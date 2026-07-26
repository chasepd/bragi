"""In-memory operation job registry with Server-Sent Events."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from time import time
from typing import Any
from uuid import uuid4

from bragi_web.bragi_adapter import bragi_runtime_bindings
from bragi_web.observability import error_fields, observe, result_shape
from bragi_web.serialization import to_jsonable

JobWorker = Callable[["JobHandle"], Awaitable[Any]]
JobChangeCallback = Callable[["JobRecord"], None]
RepositoryScopeFactory = Callable[[], Any]
ACTIVE_JOB_STATUSES = frozenset({"queued", "running"})
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
PUBLIC_JOB_FAILURE_ERROR = "Background job failed. Check diagnostics for details."
_PUBLIC_PROVIDER_FAILURE_PREFIX = "Provider request failed"


@dataclass(frozen=True)
class JobRegistryLimits:
    max_active_jobs: int = 32
    completed_ttl_seconds: float = 60 * 60
    max_completed_jobs: int = 200
    max_events_per_job: int = 500

    def __post_init__(self) -> None:
        if self.max_active_jobs < 1:
            raise ValueError("max_active_jobs must be at least 1")
        if self.completed_ttl_seconds < 0:
            raise ValueError("completed_ttl_seconds must be non-negative")
        if self.max_completed_jobs < 0:
            raise ValueError("max_completed_jobs must be non-negative")
        if self.max_events_per_job < 1:
            raise ValueError("max_events_per_job must be at least 1")


class JobRegistryFullError(RuntimeError):
    def __init__(self, max_active_jobs: int) -> None:
        super().__init__(
            "Too many active jobs; wait for one to finish before starting another."
        )
        self.max_active_jobs = max_active_jobs


class JobRegistryExclusiveKeyError(JobRegistryFullError):
    def __init__(self, exclusive_key: str, blocking_job_id: str) -> None:
        RuntimeError.__init__(
            self,
            "An active job with this exclusivity key already exists.",
        )
        self.max_active_jobs = 0
        self.exclusive_key = exclusive_key
        self.blocking_job_id = blocking_job_id


@dataclass
class JobRecord:
    id: str
    type: str
    save_id: str | None = None
    creator_user_id: str | None = None
    exclusive_key: str | None = None
    operation_queue_key: str | None = None
    status: str = "queued"
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    result: Any = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    event_offset: int = 0
    task: asyncio.Task[Any] | None = None


class JobHandle:
    def __init__(self, registry: JobRegistry, record: JobRecord) -> None:
        self._registry = registry
        self.record = record

    async def event(self, event: str, payload: Any = None) -> None:
        await self._registry.add_event(self.record.id, event, payload)


class JobRegistry:
    def __init__(
        self,
        limits: JobRegistryLimits | None = None,
        *,
        repositories: Any | None = None,
        on_change: JobChangeCallback | None = None,
        repository_scope: RepositoryScopeFactory | None = None,
    ) -> None:
        self._limits = limits or JobRegistryLimits()
        self._repositories = repositories
        self._on_change = on_change
        self._repository_scope = repository_scope
        self._jobs: dict[str, JobRecord] = {}
        self._condition = threading.Condition(threading.RLock())
        self._operation_queue_waiters: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]
        ] = []
        self._event_waiters: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]
        ] = []

    async def create(
        self,
        job_type: str,
        worker: JobWorker,
        *,
        save_id: str | None = None,
        creator_user_id: str | None = None,
        exclusive_key: str | None = None,
        operation_queue_key: str | None = None,
    ) -> JobRecord:
        record = JobRecord(
            id=uuid4().hex,
            type=job_type,
            save_id=save_id,
            creator_user_id=creator_user_id,
            exclusive_key=exclusive_key,
            operation_queue_key=operation_queue_key,
        )
        with self._condition:
            self._prune_completed_locked(time())
            if self._active_count_locked() >= self._limits.max_active_jobs:
                raise JobRegistryFullError(self._limits.max_active_jobs)
            if exclusive_key is not None:
                blocking = self._active_exclusive_job_locked(exclusive_key)
                if blocking is not None:
                    raise JobRegistryExclusiveKeyError(exclusive_key, blocking.id)
            self._jobs[record.id] = record
            queued_snapshot = self._snapshot_record(record)
            self._condition.notify_all()
        self._persist_queued(record)
        self._notify_change(queued_snapshot)
        observe("web.job.queued", job_id=record.id, job_type=record.type)
        task = asyncio.create_task(self._run(record, worker))
        with self._condition:
            if record.id in self._jobs:
                record.task = task
                snapshot = self._snapshot_locked(record)
                self._condition.notify_all()
            else:
                snapshot = self._snapshot_record(record)
        return snapshot

    def get(self, job_id: str) -> JobRecord | None:
        with self._condition:
            self._prune_completed_locked(time())
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return self._snapshot_locked(record)

    def list_active(self, *, save_id: str | None = None) -> list[JobRecord]:
        with self._condition:
            self._prune_completed_locked(time())
            return sorted(
                (
                    self._snapshot_locked(record)
                    for record in self._jobs.values()
                    if record.status in ACTIVE_JOB_STATUSES
                    and (save_id is None or record.save_id == save_id)
                ),
                key=lambda record: record.created_at,
            )

    async def cancel(self, job_id: str) -> bool:
        queued_cancel: tuple[
            JobRecord,
            Any,
            Any,
        ] | None = None
        with self._condition:
            record = self._jobs.get(job_id)
            if (
                record is None
                or record.task is None
                or record.task.done()
                or record.status in TERMINAL_JOB_STATUSES
            ):
                return False
            if record.status == "queued":
                cancel_payload = self._append_event_locked(
                    record,
                    "cancel_requested",
                    {},
                    time(),
                )
                record.status = "cancelled"
                record.error = "Cancelled"
                status_payload = self._append_event_locked(
                    record,
                    "status",
                    {"status": "cancelled"},
                    time(),
                )
                queued_cancel = (
                    self._snapshot_locked(record),
                    cancel_payload,
                    status_payload,
                )
                self._notify_operation_queue_waiters_locked()
                self._condition.notify_all()
                job_type = record.type
            else:
                if not record.task.cancel():
                    return False
                job_type = record.type
        observe("web.job.cancel_requested", job_id=job_id, job_type=job_type)
        if queued_cancel is not None:
            snapshot, cancel_payload, status_payload = queued_cancel
            self._notify_change(snapshot)
            observe(
                "web.job.event",
                level="debug",
                job_id=job_id,
                job_type=job_type,
                job_status="queued",
                event_name="cancel_requested",
                **result_shape(cancel_payload),
            )
            observe(
                "web.job.event",
                level="debug",
                job_id=job_id,
                job_type=job_type,
                job_status="cancelled",
                event_name="status",
                **result_shape(status_payload),
            )
            self._persist_cancelled(job_id, error="Cancelled")
            observe(
                "web.job.cancelled",
                job_id=job_id,
                job_type=job_type,
                duration_ms=round((time() - snapshot.created_at) * 1000, 2),
            )
            return True
        await self.add_event(job_id, "cancel_requested", {})
        return True

    async def add_event(self, job_id: str, event: str, payload: Any = None) -> None:
        with self._condition:
            record = self._jobs[job_id]
            jsonable_payload = self._append_event_locked(record, event, payload, time())
            job_type = record.type
            job_status = record.status
            snapshot = self._snapshot_locked(record)
            self._condition.notify_all()
        self._notify_change(snapshot)
        observe(
            "web.job.event",
            level="debug",
            job_id=job_id,
            job_type=job_type,
            job_status=job_status,
            event_name=event,
            **result_shape(jsonable_payload),
        )

    async def wait_for_event(self, job_id: str, last_index: int) -> int:
        while True:
            with self._condition:
                if self._has_event_after_locked(job_id, last_index):
                    return last_index
                loop = asyncio.get_running_loop()
                waiter: asyncio.Future[None] = loop.create_future()
                self._event_waiters.append((loop, waiter))
            try:
                await asyncio.wait_for(waiter, timeout=30)
            except TimeoutError:
                with self._condition:
                    self._remove_event_waiter_locked(waiter)
                return last_index
            except asyncio.CancelledError:
                with self._condition:
                    self._remove_event_waiter_locked(waiter)
                raise

    async def _run(self, record: JobRecord, worker: JobWorker) -> None:
        try:
            with _repository_scope_context(self._repository_scope):
                await self._run_scoped(record, worker)
        except asyncio.CancelledError:
            self._cancel_task_job(record)

    async def _run_scoped(self, record: JobRecord, worker: JobWorker) -> None:
        handle = JobHandle(self, record)
        try:
            if not await self._wait_for_operation_queue_turn(record.id):
                return
            if not self._start_job_if_queued(record.id):
                return
            observe("web.job.started", job_id=record.id, job_type=record.type)
            self._persist_started(record.id)
            with bragi_runtime_bindings().runtime_telemetry_context(
                repositories=self._repositories,
                job_id=record.id,
                task=_provider_task_for_job_type(record.type),
            ):
                result = to_jsonable(await worker(handle))
        except asyncio.CancelledError:
            self._cancel_task_job(record)
            return
        except Exception as exc:  # noqa: BLE001 - user-visible job boundary
            public_error = _public_job_error_for_exception(exc)
            await self._fail_job(record.id, public_error)
            self._persist_failed(record.id, error=public_error)
            observe(
                "web.job.failed",
                level="error",
                job_id=record.id,
                job_type=record.type,
                duration_ms=round((time() - record.created_at) * 1000, 2),
                **_safe_job_exception_fields(exc, public_error),
            )
            return
        await self._succeed_job(record.id, result)
        self._persist_succeeded(record.id, result=result)
        observe(
            "web.job.succeeded",
            job_id=record.id,
            job_type=record.type,
            duration_ms=round((time() - record.created_at) * 1000, 2),
            **result_shape(result),
        )

    async def _set_status_event(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        self._set_status_event_sync(job_id, status, error=error)

    def _set_status_event_sync(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        with self._condition:
            record = self._jobs[job_id]
            record.status = status
            record.error = error
            jsonable_payload = self._append_event_locked(
                record,
                "status",
                {"status": status},
                time(),
            )
            job_type = record.type
            snapshot = self._snapshot_locked(record)
            self._notify_operation_queue_waiters_locked()
            self._condition.notify_all()
        self._notify_change(snapshot)
        observe(
            "web.job.event",
            level="debug",
            job_id=job_id,
            job_type=job_type,
            job_status=status,
            event_name="status",
            **result_shape(jsonable_payload),
        )

    def _start_job_if_queued(self, job_id: str) -> bool:
        with self._condition:
            record = self._jobs.get(job_id)
            if record is None or record.status != "queued":
                return False
            record.status = "running"
            jsonable_payload = self._append_event_locked(
                record,
                "status",
                {"status": "running"},
                time(),
            )
            job_type = record.type
            snapshot = self._snapshot_locked(record)
            self._notify_operation_queue_waiters_locked()
            self._condition.notify_all()
        self._notify_change(snapshot)
        observe(
            "web.job.event",
            level="debug",
            job_id=job_id,
            job_type=job_type,
            job_status="running",
            event_name="status",
            **result_shape(jsonable_payload),
        )
        return True

    def _cancel_task_job(self, record: JobRecord) -> None:
        with self._condition:
            current = self._jobs.get(record.id)
            if current is None or current.status in TERMINAL_JOB_STATUSES:
                return
        self._set_status_event_sync(
            record.id,
            "cancelled",
            error="Cancelled",
        )
        self._persist_cancelled(record.id, error="Cancelled")
        observe(
            "web.job.cancelled",
            job_id=record.id,
            job_type=record.type,
            duration_ms=round((time() - record.created_at) * 1000, 2),
        )

    async def _fail_job(self, job_id: str, error: str) -> None:
        with self._condition:
            record = self._jobs[job_id]
            record.status = "failed"
            record.error = error
            jsonable_payload = self._append_event_locked(
                record,
                "error",
                {"error": error},
                time(),
            )
            job_type = record.type
            snapshot = self._snapshot_locked(record)
            self._notify_operation_queue_waiters_locked()
            self._condition.notify_all()
        self._notify_change(snapshot)
        observe(
            "web.job.event",
            level="debug",
            job_id=job_id,
            job_type=job_type,
            job_status="failed",
            event_name="error",
            **result_shape(jsonable_payload),
        )

    async def _succeed_job(self, job_id: str, result: Any) -> None:
        with self._condition:
            record = self._jobs[job_id]
            record.result = result
            record.status = "succeeded"
            jsonable_payload = self._append_event_locked(
                record,
                "status",
                {"status": "succeeded"},
                time(),
            )
            job_type = record.type
            snapshot = self._snapshot_locked(record)
            self._notify_operation_queue_waiters_locked()
            self._condition.notify_all()
        self._notify_change(snapshot)
        observe(
            "web.job.event",
            level="debug",
            job_id=job_id,
            job_type=job_type,
            job_status="succeeded",
            event_name="status",
            **result_shape(jsonable_payload),
        )

    def _has_event_after_locked(self, job_id: str, last_index: int) -> bool:
        record = self._jobs.get(job_id)
        if record is None:
            return True
        return (
            record.event_offset + len(record.events) > last_index
            or record.status in TERMINAL_JOB_STATUSES
        )

    def _append_event_locked(
        self,
        record: JobRecord,
        event: str,
        payload: Any,
        now: float,
    ) -> Any:
        record.updated_at = now
        jsonable_payload = to_jsonable(payload)
        record.events.append({"event": event, "payload": jsonable_payload})
        overflow = len(record.events) - self._limits.max_events_per_job
        if overflow > 0:
            del record.events[:overflow]
            record.event_offset += overflow
        self._notify_event_waiters_locked()
        return jsonable_payload

    def _active_count_locked(self) -> int:
        return sum(
            1 for record in self._jobs.values() if record.status in ACTIVE_JOB_STATUSES
        )

    def _active_exclusive_job_locked(self, exclusive_key: str) -> JobRecord | None:
        return next(
            (
                record
                for record in self._jobs.values()
                if record.status in ACTIVE_JOB_STATUSES
                and record.exclusive_key == exclusive_key
            ),
            None,
        )

    async def _wait_for_operation_queue_turn(self, job_id: str) -> bool:
        while True:
            with self._condition:
                record = self._jobs.get(job_id)
                if record is None or record.operation_queue_key is None:
                    return True
                if record.status != "queued":
                    return False
                if self._operation_queue_record_ready_locked(record):
                    return True
                loop = asyncio.get_running_loop()
                waiter: asyncio.Future[None] = loop.create_future()
                self._operation_queue_waiters.append((loop, waiter))
            try:
                await waiter
            except asyncio.CancelledError:
                with self._condition:
                    self._operation_queue_waiters = [
                        (loop, future)
                        for loop, future in self._operation_queue_waiters
                        if future is not waiter
                    ]
                raise

    def _operation_queue_record_ready_locked(self, record: JobRecord) -> bool:
        queue_key = record.operation_queue_key
        if queue_key is None:
            return True
        for other in self._jobs.values():
            if other.id == record.id:
                break
            if (
                other.status == "queued"
                and other.operation_queue_key == queue_key
            ):
                return False
        return not any(
            other.id != record.id
            and other.status == "running"
            and _job_operation_key(other) == queue_key
            for other in self._jobs.values()
        )

    def _notify_operation_queue_waiters_locked(self) -> None:
        waiters = self._operation_queue_waiters
        self._operation_queue_waiters = []
        for loop, waiter in waiters:
            if waiter.done():
                continue
            loop.call_soon_threadsafe(_complete_waiter, waiter)

    def _notify_event_waiters_locked(self) -> None:
        waiters = self._event_waiters
        self._event_waiters = []
        for loop, waiter in waiters:
            if waiter.done():
                continue
            loop.call_soon_threadsafe(_complete_waiter, waiter)

    def _remove_event_waiter_locked(self, waiter: asyncio.Future[None]) -> None:
        self._event_waiters = [
            (loop, future)
            for loop, future in self._event_waiters
            if future is not waiter
        ]

    def _prune_completed_locked(self, now: float) -> None:
        pruned = False
        if self._jobs:
            expired_ids = [
                job_id
                for job_id, record in self._jobs.items()
                if record.status in TERMINAL_JOB_STATUSES
                and now - record.updated_at >= self._limits.completed_ttl_seconds
            ]
            for job_id in expired_ids:
                del self._jobs[job_id]
                pruned = True
        completed = sorted(
            (
                record
                for record in self._jobs.values()
                if record.status in TERMINAL_JOB_STATUSES
            ),
            key=lambda record: record.updated_at,
        )
        overflow = len(completed) - self._limits.max_completed_jobs
        for record in completed[: max(0, overflow)]:
            del self._jobs[record.id]
            pruned = True
        if pruned:
            self._notify_event_waiters_locked()
            self._condition.notify_all()

    def _snapshot_locked(self, record: JobRecord) -> JobRecord:
        return self._snapshot_record(record)

    def _snapshot_record(self, record: JobRecord) -> JobRecord:
        return replace(
            record,
            events=[
                {"event": event["event"], "payload": event.get("payload")}
                for event in record.events
            ],
        )

    def _notify_change(self, record: JobRecord) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(record)
        except Exception as exc:  # noqa: BLE001 - notifications must not break jobs
            observe(
                "web.job.change_notification_failed",
                level="error",
                job_id=record.id,
                job_type=record.type,
                **error_fields(exc),
            )

    def _persist_queued(self, record: JobRecord) -> None:
        create_job = getattr(self._repositories, "create_job", None)
        if not callable(create_job):
            return
        try:
            create_job(
                job_id=record.id,
                save_id=record.save_id,
                creator_user_id=record.creator_user_id,
                type=record.type,
                status="queued",
                payload=_safe_web_job_payload(record),
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must not break jobs
            _observe_persistence_error("web.job.persist_queued_failed", record, exc)

    def _persist_started(self, job_id: str) -> None:
        start_job = getattr(self._repositories, "start_job", None)
        if not callable(start_job):
            return
        try:
            start_job(job_id)
        except Exception as exc:  # noqa: BLE001 - telemetry must not break jobs
            _observe_persistence_error_by_id(
                "web.job.persist_started_failed",
                job_id,
                exc,
            )

    def _persist_succeeded(self, job_id: str, *, result: Any) -> None:
        update_job = getattr(self._repositories, "update_job", None)
        if not callable(update_job):
            return
        try:
            safe_result = _safe_web_job_result(result)
            job = update_job(
                job_id,
                status="succeeded",
                result=safe_result,
            )
            self._persist_terminal_diagnostics(
                job,
                result=safe_result,
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must not break jobs
            _observe_persistence_error_by_id(
                "web.job.persist_succeeded_failed",
                job_id,
                exc,
            )

    def _persist_failed(self, job_id: str, *, error: str) -> None:
        update_job = getattr(self._repositories, "update_job", None)
        if not callable(update_job):
            return
        try:
            safe_result: dict[str, object] = {"result_type": "error"}
            job = update_job(
                job_id,
                status="failed",
                result=safe_result,
                error=error,
            )
            self._persist_terminal_diagnostics(
                job,
                result=safe_result,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must not break jobs
            _observe_persistence_error_by_id(
                "web.job.persist_failed_failed",
                job_id,
                exc,
            )

    def _persist_cancelled(self, job_id: str, *, error: str) -> None:
        cancel_job = getattr(self._repositories, "cancel_job", None)
        if not callable(cancel_job):
            return
        try:
            safe_result: dict[str, object] = {"result_type": "cancelled"}
            job = cancel_job(
                job_id,
                result=safe_result,
                error=error,
            )
            self._persist_terminal_diagnostics(
                job,
                result=safe_result,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must not break jobs
            _observe_persistence_error_by_id(
                "web.job.persist_cancelled_failed",
                job_id,
                exc,
            )

    def _persist_terminal_diagnostics(
        self,
        job: Any,
        *,
        result: dict[str, object] | None,
        error: str | None = None,
    ) -> None:
        set_job_diagnostics = getattr(self._repositories, "set_job_diagnostics", None)
        if not callable(set_job_diagnostics):
            return
        try:
            from bragi.services.job_diagnostics import build_job_diagnostic_snapshot

            set_job_diagnostics(
                job.id,
                build_job_diagnostic_snapshot(
                    job,
                    result=result,
                    error=error,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must not break jobs
            _observe_persistence_error_by_id(
                "web.job.persist_diagnostics_failed",
                getattr(job, "id", "unknown"),
                exc,
            )


def _repository_scope_context(
    repository_scope: RepositoryScopeFactory | None,
) -> Any:
    if repository_scope is None:
        return nullcontext()
    return repository_scope()


def _complete_waiter(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


def job_summary(record: JobRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "type": record.type,
        "save_id": record.save_id,
        "status": record.status,
        "result": to_jsonable(record.result),
        "error": _public_job_error_for_record(record),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "latest_progress": _latest_progress_event(record),
    }


def job_event_payload(record: JobRecord, event: dict[str, Any]) -> Any:
    if event.get("event") == "error" and record.status == "failed":
        return {"error": _public_job_error_for_record(record)}
    return to_jsonable(event.get("payload"))


def _latest_progress_event(record: JobRecord) -> Any:
    for event in reversed(record.events):
        if event.get("event") == "progress":
            return to_jsonable(event.get("payload"))
    return None


def _safe_web_job_payload(record: JobRecord) -> dict[str, object]:
    payload: dict[str, object] = {"source": "web"}
    try:
        from bragi.services.job_diagnostics import current_job_request_context

        payload.update(
            {
                key: value
                for key, value in current_job_request_context().items()
                if key in {"request_id", "route"}
            }
        )
    except Exception:
        pass
    if record.exclusive_key is not None:
        payload["exclusive_key"] = record.exclusive_key
    return payload


def _job_operation_key(record: JobRecord) -> str | None:
    return record.operation_queue_key or record.save_id


def _safe_web_job_result(result: object) -> dict[str, object]:
    return result_shape(result)


def _public_job_error_for_exception(exc: Exception) -> str:
    provider_details = _provider_error_details(exc)
    if provider_details is not None:
        details = [provider_details["category"]]
        status_code = provider_details.get("status_code")
        if status_code is not None:
            details.append(status_code)
        return (
            f"{_PUBLIC_PROVIDER_FAILURE_PREFIX} ({', '.join(details)}). "
            "Check diagnostics for details."
        )
    return PUBLIC_JOB_FAILURE_ERROR


def _public_job_error_for_record(record: JobRecord) -> str | None:
    if record.status != "failed":
        return record.error
    error = record.error
    if error is None:
        return None
    if error == PUBLIC_JOB_FAILURE_ERROR or _is_public_provider_failure_error(error):
        return error
    return PUBLIC_JOB_FAILURE_ERROR


def _is_public_provider_failure_error(error: str) -> bool:
    return error.startswith(f"{_PUBLIC_PROVIDER_FAILURE_PREFIX} (") and error.endswith(
        "). Check diagnostics for details."
    )


def _safe_job_exception_fields(exc: Exception, public_error: str) -> dict[str, object]:
    fields: dict[str, object] = {
        "error_class": type(exc).__name__,
        "error": public_error,
    }
    provider_details = _provider_error_details(exc)
    if provider_details is not None:
        fields["provider_error_category"] = provider_details["category"]
        status_code = provider_details.get("status_code")
        if status_code is not None:
            fields["http_status"] = int(status_code)
    return fields


def _provider_error_details(exc: Exception) -> dict[str, str] | None:
    if type(exc).__name__ != "ProviderError":
        return None
    category = getattr(exc, "category", None)
    category_value = getattr(category, "value", None)
    if not isinstance(category_value, str) or not category_value:
        return None
    details = {"category": category_value}
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        details["status_code"] = str(status_code)
    return details


def _provider_task_for_job_type(job_type: str) -> str | None:
    if job_type in {
        "chat_turn",
        "chat_regenerate",
        "chat_edit",
        "message_edit",
        "narrator_edit",
        "character_text_send",
        "character_text_spontaneous",
        "character_text_message_edit",
        "character_text_edit",
    }:
        return "chat"
    if job_type == "action_choice_regenerate":
        return "action_choice_generation"
    if job_type in {
        "image_generation",
        "character_image_generation",
        "initial_image_generation",
        "image_regeneration",
        "character_reference_image",
    }:
        return "image_generation"
    if job_type == "image_animation":
        return "video_generation"
    if job_type in {"scenario_draft", "scenario_section"}:
        return "scenario_generation"
    if job_type in {"context_cleanup", "guided_context_cleanup"}:
        return "context_cleanup"
    if job_type == "summary_backfill":
        return "summarization"
    if job_type in {
        "context_update_retry_drain",
        "character_text_world_update_retry_drain",
        "world_suggestion_review",
    }:
        return "context_update"
    if job_type == "web_maintenance_state_pruning":
        return "state_pruning"
    if job_type == "web_maintenance_world_context_retention":
        return "world_context_retention"
    if job_type == "web_maintenance_memory_consolidation":
        return "context_update"
    if job_type == "web_maintenance_character_registry_maintenance":
        return "character_registry_maintenance"
    if job_type == "state_pruning":
        return "state_pruning"
    if job_type == "model_refresh":
        return "model_listing"
    return job_type or None


def _observe_persistence_error(
    event: str,
    record: JobRecord,
    exc: Exception,
) -> None:
    observe(
        event,
        level="error",
        job_id=record.id,
        job_type=record.type,
        **error_fields(exc),
    )


def _observe_persistence_error_by_id(
    event: str,
    job_id: str,
    exc: Exception,
) -> None:
    observe(
        event,
        level="error",
        job_id=job_id,
        **error_fields(exc),
    )
