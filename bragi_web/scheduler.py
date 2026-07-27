"""Lightweight persisted scheduler for web-owned background maintenance."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import Any, cast

from bragi_web.jobs import (
    JobHandle,
    JobRegistryExclusiveKeyError,
    JobRegistryFullError,
)
from bragi_web.observability import error_fields, observe
from bragi_web.serialization import to_jsonable

WORLD_SUGGESTION_REVIEW_TASK = "world_suggestion_review"
STATE_PRUNING_TASK = "state_pruning"
STATE_EXTRACTION_RETRY_DRAIN_TASK = "state_extraction_retry_drain"
CONTEXT_UPDATE_RETRY_DRAIN_TASK = "context_update_retry_drain"
OBSERVATION_CURATION_DRAIN_TASK = "observation_curation_drain"
CHARACTER_TEXT_WORLD_UPDATE_RETRY_DRAIN_TASK = (
    "character_text_world_update_retry_drain"
)
WORLD_CONTEXT_RETENTION_TASK = "world_context_retention"
MEMORY_CONSOLIDATION_TASK = "memory_consolidation"
CHARACTER_REGISTRY_MAINTENANCE_TASK = "character_registry_maintenance"
WEB_MAINTENANCE_STATE_PRUNING_JOB = "web_maintenance_state_pruning"
WEB_MAINTENANCE_WORLD_CONTEXT_RETENTION_JOB = (
    "web_maintenance_world_context_retention"
)
WEB_MAINTENANCE_MEMORY_CONSOLIDATION_JOB = "web_maintenance_memory_consolidation"
WEB_MAINTENANCE_CHARACTER_REGISTRY_MAINTENANCE_JOB = (
    "web_maintenance_character_registry_maintenance"
)
_RETRY_DRAIN_JOB_TYPES = {
    STATE_EXTRACTION_RETRY_DRAIN_TASK: "state_extraction_retry",
    CONTEXT_UPDATE_RETRY_DRAIN_TASK: "context_update_retry",
    CHARACTER_TEXT_WORLD_UPDATE_RETRY_DRAIN_TASK: (
        "character_text_world_update_retry"
    ),
}

WORLD_SUGGESTION_REVIEW_INTERVAL_SECONDS = 60
STATE_PRUNING_INTERVAL_SECONDS = 60
STATE_EXTRACTION_RETRY_DRAIN_INTERVAL_SECONDS = 60
CONTEXT_UPDATE_RETRY_DRAIN_INTERVAL_SECONDS = 60
OBSERVATION_CURATION_DRAIN_INTERVAL_SECONDS = 60
CHARACTER_TEXT_WORLD_UPDATE_RETRY_DRAIN_INTERVAL_SECONDS = 60
WORLD_CONTEXT_RETENTION_INTERVAL_SECONDS = 15 * 60
MEMORY_CONSOLIDATION_INTERVAL_SECONDS = 10 * 60
CHARACTER_REGISTRY_MAINTENANCE_INTERVAL_SECONDS = 5 * 60
_SCHEDULER_DEFAULT_LEASE_SECONDS = 10 * 60
_SCHEDULER_STARTUP_DELAY_SECONDS = 0.25
_SCHEDULER_RETRY_SECONDS = 30
_SCHEDULER_MAX_BACKOFF_SECONDS = 30 * 60
_DUE_ROUTINE_TARGET_LIMIT = 10


@dataclass(frozen=True)
class _MaintenanceTaskDefinition:
    task_type: str
    interval_seconds: int
    progress_label: str
    runtime_method: str
    event_reason: str
    should_schedule: Callable[[Any, str], bool]
    job_type: str | None = None
    lease_seconds: int = _SCHEDULER_DEFAULT_LEASE_SECONDS
    publish_save_event: bool = True
    cache_policy_checks: bool = False


class ScheduledMaintenanceTaskError(RuntimeError):
    def __init__(self, task_type: str) -> None:
        super().__init__(f"Scheduled maintenance task failed: {task_type}")
        self.task_type = task_type


class WebMaintenanceScheduler:
    def __init__(
        self,
        state: Any,
        *,
        poll_interval_seconds: float = 15.0,
        startup_delay_seconds: float = _SCHEDULER_STARTUP_DELAY_SECONDS,
    ) -> None:
        self._state = state
        self._poll_interval_seconds = poll_interval_seconds
        self._startup_delay_seconds = startup_delay_seconds
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopped.clear()
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        self._stopped.set()
        task = self._task
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        self._task = None

    async def run_once(self) -> None:
        selected_tasks = await self._select_tasks_to_queue()
        for definition, task in selected_tasks:
            await self._queue_task(definition, task)

    async def _select_tasks_to_queue(
        self,
    ) -> list[tuple[_MaintenanceTaskDefinition, Any]]:
        if _state_repository_scope_can_run_in_worker(self._state):
            return await asyncio.to_thread(self._select_tasks_to_queue_scoped)
        return self._select_tasks_to_queue_scoped()

    def _select_tasks_to_queue_scoped(
        self,
    ) -> list[tuple[_MaintenanceTaskDefinition, Any]]:
        with _state_repository_scope(self._state):
            return self._select_tasks_to_queue_unlocked()

    def _select_tasks_to_queue_unlocked(
        self,
    ) -> list[tuple[_MaintenanceTaskDefinition, Any]]:
        repositories = self._state.repositories
        selected_tasks: list[tuple[_MaintenanceTaskDefinition, Any]] = []
        active_save_ids = _active_save_ids(self._state, repositories)
        target_save_ids_by_task = {
            definition.task_type: _task_target_save_ids(
                definition,
                repositories,
                active_save_ids,
            )
            for definition in _MAINTENANCE_TASKS
        }
        target_save_ids = {
            save_id
            for save_ids in target_save_ids_by_task.values()
            for save_id in save_ids
        }
        state_retry_active_at_start = {
            save_id: _has_active_state_extraction_retry(repositories, save_id)
            for save_id in target_save_ids
        }
        context_retry_active_at_start = {
            save_id: _has_active_context_update_retry(repositories, save_id)
            for save_id in target_save_ids
        }
        for definition in _MAINTENANCE_TASKS:
            for save_id in target_save_ids_by_task[definition.task_type]:
                if _has_active_save_job(self._state, save_id):
                    observe(
                        "web.scheduler.run_skipped",
                        level="debug",
                        task_type=definition.task_type,
                        save_id=save_id,
                        skip_reason="active_same_save_job",
                    )
                    continue
                if (
                    state_retry_active_at_start.get(save_id, False)
                    and definition.task_type in _STATE_RETRY_BLOCKED_TASKS
                ):
                    continue
                if (
                    context_retry_active_at_start.get(save_id, False)
                    and definition.task_type in _CONTEXT_RETRY_BLOCKED_TASKS
                ):
                    continue
                if definition.cache_policy_checks and not _policy_probe_due(
                    repositories,
                    definition,
                    save_id,
                ):
                    continue
                try:
                    should_schedule = definition.should_schedule(repositories, save_id)
                except Exception as exc:  # noqa: BLE001 - scheduler must keep polling
                    observe(
                        "web.scheduler.task_policy_failed",
                        level="error",
                        task_type=definition.task_type,
                        save_id=save_id,
                        **error_fields(exc),
                    )
                    continue
                if not should_schedule:
                    _record_policy_not_due(
                        repositories,
                        definition,
                        save_id,
                    )
                    continue
                existing = repositories.get_scheduled_task(
                    task_type=definition.task_type,
                    save_id=save_id,
                )
                repositories.upsert_scheduled_task(
                    task_type=definition.task_type,
                    save_id=save_id,
                    interval_seconds=definition.interval_seconds,
                    payload=_scheduled_task_payload(definition),
                    due_now=existing is None,
                )
                due_tasks = repositories.list_due_scheduled_tasks(
                    task_types=(definition.task_type,),
                    save_id=save_id,
                    limit=1,
                )
                for task in due_tasks:
                    leased = repositories.lease_scheduled_task(
                        task.id,
                        lease_seconds=definition.lease_seconds,
                    )
                    if leased is None:
                        continue
                    try:
                        should_still_schedule = definition.should_schedule(
                            repositories,
                            save_id,
                        )
                    except Exception as exc:  # noqa: BLE001 - release leased task
                        repositories.complete_scheduled_task(
                            task.id,
                            succeeded=False,
                            error=str(exc),
                            next_run_after_seconds=_SCHEDULER_RETRY_SECONDS,
                        )
                        observe(
                            "web.scheduler.task_policy_failed",
                            level="error",
                            task_type=definition.task_type,
                            save_id=save_id,
                            **error_fields(exc),
                        )
                        continue
                    if not should_still_schedule:
                        _complete_leased_task_policy_skip(
                            repositories,
                            definition,
                            task,
                        )
                        continue
                    selected_tasks.append((definition, leased))
        return selected_tasks

    async def _run_forever(self) -> None:
        if self._startup_delay_seconds > 0:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._stopped.wait(),
                    timeout=self._startup_delay_seconds,
                )
                return
        while not self._stopped.is_set():
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001 - scheduler must stay alive
                observe(
                    "web.scheduler.run_failed",
                    level="error",
                    **error_fields(exc),
                )
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._stopped.wait(),
                    timeout=self._poll_interval_seconds,
                )

    async def _queue_task(
        self,
        definition: _MaintenanceTaskDefinition,
        task: Any,
    ) -> None:
        save_id = task.save_id
        if not isinstance(save_id, str) or not save_id:
            self._state.repositories.complete_scheduled_task(
                task.id,
                succeeded=False,
                error=f"Scheduled {definition.task_type} has no save id",
                next_run_after_seconds=_SCHEDULER_RETRY_SECONDS,
            )
            return

        async def worker(handle: JobHandle) -> Any:
            await handle.event(
                "progress",
                {"label": definition.progress_label},
            )
            try:
                runtime_method = getattr(self._state.runtime, definition.runtime_method)
                if definition.task_type == WORLD_SUGGESTION_REVIEW_TASK:
                    result = await runtime_method(
                        active_save_id=save_id,
                        scheduled=True,
                    )
                else:
                    result = await runtime_method(
                        active_save_id=save_id,
                    )
            except asyncio.CancelledError:
                self._state.repositories.complete_scheduled_task(
                    task.id,
                    succeeded=True,
                    result={"status": "cancelled"},
                    last_job_id=handle.record.id,
                    next_run_after_seconds=definition.interval_seconds,
                )
                raise
            except Exception as exc:
                self._state.repositories.complete_scheduled_task(
                    task.id,
                    succeeded=False,
                    error=str(exc),
                    last_job_id=handle.record.id,
                    next_run_after_seconds=_backoff_seconds(task.failure_count),
                )
                raise
            payload = _job_result_payload(result)
            error = payload.get("error")
            succeeded = not isinstance(error, str) or not error
            self._state.repositories.complete_scheduled_task(
                task.id,
                succeeded=succeeded,
                result=payload,
                error=error if isinstance(error, str) else None,
                last_job_id=handle.record.id,
                next_run_after_seconds=(
                    definition.interval_seconds
                    if succeeded
                    else _backoff_seconds(task.failure_count)
                ),
            )
            if succeeded and definition.publish_save_event:
                self._state.save_events.publish(
                    save_id,
                    "world_data_changed",
                    {"reason": definition.event_reason},
                )
            if not succeeded:
                raise ScheduledMaintenanceTaskError(definition.task_type)
            return payload

        try:
            job_type = definition.job_type or definition.task_type
            job = await self._state.jobs.create(
                job_type,
                worker,
                save_id=save_id,
                exclusive_key=_task_exclusive_key(definition.task_type, save_id),
            )
        except JobRegistryExclusiveKeyError as exc:
            self._state.repositories.complete_scheduled_task(
                task.id,
                succeeded=False,
                error=f"{definition.task_type} job is already active",
                next_run_after_seconds=_SCHEDULER_RETRY_SECONDS,
            )
            observe(
                "web.scheduler.task_skipped",
                level="debug",
                task_type=definition.task_type,
                job_type=definition.job_type or definition.task_type,
                save_id=save_id,
                skip_reason="active_web_job",
                blocking_job_id=exc.blocking_job_id,
            )
            return
        except JobRegistryFullError as exc:
            self._state.repositories.complete_scheduled_task(
                task.id,
                succeeded=False,
                error="Web job registry is full",
                next_run_after_seconds=_SCHEDULER_RETRY_SECONDS,
            )
            observe(
                "web.scheduler.task_skipped",
                level="debug",
                task_type=definition.task_type,
                job_type=definition.job_type or definition.task_type,
                save_id=save_id,
                skip_reason="active_web_job_registry_full",
                max_active_jobs=exc.max_active_jobs,
            )
            return
        observe(
            "web.scheduler.task_scheduled",
            task_type=definition.task_type,
            job_type=definition.job_type or definition.task_type,
            save_id=save_id,
            job_id=job.id,
        )


def _state_repository_scope(state: Any) -> Any:
    state_scope = getattr(state, "repository_scope", None)
    if callable(state_scope):
        return state_scope()
    repositories = getattr(state, "repositories", None)
    repository_scope = getattr(repositories, "scope", None)
    if callable(repository_scope):
        return repository_scope()
    return nullcontext()


def _state_repository_scope_can_run_in_worker(state: Any) -> bool:
    if callable(getattr(state, "repository_scope", None)):
        return True
    repositories = getattr(state, "repositories", None)
    return callable(getattr(repositories, "scope", None))


def world_suggestion_review_exclusive_key(save_id: str | None) -> str | None:
    return _task_exclusive_key(WORLD_SUGGESTION_REVIEW_TASK, save_id)


def _task_exclusive_key(task_type: str, save_id: str | None) -> str | None:
    return f"{task_type}:{save_id}" if save_id else None


def _active_save_id(state: Any) -> str | None:
    save_id = getattr(state.runtime, "active_save_id", None)
    if isinstance(save_id, str) and save_id:
        return save_id
    return None


def _active_save_ids(state: Any, repositories: Any) -> tuple[str, ...]:
    save_ids: list[str] = []
    _append_unique_save_id(save_ids, _active_save_id(state))
    list_user_active_save_ids = getattr(repositories, "list_user_active_save_ids", None)
    if callable(list_user_active_save_ids):
        for save_id in list_user_active_save_ids():
            _append_unique_save_id(save_ids, save_id)
    return tuple(save_ids)


def _append_unique_save_id(save_ids: list[str], save_id: object) -> None:
    if isinstance(save_id, str) and save_id and save_id not in save_ids:
        save_ids.append(save_id)


def _task_target_save_ids(
    definition: _MaintenanceTaskDefinition,
    repositories: Any,
    active_save_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if definition.task_type == WORLD_SUGGESTION_REVIEW_TASK:
        list_due_reviews = getattr(
            repositories,
            "list_save_ids_with_due_context_update_suggestion_reviews",
            None,
        )
        if callable(list_due_reviews):
            return tuple(list_due_reviews(limit=_DUE_ROUTINE_TARGET_LIMIT))
    if definition.task_type == OBSERVATION_CURATION_DRAIN_TASK:
        list_due_curation = getattr(
            repositories,
            "list_save_ids_with_due_context_observation_curation",
            None,
        )
        if callable(list_due_curation):
            runnable_save_ids: list[str] = []
            offset = 0
            while len(runnable_save_ids) < _DUE_ROUTINE_TARGET_LIMIT:
                page = tuple(
                    list_due_curation(
                        limit=_DUE_ROUTINE_TARGET_LIMIT,
                        offset=offset,
                    )
                )
                if not page:
                    break
                for save_id in page:
                    if _observation_curation_drain_due(repositories, save_id):
                        _append_unique_save_id(runnable_save_ids, save_id)
                        if len(runnable_save_ids) >= _DUE_ROUTINE_TARGET_LIMIT:
                            break
                offset += len(page)
            return tuple(runnable_save_ids)
    retry_job_type = _RETRY_DRAIN_JOB_TYPES.get(definition.task_type)
    if retry_job_type is not None:
        return _save_ids_with_queued_jobs(repositories, job_type=retry_job_type)
    save_ids = list(active_save_ids)
    for save_id in _save_ids_with_due_scheduled_task(
        repositories,
        active_save_ids=active_save_ids,
        task_type=definition.task_type,
    ):
        _append_unique_save_id(save_ids, save_id)
    return tuple(save_ids)


def _save_ids_with_queued_jobs(repositories: Any, *, job_type: str) -> tuple[str, ...]:
    list_job_save_ids = getattr(repositories, "list_job_save_ids", None)
    if callable(list_job_save_ids):
        return tuple(
            sorted(
                list_job_save_ids(
                    statuses=("queued",),
                    types=(job_type,),
                )
            )
        )
    save_ids = {
        job.save_id
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == job_type and isinstance(job.save_id, str) and job.save_id
    }
    return tuple(sorted(save_ids))


def _save_ids_with_due_scheduled_task(
    repositories: Any,
    *,
    active_save_ids: tuple[str, ...],
    task_type: str,
) -> tuple[str, ...]:
    active_save_id_set = set(active_save_ids)
    tasks = repositories.list_due_scheduled_tasks(
        task_types=(task_type,),
        limit=_DUE_ROUTINE_TARGET_LIMIT,
    )
    save_ids: list[str] = []
    for task in tasks:
        save_id = getattr(task, "save_id", None)
        if (
            isinstance(save_id, str)
            and save_id not in active_save_id_set
            and _scheduled_task_active_save_only(task)
        ):
            continue
        _append_unique_save_id(save_ids, save_id)
    return tuple(save_ids)


def _scheduled_task_active_save_only(task: Any) -> bool:
    payload = getattr(task, "payload", None)
    if not isinstance(payload, Mapping):
        return True
    return payload.get("active_save_only") is not False


def _scheduled_task_payload(
    definition: _MaintenanceTaskDefinition,
) -> dict[str, object]:
    retry_job_type = _RETRY_DRAIN_JOB_TYPES.get(definition.task_type)
    if retry_job_type is not None:
        return {"active_save_only": False, "queued_job_type": retry_job_type}
    if definition.task_type == OBSERVATION_CURATION_DRAIN_TASK:
        return {"active_save_only": False}
    return {"active_save_only": True}


def _has_active_save_job(state: Any, save_id: str) -> bool:
    jobs = getattr(state, "jobs", None)
    list_active = getattr(jobs, "list_active", None)
    if not callable(list_active):
        return False
    return bool(list_active(save_id=save_id))


def _policy_probe_due(
    repositories: Any,
    definition: _MaintenanceTaskDefinition,
    save_id: str,
) -> bool:
    existing = repositories.get_scheduled_task(
        task_type=definition.task_type,
        save_id=save_id,
    )
    if existing is None:
        return True
    return bool(
        repositories.list_due_scheduled_tasks(
            task_types=(definition.task_type,),
            save_id=save_id,
            limit=1,
        )
    )


def _record_policy_not_due(
    repositories: Any,
    definition: _MaintenanceTaskDefinition,
    save_id: str,
) -> None:
    due_tasks = repositories.list_due_scheduled_tasks(
        task_types=(definition.task_type,),
        save_id=save_id,
        limit=1,
    )
    for task in due_tasks:
        leased = repositories.lease_scheduled_task(
            task.id,
            lease_seconds=definition.lease_seconds,
        )
        if leased is None:
            continue
        _complete_leased_task_policy_skip(repositories, definition, leased)
        return
    if definition.cache_policy_checks:
        existing = repositories.get_scheduled_task(
            task_type=definition.task_type,
            save_id=save_id,
        )
        if existing is None:
            repositories.upsert_scheduled_task(
                task_type=definition.task_type,
                save_id=save_id,
                interval_seconds=definition.interval_seconds,
                payload=_scheduled_task_payload(definition),
                due_now=False,
            )


def _complete_leased_task_policy_skip(
    repositories: Any,
    definition: _MaintenanceTaskDefinition,
    task: Any,
) -> None:
    repositories.complete_scheduled_task(
        task.id,
        succeeded=True,
        result={
            "status": "skipped",
            "skip_reason": "task_policy_not_due",
        },
        next_run_after_seconds=definition.interval_seconds,
    )
    observe(
        "web.scheduler.task_skipped",
        level="debug",
        task_type=definition.task_type,
        job_type=definition.job_type or definition.task_type,
        save_id=task.save_id,
        skip_reason="task_policy_not_due",
    )


def _has_pending_suggestions(repositories: Any, save_id: str) -> bool:
    has_suggestions = getattr(repositories, "has_context_update_suggestions", None)
    if callable(has_suggestions):
        return bool(has_suggestions(save_id, status="pending"))
    try:
        suggestions = repositories.list_context_update_suggestions(
            save_id,
            status="pending",
        )
    except TypeError:
        suggestions = repositories.list_context_update_suggestions(save_id)
    return any(suggestion.status == "pending" for suggestion in suggestions)


def _world_suggestion_review_due(repositories: Any, save_id: str) -> bool:
    has_due_review = getattr(
        repositories,
        "has_due_context_update_suggestion_review",
        None,
    )
    if callable(has_due_review):
        if not has_due_review(save_id):
            return False
    elif not _has_pending_suggestions(repositories, save_id):
        return False
    return _model_preference_configured(
        repositories,
        save_id=save_id,
        purpose="context_update",
    )


def _state_pruning_due(repositories: Any, save_id: str) -> bool:
    from bragi.services.state_pruning_policy import state_pruning_schedule_decision

    return state_pruning_schedule_decision(
        repositories=repositories,
        save_id=save_id,
    ).due


def _context_update_retry_drain_due(repositories: Any, save_id: str) -> bool:
    has_matching_job = getattr(repositories, "has_matching_job", None)
    if callable(has_matching_job):
        return bool(
            has_matching_job(
                statuses=("queued",),
                types=("context_update_retry",),
                save_id=save_id,
            )
        )
    return any(
        job.type == "context_update_retry" and job.save_id == save_id
        for job in repositories.list_jobs_by_status(("queued",))
    )


def _observation_curation_drain_due(repositories: Any, save_id: str) -> bool:
    from bragi.services.agentic_context import agentic_context_pipeline_enabled

    if not agentic_context_pipeline_enabled(repositories, save_id=save_id):
        return False
    if not _model_preference_configured(
        repositories,
        save_id=save_id,
        purpose="memory_curation",
    ):
        return False
    return bool(
        repositories.list_eligible_context_observations(save_id, limit=1)
    )


def _state_extraction_retry_drain_due(repositories: Any, save_id: str) -> bool:
    has_matching_job = getattr(repositories, "has_matching_job", None)
    if callable(has_matching_job):
        return bool(
            has_matching_job(
                statuses=("queued",),
                types=("state_extraction_retry",),
                save_id=save_id,
            )
        )
    return any(
        job.type == "state_extraction_retry" and job.save_id == save_id
        for job in repositories.list_jobs_by_status(("queued",))
    )


def _character_text_world_update_retry_drain_due(
    repositories: Any,
    save_id: str,
) -> bool:
    has_matching_job = getattr(repositories, "has_matching_job", None)
    if callable(has_matching_job):
        return bool(
            has_matching_job(
                statuses=("queued",),
                types=("character_text_world_update_retry",),
                save_id=save_id,
            )
        )
    return any(
        job.type == "character_text_world_update_retry" and job.save_id == save_id
        for job in repositories.list_jobs_by_status(("queued",))
    )


def _world_context_retention_due(repositories: Any, save_id: str) -> bool:
    has_retention_work = getattr(repositories, "has_world_context_retention_work", None)
    if callable(has_retention_work):
        return bool(has_retention_work(save_id))
    return False


def _memory_consolidation_due(repositories: Any, save_id: str) -> bool:
    from bragi.services.memory_consolidation_service import (
        MEMORY_CONSOLIDATION_THRESHOLD,
    )

    if _has_active_state_extraction_retry(
        repositories,
        save_id,
    ) or _has_active_context_update_retry(repositories, save_id):
        return False
    if not _model_preference_configured(
        repositories,
        save_id=save_id,
        purpose="context_update",
    ):
        return False
    count_active_memories = getattr(repositories, "count_active_memories", None)
    if callable(count_active_memories):
        return bool(count_active_memories(save_id) >= MEMORY_CONSOLIDATION_THRESHOLD)
    return len(repositories.list_memories(save_id)) >= MEMORY_CONSOLIDATION_THRESHOLD


def _character_registry_maintenance_due(repositories: Any, save_id: str) -> bool:
    if _has_active_state_extraction_retry(
        repositories,
        save_id,
    ) or _has_active_context_update_retry(repositories, save_id):
        return False
    if not _model_preference_configured(
        repositories,
        save_id=save_id,
        purpose="character_registry_maintenance",
    ):
        return False
    has_unprotected_character = getattr(
        repositories,
        "has_unprotected_character",
        None,
    )
    if callable(has_unprotected_character):
        has_actionable_character = bool(has_unprotected_character(save_id))
    else:
        has_actionable_character = any(
            not character.protected_from_maintenance
            for character in repositories.list_characters(save_id)
        )
    if not has_actionable_character:
        return False
    return _character_turn_count_is_due(
        repositories,
        save_id,
    ) and not _latest_character_maintenance_covers_current_turn(
        repositories,
        save_id,
    )


def _has_active_context_update_retry(repositories: Any, save_id: str) -> bool:
    has_matching_job = getattr(repositories, "has_matching_job", None)
    if callable(has_matching_job):
        return bool(
            has_matching_job(
                statuses=("queued", "running"),
                types=("context_update_retry",),
                save_id=save_id,
            )
        )
    return any(
        job.type == "context_update_retry" and job.save_id == save_id
        for job in repositories.list_jobs_by_status(("queued", "running"))
    )


def _has_active_state_extraction_retry(repositories: Any, save_id: str) -> bool:
    has_matching_job = getattr(repositories, "has_matching_job", None)
    if callable(has_matching_job):
        return bool(
            has_matching_job(
                statuses=("queued", "running"),
                types=("state_extraction_retry",),
                save_id=save_id,
            )
        )
    return any(
        job.type == "state_extraction_retry" and job.save_id == save_id
        for job in repositories.list_jobs_by_status(("queued", "running"))
    )


def _model_preference_configured(
    repositories: Any,
    *,
    save_id: str,
    purpose: str,
) -> bool:
    from bragi.services.model_preferences import roleplay_model_preference

    return (
        roleplay_model_preference(
            repositories=repositories,
            save_id=save_id,
            purpose=purpose,
        )
        is not None
    )


def _character_turn_count_is_due(repositories: Any, save_id: str) -> bool:
    from bragi.services.character_registry_maintenance_service import (
        CHARACTER_MAINTENANCE_TURN_CADENCE,
    )

    count_messages = getattr(repositories, "count_active_messages_by_role", None)
    if callable(count_messages):
        counts = count_messages(save_id, roles=("player", "narrator"))
        player_turns = int(counts["player"])
        narrator_turns = int(counts["narrator"])
    else:
        player_turns = sum(
            1
            for message in repositories.list_messages(save_id)
            if message.role == "player"
        )
        narrator_turns = sum(
            1
            for message in repositories.list_messages(save_id)
            if message.role == "narrator"
        )
    completed_turns = min(player_turns, narrator_turns)
    return bool(
        completed_turns > 0
        and completed_turns % CHARACTER_MAINTENANCE_TURN_CADENCE == 0
    )


def _latest_character_maintenance_covers_current_turn(
    repositories: Any,
    save_id: str,
) -> bool:
    jobs = repositories.list_recent_jobs(
        save_id=save_id,
        types=(CHARACTER_REGISTRY_MAINTENANCE_TASK,),
        seconds=0,
        limit=1,
    )
    if not jobs:
        return False
    latest = jobs[0]
    if latest.status in {"queued", "running"}:
        return True
    latest_narrator_time = _latest_narrator_message_time(repositories, save_id)
    job_time = latest.completed_at or latest.started_at
    latest_narrator_second = _timestamp_second(latest_narrator_time)
    job_second = _timestamp_second(job_time)
    return job_second is not None and (
        latest_narrator_second is None or job_second >= latest_narrator_second
    )


def _latest_narrator_message_time(repositories: Any, save_id: str) -> str | None:
    latest_created_at = getattr(repositories, "latest_active_message_created_at", None)
    if callable(latest_created_at):
        value = latest_created_at(save_id, role="narrator")
        return value if isinstance(value, str) else None
    created_at_values = [
        message.created_at
        for message in repositories.list_messages(save_id)
        if message.role == "narrator" and isinstance(message.created_at, str)
    ]
    return max(created_at_values) if created_at_values else None


def _timestamp_second(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.split(".", 1)[0]


def _job_result_payload(result: object) -> dict[str, object]:
    jsonable = to_jsonable(result)
    if isinstance(jsonable, dict):
        payload: dict[str, object] = {}
        for key, value in cast(dict[str, object], jsonable).items():
            if key in _SCHEDULER_RESULT_KEYS or key.endswith("_count"):
                payload[key] = value
        return payload or {"result_type": "object"}
    return {"result": jsonable}


def _backoff_seconds(failure_count: int) -> int:
    return int(
        min(
            _SCHEDULER_MAX_BACKOFF_SECONDS,
            _SCHEDULER_RETRY_SECONDS * (2 ** min(max(failure_count, 0), 5)),
        )
    )


_MAINTENANCE_TASKS: tuple[_MaintenanceTaskDefinition, ...] = (
    _MaintenanceTaskDefinition(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        interval_seconds=WORLD_SUGGESTION_REVIEW_INTERVAL_SECONDS,
        progress_label="Reviewing world suggestions",
        runtime_method="run_world_suggestion_review",
        event_reason="world_suggestion_review",
        should_schedule=_world_suggestion_review_due,
    ),
    _MaintenanceTaskDefinition(
        task_type=STATE_PRUNING_TASK,
        interval_seconds=STATE_PRUNING_INTERVAL_SECONDS,
        progress_label="Cleaning world state",
        runtime_method="run_state_pruning",
        event_reason="state_pruning",
        should_schedule=_state_pruning_due,
        job_type=WEB_MAINTENANCE_STATE_PRUNING_JOB,
        cache_policy_checks=True,
    ),
    _MaintenanceTaskDefinition(
        task_type=STATE_EXTRACTION_RETRY_DRAIN_TASK,
        interval_seconds=STATE_EXTRACTION_RETRY_DRAIN_INTERVAL_SECONDS,
        progress_label="Retrying state extraction",
        runtime_method="run_state_extraction_retries",
        event_reason="state_extraction_retry",
        should_schedule=_state_extraction_retry_drain_due,
    ),
    _MaintenanceTaskDefinition(
        task_type=CONTEXT_UPDATE_RETRY_DRAIN_TASK,
        interval_seconds=CONTEXT_UPDATE_RETRY_DRAIN_INTERVAL_SECONDS,
        progress_label="Retrying context updates",
        runtime_method="run_context_update_retries",
        event_reason="context_update_retry",
        should_schedule=_context_update_retry_drain_due,
    ),
    _MaintenanceTaskDefinition(
        task_type=OBSERVATION_CURATION_DRAIN_TASK,
        interval_seconds=OBSERVATION_CURATION_DRAIN_INTERVAL_SECONDS,
        progress_label="Curating observations",
        runtime_method="run_observation_curation",
        event_reason="observation_curation",
        should_schedule=_observation_curation_drain_due,
    ),
    _MaintenanceTaskDefinition(
        task_type=CHARACTER_TEXT_WORLD_UPDATE_RETRY_DRAIN_TASK,
        interval_seconds=CHARACTER_TEXT_WORLD_UPDATE_RETRY_DRAIN_INTERVAL_SECONDS,
        progress_label="Retrying text world updates",
        runtime_method="run_character_text_world_update_retries",
        event_reason="character_text_world_update_retry",
        should_schedule=_character_text_world_update_retry_drain_due,
    ),
    _MaintenanceTaskDefinition(
        task_type=WORLD_CONTEXT_RETENTION_TASK,
        interval_seconds=WORLD_CONTEXT_RETENTION_INTERVAL_SECONDS,
        progress_label="Pruning world data history",
        runtime_method="run_world_context_retention",
        event_reason="world_context_retention",
        should_schedule=_world_context_retention_due,
        job_type=WEB_MAINTENANCE_WORLD_CONTEXT_RETENTION_JOB,
        cache_policy_checks=True,
    ),
    _MaintenanceTaskDefinition(
        task_type=MEMORY_CONSOLIDATION_TASK,
        interval_seconds=MEMORY_CONSOLIDATION_INTERVAL_SECONDS,
        progress_label="Consolidating memories",
        runtime_method="run_memory_consolidation",
        event_reason="memory_consolidation",
        should_schedule=_memory_consolidation_due,
        job_type=WEB_MAINTENANCE_MEMORY_CONSOLIDATION_JOB,
        cache_policy_checks=True,
    ),
    _MaintenanceTaskDefinition(
        task_type=CHARACTER_REGISTRY_MAINTENANCE_TASK,
        interval_seconds=CHARACTER_REGISTRY_MAINTENANCE_INTERVAL_SECONDS,
        progress_label="Maintaining character registry",
        runtime_method="run_character_registry_maintenance",
        event_reason="character_registry_maintenance",
        should_schedule=_character_registry_maintenance_due,
        job_type=WEB_MAINTENANCE_CHARACTER_REGISTRY_MAINTENANCE_JOB,
        cache_policy_checks=True,
    ),
)

_STATE_RETRY_BLOCKED_TASKS = frozenset(
    {
        CONTEXT_UPDATE_RETRY_DRAIN_TASK,
        MEMORY_CONSOLIDATION_TASK,
        CHARACTER_REGISTRY_MAINTENANCE_TASK,
    }
)

_CONTEXT_RETRY_BLOCKED_TASKS = frozenset(
    {
        MEMORY_CONSOLIDATION_TASK,
        CHARACTER_REGISTRY_MAINTENANCE_TASK,
    }
)

_SCHEDULER_RESULT_KEYS = frozenset(
    {
        "active_save_id",
        "completed",
        "error",
        "expired_excess_suggestions",
        "expired_stale_suggestions",
        "failure_text",
        "pruned_audit_rows",
        "pruned_terminal_jobs",
        "status",
    }
)


WorldSuggestionReviewScheduler = WebMaintenanceScheduler
