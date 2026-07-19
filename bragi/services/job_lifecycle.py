"""Shared persisted job lifecycle helpers."""

from __future__ import annotations

from collections.abc import Iterable

from bragi.app_logging import log_error_event, log_event
from bragi.persistence.models import JobRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.redaction import redact_text
from bragi.services.job_diagnostics import (
    build_job_diagnostic_snapshot,
    current_job_request_context,
)
from bragi.services.provider_diagnostics import (
    ProviderDiagnosticsCollection,
    begin_provider_diagnostics_collection,
    finish_provider_diagnostics_collection,
    result_with_provider_diagnostics,
)

STALE_JOB_RECOVERY_ERROR = "Job was cancelled during startup recovery"


class JobLifecycleService:
    def __init__(self, *, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories
        self._provider_diagnostic_collections: dict[
            str,
            ProviderDiagnosticsCollection,
        ] = {}
        self._request_diagnostic_contexts: dict[str, dict[str, object]] = {}

    def create_queued(
        self,
        *,
        type: str,
        payload: dict[str, object],
        save_id: str | None = None,
        request_context: dict[str, object] | None = None,
    ) -> JobRecord:
        job = self.repositories.create_job(
            save_id=save_id,
            type=type,
            status="queued",
            payload=payload,
        )
        log_event(
            "job.queued",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
        )
        context = current_job_request_context()
        if request_context:
            context.update(request_context)
        if context:
            self._request_diagnostic_contexts[job.id] = context
        return job

    def start(
        self,
        job_id: str,
        *,
        collect_provider_diagnostics: bool = False,
    ) -> JobRecord:
        job = self.repositories.start_job(job_id)
        if collect_provider_diagnostics:
            self._provider_diagnostic_collections[job.id] = (
                begin_provider_diagnostics_collection()
            )
        log_event(
            "job.running",
            job_id=job.id,
            job_type=job.type,
            save_id=job.save_id,
        )
        return job

    def create_running(
        self,
        *,
        type: str,
        payload: dict[str, object],
        save_id: str | None = None,
        collect_provider_diagnostics: bool = False,
        request_context: dict[str, object] | None = None,
    ) -> JobRecord:
        job = self.create_queued(
            type=type,
            payload=payload,
            save_id=save_id,
            request_context=request_context,
        )
        return self.start(
            job.id,
            collect_provider_diagnostics=collect_provider_diagnostics,
        )

    def succeed(
        self,
        job_id: str,
        *,
        result: dict[str, object] | None = None,
        request_context: dict[str, object] | None = None,
    ) -> JobRecord:
        terminal_result = self._result_with_provider_diagnostics(job_id, result)
        job = self.repositories.update_job(
            job_id,
            status="succeeded",
            result=terminal_result,
        )
        job = self._persist_terminal_diagnostics(
            job,
            result=terminal_result,
            request_context=request_context,
        )
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=job.save_id,
        )
        return job

    def fail(
        self,
        job_id: str,
        *,
        error: str,
        result: dict[str, object] | None = None,
        request_context: dict[str, object] | None = None,
    ) -> JobRecord:
        terminal_result = self._result_with_provider_diagnostics(job_id, result)
        job = self.repositories.update_job(
            job_id,
            status="failed",
            result=terminal_result,
            error=redact_text(error),
        )
        job = self._persist_terminal_diagnostics(
            job,
            result=terminal_result,
            error=job.error,
            request_context=request_context,
        )
        log_error_event(
            "job.failed",
            job_id=job.id,
            job_type=job.type,
            save_id=job.save_id,
            error=job.error,
        )
        return job

    def cancel(
        self,
        job_id: str,
        *,
        error: str = "Job cancelled",
        result: dict[str, object] | None = None,
        request_context: dict[str, object] | None = None,
    ) -> JobRecord:
        terminal_result = self._result_with_provider_diagnostics(job_id, result)
        job = self.repositories.cancel_job(
            job_id,
            error=redact_text(error),
            result=terminal_result,
        )
        job = self._persist_terminal_diagnostics(
            job,
            result=terminal_result,
            error=job.error,
            request_context=request_context,
        )
        log_event(
            "job.cancelled",
            job_id=job.id,
            job_type=job.type,
            save_id=job.save_id,
        )
        return job

    def record_step(
        self,
        job_id: str,
        *,
        name: str,
        status: str,
        provider: str | None = None,
        model: str | None = None,
        task: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.repositories.record_job_step(
            job_id=job_id,
            name=name,
            status=status,
            provider=provider,
            model=model,
            task=task,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            error=redact_text(error),
            metadata=metadata,
        )

    def recover_stale_jobs(
        self,
        *,
        statuses: Iterable[str] = ("queued", "running"),
        error: str = STALE_JOB_RECOVERY_ERROR,
        preserve_queued_types: Iterable[str] = (),
    ) -> tuple[JobRecord, ...]:
        preserve_queued_types_set = set(preserve_queued_types)
        statuses_tuple = tuple(statuses)
        stale_jobs = self.repositories.list_jobs_by_status(statuses_tuple)

        recovered_list: list[JobRecord] = []
        for stale_job in stale_jobs:
            if (
                stale_job.status == "queued"
                and stale_job.type in preserve_queued_types_set
            ):
                continue
            terminal_result = {
                "previous_status": stale_job.status,
                "recovered_on_startup": True,
            }
            job = self.repositories.cancel_job(
                stale_job.id,
                error=redact_text(error),
                result=terminal_result,
            )
            recovered_list.append(
                self._persist_terminal_diagnostics(
                    job,
                    result=terminal_result,
                    error=job.error,
                )
            )
        recovered = tuple(recovered_list)
        for job in recovered:
            log_event(
                "job.stale_cancelled",
                job_id=job.id,
                job_type=job.type,
                save_id=job.save_id,
            )
        return recovered

    def _result_with_provider_diagnostics(
        self,
        job_id: str,
        result: dict[str, object] | None,
    ) -> dict[str, object] | None:
        collection = self._provider_diagnostic_collections.pop(job_id, None)
        if collection is None:
            return result
        provider_calls = finish_provider_diagnostics_collection(collection)
        return result_with_provider_diagnostics(
            result,
            provider_calls=provider_calls,
        )

    def _persist_terminal_diagnostics(
        self,
        job: JobRecord,
        *,
        result: dict[str, object] | None,
        error: str | None = None,
        request_context: dict[str, object] | None = None,
    ) -> JobRecord:
        diagnostics = self._terminal_diagnostics(
            job,
            result=result,
            error=error,
            request_context=request_context,
        )
        if diagnostics is None:
            return job
        return self.repositories.set_job_diagnostics(job.id, diagnostics)

    def _terminal_diagnostics(
        self,
        job: JobRecord | None,
        *,
        result: dict[str, object] | None,
        error: str | None = None,
        request_context: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if job is None:
            return None
        context = self._request_diagnostic_contexts.pop(job.id, None) or {}
        if request_context:
            context.update(request_context)
        return build_job_diagnostic_snapshot(
            job,
            request_context=context,
            result=result,
            error=error,
        )
