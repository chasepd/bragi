"""Diagnostics aggregation for provider and job failures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bragi.persistence.models import (
    JobRecord,
    RuntimePerformanceRecord,
    RuntimeSlowOperationRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.redaction import redact_text
from bragi.services.job_diagnostics import job_origin_summary
from bragi.services.provider_diagnostics import retry_summary, safe_retry_attempts
from bragi.services.text_script_policy import (
    SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
    object_text_script_violations,
)


@dataclass(frozen=True)
class ProviderDiagnostic:
    provider: str
    enabled: bool
    has_api_key: bool
    last_error: str | None


@dataclass(frozen=True)
class FailedJobDiagnostic:
    id: str
    save_id: str | None
    type: str
    status: str
    error: str | None
    attempt_count: int | None = None
    max_attempts: int | None = None
    retry_attempts: tuple[dict[str, object], ...] = ()
    retry_summary: str | None = None
    provider: str | None = None
    model: str | None = None
    origin: dict[str, str] | None = None
    detail_available: bool = False


@dataclass(frozen=True)
class FailedJobGroupDiagnostic:
    save_id: str | None
    type: str
    status: str
    provider: str | None
    count: int
    latest_error: str | None
    latest_http_status: int | None
    summary: str


@dataclass(frozen=True)
class RuntimePerformanceReport:
    job_averages: tuple[RuntimePerformanceRecord, ...]
    step_averages: tuple[RuntimePerformanceRecord, ...]
    model_averages: tuple[RuntimePerformanceRecord, ...]
    slowest_recent: tuple[RuntimeSlowOperationRecord, ...] = ()
    window_started_at: str | None = None
    limit: int | None = None


@dataclass(frozen=True)
class GeneratedTextScriptDiagnostic:
    save_id: str
    table: str
    field: str
    script: str
    count: int
    example_record_id: str


@dataclass(frozen=True)
class DiagnosticsReport:
    provider_configs: tuple[ProviderDiagnostic, ...]
    failed_jobs: tuple[FailedJobDiagnostic, ...]
    failed_job_groups: tuple[FailedJobGroupDiagnostic, ...] = ()
    generated_text_script_findings: tuple[GeneratedTextScriptDiagnostic, ...] = ()
    runtime_performance: RuntimePerformanceReport = RuntimePerformanceReport(
        job_averages=(),
        step_averages=(),
        model_averages=(),
    )
    log_file_path: str | None = None


class DiagnosticsService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        log_file_path: Path | None = None,
    ) -> None:
        self.repositories = repositories
        self.log_file_path = log_file_path

    def list_diagnostics(
        self,
        *,
        save_id: str | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> DiagnosticsReport:
        failed_jobs = tuple(self.repositories.list_failed_jobs())
        repeated_job_keys = _repeated_job_keys(failed_jobs)
        return DiagnosticsReport(
            provider_configs=tuple(
                ProviderDiagnostic(
                    provider=config.provider,
                    enabled=config.enabled,
                    has_api_key=config.has_api_key,
                    last_error=redact_diagnostic_text(config.last_error),
                )
                for config in self.repositories.list_provider_configs()
            ),
            failed_jobs=tuple(
                _failed_job_diagnostic(job)
                for job in failed_jobs
                if _job_group_key(job) not in repeated_job_keys
            ),
            failed_job_groups=_failed_job_groups(failed_jobs),
            generated_text_script_findings=_generated_text_script_findings(
                self.repositories,
            ),
            runtime_performance=RuntimePerformanceReport(
                job_averages=tuple(
                    self.repositories.runtime_job_averages(
                        save_id=save_id,
                        since=since,
                        limit=limit,
                    )
                ),
                step_averages=tuple(
                    self.repositories.runtime_step_averages(
                        save_id=save_id,
                        since=since,
                        limit=limit,
                    )
                ),
                model_averages=tuple(
                    self.repositories.runtime_model_averages(
                        save_id=save_id,
                        since=since,
                        limit=limit,
                    )
                ),
                slowest_recent=tuple(
                    self.repositories.runtime_slowest_recent_operations(
                        save_id=save_id,
                        since=since,
                        limit=limit or 10,
                    )
                ),
                window_started_at=since,
                limit=limit,
            ),
            log_file_path=(
                str(self.log_file_path.resolve()) if self.log_file_path else None
            ),
        )


def redact_diagnostic_text(value: str | None) -> str | None:
    return redact_text(value)


def _int_result_field(
    result: dict[str, object] | None,
    key: str,
) -> int | None:
    value = (result or {}).get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _failed_job_diagnostic(job: JobRecord) -> FailedJobDiagnostic:
    return FailedJobDiagnostic(
        id=job.id,
        save_id=job.save_id,
        type=job.type,
        status=job.status,
        error=redact_diagnostic_text(job.error),
        attempt_count=_int_result_field(job.result, "attempt_count"),
        max_attempts=_int_result_field(job.result, "max_attempts"),
        retry_attempts=tuple(
            safe_retry_attempts((job.result or {}).get("retry_attempts"))
        ),
        retry_summary=retry_summary(job.result),
        provider=_job_provider(job),
        model=_job_model(job),
        origin=job_origin_summary(job),
        detail_available=job.diagnostics is not None,
    )


def _failed_job_groups(
    jobs: tuple[JobRecord, ...],
) -> tuple[FailedJobGroupDiagnostic, ...]:
    grouped: dict[tuple[str | None, str, str, str | None], list[JobRecord]] = {}
    for job in jobs:
        key = _job_group_key(job)
        grouped.setdefault(key, []).append(job)
    diagnostics: list[FailedJobGroupDiagnostic] = []
    for (save_id, job_type, status, provider), group_jobs in grouped.items():
        if len(group_jobs) < 2:
            continue
        latest = group_jobs[-1]
        latest_error = redact_diagnostic_text(latest.error)
        latest_http_status = _latest_http_status(latest)
        summary = _group_summary(
            job_type=job_type,
            status=status,
            count=len(group_jobs),
            latest_http_status=latest_http_status,
            latest_error=latest_error,
        )
        diagnostics.append(
            FailedJobGroupDiagnostic(
                save_id=save_id,
                type=job_type,
                status=status,
                provider=provider,
                count=len(group_jobs),
                latest_error=latest_error,
                latest_http_status=latest_http_status,
                summary=summary,
            )
        )
    return tuple(diagnostics)


def _repeated_job_keys(
    jobs: tuple[JobRecord, ...],
) -> set[tuple[str | None, str, str, str | None]]:
    counts: dict[tuple[str | None, str, str, str | None], int] = {}
    for job in jobs:
        key = _job_group_key(job)
        counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count > 1}


def _job_group_key(job: JobRecord) -> tuple[str | None, str, str, str | None]:
    return (job.save_id, job.type, job.status, _job_provider(job))


def _job_provider(job: JobRecord) -> str | None:
    for source in (job.result or {}, job.payload):
        provider = source.get("provider")
        if isinstance(provider, str) and provider:
            return provider
        provider_calls = source.get("provider_calls")
        if isinstance(provider_calls, list | tuple):
            for call in reversed(provider_calls):
                if isinstance(call, dict):
                    call_provider = call.get("provider")
                    if isinstance(call_provider, str) and call_provider:
                        return call_provider
    return None


def _job_model(job: JobRecord) -> str | None:
    for source in (job.result or {}, job.payload):
        for key in ("model", "model_id", "final_model", "original_model"):
            model = source.get(key)
            if isinstance(model, str) and model:
                return model
        provider_calls = source.get("provider_calls")
        if isinstance(provider_calls, list | tuple):
            for call in reversed(provider_calls):
                if isinstance(call, dict):
                    model = call.get("model")
                    if isinstance(model, str) and model:
                        return model
    return None


def _latest_http_status(job: JobRecord) -> int | None:
    for source in (job.result or {}, job.payload):
        status = source.get("http_status")
        if isinstance(status, int) and not isinstance(status, bool):
            return status
        pressure = source.get("provider_pressure")
        if isinstance(pressure, dict):
            status = pressure.get("http_status")
            if isinstance(status, int) and not isinstance(status, bool):
                return status
        attempts = safe_retry_attempts(source.get("retry_attempts"))
        for attempt in reversed(attempts):
            status = attempt.get("http_status")
            if isinstance(status, int) and not isinstance(status, bool):
                return status
        provider_calls = source.get("provider_calls")
        if isinstance(provider_calls, list | tuple):
            for call in reversed(provider_calls):
                if not isinstance(call, dict):
                    continue
                status = call.get("http_status")
                if isinstance(status, int) and not isinstance(status, bool):
                    return status
                attempts = safe_retry_attempts(call.get("retry_attempts"))
                for attempt in reversed(attempts):
                    status = attempt.get("http_status")
                    if isinstance(status, int) and not isinstance(status, bool):
                        return status
    return None


_SCRIPT_SCAN_TABLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("memories", "archived_at", ("body", "tags_json")),
    ("context_sources", "archived_at", ("title", "body", "metadata_json")),
    (
        "context_observations",
        "archived_at",
        ("claim", "evidence_quote", "tags_json", "metadata_json"),
    ),
    ("world_state", "archived_at", ("key", "value_json", "category")),
)


def _generated_text_script_findings(
    repositories: PersistenceRepositories,
) -> tuple[GeneratedTextScriptDiagnostic, ...]:
    grouped: dict[
        tuple[str, str, str, str],
        tuple[int, str],
    ] = {}
    for table, inactive_column, fields in _SCRIPT_SCAN_TABLES:
        field_list = ", ".join(("id", "save_id", *fields))
        rows = repositories.connection.execute(
            f"""
            SELECT {field_list}
            FROM {table}
            WHERE {inactive_column} IS NULL
            """
        ).fetchall()
        for row in rows:
            save_id = row["save_id"]
            record_id = row["id"]
            if not isinstance(save_id, str) or not isinstance(record_id, str):
                continue
            for field in fields:
                value = _script_scan_field_value(row[field], field)
                violations = object_text_script_violations(
                    value,
                    allowed_scripts=frozenset(),
                    mode=SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
                    field_name=field,
                )
                for script in sorted({violation.script for violation in violations}):
                    key = (save_id, table, field, script)
                    count, example_id = grouped.get(key, (0, record_id))
                    grouped[key] = (count + 1, example_id)
    diagnostics = [
        GeneratedTextScriptDiagnostic(
            save_id=save_id,
            table=table,
            field=field,
            script=script,
            count=count,
            example_record_id=example_id,
        )
        for (save_id, table, field, script), (count, example_id) in grouped.items()
    ]
    diagnostics.sort(key=lambda item: (-item.count, item.table, item.field))
    return tuple(diagnostics[:20])


def _script_scan_field_value(value: object, field: str) -> object:
    if isinstance(value, str) and field.endswith("_json"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _group_summary(
    *,
    job_type: str,
    status: str,
    count: int,
    latest_http_status: int | None,
    latest_error: str | None,
) -> str:
    summary = f"{job_type} {status} {count} times"
    if latest_http_status is not None:
        summary = f"{summary}, latest HTTP {latest_http_status}"
    if latest_error:
        summary = f"{summary}: {latest_error}"
    return summary
