"""Metadata-only diagnostics for Bragi background jobs."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from bragi_web.bragi_adapter import bragi_diagnostics_bindings

_DIAGNOSTIC_JOB_TYPES = (
    "character_text_world_update_retry",
    "character_text_world_update_retry_drain",
    "context_cleanup",
    "dating_sim_maintenance",
    "guided_context_cleanup",
    "memory_consolidation",
    "post_turn_jobs",
    "state_pruning",
    "summary_backfill",
    "world_context_retention",
)
_DIAGNOSTIC_JOB_TYPE_SET = frozenset(_DIAGNOSTIC_JOB_TYPES)
_DEFAULT_LIMIT = 12
_MAX_TEXT_LENGTH = 240
_METRIC_KEYS = (
    "active_state_count",
    "batch_count",
    "completed_batch_count",
    "batch_size",
    "failed_batch_index",
    "review_only",
    "proposed_count",
    "archived_count",
    "rejected_count",
    "scanned_messages",
    "scan_batches",
    "cleanup_target_count",
    "action_batches",
    "completed_action_batches",
    "proposed_actions",
    "applied_actions",
    "queued_suggestions",
    "rejected_actions",
    "active_memory_count",
    "proposed_cluster_count",
    "rewritten_count",
    "summarized_message_count",
    "retained_recent_message_count",
    "archived_summary_count",
    "repaired_batch_count",
    "archives",
    "updates",
    "deleted_links",
    "expired_stale_suggestions",
    "expired_excess_suggestions",
    "pruned_audit_rows",
    "pruned_terminal_jobs",
)


@dataclass(frozen=True)
class MaintenanceJobDiagnostic:
    job_id: str
    job_type: str
    status: str
    save_id: str | None
    error: str | None
    started_at: str | None
    completed_at: str | None
    summary: str
    metrics: dict[str, object]


def maintenance_job_diagnostics(
    repositories: Any,
    *,
    limit: int = _DEFAULT_LIMIT,
) -> tuple[MaintenanceJobDiagnostic, ...]:
    bounded_limit = max(0, limit)
    if bounded_limit == 0:
        return ()
    jobs = _list_diagnostic_jobs(repositories, limit=bounded_limit)
    if jobs is None:
        return ()
    entries: list[MaintenanceJobDiagnostic] = []
    for job in jobs:
        if getattr(job, "type", "") not in _DIAGNOSTIC_JOB_TYPE_SET:
            continue
        if getattr(job, "status", "") not in {"failed", "cancelled"}:
            continue
        entries.append(_maintenance_job_diagnostic(job))
        if len(entries) >= bounded_limit:
            break
    return tuple(entries)


def _list_diagnostic_jobs(repositories: Any, *, limit: int) -> Iterable[Any] | None:
    recent_jobs = _list_recent_diagnostic_jobs(repositories, limit=limit)
    if recent_jobs is not None:
        return recent_jobs
    return _list_legacy_failed_jobs(repositories, limit=limit)


def _list_recent_diagnostic_jobs(
    repositories: Any,
    *,
    limit: int,
) -> Iterable[Any] | None:
    list_recent_jobs = getattr(repositories, "list_recent_jobs", None)
    if not callable(list_recent_jobs):
        return None
    parameters = inspect.signature(list_recent_jobs).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, object] = {}
    if accepts_kwargs or "types" in parameters:
        kwargs["types"] = _DIAGNOSTIC_JOB_TYPES
    if accepts_kwargs or "statuses" in parameters:
        kwargs["statuses"] = ("failed", "cancelled")
    if accepts_kwargs or "seconds" in parameters:
        kwargs["seconds"] = 0
    if accepts_kwargs or "limit" in parameters:
        kwargs["limit"] = limit
    return cast(Iterable[Any], list_recent_jobs(**kwargs))


def _list_legacy_failed_jobs(repositories: Any, *, limit: int) -> Iterable[Any] | None:
    list_failed_jobs = getattr(repositories, "list_failed_jobs", None)
    if not callable(list_failed_jobs):
        return None
    parameters = inspect.signature(list_failed_jobs).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, object] = {}
    if accepts_kwargs or "types" in parameters:
        kwargs["types"] = _DIAGNOSTIC_JOB_TYPES
    if accepts_kwargs or "limit" in parameters:
        kwargs["limit"] = limit
    return tuple(reversed(tuple(cast(Iterable[Any], list_failed_jobs(**kwargs)))))


def _maintenance_job_diagnostic(job: Any) -> MaintenanceJobDiagnostic:
    job_type = str(getattr(job, "type", ""))
    metrics = _metrics(job_type, getattr(job, "result", None))
    return MaintenanceJobDiagnostic(
        job_id=str(getattr(job, "id", "")),
        job_type=job_type,
        status=str(getattr(job, "status", "")),
        save_id=_optional_text(getattr(job, "save_id", None)),
        error=_optional_text(getattr(job, "error", None)),
        started_at=_optional_text(getattr(job, "started_at", None)),
        completed_at=_optional_text(getattr(job, "completed_at", None)),
        summary=_summary(job_type, metrics),
        metrics=metrics,
    )


def _metrics(job_type: str, result: object) -> dict[str, object]:
    if not isinstance(result, dict):
        return {}
    if job_type == "post_turn_jobs":
        return _post_turn_metrics(result)
    if job_type == "world_context_retention":
        return _world_context_retention_metrics(result)
    metrics: dict[str, object] = {}
    for key in _METRIC_KEYS:
        value = result.get(key)
        if isinstance(value, bool | int | float):
            metrics[key] = value
    return metrics


def _summary(job_type: str, metrics: dict[str, object]) -> str:
    if job_type == "state_pruning":
        return _state_pruning_summary(metrics)
    if job_type == "context_cleanup":
        return _context_cleanup_summary(metrics)
    if job_type == "guided_context_cleanup":
        return _guided_context_cleanup_summary(metrics)
    if job_type == "dating_sim_maintenance":
        return _dating_sim_maintenance_summary(metrics)
    if job_type == "memory_consolidation":
        return _memory_consolidation_summary(metrics)
    if job_type == "summary_backfill":
        return _summary_backfill_summary(metrics)
    if job_type == "post_turn_jobs":
        return _post_turn_summary(metrics)
    if job_type == "world_context_retention":
        return _world_context_retention_summary(metrics)
    return "No batch metrics"


def _dating_sim_maintenance_summary(metrics: dict[str, object]) -> str:
    applied = _int_metric(metrics, "applied_count") or 0
    deterministic = _int_metric(metrics, "deterministic_repair_count") or 0
    reviewable = _int_metric(metrics, "reviewable_repair_count") or 0
    if applied:
        return f"{applied} dating-sim repairs applied"
    return (
        f"{deterministic} deterministic and {reviewable} reviewable "
        "dating-sim repairs reported"
    )


def _world_context_retention_metrics(result: dict[str, object]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for key in _METRIC_KEYS:
        value = result.get(key)
        if isinstance(value, bool | int | float):
            metrics[key] = value
    archived = result.get("pruned_archived_rows")
    if isinstance(archived, dict):
        rows = {
            str(key): value
            for key, value in archived.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        if rows:
            metrics["pruned_archived_rows"] = rows
    return metrics


def _post_turn_metrics(result: dict[str, object]) -> dict[str, object]:
    jobs = result.get("jobs")
    if not isinstance(jobs, list):
        return {}
    failed_steps: list[str] = []
    skipped_steps: list[str] = []
    outcome_result: dict[str, object] | None = None
    for item in jobs:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        status = item.get("status")
        if not isinstance(name, str) or not isinstance(status, str):
            continue
        if status == "failed":
            failed_steps.append(name)
        elif status == "skipped":
            skipped_steps.append(name)
        if name == "outcome" and isinstance(item.get("result"), dict):
            outcome_result = item["result"]
    metrics: dict[str, object] = {}
    if failed_steps:
        metrics["failed_steps"] = failed_steps
    if skipped_steps:
        metrics["skipped_steps"] = skipped_steps
    if outcome_result is not None:
        _add_text_metric(metrics, "outcome_provider", outcome_result.get("provider"))
        _add_text_metric(metrics, "outcome_model", outcome_result.get("model"))
        _add_text_metric(metrics, "outcome_error", outcome_result.get("error"))
        _add_text_metric(
            metrics,
            "outcome_error_category",
            outcome_result.get("error_category"),
        )
        fallback_attempted = outcome_result.get("fallback_attempted")
        if isinstance(fallback_attempted, bool):
            metrics["outcome_fallback_attempted"] = fallback_attempted
        _add_text_metric(
            metrics,
            "outcome_fallback_skipped_reason",
            outcome_result.get("fallback_skipped_reason"),
        )
        _add_text_metric(
            metrics,
            "outcome_fallback_provider",
            outcome_result.get("fallback_provider"),
        )
        _add_text_metric(
            metrics,
            "outcome_fallback_model",
            outcome_result.get("fallback_model_id"),
        )
    return metrics


def _post_turn_summary(metrics: dict[str, object]) -> str:
    parts: list[str] = []
    failed_steps = _list_metric(metrics, "failed_steps")
    if failed_steps:
        parts.append(f"{', '.join(failed_steps)} failed")
    skipped_steps = _list_metric(metrics, "skipped_steps")
    if skipped_steps:
        parts.append(f"{', '.join(skipped_steps)} skipped")
    provider = _text_metric(metrics, "outcome_provider")
    model = _text_metric(metrics, "outcome_model")
    if provider and model:
        parts.append(f"outcome provider {provider}/{model}")
    elif provider:
        parts.append(f"outcome provider {provider}")
    error = _text_metric(metrics, "outcome_error")
    if error:
        parts.append(f"outcome error {error}")
    skipped_reason = _text_metric(
        metrics,
        "outcome_fallback_skipped_reason",
    )
    if skipped_reason:
        parts.append(f"outcome fallback skipped {skipped_reason}")
    elif metrics.get("outcome_fallback_attempted") is True:
        parts.append("outcome fallback attempted")
    return ", ".join(parts) or "No post-turn diagnostics"


def _state_pruning_summary(metrics: dict[str, object]) -> str:
    parts: list[str] = []
    completed = _int_metric(metrics, "completed_batch_count")
    batch_count = _int_metric(metrics, "batch_count")
    if completed is not None and batch_count is not None:
        parts.append(f"{completed}/{batch_count} batches")
    archived = _int_metric(metrics, "archived_count")
    rejected = _int_metric(metrics, "rejected_count")
    if archived is not None:
        parts.append(f"{archived} archived")
    if rejected is not None:
        parts.append(f"{rejected} rejected")
    failed_batch = _int_metric(metrics, "failed_batch_index")
    if failed_batch is not None:
        parts.append(f"failed at batch {failed_batch + 1}")
    return ", ".join(parts) or "No batch metrics"


def _context_cleanup_summary(metrics: dict[str, object]) -> str:
    parts: list[str] = []
    completed = _int_metric(metrics, "completed_action_batches")
    action_batches = _int_metric(metrics, "action_batches")
    if completed is not None and action_batches is not None:
        parts.append(f"{completed}/{action_batches} action batches")
    elif action_batches is not None:
        parts.append(f"{action_batches} action batches")
    scan_batches = _int_metric(metrics, "scan_batches")
    if scan_batches is not None:
        parts.append(f"{scan_batches} scan batches")
    applied = _int_metric(metrics, "applied_actions")
    rejected = _int_metric(metrics, "rejected_actions")
    if applied is not None:
        parts.append(f"{applied} applied")
    if rejected is not None:
        parts.append(f"{rejected} rejected")
    return ", ".join(parts) or "No batch metrics"


def _guided_context_cleanup_summary(metrics: dict[str, object]) -> str:
    parts: list[str] = []
    completed = _int_metric(metrics, "completed_action_batches")
    action_batches = _int_metric(metrics, "action_batches")
    if completed is not None and action_batches is not None:
        parts.append(f"{completed}/{action_batches} action batches")
    elif action_batches is not None:
        parts.append(f"{action_batches} action batches")
    queued = _int_metric(metrics, "queued_suggestions")
    rejected = _int_metric(metrics, "rejected_actions")
    if queued is not None:
        parts.append(f"{queued} queued")
    if rejected is not None:
        parts.append(f"{rejected} rejected")
    return ", ".join(parts) or "No batch metrics"


def _memory_consolidation_summary(metrics: dict[str, object]) -> str:
    parts: list[str] = []
    active = _int_metric(metrics, "active_memory_count")
    proposed = _int_metric(metrics, "proposed_cluster_count")
    rewritten = _int_metric(metrics, "rewritten_count")
    archived = _int_metric(metrics, "archived_count")
    rejected = _int_metric(metrics, "rejected_count")
    if active is not None:
        parts.append(f"{active} active")
    if proposed is not None:
        parts.append(f"{proposed} proposed")
    if rewritten is not None:
        parts.append(f"{rewritten} rewritten")
    if archived is not None:
        parts.append(f"{archived} archived")
    if rejected is not None:
        parts.append(f"{rejected} rejected")
    return ", ".join(parts) or "No batch metrics"


def _summary_backfill_summary(metrics: dict[str, object]) -> str:
    parts: list[str] = []
    compacted = _int_metric(metrics, "summarized_message_count")
    batches = _int_metric(metrics, "batch_count")
    archived = _int_metric(metrics, "archived_summary_count")
    repaired = _int_metric(metrics, "repaired_batch_count")
    if compacted is not None:
        parts.append(f"{compacted} messages compacted")
    if batches is not None:
        parts.append(f"{batches} batches")
    if archived is not None:
        parts.append(f"{archived} summaries archived")
    if repaired:
        parts.append(f"{repaired} repairs")
    return ", ".join(parts) or "No backfill metrics"


def _world_context_retention_summary(metrics: dict[str, object]) -> str:
    parts: list[str] = []
    stale = _int_metric(metrics, "expired_stale_suggestions")
    excess = _int_metric(metrics, "expired_excess_suggestions")
    archived = _int_dict_metric(metrics, "pruned_archived_rows")
    archived_total = sum(archived.values()) if archived else None
    audit = _int_metric(metrics, "pruned_audit_rows")
    terminal_jobs = _int_metric(metrics, "pruned_terminal_jobs")
    if stale is not None:
        parts.append(_count_phrase(stale, "stale suggestion", "expired"))
    if excess is not None:
        parts.append(_count_phrase(excess, "excess suggestion", "expired"))
    if archived_total is not None:
        parts.append(_count_phrase(archived_total, "archived row", "pruned"))
    if audit is not None:
        parts.append(_count_phrase(audit, "audit row", "pruned"))
    if terminal_jobs is not None:
        parts.append(_count_phrase(terminal_jobs, "terminal job", "pruned"))
    return ", ".join(parts) or "No retention metrics"


def _int_metric(metrics: dict[str, object], key: str) -> int | None:
    value = metrics.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _int_dict_metric(metrics: dict[str, object], key: str) -> dict[str, int]:
    value = metrics.get(key)
    if not isinstance(value, dict):
        return {}
    return {
        str(item_key): item_value
        for item_key, item_value in value.items()
        if isinstance(item_value, int) and not isinstance(item_value, bool)
    }


def _count_phrase(count: int, noun: str, verb: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {noun}{suffix} {verb}"


def _list_metric(metrics: dict[str, object], key: str) -> list[str]:
    value = metrics.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _text_metric(metrics: dict[str, object], key: str) -> str | None:
    value = metrics.get(key)
    return value if isinstance(value, str) and value else None


def _add_text_metric(
    metrics: dict[str, object],
    key: str,
    value: object,
) -> None:
    text = _optional_text(value)
    if text is not None:
        metrics[key] = text


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    redacted = bragi_diagnostics_bindings().redact_diagnostic_text(value)
    if redacted is None:
        return None
    text = redacted.strip().replace("\r", " ").replace("\n", " ")
    if len(text) > _MAX_TEXT_LENGTH:
        text = f"{text[: _MAX_TEXT_LENGTH - 3]}..."
    return text or None
