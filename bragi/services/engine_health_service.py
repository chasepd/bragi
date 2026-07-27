"""Metadata-only health checks for save continuity and prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bragi.persistence.models import JobRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.chat_history_settings import (
    chat_history_window_settings,
    narrator_planner_chat_history_window_settings,
)

CONTINUITY_JOB_TYPES = frozenset(
    {
        "context_search",
        "state_extraction",
        "state_extraction_retry",
        "context_update",
        "context_update_retry",
        "character_text_world_update",
        "character_text_world_update_retry",
        "summarization",
        "world_suggestion_review",
        "memory_consolidation",
        "state_pruning",
        "observation_curation_drain",
    }
)
STALE_PENDING_SUGGESTION_HOURS = 12
HIGH_RECENT_WINDOW_THRESHOLD = 12
HIGH_BASELINE_MESSAGE_CHARS = 32_000


@dataclass(frozen=True)
class EngineHealthWarning:
    code: str
    severity: str
    message: str
    count: int | None = None


@dataclass(frozen=True)
class ObservationCurationHealth:
    pending_count: int
    eligible_count: int
    leased_count: int
    oldest_pending_at: str | None
    oldest_pending_age_seconds: int | None
    total_attempt_count: int
    max_attempt_count: int
    terminal_failure_count: int


@dataclass(frozen=True)
class EngineHealthSnapshot:
    save_id: str
    active_message_count: int
    recent_player_message_window: int
    recent_narrator_message_window: int
    narrator_planner_recent_player_message_window: int
    narrator_planner_recent_narrator_message_window: int
    pending_suggestion_count: int
    stale_pending_suggestion_count: int
    summary_count: int
    recent_failed_continuity_job_count: int
    recent_failed_continuity_jobs_by_type: dict[str, int]
    observation_curation: ObservationCurationHealth
    latest_context_search: dict[str, object] | None
    latest_chat_prompt: dict[str, object] | None
    warnings: tuple[EngineHealthWarning, ...]


class EngineHealthService:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories

    def snapshot(self, save_id: str) -> EngineHealthSnapshot:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")

        messages = self.repositories.list_messages(save_id)
        history_settings = chat_history_window_settings(
            self.repositories,
            save_id=save_id,
        )
        planner_history_settings = narrator_planner_chat_history_window_settings(
            self.repositories,
            save_id=save_id,
        )
        pending_suggestions = self.repositories.list_context_update_suggestions(
            save_id,
            status="pending",
        )
        stale_pending_count = sum(
            1
            for suggestion in pending_suggestions
            if _is_stale_timestamp(
                suggestion.created_at,
                hours=STALE_PENDING_SUGGESTION_HOURS,
            )
        )
        summaries = self.repositories.list_summaries(save_id)
        recent_jobs = self.repositories.list_recent_jobs(
            save_id=save_id,
            seconds=14 * 24 * 60 * 60,
            limit=250,
        )
        failed_continuity_jobs = [
            job
            for job in recent_jobs
            if job.status in {"failed", "cancelled"}
            and job.type in CONTINUITY_JOB_TYPES
        ]
        failed_by_type = _job_type_counts(failed_continuity_jobs)
        latest_context_search = _latest_job_payload(
            recent_jobs,
            job_type="context_search",
        )
        latest_chat_prompt = _latest_chat_prompt_diagnostics(recent_jobs)
        curation_record = self.repositories.context_observation_curation_health(
            save_id
        )
        observation_curation = ObservationCurationHealth(
            pending_count=curation_record.pending_count,
            eligible_count=curation_record.eligible_count,
            leased_count=curation_record.leased_count,
            oldest_pending_at=curation_record.oldest_pending_at,
            oldest_pending_age_seconds=_timestamp_age_seconds(
                curation_record.oldest_pending_at
            ),
            total_attempt_count=curation_record.total_attempt_count,
            max_attempt_count=curation_record.max_attempt_count,
            terminal_failure_count=curation_record.terminal_failure_count,
        )
        warnings = _warnings(
            history_player=history_settings.player_messages,
            history_narrator=history_settings.narrator_messages,
            planner_history_player=planner_history_settings.player_messages,
            planner_history_narrator=planner_history_settings.narrator_messages,
            pending_count=len(pending_suggestions),
            stale_pending_count=stale_pending_count,
            failed_continuity_count=len(failed_continuity_jobs),
            failed_by_type=failed_by_type,
            latest_context_search=latest_context_search,
            latest_chat_prompt=latest_chat_prompt,
            observation_curation=observation_curation,
        )
        return EngineHealthSnapshot(
            save_id=save_id,
            active_message_count=len(messages),
            recent_player_message_window=history_settings.player_messages,
            recent_narrator_message_window=history_settings.narrator_messages,
            narrator_planner_recent_player_message_window=(
                planner_history_settings.player_messages
            ),
            narrator_planner_recent_narrator_message_window=(
                planner_history_settings.narrator_messages
            ),
            pending_suggestion_count=len(pending_suggestions),
            stale_pending_suggestion_count=stale_pending_count,
            summary_count=len(summaries),
            recent_failed_continuity_job_count=len(failed_continuity_jobs),
            recent_failed_continuity_jobs_by_type=failed_by_type,
            observation_curation=observation_curation,
            latest_context_search=latest_context_search,
            latest_chat_prompt=latest_chat_prompt,
            warnings=warnings,
        )


def _latest_job_payload(
    jobs: list[JobRecord],
    *,
    job_type: str,
) -> dict[str, object] | None:
    for job in jobs:
        if job.type != job_type:
            continue
        result = job.result if isinstance(job.result, dict) else {}
        return {
            "status": job.status,
            "error_present": bool(job.error),
            "result_counts": _selected_context_counts(result),
            "diagnostics": _context_search_diagnostics(result),
            "retrieval_degraded": result.get("retrieval_degraded") is True,
            **(
                {"retrieval_recovery": result["retrieval_recovery"]}
                if isinstance(result.get("retrieval_recovery"), str)
                else {}
            ),
        }
    return None


def _latest_chat_prompt_diagnostics(
    jobs: list[JobRecord],
) -> dict[str, object] | None:
    for job in jobs:
        if job.type != "chat_completion" or job.status != "succeeded":
            continue
        result = job.result if isinstance(job.result, dict) else {}
        diagnostics = result.get("prompt_context_diagnostics")
        return diagnostics if isinstance(diagnostics, dict) else None
    return None


def _selected_context_counts(result: dict[str, object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in result.items():
        if key.startswith("selected_") and isinstance(value, list):
            counts[key] = len(value)
    return counts


def _context_search_diagnostics(result: dict[str, object]) -> dict[str, object]:
    diagnostics = result.get("diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def _job_type_counts(jobs: list[JobRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.type] = counts.get(job.type, 0) + 1
    return counts


def _warnings(
    *,
    history_player: int,
    history_narrator: int,
    planner_history_player: int,
    planner_history_narrator: int,
    pending_count: int,
    stale_pending_count: int,
    failed_continuity_count: int,
    failed_by_type: dict[str, int],
    latest_context_search: dict[str, object] | None,
    latest_chat_prompt: dict[str, object] | None,
    observation_curation: ObservationCurationHealth,
) -> tuple[EngineHealthWarning, ...]:
    warnings: list[EngineHealthWarning] = []
    if observation_curation.terminal_failure_count:
        warnings.append(
            EngineHealthWarning(
                code="observation_curation_terminal_failures",
                severity="critical",
                message="Observation curation exhausted its retry budget.",
                count=observation_curation.terminal_failure_count,
            )
        )
    elif observation_curation.total_attempt_count:
        warnings.append(
            EngineHealthWarning(
                code="observation_curation_retries",
                severity="warning",
                message="Observation curation has retrying backlog.",
                count=observation_curation.pending_count,
            )
        )
    elif observation_curation.pending_count:
        warnings.append(
            EngineHealthWarning(
                code="observation_curation_backlog",
                severity="info",
                message="Observations are waiting for background curation.",
                count=observation_curation.pending_count,
            )
        )
    prose_history_total = history_player + history_narrator
    planner_history_total = planner_history_player + planner_history_narrator
    if any(
        value > HIGH_RECENT_WINDOW_THRESHOLD
        for value in (
            history_player,
            history_narrator,
            planner_history_player,
            planner_history_narrator,
        )
    ):
        warnings.append(
            EngineHealthWarning(
                code="high_recent_message_window",
                severity="warning",
                message="Recent transcript windows are high enough to crowd retrieval.",
                count=max(prose_history_total, planner_history_total),
            )
        )
    if stale_pending_count:
        warnings.append(
            EngineHealthWarning(
                code="stale_pending_suggestions",
                severity="warning",
                message=(
                    "Pending context suggestions are old enough to drift from canon."
                ),
                count=stale_pending_count,
            )
        )
    elif pending_count:
        warnings.append(
            EngineHealthWarning(
                code="pending_suggestions",
                severity="info",
                message=(
                    "Pending context suggestions may be included as noncanonical hints."
                ),
                count=pending_count,
            )
        )
    if failed_continuity_count:
        severity = "critical" if failed_by_type.get(
            "world_suggestion_review",
            0,
        ) >= 3 else "warning"
        warnings.append(
            EngineHealthWarning(
                code="failed_continuity_jobs",
                severity=severity,
                message="Recent continuity maintenance jobs failed or were cancelled.",
                count=failed_continuity_count,
            )
        )
    if _context_search_is_empty(latest_context_search):
        warnings.append(
            EngineHealthWarning(
                code="empty_context_search",
                severity="warning",
                message="Latest context search selected no retrieval context.",
            )
        )
    if _context_search_is_degraded(latest_context_search):
        warnings.append(
            EngineHealthWarning(
                code="degraded_context_search",
                severity="warning",
                message=(
                    "Latest context search recovered after provider or schema "
                    "failure."
                ),
            )
        )
    baseline_chars = _int_nested(latest_chat_prompt, "baseline_recent_message_chars")
    if baseline_chars is not None and baseline_chars > HIGH_BASELINE_MESSAGE_CHARS:
        warnings.append(
            EngineHealthWarning(
                code="large_recent_transcript",
                severity="warning",
                message=(
                    "Recent transcript baseline is large before retrieval is added."
                ),
                count=baseline_chars,
            )
        )
    return tuple(warnings)


def _context_search_is_empty(payload: dict[str, object] | None) -> bool:
    if payload is None or payload.get("status") != "succeeded":
        return False
    counts = payload.get("result_counts")
    return isinstance(counts, dict) and not any(
        isinstance(value, int) and value > 0 for value in counts.values()
    )


def _context_search_is_degraded(payload: dict[str, object] | None) -> bool:
    return (
        payload is not None
        and payload.get("status") == "succeeded"
        and payload.get("retrieval_degraded") is True
    )


def _int_nested(payload: dict[str, object] | None, key: str) -> int | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _is_stale_timestamp(value: str | None, *, hours: int) -> bool:
    parsed = _parse_sqlite_timestamp(value)
    if parsed is None:
        return False
    return datetime.now(UTC) - parsed >= timedelta(hours=hours)


def _timestamp_age_seconds(value: str | None) -> int | None:
    parsed = _parse_sqlite_timestamp(value)
    if parsed is None:
        return None
    return max(0, int((datetime.now(UTC) - parsed).total_seconds()))


def _parse_sqlite_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value[:19])
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
