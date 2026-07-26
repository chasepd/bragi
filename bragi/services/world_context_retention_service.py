"""Deterministic retention for save-owned maintenance support data."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import ContextUpdateSuggestionRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.job_lifecycle import JobLifecycleService

STALE_PENDING_SUGGESTION_DAYS = 30
MAX_PENDING_CONTEXT_UPDATE_SUGGESTIONS = 200
ARCHIVED_SUPPORT_RETENTION_DAYS = 30
MAX_CONTEXT_UPDATE_AUDIT_ROWS = 500
MAX_TERMINAL_MAINTENANCE_JOBS_PER_SAVE_TYPE = 50

ARCHIVED_SUPPORT_TABLES = (
    "world_state",
    "memories",
    "summaries",
    "context_sources",
    "active_threads",
)

TERMINAL_MAINTENANCE_JOB_TYPES = (
    "character_registry_maintenance",
    "context_cleanup",
    "context_precompute",
    "context_search",
    "context_update",
    "context_update_retry",
    "context_update_retry_drain",
    "guided_context_cleanup",
    "memory_consolidation",
    "scenario_evolution",
    "state_extraction_retry",
    "state_extraction_retry_drain",
    "state_pruning",
    "web_maintenance_character_registry_maintenance",
    "web_maintenance_memory_consolidation",
    "web_maintenance_state_pruning",
    "web_maintenance_world_context_retention",
    "world_context_retention",
    "world_suggestion_review",
)


@dataclass(frozen=True)
class WorldContextRetentionResult:
    save_id: str
    expired_stale_suggestions: int = 0
    expired_excess_suggestions: int = 0
    pruned_archived_rows: dict[str, int] = field(default_factory=dict)
    pruned_audit_rows: int = 0
    pruned_terminal_jobs: int = 0


class WorldContextRetentionService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        stale_pending_suggestion_days: int = STALE_PENDING_SUGGESTION_DAYS,
        max_pending_suggestions: int = MAX_PENDING_CONTEXT_UPDATE_SUGGESTIONS,
        archived_support_retention_days: int = ARCHIVED_SUPPORT_RETENTION_DAYS,
        max_context_update_audit_rows: int = MAX_CONTEXT_UPDATE_AUDIT_ROWS,
        max_terminal_jobs_per_save_type: int = (
            MAX_TERMINAL_MAINTENANCE_JOBS_PER_SAVE_TYPE
        ),
    ) -> None:
        self.repositories = repositories
        self.jobs = JobLifecycleService(repositories=repositories)
        self.stale_pending_suggestion_days = max(0, stale_pending_suggestion_days)
        self.max_pending_suggestions = max(0, max_pending_suggestions)
        self.archived_support_retention_days = max(0, archived_support_retention_days)
        self.max_context_update_audit_rows = max(0, max_context_update_audit_rows)
        self.max_terminal_jobs_per_save_type = max(0, max_terminal_jobs_per_save_type)

    def prune(self, save_id: str) -> WorldContextRetentionResult:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")
        job = self.jobs.create_running(
            save_id=save_id,
            type="world_context_retention",
            payload={
                "stale_pending_suggestion_days": self.stale_pending_suggestion_days,
                "max_pending_suggestions": self.max_pending_suggestions,
                "archived_support_retention_days": (
                    self.archived_support_retention_days
                ),
                "max_context_update_audit_rows": self.max_context_update_audit_rows,
                "max_terminal_jobs_per_save_type": (
                    self.max_terminal_jobs_per_save_type
                ),
            },
        )
        started_at = perf_counter()
        try:
            self.repositories.begin_transaction()
            stale = self._expire_stale_pending_suggestions(save_id)
            excess = self._expire_excess_pending_suggestions(save_id)
            pruned_archived = self._prune_old_archived_support_rows(save_id)
            pruned_audit_rows = self._prune_context_update_audit(save_id)
            pruned_terminal_jobs = self._prune_terminal_maintenance_jobs(
                save_id,
                current_job_type=job.type,
            )
            result = WorldContextRetentionResult(
                save_id=save_id,
                expired_stale_suggestions=len(stale),
                expired_excess_suggestions=len(excess),
                pruned_archived_rows=pruned_archived,
                pruned_audit_rows=pruned_audit_rows,
                pruned_terminal_jobs=pruned_terminal_jobs,
            )
            self.repositories.commit_transaction()
        except Exception as exc:
            self.repositories.rollback_transaction()
            self.jobs.fail(
                job.id,
                error=str(exc) or exc.__class__.__name__,
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        self.jobs.succeed(job.id, result=_result_json(result))
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            duration_ms=_elapsed_ms(started_at),
            expired_stale_suggestions=result.expired_stale_suggestions,
            expired_excess_suggestions=result.expired_excess_suggestions,
            pruned_audit_rows=result.pruned_audit_rows,
            pruned_terminal_jobs=result.pruned_terminal_jobs,
        )
        return result

    def _expire_stale_pending_suggestions(
        self,
        save_id: str,
    ) -> tuple[ContextUpdateSuggestionRecord, ...]:
        expired = tuple(
            self.repositories.expire_stale_context_update_suggestions(
                save_id,
                older_than_days=self.stale_pending_suggestion_days,
            )
        )
        self._add_expiration_audit(
            save_id=save_id,
            suggestions=expired,
            reason=(
                "Pending suggestion expired after "
                f"{self.stale_pending_suggestion_days} days without review."
            ),
        )
        return expired

    def _expire_excess_pending_suggestions(
        self,
        save_id: str,
    ) -> tuple[ContextUpdateSuggestionRecord, ...]:
        pending = tuple(
            self.repositories.list_context_update_suggestions(
                save_id,
                status="pending",
            )
        )
        excess_count = len(pending) - self.max_pending_suggestions
        if excess_count <= 0:
            return ()
        expired = tuple(
            self.repositories.update_context_update_suggestion_statuses(
                [suggestion.id for suggestion in pending[:excess_count]],
                status="expired",
            )
        )
        self._add_expiration_audit(
            save_id=save_id,
            suggestions=expired,
            reason=(
                "Pending suggestion expired because the review queue exceeded "
                f"{self.max_pending_suggestions} items."
            ),
        )
        return expired

    def _add_expiration_audit(
        self,
        *,
        save_id: str,
        suggestions: tuple[ContextUpdateSuggestionRecord, ...],
        reason: str,
    ) -> None:
        for suggestion in suggestions:
            self.repositories.add_context_update_audit(
                save_id=save_id,
                suggestion_id=suggestion.id,
                operation="suggestion_expired",
                entity_type=suggestion.entity_type,
                entity_id=suggestion.entity_id,
                field_path=suggestion.field_path,
                before=None,
                after=suggestion.proposed_value,
                reason=reason,
                confidence=suggestion.confidence,
                source_message_ids=suggestion.source_message_ids,
            )

    def _prune_old_archived_support_rows(self, save_id: str) -> dict[str, int]:
        pruned: dict[str, int] = {}
        for table_name in ARCHIVED_SUPPORT_TABLES:
            cursor = self.repositories.connection.execute(
                f"""
                DELETE FROM {table_name}
                WHERE save_id = ?
                  AND archived_at IS NOT NULL
                  AND julianday(archived_at) <= julianday('now', ?)
                """,
                (save_id, f"-{self.archived_support_retention_days} days"),
            )
            pruned[table_name] = max(0, cursor.rowcount)
        return pruned

    def _prune_context_update_audit(self, save_id: str) -> int:
        if self.max_context_update_audit_rows <= 0:
            keep_ids: tuple[str, ...] = ()
        else:
            rows = self.repositories.connection.execute(
                """
                SELECT id
                FROM context_update_audit
                WHERE save_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (save_id, self.max_context_update_audit_rows),
            ).fetchall()
            keep_ids = tuple(str(row["id"]) for row in rows)

        keep_clause = ""
        params: list[object] = [save_id]
        if keep_ids:
            keep_clause = f"AND id NOT IN ({_placeholders(len(keep_ids))})"
            params.extend(keep_ids)
        params.append(save_id)
        cursor = self.repositories.connection.execute(
            f"""
            DELETE FROM context_update_audit
            WHERE save_id = ?
              {keep_clause}
              AND (
                suggestion_id IS NULL
                OR suggestion_id NOT IN (
                    SELECT id
                    FROM context_update_suggestions
                    WHERE save_id = ? AND status = 'pending'
                )
              )
            """,
            tuple(params),
        )
        return max(0, cursor.rowcount)

    def _prune_terminal_maintenance_jobs(
        self,
        save_id: str,
        *,
        current_job_type: str,
    ) -> int:
        pruned = 0
        for job_type in TERMINAL_MAINTENANCE_JOB_TYPES:
            keep_count = self.max_terminal_jobs_per_save_type
            if job_type == current_job_type and keep_count > 0:
                keep_count -= 1
            rows = self.repositories.connection.execute(
                """
                SELECT id
                FROM jobs
                WHERE save_id = ?
                  AND type = ?
                  AND status IN ('succeeded', 'failed', 'cancelled')
                ORDER BY COALESCE(completed_at, started_at, created_at) DESC,
                         rowid DESC
                LIMIT -1 OFFSET ?
                """,
                (save_id, job_type, keep_count),
            ).fetchall()
            delete_ids = tuple(str(row["id"]) for row in rows)
            if not delete_ids:
                continue
            cursor = self.repositories.connection.execute(
                f"""
                DELETE FROM jobs
                WHERE id IN ({_placeholders(len(delete_ids))})
                """,
                delete_ids,
            )
            pruned += max(0, cursor.rowcount)
        return pruned


def _result_json(result: WorldContextRetentionResult) -> dict[str, object]:
    return {
        "expired_stale_suggestions": result.expired_stale_suggestions,
        "expired_excess_suggestions": result.expired_excess_suggestions,
        "pruned_archived_rows": dict(result.pruned_archived_rows),
        "pruned_audit_rows": result.pruned_audit_rows,
        "pruned_terminal_jobs": result.pruned_terminal_jobs,
    }


def _placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((perf_counter() - started_at) * 1000))
