from __future__ import annotations

from types import SimpleNamespace

from bragi_web.maintenance_diagnostics import maintenance_job_diagnostics


class _RepositoriesDouble:
    def __init__(self, jobs: list[object]) -> None:
        self._jobs = jobs

    def list_failed_jobs(self) -> list[object]:
        return self._jobs


class _FilteredRepositoriesDouble:
    def __init__(self, jobs: list[object]) -> None:
        self._jobs = jobs
        self.calls: list[dict[str, object]] = []

    def list_failed_jobs(
        self,
        *,
        types: tuple[str, ...],
        limit: int,
    ) -> list[object]:
        self.calls.append({"types": types, "limit": limit})
        return self._jobs


class _RecentRepositoriesDouble:
    def __init__(self, jobs: list[object]) -> None:
        self._jobs = jobs
        self.calls: list[dict[str, object]] = []

    def list_recent_jobs(
        self,
        *,
        types: tuple[str, ...],
        statuses: tuple[str, ...],
        seconds: int,
        limit: int,
    ) -> list[object]:
        self.calls.append(
            {
                "types": types,
                "statuses": statuses,
                "seconds": seconds,
                "limit": limit,
            }
        )
        return self._jobs


def test_memory_consolidation_jobs_are_reported_with_metrics() -> None:
    diagnostics = maintenance_job_diagnostics(
        _RepositoriesDouble(
            [
                SimpleNamespace(
                    id="job-1",
                    type="memory_consolidation",
                    status="failed",
                    save_id="save-1",
                    error="provider timed out",
                    started_at="2026-05-27T12:00:00Z",
                    completed_at="2026-05-27T12:01:00Z",
                    result={
                        "active_memory_count": 42,
                        "proposed_cluster_count": 3,
                        "rewritten_count": 1,
                        "archived_count": 4,
                        "rejected_count": 2,
                    },
                )
            ]
        )
    )

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.job_type == "memory_consolidation"
    assert diagnostic.summary == (
        "42 active, 3 proposed, 1 rewritten, 4 archived, 2 rejected"
    )
    assert diagnostic.metrics == {
        "active_memory_count": 42,
        "proposed_cluster_count": 3,
        "rewritten_count": 1,
        "archived_count": 4,
        "rejected_count": 2,
    }


def test_world_context_retention_jobs_are_reported_with_prune_metrics() -> None:
    diagnostics = maintenance_job_diagnostics(
        _RecentRepositoriesDouble(
            [
                SimpleNamespace(
                    id="job-1",
                    type="world_context_retention",
                    status="failed",
                    save_id="save-1",
                    error="retention failed",
                    started_at="2026-05-27T12:00:00Z",
                    completed_at="2026-05-27T12:01:00Z",
                    result={
                        "expired_stale_suggestions": 2,
                        "expired_excess_suggestions": 1,
                        "pruned_archived_rows": {"world_state": 4, "memories": 0},
                        "pruned_audit_rows": 7,
                        "pruned_terminal_jobs": 3,
                        "suggestion_ids": ["secret-suggestion"],
                    },
                ),
                SimpleNamespace(
                    id="job-2",
                    type="world_context_retention",
                    status="cancelled",
                    save_id="save-1",
                    error=None,
                    started_at="2026-05-27T12:02:00Z",
                    completed_at="2026-05-27T12:03:00Z",
                    result={"expired_stale_suggestions": 1},
                ),
            ]
        ),
        limit=2,
    )

    assert [diagnostic.status for diagnostic in diagnostics] == [
        "failed",
        "cancelled",
    ]
    failed = diagnostics[0]
    assert failed.summary == (
        "2 stale suggestions expired, 1 excess suggestion expired, "
        "4 archived rows pruned, 7 audit rows pruned, 3 terminal jobs pruned"
    )
    assert failed.metrics == {
        "expired_stale_suggestions": 2,
        "expired_excess_suggestions": 1,
        "pruned_archived_rows": {"world_state": 4, "memories": 0},
        "pruned_audit_rows": 7,
        "pruned_terminal_jobs": 3,
    }
    assert diagnostics[1].summary == "1 stale suggestion expired"


def test_diagnostic_errors_are_redacted_normalized_and_truncated() -> None:
    diagnostics = maintenance_job_diagnostics(
        _RepositoriesDouble(
            [
                SimpleNamespace(
                    id="job-1",
                    type="memory_consolidation",
                    status="failed",
                    save_id="save-1",
                    error=f"failed token=secret-token\n{'x' * 400}",
                    started_at=None,
                    completed_at=None,
                    result={},
                )
            ]
        )
    )

    assert len(diagnostics) == 1
    error = diagnostics[0].error
    assert error is not None
    assert len(error) == 240
    assert error.endswith("...")
    assert "\n" not in error
    assert "secret-token" not in error


def test_diagnostics_limit_latest_matching_jobs_and_pass_repo_filters() -> None:
    repositories = _FilteredRepositoriesDouble(
        [
            SimpleNamespace(
                id="unrelated",
                type="model_refresh",
                status="failed",
                save_id=None,
                error=None,
                started_at=None,
                completed_at=None,
                result={},
            ),
            SimpleNamespace(
                id="job-1",
                type="memory_consolidation",
                status="failed",
                save_id=None,
                error=None,
                started_at=None,
                completed_at=None,
                result={},
            ),
            SimpleNamespace(
                id="job-2",
                type="context_cleanup",
                status="failed",
                save_id=None,
                error=None,
                started_at=None,
                completed_at=None,
                result={},
            ),
        ]
    )

    diagnostics = maintenance_job_diagnostics(repositories, limit=1)

    assert [diagnostic.job_id for diagnostic in diagnostics] == ["job-2"]
    assert repositories.calls == [
        {
            "types": (
                "character_text_world_update_retry",
                "character_text_world_update_retry_drain",
                "context_cleanup",
                "dating_sim_maintenance",
                "guided_context_cleanup",
                "memory_consolidation",
                "observation_curation_drain",
                "post_turn_jobs",
                "state_extraction_retry",
                "state_extraction_retry_drain",
                "state_pruning",
                "summary_backfill",
                "world_context_retention",
            ),
            "limit": 1,
        }
    ]


def test_diagnostics_prefer_recent_job_filters_when_available() -> None:
    repositories = _RecentRepositoriesDouble(
        [
            SimpleNamespace(
                id="job-2",
                type="context_cleanup",
                status="failed",
                save_id=None,
                error=None,
                started_at=None,
                completed_at=None,
                result={"action_batches": 2},
            ),
            SimpleNamespace(
                id="job-1",
                type="memory_consolidation",
                status="failed",
                save_id=None,
                error=None,
                started_at=None,
                completed_at=None,
                result={},
            ),
        ]
    )

    diagnostics = maintenance_job_diagnostics(repositories, limit=1)

    assert [diagnostic.job_id for diagnostic in diagnostics] == ["job-2"]
    assert repositories.calls == [
        {
            "types": (
                "character_text_world_update_retry",
                "character_text_world_update_retry_drain",
                "context_cleanup",
                "dating_sim_maintenance",
                "guided_context_cleanup",
                "memory_consolidation",
                "observation_curation_drain",
                "post_turn_jobs",
                "state_extraction_retry",
                "state_extraction_retry_drain",
                "state_pruning",
                "summary_backfill",
                "world_context_retention",
            ),
            "statuses": ("failed", "cancelled"),
            "seconds": 0,
            "limit": 1,
        }
    ]


def test_observation_curation_diagnostics_suppress_error_text() -> None:
    diagnostics = maintenance_job_diagnostics(
        _RecentRepositoriesDouble(
            [
                SimpleNamespace(
                    id="job-curation",
                    type="observation_curation_drain",
                    status="failed",
                    save_id="save-1",
                    error="Private chronicle detail escaped from a provider.",
                    started_at="2026-07-26T12:00:00Z",
                    completed_at="2026-07-26T12:01:00Z",
                    result={},
                )
            ]
        )
    )

    assert len(diagnostics) == 1
    assert diagnostics[0].error is None
    assert diagnostics[0].summary == "No batch metrics"
