from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.world_context_retention_service import (
    WorldContextRetentionService,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_retention_expires_stale_and_excess_pending_suggestions(
    repositories: PersistenceRepositories,
) -> None:
    save, message_id = _persist_save(repositories)
    stale = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="location",
        entity_id="location-stale",
        field_path="status",
        proposed_value="quiet",
        source_message_ids=[message_id],
        suggestion_id="suggestion-stale",
    )
    suggestions = [
        repositories.add_context_update_suggestion(
            save_id=save.id,
            update_type="update",
            entity_type="location",
            entity_id=f"location-{index}",
            field_path="status",
            proposed_value=f"value-{index}",
            source_message_ids=[message_id],
            suggestion_id=f"suggestion-{index}",
        )
        for index in range(5)
    ]
    repositories.connection.execute(
        """
        UPDATE context_update_suggestions
        SET created_at = datetime('now', '-31 days')
        WHERE id = ?
        """,
        (stale.id,),
    )
    for index, suggestion in enumerate(suggestions):
        repositories.connection.execute(
            """
            UPDATE context_update_suggestions
            SET created_at = datetime('now', ?)
            WHERE id = ?
            """,
            (f"-{5 - index} minutes", suggestion.id),
        )
    repositories.connection.commit()

    result = WorldContextRetentionService(
        repositories=repositories,
        stale_pending_suggestion_days=30,
        max_pending_suggestions=3,
    ).prune(save.id)

    statuses = {
        suggestion.id: suggestion.status
        for suggestion in repositories.list_context_update_suggestions(save.id)
    }
    assert statuses["suggestion-stale"] == "expired"
    assert statuses["suggestion-0"] == "expired"
    assert statuses["suggestion-1"] == "expired"
    assert statuses["suggestion-2"] == "pending"
    assert statuses["suggestion-3"] == "pending"
    assert statuses["suggestion-4"] == "pending"
    assert result.expired_stale_suggestions == 1
    assert result.expired_excess_suggestions == 2
    audit = repositories.list_context_update_audit(save.id)
    assert [row.suggestion_id for row in audit] == [
        "suggestion-stale",
        "suggestion-0",
        "suggestion-1",
    ]
    assert {row.operation for row in audit} == {"suggestion_expired"}


def test_retention_prunes_old_archived_support_rows_without_touching_active_rows(
    repositories: PersistenceRepositories,
) -> None:
    save, message_id = _persist_save(repositories)
    active_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.current",
        value={"name": "Beacon Gallery"},
        source_message_id=message_id,
        state_id="state-active",
    )
    archived_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old",
        value={"name": "Old Hall"},
        source_message_id=message_id,
        state_id="state-archived",
    )
    active_memory = repositories.add_memory(
        save_id=save.id,
        body="The lens key is still held by Mara.",
        tags=["inventory"],
        source_message_id=message_id,
        memory_id="memory-active",
    )
    archived_memory = repositories.add_memory(
        save_id=save.id,
        body="A stale duplicate memory.",
        tags=["old"],
        source_message_id=message_id,
        memory_id="memory-archived",
    )
    active_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=message_id,
        covers_message_end_id=message_id,
        body="Mara reaches the beacon.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-active",
    )
    archived_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=message_id,
        covers_message_end_id=message_id,
        body="Old recap.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-archived",
    )
    active_source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id=active_memory.id,
        title="active",
        body="active body",
        context_source_id="context-active",
    )
    archived_source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id=archived_memory.id,
        title="archived",
        body="archived body",
        context_source_id="context-archived",
    )
    active_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Keep the beacon lit",
        source_message_id=message_id,
        thread_id="thread-active",
    )
    archived_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Old scene thread",
        source_message_id=message_id,
        thread_id="thread-archived",
    )

    repositories.archive_world_state(save_id=save.id, key=archived_state.key)
    repositories.archive_memory(archived_memory.id)
    repositories.archive_summary(archived_summary.id)
    repositories.archive_context_source(archived_source.id)
    repositories.archive_active_thread(archived_thread.id)
    for table_name, row_id in (
        ("world_state", archived_state.id),
        ("memories", archived_memory.id),
        ("summaries", archived_summary.id),
        ("context_sources", archived_source.id),
        ("active_threads", archived_thread.id),
    ):
        repositories.connection.execute(
            f"""
            UPDATE {table_name}
            SET archived_at = datetime('now', '-31 days')
            WHERE id = ?
            """,
            (row_id,),
        )
    repositories.connection.commit()

    result = WorldContextRetentionService(
        repositories=repositories,
        archived_support_retention_days=30,
    ).prune(save.id)

    assert repositories.list_world_state(save.id) == [active_state]
    assert repositories.list_memories(save.id) == [active_memory]
    assert repositories.list_summaries(save.id) == [active_summary]
    assert repositories.list_context_sources(save.id) == [active_source]
    assert repositories.list_active_threads(save.id) == [active_thread]
    assert result.pruned_archived_rows == {
        "world_state": 1,
        "memories": 1,
        "summaries": 1,
        "context_sources": 1,
        "active_threads": 1,
    }
    assert _row_count(repositories, "world_state", archived_state.id) == 0
    assert _row_count(repositories, "memories", archived_memory.id) == 0
    assert _row_count(repositories, "summaries", archived_summary.id) == 0
    assert _row_count(repositories, "context_sources", archived_source.id) == 0
    assert _row_count(repositories, "active_threads", archived_thread.id) == 0


def test_retention_prunes_audit_history_but_keeps_pending_suggestion_audit(
    repositories: PersistenceRepositories,
) -> None:
    save, message_id = _persist_save(repositories)
    pending = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="location",
        entity_id="location-pending",
        field_path="status",
        proposed_value="unstable",
        source_message_ids=[message_id],
        suggestion_id="suggestion-pending",
    )
    pending_audit = repositories.add_context_update_audit(
        save_id=save.id,
        suggestion_id=pending.id,
        operation="queued",
        entity_type="location",
        entity_id="location-pending",
        field_path="status",
        before="sealed",
        after="unstable",
        source_message_ids=[message_id],
        audit_id="audit-pending",
    )
    for index in range(5):
        audit = repositories.add_context_update_audit(
            save_id=save.id,
            operation="updated",
            entity_type="memory",
            entity_id=f"memory-{index}",
            field_path="body",
            before="old",
            after="new",
            source_message_ids=[message_id],
            audit_id=f"audit-{index}",
        )
        repositories.connection.execute(
            """
            UPDATE context_update_audit
            SET created_at = datetime('now', ?)
            WHERE id = ?
            """,
            (f"-{5 - index} minutes", audit.id),
        )
    repositories.connection.execute(
        """
        UPDATE context_update_audit
        SET created_at = datetime('now', '-60 minutes')
        WHERE id = ?
        """,
        (pending_audit.id,),
    )
    repositories.connection.commit()

    result = WorldContextRetentionService(
        repositories=repositories,
        max_context_update_audit_rows=2,
    ).prune(save.id)

    remaining_ids = [row.id for row in repositories.list_context_update_audit(save.id)]
    assert remaining_ids == ["audit-pending", "audit-3", "audit-4"]
    assert result.pruned_audit_rows == 3


def test_retention_keeps_newest_terminal_maintenance_jobs_and_active_jobs(
    repositories: PersistenceRepositories,
) -> None:
    save, _message_id = _persist_save(repositories)
    terminal_job_ids: list[str] = []
    for index in range(5):
        job = repositories.create_job(
            save_id=save.id,
            type="context_update",
            status="queued",
            payload={"index": index},
            job_id=f"job-terminal-{index}",
        )
        repositories.update_job(job.id, status="succeeded", result={"index": index})
        repositories.connection.execute(
            """
            UPDATE jobs
            SET created_at = datetime('now', ?),
                completed_at = datetime('now', ?)
            WHERE id = ?
            """,
            (f"-{5 - index} minutes", f"-{5 - index} minutes", job.id),
        )
        terminal_job_ids.append(job.id)
    queued = repositories.create_job(
        save_id=save.id,
        type="context_update",
        status="queued",
        payload={"active": True},
        job_id="job-active",
    )
    repositories.connection.commit()

    result = WorldContextRetentionService(
        repositories=repositories,
        max_terminal_jobs_per_save_type=2,
    ).prune(save.id)

    remaining = _job_ids(repositories, save.id)
    assert remaining == ["job-terminal-3", "job-terminal-4", queued.id]
    assert result.pruned_terminal_jobs == 3
    assert all(job_id not in remaining for job_id in terminal_job_ids[:3])


def test_retention_prunes_terminal_scheduler_jobs(
    repositories: PersistenceRepositories,
) -> None:
    save, _message_id = _persist_save(repositories)
    scheduler_job_types = (
        "world_suggestion_review",
        "state_extraction_retry",
        "state_extraction_retry_drain",
        "context_update_retry_drain",
        "web_maintenance_state_pruning",
        "web_maintenance_world_context_retention",
        "web_maintenance_memory_consolidation",
        "web_maintenance_character_registry_maintenance",
    )
    for job_type in scheduler_job_types:
        for index in range(3):
            job = repositories.create_job(
                save_id=save.id,
                type=job_type,
                status="queued",
                payload={"index": index},
                job_id=f"{job_type}-{index}",
            )
            repositories.update_job(job.id, status="succeeded", result={"index": index})
            repositories.connection.execute(
                """
                UPDATE jobs
                SET created_at = datetime('now', ?),
                    completed_at = datetime('now', ?)
                WHERE id = ?
                """,
                (f"-{3 - index} minutes", f"-{3 - index} minutes", job.id),
            )
    repositories.connection.commit()

    result = WorldContextRetentionService(
        repositories=repositories,
        max_terminal_jobs_per_save_type=1,
    ).prune(save.id)

    for job_type in scheduler_job_types:
        assert _job_ids(repositories, save.id, job_type=job_type) == [
            f"{job_type}-2"
        ]
    assert result.pruned_terminal_jobs == 16


def test_retention_counts_current_retention_job_toward_terminal_job_limit(
    repositories: PersistenceRepositories,
) -> None:
    save, _message_id = _persist_save(repositories)
    for index in range(5):
        job = repositories.create_job(
            save_id=save.id,
            type="world_context_retention",
            status="queued",
            payload={"index": index},
            job_id=f"job-retention-{index}",
        )
        repositories.update_job(job.id, status="succeeded", result={"index": index})
        repositories.connection.execute(
            """
            UPDATE jobs
            SET created_at = datetime('now', ?),
                completed_at = datetime('now', ?)
            WHERE id = ?
            """,
            (f"-{5 - index} minutes", f"-{5 - index} minutes", job.id),
        )
    repositories.connection.commit()

    result = WorldContextRetentionService(
        repositories=repositories,
        max_terminal_jobs_per_save_type=2,
    ).prune(save.id)

    remaining = _job_ids(repositories, save.id, job_type="world_context_retention")
    assert len(remaining) == 2
    assert "job-retention-4" in remaining
    assert result.pruned_terminal_jobs == 4


def _persist_save(repositories: PersistenceRepositories) -> tuple[SaveRecord, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon lens hums awake.",
        message_id="message-source",
    )
    return save, message.id


def _row_count(
    repositories: PersistenceRepositories,
    table_name: str,
    row_id: str,
) -> int:
    return int(
        repositories.connection.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE id = ?",
            (row_id,),
        ).fetchone()[0]
    )


def _job_ids(
    repositories: PersistenceRepositories,
    save_id: str,
    *,
    job_type: str = "context_update",
) -> list[str]:
    rows = repositories.connection.execute(
        """
        SELECT id
        FROM jobs
        WHERE save_id = ? AND type = ?
        ORDER BY created_at, rowid
        """,
        (save_id, job_type),
    ).fetchall()
    return [str(row["id"]) for row in rows]
