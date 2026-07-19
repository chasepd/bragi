from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.job_lifecycle import (
    STALE_JOB_RECOVERY_ERROR,
    JobLifecycleService,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_create_queued_then_start_sets_running_and_started_at(
    repositories: PersistenceRepositories,
) -> None:
    _save(repositories, save_id="save-1")
    service = JobLifecycleService(repositories=repositories)

    queued = service.create_queued(
        type="image_generation",
        payload={"prompt": "candlelit archive"},
        save_id="save-1",
    )
    started = service.start(queued.id)

    assert queued.status == "queued"
    assert queued.started_at is None
    assert queued.completed_at is None
    assert started.id == queued.id
    assert started.status == "running"
    assert started.started_at is not None
    assert started.completed_at is None
    assert started.error is None
    assert started.payload == {"prompt": "candlelit archive"}


def test_lifecycle_terminal_states_set_completion_and_redact_errors(
    repositories: PersistenceRepositories,
) -> None:
    service = JobLifecycleService(repositories=repositories)
    summary_job = service.create_running(
        type="summary",
        payload={"save_id": "save-1"},
    )
    image_job = service.create_running(
        type="image_generation",
        payload={"prompt": "candlelit archive"},
    )
    chat_job = service.create_running(
        type="chat_completion",
        payload={"message_id": "message-1"},
    )

    succeeded = service.succeed(summary_job.id, result={"summary_id": "summary-1"})
    failed = service.fail(
        image_job.id,
        error=(
            "provider rejected api_key: sk-secret-value "
            "and Bearer super-secret-token"
        ),
        result={"phase": "generate"},
    )
    cancelled = service.cancel(
        chat_job.id,
        error="user cancelled request with token=super-secret-token",
        result={"reason": "user_cancelled"},
    )

    assert succeeded.status == "succeeded"
    assert succeeded.completed_at is not None
    assert succeeded.duration_ms is not None
    assert succeeded.result == {"summary_id": "summary-1"}
    assert succeeded.error is None
    assert failed.status == "failed"
    assert failed.completed_at is not None
    assert failed.duration_ms is not None
    assert failed.result == {"phase": "generate"}
    assert failed.error == "provider rejected api_key: [redacted] and Bearer [redacted]"
    assert cancelled.status == "cancelled"
    assert cancelled.completed_at is not None
    assert cancelled.duration_ms is not None
    assert cancelled.result == {"reason": "user_cancelled"}
    assert cancelled.error == "user cancelled request with token=[redacted]"

    stored_failed = _job_row(repositories, failed.id)
    stored_cancelled = _job_row(repositories, cancelled.id)
    assert stored_failed["error"] == (
        "provider rejected api_key: [redacted] and Bearer [redacted]"
    )
    assert stored_cancelled["error"] == "user cancelled request with token=[redacted]"
    assert "sk-secret-value" not in stored_failed["error"]
    assert "super-secret-token" not in stored_failed["error"]
    assert "super-secret-token" not in stored_cancelled["error"]

    failed_jobs = repositories.list_failed_jobs()
    assert [job.id for job in failed_jobs] == [failed.id, cancelled.id]
    assert [job.status for job in failed_jobs] == ["failed", "cancelled"]


def test_lifecycle_persists_terminal_diagnostic_snapshot_without_raw_payload(
    repositories: PersistenceRepositories,
) -> None:
    service = JobLifecycleService(repositories=repositories)
    job = service.create_running(
        type="character_text_image_generation",
        payload={
            "job_context": "character_text_attachment",
            "provider": "fake",
            "model": "fake-image",
            "text_message_id": "text-1",
            "prompt": "raw payload prompt must not be stored here",
        },
        request_context={
            "kind": "character_text_attachment",
            "text_message_id": "text-1",
            "prompt": "admin prompt captured separately",
        },
    )

    terminal = service.fail(
        job.id,
        error="provider failed token=secret-token",
        result={
            "final_error_category": "provider_error",
            "provider_payload": {"api_key": "secret-token"},
        },
    )

    assert terminal.diagnostics is not None
    request = cast(Mapping[str, object], terminal.diagnostics["request"])
    bragi = cast(Mapping[str, object], terminal.diagnostics["bragi"])
    timing = cast(Mapping[str, object], terminal.diagnostics["timing"])
    assert request["prompt"] == (
        "admin prompt captured separately"
    )
    assert bragi["status"] == "failed"
    assert timing["completed_at"] == terminal.completed_at
    assert timing["duration_ms"] == terminal.duration_ms
    assert "provider_payload" not in repr(terminal.diagnostics)
    assert "secret-token" not in repr(terminal.diagnostics)
    stored = repositories.get_persisted_job(job.id)
    assert stored is not None
    assert stored.diagnostics == terminal.diagnostics


def test_lifecycle_records_redacted_steps(
    repositories: PersistenceRepositories,
) -> None:
    service = JobLifecycleService(repositories=repositories)
    job = service.create_running(type="chat_turn", payload={})

    service.record_step(
        job.id,
        name="provider.chat",
        status="failed",
        provider="openrouter",
        model="openrouter/chat",
        task="chat",
        duration_ms=55,
        error="chat failed token=super-secret-token",
        metadata={"token_total": 10, "prompt": "unsafe"},
    )

    steps = repositories.list_job_steps(job.id)
    assert len(steps) == 1
    assert steps[0].error == "chat failed token=[redacted]"
    assert steps[0].metadata == {"token_total": 10}


def test_recover_stale_jobs_cancels_queued_and_running_jobs_with_metadata(
    repositories: PersistenceRepositories,
) -> None:
    service = JobLifecycleService(repositories=repositories)
    queued = service.create_queued(type="summary", payload={"step": "queued"})
    running = service.create_running(
        type="image_generation",
        payload={"step": "running"},
    )
    succeeded = service.succeed(
        service.create_running(
            type="chat_completion",
            payload={"step": "succeeded"},
        ).id,
        result={"message_id": "message-1"},
    )
    failed = service.fail(
        service.create_running(type="summary", payload={"step": "failed"}).id,
        error="summary failed",
    )
    cancelled = service.cancel(
        service.create_running(
            type="image_generation",
            payload={"step": "cancelled"},
        ).id,
        error="user cancelled",
    )

    recovered = service.recover_stale_jobs()

    assert [job.id for job in recovered] == [queued.id, running.id]
    recovered_by_id = {job.id: job for job in recovered}
    recovered_queued = recovered_by_id[queued.id]
    recovered_running = recovered_by_id[running.id]
    assert recovered_queued.status == "cancelled"
    assert recovered_queued.completed_at is not None
    assert recovered_queued.started_at is None
    assert recovered_queued.result == {
        "previous_status": "queued",
        "recovered_on_startup": True,
    }
    assert recovered_queued.error == STALE_JOB_RECOVERY_ERROR
    assert recovered_queued.diagnostics is not None
    recovered_queued_bragi = cast(
        Mapping[str, object],
        recovered_queued.diagnostics["bragi"],
    )
    assert recovered_queued_bragi["status"] == "cancelled"
    assert recovered_running.status == "cancelled"
    assert recovered_running.completed_at is not None
    assert recovered_running.started_at == running.started_at
    assert recovered_running.result == {
        "previous_status": "running",
        "recovered_on_startup": True,
    }
    assert recovered_running.error == STALE_JOB_RECOVERY_ERROR
    assert recovered_running.diagnostics is not None
    recovered_running_timing = cast(
        Mapping[str, object],
        recovered_running.diagnostics["timing"],
    )
    assert recovered_running_timing["completed_at"] == recovered_running.completed_at

    terminal_statuses = {
        job_id: _job_row(repositories, job_id)["status"]
        for job_id in (succeeded.id, failed.id, cancelled.id)
    }
    assert terminal_statuses == {
        succeeded.id: "succeeded",
        failed.id: "failed",
        cancelled.id: "cancelled",
    }


def test_recover_stale_jobs_cancels_queued_context_update_retries_by_default(
    repositories: PersistenceRepositories,
) -> None:
    service = JobLifecycleService(repositories=repositories)
    queued_retry = service.create_queued(
        type="context_update_retry",
        payload={"source_message_ids": ["message-1"]},
    )
    running_retry = service.create_running(
        type="context_update_retry",
        payload={"source_message_ids": ["message-2"]},
    )
    queued_summary = service.create_queued(type="summary", payload={"step": "queued"})

    recovered = service.recover_stale_jobs()

    assert [job.id for job in recovered] == [
        queued_retry.id,
        running_retry.id,
        queued_summary.id,
    ]
    recovered_by_id = {job.id: job for job in recovered}
    assert recovered_by_id[queued_retry.id].status == "cancelled"
    assert recovered_by_id[queued_retry.id].result == {
        "previous_status": "queued",
        "recovered_on_startup": True,
    }
    assert recovered_by_id[running_retry.id].status == "cancelled"
    assert recovered_by_id[running_retry.id].result == {
        "previous_status": "running",
        "recovered_on_startup": True,
    }
    assert recovered_by_id[queued_summary.id].status == "cancelled"
    assert recovered_by_id[queued_summary.id].result == {
        "previous_status": "queued",
        "recovered_on_startup": True,
    }


def test_recover_stale_jobs_can_preserve_explicit_queued_types(
    repositories: PersistenceRepositories,
) -> None:
    service = JobLifecycleService(repositories=repositories)
    queued_retry = service.create_queued(
        type="context_update_retry",
        payload={"source_message_ids": ["message-1"]},
    )

    recovered = service.recover_stale_jobs(
        preserve_queued_types=("context_update_retry",)
    )

    assert recovered == ()
    assert _job_row(repositories, queued_retry.id)["status"] == "queued"


def test_lifecycle_rejects_invalid_recovery_statuses(
    repositories: PersistenceRepositories,
) -> None:
    service = JobLifecycleService(repositories=repositories)
    service.create_running(type="summary", payload={"save_id": "save-1"})

    with pytest.raises(ValueError, match="Unknown job status: completed"):
        service.recover_stale_jobs(statuses=("running", "completed"))


def test_lifecycle_rejects_mutating_terminal_jobs(
    repositories: PersistenceRepositories,
) -> None:
    service = JobLifecycleService(repositories=repositories)
    succeeded = service.succeed(
        service.create_running(type="summary", payload={"save_id": "save-1"}).id,
        result={"summary_id": "summary-1"},
    )
    failed = service.fail(
        service.create_running(
            type="image_generation",
            payload={"prompt": "candlelit archive"},
        ).id,
        error="image generation failed",
    )
    cancelled = service.cancel(
        service.create_queued(
            type="chat_completion",
            payload={"message_id": "message-1"},
        ).id,
        error="user cancelled",
    )

    for job in (succeeded, failed, cancelled):
        with pytest.raises(ValueError, match=f"Cannot start job {job.id}"):
            service.start(job.id)
        with pytest.raises(ValueError, match=f"Cannot update terminal job {job.id}"):
            service.succeed(job.id, result={"ignored": True})
        with pytest.raises(ValueError, match=f"Cannot update terminal job {job.id}"):
            service.fail(job.id, error="still failed")
        with pytest.raises(ValueError, match=f"Cannot cancel job {job.id}"):
            service.cancel(job.id, error="too late")


def _save(
    repositories: PersistenceRepositories,
    *,
    save_id: str = "save-1",
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
        save_id=save_id,
    )


def _job_row(repositories: PersistenceRepositories, job_id: str) -> dict[str, Any]:
    row = repositories.connection.execute(
        """
        SELECT id, status, error, started_at, completed_at, duration_ms
        FROM jobs
        WHERE id = ?
        """,
        (job_id,),
    ).fetchone()
    assert row is not None
    return dict(row)
