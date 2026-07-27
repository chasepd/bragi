from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.character_registry_maintenance_service import (
    CHARACTER_MAINTENANCE_TURN_CADENCE,
)
from bragi.services.memory_consolidation_service import MEMORY_CONSOLIDATION_THRESHOLD
from bragi.services.model_preferences import set_save_model_override_preference
from bragi_web.jobs import JobRegistry
from bragi_web.runtime import SaveEventHub
from bragi_web.scheduler import (
    CHARACTER_REGISTRY_MAINTENANCE_TASK,
    CHARACTER_TEXT_WORLD_UPDATE_RETRY_DRAIN_TASK,
    CONTEXT_UPDATE_RETRY_DRAIN_TASK,
    MEMORY_CONSOLIDATION_TASK,
    OBSERVATION_CURATION_DRAIN_TASK,
    STATE_EXTRACTION_RETRY_DRAIN_TASK,
    STATE_PRUNING_TASK,
    WEB_MAINTENANCE_CHARACTER_REGISTRY_MAINTENANCE_JOB,
    WORLD_CONTEXT_RETENTION_TASK,
    WORLD_SUGGESTION_REVIEW_TASK,
    WebMaintenanceScheduler,
)


def test_world_suggestion_scheduler_queues_review_for_active_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save_with_pending_suggestion(repositories)
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.calls == [save_id]
    task = repositories.get_scheduled_task(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        save_id=save_id,
    )
    assert task is not None
    assert task.last_job_id is not None
    assert task.failure_count == 0
    assert task.result == {
        "active_save_id": save_id,
        "error": None,
        "status": "reviewed",
    }
    jobs = repositories.list_recent_jobs(
        save_id=save_id,
        types=(WORLD_SUGGESTION_REVIEW_TASK,),
    )
    assert [job.status for job in jobs] == ["succeeded"]


def test_scheduler_persists_compact_task_result(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save_with_pending_suggestion(repositories)
    runtime = _VerboseResultRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    task = repositories.get_scheduled_task(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        save_id=save_id,
    )
    assert task is not None
    assert task.result == {
        "active_save_id": save_id,
        "error": None,
        "status": "reviewed",
    }


def test_scheduler_marks_web_job_failed_when_runtime_result_has_error(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save_with_pending_suggestion(repositories)
    runtime = _ErrorResultRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    task = repositories.get_scheduled_task(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        save_id=save_id,
    )
    assert task is not None
    assert task.failure_count == 1
    assert task.error == "review exploded"
    jobs = repositories.list_recent_jobs(
        save_id=save_id,
        types=(WORLD_SUGGESTION_REVIEW_TASK,),
    )
    assert [job.status for job in jobs] == ["failed"]


def test_world_suggestion_scheduler_skips_active_save_without_pending_suggestions(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.calls == []
    assert (
        repositories.get_scheduled_task(
            task_type=WORLD_SUGGESTION_REVIEW_TASK,
            save_id=save_id,
        )
        is None
    )
    assert repositories.list_recent_jobs(types=(WORLD_SUGGESTION_REVIEW_TASK,)) == []


def test_world_suggestion_scheduler_skips_pending_suggestions_without_model(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save_with_pending_suggestion(
        repositories,
        configure_context_update_model=False,
    )
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.calls == []
    assert (
        repositories.get_scheduled_task(
            task_type=WORLD_SUGGESTION_REVIEW_TASK,
            save_id=save_id,
        )
        is None
    )
    assert repositories.list_recent_jobs(types=(WORLD_SUGGESTION_REVIEW_TASK,)) == []


def test_scheduler_rechecks_policy_after_lease_before_queueing_job(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save_with_pending_suggestion(repositories)
    suggestion = repositories.list_context_update_suggestions(
        save_id,
        status="pending",
    )[0]
    runtime = _ReviewRuntime(active_save_id=save_id)
    disappearing_repositories = _DismissSuggestionAfterLeaseRepositories(
        repositories,
        suggestion_id=suggestion.id,
    )
    state = _scheduler_state(disappearing_repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.calls == []
    assert repositories.list_context_update_suggestions(
        save_id,
        status="pending",
    ) == []
    task = repositories.get_scheduled_task(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        save_id=save_id,
    )
    assert task is not None
    assert task.failure_count == 0
    assert task.last_job_id is None
    assert task.lease_until is None
    assert task.result == {
        "skip_reason": "task_policy_not_due",
        "status": "skipped",
    }
    assert repositories.list_recent_jobs(types=(WORLD_SUGGESTION_REVIEW_TASK,)) == []


def test_world_suggestion_scheduler_queues_due_pending_suggestions_on_inactive_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    active_save_id = _save(repositories, title="Active Save")
    inactive_save_id = _save_with_pending_suggestion(
        repositories,
        title="Inactive Save",
    )
    runtime = _ReviewRuntime(active_save_id=active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.calls == [inactive_save_id]
    assert repositories.get_scheduled_task(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        save_id=inactive_save_id,
    ) is not None


def test_world_suggestion_scheduler_queues_review_for_user_active_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    user = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    process_active_save_id = _save(repositories, title="Process Active Save")
    user_active_save_id = _save_with_pending_suggestion(
        repositories,
        title="User Active Save",
        owner_user_id=user.id,
    )
    repositories.set_user_active_save_id(
        user_id=user.id,
        save_id=user_active_save_id,
    )
    runtime = _ReviewRuntime(active_save_id=process_active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.calls == [user_active_save_id]
    task = repositories.get_scheduled_task(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        save_id=user_active_save_id,
    )
    assert task is not None
    assert task.last_job_id is not None
    assert task.failure_count == 0


def test_scheduler_runs_due_suggestion_review_for_dormant_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    process_active_save_id = _save(repositories, title="Process Active Save")
    dormant_save_id = _save_with_pending_suggestion(
        repositories,
        title="Dormant Save",
    )
    repositories.upsert_scheduled_task(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        save_id=dormant_save_id,
        interval_seconds=60,
        payload={"active_save_only": True},
        due_now=True,
    )
    runtime = _ReviewRuntime(active_save_id=process_active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.calls == [dormant_save_id]
    task = repositories.get_scheduled_task(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        save_id=dormant_save_id,
    )
    assert task is not None
    assert task.last_job_id is not None
    assert task.failure_count == 0
    assert [
        job.status
        for job in repositories.list_recent_jobs(types=(WORLD_SUGGESTION_REVIEW_TASK,))
    ] == ["succeeded"]


def test_scheduler_leaves_due_active_save_only_dormant_task_untouched(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    process_active_save_id = _save(repositories, title="Process Active Save")
    dormant_save_id = _save(repositories, title="Dormant Save")
    repositories.upsert_scheduled_task(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        save_id=dormant_save_id,
        interval_seconds=60,
        payload={"active_save_only": True},
        due_now=True,
    )
    runtime = _ReviewRuntime(active_save_id=process_active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.calls == []
    task = repositories.get_scheduled_task(
        task_type=WORLD_SUGGESTION_REVIEW_TASK,
        save_id=dormant_save_id,
    )
    assert task is not None
    assert task.last_job_id is None
    assert task.failure_count == 0
    assert task.result is None
    assert (
        repositories.list_due_scheduled_tasks(
            task_types=(WORLD_SUGGESTION_REVIEW_TASK,),
            save_id=dormant_save_id,
        )
        != []
    )


def test_scheduler_limits_due_suggestion_reviews_for_dormant_saves(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    process_active_save_id = _save(repositories, title="Process Active Save")
    dormant_save_ids = [
        _save_with_pending_suggestion(repositories, title=f"Dormant Save {index}")
        for index in range(11)
    ]
    for save_id in dormant_save_ids:
        repositories.upsert_scheduled_task(
            task_type=WORLD_SUGGESTION_REVIEW_TASK,
            save_id=save_id,
            interval_seconds=60,
            payload={"active_save_only": True},
            due_now=True,
        )
    runtime = _ReviewRuntime(active_save_id=process_active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert len(runtime.calls) == 10
    assert set(runtime.calls).issubset(set(dormant_save_ids))
    review_jobs = repositories.list_recent_jobs(
        types=(WORLD_SUGGESTION_REVIEW_TASK,),
    )
    assert len(review_jobs) == 10


def test_scheduler_skips_when_same_save_job_is_active(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save_with_pending_suggestion(repositories)
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        release = asyncio.Event()

        async def blocking_worker(_handle: object) -> object:
            await release.wait()
            return {}

        await state.jobs.create(
            "chat_turn",
            blocking_worker,
            save_id=save_id,
            exclusive_key=f"chat_turn:{save_id}",
        )
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        try:
            await scheduler.run_once()
        finally:
            release.set()
            await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.calls == []
    assert (
        repositories.get_scheduled_task(
            task_type=WORLD_SUGGESTION_REVIEW_TASK,
            save_id=save_id,
        )
        is None
    )


def test_scheduler_queues_state_pruning_for_active_save_when_due(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    repositories.set_model_preference(
        task="state_pruning",
        provider="fake",
        model_id="fake-pruner",
    )
    _append_narrator_messages(repositories, save_id=save_id, count=5)
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.state_pruning_calls == [save_id]
    task = repositories.get_scheduled_task(
        task_type=STATE_PRUNING_TASK,
        save_id=save_id,
    )
    assert task is not None
    assert task.failure_count == 0
    assert task.last_job_id is not None


def test_scheduler_drains_context_update_retry_for_active_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    retry_job = repositories.create_job(
        save_id=save_id,
        type="context_update_retry",
        status="queued",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.context_retry_calls == [save_id]
    task = repositories.get_scheduled_task(
        task_type=CONTEXT_UPDATE_RETRY_DRAIN_TASK,
        save_id=save_id,
    )
    assert task is not None
    assert task.failure_count == 0
    assert retry_job.id in {
        job.id for job in repositories.list_jobs_by_status(("queued",))
    }


def test_scheduler_drains_state_extraction_retry_for_active_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    retry_job = repositories.create_job(
        save_id=save_id,
        type="state_extraction_retry",
        status="queued",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.state_retry_calls == [save_id]
    assert runtime.context_retry_calls == []
    task = repositories.get_scheduled_task(
        task_type=STATE_EXTRACTION_RETRY_DRAIN_TASK,
        save_id=save_id,
    )
    assert task is not None
    assert task.failure_count == 0
    assert retry_job.id in {
        job.id for job in repositories.list_jobs_by_status(("queued",))
    }


def test_scheduler_drains_state_extraction_retry_for_inactive_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    active_save_id = _save(repositories, title="Active Save")
    inactive_save_id = _save(repositories, title="Inactive Save")
    retry_job = repositories.create_job(
        save_id=inactive_save_id,
        type="state_extraction_retry",
        status="queued",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    runtime = _ReviewRuntime(active_save_id=active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.state_retry_calls == [inactive_save_id]
    assert runtime.context_retry_calls == []
    task = repositories.get_scheduled_task(
        task_type=STATE_EXTRACTION_RETRY_DRAIN_TASK,
        save_id=inactive_save_id,
    )
    assert task is not None
    assert task.failure_count == 0
    assert retry_job.id in {
        job.id for job in repositories.list_jobs_by_status(("queued",))
    }


def test_scheduler_prioritizes_state_retry_before_context_retry(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    repositories.create_job(
        save_id=save_id,
        type="state_extraction_retry",
        status="queued",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    repositories.create_job(
        save_id=save_id,
        type="context_update_retry",
        status="queued",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.state_retry_calls == [save_id]
    assert runtime.context_retry_calls == []


def test_scheduler_drains_context_update_retry_for_inactive_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    active_save_id = _save(repositories, title="Active Save")
    inactive_save_id = _save(repositories, title="Inactive Save")
    retry_job = repositories.create_job(
        save_id=inactive_save_id,
        type="context_update_retry",
        status="queued",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    runtime = _ReviewRuntime(active_save_id=active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.context_retry_calls == [inactive_save_id]
    task = repositories.get_scheduled_task(
        task_type=CONTEXT_UPDATE_RETRY_DRAIN_TASK,
        save_id=inactive_save_id,
    )
    assert task is not None
    assert task.failure_count == 0
    assert retry_job.id in {
        job.id for job in repositories.list_jobs_by_status(("queued",))
    }


def test_scheduler_drains_observation_curation_for_inactive_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    active_save_id = _save(repositories, title="Active Save")
    inactive_save_id = _save(repositories, title="Inactive Save")
    repositories.set_model_preference(
        task="memory_curation",
        provider="fake",
        model_id="fake-curator",
    )
    repositories.add_context_observation(
        save_id=inactive_save_id,
        observation_type="event",
        claim="The beacon was relit.",
        evidence_quote="The beacon was relit.",
        source_message_ids=[],
        scope="durable",
        confidence=0.9,
    )
    runtime = _ReviewRuntime(active_save_id=active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.observation_curation_calls == [inactive_save_id]
    task = repositories.get_scheduled_task(
        task_type=OBSERVATION_CURATION_DRAIN_TASK,
        save_id=inactive_save_id,
    )
    assert task is not None
    assert task.payload["active_save_only"] is False


def test_scheduler_persists_only_metadata_for_curation_failures(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Private Backlog")
    repositories.add_context_observation(
        save_id=save_id,
        observation_type="event",
        claim="A private chronicle detail.",
        evidence_quote="A private chronicle detail.",
        source_message_ids=[],
        scope="durable",
        confidence=0.9,
    )
    set_save_model_override_preference(
        repositories,
        save_id=save_id,
        task="memory_curation",
        provider="fake",
        model_id="fake-curator",
    )
    runtime = _CurationErrorRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    task = repositories.get_scheduled_task(
        task_type=OBSERVATION_CURATION_DRAIN_TASK,
        save_id=save_id,
    )
    assert task is not None
    assert task.error == "observation_curation_failed"
    assert task.result == {
        "active_save_id": save_id,
        "error_present": True,
        "status": None,
    }
    assert "private chronicle" not in json.dumps(task.result).lower()


def test_scheduler_skips_unconfigured_curation_backlogs_without_starvation(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_ids = [
        _save(repositories, title=f"Backlog {index}") for index in range(11)
    ]
    for index, save_id in enumerate(save_ids):
        repositories.add_context_observation(
            save_id=save_id,
            observation_type="event",
            claim=f"Observation {index}",
            evidence_quote=f"Observation {index}",
            source_message_ids=[],
            scope="durable",
            confidence=0.9,
        )
    runnable_save_id = save_ids[-1]
    set_save_model_override_preference(
        repositories,
        save_id=runnable_save_id,
        task="memory_curation",
        provider="fake",
        model_id="fake-curator",
    )
    runtime = _ReviewRuntime(active_save_id=save_ids[0])
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.observation_curation_calls == [runnable_save_id]


def test_scheduler_replenishes_curation_candidates_after_due_checks(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_ids = [_save(repositories, title=f"Backlog {index}") for index in range(11)]
    for index, save_id in enumerate(save_ids):
        repositories.add_context_observation(
            save_id=save_id,
            observation_type="event",
            claim=f"Observation {index}",
            evidence_quote=f"Observation {index}",
            source_message_ids=[],
            scope="durable",
            confidence=0.9,
        )
        set_save_model_override_preference(
            repositories,
            save_id=save_id,
            task="memory_curation",
            provider="fake",
            model_id="fake-curator",
        )
    runtime = _ReviewRuntime(active_save_id=save_ids[0])
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)
        repositories.connection.execute(
            """
            UPDATE scheduled_tasks
            SET next_run_at = '2000-01-01 00:00:00'
            WHERE task_type = ?
            """,
            (OBSERVATION_CURATION_DRAIN_TASK,),
        )
        repositories.commit()
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert set(runtime.observation_curation_calls[:10]) == set(save_ids[:10])
    assert save_ids[10] in runtime.observation_curation_calls[10:]


def test_scheduler_drains_inactive_context_retry_while_active_save_job_runs(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    active_save_id = _save(repositories, title="Active Save")
    inactive_save_id = _save(repositories, title="Inactive Save")
    repositories.create_job(
        save_id=inactive_save_id,
        type="context_update_retry",
        status="queued",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    runtime = _ReviewRuntime(active_save_id=active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        release = asyncio.Event()

        async def blocking_worker(_handle: object) -> object:
            await release.wait()
            return {}

        await state.jobs.create(
            "chat_turn",
            blocking_worker,
            save_id=active_save_id,
            exclusive_key=f"chat_turn:{active_save_id}",
        )
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        try:
            await scheduler.run_once()
            await _wait_for_jobs_to_finish(state.jobs, save_id=inactive_save_id)
        finally:
            release.set()
            await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.context_retry_calls == [inactive_save_id]


def test_scheduler_drains_character_text_world_update_retry_for_active_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    retry_job = repositories.create_job(
        save_id=save_id,
        type="character_text_world_update_retry",
        status="queued",
        payload={"text_message_ids": ["text-message-1"]},
    )
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.character_text_world_update_retry_calls == [save_id]
    task = repositories.get_scheduled_task(
        task_type=CHARACTER_TEXT_WORLD_UPDATE_RETRY_DRAIN_TASK,
        save_id=save_id,
    )
    assert task is not None
    assert task.failure_count == 0
    assert retry_job.id in {
        job.id for job in repositories.list_jobs_by_status(("queued",))
    }


def test_scheduler_drains_character_text_world_update_retry_for_inactive_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    active_save_id = _save(repositories, title="Active Save")
    inactive_save_id = _save(repositories, title="Inactive Save")
    retry_job = repositories.create_job(
        save_id=inactive_save_id,
        type="character_text_world_update_retry",
        status="queued",
        payload={"text_message_ids": ["text-message-1"]},
    )
    runtime = _ReviewRuntime(active_save_id=active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.character_text_world_update_retry_calls == [inactive_save_id]
    task = repositories.get_scheduled_task(
        task_type=CHARACTER_TEXT_WORLD_UPDATE_RETRY_DRAIN_TASK,
        save_id=inactive_save_id,
    )
    assert task is not None
    assert task.failure_count == 0
    assert retry_job.id in {
        job.id for job in repositories.list_jobs_by_status(("queued",))
    }


def test_scheduler_skips_character_text_world_update_retry_with_active_save_job(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    repositories.create_job(
        save_id=save_id,
        type="character_text_world_update_retry",
        status="queued",
        payload={"text_message_ids": ["text-message-1"]},
    )
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        release = asyncio.Event()

        async def blocking_worker(_handle: object) -> object:
            await release.wait()
            return {}

        await state.jobs.create(
            "chat_turn",
            blocking_worker,
            save_id=save_id,
            exclusive_key=f"chat_turn:{save_id}",
        )
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        try:
            await scheduler.run_once()
        finally:
            release.set()
            await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.character_text_world_update_retry_calls == []
    assert (
        repositories.get_scheduled_task(
            task_type=CHARACTER_TEXT_WORLD_UPDATE_RETRY_DRAIN_TASK,
            save_id=save_id,
        )
        is None
    )


def test_scheduler_queues_low_priority_maintenance_for_active_save(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    repositories.set_model_preference(
        task="character_registry_maintenance",
        provider="fake",
        model_id="fake-characters",
    )
    _append_completed_turns(
        repositories,
        save_id=save_id,
        count=CHARACTER_MAINTENANCE_TURN_CADENCE,
    )
    repositories.add_character(save_id=save_id, name="Captain Ilyra")
    for index in range(MEMORY_CONSOLIDATION_THRESHOLD):
        repositories.add_memory(
            save_id=save_id,
            body=f"Memory {index}",
            tags=["test"],
        )
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        for _ in range(3):
            await scheduler.run_once()
            await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.memory_consolidation_calls == [save_id]
    assert runtime.character_maintenance_calls == [save_id]
    assert runtime.world_context_retention_calls == []
    for task_type in (
        MEMORY_CONSOLIDATION_TASK,
        CHARACTER_REGISTRY_MAINTENANCE_TASK,
        WORLD_CONTEXT_RETENTION_TASK,
    ):
        task = repositories.get_scheduled_task(task_type=task_type, save_id=save_id)
        assert task is not None
        assert task.failure_count == 0


def test_scheduler_uses_distinct_web_job_type_for_domain_character_maintenance(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    repositories.set_model_preference(
        task="character_registry_maintenance",
        provider="fake",
        model_id="fake-characters",
    )
    _append_completed_turns(
        repositories,
        save_id=save_id,
        count=CHARACTER_MAINTENANCE_TURN_CADENCE,
    )
    repositories.add_character(save_id=save_id, name="Captain Ilyra")
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        for _ in range(2):
            await scheduler.run_once()
            await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.character_maintenance_calls == [save_id]
    assert (
        repositories.list_recent_jobs(
            save_id=save_id,
            types=(CHARACTER_REGISTRY_MAINTENANCE_TASK,),
        )
        == []
    )
    wrapper_jobs = repositories.list_recent_jobs(
        save_id=save_id,
        types=(WEB_MAINTENANCE_CHARACTER_REGISTRY_MAINTENANCE_JOB,),
    )
    assert [job.status for job in wrapper_jobs] == ["succeeded"]


def test_scheduler_defers_memory_and_character_work_behind_context_retries(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    repositories.set_model_preference(
        task="character_registry_maintenance",
        provider="fake",
        model_id="fake-characters",
    )
    _append_completed_turns(
        repositories,
        save_id=save_id,
        count=CHARACTER_MAINTENANCE_TURN_CADENCE,
    )
    repositories.add_character(save_id=save_id, name="Captain Ilyra")
    for index in range(MEMORY_CONSOLIDATION_THRESHOLD):
        repositories.add_memory(
            save_id=save_id,
            body=f"Memory {index}",
            tags=["test"],
        )
    repositories.create_job(
        save_id=save_id,
        type="context_update_retry",
        status="queued",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.context_retry_calls == [save_id]
    assert runtime.memory_consolidation_calls == []
    assert runtime.character_maintenance_calls == []


def test_scheduler_defers_due_dormant_memory_work_behind_context_retry(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    process_active_save_id = _save(repositories, title="Process Active Save")
    dormant_save_id = _save(repositories, title="Dormant Save")
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    for index in range(MEMORY_CONSOLIDATION_THRESHOLD):
        repositories.add_memory(
            save_id=dormant_save_id,
            body=f"Memory {index}",
            tags=["test"],
        )
    repositories.create_job(
        save_id=dormant_save_id,
        type="context_update_retry",
        status="running",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    repositories.upsert_scheduled_task(
        task_type=MEMORY_CONSOLIDATION_TASK,
        save_id=dormant_save_id,
        interval_seconds=60,
        payload={"active_save_only": True},
        due_now=True,
    )
    runtime = _ReviewRuntime(active_save_id=process_active_save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.memory_consolidation_calls == []
    task = repositories.get_scheduled_task(
        task_type=MEMORY_CONSOLIDATION_TASK,
        save_id=dormant_save_id,
    )
    assert task is not None
    assert task.last_job_id is None


def test_scheduler_releases_context_retry_drain_lease_when_cancelled(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    repositories.create_job(
        save_id=save_id,
        type="context_update_retry",
        status="queued",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    runtime = _BlockingContextRetryRuntime(active_save_id=save_id)
    state = _scheduler_state(repositories, runtime)

    async def run() -> str:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        await asyncio.wait_for(runtime.started.wait(), timeout=1.0)
        job = next(
            job
            for job in state.jobs.list_active(save_id=save_id)
            if job.type == CONTEXT_UPDATE_RETRY_DRAIN_TASK
        )
        assert await state.jobs.cancel(job.id) is True
        if job.task is not None:
            await asyncio.wait_for(job.task, timeout=1.0)
        await _wait_for_jobs_to_finish(state.jobs)
        return str(job.id)

    job_id = asyncio.run(run())

    assert runtime.cancelled.is_set()
    task = repositories.get_scheduled_task(
        task_type=CONTEXT_UPDATE_RETRY_DRAIN_TASK,
        save_id=save_id,
    )
    assert task is not None
    assert task.lease_until is None
    assert task.failure_count == 0
    assert task.last_job_id == job_id
    assert task.result == {"status": "cancelled"}
    assert task.error is None
    persisted_job = repositories.connection.execute(
        "SELECT status, result_json, error FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert persisted_job is not None
    assert persisted_job["status"] == "cancelled"
    assert json.loads(persisted_job["result_json"]) == {"result_type": "cancelled"}
    assert persisted_job["error"] == "Cancelled"


def test_scheduler_policy_checks_use_aggregate_repository_methods(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    repositories.set_model_preference(
        task="character_registry_maintenance",
        provider="fake",
        model_id="fake-characters",
    )
    _append_completed_turns(
        repositories,
        save_id=save_id,
        count=CHARACTER_MAINTENANCE_TURN_CADENCE,
    )
    repositories.add_character(save_id=save_id, name="Captain Ilyra")
    for index in range(MEMORY_CONSOLIDATION_THRESHOLD):
        repositories.add_memory(
            save_id=save_id,
            body=f"Memory {index}",
            tags=["test"],
        )
    aggregate_only = _AggregateOnlySchedulerRepositories(repositories)
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(aggregate_only, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        for _ in range(3):
            await scheduler.run_once()
            await _wait_for_jobs_to_finish(state.jobs)

    asyncio.run(run())

    assert runtime.memory_consolidation_calls == [save_id]
    assert runtime.character_maintenance_calls == [save_id]
    assert aggregate_only.blocked_full_scan_calls == []


def test_scheduler_caches_not_due_slow_maintenance_policy_checks(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    save_id = _save(repositories, title="Night Watch")
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    repositories.set_model_preference(
        task="character_registry_maintenance",
        provider="fake",
        model_id="fake-characters",
    )
    repositories.add_character(save_id=save_id, name="Captain Ilyra")
    _append_completed_turns(repositories, save_id=save_id, count=1)
    counting = _CountingSchedulerRepositories(repositories)
    runtime = _ReviewRuntime(active_save_id=save_id)
    state = _scheduler_state(counting, runtime)

    async def run() -> None:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        await scheduler.run_once()
        first_counts = counting.policy_probe_counts.copy()
        await scheduler.run_once()
        await _wait_for_jobs_to_finish(state.jobs)
        assert counting.policy_probe_counts == first_counts

    asyncio.run(run())

    assert runtime.memory_consolidation_calls == []
    assert runtime.character_maintenance_calls == []
    assert (
        repositories.get_scheduled_task(
            task_type=MEMORY_CONSOLIDATION_TASK,
            save_id=save_id,
        )
        is not None
    )
    assert (
        repositories.get_scheduled_task(
            task_type=CHARACTER_REGISTRY_MAINTENANCE_TASK,
            save_id=save_id,
        )
        is not None
    )


def test_scheduler_selection_phase_does_not_block_event_loop() -> None:
    repositories = _BlockingScopedSchedulerRepositories(delay_seconds=0.15)
    state = SimpleNamespace(
        runtime=SimpleNamespace(active_save_id=None),
        repositories=repositories,
        repository_scope=repositories.scope,
        jobs=JobRegistry(),
        save_events=SaveEventHub(),
    )

    async def run() -> float:
        scheduler = WebMaintenanceScheduler(
            state,
            poll_interval_seconds=999,
            startup_delay_seconds=0,
        )
        started = time.perf_counter()
        task = asyncio.create_task(scheduler.run_once())
        await asyncio.sleep(0.02)
        elapsed = time.perf_counter() - started
        await task
        return elapsed

    assert asyncio.run(run()) < 0.1


class _ReviewRuntime:
    def __init__(self, *, active_save_id: str) -> None:
        self.active_save_id = active_save_id
        self.calls: list[str] = []
        self.state_pruning_calls: list[str] = []
        self.state_retry_calls: list[str] = []
        self.context_retry_calls: list[str] = []
        self.observation_curation_calls: list[str] = []
        self.character_text_world_update_retry_calls: list[str] = []
        self.memory_consolidation_calls: list[str] = []
        self.character_maintenance_calls: list[str] = []
        self.world_context_retention_calls: list[str] = []

    async def run_world_suggestion_review(
        self,
        *,
        active_save_id: str,
        scheduled: bool = False,
    ) -> object:
        self.calls.append(active_save_id)
        return {"active_save_id": active_save_id, "status": "reviewed", "error": None}

    async def run_state_pruning(self, *, active_save_id: str) -> object:
        self.state_pruning_calls.append(active_save_id)
        return {"active_save_id": active_save_id, "status": "cleaned", "error": None}

    async def run_context_update_retries(self, *, active_save_id: str) -> object:
        self.context_retry_calls.append(active_save_id)
        return {"active_save_id": active_save_id, "completed": 0, "error": None}

    async def run_observation_curation(self, *, active_save_id: str) -> object:
        self.observation_curation_calls.append(active_save_id)
        return {"active_save_id": active_save_id, "completed": 0, "error": None}

    async def run_state_extraction_retries(self, *, active_save_id: str) -> object:
        self.state_retry_calls.append(active_save_id)
        return {"active_save_id": active_save_id, "completed": 0, "error": None}

    async def run_character_text_world_update_retries(
        self,
        *,
        active_save_id: str,
    ) -> object:
        self.character_text_world_update_retry_calls.append(active_save_id)
        return {"active_save_id": active_save_id, "completed": 0, "error": None}

    async def run_memory_consolidation(self, *, active_save_id: str) -> object:
        self.memory_consolidation_calls.append(active_save_id)
        return {
            "active_save_id": active_save_id,
            "status": "consolidated",
            "error": None,
        }

    async def run_character_registry_maintenance(
        self,
        *,
        active_save_id: str,
    ) -> object:
        self.character_maintenance_calls.append(active_save_id)
        return {"active_save_id": active_save_id, "status": "maintained", "error": None}

    async def run_world_context_retention(self, *, active_save_id: str) -> object:
        self.world_context_retention_calls.append(active_save_id)
        return {"active_save_id": active_save_id, "status": "retained", "error": None}


class _VerboseResultRuntime(_ReviewRuntime):
    async def run_world_suggestion_review(
        self,
        *,
        active_save_id: str,
        scheduled: bool = False,
    ) -> object:
        self.calls.append(active_save_id)
        return {
            "active_save_id": active_save_id,
            "chronicle": {"messages": ["secret chronicle text"]},
            "custom_instructions": "secret instructions",
            "error": None,
            "media": {"assets": ["secret media"]},
            "saves": [{"save_id": active_save_id, "title": "Night Watch"}],
            "status": "reviewed",
        }


class _ErrorResultRuntime(_ReviewRuntime):
    async def run_world_suggestion_review(
        self,
        *,
        active_save_id: str,
        scheduled: bool = False,
    ) -> object:
        self.calls.append(active_save_id)
        return {
            "active_save_id": active_save_id,
            "error": "review exploded",
            "status": None,
        }


class _CurationErrorRuntime(_ReviewRuntime):
    async def run_observation_curation(self, *, active_save_id: str) -> object:
        self.observation_curation_calls.append(active_save_id)
        return {
            "active_save_id": active_save_id,
            "error": "Private chronicle detail escaped from a provider.",
            "status": None,
        }


class _BlockingContextRetryRuntime(_ReviewRuntime):
    def __init__(self, *, active_save_id: str) -> None:
        super().__init__(active_save_id=active_save_id)
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run_context_update_retries(self, *, active_save_id: str) -> object:
        self.context_retry_calls.append(active_save_id)
        self.started.set()
        try:
            await asyncio.Future()
            raise AssertionError("blocking context retry runtime resumed")
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def _repositories(tmp_path: Path) -> PersistenceRepositories:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    connection = sqlite3.connect(database_path)
    return PersistenceRepositories(connection)


def _scheduler_state(
    repositories: object,
    runtime: _ReviewRuntime,
) -> SimpleNamespace:
    return SimpleNamespace(
        runtime=runtime,
        repositories=repositories,
        jobs=JobRegistry(repositories=repositories),
        save_events=SaveEventHub(),
    )


async def _wait_for_jobs_to_finish(
    jobs: JobRegistry,
    *,
    save_id: str | None = None,
) -> None:
    for _ in range(50):
        if not jobs.list_active(save_id=save_id):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("scheduler job did not finish")


class _DismissSuggestionAfterLeaseRepositories:
    def __init__(
        self,
        repositories: PersistenceRepositories,
        *,
        suggestion_id: str,
    ) -> None:
        self._repositories = repositories
        self._suggestion_id = suggestion_id

    def __getattr__(self, name: str) -> object:
        return getattr(self._repositories, name)

    def lease_scheduled_task(
        self,
        task_id: str,
        *,
        lease_seconds: int = 300,
    ) -> object:
        leased = self._repositories.lease_scheduled_task(
            task_id,
            lease_seconds=lease_seconds,
        )
        if leased is not None:
            self._repositories.update_context_update_suggestion_status(
                self._suggestion_id,
                status="dismissed",
            )
        return leased


class _AggregateOnlySchedulerRepositories:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self._repositories = repositories
        self.blocked_full_scan_calls: list[str] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self._repositories, name)

    def list_messages(self, save_id: str, *, include_deleted: bool = False) -> object:
        self.blocked_full_scan_calls.append("list_messages")
        raise AssertionError("scheduler policy should use message aggregates")

    def list_memories(self, save_id: str) -> object:
        self.blocked_full_scan_calls.append("list_memories")
        raise AssertionError("scheduler policy should use memory aggregates")

    def list_characters(self, save_id: str) -> object:
        self.blocked_full_scan_calls.append("list_characters")
        raise AssertionError("scheduler policy should use character aggregates")

    def list_jobs_by_status(self, statuses: tuple[str, ...]) -> object:
        self.blocked_full_scan_calls.append("list_jobs_by_status")
        raise AssertionError("scheduler policy should use targeted job queries")


class _CountingSchedulerRepositories(_AggregateOnlySchedulerRepositories):
    def __init__(self, repositories: PersistenceRepositories) -> None:
        super().__init__(repositories)
        self.policy_probe_counts = {
            "messages": 0,
            "memories": 0,
            "characters": 0,
        }

    def count_active_messages_by_role(
        self,
        save_id: str,
        *,
        roles: tuple[str, ...],
        created_at_lte: str | None = None,
    ) -> dict[str, int]:
        self.policy_probe_counts["messages"] += 1
        return self._repositories.count_active_messages_by_role(
            save_id,
            roles=roles,
            created_at_lte=created_at_lte,
        )

    def count_active_memories(self, save_id: str) -> int:
        self.policy_probe_counts["memories"] += 1
        return self._repositories.count_active_memories(save_id)

    def has_unprotected_character(self, save_id: str) -> bool:
        self.policy_probe_counts["characters"] += 1
        return self._repositories.has_unprotected_character(save_id)


class _BlockingScopedSchedulerRepositories:
    def __init__(self, *, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds

    def scope(self) -> object:
        return self

    def __enter__(self) -> _BlockingScopedSchedulerRepositories:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def list_user_active_save_ids(self) -> tuple[str, ...]:
        time.sleep(self._delay_seconds)
        return ()

    def list_job_save_ids(
        self,
        *,
        statuses: tuple[str, ...],
        types: tuple[str, ...],
    ) -> tuple[str, ...]:
        return ()

    def list_due_scheduled_tasks(
        self,
        *,
        task_types: tuple[str, ...] = (),
        save_id: str | None | object = ...,
        limit: int = 10,
    ) -> list[object]:
        return []


def _save_with_pending_suggestion(
    repositories: PersistenceRepositories,
    *,
    title: str = "Night Watch",
    configure_context_update_model: bool = True,
    owner_user_id: str | None = None,
) -> str:
    save_id = _save(repositories, title=title, owner_user_id=owner_user_id)
    if configure_context_update_model:
        repositories.set_model_preference(
            task="context_update",
            provider="fake",
            model_id="fake-context",
        )
    message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens turns red.",
    )
    location = repositories.add_location(
        save_id=save_id,
        name="Beacon Gallery",
        description="The beacon lens is clear.",
        source_message_id=message.id,
    )
    repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="location",
        entity_id=location.id,
        field_path="description",
        proposed_value="The beacon lens turns red.",
        reason="Narration says the lens turns red.",
        confidence=0.9,
        source_message_ids=[message.id],
    )
    return save_id


def _append_narrator_messages(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    count: int,
) -> None:
    for index in range(count):
        repositories.append_message(
            save_id=save_id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Narrator turn {index}",
        )


def _append_completed_turns(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    count: int,
) -> None:
    for index in range(count):
        repositories.append_message(
            save_id=save_id,
            role="player",
            speaker_name="Mara",
            body=f"Player turn {index}",
        )
        repositories.append_message(
            save_id=save_id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Narrator turn {index}",
        )


def _save(
    repositories: PersistenceRepositories,
    *,
    title: str,
    owner_user_id: str | None = None,
) -> str:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title=title,
        owner_user_id=owner_user_id,
    )
    return save.id
