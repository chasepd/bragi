from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi_web.jobs import (
    CONTINUITY_READY,
    OPTIONAL_ENRICHMENTS_COMPLETE,
    RESPONSE_COMMITTED,
    JobHandle,
    JobRegistry,
    JobRegistryFullError,
    JobRegistryLimits,
    job_summary,
)


async def _ok_worker(handle: JobHandle) -> dict[str, bool]:
    return {"ok": True}


SAFE_JOB_ERROR = "Background job failed. Check diagnostics for details."
SAFE_RATE_LIMIT_ERROR = (
    "Provider request failed (rate_limited, 429). Check diagnostics for details."
)


def test_completion_level_wait_releases_before_optional_work_finishes() -> None:
    async def run_test() -> None:
        continuity_started = asyncio.Event()
        release_continuity = asyncio.Event()
        release_optional = asyncio.Event()
        registry = JobRegistry()

        async def worker(handle: JobHandle) -> dict[str, bool]:
            await handle.advance_completion_level(RESPONSE_COMMITTED)
            continuity_started.set()
            await release_continuity.wait()
            await handle.advance_completion_level(CONTINUITY_READY)
            await release_optional.wait()
            await handle.advance_completion_level(OPTIONAL_ENRICHMENTS_COMPLETE)
            return {"complete": True}

        record = await registry.create("post_turn_background", worker)
        await continuity_started.wait()
        wait_task = asyncio.create_task(
            registry.wait_for_completion_level(record.id, CONTINUITY_READY)
        )
        await asyncio.sleep(0)
        assert not wait_task.done()

        release_continuity.set()
        reached = await asyncio.wait_for(wait_task, timeout=1.0)
        assert reached is not None
        assert reached.completion_level == CONTINUITY_READY
        assert reached.status == "running"
        assert job_summary(reached)["completion_level"] == CONTINUITY_READY

        release_optional.set()
        assert record.task is not None
        await record.task
        finished = registry.get(record.id)
        assert finished is not None
        assert finished.completion_level == OPTIONAL_ENRICHMENTS_COMPLETE

    asyncio.run(run_test())


def test_completion_level_wait_releases_when_job_fails_before_barrier() -> None:
    async def run_test() -> None:
        registry = JobRegistry()

        async def worker(handle: JobHandle) -> None:
            await handle.advance_completion_level(RESPONSE_COMMITTED)
            raise RuntimeError("continuity failed")

        record = await registry.create("post_turn_background", worker)
        reached = await asyncio.wait_for(
            registry.wait_for_completion_level(record.id, CONTINUITY_READY),
            timeout=1.0,
        )
        assert reached is not None
        assert reached.status == "failed"
        assert reached.completion_level == RESPONSE_COMMITTED

    asyncio.run(run_test())


def test_active_job_cap_rejects_new_jobs_until_one_finishes() -> None:
    async def run_test() -> None:
        blocker = asyncio.Event()
        registry = JobRegistry(JobRegistryLimits(max_active_jobs=1))

        async def blocked_worker(handle: JobHandle) -> dict[str, bool]:
            await blocker.wait()
            return {"released": True}

        first = await registry.create("blocked", blocked_worker)
        with pytest.raises(JobRegistryFullError):
            await registry.create("blocked", blocked_worker)

        blocker.set()
        assert first.task is not None
        await first.task

        second = await registry.create("blocked", _ok_worker)
        assert second.task is not None
        await second.task
        assert registry.get(second.id) is not None

    asyncio.run(run_test())


def test_active_exclusive_job_rejects_same_key_until_terminal() -> None:
    async def run_test() -> None:
        blocker = asyncio.Event()
        registry = JobRegistry()

        async def blocked_worker(handle: JobHandle) -> dict[str, bool]:
            await blocker.wait()
            return {"released": True}

        first = await registry.create(
            "chat_turn",
            blocked_worker,
            save_id="save-1",
            exclusive_key="chat_turn:save-1",
        )
        same_save = await registry.create(
            "image_generation",
            _ok_worker,
            save_id="save-1",
            exclusive_key="image_generation:save-1",
        )
        other_save = await registry.create(
            "chat_turn",
            _ok_worker,
            save_id="save-2",
            exclusive_key="chat_turn:save-2",
        )

        with pytest.raises(JobRegistryFullError):
            await registry.create(
                "chat_turn",
                _ok_worker,
                save_id="save-1",
                exclusive_key="chat_turn:save-1",
            )

        assert same_save.task is not None
        assert other_save.task is not None
        await same_save.task
        await other_save.task
        blocker.set()
        assert first.task is not None
        await first.task

        retry = await registry.create(
            "chat_turn",
            _ok_worker,
            save_id="save-1",
            exclusive_key="chat_turn:save-1",
        )
        assert retry.task is not None
        await retry.task

    asyncio.run(run_test())


def test_operation_queue_runs_same_save_jobs_fifo() -> None:
    async def run_test() -> None:
        release_first = asyncio.Event()
        first_entered = asyncio.Event()
        calls: list[str] = []
        registry = JobRegistry()

        async def first_worker(handle: JobHandle) -> dict[str, str]:
            calls.append("first")
            first_entered.set()
            await release_first.wait()
            return {"job": "first"}

        async def queued_worker(handle: JobHandle) -> dict[str, str]:
            calls.append(handle.record.type)
            return {"job": handle.record.type}

        first = await registry.create(
            "first",
            first_worker,
            save_id="save-1",
            operation_queue_key="save-1",
        )
        await asyncio.wait_for(first_entered.wait(), timeout=1.0)

        second = await registry.create(
            "second",
            queued_worker,
            save_id="save-1",
            operation_queue_key="save-1",
        )
        third = await registry.create(
            "third",
            queued_worker,
            save_id="save-1",
            operation_queue_key="save-1",
        )
        await asyncio.sleep(0)

        second_queued = registry.get(second.id)
        third_queued = registry.get(third.id)
        assert second_queued is not None
        assert third_queued is not None
        assert second_queued.status == "queued"
        assert third_queued.status == "queued"
        assert calls == ["first"]

        release_first.set()
        assert first.task is not None
        assert second.task is not None
        assert third.task is not None
        await asyncio.gather(first.task, second.task, third.task)

        assert calls == ["first", "second", "third"]
        second_done = registry.get(second.id)
        third_done = registry.get(third.id)
        assert second_done is not None
        assert third_done is not None
        assert second_done.status == "succeeded"
        assert third_done.status == "succeeded"

    asyncio.run(run_test())


def test_operation_queue_does_not_block_other_save_jobs() -> None:
    async def run_test() -> None:
        release_first = asyncio.Event()
        first_entered = asyncio.Event()
        other_entered = asyncio.Event()
        calls: list[str] = []
        registry = JobRegistry()

        async def first_worker(handle: JobHandle) -> dict[str, str]:
            calls.append("first")
            first_entered.set()
            await release_first.wait()
            return {"job": "first"}

        async def other_worker(handle: JobHandle) -> dict[str, str]:
            calls.append("other")
            other_entered.set()
            return {"job": "other"}

        first = await registry.create(
            "first",
            first_worker,
            save_id="save-1",
            operation_queue_key="save-1",
        )
        await asyncio.wait_for(first_entered.wait(), timeout=1.0)

        other = await registry.create(
            "other",
            other_worker,
            save_id="save-2",
            operation_queue_key="save-2",
        )
        await asyncio.wait_for(other_entered.wait(), timeout=1.0)

        assert calls == ["first", "other"]
        other_done = registry.get(other.id)
        assert other_done is not None
        assert other_done.status == "succeeded"

        release_first.set()
        assert first.task is not None
        assert other.task is not None
        await asyncio.gather(first.task, other.task)

    asyncio.run(run_test())


def test_queued_operation_can_be_cancelled_before_it_starts() -> None:
    async def run_test() -> None:
        release_first = asyncio.Event()
        first_entered = asyncio.Event()
        queued_started = False
        registry = JobRegistry()

        async def first_worker(handle: JobHandle) -> dict[str, str]:
            first_entered.set()
            await release_first.wait()
            return {"job": "first"}

        async def queued_worker(handle: JobHandle) -> dict[str, str]:
            nonlocal queued_started
            queued_started = True
            return {"job": "queued"}

        first = await registry.create(
            "first",
            first_worker,
            save_id="save-1",
            operation_queue_key="save-1",
        )
        await asyncio.wait_for(first_entered.wait(), timeout=1.0)
        queued = await registry.create(
            "queued",
            queued_worker,
            save_id="save-1",
            operation_queue_key="save-1",
        )

        assert await registry.cancel(queued.id) is True
        assert await registry.cancel(queued.id) is False
        assert queued.task is not None
        await queued.task
        release_first.set()
        assert first.task is not None
        await first.task

        snapshot = registry.get(queued.id)
        assert snapshot is not None
        assert snapshot.status == "cancelled"
        assert queued_started is False

    asyncio.run(run_test())


def test_operation_queue_counts_queued_jobs_toward_active_cap() -> None:
    async def run_test() -> None:
        release_first = asyncio.Event()
        first_entered = asyncio.Event()
        registry = JobRegistry(JobRegistryLimits(max_active_jobs=2))

        async def first_worker(handle: JobHandle) -> dict[str, str]:
            first_entered.set()
            await release_first.wait()
            return {"job": "first"}

        await registry.create(
            "first",
            first_worker,
            save_id="save-1",
            operation_queue_key="save-1",
        )
        await asyncio.wait_for(first_entered.wait(), timeout=1.0)
        queued = await registry.create(
            "queued",
            _ok_worker,
            save_id="save-1",
            operation_queue_key="save-1",
        )

        with pytest.raises(JobRegistryFullError):
            await registry.create(
                "overflow",
                _ok_worker,
                save_id="save-2",
                operation_queue_key="save-2",
            )

        release_first.set()
        assert queued.task is not None
        active = registry.list_active()
        assert {record.id for record in active}
        await asyncio.gather(*(record.task for record in active if record.task))

    asyncio.run(run_test())


def test_completed_jobs_expire_on_registry_reads() -> None:
    async def run_test() -> None:
        registry = JobRegistry(JobRegistryLimits(completed_ttl_seconds=0))
        record = await registry.create("done", _ok_worker)
        assert record.task is not None
        await record.task

        assert registry.get(record.id) is None

    asyncio.run(run_test())


def test_completed_job_count_is_bounded_to_newest_records() -> None:
    async def run_test() -> None:
        registry = JobRegistry(
            JobRegistryLimits(completed_ttl_seconds=3600, max_completed_jobs=1)
        )
        first = await registry.create("done", _ok_worker)
        assert first.task is not None
        await first.task
        await asyncio.sleep(0.001)
        second = await registry.create("done", _ok_worker)
        assert second.task is not None
        await second.task

        assert registry.get(first.id) is None
        assert registry.get(second.id) is not None

    asyncio.run(run_test())


def test_job_event_wait_does_not_use_default_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked_to_thread(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("job event waits must not use the default executor")

    async def run_test() -> None:
        release = asyncio.Event()
        registry = JobRegistry()
        monkeypatch.setattr("bragi_web.jobs.asyncio.to_thread", blocked_to_thread)

        async def worker(_handle: JobHandle) -> dict[str, bool]:
            await release.wait()
            return {"ok": True}

        record = await registry.create("wait_probe", worker)
        waiter = asyncio.create_task(registry.wait_for_event(record.id, 0))
        await asyncio.sleep(0)
        await registry.add_event(record.id, "progress", {"ok": True})

        assert await asyncio.wait_for(waiter, timeout=1.0) == 0

        release.set()
        assert record.task is not None
        await asyncio.wait_for(record.task, timeout=1.0)

    asyncio.run(run_test())


def test_per_job_event_history_is_bounded_with_absolute_offset() -> None:
    async def run_test() -> None:
        registry = JobRegistry(JobRegistryLimits(max_events_per_job=2))

        async def noisy_worker(handle: JobHandle) -> dict[str, bool]:
            await handle.event("progress", {"label": "one"})
            await handle.event("progress", {"label": "two"})
            await handle.event("progress", {"label": "three"})
            return {"ok": True}

        record = await registry.create("noisy", noisy_worker)
        assert record.task is not None
        await record.task
        snapshot = registry.get(record.id)

        assert snapshot is not None
        assert snapshot.event_offset == 3
        assert [event["event"] for event in snapshot.events] == [
            "progress",
            "status",
        ]
        assert snapshot.events[0]["payload"] == {"label": "three"}
        assert snapshot.events[1]["payload"] == {"status": "succeeded"}

    asyncio.run(run_test())


def test_job_registry_persists_metadata_only_rows(tmp_path: Path) -> None:
    async def run_test() -> None:
        repositories = _repositories(tmp_path)
        user = repositories.create_user(
            username="Mira",
            role="user",
            password_hash="hash",
        )
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A border keep is cut off by ash storms.",
            player_role="Signal warden",
            content={"starting_scene": "The beacon gutters."},
        )
        save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
        registry = JobRegistry(repositories=repositories)
        release_worker = asyncio.Event()

        async def worker(handle: JobHandle) -> dict[str, object]:
            await handle.event("progress", {"label": "working"})
            await release_worker.wait()
            return {
                "chronicle": {"messages": [{"body": "secret chat content"}]},
                "count": 2,
            }

        record = await registry.create(
            "chat_turn",
            worker,
            save_id=save.id,
            exclusive_key=f"chat_turn:{save.id}",
            creator_user_id=user.id,
        )
        queued = repositories.list_jobs_by_status(("queued", "running"))[0]
        assert queued.id == record.id
        assert queued.creator_user_id == user.id
        assert queued.payload == {
            "source": "web",
            "exclusive_key": f"chat_turn:{save.id}",
        }
        assert "secret chat content" not in repr(queued)

        assert record.task is not None
        release_worker.set()
        await record.task

        completed = repositories.list_recent_jobs(
            types=("chat_turn",),
            statuses=("succeeded",),
            seconds=0,
            limit=1,
        )[0]
        assert completed.id == record.id
        assert completed.creator_user_id == user.id
        assert completed.duration_ms is not None
        assert completed.result == {
            "result_type": "object",
            "result_keys": ["chronicle", "count"],
            "count": 2,
        }
        assert completed.diagnostics is not None
        completed_bragi = cast(Mapping[str, object], completed.diagnostics["bragi"])
        completed_timing = cast(Mapping[str, object], completed.diagnostics["timing"])
        assert completed_bragi == {"status": "succeeded"}
        assert completed_timing["completed_at"] == completed.completed_at
        assert "secret chat content" not in repr(completed)

    asyncio.run(run_test())


def test_failed_web_job_error_is_public_in_memory_events_and_persistence(
    tmp_path: Path,
) -> None:
    async def run_test() -> None:
        repositories = _repositories(tmp_path)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A border keep is cut off by ash storms.",
            player_role="Signal warden",
            content={"starting_scene": "The beacon gutters."},
        )
        save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
        registry = JobRegistry(repositories=repositories)

        async def worker(handle: JobHandle) -> None:
            raise RuntimeError(
                "provider echoed api_key=live-secret while processing Mara's "
                "private scene prompt"
            )

        record = await registry.create("chat_turn", worker, save_id=save.id)
        assert record.task is not None
        await record.task
        snapshot = registry.get(record.id)
        failed = repositories.list_recent_jobs(
            types=("chat_turn",),
            statuses=("failed",),
            seconds=0,
            limit=1,
        )[0]

        assert snapshot is not None
        assert snapshot.status == "failed"
        assert snapshot.error == SAFE_JOB_ERROR
        assert snapshot.events[-1] == {
            "event": "error",
            "payload": {"error": SAFE_JOB_ERROR},
        }
        assert failed.error == SAFE_JOB_ERROR
        assert failed.diagnostics is not None
        failed_bragi = cast(Mapping[str, object], failed.diagnostics["bragi"])
        failed_timing = cast(Mapping[str, object], failed.diagnostics["timing"])
        assert failed_bragi == {
            "status": "failed",
            "error": SAFE_JOB_ERROR,
        }
        assert failed_timing["completed_at"] == failed.completed_at
        exposed_text = repr((snapshot.error, snapshot.events, failed.error))
        assert "live-secret" not in exposed_text
        assert "api_key" not in exposed_text
        assert "Mara" not in exposed_text
        assert "private scene prompt" not in exposed_text

    asyncio.run(run_test())


def test_provider_job_failure_uses_safe_category_summary() -> None:
    async def run_test() -> None:
        registry = JobRegistry()

        async def worker(handle: JobHandle) -> None:
            raise ProviderError(
                category=ProviderErrorCategory.RATE_LIMITED,
                status_code=429,
                message=(
                    "Venice echoed player prompt about Mara and "
                    "authorization=secret-token"
                ),
            )

        record = await registry.create("scenario_draft", worker)
        assert record.task is not None
        await record.task
        snapshot = registry.get(record.id)

        assert snapshot is not None
        assert snapshot.status == "failed"
        assert snapshot.error == SAFE_RATE_LIMIT_ERROR
        assert snapshot.events[-1] == {
            "event": "error",
            "payload": {"error": SAFE_RATE_LIMIT_ERROR},
        }
        exposed_text = repr((snapshot.error, snapshot.events))
        assert "Mara" not in exposed_text
        assert "secret-token" not in exposed_text
        assert "authorization" not in exposed_text

    asyncio.run(run_test())


def test_cancel_records_requested_event_before_terminal_status() -> None:
    async def run_test() -> None:
        registry = JobRegistry()

        async def blocked_worker(handle: JobHandle) -> None:
            await asyncio.sleep(60)

        record = await registry.create("cancel_me", blocked_worker)
        assert record.task is not None
        await asyncio.sleep(0)

        assert await registry.cancel(record.id) is True
        await record.task
        snapshot = registry.get(record.id)

        assert snapshot is not None
        assert [event["event"] for event in snapshot.events][-2:] == [
            "cancel_requested",
            "status",
        ]
        assert snapshot.events[-1]["payload"] == {"status": "cancelled"}

    asyncio.run(run_test())


def test_concurrent_create_list_and_cancel_uses_consistent_snapshots() -> None:
    async def run_test() -> None:
        registry = JobRegistry(JobRegistryLimits(max_active_jobs=50))

        async def blocked_worker(handle: JobHandle) -> None:
            await asyncio.sleep(60)

        records = await asyncio.gather(
            *(registry.create("slow", blocked_worker) for _ in range(20))
        )
        await asyncio.gather(
            *(asyncio.to_thread(registry.list_active) for _ in range(20)),
            *(registry.cancel(record.id) for record in records),
        )
        tasks = [record.task for record in records if record.task is not None]
        await asyncio.gather(*tasks)

        assert registry.list_active() == []
        snapshots = [registry.get(record.id) for record in records]
        assert all(snapshot is not None for snapshot in snapshots)
        assert all(snapshot.status == "cancelled" for snapshot in snapshots if snapshot)

    asyncio.run(run_test())


def _repositories(tmp_path: Path) -> PersistenceRepositories:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    connection = sqlite3.connect(database_path, check_same_thread=False)
    return PersistenceRepositories(connection)
