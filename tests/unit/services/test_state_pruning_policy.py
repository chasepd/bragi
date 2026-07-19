from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.state_pruning_policy import state_pruning_schedule_decision


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_state_pruning_policy_schedules_at_five_completed_narrator_turns(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _save_with_pruning_preference(repositories)
    _append_narrator_turns(repositories, save_id, 5)

    decision = state_pruning_schedule_decision(
        repositories=repositories,
        save_id=save_id,
    )

    assert decision.due is True
    assert decision.narrator_turn_count == 5
    assert decision.reason == "first_interval_reached"


def test_state_pruning_policy_skips_before_interval(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _save_with_pruning_preference(repositories)
    _append_narrator_turns(repositories, save_id, 4)

    decision = state_pruning_schedule_decision(
        repositories=repositories,
        save_id=save_id,
    )

    assert decision.due is False
    assert decision.reason == "below_interval"


def test_state_pruning_policy_skips_until_five_turns_after_last_pruning(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _save_with_pruning_preference(repositories)
    _append_narrator_turns(repositories, save_id, 5)
    jobs = JobLifecycleService(repositories=repositories)
    job = jobs.create_running(
        save_id=save_id,
        type="state_pruning",
        payload={"narrator_turn_count": 5},
    )
    jobs.succeed(job.id, result={"narrator_turn_count": 5})
    _append_narrator_turns(repositories, save_id, 4)

    decision = state_pruning_schedule_decision(
        repositories=repositories,
        save_id=save_id,
    )

    assert decision.due is False
    assert decision.narrator_turn_count == 9
    assert decision.last_pruning_turn_count == 5
    assert decision.reason == "interval_not_reached"


def test_state_pruning_policy_schedules_five_turns_after_last_pruning(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _save_with_pruning_preference(repositories)
    _append_narrator_turns(repositories, save_id, 5)
    jobs = JobLifecycleService(repositories=repositories)
    job = jobs.create_running(
        save_id=save_id,
        type="state_pruning",
        payload={"narrator_turn_count": 5},
    )
    jobs.succeed(job.id, result={"narrator_turn_count": 5})
    _append_narrator_turns(repositories, save_id, 5)

    decision = state_pruning_schedule_decision(
        repositories=repositories,
        save_id=save_id,
    )

    assert decision.due is True
    assert decision.narrator_turn_count == 10
    assert decision.last_pruning_turn_count == 5
    assert decision.reason == "interval_reached"


def test_state_pruning_policy_skips_without_model_preference(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    _append_narrator_turns(repositories, save.id, 5)

    decision = state_pruning_schedule_decision(
        repositories=repositories,
        save_id=save.id,
    )

    assert decision.due is False
    assert decision.reason == "missing_model_preference"


def _save_with_pruning_preference(repositories: PersistenceRepositories) -> str:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="state_pruning",
        provider="fake",
        model_id="fake-pruner",
    )
    return save.id


def _append_narrator_turns(
    repositories: PersistenceRepositories,
    save_id: str,
    count: int,
) -> None:
    for index in range(count):
        repositories.append_message(
            save_id=save_id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Narrator turn {index}",
            provider="fake",
            model="fake-chat",
        )
