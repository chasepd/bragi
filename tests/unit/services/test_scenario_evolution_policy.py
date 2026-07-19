from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.scenario_evolution_policy import (
    DEFAULT_SCENARIO_EVOLUTION_TURN_INTERVAL,
    SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
    save_scenario_evolution_turn_interval_setting_key,
    scenario_evolution_turn_interval,
    scenario_template_evolution_turn_interval_setting_key,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_scenario_evolution_turn_interval_uses_save_scenario_global_precedence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A keep waits under an ash storm.",
        player_role="Signal warden",
        content={"current_scene": "The warden stands at the lower gate."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")

    assert (
        scenario_evolution_turn_interval(repositories, save_id=save.id)
        == DEFAULT_SCENARIO_EVOLUTION_TURN_INTERVAL
    )

    repositories.set_app_setting(SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING, 6)
    assert scenario_evolution_turn_interval(repositories, save_id=save.id) == 6

    repositories.set_app_setting(
        scenario_template_evolution_turn_interval_setting_key(scenario.id),
        4,
    )
    assert scenario_evolution_turn_interval(repositories, save_id=save.id) == 4

    repositories.set_app_setting(
        save_scenario_evolution_turn_interval_setting_key(save.id),
        2,
    )
    assert scenario_evolution_turn_interval(repositories, save_id=save.id) == 2


def test_scenario_evolution_turn_interval_sanitizes_invalid_values(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A keep waits under an ash storm.",
        player_role="Signal warden",
        content={"current_scene": "The warden stands at the lower gate."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")

    repositories.set_app_setting(SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING, "many")
    assert (
        scenario_evolution_turn_interval(repositories, save_id=save.id)
        == DEFAULT_SCENARIO_EVOLUTION_TURN_INTERVAL
    )

    repositories.set_app_setting(
        save_scenario_evolution_turn_interval_setting_key(save.id),
        -3,
    )
    assert scenario_evolution_turn_interval(repositories, save_id=save.id) == 0
