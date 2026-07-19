from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.open_threads import (
    archive_open_thread_aggregate_state,
    has_active_thread_records,
    is_open_threads_aggregate_key,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


@pytest.mark.parametrize(
    "key",
    [
        "open_threads",
        " OPEN_THREADS ",
        "interaction.open_threads",
        "interactions.open_threads",
        " Interactions.Open_Threads ",
    ],
)
def test_is_open_threads_aggregate_key_accepts_legacy_variants(key: str) -> None:
    assert is_open_threads_aggregate_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "active_threads",
        "interaction.open_thread",
        "scene.open_threads",
        "open threads",
        "",
    ],
)
def test_is_open_threads_aggregate_key_rejects_non_aggregate_keys(key: str) -> None:
    assert not is_open_threads_aggregate_key(key)


def test_has_active_thread_records_checks_active_thread_rows(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories, title="Open threads")

    assert not has_active_thread_records(repositories, save.id)

    repositories.add_active_thread(
        save_id=save.id,
        title="Follow the bell",
        thread_id="thread-bell",
    )

    assert has_active_thread_records(repositories, save.id)


def test_archive_open_thread_aggregate_state_archives_only_matching_keys(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories, title="Open thread aggregates")
    repositories.upsert_world_state(
        save_id=save.id,
        key="open_threads",
        value={"items": ["legacy"]},
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="interaction.open_threads",
        value={"items": ["interaction"]},
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.open_threads",
        value={"items": ["not aggregate"]},
    )

    archived = archive_open_thread_aggregate_state(repositories, save.id)

    assert archived == 2
    assert [
        (state.key, state.value)
        for state in repositories.list_world_state(save.id)
    ] == [("scene.open_threads", {"items": ["not aggregate"]})]


def _create_save(
    repositories: PersistenceRepositories,
    *,
    title: str,
) -> SaveRecord:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Open Threads Scenario",
        premise="A small test scenario.",
        player_role="Tester",
        content={"starting_scene": "The bell rings."},
    )
    return repositories.create_save(scenario_id=scenario.id, title=title)
