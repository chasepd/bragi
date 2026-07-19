from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.active_thread_lifecycle import (
    active_thread_is_prompt_visible,
    active_thread_is_scene_local,
    active_thread_status_is_open,
    archive_inactive_active_threads,
    normalize_active_thread_record,
    normalize_active_thread_status,
    normalize_active_thread_visibility,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


@pytest.mark.parametrize(
    ("raw_status", "normalized", "is_open"),
    [
        ("", "active", True),
        ("open", "active", True),
        ("in-progress", "active", True),
        ("waiting", "blocked", True),
        ("paused", "deferred", True),
        ("done", "resolved", False),
        ("closed", "resolved", False),
        ("cancelled", "abandoned", False),
        ("superseded", "abandoned", False),
        ("some new label", "active", True),
        (None, "active", True),
    ],
)
def test_active_thread_status_aliases_normalize_to_lifecycle_states(
    raw_status: object,
    normalized: str,
    is_open: bool,
) -> None:
    assert normalize_active_thread_status(raw_status) == normalized
    assert active_thread_status_is_open(raw_status) is is_open


@pytest.mark.parametrize(
    ("raw_visibility", "normalized", "is_scene_local"),
    [
        ("", "public", False),
        ("neither", "public", False),
        ("private-between-characters", "private", False),
        ("gm only", "hidden", False),
        ("secret from the table", "hidden", False),
        ("current scene", "scene", True),
        ("scene-local", "scene", True),
        ("local only", "scene", True),
        ("some public note", "public", False),
        (None, "public", False),
    ],
)
def test_active_thread_visibility_aliases_normalize_prompt_scope(
    raw_visibility: object,
    normalized: str,
    is_scene_local: bool,
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories, title="Thread visibility")
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Follow the bell",
        visibility=str(raw_visibility or ""),
    )

    assert normalize_active_thread_visibility(raw_visibility) == normalized
    assert active_thread_is_scene_local(thread) is is_scene_local


@pytest.mark.parametrize(
    ("status", "visibility", "visible"),
    [
        ("open", "public", True),
        ("blocked", "private", True),
        ("deferred", "scene local", True),
        ("active", "hidden", False),
        ("resolved", "public", False),
        ("abandoned", "scene", False),
    ],
)
def test_active_thread_prompt_visibility_combines_status_and_visibility(
    repositories: PersistenceRepositories,
    status: str,
    visibility: str,
    visible: bool,
) -> None:
    save = _create_save(repositories, title="Prompt visibility")
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Follow the bell",
        status=status,
        visibility=visibility,
    )

    assert active_thread_is_prompt_visible(thread) is visible


def test_normalize_active_thread_record_returns_canonical_copy(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories, title="Normalize active thread")
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Follow the bell",
        status="in-progress",
        visibility="gm only",
    )

    normalized = normalize_active_thread_record(thread)

    assert normalized.id == thread.id
    assert normalized.status == "active"
    assert normalized.visibility == "hidden"


def test_archive_inactive_active_threads_normalizes_counts_and_archives(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories, title="Archive inactive threads")
    repositories.add_active_thread(
        save_id=save.id,
        title="Keep watching",
        status="waiting",
        visibility="current scene",
        thread_id="thread-open",
    )
    repositories.add_active_thread(
        save_id=save.id,
        title="Resolved lead",
        status="done",
        visibility="gm only",
        thread_id="thread-resolved",
    )
    repositories.add_active_thread(
        save_id=save.id,
        title="Dropped lead",
        status="stale",
        visibility="private-between-characters",
        thread_id="thread-abandoned",
    )

    archived = archive_inactive_active_threads(repositories, save.id)

    assert archived == 2
    active_threads = repositories.list_active_threads(save.id)
    assert [
        (thread.id, thread.status, thread.visibility)
        for thread in active_threads
    ] == [("thread-open", "blocked", "scene")]
    assert repositories.get_active_thread("thread-resolved") is None
    assert repositories.get_active_thread("thread-abandoned") is None


def _create_save(
    repositories: PersistenceRepositories,
    *,
    title: str,
) -> SaveRecord:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Thread Scenario",
        premise="A small test scenario.",
        player_role="Tester",
        content={"starting_scene": "The bell rings."},
    )
    return repositories.create_save(scenario_id=scenario.id, title=title)
