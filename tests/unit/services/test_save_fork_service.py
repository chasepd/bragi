from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.save_fork_service import SaveForkService
from bragi.services.turn_snapshot_service import TurnSnapshotService


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_fork_from_message_without_snapshot_fails_clearly(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A beacon tower.",
        player_role="Keeper",
        content={"opening_message": "The fog opens."},
    )
    original = repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
        custom_instructions="Keep the tone eerie.",
    )
    message = repositories.append_message(
        save_id=original.id,
        role="player",
        speaker_name="Keeper",
        body="I light the lantern.",
    )
    repositories.append_message(
        save_id=original.id,
        role="narrator",
        body="The lantern catches.",
    )
    save_ids = [save.id for save in repositories.list_saves()]

    with pytest.raises(ValueError, match="requires a turn snapshot"):
        SaveForkService(repositories).fork_from_message(
            save_id=original.id,
            message_id=message.id,
            media_dir=tmp_path / "media",
        )

    assert [save.id for save in repositories.list_saves()] == save_ids


def test_fork_from_player_message_includes_selected_player_message(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A beacon tower.",
        player_role="Keeper",
        content={},
    )
    original = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    snapshot_service = TurnSnapshotService(repositories)
    snapshot_service.capture_baseline_snapshot(original.id)
    player = repositories.append_message(
        save_id=original.id,
        role="player",
        speaker_name="Keeper",
        body="I touch the glass.",
    )
    snapshot_service.capture_message_snapshot(
        save_id=original.id,
        message_id=player.id,
    )
    repositories.append_message(
        save_id=original.id,
        role="narrator",
        body="It rings softly.",
    )

    result = SaveForkService(repositories).fork_from_message(
        save_id=original.id,
        message_id=player.id,
        media_dir=tmp_path / "media",
    )

    assert [message.body for message in repositories.list_messages(result.save.id)] == [
        "I touch the glass."
    ]
