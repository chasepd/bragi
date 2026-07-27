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
    prior_summary = repositories.add_summary(
        save_id=original.id,
        covers_message_start_id=player.id,
        covers_message_end_id=player.id,
        body="The keeper touched the glass.",
        provider="fake",
        model="fake-summary",
        source_message_ids=(player.id,),
    )
    repositories.archive_summary(prior_summary.id)
    repositories.add_summary(
        save_id=original.id,
        covers_message_start_id=player.id,
        covers_message_end_id=player.id,
        body="The keeper touched the glass and noted its response.",
        provider="fake",
        model="fake-summary",
        source_message_ids=(player.id,),
        source_summary_ids=(prior_summary.id,),
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
    forked_player = repositories.list_messages(result.save.id)[0]
    [forked_summary] = repositories.list_summaries(result.save.id)
    forked_prior_summary = repositories.connection.execute(
        """
        SELECT id, archived_at
        FROM summaries
        WHERE save_id = ? AND body = ?
        """,
        (result.save.id, "The keeper touched the glass."),
    ).fetchone()
    assert forked_prior_summary is not None
    assert forked_prior_summary["archived_at"] is not None
    assert forked_summary.source_message_ids == (forked_player.id,)
    assert forked_summary.source_summary_ids == (forked_prior_summary["id"],)
