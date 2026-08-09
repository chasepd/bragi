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


def test_fork_from_player_message_uses_preceding_turn_snapshot(
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
    opening = repositories.append_message(
        save_id=original.id,
        role="narrator",
        body="The lantern waits in darkness.",
    )
    repositories.upsert_world_state(
        save_id=original.id,
        key="lantern",
        value={"lit": False},
        source_message_id=opening.id,
    )
    snapshot_service.capture_message_snapshot(
        save_id=original.id,
        message_id=opening.id,
    )
    player = repositories.append_message(
        save_id=original.id,
        role="player",
        speaker_name="Keeper",
        body="I touch the glass.",
    )
    narrator = repositories.append_message(
        save_id=original.id,
        role="narrator",
        body="It rings softly.",
    )
    repositories.upsert_world_state(
        save_id=original.id,
        key="lantern",
        value={"lit": True},
        source_message_id=narrator.id,
    )
    snapshot_service.capture_message_snapshot(
        save_id=original.id,
        message_id=narrator.id,
    )

    result = SaveForkService(repositories).fork_from_message(
        save_id=original.id,
        message_id=player.id,
        media_dir=tmp_path / "media",
    )

    forked_messages = repositories.list_messages(result.save.id)
    assert [message.body for message in forked_messages] == [
        "The lantern waits in darkness.",
        "I touch the glass.",
    ]
    assert forked_messages[-1].role == "player"
    assert forked_messages[-1].id != player.id
    assert result.message_count == 2
    assert repositories.list_world_state(result.save.id)[0].value == {"lit": False}
    assert snapshot_service.latest_snapshot_for_message(
        save_id=result.save.id,
        message_id=forked_messages[-1].id,
    ) is not None


def test_fork_preserves_observation_epistemics_and_memory_provenance(
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
    actor = repositories.add_character(save_id=original.id, name="Mara")
    narrator = repositories.append_message(
        save_id=original.id,
        role="narrator",
        body="Mara says the eastern road is watched.",
    )
    observation = repositories.add_context_observation(
        save_id=original.id,
        observation_type="character_fact",
        claim="The eastern road is watched.",
        evidence_quote="the eastern road is watched",
        source_message_ids=(narrator.id,),
        epistemic_status="reported_speech",
        epistemic_actor_id=actor.id,
        epistemic_actor_name="Mara",
    )
    original_memory = repositories.add_memory(
        save_id=original.id,
        body="Mara reported that the eastern road is watched.",
        tags=["warning"],
        source_message_ids=(narrator.id,),
        source_observation_ids=(observation.id,),
        epistemic_status="reported_speech",
        epistemic_actor_id=actor.id,
        epistemic_actor_name="Mara",
    )
    TurnSnapshotService(repositories).capture_message_snapshot(
        save_id=original.id,
        message_id=narrator.id,
    )

    result = SaveForkService(repositories).fork_from_message(
        save_id=original.id,
        message_id=narrator.id,
        media_dir=tmp_path / "media",
    )

    [forked_observation] = repositories.list_context_observations(result.save.id)
    [forked_memory] = repositories.list_memories(result.save.id)
    assert forked_observation.epistemic_status == "reported_speech"
    assert forked_observation.epistemic_actor_name == "Mara"
    assert forked_memory.source_observation_ids == [forked_observation.id]
    assert forked_memory.epistemic_actor_id != actor.id
    assert forked_memory.claim_fingerprint != original_memory.claim_fingerprint
    duplicate = repositories.add_memory(
        save_id=result.save.id,
        body=forked_memory.body,
        tags=["warning"],
        epistemic_status=forked_memory.epistemic_status,
        epistemic_actor_id=forked_memory.epistemic_actor_id,
        epistemic_actor_name=forked_memory.epistemic_actor_name,
    )
    assert duplicate.id == forked_memory.id
