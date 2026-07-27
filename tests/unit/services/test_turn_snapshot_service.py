from __future__ import annotations

import base64
import json
import sqlite3
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.character_text_world_update_service import character_text_source_ref
from bragi.services.chat_bundle_service import ChatBundleService
from bragi.services.message_revision_service import MessageRevisionService
from bragi.services.save_fork_service import SaveForkService
from bragi.services.turn_snapshot_service import (
    TurnSnapshotService,
    _coalesce_remapped_snapshot_rows,
    _filter_character_text_snapshot_rows,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_snapshot_import_coalesces_many_to_one_memory_remaps() -> None:
    rows = _coalesce_remapped_snapshot_rows(
        "memories",
        [
            {
                "id": "merged-memory",
                "save_id": "target-save",
                "body": "The moonstone opens the archive.",
                "tags_json": '["moonstone"]',
                "importance": 0.4,
                "source_message_ids_json": '["message-one"]',
                "source_observation_ids_json": '["observation-one"]',
                "archived_at": None,
            },
            {
                "id": "merged-memory",
                "save_id": "target-save",
                "body": "the moonstone opens the archive",
                "tags_json": '["archive"]',
                "importance": 0.9,
                "source_message_ids_json": '["message-two"]',
                "source_observation_ids_json": '["observation-two"]',
                "archived_at": None,
            },
        ],
    )

    assert len(rows) == 1
    assert rows[0]["importance"] == 0.9
    assert json.loads(str(rows[0]["tags_json"])) == ["moonstone", "archive"]
    assert json.loads(str(rows[0]["source_message_ids_json"])) == [
        "message-one",
        "message-two",
    ]
    assert json.loads(str(rows[0]["source_observation_ids_json"])) == [
        "observation-one",
        "observation-two",
    ]


def test_capture_dedupes_objects_and_restore_recovers_exact_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Mara sees the red lens.",
    )
    repositories.add_location(
        save_id=save.id,
        location_id="tower",
        name="Beacon Tower",
        source_message_id=message.id,
    )
    repositories.add_character(
        save_id=save.id,
        character_id="mara",
        name="Mara",
        location_id="tower",
        source_message_id=message.id,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id="mara",
        target_type="location",
        target_id="tower",
        source_message_id=message.id,
        evidence_quote="Mara sees the tower.",
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=message.id,
        character_id="mara",
        visibility="visible",
        source="scene",
    )
    repositories.replace_message_action_choices(
        save_id=save.id,
        message_id=message.id,
        choices=("Study the lens", "Signal east"),
        provider="fake",
        model="choices",
    )
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    media_dir.joinpath("lens.png").write_bytes(b"image bytes stay on disk")
    repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path="lens.png",
        prompt="red lens",
        provider="fake",
        model="image",
        status="succeeded",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
        source_message_id=message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=1,
        world_time_day_index=1,
        world_time_day_label="monday",
        world_time_phase="morning",
        world_time_clock_minutes=8 * 60 + 30,
        world_time_period_label="bell watch",
        world_time_source_message_id=message.id,
        world_time_confidence=0.93,
        source_message_id=message.id,
    )

    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    row_object_count = _row_object_count(repositories)
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)
    assert _row_object_count(repositories) == row_object_count
    _assert_snapshot_objects_do_not_store_media_bytes(repositories)

    later = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I repaint the lens blue.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "blue"},
        source_message_id=later.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Tuesday evening",
        time_of_day="evening",
        day_of_week="tuesday",
        world_day_index=2,
        world_time_day_index=2,
        world_time_day_label="tuesday",
        world_time_phase="evening",
        world_time_clock_minutes=20 * 60 + 15,
        world_time_period_label="moon watch",
        world_time_source_message_id=later.id,
        world_time_confidence=0.42,
        source_message_id=later.id,
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    assert [message.body for message in repositories.list_messages(save.id)] == [
        "Mara sees the red lens."
    ]
    assert repositories.list_world_state(save.id)[0].value == {"color": "red"}
    assert repositories.list_character_knowledge_edges(save.id)[0].target_id == "tower"
    assert repositories.list_message_visibility(save.id)[0].message_id == message.id
    choices = repositories.latest_message_action_choices(save.id)
    assert [choice.body for choice in choices] == [
        "Study the lens",
        "Signal east",
    ]
    assert repositories.list_media_assets(save.id)[0].prompt == "red lens"
    restored_snapshot = repositories.get_scene_snapshot(save.id)
    assert restored_snapshot is not None
    assert restored_snapshot.world_day_index == 1
    assert restored_snapshot.world_time_day_index == 1
    assert restored_snapshot.world_time_day_label == "monday"
    assert restored_snapshot.world_time_phase == "morning"
    assert restored_snapshot.world_time_clock_minutes == 8 * 60 + 30
    assert restored_snapshot.world_time_period_label == "bell watch"
    assert restored_snapshot.source_message_id == message.id
    assert restored_snapshot.world_time_source_message_id == message.id
    assert restored_snapshot.world_time_confidence == 0.93


def test_snapshot_restore_preserves_unrated_narration_without_resanitizing(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    adult_body = "They had sex after returning to the inn."
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=adult_body,
    )
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    repositories.append_message(
        save_id=save.id,
        role="player",
        body="I leave the inn.",
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    restored_messages = repositories.list_messages(save.id)
    assert [restored.body for restored in restored_messages] == [adult_body]
    assert restored_messages[0].safety_transition == ""


def test_restore_rolls_back_when_snapshot_row_insert_fails(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lens burns red.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
        source_message_id=first.id,
    )
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=first.id,
    )
    later = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I repaint the lens blue.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "blue"},
        source_message_id=later.id,
    )
    repositories.connection.execute(
        """
        CREATE TRIGGER fail_lens_restore
        BEFORE INSERT ON world_state
        WHEN NEW.key = 'lens'
        BEGIN
            SELECT RAISE(ABORT, 'forced restore failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="forced restore failure"):
        service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    assert [message.body for message in repositories.list_messages(save.id)] == [
        "The lens burns red.",
        "I repaint the lens blue.",
    ]
    assert repositories.list_world_state(save.id)[0].value == {"color": "blue"}


def test_delete_from_here_restores_snapshot_before_anchor(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    baseline = service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I light the lantern.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lantern",
        value={"status": "lit"},
        source_message_id=first.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)
    second = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I douse the lantern.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lantern",
        value={"status": "dark"},
        source_message_id=second.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=second.id)

    deletion = MessageRevisionService(repositories).delete_suffix_from_message(
        save_id=save.id,
        message_id=second.id,
    )

    assert [message.id for message in deletion.deleted_messages] == [second.id]
    assert [message.id for message in repositories.list_messages(save.id)] == [first.id]
    assert repositories.list_world_state(save.id)[0].value == {"status": "lit"}

    MessageRevisionService(repositories).delete_suffix_from_message(
        save_id=save.id,
        message_id=first.id,
    )

    assert repositories.list_messages(save.id) == []
    assert repositories.list_world_state(save.id) == []
    restored_baseline = service.latest_snapshot_for_message(
        save_id=save.id,
        message_id=None,
    )
    assert restored_baseline is not None
    assert restored_baseline.id == baseline.id


def test_restore_includes_character_text_state_and_removes_later_rows(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan sends the ticket stub.",
    )
    seeded = _seed_character_text_snapshot_state(
        repositories,
        save_id=save.id,
        source_message_id=message.id,
        media_dir=media_dir,
    )
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )

    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ignore the message.",
    )
    later_text = repositories.append_character_text_message(
        message_id="text-rowan-later",
        save_id=save.id,
        thread_id=seeded["thread_id"],
        character_id=seeded["rowan_id"],
        sender="player",
        body="Not now.",
    )
    repositories.set_character_contact_state(
        save_id=save.id,
        player_character_id=seeded["player_id"],
        character_id=seeded["rowan_id"],
        player_has_character_number=False,
        character_has_player_number=True,
        source_text_message_id=later_text.id,
    )
    repositories.add_character_text_proactive_trigger(
        save_id=save.id,
        character_id=seeded["rowan_id"],
        trigger_key=f"character_intent:{seeded['rowan_id']}:later",
        trigger_type="character_intent",
        thread_id=seeded["thread_id"],
        text_message_id=later_text.id,
        source_type="character",
        source_id=seeded["rowan_id"],
        source_message_id=message.id,
        reason="Later trigger.",
        trigger_id="trigger-rowan-later",
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    text_messages = repositories.list_character_text_messages(save_id=save.id)
    assert [text_message.id for text_message in text_messages] == [
        seeded["text_message_id"]
    ]
    revisions = repositories.list_character_text_message_revisions(save_id=save.id)
    assert [(revision.id, revision.text_message_id) for revision in revisions] == [
        ("revision-rowan-text", seeded["text_message_id"])
    ]
    attachments = repositories.list_character_text_message_attachments(save_id=save.id)
    assert [
        (attachment.id, attachment.media_asset_id) for attachment in attachments
    ] == [
        ("text-attachment-ticket", seeded["media_asset_id"])
    ]
    provenance = repositories.list_character_text_provenance(save_id=save.id)
    assert [
        (row.text_message_id, row.target_type, row.target_id) for row in provenance
    ] == [
        (seeded["text_message_id"], "memory", seeded["memory_id"])
    ]
    contacts = repositories.list_character_contact_states(save.id)
    assert len(contacts) == 1
    assert contacts[0].player_has_character_number is True
    assert contacts[0].character_has_player_number is False
    assert contacts[0].source_text_message_id == seeded["text_message_id"]
    triggers = repositories.list_character_text_proactive_triggers(save.id)
    assert [(trigger.id, trigger.text_message_id) for trigger in triggers] == [
        ("trigger-rowan", seeded["text_message_id"])
    ]
    memories = repositories.list_memories(save.id)
    assert memories[0].source_message_ids == [
        character_text_source_ref(seeded["text_message_id"])
    ]


def test_snapshot_backed_fork_remaps_rows_and_copies_media(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    media_dir.joinpath("old.png").write_bytes(b"old image")
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The tower lens glows red.",
    )
    repositories.add_location(
        save_id=save.id,
        location_id="tower",
        name="Beacon Tower",
        source_message_id=first.id,
    )
    repositories.add_character(
        save_id=save.id,
        character_id="mara",
        name="Mara",
        location_id="tower",
        source_message_id=first.id,
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=first.id,
        character_id="mara",
        visibility="visible",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id="mara",
        target_type="location",
        target_id="tower",
        source_message_id=first.id,
    )
    repositories.create_media_asset(
        save_id=save.id,
        source_message_id=first.id,
        type="image",
        path="old.png",
        prompt="red lens",
        provider="fake",
        model="image",
        status="succeeded",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
        source_message_id=first.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=1,
        world_time_day_index=1,
        world_time_day_label="monday",
        world_time_phase="morning",
        world_time_clock_minutes=8 * 60 + 30,
        world_time_period_label="bell watch",
        world_time_source_message_id=first.id,
        world_time_confidence=0.93,
        source_message_id=first.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)
    second = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The tower lens turns blue.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "blue"},
        source_message_id=second.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Tuesday evening",
        time_of_day="evening",
        day_of_week="tuesday",
        world_day_index=2,
        world_time_day_index=2,
        world_time_day_label="tuesday",
        world_time_phase="evening",
        world_time_clock_minutes=20 * 60 + 15,
        world_time_period_label="moon watch",
        world_time_source_message_id=second.id,
        world_time_confidence=0.42,
        source_message_id=second.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=second.id)

    result = SaveForkService(repositories).fork_from_message(
        save_id=save.id,
        message_id=first.id,
        media_dir=media_dir,
    )

    fork_messages = repositories.list_messages(result.save.id)
    assert result.message_count == 1
    assert fork_messages[0].id != first.id
    assert fork_messages[0].body == "The tower lens glows red."
    assert repositories.list_world_state(result.save.id)[0].value == {"color": "red"}
    fork_location = repositories.list_locations(result.save.id)[0]
    fork_character = repositories.list_characters(result.save.id)[0]
    assert fork_location.id != "tower"
    assert fork_character.id != "mara"
    assert repositories.list_message_visibility(result.save.id)[0].message_id == (
        fork_messages[0].id
    )
    assert repositories.list_message_visibility(result.save.id)[0].character_id == (
        fork_character.id
    )
    assert repositories.list_character_knowledge_edges(result.save.id)[0].target_id == (
        fork_location.id
    )
    fork_media = repositories.list_media_assets(result.save.id)[0]
    assert fork_media.path != "old.png"
    assert (media_dir / fork_media.path).read_bytes() == b"old image"
    fork_snapshot = repositories.get_scene_snapshot(result.save.id)
    assert fork_snapshot is not None
    assert fork_snapshot.world_time_day_index == 1
    assert fork_snapshot.world_time_day_label == "monday"
    assert fork_snapshot.world_time_phase == "morning"
    assert fork_snapshot.world_time_clock_minutes == 8 * 60 + 30
    assert fork_snapshot.world_time_period_label == "bell watch"
    assert fork_snapshot.source_message_id == fork_messages[0].id
    assert fork_snapshot.world_time_source_message_id == fork_messages[0].id
    assert fork_snapshot.world_time_confidence == 0.93


def test_snapshot_backed_fork_rejects_media_paths_outside_media_root(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    outside_path = tmp_path / "outside.png"
    outside_path.write_bytes(b"outside image")
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower lens glows red.",
    )
    repositories.connection.execute(
        """
        INSERT INTO media_assets(
            id, save_id, source_message_id, type, path, thumbnail_path, prompt,
            provider, model, status, mime_type, metadata_json
        )
        VALUES (?, ?, ?, 'image', '../outside.png', NULL, 'escaped path',
                'fake', 'image', 'succeeded', 'image/png', '{}')
        """,
        ("escaped-media", save.id, message.id),
    )
    repositories.commit()
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)

    with pytest.raises(ValueError, match="escapes media directory"):
        SaveForkService(repositories).fork_from_message(
            save_id=save.id,
            message_id=message.id,
            media_dir=media_dir,
        )

    assert [listed_save.id for listed_save in repositories.list_saves()] == [save.id]
    assert not (media_dir / "forks").exists()


def test_snapshot_backed_fork_remaps_character_text_rows_and_attachment_media(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan sends the ticket stub.",
    )
    seeded = _seed_character_text_snapshot_state(
        repositories,
        save_id=save.id,
        source_message_id=message.id,
        media_dir=media_dir,
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="character_text_thread",
        source_id=seeded["thread_id"],
        title="Rowan text thread",
        body="Phone thread memory: Rowan found a ticket stub.",
        metadata={
            "source_message_ids": [
                character_text_source_ref(seeded["text_message_id"])
            ],
            "audience_character_ids": [
                seeded["player_id"],
                seeded["rowan_id"],
            ],
            "thread_id": seeded["thread_id"],
            "entity_ids": [seeded["thread_id"]],
        },
    )
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)

    fork = SaveForkService(repositories).fork_from_message(
        save_id=save.id,
        message_id=message.id,
        media_dir=media_dir,
    )

    fork_message = repositories.list_messages(fork.save.id)[0]
    fork_characters = {
        character.name: character
        for character in repositories.list_characters(fork.save.id)
    }
    fork_player = fork_characters["Mara"]
    fork_rowan = fork_characters["Rowan"]
    fork_thread = repositories.list_character_text_threads(fork.save.id)[0]
    fork_text = repositories.list_character_text_messages(save_id=fork.save.id)[0]
    fork_media = repositories.list_media_assets(fork.save.id)[0]
    fork_attachment = repositories.list_character_text_message_attachments(
        save_id=fork.save.id
    )[0]
    fork_memory = repositories.list_memories(fork.save.id)[0]

    assert fork_thread.id != seeded["thread_id"]
    assert fork_thread.character_id == fork_rowan.id
    assert fork_text.id != seeded["text_message_id"]
    assert fork_text.thread_id == fork_thread.id
    assert fork_text.character_id == fork_rowan.id
    assert fork_attachment.text_message_id == fork_text.id
    assert fork_attachment.character_id == fork_rowan.id
    assert fork_attachment.media_asset_id == fork_media.id
    assert fork_media.id != seeded["media_asset_id"]
    assert fork_media.path != seeded["media_path"]
    assert (media_dir / fork_media.path).read_bytes() == b"text attachment image"
    media_metadata = json.loads(fork_media.metadata_json)
    assert media_metadata["character_id"] == fork_rowan.id
    assert media_metadata["thread_id"] == fork_thread.id
    assert media_metadata["text_message_id"] == fork_text.id
    contacts = repositories.list_character_contact_states(fork.save.id)
    assert len(contacts) == 1
    assert contacts[0].player_character_id == fork_player.id
    assert contacts[0].character_id == fork_rowan.id
    assert contacts[0].source_message_id == fork_message.id
    assert contacts[0].source_text_message_id == fork_text.id
    triggers = repositories.list_character_text_proactive_triggers(fork.save.id)
    assert len(triggers) == 1
    assert triggers[0].character_id == fork_rowan.id
    assert triggers[0].thread_id == fork_thread.id
    assert triggers[0].text_message_id == fork_text.id
    assert triggers[0].source_id == fork_rowan.id
    assert triggers[0].source_message_id == fork_message.id
    assert triggers[0].trigger_key == (
        f"character_intent:{fork_rowan.id}:{fork_message.id}"
    )
    provenance = repositories.list_character_text_provenance(save_id=fork.save.id)
    assert len(provenance) == 1
    assert provenance[0].thread_id == fork_thread.id
    assert provenance[0].text_message_id == fork_text.id
    assert provenance[0].target_id == fork_memory.id
    assert fork_memory.source_message_ids == [
        character_text_source_ref(fork_text.id)
    ]
    [fork_thread_context] = repositories.list_context_sources(
        fork.save.id,
        source_type="character_text_thread",
    )
    assert fork_thread_context.source_id == fork_thread.id
    assert fork_thread_context.metadata["source_message_ids"] == [
        character_text_source_ref(fork_text.id)
    ]
    audience_character_ids = fork_thread_context.metadata["audience_character_ids"]
    assert isinstance(audience_character_ids, list)
    assert set(audience_character_ids) == {fork_player.id, fork_rowan.id}
    assert fork_thread_context.metadata["thread_id"] == fork_thread.id
    assert fork_thread_context.metadata["entity_ids"] == [fork_thread.id]
    revisions = repositories.list_character_text_message_revisions(
        save_id=fork.save.id
    )
    assert [
        (revision.text_message_id, revision.previous_body) for revision in revisions
    ] == [
        (fork_text.id, "I found a ticket stub.")
    ]


def test_restore_snapshot_replaces_scene_scratch_before_its_scene(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens stays warm.",
    )
    scene = repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The beacon lens stays warm.",
        source_message_id=message.id,
    )
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="scene_detail",
        claim="The beacon lens is warm.",
        evidence_quote="beacon lens stays warm",
        source_message_ids=[message.id],
        scope="scene",
        status="accepted",
    )
    scratch = repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id=observation.id,
        title="Warm lens",
        body=observation.claim,
        metadata={
            "observation_id": observation.id,
            "curation_action": "scene_scratch",
        },
        scene_snapshot_id=scene.id,
        scene_generation=scene.scene_generation,
        created_turn_number=1,
        expires_after_turn_number=13,
    )
    service = TurnSnapshotService(repositories)
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    restored_scene = repositories.get_scene_snapshot(save.id)
    restored_scratch = repositories.get_context_source(scratch.id)
    assert restored_scene is not None
    assert restored_scratch is not None
    assert restored_scratch.scene_snapshot_id == restored_scene.id

    fork = SaveForkService(repositories).fork_from_message(
        save_id=save.id,
        message_id=message.id,
        media_dir=tmp_path / "media",
    )
    [fork_observation] = repositories.list_context_observations(fork.save.id)
    [fork_scratch] = repositories.list_context_sources(
        fork.save.id,
        source_type="observation",
    )
    assert fork_scratch.source_id == fork_observation.id


def test_snapshot_backed_fork_remaps_character_text_reply_links(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Mara and Rowan exchange texts.",
    )
    repositories.add_character(
        character_id="mara",
        save_id=save.id,
        name="Mara",
        role="Signal warden",
        is_player_character=True,
        met=True,
        source_message_id=message.id,
    )
    rowan = repositories.add_character(
        character_id="rowan",
        save_id=save.id,
        name="Rowan",
        role="Courier",
        met=True,
        source_message_id=message.id,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=rowan.id,
        title="Rowan",
    )
    player_text = repositories.append_character_text_message(
        message_id="text-player-where",
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="player",
        body="Where are you?",
        in_world_sent_at="Friday evening after class",
        delivered_at="2026-07-01T12:05:00+00:00",
    )
    repositories.append_character_text_message(
        message_id="text-rowan-reply",
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="character",
        body="By the south gate.",
        in_world_sent_at="Friday evening after class",
        delivered_at="2026-07-01T12:06:00+00:00",
        read_at="2026-07-01T12:07:00+00:00",
        reply_to_message_id=player_text.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)

    fork = SaveForkService(repositories).fork_from_message(
        save_id=save.id,
        message_id=message.id,
        media_dir=tmp_path / "media",
    )

    fork_texts = repositories.list_character_text_messages(save_id=fork.save.id)
    fork_player_text = next(
        text for text in fork_texts if text.body == "Where are you?"
    )
    fork_reply = next(text for text in fork_texts if text.body == "By the south gate.")
    assert fork_player_text.id != player_text.id
    assert fork_reply.reply_to_message_id == fork_player_text.id
    assert fork_reply.in_world_sent_at == "Friday evening after class"
    assert fork_reply.delivered_at == "2026-07-01T12:06:00+00:00"
    assert fork_reply.read_at == "2026-07-01T12:07:00+00:00"


def test_character_text_snapshot_filter_prunes_stale_reply_chains() -> None:
    rows: dict[str, tuple[dict[str, object], ...]] = {
        "characters": ({"id": "character-rowan"},),
        "messages": (),
        "media_assets": (),
        "character_text_threads": (
            {"id": "thread-rowan", "character_id": "character-rowan"},
        ),
        "character_text_messages": (
            {
                "id": "text-keep",
                "thread_id": "thread-rowan",
                "character_id": "character-rowan",
                "reply_to_message_id": None,
            },
            {
                "id": "text-first-dependent",
                "thread_id": "thread-rowan",
                "character_id": "character-rowan",
                "reply_to_message_id": "text-pruned-target",
            },
            {
                "id": "text-second-dependent",
                "thread_id": "thread-rowan",
                "character_id": "character-rowan",
                "reply_to_message_id": "text-first-dependent",
            },
            {
                "id": "text-pruned-target",
                "thread_id": "thread-rowan",
                "character_id": "character-rowan",
                "reply_to_message_id": "text-missing",
            },
        ),
    }

    _filter_character_text_snapshot_rows(rows)

    assert [row["id"] for row in rows["character_text_messages"]] == ["text-keep"]


def test_manual_edits_are_removed_before_later_snapshot_and_included_at_head(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check the lens.",
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)
    repositories.upsert_world_state(
        save_id=save.id,
        key="manual.note",
        value={"body": "added after first message"},
        source_message_id=None,
    )
    second = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I keep walking.",
    )
    service.capture_message_snapshot(save_id=save.id, message_id=second.id)

    fork = SaveForkService(repositories).fork_from_message(
        save_id=save.id,
        message_id=second.id,
        media_dir=tmp_path / "media",
    )
    assert repositories.list_world_state(fork.save.id)[0].key == "manual.note"

    MessageRevisionService(repositories).delete_suffix_from_message(
        save_id=save.id,
        message_id=second.id,
    )

    assert repositories.list_world_state(save.id) == []


def test_dirty_head_capture_tracks_message_body_edits_without_context_change(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check the lens.",
    )
    original = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )

    repositories.update_message_body(
        save_id=save.id,
        message_id=message.id,
        body="I polish the lens.",
    )

    edited = service.capture_current_head_if_dirty(
        save.id,
        reason="message_edit",
    )

    assert edited.id != original.id
    repositories.update_message_body(
        save_id=save.id,
        message_id=message.id,
        body="I crack the lens.",
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=edited.id)

    assert repositories.list_messages(save.id)[0].body == "I polish the lens."


def test_restore_snapshot_strips_deprecated_scenario_update_sections(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    legacy_content: dict[str, object] = {
        "opening_message": "The red lens wakes.",
        "factions": "Beacon wardens",
        "rivals_and_factions": "Ash riders scout the ridge.",
        "reputation_and_contacts": "The old patrol owes Mara a warning.",
        "major_npcs": "Captain Rell guards the cracked stair.",
    }
    update = repositories.add_save_scenario_update(
        save_id=save.id,
        title="Ashfall Keep: Red Lens",
        premise="A red warning has reached the isolated border keep.",
        player_role="Signal warden",
        content=legacy_content,
        reason="Legacy snapshot fixture.",
        provider="fake-chat-provider",
        model="fake-chat-model",
    )
    service = TurnSnapshotService(repositories)
    snapshot = service.capture_baseline_snapshot(save.id)
    manifest = service._snapshot_manifest(snapshot)  # noqa: SLF001 - legacy fixture
    tables = manifest["tables"]
    assert isinstance(tables, dict)
    update_entries = tables["save_scenario_updates"]
    assert isinstance(update_entries, list)
    [update_entry] = update_entries
    assert isinstance(update_entry, dict)
    row = repositories.connection.execute(
        "SELECT * FROM save_scenario_updates WHERE id = ?",
        (update.id,),
    ).fetchone()
    legacy_row = dict(row)
    legacy_row["content_json"] = json.dumps(legacy_content, sort_keys=True)
    update_entry["object_hash"] = service._store_object(  # noqa: SLF001 - fixture
        kind="row:save_scenario_updates",
        value=legacy_row,
    )
    legacy_manifest_hash = service._store_object(  # noqa: SLF001 - fixture
        kind="snapshot_manifest",
        value=manifest,
    )
    repositories.connection.execute(
        "UPDATE save_turn_snapshots SET root_manifest_hash = ? WHERE id = ?",
        (legacy_manifest_hash, snapshot.id),
    )
    repositories.connection.execute(
        "UPDATE save_scenario_updates SET content_json = ? WHERE id = ?",
        (json.dumps({"factions": "Current wardens"}, sort_keys=True), update.id),
    )
    repositories.commit()
    _snapshot_rows, snapshot_objects = service.export_snapshot_rows(
        save_id=save.id,
        active_message_ids=(),
    )
    exported_payloads = "\n".join(
        zlib.decompress(base64.b64decode(str(row["payload_base64"]))).decode("utf-8")
        for row in snapshot_objects
    )
    assert "rivals_and_factions" not in exported_payloads
    assert "reputation_and_contacts" not in exported_payloads
    assert "major_npcs" not in exported_payloads
    assert "Ash riders scout the ridge." in exported_payloads

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    [restored] = repositories.list_save_scenario_updates(save.id)
    restored_content = json.loads(restored.content_json)
    assert restored_content["factions"] == (
        "Beacon wardens\n\n"
        "Ash riders scout the ridge.\n\n"
        "The old patrol owes Mara a warning."
    )
    assert "rivals_and_factions" not in restored_content
    assert "reputation_and_contacts" not in restored_content
    assert "major_npcs" not in restored_content


def test_export_import_preserves_snapshot_backed_fork(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I set the lens red.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
        source_message_id=first.id,
    )
    repositories.add_save_scenario_update(
        save_id=save.id,
        title="Red Lens",
        premise="The beacon lens turns red.",
        player_role="Signal warden",
        content={
            "character_starters": [
                {
                    "name": "Mara",
                    "reference_image": {
                        "media_asset_id": "starter-reference",
                        "content_rating": "r",
                    },
                }
            ],
            "_source": {"content_rating": "r"},
        },
        reason="Snapshot rating quarantine fixture.",
        provider="fake",
        model="fake-chat",
        source_message_id=first.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)
    second = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I set the lens blue.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "blue"},
        source_message_id=second.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=second.id)
    bundle_path = tmp_path / "night-watch.bragi-chat"

    ChatBundleService(repositories=repositories, media_dir=media_dir).export_save(
        save.id,
        bundle_path,
    )

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["turn_snapshots"]
    assert data["snapshot_objects"]

    imported = ChatBundleService(
        repositories=repositories,
        media_dir=media_dir,
    ).import_save(bundle_path)
    imported_messages = repositories.list_messages(imported.save_id)
    imported_first = imported_messages[0]

    fork = SaveForkService(repositories).fork_from_message(
        save_id=imported.save_id,
        message_id=imported_first.id,
        media_dir=media_dir,
    )

    fork_messages = repositories.list_messages(fork.save.id)
    assert [message.body for message in fork_messages] == ["I set the lens red."]
    assert repositories.list_world_state(fork.save.id)[0].value == {"color": "red"}
    [fork_update] = repositories.list_save_scenario_updates(fork.save.id)
    fork_content = json.loads(fork_update.content_json)
    assert fork_content["_source"]["content_rating"] == "unclassified"
    assert fork_content["character_starters"][0]["reference_image"][
        "content_rating"
    ] == "unclassified"


def test_export_import_preserves_snapshot_only_media_for_fork(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    media_dir.joinpath("old.png").write_bytes(b"archived image bytes")
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I create the first image.",
    )
    media = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=first.id,
        type="image",
        path="old.png",
        prompt="old image",
        provider="fake",
        model="image",
        status="succeeded",
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)
    repositories.archive_media_asset(save_id=save.id, media_asset_id=media.id)
    second = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I archive that image.",
    )
    service.capture_message_snapshot(save_id=save.id, message_id=second.id)
    bundle_path = tmp_path / "snapshot-media.bragi-chat"

    ChatBundleService(repositories=repositories, media_dir=media_dir).export_save(
        save.id,
        bundle_path,
    )

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["media_assets"] == []
    assert data["snapshot_media_assets"][0]["id"] == media.id

    imported = ChatBundleService(
        repositories=repositories,
        media_dir=media_dir,
    ).import_save(bundle_path)
    imported_first = repositories.list_messages(imported.save_id)[0]

    fork = SaveForkService(repositories).fork_from_message(
        save_id=imported.save_id,
        message_id=imported_first.id,
        media_dir=media_dir,
    )

    fork_media = repositories.list_media_assets(fork.save.id)[0]
    assert fork_media.path != "old.png"
    assert (media_dir / fork_media.path).read_bytes() == b"archived image bytes"


def test_export_import_preserves_character_text_snapshot_backed_fork(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan sends the ticket stub.",
    )
    _seed_character_text_snapshot_state(
        repositories,
        save_id=save.id,
        source_message_id=message.id,
        media_dir=media_dir,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)
    bundle_path = tmp_path / "character-text-snapshot.bragi-chat"

    ChatBundleService(repositories=repositories, media_dir=media_dir).export_save(
        save.id,
        bundle_path,
    )
    imported = ChatBundleService(
        repositories=repositories,
        media_dir=media_dir,
    ).import_save(bundle_path)
    imported_message = repositories.list_messages(imported.save_id)[0]

    fork = SaveForkService(repositories).fork_from_message(
        save_id=imported.save_id,
        message_id=imported_message.id,
        media_dir=media_dir,
    )

    fork_text = repositories.list_character_text_messages(save_id=fork.save.id)[0]
    fork_thread = repositories.list_character_text_threads(fork.save.id)[0]
    fork_media = repositories.list_media_assets(fork.save.id)[0]
    fork_attachment = repositories.list_character_text_message_attachments(
        save_id=fork.save.id
    )[0]
    fork_memory = repositories.list_memories(fork.save.id)[0]
    fork_rowan = next(
        character
        for character in repositories.list_characters(fork.save.id)
        if character.name == "Rowan"
    )

    assert fork_text.body == "I found a ticket stub."
    assert fork_text.thread_id == fork_thread.id
    assert fork_attachment.text_message_id == fork_text.id
    assert fork_attachment.media_asset_id == fork_media.id
    assert (media_dir / fork_media.path).read_bytes() == b"text attachment image"
    assert json.loads(fork_media.metadata_json)["text_message_id"] == fork_text.id
    assert fork_memory.source_message_ids == [
        character_text_source_ref(fork_text.id)
    ]
    assert repositories.list_character_text_provenance(
        save_id=fork.save.id,
        text_message_id=fork_text.id,
    )[0].target_id == fork_memory.id
    trigger = repositories.list_character_text_proactive_triggers(fork.save.id)[0]
    assert trigger.character_id == fork_rowan.id
    assert trigger.text_message_id == fork_text.id
    assert trigger.source_id == fork_rowan.id


def _create_save(repositories: PersistenceRepositories) -> SaveRecord:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A beacon tower.",
        player_role="Signal warden",
        content={"opening_message": "The lens hums."},
    )
    return repositories.create_save(scenario_id=scenario.id, title="Night Watch")


def _seed_character_text_snapshot_state(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    source_message_id: str,
    media_dir: Path,
) -> dict[str, str]:
    media_path = "texts/rowan-ticket.png"
    (media_dir / media_path).parent.mkdir(parents=True, exist_ok=True)
    (media_dir / media_path).write_bytes(b"text attachment image")
    player = repositories.add_character(
        character_id="mara",
        save_id=save_id,
        name="Mara",
        role="Signal warden",
        is_player_character=True,
        met=True,
        source_message_id=source_message_id,
    )
    rowan = repositories.add_character(
        character_id="rowan",
        save_id=save_id,
        name="Rowan",
        role="Courier",
        met=True,
        current_intent="Ask Mara to decode the ticket stub.",
        source_message_id=source_message_id,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=rowan.id,
        title="Rowan",
    )
    text_message = repositories.append_character_text_message(
        message_id="text-rowan-ticket",
        save_id=save_id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="character",
        body="I found a ticket stub.",
        provider="fake-text-provider",
        model="fake-text-model",
    )
    repositories.add_character_text_message_revision(
        revision_id="revision-rowan-text",
        save_id=save_id,
        text_message_id=text_message.id,
        previous_body="I found a ticket stub.",
        new_body="I found a ticket stub with blue ink.",
        diff_unified=(
            "--- previous\n"
            "+++ current\n"
            "@@ -1 +1 @@\n"
            "-I found a ticket stub.\n"
            "+I found a ticket stub with blue ink.\n"
        ),
        reconciliation_status="succeeded",
    )
    media_asset = repositories.create_media_asset(
        asset_id="media-rowan-ticket",
        save_id=save_id,
        source_message_id=None,
        type="image",
        path=media_path,
        prompt="ticket stub",
        provider="fake-image-provider",
        model="fake-image-model",
        status="succeeded",
        metadata={
            "kind": "character_text_object_context_image",
            "character_id": rowan.id,
            "thread_id": thread.id,
            "text_message_id": text_message.id,
        },
    )
    repositories.add_character_text_message_attachment(
        attachment_id="text-attachment-ticket",
        save_id=save_id,
        thread_id=thread.id,
        text_message_id=text_message.id,
        character_id=rowan.id,
        kind="object_context_image",
        status="succeeded",
        media_asset_id=media_asset.id,
        prompt="ticket stub",
        metadata={"decision_reason": "Rowan texted a concrete found object."},
    )
    memory = repositories.add_memory(
        memory_id="memory-rowan-ticket",
        save_id=save_id,
        body="Rowan found a ticket stub.",
        tags=["rowan"],
        source_message_ids=[character_text_source_ref(text_message.id)],
    )
    repositories.add_character_text_provenance(
        provenance_id="provenance-rowan-memory",
        save_id=save_id,
        thread_id=thread.id,
        text_message_id=text_message.id,
        target_type="memory",
        target_id=memory.id,
        operation="create",
        field_path="body",
    )
    repositories.upsert_character_contact_state(
        state_id="contact-rowan",
        save_id=save_id,
        player_character_id=player.id,
        character_id=rowan.id,
        player_has_character_number=True,
        character_has_player_number=False,
        source_message_id=source_message_id,
        source_text_message_id=text_message.id,
    )
    repositories.add_character_text_proactive_trigger(
        trigger_id="trigger-rowan",
        save_id=save_id,
        character_id=rowan.id,
        trigger_key=f"character_intent:{rowan.id}:{source_message_id}",
        trigger_type="character_intent",
        thread_id=thread.id,
        text_message_id=text_message.id,
        source_type="character",
        source_id=rowan.id,
        source_message_id=source_message_id,
        reason="Ask Mara to decode the ticket stub.",
    )
    return {
        "player_id": player.id,
        "rowan_id": rowan.id,
        "thread_id": thread.id,
        "text_message_id": text_message.id,
        "media_asset_id": media_asset.id,
        "media_path": media_path,
        "memory_id": memory.id,
    }


def _row_object_count(repositories: PersistenceRepositories) -> int:
    row = repositories.connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM save_snapshot_objects
        WHERE kind LIKE 'row:%'
        """
    ).fetchone()
    return int(row["count"])


def _assert_snapshot_objects_do_not_store_media_bytes(
    repositories: PersistenceRepositories,
) -> None:
    rows = repositories.connection.execute(
        """
        SELECT payload
        FROM save_snapshot_objects
        """
    ).fetchall()
    payloads = b"\n".join(zlib.decompress(bytes(row["payload"])) for row in rows)
    assert b"image bytes stay on disk" not in payloads
