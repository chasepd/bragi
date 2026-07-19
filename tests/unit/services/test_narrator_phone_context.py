from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import CharacterRecord, SaveRecord, ScenarioRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.narrator_phone_context import (
    build_narrator_phone_activity_context,
    build_narrator_phone_context,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_build_narrator_phone_context_includes_addressed_thread_with_limits(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Signal House",
        premise="Students coordinate around a tower signal.",
        player_role="Mara",
        content={"starting_scene": "The tower darkens before class."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Signal House")
    player = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="player",
        is_player_character=True,
    )
    mika = repositories.add_character(
        save_id=save.id,
        name="Mika",
        role="classmate",
        met=True,
    )
    rowan = repositories.add_character(
        save_id=save.id,
        name="Rowan",
        role="teacher",
        met=True,
    )
    repositories.upsert_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=mika.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    repositories.upsert_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=rowan.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check whether Mika has texted me back.",
    )
    mika_thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=mika.id,
        title="Mika",
    )
    for index in range(10):
        repositories.append_character_text_message(
            save_id=save.id,
            thread_id=mika_thread.id,
            character_id=mika.id,
            sender="player" if index % 2 == 0 else "character",
            body=f"Mika thread message {index}",
            in_world_sent_at=f"Monday 8:{index:02d} AM",
        )
    rowan_thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=rowan.id,
        title="Rowan",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=rowan_thread.id,
        character_id=rowan.id,
        sender="character",
        body="Rowan thread should stay out of this narrator prompt.",
    )

    context = build_narrator_phone_context(
        repositories=repositories,
        save_id=save.id,
        scenario=scenario,
        messages=[player_message],
        player_message=player_message,
        scene_snapshot=None,
        characters=tuple(repositories.list_characters(save.id)),
    )

    phone_context = "\n".join(context.lines)
    assert context.thread_count == 1
    assert context.message_count == 8
    assert "Phone thread: Mika" in phone_context
    assert "Mika thread message 2" in phone_context
    assert "Mika thread message 9" in phone_context
    assert "Mika thread message 0" not in phone_context
    assert "Rowan thread should stay out" not in phone_context


def test_build_narrator_phone_context_omits_unsent_text_messages(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Signal House",
        premise="Students coordinate around a tower signal.",
        player_role="Mara",
        content={"starting_scene": "The tower darkens before class."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Signal House")
    player = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="player",
        is_player_character=True,
    )
    mika = repositories.add_character(
        save_id=save.id,
        name="Mika",
        role="classmate",
        met=True,
    )
    repositories.upsert_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=mika.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check whether Mika has texted me back.",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=mika.id,
        title="Mika",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=mika.id,
        sender="player",
        body="Undelivered confession draft.",
        delivery_status="failed",
        delivery_error="provider failed",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=mika.id,
        sender="player",
        body="Still sending private draft.",
        delivery_status="pending",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=mika.id,
        sender="character",
        body="Delivered check-in.",
    )

    context = build_narrator_phone_context(
        repositories=repositories,
        save_id=save.id,
        scenario=scenario,
        messages=[player_message],
        player_message=player_message,
        scene_snapshot=None,
        characters=tuple(repositories.list_characters(save.id)),
    )

    phone_context = "\n".join(context.lines)
    assert "Delivered check-in." in phone_context
    assert "Undelivered confession draft." not in phone_context
    assert "Still sending private draft." not in phone_context


def test_build_narrator_phone_context_omits_unreferenced_present_thread(
    repositories: PersistenceRepositories,
) -> None:
    scenario, save, _player, mika, _rowan = _create_phone_save(repositories)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mika studies the signal tower beside Mara.",
        present_character_ids=[mika.id],
    )
    previous_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Mika waits by the tower controls.",
        created_at="2000-01-01 12:00:00",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ask Mika what she thinks about the tower signal.",
        created_at="2000-01-01 12:05:00",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=mika.id,
        title="Mika",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=mika.id,
        sender="character",
        body="Private text should not bleed into narration.",
        delivered_at="2000-01-01 12:03:00",
        read_at="2000-01-01 12:04:00",
    )

    context = build_narrator_phone_context(
        repositories=repositories,
        save_id=save.id,
        scenario=scenario,
        messages=[previous_narrator, player_message],
        player_message=player_message,
        scene_snapshot=repositories.get_scene_snapshot(save.id),
        characters=tuple(repositories.list_characters(save.id)),
    )

    assert context.lines == ()
    assert context.thread_count == 0
    assert context.message_count == 0


def test_build_narrator_phone_context_omits_unreferenced_recent_outbound_text(
    repositories: PersistenceRepositories,
) -> None:
    scenario, save, _player, mika, _rowan = _create_phone_save(repositories)
    previous_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower hall is quiet.",
        created_at="2000-01-01 12:00:00",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check the tower lock.",
        created_at="2000-01-01 12:05:00",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=mika.id,
        title="Mika",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=mika.id,
        sender="player",
        body="Outbound private plan should not appear.",
        delivered_at="2000-01-01 12:03:00",
    )

    context = build_narrator_phone_context(
        repositories=repositories,
        save_id=save.id,
        scenario=scenario,
        messages=[previous_narrator, player_message],
        player_message=player_message,
        scene_snapshot=None,
        characters=tuple(repositories.list_characters(save.id)),
    )

    assert context.lines == ()
    assert context.thread_count == 0
    assert context.message_count == 0


def test_build_narrator_phone_context_summarizes_unread_incoming_without_body(
    repositories: PersistenceRepositories,
) -> None:
    scenario, save, _player, mika, _rowan = _create_phone_save(repositories)
    previous_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower hall is quiet.",
        created_at="2000-01-01 12:00:00",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I keep watching the tower door.",
        created_at="2000-01-01 12:05:00",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=mika.id,
        title="Mika",
    )
    repositories.update_character_text_thread_memory(
        save_id=save.id,
        thread_id=thread.id,
        body="Mika's memory contains private details.",
        message_count=1,
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=mika.id,
        sender="character",
        body="Meet me behind the archive door.",
        delivered_at="2000-01-01 12:03:00",
    )

    context = build_narrator_phone_context(
        repositories=repositories,
        save_id=save.id,
        scenario=scenario,
        messages=[previous_narrator, player_message],
        player_message=player_message,
        scene_snapshot=None,
        characters=tuple(repositories.list_characters(save.id)),
    )

    assert context.lines == ()
    activity = build_narrator_phone_activity_context(
        repositories=repositories,
        save_id=save.id,
        messages=[previous_narrator, player_message],
        player_message=player_message,
        characters=tuple(repositories.list_characters(save.id)),
    )
    phone_activity = "\n".join(activity.lines)
    assert "Received a text notification from Mika." in phone_activity
    assert "Meet me behind the archive door." not in phone_activity
    assert "Mika's memory contains private details." not in phone_activity


def test_build_narrator_phone_context_omits_messages_marked_read_by_thread(
    repositories: PersistenceRepositories,
) -> None:
    scenario, save, _player, mika, _rowan = _create_phone_save(repositories)
    previous_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower hall is quiet.",
        created_at="2000-01-01 12:00:00",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I keep watching the tower door.",
        created_at="2000-01-01 12:05:00",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=mika.id,
        title="Mika",
    )
    unread = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=mika.id,
        sender="character",
        body="Meet me behind the archive door.",
        delivered_at="2000-01-01 12:03:00",
    )

    unread_context = build_narrator_phone_context(
        repositories=repositories,
        save_id=save.id,
        scenario=scenario,
        messages=[previous_narrator, player_message],
        player_message=player_message,
        scene_snapshot=None,
        characters=tuple(repositories.list_characters(save.id)),
    )
    repositories.mark_character_text_thread_read(
        save_id=save.id,
        thread_id=thread.id,
        through_message_id=unread.id,
    )
    read_context = build_narrator_phone_context(
        repositories=repositories,
        save_id=save.id,
        scenario=scenario,
        messages=[previous_narrator, player_message],
        player_message=player_message,
        scene_snapshot=None,
        characters=tuple(repositories.list_characters(save.id)),
    )

    assert unread_context.lines == ()
    assert read_context.lines == ()


def test_build_narrator_phone_context_targeted_text_omits_unrelated_unread_body(
    repositories: PersistenceRepositories,
) -> None:
    scenario, save, _player, mika, rowan = _create_phone_save(repositories)
    previous_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower hall is quiet.",
        created_at="2000-01-01 12:00:00",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I message Mika about the tower signal.",
        created_at="2000-01-01 12:05:00",
    )
    mika_thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=mika.id,
        title="Mika",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=mika_thread.id,
        character_id=mika.id,
        sender="player",
        body="Mika targeted thread should be visible.",
        delivered_at="2000-01-01 12:02:00",
    )
    rowan_thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=rowan.id,
        title="Rowan",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=rowan_thread.id,
        character_id=rowan.id,
        sender="character",
        body="Rowan unread body should not be visible.",
        delivered_at="2000-01-01 12:03:00",
    )

    context = build_narrator_phone_context(
        repositories=repositories,
        save_id=save.id,
        scenario=scenario,
        messages=[previous_narrator, player_message],
        player_message=player_message,
        scene_snapshot=None,
        characters=tuple(repositories.list_characters(save.id)),
    )

    phone_context = "\n".join(context.lines)
    assert "Phone thread: Mika" in phone_context
    assert "Mika targeted thread should be visible." in phone_context
    assert "Rowan unread body should not be visible." not in phone_context
    assert "Recent phone messages with Rowan" not in phone_context


def test_build_narrator_phone_context_omits_read_or_old_incoming_notifications(
    repositories: PersistenceRepositories,
) -> None:
    scenario, save, _player, mika, rowan = _create_phone_save(repositories)
    previous_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower hall is quiet.",
        created_at="2000-01-01 12:00:00",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I keep watching the tower door.",
        created_at="2000-01-01 12:05:00",
    )
    read_thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=mika.id,
        title="Mika",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=read_thread.id,
        character_id=mika.id,
        sender="character",
        body="Read incoming message should stay out.",
        delivered_at="2000-01-01 12:03:00",
        read_at="2000-01-01 12:04:00",
    )
    old_thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=rowan.id,
        title="Rowan",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=old_thread.id,
        character_id=rowan.id,
        sender="character",
        body="Old unread message should stay out.",
        delivered_at="2000-01-01 11:55:00",
    )

    context = build_narrator_phone_context(
        repositories=repositories,
        save_id=save.id,
        scenario=scenario,
        messages=[previous_narrator, player_message],
        player_message=player_message,
        scene_snapshot=None,
        characters=tuple(repositories.list_characters(save.id)),
    )

    assert context.lines == ()
    assert context.thread_count == 0
    assert context.message_count == 0


def test_phone_activity_context_is_body_free_and_cursor_ordered(
    repositories: PersistenceRepositories,
) -> None:
    scenario, save, _player, mika, _rowan = _create_phone_save(repositories)
    previous = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower hall is quiet.",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id, character_id=mika.id, title="Mika"
    )
    incoming = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=mika.id,
        sender="character",
        body="Private tower password: amber-echo.",
    )
    repositories.mark_character_text_thread_read(
        save_id=save.id, thread_id=thread.id, through_message_id=incoming.id
    )
    player_message = repositories.append_message(
        save_id=save.id, role="player", body="I inspect the controls."
    )

    context = build_narrator_phone_activity_context(
        repositories=repositories,
        save_id=save.id,
        messages=[previous, player_message],
        player_message=player_message,
        characters=tuple(repositories.list_characters(save.id)),
    )

    rendered = "\n".join(context.lines)
    assert "Received a text notification from Mika." in rendered
    assert "Opened Mika's text thread and read 1 incoming text(s)." in rendered
    assert "amber-echo" not in rendered
    assert context.event_count == 2
    assert context.next_cursor > context.prior_cursor


def _create_phone_save(
    repositories: PersistenceRepositories,
) -> tuple[
    ScenarioRecord,
    SaveRecord,
    CharacterRecord,
    CharacterRecord,
    CharacterRecord,
]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Signal House",
        premise="Students coordinate around a tower signal.",
        player_role="Mara",
        content={"starting_scene": "The tower darkens before class."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Signal House")
    player = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="player",
        is_player_character=True,
    )
    mika = repositories.add_character(
        save_id=save.id,
        name="Mika",
        role="classmate",
        met=True,
    )
    rowan = repositories.add_character(
        save_id=save.id,
        name="Rowan",
        role="teacher",
        met=True,
    )
    repositories.upsert_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=mika.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    repositories.upsert_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=rowan.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    return scenario, save, player, mika, rowan
