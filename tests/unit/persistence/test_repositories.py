from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from bragi.persistence import repositories as repositories_module
from bragi.persistence.repositories import (
    BragiRepository,
    PersistenceRepositories,
    canonical_claim_fingerprint,
)
from bragi.services.image_style_settings import save_image_style_preset_setting_key
from bragi.services.scenario_evolution_policy import (
    save_scenario_evolution_turn_interval_setting_key,
    scenario_template_evolution_turn_interval_setting_key,
)
from bragi.text_search import cjk_lexical_anchors


@pytest.fixture
def repositories(
    tmp_path: Path,
    migrated_database_template: Path,
) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    shutil.copy2(migrated_database_template, database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_repositories_clear_model_preference(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="scenario_generation_section_worldbuilding",
        provider="openrouter",
        model_id="openrouter/deep-world",
    )

    repositories.clear_model_preference("scenario_generation_section_worldbuilding")

    assert (
        repositories.get_model_preference("scenario_generation_section_worldbuilding")
        is None
    )


def test_message_safety_transition_round_trips_and_edits_clear_it(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Safety Test",
        premise="A neutral test scenario.",
        player_role="Traveler",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Test Save")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Rejected narrator draft.",
        safety_transition="fade_to_black",
        content_rating="r",
    )

    fetched = repositories.get_message(save_id=save.id, message_id=message.id)
    assert fetched is not None
    assert fetched.body == (
        "The intimate moment is kept off-screen. Hours later, "
        "the next scene begins."
    )
    assert fetched.safety_transition == "fade_to_black"
    assert fetched.content_rating == "r"

    edited = repositories.update_message_body(
        save_id=save.id,
        message_id=message.id,
        body="The watch continues.",
        content_rating="g",
    )
    assert edited.safety_transition == ""
    assert edited.content_rating == "g"

    raw_body = "He thrust into her before the scene changed."
    raw = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body=raw_body,
    )
    assert raw.body == raw_body
    assert raw.safety_transition == ""

    edited_body = "Their hands slid beneath her clothes."
    edited_raw = repositories.update_message_body(
        save_id=save.id,
        message_id=raw.id,
        body=edited_body,
    )
    assert edited_raw.body == edited_body
    assert edited_raw.safety_transition == ""


def test_repositories_update_save_title_trims_and_refreshes_list_order(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    opened_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Opened Watch",
    )
    renamed_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Old Watch",
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, last_opened_at = ? WHERE id = ?",
        ("2000-01-02 00:00:00", "2026-05-02 00:00:00", opened_save.id),
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, last_opened_at = ? WHERE id = ?",
        ("2000-01-01 00:00:00", "2026-05-01 00:00:00", renamed_save.id),
    )
    repositories.commit()

    updated = repositories.update_save_title(
        save_id=renamed_save.id,
        title="  Dawn Watch  ",
    )

    assert updated.title == "Dawn Watch"
    fetched = repositories.get_save(renamed_save.id)
    assert fetched is not None
    assert fetched.title == "Dawn Watch"
    assert [save.id for save in repositories.list_saves()] == [
        renamed_save.id,
        opened_save.id,
    ]


def test_repositories_list_save_and_scenario_metadata_for_library(
    repositories: PersistenceRepositories,
) -> None:
    ashfall = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    harbor = repositories.create_scenario(
        type="character_interaction",
        title="Glass Harbor",
        premise="A drowned harbor rings its bell at low tide.",
        player_role="Harbor warden",
        content={},
    )
    first_save = repositories.create_save(
        scenario_id=ashfall.id,
        title="Night Watch",
    )
    second_save = repositories.create_save(
        scenario_id=ashfall.id,
        title="Signal Tower",
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, last_opened_at = ? WHERE id = ?",
        ("2026-05-01 00:00:00", "2026-05-02 00:00:00", first_save.id),
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, last_opened_at = ? WHERE id = ?",
        ("2026-05-03 00:00:00", "2026-05-01 00:00:00", second_save.id),
    )
    repositories.commit()

    saves = repositories.list_saves()
    scenarios = {scenario.id: scenario for scenario in repositories.list_scenarios()}
    save_counts = repositories.count_saves_by_scenario()

    assert [save.id for save in saves] == [second_save.id, first_save.id]
    assert saves[0].scenario_title == "Ashfall Keep"
    assert saves[0].created_at is not None
    assert saves[0].updated_at is not None
    assert saves[0].last_opened_at == "2026-05-01 00:00:00"
    assert scenarios[harbor.id].created_at is not None
    assert scenarios[harbor.id].updated_at is not None
    assert save_counts == {ashfall.id: 2}


def test_repositories_pages_messages_before_anchor(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    messages = [
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Message {index}",
        )
        for index in range(6)
    ]

    latest = repositories.list_message_page(save.id, limit=3)
    previous = repositories.list_message_page(
        save.id,
        before_message_id=latest.messages[0].id,
        limit=3,
    )

    assert [message.body for message in latest.messages] == [
        "Message 3",
        "Message 4",
        "Message 5",
    ]
    assert latest.has_more_before is True
    assert [message.body for message in previous.messages] == [
        "Message 0",
        "Message 1",
        "Message 2",
    ]
    assert previous.has_more_before is False

    repositories.archive_messages_from(save_id=save.id, message_id=messages[1].id)
    remaining = repositories.list_message_page(save.id, limit=3)
    assert [message.id for message in remaining.messages] == [messages[0].id]
    assert remaining.has_more_before is False


def test_repositories_pages_chat_history_messages_with_sql_filters(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Player",
        body="I trim the lantern wick.",
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Fog climbs the lower stair.",
    )
    system = repositories.append_message(
        save_id=save.id,
        role="system",
        speaker_name="System",
        body="Autosave complete.",
    )
    character = repositories.append_message(
        save_id=save.id,
        role="character",
        speaker_name="Captain Ilyra",
        body="Captain Ilyra checks the signal mirror.",
    )
    second_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The mirror blooms with amber light.",
    )
    repositories.create_media_asset(
        save_id=save.id,
        source_message_id=narrator.id,
        type="image",
        path="media/save-1/narrator.png",
        thumbnail_path=None,
        prompt="Fog on stairs",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    repositories.create_media_asset(
        save_id=save.id,
        source_message_id=narrator.id,
        type="image",
        path="media/save-1/narrator-2.png",
        thumbnail_path=None,
        prompt="Fog on stairs again",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    archived_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=second_narrator.id,
        type="image",
        path="media/save-1/archived.png",
        thumbnail_path=None,
        prompt="Archived image",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    repositories.archive_media_asset(
        save_id=save.id,
        media_asset_id=archived_image.id,
    )
    repositories.archive_message(system.id)

    latest = repositories.list_chat_history_message_page(
        save.id,
        selected_filter="narrator_character",
        limit=2,
    )
    previous = repositories.list_chat_history_message_page(
        save.id,
        selected_filter="narrator_character",
        before_message_id=latest.messages[0].id,
        limit=2,
    )
    image_page = repositories.list_chat_history_message_page(
        save.id,
        selected_filter="with_images",
        limit=10,
    )

    assert repositories.count_chat_history_messages(save.id) == 4
    assert (
        repositories.count_chat_history_messages(
            save.id,
            selected_filter="narrator_character",
        )
        == 3
    )
    assert (
        repositories.count_chat_history_messages(
            save.id,
            selected_filter="with_images",
        )
        == 1
    )
    assert [message.id for message in latest.messages] == [
        character.id,
        second_narrator.id,
    ]
    assert latest.has_more_before is True
    assert [message.id for message in previous.messages] == [narrator.id]
    assert previous.has_more_before is False
    assert [message.id for message in image_page.messages] == [narrator.id]
    assert repositories.image_counts_for_messages(
        save_id=save.id,
        message_ids=[player.id, narrator.id, second_narrator.id],
    ) == {narrator.id: 2}


def test_repositories_filters_message_revision_metadata_by_message_ids(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    visible = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens cracks.",
    )
    hidden = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lower gate opens.",
    )
    other_save = repositories.create_save(scenario_id=scenario.id, title="Dawn Watch")
    other_message = repositories.append_message(
        save_id=other_save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The other tower watches.",
    )
    repositories.add_message_revision(
        save_id=save.id,
        message_id=visible.id,
        previous_body="The beacon lens holds.",
        new_body=visible.body,
        diff_unified="diff",
        reconciliation_status="succeeded",
    )
    repositories.add_message_revision(
        save_id=save.id,
        message_id=hidden.id,
        previous_body="The lower gate stays shut.",
        new_body=hidden.body,
        diff_unified="hidden diff",
        reconciliation_status="succeeded",
    )
    repositories.add_message_revision(
        save_id=other_save.id,
        message_id=other_message.id,
        previous_body="The other tower sleeps.",
        new_body=other_message.body,
        diff_unified="other diff",
        reconciliation_status="succeeded",
    )

    metadata = repositories.message_revision_metadata_for_messages(
        save.id,
        [visible.id, other_message.id],
    )

    assert set(metadata) == {visible.id}
    assert metadata[visible.id].revision_count == 1
    assert metadata[visible.id].edited_at is not None
    assert repositories.message_revision_metadata_for_messages(save.id, []) == {}


def test_repositories_round_trip_scene_snapshot_world_time(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")

    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Monday evening",
        time_of_day="evening",
        day_of_week="monday",
        world_day_index=2,
        world_time_day_index=2,
        world_time_day_label="monday",
        world_time_phase="evening",
        world_time_clock_minutes=21 * 60,
        world_time_period_label="festival week",
        world_time_source_message_id=None,
        world_time_confidence=0.92,
        locked_fields=["time_of_day"],
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "Monday evening"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "monday"
    assert snapshot.world_day_index == 2
    assert snapshot.world_time_day_index == 2
    assert snapshot.world_time_day_label == "monday"
    assert snapshot.world_time_phase == "evening"
    assert snapshot.world_time_clock_minutes == 21 * 60
    assert snapshot.world_time_period_label == "festival week"
    assert snapshot.world_time_source_message_id is None
    assert snapshot.world_time_confidence == 0.92
    assert snapshot.locked_fields == ["time_of_day"]


def test_repositories_preserve_canonical_world_time_on_unrelated_snapshot_update(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")

    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Monday evening",
        time_of_day="evening",
        day_of_week="monday",
        world_day_index=2,
        world_time_day_index=2,
        world_time_day_label="monday",
        world_time_phase="evening",
        world_time_clock_minutes=21 * 60,
        world_time_period_label="festival week",
        world_time_confidence=0.92,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The beacon lens hums.",
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.situation == "The beacon lens hums."
    assert snapshot.in_world_time == "Monday evening"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "monday"
    assert snapshot.world_day_index == 2
    assert snapshot.world_time_day_index == 2
    assert snapshot.world_time_day_label == "monday"
    assert snapshot.world_time_phase == "evening"
    assert snapshot.world_time_clock_minutes == 21 * 60
    assert snapshot.world_time_period_label == "festival week"
    assert snapshot.world_time_confidence == 0.92

    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Tuesday morning",
        time_of_day="morning",
        day_of_week="tuesday",
        world_day_index=3,
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "Tuesday morning"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "tuesday"
    assert snapshot.world_day_index == 3
    assert snapshot.world_time_day_index == 3
    assert snapshot.world_time_day_label == "tuesday"
    assert snapshot.world_time_phase == "morning"
    assert snapshot.world_time_clock_minutes == 21 * 60
    assert snapshot.world_time_period_label == "festival week"
    assert snapshot.world_time_confidence == 0.92


def test_repositories_seed_canonical_world_time_without_rewriting_legacy_label(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Oracle of Glass",
        premise="A mirrored chamber.",
        player_role="Petitioner",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Oracle Visit")

    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Friday evening after class",
        time_of_day="evening",
        day_of_week="friday",
        world_day_index=5,
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "Friday evening after class"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "friday"
    assert snapshot.world_day_index == 5
    assert snapshot.world_time_day_index == 5
    assert snapshot.world_time_day_label == "friday"
    assert snapshot.world_time_phase == "evening"


def test_repositories_round_trip_dating_route_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Ren Takahashi",
        is_player_character=True,
        met=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        relationships={player.name: "romance option for Ren Takahashi"},
        met=True,
    )
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika gives Ren her number before leaving the festival gate.",
    )

    created = repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="contact_exchanged",
        first_met_message_id=message.id,
        first_met_world_day_index=0,
        last_interaction_message_id=message.id,
        last_interaction_world_day_index=2,
        completed_interactions=1,
        dates_completed=0,
        interest_level="curious",
        trust_level="guarded",
        comfort_with_intimacy="none",
        pacing_preference="slow_burn",
        known_boundaries=["no instant commitment"],
        unresolved_questions=["why Ren helped"],
        next_reasonable_step="schedule a first date",
        source_message_id=message.id,
        route_id="route-mika",
    )
    updated = repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="first_date_planned",
        completed_interactions=2,
        dates_completed=0,
        route_id="ignored-on-conflict",
    )

    assert created.id == "route-mika"
    assert updated.id == "route-mika"
    assert updated.stage == "first_date_planned"
    assert updated.completed_interactions == 2
    assert updated.known_boundaries == ["no instant commitment"]
    assert updated.unresolved_questions == ["why Ren helped"]
    assert updated.next_reasonable_step == "schedule a first date"
    assert repositories.get_dating_route_state(created.id) == updated
    assert repositories.get_dating_route_state_for_pair(
        save.id,
        player.id,
        npc.id,
    ) == updated
    assert repositories.list_dating_route_states(save.id) == [updated]


def test_repositories_round_trip_asymmetric_character_contact_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Ren Takahashi",
        is_player_character=True,
        met=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        met=True,
    )
    source = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Ren gives Mika his number.",
    )

    character_only = repositories.upsert_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=npc.id,
        character_has_player_number=True,
        source_message_id=source.id,
        state_id="contact-mika",
    )
    both = repositories.upsert_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        state_id="ignored-on-conflict",
    )

    assert character_only.id == "contact-mika"
    assert both.id == "contact-mika"
    assert both.player_has_character_number is True
    assert both.character_has_player_number is True
    assert both.source_message_id == source.id
    assert repositories.get_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=npc.id,
    ) == both
    assert repositories.list_character_contact_states(save.id) == [both]


def test_repositories_set_character_contact_state_can_override_inferred_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Ren Takahashi",
        is_player_character=True,
        met=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        met=True,
    )
    inferred = repositories.upsert_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=True,
        state_id="contact-mika",
    )

    corrected = repositories.set_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=False,
        character_has_player_number=False,
    )

    assert corrected.id == inferred.id
    assert corrected.player_has_character_number is False
    assert corrected.character_has_player_number is False
    assert repositories.character_text_outbound_allowed(
        save_id=save.id,
        character_id=npc.id,
    ) is False
    assert repositories.can_character_proactively_text(
        save_id=save.id,
        character_id=npc.id,
    ) is False

    restored = repositories.set_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=False,
    )

    assert restored.id == inferred.id
    assert restored.player_has_character_number is True
    assert restored.character_has_player_number is False


def test_repositories_recover_interrupted_character_text_deliveries(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    pending = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Can we talk?",
        delivery_status="pending",
        delivery_job_id="job-pending",
    )
    retrying = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Still there?",
        delivery_status="retrying",
        delivery_attempt=2,
    )
    sent = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="This one went through.",
        delivery_status="sent",
    )

    recovered = repositories.recover_interrupted_character_text_deliveries(
        error="Text delivery was interrupted before completion",
    )

    assert [message.id for message in recovered] == [pending.id, retrying.id]
    messages = {
        message.id: message
        for message in repositories.list_character_text_messages(save_id=save.id)
    }
    assert messages[pending.id].delivery_status == "failed"
    assert messages[retrying.id].delivery_status == "failed"
    assert messages[sent.id].delivery_status == "sent"
    assert (
        messages[pending.id].delivery_error
        == "Text delivery was interrupted before completion"
    )
    assert repositories.has_active_character_text_delivery(
        save_id=save.id,
        thread_id=thread.id,
    ) is False


def test_repositories_create_group_text_thread_with_multiple_participants(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Mira",
        is_player_character=True,
    )
    rowan = repositories.add_character(save_id=save.id, name="Rowan", met=True)
    maya = repositories.add_character(save_id=save.id, name="Maya", met=True)
    direct = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=rowan.id,
        title=rowan.name,
    )

    group = repositories.create_character_text_group_thread(
        save_id=save.id,
        title="Arcade Crew",
        character_ids=[rowan.id, maya.id],
    )
    player_message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=group.id,
        character_id=None,
        sender="player",
        sender_character_id=player.id,
        body="Can everyone help tonight?",
    )
    rowan_reply = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=group.id,
        character_id=rowan.id,
        sender="character",
        sender_character_id=rowan.id,
        body="I can bring tokens.",
    )
    maya_reply = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=group.id,
        character_id=maya.id,
        sender="character",
        sender_character_id=maya.id,
        body="I can cover the desk.",
    )

    assert direct.kind == "direct"
    assert group.kind == "group"
    assert group.character_id is None
    assert [
        participant.character_id
        for participant in repositories.list_character_text_thread_participants(
            save_id=save.id,
            thread_id=group.id,
        )
    ] == [rowan.id, maya.id]
    messages = repositories.list_character_text_messages(
        save_id=save.id,
        thread_id=group.id,
    )
    assert [message.id for message in messages] == [
        player_message.id,
        rowan_reply.id,
        maya_reply.id,
    ]
    assert [message.sender_character_id for message in messages] == [
        player.id,
        rowan.id,
        maya.id,
    ]


def test_repositories_track_character_text_message_metadata(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    pending = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Can we talk?",
        delivery_status="pending",
        in_world_sent_at="Friday evening after class",
    )
    delivered = repositories.update_character_text_delivery(
        save_id=save.id,
        message_id=pending.id,
        status="sent",
        error=None,
    )
    reply = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Meet me by the lockers.",
        reply_to_message_id=delivered.id,
        read_at="2026-07-01T12:07:00+00:00",
    )

    messages = repositories.list_character_text_messages(
        save_id=save.id,
        thread_id=thread.id,
    )

    assert messages[0].id == delivered.id
    assert messages[0].in_world_sent_at == "Friday evening after class"
    assert messages[0].delivered_at is not None
    assert messages[0].read_at is None
    assert messages[0].reply_to_message_id is None
    assert messages[1].id == reply.id
    assert messages[1].reply_to_message_id == delivered.id
    assert messages[1].delivered_at is not None
    assert messages[1].read_at == "2026-07-01T12:07:00+00:00"


def test_repositories_mark_character_text_thread_read_through_message(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    first = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="First unread.",
        message_id="text-first",
    )
    player = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="On my way.",
        message_id="text-player",
    )
    second = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Second unread.",
        message_id="text-second",
    )
    pending = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Still sending.",
        message_id="text-pending",
        delivery_status="pending",
    )
    failed = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Failed send.",
        message_id="text-failed",
        delivery_status="failed",
    )
    deleted = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Deleted send.",
        message_id="text-deleted",
    )
    repositories.archive_character_text_messages_from(
        save_id=save.id,
        thread_id=thread.id,
        message_id=deleted.id,
    )

    marked = repositories.mark_character_text_thread_read(
        save_id=save.id,
        thread_id=thread.id,
        through_message_id=player.id,
    )

    assert [message.id for message in marked] == [first.id]
    messages = {
        message.id: message
        for message in repositories.list_character_text_messages(
            save_id=save.id,
            thread_id=thread.id,
        )
    }
    assert messages[first.id].read_at is not None
    assert messages[player.id].read_at is None
    assert messages[second.id].read_at is None
    assert messages[pending.id].read_at is None
    assert messages[failed.id].read_at is None

    marked_again = repositories.mark_character_text_thread_read(
        save_id=save.id,
        thread_id=thread.id,
        through_message_id=player.id,
    )

    assert marked_again == ()


def test_repositories_mark_character_text_thread_read_marks_all_without_boundary(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    first = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="First unread.",
    )
    second = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Second unread.",
    )

    marked = repositories.mark_character_text_thread_read(
        save_id=save.id,
        thread_id=thread.id,
    )

    assert [message.id for message in marked] == [first.id, second.id]
    assert all(message.read_at is not None for message in marked)


def test_repositories_record_body_free_character_text_activity(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer.",
        player_role="Student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id, character_id=npc.id, title=npc.name
    )
    incoming = repositories.append_character_text_message(
        save_id=save.id, thread_id=thread.id, character_id=npc.id,
        sender="character", body="Private body must not enter activity.",
    )
    repositories.mark_character_text_thread_read(
        save_id=save.id, thread_id=thread.id, through_message_id=incoming.id
    )

    events = repositories.list_character_text_activity_events_after(
        save_id=save.id, ordinal=0, limit=10
    )

    assert [(event.activity_type, event.read_count) for event in events] == [
        ("character_received", 0),
        ("thread_opened", 1),
    ]
    assert all("Private body" not in repr(event) for event in events)


def test_repositories_track_character_text_message_revisions(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Can we tak?",
    )

    updated = repositories.update_character_text_message_body(
        save_id=save.id,
        message_id=message.id,
        body="Can we talk?",
    )
    first = repositories.add_character_text_message_revision(
        save_id=save.id,
        text_message_id=message.id,
        previous_body=message.body,
        new_body=updated.body,
        diff_unified="--- previous\n+++ current\n",
        reconciliation_status="skipped",
    )
    second = repositories.add_character_text_message_revision(
        save_id=save.id,
        text_message_id=message.id,
        previous_body=updated.body,
        new_body="Can we talk after class?",
        diff_unified="--- previous\n+++ current\n",
        reconciliation_status="queued",
    )

    revisions = repositories.list_character_text_message_revisions(
        save_id=save.id,
        text_message_id=message.id,
    )
    metadata = repositories.character_text_message_revision_metadata(save.id)

    assert updated.body == "Can we talk?"
    assert [revision.revision_number for revision in revisions] == [1, 2]
    assert first.reconciled_at is not None
    assert second.reconciled_at is None
    assert metadata[message.id].revision_count == 2
    assert metadata[message.id].edited_at is not None


def test_repositories_track_character_text_message_attachments(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="I found the ticket stub.",
    )
    media = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/text-ticket.png",
        thumbnail_path="media/text-ticket.thumb.png",
        prompt="close-up of a ticket stub on a phone",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={
            "kind": "character_text_object_context_image",
            "text_message_id": message.id,
        },
    )

    succeeded = repositories.add_character_text_message_attachment(
        save_id=save.id,
        thread_id=thread.id,
        text_message_id=message.id,
        character_id=npc.id,
        kind="object_context_image",
        status="succeeded",
        media_asset_id=media.id,
        prompt="close-up of a ticket stub on a phone",
        metadata={"decision_reason": "ticket mentioned"},
    )
    failed = repositories.add_character_text_message_attachment(
        save_id=save.id,
        thread_id=thread.id,
        text_message_id=message.id,
        character_id=npc.id,
        kind="character_image",
        status="failed",
        error="provider failed with secret token",
        prompt="selfie in a red jacket",
        ordinal=1,
    )

    attachments = repositories.list_character_text_message_attachments(
        save_id=save.id,
        text_message_ids=(message.id,),
    )

    assert [attachment.id for attachment in attachments] == [succeeded.id, failed.id]
    assert attachments[0].media_asset_id == media.id
    assert attachments[0].metadata_json == '{"decision_reason":"ticket mentioned"}'
    assert attachments[1].status == "failed"
    assert attachments[1].error == "provider failed with secret token"
    assert attachments[1].ordinal == 1


def test_repositories_allow_uploaded_photo_attachment_for_message_sender(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Mira",
        role="player",
        is_player_character=True,
    )
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        sender_character_id=player.id,
        body="Do you recognize this symbol?",
    )
    media = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/text-symbol.png",
        thumbnail_path=None,
        prompt="Uploaded text photo",
        provider="local",
        model="upload",
        status="succeeded",
        metadata={
            "kind": "character_text_uploaded_photo",
            "text_message_id": message.id,
        },
    )

    attachment = repositories.add_character_text_message_attachment(
        save_id=save.id,
        thread_id=thread.id,
        text_message_id=message.id,
        character_id=player.id,
        kind="uploaded_photo",
        status="succeeded",
        media_asset_id=media.id,
        prompt="A blue symbol painted on a tile.",
    )

    rows = repositories.list_character_text_message_attachments(
        save_id=save.id,
        text_message_ids=(message.id,),
    )
    assert rows == [attachment]
    assert rows[0].character_id == player.id
    assert rows[0].kind == "uploaded_photo"


def test_repositories_replace_character_text_attachment_media_asset(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="I found the ticket stub.",
    )
    old_media = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/text-ticket.png",
        thumbnail_path=None,
        prompt="close-up of a ticket stub on a phone",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    new_media = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/text-ticket-new.png",
        thumbnail_path=None,
        prompt="edited close-up of a ticket stub",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    attachment = repositories.add_character_text_message_attachment(
        save_id=save.id,
        thread_id=thread.id,
        text_message_id=message.id,
        character_id=npc.id,
        kind="object_context_image",
        status="succeeded",
        media_asset_id=old_media.id,
        prompt="close-up of a ticket stub on a phone",
        metadata={"media_asset_id": old_media.id, "decision_reason": "ticket"},
    )

    replaced = repositories.replace_character_text_attachment_media_asset(
        save_id=save.id,
        old_media_asset_id=old_media.id,
        new_media_asset_id=new_media.id,
    )

    rows = repositories.list_character_text_message_attachments(
        save_id=save.id,
        text_message_ids=(message.id,),
    )
    assert replaced == 1
    assert rows[0].id == attachment.id
    assert rows[0].media_asset_id == new_media.id
    assert rows[0].metadata_json == (
        f'{{"decision_reason":"ticket","media_asset_id":"{new_media.id}"}}'
    )
    assert rows[0].updated_at is not None


def test_repositories_archive_and_restore_character_text_messages_after_anchor(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    anchor = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="First.",
    )
    reply = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Reply.",
    )
    later = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Later.",
    )

    archived = repositories.archive_character_text_messages_after(
        save_id=save.id,
        thread_id=thread.id,
        message_id=anchor.id,
    )

    assert [message.id for message in archived] == [reply.id, later.id]
    assert [
        message.id
        for message in repositories.list_character_text_messages(save_id=save.id)
    ] == [anchor.id]
    repositories.restore_character_text_messages({message.id for message in archived})
    assert [
        message.id
        for message in repositories.list_character_text_messages(save_id=save.id)
    ] == [anchor.id, reply.id, later.id]


def test_repositories_archive_and_restore_character_text_messages_from_anchor(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    first = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="First.",
    )
    selected = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Reply.",
    )
    later = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Later.",
    )

    archived = repositories.archive_character_text_messages_from(
        save_id=save.id,
        thread_id=thread.id,
        message_id=selected.id,
    )

    assert [message.id for message in archived] == [selected.id, later.id]
    assert [
        message.id
        for message in repositories.list_character_text_messages(save_id=save.id)
    ] == [first.id]
    repositories.restore_character_text_messages({message.id for message in archived})
    assert [
        message.id
        for message in repositories.list_character_text_messages(save_id=save.id)
    ] == [first.id, selected.id, later.id]


def test_repositories_append_message_refreshes_save_updated_order(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    stale_active_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
    )
    newer_metadata_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Signal Tower",
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ? WHERE id = ?",
        ("2000-01-01 00:00:00", stale_active_save.id),
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ? WHERE id = ?",
        ("2000-01-02 00:00:00", newer_metadata_save.id),
    )
    repositories.commit()

    repositories.append_message(
        save_id=stale_active_save.id,
        role="player",
        body="I check the beacon lens.",
    )

    saves = repositories.list_saves()

    assert [save.id for save in saves] == [
        stale_active_save.id,
        newer_metadata_save.id,
    ]
    assert saves[0].updated_at is not None
    assert saves[0].updated_at > "2000-01-02 00:00:00"


def test_repositories_append_message_preserves_imported_timestamps_without_save_touch(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ? WHERE id = ?",
        ("2026-01-02 03:04:05", save.id),
    )
    repositories.commit()

    message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I check the old beacon log.",
        created_at="2001-02-03 04:05:06.007",
        updated_at="2001-02-03 04:05:07.008",
        touch_save_updated_at=False,
    )

    fetched_save = repositories.get_save(save.id)
    assert fetched_save is not None
    assert fetched_save.updated_at == "2026-01-02 03:04:05"
    assert message.created_at == "2001-02-03 04:05:06.007"
    assert message.updated_at == "2001-02-03 04:05:07.008"


def test_repositories_store_latest_active_message_action_choices(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Library of Falling Doors",
        premise="Every shelf is a door.",
        player_role="Courier",
        content={"action_choices_enabled": True},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Library")
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The first shelf opens.",
    )
    second = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The second shelf opens.",
    )

    repositories.replace_message_action_choices(
        save_id=save.id,
        message_id=first.id,
        choices=("Wait", "Run", "Ask", "Touch"),
        provider="fake",
        model="fake-chat",
    )
    repositories.replace_message_action_choices(
        save_id=save.id,
        message_id=second.id,
        choices=("Climb", "Read", "Listen", "Knock"),
        provider="fake",
        model="fake-chat",
    )

    assert [
        choice.body for choice in repositories.latest_message_action_choices(save.id)
    ] == ["Climb", "Read", "Listen", "Knock"]

    repositories.archive_message(second.id)

    assert [
        choice.body for choice in repositories.latest_message_action_choices(save.id)
    ] == ["Wait", "Run", "Ask", "Touch"]
    assert [
        choice.body for choice in repositories.list_message_action_choices(save.id)
    ] == ["Wait", "Run", "Ask", "Touch", "Climb", "Read", "Listen", "Knock"]


def test_repositories_find_active_message_after_marker(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Library of Falling Doors",
        premise="Every shelf is a door.",
        player_role="Courier",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Library")
    other_save = repositories.create_save(scenario_id=scenario.id, title="Annex")
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Courier",
        body="Light the beacon",
    )
    marker = repositories.latest_active_message_rowid(save.id)
    repositories.append_message(
        save_id=other_save.id,
        role="player",
        speaker_name="Courier",
        body="Light the beacon",
    )
    deleted = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Courier",
        body="Light the beacon",
    )
    repositories.archive_message(deleted.id)
    committed = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Courier",
        body="Light the beacon",
    )

    found = repositories.find_active_message_after_rowid(
        save.id,
        after_rowid=marker,
        role="player",
        body="Light the beacon",
        speaker_name="Courier",
    )

    assert found == committed
    assert repositories.find_active_message_after_rowid(
        save.id,
        after_rowid=marker,
        role="narrator",
        body="Light the beacon",
        speaker_name="Courier",
    ) is None


def test_repositories_resolve_effective_scoped_setting_precedence(
    repositories: PersistenceRepositories,
) -> None:
    user = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"starting_location": "Gatehouse"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    key = "image_generation_frequency"

    repositories.set_scoped_setting(scope="global", key=key, value=1)
    repositories.set_scoped_setting(
        scope="user",
        scope_id=user.id,
        key=key,
        value=2,
    )
    repositories.set_scoped_setting(
        scope="scenario",
        scope_id=scenario.id,
        key=key,
        value=3,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=key,
        value=4,
    )

    assert (
        repositories.get_effective_setting(
            key,
            save_id=save.id,
            user_id=user.id,
        )
        == 4
    )

    repositories.delete_scoped_setting(scope="save", scope_id=save.id, key=key)
    assert (
        repositories.get_effective_setting(
            key,
            save_id=save.id,
            user_id=user.id,
        )
        == 3
    )

    repositories.delete_scoped_setting(scope="scenario", scope_id=scenario.id, key=key)
    assert (
        repositories.get_effective_setting(
            key,
            save_id=save.id,
            user_id=user.id,
        )
        == 2
    )

    repositories.delete_scoped_setting(scope="user", scope_id=user.id, key=key)
    assert repositories.get_effective_setting(key, save_id=save.id) == 1


def test_repositories_persist_context_observations_with_source_evidence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I mark the eastern signal code in my notebook.",
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The eastern code will matter if the riders return.",
    )

    observation = repositories.add_context_observation(
        observation_id="observation-signal-code",
        save_id=save.id,
        observation_type="open_thread",
        claim="The eastern signal code may matter later.",
        evidence_quote="The eastern code will matter",
        source_message_ids=[narrator.id, player.id, narrator.id],
        scope="save",
        confidence=0.86,
        tags=["signals", "future-risk", "signals"],
        metadata={"observer": "cheap"},
    )
    updated = repositories.update_context_observation(
        observation.id,
        status="accepted",
        metadata={"curation_action": "save_context"},
    )

    observations = repositories.list_context_observations(
        save.id,
        statuses=("accepted",),
    )
    assert observations == [updated]
    assert updated.source_message_ids == [narrator.id, player.id]
    assert updated.tags == ["signals", "future-risk"]
    assert updated.metadata == {
        "observer": "cheap",
        "curation_action": "save_context",
    }


def test_repositories_list_messages_by_ids_is_save_scoped(
    repositories: PersistenceRepositories,
) -> None:
    first_scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    first_save = repositories.create_save(
        scenario_id=first_scenario.id,
        title="Night Watch",
    )
    first_message = repositories.append_message(
        save_id=first_save.id,
        role="player",
        speaker_name="Mara",
        body="I mark the eastern signal code in my notebook.",
    )
    second_save = repositories.create_save(
        scenario_id=first_scenario.id,
        title="Dawn Watch",
    )
    second_message = repositories.append_message(
        save_id=second_save.id,
        role="player",
        speaker_name="Tarin",
        body="I inspect the western gate.",
    )

    messages = repositories.list_messages_by_ids(
        first_save.id,
        (first_message.id, second_message.id, first_message.id),
    )

    assert messages == [first_message]


def test_repositories_claim_context_observations_in_bounded_fifo_batches(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    observations = [
        repositories.add_context_observation(
            save_id=save.id,
            observation_type="scene_fact",
            claim=f"Observation {index}",
        )
        for index in range(3)
    ]

    eligible = repositories.list_eligible_context_observations(save.id, limit=2)
    claimed = repositories.claim_context_observations(
        [observation.id for observation in eligible],
        lease_token="worker-one",
        lease_seconds=600,
    )

    assert [observation.id for observation in claimed] == [
        observations[0].id,
        observations[1].id,
    ]
    assert [
        observation.id
        for observation in repositories.list_eligible_context_observations(
            save.id,
            limit=3,
        )
    ] == [observations[2].id]
    state = repositories.get_context_observation_curation_state(
        observations[0].id
    )
    assert state is not None
    assert state.attempt_count == 1
    assert state.lease_token == "worker-one"
    assert state.lease_until is not None


def test_repositories_normalize_observation_type_and_preserve_original(
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

    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="npc intent",
        claim="Lio intends to inspect the lens.",
        source_message_ids=[],
    )

    assert observation.observation_type == "character_intent"
    assert observation.metadata["original_observation_type"] == "npc intent"


def test_repositories_fence_stale_context_observation_curation_worker(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="scene_fact",
        claim="The eastern signal is lit.",
    )
    repositories.claim_context_observations(
        [observation.id],
        lease_token="current-worker",
        lease_seconds=600,
    )

    assert (
        repositories.complete_context_observation_curation(
            observation.id,
            lease_token="stale-worker",
            status="accepted",
            terminal_outcome="accepted",
        )
        is None
    )
    current = repositories.get_context_observation(observation.id)
    assert current is not None
    assert current.status == "pending"

    completed = repositories.complete_context_observation_curation(
        observation.id,
        lease_token="current-worker",
        status="accepted",
        terminal_outcome="accepted",
    )

    assert completed is not None
    assert completed.status == "accepted"
    state = repositories.get_context_observation_curation_state(observation.id)
    assert state is not None
    assert state.terminal_outcome == "accepted"
    assert state.lease_token is None


def test_context_observation_claims_are_exclusive_across_connections(
    tmp_path: Path,
    migrated_database_template: Path,
) -> None:
    database_path = tmp_path / "claims.sqlite3"
    shutil.copy2(migrated_database_template, database_path)
    connections = [
        sqlite3.connect(database_path, timeout=5, check_same_thread=False)
        for _ in range(2)
    ]
    repositories = [PersistenceRepositories(connection) for connection in connections]
    try:
        scenario = repositories[0].create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A border keep is cut off by ash storms.",
            player_role="Warden",
            content={},
        )
        save = repositories[0].create_save(
            scenario_id=scenario.id,
            title="Night Watch",
        )
        observations = [
            repositories[0].add_context_observation(
                save_id=save.id,
                observation_type="event",
                claim=f"Observation {index}",
            )
            for index in range(4)
        ]
        observation_ids = [observation.id for observation in observations]
        barrier = threading.Barrier(2)
        claimed_by_worker: list[list[str]] = [[], []]
        claim_errors: list[BaseException] = []

        def claim(worker_index: int) -> None:
            try:
                barrier.wait()
                claimed_by_worker[worker_index] = [
                    observation.id
                    for observation in repositories[
                        worker_index
                    ].claim_context_observations(
                        observation_ids,
                        lease_token=f"worker-{worker_index}",
                        lease_seconds=600,
                    )
                ]
            except BaseException as exc:  # pragma: no cover - asserted in parent
                claim_errors.append(exc)

        threads = [threading.Thread(target=claim, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert all(not thread.is_alive() for thread in threads)
        assert claim_errors == []
        assert set(claimed_by_worker[0]).isdisjoint(claimed_by_worker[1])
        assert sorted(claimed_by_worker[0] + claimed_by_worker[1]) == sorted(
            observation_ids
        )
    finally:
        for connection in connections:
            connection.close()


def test_memory_dedup_is_atomic_across_connections(
    tmp_path: Path,
    migrated_database_template: Path,
) -> None:
    database_path = tmp_path / "memory-dedup.sqlite3"
    shutil.copy2(migrated_database_template, database_path)
    connections = [
        sqlite3.connect(database_path, timeout=5, check_same_thread=False)
        for _ in range(2)
    ]
    repositories = [PersistenceRepositories(connection) for connection in connections]
    try:
        scenario = repositories[0].create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A border keep is cut off by ash storms.",
            player_role="Warden",
            content={},
        )
        save = repositories[0].create_save(
            scenario_id=scenario.id,
            title="Night Watch",
        )
        barrier = threading.Barrier(2)
        created_ids: list[str | None] = [None, None]
        errors: list[BaseException] = []

        def add_memory(worker_index: int) -> None:
            try:
                barrier.wait()
                record = repositories[worker_index].add_memory(
                    save_id=save.id,
                    body="The moonstone opens the eastern vault.",
                    tags=[f"worker-{worker_index}"],
                    source_message_ids=[f"message-{worker_index}"],
                )
                created_ids[worker_index] = record.id
            except BaseException as exc:  # pragma: no cover - asserted in parent
                errors.append(exc)

        threads = [
            threading.Thread(target=add_memory, args=(worker_index,))
            for worker_index in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert created_ids[0] == created_ids[1]
        [memory] = repositories[0].list_memories(save.id)
        assert set(memory.tags) == {"worker-0", "worker-1"}
        assert set(memory.source_message_ids) == {"message-0", "message-1"}
    finally:
        for connection in connections:
            connection.close()


def test_expired_context_observation_lease_cannot_complete_or_defer(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="event",
        claim="The eastern signal is lit.",
    )
    repositories.claim_context_observations(
        (observation.id,),
        lease_token="expired-worker",
        lease_seconds=600,
    )
    repositories.connection.execute(
        """
        UPDATE context_observation_curation_state
        SET lease_until = '2000-01-01 00:00:00'
        WHERE observation_id = ?
        """,
        (observation.id,),
    )
    repositories.commit()

    assert not repositories.owns_context_observation_curation_lease(
        observation.id,
        lease_token="expired-worker",
    )
    assert (
        repositories.complete_context_observation_curation(
            observation.id,
            lease_token="expired-worker",
            status="accepted",
            terminal_outcome="accepted",
        )
        is None
    )
    assert (
        repositories.defer_context_observation_curation(
            observation.id,
            lease_token="expired-worker",
            error="too late",
            retry_after_seconds=60,
            max_attempts=5,
        )
        is None
    )
    reclaimed = repositories.claim_context_observations(
        (observation.id,),
        lease_token="replacement-worker",
        lease_seconds=600,
    )
    assert [row.id for row in reclaimed] == [observation.id]


def test_repeated_expired_context_observation_leases_exhaust_retry_budget(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="event",
        claim="The eastern signal is lit.",
    )

    for attempt in range(5):
        claimed = repositories.claim_context_observations(
            (observation.id,),
            lease_token=f"worker-{attempt}",
            lease_seconds=600,
            max_attempts=5,
        )
        assert [row.id for row in claimed] == [observation.id]
        repositories.connection.execute(
            """
            UPDATE context_observation_curation_state
            SET lease_until = '2000-01-01 00:00:00'
            WHERE observation_id = ?
            """,
            (observation.id,),
        )
        repositories.commit()

    assert repositories.claim_context_observations(
        (observation.id,),
        lease_token="worker-over-budget",
        lease_seconds=600,
        max_attempts=5,
    ) == []
    state = repositories.get_context_observation_curation_state(observation.id)
    assert state is not None
    assert state.attempt_count == 5
    assert state.terminal_outcome == "retry_budget_exhausted"
    updated = repositories.get_context_observation(observation.id)
    assert updated is not None
    assert updated.status == "curation_failed"


def test_cancellation_release_cannot_clear_replacement_worker_lease(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="event",
        claim="The eastern signal is lit.",
    )
    repositories.claim_context_observations(
        (observation.id,),
        lease_token="cancelled-worker",
        lease_seconds=600,
    )
    repositories.connection.execute(
        """
        UPDATE context_observation_curation_state
        SET lease_until = '2000-01-01 00:00:00'
        WHERE observation_id = ?
        """,
        (observation.id,),
    )
    repositories.commit()
    repositories.claim_context_observations(
        (observation.id,),
        lease_token="replacement-worker",
        lease_seconds=600,
    )

    released = repositories.release_context_observation_curation_claims(
        (observation.id,),
        lease_token="cancelled-worker",
        error="cancelled",
    )

    assert released == 0
    state = repositories.get_context_observation_curation_state(observation.id)
    assert state is not None
    assert state.lease_token == "replacement-worker"


def test_repositories_archive_context_observations_for_deleted_messages(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    kept = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I keep watch.",
    )
    deleted = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A detail that will be deleted.",
    )
    portable = repositories.add_context_observation(
        save_id=save.id,
        observation_type="scene_fact",
        claim="Mara keeps watch.",
        source_message_ids=[kept.id],
    )
    stale = repositories.add_context_observation(
        save_id=save.id,
        observation_type="scene_fact",
        claim="Deleted detail.",
        source_message_ids=[deleted.id],
    )

    archived = repositories.archive_context_observations_for_deleted_messages(
        save_id=save.id,
        message_ids=frozenset({deleted.id}),
    )

    assert archived == frozenset({stale.id})
    observation_ids = [
        record.id for record in repositories.list_context_observations(save.id)
    ]
    assert observation_ids == [portable.id]


def test_repositories_map_legacy_app_setting_keys_to_scoped_settings(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"starting_location": "Gatehouse"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")

    repositories.set_app_setting(save_image_style_preset_setting_key(save.id), "ink")
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key="image_style_preset",
        value="pixel_art",
    )

    records = repositories.list_scoped_settings(
        scope="save",
        scope_id=save.id,
    )
    assert records[0].value == "pixel_art"
    assert (
        repositories.get_app_setting(save_image_style_preset_setting_key(save.id))
        == "pixel_art"
    )


def test_repositories_copy_save_scoped_settings_between_saves(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"starting_location": "Gatehouse"},
    )
    source_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
    )
    target_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Dawn Watch",
    )
    repositories.set_scoped_setting(
        scope="global",
        key="image_generation_frequency",
        value=1,
    )
    repositories.set_scoped_setting(
        scope="scenario",
        scope_id=scenario.id,
        key="image_generation_frequency",
        value=2,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=source_save.id,
        key="image_generation_frequency",
        value=3,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=source_save.id,
        key="image_style_preset",
        value="watercolor",
    )

    repositories.copy_save_scoped_settings(
        source_save_id=source_save.id,
        target_save_id=target_save.id,
    )

    copied = {
        record.key: record.value
        for record in repositories.list_scoped_settings(
            scope="save",
            scope_id=target_save.id,
        )
    }
    assert copied == {
        "image_generation_frequency": 3,
        "image_style_preset": "watercolor",
    }


def test_repositories_create_and_find_users_case_insensitively(
    repositories: PersistenceRepositories,
) -> None:
    user = repositories.create_user(
        username="  Mira ",
        role="admin",
        password_hash="argon2-hash",
    )

    assert user.username == "Mira"
    assert user.username_normalized == "mira"
    assert user.role == "admin"
    assert user.status == "active"
    assert repositories.get_user(user.id) == user
    assert repositories.get_user_by_username("MIRA") == user
    assert repositories.list_users() == [user]

    with pytest.raises(ValueError, match="username already exists"):
        repositories.create_user(
            username="mira",
            role="user",
            password_hash="other-hash",
        )


def test_repositories_validate_user_role_and_status(
    repositories: PersistenceRepositories,
) -> None:
    with pytest.raises(ValueError, match="Unknown user role"):
        repositories.create_user(
            username="mira",
            role="owner",
            password_hash="hash",
        )

    user = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )

    with pytest.raises(ValueError, match="Unknown user status"):
        repositories.update_user_status(user.id, "sleeping")

    disabled = repositories.update_user_status(user.id, "disabled")
    child = repositories.update_user_role(user.id, "child")
    updated_password = repositories.update_user_password_hash(user.id, "new-hash")

    assert disabled.status == "disabled"
    assert child.role == "child"
    assert updated_password.password_hash == "new-hash"


def test_repositories_load_only_active_unexpired_sessions(
    repositories: PersistenceRepositories,
) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    user = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    active = repositories.create_user_session(
        user_id=user.id,
        token_hash="active-token-hash",
        expires_at=now + timedelta(hours=1),
    )
    expired = repositories.create_user_session(
        user_id=user.id,
        token_hash="expired-token-hash",
        expires_at=now - timedelta(seconds=1),
    )
    revoked = repositories.create_user_session(
        user_id=user.id,
        token_hash="revoked-token-hash",
        expires_at=now + timedelta(hours=1),
    )

    repositories.revoke_user_session(revoked.id)

    assert (
        repositories.get_active_user_session_by_token_hash(
            "active-token-hash",
            now=now,
        )
        == active
    )
    assert (
        repositories.get_active_user_session_by_token_hash(
            expired.token_hash,
            now=now,
        )
        is None
    )
    assert (
        repositories.get_active_user_session_by_token_hash(
            revoked.token_hash,
            now=now,
        )
        is None
    )


def test_repositories_revoke_user_sessions_with_optional_exception(
    repositories: PersistenceRepositories,
) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    user = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    kept = repositories.create_user_session(
        user_id=user.id,
        token_hash="keep-token-hash",
        expires_at=now + timedelta(hours=1),
    )
    revoked = repositories.create_user_session(
        user_id=user.id,
        token_hash="revoke-token-hash",
        expires_at=now + timedelta(hours=1),
    )

    assert (
        repositories.revoke_user_sessions(
            user.id,
            except_token_hash=kept.token_hash,
            now=now,
        )
        == 1
    )

    kept_session = repositories.get_user_session(kept.id)
    revoked_session = repositories.get_user_session(revoked.id)
    assert kept_session is not None
    assert revoked_session is not None
    assert kept_session.revoked_at is None
    assert revoked_session.revoked_at == "2026-01-02T00:00:00+00:00"

    assert repositories.revoke_user_sessions(user.id, now=now) == 1
    kept_session = repositories.get_user_session(kept.id)
    assert kept_session is not None
    assert kept_session.revoked_at == "2026-01-02T00:00:00+00:00"


def test_repositories_persist_provider_model_generation_parameters(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/chat",
        display_name="OpenRouter Chat",
        capabilities=["chat"],
        supported_parameters=["temperature", "max_output_tokens"],
        context_window=128_000,
    )

    models = repositories.list_provider_models("openrouter")

    assert len(models) == 1
    assert models[0].supported_parameters == [
        "temperature",
        "max_output_tokens",
    ]


def test_repositories_persist_provider_model_pricing(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/chat",
        display_name="OpenRouter Chat",
        capabilities=["chat"],
        pricing={
            "input_per_million_tokens_usd": "0.15",
            "output_per_million_tokens_usd": "0.6",
        },
    )

    models = repositories.list_provider_models("openrouter")

    assert len(models) == 1
    assert models[0].pricing == {
        "input_per_million_tokens_usd": "0.15",
        "output_per_million_tokens_usd": "0.6",
    }


def test_repositories_count_provider_models(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/chat",
        display_name="OpenRouter Chat",
        capabilities=["chat"],
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/image",
        display_name="OpenRouter Image",
        capabilities=["image_generation"],
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/chat",
        display_name="Venice Chat",
        capabilities=["chat"],
    )

    assert repositories.count_provider_models("openrouter") == 2
    assert repositories.count_provider_models("venice") == 1
    assert repositories.count_provider_models("missing") == 0


def test_repositories_replace_provider_catalog_entries(
    repositories: PersistenceRepositories,
) -> None:
    repositories.replace_provider_catalog_entries(
        provider="openrouter",
        entries=[
            {
                "slug": "openai",
                "name": "OpenAI",
                "privacy_policy_url": "https://openai.com/privacy",
                "terms_of_service_url": "https://openai.com/terms",
                "status_page_url": "https://status.openai.com",
                "headquarters": "US",
                "datacenters": ["US", "IE"],
            },
            {
                "slug": "deepinfra",
                "name": "DeepInfra",
                "privacy_policy_url": None,
                "terms_of_service_url": None,
                "status_page_url": None,
                "headquarters": None,
                "datacenters": [],
            },
        ],
    )

    repositories.replace_provider_catalog_entries(
        provider="openrouter",
        entries=[
            {
                "slug": "openai",
                "name": "OpenAI Updated",
                "privacy_policy_url": "https://openai.com/privacy",
                "terms_of_service_url": "https://openai.com/terms",
                "status_page_url": "https://status.openai.com",
                "headquarters": "US",
                "datacenters": ["US"],
            },
        ],
    )

    entries = repositories.list_provider_catalog_entries("openrouter")
    assert [(entry.slug, entry.name) for entry in entries] == [
        ("openai", "OpenAI Updated"),
    ]
    assert entries[0].datacenters == ["US"]
    assert entries[0].refreshed_at is not None


def test_repositories_lease_and_complete_scheduled_tasks(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id = _persist_repository_save(repositories)

    task = repositories.upsert_scheduled_task(
        task_type="world_suggestion_review",
        save_id=save_id,
        interval_seconds=30,
        payload={"active_save_only": True},
        due_now=True,
    )

    assert task.enabled is True
    assert task.payload == {"active_save_only": True}
    assert [
        due.id
        for due in repositories.list_due_scheduled_tasks(
            task_types=("world_suggestion_review",),
            save_id=save_id,
        )
    ] == [task.id]

    leased = repositories.lease_scheduled_task(task.id, lease_seconds=60)

    assert leased is not None
    assert leased.lease_until is not None
    assert leased.last_started_at is not None
    assert repositories.list_due_scheduled_tasks(
        task_types=("world_suggestion_review",),
        save_id=save_id,
    ) == []

    job = repositories.create_job(
        type="world_suggestion_review",
        status="queued",
        payload={},
        save_id=save_id,
    )
    completed = repositories.complete_scheduled_task(
        task.id,
        succeeded=True,
        result={"applied_count": 1},
        last_job_id=job.id,
        next_run_after_seconds=120,
    )

    assert completed.lease_until is None
    assert completed.last_completed_at is not None
    assert completed.last_job_id == job.id
    assert completed.failure_count == 0
    assert completed.result == {"applied_count": 1}
    assert completed.error is None
    assert repositories.list_due_scheduled_tasks(
        task_types=("world_suggestion_review",),
        save_id=save_id,
    ) == []

    failed = repositories.complete_scheduled_task(
        task.id,
        succeeded=False,
        error="review provider unavailable",
        next_run_after_seconds=60,
    )

    assert failed.failure_count == 1
    assert failed.error == "review provider unavailable"


def test_repositories_list_scheduled_tasks_for_diagnostics(
    repositories: PersistenceRepositories,
) -> None:
    first_save_id, _message_id = _persist_repository_save(repositories)
    second_save_id, _message_id = _persist_repository_save(repositories)

    first = repositories.upsert_scheduled_task(
        task_type="world_suggestion_review",
        save_id=first_save_id,
        interval_seconds=30,
        payload={"active_save_only": True},
        due_now=True,
    )
    second = repositories.upsert_scheduled_task(
        task_type="memory_consolidation",
        save_id=second_save_id,
        interval_seconds=60,
        payload={"active_save_only": False},
        due_now=True,
    )

    assert [task.id for task in repositories.list_scheduled_tasks()] == [
        second.id,
        first.id,
    ]
    assert [
        task.id for task in repositories.list_scheduled_tasks(save_id=first_save_id)
    ] == [first.id]
    assert [
        task.id
        for task in repositories.list_scheduled_tasks(
            task_types=("memory_consolidation",),
        )
    ] == [second.id]


def test_repositories_check_context_update_suggestion_existence_by_status(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)

    assert repositories.has_context_update_suggestions(save_id) is False

    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="update",
        entity_type="location",
        field_path="status",
        proposed_value="unstable",
        reason="Narrator described the gallery shaking.",
        confidence=0.82,
        source_message_ids=[message_id],
    )

    assert repositories.has_context_update_suggestions(save_id) is True
    assert repositories.has_context_update_suggestions(
        save_id,
        status="applied",
    ) is False
    assert repositories.has_context_update_suggestions(
        save_id,
        status=None,
    ) is True

    repositories.update_context_update_suggestion_status(
        suggestion.id,
        status="applied",
    )

    assert repositories.has_context_update_suggestions(save_id) is False
    assert repositories.has_context_update_suggestions(
        save_id,
        status="applied",
    ) is True
    assert repositories.has_context_update_suggestions("missing-save") is False


def test_repositories_count_active_messages_by_role(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _persist_repository_save(repositories)
    player = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Mara",
        body="I trim the lamp wick.",
    )
    later_narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon steadies.",
    )
    deleted_narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="This line is later removed.",
    )
    other_save_id, _ = _persist_repository_save(repositories)
    repositories.append_message(
        save_id=other_save_id,
        role="player",
        speaker_name="Mara",
        body="This belongs to another save.",
    )
    for message_id, created_at in (
        (narrator_id, "2026-01-01 00:00:00"),
        (player.id, "2026-01-02 00:00:00"),
        (later_narrator.id, "2026-01-03 00:00:00"),
        (deleted_narrator.id, "2026-01-04 00:00:00"),
    ):
        repositories.connection.execute(
            "UPDATE messages SET created_at = ? WHERE id = ?",
            (created_at, message_id),
        )
    repositories.commit()
    repositories.archive_message(deleted_narrator.id)

    assert repositories.count_active_messages_by_role(
        save_id,
        roles=("narrator", "player", "system"),
    ) == {"narrator": 2, "player": 1, "system": 0}
    assert repositories.count_active_messages_by_role(
        save_id,
        roles=("narrator", "player"),
        created_at_lte="2026-01-02 00:00:00",
    ) == {"narrator": 1, "player": 1}
    assert repositories.count_active_messages_by_role(save_id, roles=()) == {}


def test_repositories_latest_active_message_created_at(
    repositories: PersistenceRepositories,
) -> None:
    save_id, first_narrator_id = _persist_repository_save(repositories)
    player = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Mara",
        body="I listen at the stair.",
    )
    latest_narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Footsteps answer below.",
    )
    deleted_latest = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="A deleted later narration.",
    )
    for message_id, created_at in (
        (first_narrator_id, "2026-02-01 00:00:00"),
        (player.id, "2026-02-02 00:00:00"),
        (latest_narrator.id, "2026-02-03 00:00:00"),
        (deleted_latest.id, "2026-02-04 00:00:00"),
    ):
        repositories.connection.execute(
            "UPDATE messages SET created_at = ? WHERE id = ?",
            (created_at, message_id),
        )
    repositories.commit()
    repositories.archive_message(deleted_latest.id)

    assert (
        repositories.latest_active_message_created_at(save_id, role="narrator")
        == "2026-02-03 00:00:00"
    )
    assert (
        repositories.latest_active_message_created_at(save_id, role="player")
        == "2026-02-02 00:00:00"
    )
    assert repositories.latest_active_message_created_at(save_id, role="system") is None


def test_repositories_count_active_memories(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    other_save_id, _ = _persist_repository_save(repositories)
    kept = repositories.add_memory(save_id=save_id, body="Kept", tags=["test"])
    archived = repositories.add_memory(save_id=save_id, body="Archived", tags=["test"])
    repositories.add_memory(save_id=other_save_id, body="Other", tags=["test"])
    repositories.archive_memory(archived.id)

    assert repositories.count_active_memories(save_id) == 1
    assert repositories.count_active_memories(other_save_id) == 1
    assert repositories.count_active_memories("missing-save") == 0
    assert repositories.list_memories(save_id) == [kept]


def test_update_memory_atomically_merges_canonical_collision(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    repositories.add_memory(
        save_id=save_id,
        body="Mara likes tea.",
        tags=["preference"],
        importance=0.7,
        source_message_ids=["message-keeper"],
    )
    duplicate = repositories.add_memory(
        save_id=save_id,
        body="Mara prefers tea.",
        tags=["dossier"],
        importance=0.9,
        source_message_ids=["message-duplicate"],
    )

    merged = repositories.update_memory(
        memory_id=duplicate.id,
        body="mara likes tea",
        tags=["dossier"],
        importance=0.9,
        source_message_ids=["message-duplicate"],
    )

    assert merged.id == duplicate.id
    assert set(merged.tags) == {"preference", "dossier"}
    assert merged.importance == 0.9
    assert set(merged.source_message_ids) == {
        "message-keeper",
        "message-duplicate",
    }
    assert repositories.list_memories(save_id) == [merged]


def test_repositories_consolidate_duplicates_without_losing_active_references(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    keeper = repositories.add_memory(
        save_id=save_id,
        body="Mara likes tea.",
        tags=["preference"],
    )
    duplicate_id = "legacy-memory-duplicate"
    repositories.connection.execute(
        "DROP INDEX idx_memories_save_claim_fingerprint_active"
    )
    repositories.connection.execute(
        """
        INSERT INTO memories(
            id, save_id, body, tags_json, importance,
            source_message_ids_json, claim_fingerprint,
            source_observation_ids_json
        )
        VALUES (?, ?, ?, '["tea"]', 0.9, '[]', ?, '[]')
        """,
        (
            duplicate_id,
            save_id,
            "mara likes tea",
            canonical_claim_fingerprint("mara likes tea"),
        ),
    )
    repositories.commit()
    character = repositories.add_character(save_id=save_id, name="Captain Ilyra")
    archived_keeper_edge = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=character.id,
        target_type="memory",
        target_id=keeper.id,
    )
    repositories.archive_character_knowledge_edge(archived_keeper_edge.id)
    repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=character.id,
        target_type="memory",
        target_id=duplicate_id,
    )
    privacy_character = repositories.add_character(
        save_id=save_id,
        name="Archivist Ren",
    )
    repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=privacy_character.id,
        target_type="memory",
        target_id=keeper.id,
        knowledge_state="knows",
        confidence=0.9,
        source_message_ids=["keeper-proof"],
    )
    repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=privacy_character.id,
        target_type="memory",
        target_id=duplicate_id,
        knowledge_state="does_not_know",
        confidence=0.7,
        source_message_ids=["duplicate-proof"],
    )
    archived_keeper_source = repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id=keeper.id,
        title="Archived keeper source",
        body=keeper.body,
    )
    repositories.archive_context_source(archived_keeper_source.id)
    repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id=duplicate_id,
        title="Active duplicate source",
        body="mara likes tea",
    )

    remapped = repositories.consolidate_active_memory_duplicates(save_id=save_id)

    assert remapped == {duplicate_id: keeper.id}
    assert [memory.id for memory in repositories.list_memories(save_id)] == [keeper.id]
    edges = repositories.list_character_knowledge_edges(save_id)
    assert {edge.target_id for edge in edges} == {keeper.id}
    privacy_edge = next(
        edge for edge in edges if edge.character_id == privacy_character.id
    )
    assert privacy_edge.knowledge_state == "does_not_know"
    assert privacy_edge.confidence == 0.9
    assert privacy_edge.source_message_ids == ["keeper-proof", "duplicate-proof"]
    [source] = repositories.list_context_sources(save_id, source_type="memory")
    assert source.source_id == keeper.id
    assert source.title == "Active duplicate source"


def test_repositories_consolidate_duplicates_keeps_body_provenance_atomic(
    repositories: PersistenceRepositories,
) -> None:
    save_id, hidden_message_id = _persist_repository_save(repositories)
    visible_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        body="The lamps are lit.",
    )
    keeper = repositories.add_memory(
        save_id=save_id,
        body="Mara likes tea.",
        tags=["preference"],
    )
    duplicate_id = "legacy-memory-duplicate"
    repositories.connection.execute(
        "DROP INDEX idx_memories_save_claim_fingerprint_active"
    )
    repositories.connection.execute(
        """
        INSERT INTO memories(
            id, save_id, body, tags_json, importance,
            source_message_ids_json, claim_fingerprint,
            source_observation_ids_json
        )
        VALUES (?, ?, 'mara likes tea', '[]', 0.5, '[]', ?, '[]')
        """,
        (
            duplicate_id,
            save_id,
            canonical_claim_fingerprint("mara likes tea"),
        ),
    )
    repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id=keeper.id,
        title="Hidden keeper",
        body="The hidden vault code is AMBER-77.",
        metadata={"source_message_ids": [hidden_message_id]},
    )
    repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id=duplicate_id,
        title="Harmless duplicate",
        body="The lamps are lit.",
        metadata={"source_message_ids": [visible_message.id]},
    )
    repositories.commit()

    repositories.consolidate_active_memory_duplicates(save_id=save_id)

    [source] = repositories.list_context_sources(
        save_id,
        source_type="memory",
    )
    assert source.source_id == keeper.id
    assert source.body == "The hidden vault code is AMBER-77."
    assert source.metadata["source_message_ids"] == [hidden_message_id]
    assert visible_message.id not in source.metadata["source_message_ids"]


def test_restore_memories_merges_active_fingerprint_collision(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    visible_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mara confirms the tea preference.",
    )
    archived = repositories.add_memory(
        save_id=save_id,
        body="Mara likes tea.",
        tags=["mara"],
        importance=0.4,
        source_message_id=message_id,
    )
    repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id=archived.id,
        title="Archived preference",
        body=archived.body,
        metadata={"source_message_ids": [message_id]},
    )
    repositories.archive_memory(archived.id)
    active = repositories.add_memory(
        save_id=save_id,
        body="mara likes tea",
        tags=["tea"],
        importance=0.9,
    )
    repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id=active.id,
        title="Replacement preference",
        body=active.body,
        metadata={"source_message_ids": [visible_message.id]},
    )
    character = repositories.add_character(
        save_id=save_id,
        name="Mara",
    )
    repositories.add_character_text_proactive_trigger(
        save_id=save_id,
        character_id=character.id,
        trigger_key=f"memory:{archived.id}",
        trigger_type="memory_changed",
        source_type="memory",
        source_id=archived.id,
        reason="Original preference",
    )
    repositories.add_character_text_proactive_trigger(
        save_id=save_id,
        character_id=character.id,
        trigger_key=f"memory:{active.id}",
        trigger_type="memory_changed",
        source_type="memory",
        source_id=active.id,
        reason="Replacement preference",
    )

    repositories.restore_memories(frozenset({archived.id}))

    [restored] = repositories.list_memories(save_id)
    assert restored.id == archived.id
    assert restored.tags == ["mara", "tea"]
    assert restored.importance == 0.9
    assert restored.source_message_ids == [message_id]
    assert repositories.get_memory(save_id, active.id) is None
    [source] = repositories.list_context_sources(
        save_id,
        source_type="memory",
    )
    assert source.source_id == archived.id
    assert source.title == "Archived preference"
    assert source.body == "Mara likes tea."
    assert source.metadata["source_message_ids"] == [message_id]
    assert visible_message.id not in source.metadata["source_message_ids"]
    [trigger] = repositories.list_character_text_proactive_triggers(save_id)
    assert trigger.trigger_key == f"memory:{archived.id}"
    assert trigger.source_id == archived.id
    assert trigger.reason == "Replacement preference"


def test_repositories_check_unprotected_character_existence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    protected = repositories.add_character(
        save_id=save_id,
        name="Protected Captain",
        protected_from_maintenance=True,
    )

    assert repositories.has_unprotected_character(save_id) is False

    unprotected = repositories.add_character(
        save_id=save_id,
        name="Watch Scout",
    )

    assert repositories.has_unprotected_character(save_id) is True

    repositories.archive_character(unprotected.id)

    assert repositories.has_unprotected_character(save_id) is False

    repositories.archive_character(protected.id)
    assert repositories.has_unprotected_character(save_id) is False


def test_repositories_query_jobs_by_status_type_and_save(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    other_save_id, _ = _persist_repository_save(repositories)
    repositories.create_job(
        save_id=save_id,
        type="context_update_retry",
        status="queued",
        payload={},
    )
    repositories.create_job(
        save_id=save_id,
        type="context_update_retry",
        status="running",
        payload={},
    )
    repositories.create_job(
        save_id=save_id,
        type="character_text_world_update_retry",
        status="queued",
        payload={},
    )
    repositories.create_job(
        save_id=other_save_id,
        type="context_update_retry",
        status="queued",
        payload={},
    )
    repositories.create_job(
        save_id=None,
        type="context_update_retry",
        status="queued",
        payload={},
    )
    terminal = repositories.create_job(
        save_id=save_id,
        type="context_update_retry",
        status="running",
        payload={},
    )
    repositories.update_job(terminal.id, status="succeeded", result={})

    assert repositories.has_matching_job(
        statuses=("queued", "running"),
        types=("context_update_retry",),
        save_id=save_id,
    ) is True
    assert repositories.has_matching_job(
        statuses=("failed",),
        types=("context_update_retry",),
        save_id=save_id,
    ) is False
    assert repositories.has_matching_job(
        statuses=("queued",),
        types=("missing_type",),
        save_id=save_id,
    ) is False
    assert repositories.list_job_save_ids(
        statuses=("queued",),
        types=("context_update_retry",),
    ) == tuple(sorted((save_id, other_save_id)))


def test_context_candidate_revision_token_ignores_current_player_message(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id = _persist_repository_save(repositories)
    before = repositories.context_candidate_revision_token(save_id)
    player = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Mara",
        body="I test whether the beacon still answers.",
    )

    ignored = repositories.context_candidate_revision_token(
        save_id,
        ignored_message_id=player.id,
    )
    included = repositories.context_candidate_revision_token(save_id)

    assert ignored == before
    assert included != before


def test_context_candidate_revision_token_ignores_current_player_message_edit(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id = _persist_repository_save(repositories)
    before = repositories.context_candidate_revision_token(save_id)
    player = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Mara",
        body="I test whether the beacon still answers.",
    )

    repositories.update_message_body(
        save_id=save_id,
        message_id=player.id,
        body="I test whether the beacon answers in the storm.",
    )

    assert (
        repositories.context_candidate_revision_token(
            save_id,
            ignored_message_id=player.id,
        )
        == before
    )
    assert repositories.context_candidate_revision_token(save_id) != before


def test_context_candidate_revision_token_changes_after_message_edit(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    before = repositories.context_candidate_revision_token(save_id)

    repositories.update_message_body(
        save_id=save_id,
        message_id=message_id,
        body="Ash claws the glass as the stair shakes.",
    )

    assert repositories.context_candidate_revision_token(save_id) != before


def test_context_candidate_revision_token_changes_after_message_archive_and_restore(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    before = repositories.context_candidate_revision_token(save_id)

    repositories.archive_message(message_id)
    archived = repositories.context_candidate_revision_token(save_id)
    repositories.restore_messages({message_id})
    restored = repositories.context_candidate_revision_token(save_id)

    assert archived != before
    assert restored != archived


def test_context_candidate_revision_token_changes_after_context_revision_bump(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    before = repositories.context_candidate_revision_token(save_id)

    repositories.add_memory(
        save_id=save_id,
        body="Mara heard the beacon answer in static.",
        tags=["beacon"],
        source_message_id=message_id,
    )

    assert repositories.context_candidate_revision_token(save_id) != before


def test_context_candidate_revision_token_does_not_scan_messages(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    player = repositories.append_message(
        save_id=save_id,
        role="player",
        body="I test whether the beacon still answers.",
    )
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        body="The beacon answers with a thin blue pulse.",
    )

    statements: list[str] = []
    repositories.connection.set_trace_callback(statements.append)
    try:
        repositories.context_candidate_revision_token(save_id)
        repositories.context_candidate_revision_token(
            save_id,
            ignored_message_id=player.id,
        )
    finally:
        repositories.connection.set_trace_callback(None)

    message_scans = [
        statement
        for statement in statements
        if "from messages" in " ".join(statement.lower().split())
    ]

    assert message_id != player.id
    assert message_scans == []


def test_character_age_round_trips_through_repository(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)

    character = repositories.add_character(
        save_id=save_id,
        name="Mara",
        age="early 30s",
    )

    fetched = repositories.get_character(character.id)
    assert character.age == "early 30s"
    assert fetched is not None
    assert fetched.age == "early 30s"
    assert repositories.list_characters(save_id)[0].age == "early 30s"

    updated = repositories.update_character(replace(character, age="ancient"))

    assert updated.age == "ancient"
    refetched = repositories.get_character(character.id)
    assert refetched is not None
    assert refetched.age == "ancient"


def test_repositories_delete_entity_links_for_inactive_stored_endpoints(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    location = repositories.add_location(
        save_id=save_id,
        name="Beacon Tower",
        source_message_id=message_id,
    )
    character = repositories.add_character(
        save_id=save_id,
        name="Mara",
        source_message_id=message_id,
    )
    memory = repositories.add_memory(
        save_id=save_id,
        body="Mara knows the beacon.",
        tags=["beacon"],
        source_message_id=message_id,
    )
    stale_memory = repositories.add_memory(
        save_id=save_id,
        body="The ash door is open.",
        tags=["ash"],
        source_message_id=message_id,
    )
    state = repositories.upsert_world_state(
        save_id=save_id,
        key="beacon.status",
        value={"lit": True},
        source_message_id=message_id,
    )
    stale_state = repositories.upsert_world_state(
        save_id=save_id,
        key="ash.status",
        value={"open": True},
        source_message_id=message_id,
    )
    summary = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=message_id,
        covers_message_end_id=message_id,
        body="The beacon is lit.",
        provider="fake",
        model="fake-chat",
    )
    stale_summary = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=message_id,
        covers_message_end_id=message_id,
        body="The ash door opened.",
        provider="fake",
        model="fake-chat",
    )
    active_link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id=memory.id,
        relation="knows",
        link_id="link-active-memory",
    )
    stale_memory_link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id=stale_memory.id,
        relation="knows",
        link_id="link-stale-memory",
    )
    stale_state_link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character.id,
        target_type="state",
        target_id=stale_state.id,
        relation="knows",
        link_id="link-stale-state",
    )
    stale_summary_link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="summary",
        entity_id=stale_summary.id,
        target_type="location",
        target_id=location.id,
        relation="mentions",
        link_id="link-stale-summary",
    )
    scenario_link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character.id,
        target_type="scenario_section",
        target_id="setting",
        relation="knows",
        link_id="link-scenario-section",
    )

    repositories.archive_memory(stale_memory.id)
    repositories.archive_world_state(save_id=save_id, key=stale_state.key)
    repositories.archive_summary(stale_summary.id)

    deleted = repositories.delete_entity_links_for_inactive_stored_endpoints(save_id)

    assert {link.id for link in deleted} == {
        stale_memory_link.id,
        stale_state_link.id,
        stale_summary_link.id,
    }
    assert repositories.list_entity_links(save_id) == [active_link, scenario_link]
    assert state.id in {item.id for item in repositories.list_world_state(save_id)}
    assert summary.id in {item.id for item in repositories.list_summaries(save_id)}


def test_repositories_persist_character_knowledge_graph_rows(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    character = repositories.add_character(
        save_id=save_id,
        name="Mara",
        source_message_id=message_id,
    )
    memory = repositories.add_memory(
        save_id=save_id,
        body="The player made the archive-code joke.",
        tags=["joke"],
        source_message_id=message_id,
    )
    edge = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=character.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="may_know",
        acquisition_method="overheard",
        confidence=0.68,
        source_message_id=message_id,
        source_message_ids=[message_id],
        evidence_quote="Mara was in the room for the joke.",
        edge_id="edge-mara-joke",
    )

    updated = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=character.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        confidence=1.0,
        source_message_id=message_id,
        evidence_quote="Mara laughed at the joke.",
    )
    visibility = repositories.add_message_visibility(
        save_id=save_id,
        message_id=message_id,
        character_id=character.id,
        visibility="visible",
        confidence=0.95,
        source="scene_presence",
        evidence="Mara was present.",
        visibility_id="visibility-mara-message",
    )

    assert edge.id == updated.id
    assert updated.knowledge_state == "knows"
    assert updated.acquisition_method == "witnessed"
    assert updated.source_message_ids == [message_id]
    assert updated.evidence_quote == "Mara laughed at the joke."
    assert repositories.list_character_knowledge_edges(save_id) == [updated]
    assert repositories.list_message_visibility(save_id) == [visibility]

    repositories.archive_character_knowledge_edge(updated.id)

    assert repositories.list_character_knowledge_edges(save_id) == []
    assert repositories.list_character_knowledge_edges(
        save_id,
        include_archived=True,
    )[0].archived_at is not None


def test_character_knowledge_updates_are_scoped_by_owner_and_target(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    mara = repositories.add_character(
        save_id=save_id,
        name="Mara",
        source_message_id=message_id,
    )
    lio = repositories.add_character(
        save_id=save_id,
        name="Lio",
        source_message_id=message_id,
    )
    memory = repositories.add_memory(
        save_id=save_id,
        body="The western archive uses a moonstone key.",
        tags=["archive"],
        source_message_id=message_id,
    )
    mara_edge = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=mara.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="may_know",
    )
    lio_edge = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=lio.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="knows",
    )

    updated_mara = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=mara.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="does_not_know",
    )
    edges = {
        edge.character_id: edge
        for edge in repositories.list_character_knowledge_edges(save_id)
    }

    assert updated_mara.id == mara_edge.id
    assert edges[mara.id].knowledge_state == "does_not_know"
    assert edges[lio.id].id == lio_edge.id
    assert edges[lio.id].knowledge_state == "knows"


def test_repositories_replace_and_list_message_scene_presence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    mara = repositories.add_character(save_id=save_id, name="Mara")
    ren = repositories.add_character(save_id=save_id, name="Ren")
    omitted = repositories.add_character(save_id=save_id, name="Omitted")

    first = repositories.replace_message_scene_presence(
        save_id=save_id,
        message_id=message_id,
        character_ids=[ren.id, mara.id, ren.id],
        source="context_snapshot",
    )

    assert {record.character_id for record in first} == {mara.id, ren.id}
    assert {record.source for record in first} == {"context_snapshot"}

    second = repositories.replace_message_scene_presence(
        save_id=save_id,
        message_id=message_id,
        character_ids=[omitted.id],
        source="manual",
    )

    assert [record.character_id for record in second] == [omitted.id]
    assert second[0].source == "manual"
    assert repositories.list_message_scene_presence(
        save_id,
        character_ids=(mara.id, ren.id),
    ) == []


def test_repositories_delete_message_scene_presence_for_archived_messages(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    character = repositories.add_character(save_id=save_id, name="Mara")
    later = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mara moves to the stairwell.",
    )
    repositories.replace_message_scene_presence(
        save_id=save_id,
        message_id=message_id,
        character_ids=[character.id],
    )
    repositories.replace_message_scene_presence(
        save_id=save_id,
        message_id=later.id,
        character_ids=[character.id],
    )

    archived = repositories.archive_messages_from(
        save_id=save_id,
        message_id=later.id,
    )

    assert [message.id for message in archived] == [later.id]
    assert [
        record.message_id
        for record in repositories.list_message_scene_presence(save_id)
    ] == [message_id]


def test_delete_save_removes_character_knowledge_graph_rows(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    character = repositories.add_character(
        save_id=save_id,
        name="Mara",
        source_message_id=message_id,
    )
    memory = repositories.add_memory(
        save_id=save_id,
        body="Mara knows the eastern signal code.",
        tags=["signals"],
        source_message_id=message_id,
    )
    repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=character.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        source_message_id=message_id,
    )
    repositories.add_message_visibility(
        save_id=save_id,
        message_id=message_id,
        character_id=character.id,
        visibility="visible",
        source="scene_presence",
    )
    repositories.replace_message_scene_presence(
        save_id=save_id,
        message_id=message_id,
        character_ids=[character.id],
    )

    assert repositories.delete_save(save_id) is True

    assert repositories.get_save(save_id) is None
    assert repositories.list_character_knowledge_edges(save_id) == []
    assert repositories.list_message_visibility(save_id) == []
    assert repositories.list_message_scene_presence(save_id) == []


def test_delete_save_removes_character_text_revisions_and_media_records(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    npc = repositories.add_character(save_id=save.id, name="Mika Arai", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Can we tak?",
    )
    revision = repositories.add_character_text_message_revision(
        save_id=save.id,
        text_message_id=message.id,
        previous_body=message.body,
        new_body="Can we talk?",
        diff_unified="--- previous\n+++ current\n",
        reconciliation_status="queued",
    )
    asset = repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="generated/text-attachment.png",
        thumbnail_path="generated/text-attachment.thumb.png",
        prompt="phone attachment",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )

    assert repositories.delete_save(save.id) is True

    assert repositories.get_save(save.id) is None
    assert repositories.list_character_text_message_revisions(save_id=save.id) == []
    assert repositories.list_character_text_messages(save_id=save.id) == []
    assert repositories.list_character_text_threads(save_id=save.id) == []
    assert repositories.list_all_media_assets(save.id) == []
    assert (
        repositories.connection.execute(
            "SELECT COUNT(*) FROM character_text_message_revisions WHERE id = ?",
            (revision.id,),
        ).fetchone()[0]
        == 0
    )
    assert (
        repositories.connection.execute(
            "SELECT COUNT(*) FROM media_assets WHERE id = ?",
            (asset.id,),
        ).fetchone()[0]
        == 0
    )


def test_repositories_persist_mvp_save_records(
    repositories: PersistenceRepositories,
) -> None:
    owner = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={
            "tone": "tense",
            "starting_location": "Gatehouse",
        },
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
        custom_instructions="  Keep player options concise and grounded.  ",
        owner_user_id=owner.id,
    )

    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon gutters in the tower.",
        provider="openrouter",
        model="anthropic/claude-sonnet",
        token_estimate=37,
    )
    messages = repositories.list_messages(save.id)

    assert scenario.title == "Ashfall Keep"
    assert save.scenario_id == scenario.id
    assert save.owner_user_id == owner.id
    assert save.custom_instructions == "Keep player options concise and grounded."
    fetched_save = repositories.get_save(save.id)
    assert fetched_save is not None
    assert fetched_save.id == save.id
    assert fetched_save.title == save.title
    assert fetched_save.owner_user_id == owner.id
    assert fetched_save.updated_at is not None
    assert message.updated_at is not None
    assert fetched_save.updated_at >= message.updated_at
    assert repositories.list_saves()[0].custom_instructions == (
        "Keep player options concise and grounded."
    )
    assert [(item.provider, item.model) for item in messages] == [
        ("openrouter", "anthropic/claude-sonnet")
    ]
    assert messages[0].token_estimate == 37
    assert messages[0].body == "The beacon gutters in the tower."
    assert messages[0].created_at is not None
    assert messages[0].deleted_at is None

    repositories.upsert_world_state(
        save_id=save.id,
        key="location.current",
        value={"name": "Gatehouse", "danger": "high"},
        category="location",
        confidence=0.8,
        source_message_id=message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="location.current",
        value={"name": "Tower", "danger": "critical"},
        category="location",
        confidence=0.95,
        source_message_id=message.id,
    )
    world_state = repositories.list_world_state(save.id)

    assert len(world_state) == 1
    assert world_state[0].key == "location.current"
    assert world_state[0].value == {"name": "Tower", "danger": "critical"}
    assert world_state[0].confidence == 0.95

    memory = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra promised to hold the east stair.",
        tags=["npc", "promise"],
        importance=0.7,
        source_message_id=message.id,
    )
    memories = repositories.list_memories(save.id)

    assert memories == [memory]
    assert memories[0].tags == ["npc", "promise"]
    assert memories[0].source_message_ids == [message.id]

    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=message.id,
        covers_message_end_id=message.id,
        body="The watch began as the tower beacon started failing.",
        provider="openrouter",
        model="anthropic/claude-sonnet",
    )
    summaries = repositories.list_summaries(save.id)

    assert summaries == [summary]
    assert summaries[0].provider == "openrouter"
    assert summaries[0].model == "anthropic/claude-sonnet"


def test_repositories_scope_saves_by_owner_assignment_and_admin(
    repositories: PersistenceRepositories,
) -> None:
    admin = repositories.create_user(
        username="Admin",
        role="admin",
        password_hash="hash",
    )
    owner = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    assigned = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    outsider = repositories.create_user(
        username="Rook",
        role="user",
        password_hash="hash",
    )
    disabled = repositories.create_user(
        username="Disabled",
        role="user",
        password_hash="hash",
        status="disabled",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"tone": "tense"},
    )
    owned = repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
        owner_user_id=owner.id,
    )
    shared = repositories.create_save(
        scenario_id=scenario.id,
        title="Signal Watch",
        owner_user_id=owner.id,
    )
    other = repositories.create_save(
        scenario_id=scenario.id,
        title="Fog Watch",
        owner_user_id=outsider.id,
    )

    repositories.grant_save_access(save_id=shared.id, user_id=assigned.id)
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, last_opened_at = ? WHERE id = ?",
        ("2026-05-03 00:00:00", "2026-05-01 00:00:00", owned.id),
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, last_opened_at = ? WHERE id = ?",
        ("2026-05-02 00:00:00", "2026-05-02 00:00:00", shared.id),
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, last_opened_at = ? WHERE id = ?",
        ("2026-05-01 00:00:00", "2026-05-03 00:00:00", other.id),
    )
    repositories.commit()

    assert [save.id for save in repositories.list_saves_for_user(admin)] == [
        owned.id,
        shared.id,
        other.id,
    ]
    assert [save.id for save in repositories.list_saves_for_user(owner)] == [
        owned.id,
        shared.id,
    ]
    assert [save.id for save in repositories.list_saves_for_user(assigned)] == [
        shared.id,
    ]
    assert repositories.user_can_access_save(owner, owned.id) is True
    assert repositories.user_can_access_save(assigned, shared.id) is True
    assert repositories.user_can_access_save(assigned, owned.id) is False
    assert repositories.user_can_access_save(outsider, owned.id) is False
    assert repositories.list_saves_for_user(disabled) == []
    assert repositories.user_can_access_save(disabled, owned.id) is False
    assert repositories.get_save_for_user(owner, other.id) is None
    admin_other = repositories.get_save_for_user(admin, other.id)
    assert admin_other is not None
    assert admin_other.id == other.id


def test_repositories_claim_unowned_saves_and_track_user_active_save(
    repositories: PersistenceRepositories,
) -> None:
    admin = repositories.create_user(
        username="Admin",
        role="admin",
        password_hash="hash",
    )
    owner = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"tone": "tense"},
    )
    legacy = repositories.create_save(scenario_id=scenario.id, title="Legacy")
    owned = repositories.create_save(
        scenario_id=scenario.id,
        title="Owned",
        owner_user_id=owner.id,
    )

    assert repositories.claim_unowned_saves(admin.id) == 1
    repositories.set_user_active_save_id(user_id=admin.id, save_id=legacy.id)
    repositories.set_user_active_save_id(user_id=owner.id, save_id=owned.id)

    claimed_legacy = repositories.get_save(legacy.id)
    claimed_owned = repositories.get_save(owned.id)
    assert claimed_legacy is not None
    assert claimed_owned is not None
    assert claimed_legacy.owner_user_id == admin.id
    assert claimed_owned.owner_user_id == owner.id
    assert repositories.get_user_active_save_id(admin.id) == legacy.id
    assert repositories.get_user_active_save_id(owner.id) == owned.id

    repositories.clear_user_active_save_id(owner.id)

    assert repositories.get_user_active_save_id(owner.id) is None


def test_repositories_list_user_active_save_ids_filters_accessible_active_users(
    repositories: PersistenceRepositories,
) -> None:
    admin = repositories.create_user(
        username="Admin",
        role="admin",
        password_hash="hash",
    )
    owner = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    assigned = repositories.create_user(
        username="Rook",
        role="user",
        password_hash="hash",
    )
    outsider = repositories.create_user(
        username="Outsider",
        role="user",
        password_hash="hash",
    )
    disabled = repositories.create_user(
        username="Disabled",
        role="user",
        password_hash="hash",
        status="disabled",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"tone": "tense"},
    )
    admin_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Admin Save",
        owner_user_id=admin.id,
    )
    owned_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Owned Save",
        owner_user_id=owner.id,
    )
    shared_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Shared Save",
        owner_user_id=owner.id,
    )
    disabled_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Disabled Save",
        owner_user_id=disabled.id,
    )
    repositories.grant_save_access(save_id=shared_save.id, user_id=assigned.id)

    repositories.set_user_active_save_id(user_id=admin.id, save_id=admin_save.id)
    repositories.set_user_active_save_id(user_id=owner.id, save_id=shared_save.id)
    repositories.set_user_active_save_id(user_id=assigned.id, save_id=shared_save.id)
    repositories.set_user_active_save_id(user_id=outsider.id, save_id=owned_save.id)
    repositories.set_user_active_save_id(user_id=disabled.id, save_id=disabled_save.id)

    assert repositories.list_user_active_save_ids() == (
        admin_save.id,
        shared_save.id,
    )


def test_repositories_update_save_custom_instructions(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"starting_location": "Gatehouse"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")

    updated = repositories.update_save_custom_instructions(
        save_id=save.id,
        custom_instructions="  Favor active choices over long monologues.  ",
    )
    cleared = repositories.update_save_custom_instructions(
        save_id=save.id,
        custom_instructions="  ",
    )

    assert updated.custom_instructions == "Favor active choices over long monologues."
    assert cleared.custom_instructions == ""
    assert repositories.get_save(save.id) == cleared


def test_repositories_persist_normalized_context_sources(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"locations": "The tower lens is cracked but still usable."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    context_source_id = f"scenario:{scenario.id}:section:locations"

    context_source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="scenario_section",
        source_id=context_source_id,
        title="locations",
        body="The tower lens is cracked but still usable.",
        metadata={"scenario_id": scenario.id, "section_id": "locations"},
        token_estimate=12,
    )
    updated_context_source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="scenario_section",
        source_id=context_source_id,
        title="locations",
        body="The tower lens is cracked, usable, and warm to the touch.",
        metadata={"scenario_id": scenario.id, "section_id": "locations"},
        token_estimate=14,
    )

    context_sources = repositories.list_context_sources(save.id)

    assert context_source.id == updated_context_source.id
    assert context_sources == [updated_context_source]
    assert context_sources[0].save_id == save.id
    assert context_sources[0].source_type == "scenario_section"
    assert context_sources[0].source_id == context_source_id
    assert context_sources[0].title == "locations"
    assert context_sources[0].body == (
        "The tower lens is cracked, usable, and warm to the touch."
    )
    assert context_sources[0].metadata == {
        "scenario_id": scenario.id,
        "section_id": "locations",
    }
    assert context_sources[0].token_estimate == 14


def test_repositories_search_context_sources_with_fts(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"locations": "The tower lens is cracked but still usable."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    target = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-moonstone",
        title="moonstone compass",
        body="The moonstone compass unlocks the eastern scriptorium.",
        metadata={"fact_type": "inventory", "importance": 0.9},
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-ledger",
        title="routine bridge ledger",
        body="Routine bridge ledger entries mention door hinges and patrol bells.",
        metadata={"fact_type": "memory", "importance": 0.2},
    )
    archived = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-archived",
        title="moonstone rumor",
        body="An archived moonstone rumor should not be retrieved.",
        metadata={"fact_type": "memory", "importance": 0.2},
    )
    repositories.archive_context_source(archived.id)

    hits = repositories.search_context_sources(
        save.id,
        query_terms={"moonstone", "scriptorium"},
        source_types={"memory"},
        limit=8,
    )

    assert [hit.record.source_id for hit in hits] == ["memory-moonstone"]
    assert hits[0].record == target
    assert isinstance(hits[0].bm25_rank, float)


def test_context_source_search_enforces_graph_scope_outside_raw_candidates(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive",
        premise="An archive holds unevenly shared secrets.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Index")
    present = repositories.add_character(save_id=save.id, name="Nira", met=True)
    absent = repositories.add_character(save_id=save.id, name="Lio", met=True)
    memory = repositories.add_memory(
        save_id=save.id,
        body="The moonstone opens the cobalt ledger.",
        tags=["moonstone"],
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id=memory.id,
        title="moonstone",
        body=memory.body,
        metadata={"indexed_by": "continuity_index"},
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=absent.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="knows",
        acquisition_method="told",
    )

    blocked = repositories.search_context_sources(
        save.id,
        query_terms={"moonstone"},
        source_types={"memory"},
        limit=1,
        visibility_character_ids={present.id},
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=present.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="knows",
        acquisition_method="told",
    )
    allowed = repositories.search_context_sources(
        save.id,
        query_terms={"moonstone"},
        source_types={"memory"},
        limit=1,
        visibility_character_ids={present.id},
    )

    assert blocked == []
    assert [hit.record.source_id for hit in allowed] == [memory.id]


def test_repositories_search_context_sources_with_unicode_terms(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    cyrillic = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-cyrillic",
        title="Северный маяк",
        body="Северный маяк открывается медным ключом.",
    )
    cjk = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-cjk",
        title="月石羅針盤",
        body="月石羅針盤は東の書庫を開く。",
    )

    cyrillic_hits = repositories.search_context_sources(
        save.id,
        query_terms={"маяк"},
        source_types={"memory"},
        limit=8,
    )
    cjk_hits = repositories.search_context_sources(
        save.id,
        query_terms={"月石羅針盤はどこ"},
        source_types={"memory"},
        limit=8,
    )

    assert [hit.record for hit in cyrillic_hits] == [cyrillic]
    assert [hit.record for hit in cjk_hits] == [cjk]


def test_repositories_unicode_match_all_cannot_be_starved_by_partial_matches(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    target = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-target",
        title="秘密の地図",
        body="秘密の地図は古い書庫にある。",
    )
    for index in range(90):
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id=f"memory-noise-{index:02d}",
            title=f"秘密 {index:02d}",
            body="秘密だけを記録した新しいメモ。",
        )

    hits = repositories.search_context_sources(
        save.id,
        query_terms={"秘密", "地図"},
        source_types={"memory"},
        limit=1,
        match_all=True,
    )

    assert [hit.record for hit in hits] == [target]


def test_repositories_indexes_middle_han_bigram_before_term_cap(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    body = "".join(chr(0x4E00 + index) for index in range(220))
    target = repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-long-han-run",
        title="長文",
        body=body,
    )

    hits = repositories.search_context_sources(
        save_id,
        query_terms={body[200:202]},
        source_types={"memory"},
        limit=1,
        match_all=True,
    )

    assert [hit.record for hit in hits] == [target]


def test_repositories_matches_middle_han_trigram_via_bigrams(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    body = "".join(chr(0x4E00 + index) for index in range(1_000))
    target = repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-long-han-trigram",
        title="長文",
        body=body,
    )
    query = body[700:703]

    hits = repositories.search_context_sources(
        save_id,
        query_terms={*cjk_lexical_anchors(query), "where", "is"},
        source_types={"memory"},
        limit=1,
        match_all=True,
    )
    broad_hits = repositories.search_context_sources(
        save_id,
        query_terms={*cjk_lexical_anchors(query), "where", "is"},
        source_types={"memory"},
        limit=1,
    )

    assert [hit.record for hit in hits] == [target]
    assert [hit.record for hit in broad_hits] == [target]


def test_repositories_exact_phrase_supports_short_ascii_identifiers(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    target = repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-short-code",
        title="Vault code",
        body="Only A 7 opens the vault.",
    )

    hits = repositories.search_context_sources(
        save_id,
        query_terms=set(),
        source_types={"memory"},
        limit=8,
        exact_phrases=("A 7",),
    )

    assert [hit.record for hit in hits] == [target]


def test_repositories_exact_identifier_cannot_be_starved_by_split_matches(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    target = repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-a7-target",
        title="Vault identifier",
        body="Only A-7 opens the old river vault.",
    )
    for index in range(90):
        repositories.upsert_context_source(
            save_id=save_id,
            source_type="memory",
            source_id=f"memory-a7-noise-{index:02d}",
            title=f"Split tokens {index:02d}",
            body=f"A 7 appears in newer record {index:02d}.",
        )
        repositories.upsert_context_source(
            save_id=save_id,
            source_type="memory",
            source_id=f"memory-a7-longer-{index:02d}",
            title=f"Longer identifier {index:02d}",
            body=f"Only A-7.{index:02d} opens the newer vault.",
        )
    repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-a70-near-match",
        title="Nearby identifier",
        body="Only A-70 opens the upper vault.",
    )

    hits = repositories.search_context_sources(
        save_id,
        query_terms={"a", "7"},
        source_types={"memory"},
        limit=1,
        exact_identifiers=("A-7",),
    )

    assert [hit.record for hit in hits] == [target]


def test_repositories_exact_identifier_lookup_does_not_scan_filter_udf(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    for index in range(200):
        repositories.upsert_context_source(
            save_id=save_id,
            source_type="memory",
            source_id=f"memory-hidden-{index:02d}",
            title=f"Hidden split code {index:02d}",
            body="A 7 appears in a private maintenance record.",
        )
    calls = 0

    def count_identifier_checks(_value: object, _identifier: object) -> int:
        nonlocal calls
        calls += 1
        return 0

    repositories.connection.create_function(
        "bragi_identifier_filter_matches",
        2,
        count_identifier_checks,
        deterministic=True,
    )

    hits = repositories.search_context_sources(
        save_id,
        query_terms=set(),
        source_types={"memory"},
        limit=1,
        exact_identifiers=("A-7",),
    )

    assert hits == []
    assert calls == 0


def test_repositories_indexes_exact_identifier_at_source_tail(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    target = repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-tail-code",
        title="Archive codes",
        body=(
            " ".join(f"ARCHIVE-{index:03d}" for index in range(129))
            + " TARGET-9999 "
            + " ".join(f"LATER-{index:03d}" for index in range(129))
        ),
    )
    repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-split-code",
        title="Nearby split code",
        body="TARGET 9999 is written without the separator.",
    )

    hits = repositories.search_context_sources(
        save_id,
        query_terms={"target", "9999"},
        source_types={"memory"},
        limit=1,
        exact_identifiers=("TARGET-9999",),
    )

    assert [hit.record for hit in hits] == [target]


def test_repositories_rebuild_preserves_exact_identifier_after_long_token(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    target = repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-long-token-tail",
        title="Archive codes",
        body="KEEP-1 " + ("A" * 70_000) + " TAIL-2",
    )

    repositories.rebuild_context_source_search_terms(save_id)

    identifiers = {
        row[0]
        for row in repositories.connection.execute(
            """
            SELECT identifier
            FROM context_source_exact_identifiers
            WHERE context_source_id = ?
            """,
            (target.id,),
        )
    }
    assert identifiers == {"keep-1", "tail-2"}


def test_repositories_restore_rebuilds_archived_exact_identifier_index(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    target = repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-restored-code",
        title="Restored vault identifier",
        body="Only SECRET-42 opens the restored vault.",
    )
    repositories.connection.execute(
        """
        UPDATE context_sources
        SET archived_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (target.id,),
    )
    repositories.connection.execute(
        "DELETE FROM context_source_exact_identifiers WHERE context_source_id = ?",
        (target.id,),
    )
    repositories.commit()

    repositories.restore_context_sources({target.id})
    hits = repositories.search_context_sources(
        save_id,
        query_terms={"secret", "42"},
        source_types={"memory"},
        limit=1,
        exact_identifiers=("SECRET-42",),
    )

    assert [hit.record.id for hit in hits] == [target.id]


def test_repositories_restore_preserves_legacy_normalized_budget_allowance(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    source = repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-legacy-budget",
        title="Legacy expansion",
        body="\ufdfa" * 32,
    )
    normalized_bytes = repositories.connection.execute(
        """
        SELECT normalized_text_bytes
        FROM context_source_index_budget_state
        WHERE save_id = ?
        """,
        (save_id,),
    ).fetchone()[0]
    repositories.connection.execute(
        """
        INSERT INTO context_source_legacy_budget_limits(
            save_id, normalized_text_bytes
        )
        VALUES (?, ?)
        """,
        (save_id, normalized_bytes),
    )
    repositories.connection.execute(
        """
        UPDATE context_sources
        SET archived_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (source.id,),
    )
    repositories.commit()
    monkeypatch.setattr(
        repositories_module,
        "MAX_CONTEXT_SOURCE_NORMALIZED_BYTES_PER_REBUILD",
        1,
    )

    repositories.restore_context_sources({source.id})

    assert repositories.connection.execute(
        "SELECT archived_at FROM context_sources WHERE id = ?",
        (source.id,),
    ).fetchone()[0] is None


def test_repositories_restore_preserves_legacy_record_budget_allowance(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    source = repositories.upsert_context_source(
        save_id=save_id,
        source_type="custom_note",
        source_id="legacy-record-budget",
        title="Legacy expansion",
        body="\ufdfa" * 32,
    )
    normalized_bytes = repositories.connection.execute(
        """
        SELECT normalized_text_bytes
        FROM context_source_normalized_budget_entries
        WHERE context_source_id = ?
        """,
        (source.id,),
    ).fetchone()[0]
    monkeypatch.setattr(
        repositories_module,
        "MAX_CONTEXT_SOURCE_NORMALIZED_BYTES_PER_RECORD",
        1,
    )
    repositories.ensure_context_source_legacy_budget_limit(
        save_id=save_id,
        normalized_text_bytes=normalized_bytes,
        normalized_record_bytes=normalized_bytes,
    )
    repositories.connection.execute(
        """
        UPDATE context_sources
        SET archived_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (source.id,),
    )
    repositories.commit()

    repositories.restore_context_sources({source.id})

    assert repositories.connection.execute(
        "SELECT archived_at FROM context_sources WHERE id = ?",
        (source.id,),
    ).fetchone()[0] is None


def test_repositories_rejects_index_budget_before_persisting_source(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-existing",
        title="Existing marker",
        body="Existing marker code EXISTING-1.",
    )
    existing_index_rows = repositories.connection.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM context_source_search_terms WHERE save_id = ?)
            + (
                SELECT COUNT(*)
                FROM context_source_exact_identifiers
                WHERE save_id = ?
            )
        """,
        (save_id, save_id),
    ).fetchone()[0]
    monkeypatch.setattr(
        repositories_module,
        "MAX_CONTEXT_INDEX_ROWS_PER_REBUILD",
        existing_index_rows,
    )

    with pytest.raises(ValueError, match="too large to rebuild"):
        repositories.upsert_context_source(
            save_id=save_id,
            source_type="memory",
            source_id="memory-rejected",
            title="Rejected marker",
            body="Rejected marker code REJECTED-2.",
        )

    assert (
        repositories.connection.execute(
            """
            SELECT COUNT(*)
            FROM context_sources
            WHERE save_id = ? AND source_id = 'memory-rejected'
            """,
            (save_id,),
        ).fetchone()[0]
        == 0
    )


def test_repositories_rejects_source_text_budget_before_persisting_source(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    monkeypatch.setattr(
        repositories_module,
        "MAX_CONTEXT_SOURCE_TEXT_BYTES_PER_REBUILD",
        16,
    )

    with pytest.raises(ValueError, match="source text is too large"):
        repositories.upsert_context_source(
            save_id=save_id,
            source_type="memory",
            source_id="memory-rejected",
            title="Rejected marker",
            body="This body exceeds the test budget.",
        )

    assert (
        repositories.connection.execute(
            """
            SELECT COUNT(*)
            FROM context_sources
            WHERE save_id = ? AND source_id = 'memory-rejected'
            """,
            (save_id,),
        ).fetchone()[0]
        == 0
    )


def test_repositories_rejects_aggregate_normalized_budget_before_persisting_source(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-existing",
        title="Existing",
        body="\ufdfa" * 32,
    )
    normalized_bytes = repositories.connection.execute(
        """
        SELECT normalized_text_bytes
        FROM context_source_index_budget_state
        WHERE save_id = ?
        """,
        (save_id,),
    ).fetchone()[0]
    monkeypatch.setattr(
        repositories_module,
        "MAX_CONTEXT_SOURCE_NORMALIZED_BYTES_PER_REBUILD",
        normalized_bytes,
    )

    with pytest.raises(ValueError, match="Normalized context source text"):
        repositories.upsert_context_source(
            save_id=save_id,
            source_type="memory",
            source_id="memory-rejected",
            title="Rejected",
            body="\ufdfa",
        )

    assert (
        repositories.connection.execute(
            """
            SELECT COUNT(*)
            FROM context_sources
            WHERE save_id = ? AND source_id = 'memory-rejected'
            """,
            (save_id,),
        ).fetchone()[0]
        == 0
    )


def test_repositories_bounds_index_rebuild_before_writing(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    first = repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-first",
        title="First memory",
        body="The amber marker opens archive seven.",
    )
    repositories.upsert_context_source(
        save_id=save_id,
        source_type="memory",
        source_id="memory-second",
        title="Second memory",
        body="The cobalt marker opens archive eight.",
    )
    repositories.connection.execute(
        "DELETE FROM context_source_search_terms WHERE context_source_id = ?",
        (first.id,),
    )
    repositories.commit()
    monkeypatch.setattr(
        repositories_module,
        "MAX_CONTEXT_INDEX_ROWS_PER_REBUILD",
        1,
    )

    with pytest.raises(ValueError, match="too large to rebuild"):
        repositories.rebuild_context_source_search_terms(save_id)

    assert repositories.connection.execute(
        """
        SELECT COUNT(*)
        FROM context_source_search_terms
        WHERE context_source_id = ?
        """,
        (first.id,),
    ).fetchone()[0] == 0


def test_repositories_mixed_unicode_match_all_requires_ascii_terms(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    target = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-target",
        title="秘密 vault marker",
        body="The mixed-script marker identifies the old vault.",
    )
    for index in range(40):
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id=f"memory-noise-{index:02d}",
            title=f"秘密 {index:02d}",
            body="秘密だけを記録した新しいメモ。",
        )

    hits = repositories.search_context_sources(
        save.id,
        query_terms={"秘密", "vault"},
        source_types={"memory"},
        limit=1,
        match_all=True,
    )

    assert [hit.record for hit in hits] == [target]


def test_repositories_bound_large_mixed_script_query(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    target = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-large-mixed-query",
        title="古い秘密の扉",
        body="古い秘密の扉は月石で開く。",
    )
    query_terms = {f"unrelatedterm{index:04d}" for index in range(1_100)}
    query_terms.add("秘密")

    hits = repositories.search_context_sources(
        save.id,
        query_terms=query_terms,
        source_types={"memory"},
        limit=8,
    )

    assert [hit.record for hit in hits] == [target]


def test_repositories_exact_phrase_precedes_bounded_all_term_matches(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    target = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-exact-phrase",
        title="Old mechanism",
        body="The copper notch under the western stair opens the vault.",
    )
    for index in range(40):
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id=f"memory-all-terms-{index:02d}",
            title=f"New mechanism {index:02d}",
            body=(
                "Western stair records say the notch inspection found copper "
                f"under shelving {index:02d}."
            ),
        )

    hits = repositories.search_context_sources(
        save.id,
        query_terms={"copper", "notch", "western", "stair"},
        source_types={"memory"},
        limit=24,
        match_all=True,
        exact_phrases=("copper notch under the western stair",),
    )

    assert hits[0].record == target


def test_repositories_prioritize_specific_phrase_over_generic_expansion(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    target = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-specific",
        title="Old mechanism",
        body="The copper notch under the western stair opens the vault.",
    )
    for index in range(40):
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id=f"memory-generic-{index:02d}",
            title=f"Archive note {index:02d}",
            body="A generic archive record.",
        )

    hits = repositories.search_context_sources(
        save.id,
        query_terms={"copper", "notch", "archive"},
        source_types={"memory"},
        limit=24,
        exact_phrases=(
            "copper notch under the western stair",
            "archive",
        ),
    )

    assert hits[0].record == target


def test_repositories_exact_phrase_uses_unicode_normalization_and_casefold(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    target = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-accented",
        title="Old signal",
        body="The CAFÉ beacon marks the eastern bridge.",
    )

    hits = repositories.search_context_sources(
        save.id,
        query_terms={"café", "beacon"},
        source_types={"memory"},
        limit=1,
        exact_phrases=("cafe\u0301 beacon",),
    )

    assert [hit.record for hit in hits] == [target]


def test_repositories_apply_context_visibility_before_search_limit(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    for index in range(12):
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id=f"hidden-{index:02d}",
            title=f"moonstone hidden {index:02d}",
            body="The moonstone opens the hidden archive.",
            metadata={"known_by": ["Lio"]},
        )
    accessible = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="accessible",
        title="moonstone public",
        body="The moonstone opens the public archive.",
        metadata={"known_by": ["mara"]},
    )

    hits = repositories.search_context_sources(
        save.id,
        query_terms={"moonstone"},
        source_types={"memory"},
        limit=1,
        allowed_owner_names={"Mara"},
    )

    assert [hit.record for hit in hits] == [accessible]


def test_repositories_apply_message_visibility_before_search_limit(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    hidden_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="A hidden moonstone rumor circulates.",
    )
    visible_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The public moonstone archive opens.",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden_message.id,
        character_id=character.id,
        visibility="not_visible",
    )
    for index in range(90):
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="observation",
            source_id=f"hidden-observation-{index:02d}",
            title=f"moonstone hidden {index:02d}",
            body="The moonstone opens the hidden archive.",
            metadata={"source_message_ids": [hidden_message.id]},
        )
    accessible = repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id="accessible-observation",
        title="moonstone public",
        body="The moonstone opens the public archive.",
        metadata={"source_message_ids": [visible_message.id]},
    )

    hits = repositories.search_context_sources(
        save.id,
        query_terms={"moonstone"},
        source_types={"observation"},
        limit=1,
        visibility_character_ids={character.id},
    )

    assert [hit.record for hit in hits] == [accessible]


def test_repositories_filter_singular_message_provenance_and_allow_visible_group(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    hidden_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="A hidden moonstone rumor circulates.",
    )
    visible_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="A public witness independently confirms the moonstone.",
    )
    character = repositories.add_character(save_id=save.id, name="Captain Ilyra")
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden_message.id,
        character_id=character.id,
        visibility="not_visible",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="singular-hidden",
        title="moonstone singular",
        body="The moonstone opens the archive.",
        metadata={"source_message_id": hidden_message.id},
    )
    accessible = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="alternative-grounding",
        title="moonstone confirmed",
        body="The moonstone opens the archive.",
        metadata={
            "source_message_ids": [hidden_message.id, visible_message.id],
            "source_provenance_groups": [
                [hidden_message.id],
                [visible_message.id],
            ],
        },
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="conjunctive-grounding",
        title="moonstone and vault code",
        body="The moonstone opens the archive and reveals the hidden vault code.",
        metadata={
            "source_message_ids": [hidden_message.id, visible_message.id],
            "source_provenance_groups": [
                [hidden_message.id],
                [visible_message.id],
            ],
            "source_provenance_mode": "all",
        },
    )

    hits = repositories.search_context_sources(
        save.id,
        query_terms={"moonstone"},
        source_types={"memory"},
        limit=8,
        visibility_character_ids={character.id},
    )

    assert [hit.record for hit in hits] == [accessible]


def test_repositories_apply_blocked_scoped_targets_before_search_limit(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    blocked: set[tuple[str, str]] = set()
    for index in range(12):
        source_id = f"hidden-{index:02d}"
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id=source_id,
            title=f"moonstone secret {index:02d}",
            body="The moonstone opens the concealed archive.",
        )
        blocked.add(("memory", source_id))
    blocked.update(
        ("memory", f"nonexistent-hidden-{index:04d}")
        for index in range(1_100)
    )
    accessible = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="accessible",
        title="moonstone public",
        body="The moonstone opens the public archive.",
    )

    hits = repositories.search_context_sources(
        save.id,
        query_terms={"moonstone"},
        source_types={"memory"},
        limit=1,
        blocked_source_keys=blocked,
    )

    assert [hit.record for hit in hits] == [accessible]


def test_repositories_expire_scene_scratch_on_generation_change(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    first_location = repositories.add_location(
        save_id=save_id,
        name="Beacon",
        source_message_id=message_id,
    )
    second_location = repositories.add_location(
        save_id=save_id,
        name="Archive",
        source_message_id=message_id,
    )
    scene = repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=first_location.id,
        source_message_id=message_id,
    )
    scratch = repositories.upsert_context_source(
        save_id=save_id,
        source_type="observation",
        source_id="scratch-observation",
        title="Temporary lens state",
        body="The lens is warm.",
        metadata={"curation_action": "scene_scratch"},
        scene_snapshot_id=scene.id,
        scene_generation=scene.scene_generation,
        created_turn_number=1,
        expires_after_turn_number=13,
    )

    same_scene = repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=first_location.id,
        source_message_id=message_id,
    )
    assert same_scene.scene_generation == scene.scene_generation
    assert repositories.get_context_source(scratch.id) is not None

    advanced_scene = repositories.advance_scene_generation(
        save_id=save_id,
        source_message_id=message_id,
    )
    same_location_next_scene = repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=first_location.id,
        situation="A new confrontation begins.",
        source_message_id=message_id,
    )
    assert same_location_next_scene.scene_generation == advanced_scene.scene_generation
    assert same_location_next_scene.scene_generation == scene.scene_generation + 1
    assert repositories.get_context_source(scratch.id) is None

    next_scratch = repositories.upsert_context_source(
        save_id=save_id,
        source_type="observation",
        source_id="next-scratch-observation",
        title="Temporary archive state",
        body="The confrontation remains unresolved.",
        metadata={"curation_action": "scene_scratch"},
        scene_snapshot_id=same_location_next_scene.id,
        scene_generation=same_location_next_scene.scene_generation,
        created_turn_number=2,
        expires_after_turn_number=14,
    )
    next_scene = repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=second_location.id,
        source_message_id=message_id,
    )

    assert next_scene.scene_generation == scene.scene_generation + 2
    assert repositories.get_context_source(next_scratch.id) is None


def test_repositories_archive_scene_scratch_after_turn_ttl(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    scene = repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="The beacon lens is warm.",
        source_message_id=message_id,
    )
    scratch = repositories.upsert_context_source(
        save_id=save_id,
        source_type="observation",
        source_id="scratch-observation",
        title="Temporary lens state",
        body="The lens is warm.",
        metadata={"curation_action": "scene_scratch"},
        scene_snapshot_id=scene.id,
        scene_generation=scene.scene_generation,
        created_turn_number=1,
        expires_after_turn_number=13,
    )

    archived_ids = repositories.archive_stale_scene_scratch(
        save_id=save_id,
        current_scene_snapshot_id=scene.id,
        current_scene_generation=scene.scene_generation,
        current_turn_number=13,
    )

    assert archived_ids == frozenset({scratch.id})
    assert repositories.get_context_source(scratch.id) is None
    assert [
        marker.id
        for marker in repositories.list_curated_observation_source_markers(save_id)
    ] == [scratch.id]


def test_repositories_hide_expired_scene_scratch_from_ordinary_reads(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    scene = repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="The beacon lens is warm.",
        source_message_id=message_id,
    )
    scratch = repositories.upsert_context_source(
        save_id=save_id,
        source_type="observation",
        source_id="scratch-observation",
        title="Temporary lens state",
        body="The lens is warm.",
        metadata={"curation_action": "scene_scratch"},
        scene_snapshot_id=scene.id,
        scene_generation=scene.scene_generation,
        created_turn_number=0,
        expires_after_turn_number=1,
    )

    repositories.append_message(
        save_id=save_id,
        role="narrator",
        body="The scene advances.",
    )

    assert repositories.get_context_source(scratch.id) is None
    assert repositories.list_context_sources(save_id) == []


def test_repositories_delete_scene_snapshot_archives_bound_scratch(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    scene = repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="The beacon lens is warm.",
        source_message_id=message_id,
    )
    scratch = repositories.upsert_context_source(
        save_id=save_id,
        source_type="observation",
        source_id="scratch-observation",
        title="Temporary lens state",
        body="The lens is warm.",
        metadata={"curation_action": "scene_scratch"},
        scene_snapshot_id=scene.id,
        scene_generation=scene.scene_generation,
    )

    deleted_id = repositories.delete_scene_snapshot(save_id)

    assert deleted_id == scene.id
    assert repositories.get_scene_snapshot(save_id) is None
    assert repositories.get_context_source(scratch.id) is None


def test_repositories_list_protected_context_sources(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    obligation = repositories.upsert_context_source(
        save_id=save.id,
        source_type="open_obligation",
        source_id="thread-hold-the-gate",
        title="Hold the gate",
        body="Hold the gate until the signal lens wakes.",
        metadata={"importance": 0.8},
    )
    voice = repositories.upsert_context_source(
        save_id=save.id,
        source_type="character_voice",
        source_id="character-ilyra",
        title="voice:Ilyra",
        body="Ilyra speaks in clipped military images.",
        metadata={"importance": 0.7},
    )
    always_include = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-promise",
        title="promise",
        body="Mara promised to keep the beacon dark.",
        metadata={"always_include_reason": "promise", "importance": 0.95},
    )
    normal = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-normal",
        title="normal",
        body="A normal memory is only searchable by text.",
        metadata={"importance": 0.1},
    )

    protected = repositories.list_protected_context_sources(save.id, limit=8)

    assert [record.source_id for record in protected] == [
        obligation.source_id,
        voice.source_id,
        always_include.source_id,
    ]
    assert normal.source_id not in {record.source_id for record in protected}


def test_repositories_persist_normalized_context_registry(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)

    location = repositories.add_location(
        save_id=save_id,
        name="Beacon Gallery",
        aliases=["upper lens room"],
        description="A hot room above the keep wall.",
        visual_description="Red glass, ash-streaked windows, brass gears.",
        connections=["Gatehouse"],
        status="sealed",
        hazards=["cracked lens"],
        source_message_id=source_message_id,
        locked_fields=["name"],
    )
    character = repositories.add_character(
        save_id=save_id,
        name="Captain Ilyra",
        aliases=["captain"],
        role="Watch captain",
        known_state="trusted",
        met=True,
        appearance="Ash-gray cloak and bronze signal horn.",
        visual_notes="Tall silhouette, red lamp glow on armor.",
        current_clothing="Fresh linen shirt under a borrowed green raincoat.",
        personality="decisive",
        voice="low and clipped",
        relationships={"player": "ally"},
        goals="Keep the beacon lit until dawn.",
        motivations="Protect the outer villages from the ash riders.",
        current_intent="Delay anyone who threatens the lens repair.",
        boundaries="Will not abandon the tower while the lens is unstable.",
        attitude_toward_player="Trusts the player under pressure.",
        cooperation_conditions="Will share the failsafe only after seeing proof.",
        status="guarding the lens",
        location_id=location.id,
        private_notes="Knows the beacon is failing.",
        source_message_id=source_message_id,
        locked_fields=["voice"],
    )
    assert character.history == "trusted"
    snapshot = repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=location.id,
        situation="The beacon is overheating.",
        objective="Stop the lens from cracking.",
        in_world_time="midnight",
        weather="ash storm",
        mood="urgent",
        nearby_objects=["signal horn", "oil lever"],
        hazards=["red-hot glass"],
        present_character_ids=[character.id],
        source_message_id=source_message_id,
        locked_fields=["objective"],
    )
    thread = repositories.add_active_thread(
        save_id=save_id,
        title="Save the beacon",
        description="The red lens may shatter before dawn.",
        priority=8,
        visibility="public",
        related_entities=[f"location:{location.id}", f"character:{character.id}"],
        source_message_id=source_message_id,
        locked_fields=["title"],
    )
    link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character.id,
        target_type="location",
        target_id=location.id,
        relation="present_at",
        source_message_id=source_message_id,
    )
    duplicate_link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character.id,
        target_type="location",
        target_id=location.id,
        relation="present_at",
        source_message_id=source_message_id,
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="update",
        entity_type="location",
        entity_id=location.id,
        field_path="status",
        proposed_value="unstable",
        reason="Narrator described the gallery shaking.",
        confidence=0.82,
        source_message_ids=[source_message_id],
    )
    audit = repositories.add_context_update_audit(
        save_id=save_id,
        suggestion_id=suggestion.id,
        operation="queued",
        entity_type="location",
        entity_id=location.id,
        field_path="status",
        before="sealed",
        after="unstable",
        reason="Queued for review.",
        confidence=0.82,
        source_message_ids=[source_message_id],
    )

    assert repositories.get_scene_snapshot(save_id) == snapshot
    assert repositories.list_locations(save_id) == [location]
    assert repositories.list_characters(save_id) == [character]
    saved_character = repositories.list_characters(save_id)[0]
    assert saved_character.goals == "Keep the beacon lit until dawn."
    assert saved_character.current_intent == (
        "Delay anyone who threatens the lens repair."
    )
    assert saved_character.current_clothing == (
        "Fresh linen shirt under a borrowed green raincoat."
    )
    assert saved_character.cooperation_conditions == (
        "Will share the failsafe only after seeing proof."
    )
    assert character.protected_from_maintenance is False
    assert repositories.list_active_threads(save_id) == [thread]
    assert repositories.list_entity_links(save_id) == [link]
    assert duplicate_link.id == link.id
    assert duplicate_link.source_message_id == source_message_id
    assert repositories.list_context_update_suggestions(save_id) == [suggestion]
    assert repositories.list_context_update_audit(save_id) == [audit]

    applied = repositories.update_context_update_suggestion_status(
        suggestion.id,
        status="applied",
    )

    assert applied.status == "applied"
    assert applied.resolved_at is not None
    superseded = repositories.update_context_update_suggestion_status(
        suggestion.id,
        status="superseded",
    )
    expired = repositories.update_context_update_suggestion_status(
        suggestion.id,
        status="expired",
    )
    pending = repositories.update_context_update_suggestion_status(
        suggestion.id,
        status="pending",
    )

    assert superseded.resolved_at is not None
    assert expired.resolved_at is not None
    assert pending.resolved_at is None

    repositories.update_location(replace(location, status="unstable"))
    updated_location = repositories.get_location(location.id)
    assert updated_location is not None
    assert updated_location.status == "unstable"

    repositories.archive_location(location.id)
    repositories.archive_character(character.id)
    repositories.archive_active_thread(thread.id)

    assert repositories.list_locations(save_id) == []
    assert repositories.get_location(location.id) is None
    assert repositories.list_characters(save_id) == []
    assert repositories.get_character(character.id) is None
    assert repositories.list_active_threads(save_id) == []
    assert repositories.get_active_thread(thread.id) is None


def test_repositories_bulk_update_context_update_suggestion_statuses(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    first = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="update",
        entity_type="location",
        entity_id="location-1",
        field_path="status",
        proposed_value="unstable",
        source_message_ids=[source_message_id],
    )
    second = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="update",
        entity_type="location",
        entity_id="location-1",
        field_path="description",
        proposed_value="glass under stress",
        source_message_ids=[source_message_id],
    )

    dismissed = repositories.update_context_update_suggestion_statuses(
        [second.id, first.id],
        status="dismissed",
    )

    assert [row.id for row in dismissed] == [first.id, second.id]
    assert {row.status for row in dismissed} == {"dismissed"}
    assert all(row.resolved_at is not None for row in dismissed)

    reset = repositories.update_context_update_suggestion_statuses(
        [first.id, second.id],
        status="pending",
    )

    assert {row.status for row in reset} == {"pending"}
    assert all(row.resolved_at is None for row in reset)


def test_repositories_expire_stale_context_update_suggestions(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    stale = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="update",
        entity_type="location",
        entity_id="location-1",
        field_path="status",
        proposed_value="unstable",
        source_message_ids=[source_message_id],
    )
    recent = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="update",
        entity_type="location",
        entity_id="location-1",
        field_path="description",
        proposed_value="glass under stress",
        source_message_ids=[source_message_id],
    )
    already_resolved = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="update",
        entity_type="active_thread",
        entity_id="thread-1",
        field_path="title",
        proposed_value="old thread",
        source_message_ids=[source_message_id],
    )
    repositories.update_context_update_suggestion_status(
        already_resolved.id,
        status="rejected",
    )
    repositories.connection.execute(
        """
        UPDATE context_update_suggestions
        SET created_at = datetime('now', '-31 days')
        WHERE id IN (?, ?)
        """,
        (stale.id, already_resolved.id),
    )

    expired = repositories.expire_stale_context_update_suggestions(
        save_id,
        older_than_days=30,
    )

    assert [row.id for row in expired] == [stale.id]
    statuses = {
        row.id: row.status
        for row in repositories.list_context_update_suggestions(save_id)
    }
    assert statuses[stale.id] == "expired"
    assert statuses[recent.id] == "pending"
    assert statuses[already_resolved.id] == "rejected"


def test_repositories_persist_character_maintenance_protection(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _source_message_id = _persist_repository_save(repositories)

    default_character = repositories.add_character(
        save_id=save_id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    protected_character = repositories.add_character(
        save_id=save_id,
        name="Oracle of Glass",
        protected_from_maintenance=True,
        character_id="character-oracle",
    )

    assert default_character.protected_from_maintenance is False
    assert protected_character.protected_from_maintenance is True
    assert repositories.get_character(protected_character.id) == protected_character
    assert {
        character.id: character.protected_from_maintenance
        for character in repositories.list_characters(save_id)
    } == {
        default_character.id: False,
        protected_character.id: True,
    }

    updated = repositories.update_character(
        replace(protected_character, protected_from_maintenance=False)
    )

    assert updated.protected_from_maintenance is False


def test_repositories_enforce_single_player_character_and_scene_presence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _source_message_id = _persist_repository_save(repositories)
    first = repositories.add_character(
        save_id=save_id,
        name="Mara Voss",
        is_player_character=True,
        character_id="character-mara",
    )
    second = repositories.add_character(
        save_id=save_id,
        name="Iris Vale",
        character_id="character-iris",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Mara watches the beacon.",
        present_character_ids=[],
        snapshot_id="snapshot-main",
    )

    first_snapshot = repositories.get_scene_snapshot(save_id)
    assert first.is_player_character is True
    assert first.protected_from_maintenance is True
    assert first_snapshot is not None
    assert first_snapshot.present_character_ids == [first.id]

    updated_second = repositories.update_character(
        replace(second, is_player_character=True)
    )
    second_snapshot = repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Iris takes point.",
        present_character_ids=[],
        snapshot_id="snapshot-main",
    )

    updated_first = repositories.get_character(first.id)
    assert updated_first is not None
    assert updated_second.is_player_character is True
    assert updated_second.protected_from_maintenance is True
    assert updated_first.is_player_character is False
    assert second_snapshot.present_character_ids == [second.id]


def test_repository_rejects_character_add_with_location_from_other_save(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    other_save_id, _ = _persist_repository_save(repositories)
    other_location = repositories.add_location(
        save_id=other_save_id,
        name="Other Gatehouse",
    )

    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        repositories.add_character(
            save_id=save_id,
            name="Captain Ilyra",
            location_id=other_location.id,
            source_message_id=source_message_id,
        )

    assert repositories.list_characters(save_id) == []


def test_repository_rejects_character_update_with_location_from_other_save(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    other_save_id, _ = _persist_repository_save(repositories)
    location = repositories.add_location(
        save_id=save_id,
        name="Beacon Gallery",
    )
    other_location = repositories.add_location(
        save_id=other_save_id,
        name="Other Gatehouse",
    )
    character = repositories.add_character(
        save_id=save_id,
        name="Captain Ilyra",
        location_id=location.id,
        source_message_id=source_message_id,
    )

    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        repositories.update_character(
            replace(character, location_id=other_location.id)
        )

    assert repositories.get_character(character.id) == character


def test_repository_rejects_scene_snapshot_upsert_with_location_from_other_save(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    other_save_id, _ = _persist_repository_save(repositories)
    other_location = repositories.add_location(
        save_id=other_save_id,
        name="Other Gatehouse",
    )

    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        repositories.upsert_scene_snapshot(
            save_id=save_id,
            current_location_id=other_location.id,
            situation="The wrong gatehouse bleeds across saves.",
            source_message_id=source_message_id,
        )

    assert repositories.get_scene_snapshot(save_id) is None


def test_repository_rejects_location_add_with_parent_from_other_save(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    other_save_id, _ = _persist_repository_save(repositories)
    other_parent = repositories.add_location(
        save_id=other_save_id,
        name="Other Gatehouse",
    )

    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        repositories.add_location(
            save_id=save_id,
            name="Beacon Gallery",
            parent_location_id=other_parent.id,
        )

    assert repositories.list_locations(save_id) == []


def test_repository_rejects_location_update_with_parent_from_other_save(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _persist_repository_save(repositories)
    other_save_id, _ = _persist_repository_save(repositories)
    parent = repositories.add_location(
        save_id=save_id,
        name="Gatehouse",
    )
    location = repositories.add_location(
        save_id=save_id,
        name="Beacon Gallery",
        parent_location_id=parent.id,
    )
    other_parent = repositories.add_location(
        save_id=other_save_id,
        name="Other Gatehouse",
    )

    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        repositories.update_location(
            replace(location, parent_location_id=other_parent.id)
        )

    assert repositories.get_location(location.id) == location


def test_location_delete_trigger_nulls_same_save_references(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    location = repositories.add_location(
        save_id=save_id,
        name="Gatehouse",
        source_message_id=source_message_id,
    )
    child_location = repositories.add_location(
        save_id=save_id,
        name="Beacon Gallery",
        parent_location_id=location.id,
        source_message_id=source_message_id,
    )
    character = repositories.add_character(
        save_id=save_id,
        name="Captain Ilyra",
        location_id=location.id,
        source_message_id=source_message_id,
    )
    snapshot = repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=location.id,
        situation="The beacon is overheating.",
        source_message_id=source_message_id,
    )

    repositories.connection.execute(
        "DELETE FROM locations WHERE save_id = ? AND id = ?",
        (save_id, location.id),
    )

    reference_row = repositories.connection.execute(
        """
        SELECT
            scene_snapshots.current_location_id,
            child_locations.parent_location_id,
            characters.location_id
        FROM scene_snapshots
        JOIN locations AS child_locations
            ON child_locations.id = ?
        JOIN characters
            ON characters.id = ?
        WHERE scene_snapshots.id = ?
            AND scene_snapshots.save_id = ?
            AND child_locations.save_id = ?
            AND characters.save_id = ?
        """,
        (
            child_location.id,
            character.id,
            snapshot.id,
            save_id,
            save_id,
            save_id,
        ),
    ).fetchone()

    assert reference_row is not None
    assert reference_row["current_location_id"] is None
    assert reference_row["parent_location_id"] is None
    assert reference_row["location_id"] is None


def test_repositories_persist_provider_jobs_and_media(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="character_interaction",
        title="The Archivist",
        premise="A forbidden archive interview.",
        player_role="Investigator",
        content={"npc": "Archivist"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Interview")
    source_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="Show me the sealed stacks.",
        provider="venice",
        model="venice-chat",
        token_estimate=12,
    )

    repositories.save_provider_model(
        provider="openrouter",
        model_id="anthropic/claude-sonnet",
        display_name="Claude Sonnet",
        capabilities=["chat", "summarization"],
        context_window=200_000,
        refreshed_at="2026-05-12T12:00:00Z",
    )
    provider_models = repositories.list_provider_models("openrouter")

    assert len(provider_models) == 1
    assert provider_models[0].model_id == "anthropic/claude-sonnet"
    assert provider_models[0].capabilities == ["chat", "summarization"]
    assert provider_models[0].context_window == 200_000

    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-sonnet",
    )
    preference = repositories.get_model_preference("chat")

    assert preference is not None
    assert preference.provider == "openrouter"
    assert preference.model_id == "anthropic/claude-sonnet"

    job = repositories.create_job(
        save_id=save.id,
        type="image_generation",
        status="queued",
        payload={"prompt": "candlelit archive"},
    )
    updated_job = repositories.update_job(
        job.id,
        status="succeeded",
        result={"asset_path": "media/archive.png"},
        error=None,
    )

    assert updated_job.id == job.id
    assert updated_job.status == "succeeded"
    assert updated_job.payload == {"prompt": "candlelit archive"}
    assert updated_job.result == {"asset_path": "media/archive.png"}
    assert updated_job.error is None
    assert updated_job.completed_at is not None

    media_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=source_message.id,
        type="image",
        path="media/archive.png",
        thumbnail_path="media/archive.thumb.png",
        prompt="candlelit archive",
        provider="venice",
        model="venice-image",
        status="ready",
    )
    media_assets = repositories.list_media_assets(save.id)

    assert [asset.id for asset in media_assets] == [media_asset.id]
    assert media_assets[0].source_message_id == source_message.id
    assert media_assets[0].provider == "venice"
    assert media_assets[0].model == "venice-image"
    assert media_assets[0].created_at is not None


def test_repositories_persist_image_media_with_new_optional_fields(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)

    media_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/archive.png",
        thumbnail_path="media/archive.thumb.png",
        prompt="candlelit archive",
        provider="venice",
        model="venice-image",
        status="ready",
        mime_type="image/png",
        metadata={"seed": 42, "revised_prompt": "warmer candlelit archive"},
    )

    media_assets = repositories.list_media_assets(save_id)

    assert [asset.id for asset in media_assets] == [media_asset.id]
    assert media_assets[0].type == "image"
    assert media_assets[0].mime_type == "image/png"
    assert json.loads(media_assets[0].metadata_json) == {
        "seed": 42,
        "revised_prompt": "warmer candlelit archive",
    }
    assert media_assets[0].source_media_asset_id is None


def test_repositories_get_media_asset_is_save_scoped_and_active(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    other_save_id, other_source_message_id = _persist_repository_save(repositories)
    active_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/active.png",
        thumbnail_path="media/thumbnails/active.png",
        prompt="active image",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )
    archived_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/archived.png",
        thumbnail_path=None,
        prompt="archived image",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )
    other_save_asset = repositories.create_media_asset(
        save_id=other_save_id,
        source_message_id=other_source_message_id,
        type="image",
        path="media/other.png",
        thumbnail_path=None,
        prompt="other save image",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )

    repositories.archive_media_asset(
        save_id=save_id,
        media_asset_id=archived_asset.id,
    )

    fetched = repositories.get_media_asset(
        save_id=save_id,
        media_asset_id=active_asset.id,
    )

    assert fetched is not None
    assert fetched.id == active_asset.id
    assert fetched.thumbnail_path == "media/thumbnails/active.png"
    assert (
        repositories.get_media_asset(
            save_id=save_id,
            media_asset_id=other_save_asset.id,
        )
        is None
    )
    assert (
        repositories.get_media_asset(
            save_id=save_id,
            media_asset_id=archived_asset.id,
        )
        is None
    )


@pytest.mark.parametrize("mime_type", ["image/png", "image/jpeg", "image/webp"])
def test_repositories_allow_supported_image_media_mime_types(
    repositories: PersistenceRepositories,
    mime_type: str,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)

    media_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path=f"media/archive-{mime_type.rsplit('/', 1)[1]}.img",
        thumbnail_path=None,
        prompt="candlelit archive",
        provider="venice",
        model="venice-image",
        status="ready",
        mime_type=mime_type,
    )

    [persisted] = repositories.list_media_assets(save_id)
    assert persisted.id == media_asset.id
    assert persisted.mime_type == mime_type


def test_repositories_allow_inert_media_mime_type(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)

    repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/imported-unknown.bin",
        thumbnail_path=None,
        prompt="imported media",
        provider="bundle-import",
        model="bundle-import",
        status="succeeded",
        mime_type="application/octet-stream",
    )

    [persisted] = repositories.list_media_assets(save_id)
    assert persisted.mime_type == "application/octet-stream"


@pytest.mark.parametrize("mime_type", ["text/html", "image/svg+xml", "video/mp4"])
def test_repositories_reject_unsupported_image_media_mime_types(
    repositories: PersistenceRepositories,
    mime_type: str,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)

    with pytest.raises(ValueError, match="Unsupported image mime type"):
        repositories.create_media_asset(
            save_id=save_id,
            source_message_id=source_message_id,
            type="image",
            path="media/imported-active.bin",
            thumbnail_path=None,
            prompt="imported media",
            provider="bundle-import",
            model="bundle-import",
            status="succeeded",
            mime_type=mime_type,
        )

    assert repositories.list_media_assets(save_id) == []


def test_repositories_persist_video_media_with_source_asset_and_metadata(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    source_image = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/archive.png",
        thumbnail_path="media/archive.thumb.png",
        prompt="candlelit archive",
        provider="venice",
        model="venice-image",
        status="succeeded",
        mime_type="image/png",
    )

    video_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        source_media_asset_id=source_image.id,
        type="video",
        mime_type="video/webm",
        path="media/archive-animation.webm",
        thumbnail_path=None,
        prompt="animate the candle smoke and drifting dust",
        provider="fake-video",
        model="fake-image-plus-text-video",
        status="succeeded",
        metadata={"duration_seconds": 4, "source_flow": "image_plus_text_to_video"},
    )

    media_assets = repositories.list_media_assets(save_id)

    assert [asset.id for asset in media_assets] == [source_image.id, video_asset.id]
    assert media_assets[1].type == "video"
    assert media_assets[1].mime_type == "video/webm"
    assert media_assets[1].source_media_asset_id == source_image.id
    assert json.loads(media_assets[1].metadata_json) == {
        "duration_seconds": 4,
        "source_flow": "image_plus_text_to_video",
    }


def test_repositories_archive_media_asset_hides_it_from_save_reads(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    first_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/archive-1.png",
        thumbnail_path=None,
        prompt="candlelit archive",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )
    second_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="video",
        path="media/archive-2.mp4",
        thumbnail_path=None,
        prompt="animated candlelit archive",
        provider="fake-video",
        model="fake-video-model",
        status="succeeded",
        mime_type="video/mp4",
    )

    archived = repositories.archive_media_asset(
        save_id=save_id,
        media_asset_id=first_asset.id,
    )

    assert archived is not None
    assert archived.id == first_asset.id
    assert archived.archived_at is not None
    assert [asset.id for asset in repositories.list_media_assets(save_id)] == [
        second_asset.id
    ]


def test_repositories_can_list_archived_media_assets_for_maintenance(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    active_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/active.png",
        thumbnail_path=None,
        prompt="active image",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )
    archived_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/archived.png",
        thumbnail_path=None,
        prompt="archived image",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )

    repositories.archive_media_asset(
        save_id=save_id,
        media_asset_id=archived_asset.id,
    )

    assert [asset.id for asset in repositories.list_media_assets(save_id)] == [
        active_asset.id
    ]
    assert [asset.id for asset in repositories.list_all_media_assets(save_id)] == [
        active_asset.id,
        archived_asset.id,
    ]
    assert repositories.list_all_media_assets(save_id)[1].archived_at is not None


def test_repositories_archive_media_asset_hides_derived_media_assets(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    source_image = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/archive.png",
        thumbnail_path=None,
        prompt="candlelit archive",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )
    repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        source_media_asset_id=source_image.id,
        type="video",
        path="media/archive.mp4",
        thumbnail_path=None,
        prompt="animated candlelit archive",
        provider="fake-video",
        model="fake-video-model",
        status="succeeded",
        mime_type="video/mp4",
    )

    repositories.archive_media_asset(
        save_id=save_id,
        media_asset_id=source_image.id,
    )

    assert repositories.list_media_assets(save_id) == []


def test_repositories_archive_media_asset_only_leaves_derived_media_assets(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    source_image = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/archive.png",
        thumbnail_path=None,
        prompt="candlelit archive",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )
    derived_video = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        source_media_asset_id=source_image.id,
        type="video",
        path="media/archive.mp4",
        thumbnail_path=None,
        prompt="animated candlelit archive",
        provider="fake-video",
        model="fake-video-model",
        status="succeeded",
        mime_type="video/mp4",
    )

    archived = repositories.archive_media_asset_only(
        save_id=save_id,
        media_asset_id=source_image.id,
    )

    assert archived is not None
    assert archived.id == source_image.id
    assert archived.archived_at is not None
    assert [asset.id for asset in repositories.list_media_assets(save_id)] == [
        derived_video.id
    ]


def test_repositories_replace_media_asset_source_references(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    old_source = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/old-source.png",
        thumbnail_path=None,
        prompt="old source",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )
    new_source = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/new-source.png",
        thumbnail_path=None,
        prompt="new source",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )
    child = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        source_media_asset_id=old_source.id,
        type="image",
        path="media/child.png",
        thumbnail_path=None,
        prompt="child image",
        provider="venice",
        model="venice-image",
        status="succeeded",
        metadata={
            "source_media_asset_id": old_source.id,
            "source_media_asset_ids": [old_source.id, "other-media"],
            "source_character_reference_asset_id": old_source.id,
            "source_character_reference_asset_ids": [old_source.id],
            "regenerated_from_media_asset_id": old_source.id,
        },
    )

    updated = repositories.replace_media_asset_source_references(
        save_id=save_id,
        old_media_asset_id=old_source.id,
        new_media_asset_id=new_source.id,
    )

    assert updated == 1
    fetched_child = repositories.get_media_asset(
        save_id=save_id,
        media_asset_id=child.id,
    )
    assert fetched_child is not None
    assert fetched_child.source_media_asset_id == new_source.id
    metadata = json.loads(fetched_child.metadata_json)
    assert metadata["source_media_asset_id"] == new_source.id
    assert metadata["source_media_asset_ids"] == [new_source.id, "other-media"]
    assert metadata["source_character_reference_asset_id"] == new_source.id
    assert metadata["source_character_reference_asset_ids"] == [new_source.id]
    assert metadata["regenerated_from_media_asset_id"] == old_source.id


def test_repositories_archive_media_asset_is_scoped_to_save(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    other_save_id, _other_source_message_id = _persist_repository_save(repositories)
    media_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type="image",
        path="media/archive.png",
        thumbnail_path=None,
        prompt="candlelit archive",
        provider="venice",
        model="venice-image",
        status="succeeded",
    )

    archived = repositories.archive_media_asset(
        save_id=other_save_id,
        media_asset_id=media_asset.id,
    )

    assert archived is None
    assert [asset.id for asset in repositories.list_media_assets(save_id)] == [
        media_asset.id
    ]
    assert repositories.list_media_assets(other_save_id) == []


@pytest.mark.parametrize(
    ("path", "mime_type", "message"),
    [
        ("media/archive.png", "image/png", "video mime type"),
        ("../escape/archive.mp4", "video/mp4", "safe relative"),
        ("/tmp/archive.mp4", "video/mp4", "safe relative"),
    ],
)
def test_repositories_validate_video_media_mime_type_and_safe_relative_path(
    repositories: PersistenceRepositories,
    path: str,
    mime_type: str,
    message: str,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)

    with pytest.raises(ValueError, match=message):
        repositories.create_media_asset(
            save_id=save_id,
            source_message_id=source_message_id,
            type="video",
            path=path,
            thumbnail_path=None,
            prompt="unsafe archive animation",
            provider="fake-video",
            model="fake-video-model",
            status="succeeded",
            mime_type=mime_type,
        )

    assert repositories.list_media_assets(save_id) == []


def test_repositories_persist_loss_conditions_audit_and_outcomes(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"starting_location": "Gatehouse"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I keep climbing after the warning bell cracks.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens splits and ash fills the stairwell.",
        provider="fake",
        model="fake-chat",
    )

    condition = repositories.add_loss_condition(
        condition_id="loss-beacon-collapse",
        save_id=save.id,
        name="Beacon collapse",
        description="The beacon may collapse if the cracked lens is ignored.",
        status="active",
        source="structured",
    )
    updated_condition = repositories.update_loss_condition(
        condition_id=condition.id,
        name="Beacon collapse",
        description="The beacon is collapsing after the lens split.",
        status="triggered",
    )
    change = repositories.add_loss_condition_change(
        change_id="loss-change-beacon-collapse",
        save_id=save.id,
        condition_id=condition.id,
        operation="update",
        before={
            "description": "The beacon may collapse if the cracked lens is ignored.",
            "status": "active",
        },
        after={
            "description": "The beacon is collapsing after the lens split.",
            "status": "triggered",
        },
        reason="The current narrator turn explicitly triggered the condition.",
        provider="fake",
        model="fake-loss-model",
        source_message_id=narrator_message.id,
    )
    outcome = repositories.create_loss_outcome(
        outcome_id="loss-outcome-beacon-collapse",
        save_id=save.id,
        condition_id=condition.id,
        condition_name=updated_condition.name,
        triggering_message_id=narrator_message.id,
        explanation="The tower collapses into the ash storm.",
        confidence=0.91,
        evidence={
            "items": [
                {
                    "source_message_id": narrator_message.id,
                    "quote": "ash fills the stairwell",
                }
            ]
        },
        provider="fake",
        model="fake-loss-model",
    )

    conditions = repositories.list_loss_conditions(save.id)
    changes = repositories.list_loss_condition_changes(save.id)
    outcomes = repositories.list_loss_outcomes(save.id)

    assert condition.id == updated_condition.id
    assert conditions == [updated_condition]
    assert conditions[0].name == "Beacon collapse"
    assert conditions[0].status == "triggered"
    assert conditions[0].source == "structured"
    assert len(changes) == 1
    assert changes[0].id == change.id
    assert changes[0].condition_id == condition.id
    assert changes[0].source_message_id == narrator_message.id
    assert changes[0].operation == "update"
    assert changes[0].after is not None
    assert changes[0].after["status"] == "triggered"
    assert outcomes == [outcome]
    assert repositories.get_active_loss_outcome(save.id) == outcome
    assert outcome.confidence == 0.91
    assert outcome.evidence["items"] == [
        {"source_message_id": narrator_message.id, "quote": "ash fills the stairwell"}
    ]

    archived_ids = repositories.archive_loss_outcomes_for_messages(
        save_id=save.id,
        message_ids=frozenset({narrator_message.id}),
    )

    assert archived_ids == frozenset({outcome.id})
    assert repositories.get_active_loss_outcome(save.id) is None
    archived_outcome = repositories.list_loss_outcomes(
        save.id,
        include_archived=True,
    )[0]
    assert archived_outcome.active is False
    assert archived_outcome.archived_at is not None


def test_repositories_persist_conditionless_terminal_outcomes(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"starting_location": "Gatehouse"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Mara dies sealing the gate. The keep is saved.",
        provider="fake",
        model="fake-chat",
    )

    outcome = repositories.create_loss_outcome(
        outcome_id="terminal-outcome-mission-complete",
        save_id=save.id,
        condition_id=None,
        condition_name="Mission complete",
        triggering_message_id=narrator_message.id,
        explanation="Mara dies sealing the gate and the mission is complete.",
        confidence=0.96,
        evidence={
            "items": [
                {
                    "source_message_id": narrator_message.id,
                    "quote": "The keep is saved.",
                }
            ],
            "epilogue": "The gate holds.",
        },
        provider="fake",
        model="fake-outcome-model",
        outcome_type="player_dead",
    )

    listed = repositories.list_loss_outcomes(save.id)

    assert listed == [outcome]
    assert repositories.get_active_loss_outcome(save.id) == outcome
    assert outcome.condition_id is None
    assert outcome.condition_name == "Mission complete"
    assert outcome.outcome_type == "player_dead"
    assert outcome.epilogue == "The gate holds."


def test_repositories_rebuild_loss_conditions_from_unarchived_change_audit(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"starting_location": "Gatehouse"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    first_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The eastern gate is buckling but still holding.",
        provider="fake",
        model="fake-chat",
    )
    later_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The eastern gate breaks open.",
        provider="fake",
        model="fake-chat",
    )
    condition = repositories.add_loss_condition(
        condition_id="loss-gate",
        save_id=save.id,
        name="Eastern gate breach",
        description="The eastern gate is buckling.",
        status="active",
        source="structured",
    )
    repositories.add_loss_condition_change(
        save_id=save.id,
        condition_id=condition.id,
        operation="add",
        before=None,
        after={
            "id": condition.id,
            "name": "Eastern gate breach",
                "description": "The eastern gate is buckling.",
            "status": "active",
            "source": "structured",
        },
        reason="Seeded from the first warning.",
        provider="fake",
        model="fake-loss-model",
        source_message_id=first_narrator.id,
    )
    repositories.update_loss_condition(
        condition_id=condition.id,
        name="Eastern gate breach",
        description="The eastern gate has broken open.",
        status="triggered",
    )
    later_change = repositories.add_loss_condition_change(
        save_id=save.id,
        condition_id=condition.id,
        operation="update",
        before={"status": "active"},
        after={
            "id": condition.id,
            "name": "Eastern gate breach",
            "description": "The eastern gate has broken open.",
            "status": "triggered",
            "source": "structured",
        },
        reason="The gate breach triggered on the later turn.",
        provider="fake",
        model="fake-loss-model",
        source_message_id=later_narrator.id,
    )
    archived_change_ids = repositories.archive_loss_condition_changes_for_messages(
        save_id=save.id,
        message_ids=frozenset({later_narrator.id}),
    )

    all_changes = repositories.list_loss_condition_changes(
        save.id,
        include_archived=True,
    )
    change_by_id = {row.id: row for row in all_changes}
    assert archived_change_ids == frozenset({later_change.id})
    assert change_by_id[later_change.id].archived_at is not None


def test_delete_save_removes_save_owned_records_without_touching_other_saves(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"starting_location": "Gatehouse"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    other_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Dawn Watch",
    )
    repositories.set_app_setting(
        save_scenario_evolution_turn_interval_setting_key(save.id),
        3,
    )
    repositories.set_app_setting(
        save_scenario_evolution_turn_interval_setting_key(other_save.id),
        5,
    )
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens cracks.",
    )
    other_message = repositories.append_message(
        save_id=other_save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The other watch remains intact.",
    )
    repositories.add_message_revision(
        save_id=save.id,
        message_id=message.id,
        previous_body="The beacon lens holds.",
        new_body=message.body,
        diff_unified="diff",
        reconciliation_status="succeeded",
    )
    other_revision = repositories.add_message_revision(
        save_id=other_save.id,
        message_id=other_message.id,
        previous_body="The other watch wavers.",
        new_body=other_message.body,
        diff_unified="other diff",
        reconciliation_status="succeeded",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Beacon Gallery"},
        category="scene",
        source_message_id=message.id,
    )
    repositories.add_state_change(
        save_id=save.id,
        operation="upsert",
        state_key="scene.location",
        after_json='{"name":"Beacon Gallery"}',
        source_message_id=message.id,
    )
    repositories.add_memory(
        save_id=save.id,
        body="The beacon lens cracked.",
        tags=["beacon"],
        source_message_id=message.id,
    )
    repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=message.id,
        covers_message_end_id=message.id,
        body="The lens cracked during the watch.",
        provider="fake",
        model="fake-chat",
    )
    repositories.record_save_scenario_evolution(
        save_id=save.id,
        title="Ashfall Keep, Changed",
        premise="The cracked beacon makes the keep vulnerable.",
        player_role="Warden",
        content={"starting_location": "Beacon Gallery"},
        reason="The beacon changed.",
        provider="fake",
        model="fake-chat",
        source_message_id=message.id,
    )
    repositories.create_job(
        save_id=save.id,
        type="image_generation",
        status="queued",
        payload={"prompt": "cracked beacon"},
    )
    repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path="generated/beacon.png",
        thumbnail_path="generated/beacon.thumb.png",
        prompt="cracked beacon",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="scenario_section",
        source_id="scenario:ashfall:section:locations",
        title="locations",
        body="Beacon Gallery",
    )
    repositories.add_context_observation(
        save_id=save.id,
        observation_type="scene_fact",
        claim="The beacon lens is cracked.",
        status="accepted",
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=message.id,
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        location_id=location.id,
        source_message_id=message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="The lens is cracking.",
        present_character_ids=[character.id],
        source_message_id=message.id,
    )
    repositories.add_active_thread(
        save_id=save.id,
        title="Repair the beacon",
        related_entities=[f"location:{location.id}"],
        source_message_id=message.id,
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="location",
        target_id=location.id,
        relation="present_at",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="location",
        entity_id=location.id,
        field_path="status",
        proposed_value="cracked",
        source_message_ids=[message.id],
    )
    repositories.add_context_update_audit(
        save_id=save.id,
        suggestion_id=suggestion.id,
        operation="applied",
        entity_type="location",
        entity_id=location.id,
        field_path="status",
        before="stable",
        after="cracked",
        source_message_ids=[message.id],
    )
    condition = repositories.add_loss_condition(
        save_id=save.id,
        name="Beacon collapse",
        description="The cracked beacon may collapse.",
        status="active",
        source="structured",
    )
    repositories.add_loss_condition_change(
        save_id=save.id,
        condition_id=condition.id,
        operation="add",
        before=None,
        after={"name": condition.name, "status": "active"},
        reason="The beacon cracked.",
        provider="fake",
        model="fake-loss-model",
        source_message_id=message.id,
    )
    repositories.create_loss_outcome(
        save_id=save.id,
        condition_id=condition.id,
        condition_name=condition.name,
        triggering_message_id=message.id,
        explanation="The tower collapses into ash.",
        confidence=0.93,
        evidence={"items": [{"source_message_id": message.id, "quote": "lens cracks"}]},
        provider="fake",
        model="fake-loss-model",
    )

    assert repositories.delete_save(save.id) is True

    assert repositories.get_save(save.id) is None
    fetched_other_save = repositories.get_save(other_save.id)
    assert fetched_other_save is not None
    assert fetched_other_save.id == other_save.id
    assert fetched_other_save.title == other_save.title
    assert repositories.list_messages(save.id) == []
    assert repositories.list_messages(other_save.id) == [other_message]
    assert repositories.list_message_revisions(save_id=save.id) == []
    assert repositories.list_message_revisions(save_id=other_save.id) == [
        other_revision
    ]
    assert repositories.list_world_state(save.id) == []
    assert repositories.list_state_changes(save.id) == []
    assert repositories.list_memories(save.id) == []
    assert repositories.list_summaries(save.id) == []
    assert repositories.list_save_scenario_updates(save.id, include_archived=True) == []
    assert repositories.list_media_assets(save.id) == []
    assert repositories.list_context_sources(save.id) == []
    assert repositories.list_context_observations(save.id) == []
    assert repositories.get_scene_snapshot(save.id) is None
    assert repositories.list_locations(save.id) == []
    assert repositories.list_characters(save.id) == []
    assert repositories.list_active_threads(save.id) == []
    assert repositories.list_entity_links(save.id) == []
    assert repositories.list_context_update_suggestions(save.id) == []
    assert repositories.list_context_update_audit(save.id) == []
    assert repositories.list_loss_conditions(save.id, include_archived=True) == []
    assert repositories.list_loss_condition_changes(
        save.id,
        include_archived=True,
    ) == []
    assert repositories.list_loss_outcomes(save.id, include_archived=True) == []
    assert (
        repositories.get_app_setting(
            save_scenario_evolution_turn_interval_setting_key(save.id)
        )
        is None
    )
    assert (
        repositories.get_app_setting(
            save_scenario_evolution_turn_interval_setting_key(other_save.id)
        )
        == 5
    )
    assert repositories.get_scenario(scenario.id) == scenario


def test_repositories_can_list_deleted_messages_for_audit(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    first = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I light the beacon.",
    )
    deleted = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The deleted response stays inspectable.",
        provider="fake",
        model="fake-chat",
    )

    repositories.archive_message(deleted.id)

    assert [message.id for message in repositories.list_messages(save.id)] == [
        first.id
    ]
    audit_messages = repositories.list_messages(save.id, include_deleted=True)
    assert [message.id for message in audit_messages] == [first.id, deleted.id]
    assert audit_messages[0].deleted_at is None
    assert audit_messages[1].deleted_at is not None


def test_archive_media_assets_for_messages_archives_derivatives(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    kept_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The kept scene remains visible.",
    )
    deleted_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The deleted scene has media.",
    )
    kept_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=kept_message.id,
        type="image",
        path="save/images/kept.png",
        prompt="kept image",
        provider="fake",
        model="fake-image",
        status="succeeded",
        asset_id="media-kept",
    )
    deleted_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=deleted_message.id,
        type="image",
        path="save/images/deleted.png",
        prompt="deleted image",
        provider="fake",
        model="fake-image",
        status="succeeded",
        asset_id="media-deleted",
    )
    derivative_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=deleted_message.id,
        source_media_asset_id=deleted_asset.id,
        type="video",
        path="save/videos/deleted.mp4",
        prompt="deleted animation",
        provider="fake",
        model="fake-video",
        status="succeeded",
        mime_type="video/mp4",
        asset_id="media-deleted-derivative",
    )

    archived_ids = repositories.archive_media_assets_for_messages(
        save_id=save.id,
        message_ids=frozenset({deleted_message.id}),
    )

    assert archived_ids == frozenset({deleted_asset.id, derivative_asset.id})
    assert [asset.id for asset in repositories.list_media_assets(save.id)] == [
        kept_asset.id
    ]
    archived_rows = repositories.connection.execute(
        """
        SELECT id, archived_at
        FROM media_assets
        WHERE id IN (?, ?)
        ORDER BY id
        """,
        (deleted_asset.id, derivative_asset.id),
    ).fetchall()
    assert {row["id"] for row in archived_rows} == {
        deleted_asset.id,
        derivative_asset.id,
    }
    assert all(row["archived_at"] is not None for row in archived_rows)


def test_delete_scenario_refuses_linked_saves_and_deletes_unlinked_scenarios(
    repositories: PersistenceRepositories,
) -> None:
    linked_scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"starting_location": "Gatehouse"},
    )
    save = repositories.create_save(
        scenario_id=linked_scenario.id,
        title="Night Watch",
    )
    unlinked_scenario = repositories.create_scenario(
        type="character_interaction",
        title="The Archivist",
        premise="A forbidden archive interview.",
        player_role="Investigator",
        content={"npc": "Archivist"},
    )
    repositories.set_app_setting(
        scenario_template_evolution_turn_interval_setting_key(linked_scenario.id),
        4,
    )
    repositories.set_app_setting(
        scenario_template_evolution_turn_interval_setting_key(unlinked_scenario.id),
        6,
    )

    with pytest.raises(ValueError, match="existing saves"):
        repositories.delete_scenario(linked_scenario.id)

    assert repositories.get_scenario(linked_scenario.id) == linked_scenario
    assert repositories.get_save(save.id) == save
    assert (
        repositories.get_app_setting(
            scenario_template_evolution_turn_interval_setting_key(linked_scenario.id)
        )
        == 4
    )
    assert repositories.delete_scenario(unlinked_scenario.id) is True
    assert repositories.get_scenario(unlinked_scenario.id) is None
    assert (
        repositories.get_app_setting(
            scenario_template_evolution_turn_interval_setting_key(
                unlinked_scenario.id
            )
        )
        is None
    )


def test_repository_rejects_invalid_job_statuses(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _source_message_id = _persist_repository_save(repositories)

    with pytest.raises(ValueError, match="Unknown job status: completed"):
        repositories.create_job(
            save_id=save_id,
            type="image_generation",
            status="completed",
            payload={"prompt": "candlelit archive"},
        )

    with pytest.raises(ValueError, match="Unsupported initial job status: failed"):
        repositories.create_job(
            save_id=save_id,
            type="image_generation",
            status="failed",
            payload={"prompt": "candlelit archive"},
        )

    job = repositories.create_job(
        save_id=save_id,
        type="image_generation",
        status="queued",
        payload={"prompt": "candlelit archive"},
    )

    for status in ("completed", "queued", "running", "cancelled"):
        with pytest.raises(ValueError, match="Unsupported job terminal status"):
            repositories.update_job(job.id, status=status)

    with pytest.raises(ValueError, match="Unknown job status: completed"):
        repositories.list_jobs_by_status(("queued", "completed"))

    with pytest.raises(ValueError, match="Unknown job status: completed"):
        repositories.cancel_stale_jobs(
            statuses=("running", "completed"),
            error="startup recovery",
        )


def test_repository_rejects_mutating_terminal_jobs(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _source_message_id = _persist_repository_save(repositories)
    succeeded = repositories.create_job(
        save_id=save_id,
        type="image_generation",
        status="running",
        payload={"prompt": "candlelit archive"},
    )
    failed = repositories.create_job(
        save_id=save_id,
        type="summary",
        status="running",
        payload={"save_id": save_id},
    )
    cancelled = repositories.create_job(
        save_id=save_id,
        type="chat_completion",
        status="queued",
        payload={"save_id": save_id},
    )

    succeeded = repositories.update_job(succeeded.id, status="succeeded")
    failed = repositories.update_job(failed.id, status="failed", error="nope")
    cancelled = repositories.cancel_job(cancelled.id, error="user cancelled")

    for job in (succeeded, failed, cancelled):
        with pytest.raises(ValueError, match=f"Cannot start job {job.id}"):
            repositories.start_job(job.id)
        with pytest.raises(ValueError, match=f"Cannot update terminal job {job.id}"):
            repositories.update_job(job.id, status="failed", error="still nope")
        with pytest.raises(ValueError, match=f"Cannot cancel job {job.id}"):
            repositories.cancel_job(job.id, error="too late")


def test_update_job_redacts_secret_values_before_persistence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _source_message_id = _persist_repository_save(repositories)
    job = repositories.create_job(
        save_id=save_id,
        type="image_generation",
        status="running",
        payload={"provider": "venice"},
    )

    updated_job = repositories.update_job(
        job.id,
        status="failed",
        error="provider rejected api_key: sk-secret-value and Bearer sk-bearer-token",
    )
    stored_error = repositories.connection.execute(
        "SELECT error FROM jobs WHERE id = ?",
        (job.id,),
    ).fetchone()[0]

    assert updated_job.error == (
        "provider rejected api_key: [redacted] and Bearer [redacted]"
    )
    assert stored_error == updated_job.error
    assert "sk-secret-value" not in stored_error
    assert "sk-bearer-token" not in stored_error


def test_repository_records_job_steps_with_safe_metadata_and_redacted_errors(
    repositories: PersistenceRepositories,
) -> None:
    job = repositories.create_job(
        type="chat_turn",
        status="running",
        payload={"body": "do not persist this"},
    )

    step = repositories.record_job_step(
        job_id=job.id,
        name="provider.chat",
        status="failed",
        provider="openrouter",
        model="anthropic/claude-sonnet",
        task="chat",
        duration_ms=123,
        error="provider rejected Bearer sk-secret-token",
        metadata={
            "openrouter_provider_attempt_statuses": [529, 200],
            "openrouter_provider_attempts": ["Together", "DeepInfra"],
            "openrouter_selected_model": "anthropic/claude-sonnet",
            "openrouter_selected_provider": "DeepInfra",
            "token_total": 42,
            "prompt": "unsafe",
            "reason": "Mara says a private phrase in the tower.",
            "skipped_reason": "provider_pressure",
            "cached": True,
            "temperature": 0.7,
        },
    )

    assert step.error == "provider rejected Bearer [redacted]"
    assert step.metadata == {
        "openrouter_provider_attempt_statuses": [529, 200],
        "openrouter_provider_attempts": ["Together", "DeepInfra"],
        "openrouter_selected_model": "anthropic/claude-sonnet",
        "openrouter_selected_provider": "DeepInfra",
        "skipped_reason": "provider_pressure",
        "temperature": 0.7,
        "token_total": 42,
    }
    assert "private phrase" not in repr(step.metadata)
    assert repositories.list_job_steps(job.id) == [step]


def test_repository_runtime_performance_aggregates_success_durations_only(
    repositories: PersistenceRepositories,
) -> None:
    succeeded = repositories.update_job(
        repositories.create_job(
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="succeeded",
    )
    failed = repositories.update_job(
        repositories.create_job(
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="failed",
        error="nope",
    )
    repositories.connection.execute(
        "UPDATE jobs SET duration_ms = 100 WHERE id = ?",
        (succeeded.id,),
    )
    repositories.connection.execute(
        "UPDATE jobs SET duration_ms = 999 WHERE id = ?",
        (failed.id,),
    )
    repositories.record_job_step(
        job_id=succeeded.id,
        name="state",
        status="succeeded",
        task="state_memory",
        duration_ms=40,
    )
    repositories.record_job_step(
        job_id=succeeded.id,
        name="state",
        status="failed",
        task="state_memory",
        duration_ms=400,
    )
    repositories.record_job_step(
        job_id=succeeded.id,
        name="state",
        status="deferred",
        task="state_memory",
        duration_ms=0,
    )
    repositories.record_job_step(
        job_id=succeeded.id,
        name="provider.generate_structured_output",
        status="succeeded",
        provider="fake",
        model="fake-structured",
        task="state_memory",
        duration_ms=70,
    )

    job_average = repositories.runtime_job_averages()[0]
    step_average = next(
        row for row in repositories.runtime_step_averages() if row.step_name == "state"
    )
    model_averages = repositories.runtime_model_averages()
    model_average = model_averages[0]

    assert job_average.job_type == "chat_turn"
    assert job_average.success_count == 1
    assert job_average.failed_count == 1
    assert job_average.average_duration_ms == 100
    assert step_average.success_count == 1
    assert step_average.failed_count == 1
    assert step_average.skipped_count == 1
    assert step_average.average_duration_ms == 40
    assert len(model_averages) == 1
    assert model_average.provider == "fake"
    assert model_average.model == "fake-structured"
    assert model_average.task == "state_memory"
    assert model_average.average_duration_ms == 70


def test_repository_runtime_performance_filters_window_and_reports_percentiles(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _source_message_id = _persist_repository_save(repositories)
    other_save_id, _other_source_message_id = _persist_repository_save(repositories)
    old = repositories.update_job(
        repositories.create_job(
            save_id=save_id,
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="succeeded",
    )
    fast = repositories.update_job(
        repositories.create_job(
            save_id=save_id,
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="succeeded",
    )
    middle = repositories.update_job(
        repositories.create_job(
            save_id=save_id,
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="succeeded",
    )
    slow = repositories.update_job(
        repositories.create_job(
            save_id=save_id,
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="succeeded",
    )
    failed = repositories.update_job(
        repositories.create_job(
            save_id=save_id,
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="failed",
        error="provider failed",
    )
    other_save = repositories.update_job(
        repositories.create_job(
            save_id=other_save_id,
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="succeeded",
    )
    rows = [
        (old.id, "2026-06-01 11:59:00", "2026-06-01 11:59:02", 20),
        (fast.id, "2026-06-01 12:00:00", "2026-06-01 12:00:01", 100),
        (middle.id, "2026-06-01 12:01:00", "2026-06-01 12:01:02", 200),
        (slow.id, "2026-06-01 12:02:00", "2026-06-01 12:02:04", 400),
        (failed.id, "2026-06-01 12:03:00", "2026-06-01 12:03:01", 900),
        (other_save.id, "2026-06-01 12:04:00", "2026-06-01 12:04:01", 50),
    ]
    for job_id, started_at, completed_at, duration_ms in rows:
        repositories.connection.execute(
            """
            UPDATE jobs
            SET created_at = datetime(?, '-2 seconds'),
                started_at = ?,
                completed_at = ?,
                duration_ms = ?
            WHERE id = ?
            """,
            (started_at, started_at, completed_at, duration_ms, job_id),
        )

    [average] = repositories.runtime_job_averages(
        save_id=save_id,
        since="2026-06-01T12:00:00+00:00",
    )

    assert average.job_type == "chat_turn"
    assert average.sample_count == 4
    assert average.success_count == 3
    assert average.failed_count == 1
    assert average.average_duration_ms == 233
    assert average.p50_duration_ms == 200
    assert average.p95_duration_ms == 400
    assert average.failure_rate == 0.25
    assert average.average_queue_wait_ms == 2000
    assert average.p95_queue_wait_ms == 2000


def test_repository_lists_terminal_jobs_and_slowest_recent_operations_safely(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _source_message_id = _persist_repository_save(repositories)
    failed = repositories.update_job(
        repositories.create_job(
            save_id=save_id,
            type="chat_turn",
            status="running",
            payload={"prompt": "secret prompt text", "provider": "fake"},
        ).id,
        status="failed",
        error="failed with token=super-secret",
    )
    succeeded = repositories.update_job(
        repositories.create_job(
            save_id=save_id,
            type="image_generation",
            status="running",
            payload={"prompt": "secret image prompt"},
        ).id,
        status="succeeded",
        result={"body": "private result body"},
    )
    repositories.connection.execute(
        """
        UPDATE jobs
        SET created_at = '2026-06-01 12:00:00',
            started_at = '2026-06-01 12:00:03',
            completed_at = '2026-06-01 12:00:10',
            duration_ms = 7000
        WHERE id = ?
        """,
        (failed.id,),
    )
    repositories.connection.execute(
        """
        UPDATE jobs
        SET created_at = '2026-06-01 12:01:00',
            started_at = '2026-06-01 12:01:01',
            completed_at = '2026-06-01 12:01:03',
            duration_ms = 2000
        WHERE id = ?
        """,
        (succeeded.id,),
    )
    repositories.record_job_step(
        job_id=failed.id,
        name="provider.chat",
        status="failed",
        provider="fake",
        model="fake-chat",
        task="chat",
        duration_ms=6500,
        error="step leaked token=super-secret",
        metadata={"prompt": "secret prompt text", "token_total": 123},
    )

    terminal = repositories.list_terminal_jobs(
        statuses=("failed",),
        save_id=save_id,
        since="2026-06-01T12:00:00+00:00",
        limit=10,
    )
    slowest = repositories.runtime_slowest_recent_operations(
        save_id=save_id,
        since="2026-06-01T12:00:00+00:00",
        limit=5,
    )

    assert [job.id for job in terminal] == [failed.id]
    assert terminal[0].payload == {"prompt": "secret prompt text", "provider": "fake"}
    assert slowest[0].job_id == failed.id
    assert slowest[0].job_type == "chat_turn"
    assert slowest[0].duration_ms == 7000
    assert slowest[0].queue_wait_ms == 3000
    assert slowest[0].slowest_step_name == "provider.chat"
    assert slowest[0].slowest_step_duration_ms == 6500
    assert "secret prompt text" not in repr(slowest)
    assert "super-secret" not in repr(slowest)


def test_nested_transaction_rollback_preserves_outer_transaction_scope(
    repositories: PersistenceRepositories,
) -> None:
    repositories.begin_transaction()
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"starting_location": "Gatehouse"},
    )
    repositories.begin_transaction()
    inner_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Rolled Back Watch",
    )

    repositories.rollback_transaction()
    repositories.commit_transaction()

    assert repositories._transaction_depth == 0
    assert repositories.get_scenario(scenario.id) == scenario
    assert (
        repositories.connection.execute(
            "SELECT id FROM saves WHERE id = ?",
            (inner_save.id,),
        ).fetchone()
        is None
    )


def test_bragi_repository_database_path_writes_are_durable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    repository = BragiRepository(database_path)
    scenario = repository.create_scenario(
        type="full_roleplay",
        title="Frostglass Hall",
        premise="A sealed hall is thawing after a century.",
        player_role="Relic hunter",
        content={"starting_location": "Mirror nave"},
    )
    save = repository.create_save(scenario_id=scenario.id, title="First Thaw")
    message = repository.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The frost cracks across the mirror floor.",
        provider="openrouter",
        model="anthropic/claude-sonnet",
        token_estimate=19,
    )
    repository.connection.close()
    del repository

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        scenario_row = connection.execute(
            "SELECT id, title FROM scenarios WHERE id = ?",
            (scenario.id,),
        ).fetchone()
        save_row = connection.execute(
            "SELECT id, scenario_id, title FROM saves WHERE id = ?",
            (save.id,),
        ).fetchone()
        message_row = connection.execute(
            "SELECT id, save_id, body, provider, model FROM messages WHERE id = ?",
            (message.id,),
        ).fetchone()

    assert dict(scenario_row) == {
        "id": scenario.id,
        "title": "Frostglass Hall",
    }
    assert dict(save_row) == {
        "id": save.id,
        "scenario_id": scenario.id,
        "title": "First Thaw",
    }
    assert dict(message_row) == {
        "id": message.id,
        "save_id": save.id,
        "body": "The frost cracks across the mirror floor.",
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet",
    }

    fresh_repository = BragiRepository(database_path)
    try:
        persisted_messages = fresh_repository.list_messages(save.id)
    finally:
        fresh_repository.connection.close()

    assert persisted_messages == [message]


def test_provider_config_facade_persists_non_secret_metadata_only(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    repository = BragiRepository(database_path)

    config = repository.upsert_provider_config(
        provider="openrouter",
        enabled=True,
        has_api_key=True,
        last_model_refresh_at="2026-05-12T18:30:00Z",
        last_error="rate limited",
    )
    fetched_config = repository.get_provider_config("openrouter")

    assert fetched_config == config
    assert fetched_config is not None
    assert fetched_config.provider == "openrouter"
    assert fetched_config.enabled is True
    assert fetched_config.has_api_key is True
    assert fetched_config.last_model_refresh_at == "2026-05-12T18:30:00Z"
    assert fetched_config.last_error == "rate limited"

    with pytest.raises(TypeError, match="api_key"):
        repository.upsert_provider_config(
            provider="openrouter",
            enabled=True,
            has_api_key=True,
            last_model_refresh_at="2026-05-12T18:30:00Z",
            last_error=None,
            **{"api_key": "sk-should-not-be-stored"},
        )

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        columns = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(provider_configs)")
        ]
        stored_values = dict(
            connection.execute(
                "SELECT * FROM provider_configs WHERE provider = ?",
                ("openrouter",),
            ).fetchone()
        )

    repository.connection.close()

    assert "api_key" not in columns
    assert "encrypted_api_key" not in columns
    assert "sk-should-not-be-stored" not in {
        value for value in stored_values.values() if isinstance(value, str)
    }


def test_repository_updates_scenario_records(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )

    updated = repositories.update_scenario(
        scenario_id=scenario.id,
        title="Ashfall Spire",
        premise="The storm has reached the upper beacon.",
        player_role="Beacon keeper",
        content={
            "starting_scene": "The lens turns red.",
            "tone_genre": "siege mystery",
        },
    )

    fetched = repositories.get_scenario(scenario.id)
    assert fetched == updated
    assert fetched is not None
    assert fetched.title == "Ashfall Spire"
    assert fetched.premise == "The storm has reached the upper beacon."
    assert fetched.player_role == "Beacon keeper"
    assert fetched.content_json == (
        '{"starting_scene":"The lens turns red.","tone_genre":"siege mystery"}'
    )


def test_save_scenario_evolution_overrides_prompt_details_without_mutating_base(
    repositories: PersistenceRepositories,
) -> None:
    base_scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "tone_genre": "siege mystery",
        },
    )
    evolved_save = repositories.create_save(
        scenario_id=base_scenario.id,
        title="Night Watch",
    )
    unchanged_save = repositories.create_save(
        scenario_id=base_scenario.id,
        title="Second Watch",
    )
    source_message = repositories.append_message(
        save_id=evolved_save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens wakes and turns the ash storm red.",
        provider="openrouter",
        model="openrouter/scenario-evolver",
    )

    repositories.record_save_scenario_evolution(
        save_id=evolved_save.id,
        source_message_id=source_message.id,
        title="Ashfall Keep: Red Lens",
        premise="The keep's beacon now burns red against the ash storm.",
        player_role="Signal warden",
        content={
            "starting_scene": "The red beacon lens watches the storm.",
            "tone_genre": "siege mystery",
            "current_scene": "The beacon gallery is hot with warning light.",
        },
        reason="The narrator established the beacon lens changed state.",
        provider="openrouter",
        model="openrouter/scenario-evolver",
    )

    evolved_details = repositories.load_save_details(evolved_save.id)
    unchanged_details = repositories.load_save_details(unchanged_save.id)
    base_details = repositories.get_scenario(base_scenario.id)
    audit = repositories.list_save_scenario_evolution_audit(
        save_id=evolved_save.id,
        include_archived=True,
    )

    assert evolved_details is not None
    assert evolved_details.scenario.id == base_scenario.id
    assert evolved_details.scenario.title == "Ashfall Keep: Red Lens"
    assert evolved_details.scenario.premise == (
        "The keep's beacon now burns red against the ash storm."
    )
    assert evolved_details.scenario.player_role == "Signal warden"
    assert _load_content(evolved_details.scenario.content_json) == {
        "starting_scene": "The red beacon lens watches the storm.",
        "tone_genre": "siege mystery",
        "current_scene": "The beacon gallery is hot with warning light.",
    }

    assert unchanged_details is not None
    assert unchanged_details.scenario == base_scenario
    assert base_details == base_scenario

    assert len(audit) == 1
    audit_row = audit[0]
    assert audit_row.save_id == evolved_save.id
    assert audit_row.source_message_id == source_message.id
    assert audit_row.reason == (
        "The narrator established the beacon lens changed state."
    )
    assert audit_row.provider == "openrouter"
    assert audit_row.model == "openrouter/scenario-evolver"
    assert audit_row.created_at is not None
    assert audit_row.active is True
    assert audit_row.archived_at is None


def test_repository_updates_and_archives_memories(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    memory = repositories.add_memory(
        save_id=save_id,
        body="Captain Ilyra promised to hold the east stair.",
        tags=["npc", "promise"],
        importance=0.7,
        source_message_id=source_message_id,
    )

    updated = repositories.update_memory(
        memory_id=memory.id,
        body="Captain Ilyra broke her promise at the east stair.",
        tags=["npc", "broken-promise"],
        importance=0.9,
        source_message_ids=[source_message_id, "message-extra"],
    )

    assert repositories.list_memories(save_id) == [updated]
    assert updated.body == "Captain Ilyra broke her promise at the east stair."
    assert updated.tags == ["npc", "broken-promise"]
    assert updated.importance == 0.9
    assert updated.source_message_ids == [source_message_id, "message-extra"]

    repositories.archive_memory(memory.id)

    assert repositories.list_memories(save_id) == []
    archived_at = repositories.connection.execute(
        "SELECT archived_at FROM memories WHERE id = ?",
        (memory.id,),
    ).fetchone()[0]
    assert archived_at is not None


def test_repository_updates_and_archives_summaries(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    summary = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=source_message_id,
        covers_message_end_id=source_message_id,
        body="The watch began as the tower beacon started failing.",
        provider="openrouter",
        model="anthropic/claude-sonnet",
    )

    updated = repositories.update_summary(
        summary_id=summary.id,
        body="The watch began after the beacon lens cracked.",
    )

    assert repositories.list_summaries(save_id) == [updated]
    assert updated.body == "The watch began after the beacon lens cracked."
    assert updated.provider == "openrouter"
    assert updated.model == "anthropic/claude-sonnet"

    repositories.archive_summary(summary.id)

    assert repositories.list_summaries(save_id) == []
    archived_at = repositories.connection.execute(
        "SELECT archived_at FROM summaries WHERE id = ?",
        (summary.id,),
    ).fetchone()[0]
    assert archived_at is not None


def test_repository_archives_world_state_and_upsert_restores_it(
    repositories: PersistenceRepositories,
) -> None:
    save_id, source_message_id = _persist_repository_save(repositories)
    state = repositories.upsert_world_state(
        save_id=save_id,
        key="scene.location",
        value={"name": "Gatehouse"},
        category="scene",
        confidence=0.8,
        source_message_id=source_message_id,
    )

    repositories.archive_world_state(save_id=save_id, key="scene.location")

    assert repositories.list_world_state(save_id) == []
    archived_at = repositories.connection.execute(
        "SELECT archived_at FROM world_state WHERE id = ?",
        (state.id,),
    ).fetchone()[0]
    assert archived_at is not None

    restored = repositories.upsert_world_state(
        save_id=save_id,
        key="scene.location",
        value={"name": "Beacon tower"},
        category="scene",
        confidence=0.95,
        source_message_id=source_message_id,
    )

    assert restored.id == state.id
    assert repositories.list_world_state(save_id) == [restored]
    assert restored.value == {"name": "Beacon tower"}
    restored_archived_at = repositories.connection.execute(
        "SELECT archived_at FROM world_state WHERE id = ?",
        (state.id,),
    ).fetchone()[0]
    assert restored_archived_at is None


def test_repository_updates_message_body_and_records_revisions(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)

    updated = repositories.update_message_body(
        save_id=save_id,
        message_id=message_id,
        body="Ash scratches the glass as the stair holds steady.",
    )
    first_revision = repositories.add_message_revision(
        save_id=save_id,
        message_id=message_id,
        previous_body="Ash scratches the glass as the stair shakes.",
        new_body=updated.body,
        diff_unified=(
            "--- previous\n"
            "+++ current\n"
            "@@ -1 +1 @@\n"
            "-Ash scratches the glass as the stair shakes.\n"
            "+Ash scratches the glass as the stair holds steady.\n"
        ),
        reconciliation_status="queued",
    )
    second_revision = repositories.add_message_revision(
        save_id=save_id,
        message_id=message_id,
        previous_body=updated.body,
        new_body="Ash scratches the glass as the tower holds steady.",
        diff_unified="diff two",
        reconciliation_status="queued",
    )

    assert updated.id == message_id
    assert repositories.list_messages(save_id)[0].body == (
        "Ash scratches the glass as the stair holds steady."
    )
    assert first_revision.revision_number == 1
    assert second_revision.revision_number == 2
    assert first_revision.diff_unified.startswith("--- previous")
    assert [
        (revision.revision_number, revision.previous_body, revision.new_body)
        for revision in repositories.list_message_revisions(
            save_id=save_id,
            message_id=message_id,
        )
    ] == [
        (
            1,
            "Ash scratches the glass as the stair shakes.",
            "Ash scratches the glass as the stair holds steady.",
        ),
        (
            2,
            "Ash scratches the glass as the stair holds steady.",
            "Ash scratches the glass as the tower holds steady.",
        ),
    ]

    repositories.mark_message_revision_reconciled(
        second_revision.id,
        status="failed",
        error="structured backend failed",
    )

    metadata = repositories.message_revision_metadata(save_id)
    revisions = repositories.list_message_revisions(
        save_id=save_id,
        message_id=message_id,
    )
    assert metadata[message_id].revision_count == 2
    assert metadata[message_id].edited_at is not None
    assert revisions[-1].reconciliation_status == "failed"
    assert revisions[-1].reconciliation_error == "structured backend failed"
    assert revisions[-1].reconciled_at is not None


def test_narration_graph_normalizes_state_alias_and_bounds_owner_rows(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    state = repositories.upsert_world_state(
        save_id=save_id,
        key="vault.code",
        value={"phrase": "ember dawn"},
        source_message_id=message_id,
    )
    characters = [
        repositories.add_character(
            save_id=save_id,
            name=f"Warden {index:02d}",
            character_id=f"character-{index:02d}",
        )
        for index in range(20)
    ]
    threshold_edge = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=characters[0].id,
        target_type="state",
        target_id=state.id,
        knowledge_state="may_know",
        confidence=0.72,
    )
    for character in characters[1:]:
        repositories.add_character_knowledge_edge(
            save_id=save_id,
            character_id=character.id,
            target_type="state",
            target_id=state.id,
            knowledge_state="knows",
        )

    edges = repositories.list_narration_character_knowledge_edges(
        save_id,
        target_keys={("world_state", state.id)},
        present_character_ids={character.id for character in characters},
        visibility_character_ids={character.id for character in characters},
    )

    assert threshold_edge.id in {edge.id for edge in edges}
    assert len(edges) == 8


def test_narration_graph_keeps_present_owner_restrictions_across_aliases(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _persist_repository_save(repositories)
    state = repositories.upsert_world_state(
        save_id=save_id,
        key="vault.code",
        value={"phrase": "ember dawn"},
        source_message_id=message_id,
    )
    first = repositories.add_character(
        save_id=save_id,
        name="First Warden",
    )
    second = repositories.add_character(
        save_id=save_id,
        name="Second Warden",
    )
    first_denial = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=first.id,
        target_type="world_state",
        target_id=state.id,
        knowledge_state="does_not_know",
    )
    repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=first.id,
        target_type="state",
        target_id=state.id,
        knowledge_state="knows",
    )
    repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=second.id,
        target_type="world_state",
        target_id=state.id,
        knowledge_state="does_not_know",
    )

    edges = repositories.list_narration_character_knowledge_edges(
        save_id,
        target_keys={("world_state", state.id)},
        present_character_ids={first.id, second.id},
        visibility_character_ids={first.id, second.id},
    )

    assert first_denial.id in {edge.id for edge in edges}


def _persist_repository_save(
    repositories: PersistenceRepositories,
) -> tuple[str, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Ash scratches the glass as the stair shakes.",
    )
    return save.id, message.id


def _load_content(content_json: str) -> dict[str, object]:
    loaded = json.loads(content_json)
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)
