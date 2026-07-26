from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.content_rating import (
    CONTENT_FILTER_RATING_SETTING,
    FADE_TO_BLACK_ENABLED_SETTING,
)
from bragi.services.message_revision_service import MessageRevisionService
from bragi.services.state_service import (
    ExtractedStateChange,
    StateExtraction,
    StateService,
)
from bragi.services.turn_snapshot_service import TurnSnapshotService
from bragi.services.world_data_service import (
    WorldDataEdits,
    WorldDataScenarioEdit,
    WorldDataService,
    WorldDataStateRow,
)


class _UnusedExtractor:
    async def extract(self, request: Any) -> StateExtraction:
        raise AssertionError("this test applies deterministic state directly")


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_rollback_keeps_manually_archived_world_state_key_archived(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "title": "Ashfall Keep",
            "premise": "A border keep is cut off by ash storms.",
            "player_role": "Signal warden",
            "starting_scene": "The beacon gutters in the tower.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    seed_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Warden Elian is posted at the lower stair.",
        provider="fake",
        model="fake-chat",
    )
    initial_state = repositories.upsert_world_state(
        save_id=save.id,
        key="npc.warden.elian",
        value={"name": "Elian", "status": "posted"},
        category="npc",
        confidence=0.8,
        source_message_id=seed_message.id,
    )

    world_data = WorldDataService(repositories=repositories, active_save_id=save.id)
    model = world_data.build_model()
    assert model.scenario is not None
    world_data.apply_edits(
        WorldDataEdits(
            scenario=WorldDataScenarioEdit(
                title=model.scenario.title,
                premise=model.scenario.premise,
                player_role=model.scenario.player_role,
                content_sections=model.scenario.content_sections,
            ),
            world_state=(
                WorldDataStateRow(
                    row_id=initial_state.id,
                    key="npc.warden.elian",
                    value_json="{not parsed when archived",
                    category="npc",
                    confidence=0.8,
                    source_message_id=seed_message.id,
                    archived=True,
                    original_key="npc.warden.elian",
                ),
            ),
        )
    )

    assert repositories.list_world_state(save.id) == []
    state_changes = repositories.list_state_changes(save.id)
    assert len(state_changes) == 1
    manual_tombstone = state_changes[0]
    assert manual_tombstone.operation == "manual_world_data_edit"
    assert manual_tombstone.state_key == "npc.warden.elian"
    assert json.loads(manual_tombstone.before_json or "") == {
        "name": "Elian",
        "status": "posted",
    }
    assert manual_tombstone.after_json is None
    assert manual_tombstone.source_message_id is None

    later_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ask whether Elian returned.",
    )
    later_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Elian returns to the stair with a cracked lantern.",
        provider="fake",
        model="fake-chat",
    )
    StateService(
        repositories=repositories,
        extractor=_UnusedExtractor(),
    ).apply_extraction(
        save_id=save.id,
        extraction=StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="npc.warden.elian",
                    value={"name": "Elian", "status": "returned"},
                    category="npc",
                    confidence=0.9,
                    source_message_id=later_narrator.id,
                ),
            ),
        ),
        allowed_source_message_ids=(later_player.id, later_narrator.id),
    )
    active_world_state = repositories.list_world_state(save.id)
    assert [(state.key, state.value) for state in active_world_state] == [
        ("npc.warden.elian", {"name": "Elian", "status": "returned"}),
    ]

    rollback = MessageRevisionService(repositories).rollback_from_message(
        save_id=save.id,
        message_id=later_player.id,
    )

    assert [message.id for message in rollback.deleted_messages] == [
        later_player.id,
        later_narrator.id,
    ]
    assert repositories.list_world_state(save.id) == []
    archived_row = repositories.connection.execute(
        """
        SELECT id, archived_at
        FROM world_state
        WHERE save_id = ? AND key = ?
        """,
        (save.id, "npc.warden.elian"),
    ).fetchone()
    assert archived_row is not None
    assert archived_row["id"] == initial_state.id
    assert archived_row["archived_at"] is not None


def test_edit_narrator_message_persists_revision_without_deleting_later_turns(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    service = MessageRevisionService(repositories)

    edit = service.edit_narrator_message(
        save_id=save_id,
        message_id=ids["narrator_2"],
        body="The corridor holds steady as ash taps the door.",
    )

    assert edit.message.id == ids["narrator_2"]
    assert edit.message.body == "The corridor holds steady as ash taps the door."
    assert edit.previous_body == "The corridor floods with ash."
    assert edit.revision.revision_number == 1
    assert "-The corridor floods with ash." in edit.revision.diff_unified
    assert "+The corridor holds steady as ash taps the door." in (
        edit.revision.diff_unified
    )
    assert [message.id for message in repositories.list_messages(save_id)] == [
        ids["player_1"],
        ids["narrator_1"],
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]


def test_edit_narrator_message_leaves_safety_review_to_async_caller(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    rejected = "He thrust into her before the scene changed."

    edit = MessageRevisionService(repositories).edit_narrator_message(
        save_id=save_id,
        message_id=ids["narrator_2"],
        body=rejected,
    )

    assert edit.message.body == rejected
    assert edit.revision.new_body == rejected
    assert rejected in edit.revision.diff_unified


def test_edit_narrator_message_respects_adult_rating_and_disabled_fade(
    repositories: PersistenceRepositories,
) -> None:
    adult = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=adult.id,
        key=CONTENT_FILTER_RATING_SETTING,
        value="r",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=adult.id,
        key=FADE_TO_BLACK_ENABLED_SETTING,
        value=False,
    )
    save_id, ids = _create_revision_save(repositories)
    replacement = "They had sex after returning to the inn."

    edit = MessageRevisionService(repositories).edit_narrator_message(
        save_id=save_id,
        message_id=ids["narrator_2"],
        body=replacement,
        current_user_id=adult.id,
    )

    assert edit.message.body == replacement
    assert edit.revision.new_body == replacement


def test_edit_narrator_message_archives_summaries_covering_edited_message(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    earlier = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=ids["player_1"],
        covers_message_end_id=ids["narrator_1"],
        body="Mara lit the beacon.",
        provider="fake",
        model="fake-summary",
    )
    covering = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=ids["player_2"],
        covers_message_end_id=ids["narrator_3"],
        body="Mara opened the door, ash flooded in, and a shadow followed.",
        provider="fake",
        model="fake-summary",
    )
    later = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=ids["player_3"],
        covers_message_end_id=ids["narrator_3"],
        body="Mara stepped through and a shadow followed.",
        provider="fake",
        model="fake-summary",
    )

    MessageRevisionService(repositories).edit_narrator_message(
        save_id=save_id,
        message_id=ids["narrator_2"],
        body="The corridor stays clear.",
    )

    active_summary_ids = {
        summary.id for summary in repositories.list_summaries(save_id)
    }
    archived_rows = repositories.connection.execute(
        """
        SELECT id, archived_at
        FROM summaries
        WHERE id IN (?, ?, ?)
        """,
        (earlier.id, covering.id, later.id),
    ).fetchall()
    archive_status = {row["id"]: row["archived_at"] for row in archived_rows}
    assert active_summary_ids == {earlier.id, later.id}
    assert archive_status[earlier.id] is None
    assert archive_status[covering.id] is not None
    assert archive_status[later.id] is None


def test_edit_message_without_resubmit_updates_player_without_deleting_later_turns(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    service = MessageRevisionService(repositories)

    edit = service.edit_message_without_resubmit(
        save_id=save_id,
        message_id=ids["player_2"],
        body="I keep the sealed door shut and listen.",
    )

    assert edit.message.id == ids["player_2"]
    assert edit.message.role == "player"
    assert edit.message.body == "I keep the sealed door shut and listen."
    assert edit.previous_body == "I open the sealed door."
    assert edit.revision.revision_number == 1
    assert edit.revision.reconciliation_status == "queued"
    assert "-I open the sealed door." in edit.revision.diff_unified
    assert "+I keep the sealed door shut and listen." in edit.revision.diff_unified
    assert [message.id for message in repositories.list_messages(save_id)] == [
        ids["player_1"],
        ids["narrator_1"],
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]


def test_edit_message_without_resubmit_archives_summaries_covering_player_message(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    earlier = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=ids["player_1"],
        covers_message_end_id=ids["narrator_1"],
        body="Mara lit the beacon.",
        provider="fake",
        model="fake-summary",
    )
    covering = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=ids["player_2"],
        covers_message_end_id=ids["narrator_3"],
        body="Mara opened the door, ash flooded in, and a shadow followed.",
        provider="fake",
        model="fake-summary",
    )
    later = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=ids["player_3"],
        covers_message_end_id=ids["narrator_3"],
        body="Mara stepped through and a shadow followed.",
        provider="fake",
        model="fake-summary",
    )

    MessageRevisionService(repositories).edit_message_without_resubmit(
        save_id=save_id,
        message_id=ids["player_2"],
        body="I keep the sealed door shut.",
    )

    active_summary_ids = {
        summary.id for summary in repositories.list_summaries(save_id)
    }
    archived_rows = repositories.connection.execute(
        """
        SELECT id, archived_at
        FROM summaries
        WHERE id IN (?, ?, ?)
        """,
        (earlier.id, covering.id, later.id),
    ).fetchall()
    archive_status = {row["id"]: row["archived_at"] for row in archived_rows}
    assert active_summary_ids == {earlier.id, later.id}
    assert archive_status[earlier.id] is None
    assert archive_status[covering.id] is not None
    assert archive_status[later.id] is None


@pytest.mark.parametrize(
    ("message_key", "body", "error"),
    [
        ("player_2", "I knock.", "Only narrator messages can be edited this way"),
        ("narrator_2", "", "Message is empty"),
        ("narrator_2", "The corridor floods with ash.", "Message was not changed"),
    ],
)
def test_edit_narrator_message_rejects_invalid_edits(
    repositories: PersistenceRepositories,
    message_key: str,
    body: str,
    error: str,
) -> None:
    save_id, ids = _create_revision_save(repositories)

    with pytest.raises(ValueError, match=error):
        MessageRevisionService(repositories).edit_narrator_message(
            save_id=save_id,
            message_id=ids[message_key],
            body=body,
        )


@pytest.mark.parametrize(
    ("message_key", "body", "error"),
    [
        ("player_2", "", "Message is empty"),
        ("player_2", "I open the sealed door.", "Message was not changed"),
    ],
)
def test_edit_message_without_resubmit_rejects_invalid_edits(
    repositories: PersistenceRepositories,
    message_key: str,
    body: str,
    error: str,
) -> None:
    save_id, ids = _create_revision_save(repositories)

    with pytest.raises(ValueError, match=error):
        MessageRevisionService(repositories).edit_message_without_resubmit(
            save_id=save_id,
            message_id=ids[message_key],
            body=body,
        )


def test_edit_message_without_resubmit_rejects_unsupported_message_roles(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ids = _create_revision_save(repositories)
    system_message = repositories.append_message(
        save_id=save_id,
        role="system",
        speaker_name=None,
        body="Internal bookkeeping.",
    )

    with pytest.raises(
        ValueError,
        match="Only player and narrator messages can be edited this way",
    ):
        MessageRevisionService(repositories).edit_message_without_resubmit(
            save_id=save_id,
            message_id=system_message.id,
            body="Updated bookkeeping.",
        )


def test_rollback_preserves_later_manual_world_state_edit_on_same_key(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "title": "Ashfall Keep",
            "premise": "A border keep is cut off by ash storms.",
            "player_role": "Signal warden",
            "starting_scene": "The beacon gutters in the tower.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    seed_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Warden Elian is posted at the lower stair.",
        provider="fake",
        model="fake-chat",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="npc.warden.elian",
        value={"name": "Elian", "status": "posted"},
        category="npc",
        confidence=0.8,
        source_message_id=seed_message.id,
    )

    later_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ask whether Elian returned.",
    )
    later_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Elian returns to the stair with a cracked lantern.",
        provider="fake",
        model="fake-chat",
    )
    StateService(
        repositories=repositories,
        extractor=_UnusedExtractor(),
    ).apply_extraction(
        save_id=save.id,
        extraction=StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="npc.warden.elian",
                    value={"name": "Elian", "status": "returned"},
                    category="npc",
                    confidence=0.9,
                    source_message_id=later_narrator.id,
                ),
            ),
        ),
        allowed_source_message_ids=(later_player.id, later_narrator.id),
    )

    world_data = WorldDataService(repositories=repositories, active_save_id=save.id)
    model = world_data.build_model()
    assert model.scenario is not None
    assert len(model.state_rows) == 1
    assert model.state_rows[0].source_message_id == later_narrator.id
    world_data.apply_edits(
        WorldDataEdits(
            scenario=WorldDataScenarioEdit(
                title=model.scenario.title,
                premise=model.scenario.premise,
                player_role=model.scenario.player_role,
                content_sections=model.scenario.content_sections,
            ),
            world_state=(
                WorldDataStateRow(
                    row_id=model.state_rows[0].row_id,
                    key="npc.warden.elian",
                    value_json=json.dumps({"name": "Elian", "status": "retired"}),
                    category="manual_npc",
                    confidence=0.42,
                    source_message_id=model.state_rows[0].source_message_id,
                    original_key="npc.warden.elian",
                ),
            ),
        )
    )
    active_state = [
        (
            state.key,
            state.value,
            state.category,
            state.confidence,
            state.source_message_id,
        )
        for state in repositories.list_world_state(save.id)
    ]
    assert active_state == [
        (
            "npc.warden.elian",
            {"name": "Elian", "status": "retired"},
            "manual_npc",
            0.42,
            None,
        ),
    ]

    rollback = MessageRevisionService(repositories).rollback_from_message(
        save_id=save.id,
        message_id=later_player.id,
    )

    assert [message.id for message in rollback.deleted_messages] == [
        later_player.id,
        later_narrator.id,
    ]
    active_state = [
        (
            state.key,
            state.value,
            state.category,
            state.confidence,
            state.source_message_id,
        )
        for state in repositories.list_world_state(save.id)
    ]
    assert active_state == [
        (
            "npc.warden.elian",
            {"name": "Elian", "status": "retired"},
            "manual_npc",
            0.42,
            None,
        ),
    ]


def test_rollback_archives_scenario_evolution_from_rolled_back_messages(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Revision Watch")
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I light the beacon.",
    )
    first_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens wakes and paints the gallery red.",
        provider="fake",
        model="fake-chat",
    )
    repositories.record_save_scenario_evolution(
        save_id=save.id,
        source_message_id=first_narrator.id,
        title="Ashfall Keep: Red Lens",
        premise="The keep's beacon now burns red against the ash storm.",
        player_role="Signal warden",
        content={"starting_scene": "The red beacon lens watches the storm."},
        reason="The beacon was lit in the first turn.",
        provider="fake",
        model="fake-scenario-evolver",
    )
    later_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I open the sealed door below the beacon.",
    )
    later_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The door opens onto a stair packed with warm ash.",
        provider="fake",
        model="fake-chat",
    )
    repositories.record_save_scenario_evolution(
        save_id=save.id,
        source_message_id=later_narrator.id,
        source_message_ids=(later_player.id, later_narrator.id),
        title="Ashfall Keep: Open Door",
        premise="The red beacon burns above an open ash-choked stair.",
        player_role="Signal warden",
        content={"starting_scene": "The sealed door stands open under red light."},
        reason="The later turn opened the sealed door.",
        provider="fake",
        model="fake-scenario-evolver",
    )
    active_details = repositories.load_save_details(save.id)
    assert active_details is not None
    assert active_details.scenario.title == "Ashfall Keep: Open Door"

    rollback = MessageRevisionService(repositories).rollback_from_message(
        save_id=save.id,
        message_id=later_player.id,
    )

    assert [message.id for message in rollback.deleted_messages] == [
        later_player.id,
        later_narrator.id,
    ]
    reverted_details = repositories.load_save_details(save.id)
    assert reverted_details is not None
    assert reverted_details.scenario.title == "Ashfall Keep: Red Lens"
    assert reverted_details.scenario.premise == (
        "The keep's beacon now burns red against the ash storm."
    )
    assert _load_content(reverted_details.scenario.content_json) == {
        "starting_scene": "The red beacon lens watches the storm."
    }

    audit = repositories.list_save_scenario_evolution_audit(
        save_id=save.id,
        include_archived=True,
    )
    audit_by_source_id = {row.source_message_id: row for row in audit}
    earlier_audit = audit_by_source_id[first_narrator.id]
    later_audit = audit_by_source_id[later_narrator.id]
    assert earlier_audit.active is True
    assert earlier_audit.archived_at is None
    assert later_audit.active is False
    assert later_audit.archived_at is not None


def test_rollback_before_loss_trigger_archives_outcome_and_rebuilds_conditions(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Revision Watch")
    first_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I brace the beacon lens.",
    )
    first_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lens is cracked but the tower still stands.",
        provider="fake",
        model="fake-chat",
    )
    condition = repositories.add_loss_condition(
        condition_id="loss-beacon-collapse",
        save_id=save.id,
        name="Beacon collapse",
        description="The beacon lens is cracked but stable.",
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
            "name": "Beacon collapse",
            "description": "The beacon lens is cracked but stable.",
            "status": "active",
            "source": "structured",
        },
        reason="The first turn established the risk.",
        provider="fake",
        model="fake-loss-model",
        source_message_id=first_narrator.id,
    )
    later_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ignore the crack and pull the wrong lever.",
    )
    later_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lens shatters and the beacon tower collapses.",
        provider="fake",
        model="fake-chat",
    )
    repositories.update_loss_condition(
        condition_id=condition.id,
        name="Beacon collapse",
        description="The beacon tower has collapsed.",
        status="triggered",
    )
    later_change = repositories.add_loss_condition_change(
        save_id=save.id,
        condition_id=condition.id,
        operation="update",
        before={"status": "active"},
        after={
            "id": condition.id,
            "name": "Beacon collapse",
            "description": "The beacon tower has collapsed.",
            "status": "triggered",
            "source": "structured",
        },
        reason="The later narrator turn explicitly triggered the loss.",
        provider="fake",
        model="fake-loss-model",
        source_message_id=later_narrator.id,
    )
    outcome = repositories.create_loss_outcome(
        save_id=save.id,
        condition_id=condition.id,
        condition_name=condition.name,
        triggering_message_id=later_narrator.id,
        explanation="The beacon falls and the watch is lost.",
        confidence=0.93,
        evidence={
            "items": [
                {
                    "source_message_id": later_narrator.id,
                    "quote": "beacon tower collapses",
                }
            ]
        },
        provider="fake",
        model="fake-loss-model",
    )

    rollback = MessageRevisionService(repositories).rollback_from_message(
        save_id=save.id,
        message_id=later_player.id,
    )

    assert [message.id for message in rollback.deleted_messages] == [
        later_player.id,
        later_narrator.id,
    ]
    assert rollback.archived_loss_outcome_ids == frozenset({outcome.id})
    assert repositories.get_active_loss_outcome(save.id) is None
    rebuilt_conditions = repositories.list_loss_conditions(save.id)
    assert [
        (row.name, row.description, row.status)
        for row in rebuilt_conditions
    ] == [
        (
            "Beacon collapse",
            "The beacon lens is cracked but stable.",
            "active",
        )
    ]
    change_rows = repositories.list_loss_condition_changes(
        save.id,
        include_archived=True,
    )
    change_by_id = {row.id: row for row in change_rows}
    assert change_by_id[later_change.id].archived_at is not None

    active_messages = repositories.list_messages(save.id)
    assert [message.id for message in active_messages] == [
        first_player.id,
        first_narrator.id,
    ]


def test_rollback_with_manually_archived_loss_condition_keeps_condition_archived(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Revision Watch")
    first_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I brace the beacon lens.",
    )
    first_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lens is cracked but the tower still stands.",
        provider="fake",
        model="fake-chat",
    )
    condition = repositories.add_loss_condition(
        condition_id="loss-beacon-collapse",
        save_id=save.id,
        name="Beacon collapse",
        description="The beacon lens is cracked but stable.",
        status="active",
        source="structured",
    )
    first_change = repositories.add_loss_condition_change(
        save_id=save.id,
        condition_id=condition.id,
        operation="add",
        before=None,
        after={
            "id": condition.id,
            "name": "Beacon collapse",
            "description": "The beacon lens is cracked but stable.",
            "status": "active",
            "source": "structured",
        },
        reason="The first turn established the risk.",
        provider="fake",
        model="fake-loss-model",
        source_message_id=first_narrator.id,
    )
    later_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ignore the crack and pull the wrong lever.",
    )
    later_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lens shatters and the beacon tower collapses.",
        provider="fake",
        model="fake-chat",
    )
    later_change = repositories.add_loss_condition_change(
        save_id=save.id,
        condition_id=condition.id,
        operation="update",
        before={"status": "active"},
        after={
            "id": condition.id,
            "name": "Beacon collapse",
            "description": "The beacon tower has collapsed.",
            "status": "triggered",
            "source": "structured",
        },
        reason="The later narrator turn explicitly triggered the loss.",
        provider="fake",
        model="fake-loss-model",
        source_message_id=later_narrator.id,
    )
    outcome = repositories.create_loss_outcome(
        save_id=save.id,
        condition_id=condition.id,
        condition_name=condition.name,
        triggering_message_id=later_narrator.id,
        explanation="The beacon falls and the watch is lost.",
        confidence=0.93,
        evidence={
            "items": [
                {
                    "source_message_id": later_narrator.id,
                    "quote": "beacon tower collapses",
                }
            ]
        },
        provider="fake",
        model="fake-loss-model",
    )
    repositories.archive_loss_condition(condition.id)

    rollback = MessageRevisionService(repositories).rollback_from_message(
        save_id=save.id,
        message_id=later_player.id,
    )

    assert [message.id for message in rollback.deleted_messages] == [
        later_player.id,
        later_narrator.id,
    ]
    assert rollback.archived_loss_outcome_ids == frozenset({outcome.id})
    assert repositories.get_active_loss_outcome(save.id) is None
    assert repositories.list_loss_conditions(save.id) == []
    archived_conditions = repositories.list_loss_conditions(
        save.id,
        include_archived=True,
    )
    assert [(row.id, row.status) for row in archived_conditions] == [
        (condition.id, "active")
    ]
    assert repositories.list_loss_condition_changes(save.id) == []
    archived_changes = repositories.list_loss_condition_changes(
        save.id,
        include_archived=True,
    )
    assert {row.id for row in archived_changes} == {first_change.id, later_change.id}
    assert all(row.archived_at is not None for row in archived_changes)
    assert [message.id for message in repositories.list_messages(save.id)] == [
        first_player.id,
        first_narrator.id,
    ]


def test_delete_suffix_from_narrator_uses_turn_boundary_and_archives_context(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    repositories.upsert_world_state(
        save_id=save_id,
        key="scene.location",
        value={"name": "Ash corridor"},
        category="scene",
        source_message_id=ids["narrator_2"],
    )
    repositories.add_state_change(
        save_id=save_id,
        operation="upsert",
        state_key="scene.location",
        before_json=json.dumps({"name": "Beacon tower"}),
        after_json=json.dumps({"name": "Ash corridor"}),
        source_message_id=ids["narrator_2"],
    )
    repositories.upsert_world_state(
        save_id=save_id,
        key="threat.shadow",
        value={"status": "following"},
        category="threat",
        source_message_id=ids["narrator_3"],
    )
    repositories.add_state_change(
        save_id=save_id,
        operation="upsert",
        state_key="threat.shadow",
        before_json=None,
        after_json=json.dumps({"status": "following"}),
        source_message_id=ids["narrator_3"],
    )
    memory = repositories.add_memory(
        save_id=save_id,
        body="Mara opened the sealed ash corridor.",
        tags=["ash", "door"],
        source_message_id=ids["narrator_2"],
    )
    repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=ids["player_2"],
        covers_message_end_id=ids["narrator_3"],
        body="Mara opened the door and a shadow followed.",
        provider="fake",
        model="fake-chat",
    )
    image = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=ids["narrator_2"],
        type="image",
        path="save/images/ash-corridor.png",
        prompt="ash corridor",
        provider="fake",
        model="fake-image",
        status="succeeded",
        asset_id="media-ash-corridor",
    )
    video = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=ids["narrator_2"],
        source_media_asset_id=image.id,
        type="video",
        path="save/videos/ash-corridor.mp4",
        prompt="ash corridor motion",
        provider="fake",
        model="fake-video",
        status="succeeded",
        mime_type="video/mp4",
        asset_id="media-ash-corridor-video",
    )
    location = repositories.add_location(
        save_id=save_id,
        name="Ash Corridor",
        source_message_id=ids["narrator_2"],
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=location.id,
        situation="The corridor floods with ash.",
        source_message_id=ids["narrator_2"],
    )
    character = repositories.add_character(
        save_id=save_id,
        name="Ash Shadow",
        location_id=location.id,
        source_message_id=ids["narrator_2"],
    )
    linked_memory = repositories.add_memory(
        save_id=save_id,
        body="The shadow knows the ash corridor.",
        tags=["shadow"],
        source_message_id=ids["narrator_1"],
    )
    source_link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id=linked_memory.id,
        relation="knows",
        source_message_id=ids["narrator_2"],
        link_id="link-source-message",
    )
    endpoint_link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character.id,
        target_type="location",
        target_id=location.id,
        relation="present",
        source_message_id=ids["narrator_1"],
        link_id="link-archived-endpoint",
    )
    thread = repositories.add_active_thread(
        save_id=save_id,
        title="Escape the ash corridor",
        source_message_id=ids["narrator_2"],
    )
    context_source = repositories.upsert_context_source(
        save_id=save_id,
        source_type="message",
        source_id=ids["narrator_2"],
        title="Deleted narrator message",
        body="The corridor floods with ash.",
        metadata={"source_message_ids": [ids["narrator_2"]]},
    )
    observation = repositories.add_context_observation(
        save_id=save_id,
        observation_type="open_thread",
        claim="The ash corridor may matter later.",
        evidence_quote="The corridor floods with ash.",
        source_message_ids=[ids["narrator_2"]],
        scope="save",
        status="accepted",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="upsert",
        entity_type="memory",
        field_path="body",
        proposed_value="Remember the ash corridor.",
        status="pending",
        source_message_ids=[ids["narrator_2"]],
    )
    audit = repositories.add_context_update_audit(
        save_id=save_id,
        operation="upsert",
        entity_type="memory",
        entity_id=memory.id,
        field_path="body",
        before=None,
        after={"body": memory.body},
        source_message_ids=[ids["narrator_2"]],
    )

    deletion = MessageRevisionService(repositories).delete_suffix_from_message(
        save_id=save_id,
        message_id=ids["narrator_2"],
    )

    assert deletion.anchor_message_id == ids["player_2"]
    assert [message.id for message in deletion.deleted_messages] == [
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]
    assert deletion.archived_media_asset_ids == frozenset({image.id, video.id})
    assert deletion.expired_context_update_suggestion_ids == frozenset(
        {suggestion.id}
    )
    assert deletion.archived_context_observation_ids == frozenset({observation.id})
    assert deletion.deleted_scene_snapshot_id is not None
    assert deletion.archived_location_ids == frozenset({location.id})
    assert deletion.archived_character_ids == frozenset({character.id})
    assert deletion.archived_active_thread_ids == frozenset({thread.id})

    assert [message.id for message in repositories.list_messages(save_id)] == [
        ids["player_1"],
        ids["narrator_1"],
    ]
    audit_messages = repositories.list_messages(save_id, include_deleted=True)
    assert [message.id for message in audit_messages] == [
        ids["player_1"],
        ids["narrator_1"],
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]
    assert all(message.deleted_at is None for message in audit_messages[:2])
    assert all(message.deleted_at is not None for message in audit_messages[2:])
    assert [
        (state.key, state.value) for state in repositories.list_world_state(save_id)
    ] == [
        ("scene.location", {"name": "Beacon tower"}),
    ]
    assert repositories.list_memories(save_id) == [linked_memory]
    assert repositories.list_summaries(save_id) == []
    assert repositories.list_media_assets(save_id) == []
    assert repositories.get_scene_snapshot(save_id) is None
    assert repositories.list_locations(save_id) == []
    assert repositories.list_characters(save_id) == []
    assert repositories.list_active_threads(save_id) == []
    assert repositories.list_entity_links(save_id) == []
    assert deletion.deleted_entity_link_ids == frozenset(
        {source_link.id, endpoint_link.id}
    )
    assert all(
        source.id != context_source.id
        for source in repositories.list_context_sources(save_id)
    )
    assert repositories.list_context_observations(save_id) == []
    assert repositories.list_context_update_suggestions(save_id)[0].status == "expired"
    assert repositories.list_context_update_audit(save_id) == [audit]


def test_rollback_archives_newer_world_data_context_and_prunes_scene_references(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    old_location = repositories.add_location(
        save_id=save_id,
        name="Beacon Tower",
        source_message_id=ids["narrator_1"],
    )
    stale_location = repositories.add_location(
        save_id=save_id,
        name="Ash Corridor",
        source_message_id=ids["narrator_2"],
    )
    old_character = repositories.add_character(
        save_id=save_id,
        name="Mara",
        source_message_id=ids["narrator_1"],
    )
    stale_character = repositories.add_character(
        save_id=save_id,
        name="Ash Shadow",
        location_id=stale_location.id,
        source_message_id=ids["narrator_2"],
    )
    thread = repositories.add_active_thread(
        save_id=save_id,
        title="Escape the ash corridor",
        source_message_id=ids["narrator_2"],
    )
    snapshot = repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=stale_location.id,
        situation="The old scene persists.",
        in_world_time="Cycle 3 night",
        time_of_day="night",
        day_of_week="cycle 3",
        world_day_index=3,
        world_time_day_index=3,
        world_time_day_label="cycle 3",
        world_time_phase="night",
        world_time_source_message_id=ids["narrator_2"],
        world_time_confidence=0.89,
        present_character_ids=[old_character.id, stale_character.id],
        source_message_id=ids["narrator_1"],
    )
    repositories.add_context_update_audit(
        save_id=save_id,
        operation="updated",
        entity_type="scene_snapshot",
        entity_id=snapshot.id,
        field_path="in_world_time",
        before="Monday morning",
        after="Cycle 3 night",
        reason="The discarded turn advanced scene time.",
        confidence=0.89,
        source_message_ids=[ids["narrator_2"]],
    )
    context_source = repositories.upsert_context_source(
        save_id=save_id,
        source_type="message",
        source_id=ids["narrator_2"],
        title="Discarded message",
        body="The corridor floods with ash.",
        metadata={"source_message_ids": [ids["narrator_2"]]},
    )
    observation = repositories.add_context_observation(
        save_id=save_id,
        observation_type="open_thread",
        claim="The ash corridor may matter later.",
        evidence_quote="The corridor floods with ash.",
        source_message_ids=[ids["narrator_2"]],
        scope="save",
        status="accepted",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="update",
        entity_type="scene_snapshot",
        entity_id=None,
        field_path="situation",
        proposed_value="The corridor floods with ash.",
        source_message_ids=[ids["narrator_2"]],
    )
    endpoint_link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=stale_character.id,
        target_type="location",
        target_id=stale_location.id,
        relation="present",
        source_message_id=ids["narrator_1"],
        link_id="link-stale-world-data",
    )
    preserved_link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=old_character.id,
        target_type="location",
        target_id=old_location.id,
        relation="present",
        source_message_id=ids["narrator_1"],
        link_id="link-preserved-world-data",
    )

    rollback = MessageRevisionService(repositories).rollback_from_message(
        save_id=save_id,
        message_id=ids["player_2"],
    )

    assert rollback.expired_context_update_suggestion_ids == frozenset(
        {suggestion.id}
    )
    assert rollback.archived_context_source_ids == frozenset({context_source.id})
    assert rollback.archived_context_observation_ids == frozenset({observation.id})
    assert rollback.archived_location_ids == frozenset({stale_location.id})
    assert rollback.archived_character_ids == frozenset({stale_character.id})
    assert rollback.archived_active_thread_ids == frozenset({thread.id})
    assert rollback.deleted_entity_link_ids == frozenset({endpoint_link.id})
    assert repositories.list_locations(save_id) == [old_location]
    assert repositories.list_characters(save_id) == [old_character]
    assert repositories.list_active_threads(save_id) == []
    assert repositories.list_entity_links(save_id) == [preserved_link]
    restored_snapshot = repositories.get_scene_snapshot(save_id)
    assert restored_snapshot is not None
    assert restored_snapshot.current_location_id is None
    assert restored_snapshot.present_character_ids == [old_character.id]
    assert restored_snapshot.in_world_time == "Monday morning"
    assert restored_snapshot.world_time_phase == "morning"
    assert restored_snapshot.world_time_source_message_id == ids["narrator_1"]
    assert restored_snapshot.world_time_confidence is None
    assert repositories.list_context_update_suggestions(save_id)[0].status == "expired"
    assert context_source.id not in {
        source.id for source in repositories.list_context_sources(save_id)
    }
    assert repositories.list_context_observations(save_id) == []


def test_rollback_preserves_protected_character_sourced_from_deleted_message(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    character = repositories.add_character(
        save_id=save_id,
        name="Captain Ilyra",
        role="Beacon watch commander",
        known_state="Keeps the tower safe.",
        protected_from_maintenance=True,
        source_message_id=ids["narrator_2"],
    )

    rollback = MessageRevisionService(repositories).rollback_from_message(
        save_id=save_id,
        message_id=ids["player_2"],
    )

    assert [message.id for message in rollback.deleted_messages] == [
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]
    assert rollback.archived_character_ids == frozenset()
    assert repositories.list_characters(save_id) == [character]


def test_rollback_reverts_later_world_data_field_updates_without_archiving_record(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    character = repositories.add_character(
        save_id=save_id,
        name="Mara",
        status="watching the beacon",
        source_message_id=ids["narrator_1"],
    )
    repositories.update_character(
        replace(
            character,
            status="lost in the ash corridor",
            source_message_id=ids["narrator_2"],
            last_updated_message_id=ids["narrator_2"],
        )
    )
    repositories.add_context_update_audit(
        save_id=save_id,
        operation="updated",
        entity_type="character",
        entity_id=character.id,
        field_path="status",
        before="watching the beacon",
        after="lost in the ash corridor",
        source_message_ids=[ids["narrator_2"]],
    )

    rollback = MessageRevisionService(repositories).rollback_from_message(
        save_id=save_id,
        message_id=ids["player_2"],
    )

    assert rollback.archived_character_ids == frozenset()
    restored = repositories.get_character(character.id)
    assert restored is not None
    assert restored.status == "watching the beacon"
    assert restored.first_seen_message_id == ids["narrator_1"]
    assert restored.last_updated_message_id == ids["narrator_1"]
    assert restored.source_message_id == ids["narrator_1"]


def test_rollback_preserves_legacy_scene_time_for_day_index_audit(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    snapshot = repositories.upsert_scene_snapshot(
        save_id=save_id,
        in_world_time="Friday festival week evening",
        time_of_day="evening",
        day_of_week="friday",
        world_day_index=9,
        world_time_day_index=9,
        world_time_day_label="friday",
        world_time_phase="evening",
        world_time_period_label="festival week",
        source_message_id=ids["narrator_2"],
        first_seen_message_id=ids["narrator_1"],
        last_updated_message_id=ids["narrator_2"],
    )
    repositories.connection.execute(
        """
        UPDATE scene_snapshots
        SET in_world_time = 'Friday evening after the festival'
        WHERE id = ?
        """,
        (snapshot.id,),
    )
    repositories.connection.commit()
    repositories.add_context_update_audit(
        save_id=save_id,
        operation="updated",
        entity_type="scene_snapshot",
        entity_id=snapshot.id,
        field_path="world_day_index",
        before=8,
        after=9,
        reason="The scene advanced one world day.",
        confidence=0.88,
        source_message_ids=[ids["narrator_2"]],
    )

    MessageRevisionService(repositories).rollback_from_message(
        save_id=save_id,
        message_id=ids["player_2"],
    )

    restored = repositories.get_scene_snapshot(save_id)
    assert restored is not None
    assert restored.in_world_time == "Friday evening after the festival"
    assert restored.time_of_day == "evening"
    assert restored.day_of_week == "friday"
    assert restored.world_day_index == 8
    assert restored.world_time_day_index == 8
    assert restored.world_time_day_label == "friday"
    assert restored.world_time_phase == "evening"
    assert restored.world_time_period_label == "festival week"


def test_restore_resubmission_restores_world_data_context_after_failed_regenerate(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    location = repositories.add_location(
        save_id=save_id,
        name="Ash Corridor",
        source_message_id=ids["narrator_2"],
    )
    character = repositories.add_character(
        save_id=save_id,
        name="Ash Shadow",
        location_id=location.id,
        source_message_id=ids["narrator_2"],
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=location.id,
        situation="The corridor floods with ash.",
        in_world_time="Cycle 4 festival week night at 21:15",
        world_time_day_index=4,
        world_time_day_label="Cycle 4",
        world_time_phase="night",
        world_time_clock_minutes=21 * 60 + 15,
        world_time_period_label="festival week",
        world_time_source_message_id=ids["narrator_2"],
        world_time_confidence=0.91,
        present_character_ids=[character.id],
        source_message_id=ids["narrator_2"],
    )
    context_source = repositories.upsert_context_source(
        save_id=save_id,
        source_type="message",
        source_id=ids["narrator_2"],
        title="Discarded message",
        body="The corridor floods with ash.",
        metadata={"source_message_ids": [ids["narrator_2"]]},
    )
    observation = repositories.add_context_observation(
        save_id=save_id,
        observation_type="open_thread",
        claim="The ash corridor may matter later.",
        evidence_quote="The corridor floods with ash.",
        source_message_ids=[ids["narrator_2"]],
        scope="save",
        status="accepted",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="update",
        entity_type="scene_snapshot",
        entity_id=None,
        field_path="situation",
        proposed_value="The corridor floods with ash.",
        source_message_ids=[ids["narrator_2"]],
    )
    link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character.id,
        target_type="location",
        target_id=location.id,
        relation="present",
        source_message_id=ids["narrator_2"],
        link_id="link-regenerate-context",
    )
    service = MessageRevisionService(repositories)
    revision = service.regenerate_message(
        save_id=save_id,
        message_id=ids["narrator_2"],
    )
    assert revision.archived_context_observation_ids == frozenset({observation.id})
    active_message_ids = frozenset(
        message.id for message in repositories.list_messages(save_id)
    )
    active_summary_ids = frozenset(
        summary.id for summary in repositories.list_summaries(save_id)
    )
    replacement = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Mara",
        body=revision.body,
    )

    service.restore_resubmission(
        save_id=save_id,
        revision=revision,
        active_message_ids_before_resubmission=active_message_ids,
        active_summary_ids_before_resubmission=active_summary_ids,
    )

    assert replacement.id not in {
        message.id for message in repositories.list_messages(save_id)
    }
    assert [message.id for message in repositories.list_messages(save_id)] == [
        ids["player_1"],
        ids["narrator_1"],
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]
    assert repositories.list_locations(save_id) == [location]
    assert repositories.list_characters(save_id) == [character]
    restored_snapshot = repositories.get_scene_snapshot(save_id)
    assert restored_snapshot is not None
    assert restored_snapshot.world_time_day_index == 4
    assert restored_snapshot.world_time_day_label == "Cycle 4"
    assert restored_snapshot.world_time_phase == "night"
    assert restored_snapshot.world_time_clock_minutes == 21 * 60 + 15
    assert restored_snapshot.world_time_period_label == "festival week"
    assert restored_snapshot.world_time_source_message_id == ids["narrator_2"]
    assert restored_snapshot.world_time_confidence == 0.91
    assert context_source.id in {
        source.id for source in repositories.list_context_sources(save_id)
    }
    observation_ids = [
        record.id for record in repositories.list_context_observations(save_id)
    ]
    assert observation_ids == [observation.id]
    assert repositories.list_context_update_suggestions(save_id)[0].status == "pending"
    assert repositories.list_context_update_suggestions(save_id)[0].id == suggestion.id
    assert repositories.list_entity_links(save_id) == [link]


def test_delete_suffix_from_player_uses_selected_player_as_anchor(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)

    deletion = MessageRevisionService(repositories).delete_suffix_from_message(
        save_id=save_id,
        message_id=ids["player_2"],
    )

    assert deletion.anchor_message_id == ids["player_2"]
    assert [message.id for message in deletion.deleted_messages] == [
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]
    assert [message.id for message in repositories.list_messages(save_id)] == [
        ids["player_1"],
        ids["narrator_1"],
    ]


def test_delete_suffix_snapshot_restore_preserves_protected_character(
    repositories: PersistenceRepositories,
) -> None:
    save_id, ids = _create_revision_save(repositories)
    TurnSnapshotService(repositories).capture_message_snapshot(
        save_id=save_id,
        message_id=ids["narrator_1"],
    )
    character = repositories.add_character(
        save_id=save_id,
        name="Captain Ilyra",
        role="Beacon watch commander",
        known_state="Keeps the tower safe.",
        protected_from_maintenance=True,
        source_message_id=ids["narrator_2"],
    )

    deletion = MessageRevisionService(repositories).delete_suffix_from_message(
        save_id=save_id,
        message_id=ids["player_2"],
    )

    assert [message.id for message in deletion.deleted_messages] == [
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]
    assert deletion.archived_character_ids == frozenset()
    assert repositories.list_characters(save_id) == [character]


def test_delete_suffix_from_opening_narrator_without_prior_player_deletes_from_itself(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Opening Watch")
    opening = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon gutters in the tower.",
        provider="fake",
        model="fake-chat",
    )
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I light the beacon.",
    )

    deletion = MessageRevisionService(repositories).delete_suffix_from_message(
        save_id=save.id,
        message_id=opening.id,
    )

    assert deletion.anchor_message_id == opening.id
    assert [message.id for message in deletion.deleted_messages] == [
        opening.id,
        player.id,
    ]
    assert repositories.list_messages(save.id) == []


def test_revision_restore_only_restores_memory_and_summary_archived_by_rollback(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Revision Watch")
    first_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I light the beacon.",
    )
    first_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon wakes.",
        provider="fake",
        model="fake-chat",
    )
    revised_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I open the sealed door.",
    )
    revised_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The corridor floods with ash.",
        provider="fake",
        model="fake-chat",
    )
    rollback_memory = repositories.add_memory(
        save_id=save.id,
        body="Mara opened the sealed ash corridor.",
        tags=["ash", "door"],
        source_message_id=revised_narrator.id,
    )
    manually_archived_memory = repositories.add_memory(
        save_id=save.id,
        body="A manually archived memory tied to the restored narrator.",
        tags=["manual"],
        source_message_id=revised_narrator.id,
    )
    repositories.archive_memory(manually_archived_memory.id)
    stable_prior_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=first_player.id,
        covers_message_end_id=first_narrator.id,
        body="Mara lit the beacon before reaching the sealed door.",
        provider="fake",
        model="fake-chat",
    )
    rollback_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=revised_player.id,
        covers_message_end_id=revised_narrator.id,
        body="Mara opened the door and ash flooded the corridor.",
        provider="fake",
        model="fake-chat",
    )
    manually_archived_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=revised_player.id,
        covers_message_end_id=revised_narrator.id,
        body="A manually archived summary tied to the restored branch.",
        provider="fake",
        model="fake-chat",
    )
    repositories.archive_summary(manually_archived_summary.id)
    service = MessageRevisionService(repositories)

    revision = service.edit_and_resubmit_message(
        save_id=save.id,
        message_id=revised_player.id,
        body="I knock once and listen at the sealed door.",
    )
    active_message_ids_before_resubmission = frozenset(
        message.id for message in repositories.list_messages(save.id)
    )
    active_summary_ids_before_resubmission = frozenset(
        summary.id for summary in repositories.list_summaries(save.id)
    )
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body=revision.body,
    )
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=f"fake narrator: {revision.body}",
        provider="fake",
        model="fake-chat",
    )
    repositories.archive_summary(stable_prior_summary.id)
    failed_attempt_prior_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=first_player.id,
        covers_message_end_id=first_narrator.id,
        body="A failed replacement summarized only pre-replacement messages.",
        provider="fake",
        model="fake-chat",
    )

    assert revision.archived_memory_ids == frozenset({rollback_memory.id})
    assert revision.archived_summary_ids == frozenset({rollback_summary.id})
    assert active_summary_ids_before_resubmission == frozenset(
        {stable_prior_summary.id}
    )
    summary_rows_before_restore = repositories.connection.execute(
        """
        SELECT id, archived_at
        FROM summaries
        WHERE id IN (?, ?)
        """,
        (stable_prior_summary.id, failed_attempt_prior_summary.id),
    ).fetchall()
    summary_archive_status_before_restore = {
        row["id"]: row["archived_at"] for row in summary_rows_before_restore
    }
    assert summary_archive_status_before_restore[stable_prior_summary.id] is not None
    assert (
        summary_archive_status_before_restore[failed_attempt_prior_summary.id] is None
    )

    service.restore_resubmission(
        save_id=save.id,
        revision=revision,
        active_message_ids_before_resubmission=active_message_ids_before_resubmission,
        active_summary_ids_before_resubmission=active_summary_ids_before_resubmission,
    )

    assert [message.id for message in repositories.list_messages(save.id)] == [
        first_player.id,
        first_narrator.id,
        revised_player.id,
        revised_narrator.id,
    ]
    memory_rows = repositories.connection.execute(
        """
        SELECT id, archived_at
        FROM memories
        WHERE id IN (?, ?)
        """,
        (rollback_memory.id, manually_archived_memory.id),
    ).fetchall()
    memory_archive_status = {row["id"]: row["archived_at"] for row in memory_rows}
    assert memory_archive_status[rollback_memory.id] is None
    assert memory_archive_status[manually_archived_memory.id] is not None

    summary_rows = repositories.connection.execute(
        """
        SELECT id, archived_at
        FROM summaries
        WHERE id IN (?, ?, ?, ?)
        """,
        (
            stable_prior_summary.id,
            rollback_summary.id,
            manually_archived_summary.id,
            failed_attempt_prior_summary.id,
        ),
    ).fetchall()
    summary_archive_status = {row["id"]: row["archived_at"] for row in summary_rows}
    assert summary_archive_status[stable_prior_summary.id] is None
    assert summary_archive_status[rollback_summary.id] is None
    assert summary_archive_status[manually_archived_summary.id] is not None
    assert summary_archive_status[failed_attempt_prior_summary.id] is not None


def _load_content(content_json: str) -> dict[str, object]:
    loaded = json.loads(content_json)
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)


def _create_revision_save(
    repositories: PersistenceRepositories,
) -> tuple[str, dict[str, str]]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "player_character_name": "Mara Voss",
            "starting_scene": "The beacon gutters in the tower.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Revision Watch")
    messages = {
        "player_1": repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I light the beacon.",
        ),
        "narrator_1": repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="The beacon wakes.",
            provider="fake",
            model="fake-chat",
        ),
        "player_2": repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I open the sealed door.",
        ),
        "narrator_2": repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="The corridor floods with ash.",
            provider="fake",
            model="fake-chat",
        ),
        "player_3": repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I step through.",
        ),
        "narrator_3": repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="A shadow follows.",
            provider="fake",
            model="fake-chat",
        ),
    }
    return save.id, {name: message.id for name, message in messages.items()}
