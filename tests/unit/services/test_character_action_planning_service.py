from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.interaction_mode import InteractionMode
from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.services.character_action_planning_service import (
    CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
    CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
    CHARACTER_ACTION_PLANNING_TASK,
    CharacterActionPlanningService,
    _character_presence_messages,
    _planning_characters_for_turn,
    _planning_evidence_sources,
    character_action_planning_enabled,
    format_character_turn_assessment,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_character_action_planning_updates_presence_and_returns_present_plans(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara lowers the storm lantern and asks what changed.",
                "intent": "keep the lens crew calm",
                "reason": "She is already beside the lantern in the scene.",
                "confidence": 0.92,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Ren": {
                "present": False,
                "action": "",
                "intent": "",
                "reason": "Ren is still cataloging the archives offscreen.",
                "confidence": 0.8,
                "evidence_source_ids": ["character:ren"],
            },
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message_id)
    )

    assert [(plan.character_name, plan.action) for plan in result.plans] == [
        ("Mara", "Mara lowers the storm lantern and asks what changed.")
    ]
    assert result.applied_presence_update is True
    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {
        characters["player"],
        characters["mara"],
    }
    assert characters["ren"] not in snapshot.present_character_ids
    assert [request.schema_name for request in provider.structured_output_requests] == [
        "character_presence_assessment",
        "character_presence_assessment",
        "character_intent_plan",
    ]
    assert "Decide only whether this character is present" in (
        provider.structured_output_requests[0].messages[0].body
    )
    intent_prompt = provider.structured_output_requests[-1].messages[0].body
    assert "Favor visible initiative over waiting for the player" in intent_prompt
    assert "interrupt, demand, refuse, leave, escalate" in intent_prompt
    assert "character_id" in provider.structured_output_requests[0].schema[
        "properties"
    ]
    schema = provider.structured_output_requests[0].schema
    for field in (
        "enters_scene",
        "leaves_scene",
    ):
        assert field in schema["properties"]
        assert field in schema["required"]
    intent_schema = provider.structured_output_requests[-1].schema
    for field in (
        "learned_memory_candidates",
        "knowledge_edge_candidates",
        "needs_review_notes",
    ):
        assert field in intent_schema["properties"]
        assert field in intent_schema["required"]


def test_character_action_planning_skips_presence_update_without_grounded_quote(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara keeps watch beside the lantern.",
                "intent": "guard the lens",
                "reason": "Mara is in the scene.",
                "confidence": 0.91,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Ren": {
                "present": False,
                "action": "Ren slips out of the beacon room.",
                "intent": "",
                "reason": "Ren is claimed absent without grounded evidence.",
                "confidence": 0.8,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
                "evidence_quote": "ruby library",
                "leaves_scene": True,
            },
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message_id)
    )

    assert result.applied_presence_update is False
    assert [(plan.character_name, plan.action) for plan in result.plans] == [
        ("Mara", "Mara keeps watch beside the lantern.")
    ]
    ren_decision = next(
        decision
        for decision in result.decisions
        if decision.character_id == characters["ren"]
    )
    assert ren_decision.evidence_source_ids == ()
    assert ren_decision.evidence_quote == ""
    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {
        characters["player"],
        characters["mara"],
        characters["ren"],
    }


def test_character_action_planning_drops_intent_guidance_with_ungrounded_quote(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A storm tower waits in the fog.",
        player_role="Signal keeper",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara steadies the storm lantern.",
        present_character_ids=[mara.id],
        snapshot_id="snapshot-1",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I ask Mara whether the corridor is clear.",
    )
    provider = SequenceCharacterDecisionProvider(
        (
            {
                "present": True,
                "enters_scene": False,
                "leaves_scene": False,
                "reason": "Mara is already present in the scene.",
                "confidence": 0.86,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            {
                "present": True,
                "action": "Mara checks the corridor.",
                "intent": "inspect the tower corridor",
                "reason": "The player asked Mara to check the corridor.",
                "confidence": 0.9,
                "evidence_source_ids": [f"message:{player_message.id}"],
                "evidence_quote": "ruby library",
                "learned_memory_candidates": [],
                "knowledge_edge_candidates": [],
                "needs_review_notes": [],
            },
        )
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=player_message.id)
    )

    assert result.plans == ()
    assert len(result.assessments) == 1
    assessment = result.assessments[0]
    assert assessment.character_id == mara.id
    assert assessment.present is True
    assert assessment.evidence_source_ids
    assert assessment.action == ""
    assert assessment.intent == ""


def test_character_action_planning_does_not_borrow_intent_evidence_for_presence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A storm tower waits in the fog.",
        player_role="Signal keeper",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The beacon gallery is empty.",
        present_character_ids=[],
        snapshot_id="snapshot-1",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I call Mara from the archive stairs.",
    )
    provider = SequenceCharacterDecisionProvider(
        (
            {
                "present": False,
                "enters_scene": True,
                "leaves_scene": False,
                "reason": "Mara is claimed to enter without support.",
                "confidence": 0.86,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
                "evidence_quote": "ruby library",
            },
            {
                "present": True,
                "action": "Mara answers from the stairwell.",
                "intent": "respond to Ily's call",
                "reason": "The player called Mara.",
                "confidence": 0.9,
                "evidence_source_ids": [f"message:{player_message.id}"],
                "evidence_quote": "call Mara",
                "learned_memory_candidates": [],
                "knowledge_edge_candidates": [],
                "needs_review_notes": [],
            },
        )
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=player_message.id)
    )

    assert [request.schema_name for request in provider.structured_output_requests] == [
        "character_presence_assessment"
    ]
    assert result.plans == ()
    assert result.applied_presence_update is False
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert mara.id not in snapshot.present_character_ids


def test_character_action_planning_does_not_create_snapshot_for_ungrounded_absence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A storm tower waits in the fog.",
        player_role="Signal keeper",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    repositories.add_character(save_id=save.id, name="Mara", met=True)
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I ask whether Mara is here.",
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": False,
                "enters_scene": False,
                "leaves_scene": False,
                "action": "",
                "intent": "",
                "reason": "Mara is claimed absent without support.",
                "confidence": 0.8,
                "evidence_source_ids": [f"message:{player_message.id}"],
                "evidence_quote": "ruby library",
            }
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=player_message.id)
    )

    assert result.applied_presence_update is False
    assert repositories.get_scene_snapshot(save.id) is None


def test_character_action_planning_skips_unmentioned_offscreen_characters(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A storm tower waits in the fog.",
        player_role="Signal keeper",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.add_character(save_id=save.id, name="Ren", met=True)
    repositories.add_character(save_id=save.id, name="Talla", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara steadies the storm lantern.",
        present_character_ids=[mara.id],
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I ask Mara what she sees in the lens.",
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara studies the lens.",
                "intent": "answer Ily",
                "reason": "Mara is in the current scene.",
                "confidence": 0.9,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=player_message.id)
    )

    requested_names = [
        _requested_character_name(request.messages[-1].body)
        for request in provider.structured_output_requests
    ]
    assert requested_names == ["Mara", "Mara"]
    assert [assessment.character_name for assessment in result.assessments] == [
        "Mara"
    ]


def test_character_action_planning_includes_named_offscreen_possible_entrant(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A storm tower waits in the fog.",
        player_role="Signal keeper",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    ren = repositories.add_character(
        save_id=save.id,
        name="Archivist Ren",
        aliases=["Ren"],
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara steadies the storm lantern.",
        present_character_ids=[mara.id],
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I call for Ren to join us by the lens.",
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara keeps watch.",
                "intent": "hold the room",
                "reason": "Mara is already present.",
                "confidence": 0.88,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Archivist Ren": {
                "present": False,
                "enters_scene": True,
                "action": "Ren steps in from the archive stairs.",
                "intent": "answer Ily's call",
                "reason": "Ily explicitly called for Ren.",
                "confidence": 0.86,
                "evidence_source_ids": [f"message:{player_message.id}"],
            },
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=player_message.id)
    )

    requested_names = [
        _requested_character_name(request.messages[-1].body)
        for request in provider.structured_output_requests
    ]
    assert requested_names == [
        "Mara",
        "Archivist Ren",
        "Mara",
        "Archivist Ren",
    ]
    assert [plan.character_name for plan in result.plans] == [
        "Mara",
        "Archivist Ren",
    ]
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert ren.id in snapshot.present_character_ids


def test_character_action_planning_can_return_tentative_presence_without_mutating_scene(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara keeps watch beside the lantern.",
                "intent": "guard the lens",
                "reason": "Mara is in the scene.",
                "confidence": 0.91,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Ren": {
                "present": False,
                "action": "Ren leaves the gallery for the archive stairs.",
                "intent": "check the old maps",
                "reason": "Ren decides to leave during the planned beat.",
                "confidence": 0.86,
                "evidence_source_ids": ["message:latest"],
                "leaves_scene": True,
            },
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            apply_presence_updates=False,
        )
    )

    assert result.applied_presence_update is False
    assert [decision.character_id for decision in result.decisions] == [
        characters["mara"],
        characters["ren"],
    ]
    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {
        characters["player"],
        characters["mara"],
        characters["ren"],
    }


def test_character_action_planning_skips_player_character(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara watches the player for a cue.",
                "intent": "wait for confirmation",
                "reason": "The player addressed her directly.",
                "confidence": 0.75,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            }
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message_id)
    )

    request_text = "\n\n".join(
        message.body
        for request in provider.structured_output_requests
        for message in request.messages
    )
    assert "Player character: Ily" in request_text
    assert "Character: Ily" not in request_text
    assert [plan.character_id for plan in result.plans] == [characters["mara"]]


def test_character_action_planning_filters_recent_messages_hidden_from_character(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A storm tower waits in the fog.",
        player_role="Signal keeper",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    ren = repositories.add_character(save_id=save.id, name="Ren", met=True)
    hidden = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The gate password is hidden from the lens crew.",
    )
    for character_id in (mara.id, ren.id):
        repositories.add_message_visibility(
            save_id=save.id,
            message_id=hidden.id,
            character_id=character_id,
            visibility="not_visible",
            confidence=1.0,
            source="scene_presence",
            evidence="The character was not present.",
        )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara and Ren are by the lens.",
        present_character_ids=[mara.id, ren.id],
    )
    source_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I ask what everyone does next.",
    )
    provider = EchoHiddenPromptDecisionProvider(
        {
            "Mara": {
                "present": True,
                "intent": "stay alert",
                "reason": "Mara is in the scene.",
                "confidence": 0.8,
                "evidence_source_ids": [],
            },
            "Ren": {
                "present": True,
                "intent": "watch the archive satchel",
                "reason": "Ren is in the scene.",
                "confidence": 0.8,
                "evidence_source_ids": [],
            },
        },
        hidden_text="gate password",
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=source_message.id)
    )

    request_text = "\n\n".join(
        message.body
        for request in provider.structured_output_requests
        for message in request.messages
    )
    plan_text = "\n".join(plan.action for plan in result.plans)
    assert hidden.body not in request_text
    assert "gate password" not in plan_text


def test_character_action_planning_includes_active_threads(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, _characters = _create_save_with_characters(
        repositories
    )
    repositories.add_active_thread(
        save_id=save_id,
        title="Guard search",
        description="Guards are sweeping toward the beacon room.",
        status="active",
        priority=3,
        visibility="scene",
        related_entities=["director_pressure"],
        source_message_id=player_message_id,
    )
    repositories.add_active_thread(
        save_id=save_id,
        title="Secret accord",
        description="Ren privately promised Mara a hidden favor.",
        status="active",
        priority=4,
        visibility="private",
        related_entities=["character:ren"],
        source_message_id=player_message_id,
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara shutters the lantern.",
                "intent": "avoid the guard search",
                "reason": "The active guard search raises the risk.",
                "confidence": 0.9,
                "evidence_source_ids": [],
            },
            "Ren": {
                "present": True,
                "action": "Ren hides the archive satchel.",
                "intent": "protect contraband notes",
                "reason": "The active guard search threatens the archive.",
                "confidence": 0.84,
                "evidence_source_ids": [],
            },
        }
    )
    _configure_planning(repositories)

    asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message_id)
    )

    request_text = "\n\n".join(
        message.body
        for request in provider.structured_output_requests
        for message in request.messages
    )
    assert "Active threads:" in request_text
    assert "Guard search (active, priority 3)" in request_text
    assert "Guards are sweeping toward the beacon room." in request_text
    assert "Secret accord" not in request_text
    assert "hidden favor" not in request_text


def test_character_turn_assessments_apply_entering_and_leaving(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Mara guards the lantern while Ren approaches from the stairs.",
        present_character_ids=[characters["mara"]],
        snapshot_id="snapshot-1",
    )
    player_message_id = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I ask Ren to take over from Mara at the lens.",
    ).id
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "enters_scene": False,
                "leaves_scene": True,
                "action": "Mara hands off the storm lantern and exits.",
                "intent": "leave Ren to inspect the lens",
                "reason": "The player asked Ren to take over.",
                "confidence": 0.83,
                "evidence_source_ids": ["message:player"],
            },
            "Ren": {
                "present": False,
                "enters_scene": True,
                "leaves_scene": False,
                "action": "Ren steps into the gallery and studies the lens.",
                "intent": "inspect the beacon mechanism",
                "reason": "The player called Ren into the room.",
                "confidence": 0.86,
                "evidence_source_ids": ["message:player"],
            },
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message_id)
    )

    assert [(plan.character_name, plan.action) for plan in result.plans] == [
        ("Mara", "Mara hands off the storm lantern and exits."),
        ("Ren", "Ren steps into the gallery and studies the lens."),
    ]
    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert characters["mara"] not in snapshot.present_character_ids
    assert characters["ren"] in snapshot.present_character_ids
    assert result.assessments[0].leaves_scene is True
    assert result.assessments[1].enters_scene is True


def test_character_turn_assessment_keeps_shadow_memory_and_edge_candidates(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A storm tower waits in the fog.",
        player_role="Signal keeper",
        content={
            "player_character_name": "Ily",
            "beacon_protocol": "The red lens responds to an ember-dawn phrase.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara waits by the beacon controls.",
        present_character_ids=[mara.id],
        snapshot_id="snapshot-shadow-candidates",
    )
    source = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I tell Mara the lens-key phrase is ember dawn.",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The lens-key phrase is ember dawn.",
        tags=["beacon"],
        memory_id="memory-lens-key",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"status": "red"},
        state_id="state-beacon-lens",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=source.id,
        covers_message_end_id=source.id,
        body="Ily told Mara the lens-key phrase.",
        provider="fake",
        model="summary",
        summary_id="summary-lens-key",
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara repeats the ember-dawn phrase under her breath.",
                "intent": "remember the lens key",
                "reason": "The latest message directly tells her the phrase.",
                "confidence": 0.91,
                "evidence_source_ids": [f"message:{source.id}"],
                "learned_memory_candidates": [
                    {
                        "body": "Mara learned that the lens-key phrase is ember dawn.",
                        "tags": ["mara", "beacon"],
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "Ily directly told Mara.",
                        "confidence": 0.9,
                        "evidence_source_ids": [f"message:{source.id}"],
                        "evidence_quote": "the lens-key phrase is ember dawn",
                    }
                ],
                "knowledge_edge_candidates": [
                    {
                        "target_type": "memory",
                        "target_id": memory.id,
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "The source message teaches the memory fact.",
                        "confidence": 0.88,
                        "evidence_source_ids": [f"message:{source.id}"],
                        "evidence_quote": "the lens-key phrase is ember dawn",
                    },
                    {
                        "target_type": "world_state",
                        "target_id": state.id,
                        "knowledge_state": "may_know",
                        "acquisition_method": "inferred_from_visible_consequence",
                        "reason": "The red lens is visible in the scene.",
                        "confidence": 0.72,
                        "evidence_source_ids": [
                            "scene_snapshot:snapshot-shadow-candidates"
                        ],
                        "evidence_quote": "Mara waits by the beacon controls.",
                    },
                    {
                        "target_type": "summary",
                        "target_id": summary.id,
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "The summary covers the same direct disclosure.",
                        "confidence": 0.81,
                        "evidence_source_ids": [f"message:{source.id}"],
                        "evidence_quote": "lens-key phrase is ember dawn",
                    },
                    {
                        "target_type": "scenario_section",
                        "target_id": f"scenario:{scenario.id}:section:beacon_protocol",
                        "knowledge_state": "may_know",
                        "acquisition_method": "told",
                        "reason": "The told phrase touches this scenario section.",
                        "confidence": 0.7,
                        "evidence_source_ids": [f"message:{source.id}"],
                        "evidence_quote": "ember dawn",
                    },
                    {
                        "target_type": "memory",
                        "target_id": "memory-other-save",
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "This should be ignored.",
                        "confidence": 1.0,
                        "evidence_source_ids": [f"message:{source.id}"],
                        "evidence_quote": "bad target",
                    },
                ],
                "needs_review_notes": [
                    "Confirm whether Mara should retain this as durable memory."
                ],
            }
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=source.id)
    )

    assessment = result.assessments[0]
    assert assessment.learned_memory_candidates[0].body == (
        "Mara learned that the lens-key phrase is ember dawn."
    )
    assert assessment.learned_memory_candidates[0].evidence_source_ids == (
        f"message:{source.id}",
    )
    target_ids = [
        candidate.target_id for candidate in assessment.knowledge_edge_candidates
    ]
    assert target_ids == [
        memory.id,
        state.id,
        summary.id,
        f"scenario:{scenario.id}:section:beacon_protocol",
    ]
    assert assessment.needs_review_notes == (
        "Confirm whether Mara should retain this as durable memory.",
    )
    assert len(repositories.list_memories(save.id)) == 1
    assert repositories.list_character_knowledge_edges(save.id) == []
    formatted = format_character_turn_assessment(assessment)
    assert "learned memory candidate (do not persist automatically)" in formatted
    assert "knowledge edge candidate (do not persist automatically)" in formatted
    assert f"evidence: message:{source.id}" in formatted


def test_character_action_planning_exposes_allowed_evidence_source_ids(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, _characters = _create_save_with_characters(
        repositories
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara studies the lens.",
                "intent": "answer Ily",
                "reason": "Mara is in the current scene.",
                "confidence": 0.9,
                "evidence_source_ids": [f"message:{player_message_id}"],
            },
            "Ren": {
                "present": False,
                "action": "",
                "intent": "",
                "reason": "Ren is offscreen.",
                "confidence": 0.8,
                "evidence_source_ids": [],
            },
        }
    )
    _configure_planning(repositories)

    asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message_id)
    )

    intent_request = provider.structured_output_requests[-1]
    evidence_schema = intent_request.schema["properties"]["evidence_source_ids"]
    assert f"message:{player_message_id}" in evidence_schema["items"]["enum"]
    assert f"message:{player_message_id}" in intent_request.messages[-1].body


def test_character_action_planning_drops_ungrounded_candidate_evidence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A storm tower waits in the fog.",
        player_role="Signal keeper",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara waits by the beacon controls.",
        present_character_ids=[mara.id],
        snapshot_id="snapshot-1",
    )
    source = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I tell Mara the lens-key phrase is ember dawn.",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The beacon answers to ember dawn.",
        tags=["beacon"],
        memory_id="memory-beacon-key",
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara repeats the ember-dawn phrase under her breath.",
                "intent": "remember the lens key",
                "reason": "The latest message directly tells her the phrase.",
                "confidence": 0.91,
                "evidence_source_ids": [f"message:{source.id}"],
                "learned_memory_candidates": [
                    {
                        "body": "Mara learned that the lens-key phrase is ember dawn.",
                        "tags": ["mara", "beacon"],
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "Ily directly told Mara.",
                        "confidence": 0.9,
                        "evidence_source_ids": [f"message:{source.id}"],
                        "evidence_quote": "lens-key phrase is ember dawn",
                    },
                    {
                        "body": "Mara learned a phrase from a missing message.",
                        "tags": ["mara"],
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "This source id is invalid.",
                        "confidence": 0.9,
                        "evidence_source_ids": ["message:missing"],
                        "evidence_quote": "lens-key phrase is ember dawn",
                    },
                    {
                        "body": "Mara learned an unsupported library clue.",
                        "tags": ["mara"],
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "This quote is invalid.",
                        "confidence": 0.9,
                        "evidence_source_ids": [f"message:{source.id}"],
                        "evidence_quote": "ruby library",
                    },
                    {
                        "body": "Mara learned without citing her own source.",
                        "tags": ["mara"],
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "Nested candidates must cite their own source.",
                        "confidence": 0.9,
                        "evidence_quote": "lens-key phrase is ember dawn",
                    },
                ],
                "knowledge_edge_candidates": [
                    {
                        "target_type": "memory",
                        "target_id": memory.id,
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "The source message teaches the memory fact.",
                        "confidence": 0.88,
                        "evidence_source_ids": [f"message:{source.id}"],
                        "evidence_quote": "ember dawn",
                    },
                    {
                        "target_type": "memory",
                        "target_id": memory.id,
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "The quote is not grounded.",
                        "confidence": 0.88,
                        "evidence_source_ids": [f"message:{source.id}"],
                        "evidence_quote": "ruby library",
                    },
                    {
                        "target_type": "memory",
                        "target_id": memory.id,
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "Nested edge candidates must cite their own source.",
                        "confidence": 0.88,
                        "evidence_quote": "ember dawn",
                    },
                ],
                "needs_review_notes": [],
            }
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=source.id)
    )

    assessment = result.assessments[0]
    assert [
        candidate.body for candidate in assessment.learned_memory_candidates
    ] == ["Mara learned that the lens-key phrase is ember dawn."]
    assert [
        candidate.evidence_quote for candidate in assessment.knowledge_edge_candidates
    ] == ["ember dawn"]


def test_character_action_planning_uses_configured_concurrency_cap(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A storm tower waits in the fog.",
        player_role="Signal keeper",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    names = ("Mara", "Ren", "Talla", "Ivo", "Senn")
    characters = [
        repositories.add_character(save_id=save.id, name=name, met=True)
        for name in names
    ]
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The whole lens crew waits in the beacon gallery.",
        present_character_ids=[character.id for character in characters],
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I ask everyone what they do next.",
    )
    provider = BlockingCharacterDecisionProvider(
        {
            name: {
                "present": True,
                "action": f"{name} acts.",
                "intent": "respond to the player",
                "reason": "The player asked the crew.",
                "confidence": 0.75,
                "evidence_source_ids": [],
            }
            for name in names
        },
        expected_concurrency=2,
    )
    _configure_planning(repositories)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
        value=2,
    )

    async def run_planning() -> None:
        task = asyncio.create_task(
            CharacterActionPlanningService(
                repositories=repositories,
                providers={"fake": provider},
            ).plan_for_turn(save_id=save.id, player_message_id=player_message.id)
        )
        await asyncio.wait_for(provider.expected_active.wait(), timeout=1)
        assert provider.max_active == 2
        assert len(provider.structured_output_requests) == 2
        provider.release.set()
        result = await task
        assert len(result.decisions) == len(names)

    asyncio.run(run_planning())
    assert provider.max_active == 2


def test_character_action_planning_includes_dating_route_escalation_policy(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={"player_character_name": "Ren Takahashi"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Ren Takahashi",
        met=True,
        is_player_character=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        met=True,
        relationships={player.name: "romance option for Ren Takahashi"},
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Ren and Mika linger near the festival gate.",
        present_character_ids=[npc.id],
        world_day_index=2,
    )
    source = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ren",
        body="I tell Mika I would like to see her again.",
    )
    repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="introduced",
        first_met_world_day_index=0,
        completed_interactions=1,
        dates_completed=0,
        next_reasonable_step="build early interest or exchange contact info",
    )
    provider = CharacterDecisionProvider(
        {
            "Mika Arai": {
                "present": True,
                "action": "Mika smiles and suggests exchanging numbers.",
                "intent": "show interest without overcommitting",
                "reason": "The route is still early.",
                "confidence": 0.83,
                "evidence_source_ids": [],
            }
        }
    )
    _configure_planning(repositories)

    asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=source.id)
    )

    prompt_text = provider.structured_output_requests[0].messages[-1].body
    assert "Dating route pacing for this character is deterministic state" in (
        prompt_text
    )
    assert "stage introduced" in prompt_text
    assert "known for 2 in-world days" in prompt_text
    assert "max plausible escalation warmth, curiosity" in prompt_text
    assert "premature now exclusivity or commitment language" in prompt_text


def test_character_action_planning_failure_preserves_existing_presence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": False,
                "action": "",
                "intent": "",
                "reason": "Mara stepped out before this beat.",
                "confidence": 0.7,
                "evidence_source_ids": [],
            }
        },
        fail_names={"Ren"},
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message_id)
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert characters["ren"] in snapshot.present_character_ids
    assert result.plans == ()
    assert result.failed_character_ids == (characters["ren"],)


def test_character_action_planning_skips_missing_catalog_row(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, _characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara takes a breath.",
                "intent": "steady the crew",
                "reason": "Mara is present in the scene.",
                "confidence": 0.7,
                "evidence_source_ids": [],
            }
        }
    )
    repositories.set_app_setting(CHARACTER_ACTION_PLANNING_ENABLED_SETTING, True)
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="unsynced-character-planning",
    )

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message_id)
    )

    assert result.skipped_reason == "model_missing"
    assert provider.structured_output_requests == []


def test_character_action_planning_is_enabled_by_default(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _, _ = _create_save_with_characters(repositories)

    assert character_action_planning_enabled(repositories, save_id=save_id) is True


def test_storyteller_planning_includes_all_characters_and_excludes_direction_evidence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="The Ceremony",
        premise="A rival waits in the wings.",
        player_role="",
        content={},
        interaction_mode=InteractionMode.STORYTELLER,
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Act One")
    rival = repositories.add_character(save_id=save.id, name="The Rival")
    witness = repositories.add_character(save_id=save.id, name="The Witness")
    direction = repositories.append_message(
        save_id=save.id,
        role="player",
        body="Have the rival interrupt the ceremony.",
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The orchestra prepares for the final movement.",
    )

    planned = _planning_characters_for_turn(
        repositories=repositories,
        save_id=save.id,
        source_message=direction,
    )
    assert {character.id for character in planned} == {rival.id, witness.id}
    evidence = _planning_evidence_sources(
        repositories=repositories,
        save_id=save.id,
        character=rival,
        source_message=direction,
        messages=(direction, narrator),
    )
    assert f"message:{direction.id}" not in evidence
    assert f"message:{narrator.id}" in evidence
    prompts = _character_presence_messages(
        repositories=repositories,
        save_id=save.id,
        character=rival,
        source_message=direction,
        messages=(direction, narrator),
        evidence_sources=evidence,
    )
    prompt = "\n".join(message.body for message in prompts)
    assert "narrator-controlled character" in prompt
    assert "non-diegetic story direction" in prompt
    assert "not canonical evidence" in prompt


def test_character_action_planning_can_be_disabled_per_save(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, _characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "action": "Mara takes a breath.",
                "intent": "",
                "reason": "",
                "confidence": 0.7,
                "evidence_source_ids": [],
            }
        }
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        value=False,
    )

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message_id)
    )

    assert result.skipped_reason == "disabled"
    assert provider.structured_output_requests == []


class CharacterDecisionProvider:
    provider_name = "fake"

    def __init__(
        self,
        decisions_by_name: dict[str, dict[str, object]],
        *,
        fail_names: set[str] | None = None,
    ) -> None:
        self.decisions_by_name = decisions_by_name
        self.fail_names = fail_names or set()
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def validate_config(self) -> ProviderConfigStatus:
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
        )

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-chat",
                display_name="Fake Chat",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise NotImplementedError

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        body = request.messages[-1].body
        name = _requested_character_name(body)
        if name in self.fail_names:
            raise ValueError(f"{name} planner failed")
        data = dict(self.decisions_by_name[name])
        data = _with_allowed_evidence_ids_and_quote(data, request)
        data["character_id"] = request.schema["properties"]["character_id"]["enum"][0]
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 7},
        )


class SequenceCharacterDecisionProvider(CharacterDecisionProvider):
    def __init__(
        self,
        decisions: tuple[dict[str, object], ...],
    ) -> None:
        super().__init__({})
        self.decisions = list(decisions)

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        if not self.decisions:
            raise AssertionError("no scripted character decision remaining")
        data = dict(self.decisions.pop(0))
        data = _with_allowed_evidence_ids_and_quote(data, request)
        data["character_id"] = request.schema["properties"]["character_id"]["enum"][0]
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 7},
        )


class EchoHiddenPromptDecisionProvider(CharacterDecisionProvider):
    def __init__(
        self,
        decisions_by_name: dict[str, dict[str, object]],
        *,
        hidden_text: str,
    ) -> None:
        super().__init__(decisions_by_name)
        self.hidden_text = hidden_text

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        body = request.messages[-1].body
        name = _requested_character_name(body)
        data = dict(self.decisions_by_name[name])
        data = _with_allowed_evidence_ids_and_quote(data, request)
        data["character_id"] = request.schema["properties"]["character_id"]["enum"][0]
        data["action"] = (
            f"{name} repeats the {self.hidden_text}."
            if self.hidden_text in body
            else f"{name} continues without hidden knowledge."
        )
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 7},
        )


class BlockingCharacterDecisionProvider(CharacterDecisionProvider):
    def __init__(
        self,
        decisions_by_name: dict[str, dict[str, object]],
        *,
        expected_concurrency: int,
    ) -> None:
        super().__init__(decisions_by_name)
        self.expected_concurrency = expected_concurrency
        self.expected_active = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        body = request.messages[-1].body
        name = _requested_character_name(body)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.max_active >= self.expected_concurrency:
            self.expected_active.set()
        try:
            await self.release.wait()
        finally:
            self.active -= 1
        data = dict(self.decisions_by_name[name])
        data = _with_allowed_evidence_ids_and_quote(data, request)
        data["character_id"] = request.schema["properties"]["character_id"]["enum"][0]
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 7},
        )


def _configure_planning(repositories: PersistenceRepositories) -> None:
    repositories.set_app_setting(CHARACTER_ACTION_PLANNING_ENABLED_SETTING, True)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )


def _create_save_with_characters(
    repositories: PersistenceRepositories,
) -> tuple[str, str, dict[str, str]]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A storm tower waits in the fog.",
        player_role="Signal keeper",
        content={
            "player_character_name": "Ily",
            "current_scene": "Ily stands by the lens with Mara nearby.",
            "opening_message": "The lens hums.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    player = repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(
        save_id=save.id,
        name="Mara",
        met=True,
        role="watch captain",
        current_intent="keep the storm lantern lit",
    )
    ren = repositories.add_character(
        save_id=save.id,
        name="Ren",
        aliases=["Archivist Ren"],
        met=True,
        role="archivist",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara steadies the storm lantern while Ren is offscreen.",
        present_character_ids=[mara.id, ren.id],
        snapshot_id="snapshot-1",
    )
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Mara steadies the storm lantern.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I ask Mara what she sees in the lens.",
    )
    return save.id, player_message.id, {
        "player": player.id,
        "mara": mara.id,
        "ren": ren.id,
    }


def _requested_character_name(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("Character: "):
            return line.removeprefix("Character: ").strip()
    raise AssertionError("request did not include a Character line")


def _with_allowed_evidence_ids_and_quote(
    value: object,
    request: StructuredOutputRequest,
) -> dict[str, object]:
    source_texts = _evidence_source_texts(request)

    def normalize(source_id: str) -> str:
        if source_id in source_texts:
            return source_id
        if source_id in {"message:latest", "message:player"}:
            return next(
                (
                    allowed_id
                    for allowed_id in reversed(tuple(source_texts))
                    if allowed_id.startswith("message:")
                ),
                source_id,
            )
        placeholder_prefixes = {
            "scene_snapshot:snapshot-1": "scene_snapshot:",
            "character:ren": "character:",
        }
        prefix = placeholder_prefixes.get(source_id)
        if prefix is None:
            return source_id
        return next(
            (
                allowed_id
                for allowed_id in source_texts
                if allowed_id.startswith(prefix)
            ),
            source_id,
        )

    def rewrite(item: object) -> object:
        if isinstance(item, dict):
            rewritten: dict[str, object] = {}
            normalized_ids: list[str] | None = None
            for key, raw_value in item.items():
                if key == "evidence_source_ids" and isinstance(raw_value, list):
                    normalized_ids = [
                        normalize(str(source_id)) for source_id in raw_value
                    ]
                    rewritten[key] = normalized_ids
                else:
                    rewritten[key] = rewrite(raw_value)
            if (
                normalized_ids
                and "evidence_quote" not in rewritten
            ):
                quote = next(
                    (
                        source_texts[source_id]
                        for source_id in normalized_ids
                        if source_id in source_texts
                    ),
                    "",
                )
                if quote:
                    rewritten["evidence_quote"] = quote
            return rewritten
        if isinstance(item, list):
            return [rewrite(child) for child in item]
        return item

    rewritten = rewrite(value)
    assert isinstance(rewritten, dict)
    return rewritten


def _evidence_source_texts(request: StructuredOutputRequest) -> dict[str, str]:
    body = request.messages[-1].body
    sources: dict[str, str] = {}
    for line in body.splitlines():
        if not line.startswith("- ") or ": " not in line:
            continue
        source_id, source_text = line.removeprefix("- ").split(": ", 1)
        if ":" in source_id:
            sources[source_id] = source_text
    return sources
