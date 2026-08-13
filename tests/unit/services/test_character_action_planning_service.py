from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
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
    CHARACTER_ACTION_PLANNING_BATCH_MAX_CHARACTERS,
    CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
    CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
    CHARACTER_ACTION_PLANNING_TASK,
    CharacterActionPlanningService,
    _character_presence_batch_messages,
    _character_presence_messages,
    _planning_batch_evidence_sources,
    _planning_characters_for_turn,
    _planning_evidence_sources,
    _presence_assessments_from_batch_data,
    _source_message_text_for_character,
    _visible_recent_messages_for_character,
    character_action_planning_enabled,
    format_character_turn_assessment,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_character_action_planning_deterministic_presence_skips_model_calls(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_message_id, characters = _create_save_with_characters(
        repositories
    )
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I study the lantern mechanism.",
    )
    provider = CharacterDecisionProvider({})
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(
            save_id=save_id,
            player_message_id=player_message.id,
            intents_absorbed=True,
        )
    )

    assert provider.structured_output_requests == []
    assert [assessment.character_name for assessment in result.assessments] == [
        "Mara",
        "Ren",
    ]
    assert all(assessment.present for assessment in result.assessments)
    assert result.deterministic_presence_count == 2
    assert result.presence_calls_made == 0
    assert result.model_calls_avoided == 3
    assert result.applied_presence_update is False
    assert characters["mara"] in {
        assessment.character_id for assessment in result.assessments
    }


def test_character_action_planning_deterministic_without_model_preference(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_message_id, characters = _create_save_with_characters(
        repositories
    )
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I study the lantern mechanism.",
    )
    provider = CharacterDecisionProvider({})

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message.id)
    )

    assert result.skipped_reason == ""
    assert provider.structured_output_requests == []
    assert result.model_calls_avoided == 0
    assert [assessment.character_name for assessment in result.assessments] == [
        "Mara",
        "Ren",
    ]
    assert all(assessment.present for assessment in result.assessments)
    assert characters["mara"] in {
        assessment.character_id for assessment in result.assessments
    }


def test_character_action_planning_reports_avoided_calls_without_absorbed_intents(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_message_id, characters = _create_save_with_characters(
        repositories
    )
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I study the lantern mechanism.",
    )
    provider = CharacterDecisionProvider({})
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(
            save_id=save_id,
            player_message_id=player_message.id,
            intents_absorbed=False,
        )
    )

    assert result.intents_absorbed is False
    assert result.model_calls_avoided == 1
    assert result.deterministic_presence_count == 2
    assert characters["mara"] in {
        assessment.character_id for assessment in result.assessments
    }


def test_character_action_planning_deterministic_assessment_is_grounded(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_message_id, characters = _create_save_with_characters(
        repositories
    )
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I study the lantern mechanism.",
    )
    provider = CharacterDecisionProvider({})
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message.id)
    )

    ren = next(
        assessment
        for assessment in result.assessments
        if assessment.character_id == characters["ren"]
    )
    assert ren.present is True
    assert ren.enters_scene is False
    assert ren.leaves_scene is False
    assert ren.confidence == 1.0
    assert ren.presence_evidence_source_ids == ("scene_snapshot:snapshot-1",)
    assert ren.presence_evidence_quote == characters["ren"]
    assert "present: yes" in format_character_turn_assessment(ren)


def test_character_action_planning_mentions_ambiguous_present_character(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "enters_scene": False,
                "leaves_scene": False,
                "reason": "Mara is already beside the lantern in the scene.",
                "confidence": 0.92,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
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
            intents_absorbed=True,
        )
    )

    assert [request.schema_name for request in provider.structured_output_requests] == [
        "character_presence_assessment"
    ]
    assert "Assess each listed character's scene presence" in (
        provider.structured_output_requests[0].messages[0].body
    )
    presence_schema = provider.structured_output_requests[0].schema
    assert presence_schema["type"] == "object"
    item_schema = presence_schema["properties"]["assessments"]["items"]
    assert "character_id" in item_schema["properties"]
    for field in (
        "enters_scene",
        "leaves_scene",
    ):
        assert field in item_schema["properties"]
        assert field in item_schema["required"]
    assert set(item_schema["properties"]["character_id"]["enum"]) == {
        characters["mara"],
    }
    assert [assessment.character_name for assessment in result.assessments] == [
        "Ren",
        "Mara",
    ]
    mara = next(
        assessment
        for assessment in result.assessments
        if assessment.character_id == characters["mara"]
    )
    assert mara.present is True
    assert mara.presence_evidence_source_ids
    assert result.deterministic_presence_count == 1
    assert result.presence_calls_made == 1
    assert result.model_calls_avoided == 2
    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {
        characters["player"],
        characters["mara"],
        characters["ren"],
    }


def test_character_action_planning_batch_presence_prompt_lists_all_characters(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    player_message = repositories.get_message(
        save_id=save_id,
        message_id=player_message_id,
    )
    assert player_message is not None
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I ask Mara and Ren what they see in the lens.",
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "reason": "Mara is in the scene.",
                "confidence": 0.9,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Ren": {
                "present": True,
                "reason": "Ren is in the scene.",
                "confidence": 0.9,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
        }
    )
    _configure_planning(repositories)

    asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message.id)
    )

    presence_body = provider.structured_output_requests[0].messages[-1].body
    assert "Character: Mara" in presence_body
    assert "Character: Ren" in presence_body
    assert f"character_id: {characters['mara']}" in presence_body
    assert f"character_id: {characters['ren']}" in presence_body
    assert "Evidence sources:" in presence_body
    assert "scene_snapshot:snapshot-1" in presence_body


def test_character_action_planning_batch_omitted_character_is_failed(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I ask Mara and Ren what they see in the lens.",
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "reason": "Mara is in the scene.",
                "confidence": 0.88,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            }
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message.id)
    )

    assert result.failed_character_ids == (characters["ren"],)
    assert [assessment.character_name for assessment in result.assessments] == [
        "Mara",
    ]
    assert len(provider.structured_output_requests) == 1
    assert [
        request.schema_name for request in provider.structured_output_requests
    ] == ["character_presence_assessment"]


def test_character_action_planning_batch_failure_falls_back_to_per_character_calls(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I ask Mara and Ren what they see in the lens.",
    )
    provider = BatchFailThenPerCharacterProvider(
        {
            "Mara": {
                "present": True,
                "reason": "Mara is in the scene.",
                "confidence": 0.88,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Ren": {
                "present": False,
                "reason": "Ren is offscreen.",
                "confidence": 0.8,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
        },
        fail_batch=True,
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(
            save_id=save_id,
            player_message_id=player_message.id,
            intents_absorbed=True,
        )
    )

    assert provider.batch_attempts == 1
    presence_requests = [
        request
        for request in provider.structured_output_requests
        if request.schema_name == "character_presence_assessment"
        and not _is_batch_presence_request(request)
    ]
    assert len(presence_requests) == 2
    assert result.presence_calls_made == 3
    assert result.model_calls_avoided == 1
    assert len(result.assessments) == 2


def test_character_action_planning_falls_back_to_per_character_beyond_batch_cap(
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
    names = tuple(
        f"Crew {index}"
        for index in range(CHARACTER_ACTION_PLANNING_BATCH_MAX_CHARACTERS + 1)
    )
    characters = [
        repositories.add_character(save_id=save.id, name=name, met=True)
        for name in names
    ]
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="A large crew gathers in the gallery.",
        present_character_ids=[character.id for character in characters],
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body="I ask " + ", ".join(names) + " what happens next.",
    )
    provider = CharacterDecisionProvider(
        {
            name: {
                "present": True,
                "reason": "The crew waits quietly.",
                "confidence": 0.5,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            }
            for name in names
        }
    )
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=player_message.id)
    )

    presence_requests = [
        request
        for request in provider.structured_output_requests
        if request.schema_name == "character_presence_assessment"
    ]
    assert len(presence_requests) == len(characters)
    assert all(
        not _is_batch_presence_request(request) for request in presence_requests
    )
    assert len(result.assessments) == len(characters)


def test_character_action_planning_skips_presence_update_without_grounded_quote(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "enters_scene": False,
                "leaves_scene": True,
                "reason": "Mara is claimed leaving without grounded evidence.",
                "confidence": 0.8,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
                "evidence_quote": "ruby library",
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
    mara_decision = next(
        decision
        for decision in result.decisions
        if decision.character_id == characters["mara"]
    )
    assert mara_decision.evidence_source_ids == ()
    assert mara_decision.evidence_quote == ""
    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {
        characters["player"],
        characters["mara"],
        characters["ren"],
    }


def test_character_action_planning_drops_entering_without_grounded_quote(
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
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": False,
                "enters_scene": True,
                "leaves_scene": False,
                "reason": "Mara is claimed to enter without support.",
                "confidence": 0.86,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
                "evidence_quote": "ruby library",
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

    assert [request.schema_name for request in provider.structured_output_requests] == [
        "character_presence_assessment"
    ]
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
        body="I study the lantern mechanism.",
    )
    provider = CharacterDecisionProvider({})
    _configure_planning(repositories)

    result = asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save.id, player_message_id=player_message.id)
    )

    assert provider.structured_output_requests == []
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
            "Archivist Ren": {
                "present": False,
                "enters_scene": True,
                "leaves_scene": False,
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
        if not _is_batch_presence_request(request)
    ]
    assert requested_names == []
    assert [assessment.character_name for assessment in result.assessments] == [
        "Mara",
        "Archivist Ren",
    ]
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert ren.id in snapshot.present_character_ids
    assert result.deterministic_presence_count == 1
    assert result.presence_calls_made == 1


def test_character_action_planning_can_return_tentative_presence_without_mutating_scene(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I ask Mara and Ren what they see in the lens.",
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "reason": "Mara is in the scene.",
                "confidence": 0.91,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Ren": {
                "present": False,
                "leaves_scene": True,
                "reason": "Ren decides to leave during the planned beat.",
                "confidence": 0.86,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
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
            player_message_id=player_message.id,
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
    assert characters["player"] not in {
        assessment.character_id for assessment in result.assessments
    }


def test_character_action_planning_projects_mixed_knowledge_per_character(
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
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden.id,
        character_id=ren.id,
        visibility="not_visible",
        confidence=1.0,
        source="scene_presence",
        evidence="Ren was not present.",
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
        body="I ask Mara and Ren what everyone does next.",
    )
    persisted_secret = repositories.add_memory(
        save_id=save.id,
        body="The archive password is ember dawn.",
        tags=["secret"],
        epistemic_status="belief",
        epistemic_actor_id=mara.id,
        epistemic_actor_name=mara.name,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=mara.id,
        target_type="memory",
        target_id=persisted_secret.id,
        knowledge_state="knows",
        acquisition_method="told",
        confidence=1.0,
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=source_message.id,
        character_id=ren.id,
        visibility="not_visible",
        confidence=1.0,
        source="private_address",
        evidence="The player addressed Mara privately.",
    )
    provider = EchoHiddenPromptDecisionProvider(
        {
            "Mara": {
                "present": True,
                "reason": "Mara is in the scene.",
                "confidence": 0.8,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Ren": {
                "present": True,
                "reason": "Ren is in the scene.",
                "confidence": 0.8,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
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

    messages = tuple(repositories.list_messages(save.id))
    mara_projection = _visible_recent_messages_for_character(
        repositories=repositories,
        save_id=save.id,
        character=mara,
        source_message=source_message,
        messages=messages,
    )
    request_text = "\n\n".join(
        message.body
        for request in provider.structured_output_requests
        for message in request.messages
    )
    assert hidden.body not in request_text
    assert len(result.assessments) == 2
    ren_projection = _visible_recent_messages_for_character(
        repositories=repositories,
        save_id=save.id,
        character=ren,
        source_message=source_message,
        messages=messages,
    )
    assert hidden in mara_projection
    assert hidden not in ren_projection
    assert _source_message_text_for_character(
        repositories=repositories,
        save_id=save.id,
        character=mara,
        source_message=source_message,
    ) == source_message.body
    assert _source_message_text_for_character(
        repositories=repositories,
        save_id=save.id,
        character=ren,
        source_message=source_message,
    ) == "[Latest source is not visible to this character.]"
    mara_evidence = _planning_evidence_sources(
        repositories=repositories,
        save_id=save.id,
        character=mara,
        source_message=source_message,
        messages=messages,
    )
    ren_evidence = _planning_evidence_sources(
        repositories=repositories,
        save_id=save.id,
        character=ren,
        source_message=source_message,
        messages=messages,
    )
    assert "archive password is ember dawn" in mara_evidence[
        f"memory:{persisted_secret.id}"
    ]
    assert f"memory:{persisted_secret.id}" not in ren_evidence
    batch_prompts = [
        "\n".join(message.body for message in request.messages)
        for request in provider.structured_output_requests
        if request.schema_name == "character_presence_assessment"
        and "Assess each listed character" in request.messages[0].body
    ]
    assert all(persisted_secret.body not in prompt for prompt in batch_prompts)
    assert result.failed_character_ids == ()


def test_character_action_planning_includes_active_threads(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, _characters = _create_save_with_characters(
        repositories
    )
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I ask Mara and Ren what they see in the lens.",
    )
    repositories.add_active_thread(
        save_id=save_id,
        title="Guard search",
        description="Guards are sweeping toward the beacon room.",
        status="active",
        priority=3,
        visibility="scene",
        related_entities=["director_pressure"],
        source_message_id=player_message.id,
    )
    repositories.add_active_thread(
        save_id=save_id,
        title="Secret accord",
        description="Ren privately promised Mara a hidden favor.",
        status="active",
        priority=4,
        visibility="private",
        related_entities=["character:ren"],
        source_message_id=player_message.id,
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "reason": "Mara is in the scene.",
                "confidence": 0.9,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Ren": {
                "present": True,
                "reason": "Ren is in the scene.",
                "confidence": 0.84,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
        }
    )
    _configure_planning(repositories)

    asyncio.run(
        CharacterActionPlanningService(
            repositories=repositories,
            providers={"fake": provider},
        ).plan_for_turn(save_id=save_id, player_message_id=player_message.id)
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
    save_id, _player_message_id, characters = _create_save_with_characters(
        repositories
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Mara guards the lantern while Ren approaches from the stairs.",
        present_character_ids=[characters["mara"]],
        snapshot_id="snapshot-1",
    )
    mara = repositories.get_character(characters["mara"])
    assert mara is not None
    repositories.update_character(replace(mara, locked_fields=["present"]))
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
                "reason": "The player asked Ren to take over.",
                "confidence": 0.83,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Ren": {
                "present": False,
                "enters_scene": True,
                "leaves_scene": False,
                "reason": "The player called Ren into the room.",
                "confidence": 0.86,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
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

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert characters["mara"] in snapshot.present_character_ids
    assert characters["ren"] in snapshot.present_character_ids
    assert result.assessments[0].leaves_scene is True
    assert result.assessments[1].enters_scene is True
    assert result.applied_presence_update is True


def test_character_action_planning_batch_exposes_allowed_evidence_source_ids(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, _characters = _create_save_with_characters(
        repositories
    )
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "reason": "Mara is in the scene.",
                "confidence": 0.9,
                "evidence_source_ids": [f"message:{player_message_id}"],
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

    presence_request = provider.structured_output_requests[0]
    evidence_schema = presence_request.schema["properties"]["assessments"]["items"][
        "properties"
    ]["evidence_source_ids"]
    assert f"message:{player_message_id}" in evidence_schema["items"]["enum"]
    assert f"message:{player_message_id}" in presence_request.messages[-1].body


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
    names = ("Mara", "Ren", "Talla", "Ivo", "Senn", "Theo", "Vega")
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
        body="I ask " + ", ".join(names) + " what they do next.",
    )
    provider = BlockingCharacterDecisionProvider(
        {
            name: {
                "present": True,
                "reason": "The player asked the crew.",
                "confidence": 0.75,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
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
        provider.release.set()
        result = await task
        assert len(provider.structured_output_requests) == len(names)
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
        body="I tell Mika Arai I would like to see her again.",
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
                "reason": "The route is still early.",
                "confidence": 0.83,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
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
                "reason": "Mara stepped out before this beat.",
                "confidence": 0.7,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            }
        },
        fail_names={"Mara"},
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
    assert characters["mara"] in snapshot.present_character_ids
    assert result.failed_character_ids == (characters["mara"],)
    assert [assessment.character_name for assessment in result.assessments] == [
        "Ren"
    ]


def test_character_action_planning_skips_missing_catalog_row(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, _characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "reason": "Mara is present in the scene.",
                "confidence": 0.7,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
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

    deterministic_present, ambiguous = _planning_characters_for_turn(
        repositories=repositories,
        save_id=save.id,
        source_message=direction,
    )
    assert deterministic_present == ()
    assert {character.id for character in ambiguous} == {rival.id, witness.id}
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


def test_storyteller_present_character_directed_to_leave_is_ambiguous(
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
    repositories.add_character(save_id=save.id, name="The Witness")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The rival stands by the stage doors.",
        present_character_ids=[rival.id],
        snapshot_id="snapshot-1",
    )
    direction = repositories.append_message(
        save_id=save.id,
        role="player",
        body="Have the rival leave the ceremony.",
    )
    provider = CharacterDecisionProvider(
        {
            "The Rival": {
                "present": True,
                "enters_scene": False,
                "leaves_scene": True,
                "reason": "The direction sends the rival out.",
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
        ).plan_for_turn(save_id=save.id, player_message_id=direction.id)
    )

    assert len(provider.structured_output_requests) == 1
    assert result.deterministic_presence_count == 0
    rival_assessment = next(
        assessment
        for assessment in result.assessments
        if assessment.character_id == rival.id
    )
    assert rival_assessment.leaves_scene is True
    assert rival_assessment.present is True


def test_storyteller_with_player_scopes_mentions_like_roleplay(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="The Ceremony",
        premise="A rival waits in the wings.",
        player_role="Director",
        content={"player_character_name": "Ily"},
        interaction_mode=InteractionMode.STORYTELLER,
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Act One")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    witness = repositories.add_character(save_id=save.id, name="The Witness")
    rival = repositories.add_character(save_id=save.id, name="The Rival")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The witness waits by the stage doors.",
        present_character_ids=[witness.id],
    )
    direction = repositories.append_message(
        save_id=save.id,
        role="player",
        body="Have the rival enter from the wings.",
    )

    deterministic_present, ambiguous = _planning_characters_for_turn(
        repositories=repositories,
        save_id=save.id,
        source_message=direction,
    )

    assert {character.id for character in deterministic_present} == {witness.id}
    assert {character.id for character in ambiguous} == {rival.id}


def test_character_action_planning_batch_prompt_excludes_direction_and_player_text(
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
    evidence_sources = _planning_batch_evidence_sources(
        repositories=repositories,
        save_id=save.id,
        characters=(rival, witness),
        source_message=direction,
        messages=(direction, narrator),
    )
    prompts = _character_presence_batch_messages(
        repositories=repositories,
        save_id=save.id,
        characters=(rival, witness),
        source_message=direction,
        messages=(direction, narrator),
        evidence_sources=evidence_sources,
    )
    prompt = "\n".join(message.body for message in prompts)
    assert "Character: The Rival" in prompt
    assert "Character: The Witness" in prompt
    assert "Player character:" not in prompt
    assert "Dating route pacing" not in prompt
    assert f"message:{direction.id}" not in evidence_sources
    assert f"message:{narrator.id}" in evidence_sources
    assert "non-diegetic story direction" in prompt
    assert "not canonical evidence" in prompt


def test_presence_assessments_from_batch_data_skips_invalid_items(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, characters = _create_save_with_characters(repositories)
    source_message = repositories.get_message(
        save_id=save_id,
        message_id=player_message_id,
    )
    assert source_message is not None
    character_records = {
        character.id: character
        for character in repositories.list_characters(save_id)
    }
    evidence_sources = _planning_batch_evidence_sources(
        repositories=repositories,
        save_id=save_id,
        characters=(
            character_records[characters["mara"]],
            character_records[characters["ren"]],
        ),
        source_message=source_message,
        messages=tuple(repositories.list_messages(save_id)),
    )
    mara_id = characters["mara"]
    ren_id = characters["ren"]
    all_characters = {
        character.id: character
        for character in repositories.list_characters(save_id)
    }
    mara_name = all_characters[mara_id].name
    valid_item = {
        "character_id": mara_id,
        "present": True,
        "enters_scene": False,
        "leaves_scene": False,
        "reason": "Mara is in the scene.",
        "confidence": 0.9,
        "evidence_source_ids": ["scene_snapshot:snapshot-1"],
        "evidence_quote": "Mara steadies the storm lantern",
    }

    assessments = _presence_assessments_from_batch_data(
        {
            "assessments": [
                valid_item,
                {"character_id": ren_id, "present": "not-a-bool"},
                {"character_id": "unknown-character-id", "present": True},
                "not-a-dict",
                valid_item,
            ]
        },
        characters_by_id=all_characters,
        evidence_sources=evidence_sources,
    )

    assert set(assessments) == {mara_id}
    assert assessments[mara_id].character_name == mara_name


def test_character_action_planning_can_be_disabled_per_save(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, _characters = _create_save_with_characters(repositories)
    provider = CharacterDecisionProvider(
        {
            "Mara": {
                "present": True,
                "reason": "",
                "confidence": 0.7,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
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
        if _is_batch_presence_request(request):
            return _batch_presence_response(
                self,
                request,
                decisions_by_name=self.decisions_by_name,
                fail_names=self.fail_names,
            )
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


class BatchFailThenPerCharacterProvider(CharacterDecisionProvider):
    def __init__(
        self,
        decisions_by_name: dict[str, dict[str, object]],
        *,
        fail_batch: bool,
    ) -> None:
        super().__init__(decisions_by_name)
        self.fail_batch = fail_batch
        self.batch_attempts = 0

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if _is_batch_presence_request(request):
            self.batch_attempts += 1
            if self.fail_batch:
                raise ValueError("batch presence provider failed")
        return await super().generate_structured_output(request)


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
        if _is_batch_presence_request(request):
            return _batch_presence_response(
                self,
                request,
                decisions_by_name=self.decisions_by_name,
                fail_names=set(),
            )
        body = request.messages[-1].body
        name = _requested_character_name(body)
        data = dict(self.decisions_by_name[name])
        data = _with_allowed_evidence_ids_and_quote(data, request)
        data["character_id"] = request.schema["properties"]["character_id"]["enum"][0]
        data["reason"] = (
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
        if _is_batch_presence_request(request):
            return _batch_presence_response(
                self,
                request,
                decisions_by_name=self.decisions_by_name,
                fail_names=set(),
            )
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


def _is_batch_presence_request(request: StructuredOutputRequest) -> bool:
    return "assessments" in request.schema.get("properties", {})


def _batch_presence_response(
    provider: CharacterDecisionProvider,
    request: StructuredOutputRequest,
    *,
    decisions_by_name: dict[str, dict[str, object]],
    fail_names: set[str],
) -> StructuredOutputResponse:
    body = request.messages[-1].body
    character_ids_by_name = _requested_batch_character_ids(body)
    items: list[dict[str, object]] = []
    for name, character_id in character_ids_by_name.items():
        if name in fail_names or name not in decisions_by_name:
            continue
        data = dict(decisions_by_name[name])
        data = _with_allowed_evidence_ids_and_quote(data, request)
        data["character_id"] = character_id
        items.append(data)
    return StructuredOutputResponse(
        data={"assessments": items},
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


def _requested_batch_character_ids(body: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    current_name: str | None = None
    for line in body.splitlines():
        if line.startswith("Character: "):
            current_name = line.removeprefix("Character: ").strip()
        elif line.startswith("character_id: ") and current_name is not None:
            pairs[current_name] = line.removeprefix("character_id: ").strip()
            current_name = None
    return pairs


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
            if normalized_ids and "evidence_quote" not in rewritten:
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
