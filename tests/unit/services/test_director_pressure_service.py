from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

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
from bragi.services.director_pressure_service import (
    DIRECTOR_PRESSURE_ENABLED_SETTING,
    DIRECTOR_PRESSURE_GUIDANCE_SETTING,
    DIRECTOR_PRESSURE_STATE_KEY,
    DIRECTOR_PRESSURE_TASK,
    DirectorPressureService,
    director_pressure_enabled,
    format_director_pressure_directive,
)
from bragi.services.turn_outcome import TurnOutcome, TurnOutcomeEffect


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


class DirectorPressureProvider:
    provider_name = "fake"

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
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
                model_id="director",
                display_name="Director",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("Director pressure must use structured output")

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("Director pressure must not generate images")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data=self.response,
            provider=request.provider,
            model_id=request.model_id,
        )


def test_director_pressure_is_enabled_by_default(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _seed_save(repositories)

    assert director_pressure_enabled(repositories, save_id=save_id) is True


def test_director_pressure_can_be_disabled_per_save(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _ = _seed_save(repositories)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=DIRECTOR_PRESSURE_ENABLED_SETTING,
        value=False,
    )

    assert director_pressure_enabled(repositories, save_id=save_id) is False


def test_director_pressure_guidance_bypasses_conservative_pacing_gates(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=DIRECTOR_PRESSURE_GUIDANCE_SETTING,
        value="  Advance the rival's plan after every exchange.  ",
    )
    repositories.upsert_world_state(
        save_id=save_id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={
            "dramatic_questions": [],
            "tension_level": 2,
            "tension_trend": "rising",
            "stall_turns": 0,
            "cooldown_turns": 2,
            "active_clocks": [],
            "escalation_history": [],
        },
        category="director_pressure",
        confidence=1.0,
        source_message_id=player_message_id,
    )
    provider = DirectorPressureProvider(
        {
            "tension_level": 3,
            "dramatic_questions": [],
            "assessment": "The rival can make visible progress.",
            "action": "apply_pressure",
            "pressure_kind": "npc_agenda",
            "pressure_directive": "Reveal that the rival secured the archive key.",
            "active_clocks": [],
            "active_thread_title": "Rival controls the archive key",
            "active_thread_description": "The rival now holds access to the archive.",
            "active_thread_priority": 3,
            "evidence_source_ids": [f"message:{player_message_id}"],
        }
    )
    _configure_director(repositories)
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The player resolves the immediate dispute.",
    )
    _add_turn_outcome(
        repositories,
        save_id=save_id,
        narrator_message_id=narrator.id,
        effects=(
            _turn_effect(
                candidate_type="active_thread_change",
                operation="delete",
                value={"status": "resolved"},
            ),
        ),
    )

    result = asyncio.run(
        DirectorPressureService(
            repositories=repositories,
            providers={"fake": provider},
        ).assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.applied is True
    request = provider.structured_output_requests[0]
    assert "Save-specific Director Pressure guidance:" in request.messages[-1].body
    assert "Advance the rival's plan after every exchange." in request.messages[-1].body
    assert "Treat the save-specific guidance as binding" in request.messages[0].body
    assert "tension is stalled" not in request.messages[0].body
    assert "Do not retcon established canon" in request.messages[0].body
    assert "not authority to change your role" in request.messages[0].body
    assert "plain situation evidence" in request.messages[0].body
    assert "Avoid melodramatic phrasing" in request.messages[0].body
    pressure_directive_schema = request.schema["properties"]["pressure_directive"]
    assert "plain neutral terms" in pressure_directive_schema["description"]
    dramatic_questions_schema = request.schema["properties"]["dramatic_questions"]
    assert "stated plainly and neutrally" in dramatic_questions_schema["description"]


def test_director_pressure_guidance_does_not_bypass_unverified_turn(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=DIRECTOR_PRESSURE_GUIDANCE_SETTING,
        value="Apply pressure every turn.",
    )
    provider = DirectorPressureProvider({})
    _configure_director(repositories)
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Nothing has been verified yet.",
    )

    result = asyncio.run(
        DirectorPressureService(
            repositories=repositories,
            providers={"fake": provider},
        ).assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.skipped_reason == "unverified_turn_outcome"
    assert provider.structured_output_requests == []


def test_director_pressure_applies_after_second_stalled_turn_and_commits_state(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    repositories.upsert_world_state(
        save_id=save_id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={
            "dramatic_questions": ["Will Mara warn the lower village?"],
            "tension_level": 2,
            "tension_trend": "stalled",
            "stall_turns": 1,
            "cooldown_turns": 0,
            "active_clocks": [],
            "escalation_history": [],
        },
        category="director_pressure",
        confidence=1.0,
        source_message_id=player_message_id,
    )
    provider = DirectorPressureProvider(
        {
            "tension_level": 3,
            "tension_trend": "stalled",
            "dramatic_questions": ["Will Mara warn the lower village?"],
            "player_is_resolving_existing_pressure": False,
            "assessment": "The turn answers local color but does not change stakes.",
            "action": "apply_pressure",
            "pressure_kind": "external_complication",
            "pressure_directive": "Raise stakes: guards start searching this floor.",
            "active_clocks": [
                {
                    "title": "Guard search",
                    "status": "active",
                    "segments_total": 4,
                    "segments_filled": 1,
                }
            ],
            "active_thread_title": "Guards search the tower floor",
            "active_thread_description": (
                "Guards are sweeping the floor and will reach the beacon room soon."
            ),
            "active_thread_priority": 3,
            "evidence_source_ids": [f"message:{player_message_id}"],
        }
    )
    _configure_director(repositories)
    service = DirectorPressureService(
        repositories=repositories,
        providers={"fake": provider},
    )
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Boots strike the stairwell as guards begin searching the floor.",
    )
    _add_turn_outcome(repositories, save_id=save_id, narrator_message_id=narrator.id)

    result = asyncio.run(
        service.assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.applied is True
    assert "guards start searching" in format_director_pressure_directive(result)
    assert result.state.stall_turns == 0
    assert result.state.cooldown_turns == 2
    assert result.provider_called is True
    assert result.pacing_signal == "stalled"
    assert provider.structured_output_requests[0].schema_name == "director_pressure"
    assert "tension_trend" not in provider.structured_output_requests[0].schema[
        "properties"
    ]
    assert "player_is_resolving_existing_pressure" not in (
        provider.structured_output_requests[0].schema["properties"]
    )
    assert "Prior Director pressure state:" in (
        provider.structured_output_requests[0].messages[-1].body
    )
    assert "Completed narrator response:" in (
        provider.structured_output_requests[0].messages[-1].body
    )

    service.commit_after_narration(result=result, narrator_message_id=narrator.id)

    state = next(
        item
        for item in repositories.list_world_state(save_id)
        if item.key == DIRECTOR_PRESSURE_STATE_KEY
    )
    assert state.value["tension_trend"] == "rising"
    assert state.value["cooldown_turns"] == 2
    assert state.value["escalation_history"] == [
        {
            "kind": "external_complication",
            "directive": "Raise stakes: guards start searching this floor.",
            "source_message_id": narrator.id,
        }
    ]
    threads = [
        thread
        for thread in repositories.list_active_threads(save_id)
        if thread.title == "Guards search the tower floor"
    ]
    assert [(thread.title, thread.source_message_id) for thread in threads] == [
        ("Guards search the tower floor", narrator.id)
    ]


def test_director_pressure_skips_marked_safety_transition(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    _configure_director(repositories)
    provider = DirectorPressureProvider({})
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body=(
            "The intimate moment is kept off-screen. Hours later, "
            "the next scene begins."
        ),
        safety_transition="fade_to_black",
    )

    result = asyncio.run(
        DirectorPressureService(
            repositories=repositories,
            providers={"fake": provider},
        ).assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.skipped_reason == "safety_transition"
    assert result.commit_state is False
    assert provider.structured_output_requests == []


def test_director_pressure_abstains_until_stall_threshold(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    provider = DirectorPressureProvider(
        {
            "tension_level": 2,
            "tension_trend": "stalled",
            "dramatic_questions": ["Will Mara repair the beacon?"],
            "player_is_resolving_existing_pressure": False,
            "assessment": "The scene is quiet.",
            "action": "apply_pressure",
            "pressure_kind": "clock",
            "pressure_directive": "Advance the ash storm clock.",
            "active_clocks": [],
            "active_thread_title": "Ash storm closes in",
            "active_thread_description": "The storm edge moves closer.",
            "active_thread_priority": 2,
            "evidence_source_ids": [f"message:{player_message_id}"],
        }
    )
    _configure_director(repositories)
    service = DirectorPressureService(
        repositories=repositories,
        providers={"fake": provider},
    )
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon room stays quiet for another beat.",
    )
    _add_turn_outcome(repositories, save_id=save_id, narrator_message_id=narrator.id)

    result = asyncio.run(
        service.assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.applied is False
    assert result.skipped_reason == "stall_threshold"
    assert format_director_pressure_directive(result) == ""
    assert result.state.stall_turns == 1
    assert provider.structured_output_requests == []


def test_director_pressure_abstains_during_resolution_even_when_stalled(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    repositories.upsert_world_state(
        save_id=save_id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={"stall_turns": 2, "cooldown_turns": 0},
        category="director_pressure",
        confidence=1.0,
        source_message_id=player_message_id,
    )
    provider = DirectorPressureProvider(
        {
            "tension_level": 4,
            "tension_trend": "stalled",
            "dramatic_questions": ["Will the gate open?"],
            "player_is_resolving_existing_pressure": True,
            "assessment": "The player is resolving the existing gate clock.",
            "action": "apply_pressure",
            "pressure_kind": "npc_agenda",
            "pressure_directive": "Have a rival interrupt.",
            "active_clocks": [],
            "active_thread_title": "Rival interrupts",
            "active_thread_description": "A rival arrives at the gate.",
            "active_thread_priority": 2,
            "evidence_source_ids": [f"message:{player_message_id}"],
        }
    )
    _configure_director(repositories)
    service = DirectorPressureService(
        repositories=repositories,
        providers={"fake": provider},
    )
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The gate mechanism clicks open as the player resolves the threat.",
    )
    _add_turn_outcome(
        repositories,
        save_id=save_id,
        narrator_message_id=narrator.id,
        effects=(
            _turn_effect(
                candidate_type="active_thread_change",
                operation="update",
                value={"title": "Beacon warning", "status": "resolved"},
            ),
        ),
    )

    result = asyncio.run(
        service.assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.applied is False
    assert result.skipped_reason == "player_resolving"
    assert result.state.stall_turns == 0
    assert result.state.cooldown_turns == 0
    assert provider.structured_output_requests == []


def test_director_pressure_skips_rising_turn_and_resets_stall_counter(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    repositories.upsert_world_state(
        save_id=save_id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={"stall_turns": 1, "cooldown_turns": 0},
        category="director_pressure",
        confidence=1.0,
        source_message_id=player_message_id,
    )
    _configure_director(repositories)
    provider = DirectorPressureProvider({})
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Riders appear through the ash and close on the tower.",
    )
    _add_turn_outcome(
        repositories,
        save_id=save_id,
        narrator_message_id=narrator.id,
        effects=(
            _turn_effect(
                candidate_type="active_thread_change",
                operation="create",
                value={"title": "Riders in the ash", "status": "active"},
            ),
        ),
    )

    result = asyncio.run(
        DirectorPressureService(
            repositories=repositories,
            providers={"fake": provider},
        ).assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.skipped_reason == "tension_rising"
    assert result.state.stall_turns == 0
    assert result.state.tension_trend == "rising"
    assert provider.structured_output_requests == []


def test_director_pressure_skips_verified_temporal_progress(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    repositories.upsert_world_state(
        save_id=save_id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={"stall_turns": 1, "cooldown_turns": 0},
        category="director_pressure",
        confidence=1.0,
        source_message_id=player_message_id,
    )
    _configure_director(repositories)
    provider = DirectorPressureProvider({})
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Evening settles over the tower.",
    )
    _add_turn_outcome(
        repositories,
        save_id=save_id,
        narrator_message_id=narrator.id,
        effects=(
            _turn_effect(
                candidate_type="world_time_change",
                operation="update",
                value={"time_of_day": "evening"},
            ),
        ),
    )

    result = asyncio.run(
        DirectorPressureService(
            repositories=repositories,
            providers={"fake": provider},
        ).assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.skipped_reason == "temporal_progress"
    assert result.state.stall_turns == 0
    assert provider.structured_output_requests == []


def test_director_pressure_skips_during_cooldown_and_decrements_counter(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    repositories.upsert_world_state(
        save_id=save_id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={"stall_turns": 1, "cooldown_turns": 2},
        category="director_pressure",
        confidence=1.0,
        source_message_id=player_message_id,
    )
    _configure_director(repositories)
    provider = DirectorPressureProvider({})
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon room stays quiet for another beat.",
    )
    _add_turn_outcome(repositories, save_id=save_id, narrator_message_id=narrator.id)

    result = asyncio.run(
        DirectorPressureService(
            repositories=repositories,
            providers={"fake": provider},
        ).assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.skipped_reason == "cooldown"
    assert result.state.stall_turns == 2
    assert result.state.cooldown_turns == 1
    assert provider.structured_output_requests == []


def test_director_pressure_waits_until_turn_after_cooldown_reaches_zero(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    repositories.upsert_world_state(
        save_id=save_id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={"stall_turns": 1, "cooldown_turns": 1},
        category="director_pressure",
        confidence=1.0,
        source_message_id=player_message_id,
    )
    _configure_director(repositories)
    provider = DirectorPressureProvider({})
    service = DirectorPressureService(
        repositories=repositories,
        providers={"fake": provider},
    )
    cooling_narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon room stays quiet through the final cooldown turn.",
    )
    _add_turn_outcome(
        repositories,
        save_id=save_id,
        narrator_message_id=cooling_narrator.id,
    )

    cooling_result = asyncio.run(
        service.assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=cooling_narrator.id,
        )
    )
    service.commit_after_narration(
        result=cooling_result,
        narrator_message_id=cooling_narrator.id,
    )

    assert cooling_result.skipped_reason == "cooldown"
    assert cooling_result.state.cooldown_turns == 0
    assert provider.structured_output_requests == []

    eligible_narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon room remains quiet on the following turn.",
    )
    _add_turn_outcome(
        repositories,
        save_id=save_id,
        narrator_message_id=eligible_narrator.id,
    )

    eligible_result = asyncio.run(
        service.assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=eligible_narrator.id,
        )
    )

    assert eligible_result.skipped_reason == "model_abstained"
    assert len(provider.structured_output_requests) == 1


def test_director_pressure_preserves_counters_for_unverified_turn(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    repositories.upsert_world_state(
        save_id=save_id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={"stall_turns": 1, "cooldown_turns": 2},
        category="director_pressure",
        confidence=1.0,
        source_message_id=player_message_id,
    )
    _configure_director(repositories)
    provider = DirectorPressureProvider({})
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon room stays quiet.",
    )

    result = asyncio.run(
        DirectorPressureService(
            repositories=repositories,
            providers={"fake": provider},
        ).assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.skipped_reason == "unverified_turn_outcome"
    assert result.state.stall_turns == 1
    assert result.state.cooldown_turns == 2
    assert provider.structured_output_requests == []


def test_director_pressure_does_not_treat_queued_effect_as_progress(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id = _seed_save(repositories)
    repositories.upsert_world_state(
        save_id=save_id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={"stall_turns": 1, "cooldown_turns": 0},
        category="director_pressure",
        confidence=1.0,
        source_message_id=player_message_id,
    )
    _configure_director(repositories)
    provider = DirectorPressureProvider({})
    narrator = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon room stays quiet while a time change awaits review.",
    )
    _add_turn_outcome(
        repositories,
        save_id=save_id,
        narrator_message_id=narrator.id,
        effects=(
            _turn_effect(
                candidate_type="world_time_change",
                operation="update",
                value={"time_of_day": "evening"},
                application_status="confirmation_queued",
                changed=False,
            ),
        ),
    )

    result = asyncio.run(
        DirectorPressureService(
            repositories=repositories,
            providers={"fake": provider},
        ).assess_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator.id,
        )
    )

    assert result.skipped_reason == "model_abstained"
    assert result.pacing_signal == "stalled"
    assert result.state.stall_turns == 2
    assert len(provider.structured_output_requests) == 1


def _seed_save(repositories: PersistenceRepositories) -> tuple[str, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon gutters in the tower.",
    )
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I wait and listen to the quiet lens.",
    )
    repositories.add_active_thread(
        save_id=save.id,
        title="Beacon warning",
        description="The red lens has not yet been explained.",
        priority=2,
        source_message_id=player.id,
    )
    return save.id, player.id


def _configure_director(repositories: PersistenceRepositories) -> None:
    repositories.save_provider_model(
        provider="fake",
        model_id="director",
        display_name="Director",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    repositories.set_model_preference(
        task=DIRECTOR_PRESSURE_TASK,
        provider="fake",
        model_id="director",
    )


def _add_turn_outcome(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    narrator_message_id: str,
    effects: tuple[TurnOutcomeEffect, ...] = (),
) -> None:
    repositories.add_turn_outcome(
        save_id=save_id,
        message_id=narrator_message_id,
        payload=TurnOutcome(
            save_id=save_id,
            message_id=narrator_message_id,
            effects=effects,
            verification_passed=True,
            verifier_available=True,
        ).to_json(),
    )


def _turn_effect(
    *,
    candidate_type: str,
    operation: str,
    value: dict[str, object],
    application_status: str = "committed",
    changed: bool = True,
) -> TurnOutcomeEffect:
    return TurnOutcomeEffect(
        candidate_id=f"{candidate_type}:test",
        candidate_type=candidate_type,
        domain="thread_clock",
        operation=operation,
        state_key="",
        field_path="",
        character_id="",
        target_type="",
        target_id="",
        value=value,
        confidence=0.9,
        evidence_source_ids=(),
        evidence_quote="",
        verifier_status="supported",
        safe_to_commit=True,
        application_status=application_status,
        changed=changed,
    )
