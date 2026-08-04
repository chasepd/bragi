from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import MessageRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.message_correction import MessageCorrectionContext
from bragi.services.prompt_inspection import PromptInspectionStore
from bragi.services.state_service import (
    ExtractedMemory,
    ExtractedStateChange,
    StateExtraction,
    StateExtractionRequest,
    StateService,
    StructuredProviderStateExtractor,
    ToolCallingProviderStateExtractor,
    _state_extraction_from_structured_data,
    _state_extraction_instruction,
    _state_extraction_schema,
)


class StructuredFakeExtractor:
    def __init__(self, extraction: StateExtraction) -> None:
        self.extraction = extraction
        self.requests: list[Any] = []

    async def extract(self, request: Any) -> StateExtraction:
        self.requests.append(request)
        return self.extraction


def test_retired_character_interaction_type_has_no_state_specialization() -> None:
    instruction = _state_extraction_instruction("character_interaction")

    lowered = instruction.casefold()
    assert "featured character" not in lowered
    assert "relationship.player_to_<name>" not in lowered
    assert "broad world-state" not in lowered


class FailingFakeExtractor:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.requests: list[Any] = []

    async def extract(self, request: Any) -> StateExtraction:
        self.requests.append(request)
        raise self.error


class SequenceToolCallProvider:
    provider_name = "fake"

    def __init__(
        self,
        responses: list[tuple[ProviderToolCall, ...]],
    ) -> None:
        self.responses = responses
        self.tool_call_requests: list[ToolCallRequest] = []

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected tool-call request")
        return ToolCallResponse(
            tool_calls=(
                self.responses[0]
                if len(self.responses) == 1
                else self.responses.pop(0)
            ),
            body="",
            provider=request.provider,
            model_id=request.model_id,
        )


class SequenceStructuredOutputProvider:
    provider_name = "fake"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected structured-output request")
        return StructuredOutputResponse(
            data=self.responses.pop(0),
            provider=request.provider,
            model_id=request.model_id,
        )


class ShapeSwitchToolCallProvider(SequenceToolCallProvider):
    """Tool-capable state provider whose tool calls 404 but structured works."""

    def __init__(
        self,
        *,
        structured_data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(responses=[])
        self.structured_data = structured_data or {
            "state_changes": [],
            "memories": [],
            "conflicts": [],
        }
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model not found",
            status_code=404,
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data=self.structured_data,
            provider=request.provider,
            model_id=request.model_id,
        )


class ShapeFailingToolCallProvider(ShapeSwitchToolCallProvider):
    """Tool-capable state provider whose tool and structured calls both 404."""

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model not found",
            status_code=404,
        )


class RateLimitedShapeSwitchToolCallProvider(ShapeSwitchToolCallProvider):
    """Tool-capable state provider whose tool calls rate-limit but structured works."""

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.RATE_LIMITED,
            "rate limited",
            status_code=429,
        )


class FailingToolCallFallbackProvider(SequenceToolCallProvider):
    provider_name = "fallback"

    def __init__(self, *, error: ProviderError) -> None:
        super().__init__(responses=[])
        self.error = error

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        raise self.error


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_extract_and_apply_turn_upserts_world_state_adds_memory_and_records_changes(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    existing_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Lower stair", "danger": "low"},
        category="scene",
        confidence=0.4,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="scene.location",
                    value={"name": "Beacon lens", "danger": "high"},
                    category="scene",
                    confidence=0.92,
                    source_message_id=narrator_message.id,
                ),
                ExtractedStateChange(
                    operation="upsert",
                    key="npc.warden.elian",
                    value={"name": "Elian", "status": "missing"},
                    category="npc",
                    confidence=0.81,
                    source_message_id=narrator_message.id,
                ),
            ),
            memories=(
                ExtractedMemory(
                    body="Mara promised to relight the Ashfall beacon.",
                    tags=("promise", "beacon"),
                    importance=0.74,
                    source_message_id=player_message.id,
                ),
            ),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert len(extractor.requests) == 1
    request = extractor.requests[0]
    assert [message.id for message in request.messages] == [
        player_message.id,
        narrator_message.id,
    ]
    assert [(state.key, state.value) for state in request.current_state] == [
        ("scene.location", {"name": "Lower stair", "danger": "low"}),
    ]
    assert "starting_scene" not in request.scenario_context
    assert "The beacon gutters in the tower." not in request.scenario_context

    world_state = repositories.list_world_state(save.id)
    assert [(state.key, state.value, state.category) for state in world_state] == [
        ("npc.warden.elian", {"name": "Elian", "status": "missing"}, "npc"),
        ("scene.location", {"name": "Beacon lens", "danger": "high"}, "scene"),
    ]
    assert world_state[1].id == existing_state.id
    assert repositories.list_memories(save.id)[0].body == (
        "Mara promised to relight the Ashfall beacon."
    )

    changes = _state_changes(repositories, save.id)
    assert [
        (change["operation"], change["state_key"], change["source_message_id"])
        for change in changes
    ] == [
        ("upsert", "scene.location", narrator_message.id),
        ("upsert", "npc.warden.elian", narrator_message.id),
    ]
    assert json.loads(changes[0]["before_json"]) == {
        "name": "Lower stair",
        "danger": "low",
    }
    assert json.loads(changes[0]["after_json"]) == {
        "name": "Beacon lens",
        "danger": "high",
    }
    assert changes[1]["before_json"] is None
    assert json.loads(changes[1]["after_json"]) == {
        "name": "Elian",
        "status": "missing",
    }


def test_extract_and_apply_turn_filters_marked_narrator_state_and_memory(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.connection.execute(
        "UPDATE messages SET body = ?, safety_transition = ? WHERE id = ?",
        (
            "The intimate moment is kept off-screen. Hours later, "
            "the next scene begins.",
            "fade_to_black",
            narrator_message.id,
        ),
    )
    repositories.commit()
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="relationship.private_detail",
                    value={"detail": "rejected draft"},
                    category="relationship",
                    confidence=0.9,
                    source_message_id=narrator_message.id,
                ),
            ),
            memories=(
                ExtractedMemory(
                    body="The player promised to keep watch.",
                    tags=("promise",),
                    importance=0.7,
                    source_message_id=player_message.id,
                ),
            ),
        )
    )

    result = asyncio.run(
        StateService(
            repositories=repositories,
            extractor=extractor,
        ).extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert extractor.requests[0].messages[-1].body.startswith(
        "The intimate moment is kept off-screen."
    )
    assert not repositories.list_world_state(save.id)
    assert [memory.body for memory in repositories.list_memories(save.id)] == [
        "The player promised to keep watch."
    ]
    assert result.suppressed_state_change_count == 1


def test_extract_and_apply_turn_preserves_replaced_durable_state_as_memory(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="character.ilyra.revealed_traits",
        value={
            "trusts_mara": True,
            "secret": "Knows the cracked red lens is a family heirloom.",
        },
        category="character",
        confidence=0.9,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="character.ilyra.revealed_traits",
                    value={
                        "trusts_mara": True,
                        "secret": "Revealed the heirloom lens to Mara.",
                    },
                    category="character",
                    confidence=0.88,
                    source_message_id=narrator_message.id,
                ),
            ),
        )
    )

    StateService(repositories=repositories, extractor=extractor).apply_extraction(
        save_id=save.id,
        extraction=extractor.extraction,
        allowed_source_message_ids=(player_message.id, narrator_message.id),
    )

    memories = repositories.list_memories(save.id)
    assert len(memories) == 1
    assert memories[0].body == (
        "Previous world state for character.ilyra.revealed_traits: "
        "secret: Knows the cracked red lens is a family heirloom."
    )
    assert set(memories[0].tags) >= {
        "state_history",
        "from_world_state",
        "state_key:character.ilyra.revealed_traits",
    }
    assert memories[0].source_message_ids == [
        player_message.id,
        narrator_message.id,
    ]


def test_extract_and_apply_turn_does_not_preserve_scene_state_overwrites(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Lower stair"},
        category="scene",
        confidence=0.9,
        source_message_id=player_message.id,
    )
    extraction = StateExtraction(
        state_changes=(
            ExtractedStateChange(
                operation="upsert",
                key="scene.location",
                value={"name": "Beacon gallery"},
                category="scene",
                confidence=0.88,
                source_message_id=narrator_message.id,
            ),
        ),
    )

    StateService(
        repositories=repositories,
        extractor=StructuredFakeExtractor(extraction),
    ).apply_extraction(
        save_id=save.id,
        extraction=extraction,
        allowed_source_message_ids=(player_message.id, narrator_message.id),
    )

    assert repositories.list_memories(save.id) == []


def test_extract_and_apply_turn_updates_time_loop_persistent_knowledge(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="time_loop",
        title="Bellwether Day",
        premise="A harbor festival repeats until the drowned bell is saved.",
        player_role="Loop-aware archivist",
        content={
            "loop_premise": "The festival day repeats until the bell is saved.",
            "persistent_knowledge": "The tower code is still unknown.",
            "npc_memory_rules": "NPCs reset to dawn memories unless excepted.",
            "current_loop_state": "Loop 2, storm phase.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Bell Loop")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I write down the tower code 4312 before the tide resets.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The tower code 4312 remains clear to you when dawn repeats.",
        provider="fake",
        model="fake-chat",
    )
    existing_knowledge = repositories.upsert_world_state(
        save_id=save.id,
        key="loop.knowledge",
        value={"summary": "Known from prior loops: Mira distrusts the tower key."},
        category="loop_persistent",
        confidence=1.0,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="loop.knowledge",
                    value={
                        "summary": (
                            "Known from prior loops: Mira distrusts the tower key; "
                            "tower code 4312 persists for the player."
                        )
                    },
                    category="loop_persistent",
                    confidence=0.93,
                    source_message_id=narrator_message.id,
                ),
            ),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert "persistent_knowledge: The tower code is still unknown" in (
        extractor.requests[0].scenario_context
    )
    assert "npc_memory_rules: NPCs reset" in extractor.requests[0].scenario_context
    assert [state.key for state in extractor.requests[0].current_state] == [
        "loop.knowledge"
    ]
    assert applied.world_state[0].id == existing_knowledge.id
    saved_knowledge = repositories.list_world_state(save.id)[0]
    assert saved_knowledge.id == existing_knowledge.id
    assert saved_knowledge.category == "loop_persistent"
    assert saved_knowledge.value == {
        "summary": (
            "Known from prior loops: Mira distrusts the tower key; tower code 4312 "
            "persists for the player."
        )
    }
    changes = _state_changes(repositories, save.id)
    assert len(changes) == 1
    assert changes[0]["operation"] == "upsert"
    assert changes[0]["state_key"] == "loop.knowledge"
    assert json.loads(changes[0]["before_json"]) == {
        "summary": "Known from prior loops: Mira distrusts the tower key."
    }
    assert json.loads(changes[0]["after_json"]) == {
        "summary": (
            "Known from prior loops: Mira distrusts the tower key; tower code 4312 "
            "persists for the player."
        )
    }


def test_extract_and_apply_turn_updates_heist_alert_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="heist_infiltration",
        title="Skybank Treaty Job",
        premise="A crew must steal a treaty from a floating bank.",
        player_role="Crew planner",
        content={
            "security_model": "Clockwork cameras and warded locks.",
            "alert_and_heat": "Suspicion low; alarm inactive.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Treaty Job")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I cut the treaty case glass even though the ward begins screaming.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=(
            "The silent alarm flips active, suspicion jumps to high, and the "
            "east guard post starts sealing the gallery."
        ),
    )
    existing_alert = repositories.upsert_world_state(
        save_id=save.id,
        key="heist.alert",
        value={"level": "low", "alarm": "inactive"},
        category="threat",
        confidence=1.0,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="heist.alert",
                    value={
                        "level": "high",
                        "alarm": "active",
                        "response": "east guard post sealing gallery",
                    },
                    category="threat",
                    confidence=0.93,
                    source_message_id=narrator_message.id,
                    evidence_quote="The silent alarm flips active",
                    persistence_scope="scene",
                ),
            )
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert len(applied.state_changes) == 1
    request = extractor.requests[0]
    assert request.scenario_type == "heist_infiltration"
    assert "security_model: Clockwork cameras and warded locks." in (
        request.scenario_context
    )
    state_by_key = {
        state.key: state for state in repositories.list_world_state(save.id)
    }
    assert state_by_key["heist.alert"].id == existing_alert.id
    assert state_by_key["heist.alert"].value == {
        "level": "high",
        "alarm": "active",
        "response": "east guard post sealing gallery",
    }
    changes = _state_changes(repositories, save.id)
    assert [(change["operation"], change["state_key"]) for change in changes] == [
        ("upsert", "heist.alert"),
    ]
    assert json.loads(changes[0]["before_json"]) == {
        "level": "low",
        "alarm": "inactive",
    }
    assert json.loads(changes[0]["after_json"]) == {
        "level": "high",
        "alarm": "active",
        "response": "east guard post sealing gallery",
    }


def test_extract_and_apply_turn_updates_political_intrigue_social_consequence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="political_intrigue",
        title="Council of Ash",
        premise="A city council vote will decide who controls the harbor.",
        player_role="Envoy holding the swing vote",
        content={
            "political_factions": "Guilds, Old Families, and reformers.",
            "factions": "Guildmaster Orro owes Mara one favor.",
            "political_pressure": "The midnight vote proceeds unless delayed.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Ash Council")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body=(
            "I publicly back the reformers and remind Orro he promised an "
            "endorsement."
        ),
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=(
            "Orro bows and says he owes Mara one public endorsement, while "
            "reformers now call Mara trusted."
        ),
    )
    existing_standing = repositories.upsert_world_state(
        save_id=save.id,
        key="faction.reformers.standing",
        value={"standing": "neutral"},
        category="reputation",
        confidence=1.0,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="obligation.orro.owed_to_mara",
                    value={"favor": "public endorsement", "status": "owed"},
                    category="obligation",
                    confidence=0.94,
                    source_message_id=narrator_message.id,
                    evidence_quote="owes Mara one public endorsement",
                    persistence_scope="durable",
                ),
                ExtractedStateChange(
                    operation="upsert",
                    key="faction.reformers.standing",
                    value={"standing": "trusted"},
                    category="reputation",
                    confidence=0.91,
                    source_message_id=narrator_message.id,
                    evidence_quote="reformers now call Mara trusted",
                    persistence_scope="durable",
                ),
            )
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert len(applied.state_changes) == 2
    request = extractor.requests[0]
    assert request.scenario_type == "political_intrigue"
    assert "political_pressure: The midnight vote proceeds unless delayed." in (
        request.scenario_context
    )
    state_by_key = {
        state.key: state for state in repositories.list_world_state(save.id)
    }
    assert state_by_key["faction.reformers.standing"].id == existing_standing.id
    assert state_by_key["faction.reformers.standing"].value == {
        "standing": "trusted"
    }
    assert state_by_key["faction.reformers.standing"].category == "reputation"
    assert state_by_key["obligation.orro.owed_to_mara"].value == {
        "favor": "public endorsement",
        "status": "owed",
    }
    assert state_by_key["obligation.orro.owed_to_mara"].category == "obligation"
    changes = _state_changes(repositories, save.id)
    assert [(change["operation"], change["state_key"]) for change in changes] == [
        ("upsert", "obligation.orro.owed_to_mara"),
        ("upsert", "faction.reformers.standing"),
    ]
    assert json.loads(changes[1]["before_json"]) == {"standing": "neutral"}
    assert json.loads(changes[1]["after_json"]) == {"standing": "trusted"}


@pytest.mark.parametrize(
    (
        "scenario_type",
        "content",
        "player_body",
        "narrator_body",
        "state_key",
        "before_value",
        "after_value",
        "category",
        "context_snippet",
    ),
    [
        (
            "settlement_builder",
            {
                "projects_and_facilities": "Repair the flood gate before spring.",
                "resources_and_indicators": "Food low; morale fragile.",
            },
            "I assign the masons to the flood gate and spend half the lumber.",
            "The flood gate reaches half-built, but the lumber store drops to low.",
            "settlement.projects",
            {"summary": "Flood gate planned; lumber adequate."},
            {"summary": "Flood gate half-built; lumber low."},
            "project",
            "projects_and_facilities: Repair the flood gate before spring.",
        ),
        (
            "monster_hunt_bounty",
            {
                "target_profile": "The Thornback avoids firelight.",
                "leads_and_clues": "Blue sap marks its trail.",
            },
            "I show the witness the blue sap from the broken arrow.",
            "She confirms the sap came from the old orchard where the beast fed.",
            "hunt.leads",
            {"summary": "Blue sap clue undiscovered."},
            {"summary": "Blue sap clue confirmed at old orchard."},
            "clue",
            "leads_and_clues: Blue sap marks its trail.",
        ),
        (
            "road_trip_pilgrimage",
            {
                "route_and_stops": "Lantern Ford, Crow Market, Saint Orra.",
                "journey_progress": "Current leg: city gate to Lantern Ford.",
            },
            "We detour around the washed bridge toward Crow Market.",
            "The party leaves the Lantern Ford road and commits to Crow Market.",
            "journey.progress",
            {"summary": "Current leg: road to Lantern Ford."},
            {"summary": "Detoured from Lantern Ford road toward Crow Market."},
            "objective",
            "journey_progress: Current leg: city gate to Lantern Ford.",
        ),
        (
            "merchant_trade_route",
            {
                "cargo_inventory": "Cedar oil: 20 jars.",
                "contracts_and_debts": "Deliver ten jars in twelve days.",
            },
            "I sell five jars to pay the axle debt.",
            "Five cedar oil jars are gone and the axle debt is paid.",
            "trade.contracts",
            {"summary": "Axle debt unpaid; twenty oil jars aboard."},
            {"summary": "Axle debt paid; fifteen oil jars aboard."},
            "contract",
            "contracts_and_debts: Deliver ten jars in twelve days.",
        ),
    ],
)
def test_extract_and_apply_turn_updates_management_template_state(
    repositories: PersistenceRepositories,
    scenario_type: str,
    content: dict[str, str],
    player_body: str,
    narrator_body: str,
    state_key: str,
    before_value: dict[str, object],
    after_value: dict[str, object],
    category: str,
    context_snippet: str,
) -> None:
    scenario = repositories.create_scenario(
        type=scenario_type,
        title="Template Scenario",
        premise="A persistent campaign tracks practical consequences.",
        player_role="Decision maker",
        content=cast(dict[str, object], dict(content)),
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Template Save")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body=player_body,
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=narrator_body,
    )
    existing_state = repositories.upsert_world_state(
        save_id=save.id,
        key=state_key,
        value=before_value,
        category=category,
        confidence=1.0,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key=state_key,
                    value=after_value,
                    category=category,
                    confidence=0.92,
                    source_message_id=narrator_message.id,
                    evidence_quote=narrator_body.split(".", 1)[0],
                    persistence_scope="durable",
                ),
            )
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert len(applied.state_changes) == 1
    request = extractor.requests[0]
    assert request.scenario_type == scenario_type
    assert context_snippet in request.scenario_context
    saved_state = repositories.list_world_state(save.id)[0]
    assert saved_state.id == existing_state.id
    assert saved_state.key == state_key
    assert saved_state.value == after_value
    assert saved_state.category == category


def test_extract_and_apply_turn_rejects_unexpected_generated_script(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    unrelated_multilingual_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="旁白说玩家正在检查灯塔。",
        provider="fake",
        model="fake-chat",
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="scene.warning",
                    value={"description": "红色透镜显示灰烬中的骑手。"},
                    category="scene",
                    confidence=0.92,
                    source_message_id=narrator_message.id,
                ),
            ),
            memories=(
                ExtractedMemory(
                    body="玩家喜欢简洁、扎实的叙事。",
                    tags=("tone",),
                    importance=0.74,
                    source_message_id=player_message.id,
                ),
            ),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(
                player_message.id,
                narrator_message.id,
                unrelated_multilingual_message.id,
            ),
        )
    )

    assert applied.world_state == ()
    assert applied.memories == ()
    assert applied.suppressed_state_change_count == 1
    assert applied.suppressed_memory_count == 1
    assert repositories.list_world_state(save.id) == []
    assert repositories.list_memories(save.id) == []


def test_structured_state_parser_wraps_string_values_and_keeps_dict_values() -> None:
    extraction = _state_extraction_from_structured_data(
        {
            "state_changes": [
                {
                    "operation": "upsert",
                    "key": "scene.mood",
                    "value": "urgent and brittle",
                    "category": "scene",
                    "confidence": 0.8,
                    "source_message_id": "message-1",
                    "evidence_quote": "urgent and brittle",
                },
                {
                    "operation": "upsert",
                    "key": "scene.location",
                    "value": {"name": "Beacon tower"},
                    "category": "scene",
                    "confidence": 0.9,
                    "source_message_id": "message-2",
                    "evidence_quote": "Beacon tower",
                },
            ],
            "memories": [],
        }
    )

    assert extraction.state_changes[0].value == {"text": "urgent and brittle"}
    assert extraction.state_changes[0].evidence_quote == "urgent and brittle"
    assert extraction.state_changes[1].value == {"name": "Beacon tower"}
    assert extraction.state_changes[1].evidence_quote == "Beacon tower"


def test_structured_state_parser_keeps_memory_grounding_quotes() -> None:
    extraction = _state_extraction_from_structured_data(
        {
            "state_changes": [],
            "memories": [
                {
                    "body": "Mara promised to relight the beacon.",
                    "tags": ["promise"],
                    "importance": 0.74,
                    "source_message_id": "message-1",
                    "evidence_quote": "promise to relight",
                }
            ],
        }
    )

    assert extraction.memories[0].evidence_quote == "promise to relight"


def test_structured_state_parser_records_conflicts_and_suppresses_same_key_patch(
) -> None:
    extraction = _state_extraction_from_structured_data(
        {
            "state_changes": [
                {
                    "operation": "upsert",
                    "key": "npc.warden.elian",
                    "value": {"status": "missing"},
                    "category": "npc",
                    "confidence": 0.8,
                    "source_message_id": "message-1",
                    "evidence_quote": "Elian's post is empty.",
                },
                {
                    "operation": "upsert",
                    "key": "scene.location",
                    "value": "Beacon tower",
                    "category": "scene",
                    "confidence": 0.9,
                    "source_message_id": "message-1",
                    "evidence_quote": "Beacon tower",
                },
            ],
            "memories": [],
            "conflicts": [
                {
                    "key": "npc.warden.elian",
                    "source_message_id": "message-1",
                    "new_evidence": "Elian's post is empty.",
                    "current_value": {"status": "on duty"},
                    "proposed_value": "missing from the post",
                    "reason": "The new turn contradicts the active duty record.",
                    "confidence": 0.7,
                }
            ],
        }
    )

    assert [change.key for change in extraction.state_changes] == ["scene.location"]
    assert extraction.state_changes[0].value == {"text": "Beacon tower"}
    assert len(extraction.conflicts) == 1
    conflict = extraction.conflicts[0]
    assert conflict.key == "npc.warden.elian"
    assert conflict.current_value == {"status": "on duty"}
    assert conflict.proposed_value == {"text": "missing from the post"}
    assert conflict.confidence == 0.7


def test_structured_state_conflict_does_not_overwrite_existing_world_state(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="npc.warden.elian",
        value={"status": "on duty"},
        category="npc",
        source_message_id=player_message.id,
    )
    provider = SequenceStructuredOutputProvider(
        responses=[
            {
                "state_changes": [
                    {
                        "operation": "upsert",
                        "key": "npc.warden.elian",
                        "value": {"status": "missing"},
                        "category": "npc",
                        "confidence": 0.8,
                        "source_message_id": narrator_message.id,
                        "evidence_quote": "empty post",
                    }
                ],
                "memories": [],
                "conflicts": [
                    {
                        "key": "npc.warden.elian",
                        "source_message_id": narrator_message.id,
                        "new_evidence": "Elian's empty post.",
                        "current_value": {"status": "on duty"},
                        "proposed_value": {"status": "missing"},
                    }
                ],
            }
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=StructuredProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-structured",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert applied.world_state == ()
    assert repositories.list_world_state(save.id)[0].value == {"status": "on duty"}
    assert _state_changes(repositories, save.id) == []
    assert _jobs(repositories, save.id, "state_extraction")[-1]["result"][
        "conflict_count"
    ] == 1


def test_structured_state_only_request_omits_memory_schema_and_accepts_absent_memories(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceStructuredOutputProvider(
        responses=[
            {
                "state_changes": [
                    {
                        "operation": "upsert",
                        "key": "scene.location",
                        "value": {"name": "Beacon gallery"},
                        "category": "scene",
                        "confidence": 0.87,
                        "source_message_id": narrator_message.id,
                        "evidence_quote": "lens flares",
                    }
                ],
                "conflicts": [
                    {
                        "key": "npc.warden.elian",
                        "source_message_id": narrator_message.id,
                        "new_evidence": "Warden Elian's empty post.",
                    }
                ],
            }
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=StructuredProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-structured",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
            include_memories=False,
        )
    )

    structured_request = provider.structured_output_requests[0]
    assert structured_request.max_output_tokens == 2048
    assert "memories" not in structured_request.schema["properties"]
    assert "memories" not in structured_request.schema["required"]
    prompt_text = "\n".join(
        message.body.casefold() for message in structured_request.messages
    )
    assert "memories" not in prompt_text
    assert [state.key for state in applied.world_state] == ["scene.location"]
    assert applied.memories == ()
    assert _jobs(repositories, save.id, "state_extraction")[-1]["result"][
        "conflict_count"
    ] == 1


def test_structured_state_extractor_applies_grounded_state_memory_and_conflict(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="npc.warden.elian",
        value={"status": "on duty"},
        category="npc",
        source_message_id=player_message.id,
    )
    provider = SequenceStructuredOutputProvider(
        responses=[_grounded_structured_response(player_message, narrator_message)]
    )
    service = StateService(
        repositories=repositories,
        extractor=StructuredProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-structured",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert [state.key for state in applied.world_state] == ["scene.location"]
    assert applied.memories[0].body == "Mara promised to relight the beacon."
    state_by_key = {
        state.key: state.value for state in repositories.list_world_state(save.id)
    }
    assert state_by_key == {
        "npc.warden.elian": {"status": "on duty"},
        "scene.location": {"name": "Beacon lens"},
    }
    assert _jobs(repositories, save.id, "state_extraction")[-1]["result"][
        "conflict_count"
    ] == 1


@pytest.mark.parametrize(
    ("case", "match"),
    [
        (
            "missing_state_quote",
            "Structured state change grounding failed: evidence_quote is required",
        ),
        (
            "bad_memory_quote",
            "Structured memory grounding failed: evidence_quote not found",
        ),
        (
            "missing_conflict_evidence",
            "Structured state conflict grounding failed: new_evidence is required",
        ),
        (
            "bad_source_id",
            "source_message_id is not in the completed turn",
        ),
    ],
)
def test_structured_state_extractor_rejects_ungrounded_entries_without_mutation(
    case: str,
    match: str,
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="npc.warden.elian",
        value={"status": "on duty"},
        category="npc",
        source_message_id=player_message.id,
    )
    response = _grounded_structured_response(player_message, narrator_message)
    if case == "missing_state_quote":
        cast(list[dict[str, Any]], response["state_changes"])[0].pop(
            "evidence_quote"
        )
    elif case == "bad_memory_quote":
        cast(list[dict[str, Any]], response["memories"])[0][
            "evidence_quote"
        ] = "not in the source message"
    elif case == "missing_conflict_evidence":
        cast(list[dict[str, Any]], response["conflicts"])[0]["new_evidence"] = ""
    elif case == "bad_source_id":
        cast(list[dict[str, Any]], response["memories"])[0][
            "source_message_id"
        ] = "missing-message"
    provider = SequenceStructuredOutputProvider(responses=[response])
    service = StateService(
        repositories=repositories,
        extractor=StructuredProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-structured",
        ),
    )

    with pytest.raises(ValueError, match=match):
        asyncio.run(
            service.extract_and_apply_turn(
                save_id=save.id,
                source_message_ids=(player_message.id, narrator_message.id),
            )
        )

    assert [
        (state.key, state.value)
        for state in repositories.list_world_state(save.id)
    ] == [("npc.warden.elian", {"status": "on duty"})]
    assert repositories.list_memories(save.id) == []
    assert _state_changes(repositories, save.id) == []
    failed_jobs = _jobs(repositories, save.id, "state_extraction")
    assert failed_jobs[-1]["status"] == "failed"
    assert match in failed_jobs[-1]["error"]


def test_tool_calling_state_only_request_omits_memory_tool_and_prompt(
    repositories: PersistenceRepositories,
) -> None:
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(responses=[()])
    extractor = ToolCallingProviderStateExtractor(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    extraction = asyncio.run(
        extractor.extract(
            StateExtractionRequest(
                save_id=save.id,
                messages=(narrator_message,),
                current_state=(),
                include_memories=False,
            )
        )
    )

    assert extraction.memories == ()
    tool_request = provider.tool_call_requests[0]
    assert [tool.name for tool in tool_request.tools] == [
        "patch_world_state",
        "flag_state_conflict",
    ]
    prompt_text = "\n".join(message.body for message in tool_request.messages)
    assert "record_memory_fact" not in prompt_text
    assert "durable memory facts" not in prompt_text


def test_state_only_structured_schema_has_source_enums_without_memories(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    schema = _state_extraction_schema(
        StateExtractionRequest(
            save_id=save.id,
            messages=(player_message, narrator_message),
            current_state=(),
            include_memories=False,
        )
    )

    properties = cast(dict[str, Any], schema["properties"])
    assert "memories" not in properties
    assert cast(list[str], schema["required"]) == ["state_changes", "conflicts"]
    assert "evidence_quote" in _schema_required_fields(properties, "state_changes")
    assert "new_evidence" in _schema_required_fields(properties, "conflicts")
    assert _schema_source_enum(properties, "state_changes") == [
        player_message.id,
        narrator_message.id,
    ]
    assert _schema_source_enum(properties, "conflicts") == [
        player_message.id,
        narrator_message.id,
    ]


def test_structured_schema_requires_grounding_for_state_memories_and_conflicts(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    schema = _state_extraction_schema(
        StateExtractionRequest(
            save_id=save.id,
            messages=(player_message, narrator_message),
            current_state=(),
        )
    )

    properties = cast(dict[str, Any], schema["properties"])
    assert "evidence_quote" in _schema_required_fields(properties, "state_changes")
    assert "evidence_quote" in _schema_required_fields(properties, "memories")
    assert "new_evidence" in _schema_required_fields(properties, "conflicts")
    assert _schema_source_enum(properties, "state_changes") == [
        player_message.id,
        narrator_message.id,
    ]
    assert _schema_source_enum(properties, "memories") == [
        player_message.id,
        narrator_message.id,
    ]
    assert _schema_source_enum(properties, "conflicts") == [
        player_message.id,
        narrator_message.id,
    ]


def test_message_correction_archives_state_and_memories_from_old_message(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.beacon",
        value={"status": "cracked"},
        category="scene",
        confidence=0.8,
        source_message_id=narrator_message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="keeper.promise",
        value={"status": "active"},
        category="memory",
        confidence=0.8,
        source_message_id=player_message.id,
    )
    repositories.add_memory(
        save_id=save.id,
        body="The beacon cracked open.",
        tags=["beacon"],
        source_message_id=narrator_message.id,
    )
    repositories.add_memory(
        save_id=save.id,
        body="Mara promised to relight the beacon.",
        tags=["promise"],
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="scene.beacon",
                    value={"status": "steady"},
                    category="scene",
                    confidence=0.9,
                    source_message_id=narrator_message.id,
                ),
            ),
            memories=(
                ExtractedMemory(
                    body="The beacon is steady.",
                    tags=("beacon",),
                    importance=0.7,
                    source_message_id=narrator_message.id,
                ),
            ),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    asyncio.run(
        service.extract_and_apply_message_correction(
            save_id=save.id,
            source_message_id=narrator_message.id,
            correction_context=MessageCorrectionContext(
                message_id=narrator_message.id,
                previous_body="The beacon cracked open.",
                new_body="The beacon is steady.",
                diff_unified="-The beacon cracked open.\n+The beacon is steady.",
            ),
        )
    )

    assert {
        state.key: state.value for state in repositories.list_world_state(save.id)
    } == {
        "keeper.promise": {"status": "active"},
        "scene.beacon": {"status": "steady"},
    }
    assert [memory.body for memory in repositories.list_memories(save.id)] == [
        "Mara promised to relight the beacon.",
        "The beacon is steady.",
    ]


def test_tool_calling_state_extractor_applies_valid_exact_quote_calls(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="state-call",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "npc.warden.elian",
                            "value_patch": {"status": "missing"},
                            "category": "npc",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "empty post",
                            "confidence": 0.83,
                        }
                    ),
                ),
                ProviderToolCall(
                    id="memory-call",
                    name="record_memory_fact",
                    arguments_json=json.dumps(
                        {
                            "body": "Mara promised to relight the beacon.",
                            "tags": ["promise"],
                            "importance": 0.71,
                            "source_message_id": player_message.id,
                            "evidence_quote": "promise to relight",
                        }
                    ),
                ),
            )
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert [state.key for state in applied.world_state] == ["npc.warden.elian"]
    assert repositories.list_world_state(save.id)[0].value == {"status": "missing"}
    assert repositories.list_memories(save.id)[0].body == (
        "Mara promised to relight the beacon."
    )
    jobs = _jobs(repositories, save.id, "state_extraction")
    diagnostics = jobs[-1]["result"]["tool_diagnostics"]
    assert diagnostics["retry_count"] == 0
    assert diagnostics["accepted_calls"][0]["name"] == "patch_world_state"


def test_tool_calling_state_extractor_retries_unexpected_generated_script(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="memory-call",
                    name="record_memory_fact",
                    arguments_json=json.dumps(
                        {
                            "body": "玩家承诺重新点燃灯塔。",
                            "tags": ["promise"],
                            "importance": 0.71,
                            "source_message_id": player_message.id,
                            "evidence_quote": "promise to relight",
                        }
                    ),
                ),
            ),
            (),
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert applied.memories == ()
    assert repositories.list_memories(save.id) == []
    assert len(provider.tool_call_requests) == 2
    diagnostics = _jobs(repositories, save.id, "state_extraction")[-1]["result"][
        "tool_diagnostics"
    ]
    assert diagnostics["retry_count"] == 1
    assert diagnostics["rejected_calls"][0]["name"] == "record_memory_fact"
    assert "script policy" in diagnostics["rejected_calls"][0]["error"]


def test_tool_calling_state_extractor_accepts_format_normalized_quotes(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, _narrator_message = _save_with_completed_turn(repositories)
    formatted_narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lens flares and reveals Warden\nElian's **empty post**.",
        provider="fake",
        model="fake-chat",
    )
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="state-call",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "npc.warden.elian",
                            "value_patch": {"status": "missing"},
                            "category": "npc",
                            "source_message_id": formatted_narrator_message.id,
                            "evidence_quote": "Warden Elian's empty post",
                            "confidence": 0.83,
                        }
                    ),
                ),
            )
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, formatted_narrator_message.id),
        )
    )

    assert [state.key for state in applied.world_state] == ["npc.warden.elian"]
    assert repositories.list_world_state(save.id)[0].value == {"status": "missing"}


def test_tool_calling_state_extractor_keeps_independent_valid_calls_after_bad_quotes(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id=f"state-call-{index}",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "npc.warden.elian",
                            "value_patch": {"status": "missing"},
                            "category": "npc",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": (
                                "not in the narrator message"
                                if index == 0
                                else "empty post"
                            ),
                            "confidence": 0.83,
                        }
                    ),
                ),
                ProviderToolCall(
                    id=f"bad-memory-{index}",
                    name="record_memory_fact",
                    arguments_json=json.dumps(
                        {
                            "body": "Mara promised to guard the lower stair.",
                            "source_message_id": player_message.id,
                            "evidence_quote": "guard the lower stair",
                        }
                    ),
                ),
            )
            for index in range(3)
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert [state.key for state in applied.world_state] == ["npc.warden.elian"]
    assert repositories.list_world_state(save.id)[0].value == {"status": "missing"}
    assert repositories.list_memories(save.id) == []
    jobs = _jobs(repositories, save.id, "state_extraction")
    assert jobs[-1]["status"] == "succeeded"
    diagnostics = jobs[-1]["result"]["tool_diagnostics"]
    assert diagnostics["partial_success"] is True
    assert diagnostics["retry_count"] == 7
    assert [call["id"] for call in diagnostics["accepted_calls"]] == [
        "state-call-1"
    ]
    assert {call["id"] for call in diagnostics["rejected_calls"]} == {
        "state-call-0",
        "bad-memory-0",
        "bad-memory-1",
        "bad-memory-2",
    }


def test_tool_calling_state_extractor_drops_partial_patch_for_unsafe_key(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="npc.warden.elian",
        value={"status": "on duty"},
        category="npc",
        source_message_id=player_message.id,
    )
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id=f"state-call-{index}",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "npc.warden.elian",
                            "value_patch": {"status": "missing"},
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "empty post",
                        }
                    ),
                ),
                ProviderToolCall(
                    id=f"bad-conflict-{index}",
                    name="flag_state_conflict",
                    arguments_json=json.dumps(
                        {
                            "key": "npc.warden.elian",
                            "source_message_id": narrator_message.id,
                            "new_evidence": "vanished from the post",
                            "current_value": {"status": "on duty"},
                            "proposed_value": {"status": "missing"},
                        }
                    ),
                ),
                ProviderToolCall(
                    id=f"memory-call-{index}",
                    name="record_memory_fact",
                    arguments_json=json.dumps(
                        {
                            "body": "Mara promised to relight the beacon.",
                            "source_message_id": player_message.id,
                            "evidence_quote": "promise to relight",
                        }
                    ),
                ),
            )
            for index in range(3)
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert applied.world_state == ()
    assert repositories.list_world_state(save.id)[0].value == {"status": "on duty"}
    assert repositories.list_memories(save.id)[0].body == (
        "Mara promised to relight the beacon."
    )
    diagnostics = _jobs(repositories, save.id, "state_extraction")[-1]["result"][
        "tool_diagnostics"
    ]
    assert diagnostics["partial_success"] is True
    assert diagnostics["partial_suppressed_state_keys"] == ["npc.warden.elian"]


def test_tool_calling_state_extractor_drops_partial_state_when_rejected_key_unknown(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id=f"state-call-{index}",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "npc.warden.elian",
                            "value_patch": {"status": "missing"},
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "empty post",
                        }
                    ),
                ),
                ProviderToolCall(
                    id=f"malformed-state-{index}",
                    name="patch_world_state",
                    arguments_json='{"source_message_id":',
                ),
                ProviderToolCall(
                    id=f"memory-call-{index}",
                    name="record_memory_fact",
                    arguments_json=json.dumps(
                        {
                            "body": "Mara promised to relight the beacon.",
                            "source_message_id": player_message.id,
                            "evidence_quote": "promise to relight",
                        }
                    ),
                ),
            )
            for index in range(3)
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert applied.world_state == ()
    assert repositories.list_world_state(save.id) == []
    assert repositories.list_memories(save.id)[0].body == (
        "Mara promised to relight the beacon."
    )
    diagnostics = _jobs(repositories, save.id, "state_extraction")[-1]["result"][
        "tool_diagnostics"
    ]
    assert diagnostics["partial_success"] is True
    assert diagnostics["partial_suppressed_state_keys"] == ["npc.warden.elian"]


def test_tool_calling_state_extractor_records_prompt_inspection(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="memory-call",
                    name="record_memory_fact",
                    arguments_json=json.dumps(
                        {
                            "body": "Mara promised to relight the beacon.",
                            "source_message_id": player_message.id,
                            "evidence_quote": "promise to relight",
                        }
                    ),
                ),
            )
        ]
    )
    store = PromptInspectionStore()
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
            prompt_inspection_store=store,
        ),
    )

    asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    prompt_text = store.prompt_for_message(narrator_message.id) or ""
    assert "State and memory tool calls" in prompt_text
    assert "Tool messages" in prompt_text
    assert "record_memory_fact" in prompt_text
    assert '"model_id": "fake-tools"' in prompt_text
    assert [entry.kind for entry in store.entries_for_message(narrator_message.id)] == [
        "state_memory_tool_calls"
    ]


def test_tool_calling_state_extractor_rejects_invalid_source_message_id(
    repositories: PersistenceRepositories,
) -> None:
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id=f"bad-source-{index}",
                    name="record_memory_fact",
                    arguments_json=json.dumps(
                        {
                            "body": "Mara promised to relight the beacon.",
                            "source_message_id": "missing-message",
                            "evidence_quote": "promise to relight",
                        }
                    ),
                ),
            )
            for index in range(3)
        ]
    )
    extractor = ToolCallingProviderStateExtractor(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            extractor.extract(
                _state_request(
                    save_id=save.id,
                    messages=(narrator_message,),
                    repositories=repositories,
                )
            )
        )

    assert "source_message_id is not in the completed turn" in str(exc_info.value)
    assert len(provider.tool_call_requests) == 7


def test_tool_calling_state_extractor_rejects_malformed_tool_args(
    repositories: PersistenceRepositories,
) -> None:
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id=f"malformed-{index}",
                    name="record_memory_fact",
                    arguments_json='{"source_message_id":',
                ),
            )
            for index in range(3)
        ]
    )
    extractor = ToolCallingProviderStateExtractor(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            extractor.extract(
                _state_request(
                    save_id=save.id,
                    messages=(narrator_message,),
                    repositories=repositories,
                )
            )
        )

    assert "Malformed JSON arguments" in str(exc_info.value)
    retry_messages = provider.tool_call_requests[1].messages
    assert any(
        "Call exactly one tool again" in message.body
        for message in retry_messages
        if message.role == "tool"
    )


def test_tool_calling_state_extractor_switches_to_structured_route_on_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-tools",
        display_name="Fake Tools",
        capabilities=["tool_calling", "structured_output"],
    )
    provider = ShapeSwitchToolCallProvider(
        structured_data=_grounded_structured_response(
            player_message,
            narrator_message,
        )
    )
    extractor = ToolCallingProviderStateExtractor(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
        repositories=repositories,
        providers={"fake": cast(Any, provider)},
    )

    extraction = asyncio.run(
        extractor.extract(
            _state_request(
                save_id=save.id,
                messages=(player_message, narrator_message),
                repositories=repositories,
            )
        )
    )

    assert len(provider.tool_call_requests) == 1
    assert len(provider.structured_output_requests) == 1
    assert [change.key for change in extraction.state_changes] == ["scene.location"]
    assert extraction.tool_diagnostics["shape_switch"] == "structured_output"
    assert extraction.tool_diagnostics["provider"] == "fake"


def test_tool_calling_state_extractor_recovers_without_fallback_infrastructure(
    repositories: PersistenceRepositories,
) -> None:
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = ShapeSwitchToolCallProvider()
    extractor = ToolCallingProviderStateExtractor(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    extraction = asyncio.run(
        extractor.extract(
            _state_request(
                save_id=save.id,
                messages=(narrator_message,),
                repositories=repositories,
            )
        )
    )

    assert len(provider.tool_call_requests) == 1
    assert len(provider.structured_output_requests) == 1
    assert extraction.state_changes == ()
    assert extraction.tool_diagnostics["shape_switch"] == "structured_output"


def test_tool_calling_state_extractor_keeps_error_when_structured_route_also_fails(
    repositories: PersistenceRepositories,
) -> None:
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-tools",
        display_name="Fake Tools",
        capabilities=["tool_calling", "structured_output"],
    )
    provider = ShapeFailingToolCallProvider()
    extractor = ToolCallingProviderStateExtractor(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
        repositories=repositories,
        providers={"fake": cast(Any, provider)},
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            extractor.extract(
                _state_request(
                    save_id=save.id,
                    messages=(narrator_message,),
                    repositories=repositories,
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.MODEL_NOT_FOUND
    assert exc_info.value.fallback_attempted is True
    assert exc_info.value.fallback_provider == "fake"
    assert len(provider.structured_output_requests) == 1


def test_tool_calling_state_extractor_recovers_when_tool_fallback_also_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-tools",
        display_name="Fake Tools",
        capabilities=["tool_calling", "structured_output"],
    )
    _configure_state_tool_fallback(repositories)
    primary = ShapeSwitchToolCallProvider(
        structured_data=_grounded_structured_response(
            player_message,
            narrator_message,
        )
    )
    fallback = FailingToolCallFallbackProvider(
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "fallback model not found",
            status_code=404,
        )
    )
    extractor = ToolCallingProviderStateExtractor(
        provider=primary,
        provider_name="fake",
        model_id="fake-tools",
        repositories=repositories,
        providers={
            "fake": cast(Any, primary),
            "fallback": cast(Any, fallback),
        },
    )

    extraction = asyncio.run(
        extractor.extract(
            _state_request(
                save_id=save.id,
                messages=(player_message, narrator_message),
                repositories=repositories,
            )
        )
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert len(primary.structured_output_requests) == 1
    assert [change.key for change in extraction.state_changes] == ["scene.location"]
    assert extraction.tool_diagnostics["shape_switch"] == "structured_output"


def test_tool_calling_state_extractor_recovers_when_tool_fallback_model_missing(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-tools",
        display_name="Fake Tools",
        capabilities=["tool_calling", "structured_output"],
    )
    _configure_state_tool_fallback(repositories)
    primary = RateLimitedShapeSwitchToolCallProvider(
        structured_data=_grounded_structured_response(
            player_message,
            narrator_message,
        )
    )
    fallback = FailingToolCallFallbackProvider(
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "fallback model missing",
            status_code=404,
        )
    )
    extractor = ToolCallingProviderStateExtractor(
        provider=primary,
        provider_name="fake",
        model_id="fake-tools",
        repositories=repositories,
        providers={
            "fake": cast(Any, primary),
            "fallback": cast(Any, fallback),
        },
    )

    extraction = asyncio.run(
        extractor.extract(
            _state_request(
                save_id=save.id,
                messages=(player_message, narrator_message),
                repositories=repositories,
            )
        )
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert len(primary.structured_output_requests) == 1
    assert [change.key for change in extraction.state_changes] == ["scene.location"]
    assert extraction.tool_diagnostics["shape_switch"] == "structured_output"


def test_tool_calling_state_extractor_keeps_fallback_result_when_fallback_succeeds(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-tools",
        display_name="Fake Tools",
        capabilities=["tool_calling", "structured_output"],
    )
    _configure_state_tool_fallback(repositories)
    primary = ShapeSwitchToolCallProvider()
    fallback = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="state-call",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "scene.location",
                            "value_patch": {"name": "Beacon lens", "danger": "high"},
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "lens flares",
                        }
                    ),
                ),
            )
        ]
    )
    extractor = ToolCallingProviderStateExtractor(
        provider=primary,
        provider_name="fake",
        model_id="fake-tools",
        repositories=repositories,
        providers={
            "fake": cast(Any, primary),
            "fallback": cast(Any, fallback),
        },
    )

    extraction = asyncio.run(
        extractor.extract(
            _state_request(
                save_id=save.id,
                messages=(player_message, narrator_message),
                repositories=repositories,
            )
        )
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert primary.structured_output_requests == []
    assert [change.key for change in extraction.state_changes] == ["scene.location"]


def test_tool_calling_state_patch_only_changes_supported_keys(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Lower stair", "danger": "low"},
        category="scene",
        source_message_id=player_message.id,
    )
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="state-call",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "scene.location",
                            "value_patch": {"danger": "high"},
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "lens flares",
                        }
                    ),
                ),
            )
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ),
    )

    asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert repositories.list_world_state(save.id)[0].value == {
        "name": "Lower stair",
        "danger": "high",
    }


def test_tool_calling_state_patch_strips_unchanged_fields(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Lower stair", "danger": "low"},
        category="scene",
        source_message_id=player_message.id,
    )
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="state-call",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "scene.location",
                            "value_patch": {
                                "name": "Lower stair",
                                "danger": "high",
                            },
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "lens flares",
                        }
                    ),
                ),
            )
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ),
    )

    asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert len(provider.tool_call_requests) == 1
    assert repositories.list_world_state(save.id)[0].value == {
        "name": "Lower stair",
        "danger": "high",
    }


def test_tool_calling_state_conflict_does_not_apply_reconciliation_patch(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="npc.warden.elian",
        value={"status": "on duty"},
        category="npc",
        source_message_id=player_message.id,
    )
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="conflict-call",
                    name="flag_state_conflict",
                    arguments_json=json.dumps(
                        {
                            "key": "npc.warden.elian",
                            "source_message_id": narrator_message.id,
                            "new_evidence": "empty post",
                            "current_value": {"status": "on duty"},
                            "proposed_value": {"status": "missing"},
                        }
                    ),
                ),
                ProviderToolCall(
                    id="invented-reconciliation",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "npc.warden.elian",
                            "value_patch": {"status": "missing"},
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "empty post",
                        }
                    ),
                ),
            ),
            (),
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert applied.world_state == ()
    assert repositories.list_world_state(save.id)[0].value == {"status": "on duty"}
    jobs = _jobs(repositories, save.id, "state_extraction")
    assert jobs[-1]["result"]["conflict_count"] == 1
    rejected = jobs[-1]["result"]["tool_diagnostics"]["rejected_calls"]
    assert rejected[0]["name"] == "patch_world_state"
    assert "without applying a patch" in rejected[0]["error"]


def test_tool_calling_state_conflict_supersedes_previous_retry_patch(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="npc.warden.elian",
        value={"status": "on duty"},
        category="npc",
        source_message_id=player_message.id,
    )
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="early-patch",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "npc.warden.elian",
                            "value_patch": {"status": "missing"},
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "empty post",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="bad-memory",
                    name="record_memory_fact",
                    arguments_json='{"source_message_id":',
                ),
            ),
            (
                ProviderToolCall(
                    id="later-conflict",
                    name="flag_state_conflict",
                    arguments_json=json.dumps(
                        {
                            "key": "npc.warden.elian",
                            "source_message_id": narrator_message.id,
                            "new_evidence": "empty post",
                            "current_value": {"status": "on duty"},
                            "proposed_value": {"status": "missing"},
                        }
                    ),
                ),
            ),
        ]
    )
    service = StateService(
        repositories=repositories,
        extractor=ToolCallingProviderStateExtractor(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ),
    )

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert applied.world_state == ()
    assert repositories.list_world_state(save.id)[0].value == {"status": "on duty"}
    diagnostics = _jobs(repositories, save.id, "state_extraction")[-1]["result"][
        "tool_diagnostics"
    ]
    assert any(
        call["id"] == "early-patch" and "superseded" in call["error"]
        for call in diagnostics["rejected_calls"]
    )


def test_tool_calling_state_prompt_uses_unknown_and_empty_arrays_for_traps(
    repositories: PersistenceRepositories,
) -> None:
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(responses=[()])
    extractor = ToolCallingProviderStateExtractor(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    extraction = asyncio.run(
        extractor.extract(
            _state_request(
                save_id=save.id,
                messages=(narrator_message,),
                repositories=repositories,
            )
        )
    )

    assert extraction.state_changes == ()
    prompt = "\n".join(
        message.body.casefold() for message in provider.tool_call_requests[0].messages
    )
    assert "unknown or empty arrays" in prompt
    assert "instead of inventing details" in prompt


def test_extract_and_apply_turn_skips_empty_world_state_upserts(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="character.NPC_A.voice.banter_tone",
                    value={},
                    category="character",
                    confidence=0.76,
                    source_message_id=narrator_message.id,
                ),
                ExtractedStateChange(
                    operation="upsert",
                    key="scene.location",
                    value={"name": "Beacon lens"},
                    category="scene",
                    confidence=0.84,
                    source_message_id=narrator_message.id,
                ),
            ),
            memories=(),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert [
        (state.key, state.value) for state in repositories.list_world_state(save.id)
    ] == [("scene.location", {"name": "Beacon lens"})]
    assert [state.key for state in applied.world_state] == ["scene.location"]
    assert [
        (change["operation"], change["state_key"])
        for change in _state_changes(repositories, save.id)
    ] == [("upsert", "scene.location")]


def test_extract_and_apply_turn_skips_aggregate_open_threads_when_threads_exist(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.add_active_thread(
        save_id=save.id,
        title="Dinner promise",
        description="Mara still owes Ilyra dinner after the beacon is safe.",
        status="active",
        priority=4,
        visibility="public",
        source_message_id=player_message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="interaction.open_threads",
        value={"old": "This aggregate should be archived."},
        category="open_threads",
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="interaction.open_threads",
                    value={
                        "items": [
                            "Mara still owes Ilyra dinner after the beacon is safe."
                        ]
                    },
                    category="interaction",
                    confidence=0.7,
                    source_message_id=narrator_message.id,
                ),
                ExtractedStateChange(
                    operation="upsert",
                    key="scene.mood",
                    value={"value": "urgent"},
                    category="scene",
                    confidence=0.83,
                    source_message_id=narrator_message.id,
                ),
            ),
            memories=(),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert [
        (state.key, state.value) for state in repositories.list_world_state(save.id)
    ] == [("scene.mood", {"value": "urgent"})]
    assert [
        (change["operation"], change["state_key"])
        for change in _state_changes(repositories, save.id)
    ] == [("upsert", "scene.mood")]


def test_scene_scoped_state_records_meaningful_existing_row_update(
    repositories: PersistenceRepositories,
) -> None:
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="character.ivy.current_emotional_state",
        value={"mood": "guarded"},
        category="scene",
        confidence=0.8,
        source_message_id=narrator_message.id,
    )
    service = StateService(
        repositories=repositories,
        extractor=StructuredFakeExtractor(
            StateExtraction(
                state_changes=(
                    ExtractedStateChange(
                        operation="upsert",
                        key="character.ivy.current_emotional_state",
                        value={"mood": "softening"},
                        category="scene",
                        confidence=0.85,
                        source_message_id=narrator_message.id,
                    ),
                    ExtractedStateChange(
                        operation="upsert",
                        key="character.ivy.hand_injury",
                        value={"status": "splinted"},
                        category="character",
                        confidence=0.9,
                        source_message_id=narrator_message.id,
                        persistence_scope="durable",
                    ),
                    ExtractedStateChange(
                        operation="upsert",
                        key="character.ivy.micro_expression",
                        value={"note": "brief smile"},
                        category="ephemeral",
                        confidence=0.7,
                        source_message_id=narrator_message.id,
                    ),
                )
            )
        ),
    )

    asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(narrator_message.id,),
        )
    )

    state = {row.key: row.value for row in repositories.list_world_state(save.id)}
    assert state == {
        "character.ivy.current_emotional_state": {"mood": "softening"},
        "character.ivy.hand_injury": {"status": "splinted"},
    }
    assert [
        (change["operation"], change["state_key"])
        for change in _state_changes(repositories, save.id)
    ] == [
        ("upsert", "character.ivy.current_emotional_state"),
        ("upsert", "character.ivy.hand_injury"),
    ]


def test_scene_scoped_state_skips_unchanged_existing_row_audit(
    repositories: PersistenceRepositories,
) -> None:
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="character.ivy.current_emotional_state",
        value={"mood": "guarded"},
        category="scene",
        confidence=0.8,
        source_message_id=narrator_message.id,
    )
    service = StateService(
        repositories=repositories,
        extractor=StructuredFakeExtractor(
            StateExtraction(
                state_changes=(
                    ExtractedStateChange(
                        operation="upsert",
                        key="character.ivy.current_emotional_state",
                        value={"mood": "guarded"},
                        category="scene",
                        confidence=0.85,
                        source_message_id=narrator_message.id,
                    ),
                )
            )
        ),
    )

    asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(narrator_message.id,),
        )
    )

    assert _state_changes(repositories, save.id) == []


def test_failed_state_extraction_preserves_completed_turn_messages(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    service = StateService(
        repositories=repositories,
        extractor=FailingFakeExtractor(RuntimeError("extractor unavailable")),
    )

    with pytest.raises(RuntimeError, match="extractor unavailable"):
        asyncio.run(
            service.extract_and_apply_turn(
                save_id=save.id,
                source_message_ids=(player_message.id, narrator_message.id),
            )
        )

    messages = repositories.list_messages(save.id)
    assert messages == [player_message, narrator_message]
    assert repositories.list_world_state(save.id) == []
    assert repositories.list_memories(save.id) == []
    failed_jobs = _jobs(repositories, save.id, "state_extraction")
    assert failed_jobs[-1]["status"] == "failed"
    assert failed_jobs[-1]["error"] == "extractor unavailable"


def test_invalid_later_state_change_fails_without_partial_state_or_memory_mutation(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Lower stair", "danger": "low"},
        category="scene",
        confidence=0.4,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="scene.location",
                    value={"name": "Beacon lens", "danger": "high"},
                    category="scene",
                    confidence=0.92,
                    source_message_id=narrator_message.id,
                ),
                ExtractedStateChange(
                    operation="teleport",
                    key="npc.warden.elian",
                    value={"name": "Elian", "status": "missing"},
                    category="npc",
                    confidence=0.81,
                    source_message_id=narrator_message.id,
                ),
            ),
            memories=(
                ExtractedMemory(
                    body="Mara promised to relight the Ashfall beacon.",
                    tags=("promise", "beacon"),
                    importance=0.74,
                    source_message_id=player_message.id,
                ),
            ),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    with pytest.raises(ValueError):
        asyncio.run(
            service.extract_and_apply_turn(
                save_id=save.id,
                source_message_ids=(player_message.id, narrator_message.id),
            )
        )

    assert [
        (state.key, state.value)
        for state in repositories.list_world_state(save.id)
    ] == [
        ("scene.location", {"name": "Lower stair", "danger": "low"}),
    ]
    assert repositories.list_memories(save.id) == []
    assert _state_changes(repositories, save.id) == []
    failed_jobs = _jobs(repositories, save.id, "state_extraction")
    assert failed_jobs[-1]["status"] == "failed"


def test_later_non_json_state_value_fails_without_partial_state_or_memory_mutation(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Lower stair", "danger": "low"},
        category="scene",
        confidence=0.4,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="scene.location",
                    value={"name": "Beacon lens", "danger": "high"},
                    category="scene",
                    confidence=0.92,
                    source_message_id=narrator_message.id,
                ),
                ExtractedStateChange(
                    operation="upsert",
                    key="npc.warden.elian",
                    value={"name": "Elian", "aliases": {"Warden"}},
                    category="npc",
                    confidence=0.81,
                    source_message_id=narrator_message.id,
                ),
            ),
            memories=(
                ExtractedMemory(
                    body="Mara promised to relight the Ashfall beacon.",
                    tags=("promise", "beacon"),
                    importance=0.74,
                    source_message_id=player_message.id,
                ),
            ),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    with pytest.raises(TypeError, match="not JSON serializable"):
        asyncio.run(
            service.extract_and_apply_turn(
                save_id=save.id,
                source_message_ids=(player_message.id, narrator_message.id),
            )
        )

    assert [
        (state.key, state.value)
        for state in repositories.list_world_state(save.id)
    ] == [
        ("scene.location", {"name": "Lower stair", "danger": "low"}),
    ]
    assert repositories.list_memories(save.id) == []
    assert _state_changes(repositories, save.id) == []
    failed_jobs = _jobs(repositories, save.id, "state_extraction")
    assert failed_jobs[-1]["status"] == "failed"
    assert "not JSON serializable" in failed_jobs[-1]["error"]


def test_unknown_memory_source_message_id_fails_without_partial_mutation(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Lower stair", "danger": "low"},
        category="scene",
        confidence=0.4,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="scene.location",
                    value={"name": "Beacon lens", "danger": "high"},
                    category="scene",
                    confidence=0.92,
                    source_message_id=narrator_message.id,
                ),
            ),
            memories=(
                ExtractedMemory(
                    body="Mara promised to relight the Ashfall beacon.",
                    tags=("promise", "beacon"),
                    importance=0.74,
                    source_message_id=player_message.id,
                ),
                ExtractedMemory(
                    body="This memory points at a message that does not exist.",
                    tags=("invalid",),
                    importance=0.2,
                    source_message_id="missing-message-id",
                ),
            ),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    with pytest.raises(
        ValueError,
        match="Unknown memory source_message_id: missing-message-id",
    ):
        asyncio.run(
            service.extract_and_apply_turn(
                save_id=save.id,
                source_message_ids=(player_message.id, narrator_message.id),
            )
        )

    assert [
        (state.key, state.value)
        for state in repositories.list_world_state(save.id)
    ] == [
        ("scene.location", {"name": "Lower stair", "danger": "low"}),
    ]
    assert repositories.list_memories(save.id) == []
    assert _state_changes(repositories, save.id) == []
    failed_jobs = _jobs(repositories, save.id, "state_extraction")
    assert failed_jobs[-1]["status"] == "failed"
    assert failed_jobs[-1]["error"] == (
        "Unknown memory source_message_id: missing-message-id"
    )


@pytest.mark.parametrize("operation", ["delete", "remove"])
def test_delete_or_remove_state_change_archives_world_state_and_records_before(
    operation: str,
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    existing_state = repositories.upsert_world_state(
        save_id=save.id,
        key="npc.warden.elian",
        value={"name": "Elian", "status": "missing"},
        category="npc",
        confidence=0.81,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation=operation,
                    key="npc.warden.elian",
                    value={},
                    category="npc",
                    confidence=1.0,
                    source_message_id=narrator_message.id,
                ),
            ),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert repositories.list_world_state(save.id) == []
    archived_row = repositories.connection.execute(
        """
        SELECT id, value_json, archived_at
        FROM world_state
        WHERE save_id = ? AND key = ?
        """,
        (save.id, "npc.warden.elian"),
    ).fetchone()
    assert archived_row is not None
    assert archived_row["id"] == existing_state.id
    assert json.loads(archived_row["value_json"]) == {
        "name": "Elian",
        "status": "missing",
    }
    assert archived_row["archived_at"] is not None
    changes = _state_changes(repositories, save.id)
    assert [(change["operation"], change["state_key"]) for change in changes] == [
        (operation, "npc.warden.elian"),
    ]
    assert json.loads(changes[0]["before_json"]) == {
        "name": "Elian",
        "status": "missing",
    }
    assert changes[0]["after_json"] is None


def test_state_extraction_cannot_delete_policy_owned_loop_current(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    loop_current = {
        "version": 1,
        "iteration": 2,
        "summary": "The loop continues.",
        "baseline_time": {"day_index": 0, "phase": "morning"},
    }
    repositories.upsert_world_state(
        save_id=save.id,
        key="loop.current",
        value=loop_current,
        category="loop_status",
        source_message_id=player_message.id,
    )
    service = StateService(
        repositories=repositories,
        extractor=StructuredFakeExtractor(
            StateExtraction(
                state_changes=(
                    ExtractedStateChange(
                        operation="delete",
                        key="loop.current",
                        value={},
                        category="loop_status",
                        confidence=1.0,
                        source_message_id=narrator_message.id,
                    ),
                ),
            )
        ),
    )

    result = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    current = next(
        state
        for state in repositories.list_world_state(save.id)
        if state.key == "loop.current"
    )
    assert current.value == loop_current
    assert result.suppressed_state_change_count == 1


def test_extract_and_apply_turn_queues_memory_and_state_when_confirmation_enabled(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.set_app_setting("manual_confirmation_memories_enabled", True)
    repositories.set_app_setting("manual_confirmation_state_changes_enabled", True)
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Lower stair", "danger": "low"},
        category="scene",
        confidence=0.4,
        source_message_id=player_message.id,
    )
    extractor = StructuredFakeExtractor(
        StateExtraction(
            state_changes=(
                ExtractedStateChange(
                    operation="upsert",
                    key="scene.location",
                    value={"name": "Beacon lens", "danger": "high"},
                    category="scene",
                    confidence=0.92,
                    source_message_id=narrator_message.id,
                ),
            ),
            memories=(
                ExtractedMemory(
                    body="Mara promised to relight the Ashfall beacon.",
                    tags=("promise", "beacon"),
                    importance=0.74,
                    source_message_id=player_message.id,
                ),
            ),
        )
    )
    service = StateService(repositories=repositories, extractor=extractor)

    applied = asyncio.run(
        service.extract_and_apply_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert applied.memories == ()
    assert applied.world_state == ()
    assert applied.state_changes == ()
    assert repositories.list_memories(save.id) == []
    assert [
        (state.key, state.value)
        for state in repositories.list_world_state(save.id)
    ] == [
        ("scene.location", {"name": "Lower stair", "danger": "low"}),
    ]
    assert _state_changes(repositories, save.id) == []

    suggestions = repositories.list_context_update_suggestions(save.id)
    assert [(item.entity_type, item.update_type) for item in suggestions] == [
        ("world_state", "upsert"),
        ("memory", "create"),
    ]
    state_suggestion = suggestions[0]
    assert state_suggestion.field_path == "scene.location"
    assert state_suggestion.proposed_value == {
        "operation": "upsert",
        "key": "scene.location",
        "value": {"name": "Beacon lens", "danger": "high"},
        "category": "scene",
        "confidence": 0.92,
        "source_message_id": narrator_message.id,
    }
    memory_suggestion = suggestions[1]
    assert memory_suggestion.field_path == "*"
    assert memory_suggestion.proposed_value == {
        "body": "Mara promised to relight the Ashfall beacon.",
        "tags": ["promise", "beacon"],
        "importance": 0.74,
        "source_message_id": player_message.id,
    }
    audit_rows = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type) for row in audit_rows] == [
        ("queued", "world_state"),
        ("queued", "memory"),
    ]


def _save_with_completed_turn(
    repositories: PersistenceRepositories,
) -> tuple[Any, MessageRecord, MessageRecord]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I promise to relight the beacon.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lens flares and reveals Warden Elian's empty post.",
        provider="fake",
        model="fake-chat",
    )
    return save, player_message, narrator_message


def _grounded_structured_response(
    player_message: MessageRecord,
    narrator_message: MessageRecord,
) -> dict[str, Any]:
    return {
        "state_changes": [
            {
                "operation": "upsert",
                "key": "scene.location",
                "value": {"name": "Beacon lens"},
                "category": "scene",
                "confidence": 0.87,
                "source_message_id": narrator_message.id,
                "evidence_quote": "lens flares",
            }
        ],
        "memories": [
            {
                "body": "Mara promised to relight the beacon.",
                "tags": ["promise"],
                "importance": 0.74,
                "source_message_id": player_message.id,
                "evidence_quote": "promise to relight",
            }
        ],
        "conflicts": [
            {
                "key": "npc.warden.elian",
                "source_message_id": narrator_message.id,
                "new_evidence": "empty post",
                "current_value": {"status": "on duty"},
                "proposed_value": {"status": "missing"},
            }
        ],
    }


def _state_changes(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[sqlite3.Row]:
    return list(
        repositories.connection.execute(
            """
            SELECT operation, state_key, source_message_id, before_json, after_json
            FROM state_changes
            WHERE save_id = ?
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )
    )


def _state_request(
    *,
    save_id: str,
    messages: tuple[MessageRecord, ...],
    repositories: PersistenceRepositories,
) -> StateExtractionRequest:
    return StateExtractionRequest(
        save_id=save_id,
        messages=messages,
        current_state=tuple(repositories.list_world_state(save_id)),
    )


def _configure_state_tool_fallback(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting("tool_call_fallback_enabled", True)
    repositories.set_model_preference(
        task="tool_call_fallback",
        provider="fallback",
        model_id="fallback-tools",
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-tools",
        display_name="Fallback Tools",
        capabilities=["tool_calling"],
    )


def _schema_source_enum(
    schema_properties: dict[str, Any],
    collection_name: str,
) -> list[str]:
    collection = cast(dict[str, Any], schema_properties[collection_name])
    items = cast(dict[str, Any], collection["items"])
    item_properties = cast(dict[str, Any], items["properties"])
    source = cast(dict[str, Any], item_properties["source_message_id"])
    return cast(list[str], source["enum"])


def _schema_required_fields(
    schema_properties: dict[str, Any],
    collection_name: str,
) -> list[str]:
    collection = cast(dict[str, Any], schema_properties[collection_name])
    items = cast(dict[str, Any], collection["items"])
    return cast(list[str], items["required"])


def _jobs(
    repositories: PersistenceRepositories,
    save_id: str,
    job_type: str,
) -> list[dict[str, Any]]:
    rows = repositories.connection.execute(
        """
        SELECT status, error, result_json
        FROM jobs
        WHERE save_id = ? AND type = ?
        ORDER BY created_at, rowid
        """,
        (save_id, job_type),
    ).fetchall()
    return [
        {
            "status": row["status"],
            "error": row["error"],
            "result": (
                json.loads(row["result_json"])
                if row["result_json"] is not None
                else None
            ),
        }
        for row in rows
    ]
