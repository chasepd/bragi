from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.scenario_evolution_service import (
    ScenarioEvolution,
    ScenarioEvolutionRequest,
    ScenarioEvolutionService,
    ScenarioSectionUpdate,
    StructuredProviderScenarioEvolver,
    ToolCallingProviderScenarioEvolver,
    _evolvable_sections,
    _scenario_evolution_instruction,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        repositories.save_provider_model(
            provider="fake",
            model_id="fake-structured",
            display_name="Fake Structured",
            capabilities=["structured_output"],
        )
        yield repositories


class FakeStructuredProvider:
    provider_name = "fake"

    def __init__(
        self,
        data: dict[str, object],
        *,
        raw_metadata: dict[str, object] | None = None,
    ) -> None:
        self.data = data
        self.raw_metadata = raw_metadata or {}
        self.requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.requests.append(request)
        return StructuredOutputResponse(
            data=self.data,
            provider=request.provider,
            model_id=request.model_id,
            raw_metadata=self.raw_metadata,
        )


class FakeToolProvider:
    provider_name = "fake"

    def __init__(self, responses: list[tuple[ProviderToolCall, ...]]) -> None:
        self.responses = responses
        self.tool_requests: list[ToolCallRequest] = []

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected tool-call request")
        return ToolCallResponse(
            tool_calls=self.responses.pop(0),
            body="",
            provider=request.provider,
            model_id=request.model_id,
        )


class ShapeSwitchToolProvider(FakeToolProvider):
    """Tool-capable evolver provider whose tool calls 404 but structured works."""

    def __init__(
        self,
        *,
        structured_data: dict[str, object] | None = None,
    ) -> None:
        super().__init__(responses=[])
        self.structured_data = structured_data or {}
        self.structured_requests: list[StructuredOutputRequest] = []

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model not found",
            status_code=404,
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        return StructuredOutputResponse(
            data=self.structured_data,
            provider=request.provider,
            model_id=request.model_id,
        )


class ShapeFailingToolProvider(ShapeSwitchToolProvider):
    """Tool-capable evolver provider whose tool and structured calls both 404."""

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model not found",
            status_code=404,
        )


class RateLimitedShapeSwitchToolProvider(ShapeSwitchToolProvider):
    """Tool-capable evolver provider that rate-limits but structured works."""

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.RATE_LIMITED,
            "rate limited",
            status_code=429,
        )


class FailingScenarioEvolutionFallbackProvider(FakeToolProvider):
    provider_name = "fallback"

    def __init__(self, *, error: ProviderError) -> None:
        super().__init__(responses=[])
        self.error = error

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_requests.append(request)
        raise self.error


def _create_full_roleplay_save(
    repositories: PersistenceRepositories,
) -> tuple[str, str, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A keep waits under an ash storm.",
        player_role="Signal warden",
        content={
            "current_scene": "The warden stands at the lower gate.",
            "lore": "The red lens is hidden in the tower.",
            "tone": "Tense but hopeful",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The warden climbs to the beacon gallery.",
    )
    return save.id, scenario.id, message.id


def _configure_scenario_evolution_tool_fallback(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-tools",
        display_name="Fake Tools",
        capabilities=["tool_calling", "structured_output"],
    )
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


def _create_first_contact_save(
    repositories: PersistenceRepositories,
) -> tuple[str, str, str]:
    scenario = repositories.create_scenario(
        type="first_contact_exploration",
        title="Songs Under Europa",
        premise="A survey crew finds patterned signals under the ice.",
        player_role="Mission linguist",
        content={
            "mission_profile": "Survey the hidden ocean.",
            "ship_or_base_status": "Habitat heat is stable for 42 hours.",
            "exploration_target": "A black-water cavern beneath the ice.",
            "unknown_intelligence": "An unseen singer answers sonar.",
            "knowledge_state": "Observed songs; unknown intent.",
            "translation_progress": "Three descending pulses may mean open water.",
            "discoveries_and_samples": "Metallic spores remain quarantined.",
            "hazards_and_escalation": "Thermal fissures are spreading.",
            "opening_message": "Blue light pulses beneath the ice.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Europa Contact")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body=(
            "The singer repeats three descending pulses while blue lights mark "
            "a safe arch. A thermal fissure splits the return tunnel."
        ),
    )
    return save.id, scenario.id, message.id


def test_mystery_evolution_instruction_and_sections_target_case_progress() -> None:
    instruction = _scenario_evolution_instruction("investigation_mystery")
    content = json.dumps(
        {
            "title": "Broken Hours",
            "case_facts": "Curator Elian vanished during a gala.",
            "clues": "Watch log gap remains undiscovered.",
            "timeline": "Alarm at 9:21.",
            "red_herrings": "Bloody glove from mannequin repair.",
            "hidden_truth": "Sera hid the smuggling ledger.",
            "case_status": "Unresolved.",
            "factions": "Museum board is stonewalling records access.",
            "opening_message": "The gallery unlocks.",
        }
    )

    sections = _evolvable_sections(
        scenario_type="investigation_mystery",
        content=content,
    )

    lowered = instruction.casefold()
    assert "discovered clues" in lowered
    assert "known public timeline" in lowered
    assert "case status" in lowered
    assert "do not rewrite hidden truth" in lowered
    assert "hidden_truth" in sections
    assert "clues" in sections
    assert "timeline" in sections
    assert "case_status" in sections
    assert "factions" in sections
    assert "title" not in sections
    assert "opening_message" not in sections


def test_retired_character_interaction_type_has_no_evolution_specialization() -> None:
    instruction = _scenario_evolution_instruction("character_interaction")
    sections = _evolvable_sections(
        scenario_type="character_interaction",
        content=json.dumps({"character_voice": "Soft questions."}),
    )

    assert "for character interactions" not in instruction.casefold()
    assert "character_voice" not in sections
    assert "current_scene" in sections
    assert "characters" not in sections


def test_first_contact_evolution_instruction_targets_discovery_progress() -> None:
    instruction = _scenario_evolution_instruction("first_contact_exploration")
    content = json.dumps(
        {
            "title": "Songs Under Europa",
            "mission_profile": "Survey the hidden ocean.",
            "ship_or_base_status": "Habitat heat is stable for 42 hours.",
            "exploration_target": "A black-water cavern beneath the ice.",
            "unknown_intelligence": "An unseen singer answers sonar.",
            "knowledge_state": "Observed songs; unknown intent.",
            "translation_progress": "Three descending pulses may mean open water.",
            "discoveries_and_samples": "Metallic spores remain quarantined.",
            "hazards_and_escalation": "Thermal fissures are spreading.",
            "opening_message": "Blue light pulses beneath the ice.",
        }
    )

    sections = _evolvable_sections(
        scenario_type="first_contact_exploration",
        content=content,
    )

    lowered = instruction.casefold()
    assert "observed facts" in lowered
    assert "hypotheses" in lowered
    assert "translation progress" in lowered
    assert "hazards" in lowered
    assert "premature exposition" in lowered
    assert "translation_progress" in sections
    assert "knowledge_state" in sections
    assert "hazards_and_escalation" in sections
    assert "current_scene" in sections
    assert "title" not in sections
    assert "opening_message" not in sections


def test_survival_evolution_targets_expedition_progress() -> None:
    instruction = _scenario_evolution_instruction("survival_expedition")
    content = json.dumps(
        {
            "title": "Whiteout Pass",
            "expedition_goal": "Reach Northwatch.",
            "route_options": "Cliff road or forest basin.",
            "resource_inventory": "Food: 2 days.",
            "travel_progress": "18 of 80 miles.",
            "opening_message": "Snow closes in.",
        }
    )

    sections = _evolvable_sections(
        scenario_type="survival_expedition",
        content=content,
    )

    lowered = instruction.casefold()
    assert "expedition goal" in lowered
    assert "route options" in lowered
    assert "resources" in lowered
    assert "camp status" in lowered
    assert "travel progress" in lowered
    assert "expedition_goal" in sections
    assert "resource_inventory" in sections
    assert "travel_progress" in sections
    assert "hazards_and_events" in sections
    assert "title" not in sections
    assert "opening_message" not in sections


def test_time_loop_evolution_targets_loop_boundaries() -> None:
    instruction = _scenario_evolution_instruction("time_loop")
    content = json.dumps(
        {
            "title": "Bellwether Day",
            "loop_premise": "The festival day repeats.",
            "persistent_knowledge": "The tower code persists for the player.",
            "npc_memory_rules": "NPCs reset to dawn memories.",
            "current_loop_state": "Loop 1, dawn phase.",
            "opening_message": "The same bell rings dawn again.",
        }
    )

    sections = _evolvable_sections(
        scenario_type="time_loop",
        content=content,
    )

    lowered = instruction.casefold()
    assert "loop rules" in lowered
    assert "persistent player/meta knowledge" in lowered
    assert "npc memory rules" in lowered
    assert "resettable world state separate from persistent knowledge" in lowered
    assert "persistent_knowledge" in sections
    assert "npc_memory_rules" in sections
    assert "current_loop_state" in sections
    assert "title" not in sections
    assert "opening_message" not in sections


def test_political_intrigue_evolution_targets_social_consequences() -> None:
    instruction = _scenario_evolution_instruction("political_intrigue")
    content = json.dumps(
        {
            "title": "Council of Ash",
            "political_arena": "The harbor council.",
            "political_factions": "Guilds and Old Families.",
            "reputation_and_standing": "Mara is trusted by reformers.",
            "obligations_and_favors": "Orro owes Mara one endorsement.",
            "political_pressure": "Midnight vote proceeds unless delayed.",
            "public_private_knowledge": "Only Mara knows Orro owes the favor.",
            "opening_message": "The council bell rings.",
        }
    )

    sections = _evolvable_sections(
        scenario_type="political_intrigue",
        content=content,
    )

    lowered = instruction.casefold()
    assert "faction positions" in lowered
    assert "reputation or standing" in lowered
    assert "obligations" in lowered
    assert "timed pressure" in lowered
    assert "public/private knowledge boundaries" in lowered
    assert "political_factions" in sections
    assert "reputation_and_standing" in sections
    assert "obligations_and_favors" in sections
    assert "political_pressure" in sections
    assert "public_private_knowledge" in sections
    assert "title" not in sections
    assert "opening_message" not in sections


def test_structured_evolver_updates_first_contact_translation_and_hazards(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_first_contact_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        provider = FakeStructuredProvider(
            {
                "content": {
                    "translation_progress": (
                        "Confirmed: three descending pulses identify a safe "
                        "passage through the ice."
                    ),
                    "hazards_and_escalation": (
                        "Thermal fissures have split the return tunnel, shortening "
                        "the crew's rescue window."
                    ),
                },
                "reason": (
                    "The latest scene confirmed one signal meaning and escalated "
                    "the environmental hazard."
                ),
                "source_message_id": message_id,
            }
        )

        evolution = await StructuredProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-structured",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        schema = cast(dict[str, Any], provider.requests[0].schema["properties"])
        updates_schema = cast(dict[str, Any], schema["updates"])
        update_items = cast(dict[str, Any], updates_schema["items"])
        update_properties = cast(dict[str, Any], update_items["properties"])
        section_schema = cast(dict[str, Any], update_properties["section_id"])
        assert "translation_progress" in section_schema["enum"]
        assert "hazards_and_escalation" in section_schema["enum"]
        assert "opening_message" not in section_schema["enum"]
        assert evolution == ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="translation_progress",
                    text=(
                        "Confirmed: three descending pulses identify a safe "
                        "passage through the ice."
                    ),
                    reason=(
                        "The latest scene confirmed one signal meaning and "
                        "escalated the environmental hazard."
                    ),
                    source_message_id=message_id,
                ),
                ScenarioSectionUpdate(
                    section_id="hazards_and_escalation",
                    text=(
                        "Thermal fissures have split the return tunnel, shortening "
                        "the crew's rescue window."
                    ),
                    reason=(
                        "The latest scene confirmed one signal meaning and "
                        "escalated the environmental hazard."
                    ),
                    source_message_id=message_id,
                ),
            )
        )

    asyncio.run(run())


def test_structured_evolver_builds_schema_messages_and_parses_content_updates(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        provider = FakeStructuredProvider(
            {
                "content": {"current_scene": "The beacon gallery hums."},
                "reason": "The location changed during play.",
                "source_message_id": message_id,
            }
        )

        evolution = await StructuredProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-structured",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        request = provider.requests[0]
        schema = cast(dict[str, Any], request.schema["properties"])
        change_type_schema = cast(dict[str, Any], schema["change_type"])
        updates_schema = cast(dict[str, Any], schema["updates"])
        update_items = cast(dict[str, Any], updates_schema["items"])
        update_properties = cast(dict[str, Any], update_items["properties"])
        section_schema = cast(dict[str, Any], update_properties["section_id"])
        source_schema = cast(dict[str, Any], schema["source_message_id"])
        assert request.schema_name == "scenario_evolution"
        assert request.max_output_tokens == 2048
        assert change_type_schema["enum"] == [
            "phase_shift",
            "no_phase_shift",
            "turn_level_change",
        ]
        assert section_schema["enum"] == [
            "current_scene",
            "factions",
            "locations",
            "lore",
            "worldbuilding",
        ]
        assert source_schema["enum"] == [message_id, None]
        assert "turn_level_change" in request.messages[0].body
        assert "no_phase_shift" in request.messages[0].body
        assert (
            "Evolvable sections: current_scene, factions, locations, lore, "
            "worldbuilding"
        ) in request.messages[1].body
        assert "tone" in request.messages[1].body
        assert evolution == ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The location changed during play.",
                    source_message_id=message_id,
                ),
            )
        )

    asyncio.run(run())


def test_tool_calling_evolver_parses_section_updates(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        provider = FakeToolProvider(
            [
                (
                    ProviderToolCall(
                        id="call-update",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "current_scene",
                                "text": "The beacon gallery hums.",
                                "reason": "The party moved into a new durable scene.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                )
            ]
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert evolution == ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The party moved into a new durable scene.",
                    source_message_id=message_id,
                ),
            )
        )
        assert [tool.name for tool in provider.tool_requests[0].tools] == [
            "update_scenario_section",
            "skip_scenario_evolution",
        ]

    asyncio.run(run())


def test_tool_calling_evolver_switches_to_structured_route_on_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        repositories.save_provider_model(
            provider="fake",
            model_id="fake-tools",
            display_name="Fake Tools",
            capabilities=["tool_calling", "structured_output"],
        )
        messages = tuple(repositories.list_messages(save_id))
        provider = ShapeSwitchToolProvider(
            structured_data={
                "content": {"current_scene": "The beacon gallery hums."},
                "reason": "The location changed during play.",
                "source_message_id": message_id,
            }
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
            providers={"fake": cast(Any, provider)},
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert len(provider.tool_requests) == 1
        assert len(provider.structured_requests) == 1
        assert evolution == ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The location changed during play.",
                    source_message_id=message_id,
                ),
            ),
            diagnostics={
                "shape_switch": "structured_output",
                "provider": "fake",
                "model": "fake-tools",
            },
        )

    asyncio.run(run())


def test_tool_calling_evolver_keeps_error_when_structured_route_also_fails(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, _message_id = _create_full_roleplay_save(
            repositories
        )
        repositories.save_provider_model(
            provider="fake",
            model_id="fake-tools",
            display_name="Fake Tools",
            capabilities=["tool_calling", "structured_output"],
        )
        messages = tuple(repositories.list_messages(save_id))
        provider = ShapeFailingToolProvider()

        with pytest.raises(ProviderError) as exc_info:
            await ToolCallingProviderScenarioEvolver(
                provider=provider,
                provider_name="fake",
                model_id="fake-tools",
                providers={"fake": cast(Any, provider)},
            ).evolve(
                ScenarioEvolutionRequest(save_id=save_id, messages=messages),
                repositories=repositories,
            )

        assert exc_info.value.category == ProviderErrorCategory.MODEL_NOT_FOUND
        assert exc_info.value.fallback_attempted is True
        assert exc_info.value.fallback_provider == "fake"
        assert len(provider.structured_requests) == 1


def test_tool_calling_evolver_recovers_when_tool_fallback_also_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        _configure_scenario_evolution_tool_fallback(repositories)
        messages = tuple(repositories.list_messages(save_id))
        primary = ShapeSwitchToolProvider(
            structured_data={
                "content": {"current_scene": "The beacon gallery hums."},
                "reason": "The location changed during play.",
                "source_message_id": message_id,
            }
        )
        fallback = FailingScenarioEvolutionFallbackProvider(
            error=ProviderError(
                ProviderErrorCategory.MODEL_NOT_FOUND,
                "fallback model not found",
                status_code=404,
            )
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=primary,
            provider_name="fake",
            model_id="fake-tools",
            providers={"fake": cast(Any, primary), "fallback": cast(Any, fallback)},
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert len(primary.tool_requests) == 1
        assert len(fallback.tool_requests) == 1
        assert len(primary.structured_requests) == 1
        assert evolution.diagnostics["shape_switch"] == "structured_output"
        assert [update.section_id for update in evolution.updates] == [
            "current_scene"
        ]

    asyncio.run(run())


def test_tool_calling_evolver_recovers_when_tool_fallback_model_missing(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        _configure_scenario_evolution_tool_fallback(repositories)
        messages = tuple(repositories.list_messages(save_id))
        primary = RateLimitedShapeSwitchToolProvider(
            structured_data={
                "content": {"current_scene": "The beacon gallery hums."},
                "reason": "The location changed during play.",
                "source_message_id": message_id,
            }
        )
        fallback = FailingScenarioEvolutionFallbackProvider(
            error=ProviderError(
                ProviderErrorCategory.MODEL_NOT_FOUND,
                "fallback model missing",
                status_code=404,
            )
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=primary,
            provider_name="fake",
            model_id="fake-tools",
            providers={"fake": cast(Any, primary), "fallback": cast(Any, fallback)},
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert len(primary.tool_requests) == 1
        assert len(fallback.tool_requests) == 1
        assert len(primary.structured_requests) == 1
        assert evolution.diagnostics["shape_switch"] == "structured_output"
        assert [update.section_id for update in evolution.updates] == [
            "current_scene"
        ]

    asyncio.run(run())


def test_tool_calling_evolver_recovers_when_tool_fallback_rate_limited(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        _configure_scenario_evolution_tool_fallback(repositories)
        messages = tuple(repositories.list_messages(save_id))
        primary = ShapeSwitchToolProvider(
            structured_data={
                "content": {"current_scene": "The beacon gallery hums."},
                "reason": "The location changed during play.",
                "source_message_id": message_id,
            }
        )
        fallback = FailingScenarioEvolutionFallbackProvider(
            error=ProviderError(
                ProviderErrorCategory.RATE_LIMITED,
                "rate limited",
                status_code=429,
            )
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=primary,
            provider_name="fake",
            model_id="fake-tools",
            providers={"fake": cast(Any, primary), "fallback": cast(Any, fallback)},
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert len(primary.tool_requests) == 1
        assert len(fallback.tool_requests) == 1
        assert len(primary.structured_requests) == 1
        assert evolution.diagnostics["shape_switch"] == "structured_output"
        assert [update.section_id for update in evolution.updates] == [
            "current_scene"
        ]

    asyncio.run(run())


def test_tool_calling_evolver_keeps_fallback_result_when_tool_fallback_succeeds(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        _configure_scenario_evolution_tool_fallback(repositories)
        messages = tuple(repositories.list_messages(save_id))
        primary = ShapeSwitchToolProvider()
        fallback = FakeToolProvider(
            responses=[
                (
                    ProviderToolCall(
                        id="evolution-call",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "current_scene",
                                "text": "The beacon gallery hums.",
                                "reason": "The location changed during play.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                )
            ]
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=primary,
            provider_name="fake",
            model_id="fake-tools",
            providers={"fake": cast(Any, primary), "fallback": cast(Any, fallback)},
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert len(primary.tool_requests) == 1
        assert len(fallback.tool_requests) == 1
        assert primary.structured_requests == []
        assert evolution.diagnostics == {}
        assert [update.section_id for update in evolution.updates] == [
            "current_scene"
        ]

    asyncio.run(run())


def test_tool_calling_evolver_retries_invalid_section(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        provider = FakeToolProvider(
            [
                (
                    ProviderToolCall(
                        id="call-bad",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "tone",
                                "text": "Too broad.",
                                "reason": "Bad section.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                ),
                (
                    ProviderToolCall(
                        id="call-skip",
                        name="skip_scenario_evolution",
                        arguments_json=json.dumps(
                            {
                                "change_type": "no_phase_shift",
                                "skip_reason": "no_phase_shift",
                            }
                        ),
                    ),
                ),
            ]
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert evolution == ScenarioEvolution(skip_reason="no_phase_shift")
        assert len(provider.tool_requests) == 2
        retry_body = provider.tool_requests[1].messages[-1].body
        assert "section_id must be one of" in retry_body

    asyncio.run(run())


@pytest.mark.parametrize(
    ("case", "expected_feedback"),
    [
        ("source", "source_message_id must be one of"),
        ("text", "Scenario section update text is too long"),
        ("skip", "skip_reason must be one of"),
    ],
)
def test_tool_calling_evolver_retries_invalid_source_text_and_skip_reason(
    repositories: PersistenceRepositories,
    case: str,
    expected_feedback: str,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        bad_calls = {
            "source": ProviderToolCall(
                id="call-bad-source",
                name="update_scenario_section",
                arguments_json=json.dumps(
                    {
                        "section_id": "current_scene",
                        "text": "The beacon gallery hums.",
                        "reason": "Bad source.",
                        "source_message_id": "missing-message",
                    }
                ),
            ),
            "text": ProviderToolCall(
                id="call-overlong",
                name="update_scenario_section",
                arguments_json=json.dumps(
                    {
                        "section_id": "current_scene",
                        "text": "x" * 1201,
                        "reason": "Too much text.",
                        "source_message_id": message_id,
                    }
                ),
            ),
            "skip": ProviderToolCall(
                id="call-bad-skip",
                name="skip_scenario_evolution",
                arguments_json=json.dumps(
                    {
                        "change_type": "no_phase_shift",
                        "skip_reason": "dramatic_vibes",
                    }
                ),
            ),
        }
        provider = FakeToolProvider(
            [
                (bad_calls[case],),
                (
                    ProviderToolCall(
                        id="call-good-update",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "current_scene",
                                "text": "The beacon gallery hums.",
                                "reason": "The party moved into a new durable scene.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                ),
            ]
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert evolution == ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The party moved into a new durable scene.",
                    source_message_id=message_id,
                ),
            )
        )
        assert len(provider.tool_requests) == 2
        feedback = "\n".join(
            message.body for message in provider.tool_requests[1].messages
        )
        assert expected_feedback in feedback

    asyncio.run(run())


def test_tool_calling_evolver_preserves_accepted_updates_after_retry(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        provider = FakeToolProvider(
            [
                (
                    ProviderToolCall(
                        id="call-update",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "current_scene",
                                "text": "The beacon gallery hums.",
                                "reason": "The party moved into a new durable scene.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                    ProviderToolCall(
                        id="call-bad",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "tone",
                                "text": "Too broad.",
                                "reason": "Bad section.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                ),
                (),
            ]
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert evolution == ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The party moved into a new durable scene.",
                    source_message_id=message_id,
                ),
            )
        )
        assert len(provider.tool_requests) == 2

    asyncio.run(run())


def test_tool_calling_evolver_recovers_from_mixed_skip_and_update(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        provider = FakeToolProvider(
            [
                (
                    ProviderToolCall(
                        id="call-update",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "current_scene",
                                "text": "The beacon gallery hums.",
                                "reason": "The party moved into a new durable scene.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                    ProviderToolCall(
                        id="call-skip",
                        name="skip_scenario_evolution",
                        arguments_json=json.dumps(
                            {
                                "change_type": "no_phase_shift",
                                "skip_reason": "no_phase_shift",
                            }
                        ),
                    ),
                ),
                (
                    ProviderToolCall(
                        id="call-corrected-update",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "current_scene",
                                "text": "The beacon gallery hums.",
                                "reason": "The party moved into a new durable scene.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                ),
            ]
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert evolution == ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The party moved into a new durable scene.",
                    source_message_id=message_id,
                ),
            )
        )
        assert len(provider.tool_requests) == 2
        retry_body = provider.tool_requests[1].messages[-1].body
        assert "Scenario evolution cannot both skip and update sections" in retry_body

    asyncio.run(run())


def test_tool_calling_evolver_does_not_preserve_skip_from_invalid_turn(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        provider = FakeToolProvider(
            [
                (
                    ProviderToolCall(
                        id="call-bad-update",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "tone",
                                "text": "Too broad.",
                                "reason": "Bad section.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                    ProviderToolCall(
                        id="call-skip",
                        name="skip_scenario_evolution",
                        arguments_json=json.dumps(
                            {
                                "change_type": "no_phase_shift",
                                "skip_reason": "no_phase_shift",
                            }
                        ),
                    ),
                ),
                (
                    ProviderToolCall(
                        id="call-corrected-update",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "current_scene",
                                "text": "The beacon gallery hums.",
                                "reason": "The party moved into a new durable scene.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                ),
            ]
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert evolution == ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The party moved into a new durable scene.",
                    source_message_id=message_id,
                ),
            )
        )
        assert len(provider.tool_requests) == 2
        retry_body = provider.tool_requests[1].messages[-1].body
        assert "Scenario evolution cannot both skip and update sections" in retry_body

    asyncio.run(run())


def test_tool_calling_evolver_does_not_preserve_update_from_invalid_skip_turn(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        provider = FakeToolProvider(
            [
                (
                    ProviderToolCall(
                        id="call-update",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "current_scene",
                                "text": "The beacon gallery hums.",
                                "reason": "The party moved into a new durable scene.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                    ProviderToolCall(
                        id="call-bad-skip",
                        name="skip_scenario_evolution",
                        arguments_json=json.dumps(
                            {
                                "change_type": "no_phase_shift",
                            }
                        ),
                    ),
                ),
                (
                    ProviderToolCall(
                        id="call-skip",
                        name="skip_scenario_evolution",
                        arguments_json=json.dumps(
                            {
                                "change_type": "no_phase_shift",
                                "skip_reason": "no_phase_shift",
                            }
                        ),
                    ),
                ),
            ]
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert evolution == ScenarioEvolution(skip_reason="no_phase_shift")
        assert len(provider.tool_requests) == 2
        retry_body = provider.tool_requests[1].messages[-2].body
        assert "Scenario evolution cannot both skip and update sections" in retry_body

    asyncio.run(run())


def test_tool_calling_evolver_reports_skip_conflict_for_malformed_skip(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        provider = FakeToolProvider(
            [
                (
                    ProviderToolCall(
                        id="call-update",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "current_scene",
                                "text": "The beacon gallery hums.",
                                "reason": "The party moved into a new durable scene.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                    ProviderToolCall(
                        id="call-bad-section",
                        name="update_scenario_section",
                        arguments_json=json.dumps(
                            {
                                "section_id": "tone",
                                "text": "Too broad.",
                                "reason": "Bad section.",
                                "source_message_id": message_id,
                            }
                        ),
                    ),
                ),
                (
                    ProviderToolCall(
                        id="call-bad-skip",
                        name="skip_scenario_evolution",
                        arguments_json=json.dumps(
                            {
                                "change_type": "no_phase_shift",
                            }
                        ),
                    ),
                ),
                (),
            ]
        )

        evolution = await ToolCallingProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert evolution == ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The party moved into a new durable scene.",
                    source_message_id=message_id,
                ),
            )
        )
        assert len(provider.tool_requests) == 3
        retry_body = provider.tool_requests[2].messages[-1].body
        assert "Scenario evolution cannot both skip and update sections" in retry_body
        assert "Missing required field: skip_reason" not in retry_body

    asyncio.run(run())


def test_structured_evolver_applies_phase_shift_updates(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        messages = tuple(repositories.list_messages(save_id))
        provider = FakeStructuredProvider(
            {
                "change_type": "phase_shift",
                "updates": [
                    {
                        "section_id": "current_scene",
                        "text": "The beacon gallery hums.",
                        "reason": "The party moved into a new durable scene phase.",
                        "source_message_id": message_id,
                    }
                ],
            }
        )

        evolution = await StructuredProviderScenarioEvolver(
            provider=provider,
            provider_name="fake",
            model_id="fake-structured",
        ).evolve(
            ScenarioEvolutionRequest(save_id=save_id, messages=messages),
            repositories=repositories,
        )

        assert evolution == ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The party moved into a new durable scene phase.",
                    source_message_id=message_id,
                ),
            )
        )

    asyncio.run(run())


def test_apply_evolution_persists_save_specific_update_without_mutating_base(
    repositories: PersistenceRepositories,
) -> None:
    save_id, scenario_id, message_id = _create_full_roleplay_save(repositories)
    service = ScenarioEvolutionService(
        repositories=repositories,
        evolver=StructuredProviderScenarioEvolver(
            provider=FakeStructuredProvider({}),
            provider_name="fake",
            model_id="fake-structured",
        ),
        provider_name="fake",
        model_id="fake-structured",
    )

    update = service.apply_evolution(
        save_id=save_id,
        evolution=ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The party moved upstairs.",
                    source_message_id=message_id,
                ),
            )
        ),
        allowed_source_message_ids=(message_id,),
    )

    assert update is not None
    assert update.source_message_id == message_id
    assert update.reason == "current_scene: The party moved upstairs."
    assert json.loads(update.content_json)["current_scene"] == (
        "The beacon gallery hums."
    )
    base_scenario = repositories.get_scenario(scenario_id)
    assert base_scenario is not None
    assert json.loads(base_scenario.content_json)["current_scene"] == (
        "The warden stands at the lower gate."
    )


def test_apply_evolution_rejects_invalid_sections_and_repairs_unknown_sources(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
    service = ScenarioEvolutionService(
        repositories=repositories,
        evolver=StructuredProviderScenarioEvolver(
            provider=FakeStructuredProvider({}),
            provider_name="fake",
            model_id="fake-structured",
        ),
        provider_name="fake",
        model_id="fake-structured",
    )

    with pytest.raises(ValueError, match="Scenario section cannot evolve: title"):
        service.apply_evolution(
            save_id=save_id,
            evolution=ScenarioEvolution(
                updates=(
                    ScenarioSectionUpdate(
                        section_id="title",
                        text="New title",
                        reason="Nope",
                        source_message_id=message_id,
                    ),
                )
            ),
        )

    update = service.apply_evolution(
        save_id=save_id,
        evolution=ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The party moved upstairs.",
                    source_message_id="missing-message",
                ),
            )
        ),
        allowed_source_message_ids=(message_id,),
    )

    assert update is not None
    assert update.source_message_id == message_id
    assert json.loads(update.source_message_ids_json) == [message_id]


def test_apply_evolution_drops_unknown_sources_when_no_allowed_source_exists(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _scenario_id, _message_id = _create_full_roleplay_save(repositories)
    service = ScenarioEvolutionService(
        repositories=repositories,
        evolver=StructuredProviderScenarioEvolver(
            provider=FakeStructuredProvider({}),
            provider_name="fake",
            model_id="fake-structured",
        ),
        provider_name="fake",
        model_id="fake-structured",
    )

    update = service.apply_evolution(
        save_id=save_id,
        evolution=ScenarioEvolution(
            updates=(
                ScenarioSectionUpdate(
                    section_id="current_scene",
                    text="The beacon gallery hums.",
                    reason="The party moved upstairs.",
                    source_message_id="missing-message",
                ),
            )
        ),
        allowed_source_message_ids=(),
    )

    assert update is None


def test_evolve_after_turn_records_successful_noop_job(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        provider = FakeStructuredProvider({"updates": []})
        service = ScenarioEvolutionService(
            repositories=repositories,
            evolver=StructuredProviderScenarioEvolver(
                provider=provider,
                provider_name="fake",
                model_id="fake-structured",
            ),
            provider_name="fake",
            model_id="fake-structured",
        )

        update = await service.evolve_after_turn(
            save_id=save_id,
            source_message_ids=(message_id,),
        )

        jobs = repositories.list_jobs_by_status(("succeeded",))
        assert update is None
        assert len(jobs) == 1
        assert jobs[0].type == "scenario_evolution"
        assert jobs[0].result == {
            "scenario_update_id": None,
            "section_update_count": 0,
            "skip_reason": "no_phase_shift",
        }

    asyncio.run(run())


def test_evolve_after_turn_skips_marked_safety_transition_before_provider(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        repositories.connection.execute(
            "UPDATE messages SET body = ?, safety_transition = ? WHERE id = ?",
            (
                "The intimate moment is kept off-screen. Hours later, "
                "the next scene begins.",
                "fade_to_black",
                message_id,
            ),
        )
        repositories.commit()
        provider = FakeStructuredProvider(
            {"updates": [{"section_id": "current_scene"}]}
        )
        service = ScenarioEvolutionService(
            repositories=repositories,
            evolver=StructuredProviderScenarioEvolver(
                provider=provider,
                provider_name="fake",
                model_id="fake-structured",
            ),
            provider_name="fake",
            model_id="fake-structured",
        )

        assert await service.evolve_after_turn(
            save_id=save_id,
            source_message_ids=(message_id,),
        ) is None
        assert provider.requests == []
        job = repositories.list_jobs_by_status(("succeeded",))[0]
        assert job.result == {
            "scenario_update_id": None,
            "section_update_count": 0,
            "skip_reason": "safety_transition",
        }

    asyncio.run(run())


def test_evolve_after_turn_persists_structured_output_retry_metadata(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        provider = FakeStructuredProvider(
            {"updates": []},
            raw_metadata={
                "_bragi_retry": {
                    "attempt_count": 2,
                    "max_attempts": 3,
                    "attempts": [
                        {
                            "attempt": 1,
                            "duration_ms": 14,
                            "error_category": "rate_limited",
                            "http_status": 429,
                        },
                        {
                            "attempt": 2,
                            "duration_ms": 9,
                            "error_category": None,
                        },
                    ],
                }
            },
        )
        service = ScenarioEvolutionService(
            repositories=repositories,
            evolver=StructuredProviderScenarioEvolver(
                provider=provider,
                provider_name="fake",
                model_id="fake-structured",
                providers={"fake": cast(Any, provider)},
            ),
            provider_name="fake",
            model_id="fake-structured",
        )

        await service.evolve_after_turn(
            save_id=save_id,
            source_message_ids=(message_id,),
        )

        jobs = repositories.list_jobs_by_status(("succeeded",))
        assert len(jobs) == 1
        assert jobs[0].result == {
            "scenario_update_id": None,
            "section_update_count": 0,
            "skip_reason": "no_phase_shift",
            "attempt_count": 2,
            "max_attempts": 3,
            "provider_call_count": 1,
            "provider_calls": [
                {
                    "task": "scenario_evolution",
                    "provider": "fake",
                    "model": "fake-structured",
                    "schema_name": "scenario_evolution",
                    "attempt_count": 2,
                    "max_attempts": 3,
                    "retry_attempts": [
                        {
                            "attempt": 1,
                            "duration_ms": 14,
                            "error_category": "rate_limited",
                            "http_status": 429,
                        },
                        {
                            "attempt": 2,
                            "duration_ms": 9,
                            "error_category": None,
                        },
                    ],
                }
            ],
            "retry_attempts": [
                {
                    "attempt": 1,
                    "duration_ms": 14,
                    "error_category": "rate_limited",
                    "http_status": 429,
                },
                {
                    "attempt": 2,
                    "duration_ms": 9,
                    "error_category": None,
                },
            ],
        }

    asyncio.run(run())


def test_evolve_after_turn_records_turn_level_skip_without_scenario_update(
    repositories: PersistenceRepositories,
) -> None:
    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        provider = FakeStructuredProvider(
            {
                "change_type": "turn_level_change",
                "updates": [
                    {
                        "section_id": "current_scene",
                        "text": "The warden blushes at the captain's praise.",
                        "reason": "An emotional beat changed the moment.",
                        "source_message_id": message_id,
                    }
                ],
            }
        )
        service = ScenarioEvolutionService(
            repositories=repositories,
            evolver=StructuredProviderScenarioEvolver(
                provider=provider,
                provider_name="fake",
                model_id="fake-structured",
            ),
            provider_name="fake",
            model_id="fake-structured",
        )

        update = await service.evolve_after_turn(
            save_id=save_id,
            source_message_ids=(message_id,),
        )

        jobs = repositories.list_jobs_by_status(("succeeded",))
        assert update is None
        assert repositories.list_save_scenario_updates(save_id) == []
        assert len(jobs) == 1
        assert jobs[0].result == {
            "scenario_update_id": None,
            "section_update_count": 0,
            "skip_reason": "turn_level_change",
        }

    asyncio.run(run())


def test_evolve_after_turn_records_redacted_failed_job(
    repositories: PersistenceRepositories,
) -> None:
    class FailingEvolver:
        async def evolve(self, *_args: object, **_kwargs: object) -> ScenarioEvolution:
            raise RuntimeError("provider token=secret-value failed")

    async def run() -> None:
        save_id, _scenario_id, message_id = _create_full_roleplay_save(repositories)
        service = ScenarioEvolutionService(
            repositories=repositories,
            evolver=cast(Any, FailingEvolver()),
            provider_name="fake",
            model_id="fake-structured",
        )

        with pytest.raises(RuntimeError, match="provider token=secret-value failed"):
            await service.evolve_after_turn(
                save_id=save_id,
                source_message_ids=(message_id,),
            )

        failed_jobs = repositories.list_jobs_by_status(("failed",))
        assert len(failed_jobs) == 1
        assert failed_jobs[0].error == "provider token=[redacted] failed"

    asyncio.run(run())
