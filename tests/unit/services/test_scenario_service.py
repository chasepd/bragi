from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Iterator, MutableMapping
from pathlib import Path
from typing import cast

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.chat_rendering import provider_chat_messages
from bragi.providers.contracts import (
    ChatPromptPurpose,
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderConfigStatus,
    ProviderModel,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.system_prompt import DEFAULT_PROSE_SAFETY_SECTION
from bragi.services.character_profile_completion import ScenarioCharacterStarter
from bragi.services.model_preferences import scenario_generation_section_model_task
from bragi.services.scenario_service import (
    ScenarioDraft,
    ScenarioGenerationProgress,
    ScenarioService,
    ScenarioType,
)
from bragi.services.sexual_content_safety import CONTENT_FILTER_TRANSITION

FULL_ROLEPLAY_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "tone_genre",
    "opening_message",
)

DATING_SIM_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_character_profile",
    "player_role",
    "tone_genre",
    "opening_message",
)

FANTASY_ROLEPLAY_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "magic_system",
    "realms_and_places",
    "factions_and_orders",
    "myths_and_creatures",
    "quest_stakes",
    "tone_genre",
    "opening_message",
)

SCIENCE_FICTION_ROLEPLAY_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "technology_level",
    "setting_scope",
    "species_and_intelligences",
    "factions_and_institutions",
    "mission_stakes",
    "tone_genre",
    "opening_message",
)

FIRST_CONTACT_EXPLORATION_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "mission_profile",
    "ship_or_base_status",
    "exploration_target",
    "unknown_intelligence",
    "knowledge_state",
    "translation_progress",
    "discoveries_and_samples",
    "hazards_and_escalation",
    "tone_genre",
    "opening_message",
)

SURVIVAL_EXPEDITION_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "expedition_goal",
    "route_options",
    "resource_inventory",
    "environmental_conditions",
    "hazards_and_events",
    "camp_status",
    "travel_progress",
    "tone_genre",
    "opening_message",
)

TIME_LOOP_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "loop_premise",
    "reset_trigger",
    "loop_duration",
    "starting_state",
    "objective",
    "failure_conditions",
    "baseline_world_state",
    "loop_schedule",
    "persistent_knowledge",
    "persistence_exceptions",
    "npc_memory_rules",
    "current_loop_state",
    "tone_genre",
    "opening_message",
)

INVESTIGATION_MYSTERY_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "case_facts",
    "clues",
    "timeline",
    "red_herrings",
    "hidden_truth",
    "case_status",
    "tone_genre",
    "opening_message",
)

HEIST_INFILTRATION_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "target_location",
    "objectives_and_stakes",
    "intel_and_access",
    "security_model",
    "alert_and_heat",
    "loadout_and_tools",
    "complications",
    "extraction_routes",
    "aftermath",
    "tone_genre",
    "opening_message",
)

POLITICAL_INTRIGUE_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "political_arena",
    "political_factions",
    "central_conflict",
    "secrets_and_leverage",
    "reputation_and_standing",
    "obligations_and_favors",
    "alliances_and_rivalries",
    "event_calendar",
    "political_pressure",
    "public_private_knowledge",
    "tone_genre",
    "opening_message",
)

SETTLEMENT_BUILDER_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "settlement_profile",
    "resources_and_indicators",
    "projects_and_facilities",
    "threats_and_opportunities",
    "calendar_and_deadlines",
    "tone_genre",
    "opening_message",
)

MONSTER_HUNT_BOUNTY_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "hunt_profile",
    "target_profile",
    "leads_and_clues",
    "hunt_locations",
    "preparation_state",
    "hunt_status",
    "tone_genre",
    "opening_message",
)

ROAD_TRIP_PILGRIMAGE_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "journey_profile",
    "route_and_stops",
    "transport_and_supplies",
    "recurring_pressures",
    "relationship_threads",
    "journey_progress",
    "tone_genre",
    "opening_message",
)

MERCHANT_TRADE_ROUTE_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "trade_profile",
    "cargo_inventory",
    "markets_and_stops",
    "contracts_and_debts",
    "route_hazards",
    "profit_and_loss",
    "tone_genre",
    "opening_message",
)

ACTION_CHOICE_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "tone_genre",
    "choice_style",
    "opening_message",
)


class RecordingScenarioProvider:
    provider_name = "openrouter"

    def __init__(self, response_sections: dict[str, str]) -> None:
        self.response_sections = response_sections
        self.section_ids = tuple(response_sections)
        self.chat_requests: list[ChatRequest] = []
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def validate_config(self) -> ProviderConfigStatus:
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
        )

    async def list_models(self) -> list[ProviderModel]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        request_index = len(self.chat_requests)
        self.chat_requests.append(request)
        try:
            section_id = self.section_ids[request_index]
        except IndexError as exc:
            raise AssertionError("unexpected extra scenario section request") from exc
        return ChatResponse(
            body=self.response_sections[section_id],
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("scenario draft generation must not request images")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data={
                "action": "allow",
                "category": "none",
                "reason": "The section stays within the content ceiling.",
                "minimum_rating": "g",
            },
            provider=request.provider,
            model_id=request.model_id,
        )


class BlockingScenarioProvider(RecordingScenarioProvider):
    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.CONTENT_BLOCKED,
            "scenario section blocked",
        )


class SequentialScenarioProvider:
    provider_name = "openrouter"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.chat_requests: list[ChatRequest] = []

    async def validate_config(self) -> ProviderConfigStatus:
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
        )

    async def list_models(self) -> list[ProviderModel]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected extra scenario section request")
        return ChatResponse(
            body=self.responses.pop(0),
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("scenario draft generation must not request images")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        return StructuredOutputResponse(
            data={
                "action": "allow",
                "category": "none",
                "reason": "The section stays within the content ceiling.",
                "minimum_rating": "g",
            },
            provider=request.provider,
            model_id=request.model_id,
        )


class ScenarioSafetyProvider(RecordingScenarioProvider):
    provider_name = "safety"

    def __init__(self, action: str) -> None:
        super().__init__({})
        self.action = action
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data={
                "action": self.action,
                "category": "violence",
                "reason": "The draft exceeds the configured ceiling.",
                "minimum_rating": "r",
            },
            provider=request.provider,
            model_id=request.model_id,
        )


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_generate_full_roleplay_draft_uses_provider_chat_and_returns_minimal_sections(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_full_roleplay_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.FULL_ROLEPLAY,
            seed="A volcanic border keep cut off by ash storms.",
        )
    )

    assert _sections(draft) == _full_roleplay_sections()
    assert tuple(_sections(draft)) == FULL_ROLEPLAY_SECTION_IDS
    assert len(provider.chat_requests) == len(FULL_ROLEPLAY_SECTION_IDS)
    for request in provider.chat_requests:
        assert request.provider == "openrouter"
        assert request.model_id == "scenario-drafter"
        assert request.prompt_purpose is ChatPromptPurpose.SCENARIO_GENERATION
        assert request.messages[-1].role in {"user", "player"}
        assert "volcanic border keep" in _request_text(request)
        assert DEFAULT_PROSE_SAFETY_SECTION in provider_chat_messages(request)[0][
            "content"
        ]
    assert _requests_include_prior_section_context(
        provider.chat_requests,
        _full_roleplay_sections(),
    )


def test_generate_action_choice_draft_uses_choice_sections(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_action_choice_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.FULL_ROLEPLAY,
            seed="A cliffside library where every shelf is a door.",
            action_choices_enabled=True,
        )
    )

    assert draft.type is ScenarioType.FULL_ROLEPLAY
    assert draft.action_choices_enabled is True
    assert _sections(draft) == _action_choice_sections()
    assert tuple(_sections(draft)) == ACTION_CHOICE_SECTION_IDS
    assert len(provider.chat_requests) == len(ACTION_CHOICE_SECTION_IDS)
    request_text = "\n".join(
        _request_text(request) for request in provider.chat_requests
    )
    assert "action choices: enabled" in request_text.casefold()
    assert "numbered options" in request_text.casefold()
    assert "concrete changed situation" in request_text.casefold()
    assert "naturally stop" not in request_text.casefold()


def test_child_scenario_generation_uses_account_policy_and_sanitizes_output(
    repositories: PersistenceRepositories,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=child.id,
        key="content_filter_rating",
        value="g",
    )
    provider = RecordingScenarioProvider(
        {"opening_message": "The blast dismembered the guard in graphic detail."}
    )
    safety_provider = ScenarioSafetyProvider("block")
    repositories.set_model_preference(
        task="content_safety",
        provider="safety",
        model_id="safety-model",
    )
    repositories.save_provider_model(
        provider="safety",
        model_id="safety-model",
        display_name="Safety Model",
        capabilities=["structured_output"],
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
        providers={"openrouter": provider, "safety": safety_provider},
        current_user_id=child.id,
    )

    result = asyncio.run(
        service.regenerate_section(
            scenario_type=ScenarioType.FULL_ROLEPLAY,
            seed="A beacon keeper survives a sudden storm.",
            section_id="opening_message",
            sections={"title": "Lantern Keep"},
        )
    )

    assert result.body == CONTENT_FILTER_TRANSITION
    assert result.minimum_rating == "g"
    assert provider.chat_requests[0].content_rating == "g"
    assert provider.chat_requests[0].fade_to_black_enabled is True
    assert len(safety_provider.structured_output_requests) == 1


def test_generated_draft_persists_section_rating_provenance(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_full_roleplay_sections())
    )
    safety_provider = ScenarioSafetyProvider("allow")
    repositories.set_model_preference(
        task="content_safety",
        provider="safety",
        model_id="safety-model",
    )
    repositories.save_provider_model(
        provider="safety",
        model_id="safety-model",
        display_name="Safety Model",
        capabilities=["structured_output"],
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
        providers={"openrouter": provider, "safety": safety_provider},
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.FULL_ROLEPLAY,
            seed="A beacon keeper survives a sudden storm.",
        )
    )

    assert draft.metadata is not None
    assert draft.metadata["content_rating"] == "g"
    assert draft.metadata["section_content_ratings"] == {
        section_id: "g" for section_id in FULL_ROLEPLAY_SECTION_IDS
    }


def test_generate_draft_accepts_explicit_allowed_sections_and_metadata(
    repositories: PersistenceRepositories,
) -> None:
    sections = {
        **_full_roleplay_sections(),
        "worldbuilding": "The border towers still answer old beacon law.",
        "current_scene": "The warden stands under ashfall.",
    }
    provider = RecordingScenarioProvider(_provider_response_sections(sections))
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.FULL_ROLEPLAY,
            seed="Continue from a long-running save.",
            section_ids=tuple(sections),
            metadata={"origin": "save_continuation"},
        )
    )

    assert _sections(draft) == sections
    assert draft.metadata is not None
    assert draft.metadata["origin"] == "save_continuation"
    assert draft.metadata["content_rating"] == "g"
    assert draft.metadata["section_content_ratings"] == {
        section_id: "g" for section_id in sections
    }
    assert [
        _requested_scenario_section(request.messages[-1].body)
        for request in provider.chat_requests
    ] == list(sections)


def test_generate_draft_uses_section_model_override_for_matching_section(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task=scenario_generation_section_model_task("opening_message"),
        provider="openrouter",
        model_id="opening-drafter",
    )
    provider = RecordingScenarioProvider(
        _provider_response_sections(_full_roleplay_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.FULL_ROLEPLAY,
            seed="A volcanic border keep cut off by ash storms.",
        )
    )

    requests_by_section = dict(
        zip(FULL_ROLEPLAY_SECTION_IDS, provider.chat_requests, strict=True)
    )
    assert requests_by_section["opening_message"].model_id == "opening-drafter"
    assert {
        section_id: request.model_id
        for section_id, request in requests_by_section.items()
        if section_id != "opening_message"
    } == {
        section_id: "scenario-drafter"
        for section_id in FULL_ROLEPLAY_SECTION_IDS
        if section_id != "opening_message"
    }


@pytest.mark.parametrize(
    ("scenario_type", "scenario_types"),
    [
        ("character_interaction", None),
        ("dating_sim", ["dating_sim", "character_interaction"]),
    ],
)
def test_generate_draft_rejects_retired_character_interaction(
    repositories: PersistenceRepositories,
    scenario_type: str,
    scenario_types: list[str] | None,
) -> None:
    provider = RecordingScenarioProvider(_provider_response_sections({}))
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    with pytest.raises(ValueError, match="no longer supported"):
        asyncio.run(
            service.generate_draft(
                scenario_type=scenario_type,
                scenario_types=scenario_types,
                seed="Retired scenario",
            )
        )

    assert provider.chat_requests == []


def test_generate_dating_sim_draft_uses_player_sections_without_character_starters(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_dating_sim_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.DATING_SIM,
            seed="A seaside summer academy with a male transfer student.",
        )
    )

    sections = _sections(draft)
    assert sections == _dating_sim_sections()
    assert tuple(sections) == DATING_SIM_SECTION_IDS
    assert len(provider.chat_requests) == len(DATING_SIM_SECTION_IDS)
    assert draft.character_starters == ()


def test_generate_non_fantasy_draft_does_not_include_starter_name_candidates(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_dating_sim_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.DATING_SIM,
            seed="A contemporary speed dating night.",
        )
    )

    request_text = "\n".join(
        _request_text(request) for request in provider.chat_requests
    )
    assert "Ordinary contemporary name candidates" in request_text
    assert "character starters" not in request_text


def test_generate_fantasy_draft_does_not_include_ordinary_name_candidates(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_fantasy_roleplay_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.FANTASY_ROLEPLAY,
            seed="A fallen moon kingdom where oath-magic binds every crown.",
        )
    )

    request_text = "\n".join(
        _request_text(request) for request in provider.chat_requests
    )
    assert "Ordinary contemporary name candidates" not in request_text


def test_generate_fantasy_roleplay_draft_uses_genre_sections(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_fantasy_roleplay_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.FANTASY_ROLEPLAY,
            seed="A fallen moon kingdom where oath-magic binds every crown.",
        )
    )

    sections = _sections(draft)
    assert sections == _fantasy_roleplay_sections()
    assert tuple(sections) == FANTASY_ROLEPLAY_SECTION_IDS
    assert len(provider.chat_requests) == len(FANTASY_ROLEPLAY_SECTION_IDS)
    magic_request = provider.chat_requests[
        FANTASY_ROLEPLAY_SECTION_IDS.index("magic_system")
    ]
    assert "costs, limits, risks" in _request_text(magic_request)
    assert "fantasy roleplay scenario" in _request_text(magic_request)


def test_generate_science_fiction_roleplay_draft_uses_genre_sections(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_science_fiction_roleplay_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.SCIENCE_FICTION_ROLEPLAY,
            seed="A first-contact salvage crew trapped near a silent megastructure.",
        )
    )

    sections = _sections(draft)
    assert sections == _science_fiction_roleplay_sections()
    assert tuple(sections) == SCIENCE_FICTION_ROLEPLAY_SECTION_IDS
    assert len(provider.chat_requests) == len(SCIENCE_FICTION_ROLEPLAY_SECTION_IDS)
    technology_request = provider.chat_requests[
        SCIENCE_FICTION_ROLEPLAY_SECTION_IDS.index("technology_level")
    ]
    request_text = _request_text(technology_request)
    assert "available technology" in request_text
    assert "constraints" in request_text
    assert "science fiction roleplay scenario" in request_text


def test_generate_hybrid_draft_merges_unique_genre_sections(
    repositories: PersistenceRepositories,
) -> None:
    expected_section_order = (
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "technology_level",
        "setting_scope",
        "species_and_intelligences",
        "factions_and_institutions",
        "mission_stakes",
        "player_character_profile",
        "tone_genre",
        "opening_message",
    )
    section_values = {
        **_science_fiction_roleplay_sections(),
        "player_character_profile": (
            "Ren is a courier caught between orbital politics and romance routes."
        ),
    }
    expected_sections = {
        section_id: section_values[section_id]
        for section_id in expected_section_order
    }
    provider = RecordingScenarioProvider(_provider_response_sections(expected_sections))
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.SCIENCE_FICTION_ROLEPLAY,
            scenario_types=(
                ScenarioType.SCIENCE_FICTION_ROLEPLAY,
                ScenarioType.DATING_SIM,
            ),
            seed="A science fiction dating sim aboard a disputed orbital academy.",
        )
    )

    sections = _sections(draft)
    assert draft.type is ScenarioType.SCIENCE_FICTION_ROLEPLAY
    assert draft.scenario_types == (
        ScenarioType.SCIENCE_FICTION_ROLEPLAY,
        ScenarioType.DATING_SIM,
    )
    assert tuple(sections) == expected_section_order
    assert sections == expected_sections
    assert len(provider.chat_requests) == len(sections)
    profile_request = provider.chat_requests[
        tuple(sections).index("player_character_profile")
    ]
    profile_prompt = _request_text(profile_request)
    assert "science fiction / dating sim hybrid scenario" in profile_prompt
    assert "romantic availability" in profile_prompt


def test_generate_first_contact_exploration_draft_uses_discovery_sections(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_first_contact_exploration_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.FIRST_CONTACT_EXPLORATION,
            seed=(
                "A survey crew on an ocean moon must translate whale-like "
                "signals before the ice shelf breaks."
            ),
        )
    )

    sections = _sections(draft)
    assert sections == _first_contact_exploration_sections()
    assert tuple(sections) == FIRST_CONTACT_EXPLORATION_SECTION_IDS
    assert len(provider.chat_requests) == len(FIRST_CONTACT_EXPLORATION_SECTION_IDS)
    knowledge_request = provider.chat_requests[
        FIRST_CONTACT_EXPLORATION_SECTION_IDS.index("knowledge_state")
    ]
    knowledge_prompt = _request_text(knowledge_request)
    assert "observed facts" in knowledge_prompt
    assert "hypotheses" in knowledge_prompt
    assert "confirmed knowledge" in knowledge_prompt
    translation_request = provider.chat_requests[
        FIRST_CONTACT_EXPLORATION_SECTION_IDS.index("translation_progress")
    ]
    translation_prompt = _request_text(translation_request)
    assert "false assumptions" in translation_prompt
    assert "confirmed meanings" in translation_prompt
    assert "Do not include JSON" in translation_prompt
    assert _requests_include_prior_section_context(
        provider.chat_requests,
        _first_contact_exploration_sections(),
    )


def test_generate_investigation_mystery_draft_uses_case_sections(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_investigation_mystery_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.INVESTIGATION_MYSTERY,
            seed="A locked museum wing, a vanished curator, and one forged alibi.",
        )
    )

    sections = _sections(draft)
    assert sections == _investigation_mystery_sections()
    assert tuple(sections) == INVESTIGATION_MYSTERY_SECTION_IDS
    assert len(provider.chat_requests) == len(INVESTIGATION_MYSTERY_SECTION_IDS)
    clue_request = provider.chat_requests[
        INVESTIGATION_MYSTERY_SECTION_IDS.index("clues")
    ]
    clue_prompt = _request_text(clue_request)
    assert "discovery status" in clue_prompt
    assert "connections to suspects" in clue_prompt
    assert "Do not include JSON" in clue_prompt
    hidden_truth_request = provider.chat_requests[
        INVESTIGATION_MYSTERY_SECTION_IDS.index("hidden_truth")
    ]
    hidden_truth_prompt = _request_text(hidden_truth_request)
    assert "not known to the player" in hidden_truth_prompt
    assert "investigation mystery scenario" in hidden_truth_prompt
    assert _requests_include_prior_section_context(
        provider.chat_requests,
        _investigation_mystery_sections(),
    )


def test_generate_survival_expedition_draft_uses_expedition_sections(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_survival_expedition_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.SURVIVAL_EXPEDITION,
            seed="A relief party must cross a whiteout pass before fever spreads.",
        )
    )

    sections = _sections(draft)
    assert sections == _survival_expedition_sections()
    assert tuple(sections) == SURVIVAL_EXPEDITION_SECTION_IDS
    assert len(provider.chat_requests) == len(SURVIVAL_EXPEDITION_SECTION_IDS)
    resource_request = provider.chat_requests[
        SURVIVAL_EXPEDITION_SECTION_IDS.index("resource_inventory")
    ]
    request_text = _request_text(resource_request)
    assert "important supplies" in request_text
    assert "equipment" in request_text
    assert "survival expedition scenario" in request_text


def test_generate_heist_infiltration_draft_uses_security_sections(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_heist_infiltration_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.HEIST_INFILTRATION,
            seed=(
                "A crew must steal a treaty from a skybank vault before the "
                "rival house frames them."
            ),
        )
    )

    sections = _sections(draft)
    assert sections == _heist_infiltration_sections()
    assert tuple(sections) == HEIST_INFILTRATION_SECTION_IDS
    assert len(provider.chat_requests) == len(HEIST_INFILTRATION_SECTION_IDS)
    security_request = provider.chat_requests[
        HEIST_INFILTRATION_SECTION_IDS.index("security_model")
    ]
    security_prompt = _request_text(security_request)
    assert "guards" in security_prompt
    assert "alarms" in security_prompt
    assert "heist / infiltration scenario" in security_prompt
    alert_request = provider.chat_requests[
        HEIST_INFILTRATION_SECTION_IDS.index("alert_and_heat")
    ]
    alert_prompt = _request_text(alert_request)
    assert "suspicion" in alert_prompt
    assert "alarm" in alert_prompt
    assert "Do not include JSON" in alert_prompt
    assert _requests_include_prior_section_context(
        provider.chat_requests,
        _heist_infiltration_sections(),
    )


def test_generate_political_intrigue_draft_uses_social_pressure_sections(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_political_intrigue_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.POLITICAL_INTRIGUE,
            seed=(
                "A swing-vote envoy navigates harbor factions before a midnight "
                "no-confidence vote."
            ),
        )
    )

    sections = _sections(draft)
    assert sections == _political_intrigue_sections()
    assert tuple(sections) == POLITICAL_INTRIGUE_SECTION_IDS
    assert len(provider.chat_requests) == len(POLITICAL_INTRIGUE_SECTION_IDS)
    factions_request = provider.chat_requests[
        POLITICAL_INTRIGUE_SECTION_IDS.index("political_factions")
    ]
    factions_prompt = _request_text(factions_request)
    assert "faction" in factions_prompt
    assert "political intrigue scenario" in factions_prompt
    pressure_request = provider.chat_requests[
        POLITICAL_INTRIGUE_SECTION_IDS.index("political_pressure")
    ]
    pressure_prompt = _request_text(pressure_request)
    assert "timed political pressure" in pressure_prompt
    assert "Do not include JSON" in pressure_prompt
    assert _requests_include_prior_section_context(
        provider.chat_requests,
        _political_intrigue_sections(),
    )


@pytest.mark.parametrize(
    (
        "scenario_type",
        "section_factory",
        "section_ids",
        "focused_section",
        "expected_snippets",
    ),
    [
        (
            ScenarioType.SETTLEMENT_BUILDER,
            "_settlement_builder_sections",
            SETTLEMENT_BUILDER_SECTION_IDS,
            "projects_and_facilities",
            ("projects", "blockers", "benefits"),
        ),
        (
            ScenarioType.MONSTER_HUNT_BOUNTY,
            "_monster_hunt_bounty_sections",
            MONSTER_HUNT_BOUNTY_SECTION_IDS,
            "leads_and_clues",
            ("clues", "discovery", "target"),
        ),
        (
            ScenarioType.ROAD_TRIP_PILGRIMAGE,
            "_road_trip_pilgrimage_sections",
            ROAD_TRIP_PILGRIMAGE_SECTION_IDS,
            "journey_progress",
            ("current leg", "detours", "destination"),
        ),
        (
            ScenarioType.MERCHANT_TRADE_ROUTE,
            "_merchant_trade_route_sections",
            MERCHANT_TRADE_ROUTE_SECTION_IDS,
            "contracts_and_debts",
            ("contracts", "deadlines", "debts"),
        ),
    ],
)
def test_generate_management_scenario_draft_uses_template_sections(
    repositories: PersistenceRepositories,
    scenario_type: ScenarioType,
    section_factory: str,
    section_ids: tuple[str, ...],
    focused_section: str,
    expected_snippets: tuple[str, ...],
) -> None:
    section_factory_func = cast(
        Callable[[], dict[str, str]],
        globals()[section_factory],
    )
    expected_sections = section_factory_func()
    provider = RecordingScenarioProvider(
        _provider_response_sections(expected_sections)
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=scenario_type,
            seed="A campaign about persistent logistics and consequences.",
        )
    )

    sections = _sections(draft)
    assert sections == expected_sections
    assert tuple(sections) == section_ids
    assert len(provider.chat_requests) == len(section_ids)
    request_text = _request_text(
        provider.chat_requests[section_ids.index(focused_section)]
    )
    for snippet in expected_snippets:
        assert snippet in request_text
    assert "Do not include JSON" in request_text
    assert _requests_include_prior_section_context(
        provider.chat_requests,
        expected_sections,
    )


def test_generate_time_loop_draft_uses_loop_boundary_sections(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_time_loop_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.TIME_LOOP,
            seed="A harbor city repeats the same festival day until the bell is saved.",
        )
    )

    sections = _sections(draft)
    assert sections == _time_loop_sections()
    assert tuple(sections) == TIME_LOOP_SECTION_IDS
    assert len(provider.chat_requests) == len(TIME_LOOP_SECTION_IDS)
    knowledge_request = provider.chat_requests[
        TIME_LOOP_SECTION_IDS.index("persistent_knowledge")
    ]
    knowledge_prompt = _request_text(knowledge_request)
    assert "player/meta knowledge" in knowledge_prompt
    assert "NPCs reset" in knowledge_prompt
    assert "Do not include JSON" in knowledge_prompt
    reset_request = provider.chat_requests[
        TIME_LOOP_SECTION_IDS.index("reset_trigger")
    ]
    assert "time loop scenario" in _request_text(reset_request)
    assert _requests_include_prior_section_context(
        provider.chat_requests,
        _time_loop_sections(),
    )


def test_generate_draft_rejects_empty_section_response_with_friendly_error(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        {
            **_provider_response_sections(_full_roleplay_sections()),
            "premise": " \n\t ",
        }
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            service.generate_draft(
                scenario_type=ScenarioType.FULL_ROLEPLAY,
                seed="A volcanic border keep cut off by ash storms.",
            )
        )

    message = str(exc_info.value)
    assert "premise" in message
    assert "empty" in message.lower()
    assert "json" not in message.lower()


def test_generate_draft_uses_chat_fallback_for_blocked_sections(
    repositories: PersistenceRepositories,
) -> None:
    primary = BlockingScenarioProvider(
        _provider_response_sections(_full_roleplay_sections())
    )
    fallback = RecordingScenarioProvider(
        _provider_response_sections(_full_roleplay_sections())
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-chat",
        display_name="Fallback Chat",
        capabilities=["chat", "fallback_marker"],
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="fallback",
        model_id="fallback-chat",
    )
    repositories.save_provider_model(
        provider="narrator",
        model_id="narrator-fallback-chat",
        display_name="Narrator Fallback Chat",
        capabilities=["chat", "fallback_marker"],
    )
    repositories.set_model_preference(
        task="narrator_fallback",
        provider="narrator",
        model_id="narrator-fallback-chat",
    )
    service = ScenarioService(
        repositories=repositories,
        provider=primary,
        provider_name="openrouter",
        model_id="scenario-drafter",
        providers={"openrouter": primary, "fallback": fallback},
    )

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.FULL_ROLEPLAY,
            seed="A volcanic border keep cut off by ash storms.",
        )
    )

    assert _sections(draft) == _full_roleplay_sections()
    assert len(primary.chat_requests) == len(FULL_ROLEPLAY_SECTION_IDS)
    assert len(fallback.chat_requests) == len(FULL_ROLEPLAY_SECTION_IDS)
    assert {request.provider for request in fallback.chat_requests} == {"fallback"}
    assert {request.model_id for request in fallback.chat_requests} == {
        "fallback-chat"
    }
    assert primary.structured_output_requests == []
    assert {
        (request.provider, request.model_id)
        for request in fallback.structured_output_requests
    } == {("fallback", "fallback-chat")}


def test_generate_draft_reports_generating_and_completed_progress(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        _provider_response_sections(_dating_sim_sections())
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )
    progress_updates: list[ScenarioGenerationProgress] = []

    draft = asyncio.run(
        service.generate_draft(
            scenario_type=ScenarioType.DATING_SIM,
            seed="A seaside summer academy.",
            progress_callback=progress_updates.append,
        )
    )

    assert _sections(draft) == _dating_sim_sections()
    assert [
        (progress.section_id, progress.status, progress.completed_count)
        for progress in progress_updates[:4]
    ] == [
        ("title", "generating", 0),
        ("title", "completed", 1),
        ("premise", "generating", 1),
        ("premise", "completed", 2),
    ]
    assert [
        (progress.section_id, progress.status, progress.completed_count)
        for progress in progress_updates[-2:]
    ] == [
        ("opening_message", "generating", len(DATING_SIM_SECTION_IDS) - 1),
        ("opening_message", "completed", len(DATING_SIM_SECTION_IDS)),
    ]
    assert all(
        progress.scenario_type is ScenarioType.DATING_SIM
        for progress in progress_updates
    )
    assert all(
        progress.total_count == len(DATING_SIM_SECTION_IDS)
        for progress in progress_updates
    )
    assert dict(progress_updates[0].completed_sections) == {}
    assert dict(progress_updates[1].completed_sections) == {
        "title": "Saltwind Hearts"
    }
    assert dict(progress_updates[-1].completed_sections) == (
        _dating_sim_sections()
    )


def test_generate_draft_reports_progress_and_preserves_completed_sections_on_failure(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingScenarioProvider(
        {
            **_provider_response_sections(_full_roleplay_sections()),
            "premise": " \n\t ",
        }
    )
    service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="openrouter",
        model_id="scenario-drafter",
    )
    progress_updates: list[ScenarioGenerationProgress] = []

    with pytest.raises(ValueError):
        asyncio.run(
            service.generate_draft(
                scenario_type=ScenarioType.FULL_ROLEPLAY,
                seed="A volcanic border keep cut off by ash storms.",
                progress_callback=progress_updates.append,
            )
        )

    assert [
        (progress.section_id, progress.status, progress.completed_count)
        for progress in progress_updates
    ] == [
        ("title", "generating", 0),
        ("title", "completed", 1),
        ("premise", "generating", 1),
        ("premise", "failed", 1),
    ]
    assert all(
        progress.scenario_type is ScenarioType.FULL_ROLEPLAY
        for progress in progress_updates
    )
    assert all(
        progress.total_count == len(FULL_ROLEPLAY_SECTION_IDS)
        for progress in progress_updates
    )
    assert dict(progress_updates[0].completed_sections) == {}
    completed_after_title = {"title": "Ashfall Keep"}
    assert dict(progress_updates[1].completed_sections) == completed_after_title
    assert dict(progress_updates[2].completed_sections) == completed_after_title
    assert dict(progress_updates[3].completed_sections) == completed_after_title
    assert "empty" in progress_updates[3].error.lower()
    assert "json" not in progress_updates[3].error.lower()


def test_apply_edits_updates_draft_sections_before_save(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_full_roleplay_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )
    draft = ScenarioDraft(
        type=ScenarioType.FULL_ROLEPLAY,
        sections=_full_roleplay_sections(),
    )

    edited = service.apply_edits(
        draft,
        {
            "premise": "The keep has one night before the ash gates fail.",
            "opening_message": "The first ember lands on your gauntlet.",
        },
    )

    assert _sections(draft)["premise"] == "Ashfall Keep is isolated by firestorms."
    assert _sections(edited)["premise"] == (
        "The keep has one night before the ash gates fail."
    )
    assert _sections(edited)["opening_message"] == (
        "The first ember lands on your gauntlet."
    )
    assert _sections(edited)["title"] == "Ashfall Keep"


def test_scenario_draft_sections_are_defensively_copied_and_immutable() -> None:
    source_sections = _full_roleplay_sections()

    draft = ScenarioDraft(
        type=ScenarioType.FULL_ROLEPLAY,
        sections=source_sections,
    )
    source_sections["title"] = "Mutated Outside Draft"

    assert draft.sections["title"] == "Ashfall Keep"
    with pytest.raises(TypeError):
        cast(MutableMapping[str, str], draft.sections)["title"] = "Mutated In Draft"


def test_save_draft_persists_scenario_content_json_and_returns_id(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_full_roleplay_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )
    draft = ScenarioDraft(
        type=ScenarioType.FULL_ROLEPLAY,
        sections={
            **_full_roleplay_sections(),
            "premise": "The keep has one night before the ash gates fail.",
        },
    )

    scenario_id = service.save_draft(draft)

    assert isinstance(scenario_id, str)
    row = repositories.connection.execute(
        """
        SELECT id, type, title, premise, player_role, content_json
        FROM scenarios
        WHERE id = ?
        """,
        (scenario_id,),
    ).fetchone()

    assert row is not None
    assert row["id"] == scenario_id
    assert row["type"] == "full_roleplay"
    assert row["title"] == "Ashfall Keep"
    assert row["premise"] == "The keep has one night before the ash gates fail."
    assert row["player_role"] == "The last signal warden"
    assert json.loads(row["content_json"]) == {
        **_full_roleplay_sections(),
        "premise": "The keep has one night before the ash gates fail.",
    }


def test_save_draft_persists_explicit_character_starters(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_dating_sim_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )
    draft = ScenarioDraft(
        type=ScenarioType.DATING_SIM,
        sections=_dating_sim_sections(),
        character_starters=(
            ScenarioCharacterStarter(
                name="Mika Arai",
                role="Student council president",
                known_state="Mika runs the festival schedule.",
                met=False,
            ),
        ),
    )

    scenario_id = service.save_draft(draft)

    scenario = repositories.get_scenario(scenario_id)
    assert scenario is not None
    content = json.loads(scenario.content_json)
    [starter] = content["character_starters"]
    assert starter["name"] == "Mika Arai"
    assert starter["role"] == "Student council president"
    assert starter["known_state"] == "Mika runs the festival schedule."
    assert starter["met"] is False


def test_save_hybrid_draft_persists_primary_type_and_genre_metadata(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_science_fiction_roleplay_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )
    sections = {
        **_science_fiction_roleplay_sections(),
        **{
            "player_character_profile": (
                "Ren is a transfer courier balancing mission duty and romance."
            ),
        },
    }
    draft = ScenarioDraft(
        type=ScenarioType.SCIENCE_FICTION_ROLEPLAY,
        scenario_types=(
            ScenarioType.SCIENCE_FICTION_ROLEPLAY,
            ScenarioType.DATING_SIM,
        ),
        sections=sections,
    )

    scenario_id = service.save_draft(draft)

    row = repositories.connection.execute(
        """
        SELECT type, content_json
        FROM scenarios
        WHERE id = ?
        """,
        (scenario_id,),
    ).fetchone()
    assert row is not None
    content = json.loads(row["content_json"])
    assert row["type"] == "science_fiction_roleplay"
    assert content["_scenario_genres"] == [
        "science_fiction_roleplay",
        "dating_sim",
    ]
    assert content["technology_level"] == sections["technology_level"]
    assert content["player_character_profile"] == sections["player_character_profile"]


def test_save_draft_folds_legacy_starting_scene_into_opening_message(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_full_roleplay_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )
    draft = ScenarioDraft(
        type=ScenarioType.FULL_ROLEPLAY,
        sections={
            **_full_roleplay_sections(),
            "starting_scene": "Ash falls over the gatehouse.",
        },
    )

    scenario_id = service.save_draft(draft)

    scenario = repositories.get_scenario(scenario_id)
    assert scenario is not None
    content = json.loads(scenario.content_json)
    assert "starting_scene" not in content
    assert content["opening_message"] == (
        "Ash falls over the gatehouse.\n\n"
        "The tower bell cracks once, then goes silent."
    )


def test_save_draft_accepts_legacy_full_roleplay_world_sections(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_detailed_full_roleplay_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )
    draft = ScenarioDraft(
        type=ScenarioType.FULL_ROLEPLAY,
        sections=_detailed_full_roleplay_sections(),
    )

    scenario_id = service.save_draft(draft)

    scenario = repositories.get_scenario(scenario_id)
    assert scenario is not None
    assert json.loads(scenario.content_json)["worldbuilding"] == (
        "Signal towers bind the border marches together."
    )


def test_save_draft_omits_deprecated_loss_condition_metadata(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_full_roleplay_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )
    draft = ScenarioDraft(
        type=ScenarioType.FULL_ROLEPLAY,
        sections=_full_roleplay_sections(),
        metadata={
            "provider": "fixture",
            "loss_conditions": [
                {
                    "name": "Beacon collapse",
                    "description": "The scenario ends if the beacon falls.",
                }
            ],
        },
    )

    scenario_id = service.save_draft(draft)

    scenario = repositories.get_scenario(scenario_id)
    assert scenario is not None
    content = json.loads(scenario.content_json)
    assert content["_source"] == {"provider": "fixture"}


def test_save_draft_persists_blank_player_character_name_when_missing(
    repositories: PersistenceRepositories,
) -> None:
    scenario_type = ScenarioType.FULL_ROLEPLAY
    section_factory = _full_roleplay_sections
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(section_factory()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )
    sections = section_factory()
    sections.pop("player_character_name")
    draft = ScenarioDraft(type=scenario_type, sections=sections)

    scenario_id = service.save_draft(draft)

    scenario = repositories.get_scenario(scenario_id)
    assert scenario is not None
    assert draft.player_character_name == ""
    assert json.loads(scenario.content_json)["player_character_name"] == ""


def test_save_draft_persists_fantasy_science_fiction_and_survival_sections(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_fantasy_roleplay_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    fantasy_id = service.save_draft(
        ScenarioDraft(
            type=ScenarioType.FANTASY_ROLEPLAY,
            sections=_fantasy_roleplay_sections(),
        )
    )
    science_id = service.save_draft(
        ScenarioDraft(
            type=ScenarioType.SCIENCE_FICTION_ROLEPLAY,
            sections=_science_fiction_roleplay_sections(),
        )
    )
    survival_id = service.save_draft(
        ScenarioDraft(
            type=ScenarioType.SURVIVAL_EXPEDITION,
            sections=_survival_expedition_sections(),
        )
    )

    fantasy = repositories.get_scenario(fantasy_id)
    science = repositories.get_scenario(science_id)
    survival = repositories.get_scenario(survival_id)
    assert fantasy is not None
    assert science is not None
    assert survival is not None
    assert fantasy.type == "fantasy_roleplay"
    assert science.type == "science_fiction_roleplay"
    assert survival.type == "survival_expedition"
    fantasy_content = json.loads(fantasy.content_json)
    science_content = json.loads(science.content_json)
    survival_content = json.loads(survival.content_json)
    assert fantasy_content["magic_system"] == (
        "Oaths pull power from named moon shards and punish broken vows."
    )
    assert science_content["technology_level"] == (
        "Patchwork interstellar salvage tech with unreliable alien interfaces."
    )
    assert survival_content["resource_inventory"] == (
        "Food for nine days, water for five, medicine for three patients."
    )


def test_save_draft_persists_first_contact_exploration_sections(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_first_contact_exploration_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    scenario_id = service.save_draft(
        ScenarioDraft(
            type=ScenarioType.FIRST_CONTACT_EXPLORATION,
            sections=_first_contact_exploration_sections(),
        )
    )

    scenario = repositories.get_scenario(scenario_id)
    assert scenario is not None
    assert scenario.type == "first_contact_exploration"
    content = json.loads(scenario.content_json)
    assert content["mission_profile"] == (
        "Survey the subglacial ocean and make non-hostile contact if possible."
    )
    assert content["knowledge_state"] == (
        "Observed: repeating pressure-wave songs beneath the ice. Hypothesis: "
        "the cadence maps safe passages. Unknown: whether the singers know the "
        "crew is present."
    )
    assert content["translation_progress"] == (
        "Learned term: three descending pulses may mean open water. False "
        "assumption: louder pulses are threats. Confirmed: blue light flashes "
        "mark attention."
    )


def test_save_draft_persists_investigation_mystery_sections(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_investigation_mystery_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    scenario_id = service.save_draft(
        ScenarioDraft(
            type=ScenarioType.INVESTIGATION_MYSTERY,
            sections=_investigation_mystery_sections(),
        )
    )

    scenario = repositories.get_scenario(scenario_id)
    assert scenario is not None
    assert scenario.type == "investigation_mystery"
    content = json.loads(scenario.content_json)
    assert content["case_facts"] == (
        "Curator Elian Vale vanished from the sealed east gallery during a gala."
    )
    assert content["clues"] == (
        "Broken display dust found outside the gallery door; undiscovered. "
        "Watch log gap from 9:10 to 9:18; reliable and tied to Sera's alibi."
    )
    assert content["hidden_truth"] == (
        "Sera staged the vanishing to hide a smuggling ledger in the restoration lift."
    )


def test_save_draft_persists_time_loop_sections(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_time_loop_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    scenario_id = service.save_draft(
        ScenarioDraft(
            type=ScenarioType.TIME_LOOP,
            sections=_time_loop_sections(),
        )
    )

    scenario = repositories.get_scenario(scenario_id)
    assert scenario is not None
    assert scenario.type == "time_loop"
    content = json.loads(scenario.content_json)
    assert content["loop_duration"] == (
        "Twenty-four hours, dawn festival bell to dawn festival bell."
    )
    assert content["persistent_knowledge"] == (
        "Player/meta knowledge persists: bell tower access code, Mira's warning, "
        "and the flooded tunnel route."
    )
    assert content["npc_memory_rules"] == (
        "NPCs reset to baseline memories unless an explicit persistence exception "
        "states otherwise."
    )


def test_save_draft_persists_political_intrigue_sections(
    repositories: PersistenceRepositories,
) -> None:
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(_political_intrigue_sections()),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    scenario_id = service.save_draft(
        ScenarioDraft(
            type=ScenarioType.POLITICAL_INTRIGUE,
            sections=_political_intrigue_sections(),
        )
    )

    scenario = repositories.get_scenario(scenario_id)
    assert scenario is not None
    assert scenario.type == "political_intrigue"
    content = json.loads(scenario.content_json)
    assert content["political_factions"] == (
        "Guilds, Old Families, and dock unions compete for harbor control."
    )
    assert content["obligations_and_favors"] == (
        "Guildmaster Orro owes Mara one public endorsement."
    )
    assert content["political_pressure"] == (
        "The midnight no-confidence vote proceeds unless Mara delays the session."
    )


@pytest.mark.parametrize(
    ("scenario_type", "section_factory", "expected_field"),
    [
        (
            ScenarioType.SETTLEMENT_BUILDER,
            "_settlement_builder_sections",
            "projects_and_facilities",
        ),
        (
            ScenarioType.MONSTER_HUNT_BOUNTY,
            "_monster_hunt_bounty_sections",
            "leads_and_clues",
        ),
        (
            ScenarioType.ROAD_TRIP_PILGRIMAGE,
            "_road_trip_pilgrimage_sections",
            "journey_progress",
        ),
        (
            ScenarioType.MERCHANT_TRADE_ROUTE,
            "_merchant_trade_route_sections",
            "contracts_and_debts",
        ),
    ],
)
def test_save_draft_persists_management_template_sections(
    repositories: PersistenceRepositories,
    scenario_type: ScenarioType,
    section_factory: str,
    expected_field: str,
) -> None:
    section_factory_func = cast(
        Callable[[], dict[str, str]],
        globals()[section_factory],
    )
    sections = section_factory_func()
    service = ScenarioService(
        repositories=repositories,
        provider=RecordingScenarioProvider(sections),
        provider_name="openrouter",
        model_id="scenario-drafter",
    )

    scenario_id = service.save_draft(
        ScenarioDraft(
            type=scenario_type,
            sections=sections,
        )
    )

    scenario = repositories.get_scenario(scenario_id)
    assert scenario is not None
    assert scenario.type == scenario_type.value
    content = json.loads(scenario.content_json)
    assert content[expected_field] == sections[expected_field]


def _full_roleplay_sections() -> dict[str, str]:
    return {
        "title": "Ashfall Keep",
        "premise": "Ashfall Keep is isolated by firestorms.",
        "player_character_name": "Mara Voss",
        "player_role": "The last signal warden",
        "tone_genre": "Tense heroic fantasy survival.",
        "opening_message": "The tower bell cracks once, then goes silent.",
    }


def _detailed_full_roleplay_sections() -> dict[str, str]:
    return {
        **_full_roleplay_sections(),
        "worldbuilding": "Signal towers bind the border marches together.",
        "lore": "The first beacon was lit from a fallen star.",
        "locations": "Gatehouse, cinder chapel, broken east stair.",
        "factions": "Wardens, ash cult, refugee caravan.",
    }


def _dating_sim_sections() -> dict[str, str]:
    return {
        "title": "Saltwind Hearts",
        "premise": "A transfer student enters a seaside academy before festival week.",
        "player_character_name": "Ren Takahashi",
        "player_character_profile": (
            "Ren is a thoughtful male transfer student trying to choose which "
            "club and which future will define his last summer."
        ),
        "player_role": "The central player character and romantic lead.",
        "tone_genre": (
            "Warm romantic drama with comedy, longing, and school-life stakes."
        ),
        "opening_message": "The station doors open onto salt air and festival banners.",
    }


def _fantasy_roleplay_sections() -> dict[str, str]:
    return {
        "title": "Moon-Oath Kingdom",
        "premise": "A fallen moon has turned every royal vow into dangerous magic.",
        "player_character_name": "Seren Vale",
        "player_role": "A disgraced knight carrying the last unbroken oath.",
        "magic_system": (
            "Oaths pull power from named moon shards and punish broken vows."
        ),
        "realms_and_places": (
            "The shattered crownlands, moon-glass roads, and the drowned abbey."
        ),
        "factions_and_orders": (
            "Moon-priests, oathbreakers, royal remnants, and wyvern riders."
        ),
        "myths_and_creatures": (
            "Wyverns nest in fallen craters and saints whisper from moonstone."
        ),
        "quest_stakes": (
            "Restore the crown oath before rival orders bind the realm by force."
        ),
        "tone_genre": "Mythic fantasy with court intrigue and perilous vows.",
        "opening_message": "The moon shard in your gauntlet speaks your old name.",
    }


def _science_fiction_roleplay_sections() -> dict[str, str]:
    return {
        "title": "Silent Ring Salvage",
        "premise": "A salvage crew finds a dead megastructure waking up around them.",
        "player_character_name": "Mara Quell",
        "player_role": "The crew's systems linguist and reluctant first-contact lead.",
        "technology_level": (
            "Patchwork interstellar salvage tech with unreliable alien interfaces."
        ),
        "setting_scope": (
            "One derelict orbital ring, nearby habitats, and deep-space law."
        ),
        "species_and_intelligences": (
            "Humans, vat-grown navigators, and an alien station mind."
        ),
        "factions_and_institutions": (
            "Salvage unions, corporate recovery fleets, and quarantine marshals."
        ),
        "mission_stakes": (
            "Decode the ring before the quarantine fleet sterilizes the site."
        ),
        "tone_genre": "Tense first-contact science fiction with hard choices.",
        "opening_message": "The ring answers your scan with your own childhood song.",
    }


def _first_contact_exploration_sections() -> dict[str, str]:
    return {
        "title": "Songs Under Europa",
        "premise": (
            "A survey crew finds patterned signals in a moon's hidden ocean while "
            "their habitat loses heat."
        ),
        "player_character_name": "Dr. Mara Voss",
        "player_role": "Mission linguist and acting contact lead.",
        "mission_profile": (
            "Survey the subglacial ocean and make non-hostile contact if possible."
        ),
        "ship_or_base_status": (
            "Habitat Kestrel has 42 hours of stable heat and one damaged drill."
        ),
        "exploration_target": (
            "A black-water cavern beneath the ice shelf with blue bioluminescent vents."
        ),
        "unknown_intelligence": (
            "An unseen whale-like intelligence answers sonar with structured "
            "pressure songs."
        ),
        "knowledge_state": (
            "Observed: repeating pressure-wave songs beneath the ice. Hypothesis: "
            "the cadence maps safe passages. Unknown: whether the singers know the "
            "crew is present."
        ),
        "translation_progress": (
            "Learned term: three descending pulses may mean open water. False "
            "assumption: louder pulses are threats. Confirmed: blue light flashes "
            "mark attention."
        ),
        "discoveries_and_samples": (
            "Ice cores contain living metallic spores; no sample may enter the habitat "
            "until quarantine clears."
        ),
        "hazards_and_escalation": (
            "Thermal fissures are spreading, contamination protocols are strained, "
            "and the rescue window closes in two days."
        ),
        "tone_genre": "Hopeful, tense exploration science fiction.",
        "opening_message": (
            "Blue light pulses under the ice before the sonar speaks back."
        ),
    }


def _survival_expedition_sections() -> dict[str, str]:
    return {
        "title": "Whiteout Pass",
        "premise": "A relief party crosses a storm-locked mountain pass.",
        "player_character_name": "Mara Voss",
        "player_role": "The expedition lead responsible for everyone surviving.",
        "expedition_goal": "Reach Northwatch before fever spreads through camp.",
        "route_options": "The cliff road is fast and exposed; the forest is slower.",
        "resource_inventory": (
            "Food for nine days, water for five, medicine for three patients."
        ),
        "environmental_conditions": "Late winter whiteout over ice-glazed slopes.",
        "hazards_and_events": "Avalanches, frostbite, wolf sign, and broken bridges.",
        "camp_status": "Two canvas tents, one cracked stove, and poor rest.",
        "travel_progress": "0 of 80 miles; retreat remains possible for one day.",
        "tone_genre": "Gritty expedition survival with hard logistical choices.",
        "opening_message": "Snow erases the last wagon tracks behind you.",
    }


def _investigation_mystery_sections() -> dict[str, str]:
    return {
        "title": "The Vanished Curator",
        "premise": "A public disappearance exposes a sealed museum conspiracy.",
        "player_character_name": "Inspector Mara Voss",
        "player_role": "The investigator assigned to reopen the impossible case.",
        "case_facts": (
            "Curator Elian Vale vanished from the sealed east gallery during a gala."
        ),
        "clues": (
            "Broken display dust found outside the gallery door; undiscovered. "
            "Watch log gap from 9:10 to 9:18; reliable and tied to Sera's alibi."
        ),
        "timeline": (
            "Public: gala toast at 9:00, alarm at 9:21. "
            "Hidden: lift moved at 9:12 and ledger was swapped at 9:15."
        ),
        "red_herrings": (
            "A bloody glove belongs to an old mannequin repair, not the culprit."
        ),
        "hidden_truth": (
            "Sera staged the vanishing to hide a smuggling ledger in the restoration "
            "lift."
        ),
        "case_status": "Unresolved; the player has only public facts.",
        "tone_genre": "Quiet investigative noir with careful clue continuity.",
        "opening_message": "Rain taps the museum glass as the east gallery unlocks.",
    }


def _heist_infiltration_sections() -> dict[str, str]:
    return {
        "title": "Skybank Treaty Job",
        "premise": (
            "A small crew must steal a sealed treaty from a floating bank before "
            "a rival house uses it to start a war."
        ),
        "player_character_name": "Mara Voss",
        "player_role": "Crew planner and face for the treaty extraction.",
        "target_location": (
            "A marble skybank with public galleries, private vault lifts, and "
            "storm moorings below."
        ),
        "objectives_and_stakes": (
            "Primary objective: recover the treaty. Optional objective: copy "
            "the blackmail ledger. Failure starts a border war."
        ),
        "intel_and_access": (
            "Known: guard shift changes at bell three and the lift code is split "
            "between two clerks. Unknown: whether the ledger is in the same vault."
        ),
        "security_model": (
            "Clockwork cameras, badge checkpoints, two warded locks, four guard "
            "patrols, and a silent alarm in the treaty case."
        ),
        "alert_and_heat": (
            "Suspicion starts low, alarm is inactive, and city heat will rise if "
            "witnesses identify the crew."
        ),
        "loadout_and_tools": (
            "Forged badges, one charm-disruptor, lockpicks, smoke pellets, and "
            "a rented storm skiff."
        ),
        "complications": (
            "A rival crew shadows the job and a surprise audit can close the "
            "vault early."
        ),
        "extraction_routes": (
            "Primary escape is the storm skiff; fallback routes are the service "
            "stairs or a public balcony drop."
        ),
        "aftermath": (
            "Clean success keeps heat low; partial success leaves the treaty "
            "safe but the crew hunted."
        ),
        "tone_genre": "Tense, system-agnostic caper with careful consequences.",
        "opening_message": "The skybank bell strikes three as the audit doors open.",
    }


def _political_intrigue_sections() -> dict[str, str]:
    return {
        "title": "Council of Ash",
        "premise": "A city council vote will decide who controls the harbor.",
        "player_character_name": "Mara Voss",
        "player_role": "Envoy holding the swing vote.",
        "political_arena": (
            "The harbor council chamber, public galleries, back corridors, and "
            "petition square outside."
        ),
        "political_factions": (
            "Guilds, Old Families, and dock unions compete for harbor control."
        ),
        "central_conflict": (
            "A midnight no-confidence vote can replace the regent and redirect "
            "the harbor charter."
        ),
        "secrets_and_leverage": (
            "Only Mara knows Orro moved missing silver through the old quay."
        ),
        "reputation_and_standing": (
            "Mara is trusted by reformers, doubted by the old houses, and watched "
            "by the guilds."
        ),
        "obligations_and_favors": (
            "Guildmaster Orro owes Mara one public endorsement."
        ),
        "alliances_and_rivalries": (
            "Reformers court Mara; old houses resist; dock unions can swing the "
            "public gallery."
        ),
        "event_calendar": "Dawn hearing, noon procession, dusk caucus, midnight vote.",
        "political_pressure": (
            "The midnight no-confidence vote proceeds unless Mara delays the session."
        ),
        "public_private_knowledge": (
            "The public knows the vote is close; only Mara knows Orro's favor and "
            "the missing silver route."
        ),
        "tone_genre": "Tense council intrigue with visible social consequences.",
        "opening_message": "The council bell rings as every gallery turns toward Mara.",
    }


def _time_loop_sections() -> dict[str, str]:
    return {
        "title": "Bellwether Day",
        "premise": "A harbor festival repeats until the drowned bell is saved.",
        "player_character_name": "Mara Voss",
        "player_role": "Archivist who notices the day repeating.",
        "loop_premise": (
            "The same festival day resets after the harbor bell sinks beneath the tide."
        ),
        "reset_trigger": "Reset occurs when the drowned bell tolls at midnight.",
        "loop_duration": "Twenty-four hours, dawn festival bell to dawn festival bell.",
        "starting_state": (
            "Mara wakes in the archive loft with a wet matchbook and no public alarm."
        ),
        "objective": "Prevent the bell from sinking and identify who sabotages it.",
        "failure_conditions": (
            "The bell sinks, Mara dies, or the day reaches midnight unresolved."
        ),
        "baseline_world_state": (
            "At dawn the harbor is intact, Mira is skeptical, the tower is locked, "
            "and the council has not evacuated the pier."
        ),
        "loop_schedule": (
            "09:00 parade forms, 12:00 tower inspection, 18:00 storm tide, "
            "23:45 sabotage window."
        ),
        "persistent_knowledge": (
            "Player/meta knowledge persists: bell tower access code, Mira's warning, "
            "and the flooded tunnel route."
        ),
        "persistence_exceptions": (
            "A salt mark on Mara's wrist and the wet matchbook persist across resets."
        ),
        "npc_memory_rules": (
            "NPCs reset to baseline memories unless an explicit persistence exception "
            "states otherwise."
        ),
        "current_loop_state": (
            "Loop 1, dawn phase, no deviations confirmed, prior-loop summary empty."
        ),
        "tone_genre": "Clockwork mystery with wistful coastal urgency.",
        "opening_message": "The same bell rings dawn again.",
    }


def _action_choice_sections() -> dict[str, str]:
    return {
        "title": "Library of Falling Doors",
        "premise": "Every shelf in the cliffside library opens onto a different fall.",
        "player_character_name": "Ily Ren",
        "player_role": "A courier carrying the only unburned index.",
        "tone_genre": "Bookish fantasy suspense with crisp chapter-like scenes.",
        "choice_style": (
            "Offer four concise, active choices with different risks: cautious, "
            "bold, social, and strange."
        ),
        "opening_message": (
            "The first shelf swings open and the sea wind reads your name."
        ),
    }


def _settlement_builder_sections() -> dict[str, str]:
    return {
        "title": "Hearthstone Landing",
        "premise": "A flood-struck river town must survive its first hard year.",
        "player_character_name": "Mara Vale",
        "player_role": "Elected settlement steward",
        "settlement_profile": (
            "Hearthstone Landing is a timber-and-stone river town founded after "
            "the old bridge collapsed."
        ),
        "resources_and_indicators": (
            "Food: low. Lumber: useful. Morale: fragile. Defenses: unfinished."
        ),
        "projects_and_facilities": (
            "Repair the palisade; build a flood gate; reopen the mill race."
        ),
        "threats_and_opportunities": (
            "Spring floods, hungry bandits, rival ferry tolls, and a generous "
            "grain compact."
        ),
        "calendar_and_deadlines": "Flood season begins in sixteen days.",
        "tone_genre": "Grounded community survival with political pressure.",
        "opening_message": "The river has risen another handspan overnight.",
    }


def _monster_hunt_bounty_sections() -> dict[str, str]:
    return {
        "title": "The Thornback Contract",
        "premise": "A bounty crew hunts a beast that learns from every failed trap.",
        "player_character_name": "Ira Voss",
        "player_role": "Licensed monster tracker",
        "hunt_profile": (
            "Find the Thornback before the harvest road closes and the bounty "
            "expires."
        ),
        "target_profile": (
            "The Thornback is armored, avoids firelight, and may be guarding "
            "something under the old orchard."
        ),
        "leads_and_clues": (
            "Three-toed tracks at Mill Creek; blue sap on broken arrows; one "
            "survivor heard bells."
        ),
        "hunt_locations": "Mill Creek, the old orchard, and the collapsed toll road.",
        "preparation_state": "Silver wire, oil snares, two borrowed hounds, one debt.",
        "hunt_status": "Unresolved; target wounded but adapting.",
        "tone_genre": "Tense investigative wilderness hunt.",
        "opening_message": "The newest tracks circle your camp twice.",
    }


def _road_trip_pilgrimage_sections() -> dict[str, str]:
    return {
        "title": "Road to Saint Orra",
        "premise": "A divided traveling party must reach the shrine before midsummer.",
        "player_character_name": "Nell Aran",
        "player_role": "Pilgrim guide and reluctant mediator",
        "journey_profile": (
            "Carry a cracked bell relic to Saint Orra's shrine before midsummer."
        ),
        "route_and_stops": (
            "Salt road to Lantern Ford, then Crow Market, then the hill shrine."
        ),
        "transport_and_supplies": (
            "One wagon, two mules, six days of oats, little coin."
        ),
        "recurring_pressures": "Border patrols, summer storms, and a silent pursuer.",
        "relationship_threads": "Tom doubts Sera; the cousins blame each other.",
        "journey_progress": (
            "Current leg: day one to Lantern Ford; destination distant."
        ),
        "tone_genre": "Warm, weary travel drama with spiritual tension.",
        "opening_message": "The shrine road starts where the city stones end.",
    }


def _merchant_trade_route_sections() -> dict[str, str]:
    return {
        "title": "Ledger Road",
        "premise": "A caravan must turn debt into profit across dangerous markets.",
        "player_character_name": "Mara Den",
        "player_role": "Caravan factor with the final signature",
        "trade_profile": "Run cedar oil and glassware from Kesh Gate to Red Harbor.",
        "cargo_inventory": "Cedar oil: 20 jars. Glassware: 8 crates. Spare axle: 1.",
        "markets_and_stops": (
            "Kesh Gate overpays for medicine; Red Harbor needs oil; Dustwell "
            "has cheap fodder."
        ),
        "contracts_and_debts": (
            "Deliver ten jars to Red Harbor in twelve days or double the debt."
        ),
        "route_hazards": "Tariff patrols, bridge bandits, summer storms, and rivals.",
        "profit_and_loss": "Current margin is thin; one lost crate erases profit.",
        "tone_genre": "Economy-lite caravan drama with hard bargains.",
        "opening_message": "The creditor stamps the contract before the ink dries.",
    }


def _provider_response_sections(sections: dict[str, str]) -> dict[str, str]:
    return {
        section_id: f"\n  {section_body}  \t"
        for section_id, section_body in sections.items()
    }


def _sections(draft: ScenarioDraft) -> dict[str, str]:
    return dict(draft.sections)


def _request_text(request: ChatRequest) -> str:
    return "\n".join(message.body for message in request.messages)


def _requested_scenario_section(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Requested field: "):
            return line.removeprefix("Requested field: ").replace(" ", "_")
    raise AssertionError(f"Prompt did not include requested field: {prompt}")


def _requests_include_prior_section_context(
    requests: list[ChatRequest],
    sections: dict[str, str],
) -> bool:
    section_ids = tuple(sections)
    return all(
        all(
            sections[section_id] in _request_text(request)
            for section_id in prior_ids
        )
        for request, prior_ids in (
            (request, section_ids[:request_index])
            for request_index, request in enumerate(requests)
        )
    )
