from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import (
    ContextObservationRecord,
    ContextSourceRecord,
    ContextSourceSearchHit,
    ContextUpdateSuggestionRecord,
    MemoryRecord,
    MessageRecord,
    SaveRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services import context_search_service as context_search_module
from bragi.services.agentic_context import ContextCurationService, CurationDecision
from bragi.services.context_search_service import (
    RECENT_MESSAGE_CANDIDATE_LIMIT,
    ContextSearchResult,
    ContextSearchService,
    _balanced_script_terms,
    _bounded_context_query_terms,
    _context_selection_instruction,
    _meaningful_terms,
    _memory_provenance_visible_to_present_characters,
)
from bragi.services.continuity_index_service import ContinuityIndexService
from bragi.services.knowledge_boundary import ScopedTargets
from bragi.services.narration_context import load_narration_context_snapshot
from bragi.services.scenario_canon import scenario_canon_is_current


class RecordingStructuredContextProvider:
    provider_name = "fake"

    def __init__(
        self,
        response_data: dict[str, object] | None = None,
        *,
        expansion_data: dict[str, object] | None = None,
    ) -> None:
        self.response_data = response_data or {"selections": []}
        self.expansion_data = expansion_data or {
            "terms": [],
            "phrases": [],
            "entity_ids": [],
        }
        self.chat_requests: list[ChatRequest] = []
        self.structured_output_requests: list[StructuredOutputRequest] = []
        self.expansion_requests: list[StructuredOutputRequest] = []

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
                model_id="fake-context",
                display_name="Fake Context",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("context search must not call normal chat")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "context_retrieval_expansion":
            self.expansion_requests.append(request)
            return StructuredOutputResponse(
                data=self.expansion_data,
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 3},
            )
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data=self.response_data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 13},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("context search must not request image generation")


class MutatingStructuredContextProvider(RecordingStructuredContextProvider):
    def __init__(
        self,
        response_data: dict[str, object],
        *,
        after_selection: Callable[[], None],
    ) -> None:
        super().__init__(response_data)
        self.after_selection = after_selection

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        response = await super().generate_structured_output(request)
        if request.schema_name != "context_retrieval_expansion":
            self.after_selection()
        return response


class ScriptedContextCurator:
    def __init__(self, decisions: tuple[CurationDecision, ...]) -> None:
        self.decisions = decisions
        self.requests: list[tuple[str, tuple[ContextObservationRecord, ...]]] = []

    async def curate(
        self,
        *,
        save_id: str,
        observations: tuple[ContextObservationRecord, ...],
    ) -> tuple[CurationDecision, ...]:
        self.requests.append((save_id, observations))
        return self.decisions


class FailingStructuredContextProvider(RecordingStructuredContextProvider):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "context_retrieval_expansion":
            return await super().generate_structured_output(request)
        self.structured_output_requests.append(request)
        raise self.error


class FallbackStructuredContextProvider(RecordingStructuredContextProvider):
    provider_name = "fallback"

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fallback-structured",
                display_name="Fallback Structured",
                capabilities=frozenset(
                    {
                        ProviderCapability.STRUCTURED_OUTPUT,
                        ProviderCapability.STRUCTURED_OUTPUT,
                    }
                ),
            )
        ]


class SequenceToolContextProvider(RecordingStructuredContextProvider):
    def __init__(
        self,
        *,
        responses: list[tuple[ProviderToolCall, ...] | Exception],
        expansion_data: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.responses = responses
        self.expansion_data = expansion_data or {
            "terms": [],
            "phrases": [],
            "entity_ids": [],
        }
        self.tool_expansion_requests: list[ToolCallRequest] = []
        self.tool_call_requests: list[ToolCallRequest] = []

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-context",
                display_name="Fake Context",
                capabilities=frozenset(
                    {
                        ProviderCapability.STRUCTURED_OUTPUT,
                        ProviderCapability.TOOL_CALLING,
                    }
                ),
            )
        ]

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "context_retrieval_expansion":
            return await super().generate_structured_output(request)
        self.structured_output_requests.append(request)
        raise AssertionError("tool-capable context search should use tool calls")

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        if [tool.name for tool in request.tools] == ["expand_context_retrieval"]:
            self.tool_expansion_requests.append(request)
            return ToolCallResponse(
                tool_calls=(
                    ProviderToolCall(
                        id="call-expansion",
                        name="expand_context_retrieval",
                        arguments_json=json.dumps(self.expansion_data),
                    ),
                ),
                body="",
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 11},
            )
        self.tool_call_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected tool-call request")
        response = (
            self.responses[0]
            if len(self.responses) == 1
            else self.responses.pop(0)
        )
        if isinstance(response, Exception):
            raise response
        return ToolCallResponse(
            tool_calls=response,
            body="",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 11},
        )


class FallbackToolContextProvider(SequenceToolContextProvider):
    provider_name = "fallback"

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fallback-tools",
                display_name="Fallback Tools",
                capabilities=frozenset({ProviderCapability.TOOL_CALLING}),
            )
        ]


class ShapeSwitchToolContextProvider(SequenceToolContextProvider):
    """Tool-capable provider whose tool calls 404 but structured output works."""

    def __init__(
        self,
        *,
        response_data: dict[str, object] | None = None,
        expansion_data: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            responses=[
                ProviderError(
                    ProviderErrorCategory.MODEL_NOT_FOUND,
                    "model not found",
                    status_code=404,
                )
            ],
            expansion_data=expansion_data,
        )
        self.response_data = response_data or {"selections": []}

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "context_retrieval_expansion":
            return await super().generate_structured_output(request)
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data=self.response_data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 13},
        )


class ShapeFailingToolContextProvider(ShapeSwitchToolContextProvider):
    """Tool-capable provider whose tool and structured calls both 404."""

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "context_retrieval_expansion":
            return await super().generate_structured_output(request)
        self.structured_output_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model not found",
            status_code=404,
        )


class ScenarioSectionSelectingProvider(RecordingStructuredContextProvider):
    def __init__(self, *, selected_text: str) -> None:
        super().__init__()
        self.selected_text = selected_text

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "context_retrieval_expansion":
            return await super().generate_structured_output(request)
        self.structured_output_requests.append(request)
        selection_properties = request.schema["properties"]["selections"]["items"][
            "properties"
        ]
        assert "scenario_claim" in selection_properties["source_type"]["enum"]
        prompt = "\n".join(message.body for message in request.messages)
        source_id = _candidate_source_id(
            prompt,
            source_type="scenario_claim",
            expected_text=self.selected_text,
        )
        return StructuredOutputResponse(
            data={
                "selections": [
                    {
                        "source_type": "scenario_claim",
                        "source_id": source_id,
                        "relevance_note": "The tower details shape the next beat.",
                    },
                ]
            },
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )


class StateChangeSelectingProvider(RecordingStructuredContextProvider):
    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "context_retrieval_expansion":
            return await super().generate_structured_output(request)
        self.structured_output_requests.append(request)
        selection_properties = request.schema["properties"]["selections"]["items"][
            "properties"
        ]
        assert "state_change" in selection_properties["source_type"]["enum"]
        prompt = "\n".join(message.body for message in request.messages)
        assert "Duplicate staging value" not in prompt
        source_id = _candidate_source_id(
            prompt,
            source_type="state_change",
            expected_text="Moon Gate",
        )
        return StructuredOutputResponse(
            data={
                "selections": [
                    {
                        "source_type": "state_change",
                        "source_id": source_id,
                        "relevance_note": "The exit changed to the Moon Gate.",
                    },
                ]
            },
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 19},
        )


class MediaAssetSelectingProvider(RecordingStructuredContextProvider):
    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "context_retrieval_expansion":
            return await super().generate_structured_output(request)
        self.structured_output_requests.append(request)
        selection_properties = request.schema["properties"]["selections"]["items"][
            "properties"
        ]
        assert "media_asset" in selection_properties["source_type"]["enum"]
        prompt = "\n".join(message.body for message in request.messages)
        assert "gold bridge lights over black water" in prompt
        for omitted_text in (
            "old successful image prompt",
            "failed recent image prompt",
            "unlinked successful image prompt",
            "media/private/recent.png",
            "media/private/thumb-recent.png",
            "SECRET FULL PROMPT TAIL SHOULD NOT APPEAR",
            "raw bytes",
            "data:",
        ):
            assert omitted_text not in prompt
        source_id = _candidate_source_id(
            prompt,
            source_type="media_asset",
            expected_text="gold bridge lights over black water",
        )
        return StructuredOutputResponse(
            data={
                "selections": [
                    {
                        "source_type": "media_asset",
                        "source_id": source_id,
                        "relevance_note": "The recent scene image anchors the bridge.",
                    },
                ]
            },
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 23},
        )


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


class CountingPersistenceRepositories(PersistenceRepositories):
    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.list_counts: dict[str, int] = {}
        self.before_context_source_search: Callable[[], None] | None = None

    def list_world_state(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[WorldStateRecord]:
        self.list_counts["world_state"] = self.list_counts.get("world_state", 0) + 1
        return super().list_world_state(save_id, limit=limit)

    def list_memories(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        self.list_counts["memories"] = self.list_counts.get("memories", 0) + 1
        return super().list_memories(save_id, limit=limit)

    def list_summaries(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[SummaryRecord]:
        self.list_counts["summaries"] = self.list_counts.get("summaries", 0) + 1
        return super().list_summaries(save_id, limit=limit)

    def list_context_sources(
        self,
        save_id: str,
        *,
        source_type: str | None = None,
    ) -> list[ContextSourceRecord]:
        self.list_counts["context_sources"] = (
            self.list_counts.get("context_sources", 0) + 1
        )
        return super().list_context_sources(save_id, source_type=source_type)

    def list_context_update_suggestions(
        self,
        save_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[ContextUpdateSuggestionRecord]:
        self.list_counts["context_update_suggestions"] = (
            self.list_counts.get("context_update_suggestions", 0) + 1
        )
        return super().list_context_update_suggestions(
            save_id,
            status=status,
            limit=limit,
        )

    def search_context_sources(
        self,
        save_id: str,
        *,
        query_terms: set[str] | frozenset[str] | list[str] | tuple[str, ...],
        source_types: set[str] | frozenset[str] | list[str] | tuple[str, ...],
        limit: int,
        allowed_owner_names: set[str] | frozenset[str] | None = None,
        reference_character_ids: set[str] | frozenset[str] | None = None,
        visibility_character_ids: set[str] | frozenset[str] | None = None,
        current_scene_snapshot_id: str | None = None,
        current_scene_generation: int | None = None,
        current_turn_number: int | None = None,
        blocked_source_keys: set[tuple[str, str]] | frozenset[tuple[str, str]]
        | None = None,
        match_all: bool = False,
        exact_phrases: tuple[str, ...] = (),
        exact_identifiers: tuple[str, ...] = (),
    ) -> list[ContextSourceSearchHit]:
        callback = self.before_context_source_search
        self.before_context_source_search = None
        if callback is not None:
            callback()
        self.list_counts["context_source_searches"] = (
            self.list_counts.get("context_source_searches", 0) + 1
        )
        return super().search_context_sources(
            save_id,
            query_terms=query_terms,
            source_types=source_types,
            limit=limit,
            allowed_owner_names=allowed_owner_names,
            reference_character_ids=reference_character_ids,
            visibility_character_ids=visibility_character_ids,
            current_scene_snapshot_id=current_scene_snapshot_id,
            current_scene_generation=current_scene_generation,
            current_turn_number=current_turn_number,
            blocked_source_keys=blocked_source_keys,
            match_all=match_all,
            exact_phrases=exact_phrases,
            exact_identifiers=exact_identifiers,
        )

    def list_protected_context_sources(
        self,
        save_id: str,
        *,
        limit: int,
        allowed_owner_names: set[str] | frozenset[str] | None = None,
        reference_character_ids: set[str] | frozenset[str] | None = None,
        visibility_character_ids: set[str] | frozenset[str] | None = None,
        current_scene_snapshot_id: str | None = None,
        current_scene_generation: int | None = None,
        current_turn_number: int | None = None,
        blocked_source_keys: set[tuple[str, str]] | frozenset[tuple[str, str]]
        | None = None,
    ) -> list[ContextSourceRecord]:
        self.list_counts["protected_context_sources"] = (
            self.list_counts.get("protected_context_sources", 0) + 1
        )
        return super().list_protected_context_sources(
            save_id,
            limit=limit,
            allowed_owner_names=allowed_owner_names,
            reference_character_ids=reference_character_ids,
            visibility_character_ids=visibility_character_ids,
            current_scene_snapshot_id=current_scene_snapshot_id,
            current_scene_generation=current_scene_generation,
            current_turn_number=current_turn_number,
            blocked_source_keys=blocked_source_keys,
        )


def test_context_selection_instruction_prioritizes_mystery_context() -> None:
    instruction = _context_selection_instruction("investigation_mystery")

    lowered = instruction.casefold()
    assert "discovered clues" in lowered
    assert "known facts" in lowered
    assert "suspects" in lowered
    assert "public timeline" in lowered
    assert "case status" in lowered
    assert "hidden truth" in lowered
    assert "do not reveal hidden truth as player knowledge" in lowered


def test_meaningful_terms_preserve_single_character_cjk_entities() -> None:
    assert {"李は東門にいる", "李", "東門"} <= _meaningful_terms("李は東門にいる")
    assert _meaningful_terms("ask 李 now") == {"ask", "李", "now"}


def test_context_query_terms_keep_short_ascii_identifiers_and_mixed_scripts() -> None:
    assert _meaningful_terms("X") == {"x"}
    assert _meaningful_terms("A-7") == {"a", "7"}
    assert {"x", "a", "7"} <= _meaningful_terms(
        "I ask X whether A-7 opens the vault."
    )
    mixed_terms = {f"漢字{index:02d}" for index in range(70)}
    mixed_terms.add("moonstone")

    bounded = _balanced_script_terms(mixed_terms, limit=64)
    bounded_query = _bounded_context_query_terms(
        " ".join((*sorted(mixed_terms - {"moonstone"}), "moonstone"))
    )

    assert len(bounded) == 64
    assert "moonstone" in bounded
    assert "moonstone" in bounded_query


def test_context_query_terms_preserve_middle_terms_across_scripts() -> None:
    query = " ".join(
        (
            *(f"漢字{index:02d}" for index in range(40)),
            "moonstone",
            *(f"漢語{index:02d}" for index in range(40)),
        )
    )

    terms = _bounded_context_query_terms(query)

    assert len(terms) == 64
    assert "moonstone" in terms


def test_context_query_terms_preserve_middle_short_identifiers() -> None:
    query = " ".join(
        (
            *(f"verboseprefix{index:02d}" for index in range(50)),
            "A-7",
            *(f"verbosesuffix{index:02d}" for index in range(50)),
        )
    )

    terms = _bounded_context_query_terms(query)

    assert {"a", "7"} <= set(terms)


def test_exact_raw_candidates_match_short_identifier_in_long_fact() -> None:
    candidate = context_search_module._ContextCandidate(
        source_type="memory",
        source_id="memory-a7",
        text="Only A-7 opens the old river vault.",
    )

    selected = context_search_module._exact_raw_candidates(
        (candidate,),
        indexed_candidates=(),
        latest_player_message="A-7",
    )

    assert selected == (candidate,)


def test_exact_raw_candidates_match_natural_cjk_query() -> None:
    candidate = context_search_module._ContextCandidate(
        source_type="memory",
        source_id="memory-cjk",
        text="月石羅針盤は東の書庫を開く。",
    )

    selected = context_search_module._exact_raw_candidates(
        (candidate,),
        indexed_candidates=(),
        latest_player_message="月石羅針盤はどこ",
    )

    assert selected == (candidate,)


def test_exact_raw_candidates_match_distinctive_term_in_natural_query() -> None:
    candidate = context_search_module._ContextCandidate(
        source_type="memory",
        source_id="memory-moonstone",
        text="The moonstone opens the eastern vault.",
    )

    selected = context_search_module._exact_raw_candidates(
        (candidate,),
        indexed_candidates=(),
        latest_player_message="Where did we learn about the moonstone?",
    )

    assert selected == (candidate,)


def test_exact_raw_candidates_match_short_code_in_natural_query() -> None:
    candidate = context_search_module._ContextCandidate(
        source_type="memory",
        source_id="memory-a7-natural",
        text="Only A-7 opens the old river vault.",
    )

    selected = context_search_module._exact_raw_candidates(
        (candidate,),
        indexed_candidates=(),
        latest_player_message="Where did we learn about A-7?",
    )

    assert selected == (candidate,)


def test_structured_identifier_matching_is_bounded_and_keeps_query_edges() -> None:
    identifiers = context_search_module._bounded_structured_identifiers(
        " ".join(f"artifact-{index}" for index in range(100))
    )

    assert len(identifiers) == 16
    assert "artifact-0" in identifiers
    assert "artifact-99" in identifiers


def test_structured_identifier_matching_preserves_tail_identifier_after_bound() -> None:
    query = f"{'noise ' * context_search_module.MAX_CONTEXT_QUERY_CHARS} artifact-99"

    bounded = context_search_module._bounded_context_query_text(query)
    identifiers = context_search_module._bounded_structured_identifiers(query)

    assert len(bounded) == context_search_module.MAX_CONTEXT_QUERY_CHARS
    assert bounded.endswith("artifact-99")
    assert "artifact-99" in identifiers


def test_context_query_terms_bound_large_raw_inputs_and_keep_tail_terms() -> None:
    query = " ".join(f"noise{index:05d}" for index in range(2_000))
    query = f"{query} moonstone"

    terms = _bounded_context_query_terms(query)

    assert len(terms) <= 64
    assert "moonstone" in terms


def test_narration_snapshot_bounds_raw_context_records(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Archive",
        premise="A guarded archive.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The archive changes.",
    )
    for index in range(3):
        repositories.upsert_world_state(
            save_id=save.id,
            key=f"archive.fact.{index}",
            value={"index": index},
        )
        repositories.add_memory(
            save_id=save.id,
            body=f"Archive memory {index}.",
            tags=["archive"],
            source_message_id=message.id,
        )
        repositories.add_context_observation(
            save_id=save.id,
            observation_type="world_fact",
            claim=f"Archive observation {index}.",
            evidence_quote="The archive changes",
            source_message_ids=[message.id],
            scope="durable",
        )

    snapshot = load_narration_context_snapshot(
        repositories,
        save_id=save.id,
        raw_record_limit=2,
    )

    assert snapshot is not None
    assert len(snapshot.world_state) == 2
    assert len(snapshot.world_state_for_scope) == 2
    assert len(snapshot.memories) == 2
    assert len(snapshot.observations) == 2


def test_context_search_bounded_snapshot_blocks_legacy_plural_memory_edge(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Archive",
        premise="A guarded archive.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary")
    older_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Nira hears that the moonstone opens the cobalt ledger.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I ask Nira about the moonstone ledger.",
    )
    present = repositories.add_character(
        save_id=save.id,
        name="Nira",
        aliases=["Nira"],
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira studies the ledger room.",
        present_character_ids=[present.id],
    )
    hidden_memory = repositories.add_memory(
        save_id=save.id,
        body="The moonstone opens the cobalt ledger.",
        tags=["moonstone"],
        source_message_id=older_message.id,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=present.id,
        target_type="memories",
        target_id=hidden_memory.id,
        knowledge_state="does_not_know",
        acquisition_method="witnessed",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert hidden_memory.body not in prompt


def test_indexed_observation_hydration_recovers_older_bounded_record(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Archive",
        premise="A guarded archive.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The old moonstone key opens the archive.",
    )
    old_observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="world_fact",
        claim="The old moonstone key opens the archive.",
        evidence_quote="old moonstone key opens the archive",
        source_message_ids=[message.id],
        scope="durable",
        status="accepted",
    )
    marker = repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id=old_observation.id,
        title="Old moonstone key",
        body=old_observation.claim,
        metadata={
            "curation_action": "save_context",
            "observation_id": old_observation.id,
            "source_message_ids": [message.id],
        },
    )
    for index in range(513):
        repositories.add_context_observation(
            save_id=save.id,
            observation_type="world_fact",
            claim=f"New archive detail {index}.",
            evidence_quote="The old moonstone key opens the archive",
            source_message_ids=[message.id],
            scope="turn",
        )
    snapshot = load_narration_context_snapshot(
        repositories,
        save_id=save.id,
        raw_record_limit=512,
    )
    assert snapshot is not None
    assert old_observation.id not in {
        observation.id for observation in snapshot.observations
    }

    hydrated = context_search_module._observations_with_indexed_sources(
        repositories,
        save_id=save.id,
        observations=snapshot.observations,
        context_sources=(marker,),
    )

    assert old_observation.id in {observation.id for observation in hydrated}


def test_recent_visible_messages_are_not_starved_by_hidden_rows(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Archive",
        premise="A guarded archive.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary")
    character = repositories.add_character(save_id=save.id, name="Ilyra")
    accessible = [
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            body=f"Accessible event {index}.",
        )
        for index in range(6)
    ]
    hidden = [
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            body=f"Hidden event {index}.",
        )
        for index in range(64)
    ]
    for message in hidden:
        repositories.add_message_visibility(
            save_id=save.id,
            message_id=message.id,
            character_id=character.id,
            visibility="not_visible",
        )

    visible = repositories.list_recent_messages_visible_to_characters(
        save.id,
        character_ids={character.id},
        limit=6,
    )

    assert visible == accessible


def test_raw_memory_provenance_allows_one_independently_visible_observation(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Archive",
        premise="A guarded archive.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary")
    hidden = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The moonstone opens the archive.",
    )
    visible = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="A public witness confirms the moonstone opens the archive.",
    )
    character = repositories.add_character(save_id=save.id, name="Ilyra")
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden.id,
        character_id=character.id,
        visibility="not_visible",
    )
    observations = [
        repositories.add_context_observation(
            save_id=save.id,
            observation_type="character_fact",
            claim="The moonstone opens the archive.",
            evidence_quote="moonstone opens the archive",
            source_message_ids=[message.id],
            scope="durable",
            status="accepted",
            metadata={
                "curation": {
                    "action": "durable_memory",
                    "memory_body": "The moonstone opens the archive.",
                }
            },
        )
        for message in (hidden, visible)
    ]
    memory = repositories.add_memory(
        save_id=save.id,
        body="The moonstone opens the archive.",
        tags=["archive"],
        source_message_ids=[hidden.id, visible.id],
        source_observation_ids=[observation.id for observation in observations],
    )

    assert _memory_provenance_visible_to_present_characters(
        memory,
        source_message_ids=tuple(memory.source_message_ids),
        observations_by_id={
            observation.id: observation for observation in observations
        },
        present_character_ids=frozenset({character.id}),
        message_visibility=repositories.list_message_visibility(save.id),
    )


def test_retired_character_interaction_type_has_no_context_specialization() -> None:
    instruction = _context_selection_instruction("character_interaction")

    lowered = instruction.casefold()
    assert "featured character" not in lowered
    assert "relationship dynamics" not in lowered
    assert "current interaction setup" not in lowered


def test_context_selection_instruction_prioritizes_survival_context() -> None:
    instruction = _context_selection_instruction("survival_expedition")

    lowered = instruction.casefold()
    assert "route progress" in lowered
    assert "resources and equipment" in lowered
    assert "party health or morale" in lowered
    assert "environmental conditions" in lowered
    assert "camp status" in lowered
    assert "unresolved survival threats" in lowered


def test_context_selection_instruction_prioritizes_time_loop_boundaries() -> None:
    instruction = _context_selection_instruction("time_loop")

    lowered = instruction.casefold()
    assert "active loop state" in lowered
    assert "reset rules" in lowered
    assert "persistent player/meta knowledge" in lowered
    assert "npc memory boundaries" in lowered
    assert "do not treat reset npcs as remembering prior loops" in lowered


def test_context_selection_instruction_prioritizes_political_intrigue_context() -> None:
    instruction = _context_selection_instruction("political_intrigue")

    lowered = instruction.casefold()
    assert "faction positions" in lowered
    assert "favors owed or held" in lowered
    assert "event calendars" in lowered
    assert "timed political pressure" in lowered
    assert "public knowledge" in lowered
    assert "private knowledge" in lowered
    assert "do not reveal secrets as player knowledge" in lowered


def test_context_search_uses_one_structured_selection_request_and_provider_order(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Bridge of Cinders",
        premise="A bridge remembers every oath broken on it.",
        player_role="Oathkeeper",
        content={"starting_scene": "Cinders drift over the bridge stones."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Crossing")
    prelude_messages = [
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Prelude ash patrol beat {index}.",
        )
        for index in range(25)
    ]
    older_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A silver bell rings beneath the bridge.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I cross the bridge and listen for the bell.",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Bridge of Cinders"},
        category="scene",
        source_message_id=older_message.id,
    )
    second_state = repositories.upsert_world_state(
        save_id=save.id,
        key="npc.bellkeeper",
        value={"name": "Orin", "mood": "watchful"},
        category="npc",
        source_message_id=older_message.id,
    )
    first_memory = repositories.add_memory(
        save_id=save.id,
        body="Mara distrusts bells that ring without wind.",
        tags=["bells", "suspicion"],
        source_message_id=older_message.id,
    )
    second_memory = repositories.add_memory(
        save_id=save.id,
        body="Mara promised Orin she would not break the bridge oath.",
        tags=["orin", "promise", "oath"],
        importance=0.9,
        source_message_id=older_message.id,
    )
    zero_overlap_memory = repositories.add_memory(
        save_id=save.id,
        body="A bakery ledger lists yesterday's rye inventory.",
        tags=["bakery", "ledger"],
        source_message_id=older_message.id,
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=prelude_messages[0].id,
        covers_message_end_id=prelude_messages[1].id,
        body="Mara reached the bridge after escaping the ash patrol.",
        provider="fake",
        model="fake-summary",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": second_memory.id,
                    "relevance_note": "The promise shapes Mara's next choice.",
                },
                {
                    "source_type": "world_state",
                    "source_id": second_state.id,
                    "relevance_note": "Orin is present and watching.",
                },
                {
                    "source_type": "memory",
                    "source_id": first_memory.id,
                    "relevance_note": "Windless bells are a known concern.",
                },
                {
                    "source_type": "world_state",
                    "source_id": state.id,
                    "relevance_note": "The current scene is the bridge.",
                },
                {
                    "source_type": "message",
                    "source_id": older_message.id,
                    "relevance_note": "The bell was introduced immediately before.",
                },
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert provider.chat_requests == []
    assert len(provider.structured_output_requests) == 1
    request = provider.structured_output_requests[0]
    assert request.provider == "fake"
    assert request.model_id == "fake-context"
    assert request.schema_name == "context_search_selection"
    selection_properties = request.schema["properties"]["selections"]["items"][
        "properties"
    ]
    selection_required = request.schema["properties"]["selections"]["items"]["required"]
    assert selection_required == [
        "source_type",
        "source_id",
        "relevance_note",
    ]
    assert "source_type" in selection_properties
    assert set(selection_properties["source_type"]["enum"]) == {
        "open_obligation",
        "world_state",
        "memory",
        "observation",
        "character_voice",
        "character_text_thread",
        "message",
        "scenario_section",
        "state_change",
        "media_asset",
        "scenario_claim",
    }
    assert "source_id" in selection_properties
    prompt = "\n".join(message.body for message in request.messages)
    assert "I cross the bridge and listen for the bell." in prompt
    assert "A silver bell rings beneath the bridge." in prompt
    assert summary.body not in prompt
    assert "A bakery ledger lists yesterday's rye inventory." not in prompt
    assert "strict JSON" not in prompt
    assert "selected_state" not in prompt

    assert [item.source_id for item in result.selected_state] == [
        second_state.id,
        state.id,
    ]
    assert [item.relevance_note for item in result.selected_state] == [
        "Orin is present and watching.",
        "The current scene is the bridge.",
    ]
    assert [item.source_id for item in result.selected_memories] == [
        second_memory.id,
        first_memory.id,
    ]
    assert [item.relevance_note for item in result.selected_memories] == [
        "The promise shapes Mara's next choice.",
        "Windless bells are a known concern.",
    ]
    assert zero_overlap_memory.id not in {
        item.source_id for item in result.selected_memories
    }
    assert result.selected_summaries == ()
    assert result.selected_recent_messages[0].source_type == "message"
    assert result.selected_recent_messages[0].source_id == older_message.id
    assert result.selected_recent_messages[0].relevance_note == (
        "The bell was introduced immediately before."
    )

    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert "Bridge of Cinders" in jobs[-1]["result_json"]
    assert "The promise shapes Mara's next choice." in jobs[-1]["result_json"]


def test_context_search_selects_context_observations(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    narrator_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="open_thread",
        claim="The silver bell may matter later.",
        evidence_quote="A silver bell rings beneath the bridge.",
        source_message_ids=[narrator_message.id],
        scope="save",
        status="accepted",
        confidence=0.84,
        tags=["bell"],
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "observation",
                    "source_id": observation.id,
                    "relevance_note": "The observation points to a future bell risk.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert [item.source_id for item in result.selected_observations] == [
        observation.id
    ]
    assert "silver bell" in result.selected_observations[0].text
    jobs = _context_search_jobs(repositories, save.id)
    job_result = json.loads(jobs[-1]["result_json"])
    assert job_result["selected_observations"][0]["source_id"] == (
        observation.id
    )


def test_context_search_only_offers_accepted_observations_as_canon(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    narrator_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    observations = {
        status: repositories.add_context_observation(
            save_id=save.id,
            observation_type="open_thread",
            claim=claim,
            evidence_quote="A silver bell rings beneath the bridge.",
            source_message_ids=[narrator_message.id],
            scope="save",
            status=status,
            confidence=0.8,
            tags=["bell"],
        )
        for status, claim in {
            "accepted": "The accepted ruby omen is settled canon.",
            "pending": "The pending jade omen awaits review.",
            "needs_confirmation": "The amber omen needs confirmation.",
            "discarded": "The discarded violet omen was rejected.",
        }.items()
    }
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "observation",
                    "source_id": observations["accepted"].id,
                    "relevance_note": "Only accepted observations are canon.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "The accepted ruby omen is settled canon." in prompt
    assert "The pending jade omen awaits review." not in prompt
    assert "The amber omen needs confirmation." not in prompt
    assert "The discarded violet omen was rejected." not in prompt
    assert [item.source_id for item in result.selected_observations] == [
        observations["accepted"].id
    ]
    job_result = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    diagnostics = job_result["diagnostics"]
    assert diagnostics["observation_status_counts"] == {
        "accepted": 1,
        "discarded": 1,
        "needs_confirmation": 1,
        "pending": 1,
    }
    assert diagnostics["included_observation_status_counts"] == {"accepted": 1}
    assert diagnostics["excluded_observation_status_counts"] == {
        "discarded": 1,
        "needs_confirmation": 1,
        "pending": 1,
    }


def test_context_search_uses_curated_observation_sources_without_raw_duplicate(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    omen_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The ruby omen means the bridge oath is fragile.",
    )
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="open_thread",
        claim="The ruby omen means the bridge oath is fragile.",
        evidence_quote="ruby omen means the bridge oath is fragile",
        source_message_ids=[omen_message.id],
        scope="save",
        status="pending",
        confidence=0.84,
        tags=["bell"],
    )
    curator = ScriptedContextCurator(
        (
            CurationDecision(
                observation_id=observation.id,
                action="save_context",
                reason="Future plot relevance.",
                confidence=0.84,
                context_title="Curated ruby omen",
                context_body="The ruby omen means the bridge oath is fragile.",
                tags=("bell",),
                grounding_status="entailed",
                supporting_evidence_quote=observation.evidence_quote,
                supporting_source_message_ids=(omen_message.id,),
            ),
        )
    )
    curation_result = asyncio.run(
        ContextCurationService(
            repositories=repositories,
            curator=curator,
        ).curate_pending(save.id)
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "observation",
                    "source_id": observation.id,
                    "relevance_note": "The curated observation is relevant.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert curation_result.accepted_count == 1
    updated_observation = repositories.get_context_observation(observation.id)
    assert updated_observation is not None
    assert updated_observation.status == "accepted"
    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert _candidate_source_id(
        prompt,
        source_type="observation",
        expected_text="Saved context",
    ) == observation.id
    assert "The ruby omen means the bridge oath is fragile." in prompt
    assert f"Evidence: {observation.evidence_quote}" not in prompt
    assert [item.source_id for item in result.selected_observations] == [
        observation.id
    ]
    assert "Saved context" in result.selected_observations[0].text
    job_result = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    diagnostics = job_result["diagnostics"]
    assert diagnostics["curated_observation_candidate_count"] == 1
    assert diagnostics["suppressed_raw_observation_count"] == 1


def test_context_search_does_not_raw_fallback_when_curated_observation_is_blocked(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    narrator_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="open_thread",
        claim="Lio-only raw observation should stay hidden.",
        evidence_quote="A silver bell rings beneath the bridge.",
        source_message_ids=[narrator_message.id],
        scope="save",
        status="accepted",
        confidence=0.84,
        tags=["bell"],
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id=observation.id,
        title="Lio-only curated observation",
        body="Lio knows the ruby omen belongs to the hidden archive.",
        metadata={
            "observation_id": observation.id,
            "observation_type": observation.observation_type,
            "source_message_ids": [narrator_message.id],
            "curation_action": "scene_scratch",
            "known_by": ["Lio"],
        },
    )
    provider = RecordingStructuredContextProvider({"selections": []})
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "Lio-only raw observation should stay hidden." not in prompt
    assert "Lio knows the ruby omen belongs to the hidden archive." not in prompt
    job_result = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    diagnostics = job_result["diagnostics"]
    assert diagnostics["curated_observation_candidate_count"] == 0
    assert diagnostics["suppressed_raw_observation_count"] == 0


def test_context_search_prefers_tool_calls_when_model_advertises_tool_calling(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.STRUCTURED_OUTPUT.value,
            ProviderCapability.TOOL_CALLING.value,
        ],
    )
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    state = repositories.list_world_state(save.id)[0]
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara distrusts bells that ring without wind.",
        tags=["bells", "suspicion"],
        source_message_id=source_message.id,
    )
    provider = SequenceToolContextProvider(
        responses=[
            (
                ProviderToolCall(
                    id="call-memory",
                    name="select_context_source",
                    arguments_json=json.dumps(
                        {
                            "source_id": memory.id,
                            "relevance_note": "Windless bells are a known concern.",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="call-state",
                    name="select_context_source",
                    arguments_json=json.dumps(
                        {
                            "source_id": state.id,
                            "source_type": "world_state",
                            "relevance_note": "The current scene is the bridge.",
                        }
                    ),
                ),
            )
        ]
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert provider.chat_requests == []
    assert provider.structured_output_requests == []
    assert len(provider.tool_call_requests) == 1
    request = provider.tool_call_requests[0]
    assert request.provider == "fake"
    assert request.model_id == "fake-context"
    assert [tool.name for tool in request.tools] == ["select_context_source"]
    tool_schema = request.tools[0].parameters
    assert tool_schema["additionalProperties"] is False
    assert tool_schema["required"] == ["source_id"]
    source_id_schema = tool_schema["properties"]["source_id"]
    assert memory.id in source_id_schema["enum"]
    prompt = "\n".join(message.body for message in request.messages)
    assert "Use the provided select_context_source tool" in prompt
    assert "strict JSON" not in prompt

    assert [item.source_id for item in result.selected_memories] == [memory.id]
    assert [item.relevance_note for item in result.selected_memories] == [
        "Windless bells are a known concern.",
    ]
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.selected_state[0].relevance_note == (
        "The current scene is the bridge."
    )


def test_context_search_tool_feedback_retains_accepted_calls(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    state = repositories.list_world_state(save.id)[0]
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara promised Orin she would not break the bridge oath.",
        tags=["orin", "promise", "oath"],
        source_message_id=source_message.id,
    )
    accepted_call = ProviderToolCall(
        id="call-memory",
        name="select_context_source",
        arguments_json=json.dumps(
            {
                "source_id": memory.id,
                "relevance_note": "The promise shapes Mara's next choice.",
            }
        ),
    )
    provider = SequenceToolContextProvider(
        responses=[
            (
                accepted_call,
                ProviderToolCall(
                    id="call-missing",
                    name="select_context_source",
                    arguments_json=json.dumps(
                        {
                            "source_id": "memory-that-is-not-a-candidate",
                            "relevance_note": "This id was not offered.",
                        }
                    ),
                ),
            ),
            (
                accepted_call,
                ProviderToolCall(
                    id="call-state",
                    name="select_context_source",
                    arguments_json=json.dumps(
                        {
                            "source_id": state.id,
                            "relevance_note": "The current scene is the bridge.",
                        }
                    ),
                ),
            ),
        ]
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(provider.tool_call_requests) == 2
    retry_messages = provider.tool_call_requests[1].messages
    assert any(
        message.role == "assistant" and message.tool_calls
        for message in retry_messages
    )
    tool_results = [
        json.loads(message.body)
        for message in retry_messages
        if message.role == "tool"
    ]
    assert {"status": "accepted", "message": "Accepted. Do not repeat this call."} in (
        tool_results
    )
    assert any(
        result["status"] == "error"
        and "memory-that-is-not-a-candidate" in result["error"]
        for result in tool_results
    )
    assert [item.source_id for item in result.selected_memories] == [memory.id]
    assert [item.source_id for item in result.selected_state] == [state.id]


def test_context_search_tool_feedback_exhaustion_preserves_accepted_calls(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara promised Orin she would not break the bridge oath.",
        tags=["orin", "promise", "oath"],
        source_message_id=source_message.id,
    )
    accepted_call = ProviderToolCall(
        id="call-memory",
        name="select_context_source",
        arguments_json=json.dumps(
            {
                "source_id": memory.id,
                "relevance_note": "The promise shapes Mara's next choice.",
            }
        ),
    )
    provider = SequenceToolContextProvider(
        responses=[
            (
                accepted_call,
                ProviderToolCall(
                    id=f"call-missing-{index}",
                    name="select_context_source",
                    arguments_json=json.dumps(
                        {
                            "source_id": f"memory-that-is-not-a-candidate-{index}",
                            "relevance_note": "This id was not offered.",
                        }
                    ),
                ),
            )
            for index in range(3)
        ]
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(provider.tool_call_requests) == 7
    assert [item.source_id for item in result.selected_memories] == [memory.id]
    assert [item.relevance_note for item in result.selected_memories] == [
        "The promise shapes Mara's next choice.",
    ]
    assert result.selected_state == ()
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"


def test_context_search_tool_feedback_exhaustion_without_accepted_calls_uses_fallback(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    state = repositories.list_world_state(save.id)[0]
    provider = SequenceToolContextProvider(
        responses=[
            (
                ProviderToolCall(
                    id=f"call-missing-{index}",
                    name="select_context_source",
                    arguments_json=json.dumps(
                        {
                            "source_id": "memory-that-is-not-a-candidate",
                            "relevance_note": "This id was not offered.",
                        }
                    ),
                ),
            )
            for index in range(3)
        ]
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(provider.tool_call_requests) == 7
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.selected_state[0].relevance_note == (
        "Selected by deterministic fallback after empty context selection."
    )
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"


def test_context_search_tool_no_selection_uses_continuity_floor(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    state = repositories.list_world_state(save.id)[0]
    provider = SequenceToolContextProvider(responses=[()])
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(provider.tool_call_requests) == 1
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.selected_state[0].relevance_note == (
        "Selected by deterministic fallback after empty context selection."
    )
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "deterministic_fallback"
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert jobs[-1]["result_json"] is not None
    job_result = json.loads(jobs[-1]["result_json"])
    assert job_result["retrieval_degraded"] is True
    assert job_result["retrieval_recovery"] == "deterministic_fallback"


def test_context_search_uses_tool_fallback_when_primary_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara distrusts bells that ring without wind.",
        tags=["bells"],
        source_message_id=source_message.id,
    )
    _configure_tool_fallback(repositories)
    primary = SequenceToolContextProvider(
        responses=[
            ProviderError(
                ProviderErrorCategory.MODEL_NOT_FOUND,
                "model not found",
                status_code=404,
            )
        ]
    )
    fallback = FallbackToolContextProvider(
        responses=[
            (
                ProviderToolCall(
                    id="fallback-memory",
                    name="select_context_source",
                    arguments_json=json.dumps(
                        {
                            "source_id": memory.id,
                            "relevance_note": "The bell concern matters now.",
                        }
                    ),
                ),
            )
        ]
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": primary, "fallback": fallback},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert fallback.tool_call_requests[0].provider == "fallback"
    assert fallback.tool_call_requests[0].model_id == "fallback-tools"
    assert [item.source_id for item in result.selected_memories] == [memory.id]
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "provider_fallback"
    assert [
        model.available
        for model in repositories.list_provider_models("fake")
        if model.model_id == "fake-context"
    ] == [True]
    job_result = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    assert job_result["retrieval_degraded"] is True
    assert job_result["retrieval_recovery"] == "provider_fallback"
    assert job_result["fallback_used"] is True
    assert job_result["fallback_provider"] == "fallback"
    assert job_result["fallback_model"] == "fallback-tools"
    assert job_result["error_category"] == ProviderErrorCategory.MODEL_NOT_FOUND.value
    assert job_result["http_status"] == 404


def test_context_search_uses_deterministic_fallback_when_tool_provider_fails(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    state = repositories.list_world_state(save.id)[0]
    provider = SequenceToolContextProvider(
        responses=[
            ProviderError(
                ProviderErrorCategory.RATE_LIMITED,
                "rate limited",
                status_code=429,
            )
        ]
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(provider.tool_call_requests) == 1
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.selected_state[0].relevance_note == (
        "Selected by deterministic fallback after empty context selection."
    )
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "deterministic_fallback"
    job_result = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    assert job_result["retrieval_degraded"] is True
    assert job_result["retrieval_recovery"] == "deterministic_fallback"
    assert job_result["fallback_used"] is False
    assert job_result["fallback_skipped_reason"] == "no_fallback_model"


def test_context_search_recovers_when_tool_fallback_model_missing(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    state = repositories.list_world_state(save.id)[0]
    _configure_tool_fallback(repositories)
    primary = ShapeSwitchToolContextProvider(
        response_data={
            "selections": [
                {
                    "source_type": "world_state",
                    "source_id": state.id,
                    "relevance_note": "Selected through the structured route.",
                }
            ]
        }
    )
    primary.responses = [
        ProviderError(
            ProviderErrorCategory.RATE_LIMITED,
            "rate limited",
            status_code=429,
        )
    ]
    fallback = FallbackToolContextProvider(
        responses=[
            ProviderError(
                ProviderErrorCategory.MODEL_NOT_FOUND,
                "fallback model missing",
                status_code=404,
            )
        ]
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": primary, "fallback": fallback},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert len(primary.structured_output_requests) == 1
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "shape_fallback"
    assert [
        model.available
        for model in repositories.list_provider_models("fallback")
        if model.model_id == "fallback-tools"
    ] == [True]
    job_result = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    assert job_result["retrieval_degraded"] is True
    assert job_result["retrieval_recovery"] == "shape_fallback"


def test_context_search_recovers_when_tool_fallback_request_over_budget(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    state = repositories.list_world_state(save.id)[0]
    _configure_tool_fallback(repositories)
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-tools",
        display_name="Fallback Tools",
        capabilities=[ProviderCapability.TOOL_CALLING.value],
        context_window=64,
    )
    primary = ShapeSwitchToolContextProvider(
        response_data={
            "selections": [
                {
                    "source_type": "world_state",
                    "source_id": state.id,
                    "relevance_note": "Selected through the structured route.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={
            "fake": primary,
            "fallback": FallbackToolContextProvider(responses=[]),
        },
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(primary.tool_call_requests) == 1
    assert len(primary.structured_output_requests) == 1
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "shape_fallback"


def test_context_search_switches_to_structured_route_when_tool_route_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    state = repositories.list_world_state(save.id)[0]
    _configure_tool_fallback(repositories)
    primary = ShapeSwitchToolContextProvider(
        response_data={
            "selections": [
                {
                    "source_type": "world_state",
                    "source_id": state.id,
                    "relevance_note": "Selected through the structured route.",
                }
            ]
        }
    )
    fallback = FallbackToolContextProvider(
        responses=[
            ProviderError(
                ProviderErrorCategory.MODEL_NOT_FOUND,
                "fallback model not found",
                status_code=404,
            )
        ]
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": primary, "fallback": fallback},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert len(primary.structured_output_requests) == 1
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.selected_state[0].relevance_note == (
        "Selected through the structured route."
    )
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "shape_fallback"
    job_result = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    assert job_result["retrieval_degraded"] is True
    assert job_result["retrieval_recovery"] == "shape_fallback"


def test_context_search_switches_to_structured_route_when_tool_fallback_missing(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    state = repositories.list_world_state(save.id)[0]
    primary = ShapeSwitchToolContextProvider(
        response_data={
            "selections": [
                {
                    "source_type": "world_state",
                    "source_id": state.id,
                    "relevance_note": "Selected through the structured route.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": primary},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(primary.tool_call_requests) == 1
    assert len(primary.structured_output_requests) == 1
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "shape_fallback"


def test_context_search_keeps_deterministic_when_structured_shape_recovery_fails(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    state = repositories.list_world_state(save.id)[0]
    _configure_tool_fallback(repositories)
    primary = ShapeFailingToolContextProvider()
    fallback = FallbackToolContextProvider(
        responses=[
            ProviderError(
                ProviderErrorCategory.MODEL_NOT_FOUND,
                "fallback model not found",
                status_code=404,
            )
        ]
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": primary, "fallback": fallback},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(primary.structured_output_requests) == 1
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "deterministic_fallback"
    job_result = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    assert job_result["retrieval_degraded"] is True
    assert job_result["retrieval_recovery"] == "deterministic_fallback"
    assert job_result["error_category"] == ProviderErrorCategory.MODEL_NOT_FOUND.value


def test_context_search_retries_model_after_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara promised Orin she would not break the bridge oath.",
        tags=["orin", "promise"],
        source_message_id=source_message.id,
    )
    provider = SequenceToolContextProvider(
        responses=[
            ProviderError(
                ProviderErrorCategory.MODEL_NOT_FOUND,
                "model not found",
                status_code=404,
            ),
            (
                ProviderToolCall(
                    id="recovered-memory",
                    name="select_context_source",
                    arguments_json=json.dumps(
                        {
                            "source_id": memory.id,
                            "relevance_note": "The promise matters on retry.",
                        }
                    ),
                ),
            ),
        ]
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    first = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )
    second = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(provider.tool_call_requests) == 2
    assert first.retrieval_recovery == "deterministic_fallback"
    assert [item.source_id for item in second.selected_memories] == [memory.id]
    assert second.retrieval_degraded is False
    assert second.retrieval_recovery is None


def test_context_search_filters_character_scoped_knowledge_by_present_character(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lens Watch")
    older_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Captain Ilyra watches the red lens tremble.",
    )
    for index in range(RECENT_MESSAGE_CANDIDATE_LIMIT):
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Lens watch bridge beat {index}.",
        )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ask Ilyra what she knows about the lens key and Lio's map.",
    )
    present = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ilyra"],
        met=True,
        character_id="character-ilyra-context-search",
    )
    inactive = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        met=True,
        character_id="character-lio-context-search",
    )
    second_present = repositories.add_character(
        save_id=save.id,
        name="Warden Rowan",
        aliases=["Rowan"],
        met=True,
        character_id="character-rowan-context-search",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Ilyra waits beside the beacon lens.",
        present_character_ids=[present.id, second_present.id],
    )
    visible_memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra knows the lens key phrase is ember dawn.",
        tags=["ilyra"],
        source_message_id=older_message.id,
    )
    hidden_memory = repositories.add_memory(
        save_id=save.id,
        body="Lio knows the crypt map is hidden under the drowned ledger.",
        tags=["lio"],
        source_message_id=older_message.id,
    )
    public_memory = repositories.add_memory(
        save_id=save.id,
        body="The beacon lens cracks faster when ash enters the gallery.",
        tags=["beacon"],
        source_message_id=older_message.id,
    )
    shared_memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra and Rowan know the shared lens watchword.",
        tags=["beacon"],
        source_message_id=older_message.id,
    )
    visible_state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens_key",
        value={"phrase": "ember dawn"},
        source_message_id=older_message.id,
    )
    hidden_state = repositories.upsert_world_state(
        save_id=save.id,
        key="crypt.map",
        value={"location": "drowned ledger"},
        source_message_id=older_message.id,
    )
    visible_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=older_message.id,
        covers_message_end_id=older_message.id,
        body="Ilyra revealed the red lens responds to the lens key.",
        provider="fake",
        model="fake-summary",
    )
    hidden_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=older_message.id,
        covers_message_end_id=older_message.id,
        body="Lio found a private route through the flooded crypt.",
        provider="fake",
        model="fake-summary",
    )
    for character, targets in (
        (
            present,
            (
                ("memory", visible_memory.id),
                ("world_state", visible_state.id),
                ("summary", visible_summary.id),
            ),
        ),
        (
            inactive,
            (
                ("memory", hidden_memory.id),
                ("world_state", hidden_state.id),
                ("summary", hidden_summary.id),
            ),
        ),
    ):
        for target_type, target_id in targets:
            repositories.add_entity_link(
                save_id=save.id,
                entity_type="character",
                entity_id=character.id,
                target_type=target_type,
                target_id=target_id,
                relation="knows",
            )
    for character in (present, second_present):
        repositories.add_entity_link(
            save_id=save.id,
            entity_type="character",
            entity_id=character.id,
            target_type="memory",
            target_id=shared_memory.id,
            relation="knows",
        )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert (
        "Character-scoped knowledge (Captain Ilyra knows): "
        "Ilyra knows the lens key phrase is ember dawn."
    ) in prompt
    assert (
        "Character-scoped knowledge (Captain Ilyra knows): "
        "beacon.lens_key: phrase: ember dawn"
    ) in prompt
    assert visible_summary.body not in prompt
    assert public_memory.body in prompt
    assert (
        "Character-scoped knowledge "
        "(Captain Ilyra knows, Warden Rowan knows): "
        "Ilyra and Rowan know the shared lens watchword."
    ) in prompt
    assert hidden_memory.body not in prompt
    assert "crypt.map" not in prompt
    assert hidden_summary.body not in prompt


def test_context_search_uses_character_knowledge_graph_for_scoped_context(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive crew tests who knows what.",
        player_role="Avery",
        content={"starting_scene": "The archive crew traces a missing map mark."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Tarin hears Avery make the archive-code joke while Nira is away.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Avery",
        body="I ask Nira if she wants to enter the chart room.",
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
        met=True,
        character_id="character-nira-knowledge-search",
    )
    tarin = repositories.add_character(
        save_id=save.id,
        name="Tarin",
        met=True,
        character_id="character-tarin-knowledge-search",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira has just arrived at the archive.",
        present_character_ids=[nira.id],
    )
    nira_memory = repositories.add_memory(
        save_id=save.id,
        body="Nira knows Avery invited her into the chart room.",
        tags=["nira"],
        source_message_id=source_message.id,
    )
    tarin_memory = repositories.add_memory(
        save_id=save.id,
        body="Tarin knows Avery made the archive-code joke within five minutes.",
        tags=["tarin", "archive-code-joke"],
        source_message_id=source_message.id,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=nira.id,
        target_type="memory",
        target_id=nira_memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        source_message_id=source_message.id,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=tarin.id,
        target_type="memory",
        target_id=tarin_memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        source_message_id=source_message.id,
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert (
        "Character-scoped knowledge (Nira knows): "
        "Nira knows Avery invited her into the chart room."
    ) in prompt
    assert "Tarin knows Avery made the archive-code joke" not in prompt


def test_context_search_keeps_recent_message_for_omniscient_narrator(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives at the archive."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    hidden = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Tarin hears Avery make the archive-code joke while Nira is away.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Avery",
        body="I ask Nira if she wants to enter the chart room.",
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
        met=True,
        character_id="character-nira-context-message-filter",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira has just arrived at the archive.",
        present_character_ids=[nira.id],
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden.id,
        character_id=nira.id,
        visibility="not_visible",
        confidence=1.0,
        source="scene_presence",
        evidence="Nira was not present for this exchange.",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert hidden.body in prompt
    assert player_message.body in prompt


def test_context_search_keeps_qualified_hidden_derivatives_for_narrator(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    hidden = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The private lens code is cobalt-seven.",
    )
    for index in range(60):
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            body=f"Routine archive beat {index}.",
        )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Avery",
        body="I ask Nira what she knows about the lens.",
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
        met=True,
    )
    mara = repositories.add_character(
        save_id=save.id,
        name="Mara",
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        present_character_ids=[nira.id, mara.id],
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden.id,
        character_id=nira.id,
        visibility="not_visible",
        confidence=1.0,
        source="scene_presence",
        evidence="Nira was absent.",
    )
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="character_fact",
        claim="The private lens code is cobalt-seven.",
        evidence_quote="private lens code is cobalt-seven",
        source_message_ids=[hidden.id],
        scope="durable",
        status="accepted",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The private lens code is cobalt-seven.",
        tags=["lens"],
        source_message_ids=[hidden.id],
        source_observation_ids=[observation.id],
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=mara.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        source_message_id=hidden.id,
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id=observation.id,
        title="Private lens code",
        body="The private lens code is cobalt-seven.",
        metadata={
            "observation_id": observation.id,
            "curation_action": "save_context",
            "source_message_ids": [hidden.id],
        },
    )
    indexed = repositories.list_context_sources(save.id, source_type="observation")
    candidates = context_search_module._indexed_context_candidates(
        indexed,
        world_state=[],
        scoped_targets=ScopedTargets(allowed={}, blocked=set()),
        reference_character_ids=frozenset({nira.id, mara.id}),
        accepted_observation_ids=frozenset({observation.id}),
        present_character_ids=frozenset({nira.id, mara.id}),
        message_visibility=repositories.list_message_visibility(save.id),
    )

    assert "cobalt-seven" in candidates[0].text
    assert "epistemic status: legacy_unclassified" in candidates[0].text

    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider()
    asyncio.run(
        ContextSearchService(
            repositories=repositories,
            providers={"fake": provider},
        ).search(save_id=save.id, player_message_id=player_message.id)
    )
    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "private lens code is cobalt-seven" in prompt


def test_context_search_keeps_message_hidden_only_from_absent_mention(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives at the archive."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    current_scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Nira lowers her voice and gives Avery the vault password.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Avery",
        body="I ask Nira whether Lio heard the vault password.",
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
        met=True,
        character_id="character-nira-present-message-filter",
    )
    lio = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        met=True,
        character_id="character-lio-absent-message-filter",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira is speaking with Avery while Lio is away.",
        present_character_ids=[nira.id],
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=current_scene_message.id,
        character_id=lio.id,
        visibility="not_visible",
        confidence=1.0,
        source="scene_presence",
        evidence="Lio was not present for this exchange.",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert current_scene_message.body in prompt
    assert player_message.body in prompt


def test_context_search_does_not_unlock_scoped_context_from_alias_substring(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lens Watch")
    older_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Archivist Lio seals the crypt ledger.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I study the rendezvous marks near the lantern.",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        met=True,
    )
    hidden_observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="character_fact",
        claim="Lio knows the crypt map is hidden under the drowned ledger.",
        evidence_quote="Lio seals the crypt ledger",
        source_message_ids=[older_message.id],
        scope="durable",
        status="accepted",
    )
    hidden_memory = repositories.add_memory(
        save_id=save.id,
        body="Lio knows the crypt map is hidden under the drowned ledger.",
        tags=["lio"],
        source_message_id=older_message.id,
        source_observation_ids=[hidden_observation.id],
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id=hidden_memory.id,
        relation="knows",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert hidden_memory.body not in prompt


def test_context_search_does_not_unlock_scoped_context_for_absent_alias_mention(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lens Watch")
    older_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Archivist Lio seals the crypt ledger.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ask Lio, what did you hide under the drowned ledger?",
    )
    present = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ilyra"],
        met=True,
        character_id="character-ilyra-present-lio-mention",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Ilyra waits in the beacon gallery while Lio is away.",
        present_character_ids=[present.id],
    )
    hidden_memory = repositories.add_memory(
        save_id=save.id,
        body="Lio knows the crypt map is hidden under the drowned ledger.",
        tags=["lio"],
        source_message_id=older_message.id,
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id=hidden_memory.id,
        relation="knows",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert hidden_memory.body not in prompt


def test_context_search_exposes_scenario_sections_as_selectable_context(
    repositories: PersistenceRepositories,
) -> None:
    selected_section = "The east tower lens is cracked."
    unselected_section = "The pantry guild argues about salted turnips."
    source_sections = {
        "factions": unselected_section,
        "locations": selected_section,
    }
    source_digest = hashlib.sha256(
        json.dumps(source_sections, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "locations": selected_section,
            "factions": unselected_section,
            "_canon_claims": {
                "version": 1,
                "source_digest": source_digest,
                "provider": "fake",
                "model": "canon",
                "claims": [
                    {
                        "claim_key": "east-lens",
                        "source_section": "locations",
                        "source_sha256": hashlib.sha256(
                            selected_section.encode()
                        ).hexdigest(),
                        "claim": selected_section,
                        "evidence_quote": selected_section,
                        "entity_anchors": [
                            {
                                "entity_type": "object",
                                "entity_key": "east-tower-lens",
                                "display_name": "the east tower lens",
                            }
                        ],
                        "fact_type": "state",
                        "fact_key": "condition",
                        "authority": "canonical",
                        "temporal_status": "current_at_scenario_start",
                        "reveal_policy": "open",
                        "known_by": [],
                        "importance": 0.45,
                    },
                    {
                        "claim_key": "pantry-guild",
                        "source_section": "factions",
                        "source_sha256": hashlib.sha256(
                            unselected_section.encode()
                        ).hexdigest(),
                        "claim": unselected_section,
                        "evidence_quote": unselected_section,
                        "entity_anchors": [
                            {
                                "entity_type": "faction",
                                "entity_key": "pantry-guild",
                                "display_name": "the pantry guild",
                            }
                        ],
                        "fact_type": "relationship",
                        "fact_key": "turnip-dispute",
                        "authority": "canonical",
                        "temporal_status": "durable",
                        "reveal_policy": "open",
                        "known_by": [],
                        "importance": 0.65,
                    },
                ],
            },
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    assert scenario_canon_is_current(json.loads(scenario.content_json))
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Ash claws at the beacon lens.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I inspect the tower lens for cracks.",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = ScenarioSectionSelectingProvider(selected_text=selected_section)
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "[scenario_claim:" in prompt
    assert selected_section in prompt
    assert unselected_section not in prompt
    assert [item.source_type for item in result.selected_scenario_sections] == [
        "scenario_claim",
    ]
    assert selected_section in result.selected_scenario_sections[0].text
    assert "Scenario-start state" in result.selected_scenario_sections[0].text
    assert result.selected_scenario_sections[0].relevance_note == (
        "The tower details shape the next beat."
    )

    jobs = _context_search_jobs(repositories, save.id)
    assert "selected_scenario_sections" in jobs[-1]["result_json"]


def test_scenario_start_claim_is_superseded_by_matching_accepted_state(
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
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="east-tower-lens.condition",
        value={"condition": "repaired"},
        category="object",
    )
    claim = repositories.upsert_context_source(
        save_id=save.id,
        source_type="scenario_claim",
        source_id="scenario-claim-lens-start",
        title="Lens at scenario start",
        body="[canonical | current_at_scenario_start | open] The lens is cracked.",
        metadata={
            "temporal_status": "current_at_scenario_start",
            "fact_key": "condition",
            "entity_anchors": [
                {
                    "entity_type": "object",
                    "entity_key": "east-tower-lens",
                    "display_name": "the east tower lens",
                }
            ],
        },
    )

    assert context_search_module._scenario_claim_is_superseded(
        claim,
        world_state=[state],
    )
    adjacent_state = repositories.upsert_world_state(
        save_id=save.id,
        key="east-tower-lens.location",
        value={"location": "workbench"},
        category="object",
    )
    assert not context_search_module._scenario_claim_is_superseded(
        claim,
        world_state=[adjacent_state],
    )
    prefixed_state = repositories.upsert_world_state(
        save_id=save.id,
        key="object.east-tower-lens.condition",
        value={"condition": "repaired"},
        category="object",
    )
    assert context_search_module._scenario_claim_is_superseded(
        claim,
        world_state=[prefixed_state],
    )


def test_narrator_only_claim_bypasses_character_known_by_filter(
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
    claim = repositories.upsert_context_source(
        save_id=save.id,
        source_type="scenario_claim",
        source_id="scenario-secret",
        title="Secret",
        body="[canonical | durable | narrator_only] The lens contains a ghost.",
        metadata={
            "reveal_policy": "narrator_only",
            "known_by": ["absent-character"],
        },
    )

    candidates = context_search_module._indexed_context_candidates(
        [claim],
        world_state=[],
        scoped_targets=ScopedTargets(allowed={}, blocked=set()),
        reference_character_ids=frozenset(),
        accepted_observation_ids=frozenset(),
        present_character_ids=frozenset(),
        message_visibility=[],
    )

    assert [candidate.source_id for candidate in candidates] == ["scenario-secret"]


def test_restricted_claim_matches_anchor_key_or_display_name_to_scope(
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
    claim = repositories.upsert_context_source(
        save_id=save.id,
        source_type="scenario_claim",
        source_id="restricted-claim",
        title="Restricted",
        body="[canonical | durable | restricted] Mira knows the signal.",
        metadata={
            "reveal_policy": "restricted",
            "known_by": ["mira"],
            "entity_anchors": [
                {
                    "entity_type": "character",
                    "entity_key": "mira",
                    "display_name": "Mira",
                }
            ],
        },
    )

    allowed = ScopedTargets(
        allowed={("character", "mira-id"): ("Mira knows",)},
        blocked=set(),
    )
    assert not context_search_module._known_by_candidate_blocked(claim, allowed)
    assert context_search_module._known_by_candidate_blocked(
        claim,
        ScopedTargets(allowed={}, blocked=set()),
    )
    assert not context_search_module._known_by_candidate_blocked(
        claim,
        ScopedTargets(allowed={}, blocked=set()),
        character_identifiers=frozenset({"mira-id", "mira"}),
    )


def test_restricted_claim_without_known_by_is_blocked(
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
    claim = repositories.upsert_context_source(
        save_id=save.id,
        source_type="scenario_claim",
        source_id="restricted-without-audience",
        title="Restricted",
        body="[canonical | durable | restricted] The lens contains a ghost.",
        metadata={"reveal_policy": "restricted", "known_by": []},
    )

    assert context_search_module._known_by_candidate_blocked(
        claim,
        ScopedTargets(allowed={}, blocked=set()),
        character_identifiers=frozenset({"mira"}),
    )


def test_scenario_supersession_key_boundaries_do_not_collide(
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
    claim = repositories.upsert_context_source(
        save_id=save.id,
        source_type="scenario_claim",
        source_id="boundary-claim",
        title="Boundary claim",
        body="An old condition.",
        metadata={
            "temporal_status": "current_at_scenario_start",
            "fact_key": "c",
            "entity_anchors": [
                {"entity_type": "object", "entity_key": "ab", "display_name": "AB"}
            ],
        },
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="a.bc",
        value={"value": "new"},
        category="object",
    )

    assert not context_search_module._scenario_claim_is_superseded(
        claim,
        world_state=[state],
    )


def test_degraded_fallback_does_not_inject_arbitrary_scenario_claim() -> None:
    claim = context_search_module._ContextCandidate(
        source_type="scenario_claim",
        source_id="irrelevant-claim",
        text="An unrelated old fact.",
    )

    assert context_search_module._fallback_candidates((claim,)) == ()


def test_context_search_exposes_state_changes_and_skips_duplicate_current_upserts(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ash Market",
        premise="A market opens only under ashfall.",
        player_role="Gatefinder",
        content={"starting_scene": "Vendors watch the sealed exits."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Gate Run")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The Moon Gate sign flickers beyond the market smoke.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I look for the exit that leads to the Moon Gate.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Ash Market"},
        category="scene",
        source_message_id=source_message.id,
    )
    selected_change = repositories.add_state_change(
        save_id=save.id,
        operation="upsert",
        state_key="scene.exit",
        before_json=json.dumps({"name": "Smoke Alley"}),
        after_json=json.dumps({"name": "Moon Gate"}),
        source_message_id=source_message.id,
    )
    repositories.add_state_change(
        save_id=save.id,
        operation="upsert",
        state_key="scene.location",
        before_json=json.dumps({"name": "Duplicate staging value"}),
        after_json=json.dumps({"name": "Ash Market"}),
        source_message_id=source_message.id,
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = StateChangeSelectingProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert [item.source_id for item in result.selected_state_changes] == [
        selected_change.id,
    ]
    assert result.selected_state_changes[0].source_type == "state_change"
    assert "Moon Gate" in result.selected_state_changes[0].text
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert "selected_state_changes" in jobs[-1]["result_json"]


def test_context_search_delete_remove_state_changes_hide_archived_before_values(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ink Vault",
        premise="A vault records which secrets were erased.",
        player_role="Archivist",
        content={"starting_scene": "The archive doors stand open."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Erasure")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The old ledger page is ash now.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check what was removed from the archive.",
    )
    delete_change = repositories.add_state_change(
        save_id=save.id,
        operation="delete",
        state_key="npc.archivist_secret",
        before_json=json.dumps(
            {"name": "Archivist Vale", "secret": "BURIED BEFORE VALUE"}
        ),
        after_json=None,
        source_message_id=source_message.id,
    )
    remove_change = repositories.add_state_change(
        save_id=save.id,
        operation="remove",
        state_key="scene.hidden_trigger",
        before_json=json.dumps(
            {"description": "A private trapdoor under the ledger"}
        ),
        after_json=None,
        source_message_id=source_message.id,
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "state_change",
                    "source_id": delete_change.id,
                    "relevance_note": "The deleted secret affects the next check.",
                },
                {
                    "source_type": "state_change",
                    "source_id": remove_change.id,
                    "relevance_note": "The removed trigger affects the room.",
                },
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    result_text = "\n".join(item.text for item in result.selected_state_changes)
    assert [item.source_id for item in result.selected_state_changes] == [
        delete_change.id,
        remove_change.id,
    ]
    assert f"[state_change:{delete_change.id}]" in prompt
    assert f"[state_change:{remove_change.id}]" in prompt
    assert "delete npc.archivist_secret" in prompt
    assert "remove scene.hidden_trigger" in prompt
    assert "delete npc.archivist_secret" in result_text
    assert "remove scene.hidden_trigger" in result_text
    for archived_value in (
        "BURIED BEFORE VALUE",
        "Archivist Vale",
        "private trapdoor",
        "before:",
    ):
        assert archived_value not in prompt
        assert archived_value not in result_text
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert "BURIED BEFORE VALUE" not in jobs[-1]["result_json"]
    assert "private trapdoor" not in jobs[-1]["result_json"]


def test_context_search_filters_state_changes_by_scope_and_message_visibility(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lens Watch")
    scoped_source = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Lio privately studies the archive.",
    )
    hidden_source = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A route opens while Nira is away.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ask Nira what Lio knows about the route.",
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
        met=True,
        character_id="character-nira-state-change-present",
    )
    lio = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        met=True,
        character_id="character-lio-state-change-absent",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira stands beside the beacon while Lio is away.",
        present_character_ids=[nira.id],
    )
    scoped_state = repositories.upsert_world_state(
        save_id=save.id,
        key="crypt.map",
        value={"location": "sealed vault"},
        source_message_id=scoped_source.id,
        state_id="world-state-lio-crypt-map",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=lio.id,
        target_type="world_state",
        target_id=scoped_state.id,
        relation="knows",
    )
    archived_state = repositories.upsert_world_state(
        save_id=save.id,
        key="crypt.route",
        value={"name": "sealed arch"},
        source_message_id=scoped_source.id,
        state_id="world-state-lio-archived-route",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=lio.id,
        target_type="world_state",
        target_id=archived_state.id,
        relation="knows",
    )
    repositories.archive_world_state(save_id=save.id, key=archived_state.key)
    scoped_change = repositories.add_state_change(
        save_id=save.id,
        operation="upsert",
        state_key="crypt.map",
        after_json=json.dumps({"location": "drowned ledger"}),
        source_message_id=scoped_source.id,
    )
    archived_scoped_change = repositories.add_state_change(
        save_id=save.id,
        operation="upsert",
        state_key="crypt.route",
        after_json=json.dumps({"name": "forgotten gate"}),
        source_message_id=scoped_source.id,
    )
    hidden_source_change = repositories.add_state_change(
        save_id=save.id,
        operation="upsert",
        state_key="route.current",
        after_json=json.dumps({"name": "ash bridge route"}),
        source_message_id=hidden_source.id,
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden_source.id,
        character_id=nira.id,
        visibility="not_visible",
        confidence=1.0,
        source="scene_presence",
        evidence="Nira was not present for this route change.",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert f"[state_change:{scoped_change.id}]" not in prompt
    assert f"[state_change:{archived_scoped_change.id}]" not in prompt
    assert f"[state_change:{hidden_source_change.id}]" in prompt
    assert "drowned ledger" not in prompt
    assert "forgotten gate" not in prompt
    assert "ash bridge route" in prompt
    assert "crypt.map" not in prompt
    assert "crypt.route" not in prompt


def test_context_search_omits_state_changes_from_archived_source_messages(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Archive Fork",
        premise="Deleted turns should not feed later context.",
        player_role="Revisionist",
        content={"starting_scene": "A clean page waits."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Clean Branch")
    archived_source = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The obsolete route led to the buried archive.",
    )
    active_source = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The active route leads to the lantern stair.",
    )
    archived_change = repositories.add_state_change(
        save_id=save.id,
        operation="upsert",
        state_key="scene.route",
        before_json=None,
        after_json=json.dumps({"name": "Buried Archive"}),
        source_message_id=archived_source.id,
    )
    active_change = repositories.add_state_change(
        save_id=save.id,
        operation="upsert",
        state_key="scene.route",
        before_json=None,
        after_json=json.dumps({"name": "Lantern Stair"}),
        source_message_id=active_source.id,
    )
    repositories.archive_message(archived_source.id)
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I follow the current route upward.",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "state_change",
                    "source_id": active_change.id,
                    "relevance_note": "Only the active source message is current.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    request = provider.structured_output_requests[0]
    selection_properties = request.schema["properties"]["selections"]["items"][
        "properties"
    ]
    source_id_enum = selection_properties["source_id"]["enum"]
    prompt = "\n".join(message.body for message in request.messages)
    assert active_change.id in source_id_enum
    assert archived_change.id not in source_id_enum
    assert "Lantern Stair" in prompt
    assert "Buried Archive" not in prompt
    assert "obsolete route" not in prompt
    assert [item.source_id for item in result.selected_state_changes] == [
        active_change.id
    ]


def test_context_search_media_candidates_include_only_recent_successful_linked_images(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Blackwater Bridge",
        premise="Lanterns reveal crossings that prose can forget.",
        player_role="Bridge scout",
        content={"starting_scene": "Rain beads on the bridge rails."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Bridge Watch")
    old_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="An old bridge image is no longer near the active exchange.",
    )
    repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="media/private/old.png",
        thumbnail_path="media/private/thumb-old.png",
        prompt="old successful image prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
        source_message_id=old_message.id,
    )
    recent_narrator = old_message
    for index in range(RECENT_MESSAGE_CANDIDATE_LIMIT):
        recent_narrator = repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Recent bridge beat {index}.",
        )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I compare the bridge lights to the latest image.",
    )
    selected_asset = repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="media/private/recent.png",
        thumbnail_path="media/private/thumb-recent.png",
        prompt=(
            "gold bridge lights over black water "
            + "mist curls " * 40
            + "SECRET FULL PROMPT TAIL SHOULD NOT APPEAR"
        ),
        provider="fake",
        model="fake-image",
        status="succeeded",
        source_message_id=recent_narrator.id,
    )
    repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="media/private/failed.png",
        prompt="failed recent image prompt with raw bytes",
        provider="fake",
        model="fake-image",
        status="failed",
        source_message_id=recent_narrator.id,
    )
    repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="media/private/unlinked.png",
        prompt="unlinked successful image prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
        source_message_id=None,
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = MediaAssetSelectingProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert [item.source_id for item in result.selected_media_assets] == [
        selected_asset.id,
    ]
    assert result.selected_media_assets[0].source_type == "media_asset"
    assert (
        "gold bridge lights over black water"
        in result.selected_media_assets[0].text
    )
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert "selected_media_assets" in jobs[-1]["result_json"]


def test_context_search_redacts_media_prompt_paths_and_data_payloads(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Redacted Bridge",
        premise="Images can guide the next beat without leaking local paths.",
        player_role="Bridge scout",
        content={"starting_scene": "Lanterns shimmer on wet stone."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Image Hygiene")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A fresh scene image shows the bridge approach.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I study the image for the safest path.",
    )
    base64_payload = "Q" * 132
    wrapped_base64_payload = "\n".join(("R" * 48, "S" * 48, "T" * 48))
    data_payload = "data:image/png;base64," + "\n".join(
        ("A" * 48, "B" * 48, "C" * 48)
    )
    substantial_payload_leak = base64_payload[:80]
    selected_asset = repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="media/private/redacted-bridge.png",
        thumbnail_path="media/private/thumb-redacted-bridge.png",
        prompt=(
            f"gold bridge lights over black water; token {base64_payload}; "
            f"wrapped {wrapped_base64_payload}; "
            "source media/private/raw.png; "
            "root /root/private/raw.png; "
            "opt /opt/bragi/raw.png; "
            "config /etc/bragi/raw.png; "
            "relative assets/raw.png; "
            r"windows C:\Users\Mara\raw.png; "
            "reference /home/user/dev/bragi/media/private/raw.png; "
            f"inline {data_payload}"
        ),
        provider="fake",
        model="fake-image",
        status="succeeded",
        source_message_id=source_message.id,
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "media_asset",
                    "source_id": selected_asset.id,
                    "relevance_note": "The image shows the bridge path.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    result_text = result.selected_media_assets[0].text
    assert "gold bridge lights over black water" in prompt
    assert "gold bridge lights over black water" in result_text
    raw_values = (
        "media/private/raw.png",
        "/root/private/raw.png",
        "/opt/bragi/raw.png",
        "/etc/bragi/raw.png",
        "assets/raw.png",
        r"C:\Users\Mara\raw.png",
        "/home/user/dev/bragi/media/private/raw.png",
        "data:image/png;base64",
        data_payload,
        base64_payload,
        wrapped_base64_payload,
        substantial_payload_leak,
    )
    for raw_value in raw_values:
        assert raw_value not in prompt
        assert raw_value not in result_text
    for redaction in (
        "[redacted media path]",
        "[redacted file path]",
        "[redacted data payload]",
    ):
        assert redaction in prompt
        assert redaction in result_text
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    for raw_value in raw_values:
        assert raw_value not in jobs[-1]["result_json"]


def test_context_search_marks_job_failed_when_provider_raises(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    provider = FailingStructuredContextProvider(
        RuntimeError("context provider unavailable")
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    with pytest.raises(RuntimeError, match="context provider unavailable"):
        asyncio.run(
            service.search(save_id=save.id, player_message_id=player_message.id)
        )

    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "failed"
    assert jobs[-1]["error"] == "context provider unavailable"
    assert jobs[-1]["result_json"] is None


def test_context_search_uses_structured_fallback_when_primary_blocks(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara distrusts bells that ring without wind.",
        tags=["bells"],
        source_message_id=source_message.id,
    )
    repositories.set_app_setting("structured_output_fallback_enabled", True)
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-structured",
        display_name="Fallback Structured",
        capabilities=["structured_output", "fallback_marker"],
    )
    repositories.set_model_preference(
        task="structured_output_fallback",
        provider="fallback",
        model_id="fallback-structured",
    )
    primary = FailingStructuredContextProvider(
        ProviderError(
            ProviderErrorCategory.CONTENT_BLOCKED,
            "primary structured output blocked",
        )
    )
    fallback = FallbackStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": memory.id,
                    "relevance_note": "The bell concern matters now.",
                },
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": primary, "fallback": fallback},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(primary.structured_output_requests) == 1
    assert len(fallback.structured_output_requests) == 1
    assert fallback.structured_output_requests[0].provider == "fallback"
    assert fallback.structured_output_requests[0].model_id == "fallback-structured"
    assert [item.source_id for item in result.selected_memories] == [memory.id]
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "provider_fallback"
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"


def test_context_search_uses_structured_fallback_for_each_primary_model_error(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara distrusts bells that ring without wind.",
        tags=["bells"],
        source_message_id=source_message.id,
    )
    repositories.set_app_setting("structured_output_fallback_enabled", True)
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-structured",
        display_name="Fallback Structured",
        capabilities=["structured_output"],
    )
    repositories.set_model_preference(
        task="structured_output_fallback",
        provider="fallback",
        model_id="fallback-structured",
    )
    primary = FailingStructuredContextProvider(
        ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "primary structured output model missing",
            status_code=404,
        )
    )
    fallback = FallbackStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": memory.id,
                    "relevance_note": "The bell concern matters now.",
                },
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": primary, "fallback": fallback},
    )

    first = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )
    second = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(primary.structured_output_requests) == 2
    assert len(fallback.structured_output_requests) == 2
    assert [item.source_id for item in first.selected_memories] == [memory.id]
    assert [item.source_id for item in second.selected_memories] == [memory.id]
    assert first.retrieval_recovery == "provider_fallback"
    assert second.retrieval_recovery == "provider_fallback"
    assert [
        model.available
        for model in repositories.list_provider_models("fake")
        if model.model_id == "fake-context"
    ] == [True]
    job_result = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    assert job_result["fallback_used"] is True
    assert job_result["fallback_provider"] == "fallback"
    assert job_result["fallback_model"] == "fallback-structured"


def test_context_search_structured_empty_selection_uses_continuity_floor(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    state = repositories.list_world_state(save.id)[0]
    provider = RecordingStructuredContextProvider({"selections": []})
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(provider.structured_output_requests) == 1
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.selected_state[0].relevance_note == (
        "Selected by deterministic fallback after empty context selection."
    )
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "deterministic_fallback"
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert jobs[-1]["result_json"] is not None
    job_result = json.loads(jobs[-1]["result_json"])
    assert job_result["retrieval_degraded"] is True
    assert job_result["retrieval_recovery"] == "deterministic_fallback"
    assert job_result["fallback_used"] is False


def test_context_search_offers_continuity_floor_when_index_has_no_hits(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="survival_expedition",
        title="Frostline",
        premise="An expedition crosses the white shelf.",
        player_role="Scout",
        content={
            "starting_scene": "The lower pass is blocked by glass ice.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="White Shelf")
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The camp marked the lower pass as unsafe.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="Okay.",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "North Ridge Camp"},
        category="scene",
        source_message_id=narrator_message.id,
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The frost lantern marks the safe camp perimeter overnight.",
        tags=["camp"],
        importance=0.9,
        source_message_id=narrator_message.id,
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": memory.id,
                    "relevance_note": "The lantern detail is continuity-critical.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "scene.location" in prompt
    assert state.id in prompt
    assert memory.body in prompt
    assert "The lower pass is blocked by glass ice." not in prompt
    assert [item.source_id for item in result.selected_memories] == [memory.id]
    jobs = _context_search_jobs(repositories, save.id)
    job_result = json.loads(jobs[-1]["result_json"])
    diagnostics = job_result["diagnostics"]
    assert diagnostics["indexed_retrieval_hit_count"] == 0
    assert diagnostics["protected_context_source_count"] >= 1
    assert diagnostics["continuity_floor_candidate_count"] >= 2


def test_context_search_rehydrates_selected_items_beyond_selector_excerpt(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    long_memory = (
        "Mara remembers the bridge bell starts with an oath phrase. "
        + " ".join(f"selector filler detail {index}" for index in range(120))
        + " SECRET TAIL THAT THE SELECTOR NEVER SAW"
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body=long_memory,
        tags=["bells"],
        source_message_id=source_message.id,
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": memory.id,
                    "relevance_note": "The oath phrase matters now.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    selected = result.selected_memories[0]
    assert selected.source_id == memory.id
    assert selected.excerpted is False
    assert "Mara remembers the bridge bell" in selected.text
    assert "SECRET TAIL THAT THE SELECTOR NEVER SAW" in selected.text
    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "SECRET TAIL THAT THE SELECTOR NEVER SAW" not in prompt
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert jobs[-1]["result_json"] is not None
    job_result = json.loads(jobs[-1]["result_json"])
    assert job_result["selected_memories"][0]["excerpted"] is False


def test_context_search_drops_source_deleted_after_selection(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara remembers that the bridge bell is cracked.",
        tags=["bells"],
        source_message_id=player_message.id,
    )
    provider = MutatingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": memory.id,
                    "relevance_note": "The cracked bell matters.",
                }
            ]
        },
        after_selection=lambda: repositories.archive_memory(memory.id),
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert result.selected_memories == ()
    job_result = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    assert job_result["selected_memories"] == []


def test_context_search_uses_composite_source_identity_for_colliding_ids(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    state = repositories.list_world_state(save.id)[0]
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara remembers the bell keeper's private warning.",
        tags=["bells", "warning"],
        source_message_id=player_message.id,
        memory_id=state.id,
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": memory.id,
                    "relevance_note": "Select the memory, not state.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert result.selected_state == ()
    assert [item.source_id for item in result.selected_memories] == [memory.id]
    assert result.selected_memories[0].text == memory.body


def test_context_search_recovers_from_unknown_structured_selection_ids(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    state = repositories.list_world_state(save.id)[0]
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": "memory-that-is-not-a-candidate",
                    "relevance_note": "This id was not offered.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert provider.chat_requests == []
    assert len(provider.structured_output_requests) == 1
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "deterministic_fallback"
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert jobs[-1]["error"] is None
    job_result = json.loads(jobs[-1]["result_json"])
    assert job_result["retrieval_degraded"] is True
    assert job_result["retrieval_recovery"] == "deterministic_fallback"
    assert job_result["fallback_used"] is False


def test_context_search_rejects_mismatched_source_type_and_source_id(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara distrusts bells that ring without wind.",
        tags=["bells", "suspicion"],
        source_message_id=player_message.id,
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "world_state",
                    "source_id": memory.id,
                    "relevance_note": "The memory id is valid despite the type.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert provider.chat_requests == []
    assert len(provider.structured_output_requests) == 1
    assert result.selected_state
    assert all(
        item.relevance_note
        == "Selected by deterministic fallback after empty context selection."
        for item in result.selected_memories
    )
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "deterministic_fallback"

    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert jobs[-1]["error"] is None
    assert jobs[-1]["result_json"] is not None


def test_context_search_requires_selected_model_capability(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[ProviderCapability.CHAT.value],
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    with pytest.raises(
        ValueError,
        match="does not advertise structured output or tool calling",
    ):
        asyncio.run(
            service.search(save_id=save.id, player_message_id=player_message.id)
        )

    assert provider.chat_requests == []
    assert provider.structured_output_requests == []
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "failed"
    assert "does not advertise structured output or tool calling" in jobs[-1]["error"]
    assert jobs[-1]["result_json"] is None


def test_context_search_rejects_missing_catalog_row_for_selected_model(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        save_model=False,
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    with pytest.raises(ValueError, match="not in the provider model catalog"):
        asyncio.run(
            service.search(save_id=save.id, player_message_id=player_message.id)
        )

    assert provider.chat_requests == []
    assert provider.structured_output_requests == []
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "failed"
    assert "not in the provider model catalog" in jobs[-1]["error"]
    assert jobs[-1]["result_json"] is None


def test_context_search_recovers_from_unavailable_selected_model(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    state = repositories.list_world_state(save.id)[0]
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert provider.chat_requests == []
    assert provider.structured_output_requests == []
    assert [item.source_id for item in result.selected_state] == [state.id]
    assert result.retrieval_degraded is True
    assert result.retrieval_recovery == "deterministic_fallback"
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert jobs[-1]["error"] is None
    job_result = json.loads(jobs[-1]["result_json"])
    assert job_result["retrieval_degraded"] is True
    assert job_result["retrieval_recovery"] == "deterministic_fallback"
    assert job_result["fallback_skipped_reason"] == "no_fallback_model"


def test_context_search_marks_job_failed_when_preferred_provider_is_missing(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    repositories.set_model_preference(
        task="context_search",
        provider="missing",
        model_id="missing-context",
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={},
    )

    with pytest.raises(KeyError, match="missing"):
        asyncio.run(
            service.search(save_id=save.id, player_message_id=player_message.id)
        )

    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "failed"
    assert "missing" in jobs[-1]["error"]
    assert jobs[-1]["result_json"] is None


def test_context_search_accepts_empty_selection_without_continuity_floor(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Bridge of Cinders",
        premise="A bridge remembers every oath broken on it.",
        player_role="Oathkeeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Crossing")
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A silver bell rings beneath the bridge.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="Continue.",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider({"selections": []})
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert result == ContextSearchResult(continuity_index_synced=True)
    request_text = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "A silver bell rings beneath the bridge." in request_text
    assert provider.chat_requests == []
    assert len(provider.structured_output_requests) == 1
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    job_result = json.loads(jobs[-1]["result_json"])
    assert job_result["empty_selection_policy"] == "accepted_no_context"
    assert job_result["diagnostics"]["continuity_floor_candidate_count"] == 0


def test_context_search_drops_unsafe_and_recent_overlapping_summaries(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Bridge of Cinders",
        premise="A bridge remembers every oath broken on it.",
        player_role="Oathkeeper",
        content={"starting_scene": "Cinders drift over the bridge stones."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Crossing")
    messages = [
        repositories.append_message(
            save_id=save.id,
            role="narrator" if index % 2 else "player",
            speaker_name="Narrator" if index % 2 else "Mara",
            body=(
                f"Bridge chronicle beat {index}: Mara studies the oath bell "
                "and the old ash marks."
            ),
        )
        for index in range(28)
    ]
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I listen for the bell under the shrine.",
    )
    safe_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[0].id,
        covers_message_end_id=messages[1].id,
        body="Mara studied the oath bell and old ash marks near the bridge.",
        provider="fake",
        model="fake-summary",
    )
    unsafe_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[2].id,
        covers_message_end_id=messages[3].id,
        body='You touch the bell rope. "What do you do next?"',
        provider="fake",
        model="fake-summary",
    )
    recent_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[-1].id,
        covers_message_end_id=player_message.id,
        body="Recent summary: Mara is already listening at the shrine.",
        provider="fake",
        model="fake-summary",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RecordingStructuredContextProvider({"selections": []})
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert result.selected_summaries == ()
    request_text = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert safe_summary.body not in request_text
    assert unsafe_summary.body not in request_text
    assert recent_summary.body not in request_text


def test_context_search_uses_ranked_index_before_structured_selection(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    exact_memory = repositories.add_memory(
        save_id=save.id,
        body="The copper notch opens only when Mara says ember dawn.",
        tags=["promise", "lens"],
        importance=0.95,
        source_message_id=source_message.id,
    )
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="ember dawn",
    )
    for index in range(130):
        repositories.add_summary(
            save_id=save.id,
            covers_message_start_id=source_message.id,
            covers_message_end_id=source_message.id,
            body=f"Low-priority pantry recap {index}: shelves and flour.",
            provider="fake",
            model="fake-summary",
        )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": exact_memory.id,
                    "relevance_note": "Exact old notch phrase.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert exact_memory.body in prompt
    assert "Low-priority pantry recap 129" not in prompt
    assert provider.expansion_requests == []
    assert result.retrieval_round_used is False
    assert [item.source_id for item in result.selected_memories] == [exact_memory.id]
    jobs = _context_search_jobs(repositories, save.id)
    result_json = json.loads(jobs[-1]["result_json"])
    assert result_json["selected_memories"][0]["source_id"] == exact_memory.id


def test_context_search_uses_structured_paraphrase_and_pronoun_expansion(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The physician keeps antivenom in the western archive.",
        tags=["medicine"],
        importance=0.8,
        source_message_id=source_message.id,
    )
    repositories.add_memory(
        save_id=save.id,
        body="The general store closes before sunset.",
        tags=["market"],
        importance=0.2,
        source_message_id=source_message.id,
    )
    physician = repositories.add_character(
        save_id=save.id,
        name="Physician Tessa",
        role="healer",
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Physician Tessa just left the western archive.",
        present_character_ids=[physician.id],
    )
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="Where did she store it?",
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": memory.id,
                    "relevance_note": "The healer refers to the physician.",
                }
            ]
        },
        expansion_data={
            "terms": ["physician", "antivenom"],
            "phrases": [],
            "entity_ids": [physician.id],
        },
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(provider.expansion_requests) == 1
    assert result.retrieval_round_used is True
    expansion_prompt = "\n".join(
        message.body for message in provider.expansion_requests[0].messages
    )
    assert "Where did she store it?" in expansion_prompt
    assert "Physician Tessa" in expansion_prompt
    selection_prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert memory.body in selection_prompt
    assert [item.source_id for item in result.selected_memories] == [memory.id]


def test_context_search_uses_post_turn_precomputed_snapshot(
    repositories: PersistenceRepositories,
) -> None:
    save, _prior_player_message = _save_with_context_search_preference(repositories)
    provider = RecordingStructuredContextProvider({"selections": []})
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )
    service.precompute_next_turn(save.id)
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I listen for the silver bell.",
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    result_json = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    assert result_json["diagnostics"]["cache_status"] == "hit"

    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.warning",
        value={"active": True},
        category="scene",
        source_message_id=None,
    )
    next_player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="What changed?",
    )
    asyncio.run(
        service.search(
            save_id=save.id,
            player_message_id=next_player_message.id,
        )
    )
    stale_result_json = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    assert stale_result_json["diagnostics"]["cache_status"] == "miss"


def test_context_search_reloads_cache_mutated_during_candidate_build(
    repositories: PersistenceRepositories,
) -> None:
    counting = CountingPersistenceRepositories(repositories.connection)
    save, _prior_player_message = _save_with_context_search_preference(counting)
    provider = RecordingStructuredContextProvider({"selections": []})
    service = ContextSearchService(
        repositories=counting,
        providers={"fake": provider},
    )
    service.precompute_next_turn(save.id)
    player_message = counting.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I listen for the silver bell.",
    )

    def mutate_context() -> None:
        counting.upsert_world_state(
            save_id=save.id,
            key="scene.warning",
            value={"active": True},
            category="scene",
            source_message_id=None,
        )

    counting.before_context_source_search = mutate_context
    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert result.narration_snapshot is not None
    assert any(
        state.key == "scene.warning"
        for state in result.narration_snapshot.world_state
    )
    result_json = json.loads(
        _context_search_jobs(counting, save.id)[-1]["result_json"]
    )
    assert result_json["diagnostics"]["cache_status"] == "stale"


def test_context_search_reloads_cache_miss_mutated_during_candidate_build(
    repositories: PersistenceRepositories,
) -> None:
    counting = CountingPersistenceRepositories(repositories.connection)
    save, player_message = _save_with_context_search_preference(counting)
    provider = RecordingStructuredContextProvider({"selections": []})
    service = ContextSearchService(
        repositories=counting,
        providers={"fake": provider},
    )

    def mutate_context() -> None:
        counting.upsert_world_state(
            save_id=save.id,
            key="scene.warning",
            value={"active": True},
            category="scene",
            source_message_id=None,
        )

    counting.before_context_source_search = mutate_context
    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert result.narration_snapshot is not None
    assert any(
        state.key == "scene.warning"
        for state in result.narration_snapshot.world_state
    )
    result_json = json.loads(
        _context_search_jobs(counting, save.id)[-1]["result_json"]
    )
    assert result_json["diagnostics"]["cache_status"] == "retried"


def test_context_search_expansion_reuses_retrieval_prelude(
    repositories: PersistenceRepositories,
) -> None:
    counting = CountingPersistenceRepositories(repositories.connection)
    save, player_message = _save_with_context_search_preference(counting)
    source_message = next(
        message
        for message in counting.list_messages(save.id)
        if message.role == "narrator"
    )
    memory = counting.add_memory(
        save_id=save.id,
        body="The physician keeps antivenom in the western archive.",
        tags=["medicine"],
        importance=0.8,
        source_message_id=source_message.id,
    )
    physician = counting.add_character(
        save_id=save.id,
        name="Physician Tessa",
        role="healer",
        met=True,
    )
    counting.upsert_scene_snapshot(
        save_id=save.id,
        situation="Physician Tessa just left the western archive.",
        present_character_ids=[physician.id],
    )
    counting.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="Where did she store it?",
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": memory.id,
                    "relevance_note": "The healer refers to the physician.",
                }
            ]
        },
        expansion_data={
            "terms": ["physician", "antivenom"],
            "phrases": [],
            "entity_ids": [physician.id],
        },
    )
    service = ContextSearchService(
        repositories=counting,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert [item.source_id for item in result.selected_memories] == [memory.id]
    assert len(provider.expansion_requests) == 1
    assert counting.list_counts["protected_context_sources"] == 1
    assert counting.list_counts["context_source_searches"] == 4


def test_context_search_uses_tool_paraphrase_despite_noisy_lexical_hit(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The physician keeps antivenom in the western archive.",
        tags=["remedy"],
        source_message_id=source_message.id,
    )
    repositories.add_memory(
        save_id=save.id,
        body="The medicine market closes before sunset.",
        tags=["medicine", "market"],
        source_message_id=source_message.id,
    )
    physician = repositories.add_character(
        save_id=save.id,
        name="Physician Tessa",
        role="healer",
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Physician Tessa just left the western archive.",
        present_character_ids=[physician.id],
    )
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="Where is the medicine stored?",
    )
    provider = SequenceToolContextProvider(
        responses=[
            (
                ProviderToolCall(
                    id="call-memory",
                    name="select_context_source",
                    arguments_json=json.dumps(
                        {
                            "source_id": memory.id,
                            "relevance_note": "The healer refers to the physician.",
                        }
                    ),
                ),
            ),
        ],
        expansion_data={
            "terms": ["physician", "antivenom"],
            "phrases": [],
            "entity_ids": [physician.id],
        },
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert len(provider.tool_expansion_requests) == 1
    assert [tool.name for tool in provider.tool_expansion_requests[0].tools] == [
        "expand_context_retrieval"
    ]
    assert memory.body in "\n".join(
        message.body for message in provider.tool_call_requests[0].messages
    )
    assert [item.source_id for item in result.selected_memories] == [memory.id]


def test_context_search_offers_indexed_character_voice_candidates(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ashknife"],
        role="Watch captain",
        met=True,
        personality="Masks fear with dry precision.",
        voice="Low clipped commands; never rambles.",
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "character_voice",
                    "source_id": character.id,
                    "relevance_note": "Ilyra's voice is relevant.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    request = provider.structured_output_requests[0]
    source_type_enum = request.schema["properties"]["selections"]["items"][
        "properties"
    ]["source_type"]["enum"]
    prompt = "\n".join(message.body for message in request.messages)
    assert "character_voice" in source_type_enum
    assert "Low clipped commands; never rambles." in prompt
    assert [item.source_id for item in result.selected_character_voice] == [
        character.id
    ]
    result_json = json.loads(
        _context_search_jobs(repositories, save.id)[-1]["result_json"]
    )
    assert result_json["selected_character_voice"][0]["source_id"] == character.id


def test_context_search_retrieves_matching_world_state_from_index(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    target_state = repositories.upsert_world_state(
        save_id=save.id,
        key="zzzz.hidden_lens_phrase",
        value={"phrase": "ember dawn opens the copper notch"},
        category="world_fact",
        confidence=1.0,
        source_message_id=source_message.id,
    )
    for index in range(5):
        repositories.upsert_world_state(
            save_id=save.id,
            key=f"aaaa.low_priority_fact_{index:02d}",
            value={"detail": f"flour shelf marker {index}"},
            category="world_fact",
            confidence=1.0,
            source_message_id=source_message.id,
        )
    ContinuityIndexService(repositories, world_state_limit=2).sync_save(save.id)
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="I test whether ember dawn opens the copper notch.",
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "world_state",
                    "source_id": target_state.id,
                    "relevance_note": "The hidden phrase answers the player.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "zzzz.hidden_lens_phrase" in prompt
    world_state_candidate_lines = [
        line for line in prompt.splitlines() if line.startswith("- [world_state:")
    ]
    assert (
        sum(
            "ember dawn opens the copper notch" in line
            for line in world_state_candidate_lines
        )
        == 1
    )
    assert [item.source_id for item in result.selected_state] == [target_state.id]


def test_context_search_indexes_legacy_state_source(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    target_state = repositories.upsert_world_state(
        save_id=save.id,
        key="zzzz.cobalt_ledger_phrase",
        value={"phrase": "moonstone opens the cobalt ledger"},
        category="world_fact",
        confidence=1.0,
        source_message_id=source_message.id,
    )
    legacy_source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="state",
        source_id=target_state.id,
        title="Cobalt ledger phrase",
        body="zzzz.cobalt_ledger_phrase: moonstone opens the cobalt ledger",
        metadata={
            "fact_type": "world_state",
            "source_message_ids": [source_message.id],
        },
    )
    assert "state" in context_search_module.INDEXED_CONTEXT_SOURCE_TYPES
    assert (
        context_search_module._indexed_candidate_source_type(
            legacy_source,
            accepted_observation_ids=frozenset(),
        )
        == "world_state"
    )
    for index in range(520):
        repositories.upsert_world_state(
            save_id=save.id,
            key=f"aaaa.low_priority_fact_{index:03d}",
            value={"detail": f"flour shelf marker {index}"},
            category="world_fact",
            confidence=1.0,
            source_message_id=source_message.id,
        )
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="Which fact says what opens the cobalt ledger?",
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "world_state",
                    "source_id": target_state.id,
                    "relevance_note": "The legacy indexed state answers the query.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "moonstone opens the cobalt ledger" in prompt
    assert [item.source_id for item in result.selected_state] == [target_state.id]


def test_context_search_retrieves_matching_memory_from_index(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    indexed_memory = repositories.add_memory(
        save_id=save.id,
        body="The bridge bell hums when the ordinary pantry ledger is opened.",
        tags=["bridge"],
        importance=1.0,
        source_message_id=source_message.id,
    )
    target_memory = repositories.add_memory(
        save_id=save.id,
        body="Mara learned the hidden lens phrase: ember dawn.",
        tags=["lens"],
        importance=1.0,
        source_message_id=source_message.id,
        )
    ContinuityIndexService(repositories, memory_limit=1).sync_save(save.id)
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="I repeat the hidden lens phrase ember dawn.",
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": target_memory.id,
                    "relevance_note": "The hidden phrase is relevant.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert target_memory.body in prompt
    assert indexed_memory.body not in prompt
    assert [item.source_id for item in result.selected_memories] == [target_memory.id]


def test_context_search_indexes_exact_memory_beyond_previous_default_bound(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    target = repositories.add_memory(
        save_id=save.id,
        body="The obsidian astrolabe opens the forgotten observatory.",
        tags=["routine"],
        importance=0.1,
        source_message_id=source_message.id,
    )
    for index in range(520):
        repositories.add_memory(
            save_id=save.id,
            body=f"High-priority bridge watch record {index:03d}.",
            tags=["promise"],
            importance=1.0,
            source_message_id=source_message.id,
        )
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body=(
            "Before acting I consider weather guards ropes lanterns maps "
            "rations witnesses schedules locks routes signals bridges towers. "
            "The obsidian astrolabe opens the forgotten observatory. "
            "Afterward I discuss supplies patrols repairs messages allies "
            "horses gates ledgers bells windows courtyards kitchens cellars."
        ),
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": target.id,
                    "relevance_note": "The exact old fact answers the action.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert target.body in prompt
    assert [item.source_id for item in result.selected_memories] == [target.id]


def test_context_search_indexes_legacy_plural_memory_source(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    target = repositories.add_memory(
        save_id=save.id,
        body="The moonstone opens the cobalt ledger.",
        tags=["ledger"],
        importance=0.1,
        source_message_id=source_message.id,
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memories",
        source_id=target.id,
        title="Moonstone ledger memory",
        body=target.body,
        metadata={"fact_type": "memory", "source_message_ids": [source_message.id]},
    )
    for index in range(520):
        repositories.add_memory(
            save_id=save.id,
            body=f"High-priority bridge watch record {index:03d}.",
            tags=["routine"],
            importance=1.0,
            source_message_id=source_message.id,
        )
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="Which memory explains what opens the cobalt ledger?",
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": target.id,
                    "relevance_note": "The legacy indexed memory answers the query.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert target.body in prompt
    assert [item.source_id for item in result.selected_memories] == [target.id]


def test_context_search_retrieves_short_code_memory_beyond_raw_bound(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    target = repositories.add_memory(
        save_id=save.id,
        body="Only A-7 opens the old river vault.",
        tags=["vault"],
        importance=0.1,
        source_message_id=source_message.id,
    )
    for index in range(520):
        repositories.add_memory(
            save_id=save.id,
            body=f"A 7 appears in newer maintenance record {index:03d}.",
            tags=["routine"],
            importance=1.0,
            source_message_id=source_message.id,
        )
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="Where did we learn about A-7?",
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": target.id,
                    "relevance_note": "The exact code answers the question.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert target.body in prompt
    assert [item.source_id for item in result.selected_memories] == [target.id]


def test_context_search_filters_indexed_known_by_facts_deterministically(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="indexed-lio-secret",
        title="Lio secret",
        body="Lio knows the crypt map is under the drowned ledger.",
        metadata={"known_by": ["Lio"], "fact_type": "memory"},
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="indexed-public-fact",
        title="Public bridge fact",
        body="The bridge bell rings without wind.",
        metadata={"fact_type": "memory"},
    )
    provider = RecordingStructuredContextProvider()
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "Lio knows the crypt map" not in prompt
    assert "The bridge bell rings without wind." in prompt


def test_context_search_retrieves_audience_scoped_character_text_thread(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    player = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="player",
        is_player_character=True,
    )
    rowan = repositories.add_character(
        save_id=save.id,
        name="Rowan",
        role="repair club",
        met=True,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=rowan.id,
        title="Rowan",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="player",
        sender_character_id=player.id,
        body="Can you bring the circuit-lantern repair code?",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="character",
        sender_character_id=rowan.id,
        body="I will keep the ember relay code ready.",
    )
    repositories.update_character_text_thread_memory(
        save_id=save.id,
        thread_id=thread.id,
        body=(
            "Phone thread memory: Rowan and Mara discussed the "
            "circuit-lantern ember relay code."
        ),
        message_count=2,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Rowan waits by the circuit lantern.",
        present_character_ids=[rowan.id],
    )
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="I ask Rowan about the circuit-lantern ember relay code.",
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "character_text_thread",
                    "source_id": thread.id,
                    "relevance_note": "Rowan's text thread has the repair code.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert [item.source_id for item in result.selected_character_text_context] == [
        thread.id
    ]
    request = provider.structured_output_requests[0]
    source_type_enum = request.schema["properties"]["selections"]["items"][
        "properties"
    ]["source_type"]["enum"]
    prompt = "\n".join(message.body for message in request.messages)
    assert "character_text_thread" in source_type_enum
    assert "Phone thread memory: Rowan and Mara discussed" in prompt
    raw_result = _context_search_jobs(repositories, save.id)[-1]["result_json"]
    assert raw_result is not None
    result_json = json.loads(raw_result)
    assert result_json["selected_character_text_context"][0]["source_id"] == thread.id
    assert result_json["diagnostics"]["source_type_counts_after_narrowing"][
        "character_text_thread"
    ] == 1


def test_context_search_retrieves_character_text_thread_for_absent_mention(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    player = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="player",
        is_player_character=True,
    )
    rowan = repositories.add_character(
        save_id=save.id,
        name="Rowan",
        role="repair club",
        met=True,
    )
    cass = repositories.add_character(
        save_id=save.id,
        name="Cass",
        role="club president",
        met=True,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=rowan.id,
        title="Rowan",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="player",
        sender_character_id=player.id,
        body="Can you bring the circuit-lantern repair code?",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="character",
        sender_character_id=rowan.id,
        body="I will keep the ember relay code ready.",
    )
    repositories.update_character_text_thread_memory(
        save_id=save.id,
        thread_id=thread.id,
        body=(
            "Phone thread memory: Rowan and Mara discussed the "
            "circuit-lantern ember relay code."
        ),
        message_count=2,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Cass waits by the arcade counter.",
        present_character_ids=[cass.id],
    )
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="I ask whether Rowan has the circuit-lantern code ready.",
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "character_text_thread",
                    "source_id": thread.id,
                    "relevance_note": "Rowan is mentioned and the text is relevant.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "Phone thread memory: Rowan and Mara discussed" in prompt
    assert [item.source_id for item in result.selected_character_text_context] == [
        thread.id
    ]


def test_context_search_suppresses_character_text_thread_outside_audience(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    player = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="player",
        is_player_character=True,
    )
    rowan = repositories.add_character(
        save_id=save.id,
        name="Rowan",
        role="repair club",
        met=True,
    )
    cass = repositories.add_character(
        save_id=save.id,
        name="Cass",
        role="club president",
        met=True,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=rowan.id,
        title="Rowan",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="player",
        sender_character_id=player.id,
        body="Can you bring the circuit-lantern repair code?",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="character",
        sender_character_id=rowan.id,
        body="I will keep the ember relay code ready.",
    )
    repositories.update_character_text_thread_memory(
        save_id=save.id,
        thread_id=thread.id,
        body=(
            "Phone thread memory: Rowan and Mara discussed the "
            "circuit-lantern ember relay code."
        ),
        message_count=2,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Cass waits by the arcade counter.",
        present_character_ids=[cass.id],
    )
    repositories.update_message_body(
        save_id=save.id,
        message_id=player_message.id,
        body="I ask Cass whether the circuit-lantern is ready.",
    )
    provider = RecordingStructuredContextProvider({"selections": []})
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(service.search(save_id=save.id, player_message_id=player_message.id))

    prompt = "\n".join(
        message.body for message in provider.structured_output_requests[0].messages
    )
    assert "Phone thread memory: Rowan and Mara discussed" not in prompt


def test_context_search_excludes_indexed_summaries_from_selection_prompt(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference(repositories)
    source_message = next(
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="summary",
        source_id="summary-indexed",
        title="Indexed summary",
        body="Indexed summary: Mara crossed the ash bridge before dusk.",
        metadata={"fact_type": "summary"},
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara distrusts bells that ring without wind.",
        tags=["bells"],
        source_message_id=source_message.id,
    )
    provider = RecordingStructuredContextProvider(
        {
            "selections": [
                {
                    "source_type": "memory",
                    "source_id": memory.id,
                    "relevance_note": "Bell distrust remains relevant.",
                }
            ]
        }
    )
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    request = provider.structured_output_requests[0]
    source_type_enum = request.schema["properties"]["selections"]["items"][
        "properties"
    ]["source_type"]["enum"]
    prompt = "\n".join(message.body for message in request.messages)
    assert "summary" not in source_type_enum
    assert "Indexed summary: Mara crossed the ash bridge" not in prompt
    assert result.selected_summaries == ()
    assert [item.source_id for item in result.selected_memories] == [memory.id]


def test_context_search_returns_empty_result_when_there_are_no_candidates(
    repositories: PersistenceRepositories,
) -> None:
    save, player_message = _save_with_context_search_preference_without_candidates(
        repositories
    )
    provider = RecordingStructuredContextProvider({"selections": []})
    service = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.search(save_id=save.id, player_message_id=player_message.id)
    )

    assert result == ContextSearchResult(continuity_index_synced=True)
    assert provider.chat_requests == []
    assert provider.structured_output_requests == []
    jobs = _context_search_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    raw_result = jobs[-1]["result_json"]
    assert raw_result is not None
    assert "continuity_index_synced" not in raw_result


def _save_with_context_search_preference(
    repositories: PersistenceRepositories,
    *,
    model_capabilities: list[str] | None = None,
    save_model: bool = True,
) -> tuple[SaveRecord, MessageRecord]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Bridge of Cinders",
        premise="A bridge remembers every oath broken on it.",
        player_role="Oathkeeper",
        content={"starting_scene": "Cinders drift over the bridge stones."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Crossing")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A silver bell rings beneath the bridge.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I cross the bridge and listen for the bell.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Bridge of Cinders"},
        category="scene",
        source_message_id=source_message.id,
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    if save_model:
        repositories.save_provider_model(
            provider="fake",
            model_id="fake-context",
            display_name="Fake Context",
            capabilities=(
                model_capabilities or [ProviderCapability.STRUCTURED_OUTPUT.value]
            ),
        )
    return save, player_message


def _save_with_context_search_preference_without_candidates(
    repositories: PersistenceRepositories,
    *,
    model_capabilities: list[str] | None = None,
) -> tuple[SaveRecord, MessageRecord]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Bridge of Cinders",
        premise="A bridge remembers every oath broken on it.",
        player_role="Oathkeeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Crossing")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I cross the bridge and listen for the bell.",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=(
            model_capabilities or [ProviderCapability.STRUCTURED_OUTPUT.value]
        ),
    )
    return save, player_message


def _configure_tool_fallback(repositories: PersistenceRepositories) -> None:
    repositories.set_model_preference(
        task="tool_call_fallback",
        provider="fallback",
        model_id="fallback-tools",
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-tools",
        display_name="Fallback Tools",
        capabilities=[ProviderCapability.TOOL_CALLING.value],
    )


def _context_search_jobs(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[sqlite3.Row]:
    return _jobs(repositories, save_id, "context_search")


def _candidate_source_id(
    prompt: str,
    *,
    source_type: str,
    expected_text: str,
) -> str:
    marker = f"- [{source_type}:"
    for line in prompt.splitlines():
        if line.startswith(marker) and expected_text in line:
            return line.removeprefix(marker).split("]", 1)[0]
    raise AssertionError(
        f"missing {source_type} candidate containing {expected_text!r}"
    )


def _jobs(
    repositories: PersistenceRepositories,
    save_id: str,
    job_type: str,
) -> list[sqlite3.Row]:
    return list(
        repositories.connection.execute(
            """
            SELECT status, result_json, error
            FROM jobs
            WHERE save_id = ? AND type = ?
            ORDER BY created_at, rowid
            """,
            (save_id, job_type),
        )
    )
