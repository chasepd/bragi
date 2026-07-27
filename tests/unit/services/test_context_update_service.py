from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterRecord,
    EntityLinkRecord,
    LocationRecord,
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
    ProviderConfigStatus,
    ProviderModel,
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.context_assembly import ContextAssemblyService
from bragi.services.message_correction import MessageCorrectionContext
from bragi.services.post_turn_inference import VerifiedPostTurnCoverage
from bragi.services.prompt_inspection import PromptInspectionStore


@pytest.fixture
def repositories(
    tmp_path: Path,
    migrated_database_template: Path,
) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    shutil.copy2(migrated_database_template, database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


class CountingPersistenceRepositories(PersistenceRepositories):
    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.list_counts: dict[str, int] = {}

    def list_messages(
        self,
        save_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[MessageRecord]:
        if not include_deleted:
            self.list_counts["messages"] = self.list_counts.get("messages", 0) + 1
        return super().list_messages(save_id, include_deleted=include_deleted)

    def list_locations(self, save_id: str) -> list[LocationRecord]:
        self.list_counts["locations"] = self.list_counts.get("locations", 0) + 1
        return super().list_locations(save_id)

    def list_characters(self, save_id: str) -> list[CharacterRecord]:
        self.list_counts["characters"] = self.list_counts.get("characters", 0) + 1
        return super().list_characters(save_id)

    def list_active_threads(self, save_id: str) -> list[ActiveThreadRecord]:
        self.list_counts["active_threads"] = (
            self.list_counts.get("active_threads", 0) + 1
        )
        return super().list_active_threads(save_id)

    def list_entity_links(self, save_id: str) -> list[EntityLinkRecord]:
        self.list_counts["entity_links"] = self.list_counts.get("entity_links", 0) + 1
        return super().list_entity_links(save_id)

    def list_memories(self, save_id: str) -> list[MemoryRecord]:
        self.list_counts["memories"] = self.list_counts.get("memories", 0) + 1
        return super().list_memories(save_id)

    def list_world_state(self, save_id: str) -> list[WorldStateRecord]:
        self.list_counts["world_state"] = self.list_counts.get("world_state", 0) + 1
        return super().list_world_state(save_id)

    def list_summaries(self, save_id: str) -> list[SummaryRecord]:
        self.list_counts["summaries"] = self.list_counts.get("summaries", 0) + 1
        return super().list_summaries(save_id)


class RecordingContextUpdateExtractor:
    def __init__(self, extraction: object) -> None:
        self.extraction = extraction
        self.requests: list[object] = []

    async def extract(self, request: object) -> object:
        self.requests.append(request)
        return _with_default_context_update_evidence(self.extraction, request)


def _with_default_context_update_evidence(
    extraction: object,
    request: object,
) -> object:
    module = _context_update_module()
    if not isinstance(extraction, module.ContextUpdateExtraction):
        return extraction
    messages_by_id = {
        message.id: message
        for message in getattr(request, "messages", ())
        if isinstance(message, MessageRecord)
    }

    def with_quote(item: object) -> Any:
        item_value = cast(Any, item)
        source_message_id = item_value.source_message_id
        if not isinstance(source_message_id, str):
            return item
        source = messages_by_id.get(source_message_id)
        if source is None or item_value.evidence_quote:
            return item
        return replace(item_value, evidence_quote=source.body)

    scene = extraction.scene
    if scene is not None:
        scene = cast(Any, with_quote(scene))
    return replace(
        extraction,
        scene=scene,
        locations=tuple(with_quote(item) for item in extraction.locations),
        characters=tuple(with_quote(item) for item in extraction.characters),
        active_threads=tuple(with_quote(item) for item in extraction.active_threads),
        entity_links=tuple(with_quote(item) for item in extraction.entity_links),
        phone_number_exchanges=tuple(
            with_quote(item) for item in extraction.phone_number_exchanges
        ),
    )


class BlockingContextUpdateExtractor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.requests: list[object] = []

    async def extract(self, request: object) -> object:
        self.requests.append(request)
        self.started.set()
        try:
            await asyncio.Future()
            raise AssertionError("blocking context update extractor resumed")
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class RecordingWorldDataEnricher:
    def __init__(self, build_enrichment: object) -> None:
        self.build_enrichment = build_enrichment
        self.requests: list[object] = []

    async def enrich(self, request: object) -> object:
        self.requests.append(request)
        if callable(self.build_enrichment):
            return self.build_enrichment(request)
        return self.build_enrichment


class RecordingContextRegistrySelector:
    def __init__(self, *, selected_body: str | None = None, fail: bool = False) -> None:
        self.selected_body = selected_body
        self.fail = fail
        self.requests: list[object] = []

    async def select_context(self, request: object) -> object:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("selector exploded")
        module = _context_update_module()
        selected: tuple[object, ...] = ()
        if self.selected_body is not None:
            for item in cast(Any, request).candidates:
                if self.selected_body in item.body:
                    selected = (
                        module.ContextRegistryItem(
                            context_source_id=item.context_source_id,
                            source_type=item.source_type,
                            source_id=item.source_id,
                            title=item.title,
                            body=item.body,
                            fact_type=item.fact_type,
                            importance=item.importance,
                            source_message_ids=item.source_message_ids,
                            relevance_note="selected for regression test",
                        ),
                    )
                    break
        return module.ContextRegistrySelection(selected_items=selected)


class RecordingExtractorWithContextSelector(RecordingContextUpdateExtractor):
    def __init__(self, extraction: object) -> None:
        super().__init__(extraction)
        self.selection_requests: list[object] = []

    async def select_context(self, request: object) -> object:
        self.selection_requests.append(request)
        module = _context_update_module()
        return module.ContextRegistrySelection(selected_items=())


class PromptInspectionStructuredProvider:
    provider_name = "fake"

    def __init__(self) -> None:
        self.requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.requests.append(request)
        if request.schema_name == "context_update_context_selection":
            selections_schema = cast(
                dict[str, Any],
                cast(dict[str, Any], request.schema["properties"])["selections"],
            )
            item_schema = cast(dict[str, Any], selections_schema["items"])
            properties = cast(dict[str, Any], item_schema["properties"])
            context_ids = cast(list[str], properties["context_source_id"]["enum"])
            return StructuredOutputResponse(
                data={
                    "selections": [
                        {
                            "context_source_id": context_ids[0],
                            "relevance_note": "Needed for inspection test.",
                        }
                    ]
                },
                provider="fake",
                model_id=request.model_id,
            )
        return StructuredOutputResponse(
            data={
                "scene": {},
                "locations": [],
                "characters": [],
                "active_threads": [],
                "entity_links": [],
            },
            provider="fake",
            model_id=request.model_id,
        )


class SequenceStructuredProvider:
    provider_name = "fake"

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected structured-output request")
        return StructuredOutputResponse(
            data=self.responses.pop(0),
            provider=request.provider,
            model_id=request.model_id,
        )


class SequenceToolCallProvider:
    def __init__(
        self,
        *,
        provider_name: str = "fake",
        responses: list[tuple[ProviderToolCall, ...] | Exception],
    ) -> None:
        self.provider_name = provider_name
        self.responses = responses
        self.tool_call_requests: list[ToolCallRequest] = []

    async def validate_config(self) -> ProviderConfigStatus:
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
        )

    async def list_models(self) -> list[ProviderModel]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("context update tool tests must not call chat")

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("context update tool tests must not generate images")

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected tool-call request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ToolCallResponse(
            tool_calls=response,
            body="",
            provider=request.provider,
            model_id=request.model_id,
        )


def test_update_after_turn_only_sends_selected_prior_context_to_extractors(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    unrelated = repositories.add_memory(
        save_id=save.id,
        body="Ancient pantry trivia that should never reach extraction.",
        tags=["old"],
        importance=0.1,
        source_message_id=player_message.id,
    )
    selected = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra owes Mara a signal flare.",
        tags=["relationship"],
        importance=0.9,
        source_message_id=narrator_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            current_location_name="Beacon Gallery",
            situation="Mara and Ilyra inspect the beacon.",
            reason="The scene stayed focused on Ilyra.",
            confidence=0.86,
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)

    def build_enrichment(request: object) -> object:
        location = cast(Any, request).locations[0]
        return module.WorldDataEnrichment(
            locations=(
                module.LocationWorldDataEnrichment(
                    location_id=location.id,
                    source_message_id=narrator_message.id,
                    evidence_quote="beacon gallery",
                    description="A red glass signal chamber.",
                    reason="Fill the sparse scene location.",
                    confidence=0.8,
                ),
            )
        )

    enricher = RecordingWorldDataEnricher(build_enrichment)
    selector = RecordingContextRegistrySelector(selected_body=selected.body)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
        world_data_enricher=enricher,
        registry_selector=selector,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    extraction_registry = module._registry_text(cast(Any, extractor.requests[0]))
    enrichment_registry = module._world_data_enrichment_registry_text(
        cast(Any, enricher.requests[0])
    )
    assert unrelated.body not in extraction_registry
    assert unrelated.body not in enrichment_registry
    assert selected.body in extraction_registry
    assert selected.body in enrichment_registry
    assert f"[memory:{selected.id}]" in extraction_registry
    assert selector.requests


def test_apply_extraction_records_asymmetric_phone_number_exchange(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
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
        body="Ren gives Mika his number, but she does not share hers yet.",
    )
    extraction = module.context_update_extraction_from_structured_data(
        {
            "scene": {},
            "locations": [],
            "characters": [],
            "active_threads": [],
            "entity_links": [],
            "phone_number_exchanges": [
                {
                    "character_id": npc.id,
                    "direction": "character_has_player_number",
                    "source_message_id": source.id,
                    "evidence_quote": (
                        "Ren gives Mika his number, but she does not share hers yet."
                    ),
                }
            ],
        }
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    service.apply_extraction(
        save_id=save.id,
        extraction=extraction,
        allowed_source_message_ids=(source.id,),
    )

    state = repositories.get_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=npc.id,
    )
    assert state is not None
    assert state.player_has_character_number is False
    assert state.character_has_player_number is True


def test_apply_extraction_filters_marked_narrator_context_sources(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
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
    extraction = module.ContextUpdateExtraction(
        locations=(
            module.ExtractedLocation(
                name="Rejected Room",
                source_message_id=narrator_message.id,
                evidence_quote="The intimate moment is kept off-screen.",
            ),
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id=player_message.id,
                evidence_quote="climb toward the beacon lens",
            ),
        ),
    )

    result = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    ).apply_extraction(
        save_id=save.id,
        extraction=extraction,
        allowed_source_message_ids=(player_message.id, narrator_message.id),
        completed_messages=tuple(repositories.list_messages(save.id)),
    )

    assert [location.name for location in result.locations] == ["Beacon Gallery"]


def test_apply_extraction_drops_invalid_scene_source_without_rewriting_provenance(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id="not-a-completed-turn-message",
            evidence_quote="Captain Ilyra",
            current_location_name="Beacon Gallery",
            situation="The lens is waking up.",
        ),
        locations=(
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id=narrator_message.id,
                evidence_quote="beacon gallery",
                description="A hot room above the keep wall.",
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    result = service.apply_extraction(
        save_id=save.id,
        extraction=extraction,
        allowed_source_message_ids=(narrator_message.id,),
        completed_messages=(narrator_message,),
    )

    assert result.scene_snapshot is None
    assert repositories.get_scene_snapshot(save.id) is None
    assert [location.name for location in result.locations] == ["Beacon Gallery"]
    locations = repositories.list_locations(save.id)
    assert [location.source_message_id for location in locations] == [
        narrator_message.id
    ]
    audit_source_ids = {
        source_id
        for row in repositories.list_context_update_audit(save.id)
        for source_id in row.source_message_ids
    }
    assert audit_source_ids == {narrator_message.id}


def test_apply_extraction_infers_reciprocal_phone_exchange(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
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
        body="Mika Arai smiles at Ren Takahashi, and they exchange numbers.",
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    service.apply_extraction(
        save_id=save.id,
        extraction=module.ContextUpdateExtraction(),
        allowed_source_message_ids=(source.id,),
        completed_messages=(source,),
    )

    state = repositories.get_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=npc.id,
    )
    assert state is not None
    assert state.player_has_character_number is True
    assert state.character_has_player_number is True


def test_apply_extraction_infers_one_way_phone_exchange(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
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
        body="Mika Arai gives Ren Takahashi her number before leaving.",
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    service.apply_extraction(
        save_id=save.id,
        extraction=module.ContextUpdateExtraction(),
        allowed_source_message_ids=(source.id,),
        completed_messages=(source,),
    )

    state = repositories.get_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=npc.id,
    )
    assert state is not None
    assert state.player_has_character_number is True
    assert state.character_has_player_number is False


def test_apply_extraction_ignores_ambiguous_stay_in_touch(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
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
        body="Mika Arai tells Ren Takahashi they should stay in touch.",
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    service.apply_extraction(
        save_id=save.id,
        extraction=module.ContextUpdateExtraction(),
        allowed_source_message_ids=(source.id,),
        completed_messages=(source,),
    )

    assert (
        repositories.get_character_contact_state(
            save_id=save.id,
            player_character_id=player.id,
            character_id=npc.id,
        )
        is None
    )


def test_apply_extraction_does_not_treat_uninvolved_player_message_as_exchange(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
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
        role="player",
        speaker_name="Ren Takahashi",
        body="Mika Arai and Sora exchange numbers after class.",
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    service.apply_extraction(
        save_id=save.id,
        extraction=module.ContextUpdateExtraction(),
        allowed_source_message_ids=(source.id,),
        completed_messages=(source,),
    )

    assert (
        repositories.get_character_contact_state(
            save_id=save.id,
            player_character_id=player.id,
            character_id=npc.id,
        )
        is None
    )


def test_update_after_turn_can_select_curated_observation_prior_context(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="open_thread",
        claim="The red lens warning may return.",
        evidence_quote="Captain Ilyra steadies Mara in the beacon gallery.",
        source_message_ids=[narrator_message.id],
        scope="save",
        status="accepted",
        confidence=0.86,
        tags=["beacon"],
    )
    context_source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id=observation.id,
        title="Red lens warning",
        body="The red lens warning means riders may still be near the tower.",
        metadata={
            "observation_id": observation.id,
            "observation_type": observation.observation_type,
            "source_message_ids": [narrator_message.id],
            "curation_action": "save_context",
        },
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            current_location_name="Beacon Gallery",
            situation="Mara and Ilyra inspect the beacon.",
            reason="The scene stayed focused on Ilyra.",
            confidence=0.86,
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    selector = RecordingContextRegistrySelector(selected_body=context_source.body)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
        registry_selector=selector,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    selector_candidates = cast(Any, selector.requests[0]).candidates
    assert context_source.id in {
        item.context_source_id for item in selector_candidates
    }
    extraction_registry = module._registry_text(cast(Any, extractor.requests[0]))
    assert context_source.body in extraction_registry
    assert f"[observation:{observation.id}]" in extraction_registry


def test_update_after_turn_reuses_read_snapshot_for_selection_and_request(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    counting = CountingPersistenceRepositories(repositories.connection)
    save, player_message, narrator_message = _save_with_completed_turn(counting)
    counting.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=narrator_message.id,
    )
    counting.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        source_message_id=narrator_message.id,
    )
    counting.add_active_thread(
        save_id=save.id,
        title="Relight the beacon",
        description="The cracked lens still needs a signal.",
        source_message_id=narrator_message.id,
    )
    counting.add_memory(
        save_id=save.id,
        body="Ilyra owes Mara a signal flare.",
        tags=["ilyra"],
        source_message_id=narrator_message.id,
    )
    extraction = module.ContextUpdateExtraction()

    class CountingExtractor(RecordingContextUpdateExtractor):
        counts_at_extract: dict[str, int]

        async def extract(self, request: object) -> object:
            self.counts_at_extract = dict(counting.list_counts)
            return await super().extract(request)

    extractor = CountingExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=counting,
        extractor=extractor,
    )
    counting.list_counts.clear()

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    request = cast(Any, extractor.requests[0])
    assert [message.id for message in request.messages] == [
        player_message.id,
        narrator_message.id,
    ]
    counts_at_extract = extractor.counts_at_extract
    assert counts_at_extract["messages"] <= 2
    assert counts_at_extract["locations"] <= 2
    assert counts_at_extract["characters"] <= 2
    assert counts_at_extract["active_threads"] <= 4
    assert counts_at_extract["memories"] <= 2
    assert counts_at_extract["world_state"] <= 2
    assert counts_at_extract["summaries"] <= 2


def test_update_after_turn_marks_lifecycle_job_cancelled_on_task_cancel(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extractor = BlockingContextUpdateExtractor()
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    async def run_and_cancel() -> None:
        task = asyncio.create_task(
            service.update_after_turn(
                save_id=save.id,
                source_message_ids=(player_message.id, narrator_message.id),
            )
        )
        try:
            await asyncio.wait_for(extractor.started.wait(), timeout=1.0)
            task.cancel()
            await asyncio.wait_for(extractor.cancelled.wait(), timeout=1.0)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(run_and_cancel())

    context_jobs = repositories.connection.execute(
        """
        SELECT status, result_json, error
        FROM jobs
        WHERE save_id = ? AND type = 'context_update'
        ORDER BY created_at, rowid
        """,
        (save.id,),
    ).fetchall()
    assert [(job["status"], job["error"]) for job in context_jobs] == [
        ("cancelled", "Context update cancelled")
    ]
    assert context_jobs[0]["result_json"] is None


def test_message_correction_archives_context_from_old_message(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    unrelated_location = repositories.add_location(
        save_id=save.id,
        name="Gatehouse",
        source_message_id=player_message.id,
    )
    unrelated_character = repositories.add_character(
        save_id=save.id,
        name="Gate Guard",
        source_message_id=player_message.id,
    )
    old_location = repositories.add_location(
        save_id=save.id,
        name="Ash-Choked Gallery",
        source_message_id=narrator_message.id,
    )
    old_character = repositories.add_character(
        save_id=save.id,
        name="Ash Phantom",
        location_id=old_location.id,
        source_message_id=narrator_message.id,
    )
    old_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Evade the ash phantom",
        source_message_id=narrator_message.id,
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=old_character.id,
        target_type="location",
        target_id=old_location.id,
        relation="present_at",
        source_message_id=narrator_message.id,
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=unrelated_character.id,
        target_type="location",
        target_id=unrelated_location.id,
        relation="guarding",
        source_message_id=narrator_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=old_location.id,
        situation="An ash phantom blocks the gallery.",
        source_message_id=narrator_message.id,
    )
    old_memory = repositories.add_memory(
        save_id=save.id,
        body="An ash phantom blocked the gallery.",
        tags=[],
        source_message_id=narrator_message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="threat.ash_phantom",
        value={"status": "blocking gallery"},
        source_message_id=narrator_message.id,
    )
    old_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=player_message.id,
        covers_message_end_id=narrator_message.id,
        body="Mara encountered an ash phantom in the gallery.",
        provider="fake",
        model="fake-summary",
    )
    old_context_source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id=old_memory.id,
        title="Old ash phantom memory",
        body="The ash phantom should not be selected during correction.",
        metadata={
            "indexed_by": "continuity_index",
            "source_message_ids": [narrator_message.id],
        },
    )
    old_message_context_source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="message",
        source_id=narrator_message.id,
        title="Old narrator context",
        body="The old narrator message introduced the ash phantom.",
        metadata={"indexed_by": "continuity_index"},
    )
    kept_context_source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="kept-memory",
        title="Kept beacon memory",
        body="The beacon lens was already important.",
        metadata={
            "indexed_by": "continuity_index",
            "source_message_ids": [player_message.id],
        },
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="character",
        entity_id=old_character.id,
        field_path="status",
        proposed_value="blocking the gallery",
        source_message_ids=[narrator_message.id],
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            current_location_name="Beacon Gallery",
            situation="Captain Ilyra steadies the beacon lens.",
            present_character_names=("Captain Ilyra",),
        ),
        locations=(
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id=narrator_message.id,
                description="A steady beacon chamber.",
            ),
        ),
        characters=(
            module.ExtractedCharacter(
                name="Captain Ilyra",
                source_message_id=narrator_message.id,
                role="Signal captain",
                location_name="Beacon Gallery",
                met=True,
            ),
        ),
        active_threads=(
            module.ExtractedActiveThread(
                title="Hold the beacon steady",
                source_message_id=narrator_message.id,
                status="active",
                visibility="scene",
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    selector = RecordingContextRegistrySelector(selected_body=kept_context_source.body)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
        registry_selector=selector,
    )

    asyncio.run(
        service.update_after_message_correction(
            save_id=save.id,
            source_message_id=narrator_message.id,
            correction_context=MessageCorrectionContext(
                message_id=narrator_message.id,
                previous_body="An ash phantom blocks the gallery.",
                new_body="Captain Ilyra steadies the beacon lens.",
                diff_unified=(
                    "-An ash phantom blocks the gallery.\n"
                    "+Captain Ilyra steadies the beacon lens."
                ),
            ),
        )
    )

    request = cast(Any, extractor.requests[0])
    assert old_location.id not in {location.id for location in request.locations}
    assert old_character.id not in {character.id for character in request.characters}
    assert old_thread.id not in {thread.id for thread in request.active_threads}
    assert old_memory.id not in {memory.id for memory in request.memories}
    assert all(
        state.source_message_id != narrator_message.id
        for state in request.world_state
    )
    assert old_summary.id not in {summary.id for summary in request.summaries}
    assert all(
        link.source_message_id != narrator_message.id
        for link in request.entity_links
    )
    selector_candidates = cast(Any, selector.requests[0]).candidates
    assert old_context_source.id not in {
        item.context_source_id for item in selector_candidates
    }
    assert old_message_context_source.id not in {
        item.context_source_id for item in selector_candidates
    }
    assert selector_candidates
    assert all(
        narrator_message.id not in item.source_message_ids
        for item in selector_candidates
    )
    assert all(
        "ash phantom" not in item.body.casefold()
        for item in selector_candidates
    )
    assert repositories.get_location(old_location.id) is None
    assert repositories.get_character(old_character.id) is None
    assert repositories.get_active_thread(old_thread.id) is None
    assert repositories.list_entity_links(save.id) == []
    assert repositories.list_context_update_suggestions(save.id, status="pending") == []
    assert [
        suggestion.status
        for suggestion in repositories.list_context_update_suggestions(save.id)
    ] == ["expired"]
    assert {location.name for location in repositories.list_locations(save.id)} == {
        "Beacon Gallery",
        unrelated_location.name,
    }
    assert {character.name for character in repositories.list_characters(save.id)} == {
        "Captain Ilyra",
        unrelated_character.name,
    }
    assert [thread.title for thread in repositories.list_active_threads(save.id)] == [
        "Hold the beacon steady"
    ]
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.situation == "Captain Ilyra steadies the beacon lens."


def test_message_correction_preserves_protected_character_from_old_message(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    protected_character = repositories.add_character(
        save_id=save.id,
        name="Heather Langley",
        aliases=["Heather"],
        role="Coalition advocate and Emma's mother",
        known_state="Rich scenario profile that should not be thrown away.",
        appearance="Carefully guarded warmth.",
        personality="Warm, approachable, and fiercely guarded.",
        source_message_id=narrator_message.id,
        protected_from_maintenance=True,
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Heather",
                source_message_id=narrator_message.id,
                status="present after the corrected exchange",
                reason="The corrected narrator message still refers to Heather.",
                confidence=0.93,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    asyncio.run(
        service.update_after_message_correction(
            save_id=save.id,
            source_message_id=narrator_message.id,
            correction_context=MessageCorrectionContext(
                message_id=narrator_message.id,
                previous_body="Heather leaves the room.",
                new_body="Heather stays and talks.",
                diff_unified="-Heather leaves the room.\n+Heather stays and talks.",
            ),
        )
    )

    request = cast(Any, extractor.requests[0])
    assert protected_character.id in {character.id for character in request.characters}

    characters = repositories.list_characters(save.id)
    assert [character.id for character in characters] == [protected_character.id]
    updated = characters[0]
    assert updated.name == "Heather Langley"
    assert updated.aliases == ["Heather"]
    assert updated.protected_from_maintenance is True
    assert updated.role == "Coalition advocate and Emma's mother"
    assert (
        updated.known_state == "Rich scenario profile that should not be thrown away."
    )
    assert updated.appearance == "Carefully guarded warmth."
    assert updated.personality == "Warm, approachable, and fiercely guarded."
    assert updated.status == "present after the corrected exchange"


def test_apply_extraction_appends_character_history_without_replacing_existing_text(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        known_state="Guarding the cracked red lens.",
        met=True,
        source_message_id=narrator_message.id,
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    service.apply_extraction(
        save_id=save.id,
        extraction=module.ContextUpdateExtraction(
            characters=(
                module.ExtractedCharacter(
                    name="Captain Ilyra",
                    source_message_id=narrator_message.id,
                    evidence_quote="Captain Ilyra",
                    known_state="Revealed the lens key phrase to Mara.",
                    reason="The completed turn adds durable character history.",
                    confidence=0.91,
                ),
            ),
        ),
    )

    updated = repositories.get_character(character.id)
    assert updated is not None
    assert updated.history == (
        "Guarding the cracked red lens.\n\n"
        "Revealed the lens key phrase to Mara."
    )
    assert updated.known_state == updated.history
    assert repositories.list_context_update_suggestions(save.id, status="pending") == []


def test_structured_context_update_records_inspection_entries_for_narrator(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra promised Mara the lens key.",
        tags=["promise"],
        importance=0.9,
        source_message_id=narrator_message.id,
    )
    store = PromptInspectionStore()
    provider = PromptInspectionStructuredProvider()
    updater = module.StructuredProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        prompt_inspection_store=store,
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=updater,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    entries = store.entries_for_message(narrator_message.id)
    assert [entry.kind for entry in entries] == [
        "context_selection",
        "context_extraction",
    ]
    prompt_text = store.prompt_for_message(narrator_message.id) or ""
    assert "Source cards" in prompt_text
    assert "Context selection" in prompt_text
    assert "Context extraction" in prompt_text
    assert "Raw requests" in prompt_text
    assert provider.requests
    assert all(request.max_output_tokens == 1024 for request in provider.requests)
    assert '"schema_name": "context_update_context_selection"' in prompt_text


def test_tool_calling_context_update_applies_valid_tool_calls(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="call-scene",
                    name="update_scene_snapshot",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "beacon gallery",
                            "current_location_name": "Beacon Gallery",
                            "situation": "Mara and Ilyra steady the beacon lens.",
                            "present_character_names": ["Captain Ilyra"],
                            "confidence": 0.86,
                        }
                    ),
                ),
                ProviderToolCall(
                    id="call-location",
                    name="upsert_location",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "beacon gallery",
                            "name": "Beacon Gallery",
                            "description": "A tower chamber around the beacon lens.",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="call-character",
                    name="upsert_character",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "Captain Ilyra",
                            "name": "Captain Ilyra",
                            "role": "Signal captain",
                            "met": True,
                            "current_intent": "Keep Mara away from the lens lever.",
                            "cooperation_conditions": (
                                "Helps after Mara proves the lens can hold."
                            ),
                            "location_name": "Beacon Gallery",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="call-thread",
                    name="upsert_active_thread",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": player_message.id,
                            "evidence_quote": "beacon lens",
                            "title": "Repair the beacon",
                            "description": "Mara is trying to restore the beacon.",
                            "status": "active",
                            "priority": 4,
                            "visibility": "scene",
                        }
                    ),
                ),
            )
        ]
    )
    updater = module.ToolCallingProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=updater,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert result.scene_snapshot is not None
    assert result.scene_snapshot.situation == (
        "Mara and Ilyra steady the beacon lens."
    )
    assert [location.name for location in result.locations] == ["Beacon Gallery"]
    assert [character.name for character in result.characters] == ["Captain Ilyra"]
    assert result.characters[0].current_intent == (
        "Keep Mara away from the lens lever."
    )
    assert result.characters[0].cooperation_conditions == (
        "Helps after Mara proves the lens can hold."
    )
    assert [thread.title for thread in result.active_threads] == [
        "Repair the beacon"
    ]
    assert len(provider.tool_call_requests) == 1
    assert [tool.name for tool in provider.tool_call_requests[0].tools] == [
        "update_scene_snapshot",
        "upsert_location",
        "upsert_character",
        "upsert_active_thread",
        "link_entities",
        "record_phone_number_exchange",
    ]


def test_tool_calling_context_update_applies_phone_number_exchange(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
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
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika Arai gives Ren Takahashi her number before leaving.",
    )
    provider = SequenceToolCallProvider(
        responses=[
            (),
            (
                ProviderToolCall(
                    id="call-phone",
                    name="record_phone_number_exchange",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "gives Ren Takahashi her number",
                            "character_id": npc.id,
                            "direction": "player_has_character_number",
                        }
                    ),
                ),
            )
        ]
    )
    updater = module.ToolCallingProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=updater,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(narrator_message.id,),
        )
    )

    state = repositories.get_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=npc.id,
    )
    assert state is not None
    assert state.player_has_character_number is True
    assert state.character_has_player_number is False


def test_structured_context_update_schema_extracts_character_agency_fields(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)

    schema = module._context_update_schema((player_message, narrator_message))
    character_schema = schema["properties"]["characters"]["items"]
    character_properties = character_schema["properties"]
    for field_name in (
        "age",
        "goals",
        "motivations",
        "current_intent",
        "boundaries",
        "attitude_toward_player",
        "cooperation_conditions",
    ):
        assert field_name in character_properties

    extraction = module.context_update_extraction_from_structured_data(
        {
            "scene": {},
            "locations": [],
            "characters": [
                {
                    "name": "Captain Ilyra",
                    "source_message_id": narrator_message.id,
                    "age": "late 40s",
                    "goals": "Keep the red lens under control.",
                    "motivations": "Protect the lower village.",
                    "current_intent": "Demand proof before sharing the failsafe.",
                    "boundaries": "Will not abandon the tower.",
                    "attitude_toward_player": "Wary trust after the repair.",
                    "cooperation_conditions": (
                        "Helps after Mara shows the brass warrant."
                    ),
                }
            ],
            "active_threads": [],
            "entity_links": [],
        }
    )

    assert save.id
    assert len(extraction.characters) == 1
    character = extraction.characters[0]
    assert character.age == "late 40s"
    assert character.goals == "Keep the red lens under control."
    assert character.motivations == "Protect the lower village."
    assert character.current_intent == "Demand proof before sharing the failsafe."
    assert character.boundaries == "Will not abandon the tower."
    assert character.attitude_toward_player == "Wary trust after the repair."
    assert character.cooperation_conditions == (
        "Helps after Mara shows the brass warrant."
    )


def test_context_update_schema_requires_evidence_quotes(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    _save, player_message, narrator_message = _save_with_completed_turn(repositories)

    schema = module._context_update_schema((player_message, narrator_message))
    properties = schema["properties"]
    item_paths = (
        properties["scene"],
        properties["locations"]["items"],
        properties["characters"]["items"],
        properties["active_threads"]["items"],
        properties["entity_links"]["items"],
        properties["phone_number_exchanges"]["items"],
    )

    for item_schema in item_paths:
        assert "evidence_quote" in item_schema["properties"]
        assert "evidence_quote" in item_schema["required"]


def test_world_data_enrichment_schema_requires_evidence_quotes(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Repair the beacon",
        status="active",
        source_message_id=narrator_message.id,
    )
    character = repositories.add_character(save_id=save.id, name="Captain Ilyra")
    request = module.WorldDataEnrichmentRequest(
        save_id=save.id,
        messages=(player_message, narrator_message),
        scenario_context="Scenario: Ashfall Keep",
        scene_snapshot=None,
        locations=(location,),
        active_threads=(thread,),
        characters=(character,),
    )

    schema = module._world_data_enrichment_schema(request)
    tool_schemas = module._world_data_enrichment_tool_schemas(request)
    item_schemas = (
        schema["properties"]["locations"]["items"],
        schema["properties"]["active_threads"]["items"],
        schema["properties"]["characters"]["items"],
        tool_schemas["enrich_location"],
        tool_schemas["enrich_active_thread"],
        tool_schemas["enrich_character"],
    )

    for item_schema in item_schemas:
        assert "evidence_quote" in item_schema["properties"]
        assert "evidence_quote" in item_schema["required"]


def test_structured_context_update_drops_items_with_ungrounded_evidence(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceStructuredProvider(
        [
            {
                "scene": {
                    "source_message_id": narrator_message.id,
                    "evidence_quote": "the ruby library",
                    "situation": "The action moves to the library.",
                },
                "locations": [
                    {
                        "name": "Beacon Gallery",
                        "source_message_id": narrator_message.id,
                        "evidence_quote": "beacon gallery",
                        "description": "A room around the beacon lens.",
                    },
                    {
                        "name": "Ruby Library",
                        "source_message_id": narrator_message.id,
                        "evidence_quote": "",
                        "description": "Unsupported by the source.",
                    },
                ],
                "characters": [],
                "active_threads": [],
                "entity_links": [],
                "phone_number_exchanges": [],
            }
        ]
    )
    updater = module.StructuredProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )
    request = module.ContextUpdateRequest(
        save_id=save.id,
        messages=(player_message, narrator_message),
        scene_snapshot=None,
        locations=(),
        characters=(),
        active_threads=(),
        entity_links=(),
    )

    extraction = asyncio.run(updater.extract(request))

    assert extraction.scene is None
    assert [location.name for location in extraction.locations] == ["Beacon Gallery"]


def test_apply_extraction_drops_items_with_ungrounded_completed_message_evidence(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    result = service.apply_extraction(
        save_id=save.id,
        extraction=module.ContextUpdateExtraction(
            locations=(
                module.ExtractedLocation(
                    name="Beacon Gallery",
                    source_message_id=narrator_message.id,
                    evidence_quote="beacon gallery",
                    description="A room around the beacon lens.",
                ),
                module.ExtractedLocation(
                    name="Ruby Library",
                    source_message_id=narrator_message.id,
                    evidence_quote="ruby library",
                    description="Unsupported by the completed turn.",
                ),
                module.ExtractedLocation(
                    name="Missing Quote",
                    source_message_id=narrator_message.id,
                    description="Missing evidence quote.",
                ),
            )
        ),
        allowed_source_message_ids=(player_message.id, narrator_message.id),
        completed_messages=(player_message, narrator_message),
    )

    assert [location.name for location in result.locations] == ["Beacon Gallery"]
    assert [location.name for location in repositories.list_locations(save.id)] == [
        "Beacon Gallery"
    ]


def test_structured_world_data_enrichment_drops_ungrounded_evidence(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    provider = SequenceStructuredProvider(
        [
            {
                "locations": [
                    {
                        "location_id": location.id,
                        "source_message_id": narrator_message.id,
                        "evidence_quote": "beacon gallery",
                        "description": "A red glass chamber around the beacon.",
                    },
                    {
                        "location_id": location.id,
                        "source_message_id": narrator_message.id,
                        "evidence_quote": "ruby library",
                        "description": "Unsupported library detail.",
                    },
                    {
                        "location_id": location.id,
                        "source_message_id": narrator_message.id,
                        "description": "Missing evidence quote.",
                    },
                ],
                "active_threads": [],
                "characters": [],
            }
        ]
    )
    updater = module.StructuredProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    enrichment = asyncio.run(
        updater.enrich(
            module.WorldDataEnrichmentRequest(
                save_id=save.id,
                messages=(player_message, narrator_message),
                scenario_context="Scenario: Ashfall Keep",
                scene_snapshot=None,
                locations=(location,),
                active_threads=(),
            )
        )
    )

    assert [item.description for item in enrichment.locations] == [
        "A red glass chamber around the beacon."
    ]


def test_tool_calling_context_update_selects_prior_context_with_feedback(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    candidate = module.ContextRegistryItem(
        context_source_id="context-memory-1",
        source_type="memory",
        source_id="memory-1",
        title="Ilyra promise",
        body="Captain Ilyra owes Mara a signal flare.",
        fact_type="promise",
        importance=0.9,
    )
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="call-bad",
                    name="select_prior_context",
                    arguments_json=json.dumps(
                        {
                            "context_source_id": "missing-context",
                            "relevance_note": "wrong id",
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-good",
                    name="select_prior_context",
                    arguments_json=json.dumps(
                        {
                            "context_source_id": candidate.context_source_id,
                            "relevance_note": "Needed to resolve Ilyra's promise.",
                        }
                    ),
                ),
            ),
        ]
    )
    updater = module.ToolCallingProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    selection = asyncio.run(
        updater.select_context(
            module.ContextRegistrySelectionRequest(
                save_id=save.id,
                messages=(player_message, narrator_message),
                scene_snapshot=None,
                locations=(),
                characters=(),
                active_threads=(),
                candidates=(candidate,),
            )
        )
    )

    assert [item.context_source_id for item in selection.selected_items] == [
        candidate.context_source_id
    ]
    assert selection.selected_items[0].relevance_note == (
        "Needed to resolve Ilyra's promise."
    )
    assert len(provider.tool_call_requests) == 2
    retry_messages = provider.tool_call_requests[1].messages
    assert retry_messages[-1].role == "tool"
    assert "missing-context" in retry_messages[-1].body


def test_tool_calling_context_update_enriches_world_data(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=narrator_message.id,
    )
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="call-enrich-location",
                    name="enrich_location",
                    arguments_json=json.dumps(
                        {
                            "location_id": location.id,
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "beacon gallery",
                            "description": "A red glass chamber around the beacon.",
                            "status": "occupied by Mara and Ilyra",
                            "confidence": 0.8,
                            "reason": "The sparse location needs stable detail.",
                        }
                    ),
                ),
            )
        ]
    )
    updater = module.ToolCallingProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    enrichment = asyncio.run(
        updater.enrich(
            module.WorldDataEnrichmentRequest(
                save_id=save.id,
                messages=(player_message, narrator_message),
                scenario_context="Scenario: Ashfall Keep",
                scene_snapshot=None,
                locations=(location,),
                active_threads=(),
            )
        )
    )

    assert [item.location_id for item in enrichment.locations] == [location.id]
    assert enrichment.locations[0].description == (
        "A red glass chamber around the beacon."
    )
    assert len(provider.tool_call_requests) == 1
    assert [tool.name for tool in provider.tool_call_requests[0].tools] == [
        "enrich_location",
        "enrich_active_thread",
        "enrich_character",
    ]


def test_tool_calling_context_update_enrichment_retries_invalid_ids(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=narrator_message.id,
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Repair the beacon",
        description="Mara is trying to restore the beacon.",
        status="active",
        source_message_id=narrator_message.id,
    )
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="bad-source",
                    name="enrich_location",
                    arguments_json=json.dumps(
                        {
                            "location_id": location.id,
                            "source_message_id": "missing-message",
                            "evidence_quote": "beacon gallery",
                            "description": "Bad source.",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="bad-location",
                    name="enrich_location",
                    arguments_json=json.dumps(
                        {
                            "location_id": "missing-location",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "beacon gallery",
                            "description": "Bad location.",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="bad-thread",
                    name="enrich_active_thread",
                    arguments_json=json.dumps(
                        {
                            "active_thread_id": "missing-thread",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "beacon gallery",
                            "description": "Bad thread.",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="bad-quote",
                    name="enrich_location",
                    arguments_json=json.dumps(
                        {
                            "location_id": location.id,
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "the ruby library",
                            "description": "Bad quote.",
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="good-location",
                    name="enrich_location",
                    arguments_json=json.dumps(
                        {
                            "location_id": location.id,
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "beacon gallery",
                            "description": "A red glass chamber around the beacon.",
                            "status": "occupied by Mara and Ilyra",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="good-thread",
                    name="enrich_active_thread",
                    arguments_json=json.dumps(
                        {
                            "active_thread_id": thread.id,
                            "source_message_id": player_message.id,
                            "evidence_quote": "beacon lens",
                            "description": "The beacon repair remains urgent.",
                            "related_entities": [location.id],
                        }
                    ),
                ),
            ),
        ]
    )
    updater = module.ToolCallingProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    enrichment = asyncio.run(
        updater.enrich(
            module.WorldDataEnrichmentRequest(
                save_id=save.id,
                messages=(player_message, narrator_message),
                scenario_context="Scenario: Ashfall Keep",
                scene_snapshot=None,
                locations=(location,),
                active_threads=(thread,),
            )
        )
    )

    assert [item.location_id for item in enrichment.locations] == [location.id]
    assert [item.active_thread_id for item in enrichment.active_threads] == [
        thread.id
    ]
    assert len(provider.tool_call_requests) == 2
    feedback = "\n".join(
        message.body for message in provider.tool_call_requests[1].messages
    )
    assert "source_message_id is not in the completed turn: missing-message" in feedback
    assert "location_id must be one of" in feedback
    assert "active_thread_id must be one of" in feedback
    assert "evidence_quote not found in source_message_id" in feedback


def test_tool_calling_context_update_records_prompt_inspection(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="call-scene",
                    name="update_scene_snapshot",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "beacon gallery",
                            "current_location_name": "Beacon Gallery",
                        }
                    ),
                ),
            )
        ]
    )
    store = PromptInspectionStore()
    updater = module.ToolCallingProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
        prompt_inspection_store=store,
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=updater,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(_player_message.id, narrator_message.id),
        )
    )

    prompt_text = store.prompt_for_message(narrator_message.id) or ""
    assert "Context extraction tool calls" in prompt_text
    assert "Tool messages" in prompt_text
    assert "update_scene_snapshot" in prompt_text
    assert '"model_id": "fake-tools"' in prompt_text
    assert [entry.kind for entry in store.entries_for_message(narrator_message.id)] == [
        "context_extraction_tool_calls"
    ]


def test_tool_calling_context_update_feedback_retains_accepted_calls(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="accepted-location",
                    name="upsert_location",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "beacon gallery",
                            "name": "Beacon Gallery",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="bad-scene",
                    name="update_scene_snapshot",
                    arguments_json='{"source_message_id":',
                ),
            ),
            (
                ProviderToolCall(
                    id="fixed-scene",
                    name="update_scene_snapshot",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "beacon gallery",
                            "current_location_name": "Beacon Gallery",
                            "situation": "The beacon lens hums awake.",
                        }
                    ),
                ),
            ),
        ]
    )
    updater = module.ToolCallingProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )
    request = module.ContextUpdateRequest(
        save_id=save.id,
        messages=(narrator_message,),
        scene_snapshot=None,
        locations=(),
        characters=(),
        active_threads=(),
        entity_links=(),
    )

    extraction = asyncio.run(updater.extract(request))

    assert [location.name for location in extraction.locations] == [
        "Beacon Gallery"
    ]
    assert extraction.scene is not None
    assert extraction.scene.situation == "The beacon lens hums awake."
    assert len(provider.tool_call_requests) == 2
    retry_messages = provider.tool_call_requests[1].messages
    tool_results = [message for message in retry_messages if message.role == "tool"]
    assert len(tool_results) == 2
    assert any('"status": "accepted"' in message.body for message in tool_results)
    assert any('"status": "error"' in message.body for message in tool_results)
    assert any("corrected arguments" in message.body for message in tool_results)


def test_tool_calling_context_update_rejects_ungrounded_evidence_quote(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id=f"bad-quote-{index}",
                    name="update_scene_snapshot",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "the ruby library",
                            "situation": "Mara studies the library.",
                        }
                    ),
                ),
            )
            for index in range(3)
        ]
    )
    updater = module.ToolCallingProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )
    request = module.ContextUpdateRequest(
        save_id=save.id,
        messages=(narrator_message,),
        scene_snapshot=None,
        locations=(),
        characters=(),
        active_threads=(),
        entity_links=(),
    )

    with pytest.raises(Exception) as exc_info:
        asyncio.run(updater.extract(request))

    assert "evidence_quote not found" in str(exc_info.value)
    assert len(provider.tool_call_requests) == 3
    retry_messages = provider.tool_call_requests[1].messages
    assert any(
        "Call exactly one tool again" in message.body
        for message in retry_messages
        if message.role == "tool"
    )


def test_tool_calling_context_update_accepts_normalized_evidence_quote(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="normalized-scene",
                    name="update_scene_snapshot",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "`Captain   Ilyra`",
                            "situation": "Ilyra steadies Mara in the gallery.",
                        }
                    ),
                ),
            )
        ]
    )
    updater = module.ToolCallingProviderContextUpdater(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )
    request = module.ContextUpdateRequest(
        save_id=save.id,
        messages=(narrator_message,),
        scene_snapshot=None,
        locations=(),
        characters=(),
        active_threads=(),
        entity_links=(),
    )

    extraction = asyncio.run(updater.extract(request))

    assert extraction.scene is not None
    assert extraction.scene.situation == "Ilyra steadies Mara in the gallery."
    assert len(provider.tool_call_requests) == 1


def test_tool_calling_context_update_uses_tool_fallback_after_feedback_exhaustion(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    primary = SequenceToolCallProvider(
        provider_name="primary",
        responses=[
            (
                ProviderToolCall(
                    id=f"bad-{index}",
                    name="update_scene_snapshot",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": "missing-message",
                            "evidence_quote": "Bad source",
                            "situation": f"Bad source {index}",
                        }
                    ),
                ),
            )
            for index in range(3)
        ],
    )
    fallback = SequenceToolCallProvider(
        provider_name="fallback",
        responses=[
            (
                ProviderToolCall(
                    id="fallback-scene",
                    name="update_scene_snapshot",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "Captain Ilyra",
                            "situation": "The fallback model extracts the scene.",
                        }
                    ),
                ),
            )
        ],
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
    updater = module.ToolCallingProviderContextUpdater(
        provider=primary,
        provider_name="primary",
        model_id="primary-tools",
        repositories=repositories,
        providers={"primary": primary, "fallback": fallback},
    )
    request = module.ContextUpdateRequest(
        save_id=save.id,
        messages=(narrator_message,),
        scene_snapshot=None,
        locations=(),
        characters=(),
        active_threads=(),
        entity_links=(),
    )

    extraction = asyncio.run(updater.extract(request))

    assert len(primary.tool_call_requests) == 3
    assert len(fallback.tool_call_requests) == 1
    assert extraction.scene is not None
    assert extraction.scene.situation == "The fallback model extracts the scene."
    assert fallback.tool_call_requests[0].provider == "fallback"
    assert fallback.tool_call_requests[0].model_id == "fallback-tools"


def test_update_after_turn_falls_back_to_deterministic_prior_context_on_selector_error(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    low_value = repositories.add_memory(
        save_id=save.id,
        body="Low value cellar dust.",
        tags=["trivia"],
        importance=0.05,
        source_message_id=player_message.id,
    )
    high_value = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra promised Mara the lens key.",
        tags=["promise"],
        importance=0.95,
        source_message_id=narrator_message.id,
    )
    extractor = RecordingContextUpdateExtractor(module.ContextUpdateExtraction())
    selector = RecordingContextRegistrySelector(fail=True)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
        registry_selector=selector,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    request = cast(Any, extractor.requests[0])
    selected_bodies = [item.body for item in request.prior_context]
    assert high_value.body in selected_bodies
    assert low_value.body not in selected_bodies or selected_bodies.index(
        high_value.body
    ) < selected_bodies.index(low_value.body)


def test_update_after_turn_skips_model_selector_when_candidates_fit_budget(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    memory = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra promised Mara the lens key.",
        tags=["promise"],
        importance=0.95,
        source_message_id=narrator_message.id,
    )
    extractor = RecordingExtractorWithContextSelector(
        module.ContextUpdateExtraction()
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert extractor.selection_requests == []
    request = cast(Any, extractor.requests[0])
    assert [item.body for item in request.prior_context] == [memory.body]


def test_update_after_turn_records_context_update_substeps(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extractor = RecordingContextUpdateExtractor(module.ContextUpdateExtraction())
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    context_job = [
        job
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.type == "context_update"
    ][0]
    steps = repositories.list_job_steps(context_job.id)
    assert [step.name for step in steps] == [
        "snapshot",
        "prior_context_selection",
        "extraction",
        "apply",
        "focused_scene",
        "world_data_enrichment",
    ]
    assert {step.status for step in steps} == {"succeeded"}
    assert all(step.duration_ms is not None for step in steps)


def test_update_after_turn_creates_context_records_links_and_audit(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extraction = module.ContextUpdateExtraction(
        locations=(
            module.ExtractedLocation(
                name="  Beacon Gallery  ",
                source_message_id=narrator_message.id,
                aliases=("upper lens room",),
                description="A hot room above the keep wall.",
                visual_description="Red glass and ash-streaked windows.",
                connections=("Gatehouse",),
                status="sealed but unstable",
                hazards=("cracked lens",),
                reason="The narrator placed the action in the beacon gallery.",
                confidence=0.91,
            ),
        ),
        characters=(
            module.ExtractedCharacter(
                name="Captain Ilyra",
                source_message_id=narrator_message.id,
                aliases=("captain",),
                role="Watch captain",
                known_state="guarding the lens",
                met=True,
                appearance="Ash-gray cloak and bronze signal horn.",
                visual_notes="Tall silhouette in red lamp glow.",
                personality="decisive",
                voice="low and clipped",
                relationships={"player": "ally"},
                status="present",
                location_name="Beacon Gallery",
                private_notes="Knows the beacon is failing.",
                reason="Ilyra speaks directly to Mara in the completed turn.",
                confidence=0.88,
            ),
        ),
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            current_location_name="Beacon Gallery",
            situation="The beacon lens is overheating.",
            objective="Stop the lens from cracking.",
            in_world_time="midnight",
            weather="ash storm",
            mood="urgent",
            nearby_objects=("signal horn", "oil lever"),
            hazards=("red-hot glass",),
            present_character_names=("Captain Ilyra",),
            reason="The turn established the immediate scene.",
            confidence=0.9,
        ),
        active_threads=(
            module.ExtractedActiveThread(
                title="Save the beacon",
                source_message_id=narrator_message.id,
                description="The red lens may shatter before dawn.",
                status="active",
                priority=8,
                visibility="public",
                related_entities=("Beacon Gallery", "Captain Ilyra"),
                reason="The cracked lens is now an active problem.",
                confidence=0.86,
            ),
        ),
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                target_type="location",
                source_message_id=narrator_message.id,
                entity_name="Captain Ilyra",
                target_name="Beacon Gallery",
                relation="present_at",
                reason="Ilyra is in the gallery.",
                confidence=0.93,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "locations") == 1
    assert _count(result, "characters") == 1
    assert _count(result, "scene_snapshot") == 1
    assert _count(result, "active_threads") == 1
    assert _count(result, "entity_links") == 1
    assert _count(result, "audit_rows") >= 5
    assert len(extractor.requests) == 1
    request = cast(Any, extractor.requests[0])
    assert request.save_id == save.id
    assert [message.id for message in request.messages] == [
        player_message.id,
        narrator_message.id,
    ]

    locations = repositories.list_locations(save.id)
    assert len(locations) == 1
    location = locations[0]
    assert location.name == "Beacon Gallery"
    assert location.aliases == ["upper lens room"]
    assert location.source_message_id == narrator_message.id

    characters = repositories.list_characters(save.id)
    assert len(characters) == 1
    character = characters[0]
    assert character.name == "Captain Ilyra"
    assert character.location_id == location.id
    assert character.source_message_id == narrator_message.id

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.current_location_id == location.id
    assert snapshot.present_character_ids == [character.id]
    assert snapshot.source_message_id == narrator_message.id

    threads = repositories.list_active_threads(save.id)
    assert len(threads) == 1
    assert threads[0].related_entities == ["Beacon Gallery", "Captain Ilyra"]

    links = repositories.list_entity_links(save.id)
    assert len(links) == 1
    assert links[0].entity_type == "character"
    assert links[0].entity_id == character.id
    assert links[0].target_type == "location"
    assert links[0].target_id == location.id
    assert links[0].relation == "present_at"
    assert links[0].source_message_id == narrator_message.id

    allowed_source_ids = {player_message.id, narrator_message.id}
    audit_rows = repositories.list_context_update_audit(save.id)
    assert len(audit_rows) >= 5
    assert {(row.operation, row.entity_type) for row in audit_rows} >= {
        ("created", "location"),
        ("created", "character"),
        ("created", "scene_snapshot"),
        ("created", "active_thread"),
        ("created", "entity_link"),
    }
    for row in audit_rows:
        assert set(row.source_message_ids) <= allowed_source_ids
        assert row.reason
        assert row.confidence > 0


def test_update_after_turn_preserves_plan_owned_mood_but_applies_other_domains(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    present_character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        source_message_id=player_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The old scene.",
        mood="planner-committed tension",
        present_character_ids=[],
        source_message_id=narrator_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        locations=(
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id=narrator_message.id,
                reason="The narrator moved the scene into the gallery.",
            ),
        ),
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            current_location_name="Beacon Gallery",
            situation="Captain Ilyra braces the failing lens.",
            mood="legacy extractor overwrite",
            present_character_names=("Captain Ilyra",),
            reason="The narrator established the new scene.",
        ),
        active_threads=(
            module.ExtractedActiveThread(
                title="Stabilize the beacon",
                source_message_id=narrator_message.id,
                description="The lens may crack.",
                reason="The failing lens remains unresolved.",
            ),
        ),
    )

    asyncio.run(
        module.ContextUpdateService(
            repositories=repositories,
            extractor=RecordingContextUpdateExtractor(extraction),
        ).update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
            verified_coverage=VerifiedPostTurnCoverage(
                scene_snapshot_fields=frozenset({"mood"}),
                applied_domains=frozenset({"scene_snapshot"}),
                committed_count=1,
            ),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.mood == "planner-committed tension"
    assert snapshot.situation == "Captain Ilyra braces the failing lens."
    assert snapshot.present_character_ids == [present_character.id]
    assert repositories.get_location(snapshot.current_location_id or "") is not None
    assert [thread.title for thread in repositories.list_active_threads(save.id)] == [
        "Stabilize the beacon"
    ]


def test_update_after_turn_enriches_scene_created_sparse_location(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            current_location_name="Beacon Gallery",
            situation="Captain Ilyra steadies Mara in the beacon gallery.",
            mood="urgent",
            reason="The scene names the beacon gallery.",
            confidence=0.82,
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)

    def build_enrichment(request: object) -> object:
        enrichment_request = cast(Any, request)
        assert "Ashfall Keep" in enrichment_request.scenario_context
        location = enrichment_request.locations[0]
        return module.WorldDataEnrichment(
            locations=(
                module.LocationWorldDataEnrichment(
                    location_id=location.id,
                    source_message_id=narrator_message.id,
                    evidence_quote="beacon gallery",
                    description=(
                        "A red glass chamber around the beacon and its lens."
                    ),
                    visual_description=(
                        "Red glass, signal brass, and a high lens assembly."
                    ),
                    status="occupied by Mara and Ilyra",
                    hazards=("unstable beacon lens",),
                    reason="The sparse current location needs stable details.",
                    confidence=0.74,
                ),
            ),
        )

    enricher = RecordingWorldDataEnricher(build_enrichment)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
        world_data_enricher=enricher,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert len(enricher.requests) == 1
    location = repositories.list_locations(save.id)[0]
    assert location.name == "Beacon Gallery"
    assert location.description == (
        "A red glass chamber around the beacon and its lens."
    )
    assert location.visual_description == (
        "Red glass, signal brass, and a high lens assembly."
    )
    assert location.status == "occupied by Mara and Ilyra"
    assert location.hazards == ["unstable beacon lens"]
    assert location.source_message_id == narrator_message.id
    assert _count(result, "locations") == 1
    assert {
        (row.operation, row.entity_type, row.field_path)
        for row in repositories.list_context_update_audit(save.id)
    } >= {
        ("created", "location", "*"),
        ("updated", "location", "description"),
        ("updated", "location", "visual_description"),
    }


def test_world_data_enrichment_only_fills_blank_unlocked_fields(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        description="Existing user-approved gallery description.",
        visual_description="",
        status="",
        locked_fields=["visual_description"],
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            current_location_name="Beacon Gallery",
            reason="The scene stayed in the beacon gallery.",
            confidence=0.8,
        ),
    )
    enricher = RecordingWorldDataEnricher(
        module.WorldDataEnrichment(
            locations=(
                module.LocationWorldDataEnrichment(
                    location_id=location.id,
                    source_message_id=narrator_message.id,
                    evidence_quote="beacon gallery",
                    description="Replacement description should not apply.",
                    visual_description="Locked visual should not apply.",
                    status="occupied by Mara and Ilyra",
                    reason="Only blank unlocked fields may be enriched.",
                    confidence=0.7,
                ),
            ),
        )
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
        world_data_enricher=enricher,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    saved = repositories.get_location(location.id)
    assert saved is not None
    assert saved.description == "Existing user-approved gallery description."
    assert saved.visual_description == ""
    assert saved.status == "occupied by Mara and Ilyra"


def test_world_data_enrichment_drops_ungrounded_items_before_apply(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        description="",
        status="",
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    result = service.apply_world_data_enrichment(
        save_id=save.id,
        enrichment=module.WorldDataEnrichment(
            locations=(
                module.LocationWorldDataEnrichment(
                    location_id=location.id,
                    source_message_id=narrator_message.id,
                    evidence_quote="ruby library",
                    description="Unsupported library detail.",
                ),
                module.LocationWorldDataEnrichment(
                    location_id=location.id,
                    source_message_id=narrator_message.id,
                    evidence_quote="beacon gallery",
                    status="occupied by Mara and Ilyra",
                ),
            )
        ),
        allowed_source_message_ids=(player_message.id, narrator_message.id),
        completed_messages=(player_message, narrator_message),
    )

    saved = repositories.get_location(location.id)
    assert saved is not None
    assert saved.description == ""
    assert saved.status == "occupied by Mara and Ilyra"
    assert _count(result, "locations") == 1


def test_world_data_enrichment_fills_only_new_sparse_character_profiles(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    old_sparse = repositories.add_character(
        save_id=save.id,
        name="Old Sparse Contact",
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Captain Ilyra",
                source_message_id=narrator_message.id,
                reason="Ilyra appears in the completed turn.",
                confidence=0.84,
            ),
        ),
    )

    def build_enrichment(request: object) -> object:
        enrichment_request = cast(Any, request)
        assert [character.name for character in enrichment_request.characters] == [
            "Captain Ilyra"
        ]
        character = enrichment_request.characters[0]
        return module.WorldDataEnrichment(
            characters=(
                module.CharacterWorldDataEnrichment(
                    character_id=character.id,
                    source_message_id=narrator_message.id,
                    evidence_quote="Captain Ilyra",
                    aliases=("Ilyra",),
                    role="Watch captain",
                    known_state="She steadied Mara in the beacon gallery.",
                    appearance="Bronze cloak clasp and salt-stained boots.",
                    visual_notes="Upright silhouette beside the beacon lens.",
                    personality="Decisive and guarded.",
                    voice="Low clipped orders.",
                    relationships={"Mara": "wary ally"},
                    status="present in the gallery",
                    reason="The new character row is sparse.",
                    confidence=0.79,
                ),
            ),
        )

    enricher = RecordingWorldDataEnricher(build_enrichment)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
        world_data_enricher=enricher,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert len(enricher.requests) == 1
    old_saved = repositories.get_character(old_sparse.id)
    assert old_saved is not None
    assert old_saved.role == ""
    new_saved = next(
        character
        for character in repositories.list_characters(save.id)
        if character.name == "Captain Ilyra"
    )
    assert new_saved.aliases == ["Ilyra"]
    assert new_saved.role == "Watch captain"
    assert new_saved.known_state == "She steadied Mara in the beacon gallery."
    assert new_saved.appearance == "Bronze cloak clasp and salt-stained boots."
    assert new_saved.visual_notes == "Upright silhouette beside the beacon lens."
    assert new_saved.personality == "Decisive and guarded."
    assert new_saved.voice == "Low clipped orders."
    assert new_saved.relationships == {"Mara": "wary ally"}
    assert new_saved.status == "present in the gallery"
    assert _count(result, "characters") == 1


def test_update_after_turn_queues_new_character_when_confirmation_enabled(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.set_app_setting(
        "manual_confirmation_character_registry_enabled",
        True,
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Captain Ilyra",
                source_message_id=narrator_message.id,
                aliases=("captain",),
                role="Watch captain",
                known_state="guarding the lens",
                met=True,
                appearance="Ash-gray cloak and bronze signal horn.",
                visual_notes="Tall silhouette in red lamp glow.",
                personality="decisive",
                voice="low and clipped",
                relationships={"player": "ally"},
                goals="Keep the red lens under control.",
                current_intent="Demand proof before sharing the failsafe.",
                cooperation_conditions="Helps after seeing proof.",
                status="present",
                private_notes="Knows the beacon is failing.",
                reason="Ilyra speaks directly to Mara in the completed turn.",
                confidence=0.88,
            ),
        ),
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            present_character_names=("Captain Ilyra",),
            reason="The scene names Ilyra as present.",
            confidence=0.89,
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "characters") == 0
    assert _count(result, "queued_suggestions") == 1
    assert _count(result, "scene_snapshot") == 1
    assert repositories.list_characters(save.id) == []

    suggestions = repositories.list_context_update_suggestions(save.id)
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.update_type == "create"
    assert suggestion.entity_type == "character"
    assert suggestion.entity_id is None
    assert suggestion.field_path == "*"
    assert suggestion.proposed_value == {
        "name": "Captain Ilyra",
        "aliases": ["captain"],
        "role": "Watch captain",
        "known_state": "guarding the lens",
        "met": True,
        "appearance": "Ash-gray cloak and bronze signal horn.",
        "visual_notes": "Tall silhouette in red lamp glow.",
        "current_clothing": "",
        "personality": "decisive",
        "voice": "low and clipped",
        "relationships": {"player": "ally"},
        "goals": "Keep the red lens under control.",
        "current_intent": "Demand proof before sharing the failsafe.",
        "cooperation_conditions": "Helps after seeing proof.",
        "status": "present",
        "location_id": None,
        "private_notes": "Knows the beacon is failing.",
        "source_message_id": narrator_message.id,
    }
    audit_rows = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type) for row in audit_rows].count(
        ("queued", "character")
    ) == 1
    assert ("created", "scene_snapshot") in {
        (row.operation, row.entity_type) for row in audit_rows
    }


def test_update_after_turn_skips_opaque_extracted_character_name(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    opaque_name = "0415a4a810a7422b8d59e95369fedc6c"
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name=opaque_name,
                source_message_id=narrator_message.id,
                role="",
                known_state="",
                met=True,
                reason="The extractor returned an internal id instead of a name.",
                confidence=0.8,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "characters") == 0
    assert repositories.list_characters(save.id) == []
    assert repositories.list_context_update_suggestions(save.id) == []
    assert repositories.list_context_update_audit(save.id) == []
    assert repositories.list_failed_jobs() == []


def test_update_after_turn_drops_blank_extracted_entities_without_failing_job(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extraction = module.ContextUpdateExtraction(
        locations=(
            module.ExtractedLocation(
                name=" ",
                source_message_id=narrator_message.id,
                description="This malformed location should be ignored.",
                reason="The extractor returned an empty location name.",
                confidence=0.6,
            ),
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id=narrator_message.id,
                description="A hot room above the keep wall.",
                reason="The narrator placed the action in the beacon gallery.",
                confidence=0.91,
            ),
        ),
        characters=(
            module.ExtractedCharacter(
                name="",
                source_message_id=narrator_message.id,
                role="Malformed row",
                reason="The extractor returned an empty character name.",
                confidence=0.6,
            ),
            module.ExtractedCharacter(
                name="Captain Ilyra",
                source_message_id=narrator_message.id,
                role="Watch captain",
                reason="Ilyra speaks directly to Mara.",
                confidence=0.88,
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "locations") == 1
    assert _count(result, "characters") == 1
    assert [location.name for location in repositories.list_locations(save.id)] == [
        "Beacon Gallery"
    ]
    assert [character.name for character in repositories.list_characters(save.id)] == [
        "Captain Ilyra"
    ]
    assert repositories.list_failed_jobs() == []
    assert {
        (row.operation, row.entity_type)
        for row in repositories.list_context_update_audit(save.id)
    } == {
        ("created", "location"),
        ("created", "character"),
    }


def test_update_after_turn_skips_opaque_present_character_name(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    opaque_name = "0415a4a810a7422b8d59e95369fedc6c"
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            situation="Mara reaches the beacon lens alone.",
            present_character_names=(opaque_name,),
            reason="The extractor returned an internal id instead of a name.",
            confidence=0.84,
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "scene_snapshot") == 1
    assert _count(result, "characters") == 0
    assert repositories.list_characters(save.id) == []

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == []

    audit_rows = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type) for row in audit_rows] == [
        ("created", "scene_snapshot")
    ]
    assert repositories.list_context_update_suggestions(save.id) == []
    assert repositories.list_failed_jobs() == []


def test_update_after_turn_updates_existing_character_from_unique_short_name(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    existing_character = repositories.add_character(
        save_id=save.id,
        name="Camille Beldem",
        aliases=[],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Camille",
                source_message_id=narrator_message.id,
                role="Signal engineer",
                known_state="guiding Mara through the lower ward",
                status="present",
                reason="The narrator refers to Camille by first name only.",
                confidence=0.9,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "characters") == 1
    characters = repositories.list_characters(save.id)
    assert len(characters) == 1
    updated_character = characters[0]
    assert updated_character.id == existing_character.id
    assert updated_character.name == "Camille Beldem"
    assert updated_character.aliases == ["Camille"]
    assert updated_character.role == "Signal engineer"
    assert updated_character.known_state == "guiding Mara through the lower ward"
    assert updated_character.status == "present"
    assert updated_character.source_message_id == narrator_message.id
    assert updated_character.first_seen_message_id == player_message.id
    assert updated_character.last_updated_message_id == narrator_message.id
    assert repositories.list_context_update_suggestions(save.id) == []

    audit_rows = repositories.list_context_update_audit(save.id)
    assert ("created", "character") not in {
        (row.operation, row.entity_type) for row in audit_rows
    }
    assert {
        row.field_path for row in audit_rows if row.entity_type == "character"
    } == {"aliases", "role", "known_state", "status"}


def test_update_after_turn_resolves_existing_alias_in_character_and_scene_present(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    existing_character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["captain"],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="captain",
                source_message_id=narrator_message.id,
                reason="The narrator refers to Ilyra by title alone.",
                confidence=0.9,
            ),
        ),
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            situation="The captain steadies Mara by the lens.",
            present_character_names=("captain",),
            reason="The scene keeps the captain present.",
            confidence=0.88,
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "characters") == 1
    assert _count(result, "scene_snapshot") == 1
    characters = repositories.list_characters(save.id)
    assert len(characters) == 1
    character = characters[0]
    assert character.id == existing_character.id
    assert character.name == "Captain Ilyra"
    assert character.aliases == ["captain"]

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == [existing_character.id]

    audit_rows = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type) for row in audit_rows] == [
        ("created", "scene_snapshot")
    ]
    assert repositories.list_context_update_suggestions(save.id) == []


def test_update_after_turn_replaces_scene_present_character_ids(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    current_character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["captain"],
        source_message_id=player_message.id,
    )
    stale_character = repositories.add_character(
        save_id=save.id,
        name="Archivist Elian",
        aliases=["elian"],
        source_message_id=player_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Ilyra and Elian stand by the beacon.",
        present_character_ids=[current_character.id, stale_character.id],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            situation="Ilyra steadies Mara by the lens.",
            present_character_names=("Captain Ilyra",),
            reason="Only Ilyra remains present in the current scene.",
            confidence=0.88,
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "scene_snapshot") == 1
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == [current_character.id]
    assert repositories.list_context_update_suggestions(save.id) == []


def test_update_after_turn_clears_scene_present_character_ids(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    stale_character = repositories.add_character(
        save_id=save.id,
        name="Archivist Elian",
        aliases=["elian"],
        source_message_id=player_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Elian stands by the beacon.",
        present_character_ids=[stale_character.id],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            situation="Mara is alone by the lens.",
            present_character_names=(),
            reason="The current scene has no named present characters.",
            confidence=0.88,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "scene_snapshot") == 1
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == []
    audit_rows = repositories.list_context_update_audit(save.id)
    assert any(
        row.entity_type == "scene_snapshot"
        and row.field_path == "present_character_ids"
        and row.before == [stale_character.id]
        and row.after == []
        for row in audit_rows
    )


def test_update_after_turn_keeps_player_character_present_when_presence_clears(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    player_character = repositories.add_character(
        save_id=save.id,
        name="Mara Voss",
        source_message_id=player_message.id,
        is_player_character=True,
    )
    stale_character = repositories.add_character(
        save_id=save.id,
        name="Archivist Elian",
        source_message_id=player_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara and Elian stand by the beacon.",
        present_character_ids=[player_character.id, stale_character.id],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            situation="The beacon lens hums with pressure.",
            present_character_names=(),
            reason="The model cleared current named scene presence.",
            confidence=0.88,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == [player_character.id]


def test_update_after_turn_preserves_scene_present_character_ids_when_presence_omitted(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    present_character = repositories.add_character(
        save_id=save.id,
        name="Archivist Elian",
        aliases=["elian"],
        source_message_id=player_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Elian stands by the beacon.",
        present_character_ids=[present_character.id],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            situation="The beacon lens hums with pressure.",
            reason="The current scene situation changed.",
            confidence=0.88,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "scene_snapshot") == 1
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == [present_character.id]
    assert not any(
        row.entity_type == "scene_snapshot"
        and row.field_path == "present_character_ids"
        for row in repositories.list_context_update_audit(save.id)
    )


def test_update_after_turn_clears_scene_local_context_after_location_change(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    old_location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=player_message.id,
    )
    stale_character = repositories.add_character(
        save_id=save.id,
        name="Archivist Elian",
        source_message_id=player_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=old_location.id,
        situation="Elian stands by the red-hot glass.",
        nearby_objects=["signal horn"],
        hazards=["red-hot glass"],
        present_character_ids=[stale_character.id],
        source_message_id=player_message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="character.archivist_elian.current_emotional_state",
        value={"mood": "focused on the failing beacon lens"},
        category="relationship",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            current_location_name="Gatehouse",
            situation="Mara reaches the gatehouse alone.",
            reason="The completed turn moved to a different location.",
            confidence=0.91,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "updated_fields") == 5
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.current_location_id != old_location.id
    assert snapshot.nearby_objects == []
    assert snapshot.hazards == []
    assert snapshot.present_character_ids == []
    audit_by_field = {
        row.field_path: row for row in repositories.list_context_update_audit(save.id)
    }
    assert audit_by_field["nearby_objects"].after == []
    assert audit_by_field["hazards"].after == []
    assert audit_by_field["present_character_ids"].after == []

    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    current_scene_text = "\n".join(assembled.current_scene_context)
    assert "signal horn" not in current_scene_text
    assert "red-hot glass" not in current_scene_text
    assert "focused on the failing beacon lens" not in current_scene_text


def test_update_after_turn_skips_ambiguous_short_character_name(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    beldem = repositories.add_character(
        save_id=save.id,
        name="Camille Beldem",
        aliases=[],
        status="at the gatehouse",
        source_message_id=player_message.id,
    )
    ardent = repositories.add_character(
        save_id=save.id,
        name="Camille Ardent",
        aliases=[],
        status="in the infirmary",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Camille",
                source_message_id=narrator_message.id,
                status="present",
                reason="The narrator uses a short name shared by two characters.",
                confidence=0.87,
            ),
        ),
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            situation="Someone named Camille calls from the stairwell.",
            present_character_names=("Camille",),
            reason="The scene mentions Camille without enough identity context.",
            confidence=0.86,
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "characters") == 0
    characters = repositories.list_characters(save.id)
    assert {character.id for character in characters} == {beldem.id, ardent.id}
    characters_by_id = {character.id: character for character in characters}
    assert characters_by_id[beldem.id].name == "Camille Beldem"
    assert characters_by_id[beldem.id].aliases == []
    assert characters_by_id[beldem.id].status == "at the gatehouse"
    assert characters_by_id[ardent.id].name == "Camille Ardent"
    assert characters_by_id[ardent.id].aliases == []
    assert characters_by_id[ardent.id].status == "in the infirmary"

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == []
    assert repositories.list_context_update_suggestions(save.id) == []

    audit_rows = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type) for row in audit_rows] == [
        ("created", "scene_snapshot")
    ]


def test_update_after_turn_skips_short_character_alias_when_full_name_is_ambiguous(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    beldem = repositories.add_character(
        save_id=save.id,
        name="Camille Beldem",
        aliases=["Camille"],
        role="Signal engineer",
        known_state="calibrating the lower ward relays",
        status="at the gatehouse",
        source_message_id=player_message.id,
    )
    ardent = repositories.add_character(
        save_id=save.id,
        name="Camille Ardent",
        aliases=[],
        role="Field medic",
        known_state="treating the injured watch",
        status="in the infirmary",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Camille",
                source_message_id=narrator_message.id,
                role="Runner",
                known_state="calling from the stairwell",
                status="present",
                reason="The narrator uses a short name shared by two characters.",
                confidence=0.87,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "characters") == 0
    characters = repositories.list_characters(save.id)
    assert {character.id for character in characters} == {beldem.id, ardent.id}
    characters_by_id = {character.id: character for character in characters}
    assert characters_by_id[beldem.id].name == "Camille Beldem"
    assert characters_by_id[beldem.id].aliases == ["Camille"]
    assert characters_by_id[beldem.id].role == "Signal engineer"
    assert (
        characters_by_id[beldem.id].known_state
        == "calibrating the lower ward relays"
    )
    assert characters_by_id[beldem.id].status == "at the gatehouse"
    assert characters_by_id[beldem.id].source_message_id == player_message.id
    assert characters_by_id[ardent.id].name == "Camille Ardent"
    assert characters_by_id[ardent.id].aliases == []
    assert characters_by_id[ardent.id].role == "Field medic"
    assert characters_by_id[ardent.id].known_state == "treating the injured watch"
    assert characters_by_id[ardent.id].status == "in the infirmary"
    assert characters_by_id[ardent.id].source_message_id == player_message.id
    assert repositories.list_context_update_suggestions(save.id) == []
    assert repositories.list_context_update_audit(save.id) == []


def test_update_after_turn_skips_ambiguous_short_alias_in_present_characters(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    beldem = repositories.add_character(
        save_id=save.id,
        name="Camille Beldem",
        aliases=["Camille"],
        status="at the gatehouse",
        source_message_id=player_message.id,
    )
    ardent = repositories.add_character(
        save_id=save.id,
        name="Camille Ardent",
        aliases=[],
        status="in the infirmary",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Camille",
                source_message_id=narrator_message.id,
                status="present",
                reason="The narrator uses a short name shared by two characters.",
                confidence=0.87,
            ),
        ),
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            situation="Camille calls from the stairwell.",
            present_character_names=("Camille",),
            reason="The scene mentions Camille without enough identity context.",
            confidence=0.86,
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "characters") == 0
    characters = repositories.list_characters(save.id)
    assert {character.id for character in characters} == {beldem.id, ardent.id}
    characters_by_id = {character.id: character for character in characters}
    assert characters_by_id[beldem.id].name == "Camille Beldem"
    assert characters_by_id[beldem.id].aliases == ["Camille"]
    assert characters_by_id[beldem.id].status == "at the gatehouse"
    assert characters_by_id[beldem.id].source_message_id == player_message.id
    assert characters_by_id[ardent.id].name == "Camille Ardent"
    assert characters_by_id[ardent.id].aliases == []
    assert characters_by_id[ardent.id].status == "in the infirmary"
    assert characters_by_id[ardent.id].source_message_id == player_message.id

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == []
    assert repositories.list_context_update_suggestions(save.id) == []

    audit_rows = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type) for row in audit_rows] == [
        ("created", "scene_snapshot")
    ]


def test_update_after_turn_does_not_merge_full_character_name_into_short_alias(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    beldem = repositories.add_character(
        save_id=save.id,
        name="Camille Beldem",
        aliases=["Camille"],
        role="Signal engineer",
        known_state="calibrating the lower ward relays",
        status="at the gatehouse",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Camille Ardent",
                source_message_id=narrator_message.id,
                role="Field medic",
                known_state="treating the injured watch",
                status="present",
                reason="The narrator identifies the character by full name.",
                confidence=0.88,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    characters = repositories.list_characters(save.id)
    beldem_after = next(
        character for character in characters if character.id == beldem.id
    )
    assert beldem_after.name == "Camille Beldem"
    assert beldem_after.aliases == ["Camille"]
    assert beldem_after.role == "Signal engineer"
    assert beldem_after.known_state == "calibrating the lower ward relays"
    assert beldem_after.status == "at the gatehouse"
    assert beldem_after.source_message_id == player_message.id

    suggestions = repositories.list_context_update_suggestions(save.id)
    assert [
        suggestion for suggestion in suggestions if suggestion.entity_id == beldem.id
    ] == []

    audit_rows = repositories.list_context_update_audit(save.id)
    assert [row for row in audit_rows if row.entity_id == beldem.id] == []


def test_update_after_turn_keeps_short_alias_owner_unchanged_for_new_full_name(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    beldem = repositories.add_character(
        save_id=save.id,
        name="Camille Beldem",
        aliases=["Camille"],
        role="Signal engineer",
        known_state="calibrating the lower ward relays",
        status="at the gatehouse",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Camille Ardent",
                source_message_id=narrator_message.id,
                role="Field medic",
                known_state="treating the injured watch",
                status="present",
                reason="The narrator identifies the character by full name.",
                confidence=0.88,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    characters = repositories.list_characters(save.id)
    beldem_after = next(
        character for character in characters if character.id == beldem.id
    )
    assert beldem_after.name == "Camille Beldem"
    assert beldem_after.aliases == ["Camille"]
    assert beldem_after.role == "Signal engineer"
    assert beldem_after.known_state == "calibrating the lower ward relays"
    assert beldem_after.status == "at the gatehouse"
    assert beldem_after.source_message_id == player_message.id
    assert any(character.name == "Camille Ardent" for character in characters)

    beldem_suggestions = [
        suggestion
        for suggestion in repositories.list_context_update_suggestions(save.id)
        if suggestion.entity_id == beldem.id
    ]
    assert beldem_suggestions == []

    beldem_audit_rows = [
        row
        for row in repositories.list_context_update_audit(save.id)
        if row.entity_id == beldem.id and row.field_path is not None
    ]
    assert beldem_audit_rows == []


def test_update_after_turn_dedupes_existing_entity_link_without_created_audit(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=player_message.id,
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        location_id=location.id,
        source_message_id=player_message.id,
    )
    existing_link = repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="location",
        target_id=location.id,
        relation="present_at",
    )
    extraction = module.ContextUpdateExtraction(
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                target_type="location",
                source_message_id=narrator_message.id,
                entity_name="Captain Ilyra",
                target_name="Beacon Gallery",
                relation="present_at",
                reason="Ilyra is already in the gallery.",
                confidence=0.93,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    links = repositories.list_entity_links(save.id)
    assert len(links) == 1
    assert links[0].id == existing_link.id

    assert repositories.list_context_update_audit(save.id) == []


def test_update_after_turn_resolves_unique_short_character_name_for_entity_link(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=player_message.id,
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Camille Beldem",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                target_type="location",
                source_message_id=narrator_message.id,
                entity_name="Camille",
                target_name="Beacon Gallery",
                relation="present_at",
                reason="Camille is said to be in the gallery.",
                confidence=0.9,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "entity_links") == 1
    assert repositories.list_characters(save.id) == [character]

    links = repositories.list_entity_links(save.id)
    assert len(links) == 1
    assert links[0].entity_type == "character"
    assert links[0].entity_id == character.id
    assert links[0].target_type == "location"
    assert links[0].target_id == location.id
    assert links[0].relation == "present_at"

    audit_rows = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type) for row in audit_rows] == [
        ("created", "entity_link")
    ]
    assert repositories.list_context_update_suggestions(save.id) == []


def test_update_after_turn_links_character_known_memory_state_and_summary(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        source_message_id=player_message.id,
        character_id="character-ilyra-knows",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra knows the lens-key phrase is ember dawn.",
        tags=["ilyra", "lens"],
        source_message_id=player_message.id,
        memory_id="memory-ilyra-knows",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"failsafe": "copper notch"},
        source_message_id=player_message.id,
        state_id="world-state-ilyra-knows",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=player_message.id,
        covers_message_end_id=narrator_message.id,
        body="Ilyra explained how to stop the beacon lens cracking.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-ilyra-knows",
    )
    extraction = module.ContextUpdateExtraction(
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                entity_id=character.id,
                target_type="memory",
                target_id=memory.id,
                source_message_id=narrator_message.id,
                relation="knows",
                reason="Ilyra treats the lens phrase as her knowledge.",
                confidence=0.91,
            ),
            module.ExtractedEntityLink(
                entity_type="character",
                entity_id=character.id,
                target_type="state",
                target_id=state.id,
                source_message_id=narrator_message.id,
                relation="knows",
                reason="Ilyra knows the current lens failsafe.",
                confidence=0.88,
            ),
            module.ExtractedEntityLink(
                entity_type="character",
                entity_id=character.id,
                target_type="summary",
                target_id=summary.id,
                source_message_id=narrator_message.id,
                relation="knows",
                reason="The summary is established as Ilyra's knowledge.",
                confidence=0.86,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "entity_links") == 3
    request = cast(Any, extractor.requests[0])
    assert [record.id for record in request.memories] == [memory.id]
    assert [record.id for record in request.world_state] == [state.id]
    assert [record.id for record in request.summaries] == [summary.id]

    links = repositories.list_entity_links(save.id)
    assert {
        (
            link.entity_type,
            link.entity_id,
            link.target_type,
            link.target_id,
            link.relation,
        )
        for link in links
    } == {
        ("character", character.id, "memory", memory.id, "knows"),
        ("character", character.id, "world_state", state.id, "knows"),
        ("character", character.id, "summary", summary.id, "knows"),
    }
    audit_rows = repositories.list_context_update_audit(save.id)
    assert [row.entity_type for row in audit_rows] == [
        "entity_link",
        "entity_link",
        "entity_link",
    ]
    assert {row.operation for row in audit_rows} == {"created"}


def test_update_after_turn_falls_back_to_names_when_entity_link_id_is_malformed(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        source_message_id=player_message.id,
        character_id="character-ilyra-knows",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"failsafe": "copper notch"},
        source_message_id=player_message.id,
        state_id="world-state-ilyra-knows",
    )
    extraction = module.ContextUpdateExtraction(
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                entity_id="character-ilyra-typo",
                entity_name="Captain Ilyra",
                target_type="world_state",
                target_id="world-state-ilyra-typo",
                target_name="beacon.lens",
                source_message_id=narrator_message.id,
                relation="knows",
                reason="Ilyra knows the lens failsafe.",
                confidence=0.88,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "entity_links") == 1
    links = repositories.list_entity_links(save.id)
    assert len(links) == 1
    assert links[0].entity_type == "character"
    assert links[0].entity_id == character.id
    assert links[0].target_type == "world_state"
    assert links[0].target_id == state.id
    assert links[0].relation == "knows"
    assert repositories.list_failed_jobs() == []


def test_update_after_turn_resolves_world_state_link_by_normalized_name(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        source_message_id=player_message.id,
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"failsafe": "copper notch"},
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                entity_name="Captain Ilyra",
                target_type="world_state",
                target_name="Beacon Lens",
                source_message_id=narrator_message.id,
                relation="knows",
                reason="Ilyra knows the lens failsafe.",
                confidence=0.88,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "entity_links") == 1
    links = repositories.list_entity_links(save.id)
    assert len(links) == 1
    assert links[0].entity_id == character.id
    assert links[0].target_type == "world_state"
    assert links[0].target_id == state.id


def test_update_after_turn_resolves_world_state_link_by_unique_value_name(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        source_message_id=player_message.id,
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.current_location",
        value={"name": "Beacon Gallery", "threat": "cracked lens"},
        source_message_id=player_message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.weather",
        value={"name": "Ash Storm"},
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                entity_name="Captain Ilyra",
                target_type="world_state",
                target_name="Beacon Gallery",
                source_message_id=narrator_message.id,
                relation="observes",
                reason="Ilyra is watching the current location state.",
                confidence=0.84,
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "entity_links") == 1
    links = repositories.list_entity_links(save.id)
    assert len(links) == 1
    assert links[0].entity_id == character.id
    assert links[0].target_id == state.id


def test_update_after_turn_skips_ambiguous_world_state_link_name(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        source_message_id=player_message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"name": "Beacon Lens", "status": "hot"},
        source_message_id=player_message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="inventory.beacon_lens",
        value={"name": "Beacon Lens", "status": "crated"},
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                entity_name="Captain Ilyra",
                target_type="world_state",
                target_name="Beacon Lens",
                source_message_id=narrator_message.id,
                relation="knows",
                reason="The extracted target name is ambiguous.",
                confidence=0.77,
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "entity_links") == 0
    assert repositories.list_entity_links(save.id) == []


def test_update_after_turn_skips_named_entity_link_with_missing_target(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                target_type="location",
                source_message_id=narrator_message.id,
                entity_name="Captain Ilyra",
                target_name="Bell Tower",
                relation="present_at",
                reason="Ilyra is said to be near the bell tower.",
                confidence=0.89,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "entity_links") == 0
    assert _count(result, "queued_suggestions") == 0
    assert _count(result, "audit_rows") == 0
    assert repositories.list_locations(save.id) == []
    assert repositories.list_characters(save.id) == [character]
    assert repositories.list_entity_links(save.id) == []
    assert repositories.list_context_update_suggestions(save.id) == []
    assert repositories.list_context_update_audit(save.id) == []
    assert repositories.list_failed_jobs() == []


def test_update_after_turn_skips_entity_link_id_with_wrong_entity_type(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                entity_id=location.id,
                target_type="location",
                target_id=location.id,
                source_message_id=narrator_message.id,
                relation="present_at",
                reason="The extracted character id is really a location id.",
                confidence=0.93,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "entity_links") == 0
    assert [record.id for record in repositories.list_locations(save.id)] == [
        location.id
    ]
    assert repositories.list_entity_links(save.id) == []
    assert repositories.list_context_update_suggestions(save.id) == []
    assert repositories.list_context_update_audit(save.id) == []

    assert repositories.list_failed_jobs() == []


@pytest.mark.parametrize(
    ("entity_type", "target_type"),
    (
        ("person", "location"),
        ("character", "person"),
    ),
)
def test_update_after_turn_skips_unsupported_named_entity_link_types(
    repositories: PersistenceRepositories,
    entity_type: str,
    target_type: str,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extraction = module.ContextUpdateExtraction(
        entity_links=(
            module.ExtractedEntityLink(
                entity_type=entity_type,
                target_type=target_type,
                source_message_id=narrator_message.id,
                entity_name="Captain Ilyra",
                target_name="Beacon Gallery",
                relation="present_at",
                reason="The link uses an unsupported endpoint type.",
                confidence=0.81,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "entity_links") == 0
    assert repositories.list_entity_links(save.id) == []
    assert repositories.list_context_update_suggestions(save.id) == []
    assert repositories.list_context_update_audit(save.id) == []
    assert repositories.list_failed_jobs() == []


def test_update_after_turn_queues_suggestion_for_unlocked_conflicting_scalar(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        status="sealed",
        description="A quiet upper lens room.",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        locations=(
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id=narrator_message.id,
                aliases=(),
                description="A quiet upper lens room.",
                visual_description="",
                connections=(),
                status="unstable",
                hazards=(),
                reason="The narrator described the gallery shaking.",
                confidence=0.82,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "queued_suggestions") == 1
    unchanged = repositories.get_location(location.id)
    assert unchanged is not None
    assert unchanged.status == "sealed"
    assert unchanged.source_message_id == player_message.id

    suggestions = repositories.list_context_update_suggestions(save.id)
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.status == "pending"
    assert suggestion.update_type == "update"
    assert suggestion.entity_type == "location"
    assert suggestion.entity_id == location.id
    assert suggestion.field_path == "status"
    assert suggestion.proposed_value == "unstable"
    assert suggestion.reason == "The narrator described the gallery shaking."
    assert suggestion.confidence == 0.82
    assert suggestion.source_message_ids == [narrator_message.id]

    audit_rows = repositories.list_context_update_audit(save.id)
    assert len(audit_rows) == 1
    audit = audit_rows[0]
    assert audit.operation == "queued"
    assert audit.suggestion_id == suggestion.id
    assert audit.entity_type == "location"
    assert audit.entity_id == location.id
    assert audit.field_path == "status"
    assert audit.before == "sealed"
    assert audit.after == "unstable"
    assert audit.reason == "The narrator described the gallery shaking."
    assert audit.confidence == 0.82
    assert audit.source_message_ids == [narrator_message.id]


def test_update_after_turn_suppresses_duplicate_pending_suggestion(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        status="sealed",
        description="A quiet upper lens room.",
        source_message_id=player_message.id,
    )
    existing = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="location",
        entity_id=location.id,
        field_path="status",
        proposed_value="unstable",
        reason="Previous extraction saw the same shake.",
        confidence=0.7,
        source_message_ids=[narrator_message.id],
    )
    extraction = module.ContextUpdateExtraction(
        locations=(
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id=narrator_message.id,
                aliases=(),
                description="A quiet upper lens room.",
                visual_description="",
                connections=(),
                status="unstable",
                hazards=(),
                reason="The narrator described the gallery shaking.",
                confidence=0.82,
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "queued_suggestions") == 0
    suggestions = repositories.list_context_update_suggestions(save.id)
    assert [suggestion.id for suggestion in suggestions] == [existing.id]
    assert suggestions[0].status == "pending"


def test_update_after_turn_applies_scene_snapshot_volatile_scalar(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    existing_snapshot = repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="quiet watch",
        objective="Keep the tower calm.",
        in_world_time="late night",
        weather="clear",
        mood="tense",
        nearby_objects=["signal horn"],
        hazards=["thin fog"],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            situation="alarms ringing",
            reason="The narrator escalated the immediate scene.",
            confidence=0.83,
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "updated_fields") == 1
    assert _count(result, "queued_suggestions") == 0
    updated = repositories.get_scene_snapshot(save.id)
    assert updated is not None
    assert updated.id == existing_snapshot.id
    assert updated.situation == "alarms ringing"
    assert updated.objective == "Keep the tower calm."
    assert updated.in_world_time == "late night"
    assert updated.weather == "clear"
    assert updated.mood == "tense"
    assert updated.nearby_objects == ["signal horn"]
    assert updated.hazards == ["thin fog"]
    assert updated.source_message_id == narrator_message.id
    assert repositories.list_context_update_suggestions(save.id) == []

    audit_rows = repositories.list_context_update_audit(save.id)
    assert len(audit_rows) == 1
    audit = audit_rows[0]
    assert audit.operation == "updated"
    assert audit.suggestion_id is None
    assert audit.entity_type == "scene_snapshot"
    assert audit.entity_id == existing_snapshot.id
    assert audit.field_path == "situation"
    assert audit.before == "quiet watch"
    assert audit.after == "alarms ringing"
    assert audit.reason == "The narrator escalated the immediate scene."
    assert audit.confidence == 0.83
    assert audit.source_message_ids == [narrator_message.id]


def test_update_after_turn_replaces_scene_current_lists(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The gallery is jammed with emergency gear.",
        nearby_objects=["signal horn", "oil lever"],
        hazards=["red-hot glass", "ash leak"],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            nearby_objects=("cooling crank",),
            hazards=("cracked lens",),
            reason="The current scene objects and hazards changed.",
            confidence=0.86,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "updated_fields") == 2
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.nearby_objects == ["cooling crank"]
    assert snapshot.hazards == ["cracked lens"]
    audit_by_field = {
        row.field_path: row for row in repositories.list_context_update_audit(save.id)
    }
    assert audit_by_field["nearby_objects"].before == ["signal horn", "oil lever"]
    assert audit_by_field["nearby_objects"].after == ["cooling crank"]
    assert audit_by_field["hazards"].before == ["red-hot glass", "ash leak"]
    assert audit_by_field["hazards"].after == ["cracked lens"]


def test_update_after_turn_clears_scene_current_lists_when_empty(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The gallery is dangerous.",
        nearby_objects=["signal horn"],
        hazards=["red-hot glass"],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            nearby_objects=(),
            hazards=(),
            reason="The completed turn clears the immediate scene.",
            confidence=0.9,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "updated_fields") == 2
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.nearby_objects == []
    assert snapshot.hazards == []
    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    current_scene_text = "\n".join(assembled.current_scene_context)
    assert "signal horn" not in current_scene_text
    assert "red-hot glass" not in current_scene_text


def test_context_update_prompt_guides_conservative_scene_time_extraction() -> None:
    module = _context_update_module()
    request = module.ContextUpdateRequest(
        save_id="save-1",
        messages=(),
        scene_snapshot=None,
        locations=(),
        characters=(),
        active_threads=(),
        entity_links=(),
    )

    prompt_text = "\n".join(
        message.body for message in module._context_update_messages(request)
    )
    schema = module._context_update_schema(())
    scene_properties = schema["properties"]["scene"]["properties"]

    assert "scene.in_world_time" in prompt_text
    assert "scene.nearby_objects" in prompt_text
    assert "scene.hazards" in prompt_text
    assert "morning, late morning, afternoon, evening, or night" in prompt_text
    assert "Do not emit vague time values like later, soon, eventually" in prompt_text
    assert "explicitly advances, waits, skips ahead" in prompt_text
    assert "complete current lists" in prompt_text
    assert "none remain" in prompt_text
    assert "description" in scene_properties["in_world_time"]
    assert "qualitative current time anchor" in scene_properties["in_world_time"][
        "description"
    ]
    assert "null" in scene_properties["nearby_objects"]["type"]
    assert "Complete current scene nearby objects" in scene_properties[
        "nearby_objects"
    ]["description"]
    assert "null" in scene_properties["hazards"]["type"]
    assert "Complete current scene hazards" in scene_properties["hazards"][
        "description"
    ]
    assert "null" in scene_properties["present_character_names"]["type"]
    assert "unchanged or unclear" in scene_properties["present_character_names"][
        "description"
    ]


def test_focused_scene_maintainer_updates_scene_time_and_present_character_emotion(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        description="A red glass signal chamber.",
    )
    ilyra = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        role="watch captain",
        location_id=location.id,
    )
    repositories.add_character(save_id=save.id, name="Archivist Ren")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        in_world_time="morning",
        present_character_ids=[ilyra.id],
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body=(
            "I wait until afternoon, then ask Ilyra whether the beacon can still "
            "fire."
        ),
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=(
            "By afternoon, Captain Ilyra exhales, guarded but relieved. "
            "Archivist Ren stays below."
        ),
        provider="fake",
        model="fake-chat",
        token_estimate=12,
    )
    extractor = RecordingContextUpdateExtractor(module.ContextUpdateExtraction())
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="time-call",
                    name="set_scene_time",
                    arguments_json=json.dumps(
                        {
                            "in_world_time": "afternoon",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "By afternoon",
                            "reason": "The narrator established the current time.",
                            "confidence": 0.91,
                        }
                    ),
                ),
            ),
            (),
            (),
            (),
            (
                ProviderToolCall(
                    id="emotion-call",
                    name="set_character_emotion",
                    arguments_json=json.dumps(
                        {
                            "character_id": ilyra.id,
                            "emotional_state": "guarded but relieved",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "guarded but relieved",
                            "reason": "Ilyra's current emotional stance changed.",
                            "confidence": 0.88,
                        }
                    ),
                ),
            ),
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "afternoon"
    state_by_key = {
        state.key: state for state in repositories.list_world_state(save.id)
    }
    assert state_by_key[
        "character.captain_ilyra.current_emotional_state"
    ].value == {"mood": "guarded but relieved"}
    assert "character.archivist_ren.current_emotional_state" not in state_by_key
    assert [
        tuple(tool.name for tool in request.tools)
        for request in provider.tool_call_requests
    ] == [
        ("set_scene_time",),
        ("set_scene_location_presence",),
        ("set_scene_surface",),
        ("set_character_relationship_posture",),
        ("set_character_emotion",),
    ]
    assert any(
        row.entity_type == "world_state"
        and row.field_path == "character.captain_ilyra.current_emotional_state"
        and row.operation in {"created", "updated"}
        for row in repositories.list_context_update_audit(save.id)
    )


def test_focused_scene_maintainer_runs_character_tools_concurrently(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    ilyra = repositories.add_character(save_id=save.id, name="Captain Ilyra")
    ren = repositories.add_character(save_id=save.id, name="Archivist Ren")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        present_character_ids=[ilyra.id, ren.id],
    )
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None

    class BlockingRelationshipProvider:
        provider_name = "fake"

        def __init__(self) -> None:
            self.tool_call_requests: list[ToolCallRequest] = []
            self.active_relationships = 0
            self.max_active_relationships = 0
            self.second_relationship_started = asyncio.Event()

        async def generate_tool_calls(
            self,
            request: ToolCallRequest,
        ) -> ToolCallResponse:
            self.tool_call_requests.append(request)
            tool_name = request.tools[0].name
            if tool_name == "set_character_relationship_posture":
                self.active_relationships += 1
                self.max_active_relationships = max(
                    self.max_active_relationships,
                    self.active_relationships,
                )
                try:
                    if self.active_relationships == 1:
                        await asyncio.wait_for(
                            self.second_relationship_started.wait(),
                            timeout=1,
                        )
                    else:
                        self.second_relationship_started.set()
                finally:
                    self.active_relationships -= 1
            return ToolCallResponse(
                tool_calls=(),
                body="",
                provider=request.provider,
                model_id=request.model_id,
            )

    provider = BlockingRelationshipProvider()
    maintainer = module.ToolCallingFocusedSceneMaintainer(
        provider=provider,
        provider_name="fake",
        model_id="fake-context-update",
    )

    async def maintain() -> object:
        return await maintainer.maintain(
            module.FocusedSceneMaintenanceRequest(
                save_id=save.id,
                messages=(player_message, narrator_message),
                scene_snapshot=snapshot,
                locations=tuple(repositories.list_locations(save.id)),
                characters=tuple(repositories.list_characters(save.id)),
            )
        )

    asyncio.run(asyncio.wait_for(maintain(), timeout=2))

    assert provider.max_active_relationships == 2
    assert [
        request.tools[0].name for request in provider.tool_call_requests
    ] == [
        "set_scene_time",
        "set_scene_location_presence",
        "set_scene_surface",
        "set_character_relationship_posture",
        "set_character_relationship_posture",
        "set_character_emotion",
        "set_character_emotion",
    ]


def test_focused_scene_maintainer_runs_scene_tools_concurrently(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        present_character_ids=[],
    )
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None

    class BlockingSceneToolProvider:
        provider_name = "fake"

        def __init__(self) -> None:
            self.tool_call_requests: list[ToolCallRequest] = []
            self.started_tools: set[str] = set()
            self.all_scene_tools_started = asyncio.Event()

        async def generate_tool_calls(
            self,
            request: ToolCallRequest,
        ) -> ToolCallResponse:
            self.tool_call_requests.append(request)
            tool_name = request.tools[0].name
            if tool_name in {
                "set_scene_time",
                "set_scene_location_presence",
                "set_scene_surface",
            }:
                self.started_tools.add(tool_name)
                if len(self.started_tools) == 3:
                    self.all_scene_tools_started.set()
                await asyncio.wait_for(
                    self.all_scene_tools_started.wait(),
                    timeout=1,
                )
            return ToolCallResponse(
                tool_calls=(),
                body="",
                provider=request.provider,
                model_id=request.model_id,
            )

    provider = BlockingSceneToolProvider()
    maintainer = module.ToolCallingFocusedSceneMaintainer(
        provider=provider,
        provider_name="fake",
        model_id="fake-context-update",
    )

    async def maintain() -> object:
        return await maintainer.maintain(
            module.FocusedSceneMaintenanceRequest(
                save_id=save.id,
                messages=(player_message, narrator_message),
                scene_snapshot=snapshot,
                locations=tuple(repositories.list_locations(save.id)),
                characters=(),
            )
        )

    asyncio.run(asyncio.wait_for(maintain(), timeout=2))

    assert provider.started_tools == {
        "set_scene_time",
        "set_scene_location_presence",
        "set_scene_surface",
    }


def test_focused_scene_manual_emotion_confirmation_dedupes_and_supersedes(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    repositories.set_app_setting("manual_confirmation_state_changes_enabled", True)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    ilyra = repositories.add_character(save_id=save.id, name="Captain Ilyra")
    emotion_key = "character.captain_ilyra.current_emotional_state"
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    service.apply_focused_scene_maintenance(
        save_id=save.id,
        maintenance=module.FocusedSceneMaintenance(
            character_emotions=(
                module.ExtractedFocusedCharacterEmotion(
                    character_id=ilyra.id,
                    emotional_state="guarded but relieved",
                    source_message_id="message-1",
                    evidence_quote="guarded but relieved",
                    confidence=0.75,
                ),
            )
        ),
    )
    service.apply_focused_scene_maintenance(
        save_id=save.id,
        maintenance=module.FocusedSceneMaintenance(
            character_emotions=(
                module.ExtractedFocusedCharacterEmotion(
                    character_id=ilyra.id,
                    emotional_state="guarded but relieved",
                    source_message_id="message-2",
                    evidence_quote="guarded but relieved",
                    confidence=0.95,
                ),
            )
        ),
    )

    suggestions = repositories.list_context_update_suggestions(save.id)
    assert len(suggestions) == 1
    assert suggestions[0].status == "pending"
    proposed_value = suggestions[0].proposed_value
    assert isinstance(proposed_value, dict)
    assert proposed_value["value"] == {"mood": "guarded but relieved"}

    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="upsert",
        entity_type="world_state",
        entity_id=None,
        field_path=emotion_key,
        proposed_value={
            "operation": "upsert",
            "key": emotion_key,
            "value": {"mood": "angry"},
            "category": "scene",
            "confidence": 0.2,
            "source_message_id": "message-stale",
        },
        status="pending",
        reason="Stale conflicting suggestion.",
        confidence=0.2,
        source_message_ids=["message-stale"],
    )
    service.apply_focused_scene_maintenance(
        save_id=save.id,
        maintenance=module.FocusedSceneMaintenance(
            character_emotions=(
                module.ExtractedFocusedCharacterEmotion(
                    character_id=ilyra.id,
                    emotional_state="guarded but relieved",
                    source_message_id="message-duplicate",
                    evidence_quote="guarded but relieved",
                    confidence=0.95,
                ),
            )
        ),
    )

    suggestions = repositories.list_context_update_suggestions(save.id)
    assert [suggestion.status for suggestion in suggestions] == [
        "pending",
        "superseded",
    ]

    service.apply_focused_scene_maintenance(
        save_id=save.id,
        maintenance=module.FocusedSceneMaintenance(
            character_emotions=(
                module.ExtractedFocusedCharacterEmotion(
                    character_id=ilyra.id,
                    emotional_state="angry",
                    source_message_id="message-3",
                    evidence_quote="angry",
                    confidence=0.84,
                ),
            )
        ),
    )

    suggestions = repositories.list_context_update_suggestions(save.id)
    assert [suggestion.status for suggestion in suggestions] == [
        "superseded",
        "superseded",
        "pending",
    ]
    proposed_value = suggestions[-1].proposed_value
    assert isinstance(proposed_value, dict)
    assert proposed_value["value"] == {"mood": "angry"}
    assert repositories.list_world_state(save.id) == []


def test_apply_focused_scene_maintenance_requires_evidence_quote(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    with pytest.raises(ValueError, match="focused scene evidence_quote is required"):
        service.apply_focused_scene_maintenance(
            save_id=save.id,
            maintenance=module.FocusedSceneMaintenance(
                scene_updates=(
                    module.ExtractedSceneSnapshot(
                        source_message_id=narrator_message.id,
                        current_location_name="Beacon Gallery",
                    ),
                )
            ),
            allowed_source_message_ids=(narrator_message.id,),
        )


def test_apply_focused_scene_maintenance_rejects_ungrounded_evidence_quote(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        in_world_time="morning",
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    with pytest.raises(
        ValueError,
        match="focused scene evidence_quote not found in source_message_id",
    ):
        service.apply_focused_scene_maintenance(
            save_id=save.id,
            maintenance=module.FocusedSceneMaintenance(
                scene_updates=(
                    module.ExtractedSceneSnapshot(
                        source_message_id=narrator_message.id,
                        evidence_quote="ruby library",
                        in_world_time="afternoon",
                    ),
                )
            ),
            allowed_source_message_ids=(narrator_message.id,),
            completed_messages=(narrator_message,),
        )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "morning"


def test_focused_scene_manual_emotion_confirmation_cleans_stale_current_state(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    repositories.set_app_setting("manual_confirmation_state_changes_enabled", True)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    ilyra = repositories.add_character(save_id=save.id, name="Captain Ilyra")
    emotion_key = "character.captain_ilyra.current_emotional_state"
    current_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Captain Ilyra keeps her guard up.",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key=emotion_key,
        value={"mood": "guarded"},
        category="scene",
        source_message_id=current_message.id,
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="upsert",
        entity_type="world_state",
        entity_id=state.id,
        field_path=emotion_key,
        proposed_value={
            "operation": "upsert",
            "key": emotion_key,
            "value": {"mood": "guarded"},
            "category": "scene",
            "confidence": 0.4,
            "source_message_id": "message-duplicate",
        },
        status="pending",
        reason="Stale same-value suggestion.",
        confidence=0.4,
        source_message_ids=["message-duplicate"],
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="upsert",
        entity_type="world_state",
        entity_id=state.id,
        field_path=emotion_key,
        proposed_value={
            "operation": "upsert",
            "key": emotion_key,
            "value": {"mood": "angry"},
            "category": "scene",
            "confidence": 0.3,
            "source_message_id": "message-stale",
        },
        status="pending",
        reason="Stale conflicting suggestion.",
        confidence=0.3,
        source_message_ids=["message-stale"],
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="upsert",
        entity_type="world_state",
        entity_id=None,
        field_path=emotion_key,
        proposed_value={
            "operation": "upsert",
            "key": emotion_key,
            "value": {"mood": "shaken"},
            "category": "scene",
            "confidence": 0.2,
            "source_message_id": "message-stale-null",
        },
        status="pending",
        reason="Stale suggestion from before the state row existed.",
        confidence=0.2,
        source_message_ids=["message-stale-null"],
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    applied = service.apply_focused_scene_maintenance(
        save_id=save.id,
        maintenance=module.FocusedSceneMaintenance(
            character_emotions=(
                module.ExtractedFocusedCharacterEmotion(
                    character_id=ilyra.id,
                    emotional_state="guarded",
                    source_message_id="message-new",
                    evidence_quote="guarded",
                    confidence=0.8,
                ),
            )
        ),
    )

    assert applied.empty
    suggestions = repositories.list_context_update_suggestions(save.id)
    assert len(suggestions) == 3
    assert [suggestion.status for suggestion in suggestions] == [
        "superseded",
        "superseded",
        "superseded",
    ]
    assert repositories.list_world_state(save.id)[0].value == {"mood": "guarded"}


def test_focused_scene_maintainer_updates_surface_threads_and_relationship_posture(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    ilyra = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        mood="quiet",
        nearby_objects=["signal horn"],
        hazards=["thin fog"],
        present_character_ids=[ilyra.id],
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Cool the beacon lens",
        description="The red lens is overheating.",
        status="active",
        visibility="scene",
        source_message_id=player_message.id,
    )
    narrator_message = repositories.update_message_body(
        save_id=save.id,
        message_id=narrator_message.id,
        body=(
            "Captain Ilyra kicks the ash bucket aside, points out the cracked "
            "lens, and tells Mara the immediate danger is handled. She now "
            "trusts Mara to make the next signal call."
        ),
    )
    provider = SequenceToolCallProvider(
        responses=[
            (),
            (),
            (
                ProviderToolCall(
                    id="surface-call",
                    name="set_scene_surface",
                    arguments_json=json.dumps(
                        {
                            "mood": "urgent but controlled",
                            "nearby_objects": ["ash bucket", "signal horn"],
                            "hazards": ["cracked lens"],
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "kicks the ash bucket aside",
                            "reason": "The current scene surface changed.",
                            "confidence": 0.86,
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="thread-call",
                    name="set_scene_thread_status",
                    arguments_json=json.dumps(
                        {
                            "active_thread_id": thread.id,
                            "status": "resolved",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "the immediate danger is handled",
                            "reason": "The scene-local hazard thread resolved.",
                            "confidence": 0.9,
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="relationship-call",
                    name="set_character_relationship_posture",
                    arguments_json=json.dumps(
                        {
                            "character_id": ilyra.id,
                            "target_name": "Mara",
                            "posture": "trusts Mara to make the next signal call",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": (
                                "trusts Mara to make the next signal call"
                            ),
                            "reason": "The current scene directly updated trust.",
                            "confidence": 0.84,
                        }
                    ),
                ),
            ),
            (),
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    applied = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.mood == "urgent but controlled"
    assert snapshot.nearby_objects == ["ash bucket", "signal horn"]
    assert snapshot.hazards == ["cracked lens"]

    assert repositories.get_active_thread(thread.id) is None
    updated_ilyra = repositories.get_character(ilyra.id)
    assert updated_ilyra is not None
    assert updated_ilyra.relationships == {
        "Mara": "trusts Mara to make the next signal call"
    }

    assert applied.job_result["focused_scene"]["scene_snapshot_updated"] is True
    assert applied.job_result["focused_scene"]["character_count"] == 1
    assert applied.job_result["focused_scene"]["active_thread_count"] == 1
    audit_by_field = {
        row.field_path: row for row in repositories.list_context_update_audit(save.id)
    }
    assert audit_by_field["mood"].after == "urgent but controlled"
    assert audit_by_field["nearby_objects"].after == ["ash bucket", "signal horn"]
    assert audit_by_field["hazards"].after == ["cracked lens"]
    assert any(
        row.entity_type == "active_thread"
        and row.operation == "archived"
        and row.entity_id == thread.id
        for row in repositories.list_context_update_audit(save.id)
    )
    assert audit_by_field["relationships"].after == {
        "Mara": "trusts Mara to make the next signal call"
    }


def test_focused_scene_thread_status_uses_thread_id_when_titles_match(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    repositories.upsert_scene_snapshot(save_id=save.id, current_location_id=location.id)
    first_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Clear the smoke",
        status="active",
        visibility="scene",
        source_message_id=player_message.id,
    )
    target_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Clear the smoke",
        status="active",
        visibility="scene",
        source_message_id=player_message.id,
    )
    narrator_message = repositories.update_message_body(
        save_id=save.id,
        message_id=narrator_message.id,
        body="Captain Ilyra steadies Mara and says the lower smoke is handled.",
    )
    provider = SequenceToolCallProvider(
        responses=[
            (),
            (),
            (),
            (
                ProviderToolCall(
                    id="thread-call",
                    name="set_scene_thread_status",
                    arguments_json=json.dumps(
                        {
                            "active_thread_id": target_thread.id,
                            "status": "resolved",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "the lower smoke is handled",
                            "reason": "Only the lower smoke thread resolved.",
                            "confidence": 0.91,
                        }
                    ),
                ),
            ),
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    applied = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert repositories.get_active_thread(first_thread.id) is not None
    assert repositories.get_active_thread(target_thread.id) is None
    assert applied.job_result["focused_scene"]["active_thread_count"] == 1
    archived = [
        row
        for row in repositories.list_context_update_audit(save.id)
        if row.entity_type == "active_thread" and row.operation == "archived"
    ]
    assert [row.entity_id for row in archived] == [target_thread.id]


def test_focused_scene_relationship_posture_queues_locked_relationship_suggestion(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    ilyra = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        relationships={},
        locked_fields=["relationships"],
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        present_character_ids=[ilyra.id],
    )
    narrator_message = repositories.update_message_body(
        save_id=save.id,
        message_id=narrator_message.id,
        body="Captain Ilyra steadies Mara and says she trusts Mara's signal call.",
    )
    provider = SequenceToolCallProvider(
        responses=[
            (),
            (),
            (),
            (
                ProviderToolCall(
                    id="relationship-call",
                    name="set_character_relationship_posture",
                    arguments_json=json.dumps(
                        {
                            "character_id": ilyra.id,
                            "target_name": "Mara",
                            "posture": "trusts Mara's signal call",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "trusts Mara's signal call",
                            "reason": "The current scene establishes trust.",
                            "confidence": 0.82,
                        }
                    ),
                ),
            ),
            (),
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    applied = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    unchanged = repositories.get_character(ilyra.id)
    assert unchanged is not None
    assert unchanged.relationships == {}
    assert applied.job_result["focused_scene"]["character_count"] == 0
    assert applied.job_result["focused_scene"]["suggestion_count"] == 1
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.entity_type == "character"
    assert suggestion.entity_id == ilyra.id
    assert suggestion.field_path == "relationships"
    assert suggestion.proposed_value == {"Mara": "trusts Mara's signal call"}


def test_focused_scene_relationship_posture_queues_conflicting_relationship_review(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    ilyra = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        relationships={"Mara": "keeps Mara at arm's length"},
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        present_character_ids=[ilyra.id],
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
    )

    applied = service.apply_focused_scene_maintenance(
        save_id=save.id,
        maintenance=module.FocusedSceneMaintenance(
            character_relationships=(
                module.ExtractedFocusedCharacterRelationship(
                    character_id=ilyra.id,
                    target_name="Mara",
                    posture="trusts Mara with the beacon key",
                    source_message_id=narrator_message.id,
                    evidence_quote="Captain Ilyra",
                    reason="The posture conflicts with the existing stance.",
                    confidence=0.83,
                ),
            )
        ),
        allowed_source_message_ids=(player_message.id, narrator_message.id),
        completed_messages=(player_message, narrator_message),
    )

    unchanged = repositories.get_character(ilyra.id)
    assert unchanged is not None
    assert unchanged.relationships == {"Mara": "keeps Mara at arm's length"}
    assert applied.characters == ()
    assert len(applied.suggestions) == 1
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.entity_type == "character"
    assert suggestion.entity_id == ilyra.id
    assert suggestion.field_path == "relationships"
    assert suggestion.proposed_value == {"Mara": "trusts Mara with the beacon key"}


def test_focused_scene_relationship_posture_queues_absent_target_review(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    ilyra = repositories.add_character(save_id=save.id, name="Captain Ilyra")
    repositories.add_character(save_id=save.id, name="Archivist Ren")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        present_character_ids=[ilyra.id],
    )
    narrator_message = repositories.update_message_body(
        save_id=save.id,
        message_id=narrator_message.id,
        body="Captain Ilyra steadies Mara while Ren remains below.",
    )
    provider = SequenceToolCallProvider(
        responses=[
            (),
            (),
            (),
            (
                ProviderToolCall(
                    id="relationship-call",
                    name="set_character_relationship_posture",
                    arguments_json=json.dumps(
                        {
                            "character_id": ilyra.id,
                            "target_name": "Archivist Ren",
                            "posture": "trusts Ren's timing",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "Ren remains below",
                            "reason": "The model tried to update an absent target.",
                            "confidence": 0.7,
                        }
                    ),
                ),
            ),
            (),
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    applied = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    unchanged = repositories.get_character(ilyra.id)
    assert unchanged is not None
    assert unchanged.relationships == {}
    assert applied.job_result["focused_scene"]["character_count"] == 0
    assert applied.job_result["focused_scene"]["suggestion_count"] == 1
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.entity_type == "character"
    assert suggestion.entity_id == ilyra.id
    assert suggestion.field_path == "relationships"
    assert suggestion.proposed_value == {"Archivist Ren": "trusts Ren's timing"}


def test_focused_scene_maintainer_updates_protected_relationships_only(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    ilyra = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        relationships={"Mara": "available route for Mara"},
        protected_from_maintenance=True,
    )
    ren = repositories.add_character(
        save_id=save.id,
        name="Archivist Ren",
        relationships={"Mara": "wary"},
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        present_character_ids=[ilyra.id],
    )
    public_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Public beacon obligation",
        status="active",
        visibility="public",
        source_message_id=player_message.id,
    )
    scene_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Scene-only lens smoke",
        status="active",
        visibility="scene",
        source_message_id=player_message.id,
    )
    provider = SequenceToolCallProvider(
        responses=[
            (),
            (),
            (),
            (),
            (
                ProviderToolCall(
                    id="relationship-call",
                    name="set_character_relationship_posture",
                    arguments_json=json.dumps(
                        {
                            "character_id": ilyra.id,
                            "target_name": "Mara",
                            "posture": "trusts Mara with the next signal call",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "Captain Ilyra steadies Mara",
                            "reason": (
                                "The protected starter's current posture changed."
                            ),
                            "confidence": 0.86,
                        }
                    ),
                ),
            ),
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    applied = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert repositories.get_active_thread(public_thread.id) is not None
    assert repositories.get_active_thread(scene_thread.id) is not None
    protected_ilyra = repositories.get_character(ilyra.id)
    assert protected_ilyra is not None
    assert protected_ilyra.protected_from_maintenance is True
    assert protected_ilyra.relationships == {
        "Mara": "trusts Mara with the next signal call"
    }
    unchanged_ren = repositories.get_character(ren.id)
    assert unchanged_ren is not None
    assert unchanged_ren.relationships == {"Mara": "wary"}
    assert applied.job_result["focused_scene"]["active_thread_count"] == 0
    assert applied.job_result["focused_scene"]["character_count"] == 1
    assert [
        tuple(tool.name for tool in request.tools)
        for request in provider.tool_call_requests
    ] == [
        ("set_scene_time",),
        ("set_scene_location_presence",),
        ("set_scene_surface",),
        ("set_scene_thread_status",),
        ("set_character_relationship_posture",),
    ]
    assert repositories.list_context_update_suggestions(save.id) == []


def test_focused_scene_tool_prompts_scope_registry_lists(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    for index in range(30):
        repositories.add_location(
            save_id=save.id,
            name=f"Overflow Location {index:02d}",
        )
    ilyra = repositories.add_character(save_id=save.id, name="Captain Ilyra")
    for index in range(40):
        repositories.add_character(
            save_id=save.id,
            name=f"Overflow Character {index:02d}",
        )
    current_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Current smoke thread",
        status="active",
        visibility="scene",
    )
    overflow_threads = []
    for index in range(12):
        overflow_threads.append(
            repositories.add_active_thread(
                save_id=save.id,
                title=f"Overflow Scene Thread {index:02d}",
                status="active",
                visibility="scene",
            )
        )
    public_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Public beacon obligation",
        status="active",
        visibility="public",
    )
    hidden_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Hidden scene secret",
        status="active",
        visibility="hidden",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        present_character_ids=[ilyra.id],
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ask Ilyra who else is nearby.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Captain Ilyra scans the gallery and keeps her voice low.",
        provider="fake",
        model="fake-chat",
    )
    provider = SequenceToolCallProvider(responses=[(), (), (), (), (), ()])
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    request_bodies = [
        _tool_call_user_body(request)
        for request in provider.tool_call_requests
    ]
    assert len(request_bodies) == 6
    assert "Known locations" not in request_bodies[0]
    assert "Known characters" not in request_bodies[0]
    assert "Known locations (showing 24 of 31; 7 omitted)" in request_bodies[1]
    assert "Known characters (showing 32 of 41; 9 omitted)" in request_bodies[1]
    assert "Beacon Gallery" in request_bodies[1]
    assert "Captain Ilyra" in request_bodies[1]
    assert "Overflow Location 29" not in request_bodies[1]
    assert "Overflow Character 39" not in request_bodies[1]
    assert "Known locations" not in request_bodies[2]
    assert "Known characters" not in request_bodies[2]
    assert "scene.nearby_objects" in request_bodies[2]
    assert (
        "Known scene-local active threads (showing 12 of 13; 1 omitted)"
        in request_bodies[3]
    )
    assert "Current smoke thread" in request_bodies[3]
    assert current_thread.id in request_bodies[3]
    assert overflow_threads[10].id in request_bodies[3]
    assert "Overflow Scene Thread 11" not in request_bodies[3]
    assert overflow_threads[11].id not in request_bodies[3]
    assert "Public beacon obligation" not in request_bodies[3]
    assert public_thread.id not in request_bodies[3]
    assert "Hidden scene secret" not in request_bodies[3]
    assert hidden_thread.id not in request_bodies[3]
    thread_request = provider.tool_call_requests[3]
    thread_schema = cast(Any, thread_request.tools[0].parameters)
    thread_id_schema = thread_schema["properties"]["active_thread_id"]
    assert thread_id_schema["enum"] == [
        current_thread.id,
        *[thread.id for thread in overflow_threads[:11]],
    ]
    assert "Relationship subject: Captain Ilyra" in request_bodies[4]
    assert "Assigned character: Captain Ilyra" in request_bodies[5]


def test_focused_scene_maintainer_ignores_vague_time_updates(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        in_world_time="morning",
    )
    provider = SequenceToolCallProvider(
        responses=[
            (
                ProviderToolCall(
                    id="vague-time-call",
                    name="set_scene_time",
                    arguments_json=json.dumps(
                        {
                            "in_world_time": "later",
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "steadies Mara",
                            "reason": "The model tried to infer vague time.",
                            "confidence": 0.5,
                        }
                    ),
                ),
            ),
            (),
            (),
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "morning"


def test_focused_scene_maintainer_does_not_create_unknown_scene_entities(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    ilyra = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        location_id=location.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        present_character_ids=[ilyra.id],
    )
    narrator_message = repositories.update_message_body(
        save_id=save.id,
        message_id=narrator_message.id,
        body="A stranger calls from the Moon Cellar, but Ilyra stays put.",
    )
    provider = SequenceToolCallProvider(
        responses=[
            (),
            (
                ProviderToolCall(
                    id="unknown-presence-call",
                    name="set_scene_location_presence",
                    arguments_json=json.dumps(
                        {
                            "current_location_name": "Moon Cellar",
                            "present_character_names": [
                                "Captain Ilyra",
                                "The Stranger",
                            ],
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "Moon Cellar",
                            "reason": "The model tried to introduce unknowns.",
                            "confidence": 0.73,
                        }
                    ),
                ),
            ),
            (),
            (),
            (),
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.current_location_id == location.id
    assert snapshot.present_character_ids == [ilyra.id]
    assert [record.name for record in repositories.list_locations(save.id)] == [
        "Beacon Gallery"
    ]
    assert [record.name for record in repositories.list_characters(save.id)] == [
        "Captain Ilyra"
    ]


def test_focused_scene_transition_expires_scratch_at_same_location(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    scene = repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="The watch studies the cracked lens.",
    )
    scratch = repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id="same-location-scratch",
        title="Temporary lens state",
        body="The cracked lens is still warm.",
        metadata={"curation_action": "scene_scratch"},
        scene_snapshot_id=scene.id,
        scene_generation=scene.scene_generation,
        created_turn_number=1,
        expires_after_turn_number=13,
    )
    narrator_message = repositories.update_message_body(
        save_id=save.id,
        message_id=narrator_message.id,
        body="The narration cuts to a new scene in the Beacon Gallery.",
    )
    provider = SequenceToolCallProvider(
        responses=[
            (),
            (
                ProviderToolCall(
                    id="same-location-scene-transition",
                    name="set_scene_location_presence",
                    arguments_json=json.dumps(
                        {
                            "current_location_name": "Beacon Gallery",
                            "scene_transition": True,
                            "source_message_id": narrator_message.id,
                            "evidence_quote": (
                                "cuts to a new scene in the Beacon Gallery"
                            ),
                            "reason": "The narration establishes a new scene.",
                            "confidence": 0.98,
                        }
                    ),
                ),
            ),
            (),
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    updated_scene = repositories.get_scene_snapshot(save.id)
    assert updated_scene is not None
    assert updated_scene.current_location_id == location.id
    assert updated_scene.scene_generation == scene.scene_generation + 1
    assert repositories.get_context_source(scratch.id) is None


def test_focused_scene_transition_advances_when_location_change_is_locked(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    old_location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
    )
    repositories.add_location(save_id=save.id, name="Gatehouse")
    scene = repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=old_location.id,
        situation="The watch studies the cracked lens.",
        locked_fields=["current_location_id"],
    )
    scratch = repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id="locked-location-scratch",
        title="Temporary lens state",
        body="The cracked lens is still warm.",
        metadata={"curation_action": "scene_scratch"},
        scene_snapshot_id=scene.id,
        scene_generation=scene.scene_generation,
        created_turn_number=1,
        expires_after_turn_number=13,
    )
    narrator_message = repositories.update_message_body(
        save_id=save.id,
        message_id=narrator_message.id,
        body="The narration cuts to a new scene in the Gatehouse.",
    )
    provider = SequenceToolCallProvider(
        responses=[
            (),
            (
                ProviderToolCall(
                    id="locked-location-scene-transition",
                    name="set_scene_location_presence",
                    arguments_json=json.dumps(
                        {
                            "current_location_name": "Gatehouse",
                            "scene_transition": True,
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "cuts to a new scene in the Gatehouse",
                            "reason": "The narration establishes a new scene.",
                            "confidence": 0.98,
                        }
                    ),
                ),
            ),
            (),
        ]
    )

    asyncio.run(
        module.ContextUpdateService(
            repositories=repositories,
            extractor=RecordingContextUpdateExtractor(
                module.ContextUpdateExtraction()
            ),
            focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
                provider=provider,
                provider_name="fake",
                model_id="fake-context-update",
            ),
        ).update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    updated = repositories.get_scene_snapshot(save.id)
    assert updated is not None
    assert updated.current_location_id == old_location.id
    assert updated.scene_generation == scene.scene_generation + 1
    assert repositories.get_context_source(scratch.id) is None
    assert any(
        suggestion.entity_type == "scene_snapshot"
        and suggestion.field_path == "current_location_id"
        for suggestion in repositories.list_context_update_suggestions(save.id)
    )


def test_scene_transition_advances_when_extracted_location_change_is_locked(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    old_location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
    )
    repositories.add_location(save_id=save.id, name="Gatehouse")
    scene = repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=old_location.id,
        situation="The watch studies the cracked lens.",
        nearby_objects=["signal horn"],
        hazards=["hot glass"],
        locked_fields=["current_location_id"],
    )
    scratch = repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id="broad-locked-location-scratch",
        title="Temporary lens state",
        body="The cracked lens is still warm.",
        metadata={"curation_action": "scene_scratch"},
        scene_snapshot_id=scene.id,
        scene_generation=scene.scene_generation,
        created_turn_number=1,
        expires_after_turn_number=13,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            current_location_name="Gatehouse",
            scene_transition=True,
            reason="The narration establishes a new scene.",
            confidence=0.98,
        )
    )

    asyncio.run(
        module.ContextUpdateService(
            repositories=repositories,
            extractor=RecordingContextUpdateExtractor(extraction),
        ).update_after_turn(
            save_id=save.id,
            source_message_ids=(narrator_message.id,),
        )
    )

    updated = repositories.get_scene_snapshot(save.id)
    assert updated is not None
    assert updated.current_location_id == old_location.id
    assert updated.scene_generation == scene.scene_generation + 1
    assert updated.nearby_objects == ["signal horn"]
    assert updated.hazards == ["hot glass"]
    assert repositories.get_context_source(scratch.id) is None


def test_focused_scene_maintainer_uses_snapshot_emotion_scope_for_invalid_presence(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    repositories.add_character(save_id=save.id, name="Captain Ilyra")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        present_character_ids=[],
    )
    narrator_message = repositories.update_message_body(
        save_id=save.id,
        message_id=narrator_message.id,
        body="Captain Ilyra and a stranger call from the Moon Cellar.",
    )
    provider = SequenceToolCallProvider(
        responses=[
            (),
            (
                ProviderToolCall(
                    id="partial-presence-call",
                    name="set_scene_location_presence",
                    arguments_json=json.dumps(
                        {
                            "present_character_names": [
                                "Captain Ilyra",
                                "The Stranger",
                            ],
                            "source_message_id": narrator_message.id,
                            "evidence_quote": "Captain Ilyra and a stranger",
                            "reason": "One proposed presence name is unknown.",
                            "confidence": 0.74,
                        }
                    ),
                ),
            ),
            (),
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(module.ContextUpdateExtraction()),
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == []
    assert [
        tuple(tool.name for tool in request.tools)
        for request in provider.tool_call_requests
    ] == [
        ("set_scene_time",),
        ("set_scene_location_presence",),
        ("set_scene_surface",),
    ]
    assert repositories.list_world_state(save.id) == []


def test_focused_scene_maintainer_failure_does_not_fail_broad_context_update(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extractor = RecordingContextUpdateExtractor(
        module.ContextUpdateExtraction(
            scene=module.ExtractedSceneSnapshot(
                current_location_name="Beacon Gallery",
                situation="The beacon gallery steadies after the climb.",
                source_message_id=narrator_message.id,
            )
        )
    )
    provider = SequenceToolCallProvider(responses=[RuntimeError("focused exploded")])
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.situation == "The beacon gallery steadies after the climb."
    assert repositories.list_jobs_by_status(("failed",)) == []
    succeeded_context_jobs = [
        job
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.type == "context_update"
    ]
    assert len(succeeded_context_jobs) == 1
    result = succeeded_context_jobs[0].result
    assert result is not None
    assert result["focused_scene"] == {
        "scene_snapshot_updated": False,
        "character_count": 0,
        "active_thread_count": 0,
        "world_state_count": 0,
        "suggestion_count": 0,
        "audit_count": 0,
        "state_change_count": 0,
        "tool_diagnostics": {"error": "focused exploded"},
    }


def test_focused_scene_provider_pressure_surfaces_in_context_job_result(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extractor = RecordingContextUpdateExtractor(
        module.ContextUpdateExtraction(
            scene=module.ExtractedSceneSnapshot(
                current_location_name="Beacon Gallery",
                source_message_id=narrator_message.id,
            )
        )
    )
    provider = SequenceToolCallProvider(
        responses=[
            ProviderError(
                category=ProviderErrorCategory.RATE_LIMITED,
                message="focused maintenance is throttled",
                status_code=429,
                retry_attempt_count=3,
                max_retry_attempts=3,
            )
        ]
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
        focused_scene_maintainer=module.ToolCallingFocusedSceneMaintainer(
            provider=provider,
            provider_name="fake",
            model_id="fake-context-update",
        ),
    )

    applied = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert applied.job_result["provider_pressure"] == {
        "reason": "provider_pressure",
        "error_category": "rate_limited",
        "http_status": 429,
        "retry_attempt_count": 3,
        "max_retry_attempts": 3,
    }
    succeeded_context_jobs = [
        job
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.type == "context_update"
    ]
    assert len(succeeded_context_jobs) == 1
    result = succeeded_context_jobs[0].result
    assert result is not None
    assert result["provider_pressure"] == applied.job_result["provider_pressure"]


def test_context_update_parses_omitted_scene_presence_as_no_change() -> None:
    module = _context_update_module()

    omitted = module.context_update_extraction_from_structured_data(
        {
            "scene": {
                "source_message_id": "message-1",
                "situation": "The lens hums under pressure.",
            },
            "locations": [],
            "characters": [],
            "active_threads": [],
            "entity_links": [],
        }
    )
    explicit_clear = module.context_update_extraction_from_structured_data(
        {
            "scene": {
                "source_message_id": "message-1",
                "situation": "The tower is empty.",
                "nearby_objects": [],
                "hazards": [],
                "present_character_names": [],
            },
            "locations": [],
            "characters": [],
            "active_threads": [],
            "entity_links": [],
        }
    )
    explicit_null = module.context_update_extraction_from_structured_data(
        {
            "scene": {
                "source_message_id": "message-1",
                "situation": "The pressure is unchanged.",
                "nearby_objects": None,
                "hazards": None,
                "present_character_names": None,
            },
            "locations": [],
            "characters": [],
            "active_threads": [],
            "entity_links": [],
        }
    )

    assert omitted.scene is not None
    assert omitted.scene.nearby_objects is None
    assert omitted.scene.hazards is None
    assert omitted.scene.present_character_names is None
    assert explicit_clear.scene is not None
    assert explicit_clear.scene.nearby_objects == ()
    assert explicit_clear.scene.hazards == ()
    assert explicit_clear.scene.present_character_names == ()
    assert explicit_null.scene is not None
    assert explicit_null.scene.nearby_objects is None
    assert explicit_null.scene.hazards is None
    assert explicit_null.scene.present_character_names is None


@pytest.mark.parametrize(
    ("extracted_time", "expected_time"),
    (
        ("It is 8 AM.", "8 AM (morning)"),
        ("around 14:30", "14:30 (afternoon)"),
        (
            "The countdown timer has 03:45 remaining until dawn. By evening, we hide.",
            "evening",
        ),
        (
            (
                "The countdown timer has 03:45 remaining until 14:30, so I wait "
                "until evening."
            ),
            "evening",
        ),
        (
            "The game clock shows 03:45, so by evening we hide.",
            "evening",
        ),
        (
            "The game clock shows 3 p.m., so by evening we hide.",
            "evening",
        ),
        (
            "I wait until evening while the countdown timer shows 03:45 remaining.",
            "evening",
        ),
        (
            "We meet in the northern quarter at 14:30.",
            "14:30 (afternoon)",
        ),
        (
            "We meet at 14:30 in the half-light.",
            "14:30 (afternoon)",
        ),
    ),
)
def test_update_after_turn_normalizes_supported_scene_time_when_no_anchor(
    repositories: PersistenceRepositories,
    extracted_time: str,
    expected_time: str,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            in_world_time=extracted_time,
            reason="The narrator stated an explicit clock time.",
            confidence=0.9,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == expected_time


def test_update_after_turn_normalizes_non_timer_clock_when_timer_readout_appears_first(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            in_world_time=(
                "The countdown timer shows 03:45 remaining, so at 14:30 we leave."
            ),
            reason="The narrator established a later departure time.",
            confidence=0.9,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "14:30 (afternoon)"


def test_update_after_turn_preserves_scene_time_for_vague_extracted_time(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="late morning",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            in_world_time="later",
            reason="The narrator used vague time.",
            confidence=0.7,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "late morning"
    assert _count(result, "updated_fields") == 0
    assert repositories.list_context_update_suggestions(save.id) == []


@pytest.mark.parametrize(
    "extracted_time",
    (
        "countdown timer: 03:45 remaining",
        "03:45 left in the round",
        "elapsed time 11:20",
        "The countdown timer has 03:45 remaining until dawn.",
        "The game clock shows 03:45.",
        "The scoreboard reads 03:45.",
        "I wait until the shot clock hits 00:12.",
        "I wait until the shot clock hits 12 seconds.",
        "The game clock shows 3:45 p.m.",
        "The game clock shows 3 p.m.",
    ),
)
def test_update_after_turn_preserves_scene_time_for_timer_and_game_readout_values(
    repositories: PersistenceRepositories,
    extracted_time: str,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=0,
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            in_world_time=extracted_time,
            reason="The narrator mentioned a countdown timer.",
            confidence=0.8,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "morning"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "monday"
    assert snapshot.world_day_index == 0
    assert _count(result, "updated_fields") == 0
    assert repositories.list_context_update_suggestions(save.id) == []


def test_update_after_turn_queues_narrator_only_scene_time_phase_jump(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="morning",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            in_world_time="dinner at night",
            reason="The narrator jumped to dinner.",
            confidence=0.8,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "morning"
    assert _count(result, "queued_suggestions") == 1
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.entity_type == "scene_snapshot"
    assert suggestion.field_path == "in_world_time"
    assert suggestion.proposed_value == "night"


@pytest.mark.parametrize(
    "player_body",
    (
        "I look at evening lanterns.",
        "I left my notebook at work.",
        "At work, I sharpen the blade.",
        "I wait to see what happens.",
    ),
)
def test_update_after_turn_queues_scene_time_for_non_advancing_player_time_phrase(
    repositories: PersistenceRepositories,
    player_body: str,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.connection.execute(
        "UPDATE messages SET body = ? WHERE id = ?",
        (player_body, player_message.id),
    )
    repositories.commit()
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="morning",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            in_world_time="evening",
            reason="The narrator described evening imagery.",
            confidence=0.82,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "morning"
    assert _count(result, "queued_suggestions") == 1
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.field_path == "in_world_time"
    assert suggestion.proposed_value == "evening"


@pytest.mark.parametrize(
    ("player_body", "extracted_time", "expected_time"),
    (
        ("I wait until dinner.", "evening", "evening"),
        ("After class, I go to her room.", "afternoon", "afternoon"),
        ("Skip ahead to evening.", "night", "night"),
        (
            "We spend the afternoon traveling to the observatory.",
            "afternoon",
            "afternoon",
        ),
        ("We wait half an hour.", "afternoon", "afternoon"),
        ("I head home after the meeting.", "evening", "evening"),
        ("Later that night, I return to the tower.", "night", "night"),
        ("The next day, I check the beacon again.", "afternoon", "afternoon"),
    ),
)
def test_update_after_turn_applies_player_authorized_scene_time_phase_change(
    repositories: PersistenceRepositories,
    player_body: str,
    extracted_time: str,
    expected_time: str,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.connection.execute(
        "UPDATE messages SET body = ? WHERE id = ?",
        (player_body, player_message.id),
    )
    repositories.commit()
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="morning",
        world_time_clock_minutes=9 * 60 + 30,
        world_time_period_label="festival week",
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            in_world_time=extracted_time,
            reason="The completed turn moved time with player permission.",
            confidence=0.87,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == f"festival week {expected_time} at 09:30"
    assert snapshot.world_time_phase == expected_time
    assert snapshot.world_time_clock_minutes == 9 * 60 + 30
    assert snapshot.world_time_period_label == "festival week"
    assert _count(result, "updated_fields") == 1
    assert repositories.list_context_update_suggestions(save.id) == []


def test_update_after_turn_respects_canonical_scene_time_phase_lock(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.connection.execute(
        "UPDATE messages SET body = ? WHERE id = ?",
        ("I wait until evening.", player_message.id),
    )
    repositories.commit()
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="morning",
        time_of_day="morning",
        world_time_phase="morning",
        locked_fields=["world_time_phase"],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            in_world_time="evening",
            reason="The completed turn moved time with player permission.",
            confidence=0.87,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "morning"
    assert snapshot.world_time_phase == "morning"
    assert _count(result, "queued_suggestions") == 1
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.field_path == "in_world_time"
    assert suggestion.proposed_value == "evening"


def test_update_after_turn_respects_legacy_scene_time_lock(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.connection.execute(
        "UPDATE messages SET body = ? WHERE id = ?",
        ("I wait until evening.", player_message.id),
    )
    repositories.commit()
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="morning",
        time_of_day="morning",
        world_time_phase="morning",
        locked_fields=["time_of_day"],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            in_world_time="evening",
            reason="The completed turn moved time with player permission.",
            confidence=0.87,
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "morning"
    assert snapshot.time_of_day == "morning"
    assert snapshot.world_time_phase == "morning"
    assert _count(result, "queued_suggestions") == 1


def test_update_after_turn_auto_applies_empty_and_additive_unlocked_fields(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        aliases=["upper lens room"],
        status="",
        hazards=["cracked lens"],
        source_message_id=player_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        locations=(
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id=narrator_message.id,
                aliases=("upper lens room", "red lens chamber"),
                description="",
                visual_description="",
                connections=(),
                status="unstable",
                hazards=("cracked lens", "ash leak"),
                reason="The narrator added details about the beacon gallery.",
                confidence=0.87,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "updated_fields") == 3
    assert _count(result, "queued_suggestions") == 0
    assert repositories.list_context_update_suggestions(save.id) == []

    updated = repositories.get_location(location.id)
    assert updated is not None
    assert updated.aliases == ["upper lens room", "red lens chamber"]
    assert updated.status == "unstable"
    assert updated.hazards == ["cracked lens", "ash leak"]
    assert updated.source_message_id == narrator_message.id

    audit_rows = repositories.list_context_update_audit(save.id)
    assert [audit.field_path for audit in audit_rows] == [
        "aliases",
        "status",
        "hazards",
    ]
    assert [audit.operation for audit in audit_rows] == [
        "updated",
        "updated",
        "updated",
    ]
    audit_by_field = {audit.field_path: audit for audit in audit_rows}
    assert audit_by_field["aliases"].before == ["upper lens room"]
    assert audit_by_field["aliases"].after == [
        "upper lens room",
        "red lens chamber",
    ]
    assert audit_by_field["status"].before == ""
    assert audit_by_field["status"].after == "unstable"
    assert audit_by_field["hazards"].before == ["cracked lens"]
    assert audit_by_field["hazards"].after == ["cracked lens", "ash leak"]
    for audit in audit_by_field.values():
        assert audit.operation == "updated"
        assert audit.entity_type == "location"
        assert audit.entity_id == location.id
        assert audit.reason == "The narrator added details about the beacon gallery."
        assert audit.confidence == 0.87
        assert audit.source_message_ids == [narrator_message.id]


def test_update_after_turn_queues_suggestion_for_locked_field_without_mutation(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        status="sealed",
        source_message_id=player_message.id,
        locked_fields=["status"],
    )
    extraction = module.ContextUpdateExtraction(
        locations=(
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id=narrator_message.id,
                aliases=(),
                description="",
                visual_description="",
                connections=(),
                status="unstable",
                hazards=(),
                reason="The narrator described the locked status changing.",
                confidence=0.84,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "queued_suggestions") == 1
    unchanged = repositories.get_location(location.id)
    assert unchanged is not None
    assert unchanged.status == "sealed"
    assert unchanged.source_message_id == player_message.id

    suggestions = repositories.list_context_update_suggestions(save.id)
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.status == "pending"
    assert suggestion.update_type == "update"
    assert suggestion.entity_type == "location"
    assert suggestion.entity_id == location.id
    assert suggestion.field_path == "status"
    assert suggestion.proposed_value == "unstable"
    assert suggestion.reason == "The narrator described the locked status changing."
    assert suggestion.confidence == 0.84
    assert suggestion.source_message_ids == [narrator_message.id]

    audit_rows = repositories.list_context_update_audit(save.id)
    assert len(audit_rows) == 1
    audit = audit_rows[0]
    assert audit.operation == "queued"
    assert audit.suggestion_id == suggestion.id
    assert audit.before == "sealed"
    assert audit.after == "unstable"
    assert audit.source_message_ids == [narrator_message.id]


def test_update_after_turn_queues_locked_character_facts_but_updates_unlocked_fields(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        appearance="Ash-stained officer coat.",
        current_clothing="Sleeveless gray work tunic.",
        voice="Clipped and dry.",
        status="present",
        source_message_id=player_message.id,
        locked_fields=["appearance", "voice"],
        character_id="character-ilyra",
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Captain Ilyra",
                source_message_id=narrator_message.id,
                appearance="Red cloak and molten-glass saber.",
                current_clothing="Borrowed green raincoat over a linen shirt.",
                voice="Booming theatrical commands.",
                status="wounded",
                reason=(
                    "The narrator updates Ilyra's state but not her locked identity."
                ),
                confidence=0.88,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "characters") == 1
    assert _count(result, "queued_suggestions") == 2
    updated = repositories.get_character(character.id)
    assert updated is not None
    assert updated.appearance == "Ash-stained officer coat."
    assert updated.current_clothing == "Borrowed green raincoat over a linen shirt."
    assert updated.voice == "Clipped and dry."
    assert updated.status == "wounded"

    suggestions = repositories.list_context_update_suggestions(save.id)
    assert {
        (suggestion.field_path, suggestion.proposed_value)
        for suggestion in suggestions
    } == {
        ("appearance", "Red cloak and molten-glass saber."),
        ("voice", "Booming theatrical commands."),
    }

    repeated = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(repeated, "queued_suggestions") == 0
    assert repositories.list_context_update_suggestions(save.id) == suggestions
    assert {
        row.field_path
        for row in repositories.list_context_update_audit(save.id)
        if row.operation == "updated"
    } == {"current_clothing", "status"}


def test_update_after_turn_keeps_protected_starter_identity_locked(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        role="Precise festival organizer.",
        appearance="Sharp blazer and silver council pin.",
        voice="Crisp formal phrasing.",
        status="available romance option at scenario start",
        relationships={"Ren Takahashi": "romance option for Ren Takahashi"},
        source_message_id=player_message.id,
        locked_fields=["name", "role", "appearance", "voice"],
        protected_from_maintenance=True,
        character_id="character-mika",
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Mika Arai",
                source_message_id=narrator_message.id,
                role="Secret rebel courier.",
                appearance="Red cloak and molten-glass saber.",
                voice="Booming theatrical commands.",
                status="guardedly hopeful after Ren helps with the festival",
                reason="The model tried to rewrite identity while updating posture.",
                confidence=0.89,
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    updated = repositories.get_character(character.id)
    assert updated is not None
    assert updated.protected_from_maintenance is True
    assert updated.role == "Precise festival organizer."
    assert updated.appearance == "Sharp blazer and silver council pin."
    assert updated.voice == "Crisp formal phrasing."
    assert updated.status == "guardedly hopeful after Ren helps with the festival"
    assert updated.relationships == {
        "Ren Takahashi": "romance option for Ren Takahashi"
    }
    assert _count(result, "characters") == 1
    assert _count(result, "queued_suggestions") == 3
    assert {
        (suggestion.field_path, suggestion.proposed_value)
        for suggestion in repositories.list_context_update_suggestions(save.id)
    } == {
        ("role", "Secret rebel courier."),
        ("appearance", "Red cloak and molten-glass saber."),
        ("voice", "Booming theatrical commands."),
    }


def test_update_after_turn_updates_unlocked_agency_fields_and_queues_locked_agency(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        goals="Keep the beacon quiet.",
        current_intent="Guard the lens stair.",
        boundaries="Will not abandon the tower.",
        source_message_id=player_message.id,
        locked_fields=["boundaries"],
        character_id="character-ilyra",
    )
    extraction = module.ContextUpdateExtraction(
        characters=(
            module.ExtractedCharacter(
                name="Captain Ilyra",
                source_message_id=narrator_message.id,
                goals="Keep the red lens under control until dawn.",
                current_intent="Demand proof before sharing the failsafe.",
                boundaries="Will not hand over the lens key.",
                reason="The completed turn updates Ilyra's immediate agency.",
                confidence=0.86,
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert _count(result, "characters") == 1
    assert _count(result, "queued_suggestions") == 1
    updated = repositories.get_character(character.id)
    assert updated is not None
    assert updated.goals == "Keep the red lens under control until dawn."
    assert updated.current_intent == "Demand proof before sharing the failsafe."
    assert updated.boundaries == "Will not abandon the tower."

    suggestions = repositories.list_context_update_suggestions(save.id)
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.entity_type == "character"
    assert suggestion.entity_id == character.id
    assert suggestion.field_path == "boundaries"
    assert suggestion.proposed_value == "Will not hand over the lens key."
    assert suggestion.source_message_ids == [narrator_message.id]

    audit_rows = repositories.list_context_update_audit(save.id)
    assert {
        row.field_path for row in audit_rows if row.operation == "updated"
    } == {"goals", "current_intent"}
    assert [
        row.field_path for row in audit_rows if row.operation == "queued"
    ] == ["boundaries"]


def test_update_after_turn_drops_unknown_source_items_and_scene_source(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, player_message, narrator_message = _save_with_completed_turn(repositories)
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id="turn_1",
            evidence_quote="Captain Ilyra",
            current_location_name="Beacon Gallery",
            situation="The lens is waking up.",
        ),
        locations=(
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id="not-a-completed-turn-message",
                evidence_quote="beacon gallery",
                aliases=(),
                description="A hot room above the keep wall.",
                visual_description="",
                connections=(),
                status="unstable",
                hazards=(),
                reason="This source id is not in the completed turn.",
                confidence=0.7,
            ),
        ),
    )
    extractor = RecordingContextUpdateExtractor(extraction)
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(player_message.id, narrator_message.id),
        )
    )

    assert result.scene_snapshot is None
    assert repositories.get_scene_snapshot(save.id) is None
    assert repositories.list_locations(save.id) == []
    assert repositories.list_characters(save.id) == []
    assert repositories.list_active_threads(save.id) == []
    assert repositories.list_entity_links(save.id) == []
    assert repositories.list_context_update_suggestions(save.id) == []

    jobs = repositories.connection.execute(
        "SELECT status FROM jobs WHERE type = 'context_update'"
    ).fetchall()
    assert [row["status"] for row in jobs] == ["succeeded"]


def test_update_after_turn_canonicalizes_location_and_thread_variants(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    existing_location = repositories.add_location(
        save_id=save.id,
        name="cafe",
        source_message_id=narrator_message.id,
    )
    existing_thread = repositories.add_active_thread(
        save_id=save.id,
        title="ivys_conditions_for_movement",
        source_message_id=narrator_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        locations=(
            module.ExtractedLocation(
                name="Ivy's cafe",
                source_message_id=narrator_message.id,
                description="Warm lights and a counter near the front window.",
            ),
        ),
        active_threads=(
            module.ExtractedActiveThread(
                title="ivy_s_conditions_for_movement",
                source_message_id=narrator_message.id,
                description="Ivy only moves when Leo gives her room.",
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(narrator_message.id,),
        )
    )

    locations = repositories.list_locations(save.id)
    threads = repositories.list_active_threads(save.id)
    assert [location.id for location in locations] == [existing_location.id]
    assert locations[0].aliases == ["Ivy's cafe"]
    assert locations[0].description == (
        "Warm lights and a counter near the front window."
    )
    assert [thread.id for thread in threads] == [existing_thread.id]
    assert threads[0].description == "Ivy only moves when Leo gives her room."


def test_update_after_turn_archives_thread_when_status_becomes_resolved(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    existing_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Dinner promise",
        description="Mara still owes Ilyra dinner after the beacon is safe.",
        status="active",
        priority=4,
        visibility="public",
        source_message_id=narrator_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        active_threads=(
            module.ExtractedActiveThread(
                title="Dinner promise",
                source_message_id=narrator_message.id,
                description="Mara thanked Ilyra and settled the dinner promise.",
                status="Completed",
                visibility="private_between",
                reason="The turn explicitly finished the promise.",
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(narrator_message.id,),
        )
    )

    assert _count(result, "active_threads") == 0
    assert repositories.get_active_thread(existing_thread.id) is None
    assert repositories.list_active_threads(save.id) == []


def test_update_after_turn_archives_preexisting_inactive_threads_before_extraction(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    stale_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Old steakhouse promise",
        description="This resolved dinner promise should not stay active.",
        status="resolved",
        priority=6,
        source_message_id=narrator_message.id,
    )
    extractor = RecordingContextUpdateExtractor(module.ContextUpdateExtraction())
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(narrator_message.id,),
        )
    )

    assert repositories.get_active_thread(stale_thread.id) is None
    assert repositories.list_active_threads(save.id) == []
    request = cast(Any, extractor.requests[0])
    assert request.active_threads == ()


def test_update_after_turn_normalizes_thread_status_and_visibility_values(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    extraction = module.ContextUpdateExtraction(
        active_threads=(
            module.ExtractedActiveThread(
                title="Keep Ivy moving",
                source_message_id=narrator_message.id,
                description="Ivy will only move while Leo gives her space.",
                status="open",
                priority=5,
                visibility="private to Avery and Ciara",
            ),
            module.ExtractedActiveThread(
                title="Watch the lens",
                source_message_id=narrator_message.id,
                description="The lens might crack if the notch slips.",
                status="in progress",
                priority=6,
                visibility="active",
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(narrator_message.id,),
        )
    )

    threads = {
        thread.title: thread for thread in repositories.list_active_threads(save.id)
    }
    assert threads["Keep Ivy moving"].status == "active"
    assert threads["Keep Ivy moving"].visibility == "private"
    assert threads["Watch the lens"].status == "active"
    assert threads["Watch the lens"].visibility == "public"


def test_update_after_turn_archives_scene_local_threads_after_location_change(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    old_location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=narrator_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=old_location.id,
        situation="The lens ticks under stress.",
        source_message_id=narrator_message.id,
    )
    scene_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Brace the gallery ladder",
        description="The ladder matters only while everyone is in the gallery.",
        status="active",
        priority=3,
        visibility="scene",
        source_message_id=narrator_message.id,
    )
    durable_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Keep the beacon lit",
        description="The beacon must stay alive across scenes.",
        status="active",
        priority=7,
        visibility="public",
        source_message_id=narrator_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        scene=module.ExtractedSceneSnapshot(
            source_message_id=narrator_message.id,
            current_location_name="Gatehouse",
            situation="The gate winch is jammed.",
            reason="The completed turn moved the scene to the gatehouse.",
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(narrator_message.id,),
        )
    )

    assert repositories.get_active_thread(scene_thread.id) is None
    assert [thread.id for thread in repositories.list_active_threads(save.id)] == [
        durable_thread.id
    ]


def test_update_after_turn_archives_aggregate_open_thread_state_when_threads_exist(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    save, _player_message, narrator_message = _save_with_completed_turn(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="interaction.open_threads",
        value={"dinner": "Mara owes Ilyra dinner after the beacon is safe."},
        category="open_threads",
        source_message_id=narrator_message.id,
    )
    extraction = module.ContextUpdateExtraction(
        active_threads=(
            module.ExtractedActiveThread(
                title="Dinner promise",
                source_message_id=narrator_message.id,
                description="Mara still owes Ilyra dinner after the beacon is safe.",
                priority=4,
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=repositories,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    asyncio.run(
        service.update_after_turn(
            save_id=save.id,
            source_message_ids=(narrator_message.id,),
        )
    )

    assert [thread.title for thread in repositories.list_active_threads(save.id)] == [
        "Dinner promise"
    ]
    assert [
        state.key for state in repositories.list_world_state(save.id)
    ] == []


def test_apply_extraction_uses_bounded_world_data_snapshot(
    repositories: PersistenceRepositories,
) -> None:
    module = _context_update_module()
    counting = CountingPersistenceRepositories(repositories.connection)
    save, _player_message, narrator_message = _save_with_completed_turn(counting)
    extraction = module.ContextUpdateExtraction(
        locations=(
            module.ExtractedLocation(
                name="Beacon Gallery",
                source_message_id=narrator_message.id,
                evidence_quote="beacon gallery",
            ),
            module.ExtractedLocation(
                name="Lens Stair",
                parent_location_name="Beacon Gallery",
                source_message_id=narrator_message.id,
                evidence_quote="beacon gallery",
            ),
        ),
        characters=(
            module.ExtractedCharacter(
                name="Captain Ilyra",
                location_name="Beacon Gallery",
                source_message_id=narrator_message.id,
                evidence_quote="Captain Ilyra",
            ),
            module.ExtractedCharacter(
                name="Orin",
                location_name="Lens Stair",
                source_message_id=narrator_message.id,
                evidence_quote="beacon gallery",
            ),
        ),
        entity_links=(
            module.ExtractedEntityLink(
                entity_type="character",
                entity_name="Captain Ilyra",
                target_type="character",
                target_name="Orin",
                relation="knows",
                source_message_id=narrator_message.id,
                evidence_quote="Captain Ilyra",
            ),
        ),
    )
    service = module.ContextUpdateService(
        repositories=counting,
        extractor=RecordingContextUpdateExtractor(extraction),
    )

    result = service.apply_extraction(
        save_id=save.id,
        extraction=extraction,
        allowed_source_message_ids=(narrator_message.id,),
    )

    assert _count(result, "locations") == 2
    assert _count(result, "characters") == 2
    assert _count(result, "entity_links") == 1
    assert counting.list_counts == {
        "locations": 1,
        "characters": 1,
        "active_threads": 1,
        "entity_links": 1,
        "memories": 1,
        "world_state": 1,
        "summaries": 1,
    }


def _context_update_module() -> Any:
    return importlib.import_module("bragi.services.context_update_service")


def _tool_call_user_body(request: ToolCallRequest) -> str:
    return "\n\n".join(
        message.body
        for message in request.messages
        if message.role == "user"
    )


def _save_with_completed_turn(
    repositories: PersistenceRepositories,
) -> tuple[SaveRecord, MessageRecord, MessageRecord]:
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
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Captain Ilyra steadies Mara in the beacon gallery.",
        provider="fake",
        model="fake-chat",
        token_estimate=9,
    )
    return save, player_message, narrator_message


def _count(result: object, name: str) -> int:
    aliases = {
        "audit_rows": "audit_entries",
        "queued_suggestions": "suggestions",
    }
    name = aliases.get(name, name)
    if isinstance(result, dict):
        if name in result:
            return int(result[name])
        counts = result.get("counts")
        if isinstance(counts, dict) and name in counts:
            return int(counts[name])
    if hasattr(result, name):
        value = getattr(result, name)
        if name == "scene_snapshot":
            return 1 if value is not None else 0
        if isinstance(value, tuple):
            return len(value)
        return int(value)
    if name == "updated_fields" and hasattr(result, "audit_entries"):
        result_with_audit = cast(Any, result)
        return sum(
            1
            for entry in result_with_audit.audit_entries
            if getattr(entry, "operation", None) == "updated"
        )
    singular_count_name = f"{name.removesuffix('s')}_count"
    if hasattr(result, singular_count_name):
        return int(getattr(result, singular_count_name))
    count_name = f"{name}_count"
    if hasattr(result, count_name):
        return int(getattr(result, count_name))
    counts = getattr(result, "counts", None)
    if isinstance(counts, dict) and name in counts:
        return int(counts[name])
    raise AssertionError(f"Missing count for {name!r} in {result!r}")
