"""Structured post-turn updates for normalized context registries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from re import IGNORECASE, findall, fullmatch, search, sub
from time import perf_counter
from typing import Any, Protocol, cast

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.content_rating_instructions import maximum_content_rating
from bragi.interaction_mode import InteractionMode
from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterRecord,
    ContextObservationRecord,
    ContextSourceRecord,
    ContextUpdateAuditRecord,
    ContextUpdateSuggestionRecord,
    EntityLinkRecord,
    LocationRecord,
    MemoryRecord,
    MessageRecord,
    SceneSnapshotRecord,
    StateChangeRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ProviderClient,
    ProviderToolCall,
    StructuredOutputProvider,
    StructuredOutputRequest,
    ToolCallMessage,
    ToolCallProvider,
    ToolCallRequest,
    ToolDefinition,
)
from bragi.providers.errors import (
    ProviderError,
    ProviderErrorCategory,
    provider_error_is_model_not_found,
)
from bragi.providers.structured_schema import normalize_strict_json_schema
from bragi.redaction import redact_text
from bragi.retry_policy import MODEL_OUTPUT_MAX_ATTEMPTS, configured_max_attempts
from bragi.services.active_thread_lifecycle import (
    ACTIVE_THREAD_STATUSES,
    ACTIVE_THREAD_VISIBILITIES,
    active_thread_is_prompt_visible,
    active_thread_is_scene_local,
    active_thread_status_is_open,
    archive_inactive_active_threads,
    normalize_active_thread_record,
    normalize_active_thread_status,
    normalize_active_thread_visibility,
)
from bragi.services.character_locks import (
    CHARACTER_AGENCY_FIELDS,
    character_field_is_locked,
)
from bragi.services.continuity_index_service import ContinuityIndexService
from bragi.services.evidence import quote_matches_source
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.maintenance_scheduler import provider_pressure_from_exception
from bragi.services.manual_confirmation import (
    manual_character_registry_confirmation_enabled,
    manual_state_change_confirmation_enabled,
)
from bragi.services.message_correction import (
    MessageCorrectionContext,
    correction_context_text,
)
from bragi.services.open_threads import archive_open_thread_aggregate_state
from bragi.services.openrouter_routing_settings import (
    request_with_openrouter_routing,
)
from bragi.services.phone_number_exchange import (
    PHONE_EXCHANGE_BOTH,
    PHONE_EXCHANGE_CHARACTER_HAS_PLAYER_NUMBER,
    PHONE_EXCHANGE_PLAYER_HAS_CHARACTER_NUMBER,
)
from bragi.services.phone_number_exchange import (
    infer_phone_number_exchanges as infer_phone_number_exchange_records,
)
from bragi.services.post_turn_inference import VerifiedPostTurnCoverage
from bragi.services.prompt_inspection import PromptInspectionStore
from bragi.services.provider_fallbacks import (
    provider_error_with_fallback_attempted,
    provider_error_with_fallback_skipped_reason,
    recover_tool_call_shape_with_structured_output,
    shape_switch_diagnostics,
    structured_output_with_fallback,
    tool_call_fallback_request,
    tool_call_fallback_skip_reason,
)
from bragi.services.request_budget import (
    budget_structured_output_request,
    budget_tool_call_request,
)
from bragi.services.scene_snapshot_locks import scene_snapshot_field_is_locked
from bragi.services.sexual_content_safety import is_fade_to_black_message
from bragi.services.tool_call_helpers import (
    accepted_tool_result,
    append_tool_feedback_messages,
    invalid_tool_result,
    parse_tool_arguments_json,
    validate_tool_arguments_shape,
)
from bragi.services.world_time_signals import (
    first_non_timer_24h_clock,
    has_world_time_advance_signal,
    text_without_timer_readout_clauses,
    timer_readout_without_clock_advance,
)
from bragi.world_time_model import (
    canonical_world_time_from_legacy,
    canonical_world_time_from_values,
    format_world_time_from_snapshot,
    legacy_world_time_fields,
)

_CHARACTER_TITLE_WORDS = frozenset(
    {
        "admiral",
        "brother",
        "captain",
        "commander",
        "dame",
        "doctor",
        "dr",
        "father",
        "general",
        "inspector",
        "king",
        "lady",
        "lieutenant",
        "lord",
        "mother",
        "prince",
        "princess",
        "prof",
        "professor",
        "queen",
        "sergeant",
        "sir",
        "sister",
        "warden",
    }
)

_VAGUE_SCENE_TIME_VALUES = frozenset(
    {
        "later",
        "soon",
        "eventually",
        "unclear",
        "unknown",
        "unspecified",
        "some time later",
        "after a while",
        "a while later",
    }
)

_SCENE_TIME_PHRASES = (
    ("late morning", "late morning"),
    ("early morning", "morning"),
    ("morning", "morning"),
    ("breakfast", "morning"),
    ("sunrise", "morning"),
    ("dawn", "morning"),
    ("noon", "afternoon"),
    ("midday", "afternoon"),
    ("afternoon", "afternoon"),
    ("midnight", "night"),
    ("late night", "night"),
    ("night", "night"),
    ("dinner", "evening"),
    ("evening", "evening"),
    ("sunset", "evening"),
    ("dusk", "evening"),
)
_SCENE_WORLD_TIME_FIELDS = frozenset(
    {"in_world_time", "time_of_day", "day_of_week", "world_day_index"}
)

MAX_CONTEXT_UPDATE_CANDIDATES = 64
MAX_CONTEXT_UPDATE_SELECTIONS = 24
MAX_CONTEXT_UPDATE_IDENTITY_LOCATIONS = 80
MAX_CONTEXT_UPDATE_IDENTITY_CHARACTERS = 120
MAX_CONTEXT_UPDATE_IDENTITY_THREADS = 80
CHARACTER_HISTORY_SUMMARY_THRESHOLD_CHARS = 6000
MAX_SELECTED_PRIOR_CONTEXT_CHARS = 700
MAX_CONTEXT_UPDATE_CANDIDATE_BODY_CHARS = 420
MAX_CONTEXT_UPDATE_TOOL_FEEDBACK_TURNS = MODEL_OUTPUT_MAX_ATTEMPTS - 1
MAX_FOCUSED_SCENE_KNOWN_LOCATIONS = 24
MAX_FOCUSED_SCENE_KNOWN_CHARACTERS = 32
MAX_FOCUSED_SCENE_ACTIVE_THREADS = 12
MAX_FOCUSED_SCENE_RELATIONSHIP_CHARACTERS = 8
MAX_FOCUSED_SCENE_EMOTION_CHARACTERS = 8
MAX_FOCUSED_SCENE_CHARACTER_TOOL_CONCURRENCY = 4

_HIGH_VALUE_CONTEXT_FACT_TYPES = frozenset(
    {
        "character_voice",
        "identity",
        "inventory",
        "location",
        "open_obligation",
        "promise",
        "relationship",
    }
)
FOCUSED_SCENE_TOOL_RETRY_INSTRUCTION = (
    "Call the focused scene tool again with corrected arguments only."
)


@dataclass(frozen=True)
class ExtractedSceneSnapshot:
    source_message_id: str
    evidence_quote: str = ""
    current_location_name: str = ""
    situation: str = ""
    objective: str = ""
    in_world_time: str = ""
    weather: str = ""
    mood: str = ""
    nearby_objects: tuple[str, ...] | None = None
    hazards: tuple[str, ...] | None = None
    present_character_names: tuple[str, ...] | None = None
    scene_transition: bool = False
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedLocation:
    name: str
    source_message_id: str
    evidence_quote: str = ""
    aliases: tuple[str, ...] = ()
    description: str = ""
    visual_description: str = ""
    parent_location_name: str = ""
    connections: tuple[str, ...] = ()
    status: str = ""
    hazards: tuple[str, ...] = ()
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedCharacter:
    name: str
    source_message_id: str
    evidence_quote: str = ""
    aliases: tuple[str, ...] = ()
    role: str = ""
    age: str = ""
    known_state: str = ""
    met: bool | None = None
    appearance: str = ""
    visual_notes: str = ""
    current_clothing: str = ""
    personality: str = ""
    voice: str = ""
    relationships: dict[str, object] | None = None
    goals: str = ""
    motivations: str = ""
    current_intent: str = ""
    boundaries: str = ""
    attitude_toward_player: str = ""
    cooperation_conditions: str = ""
    status: str = ""
    location_name: str = ""
    private_notes: str = ""
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedActiveThread:
    title: str
    source_message_id: str
    evidence_quote: str = ""
    description: str = ""
    status: str = ""
    priority: int | None = None
    visibility: str = ""
    related_entities: tuple[str, ...] = ()
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedEntityLink:
    entity_type: str
    target_type: str
    source_message_id: str
    evidence_quote: str = ""
    entity_name: str = ""
    target_name: str = ""
    entity_id: str = ""
    target_id: str = ""
    relation: str = ""
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedPhoneNumberExchange:
    character_id: str
    direction: str
    source_message_id: str
    evidence_quote: str = ""
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class ContextUpdateExtraction:
    scene: ExtractedSceneSnapshot | None = None
    locations: tuple[ExtractedLocation, ...] = ()
    characters: tuple[ExtractedCharacter, ...] = ()
    active_threads: tuple[ExtractedActiveThread, ...] = ()
    entity_links: tuple[ExtractedEntityLink, ...] = ()
    phone_number_exchanges: tuple[ExtractedPhoneNumberExchange, ...] = ()
    tool_diagnostics: dict[str, object] = field(
        default_factory=dict,
        compare=False,
    )


def _filter_scene_for_verified_coverage(
    scene: ExtractedSceneSnapshot,
    *,
    coverage: VerifiedPostTurnCoverage,
    current_snapshot: SceneSnapshotRecord | None,
    characters: tuple[CharacterRecord, ...],
) -> ExtractedSceneSnapshot:
    covered_fields = coverage.scene_snapshot_fields
    present_character_names = scene.present_character_names
    covered_presence_ids = coverage.scene_presence_character_ids
    if covered_presence_ids and (
        scene.present_character_names is not None
        or scene.current_location_name.strip()
    ):
        present_names: list[str] = []
        for name in scene.present_character_names or ():
            resolution = _resolve_character(characters, name)
            if (
                resolution.record is not None
                and resolution.record.id in covered_presence_ids
            ):
                continue
            present_names.append(name)
        current_present_ids = (
            set(current_snapshot.present_character_ids) if current_snapshot else set()
        )
        present_names.extend(
            character.name
            for character in characters
            if character.id in covered_presence_ids
            and character.id in current_present_ids
        )
        present_character_names = tuple(dict.fromkeys(present_names))
    return replace(
        scene,
        situation="" if "situation" in covered_fields else scene.situation,
        objective="" if "objective" in covered_fields else scene.objective,
        in_world_time=(
            ""
            if covered_fields & {"in_world_time", "time_of_day", "day_of_week"}
            else scene.in_world_time
        ),
        weather="" if "weather" in covered_fields else scene.weather,
        mood="" if "mood" in covered_fields else scene.mood,
        nearby_objects=(
            None if "nearby_objects" in covered_fields else scene.nearby_objects
        ),
        hazards=None if "hazards" in covered_fields else scene.hazards,
        present_character_names=present_character_names,
    )


def _filter_extraction_for_verified_coverage(
    extraction: ContextUpdateExtraction,
    *,
    coverage: VerifiedPostTurnCoverage | None,
    request: ContextUpdateRequest,
) -> ContextUpdateExtraction:
    if coverage is None or coverage.empty or extraction.scene is None:
        return extraction
    return replace(
        extraction,
        scene=_filter_scene_for_verified_coverage(
            extraction.scene,
            coverage=coverage,
            current_snapshot=request.scene_snapshot,
            characters=request.characters,
        ),
    )


@dataclass(frozen=True)
class ContextRegistryItem:
    context_source_id: str
    source_type: str
    source_id: str
    title: str
    body: str
    fact_type: str
    importance: float
    source_message_ids: tuple[str, ...] = ()
    relevance_note: str = ""


@dataclass(frozen=True)
class ContextRegistrySelection:
    selected_items: tuple[ContextRegistryItem, ...] = ()
    fallback_used: bool = False


@dataclass(frozen=True)
class ContextRegistrySelectionRequest:
    save_id: str
    messages: tuple[MessageRecord, ...]
    scene_snapshot: SceneSnapshotRecord | None
    locations: tuple[LocationRecord, ...]
    characters: tuple[CharacterRecord, ...]
    active_threads: tuple[ActiveThreadRecord, ...]
    candidates: tuple[ContextRegistryItem, ...]


@dataclass(frozen=True)
class ContextUpdateRequest:
    save_id: str
    messages: tuple[MessageRecord, ...]
    scene_snapshot: SceneSnapshotRecord | None
    locations: tuple[LocationRecord, ...]
    characters: tuple[CharacterRecord, ...]
    active_threads: tuple[ActiveThreadRecord, ...]
    entity_links: tuple[EntityLinkRecord, ...]
    memories: tuple[MemoryRecord, ...] = ()
    world_state: tuple[WorldStateRecord, ...] = ()
    summaries: tuple[SummaryRecord, ...] = ()
    prior_context: tuple[ContextRegistryItem, ...] = ()
    correction_context: MessageCorrectionContext | None = None


@dataclass(frozen=True)
class _ContextUpdateReadSnapshot:
    all_messages: tuple[MessageRecord, ...]
    messages: tuple[MessageRecord, ...]
    scene_snapshot: SceneSnapshotRecord | None
    locations: tuple[LocationRecord, ...]
    characters: tuple[CharacterRecord, ...]
    active_threads: tuple[ActiveThreadRecord, ...]
    entity_links: tuple[EntityLinkRecord, ...]
    memories: tuple[MemoryRecord, ...]
    world_state: tuple[WorldStateRecord, ...]
    summaries: tuple[SummaryRecord, ...]
    context_sources: tuple[ContextSourceRecord, ...]
    context_observations: tuple[ContextObservationRecord, ...]


@dataclass(frozen=True)
class ExtractedFocusedCharacterEmotion:
    character_id: str
    emotional_state: str
    source_message_id: str
    evidence_quote: str = ""
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedFocusedActiveThreadStatus:
    active_thread_id: str
    status: str
    source_message_id: str
    evidence_quote: str = ""
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedFocusedCharacterRelationship:
    character_id: str
    target_name: str
    posture: str
    source_message_id: str
    evidence_quote: str = ""
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class FocusedSceneMaintenanceRequest:
    save_id: str
    messages: tuple[MessageRecord, ...]
    scene_snapshot: SceneSnapshotRecord
    locations: tuple[LocationRecord, ...]
    characters: tuple[CharacterRecord, ...]
    active_threads: tuple[ActiveThreadRecord, ...] = ()
    world_state: tuple[WorldStateRecord, ...] = ()


@dataclass(frozen=True)
class FocusedSceneMaintenance:
    scene_updates: tuple[ExtractedSceneSnapshot, ...] = ()
    active_thread_updates: tuple[ExtractedFocusedActiveThreadStatus, ...] = ()
    character_relationships: tuple[ExtractedFocusedCharacterRelationship, ...] = ()
    character_emotions: tuple[ExtractedFocusedCharacterEmotion, ...] = ()
    tool_diagnostics: dict[str, object] = field(
        default_factory=dict,
        compare=False,
    )


@dataclass(frozen=True)
class AppliedContextUpdate:
    scene_snapshot: SceneSnapshotRecord | None
    locations: tuple[LocationRecord, ...]
    characters: tuple[CharacterRecord, ...]
    active_threads: tuple[ActiveThreadRecord, ...]
    entity_links: tuple[EntityLinkRecord, ...]
    suggestions: tuple[ContextUpdateSuggestionRecord, ...]
    audit_entries: tuple[ContextUpdateAuditRecord, ...]
    job_result: dict[str, object] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class AppliedFocusedSceneMaintenance:
    scene_snapshot: SceneSnapshotRecord | None = None
    characters: tuple[CharacterRecord, ...] = ()
    active_threads: tuple[ActiveThreadRecord, ...] = ()
    world_state: tuple[WorldStateRecord, ...] = ()
    suggestions: tuple[ContextUpdateSuggestionRecord, ...] = ()
    audit_entries: tuple[ContextUpdateAuditRecord, ...] = ()
    state_changes: tuple[StateChangeRecord, ...] = ()
    tool_diagnostics: dict[str, object] = field(
        default_factory=dict,
        compare=False,
    )
    provider_pressure: dict[str, object] | None = field(
        default=None,
        compare=False,
    )

    @property
    def empty(self) -> bool:
        return not (
            self.scene_snapshot
            or self.characters
            or self.active_threads
            or self.world_state
            or self.suggestions
            or self.audit_entries
            or self.state_changes
            or self.tool_diagnostics
            or self.provider_pressure
        )

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "scene_snapshot_updated": self.scene_snapshot is not None,
            "character_count": len(self.characters),
            "active_thread_count": len(self.active_threads),
            "world_state_count": len(self.world_state),
            "suggestion_count": len(self.suggestions),
            "audit_count": len(self.audit_entries),
            "state_change_count": len(self.state_changes),
        }
        if self.tool_diagnostics:
            result["tool_diagnostics"] = self.tool_diagnostics
        if self.provider_pressure is not None:
            result["provider_pressure"] = self.provider_pressure
        return result


def _load_context_update_read_snapshot(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    source_message_ids: tuple[str, ...],
) -> _ContextUpdateReadSnapshot:
    source_ids = set(source_message_ids)
    all_messages = tuple(repositories.list_messages(save_id))
    return _ContextUpdateReadSnapshot(
        all_messages=all_messages,
        messages=tuple(message for message in all_messages if message.id in source_ids),
        scene_snapshot=repositories.get_scene_snapshot(save_id),
        locations=tuple(repositories.list_locations(save_id)),
        characters=tuple(repositories.list_characters(save_id)),
        active_threads=tuple(
            thread
            for thread in repositories.list_active_threads(save_id)
            if active_thread_is_prompt_visible(thread)
        ),
        entity_links=tuple(repositories.list_entity_links(save_id)),
        memories=tuple(repositories.list_memories(save_id)),
        world_state=tuple(repositories.list_world_state(save_id)),
        summaries=tuple(repositories.list_summaries(save_id)),
        context_sources=tuple(repositories.list_context_sources(save_id)),
        context_observations=tuple(repositories.list_context_observations(save_id)),
    )


@dataclass(frozen=True)
class _CharacterResolution:
    record: CharacterRecord | None = None
    ambiguous: bool = False


@dataclass(frozen=True)
class _ValidatedContextToolCall:
    arguments: dict[str, object]
    extraction: object


class ContextUpdateExtractor(Protocol):
    async def extract(
        self, request: ContextUpdateRequest
    ) -> ContextUpdateExtraction: ...


class ContextRegistrySelector(Protocol):
    async def select_context(
        self, request: ContextRegistrySelectionRequest
    ) -> ContextRegistrySelection: ...


class FocusedSceneMaintainer(Protocol):
    async def maintain(
        self, request: FocusedSceneMaintenanceRequest
    ) -> FocusedSceneMaintenance: ...


@dataclass(frozen=True)
class LocationWorldDataEnrichment:
    location_id: str
    source_message_id: str
    evidence_quote: str = ""
    description: str = ""
    visual_description: str = ""
    connections: tuple[str, ...] = ()
    status: str = ""
    hazards: tuple[str, ...] = ()
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class ActiveThreadWorldDataEnrichment:
    active_thread_id: str
    source_message_id: str
    evidence_quote: str = ""
    description: str = ""
    related_entities: tuple[str, ...] = ()
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class CharacterWorldDataEnrichment:
    character_id: str
    source_message_id: str
    evidence_quote: str = ""
    aliases: tuple[str, ...] = ()
    role: str = ""
    age: str = ""
    known_state: str = ""
    appearance: str = ""
    visual_notes: str = ""
    current_clothing: str = ""
    personality: str = ""
    voice: str = ""
    relationships: dict[str, object] | None = None
    status: str = ""
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class WorldDataEnrichment:
    locations: tuple[LocationWorldDataEnrichment, ...] = ()
    active_threads: tuple[ActiveThreadWorldDataEnrichment, ...] = ()
    characters: tuple[CharacterWorldDataEnrichment, ...] = ()


@dataclass(frozen=True)
class WorldDataEnrichmentRequest:
    save_id: str
    messages: tuple[MessageRecord, ...]
    scenario_context: str
    scene_snapshot: SceneSnapshotRecord | None
    locations: tuple[LocationRecord, ...]
    active_threads: tuple[ActiveThreadRecord, ...]
    characters: tuple[CharacterRecord, ...] = ()
    memories: tuple[MemoryRecord, ...] = ()
    world_state: tuple[WorldStateRecord, ...] = ()
    summaries: tuple[SummaryRecord, ...] = ()
    prior_context: tuple[ContextRegistryItem, ...] = ()


class WorldDataEnricher(Protocol):
    async def enrich(
        self, request: WorldDataEnrichmentRequest
    ) -> WorldDataEnrichment: ...


class StructuredProviderContextUpdater:
    def __init__(
        self,
        *,
        provider: StructuredOutputProvider,
        provider_name: str,
        model_id: str,
        repositories: PersistenceRepositories | None = None,
        providers: dict[str, ProviderClient] | None = None,
        prompt_inspection_store: PromptInspectionStore | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.repositories = repositories
        self.providers = providers
        self.prompt_inspection_store = prompt_inspection_store

    async def extract(self, request: ContextUpdateRequest) -> ContextUpdateExtraction:
        structured_request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="context_update_extraction",
                schema=_context_update_schema(request.messages),
                messages=_context_update_messages(request),
                temperature=0.0,
            ),
            task="context_update",
            save_id=request.save_id,
        )
        self._capture_structured_request(
            message_id=_prompt_inspection_message_id(request.messages),
            kind="context_extraction",
            title="Context extraction",
            request=structured_request,
        )
        if self.repositories is not None and self.providers is not None:
            response = await structured_output_with_fallback(
                repositories=self.repositories,
                providers=self.providers,
                request=structured_request,
                task="context_update",
                save_id=request.save_id,
            )
        else:
            response = await self.provider.generate_structured_output(
                budget_structured_output_request(
                    self.repositories,
                    structured_request,
                    task="context_update",
                )
            )
        extraction = context_update_extraction_from_structured_data(response.data)
        return _filter_structured_extraction_evidence(
            extraction,
            source_messages_by_id={message.id: message for message in request.messages},
        )

    async def select_context(
        self,
        request: ContextRegistrySelectionRequest,
    ) -> ContextRegistrySelection:
        structured_request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="context_update_context_selection",
                schema=_context_registry_selection_schema(request.candidates),
                messages=_context_registry_selection_messages(request),
                temperature=0.0,
            ),
            task="context_update",
            save_id=request.save_id,
        )
        self._capture_structured_request(
            message_id=_prompt_inspection_message_id(request.messages),
            kind="context_selection",
            title="Context selection",
            request=structured_request,
        )
        if self.repositories is not None and self.providers is not None:
            response = await structured_output_with_fallback(
                repositories=self.repositories,
                providers=self.providers,
                request=structured_request,
                task="context_update",
                save_id=request.save_id,
            )
        else:
            response = await self.provider.generate_structured_output(
                budget_structured_output_request(
                    self.repositories,
                    structured_request,
                    task="context_update",
                )
            )
        return _context_registry_selection_from_structured_data(
            response.data,
            candidates=request.candidates,
        )

    async def enrich(
        self, request: WorldDataEnrichmentRequest
    ) -> WorldDataEnrichment:
        structured_request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="world_data_enrichment",
                schema=_world_data_enrichment_schema(request),
                messages=_world_data_enrichment_messages(request),
                temperature=0.35,
            ),
            task="context_update",
            save_id=request.save_id,
        )
        self._capture_structured_request(
            message_id=_prompt_inspection_message_id(request.messages),
            kind="context_enrichment",
            title="Context enrichment",
            request=structured_request,
        )
        if self.repositories is not None and self.providers is not None:
            response = await structured_output_with_fallback(
                repositories=self.repositories,
                providers=self.providers,
                request=structured_request,
                task="context_update",
                save_id=request.save_id,
            )
        else:
            response = await self.provider.generate_structured_output(
                budget_structured_output_request(
                    self.repositories,
                    structured_request,
                    task="context_update",
                )
            )
        return _filter_world_data_enrichment_evidence(
            world_data_enrichment_from_structured_data(response.data),
            source_messages_by_id={
                message.id: message for message in request.messages
            },
        )

    def _capture_structured_request(
        self,
        *,
        message_id: str | None,
        kind: str,
        title: str,
        request: StructuredOutputRequest,
    ) -> None:
        if self.prompt_inspection_store is None or message_id is None:
            return
        self.prompt_inspection_store.capture_structured_request(
            message_id=message_id,
            kind=kind,
            title=title,
            request=request,
        )


class ToolCallingProviderContextUpdater:
    def __init__(
        self,
        *,
        provider: ToolCallProvider,
        provider_name: str,
        model_id: str,
        repositories: PersistenceRepositories | None = None,
        providers: dict[str, ProviderClient] | None = None,
        prompt_inspection_store: PromptInspectionStore | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.repositories = repositories
        self.providers = providers
        self.prompt_inspection_store = prompt_inspection_store

    def _structured_extraction_run(
        self,
        request: ContextUpdateRequest,
    ) -> Callable[[], Awaitable[ContextUpdateExtraction]]:
        async def run() -> ContextUpdateExtraction:
            extraction = await self._structured_updater().extract(request)
            return replace(
                extraction,
                tool_diagnostics=shape_switch_diagnostics(
                    provider=self.provider_name,
                    model_id=self.model_id,
                ),
            )

        return run

    async def extract(self, request: ContextUpdateRequest) -> ContextUpdateExtraction:
        tool_request = request_with_openrouter_routing(
            self.repositories,
            ToolCallRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                messages=_context_update_tool_messages(request),
                tools=_context_update_tool_definitions(request.messages),
                temperature=0.0,
            ),
            task="context_update",
            save_id=request.save_id,
        )
        self._capture_tool_call_request(
            message_id=_prompt_inspection_message_id(request.messages),
            kind="context_extraction_tool_calls",
            title="Context extraction tool calls",
            request=tool_request,
        )
        try:
            return await self._extract_with_provider(
                provider=self.provider,
                request=tool_request,
                source_messages=request.messages,
                allowed_source_message_ids=tuple(
                    message.id for message in request.messages
                ),
            )
        except ProviderError as exc:
            if self.repositories is None or self.providers is None:
                recovered = await self._recover_via_structured_shape(
                    error=exc,
                    structured_run=self._structured_extraction_run(request),
                )
                if recovered is not None:
                    return recovered
                raise
            fallback_request = tool_call_fallback_request(
                repositories=self.repositories,
                providers=self.providers,
                request=tool_request,
                save_id=request.save_id,
            )
            if fallback_request is None:
                reason = tool_call_fallback_skip_reason(
                    repositories=self.repositories,
                    providers=self.providers,
                    save_id=request.save_id,
                )
                log_event(
                    "provider.tool_call_fallback_skipped",
                    provider=tool_request.provider,
                    model=tool_request.model_id,
                    task="context_update",
                    reason=reason,
                )
                recovered = await self._recover_via_structured_shape(
                    error=exc,
                    structured_run=self._structured_extraction_run(request),
                )
                if recovered is not None:
                    return recovered
                raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
            fallback_provider = self.providers[fallback_request.provider]
            if not isinstance(fallback_provider, ToolCallProvider):
                reason = "fallback_provider_unavailable"
                log_event(
                    "provider.tool_call_fallback_skipped",
                    provider=tool_request.provider,
                    model=tool_request.model_id,
                    task="context_update",
                    reason=reason,
                )
                recovered = await self._recover_via_structured_shape(
                    error=exc,
                    structured_run=self._structured_extraction_run(request),
                )
                if recovered is not None:
                    return recovered
                raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
            log_event(
                "provider.tool_call_fallback_started",
                provider=fallback_request.provider,
                model=fallback_request.model_id,
                task="context_update",
            )
            try:
                fallback_extraction = await self._extract_with_provider(
                    provider=fallback_provider,
                    request=fallback_request,
                    source_messages=request.messages,
                    allowed_source_message_ids=tuple(
                        message.id for message in request.messages
                    ),
                    fallback_used=True,
                )
            except ProviderError as fallback_exc:
                # Recover when either tool attempt ended with model_not_found:
                # the tool shape is unavailable regardless of which attempt
                # reported it, so hand the model_not_found error to the
                # structured-output route.
                recovery_error = (
                    exc if provider_error_is_model_not_found(exc) else fallback_exc
                )
                recovered = await self._recover_via_structured_shape(
                    error=recovery_error,
                    structured_run=self._structured_extraction_run(request),
                )
                if recovered is not None:
                    return recovered
                raise provider_error_with_fallback_attempted(
                    fallback_exc,
                    provider=fallback_request.provider,
                    model_id=fallback_request.model_id,
                ) from fallback_exc
            primary_diagnostics = _tool_diagnostics_from_exception(exc)
            if primary_diagnostics:
                return replace(
                    fallback_extraction,
                    tool_diagnostics=_merge_tool_diagnostics(
                        primary_diagnostics,
                        fallback_extraction.tool_diagnostics,
                        fallback_used=True,
                    ),
                )
            return fallback_extraction

    def _structured_updater(self) -> StructuredProviderContextUpdater:
        if not isinstance(self.provider, StructuredOutputProvider):
            raise ValueError("Context update provider lacks structured output")
        return StructuredProviderContextUpdater(
            provider=cast(StructuredOutputProvider, self.provider),
            provider_name=self.provider_name,
            model_id=self.model_id,
            repositories=self.repositories,
            providers=self.providers,
            prompt_inspection_store=self.prompt_inspection_store,
        )

    async def _recover_via_structured_shape[T](
        self,
        *,
        error: ProviderError,
        structured_run: Callable[[], Awaitable[T]] | None,
    ) -> T | None:
        if (
            structured_run is None
            or not provider_error_is_model_not_found(error)
        ):
            return None
        return await recover_tool_call_shape_with_structured_output(
            error=error,
            task="context_update",
            provider=self.provider_name,
            model_id=self.model_id,
            structured_run=structured_run,
        )

    async def select_context(
        self,
        request: ContextRegistrySelectionRequest,
    ) -> ContextRegistrySelection:
        tool_request = request_with_openrouter_routing(
            self.repositories,
            ToolCallRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                messages=_context_registry_selection_tool_messages(request),
                tools=_context_registry_selection_tool_definitions(request.candidates),
                temperature=0.0,
            ),
            task="context_update",
            save_id=request.save_id,
        )
        self._capture_tool_call_request(
            message_id=_prompt_inspection_message_id(request.messages),
            kind="context_selection_tool_calls",
            title="Context selection tool calls",
            request=tool_request,
        )
        return await self._run_with_tool_fallback(
            request=tool_request,
            save_id=request.save_id,
            task="context_update",
            run=lambda provider, tool_request, _fallback_used: (
                self._select_context_with_provider(
                    provider=provider,
                    request=tool_request,
                    candidates=request.candidates,
                )
            ),
            structured_run=lambda: self._structured_updater().select_context(
                request
            ),
        )

    async def enrich(
        self, request: WorldDataEnrichmentRequest
    ) -> WorldDataEnrichment:
        tool_request = request_with_openrouter_routing(
            self.repositories,
            ToolCallRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                messages=_world_data_enrichment_tool_messages(request),
                tools=_world_data_enrichment_tool_definitions(request),
                temperature=0.35,
            ),
            task="context_update",
            save_id=request.save_id,
        )
        self._capture_tool_call_request(
            message_id=_prompt_inspection_message_id(request.messages),
            kind="context_enrichment_tool_calls",
            title="Context enrichment tool calls",
            request=tool_request,
        )
        return await self._run_with_tool_fallback(
            request=tool_request,
            save_id=request.save_id,
            task="context_update",
            run=lambda provider, tool_request, _fallback_used: (
                self._enrich_with_provider(
                    provider=provider,
                    request=tool_request,
                    source_messages=request.messages,
                    locations=request.locations,
                    active_threads=request.active_threads,
                    characters=request.characters,
                )
            ),
            structured_run=lambda: self._structured_updater().enrich(request),
        )

    async def _run_with_tool_fallback[T](
        self,
        *,
        request: ToolCallRequest,
        save_id: str,
        task: str,
        run: Callable[[ToolCallProvider, ToolCallRequest, bool], Awaitable[T]],
        structured_run: Callable[[], Awaitable[T]] | None = None,
    ) -> T:
        request = request_with_openrouter_routing(
            self.repositories,
            request,
            task=task,
            save_id=save_id,
        )
        try:
            return await run(self.provider, request, False)
        except ProviderError as exc:
            if self.repositories is None or self.providers is None:
                recovered = await self._recover_via_structured_shape(
                    error=exc,
                    structured_run=structured_run,
                )
                if recovered is not None:
                    return recovered
                raise
            fallback_request = tool_call_fallback_request(
                repositories=self.repositories,
                providers=self.providers,
                request=request,
                save_id=save_id,
            )
            if fallback_request is None:
                reason = tool_call_fallback_skip_reason(
                    repositories=self.repositories,
                    providers=self.providers,
                    save_id=save_id,
                )
                log_event(
                    "provider.tool_call_fallback_skipped",
                    provider=request.provider,
                    model=request.model_id,
                    task=task,
                    reason=reason,
                )
                recovered = await self._recover_via_structured_shape(
                    error=exc,
                    structured_run=structured_run,
                )
                if recovered is not None:
                    return recovered
                raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
            fallback_provider = self.providers[fallback_request.provider]
            if not isinstance(fallback_provider, ToolCallProvider):
                reason = "fallback_provider_unavailable"
                log_event(
                    "provider.tool_call_fallback_skipped",
                    provider=request.provider,
                    model=request.model_id,
                    task=task,
                    reason=reason,
                )
                recovered = await self._recover_via_structured_shape(
                    error=exc,
                    structured_run=structured_run,
                )
                if recovered is not None:
                    return recovered
                raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
            log_event(
                "provider.tool_call_fallback_started",
                provider=fallback_request.provider,
                model=fallback_request.model_id,
                task=task,
            )
            try:
                return await run(fallback_provider, fallback_request, True)
            except ProviderError as fallback_exc:
                # Recover when either tool attempt ended with model_not_found:
                # the tool shape is unavailable regardless of which attempt
                # reported it, so hand the model_not_found error to the
                # structured-output route.
                recovery_error = (
                    exc if provider_error_is_model_not_found(exc) else fallback_exc
                )
                recovered = await self._recover_via_structured_shape(
                    error=recovery_error,
                    structured_run=structured_run,
                )
                if recovered is not None:
                    return recovered
                raise provider_error_with_fallback_attempted(
                    fallback_exc,
                    provider=fallback_request.provider,
                    model_id=fallback_request.model_id,
                ) from fallback_exc

    def _capture_tool_call_request(
        self,
        *,
        message_id: str | None,
        kind: str,
        title: str,
        request: ToolCallRequest,
    ) -> None:
        if self.prompt_inspection_store is None or message_id is None:
            return
        self.prompt_inspection_store.capture_tool_call_request(
            message_id=message_id,
            kind=kind,
            title=title,
            request=request,
        )

    async def _select_context_with_provider(
        self,
        *,
        provider: ToolCallProvider,
        request: ToolCallRequest,
        candidates: tuple[ContextRegistryItem, ...],
    ) -> ContextRegistrySelection:
        messages = list(request.messages)
        tool_schemas = {tool.name: tool.parameters for tool in request.tools}
        candidates_by_id = {item.context_source_id: item for item in candidates}
        selected: list[ContextRegistryItem] = []
        selected_ids: set[str] = set()
        last_errors: list[str] = []
        max_attempt_count = configured_max_attempts(self.repositories)

        for _turn in range(max_attempt_count):
            turn_request = budget_tool_call_request(
                self.repositories,
                replace(request, messages=tuple(messages)),
                task="context_update",
            )
            response = await provider.generate_tool_calls(turn_request)
            errors: list[str] = []
            tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
            for call in response.tool_calls:
                accepted, result, item = _validate_context_selection_tool_call(
                    call,
                    tool_schemas=tool_schemas,
                    candidates_by_id=candidates_by_id,
                )
                if accepted:
                    if item is not None and item.context_source_id not in selected_ids:
                        selected_ids.add(item.context_source_id)
                        selected.append(item)
                    tool_results.append((call, _accepted_tool_result()))
                    continue
                errors.append(result["error"])
                tool_results.append((call, result))

            if not errors:
                return ContextRegistrySelection(selected_items=tuple(selected))

            last_errors = errors
            append_tool_feedback_messages(
                messages,
                assistant_body=response.body,
                tool_calls=response.tool_calls,
                tool_results=tool_results,
            )

        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message=(
                "Context selection tool-call validation failed after feedback: "
                + "; ".join(last_errors)
            ),
        )

    async def _enrich_with_provider(
        self,
        *,
        provider: ToolCallProvider,
        request: ToolCallRequest,
        source_messages: tuple[MessageRecord, ...],
        locations: tuple[LocationRecord, ...],
        active_threads: tuple[ActiveThreadRecord, ...],
        characters: tuple[CharacterRecord, ...],
    ) -> WorldDataEnrichment:
        messages = list(request.messages)
        tool_schemas = {tool.name: tool.parameters for tool in request.tools}
        source_messages_by_id = {message.id: message for message in source_messages}
        locations_by_id = {location.id: location for location in locations}
        threads_by_id = {thread.id: thread for thread in active_threads}
        characters_by_id = {character.id: character for character in characters}
        accepted_keys: set[tuple[str, str]] = set()
        location_enrichments: list[LocationWorldDataEnrichment] = []
        thread_enrichments: list[ActiveThreadWorldDataEnrichment] = []
        character_enrichments: list[CharacterWorldDataEnrichment] = []
        last_errors: list[str] = []
        max_attempt_count = configured_max_attempts(self.repositories)

        for _turn in range(max_attempt_count):
            turn_request = budget_tool_call_request(
                self.repositories,
                replace(request, messages=tuple(messages)),
                task="context_update",
            )
            response = await provider.generate_tool_calls(turn_request)
            errors: list[str] = []
            tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
            for call in response.tool_calls:
                accepted, result, enrichment = _validate_world_enrichment_tool_call(
                    call,
                    tool_schemas=tool_schemas,
                    source_messages_by_id=source_messages_by_id,
                    locations_by_id=locations_by_id,
                    threads_by_id=threads_by_id,
                    characters_by_id=characters_by_id,
                )
                if accepted:
                    key = (
                        call.name,
                        _canonical_tool_arguments(enrichment.arguments),
                    )
                    if key not in accepted_keys:
                        accepted_keys.add(key)
                        value = enrichment.extraction
                        if isinstance(value, LocationWorldDataEnrichment):
                            location_enrichments.append(value)
                        elif isinstance(value, ActiveThreadWorldDataEnrichment):
                            thread_enrichments.append(value)
                        elif isinstance(value, CharacterWorldDataEnrichment):
                            character_enrichments.append(value)
                    tool_results.append((call, _accepted_tool_result()))
                    continue
                errors.append(result["error"])
                tool_results.append((call, result))

            if not errors:
                return WorldDataEnrichment(
                    locations=tuple(location_enrichments),
                    active_threads=tuple(thread_enrichments),
                    characters=tuple(character_enrichments),
                )

            last_errors = errors
            append_tool_feedback_messages(
                messages,
                assistant_body=response.body,
                tool_calls=response.tool_calls,
                tool_results=tool_results,
            )

        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message=(
                "World-data enrichment tool-call validation failed after feedback: "
                + "; ".join(last_errors)
            ),
        )

    async def _extract_with_provider(
        self,
        *,
        provider: ToolCallProvider,
        request: ToolCallRequest,
        source_messages: tuple[MessageRecord, ...],
        allowed_source_message_ids: tuple[str, ...],
        fallback_used: bool = False,
    ) -> ContextUpdateExtraction:
        messages = list(request.messages)
        accepted_keys: set[tuple[str, str]] = set()
        scene: ExtractedSceneSnapshot | None = None
        locations: list[ExtractedLocation] = []
        characters: list[ExtractedCharacter] = []
        active_threads: list[ExtractedActiveThread] = []
        entity_links: list[ExtractedEntityLink] = []
        phone_number_exchanges: list[ExtractedPhoneNumberExchange] = []
        tool_schemas = {tool.name: tool.parameters for tool in request.tools}
        source_messages_by_id = {message.id: message for message in source_messages}
        last_errors: list[str] = []
        max_attempt_count = configured_max_attempts(self.repositories)
        diagnostics = _initial_tool_diagnostics(
            provider=request.provider,
            model_id=request.model_id,
            fallback_used=fallback_used,
        )

        for turn in range(max_attempt_count):
            turn_request = budget_tool_call_request(
                self.repositories,
                replace(request, messages=tuple(messages)),
                task="context_update",
            )
            response = await provider.generate_tool_calls(turn_request)
            errors: list[str] = []
            tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
            raw_calls = [_tool_call_diagnostic(call) for call in response.tool_calls]
            for call in response.tool_calls:
                accepted, result, extraction = _validate_context_update_tool_call(
                    call,
                    tool_schemas=tool_schemas,
                    source_messages_by_id=source_messages_by_id,
                    allowed_source_message_ids=allowed_source_message_ids,
                )
                if accepted:
                    key = (
                        call.name,
                        _canonical_tool_arguments(extraction.arguments),
                    )
                    if key not in accepted_keys:
                        accepted_keys.add(key)
                        extracted = extraction.extraction
                        if isinstance(extracted, ExtractedSceneSnapshot):
                            scene = extracted
                        elif isinstance(extracted, ExtractedLocation):
                            locations.append(extracted)
                        elif isinstance(extracted, ExtractedCharacter):
                            characters.append(extracted)
                        elif isinstance(extracted, ExtractedActiveThread):
                            active_threads.append(extracted)
                        elif isinstance(extracted, ExtractedEntityLink):
                            entity_links.append(extracted)
                        elif isinstance(extracted, ExtractedPhoneNumberExchange):
                            phone_number_exchanges.append(extracted)
                        _append_tool_diagnostic_call(
                            diagnostics,
                            "accepted_calls",
                            call,
                        )
                    tool_results.append((call, _accepted_tool_result()))
                    continue
                errors.append(result["error"])
                _append_tool_diagnostic_call(
                    diagnostics,
                    "rejected_calls",
                    call,
                    error=result["error"],
                )
                tool_results.append((call, result))

            _append_tool_diagnostic_turn(
                diagnostics,
                turn=turn,
                raw_calls=raw_calls,
                errors=errors,
            )
            if errors:
                log_event(
                    "context_update.tool_call_validation_failed",
                    provider=request.provider,
                    model=request.model_id,
                    turn=turn,
                    error_count=len(errors),
                )
            else:
                log_event(
                    "context_update.tool_call_validation_succeeded",
                    provider=request.provider,
                    model=request.model_id,
                    turn=turn,
                    accepted_call_count=len(response.tool_calls),
                )
            if not errors:
                return ContextUpdateExtraction(
                    scene=scene,
                    locations=tuple(locations),
                    characters=tuple(characters),
                    active_threads=tuple(active_threads),
                    entity_links=tuple(entity_links),
                    phone_number_exchanges=tuple(phone_number_exchanges),
                    tool_diagnostics=_final_tool_diagnostics(diagnostics),
                )

            last_errors = errors
            diagnostics["retry_count"] = turn + 1
            messages.append(
                ToolCallMessage(
                    role="assistant",
                    body=response.body,
                    tool_calls=response.tool_calls,
                )
            )
            for call, result in tool_results:
                messages.append(
                    ToolCallMessage(
                        role="tool",
                        body=json.dumps(result, sort_keys=True),
                        tool_call_id=call.id,
                    )
                )

        exc = ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message=(
                "Context update tool-call validation failed after feedback: "
                + "; ".join(last_errors)
            ),
        )
        _attach_tool_diagnostics(exc, _final_tool_diagnostics(diagnostics))
        raise exc


class ToolCallingFocusedSceneMaintainer:
    def __init__(
        self,
        *,
        provider: ToolCallProvider,
        provider_name: str,
        model_id: str,
        repositories: PersistenceRepositories | None = None,
        prompt_inspection_store: PromptInspectionStore | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.repositories = repositories
        self.prompt_inspection_store = prompt_inspection_store

    async def maintain(
        self, request: FocusedSceneMaintenanceRequest
    ) -> FocusedSceneMaintenance:
        diagnostics = _initial_tool_diagnostics(
            provider=self.provider_name,
            model_id=self.model_id,
            fallback_used=False,
        )
        scene_updates: list[ExtractedSceneSnapshot] = []
        active_thread_updates: list[ExtractedFocusedActiveThreadStatus] = []
        relationship_updates: list[ExtractedFocusedCharacterRelationship] = []
        emotions: list[ExtractedFocusedCharacterEmotion] = []
        semaphore = asyncio.Semaphore(MAX_FOCUSED_SCENE_CHARACTER_TOOL_CONCURRENCY)

        async def run_limited_focused_tool(tool_request: ToolCallRequest) -> object:
            async with semaphore:
                return await self._run_focused_tool(
                    request=request,
                    tool_request=tool_request,
                    diagnostics=diagnostics,
                )

        async def run_character_tools(
            tool_requests: list[ToolCallRequest],
        ) -> list[object]:
            if not tool_requests:
                return []
            results = await asyncio.gather(
                *[
                    asyncio.create_task(run_limited_focused_tool(tool_request))
                    for tool_request in tool_requests
                ],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return list(results)

        scene_tool_requests: list[tuple[str, ToolCallRequest]] = [
            (
                "time",
                _focused_scene_time_tool_request(
                    request,
                    provider_name=self.provider_name,
                    model_id=self.model_id,
                ),
            ),
            (
                "location_presence",
                _focused_scene_location_presence_tool_request(
                    request,
                    provider_name=self.provider_name,
                    model_id=self.model_id,
                ),
            ),
            (
                "surface",
                _focused_scene_surface_tool_request(
                    request,
                    provider_name=self.provider_name,
                    model_id=self.model_id,
                ),
            ),
        ]
        if _focused_scene_active_threads_for_prompt(request):
            scene_tool_requests.append(
                (
                    "thread",
                    _focused_scene_thread_status_tool_request(
                        request,
                        provider_name=self.provider_name,
                        model_id=self.model_id,
                    ),
                )
            )
        scene_tool_results = await asyncio.gather(
            *[
                asyncio.create_task(
                    self._run_focused_tool(
                        request=request,
                        tool_request=tool_request,
                        diagnostics=diagnostics,
                    )
                )
                for _kind, tool_request in scene_tool_requests
            ],
            return_exceptions=True,
        )
        for result in scene_tool_results:
            if isinstance(result, BaseException):
                raise result
        scene_results = {
            kind: result
            for (kind, _tool_request), result in zip(
                scene_tool_requests,
                scene_tool_results,
                strict=True,
            )
        }

        time_update = scene_results.get("time")
        if isinstance(time_update, ExtractedSceneSnapshot):
            scene_updates.append(time_update)

        location_presence_update = scene_results.get("location_presence")
        if isinstance(location_presence_update, ExtractedSceneSnapshot):
            scene_updates.append(location_presence_update)

        surface_update = scene_results.get("surface")
        if isinstance(surface_update, ExtractedSceneSnapshot):
            scene_updates.append(surface_update)

        thread_update = scene_results.get("thread")
        if isinstance(thread_update, ExtractedFocusedActiveThreadStatus):
            active_thread_updates.append(thread_update)

        location_presence_snapshot = (
            location_presence_update
            if isinstance(location_presence_update, ExtractedSceneSnapshot)
            else None
        )
        relationship_characters = _focused_scene_relationship_characters(
            request,
            location_presence_update=location_presence_snapshot,
        )
        emotion_characters = _focused_scene_emotion_characters(
            request,
            location_presence_update=location_presence_snapshot,
        )
        character_tool_kinds: list[str] = []
        character_tool_requests: list[ToolCallRequest] = []
        for character in relationship_characters:
            character_tool_kinds.append("relationship")
            character_tool_requests.append(
                _focused_character_relationship_tool_request(
                    request,
                    character=character,
                    provider_name=self.provider_name,
                    model_id=self.model_id,
                )
            )
        for character in emotion_characters:
            character_tool_kinds.append("emotion")
            character_tool_requests.append(
                _focused_character_emotion_tool_request(
                    request,
                    character=character,
                    provider_name=self.provider_name,
                    model_id=self.model_id,
                )
            )
        character_tool_results = await run_character_tools(character_tool_requests)
        for kind, tool_result in zip(
            character_tool_kinds,
            character_tool_results,
            strict=True,
        ):
            if (
                kind == "relationship"
                and isinstance(tool_result, ExtractedFocusedCharacterRelationship)
            ):
                relationship_updates.append(tool_result)
            elif kind == "emotion" and isinstance(
                tool_result,
                ExtractedFocusedCharacterEmotion,
            ):
                emotions.append(tool_result)

        return FocusedSceneMaintenance(
            scene_updates=tuple(scene_updates),
            active_thread_updates=tuple(active_thread_updates),
            character_relationships=tuple(relationship_updates),
            character_emotions=tuple(emotions),
            tool_diagnostics=_final_tool_diagnostics(diagnostics),
        )

    async def _run_focused_tool(
        self,
        *,
        request: FocusedSceneMaintenanceRequest,
        tool_request: ToolCallRequest,
        diagnostics: dict[str, object],
    ) -> object | None:
        tool_request = request_with_openrouter_routing(
            self.repositories,
            tool_request,
            task="context_update",
            save_id=request.save_id,
        )
        self._capture_tool_call_request(
            message_id=_prompt_inspection_message_id(request.messages),
            kind=f"focused_scene_{tool_request.tools[0].name}_tool_calls",
            title=f"Focused scene {tool_request.tools[0].name} tool calls",
            request=tool_request,
        )
        messages = list(tool_request.messages)
        source_messages_by_id = {message.id: message for message in request.messages}
        allowed_source_message_ids = tuple(message.id for message in request.messages)
        tool_schemas = {tool.name: tool.parameters for tool in tool_request.tools}
        accepted: object | None = None
        last_errors: list[str] = []
        max_attempt_count = configured_max_attempts(self.repositories)

        for turn in range(max_attempt_count):
            turn_request = budget_tool_call_request(
                self.repositories,
                replace(tool_request, messages=tuple(messages)),
                task="context_update",
            )
            response = await self.provider.generate_tool_calls(turn_request)
            errors: list[str] = []
            tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
            raw_calls = [_tool_call_diagnostic(call) for call in response.tool_calls]
            for call in response.tool_calls:
                valid, result, extracted = _validate_focused_scene_tool_call(
                    call,
                    tool_schemas=tool_schemas,
                    source_messages_by_id=source_messages_by_id,
                    allowed_source_message_ids=allowed_source_message_ids,
                    request=request,
                )
                if valid:
                    if accepted is None:
                        accepted = extracted.extraction
                        _append_tool_diagnostic_call(
                            diagnostics,
                            "accepted_calls",
                            call,
                        )
                    tool_results.append((call, _accepted_tool_result()))
                    continue
                errors.append(result["error"])
                _append_tool_diagnostic_call(
                    diagnostics,
                    "rejected_calls",
                    call,
                    error=result["error"],
                )
                tool_results.append((call, result))

            _append_tool_diagnostic_turn(
                diagnostics,
                turn=turn,
                raw_calls=raw_calls,
                errors=errors,
            )
            if not errors:
                return accepted

            last_errors = errors
            retry_count = diagnostics.get("retry_count", 0)
            diagnostics["retry_count"] = (
                retry_count if isinstance(retry_count, int) else 0
            ) + 1
            append_tool_feedback_messages(
                messages,
                assistant_body=response.body,
                tool_calls=response.tool_calls,
                tool_results=tool_results,
            )

        exc = ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message=(
                "Focused scene tool-call validation failed after feedback: "
                + "; ".join(last_errors)
            ),
        )
        _attach_tool_diagnostics(exc, _final_tool_diagnostics(diagnostics))
        raise exc

    def _capture_tool_call_request(
        self,
        *,
        message_id: str | None,
        kind: str,
        title: str,
        request: ToolCallRequest,
    ) -> None:
        if self.prompt_inspection_store is None or message_id is None:
            return
        self.prompt_inspection_store.capture_tool_call_request(
            message_id=message_id,
            kind=kind,
            title=title,
            request=request,
        )


def _focused_scene_time_tool_request(
    request: FocusedSceneMaintenanceRequest,
    *,
    provider_name: str,
    model_id: str,
) -> ToolCallRequest:
    return ToolCallRequest(
        provider=provider_name,
        model_id=model_id,
        messages=_focused_scene_tool_messages(
            system=(
                "You are Bragi's tiny scene time maintainer. Read the completed "
                "turn and the current scene time. If the completed turn directly "
                "establishes that the current scene time changed, call "
                "set_scene_time once. Otherwise make no tool calls. Do not infer "
                "vague time changes from mood, pacing, or filler narration."
            ),
            user=_focused_scene_request_text(request, include_known_registry=False),
        ),
        tools=(
            ToolDefinition(
                name="set_scene_time",
                description="Set the current scene time when directly supported.",
                parameters=_focused_scene_time_tool_schema(request.messages),
            ),
        ),
        temperature=0.0,
    )


def _focused_scene_location_presence_tool_request(
    request: FocusedSceneMaintenanceRequest,
    *,
    provider_name: str,
    model_id: str,
) -> ToolCallRequest:
    return ToolCallRequest(
        provider=provider_name,
        model_id=model_id,
        messages=_focused_scene_tool_messages(
            system=(
                "You are Bragi's tiny scene location and presence maintainer. "
                "If the completed turn clearly establishes the current location "
                "or complete current named character presence, or begins a "
                "distinct new scene even at the same location, call "
                "set_scene_location_presence once. Otherwise make no tool calls. "
                "For present_character_names, only provide a complete current "
                "list; omit it when unchanged or unclear. Set scene_transition "
                "only when the text directly establishes a new scene boundary."
            ),
            user=_focused_scene_request_text(request, include_known_registry=True),
        ),
        tools=(
            ToolDefinition(
                name="set_scene_location_presence",
                description=(
                    "Set current scene location and complete present character "
                    "names when directly supported."
                ),
                parameters=_focused_scene_location_presence_tool_schema(
                    request.messages
                ),
            ),
        ),
        temperature=0.0,
    )


def _focused_scene_surface_tool_request(
    request: FocusedSceneMaintenanceRequest,
    *,
    provider_name: str,
    model_id: str,
) -> ToolCallRequest:
    return ToolCallRequest(
        provider=provider_name,
        model_id=model_id,
        messages=_focused_scene_tool_messages(
            system=(
                "You are Bragi's tiny scene surface maintainer. If the "
                "completed turn clearly establishes current scene mood, "
                "atmosphere, weather, nearby objects or props, hazards, or "
                "immediate constraints, call set_scene_surface once. Otherwise "
                "make no tool calls. For nearby_objects and hazards, provide "
                "complete current lists only; omit them when unchanged or "
                "unclear."
            ),
            user="\n\n".join(
                (
                    _focused_scene_request_text(request),
                    (
                        "Maintain only scene.mood, scene.weather, "
                        "scene.nearby_objects, and scene.hazards."
                    ),
                )
            ),
        ),
        tools=(
            ToolDefinition(
                name="set_scene_surface",
                description=(
                    "Set current scene mood, weather, nearby objects, and "
                    "hazards when directly supported."
                ),
                parameters=_focused_scene_surface_tool_schema(request.messages),
            ),
        ),
        temperature=0.0,
    )


def _focused_scene_thread_status_tool_request(
    request: FocusedSceneMaintenanceRequest,
    *,
    provider_name: str,
    model_id: str,
) -> ToolCallRequest:
    return ToolCallRequest(
        provider=provider_name,
        model_id=model_id,
        messages=_focused_scene_tool_messages(
            system=(
                "You are Bragi's tiny scene-local thread lifecycle maintainer. "
                "If the completed turn directly establishes that one existing "
                "scene-local active thread changed lifecycle status, call "
                "set_scene_thread_status once. Otherwise make no tool calls. "
                "Never create a new thread or update broad world obligations."
            ),
            user="\n\n".join(
                (
                    _focused_scene_request_text(request),
                    _focused_scene_active_threads_text(request),
                )
            ),
        ),
        tools=(
            ToolDefinition(
                name="set_scene_thread_status",
                description=(
                    "Set the status for one existing scene-local active thread."
                ),
                parameters=_focused_scene_thread_status_tool_schema(request),
            ),
        ),
        temperature=0.0,
    )


def _focused_character_relationship_tool_request(
    request: FocusedSceneMaintenanceRequest,
    *,
    character: CharacterRecord,
    provider_name: str,
    model_id: str,
) -> ToolCallRequest:
    return ToolCallRequest(
        provider=provider_name,
        model_id=model_id,
        messages=_focused_scene_tool_messages(
            system=(
                "You are Bragi's tiny relationship posture maintainer. You are "
                f"assigned only to {character.name}. If the completed turn "
                "directly establishes this character's current relationship "
                "posture toward the player or another currently present "
                "character, call set_character_relationship_posture once. "
                "Otherwise make no tool calls. Keep the posture compact and "
                "scene-grounded."
            ),
            user="\n\n".join(
                (
                    _focused_scene_request_text(request),
                    _focused_character_relationship_context_text(
                        request,
                        character,
                    ),
                )
            ),
        ),
        tools=(
            ToolDefinition(
                name="set_character_relationship_posture",
                description=(
                    "Set one assigned character's current relationship posture."
                ),
                parameters=_focused_character_relationship_tool_schema(
                    request.messages,
                    character=character,
                ),
            ),
        ),
        temperature=0.0,
    )


def _focused_character_emotion_tool_request(
    request: FocusedSceneMaintenanceRequest,
    *,
    character: CharacterRecord,
    provider_name: str,
    model_id: str,
) -> ToolCallRequest:
    return ToolCallRequest(
        provider=provider_name,
        model_id=model_id,
        messages=_focused_scene_tool_messages(
            system=(
                "You are Bragi's tiny character emotion maintainer. You are "
                f"assigned only to {character.name}. If the completed turn "
                "directly establishes this character's current emotional state "
                "changed, call set_character_emotion once. Otherwise make no "
                "tool calls. Prefer compact scene-current emotional posture, "
                "not durable personality or relationship facts."
            ),
            user="\n\n".join(
                (
                    _focused_scene_request_text(request),
                    _focused_character_emotion_context_text(request, character),
                )
            ),
        ),
        tools=(
            ToolDefinition(
                name="set_character_emotion",
                description="Set one assigned character's current emotional state.",
                parameters=_focused_character_emotion_tool_schema(
                    request.messages,
                    character=character,
                ),
            ),
        ),
        temperature=0.0,
    )


def _focused_scene_tool_messages(
    *,
    system: str,
    user: str,
) -> tuple[ToolCallMessage, ...]:
    return (
        ToolCallMessage(role="system", body=system),
        ToolCallMessage(role="user", body=user),
    )


def _focused_scene_request_text(
    request: FocusedSceneMaintenanceRequest,
    *,
    include_known_registry: bool = False,
) -> str:
    character_names = {
        character.id: character.name for character in request.characters
    }
    current_location = next(
        (
            location.name
            for location in request.locations
            if location.id == request.scene_snapshot.current_location_id
        ),
        "unknown",
    )
    present_names = [
        character_names[character_id]
        for character_id in request.scene_snapshot.present_character_ids
        if character_id in character_names
    ]
    world_time = format_world_time_from_snapshot(request.scene_snapshot) or "unknown"
    sections = [
        "Current focused scene state:\n"
        f"- location: {current_location}\n"
        f"- time: {world_time}\n"
        f"- situation: {request.scene_snapshot.situation or 'unknown'}\n"
        f"- objective: {request.scene_snapshot.objective or 'unknown'}\n"
        f"- weather: {request.scene_snapshot.weather or 'unknown'}\n"
        f"- mood: {request.scene_snapshot.mood or 'unknown'}\n"
        "- nearby objects: "
        f"{', '.join(request.scene_snapshot.nearby_objects) or 'none'}\n"
        f"- hazards: {', '.join(request.scene_snapshot.hazards) or 'none'}\n"
        f"- present characters: {', '.join(present_names) or 'none'}",
    ]
    if include_known_registry:
        sections.extend(
            (
                _focused_scene_locations_text(request),
                _focused_scene_characters_text(request),
            )
        )
    sections.append(_messages_text(request.messages))
    return "\n\n".join(section for section in sections if section)


def _focused_scene_locations_text(request: FocusedSceneMaintenanceRequest) -> str:
    locations = _focused_scene_locations_for_prompt(request)
    return _focused_scene_registry_text(
        title="Known locations",
        shown=len(locations),
        total=len(request.locations),
        names=tuple(location.name for location in locations),
    )


def _focused_scene_characters_text(request: FocusedSceneMaintenanceRequest) -> str:
    characters = _focused_scene_characters_for_prompt(request)
    return _focused_scene_registry_text(
        title="Known characters",
        shown=len(characters),
        total=len(request.characters),
        names=tuple(character.name for character in characters),
    )


def _focused_scene_active_threads_text(
    request: FocusedSceneMaintenanceRequest,
) -> str:
    threads = _focused_scene_active_threads_for_prompt(request)
    total = len(_focused_scene_active_threads(request))
    omitted = max(0, total - len(threads))
    header = f"Known scene-local active threads (showing {len(threads)} of {total}"
    if omitted:
        header += f"; {omitted} omitted"
    header += "):"
    if not threads:
        return f"{header}\n- none"
    return header + "\n" + "\n".join(
        _focused_scene_active_thread_text(thread) for thread in threads
    )


def _focused_scene_active_thread_text(thread: ActiveThreadRecord) -> str:
    description = _compact_text(thread.description, 180) or "none"
    return (
        f"- id: {thread.id}; title: {thread.title}; "
        f"status: {normalize_active_thread_status(thread.status)}; "
        f"priority: {thread.priority}; description: {description}"
    )


def _focused_scene_registry_text(
    *,
    title: str,
    shown: int,
    total: int,
    names: tuple[str, ...],
) -> str:
    omitted = max(0, total - shown)
    header = f"{title} (showing {shown} of {total}"
    if omitted:
        header += f"; {omitted} omitted"
    header += "):"
    if not names:
        return f"{header}\n- none"
    return header + "\n" + "\n".join(f"- {name}" for name in names)


def _focused_scene_locations_for_prompt(
    request: FocusedSceneMaintenanceRequest,
) -> tuple[LocationRecord, ...]:
    current_id = request.scene_snapshot.current_location_id
    ordered = sorted(
        request.locations,
        key=lambda location: (
            0 if location.id == current_id else 1,
            location.name.casefold(),
            location.id,
        ),
    )
    return tuple(ordered[:MAX_FOCUSED_SCENE_KNOWN_LOCATIONS])


def _focused_scene_characters_for_prompt(
    request: FocusedSceneMaintenanceRequest,
) -> tuple[CharacterRecord, ...]:
    present = set(request.scene_snapshot.present_character_ids)
    ordered = sorted(
        request.characters,
        key=lambda character: (
            0 if character.id in present else 1,
            character.name.casefold(),
            character.id,
        ),
    )
    return tuple(ordered[:MAX_FOCUSED_SCENE_KNOWN_CHARACTERS])


def _focused_scene_active_threads(
    request: FocusedSceneMaintenanceRequest,
) -> tuple[ActiveThreadRecord, ...]:
    return tuple(
        thread
        for thread in request.active_threads
        if active_thread_status_is_open(thread.status)
        and active_thread_is_scene_local(thread)
    )


def _focused_scene_active_threads_for_prompt(
    request: FocusedSceneMaintenanceRequest,
) -> tuple[ActiveThreadRecord, ...]:
    ordered = sorted(
        _focused_scene_active_threads(request),
        key=lambda thread: (-thread.priority, thread.title.casefold(), thread.id),
    )
    return tuple(ordered[:MAX_FOCUSED_SCENE_ACTIVE_THREADS])


def _focused_character_relationship_context_text(
    request: FocusedSceneMaintenanceRequest,
    character: CharacterRecord,
) -> str:
    current = json.dumps(character.relationships, sort_keys=True)
    targets = _focused_relationship_target_names(request, character=character)
    return (
        f"Relationship subject: {character.name}\n"
        f"Current stored relationships: {current or '{}'}\n"
        f"Eligible targets: {', '.join(targets) or 'none'}"
    )


def _focused_character_emotion_context_text(
    request: FocusedSceneMaintenanceRequest,
    character: CharacterRecord,
) -> str:
    state_key = _character_emotion_state_key(character)
    current = next(
        (
            json.dumps(state.value, sort_keys=True)
            for state in request.world_state
            if state.key == state_key
        ),
        "",
    )
    return (
        f"Assigned character: {character.name}\n"
        f"Current stored emotional state: {current or 'unknown'}"
    )


def _focused_relationship_target_names(
    request: FocusedSceneMaintenanceRequest,
    *,
    character: CharacterRecord,
) -> tuple[str, ...]:
    names: list[str] = []
    player_name = _focused_scene_player_name(request.messages)
    if player_name:
        names.append(player_name)
    present_ids = set(request.scene_snapshot.present_character_ids)
    for candidate in request.characters:
        if candidate.id == character.id or candidate.id not in present_ids:
            continue
        names.append(candidate.name)
    return tuple(dict.fromkeys(_clean_strings(names)))


def _focused_scene_player_name(messages: tuple[MessageRecord, ...]) -> str:
    for message in messages:
        speaker_name = message.speaker_name or ""
        if message.role == "player" and speaker_name.strip():
            return speaker_name.strip()
    return "player"


def _focused_scene_time_tool_schema(
    messages: tuple[MessageRecord, ...],
) -> dict[str, object]:
    return _tool_schema(
        required=["in_world_time", "source_message_id", "evidence_quote"],
        properties={
            **_focused_scene_base_tool_properties(messages),
            "in_world_time": {"type": "string"},
        },
    )


def _focused_scene_location_presence_tool_schema(
    messages: tuple[MessageRecord, ...],
) -> dict[str, object]:
    return _tool_schema(
        required=["source_message_id", "evidence_quote"],
        properties={
            **_focused_scene_base_tool_properties(messages),
            "current_location_name": {"type": "string"},
            "present_character_names": _present_character_names_schema(),
            "scene_transition": {"type": "boolean"},
        },
    )


def _focused_scene_surface_tool_schema(
    messages: tuple[MessageRecord, ...],
) -> dict[str, object]:
    return _tool_schema(
        required=["source_message_id", "evidence_quote"],
        properties={
            **_focused_scene_base_tool_properties(messages),
            "weather": {"type": "string"},
            "mood": {"type": "string"},
            "nearby_objects": _current_scene_string_array_schema("nearby objects"),
            "hazards": _current_scene_string_array_schema("hazards"),
        },
    )


def _focused_scene_thread_status_tool_schema(
    request: FocusedSceneMaintenanceRequest,
) -> dict[str, object]:
    thread_id_schema: dict[str, object] = {"type": "string"}
    thread_ids = [
        thread.id for thread in _focused_scene_active_threads_for_prompt(request)
    ]
    if thread_ids:
        thread_id_schema["enum"] = thread_ids
    return _tool_schema(
        required=[
            "active_thread_id",
            "status",
            "source_message_id",
            "evidence_quote",
        ],
        properties={
            **_focused_scene_base_tool_properties(request.messages),
            "active_thread_id": thread_id_schema,
            "status": {"type": "string", "enum": sorted(ACTIVE_THREAD_STATUSES)},
        },
    )


def _focused_character_relationship_tool_schema(
    messages: tuple[MessageRecord, ...],
    *,
    character: CharacterRecord,
) -> dict[str, object]:
    return _tool_schema(
        required=[
            "character_id",
            "target_name",
            "posture",
            "source_message_id",
            "evidence_quote",
        ],
        properties={
            **_focused_scene_base_tool_properties(messages),
            "character_id": {"type": "string", "enum": [character.id]},
            "target_name": {"type": "string"},
            "posture": {"type": "string"},
        },
    )


def _focused_character_emotion_tool_schema(
    messages: tuple[MessageRecord, ...],
    *,
    character: CharacterRecord,
) -> dict[str, object]:
    return _tool_schema(
        required=[
            "character_id",
            "emotional_state",
            "source_message_id",
            "evidence_quote",
        ],
        properties={
            **_focused_scene_base_tool_properties(messages),
            "character_id": {"type": "string", "enum": [character.id]},
            "emotional_state": {"type": "string"},
        },
    )


def _focused_scene_base_tool_properties(
    messages: tuple[MessageRecord, ...],
) -> dict[str, object]:
    source_schema: dict[str, object] = {"type": "string"}
    message_ids = [message.id for message in messages]
    if message_ids:
        source_schema["enum"] = message_ids
    return {
        "source_message_id": source_schema,
        "evidence_quote": {
            "type": "string",
            "description": "Exact substring copied from the source message.",
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }


def _focused_scene_relationship_characters(
    request: FocusedSceneMaintenanceRequest,
    *,
    location_presence_update: ExtractedSceneSnapshot | None = None,
) -> tuple[CharacterRecord, ...]:
    characters_by_id = {character.id: character for character in request.characters}
    if (
        location_presence_update is not None
        and location_presence_update.present_character_names is not None
    ):
        resolved_ids = _focused_present_character_ids(
            request.characters,
            location_presence_update.present_character_names,
        )
        if resolved_ids is not None:
            return tuple(
                characters_by_id[character_id]
                for character_id in resolved_ids
                if character_id in characters_by_id
                and not characters_by_id[character_id].is_player_character
            )[:MAX_FOCUSED_SCENE_RELATIONSHIP_CHARACTERS]
    return tuple(
        characters_by_id[character_id]
        for character_id in request.scene_snapshot.present_character_ids
        if character_id in characters_by_id
        and not characters_by_id[character_id].is_player_character
    )[:MAX_FOCUSED_SCENE_RELATIONSHIP_CHARACTERS]


def _focused_scene_emotion_characters(
    request: FocusedSceneMaintenanceRequest,
    *,
    location_presence_update: ExtractedSceneSnapshot | None = None,
) -> tuple[CharacterRecord, ...]:
    characters_by_id = {character.id: character for character in request.characters}
    if (
        location_presence_update is not None
        and location_presence_update.present_character_names is not None
    ):
        resolved_ids = _focused_present_character_ids(
            request.characters,
            location_presence_update.present_character_names,
        )
        if resolved_ids is not None:
            return tuple(
                characters_by_id[character_id]
                for character_id in resolved_ids
                if character_id in characters_by_id
                and not characters_by_id[character_id].protected_from_maintenance
            )[:MAX_FOCUSED_SCENE_EMOTION_CHARACTERS]
    return tuple(
        characters_by_id[character_id]
        for character_id in request.scene_snapshot.present_character_ids
        if character_id in characters_by_id
        and not characters_by_id[character_id].protected_from_maintenance
    )[:MAX_FOCUSED_SCENE_EMOTION_CHARACTERS]


def _validate_focused_scene_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    source_messages_by_id: dict[str, MessageRecord],
    allowed_source_message_ids: tuple[str, ...],
    request: FocusedSceneMaintenanceRequest,
) -> tuple[bool, dict[str, str], _ValidatedContextToolCall]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_focused_scene_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_focused_scene_tool_call(
            parse_error or "Tool arguments must be a JSON object"
        )
    error = _validate_tool_arguments(
        arguments,
        schema=schema,
        source_messages_by_id=source_messages_by_id,
        allowed_source_message_ids=allowed_source_message_ids,
    )
    if error is not None:
        return _invalid_focused_scene_tool_call(error)
    extraction = _focused_scene_tool_call_extraction(
        call.name,
        arguments,
        request=request,
    )
    if extraction is None:
        return _invalid_focused_scene_tool_call(
            f"Unsupported focused scene tool: {call.name}"
        )
    return True, _accepted_tool_result(), _ValidatedContextToolCall(
        arguments=arguments,
        extraction=extraction,
    )


def _focused_scene_tool_call_extraction(
    tool_name: str,
    arguments: dict[str, object],
    *,
    request: FocusedSceneMaintenanceRequest,
) -> object | None:
    if tool_name == "set_scene_time":
        return ExtractedSceneSnapshot(
            source_message_id=str(arguments.get("source_message_id", "")),
            evidence_quote=str(arguments.get("evidence_quote", "")),
            in_world_time=str(arguments.get("in_world_time", "")).strip(),
            reason=str(arguments.get("reason", "")).strip(),
            confidence=_confidence(arguments.get("confidence")),
        )
    if tool_name == "set_scene_location_presence":
        return _scene_from_data(arguments)
    if tool_name == "set_scene_surface":
        return _scene_from_data(arguments)
    if tool_name == "set_scene_thread_status":
        active_thread_id = str(arguments.get("active_thread_id", "")).strip()
        status = normalize_active_thread_status(arguments.get("status"))
        if not active_thread_id:
            return None
        return ExtractedFocusedActiveThreadStatus(
            active_thread_id=active_thread_id,
            status=status,
            source_message_id=str(arguments.get("source_message_id", "")),
            evidence_quote=str(arguments.get("evidence_quote", "")),
            reason=str(arguments.get("reason", "")).strip(),
            confidence=_confidence(arguments.get("confidence")),
        )
    if tool_name == "set_character_relationship_posture":
        character_id = str(arguments.get("character_id", "")).strip()
        if character_id not in {character.id for character in request.characters}:
            return None
        target_name = str(arguments.get("target_name", "")).strip()
        posture = str(arguments.get("posture", "")).strip()
        if not target_name or not posture:
            return None
        return ExtractedFocusedCharacterRelationship(
            character_id=character_id,
            target_name=target_name,
            posture=posture,
            source_message_id=str(arguments.get("source_message_id", "")),
            evidence_quote=str(arguments.get("evidence_quote", "")),
            reason=str(arguments.get("reason", "")).strip(),
            confidence=_confidence(arguments.get("confidence")),
        )
    if tool_name == "set_character_emotion":
        character_id = str(arguments.get("character_id", "")).strip()
        if character_id not in {character.id for character in request.characters}:
            return None
        emotional_state = str(arguments.get("emotional_state", "")).strip()
        if not emotional_state:
            return None
        return ExtractedFocusedCharacterEmotion(
            character_id=character_id,
            emotional_state=emotional_state,
            source_message_id=str(arguments.get("source_message_id", "")),
            evidence_quote=str(arguments.get("evidence_quote", "")),
            reason=str(arguments.get("reason", "")).strip(),
            confidence=_confidence(arguments.get("confidence")),
        )
    return None


def _invalid_focused_scene_tool_call(
    error: str,
) -> tuple[bool, dict[str, str], _ValidatedContextToolCall]:
    return (
        False,
        invalid_tool_result(
            error,
            retry_instruction=FOCUSED_SCENE_TOOL_RETRY_INSTRUCTION,
        ),
        _ValidatedContextToolCall(arguments={}, extraction=None),
    )


def _validate_context_update_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    source_messages_by_id: dict[str, MessageRecord],
    allowed_source_message_ids: tuple[str, ...],
) -> tuple[bool, dict[str, str], _ValidatedContextToolCall]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_tool_call(
            f"Unknown tool name: {call.name}",
        )
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_tool_call(parse_error or "Tool arguments must be a JSON object")
    error = _validate_tool_arguments(
        arguments,
        schema=schema,
        source_messages_by_id=source_messages_by_id,
        allowed_source_message_ids=allowed_source_message_ids,
    )
    if error is not None:
        return _invalid_tool_call(error)
    return True, _accepted_tool_result(), _ValidatedContextToolCall(
        arguments=arguments,
        extraction=_tool_call_extraction(call.name, arguments),
    )


def _validate_context_selection_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    candidates_by_id: dict[str, ContextRegistryItem],
) -> tuple[bool, dict[str, str], ContextRegistryItem | None]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_selection_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_selection_tool_call(
            parse_error or "Tool arguments must be a JSON object"
        )
    shape_error = validate_tool_arguments_shape(
        arguments,
        schema=schema,
        enum_error_formatter=lambda field_name, value, _allowed: (
            f"{field_name} must be one of offered values; got {value}"
        ),
    )
    if shape_error is not None:
        return _invalid_selection_tool_call(shape_error)
    context_source_id = arguments.get("context_source_id")
    if not isinstance(context_source_id, str) or not context_source_id.strip():
        return _invalid_selection_tool_call("context_source_id is required")
    candidate = candidates_by_id.get(context_source_id)
    if candidate is None:
        return _invalid_selection_tool_call(
            f"context_source_id is not a prior-context candidate: {context_source_id}"
        )
    return (
        True,
        _accepted_tool_result(),
        replace(
            candidate,
            relevance_note=str(arguments.get("relevance_note", "")).strip()
            or "Selected by context update selection.",
        ),
    )


def _invalid_selection_tool_call(
    error: str,
) -> tuple[bool, dict[str, str], None]:
    return False, invalid_tool_result(error), None


def _validate_world_enrichment_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    source_messages_by_id: dict[str, MessageRecord],
    locations_by_id: dict[str, LocationRecord],
    threads_by_id: dict[str, ActiveThreadRecord],
    characters_by_id: dict[str, CharacterRecord],
) -> tuple[bool, dict[str, str], _ValidatedContextToolCall]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_tool_call(parse_error or "Tool arguments must be a JSON object")
    shape_error = validate_tool_arguments_shape(
        arguments,
        schema=schema,
        skip_enum_fields=frozenset({"source_message_id"}),
    )
    if shape_error is not None:
        return _invalid_tool_call(shape_error)
    source_message_id = arguments.get("source_message_id")
    if not isinstance(source_message_id, str) or not source_message_id.strip():
        return _invalid_tool_call("source_message_id is required")
    source_message = source_messages_by_id.get(source_message_id)
    if source_message is None:
        return _invalid_tool_call(
            f"source_message_id is not in the completed turn: {source_message_id}"
        )
    evidence_quote = arguments.get("evidence_quote")
    if not isinstance(evidence_quote, str) or not evidence_quote.strip():
        return _invalid_tool_call("evidence_quote is required")
    if not quote_matches_source(evidence_quote, source_message.body):
        return _invalid_tool_call(
            f"evidence_quote not found in source_message_id: {source_message_id}"
        )
    if call.name == "enrich_location":
        location_id = arguments.get("location_id")
        if not isinstance(location_id, str) or location_id not in locations_by_id:
            return _invalid_tool_call(
                f"location_id is not a sparse location: {location_id}"
            )
        return True, _accepted_tool_result(), _ValidatedContextToolCall(
            arguments=arguments,
            extraction=_location_enrichment_from_data(arguments),
        )
    if call.name == "enrich_active_thread":
        active_thread_id = arguments.get("active_thread_id")
        if (
            not isinstance(active_thread_id, str)
            or active_thread_id not in threads_by_id
        ):
            return _invalid_tool_call(
                f"active_thread_id is not a sparse active thread: {active_thread_id}"
            )
        return True, _accepted_tool_result(), _ValidatedContextToolCall(
            arguments=arguments,
            extraction=_thread_enrichment_from_data(arguments),
        )
    if call.name == "enrich_character":
        character_id = arguments.get("character_id")
        if not isinstance(character_id, str) or character_id not in characters_by_id:
            return _invalid_tool_call(
                f"character_id is not a sparse character: {character_id}"
            )
        return True, _accepted_tool_result(), _ValidatedContextToolCall(
            arguments=arguments,
            extraction=_character_enrichment_from_data(arguments),
        )
    return _invalid_tool_call(f"Unknown tool name: {call.name}")


def _invalid_tool_call(
    error: str,
) -> tuple[bool, dict[str, str], _ValidatedContextToolCall]:
    return (
        False,
        invalid_tool_result(error),
        _ValidatedContextToolCall(arguments={}, extraction=None),
    )


def _accepted_tool_result() -> dict[str, str]:
    return accepted_tool_result()


def _validate_tool_arguments(
    arguments: dict[str, object],
    *,
    schema: dict[str, object],
    source_messages_by_id: dict[str, MessageRecord] | None = None,
    allowed_source_message_ids: tuple[str, ...],
) -> str | None:
    shape_error = validate_tool_arguments_shape(
        arguments,
        schema=schema,
        skip_enum_fields=frozenset({"source_message_id"}),
    )
    if shape_error is not None:
        return shape_error
    source_message_id = arguments.get("source_message_id")
    if not isinstance(source_message_id, str) or not source_message_id.strip():
        return "source_message_id is required"
    if source_message_id not in set(allowed_source_message_ids):
        return f"source_message_id is not in the completed turn: {source_message_id}"
    if source_messages_by_id is not None:
        quote_error = _validate_exact_quote(
            arguments,
            source_messages_by_id=source_messages_by_id,
            quote_field="evidence_quote",
        )
        if quote_error is not None:
            return quote_error
    return None


def _tool_call_extraction(tool_name: str, arguments: dict[str, object]) -> object:
    if tool_name == "update_scene_snapshot":
        return _scene_from_data(arguments)
    if tool_name == "upsert_location":
        return _location_from_data(arguments)
    if tool_name == "upsert_character":
        return _character_from_data(arguments)
    if tool_name == "upsert_active_thread":
        return _thread_from_data(arguments)
    if tool_name == "link_entities":
        return _entity_link_from_data(arguments)
    if tool_name == "record_phone_number_exchange":
        return _phone_number_exchange_from_data(arguments)
    raise ValueError(f"Unsupported context update tool: {tool_name}")


def _canonical_tool_arguments(arguments: dict[str, object]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def _validate_exact_quote(
    arguments: dict[str, object],
    *,
    source_messages_by_id: dict[str, MessageRecord],
    quote_field: str,
) -> str | None:
    source_message_id = arguments.get("source_message_id")
    if not isinstance(source_message_id, str) or not source_message_id.strip():
        return "source_message_id is required"
    source = source_messages_by_id.get(source_message_id)
    if source is None:
        return f"source_message_id is not in the completed turn: {source_message_id}"
    quote = arguments.get(quote_field)
    if not isinstance(quote, str) or not quote.strip():
        return f"{quote_field} is required"
    if not quote_matches_source(quote, source.body):
        return f"{quote_field} not found in source message {source_message_id}"
    return None


def _initial_tool_diagnostics(
    *,
    provider: str,
    model_id: str,
    fallback_used: bool,
) -> dict[str, object]:
    return {
        "provider": provider,
        "model": model_id,
        "fallback_used": fallback_used,
        "retry_count": 0,
        "turns": [],
        "accepted_calls": [],
        "rejected_calls": [],
        "validation_errors": [],
    }


def _tool_call_diagnostic(call: ProviderToolCall) -> dict[str, object]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments_json": call.arguments_json,
    }


def _append_tool_diagnostic_call(
    diagnostics: dict[str, object],
    key: str,
    call: ProviderToolCall,
    *,
    error: str | None = None,
) -> None:
    calls = diagnostics.get(key)
    if not isinstance(calls, list):
        return
    entry = _tool_call_diagnostic(call)
    if error:
        entry["error"] = error
    calls.append(entry)
    if error:
        errors = diagnostics.get("validation_errors")
        if isinstance(errors, list):
            errors.append(error)


def _append_tool_diagnostic_turn(
    diagnostics: dict[str, object],
    *,
    turn: int,
    raw_calls: list[dict[str, object]],
    errors: list[str],
) -> None:
    turns = diagnostics.get("turns")
    if not isinstance(turns, list):
        return
    turns.append(
        {
            "turn": turn,
            "raw_tool_calls": raw_calls,
            "validation_errors": list(errors),
        }
    )


def _final_tool_diagnostics(
    diagnostics: dict[str, object],
) -> dict[str, object]:
    final = dict(diagnostics)
    for key in ("turns", "accepted_calls", "rejected_calls", "validation_errors"):
        value = final.get(key)
        if isinstance(value, list):
            final[key] = list(value)
    return final


def _attach_tool_diagnostics(
    exc: Exception,
    diagnostics: dict[str, object],
) -> None:
    exc.tool_diagnostics = diagnostics  # type: ignore[attr-defined]


def _tool_diagnostics_from_exception(exc: Exception) -> dict[str, object]:
    diagnostics = getattr(exc, "tool_diagnostics", None)
    return diagnostics if isinstance(diagnostics, dict) else {}


def _failure_tool_diagnostics(exc: Exception) -> dict[str, object] | None:
    diagnostics = _tool_diagnostics_from_exception(exc)
    if not diagnostics:
        return None
    return {"tool_diagnostics": diagnostics}


def _merge_tool_diagnostics(
    primary: dict[str, object],
    fallback: dict[str, object],
    *,
    fallback_used: bool,
) -> dict[str, object]:
    if not primary:
        merged = dict(fallback)
        merged["fallback_used"] = fallback_used
        return merged
    if not fallback:
        merged = dict(primary)
        merged["fallback_used"] = fallback_used
        return merged
    return {
        "fallback_used": fallback_used,
        "primary": primary,
        "fallback": fallback,
    }


class ContextUpdateService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        extractor: ContextUpdateExtractor,
        world_data_enricher: WorldDataEnricher | None = None,
        registry_selector: ContextRegistrySelector | None = None,
        focused_scene_maintainer: FocusedSceneMaintainer | None = None,
    ) -> None:
        self.repositories = repositories
        self.extractor = extractor
        self.world_data_enricher = world_data_enricher
        self.registry_selector = registry_selector
        self.focused_scene_maintainer = focused_scene_maintainer
        self.jobs = JobLifecycleService(repositories=repositories)

    async def update_after_turn(
        self,
        *,
        save_id: str,
        source_message_ids: tuple[str, ...],
        verified_coverage: VerifiedPostTurnCoverage | None = None,
    ) -> AppliedContextUpdate:
        job = self.jobs.create_running(
            save_id=save_id,
            type="context_update",
            payload={"source_message_ids": list(source_message_ids)},
            collect_provider_diagnostics=True,
        )
        log_event(
            "job.running",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            source_message_count=len(source_message_ids),
        )
        messages: tuple[MessageRecord, ...] = ()
        started_at = perf_counter()

        def record_context_step(
            name: str,
            step_started_at: float,
            *,
            metadata: dict[str, object] | None = None,
        ) -> None:
            self.jobs.record_step(
                job.id,
                name=name,
                status="succeeded",
                task="context_update",
                duration_ms=_elapsed_ms(step_started_at),
                metadata=metadata,
            )

        try:
            step_started_at = perf_counter()
            if archive_inactive_active_threads(self.repositories, save_id):
                archive_open_thread_aggregate_state(self.repositories, save_id)
            ContinuityIndexService(self.repositories).sync_save(save_id)
            snapshot = _load_context_update_read_snapshot(
                self.repositories,
                save_id=save_id,
                source_message_ids=source_message_ids,
            )
            messages = snapshot.messages
            record_context_step(
                "snapshot",
                step_started_at,
                metadata={
                    "message_count": len(messages),
                    "context_source_count": len(snapshot.context_sources),
                    "context_observation_count": len(snapshot.context_observations),
                },
            )
            step_started_at = perf_counter()
            selection = await self._select_prior_context(
                save_id=save_id,
                messages=messages,
                scene_snapshot=snapshot.scene_snapshot,
                locations=snapshot.locations,
                characters=snapshot.characters,
                active_threads=snapshot.active_threads,
                context_sources=snapshot.context_sources,
                context_observations=snapshot.context_observations,
                all_messages=snapshot.all_messages,
            )
            record_context_step(
                "prior_context_selection",
                step_started_at,
                metadata={"selected_count": len(selection.selected_items)},
            )
            request = ContextUpdateRequest(
                save_id=save_id,
                messages=messages,
                scene_snapshot=snapshot.scene_snapshot,
                locations=snapshot.locations,
                characters=snapshot.characters,
                active_threads=snapshot.active_threads,
                entity_links=snapshot.entity_links,
                memories=snapshot.memories,
                world_state=snapshot.world_state,
                summaries=snapshot.summaries,
                prior_context=selection.selected_items,
            )
            step_started_at = perf_counter()
            extraction = await self.extractor.extract(request)
            extraction = _filter_extraction_for_verified_coverage(
                extraction,
                coverage=verified_coverage,
                request=request,
            )
            record_context_step("extraction", step_started_at)
            step_started_at = perf_counter()
            self.repositories.begin_transaction()
            applied = self.apply_extraction(
                save_id=save_id,
                extraction=extraction,
                allowed_source_message_ids=tuple(message.id for message in messages),
                completed_messages=messages,
            )
            self.repositories.commit_transaction()
            record_context_step(
                "apply",
                step_started_at,
                metadata={
                    "location_count": len(applied.locations),
                    "character_count": len(applied.characters),
                    "active_thread_count": len(applied.active_threads),
                    "entity_link_count": len(applied.entity_links),
                    "suggestion_count": len(applied.suggestions),
                },
            )
            step_started_at = perf_counter()
            focused_applied = await self._maintain_focused_scene_after_update(
                save_id=save_id,
                messages=messages,
                verified_coverage=verified_coverage,
            )
            record_context_step(
                "focused_scene",
                step_started_at,
                metadata=focused_applied.to_json() if not focused_applied.empty else {},
            )
            step_started_at = perf_counter()
            applied = await self._enrich_world_data_after_update(
                save_id=save_id,
                messages=messages,
                applied=applied,
                prior_context=selection.selected_items,
            )
            record_context_step("world_data_enrichment", step_started_at)
        except asyncio.CancelledError:
            self.repositories.rollback_transaction()
            self.jobs.cancel(job.id, error="Context update cancelled")
            log_event(
                "job.cancelled",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                duration_ms=_elapsed_ms(started_at),
            )
            raise
        except Exception as exc:
            self.repositories.rollback_transaction()
            failure_result = _failure_tool_diagnostics(exc)
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
                result=failure_result,
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise

        result: dict[str, object] = {
            "scene_snapshot_updated": applied.scene_snapshot is not None,
            "location_count": len(applied.locations),
            "character_count": len(applied.characters),
            "active_thread_count": len(applied.active_threads),
            "entity_link_count": len(applied.entity_links),
            "suggestion_count": len(applied.suggestions),
            "audit_count": len(applied.audit_entries),
        }
        if not focused_applied.empty:
            result["focused_scene"] = focused_applied.to_json()
            if focused_applied.provider_pressure is not None:
                result["provider_pressure"] = focused_applied.provider_pressure
        diagnostics = getattr(extraction, "tool_diagnostics", {})
        if isinstance(diagnostics, dict) and diagnostics:
            result["tool_diagnostics"] = diagnostics
        applied = replace(applied, job_result=result)
        self.jobs.succeed(job.id, result=result)
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            duration_ms=_elapsed_ms(started_at),
            location_count=len(applied.locations),
            character_count=len(applied.characters),
            active_thread_count=len(applied.active_threads),
            entity_link_count=len(applied.entity_links),
            suggestion_count=len(applied.suggestions),
            audit_count=len(applied.audit_entries),
        )
        return applied

    async def update_after_message_correction(
        self,
        *,
        save_id: str,
        source_message_id: str,
        correction_context: MessageCorrectionContext,
    ) -> AppliedContextUpdate:
        messages = tuple(
            message
            for message in self.repositories.list_messages(save_id)
            if message.id == source_message_id
        )
        if archive_inactive_active_threads(self.repositories, save_id):
            archive_open_thread_aggregate_state(self.repositories, save_id)
        ContinuityIndexService(self.repositories).sync_save(save_id)
        scene_snapshot = _scene_snapshot_without_source_message(
            self.repositories.get_scene_snapshot(save_id),
            source_message_id,
        )
        locations = tuple(
            location
            for location in self.repositories.list_locations(save_id)
            if location.source_message_id != source_message_id
        )
        characters = tuple(
            character
            for character in self.repositories.list_characters(save_id)
            if character.source_message_id != source_message_id
            or character.protected_from_maintenance
        )
        active_threads = tuple(
            thread
            for thread in self.repositories.list_active_threads(save_id)
            if thread.source_message_id != source_message_id
            and active_thread_is_prompt_visible(thread)
        )
        memories = tuple(
            memory
            for memory in self.repositories.list_memories(save_id)
            if source_message_id not in _memory_source_ids(memory)
        )
        world_state = tuple(
            state
            for state in self.repositories.list_world_state(save_id)
            if state.source_message_id != source_message_id
        )
        summaries = _summaries_without_covered_message(
            summaries=tuple(self.repositories.list_summaries(save_id)),
            messages=tuple(self.repositories.list_messages(save_id)),
            message_id=source_message_id,
        )
        entity_links = tuple(
            link
            for link in self.repositories.list_entity_links(save_id)
            if link.source_message_id != source_message_id
            and _entity_link_visible_for_correction(
                link=link,
                locations=locations,
                characters=characters,
                active_threads=active_threads,
                memories=memories,
                world_state=world_state,
            )
        )
        selection = await self._select_prior_context(
            save_id=save_id,
            messages=messages,
            scene_snapshot=scene_snapshot,
            locations=locations,
            characters=characters,
            active_threads=active_threads,
            excluded_source_message_ids=frozenset({source_message_id}),
        )
        request = ContextUpdateRequest(
            save_id=save_id,
            messages=messages,
            scene_snapshot=scene_snapshot,
            locations=locations,
            characters=characters,
            active_threads=active_threads,
            entity_links=entity_links,
            memories=memories,
            world_state=world_state,
            summaries=summaries,
            prior_context=selection.selected_items,
            correction_context=correction_context,
        )
        extraction = await self.extractor.extract(request)
        self.repositories.begin_transaction()
        try:
            self._archive_message_correction_context(
                save_id=save_id,
                source_message_id=source_message_id,
            )
            applied = self.apply_extraction(
                save_id=save_id,
                extraction=extraction,
                allowed_source_message_ids=tuple(message.id for message in messages),
                completed_messages=messages,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        return await self._enrich_world_data_after_update(
            save_id=save_id,
            messages=messages,
            applied=applied,
            prior_context=selection.selected_items,
        )

    def _archive_message_correction_context(
        self,
        *,
        save_id: str,
        source_message_id: str,
    ) -> None:
        message_ids = frozenset({source_message_id})
        self.repositories.expire_context_update_suggestions_for_messages(
            save_id=save_id,
            message_ids=message_ids,
        )
        self.repositories.archive_context_sources_for_deleted_messages(
            save_id=save_id,
            message_ids=message_ids,
        )
        self.repositories.archive_context_observations_for_deleted_messages(
            save_id=save_id,
            message_ids=message_ids,
        )
        self.repositories.delete_entity_links_for_source_message(
            save_id=save_id,
            source_message_id=source_message_id,
        )
        snapshot = self.repositories.get_scene_snapshot(save_id)
        if snapshot is not None and snapshot.source_message_id == source_message_id:
            self.repositories.delete_scene_snapshot(save_id)
        for location in self.repositories.list_locations(save_id):
            if location.source_message_id != source_message_id:
                continue
            self.repositories.delete_entity_links_for_endpoint(
                save_id=save_id,
                entity_type="location",
                entity_id=location.id,
            )
            self.repositories.archive_location(location.id)
        for character in self.repositories.list_characters(save_id):
            if character.source_message_id != source_message_id:
                continue
            if character.protected_from_maintenance:
                continue
            self.repositories.delete_entity_links_for_endpoint(
                save_id=save_id,
                entity_type="character",
                entity_id=character.id,
            )
            self.repositories.archive_character(character.id)
        for thread in self.repositories.list_active_threads(save_id):
            if thread.source_message_id != source_message_id:
                continue
            self.repositories.delete_entity_links_for_endpoint(
                save_id=save_id,
                entity_type="active_thread",
                entity_id=thread.id,
            )
            self.repositories.archive_active_thread(thread.id)

    async def _select_prior_context(
        self,
        *,
        save_id: str,
        messages: tuple[MessageRecord, ...],
        scene_snapshot: SceneSnapshotRecord | None,
        locations: tuple[LocationRecord, ...],
        characters: tuple[CharacterRecord, ...],
        active_threads: tuple[ActiveThreadRecord, ...],
        context_sources: tuple[ContextSourceRecord, ...] | None = None,
        context_observations: tuple[ContextObservationRecord, ...] | None = None,
        all_messages: tuple[MessageRecord, ...] | None = None,
        excluded_source_message_ids: frozenset[str] = frozenset(),
    ) -> ContextRegistrySelection:
        candidates = _context_registry_candidates(
            repositories=self.repositories,
            save_id=save_id,
            messages=messages,
            scene_snapshot=scene_snapshot,
            locations=locations,
            characters=characters,
            active_threads=active_threads,
            context_sources=context_sources,
            context_observations=context_observations,
            all_messages=all_messages,
        )
        if excluded_source_message_ids:
            candidates = tuple(
                candidate
                for candidate in candidates
                if not (
                    frozenset(candidate.source_message_ids)
                    & excluded_source_message_ids
                )
            )
        if not candidates:
            return ContextRegistrySelection()
        explicit_selector = self.registry_selector
        derived_selector = (
            None if explicit_selector is not None else _selector_from_extractor(
                self.extractor
            )
        )
        if (
            explicit_selector is None
            and len(candidates) <= MAX_CONTEXT_UPDATE_SELECTIONS
        ):
            if getattr(derived_selector, "prompt_inspection_store", None) is None:
                return _fallback_context_registry_selection(candidates)
        selector = explicit_selector or derived_selector
        if selector is None:
            return _fallback_context_registry_selection(candidates)
        try:
            selection = await selector.select_context(
                ContextRegistrySelectionRequest(
                    save_id=save_id,
                    messages=messages,
                    scene_snapshot=scene_snapshot,
                    locations=locations,
                    characters=characters,
                    active_threads=active_threads,
                    candidates=candidates,
                )
            )
        except Exception as exc:
            log_error_event(
                "context_update.context_selection_failed",
                save_id=save_id,
                candidate_count=len(candidates),
                **exception_log_fields(exc),
            )
            return _fallback_context_registry_selection(candidates)
        selected = _normalize_context_registry_selection(selection, candidates)
        if not selected.selected_items:
            return _fallback_context_registry_selection(candidates)
        return selected

    def apply_extraction(
        self,
        *,
        save_id: str,
        extraction: ContextUpdateExtraction,
        allowed_source_message_ids: tuple[str, ...] | None = None,
        completed_messages: tuple[MessageRecord, ...] = (),
    ) -> AppliedContextUpdate:
        extraction = _drop_invalid_extracted_entities(extraction)
        extraction = _drop_unknown_extraction_sources(
            extraction,
            allowed_source_message_ids=allowed_source_message_ids,
        )
        extraction = _drop_safety_transition_extraction_sources(
            extraction,
            blocked_source_message_ids=_safety_transition_source_ids(
                completed_messages
            ),
        )
        if completed_messages:
            extraction = _filter_structured_extraction_evidence(
                extraction,
                source_messages_by_id={
                    message.id: message for message in completed_messages
                },
            )
        extraction = _with_inferred_phone_number_exchanges(
            self.repositories,
            save_id=save_id,
            extraction=extraction,
            completed_messages=completed_messages,
        )
        _validate_extraction(
            extraction,
            allowed_source_message_ids=allowed_source_message_ids,
        )
        applier = _ContextUpdateApplier(
            repositories=self.repositories,
            save_id=save_id,
            completed_messages=completed_messages,
        )
        applier.apply(extraction)
        return applier.result()

    async def _maintain_focused_scene_after_update(
        self,
        *,
        save_id: str,
        messages: tuple[MessageRecord, ...],
        verified_coverage: VerifiedPostTurnCoverage | None = None,
    ) -> AppliedFocusedSceneMaintenance:
        if self.focused_scene_maintainer is None:
            return AppliedFocusedSceneMaintenance()
        scene_snapshot = self.repositories.get_scene_snapshot(save_id)
        if scene_snapshot is None:
            return AppliedFocusedSceneMaintenance()
        request = FocusedSceneMaintenanceRequest(
            save_id=save_id,
            messages=messages,
            scene_snapshot=scene_snapshot,
            locations=tuple(self.repositories.list_locations(save_id)),
            characters=tuple(self.repositories.list_characters(save_id)),
            active_threads=tuple(self.repositories.list_active_threads(save_id)),
            world_state=tuple(self.repositories.list_world_state(save_id)),
        )
        try:
            maintenance = await self.focused_scene_maintainer.maintain(request)
            if verified_coverage is not None and not verified_coverage.empty:
                maintenance = replace(
                    maintenance,
                    scene_updates=tuple(
                        _filter_scene_for_verified_coverage(
                            update,
                            coverage=verified_coverage,
                            current_snapshot=request.scene_snapshot,
                            characters=request.characters,
                        )
                        for update in maintenance.scene_updates
                    ),
                )
            self.repositories.begin_transaction()
            applied = self.apply_focused_scene_maintenance(
                save_id=save_id,
                maintenance=maintenance,
                allowed_source_message_ids=tuple(message.id for message in messages),
                completed_messages=messages,
            )
            self.repositories.commit_transaction()
        except Exception as exc:
            self.repositories.rollback_transaction()
            log_error_event(
                "context_update.focused_scene_maintenance_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            diagnostics = _tool_diagnostics_from_exception(exc)
            if not diagnostics:
                diagnostics = {
                    "error": redact_text(str(exc)) or exc.__class__.__name__
                }
            pressure = provider_pressure_from_exception(exc)
            provider_pressure = (
                pressure.to_result() if pressure is not None else None
            )
            if provider_pressure is not None:
                diagnostics["provider_pressure"] = provider_pressure
            return AppliedFocusedSceneMaintenance(
                tool_diagnostics=diagnostics,
                provider_pressure=provider_pressure,
            )
        return applied

    def apply_focused_scene_maintenance(
        self,
        *,
        save_id: str,
        maintenance: FocusedSceneMaintenance,
        allowed_source_message_ids: tuple[str, ...] | None = None,
        completed_messages: tuple[MessageRecord, ...] = (),
    ) -> AppliedFocusedSceneMaintenance:
        maintenance = _drop_safety_transition_focused_maintenance(
            maintenance,
            blocked_source_message_ids=_safety_transition_source_ids(
                completed_messages
            ),
        )
        _validate_focused_scene_maintenance(
            maintenance,
            allowed_source_message_ids=allowed_source_message_ids,
            completed_messages=completed_messages,
        )
        scene_snapshot: SceneSnapshotRecord | None = None
        characters: list[CharacterRecord] = []
        active_threads: list[ActiveThreadRecord] = []
        audit_entries: list[ContextUpdateAuditRecord] = []
        suggestions: list[ContextUpdateSuggestionRecord] = []
        world_state: list[WorldStateRecord] = []
        state_changes: list[StateChangeRecord] = []
        if (
            maintenance.scene_updates
            or maintenance.active_thread_updates
            or maintenance.character_relationships
        ):
            applier = _ContextUpdateApplier(
                repositories=self.repositories,
                save_id=save_id,
                completed_messages=completed_messages,
            )
            applier.scene_local_thread_ids_before_scene = {
                thread.id
                for thread in applier.snapshot.active_threads
                if active_thread_is_scene_local(thread)
            }
            for scene_update in maintenance.scene_updates:
                _apply_focused_scene_update(applier, scene_update)
            for thread_update in maintenance.active_thread_updates:
                _apply_focused_active_thread_status(applier, thread_update)
            for relationship in maintenance.character_relationships:
                _apply_focused_character_relationship(
                    applier,
                    relationship,
                    completed_messages=completed_messages,
                )
            scene_snapshot = applier.scene_snapshot
            characters.extend(applier.characters)
            active_threads.extend(applier.active_threads)
            audit_entries.extend(applier.audit_entries)
            suggestions.extend(applier.suggestions)
        for emotion in maintenance.character_emotions:
            applied = _apply_focused_character_emotion(
                repositories=self.repositories,
                save_id=save_id,
                emotion=emotion,
            )
            if applied is None:
                continue
            world_state.extend(applied.world_state)
            suggestions.extend(applied.suggestions)
            audit_entries.extend(applied.audit_entries)
            state_changes.extend(applied.state_changes)
        return AppliedFocusedSceneMaintenance(
            scene_snapshot=scene_snapshot,
            characters=tuple(characters),
            active_threads=tuple(active_threads),
            world_state=tuple(world_state),
            suggestions=tuple(suggestions),
            audit_entries=tuple(audit_entries),
            state_changes=tuple(state_changes),
            tool_diagnostics=maintenance.tool_diagnostics,
        )

    async def _enrich_world_data_after_update(
        self,
        *,
        save_id: str,
        messages: tuple[MessageRecord, ...],
        applied: AppliedContextUpdate,
        prior_context: tuple[ContextRegistryItem, ...] = (),
    ) -> AppliedContextUpdate:
        if self.world_data_enricher is None:
            return applied
        candidate_character_ids = frozenset(
            character.id for character in applied.characters
        )
        request = _world_data_enrichment_request(
            repositories=self.repositories,
            save_id=save_id,
            messages=messages,
            prior_context=prior_context,
            candidate_character_ids=candidate_character_ids,
        )
        if (
            not request.locations
            and not request.active_threads
            and not request.characters
        ):
            return applied
        try:
            enrichment = await self.world_data_enricher.enrich(request)
            self.repositories.begin_transaction()
            enriched = self.apply_world_data_enrichment(
                save_id=save_id,
                enrichment=enrichment,
                allowed_source_message_ids=tuple(message.id for message in messages),
                completed_messages=messages,
            )
            self.repositories.commit_transaction()
        except Exception as exc:
            self.repositories.rollback_transaction()
            log_error_event(
                "context_update.world_data_enrichment_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return applied
        return _merge_applied_context_updates(applied, enriched)

    def apply_world_data_enrichment(
        self,
        *,
        save_id: str,
        enrichment: WorldDataEnrichment,
        allowed_source_message_ids: tuple[str, ...] | None = None,
        completed_messages: tuple[MessageRecord, ...] = (),
    ) -> AppliedContextUpdate:
        blocked_source_message_ids = _safety_transition_source_ids(
            completed_messages
        )
        if completed_messages:
            enrichment = _filter_world_data_enrichment_evidence(
                enrichment,
                source_messages_by_id={
                    message.id: message for message in completed_messages
                },
            )
        enrichment = _drop_safety_transition_world_data_enrichment(
            enrichment,
            blocked_source_message_ids=blocked_source_message_ids,
        )
        _validate_world_data_enrichment(
            enrichment,
            allowed_source_message_ids=allowed_source_message_ids,
        )
        applier = _ContextUpdateApplier(
            repositories=self.repositories,
            save_id=save_id,
        )
        applier.apply_world_data_enrichment(enrichment)
        return applier.result()


def _safety_transition_source_ids(
    messages: tuple[MessageRecord, ...],
) -> frozenset[str]:
    return frozenset(
        message.id
        for message in messages
        if is_fade_to_black_message(
            role=message.role,
            body=message.body,
            safety_transition=message.safety_transition,
        )
    )


def _drop_safety_transition_extraction_sources(
    extraction: ContextUpdateExtraction,
    *,
    blocked_source_message_ids: frozenset[str],
) -> ContextUpdateExtraction:
    if not blocked_source_message_ids:
        return extraction
    def keep(item: object) -> bool:
        return (
            getattr(item, "source_message_id", "")
            not in blocked_source_message_ids
        )
    return replace(
        extraction,
        scene=(
            extraction.scene
            if extraction.scene is None or keep(extraction.scene)
            else None
        ),
        locations=tuple(filter(keep, extraction.locations)),
        characters=tuple(filter(keep, extraction.characters)),
        active_threads=tuple(filter(keep, extraction.active_threads)),
        entity_links=tuple(filter(keep, extraction.entity_links)),
        phone_number_exchanges=tuple(
            filter(keep, extraction.phone_number_exchanges)
        ),
    )


def _drop_safety_transition_focused_maintenance(
    maintenance: FocusedSceneMaintenance,
    *,
    blocked_source_message_ids: frozenset[str],
) -> FocusedSceneMaintenance:
    if not blocked_source_message_ids:
        return maintenance
    def keep(item: object) -> bool:
        return (
            getattr(item, "source_message_id", "")
            not in blocked_source_message_ids
        )
    return replace(
        maintenance,
        scene_updates=tuple(filter(keep, maintenance.scene_updates)),
        active_thread_updates=tuple(
            filter(keep, maintenance.active_thread_updates)
        ),
        character_relationships=tuple(
            filter(keep, maintenance.character_relationships)
        ),
        character_emotions=tuple(filter(keep, maintenance.character_emotions)),
    )


def _drop_safety_transition_world_data_enrichment(
    enrichment: WorldDataEnrichment,
    *,
    blocked_source_message_ids: frozenset[str],
) -> WorldDataEnrichment:
    if not blocked_source_message_ids:
        return enrichment
    def keep(item: object) -> bool:
        return (
            getattr(item, "source_message_id", "")
            not in blocked_source_message_ids
        )
    return replace(
        enrichment,
        locations=tuple(filter(keep, enrichment.locations)),
        active_threads=tuple(filter(keep, enrichment.active_threads)),
        characters=tuple(filter(keep, enrichment.characters)),
    )


def _validate_focused_scene_maintenance(
    maintenance: FocusedSceneMaintenance,
    *,
    allowed_source_message_ids: tuple[str, ...] | None = None,
    completed_messages: tuple[MessageRecord, ...] = (),
) -> None:
    allowed_ids = set(allowed_source_message_ids or ())
    source_messages_by_id = {message.id: message for message in completed_messages}

    def validate_evidence(
        source_message_id: str,
        evidence_quote: str,
        label: str,
    ) -> None:
        if not source_message_id:
            raise ValueError(f"{label} source_message_id is required")
        if (
            allowed_source_message_ids is not None
            and source_message_id not in allowed_ids
        ):
            raise ValueError(f"Unknown {label} source_message_id: {source_message_id}")
        if not evidence_quote.strip():
            raise ValueError(f"{label} evidence_quote is required")
        if completed_messages:
            source = source_messages_by_id.get(source_message_id)
            if source is None:
                raise ValueError(
                    f"{label} source_message_id is not in completed turn: "
                    f"{source_message_id}"
                )
            if not quote_matches_source(evidence_quote, source.body):
                raise ValueError(
                    f"{label} evidence_quote not found in source_message_id: "
                    f"{source_message_id}"
                )

    for scene in maintenance.scene_updates:
        validate_evidence(
            scene.source_message_id,
            scene.evidence_quote,
            "focused scene",
        )
    for thread in maintenance.active_thread_updates:
        if not thread.active_thread_id:
            raise ValueError("Focused active thread id is required")
        if not thread.status.strip():
            raise ValueError("Focused active thread status is required")
        validate_evidence(
            thread.source_message_id,
            thread.evidence_quote,
            "focused active thread",
        )
    for relationship in maintenance.character_relationships:
        if not relationship.character_id:
            raise ValueError("Focused relationship character_id is required")
        if not relationship.target_name.strip():
            raise ValueError("Focused relationship target_name is required")
        if not relationship.posture.strip():
            raise ValueError("Focused relationship posture is required")
        validate_evidence(
            relationship.source_message_id,
            relationship.evidence_quote,
            "focused relationship",
        )
    for emotion in maintenance.character_emotions:
        if not emotion.character_id:
            raise ValueError("Focused character emotion character_id is required")
        if not emotion.emotional_state.strip():
            raise ValueError("Focused character emotion state is required")
        validate_evidence(
            emotion.source_message_id,
            emotion.evidence_quote,
            "focused emotion",
        )


def _apply_focused_scene_update(
    applier: _ContextUpdateApplier,
    extracted: ExtractedSceneSnapshot,
) -> None:
    snapshot = applier.repositories.get_scene_snapshot(applier.save_id)
    if snapshot is None:
        return
    previous_location_id = snapshot.current_location_id
    generation_before = snapshot.scene_generation
    scene_updates: list[tuple[str, object]] = []
    current_location = _find_location(
        applier.snapshot.locations,
        extracted.current_location_name,
    )
    if current_location is not None:
        snapshot = applier._apply_scene_field(
            snapshot=snapshot,
            field_path="current_location_id",
            value=current_location.id,
            reason=extracted.reason,
            confidence=extracted.confidence,
            source_message_id=extracted.source_message_id,
        )
    if (
        extracted.scene_transition
        and snapshot.scene_generation == generation_before
    ):
        snapshot = applier.repositories.advance_scene_generation(
            save_id=applier.save_id,
            source_message_id=extracted.source_message_id,
        )
        applier._record_applied(
            operation="scene_generation_advanced",
            entity_type="scene_snapshot",
            entity_id=snapshot.id,
            field_path="scene_generation",
            before=generation_before,
            after=snapshot.scene_generation,
            reason=extracted.reason,
            confidence=extracted.confidence,
            source_message_ids=[extracted.source_message_id],
        )
    applier.scene_snapshot = snapshot
    if extracted.in_world_time.strip():
        scene_updates.append(("in_world_time", extracted.in_world_time.strip()))
    if extracted.weather.strip():
        scene_updates.append(("weather", extracted.weather.strip()))
    if extracted.mood.strip():
        scene_updates.append(("mood", extracted.mood.strip()))
    if extracted.nearby_objects is not None:
        scene_updates.append(
            ("nearby_objects", list(_clean_strings(extracted.nearby_objects)))
        )
    if extracted.hazards is not None:
        scene_updates.append(("hazards", list(_clean_strings(extracted.hazards))))
    present_character_ids = _focused_present_character_ids(
        applier.snapshot.characters,
        extracted.present_character_names,
    )
    if present_character_ids is not None:
        scene_updates.append(("present_character_ids", present_character_ids))
    original_snapshot = snapshot
    for field_path, value in scene_updates:
        snapshot = applier._apply_scene_field(
            snapshot=snapshot,
            field_path=field_path,
            value=value,
            reason=extracted.reason,
            confidence=extracted.confidence,
            source_message_id=extracted.source_message_id,
        )
    if snapshot != original_snapshot:
        applier.scene_snapshot = snapshot
    applier._archive_scene_local_threads_after_scene_change(
        previous_location_id=previous_location_id,
        current_location_id=snapshot.current_location_id,
        scene_transition=extracted.scene_transition,
        reason=extracted.reason,
        confidence=extracted.confidence,
        source_message_id=extracted.source_message_id,
    )


def _apply_focused_active_thread_status(
    applier: _ContextUpdateApplier,
    extracted: ExtractedFocusedActiveThreadStatus,
) -> None:
    thread = applier.repositories.get_active_thread(extracted.active_thread_id.strip())
    if thread is None or thread.save_id != applier.save_id:
        return
    if not active_thread_status_is_open(thread.status):
        return
    if not active_thread_is_scene_local(thread):
        return
    thread_update = ExtractedActiveThread(
        title=thread.title,
        source_message_id=extracted.source_message_id,
        status=extracted.status,
        reason=extracted.reason,
        confidence=extracted.confidence,
    )
    before_audit_count = len(applier.audit_entries)
    before_suggestion_count = len(applier.suggestions)
    thread = applier._normalize_thread(thread, thread_update)
    applier.touched_thread_ids.add(thread.id)
    raw_status = extracted.status.strip()
    normalized_status = normalize_active_thread_status(raw_status)
    thread = applier._apply_field(
        record=thread,
        entity_type="active_thread",
        entity_id=thread.id,
        field_path="status",
        value=normalized_status if raw_status else "",
        reason=extracted.reason,
        confidence=extracted.confidence,
        source_message_id=extracted.source_message_id,
        update=applier.repositories.update_active_thread,
    )
    if (
        len(applier.audit_entries) == before_audit_count
        and len(applier.suggestions) == before_suggestion_count
    ):
        return
    if raw_status and not active_thread_status_is_open(thread.status):
        applier._archive_thread(
            thread,
            reason=extracted.reason,
            confidence=extracted.confidence,
            source_message_id=extracted.source_message_id,
        )
        _append_unique(applier.active_threads, thread)
        return
    applier._append_thread(thread)


def _apply_focused_character_relationship(
    applier: _ContextUpdateApplier,
    extracted: ExtractedFocusedCharacterRelationship,
    *,
    completed_messages: tuple[MessageRecord, ...],
) -> None:
    character = applier.repositories.get_character(extracted.character_id.strip())
    if character is None or character.save_id != applier.save_id:
        return
    snapshot = applier.scene_snapshot or applier.repositories.get_scene_snapshot(
        applier.save_id
    )
    if snapshot is None or character.id not in set(snapshot.present_character_ids):
        return
    target_key = _focused_relationship_target_key(
        characters=applier.snapshot.characters,
        scene_snapshot=snapshot,
        messages=completed_messages,
        subject=character,
        target_name=extracted.target_name,
    )
    if not target_key:
        target_name = extracted.target_name.strip()
        if not target_name:
            return
        relationships = {
            **character.relationships,
            target_name: extracted.posture.strip(),
        }
        applier._queue_suggestion(
            update_type="update",
            entity_type="character",
            entity_id=character.id,
            field_path="relationships",
            before=character.relationships,
            after=relationships,
            reason=extracted.reason
            or "Focused relationship posture target needs review.",
            confidence=extracted.confidence,
            source_message_ids=[extracted.source_message_id],
        )
        return
    posture = extracted.posture.strip()
    relationships = {**character.relationships, target_key: posture}
    if relationships == character.relationships:
        return
    if _focused_relationship_requires_review(
        character,
        target_key=target_key,
    ):
        applier._queue_suggestion(
            update_type="update",
            entity_type="character",
            entity_id=character.id,
            field_path="relationships",
            before=character.relationships,
            after=relationships,
            reason=extracted.reason
            or "Focused relationship posture update needs review.",
            confidence=extracted.confidence,
            source_message_ids=[extracted.source_message_id],
        )
        return
    updated = applier.repositories.update_character(
        replace(
            character,
            relationships=relationships,
            source_message_id=extracted.source_message_id,
            last_updated_message_id=extracted.source_message_id,
        )
    )
    applier.snapshot.upsert_character(updated)
    applier._record_applied(
        operation="updated",
        entity_type="character",
        entity_id=character.id,
        field_path="relationships",
        before=character.relationships,
        after=relationships,
        reason=extracted.reason,
        confidence=extracted.confidence,
        source_message_ids=[extracted.source_message_id],
    )
    applier._append_character(updated)


def _focused_relationship_requires_review(
    character: CharacterRecord,
    *,
    target_key: str,
) -> bool:
    if character_field_is_locked(character.locked_fields, "relationships"):
        return True
    current = character.relationships.get(target_key)
    if _is_empty_update(current):
        return False
    if isinstance(current, str):
        return not _focused_relationship_posture_is_generic_route(current)
    return True


def _focused_relationship_posture_is_generic_route(value: str) -> bool:
    normalized = value.casefold()
    return any(
        phrase in normalized
        for phrase in (
            "romance option",
            "available route",
            "available romance",
            "route option",
        )
    )


def _focused_relationship_target_key(
    *,
    characters: Iterable[CharacterRecord],
    scene_snapshot: SceneSnapshotRecord,
    messages: tuple[MessageRecord, ...],
    subject: CharacterRecord,
    target_name: str,
) -> str:
    target_name = target_name.strip()
    if not target_name:
        return ""
    player_name = _focused_scene_player_name(messages)
    target_key = _character_name_key(target_name)
    player_key = _character_name_key(player_name)
    if target_key and (
        target_key == player_key
        or target_key in {"player", "the player"}
        or target_name.casefold() in {"player", "the player"}
    ):
        return player_name
    present_characters = tuple(
        character
        for character in characters
        if character.id in set(scene_snapshot.present_character_ids)
        and character.id != subject.id
    )
    resolution = _resolve_character(present_characters, target_name)
    if resolution.record is None or resolution.ambiguous:
        return ""
    return resolution.record.name


def _focused_present_character_ids(
    characters: Iterable[CharacterRecord],
    names: tuple[str, ...] | None,
) -> list[str] | None:
    if names is None:
        return None
    resolved_ids: list[str] = []
    character_records = tuple(characters)
    for name in _clean_strings(names):
        resolution = _resolve_character(character_records, name)
        if resolution.record is None or resolution.ambiguous:
            return None
        resolved_ids.append(resolution.record.id)
    return list(dict.fromkeys(resolved_ids))


def _apply_focused_character_emotion(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    emotion: ExtractedFocusedCharacterEmotion,
) -> AppliedFocusedSceneMaintenance | None:
    character = repositories.get_character(emotion.character_id)
    if character is None or character.save_id != save_id:
        return None
    key = _character_emotion_state_key(character)
    if not key:
        return None
    value: dict[str, object] = {"mood": emotion.emotional_state.strip()}
    before = _find_world_state_by_key(repositories.list_world_state(save_id), key)
    source_message_ids = [emotion.source_message_id]
    reason = emotion.reason or f"Updated {character.name}'s current emotional state."
    if manual_state_change_confirmation_enabled(repositories, save_id=save_id):
        return _queue_focused_character_emotion_confirmation(
            repositories=repositories,
            save_id=save_id,
            key=key,
            value=value,
            before=before,
            reason=reason,
            confidence=emotion.confidence,
            source_message_id=emotion.source_message_id,
        )
    if before is not None and before.value == value:
        return None

    state = repositories.upsert_world_state(
        save_id=save_id,
        key=key,
        value=value,
        category="scene",
        confidence=emotion.confidence,
        source_message_id=emotion.source_message_id,
    )
    state_change = repositories.add_state_change(
        save_id=save_id,
        source_message_id=emotion.source_message_id,
        operation="upsert",
        state_key=key,
        before_json=_dump_json_compact(before.value) if before is not None else None,
        after_json=_dump_json_compact(value),
    )
    audit = repositories.add_context_update_audit(
        save_id=save_id,
        operation="updated" if before is not None else "created",
        entity_type="world_state",
        entity_id=state.id,
        field_path=key,
        before=before.value if before is not None else None,
        after=value,
        reason=reason,
        confidence=emotion.confidence,
        source_message_ids=source_message_ids,
    )
    return AppliedFocusedSceneMaintenance(
        world_state=(state,),
        audit_entries=(audit,),
        state_changes=(state_change,),
    )


def _queue_focused_character_emotion_confirmation(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    key: str,
    value: dict[str, object],
    before: WorldStateRecord | None,
    reason: str,
    confidence: float,
    source_message_id: str,
) -> AppliedFocusedSceneMaintenance | None:
    entity_id = before.id if before is not None else None
    state_already_current = before is not None and before.value == value
    matching_pending = tuple(
        pending
        for pending in repositories.list_context_update_suggestions(
            save_id,
            status="pending",
        )
        if pending.update_type == "upsert"
        and pending.entity_type == "world_state"
        and pending.field_path == key
    )
    duplicate_pending: ContextUpdateSuggestionRecord | None = None
    for pending in matching_pending:
        if state_already_current:
            repositories.update_context_update_suggestion_status(
                pending.id,
                status="superseded",
            )
            continue
        if _world_state_suggestion_value(pending.proposed_value, key) == value:
            if duplicate_pending is None:
                duplicate_pending = pending
                continue
            repositories.update_context_update_suggestion_status(
                pending.id,
                status="superseded",
            )
            continue
        repositories.update_context_update_suggestion_status(
            pending.id,
            status="superseded",
        )
    if duplicate_pending is not None:
        log_event(
            "context_update.suggestion_suppressed",
            save_id=save_id,
            entity_type="world_state",
            entity_id=entity_id,
            field_path=key,
            suggestion_id=duplicate_pending.id,
            reason="duplicate_pending",
        )
        return None
    if state_already_current:
        log_event(
            "context_update.suggestion_suppressed",
            save_id=save_id,
            entity_type="world_state",
            entity_id=entity_id,
            field_path=key,
            reason="already_applied",
        )
        return None

    source_message_ids = [source_message_id]
    proposed_value = {
        "operation": "upsert",
        "key": key,
        "value": value,
        "category": "scene",
        "confidence": confidence,
        "source_message_id": source_message_id,
    }
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="upsert",
        entity_type="world_state",
        entity_id=entity_id,
        field_path=key,
        proposed_value=proposed_value,
        status="pending",
        reason=reason,
        confidence=confidence,
        source_message_ids=source_message_ids,
    )
    audit = repositories.add_context_update_audit(
        save_id=save_id,
        suggestion_id=suggestion.id,
        operation="queued",
        entity_type="world_state",
        entity_id=entity_id,
        field_path=key,
        before=before.value if before is not None else None,
        after=proposed_value,
        reason=reason,
        confidence=confidence,
        source_message_ids=source_message_ids,
    )
    return AppliedFocusedSceneMaintenance(
        suggestions=(suggestion,),
        audit_entries=(audit,),
    )


def _world_state_suggestion_value(
    proposed_value: object,
    key: str,
) -> object:
    if not isinstance(proposed_value, dict):
        return proposed_value
    if str(proposed_value.get("operation", "upsert")).strip() != "upsert":
        return proposed_value
    if str(proposed_value.get("key", key)).strip() != key:
        return proposed_value
    return proposed_value.get("value")


def _find_world_state_by_key(
    records: Iterable[WorldStateRecord],
    key: str,
) -> WorldStateRecord | None:
    for record in records:
        if record.key == key:
            return record
    return None


def _character_emotion_state_key(character: CharacterRecord) -> str:
    slug = _continuity_key_slug(character.name)
    if not slug:
        return ""
    return f"character.{slug}.current_emotional_state"


def _continuity_key_slug(value: str) -> str:
    parts: list[str] = []
    current: list[str] = []
    for char in value.casefold():
        if char.isalnum():
            current.append(char)
            continue
        if current:
            parts.append("".join(current))
            current.clear()
    if current:
        parts.append("".join(current))
    return "_".join(parts)


def _dump_json_compact(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass
class _SaveWorldDataSnapshot:
    repositories: PersistenceRepositories
    save_id: str
    locations: list[LocationRecord] = field(default_factory=list)
    characters: list[CharacterRecord] = field(default_factory=list)
    active_threads: list[ActiveThreadRecord] = field(default_factory=list)
    entity_links: list[EntityLinkRecord] = field(default_factory=list)
    memories: list[MemoryRecord] = field(default_factory=list)
    world_state: list[WorldStateRecord] = field(default_factory=list)
    summaries: list[SummaryRecord] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        repositories: PersistenceRepositories,
        save_id: str,
    ) -> _SaveWorldDataSnapshot:
        return cls(
            repositories=repositories,
            save_id=save_id,
            locations=repositories.list_locations(save_id),
            characters=repositories.list_characters(save_id),
            active_threads=repositories.list_active_threads(save_id),
            entity_links=repositories.list_entity_links(save_id),
            memories=repositories.list_memories(save_id),
            world_state=repositories.list_world_state(save_id),
            summaries=repositories.list_summaries(save_id),
        )

    def refresh_world_state(self) -> None:
        self.world_state = self.repositories.list_world_state(self.save_id)

    def upsert_location(self, location: LocationRecord) -> None:
        _append_unique(self.locations, location)

    def upsert_character(self, character: CharacterRecord) -> None:
        _append_unique(self.characters, character)

    def upsert_active_thread(self, thread: ActiveThreadRecord) -> None:
        _append_unique(self.active_threads, thread)

    def remove_active_thread(self, thread_id: str) -> None:
        self.active_threads = [
            thread for thread in self.active_threads if thread.id != thread_id
        ]

    def upsert_entity_link(self, link: EntityLinkRecord) -> None:
        _append_unique(self.entity_links, link)


@dataclass
class _ContextUpdateApplier:
    repositories: PersistenceRepositories
    save_id: str
    completed_messages: tuple[MessageRecord, ...] = ()
    scene_snapshot: SceneSnapshotRecord | None = None
    locations: list[LocationRecord] = field(default_factory=list)
    characters: list[CharacterRecord] = field(default_factory=list)
    active_threads: list[ActiveThreadRecord] = field(default_factory=list)
    entity_links: list[EntityLinkRecord] = field(default_factory=list)
    suggestions: list[ContextUpdateSuggestionRecord] = field(default_factory=list)
    audit_entries: list[ContextUpdateAuditRecord] = field(default_factory=list)
    queued_character_keys: set[str] = field(default_factory=set)
    scene_local_thread_ids_before_scene: set[str] = field(default_factory=set)
    touched_thread_ids: set[str] = field(default_factory=set)
    applied_suggestion_keys: set[tuple[str, str | None, str]] = field(
        default_factory=set
    )
    snapshot: _SaveWorldDataSnapshot = field(init=False)
    storyteller_mode: bool = field(init=False)

    def __post_init__(self) -> None:
        self.snapshot = _SaveWorldDataSnapshot.load(self.repositories, self.save_id)
        save = self.repositories.get_save(self.save_id)
        self.storyteller_mode = (
            save is not None
            and save.interaction_mode is InteractionMode.STORYTELLER
        )

    def apply(self, extraction: ContextUpdateExtraction) -> None:
        confirm_new_characters = manual_character_registry_confirmation_enabled(
            self.repositories,
            save_id=self.save_id,
        )
        self.scene_local_thread_ids_before_scene = {
            thread.id
            for thread in self.snapshot.active_threads
            if active_thread_is_scene_local(thread)
        }
        for location in extraction.locations:
            if not location.name.strip():
                continue
            self._apply_location(location)
        for character in extraction.characters:
            if not character.name.strip():
                continue
            if _is_probable_opaque_identifier(character.name):
                continue
            resolution = _resolve_character(
                self.snapshot.characters,
                character.name,
            )
            if resolution.ambiguous:
                continue
            if resolution.record is None and confirm_new_characters:
                self._queue_character_confirmation(character)
                continue
            self._apply_character(character)
        for thread in extraction.active_threads:
            self._apply_thread(thread)
        if self.touched_thread_ids or self.snapshot.active_threads:
            archive_open_thread_aggregate_state(self.repositories, self.save_id)
            self.snapshot.refresh_world_state()
        if extraction.scene is not None:
            self._apply_scene(extraction.scene)
        for link in extraction.entity_links:
            self._apply_entity_link(link)
        if not self.storyteller_mode:
            for exchange in extraction.phone_number_exchanges:
                self._apply_phone_number_exchange(exchange)

    def apply_world_data_enrichment(self, enrichment: WorldDataEnrichment) -> None:
        for location in enrichment.locations:
            self._apply_location_enrichment(location)
        for thread in enrichment.active_threads:
            self._apply_thread_enrichment(thread)
        for character in enrichment.characters:
            self._apply_character_enrichment(character)

    def result(self) -> AppliedContextUpdate:
        return AppliedContextUpdate(
            scene_snapshot=self.scene_snapshot,
            locations=tuple(self.locations),
            characters=tuple(self.characters),
            active_threads=tuple(self.active_threads),
            entity_links=tuple(self.entity_links),
            suggestions=tuple(self.suggestions),
            audit_entries=tuple(self.audit_entries),
        )

    def _apply_location(self, extracted: ExtractedLocation) -> LocationRecord:
        existing = _find_location(
            self.snapshot.locations,
            extracted.name,
        )
        parent = _find_location(
            self.snapshot.locations,
            extracted.parent_location_name,
        )
        parent_id = parent.id if parent else None
        if existing is None:
            location = self.repositories.add_location(
                save_id=self.save_id,
                name=extracted.name.strip(),
                aliases=list(_clean_strings(extracted.aliases)),
                description=extracted.description.strip(),
                visual_description=extracted.visual_description.strip(),
                parent_location_id=parent_id,
                connections=list(_clean_strings(extracted.connections)),
                status=extracted.status.strip(),
                hazards=list(_clean_strings(extracted.hazards)),
                source_message_id=extracted.source_message_id,
            )
            self._record_applied(
                operation="created",
                entity_type="location",
                entity_id=location.id,
                field_path="*",
                before=None,
                after=_location_audit_value(location),
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_ids=[extracted.source_message_id],
            )
            self._append_location(location)
            self.snapshot.upsert_location(location)
            return location

        location = existing
        aliases = extracted.aliases
        if _should_add_location_alias(location, extracted.name):
            aliases = (*aliases, extracted.name.strip())
        for field_path, value in (
            ("aliases", _merge_strings(location.aliases, aliases)),
            ("description", extracted.description.strip()),
            ("visual_description", extracted.visual_description.strip()),
            ("parent_location_id", parent_id),
            (
                "connections",
                _merge_strings(location.connections, extracted.connections),
            ),
            ("status", extracted.status.strip()),
            ("hazards", _merge_strings(location.hazards, extracted.hazards)),
        ):
            location = self._apply_field(
                record=location,
                entity_type="location",
                entity_id=location.id,
                field_path=field_path,
                value=value,
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_id=extracted.source_message_id,
                update=self.repositories.update_location,
            )
        self._append_location(location)
        self.snapshot.upsert_location(location)
        return location

    def _apply_location_enrichment(
        self,
        enrichment: LocationWorldDataEnrichment,
    ) -> LocationRecord | None:
        location = self.repositories.get_location(enrichment.location_id.strip())
        if location is None or location.save_id != self.save_id:
            return None
        for field_path, value in (
            ("description", enrichment.description.strip()),
            ("visual_description", enrichment.visual_description.strip()),
            (
                "connections",
                _merge_strings(location.connections, enrichment.connections),
            ),
            ("status", enrichment.status.strip()),
            ("hazards", _merge_strings(location.hazards, enrichment.hazards)),
        ):
            if not _field_is_blank_and_unlocked(location, field_path):
                continue
            location = self._apply_field(
                record=location,
                entity_type="location",
                entity_id=location.id,
                field_path=field_path,
                value=value,
                reason=enrichment.reason,
                confidence=enrichment.confidence,
                source_message_id=enrichment.source_message_id,
                update=self.repositories.update_location,
            )
        self._append_location(location)
        self.snapshot.upsert_location(location)
        return location

    def _apply_character(self, extracted: ExtractedCharacter) -> CharacterRecord:
        source_message = self.repositories.get_message(
            save_id=self.save_id,
            message_id=extracted.source_message_id,
        )
        source_content_rating = (
            source_message.content_rating
            if source_message is not None
            else "unclassified"
        )
        resolution = _resolve_character(
            self.snapshot.characters,
            extracted.name,
        )
        existing = resolution.record
        location = _find_location(
            self.snapshot.locations,
            extracted.location_name,
        )
        location_id = location.id if location else None
        relationships = dict(extracted.relationships or {})
        if existing is None:
            character = self.repositories.add_character(
                save_id=self.save_id,
                name=extracted.name.strip(),
                aliases=list(_clean_strings(extracted.aliases)),
                role=extracted.role.strip(),
                age=extracted.age.strip(),
                known_state=extracted.known_state.strip(),
                met=bool(extracted.met),
                appearance=extracted.appearance.strip(),
                visual_notes=extracted.visual_notes.strip(),
                current_clothing=extracted.current_clothing.strip(),
                personality=extracted.personality.strip(),
                voice=extracted.voice.strip(),
                relationships=relationships,
                goals=extracted.goals.strip(),
                motivations=extracted.motivations.strip(),
                current_intent=extracted.current_intent.strip(),
                boundaries=extracted.boundaries.strip(),
                attitude_toward_player=(
                    ""
                    if self.storyteller_mode
                    else extracted.attitude_toward_player.strip()
                ),
                cooperation_conditions=extracted.cooperation_conditions.strip(),
                status=extracted.status.strip(),
                location_id=location_id,
                private_notes=extracted.private_notes.strip(),
                source_message_id=extracted.source_message_id,
                content_rating=source_content_rating,
            )
            self._record_applied(
                operation="created",
                entity_type="character",
                entity_id=character.id,
                field_path="*",
                before=None,
                after=_character_audit_value(character),
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_ids=[extracted.source_message_id],
            )
            self._append_character(character)
            self.snapshot.upsert_character(character)
            return character

        character = existing
        aliases = extracted.aliases
        if _should_add_character_alias(character, extracted.name):
            aliases = (*aliases, extracted.name.strip())
        field_values: list[tuple[str, object]] = [
            ("aliases", _merge_strings(character.aliases, aliases)),
            ("role", extracted.role.strip()),
            ("age", extracted.age.strip()),
            ("known_state", extracted.known_state.strip()),
            ("appearance", extracted.appearance.strip()),
            ("visual_notes", extracted.visual_notes.strip()),
            ("current_clothing", extracted.current_clothing.strip()),
            ("personality", extracted.personality.strip()),
            ("voice", extracted.voice.strip()),
            ("relationships", {**character.relationships, **relationships}),
            ("goals", extracted.goals.strip()),
            ("motivations", extracted.motivations.strip()),
            ("current_intent", extracted.current_intent.strip()),
            ("boundaries", extracted.boundaries.strip()),
            (
                "cooperation_conditions",
                extracted.cooperation_conditions.strip(),
            ),
            ("status", extracted.status.strip()),
            ("location_id", location_id),
            ("private_notes", extracted.private_notes.strip()),
        ]
        if not self.storyteller_mode:
            field_values.append(
                (
                    "attitude_toward_player",
                    extracted.attitude_toward_player.strip(),
                )
            )
        if extracted.met is not None:
            field_values.append(("met", extracted.met))
        for field_path, value in field_values:
            character = self._apply_field(
                record=character,
                entity_type="character",
                entity_id=character.id,
                field_path=field_path,
                value=value,
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_id=extracted.source_message_id,
                update=self.repositories.update_character,
            )
        combined_rating = maximum_content_rating(
            (character.content_rating, source_content_rating)
        )
        if combined_rating != character.content_rating:
            character = self.repositories.update_character(
                replace(character, content_rating=combined_rating)
            )
        self._append_character(character)
        self.snapshot.upsert_character(character)
        return character

    def _queue_character_confirmation(self, extracted: ExtractedCharacter) -> None:
        character_key = _character_name_key(extracted.name)
        if not character_key or character_key in self.queued_character_keys:
            return
        location = _find_location(
            self.snapshot.locations,
            extracted.location_name,
        )
        proposed_value = {
            "name": extracted.name.strip(),
            "aliases": list(_clean_strings(extracted.aliases)),
            "role": extracted.role.strip(),
            "known_state": extracted.known_state.strip(),
            "met": bool(extracted.met),
            "appearance": extracted.appearance.strip(),
            "visual_notes": extracted.visual_notes.strip(),
            "current_clothing": extracted.current_clothing.strip(),
            "personality": extracted.personality.strip(),
            "voice": extracted.voice.strip(),
            "relationships": dict(extracted.relationships or {}),
            "status": extracted.status.strip(),
            "location_id": location.id if location else None,
            "private_notes": extracted.private_notes.strip(),
            "source_message_id": extracted.source_message_id,
        }
        if extracted.age.strip():
            proposed_value["age"] = extracted.age.strip()
        for field_name in sorted(CHARACTER_AGENCY_FIELDS):
            value = getattr(extracted, field_name).strip()
            if value:
                proposed_value[field_name] = value
        self._queue_suggestion(
            update_type="create",
            entity_type="character",
            entity_id=None,
            field_path="*",
            before=None,
            after=proposed_value,
            reason=extracted.reason,
            confidence=extracted.confidence,
            source_message_ids=[extracted.source_message_id],
        )
        self.queued_character_keys.add(character_key)

    def _apply_thread(
        self, extracted: ExtractedActiveThread
    ) -> ActiveThreadRecord | None:
        existing = _find_thread(
            self.snapshot.active_threads,
            extracted.title,
        )
        raw_status = extracted.status.strip()
        raw_visibility = extracted.visibility.strip()
        normalized_status = normalize_active_thread_status(raw_status)
        normalized_visibility = normalize_active_thread_visibility(raw_visibility)
        if existing is None:
            if raw_status and not active_thread_status_is_open(normalized_status):
                return None
            thread = self.repositories.add_active_thread(
                save_id=self.save_id,
                title=extracted.title.strip(),
                description=extracted.description.strip(),
                status=normalized_status,
                priority=extracted.priority or 0,
                visibility=normalized_visibility,
                related_entities=list(_clean_strings(extracted.related_entities)),
                source_message_id=extracted.source_message_id,
            )
            self.touched_thread_ids.add(thread.id)
            self._record_applied(
                operation="created",
                entity_type="active_thread",
                entity_id=thread.id,
                field_path="*",
                before=None,
                after=_thread_audit_value(thread),
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_ids=[extracted.source_message_id],
            )
            self._append_thread(thread)
            self.snapshot.upsert_active_thread(thread)
            return thread

        thread = self._normalize_thread(existing, extracted)
        self.touched_thread_ids.add(thread.id)
        field_values: list[tuple[str, object]] = [
            ("description", extracted.description.strip()),
            ("status", normalized_status if raw_status else ""),
            ("visibility", normalized_visibility if raw_visibility else ""),
            (
                "related_entities",
                _merge_strings(thread.related_entities, extracted.related_entities),
            ),
        ]
        if extracted.priority is not None:
            field_values.append(("priority", extracted.priority))
        for field_path, value in field_values:
            thread = self._apply_field(
                record=thread,
                entity_type="active_thread",
                entity_id=thread.id,
                field_path=field_path,
                value=value,
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_id=extracted.source_message_id,
                update=self.repositories.update_active_thread,
            )
        if raw_status and not active_thread_status_is_open(thread.status):
            self._archive_thread(
                thread,
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_id=extracted.source_message_id,
            )
            return None
        self._append_thread(thread)
        self.snapshot.upsert_active_thread(thread)
        return thread

    def _normalize_thread(
        self,
        thread: ActiveThreadRecord,
        extracted: ExtractedActiveThread,
    ) -> ActiveThreadRecord:
        normalized = normalize_active_thread_record(thread)
        if normalized == thread:
            return thread
        saved = self.repositories.update_active_thread(
            replace(
                normalized,
                source_message_id=extracted.source_message_id,
                last_updated_message_id=extracted.source_message_id,
            )
        )
        self.snapshot.upsert_active_thread(saved)
        if thread.status != normalized.status:
            self._record_applied(
                operation="updated",
                entity_type="active_thread",
                entity_id=thread.id,
                field_path="status",
                before=thread.status,
                after=normalized.status,
                reason=extracted.reason or "Normalized active thread status.",
                confidence=extracted.confidence,
                source_message_ids=[extracted.source_message_id],
            )
        if thread.visibility != normalized.visibility:
            self._record_applied(
                operation="updated",
                entity_type="active_thread",
                entity_id=thread.id,
                field_path="visibility",
                before=thread.visibility,
                after=normalized.visibility,
                reason=extracted.reason or "Normalized active thread visibility.",
                confidence=extracted.confidence,
                source_message_ids=[extracted.source_message_id],
            )
        return saved

    def _archive_thread(
        self,
        thread: ActiveThreadRecord,
        *,
        reason: str,
        confidence: float,
        source_message_id: str,
    ) -> None:
        self.repositories.archive_active_thread(thread.id)
        self.snapshot.remove_active_thread(thread.id)
        self.active_threads = [
            active_thread
            for active_thread in self.active_threads
            if active_thread.id != thread.id
        ]
        self._record_applied(
            operation="archived",
            entity_type="active_thread",
            entity_id=thread.id,
            field_path="*",
            before=_thread_audit_value(thread),
            after=None,
            reason=reason,
            confidence=confidence,
            source_message_ids=[source_message_id],
        )

    def _apply_thread_enrichment(
        self,
        enrichment: ActiveThreadWorldDataEnrichment,
    ) -> ActiveThreadRecord | None:
        thread = self.repositories.get_active_thread(
            enrichment.active_thread_id.strip()
        )
        if thread is None or thread.save_id != self.save_id:
            return None
        for field_path, value in (
            ("description", enrichment.description.strip()),
            (
                "related_entities",
                _merge_strings(thread.related_entities, enrichment.related_entities),
            ),
        ):
            if not _field_is_blank_and_unlocked(thread, field_path):
                continue
            thread = self._apply_field(
                record=thread,
                entity_type="active_thread",
                entity_id=thread.id,
                field_path=field_path,
                value=value,
                reason=enrichment.reason,
                confidence=enrichment.confidence,
                source_message_id=enrichment.source_message_id,
                update=self.repositories.update_active_thread,
            )
        self._append_thread(thread)
        self.snapshot.upsert_active_thread(thread)
        return thread

    def _apply_character_enrichment(
        self,
        enrichment: CharacterWorldDataEnrichment,
    ) -> CharacterRecord | None:
        character = self.repositories.get_character(enrichment.character_id.strip())
        if character is None or character.save_id != self.save_id:
            return None
        for field_path, value in (
            ("aliases", _merge_strings(character.aliases, enrichment.aliases)),
            ("role", enrichment.role.strip()),
            ("age", enrichment.age.strip()),
            ("known_state", enrichment.known_state.strip()),
            ("appearance", enrichment.appearance.strip()),
            ("visual_notes", enrichment.visual_notes.strip()),
            ("current_clothing", enrichment.current_clothing.strip()),
            ("personality", enrichment.personality.strip()),
            ("voice", enrichment.voice.strip()),
            (
                "relationships",
                {
                    **character.relationships,
                    **dict(enrichment.relationships or {}),
                },
            ),
            ("status", enrichment.status.strip()),
        ):
            if not _field_is_blank_and_unlocked(character, field_path):
                continue
            character = self._apply_field(
                record=character,
                entity_type="character",
                entity_id=character.id,
                field_path=field_path,
                value=value,
                reason=enrichment.reason,
                confidence=enrichment.confidence,
                source_message_id=enrichment.source_message_id,
                update=self.repositories.update_character,
            )
        self._append_character(character)
        self.snapshot.upsert_character(character)
        return character

    def _apply_scene(self, extracted: ExtractedSceneSnapshot) -> None:
        current_location = self._ensure_location(
            extracted.current_location_name,
            source_message_id=extracted.source_message_id,
            reason=extracted.reason,
            confidence=extracted.confidence,
        )
        present_character_ids: list[str] | None = None
        if extracted.present_character_names is not None:
            present_character_names = _clean_strings(extracted.present_character_names)
            present_character_ids = [
                character.id
                for name in present_character_names
                if (
                    character := self._ensure_character(
                        name,
                        source_message_id=extracted.source_message_id,
                        reason=extracted.reason,
                        confidence=extracted.confidence,
                    )
                )
                is not None
            ]
        existing = self.repositories.get_scene_snapshot(self.save_id)
        previous_location_id = existing.current_location_id if existing else None
        normalized_time = _normalize_scene_time(extracted.in_world_time)
        if existing is None:
            snapshot = self.repositories.upsert_scene_snapshot(
                save_id=self.save_id,
                current_location_id=current_location.id if current_location else None,
                situation=extracted.situation.strip(),
                objective=extracted.objective.strip(),
                in_world_time=normalized_time,
                world_day_index=0,
                weather=extracted.weather.strip(),
                mood=extracted.mood.strip(),
                nearby_objects=list(_clean_strings(extracted.nearby_objects or ())),
                hazards=list(_clean_strings(extracted.hazards or ())),
                present_character_ids=present_character_ids or [],
                source_message_id=extracted.source_message_id,
            )
            from bragi.services.time_loop_time_policy import TimeLoopTimePolicy

            loop_policy = TimeLoopTimePolicy(
                self.repositories,
                save_id=self.save_id,
            )
            loop_policy.ensure_baseline(snapshot)
            loop_policy.sync_current(
                snapshot,
                transition="context_scene_update",
                source_message_id=extracted.source_message_id,
            )
            self._record_applied(
                operation="created",
                entity_type="scene_snapshot",
                entity_id=snapshot.id,
                field_path="*",
                before=None,
                after=_scene_audit_value(snapshot),
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_ids=[extracted.source_message_id],
            )
            self.scene_snapshot = snapshot
            return

        next_location_id = current_location.id if current_location is not None else None
        generation_before = existing.scene_generation
        snapshot = existing
        if next_location_id is not None:
            snapshot = self._apply_scene_field(
                snapshot=snapshot,
                field_path="current_location_id",
                value=next_location_id,
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_id=extracted.source_message_id,
            )
        if (
            extracted.scene_transition
            and snapshot.scene_generation == generation_before
        ):
            existing = self.repositories.advance_scene_generation(
                save_id=self.save_id,
                source_message_id=extracted.source_message_id,
            )
            snapshot = existing
            self._record_applied(
                operation="scene_generation_advanced",
                entity_type="scene_snapshot",
                entity_id=snapshot.id,
                field_path="scene_generation",
                before=generation_before,
                after=snapshot.scene_generation,
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_ids=[extracted.source_message_id],
            )
        location_changed = snapshot.current_location_id != previous_location_id
        scene_updates: list[tuple[str, object]] = [
            ("situation", extracted.situation.strip()),
            ("objective", extracted.objective.strip()),
            ("in_world_time", normalized_time),
            ("weather", extracted.weather.strip()),
            ("mood", extracted.mood.strip()),
        ]
        if extracted.nearby_objects is not None:
            scene_updates.append(
                ("nearby_objects", list(_clean_strings(extracted.nearby_objects)))
            )
        elif location_changed:
            scene_updates.append(("nearby_objects", []))
        if extracted.hazards is not None:
            scene_updates.append(("hazards", list(_clean_strings(extracted.hazards))))
        elif location_changed:
            scene_updates.append(("hazards", []))
        if present_character_ids is not None:
            scene_updates.append(("present_character_ids", present_character_ids))
        elif location_changed:
            scene_updates.append(("present_character_ids", []))
        for field_path, value in scene_updates:
            snapshot = self._apply_scene_field(
                snapshot=snapshot,
                field_path=field_path,
                value=value,
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_id=extracted.source_message_id,
            )
        self.scene_snapshot = snapshot
        self._archive_scene_local_threads_after_scene_change(
            previous_location_id=previous_location_id,
            current_location_id=snapshot.current_location_id,
            scene_transition=extracted.scene_transition,
            reason=extracted.reason,
            confidence=extracted.confidence,
            source_message_id=extracted.source_message_id,
        )

    def _archive_scene_local_threads_after_scene_change(
        self,
        *,
        previous_location_id: str | None,
        current_location_id: str | None,
        scene_transition: bool = False,
        reason: str,
        confidence: float,
        source_message_id: str,
    ) -> None:
        if previous_location_id == current_location_id and not scene_transition:
            return
        for thread in tuple(self.snapshot.active_threads):
            if thread.id not in self.scene_local_thread_ids_before_scene:
                continue
            if thread.id in self.touched_thread_ids:
                continue
            if not active_thread_is_scene_local(thread):
                continue
            self._archive_thread(
                thread,
                reason=reason
                or "Scene-local active thread expired after the scene changed.",
                confidence=confidence,
                source_message_id=source_message_id,
            )

    def _apply_entity_link(self, extracted: ExtractedEntityLink) -> None:
        entity_type = _normalized_entity_type(extracted.entity_type)
        target_type = _normalized_entity_type(extracted.target_type)
        if entity_type not in _LINKABLE_ENTITY_TYPES:
            return
        if target_type not in _LINKABLE_ENTITY_TYPES:
            return
        if not (
            extracted.entity_id.strip() or extracted.entity_name.strip()
        ) or not (extracted.target_id.strip() or extracted.target_name.strip()):
            return
        entity_id = self._validated_or_resolved_entity_id(
            entity_type=entity_type,
            entity_id=extracted.entity_id,
            name=extracted.entity_name,
            source_message_id=extracted.source_message_id,
            reason=extracted.reason,
            confidence=extracted.confidence,
        )
        target_id = self._validated_or_resolved_entity_id(
            entity_type=target_type,
            entity_id=extracted.target_id,
            name=extracted.target_name,
            source_message_id=extracted.source_message_id,
            reason=extracted.reason,
            confidence=extracted.confidence,
        )
        if not entity_id or not target_id:
            return
        existing_link = _find_entity_link(
            self.snapshot.entity_links,
            entity_type=entity_type,
            entity_id=entity_id,
            target_type=target_type,
            target_id=target_id,
            relation=extracted.relation.strip(),
        )
        link = self.repositories.add_entity_link(
            save_id=self.save_id,
            entity_type=entity_type,
            entity_id=entity_id,
            target_type=target_type,
            target_id=target_id,
            relation=extracted.relation.strip(),
            source_message_id=extracted.source_message_id,
        )
        if existing_link is None and not any(
            existing.id == link.id for existing in self.entity_links
        ):
            self._record_applied(
                operation="created",
                entity_type="entity_link",
                entity_id=link.id,
                field_path="*",
                before=None,
                after={
                    "entity_type": link.entity_type,
                    "entity_id": link.entity_id,
                    "target_type": link.target_type,
                    "target_id": link.target_id,
                    "relation": link.relation,
                },
                reason=extracted.reason,
                confidence=extracted.confidence,
                source_message_ids=[extracted.source_message_id],
            )
        self.entity_links.append(link)
        self.snapshot.upsert_entity_link(link)

    def _apply_phone_number_exchange(
        self,
        extracted: ExtractedPhoneNumberExchange,
    ) -> None:
        player = next(
            (
                character
                for character in self.snapshot.characters
                if character.is_player_character
            ),
            None,
        )
        if player is None:
            return
        character = next(
            (
                character
                for character in self.snapshot.characters
                if character.id == extracted.character_id
                and not character.is_player_character
            ),
            None,
        )
        if character is None:
            return
        direction = extracted.direction.strip()
        self.repositories.upsert_character_contact_state(
            save_id=self.save_id,
            player_character_id=player.id,
            character_id=character.id,
            player_has_character_number=direction
            in {PHONE_EXCHANGE_PLAYER_HAS_CHARACTER_NUMBER, PHONE_EXCHANGE_BOTH},
            character_has_player_number=direction
            in {PHONE_EXCHANGE_CHARACTER_HAS_PLAYER_NUMBER, PHONE_EXCHANGE_BOTH},
            source_message_id=extracted.source_message_id,
        )

    def _validated_or_resolved_entity_id(
        self,
        *,
        entity_type: str,
        entity_id: str,
        name: str,
        source_message_id: str,
        reason: str,
        confidence: float,
    ) -> str | None:
        normalized = _normalized_entity_type(entity_type)
        entity_id = entity_id.strip()
        if entity_id:
            if self._entity_id_exists(entity_type=normalized, entity_id=entity_id):
                return entity_id
            return self._resolve_entity_id(
                entity_type=normalized,
                name=name,
                source_message_id=source_message_id,
                reason=reason,
                confidence=confidence,
            )
        return self._resolve_entity_id(
            entity_type=normalized,
            name=name,
            source_message_id=source_message_id,
            reason=reason,
            confidence=confidence,
        )

    def _entity_id_exists(self, *, entity_type: str, entity_id: str) -> bool:
        if entity_type == "location":
            location = self.repositories.get_location(entity_id)
            return location is not None and location.save_id == self.save_id
        if entity_type == "character":
            character = self.repositories.get_character(entity_id)
            return character is not None and character.save_id == self.save_id
        if entity_type == "active_thread":
            thread = self.repositories.get_active_thread(entity_id)
            return thread is not None and thread.save_id == self.save_id
        if entity_type == "memory":
            return any(
                memory.id == entity_id
                for memory in self.snapshot.memories
            )
        if entity_type == "world_state":
            return any(
                state.id == entity_id
                for state in self.snapshot.world_state
            )
        if entity_type == "summary":
            return any(
                summary.id == entity_id
                for summary in self.snapshot.summaries
            )
        return False

    def _apply_field(
        self,
        *,
        record: Any,
        entity_type: str,
        entity_id: str,
        field_path: str,
        value: object,
        reason: str,
        confidence: float,
        source_message_id: str,
        update: Callable[[Any], Any],
    ) -> Any:
        before = getattr(record, field_path)
        if _is_empty_update(value) or before == value:
            return record
        if _is_append_only_character_history_field(entity_type, field_path):
            value = _append_character_history(before, value)
            if before == value:
                return record
        field_is_locked = _record_field_is_locked(
            entity_type,
            record,
            field_path,
        )
        if field_is_locked or (
            _is_conflicting_rewrite(before, value)
            and not _is_volatile_context_field(entity_type, field_path)
            and not _is_append_only_character_history_field(entity_type, field_path)
        ):
            self._queue_suggestion(
                update_type="update",
                entity_type=entity_type,
                entity_id=entity_id,
                field_path=field_path,
                before=before,
                after=value,
                reason=reason,
                confidence=confidence,
                source_message_ids=[source_message_id],
            )
            return record
        saved = update(
            replace(
                record,
                **{
                    field_path: value,
                    "source_message_id": source_message_id,
                    "last_updated_message_id": source_message_id,
                },
            )
        )
        if entity_type == "location":
            self.snapshot.upsert_location(cast(LocationRecord, saved))
        elif entity_type == "character":
            self.snapshot.upsert_character(cast(CharacterRecord, saved))
        elif entity_type == "active_thread":
            self.snapshot.upsert_active_thread(cast(ActiveThreadRecord, saved))
        self._record_applied(
            operation="updated",
            entity_type=entity_type,
            entity_id=entity_id,
            field_path=field_path,
            before=before,
            after=value,
            reason=reason,
            confidence=confidence,
            source_message_ids=[source_message_id],
        )
        return saved

    def _apply_scene_field(
        self,
        *,
        snapshot: SceneSnapshotRecord,
        field_path: str,
        value: object,
        reason: str,
        confidence: float,
        source_message_id: str,
    ) -> SceneSnapshotRecord:
        if field_path in _SCENE_WORLD_TIME_FIELDS:
            value = _normalize_scene_time(value)
        before = getattr(snapshot, field_path)
        if before == value or (
            not _scene_field_allows_empty_replacement(field_path)
            and _is_empty_update(value)
        ):
            return snapshot
        field_is_locked = _scene_field_is_locked(snapshot, field_path)
        if field_path == "in_world_time" and (
            field_is_locked
            or (
                not _is_empty_update(before)
                and not _player_authorizes_scene_time_change(self.completed_messages)
            )
        ):
            self._queue_suggestion(
                update_type="update",
                entity_type="scene_snapshot",
                entity_id=snapshot.id,
                field_path=field_path,
                before=before,
                after=value,
                reason=reason,
                confidence=confidence,
                source_message_ids=[source_message_id],
            )
            return snapshot
        if field_is_locked or (
            _is_conflicting_rewrite(before, value)
            and not _is_volatile_context_field("scene_snapshot", field_path)
        ):
            self._queue_suggestion(
                update_type="update",
                entity_type="scene_snapshot",
                entity_id=snapshot.id,
                field_path=field_path,
                before=before,
                after=value,
                reason=reason,
                confidence=confidence,
                source_message_ids=[source_message_id],
            )
            return snapshot
        updated = _replace_scene_snapshot_field(snapshot, field_path, value)
        world_time_kwargs: dict[str, Any] = {}
        if field_path in {
            "in_world_time",
            "time_of_day",
            "day_of_week",
            "world_day_index",
        }:
            canonical_world_time = canonical_world_time_from_legacy(
                in_world_time=updated.in_world_time,
                time_of_day=(
                    "" if field_path == "in_world_time" else updated.time_of_day
                ),
                day_of_week=updated.day_of_week,
                world_day_index=updated.world_day_index,
                source_message_id=source_message_id,
                confidence=confidence,
            )
            world_time_kwargs = {
                "world_time_day_index": canonical_world_time.day_index,
                "world_time_source_message_id": (
                    canonical_world_time.source_message_id
                ),
                "world_time_confidence": canonical_world_time.confidence,
            }
            if field_path != "world_day_index":
                world_time_kwargs.update(
                    {
                        "world_time_day_label": canonical_world_time.day_label,
                        "world_time_phase": canonical_world_time.phase,
                        "world_time_clock_minutes": (
                            canonical_world_time.clock_minutes
                            if canonical_world_time.clock_minutes is not None
                            else snapshot.world_time_clock_minutes
                        ),
                        "world_time_period_label": (
                            canonical_world_time.period_label
                            or snapshot.world_time_period_label
                        ),
                    }
                )
            if field_path == "in_world_time":
                display_world_time = canonical_world_time_from_values(
                    day_index=canonical_world_time.day_index,
                    day_label=canonical_world_time.day_label,
                    phase=canonical_world_time.phase,
                    clock_minutes=(
                        canonical_world_time.clock_minutes
                        if canonical_world_time.clock_minutes is not None
                        else snapshot.world_time_clock_minutes
                    ),
                    period_label=(
                        canonical_world_time.period_label
                        or snapshot.world_time_period_label
                    ),
                    source_message_id=canonical_world_time.source_message_id,
                    confidence=canonical_world_time.confidence,
                    legacy_in_world_time=updated.in_world_time,
                    legacy_time_of_day=updated.time_of_day,
                    legacy_day_of_week=updated.day_of_week,
                    legacy_world_day_index=updated.world_day_index,
                )
                legacy_fields = legacy_world_time_fields(display_world_time)
                updated = replace(
                    updated,
                    in_world_time=cast(str, legacy_fields["in_world_time"]),
                    time_of_day=cast(str, legacy_fields["time_of_day"]),
                    day_of_week=cast(str, legacy_fields["day_of_week"]),
                    world_day_index=cast(
                        int | None,
                        legacy_fields["world_day_index"],
                    ),
                )
        if field_path in _SCENE_WORLD_TIME_FIELDS:
            from bragi.services.time_loop_time_policy import TimeLoopTimePolicy

            loop_policy = TimeLoopTimePolicy(
                self.repositories,
                save_id=self.save_id,
            )
            loop_policy.ensure_baseline(snapshot)
        saved = self.repositories.upsert_scene_snapshot(
            save_id=self.save_id,
            current_location_id=updated.current_location_id,
            situation=updated.situation,
            objective=updated.objective,
            in_world_time=updated.in_world_time,
            time_of_day=updated.time_of_day,
            day_of_week=updated.day_of_week,
            world_day_index=updated.world_day_index,
            weather=updated.weather,
            mood=updated.mood,
            nearby_objects=updated.nearby_objects,
            hazards=updated.hazards,
            present_character_ids=updated.present_character_ids,
            source_message_id=source_message_id,
            locked_fields=updated.locked_fields,
            snapshot_id=updated.id,
            first_seen_message_id=snapshot.first_seen_message_id,
            last_updated_message_id=source_message_id,
            **world_time_kwargs,
        )
        if field_path == "in_world_time":
            loop_policy.ensure_baseline(saved)
            loop_policy.sync_current(
                saved,
                transition="context_scene_update",
                source_message_id=source_message_id,
            )
        self._record_applied(
            operation="updated",
            entity_type="scene_snapshot",
            entity_id=snapshot.id,
            field_path=field_path,
            before=before,
            after=value,
            reason=reason,
            confidence=confidence,
            source_message_ids=[source_message_id],
        )
        return saved

    def _ensure_location(
        self,
        name: str,
        *,
        source_message_id: str,
        reason: str,
        confidence: float,
    ) -> LocationRecord | None:
        name = name.strip()
        if not name:
            return None
        existing = _find_location(self.snapshot.locations, name)
        if existing is not None:
            return existing
        return self._apply_location(
            ExtractedLocation(
                name=name,
                source_message_id=source_message_id,
                reason=reason,
                confidence=confidence,
            )
        )

    def _ensure_character(
        self,
        name: str,
        *,
        source_message_id: str,
        reason: str,
        confidence: float,
    ) -> CharacterRecord | None:
        if not name.strip():
            return None
        resolution = _resolve_character(
            self.snapshot.characters,
            name,
        )
        existing = resolution.record
        if existing is not None:
            if _should_add_character_alias(existing, name):
                return cast(
                    CharacterRecord,
                    self._apply_field(
                        record=existing,
                        entity_type="character",
                        entity_id=existing.id,
                        field_path="aliases",
                        value=_merge_strings(existing.aliases, (name,)),
                        reason=reason,
                        confidence=confidence,
                        source_message_id=source_message_id,
                        update=self.repositories.update_character,
                    ),
                )
            return existing
        if resolution.ambiguous:
            return None
        if _is_probable_opaque_identifier(name):
            return None
        extracted = ExtractedCharacter(
            name=name,
            source_message_id=source_message_id,
            met=True,
            reason=reason,
            confidence=confidence,
        )
        if manual_character_registry_confirmation_enabled(
            self.repositories,
            save_id=self.save_id,
        ):
            self._queue_character_confirmation(extracted)
            return None
        return self._apply_character(
            extracted
        )

    def _resolve_entity_id(
        self,
        *,
        entity_type: str,
        name: str,
        source_message_id: str,
        reason: str,
        confidence: float,
    ) -> str | None:
        normalized = _normalized_entity_type(entity_type)
        if normalized == "location":
            location = _find_location(
                self.snapshot.locations,
                name,
            )
            return location.id if location is not None else None
        if normalized == "character":
            resolution = _resolve_character(
                self.snapshot.characters,
                name,
            )
            character = resolution.record
            return character.id if character is not None else None
        if normalized == "active_thread":
            thread = _find_thread(
                self.snapshot.active_threads,
                name,
            )
            return thread.id if thread else None
        if normalized == "world_state":
            return _find_world_state_id(
                self.snapshot.world_state,
                name,
            )
        if normalized == "memory":
            return _find_memory_id(self.snapshot.memories, name)
        if normalized == "summary":
            return _find_summary_id(
                self.snapshot.summaries,
                name,
            )
        return None

    def _queue_suggestion(
        self,
        *,
        update_type: str,
        entity_type: str,
        entity_id: str | None,
        field_path: str,
        before: object | None,
        after: object,
        reason: str,
        confidence: float,
        source_message_ids: list[str],
    ) -> None:
        suggestion_key = (entity_type, entity_id, field_path)
        if suggestion_key in self.applied_suggestion_keys:
            log_event(
                "context_update.suggestion_suppressed",
                save_id=self.save_id,
                entity_type=entity_type,
                entity_id=entity_id,
                field_path=field_path,
                reason="already_applied_in_cycle",
            )
            return
        existing = self.repositories.find_pending_context_update_suggestion(
            save_id=self.save_id,
            update_type=update_type,
            entity_type=entity_type,
            entity_id=entity_id,
            field_path=field_path,
            proposed_value=after,
        )
        if existing is not None:
            log_event(
                "context_update.suggestion_suppressed",
                save_id=self.save_id,
                entity_type=entity_type,
                entity_id=entity_id,
                field_path=field_path,
                suggestion_id=existing.id,
                reason="duplicate_pending",
            )
            return
        self._supersede_pending_suggestions(
            entity_type=entity_type,
            entity_id=entity_id,
            field_path=field_path,
            update_type=update_type,
        )
        suggestion = self.repositories.add_context_update_suggestion(
            save_id=self.save_id,
            update_type=update_type,
            entity_type=entity_type,
            entity_id=entity_id,
            field_path=field_path,
            proposed_value=after,
            status="pending",
            reason=reason,
            confidence=confidence,
            source_message_ids=source_message_ids,
        )
        self.suggestions.append(suggestion)
        self.audit_entries.append(
            self.repositories.add_context_update_audit(
                save_id=self.save_id,
                suggestion_id=suggestion.id,
                operation="queued",
                entity_type=entity_type,
                entity_id=entity_id,
                field_path=field_path,
                before=before,
                after=after,
                reason=reason,
                confidence=confidence,
                source_message_ids=source_message_ids,
            )
        )

    def _record_applied(
        self,
        *,
        operation: str,
        entity_type: str,
        entity_id: str | None,
        field_path: str,
        before: object | None,
        after: object | None,
        reason: str,
        confidence: float,
        source_message_ids: list[str],
    ) -> None:
        self._supersede_pending_suggestions(
            entity_type=entity_type,
            entity_id=entity_id,
            field_path=field_path,
        )
        self.applied_suggestion_keys.add((entity_type, entity_id, field_path))
        self.audit_entries.append(
            self.repositories.add_context_update_audit(
                save_id=self.save_id,
                operation=operation,
                entity_type=entity_type,
                entity_id=entity_id,
                field_path=field_path,
                before=before,
                after=after,
                reason=reason,
                confidence=confidence,
                source_message_ids=source_message_ids,
            )
        )

    def _supersede_pending_suggestions(
        self,
        *,
        entity_type: str,
        entity_id: str | None,
        field_path: str,
        update_type: str | None = None,
    ) -> None:
        for suggestion in self.repositories.list_context_update_suggestions(
            self.save_id,
            status="pending",
        ):
            if suggestion.entity_type != entity_type:
                continue
            if suggestion.entity_id != entity_id:
                continue
            if suggestion.field_path != field_path:
                continue
            if update_type is not None and suggestion.update_type != update_type:
                continue
            self.repositories.update_context_update_suggestion_status(
                suggestion.id,
                status="superseded",
            )

    def _append_location(self, location: LocationRecord) -> None:
        _append_unique(self.locations, location)

    def _append_character(self, character: CharacterRecord) -> None:
        _append_unique(self.characters, character)

    def _append_thread(self, thread: ActiveThreadRecord) -> None:
        _append_unique(self.active_threads, thread)


def context_update_extraction_from_structured_data(
    data: dict[str, object],
) -> ContextUpdateExtraction:
    scene_data = data.get("scene")
    return ContextUpdateExtraction(
        scene=(
            _scene_from_data(scene_data)
            if isinstance(scene_data, dict) and scene_data
            else None
        ),
        locations=tuple(
            _location_from_data(item) for item in _object_list(data.get("locations"))
        ),
        characters=tuple(
            _character_from_data(item) for item in _object_list(data.get("characters"))
        ),
        active_threads=tuple(
            _thread_from_data(item) for item in _object_list(data.get("active_threads"))
        ),
        entity_links=tuple(
            _entity_link_from_data(item)
            for item in _object_list(data.get("entity_links"))
        ),
        phone_number_exchanges=tuple(
            _phone_number_exchange_from_data(item)
            for item in _object_list(data.get("phone_number_exchanges"))
        ),
    )


def world_data_enrichment_from_structured_data(
    data: dict[str, object],
) -> WorldDataEnrichment:
    return WorldDataEnrichment(
        locations=tuple(
            _location_enrichment_from_data(item)
            for item in _object_list(data.get("locations"))
        ),
        active_threads=tuple(
            _thread_enrichment_from_data(item)
            for item in _object_list(data.get("active_threads"))
        ),
        characters=tuple(
            _character_enrichment_from_data(item)
            for item in _object_list(data.get("characters"))
        ),
    )


def _location_enrichment_from_data(
    value: dict[str, object],
) -> LocationWorldDataEnrichment:
    return LocationWorldDataEnrichment(
        location_id=_string(value.get("location_id")),
        source_message_id=_string(value.get("source_message_id")),
        evidence_quote=_string(value.get("evidence_quote")),
        description=_string(value.get("description")),
        visual_description=_string(value.get("visual_description")),
        connections=_string_tuple(value.get("connections")),
        status=_string(value.get("status")),
        hazards=_string_tuple(value.get("hazards")),
        reason=_string(value.get("reason")),
        confidence=_confidence(value.get("confidence")),
    )


def _thread_enrichment_from_data(
    value: dict[str, object],
) -> ActiveThreadWorldDataEnrichment:
    return ActiveThreadWorldDataEnrichment(
        active_thread_id=_string(value.get("active_thread_id")),
        source_message_id=_string(value.get("source_message_id")),
        evidence_quote=_string(value.get("evidence_quote")),
        description=_string(value.get("description")),
        related_entities=_string_tuple(value.get("related_entities")),
        reason=_string(value.get("reason")),
        confidence=_confidence(value.get("confidence")),
    )


def _character_enrichment_from_data(
    value: dict[str, object],
) -> CharacterWorldDataEnrichment:
    return CharacterWorldDataEnrichment(
        character_id=_string(value.get("character_id")),
        source_message_id=_string(value.get("source_message_id")),
        evidence_quote=_string(value.get("evidence_quote")),
        aliases=_string_tuple(value.get("aliases")),
        role=_string(value.get("role")),
        age=_string(value.get("age")),
        known_state=_string(value.get("known_state")),
        appearance=_string(value.get("appearance")),
        visual_notes=_string(value.get("visual_notes")),
        current_clothing=_string(value.get("current_clothing")),
        personality=_string(value.get("personality")),
        voice=_string(value.get("voice")),
        relationships=_object_dict(value.get("relationships")),
        status=_string(value.get("status")),
        reason=_string(value.get("reason")),
        confidence=_confidence(value.get("confidence")),
    )


def _scene_from_data(value: dict[str, object]) -> ExtractedSceneSnapshot:
    return ExtractedSceneSnapshot(
        source_message_id=_string(value.get("source_message_id")),
        evidence_quote=_string(value.get("evidence_quote")),
        current_location_name=_string(value.get("current_location_name")),
        situation=_string(value.get("situation")),
        objective=_string(value.get("objective")),
        in_world_time=_string(value.get("in_world_time")),
        weather=_string(value.get("weather")),
        mood=_string(value.get("mood")),
        nearby_objects=(
            _optional_string_tuple(value.get("nearby_objects"))
            if "nearby_objects" in value
            else None
        ),
        hazards=(
            _optional_string_tuple(value.get("hazards"))
            if "hazards" in value
            else None
        ),
        present_character_names=_optional_string_tuple(
            value.get("present_character_names")
            if "present_character_names" in value
            else None
        ),
        scene_transition=value.get("scene_transition") is True,
        reason=_string(value.get("reason")),
        confidence=_confidence(value.get("confidence")),
    )


def _location_from_data(value: dict[str, object]) -> ExtractedLocation:
    return ExtractedLocation(
        name=_string(value.get("name")),
        source_message_id=_string(value.get("source_message_id")),
        evidence_quote=_string(value.get("evidence_quote")),
        aliases=_string_tuple(value.get("aliases")),
        description=_string(value.get("description")),
        visual_description=_string(value.get("visual_description")),
        parent_location_name=_string(value.get("parent_location_name")),
        connections=_string_tuple(value.get("connections")),
        status=_string(value.get("status")),
        hazards=_string_tuple(value.get("hazards")),
        reason=_string(value.get("reason")),
        confidence=_confidence(value.get("confidence")),
    )


def _character_from_data(value: dict[str, object]) -> ExtractedCharacter:
    return ExtractedCharacter(
        name=_string(value.get("name")),
        source_message_id=_string(value.get("source_message_id")),
        evidence_quote=_string(value.get("evidence_quote")),
        aliases=_string_tuple(value.get("aliases")),
        role=_string(value.get("role")),
        age=_string(value.get("age")),
        known_state=_string(value.get("known_state")),
        met=_optional_bool(value.get("met")),
        appearance=_string(value.get("appearance")),
        visual_notes=_string(value.get("visual_notes")),
        current_clothing=_string(value.get("current_clothing")),
        personality=_string(value.get("personality")),
        voice=_string(value.get("voice")),
        relationships=_relationships_from_data(value.get("relationships")),
        goals=_string(value.get("goals")),
        motivations=_string(value.get("motivations")),
        current_intent=_string(value.get("current_intent")),
        boundaries=_string(value.get("boundaries")),
        attitude_toward_player=_string(value.get("attitude_toward_player")),
        cooperation_conditions=_string(value.get("cooperation_conditions")),
        status=_string(value.get("status")),
        location_name=_string(value.get("location_name")),
        private_notes=_string(value.get("private_notes")),
        reason=_string(value.get("reason")),
        confidence=_confidence(value.get("confidence")),
    )


def _thread_from_data(value: dict[str, object]) -> ExtractedActiveThread:
    return ExtractedActiveThread(
        title=_string(value.get("title")),
        source_message_id=_string(value.get("source_message_id")),
        evidence_quote=_string(value.get("evidence_quote")),
        description=_string(value.get("description")),
        status=_string(value.get("status")),
        priority=_int_or_none(value.get("priority")),
        visibility=_string(value.get("visibility")),
        related_entities=_string_tuple(value.get("related_entities")),
        reason=_string(value.get("reason")),
        confidence=_confidence(value.get("confidence")),
    )


def _entity_link_from_data(value: dict[str, object]) -> ExtractedEntityLink:
    return ExtractedEntityLink(
        entity_type=_string(value.get("entity_type")),
        entity_name=_string(value.get("entity_name")),
        entity_id=_string(value.get("entity_id")),
        target_type=_string(value.get("target_type")),
        target_name=_string(value.get("target_name")),
        target_id=_string(value.get("target_id")),
        relation=_string(value.get("relation")),
        source_message_id=_string(value.get("source_message_id")),
        evidence_quote=_string(value.get("evidence_quote")),
        reason=_string(value.get("reason")),
        confidence=_confidence(value.get("confidence")),
    )


def _phone_number_exchange_from_data(
    value: dict[str, object],
) -> ExtractedPhoneNumberExchange:
    return ExtractedPhoneNumberExchange(
        character_id=_string(value.get("character_id")),
        direction=_string(value.get("direction")),
        source_message_id=_string(value.get("source_message_id")),
        evidence_quote=_string(value.get("evidence_quote")),
        reason=_string(value.get("reason")),
        confidence=_confidence(value.get("confidence")),
    )


def _with_inferred_phone_number_exchanges(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    extraction: ContextUpdateExtraction,
    completed_messages: tuple[MessageRecord, ...],
) -> ContextUpdateExtraction:
    inferred = _infer_phone_number_exchanges(
        repositories,
        save_id=save_id,
        completed_messages=completed_messages,
    )
    if not inferred:
        return extraction
    exchanges: list[ExtractedPhoneNumberExchange] = list(
        extraction.phone_number_exchanges
    )
    seen = {
        (exchange.character_id, exchange.direction, exchange.source_message_id)
        for exchange in exchanges
    }
    for exchange in inferred:
        key = (exchange.character_id, exchange.direction, exchange.source_message_id)
        if key in seen:
            continue
        seen.add(key)
        exchanges.append(exchange)
    return replace(extraction, phone_number_exchanges=tuple(exchanges))


def _infer_phone_number_exchanges(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    completed_messages: tuple[MessageRecord, ...],
) -> tuple[ExtractedPhoneNumberExchange, ...]:
    if not completed_messages:
        return ()
    characters = tuple(repositories.list_characters(save_id))
    player = next(
        (character for character in characters if character.is_player_character),
        None,
    )
    if player is None:
        return ()
    npcs = tuple(
        character for character in characters if not character.is_player_character
    )
    source_bodies = {message.id: message.body for message in completed_messages}
    return tuple(
        ExtractedPhoneNumberExchange(
            character_id=exchange.character_id,
            direction=exchange.direction,
            source_message_id=exchange.source_message_id,
            evidence_quote=source_bodies.get(exchange.source_message_id, ""),
            reason=exchange.reason,
            confidence=exchange.confidence,
        )
        for exchange in infer_phone_number_exchange_records(
            completed_messages=completed_messages,
            player=player,
            npcs=npcs,
        )
    )


def _filter_structured_extraction_evidence(
    extraction: ContextUpdateExtraction,
    *,
    source_messages_by_id: dict[str, MessageRecord],
) -> ContextUpdateExtraction:
    def valid_evidence(source_message_id: str, evidence_quote: str) -> bool:
        source = source_messages_by_id.get(source_message_id)
        return (
            source is not None
            and bool(evidence_quote.strip())
            and quote_matches_source(evidence_quote, source.body)
        )

    scene = extraction.scene
    if scene is not None and not valid_evidence(
        scene.source_message_id,
        scene.evidence_quote,
    ):
        scene = None

    return ContextUpdateExtraction(
        scene=scene,
        locations=tuple(
            location
            for location in extraction.locations
            if valid_evidence(location.source_message_id, location.evidence_quote)
        ),
        characters=tuple(
            character
            for character in extraction.characters
            if valid_evidence(character.source_message_id, character.evidence_quote)
        ),
        active_threads=tuple(
            thread
            for thread in extraction.active_threads
            if valid_evidence(thread.source_message_id, thread.evidence_quote)
        ),
        entity_links=tuple(
            link
            for link in extraction.entity_links
            if valid_evidence(link.source_message_id, link.evidence_quote)
        ),
        phone_number_exchanges=tuple(
            exchange
            for exchange in extraction.phone_number_exchanges
            if valid_evidence(exchange.source_message_id, exchange.evidence_quote)
        ),
        tool_diagnostics=extraction.tool_diagnostics,
    )


def _filter_world_data_enrichment_evidence(
    enrichment: WorldDataEnrichment,
    *,
    source_messages_by_id: dict[str, MessageRecord],
) -> WorldDataEnrichment:
    def valid_evidence(source_message_id: str, evidence_quote: str) -> bool:
        source = source_messages_by_id.get(source_message_id)
        return (
            source is not None
            and bool(evidence_quote.strip())
            and quote_matches_source(evidence_quote, source.body)
        )

    return WorldDataEnrichment(
        locations=tuple(
            location
            for location in enrichment.locations
            if valid_evidence(location.source_message_id, location.evidence_quote)
        ),
        active_threads=tuple(
            thread
            for thread in enrichment.active_threads
            if valid_evidence(thread.source_message_id, thread.evidence_quote)
        ),
        characters=tuple(
            character
            for character in enrichment.characters
            if valid_evidence(character.source_message_id, character.evidence_quote)
        ),
    )


def _validate_extraction(
    extraction: ContextUpdateExtraction,
    *,
    allowed_source_message_ids: tuple[str, ...] | None,
) -> None:
    allowed = set(allowed_source_message_ids or ())

    def validate_source(source_message_id: str, label: str) -> None:
        if not source_message_id:
            raise ValueError(f"{label} source_message_id is required")
        if allowed_source_message_ids is not None and source_message_id not in allowed:
            raise ValueError(f"Unknown {label} source_message_id: {source_message_id}")

    def validate_evidence(
        source_message_id: str,
        evidence_quote: str,
        label: str,
    ) -> None:
        validate_source(source_message_id, label)
        if not evidence_quote.strip():
            raise ValueError(f"{label} evidence_quote is required")

    if extraction.scene is not None:
        validate_evidence(
            extraction.scene.source_message_id,
            extraction.scene.evidence_quote,
            "scene",
        )
    for location in extraction.locations:
        if not location.name.strip():
            raise ValueError("Location name is required")
        validate_evidence(
            location.source_message_id,
            location.evidence_quote,
            "location",
        )
    for character in extraction.characters:
        if not character.name.strip():
            raise ValueError("Character name is required")
        validate_evidence(
            character.source_message_id,
            character.evidence_quote,
            "character",
        )
    for thread in extraction.active_threads:
        if not thread.title.strip():
            raise ValueError("Active thread title is required")
        validate_evidence(
            thread.source_message_id,
            thread.evidence_quote,
            "active thread",
        )
    for link in extraction.entity_links:
        validate_evidence(
            link.source_message_id,
            link.evidence_quote,
            "entity link",
        )
    for exchange in extraction.phone_number_exchanges:
        if not exchange.character_id.strip():
            raise ValueError("Phone number exchange character_id is required")
        if exchange.direction not in {
            "player_has_character_number",
            "character_has_player_number",
            "both",
        }:
            raise ValueError("Phone number exchange direction is invalid")
        validate_evidence(
            exchange.source_message_id,
            exchange.evidence_quote,
            "phone number exchange",
        )


def _validate_world_data_enrichment(
    enrichment: WorldDataEnrichment,
    *,
    allowed_source_message_ids: tuple[str, ...] | None,
) -> None:
    allowed = set(allowed_source_message_ids or ())

    def validate_source(source_message_id: str, label: str) -> None:
        if not source_message_id:
            raise ValueError(f"{label} source_message_id is required")
        if allowed_source_message_ids is not None and source_message_id not in allowed:
            raise ValueError(f"Unknown {label} source_message_id: {source_message_id}")

    def validate_evidence(
        source_message_id: str,
        evidence_quote: str,
        label: str,
    ) -> None:
        validate_source(source_message_id, label)
        if not evidence_quote.strip():
            raise ValueError(f"{label} evidence_quote is required")

    for location in enrichment.locations:
        if not location.location_id.strip():
            raise ValueError("Location enrichment location_id is required")
        validate_evidence(
            location.source_message_id,
            location.evidence_quote,
            "location enrichment",
        )
    for thread in enrichment.active_threads:
        if not thread.active_thread_id.strip():
            raise ValueError("Active thread enrichment active_thread_id is required")
        validate_evidence(
            thread.source_message_id,
            thread.evidence_quote,
            "active thread enrichment",
        )
    for character in enrichment.characters:
        if not character.character_id.strip():
            raise ValueError("Character enrichment character_id is required")
        validate_evidence(
            character.source_message_id,
            character.evidence_quote,
            "character enrichment",
        )


def _drop_unknown_extraction_sources(
    extraction: ContextUpdateExtraction,
    *,
    allowed_source_message_ids: tuple[str, ...] | None,
) -> ContextUpdateExtraction:
    if allowed_source_message_ids is None:
        return extraction
    allowed = set(allowed_source_message_ids)

    scene = extraction.scene
    if scene is not None and scene.source_message_id not in allowed:
        scene = None

    return ContextUpdateExtraction(
        scene=scene,
        locations=tuple(
            location
            for location in extraction.locations
            if location.source_message_id in allowed
        ),
        characters=tuple(
            character
            for character in extraction.characters
            if character.source_message_id in allowed
        ),
        active_threads=tuple(
            thread
            for thread in extraction.active_threads
            if thread.source_message_id in allowed
        ),
        entity_links=tuple(
            link
            for link in extraction.entity_links
            if link.source_message_id in allowed
        ),
        phone_number_exchanges=tuple(
            exchange
            for exchange in extraction.phone_number_exchanges
            if exchange.source_message_id in allowed
        ),
        tool_diagnostics=extraction.tool_diagnostics,
    )


def _drop_invalid_extracted_entities(
    extraction: ContextUpdateExtraction,
) -> ContextUpdateExtraction:
    return ContextUpdateExtraction(
        scene=extraction.scene,
        locations=tuple(
            location for location in extraction.locations if location.name.strip()
        ),
        characters=tuple(
            character
            for character in extraction.characters
            if character.name.strip()
        ),
        active_threads=tuple(
            thread for thread in extraction.active_threads if thread.title.strip()
        ),
        entity_links=extraction.entity_links,
        phone_number_exchanges=tuple(
            exchange
            for exchange in extraction.phone_number_exchanges
            if exchange.character_id.strip()
        ),
        tool_diagnostics=extraction.tool_diagnostics,
    )


def _selector_from_extractor(
    extractor: ContextUpdateExtractor,
) -> ContextRegistrySelector | None:
    select_context = getattr(extractor, "select_context", None)
    if not callable(select_context):
        return None
    return cast(ContextRegistrySelector, extractor)


def _prompt_inspection_message_id(
    messages: tuple[MessageRecord, ...],
) -> str | None:
    for message in reversed(messages):
        if message.role != "player":
            return message.id
    return messages[-1].id if messages else None


def _scene_snapshot_without_source_message(
    snapshot: SceneSnapshotRecord | None,
    source_message_id: str,
) -> SceneSnapshotRecord | None:
    if snapshot is not None and snapshot.source_message_id == source_message_id:
        return None
    return snapshot


def _memory_source_ids(memory: MemoryRecord) -> frozenset[str]:
    source_ids = set(memory.source_message_ids)
    if memory.source_message_id is not None:
        source_ids.add(memory.source_message_id)
    return frozenset(source_ids)


def _summaries_without_covered_message(
    *,
    summaries: tuple[SummaryRecord, ...],
    messages: tuple[MessageRecord, ...],
    message_id: str,
) -> tuple[SummaryRecord, ...]:
    message_order = {message.id: index for index, message in enumerate(messages)}
    if message_id not in message_order:
        return summaries
    return tuple(
        summary
        for summary in summaries
        if not _summary_covers_message(
            summary=summary,
            message_order=message_order,
            message_id=message_id,
        )
    )


def _summary_covers_message(
    *,
    summary: SummaryRecord,
    message_order: dict[str, int],
    message_id: str,
) -> bool:
    start = message_order.get(summary.covers_message_start_id)
    end = message_order.get(summary.covers_message_end_id)
    target = message_order.get(message_id)
    if start is None or end is None or target is None:
        return False
    lower, upper = sorted((start, end))
    return lower <= target <= upper


def _entity_link_visible_for_correction(
    *,
    link: EntityLinkRecord,
    locations: tuple[LocationRecord, ...],
    characters: tuple[CharacterRecord, ...],
    active_threads: tuple[ActiveThreadRecord, ...],
    memories: tuple[MemoryRecord, ...],
    world_state: tuple[WorldStateRecord, ...],
) -> bool:
    known_entity_ids: dict[str, set[str]] = {
        "location": {location.id for location in locations},
        "character": {character.id for character in characters},
        "active_thread": {thread.id for thread in active_threads},
        "memory": {memory.id for memory in memories},
        "world_state": {state.id for state in world_state},
    }
    return _entity_link_endpoint_visible(
        entity_type=link.entity_type,
        entity_id=link.entity_id,
        known_entity_ids=known_entity_ids,
    ) and _entity_link_endpoint_visible(
        entity_type=link.target_type,
        entity_id=link.target_id,
        known_entity_ids=known_entity_ids,
    )


def _entity_link_endpoint_visible(
    *,
    entity_type: str,
    entity_id: str,
    known_entity_ids: dict[str, set[str]],
) -> bool:
    if entity_type not in known_entity_ids:
        return True
    return entity_id in known_entity_ids[entity_type]


def _context_registry_candidates(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    messages: tuple[MessageRecord, ...],
    scene_snapshot: SceneSnapshotRecord | None,
    locations: tuple[LocationRecord, ...],
    characters: tuple[CharacterRecord, ...],
    active_threads: tuple[ActiveThreadRecord, ...],
    context_sources: tuple[ContextSourceRecord, ...] | None = None,
    context_observations: tuple[ContextObservationRecord, ...] | None = None,
    all_messages: tuple[MessageRecord, ...] | None = None,
) -> tuple[ContextRegistryItem, ...]:
    records = (
        context_sources
        if context_sources is not None
        else tuple(repositories.list_context_sources(save_id))
    )
    if not records:
        return ()
    accepted_observation_ids: frozenset[str] = frozenset()
    if any(record.source_type == "observation" for record in records):
        observations = (
            context_observations
            if context_observations is not None
            else tuple(
                repositories.list_context_observations(
                    save_id,
                    statuses={"accepted"},
                )
            )
        )
        accepted_observation_ids = frozenset(
            observation.id
            for observation in observations
            if observation.status == "accepted"
        )
    ordered_messages = (
        all_messages
        if all_messages is not None
        else tuple(repositories.list_messages(save_id))
    )
    message_order = {
        message.id: index
        for index, message in enumerate(ordered_messages)
    }
    query_terms = _context_registry_query_terms(
        messages=messages,
        scene_snapshot=scene_snapshot,
        locations=locations,
        characters=characters,
        active_threads=active_threads,
    )
    active_entity_ids = _active_context_entity_ids(
        scene_snapshot=scene_snapshot,
        active_threads=active_threads,
    )
    ranked: list[tuple[float, int, ContextRegistryItem]] = []
    for index, record in enumerate(records):
        item = _context_registry_item(
            record,
            accepted_observation_ids=accepted_observation_ids,
        )
        if item is None:
            continue
        ranked.append(
            (
                _context_registry_rank(
                    item,
                    record=record,
                    query_terms=query_terms,
                    active_entity_ids=active_entity_ids,
                    message_order=message_order,
                ),
                index,
                item,
            )
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return tuple(
        item for _score, _index, item in ranked[:MAX_CONTEXT_UPDATE_CANDIDATES]
    )


def _context_registry_item(
    record: ContextSourceRecord,
    *,
    accepted_observation_ids: frozenset[str],
) -> ContextRegistryItem | None:
    if record.metadata.get("indexed_by") != "continuity_index" and (
        _curated_observation_context_source_id(
            record,
            accepted_observation_ids=accepted_observation_ids,
        )
        is None
    ):
        return None
    if record.source_type == "scenario_section":
        return None
    body = record.body.strip()
    if not body:
        return None
    fact_type = str(
        record.metadata.get("fact_type")
        or record.metadata.get("observation_type")
        or ""
    ).strip()
    importance = record.metadata.get("importance", 0.0)
    return ContextRegistryItem(
        context_source_id=record.id,
        source_type=record.source_type,
        source_id=record.source_id,
        title=record.title,
        body=body,
        fact_type=fact_type,
        importance=float(importance) if isinstance(importance, int | float) else 0.0,
        source_message_ids=_context_source_message_ids(record),
    )


def _curated_observation_context_source_id(
    record: ContextSourceRecord,
    *,
    accepted_observation_ids: frozenset[str],
) -> str | None:
    if record.source_type != "observation":
        return None
    if record.metadata.get("curation_action") not in {
        "save_context",
        "scene_scratch",
    }:
        return None
    observation_id = record.metadata.get("observation_id")
    if not isinstance(observation_id, str) or not observation_id.strip():
        observation_id = record.source_id
    observation_id = observation_id.strip()
    if observation_id != record.source_id:
        return None
    if observation_id not in accepted_observation_ids:
        return None
    return observation_id


def _context_source_message_ids(record: ContextSourceRecord) -> tuple[str, ...]:
    source_ids: list[str] = []
    if record.source_type == "message":
        source_ids.extend(
            item.strip() for item in record.source_id.split(",") if item.strip()
        )
    source_ids.extend(_metadata_string_tuple(record.metadata.get("source_message_ids")))
    metadata_source_id = record.metadata.get("source_message_id")
    if isinstance(metadata_source_id, str) and metadata_source_id:
        source_ids.append(metadata_source_id)
    return tuple(dict.fromkeys(source_ids))


def _context_registry_rank(
    item: ContextRegistryItem,
    *,
    record: ContextSourceRecord,
    query_terms: set[str],
    active_entity_ids: set[str],
    message_order: dict[str, int],
) -> float:
    source_score = {
        "open_obligation": 8.0,
        "character_voice": 7.5,
        "memory": 6.0,
        "observation": 5.75,
        "world_state": 5.5,
        "scenario_section": 3.0,
        "summary": 2.0,
    }.get(item.source_type, 1.0)
    text = f"{item.title} {item.body} {item.fact_type}".casefold()
    lexical_score = sum(1.0 for term in query_terms if term in text)
    fact_score = 3.0 if item.fact_type in _HIGH_VALUE_CONTEXT_FACT_TYPES else 0.0
    always_score = 4.0 if record.metadata.get("always_include_reason") else 0.0
    entity_ids = set(_metadata_string_tuple(record.metadata.get("entity_ids")))
    entity_score = 3.0 if entity_ids & active_entity_ids else 0.0
    source_positions = [
        message_order[source_id]
        for source_id in item.source_message_ids
        if source_id in message_order
    ]
    recency_score = 0.0
    if source_positions and message_order:
        recency_score = (max(source_positions) + 1) / len(message_order) * 2.0
    return (
        source_score
        + (item.importance * 4.0)
        + lexical_score
        + fact_score
        + always_score
        + entity_score
        + recency_score
    )


def _context_registry_query_terms(
    *,
    messages: tuple[MessageRecord, ...],
    scene_snapshot: SceneSnapshotRecord | None,
    locations: tuple[LocationRecord, ...],
    characters: tuple[CharacterRecord, ...],
    active_threads: tuple[ActiveThreadRecord, ...],
) -> set[str]:
    parts = [message.body for message in messages]
    if scene_snapshot is not None:
        parts.extend((scene_snapshot.situation, scene_snapshot.objective))
    parts.extend(location.name for location in locations)
    parts.extend(character.name for character in characters)
    parts.extend(thread.title for thread in active_threads)
    return _meaningful_context_terms("\n".join(parts))


def _active_context_entity_ids(
    *,
    scene_snapshot: SceneSnapshotRecord | None,
    active_threads: tuple[ActiveThreadRecord, ...],
) -> set[str]:
    ids: set[str] = set()
    if scene_snapshot is not None:
        if scene_snapshot.current_location_id:
            ids.add(scene_snapshot.current_location_id)
        ids.update(scene_snapshot.present_character_ids)
    ids.update(thread.id for thread in active_threads)
    return ids


def _meaningful_context_terms(text: str) -> set[str]:
    ignored = {"and", "for", "the", "that", "this", "with", "you", "your"}
    return {
        term
        for term in findall(r"[a-z0-9']{3,}", text.casefold())
        if term not in ignored
    }


def _metadata_string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _fallback_context_registry_selection(
    candidates: tuple[ContextRegistryItem, ...],
) -> ContextRegistrySelection:
    return ContextRegistrySelection(
        selected_items=tuple(
            replace(item, relevance_note="Selected by deterministic fallback.")
            for item in candidates[:MAX_CONTEXT_UPDATE_SELECTIONS]
        ),
        fallback_used=True,
    )


def _normalize_context_registry_selection(
    selection: ContextRegistrySelection,
    candidates: tuple[ContextRegistryItem, ...],
) -> ContextRegistrySelection:
    by_context_source_id = {item.context_source_id: item for item in candidates}
    selected: list[ContextRegistryItem] = []
    seen: set[str] = set()
    for item in selection.selected_items:
        candidate = by_context_source_id.get(item.context_source_id)
        if candidate is None or candidate.context_source_id in seen:
            continue
        seen.add(candidate.context_source_id)
        selected.append(
            replace(
                candidate,
                relevance_note=item.relevance_note.strip()
                or "Selected by context update selection.",
            )
        )
        if len(selected) >= MAX_CONTEXT_UPDATE_SELECTIONS:
            break
    return ContextRegistrySelection(
        selected_items=tuple(selected),
        fallback_used=selection.fallback_used,
    )


def _context_registry_selection_schema(
    candidates: tuple[ContextRegistryItem, ...],
) -> dict[str, object]:
    return normalize_strict_json_schema({
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selections": {
                "type": "array",
                "maxItems": min(MAX_CONTEXT_UPDATE_SELECTIONS, len(candidates)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "context_source_id": {
                            "type": "string",
                            "enum": [item.context_source_id for item in candidates],
                        },
                        "relevance_note": {"type": "string"},
                    },
                    "required": ["context_source_id", "relevance_note"],
                },
            }
        },
        "required": ["selections"],
    })


def _context_registry_selection_messages(
    request: ContextRegistrySelectionRequest,
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Select the minimum prior Bragi context needed for the context "
                "update extractor to interpret the completed turn. Use the "
                "enforced schema. Select no more than the most relevant durable "
                "facts. Prefer current-scene entities, open obligations, "
                "relationships, promises, identity, location, inventory, and "
                "character voice facts. Do not select trivia that only shares "
                "generic words with the turn."
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    _context_selection_identity_text(request),
                    _messages_text(request.messages),
                    _context_registry_candidate_text(request.candidates),
                )
            ),
        ),
    )


def _context_registry_selection_tool_messages(
    request: ContextRegistrySelectionRequest,
) -> tuple[ToolCallMessage, ...]:
    messages = _context_registry_selection_messages(request)
    tool_messages: list[ToolCallMessage] = []
    for message in messages:
        body = message.body.replace(
            "Use the enforced schema.",
            (
                "Use the provided select_prior_context tool instead of prose. "
                "Call it once per selected candidate in priority order. Select "
                "nothing by making no tool calls."
            ),
        )
        tool_messages.append(
            ToolCallMessage(
                role=message.role,
                body=body,
                speaker_name=message.speaker_name,
            )
        )
    return tuple(tool_messages)


def _context_registry_selection_tool_definitions(
    candidates: tuple[ContextRegistryItem, ...],
) -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="select_prior_context",
            description=(
                "Select one offered prior context source that helps interpret "
                "the completed turn."
            ),
            parameters=_context_registry_selection_tool_schema(candidates),
        ),
    )


def _context_registry_selection_tool_schema(
    candidates: tuple[ContextRegistryItem, ...],
) -> dict[str, object]:
    return _tool_schema(
        required=["context_source_id"],
        properties={
            "context_source_id": {
                "type": "string",
                "enum": [item.context_source_id for item in candidates],
            },
            "relevance_note": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    )


def _context_selection_identity_text(
    request: ContextRegistrySelectionRequest,
) -> str:
    lines = ["Current compact registry identity:"]
    if request.scene_snapshot is not None:
        snapshot = request.scene_snapshot
        lines.append(
            "- scene: "
            f"situation={snapshot.situation}; objective={snapshot.objective}; "
            f"location_id={snapshot.current_location_id or ''}; "
            f"present={', '.join(snapshot.present_character_ids)}"
        )
    locations, omitted = _limited_items(
        request.locations,
        MAX_CONTEXT_UPDATE_IDENTITY_LOCATIONS,
    )
    for location in locations:
        lines.append(f"- location {location.id}: {location.name}")
    _append_omitted_count(lines, "locations", omitted)
    characters, omitted = _limited_items(
        request.characters,
        MAX_CONTEXT_UPDATE_IDENTITY_CHARACTERS,
    )
    for character in characters:
        lines.append(f"- character {character.id}: {character.name}")
    _append_omitted_count(lines, "characters", omitted)
    threads, omitted = _limited_items(
        request.active_threads,
        MAX_CONTEXT_UPDATE_IDENTITY_THREADS,
    )
    for thread in threads:
        lines.append(
            f"- active_thread {thread.id}: {thread.title}; priority={thread.priority}"
        )
    _append_omitted_count(lines, "active threads", omitted)
    return "\n".join(lines)


def _context_registry_candidate_text(
    candidates: tuple[ContextRegistryItem, ...],
) -> str:
    if not candidates:
        return "Prior context candidates: none"
    lines = ["Prior context candidates:"]
    for item in candidates:
        lines.append(
            f"- [{item.context_source_id}] {item.source_type}:{item.source_id}; "
            f"fact_type={item.fact_type}; importance={item.importance:.2g}; "
            f"title={item.title}; "
            f"body={_compact_text(item.body, MAX_CONTEXT_UPDATE_CANDIDATE_BODY_CHARS)}"
        )
    return "\n".join(lines)


def _context_registry_selection_from_structured_data(
    data: dict[str, object],
    *,
    candidates: tuple[ContextRegistryItem, ...],
) -> ContextRegistrySelection:
    raw_selections = data.get("selections", [])
    if not isinstance(raw_selections, list):
        raise ValueError("Structured context update selection must be a list")
    by_context_source_id = {item.context_source_id: item for item in candidates}
    selected: list[ContextRegistryItem] = []
    seen: set[str] = set()
    for raw_selection in raw_selections:
        if not isinstance(raw_selection, dict):
            raise ValueError("Structured context update selection item must be object")
        context_source_id = str(raw_selection.get("context_source_id", ""))
        item = by_context_source_id.get(context_source_id)
        if item is None:
            raise ValueError(f"Unknown context_source_id: {context_source_id}")
        if item.context_source_id in seen:
            continue
        seen.add(item.context_source_id)
        selected.append(
            replace(
                item,
                relevance_note=str(raw_selection.get("relevance_note", "")).strip(),
            )
        )
        if len(selected) >= MAX_CONTEXT_UPDATE_SELECTIONS:
            break
    return ContextRegistrySelection(selected_items=tuple(selected))


def _selected_prior_context_text(
    items: tuple[ContextRegistryItem, ...],
) -> list[str]:
    if not items:
        return ["Selected prior context: none"]
    lines = ["Selected prior context:"]
    for item in items:
        prefix = f"- [{item.source_type}:{item.source_id}]"
        metadata = []
        if item.fact_type:
            metadata.append(f"fact_type={item.fact_type}")
        if item.importance:
            metadata.append(f"importance={item.importance:.2g}")
        if item.relevance_note:
            metadata.append(f"relevance={item.relevance_note}")
        suffix = f" ({'; '.join(metadata)})" if metadata else ""
        title = f"{item.title}: " if item.title else ""
        lines.append(
            f"{prefix} {title}"
            f"{_compact_text(item.body, MAX_SELECTED_PRIOR_CONTEXT_CHARS)}{suffix}"
        )
    return lines


def _limited_items(
    items: tuple[Any, ...],
    limit: int,
) -> tuple[tuple[Any, ...], int]:
    if limit <= 0:
        return (), len(items)
    if len(items) <= limit:
        return items, 0
    return items[:limit], len(items) - limit


def _append_omitted_count(lines: list[str], label: str, omitted: int) -> None:
    if omitted > 0:
        lines.append(f"- {omitted} additional {label} omitted by payload cap")


def _compact_text(text: str, limit: int) -> str:
    compact = sub(r"\s+", " ", text.strip())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "..."


def _present_character_names_schema() -> dict[str, object]:
    return {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "description": (
            "All named characters clearly present in the current scene. Use [] "
            "only when the completed turn clearly establishes no named "
            "characters are present; use null or omit the field when presence "
            "is unchanged or unclear."
        ),
    }


def _current_scene_string_array_schema(label: str) -> dict[str, object]:
    return {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "description": (
            f"Complete current scene {label}. Use [] only when the completed "
            f"turn clearly establishes no current scene {label}; use null or "
            "omit the field when unchanged or unclear."
        ),
    }


def _context_update_schema(messages: tuple[MessageRecord, ...]) -> dict[str, object]:
    source_schema: dict[str, object] = {"type": "string"}
    message_ids = [message.id for message in messages]
    if message_ids:
        source_schema["enum"] = message_ids
    base_properties: dict[str, object] = {
        "source_message_id": source_schema,
        "evidence_quote": {
            "type": "string",
            "description": (
                "Exact substring copied from the source message that grounds "
                "this extracted item."
            ),
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    string_array = {"type": "array", "items": {"type": "string"}}
    scene_nearby_objects_schema = _current_scene_string_array_schema(
        "nearby objects"
    )
    scene_hazards_schema = _current_scene_string_array_schema("hazards")
    present_character_names_schema = _present_character_names_schema()
    relationship_array = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name", "description"],
        },
    }
    character_agency_properties = {
        field_name: {"type": "string"}
        for field_name in sorted(CHARACTER_AGENCY_FIELDS)
    }
    return normalize_strict_json_schema({
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scene": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    **base_properties,
                    "current_location_name": {"type": "string"},
                    "situation": {"type": "string"},
                    "objective": {"type": "string"},
                    "in_world_time": {
                        "type": "string",
                        "description": (
                            "Optional qualitative current time anchor for the "
                            "scene, such as morning, late morning, afternoon, "
                            "evening, night, or an explicit stated clock time "
                            "like 8 AM. Emit only when the completed turn "
                            "directly supports the time or explicit time "
                            "passage."
                        ),
                    },
                    "weather": {"type": "string"},
                    "mood": {"type": "string"},
                    "nearby_objects": scene_nearby_objects_schema,
                    "hazards": scene_hazards_schema,
                    "present_character_names": present_character_names_schema,
                },
                "required": ["source_message_id", "evidence_quote"],
            },
            "locations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        **base_properties,
                        "name": {"type": "string"},
                        "aliases": string_array,
                        "description": {"type": "string"},
                        "visual_description": {"type": "string"},
                        "parent_location_name": {"type": "string"},
                        "connections": string_array,
                        "status": {"type": "string"},
                        "hazards": string_array,
                    },
                    "required": ["name", "source_message_id", "evidence_quote"],
                },
            },
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        **base_properties,
                        "name": {"type": "string"},
                        "aliases": string_array,
                        "role": {"type": "string"},
                        "age": {"type": "string"},
                        "known_state": {"type": "string"},
                        "met": {"type": "boolean"},
                        "appearance": {"type": "string"},
                        "visual_notes": {"type": "string"},
                        "current_clothing": {
                            "type": "string",
                            "description": (
                                "Current outfit, clothing, armor, uniform, or "
                                "worn accessories explicitly supported by the "
                                "completed turn. Do not copy stable physical "
                                "identity here unless it is described as being worn."
                            ),
                        },
                        "personality": {"type": "string"},
                        "voice": {"type": "string"},
                        "relationships": relationship_array,
                        **character_agency_properties,
                        "status": {"type": "string"},
                        "location_name": {"type": "string"},
                        "private_notes": {"type": "string"},
                    },
                    "required": ["name", "source_message_id", "evidence_quote"],
                },
            },
            "active_threads": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        **base_properties,
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["", *sorted(ACTIVE_THREAD_STATUSES)],
                        },
                        "priority": {"type": "integer"},
                        "visibility": {
                            "type": "string",
                            "enum": ["", *sorted(ACTIVE_THREAD_VISIBILITIES)],
                        },
                        "related_entities": string_array,
                    },
                    "required": ["title", "source_message_id", "evidence_quote"],
                },
            },
            "entity_links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        **base_properties,
                        "entity_type": {"type": "string"},
                        "entity_name": {"type": "string"},
                        "entity_id": {"type": "string"},
                        "target_type": {"type": "string"},
                        "target_name": {"type": "string"},
                        "target_id": {"type": "string"},
                        "relation": {"type": "string"},
                    },
                    "required": [
                        "entity_type",
                        "target_type",
                        "source_message_id",
                        "evidence_quote",
                    ],
                },
            },
            "phone_number_exchanges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        **base_properties,
                        "character_id": {"type": "string"},
                        "direction": {
                            "type": "string",
                            "enum": [
                                "player_has_character_number",
                                "character_has_player_number",
                                "both",
                            ],
                        },
                    },
                    "required": [
                        "character_id",
                        "direction",
                        "source_message_id",
                        "evidence_quote",
                    ],
                },
            },
        },
        "required": [
            "locations",
            "characters",
            "active_threads",
            "entity_links",
            "phone_number_exchanges",
        ],
    })


def _context_update_messages(request: ContextUpdateRequest) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Extract structured Bragi context registry updates from the "
                "completed turn. Use the enforced response schema. Include only "
                "facts directly supported by the messages. Prefer additive or "
                "low-risk current-scene updates; uncertain rewrites should be "
                "omitted so the application can preserve existing user edits. "
                "A marked narrator safety transition is only the canonical "
                "off-screen event and elapsed time; never create intimate "
                "details, physical facts, or context records from a rejected "
                "narrator draft. "
                "Every extracted item must include source_message_id and "
                "evidence_quote; evidence_quote must be copied exactly from "
                "that source message. "
                "For characters, put explicitly stated current outfit or worn "
                "items in current_clothing; reserve appearance for stable "
                "physical identity. "
                "Prefer entity names, titles, and world-state keys over opaque "
                "database ids whenever the schema gives you name fields. When a "
                "completed turn directly supports an NPC's goals, motivations, "
                "current intent, boundaries, attitude toward the player, or "
                "cooperation conditions, emit those character agency fields; "
                "do not infer agency from vibes or genre expectations. When a "
                "character age is directly stated or clearly evidenced, emit "
                "character.age; otherwise omit it. When a "
                "character specifically knows an existing memory, world_state, "
                "or summary, emit an entity link from the character to that "
                "target with relation knows; use an existing target_id only when "
                "you can copy the id exactly from the registry text. When the "
                "completed turn directly states that the player and an NPC "
                "exchange phone numbers or contact info, emit a phone number "
                "exchange. For "
                "scene.in_world_time, use compact qualitative anchors such as "
                "morning, late morning, afternoon, evening, or night. Preserve "
                "an exact clock time only when the completed turn explicitly "
                "states it. Do not emit vague time values like later, soon, "
                "eventually, or unclear. If an existing scene time is present, "
                "emit a different scene.in_world_time only when the player "
                "explicitly advances, waits, skips ahead, travels to a later "
                "appointment, or names a later time. For "
                "scene.present_character_names, include the complete current "
                "named presence only when the turn establishes it; emit an "
                "empty list only when the turn clearly establishes that no "
                "named characters are present, and omit or use null when "
                "presence is unchanged or unclear. For scene.nearby_objects "
                "and scene.hazards, emit complete current lists when the turn "
                "clearly establishes them, emit an empty list only when the "
                "turn clearly establishes none remain, and omit or use null "
                "when unchanged or unclear. Treat fields listed as "
                "locked(read-only) in the registry as player-locked read-only "
                "facts."
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                item
                for item in (
                    _registry_text(request),
                    _messages_text(request.messages),
                    correction_context_text(request.correction_context),
                )
                if item
            ),
        ),
    )


def _context_update_tool_messages(
    request: ContextUpdateRequest,
) -> tuple[ToolCallMessage, ...]:
    messages = _context_update_messages(request)
    tool_messages: list[ToolCallMessage] = []
    for message in messages:
        body = message.body.replace(
            "Use the enforced response schema.",
            (
                "Use the provided tools instead of prose. Call only tools whose "
                "arguments are directly supported by the completed turn. Every "
                "tool call must include source_message_id and evidence_quote; "
                "evidence_quote must be copied exactly from that source message. "
                "If the source text does not provide a name or backstory, use "
                "unknown or empty arrays instead of inventing details. For "
                "character agency fields, use only directly supported goals, "
                "motivations, current intent, boundaries, attitude toward the "
                "player, or cooperation conditions; otherwise omit them. For "
                "character age, emit only directly stated or clearly evidenced "
                "age; otherwise omit it. For phone number exchanges, call "
                "record_phone_number_exchange only for explicit phone number, "
                "texting, or contact-info transfers between the player and one "
                "NPC; do not infer it from vague plans to keep in touch. For "
                "present_character_names, omit or use null unless current scene "
                "presence is clear; use an empty list only when no named "
                "characters are present. For nearby_objects and hazards, "
                "provide complete current lists only when clear; use an empty "
                "list only when none remain; otherwise omit or use null."
            ),
        )
        tool_messages.append(
            ToolCallMessage(
                role=message.role,
                body=body,
                speaker_name=message.speaker_name,
            )
        )
    return tuple(tool_messages)


def _context_update_tool_definitions(
    messages: tuple[MessageRecord, ...],
) -> tuple[ToolDefinition, ...]:
    schemas = _context_update_tool_schemas(messages)
    descriptions = {
        "update_scene_snapshot": "Propose current scene snapshot fields.",
        "upsert_location": "Propose a location create or update.",
        "upsert_character": "Propose a character create or update.",
        "upsert_active_thread": "Propose an active thread create or update.",
        "link_entities": "Propose a relationship link between existing entities.",
        "record_phone_number_exchange": (
            "Record an explicit phone number or contact-info exchange between "
            "the player and one existing NPC."
        ),
    }
    return tuple(
        ToolDefinition(
            name=name,
            description=descriptions[name],
            parameters=schema,
        )
        for name, schema in schemas.items()
    )


def _context_update_tool_schemas(
    messages: tuple[MessageRecord, ...],
) -> dict[str, dict[str, object]]:
    source_schema: dict[str, object] = {"type": "string"}
    message_ids = [message.id for message in messages]
    if message_ids:
        source_schema["enum"] = message_ids
    base_properties: dict[str, object] = {
        "source_message_id": source_schema,
        "evidence_quote": {
            "type": "string",
            "description": (
                "Exact substring copied from the source message that grounds "
                "this tool call."
            ),
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    string_array = {"type": "array", "items": {"type": "string"}}
    scene_nearby_objects_schema = _current_scene_string_array_schema(
        "nearby objects"
    )
    scene_hazards_schema = _current_scene_string_array_schema("hazards")
    present_character_names_schema = _present_character_names_schema()
    relationship_array = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name", "description"],
        },
    }
    character_agency_properties = {
        field_name: {"type": "string"}
        for field_name in sorted(CHARACTER_AGENCY_FIELDS)
    }
    return {
        "update_scene_snapshot": _tool_schema(
            required=["source_message_id", "evidence_quote"],
            properties={
                **base_properties,
                "current_location_name": {"type": "string"},
                "situation": {"type": "string"},
                "objective": {"type": "string"},
                "in_world_time": {"type": "string"},
                "weather": {"type": "string"},
                "mood": {"type": "string"},
                "nearby_objects": scene_nearby_objects_schema,
                "hazards": scene_hazards_schema,
                "present_character_names": present_character_names_schema,
                "scene_transition": {"type": "boolean"},
            },
        ),
        "upsert_location": _tool_schema(
            required=["name", "source_message_id", "evidence_quote"],
            properties={
                **base_properties,
                "name": {"type": "string"},
                "aliases": string_array,
                "description": {"type": "string"},
                "visual_description": {"type": "string"},
                "parent_location_name": {"type": "string"},
                "connections": string_array,
                "status": {"type": "string"},
                "hazards": string_array,
            },
        ),
        "upsert_character": _tool_schema(
            required=["name", "source_message_id", "evidence_quote"],
            properties={
                **base_properties,
                "name": {"type": "string"},
                "aliases": string_array,
                "role": {"type": "string"},
                "age": {"type": "string"},
                "known_state": {"type": "string"},
                "met": {"type": "boolean"},
                "appearance": {"type": "string"},
                "visual_notes": {"type": "string"},
                "current_clothing": {
                    "type": "string",
                    "description": (
                        "Current outfit or worn items directly evidenced in "
                        "the source message."
                    ),
                },
                "personality": {"type": "string"},
                "voice": {"type": "string"},
                "relationships": relationship_array,
                **character_agency_properties,
                "status": {"type": "string"},
                "location_name": {"type": "string"},
                "private_notes": {"type": "string"},
            },
        ),
        "upsert_active_thread": _tool_schema(
            required=["title", "source_message_id", "evidence_quote"],
            properties={
                **base_properties,
                "title": {"type": "string"},
                "description": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["", *sorted(ACTIVE_THREAD_STATUSES)],
                },
                "priority": {"type": "integer"},
                "visibility": {
                    "type": "string",
                    "enum": ["", *sorted(ACTIVE_THREAD_VISIBILITIES)],
                },
                "related_entities": string_array,
            },
        ),
        "link_entities": _tool_schema(
            required=[
                "entity_type",
                "target_type",
                "source_message_id",
                "evidence_quote",
            ],
            properties={
                **base_properties,
                "entity_type": {
                    "type": "string",
                    "enum": sorted(_LINKABLE_ENTITY_TYPES),
                },
                "entity_name": {"type": "string"},
                "entity_id": {"type": "string"},
                "target_type": {
                    "type": "string",
                    "enum": sorted(_LINKABLE_ENTITY_TYPES),
                },
                "target_name": {"type": "string"},
                "target_id": {"type": "string"},
                "relation": {"type": "string"},
            },
        ),
        "record_phone_number_exchange": _tool_schema(
            required=[
                "character_id",
                "direction",
                "source_message_id",
                "evidence_quote",
            ],
            properties={
                **base_properties,
                "character_id": {"type": "string"},
                "direction": {
                    "type": "string",
                    "enum": [
                        "player_has_character_number",
                        "character_has_player_number",
                        "both",
                    ],
                },
            },
        ),
    }


def _tool_schema(
    *,
    required: list[str],
    properties: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _world_data_enrichment_schema(
    request: WorldDataEnrichmentRequest,
) -> dict[str, object]:
    source_schema: dict[str, object] = {"type": "string"}
    message_ids = [message.id for message in request.messages]
    if message_ids:
        source_schema["enum"] = message_ids
    location_id_schema: dict[str, object] = {"type": "string"}
    location_ids = [location.id for location in request.locations]
    if location_ids:
        location_id_schema["enum"] = location_ids
    thread_id_schema: dict[str, object] = {"type": "string"}
    thread_ids = [thread.id for thread in request.active_threads]
    if thread_ids:
        thread_id_schema["enum"] = thread_ids
    character_id_schema: dict[str, object] = {"type": "string"}
    character_ids = [character.id for character in request.characters]
    if character_ids:
        character_id_schema["enum"] = character_ids
    base_properties: dict[str, object] = {
        "source_message_id": source_schema,
        "evidence_quote": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    string_array = {"type": "array", "items": {"type": "string"}}
    return normalize_strict_json_schema({
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "locations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        **base_properties,
                        "location_id": location_id_schema,
                        "description": {"type": "string"},
                        "visual_description": {"type": "string"},
                        "connections": string_array,
                        "status": {"type": "string"},
                        "hazards": string_array,
                    },
                    "required": [
                        "location_id",
                        "source_message_id",
                        "evidence_quote",
                    ],
                },
            },
            "active_threads": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        **base_properties,
                        "active_thread_id": thread_id_schema,
                        "description": {"type": "string"},
                        "related_entities": string_array,
                    },
                    "required": [
                        "active_thread_id",
                        "source_message_id",
                        "evidence_quote",
                    ],
                },
            },
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        **base_properties,
                        "character_id": character_id_schema,
                        "aliases": string_array,
                        "role": {"type": "string"},
                        "age": {"type": "string"},
                        "known_state": {"type": "string"},
                        "appearance": {"type": "string"},
                        "visual_notes": {"type": "string"},
                        "current_clothing": {
                            "type": "string",
                            "description": (
                                "Current outfit or worn items directly evidenced "
                                "in the source messages."
                            ),
                        },
                        "personality": {"type": "string"},
                        "voice": {"type": "string"},
                        "relationships": {"type": "object"},
                        "status": {"type": "string"},
                    },
                    "required": [
                        "character_id",
                        "source_message_id",
                        "evidence_quote",
                    ],
                },
            },
        },
        "required": ["locations", "active_threads", "characters"],
    })


def _world_data_enrichment_messages(
    request: WorldDataEnrichmentRequest,
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Enrich sparse Bragi world data after the factual extraction pass. "
                "Use the enforced response schema. Fill only blank details for "
                "the listed records when the completed turn directly supports "
                "the detail. Every item must cite a completed-turn "
                "source_message_id and copy evidence_quote exactly from that "
                "source message. For character age, fill only when directly "
                "stated or clearly evidenced. Do not invent unsupported "
                "details, rewrite nonblank fields, locked fields, player "
                "decisions, or facts that conflict with the completed turn."
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    request.scenario_context,
                    _world_data_enrichment_registry_text(request),
                    _messages_text(request.messages),
                )
            ),
        ),
    )


def _world_data_enrichment_tool_messages(
    request: WorldDataEnrichmentRequest,
) -> tuple[ToolCallMessage, ...]:
    messages = _world_data_enrichment_messages(request)
    tool_messages: list[ToolCallMessage] = []
    for message in messages:
        body = message.body.replace(
            "Use the enforced response schema.",
            (
                "Use the provided enrichment tools instead of prose. Call "
                "enrich_location, enrich_active_thread, or enrich_character only "
                "for listed sparse records and only with details that should fill "
                "blank fields. For character age, emit only directly stated or "
                "clearly evidenced age."
            ),
        )
        tool_messages.append(
            ToolCallMessage(
                role=message.role,
                body=body,
                speaker_name=message.speaker_name,
            )
        )
    return tuple(tool_messages)


def _world_data_enrichment_tool_definitions(
    request: WorldDataEnrichmentRequest,
) -> tuple[ToolDefinition, ...]:
    schemas = _world_data_enrichment_tool_schemas(request)
    return (
        ToolDefinition(
            name="enrich_location",
            description="Fill blank descriptive fields for one sparse location.",
            parameters=schemas["enrich_location"],
        ),
        ToolDefinition(
            name="enrich_active_thread",
            description="Fill blank descriptive fields for one sparse active thread.",
            parameters=schemas["enrich_active_thread"],
        ),
        ToolDefinition(
            name="enrich_character",
            description="Fill blank profile fields for one sparse character.",
            parameters=schemas["enrich_character"],
        ),
    )


def _world_data_enrichment_tool_schemas(
    request: WorldDataEnrichmentRequest,
) -> dict[str, dict[str, object]]:
    source_schema: dict[str, object] = {"type": "string"}
    message_ids = [message.id for message in request.messages]
    if message_ids:
        source_schema["enum"] = message_ids
    location_id_schema: dict[str, object] = {
        "type": "string",
        "enum": [location.id for location in request.locations],
    }
    thread_id_schema: dict[str, object] = {
        "type": "string",
        "enum": [thread.id for thread in request.active_threads],
    }
    character_id_schema: dict[str, object] = {
        "type": "string",
        "enum": [character.id for character in request.characters],
    }
    base_properties: dict[str, object] = {
        "source_message_id": source_schema,
        "evidence_quote": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "enrich_location": _tool_schema(
            required=["location_id", "source_message_id", "evidence_quote"],
            properties={
                **base_properties,
                "location_id": location_id_schema,
                "description": {"type": "string"},
                "visual_description": {"type": "string"},
                "connections": string_array,
                "status": {"type": "string"},
                "hazards": string_array,
            },
        ),
        "enrich_active_thread": _tool_schema(
            required=["active_thread_id", "source_message_id", "evidence_quote"],
            properties={
                **base_properties,
                "active_thread_id": thread_id_schema,
                "description": {"type": "string"},
                "related_entities": string_array,
            },
        ),
        "enrich_character": _tool_schema(
            required=["character_id", "source_message_id", "evidence_quote"],
            properties={
                **base_properties,
                "character_id": character_id_schema,
                "aliases": string_array,
                "role": {"type": "string"},
                "age": {"type": "string"},
                "known_state": {"type": "string"},
                "appearance": {"type": "string"},
                "visual_notes": {"type": "string"},
                "current_clothing": {
                    "type": "string",
                    "description": (
                        "Current outfit or worn items directly evidenced in "
                        "the source messages."
                    ),
                },
                "personality": {"type": "string"},
                "voice": {"type": "string"},
                "relationships": {"type": "object"},
                "status": {"type": "string"},
            },
        ),
    }


def _registry_text(request: ContextUpdateRequest) -> str:
    lines = ["Current normalized context registry:"]
    if request.scene_snapshot is None:
        lines.append("- scene: none")
    else:
        snapshot = request.scene_snapshot
        world_time = format_world_time_from_snapshot(snapshot)
        lines.append(
            "- scene: "
            f"situation={snapshot.situation}; objective={snapshot.objective}; "
            f"time={world_time}; weather={snapshot.weather}; "
            f"mood={snapshot.mood}; "
            f"locked(read-only)={','.join(snapshot.locked_fields)}"
        )
    locations, omitted = _limited_items(
        request.locations,
        MAX_CONTEXT_UPDATE_IDENTITY_LOCATIONS,
    )
    for location in locations:
        lines.append(
            "- location "
            f"{location.id}: {location.name}; status={location.status}; "
            f"locked(read-only)={','.join(location.locked_fields)}"
        )
    _append_omitted_count(lines, "locations", omitted)
    characters, omitted = _limited_items(
        request.characters,
        MAX_CONTEXT_UPDATE_IDENTITY_CHARACTERS,
    )
    for character in characters:
        agency_parts = [
            f"goals={character.goals}" if character.goals else "",
            f"motivations={character.motivations}" if character.motivations else "",
            (
                f"current_intent={character.current_intent}"
                if character.current_intent
                else ""
            ),
            f"boundaries={character.boundaries}" if character.boundaries else "",
            (
                f"attitude={character.attitude_toward_player}"
                if character.attitude_toward_player
                else ""
            ),
            (
                f"cooperation={character.cooperation_conditions}"
                if character.cooperation_conditions
                else ""
            ),
        ]
        agency_text = "; ".join(part for part in agency_parts if part)
        lines.append(
            "- character "
            f"{character.id}: {character.name}; status={character.status}; "
            f"age={character.age}; "
            f"current_clothing={character.current_clothing}; "
            f"agency={agency_text or 'none'}; "
            f"locked(read-only)={','.join(character.locked_fields)}"
        )
    _append_omitted_count(lines, "characters", omitted)
    threads, omitted = _limited_items(
        request.active_threads,
        MAX_CONTEXT_UPDATE_IDENTITY_THREADS,
    )
    for thread in threads:
        lines.append(
            "- active_thread "
            f"{thread.id}: {thread.title}; status={thread.status}; "
            f"priority={thread.priority}; "
            f"locked(read-only)={','.join(thread.locked_fields)}"
        )
    _append_omitted_count(lines, "active threads", omitted)
    lines.extend(_selected_prior_context_text(request.prior_context))
    return "\n".join(lines)


def _world_data_enrichment_registry_text(
    request: WorldDataEnrichmentRequest,
) -> str:
    lines = ["Sparse world data records to enrich:"]
    if request.scene_snapshot is not None:
        snapshot = request.scene_snapshot
        world_time = format_world_time_from_snapshot(snapshot)
        lines.append(
            "- scene: "
            f"situation={snapshot.situation}; objective={snapshot.objective}; "
            f"time={world_time}; weather={snapshot.weather}; "
            f"mood={snapshot.mood}"
        )
    locations, omitted = _limited_items(
        request.locations,
        MAX_CONTEXT_UPDATE_IDENTITY_LOCATIONS,
    )
    for location in locations:
        lines.append(
            "- location "
            f"{location.id}: {location.name}; "
            f"description={location.description}; "
            f"visual={location.visual_description}; "
            f"status={location.status}; "
            f"connections={', '.join(location.connections)}; "
            f"hazards={', '.join(location.hazards)}; "
            f"locked(read-only)={','.join(location.locked_fields)}"
        )
    _append_omitted_count(lines, "locations", omitted)
    threads, omitted = _limited_items(
        request.active_threads,
        MAX_CONTEXT_UPDATE_IDENTITY_THREADS,
    )
    for thread in threads:
        lines.append(
            "- active_thread "
            f"{thread.id}: {thread.title}; "
            f"description={thread.description}; "
            f"status={thread.status}; "
            f"related={', '.join(thread.related_entities)}; "
            f"locked(read-only)={','.join(thread.locked_fields)}"
        )
    _append_omitted_count(lines, "active threads", omitted)
    characters, omitted = _limited_items(
        request.characters,
        MAX_CONTEXT_UPDATE_IDENTITY_CHARACTERS,
    )
    for character in characters:
        lines.append(
            "- character "
            f"{character.id}: {character.name}; role={character.role}; "
            f"age={character.age}; "
            f"known_state={character.known_state}; "
            f"appearance={character.appearance}; "
            f"current_clothing={character.current_clothing}; "
            f"personality={character.personality}; "
            f"voice={character.voice}; "
            f"status={character.status}; "
            f"locked(read-only)={','.join(character.locked_fields)}"
        )
    _append_omitted_count(lines, "characters", omitted)
    lines.extend(_selected_prior_context_text(request.prior_context))
    return "\n".join(lines)


def _world_data_enrichment_request(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    messages: tuple[MessageRecord, ...],
    prior_context: tuple[ContextRegistryItem, ...] = (),
    candidate_character_ids: frozenset[str] = frozenset(),
) -> WorldDataEnrichmentRequest:
    locations = tuple(
        location
        for location in repositories.list_locations(save_id)
        if _location_needs_enrichment(location)
    )
    active_threads = tuple(
        thread
        for thread in repositories.list_active_threads(save_id)
        if active_thread_is_prompt_visible(thread)
        if _thread_needs_enrichment(thread)
    )
    characters = tuple(
        character
        for character in repositories.list_characters(save_id)
        if character.id in candidate_character_ids
        if _character_needs_enrichment(character)
    )
    return WorldDataEnrichmentRequest(
        save_id=save_id,
        messages=messages,
        scenario_context=_scenario_context_text(repositories, save_id),
        scene_snapshot=repositories.get_scene_snapshot(save_id),
        locations=locations,
        active_threads=active_threads,
        characters=characters,
        memories=tuple(repositories.list_memories(save_id)),
        world_state=tuple(repositories.list_world_state(save_id)),
        summaries=tuple(repositories.list_summaries(save_id)),
        prior_context=prior_context,
    )


def _scenario_context_text(
    repositories: PersistenceRepositories,
    save_id: str,
) -> str:
    save = repositories.get_save(save_id)
    if save is None:
        return "Scenario context: unknown save"
    scenario = repositories.get_scenario(save.scenario_id)
    if scenario is None:
        return "Scenario context: unknown scenario"
    lines = [
        "Scenario context:",
        f"- title: {scenario.title}",
        f"- premise: {scenario.premise}",
        f"- player_role: {scenario.player_role}",
    ]
    try:
        content = json.loads(scenario.content_json)
    except json.JSONDecodeError:
        content = {}
    if isinstance(content, dict):
        for key, value in content.items():
            if key.startswith("_"):
                continue
            text = _scenario_content_value_text(value)
            if text:
                lines.append(f"- {key}: {text}")
    return "\n".join(lines)


def _scenario_content_value_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return "; ".join(
            f"{key}={text}"
            for key, raw in value.items()
            if (text := _scenario_content_value_text(raw))
        )
    return str(value).strip() if value is not None else ""


def _messages_text(messages: tuple[MessageRecord, ...]) -> str:
    if not messages:
        return "Completed turn messages: none"
    return "Completed turn messages:\n" + "\n".join(
        f"- {message.id} [{message.role}] {message.body}" for message in messages
    )


def _object_list(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        cast(dict[str, object], item) for item in value if isinstance(item, dict)
    )


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _optional_string_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _string_tuple(value)


def _relationships_from_data(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, list):
        return {}
    relationships: dict[str, object] = {}
    for item in value:
        if isinstance(item, dict):
            name = _string(item.get("name")).strip()
            description = _string(item.get("description")).strip()
            if name:
                relationships[name] = description
        elif isinstance(item, str) and item.strip():
            relationships[item.strip()] = ""
    return relationships


def _object_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _confidence(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 1.0
    return min(max(float(value), 0.0), 1.0)


def _clean_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _is_probable_opaque_identifier(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    return bool(fullmatch(r"[0-9a-f]{24,64}", text, flags=IGNORECASE))


def _merge_strings(existing: Iterable[str], proposed: Iterable[str]) -> list[str]:
    return list(dict.fromkeys((*existing, *_clean_strings(proposed))))


def _normalize_scene_time(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = sub(r"\s+", " ", value.strip())
    if not text:
        return ""
    key = text.strip(" .,:;!?\"'").casefold()
    if not key or key in _VAGUE_SCENE_TIME_VALUES:
        return ""
    if not _has_supported_time_signal(key):
        return ""
    non_timer_key = text_without_timer_readout_clauses(key)
    clock = search(
        r"\b([01]?\d|2[0-3])(?::([0-5]\d))?\s*([ap]\.?m\.?)\b",
        non_timer_key,
        flags=IGNORECASE,
    )
    if clock is not None:
        hour = int(clock.group(1))
        minute = clock.group(2)
        meridiem = clock.group(3).replace(".", "").upper()
        display_hour = hour
        if meridiem == "AM" and hour == 12:
            normalized_hour = 0
        elif meridiem == "PM" and hour != 12:
            normalized_hour = hour + 12
        else:
            normalized_hour = hour
        clock_text = (
            f"{display_hour}:{minute} {meridiem}"
            if minute
            else f"{display_hour} {meridiem}"
        )
        return f"{clock_text} ({_phase_for_hour(normalized_hour)})"
    clock_24h = first_non_timer_24h_clock(key)
    if clock_24h:
        hour_text, minute_text = clock_24h.split(":", maxsplit=1)
        hour = int(hour_text)
        clock_text = f"{hour}:{minute_text}"
        return f"{clock_text} ({_phase_for_hour(hour)})"
    return _phase_from_text(non_timer_key)


def _has_supported_time_signal(text: str) -> bool:
    if timer_readout_without_clock_advance(text):
        return False
    phase = _phase_from_text(text)
    if phase:
        return True
    if (
        search(r"\b([01]?\d|2[0-3])(?::[0-5]\d)?\s*[ap]\.?m\.?\b", text)
        is not None
    ):
        return True
    return search(r"\b([01]?\d|2[0-3]):[0-5]\d\b", text) is not None


def _phase_from_text(text: str) -> str:
    for phrase, phase in _SCENE_TIME_PHRASES:
        if search(rf"\b{phrase}\b", text):
            return phase
    return ""


def _phase_for_hour(hour: int) -> str:
    if 5 <= hour <= 9:
        return "morning"
    if 10 <= hour <= 11:
        return "late morning"
    if 12 <= hour <= 16:
        return "afternoon"
    if 17 <= hour <= 20:
        return "evening"
    return "night"


def _player_authorizes_scene_time_change(messages: tuple[MessageRecord, ...]) -> bool:
    player_text = " ".join(
        message.body for message in messages if message.role == "player"
    )
    return has_world_time_advance_signal(player_text)


def _resolve_character(
    characters: Iterable[CharacterRecord],
    name: str,
) -> _CharacterResolution:
    records = tuple(characters)
    proposed_key = _character_name_key(name)
    if not proposed_key:
        return _CharacterResolution()
    proposed_is_short = len(proposed_key.split()) == 1
    if proposed_is_short:
        matches = _short_name_character_matches(records, proposed_key)
        if len(matches) == 1:
            return _CharacterResolution(record=matches[0])
        if len(matches) > 1:
            return _CharacterResolution(ambiguous=True)
        return _CharacterResolution()

    exact = _find_character(records, name)
    if exact is not None:
        return _CharacterResolution(record=exact)

    full_name_matches: list[CharacterRecord] = []
    matched_ids: set[str] = set()
    for character in records:
        if character.id in matched_ids:
            continue
        if _is_existing_short_primary_name(character, proposed_key):
            full_name_matches.append(character)
            matched_ids.add(character.id)
    if len(full_name_matches) == 1:
        return _CharacterResolution(record=full_name_matches[0])
    if len(full_name_matches) > 1:
        return _CharacterResolution(ambiguous=True)
    return _CharacterResolution()


def _short_name_character_matches(
    characters: Iterable[CharacterRecord],
    proposed_key: str,
) -> list[CharacterRecord]:
    matches: list[CharacterRecord] = []
    matched_ids: set[str] = set()
    for character in characters:
        if character.id in matched_ids:
            continue
        if _matches_short_character_name(character, proposed_key):
            matches.append(character)
            matched_ids.add(character.id)
    return matches


def _matches_short_character_name(
    character: CharacterRecord,
    proposed_key: str,
) -> bool:
    for known_name in (character.name, *character.aliases):
        known_key = _character_name_key(known_name)
        if not known_key:
            continue
        known_parts = known_key.split()
        known_short_name = known_parts[0]
        if proposed_key == known_key or proposed_key == known_short_name:
            return True
    return False


def _is_existing_short_primary_name(
    character: CharacterRecord,
    proposed_key: str,
) -> bool:
    known_key = _character_name_key(character.name)
    if not known_key:
        return False
    known_parts = known_key.split()
    proposed_parts = proposed_key.split()
    return len(known_parts) == 1 and known_parts[0] == proposed_parts[0]


def _should_add_character_alias(character: CharacterRecord, alias: str) -> bool:
    alias = alias.strip()
    if not alias:
        return False
    alias_key = alias.casefold()
    return character.name.casefold() != alias_key and alias_key not in {
        existing.casefold() for existing in character.aliases
    }


def _character_name_key(value: str) -> str:
    text = sub(r"\s+", " ", value.strip()).casefold()
    if not text:
        return ""
    parts = text.split()
    while parts and parts[0].rstrip(".") in _CHARACTER_TITLE_WORDS:
        parts = parts[1:]
    return " ".join(parts) or text


def _should_add_location_alias(location: LocationRecord, alias: str) -> bool:
    alias = alias.strip()
    if not alias:
        return False
    alias_key = _canonical_entity_name_key(alias)
    known = {
        _canonical_entity_name_key(value)
        for value in (location.name, *location.aliases)
        if value
    }
    return alias_key not in known


def _find_location(
    locations: Iterable[LocationRecord],
    name: str,
) -> LocationRecord | None:
    keys = _location_match_keys(name)
    if not keys:
        return None
    for location in locations:
        known_keys = {
            key
            for value in (location.name, *location.aliases)
            for key in _location_match_keys(value)
        }
        if keys & known_keys:
            return location
    return None


def _find_character(
    characters: Iterable[CharacterRecord],
    name: str,
) -> CharacterRecord | None:
    key = name.strip().casefold()
    if not key:
        return None
    for character in characters:
        if character.name.casefold() == key:
            return character
        if key in {alias.casefold() for alias in character.aliases}:
            return character
    return None


def _find_thread(
    threads: Iterable[ActiveThreadRecord],
    title: str,
) -> ActiveThreadRecord | None:
    key = _canonical_entity_name_key(title)
    if not key:
        return None
    for thread in threads:
        if _canonical_entity_name_key(thread.title) == key:
            return thread
    return None


def _location_match_keys(value: str) -> set[str]:
    key = _canonical_entity_name_key(value)
    if not key:
        return set()
    keys = {key}
    parts = key.split()
    if len(parts) == 2 and parts[0].endswith("s") and parts[1] in {
        "bar",
        "cafe",
        "diner",
        "pub",
        "restaurant",
        "shop",
        "store",
        "workshop",
    }:
        keys.add(parts[1])
    return keys


def _canonical_entity_name_key(value: str) -> str:
    text = value.strip().casefold()
    if not text:
        return ""
    text = text.replace("'", "")
    text = sub(r"[_/\\-]+", " ", text)
    text = sub(r"[^\w\s]", " ", text)
    parts = sub(r"\s+", " ", text).strip().split()
    while parts and parts[0] in {"a", "an", "the"}:
        parts = parts[1:]
    if len(parts) >= 2 and parts[1] == "s":
        parts = [parts[0] + "s", *parts[2:]]
    return " ".join(parts)


def _find_world_state_id(records: Iterable[WorldStateRecord], key: str) -> str | None:
    normalized = key.strip().casefold()
    if not normalized:
        return None
    records = tuple(records)
    for record in records:
        if record.key.casefold() == normalized:
            return record.id
    target_key = _canonical_entity_name_key(key)
    matches: list[WorldStateRecord] = []
    for record in records:
        if target_key in _world_state_match_keys(record):
            matches.append(record)
    unique_ids = {record.id for record in matches}
    if len(unique_ids) == 1:
        return matches[0].id
    return None


def _world_state_match_keys(record: WorldStateRecord) -> set[str]:
    keys: set[str] = set()
    key = _canonical_entity_name_key(record.key)
    if key:
        keys.add(key)
    leaf = record.key.rsplit(".", maxsplit=1)[-1]
    leaf_key = _canonical_entity_name_key(leaf)
    if leaf_key:
        keys.add(leaf_key)
    if isinstance(record.value, dict):
        for field in ("name", "title", "label"):
            value = record.value.get(field)
            if isinstance(value, str):
                value_key = _canonical_entity_name_key(value)
                if value_key:
                    keys.add(value_key)
    return keys


def _find_memory_id(records: Iterable[MemoryRecord], body: str) -> str | None:
    normalized = body.strip().casefold()
    if not normalized:
        return None
    for record in records:
        if record.body.casefold() == normalized:
            return record.id
    return None


def _find_summary_id(records: Iterable[SummaryRecord], body: str) -> str | None:
    normalized = body.strip().casefold()
    if not normalized:
        return None
    for record in records:
        if record.body.casefold() == normalized:
            return record.id
    return None


def _find_entity_link(
    links: Iterable[EntityLinkRecord],
    *,
    entity_type: str,
    entity_id: str,
    target_type: str,
    target_id: str,
    relation: str,
) -> EntityLinkRecord | None:
    for link in links:
        if (
            link.entity_type == entity_type
            and link.entity_id == entity_id
            and link.target_type == target_type
            and link.target_id == target_id
            and link.relation == relation
        ):
            return link
    return None


def _is_empty_update(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _field_is_blank_and_unlocked(record: object, field_path: str) -> bool:
    if isinstance(record, CharacterRecord):
        if character_field_is_locked(getattr(record, "locked_fields", []), field_path):
            return False
    if field_path in set(getattr(record, "locked_fields", [])):
        return False
    return _is_empty_update(getattr(record, field_path))


def _record_field_is_locked(
    entity_type: str,
    record: object,
    field_path: str,
) -> bool:
    locked_fields = getattr(record, "locked_fields", [])
    if _normalized_entity_type(entity_type) == "character":
        return character_field_is_locked(locked_fields, field_path)
    return field_path in set(locked_fields)


def _location_needs_enrichment(location: LocationRecord) -> bool:
    return any(
        _field_is_blank_and_unlocked(location, field)
        for field in ("description", "visual_description", "status")
    )


def _thread_needs_enrichment(thread: ActiveThreadRecord) -> bool:
    return any(
        _field_is_blank_and_unlocked(thread, field)
        for field in ("description",)
    )


def _character_needs_enrichment(character: CharacterRecord) -> bool:
    return any(
        _field_is_blank_and_unlocked(character, field)
        for field in (
            "aliases",
            "role",
            "age",
            "known_state",
            "appearance",
            "visual_notes",
            "personality",
            "voice",
        )
    )


def _merge_applied_context_updates(
    first: AppliedContextUpdate,
    second: AppliedContextUpdate,
) -> AppliedContextUpdate:
    locations = list(first.locations)
    for location in second.locations:
        _append_unique(locations, location)
    characters = list(first.characters)
    for character in second.characters:
        _append_unique(characters, character)
    active_threads = list(first.active_threads)
    for thread in second.active_threads:
        _append_unique(active_threads, thread)
    entity_links = list(first.entity_links)
    for link in second.entity_links:
        _append_unique(entity_links, link)
    return AppliedContextUpdate(
        scene_snapshot=second.scene_snapshot or first.scene_snapshot,
        locations=tuple(locations),
        characters=tuple(characters),
        active_threads=tuple(active_threads),
        entity_links=tuple(entity_links),
        suggestions=(*first.suggestions, *second.suggestions),
        audit_entries=(*first.audit_entries, *second.audit_entries),
    )


_LINKABLE_ENTITY_TYPES = frozenset(
    {"location", "character", "active_thread", "memory", "world_state", "summary"}
)

_VOLATILE_CONTEXT_FIELDS = frozenset(
    {
        ("active_thread", "status"),
        ("active_thread", "visibility"),
        ("character", "status"),
        ("character", "current_clothing"),
        *{("character", field_name) for field_name in CHARACTER_AGENCY_FIELDS},
        ("scene_snapshot", "current_location_id"),
        ("scene_snapshot", "situation"),
        ("scene_snapshot", "objective"),
        ("scene_snapshot", "in_world_time"),
        ("scene_snapshot", "time_of_day"),
        ("scene_snapshot", "day_of_week"),
        ("scene_snapshot", "weather"),
        ("scene_snapshot", "mood"),
        ("scene_snapshot", "nearby_objects"),
        ("scene_snapshot", "hazards"),
        ("scene_snapshot", "present_character_ids"),
    }
)


def _is_volatile_context_field(entity_type: str, field_path: str) -> bool:
    key = (_normalized_entity_type(entity_type), field_path)
    return key in _VOLATILE_CONTEXT_FIELDS


def _is_append_only_character_history_field(entity_type: str, field_path: str) -> bool:
    return _normalized_entity_type(entity_type) == "character" and field_path in {
        "known_state",
        "history",
    }


def _append_character_history(before: object, value: object) -> str:
    existing = str(before or "").strip()
    proposed = str(value or "").strip()
    if not proposed:
        return existing
    if not existing:
        return proposed
    if proposed.casefold() in existing.casefold():
        return existing
    return f"{existing}\n\n{proposed}"


def _scene_field_allows_empty_replacement(field_path: str) -> bool:
    return field_path in {"nearby_objects", "hazards", "present_character_ids"}


def _scene_field_is_locked(snapshot: SceneSnapshotRecord, field_path: str) -> bool:
    return scene_snapshot_field_is_locked(snapshot.locked_fields, field_path)


def _is_conflicting_rewrite(before: object, after: object) -> bool:
    if _is_empty_update(before) or _is_empty_update(after) or before == after:
        return False
    if isinstance(before, list) and isinstance(after, list):
        return not set(before).issubset(set(after))
    return True


def _replace_scene_snapshot_field(
    snapshot: SceneSnapshotRecord,
    field_path: str,
    value: object,
) -> SceneSnapshotRecord:
    if field_path == "current_location_id":
        return replace(snapshot, current_location_id=cast(str | None, value))
    if field_path == "situation":
        return replace(snapshot, situation=cast(str, value))
    if field_path == "objective":
        return replace(snapshot, objective=cast(str, value))
    if field_path == "in_world_time":
        return replace(snapshot, in_world_time=cast(str, value))
    if field_path == "time_of_day":
        return replace(snapshot, time_of_day=cast(str, value))
    if field_path == "day_of_week":
        return replace(snapshot, day_of_week=cast(str, value))
    if field_path == "world_day_index":
        return replace(snapshot, world_day_index=_int_or_none(value))
    if field_path == "weather":
        return replace(snapshot, weather=cast(str, value))
    if field_path == "mood":
        return replace(snapshot, mood=cast(str, value))
    if field_path == "nearby_objects":
        return replace(snapshot, nearby_objects=cast(list[str], value))
    if field_path == "hazards":
        return replace(snapshot, hazards=cast(list[str], value))
    if field_path == "present_character_ids":
        return replace(snapshot, present_character_ids=cast(list[str], value))
    raise ValueError(f"Unsupported scene snapshot field: {field_path}")


def _append_unique(records: list[Any], record: Any) -> None:
    record_id = record.id
    for index, existing in enumerate(records):
        if existing.id == record_id:
            records[index] = record
            return
    records.append(record)


def _normalized_entity_type(entity_type: str) -> str:
    value = entity_type.strip().casefold()
    if value in {"thread", "active_thread"}:
        return "active_thread"
    if value in {"state", "world_state"}:
        return "world_state"
    return value


def _location_audit_value(location: LocationRecord) -> dict[str, object]:
    return {
        "name": location.name,
        "aliases": location.aliases,
        "description": location.description,
        "visual_description": location.visual_description,
        "parent_location_id": location.parent_location_id,
        "connections": location.connections,
        "status": location.status,
        "hazards": location.hazards,
    }


def _character_audit_value(character: CharacterRecord) -> dict[str, object]:
    return {
        "name": character.name,
        "aliases": character.aliases,
        "role": character.role,
        "age": character.age,
        "known_state": character.known_state,
        "met": character.met,
        "appearance": character.appearance,
        "visual_notes": character.visual_notes,
        "current_clothing": character.current_clothing,
        "personality": character.personality,
        "voice": character.voice,
        "relationships": character.relationships,
        "goals": character.goals,
        "motivations": character.motivations,
        "current_intent": character.current_intent,
        "boundaries": character.boundaries,
        "attitude_toward_player": character.attitude_toward_player,
        "cooperation_conditions": character.cooperation_conditions,
        "status": character.status,
        "location_id": character.location_id,
        "private_notes": character.private_notes,
    }


def _thread_audit_value(thread: ActiveThreadRecord) -> dict[str, object]:
    return {
        "title": thread.title,
        "description": thread.description,
        "status": thread.status,
        "priority": thread.priority,
        "visibility": thread.visibility,
        "related_entities": thread.related_entities,
    }


def _scene_audit_value(snapshot: SceneSnapshotRecord) -> dict[str, object]:
    return {
        "current_location_id": snapshot.current_location_id,
        "situation": snapshot.situation,
        "objective": snapshot.objective,
        "in_world_time": snapshot.in_world_time,
        "world_day_index": snapshot.world_day_index,
        "weather": snapshot.weather,
        "mood": snapshot.mood,
        "nearby_objects": snapshot.nearby_objects,
        "hazards": snapshot.hazards,
        "present_character_ids": snapshot.present_character_ids,
    }


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
