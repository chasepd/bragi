"""Context selection before narrator turns."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from threading import Lock
from time import perf_counter
from typing import cast

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import (
    CharacterKnowledgeEdgeRecord,
    CharacterRecord,
    ContextObservationRecord,
    ContextSourceRecord,
    EntityLinkRecord,
    MediaAssetRecord,
    MemoryRecord,
    MessageRecord,
    MessageVisibilityRecord,
    ScenarioRecord,
    SceneSnapshotRecord,
    StateChangeRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import (
    PersistenceRepositories,
    canonical_claim_fingerprint,
)
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
from bragi.providers.errors import ProviderError
from bragi.redaction import redact_text
from bragi.services.context_assembly import scenario_section_candidates
from bragi.services.continuity_index_service import ContinuityIndexService
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.knowledge_boundary import (
    ScopedTargets,
    allowed_character_scoped_targets,
    character_scope_for_turn,
    message_visible_to_present_characters,
    scoped_owner_name,
)
from bragi.services.mention_matching import character_name_is_mentioned
from bragi.services.model_capabilities import (
    MODEL_LACKS_CAPABILITY_REASON,
    MODEL_MISSING_REASON,
    MODEL_UNAVAILABLE_REASON,
    STRUCTURED_OUTPUT_CAPABILITIES,
    TOOL_CALLING_CAPABILITIES,
    check_model_capabilities,
    find_provider_model,
    model_supports_any_capability,
    normalized_capabilities,
)
from bragi.services.model_preferences import roleplay_model_preference
from bragi.services.narration_context import (
    NarrationContextSnapshot,
    load_narration_context_snapshot,
)
from bragi.services.open_threads import is_open_threads_aggregate_key
from bragi.services.openrouter_routing_settings import (
    request_with_openrouter_routing,
)
from bragi.services.provider_fallbacks import (
    structured_output_fallback_request,
    structured_output_fallback_skip_reason,
    structured_output_with_fallback,
    tool_call_fallback_request,
    tool_call_fallback_skip_reason,
)
from bragi.services.request_budget import (
    budget_structured_output_request,
    budget_tool_call_request,
)
from bragi.services.tool_call_helpers import (
    CONTEXT_SEARCH_TOOL_RETRY_INSTRUCTION,
    accepted_tool_result,
    invalid_tool_result,
    parse_tool_arguments_json,
    validate_tool_arguments_shape,
)
from bragi.text_search import cjk_lexical_anchors, unicode_word_terms

MAX_CONTEXT_SELECTIONS = 16
MAX_CONTEXT_SEARCH_TOOL_FEEDBACK_TURNS = 2
MAX_CONTEXT_CANDIDATE_POOL = 96
MAX_CONTEXT_CANDIDATE_TEXT_CHARS = 700
MAX_CONTEXT_RESULT_TEXT_CHARS = 700
RECENT_MESSAGE_CANDIDATE_LIMIT = 24
STATE_CHANGE_CANDIDATE_LIMIT = 16
MEDIA_ASSET_CANDIDATE_LIMIT = 8
INDEXED_CONTEXT_SOURCE_RETRIEVAL_LIMIT = 80
PROTECTED_CONTEXT_SOURCE_LIMIT = 32
MAX_CONTEXT_QUERY_CHARS = 8_000
MAX_CONTEXT_QUERY_TERMS = 64
MAX_EXACT_RAW_STRUCTURED_IDENTIFIERS = 16
MAX_CONTEXT_EXACT_PHRASE_CHARS = 512
MAX_CONTEXT_EXACT_PHRASES = 4
CONTEXT_SEARCH_MESSAGE_LOAD_LIMIT = 64
RAW_CONTEXT_RECORD_LIMIT = 512
MAX_CONTEXT_SOURCE_PROVENANCE_GROUPS = 64
MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS = 64
INDEXED_CONTEXT_SOURCE_TYPES = frozenset(
    {
        "character_text_thread",
        "open_obligation",
        "scenario_section",
        "world_state",
        "memory",
        "observation",
        "character_voice",
    }
)
MEDIA_PROMPT_EXCERPT_MAX_CHARS = 400
DATA_PAYLOAD_PATTERN = re.compile(
    r"\bdata:[^\s;]+;base64,(?:[A-Za-z0-9+/=]{20,}\s*)+"
)
PRIVATE_MEDIA_PATH_PATTERN = re.compile(
    r"\b(?:[\w.-]+/)*(?:media/private|private/media)/[^\s]+"
)
WINDOWS_PATH_PATTERN = re.compile(r"\b[A-Za-z]:\\[^\s]+")
ABSOLUTE_PATH_PATTERN = re.compile(r"(?<!\w)/(?:[^/\s]+/)+[^\s]+")
RELATIVE_PATH_PATTERN = re.compile(
    r"\b(?:\.{1,2}/)?(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{2,8}\b"
)
BASE64_LIKE_TOKEN_PATTERN = re.compile(r"\b(?:[A-Za-z0-9+/]{40,}={0,2}\s*){2,}")
CONTEXT_SEARCH_SELECTION_TOOL = "select_context_source"
CONTEXT_RETRIEVAL_EXPANSION_TOOL = "expand_context_retrieval"
CONTEXT_RETRIEVAL_RECOVERY_PROVIDER_FALLBACK = "provider_fallback"
CONTEXT_RETRIEVAL_RECOVERY_DETERMINISTIC = "deterministic_fallback"


@dataclass(frozen=True)
class SelectedContextItem:
    source_type: str
    source_id: str
    text: str
    relevance_note: str
    excerpted: bool = False

    def format_for_prompt(self) -> str:
        return f"[{self.source_type}:{self.source_id}] {self.text}"


@dataclass(frozen=True)
class ContextSearchResult:
    selected_open_obligations: tuple[SelectedContextItem, ...] = ()
    selected_scenario_sections: tuple[SelectedContextItem, ...] = ()
    selected_state: tuple[SelectedContextItem, ...] = ()
    selected_state_changes: tuple[SelectedContextItem, ...] = ()
    selected_media_assets: tuple[SelectedContextItem, ...] = ()
    selected_character_text_context: tuple[SelectedContextItem, ...] = ()
    selected_memories: tuple[SelectedContextItem, ...] = ()
    selected_observations: tuple[SelectedContextItem, ...] = ()
    selected_character_voice: tuple[SelectedContextItem, ...] = ()
    selected_summaries: tuple[SelectedContextItem, ...] = ()
    selected_recent_messages: tuple[SelectedContextItem, ...] = ()
    continuity_index_synced: bool = False
    retrieval_degraded: bool = False
    retrieval_recovery: str | None = None
    narration_snapshot: NarrationContextSnapshot | None = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class _ContextCandidate:
    source_type: str
    source_id: str
    text: str
    selection_text: str | None = None


@dataclass(frozen=True)
class _ContextCandidateDiagnostics:
    candidate_count_before_narrowing: int
    candidate_count_after_narrowing: int
    source_type_counts_before_narrowing: dict[str, int]
    source_type_counts_after_narrowing: dict[str, int]
    dropped_source_type_counts: dict[str, int]
    observation_status_counts: dict[str, int]
    included_observation_status_counts: dict[str, int]
    excluded_observation_status_counts: dict[str, int]
    curated_observation_candidate_count: int
    suppressed_raw_observation_count: int
    retrieval_diagnostics: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_count_before_narrowing": (
                self.candidate_count_before_narrowing
            ),
            "candidate_count_after_narrowing": self.candidate_count_after_narrowing,
            "source_type_counts_before_narrowing": (
                self.source_type_counts_before_narrowing
            ),
            "source_type_counts_after_narrowing": (
                self.source_type_counts_after_narrowing
            ),
            "dropped_source_type_counts": self.dropped_source_type_counts,
            "observation_status_counts": self.observation_status_counts,
            "included_observation_status_counts": (
                self.included_observation_status_counts
            ),
            "excluded_observation_status_counts": (
                self.excluded_observation_status_counts
            ),
            "curated_observation_candidate_count": (
                self.curated_observation_candidate_count
            ),
            "suppressed_raw_observation_count": (
                self.suppressed_raw_observation_count
            ),
            **self.retrieval_diagnostics,
        }


@dataclass(frozen=True)
class _ContextCandidateSet:
    candidates: tuple[_ContextCandidate, ...]
    diagnostics: _ContextCandidateDiagnostics


@dataclass(frozen=True)
class _IndexedContextSourceRetrieval:
    records: tuple[ContextSourceRecord, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _NextTurnContextCacheEntry:
    fingerprint: str
    snapshot: NarrationContextSnapshot
    digest: str


@dataclass(frozen=True)
class _ContextSelectionOutcome:
    result: ContextSearchResult
    fallback_allowed: bool = False
    primary_provider: str | None = None
    primary_model_id: str | None = None
    final_provider: str | None = None
    final_model_id: str | None = None
    fallback_used: bool = False
    fallback_provider: str | None = None
    fallback_model_id: str | None = None
    fallback_skipped_reason: str | None = None
    error_category: str | None = None
    http_status: int | None = None


class ContextSearchService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.jobs = JobLifecycleService(repositories=repositories)
        self._next_turn_cache: dict[str, _NextTurnContextCacheEntry] = {}
        self._next_turn_cache_lock = Lock()

    def precompute_next_turn(self, save_id: str) -> None:
        started_at = perf_counter()
        job = self.jobs.create_running(
            save_id=save_id,
            type="context_precompute",
            payload={"scope": "next_turn_context_candidates"},
        )
        try:
            details = self.repositories.load_save_details(
                save_id,
                message_limit=CONTEXT_SEARCH_MESSAGE_LOAD_LIMIT,
            )
            if details is not None:
                _sync_continuity_index_for_search(self.repositories, save_id)
            initial_fingerprint = self.repositories.context_candidate_revision_token(
                save_id
            )
            snapshot = load_narration_context_snapshot(
                self.repositories,
                save_id=save_id,
                details=details,
                include_context_sources=False,
                raw_record_limit=RAW_CONTEXT_RECORD_LIMIT,
            )
            if snapshot is None:
                raise ValueError(f"Unknown save id: {save_id}")
            retrieved_sources = _indexed_context_source_retrieval(
                self.repositories,
                save_id=save_id,
                latest_player_message="",
                scene_snapshot=snapshot.scene_snapshot,
                characters=list(snapshot.characters),
                character_knowledge_edges=list(snapshot.character_knowledge_edges),
                entity_links=list(snapshot.entity_links),
                message_visibility=list(snapshot.message_visibility),
            )
            candidate_observations = _observations_with_indexed_sources(
                self.repositories,
                save_id=save_id,
                observations=snapshot.observations,
                context_sources=retrieved_sources.records,
            )
            candidates = _next_turn_context_candidates(
                scenario=snapshot.details.scenario,
                scene_snapshot=snapshot.scene_snapshot,
                characters=list(snapshot.characters),
                character_knowledge_edges=list(snapshot.character_knowledge_edges),
                message_visibility=list(snapshot.message_visibility),
                entity_links=list(snapshot.entity_links),
                world_state=list(snapshot.world_state),
                world_state_for_scope=list(snapshot.world_state_for_scope),
                state_changes=list(snapshot.state_changes),
                media_assets=list(snapshot.media_assets),
                memories=list(snapshot.memories),
                summaries=list(snapshot.summaries),
                observations=list(candidate_observations),
                context_sources=list(retrieved_sources.records),
                recent_messages=snapshot.details.messages,
                include_missing_raw_candidates=False,
            )
            final_fingerprint = self.repositories.context_candidate_revision_token(
                save_id
            )
            cache_digest = _candidate_digest(candidates)
            cache_status = "stored"
            if initial_fingerprint != final_fingerprint:
                cache_status = "skipped_stale"
                with self._next_turn_cache_lock:
                    existing = self._next_turn_cache.get(save_id)
                    if (
                        existing is not None
                        and existing.fingerprint != final_fingerprint
                    ):
                        self._next_turn_cache.pop(save_id, None)
            else:
                with self._next_turn_cache_lock:
                    self._next_turn_cache[save_id] = _NextTurnContextCacheEntry(
                        fingerprint=final_fingerprint,
                        snapshot=snapshot,
                        digest=cache_digest,
                    )
            result = {
                "scope": "next_turn_context_candidates",
                "cache_status": cache_status,
                "candidate_count": len(candidates),
                "source_type_counts": _source_type_counts_json(candidates),
                "indexed_candidate_count": sum(
                    1 for candidate in candidates if candidate.source_type != "message"
                ),
                "cache_digest": cache_digest,
                "duration_ms": _elapsed_ms(started_at),
                **retrieved_sources.diagnostics,
            }
        except asyncio.CancelledError:
            try:
                self.jobs.cancel(job.id, error="Context search cancelled")
            except ValueError:
                pass
            raise
        except Exception as exc:
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                **exception_log_fields(exc),
            )
            raise
        self.jobs.succeed(job.id, result=result)
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            cache_status=result["cache_status"],
            candidate_count=result["candidate_count"],
        )

    async def search(
        self,
        *,
        save_id: str,
        player_message_id: str,
    ) -> ContextSearchResult:
        return await self._search(
            save_id=save_id,
            player_message_id=player_message_id,
            focus_message=None,
        )

    async def search_for_focus(
        self,
        *,
        save_id: str,
        focus_message: MessageRecord,
    ) -> ContextSearchResult:
        return await self._search(
            save_id=save_id,
            player_message_id=focus_message.id,
            focus_message=focus_message,
        )

    async def _search(
        self,
        *,
        save_id: str,
        player_message_id: str,
        focus_message: MessageRecord | None,
    ) -> ContextSearchResult:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="context_search",
        )
        if preference is None:
            raise ValueError("No context-search model preference configured")
        job = self.jobs.create_running(
            save_id=save_id,
            type="context_search",
            payload=(
                {"player_message_id": player_message_id}
                if focus_message is None
                else {
                    "focus": "look_around",
                    "query_chars": len(focus_message.body),
                }
            ),
            collect_provider_diagnostics=True,
        )
        log_event(
            "job.running",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
        )
        try:
            provider = self.providers[preference.provider]
            details = self.repositories.load_save_details(
                save_id,
                message_limit=CONTEXT_SEARCH_MESSAGE_LOAD_LIMIT,
            )
            if details is None:
                raise ValueError(f"Unknown save id: {save_id}")
            messages = (
                details.messages
                if focus_message is None
                else [*details.messages, focus_message]
            )
            scenario = details.scenario if details is not None else None
            player_message = _message_body(messages, player_message_id)
            cache_entry = (
                self._valid_next_turn_cache(
                    save_id,
                    player_message_id=player_message_id,
                )
                if focus_message is None
                else None
            )
            continuity_index_synced = False
            narration_snapshot: NarrationContextSnapshot | None = None
            if cache_entry is None:
                _sync_continuity_index_for_search(self.repositories, save_id)
                continuity_index_synced = True
                narration_snapshot = load_narration_context_snapshot(
                    self.repositories,
                    save_id=save_id,
                    details=details,
                    include_context_sources=False,
                    raw_record_limit=RAW_CONTEXT_RECORD_LIMIT,
                )
                if narration_snapshot is None:
                    raise ValueError(f"Unknown save id: {save_id}")
                messages = _context_search_visible_messages(
                    self.repositories,
                    save_id=save_id,
                    scene_snapshot=narration_snapshot.scene_snapshot,
                    required_messages=tuple(
                        message
                        for message in messages
                        if message.id == player_message_id
                    ),
                )
                retrieved_sources = await _retrieve_indexed_context_sources(
                    self.repositories,
                    provider=provider,
                    provider_name=preference.provider,
                    model_id=preference.model_id,
                    save_id=save_id,
                    latest_player_message=player_message,
                    scene_snapshot=narration_snapshot.scene_snapshot,
                    characters=list(narration_snapshot.characters),
                    character_knowledge_edges=list(
                        narration_snapshot.character_knowledge_edges
                    ),
                    entity_links=list(narration_snapshot.entity_links),
                    recent_messages=messages,
                    message_visibility=list(narration_snapshot.message_visibility),
                )
                candidate_observations = _observations_with_indexed_sources(
                    self.repositories,
                    save_id=save_id,
                    observations=narration_snapshot.observations,
                    context_sources=retrieved_sources.records,
                )
                candidate_set = _context_candidate_set(
                    scenario=scenario,
                    scene_snapshot=narration_snapshot.scene_snapshot,
                    characters=list(narration_snapshot.characters),
                    character_knowledge_edges=list(
                        narration_snapshot.character_knowledge_edges
                    ),
                    message_visibility=list(narration_snapshot.message_visibility),
                    entity_links=list(narration_snapshot.entity_links),
                    world_state=list(narration_snapshot.world_state),
                    world_state_for_scope=(
                        list(narration_snapshot.world_state_for_scope)
                    ),
                    state_changes=list(narration_snapshot.state_changes),
                    media_assets=list(narration_snapshot.media_assets),
                    memories=list(narration_snapshot.memories),
                    summaries=list(narration_snapshot.summaries),
                    observations=list(candidate_observations),
                    context_sources=list(retrieved_sources.records),
                    recent_messages=messages,
                    player_message_id=player_message_id,
                    include_missing_raw_candidates=False,
                    retrieval_diagnostics=retrieved_sources.diagnostics,
                )
                candidates = candidate_set.candidates
                candidate_diagnostics = candidate_set.diagnostics
                cache_status = "miss"
            else:
                fresh_pending_suggestions = tuple(
                    self.repositories.list_context_update_suggestions(
                        save_id,
                        status="pending",
                        limit=RAW_CONTEXT_RECORD_LIMIT,
                    )
                )
                pending_source_message_ids = {
                    source_id
                    for suggestion in fresh_pending_suggestions
                    for source_id in suggestion.source_message_ids
                }
                present_character_ids = (
                    set(cache_entry.snapshot.scene_snapshot.present_character_ids)
                    if cache_entry.snapshot.scene_snapshot is not None
                    else set()
                )
                refreshed_pending_visibility = tuple(
                    self.repositories.list_message_visibility(
                        save_id,
                        character_ids=present_character_ids,
                        message_ids=pending_source_message_ids,
                    )
                )
                snapshot = replace(
                    cache_entry.snapshot,
                    details=details,
                    pending_context_suggestions=fresh_pending_suggestions,
                    message_visibility=tuple(
                        visibility
                        for visibility in cache_entry.snapshot.message_visibility
                        if visibility.message_id not in pending_source_message_ids
                    ),
                )
                snapshot = replace(
                    snapshot,
                    message_visibility=(
                        *snapshot.message_visibility,
                        *refreshed_pending_visibility,
                    ),
                )
                narration_snapshot = snapshot
                messages = _context_search_visible_messages(
                    self.repositories,
                    save_id=save_id,
                    scene_snapshot=snapshot.scene_snapshot,
                    required_messages=tuple(
                        message
                        for message in messages
                        if message.id == player_message_id
                    ),
                )
                retrieved_sources = await _retrieve_indexed_context_sources(
                    self.repositories,
                    provider=provider,
                    provider_name=preference.provider,
                    model_id=preference.model_id,
                    save_id=save_id,
                    latest_player_message=player_message,
                    scene_snapshot=snapshot.scene_snapshot,
                    characters=list(snapshot.characters),
                    character_knowledge_edges=list(
                        snapshot.character_knowledge_edges
                    ),
                    entity_links=list(snapshot.entity_links),
                    recent_messages=messages,
                    message_visibility=list(snapshot.message_visibility),
                )
                candidate_observations = _observations_with_indexed_sources(
                    self.repositories,
                    save_id=save_id,
                    observations=snapshot.observations,
                    context_sources=retrieved_sources.records,
                )
                candidate_set = _context_candidate_set(
                    scenario=scenario,
                    scene_snapshot=snapshot.scene_snapshot,
                    characters=list(snapshot.characters),
                    character_knowledge_edges=list(
                        snapshot.character_knowledge_edges
                    ),
                    message_visibility=list(snapshot.message_visibility),
                    entity_links=list(snapshot.entity_links),
                    world_state=list(snapshot.world_state),
                    world_state_for_scope=list(snapshot.world_state_for_scope),
                    state_changes=list(snapshot.state_changes),
                    media_assets=list(snapshot.media_assets),
                    memories=list(snapshot.memories),
                    summaries=list(snapshot.summaries),
                    observations=list(candidate_observations),
                    context_sources=list(retrieved_sources.records),
                    recent_messages=messages,
                    player_message_id=player_message_id,
                    include_missing_raw_candidates=False,
                    retrieval_diagnostics=retrieved_sources.diagnostics,
                )
                candidates = candidate_set.candidates
                candidate_diagnostics = candidate_set.diagnostics
                cache_status = "hit"
                continuity_index_synced = True
            log_event(
                "context_search.candidates_built",
                save_id=save_id,
                player_message_id=player_message_id,
                candidate_count=len(candidates),
                cache_status=cache_status,
                **_candidate_count_fields(candidates),
            )
            preselection_revision = (
                self.repositories.context_candidate_revision_token(
                    save_id,
                    ignored_message_id=player_message_id,
                )
            )
            requirement_error = _model_requirement_error(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
            )
            primary_model_unavailable = _model_is_unavailable(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
            )
            if requirement_error is not None and not primary_model_unavailable:
                raise ValueError(requirement_error)
            supports_tool_calling = _model_supports_tool_calling(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
            )
            supports_structured_output = _model_supports_structured_output(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
            )
            advertises_tool_calling = _model_advertises_tool_calling(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
            )
            advertises_structured_output = _model_advertises_structured_output(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
            )
            if (
                not primary_model_unavailable
                and not supports_tool_calling
                and not supports_structured_output
            ):
                raise ValueError(
                    "Context-search model does not advertise structured output "
                    "or tool calling"
                )
            tool_provider = (
                cast(ToolCallProvider, provider)
                if supports_tool_calling and isinstance(provider, ToolCallProvider)
                else None
            )
            structured_provider = (
                cast(StructuredOutputProvider, provider)
                if supports_structured_output
                and isinstance(provider, StructuredOutputProvider)
                else None
            )
            if (
                not primary_model_unavailable
                and tool_provider is None
                and structured_provider is None
            ):
                raise ValueError(
                    "Context-search provider does not support structured output "
                    "or tool calling"
                )
            if not candidates:
                selection = _ContextSelectionOutcome(
                    ContextSearchResult(),
                    primary_provider=preference.provider,
                    primary_model_id=preference.model_id,
                )
            elif primary_model_unavailable and advertises_tool_calling:
                selection = await _select_context_with_unavailable_tool_route(
                    repositories=self.repositories,
                    providers=self.providers,
                    provider_name=preference.provider,
                    model_id=preference.model_id,
                    save_id=save_id,
                    scenario=scenario,
                    player_message=player_message,
                    candidates=candidates,
                )
            elif primary_model_unavailable and advertises_structured_output:
                selection = await _select_context_with_unavailable_structured_route(
                    repositories=self.repositories,
                    providers=self.providers,
                    provider_name=preference.provider,
                    model_id=preference.model_id,
                    save_id=save_id,
                    scenario=scenario,
                    player_message=player_message,
                    candidates=candidates,
                )
            elif primary_model_unavailable:
                raise ValueError(
                    requirement_error or "Context-search model is unavailable"
                )
            elif tool_provider is not None:
                selection = await _select_context_with_tool_calls(
                    repositories=self.repositories,
                    providers=self.providers,
                    provider=tool_provider,
                    provider_name=preference.provider,
                    model_id=preference.model_id,
                    save_id=save_id,
                    scenario=scenario,
                    player_message=player_message,
                    candidates=candidates,
                )
            else:
                selection = await _select_context_with_structured_output(
                    repositories=self.repositories,
                    providers=self.providers,
                    provider_name=preference.provider,
                    model_id=preference.model_id,
                    save_id=save_id,
                    scenario=scenario,
                    player_message=player_message,
                    candidates=candidates,
                )
            result = selection.result
            if (
                candidates
                and selection.fallback_allowed
                and _context_result_is_empty(result)
            ):
                result = _fallback_context_result(candidates)
                result = replace(
                    result,
                    retrieval_degraded=True,
                    retrieval_recovery=CONTEXT_RETRIEVAL_RECOVERY_DETERMINISTIC,
                )
                selection = replace(
                    selection,
                    result=result,
                    fallback_allowed=False,
                    fallback_used=False,
                )
                log_event(
                    "context_search.fallback_selected",
                    save_id=save_id,
                    player_message_id=player_message_id,
                    candidate_count=len(candidates),
                    scenario_section_count=len(result.selected_scenario_sections),
                    open_obligation_count=len(result.selected_open_obligations),
                    state_count=len(result.selected_state),
                    state_change_count=len(result.selected_state_changes),
                    media_asset_count=len(result.selected_media_assets),
                    character_text_context_count=len(
                        result.selected_character_text_context
                    ),
                    memory_count=len(result.selected_memories),
                    observation_count=len(result.selected_observations),
                    character_voice_count=len(result.selected_character_voice),
                    summary_count=len(result.selected_summaries),
                    recent_message_count=len(result.selected_recent_messages),
                )
            result, narration_snapshot = _rehydrate_selected_context(
                repositories=self.repositories,
                save_id=save_id,
                player_message_id=player_message_id,
                focus_message=focus_message,
                result=result,
                fallback_snapshot=narration_snapshot,
                preselection_revision=preselection_revision,
            )
            result = replace(
                result,
                continuity_index_synced=continuity_index_synced,
                narration_snapshot=narration_snapshot,
            )
            log_event(
                "context_search.context_selected",
                save_id=save_id,
                player_message_id=player_message_id,
                scenario_section_count=len(result.selected_scenario_sections),
                open_obligation_count=len(result.selected_open_obligations),
                state_count=len(result.selected_state),
                state_change_count=len(result.selected_state_changes),
                media_asset_count=len(result.selected_media_assets),
                character_text_context_count=len(
                    result.selected_character_text_context
                ),
                memory_count=len(result.selected_memories),
                observation_count=len(result.selected_observations),
                character_voice_count=len(result.selected_character_voice),
                summary_count=len(result.selected_summaries),
                recent_message_count=len(result.selected_recent_messages),
            )
        except asyncio.CancelledError:
            try:
                self.jobs.cancel(job.id, error="Context search cancelled")
            except ValueError:
                pass
            raise
        except Exception as exc:
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                **exception_log_fields(exc),
            )
            raise
        self.jobs.succeed(
            job.id,
            result=_result_json(
                result,
                diagnostics={
                    "cache_status": cache_status,
                    **candidate_diagnostics.to_json(),
                },
                selection=selection,
            ),
        )
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
        )
        return result

    def _valid_next_turn_cache(
        self,
        save_id: str,
        *,
        player_message_id: str,
    ) -> _NextTurnContextCacheEntry | None:
        with self._next_turn_cache_lock:
            entry = self._next_turn_cache.get(save_id)
        if entry is None:
            return None
        if entry.fingerprint == self.repositories.context_candidate_revision_token(
            save_id,
            ignored_message_id=player_message_id,
        ):
            return entry
        with self._next_turn_cache_lock:
            current_entry = self._next_turn_cache.get(save_id)
            if current_entry is entry:
                self._next_turn_cache.pop(save_id, None)
        log_event("context_search.precompute_cache_stale", save_id=save_id)
        return None


def _sync_continuity_index_for_search(
    repositories: PersistenceRepositories,
    save_id: str,
) -> None:
    result = ContinuityIndexService(repositories).sync_save(save_id)
    if not result.complete:
        raise RuntimeError(
            "Continuity index maintenance backlog is still draining; retry the turn"
        )


def _rehydrate_selected_context(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    player_message_id: str,
    focus_message: MessageRecord | None,
    result: ContextSearchResult,
    fallback_snapshot: NarrationContextSnapshot | None,
    preselection_revision: str,
) -> tuple[ContextSearchResult, NarrationContextSnapshot | None]:
    selected_items = _selected_context_items(result)
    if not selected_items:
        return result, fallback_snapshot

    current_revision = repositories.context_candidate_revision_token(
        save_id,
        ignored_message_id=player_message_id,
    )
    if current_revision != preselection_revision:
        _sync_continuity_index_for_search(repositories, save_id)
    details = repositories.load_save_details(
        save_id,
        message_limit=CONTEXT_SEARCH_MESSAGE_LOAD_LIMIT,
    )
    if details is None:
        raise ValueError(f"Unknown save id: {save_id}")
    messages = (
        details.messages
        if focus_message is None
        else [*details.messages, focus_message]
    )
    snapshot = load_narration_context_snapshot(
        repositories,
        save_id=save_id,
        details=details,
        include_context_sources=False,
        raw_record_limit=RAW_CONTEXT_RECORD_LIMIT,
    )
    if snapshot is None:
        raise ValueError(f"Unknown save id: {save_id}")
    messages = _context_search_visible_messages(
        repositories,
        save_id=save_id,
        scene_snapshot=snapshot.scene_snapshot,
        required_messages=tuple(
            message for message in messages if message.id == player_message_id
        ),
    )
    selected_context_sources = tuple(
        repositories.list_context_sources_by_keys(
            save_id,
            {
                (item.source_type, item.source_id)
                for item in selected_items
            },
        )
    )
    selected_source_message_ids = {
        source_id
        for source in selected_context_sources
        for source_id in _context_source_message_ids(source)
    }
    selected_message_visibility = tuple(
        repositories.list_message_visibility(
            save_id,
            character_ids=(
                set(snapshot.scene_snapshot.present_character_ids)
                if snapshot.scene_snapshot is not None
                else set()
            ),
            message_ids=selected_source_message_ids,
        )
    )
    candidate_message_visibility = list(
        {
            visibility.id: visibility
            for visibility in (
                *snapshot.message_visibility,
                *selected_message_visibility,
            )
        }.values()
    )
    candidate_observations = _observations_with_indexed_sources(
        repositories,
        save_id=save_id,
        observations=snapshot.observations,
        context_sources=selected_context_sources,
    )
    fresh_candidates = _context_candidate_set(
        scenario=details.scenario,
        scene_snapshot=snapshot.scene_snapshot,
        characters=list(snapshot.characters),
        character_knowledge_edges=list(snapshot.character_knowledge_edges),
        message_visibility=candidate_message_visibility,
        entity_links=list(snapshot.entity_links),
        world_state=list(snapshot.world_state),
        world_state_for_scope=list(snapshot.world_state_for_scope),
        state_changes=list(snapshot.state_changes),
        media_assets=list(snapshot.media_assets),
        memories=list(snapshot.memories),
        summaries=list(snapshot.summaries),
        observations=list(candidate_observations),
        context_sources=list(selected_context_sources),
        recent_messages=messages,
        player_message_id=player_message_id,
        include_missing_raw_candidates=True,
        narrow=False,
    ).candidates
    candidates_by_key = {
        (candidate.source_type, candidate.source_id): candidate
        for candidate in fresh_candidates
    }
    rehydrated_items: list[SelectedContextItem] = []
    dropped_keys: list[tuple[str, str]] = []
    for item in selected_items:
        key = (item.source_type, item.source_id)
        candidate = candidates_by_key.get(key)
        if candidate is None:
            dropped_keys.append(key)
            continue
        rehydrated_items.append(
            _selected_context_item(
                candidate,
                relevance_note=item.relevance_note,
            )
        )
    if dropped_keys:
        log_event(
            "context_search.stale_selected_sources_dropped",
            save_id=save_id,
            source_keys=[
                f"{source_type}:{source_id}"
                for source_type, source_id in dropped_keys
            ],
        )
    rehydrated = _context_result_from_items(rehydrated_items)
    return (
        replace(
            rehydrated,
            continuity_index_synced=result.continuity_index_synced,
            retrieval_degraded=result.retrieval_degraded,
            retrieval_recovery=result.retrieval_recovery,
        ),
        snapshot,
    )


def _indexed_context_source_retrieval(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    latest_player_message: str,
    scene_snapshot: SceneSnapshotRecord | None,
    characters: list[CharacterRecord],
    character_knowledge_edges: list[CharacterKnowledgeEdgeRecord],
    entity_links: list[EntityLinkRecord],
    message_visibility: list[MessageVisibilityRecord],
    additional_query_terms: tuple[str, ...] = (),
) -> _IndexedContextSourceRetrieval:
    started_at = perf_counter()
    query_text = _bounded_context_query_text(
        " ".join((latest_player_message, *additional_query_terms))
    )
    query_terms = list(_bounded_context_query_terms(query_text))
    exact_phrases = tuple(
        dict.fromkeys(
            phrase.strip()[:MAX_CONTEXT_EXACT_PHRASE_CHARS]
            for phrase in (latest_player_message, *additional_query_terms)
            if " " in phrase.strip()
        )
    )[:MAX_CONTEXT_EXACT_PHRASES]
    exact_identifiers = _bounded_structured_identifiers(query_text)
    turn_scope = character_scope_for_turn(
        scene_snapshot=scene_snapshot,
        characters=characters,
        latest_player_message=latest_player_message,
    )
    scoped_targets = allowed_character_scoped_targets(
        scene_snapshot=scene_snapshot,
        characters=characters,
        character_knowledge_edges=character_knowledge_edges,
        entity_links=entity_links,
        latest_player_message=latest_player_message,
        message_visibility=message_visibility,
    )
    allowed_owner_names = {
        scoped_owner_name(owner)
        for owners in scoped_targets.allowed.values()
        for owner in owners
    }
    current_turn_number = repositories.count_active_messages_by_role(
        save_id,
        roles=("narrator",),
    )["narrator"]
    reference_character_ids = set(turn_scope.reference_character_ids)
    reference_character_ids.difference_update(
        character.id for character in characters if character.is_player_character
    )
    current_scene_snapshot_id = (
        scene_snapshot.id if scene_snapshot is not None else None
    )
    current_scene_generation = (
        scene_snapshot.scene_generation if scene_snapshot is not None else None
    )
    repositories.archive_stale_scene_scratch(
        save_id=save_id,
        current_scene_snapshot_id=current_scene_snapshot_id,
        current_scene_generation=current_scene_generation,
        current_turn_number=current_turn_number,
    )
    protected = repositories.list_protected_context_sources(
        save_id,
        limit=PROTECTED_CONTEXT_SOURCE_LIMIT,
        allowed_owner_names=allowed_owner_names,
        reference_character_ids=reference_character_ids,
        visibility_character_ids=set(turn_scope.present_character_ids),
        current_scene_snapshot_id=current_scene_snapshot_id,
        current_scene_generation=current_scene_generation,
        current_turn_number=current_turn_number,
        blocked_source_keys=scoped_targets.blocked,
    )
    exact_hits = repositories.search_context_sources(
        save_id,
        query_terms=query_terms,
        source_types=INDEXED_CONTEXT_SOURCE_TYPES,
        limit=min(24, INDEXED_CONTEXT_SOURCE_RETRIEVAL_LIMIT),
        allowed_owner_names=allowed_owner_names,
        reference_character_ids=reference_character_ids,
        visibility_character_ids=set(turn_scope.present_character_ids),
        current_scene_snapshot_id=current_scene_snapshot_id,
        current_scene_generation=current_scene_generation,
        current_turn_number=current_turn_number,
        blocked_source_keys=scoped_targets.blocked,
        match_all=True,
        exact_phrases=exact_phrases,
        exact_identifiers=exact_identifiers,
    )
    broad_hits = repositories.search_context_sources(
        save_id,
        query_terms=query_terms,
        source_types=INDEXED_CONTEXT_SOURCE_TYPES,
        limit=INDEXED_CONTEXT_SOURCE_RETRIEVAL_LIMIT,
        allowed_owner_names=allowed_owner_names,
        reference_character_ids=reference_character_ids,
        visibility_character_ids=set(turn_scope.present_character_ids),
        current_scene_snapshot_id=current_scene_snapshot_id,
        current_scene_generation=current_scene_generation,
        current_turn_number=current_turn_number,
        blocked_source_keys=scoped_targets.blocked,
        exact_identifiers=exact_identifiers,
    )
    hits = tuple(
        {
            hit.record.id: hit
            for hit in (*exact_hits, *broad_hits)
        }.values()
    )
    records: list[ContextSourceRecord] = []
    seen_ids: set[str] = set()
    for record in protected:
        if record.id in seen_ids:
            continue
        seen_ids.add(record.id)
        records.append(record)
    for hit in hits:
        if hit.record.id in seen_ids:
            continue
        seen_ids.add(hit.record.id)
        records.append(hit.record)
    for record in repositories.list_curated_observation_source_markers(
        save_id,
        limit=RAW_CONTEXT_RECORD_LIMIT,
    ):
        if (
            record.id in seen_ids
            or record.metadata.get("curation_action")
            not in {"save_context", "scene_scratch"}
        ):
            continue
        seen_ids.add(record.id)
        records.append(
            replace(
                record,
                metadata={**record.metadata, "suppression_only": True},
            )
        )
    return _IndexedContextSourceRetrieval(
        records=tuple(records),
        diagnostics={
            "indexed_retrieval_enabled": True,
            "indexed_retrieval_query_term_count": len(query_terms),
            "indexed_retrieval_hit_count": len(hits),
            "protected_context_source_count": len(protected),
            "indexed_retrieval_duration_ms": _elapsed_ms(started_at),
            "first_pass_source_type_counts": _context_source_type_counts(records),
        },
    )


async def _retrieve_indexed_context_sources(
    repositories: PersistenceRepositories,
    *,
    provider: ProviderClient,
    provider_name: str,
    model_id: str,
    save_id: str,
    latest_player_message: str,
    scene_snapshot: SceneSnapshotRecord | None,
    characters: list[CharacterRecord],
    character_knowledge_edges: list[CharacterKnowledgeEdgeRecord],
    entity_links: list[EntityLinkRecord],
    recent_messages: list[MessageRecord],
    message_visibility: list[MessageVisibilityRecord],
) -> _IndexedContextSourceRetrieval:
    initial = _indexed_context_source_retrieval(
        repositories,
        save_id=save_id,
        latest_player_message=latest_player_message,
        scene_snapshot=scene_snapshot,
        characters=characters,
        character_knowledge_edges=character_knowledge_edges,
        entity_links=entity_links,
        message_visibility=message_visibility,
    )
    if not latest_player_message.strip():
        return initial
    supports_tools = (
        isinstance(provider, ToolCallProvider)
        and _model_supports_tool_calling(
            repositories=repositories,
            provider=provider_name,
            model_id=model_id,
        )
    )
    supports_structured = (
        isinstance(provider, StructuredOutputProvider)
        and _model_supports_structured_output(
            repositories=repositories,
            provider=provider_name,
            model_id=model_id,
        )
    )
    if supports_tools:
        expanded_terms = await _tool_retrieval_expansion(
            repositories=repositories,
            provider=cast(ToolCallProvider, provider),
            provider_name=provider_name,
            model_id=model_id,
            save_id=save_id,
            latest_player_message=latest_player_message,
            scene_snapshot=scene_snapshot,
            characters=characters,
            recent_messages=recent_messages,
            message_visibility=message_visibility,
        )
    elif supports_structured:
        expanded_terms = await _structured_retrieval_expansion(
            repositories=repositories,
            provider=cast(StructuredOutputProvider, provider),
            provider_name=provider_name,
            model_id=model_id,
            save_id=save_id,
            latest_player_message=latest_player_message,
            scene_snapshot=scene_snapshot,
            characters=characters,
            recent_messages=recent_messages,
            message_visibility=message_visibility,
        )
    else:
        return initial
    if not expanded_terms:
        return initial
    expanded = _indexed_context_source_retrieval(
        repositories,
        save_id=save_id,
        latest_player_message=latest_player_message,
        scene_snapshot=scene_snapshot,
        characters=characters,
        character_knowledge_edges=character_knowledge_edges,
        entity_links=entity_links,
        message_visibility=message_visibility,
        additional_query_terms=expanded_terms,
    )
    return replace(
        expanded,
        diagnostics={
            **expanded.diagnostics,
            "retrieval_expansion_used": True,
            "retrieval_expansion_term_count": len(expanded_terms),
        },
    )


async def _structured_retrieval_expansion(
    *,
    repositories: PersistenceRepositories,
    provider: StructuredOutputProvider,
    provider_name: str,
    model_id: str,
    save_id: str,
    latest_player_message: str,
    scene_snapshot: SceneSnapshotRecord | None,
    characters: list[CharacterRecord],
    recent_messages: list[MessageRecord],
    message_visibility: list[MessageVisibilityRecord],
) -> tuple[str, ...]:
    present_ids = frozenset(
        scene_snapshot.present_character_ids if scene_snapshot is not None else ()
    )
    visible_recent = _latest_visible_messages(
        recent_messages,
        limit=6,
        present_character_ids=present_ids,
        message_visibility=message_visibility,
    )
    eligible_characters = [
        character
        for character in characters
        if character.id in present_ids
        or character_name_is_mentioned(
            name=character.name,
            aliases=character.aliases,
            text=" ".join(message.body for message in visible_recent),
        )
    ]
    entity_ids = [character.id for character in eligible_characters]
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "terms": {
                "type": "array",
                "maxItems": 12,
                "items": {"type": "string"},
            },
            "phrases": {
                "type": "array",
                "maxItems": 6,
                "items": {"type": "string"},
            },
            "entity_ids": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "enum": entity_ids},
            },
        },
        "required": ["terms", "phrases", "entity_ids"],
    }
    entity_text = "\n".join(
        f"- {character.id}: {character.name}; aliases={', '.join(character.aliases)}"
        for character in eligible_characters
    ) or "none"
    recent_text = "\n".join(
        f"- {message.speaker_name or message.role}: {message.body}"
        for message in visible_recent
    )
    request = request_with_openrouter_routing(
        repositories,
        StructuredOutputRequest(
            provider=provider_name,
            model_id=model_id,
            schema_name="context_retrieval_expansion",
            schema=schema,
            messages=(
                ChatMessage(
                    role="system",
                    body=(
                        "Expand the player's continuity query with short synonymous "
                        "terms, phrases, and visible entity references. Use only the "
                        "provided entity IDs and enforced schema."
                    ),
                ),
                ChatMessage(
                    role="user",
                    body=(
                        f"Player query:\n{latest_player_message}\n\n"
                        f"Visible entities:\n{entity_text}\n\n"
                        f"Visible recent context:\n{recent_text}"
                    ),
                ),
            ),
            temperature=0.0,
        ),
        task="context_search",
        save_id=save_id,
    )
    try:
        response = await provider.generate_structured_output(
            budget_structured_output_request(
                repositories,
                request,
                task="context_search",
            )
        )
    except (ProviderError, ValueError, KeyError, TypeError, AssertionError):
        return ()
    data = response.data
    terms = _string_values(data.get("terms"), limit=12)
    phrases = _string_values(data.get("phrases"), limit=6)
    selected_ids = set(_string_values(data.get("entity_ids"), limit=8))
    entity_terms = tuple(
        value
        for character in eligible_characters
        if character.id in selected_ids
        for value in (character.name, *character.aliases)
    )
    return tuple(dict.fromkeys((*terms, *phrases, *entity_terms)))


async def _tool_retrieval_expansion(
    *,
    repositories: PersistenceRepositories,
    provider: ToolCallProvider,
    provider_name: str,
    model_id: str,
    save_id: str,
    latest_player_message: str,
    scene_snapshot: SceneSnapshotRecord | None,
    characters: list[CharacterRecord],
    recent_messages: list[MessageRecord],
    message_visibility: list[MessageVisibilityRecord],
) -> tuple[str, ...]:
    present_ids = frozenset(
        scene_snapshot.present_character_ids if scene_snapshot is not None else ()
    )
    visible_recent = _latest_visible_messages(
        recent_messages,
        limit=6,
        present_character_ids=present_ids,
        message_visibility=message_visibility,
    )
    eligible_characters = [
        character
        for character in characters
        if character.id in present_ids
        or character_name_is_mentioned(
            name=character.name,
            aliases=character.aliases,
            text=" ".join(message.body for message in visible_recent),
        )
    ]
    entity_ids = [character.id for character in eligible_characters]
    parameters: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "terms": {
                "type": "array",
                "maxItems": 12,
                "items": {"type": "string"},
            },
            "phrases": {
                "type": "array",
                "maxItems": 6,
                "items": {"type": "string"},
            },
            "entity_ids": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "enum": entity_ids},
            },
        },
        "required": ["terms", "phrases", "entity_ids"],
    }
    entity_text = "\n".join(
        f"- {character.id}: {character.name}; aliases={', '.join(character.aliases)}"
        for character in eligible_characters
    ) or "none"
    recent_text = "\n".join(
        f"- {message.speaker_name or message.role}: {message.body}"
        for message in visible_recent
    )
    request = request_with_openrouter_routing(
        repositories,
        ToolCallRequest(
            provider=provider_name,
            model_id=model_id,
            messages=(
                ToolCallMessage(
                    role="system",
                    body=(
                        "Call expand_context_retrieval once with short synonymous "
                        "terms, phrases, and visible entity references. Use only "
                        "the provided entity IDs."
                    ),
                ),
                ToolCallMessage(
                    role="user",
                    body=(
                        f"Player query:\n{latest_player_message}\n\n"
                        f"Visible entities:\n{entity_text}\n\n"
                        f"Visible recent context:\n{recent_text}"
                    ),
                ),
            ),
            tools=(
                ToolDefinition(
                    name=CONTEXT_RETRIEVAL_EXPANSION_TOOL,
                    description="Expand a continuity query for local retrieval.",
                    parameters=parameters,
                ),
            ),
            temperature=0.0,
        ),
        task="context_search",
        save_id=save_id,
    )
    try:
        response = await provider.generate_tool_calls(
            budget_tool_call_request(
                repositories,
                request,
                task="context_search",
            )
        )
    except (ProviderError, ValueError, KeyError, TypeError):
        return ()
    call = next(
        (
            item
            for item in response.tool_calls
            if item.name == CONTEXT_RETRIEVAL_EXPANSION_TOOL
        ),
        None,
    )
    if call is None:
        return ()
    data, parse_error = parse_tool_arguments_json(call.arguments_json)
    if data is None or parse_error is not None:
        return ()
    terms = _string_values(data.get("terms"), limit=12)
    phrases = _string_values(data.get("phrases"), limit=6)
    selected_ids = set(_string_values(data.get("entity_ids"), limit=8))
    entity_terms = tuple(
        value
        for character in eligible_characters
        if character.id in selected_ids
        for value in (character.name, *character.aliases)
    )
    return tuple(dict.fromkeys((*terms, *phrases, *entity_terms)))


def _string_values(value: object, *, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        text
        for item in value[:limit]
        if (text := str(item).strip())
    )


def _context_source_type_counts(
    records: list[ContextSourceRecord] | tuple[ContextSourceRecord, ...],
) -> dict[str, int]:
    return dict(Counter(record.source_type for record in records))


def _context_candidates(
    *,
    scenario: ScenarioRecord | None,
    world_state: list[WorldStateRecord],
    world_state_for_scope: list[WorldStateRecord] | None = None,
    state_changes: list[StateChangeRecord],
    media_assets: list[MediaAssetRecord],
    memories: list[MemoryRecord],
    summaries: list[SummaryRecord],
    observations: list[ContextObservationRecord] | None = None,
    context_sources: list[ContextSourceRecord] | None = None,
    recent_messages: list[MessageRecord],
    player_message_id: str,
    scene_snapshot: SceneSnapshotRecord | None = None,
    characters: list[CharacterRecord] | None = None,
    character_knowledge_edges: list[CharacterKnowledgeEdgeRecord] | None = None,
    message_visibility: list[MessageVisibilityRecord] | None = None,
    entity_links: list[EntityLinkRecord] | None = None,
    active_message_ids: set[str] | None = None,
    include_missing_raw_candidates: bool = True,
) -> tuple[_ContextCandidate, ...]:
    return _context_candidate_set(
        scenario=scenario,
        scene_snapshot=scene_snapshot,
        characters=characters,
        character_knowledge_edges=character_knowledge_edges,
        message_visibility=message_visibility,
        entity_links=entity_links,
        world_state=world_state,
        world_state_for_scope=world_state_for_scope,
        state_changes=state_changes,
        media_assets=media_assets,
        memories=memories,
        summaries=summaries,
        observations=observations,
        context_sources=context_sources,
        recent_messages=recent_messages,
        player_message_id=player_message_id,
        active_message_ids=active_message_ids,
        include_missing_raw_candidates=include_missing_raw_candidates,
    ).candidates


def _context_candidate_set(
    *,
    scenario: ScenarioRecord | None,
    world_state: list[WorldStateRecord],
    world_state_for_scope: list[WorldStateRecord] | None = None,
    state_changes: list[StateChangeRecord],
    media_assets: list[MediaAssetRecord],
    memories: list[MemoryRecord],
    summaries: list[SummaryRecord],
    observations: list[ContextObservationRecord] | None = None,
    context_sources: list[ContextSourceRecord] | None = None,
    recent_messages: list[MessageRecord],
    player_message_id: str,
    scene_snapshot: SceneSnapshotRecord | None = None,
    characters: list[CharacterRecord] | None = None,
    character_knowledge_edges: list[CharacterKnowledgeEdgeRecord] | None = None,
    message_visibility: list[MessageVisibilityRecord] | None = None,
    entity_links: list[EntityLinkRecord] | None = None,
    active_message_ids: set[str] | None = None,
    include_missing_raw_candidates: bool = True,
    retrieval_diagnostics: Mapping[str, object] | None = None,
    narrow: bool = True,
) -> _ContextCandidateSet:
    latest_player_message = _message_body(recent_messages, player_message_id)
    character_records = characters or []
    turn_scope = character_scope_for_turn(
        scene_snapshot=scene_snapshot,
        characters=character_records,
        latest_player_message=latest_player_message,
    )
    characters_by_id = {character.id: character for character in character_records}
    audience_reference_character_ids = frozenset(
        character_id
        for character_id in turn_scope.reference_character_ids
        if not (
            (character := characters_by_id.get(character_id)) is not None
            and character.is_player_character
        )
    )
    message_candidates = _message_candidates(
        recent_messages,
        player_message_id,
        present_character_ids=turn_scope.present_character_ids,
        message_visibility=message_visibility or [],
    )
    recent_message_ids = {candidate.source_id for candidate in message_candidates}
    if active_message_ids is None:
        active_message_ids = {message.id for message in recent_messages}
    scoped_targets = allowed_character_scoped_targets(
        scene_snapshot=scene_snapshot,
        characters=character_records,
        character_knowledge_edges=character_knowledge_edges or [],
        entity_links=entity_links or [],
        latest_player_message=latest_player_message,
        message_visibility=message_visibility or [],
    )
    context_source_records = context_sources or []
    observation_records = observations or []
    accepted_observation_ids = _accepted_observation_ids(observation_records)
    curated_observation_source_ids = _curated_observation_source_ids(
        context_source_records,
        memories=memories,
        accepted_observation_ids=accepted_observation_ids,
    )
    indexed_candidates = _indexed_context_candidates(
        context_source_records,
        scoped_targets=scoped_targets,
        reference_character_ids=audience_reference_character_ids,
        accepted_observation_ids=accepted_observation_ids,
        present_character_ids=turn_scope.present_character_ids,
        message_visibility=message_visibility or [],
    )
    raw_state_candidates = _state_candidates(
        world_state,
        scoped_targets=scoped_targets,
        exclude_open_thread_aggregates=bool(indexed_candidates),
        present_character_ids=turn_scope.present_character_ids,
        message_visibility=message_visibility or [],
    )
    raw_memory_candidates = _memory_candidates(
        memories,
        scoped_targets=scoped_targets,
        present_character_ids=turn_scope.present_character_ids,
        message_visibility=message_visibility or [],
        observations_by_id={
            observation.id: observation for observation in observation_records
        },
    )
    state_candidates = (
        raw_state_candidates if include_missing_raw_candidates else ()
    )
    memory_candidates = (
        raw_memory_candidates if include_missing_raw_candidates else ()
    )
    exact_raw_candidates = (
        ()
        if include_missing_raw_candidates
        else _exact_raw_candidates(
            (*raw_state_candidates, *raw_memory_candidates),
            indexed_candidates=indexed_candidates,
            latest_player_message=latest_player_message,
        )
    )
    observation_candidates = _observation_candidates(
        observation_records,
        excluded_observation_ids=curated_observation_source_ids,
        present_character_ids=turn_scope.present_character_ids,
        message_visibility=message_visibility or [],
    )
    if indexed_candidates:
        canonical_candidates = (
            *indexed_candidates,
            *exact_raw_candidates,
            *(
                _missing_raw_candidates(
                    state_candidates,
                    memory_candidates,
                    indexed_candidates=indexed_candidates,
                )
                if include_missing_raw_candidates
                else ()
            ),
        )
    elif include_missing_raw_candidates:
        canonical_candidates = (
            *_scenario_section_candidates(scenario),
            *state_candidates,
            *memory_candidates,
        )
    else:
        canonical_candidates = exact_raw_candidates
    candidates = (
        *canonical_candidates,
        *observation_candidates,
        *_state_change_candidates(
            state_changes,
            world_state_for_scope or world_state,
            active_message_ids,
            scoped_targets=scoped_targets,
            present_character_ids=turn_scope.present_character_ids,
            message_visibility=message_visibility or [],
        ),
        *_media_asset_candidates(media_assets, recent_message_ids),
        *message_candidates,
    )
    narrowed = (
        _narrow_context_candidates(
            candidates,
            latest_player_message=latest_player_message,
        )
        if narrow
        else candidates
    )
    return _ContextCandidateSet(
        candidates=narrowed,
        diagnostics=_candidate_diagnostics(
            before=candidates,
            after=narrowed,
            observations=observation_records,
            curated_observation_source_ids=curated_observation_source_ids,
            retrieval_diagnostics=retrieval_diagnostics,
        ),
    )


def _next_turn_context_candidates(
    *,
    scenario: ScenarioRecord | None,
    world_state: list[WorldStateRecord],
    world_state_for_scope: list[WorldStateRecord] | None = None,
    state_changes: list[StateChangeRecord],
    media_assets: list[MediaAssetRecord],
    memories: list[MemoryRecord],
    summaries: list[SummaryRecord],
    observations: list[ContextObservationRecord] | None = None,
    context_sources: list[ContextSourceRecord] | None = None,
    recent_messages: list[MessageRecord],
    scene_snapshot: SceneSnapshotRecord | None = None,
    characters: list[CharacterRecord] | None = None,
    character_knowledge_edges: list[CharacterKnowledgeEdgeRecord] | None = None,
    message_visibility: list[MessageVisibilityRecord] | None = None,
    entity_links: list[EntityLinkRecord] | None = None,
    include_missing_raw_candidates: bool = True,
) -> tuple[_ContextCandidate, ...]:
    return _context_candidates(
        scenario=scenario,
        scene_snapshot=scene_snapshot,
        characters=characters,
        character_knowledge_edges=character_knowledge_edges,
        message_visibility=message_visibility,
        entity_links=entity_links,
        world_state=world_state,
        world_state_for_scope=world_state_for_scope,
        state_changes=state_changes,
        media_assets=media_assets,
        memories=memories,
        summaries=summaries,
        observations=observations,
        context_sources=context_sources,
        recent_messages=recent_messages,
        player_message_id="",
        active_message_ids={message.id for message in recent_messages},
        include_missing_raw_candidates=include_missing_raw_candidates,
    )


async def _select_context_with_structured_output(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider_name: str,
    model_id: str,
    save_id: str,
    scenario: ScenarioRecord | None,
    player_message: str,
    candidates: tuple[_ContextCandidate, ...],
) -> _ContextSelectionOutcome:
    started_at = perf_counter()
    request = request_with_openrouter_routing(
        repositories,
        StructuredOutputRequest(
            provider=provider_name,
            model_id=model_id,
            schema_name="context_search_selection",
            schema=_context_selection_schema(candidates),
            messages=_context_selection_messages(
                scenario=scenario,
                player_message=player_message,
                candidates=candidates,
            ),
            temperature=0.0,
        ),
        task="context_search",
        save_id=save_id,
    )
    try:
        response = await structured_output_with_fallback(
            repositories=repositories,
            providers=providers,
            request=request,
            task="context_search",
            save_id=save_id,
        )
        result = _context_result_from_structured_data(response.data, candidates)
    except ProviderError as exc:
        log_error_event(
            "provider.structured_output_failed",
            provider=provider_name,
            model=model_id,
            task="context_search",
            duration_ms=_elapsed_ms(started_at),
            candidate_count=len(candidates),
            **exception_log_fields(exc),
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=provider_name,
            primary_model_id=model_id,
            exc=exc,
        )
    except (TimeoutError, ValueError, TypeError) as exc:
        log_error_event(
            "provider.structured_output_failed",
            provider=provider_name,
            model=model_id,
            task="context_search",
            duration_ms=_elapsed_ms(started_at),
            candidate_count=len(candidates),
            **exception_log_fields(exc),
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=provider_name,
            primary_model_id=model_id,
            error_category="schema_validation_failed",
        )
    log_event(
        "provider.structured_output_succeeded",
        provider=response.provider,
        model=response.model_id,
        task="context_search",
        duration_ms=_elapsed_ms(started_at),
        candidate_count=len(candidates),
        player_message_chars=len(player_message),
        token_usage=response.token_usage,
    )
    fallback_used = (
        response.provider != provider_name or response.model_id != model_id
    )
    if fallback_used:
        result = replace(
            result,
            retrieval_degraded=True,
            retrieval_recovery=CONTEXT_RETRIEVAL_RECOVERY_PROVIDER_FALLBACK,
        )
    return _ContextSelectionOutcome(
        result,
        primary_provider=provider_name,
        primary_model_id=model_id,
        final_provider=response.provider,
        final_model_id=response.model_id,
        fallback_used=fallback_used,
        fallback_provider=response.provider if fallback_used else None,
        fallback_model_id=response.model_id if fallback_used else None,
    )


async def _select_context_with_unavailable_structured_route(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider_name: str,
    model_id: str,
    save_id: str,
    scenario: ScenarioRecord | None,
    player_message: str,
    candidates: tuple[_ContextCandidate, ...],
) -> _ContextSelectionOutcome:
    request = request_with_openrouter_routing(
        repositories,
        StructuredOutputRequest(
            provider=provider_name,
            model_id=model_id,
            schema_name="context_search_selection",
            schema=_context_selection_schema(candidates),
            messages=_context_selection_messages(
                scenario=scenario,
                player_message=player_message,
                candidates=candidates,
            ),
            temperature=0.0,
        ),
        task="context_search",
        save_id=save_id,
    )
    log_event(
        "context_search.primary_model_unavailable",
        provider=provider_name,
        model=model_id,
        task="context_search",
    )
    return await _recover_context_structured_selection(
        repositories=repositories,
        providers=providers,
        request=request,
        candidates=candidates,
        save_id=save_id,
        primary_error=None,
    )


async def _recover_context_structured_selection(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    request: StructuredOutputRequest,
    candidates: tuple[_ContextCandidate, ...],
    save_id: str | None,
    primary_error: Exception | None,
) -> _ContextSelectionOutcome:
    try:
        fallback_request = structured_output_fallback_request(
            repositories=repositories,
            providers=providers,
            request=request,
            save_id=save_id,
            task="context_search",
        )
    except ProviderError as exc:
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_skipped_reason="fallback_request_over_budget",
            exc=exc,
        )
    if fallback_request is None:
        reason = structured_output_fallback_skip_reason(
            repositories=repositories,
            providers=providers,
            save_id=save_id,
        )
        log_event(
            "provider.structured_output_fallback_skipped",
            provider=request.provider,
            model=request.model_id,
            task="context_search",
            reason=reason,
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_skipped_reason=reason,
            exc=primary_error,
        )
    fallback_provider: object | None = providers.get(fallback_request.provider)
    if not isinstance(fallback_provider, StructuredOutputProvider):
        reason = "fallback_provider_unavailable"
        log_event(
            "provider.structured_output_fallback_skipped",
            provider=request.provider,
            model=request.model_id,
            task="context_search",
            reason=reason,
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_provider=fallback_request.provider,
            fallback_model_id=fallback_request.model_id,
            fallback_skipped_reason=reason,
            exc=primary_error,
        )
    log_event(
        "provider.structured_output_fallback_started",
        provider=fallback_request.provider,
        model=fallback_request.model_id,
        task="context_search",
    )
    try:
        response = await fallback_provider.generate_structured_output(
            fallback_request
        )
        result = _context_result_from_structured_data(response.data, candidates)
    except ProviderError as exc:
        log_error_event(
            "provider.structured_output_fallback_failed",
            provider=fallback_request.provider,
            model=fallback_request.model_id,
            task="context_search",
            candidate_count=len(candidates),
            **exception_log_fields(exc),
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_provider=fallback_request.provider,
            fallback_model_id=fallback_request.model_id,
            exc=exc,
        )
    except (TimeoutError, ValueError, TypeError) as exc:
        log_error_event(
            "provider.structured_output_fallback_failed",
            provider=fallback_request.provider,
            model=fallback_request.model_id,
            task="context_search",
            candidate_count=len(candidates),
            **exception_log_fields(exc),
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_provider=fallback_request.provider,
            fallback_model_id=fallback_request.model_id,
            error_category="schema_validation_failed",
        )
    log_event(
        "provider.structured_output_succeeded",
        provider=response.provider,
        model=response.model_id,
        task="context_search",
        candidate_count=len(candidates),
        token_usage=response.token_usage,
    )
    result = replace(
        result,
        retrieval_degraded=True,
        retrieval_recovery=CONTEXT_RETRIEVAL_RECOVERY_PROVIDER_FALLBACK,
    )
    return _ContextSelectionOutcome(
        result,
        primary_provider=request.provider,
        primary_model_id=request.model_id,
        final_provider=response.provider,
        final_model_id=response.model_id,
        fallback_used=True,
        fallback_provider=fallback_request.provider,
        fallback_model_id=fallback_request.model_id,
        error_category=_error_category(primary_error),
        http_status=_http_status(primary_error),
    )


async def _select_context_with_tool_calls(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider: ToolCallProvider,
    provider_name: str,
    model_id: str,
    save_id: str,
    scenario: ScenarioRecord | None,
    player_message: str,
    candidates: tuple[_ContextCandidate, ...],
) -> _ContextSelectionOutcome:
    started_at = perf_counter()
    request = request_with_openrouter_routing(
        repositories,
        ToolCallRequest(
            provider=provider_name,
            model_id=model_id,
            messages=_context_selection_tool_messages(
                scenario=scenario,
                player_message=player_message,
                candidates=candidates,
            ),
            tools=_context_selection_tool_definitions(candidates),
            temperature=0.0,
        ),
        task="context_search",
        save_id=save_id,
    )
    try:
        result = await _select_context_with_tool_feedback(
            repositories=repositories,
            provider=provider,
            request=request,
            candidates=candidates,
        )
    except _ContextSearchToolValidationFailed as exc:
        log_error_event(
            "provider.tool_call_failed",
            provider=provider_name,
            model=model_id,
            task="context_search",
            duration_ms=_elapsed_ms(started_at),
            candidate_count=len(candidates),
            error=str(exc),
        )
        return _ContextSelectionOutcome(
            ContextSearchResult(
                retrieval_degraded=True,
                retrieval_recovery=CONTEXT_RETRIEVAL_RECOVERY_DETERMINISTIC,
            ),
            fallback_allowed=True,
            primary_provider=provider_name,
            primary_model_id=model_id,
            final_provider=provider_name,
            final_model_id=model_id,
            error_category="schema_validation_failed",
        )
    except ProviderError as exc:
        log_error_event(
            "provider.tool_call_failed",
            provider=provider_name,
            model=model_id,
            task="context_search",
            duration_ms=_elapsed_ms(started_at),
            candidate_count=len(candidates),
            **exception_log_fields(exc),
        )
        return await _recover_context_tool_selection(
            repositories=repositories,
            providers=providers,
            request=request,
            candidates=candidates,
            save_id=save_id,
            primary_error=exc,
        )
    except TimeoutError as exc:
        log_error_event(
            "provider.tool_call_failed",
            provider=provider_name,
            model=model_id,
            task="context_search",
            duration_ms=_elapsed_ms(started_at),
            candidate_count=len(candidates),
            **exception_log_fields(exc),
        )
        return await _recover_context_tool_selection(
            repositories=repositories,
            providers=providers,
            request=request,
            candidates=candidates,
            save_id=save_id,
            primary_error=exc,
        )
    log_event(
        "provider.tool_call_succeeded",
        provider=provider_name,
        model=model_id,
        task="context_search",
        duration_ms=_elapsed_ms(started_at),
        candidate_count=len(candidates),
        player_message_chars=len(player_message),
    )
    return _ContextSelectionOutcome(
        result,
        primary_provider=provider_name,
        primary_model_id=model_id,
        final_provider=provider_name,
        final_model_id=model_id,
    )


async def _select_context_with_unavailable_tool_route(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider_name: str,
    model_id: str,
    save_id: str,
    scenario: ScenarioRecord | None,
    player_message: str,
    candidates: tuple[_ContextCandidate, ...],
) -> _ContextSelectionOutcome:
    request = request_with_openrouter_routing(
        repositories,
        ToolCallRequest(
            provider=provider_name,
            model_id=model_id,
            messages=_context_selection_tool_messages(
                scenario=scenario,
                player_message=player_message,
                candidates=candidates,
            ),
            tools=_context_selection_tool_definitions(candidates),
            temperature=0.0,
        ),
        task="context_search",
        save_id=save_id,
    )
    log_event(
        "context_search.primary_model_unavailable",
        provider=provider_name,
        model=model_id,
        task="context_search",
    )
    return await _recover_context_tool_selection(
        repositories=repositories,
        providers=providers,
        request=request,
        candidates=candidates,
        save_id=save_id,
        primary_error=None,
    )


async def _recover_context_tool_selection(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    request: ToolCallRequest,
    candidates: tuple[_ContextCandidate, ...],
    save_id: str | None,
    primary_error: Exception | None,
) -> _ContextSelectionOutcome:
    try:
        fallback_request = tool_call_fallback_request(
            repositories=repositories,
            providers=providers,
            request=request,
            save_id=save_id,
            task="context_search",
        )
    except ProviderError as exc:
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_skipped_reason="fallback_request_over_budget",
            exc=exc,
        )
    if fallback_request is None:
        reason = tool_call_fallback_skip_reason(
            repositories=repositories,
            providers=providers,
            save_id=save_id,
        )
        log_event(
            "provider.tool_call_fallback_skipped",
            provider=request.provider,
            model=request.model_id,
            task="context_search",
            reason=reason,
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_skipped_reason=reason,
            exc=primary_error,
        )
    fallback_provider: object | None = providers.get(fallback_request.provider)
    if not isinstance(fallback_provider, ToolCallProvider):
        reason = "fallback_provider_unavailable"
        log_event(
            "provider.tool_call_fallback_skipped",
            provider=request.provider,
            model=request.model_id,
            task="context_search",
            reason=reason,
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_provider=fallback_request.provider,
            fallback_model_id=fallback_request.model_id,
            fallback_skipped_reason=reason,
            exc=primary_error,
        )
    log_event(
        "provider.tool_call_fallback_started",
        provider=fallback_request.provider,
        model=fallback_request.model_id,
        task="context_search",
    )
    try:
        result = await _select_context_with_tool_feedback(
            repositories=repositories,
            provider=fallback_provider,
            request=fallback_request,
            candidates=candidates,
        )
    except _ContextSearchToolValidationFailed as exc:
        log_error_event(
            "provider.tool_call_failed",
            provider=fallback_request.provider,
            model=fallback_request.model_id,
            task="context_search",
            candidate_count=len(candidates),
            error=str(exc),
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_provider=fallback_request.provider,
            fallback_model_id=fallback_request.model_id,
            error_category="schema_validation_failed",
        )
    except ProviderError as exc:
        log_error_event(
            "provider.tool_call_fallback_failed",
            provider=fallback_request.provider,
            model=fallback_request.model_id,
            task="context_search",
            candidate_count=len(candidates),
            **exception_log_fields(exc),
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_provider=fallback_request.provider,
            fallback_model_id=fallback_request.model_id,
            exc=exc,
        )
    except TimeoutError as exc:
        log_error_event(
            "provider.tool_call_fallback_failed",
            provider=fallback_request.provider,
            model=fallback_request.model_id,
            task="context_search",
            candidate_count=len(candidates),
            **exception_log_fields(exc),
        )
        return _deterministic_context_selection(
            candidates,
            primary_provider=request.provider,
            primary_model_id=request.model_id,
            fallback_provider=fallback_request.provider,
            fallback_model_id=fallback_request.model_id,
            exc=exc,
        )
    result = replace(
        result,
        retrieval_degraded=True,
        retrieval_recovery=CONTEXT_RETRIEVAL_RECOVERY_PROVIDER_FALLBACK,
    )
    return _ContextSelectionOutcome(
        result,
        primary_provider=request.provider,
        primary_model_id=request.model_id,
        final_provider=fallback_request.provider,
        final_model_id=fallback_request.model_id,
        fallback_used=True,
        fallback_provider=fallback_request.provider,
        fallback_model_id=fallback_request.model_id,
        error_category=_error_category(primary_error),
        http_status=_http_status(primary_error),
    )


def _deterministic_context_selection(
    candidates: tuple[_ContextCandidate, ...],
    *,
    primary_provider: str,
    primary_model_id: str,
    fallback_provider: str | None = None,
    fallback_model_id: str | None = None,
    fallback_skipped_reason: str | None = None,
    exc: Exception | None = None,
    error_category: str | None = None,
    http_status: int | None = None,
) -> _ContextSelectionOutcome:
    result = replace(
        _fallback_context_result(candidates),
        retrieval_degraded=True,
        retrieval_recovery=CONTEXT_RETRIEVAL_RECOVERY_DETERMINISTIC,
    )
    return _ContextSelectionOutcome(
        result,
        primary_provider=primary_provider,
        primary_model_id=primary_model_id,
        fallback_used=False,
        fallback_provider=fallback_provider,
        fallback_model_id=fallback_model_id,
        fallback_skipped_reason=fallback_skipped_reason,
        error_category=error_category or _error_category(exc),
        http_status=http_status if http_status is not None else _http_status(exc),
    )


def _error_category(exc: Exception | None) -> str | None:
    return exc.category.value if isinstance(exc, ProviderError) else None


def _http_status(exc: Exception | None) -> int | None:
    return exc.status_code if isinstance(exc, ProviderError) else None


async def _select_context_with_tool_feedback(
    *,
    repositories: PersistenceRepositories,
    provider: ToolCallProvider,
    request: ToolCallRequest,
    candidates: tuple[_ContextCandidate, ...],
) -> ContextSearchResult:
    messages = list(request.messages)
    candidates_by_key = {
        (candidate.source_type, candidate.source_id): candidate
        for candidate in candidates
    }
    candidate_keys_by_id: dict[str, list[tuple[str, str]]] = {}
    for key in candidates_by_key:
        candidate_keys_by_id.setdefault(key[1], []).append(key)
    tool_schemas = {tool.name: tool.parameters for tool in request.tools}
    selected_keys: set[tuple[str, str]] = set()
    selected_items: list[SelectedContextItem] = []
    last_errors: list[str] = []

    for _turn in range(MAX_CONTEXT_SEARCH_TOOL_FEEDBACK_TURNS + 1):
        turn_request = budget_tool_call_request(
            repositories,
            replace(request, messages=tuple(messages)),
            task="context_search",
        )
        response = await provider.generate_tool_calls(turn_request)
        errors: list[str] = []
        tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
        for call in response.tool_calls:
            accepted, result, item = _validate_context_search_tool_call(
                call,
                tool_schemas=tool_schemas,
                candidates_by_key=candidates_by_key,
                candidate_keys_by_id=candidate_keys_by_id,
            )
            if accepted:
                if item is not None:
                    selected_key = (item.source_type, item.source_id)
                    if selected_key not in selected_keys:
                        selected_keys.add(selected_key)
                        selected_items.append(item)
                tool_results.append((call, _accepted_tool_result()))
                continue
            errors.append(result["error"])
            tool_results.append((call, result))

        if not errors:
            return _context_result_from_items(selected_items)

        last_errors = errors
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

    if selected_items:
        log_error_event(
            "provider.tool_call_validation_partial",
            task="context_search",
            accepted_count=len(selected_items),
            error="; ".join(last_errors),
        )
        return _context_result_from_items(selected_items)
    raise _ContextSearchToolValidationFailed(
        "Context search tool-call validation failed after feedback: "
        + "; ".join(last_errors)
    )


def _validate_context_search_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    candidates_by_key: dict[tuple[str, str], _ContextCandidate],
    candidate_keys_by_id: dict[str, list[tuple[str, str]]],
) -> tuple[bool, dict[str, str], SelectedContextItem | None]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_tool_call(parse_error or "Tool arguments must be a JSON object")
    error = _validate_context_search_tool_arguments(arguments, schema=schema)
    if error is not None:
        return _invalid_tool_call(error)
    source_id = cast(str, arguments["source_id"])
    source_type = arguments.get("source_type")
    if isinstance(source_type, str) and source_type:
        candidate = candidates_by_key.get((source_type, source_id))
    else:
        matching_keys = candidate_keys_by_id.get(source_id, [])
        if len(matching_keys) != 1:
            return _invalid_tool_call(
                "source_type is required when a source_id is shared by "
                f"multiple context candidates: {source_id}"
            )
        candidate = candidates_by_key[matching_keys[0]]
    if candidate is None:
        return _invalid_tool_call(
            "source_type/source_id is not a context candidate: "
            f"{source_type}:{source_id}"
        )
    note = str(arguments.get("relevance_note", "")).strip()
    return (
        True,
        _accepted_tool_result(),
        _selected_context_item(
            candidate,
            relevance_note=note or "Selected by context selection model.",
        ),
    )


def _validate_context_search_tool_arguments(
    arguments: dict[str, object],
    *,
    schema: dict[str, object],
) -> str | None:
    return validate_tool_arguments_shape(
        arguments,
        schema=schema,
        enum_error_formatter=lambda field_name, value, _allowed: (
            f"{field_name} must be one of offered values; got {value}"
        ),
    )


def _invalid_tool_call(error: str) -> tuple[bool, dict[str, str], None]:
    return (
        False,
        invalid_tool_result(
            error,
            retry_instruction=CONTEXT_SEARCH_TOOL_RETRY_INSTRUCTION,
        ),
        None,
    )


def _accepted_tool_result() -> dict[str, str]:
    return accepted_tool_result()


class _ContextSearchToolValidationFailed(Exception):
    pass


def _context_selection_schema(
    candidates: tuple[_ContextCandidate, ...],
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selections": {
                "type": "array",
                "maxItems": min(MAX_CONTEXT_SELECTIONS, len(candidates)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source_type": {
                            "type": "string",
                            "enum": [
                                "open_obligation",
                                "scenario_section",
                                "world_state",
                                "state_change",
                                "media_asset",
                                "character_text_thread",
                                "memory",
                                "observation",
                                "character_voice",
                                "message",
                            ],
                        },
                        "source_id": {
                            "type": "string",
                            "enum": [candidate.source_id for candidate in candidates],
                        },
                        "relevance_note": {"type": "string"},
                    },
                    "required": ["source_type", "source_id", "relevance_note"],
                },
            },
        },
        "required": ["selections"],
    }


def _context_selection_messages(
    *,
    scenario: ScenarioRecord | None,
    player_message: str,
    candidates: tuple[_ContextCandidate, ...],
) -> tuple[ChatMessage, ...]:
    scenario_type = scenario.type if scenario is not None else ""
    return (
        ChatMessage(
            role="system",
            body=_context_selection_instruction(scenario_type),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    _scenario_context_text(scenario),
                    f"Latest player message:\n{player_message}",
                    _candidate_list_text(candidates),
                )
            ),
        ),
    )


def _context_selection_tool_messages(
    *,
    scenario: ScenarioRecord | None,
    player_message: str,
    candidates: tuple[_ContextCandidate, ...],
) -> tuple[ToolCallMessage, ...]:
    messages = _context_selection_messages(
        scenario=scenario,
        player_message=player_message,
        candidates=candidates,
    )
    tool_messages: list[ToolCallMessage] = []
    for message in messages:
        body = message.body.replace(
            "Use the enforced schema.",
            "Use the provided select_context_source tool instead of prose.",
        ).replace(
            (
                "Return selected source IDs in priority order with one short "
                "relevance note each."
            ),
            (
                "Call select_context_source once per selected candidate in "
                "priority order with one short relevance_note. Select no items "
                "by making no tool calls."
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


def _context_selection_tool_definitions(
    candidates: tuple[_ContextCandidate, ...],
) -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name=CONTEXT_SEARCH_SELECTION_TOOL,
            description=(
                "Select one offered context source that is useful for the next "
                "roleplay response."
            ),
            parameters=_context_selection_tool_schema(candidates),
        ),
    )


def _context_selection_tool_schema(
    candidates: tuple[_ContextCandidate, ...],
) -> dict[str, object]:
    source_types = sorted({candidate.source_type for candidate in candidates})
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_id": {
                "type": "string",
                "enum": [candidate.source_id for candidate in candidates],
            },
            "source_type": {
                "type": "string",
                "enum": source_types,
            },
            "relevance_note": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["source_id"],
    }


def _context_selection_instruction(scenario_type: str) -> str:
    base = (
        "Select the minimum Bragi context needed for the roleplay model to "
        "decide its next response to the latest player message. Use the "
        "enforced schema. Select no items only when none of the candidates are "
        "useful. Return selected source IDs in priority order with one short "
        "relevance note each. "
        "Do not select filler, trivia, or context that merely shares words "
        "with the player message. Assume the roleplay model will not receive "
        "the full transcript; select prior messages only when they contain "
        "irreducible immediate context not captured by state, memories, "
        "scenario setup, or the small baseline recent transcript window. "
        "The latest safe rolling summary is supplied separately and is not a "
        "selectable context candidate."
    )
    if scenario_type == "dating_sim":
        return (
            base + " For dating sim roleplays, prioritize the active romance "
            "options' revealed traits, preferences, boundaries, current "
            "emotional states, relationship dynamics with the player character, "
            "and choice pressure. Avoid broad world lore unless the player "
            "message directly depends on it."
        )
    if scenario_type == "fantasy_roleplay":
        return (
            base + " For fantasy roleplays, prioritize active scene status, "
            "magic rules and costs, realms or places in play, factions or "
            "orders applying pressure, mythic threats, promises, quest stakes, "
            "inventory-like facts, and unresolved threads."
        )
    if scenario_type == "science_fiction_roleplay":
        return (
            base + " For science fiction roleplays, prioritize active scene "
            "status, technology constraints and failure modes, setting scale, "
            "species or artificial intelligences in play, institutions applying "
            "pressure, mission stakes, equipment-like facts, and unresolved "
            "threads."
        )
    if scenario_type == "first_contact_exploration":
        return (
            base + " For first-contact and exploration roleplays, prioritize "
            "active scene status, mission constraints, ship or base status, crew "
            "expertise and conflict, observed facts, hypotheses, confirmed "
            "knowledge, unknowns, translation progress, false assumptions, "
            "unknown intelligence behavior, diplomatic tension, discoveries, "
            "samples, sensor findings, contamination risk, environmental hazards, "
            "equipment damage, rescue windows, deadlines, and unresolved research "
            "questions. Preserve the distinction between what is observed, "
            "suspected, misunderstood, confirmed, and still unknown."
        )
    if scenario_type == "survival_expedition":
        return (
            base + " For survival expeditions, prioritize active scene status, "
            "route progress, resources and equipment, party health or morale, "
            "environmental conditions, hazards, camp status, landmarks, delays, "
            "detours, retreat status, and unresolved survival threats."
        )
    if scenario_type == "time_loop":
        return (
            base + " For time loop roleplays, prioritize active loop state, "
            "reset rules, loop counter or phase, baseline resettable facts, "
            "known schedules, windows of opportunity, persistent player/meta "
            "knowledge, persistence exceptions, NPC memory boundaries, prior-loop "
            "summaries, and deviations from the baseline. Do not treat reset NPCs "
            "as remembering prior loops unless a persistence exception says they do."
        )
    if scenario_type == "investigation_mystery":
        return (
            base + " For investigation mystery roleplays, prioritize discovered "
            "clues, known facts, suspects or witnesses currently in play, the "
            "public timeline, intentional red herrings, case status, and "
            "deduction progress. Select hidden truth only when needed to keep "
            "narration consistent, and do not reveal hidden truth as player "
            "knowledge."
        )
    if scenario_type == "heist_infiltration":
        return (
            base + " For heist and infiltration roleplays, prioritize target "
            "layout, objective progress, crew and contact status, current intel, "
            "access credentials or covers, security layers, guards, locks, alarms, "
            "cameras, patrols, suspicion, alarm state, heat, loadout, complications, "
            "extraction route status, pursuit pressure, and aftermath consequences. "
            "Preserve what the crew knows, suspects, and has not yet discovered."
        )
    if scenario_type == "political_intrigue":
        return (
            base + " For political intrigue roleplays, prioritize faction "
            "positions and pressure points, key NPC loyalties and grudges, "
            "reputation or standing, favors owed or held, bargains, promises, "
            "blackmail material, alliances, rivalries, event calendars, timed "
            "political pressure, public knowledge, private knowledge, and "
            "unresolved agenda conflicts. Preserve public/private boundaries "
            "and do not reveal secrets as player knowledge unless play has "
            "established the player learned them."
        )
    return (
        base + " For generic roleplays, prioritize active scene status, current "
        "objective, durable character facts, immediate threats, promises, "
        "inventory-like facts, and unresolved threads."
    )


def _candidate_list_text(candidates: tuple[_ContextCandidate, ...]) -> str:
    if not candidates:
        return "Context candidates: none"
    lines = ["Context candidates:"]
    lines.extend(
        _candidate_prompt_line(candidate)
        for candidate in candidates
    )
    return "\n".join(lines)


def _candidate_prompt_line(candidate: _ContextCandidate) -> str:
    text, _excerpted = _selector_visible_context_text(candidate)
    return f"- [{candidate.source_type}:{candidate.source_id}] {text}"


def _context_result_from_structured_data(
    data: dict[str, object],
    candidates: tuple[_ContextCandidate, ...],
) -> ContextSearchResult:
    raw_selections = data.get("selections", [])
    if not isinstance(raw_selections, list):
        raise ValueError("Structured context selection selections must be a list")
    candidates_by_key = {
        (candidate.source_type, candidate.source_id): candidate
        for candidate in candidates
    }
    selected_items: list[SelectedContextItem] = []
    selected_keys: set[tuple[str, str]] = set()
    for raw_selection in raw_selections:
        if not isinstance(raw_selection, dict):
            raise ValueError("Structured context selection item must be an object")
        source_type = str(raw_selection.get("source_type", ""))
        source_id = str(raw_selection.get("source_id", ""))
        key = (source_type, source_id)
        candidate = candidates_by_key.get(key)
        if candidate is None:
            raise ValueError(
                "Unknown context source_type/source_id: "
                f"{source_type}:{source_id}"
            )
        if key in selected_keys:
            continue
        selected_keys.add(key)
        note = str(raw_selection.get("relevance_note", "")).strip()
        selected_items.append(
            _selected_context_item(
                candidate,
                relevance_note=note or "Selected by context selection model.",
            )
        )
    return _context_result_from_items(selected_items)


def _selected_context_item(
    candidate: _ContextCandidate,
    *,
    relevance_note: str,
) -> SelectedContextItem:
    return SelectedContextItem(
        source_type=candidate.source_type,
        source_id=candidate.source_id,
        text=candidate.text,
        relevance_note=relevance_note,
        excerpted=False,
    )


def _selector_visible_context_text(candidate: _ContextCandidate) -> tuple[str, bool]:
    source_text = candidate.selection_text or candidate.text
    compact_source = " ".join(source_text.split())
    text = _compact_text(source_text, MAX_CONTEXT_CANDIDATE_TEXT_CHARS)
    return text, text != compact_source


def _context_result_from_items(
    items: list[SelectedContextItem],
) -> ContextSearchResult:
    selected_state: list[SelectedContextItem] = []
    selected_open_obligations: list[SelectedContextItem] = []
    selected_state_changes: list[SelectedContextItem] = []
    selected_media_assets: list[SelectedContextItem] = []
    selected_character_text_context: list[SelectedContextItem] = []
    selected_scenario_sections: list[SelectedContextItem] = []
    selected_memories: list[SelectedContextItem] = []
    selected_observations: list[SelectedContextItem] = []
    selected_character_voice: list[SelectedContextItem] = []
    selected_summaries: list[SelectedContextItem] = []
    selected_recent_messages: list[SelectedContextItem] = []
    for item in items:
        if item.source_type == "open_obligation":
            selected_open_obligations.append(item)
        elif item.source_type == "scenario_section":
            selected_scenario_sections.append(item)
        elif item.source_type == "world_state":
            selected_state.append(item)
        elif item.source_type == "state_change":
            selected_state_changes.append(item)
        elif item.source_type == "media_asset":
            selected_media_assets.append(item)
        elif item.source_type == "character_text_thread":
            selected_character_text_context.append(item)
        elif item.source_type == "memory":
            selected_memories.append(item)
        elif item.source_type == "observation":
            selected_observations.append(item)
        elif item.source_type == "character_voice":
            selected_character_voice.append(item)
        elif item.source_type == "summary":
            selected_summaries.append(item)
        elif item.source_type == "message":
            selected_recent_messages.append(item)
    return ContextSearchResult(
        selected_open_obligations=tuple(selected_open_obligations),
        selected_scenario_sections=tuple(selected_scenario_sections),
        selected_state=tuple(selected_state),
        selected_state_changes=tuple(selected_state_changes),
        selected_media_assets=tuple(selected_media_assets),
        selected_character_text_context=tuple(selected_character_text_context),
        selected_memories=tuple(selected_memories),
        selected_observations=tuple(selected_observations),
        selected_character_voice=tuple(selected_character_voice),
        selected_summaries=tuple(selected_summaries),
        selected_recent_messages=tuple(selected_recent_messages),
    )


def _context_result_is_empty(result: ContextSearchResult) -> bool:
    return not (
        result.selected_scenario_sections
        or result.selected_open_obligations
        or result.selected_state
        or result.selected_state_changes
        or result.selected_media_assets
        or result.selected_character_text_context
        or result.selected_memories
        or result.selected_observations
        or result.selected_character_voice
        or result.selected_summaries
        or result.selected_recent_messages
    )


def _selected_context_items(
    result: ContextSearchResult,
) -> tuple[SelectedContextItem, ...]:
    return (
        *result.selected_open_obligations,
        *result.selected_scenario_sections,
        *result.selected_state,
        *result.selected_state_changes,
        *result.selected_media_assets,
        *result.selected_character_text_context,
        *result.selected_memories,
        *result.selected_observations,
        *result.selected_character_voice,
        *result.selected_summaries,
        *result.selected_recent_messages,
    )


def _fallback_context_result(
    candidates: tuple[_ContextCandidate, ...],
) -> ContextSearchResult:
    items: list[SelectedContextItem] = []
    for candidate in _fallback_candidates(candidates):
        items.append(
            _selected_context_item(
                candidate,
                relevance_note=(
                    "Selected by deterministic fallback after empty context selection."
                ),
            )
        )
    return _context_result_from_items(items)


def _fallback_candidates(
    candidates: tuple[_ContextCandidate, ...],
) -> tuple[_ContextCandidate, ...]:
    selected: list[_ContextCandidate] = []
    for source_type in (
        "world_state",
        "open_obligation",
        "memory",
        "character_text_thread",
        "observation",
        "character_voice",
        "state_change",
        "media_asset",
        "message",
        "scenario_section",
    ):
        candidate = _fallback_candidate_for_type(candidates, source_type)
        if candidate is not None:
            selected.append(candidate)
    return tuple(selected)


def _fallback_candidate_for_type(
    candidates: tuple[_ContextCandidate, ...],
    source_type: str,
) -> _ContextCandidate | None:
    matching = [
        candidate for candidate in candidates if candidate.source_type == source_type
    ]
    if not matching:
        return None
    if source_type in {"memory", "message"}:
        return matching[-1]
    return matching[0]


def _candidate_count_fields(
    candidates: tuple[_ContextCandidate, ...],
) -> dict[str, int]:
    counts = {
        "state_candidate_count": 0,
        "open_obligation_candidate_count": 0,
        "state_change_candidate_count": 0,
        "media_asset_candidate_count": 0,
        "character_text_context_candidate_count": 0,
        "scenario_section_candidate_count": 0,
        "memory_candidate_count": 0,
        "observation_candidate_count": 0,
        "character_voice_candidate_count": 0,
        "summary_candidate_count": 0,
        "recent_message_candidate_count": 0,
    }
    for candidate in candidates:
        if candidate.source_type == "scenario_section":
            counts["scenario_section_candidate_count"] += 1
        elif candidate.source_type == "open_obligation":
            counts["open_obligation_candidate_count"] += 1
        elif candidate.source_type == "world_state":
            counts["state_candidate_count"] += 1
        elif candidate.source_type == "state_change":
            counts["state_change_candidate_count"] += 1
        elif candidate.source_type == "media_asset":
            counts["media_asset_candidate_count"] += 1
        elif candidate.source_type == "character_text_thread":
            counts["character_text_context_candidate_count"] += 1
        elif candidate.source_type == "memory":
            counts["memory_candidate_count"] += 1
        elif candidate.source_type == "observation":
            counts["observation_candidate_count"] += 1
        elif candidate.source_type == "character_voice":
            counts["character_voice_candidate_count"] += 1
        elif candidate.source_type == "summary":
            counts["summary_candidate_count"] += 1
        elif candidate.source_type == "message":
            counts["recent_message_candidate_count"] += 1
    return counts


def _candidate_diagnostics(
    *,
    before: tuple[_ContextCandidate, ...],
    after: tuple[_ContextCandidate, ...],
    observations: list[ContextObservationRecord] | tuple[ContextObservationRecord, ...],
    curated_observation_source_ids: frozenset[str],
    retrieval_diagnostics: Mapping[str, object] | None = None,
) -> _ContextCandidateDiagnostics:
    before_counts = _source_type_counts_json(before)
    after_counts = _source_type_counts_json(after)
    dropped = {
        source_type: count - after_counts.get(source_type, 0)
        for source_type, count in before_counts.items()
        if count > after_counts.get(source_type, 0)
    }
    observation_status_counts = Counter(record.status for record in observations)
    included_observation_ids = {
        candidate.source_id
        for candidate in after
        if candidate.source_type == "observation"
    }
    observation_status_by_id = {
        record.id: record.status for record in observations
    }
    included_observation_status_counts = Counter(
        observation_status_by_id[observation_id]
        for observation_id in included_observation_ids
        if observation_id in observation_status_by_id
    )
    excluded_observation_status_counts: Counter[str] = Counter()
    for status, count in observation_status_counts.items():
        excluded = count - included_observation_status_counts.get(status, 0)
        if excluded > 0:
            excluded_observation_status_counts[status] = excluded
    return _ContextCandidateDiagnostics(
        candidate_count_before_narrowing=len(before),
        candidate_count_after_narrowing=len(after),
        source_type_counts_before_narrowing=before_counts,
        source_type_counts_after_narrowing=after_counts,
        dropped_source_type_counts=dropped,
        observation_status_counts=_counter_json(observation_status_counts),
        included_observation_status_counts=_counter_json(
            included_observation_status_counts
        ),
        excluded_observation_status_counts=_counter_json(
            excluded_observation_status_counts
        ),
        curated_observation_candidate_count=sum(
            1
            for candidate in after
            if candidate.source_type == "observation"
            if candidate.source_id in curated_observation_source_ids
        ),
        suppressed_raw_observation_count=sum(
            1
            for record in observations
            if record.status == "accepted"
            if record.id in curated_observation_source_ids
        ),
        retrieval_diagnostics=dict(retrieval_diagnostics or {}),
    )


def _source_type_counts_json(
    candidates: tuple[_ContextCandidate, ...],
) -> dict[str, int]:
    return dict(Counter(candidate.source_type for candidate in candidates))


def _counter_json(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _candidate_digest(candidates: tuple[_ContextCandidate, ...]) -> str:
    return _json_digest(
        [
            {
                "source_type": candidate.source_type,
                "source_id": candidate.source_id,
                "text_hash": _text_digest(candidate.text),
                "selection_text_hash": _text_digest(candidate.selection_text or ""),
            }
            for candidate in candidates
        ]
    )


def _json_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _text_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _scenario_context_text(scenario: ScenarioRecord | None) -> str:
    if scenario is None:
        return "Scenario context: unavailable"
    lines = [
        "Scenario context:",
        f"- type: {scenario.type}",
        f"- title: {scenario.title}",
        f"- premise/setup: {scenario.premise}",
        f"- player role: {scenario.player_role}",
    ]
    return "\n".join(lines)


def _model_supports_structured_output(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> bool:
    return model_supports_any_capability(
        repositories,
        provider=provider,
        model_id=model_id,
        required=STRUCTURED_OUTPUT_CAPABILITIES,
    )


def _model_supports_tool_calling(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> bool:
    return model_supports_any_capability(
        repositories,
        provider=provider,
        model_id=model_id,
        required=TOOL_CALLING_CAPABILITIES,
    )


def _model_is_unavailable(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> bool:
    model = find_provider_model(repositories, provider=provider, model_id=model_id)
    return model is not None and not model.available


def _model_advertises_tool_calling(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> bool:
    return _model_advertises_any_capability(
        repositories=repositories,
        provider=provider,
        model_id=model_id,
        required=TOOL_CALLING_CAPABILITIES,
    )


def _model_advertises_structured_output(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> bool:
    return _model_advertises_any_capability(
        repositories=repositories,
        provider=provider,
        model_id=model_id,
        required=STRUCTURED_OUTPUT_CAPABILITIES,
    )


def _model_advertises_any_capability(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
    required: frozenset[str],
) -> bool:
    model = find_provider_model(repositories, provider=provider, model_id=model_id)
    if model is None:
        return False
    return bool(normalized_capabilities(model.capabilities) & required)


def _model_requirement_error(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> str | None:
    structured_check = check_model_capabilities(
        repositories,
        provider=provider,
        model_id=model_id,
        required=STRUCTURED_OUTPUT_CAPABILITIES,
    )
    tool_check = check_model_capabilities(
        repositories,
        provider=provider,
        model_id=model_id,
        required=TOOL_CALLING_CAPABILITIES,
    )
    if structured_check.reason == MODEL_MISSING_REASON:
        return f"Context-search model is not in the provider model catalog: {model_id}"
    if structured_check.reason == MODEL_UNAVAILABLE_REASON:
        return f"Context-search model is unavailable: {model_id}"
    if (
        structured_check.reason == MODEL_LACKS_CAPABILITY_REASON
        and tool_check.reason == MODEL_LACKS_CAPABILITY_REASON
    ):
        return (
            "Context-search model does not advertise structured output "
            "or tool calling"
        )
    return None


def _indexed_context_candidates(
    records: list[ContextSourceRecord],
    *,
    scoped_targets: ScopedTargets,
    reference_character_ids: frozenset[str],
    accepted_observation_ids: frozenset[str],
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> tuple[_ContextCandidate, ...]:
    candidates: list[_ContextCandidate] = []
    for record in records:
        if record.metadata.get("suppression_only") is True:
            continue
        source_type = _indexed_candidate_source_type(
            record,
            accepted_observation_ids=accepted_observation_ids,
        )
        if source_type is None:
            continue
        if not _metadata_provenance_visible_to_present_characters(
            record.metadata,
            present_character_ids=present_character_ids,
            message_visibility=message_visibility,
        ):
            continue
        if _audience_candidate_blocked(record, reference_character_ids):
            continue
        if _known_by_candidate_blocked(record, scoped_targets):
            continue
        source_text = _indexed_context_candidate_text(record, source_type=source_type)
        if not source_text.strip():
            continue
        text = _scoped_candidate_text(
            source_type=source_type,
            source_id=record.source_id,
            text=source_text,
            scoped_targets=scoped_targets,
        )
        if text is None:
            continue
        metadata_text = _indexed_metadata_selection_text(record, text=text)
        candidates.append(
            _ContextCandidate(
                source_type=source_type,
                source_id=record.source_id,
                text=text,
                selection_text=metadata_text,
            )
        )
    return tuple(candidates)


def _missing_raw_candidates(
    state_candidates: tuple[_ContextCandidate, ...],
    memory_candidates: tuple[_ContextCandidate, ...],
    *,
    indexed_candidates: tuple[_ContextCandidate, ...],
) -> tuple[_ContextCandidate, ...]:
    indexed_keys = {
        (candidate.source_type, candidate.source_id) for candidate in indexed_candidates
    }
    return tuple(
        candidate
        for candidate in (*state_candidates, *memory_candidates)
        if (candidate.source_type, candidate.source_id) not in indexed_keys
    )


def _exact_raw_candidates(
    candidates: tuple[_ContextCandidate, ...],
    *,
    indexed_candidates: tuple[_ContextCandidate, ...],
    latest_player_message: str,
    limit: int = 24,
) -> tuple[_ContextCandidate, ...]:
    query_terms = set(_bounded_context_query_terms(latest_player_message))
    if not query_terms or limit <= 0:
        return ()
    structured_identifiers = _bounded_structured_identifiers(latest_player_message)
    indexed_keys = {
        (candidate.source_type, candidate.source_id)
        for candidate in indexed_candidates
    }
    ranked: list[tuple[float, _ContextCandidate]] = []
    ordinary_query_terms = {
        "about",
        "could",
        "did",
        "does",
        "from",
        "have",
        "learn",
        "remember",
        "tell",
        "that",
        "what",
        "when",
        "where",
        "which",
        "who",
        "would",
    }
    for candidate in candidates:
        if (candidate.source_type, candidate.source_id) in indexed_keys:
            continue
        candidate_text = (candidate.selection_text or candidate.text).casefold()
        candidate_terms = set(
            _bounded_context_query_terms(candidate_text)
        )
        overlap_terms = query_terms & candidate_terms
        overlap_count = len(overlap_terms)
        distinctive_identifier_match = any(
            len(term) >= 6 and term not in ordinary_query_terms
            for term in overlap_terms
        ) or any(
            identifier in candidate_text
            for identifier in structured_identifiers
        )
        if (
            overlap_count < min(2, len(query_terms))
            and not distinctive_identifier_match
        ):
            continue
        coverage = overlap_count / len(candidate_terms)
        short_identifier_match = (
            len(query_terms) <= 2 and overlap_count == len(query_terms)
        )
        if (
            coverage < 0.5
            and not short_identifier_match
            and not distinctive_identifier_match
        ):
            continue
        ranked.append((coverage + overlap_count, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return tuple(candidate for _score, candidate in ranked[:limit])


def _bounded_structured_identifiers(text: str) -> tuple[str, ...]:
    ordered = tuple(
        dict.fromkeys(
            identifier.casefold()
            for identifier in re.findall(
                r"(?<!\w)[^\W_]+(?:[-_.][^\W_]+)+(?!\w)",
                _bounded_context_query_text(text),
                flags=re.UNICODE,
            )
        )
    )
    if len(ordered) <= MAX_EXACT_RAW_STRUCTURED_IDENTIFIERS:
        return ordered
    edge_count = MAX_EXACT_RAW_STRUCTURED_IDENTIFIERS // 2
    return tuple(
        dict.fromkeys((*ordered[:edge_count], *ordered[-edge_count:]))
    )


def _known_by_candidate_blocked(
    record: ContextSourceRecord,
    scoped_targets: ScopedTargets,
) -> bool:
    audience_character_ids = record.metadata.get("audience_character_ids")
    if isinstance(audience_character_ids, list) and audience_character_ids:
        return False
    known_by = record.metadata.get("known_by")
    if not isinstance(known_by, list) or not known_by:
        return False
    allowed_owners = {
        scoped_owner_name(owner).casefold()
        for owners in scoped_targets.allowed.values()
        for owner in owners
    }
    normalized_known_by = {str(item).casefold() for item in known_by}
    return not bool(normalized_known_by & allowed_owners)


def _audience_candidate_blocked(
    record: ContextSourceRecord,
    reference_character_ids: frozenset[str],
) -> bool:
    audience_character_ids = record.metadata.get("audience_character_ids")
    requires_audience = record.metadata.get("requires_audience") is True
    if not isinstance(audience_character_ids, list) or not audience_character_ids:
        return requires_audience
    normalized_audience = {
        str(character_id)
        for character_id in audience_character_ids
        if str(character_id).strip()
    }
    return not bool(normalized_audience & set(reference_character_ids))


def _indexed_candidate_source_type(
    record: ContextSourceRecord,
    *,
    accepted_observation_ids: frozenset[str],
) -> str | None:
    if record.source_type == "observation":
        if (
            _curated_observation_source_id(
                record,
                accepted_observation_ids=accepted_observation_ids,
            )
            is not None
        ):
            return "observation"
        return None
    if record.source_type in {
        "character_text_thread",
        "open_obligation",
        "scenario_section",
        "world_state",
        "memory",
        "character_voice",
    }:
        return record.source_type
    return None


def _indexed_context_candidate_text(
    record: ContextSourceRecord,
    *,
    source_type: str,
) -> str:
    body = record.body.strip()
    if source_type != "observation":
        return body
    title = record.title.strip()
    if title and body:
        if title.casefold() in body.casefold():
            return body
        return f"{title}: {body}"
    return title or body


def _indexed_metadata_selection_text(
    record: ContextSourceRecord,
    *,
    text: str,
) -> str:
    fact_type = str(
        record.metadata.get("fact_type")
        or record.metadata.get("observation_type")
        or ""
    ).strip()
    importance = record.metadata.get("importance")
    parts = [text]
    if fact_type:
        parts.append(f"fact_type: {fact_type}")
    if isinstance(importance, int | float):
        parts.append(f"importance: {float(importance):.2g}")
    known_by = record.metadata.get("known_by")
    if isinstance(known_by, list) and known_by:
        parts.append("known_by: " + ", ".join(str(item) for item in known_by))
    return " | ".join(parts)


def _accepted_observation_ids(
    records: list[ContextObservationRecord] | tuple[ContextObservationRecord, ...],
) -> frozenset[str]:
    return frozenset(record.id for record in records if record.status == "accepted")


def _observations_with_indexed_sources(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    observations: tuple[ContextObservationRecord, ...],
    context_sources: tuple[ContextSourceRecord, ...],
) -> tuple[ContextObservationRecord, ...]:
    existing_ids = {observation.id for observation in observations}
    missing_ids = {
        source.source_id
        for source in context_sources
        if source.source_type == "observation"
        and source.source_id not in existing_ids
    }
    if not missing_ids:
        return observations
    return (
        *observations,
        *repositories.list_context_observations_by_ids(save_id, missing_ids),
    )


def _context_source_message_ids(
    source: ContextSourceRecord,
) -> frozenset[str]:
    source_ids: set[str] = set()
    for metadata_field in ("source_message_id", "last_seen_message_id"):
        value = source.metadata.get(metadata_field)
        if isinstance(value, str) and value:
            source_ids.add(value)
    raw_ids = source.metadata.get("source_message_ids")
    if isinstance(raw_ids, list):
        source_ids.update(
            value for value in raw_ids if isinstance(value, str) and value
        )
    raw_groups = source.metadata.get("source_provenance_groups")
    if isinstance(raw_groups, list):
        source_ids.update(
            value
            for group in raw_groups
            if isinstance(group, list)
            for value in group
            if isinstance(value, str) and value
        )
    return frozenset(source_ids)


def _curated_observation_source_ids(
    records: list[ContextSourceRecord],
    *,
    memories: list[MemoryRecord],
    accepted_observation_ids: frozenset[str],
) -> frozenset[str]:
    context_observation_ids = frozenset(
        observation_id
        for record in records
        if (
            observation_id := _curated_observation_source_id(
                record,
                accepted_observation_ids=accepted_observation_ids,
            )
        )
        is not None
    )
    memory_observation_ids = frozenset(
        observation_id
        for memory in memories
        for observation_id in memory.source_observation_ids
        if observation_id in accepted_observation_ids
    )
    return context_observation_ids | memory_observation_ids


def _curated_observation_source_id(
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
    if not (record.title.strip() or record.body.strip()):
        return None
    return observation_id


def _narrow_context_candidates(
    candidates: tuple[_ContextCandidate, ...],
    *,
    latest_player_message: str,
) -> tuple[_ContextCandidate, ...]:
    if len(candidates) <= MAX_CONTEXT_CANDIDATE_POOL:
        return candidates
    query_terms = _meaningful_terms(latest_player_message)
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            -_candidate_rank(item[1], query_terms=query_terms),
            item[0],
        ),
    )
    selected_indexes: set[int] = set()
    selected_type_counts: Counter[str] = Counter()
    for source_type in dict.fromkeys(
        candidate.source_type for candidate in candidates
    ):
        reserved = [
            item for item in ranked if item[1].source_type == source_type
        ][:4]
        for index, candidate in reserved:
            selected_indexes.add(index)
            selected_type_counts[candidate.source_type] += 1
    for index, candidate in ranked:
        if len(selected_indexes) >= MAX_CONTEXT_CANDIDATE_POOL:
            break
        if index in selected_indexes:
            continue
        if selected_type_counts[candidate.source_type] >= 24:
            continue
        selected_indexes.add(index)
        selected_type_counts[candidate.source_type] += 1
    selected = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if index in selected_indexes
    ]
    return tuple(candidate for _index, candidate in selected)


def _candidate_rank(
    candidate: _ContextCandidate,
    *,
    query_terms: set[str],
) -> float:
    source_priority = {
        "open_obligation": 9.0,
        "world_state": 7.0,
        "memory": 6.5,
        "character_voice": 6.0,
        "observation": 5.75,
        "state_change": 5.5,
        "media_asset": 4.5,
        "message": 4.0,
        "scenario_section": 3.0,
        "summary": 1.5,
    }.get(candidate.source_type, 0.0)
    text = (candidate.selection_text or candidate.text).casefold()
    lexical_score = sum(1.0 for term in query_terms if term in text)
    high_value_score = 0.0
    if any(
        marker in text
        for marker in (
            "fact_type: promise",
            "fact_type: open_obligation",
            "fact_type: character_voice",
            "fact_type: relationship",
            "fact_type: inventory",
        )
    ):
        high_value_score = 4.0
    return source_priority + lexical_score + high_value_score


def _meaningful_terms(text: str) -> set[str]:
    return set(_ordered_meaningful_query_terms(text))


def _ordered_meaningful_query_terms(text: str) -> tuple[str, ...]:
    terms = (*unicode_word_terms(text), *cjk_lexical_anchors(text))
    return tuple(
        dict.fromkeys(
            term
            for term in terms
            if term
            and term
            not in {
                "and",
                "for",
                "i",
                "the",
                "that",
                "this",
                "with",
                "you",
                "your",
            }
        )
    )


def _bounded_context_query_text(text: str) -> str:
    if len(text) <= MAX_CONTEXT_QUERY_CHARS:
        return text
    half = MAX_CONTEXT_QUERY_CHARS // 2
    return f"{text[:half]} {text[-half:]}"


def _bounded_context_query_terms(text: str) -> tuple[str, ...]:
    ordered = _ordered_meaningful_query_terms(_bounded_context_query_text(text))
    if len(ordered) <= MAX_CONTEXT_QUERY_TERMS:
        return _balanced_script_terms(
            set(ordered),
            limit=MAX_CONTEXT_QUERY_TERMS,
        )
    edge_reserve = MAX_CONTEXT_QUERY_TERMS // 4
    identifier_reserve = MAX_CONTEXT_QUERY_TERMS // 8
    identifier_terms = tuple(
        term
        for term in ordered
        if len(term) <= 2
        and term.isascii()
        and term.isalnum()
    )[:identifier_reserve]
    selected = list(
        dict.fromkeys(
            (
                *identifier_terms,
                *ordered[:edge_reserve],
                *ordered[-edge_reserve:],
            )
        )
    )
    remaining = set(ordered) - set(selected)
    selected.extend(
        _balanced_script_terms(
            remaining,
            limit=MAX_CONTEXT_QUERY_TERMS - len(selected),
        )
    )
    return tuple(selected[:MAX_CONTEXT_QUERY_TERMS])


def _balanced_script_terms(
    terms: set[str] | frozenset[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    if limit <= 0:
        return ()
    ordered = sorted(terms, key=lambda term: (-len(term), term))
    non_ascii = [
        term
        for term in ordered
        if any(ord(character) > 127 for character in term)
    ]
    non_ascii_set = set(non_ascii)
    ascii_terms = [term for term in ordered if term not in non_ascii_set]
    if not non_ascii or not ascii_terms:
        return tuple(ordered[:limit])
    reserve = limit // 2
    selected = [*non_ascii[:reserve], *ascii_terms[:reserve]]
    selected_set = set(selected)
    selected.extend(
        term
        for term in ordered
        if term not in selected_set
    )
    return tuple(selected[:limit])


def _state_candidates(
    records: list[WorldStateRecord],
    *,
    scoped_targets: ScopedTargets,
    exclude_open_thread_aggregates: bool = False,
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> tuple[_ContextCandidate, ...]:
    candidates: list[_ContextCandidate] = []
    for record in records:
        if exclude_open_thread_aggregates and is_open_threads_aggregate_key(record.key):
            continue
        if record.source_message_id and not (
            _source_messages_visible_to_present_characters(
                (record.source_message_id,),
                present_character_ids=present_character_ids,
                message_visibility=message_visibility,
            )
        ):
            continue
        text = _scoped_candidate_text(
            source_type="world_state",
            source_id=record.id,
            text=f"{record.key}: {_format_state_value(record.value)}",
            scoped_targets=scoped_targets,
        )
        if text is not None:
            candidates.append(
                _ContextCandidate(
                    source_type="world_state",
                    source_id=record.id,
                    text=text,
                )
            )
    return tuple(candidates)


def _state_change_candidates(
    records: list[StateChangeRecord],
    world_state: list[WorldStateRecord],
    active_message_ids: set[str],
    *,
    scoped_targets: ScopedTargets,
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> tuple[_ContextCandidate, ...]:
    current_values = {record.key: record.value for record in world_state}
    blocked_state_keys = _blocked_world_state_keys(world_state, scoped_targets)
    candidates = [
        record
        for record in records
        if _has_active_source_message(record, active_message_ids)
        and _state_change_source_visible_to_present_characters(
            record,
            present_character_ids=present_character_ids,
            message_visibility=message_visibility,
        )
        and record.state_key not in blocked_state_keys
        and not _is_manual_archive_change(record)
        and not _duplicates_current_state(record, current_values)
    ][-STATE_CHANGE_CANDIDATE_LIMIT:]
    return tuple(
        _ContextCandidate(
            source_type="state_change",
            source_id=record.id,
            text=_state_change_text(record),
        )
        for record in candidates
    )


def _blocked_world_state_keys(
    world_state: list[WorldStateRecord],
    scoped_targets: ScopedTargets,
) -> frozenset[str]:
    blocked_world_state_ids = {
        target_id
        for target_type, target_id in scoped_targets.blocked
        if target_type == "world_state"
    }
    if not blocked_world_state_ids:
        return frozenset()
    return frozenset(
        record.key for record in world_state if record.id in blocked_world_state_ids
    )


def _state_change_source_visible_to_present_characters(
    record: StateChangeRecord,
    *,
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> bool:
    if record.source_message_id is None:
        return True
    return message_visible_to_present_characters(
        message_id=record.source_message_id,
        present_character_ids=present_character_ids,
        message_visibility=message_visibility,
    )


def _media_asset_candidates(
    records: list[MediaAssetRecord],
    recent_message_ids: set[str],
) -> tuple[_ContextCandidate, ...]:
    candidates = [
        record
        for record in records
        if record.type == "image"
        and record.status == "succeeded"
        and record.source_message_id in recent_message_ids
    ][-MEDIA_ASSET_CANDIDATE_LIMIT:]
    return tuple(
        _ContextCandidate(
            source_type="media_asset",
            source_id=record.id,
            text=_media_asset_text(record),
        )
        for record in candidates
    )


def _scoped_candidate_text(
    *,
    source_type: str,
    source_id: str,
    text: str,
    scoped_targets: ScopedTargets,
) -> str | None:
    key = (source_type, source_id)
    owners = scoped_targets.allowed.get(key)
    if owners:
        return f"Character-scoped knowledge ({', '.join(owners)}): {text}"
    if key in scoped_targets.blocked:
        return None
    return text


def _scenario_section_candidates(
    scenario: ScenarioRecord | None,
) -> tuple[_ContextCandidate, ...]:
    return tuple(
        _ContextCandidate(
            source_type="scenario_section",
            source_id=source_id,
            text=text,
            selection_text=f"{section_id}: {text}",
        )
        for source_id, section_id, text in scenario_section_candidates(scenario)
    )


def _memory_candidates(
    records: list[MemoryRecord],
    *,
    scoped_targets: ScopedTargets,
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
    observations_by_id: dict[str, ContextObservationRecord],
) -> tuple[_ContextCandidate, ...]:
    candidates: list[_ContextCandidate] = []
    for record in records:
        source_message_ids = tuple(
            dict.fromkeys(
                (
                    *record.source_message_ids,
                    *((record.source_message_id,) if record.source_message_id else ()),
                )
            )
        )
        if not _memory_provenance_visible_to_present_characters(
            record,
            source_message_ids=source_message_ids,
            observations_by_id=observations_by_id,
            present_character_ids=present_character_ids,
            message_visibility=message_visibility,
        ):
            continue
        text = _scoped_candidate_text(
            source_type="memory",
            source_id=record.id,
            text=record.body,
            scoped_targets=scoped_targets,
        )
        if text is None:
            continue
        candidates.append(
            _ContextCandidate(
                source_type="memory",
                source_id=record.id,
                text=text,
                selection_text=_memory_selection_text(record, text=text),
            )
        )
    return tuple(candidates)


def _memory_provenance_visible_to_present_characters(
    memory: MemoryRecord,
    *,
    source_message_ids: tuple[str, ...],
    observations_by_id: dict[str, ContextObservationRecord],
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> bool:
    groups: list[tuple[str, ...]] = []
    memory_fingerprint = (
        memory.claim_fingerprint or canonical_claim_fingerprint(memory.body)
    )
    provenance_mode = "any"
    grouped_source_ids: set[str] = set()
    for observation_id in memory.source_observation_ids:
        observation = observations_by_id.get(observation_id)
        if observation is None:
            provenance_mode = "all"
            continue
        group = tuple(dict.fromkeys(observation.source_message_ids))
        if group:
            groups.append(group)
            grouped_source_ids.update(group)
        curation = observation.metadata.get("curation")
        curated_body = (
            curation.get("memory_body")
            if isinstance(curation, dict)
            else None
        )
        supporting_body = (
            curated_body.strip()
            if isinstance(curated_body, str) and curated_body.strip()
            else observation.claim
        )
        if canonical_claim_fingerprint(supporting_body) != memory_fingerprint:
            provenance_mode = "all"
    ungrouped = tuple(
        source_id
        for source_id in source_message_ids
        if source_id not in grouped_source_ids
    )
    if ungrouped:
        groups.append(ungrouped)
    if not groups:
        groups.append(source_message_ids)
    visible_groups = (
        _source_messages_visible_to_present_characters(
            group,
            present_character_ids=present_character_ids,
            message_visibility=message_visibility,
        )
        for group in groups
    )
    return all(visible_groups) if provenance_mode == "all" else any(visible_groups)


def _observation_candidates(
    records: list[ContextObservationRecord],
    *,
    excluded_observation_ids: frozenset[str],
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> tuple[_ContextCandidate, ...]:
    candidates: list[_ContextCandidate] = []
    for record in records:
        if record.status != "accepted":
            continue
        if record.id in excluded_observation_ids:
            continue
        if not _source_messages_visible_to_present_characters(
            tuple(record.source_message_ids),
            present_character_ids=present_character_ids,
            message_visibility=message_visibility,
        ):
            continue
        text = (
            f"{record.claim} Evidence: {record.evidence_quote or 'not quoted'}; "
            f"sources: {', '.join(record.source_message_ids) or 'none'}; "
            f"scope: {record.scope}; status: {record.status}."
        )
        candidates.append(
            _ContextCandidate(
                source_type="observation",
                source_id=record.id,
                text=text,
                selection_text=(
                    f"{record.observation_type}: {record.claim} "
                    f"(evidence: {record.evidence_quote})"
                ),
            )
        )
    return tuple(candidates)


def _metadata_source_message_ids(
    metadata: Mapping[str, object],
) -> tuple[str, ...]:
    raw_ids = metadata.get("source_message_ids")
    values = (
        [
            str(item)
            for item in raw_ids
            if isinstance(item, str) and item.strip()
        ]
        if isinstance(raw_ids, list)
        else []
    )
    for field_name in ("source_message_id", "last_seen_message_id"):
        value = metadata.get(field_name)
        if isinstance(value, str) and value.strip():
            values.append(value)
    return tuple(dict.fromkeys(values))


def _metadata_provenance_visible_to_present_characters(
    metadata: Mapping[str, object],
    *,
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> bool:
    raw_groups = metadata.get("source_provenance_groups")
    mode = metadata.get("source_provenance_mode")
    if mode is not None and mode not in {"all", "any"}:
        return False
    if raw_groups is not None and (
        not isinstance(raw_groups, list)
        or len(raw_groups) > MAX_CONTEXT_SOURCE_PROVENANCE_GROUPS
        or not all(
            isinstance(group, list)
            and bool(group)
            and len(group) <= MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS
            and all(isinstance(source_id, str) and source_id for source_id in group)
            for group in raw_groups
        )
    ):
        return False
    groups = (
        tuple(
            tuple(
                str(source_id)
                for source_id in group
                if isinstance(source_id, str) and source_id.strip()
            )
            for group in raw_groups
            if isinstance(group, list)
        )
        if isinstance(raw_groups, list)
        else ()
    )
    nonempty_groups = tuple(group for group in groups if group)
    if nonempty_groups:
        grouped_source_ids = {
            source_id for group in nonempty_groups for source_id in group
        }
        if not set(_metadata_source_message_ids(metadata)).issubset(
            grouped_source_ids
        ):
            return False
        group_visibility = (
            _source_messages_visible_to_present_characters(
                group,
                present_character_ids=present_character_ids,
                message_visibility=message_visibility,
            )
            for group in nonempty_groups
        )
        if mode == "all":
            return all(group_visibility)
        return any(group_visibility)
    return _source_messages_visible_to_present_characters(
        _metadata_source_message_ids(metadata),
        present_character_ids=present_character_ids,
        message_visibility=message_visibility,
    )


def _source_messages_visible_to_present_characters(
    source_message_ids: tuple[str, ...],
    *,
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> bool:
    return all(
        message_visible_to_present_characters(
            message_id=message_id,
            present_character_ids=present_character_ids,
            message_visibility=message_visibility,
        )
        for message_id in source_message_ids
    )


def _message_candidates(
    records: list[MessageRecord],
    player_message_id: str,
    *,
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> tuple[_ContextCandidate, ...]:
    visible_records = _latest_visible_messages(
        records,
        limit=RECENT_MESSAGE_CANDIDATE_LIMIT,
        present_character_ids=present_character_ids,
        message_visibility=message_visibility,
        excluded_message_id=player_message_id,
    )
    return tuple(
        _ContextCandidate(
            source_type="message",
            source_id=record.id,
            text=f"{record.speaker_name or record.role}: {record.body}",
        )
        for record in visible_records
    )


def _latest_visible_messages(
    records: list[MessageRecord],
    *,
    limit: int,
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
    excluded_message_id: str | None = None,
) -> list[MessageRecord]:
    if limit <= 0:
        return []
    hidden_message_ids = {
        record.message_id
        for record in message_visibility
        if record.character_id in present_character_ids
        and record.visibility == "not_visible"
    }
    selected: list[MessageRecord] = []
    for record in reversed(records):
        if record.id == excluded_message_id or record.id in hidden_message_ids:
            continue
        selected.append(record)
        if len(selected) >= limit:
            break
    selected.reverse()
    return selected


def _memory_selection_text(record: MemoryRecord, *, text: str | None = None) -> str:
    tags = ", ".join(record.tags)
    body = text or record.body
    if not tags:
        return body
    return f"{body} (tags: {tags}; importance: {record.importance:g})"


def _format_state_value(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}: {item}" for key, item in value.items())
    return str(value)


def _duplicates_current_state(
    record: StateChangeRecord,
    current_values: Mapping[str, object],
) -> bool:
    if record.operation != "upsert" or record.after_json is None:
        return False
    if record.state_key not in current_values:
        return False
    try:
        after_value: object = json.loads(record.after_json)
    except json.JSONDecodeError:
        return False
    return after_value == current_values[record.state_key]


def _is_manual_archive_change(record: StateChangeRecord) -> bool:
    return (
        record.operation == "manual_world_data_edit"
        and record.source_message_id is None
        and record.after_json is None
    )


def _has_active_source_message(
    record: StateChangeRecord,
    active_message_ids: set[str],
) -> bool:
    return (
        record.source_message_id is None
        or record.source_message_id in active_message_ids
    )


def _state_change_text(record: StateChangeRecord) -> str:
    parts = [
        f"{record.operation} {record.state_key}",
        f"source_message_id: {record.source_message_id or 'unknown'}",
    ]
    before = (
        ""
        if _removes_state(record)
        else _state_change_json_text(record.before_json)
    )
    after = _state_change_json_text(record.after_json)
    if before:
        parts.append(f"before: {before}")
    if after:
        parts.append(f"after: {after}")
    return "; ".join(parts)


def _removes_state(record: StateChangeRecord) -> bool:
    return record.after_json is None and record.operation in {"delete", "remove"}


def _state_change_json_text(value: str | None) -> str:
    if not value:
        return ""
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return _format_state_value(loaded)


def _media_asset_text(record: MediaAssetRecord) -> str:
    parts = [
        "generated image metadata",
        f"source_message_id: {record.source_message_id or 'unknown'}",
    ]
    model_text = "/".join(part for part in (record.provider, record.model) if part)
    if model_text:
        parts.append(f"model: {model_text}")
    if record.prompt:
        prompt_excerpt = _compact_candidate_text(_redact_media_prompt(record.prompt))
        parts.append(
            f"prompt excerpt: {prompt_excerpt}"
        )
    return "; ".join(parts)


def _redact_media_prompt(text: str) -> str:
    redacted = DATA_PAYLOAD_PATTERN.sub("[redacted data payload]", text)
    redacted = PRIVATE_MEDIA_PATH_PATTERN.sub("[redacted media path]", redacted)
    redacted = WINDOWS_PATH_PATTERN.sub("[redacted file path]", redacted)
    redacted = ABSOLUTE_PATH_PATTERN.sub("[redacted file path]", redacted)
    redacted = RELATIVE_PATH_PATTERN.sub("[redacted file path]", redacted)
    return BASE64_LIKE_TOKEN_PATTERN.sub("[redacted data payload]", redacted)


def _compact_candidate_text(text: str) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= MEDIA_PROMPT_EXCERPT_MAX_CHARS:
        return compacted
    marker = "..."
    return compacted[: MEDIA_PROMPT_EXCERPT_MAX_CHARS - len(marker)].rstrip() + marker


def _message_body(messages: list[MessageRecord], message_id: str) -> str:
    for message in messages:
        if message.id == message_id:
            return message.body
    return ""


def _context_search_visible_messages(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    scene_snapshot: SceneSnapshotRecord | None,
    required_messages: tuple[MessageRecord, ...] = (),
) -> list[MessageRecord]:
    visible_messages = repositories.list_recent_messages_visible_to_characters(
        save_id,
        character_ids=(
            set(scene_snapshot.present_character_ids)
            if scene_snapshot is not None
            else set()
        ),
        limit=CONTEXT_SEARCH_MESSAGE_LOAD_LIMIT,
    )
    visible_ids = {message.id for message in visible_messages}
    visible_messages.extend(
        message for message in required_messages if message.id not in visible_ids
    )
    return visible_messages


def _result_json(
    result: ContextSearchResult,
    *,
    diagnostics: dict[str, object] | None = None,
    selection: _ContextSelectionOutcome | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "selected_open_obligations": [
            _item_json(item) for item in result.selected_open_obligations
        ],
        "selected_scenario_sections": [
            _item_json(item) for item in result.selected_scenario_sections
        ],
        "selected_state": [_item_json(item) for item in result.selected_state],
        "selected_state_changes": [
            _item_json(item) for item in result.selected_state_changes
        ],
        "selected_media_assets": [
            _item_json(item) for item in result.selected_media_assets
        ],
        "selected_character_text_context": [
            _item_json(item) for item in result.selected_character_text_context
        ],
        "selected_memories": [_item_json(item) for item in result.selected_memories],
        "selected_observations": [
            _item_json(item) for item in result.selected_observations
        ],
        "selected_character_voice": [
            _item_json(item) for item in result.selected_character_voice
        ],
        "selected_summaries": [_item_json(item) for item in result.selected_summaries],
        "selected_recent_messages": [
            _item_json(item) for item in result.selected_recent_messages
        ],
        "retrieval_degraded": result.retrieval_degraded,
    }
    if result.retrieval_recovery is not None:
        payload["retrieval_recovery"] = result.retrieval_recovery
    if selection is not None:
        payload.update(_selection_metadata(selection))
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    return payload


def _selection_metadata(selection: _ContextSelectionOutcome) -> dict[str, object]:
    metadata: dict[str, object] = {"fallback_used": selection.fallback_used}
    for key, value in (
        ("provider", selection.primary_provider),
        ("model", selection.primary_model_id),
        ("final_provider", selection.final_provider),
        ("final_model", selection.final_model_id),
        ("fallback_provider", selection.fallback_provider),
        ("fallback_model", selection.fallback_model_id),
        ("fallback_skipped_reason", selection.fallback_skipped_reason),
        ("error_category", selection.error_category),
    ):
        if value:
            metadata[key] = value
    if selection.http_status is not None:
        metadata["http_status"] = selection.http_status
    return metadata


def _item_json(item: SelectedContextItem) -> dict[str, object]:
    return {
        "source_type": item.source_type,
        "source_id": item.source_id,
        "text": _compact_text(item.text, MAX_CONTEXT_RESULT_TEXT_CHARS),
        "relevance_note": _compact_text(item.relevance_note, 240),
        "excerpted": item.excerpted,
    }


def _compact_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "..."


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
