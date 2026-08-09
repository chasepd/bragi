"""Import-safe runtime controller models shared by the web shell."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Mapping,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.application.chat_history import ChatHistoryModel, build_chat_history_model
from bragi.application.chronicle import (
    ChronicleMessageModel,
    ChronicleModel,
    MessageRevisionMetadata,
    build_chronicle_model,
    parse_message_markdown,
)
from bragi.application.media import MediaModel, build_media_model
from bragi.application.scenario_wizard import (
    ScenarioWizardModel,
    build_scenario_wizard_model,
)
from bragi.application.scene_presence import character_image_eligible_message_ids
from bragi.application.settings import (
    DEFAULT_AUTOMATIC_IMAGE_GENERATION_ENABLED,
    DEFAULT_AUTOMATIC_SUMMARIZATION_ENABLED,
    DEFAULT_IMAGE_GENERATION_FREQUENCY,
    DEFAULT_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD,
    MAX_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD,
    MIN_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD,
)
from bragi.content_rating_instructions import (
    CONTENT_RATING_PROHIBITED,
    content_rating_exceeds,
    maximum_content_rating,
)
from bragi.interaction_mode import InteractionMode, normalize_interaction_mode
from bragi.persistence.models import (
    CharacterRecord,
    MemoryRecord,
    MessageRecord,
    ModelPreferenceRecord,
    SaveRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ProviderCapability,
    ProviderClient,
    ProviderRetryProgressCallback,
    StructuredOutputProvider,
    ToolCallProvider,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.retry import exhausted_retry_attempt_count
from bragi.redaction import redact_text
from bragi.safety import (
    CONTENT_FILTER_TRANSITION,
    CONTENT_FILTER_TRANSITION_KIND,
    FADE_TO_BLACK_TRANSITION,
    FADE_TO_BLACK_TRANSITION_KIND,
)
from bragi.services.action_choice_flags import (
    content_with_action_choices_enabled,
    scenario_action_choices_enabled,
)
from bragi.services.action_choice_service import (
    ActionChoiceService,
    PreparedActionChoiceGeneration,
)
from bragi.services.character_bundle_service import (
    CharacterBundlePreview,
    CharacterBundleService,
)
from bragi.services.character_locks import (
    CHARACTER_AGENCY_FIELDS,
    character_field_is_locked,
    normalize_character_locked_fields,
)
from bragi.services.character_profile_completion import (
    CHARACTER_FIELD_ENHANCEMENT_FIELDS,
    CHARACTER_STARTER_AGENCY_LOCK_FIELDS,
    CHARACTER_STARTER_IDENTITY_LOCK_FIELDS,
    CharacterFieldEnhancementRequest,
    CharacterProfileCompletionRequest,
    CharacterStarterGenerationRequest,
    ScenarioCharacterStarter,
    StructuredProviderCharacterProfileCompleter,
    ToolCallingProviderCharacterProfileCompleter,
    complete_character_starters,
    content_with_character_starters,
    normalize_scenario_character_starters,
    scenario_character_starter_to_json,
    scenario_character_starters_for_content,
    scenario_context_text,
    starter_identity_locked_fields,
)
from bragi.services.character_registry_maintenance_service import (
    CharacterMaintenanceResult,
    CharacterRegistryMaintenanceService,
)
from bragi.services.character_registry_service import (
    CharacterFieldEnhanceResult,
    CharacterRegistryEdits,
    CharacterRegistryModel,
    CharacterRegistryRow,
    CharacterRegistryService,
)
from bragi.services.character_text_world_update_service import (
    CharacterTextWorldUpdateService,
)
from bragi.services.chat_bundle_service import (
    ChatBundlePreview,
    ChatBundleService,
    ImportedChatBundle,
)
from bragi.services.chat_service import (
    CHAT_TURN_CANCELLED_ERROR,
    TIMESKIP_SPEAKER_NAME,
    CancellationToken,
    ChatService,
    ChatTurnCancelled,
    NarratorStreamCallback,
    PostTurnProgressCallback,
    TurnProgressCallback,
    TurnRevisionBoundary,
    timeskip_message_body,
)
from bragi.services.content_rating import effective_content_safety_policy
from bragi.services.content_safety_service import (
    ContentSafetyAction,
    ContentSafetyResult,
    ContentSafetyService,
)
from bragi.services.context_cleanup_service import (
    CONTEXT_CLEANUP_ACTIONS_TASK,
    CONTEXT_CLEANUP_SCAN_TASK,
    GUIDED_CONTEXT_CLEANUP_TASK,
    ContextCleanupService,
)
from bragi.services.context_search_service import (
    ContextSearchResult,
    ContextSearchService,
)
from bragi.services.continuation_scenario_service import (
    ContinuationScenarioService,
    seed_continuation_characters,
)
from bragi.services.dating_route_service import DatingRouteService
from bragi.services.knowledge_boundary import (
    knowledge_edge_allows_prompt_use,
    knowledge_edge_scope_label,
    normalized_knowledge_target_type,
)
from bragi.services.media_service import MediaService
from bragi.services.memory_consolidation_service import MemoryConsolidationService
from bragi.services.message_reconciliation_service import MessageReconciliationService
from bragi.services.message_revision_service import MessageRevisionService
from bragi.services.model_capabilities import (
    IMAGE_TO_IMAGE_CAPABILITIES,
    IMAGE_TO_VIDEO_CAPABILITIES,
    STRUCTURED_OUTPUT_CAPABILITIES,
    TOOL_CALLING_CAPABILITIES,
    check_model_capabilities,
    known_model_is_unavailable,
    model_supports_any_capability,
    model_supports_any_capability_or_unknown,
)
from bragi.services.model_preferences import (
    CHARACTER_IMAGE_EDIT_PURPOSE,
    ROLEPLAY_TYPES,
    character_enhancement_model_preference,
    image_edit_model_preference,
    roleplay_model_preference,
    roleplay_model_preference_with_fallbacks,
    scenario_generation_model_preference,
)
from bragi.services.prompt_inspection import PromptInspectionStore
from bragi.services.save_fork_service import SaveForkService
from bragi.services.save_service import SaveService
from bragi.services.scenario_bundle_service import (
    ScenarioBundlePreview,
    ScenarioBundleService,
)
from bragi.services.scenario_content_rating import (
    metadata_with_scenario_content_ratings,
    scenario_content_rating,
)
from bragi.services.scenario_name_sources import (
    ordinary_name_starter_generation_context,
)
from bragi.services.scenario_service import (
    CHOOSE_YOUR_OWN_ADVENTURE_ALLOWED_SECTIONS,
    CHOOSE_YOUR_OWN_ADVENTURE_SECTIONS,
    DATING_SIM_ALLOWED_SECTIONS,
    DATING_SIM_SECTIONS,
    FANTASY_ROLEPLAY_ALLOWED_SECTIONS,
    FANTASY_ROLEPLAY_SECTIONS,
    FIRST_CONTACT_EXPLORATION_ALLOWED_SECTIONS,
    FIRST_CONTACT_EXPLORATION_SECTIONS,
    FULL_ROLEPLAY_ALLOWED_SECTIONS,
    FULL_ROLEPLAY_SECTIONS,
    HEIST_INFILTRATION_ALLOWED_SECTIONS,
    HEIST_INFILTRATION_SECTIONS,
    INVESTIGATION_MYSTERY_ALLOWED_SECTIONS,
    INVESTIGATION_MYSTERY_SECTIONS,
    MERCHANT_TRADE_ROUTE_ALLOWED_SECTIONS,
    MERCHANT_TRADE_ROUTE_SECTIONS,
    MONSTER_HUNT_BOUNTY_ALLOWED_SECTIONS,
    MONSTER_HUNT_BOUNTY_SECTIONS,
    POLITICAL_INTRIGUE_ALLOWED_SECTIONS,
    POLITICAL_INTRIGUE_SECTIONS,
    RETIRED_SCENARIO_REASON,
    ROAD_TRIP_PILGRIMAGE_ALLOWED_SECTIONS,
    ROAD_TRIP_PILGRIMAGE_SECTIONS,
    SCENARIO_GENRES_CONTENT_KEY,
    SCIENCE_FICTION_ROLEPLAY_ALLOWED_SECTIONS,
    SCIENCE_FICTION_ROLEPLAY_SECTIONS,
    SETTLEMENT_BUILDER_ALLOWED_SECTIONS,
    SETTLEMENT_BUILDER_SECTIONS,
    SURVIVAL_EXPEDITION_ALLOWED_SECTIONS,
    SURVIVAL_EXPEDITION_SECTIONS,
    TIME_LOOP_ALLOWED_SECTIONS,
    TIME_LOOP_SECTIONS,
    ScenarioDraft,
    ScenarioGenerationProgress,
    ScenarioService,
    ScenarioType,
    normalize_scenario_definition,
    normalize_scenario_draft_sections,
    normalized_scenario_types_and_flag,
    scenario_record_is_retired,
)
from bragi.services.state_pruning_service import StatePruningService
from bragi.services.summary_backfill_service import SummaryBackfillService
from bragi.services.summary_service import SummaryService
from bragi.services.turn_snapshot_service import TurnSnapshotService
from bragi.services.world_context_retention_service import WorldContextRetentionService
from bragi.services.world_data_service import WorldDataModel, WorldDataService
from bragi.services.world_suggestion_review_service import WorldSuggestionReviewService
from bragi_common.story_continuation import is_story_continuation_message


@dataclass(frozen=True)
class ManualScenarioInput:
    scenario_type: str
    title: str
    premise: str
    player_role: str
    interaction_mode: InteractionMode | str = InteractionMode.ROLEPLAY
    player_character_name: str = ""
    worldbuilding: str = ""
    lore: str = ""
    locations: str = ""
    factions: str = ""
    magic_system: str = ""
    realms_and_places: str = ""
    factions_and_orders: str = ""
    myths_and_creatures: str = ""
    quest_stakes: str = ""
    technology_level: str = ""
    setting_scope: str = ""
    species_and_intelligences: str = ""
    factions_and_institutions: str = ""
    mission_stakes: str = ""
    mission_profile: str = ""
    ship_or_base_status: str = ""
    exploration_target: str = ""
    unknown_intelligence: str = ""
    knowledge_state: str = ""
    translation_progress: str = ""
    discoveries_and_samples: str = ""
    hazards_and_escalation: str = ""
    expedition_goal: str = ""
    route_options: str = ""
    resource_inventory: str = ""
    environmental_conditions: str = ""
    hazards_and_events: str = ""
    camp_status: str = ""
    travel_progress: str = ""
    loop_premise: str = ""
    reset_trigger: str = ""
    loop_duration: str = ""
    starting_state: str = ""
    objective: str = ""
    failure_conditions: str = ""
    baseline_world_state: str = ""
    loop_schedule: str = ""
    persistent_knowledge: str = ""
    persistence_exceptions: str = ""
    npc_memory_rules: str = ""
    current_loop_state: str = ""
    case_facts: str = ""
    clues: str = ""
    timeline: str = ""
    red_herrings: str = ""
    hidden_truth: str = ""
    case_status: str = ""
    target_location: str = ""
    objectives_and_stakes: str = ""
    intel_and_access: str = ""
    security_model: str = ""
    alert_and_heat: str = ""
    loadout_and_tools: str = ""
    complications: str = ""
    extraction_routes: str = ""
    aftermath: str = ""
    political_arena: str = ""
    political_factions: str = ""
    central_conflict: str = ""
    secrets_and_leverage: str = ""
    reputation_and_standing: str = ""
    obligations_and_favors: str = ""
    alliances_and_rivalries: str = ""
    event_calendar: str = ""
    political_pressure: str = ""
    public_private_knowledge: str = ""
    settlement_profile: str = ""
    resources_and_indicators: str = ""
    projects_and_facilities: str = ""
    threats_and_opportunities: str = ""
    calendar_and_deadlines: str = ""
    hunt_profile: str = ""
    target_profile: str = ""
    leads_and_clues: str = ""
    hunt_locations: str = ""
    preparation_state: str = ""
    hunt_status: str = ""
    journey_profile: str = ""
    route_and_stops: str = ""
    transport_and_supplies: str = ""
    recurring_pressures: str = ""
    relationship_threads: str = ""
    journey_progress: str = ""
    trade_profile: str = ""
    cargo_inventory: str = ""
    markets_and_stops: str = ""
    contracts_and_debts: str = ""
    route_hazards: str = ""
    profit_and_loss: str = ""
    tone_genre: str = ""
    player_character_profile: str = ""
    starting_scene: str = ""
    action_choices_enabled: bool = False
    choice_style: str = ""
    opening_message: str = ""
    save_title: str = ""
    loss_conditions: tuple[tuple[str, str], ...] = ()
    scenario_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class SaveListItemModel:
    save_id: str
    title: str
    active: bool
    scenario_id: str | None = None
    scenario_title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_opened_at: str | None = None
    supported: bool = True
    unsupported_reason: str | None = None
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY


@dataclass(frozen=True)
class SavedScenarioModel:
    scenario_id: str
    scenario_type: str
    title: str
    premise: str
    player_role: str
    opening_message: str | None
    save_count: int
    created_at: str | None = None
    updated_at: str | None = None
    scenario_types: tuple[str, ...] = ()
    has_generation_prompt: bool = False
    action_choices_enabled: bool = False
    supported: bool = True
    unsupported_reason: str | None = None
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY


@dataclass(frozen=True)
class ScenarioDraftModel:
    scenario_type: str
    sections: tuple[tuple[str, str], ...]
    regeneration_seed: str = ""
    source_metadata: tuple[tuple[str, object], ...] = ()
    action_choices_enabled: bool = False
    scenario_types: tuple[str, ...] = ()
    character_starters: tuple[dict[str, object], ...] = ()
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY


@dataclass(frozen=True)
class ScenarioDraftProgressModel:
    scenario_type: str
    section_id: str
    status: str
    completed_sections: tuple[tuple[str, str], ...]
    completed_count: int
    total_count: int
    action_choices_enabled: bool = False
    error: str = ""
    scenario_types: tuple[str, ...] = ()
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY


@dataclass(frozen=True)
class ActionChoiceModel:
    choice_id: str
    ordinal: int
    body: str
    content_rating: str = "unclassified"


@dataclass(frozen=True)
class ActionChoicesModel:
    narrator_message_id: str
    choices: tuple[ActionChoiceModel, ...]


ScenarioDraftProgressCallback = Callable[
    [ScenarioDraftProgressModel],
    Awaitable[None] | None,
]


@dataclass(frozen=True)
class RuntimeModel:
    saves: tuple[SaveListItemModel, ...]
    active_save_id: str | None
    active_save_title: str | None
    custom_instructions: str
    scenario_title: str | None
    scene_title: str
    chronicle: ChronicleModel
    media: MediaModel | None
    action_choices: ActionChoicesModel | None
    action_choices_enabled: bool
    scenario_wizard: ScenarioWizardModel
    scenario_draft: ScenarioDraftModel | None
    model_indicator: str
    failed_save: bool = False
    composer_enabled: bool = True
    failure_text: str | None = None
    status: str | None = None
    error: str | None = None
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY


@dataclass(frozen=True)
class ChatTurnDeltaModel:
    save_id: str
    status: str | None
    error: str | None
    player_message_id: str | None
    narrator_message_id: str | None
    messages: tuple[ChronicleMessageModel, ...]
    action_choices: ActionChoicesModel | None
    save: SaveListItemModel | None
    fallback_used: bool = False
    context_trimmed: bool = False
    requires_full_refresh: bool = False
    kind: str = "chat_turn_delta"
    version: int = 1


@dataclass(frozen=True)
class SubmittedRuntimeTurn:
    model: RuntimeModel | None = None
    save_id: str | None = None
    player_message_id: str | None = None
    narrator_message_id: str | None = None
    turn_revision: TurnRevisionBoundary | None = None
    context_trimmed: bool = False
    prepared_action_choices: PreparedActionChoiceGeneration | None = None
    delta: ChatTurnDeltaModel | None = None

    @property
    def input_committed(self) -> bool:
        return self.player_message_id is not None

    @property
    def error(self) -> str | None:
        if self.model is not None:
            return self.model.error
        if self.delta is not None:
            return self.delta.error
        return None

    @property
    def has_post_turn_jobs(self) -> bool:
        return (
            self.save_id is not None
            and self.player_message_id is not None
            and self.narrator_message_id is not None
            and self.error is None
        )


class ContextSearchRunner(Protocol):
    async def search(
        self,
        *,
        save_id: str,
        player_message_id: str,
    ) -> ContextSearchResult: ...


class BragiRuntime:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
        media_dir: Path,
        context_search_service: ContextSearchRunner | None = None,
        summary_service: SummaryService | None = None,
        active_save_id: str | None = None,
        prompt_inspection_store: PromptInspectionStore | None = None,
        chronicle_message_limit: int | None = None,
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.media_dir = media_dir
        self.active_save_id = active_save_id
        self.prompt_inspection_store = (
            prompt_inspection_store or PromptInspectionStore()
        )
        self.context_search_service = context_search_service or ContextSearchService(
            repositories=repositories,
            providers=providers,
        )
        self.summary_service = summary_service
        self.chronicle_message_limit = chronicle_message_limit
        self._active_chat_cancellations: dict[str, CancellationToken] = {}
        self._queued_chat_submissions: set[str] = set()
        self._pending_chat_cancellations: set[str] = set()
        self._save_operation_locks: dict[str, threading.Lock] = {}
        self._save_operation_locks_guard = threading.Lock()
        self._maintenance_retry_drain_counts: dict[str, int] = {}
        self._maintenance_retry_drain_guard = threading.Lock()
        self._context_trimmed_narrator_message_ids: set[str] = set()
        self._deferred_automatic_image_payloads: dict[
            tuple[str, str], dict[str, object]
        ] = {}

    def cancel_active_chat_turn(self, save_id: str | None = None) -> bool:
        resolved_save_id = self.active_save_id if save_id is None else save_id
        if resolved_save_id is None:
            return False
        token = self._active_chat_cancellations.get(resolved_save_id)
        if token is None:
            if resolved_save_id not in self._queued_chat_submissions:
                return False
            self._pending_chat_cancellations.add(resolved_save_id)
        else:
            if not token.cancel():
                return False
        log_event("runtime.chat_turn_cancel_requested", save_id=resolved_save_id)
        return True

    def cancel_active_submit(self, *, save_id: str | None = None) -> bool:
        return self.cancel_active_chat_turn(save_id)

    def register_pending_chat_submit(self, *, save_id: str | None = None) -> bool:
        resolved_save_id = self.active_save_id if save_id is None else save_id
        if resolved_save_id is None:
            return False
        self._queued_chat_submissions.add(resolved_save_id)
        return True

    def build_model(
        self,
        *,
        status: str | None = None,
        error: str | None = None,
        active_save_id: str | None | object = ...,
        chronicle_message_limit: int | None | object = ...,
    ) -> RuntimeModel:
        requested_save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        requested_chronicle_message_limit = (
            self.chronicle_message_limit
            if chronicle_message_limit is ...
            else cast(int | None, chronicle_message_limit)
        )
        active_save = _active_save(self.repositories, requested_save_id)
        details = (
            self.repositories.load_chronicle_details(
                active_save.id,
                message_limit=requested_chronicle_message_limit,
            )
            if active_save is not None
            else None
        )
        messages = details.messages if details is not None else []
        roleplay_save = details is not None and details.scenario.type in ROLEPLAY_TYPES
        action_choices_enabled = (
            details is not None
            and details.save.interaction_mode is InteractionMode.ROLEPLAY
            and scenario_action_choices_enabled(details.scenario)
        )
        eligible_character_image_message_ids = (
            character_image_eligible_message_ids(
                self.repositories,
                save_id=active_save.id,
                messages=messages,
            )
            if active_save is not None and roleplay_save
            else frozenset()
        )
        return RuntimeModel(
            saves=tuple(
                _save_list_item_model(
                    self.repositories,
                    save,
                    active_save_id=active_save.id if active_save is not None else None,
                )
                for save in self.repositories.list_saves()
            ),
            active_save_id=active_save.id if active_save else None,
            active_save_title=active_save.title if active_save else None,
            custom_instructions=active_save.custom_instructions if active_save else "",
            scenario_title=details.scenario.title if details else None,
            scene_title=details.scenario.title if details else "No save loaded",
            chronicle=build_chronicle_model(
                messages,
                has_more_before=(
                    details.has_more_messages_before if details is not None else False
                ),
                player_speaker_name=(
                    _default_player_character_name(
                        repositories=self.repositories,
                        save_id=active_save.id,
                        scenario=details.scenario,
                    )
                    if details is not None and active_save is not None
                    else None
                ),
                character_image_actions_enabled=roleplay_save,
                character_image_message_ids=eligible_character_image_message_ids,
                scene_presence_actions_enabled=roleplay_save,
                debug_prompt_text_by_message_id=(
                    self.prompt_inspection_store.prompts_by_message_id()
                ),
                debug_provider_payload_text_by_message_id=(
                    self.prompt_inspection_store.provider_payloads_by_message_id()
                ),
                revision_metadata_by_message_id=(
                    _message_revision_metadata_for_messages(
                        self.repositories,
                        save_id=active_save.id,
                        messages=messages,
                    )
                    if active_save is not None
                    else {}
                ),
                debug_prompts_enabled=bool(
                    self.repositories.get_app_setting("debug_logging_enabled")
                ),
            ),
            media=(
                build_media_model(
                    repositories=self.repositories,
                    save_id=active_save.id,
                    providers=self.providers,
                    media_dir=self.media_dir,
                )
                if active_save
                else None
            ),
            action_choices=(
                _action_choices_model(self.repositories, save_id=active_save.id)
                if active_save is not None and action_choices_enabled
                else None
            ),
            action_choices_enabled=action_choices_enabled,
            scenario_wizard=build_scenario_wizard_model(),
            scenario_draft=None,
            model_indicator=_model_indicator(self.repositories),
            failed_save=False,
            composer_enabled=active_save is not None,
            failure_text=None,
            status=status,
            error=error,
            interaction_mode=(
                active_save.interaction_mode
                if active_save is not None
                else InteractionMode.ROLEPLAY
            ),
        )

    def build_shell_model(
        self,
        *,
        status: str | None = None,
        error: str | None = None,
        active_save_id: str | None | object = ...,
        chronicle_message_limit: int | None | object = ...,
    ) -> RuntimeModel:
        requested_save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        requested_chronicle_message_limit = (
            self.chronicle_message_limit
            if chronicle_message_limit is ...
            else cast(int | None, chronicle_message_limit)
        )
        active_save = _active_save(self.repositories, requested_save_id)
        details = (
            self.repositories.load_chronicle_details(
                active_save.id,
                message_limit=requested_chronicle_message_limit,
            )
            if active_save is not None
            else None
        )
        messages = details.messages if details is not None else []
        roleplay_save = details is not None and details.scenario.type in ROLEPLAY_TYPES
        action_choices_enabled = (
            details is not None
            and details.save.interaction_mode is InteractionMode.ROLEPLAY
            and scenario_action_choices_enabled(details.scenario)
        )
        eligible_character_image_message_ids = (
            character_image_eligible_message_ids(
                self.repositories,
                save_id=active_save.id,
                messages=messages,
            )
            if active_save is not None and roleplay_save
            else frozenset()
        )
        return RuntimeModel(
            saves=tuple(
                _save_list_item_model(
                    self.repositories,
                    save,
                    active_save_id=active_save.id if active_save is not None else None,
                )
                for save in self.repositories.list_saves()
            ),
            active_save_id=active_save.id if active_save else None,
            active_save_title=active_save.title if active_save else None,
            custom_instructions=active_save.custom_instructions if active_save else "",
            scenario_title=details.scenario.title if details else None,
            scene_title=details.scenario.title if details else "No save loaded",
            chronicle=build_chronicle_model(
                messages,
                has_more_before=(
                    details.has_more_messages_before if details is not None else False
                ),
                player_speaker_name=(
                    _default_player_character_name(
                        repositories=self.repositories,
                        save_id=active_save.id,
                        scenario=details.scenario,
                    )
                    if details is not None and active_save is not None
                    else None
                ),
                character_image_actions_enabled=roleplay_save,
                character_image_message_ids=eligible_character_image_message_ids,
                scene_presence_actions_enabled=roleplay_save,
                debug_prompt_text_by_message_id=(
                    self.prompt_inspection_store.prompts_by_message_id()
                ),
                debug_provider_payload_text_by_message_id=(
                    self.prompt_inspection_store.provider_payloads_by_message_id()
                ),
                revision_metadata_by_message_id=(
                    _message_revision_metadata_for_messages(
                        self.repositories,
                        save_id=active_save.id,
                        messages=messages,
                    )
                    if active_save is not None
                    else {}
                ),
                debug_prompts_enabled=bool(
                    self.repositories.get_app_setting("debug_logging_enabled")
                ),
            ),
            media=None,
            action_choices=(
                _action_choices_model(self.repositories, save_id=active_save.id)
                if active_save is not None and action_choices_enabled
                else None
            ),
            action_choices_enabled=action_choices_enabled,
            scenario_wizard=build_scenario_wizard_model(),
            scenario_draft=None,
            model_indicator=_model_indicator(self.repositories),
            failed_save=False,
            composer_enabled=active_save is not None,
            failure_text=None,
            status=status,
            error=error,
            interaction_mode=(
                active_save.interaction_mode
                if active_save is not None
                else InteractionMode.ROLEPLAY
            ),
        )

    def build_chronicle_page_model(
        self,
        *,
        save_id: str,
        before_message_id: str | None = None,
        limit: int = 80,
    ) -> ChronicleModel:
        details = self.repositories.load_chronicle_details(
            save_id,
            message_limit=limit,
            before_message_id=before_message_id,
        )
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        messages = details.messages
        roleplay_save = details.scenario.type in ROLEPLAY_TYPES
        eligible_character_image_message_ids = (
            character_image_eligible_message_ids(
                self.repositories,
                save_id=save_id,
                messages=messages,
            )
            if roleplay_save
            else frozenset()
        )
        return build_chronicle_model(
            messages,
            has_more_before=details.has_more_messages_before,
            player_speaker_name=_default_player_character_name(
                repositories=self.repositories,
                save_id=save_id,
                scenario=details.scenario,
            ),
            character_image_actions_enabled=roleplay_save,
            character_image_message_ids=eligible_character_image_message_ids,
            scene_presence_actions_enabled=roleplay_save,
            debug_prompt_text_by_message_id=(
                self.prompt_inspection_store.prompts_by_message_id()
            ),
            debug_provider_payload_text_by_message_id=(
                self.prompt_inspection_store.provider_payloads_by_message_id()
            ),
            revision_metadata_by_message_id={
                message_id: metadata
                for message_id, metadata in _message_revision_metadata_for_messages(
                    self.repositories,
                    save_id=save_id,
                    messages=messages,
                ).items()
            },
            debug_prompts_enabled=bool(
                self.repositories.get_app_setting("debug_logging_enabled")
            ),
        )

    def build_chat_turn_delta(
        self,
        *,
        save_id: str,
        player_message: MessageRecord,
        narrator_message: MessageRecord,
        status: str | None,
        error: str | None = None,
        fallback_used: bool = False,
        context_trimmed: bool = False,
    ) -> ChatTurnDeltaModel:
        details = self.repositories.load_save_details(save_id, message_limit=1)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        messages = [player_message, narrator_message]
        roleplay_save = details.scenario.type in ROLEPLAY_TYPES
        eligible_character_image_message_ids = (
            character_image_eligible_message_ids(
                self.repositories,
                save_id=save_id,
                messages=messages,
            )
            if roleplay_save
            else frozenset()
        )
        action_choices_enabled = scenario_action_choices_enabled(details.scenario)
        save = self.repositories.get_save(save_id)
        chronicle = build_chronicle_model(
            messages,
            player_speaker_name=_default_player_character_name(
                repositories=self.repositories,
                save_id=save_id,
                scenario=details.scenario,
            ),
            character_image_actions_enabled=roleplay_save,
            character_image_message_ids=eligible_character_image_message_ids,
            scene_presence_actions_enabled=roleplay_save,
            debug_prompt_text_by_message_id=(
                self.prompt_inspection_store.prompts_by_message_id()
            ),
            debug_provider_payload_text_by_message_id=(
                self.prompt_inspection_store.provider_payloads_by_message_id()
            ),
            revision_metadata_by_message_id=_message_revision_metadata_for_messages(
                self.repositories,
                save_id=save_id,
                messages=messages,
            ),
            debug_prompts_enabled=bool(
                self.repositories.get_app_setting("debug_logging_enabled")
            ),
        )
        return ChatTurnDeltaModel(
            save_id=save_id,
            status=status,
            error=error,
            player_message_id=(
                None
                if is_story_continuation_message(player_message)
                else player_message.id
            ),
            narrator_message_id=narrator_message.id,
            messages=chronicle.messages,
            action_choices=(
                _action_choices_model(self.repositories, save_id=save_id)
                if action_choices_enabled
                else None
            ),
            save=(
                _save_list_item_model(
                    self.repositories,
                    save,
                    active_save_id=save_id,
                )
                if save is not None
                else None
            ),
            fallback_used=fallback_used,
            context_trimmed=context_trimmed,
        )

    def build_model_with_pending_player_message(
        self,
        *,
        save_id: str,
        body: str,
        speaker_name: str | None = None,
        status: str | None = None,
    ) -> RuntimeModel:
        base = self.build_model(status=status, active_save_id=save_id)
        text = body.strip()
        if not text or base.active_save_id != save_id:
            return base
        display_speaker_name = _player_speaker_name(
            repositories=self.repositories,
            save_id=save_id,
            requested_name=speaker_name,
        )
        pending_message = ChronicleMessageModel(
            message_id="pending-player-message",
            role="player",
            speaker_name=display_speaker_name,
            body=text,
            markdown_blocks=parse_message_markdown(text),
            actions=(),
        )
        return replace(
            base,
            chronicle=replace(
                base.chronicle,
                messages=(*base.chronicle.messages, pending_message),
            ),
        )

    async def regenerate_action_choices(
        self,
        *,
        narrator_message_id: str,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.action_choice_regeneration_failed",
                narrator_message_id=narrator_message_id,
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")
        try:
            await ActionChoiceService(
                repositories=self.repositories,
                providers=self.providers,
            ).generate_for_message(
                save_id=save_id,
                narrator_message_id=narrator_message_id,
                current_user_id=current_user_id,
                retry_progress_callback=retry_progress_callback,
            )
        except Exception as exc:  # noqa: BLE001 - provider failures become model errors
            log_error_event(
                "runtime.action_choice_regeneration_failed",
                save_id=save_id,
                narrator_message_id=narrator_message_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.action_choice_regenerated",
            save_id=save_id,
            narrator_message_id=narrator_message_id,
        )
        TurnSnapshotService(self.repositories).capture_current_head_if_dirty(
            save_id,
            reason="action_choices",
        )
        return self.build_model(
            status="Action choices regenerated",
            active_save_id=save_id,
        )

    def update_custom_instructions(
        self,
        *,
        custom_instructions: str,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No save loaded")
        try:
            save = self.repositories.update_save_custom_instructions(
                save_id=save_id,
                custom_instructions=custom_instructions,
            )
        except Exception as exc:
            log_error_event(
                "runtime.custom_instructions_update_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        status = (
            "Response guidance saved"
            if save.custom_instructions
            else "Response guidance cleared"
        )
        return self.build_model(status=status, active_save_id=save_id)

    async def generate_scenario_draft(
        self,
        *,
        scenario_type: str,
        scenario_types: Iterable[str] | None = None,
        seed: str,
        interaction_mode: InteractionMode | str = InteractionMode.ROLEPLAY,
        action_choices_enabled: bool = False,
        progress_callback: ScenarioDraftProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        text = seed.strip()
        if not text:
            log_error_event(
                "runtime.scenario_draft_failed",
                scenario_type=scenario_type,
                error="Scenario seed is required",
            )
            return self.build_model(error="Scenario seed is required")
        preference = _scenario_generation_preference(self.repositories)
        if preference is None:
            log_error_event(
                "runtime.scenario_draft_failed",
                scenario_type=scenario_type,
                error="No scenario generation model preference configured",
            )
            return self.build_model(
                error="No scenario generation model preference configured",
            )
        if preference.provider not in self.providers:
            log_error_event(
                "runtime.scenario_draft_failed",
                scenario_type=scenario_type,
                provider=preference.provider,
                error="Scenario provider is unavailable",
            )
            return self.build_model(
                error=f"Scenario provider is unavailable: {preference.provider}",
            )
        try:
            async def notify_progress(
                progress: ScenarioGenerationProgress,
            ) -> None:
                if progress_callback is None:
                    return
                result = progress_callback(_scenario_progress_model(progress))
                if result is not None:
                    await result

            draft = await ScenarioService(
                repositories=self.repositories,
                provider=self.providers[preference.provider],
                provider_name=preference.provider,
                model_id=preference.model_id,
                providers=self.providers,
                current_user_id=current_user_id,
            ).generate_draft(
                scenario_type=scenario_type,
                scenario_types=scenario_types,
                seed=text,
                interaction_mode=interaction_mode,
                action_choices_enabled=action_choices_enabled,
                progress_callback=notify_progress,
            )
        except Exception as exc:
            log_error_event(
                "runtime.scenario_draft_failed",
                scenario_type=scenario_type,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(error=_user_visible_error(exc))
        log_event(
            "runtime.scenario_draft_succeeded",
            scenario_type=scenario_type,
            provider=preference.provider,
            model=preference.model_id,
        )
        return self._model_with_draft(draft, status="Scenario draft generated")

    async def generate_continuation_scenario_draft(
        self,
        *,
        active_save_id: str | None | object = ...,
        chapter_start_instructions: str = "",
        progress_callback: ScenarioDraftProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.continuation_scenario_draft_failed",
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")
        preference = _scenario_generation_preference(self.repositories)
        if preference is None:
            log_error_event(
                "runtime.continuation_scenario_draft_failed",
                save_id=save_id,
                error="No scenario generation model preference configured",
            )
            return self.build_model(
                error="No scenario generation model preference configured",
                active_save_id=save_id,
            )
        if preference.provider not in self.providers:
            log_error_event(
                "runtime.continuation_scenario_draft_failed",
                save_id=save_id,
                provider=preference.provider,
                error="Scenario provider is unavailable",
            )
            return self.build_model(
                error=f"Scenario provider is unavailable: {preference.provider}",
                active_save_id=save_id,
            )
        try:
            async def notify_progress(
                progress: ScenarioGenerationProgress,
            ) -> None:
                if progress_callback is None:
                    return
                result = progress_callback(_scenario_progress_model(progress))
                if result is not None:
                    await result

            draft = await ContinuationScenarioService(
                repositories=self.repositories,
                scenario_service=ScenarioService(
                    repositories=self.repositories,
                    provider=self.providers[preference.provider],
                    provider_name=preference.provider,
                    model_id=preference.model_id,
                    providers=self.providers,
                    current_user_id=current_user_id,
                ),
            ).generate_draft(
                save_id=save_id,
                chapter_start_instructions=chapter_start_instructions,
                progress_callback=notify_progress,
            )
        except Exception as exc:
            log_error_event(
                "runtime.continuation_scenario_draft_failed",
                save_id=save_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.continuation_scenario_draft_succeeded",
            save_id=save_id,
            provider=preference.provider,
            model=preference.model_id,
        )
        return self._model_with_draft(draft, status="Continuation draft generated")

    async def regenerate_scenario_section(
        self,
        *,
        scenario_type: str,
        scenario_types: Iterable[str] | None = None,
        seed: str,
        section_id: str,
        sections: dict[str, str],
        interaction_mode: InteractionMode | str = InteractionMode.ROLEPLAY,
        action_choices_enabled: bool = False,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        text = seed.strip()
        if not text:
            return self.build_model(error="Scenario seed is required")
        preference = _scenario_generation_preference(
            self.repositories,
            section_id=section_id,
        )
        if preference is None:
            return self.build_model(
                error="No scenario generation model preference configured",
            )
        if preference.provider not in self.providers:
            return self.build_model(
                error=f"Scenario provider is unavailable: {preference.provider}",
            )
        try:
            section_result = await ScenarioService(
                repositories=self.repositories,
                provider=self.providers[preference.provider],
                provider_name=preference.provider,
                model_id=preference.model_id,
                providers=self.providers,
                current_user_id=current_user_id,
            ).regenerate_section(
                scenario_type=scenario_type,
                scenario_types=scenario_types,
                seed=text,
                section_id=section_id,
                sections=sections,
                interaction_mode=interaction_mode,
                action_choices_enabled=action_choices_enabled,
            )
        except Exception as exc:
            log_error_event(
                "runtime.scenario_section_regeneration_failed",
                scenario_type=scenario_type,
                section_id=section_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(error=_user_visible_error(exc))
        updated_sections = {**sections, section_id: section_result.body}
        draft_type, draft_genres, normalized_action_choices_enabled = (
            normalized_scenario_types_and_flag(
                scenario_type,
                scenario_types=scenario_types,
                action_choices_enabled=action_choices_enabled,
            )
        )
        draft = ScenarioDraft(
            type=draft_type,
            interaction_mode=normalize_interaction_mode(interaction_mode),
            scenario_types=draft_genres,
            sections=updated_sections,
            metadata=metadata_with_scenario_content_ratings(
                None,
                aggregate_rating=effective_content_safety_policy(
                    self.repositories,
                    user_id=current_user_id,
                ).rating,
                section_ratings={
                    section_id: section_result.minimum_rating,
                },
            ),
            regeneration_seed=text,
            action_choices_enabled=normalized_action_choices_enabled,
        )
        log_event(
            "runtime.scenario_section_regenerated",
            scenario_type=scenario_type,
            section_id=section_id,
            provider=preference.provider,
            model=preference.model_id,
        )
        return self._model_with_draft(draft, status="Section regenerated")

    async def generate_scenario_draft_character_starters(
        self,
        *,
        scenario_type: str,
        scenario_types: Iterable[str] | None = None,
        sections: dict[str, str],
        interaction_mode: InteractionMode | str = InteractionMode.ROLEPLAY,
        character_starters: Iterable[Mapping[str, object]] | None = None,
        count: int | None = None,
        custom_description: str = "",
        action_choices_enabled: bool = False,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        try:
            draft_type, draft_genres, normalized_action_choices_enabled = (
                normalized_scenario_types_and_flag(
                    scenario_type,
                    scenario_types=scenario_types,
                    action_choices_enabled=action_choices_enabled,
                )
            )
            normalized_sections = normalize_scenario_draft_sections(
                draft_type,
                sections,
            )
            existing_starters = normalize_scenario_character_starters(
                list(character_starters or ()),
                strict=True,
            )
            description = custom_description.strip()
            if description:
                requested_count = 1
            else:
                if count is None or isinstance(count, bool):
                    return self.build_model(
                        error=(
                            "Number of characters or custom character "
                            "description is required"
                        )
                    )
                requested_count = count
            if requested_count < 1 or requested_count > 12:
                return self.build_model(
                    error="Number of characters must be between 1 and 12"
                )
            starter_type = _character_starter_scenario_type(
                draft_type,
                draft_genres,
            )
            preference = _context_update_preference_for_scenario_type(
                repositories=self.repositories,
                scenario_type=starter_type,
            )
            if preference is None:
                return self.build_model(
                    error="No context update model preference configured",
                )
            if preference.provider not in self.providers:
                return self.build_model(
                    error=(
                        "Context Update provider is unavailable: "
                        f"{preference.provider}"
                    ),
                )
            provider = self.providers[preference.provider]
            structured_completer = self._structured_character_profile_completer(
                preference,
                provider,
            )
            if structured_completer is None:
                return self.build_model(
                    error=(
                        "Character starter generation requires a "
                        "structured-output model"
                    ),
                )
            player_name = normalized_sections.get("player_character_name", "")
            name_context = ordinary_name_starter_generation_context(
                scenario_type=tuple(genre.value for genre in draft_genres),
                seed=normalized_sections.get("premise", ""),
                sections=normalized_sections,
                player_character_name=player_name,
                existing_starter_names=_starter_generation_existing_names(
                    existing_starters
                ),
            )
            generated_starters = await structured_completer.generate_starters(
                CharacterStarterGenerationRequest(
                    scenario_type=starter_type,
                    scenario_types=tuple(genre.value for genre in draft_genres),
                    scenario_context=scenario_context_text(
                        scenario_type=starter_type,
                        content=normalized_sections,
                    ),
                    content=normalized_sections,
                    existing_starters=existing_starters,
                    count=requested_count,
                    custom_description=description,
                    name_candidate_context=name_context,
                    save_id=None,
                )
            )
            generated_starters, generated_starters_rating = (
                await self._review_scenario_character_starters(
                    starters=generated_starters,
                    save_id=None,
                    current_user_id=current_user_id,
                    roleplay_type=draft_type.value,
                )
            )
            draft = ScenarioDraft(
                type=draft_type,
                interaction_mode=normalize_interaction_mode(interaction_mode),
                scenario_types=draft_genres,
                sections=normalized_sections,
                metadata=metadata_with_scenario_content_ratings(
                    None,
                    aggregate_rating=generated_starters_rating or "unclassified",
                    section_ratings=(
                        {"character_starters": generated_starters_rating}
                        if generated_starters_rating is not None
                        else {}
                    ),
                ),
                action_choices_enabled=normalized_action_choices_enabled,
                character_starters=(*existing_starters, *generated_starters),
            )
            log_event(
                "runtime.scenario_draft_character_starters_generated",
                scenario_type=draft_type.value,
                scenario_types=tuple(genre.value for genre in draft_genres),
                provider=preference.provider,
                model=preference.model_id,
                generated_count=len(generated_starters),
                current_user_id=current_user_id,
            )
            return self._model_with_draft(
                draft,
                status="Character starters generated",
            )
        except Exception as exc:
            log_error_event(
                "runtime.scenario_draft_character_starters_failed",
                scenario_type=scenario_type,
                **exception_log_fields(exc),
            )
            return self.build_model(error=_user_visible_error(exc))

    async def save_scenario_draft(
        self,
        *,
        scenario_type: str,
        scenario_types: Iterable[str] | None = None,
        sections: dict[str, str],
        interaction_mode: InteractionMode | str = InteractionMode.ROLEPLAY,
        character_starters: Iterable[Mapping[str, object]] | None = None,
        action_choices_enabled: bool = False,
        save_title: str = "",
        source_metadata: dict[str, object] | None = None,
        owner_user_id: str | None = None,
        remember_process_active_save: bool = True,
        current_user_id: str | None = None,
        defer_opening_action_choices: bool = False,
    ) -> RuntimeModel:
        try:
            source_metadata = _scenario_source_metadata_without_loss_conditions(
                source_metadata
            )
            normalized_interaction_mode = normalize_interaction_mode(
                interaction_mode
            )
            draft_type, draft_genres, action_choices_enabled = (
                normalized_scenario_types_and_flag(
                    scenario_type,
                    scenario_types=scenario_types,
                    action_choices_enabled=action_choices_enabled,
                )
            )
            normalized_sections = normalize_scenario_draft_sections(
                draft_type,
                sections,
            )
            reviewed_sections: dict[str, str] = {}
            section_content_ratings: dict[str, str] = {}
            for section_id, section_body in normalized_sections.items():
                safety = await self._review_actor_content(
                    body=section_body,
                    save_id=None,
                    current_user_id=current_user_id,
                    roleplay_type=draft_type.value,
                )
                reviewed_sections[section_id] = safety.body
                section_content_ratings[section_id] = (
                    safety.reviewed_content_rating
                )
            normalized_sections = reviewed_sections
            normalized_starters = normalize_scenario_character_starters(
                list(character_starters or ()),
                strict=True,
            )
            (
                normalized_starters,
                character_starters_rating,
            ) = await self._review_scenario_character_starters(
                starters=normalized_starters,
                save_id=None,
                current_user_id=current_user_id,
                roleplay_type=draft_type.value,
            )
            if character_starters_rating is not None:
                section_content_ratings["character_starters"] = (
                    character_starters_rating
                )
            source_metadata = metadata_with_scenario_content_ratings(
                source_metadata,
                aggregate_rating=maximum_content_rating(
                    tuple(section_content_ratings.values())
                ),
                section_ratings=section_content_ratings,
            )
            draft = ScenarioDraft(
                type=draft_type,
                interaction_mode=normalized_interaction_mode,
                scenario_types=draft_genres,
                sections=normalized_sections,
                metadata=source_metadata,
                action_choices_enabled=action_choices_enabled,
                character_starters=normalized_starters,
            )
            scenario_id = _persist_scenario_draft(
                self.repositories,
                draft,
            )
            save = SaveService(self.repositories).create_save(
                scenario_id=scenario_id,
                title=save_title.strip() or draft.title,
                owner_user_id=owner_user_id,
            )
            opening = draft.sections.get("opening_message", "").strip()
            opening_message_id: str | None = None
            if opening:
                safety = await self._review_actor_content(
                    body=opening,
                    save_id=save.id,
                    current_user_id=current_user_id,
                )
                opening = safety.body
                message = self.repositories.append_message(
                    save_id=save.id,
                    role="narrator",
                    speaker_name="Narrator",
                    body=opening,
                    content_rating=safety.reviewed_content_rating,
                    safety_transition=_content_safety_transition(safety),
                )
                opening_message_id = message.id
            seeded_character_count = (
                seed_continuation_characters(
                    repositories=self.repositories,
                    save_id=save.id,
                    metadata=source_metadata or {},
                    source_message_id=opening_message_id,
                    include_player_character=(
                        normalized_interaction_mode is InteractionMode.ROLEPLAY
                    ),
                )
            )
            persisted_scenario = self.repositories.get_scenario(scenario_id)
            seed_content = (
                _scenario_content(persisted_scenario.content_json)
                if persisted_scenario is not None
                else dict(draft.sections)
            )
            seeded_character_count += (
                _seed_initial_character_registry(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=draft.type,
                scenario_types=draft.scenario_types,
                content=seed_content,
                source_message_id=opening_message_id,
                media_service=self._media_service(),
                interaction_mode=normalized_interaction_mode,
            )
            )
            seeded_first_contact_state_count = _seed_first_contact_exploration_state(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=draft.type,
                scenario_types=draft.scenario_types,
                content=seed_content,
                source_message_id=opening_message_id,
            )
            seeded_expedition_state_count = _seed_survival_expedition_state(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=draft.type,
                scenario_types=draft.scenario_types,
                content=seed_content,
                source_message_id=opening_message_id,
            )
            seeded_loop_state_count = _seed_time_loop_state(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=draft.type,
                scenario_types=draft.scenario_types,
                content=seed_content,
                source_message_id=opening_message_id,
            )
            seeded_heist_state_count = _seed_heist_infiltration_state(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=draft.type,
                scenario_types=draft.scenario_types,
                content=seed_content,
                source_message_id=opening_message_id,
            )
            seeded_intrigue_state_count = _seed_political_intrigue_state(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=draft.type,
                scenario_types=draft.scenario_types,
                content=seed_content,
                source_message_id=opening_message_id,
            )
            seeded_settlement_state_count = _seed_settlement_builder_state(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=draft.type,
                scenario_types=draft.scenario_types,
                content=seed_content,
                source_message_id=opening_message_id,
            )
            seeded_hunt_state_count = _seed_monster_hunt_bounty_state(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=draft.type,
                scenario_types=draft.scenario_types,
                content=seed_content,
                source_message_id=opening_message_id,
            )
            seeded_journey_state_count = _seed_road_trip_pilgrimage_state(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=draft.type,
                scenario_types=draft.scenario_types,
                content=seed_content,
                source_message_id=opening_message_id,
            )
            seeded_trade_state_count = _seed_merchant_trade_route_state(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=draft.type,
                scenario_types=draft.scenario_types,
                content=seed_content,
                source_message_id=opening_message_id,
            )
            if not defer_opening_action_choices:
                await self._generate_opening_action_choices_if_configured(
                    save_id=save.id,
                    opening_message_id=opening_message_id,
                    current_user_id=current_user_id,
                )
            TurnSnapshotService(self.repositories).capture_current_head_if_dirty(
                save.id,
                reason="opening_message",
            )
            if remember_process_active_save:
                self.active_save_id = save.id
        except Exception as exc:
            log_error_event(
                "runtime.save_scenario_draft_failed",
                scenario_type=scenario_type,
                **exception_log_fields(exc),
            )
            return self.build_model(error=_user_visible_error(exc))
        log_event(
            "runtime.save_scenario_draft_succeeded",
            scenario_type=draft.type.value,
            save_id=save.id,
            scenario_id=scenario_id,
            opening_message_chars=len(opening),
            seeded_character_count=seeded_character_count,
            seeded_first_contact_state_count=seeded_first_contact_state_count,
            seeded_expedition_state_count=seeded_expedition_state_count,
            seeded_loop_state_count=seeded_loop_state_count,
            seeded_heist_state_count=seeded_heist_state_count,
            seeded_intrigue_state_count=seeded_intrigue_state_count,
            seeded_settlement_state_count=seeded_settlement_state_count,
            seeded_hunt_state_count=seeded_hunt_state_count,
            seeded_journey_state_count=seeded_journey_state_count,
            seeded_trade_state_count=seeded_trade_state_count,
        )
        return self.build_model(
            status=f"Created save: {save.title}",
            active_save_id=save.id,
        )

    def create_manual_scenario(
        self,
        scenario: ManualScenarioInput,
        *,
        owner_user_id: str | None = None,
        remember_process_active_save: bool = True,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        title = _required_text(scenario.title, "Scenario title")
        interaction_mode = normalize_interaction_mode(scenario.interaction_mode)
        scenario_type, scenario_types, action_choices_enabled = (
            normalized_scenario_types_and_flag(
                scenario.scenario_type,
                scenario_types=scenario.scenario_types,
                action_choices_enabled=scenario.action_choices_enabled,
            )
        )
        content = _manual_scenario_content(
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
        if interaction_mode is InteractionMode.STORYTELLER:
            for player_section in (
                "player_character_name",
                "player_character_profile",
                "player_role",
            ):
                content.pop(player_section, None)
            content = content_with_action_choices_enabled(content, enabled=False)
        premise, content = normalize_scenario_definition(
            scenario_type=scenario_type,
            premise=_required_text(scenario.premise, "Premise"),
            content=content,
        )
        content = self._content_with_completed_character_starters(
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=content,
        )
        reviewed_content = dict(content)
        section_content_ratings: dict[str, str] = {}
        for section_id, section_body in content.items():
            if not isinstance(section_body, str) or not section_body.strip():
                continue
            safety = self._review_actor_content_blocking(
                body=section_body,
                save_id=None,
                current_user_id=current_user_id,
                roleplay_type=scenario_type.value,
            )
            reviewed_content[section_id] = safety.body
            section_content_ratings[section_id] = safety.reviewed_content_rating
        content = reviewed_content
        starter_type = _character_starter_scenario_type(
            scenario_type,
            scenario_types,
        )
        starters = scenario_character_starters_for_content(
            scenario_type=starter_type,
            content=content,
        )
        starters, character_starters_rating = (
            self._review_scenario_character_starters_blocking(
                starters=starters,
                save_id=None,
                current_user_id=current_user_id,
                roleplay_type=scenario_type.value,
            )
        )
        content = content_with_character_starters(
            scenario_type=starter_type,
            content=content,
            starters=starters,
        )
        if character_starters_rating is not None:
            section_content_ratings["character_starters"] = (
                character_starters_rating
            )
        title = str(content.get("title", title)).strip() or title
        premise = str(content.get("premise", premise)).strip() or premise
        player_role = str(
            content.get("player_role", scenario.player_role)
        ).strip()
        if interaction_mode is InteractionMode.ROLEPLAY:
            player_role = player_role or _required_text(
                scenario.player_role,
                "Player role",
            )
        source_metadata = content.get("_source")
        content["_source"] = metadata_with_scenario_content_ratings(
            source_metadata if isinstance(source_metadata, Mapping) else None,
            aggregate_rating=maximum_content_rating(
                tuple(section_content_ratings.values())
            ),
            section_ratings=section_content_ratings,
        )
        record = self.repositories.create_scenario(
            type=scenario_type.value,
            title=title,
            premise=premise,
            player_role=player_role,
            content=content,
            interaction_mode=interaction_mode,
        )
        save = SaveService(self.repositories).create_save(
            scenario_id=record.id,
            title=scenario.save_title.strip() or title,
            owner_user_id=owner_user_id,
        )
        opening_value = content.get("opening_message")
        opening = opening_value.strip() if isinstance(opening_value, str) else ""
        opening_message_id: str | None = None
        if opening:
            safety = self._review_actor_content_blocking(
                body=opening,
                save_id=save.id,
                current_user_id=current_user_id,
            )
            opening = safety.body
            message = self.repositories.append_message(
                save_id=save.id,
                role="narrator",
                speaker_name="Narrator",
                body=opening,
                content_rating=safety.reviewed_content_rating,
                safety_transition=_content_safety_transition(safety),
            )
            opening_message_id = message.id
        seeded_character_count = (
            _seed_initial_character_registry(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=scenario_type,
                scenario_types=scenario_types,
                content=content,
                source_message_id=opening_message_id,
                media_service=self._media_service(),
                interaction_mode=interaction_mode,
            )
        )
        seeded_first_contact_state_count = _seed_first_contact_exploration_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=content,
            source_message_id=opening_message_id,
        )
        seeded_expedition_state_count = _seed_survival_expedition_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=content,
            source_message_id=opening_message_id,
        )
        seeded_loop_state_count = _seed_time_loop_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=content,
            source_message_id=opening_message_id,
        )
        seeded_heist_state_count = _seed_heist_infiltration_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=content,
            source_message_id=opening_message_id,
        )
        seeded_intrigue_state_count = _seed_political_intrigue_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=content,
            source_message_id=opening_message_id,
        )
        seeded_settlement_state_count = _seed_settlement_builder_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=content,
            source_message_id=opening_message_id,
        )
        seeded_hunt_state_count = _seed_monster_hunt_bounty_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=content,
            source_message_id=opening_message_id,
        )
        seeded_journey_state_count = _seed_road_trip_pilgrimage_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=content,
            source_message_id=opening_message_id,
        )
        seeded_trade_state_count = _seed_merchant_trade_route_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=content,
            source_message_id=opening_message_id,
        )
        if interaction_mode is InteractionMode.ROLEPLAY:
            self._generate_opening_action_choices_if_configured_blocking(
                save_id=save.id,
                opening_message_id=opening_message_id,
                current_user_id=current_user_id,
            )
        TurnSnapshotService(self.repositories).capture_current_head_if_dirty(
            save.id,
            reason="opening_message",
        )
        if remember_process_active_save:
            self.active_save_id = save.id
        log_event(
            "runtime.manual_scenario_created",
            scenario_type=scenario_type.value,
            save_id=save.id,
            scenario_id=record.id,
            opening_message_chars=len(opening),
            seeded_character_count=seeded_character_count,
            seeded_first_contact_state_count=seeded_first_contact_state_count,
            seeded_expedition_state_count=seeded_expedition_state_count,
            seeded_loop_state_count=seeded_loop_state_count,
            seeded_heist_state_count=seeded_heist_state_count,
            seeded_intrigue_state_count=seeded_intrigue_state_count,
            seeded_settlement_state_count=seeded_settlement_state_count,
            seeded_hunt_state_count=seeded_hunt_state_count,
            seeded_journey_state_count=seeded_journey_state_count,
            seeded_trade_state_count=seeded_trade_state_count,
        )
        return self.build_model(
            status=f"Created save: {save.title}",
            active_save_id=save.id,
        )

    def list_saved_scenarios(
        self,
        *,
        current_user_id: str | None = None,
    ) -> tuple[SavedScenarioModel, ...]:
        save_counts = self.repositories.count_saves_by_scenario()
        scenarios: list[SavedScenarioModel] = []
        allowed_rating = effective_content_safety_policy(
            self.repositories,
            user_id=current_user_id,
        ).rating
        for scenario in self.repositories.list_scenarios():
            if content_rating_exceeds(
                minimum_rating=scenario_content_rating(scenario.content_json),
                allowed_rating=allowed_rating,
            ):
                continue
            content = _scenario_content(scenario.content_json)
            supported = not scenario_record_is_retired(scenario.type, content)
            scenario_type, scenario_types = _saved_scenario_type_values(
                scenario.type,
                content,
            )
            scenarios.append(
                SavedScenarioModel(
                    scenario_id=scenario.id,
                    scenario_type=scenario_type,
                    scenario_types=scenario_types,
                    title=scenario.title,
                    premise=scenario.premise,
                    player_role=scenario.player_role,
                    opening_message=_scenario_opening_message(scenario.content_json),
                    save_count=save_counts.get(scenario.id, 0),
                    created_at=scenario.created_at,
                    updated_at=scenario.updated_at,
                    has_generation_prompt=(
                        _scenario_generation_prompt(scenario.content_json) is not None
                    ),
                    action_choices_enabled=scenario_action_choices_enabled(scenario),
                    supported=supported,
                    unsupported_reason=(
                        None if supported else RETIRED_SCENARIO_REASON
                    ),
                    interaction_mode=scenario.interaction_mode,
                )
            )
        return tuple(scenarios)

    def start_saved_scenario(
        self,
        *,
        scenario_id: str,
        save_title: str = "",
        owner_user_id: str | None = None,
        remember_process_active_save: bool = True,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        scenario = self.repositories.get_scenario(scenario_id)
        if scenario is None:
            log_error_event(
                "runtime.saved_scenario_start_failed",
                scenario_id=scenario_id,
                error="Unknown scenario id",
            )
            return self.build_model(error=f"Unknown scenario id: {scenario_id}")
        if content_rating_exceeds(
            minimum_rating=scenario_content_rating(scenario.content_json),
            allowed_rating=effective_content_safety_policy(
                self.repositories,
                user_id=current_user_id,
            ).rating,
        ):
            return self.build_model(error="Scenario exceeds your content rating")
        if scenario_record_is_retired(
            scenario.type,
            _scenario_content(scenario.content_json),
        ):
            return self.build_model(error=RETIRED_SCENARIO_REASON)

        save = SaveService(self.repositories).create_save(
            scenario_id=scenario.id,
            title=save_title.strip() or scenario.title,
            owner_user_id=owner_user_id,
        )
        opening = _scenario_opening_message(scenario.content_json) or ""
        opening_message_id: str | None = None
        if opening:
            safety = self._review_actor_content_blocking(
                body=opening,
                save_id=save.id,
                current_user_id=current_user_id,
            )
            opening = safety.body
            message = self.repositories.append_message(
                save_id=save.id,
                role="narrator",
                speaker_name="Narrator",
                body=opening,
                content_rating=safety.reviewed_content_rating,
                safety_transition=_content_safety_transition(safety),
            )
            opening_message_id = message.id
        seeded_character_count = (
            seed_continuation_characters(
                repositories=self.repositories,
                save_id=save.id,
                metadata=_scenario_source_metadata(scenario.content_json),
                source_message_id=opening_message_id,
                include_player_character=(
                    save.interaction_mode is InteractionMode.ROLEPLAY
                ),
            )
        )
        scenario_type = ScenarioType(scenario.type)
        scenario_content = _scenario_content(scenario.content_json)
        scenario_types = _scenario_types_from_content(
            scenario_type,
            scenario_content,
        )
        seeded_character_count += (
            _seed_initial_character_registry(
                repositories=self.repositories,
                save_id=save.id,
                scenario_type=scenario_type,
                scenario_types=scenario_types,
                content=scenario_content,
                source_message_id=opening_message_id,
                media_service=self._media_service(),
                interaction_mode=save.interaction_mode,
            )
        )
        seeded_first_contact_state_count = _seed_first_contact_exploration_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=scenario_content,
            source_message_id=opening_message_id,
        )
        seeded_expedition_state_count = _seed_survival_expedition_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=scenario_content,
            source_message_id=opening_message_id,
        )
        seeded_loop_state_count = _seed_time_loop_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=scenario_content,
            source_message_id=opening_message_id,
        )
        seeded_heist_state_count = _seed_heist_infiltration_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=scenario_content,
            source_message_id=opening_message_id,
        )
        seeded_intrigue_state_count = _seed_political_intrigue_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=scenario_content,
            source_message_id=opening_message_id,
        )
        seeded_settlement_state_count = _seed_settlement_builder_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=scenario_content,
            source_message_id=opening_message_id,
        )
        seeded_hunt_state_count = _seed_monster_hunt_bounty_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=scenario_content,
            source_message_id=opening_message_id,
        )
        seeded_journey_state_count = _seed_road_trip_pilgrimage_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=scenario_content,
            source_message_id=opening_message_id,
        )
        seeded_trade_state_count = _seed_merchant_trade_route_state(
            repositories=self.repositories,
            save_id=save.id,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            content=scenario_content,
            source_message_id=opening_message_id,
        )
        if save.interaction_mode is InteractionMode.ROLEPLAY:
            self._generate_opening_action_choices_if_configured_blocking(
                save_id=save.id,
                opening_message_id=opening_message_id,
                current_user_id=current_user_id,
            )
        TurnSnapshotService(self.repositories).capture_current_head_if_dirty(
            save.id,
            reason="opening_message",
        )
        if remember_process_active_save:
            self.active_save_id = save.id
        log_event(
            "runtime.saved_scenario_started",
            save_id=save.id,
            scenario_id=scenario.id,
            opening_message_chars=len(opening),
            seeded_character_count=seeded_character_count,
            seeded_first_contact_state_count=seeded_first_contact_state_count,
            seeded_expedition_state_count=seeded_expedition_state_count,
            seeded_loop_state_count=seeded_loop_state_count,
            seeded_heist_state_count=seeded_heist_state_count,
            seeded_intrigue_state_count=seeded_intrigue_state_count,
            seeded_settlement_state_count=seeded_settlement_state_count,
            seeded_hunt_state_count=seeded_hunt_state_count,
            seeded_journey_state_count=seeded_journey_state_count,
            seeded_trade_state_count=seeded_trade_state_count,
        )
        return self.build_model(
            status=f"Created save: {save.title}",
            active_save_id=save.id,
        )

    async def _review_actor_content(
        self,
        *,
        body: str,
        save_id: str | None,
        current_user_id: str | None,
        roleplay_type: str | None = None,
    ) -> ContentSafetyResult:
        policy = effective_content_safety_policy(
            self.repositories,
            user_id=current_user_id,
        )
        return await ContentSafetyService(
            repositories=self.repositories,
            providers=self.providers,
        ).review_narration(
            body=body,
            content_rating=policy.rating,
            fade_to_black_enabled=policy.fade_to_black_enabled,
            save_id=save_id,
            roleplay_type=roleplay_type,
        )

    def _review_actor_content_blocking(
        self,
        *,
        body: str,
        save_id: str | None,
        current_user_id: str | None,
        roleplay_type: str | None = None,
    ) -> ContentSafetyResult:
        result: list[ContentSafetyResult] = []

        async def review() -> None:
            result.append(
                await self._review_actor_content(
                    body=body,
                    save_id=save_id,
                    current_user_id=current_user_id,
                    roleplay_type=roleplay_type,
                )
            )

        _run_coroutine_blocking(review())
        return result[0]

    async def _generate_opening_action_choices_if_configured(
        self,
        *,
        save_id: str,
        opening_message_id: str | None,
        current_user_id: str | None = None,
    ) -> None:
        if opening_message_id is None:
            return
        try:
            await ActionChoiceService(
                repositories=self.repositories,
                providers=self.providers,
            ).generate_for_message(
                save_id=save_id,
                narrator_message_id=opening_message_id,
                current_user_id=current_user_id,
            )
        except Exception as exc:
            log_error_event(
                "runtime.opening_action_choice_generation_failed",
                save_id=save_id,
                narrator_message_id=opening_message_id,
                **exception_log_fields(exc),
            )

    def _generate_opening_action_choices_if_configured_blocking(
        self,
        *,
        save_id: str,
        opening_message_id: str | None,
        current_user_id: str | None = None,
    ) -> None:
        _run_coroutine_blocking(
            self._generate_opening_action_choices_if_configured(
                save_id=save_id,
                opening_message_id=opening_message_id,
                current_user_id=current_user_id,
            )
        )

    def delete_saved_scenario(self, scenario_id: str) -> RuntimeModel:
        scenario = self.repositories.get_scenario(scenario_id)
        if scenario is None:
            log_error_event(
                "runtime.saved_scenario_delete_failed",
                scenario_id=scenario_id,
                error="Unknown scenario id",
            )
            return self.build_model(error=f"Unknown scenario id: {scenario_id}")

        save_count = self.repositories.count_saves_for_scenario(scenario_id)
        if save_count > 0:
            message = (
                "Cannot delete a scenario that has existing saves. "
                "Delete those saves first."
            )
            log_error_event(
                "runtime.saved_scenario_delete_failed",
                scenario_id=scenario_id,
                linked_save_count=save_count,
                error="Scenario has existing saves",
            )
            return self.build_model(error=message)

        self.repositories.delete_scenario(scenario_id)
        log_event(
            "runtime.saved_scenario_deleted",
            scenario_id=scenario_id,
        )
        return self.build_model(status=f"Deleted scenario: {scenario.title}")

    def load_save(
        self,
        save_id: str,
        *,
        remember_process_active_save: bool = True,
    ) -> RuntimeModel:
        if _save_has_retired_scenario(self.repositories, save_id):
            return self.build_model(error=RETIRED_SCENARIO_REASON)
        SaveService(self.repositories).load_save(save_id)
        if remember_process_active_save:
            self.active_save_id = save_id
        log_event("runtime.save_loaded", save_id=save_id)
        return self.build_model(status="Save loaded", active_save_id=save_id)

    def rename_save(
        self,
        *,
        save_id: str,
        title: str,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        selected_save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if _save_has_retired_scenario(self.repositories, save_id):
            return self.build_model(
                error=RETIRED_SCENARIO_REASON,
                active_save_id=selected_save_id,
            )
        try:
            with self._thread_save_operation_lock(save_id):
                save = SaveService(self.repositories).rename_save(save_id, title)
        except Exception as exc:
            log_error_event(
                "runtime.save_rename_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=selected_save_id,
            )
        log_event(
            "runtime.save_renamed",
            save_id=save_id,
            title_chars=len(save.title),
        )
        return self.build_model(
            status=f"Renamed save: {save.title}",
            active_save_id=selected_save_id,
        )

    def build_chat_history_model(
        self,
        *,
        selected_filter: str = "all",
        active_save_id: str | None | object = ...,
        before_message_id: str | None = None,
        limit: int = 80,
    ) -> ChatHistoryModel:
        requested_save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        active_save = _active_save(self.repositories, requested_save_id)
        if active_save is None:
            return build_chat_history_model(
                repositories=self.repositories,
                save_id=None,
                selected_filter=selected_filter,
                before_message_id=before_message_id,
                limit=limit,
            )
        details = self.repositories.load_save_details(active_save.id)
        return build_chat_history_model(
            repositories=self.repositories,
            save_id=active_save.id,
            save_title=active_save.title,
            selected_filter=selected_filter,
            before_message_id=before_message_id,
            limit=limit,
            player_speaker_name=(
                _default_player_character_name(
                    repositories=self.repositories,
                    save_id=active_save.id,
                    scenario=details.scenario,
                )
                if details is not None
                else None
            ),
        )

    def delete_save(self, save_id: str) -> RuntimeModel:
        save = self.repositories.get_save(save_id)
        if save is None:
            log_error_event(
                "runtime.save_delete_failed",
                save_id=save_id,
                error="Unknown save id",
            )
            return self.build_model(error=f"Unknown save id: {save_id}")

        with self._thread_save_operation_lock(save_id):
            try:
                SaveService(self.repositories).delete_save(
                    save_id, media_dir=self.media_dir
                )
            except Exception as exc:
                log_error_event(
                    "runtime.save_delete_failed",
                    save_id=save_id,
                    **exception_log_fields(exc),
                )
                return self.build_model(
                    error=_user_visible_error(exc),
                    active_save_id=self.active_save_id,
                )
        self._active_chat_cancellations.pop(save_id, None)
        self._queued_chat_submissions.discard(save_id)
        self._pending_chat_cancellations.discard(save_id)
        if self.active_save_id == save_id:
            self.active_save_id = None
        log_event("runtime.save_deleted", save_id=save_id)
        return self.build_model(status=f"Deleted save: {save.title}")

    def delete_media_asset(
        self,
        media_asset_id: str,
        *,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.media_asset_delete_failed",
                media_asset_id=media_asset_id,
                error="No save loaded",
            )
            return self.build_model(error="No save loaded", active_save_id=save_id)

        with self._thread_save_operation_lock(save_id):
            asset = self.repositories.archive_media_asset(
                save_id=save_id,
                media_asset_id=media_asset_id,
            )
        if asset is None:
            log_error_event(
                "runtime.media_asset_delete_failed",
                save_id=save_id,
                media_asset_id=media_asset_id,
                error="Unknown media asset id",
            )
            return self.build_model(
                error="Media asset not found",
                active_save_id=save_id,
            )

        log_event(
            "runtime.media_asset_deleted",
            save_id=save_id,
            media_asset_id=asset.id,
            media_type=asset.type,
        )
        return self.build_model(status="Media deleted", active_save_id=save_id)

    def set_character_reference_image(
        self,
        media_asset_id: str,
        *,
        character_id: str | None = None,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.character_reference_update_failed",
                media_asset_id=media_asset_id,
                error="No save loaded",
            )
            return self.build_model(error="No save loaded", active_save_id=save_id)

        try:
            with self._thread_save_operation_lock(save_id):
                asset = self._media_service().set_character_reference_image(
                    save_id=save_id,
                    media_asset_id=media_asset_id,
                    character_id=character_id,
                )
        except Exception as exc:
            log_error_event(
                "runtime.character_reference_update_failed",
                save_id=save_id,
                character_id=character_id,
                media_asset_id=media_asset_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )

        log_event(
            "runtime.character_reference_updated",
            save_id=save_id,
            character_id=character_id,
            media_asset_id=asset.id,
        )
        return self.build_model(
            status="Character reference image updated",
            active_save_id=save_id,
        )

    def upload_character_reference_image(
        self,
        *,
        image_bytes: bytes,
        filename: str | None = None,
        character_id: str | None = None,
        replace_existing: bool = False,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.character_reference_upload_failed",
                error="No save loaded",
            )
            return self.build_model(error="No save loaded", active_save_id=save_id)

        try:
            with self._thread_save_operation_lock(save_id):
                asset = self._media_service().upload_character_reference_image(
                    save_id=save_id,
                    image_bytes=image_bytes,
                    filename=filename,
                    character_id=character_id,
                    replace_existing=replace_existing,
                )
        except Exception as exc:
            log_error_event(
                "runtime.character_reference_upload_failed",
                save_id=save_id,
                character_id=character_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )

        log_event(
            "runtime.character_reference_uploaded",
            save_id=save_id,
            character_id=character_id,
            media_asset_id=asset.id,
            replaced=replace_existing,
        )
        return self.build_model(
            status=(
                "Character reference image replaced"
                if replace_existing
                else "Character reference image uploaded"
            ),
            active_save_id=save_id,
        )

    def remove_character_reference_image(
        self,
        *,
        character_id: str | None = None,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.character_reference_remove_failed",
                error="No save loaded",
            )
            return self.build_model(error="No save loaded", active_save_id=save_id)

        try:
            with self._thread_save_operation_lock(save_id):
                removed = self._media_service().remove_character_reference_image(
                    save_id=save_id,
                    character_id=character_id,
                )
        except Exception as exc:
            log_error_event(
                "runtime.character_reference_remove_failed",
                save_id=save_id,
                character_id=character_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        if removed is None:
            return self.build_model(
                error="No character reference image is set",
                active_save_id=save_id,
            )

        log_event(
            "runtime.character_reference_removed",
            save_id=save_id,
            character_id=character_id,
            media_asset_id=removed.id,
        )
        return self.build_model(
            status="Character reference image removed",
            active_save_id=save_id,
        )

    def upload_scenario_starter_reference_image(
        self,
        *,
        scenario_id: str,
        image_bytes: bytes,
        filename: str | None = None,
        starter_id: str | None = None,
        starter_name: str = "",
        replace_existing: bool = False,
    ) -> WorldDataModel:
        self._media_service().upload_scenario_starter_reference_image(
            scenario_id=scenario_id,
            image_bytes=image_bytes,
            filename=filename,
            starter_id=starter_id,
            starter_name=starter_name,
            replace_existing=replace_existing,
        )
        return WorldDataService(self.repositories).build_scenario_definition_model(
            scenario_id
        )

    def remove_scenario_starter_reference_image(
        self,
        *,
        scenario_id: str,
        starter_id: str | None = None,
        starter_name: str = "",
    ) -> WorldDataModel:
        self._media_service().remove_scenario_starter_reference_image(
            scenario_id=scenario_id,
            starter_id=starter_id,
            starter_name=starter_name,
        )
        return WorldDataService(self.repositories).build_scenario_definition_model(
            scenario_id
        )

    async def animate_media_asset(
        self,
        media_asset_id: str,
        *,
        motion_prompt: str = "",
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.image_animation_failed",
                media_asset_id=media_asset_id,
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")
        missing = _missing_image_animation_requirement(
            self.repositories,
            self.providers,
            save_id=save_id,
        )
        if missing is not None:
            log_error_event(
                "runtime.image_animation_failed",
                save_id=save_id,
                media_asset_id=media_asset_id,
                error=missing,
            )
            return self.build_model(
                error=missing,
                active_save_id=save_id,
            )
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="image_animation",
        )
        if preference is None:
            raise AssertionError("image animation requirement check failed")
        try:
            media_service = self._media_service()
            kwargs: dict[str, object] = {
                "save_id": save_id,
                "media_asset_id": media_asset_id,
                "motion_prompt": motion_prompt,
                "job_context": "manual_image_animation",
            }
            if _call_accepts_keyword(media_service.animate_image, "current_user_id"):
                kwargs["current_user_id"] = current_user_id
            await cast(Any, media_service.animate_image)(**kwargs)
        except Exception as exc:
            log_error_event(
                "runtime.image_animation_failed",
                save_id=save_id,
                media_asset_id=media_asset_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.image_animation_succeeded",
            save_id=save_id,
            media_asset_id=media_asset_id,
            provider=preference.provider,
            model=preference.model_id,
        )
        return self.build_model(status="Image animated", active_save_id=save_id)

    def delete_messages_from_here(
        self,
        *,
        message_id: str,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.message_delete_failed",
                message_id=message_id,
                error="No save loaded",
            )
            return self.build_model(error="No save loaded", active_save_id=save_id)

        try:
            with self._thread_save_operation_lock(save_id):
                self.repositories.begin_transaction()
                deletion = MessageRevisionService(
                    self.repositories
                ).delete_suffix_from_message(
                    save_id=save_id,
                    message_id=message_id,
                )
                self.repositories.commit_transaction()
        except Exception as exc:
            self.repositories.rollback_transaction()
            log_error_event(
                "runtime.message_delete_failed",
                save_id=save_id,
                message_id=message_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )

        log_event(
            "runtime.messages_deleted",
            save_id=save_id,
            message_id=message_id,
            anchor_message_id=deletion.anchor_message_id,
            deleted_count=len(deletion.deleted_messages),
        )
        return self.build_model(status="Messages deleted", active_save_id=save_id)

    def fork_save_from_message(
        self,
        *,
        message_id: str,
        active_save_id: str | None | object = ...,
        owner_user_id: str | None = None,
        remember_process_active_save: bool = True,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.save_fork_failed",
                message_id=message_id,
                error="No save loaded",
            )
            return self.build_model(error="No save loaded", active_save_id=save_id)

        try:
            with self._thread_save_operation_lock(save_id):
                result = SaveForkService(self.repositories).fork_from_message(
                    save_id=save_id,
                    message_id=message_id,
                    media_dir=self.media_dir,
                    owner_user_id=owner_user_id,
                )
        except Exception as exc:
            log_error_event(
                "runtime.save_fork_failed",
                save_id=save_id,
                message_id=message_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )

        if remember_process_active_save:
            self.active_save_id = result.save.id
        log_event(
            "runtime.save_forked",
            source_save_id=save_id,
            fork_save_id=result.save.id,
            message_id=message_id,
            message_count=result.message_count,
            media_count=result.media_count,
        )
        return self.build_model(status="Save forked", active_save_id=result.save.id)

    def preview_import_bundle(self, bundle_path: Path) -> ChatBundlePreview:
        preview = ChatBundleService(
            repositories=self.repositories,
            media_dir=self.media_dir,
        ).preview_import(bundle_path)
        log_event(
            "runtime.chat_bundle_previewed",
            bundle_path=str(bundle_path),
            title_chars=len(preview.title),
            message_count=preview.message_count,
            media_count=preview.media_count,
        )
        return preview

    def export_active_save(
        self,
        bundle_path: Path,
        *,
        include_message_revisions: bool = False,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.chat_bundle_export_failed",
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")
        try:
            with self._thread_save_operation_lock(save_id):
                manifest = ChatBundleService(
                    repositories=self.repositories,
                    media_dir=self.media_dir,
                ).export_save(
                    save_id,
                    bundle_path,
                    include_message_revisions=include_message_revisions,
                )
        except Exception as exc:
            log_error_event(
                "runtime.chat_bundle_export_failed",
                save_id=save_id,
                bundle_path=str(bundle_path),
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.chat_bundle_exported",
            save_id=save_id,
            bundle_path=str(bundle_path),
            message_count=manifest.message_count,
            media_count=manifest.media_count,
        )
        return self.build_model(
            status=f"Exported chat: {manifest.title}",
            active_save_id=save_id,
        )

    def import_save_bundle(
        self,
        bundle_path: Path,
        *,
        owner_user_id: str | None = None,
        remember_process_active_save: bool = True,
    ) -> RuntimeModel:
        try:
            active_save_id = (
                self.active_save_id if remember_process_active_save else None
            )
            def import_bundle() -> ImportedChatBundle:
                service = ChatBundleService(
                    repositories=self.repositories,
                    media_dir=self.media_dir,
                )
                if owner_user_id is None:
                    return service.import_save(bundle_path)
                return service.import_save(bundle_path, owner_user_id=owner_user_id)

            if active_save_id is None:
                imported = import_bundle()
            else:
                with self._thread_save_operation_lock(active_save_id):
                    imported = import_bundle()
        except Exception as exc:
            log_error_event(
                "runtime.chat_bundle_import_failed",
                bundle_path=str(bundle_path),
                **exception_log_fields(exc),
            )
            return self.build_model(error=_user_visible_error(exc))
        if remember_process_active_save:
            self.active_save_id = imported.save_id
        log_event(
            "runtime.chat_bundle_imported",
            save_id=imported.save_id,
            scenario_id=imported.scenario_id,
            message_count=imported.message_count,
            media_count=imported.media_count,
            skipped_media_count=imported.skipped_media_count,
        )
        return self.build_model(
            status=f"Imported chat: {imported.title}",
            active_save_id=imported.save_id,
        )

    def preview_import_scenario_bundle(
        self,
        bundle_path: Path,
    ) -> ScenarioBundlePreview:
        preview = ScenarioBundleService(
            repositories=self.repositories,
            media_dir=self.media_dir,
        ).preview_import(bundle_path)
        log_event(
            "runtime.scenario_bundle_previewed",
            bundle_path=str(bundle_path),
            title_chars=len(preview.title),
            scenario_type=preview.scenario_type,
        )
        return preview

    def export_saved_scenario(
        self,
        scenario_id: str,
        bundle_path: Path,
    ) -> RuntimeModel:
        try:
            manifest = ScenarioBundleService(
                repositories=self.repositories,
                media_dir=self.media_dir,
            ).export_scenario(scenario_id, bundle_path)
        except Exception as exc:
            log_error_event(
                "runtime.scenario_bundle_export_failed",
                scenario_id=scenario_id,
                bundle_path=str(bundle_path),
                **exception_log_fields(exc),
            )
            return self.build_model(error=_user_visible_error(exc))
        log_event(
            "runtime.scenario_bundle_exported",
            scenario_id=scenario_id,
            bundle_path=str(bundle_path),
            scenario_type=manifest.scenario_type,
        )
        return self.build_model(status=f"Exported scenario: {manifest.title}")

    def import_scenario_bundle(self, bundle_path: Path) -> RuntimeModel:
        try:
            imported = ScenarioBundleService(
                repositories=self.repositories,
                media_dir=self.media_dir,
            ).import_scenario(bundle_path)
        except Exception as exc:
            log_error_event(
                "runtime.scenario_bundle_import_failed",
                bundle_path=str(bundle_path),
                **exception_log_fields(exc),
            )
            return self.build_model(error=_user_visible_error(exc))
        log_event(
            "runtime.scenario_bundle_imported",
            scenario_id=imported.scenario_id,
            scenario_type=imported.scenario_type,
        )
        return self.build_model(status=f"Imported scenario: {imported.title}")

    def preview_import_character_bundle(
        self,
        bundle_path: Path,
        *,
        target_save_id: str | None = None,
    ) -> CharacterBundlePreview:
        preview = CharacterBundleService(
            repositories=self.repositories,
            media_dir=self.media_dir,
        ).preview_import(
            bundle_path,
            target_save_id=target_save_id or self.active_save_id,
        )
        log_event(
            "runtime.character_bundle_previewed",
            bundle_path=str(bundle_path),
            name_chars=len(preview.name),
            media_count=preview.media_count,
        )
        return preview

    def export_character_bundle(
        self,
        character_id: str,
        bundle_path: Path,
        *,
        active_save_id: str | None | object = ...,
        include_private_notes: bool = False,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.character_bundle_export_failed",
                error="No save loaded",
            )
            return self.build_model(error="No save loaded", active_save_id=save_id)
        character = self.repositories.get_character(character_id)
        if character is None or character.save_id != save_id:
            log_error_event(
                "runtime.character_bundle_export_failed",
                save_id=save_id,
                character_id=character_id,
                error="Character does not belong to the active save",
            )
            return self.build_model(
                error="Character does not belong to the active save",
                active_save_id=save_id,
            )
        try:
            with self._thread_save_operation_lock(save_id):
                manifest = CharacterBundleService(
                    repositories=self.repositories,
                    media_dir=self.media_dir,
                ).export_character(
                    character_id,
                    bundle_path,
                    include_private_notes=include_private_notes,
                )
        except Exception as exc:
            log_error_event(
                "runtime.character_bundle_export_failed",
                save_id=save_id,
                character_id=character_id,
                bundle_path=str(bundle_path),
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.character_bundle_exported",
            save_id=save_id,
            character_id=character_id,
            bundle_path=str(bundle_path),
            media_count=manifest.media_count,
        )
        return self.build_model(
            status=f"Exported character: {manifest.name}",
            active_save_id=save_id,
        )

    def import_character_bundle(
        self,
        bundle_path: Path,
        *,
        target_save_id: str | None = None,
        name: str | None = None,
        remember_process_active_save: bool = True,
    ) -> CharacterRegistryModel:
        save_id = target_save_id or (
            self.active_save_id if remember_process_active_save else None
        )
        if save_id is None:
            return CharacterRegistryModel(
                active_save_id=None,
                save=None,
                error="No save loaded",
            )
        try:
            with self._thread_save_operation_lock(save_id):
                imported = CharacterBundleService(
                    repositories=self.repositories,
                    media_dir=self.media_dir,
                ).import_character(
                    bundle_path,
                    target_save_id=save_id,
                    name=name,
                )
        except Exception as exc:
            log_error_event(
                "runtime.character_bundle_import_failed",
                save_id=save_id,
                bundle_path=str(bundle_path),
                **exception_log_fields(exc),
            )
            return CharacterRegistryModel(
                active_save_id=save_id,
                save=self.repositories.get_save(save_id),
                error=_user_visible_error(exc),
            )
        if remember_process_active_save:
            self.active_save_id = save_id
        log_event(
            "runtime.character_bundle_imported",
            save_id=save_id,
            character_id=imported.character_id,
            media_count=imported.media_count,
            skipped_media_count=imported.skipped_media_count,
        )
        return self.build_character_registry_model(active_save_id=save_id)

    async def _review_scenario_character_starters(
        self,
        *,
        starters: tuple[ScenarioCharacterStarter, ...],
        save_id: str | None,
        current_user_id: str | None,
        roleplay_type: str,
    ) -> tuple[tuple[ScenarioCharacterStarter, ...], str | None]:
        reviewed: list[ScenarioCharacterStarter] = []
        ratings: list[str] = []
        for starter in starters:
            safety = await self._review_actor_content(
                body=_scenario_character_starter_safety_body(starter),
                save_id=save_id,
                current_user_id=current_user_id,
                roleplay_type=roleplay_type,
            )
            reviewed.append(
                starter
                if safety.action is ContentSafetyAction.ALLOW
                else _scenario_character_starter_with_safe_transition(
                    starter,
                    replacement=safety.body,
                )
            )
            ratings.append(safety.reviewed_content_rating)
        return (
            tuple(reviewed),
            maximum_content_rating(tuple(ratings)) if ratings else None,
        )

    def _review_scenario_character_starters_blocking(
        self,
        *,
        starters: tuple[ScenarioCharacterStarter, ...],
        save_id: str | None,
        current_user_id: str | None,
        roleplay_type: str,
    ) -> tuple[tuple[ScenarioCharacterStarter, ...], str | None]:
        reviewed: list[ScenarioCharacterStarter] = []
        ratings: list[str] = []
        for starter in starters:
            safety = self._review_actor_content_blocking(
                body=_scenario_character_starter_safety_body(starter),
                save_id=save_id,
                current_user_id=current_user_id,
                roleplay_type=roleplay_type,
            )
            reviewed.append(
                starter
                if safety.action is ContentSafetyAction.ALLOW
                else _scenario_character_starter_with_safe_transition(
                    starter,
                    replacement=safety.body,
                )
            )
            ratings.append(safety.reviewed_content_rating)
        return (
            tuple(reviewed),
            maximum_content_rating(tuple(ratings)) if ratings else None,
        )

    def _content_with_completed_character_starters(
        self,
        *,
        scenario_type: str | ScenarioType,
        scenario_types: Iterable[str | ScenarioType] | None = None,
        content: Mapping[str, object],
        save_id: str | None = None,
    ) -> dict[str, object]:
        normalized_type, normalized_genres, _action_choices_enabled = (
            normalized_scenario_types_and_flag(
                scenario_type,
                scenario_types=scenario_types,
                action_choices_enabled=False,
            )
        )
        starter_type = _character_starter_scenario_type(
            normalized_type,
            normalized_genres,
        )
        starters = scenario_character_starters_for_content(
            scenario_type=starter_type,
            content=content,
        )
        if not starters:
            return content_with_character_starters(
                scenario_type=starter_type,
                content=content,
                starters=starters,
            )
        completer = self._character_profile_completer(starter_type)
        if completer is None:
            return content_with_character_starters(
                scenario_type=starter_type,
                content=content,
                starters=starters,
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            completed = asyncio.run(
                complete_character_starters(
                    completer=completer,
                    scenario_type=starter_type,
                    scenario_types=tuple(genre.value for genre in normalized_genres),
                    content=content,
                    save_id=save_id,
                )
            )
            return content_with_character_starters(
                scenario_type=starter_type,
                content=content,
                starters=completed,
            )
        return content_with_character_starters(
            scenario_type=starter_type,
            content=content,
            starters=starters,
        )

    def _character_profile_completer(
        self,
        scenario_type: str,
    ) -> object | None:
        preference = _context_update_preference_for_scenario_type(
            repositories=self.repositories,
            scenario_type=scenario_type,
        )
        return self._character_profile_completer_for_preference(
            preference,
            prefer_structured_output=False,
        )

    def _character_field_enhancement_completer(
        self,
        active_save_id: str,
    ) -> object | None:
        preference = character_enhancement_model_preference(
            repositories=self.repositories,
            save_id=active_save_id,
        )
        return self._character_profile_completer_for_preference(
            preference,
            prefer_structured_output=True,
        )

    def _character_profile_completer_for_preference(
        self,
        preference: ModelPreferenceRecord | None,
        *,
        prefer_structured_output: bool,
    ) -> object | None:
        if preference is None or preference.provider not in self.providers:
            return None
        provider = self.providers[preference.provider]
        structured_completer = self._structured_character_profile_completer(
            preference,
            provider,
        )
        tool_completer = self._tool_character_profile_completer(
            preference,
            provider,
        )
        if prefer_structured_output:
            return structured_completer or tool_completer
        return tool_completer or structured_completer

    def _structured_character_profile_completer(
        self,
        preference: ModelPreferenceRecord,
        provider: ProviderClient,
    ) -> StructuredProviderCharacterProfileCompleter | None:
        if not (
            _model_supports_structured_output(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
            )
            and isinstance(provider, StructuredOutputProvider)
        ):
            return None
        return StructuredProviderCharacterProfileCompleter(
            provider=provider,
            provider_name=preference.provider,
            model_id=preference.model_id,
            repositories=self.repositories,
            providers=self.providers,
        )

    def _tool_character_profile_completer(
        self,
        preference: ModelPreferenceRecord,
        provider: ProviderClient,
    ) -> object | None:
        if not (
            _model_supports_tool_calling(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
            )
            and isinstance(provider, ToolCallProvider)
        ):
            return None
        return ToolCallingProviderCharacterProfileCompleter(
            provider=provider,
            provider_name=preference.provider,
            model_id=preference.model_id,
            repositories=self.repositories,
            providers=self.providers,
        )

    def _model_with_draft(
        self,
        draft: ScenarioDraft,
        *,
        status: str,
    ) -> RuntimeModel:
        base = self.build_model(status=status)
        return RuntimeModel(
            saves=base.saves,
            active_save_id=base.active_save_id,
            active_save_title=base.active_save_title,
            custom_instructions=base.custom_instructions,
            scenario_title=base.scenario_title,
            scene_title=base.scene_title,
            chronicle=base.chronicle,
            media=base.media,
            action_choices=base.action_choices,
            action_choices_enabled=base.action_choices_enabled,
            scenario_wizard=base.scenario_wizard,
            scenario_draft=ScenarioDraftModel(
                scenario_type=draft.type.value,
                interaction_mode=draft.interaction_mode,
                sections=tuple(draft.sections.items()),
                regeneration_seed=draft.regeneration_seed,
                source_metadata=tuple((draft.metadata or {}).items()),
                action_choices_enabled=draft.action_choices_enabled,
                scenario_types=tuple(genre.value for genre in draft.scenario_types),
                character_starters=tuple(
                    scenario_character_starter_to_json(starter)
                    for starter in draft.character_starters
                ),
            ),
            model_indicator=base.model_indicator,
            status=base.status,
            error=base.error,
            interaction_mode=base.interaction_mode,
        )

    async def submit_player_message(
        self,
        *,
        body: str,
        speaker_name: str | None = None,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        narrator_stream_callback: NarratorStreamCallback | None = None,
        turn_progress_callback: TurnProgressCallback | None = None,
        post_input_catchup: Callable[[], Awaitable[None]] | None = None,
    ) -> RuntimeModel:
        return _required_turn_model(
            await self._submit_player_message(
                body=body,
                speaker_name=speaker_name,
                run_post_turn_jobs=True,
                active_save_id=active_save_id,
                current_user_id=current_user_id,
                retry_progress_callback=retry_progress_callback,
                narrator_stream_callback=narrator_stream_callback,
                turn_progress_callback=turn_progress_callback,
                post_input_catchup=post_input_catchup,
            )
        )

    async def submit_player_message_for_initial_render(
        self,
        *,
        body: str,
        speaker_name: str | None = None,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        narrator_stream_callback: NarratorStreamCallback | None = None,
        turn_progress_callback: TurnProgressCallback | None = None,
        post_input_catchup: Callable[[], Awaitable[None]] | None = None,
    ) -> SubmittedRuntimeTurn:
        return await self._submit_player_message(
            body=body,
            speaker_name=speaker_name,
            run_post_turn_jobs=False,
            active_save_id=active_save_id,
            current_user_id=current_user_id,
            retry_progress_callback=retry_progress_callback,
            narrator_stream_callback=narrator_stream_callback,
            turn_progress_callback=turn_progress_callback,
            post_input_catchup=post_input_catchup,
        )

    async def submit_timeskip(
        self,
        *,
        instruction: str,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        narrator_stream_callback: NarratorStreamCallback | None = None,
        turn_progress_callback: TurnProgressCallback | None = None,
        post_input_catchup: Callable[[], Awaitable[None]] | None = None,
    ) -> RuntimeModel:
        return _required_turn_model(
            await self._submit_timeskip(
                instruction=instruction,
                run_post_turn_jobs=True,
                active_save_id=active_save_id,
                current_user_id=current_user_id,
                retry_progress_callback=retry_progress_callback,
                narrator_stream_callback=narrator_stream_callback,
                turn_progress_callback=turn_progress_callback,
                post_input_catchup=post_input_catchup,
            )
        )

    async def submit_timeskip_for_initial_render(
        self,
        *,
        instruction: str,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        narrator_stream_callback: NarratorStreamCallback | None = None,
        turn_progress_callback: TurnProgressCallback | None = None,
        post_input_catchup: Callable[[], Awaitable[None]] | None = None,
    ) -> SubmittedRuntimeTurn:
        return await self._submit_timeskip(
            instruction=instruction,
            run_post_turn_jobs=False,
            active_save_id=active_save_id,
            current_user_id=current_user_id,
            retry_progress_callback=retry_progress_callback,
            narrator_stream_callback=narrator_stream_callback,
            turn_progress_callback=turn_progress_callback,
            post_input_catchup=post_input_catchup,
        )

    async def look_around(
        self,
        *,
        query: str,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> object:
        submitted_save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        text = query.strip()
        if not text:
            raise ValueError("Look Around query is required")
        if submitted_save_id is None:
            raise ValueError("No save loaded")
        missing = _missing_chat_requirement(
            self.repositories,
            self.providers,
            save_id=submitted_save_id,
        )
        if missing is not None:
            raise ValueError(missing)
        async with self._save_operation_lock(submitted_save_id):
            chat_service = ChatService(
                repositories=self.repositories,
                providers=self.providers,
                context_search_service=self.context_search_service,
                summary_service=self._summary_service(),
                media_service=self._media_service(),
                prompt_inspection_store=self._prompt_inspection_store_if_enabled(),
            )
            return await chat_service.look_around(
                save_id=submitted_save_id,
                query=text,
                current_user_id=current_user_id,
                retry_progress_callback=retry_progress_callback,
            )

    async def run_post_turn_jobs(
        self,
        *,
        save_id: str,
        player_message_id: str,
        narrator_message_id: str,
        turn_revision: TurnRevisionBoundary | dict[str, object] | None = None,
        progress_callback: PostTurnProgressCallback | None = None,
        current_user_id: str | None = None,
        defer_image_generation: bool = False,
    ) -> RuntimeModel:
        try:
            chat_service = ChatService(
                repositories=self.repositories,
                providers=self.providers,
                context_search_service=self.context_search_service,
                summary_service=self._summary_service(),
                media_service=self._media_service(),
                prompt_inspection_store=self._prompt_inspection_store_if_enabled(),
            )
            kwargs: dict[str, object] = {
                "save_id": save_id,
                "player_message_id": player_message_id,
                "narrator_message_id": narrator_message_id,
            }
            parameters = inspect.signature(chat_service.run_post_turn_jobs).parameters
            if turn_revision is not None and "turn_revision" in parameters:
                kwargs["turn_revision"] = turn_revision
            if progress_callback is not None and "progress_callback" in parameters:
                kwargs["progress_callback"] = progress_callback
            if "current_user_id" in parameters:
                kwargs["current_user_id"] = current_user_id
            if defer_image_generation and "defer_image_generation" in parameters:
                kwargs["defer_image_generation"] = True

            async def run_post_turn() -> dict[str, object]:
                if "world_update_context" in parameters:

                    def context_factory() -> AbstractAsyncContextManager[None]:
                        return self._save_operation_lock(save_id)

                    kwargs["world_update_context"] = context_factory
                    return cast(
                        dict[str, object],
                        await cast(Any, chat_service.run_post_turn_jobs)(**kwargs),
                    )
                async with self._save_operation_lock(save_id):
                    return cast(
                        dict[str, object],
                        await cast(Any, chat_service.run_post_turn_jobs)(**kwargs),
                    )

            coordinator_result = await run_post_turn()
            prepared = _prepared_image_from_coordinator_result(coordinator_result)
            if prepared is not None:
                if len(self._deferred_automatic_image_payloads) > 64:
                    self._deferred_automatic_image_payloads.clear()
                self._deferred_automatic_image_payloads[
                    (save_id, narrator_message_id)
                ] = prepared
        except Exception as exc:
            log_error_event(
                "runtime.post_turn_jobs_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        return self.build_model(
            status=_turn_complete_status(
                context_trimmed=narrator_message_id
                in self._context_trimmed_narrator_message_ids,
            ),
            active_save_id=save_id,
        )

    async def run_prepared_action_choices(
        self,
        *,
        prepared_action_choices: PreparedActionChoiceGeneration,
        current_user_id: str | None = None,
    ) -> str:
        try:
            records = await ActionChoiceService(
                repositories=self.repositories,
                providers=self.providers,
            ).generate_prepared(
                replace(
                    prepared_action_choices,
                    current_user_id=current_user_id,
                )
            )
        except Exception as exc:
            log_error_event(
                "runtime.action_choice_generation_failed",
                save_id=prepared_action_choices.save_id,
                narrator_message_id=prepared_action_choices.narrator_message_id,
                **exception_log_fields(exc),
            )
            return "failed"
        if records:
            return "succeeded"
        if (
            self.repositories.latest_active_message_id(
                prepared_action_choices.save_id
            )
            != prepared_action_choices.narrator_message_id
        ):
            return "obsolete"
        return "skipped"

    def consume_deferred_automatic_image(
        self,
        *,
        save_id: str,
        narrator_message_id: str,
    ) -> dict[str, object] | None:
        return self._deferred_automatic_image_payloads.pop(
            (save_id, narrator_message_id),
            None,
        )

    async def run_deferred_automatic_image(
        self,
        *,
        save_id: str,
        prepared_automatic_image: Mapping[str, object],
        current_user_id: str | None = None,
    ) -> str:
        chat_service = ChatService(
            repositories=self.repositories,
            providers=self.providers,
            context_search_service=self.context_search_service,
            summary_service=self._summary_service(),
            media_service=self._media_service(),
            prompt_inspection_store=self._prompt_inspection_store_if_enabled(),
        )
        kwargs: dict[str, object] = {
            "prepared_automatic_image": prepared_automatic_image,
        }
        parameters = inspect.signature(
            chat_service.generate_deferred_automatic_image
        ).parameters
        if current_user_id is not None and "current_user_id" in parameters:
            kwargs["current_user_id"] = current_user_id
        try:
            return cast(
                str,
                await cast(
                    Any,
                    chat_service.generate_deferred_automatic_image,
                )(**kwargs),
            )
        except Exception as exc:
            log_error_event(
                "runtime.deferred_automatic_image_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return "failed"

    async def run_state_pruning(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No active save selected")
        try:
            service = StatePruningService(
                repositories=self.repositories,
                providers=self.providers,
            )
            await service.prune(
                save_id=save_id,
                review_only=False,
                apply_guard=lambda: self._save_operation_lock(save_id),
            )
        except Exception as exc:
            log_error_event(
                "runtime.state_pruning_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        return self.build_model(
            status="World state cleanup complete.",
            active_save_id=save_id,
        )

    async def run_context_update_retries(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No active save selected")
        try:
            self._begin_maintenance_retry_drain(save_id)
            try:
                async with self._save_operation_lock(save_id):
                    completed = await ChatService(
                        repositories=self.repositories,
                        providers=self.providers,
                        context_search_service=self.context_search_service,
                        summary_service=self._summary_service(),
                        media_service=self._media_service(),
                        prompt_inspection_store=(
                            self._prompt_inspection_store_if_enabled()
                        ),
                    ).run_context_update_retries(save_id=save_id)
            finally:
                self._end_maintenance_retry_drain(save_id)
        except Exception as exc:
            log_error_event(
                "runtime.context_update_retries_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        return self.build_model(
            status=f"Context update retries finished: {completed} completed.",
            active_save_id=save_id,
        )

    async def run_observation_curation(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No active save selected")
        try:
            self._begin_maintenance_retry_drain(save_id)
            try:
                result = await ChatService(
                    repositories=self.repositories,
                    providers=self.providers,
                    context_search_service=self.context_search_service,
                    summary_service=self._summary_service(),
                    media_service=self._media_service(),
                    prompt_inspection_store=self._prompt_inspection_store_if_enabled(),
                ).run_observation_curation(
                    save_id=save_id,
                    apply_guard=lambda: self._save_operation_lock(save_id),
                )
            finally:
                self._end_maintenance_retry_drain(save_id)
        except Exception as exc:
            log_error_event(
                "runtime.observation_curation_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        curation = result.get("curation") if result is not None else None
        considered = (
            curation.get("considered_count", 0)
            if isinstance(curation, Mapping)
            else 0
        )
        return self.build_model(
            status=f"Observation curation finished: {considered} considered.",
            active_save_id=save_id,
        )

    async def run_state_extraction_retries(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No active save selected")
        try:
            self._begin_maintenance_retry_drain(save_id)
            try:
                async with self._save_operation_lock(save_id):
                    completed = await ChatService(
                        repositories=self.repositories,
                        providers=self.providers,
                        context_search_service=self.context_search_service,
                        summary_service=self._summary_service(),
                        media_service=self._media_service(),
                        prompt_inspection_store=(
                            self._prompt_inspection_store_if_enabled()
                        ),
                    ).run_state_extraction_retries(save_id=save_id)
            finally:
                self._end_maintenance_retry_drain(save_id)
        except Exception as exc:
            log_error_event(
                "runtime.state_extraction_retries_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        return self.build_model(
            status=f"State extraction retries finished: {completed} completed.",
            active_save_id=save_id,
        )

    async def run_character_text_world_update_retries(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No active save selected")
        try:
            async with self._save_operation_lock(save_id):
                completed = await CharacterTextWorldUpdateService(
                    repositories=self.repositories,
                    providers=self.providers,
                ).run_retries(save_id=save_id)
        except Exception as exc:
            log_error_event(
                "runtime.character_text_world_update_retries_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        return self.build_model(
            status=f"Text world update retries finished: {completed} completed.",
            active_save_id=save_id,
        )

    async def run_summary_backfill(
        self,
        *,
        active_save_id: str | None | object = ...,
        apply_recommended_windows: bool = False,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No save loaded")
        try:
            async with self._save_operation_lock(save_id):
                result = await SummaryBackfillService(
                    repositories=self.repositories,
                    providers=self.providers,
                ).backfill_save(
                    save_id,
                    apply_recommended_windows=apply_recommended_windows,
                )
        except Exception as exc:
            log_error_event(
                "runtime.summary_backfill_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        if result.skipped_reason:
            status = f"Summary backfill skipped: {result.skipped_reason}."
        else:
            batch_label = "batch" if result.batch_count == 1 else "batches"
            status = (
                "Summary backfill finished: "
                f"{result.summarized_message_count} messages compacted across "
                f"{result.batch_count} {batch_label}, "
                f"{result.archived_summary_count} old summaries archived."
            )
            if result.applied_window_changes:
                status += " Chat history windows updated."
            elif (
                result.recommended_player_window < result.current_player_window
                or result.recommended_narrator_window < result.current_narrator_window
            ):
                status += " Recommended chat history windows are available."
        log_event(
            "runtime.summary_backfill_succeeded",
            save_id=save_id,
            **result.to_result(),
        )
        return self.build_model(status=status, active_save_id=save_id)

    async def run_memory_consolidation(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No active save selected")
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="context_update",
        )
        if preference is None:
            return self.build_model(
                status=(
                    "Memory consolidation skipped: no context-update model "
                    "configured."
                ),
                active_save_id=save_id,
            )
        provider = self.providers.get(preference.provider)
        supports_tool_calling = (
            provider is not None
            and isinstance(provider, ToolCallProvider)
            and _model_supports_capability(
                self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                capability=ProviderCapability.TOOL_CALLING.value,
            )
        )
        supports_structured_output = (
            provider is not None
            and isinstance(provider, StructuredOutputProvider)
            and _model_supports_capability(
                self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                capability=ProviderCapability.STRUCTURED_OUTPUT.value,
            )
        )
        if not supports_tool_calling and not supports_structured_output:
            return self.build_model(
                status=(
                    "Memory consolidation skipped: context-update model does not "
                    "support structured output or tool calling."
                ),
                active_save_id=save_id,
            )
        try:
            result = await MemoryConsolidationService(
                repositories=self.repositories,
                provider=cast(
                    StructuredOutputProvider | ToolCallProvider,
                    provider,
                ),
                provider_name=preference.provider,
                model_id=preference.model_id,
                providers=self.providers,
                prompt_inspection_store=self._prompt_inspection_store_if_enabled(),
                prefer_tool_calls=supports_tool_calling,
            ).consolidate_if_needed(
                save_id,
                apply_guard=lambda: self._save_operation_lock(save_id),
            )
        except Exception as exc:
            log_error_event(
                "runtime.memory_consolidation_failed",
                save_id=save_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        if result.skipped_reason:
            status = f"Memory consolidation skipped: {result.skipped_reason}."
        else:
            status = (
                "Memory consolidation finished: "
                f"{result.rewritten_count} rewritten, "
                f"{result.archived_count} archived, "
                f"{result.rejected_count} rejected."
            )
        return self.build_model(status=status, active_save_id=save_id)

    async def run_character_registry_maintenance(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No active save selected")
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="character_registry_maintenance",
        )
        if preference is None:
            return self.build_model(
                status="Character maintenance skipped: no model configured.",
                active_save_id=save_id,
            )
        try:
            async with self._save_operation_lock(save_id):
                result = await CharacterRegistryMaintenanceService(
                    repositories=self.repositories,
                    providers=self.providers,
                ).maintain_if_due(save_id=save_id, force=False)
        except Exception as exc:
            log_error_event(
                "runtime.character_maintenance_failed",
                save_id=save_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.character_maintenance_succeeded",
            save_id=save_id,
            provider=preference.provider,
            model=preference.model_id,
            proposed_count=len(result.proposed),
            applied_count=len(result.applied),
            rejected_count=len(result.rejected),
            skipped_reason=result.skipped_reason,
        )
        return self.build_model(
            status=_manual_character_maintenance_status(result).strip(),
            active_save_id=save_id,
        )

    async def run_world_context_retention(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No active save selected")
        try:
            async with self._save_operation_lock(save_id):
                result = WorldContextRetentionService(
                    repositories=self.repositories,
                ).prune(save_id)
        except Exception as exc:
            log_error_event(
                "runtime.world_context_retention_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        pruned_archived = sum(result.pruned_archived_rows.values())
        status = (
            "World context retention finished: "
            f"{result.expired_stale_suggestions + result.expired_excess_suggestions} "
            "suggestions expired, "
            f"{pruned_archived} archived rows pruned, "
            f"{result.pruned_terminal_jobs} terminal jobs pruned."
        )
        return self.build_model(status=status, active_save_id=save_id)

    async def run_context_cleanup(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event("runtime.context_cleanup_failed", error="No save loaded")
            return self.build_model(error="No save loaded")
        missing = _missing_context_cleanup_requirement(
            self.repositories,
            self.providers,
            save_id=save_id,
            purposes=(CONTEXT_CLEANUP_SCAN_TASK, CONTEXT_CLEANUP_ACTIONS_TASK),
        )
        if missing is not None:
            log_error_event(
                "runtime.context_cleanup_failed",
                save_id=save_id,
                error=missing,
            )
            return self.build_model(error=missing, active_save_id=save_id)
        task_preferences = _context_cleanup_preferences(
            repositories=self.repositories,
            save_id=save_id,
            purposes=(CONTEXT_CLEANUP_SCAN_TASK, CONTEXT_CLEANUP_ACTIONS_TASK),
        )
        preference = task_preferences.get(CONTEXT_CLEANUP_ACTIONS_TASK)
        if preference is None:
            raise AssertionError("context cleanup requirement check failed")
        provider = self.providers[preference.provider]
        prefer_tool_call_tasks = _context_cleanup_tool_call_tasks(
            self.repositories,
            self.providers,
            task_preferences,
        )
        try:
            result = await ContextCleanupService(
                repositories=self.repositories,
                provider=cast(StructuredOutputProvider | ToolCallProvider, provider),
                provider_name=preference.provider,
                model_id=preference.model_id,
                providers=self.providers,
                task_model_preferences=task_preferences,
                prefer_tool_call_tasks=prefer_tool_call_tasks,
            ).analyze_and_apply(
                save_id,
                apply_guard=lambda: self._save_operation_lock(save_id),
            )
        except Exception as exc:
            log_error_event(
                "runtime.context_cleanup_failed",
                save_id=save_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )

        await self._run_manual_memory_consolidation(save_id, preference=preference)
        maintenance_status = await self._run_manual_character_maintenance(save_id)
        status = (
            "Context cleanup finished: "
            f"{result.applied_actions} changes applied, "
            f"{result.rejected_actions} rejected."
            f"{maintenance_status} Details in World data audit."
        )
        log_event(
            "runtime.context_cleanup_succeeded",
            save_id=save_id,
            provider=preference.provider,
            model=preference.model_id,
            applied_actions=result.applied_actions,
            rejected_actions=result.rejected_actions,
        )
        return self.build_model(status=status, active_save_id=save_id)

    async def _run_manual_memory_consolidation(
        self,
        save_id: str,
        *,
        preference: ModelPreferenceRecord,
    ) -> None:
        provider = self.providers.get(preference.provider)
        supports_tool_calling = (
            provider is not None
            and isinstance(provider, ToolCallProvider)
            and _model_supports_capability(
                self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                capability=ProviderCapability.TOOL_CALLING.value,
            )
        )
        supports_structured_output = (
            provider is not None
            and isinstance(provider, StructuredOutputProvider)
            and _model_supports_capability(
                self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                capability=ProviderCapability.STRUCTURED_OUTPUT.value,
            )
        )
        if not supports_tool_calling and not supports_structured_output:
            return
        try:
            await MemoryConsolidationService(
                repositories=self.repositories,
                provider=cast(StructuredOutputProvider | ToolCallProvider, provider),
                provider_name=preference.provider,
                model_id=preference.model_id,
                providers=self.providers,
                prefer_tool_calls=supports_tool_calling,
            ).consolidate_if_needed(save_id)
        except Exception as exc:
            log_error_event(
                "runtime.memory_consolidation_failed",
                save_id=save_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )

    async def run_guided_context_cleanup(
        self,
        *,
        instruction: str,
        active_save_id: str | None | object = ...,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.guided_context_cleanup_failed",
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")
        if not instruction.strip():
            return self.build_model(
                error="Cleanup instructions are required",
                active_save_id=save_id,
            )
        missing = _missing_context_cleanup_requirement(
            self.repositories,
            self.providers,
            save_id=save_id,
            purposes=(GUIDED_CONTEXT_CLEANUP_TASK,),
        )
        if missing is not None:
            log_error_event(
                "runtime.guided_context_cleanup_failed",
                save_id=save_id,
                error=missing,
            )
            return self.build_model(error=missing, active_save_id=save_id)
        task_preferences = _context_cleanup_preferences(
            repositories=self.repositories,
            save_id=save_id,
            purposes=(GUIDED_CONTEXT_CLEANUP_TASK,),
        )
        preference = task_preferences.get(GUIDED_CONTEXT_CLEANUP_TASK)
        if preference is None:
            raise AssertionError("context cleanup requirement check failed")
        provider = self.providers[preference.provider]
        prefer_tool_call_tasks = _context_cleanup_tool_call_tasks(
            self.repositories,
            self.providers,
            task_preferences,
        )
        try:
            result = await ContextCleanupService(
                repositories=self.repositories,
                provider=cast(StructuredOutputProvider | ToolCallProvider, provider),
                provider_name=preference.provider,
                model_id=preference.model_id,
                providers=self.providers,
                task_model_preferences=task_preferences,
                prefer_tool_call_tasks=prefer_tool_call_tasks,
            ).propose_guided_cleanup(
                save_id,
                instruction=instruction,
                apply_guard=lambda: self._save_operation_lock(save_id),
            )
        except Exception as exc:
            log_error_event(
                "runtime.guided_context_cleanup_failed",
                save_id=save_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )

        status = (
            "Guided cleanup queued: "
            f"{result.queued_suggestions} suggestions ready for review, "
            f"{result.rejected_actions} rejected. "
            "They will be reviewed automatically."
        )
        log_event(
            "runtime.guided_context_cleanup_succeeded",
            save_id=save_id,
            provider=preference.provider,
            model=preference.model_id,
            queued_suggestions=result.queued_suggestions,
            rejected_actions=result.rejected_actions,
        )
        return self.build_model(status=status, active_save_id=save_id)

    async def run_world_suggestion_review(
        self,
        *,
        active_save_id: str | None | object = ...,
        scheduled: bool = False,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            return self.build_model(error="No save loaded")
        if self.repositories.get_save(save_id) is None:
            return self.build_model(error="No save loaded", active_save_id=save_id)
        pending = [
            suggestion
            for suggestion in self.repositories.list_context_update_suggestions(save_id)
            if suggestion.status == "pending"
        ]
        if not pending:
            return self.build_model(
                status="World suggestion review skipped: no pending suggestions.",
                active_save_id=save_id,
            )
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="context_update",
        )
        if preference is None:
            return self.build_model(
                status=(
                    "World suggestion review skipped: "
                    "no context update model configured."
                ),
                active_save_id=save_id,
            )
        provider = self.providers.get(preference.provider)
        if provider is None:
            return self.build_model(
                status=(
                    "World suggestion review skipped: "
                    f"provider is unavailable: {preference.provider}."
                ),
                active_save_id=save_id,
            )
        supports_tool_calling = isinstance(
            provider,
            ToolCallProvider,
        ) and _model_supports_capability(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            capability=ProviderCapability.TOOL_CALLING.value,
        )
        supports_structured_output = isinstance(
            provider,
            StructuredOutputProvider,
        ) and _model_supports_capability(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            capability=ProviderCapability.STRUCTURED_OUTPUT.value,
        )
        if not supports_tool_calling and not supports_structured_output:
            return self.build_model(
                status=(
                    "World suggestion review skipped: context update model does "
                    "not advertise structured output or tool calling."
                ),
                active_save_id=save_id,
            )
        try:
            async with self._save_operation_lock(save_id):
                result = await WorldSuggestionReviewService(
                    repositories=self.repositories,
                    provider=cast(
                        StructuredOutputProvider | ToolCallProvider,
                        provider,
                    ),
                    provider_name=preference.provider,
                    model_id=preference.model_id,
                    providers=self.providers,
                    prefer_tool_calls=supports_tool_calling,
                ).review_pending(save_id, due_only=scheduled)
        except Exception as exc:
            log_error_event(
                "runtime.world_suggestion_review_failed",
                save_id=save_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )

        if result.error:
            status = (
                "World suggestion review deferred: "
                f"{result.deferred_count} suggestions will be retried."
            )
        else:
            status = (
                "World suggestion review finished: "
                f"{result.applied_count} applied, "
                f"{result.rejected_count} rejected, "
                f"{result.deferred_count} deferred."
            )
        log_event(
            "runtime.world_suggestion_review_succeeded",
            save_id=save_id,
            provider=preference.provider,
            model=preference.model_id,
            reviewed_count=result.reviewed_count,
            applied_count=result.applied_count,
            rejected_count=result.rejected_count,
            deferred_count=result.deferred_count,
        )
        return self.build_model(status=status, active_save_id=save_id)

    async def _run_manual_character_maintenance(self, save_id: str) -> str:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="character_registry_maintenance",
        )
        if preference is None:
            return " Character maintenance skipped: no model configured."
        try:
            result = await CharacterRegistryMaintenanceService(
                repositories=self.repositories,
                providers=self.providers,
            ).maintain_if_due(save_id=save_id, force=True)
        except Exception as exc:
            log_error_event(
                "runtime.character_maintenance_failed",
                save_id=save_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return " Character maintenance failed."
        log_event(
            "runtime.character_maintenance_succeeded",
            save_id=save_id,
            provider=preference.provider,
            model=preference.model_id,
            proposed_count=len(result.proposed),
            applied_count=len(result.applied),
            rejected_count=len(result.rejected),
            skipped_reason=result.skipped_reason,
        )
        return _manual_character_maintenance_status(result)

    async def _submit_player_message(
        self,
        *,
        body: str,
        speaker_name: str | None,
        run_post_turn_jobs: bool,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        narrator_stream_callback: NarratorStreamCallback | None = None,
        turn_progress_callback: TurnProgressCallback | None = None,
        post_input_catchup: Callable[[], Awaitable[None]] | None = None,
    ) -> SubmittedRuntimeTurn:
        submitted_save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        try:
            text = body.strip()
            if not text:
                log_error_event("runtime.chat_turn_failed", error="Message is empty")
                return SubmittedRuntimeTurn(
                    model=self.build_model(
                        error="Message is empty",
                        active_save_id=submitted_save_id,
                    ),
                    save_id=submitted_save_id,
                )
            if submitted_save_id is None:
                log_error_event("runtime.chat_turn_failed", error="No save loaded")
                return SubmittedRuntimeTurn(
                    model=self.build_model(error="No save loaded")
                )
            missing = _missing_chat_requirement(
                self.repositories,
                self.providers,
                save_id=submitted_save_id,
            )
            if missing is not None:
                log_error_event("runtime.chat_turn_failed", error=missing)
                return SubmittedRuntimeTurn(
                    model=self.build_model(
                        error=missing,
                        active_save_id=submitted_save_id,
                    ),
                    save_id=submitted_save_id,
                )

            self._queued_chat_submissions.add(submitted_save_id)
            if self._defer_submit_lock_until_after_input(
                save_id=submitted_save_id,
                post_input_catchup=post_input_catchup,
            ):
                return await self._submit_player_message_unlocked(
                    body=text,
                    speaker_name=speaker_name,
                    run_post_turn_jobs=run_post_turn_jobs,
                    submitted_save_id=submitted_save_id,
                    current_user_id=current_user_id,
                    retry_progress_callback=retry_progress_callback,
                    narrator_stream_callback=narrator_stream_callback,
                    turn_progress_callback=turn_progress_callback,
                    post_input_context=lambda: self._post_input_submit_context(
                        save_id=submitted_save_id,
                        post_input_catchup=post_input_catchup,
                    ),
                )
            async with self._save_operation_lock(submitted_save_id):
                return await self._submit_player_message_unlocked(
                    body=text,
                    speaker_name=speaker_name,
                    run_post_turn_jobs=run_post_turn_jobs,
                    submitted_save_id=submitted_save_id,
                    current_user_id=current_user_id,
                    retry_progress_callback=retry_progress_callback,
                    narrator_stream_callback=narrator_stream_callback,
                    turn_progress_callback=turn_progress_callback,
                    post_input_context=None,
                )
        finally:
            if submitted_save_id is not None:
                self._queued_chat_submissions.discard(submitted_save_id)
                self._pending_chat_cancellations.discard(submitted_save_id)

    async def _submit_timeskip(
        self,
        *,
        instruction: str,
        run_post_turn_jobs: bool,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        narrator_stream_callback: NarratorStreamCallback | None = None,
        turn_progress_callback: TurnProgressCallback | None = None,
        post_input_catchup: Callable[[], Awaitable[None]] | None = None,
    ) -> SubmittedRuntimeTurn:
        submitted_save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        try:
            text = instruction.strip()
            if not text:
                log_error_event(
                    "runtime.chat_turn_failed",
                    error="Timeskip instruction is required",
                )
                return SubmittedRuntimeTurn(
                    model=self.build_model(
                        error="Timeskip instruction is required",
                        active_save_id=submitted_save_id,
                    ),
                    save_id=submitted_save_id,
                )
            if submitted_save_id is None:
                log_error_event("runtime.chat_turn_failed", error="No save loaded")
                return SubmittedRuntimeTurn(
                    model=self.build_model(error="No save loaded")
                )
            missing = _missing_chat_requirement(
                self.repositories,
                self.providers,
                save_id=submitted_save_id,
            )
            if missing is not None:
                log_error_event("runtime.chat_turn_failed", error=missing)
                return SubmittedRuntimeTurn(
                    model=self.build_model(
                        error=missing,
                        active_save_id=submitted_save_id,
                    ),
                    save_id=submitted_save_id,
                )

            self._queued_chat_submissions.add(submitted_save_id)
            if self._defer_submit_lock_until_after_input(
                save_id=submitted_save_id,
                post_input_catchup=post_input_catchup,
            ):
                return await self._submit_timeskip_unlocked(
                    instruction=text,
                    run_post_turn_jobs=run_post_turn_jobs,
                    submitted_save_id=submitted_save_id,
                    current_user_id=current_user_id,
                    retry_progress_callback=retry_progress_callback,
                    narrator_stream_callback=narrator_stream_callback,
                    turn_progress_callback=turn_progress_callback,
                    post_input_context=lambda: self._post_input_submit_context(
                        save_id=submitted_save_id,
                        post_input_catchup=post_input_catchup,
                    ),
                )
            async with self._save_operation_lock(submitted_save_id):
                return await self._submit_timeskip_unlocked(
                    instruction=text,
                    run_post_turn_jobs=run_post_turn_jobs,
                    submitted_save_id=submitted_save_id,
                    current_user_id=current_user_id,
                    retry_progress_callback=retry_progress_callback,
                    narrator_stream_callback=narrator_stream_callback,
                    turn_progress_callback=turn_progress_callback,
                    post_input_context=None,
                )
        finally:
            if submitted_save_id is not None:
                self._queued_chat_submissions.discard(submitted_save_id)
                self._pending_chat_cancellations.discard(submitted_save_id)

    async def _submit_player_message_unlocked(
        self,
        *,
        body: str,
        speaker_name: str | None,
        run_post_turn_jobs: bool,
        submitted_save_id: str,
        current_user_id: str | None,
        retry_progress_callback: ProviderRetryProgressCallback | None,
        narrator_stream_callback: NarratorStreamCallback | None,
        turn_progress_callback: TurnProgressCallback | None,
        post_input_context: Callable[[], AbstractAsyncContextManager[None]] | None,
    ) -> SubmittedRuntimeTurn:
        cancellation_token = CancellationToken()
        self._active_chat_cancellations[submitted_save_id] = cancellation_token
        self._queued_chat_submissions.discard(submitted_save_id)
        try:
            if submitted_save_id in self._pending_chat_cancellations:
                self._pending_chat_cancellations.remove(submitted_save_id)
                cancellation_token.cancel()
            message_marker = self.repositories.latest_active_message_rowid(
                submitted_save_id
            )
            display_speaker_name = _player_speaker_name(
                repositories=self.repositories,
                save_id=submitted_save_id,
                requested_name=speaker_name,
            )

            try:
                chat_service = ChatService(
                    repositories=self.repositories,
                    providers=self.providers,
                    context_search_service=self.context_search_service,
                    summary_service=self._summary_service(),
                    media_service=self._media_service(),
                    prompt_inspection_store=self._prompt_inspection_store_if_enabled(),
                )
                submitted_turn = await _submit_player_turn_with_optional_cancellation(
                    chat_service.submit_player_turn,
                    save_id=submitted_save_id,
                    body=body,
                    speaker_name=display_speaker_name,
                    run_post_turn_jobs=run_post_turn_jobs,
                    defer_action_choices=not run_post_turn_jobs,
                    current_user_id=current_user_id,
                    cancellation_token=cancellation_token,
                    retry_progress_callback=retry_progress_callback,
                    narrator_stream_callback=narrator_stream_callback,
                    turn_progress_callback=turn_progress_callback,
                    post_input_context=post_input_context,
                )
            except ChatTurnCancelled:
                log_event(
                    "runtime.chat_turn_cancelled",
                    save_id=submitted_save_id,
                )
                player_message_id = _committed_source_message_id(
                    repositories=self.repositories,
                    save_id=submitted_save_id,
                    after_rowid=message_marker,
                    role="player",
                    body=body,
                    speaker_name=display_speaker_name,
                )
                return SubmittedRuntimeTurn(
                    model=self.build_model(
                        error=CHAT_TURN_CANCELLED_ERROR,
                        active_save_id=submitted_save_id,
                    ),
                    save_id=submitted_save_id,
                    player_message_id=player_message_id,
                )
            except asyncio.CancelledError:
                cancellation_token.cancel()
                log_event(
                    "runtime.chat_turn_cancelled",
                    save_id=submitted_save_id,
                )
                player_message_id = _committed_source_message_id(
                    repositories=self.repositories,
                    save_id=submitted_save_id,
                    after_rowid=message_marker,
                    role="player",
                    body=body,
                    speaker_name=display_speaker_name,
                )
                return SubmittedRuntimeTurn(
                    model=self.build_model(
                        error=CHAT_TURN_CANCELLED_ERROR,
                        active_save_id=submitted_save_id,
                    ),
                    save_id=submitted_save_id,
                    player_message_id=player_message_id,
                )
            except Exception as exc:
                log_error_event(
                    "runtime.chat_turn_failed",
                    save_id=submitted_save_id,
                    **exception_log_fields(exc),
                )
                player_message_id = _committed_source_message_id(
                    repositories=self.repositories,
                    save_id=submitted_save_id,
                    after_rowid=message_marker,
                    role="player",
                    body=body,
                    speaker_name=display_speaker_name,
                )
                return SubmittedRuntimeTurn(
                    model=self.build_model(
                        error=_user_visible_error(exc),
                        active_save_id=submitted_save_id,
                    ),
                    save_id=submitted_save_id,
                    player_message_id=player_message_id,
                )
            log_event(
                "runtime.chat_turn_succeeded",
                save_id=submitted_save_id,
                body_chars=len(body),
            )
            fallback_used = bool(
                getattr(submitted_turn, "fallback_used", False)
            ) or _chat_completion_used_fallback(
                repositories=self.repositories,
                narrator_message_id=submitted_turn.narrator_message.id,
            )
            context_trimmed = bool(getattr(submitted_turn, "context_trimmed", False))
            if context_trimmed:
                self._context_trimmed_narrator_message_ids.add(
                    submitted_turn.narrator_message.id
                )
            status = _turn_complete_status(
                fallback_used=fallback_used,
                context_trimmed=context_trimmed,
            )
            model = (
                self.build_model(
                    status=status,
                    active_save_id=submitted_save_id,
                )
                if run_post_turn_jobs
                else None
            )
            delta = (
                None
                if run_post_turn_jobs
                else self.build_chat_turn_delta(
                    save_id=submitted_save_id,
                    player_message=submitted_turn.player_message,
                    narrator_message=submitted_turn.narrator_message,
                    status=status,
                    fallback_used=fallback_used,
                    context_trimmed=context_trimmed,
                )
            )
            return SubmittedRuntimeTurn(
                model=model,
                save_id=submitted_save_id,
                player_message_id=submitted_turn.player_message.id,
                narrator_message_id=submitted_turn.narrator_message.id,
                turn_revision=getattr(submitted_turn, "turn_revision", None),
                context_trimmed=context_trimmed,
                prepared_action_choices=getattr(
                    submitted_turn,
                    "prepared_action_choices",
                    None,
                ),
                delta=delta,
            )
        finally:
            self._active_chat_cancellations.pop(submitted_save_id, None)

    async def _submit_timeskip_unlocked(
        self,
        *,
        instruction: str,
        run_post_turn_jobs: bool,
        submitted_save_id: str,
        current_user_id: str | None,
        retry_progress_callback: ProviderRetryProgressCallback | None,
        narrator_stream_callback: NarratorStreamCallback | None,
        turn_progress_callback: TurnProgressCallback | None,
        post_input_context: Callable[[], AbstractAsyncContextManager[None]] | None,
    ) -> SubmittedRuntimeTurn:
        cancellation_token = CancellationToken()
        self._active_chat_cancellations[submitted_save_id] = cancellation_token
        self._queued_chat_submissions.discard(submitted_save_id)
        timeskip_body = _timeskip_body(instruction)
        try:
            if submitted_save_id in self._pending_chat_cancellations:
                self._pending_chat_cancellations.remove(submitted_save_id)
                cancellation_token.cancel()
            message_marker = self.repositories.latest_active_message_rowid(
                submitted_save_id
            )

            try:
                chat_service = ChatService(
                    repositories=self.repositories,
                    providers=self.providers,
                    context_search_service=self.context_search_service,
                    summary_service=self._summary_service(),
                    media_service=self._media_service(),
                    prompt_inspection_store=self._prompt_inspection_store_if_enabled(),
                )
                submitted_turn = await _submit_timeskip_turn_with_optional_cancellation(
                    chat_service.submit_timeskip_turn,
                    save_id=submitted_save_id,
                    instruction=instruction,
                    run_post_turn_jobs=run_post_turn_jobs,
                    defer_action_choices=not run_post_turn_jobs,
                    current_user_id=current_user_id,
                    cancellation_token=cancellation_token,
                    retry_progress_callback=retry_progress_callback,
                    narrator_stream_callback=narrator_stream_callback,
                    turn_progress_callback=turn_progress_callback,
                    post_input_context=post_input_context,
                )
            except ChatTurnCancelled:
                log_event(
                    "runtime.chat_turn_cancelled",
                    save_id=submitted_save_id,
                )
                source_message_id = _committed_source_message_id(
                    repositories=self.repositories,
                    save_id=submitted_save_id,
                    after_rowid=message_marker,
                    role="system",
                    body=timeskip_body,
                    speaker_name=TIMESKIP_SPEAKER_NAME,
                )
                return SubmittedRuntimeTurn(
                    model=self.build_model(
                        error=CHAT_TURN_CANCELLED_ERROR,
                        active_save_id=submitted_save_id,
                    ),
                    save_id=submitted_save_id,
                    player_message_id=source_message_id,
                )
            except asyncio.CancelledError:
                cancellation_token.cancel()
                log_event(
                    "runtime.chat_turn_cancelled",
                    save_id=submitted_save_id,
                )
                source_message_id = _committed_source_message_id(
                    repositories=self.repositories,
                    save_id=submitted_save_id,
                    after_rowid=message_marker,
                    role="system",
                    body=timeskip_body,
                    speaker_name=TIMESKIP_SPEAKER_NAME,
                )
                return SubmittedRuntimeTurn(
                    model=self.build_model(
                        error=CHAT_TURN_CANCELLED_ERROR,
                        active_save_id=submitted_save_id,
                    ),
                    save_id=submitted_save_id,
                    player_message_id=source_message_id,
                )
            except Exception as exc:
                log_error_event(
                    "runtime.chat_turn_failed",
                    save_id=submitted_save_id,
                    **exception_log_fields(exc),
                )
                source_message_id = _committed_source_message_id(
                    repositories=self.repositories,
                    save_id=submitted_save_id,
                    after_rowid=message_marker,
                    role="system",
                    body=timeskip_body,
                    speaker_name=TIMESKIP_SPEAKER_NAME,
                )
                return SubmittedRuntimeTurn(
                    model=self.build_model(
                        error=_user_visible_error(exc),
                        active_save_id=submitted_save_id,
                    ),
                    save_id=submitted_save_id,
                    player_message_id=source_message_id,
                )
            log_event(
                "runtime.chat_turn_succeeded",
                save_id=submitted_save_id,
                body_chars=len(instruction),
                turn_type="timeskip",
            )
            fallback_used = bool(
                getattr(submitted_turn, "fallback_used", False)
            ) or _chat_completion_used_fallback(
                repositories=self.repositories,
                narrator_message_id=submitted_turn.narrator_message.id,
            )
            context_trimmed = bool(getattr(submitted_turn, "context_trimmed", False))
            if context_trimmed:
                self._context_trimmed_narrator_message_ids.add(
                    submitted_turn.narrator_message.id
                )
            status = _turn_complete_status(
                fallback_used=fallback_used,
                context_trimmed=context_trimmed,
            )
            model = (
                self.build_model(
                    status=status,
                    active_save_id=submitted_save_id,
                )
                if run_post_turn_jobs
                else None
            )
            delta = (
                None
                if run_post_turn_jobs
                else self.build_chat_turn_delta(
                    save_id=submitted_save_id,
                    player_message=submitted_turn.player_message,
                    narrator_message=submitted_turn.narrator_message,
                    status=status,
                    fallback_used=fallback_used,
                    context_trimmed=context_trimmed,
                )
            )
            return SubmittedRuntimeTurn(
                model=model,
                save_id=submitted_save_id,
                player_message_id=submitted_turn.player_message.id,
                narrator_message_id=submitted_turn.narrator_message.id,
                turn_revision=getattr(submitted_turn, "turn_revision", None),
                context_trimmed=context_trimmed,
                prepared_action_choices=getattr(
                    submitted_turn,
                    "prepared_action_choices",
                    None,
                ),
                delta=delta,
            )
        finally:
            self._active_chat_cancellations.pop(submitted_save_id, None)

    async def regenerate_message(
        self,
        *,
        message_id: str,
        active_save_id: str | None | object = ...,
        regeneration_feedback: str = "",
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> RuntimeModel:
        turn = await self.regenerate_message_for_initial_render(
            message_id=message_id,
            active_save_id=active_save_id,
            regeneration_feedback=regeneration_feedback,
            current_user_id=current_user_id,
            retry_progress_callback=retry_progress_callback,
        )
        if turn.has_post_turn_jobs:
            return await self.run_post_turn_jobs(
                save_id=cast(str, turn.save_id),
                player_message_id=cast(str, turn.player_message_id),
                narrator_message_id=cast(str, turn.narrator_message_id),
                current_user_id=current_user_id,
            )
        return _required_turn_model(turn)

    async def regenerate_message_for_initial_render(
        self,
        *,
        message_id: str,
        active_save_id: str | None | object = ...,
        regeneration_feedback: str = "",
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> SubmittedRuntimeTurn:
        return await self._revise_message(
            message_id=message_id,
            body=None,
            active_save_id=active_save_id,
            status="Message regenerated",
            action_name="regenerate",
            regeneration_feedback=regeneration_feedback.strip(),
            current_user_id=current_user_id,
            retry_progress_callback=retry_progress_callback,
        )

    async def edit_and_resubmit_message(
        self,
        *,
        message_id: str,
        body: str,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        on_revision_committed: Callable[[RuntimeModel], None] | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> RuntimeModel:
        turn = await self.edit_and_resubmit_message_for_initial_render(
            message_id=message_id,
            body=body,
            active_save_id=active_save_id,
            current_user_id=current_user_id,
            on_revision_committed=on_revision_committed,
            retry_progress_callback=retry_progress_callback,
        )
        if turn.has_post_turn_jobs:
            return await self.run_post_turn_jobs(
                save_id=cast(str, turn.save_id),
                player_message_id=cast(str, turn.player_message_id),
                narrator_message_id=cast(str, turn.narrator_message_id),
                current_user_id=current_user_id,
            )
        return _required_turn_model(turn)

    async def edit_and_resubmit_message_for_initial_render(
        self,
        *,
        message_id: str,
        body: str,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        on_revision_committed: Callable[[RuntimeModel], None] | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> SubmittedRuntimeTurn:
        return await self._revise_message(
            message_id=message_id,
            body=body,
            active_save_id=active_save_id,
            status="Edited message resubmitted",
            action_name="edit_resubmit",
            current_user_id=current_user_id,
            on_revision_committed=on_revision_committed,
            retry_progress_callback=retry_progress_callback,
        )

    async def edit_narrator_message(
        self,
        *,
        message_id: str,
        body: str,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        on_revision_committed: Callable[[RuntimeModel], None] | None = None,
    ) -> RuntimeModel:
        return await self._edit_message_without_resubmit(
            message_id=message_id,
            body=body,
            active_save_id=active_save_id,
            current_user_id=current_user_id,
            on_revision_committed=on_revision_committed,
            service_method_name="edit_narrator_message",
            event_name="narrator_message_edit",
            success_status="Narrator message edited",
            failed_reconciliation_status=(
                "Narrator message edited; reconciliation failed"
            ),
        )

    async def edit_message_without_resubmit(
        self,
        *,
        message_id: str,
        body: str,
        active_save_id: str | None | object = ...,
        current_user_id: str | None = None,
        on_revision_committed: Callable[[RuntimeModel], None] | None = None,
    ) -> RuntimeModel:
        return await self._edit_message_without_resubmit(
            message_id=message_id,
            body=body,
            active_save_id=active_save_id,
            current_user_id=current_user_id,
            on_revision_committed=on_revision_committed,
            service_method_name="edit_message_without_resubmit",
            event_name="message_edit",
            success_status="Message edited",
            failed_reconciliation_status="Message edited; reconciliation failed",
        )

    async def _edit_message_without_resubmit(
        self,
        *,
        message_id: str,
        body: str,
        active_save_id: str | None | object,
        current_user_id: str | None,
        on_revision_committed: Callable[[RuntimeModel], None] | None,
        service_method_name: str,
        event_name: str,
        success_status: str,
        failed_reconciliation_status: str,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                f"runtime.{event_name}_failed",
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")

        try:
            async with self._save_operation_lock(save_id):
                reviewed_body = body
                content_rating = "unclassified"
                safety_transition = ""
                existing_message = self.repositories.get_message(
                    save_id=save_id,
                    message_id=message_id,
                )
                if existing_message is not None:
                    safety = await self._review_actor_content(
                        body=body,
                        save_id=save_id,
                        current_user_id=current_user_id,
                    )
                    if existing_message.role == "narrator":
                        reviewed_body = safety.body
                        content_rating = safety.reviewed_content_rating
                        safety_transition = _content_safety_transition(safety)
                    else:
                        content_rating = (
                            safety.minimum_rating
                            if safety.action is ContentSafetyAction.ALLOW
                            else CONTENT_RATING_PROHIBITED
                        )
                self.repositories.begin_transaction()
                revision_service = MessageRevisionService(self.repositories)
                edit = cast(Any, getattr(revision_service, service_method_name))(
                    save_id=save_id,
                    message_id=message_id,
                    body=reviewed_body,
                    current_user_id=current_user_id,
                    content_rating=content_rating,
                    safety_transition=safety_transition,
                )
                self.repositories.commit_transaction()

                if on_revision_committed is not None:
                    try:
                        on_revision_committed(
                            self.build_model(
                                status="Reconciling message edit...",
                                active_save_id=save_id,
                            )
                        )
                    except Exception as exc:
                        log_error_event(
                            f"runtime.{event_name}_callback_failed",
                            save_id=save_id,
                            message_id=message_id,
                            **exception_log_fields(exc),
                        )

                result = await MessageReconciliationService(
                    repositories=self.repositories,
                    providers=self.providers,
                ).reconcile_revision(revision=edit.revision)
        except Exception as exc:
            self.repositories.rollback_transaction()
            log_error_event(
                f"runtime.{event_name}_failed",
                save_id=save_id,
                message_id=message_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )

        status = success_status
        error: str | None = None
        if result.status == "failed":
            status = failed_reconciliation_status
            error = result.error
        log_event(
            f"runtime.{event_name}_succeeded",
            save_id=save_id,
            message_id=message_id,
            revision_id=edit.revision.id,
            reconciliation_status=result.status,
        )
        return self.build_model(
            status=status,
            error=error,
            active_save_id=save_id,
        )

    async def _revise_message(
        self,
        *,
        message_id: str,
        body: str | None,
        active_save_id: str | None | object,
        status: str,
        action_name: str,
        regeneration_feedback: str = "",
        current_user_id: str | None = None,
        on_revision_committed: Callable[[RuntimeModel], None] | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> SubmittedRuntimeTurn:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event("runtime.message_revision_failed", error="No save loaded")
            return SubmittedRuntimeTurn(model=self.build_model(error="No save loaded"))
        missing = _missing_chat_requirement(
            self.repositories,
            self.providers,
            save_id=save_id,
        )
        if missing is not None:
            log_error_event(
                "runtime.message_revision_failed",
                save_id=save_id,
                message_id=message_id,
                action=action_name,
                error=missing,
            )
            return SubmittedRuntimeTurn(
                model=self.build_model(error=missing, active_save_id=save_id),
                save_id=save_id,
            )

        self._queued_chat_submissions.add(save_id)
        try:
            async with self._save_operation_lock(save_id):
                cancellation_token = CancellationToken()
                self._active_chat_cancellations[save_id] = cancellation_token
                self._queued_chat_submissions.discard(save_id)
                try:
                    if save_id in self._pending_chat_cancellations:
                        self._pending_chat_cancellations.remove(save_id)
                        cancellation_token.cancel()
                    if cancellation_token.cancelled:
                        raise asyncio.CancelledError(CHAT_TURN_CANCELLED_ERROR)
                    return await self._revise_message_unlocked(
                        message_id=message_id,
                        body=body,
                        save_id=save_id,
                        status=status,
                        action_name=action_name,
                        regeneration_feedback=regeneration_feedback,
                        current_user_id=current_user_id,
                        on_revision_committed=on_revision_committed,
                        retry_progress_callback=retry_progress_callback,
                        cancellation_token=cancellation_token,
                    )
                finally:
                    self._active_chat_cancellations.pop(save_id, None)
        finally:
            self._queued_chat_submissions.discard(save_id)
            self._pending_chat_cancellations.discard(save_id)

    async def _revise_message_unlocked(
        self,
        *,
        message_id: str,
        body: str | None,
        save_id: str,
        status: str,
        action_name: str,
        regeneration_feedback: str = "",
        current_user_id: str | None = None,
        on_revision_committed: Callable[[RuntimeModel], None] | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        cancellation_token: CancellationToken,
    ) -> SubmittedRuntimeTurn:
        replacement_player_message_id: str | None = None
        try:
            self.repositories.begin_transaction()
            revision_service = MessageRevisionService(self.repositories)
            if body is None:
                revision = revision_service.regenerate_message(
                    save_id=save_id,
                    message_id=message_id,
                )
            else:
                revision = revision_service.edit_and_resubmit_message(
                    save_id=save_id,
                    message_id=message_id,
                    body=body,
                )
            display_speaker_name = _player_speaker_name(
                repositories=self.repositories,
                save_id=save_id,
                requested_name=revision.speaker_name,
            )
            active_message_ids_before_resubmission = frozenset(
                message.id for message in self.repositories.list_messages(save_id)
            )
            active_summary_ids_before_resubmission = frozenset(
                summary.id for summary in self.repositories.list_summaries(save_id)
            )
            if body is not None:
                replacement_player = self.repositories.append_message(
                    save_id=save_id,
                    role="player",
                    speaker_name=display_speaker_name,
                    body=revision.body,
                )
                replacement_player_message_id = replacement_player.id
            self.repositories.commit_transaction()
        except asyncio.CancelledError:
            self.repositories.rollback_transaction()
            cancellation_token.cancel()
            raise
        except Exception as exc:
            self.repositories.rollback_transaction()
            log_error_event(
                "runtime.message_revision_failed",
                save_id=save_id,
                message_id=message_id,
                action=action_name,
                **exception_log_fields(exc),
            )
            return SubmittedRuntimeTurn(
                model=self.build_model(
                    error=_user_visible_error(exc),
                    active_save_id=save_id,
                ),
                save_id=save_id,
            )

        if replacement_player_message_id is not None and on_revision_committed:
            try:
                on_revision_committed(
                    self.build_model(
                        status="Resubmitting message...",
                        active_save_id=save_id,
                    )
                )
            except Exception as exc:
                log_error_event(
                    "runtime.message_revision_commit_callback_failed",
                    save_id=save_id,
                    message_id=message_id,
                    action=action_name,
                    **exception_log_fields(exc),
                )

        def restore_resubmission_after_failure() -> None:
            if replacement_player_message_id is not None:
                return
            try:
                self.repositories.begin_transaction()
                MessageRevisionService(self.repositories).restore_resubmission(
                    save_id=save_id,
                    revision=revision,
                    active_message_ids_before_resubmission=(
                        active_message_ids_before_resubmission
                    ),
                    active_summary_ids_before_resubmission=(
                        active_summary_ids_before_resubmission
                    ),
                )
                self.repositories.commit_transaction()
            except Exception as restore_exc:
                self.repositories.rollback_transaction()
                log_error_event(
                    "runtime.message_revision_restore_failed",
                    save_id=save_id,
                    message_id=message_id,
                    action=action_name,
                    **exception_log_fields(restore_exc),
                )

        try:
            chat_service = ChatService(
                repositories=self.repositories,
                providers=self.providers,
                context_search_service=self.context_search_service,
                summary_service=self._summary_service(),
                media_service=self._media_service(),
                prompt_inspection_store=self._prompt_inspection_store_if_enabled(),
            )
            if replacement_player_message_id is None:
                if regeneration_feedback:
                    kwargs: dict[str, object] = {
                        "save_id": save_id,
                        "body": revision.body,
                        "speaker_name": display_speaker_name,
                        "run_post_turn_jobs": False,
                        "regeneration_feedback": regeneration_feedback,
                    }
                else:
                    kwargs = {
                        "save_id": save_id,
                        "body": revision.body,
                        "speaker_name": display_speaker_name,
                        "run_post_turn_jobs": False,
                    }
                if _call_accepts_keyword(
                    chat_service.submit_player_turn,
                    "current_user_id",
                ):
                    kwargs["current_user_id"] = current_user_id
                if (
                    retry_progress_callback is not None
                    and _call_accepts_keyword(
                        chat_service.submit_player_turn,
                        "retry_progress_callback",
                    )
                ):
                    kwargs["retry_progress_callback"] = retry_progress_callback
                if _call_accepts_keyword(
                    chat_service.submit_player_turn,
                    "cancellation_requested",
                ):
                    kwargs["cancellation_requested"] = (
                        lambda: cancellation_token.cancelled
                    )
                if _call_accepts_keyword(
                    chat_service.submit_player_turn,
                    "cancellation_token",
                ):
                    kwargs["cancellation_token"] = cancellation_token
                submitted_turn = await cast(Any, chat_service.submit_player_turn)(
                    **kwargs
                )
            else:
                kwargs = {
                    "save_id": save_id,
                    "player_message_id": replacement_player_message_id,
                    "run_post_turn_jobs": False,
                }
                if _call_accepts_keyword(
                    chat_service.submit_existing_player_turn,
                    "current_user_id",
                ):
                    kwargs["current_user_id"] = current_user_id
                if (
                    retry_progress_callback is not None
                    and _call_accepts_keyword(
                        chat_service.submit_existing_player_turn,
                        "retry_progress_callback",
                    )
                ):
                    kwargs["retry_progress_callback"] = retry_progress_callback
                if _call_accepts_keyword(
                    chat_service.submit_existing_player_turn,
                    "cancellation_requested",
                ):
                    kwargs["cancellation_requested"] = (
                        lambda: cancellation_token.cancelled
                    )
                if _call_accepts_keyword(
                    chat_service.submit_existing_player_turn,
                    "cancellation_token",
                ):
                    kwargs["cancellation_token"] = cancellation_token
                submitted_turn = await cast(
                    Any,
                    chat_service.submit_existing_player_turn,
                )(**kwargs)
        except asyncio.CancelledError:
            cancellation_token.cancel()
            log_event(
                "runtime.message_revision_cancelled",
                save_id=save_id,
                message_id=message_id,
                action=action_name,
            )
            restore_resubmission_after_failure()
            raise
        except Exception as exc:
            log_error_event(
                "runtime.message_revision_failed",
                save_id=save_id,
                message_id=message_id,
                action=action_name,
                **exception_log_fields(exc),
            )
            restore_resubmission_after_failure()
            return SubmittedRuntimeTurn(
                model=self.build_model(
                    error=_user_visible_error(exc),
                    active_save_id=save_id,
                ),
                save_id=save_id,
                player_message_id=replacement_player_message_id,
            )

        log_event(
            "runtime.message_revision_succeeded",
            save_id=save_id,
            message_id=message_id,
            action=action_name,
            player_message_id=submitted_turn.player_message.id,
            narrator_message_id=submitted_turn.narrator_message.id,
        )
        return SubmittedRuntimeTurn(
            model=self.build_model(status=status, active_save_id=save_id),
            save_id=save_id,
            player_message_id=submitted_turn.player_message.id,
            narrator_message_id=submitted_turn.narrator_message.id,
        )

    @asynccontextmanager
    async def _save_operation_lock(self, save_id: str) -> AsyncIterator[None]:
        lock = self._thread_save_operation_lock(save_id)
        await self._acquire_thread_save_operation_lock(lock, save_id=save_id)
        try:
            yield
        finally:
            lock.release()

    @asynccontextmanager
    async def _post_input_submit_context(
        self,
        *,
        save_id: str,
        post_input_catchup: Callable[[], Awaitable[None]] | None,
    ) -> AsyncIterator[None]:
        if post_input_catchup is not None:
            await post_input_catchup()
        async with self._save_operation_lock(save_id):
            yield

    def _defer_submit_lock_until_after_input(
        self,
        *,
        save_id: str,
        post_input_catchup: Callable[[], Awaitable[None]] | None,
    ) -> bool:
        return (
            post_input_catchup is not None
            or self._maintenance_retry_drain_active(save_id)
        )

    def _begin_maintenance_retry_drain(self, save_id: str) -> None:
        with self._maintenance_retry_drain_guard:
            self._maintenance_retry_drain_counts[save_id] = (
                self._maintenance_retry_drain_counts.get(save_id, 0) + 1
            )

    def _end_maintenance_retry_drain(self, save_id: str) -> None:
        with self._maintenance_retry_drain_guard:
            count = self._maintenance_retry_drain_counts.get(save_id, 0)
            if count <= 1:
                self._maintenance_retry_drain_counts.pop(save_id, None)
                return
            self._maintenance_retry_drain_counts[save_id] = count - 1

    def _maintenance_retry_drain_active(self, save_id: str) -> bool:
        with self._maintenance_retry_drain_guard:
            return self._maintenance_retry_drain_counts.get(save_id, 0) > 0

    async def _acquire_thread_save_operation_lock(
        self,
        lock: threading.Lock,
        *,
        save_id: str,
    ) -> None:
        acquire_task = asyncio.create_task(asyncio.to_thread(lock.acquire))
        try:
            acquired = await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            self._release_thread_save_operation_lock_when_acquired(
                acquire_task,
                lock,
                save_id=save_id,
            )
            raise
        if not acquired:
            raise RuntimeError(f"Failed to acquire save operation lock: {save_id}")

    def _release_thread_save_operation_lock_when_acquired(
        self,
        acquire_task: asyncio.Task[bool],
        lock: threading.Lock,
        *,
        save_id: str,
    ) -> None:
        def release_if_acquired(task: asyncio.Task[bool]) -> None:
            try:
                acquired = task.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 - defensive cleanup callback
                log_error_event(
                    "runtime.save_operation_lock_acquire_failed_after_cancel",
                    save_id=save_id,
                    **exception_log_fields(exc),
                )
                return
            if not acquired:
                return
            try:
                lock.release()
            except RuntimeError as exc:
                log_error_event(
                    "runtime.save_operation_lock_release_failed_after_cancel",
                    save_id=save_id,
                    **exception_log_fields(exc),
                )
                return
            log_event(
                "runtime.save_operation_lock_released_after_cancelled_waiter",
                save_id=save_id,
            )

        if acquire_task.done():
            release_if_acquired(acquire_task)
        else:
            acquire_task.add_done_callback(release_if_acquired)

    def _thread_save_operation_lock(self, save_id: str) -> threading.Lock:
        with self._save_operation_locks_guard:
            lock = self._save_operation_locks.get(save_id)
            if lock is None:
                lock = threading.Lock()
                self._save_operation_locks[save_id] = lock
            return lock

    def _prompt_inspection_store_if_enabled(
        self,
    ) -> PromptInspectionStore | None:
        if not self.repositories.get_app_setting("debug_logging_enabled"):
            return None
        return self.prompt_inspection_store

    async def generate_image(
        self,
        *,
        source_message_id: str,
        active_save_id: str | None | object = ...,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event("runtime.image_generation_failed", error="No save loaded")
            return self.build_model(error="No save loaded")
        missing = _missing_image_generation_requirement(
            self.repositories,
            self.providers,
            save_id=save_id,
        )
        if missing is not None:
            log_error_event(
                "runtime.image_generation_failed",
                save_id=save_id,
                source_message_id=source_message_id,
                error=missing,
            )
            return self.build_model(
                error=missing,
                active_save_id=save_id,
            )
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="image_generation",
        )
        if preference is None:
            raise AssertionError("image generation requirement check failed")
        try:
            async with self._save_operation_lock(save_id):
                media_service = self._media_service()
                kwargs: dict[str, object] = {
                    "save_id": save_id,
                    "source_message_id": source_message_id,
                }
                if (
                    retry_progress_callback is not None
                    and _call_accepts_keyword(
                        media_service.generate_for_message,
                        "retry_progress_callback",
                    )
                ):
                    kwargs["retry_progress_callback"] = retry_progress_callback
                if _call_accepts_keyword(
                    media_service.generate_for_message,
                    "current_user_id",
                ):
                    kwargs["current_user_id"] = current_user_id
                await cast(Any, media_service.generate_for_message)(**kwargs)
        except Exception as exc:
            log_error_event(
                "runtime.image_generation_failed",
                save_id=save_id,
                source_message_id=source_message_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.image_generation_succeeded",
            save_id=save_id,
            source_message_id=source_message_id,
            provider=preference.provider,
            model=preference.model_id,
        )
        return self.build_model(status="Image generated", active_save_id=save_id)

    async def regenerate_media_asset(
        self,
        media_asset_id: str,
        *,
        prompt: str,
        active_save_id: str | None | object = ...,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.image_regeneration_failed",
                media_asset_id=media_asset_id,
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")
        try:
            async with self._save_operation_lock(save_id):
                media_service = self._media_service()
                kwargs: dict[str, object] = {
                    "save_id": save_id,
                    "media_asset_id": media_asset_id,
                    "prompt": prompt,
                }
                if (
                    retry_progress_callback is not None
                    and _call_accepts_keyword(
                        media_service.regenerate_asset_with_prompt,
                        "retry_progress_callback",
                    )
                ):
                    kwargs["retry_progress_callback"] = retry_progress_callback
                if _call_accepts_keyword(
                    media_service.regenerate_asset_with_prompt,
                    "current_user_id",
                ):
                    kwargs["current_user_id"] = current_user_id
                asset = await cast(Any, media_service.regenerate_asset_with_prompt)(
                    **kwargs
                )
        except Exception as exc:
            log_error_event(
                "runtime.image_regeneration_failed",
                save_id=save_id,
                media_asset_id=media_asset_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.image_regeneration_succeeded",
            save_id=save_id,
            media_asset_id=asset.id,
            replaced_media_asset_id=media_asset_id,
            provider=asset.provider,
            model=asset.model,
        )
        return self.build_model(status="Image regenerated", active_save_id=save_id)

    async def generate_character_image(
        self,
        *,
        source_message_id: str,
        character_id: str,
        active_save_id: str | None | object = ...,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.character_image_generation_failed",
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")
        if _active_save_scenario_type(
            self.repositories,
            save_id=save_id,
        ) not in ROLEPLAY_TYPES:
            message = "Character images require a roleplay save"
            log_error_event(
                "runtime.character_image_generation_failed",
                save_id=save_id,
                source_message_id=source_message_id,
                character_id=character_id,
                error=message,
            )
            return self.build_model(error=message, active_save_id=save_id)
        missing = _missing_character_image_requirement(
            self.repositories,
            self.providers,
            save_id=save_id,
        )
        if missing is not None:
            log_error_event(
                "runtime.character_image_generation_failed",
                save_id=save_id,
                source_message_id=source_message_id,
                character_id=character_id,
                error=missing,
            )
            return self.build_model(error=missing, active_save_id=save_id)
        preference = image_edit_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose=CHARACTER_IMAGE_EDIT_PURPOSE,
        )
        if preference is None:
            raise AssertionError("character image requirement check failed")
        try:
            async with self._save_operation_lock(save_id):
                media_service = self._media_service()
                kwargs: dict[str, object] = {
                    "save_id": save_id,
                    "source_message_id": source_message_id,
                    "character_id": character_id,
                    "job_context": "manual_character_image",
                }
                if (
                    retry_progress_callback is not None
                    and _call_accepts_keyword(
                        media_service.generate_character_image_for_message,
                        "retry_progress_callback",
                    )
                ):
                    kwargs["retry_progress_callback"] = retry_progress_callback
                if _call_accepts_keyword(
                    media_service.generate_character_image_for_message,
                    "current_user_id",
                ):
                    kwargs["current_user_id"] = current_user_id
                await cast(Any, media_service.generate_character_image_for_message)(
                    **kwargs
                )
        except Exception as exc:
            log_error_event(
                "runtime.character_image_generation_failed",
                save_id=save_id,
                source_message_id=source_message_id,
                character_id=character_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.character_image_generation_succeeded",
            save_id=save_id,
            source_message_id=source_message_id,
            character_id=character_id,
            provider=preference.provider,
            model=preference.model_id,
        )
        return self.build_model(
            status="Character image generated",
            active_save_id=save_id,
        )

    async def generate_character_registry_image(
        self,
        character_id: str,
        *,
        instructions: str = "",
        active_save_id: str | None | object = ...,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.character_image_generation_failed",
                character_id=character_id,
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")
        missing = _missing_character_image_requirement(
            self.repositories,
            self.providers,
            save_id=save_id,
        )
        if missing is not None:
            log_error_event(
                "runtime.character_image_generation_failed",
                save_id=save_id,
                character_id=character_id,
                error=missing,
            )
            return self.build_model(error=missing, active_save_id=save_id)
        preference = image_edit_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose=CHARACTER_IMAGE_EDIT_PURPOSE,
        )
        if preference is None:
            raise AssertionError("character image requirement check failed")
        try:
            async with self._save_operation_lock(save_id):
                media_service = self._media_service()
                kwargs: dict[str, object] = {
                    "save_id": save_id,
                    "character_id": character_id,
                    "instructions": instructions,
                    "job_context": "manual_registry_character_image",
                }
                if (
                    retry_progress_callback is not None
                    and _call_accepts_keyword(
                        media_service.generate_character_image_for_character,
                        "retry_progress_callback",
                    )
                ):
                    kwargs["retry_progress_callback"] = retry_progress_callback
                if _call_accepts_keyword(
                    media_service.generate_character_image_for_character,
                    "current_user_id",
                ):
                    kwargs["current_user_id"] = current_user_id
                await cast(Any, media_service.generate_character_image_for_character)(
                    **kwargs
                )
        except Exception as exc:
            log_error_event(
                "runtime.character_image_generation_failed",
                save_id=save_id,
                character_id=character_id,
                provider=preference.provider,
                model=preference.model_id,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.character_image_generation_succeeded",
            save_id=save_id,
            character_id=character_id,
            provider=preference.provider,
            model=preference.model_id,
        )
        return self.build_model(
            status="Character image generated",
            active_save_id=save_id,
        )

    async def generate_initial_scenario_image(
        self,
        *,
        source_message_id: str,
        active_save_id: str | None | object = ...,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.initial_scenario_image_failed",
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")
        missing = _missing_image_generation_requirement(
            self.repositories,
            self.providers,
            save_id=save_id,
        )
        if missing is not None:
            log_error_event(
                "runtime.initial_scenario_image_failed",
                save_id=save_id,
                opening_message_id=source_message_id,
                source_message_id=source_message_id,
                error=missing,
            )
            return self.build_model(
                error=missing,
                active_save_id=save_id,
            )
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="image_generation",
        )
        if preference is None:
            raise AssertionError("image generation requirement check failed")
        provider_name = preference.provider
        model_name = preference.model_id
        log_event(
            "runtime.initial_scenario_image_started",
            save_id=save_id,
            opening_message_id=source_message_id,
            source_message_id=source_message_id,
            provider=provider_name,
            model=model_name,
        )
        try:
            async with self._save_operation_lock(save_id):
                media_service = self._media_service()
                kwargs: dict[str, object] = {
                    "save_id": save_id,
                    "source_message_id": source_message_id,
                    "job_context": "initial_scenario_image",
                }
                if (
                    retry_progress_callback is not None
                    and _call_accepts_keyword(
                        media_service.generate_for_message,
                        "retry_progress_callback",
                    )
                ):
                    kwargs["retry_progress_callback"] = retry_progress_callback
                if _call_accepts_keyword(
                    media_service.generate_for_message,
                    "current_user_id",
                ):
                    kwargs["current_user_id"] = current_user_id
                await cast(Any, media_service.generate_for_message)(**kwargs)
        except Exception as exc:
            log_error_event(
                "runtime.initial_scenario_image_failed",
                save_id=save_id,
                opening_message_id=source_message_id,
                source_message_id=source_message_id,
                provider=provider_name,
                model=model_name,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )
        log_event(
            "runtime.initial_scenario_image_succeeded",
            save_id=save_id,
            opening_message_id=source_message_id,
            source_message_id=source_message_id,
            provider=provider_name,
            model=model_name,
        )
        return self.build_model(
            status="Initial scene image generated",
            active_save_id=save_id,
        )

    async def generate_character_reference_image(
        self,
        character_id: str,
        *,
        source_message_id: str | None = None,
        replace_existing: bool = False,
        active_save_id: str | None | object = ...,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> RuntimeModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        if save_id is None:
            log_error_event(
                "runtime.character_reference_generation_failed",
                character_id=character_id,
                error="No save loaded",
            )
            return self.build_model(error="No save loaded")

        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="image_generation",
        )
        provider_name = (
            preference.provider if preference is not None else "unconfigured"
        )
        model_name = preference.model_id if preference is not None else "unconfigured"
        try:
            async with self._save_operation_lock(save_id):
                media_service = self._media_service()
                kwargs: dict[str, object] = {
                    "save_id": save_id,
                    "character_id": character_id,
                    "source_message_id": source_message_id,
                    "job_context": "manual_character_reference",
                    "replace_existing": replace_existing,
                }
                if (
                    retry_progress_callback is not None
                    and _call_accepts_keyword(
                        media_service.generate_character_reference,
                        "retry_progress_callback",
                    )
                ):
                    kwargs["retry_progress_callback"] = retry_progress_callback
                if _call_accepts_keyword(
                    media_service.generate_character_reference,
                    "current_user_id",
                ):
                    kwargs["current_user_id"] = current_user_id
                await cast(Any, media_service.generate_character_reference)(**kwargs)
        except Exception as exc:
            log_error_event(
                "runtime.character_reference_generation_failed",
                save_id=save_id,
                character_id=character_id,
                provider=provider_name,
                model=model_name,
                **exception_log_fields(exc),
            )
            return self.build_model(
                error=_user_visible_error(exc),
                active_save_id=save_id,
            )

        log_event(
            "runtime.character_reference_generation_succeeded",
            save_id=save_id,
            character_id=character_id,
            provider=provider_name,
            model=model_name,
            replaced=replace_existing,
        )
        return self.build_model(
            status=(
                "Character reference image replaced"
                if replace_existing
                else "Character reference image generated"
            ),
            active_save_id=save_id,
        )

    def build_world_data_model(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> WorldDataModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        return WorldDataService(
            self.repositories,
            active_save_id=save_id,
        ).build_model(active_save_id=save_id if save_id else ...)

    def open_world_data(self) -> RuntimeModel:
        model = self.build_world_data_model()
        if model.error is not None:
            log_error_event("runtime.world_data_open_failed", error=model.error)
            return self.build_model(error=model.error)
        log_event("runtime.world_data_opened", save_id=model.save_id)
        return self.build_model(status="World data opened")

    def open_world_data_editor(self) -> RuntimeModel:
        return self.open_world_data()

    def build_character_registry_model(
        self,
        *,
        active_save_id: str | None | object = ...,
    ) -> CharacterRegistryModel:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        return CharacterRegistryService(
            self.repositories,
            active_save_id=save_id,
        ).build_model(active_save_id=save_id if save_id else ...)

    def open_character_registry(self) -> RuntimeModel:
        model = self.build_character_registry_model()
        if model.error is not None:
            log_error_event("runtime.character_registry_open_failed", error=model.error)
            return self.build_model(error=model.error)
        log_event("runtime.character_registry_opened", save_id=model.save_id)
        return self.build_model(status="Characters opened")

    def complete_sparse_character_profiles(
        self,
        *,
        active_save_id: str,
        character_ids: tuple[str, ...],
    ) -> int:
        save = self.repositories.get_save(active_save_id)
        if save is None:
            return 0
        scenario = self.repositories.get_scenario(save.scenario_id)
        if scenario is None:
            return 0
        content = _scenario_content(scenario.content_json)
        starters = tuple(
            _character_profile_starter(character)
            for character_id in character_ids
            if (character := self.repositories.get_character(character_id)) is not None
            and character.save_id == active_save_id
            and _character_needs_profile_completion(character)
        )
        if not starters:
            return 0
        completer = self._character_profile_completer(scenario.type)
        if completer is None:
            return 0
        request = CharacterProfileCompletionRequest(
            scenario_type=scenario.type,
            scenario_context=scenario_context_text(
                scenario_type=scenario.type,
                content=content,
            ),
            starters=starters,
            save_id=active_save_id,
        )
        complete = getattr(completer, "complete", None)
        if not callable(complete):
            return 0
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            completed = asyncio.run(complete(request))
        else:
            return 0
        completed_by_key = {
            _character_key(starter.name): starter for starter in tuple(completed)
        }
        updated_count = 0
        for character_id in character_ids:
            character = self.repositories.get_character(character_id)
            if character is None or character.save_id != active_save_id:
                continue
            starter = completed_by_key.get(_character_key(character.name))
            if starter is None:
                continue
            updated = _apply_completed_character_profile(character, starter)
            if updated != character:
                self.repositories.update_character(updated)
                updated_count += 1
        return updated_count

    def complete_new_character_agency(
        self,
        *,
        active_save_id: str,
        character_ids: tuple[str, ...],
        current_user_id: str | None = None,
    ) -> int:
        save = self.repositories.get_save(active_save_id)
        if save is None:
            return 0
        scenario = self.repositories.get_scenario(save.scenario_id)
        if scenario is None:
            return 0
        content = _scenario_content(scenario.content_json)
        starters = tuple(
            _character_profile_starter(character)
            for character_id in character_ids
            if (character := self.repositories.get_character(character_id)) is not None
            and character.save_id == active_save_id
            and _character_needs_agency_completion(character)
        )
        if not starters:
            return 0
        completer = self._character_profile_completer(scenario.type)
        if completer is None:
            return 0
        complete = getattr(completer, "complete", None)
        if not callable(complete):
            return 0
        request = CharacterProfileCompletionRequest(
            scenario_type=scenario.type,
            scenario_context=scenario_context_text(
                scenario_type=scenario.type,
                content=content,
            ),
            starters=starters,
            save_id=active_save_id,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                completed = asyncio.run(complete(request))
            except Exception as exc:
                log_error_event(
                    "runtime.new_character_agency_completion_failed",
                    save_id=active_save_id,
                    character_count=len(character_ids),
                    **exception_log_fields(exc),
                )
                return 0
        else:
            return 0
        completed_by_key = {
            _character_key(starter.name): starter for starter in tuple(completed)
        }
        updated_count = 0
        for character_id in character_ids:
            character = self.repositories.get_character(character_id)
            if character is None or character.save_id != active_save_id:
                continue
            starter = completed_by_key.get(_character_key(character.name))
            if starter is None:
                continue
            updated = _apply_completed_character_agency(character, starter)
            safety = self._review_actor_content_blocking(
                body=_character_record_safety_body(updated),
                save_id=active_save_id,
                current_user_id=current_user_id,
            )
            updated = (
                replace(updated, content_rating=safety.reviewed_content_rating)
                if safety.action is ContentSafetyAction.ALLOW
                else _character_record_with_safe_transition(
                    updated,
                    replacement=safety.body,
                    content_rating=safety.reviewed_content_rating,
                )
            )
            if updated != character:
                self.repositories.update_character(updated)
                updated_count += 1
        log_event(
            "runtime.new_character_agency_completion_completed",
            save_id=active_save_id,
            character_count=len(character_ids),
            updated_count=updated_count,
        )
        return updated_count

    def enhance_character_registry_field(
        self,
        *,
        active_save_id: str,
        character_id: str,
        field_name: str,
        row: CharacterRegistryRow,
        current_user_id: str | None = None,
    ) -> CharacterFieldEnhanceResult:
        field_name = _validated_character_enhancement_field(field_name)
        if row.character_id != character_id:
            raise ValueError("Character enhancement row does not match request")
        if row.archived:
            raise ValueError("Character cannot be archived by enhancement")
        if row.merge_into_character_id is not None:
            raise ValueError("Clear merge target before auto-enhancing")
        if not row.name.strip():
            raise ValueError("Character name must not be blank")
        save = self.repositories.get_save(active_save_id)
        if save is None:
            raise ValueError("No save loaded")
        scenario = self.repositories.get_scenario(save.scenario_id)
        if scenario is None:
            raise ValueError("No scenario loaded")
        character = self.repositories.get_character(character_id)
        if character is None or character.save_id != active_save_id:
            raise ValueError("Character edit does not belong to the active save")
        preference = character_enhancement_model_preference(
            repositories=self.repositories,
            save_id=active_save_id,
        )
        provider_name = preference.provider if preference is not None else None
        model_id = preference.model_id if preference is not None else None
        completer = self._character_field_enhancement_completer(active_save_id)
        enhance_field = getattr(completer, "enhance_field", None)
        if not callable(enhance_field):
            raise ValueError("Character enhancement model is unavailable")
        if field_name in CHARACTER_AGENCY_FIELDS:
            scenario_context, evidence_source_ids = (
                _character_registry_agency_enhancement_context(
                    repositories=self.repositories,
                    save_id=active_save_id,
                    scenario_id=scenario.id,
                    scenario_type=scenario.type,
                    content=_scenario_content(scenario.content_json),
                    row=row,
                )
            )
        else:
            scenario_context = _character_registry_enhancement_context(
                repositories=self.repositories,
                save_id=active_save_id,
                scenario_type=scenario.type,
                content=_scenario_content(scenario.content_json),
                row=row,
            )
            evidence_source_ids = ()
        request = CharacterFieldEnhancementRequest(
            scenario_type=scenario.type,
            scenario_context=scenario_context,
            character=_character_profile_starter_from_registry_row(row),
            field_name=field_name,
            save_id=active_save_id,
            evidence_source_ids=evidence_source_ids,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                enhanced = asyncio.run(enhance_field(request))
            except Exception as exc:
                log_error_event(
                    "runtime.character_field_enhancement_failed",
                    save_id=active_save_id,
                    character_id=character_id,
                    provider=provider_name,
                    model=model_id,
                    field_name=field_name,
                    **exception_log_fields(exc),
                )
                raise ValueError(_user_visible_error(exc)) from exc
        else:
            raise ValueError("Character enhancement is unavailable while busy")
        enhanced_row = _enhanced_character_registry_row(
            row,
            field_name=field_name,
            enhanced=enhanced,
            existing_locked_fields=tuple(character.locked_fields),
        )
        safety = self._review_actor_content_blocking(
            body=_character_registry_row_safety_body(enhanced_row),
            save_id=active_save_id,
            current_user_id=current_user_id,
        )
        enhanced_row = (
            replace(
                enhanced_row,
                content_rating=safety.reviewed_content_rating,
            )
            if safety.action is ContentSafetyAction.ALLOW
            else _character_registry_row_with_safe_transition(
                enhanced_row,
                replacement=safety.body,
                content_rating=safety.reviewed_content_rating,
            )
        )
        field_changed = _enhanced_character_registry_field_changed(
            row,
            enhanced_row,
            field_name=field_name,
        )
        applied_row = enhanced_row if field_changed else row
        notice = None
        if not field_changed:
            notice = _character_field_enhancement_noop_notice(field_name)
        result = CharacterRegistryService(
            self.repositories,
            active_save_id=active_save_id,
        ).apply_edits(
            CharacterRegistryEdits(characters=(applied_row,)),
            active_save_id=active_save_id,
        )
        log_event(
            "runtime.character_field_enhancement_completed",
            save_id=active_save_id,
            character_id=character_id,
            provider=provider_name,
            model=model_id,
            field_name=field_name,
            field_changed=field_changed,
            created_count=result.created_count,
            updated_count=result.updated_count,
            archived_count=result.archived_count,
        )
        return CharacterFieldEnhanceResult(
            model=result.model,
            character_id=character_id,
            field_name=field_name,
            created_count=result.created_count,
            updated_count=result.updated_count,
            archived_count=result.archived_count,
            field_changed=field_changed,
            notice=notice,
        )

    def _media_service(self) -> MediaService:
        return MediaService(
            repositories=self.repositories,
            providers=self.providers,
            media_dir=self.media_dir,
            automatic_enabled=_automatic_image_generation_enabled(self.repositories),
            auto_frequency=_image_frequency(self.repositories),
        )

    def _summary_service(self) -> SummaryService:
        if self.summary_service is not None:
            return self.summary_service
        return SummaryService(
            repositories=self.repositories,
            providers=self.providers,
            enabled=_automatic_summarization_enabled(self.repositories),
            threshold=_summary_threshold(self.repositories),
        )


def _scenario_opening_message(content_json: str) -> str | None:
    content = _scenario_content(content_json)
    opening = content.get("opening_message")
    if not isinstance(opening, str):
        return None
    text = opening.strip()
    return text or None


def _scenario_content(content_json: str) -> dict[str, object]:
    try:
        content = json.loads(content_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(content, dict):
        return {}
    return content


def _scenario_types_from_content(
    primary: ScenarioType,
    content: Mapping[str, object],
) -> tuple[ScenarioType, ...]:
    raw_genres = content.get(SCENARIO_GENRES_CONTENT_KEY)
    if not isinstance(raw_genres, list):
        return (primary,)
    genres: list[ScenarioType] = []
    for raw_genre in raw_genres:
        if not isinstance(raw_genre, str):
            continue
        try:
            genre = ScenarioType(raw_genre)
        except ValueError:
            continue
        if genre not in genres:
            genres.append(genre)
    if not genres or genres[0] is not primary:
        return (primary,)
    return tuple(genres[:2])


def _saved_scenario_type_values(
    primary: str,
    content: Mapping[str, object],
) -> tuple[str, tuple[str, ...]]:
    try:
        primary_type = ScenarioType(primary)
    except ValueError:
        return primary, (primary,)
    return primary_type.value, tuple(
        genre.value for genre in _scenario_types_from_content(primary_type, content)
    )


def _content_with_scenario_genres(
    content: Mapping[str, object],
    scenario_types: tuple[ScenarioType, ...],
) -> dict[str, object]:
    normalized = dict(content)
    if len(scenario_types) > 1:
        normalized[SCENARIO_GENRES_CONTENT_KEY] = [
            scenario_type.value for scenario_type in scenario_types
        ]
    return normalized


def _scenario_source_metadata(content_json: str) -> dict[str, object]:
    source = _scenario_content(content_json).get("_source")
    return dict(source) if isinstance(source, dict) else {}


def _scenario_generation_prompt(content_json: str) -> str | None:
    prompt = _scenario_source_metadata(content_json).get("generation_prompt")
    if not isinstance(prompt, str):
        return None
    text = prompt.strip()
    return text or None


def _character_profile_starter(character: CharacterRecord) -> ScenarioCharacterStarter:
    return ScenarioCharacterStarter(
        name=character.name,
        aliases=tuple(character.aliases),
        role=character.role,
        age=character.age,
        known_state=character.known_state,
        appearance=character.appearance,
        visual_notes=character.visual_notes,
        personality=character.personality,
        voice=character.voice,
        texting_style=character.texting_style,
        relationships=dict(character.relationships),
        goals=character.goals,
        motivations=character.motivations,
        current_intent=character.current_intent,
        boundaries=character.boundaries,
        attitude_toward_player=character.attitude_toward_player,
        cooperation_conditions=character.cooperation_conditions,
        status=character.status,
        met=character.met,
        locked_fields=tuple(character.locked_fields),
    )


def _character_needs_profile_completion(character: CharacterRecord) -> bool:
    return any(
        _character_profile_field_blank_and_unlocked(character, field)
        for field in (
            *CHARACTER_STARTER_IDENTITY_LOCK_FIELDS,
            *CHARACTER_STARTER_AGENCY_LOCK_FIELDS,
        )
        if field not in {"name", "age"}
    )


def _character_needs_agency_completion(character: CharacterRecord) -> bool:
    return any(
        _character_profile_field_blank_and_unlocked(character, field)
        for field in CHARACTER_STARTER_AGENCY_LOCK_FIELDS
    )


def _apply_completed_character_profile(
    character: CharacterRecord,
    starter: ScenarioCharacterStarter,
) -> CharacterRecord:
    updates: dict[str, object] = {}
    generated_locks: list[str] = []
    for field in (
        *CHARACTER_STARTER_IDENTITY_LOCK_FIELDS,
        *CHARACTER_STARTER_AGENCY_LOCK_FIELDS,
    ):
        if field == "name":
            continue
        if not _character_profile_field_blank_and_unlocked(character, field):
            continue
        value = getattr(starter, field)
        if _empty_profile_value(value):
            continue
        updates[field] = value
        generated_locks.append(field)
    if not character.relationships and starter.relationships:
        updates["relationships"] = dict(starter.relationships)
    if not character.status.strip() and starter.status.strip():
        updates["status"] = starter.status.strip()
    if not updates:
        return character
    return replace(
        character,
        aliases=list(cast(tuple[str, ...], updates.get("aliases", character.aliases))),
        role=cast(str, updates.get("role", character.role)),
        age=cast(str, updates.get("age", character.age)),
        known_state=cast(str, updates.get("known_state", character.known_state)),
        appearance=cast(str, updates.get("appearance", character.appearance)),
        visual_notes=cast(str, updates.get("visual_notes", character.visual_notes)),
        personality=cast(str, updates.get("personality", character.personality)),
        voice=cast(str, updates.get("voice", character.voice)),
        texting_style=cast(
            str,
            updates.get("texting_style", character.texting_style),
        ),
        relationships=cast(
            dict[str, object],
            updates.get("relationships", character.relationships),
        ),
        goals=cast(str, updates.get("goals", character.goals)),
        motivations=cast(str, updates.get("motivations", character.motivations)),
        current_intent=cast(
            str,
            updates.get("current_intent", character.current_intent),
        ),
        boundaries=cast(str, updates.get("boundaries", character.boundaries)),
        attitude_toward_player=cast(
            str,
            updates.get("attitude_toward_player", character.attitude_toward_player),
        ),
        cooperation_conditions=cast(
            str,
            updates.get("cooperation_conditions", character.cooperation_conditions),
        ),
        status=cast(str, updates.get("status", character.status)),
        locked_fields=(
            normalize_character_locked_fields(
                (*character.locked_fields, *generated_locks),
                preserve_unknown=True,
            )
            if generated_locks
            else character.locked_fields
        ),
    )


def _apply_completed_character_agency(
    character: CharacterRecord,
    starter: ScenarioCharacterStarter,
) -> CharacterRecord:
    updates: dict[str, str] = {}
    generated_locks: list[str] = []
    for field in CHARACTER_STARTER_AGENCY_LOCK_FIELDS:
        if not _character_profile_field_blank_and_unlocked(character, field):
            continue
        value = getattr(starter, field)
        if not isinstance(value, str) or not value.strip():
            continue
        updates[field] = value.strip()
        generated_locks.append(field)
    if not updates:
        return character
    return replace(
        character,
        goals=updates.get("goals", character.goals),
        motivations=updates.get("motivations", character.motivations),
        current_intent=updates.get("current_intent", character.current_intent),
        boundaries=updates.get("boundaries", character.boundaries),
        attitude_toward_player=updates.get(
            "attitude_toward_player",
            character.attitude_toward_player,
        ),
        cooperation_conditions=updates.get(
            "cooperation_conditions",
            character.cooperation_conditions,
        ),
        locked_fields=normalize_character_locked_fields(
            (*character.locked_fields, *generated_locks),
            preserve_unknown=True,
        ),
    )


_CHARACTER_SAFETY_TEXT_FIELDS = (
    "name",
    "role",
    "age",
    "known_state",
    "history",
    "appearance",
    "visual_notes",
    "current_clothing",
    "personality",
    "voice",
    "texting_style",
    "goals",
    "motivations",
    "current_intent",
    "boundaries",
    "attitude_toward_player",
    "cooperation_conditions",
    "status",
    "private_notes",
    "contact_name",
)


_SCENARIO_CHARACTER_STARTER_SAFETY_TEXT_FIELDS = (
    "name",
    "role",
    "age",
    "known_state",
    "appearance",
    "visual_notes",
    "personality",
    "voice",
    "texting_style",
    "goals",
    "motivations",
    "current_intent",
    "boundaries",
    "attitude_toward_player",
    "cooperation_conditions",
    "status",
)


def _scenario_character_starter_safety_body(
    starter: ScenarioCharacterStarter,
) -> str:
    content = {
        field: getattr(starter, field)
        for field in _SCENARIO_CHARACTER_STARTER_SAFETY_TEXT_FIELDS
    }
    content["aliases"] = starter.aliases
    content["relationships"] = starter.relationships
    if starter.reference_image is not None:
        content["reference_image_prompt"] = starter.reference_image.prompt_preview
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _scenario_character_starter_with_safe_transition(
    starter: ScenarioCharacterStarter,
    *,
    replacement: str,
) -> ScenarioCharacterStarter:
    updates: dict[str, object] = {
        field: replacement if getattr(starter, field) else ""
        for field in _SCENARIO_CHARACTER_STARTER_SAFETY_TEXT_FIELDS
    }
    updates["aliases"] = ()
    updates["relationships"] = {}
    if starter.reference_image is not None:
        updates["reference_image"] = replace(
            starter.reference_image,
            prompt_preview=(
                replacement if starter.reference_image.prompt_preview else ""
            ),
        )
    return replace(starter, **updates)  # type: ignore[arg-type]


def _character_record_safety_body(character: CharacterRecord) -> str:
    content = {
        field: getattr(character, field)
        for field in _CHARACTER_SAFETY_TEXT_FIELDS
    }
    content["aliases"] = character.aliases
    content["relationships"] = character.relationships
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _character_registry_row_safety_body(row: CharacterRegistryRow) -> str:
    content = {
        field: getattr(row, field)
        for field in _CHARACTER_SAFETY_TEXT_FIELDS
    }
    content["aliases"] = row.aliases_text
    content["relationships"] = row.relationships_json
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _character_record_with_safe_transition(
    character: CharacterRecord,
    *,
    replacement: str,
    content_rating: str,
) -> CharacterRecord:
    updates: dict[str, object] = {
        field: replacement if getattr(character, field) else ""
        for field in _CHARACTER_SAFETY_TEXT_FIELDS
    }
    updates["aliases"] = []
    updates["relationships"] = {}
    updates["content_rating"] = content_rating
    return replace(character, **updates)  # type: ignore[arg-type]


def _character_registry_row_with_safe_transition(
    row: CharacterRegistryRow,
    *,
    replacement: str,
    content_rating: str,
) -> CharacterRegistryRow:
    updates: dict[str, object] = {
        field: replacement if getattr(row, field) else ""
        for field in _CHARACTER_SAFETY_TEXT_FIELDS
    }
    updates["aliases_text"] = ""
    updates["relationships_json"] = "{}"
    updates["content_rating"] = content_rating
    return replace(row, **updates)  # type: ignore[arg-type]


def _character_profile_field_blank_and_unlocked(
    character: CharacterRecord,
    field: str,
) -> bool:
    if character_field_is_locked(character.locked_fields, field):
        return False
    return _empty_profile_value(getattr(character, field))


def _empty_profile_value(value: object) -> bool:
    return value == "" or value == () or value == [] or value == {}


def _validated_character_enhancement_field(field_name: str) -> str:
    normalized = field_name.strip()
    if normalized not in CHARACTER_FIELD_ENHANCEMENT_FIELDS:
        raise ValueError(f"Unsupported character enhancement field: {field_name}")
    return normalized


def _character_profile_starter_from_registry_row(
    row: CharacterRegistryRow,
) -> ScenarioCharacterStarter:
    return ScenarioCharacterStarter(
        name=row.name.strip(),
        aliases=tuple(_csv_text(row.aliases_text)),
        role=row.role.strip(),
        age=row.age.strip(),
        known_state=(row.known_state or row.history).strip(),
        appearance=row.appearance.strip(),
        visual_notes=row.visual_notes.strip(),
        personality=row.personality.strip(),
        voice=row.voice.strip(),
        texting_style=row.texting_style.strip(),
        relationships=_relationships_from_json(row.relationships_json),
        goals=row.goals.strip(),
        motivations=row.motivations.strip(),
        current_intent=row.current_intent.strip(),
        boundaries=row.boundaries.strip(),
        attitude_toward_player=row.attitude_toward_player.strip(),
        cooperation_conditions=row.cooperation_conditions.strip(),
        status=row.status.strip(),
        met=row.met,
        locked_fields=tuple(row.locked_fields or ()),
    )


def _character_registry_enhancement_context(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: str,
    content: Mapping[str, object],
    row: CharacterRegistryRow,
) -> str:
    lines = [
        scenario_context_text(scenario_type=scenario_type, content=content),
        "Current edited character profile:",
        _character_registry_row_context(row),
    ]
    location = _character_row_location_context(
        repositories=repositories,
        save_id=save_id,
        location_id=row.location_id,
    )
    if location:
        lines.extend(("Selected location:", location))
    linked = _character_row_linked_context(
        repositories=repositories,
        save_id=save_id,
        row=row,
    )
    if linked:
        lines.extend(("Linked character knowledge:", linked))
    graph = _character_row_knowledge_graph_context(
        repositories=repositories,
        save_id=save_id,
        content=content,
        row=row,
    )
    if graph:
        lines.extend(("Character knowledge graph:", graph))
    return "\n".join(line for line in lines if line)


MAX_CHARACTER_AGENCY_CONTEXT_ITEM_CHARS = 900
MAX_CHARACTER_AGENCY_TRANSCRIPT_MESSAGES = 24
MAX_CHARACTER_AGENCY_CONTEXT_CHARS = 32000


def _character_registry_agency_enhancement_context(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_id: str,
    scenario_type: str,
    content: Mapping[str, object],
    row: CharacterRegistryRow,
) -> tuple[str, tuple[str, ...]]:
    source_ids: list[str] = []
    lines = [
        "Source-labeled save context for agency enhancement.",
        (
            "Use these source IDs as evidence_source_ids when the target agency "
            "field is directly supported."
        ),
    ]

    def add_source(source_id: str, title: str, body: str) -> None:
        source_id = source_id.strip()
        body = body.strip()
        if not source_id or not body:
            return
        source_ids.append(source_id)
        lines.append(
            f"[{source_id}] {title}: "
            f"{_compact_character_agency_context_text(body)}"
        )

    add_source(
        f"scenario:{scenario_id}",
        "Scenario",
        scenario_context_text(scenario_type=scenario_type, content=content),
    )
    current_character_source_id = (
        f"character:{row.character_id}" if row.character_id else "character:current"
    )
    add_source(
        current_character_source_id,
        "Current edited character row",
        _character_registry_row_context(row),
    )
    graph_context = _character_row_knowledge_graph_context(
        repositories=repositories,
        save_id=save_id,
        content=content,
        row=row,
    )
    if graph_context:
        add_source(
            (
                f"character_knowledge:{row.character_id}"
                if row.character_id
                else "character_knowledge:current"
            ),
            "Current character knowledge graph",
            graph_context,
        )

    snapshot = repositories.get_scene_snapshot(save_id)
    if snapshot is not None:
        add_source(
            f"scene:{snapshot.id}",
            "Current scene",
            "\n".join(
                item
                for item in (
                    f"location_id: {snapshot.current_location_id or ''}",
                    f"situation: {snapshot.situation}",
                    f"objective: {snapshot.objective}",
                    f"time: {snapshot.in_world_time}",
                    f"weather: {snapshot.weather}",
                    f"mood: {snapshot.mood}",
                    f"nearby_objects: {', '.join(snapshot.nearby_objects)}",
                    f"hazards: {', '.join(snapshot.hazards)}",
                    (
                        "present_character_ids: "
                        f"{', '.join(snapshot.present_character_ids)}"
                    ),
                )
                if item.strip()
            ),
        )

    for character in repositories.list_characters(save_id):
        add_source(
            f"character:{character.id}",
            f"Character {character.name}",
            _character_record_agency_context(character),
        )
    for location in repositories.list_locations(save_id):
        add_source(
            f"location:{location.id}",
            f"Location {location.name}",
            "\n".join(
                item
                for item in (
                    f"description: {location.description}",
                    f"visual_description: {location.visual_description}",
                    f"status: {location.status}",
                    f"connections: {', '.join(location.connections)}",
                    f"hazards: {', '.join(location.hazards)}",
                )
                if item.strip()
            ),
        )
    for state in repositories.list_world_state(save_id):
        add_source(
            f"world_state:{state.id}",
            f"World state {state.key}",
            (
                f"key: {state.key}\n"
                f"category: {state.category}\n"
                f"value: {_json_compact(state.value)}"
            ),
        )
    for memory in repositories.list_memories(save_id):
        add_source(
            f"memory:{memory.id}",
            "Memory",
            "\n".join(
                item
                for item in (
                    memory.body,
                    f"tags: {', '.join(memory.tags)}",
                    f"source_message_ids: {', '.join(memory.source_message_ids)}",
                )
                if item.strip()
            ),
        )
    for summary in repositories.list_summaries(save_id):
        add_source(
            f"summary:{summary.id}",
            "Summary",
            "\n".join(
                item
                for item in (
                    summary.body,
                    (
                        "covers: "
                        f"{summary.covers_message_start_id}"
                        f"..{summary.covers_message_end_id}"
                    ),
                )
                if item.strip()
            ),
        )
    for observation in repositories.list_context_observations(save_id):
        add_source(
            f"observation:{observation.id}",
            f"Observation {observation.observation_type}",
            "\n".join(
                item
                for item in (
                    f"claim: {observation.claim}",
                    f"evidence_quote: {observation.evidence_quote}",
                    f"status: {observation.status}",
                    f"source_message_ids: {', '.join(observation.source_message_ids)}",
                )
                if item.strip()
            ),
        )
    for source in repositories.list_context_sources(save_id):
        add_source(
            f"context_source:{source.id}",
            f"Context source {source.source_type}:{source.source_id}",
            "\n".join(
                item
                for item in (
                    source.title,
                    source.body,
                    f"metadata: {_json_compact(source.metadata)}",
                )
                if item.strip()
            ),
        )
    for suggestion in repositories.list_context_update_suggestions(
        save_id,
        status="pending",
    ):
        add_source(
            f"suggestion:{suggestion.id}",
            f"Pending suggestion {suggestion.entity_type}.{suggestion.field_path}",
            "\n".join(
                item
                for item in (
                    f"update_type: {suggestion.update_type}",
                    f"entity_id: {suggestion.entity_id or ''}",
                    f"proposed_value: {_json_compact(suggestion.proposed_value)}",
                    f"reason: {suggestion.reason}",
                    f"source_message_ids: {', '.join(suggestion.source_message_ids)}",
                )
                if item.strip()
            ),
        )

    messages = repositories.list_messages(save_id)
    if messages:
        lines.append("Transcript history:")
    for message in messages[-MAX_CHARACTER_AGENCY_TRANSCRIPT_MESSAGES:]:
        speaker = f" ({message.speaker_name})" if message.speaker_name else ""
        add_source(
            f"message:{message.id}",
            f"Message {message.role}{speaker}",
            message.body,
        )
    context_text, visible_source_ids = _truncate_character_agency_context_lines(lines)
    return (
        context_text,
        tuple(source_id for source_id in visible_source_ids if source_id in source_ids),
    )


def _character_registry_row_context(row: CharacterRegistryRow) -> str:
    relationships = _relationships_from_json(row.relationships_json)
    parts = [
        f"name: {row.name.strip()}",
        f"aliases: {row.aliases_text.strip()}" if row.aliases_text.strip() else "",
        f"role: {row.role.strip()}" if row.role.strip() else "",
        f"age: {row.age.strip()}" if row.age.strip() else "",
        f"met: {_json_bool(row.met)}",
        f"known_state: {row.known_state.strip()}" if row.known_state.strip() else "",
        f"appearance: {row.appearance.strip()}" if row.appearance.strip() else "",
        f"visual_notes: {row.visual_notes.strip()}" if row.visual_notes.strip() else "",
        (
            f"current_clothing: {row.current_clothing.strip()}"
            if row.current_clothing.strip()
            else ""
        ),
        f"personality: {row.personality.strip()}" if row.personality.strip() else "",
        f"voice: {row.voice.strip()}" if row.voice.strip() else "",
        (
            f"texting_style: {row.texting_style.strip()}"
            if row.texting_style.strip()
            else ""
        ),
        f"goals: {row.goals.strip()}" if row.goals.strip() else "",
        f"motivations: {row.motivations.strip()}" if row.motivations.strip() else "",
        (
            f"current_intent: {row.current_intent.strip()}"
            if row.current_intent.strip()
            else ""
        ),
        f"boundaries: {row.boundaries.strip()}" if row.boundaries.strip() else "",
        (
            f"attitude_toward_player: {row.attitude_toward_player.strip()}"
            if row.attitude_toward_player.strip()
            else ""
        ),
        (
            f"cooperation_conditions: {row.cooperation_conditions.strip()}"
            if row.cooperation_conditions.strip()
            else ""
        ),
        f"status: {row.status.strip()}" if row.status.strip() else "",
        f"location_id: {row.location_id}" if row.location_id else "",
        f"private_notes: {row.private_notes.strip()}"
        if row.private_notes.strip()
        else "",
        f"contact_name: {row.contact_name.strip()}" if row.contact_name.strip() else "",
        f"present: {_json_bool(row.present)}",
        (
            "relationships: "
            + "; ".join(f"{key}: {value}" for key, value in relationships.items())
            if relationships
            else ""
        ),
    ]
    return "\n".join(part for part in parts if part)


def _character_record_agency_context(character: CharacterRecord) -> str:
    parts = [
        f"name: {character.name}",
        f"aliases: {', '.join(character.aliases)}" if character.aliases else "",
        f"role: {character.role}" if character.role else "",
        f"age: {character.age}" if character.age else "",
        f"known_state: {character.known_state}" if character.known_state else "",
        (
            f"current_clothing: {character.current_clothing}"
            if character.current_clothing
            else ""
        ),
        f"status: {character.status}" if character.status else "",
        f"location_id: {character.location_id or ''}" if character.location_id else "",
        f"goals: {character.goals}" if character.goals else "",
        f"motivations: {character.motivations}" if character.motivations else "",
        (
            f"current_intent: {character.current_intent}"
            if character.current_intent
            else ""
        ),
        f"boundaries: {character.boundaries}" if character.boundaries else "",
        (
            f"attitude_toward_player: {character.attitude_toward_player}"
            if character.attitude_toward_player
            else ""
        ),
        (
            f"cooperation_conditions: {character.cooperation_conditions}"
            if character.cooperation_conditions
            else ""
        ),
        (
            "relationships: "
            + "; ".join(
                f"{key}: {value}" for key, value in character.relationships.items()
            )
            if character.relationships
            else ""
        ),
        (
            f"locked_fields: {', '.join(character.locked_fields)}"
            if character.locked_fields
            else ""
        ),
    ]
    return "\n".join(part for part in parts if part)


def _compact_character_agency_context_text(value: str) -> str:
    compact = re.sub(r"\s+", " ", value.strip())
    if len(compact) <= MAX_CHARACTER_AGENCY_CONTEXT_ITEM_CHARS:
        return compact
    return compact[: MAX_CHARACTER_AGENCY_CONTEXT_ITEM_CHARS - 1].rstrip() + "..."


def _truncate_character_agency_context_lines(
    lines: list[str],
) -> tuple[str, tuple[str, ...]]:
    output: list[str] = []
    visible_source_ids: list[str] = []
    size = 0
    for line in lines:
        extra = len(line) + (1 if output else 0)
        if size + extra <= MAX_CHARACTER_AGENCY_CONTEXT_CHARS:
            output.append(line)
            _append_visible_agency_source_id(visible_source_ids, line)
            size += extra
            continue
        remaining = MAX_CHARACTER_AGENCY_CONTEXT_CHARS - size - (1 if output else 0)
        if remaining > 20:
            truncated_line = line[: remaining - 1].rstrip() + "..."
            output.append(truncated_line)
            _append_visible_agency_source_id(visible_source_ids, truncated_line)
        output.append("[context truncated deterministically]")
        break
    return "\n".join(output), tuple(dict.fromkeys(visible_source_ids))


def _append_visible_agency_source_id(source_ids: list[str], line: str) -> None:
    match = re.match(r"\[([^\]]+)\]", line)
    if match is not None:
        source_ids.append(match.group(1))


def _json_compact(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def _json_bool(value: bool) -> str:
    return "true" if value else "false"


def _character_row_knowledge_graph_context(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    content: Mapping[str, object],
    row: CharacterRegistryRow,
) -> str:
    if not row.character_id:
        return ""
    character = repositories.get_character(row.character_id)
    if character is None or character.save_id != save_id:
        return ""
    memories = {memory.id: memory for memory in repositories.list_memories(save_id)}
    states = {state.id: state for state in repositories.list_world_state(save_id)}
    summaries = {
        summary.id: summary for summary in repositories.list_summaries(save_id)
    }
    lines: list[str] = []
    for edge in repositories.list_character_knowledge_edges(
        save_id,
        character_ids={row.character_id},
    ):
        if not knowledge_edge_allows_prompt_use(edge):
            continue
        target_type = normalized_knowledge_target_type(edge.target_type)
        target = _character_knowledge_target_text(
            target_type=target_type,
            target_id=edge.target_id,
            content=content,
            memories=memories,
            states=states,
            summaries=summaries,
        )
        if not target:
            continue
        metadata = [
            knowledge_edge_scope_label(edge, character),
            f"{target_type}:{edge.target_id}",
            f"acquired: {edge.acquisition_method}",
            f"confidence: {edge.confidence:.2f}",
        ]
        if edge.evidence_quote:
            metadata.append(f"evidence: {edge.evidence_quote}")
        if edge.source_message_ids:
            metadata.append(
                f"source_message_ids: {', '.join(edge.source_message_ids)}"
            )
        lines.append(f"{'; '.join(metadata)} - {target}")
    return "\n".join(lines)


def _character_knowledge_target_text(
    *,
    target_type: str,
    target_id: str,
    content: Mapping[str, object],
    memories: Mapping[str, MemoryRecord],
    states: Mapping[str, WorldStateRecord],
    summaries: Mapping[str, SummaryRecord],
) -> str:
    if target_type == "memory":
        memory = memories.get(target_id)
        return f"memory: {memory.body}" if memory is not None else ""
    if target_type == "world_state":
        state = states.get(target_id)
        if state is None:
            return ""
        return f"world_state {state.key}: {_json_compact(state.value)}"
    if target_type == "summary":
        summary = summaries.get(target_id)
        return f"summary: {summary.body}" if summary is not None else ""
    if target_type == "scenario_section":
        value = content.get(target_id)
        if value is None:
            return ""
        return f"scenario_section {target_id}: {_character_context_value_text(value)}"
    return ""


def _character_context_value_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return _json_compact(value)


def _character_row_location_context(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    location_id: str | None,
) -> str:
    if not location_id:
        return ""
    location = repositories.get_location(location_id)
    if location is None or location.save_id != save_id:
        return ""
    details = [
        location.name,
        location.description,
        location.visual_description,
    ]
    return " — ".join(detail.strip() for detail in details if detail.strip())


def _character_row_linked_context(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    row: CharacterRegistryRow,
) -> str:
    lines: list[str] = []
    memories = {memory.id: memory for memory in repositories.list_memories(save_id)}
    states = {state.id: state for state in repositories.list_world_state(save_id)}
    summaries = {
        summary.id: summary for summary in repositories.list_summaries(save_id)
    }
    for memory_id in row.linked_memory_ids:
        memory = memories.get(memory_id)
        if memory is not None:
            lines.append(f"memory: {memory.body}")
    for state_id in row.linked_state_ids:
        state = states.get(state_id)
        if state is not None:
            lines.append(f"fact {state.key}: {state.value}")
    for summary_id in row.linked_summary_ids:
        summary = summaries.get(summary_id)
        if summary is not None:
            lines.append(f"summary: {summary.body}")
    return "\n".join(lines)


def _enhanced_character_registry_row(
    row: CharacterRegistryRow,
    *,
    field_name: str,
    enhanced: ScenarioCharacterStarter,
    existing_locked_fields: tuple[str, ...],
) -> CharacterRegistryRow:
    base_locked_fields = (
        existing_locked_fields
        if row.locked_fields is None
        else tuple(row.locked_fields)
    )
    locked_fields = tuple(
        normalize_character_locked_fields(
            (*base_locked_fields, field_name),
            preserve_unknown=True,
        )
    )
    if field_name == "relationships":
        generated_relationships = enhanced.relationships
        if not generated_relationships:
            raise ValueError("Character enhancement did not return relationships")
        relationships = _merge_enhanced_relationships(
            _relationships_from_json(row.relationships_json),
            generated_relationships,
        )
        return replace(
            row,
            relationships_json=_relationships_to_json(relationships),
            locked_fields=locked_fields,
        )
    existing = cast(str, getattr(row, field_name)).strip()
    generated_value = getattr(enhanced, field_name)
    generated_text = (
        generated_value.strip() if isinstance(generated_value, str) else ""
    )
    if not generated_text:
        raise ValueError(
            f"Character enhancement did not return {field_name.replace('_', ' ')}"
        )
    return _replace_character_registry_text_field(
        row,
        field_name=field_name,
        value=_merge_enhanced_text(existing, generated_text),
        locked_fields=locked_fields,
    )


def _enhanced_character_registry_field_changed(
    before: CharacterRegistryRow,
    after: CharacterRegistryRow,
    *,
    field_name: str,
) -> bool:
    if field_name == "relationships":
        return _normalized_character_relationships(before) != (
            _normalized_character_relationships(after)
        )
    return _normalized_character_field_text(before, field_name) != (
        _normalized_character_field_text(after, field_name)
    )


def _normalized_character_field_text(
    row: CharacterRegistryRow,
    field_name: str,
) -> str:
    value = getattr(row, field_name)
    text = value.strip() if isinstance(value, str) else ""
    return _compact_text_key(text) or text


def _normalized_character_relationships(
    row: CharacterRegistryRow,
) -> dict[str, object]:
    return {
        key: _normalized_character_relationship_value(value)
        for key, value in _relationships_from_json(row.relationships_json).items()
    }


def _normalized_character_relationship_value(value: object) -> object:
    if isinstance(value, str):
        text = value.strip()
        return _compact_text_key(text) or text
    return value


def _character_field_enhancement_noop_notice(field_name: str) -> str:
    field_label = field_name.replace("_", " ").title()
    return f"No new {field_label} details were found; the field was left unchanged."


def _replace_character_registry_text_field(
    row: CharacterRegistryRow,
    *,
    field_name: str,
    value: str,
    locked_fields: tuple[str, ...],
) -> CharacterRegistryRow:
    if field_name == "known_state":
        return replace(
            row,
            known_state=value,
            history=value,
            locked_fields=locked_fields,
        )
    if field_name == "appearance":
        return replace(row, appearance=value, locked_fields=locked_fields)
    if field_name == "visual_notes":
        return replace(row, visual_notes=value, locked_fields=locked_fields)
    if field_name == "personality":
        return replace(row, personality=value, locked_fields=locked_fields)
    if field_name == "voice":
        return replace(row, voice=value, locked_fields=locked_fields)
    if field_name == "texting_style":
        return replace(row, texting_style=value, locked_fields=locked_fields)
    if field_name == "goals":
        return replace(row, goals=value, locked_fields=locked_fields)
    if field_name == "motivations":
        return replace(row, motivations=value, locked_fields=locked_fields)
    if field_name == "current_intent":
        return replace(row, current_intent=value, locked_fields=locked_fields)
    if field_name == "boundaries":
        return replace(row, boundaries=value, locked_fields=locked_fields)
    if field_name == "attitude_toward_player":
        return replace(row, attitude_toward_player=value, locked_fields=locked_fields)
    if field_name == "cooperation_conditions":
        return replace(row, cooperation_conditions=value, locked_fields=locked_fields)
    if field_name == "status":
        return replace(row, status=value, locked_fields=locked_fields)
    raise ValueError(f"Unsupported character enhancement field: {field_name}")


def _merge_enhanced_text(existing: str, generated: str) -> str:
    existing = existing.strip()
    generated = generated.strip()
    if not existing:
        return generated
    if not generated:
        return existing
    existing_key = _compact_text_key(existing)
    generated_key = _compact_text_key(generated)
    if existing_key and existing_key in generated_key:
        return generated
    if generated_key and generated_key in existing_key:
        return existing
    return f"{existing}\n\n{generated}"


def _merge_enhanced_relationships(
    existing: dict[str, object],
    generated: dict[str, object],
) -> dict[str, object]:
    merged = dict(existing)
    for key, generated_value in generated.items():
        if key not in merged:
            merged[key] = generated_value
            continue
        existing_value = merged[key]
        if existing_value == generated_value:
            continue
        if isinstance(existing_value, str) and isinstance(generated_value, str):
            merged[key] = _merge_enhanced_text(existing_value, generated_value)
        else:
            merged[key] = [existing_value, generated_value]
    return merged


def _relationships_from_json(value: str) -> dict[str, object]:
    try:
        loaded = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Character relationships must be valid JSON") from exc
    if not isinstance(loaded, dict) or any(
        not isinstance(key, str) for key in loaded
    ):
        raise ValueError("Character relationships must be a JSON object")
    return cast(dict[str, object], loaded)


def _relationships_to_json(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _character_starter_scenario_type(
    primary: ScenarioType,
    scenario_types: tuple[ScenarioType, ...],
) -> str:
    if ScenarioType.DATING_SIM in scenario_types:
        return ScenarioType.DATING_SIM.value
    if ScenarioType.FIRST_CONTACT_EXPLORATION in scenario_types:
        return ScenarioType.FIRST_CONTACT_EXPLORATION.value
    if ScenarioType.POLITICAL_INTRIGUE in scenario_types:
        return ScenarioType.POLITICAL_INTRIGUE.value
    return primary.value


def _csv_text(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _starter_generation_existing_names(
    starters: Iterable[ScenarioCharacterStarter],
) -> tuple[str, ...]:
    names: list[str] = []
    for starter in starters:
        names.append(starter.name)
        names.extend(starter.aliases)
    return tuple(name for name in names if name.strip())


def _compact_text_key(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def _seed_initial_character_registry(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    content: Mapping[str, object],
    source_message_id: str | None,
    media_service: MediaService | None = None,
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY,
) -> int:
    content_rating = _scenario_content_mapping_rating(content)
    normalized_genres = scenario_types or (scenario_type,)
    starter_type = _character_starter_scenario_type(
        scenario_type,
        normalized_genres,
    )
    entries = scenario_character_starters_for_content(
        scenario_type=starter_type,
        content=content,
    )
    created_count = 0
    if interaction_mode is InteractionMode.ROLEPLAY:
        created_count = _seed_player_character_from_scenario(
            repositories=repositories,
            save_id=save_id,
            content=content,
            source_message_id=source_message_id,
            content_rating=content_rating,
        )
    if not entries:
        if (
            interaction_mode is InteractionMode.ROLEPLAY
            and ScenarioType.DATING_SIM in normalized_genres
        ):
            DatingRouteService(repositories).seed_routes_for_save(
                save_id,
                source_message_id=source_message_id,
            )
        return created_count

    existing_keys = {
        _character_key(name)
        for character in repositories.list_characters(save_id)
        for name in (character.name, *character.aliases)
        if _character_key(name)
    }
    for entry in entries:
        key = _character_key(entry.name)
        if not key or key in existing_keys:
            continue
        aliases = [
            alias.strip()
            for alias in entry.aliases
            if alias.strip() and _character_key(alias) != key
        ]
        alias_keys = {_character_key(alias) for alias in aliases}
        if alias_keys & existing_keys:
            continue
        character = repositories.add_character(
            save_id=save_id,
            name=entry.name.strip(),
            aliases=aliases,
            role=entry.role.strip(),
            age=entry.age.strip(),
            known_state=entry.known_state.strip(),
            met=entry.met,
            appearance=entry.appearance.strip(),
            visual_notes=entry.visual_notes.strip(),
            personality=entry.personality.strip(),
            voice=entry.voice.strip(),
            texting_style=entry.texting_style.strip(),
            relationships=dict(entry.relationships or {}),
            goals=entry.goals.strip(),
            motivations=entry.motivations.strip(),
            current_intent=entry.current_intent.strip(),
            boundaries=entry.boundaries.strip(),
            attitude_toward_player=(
                entry.attitude_toward_player.strip()
                if interaction_mode is InteractionMode.ROLEPLAY
                else ""
            ),
            cooperation_conditions=entry.cooperation_conditions.strip(),
            status=entry.status.strip() or "present at scenario start",
            source_message_id=source_message_id,
            locked_fields=starter_identity_locked_fields(entry),
            protected_from_maintenance=True,
            content_rating=content_rating,
        )
        if media_service is not None and entry.reference_image is not None:
            try:
                media_service.create_character_reference_from_scenario_starter(
                    save_id=save_id,
                    character_id=character.id,
                    starter=entry,
                )
            except Exception as exc:
                log_error_event(
                    "runtime.scenario_starter_reference_seed_failed",
                    save_id=save_id,
                    character_id=character.id,
                    starter_id=entry.starter_id,
                    starter_name=entry.name,
                    **exception_log_fields(exc),
                )
        existing_keys.add(key)
        existing_keys.update(alias_keys)
        created_count += 1
    if ScenarioType.DATING_SIM in normalized_genres:
        DatingRouteService(repositories).seed_routes_for_save(
            save_id,
            source_message_id=source_message_id,
        )
    return created_count


def _seed_player_character_from_scenario(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    content: Mapping[str, object],
    source_message_id: str | None,
    content_rating: str,
) -> int:
    player_name = _content_text(content, "player_character_name")
    if not player_name:
        return 0
    characters = repositories.list_characters(save_id)
    if any(character.is_player_character for character in characters):
        return 0
    matching = tuple(
        character
        for character in characters
        if _character_matches_name(character, player_name)
    )
    if len(matching) == 1:
        character = matching[0]
        player_profile = _content_text(content, "player_character_profile")
        repositories.update_character(
            replace(
                character,
                known_state=character.known_state or player_profile,
                met=True,
                status=character.status or "present at scenario start",
                protected_from_maintenance=True,
                is_player_character=True,
                content_rating=maximum_content_rating(
                    (character.content_rating, content_rating)
                ),
            )
        )
        return 0
    if matching:
        return 0
    repositories.add_character(
        save_id=save_id,
        name=player_name.strip(),
        role=_content_text(content, "player_role"),
        known_state=_content_text(content, "player_character_profile")
        or "Player character.",
        met=True,
        status="present at scenario start",
        source_message_id=source_message_id,
        protected_from_maintenance=True,
        is_player_character=True,
        content_rating=content_rating,
    )
    return 1


def _scenario_content_mapping_rating(content: Mapping[str, object]) -> str:
    source = content.get("_source")
    if not isinstance(source, Mapping):
        return "unclassified"
    rating = source.get("content_rating")
    return (
        rating.strip()
        if isinstance(rating, str) and rating.strip()
        else "unclassified"
    )


def _character_matches_name(character: CharacterRecord, name: str) -> bool:
    target = _character_key(name)
    if not target:
        return False
    return any(
        _character_key(candidate) == target
        for candidate in (character.name, *character.aliases)
    )


def _scenario_source_metadata_without_loss_conditions(
    metadata: dict[str, object] | None,
) -> dict[str, object] | None:
    if not metadata:
        return None
    sanitized = dict(metadata)
    sanitized.pop("loss_conditions", None)
    return sanitized or None


def _content_text(content: Mapping[str, object], key: str) -> str:
    value = content.get(key)
    return value.strip() if isinstance(value, str) else ""


_SURVIVAL_EXPEDITION_STATE_SEEDS = (
    ("expedition_goal", "expedition.goal", "expedition"),
    ("route_options", "expedition.route", "expedition"),
    ("resource_inventory", "expedition.resources", "inventory"),
    ("environmental_conditions", "expedition.environment", "expedition"),
    ("hazards_and_events", "expedition.hazards", "threat"),
    ("camp_status", "expedition.camp", "expedition"),
    ("travel_progress", "expedition.progress", "objective"),
)


_FIRST_CONTACT_EXPLORATION_STATE_SEEDS = (
    ("mission_profile", "contact.mission", "mission"),
    ("ship_or_base_status", "contact.base", "base"),
    ("exploration_target", "contact.target", "location"),
    ("unknown_intelligence", "contact.intelligence", "contact"),
    ("knowledge_state", "contact.knowledge", "knowledge"),
    ("translation_progress", "contact.translation", "translation"),
    ("discoveries_and_samples", "contact.discoveries", "discovery"),
    ("hazards_and_escalation", "contact.hazards", "threat"),
)


_TIME_LOOP_STATE_SEEDS = (
    ("starting_state", "loop.starting_state", "loop_resettable"),
    ("objective", "loop.objective", "objective"),
    ("baseline_world_state", "loop.baseline", "loop_resettable"),
    ("loop_schedule", "loop.schedule", "loop_schedule"),
    ("persistent_knowledge", "loop.knowledge", "loop_persistent"),
    ("persistence_exceptions", "loop.persistence", "loop_persistent"),
    ("npc_memory_rules", "loop.npc_memory", "loop_boundary"),
    ("current_loop_state", "loop.current", "loop_status"),
)


_HEIST_INFILTRATION_STATE_SEEDS = (
    ("target_location", "heist.target", "location"),
    ("objectives_and_stakes", "heist.objectives", "objective"),
    ("intel_and_access", "heist.intel", "intel"),
    ("security_model", "heist.security", "security"),
    ("alert_and_heat", "heist.alert", "threat"),
    ("loadout_and_tools", "heist.loadout", "inventory"),
    ("complications", "heist.complications", "threat"),
    ("extraction_routes", "heist.extraction", "objective"),
    ("aftermath", "heist.aftermath", "consequence"),
)

_POLITICAL_INTRIGUE_STATE_SEEDS = (
    ("political_arena", "intrigue.arena", "intrigue"),
    ("political_factions", "intrigue.factions", "faction"),
    ("central_conflict", "intrigue.conflict", "objective"),
    ("secrets_and_leverage", "intrigue.secrets", "leverage"),
    ("reputation_and_standing", "intrigue.standing", "reputation"),
    ("obligations_and_favors", "intrigue.obligations", "obligation"),
    ("alliances_and_rivalries", "intrigue.alliances", "relationship"),
    ("event_calendar", "intrigue.calendar", "schedule"),
    ("political_pressure", "intrigue.pressure", "deadline"),
    ("public_private_knowledge", "intrigue.knowledge", "knowledge_boundary"),
)

_SETTLEMENT_BUILDER_STATE_SEEDS = (
    ("settlement_profile", "settlement.profile", "settlement"),
    ("resources_and_indicators", "settlement.resources", "resource"),
    ("projects_and_facilities", "settlement.projects", "project"),
    ("threats_and_opportunities", "settlement.pressures", "threat"),
    ("calendar_and_deadlines", "settlement.calendar", "schedule"),
)

_MONSTER_HUNT_BOUNTY_STATE_SEEDS = (
    ("hunt_profile", "hunt.profile", "hunt"),
    ("target_profile", "hunt.target", "threat"),
    ("leads_and_clues", "hunt.leads", "clue"),
    ("hunt_locations", "hunt.locations", "location"),
    ("preparation_state", "hunt.preparation", "inventory"),
    ("hunt_status", "hunt.status", "objective"),
)

_ROAD_TRIP_PILGRIMAGE_STATE_SEEDS = (
    ("journey_profile", "journey.profile", "journey"),
    ("route_and_stops", "journey.route", "location"),
    ("transport_and_supplies", "journey.supplies", "inventory"),
    ("recurring_pressures", "journey.pressures", "threat"),
    ("relationship_threads", "journey.relationships", "relationship"),
    ("journey_progress", "journey.progress", "objective"),
)

_MERCHANT_TRADE_ROUTE_STATE_SEEDS = (
    ("trade_profile", "trade.profile", "trade"),
    ("cargo_inventory", "trade.cargo", "inventory"),
    ("markets_and_stops", "trade.markets", "location"),
    ("contracts_and_debts", "trade.contracts", "contract"),
    ("route_hazards", "trade.hazards", "threat"),
    ("profit_and_loss", "trade.ledger", "finance"),
)


def _seed_first_contact_exploration_state(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    content: Mapping[str, object],
    source_message_id: str | None,
) -> int:
    if ScenarioType.FIRST_CONTACT_EXPLORATION not in (
        scenario_types or (scenario_type,)
    ):
        return 0
    return _seed_template_section_state(
        repositories=repositories,
        save_id=save_id,
        content=content,
        source_message_id=source_message_id,
        seeds=_FIRST_CONTACT_EXPLORATION_STATE_SEEDS,
    )


def _seed_survival_expedition_state(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    content: Mapping[str, object],
    source_message_id: str | None,
) -> int:
    if ScenarioType.SURVIVAL_EXPEDITION not in (scenario_types or (scenario_type,)):
        return 0
    created_count = 0
    for section_id, key, category in _SURVIVAL_EXPEDITION_STATE_SEEDS:
        summary = _content_text(content, section_id)
        if not summary:
            continue
        repositories.upsert_world_state(
            save_id=save_id,
            key=key,
            value={"summary": summary},
            category=category,
            confidence=1.0,
            source_message_id=source_message_id,
        )
        created_count += 1
    return created_count


def _seed_time_loop_state(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    content: Mapping[str, object],
    source_message_id: str | None,
) -> int:
    if ScenarioType.TIME_LOOP not in (scenario_types or (scenario_type,)):
        return 0
    created_count = 0
    rules_summary = _time_loop_rules_summary(content)
    if rules_summary:
        repositories.upsert_world_state(
            save_id=save_id,
            key="loop.rules",
            value={"summary": rules_summary},
            category="loop_rule",
            confidence=1.0,
            source_message_id=source_message_id,
        )
        created_count += 1
    for section_id, key, category in _TIME_LOOP_STATE_SEEDS:
        summary = _content_text(content, section_id)
        if not summary:
            continue
        repositories.upsert_world_state(
            save_id=save_id,
            key=key,
            value={"summary": summary},
            category=category,
            confidence=1.0,
            source_message_id=source_message_id,
        )
        created_count += 1
    return created_count


def _seed_heist_infiltration_state(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    content: Mapping[str, object],
    source_message_id: str | None,
) -> int:
    if ScenarioType.HEIST_INFILTRATION not in (scenario_types or (scenario_type,)):
        return 0
    created_count = 0
    for section_id, key, category in _HEIST_INFILTRATION_STATE_SEEDS:
        summary = _content_text(content, section_id)
        if not summary:
            continue
        repositories.upsert_world_state(
            save_id=save_id,
            key=key,
            value={"summary": summary},
            category=category,
            confidence=1.0,
            source_message_id=source_message_id,
        )
        created_count += 1
    return created_count


def _seed_political_intrigue_state(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    content: Mapping[str, object],
    source_message_id: str | None,
) -> int:
    if ScenarioType.POLITICAL_INTRIGUE not in (scenario_types or (scenario_type,)):
        return 0
    created_count = 0
    for section_id, key, category in _POLITICAL_INTRIGUE_STATE_SEEDS:
        summary = _content_text(content, section_id)
        if not summary:
            continue
        repositories.upsert_world_state(
            save_id=save_id,
            key=key,
            value={"summary": summary},
            category=category,
            confidence=1.0,
            source_message_id=source_message_id,
        )
        created_count += 1
    return created_count


def _seed_settlement_builder_state(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    content: Mapping[str, object],
    source_message_id: str | None,
) -> int:
    if ScenarioType.SETTLEMENT_BUILDER not in (scenario_types or (scenario_type,)):
        return 0
    return _seed_template_section_state(
        repositories=repositories,
        save_id=save_id,
        content=content,
        source_message_id=source_message_id,
        seeds=_SETTLEMENT_BUILDER_STATE_SEEDS,
    )


def _seed_monster_hunt_bounty_state(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    content: Mapping[str, object],
    source_message_id: str | None,
) -> int:
    if ScenarioType.MONSTER_HUNT_BOUNTY not in (scenario_types or (scenario_type,)):
        return 0
    return _seed_template_section_state(
        repositories=repositories,
        save_id=save_id,
        content=content,
        source_message_id=source_message_id,
        seeds=_MONSTER_HUNT_BOUNTY_STATE_SEEDS,
    )


def _seed_road_trip_pilgrimage_state(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    content: Mapping[str, object],
    source_message_id: str | None,
) -> int:
    if ScenarioType.ROAD_TRIP_PILGRIMAGE not in (scenario_types or (scenario_type,)):
        return 0
    return _seed_template_section_state(
        repositories=repositories,
        save_id=save_id,
        content=content,
        source_message_id=source_message_id,
        seeds=_ROAD_TRIP_PILGRIMAGE_STATE_SEEDS,
    )


def _seed_merchant_trade_route_state(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    content: Mapping[str, object],
    source_message_id: str | None,
) -> int:
    if ScenarioType.MERCHANT_TRADE_ROUTE not in (scenario_types or (scenario_type,)):
        return 0
    return _seed_template_section_state(
        repositories=repositories,
        save_id=save_id,
        content=content,
        source_message_id=source_message_id,
        seeds=_MERCHANT_TRADE_ROUTE_STATE_SEEDS,
    )


def _seed_template_section_state(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    content: Mapping[str, object],
    source_message_id: str | None,
    seeds: tuple[tuple[str, str, str], ...],
) -> int:
    created_count = 0
    for section_id, key, category in seeds:
        summary = _content_text(content, section_id)
        if not summary:
            continue
        repositories.upsert_world_state(
            save_id=save_id,
            key=key,
            value={"summary": summary},
            category=category,
            confidence=1.0,
            source_message_id=source_message_id,
        )
        created_count += 1
    return created_count


def _time_loop_rules_summary(content: Mapping[str, object]) -> str:
    return _join_nonempty_paragraphs(
        _content_text(content, "loop_premise"),
        _prefixed_text("Reset trigger", _content_text(content, "reset_trigger")),
        _prefixed_text("Loop duration", _content_text(content, "loop_duration")),
        _prefixed_text(
            "Failure conditions",
            _content_text(content, "failure_conditions"),
        ),
    )


def _join_nonempty_paragraphs(*values: str) -> str:
    return "\n\n".join(value.strip() for value in values if value.strip())


def _prefixed_text(label: str, value: str) -> str:
    if not value:
        return ""
    return f"{label}: {value}"


def _character_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _active_save(
    repositories: PersistenceRepositories,
    active_save_id: str | None,
) -> SaveRecord | None:
    if active_save_id is None:
        return None
    return repositories.get_save(active_save_id)


def _required_turn_model(turn: SubmittedRuntimeTurn) -> RuntimeModel:
    if turn.model is None:
        raise RuntimeError("Chat turn did not include a runtime model")
    return turn.model


def _save_list_item_model(
    repositories: PersistenceRepositories,
    save: SaveRecord,
    *,
    active_save_id: str | None,
) -> SaveListItemModel:
    supported = not _save_has_retired_scenario(repositories, save.id)
    return SaveListItemModel(
        save_id=save.id,
        title=save.title,
        active=save.id == active_save_id,
        scenario_id=save.scenario_id,
        scenario_title=save.scenario_title,
        created_at=save.created_at,
        updated_at=save.updated_at,
        last_opened_at=save.last_opened_at,
        supported=supported,
        unsupported_reason=None if supported else RETIRED_SCENARIO_REASON,
        interaction_mode=save.interaction_mode,
    )


def _save_has_retired_scenario(
    repositories: PersistenceRepositories,
    save_id: str,
) -> bool:
    save = repositories.get_save(save_id)
    if save is None:
        return False
    scenario = repositories.get_scenario(save.scenario_id)
    return scenario is not None and scenario_record_is_retired(
        scenario.type,
        _scenario_content(scenario.content_json),
    )


def _message_revision_metadata_for_messages(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    messages: Iterable[object],
) -> dict[str, MessageRevisionMetadata]:
    message_ids: list[str] = []
    for message in messages:
        message_id = getattr(message, "id", None)
        if isinstance(message_id, str):
            message_ids.append(message_id)
    if not message_ids:
        return {}
    metadata_for_messages = getattr(
        repositories,
        "message_revision_metadata_for_messages",
        None,
    )
    if callable(metadata_for_messages):
        raw_metadata = metadata_for_messages(save_id, message_ids)
    else:
        visible_ids = set(message_ids)
        raw_metadata = {
            message_id: metadata
            for message_id, metadata in repositories.message_revision_metadata(
                save_id
            ).items()
            if message_id in visible_ids
        }
    return {
        message_id: MessageRevisionMetadata(
            revision_count=int(getattr(metadata, "revision_count", 0)),
            edited_at=cast(str | None, getattr(metadata, "edited_at", None)),
        )
        for message_id, metadata in raw_metadata.items()
    }


def _action_choices_model(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> ActionChoicesModel | None:
    narrator_message = next(
        (
            message
            for message in reversed(repositories.list_messages(save_id))
            if message.role == "narrator" and message.deleted_at is None
        ),
        None,
    )
    if narrator_message is None:
        return None
    choices = repositories.list_message_action_choices(
        save_id,
        message_id=narrator_message.id,
    )
    models = tuple(
        ActionChoiceModel(
            choice_id=choice.id,
            ordinal=choice.ordinal,
            body=choice.body,
            content_rating=choice.content_rating,
        )
        for choice in choices
        if choice.message_id == narrator_message.id
    )
    if models and len(models) != 4:
        return None
    return ActionChoicesModel(
        narrator_message_id=narrator_message.id,
        choices=models,
    )


def _player_speaker_name(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    requested_name: str | None,
) -> str:
    if requested_name is not None:
        text = requested_name.strip()
        if text:
            return text
    details = repositories.load_save_details(save_id)
    if details is not None:
        player_character_name = _default_player_character_name(
            repositories=repositories,
            save_id=save_id,
            scenario=details.scenario,
        )
        if player_character_name:
            return player_character_name
    return "Player"


def _default_player_character_name(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario: object,
) -> str:
    registry_name = _active_registry_player_character_name(
        repositories=repositories,
        save_id=save_id,
    )
    if registry_name:
        return registry_name
    return _active_player_character_name(scenario)


def _active_registry_player_character_name(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> str:
    for character in repositories.list_characters(save_id):
        if character.is_player_character:
            return character.name.strip()
    return ""


def _active_player_character_name(scenario: object) -> str:
    content_json = getattr(scenario, "content_json", "")
    try:
        content = json.loads(content_json)
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(content, dict):
        return ""
    value = content.get("player_character_name")
    return value.strip() if isinstance(value, str) else ""


def _manual_scenario_content(
    *,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...],
    scenario: ManualScenarioInput,
    action_choices_enabled: bool,
) -> dict[str, object]:
    if len(scenario_types) > 1:
        return _manual_hybrid_scenario_content(
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.DATING_SIM:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_character_profile": scenario.player_character_profile.strip(),
                "player_role": scenario.player_role.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.FANTASY_ROLEPLAY:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "magic_system": scenario.magic_system.strip(),
                "realms_and_places": scenario.realms_and_places.strip(),
                "factions_and_orders": scenario.factions_and_orders.strip(),
                "myths_and_creatures": scenario.myths_and_creatures.strip(),
                "quest_stakes": scenario.quest_stakes.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.SCIENCE_FICTION_ROLEPLAY:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "technology_level": scenario.technology_level.strip(),
                "setting_scope": scenario.setting_scope.strip(),
                "species_and_intelligences": (
                    scenario.species_and_intelligences.strip()
                ),
                "factions_and_institutions": (
                    scenario.factions_and_institutions.strip()
                ),
                "mission_stakes": scenario.mission_stakes.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.FIRST_CONTACT_EXPLORATION:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "mission_profile": scenario.mission_profile.strip(),
                "ship_or_base_status": scenario.ship_or_base_status.strip(),
                "exploration_target": scenario.exploration_target.strip(),
                "unknown_intelligence": scenario.unknown_intelligence.strip(),
                "knowledge_state": scenario.knowledge_state.strip(),
                "translation_progress": scenario.translation_progress.strip(),
                "discoveries_and_samples": scenario.discoveries_and_samples.strip(),
                "hazards_and_escalation": scenario.hazards_and_escalation.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.SURVIVAL_EXPEDITION:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "expedition_goal": scenario.expedition_goal.strip(),
                "route_options": scenario.route_options.strip(),
                "resource_inventory": scenario.resource_inventory.strip(),
                "environmental_conditions": (
                    scenario.environmental_conditions.strip()
                ),
                "hazards_and_events": scenario.hazards_and_events.strip(),
                "camp_status": scenario.camp_status.strip(),
                "travel_progress": scenario.travel_progress.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.TIME_LOOP:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "loop_premise": scenario.loop_premise.strip(),
                "reset_trigger": scenario.reset_trigger.strip(),
                "loop_duration": scenario.loop_duration.strip(),
                "starting_state": scenario.starting_state.strip(),
                "objective": scenario.objective.strip(),
                "failure_conditions": scenario.failure_conditions.strip(),
                "baseline_world_state": scenario.baseline_world_state.strip(),
                "loop_schedule": scenario.loop_schedule.strip(),
                "persistent_knowledge": scenario.persistent_knowledge.strip(),
                "persistence_exceptions": scenario.persistence_exceptions.strip(),
                "npc_memory_rules": scenario.npc_memory_rules.strip(),
                "current_loop_state": scenario.current_loop_state.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.INVESTIGATION_MYSTERY:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "case_facts": scenario.case_facts.strip(),
                "clues": scenario.clues.strip(),
                "timeline": scenario.timeline.strip(),
                "red_herrings": scenario.red_herrings.strip(),
                "hidden_truth": scenario.hidden_truth.strip(),
                "case_status": scenario.case_status.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.HEIST_INFILTRATION:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "target_location": scenario.target_location.strip(),
                "objectives_and_stakes": scenario.objectives_and_stakes.strip(),
                "intel_and_access": scenario.intel_and_access.strip(),
                "security_model": scenario.security_model.strip(),
                "alert_and_heat": scenario.alert_and_heat.strip(),
                "loadout_and_tools": scenario.loadout_and_tools.strip(),
                "complications": scenario.complications.strip(),
                "extraction_routes": scenario.extraction_routes.strip(),
                "aftermath": scenario.aftermath.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.POLITICAL_INTRIGUE:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "political_arena": scenario.political_arena.strip(),
                "political_factions": scenario.political_factions.strip(),
                "central_conflict": scenario.central_conflict.strip(),
                "secrets_and_leverage": scenario.secrets_and_leverage.strip(),
                "reputation_and_standing": (
                    scenario.reputation_and_standing.strip()
                ),
                "obligations_and_favors": (
                    scenario.obligations_and_favors.strip()
                ),
                "alliances_and_rivalries": (
                    scenario.alliances_and_rivalries.strip()
                ),
                "event_calendar": scenario.event_calendar.strip(),
                "political_pressure": scenario.political_pressure.strip(),
                "public_private_knowledge": (
                    scenario.public_private_knowledge.strip()
                ),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.SETTLEMENT_BUILDER:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "settlement_profile": scenario.settlement_profile.strip(),
                "resources_and_indicators": (
                    scenario.resources_and_indicators.strip()
                ),
                "projects_and_facilities": (
                    scenario.projects_and_facilities.strip()
                ),
                "threats_and_opportunities": (
                    scenario.threats_and_opportunities.strip()
                ),
                "calendar_and_deadlines": (
                    scenario.calendar_and_deadlines.strip()
                ),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.MONSTER_HUNT_BOUNTY:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "hunt_profile": scenario.hunt_profile.strip(),
                "target_profile": scenario.target_profile.strip(),
                "leads_and_clues": scenario.leads_and_clues.strip(),
                "hunt_locations": scenario.hunt_locations.strip(),
                "preparation_state": scenario.preparation_state.strip(),
                "hunt_status": scenario.hunt_status.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.ROAD_TRIP_PILGRIMAGE:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "journey_profile": scenario.journey_profile.strip(),
                "route_and_stops": scenario.route_and_stops.strip(),
                "transport_and_supplies": (
                    scenario.transport_and_supplies.strip()
                ),
                "recurring_pressures": scenario.recurring_pressures.strip(),
                "relationship_threads": scenario.relationship_threads.strip(),
                "journey_progress": scenario.journey_progress.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.MERCHANT_TRADE_ROUTE:
        return _manual_action_choice_content(
            {
                "title": scenario.title.strip(),
                "premise": scenario.premise.strip(),
                "player_character_name": scenario.player_character_name.strip(),
                "player_role": scenario.player_role.strip(),
                "trade_profile": scenario.trade_profile.strip(),
                "cargo_inventory": scenario.cargo_inventory.strip(),
                "markets_and_stops": scenario.markets_and_stops.strip(),
                "contracts_and_debts": scenario.contracts_and_debts.strip(),
                "route_hazards": scenario.route_hazards.strip(),
                "profit_and_loss": scenario.profit_and_loss.strip(),
                "tone_genre": scenario.tone_genre.strip(),
                "opening_message": scenario.opening_message.strip(),
            },
            scenario=scenario,
            action_choices_enabled=action_choices_enabled,
        )
    return _manual_action_choice_content(
        {
            "title": scenario.title.strip(),
            "premise": scenario.premise.strip(),
            "player_character_name": scenario.player_character_name.strip(),
            "player_role": scenario.player_role.strip(),
            "worldbuilding": scenario.worldbuilding.strip(),
            "lore": scenario.lore.strip(),
            "locations": scenario.locations.strip(),
            "factions": scenario.factions.strip(),
            "tone_genre": scenario.tone_genre.strip(),
            "starting_scene": scenario.starting_scene.strip(),
            "opening_message": scenario.opening_message.strip(),
        },
        scenario=scenario,
        action_choices_enabled=action_choices_enabled,
    )


def _manual_hybrid_scenario_content(
    *,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...],
    scenario: ManualScenarioInput,
    action_choices_enabled: bool,
) -> dict[str, object]:
    content: dict[str, object] = {}
    for section_id in _scenario_allowed_section_ids(
        scenario_type,
        scenario_types=scenario_types,
    ):
        value = getattr(scenario, section_id, None)
        if isinstance(value, str):
            content[section_id] = value.strip()
    content = _manual_action_choice_content(
        content,
        scenario=scenario,
        action_choices_enabled=action_choices_enabled,
    )
    content[SCENARIO_GENRES_CONTENT_KEY] = [
        scenario_type.value for scenario_type in scenario_types
    ]
    return content


def _manual_action_choice_content(
    content: dict[str, object],
    *,
    scenario: ManualScenarioInput,
    action_choices_enabled: bool,
) -> dict[str, object]:
    if action_choices_enabled:
        content["choice_style"] = scenario.choice_style.strip()
    return content_with_action_choices_enabled(
        content,
        enabled=action_choices_enabled,
    )


def _scenario_type_and_action_choices(
    scenario_type: str,
    *,
    action_choices_enabled: bool,
) -> tuple[ScenarioType, bool]:
    requested_type = ScenarioType(scenario_type)
    if requested_type is ScenarioType.CHOOSE_YOUR_OWN_ADVENTURE:
        return ScenarioType.FULL_ROLEPLAY, True
    return requested_type, action_choices_enabled


def _required_text(value: str, label: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _content_safety_transition(result: ContentSafetyResult) -> str:
    if result.body == FADE_TO_BLACK_TRANSITION:
        return FADE_TO_BLACK_TRANSITION_KIND
    if result.body == CONTENT_FILTER_TRANSITION:
        return CONTENT_FILTER_TRANSITION_KIND
    return ""


def _run_coroutine_blocking(coroutine: Coroutine[Any, Any, None]) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coroutine)
        return

    error: list[BaseException] = []

    def run() -> None:
        try:
            asyncio.run(coroutine)
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]


def _user_visible_error(exc: Exception) -> str:
    if isinstance(exc, ProviderError):
        exhausted_attempts = exhausted_retry_attempt_count(exc)
        if exhausted_attempts is not None:
            if exc.category == ProviderErrorCategory.RATE_LIMITED:
                return (
                    "The provider is rate-limited after "
                    f"{exhausted_attempts} attempts. Retry attempts were exhausted; "
                    "wait a moment and try again."
                )
            return (
                "Provider retry attempts were exhausted after "
                f"{exhausted_attempts} attempts ({exc.category.value})."
            )
    if (
        isinstance(exc, ProviderError)
        and exc.category == ProviderErrorCategory.RATE_LIMITED
    ):
        return "The provider is rate-limited. Wait a moment and retry."
    return redact_text(str(exc)) or exc.__class__.__name__


def _chat_completion_used_fallback(
    *,
    repositories: PersistenceRepositories,
    narrator_message_id: str,
) -> bool:
    job = repositories.find_chat_completion_job_for_narrator_message(
        narrator_message_id
    )
    if job is None or not isinstance(job.result, dict):
        return False
    return bool(job.result.get("fallback_used"))


def _committed_source_message_id(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    after_rowid: int | None,
    role: str,
    body: str,
    speaker_name: str | None,
) -> str | None:
    message = repositories.find_active_message_after_rowid(
        save_id,
        after_rowid=after_rowid,
        role=role,
        body=body,
        speaker_name=speaker_name,
    )
    return message.id if message is not None else None


def _missing_chat_requirement(
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    *,
    save_id: str,
) -> str | None:
    chat_preference = _chat_model_preference_for_save(
        repositories=repositories,
        save_id=save_id,
    )
    if chat_preference is None:
        return "No chat model preference configured"
    if chat_preference.provider not in providers:
        return f"Chat provider is unavailable: {chat_preference.provider}"
    if known_model_is_unavailable(
        repositories,
        provider=chat_preference.provider,
        model_id=chat_preference.model_id,
    ):
        return f"Chat model is unavailable: {chat_preference.model_id}"

    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="context_search",
    )
    if preference is None:
        return "No context search model preference configured"
    if preference.provider not in providers:
        return f"Context Search provider is unavailable: {preference.provider}"
    if known_model_is_unavailable(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
    ):
        return f"Context Search model is unavailable: {preference.model_id}"
    return None


def _context_cleanup_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    purpose: str = "context_cleanup",
) -> ModelPreferenceRecord | None:
    purposes = _dedupe_preserving_order((purpose, "context_cleanup", "context_update"))
    return roleplay_model_preference_with_fallbacks(
        repositories=repositories,
        save_id=save_id,
        purposes=purposes,
    )


def _dedupe_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _context_cleanup_preferences(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    purposes: tuple[str, ...],
) -> dict[str, ModelPreferenceRecord]:
    preferences: dict[str, ModelPreferenceRecord] = {}
    for purpose in purposes:
        preference = _context_cleanup_preference(
            repositories=repositories,
            save_id=save_id,
            purpose=purpose,
        )
        if preference is not None:
            preferences[purpose] = preference
    return preferences


def _context_cleanup_tool_call_tasks(
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    task_preferences: Mapping[str, ModelPreferenceRecord],
) -> frozenset[str]:
    tasks: set[str] = set()
    for task, preference in task_preferences.items():
        provider: object = providers.get(preference.provider)
        if not isinstance(provider, ToolCallProvider):
            continue
        if _model_supports_capability(
            repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            capability=ProviderCapability.TOOL_CALLING.value,
        ):
            tasks.add(task)
    return frozenset(tasks)


def _context_update_preference_for_scenario_type(
    *,
    repositories: PersistenceRepositories,
    scenario_type: str,
) -> ModelPreferenceRecord | None:
    for task in (f"{scenario_type}_context_update", "context_update"):
        preference = repositories.get_model_preference(task)
        if preference is not None:
            return preference
    return None


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


def _missing_context_cleanup_requirement(
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    *,
    save_id: str,
    purposes: tuple[str, ...] = ("context_cleanup",),
) -> str | None:
    for purpose in purposes:
        preference = _context_cleanup_preference(
            repositories=repositories,
            save_id=save_id,
            purpose=purpose,
        )
        if preference is None:
            return "No context cleanup model preference configured"
        provider = providers.get(preference.provider)
        if provider is None:
            return f"Context Cleanup provider is unavailable: {preference.provider}"
        if known_model_is_unavailable(
            repositories,
            provider=preference.provider,
            model_id=preference.model_id,
        ):
            return f"Context cleanup model is unavailable: {preference.model_id}"
        provider_supports_structured_output = isinstance(
            provider,
            StructuredOutputProvider,
        )
        provider_supports_tool_calling = isinstance(provider, ToolCallProvider)
        if (
            not provider_supports_structured_output
            and not provider_supports_tool_calling
        ):
            return (
                "Context Cleanup provider does not support structured output "
                "or tool calling"
            )
        supports_structured_output = isinstance(
            provider,
            StructuredOutputProvider,
        ) and _model_supports_capability(
            repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            capability="structured_output",
        )
        supports_tool_calling = isinstance(
            provider,
            ToolCallProvider,
        ) and _model_supports_capability(
            repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            capability=ProviderCapability.TOOL_CALLING.value,
        )
        if not supports_structured_output and not supports_tool_calling:
            return (
                "Context cleanup model does not advertise structured output "
                "or tool calling"
            )
    return None


def _manual_character_maintenance_status(
    result: CharacterMaintenanceResult,
) -> str:
    if result.skipped_reason:
        return f" Character maintenance skipped: {result.skipped_reason}."
    return (
        " Character registry maintenance finished: "
        f"{len(result.proposed)} proposed, "
        f"{len(result.applied)} applied, "
        f"{len(result.rejected)} rejected."
    )


def _model_supports_capability(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
    capability: str,
) -> bool:
    aliases = {
        "structured_output": STRUCTURED_OUTPUT_CAPABILITIES,
        "tool_calling": TOOL_CALLING_CAPABILITIES,
        "image_to_image": IMAGE_TO_IMAGE_CAPABILITIES,
    }.get(capability, frozenset({capability}))
    capability_is_schema_critical = capability in {
        "structured_output",
        "tool_calling",
    }
    supports = (
        model_supports_any_capability
        if capability_is_schema_critical
        else model_supports_any_capability_or_unknown
    )
    return supports(
        repositories,
        provider=provider,
        model_id=model_id,
        required=aliases,
    )


def _chat_model_preference_for_save(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> ModelPreferenceRecord | None:
    return roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="chat",
    )


def _missing_image_generation_requirement(
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    *,
    save_id: str,
) -> str | None:
    for task, label in (("image_generation", "image generation"),):
        preference = roleplay_model_preference(
            repositories=repositories,
            save_id=save_id,
            purpose=task,
        )
        if preference is None:
            return f"No {label} model preference configured"
        if preference.provider not in providers:
            return f"{label.title()} provider is unavailable: {preference.provider}"
        if known_model_is_unavailable(
            repositories,
            provider=preference.provider,
            model_id=preference.model_id,
        ):
            return f"{label.title()} model is unavailable: {preference.model_id}"
    image_prompt_preference = _image_prompt_preference(
        repositories=repositories,
        save_id=save_id,
    )
    if image_prompt_preference is None:
        return "No image prompt model preference configured"
    if image_prompt_preference.provider not in providers:
        return (
            f"Image Prompt provider is unavailable: {image_prompt_preference.provider}"
        )
    return None


def _missing_character_image_requirement(
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    *,
    save_id: str,
) -> str | None:
    preference = image_edit_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=CHARACTER_IMAGE_EDIT_PURPOSE,
    )
    if preference is None:
        return "No image-to-image generation model preference configured"
    if preference.provider not in providers:
        return (
            "Image-to-image generation provider is unavailable: "
            f"{preference.provider}"
        )
    check = check_model_capabilities(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        required=IMAGE_TO_IMAGE_CAPABILITIES,
    )
    if not check.available and check.found:
        return (
            "Image-to-image generation model is unavailable: "
            f"{preference.model_id}"
        )
    if check.found and not check.supported:
        return (
            "Image-to-image generation model does not advertise "
            "image-to-image support"
        )
    return None


def _missing_image_animation_requirement(
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    *,
    save_id: str,
) -> str | None:
    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="image_animation",
    )
    if preference is None:
        return "No image animation model preference configured"
    if preference.provider not in providers:
        return f"Image Animation provider is unavailable: {preference.provider}"
    if not callable(getattr(providers[preference.provider], "generate_video", None)):
        return (
            "Image Animation provider does not support video: "
            f"{preference.provider}"
        )
    check = check_model_capabilities(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        required=IMAGE_TO_VIDEO_CAPABILITIES,
    )
    if check.found:
        if not check.available:
            return f"Image animation model is unavailable: {preference.model_id}"
        if check.supported:
            return None
        return (
            "Image animation model does not support image-to-video: "
            f"{preference.model_id}"
        )
    return None


def _active_save_scenario_type(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> str | None:
    save = repositories.get_save(save_id)
    if save is None:
        return None
    scenario = repositories.get_scenario(save.scenario_id)
    return scenario.type if scenario is not None else None


def _scenario_progress_model(
    progress: ScenarioGenerationProgress,
) -> ScenarioDraftProgressModel:
    return ScenarioDraftProgressModel(
        scenario_type=progress.scenario_type.value,
        section_id=progress.section_id,
        status=progress.status,
        completed_sections=tuple(progress.completed_sections.items()),
        completed_count=progress.completed_count,
        total_count=progress.total_count,
        action_choices_enabled=progress.action_choices_enabled,
        error=progress.error,
        scenario_types=tuple(genre.value for genre in progress.scenario_types),
        interaction_mode=progress.interaction_mode,
    )


def _scenario_generation_preference(
    repositories: PersistenceRepositories,
    *,
    section_id: str | None = None,
) -> ModelPreferenceRecord | None:
    return scenario_generation_model_preference(
        repositories,
        section_id=section_id,
    )


def _image_prompt_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> ModelPreferenceRecord | None:
    return roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="image_prompt",
    )


def _persist_scenario_draft(
    repositories: PersistenceRepositories,
    draft: ScenarioDraft,
    *,
    character_starters: tuple[ScenarioCharacterStarter, ...] | None = None,
) -> str:
    required_sections = _scenario_section_ids(
        draft.type,
        scenario_types=draft.scenario_types,
        action_choices_enabled=draft.action_choices_enabled,
    )
    if draft.interaction_mode is InteractionMode.STORYTELLER:
        required_sections = tuple(
            section_id
            for section_id in required_sections
            if section_id
            not in {
                "player_character_name",
                "player_character_profile",
                "player_role",
            }
        )
    allowed_sections = _scenario_allowed_section_ids(
        draft.type,
        scenario_types=draft.scenario_types,
    )
    optional_sections = _optional_scenario_sections(
        draft.type,
        scenario_types=draft.scenario_types,
    )
    normalized_sections = normalize_scenario_draft_sections(
        draft.type,
        draft.sections,
    )
    extra_sections = set(normalized_sections) - set(allowed_sections)
    if extra_sections:
        raise ValueError(
            f"Scenario draft has unknown sections: {sorted(extra_sections)}"
        )
    missing_sections = [
        section_id
        for section_id in required_sections
        if (
            section_id not in normalized_sections
            and section_id not in optional_sections
        )
    ]
    if missing_sections:
        raise ValueError(f"Scenario draft missing sections: {missing_sections}")
    sections = {
        section_id: normalized_sections.get(section_id, "").strip()
        for section_id in required_sections
    }
    for section_id in allowed_sections:
        if (
            section_id not in sections
            and normalized_sections.get(section_id, "").strip()
        ):
            sections[section_id] = normalized_sections[section_id].strip()
    missing_required_values = [
        section_id
        for section_id in required_sections
        if section_id not in optional_sections and not sections[section_id]
    ]
    if missing_required_values:
        raise ValueError(
            f"Scenario draft has empty required sections: {missing_required_values}"
        )
    content: dict[str, object] = content_with_action_choices_enabled(
        content_with_character_starters(
            scenario_type=_character_starter_scenario_type(
                draft.type,
                draft.scenario_types,
            ),
            content=_content_with_scenario_genres(
                sections,
                draft.scenario_types,
            ),
            starters=draft.character_starters
            if character_starters is None
            else character_starters,
        ),
        enabled=draft.action_choices_enabled,
    )
    if draft.metadata:
        content["_source"] = dict(draft.metadata)
    scenario = repositories.create_scenario(
        type=draft.type.value,
        title=sections["title"],
        premise=sections.get("premise", ""),
        player_role=sections.get("player_role", ""),
        interaction_mode=draft.interaction_mode,
        content=content,
    )
    return scenario.id


def _scenario_section_ids(
    scenario_type: ScenarioType,
    *,
    scenario_types: tuple[ScenarioType, ...] = (),
    action_choices_enabled: bool = False,
) -> tuple[str, ...]:
    normalized_genres = scenario_types or (scenario_type,)
    if len(normalized_genres) > 1:
        return _section_ids_with_choice_style(
            _merged_scenario_section_ids(
                _scenario_section_ids(genre) for genre in normalized_genres
            ),
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.FULL_ROLEPLAY:
        return _section_ids_with_choice_style(
            FULL_ROLEPLAY_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.FANTASY_ROLEPLAY:
        return _section_ids_with_choice_style(
            FANTASY_ROLEPLAY_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.SCIENCE_FICTION_ROLEPLAY:
        return _section_ids_with_choice_style(
            SCIENCE_FICTION_ROLEPLAY_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.FIRST_CONTACT_EXPLORATION:
        return _section_ids_with_choice_style(
            FIRST_CONTACT_EXPLORATION_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.SURVIVAL_EXPEDITION:
        return _section_ids_with_choice_style(
            SURVIVAL_EXPEDITION_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.TIME_LOOP:
        return _section_ids_with_choice_style(
            TIME_LOOP_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.INVESTIGATION_MYSTERY:
        return _section_ids_with_choice_style(
            INVESTIGATION_MYSTERY_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.HEIST_INFILTRATION:
        return _section_ids_with_choice_style(
            HEIST_INFILTRATION_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.POLITICAL_INTRIGUE:
        return _section_ids_with_choice_style(
            POLITICAL_INTRIGUE_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.SETTLEMENT_BUILDER:
        return _section_ids_with_choice_style(
            SETTLEMENT_BUILDER_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.MONSTER_HUNT_BOUNTY:
        return _section_ids_with_choice_style(
            MONSTER_HUNT_BOUNTY_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.ROAD_TRIP_PILGRIMAGE:
        return _section_ids_with_choice_style(
            ROAD_TRIP_PILGRIMAGE_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.MERCHANT_TRADE_ROUTE:
        return _section_ids_with_choice_style(
            MERCHANT_TRADE_ROUTE_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    if scenario_type is ScenarioType.DATING_SIM:
        return _section_ids_with_choice_style(
            DATING_SIM_SECTIONS,
            action_choices_enabled=action_choices_enabled,
        )
    return CHOOSE_YOUR_OWN_ADVENTURE_SECTIONS


def _merged_scenario_section_ids(
    section_groups: Iterable[tuple[str, ...]],
) -> tuple[str, ...]:
    opening_ids = ("tone_genre", "choice_style", "opening_message")
    body: list[str] = []
    opening: list[str] = []
    seen: set[str] = set()
    for section_ids in section_groups:
        for section_id in section_ids:
            if section_id in seen:
                continue
            seen.add(section_id)
            if section_id in opening_ids:
                opening.append(section_id)
            else:
                body.append(section_id)
    ordered_opening = [
        section_id for section_id in opening_ids if section_id in opening
    ]
    return (*body, *ordered_opening)


def _section_ids_with_choice_style(
    section_ids: tuple[str, ...],
    *,
    action_choices_enabled: bool,
) -> tuple[str, ...]:
    if not action_choices_enabled or "choice_style" in section_ids:
        return section_ids
    if "opening_message" not in section_ids:
        return (*section_ids, "choice_style")
    return tuple(
        item
        for section_id in section_ids
        for item in (
            ("choice_style", section_id)
            if section_id == "opening_message"
            else (section_id,)
        )
    )


def _scenario_allowed_section_ids(
    scenario_type: ScenarioType,
    *,
    scenario_types: tuple[ScenarioType, ...] = (),
) -> tuple[str, ...]:
    normalized_genres = scenario_types or (scenario_type,)
    if len(normalized_genres) > 1:
        return _merged_scenario_section_ids(
            _scenario_allowed_section_ids(genre) for genre in normalized_genres
        )
    if scenario_type is ScenarioType.FULL_ROLEPLAY:
        return FULL_ROLEPLAY_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.FANTASY_ROLEPLAY:
        return FANTASY_ROLEPLAY_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.SCIENCE_FICTION_ROLEPLAY:
        return SCIENCE_FICTION_ROLEPLAY_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.FIRST_CONTACT_EXPLORATION:
        return FIRST_CONTACT_EXPLORATION_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.SURVIVAL_EXPEDITION:
        return SURVIVAL_EXPEDITION_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.TIME_LOOP:
        return TIME_LOOP_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.INVESTIGATION_MYSTERY:
        return INVESTIGATION_MYSTERY_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.HEIST_INFILTRATION:
        return HEIST_INFILTRATION_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.POLITICAL_INTRIGUE:
        return POLITICAL_INTRIGUE_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.SETTLEMENT_BUILDER:
        return SETTLEMENT_BUILDER_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.MONSTER_HUNT_BOUNTY:
        return MONSTER_HUNT_BOUNTY_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.ROAD_TRIP_PILGRIMAGE:
        return ROAD_TRIP_PILGRIMAGE_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.MERCHANT_TRADE_ROUTE:
        return MERCHANT_TRADE_ROUTE_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.DATING_SIM:
        return DATING_SIM_ALLOWED_SECTIONS
    return CHOOSE_YOUR_OWN_ADVENTURE_ALLOWED_SECTIONS


_OPTIONAL_SCENARIO_SECTIONS = frozenset({"player_character_name", "relationship_seed"})


def _optional_scenario_sections(
    scenario_type: ScenarioType,
    *,
    scenario_types: tuple[ScenarioType, ...] = (),
) -> frozenset[str]:
    normalized_genres = scenario_types or (scenario_type,)
    optional = {"relationship_seed"}
    if ScenarioType.DATING_SIM not in normalized_genres:
        optional.add("player_character_name")
    return frozenset(optional)


def _model_indicator(repositories: PersistenceRepositories) -> str:
    preference = repositories.get_model_preference("chat")
    if preference is None:
        return "No chat model selected"
    return f"{preference.provider} / {preference.model_id}"


def _image_frequency(repositories: PersistenceRepositories) -> int:
    value = repositories.get_app_setting("image_generation_frequency")
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else DEFAULT_IMAGE_GENERATION_FREQUENCY
    )


def _automatic_image_generation_enabled(
    repositories: PersistenceRepositories,
) -> bool:
    value = repositories.get_app_setting("automatic_image_generation_enabled")
    return (
        value if isinstance(value, bool) else DEFAULT_AUTOMATIC_IMAGE_GENERATION_ENABLED
    )


def _automatic_summarization_enabled(
    repositories: PersistenceRepositories,
) -> bool:
    value = repositories.get_app_setting("automatic_summarization_enabled")
    return value if isinstance(value, bool) else DEFAULT_AUTOMATIC_SUMMARIZATION_ENABLED


def _summary_threshold(repositories: PersistenceRepositories) -> float:
    value = repositories.get_app_setting("summarization_context_pressure_threshold")
    threshold = (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else DEFAULT_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD
    )
    return min(
        max(threshold, MIN_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD),
        MAX_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD,
    )


def _turn_complete_status(
    *,
    fallback_used: bool = False,
    context_trimmed: bool = False,
) -> str:
    notes = ["Turn complete"]
    if fallback_used:
        notes.append("fallback model used")
    if context_trimmed:
        notes.append("context budget trimmed older context")
    return "; ".join(notes)


def _prepared_image_from_coordinator_result(
    coordinator_result: dict[str, object] | None,
) -> dict[str, object] | None:
    if not coordinator_result:
        return None
    for job in cast(Iterable[object], coordinator_result.get("jobs", [])):
        if not isinstance(job, dict) or job.get("name") != "image":
            continue
        result = job.get("result")
        if not isinstance(result, dict) or not result.get(
            "deferred_to_background"
        ):
            continue
        candidate = result.get("prepared_automatic_image")
        if isinstance(candidate, dict):
            return candidate
    return None


async def _submit_player_turn_with_optional_cancellation(
    submit_player_turn: Callable[..., Awaitable[Any]],
    *,
    save_id: str,
    body: str,
    speaker_name: str | None,
    run_post_turn_jobs: bool,
    defer_action_choices: bool = False,
    cancellation_token: CancellationToken,
    current_user_id: str | None = None,
    retry_progress_callback: ProviderRetryProgressCallback | None = None,
    narrator_stream_callback: NarratorStreamCallback | None = None,
    turn_progress_callback: TurnProgressCallback | None = None,
    post_input_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
) -> Any:
    kwargs: dict[str, object] = {
        "save_id": save_id,
        "body": body,
        "speaker_name": speaker_name,
        "run_post_turn_jobs": run_post_turn_jobs,
    }
    accepts_requested = _call_accepts_keyword(
        submit_player_turn,
        "cancellation_requested",
    )
    accepts_token = _call_accepts_keyword(submit_player_turn, "cancellation_token")
    if accepts_requested:
        kwargs["cancellation_requested"] = lambda: cancellation_token.cancelled
    if accepts_token:
        kwargs["cancellation_token"] = cancellation_token
    if _call_accepts_keyword(submit_player_turn, "defer_action_choices"):
        kwargs["defer_action_choices"] = defer_action_choices
    if _call_accepts_keyword(submit_player_turn, "current_user_id"):
        kwargs["current_user_id"] = current_user_id
    if (
        retry_progress_callback is not None
        and _call_accepts_keyword(submit_player_turn, "retry_progress_callback")
    ):
        kwargs["retry_progress_callback"] = retry_progress_callback
    if (
        narrator_stream_callback is not None
        and _call_accepts_keyword(submit_player_turn, "narrator_stream_callback")
    ):
        kwargs["narrator_stream_callback"] = narrator_stream_callback
    if (
        turn_progress_callback is not None
        and _call_accepts_keyword(submit_player_turn, "turn_progress_callback")
    ):
        kwargs["turn_progress_callback"] = turn_progress_callback
    if (
        post_input_context is not None
        and _call_accepts_keyword(submit_player_turn, "post_input_context")
    ):
        kwargs["post_input_context"] = post_input_context
    return await submit_player_turn(**kwargs)


async def _submit_timeskip_turn_with_optional_cancellation(
    submit_timeskip_turn: Callable[..., Awaitable[Any]],
    *,
    save_id: str,
    instruction: str,
    run_post_turn_jobs: bool,
    defer_action_choices: bool = False,
    cancellation_token: CancellationToken,
    current_user_id: str | None = None,
    retry_progress_callback: ProviderRetryProgressCallback | None = None,
    narrator_stream_callback: NarratorStreamCallback | None = None,
    turn_progress_callback: TurnProgressCallback | None = None,
    post_input_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
) -> Any:
    kwargs: dict[str, object] = {
        "save_id": save_id,
        "instruction": instruction,
        "run_post_turn_jobs": run_post_turn_jobs,
    }
    accepts_requested = _call_accepts_keyword(
        submit_timeskip_turn,
        "cancellation_requested",
    )
    accepts_token = _call_accepts_keyword(submit_timeskip_turn, "cancellation_token")
    if accepts_requested:
        kwargs["cancellation_requested"] = lambda: cancellation_token.cancelled
    if accepts_token:
        kwargs["cancellation_token"] = cancellation_token
    if _call_accepts_keyword(submit_timeskip_turn, "defer_action_choices"):
        kwargs["defer_action_choices"] = defer_action_choices
    if _call_accepts_keyword(submit_timeskip_turn, "current_user_id"):
        kwargs["current_user_id"] = current_user_id
    if (
        retry_progress_callback is not None
        and _call_accepts_keyword(submit_timeskip_turn, "retry_progress_callback")
    ):
        kwargs["retry_progress_callback"] = retry_progress_callback
    if (
        narrator_stream_callback is not None
        and _call_accepts_keyword(submit_timeskip_turn, "narrator_stream_callback")
    ):
        kwargs["narrator_stream_callback"] = narrator_stream_callback
    if (
        turn_progress_callback is not None
        and _call_accepts_keyword(submit_timeskip_turn, "turn_progress_callback")
    ):
        kwargs["turn_progress_callback"] = turn_progress_callback
    if (
        post_input_context is not None
        and _call_accepts_keyword(submit_timeskip_turn, "post_input_context")
    ):
        kwargs["post_input_context"] = post_input_context
    return await submit_timeskip_turn(**kwargs)


def _timeskip_body(instruction: str) -> str:
    return timeskip_message_body(instruction)


def _call_accepts_keyword(callback: Callable[..., Any], keyword: str) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == keyword
        for parameter in signature.parameters.values()
    )
