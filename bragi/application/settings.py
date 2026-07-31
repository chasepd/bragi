"""Import-safe settings view models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from bragi.persistence.models import (
    ModelPreferenceRecord,
    ProviderConfigRecord,
    ProviderModelRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import ProviderGenerationParameter
from bragi.retry_policy import (
    MAX_RETRY_COUNT,
    MIN_RETRY_COUNT,
    RETRY_COUNT_SETTING,
    RETRY_COUNT_STEP,
    sanitize_retry_count,
)
from bragi.services.agentic_context import (
    AGENTIC_CONTEXT_PIPELINE_DEFAULT,
    AGENTIC_CONTEXT_PIPELINE_SETTING,
    PLAN_FIRST_NARRATOR_DEFAULT,
    PLAN_FIRST_NARRATOR_SETTING,
)
from bragi.services.character_action_planning_service import (
    CHARACTER_ACTION_PLANNING_ENABLED_DEFAULT,
    CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
    CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
    MAX_CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY,
    MIN_CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY,
    sanitize_character_action_planning_max_concurrency,
)
from bragi.services.character_text_service import (
    CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
    CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
    CHARACTER_TEXTS_ENABLED_SETTING,
    MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT,
    MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS,
    MIN_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT,
    MIN_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS,
    character_text_proactive_random_chance_percent,
    character_text_proactive_random_cooldown_turns,
)
from bragi.services.chat_history_settings import (
    MAX_RECENT_MESSAGE_WINDOW,
    MIN_RECENT_MESSAGE_WINDOW,
    NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    chat_history_window_settings,
    narrator_planner_chat_history_window_settings,
)
from bragi.services.content_rating import (
    CHILD_ADMIN_CONTENT_RATING_OPTIONS,
    CHILD_SELF_SERVICE_CONTENT_RATING_OPTIONS,
    CONTENT_FILTER_RATING_SETTING,
    CONTENT_RATING_OPTIONS,
    CONTENT_RATING_PG_13,
    FADE_TO_BLACK_ENABLED_SETTING,
    effective_content_safety_policy,
)
from bragi.services.context_assembly import (
    CONTEXT_BUDGET_MODES,
    DEFAULT_CONTEXT_BUDGET_ADAPTIVE_FRACTION,
    DEFAULT_CONTEXT_BUDGET_FIXED_TOTAL_CHARS,
    DEFAULT_CONTEXT_BUDGET_MODE,
    context_budget_settings,
)
from bragi.services.diagnostics_service import redact_diagnostic_text
from bragi.services.director_pressure_service import (
    DIRECTOR_PRESSURE_ENABLED_DEFAULT,
    DIRECTOR_PRESSURE_ENABLED_SETTING,
)
from bragi.services.generation_settings import (
    CHAT_MAX_OUTPUT_TOKENS_ENABLED_SETTING,
    CHAT_MAX_OUTPUT_TOKENS_SETTING,
    CHAT_TEMPERATURE_ENABLED_SETTING,
    CHAT_TEMPERATURE_SETTING,
    IMAGE_DIMENSION_PRESET_SETTING,
    MAX_CHAT_MAX_OUTPUT_TOKENS,
    MAX_CHAT_TEMPERATURE,
    MIN_CHAT_MAX_OUTPUT_TOKENS,
    MIN_CHAT_TEMPERATURE,
    MODEL_THINKING_PREFERENCES_SETTING,
    STEP_CHAT_MAX_OUTPUT_TOKENS,
    STEP_CHAT_TEMPERATURE,
    THINKING_LEVEL_OFF,
    THINKING_LEVEL_PROVIDER_DEFAULT,
    image_dimension_preset_options,
    model_supports_generation_parameter,
    model_thinking_preference_level,
    model_thinking_support,
    sanitize_chat_max_output_tokens,
    sanitize_chat_temperature,
    sanitize_image_dimension_preset,
    selected_image_dimension_preset,
)
from bragi.services.image_style_settings import (
    IMAGE_STYLE_PRESET_SETTING,
    image_style_preset_options,
    selected_image_style_preset,
)
from bragi.services.manual_confirmation import (
    MANUAL_CONFIRMATION_CHARACTER_REGISTRY_SETTING,
    MANUAL_CONFIRMATION_MEMORIES_SETTING,
    MANUAL_CONFIRMATION_STATE_CHANGES_SETTING,
)
from bragi.services.model_preferences import (
    ACTION_CHOICE_GENERATION_PURPOSE,
    CHARACTER_ENHANCEMENT_PURPOSE,
    CHARACTER_IMAGE_EDIT_PURPOSE,
    CHARACTER_INTENT_PLANNING_PURPOSE,
    CHARACTER_PRESENCE_ASSESSMENT_PURPOSE,
    CONTENT_SAFETY_PURPOSE,
    CONTEXT_CLEANUP_ACTIONS_PURPOSE,
    CONTEXT_CLEANUP_SCAN_PURPOSE,
    DATING_ROUTE_PROFILE_PURPOSE,
    DATING_SIM_TYPE,
    FANTASY_ROLEPLAY_TYPE,
    FIRST_CONTACT_EXPLORATION_TYPE,
    FULL_ROLEPLAY_TYPE,
    GUIDED_CONTEXT_CLEANUP_PURPOSE,
    HEIST_INFILTRATION_TYPE,
    IMAGE_EDIT_FALLBACK_PURPOSE,
    IMAGE_TO_IMAGE_GENERATION_PURPOSE,
    INVESTIGATION_MYSTERY_TYPE,
    POLITICAL_INTRIGUE_TYPE,
    ROLEPLAY_MODEL_PURPOSES,
    ROLEPLAY_SHARED_MODE_SETTING,
    ROLEPLAY_SHARED_TYPE,
    ROLEPLAY_TYPES,
    SCENARIO_GENERATION_SECTION_GROUPS,
    SCENE_IMAGE_EDIT_PURPOSE,
    SCIENCE_FICTION_ROLEPLAY_TYPE,
    SURVIVAL_EXPEDITION_TYPE,
    TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
    TIME_LOOP_TYPE,
    model_preference_for_selector,
    roleplay_model_task,
    save_model_override_preference,
    scenario_generation_model_preference,
    scenario_generation_section_id_from_task,
    scenario_generation_section_model_task,
    shared_roleplay_models_enabled,
)
from bragi.services.model_routing_profiles import (
    ModelRoutingProfilesModel,
    model_routing_profiles_model,
)
from bragi.services.npc_knowledge_audit_service import (
    NPC_KNOWLEDGE_AUDIT_MODE_OPTIONS,
    NPC_KNOWLEDGE_AUDIT_MODE_SETTING,
    npc_knowledge_audit_mode,
)
from bragi.services.openrouter_routing_settings import (
    OpenRouterRoutingSettingsModel,
    openrouter_routing_settings_model,
)
from bragi.services.pending_jobs_settings import (
    PENDING_JOBS_DISPLAY_MODE_OPTIONS,
    PENDING_JOBS_DISPLAY_MODE_SETTING,
    sanitize_pending_jobs_display_mode,
)
from bragi.services.phrase_denylist import (
    GENERATED_PHRASE_DENYLIST_SETTING,
    SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
    generated_phrase_denylist_text,
    save_generated_phrase_denylist_text,
)
from bragi.services.post_turn_inference import (
    POST_TURN_INFERENCE_MODE_OPTIONS,
    POST_TURN_INFERENCE_MODE_SETTING,
    post_turn_inference_mode,
)
from bragi.services.settings_policy import scoped_setting_policy
from bragi.services.text_script_policy import (
    SCRIPT_GUARD_MODE_OPTIONS,
    SCRIPT_GUARD_MODE_SETTING,
    script_guard_mode,
)
from bragi.services.user_narration_guidance import (
    USER_NARRATION_GUIDANCE_SETTING,
    sanitize_user_narration_guidance,
)

TASKS = (
    "chat",
    "chat_full_roleplay",
    "chat_fantasy_roleplay",
    "chat_science_fiction_roleplay",
    "chat_first_contact_exploration",
    "chat_survival_expedition",
    "chat_time_loop",
    "chat_investigation_mystery",
    "chat_heist_infiltration",
    "chat_political_intrigue",
    "chat_dating_sim",
    "narrator_fallback",
    "chat_fallback",
    "structured_output_fallback",
    "tool_call_fallback",
    "scenario_generation",
    "context_search",
    "summarization",
    "state_memory",
    "context_update",
    CHARACTER_ENHANCEMENT_PURPOSE,
    "fact_observation",
    "memory_curation",
    "response_planning",
    "response_verification",
    CONTENT_SAFETY_PURPOSE,
    "director_pressure",
    ACTION_CHOICE_GENERATION_PURPOSE,
    CHARACTER_PRESENCE_ASSESSMENT_PURPOSE,
    CHARACTER_INTENT_PLANNING_PURPOSE,
    DATING_ROUTE_PROFILE_PURPOSE,
    "character_action_planning",
    "character_registry_maintenance",
    CONTEXT_CLEANUP_SCAN_PURPOSE,
    CONTEXT_CLEANUP_ACTIONS_PURPOSE,
    GUIDED_CONTEXT_CLEANUP_PURPOSE,
    "context_cleanup",
    "state_pruning",
    "scenario_evolution",
    "npc_knowledge_audit",
    "image_prompt",
    "image_generation",
    IMAGE_TO_IMAGE_GENERATION_PURPOSE,
    SCENE_IMAGE_EDIT_PURPOSE,
    CHARACTER_IMAGE_EDIT_PURPOSE,
    TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
    "video_generation",
    "image_animation",
    "image_fallback",
    IMAGE_EDIT_FALLBACK_PURPOSE,
    "video_fallback",
    "character_image_description",
)

TASK_CAPABILITIES = {
    "chat": "chat",
    "chat_full_roleplay": "chat",
    "chat_fantasy_roleplay": "chat",
    "chat_science_fiction_roleplay": "chat",
    "chat_first_contact_exploration": "chat",
    "chat_survival_expedition": "chat",
    "chat_time_loop": "chat",
    "chat_investigation_mystery": "chat",
    "chat_heist_infiltration": "chat",
    "chat_political_intrigue": "chat",
    "chat_dating_sim": "chat",
    "narrator_fallback": "chat",
    "chat_fallback": "chat",
    "structured_output_fallback": "structured_output",
    "tool_call_fallback": "tool_calling",
    "scenario_generation": "chat",
    "context_search": "structured_output",
    "summarization": "chat",
    "state_memory": "structured_output",
    "context_update": "structured_output",
    CHARACTER_ENHANCEMENT_PURPOSE: "structured_output",
    "fact_observation": "structured_output",
    "memory_curation": "structured_output",
    "response_planning": "structured_output",
    "response_verification": "structured_output",
    CONTENT_SAFETY_PURPOSE: "structured_output",
    "director_pressure": "structured_output",
    ACTION_CHOICE_GENERATION_PURPOSE: "structured_output",
    CHARACTER_PRESENCE_ASSESSMENT_PURPOSE: "structured_output",
    CHARACTER_INTENT_PLANNING_PURPOSE: "structured_output",
    DATING_ROUTE_PROFILE_PURPOSE: "structured_output",
    "character_action_planning": "structured_output",
    "character_registry_maintenance": "structured_output",
    CONTEXT_CLEANUP_SCAN_PURPOSE: "structured_output",
    CONTEXT_CLEANUP_ACTIONS_PURPOSE: "structured_output",
    GUIDED_CONTEXT_CLEANUP_PURPOSE: "structured_output",
    "context_cleanup": "structured_output",
    "state_pruning": "structured_output",
    "scenario_evolution": "structured_output",
    "npc_knowledge_audit": "structured_output",
    "image_prompt": "chat",
    "image_generation": "image_generation",
    IMAGE_TO_IMAGE_GENERATION_PURPOSE: "image_to_image",
    SCENE_IMAGE_EDIT_PURPOSE: "image_to_image",
    CHARACTER_IMAGE_EDIT_PURPOSE: "image_to_image",
    TEXT_MESSAGE_IMAGE_EDIT_PURPOSE: "image_to_image",
    "video_generation": "text_to_video",
    "image_animation": "image_to_video",
    "image_fallback": "image_generation",
    IMAGE_EDIT_FALLBACK_PURPOSE: "image_to_image",
    "video_fallback": "text_to_video",
    "character_image_description": "vision",
}

SCENARIO_CHAT_TASK_FALLBACKS = {
    "chat_full_roleplay": "chat",
    "chat_fantasy_roleplay": "chat",
    "chat_science_fiction_roleplay": "chat",
    "chat_first_contact_exploration": "chat",
    "chat_survival_expedition": "chat",
    "chat_time_loop": "chat",
    "chat_investigation_mystery": "chat",
    "chat_heist_infiltration": "chat",
    "chat_political_intrigue": "chat",
    "chat_dating_sim": "chat",
}

CAPABILITY_ALIASES = {
    "chat": frozenset({"chat", "chat_completion"}),
    "structured_output": frozenset(
        {
            "structured_output",
            "structured",
            "json_schema",
        }
    ),
    "tool_calling": frozenset(
        {
            "tool_calling",
            "tools",
            "function_calling",
        }
    ),
    "image_generation": frozenset({"image_generation", "image"}),
    "image_to_image": frozenset(
        {"image_to_image", "image_edit", "image_editing", "edit", "inpaint"}
    ),
    "text_to_video": frozenset({"text_to_video", "video_generation", "video"}),
    "image_to_video": frozenset(
        {
            "image_to_video",
            "image_plus_text_to_video",
            "image_text_to_video",
            "image_animation",
        }
    ),
    "vision": frozenset(
        {
            "vision",
            "image_input",
            "image_understanding",
            "image_analysis",
            "multimodal",
        }
    ),
}

DEFAULT_IMAGE_GENERATION_FREQUENCY = 3
DEFAULT_AUTOMATIC_SUMMARIZATION_ENABLED = True
DEFAULT_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD = 0.75
MIN_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD = 0.10
MAX_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD = 1.00
STEP_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD = 0.05
DEFAULT_AUTOMATIC_IMAGE_GENERATION_ENABLED = False
DEFAULT_VENICE_IMAGE_SAFE_MODE = True

@dataclass(frozen=True)
class ProviderCardModel:
    provider: str
    enabled: bool
    has_api_key: bool
    model_count: int
    last_model_refresh_at: str | None
    refresh_status: str
    last_error: str | None


@dataclass(frozen=True)
class ModelOption:
    provider: str
    model_id: str
    display_name: str
    available: bool
    capabilities: tuple[str, ...]
    pricing: ModelPricing | None = None
    thinking: ModelThinkingSupport | None = None


@dataclass(frozen=True)
class ModelThinkingSupport:
    levels: tuple[str, ...]
    default_level: str | None = None
    default_enabled: bool | None = None
    mandatory: bool = False
    supports_max_tokens: bool = False


@dataclass(frozen=True)
class ThinkingLevelControl:
    setting_key: str
    task: str
    selected: str
    supported: bool
    options: tuple[str, ...]
    provider: str | None
    model_id: str | None
    default_level: str | None = None
    default_enabled: bool | None = None
    mandatory: bool = False
    disabled_reason: str | None = None


@dataclass(frozen=True)
class ModelPricing:
    input_per_million_tokens_usd: str | None = None
    output_per_million_tokens_usd: str | None = None
    cache_read_per_million_tokens_usd: str | None = None
    cache_write_per_million_tokens_usd: str | None = None
    request_usd: str | None = None
    image_usd: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class TaskModelSelector:
    task: str
    selected_provider: str | None
    selected_model_id: str | None
    selected_available: bool
    warning: str | None
    options: tuple[ModelOption, ...]
    label: str | None = None
    section_id: str | None = None
    inherited_provider: str | None = None
    inherited_model_id: str | None = None
    clearable: bool = False
    thinking: ThinkingLevelControl | None = None


@dataclass(frozen=True)
class RoleplayModelGroup:
    roleplay_type: str
    label: str
    selectors: tuple[TaskModelSelector, ...]


@dataclass(frozen=True)
class ToggleControl:
    setting_key: str
    enabled: bool


@dataclass(frozen=True)
class NumberControl:
    setting_key: str
    value: int | float
    minimum: int | float
    maximum: int | float | None = None
    step: int | float = 1


@dataclass(frozen=True)
class OptionalNumberControl:
    setting_key: str
    enabled_setting_key: str
    enabled: bool
    supported: bool
    value: int | float
    minimum: int | float
    maximum: int | float
    step: int | float = 1


@dataclass(frozen=True)
class FractionControl:
    setting_key: str
    value: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ChoiceControl:
    setting_key: str
    selected: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class ContentRatingControl:
    setting_key: str
    selected: str
    options: tuple[str, ...]
    admin_granted: bool


@dataclass(frozen=True)
class TextControl:
    setting_key: str
    value: str


@dataclass(frozen=True)
class SupportedChoiceControl:
    setting_key: str
    selected: str
    options: tuple[str, ...]
    supported: bool


@dataclass(frozen=True)
class ContextBudgetControls:
    mode: ChoiceControl
    fixed_total_chars: NumberControl
    adaptive_fraction: FractionControl


@dataclass(frozen=True)
class ChatHistoryControls:
    planner_player_messages: NumberControl
    planner_narrator_messages: NumberControl
    player_messages: NumberControl
    narrator_messages: NumberControl


@dataclass(frozen=True)
class ManualConfirmationControls:
    memories: ToggleControl
    character_registry: ToggleControl
    state_changes: ToggleControl


@dataclass(frozen=True)
class DiagnosticEntry:
    kind: str
    error: str | None
    provider: str | None = None
    job_type: str | None = None
    save_id: str | None = None
    path: str | None = None
    retry_summary: str | None = None


@dataclass(frozen=True)
class SettingsModel:
    provider_cards: tuple[ProviderCardModel, ...]
    task_model_selectors: tuple[TaskModelSelector, ...]
    save_model_override_selectors: tuple[TaskModelSelector, ...]
    roleplay_shared_models: ToggleControl | None
    roleplay_model_groups: tuple[RoleplayModelGroup, ...]
    scenario_section_model_selectors: tuple[TaskModelSelector, ...]
    model_routing_profiles: ModelRoutingProfilesModel | None
    retry_count: NumberControl | None
    automatic_summarization: ToggleControl | None
    summarization_context_pressure_threshold: NumberControl | None
    summarization_visibility: ToggleControl | None
    agentic_context_pipeline: ToggleControl | None
    plan_first_narrator: ToggleControl | None
    director_pressure: ToggleControl | None
    character_action_planning: ToggleControl | None
    character_action_planning_max_concurrency: NumberControl | None
    character_texts: ToggleControl | None
    character_text_proactive_random_chance: NumberControl | None
    character_text_proactive_random_cooldown: NumberControl | None
    post_turn_inference_mode: ChoiceControl | None
    npc_knowledge_audit_mode: ChoiceControl | None
    generated_text_script_guard_mode: ChoiceControl | None
    generated_phrase_denylist: TextControl | None
    save_generated_phrase_denylist: TextControl | None
    chat_fallback: ToggleControl | None
    structured_output_fallback: ToggleControl | None
    tool_call_fallback: ToggleControl | None
    image_fallback: ToggleControl | None
    video_fallback: ToggleControl | None
    venice_image_safe_mode: ToggleControl | None
    debug_logging: ToggleControl | None
    pending_jobs_display_mode: ChoiceControl | None
    user_narration_guidance: TextControl | None
    content_rating: ContentRatingControl
    fade_to_black: ToggleControl | None
    automatic_image_generation: ToggleControl | None
    image_style_preset: ChoiceControl | None
    chat_temperature: OptionalNumberControl | None
    chat_max_output_tokens: OptionalNumberControl | None
    image_dimension_preset: SupportedChoiceControl | None
    openrouter_routing: OpenRouterRoutingSettingsModel | None
    automatic_media_mode: ChoiceControl | None
    image_frequency: NumberControl | None
    manual_confirmation: ManualConfirmationControls | None
    chat_history: ChatHistoryControls | None
    context_budget: ContextBudgetControls | None
    secret_storage_warning: str | None
    visible_sections: tuple[str, ...]

    @property
    def summarization_threshold(self) -> NumberControl | None:
        return self.summarization_context_pressure_threshold


@dataclass(frozen=True)
class ProviderSettingsModel:
    provider_cards: tuple[ProviderCardModel, ...]
    secret_storage_warning: str | None


@dataclass(frozen=True)
class LocalSettingsModel:
    pending_jobs_display_mode: ChoiceControl | None
    user_narration_guidance: TextControl | None
    content_rating: ContentRatingControl
    fade_to_black: ToggleControl | None
    debug_logging: ToggleControl | None


def build_provider_settings_model(
    *,
    repositories: PersistenceRepositories,
    providers: tuple[str, ...],
    current_user_role: str | None = None,
    secret_storage_warning: str | None = None,
) -> ProviderSettingsModel:
    is_admin = current_user_role in {None, "admin"}
    return ProviderSettingsModel(
        provider_cards=tuple(
            _provider_card(repositories=repositories, provider=provider)
            for provider in providers
        )
        if is_admin
        else (),
        secret_storage_warning=secret_storage_warning if is_admin else None,
    )


def build_local_settings_model(
    *,
    repositories: PersistenceRepositories,
    current_user_role: str | None = None,
    current_user_id: str | None = None,
) -> LocalSettingsModel:
    is_admin = current_user_role in {None, "admin"}
    is_child = current_user_role == "child"
    content_safety = effective_content_safety_policy(
        repositories,
        user_id=current_user_id,
    )
    child_admin_grant = is_child and content_safety.rating == CONTENT_RATING_PG_13
    return LocalSettingsModel(
        pending_jobs_display_mode=_pending_jobs_display_mode_control(
            repositories,
            current_user_id=current_user_id,
        ),
        user_narration_guidance=TextControl(
            setting_key=USER_NARRATION_GUIDANCE_SETTING,
            value=sanitize_user_narration_guidance(
                _setting_value(
                    repositories,
                    USER_NARRATION_GUIDANCE_SETTING,
                    active_save_id=None,
                    current_user_id=current_user_id,
                )
            ),
        ),
        content_rating=ContentRatingControl(
            setting_key=CONTENT_FILTER_RATING_SETTING,
            selected=content_safety.rating,
            options=(
                CHILD_ADMIN_CONTENT_RATING_OPTIONS
                if child_admin_grant
                else CHILD_SELF_SERVICE_CONTENT_RATING_OPTIONS
            )
            if is_child
            else CONTENT_RATING_OPTIONS,
            admin_granted=child_admin_grant,
        ),
        fade_to_black=None
        if is_child
        else ToggleControl(
            setting_key=FADE_TO_BLACK_ENABLED_SETTING,
            enabled=content_safety.fade_to_black_enabled,
        ),
        debug_logging=ToggleControl(
            setting_key="debug_logging_enabled",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "debug_logging_enabled",
                    active_save_id=None,
                    current_user_id=current_user_id,
                )
            ),
        )
        if is_admin
        else None,
    )


def build_settings_model(
    *,
    repositories: PersistenceRepositories,
    providers: tuple[str, ...],
    active_save_id: str | None = None,
    current_user_role: str | None = None,
    current_user_id: str | None = None,
    log_file_path: Path | None = None,
    secret_storage_warning: str | None = None,
) -> SettingsModel:
    is_admin = current_user_role in {None, "admin"}
    is_user = current_user_role == "user"
    is_child = current_user_role == "child"
    save_controls_visible = is_admin or is_user
    visible_sections = _visible_sections(current_user_role)
    content_safety = effective_content_safety_policy(
        repositories,
        user_id=current_user_id,
    )
    child_admin_grant = is_child and content_safety.rating == CONTENT_RATING_PG_13
    return SettingsModel(
        provider_cards=tuple(
            _provider_card(repositories=repositories, provider=provider)
            for provider in providers
        )
        if is_admin
        else (),
        task_model_selectors=tuple(
            _task_selector(repositories=repositories, task=task) for task in TASKS
        )
        if is_admin
        else (),
        save_model_override_selectors=_save_model_override_selectors(
            repositories=repositories,
            active_save_id=active_save_id,
        )
        if is_admin and active_save_id
        else (),
        roleplay_shared_models=ToggleControl(
            setting_key=ROLEPLAY_SHARED_MODE_SETTING,
            enabled=shared_roleplay_models_enabled(repositories),
        )
        if is_admin
        else None,
        roleplay_model_groups=_roleplay_model_groups(repositories) if is_admin else (),
        scenario_section_model_selectors=_scenario_section_model_selectors(
            repositories
        )
        if is_admin
        else (),
        model_routing_profiles=model_routing_profiles_model(repositories)
        if is_admin
        else None,
        retry_count=NumberControl(
            setting_key=RETRY_COUNT_SETTING,
            value=sanitize_retry_count(
                _setting_value(
                    repositories,
                    RETRY_COUNT_SETTING,
                    active_save_id=None,
                    current_user_id=None,
                )
            ),
            minimum=MIN_RETRY_COUNT,
            maximum=MAX_RETRY_COUNT,
            step=RETRY_COUNT_STEP,
        )
        if is_admin
        else None,
        automatic_summarization=ToggleControl(
            setting_key="automatic_summarization_enabled",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "automatic_summarization_enabled",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                ),
                default=DEFAULT_AUTOMATIC_SUMMARIZATION_ENABLED,
            ),
        )
        if save_controls_visible
        else None,
        summarization_context_pressure_threshold=NumberControl(
            setting_key="summarization_context_pressure_threshold",
            value=_float_setting(
                _setting_value(
                    repositories,
                    "summarization_context_pressure_threshold",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                ),
                default=DEFAULT_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD,
                minimum=MIN_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD,
                maximum=MAX_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD,
            ),
            minimum=MIN_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD,
            maximum=MAX_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD,
            step=STEP_SUMMARIZATION_CONTEXT_PRESSURE_THRESHOLD,
        )
        if save_controls_visible
        else None,
        summarization_visibility=ToggleControl(
            setting_key="show_summarization_activity",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "show_summarization_activity",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                )
            ),
        )
        if save_controls_visible
        else None,
        agentic_context_pipeline=ToggleControl(
            setting_key=AGENTIC_CONTEXT_PIPELINE_SETTING,
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    AGENTIC_CONTEXT_PIPELINE_SETTING,
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                ),
                default=AGENTIC_CONTEXT_PIPELINE_DEFAULT,
            ),
        )
        if save_controls_visible
        else None,
        plan_first_narrator=ToggleControl(
            setting_key=PLAN_FIRST_NARRATOR_SETTING,
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    PLAN_FIRST_NARRATOR_SETTING,
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                ),
                default=PLAN_FIRST_NARRATOR_DEFAULT,
            ),
        )
        if save_controls_visible
        else None,
        director_pressure=ToggleControl(
            setting_key=DIRECTOR_PRESSURE_ENABLED_SETTING,
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    DIRECTOR_PRESSURE_ENABLED_SETTING,
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                ),
                default=DIRECTOR_PRESSURE_ENABLED_DEFAULT,
            ),
        )
        if save_controls_visible
        else None,
        character_action_planning=ToggleControl(
            setting_key=CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                ),
                default=CHARACTER_ACTION_PLANNING_ENABLED_DEFAULT,
            ),
        )
        if save_controls_visible
        else None,
        character_action_planning_max_concurrency=NumberControl(
            setting_key=CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
            value=sanitize_character_action_planning_max_concurrency(
                _setting_value(
                    repositories,
                    CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                )
            ),
            minimum=MIN_CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY,
            maximum=MAX_CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY,
            step=1,
        )
        if save_controls_visible
        else None,
        character_texts=ToggleControl(
            setting_key=CHARACTER_TEXTS_ENABLED_SETTING,
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    CHARACTER_TEXTS_ENABLED_SETTING,
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                ),
                default=_character_texts_default_enabled(
                    repositories,
                    active_save_id=active_save_id,
                ),
            ),
        )
        if save_controls_visible
        else None,
        character_text_proactive_random_chance=NumberControl(
            setting_key=CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
            value=character_text_proactive_random_chance_percent(
                repositories,
                save_id=active_save_id,
            ),
            minimum=MIN_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT,
            maximum=MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT,
            step=1,
        )
        if save_controls_visible
        else None,
        character_text_proactive_random_cooldown=NumberControl(
            setting_key=CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
            value=character_text_proactive_random_cooldown_turns(
                repositories,
                save_id=active_save_id,
            ),
            minimum=MIN_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS,
            maximum=MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS,
            step=1,
        )
        if save_controls_visible
        else None,
        post_turn_inference_mode=ChoiceControl(
            setting_key=POST_TURN_INFERENCE_MODE_SETTING,
            selected=post_turn_inference_mode(
                repositories,
                save_id=active_save_id,
            ),
            options=POST_TURN_INFERENCE_MODE_OPTIONS,
        )
        if save_controls_visible
        else None,
        npc_knowledge_audit_mode=ChoiceControl(
            setting_key=NPC_KNOWLEDGE_AUDIT_MODE_SETTING,
            selected=npc_knowledge_audit_mode(
                repositories,
                save_id=active_save_id,
            ),
            options=NPC_KNOWLEDGE_AUDIT_MODE_OPTIONS,
        )
        if save_controls_visible
        else None,
        generated_text_script_guard_mode=ChoiceControl(
            setting_key=SCRIPT_GUARD_MODE_SETTING,
            selected=script_guard_mode(repositories, save_id=active_save_id),
            options=SCRIPT_GUARD_MODE_OPTIONS,
        )
        if save_controls_visible
        else None,
        generated_phrase_denylist=TextControl(
            setting_key=GENERATED_PHRASE_DENYLIST_SETTING,
            value=generated_phrase_denylist_text(repositories),
        )
        if is_admin
        else None,
        save_generated_phrase_denylist=TextControl(
            setting_key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
            value=save_generated_phrase_denylist_text(
                repositories,
                save_id=active_save_id,
            ),
        )
        if save_controls_visible
        else None,
        chat_fallback=ToggleControl(
            setting_key="chat_fallback_enabled",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "chat_fallback_enabled",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                )
            ),
        )
        if is_admin
        else None,
        structured_output_fallback=ToggleControl(
            setting_key="structured_output_fallback_enabled",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "structured_output_fallback_enabled",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                )
            ),
        )
        if is_admin
        else None,
        tool_call_fallback=ToggleControl(
            setting_key="tool_call_fallback_enabled",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "tool_call_fallback_enabled",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                )
            ),
        )
        if is_admin
        else None,
        image_fallback=ToggleControl(
            setting_key="image_fallback_enabled",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "image_fallback_enabled",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                )
            ),
        )
        if is_admin
        else None,
        video_fallback=ToggleControl(
            setting_key="video_fallback_enabled",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "video_fallback_enabled",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                )
            ),
        )
        if is_admin
        else None,
        venice_image_safe_mode=ToggleControl(
            setting_key="venice_image_safe_mode",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "venice_image_safe_mode",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                ),
                default=DEFAULT_VENICE_IMAGE_SAFE_MODE,
            ),
        )
        if save_controls_visible
        else None,
        debug_logging=ToggleControl(
            setting_key="debug_logging_enabled",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "debug_logging_enabled",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                )
            ),
        )
        if is_admin
        else None,
        pending_jobs_display_mode=_pending_jobs_display_mode_control(
            repositories,
            current_user_id=current_user_id,
        ),
        user_narration_guidance=TextControl(
            setting_key=USER_NARRATION_GUIDANCE_SETTING,
            value=sanitize_user_narration_guidance(
                _setting_value(
                    repositories,
                    USER_NARRATION_GUIDANCE_SETTING,
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                )
            ),
        ),
        content_rating=ContentRatingControl(
            setting_key=CONTENT_FILTER_RATING_SETTING,
            selected=content_safety.rating,
            options=(
                CHILD_ADMIN_CONTENT_RATING_OPTIONS
                if child_admin_grant
                else CHILD_SELF_SERVICE_CONTENT_RATING_OPTIONS
            )
            if is_child
            else CONTENT_RATING_OPTIONS,
            admin_granted=child_admin_grant,
        ),
        fade_to_black=None
        if is_child
        else ToggleControl(
            setting_key=FADE_TO_BLACK_ENABLED_SETTING,
            enabled=content_safety.fade_to_black_enabled,
        ),
        automatic_image_generation=ToggleControl(
            setting_key="automatic_image_generation_enabled",
            enabled=_bool_setting(
                _setting_value(
                    repositories,
                    "automatic_image_generation_enabled",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                ),
                default=DEFAULT_AUTOMATIC_IMAGE_GENERATION_ENABLED,
            ),
        )
        if save_controls_visible
        else None,
        image_style_preset=_image_style_preset_control(
            repositories,
            active_save_id=active_save_id,
        )
        if save_controls_visible
        else None,
        chat_temperature=_chat_temperature_control(
            repositories,
            active_save_id=active_save_id,
        )
        if save_controls_visible
        else None,
        chat_max_output_tokens=_chat_max_output_tokens_control(
            repositories,
            active_save_id=active_save_id,
        )
        if save_controls_visible
        else None,
        image_dimension_preset=_image_dimension_preset_control(
            repositories,
            active_save_id=active_save_id,
        )
        if save_controls_visible
        else None,
        openrouter_routing=openrouter_routing_settings_model(repositories)
        if is_admin
        else None,
        automatic_media_mode=_automatic_media_mode_control(
            repositories,
            active_save_id=active_save_id,
        )
        if save_controls_visible
        else None,
        image_frequency=NumberControl(
            setting_key="image_generation_frequency",
            value=_int_setting(
                _setting_value(
                    repositories,
                    "image_generation_frequency",
                    active_save_id=active_save_id,
                    current_user_id=current_user_id,
                )
            ),
            minimum=0,
            maximum=999,
            step=1,
        )
        if save_controls_visible
        else None,
        manual_confirmation=_manual_confirmation_controls(
            repositories,
            active_save_id=active_save_id,
        )
        if save_controls_visible
        else None,
        chat_history=_chat_history_controls(
            repositories,
            active_save_id=active_save_id,
        )
        if save_controls_visible
        else None,
        context_budget=_context_budget_controls(
            repositories,
            active_save_id=active_save_id,
        )
        if save_controls_visible
        else None,
        secret_storage_warning=secret_storage_warning if is_admin else None,
        visible_sections=visible_sections,
    )


def _provider_card(
    *,
    repositories: PersistenceRepositories,
    provider: str,
) -> ProviderCardModel:
    config = repositories.get_provider_config(provider)
    model_count = repositories.count_provider_models(provider)
    last_error = redact_diagnostic_text(config.last_error) if config else None
    last_model_refresh_at = config.last_model_refresh_at if config else None
    return ProviderCardModel(
        provider=provider,
        enabled=config.enabled if config else False,
        has_api_key=config.has_api_key if config else False,
        model_count=model_count,
        last_model_refresh_at=last_model_refresh_at,
        refresh_status=_provider_refresh_status(
            config=config,
            last_error=last_error,
            model_count=model_count,
        ),
        last_error=last_error,
    )


def _provider_refresh_status(
    *,
    config: ProviderConfigRecord | None,
    last_error: str | None,
    model_count: int,
) -> str:
    if config is None:
        return "Not configured"
    if not config.has_api_key:
        return "No API key"
    if last_error:
        return "Refresh failed"
    if config.last_model_refresh_at:
        return f"Refreshed {config.last_model_refresh_at}"
    if config.enabled and model_count > 0:
        return "Models available; refresh time unknown"
    if config.enabled:
        return "Configured; not refreshed"
    return "Disabled"


def _int_setting(value: object | None) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else DEFAULT_IMAGE_GENERATION_FREQUENCY
    )


def _bool_setting(value: object | None, *, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _character_texts_default_enabled(
    repositories: PersistenceRepositories,
    *,
    active_save_id: str | None,
) -> bool:
    if active_save_id is None:
        return False
    details = repositories.load_save_details(active_save_id)
    return details is not None and details.scenario.type == "dating_sim"


def _visible_sections(current_user_role: str | None) -> tuple[str, ...]:
    if current_user_role in {None, "admin"}:
        return (
            "providers",
            "openrouter",
            "models",
            "save",
            "local",
            "diagnostics",
            "users",
        )
    if current_user_role == "child":
        return ("local",)
    return ("save", "local", "diagnostics")


def _setting_value(
    repositories: PersistenceRepositories,
    key: str,
    *,
    active_save_id: str | None,
    current_user_id: str | None,
) -> object | None:
    try:
        policy = scoped_setting_policy(key)
    except ValueError:
        return repositories.get_effective_setting(key)
    return repositories.get_effective_setting(
        key,
        save_id=active_save_id if policy.scope == "save" else None,
        user_id=current_user_id if policy.scope == "user" else None,
    )


def _float_setting(
    value: object | None,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    number = (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else default
    )
    return min(max(number, minimum), maximum)


def _context_budget_controls(
    repositories: PersistenceRepositories,
    *,
    active_save_id: str | None,
) -> ContextBudgetControls:
    settings = context_budget_settings(repositories, save_id=active_save_id)
    return ContextBudgetControls(
        mode=ChoiceControl(
            setting_key="context_budget_mode",
            selected=settings.mode or DEFAULT_CONTEXT_BUDGET_MODE,
            options=tuple(sorted(CONTEXT_BUDGET_MODES)),
        ),
        fixed_total_chars=NumberControl(
            setting_key="context_budget_fixed_total_chars",
            value=(
                settings.fixed_total_chars or DEFAULT_CONTEXT_BUDGET_FIXED_TOTAL_CHARS
            ),
            minimum=1,
        ),
        adaptive_fraction=FractionControl(
            setting_key="context_budget_adaptive_fraction",
            value=(
                settings.adaptive_fraction or DEFAULT_CONTEXT_BUDGET_ADAPTIVE_FRACTION
            ),
            minimum=0.01,
            maximum=1.0,
        ),
    )


def _chat_history_controls(
    repositories: PersistenceRepositories,
    *,
    active_save_id: str | None,
) -> ChatHistoryControls:
    prose_settings = chat_history_window_settings(repositories, save_id=active_save_id)
    planner_settings = narrator_planner_chat_history_window_settings(
        repositories,
        save_id=active_save_id,
    )
    return ChatHistoryControls(
        planner_player_messages=NumberControl(
            setting_key=NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
            value=planner_settings.player_messages,
            minimum=MIN_RECENT_MESSAGE_WINDOW,
            maximum=MAX_RECENT_MESSAGE_WINDOW,
            step=1,
        ),
        planner_narrator_messages=NumberControl(
            setting_key=NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
            value=planner_settings.narrator_messages,
            minimum=MIN_RECENT_MESSAGE_WINDOW,
            maximum=MAX_RECENT_MESSAGE_WINDOW,
            step=1,
        ),
        player_messages=NumberControl(
            setting_key=RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
            value=prose_settings.player_messages,
            minimum=MIN_RECENT_MESSAGE_WINDOW,
            maximum=MAX_RECENT_MESSAGE_WINDOW,
            step=1,
        ),
        narrator_messages=NumberControl(
            setting_key=RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
            value=prose_settings.narrator_messages,
            minimum=MIN_RECENT_MESSAGE_WINDOW,
            maximum=MAX_RECENT_MESSAGE_WINDOW,
            step=1,
        ),
    )


def _manual_confirmation_controls(
    repositories: PersistenceRepositories,
    *,
    active_save_id: str | None,
) -> ManualConfirmationControls:
    return ManualConfirmationControls(
        memories=ToggleControl(
            setting_key=MANUAL_CONFIRMATION_MEMORIES_SETTING,
            enabled=_bool_setting(
                repositories.get_effective_setting(
                    MANUAL_CONFIRMATION_MEMORIES_SETTING,
                    save_id=active_save_id,
                )
            ),
        ),
        character_registry=ToggleControl(
            setting_key=MANUAL_CONFIRMATION_CHARACTER_REGISTRY_SETTING,
            enabled=_bool_setting(
                repositories.get_effective_setting(
                    MANUAL_CONFIRMATION_CHARACTER_REGISTRY_SETTING,
                    save_id=active_save_id,
                )
            ),
        ),
        state_changes=ToggleControl(
            setting_key=MANUAL_CONFIRMATION_STATE_CHANGES_SETTING,
            enabled=_bool_setting(
                repositories.get_effective_setting(
                    MANUAL_CONFIRMATION_STATE_CHANGES_SETTING,
                    save_id=active_save_id,
                )
            ),
        ),
    )


def _task_selector(
    *,
    repositories: PersistenceRepositories,
    task: str,
) -> TaskModelSelector:
    direct_preference = _normalized_preference(repositories.get_model_preference(task))
    return _task_selector_from_preference(
        repositories=repositories,
        task=task,
        preference=_model_preference_for_selector(repositories, task),
        clearable=direct_preference is not None,
    )


def _task_selector_from_preference(
    *,
    repositories: PersistenceRepositories,
    task: str,
    preference: ModelPreferenceRecord | None,
    label: str | None = None,
    section_id: str | None = None,
    inherited_provider: str | None = None,
    inherited_model_id: str | None = None,
    clearable: bool = False,
    save_id: str | None = None,
) -> TaskModelSelector:
    selected_provider = preference.provider if preference else None
    selected_model_id = preference.model_id if preference else None
    options = tuple(
        _model_option(model)
        for provider in _provider_names(
            repositories,
            preference_provider=selected_provider,
        )
        for model in repositories.list_provider_models(provider)
        if _supports_task(model, task)
    )
    selected_available = _selected_available(
        options=options,
        provider=selected_provider,
        model_id=selected_model_id,
    )
    return TaskModelSelector(
        task=task,
        selected_provider=selected_provider,
        selected_model_id=selected_model_id,
        selected_available=selected_available,
        warning=(
            "Selected model is unavailable"
            if preference is not None and not selected_available
            else None
        ),
        options=options,
        label=label,
        section_id=section_id,
        inherited_provider=inherited_provider,
        inherited_model_id=inherited_model_id,
        clearable=clearable,
        thinking=_thinking_level_control(
            repositories,
            task=task,
            provider=selected_provider,
            model_id=selected_model_id,
            save_id=save_id,
        ),
    )


def _save_model_override_selectors(
    *,
    repositories: PersistenceRepositories,
    active_save_id: str,
) -> tuple[TaskModelSelector, ...]:
    selectors: list[TaskModelSelector] = []
    for task in TASKS:
        override = _normalized_preference(
            save_model_override_preference(
                repositories,
                save_id=active_save_id,
                task=task,
            )
        )
        inherited = _normalized_preference(
            model_preference_for_selector(repositories, task)
        )
        preference = override or inherited
        selectors.append(
            _task_selector_from_preference(
                repositories=repositories,
                task=task,
                preference=preference,
                inherited_provider=inherited.provider if inherited else None,
                inherited_model_id=inherited.model_id if inherited else None,
                clearable=override is not None,
                save_id=active_save_id,
            )
        )
    return tuple(selectors)


def _roleplay_model_groups(
    repositories: PersistenceRepositories,
) -> tuple[RoleplayModelGroup, ...]:
    if shared_roleplay_models_enabled(repositories):
        return (
            RoleplayModelGroup(
                roleplay_type=ROLEPLAY_SHARED_TYPE,
                label="Shared Roleplay",
                selectors=tuple(
                    _task_selector(
                        repositories=repositories,
                        task=roleplay_model_task(
                            roleplay_type=ROLEPLAY_SHARED_TYPE,
                            purpose=purpose,
                        ),
                    )
                    for purpose in ROLEPLAY_MODEL_PURPOSES
                ),
            ),
        )
    return (
        _roleplay_model_group(
            repositories=repositories,
            roleplay_type=FULL_ROLEPLAY_TYPE,
            label="Generic Roleplay",
        ),
        _roleplay_model_group(
            repositories=repositories,
            roleplay_type=FANTASY_ROLEPLAY_TYPE,
            label="Fantasy",
        ),
        _roleplay_model_group(
            repositories=repositories,
            roleplay_type=SCIENCE_FICTION_ROLEPLAY_TYPE,
            label="Science Fiction",
        ),
        _roleplay_model_group(
            repositories=repositories,
            roleplay_type=FIRST_CONTACT_EXPLORATION_TYPE,
            label="First Contact / Exploration",
        ),
        _roleplay_model_group(
            repositories=repositories,
            roleplay_type=SURVIVAL_EXPEDITION_TYPE,
            label="Survival Expedition",
        ),
        _roleplay_model_group(
            repositories=repositories,
            roleplay_type=TIME_LOOP_TYPE,
            label="Time Loop",
        ),
        _roleplay_model_group(
            repositories=repositories,
            roleplay_type=INVESTIGATION_MYSTERY_TYPE,
            label="Investigation Mystery",
        ),
        _roleplay_model_group(
            repositories=repositories,
            roleplay_type=HEIST_INFILTRATION_TYPE,
            label="Heist / Infiltration",
        ),
        _roleplay_model_group(
            repositories=repositories,
            roleplay_type=POLITICAL_INTRIGUE_TYPE,
            label="Political Intrigue",
        ),
        _roleplay_model_group(
            repositories=repositories,
            roleplay_type=DATING_SIM_TYPE,
            label="Dating Sim",
        ),
    )


def _roleplay_model_group(
    *,
    repositories: PersistenceRepositories,
    roleplay_type: str,
    label: str,
) -> RoleplayModelGroup:
    return RoleplayModelGroup(
        roleplay_type=roleplay_type,
        label=label,
        selectors=tuple(
            _task_selector(
                repositories=repositories,
                task=roleplay_model_task(roleplay_type=roleplay_type, purpose=purpose),
            )
            for purpose in ROLEPLAY_MODEL_PURPOSES
        ),
    )


def _scenario_section_model_selectors(
    repositories: PersistenceRepositories,
) -> tuple[TaskModelSelector, ...]:
    inherited = _normalized_preference(
        scenario_generation_model_preference(repositories)
    )
    selectors: list[TaskModelSelector] = []
    for _group_label, section_ids in SCENARIO_GENERATION_SECTION_GROUPS:
        for section_id in section_ids:
            task = scenario_generation_section_model_task(section_id)
            preference = _normalized_preference(repositories.get_model_preference(task))
            selectors.append(
                _task_selector_from_preference(
                    repositories=repositories,
                    task=task,
                    preference=preference,
                    label=_section_display_name(section_id),
                    section_id=section_id,
                    inherited_provider=inherited.provider if inherited else None,
                    inherited_model_id=inherited.model_id if inherited else None,
                    clearable=preference is not None,
                )
            )
    return tuple(selectors)


def _model_preference_for_selector(
    repositories: PersistenceRepositories,
    task: str,
) -> ModelPreferenceRecord | None:
    return _normalized_preference(model_preference_for_selector(repositories, task))


def _selector_fallback_purposes(purpose: str) -> tuple[str, ...]:
    explicit = {
        ACTION_CHOICE_GENERATION_PURPOSE: ("character_action_planning",),
        CHARACTER_PRESENCE_ASSESSMENT_PURPOSE: ("character_action_planning",),
        CHARACTER_INTENT_PLANNING_PURPOSE: ("character_action_planning",),
        CONTEXT_CLEANUP_SCAN_PURPOSE: ("context_cleanup", "context_update"),
        CONTEXT_CLEANUP_ACTIONS_PURPOSE: ("context_cleanup", "context_update"),
        GUIDED_CONTEXT_CLEANUP_PURPOSE: ("context_cleanup", "context_update"),
        "context_cleanup": ("context_update",),
    }.get(purpose, ())
    derived: tuple[str, ...] = ()
    character_enhancement = _character_enhancement_fallback_purpose(purpose)
    if character_enhancement is not None:
        derived = (*derived, character_enhancement)
    image_edit = _image_edit_fallback_purpose(purpose)
    if image_edit is not None:
        derived = (*derived, image_edit)
    return (*explicit, *derived)


def _image_edit_fallback_purpose(purpose: str) -> str | None:
    if purpose in {
        SCENE_IMAGE_EDIT_PURPOSE,
        CHARACTER_IMAGE_EDIT_PURPOSE,
        TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
    }:
        return IMAGE_TO_IMAGE_GENERATION_PURPOSE
    return None


def _character_enhancement_fallback_purpose(purpose: str) -> str | None:
    if purpose == CHARACTER_ENHANCEMENT_PURPOSE:
        return "context_update"
    return None


def _normalized_preference(
    preference: ModelPreferenceRecord | None,
) -> ModelPreferenceRecord | None:
    if preference is None:
        return None
    provider = preference.provider.strip()
    model_id = preference.model_id.strip()
    if not provider or not model_id:
        return None
    return ModelPreferenceRecord(
        id=preference.id,
        task=preference.task,
        provider=provider,
        model_id=model_id,
    )


def _provider_names(
    repositories: PersistenceRepositories,
    *,
    preference_provider: str | None,
) -> tuple[str, ...]:
    names = {config.provider for config in repositories.list_provider_configs()}
    if preference_provider:
        names.add(preference_provider)
    return tuple(sorted(names))


def _model_option(model: ProviderModelRecord) -> ModelOption:
    return ModelOption(
        provider=model.provider,
        model_id=model.model_id,
        display_name=model.display_name,
        available=model.available,
        capabilities=tuple(model.capabilities),
        pricing=_model_pricing(model.pricing),
        thinking=_model_thinking(model.thinking),
    )


def _thinking_level_control(
    repositories: PersistenceRepositories,
    *,
    task: str,
    provider: str | None,
    model_id: str | None,
    save_id: str | None = None,
) -> ThinkingLevelControl:
    if provider is None or model_id is None:
        return ThinkingLevelControl(
            setting_key=MODEL_THINKING_PREFERENCES_SETTING,
            task=task,
            selected=THINKING_LEVEL_PROVIDER_DEFAULT,
            supported=False,
            options=(THINKING_LEVEL_PROVIDER_DEFAULT,),
            provider=provider,
            model_id=model_id,
            disabled_reason="Choose a model first",
        )
    support = model_thinking_support(
        repositories,
        provider=provider,
        model_id=model_id,
    )
    if support is None:
        return ThinkingLevelControl(
            setting_key=MODEL_THINKING_PREFERENCES_SETTING,
            task=task,
            selected=THINKING_LEVEL_PROVIDER_DEFAULT,
            supported=False,
            options=(THINKING_LEVEL_PROVIDER_DEFAULT,),
            provider=provider,
            model_id=model_id,
            disabled_reason="Selected model does not support thinking level",
        )
    levels = _thinking_levels(support)
    mandatory = support.get("mandatory") is True
    off_options: tuple[str, ...] = () if mandatory else (THINKING_LEVEL_OFF,)
    options = (THINKING_LEVEL_PROVIDER_DEFAULT, *off_options, *levels)
    return ThinkingLevelControl(
        setting_key=MODEL_THINKING_PREFERENCES_SETTING,
        task=task,
        selected=model_thinking_preference_level(
            repositories,
            task=task,
            provider=provider,
            model_id=model_id,
            save_id=save_id,
        ),
        supported=True,
        options=options,
        provider=provider,
        model_id=model_id,
        default_level=_optional_text(support.get("default_level")),
        default_enabled=_optional_bool(support.get("default_enabled")),
        mandatory=mandatory,
        disabled_reason=None,
    )


def _model_thinking(thinking: dict[str, object]) -> ModelThinkingSupport | None:
    levels = _thinking_levels(thinking)
    if not levels:
        return None
    return ModelThinkingSupport(
        levels=levels,
        default_level=_optional_text(thinking.get("default_level")),
        default_enabled=_optional_bool(thinking.get("default_enabled")),
        mandatory=thinking.get("mandatory") is True,
        supports_max_tokens=thinking.get("supports_max_tokens") is True,
    )


def _thinking_levels(thinking: Mapping[str, object]) -> tuple[str, ...]:
    levels = thinking.get("levels")
    if not isinstance(levels, list | tuple):
        return ()
    return tuple(
        level.strip().casefold().replace("-", "_")
        for level in levels
        if isinstance(level, str) and level.strip()
    )


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _model_pricing(pricing: dict[str, str]) -> ModelPricing | None:
    if not pricing:
        return None
    model_pricing = ModelPricing(
        input_per_million_tokens_usd=_pricing_field(
            pricing,
            "input_per_million_tokens_usd",
        ),
        output_per_million_tokens_usd=_pricing_field(
            pricing,
            "output_per_million_tokens_usd",
        ),
        cache_read_per_million_tokens_usd=_pricing_field(
            pricing,
            "cache_read_per_million_tokens_usd",
        ),
        cache_write_per_million_tokens_usd=_pricing_field(
            pricing,
            "cache_write_per_million_tokens_usd",
        ),
        request_usd=_pricing_field(pricing, "request_usd"),
        image_usd=_pricing_field(pricing, "image_usd"),
        note=_pricing_field(pricing, "note"),
    )
    return model_pricing if any(model_pricing.__dict__.values()) else None


def _pricing_field(pricing: dict[str, str], key: str) -> str | None:
    value = pricing.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _supports_task(model: ProviderModelRecord, task: str) -> bool:
    purpose = _task_purpose(task)
    if purpose == "chat_fallback":
        return _has_capability(model, "chat")
    if purpose == "structured_output_fallback":
        return _has_capability(model, "structured_output")
    if purpose == "tool_call_fallback":
        return _has_capability(model, "tool_calling")
    if purpose in {
        "context_search",
        "state_memory",
        "context_update",
        CHARACTER_ENHANCEMENT_PURPOSE,
        "fact_observation",
        "memory_curation",
        "response_planning",
        "response_verification",
        "character_registry_maintenance",
        CONTEXT_CLEANUP_SCAN_PURPOSE,
        CONTEXT_CLEANUP_ACTIONS_PURPOSE,
        GUIDED_CONTEXT_CLEANUP_PURPOSE,
        "context_cleanup",
        "state_pruning",
        "scenario_evolution",
    }:
        return _has_capability(model, "structured_output") or _has_capability(
            model,
            "tool_calling",
        )
    if purpose == "image_fallback":
        return _has_capability(model, "image_generation")
    if purpose == IMAGE_EDIT_FALLBACK_PURPOSE:
        return _has_capability(model, "image_to_image")
    if purpose == "video_fallback":
        return _has_capability(model, "text_to_video")
    required_capability = TASK_CAPABILITIES[purpose]
    return _has_capability(model, required_capability)


def _task_purpose(task: str) -> str:
    if scenario_generation_section_id_from_task(task) is not None:
        return "scenario_generation"
    for roleplay_type in ROLEPLAY_TYPES:
        prefix = f"{roleplay_type}_"
        if task.startswith(prefix):
            return task.removeprefix(prefix)
    if task in SCENARIO_CHAT_TASK_FALLBACKS:
        return "chat"
    return task


def _section_display_name(section_id: str) -> str:
    return section_id.replace("_", " ").title()


def _has_capability(model: ProviderModelRecord, capability: str) -> bool:
    aliases = CAPABILITY_ALIASES.get(capability, frozenset({capability}))
    return any(_normalized_capability(value) in aliases for value in model.capabilities)


def _automatic_media_mode_control(
    repositories: PersistenceRepositories,
    *,
    active_save_id: str | None,
) -> ChoiceControl:
    options = ["image"]
    if _has_available_text_to_video_model(repositories):
        options.append("video")
    selected = repositories.get_effective_setting(
        "automatic_media_mode",
        save_id=active_save_id,
    )
    if selected not in options:
        selected = "image"
    return ChoiceControl(
        setting_key="automatic_media_mode",
        selected=str(selected),
        options=tuple(options),
    )


def _pending_jobs_display_mode_control(
    repositories: PersistenceRepositories,
    *,
    current_user_id: str | None,
) -> ChoiceControl:
    return ChoiceControl(
        setting_key=PENDING_JOBS_DISPLAY_MODE_SETTING,
        selected=sanitize_pending_jobs_display_mode(
            repositories.get_effective_setting(
                PENDING_JOBS_DISPLAY_MODE_SETTING,
                user_id=current_user_id,
            )
        ),
        options=PENDING_JOBS_DISPLAY_MODE_OPTIONS,
    )


def _image_style_preset_control(
    repositories: PersistenceRepositories,
    *,
    active_save_id: str | None,
) -> ChoiceControl:
    return ChoiceControl(
        setting_key=IMAGE_STYLE_PRESET_SETTING,
        selected=selected_image_style_preset(
            repositories,
            save_id=active_save_id,
        ),
        options=image_style_preset_options(),
    )


def _chat_temperature_control(
    repositories: PersistenceRepositories,
    *,
    active_save_id: str | None,
) -> OptionalNumberControl:
    return OptionalNumberControl(
        setting_key=CHAT_TEMPERATURE_SETTING,
        enabled_setting_key=CHAT_TEMPERATURE_ENABLED_SETTING,
        enabled=_bool_setting(
            repositories.get_effective_setting(
                CHAT_TEMPERATURE_ENABLED_SETTING,
                save_id=active_save_id,
            )
        ),
        supported=_selected_roleplay_model_supports_parameter(
            repositories,
            purpose="chat",
            parameter=ProviderGenerationParameter.TEMPERATURE,
        ),
        value=sanitize_chat_temperature(
            repositories.get_effective_setting(
                CHAT_TEMPERATURE_SETTING,
                save_id=active_save_id,
            )
        ),
        minimum=MIN_CHAT_TEMPERATURE,
        maximum=MAX_CHAT_TEMPERATURE,
        step=STEP_CHAT_TEMPERATURE,
    )


def _chat_max_output_tokens_control(
    repositories: PersistenceRepositories,
    *,
    active_save_id: str | None,
) -> OptionalNumberControl:
    return OptionalNumberControl(
        setting_key=CHAT_MAX_OUTPUT_TOKENS_SETTING,
        enabled_setting_key=CHAT_MAX_OUTPUT_TOKENS_ENABLED_SETTING,
        enabled=_bool_setting(
            repositories.get_effective_setting(
                CHAT_MAX_OUTPUT_TOKENS_ENABLED_SETTING,
                save_id=active_save_id,
            )
        ),
        supported=_selected_roleplay_model_supports_parameter(
            repositories,
            purpose="chat",
            parameter=ProviderGenerationParameter.MAX_OUTPUT_TOKENS,
        ),
        value=sanitize_chat_max_output_tokens(
            repositories.get_effective_setting(
                CHAT_MAX_OUTPUT_TOKENS_SETTING,
                save_id=active_save_id,
            )
        ),
        minimum=MIN_CHAT_MAX_OUTPUT_TOKENS,
        maximum=MAX_CHAT_MAX_OUTPUT_TOKENS,
        step=STEP_CHAT_MAX_OUTPUT_TOKENS,
    )


def _image_dimension_preset_control(
    repositories: PersistenceRepositories,
    *,
    active_save_id: str | None,
) -> SupportedChoiceControl:
    return SupportedChoiceControl(
        setting_key=IMAGE_DIMENSION_PRESET_SETTING,
        selected=sanitize_image_dimension_preset(
            selected_image_dimension_preset(repositories, save_id=active_save_id)
        ),
        options=image_dimension_preset_options(),
        supported=_selected_roleplay_model_supports_parameter(
            repositories,
            purpose="image_generation",
            parameter=ProviderGenerationParameter.IMAGE_DIMENSIONS,
        ),
    )


def _selected_roleplay_model_supports_parameter(
    repositories: PersistenceRepositories,
    *,
    purpose: str,
    parameter: ProviderGenerationParameter,
) -> bool:
    roleplay_types = (
        (ROLEPLAY_SHARED_TYPE,)
        if shared_roleplay_models_enabled(repositories)
        else ROLEPLAY_TYPES
    )
    return any(
        _selected_task_model_supports_parameter(
            repositories,
            task=roleplay_model_task(roleplay_type=roleplay_type, purpose=purpose),
            parameter=parameter,
        )
        for roleplay_type in roleplay_types
    )


def _selected_task_model_supports_parameter(
    repositories: PersistenceRepositories,
    *,
    task: str,
    parameter: ProviderGenerationParameter,
) -> bool:
    preference = _model_preference_for_selector(repositories, task)
    if preference is None:
        return False
    return model_supports_generation_parameter(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        parameter=parameter,
    )


def _has_available_text_to_video_model(repositories: PersistenceRepositories) -> bool:
    provider_names = {
        config.provider for config in repositories.list_provider_configs()
    }
    preference = repositories.get_model_preference("video_generation")
    if preference is not None:
        provider_names.add(preference.provider)
    return any(
        model.available and _has_capability(model, "text_to_video")
        for provider in provider_names
        for model in repositories.list_provider_models(provider)
    )


def _normalized_capability(value: str) -> str:
    return value.lower().replace("-", "_")


def _selected_available(
    *,
    options: tuple[ModelOption, ...],
    provider: str | None,
    model_id: str | None,
) -> bool:
    return any(
        option.provider == provider and option.model_id == model_id and option.available
        for option in options
    )


def configuration_diagnostics(
    _repositories: PersistenceRepositories,
) -> tuple[DiagnosticEntry, ...]:
    return ()
