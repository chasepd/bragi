"""Scoped settings scope and role policy."""

from __future__ import annotations

from dataclasses import dataclass

from bragi.retry_policy import PROVIDER_CALL_DEADLINE_SETTING, RETRY_COUNT_SETTING
from bragi.services.agentic_context import (
    AGENTIC_CONTEXT_PIPELINE_SETTING,
    PLAN_FIRST_NARRATOR_SETTING,
)
from bragi.services.character_action_planning_service import (
    CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
    CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
)
from bragi.services.character_text_service import (
    CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
    CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
    CHARACTER_TEXTS_ENABLED_SETTING,
)
from bragi.services.chat_history_settings import (
    NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
)
from bragi.services.content_rating import (
    CONTENT_FILTER_RATING_SETTING,
    FADE_TO_BLACK_ENABLED_SETTING,
)
from bragi.services.director_pressure_service import (
    DIRECTOR_PRESSURE_ENABLED_SETTING,
    DIRECTOR_PRESSURE_GUIDANCE_SETTING,
)
from bragi.services.generation_settings import (
    CHAT_MAX_OUTPUT_TOKENS_ENABLED_SETTING,
    CHAT_MAX_OUTPUT_TOKENS_SETTING,
    CHAT_TEMPERATURE_ENABLED_SETTING,
    CHAT_TEMPERATURE_SETTING,
    IMAGE_DIMENSION_PRESET_SETTING,
    MODEL_THINKING_PREFERENCES_SETTING,
    OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
)
from bragi.services.image_style_settings import IMAGE_STYLE_PRESET_SETTING
from bragi.services.manual_confirmation import (
    MANUAL_CONFIRMATION_CHARACTER_REGISTRY_SETTING,
    MANUAL_CONFIRMATION_MEMORIES_SETTING,
    MANUAL_CONFIRMATION_STATE_CHANGES_SETTING,
)
from bragi.services.model_preferences import ROLEPLAY_SHARED_MODE_SETTING
from bragi.services.model_routing_profiles import MODEL_ROUTING_PROFILES_SETTING
from bragi.services.npc_knowledge_audit_service import (
    NPC_KNOWLEDGE_AUDIT_MODE_SETTING,
)
from bragi.services.openrouter_routing_settings import (
    OPENROUTER_ROUTING_PROFILES_SETTING,
)
from bragi.services.pending_jobs_settings import PENDING_JOBS_DISPLAY_MODE_SETTING
from bragi.services.phrase_denylist import (
    GENERATED_PHRASE_DENYLIST_SETTING,
    SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
)
from bragi.services.post_turn_inference import POST_TURN_INFERENCE_MODE_SETTING
from bragi.services.provider_fallbacks import (
    CHAT_FALLBACK_ENABLED_SETTING,
    STRUCTURED_OUTPUT_FALLBACK_ENABLED_SETTING,
    TOOL_CALL_FALLBACK_ENABLED_SETTING,
)
from bragi.services.scenario_evolution_policy import (
    SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
)
from bragi.services.text_script_policy import SCRIPT_GUARD_MODE_SETTING
from bragi.services.user_narration_guidance import USER_NARRATION_GUIDANCE_SETTING


@dataclass(frozen=True)
class ScopedSettingPolicy:
    scope: str
    admin_only: bool = False
    child_allowed: bool = False


LocalSettingPolicy = ScopedSettingPolicy


_GLOBAL_ADMIN_SCOPED_SETTINGS = frozenset(
    {
        ROLEPLAY_SHARED_MODE_SETTING,
        CHAT_FALLBACK_ENABLED_SETTING,
        STRUCTURED_OUTPUT_FALLBACK_ENABLED_SETTING,
        TOOL_CALL_FALLBACK_ENABLED_SETTING,
        "image_fallback_enabled",
        "video_fallback_enabled",
        "debug_logging_enabled",
        OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
        MODEL_THINKING_PREFERENCES_SETTING,
        OPENROUTER_ROUTING_PROFILES_SETTING,
        MODEL_ROUTING_PROFILES_SETTING,
        GENERATED_PHRASE_DENYLIST_SETTING,
        RETRY_COUNT_SETTING,
        PROVIDER_CALL_DEADLINE_SETTING,
    }
)
_USER_SCOPED_SETTINGS = frozenset(
    {
        CONTENT_FILTER_RATING_SETTING,
        FADE_TO_BLACK_ENABLED_SETTING,
        PENDING_JOBS_DISPLAY_MODE_SETTING,
        USER_NARRATION_GUIDANCE_SETTING,
    }
)
_CHILD_ALLOWED_USER_SCOPED_SETTINGS = frozenset(
    {CONTENT_FILTER_RATING_SETTING, PENDING_JOBS_DISPLAY_MODE_SETTING}
)
_SAVE_SCOPED_SETTINGS = frozenset(
    {
        "automatic_summarization_enabled",
        "summarization_context_pressure_threshold",
        "show_summarization_activity",
        AGENTIC_CONTEXT_PIPELINE_SETTING,
        PLAN_FIRST_NARRATOR_SETTING,
        DIRECTOR_PRESSURE_ENABLED_SETTING,
        DIRECTOR_PRESSURE_GUIDANCE_SETTING,
        CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
        CHARACTER_TEXTS_ENABLED_SETTING,
        CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        POST_TURN_INFERENCE_MODE_SETTING,
        NPC_KNOWLEDGE_AUDIT_MODE_SETTING,
        "automatic_image_generation_enabled",
        "automatic_media_mode",
        "image_generation_frequency",
        "venice_image_safe_mode",
        CHAT_TEMPERATURE_ENABLED_SETTING,
        CHAT_TEMPERATURE_SETTING,
        CHAT_MAX_OUTPUT_TOKENS_ENABLED_SETTING,
        CHAT_MAX_OUTPUT_TOKENS_SETTING,
        IMAGE_DIMENSION_PRESET_SETTING,
        IMAGE_STYLE_PRESET_SETTING,
        NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
        NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
        RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
        RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
        "context_budget_mode",
        "context_budget_fixed_total_chars",
        "context_budget_adaptive_fraction",
        MANUAL_CONFIRMATION_MEMORIES_SETTING,
        MANUAL_CONFIRMATION_CHARACTER_REGISTRY_SETTING,
        MANUAL_CONFIRMATION_STATE_CHANGES_SETTING,
        SCRIPT_GUARD_MODE_SETTING,
        SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
    }
)


def scoped_setting_policy(key: str) -> ScopedSettingPolicy:
    if key in _GLOBAL_ADMIN_SCOPED_SETTINGS:
        return ScopedSettingPolicy(scope="global", admin_only=True)
    if key in _USER_SCOPED_SETTINGS:
        return ScopedSettingPolicy(
            scope="user",
            child_allowed=key in _CHILD_ALLOWED_USER_SCOPED_SETTINGS,
        )
    if key in _SAVE_SCOPED_SETTINGS:
        return ScopedSettingPolicy(scope="save")
    raise ValueError(f"Unknown scoped setting: {key}")


def local_setting_policy(key: str) -> ScopedSettingPolicy:
    return scoped_setting_policy(key)


def role_can_write_scoped_setting(role: str, key: str) -> bool:
    if role == "admin":
        return True
    policy = scoped_setting_policy(key)
    if policy.admin_only:
        return False
    if role == "child":
        return policy.child_allowed
    return role == "user"


def role_can_write_local_setting(role: str, key: str) -> bool:
    return role_can_write_scoped_setting(role, key)
