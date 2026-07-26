"""Helpers for roleplay-aware model preference lookup."""

from __future__ import annotations

from collections.abc import Mapping

from bragi.model_tasks import is_retired_model_task
from bragi.persistence.models import ModelPreferenceRecord
from bragi.persistence.repositories import PersistenceRepositories

ROLEPLAY_SHARED_MODE_SETTING = "use_shared_roleplay_models"
SAVE_MODEL_OVERRIDES_SETTING = "save_model_overrides"
ROLEPLAY_SHARED_TYPE = "shared"
FULL_ROLEPLAY_TYPE = "full_roleplay"
FANTASY_ROLEPLAY_TYPE = "fantasy_roleplay"
SCIENCE_FICTION_ROLEPLAY_TYPE = "science_fiction_roleplay"
FIRST_CONTACT_EXPLORATION_TYPE = "first_contact_exploration"
SURVIVAL_EXPEDITION_TYPE = "survival_expedition"
TIME_LOOP_TYPE = "time_loop"
INVESTIGATION_MYSTERY_TYPE = "investigation_mystery"
HEIST_INFILTRATION_TYPE = "heist_infiltration"
POLITICAL_INTRIGUE_TYPE = "political_intrigue"
DATING_SIM_TYPE = "dating_sim"
CHOOSE_YOUR_OWN_ADVENTURE_TYPE = "choose_your_own_adventure"

ROLEPLAY_TYPES = (
    FULL_ROLEPLAY_TYPE,
    FANTASY_ROLEPLAY_TYPE,
    SCIENCE_FICTION_ROLEPLAY_TYPE,
    FIRST_CONTACT_EXPLORATION_TYPE,
    SURVIVAL_EXPEDITION_TYPE,
    TIME_LOOP_TYPE,
    INVESTIGATION_MYSTERY_TYPE,
    HEIST_INFILTRATION_TYPE,
    POLITICAL_INTRIGUE_TYPE,
    DATING_SIM_TYPE,
)
NARRATOR_FALLBACK_PURPOSE = "narrator_fallback"
CHAT_FALLBACK_PURPOSE = "chat_fallback"
CHARACTER_ENHANCEMENT_PURPOSE = "character_enhancement"
ACTION_CHOICE_GENERATION_PURPOSE = "action_choice_generation"
CHARACTER_PRESENCE_ASSESSMENT_PURPOSE = "character_presence_assessment"
CHARACTER_INTENT_PLANNING_PURPOSE = "character_intent_planning"
DATING_ROUTE_PROFILE_PURPOSE = "dating_route_profile"
CONTEXT_CLEANUP_SCAN_PURPOSE = "context_cleanup_scan"
CONTEXT_CLEANUP_ACTIONS_PURPOSE = "context_cleanup_actions"
GUIDED_CONTEXT_CLEANUP_PURPOSE = "guided_context_cleanup"
IMAGE_TO_IMAGE_GENERATION_PURPOSE = "image_to_image_generation"
SCENE_IMAGE_EDIT_PURPOSE = "scene_image_edit_generation"
CHARACTER_IMAGE_EDIT_PURPOSE = "character_image_edit_generation"
TEXT_MESSAGE_IMAGE_EDIT_PURPOSE = "text_message_image_edit_generation"
IMAGE_EDIT_FALLBACK_PURPOSE = "image_edit_fallback"
IMAGE_EDIT_FLOW_PURPOSES = frozenset(
    {
        SCENE_IMAGE_EDIT_PURPOSE,
        CHARACTER_IMAGE_EDIT_PURPOSE,
        TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
    }
)

ROLEPLAY_MODEL_PURPOSES = (
    "chat",
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
    NARRATOR_FALLBACK_PURPOSE,
    CHAT_FALLBACK_PURPOSE,
    "structured_output_fallback",
    "tool_call_fallback",
    "image_fallback",
    IMAGE_EDIT_FALLBACK_PURPOSE,
    "video_fallback",
)

SCENARIO_GENERATION_SECTION_MODEL_TASK_PREFIX = "scenario_generation_section_"
EXTRACTION_TOOL_PRIMARY_PROVIDER = "openrouter"
EXTRACTION_TOOL_PRIMARY_MODEL_ID = "deepseek/deepseek-v4-flash"
EXTRACTION_TOOL_FALLBACK_PROVIDER = "venice"
EXTRACTION_TOOL_FALLBACK_MODEL_ID = "qwen3-5-9b"

_EXTRACTION_TOOL_PURPOSES = frozenset({"state_memory", "context_update"})
_AGENTIC_STRUCTURED_PURPOSES = frozenset(
    {
        "fact_observation",
        "memory_curation",
        "response_planning",
        "response_verification",
        "director_pressure",
        "character_action_planning",
    }
)
_TOOL_CALL_CAPABILITIES = frozenset(
    {
        "tool_calling",
        "tools",
        "function_calling",
    }
)

SCENARIO_GENERATION_SECTION_GROUPS = (
    (
        "Common",
        (
            "title",
            "premise",
            "player_character_name",
            "player_role",
            "choice_style",
            "opening_message",
        ),
    ),
    (
        "Generic Roleplay",
        (
            "worldbuilding",
            "lore",
            "locations",
            "factions",
            "tone_genre",
            "current_scene",
        ),
    ),
    (
        "Fantasy",
        (
            "magic_system",
            "realms_and_places",
            "factions_and_orders",
            "myths_and_creatures",
            "quest_stakes",
        ),
    ),
    (
        "Science Fiction",
        (
            "technology_level",
            "setting_scope",
            "species_and_intelligences",
            "factions_and_institutions",
            "mission_stakes",
        ),
    ),
    (
        "First Contact / Exploration",
        (
            "mission_profile",
            "ship_or_base_status",
            "exploration_target",
            "unknown_intelligence",
            "knowledge_state",
            "translation_progress",
            "discoveries_and_samples",
            "hazards_and_escalation",
        ),
    ),
    (
        "Survival Expedition",
        (
            "expedition_goal",
            "route_options",
            "resource_inventory",
            "environmental_conditions",
            "hazards_and_events",
            "camp_status",
            "travel_progress",
        ),
    ),
    (
        "Time Loop",
        (
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
        ),
    ),
    (
        "Investigation Mystery",
        (
            "case_facts",
            "clues",
            "timeline",
            "red_herrings",
            "hidden_truth",
            "case_status",
        ),
    ),
    (
        "Heist / Infiltration",
        (
            "target_location",
            "objectives_and_stakes",
            "intel_and_access",
            "security_model",
            "alert_and_heat",
            "loadout_and_tools",
            "complications",
            "extraction_routes",
            "aftermath",
        ),
    ),
    (
        "Political Intrigue",
        (
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
        ),
    ),
    (
        "Settlement Builder",
        (
            "settlement_profile",
            "resources_and_indicators",
            "projects_and_facilities",
            "threats_and_opportunities",
            "calendar_and_deadlines",
        ),
    ),
    (
        "Monster Hunt / Bounty",
        (
            "hunt_profile",
            "target_profile",
            "leads_and_clues",
            "hunt_locations",
            "preparation_state",
            "hunt_status",
        ),
    ),
    (
        "Road Trip / Pilgrimage",
        (
            "journey_profile",
            "route_and_stops",
            "transport_and_supplies",
            "recurring_pressures",
            "relationship_threads",
            "journey_progress",
        ),
    ),
    (
        "Merchant / Trade Route",
        (
            "trade_profile",
            "cargo_inventory",
            "markets_and_stops",
            "contracts_and_debts",
            "route_hazards",
            "profit_and_loss",
        ),
    ),
    (
        "Dating Sim",
        (
            "player_character_profile",
        ),
    ),
)

SCENARIO_GENERATION_SECTION_IDS = tuple(
    section_id
    for _group_label, section_ids in SCENARIO_GENERATION_SECTION_GROUPS
    for section_id in section_ids
)
_SCENARIO_GENERATION_SECTION_ID_SET = frozenset(SCENARIO_GENERATION_SECTION_IDS)

_ROLEPLAY_CHAT_TASKS = {
    FULL_ROLEPLAY_TYPE: "chat_full_roleplay",
    FANTASY_ROLEPLAY_TYPE: "chat_fantasy_roleplay",
    SCIENCE_FICTION_ROLEPLAY_TYPE: "chat_science_fiction_roleplay",
    FIRST_CONTACT_EXPLORATION_TYPE: "chat_first_contact_exploration",
    SURVIVAL_EXPEDITION_TYPE: "chat_survival_expedition",
    TIME_LOOP_TYPE: "chat_time_loop",
    INVESTIGATION_MYSTERY_TYPE: "chat_investigation_mystery",
    HEIST_INFILTRATION_TYPE: "chat_heist_infiltration",
    POLITICAL_INTRIGUE_TYPE: "chat_political_intrigue",
    DATING_SIM_TYPE: "chat_dating_sim",
    CHOOSE_YOUR_OWN_ADVENTURE_TYPE: "chat_choose_your_own_adventure",
}
_SCENARIO_CHAT_TASK_FALLBACKS = {
    value: "chat" for value in _ROLEPLAY_CHAT_TASKS.values()
}
_THINKING_LEVEL_VALUES = frozenset(
    {
        "provider_default",
        "off",
        "max",
        "xhigh",
        "high",
        "medium",
        "low",
        "minimal",
        "none",
    }
)


def sanitize_save_model_overrides(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    preferences = _sanitize_model_override_preferences(value.get("preferences"))
    thinking = _sanitize_model_override_thinking(value.get("thinking"))
    sanitized: dict[str, object] = {}
    if preferences:
        sanitized["preferences"] = preferences
    if thinking:
        sanitized["thinking"] = thinking
    return sanitized


def save_model_override_preference(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    task: str,
) -> ModelPreferenceRecord | None:
    preferences = save_model_override_preferences(repositories, save_id=save_id)
    value = preferences.get(task.strip())
    if value is None:
        return None
    return ModelPreferenceRecord(
        id=f"save_override:{save_id}:{task.strip()}",
        task=task.strip(),
        provider=value["provider"],
        model_id=value["model_id"],
    )


def save_model_override_preferences(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> dict[str, dict[str, str]]:
    overrides = _save_model_overrides(repositories, save_id=save_id)
    preferences = overrides.get("preferences")
    return dict(preferences) if isinstance(preferences, dict) else {}


def set_save_model_override_preference(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    task: str,
    provider: str,
    model_id: str,
) -> ModelPreferenceRecord:
    normalized_task = _required_text(task, "task")
    normalized_provider = _required_text(provider, "provider").casefold()
    normalized_model_id = _required_text(model_id, "model_id")
    overrides = _save_model_overrides(repositories, save_id=save_id)
    preferences = _override_preferences_from_settings(overrides)
    preferences[normalized_task] = {
        "provider": normalized_provider,
        "model_id": normalized_model_id,
    }
    overrides["preferences"] = preferences
    _write_save_model_overrides(repositories, save_id=save_id, overrides=overrides)
    return ModelPreferenceRecord(
        id=f"save_override:{save_id}:{normalized_task}",
        task=normalized_task,
        provider=normalized_provider,
        model_id=normalized_model_id,
    )


def clear_save_model_override_preference(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    task: str,
) -> None:
    normalized_task = task.strip()
    overrides = _save_model_overrides(repositories, save_id=save_id)
    preferences = _override_preferences_from_settings(overrides)
    preferences.pop(normalized_task, None)
    if preferences:
        overrides["preferences"] = preferences
    else:
        overrides.pop("preferences", None)
    _write_save_model_overrides(repositories, save_id=save_id, overrides=overrides)


def save_model_thinking_preference(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    task: str,
) -> dict[str, str] | None:
    thinking = save_model_thinking_preferences(repositories, save_id=save_id)
    return thinking.get(task.strip())


def save_model_thinking_preferences(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> dict[str, dict[str, str]]:
    overrides = _save_model_overrides(repositories, save_id=save_id)
    thinking = overrides.get("thinking")
    return dict(thinking) if isinstance(thinking, dict) else {}


def set_save_model_thinking_preference(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    task: str,
    provider: str,
    model_id: str,
    level: str,
) -> None:
    normalized_task = _required_text(task, "task")
    normalized_provider = _required_text(provider, "provider").casefold()
    normalized_model_id = _required_text(model_id, "model_id")
    normalized_level = _sanitize_thinking_level(level)
    if normalized_level is None:
        raise ValueError("Unknown thinking level")
    overrides = _save_model_overrides(repositories, save_id=save_id)
    thinking = _override_thinking_from_settings(overrides)
    thinking[normalized_task] = {
        "provider": normalized_provider,
        "model_id": normalized_model_id,
        "level": normalized_level,
    }
    overrides["thinking"] = thinking
    _write_save_model_overrides(repositories, save_id=save_id, overrides=overrides)


def clear_save_model_thinking_preference(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    task: str,
) -> None:
    normalized_task = task.strip()
    overrides = _save_model_overrides(repositories, save_id=save_id)
    thinking = _override_thinking_from_settings(overrides)
    thinking.pop(normalized_task, None)
    if thinking:
        overrides["thinking"] = thinking
    else:
        overrides.pop("thinking", None)
    _write_save_model_overrides(repositories, save_id=save_id, overrides=overrides)


def model_preference_for_selector(
    repositories: PersistenceRepositories,
    task: str,
    *,
    save_id: str | None = None,
) -> ModelPreferenceRecord | None:
    preference = _preference_for_task(repositories, task, save_id=save_id)
    if preference is not None:
        return preference
    for roleplay_type in ROLEPLAY_TYPES:
        prefix = f"{roleplay_type}_"
        if task.startswith(prefix):
            purpose = task.removeprefix(prefix)
            shared_preference = _preference_for_task(
                repositories,
                purpose,
                save_id=save_id,
            )
            if shared_preference is not None:
                return shared_preference
            for fallback_purpose in _selector_fallback_purposes(purpose):
                roleplay_fallback = _preference_for_task(
                    repositories,
                    roleplay_model_task(
                        roleplay_type=roleplay_type,
                        purpose=fallback_purpose,
                    ),
                    save_id=save_id,
                )
                if roleplay_fallback is not None:
                    return roleplay_fallback
                shared_fallback = _preference_for_task(
                    repositories,
                    fallback_purpose,
                    save_id=save_id,
                )
                if shared_fallback is not None:
                    return shared_fallback
            return None
    for fallback_task in _selector_fallback_purposes(task):
        fallback_preference = _preference_for_task(
            repositories,
            fallback_task,
            save_id=save_id,
        )
        if fallback_preference is not None:
            return fallback_preference
    scenario_fallback_task = _SCENARIO_CHAT_TASK_FALLBACKS.get(task)
    if scenario_fallback_task is None:
        return None
    return _preference_for_task(
        repositories,
        scenario_fallback_task,
        save_id=save_id,
    )


def shared_roleplay_models_enabled(repositories: PersistenceRepositories) -> bool:
    value = repositories.get_app_setting(ROLEPLAY_SHARED_MODE_SETTING)
    if isinstance(value, bool):
        return value
    return not _has_legacy_scenario_chat_preference(repositories)


def roleplay_model_task(*, roleplay_type: str, purpose: str) -> str:
    if roleplay_type == ROLEPLAY_SHARED_TYPE:
        return purpose
    if purpose == "scenario_generation":
        return purpose
    if purpose == "chat":
        return _ROLEPLAY_CHAT_TASKS.get(roleplay_type, purpose)
    return f"{roleplay_type}_{purpose}"


def roleplay_model_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    purpose: str,
) -> ModelPreferenceRecord | None:
    if shared_roleplay_models_enabled(repositories):
        return _shared_preference(repositories, purpose, save_id=save_id)

    scenario_type = _scenario_type_for_save(repositories=repositories, save_id=save_id)
    if scenario_type in ROLEPLAY_TYPES:
        preference = _preference_for_task(
            repositories,
            roleplay_model_task(roleplay_type=scenario_type, purpose=purpose),
            save_id=save_id,
        )
        if preference is not None:
            return preference
    return _shared_preference(repositories, purpose, save_id=save_id)


def roleplay_model_preference_with_fallbacks(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    purposes: tuple[str, ...],
) -> ModelPreferenceRecord | None:
    for purpose in purposes:
        preference = roleplay_model_preference(
            repositories=repositories,
            save_id=save_id,
            purpose=purpose,
        )
        if preference is not None:
            return preference
    return None


def image_edit_model_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    purpose: str,
) -> ModelPreferenceRecord | None:
    if purpose not in IMAGE_EDIT_FLOW_PURPOSES:
        raise ValueError(f"Unknown image edit flow purpose: {purpose}")
    return roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=purpose,
    ) or roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=IMAGE_TO_IMAGE_GENERATION_PURPOSE,
    )


def character_enhancement_model_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> ModelPreferenceRecord | None:
    return roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=CHARACTER_ENHANCEMENT_PURPOSE,
    ) or roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="context_update",
    )


def scenario_generation_section_model_task(section_id: str) -> str:
    normalized = section_id.strip()
    if normalized not in _SCENARIO_GENERATION_SECTION_ID_SET:
        raise ValueError(f"Unknown scenario generation section: {section_id}")
    return f"{SCENARIO_GENERATION_SECTION_MODEL_TASK_PREFIX}{normalized}"


def scenario_generation_section_id_from_task(task: str) -> str | None:
    if not task.startswith(SCENARIO_GENERATION_SECTION_MODEL_TASK_PREFIX):
        return None
    section_id = task.removeprefix(SCENARIO_GENERATION_SECTION_MODEL_TASK_PREFIX)
    if section_id not in _SCENARIO_GENERATION_SECTION_ID_SET:
        return None
    return section_id


def scenario_generation_section_model_preference(
    repositories: PersistenceRepositories,
    *,
    section_id: str,
) -> ModelPreferenceRecord | None:
    return repositories.get_model_preference(
        scenario_generation_section_model_task(section_id)
    )


def scenario_generation_model_preference(
    repositories: PersistenceRepositories,
    *,
    section_id: str | None = None,
) -> ModelPreferenceRecord | None:
    if section_id is not None:
        preference = scenario_generation_section_model_preference(
            repositories,
            section_id=section_id,
        )
        if preference is not None:
            return preference
    return repositories.get_model_preference(
        "scenario_generation",
    ) or repositories.get_model_preference("chat")


def narrator_fallback_model_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> ModelPreferenceRecord | None:
    return roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=NARRATOR_FALLBACK_PURPOSE,
    ) or roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=CHAT_FALLBACK_PURPOSE,
    )


def _shared_preference(
    repositories: PersistenceRepositories,
    purpose: str,
    *,
    save_id: str | None = None,
) -> ModelPreferenceRecord | None:
    preference = _preference_for_task(repositories, purpose, save_id=save_id)
    if preference is not None:
        return preference
    if purpose in _EXTRACTION_TOOL_PURPOSES:
        return _recommended_tool_model_preference(
            repositories,
            task=purpose,
            provider=EXTRACTION_TOOL_PRIMARY_PROVIDER,
            model_id=EXTRACTION_TOOL_PRIMARY_MODEL_ID,
        )
    if purpose in _AGENTIC_STRUCTURED_PURPOSES:
        return (
            _preference_for_task(repositories, "context_update", save_id=save_id)
            or _preference_for_task(repositories, "context_search", save_id=save_id)
        )
    if purpose == "image_prompt":
        return _preference_for_task(repositories, "chat", save_id=save_id)
    return None


def recommended_tool_call_fallback_preference(
    repositories: PersistenceRepositories,
) -> ModelPreferenceRecord | None:
    return _recommended_tool_model_preference(
        repositories,
        task="tool_call_fallback",
        provider=EXTRACTION_TOOL_FALLBACK_PROVIDER,
        model_id=EXTRACTION_TOOL_FALLBACK_MODEL_ID,
    )


def _recommended_tool_model_preference(
    repositories: PersistenceRepositories,
    *,
    task: str,
    provider: str,
    model_id: str,
) -> ModelPreferenceRecord | None:
    for model in repositories.list_provider_models(provider):
        if model.model_id != model_id or not model.available:
            continue
        capabilities = {
            capability.strip().casefold().replace("-", "_")
            for capability in model.capabilities
        }
        if not capabilities & _TOOL_CALL_CAPABILITIES:
            return None
        return ModelPreferenceRecord(
            id=f"recommended:{task}:{provider}:{model_id}",
            task=task,
            provider=provider,
            model_id=model_id,
        )
    return None


def _preference_for_task(
    repositories: PersistenceRepositories,
    task: str,
    *,
    save_id: str | None = None,
) -> ModelPreferenceRecord | None:
    normalized_task = task.strip()
    if save_id is not None:
        preference = save_model_override_preference(
            repositories,
            save_id=save_id,
            task=normalized_task,
        )
        if preference is not None:
            return _normalized_preference(preference)
    return _normalized_preference(repositories.get_model_preference(normalized_task))


def _selector_fallback_purposes(purpose: str) -> tuple[str, ...]:
    explicit = {
        ACTION_CHOICE_GENERATION_PURPOSE: ("character_action_planning",),
        CHARACTER_PRESENCE_ASSESSMENT_PURPOSE: ("character_action_planning",),
        CHARACTER_INTENT_PLANNING_PURPOSE: ("character_action_planning",),
        DATING_ROUTE_PROFILE_PURPOSE: (
            CHARACTER_INTENT_PLANNING_PURPOSE,
            "character_action_planning",
            "context_update",
        ),
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
    if purpose in IMAGE_EDIT_FLOW_PURPOSES:
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


def _save_model_overrides(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> dict[str, object]:
    return sanitize_save_model_overrides(
        repositories.get_scoped_setting(
            scope="save",
            scope_id=save_id,
            key=SAVE_MODEL_OVERRIDES_SETTING,
        )
    )


def _write_save_model_overrides(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    overrides: Mapping[str, object],
) -> None:
    sanitized = sanitize_save_model_overrides(overrides)
    if not sanitized:
        repositories.delete_scoped_setting(
            scope="save",
            scope_id=save_id,
            key=SAVE_MODEL_OVERRIDES_SETTING,
        )
        return
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=SAVE_MODEL_OVERRIDES_SETTING,
        value=sanitized,
    )


def _override_preferences_from_settings(
    overrides: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    preferences = overrides.get("preferences")
    return dict(preferences) if isinstance(preferences, dict) else {}


def _override_thinking_from_settings(
    overrides: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    thinking = overrides.get("thinking")
    return dict(thinking) if isinstance(thinking, dict) else {}


def _sanitize_model_override_preferences(
    value: object,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        return {}
    sanitized: dict[str, dict[str, str]] = {}
    for raw_task, raw_config in value.items():
        task = _optional_text(raw_task)
        if (
            task is None
            or is_retired_model_task(task)
            or not isinstance(raw_config, Mapping)
        ):
            continue
        provider = _optional_text(raw_config.get("provider"))
        model_id = _optional_text(raw_config.get("model_id"))
        if provider is None or model_id is None:
            continue
        sanitized[task] = {"provider": provider.casefold(), "model_id": model_id}
    return sanitized


def _sanitize_model_override_thinking(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        return {}
    sanitized: dict[str, dict[str, str]] = {}
    for raw_task, raw_config in value.items():
        task = _optional_text(raw_task)
        if (
            task is None
            or is_retired_model_task(task)
            or not isinstance(raw_config, Mapping)
        ):
            continue
        provider = _optional_text(raw_config.get("provider"))
        model_id = _optional_text(raw_config.get("model_id"))
        level = _sanitize_thinking_level(raw_config.get("level"))
        if provider is None or model_id is None or level is None:
            continue
        sanitized[task] = {
            "provider": provider.casefold(),
            "model_id": model_id,
            "level": level,
        }
    return sanitized


def _required_text(value: object, label: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{label} is required")
    return text


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _sanitize_thinking_level(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold().replace("-", "_")
    return normalized if normalized in _THINKING_LEVEL_VALUES else None


def _scenario_type_for_save(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> str | None:
    save = repositories.get_save(save_id)
    if save is None:
        return None
    scenario = repositories.get_scenario(save.scenario_id)
    return scenario.type if scenario is not None else None


def _has_legacy_scenario_chat_preference(
    repositories: PersistenceRepositories,
) -> bool:
    return any(
        repositories.get_model_preference(task) is not None
        for task in _ROLEPLAY_CHAT_TASKS.values()
    )
