"""Saved model routing profile settings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from uuid import uuid4

from bragi.persistence.models import ModelPreferenceRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.generation_settings import (
    MODEL_THINKING_PREFERENCES_SETTING,
    sanitize_model_thinking_preferences,
)
from bragi.services.model_preferences import (
    ROLEPLAY_MODEL_PURPOSES,
    ROLEPLAY_SHARED_MODE_SETTING,
    ROLEPLAY_SHARED_TYPE,
    ROLEPLAY_TYPES,
    SCENARIO_GENERATION_SECTION_GROUPS,
    roleplay_model_task,
    scenario_generation_section_model_task,
    shared_roleplay_models_enabled,
)

MODEL_ROUTING_PROFILES_SETTING = "model_routing_profiles"
_MAX_PROFILE_NAME_LENGTH = 80
_EXTRA_TASKS = ("character_image_description",)


@dataclass(frozen=True)
class ModelRoutingProfilePreferenceModel:
    task: str
    provider: str
    model_id: str


@dataclass(frozen=True)
class ModelRoutingProfileModel:
    id: str
    name: str
    roleplay_shared_models_enabled: bool
    preference_count: int
    preferences: tuple[ModelRoutingProfilePreferenceModel, ...]


@dataclass(frozen=True)
class ModelRoutingProfilesModel:
    setting_key: str
    last_loaded_profile_id: str | None
    profiles: tuple[ModelRoutingProfileModel, ...]


def default_model_routing_profiles() -> dict[str, object]:
    return {"profiles": [], "last_loaded_profile_id": None}


def sanitize_model_routing_profiles(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return default_model_routing_profiles()
    profile_ids: set[str] = set()
    profiles: list[dict[str, object]] = []
    for item in _sequence_items(value.get("profiles")):
        profile = _sanitize_profile(item)
        if profile is None:
            continue
        profile_id = str(profile["id"])
        if profile_id in profile_ids:
            continue
        profile_ids.add(profile_id)
        profiles.append(profile)
    last_loaded = _trimmed_text(value.get("last_loaded_profile_id"))
    return {
        "profiles": profiles,
        "last_loaded_profile_id": last_loaded if last_loaded in profile_ids else None,
    }


def model_routing_profiles_model(
    repositories: PersistenceRepositories,
) -> ModelRoutingProfilesModel:
    sanitized = sanitize_model_routing_profiles(
        repositories.get_app_setting(MODEL_ROUTING_PROFILES_SETTING)
    )
    return ModelRoutingProfilesModel(
        setting_key=MODEL_ROUTING_PROFILES_SETTING,
        last_loaded_profile_id=_optional_text(sanitized["last_loaded_profile_id"]),
        profiles=tuple(
            _profile_model(profile)
            for profile in _profile_mappings(sanitized)
        ),
    )


def save_current_model_routing_profile(
    repositories: PersistenceRepositories,
    *,
    name: str,
    profile_id: str | None = None,
) -> ModelRoutingProfileModel:
    normalized_name = _profile_name(name)
    sanitized = sanitize_model_routing_profiles(
        repositories.get_app_setting(MODEL_ROUTING_PROFILES_SETTING)
    )
    raw_profiles = list(_profile_mappings(sanitized))
    existing_index = _profile_index(raw_profiles, profile_id) if profile_id else None
    if profile_id is not None and existing_index is None:
        raise ValueError("Unknown model routing profile")
    _raise_duplicate_name(raw_profiles, normalized_name, profile_id)
    next_profile_id = profile_id or uuid4().hex
    preferences = _snapshot_visible_preferences(repositories)
    raw_profile = {
        "id": next_profile_id,
        "name": normalized_name,
        "roleplay_shared_models_enabled": shared_roleplay_models_enabled(
            repositories
        ),
        "preferences": [
            {
                "task": preference.task,
                "provider": preference.provider,
                "model_id": preference.model_id,
            }
            for preference in preferences
        ],
        "thinking_preferences": _snapshot_visible_thinking_preferences(repositories),
    }
    if existing_index is None:
        raw_profiles.append(raw_profile)
    else:
        raw_profiles[existing_index] = raw_profile
    sanitized["profiles"] = raw_profiles
    sanitized["last_loaded_profile_id"] = next_profile_id
    repositories.set_app_setting(MODEL_ROUTING_PROFILES_SETTING, sanitized)
    return _profile_model(raw_profile)


def apply_model_routing_profile(
    repositories: PersistenceRepositories,
    profile_id: str,
) -> ModelRoutingProfileModel:
    sanitized = sanitize_model_routing_profiles(
        repositories.get_app_setting(MODEL_ROUTING_PROFILES_SETTING)
    )
    profile = _profile_by_id(sanitized["profiles"], profile_id)
    if profile is None:
        raise ValueError("Unknown model routing profile")
    model = _profile_model(profile)
    repositories.begin_transaction()
    try:
        repositories.set_app_setting(
            ROLEPLAY_SHARED_MODE_SETTING,
            model.roleplay_shared_models_enabled,
        )
        for task in model_routing_profile_task_universe():
            repositories.clear_model_preference(task)
        for preference in model.preferences:
            repositories.set_model_preference(
                task=preference.task,
                provider=preference.provider,
                model_id=preference.model_id,
            )
        _apply_profile_thinking_preferences(repositories, profile)
        sanitized["last_loaded_profile_id"] = model.id
        repositories.set_app_setting(MODEL_ROUTING_PROFILES_SETTING, sanitized)
        repositories.commit_transaction()
    except Exception:
        repositories.rollback_transaction()
        raise
    return model


def delete_model_routing_profile(
    repositories: PersistenceRepositories,
    profile_id: str,
) -> None:
    sanitized = sanitize_model_routing_profiles(
        repositories.get_app_setting(MODEL_ROUTING_PROFILES_SETTING)
    )
    raw_profiles = list(_profile_mappings(sanitized))
    profiles = [
        profile
        for profile in raw_profiles
        if profile.get("id") != profile_id
    ]
    if len(profiles) == len(raw_profiles):
        raise ValueError("Unknown model routing profile")
    sanitized["profiles"] = profiles
    if sanitized.get("last_loaded_profile_id") == profile_id:
        sanitized["last_loaded_profile_id"] = None
    repositories.set_app_setting(MODEL_ROUTING_PROFILES_SETTING, sanitized)


def model_routing_profile_task_universe() -> tuple[str, ...]:
    tasks: list[str] = []
    for roleplay_type in (
        ROLEPLAY_SHARED_TYPE,
        *ROLEPLAY_TYPES,
    ):
        for purpose in ROLEPLAY_MODEL_PURPOSES:
            tasks.append(
                roleplay_model_task(roleplay_type=roleplay_type, purpose=purpose)
            )
    for _group_label, section_ids in SCENARIO_GENERATION_SECTION_GROUPS:
        for section_id in section_ids:
            tasks.append(scenario_generation_section_model_task(section_id))
    tasks.extend(_EXTRA_TASKS)
    return _dedupe_preserving_order(tasks)


def _visible_profile_tasks(
    repositories: PersistenceRepositories,
) -> frozenset[str]:
    tasks: list[str] = []
    if shared_roleplay_models_enabled(repositories):
        tasks.extend(
            roleplay_model_task(roleplay_type=ROLEPLAY_SHARED_TYPE, purpose=purpose)
            for purpose in ROLEPLAY_MODEL_PURPOSES
        )
    else:
        for roleplay_type in ROLEPLAY_TYPES:
            tasks.extend(
                roleplay_model_task(roleplay_type=roleplay_type, purpose=purpose)
                for purpose in ROLEPLAY_MODEL_PURPOSES
            )
        tasks.append("chat")
    for _group_label, section_ids in SCENARIO_GENERATION_SECTION_GROUPS:
        for section_id in section_ids:
            tasks.append(scenario_generation_section_model_task(section_id))
    tasks.extend(_EXTRA_TASKS)
    return frozenset(tasks)


def _snapshot_visible_preferences(
    repositories: PersistenceRepositories,
) -> tuple[ModelPreferenceRecord, ...]:
    visible_tasks = _visible_profile_tasks(repositories)
    return tuple(
        preference
        for preference in repositories.list_model_preferences()
        if preference.task in visible_tasks
    )


def _sanitize_profile(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    profile_id = _trimmed_text(value.get("id"))
    name = _trimmed_text(value.get("name"), max_length=_MAX_PROFILE_NAME_LENGTH)
    if not profile_id or not name:
        return None
    return {
        "id": profile_id,
        "name": name,
        "roleplay_shared_models_enabled": (
            value.get("roleplay_shared_models_enabled") is True
        ),
        "preferences": _sanitize_preferences(value.get("preferences")),
        "thinking_preferences": _sanitize_thinking_preferences(
            value.get("thinking_preferences")
        ),
    }


def _sanitize_preferences(value: object) -> list[dict[str, str]]:
    task_universe = set(model_routing_profile_task_universe())
    seen_tasks: set[str] = set()
    preferences: list[dict[str, str]] = []
    for item in _sequence_items(value):
        if not isinstance(item, Mapping):
            continue
        task = _trimmed_text(item.get("task"))
        provider = _trimmed_text(item.get("provider"))
        model_id = _trimmed_text(item.get("model_id"))
        if not task or not provider or not model_id or task not in task_universe:
            continue
        if task in seen_tasks:
            continue
        seen_tasks.add(task)
        preferences.append({"task": task, "provider": provider, "model_id": model_id})
    return preferences


def _sanitize_thinking_preferences(value: object) -> list[dict[str, str]]:
    task_universe = set(model_routing_profile_task_universe())
    seen_tasks: set[str] = set()
    preferences: list[dict[str, str]] = []
    for item in _sequence_items(value):
        if not isinstance(item, Mapping):
            continue
        task = _trimmed_text(item.get("task"))
        provider = _trimmed_text(item.get("provider"))
        model_id = _trimmed_text(item.get("model_id"))
        level = _trimmed_text(item.get("level"))
        if not task or not provider or not model_id or not level:
            continue
        if task not in task_universe or task in seen_tasks:
            continue
        sanitized = sanitize_model_thinking_preferences(
            {task: {"provider": provider, "model_id": model_id, "level": level}}
        )
        if task not in sanitized:
            continue
        seen_tasks.add(task)
        preferences.append({"task": task, **sanitized[task]})
    return preferences


def _snapshot_visible_thinking_preferences(
    repositories: PersistenceRepositories,
) -> list[dict[str, str]]:
    visible_tasks = _visible_profile_tasks(repositories)
    preferences = sanitize_model_thinking_preferences(
        repositories.get_app_setting(MODEL_THINKING_PREFERENCES_SETTING)
    )
    return [
        {"task": task, **preference}
        for task, preference in preferences.items()
        if task in visible_tasks
    ]


def _apply_profile_thinking_preferences(
    repositories: PersistenceRepositories,
    profile: Mapping[str, object],
) -> None:
    task_universe = set(model_routing_profile_task_universe())
    current = sanitize_model_thinking_preferences(
        repositories.get_app_setting(MODEL_THINKING_PREFERENCES_SETTING)
    )
    for task in task_universe:
        current.pop(task, None)
    for item in _sequence_items(profile.get("thinking_preferences")):
        if not isinstance(item, Mapping):
            continue
        task = _trimmed_text(item.get("task"))
        if task and task in task_universe:
            sanitized = sanitize_model_thinking_preferences({task: item})
            if task in sanitized:
                current[task] = sanitized[task]
    repositories.set_app_setting(MODEL_THINKING_PREFERENCES_SETTING, current)


def _profile_model(profile: Mapping[str, object]) -> ModelRoutingProfileModel:
    preferences = tuple(
        ModelRoutingProfilePreferenceModel(
            task=str(preference["task"]),
            provider=str(preference["provider"]),
            model_id=str(preference["model_id"]),
        )
        for preference in _sequence_items(profile.get("preferences"))
        if isinstance(preference, Mapping)
    )
    return ModelRoutingProfileModel(
        id=str(profile["id"]),
        name=str(profile["name"]),
        roleplay_shared_models_enabled=profile.get(
            "roleplay_shared_models_enabled"
        )
        is True,
        preference_count=len(preferences),
        preferences=preferences,
    )


def _profile_index(
    profiles: list[Mapping[str, object]],
    profile_id: str | None,
) -> int | None:
    for index, profile in enumerate(profiles):
        if profile.get("id") == profile_id:
            return index
    return None


def _profile_by_id(value: object, profile_id: str) -> Mapping[str, object] | None:
    for profile in _sequence_items(value):
        if isinstance(profile, Mapping) and profile.get("id") == profile_id:
            return profile
    return None


def _raise_duplicate_name(
    profiles: list[Mapping[str, object]],
    name: str,
    profile_id: str | None,
) -> None:
    normalized = name.casefold()
    for profile in profiles:
        if profile.get("id") == profile_id:
            continue
        existing_name = _trimmed_text(profile.get("name"))
        if existing_name and existing_name.casefold() == normalized:
            raise ValueError("A model routing profile with that name already exists")


def _profile_name(value: str) -> str:
    name = _trimmed_text(value, max_length=_MAX_PROFILE_NAME_LENGTH)
    if not name:
        raise ValueError("Model routing profile name is required")
    return name


def _trimmed_text(value: object, *, max_length: int | None = None) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if max_length is not None:
        return text[:max_length]
    return text


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _profile_mappings(
    sanitized: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        profile
        for profile in _sequence_items(sanitized.get("profiles"))
        if isinstance(profile, Mapping)
    )


def _sequence_items(value: object) -> tuple[object, ...]:
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
        return tuple(value)
    return ()


def _dedupe_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)
