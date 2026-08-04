"""OpenRouter provider routing settings and request helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

from bragi.persistence.models import ProviderCatalogEntryRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ImageDescriptionRequest,
    ImageRequest,
    StructuredOutputRequest,
    ToolCallRequest,
)
from bragi.services.generation_settings import (
    request_with_model_thinking_preference,
)

OPENROUTER_PROVIDER_NAME = "openrouter"
OPENROUTER_ROUTING_PROFILES_SETTING = "openrouter_routing_profiles"
OPENROUTER_ROUTING_TASK_FAMILIES = (
    "narrator",
    "background_text",
    "structured_tool",
    "media",
)
OPENROUTER_ROUTING_TASK_LABELS = {
    "narrator": "Narrator",
    "background_text": "Background Text",
    "structured_tool": "Structured & Tool Work",
    "media": "Media",
}
OPENROUTER_ROUTING_SORT_OPTIONS = ("default", "price", "throughput", "latency")
OPENROUTER_ROUTING_PARTITION_OPTIONS = ("model", "none")
OPENROUTER_ROUTING_DATA_COLLECTION_OPTIONS = ("allow", "deny")
OPENROUTER_ROUTING_QUANTIZATION_OPTIONS = (
    "int4",
    "int8",
    "fp4",
    "fp6",
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
)
OPENROUTER_ROUTING_PERCENTILES = ("p50", "p75", "p90", "p99")
OPENROUTER_ROUTING_MAX_PRICE_FIELDS = ("prompt", "completion", "request", "image")
_SCENARIO_GENERATION_SECTION_TASK_PREFIX = "scenario_generation_section_"
_OPENROUTER_APP_TITLE = "Bragi"

_NARRATOR_TASKS = frozenset(
    {
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
        "chat_choose_your_own_adventure",
        "narrator_fallback",
    }
)
_BACKGROUND_TEXT_TASKS = frozenset(
    {"scenario_generation", "summarization", "image_prompt", "chat_fallback"}
)
_STRUCTURED_TOOL_TASKS = frozenset(
    {
        "context_search",
        "state_memory",
        "context_update",
        "character_enhancement",
        "fact_observation",
        "memory_curation",
        "response_planning",
        "response_verification",
        "content_safety",
        "director_pressure",
        "action_choice_generation",
        "character_presence_assessment",
        "character_intent_planning",
        "dating_route_profile",
        "character_action_planning",
        "character_registry_maintenance",
        "context_cleanup_scan",
        "context_cleanup_actions",
        "context_cleanup",
        "guided_context_cleanup",
        "state_pruning",
        "scenario_evolution",
        "npc_knowledge_audit",
        "structured_output_fallback",
        "tool_call_fallback",
    }
)
_MEDIA_TASKS = frozenset(
    {
        "image_generation",
        "image_to_image_generation",
        "scene_image_edit_generation",
        "character_image_edit_generation",
        "text_message_image_edit_generation",
        "image_fallback",
        "image_edit_fallback",
        "video_generation",
        "image_animation",
        "video_fallback",
        "character_image_description",
    }
)
_ROLEPLAY_PREFIXES = (
    "full_roleplay_",
    "fantasy_roleplay_",
    "science_fiction_roleplay_",
    "first_contact_exploration_",
    "survival_expedition_",
    "time_loop_",
    "investigation_mystery_",
    "heist_infiltration_",
    "political_intrigue_",
    "dating_sim_",
    "shared_roleplay_",
)
_SLUG_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_./"
)
@dataclass(frozen=True)
class OpenRouterProviderCatalogEntryModel:
    slug: str
    name: str
    privacy_policy_url: str | None
    terms_of_service_url: str | None
    status_page_url: str | None
    headquarters: str | None
    datacenters: tuple[str, ...]


@dataclass(frozen=True)
class OpenRouterRoutingTaskOverrideModel:
    task_family: str
    label: str
    enabled: bool
    profile: dict[str, object]
    provider_payload: dict[str, object]
    effective_provider_payload: dict[str, object]


@dataclass(frozen=True)
class OpenRouterRoutingSettingsModel:
    setting_key: str
    global_profile: dict[str, object]
    global_provider_payload: dict[str, object]
    task_overrides: tuple[OpenRouterRoutingTaskOverrideModel, ...]
    provider_catalog: tuple[OpenRouterProviderCatalogEntryModel, ...]
    provider_catalog_refreshed_at: str | None
    sort_options: tuple[str, ...]
    partition_options: tuple[str, ...]
    data_collection_options: tuple[str, ...]
    quantization_options: tuple[str, ...]
    percentile_options: tuple[str, ...]
    max_price_fields: tuple[str, ...]


def default_openrouter_routing_profiles() -> dict[str, object]:
    return {
        "global": _default_profile(),
        "task_overrides": {
            family: {"enabled": False, "profile": _default_profile()}
            for family in OPENROUTER_ROUTING_TASK_FAMILIES
        },
    }


def sanitize_openrouter_routing_profiles(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return default_openrouter_routing_profiles()
    task_overrides = value.get("task_overrides")
    raw_overrides = task_overrides if isinstance(task_overrides, Mapping) else {}
    return {
        "global": _sanitize_profile(value.get("global")),
        "task_overrides": {
            family: _sanitize_task_override(raw_overrides.get(family))
            for family in OPENROUTER_ROUTING_TASK_FAMILIES
        },
    }


def openrouter_routing_settings_model(
    repositories: PersistenceRepositories,
) -> OpenRouterRoutingSettingsModel:
    profiles = sanitize_openrouter_routing_profiles(
        repositories.get_app_setting(OPENROUTER_ROUTING_PROFILES_SETTING)
    )
    global_profile = _profile_from_sanitized(profiles["global"])
    global_payload = openrouter_provider_payload(global_profile)
    overrides = cast(dict[str, object], profiles["task_overrides"])
    provider_catalog = repositories.list_provider_catalog_entries(
        OPENROUTER_PROVIDER_NAME
    )
    return OpenRouterRoutingSettingsModel(
        setting_key=OPENROUTER_ROUTING_PROFILES_SETTING,
        global_profile=global_profile,
        global_provider_payload=global_payload,
        task_overrides=tuple(
            _task_override_model(
                family=family,
                raw_override=overrides[family],
                global_payload=global_payload,
            )
            for family in OPENROUTER_ROUTING_TASK_FAMILIES
        ),
        provider_catalog=tuple(
            _provider_catalog_entry_model(entry) for entry in provider_catalog
        ),
        provider_catalog_refreshed_at=max(
            (
                entry.refreshed_at
                for entry in provider_catalog
                if entry.refreshed_at is not None
            ),
            default=None,
        ),
        sort_options=OPENROUTER_ROUTING_SORT_OPTIONS,
        partition_options=OPENROUTER_ROUTING_PARTITION_OPTIONS,
        data_collection_options=OPENROUTER_ROUTING_DATA_COLLECTION_OPTIONS,
        quantization_options=OPENROUTER_ROUTING_QUANTIZATION_OPTIONS,
        percentile_options=OPENROUTER_ROUTING_PERCENTILES,
        max_price_fields=OPENROUTER_ROUTING_MAX_PRICE_FIELDS,
    )


def _provider_catalog_entry_model(
    entry: ProviderCatalogEntryRecord,
) -> OpenRouterProviderCatalogEntryModel:
    return OpenRouterProviderCatalogEntryModel(
        slug=entry.slug,
        name=entry.name,
        privacy_policy_url=entry.privacy_policy_url,
        terms_of_service_url=entry.terms_of_service_url,
        status_page_url=entry.status_page_url,
        headquarters=entry.headquarters,
        datacenters=tuple(entry.datacenters),
    )


def openrouter_routing_payload_for_task(
    repositories: PersistenceRepositories | None,
    *,
    provider: str,
    task: str,
) -> dict[str, object] | None:
    if repositories is None or provider != OPENROUTER_PROVIDER_NAME:
        return None
    profiles = sanitize_openrouter_routing_profiles(
        repositories.get_app_setting(OPENROUTER_ROUTING_PROFILES_SETTING)
    )
    family = openrouter_routing_task_family(task)
    profile = _profile_from_sanitized(profiles["global"])
    if family is not None:
        overrides = cast(dict[str, object], profiles["task_overrides"])
        override = cast(dict[str, object], overrides[family])
        if override["enabled"] is True:
            profile = _profile_from_sanitized(override["profile"])
    payload = openrouter_provider_payload(profile)
    return payload or None


def openrouter_routing_task_family(task: str) -> str | None:
    if task.startswith(_SCENARIO_GENERATION_SECTION_TASK_PREFIX):
        return "background_text"
    normalized = task
    for prefix in _ROLEPLAY_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    if normalized in _NARRATOR_TASKS:
        return "narrator"
    if normalized in _BACKGROUND_TEXT_TASKS:
        return "background_text"
    if normalized in _STRUCTURED_TOOL_TASKS:
        return "structured_tool"
    if normalized in _MEDIA_TASKS:
        return "media"
    return None


def openrouter_app_title_for_task(_task: str) -> str:
    return _OPENROUTER_APP_TITLE


def request_with_openrouter_routing[
    OpenRouterRoutableRequest: (
        ChatRequest,
        StructuredOutputRequest,
        ToolCallRequest,
        ImageRequest,
        ImageDescriptionRequest,
    )
](
    repositories: PersistenceRepositories | None,
    request: OpenRouterRoutableRequest,
    *,
    task: str,
    save_id: str | None = None,
) -> OpenRouterRoutableRequest:
    if repositories is not None and isinstance(request, ChatRequest):
        request = request_with_model_thinking_preference(
            repositories,
            request,
            task=task,
            save_id=save_id,
        )
    payload = openrouter_routing_payload_for_task(
        repositories,
        provider=request.provider,
        task=task,
    )
    app_title = (
        openrouter_app_title_for_task(task)
        if request.provider == OPENROUTER_PROVIDER_NAME
        else None
    )
    if (
        request.openrouter_provider_routing == payload
        and request.openrouter_app_title == app_title
    ):
        return request
    return replace(
        request,
        openrouter_provider_routing=payload,
        openrouter_app_title=app_title,
    )


def openrouter_provider_payload(profile: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    _copy_non_empty_list(payload, profile, "order")
    allow_fallbacks = profile.get("allow_fallbacks")
    if isinstance(allow_fallbacks, bool):
        payload["allow_fallbacks"] = allow_fallbacks
    if profile.get("require_parameters") is True:
        payload["require_parameters"] = True
    if profile.get("data_collection") == "deny":
        payload["data_collection"] = "deny"
    if profile.get("zdr") is True:
        payload["zdr"] = True
    if profile.get("enforce_distillable_text") is True:
        payload["enforce_distillable_text"] = True
    _copy_non_empty_list(payload, profile, "only")
    _copy_non_empty_list(payload, profile, "ignore")
    _copy_non_empty_list(payload, profile, "quantizations")
    sort = profile.get("sort")
    if sort in {"price", "throughput", "latency"}:
        if profile.get("sort_partition") == "none":
            payload["sort"] = {"by": sort, "partition": "none"}
        else:
            payload["sort"] = sort
    _copy_non_empty_mapping(payload, profile, "preferred_min_throughput")
    _copy_non_empty_mapping(payload, profile, "preferred_max_latency")
    _copy_non_empty_mapping(payload, profile, "max_price")
    return payload


def _task_override_model(
    *,
    family: str,
    raw_override: object,
    global_payload: dict[str, object],
) -> OpenRouterRoutingTaskOverrideModel:
    override = cast(dict[str, object], raw_override)
    profile = _profile_from_sanitized(override["profile"])
    payload = openrouter_provider_payload(profile)
    enabled = override["enabled"] is True
    return OpenRouterRoutingTaskOverrideModel(
        task_family=family,
        label=OPENROUTER_ROUTING_TASK_LABELS[family],
        enabled=enabled,
        profile=profile,
        provider_payload=payload,
        effective_provider_payload=payload if enabled else global_payload,
    )


def _sanitize_task_override(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {"enabled": False, "profile": _default_profile()}
    return {
        "enabled": value.get("enabled") is True,
        "profile": _sanitize_profile(value.get("profile")),
    }


def _sanitize_profile(value: object) -> dict[str, object]:
    raw = value if isinstance(value, Mapping) else {}
    sort = _choice(raw.get("sort"), OPENROUTER_ROUTING_SORT_OPTIONS, "default")
    return {
        "order": _slug_list(raw.get("order")),
        "allow_fallbacks": _optional_bool(raw.get("allow_fallbacks")),
        "require_parameters": raw.get("require_parameters") is True,
        "data_collection": _choice(
            raw.get("data_collection"),
            OPENROUTER_ROUTING_DATA_COLLECTION_OPTIONS,
            "allow",
        ),
        "zdr": raw.get("zdr") is True,
        "enforce_distillable_text": raw.get("enforce_distillable_text") is True,
        "only": _slug_list(raw.get("only")),
        "ignore": _slug_list(raw.get("ignore")),
        "quantizations": _choice_list(
            raw.get("quantizations"),
            OPENROUTER_ROUTING_QUANTIZATION_OPTIONS,
        ),
        "sort": sort,
        "sort_partition": (
            _choice(
                raw.get("sort_partition"),
                OPENROUTER_ROUTING_PARTITION_OPTIONS,
                "model",
            )
            if sort != "default"
            else "model"
        ),
        "preferred_min_throughput": _number_map(
            raw.get("preferred_min_throughput"),
            OPENROUTER_ROUTING_PERCENTILES,
            minimum_exclusive=0.0,
        ),
        "preferred_max_latency": _number_map(
            raw.get("preferred_max_latency"),
            OPENROUTER_ROUTING_PERCENTILES,
            minimum_exclusive=0.0,
        ),
        "max_price": _number_map(
            raw.get("max_price"),
            OPENROUTER_ROUTING_MAX_PRICE_FIELDS,
            minimum_inclusive=0.0,
        ),
    }


def _default_profile() -> dict[str, object]:
    return {
        "order": [],
        "allow_fallbacks": None,
        "require_parameters": False,
        "data_collection": "allow",
        "zdr": False,
        "enforce_distillable_text": False,
        "only": [],
        "ignore": [],
        "quantizations": [],
        "sort": "default",
        "sort_partition": "model",
        "preferred_min_throughput": {},
        "preferred_max_latency": {},
        "max_price": {},
    }


def _profile_from_sanitized(value: object) -> dict[str, object]:
    return dict(cast(dict[str, object], value))


def _copy_non_empty_list(
    payload: dict[str, object],
    profile: Mapping[str, object],
    key: str,
) -> None:
    values = profile.get(key)
    if isinstance(values, list) and values:
        payload[key] = list(values)


def _copy_non_empty_mapping(
    payload: dict[str, object],
    profile: Mapping[str, object],
    key: str,
) -> None:
    value = profile.get(key)
    if isinstance(value, Mapping) and value:
        payload[key] = dict(value)


def _slug_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    slugs: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        slug = item.strip().lower()
        if not slug or slug in seen or len(slug) > 128:
            continue
        if any(character not in _SLUG_CHARS for character in slug):
            continue
        seen.add(slug)
        slugs.append(slug)
    return slugs


def _choice_list(value: object, options: tuple[str, ...]) -> list[str]:
    if not isinstance(value, list):
        return []
    allowed = set(options)
    seen: set[str] = set()
    selected: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip().casefold().replace("-", "_")
        if normalized not in allowed or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
    return selected


def _choice(value: object, options: tuple[str, ...], default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().casefold().replace("-", "_")
        if normalized in options:
            return normalized
    return default


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _number_map(
    value: object,
    keys: tuple[str, ...],
    *,
    minimum_exclusive: float | None = None,
    minimum_inclusive: float | None = None,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key in keys:
        number = value.get(key)
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            continue
        parsed = float(number)
        if minimum_exclusive is not None and parsed <= minimum_exclusive:
            continue
        if minimum_inclusive is not None and parsed < minimum_inclusive:
            continue
        result[key] = parsed
    return result
