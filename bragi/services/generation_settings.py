"""User-configurable provider generation settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from bragi.model_tasks import is_retired_model_task
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatReasoningConfig,
    ChatRequest,
    ImageDescriptionRequest,
    ProviderGenerationParameter,
    StructuredOutputRequest,
    ToolCallRequest,
)
from bragi.services.model_preferences import save_model_thinking_preference

CHAT_TEMPERATURE_ENABLED_SETTING = "chat_temperature_enabled"
CHAT_TEMPERATURE_SETTING = "chat_temperature"
DEFAULT_CHAT_TEMPERATURE = 0.7
MIN_CHAT_TEMPERATURE = 0.0
MAX_CHAT_TEMPERATURE = 2.0
STEP_CHAT_TEMPERATURE = 0.05

CHAT_MAX_OUTPUT_TOKENS_ENABLED_SETTING = "chat_max_output_tokens_enabled"
CHAT_MAX_OUTPUT_TOKENS_SETTING = "chat_max_output_tokens"
DEFAULT_CHAT_MAX_OUTPUT_TOKENS = 2048
MIN_CHAT_MAX_OUTPUT_TOKENS = 64
MAX_CHAT_MAX_OUTPUT_TOKENS = 8192
STEP_CHAT_MAX_OUTPUT_TOKENS = 64

IMAGE_DIMENSION_PRESET_SETTING = "image_dimension_preset"
DEFAULT_IMAGE_DIMENSION_PRESET = "square_1024x1024"

OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING = "openrouter_chat_reasoning_overrides"
OPENROUTER_PROVIDER_NAME = "openrouter"
OPENROUTER_REASONING_EFFORTS = frozenset(
    {"max", "xhigh", "high", "medium", "low", "minimal", "none"}
)
MODEL_THINKING_PREFERENCES_SETTING = "model_thinking_preferences"
THINKING_LEVEL_PROVIDER_DEFAULT = "provider_default"
THINKING_LEVEL_OFF = "off"
THINKING_LEVEL_VALUES = (
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
)

ThinkingRequest = (
    ChatRequest | StructuredOutputRequest | ToolCallRequest | ImageDescriptionRequest
)


@dataclass(frozen=True)
class ChatGenerationSettings:
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class ImageDimensionPreset:
    id: str
    label: str
    dimensions: tuple[int, int] | None


_IMAGE_DIMENSION_PRESETS = (
    ImageDimensionPreset(
        id="provider_default",
        label="Provider default",
        dimensions=None,
    ),
    ImageDimensionPreset(
        id="square_1024x1024",
        label="Square 1024x1024",
        dimensions=(1024, 1024),
    ),
    ImageDimensionPreset(
        id="landscape_1024x768",
        label="Landscape 1024x768",
        dimensions=(1024, 768),
    ),
    ImageDimensionPreset(
        id="portrait_768x1024",
        label="Portrait 768x1024",
        dimensions=(768, 1024),
    ),
    ImageDimensionPreset(
        id="wide_1024x576",
        label="Wide 1024x576",
        dimensions=(1024, 576),
    ),
    ImageDimensionPreset(
        id="tall_576x1024",
        label="Tall 576x1024",
        dimensions=(576, 1024),
    ),
)
_IMAGE_DIMENSION_PRESETS_BY_ID = {
    preset.id: preset for preset in _IMAGE_DIMENSION_PRESETS
}


def sanitize_chat_temperature(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(max(float(value), MIN_CHAT_TEMPERATURE), MAX_CHAT_TEMPERATURE)
    return DEFAULT_CHAT_TEMPERATURE


def sanitize_chat_max_output_tokens(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return min(
            max(value, MIN_CHAT_MAX_OUTPUT_TOKENS),
            MAX_CHAT_MAX_OUTPUT_TOKENS,
        )
    return DEFAULT_CHAT_MAX_OUTPUT_TOKENS


def sanitize_image_dimension_preset(value: object) -> str:
    if isinstance(value, str):
        candidate = value.strip().lower().replace("-", "_")
        if candidate in _IMAGE_DIMENSION_PRESETS_BY_ID:
            return candidate
    return DEFAULT_IMAGE_DIMENSION_PRESET


def sanitize_openrouter_chat_reasoning_overrides(
    value: object,
) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        return {}
    sanitized: dict[str, dict[str, object]] = {}
    for raw_model_id, raw_config in value.items():
        model_id = _sanitize_openrouter_model_id(raw_model_id)
        if model_id is None:
            continue
        config = _sanitize_openrouter_reasoning_config(raw_config)
        if config:
            sanitized[model_id] = config
    return sanitized


def sanitize_model_thinking_preferences(
    value: object,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        return {}
    sanitized: dict[str, dict[str, str]] = {}
    for raw_task, raw_config in value.items():
        task = _sanitize_task(raw_task)
        if (
            task is None
            or is_retired_model_task(task)
            or not isinstance(raw_config, Mapping)
        ):
            continue
        provider = _sanitize_provider(raw_config.get("provider"))
        model_id = _sanitize_model_id(raw_config.get("model_id"))
        level = sanitize_thinking_level(raw_config.get("level"))
        if provider is None or model_id is None or level is None:
            continue
        sanitized[task] = {
            "provider": provider,
            "model_id": model_id,
            "level": level,
        }
    return sanitized


def sanitize_thinking_level(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold().replace("-", "_")
    if normalized in {THINKING_LEVEL_PROVIDER_DEFAULT, THINKING_LEVEL_OFF}:
        return normalized
    return normalized if normalized in THINKING_LEVEL_VALUES else None


def image_dimension_preset_options() -> tuple[str, ...]:
    return tuple(preset.id for preset in _IMAGE_DIMENSION_PRESETS)


def image_dimension_preset_label(preset_id: object) -> str:
    return _IMAGE_DIMENSION_PRESETS_BY_ID[
        sanitize_image_dimension_preset(preset_id)
    ].label


def selected_image_dimension_preset(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> str:
    return sanitize_image_dimension_preset(
        repositories.get_effective_setting(
            IMAGE_DIMENSION_PRESET_SETTING,
            save_id=save_id,
        )
    )


def chat_generation_settings(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
    save_id: str | None = None,
) -> ChatGenerationSettings:
    temperature = None
    if _bool_setting(
        repositories.get_effective_setting(
            CHAT_TEMPERATURE_ENABLED_SETTING,
            save_id=save_id,
        )
    ):
        if model_supports_generation_parameter(
            repositories,
            provider=provider,
            model_id=model_id,
            parameter=ProviderGenerationParameter.TEMPERATURE,
        ):
            temperature = sanitize_chat_temperature(
                repositories.get_effective_setting(
                    CHAT_TEMPERATURE_SETTING,
                    save_id=save_id,
                )
            )

    max_output_tokens = None
    if _bool_setting(
        repositories.get_effective_setting(
            CHAT_MAX_OUTPUT_TOKENS_ENABLED_SETTING,
            save_id=save_id,
        )
    ):
        if model_supports_generation_parameter(
            repositories,
            provider=provider,
            model_id=model_id,
            parameter=ProviderGenerationParameter.MAX_OUTPUT_TOKENS,
        ):
            max_output_tokens = sanitize_chat_max_output_tokens(
                repositories.get_effective_setting(
                    CHAT_MAX_OUTPUT_TOKENS_SETTING,
                    save_id=save_id,
                )
            )

    return ChatGenerationSettings(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def openrouter_chat_reasoning_config(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
    save_id: str | None = None,
) -> ChatReasoningConfig | None:
    if provider != OPENROUTER_PROVIDER_NAME:
        return None
    overrides = sanitize_openrouter_chat_reasoning_overrides(
        repositories.get_effective_setting(
            OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
            save_id=save_id,
        )
    )
    config = overrides.get(model_id)
    if not config:
        return None
    return ChatReasoningConfig(
        enabled=_optional_bool(config.get("enabled")),
        effort=_optional_str(config.get("effort")),
        max_tokens=_optional_int(config.get("max_tokens")),
        exclude=_optional_bool(config.get("exclude")),
    )


def model_thinking_reasoning_config(
    repositories: PersistenceRepositories,
    *,
    task: str,
    provider: str,
    model_id: str,
    save_id: str | None = None,
) -> ChatReasoningConfig | None:
    if save_id is not None:
        preference = save_model_thinking_preference(
            repositories,
            save_id=save_id,
            task=task,
        )
    else:
        preference = None
    if preference is None:
        preference = sanitize_model_thinking_preferences(
            repositories.get_app_setting(MODEL_THINKING_PREFERENCES_SETTING)
        ).get(task)
    if not preference:
        return None
    if preference["provider"] != provider or preference["model_id"] != model_id:
        return None
    support = model_thinking_support(
        repositories,
        provider=provider,
        model_id=model_id,
    )
    if not support:
        return None
    level = preference["level"]
    if level == THINKING_LEVEL_PROVIDER_DEFAULT:
        return None
    if level == THINKING_LEVEL_OFF:
        return ChatReasoningConfig(effort="none", exclude=True)
    if level not in _support_levels(support):
        return None
    return ChatReasoningConfig(effort=level, exclude=True)


def model_thinking_support(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
) -> dict[str, object] | None:
    for model in repositories.list_provider_models(provider):
        if model.model_id != model_id or not model.available or not model.thinking:
            continue
        levels = _support_levels(model.thinking)
        if not levels:
            return None
        return model.thinking
    return None


def model_thinking_preference_level(
    repositories: PersistenceRepositories,
    *,
    task: str,
    provider: str,
    model_id: str,
    save_id: str | None = None,
) -> str:
    if save_id is not None:
        preference = save_model_thinking_preference(
            repositories,
            save_id=save_id,
            task=task,
        )
    else:
        preference = None
    if preference is None:
        preference = sanitize_model_thinking_preferences(
            repositories.get_app_setting(MODEL_THINKING_PREFERENCES_SETTING)
        ).get(task)
    if (
        preference is None
        or preference["provider"] != provider
        or preference["model_id"] != model_id
    ):
        return THINKING_LEVEL_PROVIDER_DEFAULT
    support = model_thinking_support(
        repositories,
        provider=provider,
        model_id=model_id,
    )
    if support is None:
        return THINKING_LEVEL_PROVIDER_DEFAULT
    level = preference["level"]
    if level == THINKING_LEVEL_OFF:
        return level
    if level in _support_levels(support):
        return level
    return THINKING_LEVEL_PROVIDER_DEFAULT


def chat_request_with_reasoning_override(
    repositories: PersistenceRepositories,
    request: ChatRequest,
    *,
    task: str | None = None,
    save_id: str | None = None,
) -> ChatRequest:
    if request.reasoning is not None:
        return request
    if task is not None:
        config = model_thinking_reasoning_config(
            repositories,
            task=task,
            provider=request.provider,
            model_id=request.model_id,
            save_id=save_id,
        )
        if config is not None:
            return replace(request, reasoning=config)
    config = openrouter_chat_reasoning_config(
        repositories,
        provider=request.provider,
        model_id=request.model_id,
        save_id=save_id,
    )
    if config is None:
        return request
    return replace(request, reasoning=config)


def request_with_model_thinking_preference[ThinkingRequestT: ThinkingRequest](
    repositories: PersistenceRepositories,
    request: ThinkingRequestT,
    *,
    task: str,
    save_id: str | None = None,
) -> ThinkingRequestT:
    if request.reasoning is not None:
        return request
    config = model_thinking_reasoning_config(
        repositories,
        task=task,
        provider=request.provider,
        model_id=request.model_id,
        save_id=save_id,
    )
    if config is None:
        return request
    return replace(request, reasoning=config)


def image_generation_dimensions(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
    save_id: str | None = None,
) -> tuple[int, int] | None:
    if not model_supports_generation_parameter(
        repositories,
        provider=provider,
        model_id=model_id,
        parameter=ProviderGenerationParameter.IMAGE_DIMENSIONS,
    ):
        return None
    preset = _IMAGE_DIMENSION_PRESETS_BY_ID[
        selected_image_dimension_preset(repositories, save_id=save_id)
    ]
    return preset.dimensions


def model_supports_generation_parameter(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
    parameter: ProviderGenerationParameter,
) -> bool:
    for model in repositories.list_provider_models(provider):
        if model.model_id != model_id or not model.available:
            continue
        return parameter.value in {
            value.lower().replace("-", "_") for value in model.supported_parameters
        }
    return False


def _support_levels(support: Mapping[str, object]) -> tuple[str, ...]:
    levels = support.get("levels")
    if not isinstance(levels, list | tuple):
        return ()
    return tuple(
        level
        for item in levels
        if (level := sanitize_thinking_level(item)) in THINKING_LEVEL_VALUES
    )


def _bool_setting(value: object | None) -> bool:
    return value if isinstance(value, bool) else False


def _sanitize_task(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    task = value.strip()
    if not task or any(character.isspace() for character in task):
        return None
    return task


def _sanitize_provider(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    provider = value.strip().casefold()
    if not provider or any(character.isspace() for character in provider):
        return None
    return provider


def _sanitize_model_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    model_id = value.strip()
    if not model_id or any(character.isspace() for character in model_id):
        return None
    return model_id


def _sanitize_openrouter_model_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if "/" not in candidate:
        return None
    if any(character.isspace() for character in candidate):
        return None
    return candidate or None


def _sanitize_openrouter_reasoning_config(
    value: object,
) -> dict[str, object]:
    if isinstance(value, str):
        normalized = value.strip().casefold().replace("-", "_")
        if normalized == "disabled":
            return {"enabled": False, "exclude": True}
        if normalized in OPENROUTER_REASONING_EFFORTS:
            return {"effort": normalized, "exclude": True}
        return {}
    if not isinstance(value, Mapping):
        return {}
    config: dict[str, object] = {}
    enabled = value.get("enabled")
    if isinstance(enabled, bool):
        config["enabled"] = enabled
    effort = value.get("effort")
    if isinstance(effort, str):
        normalized_effort = effort.strip().casefold().replace("-", "_")
        if normalized_effort in OPENROUTER_REASONING_EFFORTS:
            config["effort"] = normalized_effort
    max_tokens = value.get("max_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
        if max_tokens > 0:
            config["max_tokens"] = max_tokens
    exclude = value.get("exclude")
    if isinstance(exclude, bool):
        config["exclude"] = exclude
    if "effort" in config and "max_tokens" in config:
        config.pop("max_tokens")
    return config


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
