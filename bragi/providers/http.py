"""Shared provider normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from bragi.providers.contracts import (
    ProviderCapability,
    ProviderGenerationParameter,
    ProviderModel,
    ProviderModelPricing,
    ProviderThinkingLevelSupport,
)
from bragi.providers.errors import ProviderErrorCategory, map_http_status_to_category


def normalize_capabilities(values: Iterable[str]) -> frozenset[ProviderCapability]:
    capabilities: set[ProviderCapability] = set()
    for value in values:
        normalized = value.lower().replace("-", "_")
        if normalized in {"text", "chat", "chat_completion", "completion"}:
            capabilities.add(ProviderCapability.CHAT)
        elif normalized in {"image", "image_generation"}:
            capabilities.add(ProviderCapability.IMAGE_GENERATION)
        elif normalized in {
            "image_to_image",
            "image_edit",
            "image_editing",
            "edit",
            "inpaint",
            "inpainting",
        }:
            capabilities.add(ProviderCapability.IMAGE_TO_IMAGE)
        elif normalized in {"video", "video_generation", "text_to_video"}:
            capabilities.add(ProviderCapability.TEXT_TO_VIDEO)
        elif normalized in {"image_to_video", "animate_image", "image_animation"}:
            capabilities.add(ProviderCapability.IMAGE_TO_VIDEO)
        elif normalized in {
            "image_plus_text_to_video",
            "image_text_to_video",
            "image_and_text_to_video",
        }:
            capabilities.add(ProviderCapability.IMAGE_PLUS_TEXT_TO_VIDEO)
        elif normalized in {
            "vision",
            "image_input",
            "image_understanding",
            "image_analysis",
            "multimodal",
        }:
            capabilities.add(ProviderCapability.VISION)
        elif normalized in {"model_listing", "models"}:
            capabilities.add(ProviderCapability.MODEL_LISTING)
        elif normalized in {
            "structured",
            "structured_output",
            "json_schema",
        }:
            capabilities.add(ProviderCapability.STRUCTURED_OUTPUT)
        elif normalized in {"tool_calling", "tools", "function_calling"}:
            capabilities.add(ProviderCapability.TOOL_CALLING)
        elif normalized in {
            "blocked_output_fallback",
            "uncensored",
            "unmoderated",
            "unmoderated_fallback",
        }:
            capabilities.add(ProviderCapability.BLOCKED_OUTPUT_FALLBACK)
    return frozenset(capabilities)


def normalize_generation_parameters(
    values: Iterable[str],
) -> frozenset[ProviderGenerationParameter]:
    parameters: set[ProviderGenerationParameter] = set()
    for value in values:
        normalized = value.lower().replace("-", "_")
        if normalized == "temperature":
            parameters.add(ProviderGenerationParameter.TEMPERATURE)
        elif normalized in {
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "output_tokens",
        }:
            parameters.add(ProviderGenerationParameter.MAX_OUTPUT_TOKENS)
        elif normalized in {
            "dimensions",
            "image_dimensions",
            "width",
            "height",
            "aspect_ratio",
            "image_size",
        }:
            parameters.add(ProviderGenerationParameter.IMAGE_DIMENSIONS)
        elif normalized in {"safe_mode", "image_safe_mode"}:
            parameters.add(ProviderGenerationParameter.IMAGE_SAFE_MODE)
    return frozenset(parameters)


THINKING_LEVELS: tuple[str, ...] = (
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
)


def normalize_model_record(
    *,
    provider: str,
    payload: dict[str, Any],
    capability_hints: Iterable[str] = (),
    default_to_chat: bool = True,
) -> ProviderModel:
    model_id = str(payload.get("id") or payload.get("model") or payload["name"])
    display_name = str(payload.get("name") or payload.get("display_name") or model_id)
    context_window = payload.get("context_length") or payload.get("context_window")
    capabilities = normalize_capabilities(
        [
            *[str(item) for item in payload.get("capabilities", [])],
            *capability_hints,
        ]
    )
    raw_supported_parameters = payload.get("supported_parameters", [])
    if not isinstance(raw_supported_parameters, list):
        raw_supported_parameters = []
    supported_parameters = normalize_generation_parameters(
        [str(item) for item in raw_supported_parameters]
    )
    if not capabilities and default_to_chat:
        capabilities = frozenset({ProviderCapability.CHAT})
    pricing = payload.get("pricing")
    return ProviderModel(
        provider=provider,
        model_id=model_id,
        display_name=display_name,
        capabilities=capabilities | {ProviderCapability.MODEL_LISTING},
        context_window=int(context_window) if context_window is not None else None,
        supported_parameters=supported_parameters,
        pricing=pricing if isinstance(pricing, ProviderModelPricing) else None,
        thinking=normalize_thinking_level_support(payload.get("thinking")),
    )


def normalize_thinking_level_support(
    value: object,
) -> ProviderThinkingLevelSupport | None:
    if not isinstance(value, Mapping):
        return None
    levels = _thinking_levels(value.get("levels"))
    default_level = _thinking_level(value.get("default_level"))
    if default_level is not None and default_level not in levels:
        default_level = None
    default_enabled = value.get("default_enabled")
    supports_max_tokens = value.get("supports_max_tokens") is True
    if not levels and not supports_max_tokens:
        return None
    return ProviderThinkingLevelSupport(
        levels=levels,
        default_level=default_level,
        default_enabled=default_enabled if isinstance(default_enabled, bool) else None,
        mandatory=value.get("mandatory") is True,
        supports_max_tokens=supports_max_tokens,
    )


def _thinking_levels(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    levels: list[str] = []
    for item in value:
        level = _thinking_level(item)
        if level is not None and level not in levels:
            levels.append(level)
    return tuple(levels)


def _thinking_level(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold().replace("-", "_")
    return normalized if normalized in THINKING_LEVELS else None


def map_provider_status(status_code: int) -> ProviderErrorCategory:
    return map_http_status_to_category(status_code)
