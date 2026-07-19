"""Shared provider model capability checks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bragi.persistence.models import ProviderModelRecord
from bragi.persistence.repositories import PersistenceRepositories

CHAT_CAPABILITIES = frozenset({"chat", "chat_completion"})
STRUCTURED_OUTPUT_CAPABILITIES = frozenset(
    {"structured_output", "structured", "json_schema"}
)
TOOL_CALLING_CAPABILITIES = frozenset(
    {"tool_calling", "tools", "function_calling"}
)
IMAGE_GENERATION_CAPABILITIES = frozenset({"image_generation", "image"})
IMAGE_TO_IMAGE_CAPABILITIES = frozenset(
    {"image_to_image", "image_edit", "image_editing", "edit", "inpaint"}
)
TEXT_TO_VIDEO_CAPABILITIES = frozenset(
    {"text_to_video", "video_generation", "video"}
)
IMAGE_TO_VIDEO_CAPABILITIES = frozenset(
    {
        "image_to_video",
        "image_plus_text_to_video",
        "image_text_to_video",
        "image_animation",
    }
)
VISION_CAPABILITIES = frozenset({"vision"})

MODEL_MISSING_REASON = "model_missing"
MODEL_UNAVAILABLE_REASON = "model_unavailable"
MODEL_LACKS_CAPABILITY_REASON = "model_lacks_required_capabilities"


@dataclass(frozen=True)
class ModelCapabilityCheck:
    model: ProviderModelRecord | None
    found: bool
    available: bool
    supported: bool
    reason: str | None


def find_provider_model(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
) -> ProviderModelRecord | None:
    for model in repositories.list_provider_models(provider):
        if model.model_id == model_id:
            return model
    return None


def check_model_capabilities(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
    required: Iterable[str],
) -> ModelCapabilityCheck:
    model = find_provider_model(
        repositories,
        provider=provider,
        model_id=model_id,
    )
    if model is None:
        return ModelCapabilityCheck(
            model=None,
            found=False,
            available=False,
            supported=False,
            reason=MODEL_MISSING_REASON,
        )
    if not model.available:
        return ModelCapabilityCheck(
            model=model,
            found=True,
            available=False,
            supported=False,
            reason=MODEL_UNAVAILABLE_REASON,
        )
    required_capabilities = normalized_capabilities(required)
    if not normalized_capabilities(model.capabilities) & required_capabilities:
        return ModelCapabilityCheck(
            model=model,
            found=True,
            available=True,
            supported=False,
            reason=MODEL_LACKS_CAPABILITY_REASON,
        )
    return ModelCapabilityCheck(
        model=model,
        found=True,
        available=True,
        supported=True,
        reason=None,
    )


def model_supports_any_capability(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
    required: Iterable[str],
) -> bool:
    return check_model_capabilities(
        repositories,
        provider=provider,
        model_id=model_id,
        required=required,
    ).supported


def model_supports_any_capability_or_unknown(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
    required: Iterable[str],
) -> bool:
    check = check_model_capabilities(
        repositories,
        provider=provider,
        model_id=model_id,
        required=required,
    )
    return not check.found or check.supported


def known_model_is_unavailable(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
) -> bool:
    model = find_provider_model(
        repositories,
        provider=provider,
        model_id=model_id,
    )
    return model is not None and not model.available


def normalized_capabilities(values: Iterable[str]) -> frozenset[str]:
    return frozenset(_normalized_capability(value) for value in values)


def _normalized_capability(value: str) -> str:
    return value.strip().casefold().replace("-", "_")
