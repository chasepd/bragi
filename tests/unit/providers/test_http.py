from __future__ import annotations

from bragi.providers.contracts import ProviderCapability
from bragi.providers.errors import ProviderErrorCategory
from bragi.providers.http import (
    map_provider_status,
    normalize_capabilities,
    normalize_model_record,
)


def test_normalize_capabilities_maps_provider_aliases() -> None:
    assert normalize_capabilities(
        [
            "chat-completion",
            "image_generation",
            "text-to-video",
            "image_animation",
            "image_and_text_to_video",
            "image-to-image",
            "image_edit",
            "multimodal",
            "json_schema",
            "unmoderated",
            "unknown",
        ]
    ) == frozenset(
        {
            ProviderCapability.CHAT,
            ProviderCapability.IMAGE_GENERATION,
            ProviderCapability.TEXT_TO_VIDEO,
            ProviderCapability.IMAGE_TO_VIDEO,
            ProviderCapability.IMAGE_PLUS_TEXT_TO_VIDEO,
            ProviderCapability.IMAGE_TO_IMAGE,
            ProviderCapability.VISION,
            ProviderCapability.STRUCTURED_OUTPUT,
            ProviderCapability.BLOCKED_OUTPUT_FALLBACK,
        }
    )


def test_normalize_model_record_uses_id_name_context_and_capability_hints() -> None:
    model = normalize_model_record(
        provider="openrouter",
        payload={
            "id": "provider/model-a",
            "name": "Model A",
            "context_length": "16384",
            "capabilities": ["vision"],
        },
        capability_hints=["structured_output"],
    )

    assert model.provider == "openrouter"
    assert model.model_id == "provider/model-a"
    assert model.display_name == "Model A"
    assert model.context_window == 16384
    assert model.capabilities == frozenset(
        {
            ProviderCapability.VISION,
            ProviderCapability.STRUCTURED_OUTPUT,
            ProviderCapability.MODEL_LISTING,
        }
    )


def test_normalize_model_record_defaults_to_chat_when_no_capabilities() -> None:
    model = normalize_model_record(provider="venice", payload={"name": "venice-chat"})

    assert model.model_id == "venice-chat"
    assert model.capabilities == frozenset(
        {ProviderCapability.CHAT, ProviderCapability.MODEL_LISTING}
    )


def test_map_provider_status_delegates_http_category_mapping() -> None:
    assert map_provider_status(429) == ProviderErrorCategory.RATE_LIMITED
    assert map_provider_status(503) == ProviderErrorCategory.PROVIDER_ERROR
