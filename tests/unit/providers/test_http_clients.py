from __future__ import annotations

import pytest

from bragi.providers.contracts import ProviderCapability
from bragi.providers.errors import (
    ProviderErrorCategory,
    map_exception_to_category,
    map_http_status_to_category,
)
from bragi.providers.openrouter import normalize_openrouter_models
from bragi.providers.venice import normalize_venice_models


def test_openrouter_model_normalization_maps_capabilities_and_context_windows() -> None:
    payload = {
        "data": [
            {
                "id": "anthropic/claude-3.5-sonnet",
                "canonical_slug": "anthropic/claude-3.5-sonnet",
                "name": "Claude 3.5 Sonnet",
                "context_length": 200_000,
                "architecture": {
                    "modality": "text->text",
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["temperature", "max_tokens"],
            },
            {
                "id": "google/gemini-2.5-flash-image-preview",
                "canonical_slug": "google/gemini-2.5-flash-image-preview",
                "name": "Gemini 2.5 Flash Image",
                "context_length": 32_768,
                "architecture": {
                    "modality": "text->image",
                    "input_modalities": ["text"],
                    "output_modalities": ["image"],
                },
                "supported_parameters": ["temperature"],
            },
        ]
    }

    models = normalize_openrouter_models(payload)

    models_by_id = {model.model_id: model for model in models}
    assert set(models_by_id) == {
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.5-flash-image-preview",
    }
    assert models_by_id["anthropic/claude-3.5-sonnet"].provider == "openrouter"
    assert (
        models_by_id["anthropic/claude-3.5-sonnet"].display_name
        == "Claude 3.5 Sonnet"
    )
    assert (
        ProviderCapability.CHAT
        in models_by_id["anthropic/claude-3.5-sonnet"].capabilities
    )
    assert models_by_id["anthropic/claude-3.5-sonnet"].context_window == 200_000
    assert (
        models_by_id["google/gemini-2.5-flash-image-preview"].provider
        == "openrouter"
    )
    assert (
        ProviderCapability.IMAGE_GENERATION
        in models_by_id["google/gemini-2.5-flash-image-preview"].capabilities
    )
    assert (
        models_by_id["google/gemini-2.5-flash-image-preview"].context_window
        == 32_768
    )


def test_openrouter_text_output_vision_model_is_chat_not_image_generation() -> None:
    payload = {
        "data": [
            {
                "id": "openai/gpt-4o",
                "name": "GPT-4o",
                "context_length": 128_000,
                "architecture": {
                    "modality": "text+image->text",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["temperature", "max_tokens"],
            }
        ]
    }

    models = normalize_openrouter_models(payload)

    assert len(models) == 1
    capabilities = models[0].capabilities
    assert ProviderCapability.CHAT in capabilities
    assert ProviderCapability.IMAGE_GENERATION not in capabilities


def test_venice_model_normalization_maps_model_spec_fields() -> None:
    payload = {
        "data": [
            {
                "created": 1_727_966_436,
                "id": "llama-3.2-3b",
                "model_spec": {
                    "availableContextTokens": 131_072,
                    "name": "Llama 3.2 3B",
                    "capabilities": {
                        "supportsFunctionCalling": True,
                        "supportsVision": False,
                    },
                },
                "object": "model",
                "owned_by": "venice.ai",
                "type": "text",
            },
            {
                "created": 1_727_966_500,
                "id": "hidream",
                "model_spec": {
                    "name": "HiDream",
                    "capabilities": {
                        "supportsMultipleImages": False,
                    },
                },
                "object": "model",
                "owned_by": "venice.ai",
                "type": "image",
            },
        ],
        "object": "list",
        "type": "all",
    }

    models = normalize_venice_models(payload)

    models_by_id = {model.model_id: model for model in models}
    assert set(models_by_id) == {"llama-3.2-3b", "hidream"}
    assert models_by_id["llama-3.2-3b"].provider == "venice"
    assert models_by_id["llama-3.2-3b"].display_name == "Llama 3.2 3B"
    assert ProviderCapability.CHAT in models_by_id["llama-3.2-3b"].capabilities
    assert models_by_id["llama-3.2-3b"].context_window == 131_072
    assert models_by_id["hidream"].provider == "venice"
    assert models_by_id["hidream"].display_name == "HiDream"
    assert ProviderCapability.IMAGE_GENERATION in models_by_id["hidream"].capabilities
    assert models_by_id["hidream"].context_window is None


@pytest.mark.parametrize(
    ("status_code", "expected_category"),
    [
        (401, ProviderErrorCategory.AUTHENTICATION_FAILED),
        (403, ProviderErrorCategory.AUTHENTICATION_FAILED),
        (404, ProviderErrorCategory.MODEL_NOT_FOUND),
        (408, ProviderErrorCategory.NETWORK_ERROR),
        (413, ProviderErrorCategory.CONTEXT_LIMIT_EXCEEDED),
        (429, ProviderErrorCategory.RATE_LIMITED),
        (500, ProviderErrorCategory.PROVIDER_ERROR),
    ],
)
def test_http_status_error_mapping_uses_app_categories(
    status_code: int,
    expected_category: ProviderErrorCategory,
) -> None:
    assert map_http_status_to_category(status_code) == expected_category


@pytest.mark.parametrize(
    ("exc", "expected_category"),
    [
        (TimeoutError("provider timed out"), ProviderErrorCategory.NETWORK_ERROR),
        (ConnectionError("connection reset"), ProviderErrorCategory.NETWORK_ERROR),
        (
            RuntimeError("provider sent malformed JSON"),
            ProviderErrorCategory.PROVIDER_ERROR,
        ),
    ],
)
def test_exception_error_mapping_uses_app_categories(
    exc: Exception,
    expected_category: ProviderErrorCategory,
) -> None:
    assert map_exception_to_category(exc) == expected_category
