from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from bragi.providers.contracts import (
    ChatMessage,
    ChatReasoningConfig,
    ChatRequest,
    ImageDescriptionRequest,
    ImageRequest,
    ProviderCapability,
    ProviderGenerationParameter,
    StructuredOutputRequest,
    ToolCallMessage,
    ToolCallRequest,
    ToolDefinition,
    VideoRequest,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.http_client import BinaryHttpResponse, JsonHttpResponse
from bragi.providers.openrouter import (
    OpenRouterClient,
    _image_config,
    _parse_structured_content,
)
from bragi.services.secrets import InMemorySecretStore, SecretStorageError

OPENROUTER_ALL_MODALITIES_MODELS_PATH = "/api/v1/models?output_modalities=all"
OPENROUTER_PROVIDERS_PATH = "/api/v1/providers"
OPENROUTER_DEFAULT_APP_URL = "https://github.com/chasepd/bragi"


class RecordingTransport:
    def __init__(self, responses: list[JsonHttpResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> JsonHttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected OpenRouter transport call")
        return self.responses.pop(0)


class RecordingBinaryTransport:
    def __init__(self, responses: list[BinaryHttpResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> BinaryHttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected OpenRouter binary transport call")
        return self.responses.pop(0)


class RecordingStreamTransport:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> Any:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout": timeout,
            }
        )
        for event in self.events:
            yield event


class FailingSecretStore:
    def set_api_key(self, provider: str, api_key: str) -> None:
        raise AssertionError("provider config validation should only read secrets")

    def delete_api_key(self, provider: str) -> None:
        raise AssertionError("provider config validation should only read secrets")

    def has_api_key(self, provider: str) -> bool:
        raise AssertionError("provider config validation should call get_api_key")

    def get_api_key(self, provider: str) -> str | None:
        raise SecretStorageError("Secret Service keyring read failed")


def test_openrouter_validate_config_uses_secret_key_for_models_probe() -> None:
    transport = RecordingTransport([JsonHttpResponse(status_code=200, payload={})])
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    status = asyncio.run(client.validate_config())

    assert status.provider == "openrouter"
    assert status.configured is True
    assert status.authenticated is True
    assert status.error is None
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["method"] == "GET"
    assert call["url"].endswith(OPENROUTER_ALL_MODALITIES_MODELS_PATH)
    assert call["headers"]["Authorization"] == "Bearer or-secret"
    assert call["payload"] is None


def test_openrouter_validate_config_reports_secret_storage_error() -> None:
    transport = RecordingTransport([])
    client = OpenRouterClient(
        secret_store=FailingSecretStore(),
        transport=transport,
    )

    status = asyncio.run(client.validate_config())

    assert status.provider == "openrouter"
    assert status.configured is True
    assert status.authenticated is False
    assert status.error == ProviderErrorCategory.SECRET_STORAGE_ERROR.value
    assert transport.calls == []


def test_openrouter_list_models_gets_models_with_bearer_auth_and_normalizes() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "openai/gpt-4o-mini",
                            "name": "GPT-4o mini",
                            "context_length": 128_000,
                            "architecture": {"output_modalities": ["text"]},
                        },
                        {
                            "id": "google/gemini-flash-image",
                            "name": "Gemini Flash Image",
                            "context_length": 32_768,
                            "architecture": {"output_modalities": ["image"]},
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    call = transport.calls[0]
    assert call["method"] == "GET"
    assert call["url"].endswith(OPENROUTER_ALL_MODALITIES_MODELS_PATH)
    assert call["headers"]["Authorization"] == "Bearer or-secret"
    models_by_id = {model.model_id: model for model in models}
    assert set(models_by_id) == {
        "openai/gpt-4o-mini",
        "google/gemini-flash-image",
    }
    assert ProviderCapability.CHAT in models_by_id["openai/gpt-4o-mini"].capabilities
    assert (
        ProviderCapability.IMAGE_GENERATION
        in models_by_id["google/gemini-flash-image"].capabilities
    )


def test_openrouter_list_providers_gets_catalog_with_bearer_auth_and_normalizes(
) -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "slug": "openai",
                            "name": "OpenAI",
                            "privacy_policy_url": "https://openai.com/privacy",
                            "terms_of_service_url": "https://openai.com/terms",
                            "status_page_url": "https://status.openai.com",
                            "headquarters": "US",
                            "datacenters": ["US", "IE"],
                        },
                        {"slug": "", "name": "Missing Slug"},
                        {"slug": "deepinfra/turbo", "name": 123},
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    providers = asyncio.run(client.list_providers())

    call = transport.calls[0]
    assert call["method"] == "GET"
    assert call["url"].endswith(OPENROUTER_PROVIDERS_PATH)
    assert call["headers"]["Authorization"] == "Bearer or-secret"
    assert call["payload"] is None
    assert [(provider.slug, provider.name) for provider in providers] == [
        ("openai", "OpenAI"),
        ("deepinfra/turbo", "deepinfra/turbo"),
    ]
    assert providers[0].privacy_policy_url == "https://openai.com/privacy"
    assert providers[0].terms_of_service_url == "https://openai.com/terms"
    assert providers[0].status_page_url == "https://status.openai.com"
    assert providers[0].headquarters == "US"
    assert providers[0].datacenters == ("US", "IE")


def test_openrouter_list_models_exposes_pricing_per_million_tokens() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "openai/gpt-4o-mini",
                            "name": "GPT-4o mini",
                            "context_length": 128_000,
                            "architecture": {"output_modalities": ["text"]},
                            "pricing": {
                                "prompt": "0.00000015",
                                "completion": "0.0000006",
                                "request": "0",
                                "image": "0.025",
                                "input_cache_read": "0.00000001",
                                "input_cache_write": "0.00000002",
                            },
                        }
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    pricing = models[0].pricing
    assert pricing is not None
    assert pricing.input_per_million_tokens_usd == "0.15"
    assert pricing.output_per_million_tokens_usd == "0.6"
    assert pricing.cache_read_per_million_tokens_usd == "0.01"
    assert pricing.cache_write_per_million_tokens_usd == "0.02"
    assert pricing.request_usd == "0"
    assert pricing.image_usd == "0.025"


def test_openrouter_list_models_notes_unknown_billable_pricing_fields() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "reasoning/search-model",
                            "name": "Reasoning Search Model",
                            "architecture": {"output_modalities": ["text"]},
                            "pricing": {
                                "prompt": "0.0000002",
                                "completion": "0.0000008",
                                "web_search": "0.002",
                                "internal_reasoning": "0.0000001",
                            },
                        }
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    pricing = models[0].pricing
    assert pricing is not None
    assert pricing.note == (
        "Additional pricing fields: internal_reasoning, web_search"
    )


def test_openrouter_list_models_normalizes_reasoning_metadata() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "google/gemini-3.5-flash",
                            "name": "Gemini 3.5 Flash",
                            "architecture": {"output_modalities": ["text"]},
                            "supported_parameters": ["reasoning", "temperature"],
                            "reasoning": {
                                "supported_efforts": [
                                    "high",
                                    "medium",
                                    "low",
                                    "minimal",
                                ],
                                "default_effort": "medium",
                                "default_enabled": True,
                                "mandatory": True,
                                "supports_max_tokens": False,
                            },
                        }
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    thinking = models[0].thinking
    assert thinking is not None
    assert thinking.levels == ("high", "medium", "low", "minimal")
    assert thinking.default_level == "medium"
    assert thinking.default_enabled is True
    assert thinking.mandatory is True
    assert thinking.supports_max_tokens is False


def test_openrouter_marks_chat_model_for_blocked_output_fallback() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "acme/uncensored-frontier-chat",
                            "name": "Uncensored Frontier Chat",
                            "description": "Unfiltered frontier roleplay chat model.",
                            "context_length": 32_768,
                            "architecture": {"output_modalities": ["text"]},
                            "top_provider": {"is_moderated": False},
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert ProviderCapability.CHAT in models[0].capabilities
    assert ProviderCapability.BLOCKED_OUTPUT_FALLBACK in models[0].capabilities


def test_openrouter_unmarked_chat_model_is_not_blocked_output_fallback() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "acme/frontier-chat",
                            "name": "Frontier Chat",
                            "description": "General purpose frontier chat model.",
                            "context_length": 32_768,
                            "architecture": {"output_modalities": ["text"]},
                            "top_provider": {"is_moderated": False},
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert ProviderCapability.CHAT in models[0].capabilities
    assert ProviderCapability.BLOCKED_OUTPUT_FALLBACK not in models[0].capabilities


def test_openrouter_image_fallback_requires_all_signals() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "acme/uncensored-image",
                            "name": "Uncensored Image",
                            "description": "Unfiltered image generation.",
                            "architecture": {"output_modalities": ["image"]},
                            "top_provider": {"is_moderated": False},
                        },
                        {
                            "id": "acme/moderated-uncensored-image",
                            "name": "Uncensored Image Moderated",
                            "description": "Unfiltered image generation.",
                            "architecture": {"output_modalities": ["image"]},
                            "top_provider": {"is_moderated": True},
                        },
                        {
                            "id": "acme/general-image",
                            "name": "General Image",
                            "description": "General image generation.",
                            "architecture": {"output_modalities": ["image"]},
                            "top_provider": {"is_moderated": False},
                        },
                        {
                            "id": "acme/uncensored-audio",
                            "name": "Uncensored Audio",
                            "description": "Unmoderated audio generation.",
                            "architecture": {"output_modalities": ["audio"]},
                            "top_provider": {"is_moderated": False},
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    models_by_id = {model.model_id: model for model in models}
    assert models_by_id["acme/uncensored-image"].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_GENERATION,
            ProviderCapability.MODEL_LISTING,
            ProviderCapability.BLOCKED_OUTPUT_FALLBACK,
        }
    )
    assert models_by_id["acme/moderated-uncensored-image"].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_GENERATION,
            ProviderCapability.MODEL_LISTING,
        }
    )
    assert models_by_id["acme/general-image"].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_GENERATION,
            ProviderCapability.MODEL_LISTING,
        }
    )
    assert models_by_id["acme/uncensored-audio"].capabilities == frozenset(
        {ProviderCapability.MODEL_LISTING}
    )


def test_openrouter_list_models_keeps_non_bragi_outputs_listing_only() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "openai/gpt-4o-mini",
                            "name": "GPT-4o mini",
                            "context_length": 128_000,
                            "architecture": {"output_modalities": ["text"]},
                        },
                        {
                            "id": "google/gemini-flash-image",
                            "name": "Gemini Flash Image",
                            "context_length": 32_768,
                            "architecture": {"output_modalities": ["image"]},
                        },
                        {
                            "id": "google/gemini-flash-image-preview",
                            "name": "Gemini Flash Image Preview",
                            "context_length": 32_768,
                            "architecture": {"output_modalities": ["text", "image"]},
                        },
                        {
                            "id": "openai/gpt-4o-mini-vision",
                            "name": "GPT-4o mini Vision",
                            "context_length": 128_000,
                            "architecture": {
                                "input_modalities": ["text", "image"],
                                "output_modalities": ["text"],
                            },
                        },
                        {
                            "id": "acme/audio-only",
                            "name": "Audio Only",
                            "architecture": {"output_modalities": ["audio"]},
                        },
                        {
                            "id": "acme/text-audio",
                            "name": "Text Audio",
                            "architecture": {"output_modalities": ["text", "audio"]},
                        },
                        {
                            "id": "acme/embedder",
                            "name": "Embedder",
                            "architecture": {"output_modalities": ["embeddings"]},
                        },
                        {
                            "id": "acme/video-only",
                            "name": "Video Only",
                            "architecture": {"output_modalities": ["video"]},
                        },
                        {
                            "id": "acme/text-video",
                            "name": "Text Video",
                            "architecture": {"output_modalities": ["text", "video"]},
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    models_by_id = {model.model_id: model for model in models}
    assert models_by_id["openai/gpt-4o-mini"].capabilities == frozenset(
        {
            ProviderCapability.CHAT,
            ProviderCapability.MODEL_LISTING,
        }
    )
    assert models_by_id["google/gemini-flash-image"].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_GENERATION,
            ProviderCapability.MODEL_LISTING,
        }
    )
    assert models_by_id["google/gemini-flash-image-preview"].capabilities == frozenset(
        {
            ProviderCapability.CHAT,
            ProviderCapability.IMAGE_GENERATION,
            ProviderCapability.MODEL_LISTING,
        }
    )
    assert models_by_id["openai/gpt-4o-mini-vision"].capabilities == frozenset(
        {
            ProviderCapability.CHAT,
            ProviderCapability.MODEL_LISTING,
            ProviderCapability.VISION,
        }
    )
    for model_id in (
        "acme/audio-only",
        "acme/text-audio",
        "acme/embedder",
        "acme/video-only",
        "acme/text-video",
    ):
        assert models_by_id[model_id].capabilities == frozenset(
            {ProviderCapability.MODEL_LISTING}
        )


def test_openrouter_text_model_with_file_input_remains_chat_selectable() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "mistralai/mistral-large-2512",
                            "name": "Mistral: Mistral Large 3 2512",
                            "context_length": 262_144,
                            "architecture": {
                                "modality": "text+image+file->text",
                                "input_modalities": ["text", "image", "file"],
                                "output_modalities": ["text"],
                            },
                            "supported_parameters": [
                                "max_tokens",
                                "response_format",
                                "structured_outputs",
                                "temperature",
                                "tool_choice",
                                "tools",
                            ],
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert models[0].model_id == "mistralai/mistral-large-2512"
    assert ProviderCapability.CHAT in models[0].capabilities
    assert ProviderCapability.VISION in models[0].capabilities
    assert ProviderCapability.STRUCTURED_OUTPUT in models[0].capabilities
    assert ProviderCapability.TOOL_CALLING in models[0].capabilities
    assert ProviderGenerationParameter.MAX_OUTPUT_TOKENS in (
        models[0].supported_parameters
    )


def test_text_model_without_metadata_has_no_structured_output() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "openai/gpt-4o-mini",
                            "name": "GPT-4o mini",
                            "context_length": 128_000,
                            "architecture": {"output_modalities": ["text"]},
                            "supported_parameters": ["temperature", "max_tokens"],
                        }
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert ProviderCapability.CHAT in models[0].capabilities
    assert ProviderCapability.STRUCTURED_OUTPUT not in models[0].capabilities


def test_openrouter_tools_metadata_marks_tool_calling_not_structured_output() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "openai/gpt-4o-mini",
                            "name": "GPT-4o mini",
                            "context_length": 128_000,
                            "architecture": {"output_modalities": ["text"]},
                            "supported_parameters": [
                                "temperature",
                                "tools",
                                "tool_choice",
                            ],
                        }
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert ProviderCapability.CHAT in models[0].capabilities
    assert ProviderCapability.TOOL_CALLING in models[0].capabilities
    assert ProviderCapability.STRUCTURED_OUTPUT not in models[0].capabilities


def test_openrouter_models_expose_supported_generation_parameters() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "openai/gpt-4o-mini",
                            "name": "GPT-4o mini",
                            "architecture": {"output_modalities": ["text"]},
                            "supported_parameters": ["temperature", "max_tokens"],
                        },
                        {
                            "id": "google/gemini-flash-image",
                            "name": "Gemini Flash Image",
                            "architecture": {"output_modalities": ["image"]},
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = {model.model_id: model for model in asyncio.run(client.list_models())}

    assert models["openai/gpt-4o-mini"].supported_parameters == frozenset(
        {
            ProviderGenerationParameter.TEMPERATURE,
            ProviderGenerationParameter.MAX_OUTPUT_TOKENS,
        }
    )
    assert models["google/gemini-flash-image"].supported_parameters == frozenset(
        {ProviderGenerationParameter.IMAGE_DIMENSIONS}
    )


@pytest.mark.parametrize(
    "supported_parameters",
    [
        ["temperature", "response_format"],
        ["temperature", "json_schema"],
    ],
)
def test_openrouter_structured_output_requires_schema_enforced_metadata(
    supported_parameters: list[str],
) -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "openai/gpt-4o-mini",
                            "name": "GPT-4o mini",
                            "context_length": 128_000,
                            "architecture": {"output_modalities": ["text"]},
                            "supported_parameters": supported_parameters,
                        }
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert ProviderCapability.STRUCTURED_OUTPUT in models[0].capabilities


def test_openrouter_tool_call_payload_uses_tools_and_preserves_arguments() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "update_scene_snapshot",
                                            "arguments": '{"source_message_id":',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"total_tokens": 11},
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.generate_tool_calls(
            ToolCallRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(
                    ToolCallMessage(role="system", body="Extract updates."),
                    ToolCallMessage(role="user", body="Completed turn."),
                ),
                tools=(
                    ToolDefinition(
                        name="update_scene_snapshot",
                        description="Updates the current scene.",
                        parameters={
                            "type": "object",
                            "properties": {"source_message_id": {"type": "string"}},
                            "required": ["source_message_id"],
                            "additionalProperties": False,
                        },
                    ),
                ),
                temperature=0.0,
                max_output_tokens=400,
                openrouter_provider_routing={
                    "require_parameters": True,
                    "data_collection": "deny",
                },
                openrouter_app_title="Bragi",
            )
        )
    )

    assert (
        transport.calls[0]["headers"]["X-OpenRouter-Title"]
        == "Bragi"
    )
    payload = transport.calls[0]["payload"]
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "update_scene_snapshot",
                "description": "Updates the current scene.",
                "parameters": {
                    "type": "object",
                    "properties": {"source_message_id": {"type": "string"}},
                    "required": ["source_message_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert payload["max_tokens"] == 400
    assert payload["provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert response.tool_calls[0].arguments_json == '{"source_message_id":'


def test_openrouter_chat_posts_contextual_completion_and_parses_usage() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": "The archive door opens."}},
                    ],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 5,
                        "total_tokens": 13,
                    },
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(
                    ChatMessage(role="player", body="I press the silver latch."),
                ),
                scenario_instructions="Keep the scene tense and grounded.",
                custom_instructions="Favor terse second-person prompts.",
                regeneration_feedback="Make the replacement more ominous.",
                retrieved_state=("location=archive",),
                retrieved_memories=("Mara distrusts silver locks.",),
                current_scene_recap=(
                    "scene.location: Archive\n"
                    "scene.present_characters: Mara, Archivist Venn\n"
                    "Recent: Mara is pressing the silver latch.",
                ),
                character_voice_profiles=(
                    "Archivist Venn voice: soft, precise, never uses contractions.",
                ),
                summary="Mara reached the archive after dusk.",
                temperature=0.7,
                max_output_tokens=256,
                openrouter_provider_routing={
                    "sort": "price",
                    "allow_fallbacks": False,
                },
                openrouter_app_title="Bragi",
            )
        )
    )

    assert response.body == "The archive door opens."
    assert response.provider == "openrouter"
    assert response.model_id == "openai/gpt-4o-mini"
    assert response.token_usage["total_tokens"] == 13
    assert response.token_usage["total"] == 13
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer or-secret"
    assert call["headers"]["X-OpenRouter-Title"] == "Bragi"
    payload = call["payload"]
    assert payload["model"] == "openai/gpt-4o-mini"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 256
    assert payload["provider"] == {
        "sort": "price",
        "allow_fallbacks": False,
    }
    assert payload["messages"][0]["role"] == "system"
    system_body = payload["messages"][0]["content"]
    assert "Response style:" in system_body
    assert "- Keep responses reasonably short." in system_body
    assert "- Put dialogue in quotation marks." in system_body
    assert "- Put non-dialogue narration in italics." in system_body
    assert "- Format text messages with > at the beginning of each message." in (
        system_body
    )
    assert system_body.index("Response style:") < system_body.index(
        "Keep the scene tense and grounded."
    )
    assert "Keep the scene tense and grounded." in system_body
    assert "Save response guidance:" in system_body
    assert "Favor terse second-person prompts." in system_body
    assert "Current scene recap:" in system_body
    assert (
        "Current scene recap:\n"
        "- scene.location: Archive\n"
        "scene.present_characters: Mara, Archivist Venn\n"
        "Recent: Mara is pressing the silver latch."
    ) in system_body
    assert "Summary:" in system_body
    assert system_body.index("Summary:") < system_body.index(
        "Current scene recap:"
    )
    assert system_body.index("Current scene recap:") < system_body.index(
        "Regeneration feedback:"
    )
    assert "Character voice profiles:" in system_body
    assert system_body.index("Summary:") < system_body.index(
        "Character voice profiles:"
    )
    assert system_body.index("Character voice profiles:") < system_body.index(
        "Retrieved state:"
    )
    assert "location=archive" in system_body
    assert "Archivist Venn voice: soft, precise" in system_body
    assert "Mara distrusts silver locks." in system_body
    assert "Mara reached the archive after dusk." in system_body
    assert "Make the replacement more ominous." in system_body
    assert payload["messages"][1] == {
        "role": "user",
        "content": "I press the silver latch.",
    }


def test_openrouter_chat_sends_configured_app_referer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAGI_OPENROUTER_APP_URL", "https://bragi.example.test")
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The gate opens."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="player", body="Open the gate."),),
                openrouter_app_title="Bragi",
            )
        )
    )

    headers = transport.calls[0]["headers"]
    assert headers["X-OpenRouter-Title"] == "Bragi"
    assert headers["HTTP-Referer"] == "https://bragi.example.test"


def test_openrouter_chat_ignores_deprecated_agent_specific_app_title() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "Search complete."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="player", body="Find context."),),
                openrouter_app_title="Bragi - Context Search Agent",
            )
        )
    )

    headers = transport.calls[0]["headers"]
    assert headers["X-OpenRouter-Title"] == "Bragi"
    assert headers["HTTP-Referer"] == OPENROUTER_DEFAULT_APP_URL


@pytest.mark.parametrize("app_title", [None, "", "   "])
def test_openrouter_chat_uses_default_app_title_when_missing(
    app_title: str | None,
) -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "Defaulted."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="player", body="Use defaults."),),
                openrouter_app_title=app_title,
            )
        )
    )

    headers = transport.calls[0]["headers"]
    assert headers["X-OpenRouter-Title"] == "Bragi"
    assert headers["HTTP-Referer"] == OPENROUTER_DEFAULT_APP_URL


@pytest.mark.parametrize(
    "app_url",
    [
        None,
        "",
        "bragi",
        "ftp://bragi.example.test",
        "https://bragi.example.test\r\nX-Bad: yep",
    ],
)
def test_openrouter_chat_uses_default_app_referer_when_unconfigured_or_invalid(
    monkeypatch: pytest.MonkeyPatch,
    app_url: str | None,
) -> None:
    if app_url is None:
        monkeypatch.delenv("BRAGI_OPENROUTER_APP_URL", raising=False)
    else:
        monkeypatch.setenv("BRAGI_OPENROUTER_APP_URL", app_url)
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The gate opens."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="player", body="Open the gate."),),
                openrouter_app_title="Bragi",
            )
        )
    )

    headers = transport.calls[0]["headers"]
    assert headers["X-OpenRouter-Title"] == "Bragi"
    assert headers["HTTP-Referer"] == OPENROUTER_DEFAULT_APP_URL


def test_openrouter_chat_renders_timeskip_turn_directive_only_when_present() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "Dawn breaks."}}]},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The latch opens."}}]},
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(
                    ChatMessage(
                        role="system",
                        speaker_name="Timeskip",
                        body="Timeskip request: Skip to dawn at the city gates.",
                    ),
                ),
                turn_directive="Timeskip request: Skip to dawn at the city gates.",
            )
        )
    )
    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="player", body="I test the latch."),),
            )
        )
    )

    timeskip_system_body = transport.calls[0]["payload"]["messages"][0]["content"]
    normal_system_body = transport.calls[1]["payload"]["messages"][0]["content"]
    assert "Turn directive:" in timeskip_system_body
    assert "Timeskip request: Skip to dawn at the city gates." in timeskip_system_body
    assert "may advance time, location, and immediate player circumstances" in (
        timeskip_system_body
    )
    assert "Turn directive:" not in normal_system_body
    assert "may advance time, location, and immediate player circumstances" not in (
        normal_system_body
    )


def test_openrouter_stream_chat_posts_streaming_completion_and_yields_chunks() -> None:
    stream = RecordingStreamTransport(
        [
            {"choices": [{"delta": {"content": "The archive"}}]},
            {"choices": [{"delta": {"content": " door opens."}}]},
            {"choices": [], "usage": {"total_tokens": 13}},
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, stream_transport=stream)

    async def collect() -> list[str]:
        chunks = []
        async for chunk in client.stream_chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="player", body="Open it."),),
                temperature=0.7,
                max_output_tokens=256,
                openrouter_provider_routing={
                    "sort": {"by": "throughput", "partition": "none"},
                },
                openrouter_app_title="Bragi",
            )
        ):
            if chunk.delta:
                chunks.append(chunk.delta)
            if chunk.token_usage:
                assert chunk.token_usage["total"] == 13
        return chunks

    assert asyncio.run(collect()) == ["The archive", " door opens."]
    call = stream.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer or-secret"
    assert call["headers"]["X-OpenRouter-Title"] == "Bragi"
    payload = call["payload"]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 256
    assert payload["provider"] == {
        "sort": {"by": "throughput", "partition": "none"},
    }


def test_openrouter_chat_renders_state_change_and_media_asset_sections() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The gate opens."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="player", body="I follow the lights."),),
                retrieved_state=("[world_state:state-1] scene.location: Bridge",),
                retrieved_state_changes=(
                    "[state_change:change-1] scene.exit changed to Moon Gate",
                ),
                retrieved_recent_messages=(
                    "[message:message-1] Narrator: The bridge answered with bells.",
                ),
                retrieved_memories=("[memory:memory-1] Mara distrusts bells.",),
                retrieved_media_assets=(
                    "[media_asset:media-1] Image prompt: gold bridge lights",
                ),
            )
        )
    )

    system_body = transport.calls[0]["payload"]["messages"][0]["content"]
    assert (
        "Retrieved state:\n"
        "- [world_state:state-1] scene.location: Bridge"
    ) in system_body
    assert (
        "Retrieved state changes:\n"
        "- [state_change:change-1] scene.exit changed to Moon Gate"
    ) in system_body
    assert (
        "Retrieved chronicle:\n"
        "- [message:message-1] Narrator: The bridge answered with bells."
    ) in system_body
    assert (
        "Retrieved memories:\n- [memory:memory-1] Mara distrusts bells."
        in system_body
    )
    assert (
        "Retrieved media assets:\n"
        "- [media_asset:media-1] Image prompt: gold bridge lights"
    ) in system_body
    assert system_body.index("Retrieved memories:") < system_body.index(
        "Retrieved media assets:"
    )
    assert system_body.index("Retrieved media assets:") < system_body.index(
        "Retrieved chronicle:"
    )
    assert system_body.index("Retrieved chronicle:") < system_body.index(
        "Retrieved state changes:"
    )
    assert system_body.index("Retrieved state changes:") < system_body.index(
        "Retrieved state:"
    )


def test_openrouter_chat_raw_metadata_includes_safe_response_headers() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": "The archive door opens."}},
                    ],
                },
                headers={
                    "x-venice-is-content-violation": "true",
                    "x-request-id": "req-123",
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="player", body="I test the latch."),),
            )
        )
    )

    assert response.raw_metadata["_bragi_headers"] == {
        "x-venice-is-content-violation": "true",
        "x-request-id": "req-123",
    }


def test_openrouter_chat_opts_into_router_metadata() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": "The relay clicks on."}},
                    ],
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="qwen/qwen3-235b-a22b-2507",
                messages=(ChatMessage(role="player", body="Check the relay."),),
            )
        )
    )

    assert transport.calls[0]["headers"]["X-OpenRouter-Metadata"] == "enabled"


def test_openrouter_chat_normalizes_provider_message_names() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The crews answer."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(
                    ChatMessage(
                        role="player",
                        body="I hail the bridge.",
                        speaker_name="Mara Jade",
                    ),
                    ChatMessage(
                        role="narrator",
                        body="Captain Voss answers from the mist.",
                        speaker_name="Captain Voss (NPC)",
                    ),
                    ChatMessage(
                        role="player",
                        body="I wait for the static to clear.",
                        speaker_name="!!!",
                    ),
                ),
            )
        )
    )

    messages = transport.calls[0]["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert "Response style:" in messages[0]["content"]
    assert messages[1:] == [
        {
            "role": "user",
            "content": "I hail the bridge.",
            "name": "Mara_Jade",
        },
        {
            "role": "assistant",
            "content": "Captain Voss answers from the mist.",
            "name": "Captain_Voss_NPC",
        },
        {
            "role": "user",
            "content": "I wait for the static to clear.",
        },
    ]
    assert "name" not in messages[3]


def test_openrouter_chat_malformed_success_response_raises_provider_error() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": ["not text"]}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.chat(
                ChatRequest(
                    provider="openrouter",
                    model_id="openai/gpt-4o-mini",
                    messages=(ChatMessage(role="player", body="Hello?"),),
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR
    assert "content must be a string" in exc_info.value.message


def test_openrouter_chat_reasoning_only_length_response_is_diagnosed() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "native_finish_reason": "length",
                            "message": {
                                "content": None,
                                "reasoning": "private thinking text",
                                "reasoning_details": [
                                    {
                                        "type": "reasoning.text",
                                        "text": "private detail",
                                    },
                                    {
                                        "type": "reasoning.encrypted",
                                        "encrypted_content": "secret",
                                    },
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "completion_tokens_details": {
                            "reasoning_tokens": 20,
                        },
                    },
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.chat(
                ChatRequest(
                    provider="openrouter",
                    model_id="openai/gpt-5-mini",
                    messages=(ChatMessage(role="player", body="Hello?"),),
                    max_output_tokens=20,
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR
    assert "reasoning-only response" in exc_info.value.message
    assert "increase max_output_tokens" in exc_info.value.message
    assert exc_info.value.diagnostics == {
        "finish_reason": "length",
        "native_finish_reason": "length",
        "reasoning_tokens": 20,
        "reasoning_detail_types": ["reasoning.text", "reasoning.encrypted"],
    }
    assert "private thinking text" not in repr(exc_info.value.diagnostics)
    assert "private detail" not in repr(exc_info.value.diagnostics)
    assert "secret" not in repr(exc_info.value.diagnostics)


def test_openrouter_chat_sends_reasoning_config() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The crews answer."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="z-ai/glm-4.7",
                messages=(ChatMessage(role="player", body="Hello?"),),
                reasoning=ChatReasoningConfig(
                    effort="low",
                    exclude=True,
                ),
            )
        )
    )

    assert transport.calls[0]["payload"]["reasoning"] == {
        "effort": "low",
        "exclude": True,
    }


def test_openrouter_chat_sends_disabled_reasoning_config() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The crews answer."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="z-ai/glm-4.7",
                messages=(ChatMessage(role="player", body="Hello?"),),
                reasoning=ChatReasoningConfig(
                    enabled=False,
                    exclude=True,
                ),
            )
        )
    )

    assert transport.calls[0]["payload"]["reasoning"] == {
        "enabled": False,
        "exclude": True,
    }


def test_openrouter_structured_output_sends_reasoning_config() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": '{"result":"ok"}',
                            }
                        }
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.generate_structured_output(
            StructuredOutputRequest(
                provider="openrouter",
                model_id="z-ai/glm-4.7",
                messages=(ChatMessage(role="user", body="Extract."),),
                schema_name="result",
                schema={
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                    "additionalProperties": False,
                },
                reasoning=ChatReasoningConfig(effort="medium", exclude=True),
            )
        )
    )

    assert transport.calls[0]["payload"]["reasoning"] == {
        "effort": "medium",
        "exclude": True,
    }


def test_openrouter_structured_output_rejects_local_schema_violation() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": '{"result":17,"extra":true}'}}
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_structured_output(
                StructuredOutputRequest(
                    provider="openrouter",
                    model_id="openai/gpt-4o-mini",
                    messages=(ChatMessage(role="user", body="Extract."),),
                    schema_name="result_contract",
                    schema={
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                        "required": ["result"],
                        "additionalProperties": False,
                    },
                )
            )
        )

    assert (
        exc_info.value.category
        == ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID
    )
    assert exc_info.value.diagnostics["schema_name"] == "result_contract"
    assert exc_info.value.diagnostics["error_count"] == 2
    assert len(transport.calls) == 1


def test_openrouter_structured_output_types_non_object_schema_violation() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "[]"}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_structured_output(
                StructuredOutputRequest(
                    provider="openrouter",
                    model_id="openai/gpt-4o-mini",
                    messages=(ChatMessage(role="user", body="Extract."),),
                    schema_name="object_contract",
                    schema={"type": "object", "additionalProperties": False},
                )
            )
        )

    assert (
        exc_info.value.category
        == ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID
    )
    assert len(transport.calls) == 1


def test_openrouter_structured_content_preserves_provider_diagnostics() -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "length",
                "native_finish_reason": "length",
                "message": {
                    "content": None,
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "private detail"},
                    ],
                },
            }
        ],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 20}},
    }

    with pytest.raises(ProviderError) as exc_info:
        _parse_structured_content(payload)

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR
    assert "reasoning-only response" in exc_info.value.message
    assert exc_info.value.diagnostics == {
        "finish_reason": "length",
        "native_finish_reason": "length",
        "reasoning_tokens": 20,
        "reasoning_detail_types": ["reasoning.text"],
    }
    assert "private detail" not in repr(exc_info.value.diagnostics)


def test_openrouter_chat_accepts_multimodal_text_content() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "The canal "},
                                    {"type": "text", "text": "fog answers."},
                                ]
                            }
                        }
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="player", body="Hello?"),),
            )
        )
    )

    assert response.body == "The canal fog answers."


def test_openrouter_structured_output_retries_non_json_success_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retry_progress: list[object] = []

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", no_sleep)
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": "I could not produce JSON."}},
                    ],
                },
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"state_changes":[],"memories":[],'
                                    '"metadata":null}'
                                )
                            }
                        },
                    ],
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.generate_structured_output(
            StructuredOutputRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(
                    ChatMessage(role="user", body="Mara pockets the brass key."),
                ),
                schema_name="state_memory_update",
                schema={
                    "type": "object",
                    "properties": {
                        "state_changes": {"type": "array"},
                        "memories": {"type": "array"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["state_changes", "memories"],
                    "additionalProperties": False,
                },
                openrouter_provider_routing={"zdr": True},
                openrouter_app_title="Bragi",
                retry_progress_callback=retry_progress.append,
            )
        )
    )

    assert response.data == {
        "state_changes": [],
        "memories": [],
        "metadata": None,
    }
    assert len(transport.calls) == 2
    assert [call["method"] for call in transport.calls] == ["POST", "POST"]
    assert [
        call["headers"]["X-OpenRouter-Title"] for call in transport.calls
    ] == [
        "Bragi",
        "Bragi",
    ]
    assert transport.calls[0]["payload"]["provider"] == {"zdr": True}
    sent_schema = transport.calls[0]["payload"]["response_format"]["json_schema"][
        "schema"
    ]
    assert sent_schema["required"] == ["state_changes", "memories", "metadata"]
    assert sent_schema["properties"]["metadata"]["type"] == ["string", "null"]
    retry_metadata = response.raw_metadata["_bragi_retry"]
    assert retry_metadata["attempt_count"] == 2
    assert retry_metadata["max_attempts"] == 3
    assert len(retry_progress) == 1

    attempts = retry_metadata["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["attempt"] == 1
    assert attempts[0]["error_category"] == ProviderErrorCategory.PROVIDER_ERROR.value
    assert isinstance(attempts[0]["duration_ms"], int)
    assert attempts[1]["attempt"] == 2
    assert attempts[1]["error_category"] is None
    assert isinstance(attempts[1]["duration_ms"], int)


def test_openrouter_structured_output_enforces_async_transport_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport_calls: list[dict[str, Any]] = []

    async def blocked_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        transport_calls.append({"function": function, "kwargs": dict(kwargs)})
        await asyncio.Future()

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(
        "bragi.providers.openrouter.asyncio.to_thread",
        blocked_to_thread,
    )
    monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", no_sleep)
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(
        secret_store=secrets,
        timeout=0.01,
    )

    async def generate() -> None:
        await client.generate_structured_output(
            StructuredOutputRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="user", body="Plan the next beat."),),
                schema_name="narrator_message_plan",
                schema={
                    "type": "object",
                    "properties": {"source_ids": {"type": "array"}},
                    "required": ["source_ids"],
                    "additionalProperties": False,
                },
            )
        )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(asyncio.wait_for(generate(), timeout=0.5))

    assert exc_info.value.category == ProviderErrorCategory.NETWORK_ERROR
    assert "timed out" in exc_info.value.message
    assert exc_info.value.retry_attempt_count == 3
    assert exc_info.value.max_retry_attempts == 3
    assert len(transport_calls) == 3
    assert all(
        call["kwargs"]["task"] == "structured_output" for call in transport_calls
    )
    assert all(call["kwargs"]["timeout"] == 0.01 for call in transport_calls)


def test_openrouter_chat_retries_transient_failure_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", no_sleep)
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=429,
                payload={"error": {"message": "slow down"}},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": "The archive door opens."}},
                    ],
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.chat(
            ChatRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                messages=(ChatMessage(role="player", body="I test the latch."),),
            )
        )
    )

    assert len(transport.calls) == 2
    assert [call["method"] for call in transport.calls] == ["POST", "POST"]
    retry_metadata = response.raw_metadata["_bragi_retry"]
    assert retry_metadata["attempt_count"] == 2
    assert retry_metadata["max_attempts"] == 3

    attempts = retry_metadata["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["attempt"] == 1
    assert attempts[0]["error_category"] == ProviderErrorCategory.RATE_LIMITED.value
    assert isinstance(attempts[0]["duration_ms"], int)
    assert attempts[1]["attempt"] == 2
    assert attempts[1]["error_category"] is None
    assert isinstance(attempts[1]["duration_ms"], int)


def test_openrouter_chat_does_not_retry_authentication_failure() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=401,
                payload={"error": {"message": "bad key"}},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.chat(
                ChatRequest(
                    provider="openrouter",
                    model_id="openai/gpt-4o-mini",
                    messages=(ChatMessage(role="player", body="Hello?"),),
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.AUTHENTICATION_FAILED
    assert len(transport.calls) == 1


def test_openrouter_chat_does_not_retry_non_5xx_provider_error() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=400,
                payload={"error": {"message": "bad request"}},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": "This should not be reached."}},
                    ],
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.chat(
                ChatRequest(
                    provider="openrouter",
                    model_id="openai/gpt-4o-mini",
                    messages=(ChatMessage(role="player", body="Hello?"),),
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR
    assert exc_info.value.status_code == 400
    assert len(transport.calls) == 1


def test_openrouter_generate_image_posts_modalities_and_decodes_data_url() -> None:
    image_bytes = b"fake-openrouter-image"
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "google/gemini-flash-image",
                            "name": "Gemini Flash Image",
                            "architecture": {"output_modalities": ["image"]},
                        }
                    ]
                },
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "images": [{"image_url": {"url": data_url}}],
                            }
                        }
                    ]
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="openrouter",
                model_id="google/gemini-flash-image",
                prompt="A candlelit archive door.",
                source_save_id="save-1",
                source_message_id="message-1",
                dimensions=(1024, 768),
                openrouter_provider_routing={
                    "quantizations": ["fp8"],
                    "max_price": {"image": 0.05},
                },
                openrouter_app_title="Bragi",
            )
        )
    )

    assert response.provider == "openrouter"
    assert response.model_id == "google/gemini-flash-image"
    assert response.image_bytes == image_bytes
    assert len(transport.calls) == 2
    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[0]["url"].endswith(OPENROUTER_ALL_MODALITIES_MODELS_PATH)
    assert (
        transport.calls[0]["headers"]["X-OpenRouter-Title"]
        == "Bragi"
    )
    call = transport.calls[1]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/chat/completions")
    assert call["headers"]["X-OpenRouter-Title"] == "Bragi"
    payload = call["payload"]
    assert payload["model"] == "google/gemini-flash-image"
    assert payload["modalities"] == ["image"]
    assert payload["messages"][-1]["content"] == "A candlelit archive door."
    assert payload["provider"] == {
        "quantizations": ["fp8"],
        "max_price": {"image": 0.05},
    }
    assert payload["image_config"] == {
        "aspect_ratio": "4:3",
        "image_size": "1K",
    }


def test_openrouter_image_to_image_posts_source_image_content(
    tmp_path: Path,
) -> None:
    image_bytes = b"fake-openrouter-image"
    reference_bytes = b"fake-reference-png"
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(reference_bytes)
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "recraft/recraft-v3",
                            "name": "Recraft V3",
                            "architecture": {
                                "input_modalities": ["text", "image"],
                                "output_modalities": ["image"],
                            },
                        }
                    ]
                },
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "images": [{"image_url": {"url": data_url}}],
                            }
                        }
                    ]
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())
    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="openrouter",
                model_id="recraft/recraft-v3",
                prompt="Keep the character identity while changing the pose.",
                source_save_id="save-1",
                source_message_id="message-1",
                source_media_asset_id="reference-asset",
                source_media_path=reference_path,
            )
        )
    )

    assert ProviderCapability.IMAGE_TO_IMAGE in models[0].capabilities
    assert response.image_bytes == image_bytes
    payload = transport.calls[1]["payload"]
    content = payload["messages"][-1]["content"]
    assert content[0] == {
        "type": "text",
        "text": "Keep the character identity while changing the pose.",
    }
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == (
        "data:image/png;base64,"
        + base64.b64encode(reference_bytes).decode("ascii")
    )


def test_openrouter_generate_image_posts_multiple_reference_images(
    tmp_path: Path,
) -> None:
    image_bytes = b"generated-image"
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    first_reference = tmp_path / "first.png"
    second_reference = tmp_path / "second.png"
    first_reference.write_bytes(b"first-reference")
    second_reference.write_bytes(b"second-reference")
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "recraft/recraft-v3",
                            "name": "Recraft V3",
                            "architecture": {
                                "input_modalities": ["text", "image"],
                                "output_modalities": ["image"],
                            },
                        }
                    ]
                },
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "images": [{"image_url": {"url": data_url}}],
                            }
                        }
                    ]
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)
    asyncio.run(client.list_models())

    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="openrouter",
                model_id="recraft/recraft-v3",
                prompt="Keep both character identities in the scene.",
                source_save_id="save-1",
                source_message_id="message-1",
                source_media_asset_ids=("first-asset", "second-asset"),
                source_media_paths=(first_reference, second_reference),
            )
        )
    )

    assert response.image_bytes == image_bytes
    content = transport.calls[1]["payload"]["messages"][-1]["content"]
    assert content[0] == {
        "type": "text",
        "text": "Keep both character identities in the scene.",
    }
    assert [item["type"] for item in content[1:]] == ["image_url", "image_url"]
    assert content[1]["image_url"]["url"] == (
        "data:image/png;base64,"
        + base64.b64encode(b"first-reference").decode("ascii")
    )
    assert content[2]["image_url"]["url"] == (
        "data:image/png;base64,"
        + base64.b64encode(b"second-reference").decode("ascii")
    )


def test_openrouter_image_config_maps_dimensions_to_supported_values() -> None:
    assert _image_config((1200, 500)) == {
        "aspect_ratio": "21:9",
        "image_size": "2K",
    }
    assert _image_config((512, 384)) == {
        "aspect_ratio": "4:3",
        "image_size": "1K",
    }


def test_openrouter_generate_video_submits_polls_and_downloads_content() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=202,
                payload={
                    "id": "job-123",
                    "polling_url": "/api/v1/videos/job-123",
                    "status": "pending",
                    "generation_id": "gen-123",
                },
            ),
            JsonHttpResponse(
                status_code=200,
                payload={"id": "job-123", "status": "in_progress"},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "id": "job-123",
                    "generation_id": "gen-123",
                    "status": "completed",
                    "unsigned_urls": [
                        "https://storage.example.com/private-video.mp4",
                    ],
                    "usage": {"cost": 0.25},
                },
            ),
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [
            BinaryHttpResponse(
                status_code=200,
                body=b"fake-mp4",
                headers={"content-type": "video/mp4"},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    response = asyncio.run(
        client.generate_video(
            VideoRequest(
                provider="openrouter",
                model_id="google/veo-3.1",
                prompt="A serene mountain landscape at sunset",
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
    )

    assert response.provider == "openrouter"
    assert response.model_id == "google/veo-3.1"
    assert response.mime_type == "video/mp4"
    assert response.video_bytes == b"fake-mp4"
    assert response.raw_metadata["job_id"] == "job-123"
    assert response.raw_metadata["generation_id"] == "gen-123"
    assert response.raw_metadata["status"] == "completed"
    assert response.raw_metadata["poll_count"] == 2
    assert response.raw_metadata["usage"] == {"cost": 0.25}
    assert "unsigned_urls" not in response.raw_metadata

    submit_call = transport.calls[0]
    assert submit_call["method"] == "POST"
    assert submit_call["url"].endswith("/api/v1/videos")
    assert submit_call["headers"]["Authorization"] == "Bearer or-secret"
    assert submit_call["payload"] == {
        "model": "google/veo-3.1",
        "prompt": "A serene mountain landscape at sunset",
    }
    assert [call["method"] for call in transport.calls[1:]] == ["GET", "GET"]
    assert [call["url"] for call in transport.calls[1:]] == [
        "https://openrouter.ai/api/v1/videos/job-123",
        "https://openrouter.ai/api/v1/videos/job-123",
    ]
    download_call = binary_transport.calls[0]
    assert download_call["method"] == "GET"
    assert download_call["url"] == (
        "https://openrouter.ai/api/v1/videos/job-123/content?index=0"
    )
    assert download_call["headers"]["Authorization"] == "Bearer or-secret"
    assert download_call["payload"] is None


def test_openrouter_generate_video_sends_source_image_as_first_frame(
    tmp_path: Path,
) -> None:
    reference_bytes = b"fake-reference"
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(reference_bytes)
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=202,
                payload={"id": "job-123", "status": "pending"},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={"id": "job-123", "status": "completed"},
            ),
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [
            BinaryHttpResponse(
                status_code=200,
                body=b"fake-webm",
                headers={"content-type": "video/webm; charset=binary"},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    response = asyncio.run(
        client.generate_video(
            VideoRequest(
                provider="openrouter",
                model_id="alibaba/wan-2.7",
                prompt="Camera slowly zooms in.",
                source_save_id="save-1",
                source_message_id="message-1",
                source_media_asset_id="media-1",
                source_media_path=reference_path,
            )
        )
    )

    assert response.mime_type == "video/webm"
    assert response.video_bytes == b"fake-webm"
    assert "data:image" not in repr(response.raw_metadata)
    payload = transport.calls[0]["payload"]
    assert payload["model"] == "alibaba/wan-2.7"
    assert payload["prompt"] == "Camera slowly zooms in."
    assert payload["frame_images"] == [
        {
            "type": "image_url",
            "image_url": {
                "url": (
                    "data:image/png;base64,"
                    + base64.b64encode(reference_bytes).decode("ascii")
                ),
            },
            "frame_type": "first_frame",
        }
    ]


def test_openrouter_generate_video_maps_content_policy_failure_to_blocked() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=202,
                payload={"id": "job-123", "status": "pending"},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "id": "job-123",
                    "status": "failed",
                    "error": "Content policy violation",
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=RecordingBinaryTransport([]),
        video_poll_interval=0,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="openrouter",
                    model_id="google/veo-3.1",
                    prompt="blocked prompt",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category is ProviderErrorCategory.CONTENT_BLOCKED
    assert "Content policy violation" in exc_info.value.message


def test_openrouter_generate_video_times_out_polling_pending_job() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=202,
                payload={"id": "job-123", "status": "pending"},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={"id": "job-123", "status": "pending"},
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=RecordingBinaryTransport([]),
        video_poll_interval=0,
        video_timeout=0,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="openrouter",
                    model_id="google/veo-3.1",
                    prompt="slow prompt",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category is ProviderErrorCategory.PROVIDER_ERROR
    assert "timed out" in exc_info.value.message


def test_openrouter_generate_video_rejects_unknown_job_status() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=202,
                payload={"id": "job-123", "status": "pending"},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={"id": "job-123", "status": "wat"},
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=RecordingBinaryTransport([]),
        video_poll_interval=0,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="openrouter",
                    model_id="google/veo-3.1",
                    prompt="prompt",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category is ProviderErrorCategory.PROVIDER_ERROR
    assert "unsupported status: wat" in exc_info.value.message


def test_openrouter_generate_video_rejects_missing_video_content_type() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=202,
                payload={"id": "job-123", "status": "pending"},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={"id": "job-123", "status": "completed"},
            ),
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [BinaryHttpResponse(status_code=200, body=b"fake-video", headers={})]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="openrouter",
                    model_id="google/veo-3.1",
                    prompt="prompt",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category is ProviderErrorCategory.IMAGE_GENERATION_FAILED
    assert "supported video content type" in exc_info.value.message


def test_openrouter_image_decode_rejects_oversized_payload_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bragi.providers.openrouter.MAX_PROVIDER_IMAGE_BYTES",
        8,
    )

    def fail_decode(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("oversized image payload should be rejected before decode")

    encoded = base64.b64encode(b"x" * 9).decode("ascii")
    data_url = "data:image/png;base64," + encoded
    monkeypatch.setattr("bragi.providers.openrouter.base64.b64decode", fail_decode)
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "google/gemini-flash-image",
                            "name": "Gemini Flash Image",
                            "architecture": {"output_modalities": ["image"]},
                        }
                    ]
                },
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "images": [{"image_url": {"url": data_url}}],
                            }
                        }
                    ]
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_image(
                ImageRequest(
                    provider="openrouter",
                    model_id="google/gemini-flash-image",
                    prompt="A candlelit archive door.",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.IMAGE_GENERATION_FAILED
    assert "too large" in exc_info.value.message.casefold() or (
        "exceeds" in exc_info.value.message.casefold()
        or "exceeded" in exc_info.value.message.casefold()
    )


def test_openrouter_generate_image_refreshes_cache_for_image_text_modalities() -> None:
    image_bytes = b"fake-openrouter-image"
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "google/gemini-flash-image",
                            "name": "Gemini Flash Image",
                            "architecture": {
                                "output_modalities": ["text", "image"],
                            },
                        }
                    ]
                },
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "images": [{"image_url": {"url": data_url}}],
                            }
                        }
                    ]
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="openrouter",
                model_id="google/gemini-flash-image",
                prompt="A candlelit archive door.",
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
    )

    assert response.image_bytes == image_bytes
    assert len(transport.calls) == 2
    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[0]["url"].endswith(OPENROUTER_ALL_MODALITIES_MODELS_PATH)
    assert transport.calls[1]["method"] == "POST"
    assert transport.calls[1]["payload"]["modalities"] == ["image", "text"]


def test_openrouter_generate_image_missing_metadata_raises_model_not_found() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "google/imagen",
                            "name": "Imagen",
                            "architecture": {"output_modalities": ["image"]},
                        }
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_image(
                ImageRequest(
                    provider="openrouter",
                    model_id="google/gemini-flash-image",
                    prompt="A candlelit archive door.",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.MODEL_NOT_FOUND
    assert len(transport.calls) == 1
    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[0]["url"].endswith(OPENROUTER_ALL_MODALITIES_MODELS_PATH)


def test_openrouter_generate_image_posts_image_only_modalities_after_listing() -> None:
    image_bytes = b"fake-openrouter-image"
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "google/imagen",
                            "name": "Imagen",
                            "architecture": {"output_modalities": ["image"]},
                        }
                    ]
                },
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "images": [{"image_url": {"url": data_url}}],
                            }
                        }
                    ]
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())
    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="openrouter",
                model_id="google/imagen",
                prompt="A brass observatory under snow.",
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
    )

    assert ProviderCapability.IMAGE_GENERATION in models[0].capabilities
    assert response.image_bytes == image_bytes
    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[1]["method"] == "POST"
    assert transport.calls[1]["payload"]["modalities"] == ["image"]


def test_openrouter_generate_image_posts_image_text_modalities_after_listing() -> None:
    image_bytes = b"fake-openrouter-image"
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "google/gemini-flash-image",
                            "name": "Gemini Flash Image",
                            "architecture": {
                                "output_modalities": ["text", "image"],
                            },
                        }
                    ]
                },
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "images": [{"image_url": {"url": data_url}}],
                            }
                        }
                    ]
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())
    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="openrouter",
                model_id="google/gemini-flash-image",
                prompt="A candlelit archive door.",
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
    )

    assert ProviderCapability.CHAT in models[0].capabilities
    assert ProviderCapability.IMAGE_GENERATION in models[0].capabilities
    assert response.image_bytes == image_bytes
    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[1]["method"] == "POST"
    assert transport.calls[1]["payload"]["modalities"] == ["image", "text"]


def test_openrouter_describe_image_posts_multimodal_chat_request() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "model": "openai/gpt-4o-mini-vision",
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "  A person in a red coat and round glasses.  "
                                )
                            }
                        },
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 8,
                        "total_tokens": 20,
                    },
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.describe_image(
            ImageDescriptionRequest(
                provider="openrouter",
                model_id="openai/gpt-4o-mini-vision",
                image_url="https://cdn.example.test/oracle.png",
                prompt="Describe the character.",
                temperature=0.2,
                max_output_tokens=700,
                openrouter_provider_routing={
                    "preferred_max_latency": {"p90": 3.0},
                },
                openrouter_app_title="Bragi",
            )
        )
    )

    assert response.provider == "openrouter"
    assert response.model_id == "openai/gpt-4o-mini-vision"
    assert response.description == "A person in a red coat and round glasses."
    assert response.token_usage["total"] == 20
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/chat/completions")
    assert (
        call["headers"]["X-OpenRouter-Title"]
        == "Bragi"
    )
    payload = call["payload"]
    assert payload["model"] == "openai/gpt-4o-mini-vision"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 700
    assert payload["provider"] == {
        "preferred_max_latency": {"p90": 3.0},
    }
    assert "modalities" not in payload
    assert "image_config" not in payload
    assert payload["messages"] == [
        {
            "role": "system",
            "content": (
                "Describe the visible physical appearance in the image as natural "
                "prose for consistent future image generation. Include stable "
                "visible details such as skin tone, complexion or undertone, "
                "face shape and features, hair color and texture, eyes, clothing "
                "or styling, posture, and overall presence. When the text prompt "
                "does not specify a visible detail, infer that detail from the "
                "image. Use explicitly provided character context when it "
                "supplies or clearly establishes race, ethnicity, ancestry, "
                "or cultural appearance details, and do not omit those details "
                "from the appearance description."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the character."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://cdn.example.test/oracle.png"},
                },
            ],
        },
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": {"images": []}}]},
        {
            "choices": [
                {
                    "message": {
                        "images": [
                            {
                                "image_url": {
                                    "url": "data:image/png;base64,abc",
                                }
                            }
                        ],
                    }
                }
            ]
        },
    ],
)
def test_openrouter_generate_image_malformed_success_response_raises_image_error(
    payload: dict[str, Any],
) -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "google/gemini-flash-image",
                            "name": "Gemini Flash Image",
                            "architecture": {"output_modalities": ["image"]},
                        }
                    ]
                },
            ),
            JsonHttpResponse(status_code=200, payload=payload),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_image(
                ImageRequest(
                    provider="openrouter",
                    model_id="google/gemini-flash-image",
                    prompt="A candlelit archive door.",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.IMAGE_GENERATION_FAILED
    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[1]["method"] == "POST"


def test_openrouter_missing_api_key_validation_skips_transport() -> None:
    transport = RecordingTransport([])
    client = OpenRouterClient(
        secret_store=InMemorySecretStore(),
        transport=transport,
    )

    status = asyncio.run(client.validate_config())

    assert status.provider == "openrouter"
    assert status.configured is False
    assert status.authenticated is False
    assert transport.calls == []


def test_openrouter_http_error_uses_provider_error_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", no_sleep)
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=429,
                payload={"error": {"message": "slow down"}},
            ),
            JsonHttpResponse(
                status_code=429,
                payload={"error": {"message": "still slow"}},
            ),
            JsonHttpResponse(
                status_code=429,
                payload={"error": {"message": "too slow"}},
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("openrouter", "or-secret")
    client = OpenRouterClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(client.list_models())

    assert exc_info.value.category == ProviderErrorCategory.RATE_LIMITED
    assert exc_info.value.message == "rate_limited (429)"
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_attempt_count == 3
    assert exc_info.value.max_retry_attempts == 3
    assert "slow down" not in exc_info.value.message
    assert len(transport.calls) == 3
