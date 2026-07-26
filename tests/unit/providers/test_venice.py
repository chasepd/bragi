from __future__ import annotations

import asyncio
import base64
import json
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
from bragi.providers.venice import (
    VENICE_IMAGE_PROMPT_MAX_CHARS,
    VENICE_VIDEO_PROMPT_MAX_CHARS,
    VeniceClient,
    _parse_chat_content,
    _provider_url,
)
from bragi.services.secrets import InMemorySecretStore, SecretStorageError

VENICE_ALL_MODELS_PATH = "/api/v1/models?type=all"


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
            raise AssertionError("unexpected Venice transport call")
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
            raise AssertionError("unexpected Venice binary transport call")
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


def _json_binary_response(
    payload: dict[str, Any],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> BinaryHttpResponse:
    return BinaryHttpResponse(
        status_code=status_code,
        body=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", **(headers or {})},
    )


def test_venice_validate_config_reports_secret_storage_error() -> None:
    transport = RecordingTransport([])
    client = VeniceClient(
        secret_store=FailingSecretStore(),
        transport=transport,
    )

    status = asyncio.run(client.validate_config())

    assert status.provider == "venice"
    assert status.configured is True
    assert status.authenticated is False
    assert status.error == ProviderErrorCategory.SECRET_STORAGE_ERROR.value
    assert transport.calls == []


def test_venice_list_models_gets_models_with_bearer_auth_and_normalizes() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "llama-3.2-3b",
                            "model_spec": {
                                "availableContextTokens": 131_072,
                                "name": "Llama 3.2 3B",
                                "capabilities": {
                                    "supportsResponseSchema": True,
                                    "supportsVision": True,
                                },
                            },
                            "type": "text",
                        },
                        {
                            "id": "hidream",
                            "model_spec": {"name": "HiDream"},
                            "type": "image",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    call = transport.calls[0]
    assert call["method"] == "GET"
    assert call["url"].endswith(VENICE_ALL_MODELS_PATH)
    assert call["headers"]["Authorization"] == "Bearer venice-secret"
    models_by_id = {model.model_id: model for model in models}
    assert set(models_by_id) == {"llama-3.2-3b", "hidream"}
    assert ProviderCapability.CHAT in models_by_id["llama-3.2-3b"].capabilities
    assert (
        ProviderCapability.STRUCTURED_OUTPUT
        in models_by_id["llama-3.2-3b"].capabilities
    )
    assert ProviderCapability.VISION in models_by_id["llama-3.2-3b"].capabilities
    assert (
        ProviderCapability.IMAGE_GENERATION
        not in models_by_id["llama-3.2-3b"].capabilities
    )
    assert ProviderCapability.IMAGE_GENERATION in models_by_id["hidream"].capabilities
    assert models_by_id["llama-3.2-3b"].context_window == 131_072


def test_venice_list_models_normalizes_reasoning_effort_metadata() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "glm-5",
                            "model_spec": {
                                "name": "GLM 5",
                                "capabilities": {
                                    "supportsReasoning": True,
                                    "supportsReasoningEffort": True,
                                },
                                "constraints": {
                                    "reasoning_effort": {
                                        "allowed_values": ["high", "medium", "low"],
                                        "default": "medium",
                                    }
                                },
                            },
                            "type": "text",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    thinking = models[0].thinking
    assert thinking is not None
    assert thinking.levels == ("high", "medium", "low")
    assert thinking.default_level == "medium"
    assert thinking.default_enabled is True
    assert thinking.mandatory is True
    assert thinking.supports_max_tokens is False


def test_venice_list_models_exposes_text_and_variable_pricing() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "llama-3.2-3b",
                            "model_spec": {
                                "availableContextTokens": 131_072,
                                "name": "Llama 3.2 3B",
                                "pricing": {
                                    "input": {"usd": 0.15},
                                    "output": {"usd": 0.6},
                                    "cache_read": {"usd": 0.01},
                                    "cache_write": {"usd": 0.02},
                                },
                            },
                            "type": "text",
                        },
                        {
                            "id": "gpt-image-2",
                            "model_spec": {
                                "name": "GPT Image 2",
                                "pricing": {
                                    "quality": {
                                        "1K": {
                                            "low": {"usd": 0.02},
                                            "medium": {"usd": 0.07},
                                        }
                                    }
                                },
                            },
                            "type": "image",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    models_by_id = {model.model_id: model for model in models}
    text_pricing = models_by_id["llama-3.2-3b"].pricing
    image_pricing = models_by_id["gpt-image-2"].pricing
    assert text_pricing is not None
    assert text_pricing.input_per_million_tokens_usd == "0.15"
    assert text_pricing.output_per_million_tokens_usd == "0.6"
    assert text_pricing.cache_read_per_million_tokens_usd == "0.01"
    assert text_pricing.cache_write_per_million_tokens_usd == "0.02"
    assert image_pricing is not None
    assert image_pricing.note == "Variable pricing"


@pytest.mark.parametrize(
    "record",
    [
        {
            "id": "venice-trait-signal",
            "model_spec": {"name": "Trait Signal"},
            "type": "text",
            "traits": ["uncensored"],
        },
        {
            "id": "venice-most-uncensored-trait-signal",
            "model_spec": {"name": "Most Uncensored Trait Signal"},
            "type": "text",
            "traits": ["most_uncensored"],
        },
        {
            "id": "venice-tag-signal",
            "model_spec": {"name": "Tag Signal"},
            "type": "text",
            "tags": ["unmoderated"],
        },
        {
            "id": "venice-top-level-capability-signal",
            "model_spec": {"name": "Top Level Capability Signal"},
            "type": "text",
            "capabilities": {"uncensored": True},
        },
        {
            "id": "venice-model-spec-capability-signal",
            "model_spec": {
                "name": "Model Spec Capability Signal",
                "capabilities": {"uncensored": True},
            },
            "type": "text",
        },
        {
            "id": "venice-uncensored-1-2",
            "model_spec": {"name": "Venice 1.2"},
            "type": "text",
            "traits": [],
        },
        {
            "id": "venice-1-2",
            "model_spec": {"name": "Venice Uncensored 1.2"},
            "type": "text",
            "traits": [],
        },
        {
            "id": "venice-1-2-dialogue",
            "model_spec": {
                "name": "Venice 1.2 Dialogue",
                "description": "Built for unfiltered dialogue in roleplay scenes.",
            },
            "type": "text",
            "traits": [],
        },
    ],
)
def test_venice_catalog_signals_mark_blocked_output_fallback_capable(
    record: dict[str, Any],
) -> None:
    transport = RecordingTransport(
        [JsonHttpResponse(status_code=200, payload={"data": [record]})]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert ProviderCapability.CHAT in models[0].capabilities
    assert ProviderCapability.BLOCKED_OUTPUT_FALLBACK in models[0].capabilities


def test_venice_normal_text_model_is_not_blocked_output_fallback_capable() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "llama-3.2-3b",
                            "model_spec": {
                                "name": "Llama 3.2 3B",
                                "description": "General-purpose text model.",
                            },
                            "type": "text",
                            "traits": [],
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert ProviderCapability.CHAT in models[0].capabilities
    assert ProviderCapability.BLOCKED_OUTPUT_FALLBACK not in models[0].capabilities


def test_venice_blocked_output_image_model_is_image_fallback_capable() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "hidream-uncensored",
                            "model_spec": {
                                "name": "HiDream Uncensored",
                                "description": "Unfiltered image generation.",
                            },
                            "type": "image",
                        },
                        {
                            "id": "hidream",
                            "model_spec": {
                                "name": "HiDream",
                                "description": "General image generation.",
                            },
                            "type": "image",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    models_by_id = {model.model_id: model for model in models}
    assert models_by_id["hidream-uncensored"].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_GENERATION,
            ProviderCapability.MODEL_LISTING,
            ProviderCapability.BLOCKED_OUTPUT_FALLBACK,
        }
    )
    assert models_by_id["hidream"].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_GENERATION,
            ProviderCapability.MODEL_LISTING,
        }
    )


def test_venice_edit_model_is_image_to_image_capable() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "qwen-image-2-edit",
                            "model_spec": {
                                "name": "Qwen Image Edit",
                                "description": "Prompt-driven image editing.",
                            },
                            "type": "image",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert models[0].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_TO_IMAGE,
            ProviderCapability.MODEL_LISTING,
        }
    )


def test_venice_inpaint_model_type_is_image_to_image_capable() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "gpt-image-2-edit",
                            "model_spec": {
                                "name": "GPT Image 2 Edit",
                                "description": "Modify existing images.",
                            },
                            "type": "inpaint",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert models[0].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_TO_IMAGE,
            ProviderCapability.MODEL_LISTING,
        }
    )
    assert (
        ProviderGenerationParameter.IMAGE_DIMENSIONS
        in models[0].supported_parameters
    )


def test_venice_video_model_type_constraints_drive_video_capabilities() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "wan-preview-image",
                            "model_spec": {
                                "name": "WAN Preview Image",
                                "constraints": {"model_type": "image-to-video"},
                            },
                            "type": "video",
                        },
                        {
                            "id": "seedance-text",
                            "model_spec": {
                                "name": "Seedance Text",
                                "constraints": {"model_type": "text-to-video"},
                            },
                            "type": "video",
                        },
                        {
                            "id": "wan-reference",
                            "model_spec": {
                                "name": "WAN Reference",
                                "constraints": {"model_type": "reference-to-video"},
                            },
                            "type": "video",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    models_by_id = {model.model_id: model for model in models}
    assert models_by_id["wan-preview-image"].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_TO_VIDEO,
            ProviderCapability.IMAGE_PLUS_TEXT_TO_VIDEO,
            ProviderCapability.MODEL_LISTING,
        }
    )
    assert models_by_id["seedance-text"].capabilities == frozenset(
        {
            ProviderCapability.TEXT_TO_VIDEO,
            ProviderCapability.MODEL_LISTING,
        }
    )
    assert models_by_id["wan-reference"].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_TO_VIDEO,
            ProviderCapability.IMAGE_PLUS_TEXT_TO_VIDEO,
            ProviderCapability.MODEL_LISTING,
        }
    )


def test_venice_list_models_maps_media_and_misc_types() -> None:
    unsupported_records: list[dict[str, Any]] = [
        {
            "id": f"venice-{model_type}",
            "model_spec": {"name": f"Venice {model_type}"},
            "type": model_type,
            "capabilities": ["text"],
        }
        for model_type in (
            "tts",
            "asr",
            "embedding",
            "music",
            "upscale",
            "inpaint",
            "video",
        )
    ]
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "llama-3.2-3b",
                            "model_spec": {
                                "availableContextTokens": 131_072,
                                "name": "Llama 3.2 3B",
                                "capabilities": {
                                    "supportsResponseSchema": True,
                                    "supportsVision": True,
                                },
                            },
                            "type": "text",
                        },
                        {
                            "id": "hidream",
                            "model_spec": {"name": "HiDream"},
                            "type": "image",
                        },
                        *unsupported_records,
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    models_by_id = {model.model_id: model for model in models}
    assert models_by_id["llama-3.2-3b"].capabilities == frozenset(
        {
            ProviderCapability.CHAT,
            ProviderCapability.MODEL_LISTING,
            ProviderCapability.STRUCTURED_OUTPUT,
            ProviderCapability.VISION,
        }
    )
    assert (
        ProviderCapability.IMAGE_GENERATION
        not in models_by_id["llama-3.2-3b"].capabilities
    )
    assert models_by_id["hidream"].capabilities == frozenset(
        {
            ProviderCapability.IMAGE_GENERATION,
            ProviderCapability.MODEL_LISTING,
        }
    )
    for record in unsupported_records:
        if record["type"] == "video":
            assert models_by_id[record["id"]].capabilities == frozenset(
                {
                    ProviderCapability.MODEL_LISTING,
                    ProviderCapability.TEXT_TO_VIDEO,
                }
            )
            continue
        if record["type"] == "inpaint":
            assert models_by_id[record["id"]].capabilities == frozenset(
                {
                    ProviderCapability.IMAGE_TO_IMAGE,
                    ProviderCapability.MODEL_LISTING,
                }
            )
            continue
        assert models_by_id[record["id"]].capabilities == frozenset(
            {ProviderCapability.MODEL_LISTING}
        )


@pytest.mark.parametrize(
    ("capabilities", "expected_structured"),
    [
        ({"supportsResponseSchema": True}, True),
        ({"supportsResponseSchema": False}, False),
        ({}, False),
    ],
)
def test_venice_structured_output_requires_response_schema_capability(
    capabilities: dict[str, object],
    expected_structured: bool,
) -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "llama-3.2-3b",
                            "model_spec": {
                                "availableContextTokens": 131_072,
                                "name": "Llama 3.2 3B",
                                "capabilities": capabilities,
                            },
                            "type": "text",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert ProviderCapability.CHAT in models[0].capabilities
    assert (
        ProviderCapability.STRUCTURED_OUTPUT in models[0].capabilities
    ) is expected_structured


@pytest.mark.parametrize(
    ("capabilities", "expected_tool_calling"),
    [
        ({"supportsFunctionCalling": True}, True),
        ({"supportsFunctionCalling": False}, False),
        ({}, False),
    ],
)
def test_venice_tool_calling_requires_function_calling_capability(
    capabilities: dict[str, object],
    expected_tool_calling: bool,
) -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "llama-3.2-3b",
                            "model_spec": {
                                "availableContextTokens": 131_072,
                                "name": "Llama 3.2 3B",
                                "capabilities": capabilities,
                            },
                            "type": "text",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    assert len(models) == 1
    assert ProviderCapability.CHAT in models[0].capabilities
    assert (
        ProviderCapability.TOOL_CALLING in models[0].capabilities
    ) is expected_tool_calling
    assert ProviderCapability.STRUCTURED_OUTPUT not in models[0].capabilities


def test_venice_models_expose_supported_generation_parameters() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "llama-3.2-3b",
                            "model_spec": {
                                "availableContextTokens": 131_072,
                                "name": "Llama 3.2 3B",
                                "capabilities": {},
                            },
                            "type": "text",
                        },
                        {
                            "id": "hidream",
                            "name": "HiDream",
                            "type": "image",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = {model.model_id: model for model in asyncio.run(client.list_models())}

    assert models["llama-3.2-3b"].supported_parameters == frozenset(
        {
            ProviderGenerationParameter.TEMPERATURE,
            ProviderGenerationParameter.MAX_OUTPUT_TOKENS,
        }
    )
    assert models["hidream"].supported_parameters == frozenset(
        {
            ProviderGenerationParameter.IMAGE_DIMENSIONS,
            ProviderGenerationParameter.IMAGE_SAFE_MODE,
        }
    )


def test_venice_chat_posts_completion_and_parses_usage() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": "The canal fog answers."}},
                    ],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 7,
                        "total_tokens": 16,
                    },
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.chat(
            ChatRequest(
                provider="venice",
                model_id="llama-3.2-3b",
                messages=(ChatMessage(role="player", body="I listen at the canal."),),
            )
        )
    )

    assert response.body == "The canal fog answers."
    assert response.provider == "venice"
    assert response.model_id == "llama-3.2-3b"
    assert response.token_usage["total_tokens"] == 16
    assert response.token_usage["total"] == 16
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer venice-secret"
    payload = call["payload"]
    assert payload["model"] == "llama-3.2-3b"
    assert payload["messages"][0]["role"] == "system"
    assert "Response style:" in payload["messages"][0]["content"]
    assert payload["messages"][1:] == [
        {"role": "user", "content": "I listen at the canal."},
    ]
    assert "response_format" not in payload


def test_venice_chat_sends_reasoning_effort() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": "The canal fog answers."}},
                    ],
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="venice",
                model_id="glm-5",
                messages=(ChatMessage(role="player", body="Think."),),
                reasoning=ChatReasoningConfig(effort="high", exclude=True),
            )
        )
    )

    assert transport.calls[0]["payload"]["reasoning_effort"] == "high"


def test_venice_stream_chat_posts_streaming_completion_and_yields_chunks() -> None:
    stream = RecordingStreamTransport(
        [
            {"choices": [{"delta": {"content": "The canal"}}]},
            {"choices": [{"delta": {"content": " fog answers."}}]},
            {"choices": [], "usage": {"total_tokens": 16}},
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, stream_transport=stream)

    async def collect() -> list[str]:
        chunks = []
        async for chunk in client.stream_chat(
            ChatRequest(
                provider="venice",
                model_id="llama-3.2-3b",
                messages=(ChatMessage(role="player", body="Listen."),),
                temperature=0.7,
                max_output_tokens=128,
            )
        ):
            if chunk.delta:
                chunks.append(chunk.delta)
            if chunk.token_usage:
                assert chunk.token_usage["total"] == 16
        return chunks

    assert asyncio.run(collect()) == ["The canal", " fog answers."]
    call = stream.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer venice-secret"
    payload = call["payload"]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["temperature"] == 0.7
    assert payload["max_completion_tokens"] == 128


def test_venice_chat_renders_current_scene_recap_system_section() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The room stills."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="venice",
                model_id="llama-3.2-3b",
                messages=(ChatMessage(role="player", body="I stay seated."),),
                custom_instructions="Keep responses intimate and concrete.",
                regeneration_feedback="Lean harder into suspicion.",
                current_scene_recap=(
                    "scene.location: Manor dining room\n"
                    "scene.present_characters: Mara, Lord Vale\n"
                    "Recent: Everyone is sitting here after the toast.",
                ),
                character_voice_profiles=(
                    "Lord Vale voice: formal, brittle, clipped politeness.",
                ),
                summary="Long-term summary: Mara survived the border war.",
            )
        )
    )

    messages = transport.calls[0]["payload"]["messages"]
    assert messages[0]["role"] == "system"
    system_body = messages[0]["content"]
    assert "Response style:" in system_body
    assert "- Keep responses reasonably short." in system_body
    assert "- Put dialogue in quotation marks." in system_body
    assert "- Put non-dialogue narration in italics." in system_body
    assert "- Format text messages with > at the beginning of each message." in (
        system_body
    )
    assert system_body.index("Response style:") < system_body.index(
        "Current scene recap:"
    )
    assert (
        "Current scene recap:\n"
        "- scene.location: Manor dining room\n"
        "scene.present_characters: Mara, Lord Vale\n"
        "Recent: Everyone is sitting here after the toast."
    ) in system_body
    assert "Character voice profiles:" in system_body
    assert "Lord Vale voice: formal, brittle" in system_body
    assert "Summary:\n- Long-term summary: Mara survived the border war." in system_body
    assert "Save response guidance:" in system_body
    assert "Keep responses intimate and concrete." in system_body
    assert "Regeneration feedback:" in system_body
    assert "Lean harder into suspicion." in system_body
    assert system_body.index("Character voice profiles:") < system_body.index(
        "Summary:"
    )
    assert system_body.index("Current scene recap:") < system_body.index("Summary:")
    assert system_body.index("Summary:") < system_body.index("Regeneration feedback:")
    assert messages[1] == {"role": "user", "content": "I stay seated."}
    assert "response_format" not in transport.calls[0]["payload"]


def test_venice_chat_renders_timeskip_turn_directive_only_when_present() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "Dawn breaks."}}]},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The room stills."}}]},
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="venice",
                model_id="llama-3.2-3b",
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
                provider="venice",
                model_id="llama-3.2-3b",
                messages=(ChatMessage(role="player", body="I stay seated."),),
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


def test_venice_chat_renders_state_change_and_media_asset_context_sections() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The gate opens."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="venice",
                model_id="llama-3.2-3b",
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
    assert system_body.index("Retrieved state:") < system_body.index(
        "Retrieved state changes:"
    )
    assert system_body.index("Retrieved state changes:") < system_body.index(
        "Retrieved chronicle:"
    )
    assert system_body.index("Retrieved chronicle:") < system_body.index(
        "Retrieved memories:"
    )
    assert system_body.index("Retrieved memories:") < system_body.index(
        "Retrieved media assets:"
    )


def test_venice_structured_output_payload_uses_schema_and_json_hint() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": '{"state_changes": [], "memories": []}'
                            }
                        },
                    ],
                    "usage": {"total_tokens": 11},
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.generate_structured_output(
            StructuredOutputRequest(
                provider="venice",
                model_id="llama-3.2-3b",
                messages=(
                    ChatMessage(
                        role="system",
                        body="Extract deterministic state changes from prose.",
                    ),
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
                temperature=0.1,
                max_output_tokens=300,
            )
        )
    )

    assert response.data == {"state_changes": [], "memories": []}
    payload = transport.calls[0]["payload"]
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "state_memory_update",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "state_changes": {"type": "array"},
                    "memories": {"type": "array"},
                    "metadata": {
                        "type": ["string", "null"],
                        "description": (
                            "Use plain text; do not emit a free-form object."
                        ),
                    },
                },
                "required": ["state_changes", "memories", "metadata"],
                "additionalProperties": False,
            },
        },
    }
    assert any(
        "json" in str(message["content"]).casefold() for message in payload["messages"]
    )


def test_venice_tool_call_payload_uses_tools_and_preserves_arguments() -> None:
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
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.generate_tool_calls(
            ToolCallRequest(
                provider="venice",
                model_id="llama-3.2-3b",
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
            )
        )
    )

    payload = transport.calls[0]["payload"]
    assert payload["tools"][0]["function"]["name"] == "update_scene_snapshot"
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert payload["max_completion_tokens"] == 400
    assert response.tool_calls[0].arguments_json == '{"source_message_id":'


def test_venice_structured_output_retries_empty_success_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", no_sleep)
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": ""}}]},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": '{"state_changes": [], "memories": []}'
                            }
                        },
                    ],
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.generate_structured_output(
            StructuredOutputRequest(
                provider="venice",
                model_id="llama-3.2-3b",
                messages=(
                    ChatMessage(role="user", body="Mara pockets the brass key."),
                ),
                schema_name="state_memory_update",
                schema={
                    "type": "object",
                    "properties": {
                        "state_changes": {"type": "array"},
                        "memories": {"type": "array"},
                    },
                    "required": ["state_changes", "memories"],
                    "additionalProperties": False,
                },
            )
        )
    )

    assert response.data == {"state_changes": [], "memories": []}
    assert len(transport.calls) == 2
    assert [call["method"] for call in transport.calls] == ["POST", "POST"]
    retry_metadata = response.raw_metadata["_bragi_retry"]
    assert retry_metadata["attempt_count"] == 2
    assert retry_metadata["max_attempts"] == 3

    attempts = retry_metadata["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["attempt"] == 1
    assert attempts[0]["error_category"] == ProviderErrorCategory.PROVIDER_ERROR.value
    assert isinstance(attempts[0]["duration_ms"], int)
    assert attempts[1]["attempt"] == 2
    assert attempts[1]["error_category"] is None
    assert isinstance(attempts[1]["duration_ms"], int)


def test_venice_structured_output_enforces_async_transport_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked_to_thread(*_args: object, **_kwargs: object) -> object:
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("bragi.providers.venice.asyncio.to_thread", blocked_to_thread)
    monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", no_sleep)
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, timeout=0.01)

    async def generate() -> None:
        await client.generate_structured_output(
            StructuredOutputRequest(
                provider="venice",
                model_id="llama-3.2-3b",
                messages=(ChatMessage(role="user", body="Mara opens the door."),),
                schema_name="state_memory_update",
                schema={
                    "type": "object",
                    "properties": {"state_changes": {"type": "array"}},
                    "required": ["state_changes"],
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


def test_venice_chat_content_preserves_provider_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_with_diagnostics(_payload: dict[str, Any]) -> str:
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "reasoning-only response",
            diagnostics={"finish_reason": "length", "reasoning_tokens": 20},
        )

    monkeypatch.setattr("bragi.providers.venice._chat_content", fail_with_diagnostics)

    with pytest.raises(ProviderError) as exc_info:
        _parse_chat_content({"choices": []})

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR
    assert exc_info.value.message == "reasoning-only response"
    assert exc_info.value.diagnostics == {
        "finish_reason": "length",
        "reasoning_tokens": 20,
    }


def test_venice_chat_normalizes_provider_message_names() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "The crews answer."}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.chat(
            ChatRequest(
                provider="venice",
                model_id="llama-3.2-3b",
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


def test_venice_chat_malformed_success_response_raises_provider_error() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": ["not text"]}}]},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.chat(
                ChatRequest(
                    provider="venice",
                    model_id="llama-3.2-3b",
                    messages=(ChatMessage(role="player", body="Hello?"),),
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR
    assert "content must be a string" in exc_info.value.message


def test_venice_chat_accepts_multimodal_text_content() -> None:
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
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.chat(
            ChatRequest(
                provider="venice",
                model_id="llama-3.2-3b",
                messages=(ChatMessage(role="player", body="Hello?"),),
            )
        )
    )

    assert response.body == "The canal fog answers."


def test_venice_chat_retries_transient_failure_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", no_sleep)
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=500,
                payload={"error": {"message": "upstream failed"}},
            ),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": "The canal fog answers."}},
                    ],
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.chat(
            ChatRequest(
                provider="venice",
                model_id="llama-3.2-3b",
                messages=(ChatMessage(role="player", body="I listen."),),
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
    assert attempts[0]["error_category"] == ProviderErrorCategory.PROVIDER_ERROR.value
    assert isinstance(attempts[0]["duration_ms"], int)
    assert attempts[1]["attempt"] == 2
    assert attempts[1]["error_category"] is None
    assert isinstance(attempts[1]["duration_ms"], int)


def test_venice_chat_does_not_retry_authentication_failure() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=401,
                payload={"error": {"message": "bad key"}},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.chat(
                ChatRequest(
                    provider="venice",
                    model_id="llama-3.2-3b",
                    messages=(ChatMessage(role="player", body="Hello?"),),
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.AUTHENTICATION_FAILED
    assert len(transport.calls) == 1


def test_venice_generate_image_posts_native_request_and_decodes_images_shape() -> None:
    image_bytes = b"fake-venice-image"
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="venice",
                model_id="hidream",
                prompt="A narrow canal.",
                source_save_id="save-1",
                source_message_id="message-1",
                dimensions=(1024, 768),
            )
        )
    )

    assert response.provider == "venice"
    assert response.model_id == "hidream"
    assert response.image_bytes == image_bytes
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/image/generate")
    assert call["headers"]["Authorization"] == "Bearer venice-secret"
    payload = call["payload"]
    assert payload["model"] == "hidream"
    assert payload["prompt"] == "A narrow canal."
    assert payload["format"] == "png"
    assert payload["return_binary"] is False
    assert payload["safe_mode"] is True
    assert payload["width"] == 1024
    assert payload["height"] == 768
    assert "aspect_ratio" not in payload
    assert "response_format" not in payload
    assert "size" not in payload


@pytest.mark.parametrize(
    ("safe_mode", "expected_payload_value"),
    [
        (None, True),
        (False, False),
        (True, True),
    ],
)
def test_venice_generate_image_posts_native_safe_mode(
    safe_mode: bool | None,
    expected_payload_value: bool | None,
) -> None:
    image_bytes = b"fake-venice-image"
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="venice",
                model_id="hidream",
                prompt="A narrow canal.",
                source_save_id="save-1",
                source_message_id="message-1",
                safe_mode=safe_mode,
            )
        )
    )

    payload = transport.calls[0]["payload"]
    assert payload["safe_mode"] is expected_payload_value


def test_venice_generate_image_uses_aspect_ratio_for_ratio_models() -> None:
    image_bytes = b"fake-venice-image"
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="venice",
                model_id="qwen-image-2",
                prompt="A narrow canal.",
                source_save_id="save-1",
                source_message_id="message-1",
                dimensions=(1024, 768),
            )
        )
    )

    assert response.provider == "venice"
    assert response.model_id == "qwen-image-2"
    assert response.image_bytes == image_bytes
    payload = transport.calls[0]["payload"]
    assert payload["model"] == "qwen-image-2"
    assert payload["aspect_ratio"] == "4:3"
    assert "width" not in payload
    assert "height" not in payload


def test_venice_image_to_image_posts_edit_request(
    tmp_path: Path,
) -> None:
    reference_bytes = b"fake-reference-png"
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(reference_bytes)
    binary_transport = RecordingBinaryTransport(
        [
            BinaryHttpResponse(
                status_code=200,
                body=b"fake-edited-image",
                headers={"content-type": "image/png"},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=RecordingTransport([]),
        binary_transport=binary_transport,
    )

    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="venice",
                model_id="qwen-image-2-edit",
                prompt="Keep the character identity while changing the pose.",
                source_save_id="save-1",
                source_message_id="message-1",
                source_media_asset_id="reference-asset",
                source_media_path=reference_path,
                dimensions=(1024, 768),
            )
        )
    )

    assert response.provider == "venice"
    assert response.model_id == "qwen-image-2-edit"
    assert response.image_bytes == b"fake-edited-image"
    call = binary_transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/image/edit")
    payload = call["payload"]
    assert payload["model"] == "qwen-image-2-edit"
    assert payload["prompt"] == "Keep the character identity while changing the pose."
    assert payload["image"] == base64.b64encode(reference_bytes).decode("ascii")
    assert payload["output_format"] == "png"
    assert payload["aspect_ratio"] == "4:3"
    assert payload["safe_mode"] is True


def test_venice_image_to_image_with_disabled_safe_mode_uses_multi_edit(
    tmp_path: Path,
) -> None:
    reference_bytes = b"fake-reference-png"
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(reference_bytes)
    binary_transport = RecordingBinaryTransport(
        [
            BinaryHttpResponse(
                status_code=200,
                body=b"fake-edited-image",
                headers={"content-type": "image/png"},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=RecordingTransport([]),
        binary_transport=binary_transport,
    )

    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="venice",
                model_id="qwen-image-2-edit",
                prompt="Keep the character identity while changing the pose.",
                source_save_id="save-1",
                source_message_id="message-1",
                source_media_asset_id="reference-asset",
                source_media_path=reference_path,
                dimensions=(1024, 768),
                safe_mode=False,
            )
        )
    )

    assert response.raw_metadata["source_media_asset_id"] == "reference-asset"
    call = binary_transport.calls[0]
    assert call["url"].endswith("/api/v1/image/multi-edit")
    payload = call["payload"]
    assert payload["modelId"] == "qwen-image-2-edit"
    assert payload["prompt"] == "Keep the character identity while changing the pose."
    assert payload["images"] == [base64.b64encode(reference_bytes).decode("ascii")]
    assert payload["output_format"] == "png"
    assert payload["aspect_ratio"] == "4:3"
    assert payload["safe_mode"] is False


def test_venice_generate_image_posts_multi_edit_references(tmp_path: Path) -> None:
    first_reference = tmp_path / "first.png"
    second_reference = tmp_path / "second.png"
    first_reference.write_bytes(b"first-reference")
    second_reference.write_bytes(b"second-reference")
    binary_transport = RecordingBinaryTransport(
        [
            BinaryHttpResponse(
                status_code=200,
                body=b"fake-edited-image",
                headers={"content-type": "image/png"},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=RecordingTransport([]),
        binary_transport=binary_transport,
    )

    response = asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="venice",
                model_id="qwen-image-2-edit",
                prompt="Keep both character identities in the scene.",
                source_save_id="save-1",
                source_message_id="message-1",
                source_media_asset_ids=("first-asset", "second-asset"),
                source_media_paths=(first_reference, second_reference),
                dimensions=(1024, 768),
                safe_mode=False,
            )
        )
    )

    assert response.image_bytes == b"fake-edited-image"
    assert response.raw_metadata["source_media_asset_ids"] == [
        "first-asset",
        "second-asset",
    ]
    call = binary_transport.calls[0]
    assert call["url"].endswith("/api/v1/image/multi-edit")
    payload = call["payload"]
    assert payload["modelId"] == "qwen-image-2-edit"
    assert payload["prompt"] == "Keep both character identities in the scene."
    assert payload["images"] == [
        base64.b64encode(b"first-reference").decode("ascii"),
        base64.b64encode(b"second-reference").decode("ascii"),
    ]
    assert payload["output_format"] == "png"
    assert payload["aspect_ratio"] == "4:3"
    assert payload["safe_mode"] is False


def test_venice_generate_video_queues_polls_and_reads_inline_mp4() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "model": "wan-2.5-preview-text-to-video",
                    "queue_id": "queue-123",
                },
                headers={"x-request-id": "queue-req"},
            )
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [
            _json_binary_response(
                {
                    "status": "PROCESSING",
                    "average_execution_time": 145000,
                    "execution_duration": 53200,
                },
                headers={"x-request-id": "retrieve-1"},
            ),
            BinaryHttpResponse(
                status_code=200,
                body=b"fake-mp4",
                headers={
                    "content-type": "video/mp4",
                    "x-request-id": "retrieve-2",
                    "authorization": "Bearer leaked",
                },
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    response = asyncio.run(
        client.generate_video(
            VideoRequest(
                provider="venice",
                model_id="wan-2.5-preview-text-to-video",
                prompt="A gondola glides through moonlit fog.",
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
    )

    assert response.provider == "venice"
    assert response.model_id == "wan-2.5-preview-text-to-video"
    assert response.mime_type == "video/mp4"
    assert response.video_bytes == b"fake-mp4"
    assert response.raw_metadata["queue_id"] == "queue-123"
    assert response.raw_metadata["status"] == "COMPLETED"
    assert response.raw_metadata["poll_count"] == 2
    assert response.raw_metadata["_bragi_headers"] == {
        "content-type": "video/mp4",
        "x-request-id": "retrieve-2",
    }
    assert response.raw_metadata["queue_headers"] == {"x-request-id": "queue-req"}
    assert "prompt" not in response.raw_metadata

    queue_call = transport.calls[0]
    assert queue_call["method"] == "POST"
    assert queue_call["url"].endswith("/api/v1/video/queue")
    assert queue_call["headers"]["Authorization"] == "Bearer venice-secret"
    assert queue_call["payload"] == {
        "model": "wan-2.5-preview-text-to-video",
        "prompt": "A gondola glides through moonlit fog.",
        "duration": "5s",
    }
    assert [call["method"] for call in binary_transport.calls] == ["POST", "POST"]
    assert [call["url"] for call in binary_transport.calls] == [
        "https://api.venice.ai/api/v1/video/retrieve",
        "https://api.venice.ai/api/v1/video/retrieve",
    ]
    assert binary_transport.calls[0]["payload"] == {
        "model": "wan-2.5-preview-text-to-video",
        "queue_id": "queue-123",
    }


def test_venice_generate_video_sends_source_image_data_url(tmp_path: Path) -> None:
    reference_bytes = b"fake-reference-png"
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(reference_bytes)
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"model": "wan-2.5-preview-image-to-video", "queue_id": "q"},
            )
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
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    response = asyncio.run(
        client.generate_video(
            VideoRequest(
                provider="venice",
                model_id="wan-2.5-preview-image-to-video",
                prompt="Camera slowly pushes in.",
                source_save_id="save-1",
                source_message_id="message-1",
                source_media_asset_id="media-1",
                source_media_path=reference_path,
                safe_mode=True,
            )
        )
    )

    assert response.video_bytes == b"fake-mp4"
    assert "data:image" not in repr(response.raw_metadata)
    payload = transport.calls[0]["payload"]
    assert payload == {
        "model": "wan-2.5-preview-image-to-video",
        "prompt": "Camera slowly pushes in.",
        "duration": "5s",
        "image_url": (
            "data:image/png;base64,"
            + base64.b64encode(reference_bytes).decode("ascii")
        ),
        "safe_mode": True,
    }


def test_venice_generate_video_sends_reference_image_for_reference_models(
    tmp_path: Path,
) -> None:
    reference_bytes = b"fake-reference-png"
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(reference_bytes)
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"model": "wan-2-7-reference-to-video", "queue_id": "q"},
            )
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
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    response = asyncio.run(
        client.generate_video(
            VideoRequest(
                provider="venice",
                model_id="wan-2-7-reference-to-video",
                prompt="Camera slowly pushes in.",
                source_save_id="save-1",
                source_message_id="message-1",
                source_media_asset_id="media-1",
                source_media_path=reference_path,
            )
        )
    )

    assert response.video_bytes == b"fake-mp4"
    payload = transport.calls[0]["payload"]
    assert payload["model"] == "wan-2-7-reference-to-video"
    assert payload["prompt"].startswith("@Image1 ")
    assert "Camera slowly pushes in." in payload["prompt"]
    assert payload["reference_image_urls"] == [
        "data:image/png;base64," + base64.b64encode(reference_bytes).decode("ascii")
    ]
    assert "image_url" not in payload


def test_venice_generate_video_caps_prompt_to_default_video_limit() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "model": "wan-2.5-preview-text-to-video",
                    "queue_id": "queue-123",
                },
            )
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
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    asyncio.run(
        client.generate_video(
            VideoRequest(
                provider="venice",
                model_id="wan-2.5-preview-text-to-video",
                prompt="front " + ("middle " * 500) + "back",
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
    )

    prompt = transport.calls[0]["payload"]["prompt"]
    assert len(prompt) <= VENICE_VIDEO_PROMPT_MAX_CHARS
    assert prompt.startswith("front ")
    assert prompt.endswith("back")
    assert "middle " * 500 not in prompt


def test_venice_video_queue_error_includes_safe_validation_details() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=400,
                payload={
                    "detail": [
                        {"msg": "prompt must be less than 2500 characters"},
                        {
                            "msg": (
                                "At least one reference is required for this model"
                            )
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=RecordingBinaryTransport([]),
        video_poll_interval=0,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="venice",
                    model_id="wan-2-7-reference-to-video",
                    prompt="Camera slowly pushes in.",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR
    assert exc_info.value.status_code == 400
    assert "prompt must be less than 2500 characters" in exc_info.value.message
    assert "At least one reference is required" in exc_info.value.message


def test_venice_generate_video_downloads_private_url_without_metadata() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "model": "grok-imagine-text-to-video-private",
                    "queue_id": "private-q",
                    "download_url": "https://private-share.venice.ai/v1/share/read/token",
                },
            )
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [
            _json_binary_response({"status": "PROCESSING"}),
            _json_binary_response({"status": "COMPLETED"}),
            BinaryHttpResponse(
                status_code=200,
                body=b"private-mp4",
                headers={"content-type": "video/mp4", "x-request-id": "download-req"},
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    response = asyncio.run(
        client.generate_video(
            VideoRequest(
                provider="venice",
                model_id="grok-imagine-text-to-video-private",
                prompt="A lantern swings in the rain.",
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
    )

    assert response.video_bytes == b"private-mp4"
    assert response.raw_metadata["queue_id"] == "private-q"
    assert response.raw_metadata["status"] == "COMPLETED"
    assert response.raw_metadata["poll_count"] == 2
    assert response.raw_metadata["_bragi_headers"] == {
        "content-type": "video/mp4",
        "x-request-id": "download-req",
    }
    assert "download_url" not in response.raw_metadata
    assert "private-share.venice.ai" not in repr(response.raw_metadata)
    download_call = binary_transport.calls[2]
    assert download_call["method"] == "GET"
    assert download_call["url"] == (
        "https://private-share.venice.ai/v1/share/read/token"
    )
    assert "Authorization" not in download_call["headers"]
    assert download_call["payload"] is None


@pytest.mark.parametrize(
    "download_url",
    [
        "http://private-share.venice.ai/v1/share/read/token",
        "https://127.0.0.1/v1/share/read/token",
        "https://[::1]/v1/share/read/token",
        "https://169.254.169.254/latest/meta-data",
        "https://evil.example/v1/share/read/token",
        "https://private-share.venice.ai.evil.example/v1/share/read/token",
        "https://user:pass@private-share.venice.ai/v1/share/read/token",
        "https://private-share.venice.ai:444/v1/share/read/token",
        "https://private-share.venice.ai:bad/v1/share/read/token",
        "https://private-share.venice.ai/not-share/token",
    ],
)
def test_venice_generate_video_rejects_unsafe_private_download_url(
    download_url: str,
) -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "model": "grok-imagine-text-to-video-private",
                    "queue_id": "private-q",
                    "download_url": download_url,
                },
            )
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [
            _json_binary_response({"status": "PROCESSING"}),
            _json_binary_response({"status": "COMPLETED"}),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="venice",
                    model_id="grok-imagine-text-to-video-private",
                    prompt="A lantern swings in the rain.",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.IMAGE_GENERATION_FAILED
    assert "unsafe private download URL" in exc_info.value.message
    assert download_url not in exc_info.value.message
    assert len(binary_transport.calls) == 2


def test_venice_provider_url_rejects_unapproved_absolute_urls() -> None:
    with pytest.raises(ProviderError) as exc_info:
        _provider_url("https://api.venice.ai/api/v1", "https://evil.example/models")

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR
    assert "relative" in exc_info.value.message


def test_venice_generate_video_times_out_polling_pending_job() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"model": "wan-2.5-preview-text-to-video", "queue_id": "q"},
            )
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [_json_binary_response({"status": "PROCESSING"})]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
        video_timeout=0,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="venice",
                    model_id="wan-2.5-preview-text-to-video",
                    prompt="slow prompt",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category is ProviderErrorCategory.PROVIDER_ERROR
    assert "timed out" in exc_info.value.message


def test_venice_generate_video_extends_timeout_from_provider_timing_estimate() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"model": "wan-2.5-preview-text-to-video", "queue_id": "q"},
            )
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [
            _json_binary_response(
                {
                    "status": "PROCESSING",
                    "average_execution_time": 2_234_099,
                    "execution_duration": 1_161_472,
                }
            ),
            BinaryHttpResponse(
                status_code=200,
                body=b"fake-mp4",
                headers={"content-type": "video/mp4"},
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
        video_timeout=0,
    )

    response = asyncio.run(
        client.generate_video(
            VideoRequest(
                provider="venice",
                model_id="wan-2.5-preview-text-to-video",
                prompt="slow prompt",
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
    )

    assert response.video_bytes == b"fake-mp4"
    assert len(binary_transport.calls) == 2
    assert response.raw_metadata["poll_count"] == 2


def test_venice_generate_video_rejects_unsupported_status() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"model": "wan-2.5-preview-text-to-video", "queue_id": "q"},
            )
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [_json_binary_response({"status": "WHAT"})]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="venice",
                    model_id="wan-2.5-preview-text-to-video",
                    prompt="prompt",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category is ProviderErrorCategory.PROVIDER_ERROR
    assert "unsupported status: WHAT" in exc_info.value.message


def test_venice_generate_video_rejects_malformed_queue_response() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"model": "wan-2.5-preview-text-to-video"},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=RecordingBinaryTransport([]),
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="venice",
                    model_id="wan-2.5-preview-text-to-video",
                    prompt="prompt",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category is ProviderErrorCategory.PROVIDER_ERROR
    assert "queue_id" in exc_info.value.message


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"content-type": "video/webm"},
        {"content-type": "application/octet-stream"},
    ],
)
def test_venice_generate_video_rejects_non_mp4_content_type(
    headers: dict[str, str],
) -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"model": "wan-2.5-preview-text-to-video", "queue_id": "q"},
            )
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [BinaryHttpResponse(status_code=200, body=b"not-mp4", headers=headers)]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="venice",
                    model_id="wan-2.5-preview-text-to-video",
                    prompt="prompt",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category is ProviderErrorCategory.IMAGE_GENERATION_FAILED
    assert "video/mp4" in exc_info.value.message


def test_venice_generate_video_maps_content_policy_failure_to_blocked() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={"model": "wan-2.5-preview-text-to-video", "queue_id": "q"},
            )
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [
            _json_binary_response(
                {"status": "FAILED", "error": "Content policy violation"}
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_video(
                VideoRequest(
                    provider="venice",
                    model_id="wan-2.5-preview-text-to-video",
                    prompt="blocked prompt",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category is ProviderErrorCategory.CONTENT_BLOCKED
    assert "Content policy violation" in exc_info.value.message


def test_venice_generate_video_preserves_retry_metadata_from_video_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.0)
    transport = RecordingTransport(
        [
            JsonHttpResponse(status_code=500, payload={"error": "queue unavailable"}),
            JsonHttpResponse(
                status_code=200,
                payload={
                    "model": "grok-imagine-text-to-video-private",
                    "queue_id": "private-q",
                    "download_url": "https://private-share.venice.ai/v1/share/read/t",
                },
            ),
        ]
    )
    binary_transport = RecordingBinaryTransport(
        [
            BinaryHttpResponse(
                status_code=503,
                body=b'{"status":"PROCESSING"}',
                headers={"content-type": "application/json"},
            ),
            _json_binary_response({"status": "COMPLETED"}),
            BinaryHttpResponse(
                status_code=500,
                body=b"",
                headers={"content-type": "text/plain"},
            ),
            BinaryHttpResponse(
                status_code=200,
                body=b"retry-mp4",
                headers={"content-type": "video/mp4"},
            ),
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(
        secret_store=secrets,
        transport=transport,
        binary_transport=binary_transport,
        video_poll_interval=0,
    )

    response = asyncio.run(
        client.generate_video(
            VideoRequest(
                provider="venice",
                model_id="grok-imagine-text-to-video-private",
                prompt="A storm rolls over the harbor.",
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
    )

    assert response.video_bytes == b"retry-mp4"
    assert response.raw_metadata["queue_retry"]["attempt_count"] == 2
    assert response.raw_metadata["retrieve_retry"]["attempt_count"] == 2
    assert response.raw_metadata["download_retry"]["attempt_count"] == 2
    assert response.raw_metadata["_bragi_retry"] == response.raw_metadata[
        "download_retry"
    ]


def test_venice_video_models_only_expose_safe_mode_when_metadata_supports_it() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "video-safe",
                            "model_spec": {
                                "name": "Video Safe",
                                "capabilities": {"supportsSafeMode": True},
                            },
                            "type": "video",
                        },
                        {
                            "id": "video-plain",
                            "model_spec": {
                                "name": "Video Plain",
                                "capabilities": {"supportsAudioConfig": True},
                            },
                            "type": "video",
                        },
                    ]
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    models = asyncio.run(client.list_models())

    models_by_id = {model.model_id: model for model in models}
    assert (
        ProviderGenerationParameter.IMAGE_SAFE_MODE
        in models_by_id["video-safe"].supported_parameters
    )
    assert (
        ProviderGenerationParameter.IMAGE_SAFE_MODE
        not in models_by_id["video-plain"].supported_parameters
    )


def test_venice_describe_image_posts_multimodal_chat_request() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "model": "qwen-vision",
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "  A tall character with silver hair and a blue "
                                    "cloak.  "
                                )
                            }
                        },
                    ],
                    "usage": {
                        "prompt_tokens": 14,
                        "completion_tokens": 10,
                        "total_tokens": 24,
                    },
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    response = asyncio.run(
        client.describe_image(
            ImageDescriptionRequest(
                provider="venice",
                model_id="qwen-vision",
                image_url="https://cdn.example.test/oracle.png",
                prompt="Describe the character.",
                temperature=0.2,
                max_output_tokens=700,
            )
        )
    )

    assert response.provider == "venice"
    assert response.model_id == "qwen-vision"
    assert response.description == "A tall character with silver hair and a blue cloak."
    assert response.token_usage["total"] == 24
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/chat/completions")
    payload = call["payload"]
    assert payload["model"] == "qwen-vision"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.2
    assert payload["max_completion_tokens"] == 700
    assert "safe_mode" not in payload
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


def test_venice_generate_image_caps_prompt_to_provider_limit() -> None:
    image_bytes = b"fake-venice-image"
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=200,
                payload={
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                },
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)
    prompt = (
        "opening detail "
        + ("middle " * VENICE_IMAGE_PROMPT_MAX_CHARS)
        + "closing detail"
    )

    asyncio.run(
        client.generate_image(
            ImageRequest(
                provider="venice",
                model_id="hidream",
                prompt=prompt,
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
    )

    submitted_prompt = transport.calls[0]["payload"]["prompt"]
    assert isinstance(submitted_prompt, str)
    assert len(prompt) > VENICE_IMAGE_PROMPT_MAX_CHARS
    assert len(submitted_prompt) <= VENICE_IMAGE_PROMPT_MAX_CHARS
    assert submitted_prompt.startswith("opening detail")
    assert submitted_prompt.endswith("closing detail")
    assert transport.calls[0]["url"].endswith("/api/v1/image/generate")


@pytest.mark.parametrize(
    "payload",
    [
        {"images": []},
        {"images": ["abc"]},
    ],
)
def test_venice_generate_image_malformed_success_response_raises_image_error(
    payload: dict[str, Any],
) -> None:
    transport = RecordingTransport([JsonHttpResponse(status_code=200, payload=payload)])
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_image(
                ImageRequest(
                    provider="venice",
                    model_id="hidream",
                    prompt="A narrow canal.",
                    source_save_id="save-1",
                    source_message_id="message-1",
                )
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.IMAGE_GENERATION_FAILED


def test_venice_image_decode_rejects_oversized_payload_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bragi.providers.venice.MAX_PROVIDER_IMAGE_BYTES",
        8,
    )

    def fail_decode(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("oversized image payload should be rejected before decode")

    encoded = base64.b64encode(b"x" * 9).decode("ascii")
    monkeypatch.setattr("bragi.providers.venice.base64.b64decode", fail_decode)
    transport = RecordingTransport(
        [JsonHttpResponse(status_code=200, payload={"images": [encoded]})]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            client.generate_image(
                ImageRequest(
                    provider="venice",
                    model_id="hidream",
                    prompt="A narrow canal.",
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


def test_venice_missing_api_key_validation_skips_transport() -> None:
    transport = RecordingTransport([])
    client = VeniceClient(
        secret_store=InMemorySecretStore(),
        transport=transport,
    )

    status = asyncio.run(client.validate_config())

    assert status.provider == "venice"
    assert status.configured is False
    assert status.authenticated is False
    assert transport.calls == []


def test_venice_http_error_uses_provider_error_category() -> None:
    transport = RecordingTransport(
        [
            JsonHttpResponse(
                status_code=401,
                payload={"error": {"message": "bad key"}},
            )
        ]
    )
    secrets = InMemorySecretStore()
    secrets.set_api_key("venice", "venice-secret")
    client = VeniceClient(secret_store=secrets, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(client.list_models())

    assert exc_info.value.category == ProviderErrorCategory.AUTHENTICATION_FAILED
    assert exc_info.value.message == "authentication_failed (401)"
    assert "bad key" not in exc_info.value.message
