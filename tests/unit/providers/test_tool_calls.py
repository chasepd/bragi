from __future__ import annotations

import pytest

from bragi.providers.contracts import (
    ProviderToolCall,
    ToolCallMessage,
    ToolDefinition,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.tool_calls import (
    parse_tool_call_response,
    tool_definition_payload,
    tool_message_payload,
)


def test_tool_definition_payload_formats_openai_function_tool() -> None:
    tool = ToolDefinition(
        name="record_memory",
        description="Records durable memories.",
        parameters={
            "type": "object",
            "properties": {"body": {"type": "string"}},
            "required": ["body"],
            "additionalProperties": False,
        },
    )

    assert tool_definition_payload(tool) == {
        "type": "function",
        "function": {
            "name": "record_memory",
            "description": "Records durable memories.",
            "parameters": {
                "type": "object",
                "properties": {"body": {"type": "string"}},
                "required": ["body"],
                "additionalProperties": False,
            },
        },
    }


def test_tool_message_payload_maps_roles_and_safe_names() -> None:
    payload = tool_message_payload(
        ToolCallMessage(
            role="player",
            body="I mark the bridge stone.",
            speaker_name="  Mara Vale, Signal-Warden!  ",
        )
    )

    assert payload == {
        "role": "user",
        "content": "I mark the bridge stone.",
        "name": "Mara_Vale_Signal-Warden",
    }


def test_tool_message_payload_formats_assistant_tool_calls_with_null_content() -> None:
    payload = tool_message_payload(
        ToolCallMessage(
            role="narrator",
            body="",
            speaker_name="Narrator",
            tool_calls=(
                ProviderToolCall(
                    id="call-1",
                    name="update_scene_snapshot",
                    arguments_json='{"location":"bridge"}',
                ),
            ),
        )
    )

    assert payload == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "update_scene_snapshot",
                    "arguments": '{"location":"bridge"}',
                },
            }
        ],
        "name": "Narrator",
    }


def test_tool_message_payload_includes_tool_call_id_for_tool_result() -> None:
    payload = tool_message_payload(
        ToolCallMessage(
            role="tool",
            body="Accepted memory.",
            speaker_name="Result ignored for tool role",
            tool_call_id="call-1",
        )
    )

    assert payload == {
        "role": "tool",
        "content": "Accepted memory.",
        "tool_call_id": "call-1",
    }


def test_parse_tool_call_response_accepts_ids_and_generates_fallback_ids() -> None:
    body, tool_calls = parse_tool_call_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "Done.",
                        "tool_calls": [
                            {
                                "id": "provider-call",
                                "function": {
                                    "name": "record_memory",
                                    "arguments": '{"body":"The bell rang."}',
                                },
                            },
                            {
                                "id": "",
                                "function": {
                                    "name": "update_scene_snapshot",
                                    "arguments": {"location": "bridge"},
                                },
                            },
                        ],
                    }
                }
            ]
        }
    )

    assert body == "Done."
    assert tool_calls == (
        ProviderToolCall(
            id="provider-call",
            name="record_memory",
            arguments_json='{"body":"The bell rang."}',
        ),
        ProviderToolCall(
            id="call-1",
            name="update_scene_snapshot",
            arguments_json='{"location":"bridge"}',
        ),
    )


def test_parse_tool_call_response_serializes_list_arguments_as_json() -> None:
    _body, tool_calls = parse_tool_call_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "record_observations",
                                    "arguments": [
                                        {"kind": "scene", "value": "bridge"},
                                        "lantern",
                                    ],
                                }
                            }
                        ],
                    }
                }
            ]
        }
    )

    assert tool_calls == (
        ProviderToolCall(
            id="call-0",
            name="record_observations",
            arguments_json='[{"kind":"scene","value":"bridge"},"lantern"]',
        ),
    )


@pytest.mark.parametrize("arguments", [None, 42, False])
def test_parse_tool_call_response_rejects_scalar_arguments(arguments: object) -> None:
    with pytest.raises(ProviderError) as exc_info:
        parse_tool_call_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "record_memory",
                                        "arguments": arguments,
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR


@pytest.mark.parametrize(
    "tool_calls_value",
    [
        None,
        [],
    ],
)
def test_parse_tool_call_response_accepts_empty_tool_calls(
    tool_calls_value: object,
) -> None:
    body, tool_calls = parse_tool_call_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": tool_calls_value,
                    }
                }
            ]
        }
    )

    assert body == ""
    assert tool_calls == ()


def test_parse_tool_call_response_joins_multimodal_text_content() -> None:
    body, tool_calls = parse_tool_call_response(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "First "},
                            {"type": "image_url", "image_url": {"url": "ignored"}},
                            {"type": "text", "text": "second."},
                        ],
                        "tool_calls": [],
                    }
                }
            ]
        }
    )

    assert body == "First second."
    assert tool_calls == ()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": None},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {"tool_calls": "not a list"}}]},
        {"choices": [{"message": {"tool_calls": ["not an object"]}}]},
        {"choices": [{"message": {"tool_calls": [{}]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {}}]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {"name": "  "}}]}}]},
    ],
)
def test_parse_tool_call_response_rejects_malformed_provider_payloads(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ProviderError) as exc_info:
        parse_tool_call_response(payload)

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR


def test_parse_tool_call_response_rejects_non_object_choice() -> None:
    with pytest.raises(ProviderError) as exc_info:
        parse_tool_call_response({"choices": ["not an object"]})

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR


def test_parse_tool_call_response_rejects_non_text_content() -> None:
    with pytest.raises(ProviderError) as exc_info:
        parse_tool_call_response(
            {"choices": [{"message": {"content": [{"type": "image_url"}]}}]}
        )

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR
