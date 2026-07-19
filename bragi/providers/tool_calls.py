"""OpenAI-style provider tool-call payload helpers."""

from __future__ import annotations

import json
from typing import Any

from bragi.providers.contracts import (
    ProviderToolCall,
    ToolCallMessage,
    ToolDefinition,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.message_names import provider_message_name


def tool_definition_payload(tool: ToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def tool_message_payload(message: ToolCallMessage) -> dict[str, object]:
    role = {
        "player": "user",
        "narrator": "assistant",
    }.get(message.role, message.role)
    payload: dict[str, object] = {"role": role}
    if message.tool_calls:
        payload["content"] = message.body or None
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments_json,
                },
            }
            for tool_call in message.tool_calls
        ]
    else:
        payload["content"] = message.body
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    safe_name = provider_message_name(message.speaker_name)
    if safe_name and role in {"user", "assistant"}:
        payload["name"] = safe_name
    return payload


def parse_tool_call_response(
    payload: dict[str, Any],
) -> tuple[str, tuple[ProviderToolCall, ...]]:
    message = _first_choice_message(payload)
    body = _message_content(message)
    raw_tool_calls = message.get("tool_calls", [])
    if raw_tool_calls is None:
        raw_tool_calls = []
    if not isinstance(raw_tool_calls, list):
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message="Provider tool_calls must be a list",
        )
    tool_calls: list[ProviderToolCall] = []
    for index, raw_call in enumerate(raw_tool_calls):
        if not isinstance(raw_call, dict):
            raise ProviderError(
                category=ProviderErrorCategory.PROVIDER_ERROR,
                message="Provider tool call must be an object",
            )
        function = raw_call.get("function")
        if not isinstance(function, dict):
            raise ProviderError(
                category=ProviderErrorCategory.PROVIDER_ERROR,
                message="Provider tool call did not include a function object",
            )
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProviderError(
                category=ProviderErrorCategory.PROVIDER_ERROR,
                message="Provider tool call function name is required",
            )
        arguments = function.get("arguments", "")
        raw_id = raw_call.get("id")
        tool_calls.append(
            ProviderToolCall(
                id=raw_id if isinstance(raw_id, str) and raw_id else f"call-{index}",
                name=name,
                arguments_json=_normalize_arguments_json(arguments),
            )
        )
    return body, tuple(tool_calls)


def _first_choice_message(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message="Provider response did not include chat choices",
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message="Provider chat choice must be an object",
        )
    message = first.get("message")
    if not isinstance(message, dict):
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message="Provider chat choice did not include a message",
        )
    return message


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "".join(text_parts)
    raise ProviderError(
        category=ProviderErrorCategory.PROVIDER_ERROR,
        message="Provider tool-call content must be a string",
    )


def _normalize_arguments_json(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict | list):
        try:
            return json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                category=ProviderErrorCategory.PROVIDER_ERROR,
                message="Provider tool-call arguments must be JSON serializable",
            ) from exc
    raise ProviderError(
        category=ProviderErrorCategory.PROVIDER_ERROR,
        message="Provider tool-call arguments must be a JSON string, object, or array",
    )
