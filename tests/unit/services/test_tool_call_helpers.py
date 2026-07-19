from __future__ import annotations

import json

from bragi.providers.contracts import ProviderToolCall, ToolCallMessage
from bragi.services.tool_call_helpers import (
    accepted_tool_result,
    append_tool_feedback_messages,
    invalid_tool_result,
    parse_tool_arguments_json,
    validate_tool_arguments_shape,
)


def test_parse_tool_arguments_json_accepts_object_arguments() -> None:
    arguments, error = parse_tool_arguments_json('{"source_id":"memory-1"}')

    assert error is None
    assert arguments == {"source_id": "memory-1"}


def test_parse_tool_arguments_json_rejects_malformed_and_non_object_arguments() -> None:
    arguments, error = parse_tool_arguments_json('{"source_id":')

    assert arguments is None
    assert error == "Malformed JSON arguments: Expecting value"

    arguments, error = parse_tool_arguments_json('["memory-1"]')

    assert arguments is None
    assert error == "Tool arguments must be a JSON object"


def test_validate_tool_arguments_shape_checks_required_fields_and_extras() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_id": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["source_id"],
    }

    assert (
        validate_tool_arguments_shape({"confidence": 0.5}, schema=schema)
        == "Missing required field: source_id"
    )
    assert (
        validate_tool_arguments_shape(
            {"source_id": "memory-1", "extra": True},
            schema=schema,
        )
        == "Unexpected field: extra"
    )


def test_validate_tool_arguments_shape_checks_types_enums_and_ranges() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "operation": {"type": "string", "enum": ["upsert", "delete"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "tags": {"type": "array", "items": {"type": "string"}},
            "priority": {"type": "integer"},
        },
        "required": ["source_id"],
    }

    assert (
        validate_tool_arguments_shape({"source_id": 123}, schema=schema)
        == "source_id must be a string"
    )
    assert (
        validate_tool_arguments_shape(
            {"source_id": "memory-1", "operation": "patch"},
            schema=schema,
        )
        == "operation must be one of: upsert, delete"
    )
    assert (
        validate_tool_arguments_shape(
            {"source_id": "memory-1", "confidence": 1.2},
            schema=schema,
        )
        == "confidence must be at most 1"
    )
    assert (
        validate_tool_arguments_shape(
            {"source_id": "memory-1", "tags": ["memory", 7]},
            schema=schema,
        )
        == "tags must contain only strings"
    )
    assert (
        validate_tool_arguments_shape(
            {"source_id": "memory-1", "priority": 1.5},
            schema=schema,
        )
        == "priority must be an integer"
    )


def test_validate_tool_arguments_shape_supports_custom_enum_errors() -> None:
    error = validate_tool_arguments_shape(
        {"source_id": "missing"},
        schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "enum": ["memory-1"]},
            },
        },
        enum_error_formatter=lambda field, value, _allowed: (
            f"{field} must be one of offered values; got {value}"
        ),
    )

    assert error == "source_id must be one of offered values; got missing"


def test_validate_tool_arguments_shape_checks_union_and_array_enums() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "value": {
                "type": ["object", "array", "string", "number", "boolean", "null"]
            },
            "memory_ids": {
                "type": "array",
                "items": {"type": "string", "enum": ["memory-1", "memory-2"]},
            },
        },
        "required": ["value"],
    }

    assert validate_tool_arguments_shape({"value": None}, schema=schema) is None
    assert validate_tool_arguments_shape({"value": {"ok": True}}, schema=schema) is None
    assert (
        validate_tool_arguments_shape(
            {"value": object(), "memory_ids": ["memory-3"]},
            schema=schema,
        )
        == (
            "value must be one of these JSON types: object, array, string, "
            "number, boolean, null"
        )
    )
    assert (
        validate_tool_arguments_shape(
            {"value": "ok", "memory_ids": ["memory-3"]},
            schema=schema,
        )
        == "memory_ids items must be one of: memory-1, memory-2"
    )


def test_append_tool_feedback_messages_preserves_assistant_and_tool_payloads() -> None:
    messages = [ToolCallMessage(role="user", body="Use tools.")]
    call = ProviderToolCall(
        id="call-1",
        name="select_context_source",
        arguments_json='{"source_id":"memory-1"}',
    )

    append_tool_feedback_messages(
        messages,
        assistant_body="",
        tool_calls=(call,),
        tool_results=[(call, accepted_tool_result())],
    )

    assert [message.role for message in messages] == ["user", "assistant", "tool"]
    assert messages[1].tool_calls == (call,)
    assert messages[2].tool_call_id == "call-1"
    assert json.loads(messages[2].body)["status"] == "accepted"


def test_tool_result_helpers_match_feedback_payloads() -> None:
    assert accepted_tool_result() == {
        "status": "accepted",
        "message": "Accepted. Do not repeat this call.",
    }
    assert invalid_tool_result(
        "Unknown tool name: nope",
        retry_instruction="Call this tool again with corrected arguments only.",
    ) == {
        "status": "error",
        "error": "Unknown tool name: nope",
        "retry_instruction": "Call this tool again with corrected arguments only.",
    }
