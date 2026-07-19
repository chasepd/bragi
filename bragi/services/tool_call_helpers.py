"""Shared helpers for provider tool-call validation and feedback."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from bragi.providers.contracts import ProviderToolCall, ToolCallMessage

DEFAULT_TOOL_RETRY_INSTRUCTION = (
    "Call exactly one tool again with corrected arguments only."
)
CONTEXT_SEARCH_TOOL_RETRY_INSTRUCTION = (
    "Call this tool again with corrected arguments only."
)

EnumErrorFormatter = Callable[[str, object, list[object]], str]


def parse_tool_arguments_json(
    arguments_json: str,
) -> tuple[dict[str, object] | None, str | None]:
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return None, f"Malformed JSON arguments: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, "Tool arguments must be a JSON object"
    return cast(dict[str, object], parsed), None


def accepted_tool_result() -> dict[str, str]:
    return {
        "status": "accepted",
        "message": "Accepted. Do not repeat this call.",
    }


def invalid_tool_result(
    error: str,
    *,
    retry_instruction: str = DEFAULT_TOOL_RETRY_INSTRUCTION,
) -> dict[str, str]:
    return {
        "status": "error",
        "error": error,
        "retry_instruction": retry_instruction,
    }


def append_tool_feedback_messages(
    messages: list[ToolCallMessage],
    *,
    assistant_body: str,
    tool_calls: tuple[ProviderToolCall, ...],
    tool_results: list[tuple[ProviderToolCall, dict[str, str]]],
) -> None:
    messages.append(
        ToolCallMessage(
            role="assistant",
            body=assistant_body,
            tool_calls=tool_calls,
        )
    )
    for call, result in tool_results:
        messages.append(
            ToolCallMessage(
                role="tool",
                body=json.dumps(result, sort_keys=True),
                tool_call_id=call.id,
            )
        )


def validate_tool_arguments_shape(
    arguments: dict[str, object],
    *,
    schema: dict[str, object],
    enum_error_formatter: EnumErrorFormatter | None = None,
    skip_enum_fields: frozenset[str] = frozenset(),
) -> str | None:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return "Tool schema properties must be an object"
    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    for field_name in required:
        if not isinstance(field_name, str):
            continue
        if field_name not in arguments:
            return f"Missing required field: {field_name}"
        if isinstance(arguments.get(field_name), str) and not cast(
            str,
            arguments[field_name],
        ).strip():
            return f"Required field must not be blank: {field_name}"
    if schema.get("additionalProperties") is False:
        allowed = {key for key in properties if isinstance(key, str)}
        extra = sorted(set(arguments) - allowed)
        if extra:
            return f"Unexpected field: {extra[0]}"
    for field_name, value in arguments.items():
        raw_field_schema = properties.get(field_name)
        if not isinstance(raw_field_schema, dict):
            continue
        field_schema = cast(dict[str, object], raw_field_schema)
        type_error = _validate_json_type(
            field_name,
            value,
            field_schema,
            enum_error_formatter=enum_error_formatter,
            skip_enum_fields=skip_enum_fields,
        )
        if type_error is not None:
            return type_error
        enum_values = field_schema.get("enum")
        if (
            field_name not in skip_enum_fields
            and isinstance(enum_values, list)
            and value not in enum_values
        ):
            if enum_error_formatter is not None:
                return enum_error_formatter(field_name, value, enum_values)
            allowed_values = ", ".join(map(str, enum_values))
            return f"{field_name} must be one of: {allowed_values}"
        range_error = _validate_number_range(field_name, value, field_schema)
        if range_error is not None:
            return range_error
    return None


def _validate_json_type(
    field_name: str,
    value: object,
    schema: dict[str, object],
    *,
    enum_error_formatter: EnumErrorFormatter | None,
    skip_enum_fields: frozenset[str],
) -> str | None:
    expected = schema.get("type")
    if isinstance(expected, list):
        return _validate_json_type_union(
            field_name,
            value,
            expected,
            schema,
            enum_error_formatter=enum_error_formatter,
            skip_enum_fields=skip_enum_fields,
        )
    if expected == "string":
        return None if isinstance(value, str) else f"{field_name} must be a string"
    if expected == "boolean":
        return None if isinstance(value, bool) else f"{field_name} must be a boolean"
    if expected == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return None
        return f"{field_name} must be an integer"
    if expected == "number":
        if isinstance(value, int | float) and not isinstance(value, bool):
            return None
        return f"{field_name} must be a number"
    if expected == "array":
        if not isinstance(value, list):
            return f"{field_name} must be an array"
        return _validate_array_items(
            field_name,
            value,
            schema,
            enum_error_formatter=enum_error_formatter,
            skip_enum_fields=skip_enum_fields,
        )
    if expected == "object":
        return None if isinstance(value, dict) else f"{field_name} must be an object"
    if expected == "null":
        return None if value is None else f"{field_name} must be null"
    return None


def _validate_json_type_union(
    field_name: str,
    value: object,
    expected: list[object],
    schema: dict[str, object],
    *,
    enum_error_formatter: EnumErrorFormatter | None,
    skip_enum_fields: frozenset[str],
) -> str | None:
    for item_type in expected:
        if item_type == "null" and value is None:
            return None
        if not isinstance(item_type, str):
            continue
        item_schema = dict(schema)
        item_schema["type"] = item_type
        if (
            _validate_json_type(
                field_name,
                value,
                item_schema,
                enum_error_formatter=enum_error_formatter,
                skip_enum_fields=skip_enum_fields,
            )
            is None
        ):
            return None
    expected_text = ", ".join(str(item) for item in expected)
    return f"{field_name} must be one of these JSON types: {expected_text}"


def _validate_array_items(
    field_name: str,
    value: list[object],
    schema: dict[str, object],
    *,
    enum_error_formatter: EnumErrorFormatter | None,
    skip_enum_fields: frozenset[str],
) -> str | None:
    item_schema = schema.get("items", {})
    if not isinstance(item_schema, dict):
        return None
    item_type = item_schema.get("type")
    if item_type == "string" and any(not isinstance(item, str) for item in value):
        return f"{field_name} must contain only strings"
    enum_values = item_schema.get("enum")
    if item_type == "string" and isinstance(enum_values, list):
        for item in value:
            if item not in enum_values:
                if enum_error_formatter is not None:
                    return enum_error_formatter(field_name, item, enum_values)
                allowed_values = ", ".join(map(str, enum_values))
                return f"{field_name} items must be one of: {allowed_values}"
    if item_type == "object":
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                return f"{field_name}[{index}] must be an object"
            error = validate_tool_arguments_shape(
                cast(dict[str, object], item),
                schema=item_schema,
                enum_error_formatter=enum_error_formatter,
                skip_enum_fields=skip_enum_fields,
            )
            if error is not None:
                return f"{field_name}[{index}]: {error}"
    return None


def _validate_number_range(
    field_name: str,
    value: object,
    schema: dict[str, object],
) -> str | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    minimum = schema.get("minimum")
    if isinstance(minimum, int | float) and value < minimum:
        return f"{field_name} must be at least {minimum}"
    maximum = schema.get("maximum")
    if isinstance(maximum, int | float) and value > maximum:
        return f"{field_name} must be at most {maximum}"
    return None
