from __future__ import annotations

import pytest

from bragi.providers.structured_output_validation import (
    StructuredOutputValidationError,
    validate_structured_output,
)


def test_validate_structured_output_reports_typed_contract_diagnostics() -> None:
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["accepted", "rejected"]},
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "required": ["status", "source_ids"],
        "additionalProperties": False,
    }

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        validate_structured_output(
            {"status": "maybe", "source_ids": []},
            schema=schema,
            schema_name="context_selection",
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["schema_name"] == "context_selection"
    assert diagnostics["error_count"] == 2
    errors = diagnostics["errors"]
    assert isinstance(errors, list)
    assert {error["validator"] for error in errors} == {"enum", "minItems"}
    assert all("instance_value" not in error for error in errors)


def test_validate_structured_output_accepts_nested_valid_data() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    validate_structured_output(
        {"items": [{"id": "source-1"}]},
        schema=schema,
        schema_name="nested",
    )


def test_validation_diagnostics_do_not_echo_provider_property_names() -> None:
    provider_property = "private-provider-property"

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        validate_structured_output(
            {provider_property: {"value": 17}},
            schema={
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
            schema_name="safe_diagnostics",
        )

    assert provider_property not in repr(exc_info.value.diagnostics)
