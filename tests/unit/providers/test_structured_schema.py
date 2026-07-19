from __future__ import annotations

from typing import Any, cast

import pytest

from bragi.persistence.models import MessageRecord
from bragi.providers.contracts import (
    ChatMessage,
    StructuredOutputRequest,
)
from bragi.providers.structured_schema import (
    normalize_strict_json_schema,
    validate_strict_json_schema,
)
from bragi.services import (
    context_update_service,
    scenario_evolution_service,
    state_service,
)


def test_maintenance_schemas_are_strict_provider_schemas() -> None:
    messages = (
        MessageRecord(
            id="player-message",
            save_id="save-1",
            role="player",
            body="I open the cabinet.",
            speaker_name="Mara",
            provider=None,
            model=None,
            token_estimate=None,
            deleted_at=None,
        ),
        MessageRecord(
            id="narrator-message",
            save_id="save-1",
            role="narrator",
            body="The cabinet holds a chipped mug.",
            speaker_name="Narrator",
            provider="fake",
            model="fake-chat",
            token_estimate=None,
            deleted_at=None,
        ),
    )
    schemas = (
        state_service._state_extraction_schema(
            state_service.StateExtractionRequest(
                save_id="save-1",
                messages=messages,
                current_state=(),
            )
        ),
        context_update_service._context_update_schema(messages),
        scenario_evolution_service._scenario_evolution_schema(
            allowed_sections=("current_scene",),
            messages=messages,
        ),
    )

    for schema in schemas:
        validate_strict_json_schema(schema)

    state_properties = cast(dict[str, Any], schemas[0]["properties"])
    state_changes_schema = cast(dict[str, Any], state_properties["state_changes"])
    state_change_items = cast(dict[str, Any], state_changes_schema["items"])
    state_change_properties = cast(dict[str, Any], state_change_items["properties"])
    state_value_schema = cast(dict[str, Any], state_change_properties["value"])
    assert state_value_schema["type"] == "string"
    conflicts_schema = cast(dict[str, Any], state_properties["conflicts"])
    conflict_items = cast(dict[str, Any], conflicts_schema["items"])
    conflict_properties = cast(dict[str, Any], conflict_items["properties"])
    current_value_schema = cast(dict[str, Any], conflict_properties["current_value"])
    proposed_value_schema = cast(dict[str, Any], conflict_properties["proposed_value"])
    assert current_value_schema["type"] == ["string", "null"]
    assert proposed_value_schema["type"] == ["string", "null"]


def test_structured_output_request_normalizes_schema_before_provider_use() -> None:
    request = StructuredOutputRequest(
        provider="fake",
        model_id="fake-structured",
        messages=(ChatMessage(role="user", body="Extract facts."),),
        schema_name="loose_payload",
        schema={
            "type": "object",
            "properties": {
                "payload": {"type": "object"},
                "note": {"type": "string"},
            },
            "required": ["note"],
        },
    )

    validate_strict_json_schema(request.schema)
    assert request.schema["additionalProperties"] is False
    assert request.schema["required"] == ["payload", "note"]
    request_properties = cast(dict[str, Any], request.schema["properties"])
    payload_schema = cast(dict[str, Any], request_properties["payload"])
    assert payload_schema["type"] == ["string", "null"]


def test_strict_schema_normalizes_combinator_branches() -> None:
    normalized = normalize_strict_json_schema(
        {
            "type": "object",
            "properties": {
                "payload": {
                    "anyOf": [
                        {"type": "object", "additionalProperties": True},
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"label": {"type": "string"}},
                                "required": ["label"],
                            },
                        },
                        {"type": "string"},
                    ]
                }
            },
            "required": ["payload"],
        }
    )

    validate_strict_json_schema(normalized)
    root_properties = cast(dict[str, Any], normalized["properties"])
    payload_schema = cast(dict[str, Any], root_properties["payload"])
    branches = cast(list[dict[str, Any]], payload_schema["anyOf"])
    assert branches[0]["type"] == "string"
    assert "free-form object" in branches[0]["description"]
    array_items = cast(dict[str, Any], branches[1]["items"])
    assert array_items["additionalProperties"] is False
    assert array_items["required"] == ["label"]


def test_strict_schema_validation_rejects_loose_combinator_branches() -> None:
    with pytest.raises(ValueError, match=r"\$\.properties\.payload\.anyOf\[0\]"):
        validate_strict_json_schema(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "payload": {
                        "anyOf": [
                            {"type": "object", "additionalProperties": True},
                            {"type": "string"},
                        ]
                    }
                },
                "required": ["payload"],
            }
        )
