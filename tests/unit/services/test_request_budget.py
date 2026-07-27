from __future__ import annotations

import pytest

from bragi.providers.contracts import (
    ChatMessage,
    ChatPromptPurpose,
    ChatRequest,
    StructuredOutputRequest,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.request_budget import (
    enforce_chat_request_budget,
    enforce_structured_output_request_budget,
)


def test_chat_budget_rejects_irreducible_core_before_dispatch() -> None:
    request = ChatRequest(
        provider="fake",
        model_id="tiny",
        messages=(ChatMessage(role="user", body="界" * 300),),
        prompt_purpose=ChatPromptPurpose.CHARACTER_TEXT,
        max_output_tokens=64,
    )

    with pytest.raises(ProviderError) as exc_info:
        enforce_chat_request_budget(
            request,
            model_context_window=100,
            task="character_text",
        )

    assert exc_info.value.category == ProviderErrorCategory.CONTEXT_LIMIT_EXCEEDED
    assert exc_info.value.diagnostics["task"] == "character_text"
    assert exc_info.value.diagnostics["model_context_window"] == 100
    assert exc_info.value.diagnostics["still_over_budget"] is True


def test_structured_budget_reserves_output_and_schema_tokens() -> None:
    request = StructuredOutputRequest(
        provider="fake",
        model_id="structured",
        messages=(ChatMessage(role="user", body="Select the matching source."),),
        schema_name="selection",
        schema={
            "type": "object",
            "properties": {
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["source-" + "x" * 800]},
                }
            },
            "required": ["source_ids"],
            "additionalProperties": False,
        },
        max_output_tokens=64,
    )

    with pytest.raises(ProviderError) as exc_info:
        enforce_structured_output_request_budget(
            request,
            model_context_window=200,
            task="context_search",
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["schema_name"] == "selection"
    assert diagnostics["reserved_output_tokens"] == 64
    estimated_schema_tokens = diagnostics["estimated_schema_tokens"]
    assert isinstance(estimated_schema_tokens, int)
    assert estimated_schema_tokens > 100
    assert diagnostics["still_over_budget"] is True
