from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import pytest

from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ChatPromptPurpose,
    ChatRequest,
    StructuredOutputRequest,
    ToolCallMessage,
    ToolCallRequest,
    ToolDefinition,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.request_budget import (
    budget_chat_request,
    budget_structured_output_request,
    budget_tool_call_request,
    enforce_chat_request_budget,
    enforce_structured_output_request_budget,
    enforce_tool_call_request_budget,
)


def test_unknown_windows_still_apply_provider_output_caps() -> None:
    repository_mock = Mock()
    repository_mock.list_provider_models.return_value = []
    repositories = cast(PersistenceRepositories, repository_mock)

    chat = budget_chat_request(
        repositories,
        ChatRequest(
            provider="fake",
            model_id="unknown",
            messages=(ChatMessage(role="user", body="Continue."),),
        ),
        task="character_text",
    )
    structured = budget_structured_output_request(
        repositories,
        StructuredOutputRequest(
            provider="fake",
            model_id="unknown",
            messages=(ChatMessage(role="user", body="Select."),),
            schema_name="selection",
            schema={"type": "object", "additionalProperties": False},
        ),
        task="context_search",
    )
    tool = budget_tool_call_request(
        repositories,
        ToolCallRequest(
            provider="fake",
            model_id="unknown",
            messages=(ToolCallMessage(role="user", body="Select."),),
            tools=(
                ToolDefinition(
                    name="select",
                    description="Select.",
                    parameters={"type": "object", "additionalProperties": False},
                ),
            ),
        ),
        task="context_search",
    )

    assert chat.max_output_tokens == 256
    assert structured.max_output_tokens == 128
    assert tool.max_output_tokens == 128


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


def test_chat_budget_sets_provider_output_cap_from_reserve() -> None:
    request = ChatRequest(
        provider="fake",
        model_id="chat",
        messages=(ChatMessage(role="user", body="Continue."),),
    )

    budgeted = enforce_chat_request_budget(
        request,
        model_context_window=4096,
        task="character_text",
    )

    assert budgeted.max_output_tokens == 256


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


def test_tool_call_budget_reserves_output_and_tool_schema_tokens() -> None:
    request = ToolCallRequest(
        provider="fake",
        model_id="tools",
        messages=(ToolCallMessage(role="user", body="Select matching context."),),
        tools=(
            ToolDefinition(
                name="select_context",
                description="Select matching context.",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": "string",
                            "enum": ["source-" + "x" * 800],
                        }
                    },
                    "required": ["source_id"],
                    "additionalProperties": False,
                },
            ),
        ),
        max_output_tokens=64,
    )

    with pytest.raises(ProviderError) as exc_info:
        enforce_tool_call_request_budget(
            request,
            model_context_window=200,
            task="context_search",
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["reserved_output_tokens"] == 64
    estimated_tool_tokens = diagnostics["estimated_tool_tokens"]
    assert isinstance(estimated_tool_tokens, int)
    assert estimated_tool_tokens > 100
    assert diagnostics["still_over_budget"] is True


def test_structured_and_tool_budgets_set_provider_output_caps() -> None:
    structured = enforce_structured_output_request_budget(
        StructuredOutputRequest(
            provider="fake",
            model_id="structured",
            messages=(ChatMessage(role="user", body="Select."),),
            schema_name="selection",
            schema={"type": "object", "additionalProperties": False},
        ),
        model_context_window=4096,
        task="context_search",
    )
    tool = enforce_tool_call_request_budget(
        ToolCallRequest(
            provider="fake",
            model_id="tools",
            messages=(ToolCallMessage(role="user", body="Select."),),
            tools=(
                ToolDefinition(
                    name="select",
                    description="Select.",
                    parameters={"type": "object", "additionalProperties": False},
                ),
            ),
        ),
        model_context_window=4096,
        task="context_search",
    )

    assert structured.max_output_tokens == 128
    assert tool.max_output_tokens == 128


def test_unknown_task_budgets_apply_default_output_reserves() -> None:
    chat = enforce_chat_request_budget(
        ChatRequest(
            provider="fake",
            model_id="chat",
            messages=(ChatMessage(role="user", body="Continue."),),
        ),
        model_context_window=8192,
        task="unknown_task",
    )
    structured = enforce_structured_output_request_budget(
        StructuredOutputRequest(
            provider="fake",
            model_id="structured",
            messages=(ChatMessage(role="user", body="Select."),),
            schema_name="selection",
            schema={"type": "object", "additionalProperties": False},
        ),
        model_context_window=8192,
        task="unknown_task",
    )
    tool = enforce_tool_call_request_budget(
        ToolCallRequest(
            provider="fake",
            model_id="tools",
            messages=(ToolCallMessage(role="user", body="Select."),),
            tools=(
                ToolDefinition(
                    name="select",
                    description="Select.",
                    parameters={"type": "object", "additionalProperties": False},
                ),
            ),
        ),
        model_context_window=16384,
        task="unknown_task",
    )

    assert chat.max_output_tokens == 2048
    assert structured.max_output_tokens == 2048
    assert tool.max_output_tokens == 1024
