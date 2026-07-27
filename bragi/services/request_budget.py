"""Hard, model-aware budgets for provider text requests."""

from __future__ import annotations

import json
from dataclasses import replace

from bragi.app_logging import log_event
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.chat_rendering import (
    estimate_chat_request_tokens,
)
from bragi.providers.contracts import (
    ChatRequest,
    StructuredOutputRequest,
    ToolCallRequest,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.token_accounting import estimate_text_tokens

DEFAULT_CHAT_OUTPUT_RESERVE = 1024
DEFAULT_STRUCTURED_OUTPUT_RESERVE = 1024
DEFAULT_TOOL_CALL_OUTPUT_RESERVE = 512
_CHAT_OUTPUT_RESERVES = {
    "character_text": 256,
    "image_prompt": 512,
    "summarization": 256,
}
_STRUCTURED_OUTPUT_RESERVES = {
    "context_search": 128,
    "narrator_message_verification": 512,
}


def budget_chat_request(
    repositories: PersistenceRepositories,
    request: ChatRequest,
    *,
    task: str,
) -> ChatRequest:
    context_window = model_context_window(
        repositories,
        provider=request.provider,
        model_id=request.model_id,
    )
    if context_window is None:
        _log_unenforced_budget(request.provider, request.model_id, task)
        reserved_output_tokens = _reserved_output_tokens(
            request.max_output_tokens,
            _CHAT_OUTPUT_RESERVES.get(task, DEFAULT_CHAT_OUTPUT_RESERVE),
        )
        return _chat_request_with_budget_diagnostics(
            replace(request, max_output_tokens=reserved_output_tokens),
            {
                "task": task,
                "model_context_window": None,
                "reserved_output_tokens": reserved_output_tokens,
                "enforced": False,
                "reason": "no_model_context_window",
                "still_over_budget": None,
            },
        )
    return enforce_chat_request_budget(
        request,
        model_context_window=context_window,
        task=task,
    )


def enforce_chat_request_budget(
    request: ChatRequest,
    *,
    model_context_window: int,
    task: str,
) -> ChatRequest:
    reserved_output_tokens = _reserved_output_tokens(
        request.max_output_tokens,
        _CHAT_OUTPUT_RESERVES.get(task, DEFAULT_CHAT_OUTPUT_RESERVE),
    )
    estimated_input_tokens = estimate_chat_request_tokens(request)
    diagnostics = _budget_diagnostics(
        task=task,
        provider=request.provider,
        model_id=request.model_id,
        model_context_window=model_context_window,
        reserved_output_tokens=reserved_output_tokens,
        estimated_input_tokens=estimated_input_tokens,
    )
    if diagnostics["still_over_budget"] is True:
        raise _overflow_error(diagnostics)
    capped_request = replace(
        request,
        max_output_tokens=reserved_output_tokens,
    )
    return _chat_request_with_budget_diagnostics(capped_request, diagnostics)


def budget_structured_output_request(
    repositories: PersistenceRepositories | None,
    request: StructuredOutputRequest,
    *,
    task: str,
) -> StructuredOutputRequest:
    if repositories is None:
        log_event(
            "provider.request_budget_unenforced",
            provider=request.provider,
            model=request.model_id,
            task=task,
            reason="no_repository_context",
        )
        return replace(
            request,
            max_output_tokens=_reserved_output_tokens(
                request.max_output_tokens,
                _STRUCTURED_OUTPUT_RESERVES.get(
                    task,
                    DEFAULT_STRUCTURED_OUTPUT_RESERVE,
                ),
            ),
        )
    context_window = model_context_window(
        repositories,
        provider=request.provider,
        model_id=request.model_id,
    )
    if context_window is None:
        _log_unenforced_budget(request.provider, request.model_id, task)
        return replace(
            request,
            max_output_tokens=_reserved_output_tokens(
                request.max_output_tokens,
                _STRUCTURED_OUTPUT_RESERVES.get(
                    task,
                    DEFAULT_STRUCTURED_OUTPUT_RESERVE,
                ),
            ),
        )
    return enforce_structured_output_request_budget(
        request,
        model_context_window=context_window,
        task=task,
    )


def enforce_structured_output_request_budget(
    request: StructuredOutputRequest,
    *,
    model_context_window: int,
    task: str,
) -> StructuredOutputRequest:
    reserved_output_tokens = _reserved_output_tokens(
        request.max_output_tokens,
        _STRUCTURED_OUTPUT_RESERVES.get(
            task,
            DEFAULT_STRUCTURED_OUTPUT_RESERVE,
        ),
    )
    estimated_message_tokens = sum(
        estimate_text_tokens(
            json.dumps(
                {
                    "role": message.role,
                    "body": message.body,
                    "speaker_name": message.speaker_name,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        + 4
        for message in request.messages
    )
    schema_text = json.dumps(
        request.schema,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    estimated_schema_tokens = (
        estimate_text_tokens(schema_text)
        + estimate_text_tokens(request.schema_name)
        + 32
    )
    diagnostics = _budget_diagnostics(
        task=task,
        provider=request.provider,
        model_id=request.model_id,
        model_context_window=model_context_window,
        reserved_output_tokens=reserved_output_tokens,
        estimated_input_tokens=(
            estimated_message_tokens + estimated_schema_tokens
        ),
    )
    diagnostics.update(
        {
            "schema_name": request.schema_name,
            "estimated_message_tokens": estimated_message_tokens,
            "estimated_schema_tokens": estimated_schema_tokens,
        }
    )
    if diagnostics["still_over_budget"] is True:
        raise _overflow_error(diagnostics)
    return replace(request, max_output_tokens=reserved_output_tokens)


def budget_tool_call_request(
    repositories: PersistenceRepositories | None,
    request: ToolCallRequest,
    *,
    task: str,
) -> ToolCallRequest:
    if repositories is None:
        log_event(
            "provider.request_budget_unenforced",
            provider=request.provider,
            model=request.model_id,
            task=task,
            reason="no_repository_context",
        )
        return replace(
            request,
            max_output_tokens=_reserved_output_tokens(
                request.max_output_tokens,
                _STRUCTURED_OUTPUT_RESERVES.get(
                    task,
                    DEFAULT_TOOL_CALL_OUTPUT_RESERVE,
                ),
            ),
        )
    context_window = model_context_window(
        repositories,
        provider=request.provider,
        model_id=request.model_id,
    )
    if context_window is None:
        _log_unenforced_budget(request.provider, request.model_id, task)
        return replace(
            request,
            max_output_tokens=_reserved_output_tokens(
                request.max_output_tokens,
                _STRUCTURED_OUTPUT_RESERVES.get(
                    task,
                    DEFAULT_TOOL_CALL_OUTPUT_RESERVE,
                ),
            ),
        )
    return enforce_tool_call_request_budget(
        request,
        model_context_window=context_window,
        task=task,
    )


def enforce_tool_call_request_budget(
    request: ToolCallRequest,
    *,
    model_context_window: int,
    task: str,
) -> ToolCallRequest:
    reserved_output_tokens = _reserved_output_tokens(
        request.max_output_tokens,
        _STRUCTURED_OUTPUT_RESERVES.get(
            task,
            DEFAULT_TOOL_CALL_OUTPUT_RESERVE,
        ),
    )
    estimated_message_tokens = sum(
        estimate_text_tokens(
            json.dumps(
                {
                    "role": message.role,
                    "body": message.body,
                    "speaker_name": message.speaker_name,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "arguments_json": call.arguments_json,
                        }
                        for call in message.tool_calls
                    ],
                    "tool_call_id": message.tool_call_id,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        + 4
        for message in request.messages
    )
    tool_text = json.dumps(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in request.tools
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    estimated_tool_tokens = estimate_text_tokens(tool_text) + 32
    diagnostics = _budget_diagnostics(
        task=task,
        provider=request.provider,
        model_id=request.model_id,
        model_context_window=model_context_window,
        reserved_output_tokens=reserved_output_tokens,
        estimated_input_tokens=(
            estimated_message_tokens + estimated_tool_tokens
        ),
    )
    diagnostics.update(
        {
            "estimated_message_tokens": estimated_message_tokens,
            "estimated_tool_tokens": estimated_tool_tokens,
            "tool_count": len(request.tools),
        }
    )
    if diagnostics["still_over_budget"] is True:
        raise _overflow_error(diagnostics)
    return replace(request, max_output_tokens=reserved_output_tokens)


def model_context_window(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
) -> int | None:
    for model in repositories.list_provider_models(provider):
        if model.model_id == model_id:
            return model.context_window
    return None


def _budget_diagnostics(
    *,
    task: str,
    provider: str,
    model_id: str,
    model_context_window: int,
    reserved_output_tokens: int,
    estimated_input_tokens: int,
) -> dict[str, object]:
    input_limit_tokens = max(0, model_context_window - reserved_output_tokens)
    still_over_budget = estimated_input_tokens > input_limit_tokens
    return {
        "task": task,
        "provider": provider,
        "model_id": model_id,
        "model_context_window": model_context_window,
        "reserved_output_tokens": reserved_output_tokens,
        "input_limit_tokens": input_limit_tokens,
        "estimated_input_tokens": estimated_input_tokens,
        "enforced": True,
        "still_over_budget": still_over_budget,
        "reason": (
            "irreducible_core_context_overflow"
            if still_over_budget
            else "within_model_context_window"
        ),
    }


def _reserved_output_tokens(configured: int | None, default: int) -> int:
    if configured is not None and configured > 0:
        return configured
    return default


def _overflow_error(diagnostics: dict[str, object]) -> ProviderError:
    return ProviderError(
        category=ProviderErrorCategory.CONTEXT_LIMIT_EXCEEDED,
        message=(
            "Provider request core context cannot fit the selected model "
            "after reserving output tokens"
        ),
        diagnostics=diagnostics,
    )


def _chat_request_with_budget_diagnostics(
    request: ChatRequest,
    diagnostics: dict[str, object],
) -> ChatRequest:
    context_breakdown = dict(request.context_breakdown)
    context_breakdown["request_budget"] = diagnostics
    return replace(request, context_breakdown=context_breakdown)


def _log_unenforced_budget(provider: str, model_id: str, task: str) -> None:
    log_event(
        "provider.request_budget_unenforced",
        provider=provider,
        model=model_id,
        task=task,
        reason="no_model_context_window",
    )
