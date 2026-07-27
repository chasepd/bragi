"""Hard, model-aware budgets for provider text requests."""

from __future__ import annotations

import json
from dataclasses import replace

from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.chat_rendering import (
    estimate_chat_request_tokens,
)
from bragi.providers.contracts import ChatRequest, StructuredOutputRequest
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.token_accounting import estimate_text_tokens

DEFAULT_CHAT_OUTPUT_RESERVE = 1024
DEFAULT_STRUCTURED_OUTPUT_RESERVE = 1024
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
        return _chat_request_with_budget_diagnostics(
            request,
            {
                "task": task,
                "model_context_window": None,
                "enforced": False,
                "reason": "no_model_context_window",
                "still_over_budget": False,
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
    return _chat_request_with_budget_diagnostics(request, diagnostics)


def budget_structured_output_request(
    repositories: PersistenceRepositories,
    request: StructuredOutputRequest,
    *,
    task: str,
) -> StructuredOutputRequest:
    context_window = model_context_window(
        repositories,
        provider=request.provider,
        model_id=request.model_id,
    )
    if context_window is None:
        return request
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
    messages_text = "\n\n".join(
        f"{message.role}:\n{message.body}" for message in request.messages
    )
    schema_text = json.dumps(
        request.schema,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    estimated_message_tokens = estimate_text_tokens(messages_text)
    estimated_schema_tokens = estimate_text_tokens(schema_text)
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
    return request


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
