"""Save-scoped turn responsiveness policy."""

from __future__ import annotations

from dataclasses import replace

from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatReasoningConfig,
    StructuredOutputRequest,
    ToolCallRequest,
)
from bragi.retry_policy import RetryExecutionClass, current_retry_execution_class

TURN_RESPONSIVENESS_MODE_SETTING = "turn_responsiveness_mode"
TURN_RESPONSIVENESS_MODE_QUALITY = "quality"
TURN_RESPONSIVENESS_MODE_RESPONSIVE = "responsive"
TURN_RESPONSIVENESS_MODE_OPTIONS = (
    TURN_RESPONSIVENESS_MODE_QUALITY,
    TURN_RESPONSIVENESS_MODE_RESPONSIVE,
)
DEFAULT_TURN_RESPONSIVENESS_MODE = TURN_RESPONSIVENESS_MODE_QUALITY

RESPONSIVE_STRUCTURED_HELPER_MAX_OUTPUT_TOKENS = 2_048
RESPONSIVE_PLANNER_MESSAGE_WINDOW = 8


def sanitize_turn_responsiveness_mode(value: object) -> str:
    """Return a supported mode, preserving quality as the safe default."""

    if value in TURN_RESPONSIVENESS_MODE_OPTIONS:
        return str(value)
    return DEFAULT_TURN_RESPONSIVENESS_MODE


def turn_responsiveness_mode(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None,
) -> str:
    try:
        return sanitize_turn_responsiveness_mode(
            repositories.get_effective_setting(
                TURN_RESPONSIVENESS_MODE_SETTING,
                save_id=save_id,
            )
        )
    except Exception:  # noqa: BLE001 - storage failures retain quality behavior
        return DEFAULT_TURN_RESPONSIVENESS_MODE


def retry_execution_class_for_save(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> RetryExecutionClass:
    if (
        turn_responsiveness_mode(repositories, save_id=save_id)
        == TURN_RESPONSIVENESS_MODE_RESPONSIVE
    ):
        return RetryExecutionClass.RESPONSIVE_FOREGROUND
    return RetryExecutionClass.QUALITY_FOREGROUND


def responsive_structured_helper_request(
    repositories: PersistenceRepositories | None,
    request: StructuredOutputRequest,
) -> StructuredOutputRequest:
    """Apply responsive-only output and optional-reasoning limits."""

    if (
        current_retry_execution_class()
        is not RetryExecutionClass.RESPONSIVE_FOREGROUND
    ):
        return request
    configured_max = request.max_output_tokens
    max_output_tokens = (
        RESPONSIVE_STRUCTURED_HELPER_MAX_OUTPUT_TOKENS
        if configured_max is None
        else min(configured_max, RESPONSIVE_STRUCTURED_HELPER_MAX_OUTPUT_TOKENS)
    )
    reasoning = request.reasoning
    if not _model_requires_thinking(
        repositories,
        provider=request.provider,
        model_id=request.model_id,
    ):
        reasoning = ChatReasoningConfig(effort="none", exclude=True)
    return replace(
        request,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
    )


def responsive_tool_helper_request(
    repositories: PersistenceRepositories | None,
    request: ToolCallRequest,
) -> ToolCallRequest:
    """Apply the same responsive helper policy to tool-enforced output."""

    if (
        current_retry_execution_class()
        is not RetryExecutionClass.RESPONSIVE_FOREGROUND
    ):
        return request
    configured_max = request.max_output_tokens
    max_output_tokens = (
        RESPONSIVE_STRUCTURED_HELPER_MAX_OUTPUT_TOKENS
        if configured_max is None
        else min(configured_max, RESPONSIVE_STRUCTURED_HELPER_MAX_OUTPUT_TOKENS)
    )
    reasoning = request.reasoning
    if not _model_requires_thinking(
        repositories,
        provider=request.provider,
        model_id=request.model_id,
    ):
        reasoning = ChatReasoningConfig(effort="none", exclude=True)
    return replace(
        request,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
    )


def _model_requires_thinking(
    repositories: PersistenceRepositories | None,
    *,
    provider: str,
    model_id: str,
) -> bool:
    if repositories is None:
        return False
    return any(
        model.model_id == model_id
        and model.available
        and model.thinking.get("mandatory") is True
        for model in repositories.list_provider_models(provider)
    )
