"""Shared provider-agnostic reasoning diagnostics.

Both the OpenRouter and Venice providers surface reasoning-model output in the
OpenAI-compatible chat-completions shape. The helpers in this module normalize
the response so callers can detect a "reasoning-only" or "reasoning-truncated"
response without duplicating the per-provider shape handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReasoningResponseSignals:
    finish_reason: str | None
    reasoning_tokens: int | None
    detail_types: tuple[str, ...]
    completion_tokens: int | None


def extract_reasoning_signals(payload: dict[str, Any]) -> ReasoningResponseSignals:
    """Pull reasoning-related signals from a chat-completions style payload.

    Tolerant of provider-specific keys: OpenRouter exposes
    ``usage.completion_tokens_details.reasoning_tokens`` and
    ``message.reasoning_details``; Venice exposes ``usage.completion_tokens``
    with reasoning fields and a ``reasoning`` text blob on the message.
    """

    choice = _first_choice(payload)
    message = _first_message(choice) if choice is not None else None
    finish_reason = choice.get("finish_reason") if choice is not None else None
    reasoning_tokens = _reasoning_tokens_from_usage(payload.get("usage"))
    detail_types = (
        tuple(_string_list(message.get("reasoning_details"))) if message else ()
    )
    completion_tokens = _completion_tokens(payload.get("usage"))
    return ReasoningResponseSignals(
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        reasoning_tokens=reasoning_tokens,
        detail_types=detail_types,
        completion_tokens=completion_tokens,
    )


def is_reasoning_only_chat_response(
    signals: ReasoningResponseSignals,
    *,
    completion_tokens_ceiling: int = 1,
) -> bool:
    """True when the response is reasoning-only with no visible assistant text.

    A "reasoning-only" chat response is the signature of a reasoning model
    that consumed the entire ``max_output_tokens`` budget on reasoning and
    returned no visible body. The user-visible impact is an empty chat
    message and a slow round trip.
    """

    if signals.finish_reason != "length":
        return False
    if signals.completion_tokens is None:
        return signals.reasoning_tokens is not None or bool(signals.detail_types)
    return (
        signals.completion_tokens <= completion_tokens_ceiling
        and (
            (signals.reasoning_tokens is not None and signals.reasoning_tokens > 0)
            or bool(signals.detail_types)
        )
    )


def is_reasoning_truncated_structured_response(
    signals: ReasoningResponseSignals,
    *,
    completion_tokens_ceiling: int = 1,
) -> bool:
    """True when a structured response was truncated by reasoning.

    Distinct from a normal ``finish_reason="length"`` schema violation: the
    signature is that the visible completion token count is far below the
    requested budget while reasoning consumed the rest. The structured output
    in this case is almost always an empty object or a partial fragment.
    """

    if signals.finish_reason != "length":
        return False
    if signals.completion_tokens is None:
        return False
    if signals.completion_tokens > completion_tokens_ceiling:
        return False
    return (
        signals.reasoning_tokens is not None and signals.reasoning_tokens > 0
    ) or bool(signals.detail_types)


def _first_choice(payload: dict[str, Any]) -> dict[str, Any] | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    return first if isinstance(first, dict) else None


def _first_message(choice: dict[str, Any]) -> dict[str, Any] | None:
    message = choice.get("message")
    return message if isinstance(message, dict) else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str):
            items.append(item)
        elif isinstance(item, dict):
            type_name = item.get("type")
            if isinstance(type_name, str) and type_name:
                items.append(type_name)
    return items


def _reasoning_tokens_from_usage(usage: object) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in (
        "completion_tokens_details",
        "output_tokens_details",
    ):
        details = usage.get(key)
        if not isinstance(details, dict):
            continue
        tokens = details.get("reasoning_tokens")
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            return tokens
    return None


def _completion_tokens(usage: object) -> int | None:
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("completion_tokens")
    if isinstance(tokens, int) and not isinstance(tokens, bool):
        return tokens
    return None
