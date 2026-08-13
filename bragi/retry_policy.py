"""Central, admin-configurable retry policy.

The user-facing setting counts retries after the initial operation. Provider
and model-output callers convert that value to total attempts by adding one;
deferred retry jobs keep the literal retry count in their payloads.

A separate per-call deadline bounds the total wall time across all attempts.
Without it, a misbehaving provider can stretch a single call out to
``max_attempts * per_attempt_timeout`` (seven minutes by default) which is
far longer than a human user is willing to wait for one step of a regenerate
turn.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

RETRY_COUNT_SETTING = "retry_count"
DEFAULT_RETRY_COUNT = 6
MIN_RETRY_COUNT = 0
MAX_RETRY_COUNT = 10
RETRY_COUNT_STEP = 1

PROVIDER_CALL_DEADLINE_SETTING = "provider_call_deadline_seconds"
DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS = 120.0
MIN_PROVIDER_CALL_DEADLINE_SECONDS = 5.0
MAX_PROVIDER_CALL_DEADLINE_SECONDS = 600.0
PROVIDER_CALL_DEADLINE_STEP = 1.0

# Compatibility defaults for low-level callers that do not have a repository
# (including standalone provider-client tests).
PROVIDER_MAX_ATTEMPTS = DEFAULT_RETRY_COUNT + 1
MODEL_OUTPUT_MAX_ATTEMPTS = DEFAULT_RETRY_COUNT + 1
DEFERRED_WORK_MAX_ATTEMPTS = DEFAULT_RETRY_COUNT + 1

RESPONSIVE_FOREGROUND_MAX_PROVIDER_ATTEMPTS = 2
RESPONSIVE_FOREGROUND_CALL_DEADLINE_SECONDS = 45.0
RESPONSIVE_FOREGROUND_VERIFICATION_MAX_ATTEMPTS = 2


class RetryExecutionClass(StrEnum):
    """Request class used to resolve internal retry and deadline budgets."""

    QUALITY_FOREGROUND = "quality_foreground"
    RESPONSIVE_FOREGROUND = "responsive_foreground"
    BACKGROUND = "background"


@dataclass(frozen=True)
class ResolvedRetryBudget:
    provider_max_attempts: int
    provider_call_deadline_seconds: float
    automatic_turn_retry_allowed: bool
    verification_max_attempts: int


_RETRY_EXECUTION_CLASS: ContextVar[RetryExecutionClass] = ContextVar(
    "bragi_retry_execution_class",
    default=RetryExecutionClass.QUALITY_FOREGROUND,
)


def sanitize_retry_count(value: object) -> int:
    """Return a safe retry count for persisted or API-provided values."""

    if not isinstance(value, int) or isinstance(value, bool):
        return DEFAULT_RETRY_COUNT
    return min(max(value, MIN_RETRY_COUNT), MAX_RETRY_COUNT)


def sanitize_provider_call_deadline_seconds(value: object) -> float:
    """Return a safe per-call deadline (seconds) for provider retries."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS
    return float(
        min(
            max(float(value), MIN_PROVIDER_CALL_DEADLINE_SECONDS),
            MAX_PROVIDER_CALL_DEADLINE_SECONDS,
        )
    )


def configured_retry_count(repositories: Any | None = None) -> int:
    """Read the current global retry count, falling back to the default."""

    if repositories is None:
        return DEFAULT_RETRY_COUNT
    getter = getattr(repositories, "get_effective_setting", None)
    if not callable(getter):
        return DEFAULT_RETRY_COUNT
    try:
        value = getter(RETRY_COUNT_SETTING)
    except Exception:  # noqa: BLE001 - fail closed if settings storage is unavailable
        return DEFAULT_RETRY_COUNT
    return sanitize_retry_count(value)


def configured_provider_call_deadline_seconds(
    repositories: Any | None = None,
) -> float:
    """Read the current per-call deadline, falling back to the default."""

    if repositories is None:
        return DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS
    getter = getattr(repositories, "get_effective_setting", None)
    if not callable(getter):
        return DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS
    try:
        value = getter(PROVIDER_CALL_DEADLINE_SETTING)
    except Exception:  # noqa: BLE001 - fail closed if settings storage is unavailable
        return DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS
    return sanitize_provider_call_deadline_seconds(value)


def configured_max_attempts(repositories: Any | None = None) -> int:
    """Return initial attempt plus the configured retry count."""

    return configured_retry_count(repositories) + 1


@contextmanager
def retry_execution_context(
    execution_class: RetryExecutionClass,
) -> Iterator[None]:
    """Temporarily apply retry behavior to provider calls in this async context."""

    token = _RETRY_EXECUTION_CLASS.set(execution_class)
    try:
        yield
    finally:
        _RETRY_EXECUTION_CLASS.reset(token)


def current_retry_execution_class() -> RetryExecutionClass:
    return _RETRY_EXECUTION_CLASS.get()


def resolved_retry_budget(
    repositories: Any | None = None,
) -> ResolvedRetryBudget:
    """Resolve provider and turn retry limits for the current execution context."""

    provider_max_attempts = configured_max_attempts(repositories)
    provider_call_deadline_seconds = configured_provider_call_deadline_seconds(
        repositories
    )
    execution_class = current_retry_execution_class()
    if execution_class is RetryExecutionClass.RESPONSIVE_FOREGROUND:
        return ResolvedRetryBudget(
            provider_max_attempts=min(
                provider_max_attempts,
                RESPONSIVE_FOREGROUND_MAX_PROVIDER_ATTEMPTS,
            ),
            provider_call_deadline_seconds=min(
                provider_call_deadline_seconds,
                RESPONSIVE_FOREGROUND_CALL_DEADLINE_SECONDS,
            ),
            automatic_turn_retry_allowed=False,
            verification_max_attempts=min(
                provider_max_attempts,
                RESPONSIVE_FOREGROUND_VERIFICATION_MAX_ATTEMPTS,
            ),
        )
    return ResolvedRetryBudget(
        provider_max_attempts=provider_max_attempts,
        provider_call_deadline_seconds=provider_call_deadline_seconds,
        automatic_turn_retry_allowed=True,
        verification_max_attempts=provider_max_attempts,
    )


RetryCountResolver = Callable[[], int]
ProviderCallDeadlineResolver = Callable[[], float]
