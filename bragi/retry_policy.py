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

from collections.abc import Callable
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


RetryCountResolver = Callable[[], int]
ProviderCallDeadlineResolver = Callable[[], float]
