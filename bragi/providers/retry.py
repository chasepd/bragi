"""Provider retry helpers for transient upstream failures."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import NoReturn, cast

from bragi.app_logging import log_error_event, log_event
from bragi.providers.contracts import (
    ProviderRetryProgress,
    ProviderRetryProgressCallback,
)
from bragi.providers.errors import (
    PROVIDER_ERROR_MESSAGE_DIAGNOSTIC,
    ProviderError,
    ProviderErrorCategory,
)
from bragi.retry_policy import (
    DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS,
    PROVIDER_MAX_ATTEMPTS,
)

DEFAULT_PROVIDER_ATTEMPTS = PROVIDER_MAX_ATTEMPTS
DEFAULT_BACKOFF_SECONDS = 0.4
MAX_BACKOFF_SECONDS = 30.0
RATE_LIMIT_BACKOFF_MULTIPLIER = 5.0
NO_ENDPOINTS_ROUTING_ERROR_MESSAGE = (
    "No endpoints found that can handle the requested parameters"
)
_TRANSIENT_CATEGORIES = frozenset(
    {
        ProviderErrorCategory.NETWORK_ERROR,
        ProviderErrorCategory.RATE_LIMITED,
    }
)


@dataclass(frozen=True)
class RetryAttempt:
    attempt: int
    error_category: str | None
    duration_ms: int


async def call_with_provider_retries[T](
    operation: Callable[[], Awaitable[T]],
    *,
    provider: str,
    task: str,
    max_attempts: int = DEFAULT_PROVIDER_ATTEMPTS,
    base_delay: float = DEFAULT_BACKOFF_SECONDS,
    retry_progress_callback: ProviderRetryProgressCallback | None = None,
    call_deadline_seconds: float = DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS,
) -> T:
    attempts = max(1, max_attempts)
    deadline_seconds = max(0.0, float(call_deadline_seconds))
    attempt_diagnostics: list[dict[str, object]] = []
    call_started_at = perf_counter()
    # OpenRouter "no endpoints found" 404s run no inference and bill nothing, so
    # once one is seen the loop keeps retrying without the attempt budget or the
    # per-call deadline until it succeeds or the failure type changes.
    unlimited_no_endpoints = False
    attempt = 0
    while True:
        attempt += 1
        no_endpoints_attempt = False
        if (
            not unlimited_no_endpoints
            and _elapsed_seconds(call_started_at) > deadline_seconds
        ):
            raise _deadline_exceeded_error(
                provider=provider,
                task=task,
                attempt=attempt,
                max_attempts=attempts,
                deadline_seconds=deadline_seconds,
                attempt_diagnostics=attempt_diagnostics,
            )
        started_at = perf_counter()
        try:
            result = await operation()
        except Exception as exc:
            duration_ms = _elapsed_ms(started_at)
            category = _error_category(exc)
            attempt_diagnostics.append(
                _attempt_diagnostics(
                    attempt=attempt,
                    exc=exc,
                    duration_ms=duration_ms,
                )
            )
            no_endpoints_attempt = is_no_endpoints_routing_error(exc)
            if unlimited_no_endpoints and not no_endpoints_attempt:
                _raise_retry_exhausted_error(
                    provider=provider,
                    task=task,
                    attempt=attempt,
                    max_attempts=attempts,
                    duration_ms=duration_ms,
                    category=category,
                    exc=exc,
                    attempt_diagnostics=attempt_diagnostics,
                    no_endpoints_terminated=True,
                )
            if (
                not no_endpoints_attempt
                and _elapsed_seconds(call_started_at) > deadline_seconds
            ):
                raise _deadline_exceeded_error(
                    provider=provider,
                    task=task,
                    attempt=attempt,
                    max_attempts=attempts,
                    deadline_seconds=deadline_seconds,
                    attempt_diagnostics=attempt_diagnostics,
                ) from exc
            if not no_endpoints_attempt and (
                attempt >= attempts or not _is_transient(exc)
            ):
                _raise_retry_exhausted_error(
                    provider=provider,
                    task=task,
                    attempt=attempt,
                    max_attempts=attempts,
                    duration_ms=duration_ms,
                    category=category,
                    exc=exc,
                    attempt_diagnostics=attempt_diagnostics,
                )
            delay = _retry_delay(
                attempt=attempt,
                base_delay=base_delay,
                error_category=category,
            )
            if not no_endpoints_attempt:
                remaining = deadline_seconds - _elapsed_seconds(call_started_at)
                if remaining <= 0:
                    raise _deadline_exceeded_error(
                        provider=provider,
                        task=task,
                        attempt=attempt,
                        max_attempts=attempts,
                        deadline_seconds=deadline_seconds,
                        attempt_diagnostics=attempt_diagnostics,
                    ) from exc
                if delay > remaining:
                    delay = remaining
            log_error_event(
                "provider.retry_scheduled",
                provider=provider,
                task=task,
                attempt=attempt,
                max_attempts=attempts,
                duration_ms=duration_ms,
                retry_delay_ms=int(delay * 1000),
                error_category=category,
                error=str(exc),
                no_endpoints_routing=no_endpoints_attempt,
            )
            _publish_retry_progress(
                retry_progress_callback,
                ProviderRetryProgress(
                    provider=provider,
                    task=task,
                    failed_attempt=attempt,
                    next_attempt=attempt + 1,
                    max_attempts=attempts,
                    retry_delay_ms=int(delay * 1000),
                    error_category=category,
                    http_status=(
                        exc.status_code if isinstance(exc, ProviderError) else None
                    ),
                    unlimited=no_endpoints_attempt,
                ),
            )
            await asyncio.sleep(delay)
            if no_endpoints_attempt:
                unlimited_no_endpoints = True
            continue

        duration_ms = _elapsed_ms(started_at)
        if attempt_diagnostics:
            attempt_diagnostics.append(
                {
                    "attempt": attempt,
                    "error_category": None,
                    "duration_ms": duration_ms,
                }
            )
        if attempt > 1:
            log_event(
                "provider.retry_succeeded",
                provider=provider,
                task=task,
                attempt=attempt,
                max_attempts=attempts,
                duration_ms=duration_ms,
            )
        return _with_retry_metadata(
            result,
            attempt_count=attempt,
            max_attempts=attempts,
            retry_attempts=tuple(attempt_diagnostics),
        )


def is_no_endpoints_routing_error(exc: Exception) -> bool:
    """Match OpenRouter's transient routing 404 for no available endpoints.

    OpenRouter returns this 404 (with the guidance message) when no endpoint
    can currently handle the request. No inference ran and no billing applies,
    so callers may keep retrying without regard for the retry budget.

    The match relies on OpenRouter's message wording; if it changes, behavior
    degrades safely to the ordinary terminal handling for 404s.
    """

    if not isinstance(exc, ProviderError):
        return False
    if exc.status_code != 404:
        return False
    message = exc.diagnostics.get(PROVIDER_ERROR_MESSAGE_DIAGNOSTIC)
    return (
        isinstance(message, str)
        and NO_ENDPOINTS_ROUTING_ERROR_MESSAGE in message
    )


def is_transient_provider_error(exc: Exception) -> bool:
    return _is_transient(exc)


def retry_metadata_from_provider_error(exc: ProviderError) -> dict[str, object]:
    retry: dict[str, object] = {}
    if exc.retry_attempt_count is not None:
        retry["attempt_count"] = exc.retry_attempt_count
    if exc.max_retry_attempts is not None:
        retry["max_attempts"] = exc.max_retry_attempts
    if exc.retry_attempts:
        retry["attempts"] = [dict(attempt) for attempt in exc.retry_attempts]
    return {"_bragi_retry": retry} if retry else {}


def exhausted_retry_attempt_count(exc: Exception) -> int | None:
    if not isinstance(exc, ProviderError):
        return None
    if exc.retry_attempt_count is None or exc.max_retry_attempts is None:
        return None
    if exc.retry_attempt_count < exc.max_retry_attempts:
        return None
    if exc.retry_attempt_count <= 1:
        return None
    return exc.retry_attempt_count


def _is_transient(exc: Exception) -> bool:
    if not isinstance(exc, ProviderError):
        return False
    if exc.category in _TRANSIENT_CATEGORIES:
        return True
    return exc.category is ProviderErrorCategory.PROVIDER_ERROR and (
        exc.status_code is None or exc.status_code >= 500
    )


def _error_category(exc: Exception) -> str:
    if isinstance(exc, ProviderError):
        return exc.category.value
    return exc.__class__.__name__


def _retry_delay(
    *,
    attempt: int,
    base_delay: float,
    error_category: str | None = None,
) -> float:
    retry_base_delay = (
        base_delay * RATE_LIMIT_BACKOFF_MULTIPLIER
        if error_category == ProviderErrorCategory.RATE_LIMITED.value
        else base_delay
    )
    exponential = float(
        min(MAX_BACKOFF_SECONDS, retry_base_delay * (2 ** (attempt - 1)))
    )
    jitter = float(random.uniform(0, exponential * 0.25))
    return exponential + jitter


def _with_retry_metadata[T](
    result: T,
    *,
    attempt_count: int,
    max_attempts: int,
    retry_attempts: tuple[dict[str, object], ...],
) -> T:
    if not isinstance(result, dict):
        return result
    metadata = dict(result)
    metadata["_bragi_retry"] = {
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
    }
    if retry_attempts:
        metadata["_bragi_retry"]["attempts"] = list(retry_attempts)
    return cast(T, metadata)


def _attempt_diagnostics(
    *,
    attempt: int,
    exc: Exception,
    duration_ms: int,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "attempt": attempt,
        "error_category": _error_category(exc),
        "duration_ms": duration_ms,
    }
    if isinstance(exc, ProviderError) and exc.status_code is not None:
        diagnostics["http_status"] = exc.status_code
    return diagnostics


def _publish_retry_progress(
    callback: ProviderRetryProgressCallback | None,
    progress: ProviderRetryProgress,
) -> None:
    if callback is None:
        return
    try:
        callback(progress)
    except Exception as exc:
        log_error_event(
            "provider.retry_progress_callback_failed",
            provider=progress.provider,
            task=progress.task,
            failed_attempt=progress.failed_attempt,
            next_attempt=progress.next_attempt,
            max_attempts=progress.max_attempts,
            error_category=type(exc).__name__,
            error=str(exc),
        )


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def _elapsed_seconds(started_at: float) -> float:
    return perf_counter() - started_at


def _deadline_exceeded_error(
    *,
    provider: str,
    task: str,
    attempt: int,
    max_attempts: int,
    deadline_seconds: float,
    attempt_diagnostics: list[dict[str, object]],
) -> ProviderError:
    log_error_event(
        "provider.retry_deadline_exceeded",
        provider=provider,
        task=task,
        attempt=attempt,
        max_attempts=max_attempts,
        deadline_seconds=deadline_seconds,
    )
    return ProviderError(
        category=ProviderErrorCategory.NETWORK_ERROR,
        message=(
            f"Provider call exceeded the global deadline of {deadline_seconds:g}s "
            f"after {attempt} attempts"
        ),
        retry_attempt_count=attempt,
        max_retry_attempts=max_attempts,
        retry_attempts=tuple(attempt_diagnostics),
        diagnostics={
            "deadline_seconds": deadline_seconds,
            "deadline_exceeded": True,
        },
    )


def _raise_retry_exhausted_error(
    *,
    provider: str,
    task: str,
    attempt: int,
    max_attempts: int,
    duration_ms: int,
    category: str,
    exc: Exception,
    attempt_diagnostics: list[dict[str, object]],
    no_endpoints_terminated: bool = False,
) -> NoReturn:
    log_error_event(
        "provider.retry_exhausted",
        provider=provider,
        task=task,
        attempt=attempt,
        max_attempts=max_attempts,
        duration_ms=duration_ms,
        error_category=category,
        error=str(exc),
    )
    if isinstance(exc, ProviderError):
        diagnostics = dict(exc.diagnostics)
        if no_endpoints_terminated:
            diagnostics["no_endpoints_retry_terminated"] = True
        raise ProviderError(
            category=exc.category,
            message=exc.message,
            status_code=exc.status_code,
            retry_attempt_count=attempt,
            max_retry_attempts=max_attempts,
            retry_attempts=tuple(attempt_diagnostics),
            diagnostics=diagnostics,
        ) from exc
    raise exc
