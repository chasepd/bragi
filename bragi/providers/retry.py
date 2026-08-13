"""Provider retry helpers for transient upstream failures."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import cast

from bragi.app_logging import log_error_event, log_event
from bragi.providers.contracts import (
    ProviderRetryProgress,
    ProviderRetryProgressCallback,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.retry_policy import (
    DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS,
    PROVIDER_MAX_ATTEMPTS,
)

DEFAULT_PROVIDER_ATTEMPTS = PROVIDER_MAX_ATTEMPTS
DEFAULT_BACKOFF_SECONDS = 0.4
MAX_BACKOFF_SECONDS = 30.0
RATE_LIMIT_BACKOFF_MULTIPLIER = 5.0
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
    for attempt in range(1, attempts + 1):
        remaining = deadline_seconds - _elapsed_seconds(call_started_at)
        if remaining <= 0:
            raise _deadline_exceeded_error(
                provider=provider,
                task=task,
                attempt=attempt,
                max_attempts=attempts,
                deadline_seconds=deadline_seconds,
                attempt_diagnostics=attempt_diagnostics,
            )
        started_at = perf_counter()
        attempt_timeout = asyncio.timeout(remaining)
        try:
            async with attempt_timeout:
                result = await operation()
        except Exception as caught:
            deadline_expired = isinstance(caught, TimeoutError) and bool(
                getattr(attempt_timeout, "expired", lambda: True)()
            )
            exc = _normalize_timeout_error(caught)
            duration_ms = _elapsed_ms(started_at)
            category = _error_category(exc)
            attempt_diagnostics.append(
                _attempt_diagnostics(
                    attempt=attempt,
                    exc=exc,
                    duration_ms=duration_ms,
                )
            )
            if (
                deadline_expired
                or _elapsed_seconds(call_started_at) >= deadline_seconds
            ):
                raise _deadline_exceeded_error(
                    provider=provider,
                    task=task,
                    attempt=attempt,
                    max_attempts=attempts,
                    deadline_seconds=deadline_seconds,
                    attempt_diagnostics=attempt_diagnostics,
                ) from exc
            if attempt >= attempts or not _is_transient(exc):
                log_error_event(
                    "provider.retry_exhausted",
                    provider=provider,
                    task=task,
                    attempt=attempt,
                    max_attempts=attempts,
                    duration_ms=duration_ms,
                    error_category=category,
                    error=str(exc),
                )
                if isinstance(exc, ProviderError):
                    raise ProviderError(
                        category=exc.category,
                        message=exc.message,
                        status_code=exc.status_code,
                        retry_attempt_count=attempt,
                        max_retry_attempts=attempts,
                        retry_attempts=tuple(attempt_diagnostics),
                        diagnostics=dict(exc.diagnostics),
                    ) from exc
                raise
            delay = _retry_delay(
                attempt=attempt,
                base_delay=base_delay,
                error_category=category,
            )
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
            if delay >= remaining:
                raise _deadline_exceeded_error(
                    provider=provider,
                    task=task,
                    attempt=attempt,
                    max_attempts=attempts,
                    deadline_seconds=deadline_seconds,
                    attempt_diagnostics=attempt_diagnostics,
                ) from exc
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
                ),
            )
            await asyncio.sleep(delay)
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

    raise AssertionError("provider retry loop exited unexpectedly")


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


async def stream_with_provider_deadline[T](
    stream: AsyncIterator[T],
    *,
    provider: str,
    task: str,
    call_deadline_seconds: float,
) -> AsyncIterator[T]:
    """Yield a provider stream within one hard, transient timeout attempt."""

    deadline_seconds = max(0.0, float(call_deadline_seconds))
    started_at = perf_counter()
    timeout_context = asyncio.timeout(deadline_seconds)
    try:
        async with timeout_context:
            async for item in stream:
                yield item
    except TimeoutError as caught:
        if bool(getattr(timeout_context, "expired", lambda: True)()):
            raise _deadline_exceeded_error(
                provider=provider,
                task=task,
                attempt=1,
                max_attempts=1,
                deadline_seconds=deadline_seconds,
                attempt_diagnostics=[
                    {
                        "attempt": 1,
                        "error_category": ProviderErrorCategory.NETWORK_ERROR.value,
                        "duration_ms": _elapsed_ms(started_at),
                    }
                ],
            ) from caught
        normalized = _normalize_timeout_error(caught)
        if not isinstance(normalized, ProviderError):
            raise
        raise ProviderError(
            category=normalized.category,
            message=normalized.message,
            retry_attempt_count=1,
            max_retry_attempts=1,
            retry_attempts=(
                _attempt_diagnostics(
                    attempt=1,
                    exc=normalized,
                    duration_ms=_elapsed_ms(started_at),
                ),
            ),
            diagnostics=dict(normalized.diagnostics),
        ) from caught


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


def _normalize_timeout_error(exc: Exception) -> Exception:
    if not isinstance(exc, TimeoutError) or isinstance(exc, ProviderError):
        return exc
    return ProviderError(
        category=ProviderErrorCategory.NETWORK_ERROR,
        message="Provider request timed out",
        diagnostics={"timeout": True},
    )


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
