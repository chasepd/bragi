from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from bragi.providers.errors import (
    PROVIDER_ERROR_MESSAGE_DIAGNOSTIC,
    ProviderError,
    ProviderErrorCategory,
)
from bragi.providers.retry import (
    DEFAULT_PROVIDER_ATTEMPTS,
    MAX_BACKOFF_SECONDS,
    NO_ENDPOINTS_ROUTING_ERROR_MESSAGE,
    call_with_provider_retries,
    is_no_endpoints_routing_error,
    is_transient_provider_error,
)


def test_call_with_provider_retries_succeeds_after_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0
        sleeps: list[float] = []
        events: list[tuple[str, dict[str, object]]] = []

        async def operation() -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderError(
                    ProviderErrorCategory.RATE_LIMITED,
                    "slow down",
                    status_code=429,
                )
            return {"body": "ok"}

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", fake_sleep)
        monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.01)
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda event_name, **fields: events.append((event_name, fields)),
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda event_name, **fields: events.append((event_name, fields)),
        )

        result = await call_with_provider_retries(
            operation,
            provider="fake",
            task="chat",
            max_attempts=3,
        )

        assert attempts == 2
        assert sleeps == [0.01]
        assert result["body"] == "ok"
        retry = cast(dict[str, Any], result["_bragi_retry"])
        retry_attempts = cast(list[dict[str, object]], retry["attempts"])
        assert retry["attempt_count"] == 2
        assert retry["max_attempts"] == 3
        assert retry_attempts[0]["error_category"] == "rate_limited"
        assert retry_attempts[1]["error_category"] is None
        assert [event[0] for event in events] == [
            "provider.retry_scheduled",
            "provider.retry_succeeded",
        ]

    asyncio.run(run())


def test_call_with_provider_retries_reports_progress_before_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0
        progress: list[Any] = []
        sleeps_after_progress: list[int] = []

        async def operation() -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderError(
                    ProviderErrorCategory.RATE_LIMITED,
                    "slow down",
                    status_code=429,
                )
            return {"body": "ok"}

        async def fake_sleep(_delay: float) -> None:
            sleeps_after_progress.append(len(progress))

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", fake_sleep)
        monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.25)
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda *_args, **_kwargs: None,
        )

        await call_with_provider_retries(
            operation,
            provider="fake",
            task="chat",
            max_attempts=3,
            retry_progress_callback=progress.append,
        )

        assert sleeps_after_progress == [1]
        [reported] = progress
        assert reported.provider == "fake"
        assert reported.task == "chat"
        assert reported.failed_attempt == 1
        assert reported.next_attempt == 2
        assert reported.max_attempts == 3
        assert reported.retry_delay_ms == 250
        assert reported.error_category == "rate_limited"
        assert reported.http_status == 429
        assert reported.unlimited is False

    asyncio.run(run())


def test_call_with_provider_retries_ignores_progress_callback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0
        events: list[tuple[str, dict[str, object]]] = []

        async def operation() -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderError(
                    ProviderErrorCategory.NETWORK_ERROR,
                    "socket closed",
                )
            return {"body": "ok"}

        def callback(_progress: object) -> None:
            raise RuntimeError("callback exploded")

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", _no_sleep)
        monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.01)
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda event_name, **fields: events.append((event_name, fields)),
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda *_args, **_kwargs: None,
        )

        result = await call_with_provider_retries(
            operation,
            provider="fake",
            task="image_generation",
            max_attempts=2,
            retry_progress_callback=callback,
        )

        assert result["body"] == "ok"
        assert attempts == 2
        assert any(
            event_name == "provider.retry_progress_callback_failed"
            for event_name, _fields in events
        )

    asyncio.run(run())


def test_call_with_provider_retries_uses_longer_delay_for_rate_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0
        sleeps: list[float] = []

        async def operation() -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderError(
                    ProviderErrorCategory.RATE_LIMITED,
                    "slow down",
                    status_code=429,
                )
            return {"body": "ok"}

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", fake_sleep)
        monkeypatch.setattr(
            "bragi.providers.retry.random.uniform",
            lambda _start, _stop: 0.0,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda *_args, **_kwargs: None,
        )

        await call_with_provider_retries(
            operation,
            provider="fake",
            task="structured_output",
            max_attempts=3,
            base_delay=0.4,
        )

        assert sleeps == [2.0]

    asyncio.run(run())


def test_quality_first_provider_retry_defaults() -> None:
    assert DEFAULT_PROVIDER_ATTEMPTS == 7
    assert MAX_BACKOFF_SECONDS == 30.0


def test_default_provider_retry_budget_allows_success_on_seventh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempt_count = 0

        async def operation() -> dict[str, object]:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 7:
                raise ProviderError(
                    ProviderErrorCategory.NETWORK_ERROR,
                    "temporary outage",
                )
            return {"body": "recovered"}

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", _no_sleep)
        monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.0)
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda *_args, **_kwargs: None,
        )

        result = await call_with_provider_retries(
            operation,
            provider="fake",
            task="chat",
        )

        assert attempt_count == 7
        retry = cast(dict[str, Any], result["_bragi_retry"])
        assert retry["attempt_count"] == 7
        assert retry["max_attempts"] == 7

    asyncio.run(run())


def test_call_with_provider_retries_retries_408_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0
        sleeps: list[float] = []

        async def operation() -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderError(
                    ProviderErrorCategory.NETWORK_ERROR,
                    "request timed out",
                    status_code=408,
                )
            return {"body": "ok"}

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", fake_sleep)
        monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.01)
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda *_args, **_kwargs: None,
        )

        result = await call_with_provider_retries(
            operation,
            provider="fake",
            task="chat",
            max_attempts=3,
        )

        assert attempts == 2
        assert sleeps == [0.01]
        assert result["body"] == "ok"

    asyncio.run(run())


def test_call_with_provider_retries_exhaustion_preserves_retry_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        sleeps: list[float] = []

        async def operation() -> object:
            raise ProviderError(
                ProviderErrorCategory.PROVIDER_ERROR,
                "upstream unavailable",
                status_code=503,
                diagnostics={"finish_reason": "length", "reasoning_tokens": 20},
            )

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", fake_sleep)
        monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.01)
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda *_args, **_kwargs: None,
        )

        with pytest.raises(ProviderError) as exc_info:
            await call_with_provider_retries(
                operation,
                provider="fake",
                task="chat",
                max_attempts=2,
            )

        assert sleeps == [0.01]
        assert exc_info.value.retry_attempt_count == 2
        assert exc_info.value.max_retry_attempts == 2
        assert exc_info.value.diagnostics == {
            "finish_reason": "length",
            "reasoning_tokens": 20,
        }
        assert [
            attempt["http_status"] for attempt in exc_info.value.retry_attempts
        ] == [503, 503]

    asyncio.run(run())


def test_call_with_provider_retries_does_not_retry_non_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0

        async def operation() -> object:
            nonlocal attempts
            attempts += 1
            raise ProviderError(
                ProviderErrorCategory.AUTHENTICATION_FAILED,
                "bad key",
                status_code=401,
            )

        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda *_args, **_kwargs: None,
        )

        with pytest.raises(ProviderError) as exc_info:
            await call_with_provider_retries(
                operation,
                provider="fake",
                task="chat",
                max_attempts=3,
            )

        assert attempts == 1
        assert exc_info.value.retry_attempt_count == 1
        assert exc_info.value.retry_attempts[0]["http_status"] == 401

    asyncio.run(run())


def test_is_transient_provider_error_classifies_retryable_categories() -> None:
    assert is_transient_provider_error(
        ProviderError(ProviderErrorCategory.NETWORK_ERROR, "socket closed")
    )
    assert is_transient_provider_error(
        ProviderError(ProviderErrorCategory.PROVIDER_ERROR, "bad gateway")
    )
    assert is_transient_provider_error(
        ProviderError(ProviderErrorCategory.PROVIDER_ERROR, "bad gateway", 502)
    )
    assert not is_transient_provider_error(
        ProviderError(ProviderErrorCategory.PROVIDER_ERROR, "bad request", 400)
    )


def test_call_with_provider_retries_aborts_on_global_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0
        events: list[tuple[str, dict[str, object]]] = []
        current_time = {"value": 0.0}

        async def operation() -> object:
            nonlocal attempts
            attempts += 1
            current_time["value"] += 1.0
            raise ProviderError(
                ProviderErrorCategory.NETWORK_ERROR,
                "slow upstream",
            )

        async def scheduled_sleep(delay: float) -> None:
            current_time["value"] += delay

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", scheduled_sleep)
        monkeypatch.setattr(
            "bragi.providers.retry.perf_counter",
            lambda: current_time["value"],
        )
        monkeypatch.setattr(
            "bragi.providers.retry._retry_delay",
            lambda **_kwargs: 0.0,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda event_name, **fields: events.append((event_name, fields)),
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda *_args, **_kwargs: None,
        )

        with pytest.raises(ProviderError) as exc_info:
            await call_with_provider_retries(
                operation,
                provider="fake",
                task="structured_output",
                max_attempts=7,
                call_deadline_seconds=2.5,
            )

        assert exc_info.value.category is ProviderErrorCategory.NETWORK_ERROR
        assert "global deadline" in str(exc_info.value)
        assert exc_info.value.diagnostics.get("deadline_exceeded") is True
        assert exc_info.value.diagnostics.get("deadline_seconds") == 2.5
        assert attempts < 7
        assert any(
            event_name == "provider.retry_deadline_exceeded"
            for event_name, _fields in events
        )

    asyncio.run(run())


def test_call_with_provider_retries_clamps_retry_delay_to_deadline_remaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0
        current_time = {"value": 0.0}
        sleeps: list[float] = []

        async def operation() -> object:
            nonlocal attempts
            attempts += 1
            current_time["value"] += 0.1
            if attempts < 2:
                raise ProviderError(
                    ProviderErrorCategory.NETWORK_ERROR,
                    "flaky",
                )
            return {"body": "ok"}

        async def scheduled_sleep(delay: float) -> None:
            sleeps.append(delay)
            current_time["value"] += delay

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", scheduled_sleep)
        monkeypatch.setattr(
            "bragi.providers.retry.perf_counter",
            lambda: current_time["value"],
        )
        monkeypatch.setattr(
            "bragi.providers.retry._retry_delay",
            lambda **_kwargs: 5.0,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda *_args, **_kwargs: None,
        )

        result = await call_with_provider_retries(
            operation,
            provider="fake",
            task="chat",
            max_attempts=5,
            call_deadline_seconds=2.0,
        )

        body = cast(dict[str, object], result)["body"]
        assert body == "ok"
        assert sleeps[0] == 1.9

    asyncio.run(run())


def _no_endpoints_error() -> ProviderError:
    return ProviderError(
        ProviderErrorCategory.MODEL_NOT_FOUND,
        "model_not_found (404)",
        status_code=404,
        diagnostics={
            PROVIDER_ERROR_MESSAGE_DIAGNOSTIC: (
                f"{NO_ENDPOINTS_ROUTING_ERROR_MESSAGE}. To learn more about "
                "provider routing, visit: "
                "https://openrouter.ai/docs/guides/routing/provider-selection"
            )
        },
    )


def test_is_no_endpoints_routing_error_matches_routing_404_only() -> None:
    assert is_no_endpoints_routing_error(_no_endpoints_error())
    assert not is_no_endpoints_routing_error(
        ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model_not_found (404)",
            status_code=404,
        )
    )
    assert not is_no_endpoints_routing_error(
        ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model_not_found (404)",
            status_code=404,
            diagnostics={PROVIDER_ERROR_MESSAGE_DIAGNOSTIC: "Not Found"},
        )
    )
    assert not is_no_endpoints_routing_error(
        ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "bad request",
            status_code=400,
            diagnostics={
                PROVIDER_ERROR_MESSAGE_DIAGNOSTIC: NO_ENDPOINTS_ROUTING_ERROR_MESSAGE
            },
        )
    )
    assert not is_no_endpoints_routing_error(RuntimeError("boom"))


def test_no_endpoints_404_keeps_retrying_beyond_attempt_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0
        sleeps: list[float] = []
        events: list[tuple[str, dict[str, object]]] = []

        async def operation() -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            if attempts < 10:
                raise _no_endpoints_error()
            return {"body": "ok"}

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", fake_sleep)
        monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.01)
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda event_name, **fields: events.append((event_name, fields)),
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda event_name, **fields: events.append((event_name, fields)),
        )

        result = await call_with_provider_retries(
            operation,
            provider="openrouter",
            task="tool_calling",
            max_attempts=3,
        )

        assert attempts == 10
        assert sleeps == [0.01] * 9
        assert result["body"] == "ok"
        retry = cast(dict[str, Any], result["_bragi_retry"])
        assert retry["attempt_count"] == 10
        assert retry["max_attempts"] == 3
        scheduled = [
            fields
            for event_name, fields in events
            if event_name == "provider.retry_scheduled"
        ]
        assert len(scheduled) == 9
        assert all(fields.get("no_endpoints_routing") is True for fields in scheduled)

    asyncio.run(run())


def test_no_endpoints_404_bypasses_global_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0
        current_time = {"value": 0.0}

        async def operation() -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            current_time["value"] += 0.5
            if attempts < 10:
                raise _no_endpoints_error()
            return {"body": "ok"}

        async def scheduled_sleep(delay: float) -> None:
            current_time["value"] += delay

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", scheduled_sleep)
        monkeypatch.setattr(
            "bragi.providers.retry.perf_counter",
            lambda: current_time["value"],
        )
        monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.0)
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda *_args, **_kwargs: None,
        )

        result = await call_with_provider_retries(
            operation,
            provider="openrouter",
            task="structured_output",
            max_attempts=7,
            call_deadline_seconds=2.5,
        )

        assert attempts == 10
        assert current_time["value"] > 2.5
        assert result["body"] == "ok"

    asyncio.run(run())


def test_no_endpoints_404_ends_on_different_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0

        async def operation() -> object:
            nonlocal attempts
            attempts += 1
            if attempts < 4:
                raise _no_endpoints_error()
            raise ProviderError(
                ProviderErrorCategory.AUTHENTICATION_FAILED,
                "bad key",
                status_code=401,
            )

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", _no_sleep)
        monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.0)
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda *_args, **_kwargs: None,
        )

        with pytest.raises(ProviderError) as exc_info:
            await call_with_provider_retries(
                operation,
                provider="openrouter",
                task="tool_calling",
                max_attempts=2,
            )

        assert attempts == 4
        assert exc_info.value.category is ProviderErrorCategory.AUTHENTICATION_FAILED
        assert exc_info.value.retry_attempt_count == 4
        assert exc_info.value.diagnostics.get("no_endpoints_retry_terminated") is True
        assert [
            attempt["http_status"]
            for attempt in exc_info.value.retry_attempts
            if attempt.get("http_status") is not None
        ] == [404, 404, 404, 401]

    asyncio.run(run())


def test_no_endpoints_404_reports_unlimited_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        attempts = 0
        progress: list[Any] = []

        async def operation() -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise _no_endpoints_error()
            return {"body": "ok"}

        monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", _no_sleep)
        monkeypatch.setattr("bragi.providers.retry._retry_delay", lambda **_: 0.01)
        monkeypatch.setattr(
            "bragi.providers.retry.log_error_event",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "bragi.providers.retry.log_event",
            lambda *_args, **_kwargs: None,
        )

        await call_with_provider_retries(
            operation,
            provider="openrouter",
            task="tool_calling",
            max_attempts=3,
            retry_progress_callback=progress.append,
        )

        [reported] = progress
        assert reported.failed_attempt == 1
        assert reported.next_attempt == 2
        assert reported.unlimited is True

    asyncio.run(run())


async def _no_sleep(_delay: float) -> None:
    return None
