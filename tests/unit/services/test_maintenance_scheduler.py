from __future__ import annotations

import pytest

from bragi.persistence.models import JobRecord
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.maintenance_scheduler import (
    provider_pressure_from_exception,
    provider_pressure_from_jobs,
    provider_pressure_from_result,
)


def test_provider_pressure_requires_exhausted_retry_metadata_when_present() -> None:
    pressure = provider_pressure_from_exception(
        ProviderError(
            category=ProviderErrorCategory.RATE_LIMITED,
            message="rate limited",
            status_code=429,
            retry_attempt_count=2,
            max_retry_attempts=3,
            retry_attempts=(
                {"attempt": 1, "error_category": "rate_limited", "duration_ms": 10},
                {"attempt": 2, "error_category": "rate_limited", "duration_ms": 10},
            ),
        )
    )

    assert pressure is None


def test_provider_pressure_detects_exhausted_rate_limit_errors() -> None:
    pressure = provider_pressure_from_exception(
        ProviderError(
            category=ProviderErrorCategory.RATE_LIMITED,
            message="rate limited",
            status_code=429,
            retry_attempt_count=3,
            max_retry_attempts=3,
            retry_attempts=(
                {"attempt": 1, "error_category": "rate_limited", "duration_ms": 10},
                {"attempt": 2, "error_category": "rate_limited", "duration_ms": 10},
                {"attempt": 3, "error_category": "rate_limited", "duration_ms": 10},
            ),
        )
    )

    assert pressure is not None
    assert pressure.reason == "provider_pressure"
    assert pressure.error_category == "rate_limited"
    assert pressure.http_status == 429
    assert pressure.retry_attempt_count == 3
    assert pressure.max_retry_attempts == 3


def test_provider_pressure_ignores_queued_retry_payload_marker() -> None:
    pressure = provider_pressure_from_jobs(
        [
            JobRecord(
                id="retry-1",
                save_id="save-1",
                type="context_update_retry",
                status="queued",
                payload={
                    "last_pressure_category": "rate_limited",
                    "last_pressure_http_status": 429,
                    "retry_attempt": 1,
                    "max_retry_attempts": 3,
                },
                result=None,
                error=None,
                started_at=None,
                completed_at=None,
            )
        ]
    )

    assert pressure is None


def test_provider_pressure_from_result_prefers_embedded_pressure_payload() -> None:
    pressure = provider_pressure_from_result(
        {
            "error_category": "content_blocked",
            "provider_pressure": {
                "error_category": ProviderErrorCategory.NETWORK_ERROR.value,
                "http_status": 503,
                "attempt_count": 3,
                "max_attempts": 3,
            },
        }
    )

    assert pressure is not None
    assert pressure.error_category == "network_error"
    assert pressure.http_status == 503
    assert pressure.retry_attempt_count == 3
    assert pressure.max_retry_attempts == 3


def test_provider_pressure_from_result_uses_latest_nested_provider_call() -> None:
    pressure = provider_pressure_from_result(
        {
            "provider_calls": [
                {
                    "provider": "first",
                    "error_category": ProviderErrorCategory.NETWORK_ERROR.value,
                    "http_status": 502,
                    "attempt_count": 2,
                    "max_attempts": 3,
                },
                {
                    "provider": "second",
                    "error_category": ProviderErrorCategory.RATE_LIMITED.value,
                    "http_status": 429,
                    "retry_attempt_count": 3,
                    "max_retry_attempts": 3,
                },
            ]
        }
    )

    assert pressure is not None
    assert pressure.error_category == "rate_limited"
    assert pressure.http_status == 429
    assert pressure.retry_attempt_count == 3
    assert pressure.max_retry_attempts == 3


def test_provider_pressure_from_result_ignores_unexhausted_nested_call() -> None:
    pressure = provider_pressure_from_result(
        {
            "provider_calls": [
                {
                    "error_category": ProviderErrorCategory.RATE_LIMITED.value,
                    "http_status": 429,
                    "attempt_count": 1,
                    "max_attempts": 3,
                }
            ]
        }
    )

    assert pressure is None


@pytest.mark.parametrize(
    ("category", "http_status", "expected"),
    [
        (ProviderErrorCategory.PROVIDER_ERROR.value, 500, True),
        (ProviderErrorCategory.PROVIDER_ERROR.value, 503, True),
        (ProviderErrorCategory.PROVIDER_ERROR.value, None, True),
        (ProviderErrorCategory.PROVIDER_ERROR.value, 499, False),
        (ProviderErrorCategory.CONTENT_BLOCKED.value, 503, False),
    ],
)
def test_provider_pressure_from_result_treats_5xx_provider_errors_as_pressure(
    category: str,
    http_status: int | None,
    expected: bool,
) -> None:
    payload: dict[str, object] = {"error_category": category}
    if http_status is not None:
        payload["http_status"] = http_status

    pressure = provider_pressure_from_result(payload)

    assert (pressure is not None) is expected
    if pressure is not None:
        assert pressure.error_category == category
        assert pressure.http_status == http_status


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("context update timed out"),
        ConnectionError("connection reset by provider"),
    ],
)
def test_provider_pressure_detects_timeout_and_connection_exceptions(
    exc: Exception,
) -> None:
    pressure = provider_pressure_from_exception(exc)

    assert pressure is not None
    assert pressure.error_category == ProviderErrorCategory.NETWORK_ERROR.value


@pytest.mark.parametrize(
    "error",
    [
        "rate limited by provider",
        "request timed out waiting for response",
        "connection reset by provider",
    ],
)
def test_provider_pressure_from_jobs_parses_failed_job_error_text(error: str) -> None:
    pressure = provider_pressure_from_jobs(
        [
            JobRecord(
                id="job-pressure",
                save_id="save-1",
                type="context_update",
                status="failed",
                payload={},
                result=None,
                error=error,
                started_at=None,
                completed_at=None,
            )
        ]
    )

    assert pressure is not None
    assert pressure.source_job_id == "job-pressure"
    assert pressure.error_category in {
        ProviderErrorCategory.RATE_LIMITED.value,
        ProviderErrorCategory.NETWORK_ERROR.value,
    }
