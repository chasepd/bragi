"""Provider-pressure decisions for post-turn maintenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bragi.persistence.models import JobRecord
from bragi.providers.errors import ProviderError, ProviderErrorCategory

PROVIDER_PRESSURE_COOLDOWN_SECONDS = 15 * 60
CONTEXT_UPDATE_RETRY_MAX_ATTEMPTS = 3
CONTEXT_UPDATE_RETRY_DRAIN_LIMIT = 3

_PRESSURE_CATEGORIES = frozenset(
    {
        ProviderErrorCategory.RATE_LIMITED.value,
        ProviderErrorCategory.NETWORK_ERROR.value,
    }
)


@dataclass(frozen=True)
class ProviderPressure:
    reason: str
    error_category: str
    http_status: int | None = None
    retry_attempt_count: int | None = None
    max_retry_attempts: int | None = None
    source_job_id: str | None = None

    def to_result(self) -> dict[str, object]:
        result: dict[str, object] = {
            "reason": self.reason,
            "error_category": self.error_category,
        }
        if self.http_status is not None:
            result["http_status"] = self.http_status
        if self.retry_attempt_count is not None:
            result["retry_attempt_count"] = self.retry_attempt_count
        if self.max_retry_attempts is not None:
            result["max_retry_attempts"] = self.max_retry_attempts
        if self.source_job_id is not None:
            result["source_job_id"] = self.source_job_id
        return result


def provider_pressure_from_exception(exc: Exception) -> ProviderPressure | None:
    if isinstance(exc, ProviderError):
        category = exc.category.value
        http_status = exc.status_code
        if not _is_pressure_category(category, http_status):
            return None
        if not _retry_exhausted_or_unknown(
            retry_attempt_count=exc.retry_attempt_count,
            max_retry_attempts=exc.max_retry_attempts,
        ):
            return None
        return ProviderPressure(
            reason="provider_pressure",
            error_category=category,
            http_status=http_status,
            retry_attempt_count=exc.retry_attempt_count,
            max_retry_attempts=exc.max_retry_attempts,
        )
    if isinstance(exc, TimeoutError | ConnectionError):
        return ProviderPressure(
            reason="provider_pressure",
            error_category=ProviderErrorCategory.NETWORK_ERROR.value,
        )
    return None


def provider_pressure_from_result(
    result: Mapping[str, object] | None,
) -> ProviderPressure | None:
    if result is None:
        return None
    existing = result.get("provider_pressure")
    if isinstance(existing, Mapping):
        pressure = _pressure_from_mapping(existing)
        if pressure is not None:
            return pressure

    pressure = _pressure_from_mapping(result)
    if pressure is not None:
        return pressure

    calls = result.get("provider_calls")
    if isinstance(calls, list | tuple):
        for call in reversed(calls):
            if isinstance(call, Mapping):
                pressure = _pressure_from_mapping(call)
                if pressure is not None:
                    return pressure
    return None


def provider_pressure_from_jobs(
    jobs: tuple[JobRecord, ...] | list[JobRecord],
) -> ProviderPressure | None:
    for job in jobs:
        pressure = provider_pressure_from_result(job.result)
        if pressure is not None:
            return ProviderPressure(
                reason=pressure.reason,
                error_category=pressure.error_category,
                http_status=pressure.http_status,
                retry_attempt_count=pressure.retry_attempt_count,
                max_retry_attempts=pressure.max_retry_attempts,
                source_job_id=job.id,
            )
        if job.status != "failed":
            continue
        pressure = _pressure_from_error_text(job.error)
        if pressure is not None:
            return ProviderPressure(
                reason=pressure.reason,
                error_category=pressure.error_category,
                http_status=pressure.http_status,
                retry_attempt_count=pressure.retry_attempt_count,
                max_retry_attempts=pressure.max_retry_attempts,
                source_job_id=job.id,
            )
    return None


def _pressure_from_mapping(
    value: Mapping[str, object],
) -> ProviderPressure | None:
    category = value.get("error_category")
    if not isinstance(category, str):
        return None
    http_status = _optional_int(value.get("http_status"))
    if not _is_pressure_category(category, http_status):
        return None
    retry_attempt_count = _optional_int(
        value.get("retry_attempt_count", value.get("attempt_count"))
    )
    max_retry_attempts = _optional_int(
        value.get("max_retry_attempts", value.get("max_attempts"))
    )
    if not _retry_exhausted_or_unknown(
        retry_attempt_count=retry_attempt_count,
        max_retry_attempts=max_retry_attempts,
    ):
        return None
    return ProviderPressure(
        reason="provider_pressure",
        error_category=category,
        http_status=http_status,
        retry_attempt_count=retry_attempt_count,
        max_retry_attempts=max_retry_attempts,
    )


def _pressure_from_error_text(error: str | None) -> ProviderPressure | None:
    if not error:
        return None
    text = error.casefold()
    if "rate_limited" in text or "rate limited" in text:
        return ProviderPressure(
            reason="provider_pressure",
            error_category=ProviderErrorCategory.RATE_LIMITED.value,
            http_status=429,
        )
    if (
        "network_error" in text
        or "timeout" in text
        or "timed out" in text
        or "connection" in text
    ):
        return ProviderPressure(
            reason="provider_pressure",
            error_category=ProviderErrorCategory.NETWORK_ERROR.value,
        )
    return None


def _is_pressure_category(category: str, http_status: int | None) -> bool:
    if category in _PRESSURE_CATEGORIES:
        return True
    return category == ProviderErrorCategory.PROVIDER_ERROR.value and (
        http_status is None or http_status >= 500
    )


def _retry_exhausted_or_unknown(
    *,
    retry_attempt_count: int | None,
    max_retry_attempts: int | None,
) -> bool:
    if retry_attempt_count is None or max_retry_attempts is None:
        return True
    return retry_attempt_count >= max_retry_attempts


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
