"""Provider error categories used by services and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ProviderErrorCategory(StrEnum):
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    AUTHENTICATION_FAILED = "authentication_failed"
    MODEL_NOT_FOUND = "model_not_found"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    SECRET_STORAGE_ERROR = "secret_storage_error"
    PROVIDER_ERROR = "provider_error"
    IMAGE_GENERATION_FAILED = "image_generation_failed"
    CONTENT_BLOCKED = "content_blocked"


@dataclass
class ProviderError(Exception):
    category: ProviderErrorCategory
    message: str
    status_code: int | None = None
    retry_attempt_count: int | None = None
    max_retry_attempts: int | None = None
    retry_attempts: tuple[dict[str, object], ...] = ()
    fallback_attempted: bool | None = None
    fallback_skipped_reason: str | None = None
    fallback_provider: str | None = None
    fallback_model_id: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        message = self.message.strip()
        fallback_details = _fallback_details(self)
        if fallback_details:
            detail_text = "; ".join(fallback_details)
            message = f"{message} ({detail_text})" if message else detail_text
        if not message:
            return self.category.value
        return f"{self.category.value}: {message}"


def _fallback_details(exc: ProviderError) -> list[str]:
    details: list[str] = []
    if exc.fallback_attempted is not None:
        details.append(
            f"fallback_attempted={str(exc.fallback_attempted).lower()}"
        )
    if exc.fallback_skipped_reason:
        details.append(f"fallback_skipped_reason={exc.fallback_skipped_reason}")
    if exc.fallback_provider:
        details.append(f"fallback_provider={exc.fallback_provider}")
    if exc.fallback_model_id:
        details.append(f"fallback_model_id={exc.fallback_model_id}")
    return details


def map_http_status_to_category(status_code: int) -> ProviderErrorCategory:
    if status_code in {401, 403}:
        return ProviderErrorCategory.AUTHENTICATION_FAILED
    if status_code == 404:
        return ProviderErrorCategory.MODEL_NOT_FOUND
    if status_code == 408:
        return ProviderErrorCategory.NETWORK_ERROR
    if status_code == 413:
        return ProviderErrorCategory.CONTEXT_LIMIT_EXCEEDED
    if status_code == 429:
        return ProviderErrorCategory.RATE_LIMITED
    if 500 <= status_code:
        return ProviderErrorCategory.PROVIDER_ERROR
    return ProviderErrorCategory.PROVIDER_ERROR


def map_exception_to_category(exc: Exception) -> ProviderErrorCategory:
    if isinstance(exc, TimeoutError | ConnectionError):
        return ProviderErrorCategory.NETWORK_ERROR
    return ProviderErrorCategory.PROVIDER_ERROR
