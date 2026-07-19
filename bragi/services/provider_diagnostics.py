"""Safe provider retry diagnostics for persisted job results."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from bragi.providers.errors import ProviderError

_PROVIDER_CALLS: ContextVar[list[dict[str, object]] | None] = ContextVar(
    "bragi_provider_diagnostic_calls",
    default=None,
)


@dataclass(frozen=True)
class ProviderDiagnosticsCollection:
    calls: list[dict[str, object]]
    token: Token[list[dict[str, object]] | None]


def begin_provider_diagnostics_collection() -> ProviderDiagnosticsCollection:
    calls: list[dict[str, object]] = []
    return ProviderDiagnosticsCollection(
        calls=calls,
        token=_PROVIDER_CALLS.set(calls),
    )


def finish_provider_diagnostics_collection(
    collection: ProviderDiagnosticsCollection,
) -> tuple[dict[str, object], ...]:
    _PROVIDER_CALLS.reset(collection.token)
    return tuple(_safe_provider_call(call) for call in collection.calls)


def record_provider_response(
    *,
    task: str,
    provider: str,
    model_id: str | None = None,
    raw_metadata: dict[str, object] | None = None,
    operation: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    retry = retry_diagnostics_from_metadata(raw_metadata or {})
    if not retry:
        return
    _record_provider_call(
        {
            "task": task,
            "operation": operation or task,
            "provider": provider,
            **({"model": model_id} if model_id is not None else {}),
            **_safe_extra(extra),
            **retry,
        }
    )


def record_provider_error(
    *,
    task: str,
    provider: str,
    model_id: str | None = None,
    exc: Exception,
    operation: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    retry = retry_diagnostics_from_error(exc)
    if not retry:
        return
    fields: dict[str, object] = {
        "task": task,
        "operation": operation or task,
        "provider": provider,
        **({"model": model_id} if model_id is not None else {}),
        **_safe_extra(extra),
        **retry,
    }
    if isinstance(exc, ProviderError):
        fields["error_category"] = exc.category.value
        if exc.status_code is not None:
            fields["http_status"] = exc.status_code
    _record_provider_call(fields)


def retry_diagnostics_from_metadata(
    raw_metadata: dict[str, object],
) -> dict[str, object]:
    retry = raw_metadata.get("_bragi_retry")
    if not isinstance(retry, dict):
        return {}
    attempt_count = retry.get("attempt_count")
    max_attempts = retry.get("max_attempts")
    result: dict[str, object] = {}
    if isinstance(attempt_count, int) and not isinstance(attempt_count, bool):
        result["attempt_count"] = attempt_count
    if isinstance(max_attempts, int) and not isinstance(max_attempts, bool):
        result["max_attempts"] = max_attempts
    retry_attempts = safe_retry_attempts(retry.get("attempts"))
    if retry_attempts:
        result["retry_attempts"] = retry_attempts
    return result


def retry_diagnostics_from_error(exc: Exception) -> dict[str, object]:
    if not isinstance(exc, ProviderError):
        return {}
    result: dict[str, object] = {}
    if exc.retry_attempt_count is not None:
        result["attempt_count"] = exc.retry_attempt_count
    if exc.max_retry_attempts is not None:
        result["max_attempts"] = exc.max_retry_attempts
    retry_attempts = safe_retry_attempts(exc.retry_attempts)
    if retry_attempts:
        result["retry_attempts"] = retry_attempts
    return result


def safe_provider_error_diagnostics(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    safe: dict[str, object] = {}
    finish_reason = value.get("finish_reason")
    if isinstance(finish_reason, str):
        safe["finish_reason"] = finish_reason
    native_finish_reason = value.get("native_finish_reason")
    if isinstance(native_finish_reason, str):
        safe["native_finish_reason"] = native_finish_reason
    reasoning_tokens = value.get("reasoning_tokens")
    if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
        safe["reasoning_tokens"] = reasoning_tokens
    detail_types = value.get("reasoning_detail_types")
    if isinstance(detail_types, list | tuple):
        sanitized_detail_types = [
            detail_type
            for detail_type in detail_types
            if isinstance(detail_type, str) and detail_type
        ]
        if sanitized_detail_types:
            safe["reasoning_detail_types"] = sanitized_detail_types
    return safe


def safe_retry_attempts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list | tuple):
        return []
    attempts: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        attempt = item.get("attempt")
        duration_ms = item.get("duration_ms")
        if not isinstance(attempt, int) or isinstance(attempt, bool):
            continue
        if not isinstance(duration_ms, int) or isinstance(duration_ms, bool):
            continue
        normalized: dict[str, object] = {
            "attempt": attempt,
            "duration_ms": duration_ms,
        }
        error_category = item.get("error_category")
        if isinstance(error_category, str) or error_category is None:
            normalized["error_category"] = error_category
        http_status = item.get("http_status")
        if isinstance(http_status, int) and not isinstance(http_status, bool):
            normalized["http_status"] = http_status
        attempts.append(normalized)
    return attempts


def result_with_provider_diagnostics(
    result: dict[str, object] | None,
    *,
    provider_calls: tuple[dict[str, object], ...],
) -> dict[str, object] | None:
    safe_calls = [_safe_provider_call(call) for call in provider_calls]
    safe_calls = [call for call in safe_calls if _has_retry_diagnostics(call)]
    if not safe_calls:
        return result

    merged: dict[str, object] = dict(result or {})
    merged["provider_call_count"] = len(safe_calls)
    merged["provider_calls"] = safe_calls
    if len(safe_calls) == 1:
        call = safe_calls[0]
        for key in ("attempt_count", "max_attempts", "retry_attempts"):
            if key in call and key not in merged:
                merged[key] = call[key]
    return merged


def retry_summary(result: dict[str, object] | None) -> str | None:
    if not result:
        return None
    attempt_count = result.get("attempt_count")
    max_attempts = result.get("max_attempts")
    if isinstance(attempt_count, int) and isinstance(max_attempts, int):
        return f"retry={attempt_count}/{max_attempts}"
    provider_calls = result.get("provider_calls")
    if not isinstance(provider_calls, list | tuple):
        return None
    for call in reversed(provider_calls):
        if not isinstance(call, dict):
            continue
        attempt_count = call.get("attempt_count")
        max_attempts = call.get("max_attempts")
        if isinstance(attempt_count, int) and isinstance(max_attempts, int):
            return f"retry={attempt_count}/{max_attempts}"
    return None


def _record_provider_call(fields: dict[str, object]) -> None:
    calls = _PROVIDER_CALLS.get()
    if calls is None:
        return
    safe_call = _safe_provider_call(fields)
    if _has_retry_diagnostics(safe_call):
        calls.append(safe_call)


def _safe_extra(extra: dict[str, object] | None) -> dict[str, object]:
    if not extra:
        return {}
    return {
        key: value
        for key, value in extra.items()
        if key in {"section_id", "schema_name"} and isinstance(value, str)
    }


def _safe_provider_call(call: dict[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key in ("task", "provider", "model", "section_id", "schema_name"):
        value = call.get(key)
        if isinstance(value, str) and value:
            safe[key] = value
    operation = call.get("operation")
    if (
        isinstance(operation, str)
        and operation
        and operation != safe.get("task")
    ):
        safe["operation"] = operation
    for key in ("attempt_count", "max_attempts", "http_status"):
        value = call.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            safe[key] = value
    error_category = call.get("error_category")
    if isinstance(error_category, str):
        safe["error_category"] = error_category
    retry_attempts = safe_retry_attempts(call.get("retry_attempts"))
    if retry_attempts:
        safe["retry_attempts"] = retry_attempts
    return safe


def _has_retry_diagnostics(call: dict[str, object]) -> bool:
    return "attempt_count" in call or "max_attempts" in call or "retry_attempts" in call
