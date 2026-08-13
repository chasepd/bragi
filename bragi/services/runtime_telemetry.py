"""Metadata-only runtime telemetry helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from bragi.app_logging import exception_log_fields, log_error_event

_PROVIDER_METHOD_TASKS = {
    "validate_config": "model_listing",
    "list_models": "model_listing",
    "list_models_with_metadata": "model_listing",
    "chat": "chat",
    "stream_chat": "chat",
    "generate_image": "image_generation",
    "generate_video": "video_generation",
    "describe_image": "vision",
    "generate_structured_output": "structured_output",
    "generate_tool_calls": "tool_calling",
}


@dataclass(frozen=True)
class RuntimeTelemetryContext:
    repositories: Any
    job_id: str
    task: str | None = None
    root_repositories: Any | None = None
    root_job_id: str | None = None


_CURRENT_CONTEXT: ContextVar[RuntimeTelemetryContext | None] = ContextVar(
    "bragi_runtime_telemetry_context",
    default=None,
)
_SAFE_TEXT_METADATA_KEYS = frozenset(
    {
        "openrouter_selected_model",
        "openrouter_selected_provider",
        "turn_responsiveness_mode",
    }
)
_SAFE_TEXT_LIST_METADATA_KEYS = frozenset({"openrouter_provider_attempts"})
_SAFE_NUMBER_LIST_METADATA_KEYS = frozenset(
    {"openrouter_provider_attempt_statuses"}
)
_MAX_SAFE_METADATA_TEXT_LENGTH = 200
_MAX_SAFE_METADATA_LIST_ITEMS = 20


@contextmanager
def runtime_telemetry_context(
    *,
    repositories: Any,
    job_id: str,
    task: str | None = None,
) -> Any:
    current = _CURRENT_CONTEXT.get()
    token = _CURRENT_CONTEXT.set(
        RuntimeTelemetryContext(
            repositories=repositories,
            job_id=job_id,
            task=task,
            root_repositories=(
                current.root_repositories or current.repositories
                if current is not None
                else repositories
            ),
            root_job_id=(
                current.root_job_id or current.job_id
                if current is not None
                else job_id
            ),
        )
    )
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


@contextmanager
def provider_task_context(task: str | None) -> Any:
    context = _CURRENT_CONTEXT.get()
    if context is None:
        yield
        return
    token = _CURRENT_CONTEXT.set(replace(context, task=task))
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def current_runtime_telemetry_context() -> RuntimeTelemetryContext | None:
    return _CURRENT_CONTEXT.get()


@contextmanager
def runtime_job_step(
    name: str,
    *,
    root: bool = False,
    metadata: dict[str, object] | None = None,
) -> Any:
    """Record one metadata-only runtime span without affecting application work."""
    context = _CURRENT_CONTEXT.get()
    if context is None:
        yield
        return
    repositories = (
        context.root_repositories or context.repositories
        if root
        else context.repositories
    )
    job_id = context.root_job_id or context.job_id if root else context.job_id
    started = perf_counter()
    started_at = _utc_now()
    try:
        yield
    except asyncio.CancelledError:
        _record_runtime_job_span(
            repositories=repositories,
            job_id=job_id,
            name=name,
            task=context.task,
            status="cancelled",
            started=started,
            started_at=started_at,
            error="Cancelled",
            metadata=metadata,
        )
        raise
    except Exception as exc:
        _record_runtime_job_span(
            repositories=repositories,
            job_id=job_id,
            name=name,
            task=context.task,
            status="failed",
            started=started,
            started_at=started_at,
            error=exc.__class__.__name__,
            metadata=metadata,
        )
        raise
    _record_runtime_job_span(
        repositories=repositories,
        job_id=job_id,
        name=name,
        task=context.task,
        status="succeeded",
        started=started,
        started_at=started_at,
        metadata=metadata,
    )


def _record_runtime_job_span(
    *,
    repositories: Any,
    job_id: str,
    name: str,
    task: str | None,
    status: str,
    started: float,
    started_at: str,
    error: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    record_job_step(
        repositories=repositories,
        job_id=job_id,
        name=name,
        status=status,
        task=task,
        started_at=started_at,
        completed_at=_utc_now(),
        duration_ms=_elapsed_ms(started),
        error=error,
        metadata=metadata,
    )


def record_current_job_step(
    *,
    name: str,
    status: str,
    provider: str | None = None,
    model: str | None = None,
    task: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    duration_ms: int | None = None,
    error: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    context = _CURRENT_CONTEXT.get()
    if context is None:
        return
    record_job_step(
        repositories=context.repositories,
        job_id=context.job_id,
        name=name,
        status=status,
        provider=provider,
        model=model,
        task=task or context.task,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        error=error,
        metadata=metadata,
    )


def record_job_step(
    *,
    repositories: Any,
    job_id: str,
    name: str,
    status: str,
    provider: str | None = None,
    model: str | None = None,
    task: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    duration_ms: int | None = None,
    error: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    recorder = getattr(repositories, "record_job_step", None)
    if not callable(recorder):
        return
    try:
        recorder(
            job_id=job_id,
            name=name,
            status=status,
            provider=provider,
            model=model,
            task=task,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            error=error,
            metadata=safe_job_step_metadata(metadata),
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must not break runtime work
        log_error_event(
            "runtime.telemetry_step_record_failed",
            job_id=job_id,
            step_name=name,
            step_status=status,
            **exception_log_fields(exc),
        )


def safe_job_step_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in (metadata or {}).items():
        normalized_key = str(key)
        if isinstance(value, bool):
            safe[normalized_key] = value
            continue
        if isinstance(value, int | float):
            safe[normalized_key] = value
            continue
        if normalized_key in _SAFE_TEXT_METADATA_KEYS:
            text = _safe_metadata_text(value)
            if text is not None:
                safe[normalized_key] = text
            continue
        if normalized_key in _SAFE_TEXT_LIST_METADATA_KEYS:
            text_values = _safe_metadata_text_list(value)
            if text_values:
                safe[normalized_key] = text_values
            continue
        if normalized_key in _SAFE_NUMBER_LIST_METADATA_KEYS:
            number_values = _safe_metadata_number_list(value)
            if number_values:
                safe[normalized_key] = number_values
    return safe


def wrap_provider_clients_for_telemetry(
    providers: Mapping[str, Any],
    *,
    repositories: Any,
) -> dict[str, Any]:
    return {
        name: TelemetryProviderClient(provider, repositories=repositories)
        for name, provider in providers.items()
    }


class TelemetryProviderClient:
    provider_name: str

    def __init__(self, delegate: Any, *, repositories: Any) -> None:
        self._delegate = delegate
        self._repositories = repositories
        provider_name = getattr(delegate, "provider_name", "")
        self.provider_name = provider_name if isinstance(provider_name, str) else ""
        for method_name in _PROVIDER_METHOD_TASKS:
            method = getattr(delegate, method_name, None)
            if not callable(method):
                continue
            if method_name == "stream_chat":
                setattr(self, method_name, self._stream_wrapper(method_name, method))
            else:
                setattr(self, method_name, self._async_wrapper(method_name, method))
        for passthrough_name in ("image_reference_limit",):
            passthrough = getattr(delegate, passthrough_name, None)
            if callable(passthrough):
                setattr(self, passthrough_name, passthrough)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def _async_wrapper(
        self,
        method_name: str,
        method: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = perf_counter()
            started_at = _utc_now()
            request = _request_from_args(args, kwargs)
            try:
                response = await method(*args, **kwargs)
            except asyncio.CancelledError:
                _record_provider_call(
                    method_name=method_name,
                    request=request,
                    response=None,
                    provider_name=self.provider_name,
                    status="cancelled",
                    started=started,
                    started_at=started_at,
                    error="Cancelled",
                )
                raise
            except Exception as exc:
                _record_provider_call(
                    method_name=method_name,
                    request=request,
                    response=None,
                    provider_name=self.provider_name,
                    status="failed",
                    started=started,
                    started_at=started_at,
                    error=str(exc) or exc.__class__.__name__,
                    metadata=_provider_error_metadata(exc),
                )
                raise
            _record_provider_call(
                method_name=method_name,
                request=request,
                response=response,
                provider_name=self.provider_name,
                status="succeeded",
                started=started,
                started_at=started_at,
            )
            return response

        return wrapped

    def _stream_wrapper(
        self,
        method_name: str,
        method: Callable[..., AsyncIterator[Any]],
    ) -> Callable[..., AsyncIterator[Any]]:
        def wrapped(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            request = _request_from_args(args, kwargs)
            started = perf_counter()
            started_at = _utc_now()
            token_usage: dict[str, int] = {}
            stream_metadata: dict[str, object] = {}
            first_chunk_ms: int | None = None
            output_chars = 0

            async def stream() -> AsyncIterator[Any]:
                nonlocal first_chunk_ms, output_chars
                try:
                    async for chunk in method(*args, **kwargs):
                        delta = getattr(chunk, "delta", "")
                        if isinstance(delta, str) and delta:
                            output_chars += len(delta)
                            if first_chunk_ms is None:
                                first_chunk_ms = _elapsed_ms(started)
                        usage = getattr(chunk, "token_usage", None)
                        if isinstance(usage, Mapping):
                            token_usage.update(
                                {
                                    str(key): int(value)
                                    for key, value in usage.items()
                                    if isinstance(value, int)
                                    and not isinstance(value, bool)
                                }
                            )
                        raw_metadata = getattr(chunk, "raw_metadata", None)
                        if isinstance(raw_metadata, Mapping):
                            stream_metadata.update(
                                _openrouter_provider_metadata(raw_metadata)
                            )
                        yield chunk
                except asyncio.CancelledError:
                    _record_provider_call(
                        method_name=method_name,
                        request=request,
                        response=None,
                        provider_name=self.provider_name,
                        status="cancelled",
                        started=started,
                        started_at=started_at,
                        error="Cancelled",
                        metadata={
                            **_token_usage_metadata(token_usage),
                            **stream_metadata,
                            **_stream_performance_metadata(
                                started=started,
                                first_chunk_ms=first_chunk_ms,
                                output_chars=output_chars,
                            ),
                        },
                    )
                    raise
                except Exception as exc:
                    _record_provider_call(
                        method_name=method_name,
                        request=request,
                        response=None,
                        provider_name=self.provider_name,
                        status="failed",
                        started=started,
                        started_at=started_at,
                        error=str(exc) or exc.__class__.__name__,
                        metadata={
                            **_token_usage_metadata(token_usage),
                            **stream_metadata,
                            **_provider_error_metadata(exc),
                            **_stream_performance_metadata(
                                started=started,
                                first_chunk_ms=first_chunk_ms,
                                output_chars=output_chars,
                            ),
                        },
                    )
                    raise
                _record_provider_call(
                    method_name=method_name,
                    request=request,
                    response=None,
                    provider_name=self.provider_name,
                    status="succeeded",
                    started=started,
                    started_at=started_at,
                    metadata={
                        **_token_usage_metadata(token_usage),
                        **stream_metadata,
                        **_stream_performance_metadata(
                            started=started,
                            first_chunk_ms=first_chunk_ms,
                            output_chars=output_chars,
                        ),
                    },
                )

            return stream()

        return wrapped


def _stream_performance_metadata(
    *,
    started: float,
    first_chunk_ms: int | None,
    output_chars: int,
) -> dict[str, object]:
    elapsed_seconds = max(perf_counter() - started, 0.0)
    metadata: dict[str, object] = {"stream_output_chars": output_chars}
    if first_chunk_ms is not None:
        metadata["stream_first_chunk_ms"] = first_chunk_ms
    if elapsed_seconds > 0:
        metadata["stream_output_chars_per_second"] = round(
            output_chars / elapsed_seconds,
            2,
        )
    return metadata


def _record_provider_call(
    *,
    method_name: str,
    request: object,
    response: object,
    provider_name: str,
    status: str,
    started: float,
    started_at: str,
    error: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    context = _CURRENT_CONTEXT.get()
    if context is None:
        return
    provider = _text_attr(request, "provider") or _text_attr(response, "provider")
    provider = provider or provider_name or None
    model = _text_attr(request, "model_id") or _text_attr(response, "model_id")
    task = context.task or _PROVIDER_METHOD_TASKS[method_name]
    record_job_step(
        repositories=context.repositories,
        job_id=context.job_id,
        name=f"provider.{method_name}",
        status=status,
        provider=provider,
        model=model,
        task=task,
        started_at=started_at,
        completed_at=_utc_now(),
        duration_ms=_elapsed_ms(started),
        error=error,
        metadata={
            **_request_metadata(request),
            **_response_metadata(response),
            **(metadata or {}),
        },
    )


def _request_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
    if args:
        return args[0]
    return kwargs.get("request")


def _request_metadata(request: object) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for attr in ("temperature", "max_output_tokens"):
        value = getattr(request, attr, None)
        if isinstance(value, int | float) and not isinstance(value, bool):
            metadata[attr] = value
    dimensions = getattr(request, "dimensions", None)
    if (
        isinstance(dimensions, tuple)
        and len(dimensions) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in dimensions
        )
    ):
        metadata["width"] = dimensions[0]
        metadata["height"] = dimensions[1]
    return metadata


def _response_metadata(response: object) -> dict[str, object]:
    metadata: dict[str, object] = {}
    raw_metadata = getattr(response, "raw_metadata", None)
    if isinstance(raw_metadata, Mapping):
        metadata.update(
            {
                str(key): value
                for key, value in raw_metadata.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
        )
        metadata.update(_openrouter_provider_metadata(raw_metadata))
        metadata.update(_retry_metadata(raw_metadata.get("_bragi_retry")))
    token_usage = getattr(response, "token_usage", None)
    if isinstance(token_usage, Mapping):
        metadata.update(_token_usage_metadata(token_usage))
    return metadata


def _provider_error_metadata(exc: Exception) -> dict[str, object]:
    attempt_count = getattr(exc, "retry_attempt_count", None)
    max_attempts = getattr(exc, "max_retry_attempts", None)
    return _retry_count_metadata(
        attempt_count=attempt_count,
        max_attempts=max_attempts,
    )


def _retry_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return _retry_count_metadata(
        attempt_count=value.get("attempt_count"),
        max_attempts=value.get("max_attempts"),
    )


def _retry_count_metadata(
    *,
    attempt_count: object,
    max_attempts: object,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if isinstance(attempt_count, int) and not isinstance(attempt_count, bool):
        metadata["attempt_count"] = attempt_count
        metadata["retry_count"] = max(0, attempt_count - 1)
    if isinstance(max_attempts, int) and not isinstance(max_attempts, bool):
        metadata["max_attempts"] = max_attempts
    return metadata


def _token_usage_metadata(token_usage: Mapping[str, object]) -> dict[str, object]:
    return {
        f"token_{key}": value
        for key, value in token_usage.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _text_attr(value: object, attr: str) -> str | None:
    attr_value = getattr(value, attr, None)
    return attr_value if isinstance(attr_value, str) and attr_value else None


def _openrouter_provider_metadata(
    raw_metadata: Mapping[str, object],
) -> dict[str, object]:
    router_metadata = raw_metadata.get("openrouter_metadata")
    if not isinstance(router_metadata, Mapping):
        return {}

    metadata: dict[str, object] = {}
    endpoints = router_metadata.get("endpoints")
    if isinstance(endpoints, Mapping):
        total = endpoints.get("total")
        if isinstance(total, int | float) and not isinstance(total, bool):
            metadata["openrouter_available_provider_count"] = total
        selected_endpoint = _selected_openrouter_endpoint(endpoints)
        if selected_endpoint is not None:
            provider = _safe_metadata_text(selected_endpoint.get("provider"))
            model = _safe_metadata_text(selected_endpoint.get("model"))
            if provider is not None:
                metadata["openrouter_selected_provider"] = provider
            if model is not None:
                metadata["openrouter_selected_model"] = model

    attempts = router_metadata.get("attempts")
    if isinstance(attempts, list | tuple):
        providers: list[str] = []
        statuses: list[int | float] = []
        for attempt in attempts[:_MAX_SAFE_METADATA_LIST_ITEMS]:
            if not isinstance(attempt, Mapping):
                continue
            provider = _safe_metadata_text(attempt.get("provider"))
            if provider is not None:
                providers.append(provider)
            status = attempt.get("status")
            if isinstance(status, int | float) and not isinstance(status, bool):
                statuses.append(status)
        if providers:
            metadata["openrouter_provider_attempts"] = providers
        if statuses:
            metadata["openrouter_provider_attempt_statuses"] = statuses

    return metadata


def _selected_openrouter_endpoint(
    endpoints: Mapping[str, object],
) -> Mapping[str, object] | None:
    available = endpoints.get("available")
    if not isinstance(available, list | tuple):
        return None
    for endpoint in available:
        if isinstance(endpoint, Mapping) and endpoint.get("selected") is True:
            return endpoint
    return None


def _safe_metadata_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > _MAX_SAFE_METADATA_TEXT_LENGTH:
        return None
    return text


def _safe_metadata_text_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    values: list[str] = []
    for item in value[:_MAX_SAFE_METADATA_LIST_ITEMS]:
        text = _safe_metadata_text(item)
        if text is not None:
            values.append(text)
    return values


def _safe_metadata_number_list(value: object) -> list[int | float]:
    if not isinstance(value, list | tuple):
        return []
    values: list[int | float] = []
    for item in value[:_MAX_SAFE_METADATA_LIST_ITEMS]:
        if isinstance(item, bool):
            continue
        if isinstance(item, int | float):
            values.append(item)
    return values


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))
