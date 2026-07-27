"""Small JSON HTTP transport used by provider clients."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from email.message import Message
from time import perf_counter
from typing import Any, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.providers.errors import (
    ProviderError,
    ProviderErrorCategory,
    map_exception_to_category,
    map_http_status_to_category,
)
from bragi.redaction import redact_text


@dataclass(frozen=True)
class JsonHttpResponse:
    status_code: int
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BinaryHttpResponse:
    status_code: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


MAX_PROVIDER_RESPONSE_BYTES = 64 * 1024 * 1024
PROVIDER_RESPONSE_READ_CHUNK_BYTES = 64 * 1024
SAFE_PROVIDER_RESPONSE_HEADERS = frozenset(
    {
        "cf-ray",
        "content-type",
        "x-request-id",
        "x-retry-count",
        "x-venice-contains-minor",
        "x-venice-is-adult-model-content-violation",
        "x-venice-is-blurred",
        "x-venice-is-content-violation",
    }
)
SENSITIVE_PROVIDER_RESPONSE_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "proxy-authenticate",
        "set-cookie",
        "www-authenticate",
    }
)
_SOURCE_ID_PATTERN = re.compile(
    r"\[([a-z][a-z0-9_]*):([A-Za-z0-9][A-Za-z0-9._:-]{0,95})\]"
    r"(?![A-Za-z0-9._:-])"
)
_MAX_CAPTURED_SOURCE_IDS = 64
_TRUSTED_SOURCE_ID_TYPES = frozenset(
    {
        "character_text_thread",
        "character_voice",
        "media_asset",
        "memory",
        "message",
        "observation",
        "open_obligation",
        "scenario_section",
        "state_change",
        "summary",
        "world_state",
    }
)


@dataclass(frozen=True)
class _ProviderDeadline:
    timeout: float
    expires_at: float


class JsonHttpTransport(Protocol):
    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> JsonHttpResponse:
        ...


class BinaryHttpTransport(Protocol):
    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> BinaryHttpResponse:
        ...


async def request_sse_json(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: dict[str, Any] | None,
    timeout: float,
    provider: str | None = None,
    task: str | None = None,
    model: str | None = None,
    schema_name: str | None = None,
    max_response_bytes: int = MAX_PROVIDER_RESPONSE_BYTES,
) -> AsyncIterator[dict[str, Any]]:
    started_at = perf_counter()
    deadline = _provider_deadline(timeout, started_at=started_at)
    path = urlsplit(url).path
    body = None
    request_headers = dict(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request_headers.setdefault("Accept", "text/event-stream")
    request = Request(url=url, data=body, headers=request_headers, method=method)
    log_event(
        "provider.http_stream_started",
        method=method,
        path=path,
        timeout=timeout,
        payload_bytes=len(body) if body is not None else 0,
        **_safe_started_diagnostics(
            provider=provider,
            task=task,
            model=model,
            schema_name=schema_name,
            payload=payload,
        ),
    )
    queue: asyncio.Queue[object] = asyncio.Queue()
    done = object()
    loop = asyncio.get_running_loop()
    stop_event = threading.Event()
    response_lock = threading.Lock()
    response_holder: dict[str, Any] = {}

    def publish(item: object) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            pass

    def worker() -> None:
        byte_count = 0
        try:
            with _NO_REDIRECT_OPENER.open(
                request,
                timeout=_remaining_provider_timeout(deadline),
            ) as response:
                with response_lock:
                    response_holder["response"] = response
                for event_data in _iter_sse_data(response, deadline=deadline):
                    if stop_event.is_set():
                        break
                    byte_count += len(event_data.encode("utf-8"))
                    if byte_count > max_response_bytes:
                        raise ProviderError(
                            category=ProviderErrorCategory.PROVIDER_ERROR,
                            message=(
                                "Provider stream exceeded "
                                f"{max_response_bytes} bytes"
                            ),
                        )
                    compact = event_data.strip()
                    if not compact or compact == "[DONE]":
                        continue
                    payload = json.loads(compact)
                    if not isinstance(payload, dict):
                        raise ValueError("Provider stream event must be a JSON object")
                    publish(payload)
            if stop_event.is_set():
                return
            log_event(
                "provider.http_stream_succeeded",
                method=method,
                path=path,
                duration_ms=_elapsed_ms(started_at),
            )
        except HTTPError as exc:
            category = map_http_status_to_category(exc.code)
            log_error_event(
                "provider.http_stream_failed",
                method=method,
                path=path,
                status_code=exc.code,
                duration_ms=_elapsed_ms(started_at),
                error_category=category.value,
                error=f"Provider HTTP request failed with {exc.code}",
            )
            publish(
                ProviderError(
                    category=category,
                    message=_safe_http_error_message(exc.code),
                    status_code=exc.code,
                )
            )
        except URLError as exc:
            if stop_event.is_set():
                return
            log_error_event(
                "provider.http_stream_failed",
                method=method,
                path=path,
                duration_ms=_elapsed_ms(started_at),
                error_category=ProviderErrorCategory.NETWORK_ERROR.value,
                error=str(exc.reason),
            )
            publish(
                ProviderError(
                    category=ProviderErrorCategory.NETWORK_ERROR,
                    message=str(exc.reason),
                )
            )
        except TimeoutError as exc:
            if stop_event.is_set():
                return
            log_error_event(
                "provider.http_stream_failed",
                method=method,
                path=path,
                duration_ms=_elapsed_ms(started_at),
                error_category=ProviderErrorCategory.NETWORK_ERROR.value,
                error=str(exc),
            )
            publish(
                ProviderError(
                    category=ProviderErrorCategory.NETWORK_ERROR,
                    message=str(exc),
                )
            )
        except Exception as exc:
            if stop_event.is_set():
                return
            category = map_exception_to_category(exc)
            log_error_event(
                "provider.http_stream_failed",
                method=method,
                path=path,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            publish(ProviderError(category=category, message=str(exc)))
        finally:
            with response_lock:
                response_holder.pop("response", None)
            publish(done)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=_remaining_provider_timeout(deadline),
                )
            except TimeoutError as exc:
                timeout_error = _provider_timeout_error(deadline)
                log_error_event(
                    "provider.http_stream_failed",
                    method=method,
                    path=path,
                    duration_ms=_elapsed_ms(started_at),
                    error_category=ProviderErrorCategory.NETWORK_ERROR.value,
                    error=str(timeout_error),
                )
                raise ProviderError(
                    category=ProviderErrorCategory.NETWORK_ERROR,
                    message=str(timeout_error),
                ) from exc
            if item is done:
                break
            if isinstance(item, Exception):
                raise item
            yield cast(dict[str, Any], item)
    finally:
        stop_event.set()
        with response_lock:
            response = response_holder.get("response")
        close = getattr(response, "close", None)
        if callable(close):
            close()


def request_json(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: dict[str, Any] | None,
    timeout: float,
    provider: str | None = None,
    task: str | None = None,
    model: str | None = None,
    schema_name: str | None = None,
    max_response_bytes: int = MAX_PROVIDER_RESPONSE_BYTES,
) -> JsonHttpResponse:
    started_at = perf_counter()
    deadline = _provider_deadline(timeout, started_at=started_at)
    path = urlsplit(url).path
    body = None
    request_headers = dict(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = Request(url=url, data=body, headers=request_headers, method=method)
    log_event(
        "provider.http_started",
        method=method,
        path=path,
        timeout=timeout,
        payload_bytes=len(body) if body is not None else 0,
        **_safe_started_diagnostics(
            provider=provider,
            task=task,
            model=model,
            schema_name=schema_name,
            payload=payload,
        ),
    )
    try:
        with _NO_REDIRECT_OPENER.open(
            request,
            timeout=_remaining_provider_timeout(deadline),
        ) as response:
            raw_body = _read_limited_text(
                response,
                max_bytes=max_response_bytes,
                description="Provider response",
                deadline=deadline,
            )
            result = JsonHttpResponse(
                status_code=response.status,
                payload=_decode_json(raw_body),
                headers=_safe_response_headers(response.headers),
            )
            log_event(
                "provider.http_succeeded",
                method=method,
                path=path,
                status_code=response.status,
                duration_ms=_elapsed_ms(started_at),
            )
            return result
    except ProviderError as exc:
        log_error_event(
            "provider.http_failed",
            method=method,
            path=path,
            duration_ms=_elapsed_ms(started_at),
            error_category=exc.category.value,
            error=exc.message,
        )
        raise
    except HTTPError as exc:
        category = map_http_status_to_category(exc.code)
        venice_error_fields = _venice_http_error_diagnostics(
            provider=provider,
            error=exc,
        )
        log_error_event(
            "provider.http_failed",
            method=method,
            path=path,
            status_code=exc.code,
            duration_ms=_elapsed_ms(started_at),
            error_category=category.value,
            error=f"Provider HTTP request failed with {exc.code}",
            **_safe_started_diagnostics(
                provider=provider,
                task=task,
                model=model,
                schema_name=schema_name,
                payload=payload,
            ),
            **venice_error_fields,
        )
        raise ProviderError(
            category=category,
            message=_safe_http_error_message(exc.code),
            status_code=exc.code,
        ) from exc
    except URLError as exc:
        log_error_event(
            "provider.http_failed",
            method=method,
            path=path,
            duration_ms=_elapsed_ms(started_at),
            error_category=ProviderErrorCategory.NETWORK_ERROR.value,
            error=str(exc.reason),
        )
        raise ProviderError(
            category=ProviderErrorCategory.NETWORK_ERROR,
            message=str(exc.reason),
        ) from exc
    except TimeoutError as exc:
        log_error_event(
            "provider.http_failed",
            method=method,
            path=path,
            duration_ms=_elapsed_ms(started_at),
            error_category=ProviderErrorCategory.NETWORK_ERROR.value,
            error=str(exc),
        )
        raise ProviderError(
            category=ProviderErrorCategory.NETWORK_ERROR,
            message=str(exc),
        ) from exc
    except Exception as exc:
        category = map_exception_to_category(exc)
        log_error_event(
            "provider.http_failed",
            method=method,
            path=path,
            duration_ms=_elapsed_ms(started_at),
            **exception_log_fields(exc),
        )
        raise ProviderError(
            category=category,
            message=str(exc),
        ) from exc


def request_bytes(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: dict[str, Any] | None,
    timeout: float,
    provider: str | None = None,
    task: str | None = None,
    model: str | None = None,
    schema_name: str | None = None,
    max_response_bytes: int = MAX_PROVIDER_RESPONSE_BYTES,
) -> BinaryHttpResponse:
    started_at = perf_counter()
    deadline = _provider_deadline(timeout, started_at=started_at)
    path = urlsplit(url).path
    body = None
    request_headers = dict(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = Request(url=url, data=body, headers=request_headers, method=method)
    log_event(
        "provider.http_started",
        method=method,
        path=path,
        timeout=timeout,
        payload_bytes=len(body) if body is not None else 0,
        **_safe_started_diagnostics(
            provider=provider,
            task=task,
            model=model,
            schema_name=schema_name,
            payload=payload,
        ),
    )
    try:
        with _NO_REDIRECT_OPENER.open(
            request,
            timeout=_remaining_provider_timeout(deadline),
        ) as response:
            raw_body = _read_limited_bytes(
                response,
                max_bytes=max_response_bytes,
                description="Provider response",
                deadline=deadline,
            )
            result = BinaryHttpResponse(
                status_code=response.status,
                body=raw_body,
                headers=_safe_response_headers(response.headers),
            )
            log_event(
                "provider.http_succeeded",
                method=method,
                path=path,
                status_code=response.status,
                duration_ms=_elapsed_ms(started_at),
            )
            return result
    except ProviderError as exc:
        log_error_event(
            "provider.http_failed",
            method=method,
            path=path,
            duration_ms=_elapsed_ms(started_at),
            error_category=exc.category.value,
            error=exc.message,
        )
        raise
    except HTTPError as exc:
        category = map_http_status_to_category(exc.code)
        venice_error_fields = _venice_http_error_diagnostics(
            provider=provider,
            error=exc,
        )
        log_error_event(
            "provider.http_failed",
            method=method,
            path=path,
            status_code=exc.code,
            duration_ms=_elapsed_ms(started_at),
            error_category=category.value,
            error=f"Provider HTTP request failed with {exc.code}",
            **_safe_started_diagnostics(
                provider=provider,
                task=task,
                model=model,
                schema_name=schema_name,
                payload=payload,
            ),
            **venice_error_fields,
        )
        raise ProviderError(
            category=category,
            message=_safe_http_error_message(exc.code),
            status_code=exc.code,
        ) from exc
    except URLError as exc:
        log_error_event(
            "provider.http_failed",
            method=method,
            path=path,
            duration_ms=_elapsed_ms(started_at),
            error_category=ProviderErrorCategory.NETWORK_ERROR.value,
            error=str(exc.reason),
        )
        raise ProviderError(
            category=ProviderErrorCategory.NETWORK_ERROR,
            message=str(exc.reason),
        ) from exc
    except TimeoutError as exc:
        log_error_event(
            "provider.http_failed",
            method=method,
            path=path,
            duration_ms=_elapsed_ms(started_at),
            error_category=ProviderErrorCategory.NETWORK_ERROR.value,
            error=str(exc),
        )
        raise ProviderError(
            category=ProviderErrorCategory.NETWORK_ERROR,
            message=str(exc),
        ) from exc
    except Exception as exc:
        category = map_exception_to_category(exc)
        log_error_event(
            "provider.http_failed",
            method=method,
            path=path,
            duration_ms=_elapsed_ms(started_at),
            **exception_log_fields(exc),
        )
        raise ProviderError(
            category=category,
            message=str(exc),
        ) from exc


def ensure_success(response: JsonHttpResponse) -> dict[str, Any]:
    if 200 <= response.status_code < 300:
        return _payload_with_safe_headers(response)
    raise ProviderError(
        category=map_http_status_to_category(response.status_code),
        message=_safe_http_error_message(response.status_code),
        status_code=response.status_code,
    )


def ensure_binary_success(response: BinaryHttpResponse) -> bytes:
    if 200 <= response.status_code < 300:
        return response.body
    raise ProviderError(
        category=map_http_status_to_category(response.status_code),
        message=_safe_http_error_message(response.status_code),
        status_code=response.status_code,
    )


def _decode_json(raw_body: str) -> dict[str, Any]:
    if not raw_body:
        return {}
    payload = json.loads(raw_body)
    if not isinstance(payload, dict):
        raise ValueError("Provider response must be a JSON object")
    return payload


def _safe_http_error_message(status_code: int) -> str:
    category = map_http_status_to_category(status_code)
    return f"{category.value} ({status_code})"


def _payload_with_safe_headers(response: JsonHttpResponse) -> dict[str, Any]:
    if not response.headers:
        return response.payload
    payload = dict(response.payload)
    payload["_bragi_headers"] = dict(response.headers)
    return payload


def _safe_response_headers(headers: Mapping[str, object]) -> dict[str, str]:
    safe_headers: dict[str, str] = {}
    for key, value in headers.items():
        normalized = str(key).strip().lower()
        if normalized not in SAFE_PROVIDER_RESPONSE_HEADERS:
            continue
        safe_headers[normalized] = str(value)
    return safe_headers


def _venice_http_error_diagnostics(
    *,
    provider: str | None,
    error: HTTPError,
) -> dict[str, object]:
    if provider != "venice":
        return {}
    return {
        "response_body_suppressed": True,
        "response_body_suppressed_reason": (
            "provider_error_body_may_contain_private_content"
        ),
        "response_headers": _redacted_response_headers(error.headers),
    }


def _redacted_response_headers(
    headers: Mapping[str, object] | Message[str, str],
) -> dict[str, str]:
    redacted_headers: dict[str, str] = {}
    for key, value in headers.items():
        normalized = str(key).strip().lower()
        if not normalized or normalized in SENSITIVE_PROVIDER_RESPONSE_HEADERS:
            continue
        redacted_headers[normalized] = redact_text(str(value)) or ""
    return redacted_headers


def _safe_started_diagnostics(
    *,
    provider: str | None,
    task: str | None,
    model: str | None,
    schema_name: str | None,
    payload: dict[str, Any] | None,
) -> dict[str, object]:
    fields: dict[str, object] = {}
    if provider:
        fields["provider"] = provider
    if task:
        fields["task"] = task
    if model:
        fields["model"] = model
    if schema_name:
        fields["schema_name"] = schema_name
    if payload is not None:
        canonical_payload = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        fields["payload_sha256"] = hashlib.sha256(canonical_payload).hexdigest()
        message_diagnostics = _safe_message_diagnostics(payload)
        fields.update(message_diagnostics)
    schema_enum_value_count = _schema_enum_value_count_from_payload(payload)
    if schema_enum_value_count is not None:
        fields["schema_enum_value_count"] = schema_enum_value_count
    return fields


def _safe_message_diagnostics(payload: dict[str, Any]) -> dict[str, object]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return {}
    content_chars = 0
    source_ids: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            content_chars += len(content)
            _append_source_ids(source_ids, content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                content_chars += len(text)
                _append_source_ids(source_ids, text)
    diagnostics: dict[str, object] = {
        "message_count": len(messages),
        "message_content_chars": content_chars,
    }
    if source_ids:
        diagnostics["source_id_count"] = len(source_ids)
        diagnostics["source_id_sha256"] = [
            hashlib.sha256(source_id.encode("utf-8")).hexdigest()
            for source_id in source_ids
        ]
    return diagnostics


def _append_source_ids(source_ids: list[str], text: str) -> None:
    if len(source_ids) >= _MAX_CAPTURED_SOURCE_IDS:
        return
    for source_type, identifier in _SOURCE_ID_PATTERN.findall(text):
        if source_type not in _TRUSTED_SOURCE_ID_TYPES:
            continue
        source_id = f"{source_type}:{identifier}"
        if source_id in source_ids:
            continue
        if len(source_ids) >= _MAX_CAPTURED_SOURCE_IDS:
            return
        source_ids.append(source_id)


def _schema_enum_value_count_from_payload(payload: dict[str, Any] | None) -> int | None:
    if payload is None:
        return None
    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return None
    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        return None
    return _schema_enum_value_count(schema)


def _schema_enum_value_count(value: object) -> int:
    if isinstance(value, dict):
        enum = value.get("enum")
        total = len(enum) if isinstance(enum, list) else 0
        return total + sum(_schema_enum_value_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(_schema_enum_value_count(child) for child in value)
    return 0


def _read_limited_text(
    handle: Any,
    *,
    max_bytes: int,
    description: str,
    deadline: _ProviderDeadline | None = None,
) -> str:
    raw_body = _read_limited_bytes(
        handle,
        max_bytes=max_bytes,
        description=description,
        deadline=deadline,
    )
    return raw_body.decode("utf-8")


def _read_limited_bytes(
    handle: Any,
    *,
    max_bytes: int,
    description: str,
    deadline: _ProviderDeadline | None = None,
) -> bytes:
    raw_body = bytearray()
    while len(raw_body) <= max_bytes:
        if deadline is not None:
            _apply_remaining_provider_timeout(handle, deadline)
        chunk_size = max(1, int(PROVIDER_RESPONSE_READ_CHUNK_BYTES))
        read_size = min(
            chunk_size,
            max_bytes + 1 - len(raw_body),
        )
        chunk = bytes(handle.read(read_size))
        if not chunk:
            break
        raw_body.extend(chunk)
        if deadline is not None:
            _raise_if_provider_deadline_expired(deadline)
        if len(raw_body) > max_bytes:
            break
        if len(chunk) < read_size:
            break
    if len(raw_body) > max_bytes:
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message=f"{description} exceeded {max_bytes} bytes",
        )
    return bytes(raw_body)


def _provider_deadline(timeout: float, *, started_at: float) -> _ProviderDeadline:
    normalized_timeout = max(0.0, timeout)
    return _ProviderDeadline(
        timeout=normalized_timeout,
        expires_at=started_at + normalized_timeout,
    )


def _remaining_provider_timeout(deadline: _ProviderDeadline) -> float:
    remaining = deadline.expires_at - perf_counter()
    if remaining <= 0:
        raise _provider_timeout_error(deadline)
    return remaining


def _apply_remaining_provider_timeout(
    handle: Any,
    deadline: _ProviderDeadline,
) -> None:
    remaining = _remaining_provider_timeout(deadline)
    for candidate in _timeout_candidates(handle):
        settimeout = getattr(candidate, "settimeout", None)
        if not callable(settimeout):
            continue
        try:
            settimeout(remaining)
            return
        except Exception:
            continue


def _timeout_candidates(handle: Any) -> Iterator[Any]:
    stack = [handle]
    seen: set[int] = set()
    for _ in range(24):
        if not stack:
            return
        candidate = stack.pop()
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        yield candidate
        for attr in ("fp", "raw", "_sock", "sock", "socket"):
            try:
                child = getattr(candidate, attr)
            except Exception:
                continue
            if child is not None:
                stack.append(child)


def _raise_if_provider_deadline_expired(deadline: _ProviderDeadline) -> None:
    if perf_counter() >= deadline.expires_at:
        raise _provider_timeout_error(deadline)


def _provider_timeout_error(deadline: _ProviderDeadline) -> TimeoutError:
    return TimeoutError(
        f"Provider request timed out after {deadline.timeout:g} seconds"
    )


def _iter_sse_data(
    handle: Any,
    *,
    deadline: _ProviderDeadline | None = None,
) -> Iterator[str]:
    data_lines: list[str] = []
    while True:
        if deadline is not None:
            _apply_remaining_provider_timeout(handle, deadline)
        raw_line = handle.readline()
        if not raw_line:
            break
        line = bytes(raw_line).decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            if deadline is not None:
                _raise_if_provider_deadline_expired(deadline)
            continue
        field, separator, value = line.partition(":")
        if field != "data":
            if deadline is not None:
                _raise_if_provider_deadline_expired(deadline)
            continue
        data_lines.append(value[1:] if separator and value.startswith(" ") else value)
        if deadline is not None:
            _raise_if_provider_deadline_expired(deadline)
    if data_lines:
        yield "\n".join(data_lines)


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_NO_REDIRECT_OPENER = build_opener(_NoRedirectHandler)


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
