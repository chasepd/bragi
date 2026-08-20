from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
from collections.abc import AsyncIterator
from email.message import Message
from urllib.error import HTTPError

import pytest

from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.http_client import (
    JsonHttpResponse,
    _iter_sse_data,
    _safe_started_diagnostics,
    dispatch_transport,
    ensure_success,
    httpx_request_bytes,
    httpx_request_json,
    httpx_request_sse_json,
    request_bytes,
    request_json,
    request_sse_json,
)


def test_ensure_success_returns_payload_for_success_response() -> None:
    payload = {"data": [{"id": "model-1"}]}

    assert ensure_success(JsonHttpResponse(status_code=200, payload=payload)) == payload


def test_iter_sse_data_ignores_comments_and_collects_multiline_data() -> None:
    stream = io.BytesIO(
        b": keepalive\n"
        b"event: chunk\n"
        b"data: {\"choices\":[{\"delta\":{\"content\":\"The\"}}]}\n"
        b"\n"
        b"data: line one\n"
        b"data: line two\n"
        b"\n"
        b"data: [DONE]\n"
        b"\n"
    )

    assert list(_iter_sse_data(stream)) == [
        '{"choices":[{"delta":{"content":"The"}}]}',
        "line one\nline two",
        "[DONE]",
    ]


def test_request_sse_json_enforces_total_timeout_without_stream_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingStreamResponse:
        status = 200

        def __init__(self) -> None:
            self.headers: Message[str, str] = Message()
            self.closed = False
            self.close_requested = threading.Event()

        def __enter__(self) -> BlockingStreamResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def __iter__(self) -> BlockingStreamResponse:
            return self

        def __next__(self) -> bytes:
            self.close_requested.wait(timeout=2.0)
            if self.closed:
                raise StopIteration
            return b""

        def readline(self) -> bytes:
            self.close_requested.wait(timeout=2.0)
            return b""

        def close(self) -> None:
            self.closed = True
            self.close_requested.set()

    response = BlockingStreamResponse()

    class FakeOpener:
        def open(self, *_args: object, **_kwargs: object) -> object:
            return response

    async def collect_stream() -> None:
        async for _event in request_sse_json(
            method="POST",
            url="https://provider.example/v1/chat/completions",
            headers={},
            payload={"stream": True},
            timeout=0.01,
            provider="venice",
            task="chat_completion",
        ):
            raise AssertionError("stream should not emit an event")

    monkeypatch.setattr("bragi.providers.http_client._NO_REDIRECT_OPENER", FakeOpener())

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(asyncio.wait_for(collect_stream(), timeout=0.5))

    assert exc_info.value.category == ProviderErrorCategory.NETWORK_ERROR
    assert "timed out" in exc_info.value.message
    assert response.closed is True


def test_ensure_success_raises_provider_error_with_payload_message() -> None:
    with pytest.raises(ProviderError) as exc_info:
        ensure_success(
            JsonHttpResponse(
                status_code=429,
                payload={"error": {"message": "slow down"}},
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.RATE_LIMITED
    assert exc_info.value.message == "rate_limited (429)"
    assert "slow down" not in exc_info.value.message


def test_request_json_success_reads_response_with_explicit_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundedBody:
        def __init__(self) -> None:
            self.read_sizes: list[int | None] = []

        def read(self, size: int | None = None) -> bytes:
            self.read_sizes.append(size)
            if size is None or size < 0:
                raise AssertionError("provider success body read must be bounded")
            return b'{"ok": true}'

    class FakeResponse:
        status = 200

        def __init__(self, body: BoundedBody) -> None:
            self.body = body
            self.headers: Message[str, str] = Message()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int | None = None) -> bytes:
            return self.body.read(size)

    body = BoundedBody()

    class FakeOpener:
        def open(self, *_args: object, **_kwargs: object) -> object:
            return FakeResponse(body)

    monkeypatch.setattr("bragi.providers.http_client._NO_REDIRECT_OPENER", FakeOpener())

    response = request_json(
        method="GET",
        url="https://provider.example/api",
        headers={},
        payload=None,
        timeout=1.0,
    )

    assert response.payload == {"ok": True}
    assert body.read_sizes
    assert all(size is not None and size > 0 for size in body.read_sizes)


def test_request_json_success_captures_only_safe_response_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.headers: Message[str, str] = Message()
            self.headers["Content-Type"] = "video/mp4"
            self.headers["X-Venice-Is-Content-Violation"] = "true"
            self.headers["X-Request-ID"] = "req-123"
            self.headers["Set-Cookie"] = "session=secret"

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int | None = None) -> bytes:
            if size is None or size < 0:
                raise AssertionError("provider success body read must be bounded")
            return b'{"ok": true}'

    class FakeOpener:
        def open(self, *_args: object, **_kwargs: object) -> object:
            return FakeResponse()

    monkeypatch.setattr("bragi.providers.http_client._NO_REDIRECT_OPENER", FakeOpener())

    response = request_json(
        method="GET",
        url="https://provider.example/api",
        headers={},
        payload=None,
        timeout=1.0,
    )

    assert response.headers == {
        "content-type": "video/mp4",
        "x-venice-is-content-violation": "true",
        "x-request-id": "req-123",
    }


def test_request_json_enforces_total_timeout_while_reading_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0

    def fake_perf_counter() -> float:
        return now

    class SlowBody:
        def __init__(self) -> None:
            self.chunks = [b'{"ok": ', b"true}"]
            self.read_sizes: list[int | None] = []

        def read(self, size: int | None = None) -> bytes:
            nonlocal now
            self.read_sizes.append(size)
            now += 0.6
            return self.chunks.pop(0) if self.chunks else b""

    class FakeResponse:
        status = 200

        def __init__(self, body: SlowBody) -> None:
            self.body = body
            self.headers: Message[str, str] = Message()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int | None = None) -> bytes:
            return self.body.read(size)

    body = SlowBody()

    class FakeOpener:
        def open(self, *_args: object, **kwargs: object) -> object:
            assert kwargs["timeout"] == 1.0
            return FakeResponse(body)

    monkeypatch.setattr(
        "bragi.providers.http_client.perf_counter",
        fake_perf_counter,
    )
    monkeypatch.setattr(
        "bragi.providers.http_client.PROVIDER_RESPONSE_READ_CHUNK_BYTES",
        len(body.chunks[0]),
    )
    monkeypatch.setattr("bragi.providers.http_client._NO_REDIRECT_OPENER", FakeOpener())

    with pytest.raises(ProviderError) as exc_info:
        request_json(
            method="GET",
            url="https://provider.example/api",
            headers={},
            payload=None,
            timeout=1.0,
        )

    assert exc_info.value.category == ProviderErrorCategory.NETWORK_ERROR
    assert "timed out" in exc_info.value.message
    assert len(body.read_sizes) > 1
    assert all(size is not None and size > 0 for size in body.read_sizes)


def test_request_json_logs_safe_started_event_before_opening_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    payload = {
        "model": "payload-model-secret",
        "messages": [
            {
                "role": "system",
                "content": "body-secret [memory:memory-1]",
            }
        ],
    }

    def capture_log_event(event_name: str, **fields: object) -> None:
        events.append((event_name, fields))

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.headers: Message[str, str] = Message()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int | None = None) -> bytes:
            if size is None or size < 0:
                raise AssertionError("provider success body read must be bounded")
            return b'{"ok": true}'

    class FakeOpener:
        def open(self, *_args: object, **_kwargs: object) -> object:
            assert events
            assert events[0][0] == "provider.http_started"
            return FakeResponse()

    monkeypatch.setattr("bragi.providers.http_client.log_event", capture_log_event)
    monkeypatch.setattr("bragi.providers.http_client._NO_REDIRECT_OPENER", FakeOpener())

    request_json(
        method="POST",
        url="https://provider.example/v1/chat/completions?api_key=query-secret",
        headers={"Authorization": "Bearer header-secret"},
        payload=payload,
        timeout=3.5,
        provider="openrouter",
        task="chat_completion",
        model="safe-model-id",
    )

    assert events
    event_name, fields = events[0]
    assert event_name == "provider.http_started"
    assert fields == {
        "method": "POST",
        "path": "/v1/chat/completions",
        "timeout": 3.5,
        "payload_bytes": len(json.dumps(payload).encode("utf-8")),
        "payload_sha256": hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "provider": "openrouter",
        "task": "chat_completion",
        "model": "safe-model-id",
        "message_count": 1,
        "message_content_chars": len("body-secret [memory:memory-1]"),
        "source_id_count": 1,
        "source_id_sha256": [
            hashlib.sha256(b"memory:memory-1").hexdigest(),
        ],
    }
    assert "header-secret" not in repr(fields)
    assert "body-secret" not in repr(fields)
    assert "payload-model-secret" not in repr(fields)
    assert "query-secret" not in repr(fields)
    assert "Authorization" not in repr(fields)
    assert "messages" not in repr(fields)


def test_safe_started_diagnostics_bounds_trusted_source_ids() -> None:
    diagnostics = _safe_started_diagnostics(
        provider="openrouter",
        task="chat",
        model="safe-model",
        schema_name=None,
        payload={
            "messages": [
                {
                    "role": "user",
                    "content": "[secret:account-password]",
                },
                *(
                    {
                        "role": "system",
                        "content": (
                            f"[memory:memory-{index}] safe marker "
                            f"[memory:{'x' * 200}]"
                        ),
                    }
                    for index in range(10_000)
                ),
            ]
        },
    )

    source_id_hashes = diagnostics["source_id_sha256"]
    assert diagnostics["source_id_count"] == 64
    assert isinstance(source_id_hashes, list)
    assert len(source_id_hashes) == 64
    assert source_id_hashes[0] == hashlib.sha256(b"memory:memory-0").hexdigest()
    assert "secret:account-password" not in repr(diagnostics)
    assert all(len(source_id_hash) == 64 for source_id_hash in source_id_hashes)


def test_ensure_success_includes_safe_response_headers_in_payload() -> None:
    payload = {"choices": [{"message": {"content": "prose"}}]}

    result = ensure_success(
        JsonHttpResponse(
            status_code=200,
            payload=payload,
            headers={"x-venice-is-content-violation": "true"},
        )
    )

    assert result == {
        "choices": [{"message": {"content": "prose"}}],
        "_bragi_headers": {"x-venice-is-content-violation": "true"},
    }
    assert payload == {"choices": [{"message": {"content": "prose"}}]}


def test_request_json_http_error_with_non_json_body_uses_safe_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpener:
        def open(self, *_args: object, **_kwargs: object) -> object:
            headers: Message[str, str] = Message()
            raise HTTPError(
                url="https://provider.example/api",
                code=500,
                msg="Internal Server Error",
                hdrs=headers,
                fp=io.BytesIO(b"<html>provider stack trace</html>"),
            )

    monkeypatch.setattr("bragi.providers.http_client._NO_REDIRECT_OPENER", FakeOpener())

    with pytest.raises(ProviderError) as exc_info:
        request_json(
            method="GET",
            url="https://provider.example/api",
            headers={},
            payload=None,
            timeout=1.0,
        )

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR
    assert exc_info.value.message == "provider_error (500)"
    assert "<html>" not in exc_info.value.message
    assert "stack trace" not in exc_info.value.message


def test_request_json_http_error_skips_body_read_and_redacts_provider_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundedErrorBody(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b'{"error": {"message": "raw provider quota details"}}')
            self.read_sizes: list[int | None] = []

        def read(self, size: int | None = None) -> bytes:
            self.read_sizes.append(size)
            return super().read(-1 if size is None else size)

    error_body = BoundedErrorBody()

    class FakeOpener:
        def open(self, *_args: object, **_kwargs: object) -> object:
            headers: Message[str, str] = Message()
            raise HTTPError(
                url="https://provider.example/api",
                code=429,
                msg="Too Many Requests",
                hdrs=headers,
                fp=error_body,
            )

    monkeypatch.setattr("bragi.providers.http_client._NO_REDIRECT_OPENER", FakeOpener())

    with pytest.raises(ProviderError) as exc_info:
        request_json(
            method="GET",
            url="https://provider.example/api",
            headers={},
            payload=None,
            timeout=1.0,
        )

    assert exc_info.value.category == ProviderErrorCategory.RATE_LIMITED
    assert exc_info.value.message == "rate_limited (429)"
    assert "raw provider quota details" not in exc_info.value.message
    assert error_body.read_sizes == []


def test_request_json_suppresses_venice_http_error_body_and_keeps_safe_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def capture_error_event(event_name: str, **fields: object) -> None:
        events.append((event_name, fields))

    class BoundedErrorBody(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(
                b'{"error":{"message":"blocked private prompt for '
                b'api_key=venice-secret"}}'
            )
            self.read_sizes: list[int | None] = []

        def read(self, size: int | None = None) -> bytes:
            self.read_sizes.append(size)
            return super().read(size)

    error_body = BoundedErrorBody()

    class FakeOpener:
        def open(self, *_args: object, **_kwargs: object) -> object:
            headers: Message[str, str] = Message()
            headers["X-Request-ID"] = "req-venice"
            headers["X-Venice-Is-Content-Violation"] = "true"
            headers["Set-Cookie"] = "session=secret-cookie"
            raise HTTPError(
                url="https://api.venice.ai/api/v1/chat/completions",
                code=400,
                msg="Bad Request",
                hdrs=headers,
                fp=error_body,
            )

    monkeypatch.setattr(
        "bragi.providers.http_client.log_error_event",
        capture_error_event,
    )
    monkeypatch.setattr("bragi.providers.http_client._NO_REDIRECT_OPENER", FakeOpener())

    with pytest.raises(ProviderError):
        request_json(
            method="POST",
            url="https://api.venice.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer request-secret"},
            payload={"model": "venice-model"},
            timeout=1.0,
            provider="venice",
            task="chat",
            model="venice-model",
        )

    assert error_body.read_sizes == []
    event_name, fields = events[-1]
    assert event_name == "provider.http_failed"
    assert fields["provider"] == "venice"
    assert fields["task"] == "chat"
    assert fields["model"] == "venice-model"
    assert "response_body" not in fields
    assert fields["response_body_suppressed"] is True
    assert (
        fields["response_body_suppressed_reason"]
        == "provider_error_body_may_contain_private_content"
    )
    assert fields["response_headers"] == {
        "x-request-id": "req-venice",
        "x-venice-is-content-violation": "true",
    }
    assert "secret-cookie" not in repr(fields)
    assert "request-secret" not in repr(fields)
    assert "private prompt" not in repr(fields)
    assert "venice-secret" not in repr(fields)


def test_request_binary_suppresses_venice_http_error_body_and_keeps_safe_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def capture_error_event(event_name: str, **fields: object) -> None:
        events.append((event_name, fields))

    class BoundedErrorBody(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(
                b'{"error":{"message":"blocked private image prompt for '
                b'api_key=venice-secret"}}'
            )
            self.read_sizes: list[int | None] = []

        def read(self, size: int | None = None) -> bytes:
            self.read_sizes.append(size)
            return super().read(size)

    error_body = BoundedErrorBody()

    class FakeOpener:
        def open(self, *_args: object, **_kwargs: object) -> object:
            headers: Message[str, str] = Message()
            headers["X-Request-ID"] = "req-venice-binary"
            headers["Set-Cookie"] = "session=secret-cookie"
            raise HTTPError(
                url="https://api.venice.ai/api/v1/image/generate",
                code=422,
                msg="Unprocessable Entity",
                hdrs=headers,
                fp=error_body,
            )

    monkeypatch.setattr(
        "bragi.providers.http_client.log_error_event",
        capture_error_event,
    )
    monkeypatch.setattr("bragi.providers.http_client._NO_REDIRECT_OPENER", FakeOpener())

    with pytest.raises(ProviderError):
        request_bytes(
            method="POST",
            url="https://api.venice.ai/api/v1/image/generate",
            headers={"Authorization": "Bearer request-secret"},
            payload={"model": "venice-image-model"},
            timeout=1.0,
            provider="venice",
            task="image_generation",
            model="venice-image-model",
        )

    assert error_body.read_sizes == []
    event_name, fields = events[-1]
    assert event_name == "provider.http_failed"
    assert fields["provider"] == "venice"
    assert fields["task"] == "image_generation"
    assert fields["model"] == "venice-image-model"
    assert "response_body" not in fields
    assert fields["response_body_suppressed"] is True
    assert fields["response_headers"] == {"x-request-id": "req-venice-binary"}
    assert "secret-cookie" not in repr(fields)
    assert "request-secret" not in repr(fields)
    assert "private image prompt" not in repr(fields)
    assert "venice-secret" not in repr(fields)


def test_httpx_request_json_parses_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeStreamResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        chunks=(b'{"ok": true, "items": 3}',)
    )
    client = _FakeStreamClient(response)
    monkeypatch.setattr(
        "bragi.providers.http_client._get_httpx_client",
        lambda **_: client,
    )

    result = asyncio.run(
        httpx_request_json(
            method="POST",
            url="https://api.example.test/chat",
            headers={"Authorization": "Bearer token"},
            payload={"model": "fake-chat"},
            timeout=30.0,
            provider="fake",
            task="chat",
            model="fake-chat",
            schema_name="narrator",
        )
    )

    assert result.status_code == 200
    assert result.payload == {"ok": True, "items": 3}
    assert result.headers == {"content-type": "application/json"}
    assert client.requests[0]["method"] == "POST"
    assert client.requests[0]["url"] == "https://api.example.test/chat"
    assert client.requests[0]["content"] == b'{"model": "fake-chat"}'


def test_httpx_request_json_raises_on_non_success_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeStreamResponse(
        status_code=429,
        headers={"retry-after": "2"},
    )
    monkeypatch.setattr(
        "bragi.providers.http_client._get_httpx_client",
        lambda **kwargs: _FakeStreamClient(response),
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            httpx_request_json(
                method="POST",
                url="https://api.example.test/chat",
                headers={},
                payload=None,
                timeout=30.0,
                provider="fake",
                task="chat",
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.RATE_LIMITED
    assert exc_info.value.status_code == 429


def test_httpx_request_bytes_raises_when_response_exceeds_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeStreamResponse(
        status_code=200,
        chunks=(b"0123456789", b"abcdef"),
    )
    monkeypatch.setattr(
        "bragi.providers.http_client._get_httpx_client",
        lambda **kwargs: _FakeStreamClient(response),
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            httpx_request_bytes(
                method="GET",
                url="https://api.example.test/image",
                headers={},
                payload=None,
                timeout=30.0,
                max_response_bytes=4,
                provider="fake",
                task="image_generation",
            )
        )

    assert exc_info.value.category == ProviderErrorCategory.PROVIDER_ERROR


async def _collect_stream(
    stream: AsyncIterator[dict[str, object]],
) -> list[dict[str, object]]:
    return [event async for event in stream]


class _FakeStreamResponse:
    def __init__(
        self,
        *,
        status_code: int,
        headers: dict[str, str] | None = None,
        chunks: tuple[bytes, ...] = (),
        lines: tuple[str, ...] = (),
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *args: object) -> None:
        return None


class _FakeStreamClient:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response
        self.requests: list[dict[str, object]] = []

    def stream(self, **kwargs: object) -> _FakeStreamContext:
        self.requests.append(kwargs)
        return _FakeStreamContext(self._response)


def test_httpx_request_sse_json_yields_parsed_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeStreamResponse(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        lines=(
            ": keepalive",
            "",
            'data: {"choices":[{"delta":{"content":"The"}}]}',
            "",
            "data: [DONE]",
            "",
        ),
    )
    monkeypatch.setattr(
        "bragi.providers.http_client._get_httpx_client",
        lambda **kwargs: _FakeStreamClient(response),
    )

    events = asyncio.run(
        _collect_stream(
            httpx_request_sse_json(
                method="POST",
                url="https://api.example.test/chat",
                headers={},
                payload={"stream": True},
                timeout=30.0,
            )
        )
    )

    assert events == [{"choices": [{"delta": {"content": "The"}}]}]


def test_dispatch_transport_runs_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def async_transport(**kwargs: object) -> str:
        return f"async-{kwargs['value']}"

    def sync_transport(**kwargs: object) -> str:
        return f"sync-{kwargs['value']}"

    async def run() -> tuple[str, str]:
        async_result = await dispatch_transport(async_transport, value=1)
        sync_result = await dispatch_transport(sync_transport, value=2)
        return async_result, sync_result

    assert asyncio.run(run()) == ("async-1", "sync-2")
