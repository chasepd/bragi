"""OpenRouter provider client helpers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import mimetypes
from collections.abc import AsyncIterator, Awaitable, Callable
from decimal import Decimal, InvalidOperation
from os import environ
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from bragi.app_logging import log_error_event
from bragi.providers.chat_rendering import provider_chat_message, provider_chat_messages
from bragi.providers.contracts import (
    ChatReasoningConfig,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ImageDescriptionRequest,
    ImageDescriptionResponse,
    ImageRequest,
    ImageResponse,
    ProviderCatalogEntry,
    ProviderConfigStatus,
    ProviderModel,
    ProviderModelListResponse,
    ProviderModelPricing,
    ProviderRetryProgressCallback,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
    VideoRequest,
    VideoResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.http import THINKING_LEVELS, normalize_model_record
from bragi.providers.http_client import (
    BinaryHttpResponse,
    BinaryHttpTransport,
    JsonHttpTransport,
    dispatch_transport,
    ensure_binary_success,
    ensure_success,
    httpx_request_bytes,
    httpx_request_json,
    httpx_request_sse_json,
    request_bytes,
    request_json,
    request_sse_json,
)
from bragi.providers.reasoning_diagnostics import (
    extract_reasoning_signals,
    is_reasoning_truncated_structured_response,
)
from bragi.providers.retry import (
    call_with_provider_retries,
    retry_metadata_from_provider_error,
    stream_with_provider_deadline,
)
from bragi.providers.structured_output_validation import (
    StructuredOutputValidationError,
    validate_structured_output,
)
from bragi.providers.tool_calls import (
    parse_tool_call_response,
    tool_definition_payload,
    tool_message_payload,
)
from bragi.retry_policy import (
    DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS,
    PROVIDER_MAX_ATTEMPTS,
)
from bragi.services.secrets import SecretStorageError, SecretStore

OPENROUTER_PROVIDER_NAME = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_APP_URL = "https://github.com/chasepd/bragi"
OPENROUTER_MODEL_LIST_PATH = "/models?output_modalities=all"
OPENROUTER_PROVIDER_LIST_PATH = "/providers"
OPENROUTER_METADATA_HEADER = "X-OpenRouter-Metadata"
MAX_PROVIDER_IMAGE_BYTES = 25 * 1024 * 1024
OPENROUTER_VIDEO_POLL_INTERVAL_SECONDS = 30.0
OPENROUTER_VIDEO_TIMEOUT_SECONDS = 15 * 60.0
_STRUCTURED_DATA_METADATA_KEY = "_bragi_structured_data"
OPENROUTER_DEFAULT_APP_TITLE = "Bragi"
OPENROUTER_IMAGE_ASPECT_RATIOS: tuple[str, ...] = (
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
)
OPENROUTER_KNOWN_PRICING_KEYS = frozenset(
    {
        "prompt",
        "completion",
        "request",
        "image",
        "input_cache_read",
        "input_cache_write",
    }
)


class OpenRouterClient:
    provider_name = OPENROUTER_PROVIDER_NAME

    def __init__(
        self,
        *,
        secret_store: SecretStore,
        base_url: str = OPENROUTER_BASE_URL,
        transport: JsonHttpTransport = request_json,
        binary_transport: BinaryHttpTransport = request_bytes,
        stream_transport: Any = request_sse_json,
        timeout: float = 60.0,
        video_poll_interval: float = OPENROUTER_VIDEO_POLL_INTERVAL_SECONDS,
        video_timeout: float = OPENROUTER_VIDEO_TIMEOUT_SECONDS,
        retry_max_attempts: Callable[[], int] | None = None,
        call_deadline_seconds: Callable[[], float] | None = None,
    ) -> None:
        self.secret_store = secret_store
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.binary_transport = binary_transport
        self.stream_transport = stream_transport
        self.timeout = timeout
        self.video_poll_interval = max(0.0, video_poll_interval)
        self.video_timeout = max(0.0, video_timeout)
        self.retry_max_attempts = retry_max_attempts
        self.call_deadline_seconds = call_deadline_seconds
        self._model_output_modalities: dict[str, tuple[str, ...]] = {}

    def _configured_max_attempts(self) -> int:
        if self.retry_max_attempts is None:
            return PROVIDER_MAX_ATTEMPTS
        return max(1, int(self.retry_max_attempts()))

    def _configured_call_deadline_seconds(self) -> float:
        if self.call_deadline_seconds is None:
            return DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS
        return max(0.0, float(self.call_deadline_seconds()))

    async def validate_config(self) -> ProviderConfigStatus:
        try:
            api_key = self.secret_store.get_api_key(self.provider_name)
        except SecretStorageError:
            return ProviderConfigStatus(
                provider=self.provider_name,
                configured=True,
                authenticated=False,
                error=ProviderErrorCategory.SECRET_STORAGE_ERROR.value,
            )
        if not api_key:
            return ProviderConfigStatus(
                provider=self.provider_name,
                configured=False,
                authenticated=False,
                error=ProviderErrorCategory.PROVIDER_NOT_CONFIGURED.value,
            )
        try:
            payload = await self._get_json(path=OPENROUTER_MODEL_LIST_PATH)
            self._remember_output_modalities(payload)
        except ProviderError as exc:
            return ProviderConfigStatus(
                provider=self.provider_name,
                configured=True,
                authenticated=False,
                error=exc.category.value,
                diagnostics=retry_metadata_from_provider_error(exc),
            )
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
            diagnostics=payload,
        )

    async def list_models(self) -> list[ProviderModel]:
        return (await self.list_models_with_metadata()).models

    async def list_models_with_metadata(self) -> ProviderModelListResponse:
        payload = await self._get_json(path=OPENROUTER_MODEL_LIST_PATH)
        models = normalize_openrouter_models(payload)
        self._model_output_modalities = _output_modalities_by_model(payload)
        return ProviderModelListResponse(models=models, raw_metadata=payload)

    async def list_providers(self) -> list[ProviderCatalogEntry]:
        payload = await self._get_json(path=OPENROUTER_PROVIDER_LIST_PATH)
        return normalize_openrouter_providers(payload)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": provider_chat_messages(request),
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.reasoning is not None:
            reasoning = _reasoning_payload(request.reasoning)
            if reasoning:
                payload["reasoning"] = reasoning
        _apply_provider_routing(payload, request.openrouter_provider_routing)
        response = await self._post_json(
            path="/chat/completions",
            payload=payload,
            task="chat",
            app_title=request.openrouter_app_title,
            retry_progress_callback=request.retry_progress_callback,
        )
        return ChatResponse(
            body=_parse_chat_content(response),
            provider=self.provider_name,
            model_id=str(response.get("model") or request.model_id),
            token_usage=_usage(response),
            raw_metadata=response,
            raw_request_payload=payload,
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": provider_chat_messages(request),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.reasoning is not None:
            reasoning = _reasoning_payload(request.reasoning)
            if reasoning:
                payload["reasoning"] = reasoning
        _apply_provider_routing(payload, request.openrouter_provider_routing)
        async for event in self._post_stream(
            path="/chat/completions",
            payload=payload,
            task="chat",
            app_title=request.openrouter_app_title,
        ):
            yield _parse_chat_stream_chunk(event)

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": _image_generation_message_content(request),
                }
            ],
            "modalities": await self._image_modalities(
                request.model_id,
                app_title=request.openrouter_app_title,
            ),
            "stream": False,
        }
        image_config = _image_config(request.dimensions)
        if image_config:
            payload["image_config"] = image_config
        _apply_provider_routing(payload, request.openrouter_provider_routing)
        response = await self._post_json(
            path="/chat/completions",
            payload=payload,
            task="image_generation",
            app_title=request.openrouter_app_title,
            retry_progress_callback=request.retry_progress_callback,
        )
        return ImageResponse(
            provider=self.provider_name,
            model_id=str(response.get("model") or request.model_id),
            image_bytes=_parse_image_bytes(response),
            raw_metadata=response,
        )

    def image_reference_limit(self, model_id: str) -> int:
        return 4

    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        submit_response = await self._post_json(
            path="/videos",
            payload=_video_generation_payload(request),
            task="video_generation",
            app_title=request.openrouter_app_title,
        )
        job_id = _video_job_id(submit_response)
        poll_response, poll_count = await self._poll_video_job(
            job_id=job_id,
            app_title=request.openrouter_app_title,
        )
        content_response = await self._get_bytes(
            path=f"/videos/{job_id}/content?index=0",
            task="video_generation",
            app_title=request.openrouter_app_title,
        )
        video_bytes = ensure_binary_success(content_response)
        mime_type = _video_content_type(content_response)
        return VideoResponse(
            provider=self.provider_name,
            model_id=_video_response_model(
                request=request,
                submit_response=submit_response,
                poll_response=poll_response,
            ),
            mime_type=mime_type,
            video_bytes=video_bytes,
            raw_metadata=_video_raw_metadata(
                submit_response=submit_response,
                poll_response=poll_response,
                content_response=content_response,
                poll_count=poll_count,
            ),
        )

    async def _poll_video_job(
        self,
        *,
        job_id: str,
        app_title: str | None = None,
    ) -> tuple[dict[str, Any], int]:
        started_at = perf_counter()
        poll_count = 0
        while True:
            response = await self._get_json(
                path=f"/videos/{job_id}",
                app_title=app_title,
            )
            poll_count += 1
            status = _video_status(response)
            if status == "completed":
                return response, poll_count
            if status in {"failed", "cancelled", "canceled", "expired"}:
                raise _video_terminal_error(response)
            if status not in {"pending", "queued", "in_progress", "processing"}:
                raise ProviderError(
                    category=ProviderErrorCategory.PROVIDER_ERROR,
                    message=(
                        "OpenRouter video generation returned unsupported status: "
                        f"{status or '<missing>'}"
                    ),
                )
            if perf_counter() - started_at >= self.video_timeout:
                raise ProviderError(
                    category=ProviderErrorCategory.PROVIDER_ERROR,
                    message=f"OpenRouter video generation timed out: {job_id}",
                )
            await asyncio.sleep(self.video_poll_interval)

    async def describe_image(
        self,
        request: ImageDescriptionRequest,
    ) -> ImageDescriptionResponse:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": _image_description_messages(request),
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.reasoning is not None:
            reasoning = _reasoning_payload(request.reasoning)
            if reasoning:
                payload["reasoning"] = reasoning
        _apply_provider_routing(payload, request.openrouter_provider_routing)
        response = await self._post_json(
            path="/chat/completions",
            payload=payload,
            task="image_description",
            app_title=request.openrouter_app_title,
        )
        return ImageDescriptionResponse(
            description=_parse_chat_content(response).strip(),
            provider=self.provider_name,
            model_id=str(response.get("model") or request.model_id),
            token_usage=_usage(response),
            raw_metadata=response,
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": [
                provider_chat_message(message) for message in request.messages
            ],
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": True,
                    "schema": request.schema,
                },
            },
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.reasoning is not None:
            reasoning = _reasoning_payload(request.reasoning)
            if reasoning:
                payload["reasoning"] = reasoning
        _apply_provider_routing(payload, request.openrouter_provider_routing)
        response = await call_with_provider_retries(
            lambda: self._request_structured_output(
                payload,
                app_title=request.openrouter_app_title,
            ),
            provider=self.provider_name,
            task="structured_output",
            max_attempts=self._configured_max_attempts(),
            call_deadline_seconds=self._configured_call_deadline_seconds(),
            retry_progress_callback=request.retry_progress_callback,
        )
        raw_metadata = dict(response)
        data = raw_metadata.pop(_STRUCTURED_DATA_METADATA_KEY)
        _raise_if_reasoning_truncated(
            response=response,
            data=data,
            schema_name=request.schema_name,
        )
        try:
            validate_structured_output(
                data,
                schema=request.schema,
                schema_name=request.schema_name,
            )
        except StructuredOutputValidationError as exc:
            raise ProviderError(
                category=ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID,
                message=str(exc),
                diagnostics=exc.diagnostics,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(
                category=ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID,
                message="Structured provider response violated its JSON Schema",
                diagnostics={
                    "schema_name": request.schema_name,
                    "error_count": 1,
                    "errors": [
                        {
                            "schema_path": "$.type",
                            "validator": "type",
                            "message": "Value does not satisfy schema constraint",
                        }
                    ],
                },
            )
        return StructuredOutputResponse(
            data=data,
            provider=self.provider_name,
            model_id=str(response.get("model") or request.model_id),
            token_usage=_usage(response),
            raw_metadata=raw_metadata,
        )

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": [tool_message_payload(message) for message in request.messages],
            "stream": False,
            "tools": [tool_definition_payload(tool) for tool in request.tools],
            "tool_choice": "auto",
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.reasoning is not None:
            reasoning = _reasoning_payload(request.reasoning)
            if reasoning:
                payload["reasoning"] = reasoning
        _apply_provider_routing(payload, request.openrouter_provider_routing)
        response = await self._post_json(
            path="/chat/completions",
            payload=payload,
            task="tool_calling",
            app_title=request.openrouter_app_title,
        )
        body, tool_calls = parse_tool_call_response(response)
        return ToolCallResponse(
            tool_calls=tool_calls,
            body=body,
            provider=self.provider_name,
            model_id=str(response.get("model") or request.model_id),
            token_usage=_usage(response),
            raw_metadata=response,
            raw_request_payload=payload,
        )

    async def _request_structured_output(
        self,
        payload: dict[str, Any],
        *,
        app_title: str | None = None,
    ) -> dict[str, Any]:
        response = await self._request_json(
            method="POST",
            path="/chat/completions",
            payload=payload,
            timeout=self.timeout,
            task="structured_output",
            app_title=app_title,
        )
        return {
            **response,
            _STRUCTURED_DATA_METADATA_KEY: _parse_structured_content(response),
        }

    async def _get_json(
        self,
        *,
        path: str,
        app_title: str | None = None,
    ) -> dict[str, Any]:
        return await call_with_provider_retries(
            lambda: self._request_json(
                method="GET",
                path=path,
                payload=None,
                timeout=self.timeout,
                task="model_listing",
                app_title=app_title,
            ),
            provider=self.provider_name,
            task="model_listing",
            max_attempts=self._configured_max_attempts(),
            call_deadline_seconds=self._configured_call_deadline_seconds(),
        )

    async def _post_json(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        task: str,
        app_title: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> dict[str, Any]:
        return await call_with_provider_retries(
            lambda: self._request_json(
                method="POST",
                path=path,
                payload=payload,
                timeout=self.timeout,
                task=task,
                app_title=app_title,
            ),
            provider=self.provider_name,
            task=task,
            max_attempts=self._configured_max_attempts(),
            call_deadline_seconds=self._configured_call_deadline_seconds(),
            retry_progress_callback=retry_progress_callback,
        )

    def _json_transport_kwargs(
        self,
        payload: dict[str, Any] | None,
        *,
        task: str,
    ) -> dict[str, Any]:
        if self.transport is request_json or self.transport is httpx_request_json:
            return {
                "provider": self.provider_name,
                "task": task,
                "model": _payload_model(payload),
                "schema_name": _schema_name(payload),
            }
        return {}

    def _bytes_transport_kwargs(
        self,
        payload: dict[str, Any] | None,
        *,
        task: str,
    ) -> dict[str, Any]:
        if (
            self.binary_transport is request_bytes
            or self.binary_transport is httpx_request_bytes
        ):
            return {
                "provider": self.provider_name,
                "task": task,
                "model": _payload_model(payload),
            }
        return {}

    def _stream_transport_kwargs(
        self,
        payload: dict[str, Any],
        *,
        task: str,
    ) -> dict[str, Any]:
        if (
            self.stream_transport is request_sse_json
            or self.stream_transport is httpx_request_sse_json
        ):
            return {
                "provider": self.provider_name,
                "task": task,
                "model": _payload_model(payload),
                "schema_name": _schema_name(payload),
            }
        return {}

    async def _post_stream(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        task: str,
        app_title: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        deadline_seconds = self._configured_call_deadline_seconds()
        stream = self.stream_transport(
            method="POST",
            url=f"{self.base_url}{path}",
            headers=self._headers(app_title=app_title),
            payload=payload,
            timeout=min(self.timeout, deadline_seconds),
            **self._stream_transport_kwargs(payload, task=task),
        )
        async for event in stream_with_provider_deadline(
            stream,
            provider=self.provider_name,
            task=task,
            call_deadline_seconds=deadline_seconds,
        ):
            yield event

    async def _get_bytes(
        self,
        *,
        path: str,
        task: str,
        app_title: str | None = None,
    ) -> BinaryHttpResponse:
        return await call_with_provider_retries(
            lambda: self._request_success_bytes(
                method="GET",
                path=path,
                payload=None,
                timeout=self.video_timeout,
                task=task,
                app_title=app_title,
            ),
            provider=self.provider_name,
            task=task,
            max_attempts=self._configured_max_attempts(),
            call_deadline_seconds=self._configured_call_deadline_seconds(),
        )

    async def _request_json(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float,
        task: str,
        app_title: str | None = None,
    ) -> dict[str, Any]:
        response = await _await_provider_transport(
            dispatch_transport(
                self.transport,
                method=method,
                url=f"{self.base_url}{path}",
                headers=self._headers(app_title=app_title),
                payload=payload,
                timeout=timeout,
                **self._json_transport_kwargs(payload, task=task),
            ),
            timeout=timeout,
            method=method,
            path=path,
            provider=self.provider_name,
            task=task,
            model=_payload_model(payload),
            schema_name=_schema_name(payload),
        )
        return ensure_success(response)

    async def _request_success_bytes(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float,
        task: str,
        app_title: str | None = None,
    ) -> BinaryHttpResponse:
        response = await self._request_bytes(
            method=method,
            path=path,
            payload=payload,
            timeout=timeout,
            task=task,
            app_title=app_title,
        )
        ensure_binary_success(response)
        return response

    async def _request_bytes(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float,
        task: str,
        app_title: str | None = None,
    ) -> BinaryHttpResponse:
        return await _await_provider_transport(
            dispatch_transport(
                self.binary_transport,
                method=method,
                url=f"{self.base_url}{path}",
                headers=self._headers(app_title=app_title),
                payload=payload,
                timeout=timeout,
                **self._bytes_transport_kwargs(payload, task=task),
            ),
            timeout=timeout,
            method=method,
            path=path,
            provider=self.provider_name,
            task=task,
            model=_payload_model(payload),
        )

    def _headers(self, *, app_title: str | None = None) -> dict[str, str]:
        try:
            api_key = self.secret_store.get_api_key(self.provider_name)
        except SecretStorageError as exc:
            raise ProviderError(
                category=ProviderErrorCategory.SECRET_STORAGE_ERROR,
                message=str(exc),
            ) from exc
        if not api_key:
            raise ProviderError(
                category=ProviderErrorCategory.PROVIDER_NOT_CONFIGURED,
                message="OpenRouter API key is not configured",
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            OPENROUTER_METADATA_HEADER: "enabled",
        }
        headers["X-OpenRouter-Title"] = OPENROUTER_DEFAULT_APP_TITLE
        headers["HTTP-Referer"] = _openrouter_app_url()
        return headers

    async def _image_modalities(
        self,
        model_id: str,
        *,
        app_title: str | None = None,
    ) -> list[str]:
        if model_id not in self._model_output_modalities:
            self._remember_output_modalities(
                await self._get_json(
                    path=OPENROUTER_MODEL_LIST_PATH,
                    app_title=app_title,
                )
            )

        output_modalities = self._model_output_modalities.get(model_id)
        if output_modalities is None:
            raise ProviderError(
                category=ProviderErrorCategory.MODEL_NOT_FOUND,
                message=(
                    "OpenRouter model metadata is unavailable for image "
                    f"generation model: {model_id}"
                ),
            )
        if "image" not in output_modalities:
            raise ProviderError(
                category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
                message=f"OpenRouter model does not advertise image output: {model_id}",
            )
        if "image" in output_modalities and "text" in output_modalities:
            return ["image", "text"]
        return ["image"]

    def _remember_output_modalities(self, payload: dict[str, Any]) -> None:
        self._model_output_modalities.update(_output_modalities_by_model(payload))


def _openrouter_app_url() -> str:
    value = environ.get("BRAGI_OPENROUTER_APP_URL")
    if value is None:
        return OPENROUTER_DEFAULT_APP_URL
    text = value.strip()
    if not text or "\r" in text or "\n" in text:
        return OPENROUTER_DEFAULT_APP_URL
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return OPENROUTER_DEFAULT_APP_URL
    return text


def _apply_provider_routing(
    payload: dict[str, Any],
    provider_routing: dict[str, Any] | None,
) -> None:
    if provider_routing:
        payload["provider"] = dict(provider_routing)


def _video_generation_payload(request: VideoRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model_id,
        "prompt": request.prompt,
    }
    if request.source_media_path is not None:
        payload["frame_images"] = [
            {
                "type": "image_url",
                "image_url": {"url": _local_image_data_url(request.source_media_path)},
                "frame_type": "first_frame",
            }
        ]
    return payload


def _video_job_id(payload: dict[str, Any]) -> str:
    job_id = payload.get("id")
    if isinstance(job_id, str) and job_id.strip():
        return job_id.strip()
    raise ProviderError(
        category=ProviderErrorCategory.PROVIDER_ERROR,
        message="OpenRouter video response did not include a job id",
    )


def _video_status(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    return status.strip().lower() if isinstance(status, str) else ""


def _video_terminal_error(payload: dict[str, Any]) -> ProviderError:
    status = _video_status(payload)
    message = _video_error_message(payload)
    return ProviderError(
        category=(
            ProviderErrorCategory.CONTENT_BLOCKED
            if _video_error_indicates_content_block(message)
            else ProviderErrorCategory.PROVIDER_ERROR
        ),
        message=f"OpenRouter video generation {status}: {message}",
    )


def _video_error_message(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return "unknown error"


def _video_error_indicates_content_block(message: str) -> bool:
    normalized = message.strip().casefold()
    return any(
        marker in normalized
        for marker in (
            "blocked",
            "content policy",
            "content violation",
            "moderation",
            "policy violation",
            "prohibited",
            "refusal",
            "safety",
        )
    )


def _video_response_model(
    *,
    request: VideoRequest,
    submit_response: dict[str, Any],
    poll_response: dict[str, Any],
) -> str:
    for payload in (poll_response, submit_response):
        model = payload.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return request.model_id


def _video_content_type(response: BinaryHttpResponse) -> str:
    raw_content_type = response.headers.get("content-type", "")
    mime_type = raw_content_type.split(";", 1)[0].strip().lower()
    if mime_type in {"video/mp4", "video/webm"}:
        return mime_type
    raise ProviderError(
        category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
        message=(
            "OpenRouter video download did not include a supported video "
            "content type"
        ),
    )


def _video_raw_metadata(
    *,
    submit_response: dict[str, Any],
    poll_response: dict[str, Any],
    content_response: BinaryHttpResponse,
    poll_count: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "job_id": _video_job_id(submit_response),
        "status": _video_status(poll_response),
        "poll_count": poll_count,
    }
    generation_id = poll_response.get("generation_id") or submit_response.get(
        "generation_id"
    )
    if isinstance(generation_id, str) and generation_id.strip():
        metadata["generation_id"] = generation_id.strip()
    usage = poll_response.get("usage")
    if isinstance(usage, dict):
        metadata["usage"] = dict(usage)
    if content_response.headers:
        metadata["_bragi_headers"] = dict(content_response.headers)
    submit_retry = submit_response.get("_bragi_retry")
    poll_retry = poll_response.get("_bragi_retry")
    if isinstance(submit_retry, dict):
        metadata["submit_retry"] = dict(submit_retry)
    if isinstance(poll_retry, dict):
        metadata["poll_retry"] = dict(poll_retry)
        metadata["_bragi_retry"] = dict(poll_retry)
    elif isinstance(submit_retry, dict):
        metadata["_bragi_retry"] = dict(submit_retry)
    return metadata


def _payload_model(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    model = payload.get("model")
    return model if isinstance(model, str) else None


async def _await_provider_transport[T](
    operation: Awaitable[T],
    *,
    timeout: float,
    method: str,
    path: str,
    provider: str,
    task: str,
    model: str | None = None,
    schema_name: str | None = None,
) -> T:
    started_at = perf_counter()
    normalized_timeout = max(0.0, timeout)
    try:
        return await asyncio.wait_for(operation, timeout=normalized_timeout)
    except TimeoutError as exc:
        message = _provider_timeout_message(normalized_timeout)
        log_fields: dict[str, object] = {
            "method": method,
            "path": path,
            "duration_ms": int((perf_counter() - started_at) * 1000),
            "error_category": ProviderErrorCategory.NETWORK_ERROR.value,
            "error": message,
            "provider": provider,
            "task": task,
        }
        if model:
            log_fields["model"] = model
        if schema_name:
            log_fields["schema_name"] = schema_name
        log_error_event("provider.http_failed", **log_fields)
        raise ProviderError(
            category=ProviderErrorCategory.NETWORK_ERROR,
            message=message,
        ) from exc


def _provider_timeout_message(timeout: float) -> str:
    return f"Provider request timed out after {timeout:g} seconds"


def _schema_name(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return None
    name = json_schema.get("name")
    return name if isinstance(name, str) else None


def normalize_openrouter_models(payload: dict[str, Any]) -> list[ProviderModel]:
    records = payload.get("data", [])
    if not isinstance(records, list):
        return []
    return [
        normalize_model_record(
            provider=OPENROUTER_PROVIDER_NAME,
            payload=_with_openrouter_capabilities(dict(record)),
            default_to_chat=False,
        )
        for record in records
        if isinstance(record, dict)
    ]


def normalize_openrouter_providers(
    payload: dict[str, Any],
) -> list[ProviderCatalogEntry]:
    records = payload.get("data", [])
    if not isinstance(records, list):
        return []
    providers: list[ProviderCatalogEntry] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        slug = _catalog_text(record.get("slug"))
        if slug is None:
            continue
        normalized_slug = slug.casefold()
        if normalized_slug in seen:
            continue
        seen.add(normalized_slug)
        providers.append(
            ProviderCatalogEntry(
                slug=normalized_slug,
                name=_catalog_text(record.get("name")) or normalized_slug,
                privacy_policy_url=_catalog_text(record.get("privacy_policy_url")),
                terms_of_service_url=_catalog_text(
                    record.get("terms_of_service_url")
                ),
                status_page_url=_catalog_text(record.get("status_page_url")),
                headquarters=_catalog_text(record.get("headquarters")),
                datacenters=tuple(_catalog_text_list(record.get("datacenters"))),
            )
        )
    return providers


def _catalog_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _catalog_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _catalog_text(item)
        if text is None or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return values


def _with_openrouter_capabilities(record: dict[str, Any]) -> dict[str, Any]:
    input_modalities = _openrouter_input_modalities(record)
    output_modalities = _openrouter_output_modalities(record)
    capabilities = _openrouter_capability_hints(input_modalities, output_modalities)
    if not output_modalities and not _declares_openrouter_output_modalities(record):
        capabilities.append("chat")
    if (
        ("chat" in capabilities or "image_generation" in capabilities)
        and _is_unmoderated_openrouter_model(record)
        and _has_blocked_output_fallback_openrouter_marker(record)
    ):
        capabilities.append("blocked_output_fallback")
    if "chat" in capabilities and _supports_structured_output(record):
        capabilities.append("structured_output")
    if "chat" in capabilities and _supports_tool_calling(record):
        capabilities.append("tool_calling")
    record["capabilities"] = capabilities
    record["supported_parameters"] = _openrouter_supported_parameters(
        record,
        capabilities=capabilities,
    )
    record["pricing"] = _openrouter_model_pricing(record.get("pricing"))
    record["thinking"] = _openrouter_thinking_level_support(record)
    return record


def _openrouter_model_pricing(raw_pricing: Any) -> ProviderModelPricing | None:
    if not isinstance(raw_pricing, dict):
        return None
    pricing = ProviderModelPricing(
        input_per_million_tokens_usd=_decimal_price(
            raw_pricing.get("prompt"),
            multiplier=Decimal("1000000"),
        ),
        output_per_million_tokens_usd=_decimal_price(
            raw_pricing.get("completion"),
            multiplier=Decimal("1000000"),
        ),
        cache_read_per_million_tokens_usd=_decimal_price(
            raw_pricing.get("input_cache_read"),
            multiplier=Decimal("1000000"),
        ),
        cache_write_per_million_tokens_usd=_decimal_price(
            raw_pricing.get("input_cache_write"),
            multiplier=Decimal("1000000"),
        ),
        request_usd=_decimal_price(raw_pricing.get("request")),
        image_usd=_decimal_price(raw_pricing.get("image")),
        note=_unknown_openrouter_pricing_note(raw_pricing),
    )
    return pricing if _pricing_has_value(pricing) else None


def _unknown_openrouter_pricing_note(raw_pricing: dict[str, Any]) -> str | None:
    unknown_keys = sorted(
        key
        for key, value in raw_pricing.items()
        if key not in OPENROUTER_KNOWN_PRICING_KEYS and _is_positive_decimal(value)
    )
    if not unknown_keys:
        return None
    return f"Additional pricing fields: {', '.join(unknown_keys)}"


def _is_positive_decimal(value: Any) -> bool:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return False
    return decimal.is_finite() and decimal > 0


def _decimal_price(value: Any, *, multiplier: Decimal = Decimal("1")) -> str | None:
    if value is None:
        return None
    try:
        decimal = Decimal(str(value)) * multiplier
    except (InvalidOperation, ValueError):
        return None
    if not decimal.is_finite() or decimal < 0:
        return None
    return format(decimal.normalize(), "f")


def _pricing_has_value(pricing: ProviderModelPricing) -> bool:
    return any(
        value
        for value in (
            pricing.input_per_million_tokens_usd,
            pricing.output_per_million_tokens_usd,
            pricing.cache_read_per_million_tokens_usd,
            pricing.cache_write_per_million_tokens_usd,
            pricing.request_usd,
            pricing.image_usd,
            pricing.note,
        )
    )


def _openrouter_supported_parameters(
    record: dict[str, Any],
    *,
    capabilities: list[str],
) -> list[str]:
    raw_parameters = record.get("supported_parameters")
    parameters = (
        [str(parameter) for parameter in raw_parameters]
        if isinstance(raw_parameters, list)
        else []
    )
    if "image_generation" in capabilities or "image_to_image" in capabilities:
        parameters.append("image_dimensions")
    return _dedupe_strings(parameters)


def _openrouter_thinking_level_support(
    record: dict[str, Any],
) -> dict[str, object] | None:
    supported_parameters = record.get("supported_parameters")
    normalized_parameters = (
        {
            str(parameter).strip().casefold().replace("-", "_")
            for parameter in supported_parameters
        }
        if isinstance(supported_parameters, list)
        else set()
    )
    reasoning = record.get("reasoning")
    if not isinstance(reasoning, dict) and not (
        normalized_parameters & {"reasoning", "reasoning_effort"}
    ):
        return None
    if isinstance(reasoning, dict):
        raw_efforts = reasoning.get("supported_efforts")
        if isinstance(raw_efforts, list):
            levels = [
                effort
                for effort in _dedupe_strings(
                    [
                        str(item).strip().casefold().replace("-", "_")
                        for item in raw_efforts
                    ]
                )
                if effort in THINKING_LEVELS
            ]
        elif raw_efforts is None:
            levels = list(THINKING_LEVELS)
        else:
            levels = []
        return {
            "levels": levels or list(THINKING_LEVELS),
            "default_level": reasoning.get("default_effort"),
            "default_enabled": reasoning.get("default_enabled"),
            "mandatory": reasoning.get("mandatory") is True,
            "supports_max_tokens": reasoning.get("supports_max_tokens") is True,
        }
    return {
        "levels": list(THINKING_LEVELS),
        "default_level": None,
        "default_enabled": None,
        "mandatory": False,
        "supports_max_tokens": False,
    }


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _openrouter_capability_hints(
    input_modalities: tuple[str, ...],
    output_modalities: tuple[str, ...],
) -> list[str]:
    if any(
        modality not in {"text", "image", "video"} for modality in output_modalities
    ):
        return []
    if "video" in output_modalities and not input_modalities:
        return []

    capabilities: list[str] = []
    text_input_supported = not input_modalities or "text" in input_modalities
    if "text" in output_modalities and text_input_supported:
        capabilities.append("chat")
    if "image" in output_modalities:
        capabilities.append("image_generation")
    if "image" in input_modalities and "image" in output_modalities:
        capabilities.append("image_to_image")
    if "video" in output_modalities and "text" in input_modalities:
        capabilities.append("text_to_video")
    if "video" in output_modalities and "image" in input_modalities:
        capabilities.append("image_to_video")
        if "text" in input_modalities:
            capabilities.append("image_plus_text_to_video")
    if "image" in input_modalities and "text" in output_modalities:
        capabilities.append("vision")
    return capabilities


def _image_config(dimensions: tuple[int, int] | None) -> dict[str, str]:
    if dimensions is None:
        return {}
    width, height = dimensions
    if width <= 0 or height <= 0:
        return {}
    return {
        "aspect_ratio": _aspect_ratio(dimensions),
        "image_size": _image_size(dimensions),
    }


def _aspect_ratio(dimensions: tuple[int, int]) -> str:
    width, height = dimensions
    target_ratio = width / height
    return min(
        OPENROUTER_IMAGE_ASPECT_RATIOS,
        key=lambda ratio: abs(_ratio_value(ratio) - target_ratio),
    )


def _ratio_value(ratio: str) -> float:
    width, height = ratio.split(":", 1)
    return int(width) / int(height)


def _image_size(dimensions: tuple[int, int]) -> str:
    longest_side = max(dimensions)
    if longest_side <= 1024:
        return "1K"
    if longest_side <= 2048:
        return "2K"
    return "4K"


def _image_generation_message_content(
    request: ImageRequest,
) -> str | list[dict[str, object]]:
    source_paths = _source_image_paths(request)
    if not source_paths:
        return request.prompt
    content: list[dict[str, object]] = [
        {"type": "text", "text": request.prompt},
    ]
    content.extend(
        [
            {
                "type": "image_url",
                "image_url": {"url": _local_image_data_url(source_path)},
            }
            for source_path in source_paths
        ]
    )
    return content


def _source_image_paths(request: ImageRequest) -> tuple[Path, ...]:
    if request.source_media_paths:
        return request.source_media_paths
    if request.source_media_path is not None:
        return (request.source_media_path,)
    return ()


def _local_image_data_url(path: Any) -> str:
    source = Path(path)
    if not source.is_file():
        raise ProviderError(
            category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
            message=f"Source image is unavailable: {source}",
        )
    if source.stat().st_size > MAX_PROVIDER_IMAGE_BYTES:
        raise ProviderError(
            category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
            message=f"Source image exceeded {MAX_PROVIDER_IMAGE_BYTES} bytes",
        )
    mime_type = mimetypes.guess_type(source.name)[0] or "image/png"
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _output_modalities_by_model(
    payload: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    records = payload.get("data", [])
    if not isinstance(records, list):
        return {}

    modalities_by_model: dict[str, tuple[str, ...]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        model_id = record.get("id") or record.get("model") or record.get("name")
        if model_id is None:
            continue
        output_modalities = _openrouter_output_modalities(record)
        if output_modalities:
            modalities_by_model[str(model_id)] = output_modalities
    return modalities_by_model


def _openrouter_input_modalities(record: dict[str, Any]) -> tuple[str, ...]:
    architecture = record.get("architecture")
    if not isinstance(architecture, dict):
        return ()
    raw_input = architecture.get("input_modalities", [])
    if not isinstance(raw_input, list):
        return ()

    input_modalities: list[str] = []
    for item in raw_input:
        normalized = str(item).strip().lower()
        if normalized and normalized not in input_modalities:
            input_modalities.append(normalized)
    return tuple(input_modalities)


def _openrouter_output_modalities(record: dict[str, Any]) -> tuple[str, ...]:
    architecture = record.get("architecture")
    if not isinstance(architecture, dict):
        return ()
    raw_output = architecture.get("output_modalities", [])
    if not isinstance(raw_output, list):
        return ()

    output_modalities: list[str] = []
    for item in raw_output:
        normalized = str(item).strip().lower()
        if normalized and normalized not in output_modalities:
            output_modalities.append(normalized)
    return tuple(output_modalities)


def _declares_openrouter_output_modalities(record: dict[str, Any]) -> bool:
    architecture = record.get("architecture")
    return isinstance(architecture, dict) and isinstance(
        architecture.get("output_modalities"),
        list,
    )


def _is_unmoderated_openrouter_model(record: dict[str, Any]) -> bool:
    top_provider = record.get("top_provider")
    return isinstance(top_provider, dict) and top_provider.get("is_moderated") is False


def _has_blocked_output_fallback_openrouter_marker(
    record: dict[str, Any],
) -> bool:
    values = [
        value
        for key in ("id", "name", "description")
        if isinstance(value := record.get(key), str)
    ]
    return any(
        _is_blocked_output_fallback_openrouter_marker(value) for value in values
    )


def _is_blocked_output_fallback_openrouter_marker(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return any(
        marker in normalized for marker in ("uncensored", "unmoderated", "unfiltered")
    )


def _supports_structured_output(record: dict[str, Any]) -> bool:
    supported_parameters = record.get("supported_parameters")
    if not isinstance(supported_parameters, list):
        return False
    normalized = {str(parameter).lower() for parameter in supported_parameters}
    return bool(
        normalized
        & {
            "response_format",
            "json_schema",
            "structured_output",
            "structured_outputs",
        }
    )


def _supports_tool_calling(record: dict[str, Any]) -> bool:
    supported_parameters = record.get("supported_parameters")
    if not isinstance(supported_parameters, list):
        return False
    normalized = {str(parameter).lower() for parameter in supported_parameters}
    return bool(
        normalized
        & {
            "tools",
            "tool_choice",
            "tool_calling",
            "function_calling",
        }
    )


def _image_description_messages(
    request: ImageDescriptionRequest,
) -> list[dict[str, object]]:
    system_prompt = request.system_prompt or (
        "Describe the visible physical appearance in the image as "
        "natural prose for consistent future image generation. Include "
        "stable visible details such as skin tone, complexion or "
        "undertone, face shape and features, hair color and texture, "
        "eyes, clothing or styling, posture, and overall presence. When "
        "the text prompt does not specify a visible detail, infer that "
        "detail from the image. Use explicitly provided character context "
        "when it supplies or clearly establishes race, ethnicity, "
        "ancestry, or cultural appearance details, and do not omit those "
        "details from the appearance description."
    )
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": request.prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": request.image_url},
                },
            ],
        },
    ]


def _chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Provider response did not include chat choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Provider chat choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Provider chat choice did not include a message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        diagnostics = _reasoning_only_diagnostics(
            payload=payload,
            choice=first,
            message=message,
        )
        if diagnostics is not None:
            raise ProviderError(
                category=ProviderErrorCategory.PROVIDER_ERROR,
                message=(
                    "Provider returned a reasoning-only response with no visible "
                    "assistant text; increase max_output_tokens or disable/reduce "
                    "reasoning for this model"
                ),
                diagnostics=diagnostics,
            )
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "".join(text_parts)
    raise ValueError("Provider chat response content must be a string")


def _parse_chat_content(payload: dict[str, Any]) -> str:
    try:
        return _chat_content(payload)
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message=str(exc),
        ) from exc


def _reasoning_payload(config: ChatReasoningConfig) -> dict[str, object]:
    payload: dict[str, object] = {}
    if config.enabled is not None:
        payload["enabled"] = config.enabled
    if config.effort is not None:
        payload["effort"] = config.effort
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    if config.exclude is not None:
        payload["exclude"] = config.exclude
    return payload


def _reasoning_only_diagnostics(
    *,
    payload: dict[str, Any],
    choice: dict[str, Any],
    message: dict[str, Any],
) -> dict[str, object] | None:
    finish_reason = choice.get("finish_reason")
    if finish_reason != "length":
        return None
    reasoning_tokens = _reasoning_tokens(payload)
    detail_types = _reasoning_detail_types(message.get("reasoning_details"))
    has_reasoning_field = message.get("reasoning") is not None
    if reasoning_tokens is None and not detail_types and not has_reasoning_field:
        return None

    diagnostics: dict[str, object] = {"finish_reason": finish_reason}
    native_finish_reason = choice.get("native_finish_reason")
    if isinstance(native_finish_reason, str):
        diagnostics["native_finish_reason"] = native_finish_reason
    if reasoning_tokens is not None:
        diagnostics["reasoning_tokens"] = reasoning_tokens
    if detail_types:
        diagnostics["reasoning_detail_types"] = detail_types
    return diagnostics


def _reasoning_tokens(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(key)
        if not isinstance(details, dict):
            continue
        tokens = details.get("reasoning_tokens")
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            return tokens
    return None


def _reasoning_detail_types(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    detail_types: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        detail_type = item.get("type")
        if isinstance(detail_type, str) and detail_type:
            detail_types.append(detail_type)
    return detail_types


def _raise_if_reasoning_truncated(
    *,
    response: dict[str, Any],
    data: Any,
    schema_name: str,
) -> None:
    if not isinstance(data, dict):
        return
    if data:
        return
    signals = extract_reasoning_signals(response)
    if not is_reasoning_truncated_structured_response(signals):
        return
    raise ProviderError(
        category=ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID,
        message=(
            "Structured response was truncated; reasoning consumed the output "
            "budget before any visible JSON was emitted. Increase "
            "max_output_tokens or disable reasoning for this model."
        ),
        diagnostics={
            "schema_name": schema_name,
            "finish_reason": signals.finish_reason,
            "reasoning_tokens": signals.reasoning_tokens,
            "reasoning_detail_types": list(signals.detail_types),
            "completion_tokens": signals.completion_tokens,
            "truncated": True,
        },
    )


def _parse_chat_stream_chunk(payload: dict[str, Any]) -> ChatStreamChunk:
    try:
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            raise ValueError(str(message) if message else "Provider stream error")
        choices = payload.get("choices")
        delta = ""
        if isinstance(choices, list) and choices:
            first = choices[0]
            if not isinstance(first, dict):
                raise ValueError("Provider stream choice must be an object")
            delta = _stream_delta_text(first)
        elif choices not in (None, []):
            raise ValueError("Provider stream choices must be a list")
        usage = _usage(payload)
        return ChatStreamChunk(
            delta=delta,
            token_usage=usage,
            raw_metadata=payload,
            done=bool(usage and not delta),
        )
    except Exception as exc:
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message=str(exc),
        ) from exc


def _stream_delta_text(choice: dict[str, Any]) -> str:
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)
    return ""


def _parse_structured_content(payload: dict[str, Any]) -> Any:
    try:
        content = _chat_content(payload)
        loaded = json.loads(content)
        return loaded
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message=str(exc),
        ) from exc


def _usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    normalized = {key: value for key, value in usage.items() if isinstance(value, int)}
    if "total_tokens" in normalized:
        normalized["total"] = normalized["total_tokens"]
    return normalized


def _openrouter_image_bytes(payload: dict[str, Any]) -> bytes:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Provider response did not include image choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Provider image choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Provider image choice did not include a message")
    images = message.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Provider image response did not include images")
    first_image = images[0]
    if not isinstance(first_image, dict):
        raise ValueError("Provider image entry must be an object")
    image_url = first_image.get("image_url")
    if not isinstance(image_url, dict):
        raise ValueError("Provider image entry did not include image_url")
    url = image_url.get("url")
    if not isinstance(url, str):
        raise ValueError("Provider image URL must be a string")
    return _decode_data_url(url)


def _decode_data_url(value: str) -> bytes:
    if "," in value and value.startswith("data:"):
        value = value.split(",", 1)[1]
    compact = "".join(value.split())
    if _maximum_decoded_base64_bytes(compact) > MAX_PROVIDER_IMAGE_BYTES:
        raise ValueError(
            f"Provider image payload exceeded {MAX_PROVIDER_IMAGE_BYTES} bytes"
        )
    decoded = base64.b64decode(compact, validate=True)
    if len(decoded) > MAX_PROVIDER_IMAGE_BYTES:
        raise ValueError(
            f"Provider image payload exceeded {MAX_PROVIDER_IMAGE_BYTES} bytes"
        )
    return decoded


def _maximum_decoded_base64_bytes(value: str) -> int:
    if not value:
        return 0
    padding = len(value) - len(value.rstrip("="))
    return ((len(value) + 3) // 4) * 3 - min(padding, 2)


def _parse_image_bytes(payload: dict[str, Any]) -> bytes:
    try:
        return _openrouter_image_bytes(payload)
    except (binascii.Error, ValueError) as exc:
        raise ProviderError(
            category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
            message=str(exc),
        ) from exc
