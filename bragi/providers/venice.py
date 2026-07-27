"""Venice.ai provider client helpers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import json
import mimetypes
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import ParseResult, urlparse

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
from bragi.providers.http import normalize_model_record
from bragi.providers.http_client import (
    SAFE_PROVIDER_RESPONSE_HEADERS,
    BinaryHttpResponse,
    BinaryHttpTransport,
    JsonHttpTransport,
    ensure_binary_success,
    ensure_success,
    request_bytes,
    request_json,
    request_sse_json,
)
from bragi.providers.retry import (
    call_with_provider_retries,
    retry_metadata_from_provider_error,
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
from bragi.redaction import redact_text
from bragi.services.secrets import SecretStorageError, SecretStore

VENICE_PROVIDER_NAME = "venice"
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VENICE_MODEL_LIST_PATH = "/models?type=all"
VENICE_IMAGE_PROMPT_MAX_CHARS = 7500
VENICE_VIDEO_PROMPT_MAX_CHARS = 2500
VENICE_IMAGE_TIMEOUT_SECONDS = 180.0
VENICE_VIDEO_POLL_INTERVAL_SECONDS = 5.0
VENICE_VIDEO_TIMEOUT_SECONDS = 15 * 60.0
VENICE_VIDEO_TIMEOUT_BUFFER_SECONDS = 5 * 60.0
VENICE_VIDEO_MAX_TIMEOUT_SECONDS = 60 * 60.0
VENICE_PRIVATE_VIDEO_DOWNLOAD_HOST = "private-share.venice.ai"
VENICE_PRIVATE_VIDEO_DOWNLOAD_PATH_PREFIX = "/v1/share/read/"
MAX_PROVIDER_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_VENICE_VALIDATION_SUMMARY_CHARS = 1000
_STRUCTURED_DATA_METADATA_KEY = "_bragi_structured_data"
VENICE_PIXEL_DIMENSION_MODEL_IDS = frozenset(
    {
        "hidream",
        "qwen-image",
        "venice-sd35",
    }
)


@dataclass(frozen=True)
class _VeniceVideoContent:
    response: BinaryHttpResponse
    status: str
    poll_count: int
    retrieve_retry: dict[str, Any] | None = None
    download_retry: dict[str, Any] | None = None


class VeniceClient:
    provider_name = VENICE_PROVIDER_NAME

    def __init__(
        self,
        *,
        secret_store: SecretStore,
        base_url: str = VENICE_BASE_URL,
        transport: JsonHttpTransport = request_json,
        binary_transport: BinaryHttpTransport = request_bytes,
        stream_transport: Any = request_sse_json,
        timeout: float = 60.0,
        image_timeout: float = VENICE_IMAGE_TIMEOUT_SECONDS,
        video_poll_interval: float = VENICE_VIDEO_POLL_INTERVAL_SECONDS,
        video_timeout: float = VENICE_VIDEO_TIMEOUT_SECONDS,
    ) -> None:
        self.secret_store = secret_store
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.binary_transport = binary_transport
        self.stream_transport = stream_transport
        self.timeout = timeout
        self.image_timeout = image_timeout
        self.video_poll_interval = max(0.0, video_poll_interval)
        self.video_timeout = max(0.0, video_timeout)

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
            payload = await self._get_json(path=VENICE_MODEL_LIST_PATH)
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
        payload = await self._get_json(path=VENICE_MODEL_LIST_PATH)
        return ProviderModelListResponse(
            models=normalize_venice_models(payload),
            raw_metadata=payload,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": provider_chat_messages(request),
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_completion_tokens"] = request.max_output_tokens
        reasoning_effort = _reasoning_effort(request.reasoning)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        response = await self._post_json(
            path="/chat/completions",
            payload=payload,
            task="chat",
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
            payload["max_completion_tokens"] = request.max_output_tokens
        reasoning_effort = _reasoning_effort(request.reasoning)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        async for event in self._post_stream(
            path="/chat/completions",
            payload=payload,
            task="chat",
        ):
            yield _parse_chat_stream_chunk(event)

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        if _source_image_paths(request):
            return await self._edit_image(request)
        payload: dict[str, Any] = {
            "model": request.model_id,
            "prompt": _compact_image_prompt(request.prompt),
            "format": "png",
            "return_binary": False,
            "safe_mode": _image_safe_mode(request),
        }
        payload.update(_image_sizing_payload(request.model_id, request.dimensions))
        response = await self._post_json(
            path="/image/generate",
            payload=payload,
            timeout=self.image_timeout,
            task="image_generation",
            retry_progress_callback=request.retry_progress_callback,
        )
        return ImageResponse(
            provider=self.provider_name,
            model_id=request.model_id,
            image_bytes=_parse_image_bytes(response),
            raw_metadata=response,
        )

    def image_reference_limit(self, model_id: str) -> int:
        return 3

    async def _edit_image(self, request: ImageRequest) -> ImageResponse:
        source_paths = _source_image_paths(request)
        source_asset_ids = _source_image_asset_ids(request)
        if len(source_paths) > 1 or request.safe_mode is False:
            multi_edit_payload: dict[str, Any] = {
                "modelId": request.model_id,
                "prompt": _compact_image_prompt(request.prompt),
                "images": [_source_image_base64(path) for path in source_paths[:3]],
                "output_format": "png",
                "safe_mode": _image_safe_mode(request),
            }
            multi_edit_payload.update(_image_edit_sizing_payload(request.dimensions))
            response = await self._post_bytes(
                path="/image/multi-edit",
                payload=multi_edit_payload,
                timeout=self.image_timeout,
                task="image_generation",
                retry_progress_callback=request.retry_progress_callback,
            )
            return ImageResponse(
                provider=self.provider_name,
                model_id=request.model_id,
                image_bytes=ensure_binary_success(response),
                raw_metadata={
                    "_bragi_headers": response.headers,
                    "byte_count": len(response.body),
                    "source_media_asset_id": (
                        source_asset_ids[0] if source_asset_ids else ""
                    ),
                    "source_media_asset_ids": list(source_asset_ids[:3]),
                },
            )
        edit_payload: dict[str, Any] = {
            "model": request.model_id,
            "prompt": _compact_image_prompt(request.prompt),
            "image": _source_image_base64(source_paths[0]),
            "output_format": "png",
            "safe_mode": _image_safe_mode(request),
        }
        edit_payload.update(_image_edit_sizing_payload(request.dimensions))
        response = await self._post_bytes(
            path="/image/edit",
            payload=edit_payload,
            timeout=self.image_timeout,
            task="image_generation",
            retry_progress_callback=request.retry_progress_callback,
        )
        return ImageResponse(
            provider=self.provider_name,
            model_id=request.model_id,
            image_bytes=ensure_binary_success(response),
            raw_metadata={
                "_bragi_headers": response.headers,
                "byte_count": len(response.body),
                "source_media_asset_id": (
                    source_asset_ids[0] if source_asset_ids else ""
                ),
                "source_media_asset_ids": list(source_asset_ids[:1]),
            },
        )

    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        queue_response = await self._post_json(
            path="/video/queue",
            payload=_video_generation_payload(request),
            timeout=self.video_timeout,
            task="video_generation",
        )
        queue_id = _venice_video_queue_id(queue_response)
        content = await self._poll_video(
            request=request,
            queue_response=queue_response,
            queue_id=queue_id,
        )
        video_bytes = ensure_binary_success(content.response)
        mime_type = _venice_video_content_type(content.response)
        return VideoResponse(
            provider=self.provider_name,
            model_id=_venice_video_response_model(
                request=request,
                queue_response=queue_response,
            ),
            mime_type=mime_type,
            video_bytes=video_bytes,
            raw_metadata=_venice_video_raw_metadata(
                queue_response=queue_response,
                content=content,
            ),
        )

    async def _poll_video(
        self,
        *,
        request: VideoRequest,
        queue_response: dict[str, Any],
        queue_id: str,
    ) -> _VeniceVideoContent:
        started_at = perf_counter()
        poll_count = 0
        retrieve_retry: dict[str, Any] | None = None
        effective_timeout = self.video_timeout
        retrieve_payload = {
            "model": _venice_video_response_model(
                request=request,
                queue_response=queue_response,
            ),
            "queue_id": queue_id,
        }
        while True:
            response, retry = await self._post_bytes_with_metadata(
                path="/video/retrieve",
                payload=retrieve_payload,
                timeout=self.video_timeout,
                task="video_generation",
            )
            poll_count += 1
            if retry is not None:
                retrieve_retry = retry
            content_type = _binary_content_type(response)
            if content_type == "video/mp4":
                return _VeniceVideoContent(
                    response=response,
                    status="COMPLETED",
                    poll_count=poll_count,
                    retrieve_retry=retrieve_retry,
                )
            if content_type != "application/json":
                _venice_video_content_type(response)
            payload = _venice_video_json_response(response)
            status = _venice_video_status(payload)
            effective_timeout = _venice_video_effective_timeout(
                payload,
                current_timeout=effective_timeout,
            )
            if status == "COMPLETED":
                download_url = _venice_video_download_url(queue_response)
                if not download_url:
                    raise ProviderError(
                        category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
                        message=(
                            "Venice video generation completed without video/mp4 "
                            "content or a private download URL"
                        ),
                    )
                download_response, download_retry = await self._get_bytes_with_metadata(
                    path=download_url,
                    timeout=self.video_timeout,
                    task="video_generation",
                    include_auth=False,
                    allowed_absolute_hosts=frozenset(
                        {VENICE_PRIVATE_VIDEO_DOWNLOAD_HOST}
                    ),
                )
                _venice_video_content_type(download_response)
                return _VeniceVideoContent(
                    response=download_response,
                    status=status,
                    poll_count=poll_count,
                    retrieve_retry=retrieve_retry,
                    download_retry=download_retry,
                )
            if status in {
                "FAILED",
                "ERROR",
                "CANCELED",
                "CANCELLED",
                "EXPIRED",
                "BLOCKED",
                "CONTENT_BLOCKED",
                "MODERATION_FAILED",
                "REJECTED",
            }:
                raise _venice_video_terminal_error(payload)
            if status not in {"PROCESSING", "PENDING", "QUEUED", "RUNNING"}:
                raise ProviderError(
                    category=ProviderErrorCategory.PROVIDER_ERROR,
                    message=(
                        "Venice video generation returned unsupported status: "
                        f"{status or '<missing>'}"
                    ),
                )
            if perf_counter() - started_at >= effective_timeout:
                raise ProviderError(
                    category=ProviderErrorCategory.PROVIDER_ERROR,
                    message=f"Venice video generation timed out: {queue_id}",
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
            payload["max_completion_tokens"] = request.max_output_tokens
        reasoning_effort = _reasoning_effort(request.reasoning)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        response = await self._post_json(
            path="/chat/completions",
            payload=payload,
            timeout=self.timeout,
            task="image_description",
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
            "messages": _structured_output_messages(request),
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
            payload["max_completion_tokens"] = request.max_output_tokens
        reasoning_effort = _reasoning_effort(request.reasoning)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        response = await call_with_provider_retries(
            lambda: self._request_structured_output(payload),
            provider=self.provider_name,
            task="structured_output",
        )
        raw_metadata = dict(response)
        data = raw_metadata.pop(_STRUCTURED_DATA_METADATA_KEY)
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
            "parallel_tool_calls": request.parallel_tool_calls,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_completion_tokens"] = request.max_output_tokens
        reasoning_effort = _reasoning_effort(request.reasoning)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        response = await self._post_json(
            path="/chat/completions",
            payload=payload,
            task="tool_calling",
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
    ) -> dict[str, Any]:
        response = await self._request_json(
            method="POST",
            path="/chat/completions",
            payload=payload,
            timeout=self.timeout,
            task="structured_output",
        )
        return {
            **response,
            _STRUCTURED_DATA_METADATA_KEY: _parse_structured_content(response),
        }

    async def _get_json(self, *, path: str) -> dict[str, Any]:
        return await call_with_provider_retries(
            lambda: self._request_json(
                method="GET",
                path=path,
                payload=None,
                timeout=self.timeout,
                task="model_listing",
            ),
            provider=self.provider_name,
            task="model_listing",
        )

    async def _post_json(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        task: str,
        timeout: float | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> dict[str, Any]:
        return await call_with_provider_retries(
            lambda: self._request_json(
                method="POST",
                path=path,
                payload=payload,
                timeout=self.timeout if timeout is None else timeout,
                task=task,
            ),
            provider=self.provider_name,
            task=task,
            retry_progress_callback=retry_progress_callback,
        )

    async def _post_stream(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        task: str,
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        transport_kwargs: dict[str, Any] = {}
        if self.stream_transport is request_sse_json:
            transport_kwargs = {
                "provider": self.provider_name,
                "task": task,
                "model": _payload_model(payload),
                "schema_name": _schema_name(payload),
            }
        async for event in self.stream_transport(
            method="POST",
            url=_provider_url(self.base_url, path),
            headers=self._headers(),
            payload=payload,
            timeout=self.timeout if timeout is None else timeout,
            **transport_kwargs,
        ):
            yield event

    async def _request_json(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float,
        task: str,
    ) -> dict[str, Any]:
        transport_kwargs: dict[str, Any] = {}
        if self.transport is request_json:
            transport_kwargs = {
                "provider": self.provider_name,
                "task": task,
                "model": _payload_model(payload),
                "schema_name": _schema_name(payload),
            }
        response = await _await_provider_transport(
            asyncio.to_thread(
                self.transport,
                method=method,
                url=_provider_url(self.base_url, path),
                headers=self._headers(),
                payload=payload,
                timeout=timeout,
                **transport_kwargs,
            ),
            timeout=timeout,
            method=method,
            path=path,
            provider=self.provider_name,
            task=task,
            model=_payload_model(payload),
            schema_name=_schema_name(payload),
        )
        try:
            return ensure_success(response)
        except ProviderError as exc:
            raise _venice_video_http_error(
                exc,
                task=task,
                payload=response.payload,
            ) from exc

    async def _post_bytes(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        task: str,
        timeout: float | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> BinaryHttpResponse:
        return await call_with_provider_retries(
            lambda: self._request_bytes(
                method="POST",
                path=path,
                payload=payload,
                timeout=timeout or self.timeout,
                task=task,
            ),
            provider=self.provider_name,
            task=task,
            retry_progress_callback=retry_progress_callback,
        )

    async def _post_bytes_with_metadata(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        task: str,
        timeout: float,
    ) -> tuple[BinaryHttpResponse, dict[str, Any] | None]:
        result = await call_with_provider_retries(
            lambda: self._request_success_bytes_result(
                method="POST",
                path=path,
                payload=payload,
                timeout=timeout,
                task=task,
            ),
            provider=self.provider_name,
            task=task,
        )
        return _bytes_result_response_and_retry(result)

    async def _get_bytes_with_metadata(
        self,
        *,
        path: str,
        task: str,
        timeout: float,
        include_auth: bool = True,
        allowed_absolute_hosts: frozenset[str] = frozenset(),
    ) -> tuple[BinaryHttpResponse, dict[str, Any] | None]:
        result = await call_with_provider_retries(
            lambda: self._request_success_bytes_result(
                method="GET",
                path=path,
                payload=None,
                timeout=timeout,
                task=task,
                include_auth=include_auth,
                allowed_absolute_hosts=allowed_absolute_hosts,
            ),
            provider=self.provider_name,
            task=task,
        )
        return _bytes_result_response_and_retry(result)

    async def _request_success_bytes_result(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float,
        task: str,
        include_auth: bool = True,
        allowed_absolute_hosts: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        response = await self._request_bytes(
            method=method,
            path=path,
            payload=payload,
            timeout=timeout,
            task=task,
            include_auth=include_auth,
            allowed_absolute_hosts=allowed_absolute_hosts,
        )
        try:
            ensure_binary_success(response)
        except ProviderError as exc:
            raise _venice_video_http_error(exc, task=task) from exc
        return {"_bragi_response": response}

    async def _request_bytes(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float,
        task: str,
        include_auth: bool = True,
        allowed_absolute_hosts: frozenset[str] = frozenset(),
    ) -> BinaryHttpResponse:
        transport_kwargs: dict[str, Any] = {}
        if self.binary_transport is request_bytes:
            transport_kwargs = {
                "provider": self.provider_name,
                "task": task,
                "model": _payload_model(payload),
            }
        return await _await_provider_transport(
            asyncio.to_thread(
                self.binary_transport,
                method=method,
                url=_provider_url(
                    self.base_url,
                    path,
                    allowed_absolute_hosts=allowed_absolute_hosts,
                ),
                headers=self._headers() if include_auth else {},
                payload=payload,
                timeout=timeout,
                **transport_kwargs,
            ),
            timeout=timeout,
            method=method,
            path=path,
            provider=self.provider_name,
            task=task,
            model=_payload_model(payload),
        )

    def _headers(self) -> dict[str, str]:
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
                message="Venice API key is not configured",
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


def _video_generation_payload(request: VideoRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model_id,
        "prompt": _video_prompt_for_payload(request),
        "duration": "5s",
    }
    if request.source_media_path is not None:
        source_image_url = _source_image_data_url(request.source_media_path)
        if _is_reference_to_video_model(request.model_id):
            payload["reference_image_urls"] = [source_image_url]
        else:
            payload["image_url"] = source_image_url
    if request.safe_mode is not None:
        payload["safe_mode"] = request.safe_mode
    return payload


def _image_safe_mode(request: ImageRequest) -> bool:
    return request.safe_mode if request.safe_mode is not None else True


def _source_image_paths(request: ImageRequest) -> tuple[Path, ...]:
    if request.source_media_paths:
        return request.source_media_paths
    if request.source_media_path is not None:
        return (request.source_media_path,)
    return ()


def _source_image_asset_ids(request: ImageRequest) -> tuple[str, ...]:
    if request.source_media_asset_ids:
        return request.source_media_asset_ids
    if request.source_media_asset_id is not None:
        return (request.source_media_asset_id,)
    return ()


def _video_prompt_for_payload(request: VideoRequest) -> str:
    prompt = request.prompt.strip()
    is_reference_request = (
        request.source_media_path is not None
        and _is_reference_to_video_model(request.model_id)
    )
    if is_reference_request and "@image1" not in prompt.casefold():
        prompt = f"@Image1 {prompt}".strip()
    compacted = _compact_video_prompt(prompt)
    if is_reference_request and "@image1" not in compacted.casefold():
        compacted = _compact_video_prompt(f"@Image1 {compacted}".strip())
    return compacted


def _is_reference_to_video_model(model_id: str) -> bool:
    normalized = _normalize_venice_marker(model_id)
    markers = set(normalized.split("_"))
    return "reference_to_video" in normalized or "r2v" in markers


def _source_image_data_url(path: Path | None) -> str:
    if path is None:
        raise ProviderError(
            category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
            message="Source image is unavailable",
        )
    source = Path(path)
    mime_type = mimetypes.guess_type(source.name)[0]
    if mime_type is None or not mime_type.startswith("image/"):
        raise ProviderError(
            category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
            message=f"Source image MIME type is unsupported: {source.name}",
        )
    return f"data:{mime_type};base64,{_source_image_base64(source)}"


def _venice_video_queue_id(payload: dict[str, Any]) -> str:
    for key in ("queue_id", "id"):
        queue_id = payload.get(key)
        if isinstance(queue_id, str) and queue_id.strip():
            return queue_id.strip()
    raise ProviderError(
        category=ProviderErrorCategory.PROVIDER_ERROR,
        message="Venice video response did not include a queue_id",
    )


def _venice_video_download_url(payload: dict[str, Any]) -> str:
    value = payload.get("download_url")
    if not isinstance(value, str) or not value.strip():
        return ""
    text = value.strip()
    parsed = urlparse(text)
    if not _is_safe_venice_private_video_download_url(parsed):
        raise ProviderError(
            category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
            message="Venice video generation returned an unsafe private download URL",
        )
    return text


def _is_safe_venice_private_video_download_url(parsed: ParseResult) -> bool:
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    try:
        port = parsed.port
    except ValueError:
        return False
    if port not in {None, 443}:
        return False
    hostname = parsed.hostname or ""
    if _is_ip_literal(hostname):
        return False
    if hostname != VENICE_PRIVATE_VIDEO_DOWNLOAD_HOST:
        return False
    return parsed.path.startswith(VENICE_PRIVATE_VIDEO_DOWNLOAD_PATH_PREFIX) and (
        len(parsed.path) > len(VENICE_PRIVATE_VIDEO_DOWNLOAD_PATH_PREFIX)
    )


def _is_ip_literal(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


def _venice_video_response_model(
    *,
    request: VideoRequest,
    queue_response: dict[str, Any],
) -> str:
    model = queue_response.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return request.model_id


def _venice_video_status(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    return status.strip().upper() if isinstance(status, str) else ""


def _venice_video_effective_timeout(
    payload: dict[str, Any],
    *,
    current_timeout: float,
) -> float:
    timeout = current_timeout
    for key in ("average_execution_time", "execution_duration"):
        seconds = _venice_video_timing_seconds(payload.get(key))
        if seconds is None:
            continue
        timeout = max(timeout, seconds + VENICE_VIDEO_TIMEOUT_BUFFER_SECONDS)
    return max(current_timeout, min(timeout, VENICE_VIDEO_MAX_TIMEOUT_SECONDS))


def _venice_video_timing_seconds(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        milliseconds = float(value)
    elif isinstance(value, str):
        try:
            milliseconds = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if milliseconds <= 0:
        return None
    return milliseconds / 1000.0


def _venice_video_json_response(response: BinaryHttpResponse) -> dict[str, Any]:
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message="Venice video retrieve response was not valid JSON",
        ) from exc
    if not isinstance(payload, dict):
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message="Venice video retrieve response must be a JSON object",
        )
    return payload


def _venice_video_terminal_error(payload: dict[str, Any]) -> ProviderError:
    status = _venice_video_status(payload) or "FAILED"
    message = _venice_video_error_message(payload)
    return ProviderError(
        category=(
            ProviderErrorCategory.CONTENT_BLOCKED
            if status
            in {"BLOCKED", "CONTENT_BLOCKED", "MODERATION_FAILED", "REJECTED"}
            or _video_error_indicates_content_block(message)
            else ProviderErrorCategory.PROVIDER_ERROR
        ),
        message=f"Venice video generation {status}: {message}",
    )


def _venice_video_error_message(payload: dict[str, Any]) -> str:
    for key in ("error", "message", "details"):
        error = payload.get(key)
        if isinstance(error, str) and error.strip():
            return error.strip()
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail")
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
            "violation",
        )
    )


def _venice_video_content_type(response: BinaryHttpResponse) -> str:
    mime_type = _binary_content_type(response)
    if mime_type == "video/mp4":
        return mime_type
    raise ProviderError(
        category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
        message="Venice video download did not include video/mp4 content",
    )


def _binary_content_type(response: BinaryHttpResponse) -> str:
    raw_content_type = ""
    for key, value in response.headers.items():
        if key.strip().lower() == "content-type":
            raw_content_type = value
            break
    return raw_content_type.split(";", 1)[0].strip().lower()


def _venice_video_raw_metadata(
    *,
    queue_response: dict[str, Any],
    content: _VeniceVideoContent,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "queue_id": _venice_video_queue_id(queue_response),
        "status": content.status,
        "poll_count": content.poll_count,
        "byte_count": len(content.response.body),
    }
    queue_headers = _safe_metadata_headers(queue_response.get("_bragi_headers"))
    if queue_headers:
        metadata["queue_headers"] = queue_headers
    final_headers = _safe_metadata_headers(content.response.headers)
    if final_headers:
        metadata["_bragi_headers"] = final_headers
    queue_retry = queue_response.get("_bragi_retry")
    if isinstance(queue_retry, dict):
        metadata["queue_retry"] = dict(queue_retry)
    if content.retrieve_retry is not None:
        metadata["retrieve_retry"] = dict(content.retrieve_retry)
    if content.download_retry is not None:
        metadata["download_retry"] = dict(content.download_retry)
    retry = (
        content.download_retry
        or content.retrieve_retry
        or (queue_retry if isinstance(queue_retry, dict) else None)
    )
    if retry is not None:
        metadata["_bragi_retry"] = dict(retry)
    return metadata


def _safe_metadata_headers(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    safe_headers: dict[str, str] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str) or not isinstance(raw_value, str):
            continue
        normalized = key.strip().lower()
        if normalized in SAFE_PROVIDER_RESPONSE_HEADERS:
            safe_headers[normalized] = raw_value
    return safe_headers


def _bytes_result_response_and_retry(
    result: dict[str, Any],
) -> tuple[BinaryHttpResponse, dict[str, Any] | None]:
    response = result.get("_bragi_response")
    if not isinstance(response, BinaryHttpResponse):
        raise AssertionError("Venice binary retry result did not include a response")
    retry = result.get("_bragi_retry")
    return response, dict(retry) if isinstance(retry, dict) else None


def _venice_video_http_error(
    exc: ProviderError,
    *,
    task: str,
    payload: dict[str, Any] | None = None,
) -> ProviderError:
    if task == "video_generation" and exc.status_code == 422:
        return ProviderError(
            category=ProviderErrorCategory.CONTENT_BLOCKED,
            message="Venice video generation content violation (422)",
            status_code=exc.status_code,
        )
    if task == "video_generation":
        validation_summary = _venice_validation_summary(payload)
        if validation_summary:
            return ProviderError(
                category=exc.category,
                message=f"{exc.message}: {validation_summary}",
                status_code=exc.status_code,
            )
    return exc


def _venice_validation_summary(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return ""
    messages: list[str] = []
    _collect_venice_validation_messages(payload, messages)
    unique_messages: list[str] = []
    seen: set[str] = set()
    for message in messages:
        normalized = " ".join(message.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_messages.append(normalized)
    summary = "; ".join(unique_messages)
    if not summary:
        return ""
    redacted = redact_text(summary) or ""
    if len(redacted) <= _MAX_VENICE_VALIDATION_SUMMARY_CHARS:
        return redacted
    return f"{redacted[: _MAX_VENICE_VALIDATION_SUMMARY_CHARS - 3].rstrip()}..."


def _collect_venice_validation_messages(
    value: object,
    messages: list[str],
) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            messages.append(text)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _collect_venice_validation_messages(item, messages)
        return
    if not isinstance(value, dict):
        return
    for raw_key, item in value.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip().lower().replace("-", "_")
        if key in {"error", "errors", "message", "msg", "detail", "details"}:
            _collect_venice_validation_messages(item, messages)


def _provider_url(
    base_url: str,
    path: str,
    *,
    allowed_absolute_hosts: frozenset[str] = frozenset(),
) -> str:
    parsed = urlparse(path)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        if not _is_allowed_absolute_provider_url(
            parsed,
            allowed_hosts=allowed_absolute_hosts,
        ):
            raise ProviderError(
                category=ProviderErrorCategory.PROVIDER_ERROR,
                message="Venice provider URL must be relative",
            )
        return path
    return f"{base_url}{path}"


def _is_allowed_absolute_provider_url(
    parsed: ParseResult,
    *,
    allowed_hosts: frozenset[str],
) -> bool:
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    try:
        port = parsed.port
    except ValueError:
        return False
    if port not in {None, 443}:
        return False
    return (parsed.hostname or "") in allowed_hosts


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
    if normalized_timeout <= 0:
        return await operation
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


def normalize_venice_models(payload: dict[str, Any]) -> list[ProviderModel]:
    records = payload.get("data") or payload.get("models") or []
    if not isinstance(records, list):
        return []
    normalized: list[ProviderModel] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        normalized_record = _flatten_venice_model(dict(record))
        capability_hints = _venice_capability_hints(normalized_record)
        if "chat" in capability_hints and _supports_response_schema(record):
            capability_hints.append("structured_output")
        if "chat" in capability_hints and _supports_function_calling(record):
            capability_hints.append("tool_calling")
        normalized_payload = dict(normalized_record)
        normalized_payload["capabilities"] = capability_hints
        normalized_payload["supported_parameters"] = (
            _venice_supported_parameters(record, capabilities=capability_hints)
        )
        normalized_payload["pricing"] = _venice_model_pricing(record)
        normalized_payload["thinking"] = _venice_thinking_level_support(record)
        normalized.append(
            normalize_model_record(
                provider=VENICE_PROVIDER_NAME,
                payload=normalized_payload,
                default_to_chat=False,
            )
        )
    return normalized


def _venice_model_pricing(record: dict[str, Any]) -> ProviderModelPricing | None:
    model_spec = record.get("model_spec")
    if not isinstance(model_spec, dict):
        return None
    raw_pricing = model_spec.get("pricing")
    if not isinstance(raw_pricing, dict):
        return None
    pricing = ProviderModelPricing(
        input_per_million_tokens_usd=_usd_price(raw_pricing.get("input")),
        output_per_million_tokens_usd=_usd_price(raw_pricing.get("output")),
        cache_read_per_million_tokens_usd=_usd_price(
            _first_present(raw_pricing, "cache_read", "input_cache_read")
        ),
        cache_write_per_million_tokens_usd=_usd_price(
            _first_present(raw_pricing, "cache_write", "input_cache_write")
        ),
        request_usd=_usd_price(
            _first_present(raw_pricing, "request", "generation")
        ),
        image_usd=_usd_price(
            _first_present(raw_pricing, "image", "per_image")
        ),
        note="Variable pricing" if _has_variable_venice_pricing(raw_pricing) else None,
    )
    return pricing if _pricing_has_value(pricing) else None


def _first_present(values: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _usd_price(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("usd")
    if value is None:
        return None
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not decimal.is_finite() or decimal < 0:
        return None
    return format(decimal.normalize(), "f")


def _has_variable_venice_pricing(raw_pricing: dict[str, Any]) -> bool:
    return any(
        isinstance(raw_pricing.get(key), dict)
        for key in ("quality", "resolutions", "duration", "durations")
    )


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


def _venice_supported_parameters(
    record: dict[str, Any],
    *,
    capabilities: list[str],
) -> list[str]:
    parameters: list[str] = []
    if "chat" in capabilities:
        parameters.extend(("temperature", "max_output_tokens"))
    if "image_generation" in capabilities:
        parameters.extend(("image_dimensions", "image_safe_mode"))
    if "image_to_image" in capabilities:
        parameters.append("image_dimensions")
    if (
        any(
            capability in capabilities
            for capability in (
                "text_to_video",
                "image_to_video",
                "image_plus_text_to_video",
            )
        )
        and _supports_safe_mode(record)
    ):
        parameters.append("image_safe_mode")
    return parameters


def _reasoning_effort(config: ChatReasoningConfig | None) -> str | None:
    if config is None:
        return None
    if config.enabled is False:
        return "none"
    if config.effort:
        return config.effort
    return None


def _venice_thinking_level_support(
    record: dict[str, Any],
) -> dict[str, object] | None:
    model_spec = record.get("model_spec")
    if not isinstance(model_spec, dict):
        return None
    capabilities = model_spec.get("capabilities")
    if not isinstance(capabilities, dict):
        return None
    if capabilities.get("supportsReasoningEffort") is not True:
        return None
    levels = _venice_constraint_levels(model_spec)
    if not levels:
        levels = _known_venice_reasoning_levels(record)
    return {
        "levels": levels,
        "default_level": _venice_default_reasoning_level(model_spec, levels),
        "default_enabled": (
            True if capabilities.get("supportsReasoning") is True else None
        ),
        "mandatory": "none" not in levels,
        "supports_max_tokens": False,
    }


def _venice_constraint_levels(model_spec: dict[str, Any]) -> list[str]:
    constraints = model_spec.get("constraints")
    if not isinstance(constraints, dict):
        return []
    reasoning = constraints.get("reasoning_effort")
    if not isinstance(reasoning, dict):
        return []
    for key in ("values", "allowed_values", "enum", "options"):
        values = reasoning.get(key)
        if not isinstance(values, list):
            continue
        levels = [
            level
            for value in values
            if (level := _venice_reasoning_level(value)) is not None
        ]
        if levels:
            return _dedupe_strings(levels)
    return []


def _venice_default_reasoning_level(
    model_spec: dict[str, Any],
    levels: list[str],
) -> str | None:
    constraints = model_spec.get("constraints")
    if not isinstance(constraints, dict):
        return "medium" if "medium" in levels else levels[0] if levels else None
    reasoning = constraints.get("reasoning_effort")
    if isinstance(reasoning, dict):
        default = _venice_reasoning_level(reasoning.get("default"))
        if default in levels:
            return default
    return "medium" if "medium" in levels else levels[0] if levels else None


def _known_venice_reasoning_levels(record: dict[str, Any]) -> list[str]:
    haystack = " ".join(
        str(value)
        for value in (
            record.get("id"),
            record.get("name"),
            _model_spec_text(record, "name"),
            _model_spec_text(record, "description"),
        )
        if value
    ).casefold()
    if "gpt-5.2 codex" in haystack or "gpt-5.3 codex" in haystack:
        return ["xhigh", "high", "medium", "low"]
    if "gpt-5.2" in haystack:
        return ["xhigh", "high", "medium", "low", "none"]
    if "claude opus 4.6" in haystack or "opus 4.6" in haystack:
        return ["max", "high", "medium", "low"]
    if any(
        name in haystack
        for name in ("claude opus 4.5", "sonnet 4.5", "sonnet 4.6")
    ):
        return ["high", "medium", "low"]
    if "gemini 3 flash" in haystack:
        return ["high", "medium", "low", "minimal"]
    if "gemini 3.1 pro" in haystack:
        return ["high", "medium", "low"]
    if "gemini 3 pro" in haystack:
        return ["high", "low"]
    return ["high", "medium", "low"]


def _model_spec_text(record: dict[str, Any], key: str) -> str:
    model_spec = record.get("model_spec")
    if not isinstance(model_spec, dict):
        return ""
    value = model_spec.get(key)
    return value if isinstance(value, str) else ""


def _venice_reasoning_level(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold().replace("-", "_")
    return (
        normalized
        if normalized in {"max", "xhigh", "high", "medium", "low", "minimal", "none"}
        else None
    )


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _supports_safe_mode(record: dict[str, Any]) -> bool:
    values: list[str] = []
    _collect_safe_mode_metadata_values(record, values)
    squashed = {
        value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        for value in values
    }
    return bool(
        squashed
        & {
            "safemode",
            "imagesafemode",
            "mediasafemode",
            "supportssafemode",
            "videosafemode",
        }
    )


def _collect_safe_mode_metadata_values(value: object, values: list[str]) -> None:
    if isinstance(value, list):
        values.extend(str(item) for item in value)
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if item is True:
            values.append(str(key))
        if key in {
            "capabilities",
            "features",
            "model_spec",
            "parameters",
            "supported_parameters",
            "supportedParameters",
            "traits",
        }:
            _collect_safe_mode_metadata_values(item, values)


def _venice_capability_hints(record: dict[str, Any]) -> list[str]:
    model_type = str(record.get("type", "text")).strip().lower().replace("-", "_")
    if model_type == "text":
        capabilities = ["chat"]
        if _supports_vision(record):
            capabilities.append("vision")
        if _has_blocked_output_fallback_venice_marker(record):
            capabilities.append("blocked_output_fallback")
        return capabilities
    if model_type in {"image", "inpaint"}:
        capabilities = (
            ["image_to_image"]
            if model_type == "inpaint" or _supports_image_edit(record)
            else ["image_generation"]
        )
        if _has_blocked_output_fallback_venice_marker(record):
            capabilities.append("blocked_output_fallback")
        return capabilities
    if model_type == "video":
        video_capabilities: list[str] = []
        if _supports_text_to_video(record):
            video_capabilities.append("text_to_video")
        if _supports_image_to_video(record):
            video_capabilities.append("image_to_video")
            video_capabilities.append("image_plus_text_to_video")
        if not video_capabilities:
            video_capabilities.append("text_to_video")
        if _has_blocked_output_fallback_venice_marker(record):
            video_capabilities.append("blocked_output_fallback")
        return video_capabilities
    return []


def _supports_text_to_video(record: dict[str, Any]) -> bool:
    explicit_model_types = _venice_video_constraint_model_types(record)
    if explicit_model_types & {"text_to_video"}:
        return True
    if explicit_model_types and explicit_model_types != {"video"}:
        return False
    return _has_venice_metadata_marker(
        record,
        {
            "text_to_video",
            "text_video",
            "text2video",
            "t2v",
        },
    )


def _supports_image_to_video(record: dict[str, Any]) -> bool:
    explicit_model_types = _venice_video_constraint_model_types(record)
    if explicit_model_types & {"image_to_video", "reference_to_video", "r2v"}:
        return True
    if explicit_model_types - {"video"}:
        return False
    return _has_venice_metadata_marker(
        record,
        {
            "image_to_video",
            "image_plus_text_to_video",
            "image_text_to_video",
            "image_animation",
            "image2video",
            "i2v",
            "reference_to_video",
            "r2v",
        },
    )


def _has_venice_metadata_marker(
    record: dict[str, Any],
    markers: set[str],
) -> bool:
    values: list[str] = []
    for key in ("id", "name", "description"):
        raw = record.get(key)
        if isinstance(raw, str):
            values.append(raw)
    for key in ("traits", "tags", "capabilities"):
        raw = record.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
        elif isinstance(raw, dict):
            values.extend(str(key) for key, value in raw.items() if value is True)
    model_spec = record.get("model_spec")
    if isinstance(model_spec, dict):
        for key in ("id", "name", "description"):
            raw = model_spec.get(key)
            if isinstance(raw, str):
                values.append(raw)
        capabilities = model_spec.get("capabilities")
        if isinstance(capabilities, dict):
            values.extend(
                str(key) for key, value in capabilities.items() if value is True
            )
    normalized_values = {_normalize_venice_marker(value) for value in values}
    return any(
        marker == value or marker in value
        for value in normalized_values
        for marker in markers
    )


def _venice_video_constraint_model_types(record: dict[str, Any]) -> set[str]:
    values: list[str] = []
    model_spec = record.get("model_spec")
    for container in (record, model_spec):
        if not isinstance(container, dict):
            continue
        for key in ("model_type", "modelType"):
            raw = container.get(key)
            if isinstance(raw, str):
                values.append(raw)
    if isinstance(model_spec, dict):
        constraints = model_spec.get("constraints")
        if isinstance(constraints, dict):
            for key in ("model_type", "modelType"):
                raw = constraints.get(key)
                if isinstance(raw, str):
                    values.append(raw)
    return {_normalize_venice_marker(value) for value in values}


def _normalize_venice_marker(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _supports_image_edit(record: dict[str, Any]) -> bool:
    values: list[str] = []
    for key in ("id", "name", "description", "type"):
        raw = record.get(key)
        if isinstance(raw, str):
            values.append(raw)
    for key in ("traits", "tags", "capabilities"):
        raw = record.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
        elif isinstance(raw, dict):
            values.extend(str(key) for key, value in raw.items() if value is True)
    model_spec = record.get("model_spec")
    if isinstance(model_spec, dict):
        for key in ("name", "description", "type"):
            raw = model_spec.get(key)
            if isinstance(raw, str):
                values.append(raw)
        capabilities = model_spec.get("capabilities")
        if isinstance(capabilities, dict):
            values.extend(
                str(key) for key, value in capabilities.items() if value is True
            )
    normalized = {_normalize_venice_marker(value) for value in values}
    return any(
        marker in normalized
        for marker in (
            "image_to_image",
            "image_edit",
            "image_editing",
            "edit",
            "inpaint",
            "inpainting",
        )
    ) or any(value.endswith("_edit") for value in normalized)


def _has_blocked_output_fallback_venice_marker(record: dict[str, Any]) -> bool:
    values: list[str] = []
    for key in ("id", "name", "description"):
        raw = record.get(key)
        if isinstance(raw, str):
            values.append(raw)
    for key in ("traits", "tags", "capabilities"):
        raw = record.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
        elif isinstance(raw, dict):
            values.extend(str(key) for key, value in raw.items() if value is True)
    model_spec = record.get("model_spec")
    if isinstance(model_spec, dict):
        for key in ("id", "name", "description"):
            raw = model_spec.get(key)
            if isinstance(raw, str):
                values.append(raw)
        for key in ("traits", "tags"):
            raw = model_spec.get(key)
            if isinstance(raw, list):
                values.extend(str(item) for item in raw)
        capabilities = model_spec.get("capabilities")
        if isinstance(capabilities, dict):
            values.extend(
                str(key) for key, value in capabilities.items() if value is True
            )
    return any(_is_blocked_output_fallback_marker(value) for value in values)


def _is_blocked_output_fallback_marker(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return any(
        marker in normalized
        for marker in (
            "uncensored",
            "unmoderated",
            "unfiltered",
            "most_uncensored",
        )
    )


def _flatten_venice_model(record: dict[str, Any]) -> dict[str, Any]:
    model_spec = record.get("model_spec")
    if not isinstance(model_spec, dict):
        return record
    if "name" in model_spec:
        record["name"] = model_spec["name"]
    context_window = model_spec.get("availableContextTokens")
    if context_window is not None:
        record["context_window"] = context_window
    return record


def _supports_response_schema(record: dict[str, Any]) -> bool:
    model_spec = record.get("model_spec")
    if not isinstance(model_spec, dict):
        return False
    capabilities = model_spec.get("capabilities")
    if not isinstance(capabilities, dict):
        return False
    return capabilities.get("supportsResponseSchema") is True


def _supports_function_calling(record: dict[str, Any]) -> bool:
    model_spec = record.get("model_spec")
    if not isinstance(model_spec, dict):
        return False
    capabilities = model_spec.get("capabilities")
    if not isinstance(capabilities, dict):
        return False
    return capabilities.get("supportsFunctionCalling") is True


def _supports_vision(record: dict[str, Any]) -> bool:
    model_spec = record.get("model_spec")
    if not isinstance(model_spec, dict):
        return False
    capabilities = model_spec.get("capabilities")
    if not isinstance(capabilities, dict):
        return False
    return capabilities.get("supportsVision") is True


def _compact_image_prompt(prompt: str) -> str:
    return _compact_prompt(
        prompt,
        max_chars=VENICE_IMAGE_PROMPT_MAX_CHARS,
        label="Venice image",
    )


def _compact_video_prompt(prompt: str) -> str:
    return _compact_prompt(
        prompt,
        max_chars=VENICE_VIDEO_PROMPT_MAX_CHARS,
        label="Venice video",
    )


def _compact_prompt(prompt: str, *, max_chars: int, label: str) -> str:
    prompt = prompt.strip()
    if len(prompt) <= max_chars:
        return prompt

    marker = f"\n...[truncated for {label} prompt limit]...\n"
    available = max_chars - len(marker)
    if available <= 0:
        return prompt[:max_chars]

    head_chars = max(1, available // 2)
    tail_chars = max(1, available - head_chars)
    return (prompt[:head_chars].rstrip() + marker + prompt[-tail_chars:].lstrip())[
        :max_chars
    ]


def _image_sizing_payload(
    model_id: str,
    dimensions: tuple[int, int] | None,
) -> dict[str, int | str]:
    if dimensions is None:
        return {}
    width, height = dimensions
    if width <= 0 or height <= 0:
        return {}
    if _uses_pixel_dimensions(model_id):
        return {"width": width, "height": height}
    return {"aspect_ratio": _aspect_ratio(dimensions)}


def _image_edit_sizing_payload(
    dimensions: tuple[int, int] | None,
) -> dict[str, str]:
    if dimensions is None:
        return {}
    width, height = dimensions
    if width <= 0 or height <= 0:
        return {}
    return {"aspect_ratio": _aspect_ratio(dimensions)}


def _source_image_base64(path: Path | None) -> str:
    if path is None or not path.is_file():
        raise ProviderError(
            category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
            message=f"Source image is unavailable: {path}",
        )
    if path.stat().st_size > MAX_PROVIDER_IMAGE_BYTES:
        raise ProviderError(
            category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
            message=f"Source image exceeded {MAX_PROVIDER_IMAGE_BYTES} bytes",
        )
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _uses_pixel_dimensions(model_id: str) -> bool:
    normalized = model_id.strip().casefold()
    return normalized in VENICE_PIXEL_DIMENSION_MODEL_IDS


def _aspect_ratio(dimensions: tuple[int, int]) -> str:
    width, height = dimensions
    divisor = _greatest_common_divisor(width, height)
    return f"{width // divisor}:{height // divisor}"


def _greatest_common_divisor(left: int, right: int) -> int:
    left = abs(left)
    right = abs(right)
    while right:
        left, right = right, left % right
    return max(left, 1)


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


def _structured_output_messages(
    request: StructuredOutputRequest,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Return data using the enforced JSON schema.",
        },
        *(provider_chat_message(message) for message in request.messages),
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


def _venice_image_bytes(payload: dict[str, Any]) -> bytes:
    images = payload.get("images")
    if isinstance(images, list) and images:
        first_image = images[0]
        if not isinstance(first_image, str):
            raise ValueError("Provider image entry must be base64 text")
        return _decode_b64_image(first_image)

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("Provider image response did not include images")
    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("Provider image entry must be an object")
    b64_json = first.get("b64_json")
    if not isinstance(b64_json, str):
        raise ValueError("Provider image response did not include b64_json")
    return _decode_b64_image(b64_json)


def _decode_b64_image(value: str) -> bytes:
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
        return _venice_image_bytes(payload)
    except (binascii.Error, ValueError) as exc:
        raise ProviderError(
            category=ProviderErrorCategory.IMAGE_GENERATION_FAILED,
            message=str(exc),
        ) from exc
