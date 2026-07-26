"""Deterministic provider client for tests and local UI wiring."""

from __future__ import annotations

from pathlib import Path

from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageDescriptionRequest,
    ImageDescriptionResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
    VideoRequest,
    VideoResponse,
)


class FakeProviderClient:
    provider_name = "fake"

    def __init__(
        self,
        configured: bool = True,
        structured_output: dict[str, object] | None = None,
        tool_calls: tuple[ProviderToolCall, ...] = (),
    ) -> None:
        self.configured = configured
        self.structured_output = structured_output or {}
        self.tool_calls = tool_calls
        self.structured_output_requests: list[StructuredOutputRequest] = []
        self.tool_call_requests: list[ToolCallRequest] = []
        self.image_description_requests: list[ImageDescriptionRequest] = []

    async def validate_config(self) -> ProviderConfigStatus:
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=self.configured,
            authenticated=self.configured,
            error=None if self.configured else "Missing fake provider config",
        )

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-chat",
                display_name="Fake Chat",
                capabilities=frozenset(
                    {
                        ProviderCapability.CHAT,
                        ProviderCapability.MODEL_LISTING,
                        ProviderCapability.STRUCTURED_OUTPUT,
                        ProviderCapability.TOOL_CALLING,
                        ProviderCapability.VISION,
                    }
                ),
                context_window=8192,
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-image",
                display_name="Fake Image",
                capabilities=frozenset({ProviderCapability.IMAGE_GENERATION}),
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-edit",
                display_name="Fake Edit",
                capabilities=frozenset({ProviderCapability.IMAGE_TO_IMAGE}),
            ),
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        latest = request.messages[-1].body if request.messages else ""
        payload = {
            "model": request.model_id,
            "messages": [
                {"role": message.role, "content": message.body}
                for message in request.messages
            ],
            "stream": False,
        }
        return ChatResponse(
            body=f"echo: {latest}",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 5},
            raw_request_payload=payload,
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        return ImageResponse(
            provider=request.provider,
            model_id=request.model_id,
            image_path=(
                Path(request.source_save_id) / f"{request.source_message_id}.png"
            ),
            image_bytes=b"fake-image",
        )

    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        return VideoResponse(
            provider=request.provider,
            model_id=request.model_id,
            mime_type="video/mp4",
            video_path=(
                Path(request.source_save_id) / f"{request.source_message_id}.mp4"
            ),
            video_bytes=b"fake-video",
        )

    async def describe_image(
        self,
        request: ImageDescriptionRequest,
    ) -> ImageDescriptionResponse:
        self.image_description_requests.append(request)
        return ImageDescriptionResponse(
            description="A detailed fake character portrait description.",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 11},
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        data = dict(self.structured_output)
        if request.schema_name == "content_safety_review" and not data:
            data = {
                "action": "allow",
                "category": "none",
                "reason": "Fake provider content is suitable for general audiences.",
                "minimum_rating": "g",
            }
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 7},
        )

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        return ToolCallResponse(
            tool_calls=self.tool_calls,
            body="",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 7},
        )
