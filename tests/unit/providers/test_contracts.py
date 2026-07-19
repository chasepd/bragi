from __future__ import annotations

import asyncio
from pathlib import Path

import bragi.providers as provider_exports
from bragi.providers import (
    ChatMessage,
    ChatReasoningConfig,
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderClient,
    ProviderConfigStatus,
    ProviderModel,
    ProviderThinkingLevelSupport,
    ProviderToolCall,
    StructuredOutputProvider,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallMessage,
    ToolCallProvider,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)
from bragi.providers.chat_rendering import chat_system_body
from bragi.providers.contracts import VideoProvider, VideoRequest, VideoResponse


class FakeProvider:
    provider_name = "fake"

    async def validate_config(self) -> ProviderConfigStatus:
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
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
                    }
                ),
                context_window=8192,
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            body=f"echo: {request.messages[-1].body}",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 5},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        return ImageResponse(
            provider=request.provider,
            model_id=request.model_id,
            image_bytes=b"fake-image",
        )


class FakeStructuredProvider(FakeProvider):
    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-state-memory",
                display_name="Fake State Memory",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
                context_window=8192,
            )
        ]

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        assert "json" not in "\n".join(
            message.body.lower() for message in request.messages
        )
        assert request.schema_name == "state_memory_extraction"
        assert "state_changes" in request.schema["properties"]
        assert "memories" in request.schema["properties"]
        assert "conflicts" in request.schema["properties"]
        return StructuredOutputResponse(
            data={
                "state_changes": [
                    {
                        "operation": "upsert",
                        "key": "scene.location",
                        "value": {"name": "Moonwell"},
                        "category": "scene",
                        "confidence": 0.9,
                    }
                ],
                "memories": [
                    {
                        "body": "Mara trusts Elian near the Moonwell.",
                        "tags": ["moonwell", "elian"],
                        "importance": 0.8,
                    }
                ],
                "conflicts": [],
            },
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 7},
        )


class FakeVideoProvider:
    provider_name = "fake-video"

    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        return VideoResponse(
            provider=request.provider,
            model_id=request.model_id,
            mime_type="video/mp4",
            video_bytes=b"fake-video",
            revised_prompt=f"revised: {request.prompt}",
        )


class FakeToolCallProvider(FakeProvider):
    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-tools",
                display_name="Fake Tools",
                capabilities=frozenset({ProviderCapability.TOOL_CALLING}),
                context_window=8192,
            )
        ]

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        assert request.tools[0].name == "record_fact"
        assert request.messages[-1].role == "user"
        return ToolCallResponse(
            tool_calls=(
                ProviderToolCall(
                    id="call-1",
                    name="record_fact",
                    arguments_json='{"body":"The Moonwell glows."}',
                ),
            ),
            body="",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 7},
        )


def test_fake_provider_satisfies_provider_protocol() -> None:
    assert isinstance(FakeProvider(), ProviderClient)


def test_fake_structured_provider_satisfies_structured_output_protocol() -> None:
    assert isinstance(FakeStructuredProvider(), StructuredOutputProvider)


def test_fake_video_provider_satisfies_video_provider_protocol() -> None:
    assert isinstance(FakeVideoProvider(), VideoProvider)


def test_fake_tool_call_provider_satisfies_tool_call_protocol() -> None:
    assert isinstance(FakeToolCallProvider(), ToolCallProvider)


def test_provider_package_exports_video_contracts() -> None:
    assert provider_exports.VideoRequest is VideoRequest
    assert provider_exports.VideoResponse is VideoResponse


def test_provider_package_exports_tool_call_contracts() -> None:
    assert provider_exports.ToolDefinition is ToolDefinition
    assert provider_exports.ToolCallMessage is ToolCallMessage
    assert provider_exports.ProviderToolCall is ProviderToolCall
    assert provider_exports.ToolCallRequest is ToolCallRequest
    assert provider_exports.ToolCallResponse is ToolCallResponse
    assert provider_exports.ToolCallProvider is ToolCallProvider
    assert provider_exports.VideoProvider is VideoProvider
    assert provider_exports.ChatReasoningConfig is ChatReasoningConfig
    assert provider_exports.ProviderThinkingLevelSupport is ProviderThinkingLevelSupport


def test_provider_capabilities_include_distinct_video_flows() -> None:
    assert ProviderCapability.TEXT_TO_VIDEO.value == "text_to_video"
    assert ProviderCapability.IMAGE_TO_VIDEO.value == "image_to_video"
    assert ProviderCapability.IMAGE_PLUS_TEXT_TO_VIDEO.value == (
        "image_plus_text_to_video"
    )


def test_provider_capabilities_keep_tool_calling_distinct_from_structured_output(
) -> None:
    assert ProviderCapability.TOOL_CALLING.value == "tool_calling"
    assert ProviderCapability.TOOL_CALLING is not ProviderCapability.STRUCTURED_OUTPUT


def test_provider_contracts_support_fake_chat_client() -> None:
    async def run() -> None:
        provider = FakeProvider()
        response = await provider.chat(
            ChatRequest(
                provider="fake",
                model_id="fake-chat",
                messages=(ChatMessage(role="player", body="hello"),),
                retrieved_state=("weather=rain",),
                reasoning=ChatReasoningConfig(effort="low", exclude=True),
            )
        )

        assert response.body == "echo: hello"
        assert response.token_usage == {"total": 5}

    asyncio.run(run())


def test_provider_contracts_support_provider_neutral_video_generation() -> None:
    async def run() -> None:
        provider = FakeVideoProvider()
        response = await provider.generate_video(
            VideoRequest(
                provider="fake-video",
                model_id="fake-text-to-video",
                prompt="A lantern drifts over the ash bridge.",
                source_save_id="save-1",
                source_message_id="message-1",
                source_media_asset_id="media-image-1",
                source_media_path=Path("save-1/images/bridge.png"),
                dimensions=(1024, 576),
                safe_mode=True,
            )
        )

        assert response.provider == "fake-video"
        assert response.model_id == "fake-text-to-video"
        assert response.mime_type == "video/mp4"
        assert response.video_bytes == b"fake-video"
        assert response.revised_prompt == (
            "revised: A lantern drifts over the ash bridge."
        )

    asyncio.run(run())


def test_chat_request_keeps_current_scene_recap_distinct_from_summary() -> None:
    request = ChatRequest(
        provider="fake",
        model_id="fake-chat",
        messages=(ChatMessage(role="player", body="I stay alert."),),
        current_scene_recap=(
            "scene.location: Manor dining room\n"
            "scene.present_characters: Mara, Lord Vale",
        ),
        summary="Long-term summary: Mara crossed the ash bridge last week.",
    )

    assert request.current_scene_recap == (
        "scene.location: Manor dining room\n"
        "scene.present_characters: Mara, Lord Vale",
    )
    assert request.summary == (
        "Long-term summary: Mara crossed the ash bridge last week."
    )
    assert request.current_scene_recap != request.summary


def test_chat_system_body_uses_user_guidance_only_without_save_guidance() -> None:
    user_guided = ChatRequest(
        provider="fake",
        model_id="fake-chat",
        messages=(ChatMessage(role="player", body="I stay alert."),),
        user_narration_guidance="Keep narrator responses to two paragraphs or less.",
    )
    save_guided = ChatRequest(
        provider="fake",
        model_id="fake-chat",
        messages=(ChatMessage(role="player", body="I stay alert."),),
        custom_instructions="Keep this save tense and clipped.",
        user_narration_guidance="Keep narrator responses to two paragraphs or less.",
    )

    user_body = chat_system_body(user_guided)
    save_body = chat_system_body(save_guided)

    assert "User narration guidance:" in user_body
    assert "Keep narrator responses to two paragraphs or less." in user_body
    assert "Save response guidance:" not in user_body
    assert "Save response guidance:" in save_body
    assert "Keep this save tense and clipped." in save_body
    assert "User narration guidance:" not in save_body
    assert "Keep narrator responses to two paragraphs or less." not in save_body


def test_chat_request_keeps_retrieved_state_changes_and_media_assets_distinct() -> None:
    request = ChatRequest(
        provider="fake",
        model_id="fake-chat",
        messages=(ChatMessage(role="player", body="I study the bridge image."),),
        retrieved_state=("[world_state:state-1] scene.location: Ash Market",),
        retrieved_state_changes=(
            "[state_change:change-1] scene.exit changed to Moon Gate",
        ),
        retrieved_recent_messages=(
            "[message:message-1] Narrator: The bridge answered with bells.",
        ),
        retrieved_memories=("[memory:memory-1] Mara distrusts quiet bells.",),
        retrieved_media_assets=(
            "[media_asset:media-1] Image prompt: gold bridge lights",
        ),
    )

    assert request.retrieved_state == (
        "[world_state:state-1] scene.location: Ash Market",
    )
    assert request.retrieved_state_changes == (
        "[state_change:change-1] scene.exit changed to Moon Gate",
    )
    assert request.retrieved_recent_messages == (
        "[message:message-1] Narrator: The bridge answered with bells.",
    )
    assert request.retrieved_memories == (
        "[memory:memory-1] Mara distrusts quiet bells.",
    )
    assert request.retrieved_media_assets == (
        "[media_asset:media-1] Image prompt: gold bridge lights",
    )
    assert request.retrieved_state_changes != request.retrieved_state
    assert request.retrieved_media_assets != request.retrieved_memories


def test_chat_request_keeps_pending_context_suggestions_distinct() -> None:
    request = ChatRequest(
        provider="fake",
        model_id="fake-chat",
        messages=(ChatMessage(role="player", body="I wait for review."),),
        pending_context_suggestions=(
            "Pending review (not canon yet): update world_state/storm.mood -> wary",
        ),
        retrieved_state=("[world_state:state-1] storm.mood: calm",),
        summary="Long-term summary: the storm has been calm.",
    )

    assert request.pending_context_suggestions == (
        "Pending review (not canon yet): update world_state/storm.mood -> wary",
    )
    assert request.pending_context_suggestions != request.retrieved_state
    assert request.pending_context_suggestions != (request.summary,)


def test_provider_contracts_support_schema_enforced_structured_output() -> None:
    async def run() -> None:
        provider = FakeStructuredProvider()
        response = await provider.generate_structured_output(
            StructuredOutputRequest(
                provider="fake",
                model_id="fake-state-memory",
                messages=(
                    ChatMessage(
                        role="system",
                        body="Extract durable state and memory from the turn.",
                    ),
                    ChatMessage(role="player", body="I inspect the Moonwell."),
                ),
                schema_name="state_memory_extraction",
                schema={
                    "type": "object",
                    "properties": {
                        "state_changes": {"type": "array"},
                        "memories": {"type": "array"},
                        "conflicts": {"type": "array"},
                    },
                    "required": ["state_changes", "memories", "conflicts"],
                },
            )
        )

        assert response.provider == "fake"
        assert response.model_id == "fake-state-memory"
        assert response.data["state_changes"][0]["key"] == "scene.location"
        assert response.data["memories"][0]["tags"] == ["moonwell", "elian"]
        assert response.data["conflicts"] == []

    asyncio.run(run())


def test_provider_contracts_support_tool_calling_with_raw_arguments() -> None:
    async def run() -> None:
        provider = FakeToolCallProvider()
        response = await provider.generate_tool_calls(
            ToolCallRequest(
                provider="fake",
                model_id="fake-tools",
                messages=(ToolCallMessage(role="user", body="Record the fact."),),
                tools=(
                    ToolDefinition(
                        name="record_fact",
                        description="Records a durable fact.",
                        parameters={
                            "type": "object",
                            "properties": {"body": {"type": "string"}},
                            "required": ["body"],
                            "additionalProperties": False,
                        },
                    ),
                ),
            )
        )

        assert response.provider == "fake"
        assert response.model_id == "fake-tools"
        assert response.tool_calls[0].arguments_json == '{"body":"The Moonwell glows."}'

    asyncio.run(run())
