from __future__ import annotations

import asyncio

from bragi.providers.contracts import (
    ChatMessage,
    ChatRequest,
    ImageDescriptionRequest,
    ImageRequest,
    ProviderCapability,
    StructuredOutputRequest,
    VideoRequest,
)
from bragi.providers.fake import FakeProviderClient


def test_fake_provider_reports_config_status_and_models() -> None:
    async def run() -> None:
        configured = FakeProviderClient(configured=True)
        missing = FakeProviderClient(configured=False)

        configured_status = await configured.validate_config()
        missing_status = await missing.validate_config()
        models = await configured.list_models()
        models_by_id = {model.model_id: model for model in models}

        assert configured_status.configured is True
        assert configured_status.authenticated is True
        assert missing_status.configured is False
        assert missing_status.error == "Missing fake provider config"
        assert set(models_by_id) == {
            "fake-chat",
            "fake-image",
            "fake-edit",
        }
        assert models_by_id["fake-chat"].provider == "fake"
        assert models_by_id["fake-chat"].context_window == 16_384
        assert ProviderCapability.CHAT in models_by_id["fake-chat"].capabilities
        assert (
            ProviderCapability.MODEL_LISTING
            in models_by_id["fake-chat"].capabilities
        )
        assert (
            ProviderCapability.STRUCTURED_OUTPUT
            in models_by_id["fake-chat"].capabilities
        )
        assert models_by_id["fake-image"].provider == "fake"
        assert (
            ProviderCapability.IMAGE_GENERATION
            in models_by_id["fake-image"].capabilities
        )
        assert models_by_id["fake-edit"].provider == "fake"
        assert (
            ProviderCapability.IMAGE_TO_IMAGE
            in models_by_id["fake-edit"].capabilities
        )

    asyncio.run(run())


def test_fake_provider_chat_and_media_methods_are_deterministic() -> None:
    async def run() -> None:
        provider = FakeProviderClient()

        chat = await provider.chat(
            ChatRequest(
                provider="fake",
                model_id="fake-chat",
                messages=(ChatMessage(role="player", body="Open the gate."),),
            )
        )
        image = await provider.generate_image(
            ImageRequest(
                provider="fake",
                model_id="fake-image",
                prompt="A brass gate",
                source_save_id="save-1",
                source_message_id="message-1",
            )
        )
        video = await provider.generate_video(
            VideoRequest(
                provider="fake",
                model_id="fake-video",
                prompt="The gate opens",
                source_save_id="save-1",
                source_message_id="message-2",
            )
        )

        assert chat.body == "echo: Open the gate."
        assert chat.raw_request_payload == {
            "model": "fake-chat",
            "messages": [{"role": "player", "content": "Open the gate."}],
            "stream": False,
        }
        assert image.image_path is not None
        assert image.image_path.as_posix() == "save-1/message-1.png"
        assert image.image_bytes == b"fake-image"
        assert video.video_path is not None
        assert video.video_path.as_posix() == "save-1/message-2.mp4"
        assert video.video_bytes == b"fake-video"

    asyncio.run(run())


def test_fake_provider_structured_and_vision_methods_capture_requests() -> None:
    async def run() -> None:
        provider = FakeProviderClient(structured_output={"ok": True})
        structured_request = StructuredOutputRequest(
            provider="fake",
            model_id="fake-chat",
            messages=(ChatMessage(role="system", body="Extract facts."),),
            schema_name="facts",
            schema={"type": "object"},
        )
        vision_request = ImageDescriptionRequest(
            provider="fake",
            model_id="fake-chat",
            image_url="file:///portrait.png",
            prompt="Describe this portrait.",
        )

        structured = await provider.generate_structured_output(structured_request)
        vision = await provider.describe_image(vision_request)

        assert structured.data == {"ok": True}
        assert structured.token_usage == {"total": 7}
        assert provider.structured_output_requests == [structured_request]
        assert vision.description == "A detailed fake character portrait description."
        assert vision.token_usage == {"total": 11}
        assert provider.image_description_requests == [vision_request]

    asyncio.run(run())


def test_fake_provider_allows_content_safety_review_by_default() -> None:
    async def run() -> None:
        provider = FakeProviderClient()
        response = await provider.generate_structured_output(
            StructuredOutputRequest(
                provider="fake",
                model_id="fake-chat",
                messages=(),
                schema_name="content_safety_review",
                schema={"type": "object"},
            )
        )

        assert response.data == {
            "action": "allow",
            "category": "none",
            "reason": "Fake provider content is suitable for general audiences.",
            "minimum_rating": "g",
        }

    asyncio.run(run())
