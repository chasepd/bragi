from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import stat
import struct
import zlib
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import pytest

from bragi.persistence.models import MediaAssetRecord, MessageRecord, SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
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
    StructuredOutputRequest,
    StructuredOutputResponse,
    VideoRequest,
    VideoResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services import media_service as media_service_module
from bragi.services.character_profile_completion import (
    ScenarioCharacterStarter,
    ScenarioStarterReferenceImage,
)
from bragi.services.content_safety_service import (
    ContentSafetyAction,
    ContentSafetyResult,
    ContentSafetyService,
)
from bragi.services.image_style_settings import save_image_style_preset_setting_key
from bragi.services.media_service import MediaService
from bragi.services.model_preferences import (
    CHARACTER_IMAGE_EDIT_PURPOSE,
    IMAGE_EDIT_FALLBACK_PURPOSE,
    IMAGE_TO_IMAGE_GENERATION_PURPOSE,
    ROLEPLAY_SHARED_MODE_SETTING,
    SAVE_MODEL_OVERRIDES_SETTING,
    SCENE_IMAGE_EDIT_PURPOSE,
    roleplay_model_task,
)

_UNSET = object()
_VALID_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000b49444154789c6360000200000500017a5eab3f"
    "0000000049454e44ae426082"
)
_VALID_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42bragi-test-video"


class BlockingMediaSafetyService(ContentSafetyService):
    def __init__(self) -> None:
        pass

    async def review_media_prompt(
        self,
        *,
        prompt: str,
        content_rating: str,
        save_id: str,
        source_provider: str | None = None,
        source_model_id: str | None = None,
    ) -> ContentSafetyResult:
        del (
            prompt,
            content_rating,
            save_id,
            source_provider,
            source_model_id,
        )
        return ContentSafetyResult(
            body="",
            action=ContentSafetyAction.BLOCK,
            minimum_rating="r",
            agent_ran=True,
        )


def _mark_message_as_fade_transition(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    message_id: str,
) -> MessageRecord:
    repositories.connection.execute(
        """
        UPDATE messages
        SET body = ?, safety_transition = 'fade_to_black'
        WHERE save_id = ? AND id = ?
        """,
        (
            "The intimate moment is kept off-screen. Hours later, "
            "the next scene begins.",
            save_id,
            message_id,
        ),
    )
    repositories.commit()
    message = repositories.get_message(save_id=save_id, message_id=message_id)
    assert message is not None
    return message


def _assert_realistic_prompt(prompt: str, base_prompt: str) -> None:
    assert prompt.startswith(f"{base_prompt}\n\n")
    assert "Style preset: Realistic." in prompt
    assert "photoreal image style" in prompt


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


class RecordingImageProvider:
    provider_name = "fake"

    def __init__(
        self,
        image_bytes: bytes = b"fake-scene-image",
        drafted_prompt: str = "cinematic drafted image prompt",
        image_reference_limit: int = 1,
    ) -> None:
        self.image_bytes = image_bytes
        self.drafted_prompt = drafted_prompt
        self._image_reference_limit = image_reference_limit
        self.chat_requests: list[ChatRequest] = []
        self.image_requests: list[ImageRequest] = []

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
                        ProviderCapability.STRUCTURED_OUTPUT,
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
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        return ChatResponse(
            body=self.drafted_prompt,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 13},
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name != "content_safety_review":
            raise AssertionError(f"unexpected structured schema: {request.schema_name}")
        return StructuredOutputResponse(
            data={
                "action": "allow",
                "category": "none",
                "reason": "Test fixture content is within the ceiling.",
                "minimum_rating": "g",
            },
            provider=request.provider,
            model_id=request.model_id,
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        self.image_requests.append(request)
        return ImageResponse(
            provider=request.provider,
            model_id=request.model_id,
            image_bytes=self.image_bytes,
            revised_prompt=f"revised: {request.prompt}",
        )

    def image_reference_limit(self, model_id: str) -> int:
        return self._image_reference_limit


class RecordingVisionProvider(RecordingImageProvider):
    def __init__(self, description: str) -> None:
        super().__init__(image_bytes=_VALID_PNG_BYTES)
        self.description = description
        self.image_description_requests: list[ImageDescriptionRequest] = []

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-vision",
                display_name="Fake Vision",
                capabilities=frozenset(
                    {
                        ProviderCapability.VISION,
                        ProviderCapability.STRUCTURED_OUTPUT,
                    }
                ),
                context_window=8192,
            )
        ]

    async def describe_image(
        self,
        request: ImageDescriptionRequest,
    ) -> ImageDescriptionResponse:
        self.image_description_requests.append(request)
        return ImageDescriptionResponse(
            description=self.description,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 23},
        )


class RelativePathImageProvider(RecordingImageProvider):
    def __init__(self, image_path: Path) -> None:
        super().__init__()
        self.image_path = image_path

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        self.image_requests.append(request)
        return ImageResponse(
            provider=request.provider,
            model_id=request.model_id,
            image_path=self.image_path,
            revised_prompt=f"revised: {request.prompt}",
        )


class FailingImageProvider(RecordingImageProvider):
    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        self.image_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.IMAGE_GENERATION_FAILED,
            "image backend unavailable",
        )


class FailingPromptProvider(RecordingImageProvider):
    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "prompt drafting unavailable",
        )


class SequenceImageProvider(RecordingImageProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        outcomes: list[ImageResponse | Exception],
        drafted_prompt: str = "cinematic drafted image prompt",
    ) -> None:
        super().__init__(image_bytes=_VALID_PNG_BYTES, drafted_prompt=drafted_prompt)
        self.provider_name = provider_name
        self.outcomes = outcomes

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        self.image_requests.append(request)
        if not self.outcomes:
            raise AssertionError(f"unexpected {self.provider_name} image request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingVideoProvider(RecordingImageProvider):
    def __init__(
        self,
        video_bytes: bytes = _VALID_MP4_BYTES,
        mime_type: str = "video/mp4",
        drafted_prompt: str = "cinematic drafted video prompt",
    ) -> None:
        super().__init__(image_bytes=_VALID_PNG_BYTES, drafted_prompt=drafted_prompt)
        self.video_bytes = video_bytes
        self.mime_type = mime_type
        self.video_requests: list[VideoRequest] = []

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-chat",
                display_name="Fake Chat",
                capabilities=frozenset(
                    {
                        ProviderCapability.CHAT,
                        ProviderCapability.STRUCTURED_OUTPUT,
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
                model_id="fake-video",
                display_name="Fake Video",
                capabilities=frozenset({ProviderCapability.TEXT_TO_VIDEO}),
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-image-video",
                display_name="Fake Image Video",
                capabilities=frozenset(
                    {ProviderCapability.IMAGE_PLUS_TEXT_TO_VIDEO}
                ),
            ),
        ]

    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        self.video_requests.append(request)
        return VideoResponse(
            provider=request.provider,
            model_id=request.model_id,
            mime_type=self.mime_type,
            video_bytes=self.video_bytes,
            revised_prompt=f"revised: {request.prompt}",
        )


class SequenceVideoProvider(RecordingVideoProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        outcomes: list[VideoResponse | Exception],
        drafted_prompt: str = "cinematic drafted video prompt",
    ) -> None:
        super().__init__(video_bytes=_VALID_MP4_BYTES, drafted_prompt=drafted_prompt)
        self.provider_name = provider_name
        self.outcomes = outcomes

    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        self.video_requests.append(request)
        if not self.outcomes:
            raise AssertionError(f"unexpected {self.provider_name} video request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def repositories(
    tmp_path: Path,
    migrated_database_template: Path,
) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    shutil.copy2(migrated_database_template, database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_generate_for_message_rejects_fade_transition_sources(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    fade = _mark_message_as_fade_transition(
        repositories,
        save_id=save.id,
        message_id=messages[-1].id,
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )

    with pytest.raises(ValueError, match="cannot be media sources"):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=fade.id,
            )
        )

    assert provider.chat_requests == []
    assert provider.image_requests == []
    assert repositories.list_media_assets(save.id) == []


def test_generate_for_message_drafts_prompt_and_persists_asset_and_job(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(provider.chat_requests) == 1
    chat_request = provider.chat_requests[0]
    assert chat_request.provider == "fake"
    assert chat_request.model_id == "fake-chat"
    system_text = _chat_request_system_message(chat_request)
    for expected_guidance in (
        "visible subject",
        "setting",
        "what each visible character is wearing",
        "action",
        "facial expression",
        "objects",
        "lighting",
        "weather",
        "time of day",
        "mood",
        "composition",
        "continuity",
        "selected scene message",
        "highest-priority current moment",
        "unsupported",
        "internal",
        "future details",
    ):
        assert expected_guidance in system_text
    chat_context = _chat_request_context(chat_request)
    assert "Bridge of Cinders" in chat_context
    assert "A bridge remembers every oath broken on it." in chat_context
    assert "Oathkeeper" in chat_context
    assert "Mara: I step onto the ash bridge." in chat_context
    assert "Narrator: A bell rings under the span." in chat_context

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.provider == "fake"
    assert request.model_id == "fake-image"
    assert request.source_save_id == save.id
    assert request.source_message_id == messages[-1].id
    _assert_realistic_prompt(request.prompt, "cinematic drafted image prompt")

    media_assets = repositories.list_media_assets(save.id)
    assert [item.id for item in media_assets] == [asset.id]
    media_asset = media_assets[0]
    asset_path = _asset_path(media_dir, media_asset.path)
    assert asset_path.read_bytes() == _VALID_PNG_BYTES
    _assert_private_modes(asset_path)
    assert asset.thumbnail_path == media_asset.thumbnail_path
    _assert_private_thumbnail_if_present(media_dir=media_dir, asset=media_asset)
    assert media_asset.type == "image"
    assert media_asset.source_message_id == messages[-1].id
    assert media_asset.status == "succeeded"
    assert media_asset.provider == "fake"
    assert media_asset.model == "fake-image"
    assert media_asset.prompt == request.prompt

    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["payload"]["save_id"] == save.id
    assert jobs[0]["payload"]["source_message_id"] == messages[-1].id
    assert jobs[0]["payload"]["provider"] == "fake"
    assert jobs[0]["payload"]["model"] == "fake-image"
    assert jobs[0]["result"]["media_asset_id"] == media_asset.id
    assert jobs[0]["result"]["path"] == media_asset.path
    assert jobs[0]["result"]["prompt_chars"] == len(request.prompt)


def test_upload_character_text_player_photo_describes_and_persists_upload(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="After School",
        premise="A small town romance.",
        player_role="Mira",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="After School")
    player = repositories.add_character(
        save_id=save.id,
        name="Mira",
        role="player",
        is_player_character=True,
    )
    npc = repositories.add_character(save_id=save.id, name="Rowan", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        sender_character_id=player.id,
        body="Do you recognize this mark?",
        delivery_status="pending",
    )
    repositories.set_model_preference(
        task="character_image_description",
        provider="fake",
        model_id="fake-vision",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-vision",
        display_name="Fake Vision",
        capabilities=[ProviderCapability.VISION.value],
    )
    provider = RecordingVisionProvider(
        "A brass sigil on cracked blue tile, photographed close up."
    )
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    upload = asyncio.run(
        service.upload_character_text_player_photo(
            save_id=save.id,
            text_message=message,
            sender_character_id=player.id,
            image_bytes=_VALID_PNG_BYTES,
            filename="sigil.png",
        )
    )

    assert upload.description == (
        "A brass sigil on cracked blue tile, photographed close up."
    )
    assert len(provider.image_description_requests) == 1
    request = provider.image_description_requests[0]
    assert request.provider == "fake"
    assert request.model_id == "fake-vision"
    assert request.image_url.startswith("data:image/png;base64,")
    assert request.system_prompt is not None
    assert "text-message recipient" in request.system_prompt
    assert "physical appearance" not in request.system_prompt
    assert "Do you recognize this mark?" in request.prompt
    assert "sigil.png" not in request.prompt
    assert upload.asset.provider == "local"
    assert upload.asset.model == "upload"
    assert upload.asset.mime_type == "image/png"
    assert _asset_path(media_dir, upload.asset.path).read_bytes() == _VALID_PNG_BYTES
    metadata = json.loads(upload.asset.metadata_json)
    assert metadata["kind"] == "character_text_uploaded_photo"
    assert metadata["thread_id"] == thread.id
    assert metadata["text_message_id"] == message.id
    assert metadata["sender_character_id"] == player.id
    assert "filename" not in metadata
    assert metadata["description"] == upload.description
    assert metadata["vision_provider"] == "fake"
    assert metadata["vision_model"] == "fake-vision"
    _assert_private_thumbnail_if_present(media_dir=media_dir, asset=upload.asset)


def test_upload_character_text_player_photo_uses_save_scoped_image_details_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="After School",
        premise="A small town romance.",
        player_role="Mira",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="After School")
    player = repositories.add_character(
        save_id=save.id,
        name="Mira",
        role="player",
        is_player_character=True,
    )
    npc = repositories.add_character(save_id=save.id, name="Rowan", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        sender_character_id=player.id,
        body="Is this the seal?",
        delivery_status="pending",
    )
    repositories.set_model_preference(
        task="character_image_description",
        provider="fake",
        model_id="global-vision",
    )
    for model_id in ("global-vision", "save-vision"):
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=model_id,
            capabilities=[ProviderCapability.VISION.value],
        )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_MODEL_OVERRIDES_SETTING,
        value={
            "preferences": {
                "character_image_description": {
                    "provider": "fake",
                    "model_id": "save-vision",
                }
            }
        },
    )
    provider = RecordingVisionProvider("A stamped wax seal in red light.")
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )

    upload = asyncio.run(
        service.upload_character_text_player_photo(
            save_id=save.id,
            text_message=message,
            sender_character_id=player.id,
            image_bytes=_VALID_PNG_BYTES,
        )
    )

    assert upload.description == "A stamped wax seal in red light."
    assert len(provider.image_description_requests) == 1
    assert provider.image_description_requests[0].model_id == "save-vision"
    metadata = json.loads(upload.asset.metadata_json)
    assert metadata["vision_model"] == "save-vision"


def test_cleanup_character_text_uploaded_photo_archives_asset_and_deletes_files(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="After School",
        premise="A small town romance.",
        player_role="Mira",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="After School")
    player = repositories.add_character(
        save_id=save.id,
        name="Mira",
        role="player",
        is_player_character=True,
    )
    npc = repositories.add_character(save_id=save.id, name="Rowan", met=True)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        sender_character_id=player.id,
        body="Do you recognize this mark?",
        delivery_status="pending",
    )
    repositories.set_model_preference(
        task="character_image_description",
        provider="fake",
        model_id="fake-vision",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-vision",
        display_name="Fake Vision",
        capabilities=[ProviderCapability.VISION.value],
    )
    provider = RecordingVisionProvider("A brass sigil on cracked blue tile.")
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )
    upload = asyncio.run(
        service.upload_character_text_player_photo(
            save_id=save.id,
            text_message=message,
            sender_character_id=player.id,
            image_bytes=_VALID_PNG_BYTES,
        )
    )
    asset_path = _asset_path(media_dir, upload.asset.path)
    thumbnail_path = (
        _asset_path(media_dir, upload.asset.thumbnail_path)
        if upload.asset.thumbnail_path
        else None
    )
    assert asset_path.is_file()

    service.cleanup_character_text_uploaded_photo(
        save_id=save.id,
        asset=upload.asset,
    )

    assert repositories.list_media_assets(save.id) == []
    assert not asset_path.exists()
    if thumbnail_path is not None:
        assert not thumbnail_path.exists()


def test_generate_character_text_character_image_includes_visual_direction(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    character = repositories.get_character(character_id)
    assert character is not None
    character = replace(character, content_rating="pg-13")
    repositories.update_character(character)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=character.id,
    )
    text_message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=character.id,
        sender="character",
        body="The ritual robe survived the rain. Sending proof.",
        content_rating="r",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="full_roleplay_text_message_image_edit_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    reference = _persist_character_reference(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        character_id=character.id,
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    asset = asyncio.run(
        service.generate_character_text_character_image(
            save_id=save.id,
            text_message=text_message,
            character=character,
            visual_prompt=(
                "Oracle mirror selfie\n\n"
                "Character visual direction for Oracle of Glass:\n"
                "Wearing: rain-darkened blue glass robes.\n"
                "Current action/pose: holding the mirror toward the beads.\n"
                "Facial expression: small relieved smile."
            ),
            scene_context="Phone thread:\nMara: Did the robe survive?",
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.source_media_asset_id == reference.id
    assert "Character visual direction for Oracle of Glass" in request.prompt
    assert "Wearing: rain-darkened blue glass robes." in request.prompt
    assert "Current action/pose: holding the mirror toward the beads." in request.prompt
    assert "Facial expression: small relieved smile." in request.prompt
    metadata = json.loads(asset.metadata_json)
    assert metadata["kind"] == "character_text_character_image"
    assert metadata["content_rating"] == "r"


def test_generate_character_text_object_image_persists_openrouter_request_alias(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _messages = _save_with_image_preference(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        met=True,
        content_rating="r",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=character.id,
        title=character.name,
    )
    text_message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=character.id,
        sender="character",
        body="This is the corner booth I mentioned.",
        content_rating="pg-13",
    )
    requested_model = "google/gemini-3.1-flash-lite-image"
    response_model = "google/gemini-3.1-flash-lite-image-20260630"
    repositories.save_provider_model(
        provider="openrouter",
        model_id=requested_model,
        display_name="Gemini Flash Lite Image",
        capabilities=["image_generation"],
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="openrouter",
        model_id=requested_model,
    )
    provider = SequenceImageProvider(
        provider_name="openrouter",
        outcomes=[
            ImageResponse(
                provider="openrouter",
                model_id=response_model,
                image_bytes=_VALID_PNG_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"openrouter": provider},
        media_dir=tmp_path / "media",
    )

    asset = asyncio.run(
        service.generate_character_text_object_context_image(
            save_id=save.id,
            text_message=text_message,
            character=character,
            visual_prompt="a quiet restaurant corner booth",
            scene_context="Phone thread:\nMika: This is the corner booth.",
        )
    )

    assert len(provider.image_requests) == 1
    assert provider.image_requests[0].model_id == requested_model
    assert asset.provider == "openrouter"
    assert asset.model == requested_model
    metadata = json.loads(asset.metadata_json)
    assert metadata["kind"] == "character_text_object_context_image"
    assert metadata["content_rating"] == "r"
    assert metadata["requested_model_id"] == requested_model
    assert metadata["response_model_id"] == response_model


def test_generate_character_text_object_image_enforces_child_content_rating(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=child.id,
        key="content_filter_rating",
        value="g",
    )
    save, _messages = _save_with_image_preference(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        met=True,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=character.id,
        title=character.name,
    )
    text_message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=character.id,
        sender="character",
        body="This is what I found.",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        content_safety_service=BlockingMediaSafetyService(),
    )

    with pytest.raises(ValueError, match="selected content rating"):
        asyncio.run(
            service.generate_character_text_object_context_image(
                save_id=save.id,
                text_message=text_message,
                character=character,
                visual_prompt="a gun resting on the table",
                scene_context="Phone thread",
                current_user_id=child.id,
            )
        )

    assert provider.image_requests == []


def test_generate_scene_image_includes_character_visual_direction_without_reference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    source_message = messages[-1]
    character = repositories.add_character(
        save_id=save.id,
        name="Bell Warden",
        met=True,
        visual_notes="soot-dark cloak with brass chimes",
        current_clothing="borrowed green raincoat over a linen shirt",
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=source_message.id,
        character_id=character.id,
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=source_message.id,
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.source_media_asset_id is None
    assert "cinematic drafted image prompt" in request.prompt
    assert "Character visual direction for Bell Warden" in request.prompt
    assert "Wearing: borrowed green raincoat over a linen shirt." in request.prompt
    assert "Wearing: soot-dark cloak with brass chimes." not in request.prompt
    assert "Current action/pose: The echo answers from below." in request.prompt
    assert "Facial expression: expression grounded in this moment" in request.prompt


def test_generate_for_message_does_not_persist_unusable_thumbnail_when_scaling_fails(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    provider = RecordingImageProvider(b"full-size image bytes")
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    def fail_thumbnail_scaling(*, image_path: Path, thumbnail_path: Path) -> bool:
        assert image_path.is_file()
        assert thumbnail_path.name.endswith(".png")
        return False

    monkeypatch.setattr(
        media_service_module,
        "_write_scaled_thumbnail",
        fail_thumbnail_scaling,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    media_asset = repositories.list_media_assets(save.id)[0]
    assert asset.thumbnail_path is None
    assert media_asset.thumbnail_path is None
    assert (
        _asset_path(media_dir, media_asset.path).read_bytes()
        == b"full-size image bytes"
    )
    thumbnail_dir = _asset_path(media_dir, media_asset.path).parent / "thumbnails"
    assert not thumbnail_dir.exists() or list(thumbnail_dir.iterdir()) == []


def test_generate_for_message_persists_private_thumbnail_when_scaling_succeeds(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    provider = RecordingImageProvider(b"full-size image bytes")
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    def write_thumbnail(*, image_path: Path, thumbnail_path: Path) -> bool:
        assert image_path.read_bytes() == b"full-size image bytes"
        thumbnail_path.write_bytes(b"scaled thumbnail bytes")
        return True

    monkeypatch.setattr(
        media_service_module,
        "_write_scaled_thumbnail",
        write_thumbnail,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert asset.thumbnail_path is not None
    thumbnail_path = _asset_path(media_dir, asset.thumbnail_path)
    assert thumbnail_path.read_bytes() == b"scaled thumbnail bytes"
    _assert_private_thumbnail_if_present(media_dir=media_dir, asset=asset)


def test_generate_for_message_sends_disabled_venice_safe_mode(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_app_setting("venice_image_safe_mode", False)
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/image",
        display_name="Venice Image",
        capabilities=["image_generation"],
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="venice",
        model_id="venice/image",
    )
    prompt_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    image_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": prompt_provider, "venice": image_provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(image_provider.image_requests) == 1
    assert image_provider.image_requests[0].safe_mode is False
    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["payload"]["venice_safe_mode"] is False
    assert jobs[0]["result"]["primary_venice_safe_mode"] is False


def test_generate_for_message_forces_venice_safe_mode_for_child_account(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    save, messages = _save_with_image_preference(repositories)
    repositories.set_app_setting("venice_image_safe_mode", False)
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/image",
        display_name="Venice Image",
        capabilities=["image_generation"],
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="venice",
        model_id="venice/image",
    )
    image_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": RecordingImageProvider(_VALID_PNG_BYTES),
            "venice": image_provider,
        },
        media_dir=tmp_path / "media",
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
            current_user_id=child.id,
        )
    )

    assert image_provider.image_requests[0].safe_mode is True


def test_generate_for_message_rejects_non_safe_mode_provider_for_child_account(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    save, messages = _save_with_image_preference(repositories)
    repositories.save_provider_model(
        provider="openrouter",
        model_id="vendor/image",
        display_name="OpenRouter Image",
        capabilities=["image_generation"],
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="openrouter",
        model_id="vendor/image",
    )
    image_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": RecordingImageProvider(_VALID_PNG_BYTES),
            "openrouter": image_provider,
        },
        media_dir=tmp_path / "media",
    )

    with pytest.raises(ValueError, match="enforced safe mode"):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
                current_user_id=child.id,
            )
        )

    assert image_provider.image_requests == []


def test_generate_for_message_applies_supported_image_dimension_setting(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-image",
        display_name="Fake Image",
        capabilities=["image_generation"],
        supported_parameters=["image_dimensions"],
    )
    repositories.set_app_setting("image_dimension_preset", "landscape_1024x768")
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(provider.image_requests) == 1
    assert provider.image_requests[0].dimensions == (1024, 768)


def test_generate_for_message_uses_image_prompt_preference_for_prompt_drafting(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_prompt",
        provider="prompt",
        model_id="prompt/drafter",
    )
    prompt_provider = RecordingImageProvider(
        _VALID_PNG_BYTES,
        drafted_prompt="image-prompt-model drafted prompt",
    )
    image_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": image_provider, "prompt": prompt_provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert image_provider.chat_requests == []
    assert len(prompt_provider.chat_requests) == 1
    chat_request = prompt_provider.chat_requests[0]
    assert chat_request.provider == "prompt"
    assert chat_request.model_id == "prompt/drafter"
    assert len(image_provider.image_requests) == 1
    assert image_provider.image_requests[0].provider == "fake"
    assert image_provider.image_requests[0].model_id == "fake-image"
    _assert_realistic_prompt(
        image_provider.image_requests[0].prompt,
        "image-prompt-model drafted prompt",
    )


def test_generate_for_message_applies_selected_image_style_preset(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    other_save = repositories.create_save(
        scenario_id=save.scenario_id,
        title="Signal Tower",
    )
    repositories.set_app_setting(
        save_image_style_preset_setting_key(save.id),
        "low_poly",
    )
    repositories.set_app_setting(
        save_image_style_preset_setting_key(other_save.id),
        "anime",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.prompt.startswith("cinematic drafted image prompt\n\n")
    assert "Style preset: Low Poly." in request.prompt
    assert "faceted geometry" in request.prompt
    assert asset.prompt == request.prompt


def test_generate_for_message_skips_image_only_prompt_preference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_prompt",
        provider="fake",
        model_id="fake-image",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-image",
        display_name="Fake Image",
        capabilities=["image_generation"],
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat"],
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert [request.model_id for request in provider.chat_requests] == ["fake-chat"]


def test_generate_for_message_recovers_from_empty_shared_image_prompt_with_shared_chat(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_prompt",
        provider="shared-prompt",
        model_id="shared/prompt-drafter",
    )
    repositories.set_model_preference(
        task="chat",
        provider="shared-chat",
        model_id="shared/chat-drafter",
    )
    image_prompt_provider = RecordingImageProvider(drafted_prompt=" \n\t ")
    shared_chat_provider = RecordingImageProvider(
        drafted_prompt="shared chat drafted image prompt",
    )
    image_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": image_provider,
            "shared-prompt": image_prompt_provider,
            "shared-chat": shared_chat_provider,
        },
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(image_prompt_provider.chat_requests) == 1
    assert image_prompt_provider.chat_requests[0].provider == "shared-prompt"
    assert image_prompt_provider.chat_requests[0].model_id == "shared/prompt-drafter"
    assert len(shared_chat_provider.chat_requests) == 1
    assert shared_chat_provider.chat_requests[0].provider == "shared-chat"
    assert shared_chat_provider.chat_requests[0].model_id == "shared/chat-drafter"
    assert image_provider.chat_requests == []
    assert len(image_provider.image_requests) == 1
    assert image_provider.image_requests[0].provider == "fake"
    assert image_provider.image_requests[0].model_id == "fake-image"
    _assert_realistic_prompt(
        image_provider.image_requests[0].prompt,
        "shared chat drafted image prompt",
    )
    assert asset.prompt == image_provider.image_requests[0].prompt


def test_generate_for_message_uses_chat_preference_for_prompt_drafting_fallback(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(provider.chat_requests) == 1
    chat_request = provider.chat_requests[0]
    assert chat_request.provider == "fake"
    assert chat_request.model_id == "fake-chat"


def test_generate_for_message_recovers_from_empty_scenario_image_prompt(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_app_setting("use_shared_roleplay_models", False)
    repositories.set_model_preference(
        task="full_roleplay_image_prompt",
        provider="scenario-prompt",
        model_id="scenario/prompt-drafter",
    )
    repositories.set_model_preference(
        task="image_prompt",
        provider="shared-prompt",
        model_id="shared/prompt-drafter",
    )
    image_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    scenario_prompt_provider = RecordingImageProvider(drafted_prompt=" \n\t ")
    shared_prompt_provider = RecordingImageProvider(
        drafted_prompt="shared image prompt after empty scenario prompt",
    )
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": image_provider,
            "scenario-prompt": scenario_prompt_provider,
            "shared-prompt": shared_prompt_provider,
        },
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(scenario_prompt_provider.chat_requests) == 1
    assert scenario_prompt_provider.chat_requests[0].provider == "scenario-prompt"
    assert scenario_prompt_provider.chat_requests[0].model_id == (
        "scenario/prompt-drafter"
    )
    assert len(shared_prompt_provider.chat_requests) == 1
    assert shared_prompt_provider.chat_requests[0].provider == "shared-prompt"
    assert shared_prompt_provider.chat_requests[0].model_id == "shared/prompt-drafter"
    assert len(image_provider.image_requests) == 1
    _assert_realistic_prompt(
        image_provider.image_requests[0].prompt,
        "shared image prompt after empty scenario prompt",
    )
    assert asset.prompt == image_provider.image_requests[0].prompt


def test_generate_for_message_omits_unselected_full_scenario_sections(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    selected_moment = "The beacon lens flashes green as Mara raises the shutter."
    unselected_lore = "The buried legion names every signal warden in copper."
    unselected_factions = "The pantry guild argues about salted turnips."
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "locations": unselected_lore,
            "factions": unselected_factions,
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=selected_moment,
        provider="fake",
        model="fake-chat",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=source_message.id,
        )
    )

    chat_context = _chat_request_context(provider.chat_requests[0])
    assert "Ashfall Keep" in chat_context
    assert "A border keep is cut off by ash storms." in chat_context
    assert "Signal warden" in chat_context
    assert selected_moment in chat_context
    assert unselected_lore not in chat_context
    assert unselected_factions not in chat_context


def test_generate_for_message_repeated_calls_keep_distinct_files(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=media_dir,
        auto_frequency=3,
    )

    first = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )
    second = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert first.path != second.path
    first_path = _asset_path(media_dir, first.path)
    second_path = _asset_path(media_dir, second.path)
    assert first_path.is_file()
    assert second_path.is_file()
    assert first_path.read_bytes() == _VALID_PNG_BYTES
    assert second_path.read_bytes() == _VALID_PNG_BYTES
    _assert_private_thumbnail_if_present(media_dir=media_dir, asset=first)
    _assert_private_thumbnail_if_present(media_dir=media_dir, asset=second)
    if first.thumbnail_path is not None and second.thumbnail_path is not None:
        assert first.thumbnail_path != second.thumbnail_path
    media_assets = repositories.list_media_assets(save.id)
    assert [asset.path for asset in media_assets] == [first.path, second.path]


def test_regenerate_asset_with_prompt_replaces_image_without_drafting_prompt(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "scene_image"},
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )

    new_asset = asyncio.run(
        service.regenerate_asset_with_prompt(
            save_id=save.id,
            media_asset_id=old_asset.id,
            prompt="edited lantern prompt",
        )
    )

    assert provider.chat_requests == []
    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.prompt == "edited lantern prompt"
    assert request.source_message_id == messages[-1].id
    assert new_asset.source_message_id == old_asset.source_message_id
    assert new_asset.prompt == "edited lantern prompt"
    assert json.loads(new_asset.metadata_json) == {
        "content_rating": "g",
        "kind": "scene_image",
        "regenerated_from_media_asset_id": old_asset.id,
    }
    assert [asset.id for asset in repositories.list_media_assets(save.id)] == [
        new_asset.id,
    ]
    archived = repositories.list_all_media_assets(save.id)[0]
    assert archived.id == old_asset.id
    assert archived.archived_at is not None


@pytest.mark.parametrize(
    "prompt",
    (
        "He thrust into her as the lanterns went dark.",
        "Their hands slipped beneath each other's clothes.",
    ),
)
def test_regenerate_asset_with_prompt_rejects_intimate_prompt_before_job_or_provider(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    prompt: str,
) -> None:
    save, _messages = _save_with_image_preference(repositories)
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "scene_image"},
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        content_safety_service=BlockingMediaSafetyService(),
    )

    with pytest.raises(ValueError, match="selected content rating"):
        asyncio.run(
            service.regenerate_asset_with_prompt(
                save_id=save.id,
                media_asset_id=old_asset.id,
                prompt=prompt,
            )
        )

    assert provider.chat_requests == []
    assert provider.image_requests == []
    assert _media_jobs(repositories, save.id, "image_regeneration") == []
    assert repositories.list_media_assets(save.id) == [old_asset]
    assert prompt not in repr(repositories.list_all_media_assets(save.id))


def test_regenerate_asset_with_prompt_enforces_child_content_rating(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=child.id,
        key="content_filter_rating",
        value="g",
    )
    save, _messages = _save_with_image_preference(repositories)
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "scene_image"},
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        content_safety_service=BlockingMediaSafetyService(),
    )

    with pytest.raises(ValueError, match="selected content rating"):
        asyncio.run(
            service.regenerate_asset_with_prompt(
                save_id=save.id,
                media_asset_id=old_asset.id,
                prompt="They kiss beneath the lanterns.",
                current_user_id=child.id,
            )
        )

    assert provider.image_requests == []
    assert _media_jobs(repositories, save.id, "image_regeneration") == []


def test_regenerate_child_explicit_act_prompt_is_rejected_before_venice(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    save, _messages = _save_with_image_preference(repositories)
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="venice",
        model="venice/image",
        status="succeeded",
        metadata={"kind": "scene_image"},
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"venice": provider},
        media_dir=tmp_path / "media",
        content_safety_service=BlockingMediaSafetyService(),
    )

    with pytest.raises(ValueError, match="selected content rating"):
        asyncio.run(
            service.regenerate_asset_with_prompt(
                save_id=save.id,
                media_asset_id=old_asset.id,
                prompt="A sex act on a bed.",
                current_user_id=child.id,
            )
        )

    assert provider.image_requests == []
    assert _media_jobs(repositories, save.id, "image_regeneration") == []


def test_regenerate_child_prompt_requires_provider_with_enforced_safe_mode(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    save, _messages = _save_with_image_preference(repositories)
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="openrouter",
        model="vendor/image",
        status="succeeded",
        metadata={"kind": "scene_image"},
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"openrouter": provider},
        media_dir=tmp_path / "media",
    )

    with pytest.raises(ValueError, match="enforced safe mode"):
        asyncio.run(
            service.regenerate_asset_with_prompt(
                save_id=save.id,
                media_asset_id=old_asset.id,
                prompt=(
                    "An unclothed adult showing their private parts on a bed."
                ),
                current_user_id=child.id,
            )
        )

    assert provider.image_requests == []
    assert _media_jobs(repositories, save.id, "image_regeneration") == []


def test_regenerate_asset_with_prompt_rejects_fade_transition_source(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    source = _mark_message_as_fade_transition(
        repositories,
        save_id=save.id,
        message_id=messages[-1].id,
    )
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=source.id,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "scene_image"},
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )

    with pytest.raises(ValueError, match="cannot be media sources"):
        asyncio.run(
            service.regenerate_asset_with_prompt(
                save_id=save.id,
                media_asset_id=old_asset.id,
                prompt="a safe lantern scene",
            )
        )

    assert provider.chat_requests == []
    assert provider.image_requests == []
    assert _media_jobs(repositories, save.id, "image_regeneration") == []
    assert repositories.list_media_assets(save.id) == [old_asset]


def test_regenerate_asset_with_prompt_keeps_old_asset_when_provider_fails(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    provider = FailingImageProvider()
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )

    with pytest.raises(Exception, match="image backend unavailable"):
        asyncio.run(
            service.regenerate_asset_with_prompt(
                save_id=save.id,
                media_asset_id=old_asset.id,
                prompt="edited lantern prompt",
            )
        )

    assert [asset.id for asset in repositories.list_media_assets(save.id)] == [
        old_asset.id,
    ]
    assert len(repositories.list_all_media_assets(save.id)) == 1


def test_regenerate_asset_with_prompt_does_not_use_image_fallback(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    _configure_image_fallback(repositories, enabled=True)
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    primary_provider = FailingImageProvider()
    fallback_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": primary_provider, "fallback": fallback_provider},
        media_dir=tmp_path / "media",
    )

    with pytest.raises(Exception, match="image backend unavailable"):
        asyncio.run(
            service.regenerate_asset_with_prompt(
                save_id=save.id,
                media_asset_id=old_asset.id,
                prompt="edited lantern prompt",
            )
        )

    assert len(primary_provider.image_requests) == 1
    assert fallback_provider.image_requests == []
    assert [asset.id for asset in repositories.list_media_assets(save.id)] == [
        old_asset.id,
    ]
    assert len(repositories.list_all_media_assets(save.id)) == 1
    job = _media_jobs(repositories, save.id, "image_regeneration")[0]
    assert job["status"] == "failed"
    assert job["result"]["fallback_used"] is False
    assert job["result"]["fallback_skipped_reason"] == "disabled_for_regeneration"


def test_regenerate_asset_with_prompt_disables_openrouter_provider_fallback(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="openrouter",
        model="openrouter/image",
        status="succeeded",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"openrouter": provider},
        media_dir=tmp_path / "media",
    )

    asyncio.run(
        service.regenerate_asset_with_prompt(
            save_id=save.id,
            media_asset_id=old_asset.id,
            prompt="edited lantern prompt",
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.provider == "openrouter"
    assert request.model_id == "openrouter/image"
    assert request.openrouter_provider_routing == {"allow_fallbacks": False}


def test_regenerate_asset_with_prompt_uses_openrouter_alias_for_dated_response_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _messages = _save_with_image_preference(repositories)
    requested_model = "google/gemini-3.1-flash-lite-image"
    response_model = "google/gemini-3.1-flash-lite-image-20260630"
    repositories.set_model_preference(
        task="image_generation",
        provider="openrouter",
        model_id="google/gemini-3.1-flash-image",
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id=requested_model,
        display_name="Gemini Flash Lite Image",
        capabilities=["image_generation"],
    )
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/text-location.png",
        thumbnail_path=None,
        prompt="original generated text attachment prompt",
        provider="openrouter",
        model=response_model,
        status="succeeded",
        metadata={
            "kind": "character_text_object_context_image",
            "text_message_id": "text-message-1",
        },
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"openrouter": provider},
        media_dir=tmp_path / "media",
    )

    new_asset = asyncio.run(
        service.regenerate_asset_with_prompt(
            save_id=save.id,
            media_asset_id=old_asset.id,
            prompt="edited text attachment prompt",
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.provider == "openrouter"
    assert request.model_id == requested_model
    assert new_asset.provider == "openrouter"
    assert new_asset.model == requested_model
    assert json.loads(new_asset.metadata_json)["requested_model_id"] == requested_model


def test_regenerate_asset_with_prompt_prefers_openrouter_requested_model_metadata(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _messages = _save_with_image_preference(repositories)
    requested_model = "google/gemini-3.1-flash-lite-image"
    response_model = "google/gemini-3.1-flash-lite-image-20260630"
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/text-location.png",
        thumbnail_path=None,
        prompt="original generated text attachment prompt",
        provider="openrouter",
        model=response_model,
        status="succeeded",
        metadata={
            "kind": "character_text_object_context_image",
            "text_message_id": "text-message-1",
            "requested_model_id": requested_model,
            "response_model_id": response_model,
        },
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"openrouter": provider},
        media_dir=tmp_path / "media",
    )

    new_asset = asyncio.run(
        service.regenerate_asset_with_prompt(
            save_id=save.id,
            media_asset_id=old_asset.id,
            prompt="edited text attachment prompt",
        )
    )

    assert len(provider.image_requests) == 1
    assert provider.image_requests[0].model_id == requested_model
    assert new_asset.model == requested_model
    metadata = json.loads(new_asset.metadata_json)
    assert metadata["requested_model_id"] == requested_model
    assert "response_model_id" not in metadata


def test_regenerate_text_character_image_uses_image_edit_fallback_alias(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    character = repositories.get_character(character_id)
    assert character is not None
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=character.id,
        title=character.name,
    )
    text_message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=character.id,
        sender="character",
        body="The mirror caught the robe clearly.",
    )
    requested_model = "openrouter/text-message-edit-image"
    response_model = "openrouter/text-message-edit-image-20260630"
    repositories.set_model_preference(
        task=IMAGE_TO_IMAGE_GENERATION_PURPOSE,
        provider="openrouter",
        model_id=requested_model,
    )
    media_dir = tmp_path / "media"
    reference = _persist_character_reference(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        character_id=character.id,
    )
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/text-selfie.png",
        thumbnail_path=None,
        prompt="original generated text selfie prompt",
        provider="openrouter",
        model=response_model,
        status="succeeded",
        metadata={
            "kind": "character_text_character_image",
            "text_message_id": text_message.id,
            "thread_id": thread.id,
            "character_id": character.id,
            "source_character_reference_asset_id": reference.id,
            "source_character_reference_asset_ids": [reference.id],
        },
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"openrouter": provider},
        media_dir=media_dir,
    )

    new_asset = asyncio.run(
        service.regenerate_asset_with_prompt(
            save_id=save.id,
            media_asset_id=old_asset.id,
            prompt="edited text selfie prompt",
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.model_id == requested_model
    assert request.source_media_asset_id == reference.id
    assert request.source_media_path == media_dir / reference.path
    assert new_asset.model == requested_model
    assert json.loads(new_asset.metadata_json)["requested_model_id"] == requested_model


def test_regenerate_asset_with_prompt_keeps_derived_assets_active(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    derived_video = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        source_media_asset_id=old_asset.id,
        type="video",
        path="media/old-scene.mp4",
        thumbnail_path=None,
        prompt="animate original scene",
        provider="fake-video",
        model="fake-video",
        status="succeeded",
        mime_type="video/mp4",
        metadata={
            "source_media_asset_id": old_asset.id,
            "source_media_asset_ids": [old_asset.id],
            "source_character_reference_asset_id": old_asset.id,
            "source_character_reference_asset_ids": [old_asset.id],
        },
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=tmp_path / "media",
    )

    new_asset = asyncio.run(
        service.regenerate_asset_with_prompt(
            save_id=save.id,
            media_asset_id=old_asset.id,
            prompt="edited lantern prompt",
        )
    )

    assert [asset.id for asset in repositories.list_media_assets(save.id)] == [
        derived_video.id,
        new_asset.id,
    ]
    refreshed_video = repositories.get_media_asset(
        save_id=save.id,
        media_asset_id=derived_video.id,
    )
    assert refreshed_video is not None
    assert refreshed_video.source_media_asset_id == new_asset.id
    refreshed_metadata = json.loads(refreshed_video.metadata_json)
    assert refreshed_metadata["source_media_asset_id"] == new_asset.id
    assert refreshed_metadata["source_media_asset_ids"] == [new_asset.id]
    assert refreshed_metadata["source_character_reference_asset_id"] == new_asset.id
    assert refreshed_metadata["source_character_reference_asset_ids"] == [
        new_asset.id
    ]


def test_regenerate_asset_with_prompt_keeps_old_asset_when_job_success_fails(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="media/old-scene.png",
        thumbnail_path=None,
        prompt="original generated prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=media_dir,
    )

    def fail_succeed(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("job success write failed")

    monkeypatch.setattr(service.jobs, "succeed", fail_succeed)

    with pytest.raises(RuntimeError, match="job success write failed"):
        asyncio.run(
            service.regenerate_asset_with_prompt(
                save_id=save.id,
                media_asset_id=old_asset.id,
                prompt="edited lantern prompt",
            )
        )

    assert [asset.id for asset in repositories.list_media_assets(save.id)] == [
        old_asset.id,
    ]
    assert len(repositories.list_all_media_assets(save.id)) == 1
    assert [path for path in media_dir.rglob("*") if path.is_file()] == []
    job = _media_jobs(repositories, save.id, "image_regeneration")[0]
    assert job["status"] == "failed"


def test_regenerate_text_image_with_prompt_replaces_attachment_inline(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    character = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        met=True,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=character.id,
        title=character.name,
    )
    text_message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=character.id,
        sender="character",
        body="I found the ticket stub.",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    old_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/text-ticket.png",
        thumbnail_path=None,
        prompt="close-up of a ticket stub on a phone",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={
            "kind": "character_text_object_context_image",
            "text_message_id": text_message.id,
            "thread_id": thread.id,
            "character_id": character.id,
        },
    )
    repositories.add_character_text_message_attachment(
        save_id=save.id,
        thread_id=thread.id,
        text_message_id=text_message.id,
        character_id=character.id,
        kind="object_context_image",
        status="succeeded",
        media_asset_id=old_asset.id,
        prompt="close-up of a ticket stub on a phone",
        metadata={"media_asset_id": old_asset.id},
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )

    new_asset = asyncio.run(
        service.regenerate_asset_with_prompt(
            save_id=save.id,
            media_asset_id=old_asset.id,
            prompt="edited ticket stub prompt",
        )
    )

    attachment = repositories.list_character_text_message_attachments(
        save_id=save.id,
        text_message_ids=(text_message.id,),
    )[0]
    assert provider.image_requests[0].prompt == "edited ticket stub prompt"
    assert provider.image_requests[0].source_message_id == text_message.id
    assert new_asset.source_message_id is None
    assert attachment.media_asset_id == new_asset.id
    assert [asset.id for asset in repositories.list_media_assets(save.id)] == [
        new_asset.id,
    ]


def test_generate_for_message_prompt_uses_selected_message_without_future_context(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    selected_message = messages[1]
    for index in range(10):
        repositories.append_message(
            save_id=save.id,
            role="narrator" if index % 2 else "player",
            speaker_name="Narrator" if index % 2 else "Mara",
            body=f"Future scene beat {index} that must not shape the old image.",
            provider="fake" if index % 2 else None,
            model="fake-chat" if index % 2 else None,
        )
    provider = RecordingImageProvider()
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=selected_message.id,
        )
    )

    assert len(provider.chat_requests) == 1
    chat_context = _chat_request_context(provider.chat_requests[0])
    assert len(provider.image_requests) == 1
    assert provider.image_requests[0].source_message_id == selected_message.id
    _assert_realistic_prompt(
        provider.image_requests[0].prompt,
        "cinematic drafted image prompt",
    )
    assert "Narrator: A bell rings under the span." in chat_context
    assert "Future scene beat" not in chat_context


def test_generate_for_message_prompt_omits_future_deterministic_scene_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    selected_message = messages[1]
    future_chronicle_text = (
        "FUTURE_CHRONICLE_sunspire_gate: the sealed lift opens above the bridge."
    )
    future_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=future_chronicle_text,
        provider="fake",
        model="fake-chat",
    )
    future_location = repositories.add_location(
        save_id=save.id,
        name="FUTURE_LOCATION_Sunspire Gate",
        description="FUTURE_LOCATION_DESC_copper doors overlooking noon clouds.",
        visual_description="FUTURE_LOCATION_VISUAL_gold glass ribs over white stone.",
        status="FUTURE_LOCATION_STATUS_locked-open",
        hazards=["FUTURE_LOCATION_HAZARD_solar flare glass"],
        source_message_id=future_message.id,
    )
    future_character = repositories.add_character(
        save_id=save.id,
        name="FUTURE_CHARACTER_Aurel",
        role="FUTURE_CHARACTER_ROLE_gate herald",
        visual_notes="FUTURE_CHARACTER_VISUAL_mirror mask and blue cloak.",
        location_id=future_location.id,
        met=True,
        source_message_id=future_message.id,
    )
    future_thread = repositories.add_active_thread(
        save_id=save.id,
        title="FUTURE_THREAD_sunspire alarm",
        description="FUTURE_THREAD_DESC_the gate bell calls the noon guard.",
        related_entities=[future_location.id, future_character.id],
        source_message_id=future_message.id,
    )
    future_memory = repositories.add_memory(
        save_id=save.id,
        body="FUTURE_MEMORY_gate password is ember-at-noon.",
        tags=["future", "sunspire"],
        source_message_id=future_message.id,
    )
    future_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={
            "name": "FUTURE_WORLD_STATE_location_sunspire",
            "visual": "FUTURE_WORLD_STATE_VISUAL_noon prisms on the bridge.",
        },
        category="location",
        source_message_id=future_message.id,
    )
    future_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=future_message.id,
        covers_message_end_id=future_message.id,
        body="FUTURE_SUMMARY_the bridge has already become a sun gate.",
        provider="fake",
        model="fake-chat",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=future_location.id,
        situation="FUTURE_SCENE_SITUATION_noon gate revealed above the ash span.",
        objective="FUTURE_SCENE_OBJECTIVE_cross into the sunspire lift.",
        in_world_time="FUTURE_SCENE_TIME_bright noon",
        weather="FUTURE_SCENE_WEATHER_glass rain",
        mood="FUTURE_SCENE_MOOD_triumphant glare",
        nearby_objects=["FUTURE_SCENE_OBJECT_prism key"],
        hazards=["FUTURE_SCENE_HAZARD_burning sigils"],
        present_character_ids=[future_character.id],
        source_message_id=future_message.id,
    )
    for entity_type, entity_id in (
        ("location", future_location.id),
        ("character", future_character.id),
        ("active_thread", future_thread.id),
    ):
        for target_type, target_id in (
            ("memory", future_memory.id),
            ("world_state", future_state.id),
            ("summary", future_summary.id),
        ):
            repositories.add_entity_link(
                save_id=save.id,
                entity_type=entity_type,
                entity_id=entity_id,
                target_type=target_type,
                target_id=target_id,
            )

    unprovenanced_location = repositories.add_location(
        save_id=save.id,
        name="UNPROVENANCED_FUTURE_LOCATION_Moonwell Annex",
        description="UNPROVENANCED_FUTURE_LOCATION_DESC_silver doors above the ash.",
        visual_description="UNPROVENANCED_FUTURE_LOCATION_VISUAL_moonlit marble arch.",
        status="UNPROVENANCED_FUTURE_LOCATION_STATUS_newly-open",
        hazards=["UNPROVENANCED_FUTURE_LOCATION_HAZARD_falling moon glass"],
    )
    unprovenanced_character = repositories.add_character(
        save_id=save.id,
        name="UNPROVENANCED_FUTURE_CHARACTER_Ser Vale",
        role="UNPROVENANCED_FUTURE_CHARACTER_ROLE_moonwell guide",
        visual_notes="UNPROVENANCED_FUTURE_CHARACTER_VISUAL_opal veil and lamp.",
        location_id=unprovenanced_location.id,
        met=True,
    )
    unprovenanced_thread = repositories.add_active_thread(
        save_id=save.id,
        title="UNPROVENANCED_FUTURE_THREAD_moonwell bargain",
        description="UNPROVENANCED_FUTURE_THREAD_DESC_the guide demands silver ash.",
        related_entities=[
            unprovenanced_location.id,
            unprovenanced_character.id,
        ],
    )
    unprovenanced_memory = repositories.add_memory(
        save_id=save.id,
        body="UNPROVENANCED_FUTURE_MEMORY_moonwell opens only after the bell.",
        tags=["unprovenanced", "future"],
    )
    unprovenanced_state = repositories.upsert_world_state(
        save_id=save.id,
        key="UNPROVENANCED_FUTURE_WORLD_STATE_scene.location",
        value={
            "name": "UNPROVENANCED_FUTURE_WORLD_STATE_location_moonwell",
            "visual": "UNPROVENANCED_FUTURE_WORLD_STATE_VISUAL_lunar mist bridge.",
        },
        category="location",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=unprovenanced_location.id,
        situation="UNPROVENANCED_FUTURE_SCENE_SITUATION_moonwell revealed.",
        objective="UNPROVENANCED_FUTURE_SCENE_OBJECTIVE_bargain with the guide.",
        in_world_time="UNPROVENANCED_FUTURE_SCENE_TIME_blue midnight",
        weather="UNPROVENANCED_FUTURE_SCENE_WEATHER_silver rain",
        mood="UNPROVENANCED_FUTURE_SCENE_MOOD_hushed omen",
        nearby_objects=["UNPROVENANCED_FUTURE_SCENE_OBJECT_moon key"],
        hazards=["UNPROVENANCED_FUTURE_SCENE_HAZARD_lunar glassfall"],
        present_character_ids=[unprovenanced_character.id],
    )
    for entity_type, entity_id in (
        ("location", unprovenanced_location.id),
        ("character", unprovenanced_character.id),
        ("active_thread", unprovenanced_thread.id),
    ):
        for target_type, target_id in (
            ("memory", unprovenanced_memory.id),
            ("world_state", unprovenanced_state.id),
        ):
            repositories.add_entity_link(
                save_id=save.id,
                entity_type=entity_type,
                entity_id=entity_id,
                target_type=target_type,
                target_id=target_id,
            )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=selected_message.id,
        )
    )

    assert len(provider.chat_requests) == 1
    chat_context = _chat_request_context(provider.chat_requests[0])
    assert (
        "Selected scene message:\nNarrator: A bell rings under the span."
        in chat_context
    )
    for future_text in (
        future_chronicle_text,
        "The echo answers from below.",
        "FUTURE_LOCATION_Sunspire Gate",
        "FUTURE_LOCATION_DESC_copper doors overlooking noon clouds.",
        "FUTURE_LOCATION_VISUAL_gold glass ribs over white stone.",
        "FUTURE_LOCATION_STATUS_locked-open",
        "FUTURE_LOCATION_HAZARD_solar flare glass",
        "FUTURE_CHARACTER_Aurel",
        "FUTURE_CHARACTER_ROLE_gate herald",
        "FUTURE_CHARACTER_VISUAL_mirror mask and blue cloak.",
        "FUTURE_THREAD_sunspire alarm",
        "FUTURE_THREAD_DESC_the gate bell calls the noon guard.",
        "FUTURE_MEMORY_gate password is ember-at-noon.",
        "FUTURE_WORLD_STATE_location_sunspire",
        "FUTURE_WORLD_STATE_VISUAL_noon prisms on the bridge.",
        "FUTURE_SUMMARY_the bridge has already become a sun gate.",
        "FUTURE_SCENE_SITUATION_noon gate revealed above the ash span.",
        "FUTURE_SCENE_OBJECTIVE_cross into the sunspire lift.",
        "FUTURE_SCENE_TIME_bright noon",
        "FUTURE_SCENE_WEATHER_glass rain",
        "FUTURE_SCENE_MOOD_triumphant glare",
        "FUTURE_SCENE_OBJECT_prism key",
        "FUTURE_SCENE_HAZARD_burning sigils",
        "UNPROVENANCED_FUTURE_LOCATION_Moonwell Annex",
        "UNPROVENANCED_FUTURE_LOCATION_DESC_silver doors above the ash.",
        "UNPROVENANCED_FUTURE_LOCATION_VISUAL_moonlit marble arch.",
        "UNPROVENANCED_FUTURE_LOCATION_STATUS_newly-open",
        "UNPROVENANCED_FUTURE_LOCATION_HAZARD_falling moon glass",
        "UNPROVENANCED_FUTURE_CHARACTER_Ser Vale",
        "UNPROVENANCED_FUTURE_CHARACTER_ROLE_moonwell guide",
        "UNPROVENANCED_FUTURE_CHARACTER_VISUAL_opal veil and lamp.",
        "UNPROVENANCED_FUTURE_THREAD_moonwell bargain",
        "UNPROVENANCED_FUTURE_THREAD_DESC_the guide demands silver ash.",
        "UNPROVENANCED_FUTURE_MEMORY_moonwell opens only after the bell.",
        "UNPROVENANCED_FUTURE_WORLD_STATE_scene.location",
        "UNPROVENANCED_FUTURE_WORLD_STATE_location_moonwell",
        "UNPROVENANCED_FUTURE_WORLD_STATE_VISUAL_lunar mist bridge.",
        "UNPROVENANCED_FUTURE_SCENE_SITUATION_moonwell revealed.",
        "UNPROVENANCED_FUTURE_SCENE_OBJECTIVE_bargain with the guide.",
        "UNPROVENANCED_FUTURE_SCENE_TIME_blue midnight",
        "UNPROVENANCED_FUTURE_SCENE_WEATHER_silver rain",
        "UNPROVENANCED_FUTURE_SCENE_MOOD_hushed omen",
        "UNPROVENANCED_FUTURE_SCENE_OBJECT_moon key",
        "UNPROVENANCED_FUTURE_SCENE_HAZARD_lunar glassfall",
    ):
        assert future_text not in chat_context


def test_generate_for_latest_message_includes_unprovenanced_current_scene_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    selected_message = messages[-1]
    current_location = repositories.add_location(
        save_id=save.id,
        name="CURRENT_UNPROVENANCED_LOCATION_Echo Undercroft",
        description="CURRENT_UNPROVENANCED_LOCATION_DESC_bell roots below ash.",
        visual_description="CURRENT_UNPROVENANCED_LOCATION_VISUAL_green bells in fog.",
        status="CURRENT_UNPROVENANCED_LOCATION_STATUS_open",
        hazards=["CURRENT_UNPROVENANCED_LOCATION_HAZARD_echo tide"],
    )
    current_character = repositories.add_character(
        save_id=save.id,
        name="CURRENT_UNPROVENANCED_CHARACTER_Lysa",
        role="CURRENT_UNPROVENANCED_CHARACTER_ROLE_echo keeper",
        visual_notes="CURRENT_UNPROVENANCED_CHARACTER_VISUAL_copper mask and lantern.",
        location_id=current_location.id,
        met=True,
    )
    current_thread = repositories.add_active_thread(
        save_id=save.id,
        title="CURRENT_UNPROVENANCED_THREAD_answering echo",
        description="CURRENT_UNPROVENANCED_THREAD_DESC_the bell asks for a name.",
        related_entities=[current_location.id, current_character.id],
    )
    current_memory = repositories.add_memory(
        save_id=save.id,
        body="CURRENT_UNPROVENANCED_MEMORY_echo names the debt beneath the bridge.",
        tags=["current", "echo"],
    )
    current_state = repositories.upsert_world_state(
        save_id=save.id,
        key="CURRENT_UNPROVENANCED_WORLD_STATE_scene.location",
        value={
            "name": "CURRENT_UNPROVENANCED_WORLD_STATE_location_undercroft",
            "visual": "CURRENT_UNPROVENANCED_WORLD_STATE_VISUAL_lantern smoke below.",
        },
        category="location",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="CURRENT_UNPROVENANCED_SCENE_SITUATION_echo answers below.",
        objective="CURRENT_UNPROVENANCED_SCENE_OBJECTIVE_follow the bell rope.",
        in_world_time="CURRENT_UNPROVENANCED_SCENE_TIME_blue evening",
        weather="CURRENT_UNPROVENANCED_SCENE_WEATHER_ash drizzle",
        mood="CURRENT_UNPROVENANCED_SCENE_MOOD_waiting omen",
        nearby_objects=["CURRENT_UNPROVENANCED_SCENE_OBJECT_bronze clapper"],
        hazards=["CURRENT_UNPROVENANCED_SCENE_HAZARD_resonant drop"],
        present_character_ids=[current_character.id],
    )
    for entity_type, entity_id in (
        ("location", current_location.id),
        ("character", current_character.id),
        ("active_thread", current_thread.id),
    ):
        for target_type, target_id in (
            ("memory", current_memory.id),
            ("world_state", current_state.id),
        ):
            repositories.add_entity_link(
                save_id=save.id,
                entity_type=entity_type,
                entity_id=entity_id,
                target_type=target_type,
                target_id=target_id,
            )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=selected_message.id,
        )
    )

    assert len(provider.chat_requests) == 1
    chat_context = _chat_request_context(provider.chat_requests[0])
    assert (
        "Selected scene message:\nNarrator: The echo answers from below."
        in chat_context
    )
    for current_text in (
        "CURRENT_UNPROVENANCED_LOCATION_Echo Undercroft",
        "CURRENT_UNPROVENANCED_LOCATION_VISUAL_green bells in fog.",
        "CURRENT_UNPROVENANCED_LOCATION_STATUS_open",
        "CURRENT_UNPROVENANCED_LOCATION_HAZARD_echo tide",
        "CURRENT_UNPROVENANCED_CHARACTER_Lysa",
        "CURRENT_UNPROVENANCED_CHARACTER_VISUAL_copper mask and lantern.",
        "CURRENT_UNPROVENANCED_THREAD_answering echo",
        "CURRENT_UNPROVENANCED_THREAD_DESC_the bell asks for a name.",
        "CURRENT_UNPROVENANCED_MEMORY_echo names the debt beneath the bridge.",
        "CURRENT_UNPROVENANCED_WORLD_STATE_scene.location",
        "CURRENT_UNPROVENANCED_WORLD_STATE_location_undercroft",
        "CURRENT_UNPROVENANCED_WORLD_STATE_VISUAL_lantern smoke below.",
        "CURRENT_UNPROVENANCED_SCENE_SITUATION_echo answers below.",
        "CURRENT_UNPROVENANCED_SCENE_OBJECTIVE_follow the bell rope.",
        "CURRENT_UNPROVENANCED_SCENE_TIME_blue evening",
        "CURRENT_UNPROVENANCED_SCENE_WEATHER_ash drizzle",
        "CURRENT_UNPROVENANCED_SCENE_MOOD_waiting omen",
        "CURRENT_UNPROVENANCED_SCENE_OBJECT_bronze clapper",
        "CURRENT_UNPROVENANCED_SCENE_HAZARD_resonant drop",
    ):
        assert current_text in chat_context
    assert "CURRENT_UNPROVENANCED_CHARACTER_ROLE_echo keeper" not in chat_context


def test_generate_for_message_context_marks_prior_image_as_continuity_only(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    selected_message = messages[-1]
    future_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A future lighthouse reveals the bridge from above.",
        provider="fake",
        model="fake-chat",
    )
    repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="latest-prior.png",
        prompt=(
            "Opening-scene prompt that must not be reused as the current scene: "
            "wide shot of the first ash bridge, cinders, bell tower, dawn fog."
        ),
        provider="continuity",
        model="continuity-image",
        status="succeeded",
        source_message_id=messages[2].id,
    )
    repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="regenerated-older-prior.png",
        prompt="regenerated later older prior prompt must not appear",
        provider="fake",
        model="fake-image",
        status="succeeded",
        source_message_id=messages[0].id,
    )
    repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="failed-prior.png",
        prompt="failed prior prompt should not appear",
        provider="fake",
        model="fake-image",
        status="failed",
        source_message_id=messages[1].id,
    )
    future_asset = repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="future.png",
        prompt="future prompt must never leak",
        provider="fake",
        model="fake-image",
        status="succeeded",
        source_message_id=future_message.id,
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=selected_message.id,
        )
    )

    chat_context = _chat_request_context(provider.chat_requests[0])
    assert "Prior image continuity before selected moment:" in chat_context
    assert f"source_message_id: {messages[2].id}" in chat_context
    assert "Reuse only stable visual continuity" in chat_context
    assert "Prior image prompt:" not in chat_context
    assert "prior visible prompt:" not in chat_context
    assert "Opening-scene prompt that must not be reused" not in chat_context
    assert "wide shot of the first ash bridge" not in chat_context
    assert "prior image model: continuity/continuity-image" in chat_context
    assert "regenerated later older prior prompt must not appear" not in chat_context
    assert "failed prior prompt should not appear" not in chat_context
    assert "future prompt must never leak" not in chat_context
    assert f"source_message_id: {future_asset.source_message_id}" not in chat_context
    assert "A future lighthouse reveals the bridge from above." not in chat_context


def test_generate_for_message_fixed_budget_keeps_selected_message(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    selected_message = messages[-1]
    repositories.set_app_setting("context_budget_mode", "fixed_chars")
    repositories.set_app_setting("context_budget_fixed_total_chars", 1)
    provider = RecordingImageProvider()
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=selected_message.id,
        )
    )

    chat_context = _chat_request_context(provider.chat_requests[0])
    assert "Narrator: The echo answers from below." in chat_context


def test_generate_for_message_rejects_provider_image_path_outside_media_dir(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    media_dir = tmp_path / "media"
    source_bytes = b"provider path scene bytes"
    traversal_source = tmp_path / "outside.png"
    traversal_source.write_bytes(source_bytes)
    provider = RelativePathImageProvider(Path("../outside.png"))
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    with pytest.raises(ValueError, match="escapes media directory"):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert repositories.list_media_assets(save.id) == []
    assert traversal_source.read_bytes() == source_bytes
    assert [path for path in media_dir.rglob("*") if path.is_file()] == []

    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    result = jobs[0]["result"]
    assert result["classification"] == "primary_image_not_stored"
    assert result["fallback_used"] is False
    assert result["fallback_skipped_reason"] == "no_fallback_model"
    assert result["primary_error_message"] == (
        "Resolved image path escapes media directory"
    )
    assert "escapes media directory" in jobs[0]["error"]


def test_generate_for_message_copies_provider_image_path_under_media_dir(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    media_dir = tmp_path / "media"
    source_path = media_dir / "provider-output" / "scene.png"
    source_bytes = _VALID_PNG_BYTES
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source_bytes)
    provider = RelativePathImageProvider(Path("provider-output/scene.png"))
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    asset_path = _asset_path(media_dir, asset.path)
    assert asset_path.resolve().is_relative_to(media_dir.resolve())
    assert asset_path.read_bytes() == source_bytes
    _assert_private_modes(asset_path)
    assert source_path.read_bytes() == source_bytes

    media_assets = repositories.list_media_assets(save.id)
    assert [item.id for item in media_assets] == [asset.id]
    assert media_assets[0].path == asset.path
    assert media_assets[0].thumbnail_path == asset.thumbnail_path
    _assert_private_thumbnail_if_present(media_dir=media_dir, asset=media_assets[0])

    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["result"]["path"] == asset.path


def test_generate_for_message_fails_when_provider_references_missing_output_path(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    media_dir = tmp_path / "media"
    derived_relative_path = Path(save.id) / f"{messages[-1].id}.png"
    output_path = media_dir / derived_relative_path
    provider = RelativePathImageProvider(derived_relative_path)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    with pytest.raises(ValueError, match="missing image file"):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert repositories.list_media_assets(save.id) == []
    assert not output_path.exists()
    assert [path for path in media_dir.rglob("*") if path.is_file()] == []

    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    result = jobs[0]["result"]
    assert result["classification"] == "primary_image_not_stored"
    assert result["fallback_used"] is False
    assert result["fallback_skipped_reason"] == "no_fallback_model"
    assert result["primary_error_message"] == (
        "Image provider returned a missing image file"
    )
    assert "missing image file" in jobs[0]["error"]


def test_generate_for_message_sanitizes_exact_dot_segment_ids(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, message = _save_with_custom_ids_and_image_preference(
        repositories=repositories,
        save_id="..",
        message_id="..",
    )
    provider = RecordingImageProvider(b"exact dot segment scene bytes")
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=message.id,
        )
    )

    persisted_path = Path(asset.path)
    assert "." not in persisted_path.parts
    assert ".." not in persisted_path.parts
    asset_path = _asset_path(media_dir, asset.path)
    assert asset_path.resolve().is_relative_to(media_dir.resolve())
    assert asset_path.read_bytes() == b"exact dot segment scene bytes"

    media_assets = repositories.list_media_assets(save.id)
    assert [item.id for item in media_assets] == [asset.id]
    assert Path(media_assets[0].path).parts == persisted_path.parts

    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["result"]["path"] == asset.path


def test_generate_for_message_sanitizes_traversal_segments_from_save_and_message_ids(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, message = _save_with_custom_ids_and_image_preference(
        repositories=repositories,
        save_id="../escape-save",
        message_id="../../escape-message",
    )
    provider = RecordingImageProvider(b"safe id scene bytes")
    media_dir = tmp_path / "media"
    outside_candidates = [
        tmp_path / "escape-save",
        tmp_path.parent / "escape-message.png",
    ]
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=message.id,
        )
    )

    persisted_path = Path(asset.path)
    asset_path = _asset_path(media_dir, asset.path)
    assert ".." not in persisted_path.parts
    assert asset_path.resolve().is_relative_to(media_dir.resolve())
    assert asset_path.read_bytes() == b"safe id scene bytes"
    assert all(not path.exists() for path in outside_candidates)

    media_assets = repositories.list_media_assets(save.id)
    assert [item.id for item in media_assets] == [asset.id]
    assert ".." not in Path(media_assets[0].path).parts

    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["result"]["path"] == asset.path


@pytest.mark.parametrize("auto_frequency", [_UNSET, 3])
def test_automatic_generation_runs_only_when_narrator_count_reaches_frequency(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    auto_frequency: object,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    provider = RecordingImageProvider()
    service = _media_service(
        repositories=repositories,
        provider=provider,
        media_dir=tmp_path / "media",
        auto_frequency=auto_frequency,
    )

    result = asyncio.run(service.generate_automatic_if_due(save_id=save.id))

    assert result is None
    assert provider.image_requests == []
    assert repositories.list_media_assets(save.id) == []
    assert _image_generation_jobs(repositories, save.id) == []

    latest_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oath-brands glow under the bridge stones.",
        provider="fake",
        model="fake-chat",
        token_estimate=51,
    )

    asset = asyncio.run(service.generate_automatic_if_due(save_id=save.id))

    assert asset is not None
    assert len(provider.chat_requests) == 1
    assert len(provider.image_requests) == 1
    assert provider.image_requests[0].source_message_id == latest_narrator.id
    assert "The oath-brands glow under the bridge stones." in _chat_request_context(
        provider.chat_requests[0]
    )
    _assert_realistic_prompt(
        provider.image_requests[0].prompt,
        "cinematic drafted image prompt",
    )
    assert repositories.list_media_assets(save.id)[0].source_message_id == (
        latest_narrator.id
    )


def test_automatic_generation_skips_fade_transition_sources(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    source = _mark_message_as_fade_transition(
        repositories,
        save_id=save.id,
        message_id=messages[-1].id,
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = _media_service(
        repositories=repositories,
        provider=provider,
        media_dir=tmp_path / "media",
        auto_frequency=2,
    )

    assert service.prepare_automatic_if_due(
        save_id=save.id,
        source_message_id=source.id,
    ) is None

    assert provider.chat_requests == []
    assert provider.image_requests == []
    assert _image_generation_jobs(repositories, save.id) == []
    assert repositories.list_media_assets(save.id) == []


def test_automatic_generation_uses_deferred_source_message_ordinal(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    third_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Third narrator beat should trigger the image.",
        provider="fake",
        model="fake-chat",
        token_estimate=51,
    )
    fourth_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Fourth narrator beat exists before the delayed job runs.",
        provider="fake",
        model="fake-chat",
        token_estimate=52,
    )
    provider = RecordingImageProvider()
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        automatic_enabled=True,
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_automatic_if_due(
            save_id=save.id,
            source_message_id=third_narrator.id,
        )
    )

    assert asset is not None
    assert len(provider.chat_requests) == 1
    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.source_message_id == third_narrator.id
    chat_context = _chat_request_context(provider.chat_requests[0])
    assert "Third narrator beat should trigger the image." in chat_context
    assert "Fourth narrator beat exists before the delayed job runs." not in (
        chat_context
    )
    _assert_realistic_prompt(request.prompt, "cinematic drafted image prompt")
    assert repositories.list_media_assets(save.id)[0].source_message_id == (
        third_narrator.id
    )
    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["payload"]["source_message_id"] == third_narrator.id
    assert jobs[0]["payload"]["source_message_id"] != fourth_narrator.id
    narrator_ids = [
        message.id
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    ]
    assert narrator_ids == [
        messages[1].id,
        messages[3].id,
        third_narrator.id,
        fourth_narrator.id,
    ]


def test_generate_prepared_automatic_uses_context_captured_during_prepare(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    provider = RecordingImageProvider()
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        automatic_enabled=True,
        auto_frequency=2,
    )

    prepared = service.prepare_automatic_if_due(
        save_id=save.id,
        source_message_id=messages[-1].id,
    )
    assert prepared is not None
    assert prepared.source_message_id == messages[-1].id
    assert "POST_PREPARE_SCENE_MUTATION" not in prepared.scene_context

    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="POST_PREPARE_SCENE_MUTATION the lens cracks after prepare.",
        objective="POST_PREPARE_OBJECTIVE choose a new path after prepare.",
        in_world_time="POST_PREPARE_TIME midnight after prepare.",
        weather="POST_PREPARE_WEATHER red ash after prepare.",
        mood="POST_PREPARE_MOOD alarm after prepare.",
        nearby_objects=["POST_PREPARE_OBJECT fractured lens"],
        hazards=["POST_PREPARE_HAZARD falling glass"],
    )
    repositories.add_memory(
        save_id=save.id,
        body="POST_PREPARE_MEMORY this should not shape the prepared image.",
        tags=["post-prepare"],
    )

    asset = asyncio.run(service.generate_prepared_automatic(prepared))

    assert asset is not None
    assert len(provider.chat_requests) == 1
    chat_context = _chat_request_context(provider.chat_requests[0])
    assert "Narrator: The echo answers from below." in chat_context
    for post_prepare_text in (
        "POST_PREPARE_SCENE_MUTATION",
        "POST_PREPARE_OBJECTIVE",
        "POST_PREPARE_TIME",
        "POST_PREPARE_WEATHER",
        "POST_PREPARE_MOOD",
        "POST_PREPARE_OBJECT",
        "POST_PREPARE_HAZARD",
        "POST_PREPARE_MEMORY",
    ):
        assert post_prepare_text not in chat_context
    assert provider.image_requests[0].source_message_id == messages[-1].id
    assert repositories.list_media_assets(save.id)[0].source_message_id == (
        messages[-1].id
    )


def test_generate_prepared_automatic_rechecks_fade_transition_source(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = _media_service(
        repositories=repositories,
        provider=provider,
        media_dir=tmp_path / "media",
        auto_frequency=2,
    )

    prepared = service.prepare_automatic_if_due(
        save_id=save.id,
        source_message_id=messages[-1].id,
    )
    assert prepared is not None

    _mark_message_as_fade_transition(
        repositories,
        save_id=save.id,
        message_id=messages[-1].id,
    )

    assert asyncio.run(service.generate_prepared_automatic(prepared)) is None
    assert provider.chat_requests == []
    assert provider.image_requests == []
    assert _image_generation_jobs(repositories, save.id) == []
    assert repositories.list_media_assets(save.id) == []


def test_generate_prepared_automatic_rejects_unavailable_image_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-image",
        display_name="Fake Image",
        capabilities=["image_generation"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )
    provider = RecordingImageProvider()
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        automatic_enabled=True,
        auto_frequency=2,
    )
    prepared = service.prepare_automatic_if_due(
        save_id=save.id,
        source_message_id=messages[-1].id,
    )
    assert prepared is not None

    with pytest.raises(ValueError, match="Image generation model is unavailable"):
        asyncio.run(service.generate_prepared_automatic(prepared))

    assert provider.chat_requests == []
    assert provider.image_requests == []
    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["error"] == "Image generation model is unavailable: fake-image"


def test_automatic_generation_skips_text_message_beats(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _messages = _save_with_image_preference(repositories)
    text_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=(
            "Your phone buzzes after a minute.\n\n"
            "---\n\n"
            "**Jade:** Hey player character. Glad I met you too. See you tomorrow."
        ),
        provider="fake",
        model="fake-chat",
        token_estimate=50,
    )
    provider = RecordingImageProvider()
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        automatic_enabled=True,
        auto_frequency=3,
    )

    prepared = service.prepare_automatic_if_due(
        save_id=save.id,
        source_message_id=text_message.id,
    )
    asset = asyncio.run(
        service.generate_automatic_if_due(
            save_id=save.id,
            source_message_id=text_message.id,
        )
    )

    assert prepared is None
    assert asset is None
    assert provider.chat_requests == []
    assert provider.image_requests == []
    assert repositories.list_media_assets(save.id) == []
    assert _image_generation_jobs(repositories, save.id) == []


def test_automatic_generation_can_create_video_without_image_duplicate_blocking(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_app_setting("automatic_media_mode", "video")
    repositories.set_model_preference(
        task="video_generation",
        provider="fake",
        model_id="fake-video",
    )
    repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="existing-image.png",
        prompt="already generated still image",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    provider = RecordingVideoProvider(_VALID_MP4_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        automatic_enabled=True,
        auto_frequency=2,
    )

    asset = asyncio.run(service.generate_automatic_if_due(save_id=save.id))
    second = asyncio.run(service.generate_automatic_if_due(save_id=save.id))

    assert asset is not None
    assert asset.type == "video"
    assert asset.mime_type == "video/mp4"
    assert asset.source_message_id == messages[-1].id
    assert asset.path.endswith(".mp4")
    assert second is None
    assert provider.image_requests == []
    assert len(provider.video_requests) == 1
    assert provider.video_requests[0].provider == "fake"
    assert provider.video_requests[0].model_id == "fake-video"
    assert provider.video_requests[0].source_message_id == messages[-1].id
    media_assets = repositories.list_media_assets(save.id)
    assert [item.type for item in media_assets] == ["image", "video"]
    jobs = _video_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["payload"]["job_context"] == "automatic_post_turn"
    assert jobs[0]["result"]["media_asset_id"] == asset.id
    assert jobs[0]["result"]["mime_type"] == "video/mp4"


def test_automatic_video_generation_labels_openrouter_request(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_app_setting("automatic_media_mode", "video")
    repositories.set_model_preference(
        task="video_generation",
        provider="openrouter",
        model_id="fake-video",
    )
    repositories.set_model_preference(
        task="image_prompt",
        provider="openrouter",
        model_id="fake-chat",
    )
    provider = RecordingVideoProvider(_VALID_MP4_BYTES)
    provider.provider_name = "openrouter"
    service = MediaService(
        repositories=repositories,
        providers={"openrouter": provider},
        media_dir=tmp_path / "media",
        automatic_enabled=True,
        auto_frequency=2,
    )

    asset = asyncio.run(service.generate_automatic_if_due(save_id=save.id))

    assert asset is not None
    assert len(provider.video_requests) == 1
    assert (
        provider.video_requests[0].openrouter_app_title
        == "Bragi"
    )


def test_automatic_generation_rejects_unavailable_video_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_app_setting("automatic_media_mode", "video")
    repositories.set_model_preference(
        task="video_generation",
        provider="fake",
        model_id="fake-video",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-video",
        display_name="Fake Video",
        capabilities=["text_to_video"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )
    provider = RecordingVideoProvider(_VALID_MP4_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        automatic_enabled=True,
        auto_frequency=2,
    )

    with pytest.raises(ValueError, match="Video generation model is unavailable"):
        asyncio.run(
            service.generate_automatic_if_due(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert provider.chat_requests == []
    assert provider.video_requests == []
    jobs = _video_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["error"] == "Video generation model is unavailable: fake-video"


@pytest.mark.parametrize(
    ("supported_parameters", "expected_safe_mode"),
    [
        (["image_safe_mode"], False),
        ([], None),
    ],
)
def test_generate_video_gates_venice_safe_mode_by_video_model_metadata(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    supported_parameters: list[str],
    expected_safe_mode: bool | None,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_app_setting("venice_image_safe_mode", False)
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/video",
        display_name="Venice Video",
        capabilities=["text_to_video"],
        supported_parameters=supported_parameters,
    )
    repositories.set_model_preference(
        task="video_generation",
        provider="venice",
        model_id="venice/video",
    )
    prompt_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    video_provider = SequenceVideoProvider(
        provider_name="venice",
        outcomes=[
            VideoResponse(
                provider="venice",
                model_id="venice/video",
                mime_type="video/mp4",
                video_bytes=_VALID_MP4_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": prompt_provider, "venice": video_provider},
        media_dir=tmp_path / "media",
        auto_frequency=2,
    )

    asset = asyncio.run(
        service.generate_video_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert asset.type == "video"
    assert len(video_provider.video_requests) == 1
    assert video_provider.video_requests[0].safe_mode is expected_safe_mode
    jobs = _video_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    if expected_safe_mode is None:
        assert "venice_safe_mode" not in jobs[0]["payload"]
        assert "primary_venice_safe_mode" not in jobs[0]["result"]
    else:
        assert jobs[0]["payload"]["venice_safe_mode"] is expected_safe_mode
        assert jobs[0]["result"]["primary_venice_safe_mode"] is expected_safe_mode


def test_generate_video_forces_venice_safe_mode_for_child_without_model_metadata(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    save, messages = _save_with_image_preference(repositories)
    repositories.set_app_setting("venice_image_safe_mode", False)
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/video",
        display_name="Venice Video",
        capabilities=["text_to_video"],
        supported_parameters=[],
    )
    repositories.set_model_preference(
        task="video_generation",
        provider="venice",
        model_id="venice/video",
    )
    video_provider = SequenceVideoProvider(
        provider_name="venice",
        outcomes=[
            VideoResponse(
                provider="venice",
                model_id="venice/video",
                mime_type="video/mp4",
                video_bytes=_VALID_MP4_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": RecordingImageProvider(_VALID_PNG_BYTES),
            "venice": video_provider,
        },
        media_dir=tmp_path / "media",
    )

    asyncio.run(
        service.generate_video_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
            current_user_id=child.id,
        )
    )

    assert video_provider.video_requests[0].safe_mode is True


def test_generate_video_rejects_non_safe_mode_provider_for_child_account(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    save, messages = _save_with_image_preference(repositories)
    repositories.save_provider_model(
        provider="openrouter",
        model_id="vendor/video",
        display_name="OpenRouter Video",
        capabilities=["text_to_video"],
    )
    repositories.set_model_preference(
        task="video_generation",
        provider="openrouter",
        model_id="vendor/video",
    )
    video_provider = RecordingVideoProvider(_VALID_MP4_BYTES)
    video_provider.provider_name = "openrouter"
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": RecordingImageProvider(_VALID_PNG_BYTES),
            "openrouter": video_provider,
        },
        media_dir=tmp_path / "media",
    )

    with pytest.raises(ValueError, match="enforced safe mode"):
        asyncio.run(
            service.generate_video_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
                current_user_id=child.id,
            )
        )

    assert video_provider.video_requests == []


def test_animate_image_generates_video_linked_to_source_media_asset(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_animation",
        provider="fake",
        model_id="fake-image-video",
    )
    media_dir = tmp_path / "media"
    source_path = media_dir / "source-image.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_VALID_PNG_BYTES)
    source_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="source-image.png",
        prompt="still frame of a bell under the ash bridge",
        provider="fake",
        model="fake-image",
        status="succeeded",
        mime_type="image/png",
        metadata={"content_rating": "g"},
    )
    provider = RecordingVideoProvider(_VALID_MP4_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=2,
    )

    asset = asyncio.run(
        service.animate_image(
            save_id=save.id,
            media_asset_id=source_image.id,
            motion_prompt="make the bell swing once",
        )
    )

    assert asset.type == "video"
    assert asset.source_message_id == messages[-1].id
    assert asset.source_media_asset_id == source_image.id
    assert asset.mime_type == "video/mp4"
    assert len(provider.video_requests) == 1
    request = provider.video_requests[0]
    assert request.model_id == "fake-image-video"
    assert request.source_media_asset_id == source_image.id
    assert request.source_media_path == source_path
    assert "make the bell swing once" in request.prompt
    jobs = _media_jobs(repositories, save.id, "image_animation")
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["payload"]["source_media_asset_id"] == source_image.id


def test_animate_image_rejects_source_message_above_actor_ceiling(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.connection.execute(
        "UPDATE messages SET content_rating = 'r' WHERE id = ?",
        (messages[-1].id,),
    )
    repositories.commit()
    repositories.set_model_preference(
        task="image_animation",
        provider="fake",
        model_id="fake-image-video",
    )
    media_dir = tmp_path / "media"
    source_path = media_dir / "restricted-source.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_VALID_PNG_BYTES)
    source_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="restricted-source.png",
        prompt="Benign source prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"content_rating": "g"},
    )
    provider = RecordingVideoProvider(_VALID_MP4_BYTES)

    with pytest.raises(
        ValueError,
        match="Source media exceeds the selected content rating",
    ):
        asyncio.run(
            MediaService(
                repositories=repositories,
                providers={"fake": provider},
                media_dir=media_dir,
            ).animate_image(
                save_id=save.id,
                media_asset_id=source_image.id,
            )
        )

    assert provider.video_requests == []


def test_animate_image_compacts_venice_prompt_before_provider_call(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_animation",
        provider="venice",
        model_id="wan-2-7-reference-to-video",
    )
    media_dir = tmp_path / "media"
    source_path = media_dir / "source-image.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_VALID_PNG_BYTES)
    source_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="source-image.png",
        prompt="still frame " + ("of the glowing bell " * 400) + "under ash",
        provider="fake",
        model="fake-image",
        status="succeeded",
        mime_type="image/png",
        metadata={"content_rating": "g"},
    )
    provider = RecordingVideoProvider(_VALID_MP4_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"venice": provider},
        media_dir=media_dir,
        auto_frequency=2,
    )

    asset = asyncio.run(
        service.animate_image(
            save_id=save.id,
            media_asset_id=source_image.id,
            motion_prompt="make the bell swing once",
        )
    )

    request = provider.video_requests[0]
    assert len(request.prompt) <= 2400
    assert "make the bell swing once" in request.prompt
    assert "still frame" in request.prompt
    assert "of the glowing bell " * 400 not in request.prompt
    assert asset.prompt == request.prompt
    jobs = _media_jobs(repositories, save.id, "image_animation")
    assert jobs[0]["payload"]["prompt_chars"] == len(request.prompt)
    assert jobs[0]["result"]["prompt_chars"] == len(request.prompt)


def test_animate_image_rejects_missing_source_file_before_provider_call(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_animation",
        provider="fake",
        model_id="fake-image-video",
    )
    source_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="missing-source-image.png",
        prompt="still frame of a bell under the ash bridge",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"content_rating": "g"},
        mime_type="image/png",
    )
    provider = RecordingVideoProvider(_VALID_MP4_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )

    with pytest.raises(ValueError, match="Source image is unavailable"):
        asyncio.run(
            service.animate_image(
                save_id=save.id,
                media_asset_id=source_image.id,
                motion_prompt="make the bell swing once",
            )
        )

    assert provider.video_requests == []
    jobs = _media_jobs(repositories, save.id, "image_animation")
    assert jobs == []


def test_animate_image_failed_job_records_provider_validation_details(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_animation",
        provider="venice",
        model_id="wan-2-7-reference-to-video",
    )
    media_dir = tmp_path / "media"
    source_path = media_dir / "source-image.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_VALID_PNG_BYTES)
    source_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="source-image.png",
        prompt="still frame of a bell under the ash bridge",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"content_rating": "g"},
        mime_type="image/png",
    )
    provider = SequenceVideoProvider(
        provider_name="venice",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.PROVIDER_ERROR,
                (
                    "Venice video queue validation failed: prompt must be less "
                    "than 2500 characters; At least one reference is required"
                ),
                status_code=400,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"venice": provider},
        media_dir=media_dir,
    )

    with pytest.raises(ValueError, match="prompt must be less than 2500"):
        asyncio.run(
            service.animate_image(
                save_id=save.id,
                media_asset_id=source_image.id,
                motion_prompt="make the bell swing once",
            )
        )

    assert len(provider.video_requests) == 1
    jobs = _media_jobs(repositories, save.id, "image_animation")
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert "prompt must be less than 2500" in jobs[0]["error"]
    result = jobs[0]["result"]
    assert result["primary_http_status"] == 400
    assert "prompt must be less than 2500" in result["primary_error_message"]
    assert "At least one reference is required" in result["final_error_message"]


def test_automatic_generation_is_disabled_by_default(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _messages = _save_with_image_preference(repositories)
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=1,
    )

    result = asyncio.run(service.generate_automatic_if_due(save_id=save.id))

    assert result is None
    assert provider.chat_requests == []
    assert provider.image_requests == []
    assert repositories.list_media_assets(save.id) == []
    assert _image_generation_jobs(repositories, save.id) == []


def test_generate_for_message_ignores_automatic_disabled_setting(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        automatic_enabled=False,
        auto_frequency=1,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(provider.image_requests) == 1
    assert [item.id for item in repositories.list_media_assets(save.id)] == [asset.id]
    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"


def test_automatic_generation_frequency_zero_is_disabled(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _messages = _save_with_image_preference(repositories)
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        automatic_enabled=True,
        auto_frequency=0,
    )

    result = asyncio.run(service.generate_automatic_if_due(save_id=save.id))

    assert result is None
    assert provider.chat_requests == []
    assert provider.image_requests == []
    assert repositories.list_media_assets(save.id) == []
    assert _image_generation_jobs(repositories, save.id) == []


def test_provider_failure_marks_job_failed_without_creating_asset_or_file(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    media_dir = tmp_path / "media"
    provider = FailingImageProvider()
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    with pytest.raises(Exception, match="image backend unavailable"):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert repositories.list_media_assets(save.id) == []
    assert list(media_dir.rglob("*")) == []
    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    result = jobs[0]["result"]
    assert result["classification"] == "primary_image_generation_failed"
    assert result["fallback_used"] is False
    assert result["fallback_skipped_reason"] == "no_fallback_model"
    assert result["primary_error_category"] == (
        ProviderErrorCategory.IMAGE_GENERATION_FAILED.value
    )
    assert "image backend unavailable" in jobs[0]["error"]


def test_generate_retries_image_fallback_for_blocked_venice_headers(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_generation",
        provider="venice",
        model_id="venice/safe-image",
    )
    _configure_image_fallback(repositories, enabled=True)
    prompt_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    primary_provider = SequenceImageProvider(
        provider_name="venice",
        outcomes=[
            ImageResponse(
                provider="venice",
                model_id="venice/safe-image",
                raw_metadata={
                    "_bragi_headers": {
                        "x-request-id": "primary-req",
                        "x-venice-is-blurred": "true",
                        "x-venice-is-content-violation": "true",
                        "authorization": "Bearer sk-primary-leak",
                    },
                    "prompt": "leaked primary prompt bytes",
                    "body": "leaked primary response body",
                },
            )
        ],
    )
    fallback_provider = SequenceImageProvider(
        provider_name="fallback",
        outcomes=[
            ImageResponse(
                provider="fallback",
                model_id="fallback/image",
                image_bytes=_VALID_PNG_BYTES,
                raw_metadata={
                    "_bragi_headers": {
                        "x-request-id": "fallback-req",
                        "authorization": "Bearer sk-fallback-leak",
                    },
                    "prompt": "leaked fallback prompt bytes",
                    "body": "leaked fallback response body",
                },
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": prompt_provider,
            "venice": primary_provider,
            "fallback": fallback_provider,
        },
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(primary_provider.image_requests) == 1
    assert primary_provider.image_requests[0].safe_mode is True
    assert len(fallback_provider.image_requests) == 1
    assert fallback_provider.image_requests[0].provider == "fallback"
    assert fallback_provider.image_requests[0].model_id == "fallback/image"
    assert fallback_provider.image_requests[0].safe_mode is None
    assert asset.provider == "fallback"
    assert asset.model == "fallback/image"
    media_assets = repositories.list_media_assets(save.id)
    assert media_assets[0].provider == "fallback"
    assert media_assets[0].model == "fallback/image"

    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["payload"]["venice_safe_mode"] is True
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["classification"] == "suspected_blocked_image_output"
    assert result["fallback_used"] is True
    assert result["primary_venice_safe_mode"] is True
    assert result["original_provider"] == "venice"
    assert result["original_model"] == "venice/safe-image"
    assert result["fallback_provider"] == "fallback"
    assert result["fallback_model"] == "fallback/image"
    assert result["final_provider"] == "fallback"
    assert result["final_model"] == "fallback/image"
    assert result["primary_provider_headers"] == {
        "x-request-id": "primary-req",
        "x-venice-is-blurred": "true",
        "x-venice-is-content-violation": "true",
    }
    assert result["provider_headers"] == {"x-request-id": "fallback-req"}
    result_repr = repr(result)
    for leaked_text in (
        "sk-primary-leak",
        "sk-fallback-leak",
        "leaked primary prompt bytes",
        "leaked primary response body",
        "leaked fallback prompt bytes",
        "leaked fallback response body",
    ):
        assert leaked_text not in result_repr


def test_child_image_generation_does_not_fallback_to_provider_without_safe_mode(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_generation",
        provider="venice",
        model_id="venice/safe-image",
    )
    _configure_image_fallback(repositories, enabled=True)
    primary_provider = SequenceImageProvider(
        provider_name="venice",
        outcomes=[
            ImageResponse(
                provider="venice",
                model_id="venice/safe-image",
                raw_metadata={
                    "_bragi_headers": {
                        "x-venice-is-content-violation": "true",
                    },
                },
            )
        ],
    )
    fallback_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": RecordingImageProvider(_VALID_PNG_BYTES),
            "venice": primary_provider,
            "fallback": fallback_provider,
        },
        media_dir=tmp_path / "media",
    )

    with pytest.raises(ValueError, match="Image provider returned no image data"):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
                current_user_id=child.id,
            )
        )

    assert primary_provider.image_requests[0].safe_mode is True
    assert fallback_provider.image_requests == []


def test_generate_uses_image_fallback_when_toggle_false(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_generation",
        provider="venice",
        model_id="venice/safe-image",
    )
    _configure_image_fallback(repositories, enabled=False)
    prompt_provider = RecordingImageProvider(_VALID_PNG_BYTES)
    primary_provider = SequenceImageProvider(
        provider_name="venice",
        outcomes=[
            ImageResponse(
                provider="venice",
                model_id="venice/safe-image",
                raw_metadata={
                    "_bragi_headers": {
                        "x-request-id": "primary-req",
                        "x-venice-is-blurred": "true",
                        "authorization": "Bearer sk-primary-leak",
                    },
                    "prompt": "leaked primary prompt bytes",
                    "body": "leaked primary response body",
                },
            )
        ],
    )
    fallback_provider = SequenceImageProvider(
        provider_name="fallback",
        outcomes=[
            ImageResponse(
                provider="fallback",
                model_id="fallback/image",
                image_bytes=_VALID_PNG_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": prompt_provider,
            "venice": primary_provider,
            "fallback": fallback_provider,
        },
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(primary_provider.image_requests) == 1
    assert primary_provider.image_requests[0].safe_mode is True
    assert len(fallback_provider.image_requests) == 1
    assert asset.provider == "fallback"
    assert asset.model == "fallback/image"
    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["payload"]["venice_safe_mode"] is True
    result = jobs[0]["result"]
    assert result["classification"] == "suspected_blocked_image_output"
    assert result["fallback_used"] is True
    assert result["fallback_provider"] == "fallback"
    assert result["fallback_model"] == "fallback/image"
    assert result["final_provider"] == "fallback"
    assert result["final_model"] == "fallback/image"
    assert result["primary_venice_safe_mode"] is True
    assert result["primary_provider_headers"] == {
        "x-request-id": "primary-req",
        "x-venice-is-blurred": "true",
    }
    result_repr = repr(result)
    assert "sk-primary-leak" not in result_repr
    assert "leaked primary prompt bytes" not in result_repr
    assert "leaked primary response body" not in result_repr


def test_generate_retries_image_fallback_for_content_blocked_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    _configure_image_fallback(repositories, enabled=True)
    primary_provider = SequenceImageProvider(
        provider_name="fake",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.CONTENT_BLOCKED,
                "image output was blocked",
            )
        ],
    )
    fallback_provider = SequenceImageProvider(
        provider_name="fallback",
        outcomes=[
            ImageResponse(
                provider="fallback",
                model_id="fallback/image",
                image_bytes=_VALID_PNG_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": primary_provider, "fallback": fallback_provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(primary_provider.image_requests) == 1
    assert len(fallback_provider.image_requests) == 1
    assert asset.provider == "fallback"
    assert asset.model == "fallback/image"
    jobs = _image_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["classification"] == "suspected_blocked_image_output"
    assert result["primary_error_category"] == (
        ProviderErrorCategory.CONTENT_BLOCKED.value
    )
    assert result["fallback_used"] is True
    assert result["final_provider"] == "fallback"
    assert result["final_model"] == "fallback/image"


def test_generate_retries_image_fallback_for_any_provider_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    _configure_image_fallback(repositories, enabled=True)
    primary_provider = SequenceImageProvider(
        provider_name="fake",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.AUTHENTICATION_FAILED,
                "image provider credentials were rejected",
            )
        ],
    )
    fallback_provider = SequenceImageProvider(
        provider_name="fallback",
        outcomes=[
            ImageResponse(
                provider="fallback",
                model_id="fallback/image",
                image_bytes=_VALID_PNG_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": primary_provider, "fallback": fallback_provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(primary_provider.image_requests) == 1
    assert len(fallback_provider.image_requests) == 1
    assert asset.provider == "fallback"
    assert asset.model == "fallback/image"
    jobs = _image_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["classification"] == "primary_image_generation_failed"
    assert result["primary_error_category"] == (
        ProviderErrorCategory.AUTHENTICATION_FAILED.value
    )
    assert result["fallback_used"] is True
    assert result["fallback_task"] == "image_fallback"
    assert result["final_provider"] == "fallback"
    assert result["final_model"] == "fallback/image"


def test_generate_retries_image_fallback_when_primary_output_file_is_missing(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    _configure_image_fallback(repositories, enabled=True)
    media_dir = tmp_path / "media"
    missing_relative_path = Path(save.id) / f"{messages[-1].id}.png"
    primary_provider = SequenceImageProvider(
        provider_name="fake",
        outcomes=[
            ImageResponse(
                provider="fake",
                model_id="fake-image",
                image_path=missing_relative_path,
            )
        ],
    )
    fallback_provider = SequenceImageProvider(
        provider_name="fallback",
        outcomes=[
            ImageResponse(
                provider="fallback",
                model_id="fallback/image",
                image_bytes=_VALID_PNG_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": primary_provider, "fallback": fallback_provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(primary_provider.image_requests) == 1
    assert len(fallback_provider.image_requests) == 1
    assert asset.provider == "fallback"
    assert asset.model == "fallback/image"
    assert _asset_path(media_dir, asset.path).read_bytes() == _VALID_PNG_BYTES
    jobs = _image_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["classification"] == "primary_image_not_stored"
    assert result["primary_error_message"] == (
        "Image provider returned a missing image file"
    )
    assert result["fallback_used"] is True
    assert result["fallback_task"] == "image_fallback"
    assert result["final_provider"] == "fallback"
    assert result["final_model"] == "fallback/image"


def test_generate_retries_image_fallback_when_primary_output_is_too_large(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    _configure_image_fallback(repositories, enabled=True)
    monkeypatch.setattr(media_service_module, "_MAX_PERSISTED_IMAGE_BYTES", 4)
    primary_provider = SequenceImageProvider(
        provider_name="fake",
        outcomes=[
            ImageResponse(
                provider="fake",
                model_id="fake-image",
                image_bytes=b"primary-image-too-large",
            )
        ],
    )
    fallback_provider = SequenceImageProvider(
        provider_name="fallback",
        outcomes=[
            ImageResponse(
                provider="fallback",
                model_id="fallback/image",
                image_bytes=_VALID_PNG_BYTES[:4],
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": primary_provider, "fallback": fallback_provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(fallback_provider.image_requests) == 1
    assert asset.provider == "fallback"
    jobs = _image_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["classification"] == "primary_image_not_stored"
    assert result["primary_error_message"] == "Generated image exceeded 4 bytes"
    assert result["fallback_used"] is True


def test_generate_retries_image_fallback_for_temporarily_unavailable_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    _configure_image_fallback(
        repositories,
        enabled=True,
        capabilities=["image_generation"],
    )
    primary_provider = SequenceImageProvider(
        provider_name="fake",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.MODEL_NOT_FOUND,
                "image model is temporarily unavailable",
                status_code=404,
            )
        ],
    )
    fallback_provider = SequenceImageProvider(
        provider_name="fallback",
        outcomes=[
            ImageResponse(
                provider="fallback",
                model_id="fallback/image",
                image_bytes=_VALID_PNG_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": primary_provider, "fallback": fallback_provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(fallback_provider.image_requests) == 1
    assert asset.provider == "fallback"
    jobs = _image_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["primary_error_category"] == (
        ProviderErrorCategory.MODEL_NOT_FOUND.value
    )
    assert result["fallback_used"] is True


def test_generate_skips_unavailable_image_fallback_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    _configure_image_fallback(
        repositories,
        enabled=True,
        capabilities=["image_generation"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="fallback",
        available_model_ids=set(),
    )
    primary_provider = SequenceImageProvider(
        provider_name="fake",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.CONTENT_BLOCKED,
                "image output was blocked",
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": primary_provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    with pytest.raises(Exception, match="image output was blocked"):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert len(primary_provider.image_requests) == 1
    jobs = _image_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "failed"
    assert result["fallback_used"] is False
    assert result["fallback_skipped_reason"] == "fallback_model_unavailable"


def test_generate_video_retries_video_fallback_for_blocked_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="video_generation",
        provider="primary",
        model_id="primary/safe-video",
    )
    _configure_video_fallback(
        repositories,
        enabled=True,
        capabilities=["text_to_video", "fallback_marker"],
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/video",
        display_name="OpenRouter Video Fallback",
        capabilities=["text_to_video", "fallback_marker"],
    )
    repositories.set_model_preference(
        task="video_fallback",
        provider="openrouter",
        model_id="openrouter/video",
    )
    prompt_provider = RecordingImageProvider(
        drafted_prompt="fallback-safe drafted video prompt",
    )
    primary_provider = SequenceVideoProvider(
        provider_name="primary",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.CONTENT_BLOCKED,
                "video output was blocked",
            )
        ],
    )
    fallback_provider = SequenceVideoProvider(
        provider_name="openrouter",
        outcomes=[
            VideoResponse(
                provider="openrouter",
                model_id="openrouter/video",
                mime_type="video/mp4",
                video_bytes=_VALID_MP4_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": prompt_provider,
            "primary": primary_provider,
            "openrouter": fallback_provider,
        },
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_video_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(primary_provider.video_requests) == 1
    assert len(fallback_provider.video_requests) == 1
    assert fallback_provider.video_requests[0].prompt == (
        primary_provider.video_requests[0].prompt
    )
    assert fallback_provider.video_requests[0].source_media_asset_id is None
    assert (
        fallback_provider.video_requests[0].openrouter_app_title
        == "Bragi"
    )
    assert asset.type == "video"
    assert asset.provider == "openrouter"
    assert asset.model == "openrouter/video"
    jobs = _video_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["classification"] == "suspected_blocked_video_output"
    assert result["primary_error_category"] == (
        ProviderErrorCategory.CONTENT_BLOCKED.value
    )
    assert result["fallback_used"] is True
    assert result["fallback_provider"] == "openrouter"
    assert result["fallback_model"] == "openrouter/video"
    assert result["final_provider"] == "openrouter"
    assert result["final_model"] == "openrouter/video"


def test_generate_video_uses_fallback_when_toggle_false(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="video_generation",
        provider="primary",
        model_id="primary/safe-video",
    )
    _configure_video_fallback(
        repositories,
        enabled=False,
        capabilities=["text_to_video", "fallback_marker"],
    )
    prompt_provider = RecordingImageProvider(
        drafted_prompt="fallback-safe drafted video prompt",
    )
    primary_provider = SequenceVideoProvider(
        provider_name="primary",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.CONTENT_BLOCKED,
                "video output was blocked",
            )
        ],
    )
    fallback_provider = SequenceVideoProvider(
        provider_name="fallback",
        outcomes=[
            VideoResponse(
                provider="fallback",
                model_id="fallback/video",
                mime_type="video/mp4",
                video_bytes=_VALID_MP4_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": prompt_provider,
            "primary": primary_provider,
            "fallback": fallback_provider,
        },
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_video_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(primary_provider.video_requests) == 1
    assert len(fallback_provider.video_requests) == 1
    assert asset.provider == "fallback"
    assert asset.model == "fallback/video"
    result = _video_generation_jobs(repositories, save.id)[0]["result"]
    assert result["fallback_used"] is True
    assert result["fallback_provider"] == "fallback"
    assert result["fallback_model"] == "fallback/video"


def test_generate_video_retries_video_fallback_for_blocked_metadata(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="video_generation",
        provider="primary",
        model_id="primary/safe-video",
    )
    _configure_video_fallback(
        repositories,
        enabled=True,
        capabilities=["text_to_video", "fallback_marker"],
    )
    prompt_provider = RecordingImageProvider(
        drafted_prompt="fallback-safe drafted video prompt",
    )
    primary_provider = SequenceVideoProvider(
        provider_name="primary",
        outcomes=[
            VideoResponse(
                provider="primary",
                model_id="primary/safe-video",
                mime_type="video/mp4",
                video_bytes=_VALID_MP4_BYTES,
                raw_metadata={
                    "native_finish_reason": "content_filter",
                    "_bragi_headers": {
                        "x-request-id": "primary-video-req",
                        "authorization": "Bearer sk-video-leak",
                    },
                },
            )
        ],
    )
    fallback_provider = SequenceVideoProvider(
        provider_name="fallback",
        outcomes=[
            VideoResponse(
                provider="fallback",
                model_id="fallback/video",
                mime_type="video/mp4",
                video_bytes=_VALID_MP4_BYTES,
                raw_metadata={"_bragi_headers": {"x-request-id": "fallback-req"}},
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": prompt_provider,
            "primary": primary_provider,
            "fallback": fallback_provider,
        },
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_video_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(primary_provider.video_requests) == 1
    assert len(fallback_provider.video_requests) == 1
    assert asset.provider == "fallback"
    jobs = _video_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["classification"] == "suspected_blocked_video_output"
    assert result["fallback_used"] is True
    assert result["primary_provider_headers"] == {
        "x-request-id": "primary-video-req"
    }
    assert result["provider_headers"] == {"x-request-id": "fallback-req"}
    assert "sk-video-leak" not in repr(result)


def test_generate_video_records_retry_diagnostics_when_fallback_succeeds(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="video_generation",
        provider="primary",
        model_id="primary/safe-video",
    )
    _configure_video_fallback(
        repositories,
        enabled=True,
        capabilities=["text_to_video", "fallback_marker"],
    )
    prompt_provider = RecordingImageProvider(
        drafted_prompt="fallback-safe drafted video prompt",
    )
    primary_provider = SequenceVideoProvider(
        provider_name="primary",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.PROVIDER_ERROR,
                "provider exhausted fast retries",
                status_code=503,
                retry_attempt_count=2,
                max_retry_attempts=2,
                retry_attempts=(
                    {
                        "attempt": 1,
                        "duration_ms": 20,
                        "error_category": "provider_error",
                        "http_status": 503,
                    },
                    {
                        "attempt": 2,
                        "duration_ms": 25,
                        "error_category": "provider_error",
                        "http_status": 503,
                    },
                ),
            )
        ],
    )
    fallback_provider = SequenceVideoProvider(
        provider_name="fallback",
        outcomes=[
            VideoResponse(
                provider="fallback",
                model_id="fallback/video",
                mime_type="video/mp4",
                video_bytes=_VALID_MP4_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": prompt_provider,
            "primary": primary_provider,
            "fallback": fallback_provider,
        },
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asyncio.run(
        service.generate_video_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    result = _video_generation_jobs(repositories, save.id)[0]["result"]
    assert result["classification"] == "suspected_blocked_video_output"
    assert result["primary_error_category"] == "provider_error"
    assert result["primary_http_status"] == 503
    assert result["primary_attempt_count"] == 2
    assert result["primary_max_attempts"] == 2
    assert result["primary_retry_attempts"] == [
        {
            "attempt": 1,
            "duration_ms": 20,
            "error_category": "provider_error",
            "http_status": 503,
        },
        {
            "attempt": 2,
            "duration_ms": 25,
            "error_category": "provider_error",
            "http_status": 503,
        },
    ]
    assert result["fallback_used"] is True


def test_generate_video_fails_with_diagnostics_when_primary_returns_no_video_data(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="video_generation",
        provider="primary",
        model_id="primary/safe-video",
    )
    prompt_provider = RecordingImageProvider(
        drafted_prompt="fallback-safe drafted video prompt",
    )
    primary_provider = SequenceVideoProvider(
        provider_name="primary",
        outcomes=[
            VideoResponse(
                provider="primary",
                model_id="primary/safe-video",
                mime_type="video/mp4",
                raw_metadata={
                    "_bragi_retry": {
                        "attempt_count": 3,
                        "max_attempts": 3,
                        "attempts": [
                            {
                                "attempt": 1,
                                "duration_ms": 10,
                                "error_category": "provider_error",
                            },
                            {
                                "attempt": 3,
                                "duration_ms": 12,
                                "error_category": "provider_error",
                            },
                        ],
                    }
                },
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": prompt_provider, "primary": primary_provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    with pytest.raises(ValueError, match="Video provider returned no video data"):
        asyncio.run(
            service.generate_video_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    jobs = _video_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "failed"
    assert result["classification"] == "suspected_blocked_video_output"
    assert result["fallback_used"] is False
    assert result["fallback_skipped_reason"] == "no_fallback_model"
    assert result["primary_attempt_count"] == 3
    assert result["primary_max_attempts"] == 3
    assert result["primary_retry_attempts"] == [
        {"attempt": 1, "duration_ms": 10, "error_category": "provider_error"},
        {"attempt": 3, "duration_ms": 12, "error_category": "provider_error"},
    ]


def test_generate_video_skips_unavailable_video_fallback_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="video_generation",
        provider="primary",
        model_id="primary/safe-video",
    )
    _configure_video_fallback(
        repositories,
        enabled=True,
        capabilities=["text_to_video", "fallback_marker"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="fallback",
        available_model_ids=set(),
    )
    prompt_provider = RecordingImageProvider(
        drafted_prompt="fallback-safe drafted video prompt",
    )
    primary_provider = SequenceVideoProvider(
        provider_name="primary",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.CONTENT_BLOCKED,
                "video output was blocked",
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={
            "fake": prompt_provider,
            "primary": primary_provider,
        },
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    with pytest.raises(Exception, match="video output was blocked"):
        asyncio.run(
            service.generate_video_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert len(primary_provider.video_requests) == 1
    jobs = _video_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "failed"
    assert result["fallback_used"] is False
    assert result["fallback_skipped_reason"] == "fallback_model_unavailable"


def test_generate_video_fails_when_provider_references_missing_output_path(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="video_generation",
        provider="fake",
        model_id="fake-video",
    )
    provider = SequenceVideoProvider(
        provider_name="fake",
        outcomes=[
            VideoResponse(
                provider="fake",
                model_id="fake-video",
                mime_type="video/mp4",
                video_path=Path("missing-provider-output.mp4"),
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    with pytest.raises(ValueError, match="missing video file"):
        asyncio.run(
            service.generate_video_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert repositories.list_media_assets(save.id) == []
    assert _video_generation_jobs(repositories, save.id)[0]["result"] is None


def test_generate_video_rejects_oversized_provider_video_file(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="video_generation",
        provider="fake",
        model_id="fake-video",
    )
    media_dir = tmp_path / "media"
    source_path = media_dir / "provider-output.mp4"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_VALID_MP4_BYTES + b"x" * 8)
    monkeypatch.setattr(media_service_module, "_MAX_PERSISTED_VIDEO_BYTES", 8)
    provider = SequenceVideoProvider(
        provider_name="fake",
        outcomes=[
            VideoResponse(
                provider="fake",
                model_id="fake-video",
                mime_type="video/mp4",
                video_path=source_path,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    with pytest.raises(ValueError, match="Generated video exceeded"):
        asyncio.run(
            service.generate_video_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert repositories.list_media_assets(save.id) == []
    assert _video_generation_jobs(repositories, save.id)[0]["result"] is None


def test_generate_for_message_uses_image_fallback_when_toggle_false(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    _configure_image_fallback(repositories, enabled=False)
    primary_provider = SequenceImageProvider(
        provider_name="fake",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.CONTENT_BLOCKED,
                "image output was blocked",
            )
        ],
    )
    fallback_provider = SequenceImageProvider(
        provider_name="fallback",
        outcomes=[
            ImageResponse(
                provider="fallback",
                model_id="fallback/image",
                image_bytes=_VALID_PNG_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": primary_provider, "fallback": fallback_provider},
        media_dir=tmp_path / "media",
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=messages[-1].id,
        )
    )

    assert len(primary_provider.image_requests) == 1
    assert len(fallback_provider.image_requests) == 1
    assert asset.provider == "fallback"
    assert asset.model == "fallback/image"
    jobs = _image_generation_jobs(repositories, save.id)
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["classification"] == "suspected_blocked_image_output"
    assert result["fallback_used"] is True
    assert result["primary_error_category"] == (
        ProviderErrorCategory.CONTENT_BLOCKED.value
    )
    assert result["final_provider"] == "fallback"
    assert result["final_model"] == "fallback/image"


def test_animate_image_skips_video_fallback_without_matching_video_flow(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_animation",
        provider="primary",
        model_id="primary/image-video",
    )
    _configure_video_fallback(
        repositories,
        enabled=True,
        capabilities=["text_to_video", "fallback_marker"],
    )
    media_dir = tmp_path / "media"
    source_path = media_dir / "source-image.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_VALID_PNG_BYTES)
    source_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="source-image.png",
        prompt="still frame of a bell under the ash bridge",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"content_rating": "g"},
    )
    primary_provider = SequenceVideoProvider(
        provider_name="primary",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.CONTENT_BLOCKED,
                "image animation was blocked",
            )
        ],
    )
    fallback_provider = SequenceVideoProvider(
        provider_name="fallback",
        outcomes=[
            VideoResponse(
                provider="fallback",
                model_id="fallback/video",
                mime_type="video/mp4",
                video_bytes=_VALID_MP4_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"primary": primary_provider, "fallback": fallback_provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    with pytest.raises(ValueError, match="image animation was blocked"):
        asyncio.run(
            service.animate_image(
                save_id=save.id,
                media_asset_id=source_image.id,
                motion_prompt="make the bell swing once",
            )
        )

    assert len(primary_provider.video_requests) == 1
    assert fallback_provider.video_requests == []
    assert [asset.id for asset in repositories.list_media_assets(save.id)] == [
        source_image.id
    ]
    jobs = _media_jobs(repositories, save.id, "image_animation")
    assert len(jobs) == 1
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "failed"
    assert result["classification"] == "suspected_blocked_video_output"
    assert result["fallback_used"] is False
    assert result["fallback_skipped_reason"] == (
        "fallback_model_lacks_required_capabilities"
    )
    assert result["primary_error_category"] == (
        ProviderErrorCategory.CONTENT_BLOCKED.value
    )


def test_animate_image_uses_image_plus_text_video_fallback(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="image_animation",
        provider="primary",
        model_id="primary/image-video",
    )
    _configure_video_fallback(
        repositories,
        enabled=True,
        capabilities=["image_plus_text_to_video", "fallback_marker"],
    )
    media_dir = tmp_path / "media"
    source_path = media_dir / "source-image.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_VALID_PNG_BYTES)
    source_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path="source-image.png",
        prompt="still frame of a bell under the ash bridge",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"content_rating": "g"},
    )
    primary_provider = SequenceVideoProvider(
        provider_name="primary",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.CONTENT_BLOCKED,
                "image animation was blocked",
            )
        ],
    )
    fallback_provider = SequenceVideoProvider(
        provider_name="fallback",
        outcomes=[
            VideoResponse(
                provider="fallback",
                model_id="fallback/video",
                mime_type="video/mp4",
                video_bytes=_VALID_MP4_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"primary": primary_provider, "fallback": fallback_provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    asset = asyncio.run(
        service.animate_image(
            save_id=save.id,
            media_asset_id=source_image.id,
            motion_prompt="make the bell swing once",
        )
    )

    assert len(primary_provider.video_requests) == 1
    assert len(fallback_provider.video_requests) == 1
    fallback_request = fallback_provider.video_requests[0]
    assert fallback_request.source_media_asset_id == source_image.id
    assert fallback_request.source_media_path == source_path
    assert asset.type == "video"
    assert asset.provider == "fallback"
    assert asset.model == "fallback/video"
    assert asset.source_media_asset_id == source_image.id
    jobs = _media_jobs(repositories, save.id, "image_animation")
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    result = jobs[0]["result"]
    assert result["fallback_used"] is True
    assert result["fallback_provider"] == "fallback"
    assert result["fallback_model"] == "fallback/video"
    assert result["final_provider"] == "fallback"
    assert result["final_model"] == "fallback/video"


def test_prompt_drafting_failure_marks_job_failed_without_creating_asset_or_file(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    media_dir = tmp_path / "media"
    provider = FailingPromptProvider()
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    with pytest.raises(ProviderError, match="prompt drafting unavailable"):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert len(provider.chat_requests) == 1
    assert provider.image_requests == []
    assert repositories.list_media_assets(save.id) == []
    assert list(media_dir.rglob("*")) == []
    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["result"] is None
    assert jobs[0]["payload"]["save_id"] == save.id
    assert jobs[0]["payload"]["source_message_id"] == messages[-1].id
    assert jobs[0]["payload"]["provider"] == "fake"
    assert jobs[0]["payload"]["model"] == "fake-image"
    assert "prompt drafting unavailable" in jobs[0]["error"]


def test_generate_for_message_cleans_up_files_when_media_asset_persistence_fails(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    media_dir = tmp_path / "media"
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )

    def fail_create_media_asset(**_kwargs: object) -> MediaAssetRecord:
        raise RuntimeError("database write failed after image persisted")

    monkeypatch.setattr(repositories, "create_media_asset", fail_create_media_asset)

    with pytest.raises(RuntimeError, match="database write failed"):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert repositories.list_media_assets(save.id) == []
    assert [path for path in media_dir.rglob("*") if path.is_file()] == []
    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["result"] is None
    assert "database write failed after image persisted" in jobs[0]["error"]


def test_seeded_scenario_starter_reference_cleans_up_files_when_persistence_fails(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, _messages = _save_with_image_preference(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
    )
    media_dir = tmp_path / "media"
    source_relative_path = Path("scenario-starters") / "scenario-1" / "ilyra.png"
    source_path = media_dir / source_relative_path
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=media_dir,
    )
    starter = ScenarioCharacterStarter(
        name="Captain Ilyra",
        starter_id="starter-ilyra",
        reference_image=ScenarioStarterReferenceImage(
            id="starter-ref-ilyra",
            path=source_relative_path.as_posix(),
            mime_type="image/png",
        ),
    )

    def fail_create_media_asset(**_kwargs: object) -> MediaAssetRecord:
        raise RuntimeError("database write failed after reference copy")

    monkeypatch.setattr(repositories, "create_media_asset", fail_create_media_asset)

    with pytest.raises(RuntimeError, match="database write failed"):
        service.create_character_reference_from_scenario_starter(
            save_id=save.id,
            character_id=character.id,
            starter=starter,
        )

    assert repositories.list_media_assets(save.id) == []
    assert [path for path in (media_dir / save.id).rglob("*") if path.is_file()] == []
    assert source_path.read_bytes() == _VALID_PNG_BYTES


def test_generate_for_message_keeps_files_when_job_success_update_fails(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    media_dir = tmp_path / "media"
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )
    original_update_job = repositories.update_job

    def fail_success_update(
        job_id: str,
        *,
        status: str,
        result: dict[str, object] | None = None,
        error: str | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> object:
        if status == "succeeded":
            raise RuntimeError("job success update failed after asset persisted")
        return original_update_job(
            job_id,
            status=status,
            result=result,
            error=error,
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(repositories, "update_job", fail_success_update)

    with pytest.raises(RuntimeError, match="job success update failed"):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    media_assets = repositories.list_media_assets(save.id)
    assert len(media_assets) == 1
    media_asset = media_assets[0]
    asset_path = _asset_path(media_dir, media_asset.path)
    assert asset_path.is_file()
    assert asset_path.read_bytes() == _VALID_PNG_BYTES
    _assert_private_thumbnail_if_present(media_dir=media_dir, asset=media_asset)

    jobs = _image_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["result"] is None
    assert "job success update failed after asset persisted" in jobs[0]["error"]


def test_generate_video_keeps_file_when_job_success_update_fails(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="video_generation",
        provider="fake",
        model_id="fake-video",
    )
    media_dir = tmp_path / "media"
    provider = RecordingVideoProvider(_VALID_MP4_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )
    original_update_job = repositories.update_job

    def fail_success_update(
        job_id: str,
        *,
        status: str,
        result: dict[str, object] | None = None,
        error: str | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> object:
        if status == "succeeded":
            raise RuntimeError("job success update failed after video persisted")
        return original_update_job(
            job_id,
            status=status,
            result=result,
            error=error,
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(repositories, "update_job", fail_success_update)

    with pytest.raises(RuntimeError, match="job success update failed"):
        asyncio.run(
            service.generate_video_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    media_assets = repositories.list_media_assets(save.id)
    assert len(media_assets) == 1
    media_asset = media_assets[0]
    asset_path = _asset_path(media_dir, media_asset.path)
    assert media_asset.type == "video"
    assert media_asset.mime_type == "video/mp4"
    assert asset_path.is_file()
    assert asset_path.read_bytes() == _VALID_MP4_BYTES

    jobs = _video_generation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["result"] is None
    assert "job success update failed after video persisted" in jobs[0]["error"]


def test_generate_video_logs_cleanup_failure_without_masking_original_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    repositories.set_model_preference(
        task="video_generation",
        provider="fake",
        model_id="fake-video",
    )
    media_dir = tmp_path / "media"
    provider = RecordingVideoProvider(_VALID_MP4_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        auto_frequency=3,
    )
    logged_events: list[dict[str, object]] = []

    def fail_create_media_asset(**_kwargs: object) -> MediaAssetRecord:
        raise RuntimeError("database write failed after video persisted")

    def fail_unlink(self: Path) -> None:
        raise PermissionError(f"cleanup denied for {self.name}")

    def capture_log_error_event(event: str, **fields: object) -> None:
        logged_events.append({"event": event, **fields})

    monkeypatch.setattr(repositories, "create_media_asset", fail_create_media_asset)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(
        media_service_module,
        "log_error_event",
        capture_log_error_event,
    )

    with pytest.raises(RuntimeError, match="database write failed"):
        asyncio.run(
            service.generate_video_for_message(
                save_id=save.id,
                source_message_id=messages[-1].id,
            )
        )

    assert repositories.list_media_assets(save.id) == []
    video_files = [path for path in media_dir.rglob("*.mp4") if path.is_file()]
    assert len(video_files) == 1
    assert any(
        event["event"] == "media.cleanup_failed"
        and str(event.get("path", "")).endswith(".mp4")
        for event in logged_events
    )
    jobs = _video_generation_jobs(repositories, save.id)
    assert jobs[0]["status"] == "failed"
    assert "database write failed after video persisted" in jobs[0]["error"]


def test_generate_character_reference_persists_character_link(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(
        repositories,
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    asset = asyncio.run(
        service.generate_character_reference(
            save_id=save.id,
            source_message_id=opening_message.id,
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.model_id == "fake-image"
    assert request.source_media_asset_id is None
    assert "Character reference portrait for Oracle of Glass" in request.prompt
    assert "mirrored silver eyes" in request.prompt
    media_assets = repositories.list_media_assets(save.id)
    assert [item.id for item in media_assets] == [asset.id]
    assert _asset_path(media_dir, asset.path).read_bytes() == _VALID_PNG_BYTES
    assert json.loads(asset.metadata_json) == {
        "content_rating": "g",
        "kind": "character_reference",
        "character_id": character_id,
    }
    links = repositories.list_entity_links(save.id)
    assert sorted(
        (
            link.entity_type,
            link.entity_id,
            link.target_type,
            link.target_id,
            link.relation,
        )
        for link in links
    ) == [
        ("character", character_id, "media_asset", asset.id, "reference_image")
    ]


def test_generate_scoped_reference_does_not_promote_first_image(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    first_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=opening_message.id,
        type="image",
        path=(Path(save.id) / opening_message.id / "first.png").as_posix(),
        thumbnail_path=None,
        prompt="first generated image",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )

    asset = asyncio.run(
        service.generate_character_reference(
            save_id=save.id,
            character_id=character_id,
            source_message_id=opening_message.id,
        )
    )

    assert asset.id != first_image.id
    assert len(provider.image_requests) == 1
    links = [
        link
        for link in repositories.list_entity_links(save.id)
        if link.relation == "reference_image"
    ]
    assert [
        (link.entity_type, link.entity_id, link.target_id) for link in links
    ] == [("character", character_id, asset.id)]


def test_generate_character_reference_keeps_prompt_compact_and_visual(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(
        repositories,
    )
    character = repositories.get_character(character_id)
    assert character is not None
    long_visual_description = (
        "Unique visible identity marker: pale green eyes, loose dark waves, "
        "soft blue cardigan, careful posture. "
        + "faint freckle detail " * 180
    )
    repositories.update_character(
        replace(
            character,
            appearance=long_visual_description,
            visual_notes=long_visual_description,
            current_clothing="Do not bake this temporary raincoat into the reference.",
            role="nonvisual relationship backstory marker " * 120,
            known_state="nonvisual current emotional state marker " * 80,
            personality="nonvisual personality marker " * 100,
        )
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    asyncio.run(
        service.generate_character_reference(
            save_id=save.id,
            source_message_id=opening_message.id,
        )
    )

    assert len(provider.image_requests) == 1
    prompt = provider.image_requests[0].prompt
    assert "Unique visible identity marker" in prompt
    assert 1 <= prompt.count("Unique visible identity marker") <= 2
    assert "nonvisual relationship backstory marker" not in prompt
    assert "nonvisual current emotional state marker" not in prompt
    assert "nonvisual personality marker" not in prompt
    assert "temporary raincoat" not in prompt
    assert "Character visual direction for Oracle of Glass" in prompt
    assert "Wearing: Unique visible identity marker" in prompt
    assert "Current action/pose: stable reusable reference portrait pose" in prompt
    assert (
        "Facial expression: neutral, reusable character-reference expression"
        in prompt
    )
    assert len(prompt) <= 1800


def test_generate_character_reference_rejects_unavailable_image_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, _character_id = _full_roleplay_save(
        repositories,
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-image",
        display_name="Fake Image",
        capabilities=["image_generation"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )

    with pytest.raises(ValueError, match="Image generation model is unavailable"):
        asyncio.run(
            service.generate_character_reference(
                save_id=save.id,
                source_message_id=opening_message.id,
            )
        )

    assert provider.image_requests == []
    jobs = _media_jobs(repositories, save.id, "character_reference_image")
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["error"] == "Image generation model is unavailable: fake-image"


def test_generate_character_reference_allows_full_roleplay_character(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Bell Warden",
        role="Keeps the ash bridge bells.",
        appearance="A rangy warden in a soot-dark cloak with brass chimes.",
        visual_notes="Soot-dark cloak, brass chimes, weathered face.",
        current_clothing="Borrowed green raincoat over a linen shirt.",
    )
    media_dir = tmp_path / "media"
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    asset = asyncio.run(
        service.generate_character_reference(
            save_id=save.id,
            character_id=character.id,
            source_message_id=messages[1].id,
        )
    )

    assert provider.image_requests[0].source_media_asset_id is None
    assert (
        "Character reference portrait for Bell Warden"
        in provider.image_requests[0].prompt
    )
    assert "Borrowed green raincoat" not in provider.image_requests[0].prompt
    links = repositories.list_entity_links(save.id)
    assert [
        (link.entity_type, link.entity_id, link.target_id, link.relation)
        for link in links
    ] == [("character", character.id, asset.id, "reference_image")]


def test_scene_generation_uses_present_character_reference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle turns toward the moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-scene-edit",
        display_name="Fake Scene Edit",
        capabilities=["image_to_image"],
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-default-edit",
        display_name="Fake Default Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task=roleplay_model_task(
            roleplay_type="full_roleplay",
            purpose=SCENE_IMAGE_EDIT_PURPOSE,
        ),
        provider="fake",
        model_id="fake-scene-edit",
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-default-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )
    reference = asyncio.run(
        service.generate_character_reference(
            save_id=save.id,
            source_message_id=opening_message.id,
        )
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The oracle is present.",
        present_character_ids=[character_id],
        source_message_id=scene_message.id,
    )

    scene_asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=scene_message.id,
        )
    )

    scene_request = provider.image_requests[-1]
    assert reference.id in {
        asset.id for asset in repositories.list_media_assets(save.id)
    }
    assert scene_request.model_id == "fake-scene-edit"
    assert scene_request.source_media_asset_id == reference.id
    assert scene_request.source_media_path == media_dir / reference.path
    assert scene_request.source_media_asset_ids == (reference.id,)
    assert scene_request.source_media_paths == (media_dir / reference.path,)
    assert "Use the attached character reference image" in scene_request.prompt
    assert "Character visual direction for Oracle of Glass" in scene_request.prompt
    expected_wearing = (
        "Wearing: Tall, still, mirrored silver eyes, white hair, blue glass robes."
    )
    assert expected_wearing in scene_request.prompt
    assert "Current action/pose: The oracle turns toward the moonlit window." in (
        scene_request.prompt
    )
    expected_expression = "Facial expression: expression grounded in this moment"
    assert expected_expression in scene_request.prompt
    assert scene_asset.source_media_asset_id == reference.id
    assert json.loads(scene_asset.metadata_json) == {
        "content_rating": "g",
        "kind": "scene_image",
        "source_character_reference_asset_id": reference.id,
        "source_character_reference_asset_ids": [reference.id],
        "source_character_reference_character_ids": [character_id],
        "source_character_reference_character_names": ["Oracle of Glass"],
    }


def test_scene_generation_uses_present_and_mentioned_character_references_up_to_cap(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    present = repositories.add_character(
        save_id=save.id,
        name="Mara Voss",
        aliases=["Oathkeeper"],
        role="Player character",
    )
    mentioned = repositories.add_character(
        save_id=save.id,
        name="Bell Warden",
        aliases=["Warden"],
        role="Bridge keeper",
    )
    omitted = repositories.add_character(
        save_id=save.id,
        name="Zephyr Saint",
        role="Distant figure",
    )
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Bell Warden gestures while Zephyr Saint watches from the ash.",
        provider="fake",
        model="fake-chat",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="At the bridge.",
        present_character_ids=[present.id],
        source_message_id=messages[-1].id,
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    media_dir = tmp_path / "media"
    present_reference = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=messages[-1].id,
        filename="present.png",
        prompt="present reference",
        metadata={"kind": "character_reference", "character_id": present.id},
    )
    mentioned_reference = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=messages[-1].id,
        filename="mentioned.png",
        prompt="mentioned reference",
        metadata={"kind": "character_reference", "character_id": mentioned.id},
    )
    omitted_reference = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=messages[-1].id,
        filename="omitted.png",
        prompt="omitted reference",
        metadata={"kind": "character_reference", "character_id": omitted.id},
    )
    for character, reference in (
        (present, present_reference),
        (mentioned, mentioned_reference),
        (omitted, omitted_reference),
    ):
        repositories.add_entity_link(
            save_id=save.id,
            entity_type="character",
            entity_id=character.id,
            target_type="media_asset",
            target_id=reference.id,
            relation="reference_image",
        )
    provider = RecordingImageProvider(
        _VALID_PNG_BYTES,
        image_reference_limit=2,
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    asset = asyncio.run(
        service.generate_for_message(
            save_id=save.id,
            source_message_id=scene_message.id,
        )
    )

    request = provider.image_requests[0]
    assert request.model_id == "fake-edit"
    assert request.source_media_asset_ids == (
        present_reference.id,
        mentioned_reference.id,
    )
    assert request.source_media_paths == (
        media_dir / present_reference.path,
        media_dir / mentioned_reference.path,
    )
    assert omitted_reference.id not in request.source_media_asset_ids
    assert asset.source_media_asset_id == present_reference.id
    assert json.loads(asset.metadata_json) == {
        "content_rating": "g",
        "kind": "scene_image",
        "source_character_reference_asset_id": present_reference.id,
        "source_character_reference_asset_ids": [
            present_reference.id,
            mentioned_reference.id,
        ],
        "source_character_reference_character_ids": [present.id, mentioned.id],
        "source_character_reference_character_names": ["Mara Voss", "Bell Warden"],
    }


def test_scene_generation_requires_image_to_image_when_reference_is_selected(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle raises a prism toward the window.",
        provider="fake",
        model="fake-chat",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )
    asyncio.run(
        service.generate_character_reference(
            save_id=save.id,
            source_message_id=opening_message.id,
        )
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The oracle is present.",
        present_character_ids=[character_id],
        source_message_id=scene_message.id,
    )

    with pytest.raises(
        ValueError,
        match="No image-to-image generation model preference configured",
    ):
        asyncio.run(
            service.generate_for_message(
                save_id=save.id,
                source_message_id=scene_message.id,
            )
        )

    assert len(provider.image_requests) == 1


def test_character_image_generation_uses_reference_image_to_image(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle turns toward the moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-character-edit",
        display_name="Fake Character Edit",
        capabilities=["image_to_image"],
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-default-edit",
        display_name="Fake Default Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task=roleplay_model_task(
            roleplay_type="full_roleplay",
            purpose=CHARACTER_IMAGE_EDIT_PURPOSE,
        ),
        provider="fake",
        model_id="fake-character-edit",
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-default-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    reference = _persist_character_reference(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        character_id=character_id,
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    asset = asyncio.run(
        service.generate_character_image_for_message(
            save_id=save.id,
            source_message_id=scene_message.id,
            character_id=character_id,
        )
    )

    assert len(provider.image_requests) == 1
    character_request = provider.image_requests[0]
    assert character_request.model_id == "fake-character-edit"
    assert character_request.source_media_asset_id == reference.id
    assert character_request.source_media_path == media_dir / reference.path
    assert "Use the attached character reference image" in character_request.prompt
    assert "Show only this one character" in character_request.prompt
    assert "Do not include other people" in character_request.prompt
    assert "The oracle turns toward the moonlit window" in character_request.prompt
    assert "Character visual direction for Oracle of Glass" in character_request.prompt
    expected_wearing = (
        "Wearing: Tall, still, mirrored silver eyes, white hair, blue glass robes."
    )
    assert expected_wearing in character_request.prompt
    assert "Current action/pose: The oracle turns toward the moonlit window." in (
        character_request.prompt
    )
    assert (
        "Facial expression: expression grounded in this moment"
        in character_request.prompt
    )
    assert asset.source_media_asset_id == character_request.source_media_asset_id
    assert json.loads(asset.metadata_json) == {
        "content_rating": "g",
        "kind": "character_image",
        "character_id": character_id,
        "character_name": "Oracle of Glass",
        "origin": "message_scene",
        "source_character_reference_asset_id": character_request.source_media_asset_id,
        "source_character_reference_asset_ids": [
            character_request.source_media_asset_id
        ],
    }


def test_character_image_generation_uses_image_edit_fallback(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle turns toward the moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="primary",
        model_id="primary/edit",
        display_name="Primary Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task=CHARACTER_IMAGE_EDIT_PURPOSE,
        provider="primary",
        model_id="primary/edit",
    )
    _configure_image_edit_fallback(repositories, enabled=True)
    media_dir = tmp_path / "media"
    reference = _persist_character_reference(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        character_id=character_id,
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )
    primary_provider = SequenceImageProvider(
        provider_name="primary",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.PROVIDER_ERROR,
                "primary edit failed",
            )
        ],
    )
    fallback_provider = SequenceImageProvider(
        provider_name="fallback-edit",
        outcomes=[
            ImageResponse(
                provider="fallback-edit",
                model_id="fallback/edit",
                image_bytes=_VALID_PNG_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"primary": primary_provider, "fallback-edit": fallback_provider},
        media_dir=media_dir,
    )

    asset = asyncio.run(
        service.generate_character_image_for_message(
            save_id=save.id,
            source_message_id=scene_message.id,
            character_id=character_id,
        )
    )

    assert len(primary_provider.image_requests) == 1
    assert len(fallback_provider.image_requests) == 1
    fallback_request = fallback_provider.image_requests[0]
    assert fallback_request.provider == "fallback-edit"
    assert fallback_request.model_id == "fallback/edit"
    assert fallback_request.source_media_asset_id == reference.id
    assert fallback_request.source_media_path == media_dir / reference.path
    assert asset.provider == "fallback-edit"
    assert asset.model == "fallback/edit"
    assert asset.source_media_asset_id == reference.id
    jobs = _media_jobs(repositories, save.id, "character_image_generation")
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["fallback_used"] is True
    assert result["fallback_task"] == IMAGE_EDIT_FALLBACK_PURPOSE
    assert result["fallback_provider"] == "fallback-edit"
    assert result["fallback_model"] == "fallback/edit"


def test_character_image_generation_uses_legacy_image_fallback_for_edit(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle turns toward the moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="primary",
        model_id="primary/edit",
        display_name="Primary Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task=CHARACTER_IMAGE_EDIT_PURPOSE,
        provider="primary",
        model_id="primary/edit",
    )
    _configure_image_fallback(
        repositories,
        enabled=True,
        capabilities=["image_to_image", "fallback_marker"],
    )
    media_dir = tmp_path / "media"
    reference = _persist_character_reference(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        character_id=character_id,
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )
    primary_provider = SequenceImageProvider(
        provider_name="primary",
        outcomes=[
            ProviderError(
                ProviderErrorCategory.PROVIDER_ERROR,
                "primary edit failed",
            )
        ],
    )
    fallback_provider = SequenceImageProvider(
        provider_name="fallback",
        outcomes=[
            ImageResponse(
                provider="fallback",
                model_id="fallback/image",
                image_bytes=_VALID_PNG_BYTES,
            )
        ],
    )
    service = MediaService(
        repositories=repositories,
        providers={"primary": primary_provider, "fallback": fallback_provider},
        media_dir=media_dir,
    )

    asset = asyncio.run(
        service.generate_character_image_for_message(
            save_id=save.id,
            source_message_id=scene_message.id,
            character_id=character_id,
        )
    )

    assert len(fallback_provider.image_requests) == 1
    fallback_request = fallback_provider.image_requests[0]
    assert fallback_request.source_media_asset_id == reference.id
    assert fallback_request.source_media_path == media_dir / reference.path
    assert asset.provider == "fallback"
    assert asset.model == "fallback/image"
    jobs = _media_jobs(repositories, save.id, "character_image_generation")
    result = jobs[0]["result"]
    assert jobs[0]["status"] == "succeeded"
    assert result["fallback_used"] is True
    assert result["fallback_task"] == "image_fallback"
    assert result["fallback_provider"] == "fallback"
    assert result["fallback_model"] == "fallback/image"


def test_character_image_generation_allows_full_roleplay_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, messages = _save_with_image_preference(repositories)
    scene_message = messages[-1]
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        met=True,
        status="present",
        visual_notes="A storm-cloaked oathkeeper with a brass lantern.",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    reference = _persist_character_reference(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=messages[0].id,
        character_id=character.id,
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character.id,
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    asset = asyncio.run(
        service.generate_character_image_for_message(
            save_id=save.id,
            source_message_id=scene_message.id,
            character_id=character.id,
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.model_id == "fake-edit"
    assert request.source_media_asset_id == reference.id
    assert request.source_media_path == media_dir / reference.path
    assert "Show only this one character" in request.prompt
    assert "Do not include other people" in request.prompt
    assert scene_message.body in request.prompt
    assert "Character visual direction for Mara" in request.prompt
    assert "Wearing: A storm-cloaked oathkeeper with a brass lantern." in request.prompt
    assert "Current action/pose: The echo answers from below." in request.prompt
    assert "Facial expression: expression grounded in this moment" in request.prompt
    assert asset.source_media_asset_id == reference.id
    assert json.loads(asset.metadata_json) == {
        "content_rating": "g",
        "kind": "character_image",
        "character_id": character.id,
        "character_name": "Mara",
        "origin": "message_scene",
        "source_character_reference_asset_id": reference.id,
        "source_character_reference_asset_ids": [reference.id],
    }


def test_character_image_generation_requires_selected_character_presence(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle turns toward the moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    _persist_character_reference(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        character_id=character_id,
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    with pytest.raises(ValueError, match="not present"):
        asyncio.run(
            service.generate_character_image_for_message(
                save_id=save.id,
                source_message_id=scene_message.id,
                character_id=character_id,
            )
        )

    assert provider.image_requests == []


def test_character_registry_image_generation_uses_reference_without_message_link(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    reference = _persist_character_reference(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        character_id=character_id,
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    asset = asyncio.run(
        service.generate_character_image_for_character(
            save_id=save.id,
            character_id=character_id,
            instructions="blue dawn rim light",
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.source_media_asset_id == reference.id
    assert "blue dawn rim light" in request.prompt
    assert "Show only this one character" in request.prompt
    assert "Character visual direction for Oracle of Glass" in request.prompt
    expected_wearing = (
        "Wearing: Tall, still, mirrored silver eyes, white hair, blue glass robes."
    )
    assert expected_wearing in request.prompt
    assert "Current action/pose: blue dawn rim light." in request.prompt
    expected_expression = "Facial expression: expression grounded in this moment"
    assert expected_expression in request.prompt
    assert asset.source_message_id is None
    assert asset.source_media_asset_id == reference.id
    assert json.loads(asset.metadata_json) == {
        "content_rating": "g",
        "kind": "character_image",
        "character_id": character_id,
        "character_name": "Oracle of Glass",
        "origin": "character_registry",
        "source_character_reference_asset_id": reference.id,
        "source_character_reference_asset_ids": [reference.id],
    }


@pytest.mark.parametrize("promoted_kind", ["character_image", "scene_image"])
def test_set_character_reference_image_promotes_generated_image(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    promoted_kind: str,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    secondary = repositories.add_character(
        save_id=save.id,
        name="Zephyr Attendant",
        role="Secondary character",
    )
    media_dir = tmp_path / "media"
    secondary_reference = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        filename="secondary-reference.png",
        prompt="secondary reference",
        metadata={"kind": "character_reference", "character_id": secondary.id},
    )
    current_reference = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        filename="current-reference.png",
        prompt="current reference",
        metadata={"kind": "character_reference", "character_id": character_id},
    )
    promoted = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        filename=f"promoted-{promoted_kind}.png",
        prompt=f"promoted {promoted_kind}",
        metadata={
            "kind": promoted_kind,
            "source_character_reference_asset_id": current_reference.id,
        },
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character_id,
        target_type="media_asset",
        target_id=current_reference.id,
        relation="reference_image",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=secondary.id,
        target_type="media_asset",
        target_id=secondary_reference.id,
        relation="reference_image",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    selected = service.set_character_reference_image(
        save_id=save.id,
        media_asset_id=promoted.id,
    )

    assert selected.id == promoted.id
    assert provider.image_requests == []
    links = [
        link
        for link in repositories.list_entity_links(save.id)
        if link.relation == "reference_image"
    ]
    assert sorted((link.entity_id, link.target_id) for link in links) == sorted(
        [
            (secondary.id, secondary_reference.id),
            (character_id, promoted.id),
        ]
    )
    assert {asset.id for asset in repositories.list_media_assets(save.id)} == {
        secondary_reference.id,
        current_reference.id,
        promoted.id,
    }
    assert json.loads(promoted.metadata_json)["kind"] == promoted_kind


@pytest.mark.parametrize("promoted_kind", ["character_image", "scene_image"])
def test_character_image_generation_uses_promoted_reference_image(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    promoted_kind: str,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    media_dir = tmp_path / "media"
    promoted = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        filename=f"promoted-{promoted_kind}.png",
        prompt=f"promoted {promoted_kind}",
        metadata={"kind": promoted_kind},
    )
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle turns toward the moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )
    service.set_character_reference_image(
        save_id=save.id,
        media_asset_id=promoted.id,
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )

    asset = asyncio.run(
        service.generate_character_image_for_message(
            save_id=save.id,
            source_message_id=scene_message.id,
            character_id=character_id,
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.model_id == "fake-edit"
    assert request.source_media_asset_id == promoted.id
    assert request.source_media_path == media_dir / promoted.path
    assert asset.source_media_asset_id == promoted.id


def test_character_churn_does_not_reuse_retired_save_level_reference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    media_dir = tmp_path / "media"
    selected_reference = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        filename="selected-reference.png",
        prompt="selected generated reference",
        metadata={"kind": "scene_image"},
    )
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle turns toward the moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )
    service.set_character_reference_image(
        save_id=save.id,
        media_asset_id=selected_reference.id,
    )
    repositories.archive_character(character_id)
    recreated = repositories.add_character(
        save_id=save.id,
        name="Oracle of Glass",
        role="Recreated primary character row.",
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=recreated.id,
    )

    with pytest.raises(
        ValueError,
        match="Selected character does not have a reference image",
    ):
        asyncio.run(
            service.generate_character_image_for_message(
                save_id=save.id,
                source_message_id=scene_message.id,
                character_id=recreated.id,
            )
        )

    assert provider.image_requests == []
    assert _media_jobs(repositories, save.id, "character_reference_image") == []


@pytest.mark.parametrize(
    ("scenario_type", "asset_type", "status", "metadata", "message"),
    [
        (
            "full_roleplay",
            "video",
            "succeeded",
            {"kind": "character_image"},
            "Only generated character, scene, or uploaded reference images",
        ),
        (
            "full_roleplay",
            "image",
            "failed",
            {"kind": "character_image"},
            "Only generated character, scene, or uploaded reference images",
        ),
        (
            "full_roleplay",
            "image",
            "succeeded",
            {"kind": "opening_image"},
            "Only generated character, scene, or uploaded reference images",
        ),
    ],
)
def test_set_character_reference_image_rejects_ineligible_assets(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    scenario_type: str,
    asset_type: str,
    status: str,
    metadata: dict[str, object],
    message: str,
) -> None:
    if scenario_type == "full_roleplay":
        save, opening_message, _character_id = _full_roleplay_save(
            repositories
        )
    else:
        save, messages = _save_with_image_preference(repositories)
        opening_message = messages[0]
    media_dir = tmp_path / "media"
    asset = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        filename="candidate.png",
        prompt="candidate",
        media_type=asset_type,
        status=status,
        metadata=metadata,
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=media_dir,
    )

    with pytest.raises(ValueError, match=message):
        service.set_character_reference_image(
            save_id=save.id,
            media_asset_id=asset.id,
        )


def test_character_image_generation_allows_missing_image_to_image_catalog_row(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle turns toward the moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-unsynced-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    reference = _persist_character_reference(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        character_id=character_id,
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    asset = asyncio.run(
        service.generate_character_image_for_message(
            save_id=save.id,
            source_message_id=scene_message.id,
            character_id=character_id,
        )
    )

    assert len(provider.image_requests) == 1
    character_request = provider.image_requests[0]
    assert character_request.model_id == "fake-unsynced-edit"
    assert character_request.source_media_asset_id == reference.id
    assert asset.source_media_asset_id == character_request.source_media_asset_id


def test_character_image_generation_rejects_unavailable_image_to_image_before_reference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle turns toward the moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )

    with pytest.raises(
        ValueError,
        match="Image-to-image generation model is unavailable",
    ):
        asyncio.run(
            service.generate_character_image_for_message(
                save_id=save.id,
                source_message_id=scene_message.id,
                character_id=character_id,
            )
        )

    assert provider.chat_requests == []
    assert provider.image_requests == []
    assert repositories.list_media_assets(save.id) == []
    assert _media_jobs(repositories, save.id, "character_reference_image") == []
    assert _media_jobs(repositories, save.id, "character_image_generation") == []


def test_character_image_generation_rejects_scene_image_without_character_reference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    media_dir = tmp_path / "media"
    scene_image_path = Path(save.id) / opening_message.id / "scene.png"
    scene_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=opening_message.id,
        type="image",
        path=scene_image_path.as_posix(),
        thumbnail_path=None,
        prompt="a text-to-image scene",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "scene_image"},
    )
    scene_file = media_dir / scene_image.path
    scene_file.parent.mkdir(parents=True, exist_ok=True)
    scene_file.write_bytes(_VALID_PNG_BYTES)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle leans close to the silver basin.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )

    with pytest.raises(ValueError, match="reference image"):
        asyncio.run(
            service.generate_character_image_for_message(
                save_id=save.id,
                source_message_id=scene_message.id,
                character_id=character_id,
            )
        )

    assert provider.image_requests == []
    assert scene_image.source_media_asset_id is None


def test_character_image_generation_does_not_promote_first_image_as_reference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    media_dir = tmp_path / "media"
    first_image_path = Path(save.id) / opening_message.id / "first.png"
    first_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=opening_message.id,
        type="image",
        path=first_image_path.as_posix(),
        thumbnail_path=None,
        prompt="first generated character image",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    first_file = media_dir / first_image.path
    first_file.parent.mkdir(parents=True, exist_ok=True)
    first_file.write_bytes(_VALID_PNG_BYTES)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle leans close to the silver basin.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )

    with pytest.raises(ValueError, match="reference image"):
        asyncio.run(
            service.generate_character_image_for_message(
                save_id=save.id,
                source_message_id=scene_message.id,
                character_id=character_id,
            )
        )

    assert provider.image_requests == []
    links = repositories.list_entity_links(save.id)
    assert not any(link.target_id == first_image.id for link in links)


def test_character_image_generation_rejects_failed_linked_reference_without_fallback(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    media_dir = tmp_path / "media"
    failed_reference = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        filename="failed-reference.png",
        prompt="failed linked reference",
        status="failed",
    )
    fallback_reference = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        filename="fallback-generated.png",
        prompt="first usable generated image",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character_id,
        target_type="media_asset",
        target_id=failed_reference.id,
        relation="reference_image",
    )
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle leans close to the silver basin.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )

    with pytest.raises(ValueError, match="reference image"):
        asyncio.run(
            service.generate_character_image_for_message(
                save_id=save.id,
                source_message_id=scene_message.id,
                character_id=character_id,
            )
        )

    assert provider.image_requests == []
    links = repositories.list_entity_links(save.id)
    assert [
        (link.entity_type, link.entity_id, link.target_id, link.relation)
        for link in links
    ] == [
        (
            "character",
            character_id,
            failed_reference.id,
            "reference_image",
        )
    ]
    assert not any(
        link.target_id == fallback_reference.id and link.relation == "reference_image"
        for link in links
    )
    assert _media_jobs(repositories, save.id, "character_reference_image") == []


def test_upload_character_reference_persists_private_media_and_link(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _opening_message, character_id = _full_roleplay_save(repositories)
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )

    asset = service.upload_character_reference_image(
        save_id=save.id,
        image_bytes=_VALID_PNG_BYTES,
        filename="private-portrait.png",
    )

    assert provider.image_requests == []
    assert asset.source_message_id is None
    assert asset.type == "image"
    assert asset.mime_type == "image/png"
    assert asset.provider == "local"
    assert asset.model == "upload"
    assert asset.status == "succeeded"
    assert asset.prompt == "Uploaded character reference image"
    assert json.loads(asset.metadata_json) == {
        "kind": "character_reference",
        "source": "uploaded",
        "character_id": character_id,
    }
    image_path = _asset_path(media_dir, asset.path)
    assert image_path.read_bytes() == _VALID_PNG_BYTES
    _assert_private_modes(image_path)
    _assert_private_thumbnail_if_present(media_dir=media_dir, asset=asset)
    assert "private-portrait" not in asset.path
    links = repositories.list_entity_links(save.id)
    assert sorted(
        (
            link.entity_type,
            link.entity_id,
            link.target_type,
            link.target_id,
            link.relation,
        )
        for link in links
        if link.relation == "reference_image"
    ) == [
        ("character", character_id, "media_asset", asset.id, "reference_image")
    ]


def test_upload_character_reference_allows_full_roleplay_character(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _messages = _save_with_image_preference(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Bell Warden",
        role="Keeps the ash bridge bells.",
    )
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=tmp_path / "media",
    )

    asset = service.upload_character_reference_image(
        save_id=save.id,
        character_id=character.id,
        image_bytes=_VALID_PNG_BYTES,
        filename="reference.png",
    )

    assert json.loads(asset.metadata_json)["character_id"] == character.id
    links = [
        link
        for link in repositories.list_entity_links(save.id)
        if link.relation == "reference_image"
    ]
    assert [
        (link.entity_type, link.entity_id, link.target_id)
        for link in links
    ] == [("character", character.id, asset.id)]


def test_upload_scoped_reference_does_not_create_retired_save_compat_link(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _opening_message, character_id = _full_roleplay_save(repositories)
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=tmp_path / "media",
    )

    asset = service.upload_character_reference_image(
        save_id=save.id,
        character_id=character_id,
        image_bytes=_VALID_PNG_BYTES,
        filename="primary-reference.png",
    )

    links = [
        link
        for link in repositories.list_entity_links(save.id)
        if link.relation == "reference_image"
    ]
    assert [
        (link.entity_type, link.entity_id, link.target_id) for link in links
    ] == [("character", character_id, asset.id)]


def test_upload_character_reference_replaces_link_without_archiving_old_source(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=media_dir,
    )
    old_reference = service.upload_character_reference_image(
        save_id=save.id,
        image_bytes=_VALID_PNG_BYTES,
        filename="old.png",
    )
    scene_asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=opening_message.id,
        source_media_asset_id=old_reference.id,
        type="image",
        path=(Path(save.id) / "scene.png").as_posix(),
        thumbnail_path=None,
        prompt="scene from uploaded reference",
        provider="fake",
        model="fake-edit",
        status="succeeded",
        metadata={"kind": "character_image"},
    )

    with pytest.raises(ValueError, match="already exists"):
        service.upload_character_reference_image(
            save_id=save.id,
            image_bytes=_VALID_PNG_BYTES,
            filename="new.png",
        )
    new_reference = service.upload_character_reference_image(
        save_id=save.id,
        image_bytes=_VALID_PNG_BYTES,
        filename="new.png",
        replace_existing=True,
    )

    assert repositories.list_media_assets(save.id)[0].archived_at is None
    assert repositories.list_media_assets(save.id)[1].archived_at is None
    assert scene_asset.source_media_asset_id == old_reference.id
    links = [
        link
        for link in repositories.list_entity_links(save.id)
        if link.relation == "reference_image"
    ]
    assert [
        (link.entity_type, link.entity_id, link.target_id) for link in links
    ] == [("character", character_id, new_reference.id)]


def test_remove_character_reference_only_unlinks_active_reference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _opening_message, _character_id = _full_roleplay_save(repositories)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=media_dir,
    )
    reference = service.upload_character_reference_image(
        save_id=save.id,
        image_bytes=_VALID_PNG_BYTES,
        filename="reference.webp",
    )

    removed = service.remove_character_reference_image(save_id=save.id)

    assert removed is not None
    assert removed.id == reference.id
    assert repositories.list_entity_links(save.id) == []
    assert repositories.list_media_assets(save.id)[0].archived_at is None
    assert _asset_path(media_dir, reference.path).is_file()
    assert service.remove_character_reference_image(save_id=save.id) is None


def test_remove_scoped_reference_clears_character_link(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _opening_message, character_id = _full_roleplay_save(repositories)
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=tmp_path / "media",
    )
    reference = service.upload_character_reference_image(
        save_id=save.id,
        image_bytes=_VALID_PNG_BYTES,
        filename="reference.webp",
    )

    removed = service.remove_character_reference_image(
        save_id=save.id,
        character_id=character_id,
    )

    assert removed is not None
    assert removed.id == reference.id
    assert [
        link
        for link in repositories.list_entity_links(save.id)
        if link.relation == "reference_image"
    ] == []


def test_upload_character_reference_validates_type_size_and_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _opening_message, _character_id = _full_roleplay_save(repositories)
    full_roleplay_save, _messages = _save_with_image_preference(repositories)
    service = MediaService(
        repositories=repositories,
        providers={"fake": RecordingImageProvider(_VALID_PNG_BYTES)},
        media_dir=tmp_path / "media",
    )

    with pytest.raises(ValueError, match="Unsupported image upload type"):
        service.upload_character_reference_image(
            save_id=save.id,
            image_bytes=b"not an image",
            filename="notes.txt",
        )
    with pytest.raises(ValueError, match="Uploaded image exceeded"):
        service.upload_character_reference_image(
            save_id=save.id,
            image_bytes=b"\x89PNG\r\n\x1a\n" + (b"x" * (25 * 1024 * 1024)),
            filename="huge.png",
        )
    with pytest.raises(ValueError, match="Unknown save id"):
        service.upload_character_reference_image(
            save_id="missing-save",
            image_bytes=_VALID_PNG_BYTES,
            filename="reference.png",
        )
    with pytest.raises(ValueError, match="No character is available"):
        service.upload_character_reference_image(
            save_id=full_roleplay_save.id,
            image_bytes=_VALID_PNG_BYTES,
            filename="reference.png",
        )


def test_uploaded_image_validation_rejects_malformed_image_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed_png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", b"not-a-zlib-stream")
        + _png_chunk(b"IEND", b"")
    )
    nonempty_iend_png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
        + _png_chunk(b"IEND", b"not-empty")
    )
    indexed_png_without_plte = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", 1, 1, 8, 3, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00"))
        + _png_chunk(b"IEND", b"")
    )
    malformed_jpeg = (
        b"\xff\xd8"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00"
        b"\x00"
        b"\xff\xd9"
    )
    malformed_webp_chunk = b"not-vp8!!!"
    malformed_webp = (
        b"RIFF"
        + (len(malformed_webp_chunk) + 12).to_bytes(4, "little")
        + b"WEBP"
        + b"VP8 "
        + len(malformed_webp_chunk).to_bytes(4, "little")
        + malformed_webp_chunk
    )
    header_only_vp8 = b"\x00\x00\x00\x9d\x01\x2a\x01\x00\x01\x00"
    header_only_webp = (
        b"RIFF"
        + (len(header_only_vp8) + 12).to_bytes(4, "little")
        + b"WEBP"
        + b"VP8 "
        + len(header_only_vp8).to_bytes(4, "little")
        + header_only_vp8
    )

    for image_bytes in (
        malformed_png,
        nonempty_iend_png,
        indexed_png_without_plte,
        malformed_jpeg,
        malformed_webp,
        header_only_webp,
    ):
        with pytest.raises(ValueError, match="Unsupported image upload type"):
            media_service_module._uploaded_image_mime_type(image_bytes)

    monkeypatch.setattr(
        media_service_module,
        "_MAX_UPLOADED_IMAGE_DECODED_BYTES",
        1024,
    )
    oversized_dimension_png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", 1, 300, 8, 6, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(b""))
        + _png_chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="Unsupported image upload type"):
        media_service_module._uploaded_image_mime_type(oversized_dimension_png)

    monkeypatch.setattr(
        media_service_module,
        "_MAX_UPLOADED_IMAGE_DECODED_BYTES",
        4,
    )
    with pytest.raises(ValueError, match="Unsupported image upload type"):
        media_service_module._uploaded_image_mime_type(_VALID_PNG_BYTES)


def test_uploaded_png_rejects_when_available_decoder_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectingLoader:
        def write(self, _image_bytes: bytes) -> None:
            return None

        def close(self) -> None:
            raise ValueError("decoder rejected image")

    class RejectingPixbufLoader:
        @staticmethod
        def new_with_mime_type(_mime_type: str) -> RejectingLoader:
            return RejectingLoader()

    class RejectingGdkPixbuf:
        PixbufLoader = RejectingPixbufLoader

    monkeypatch.setattr(
        media_service_module,
        "_gdk_pixbuf_module",
        lambda: RejectingGdkPixbuf,
    )

    with pytest.raises(ValueError, match="Unsupported image upload type"):
        media_service_module._uploaded_image_mime_type(_VALID_PNG_BYTES)


def test_uploaded_character_reference_is_reused_for_character_image_generation(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, _opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle turns toward the moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )
    reference = service.upload_character_reference_image(
        save_id=save.id,
        image_bytes=_VALID_PNG_BYTES,
        filename="reference.png",
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )

    asset = asyncio.run(
        service.generate_character_image_for_message(
            save_id=save.id,
            source_message_id=scene_message.id,
            character_id=character_id,
        )
    )

    assert len(provider.image_requests) == 1
    request = provider.image_requests[0]
    assert request.source_media_asset_id == reference.id
    assert request.source_media_path == media_dir / reference.path
    assert asset.source_media_asset_id == reference.id


def test_character_image_generation_requires_image_to_image_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle lifts a prism in one hand.",
        provider="fake",
        model="fake-chat",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
    )
    _persist_character_reference(
        repositories,
        media_dir=media_dir,
        save_id=save.id,
        source_message_id=opening_message.id,
        character_id=character_id,
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )

    with pytest.raises(ValueError, match="image-to-image generation"):
        asyncio.run(
            service.generate_character_image_for_message(
                save_id=save.id,
                source_message_id=scene_message.id,
                character_id=character_id,
            )
        )

    assert len(repositories.list_media_assets(save.id)) == 1


def test_automatic_scene_generation_uses_present_character_reference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save, opening_message, character_id = _full_roleplay_save(repositories)
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle points toward a moonlit window.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-scene-edit",
        display_name="Fake Scene Edit",
        capabilities=["image_to_image"],
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-default-edit",
        display_name="Fake Default Edit",
        capabilities=["image_to_image"],
    )
    repositories.set_model_preference(
        task=roleplay_model_task(
            roleplay_type="full_roleplay",
            purpose=SCENE_IMAGE_EDIT_PURPOSE,
        ),
        provider="fake",
        model_id="fake-scene-edit",
    )
    repositories.set_model_preference(
        task="full_roleplay_image_to_image_generation",
        provider="fake",
        model_id="fake-default-edit",
    )
    provider = RecordingImageProvider(_VALID_PNG_BYTES)
    media_dir = tmp_path / "media"
    service = MediaService(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=media_dir,
        automatic_enabled=True,
        auto_frequency=2,
    )
    reference = asyncio.run(
        service.generate_character_reference(
            save_id=save.id,
            source_message_id=opening_message.id,
        )
    )
    _mark_character_present(
        repositories,
        save_id=save.id,
        message_id=scene_message.id,
        character_id=character_id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The oracle is present.",
        present_character_ids=[character_id],
        source_message_id=scene_message.id,
    )

    asset = asyncio.run(
        service.generate_automatic_if_due(
            save_id=save.id,
            source_message_id=scene_message.id,
        )
    )

    assert asset is not None
    request = provider.image_requests[-1]
    assert request.model_id == "fake-scene-edit"
    assert request.source_media_asset_id == reference.id
    assert request.source_media_path == media_dir / reference.path
    assert asset.source_media_asset_id == reference.id
    assert json.loads(asset.metadata_json) == {
        "content_rating": "g",
        "kind": "scene_image",
        "source_character_reference_asset_id": reference.id,
        "source_character_reference_asset_ids": [reference.id],
        "source_character_reference_character_ids": [character_id],
        "source_character_reference_character_names": ["Oracle of Glass"],
    }


def _media_service(
    *,
    repositories: PersistenceRepositories,
    provider: RecordingImageProvider,
    media_dir: Path,
    auto_frequency: object,
) -> MediaService:
    kwargs: dict[str, Any] = {
        "repositories": repositories,
        "providers": {"fake": provider},
        "media_dir": media_dir,
    }
    if auto_frequency is not _UNSET:
        kwargs["auto_frequency"] = auto_frequency
    kwargs["automatic_enabled"] = True
    return MediaService(**kwargs)


def _save_with_image_preference(
    repositories: PersistenceRepositories,
) -> tuple[SaveRecord, list[MessageRecord]]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Bridge of Cinders",
        premise="A bridge remembers every oath broken on it.",
        player_role="Oathkeeper",
        content={"starting_scene": "Cinders drift over the bridge stones."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Crossing")
    messages = [
        repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I step onto the ash bridge.",
            token_estimate=55,
            content_rating="g",
        ),
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="A bell rings under the span.",
            provider="fake",
            model="fake-chat",
            token_estimate=65,
            content_rating="g",
        ),
        repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I ask who rang the bell.",
            token_estimate=45,
            content_rating="g",
        ),
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="The echo answers from below.",
            provider="fake",
            model="fake-chat",
            token_estimate=50,
            content_rating="g",
        ),
    ]
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    return save, messages


def _save_with_custom_ids_and_image_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    message_id: str,
) -> tuple[SaveRecord, MessageRecord]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Bridge of Cinders",
        premise="A bridge remembers every oath broken on it.",
        player_role="Oathkeeper",
        content={"starting_scene": "Cinders drift over the bridge stones."},
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Crossing",
        save_id=save_id,
    )
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The echo answers from below.",
        provider="fake",
        model="fake-chat",
        content_rating="g",
        token_estimate=50,
        message_id=message_id,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    return save, message


def _full_roleplay_save(
    repositories: PersistenceRepositories,
) -> tuple[SaveRecord, MessageRecord, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Oracle of Glass",
        premise="A private audience with the oracle.",
        player_role="A careful petitioner",
        content={
            "character_name": "Oracle of Glass",
            "character_description": "An ancient diviner bound to mirrored halls.",
            "character_physical_description": (
                "Tall, still, mirrored silver eyes, white hair, blue glass robes."
            ),
            "character_personality": "Patient, precise, and unsettlingly kind.",
            "character_voice": "Soft ceremonial phrases.",
            "setup_line": "The oracle is waiting behind a curtain of beads.",
            "opening_message": "The oracle studies your reflection before your face.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Audience")
    repositories.set_app_setting(ROLEPLAY_SHARED_MODE_SETTING, False)
    opening_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle studies your reflection before your face.",
        provider="fake",
        model="fake-chat",
        content_rating="g",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Oracle of Glass",
        aliases=[],
        role="An ancient diviner bound to mirrored halls.",
        known_state="The oracle is waiting behind a curtain of beads.",
        met=True,
        appearance="Tall, still, mirrored silver eyes, white hair, blue glass robes.",
        visual_notes="Tall, still, mirrored silver eyes, white hair, blue glass robes.",
        personality="Patient, precise, and unsettlingly kind.",
        voice="Soft ceremonial phrases.",
        relationships={},
        status="present at scenario start",
        location_id=None,
        private_notes="",
        source_message_id=opening_message.id,
        protected_from_maintenance=True,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    return save, opening_message, character.id


def _persist_test_image_asset(
    repositories: PersistenceRepositories,
    *,
    media_dir: Path,
    save_id: str,
    source_message_id: str,
    filename: str,
    prompt: str,
    media_type: str = "image",
    status: str = "succeeded",
    metadata: dict[str, object] | None = None,
) -> MediaAssetRecord:
    relative_path = Path(save_id) / source_message_id / filename
    asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=source_message_id,
        type=media_type,
        path=relative_path.as_posix(),
        thumbnail_path=None,
        prompt=prompt,
        provider="fake",
        model="fake-image",
        status=status,
        mime_type="video/mp4" if media_type == "video" else None,
        metadata=metadata,
    )
    output_path = media_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_VALID_PNG_BYTES)
    return asset


def _mark_character_present(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    message_id: str,
    character_id: str,
) -> None:
    repositories.replace_message_scene_presence(
        save_id,
        message_id,
        [character_id],
        source="context_snapshot",
    )


def _persist_character_reference(
    repositories: PersistenceRepositories,
    *,
    media_dir: Path,
    save_id: str,
    source_message_id: str,
    character_id: str,
    filename: str = "reference.png",
) -> MediaAssetRecord:
    reference = _persist_test_image_asset(
        repositories,
        media_dir=media_dir,
        save_id=save_id,
        source_message_id=source_message_id,
        filename=filename,
        prompt="character reference",
        metadata={"kind": "character_reference", "character_id": character_id},
    )
    repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character_id,
        target_type="media_asset",
        target_id=reference.id,
        relation="reference_image",
    )
    return reference


def _configure_image_fallback(
    repositories: PersistenceRepositories,
    *,
    enabled: bool,
    capabilities: list[str] | None = None,
) -> None:
    repositories.set_app_setting("image_fallback_enabled", enabled)
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback/image",
        display_name="Fallback Fallback Image",
        capabilities=capabilities or ["image_generation", "fallback_marker"],
    )
    repositories.set_model_preference(
        task="image_fallback",
        provider="fallback",
        model_id="fallback/image",
    )


def _configure_image_edit_fallback(
    repositories: PersistenceRepositories,
    *,
    enabled: bool,
    capabilities: list[str] | None = None,
) -> None:
    repositories.set_app_setting("image_fallback_enabled", enabled)
    repositories.save_provider_model(
        provider="fallback-edit",
        model_id="fallback/edit",
        display_name="Fallback Edit Image",
        capabilities=capabilities or ["image_to_image", "fallback_marker"],
    )
    repositories.set_model_preference(
        task=IMAGE_EDIT_FALLBACK_PURPOSE,
        provider="fallback-edit",
        model_id="fallback/edit",
    )


def _configure_video_fallback(
    repositories: PersistenceRepositories,
    *,
    enabled: bool,
    capabilities: list[str],
) -> None:
    repositories.set_app_setting("video_fallback_enabled", enabled)
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback/video",
        display_name="Fallback Fallback Video",
        capabilities=capabilities,
    )
    repositories.set_model_preference(
        task="video_fallback",
        provider="fallback",
        model_id="fallback/video",
    )


def _chat_request_context(request: ChatRequest) -> str:
    parts = [
        request.scenario_instructions,
        request.summary or "",
        *request.retrieved_state,
        *request.retrieved_memories,
        *(message.body for message in request.messages),
    ]
    return "\n".join(part for part in parts if part)


def _chat_request_system_message(request: ChatRequest) -> str:
    system_messages = [
        message.body for message in request.messages if message.role == "system"
    ]
    assert len(system_messages) == 1
    return system_messages[0]


def _image_generation_jobs(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[dict[str, Any]]:
    return _media_jobs(repositories, save_id, "image_generation")


def _video_generation_jobs(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[dict[str, Any]]:
    return _media_jobs(repositories, save_id, "video_generation")


def _media_jobs(
    repositories: PersistenceRepositories,
    save_id: str,
    job_type: str,
) -> list[dict[str, Any]]:
    rows = repositories.connection.execute(
        """
        SELECT status, payload_json, result_json, error
        FROM jobs
        WHERE save_id = ? AND type = ?
        ORDER BY created_at, rowid
        """,
        (save_id, job_type),
    )
    jobs: list[dict[str, Any]] = []
    for row in rows:
        jobs.append(
            {
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
                "result": (
                    json.loads(row["result_json"])
                    if row["result_json"] is not None
                    else None
                ),
                "error": row["error"],
            }
        )
    return jobs


def _asset_path(media_dir: Path, persisted_path: str) -> Path:
    path = Path(persisted_path)
    if path.is_absolute():
        return path
    return media_dir / path


def _assert_private_thumbnail_if_present(
    *,
    media_dir: Path,
    asset: MediaAssetRecord,
) -> None:
    if asset.thumbnail_path is None:
        return
    assert asset.thumbnail_path != asset.path
    thumbnail_path = _asset_path(media_dir, asset.thumbnail_path)
    assert thumbnail_path.resolve().is_relative_to(media_dir.resolve())
    assert thumbnail_path.is_file()
    _assert_private_modes(thumbnail_path)


def _assert_private_modes(path: Path) -> None:
    if os.name == "nt":
        return
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
