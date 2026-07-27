from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, cast

import pytest

from bragi.interaction_mode import InteractionMode
from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import (
    CharacterRecord,
    CharacterTextMessageRecord,
    MediaAssetRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.chat_rendering import chat_system_body
from bragi.providers.contracts import (
    CHAT_TURN_DIRECTIVE_PURPOSE_CHARACTER_TEXT,
    ChatRequest,
    ChatResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.providers.system_prompt import CHARACTER_TEXT_RESPONSE_STYLE_SECTION
from bragi.services.character_text_service import (
    CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
    CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
    CHARACTER_TEXTS_ENABLED_SETTING,
    CharacterTextService,
    _update_text_route,
)
from bragi.services.content_rating import set_user_content_rating
from bragi.services.content_safety_service import (
    ContentSafetyAction,
    ContentSafetyResult,
    ContentSafetyService,
)
from bragi.services.media_service import CharacterTextUploadedPhoto
from bragi.services.phrase_denylist import (
    GENERATED_PHRASE_DENYLIST_SETTING,
    SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
)
from bragi.services.prompt_inspection import PromptInspectionStore
from bragi.services.text_script_policy import (
    SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
    SCRIPT_GUARD_MODE_OFF,
    SCRIPT_GUARD_MODE_SETTING,
    SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT,
)


class RecordingTextProvider:
    provider_name = "fake"

    def __init__(
        self,
        response_body: str = "Sounds good. Meet me by the arcade after class?",
    ) -> None:
        self.chat_requests: list[ChatRequest] = []
        self.response_body = response_body

    async def validate_config(self) -> ProviderConfigStatus:
        return ProviderConfigStatus(
            provider="fake",
            configured=True,
            authenticated=True,
        )

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider="fake",
                model_id="fake-chat",
                display_name="Fake Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=32768,
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        return ChatResponse(
            body=self.response_body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        return StructuredOutputResponse(
            data={
                "action": "allow",
                "category": "none",
                "reason": "The text stays within the content ceiling.",
                "minimum_rating": "g",
            },
            provider=request.provider,
            model_id=request.model_id,
        )


class SequenceTextProvider(RecordingTextProvider):
    def __init__(self, response_bodies: tuple[str, ...]) -> None:
        super().__init__(response_body="")
        self.response_bodies = response_bodies

    async def chat(self, request: ChatRequest) -> ChatResponse:
        request_index = len(self.chat_requests)
        self.chat_requests.append(request)
        body = self.response_bodies[min(request_index, len(self.response_bodies) - 1)]
        return ChatResponse(
            body=body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )


class RecordingContentSafetyService:
    def __init__(self) -> None:
        self.ratings: list[tuple[str, bool]] = []

    async def review_narration(
        self,
        *,
        body: str,
        content_rating: str,
        fade_to_black_enabled: bool,
        **_kwargs: object,
    ) -> ContentSafetyResult:
        self.ratings.append((content_rating, fade_to_black_enabled))
        return ContentSafetyResult(
            body=body,
            action=ContentSafetyAction.ALLOW,
            minimum_rating="g",
            agent_ran=content_rating != "unrated",
        )


class FailingTextProvider(RecordingTextProvider):
    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise RuntimeError("text provider failed")


class BlankTextProvider(RecordingTextProvider):
    def __init__(self) -> None:
        super().__init__(response_body=" \n\t ")


class StructuredTextWorldProvider(RecordingTextProvider):
    provider_name = "fake"

    def __init__(
        self,
        *,
        response_body: str = "I promised I would bring the repair notes.",
        structured_data: dict[str, object] | Exception,
    ) -> None:
        super().__init__(response_body=response_body)
        self.structured_data = structured_data
        self.structured_requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        if request.schema_name == "content_safety_review":
            return await super().generate_structured_output(request)
        if isinstance(self.structured_data, Exception):
            raise self.structured_data
        return StructuredOutputResponse(
            data=self.structured_data,
            provider=request.provider,
            model_id=request.model_id,
        )


class QueuedStructuredTextProvider(RecordingTextProvider):
    provider_name = "fake"

    def __init__(
        self,
        *,
        chat_responses: list[str],
        structured_responses: list[dict[str, object]] | Exception,
    ) -> None:
        super().__init__(response_body="")
        self.chat_responses = list(chat_responses)
        self.structured_responses = structured_responses
        self.structured_requests: list[StructuredOutputRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        body = self.chat_responses.pop(0)
        return ChatResponse(
            body=body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        if request.schema_name == "content_safety_review":
            return await super().generate_structured_output(request)
        if isinstance(self.structured_responses, Exception):
            raise self.structured_responses
        return StructuredOutputResponse(
            data=self.structured_responses.pop(0),
            provider=request.provider,
            model_id=request.model_id,
        )


class RecordingCharacterTextMediaRunner:
    def __init__(
        self,
        repositories: PersistenceRepositories,
        *,
        fail: bool = False,
    ) -> None:
        self.repositories = repositories
        self.fail = fail
        self.character_calls: list[tuple[str, str, str]] = []
        self.object_calls: list[tuple[str, str, str]] = []
        self.upload_calls: list[tuple[str, bytes, str | None]] = []
        self.cleanup_calls: list[str] = []

    async def upload_character_text_player_photo(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        sender_character_id: str,
        image_bytes: bytes,
        filename: str | None = None,
        retry_progress_callback: object | None = None,
    ) -> CharacterTextUploadedPhoto:
        del retry_progress_callback
        self.upload_calls.append((text_message.id, image_bytes, filename))
        if self.fail:
            raise RuntimeError("vision provider failed")
        asset = self.repositories.create_media_asset(
            save_id=save_id,
            source_message_id=None,
            type="image",
            mime_type="image/png",
            provider="local",
            model="upload",
            path=f"{save_id}/{text_message.id}-uploaded.png",
            prompt="Uploaded text photo",
            status="succeeded",
            metadata={
                "kind": "character_text_uploaded_photo",
                "thread_id": text_message.thread_id,
                "text_message_id": text_message.id,
                "sender_character_id": sender_character_id,
            },
        )
        return CharacterTextUploadedPhoto(
            asset=asset,
            description="A cracked blue key lies beside a torn paper map.",
        )

    def cleanup_character_text_uploaded_photo(
        self,
        *,
        save_id: str,
        asset: MediaAssetRecord,
    ) -> None:
        self.cleanup_calls.append(asset.id)
        self.repositories.archive_media_asset_only(
            save_id=save_id,
            media_asset_id=asset.id,
        )

    async def generate_character_text_character_image(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        character: CharacterRecord,
        visual_prompt: str,
        scene_context: str,
        retry_progress_callback: object | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        del retry_progress_callback, current_user_id
        self.character_calls.append((text_message.id, visual_prompt, scene_context))
        if self.fail:
            raise RuntimeError("image provider failed")
        return self._create_asset(
            save_id=save_id,
            text_message=text_message,
            character=character,
            visual_prompt=visual_prompt,
            kind="character_text_character_image",
            path_suffix="character",
        )

    async def generate_character_text_object_context_image(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        character: CharacterRecord,
        visual_prompt: str,
        scene_context: str,
        retry_progress_callback: object | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        del retry_progress_callback, current_user_id
        self.object_calls.append((text_message.id, visual_prompt, scene_context))
        if self.fail:
            raise RuntimeError("image provider failed")
        return self._create_asset(
            save_id=save_id,
            text_message=text_message,
            character=character,
            visual_prompt=visual_prompt,
            kind="character_text_object_context_image",
            path_suffix="object",
        )

    def _create_asset(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        character: CharacterRecord,
        visual_prompt: str,
        kind: str,
        path_suffix: str,
    ) -> MediaAssetRecord:
        return self.repositories.create_media_asset(
            save_id=save_id,
            source_message_id=None,
            type="image",
            mime_type="image/png",
            provider="fake-image",
            model="fake-image-model",
            path=f"{save_id}/{text_message.id}-{path_suffix}.png",
            prompt=visual_prompt,
            status="succeeded",
            metadata={
                "kind": kind,
                "thread_id": text_message.thread_id,
                "text_message_id": text_message.id,
                "character_id": character.id,
            },
        )


class BlockingCharacterTextMediaRunner(RecordingCharacterTextMediaRunner):
    def __init__(self, repositories: PersistenceRepositories) -> None:
        super().__init__(repositories)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate_character_text_character_image(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        character: CharacterRecord,
        visual_prompt: str,
        scene_context: str,
        retry_progress_callback: object | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        self.started.set()
        await self.release.wait()
        return await super().generate_character_text_character_image(
            save_id=save_id,
            text_message=text_message,
            character=character,
            visual_prompt=visual_prompt,
            scene_context=scene_context,
            retry_progress_callback=retry_progress_callback,
            current_user_id=current_user_id,
        )

    async def generate_character_text_object_context_image(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        character: CharacterRecord,
        visual_prompt: str,
        scene_context: str,
        retry_progress_callback: object | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        self.started.set()
        await self.release.wait()
        return await super().generate_character_text_object_context_image(
            save_id=save_id,
            text_message=text_message,
            character=character,
            visual_prompt=visual_prompt,
            scene_context=scene_context,
            retry_progress_callback=retry_progress_callback,
            current_user_id=current_user_id,
        )


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_dating_saves_enable_character_texts_by_default(
    repositories: PersistenceRepositories,
) -> None:
    dating_save_id = _create_save_with_characters(
        repositories,
        scenario_type="dating_sim",
    )
    roleplay_save_id = _create_save_with_characters(
        repositories,
        scenario_type="full_roleplay",
    )

    service = CharacterTextService(repositories=repositories, providers={})

    assert service.is_enabled(dating_save_id) is True
    assert service.is_enabled(roleplay_save_id) is False

    repositories.set_scoped_setting(
        scope="save",
        key=CHARACTER_TEXTS_ENABLED_SETTING,
        value=True,
        scope_id=roleplay_save_id,
    )

    assert service.is_enabled(roleplay_save_id) is True


def test_storyteller_saves_disable_character_texts_even_when_enabled(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(
        repositories,
        scenario_type="dating_sim",
        interaction_mode=InteractionMode.STORYTELLER,
    )
    repositories.set_scoped_setting(
        scope="save",
        key=CHARACTER_TEXTS_ENABLED_SETTING,
        value=True,
        scope_id=save_id,
    )
    service = CharacterTextService(repositories=repositories, providers={})

    assert service.is_enabled(save_id) is False
    with pytest.raises(
        ValueError,
        match="unavailable in storyteller mode",
    ):
        service.prepare_spontaneous_text(
            save_id=save_id,
            character_id=next(
                character.id
                for character in repositories.list_characters(save_id)
                if not character.is_player_character
            ),
        )


def test_send_text_persists_side_channel_messages_without_chronicle_append(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk after class?",
        )
    )

    assert result.reply.body == "Sounds good. Meet me by the arcade after class?"
    details = repositories.load_save_details(save_id)
    assert details is not None
    assert details.messages == []
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=result.thread.id,
    )
    assert [(message.sender, message.body) for message in messages] == [
        ("player", "Can we talk after class?"),
        ("character", "Sounds good. Meet me by the arcade after class?"),
    ]
    assert messages[1].provider == "fake"
    assert messages[1].model == "fake-chat"
    provenance = repositories.list_character_text_provenance(
        save_id=save_id,
        thread_id=result.thread.id,
    )
    assert any(
        row.text_message_id == result.reply.id
        and row.target_type == "dating_route_state"
        for row in provenance
    )


@pytest.mark.parametrize(
    ("mode", "expected_requests", "expected_reply"),
    [
        (SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT, 2, "Meet me by the arcade."),
        (SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT, 2, "Meet me by the arcade."),
        (SCRIPT_GUARD_MODE_OFF, 1, "玩家喜欢简洁叙事。"),
    ],
)
def test_send_text_applies_script_guard_before_persisting_character_reply(
    repositories: PersistenceRepositories,
    mode: str,
    expected_requests: int,
    expected_reply: str,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, mode)
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceTextProvider(
        ("玩家喜欢简洁叙事。", "Meet me by the arcade."),
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk after class?",
        )
    )

    assert len(provider.chat_requests) == expected_requests
    if expected_requests == 2:
        assert "unsupported writing script" in (
            provider.chat_requests[1].regeneration_feedback
        )
    assert result.reply.body == expected_reply
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=result.thread.id,
    )
    assert [(message.sender, message.body) for message in messages] == [
        ("player", "Can we talk after class?"),
        ("character", expected_reply),
    ]


def test_send_text_applies_phrase_guard_before_delivering_character_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="save-only phrase",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceTextProvider(
        ("That's a save-only phrase.", "Meet me by the arcade after class?")
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk after class?",
        )
    )

    assert result.reply.body == "Meet me by the arcade after class?"
    assert len(provider.chat_requests) == 2
    assert "save-only phrase" in provider.chat_requests[1].regeneration_feedback


@pytest.mark.parametrize(
    ("rating", "expected_fade"),
    (("g", True), ("pg", True), ("unrated", False)),
)
def test_character_text_guard_retry_preserves_actor_content_policy(
    repositories: PersistenceRepositories,
    rating: str,
    expected_fade: bool,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    actor = repositories.create_user(
        username=f"actor-{rating}",
        role="user",
        password_hash="hash",
    )
    set_user_content_rating(
        repositories,
        user_id=actor.id,
        rating=rating,
    )
    repositories.set_app_setting(
        SCRIPT_GUARD_MODE_SETTING,
        SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
    )
    _configure_text_reply_model(repositories)
    provider = SequenceTextProvider(
        ("玩家喜欢简洁叙事。", "Meet me by the arcade."),
    )
    safety = RecordingContentSafetyService()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        content_safety_service=cast(ContentSafetyService, safety),
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk after class?",
            current_user_id=actor.id,
        )
    )

    assert [
        (request.content_rating, request.fade_to_black_enabled)
        for request in provider.chat_requests
    ] == [(rating, expected_fade), (rating, expected_fade)]
    assert safety.ratings == [
        (rating, False),
        (rating, expected_fade),
    ]


def test_queue_text_send_persists_pending_message_without_provider_call(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    queued = service.queue_text_send(
        save_id=save_id,
        character_id=npc.id,
        body="Can we talk after class?",
    )

    assert provider.chat_requests == []
    assert queued.player_message.delivery_status == "pending"
    assert queued.player_message.delivery_error is None
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=queued.thread.id,
    )
    assert [
        (message.sender, message.body, message.delivery_status)
        for message in messages
    ] == [
        ("player", "Can we talk after class?", "pending"),
    ]


def test_queue_spontaneous_text_persists_pending_character_without_provider_call(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(
        repositories,
        save_id,
        npc.id,
        player_has_character_number=True,
    )

    queued = service.queue_spontaneous_text(
        save_id=save_id,
        character_id=npc.id,
    )

    assert provider.chat_requests == []
    assert queued.message.sender == "character"
    assert queued.message.body == ""
    assert queued.message.delivery_status == "pending"
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=queued.thread.id,
    )
    assert [
        (message.id, message.sender, message.body, message.delivery_status)
        for message in messages
    ] == [
        (queued.message.id, "character", "", "pending"),
    ]


def test_mark_thread_read_updates_incoming_messages_and_contact_summary(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    player_message = repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Can we talk after class?",
    )
    first_reply = repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Meet me by the lockers.",
    )
    latest_reply = repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Bring the brass token.",
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={},
    )

    result = service.mark_thread_read(
        save_id=save_id,
        thread_id=thread.id,
        through_message_id=latest_reply.id,
    )

    assert result.updated_message_ids == (first_reply.id, latest_reply.id)
    assert [message.id for message in result.thread.messages] == [
        player_message.id,
        first_reply.id,
        latest_reply.id,
    ]
    assert result.thread.messages[0].read_at is None
    assert result.thread.messages[1].read_at is not None
    assert result.thread.messages[2].read_at is not None
    contact = next(
        item
        for item in service.build_model(save_id).contacts
        if item.thread_id == thread.id
    )
    assert contact.latest_message_id == latest_reply.id
    assert contact.latest_message_read_at is not None


def test_complete_queued_text_send_marks_sent_and_appends_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    queued = service.queue_text_send(
        save_id=save_id,
        character_id=npc.id,
        body="Can we talk after class?",
    )

    result = asyncio.run(
        service.complete_queued_text_send(
            save_id=save_id,
            player_message_id=queued.player_message.id,
        )
    )

    assert result.player_message.delivery_status == "sent"
    assert result.reply.body == "Sounds good. Meet me by the arcade after class?"
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=result.thread.id,
    )
    assert [(message.sender, message.delivery_status) for message in messages] == [
        ("player", "sent"),
        ("character", "sent"),
    ]


def test_complete_queued_text_send_includes_uploaded_photo_description(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(response_body="That looks like the old gate key.")
    media = RecordingCharacterTextMediaRunner(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    queued = service.queue_text_send(
        save_id=save_id,
        character_id=npc.id,
        body="Do you recognize this?",
    )

    result = asyncio.run(
        service.complete_queued_text_send(
            save_id=save_id,
            player_message_id=queued.player_message.id,
            uploaded_photo_bytes=b"fake-photo-bytes",
            uploaded_photo_filename="gate-key.png",
        )
    )

    assert result.player_message.delivery_status == "sent"
    assert media.upload_calls == [
        (queued.player_message.id, b"fake-photo-bytes", "gate-key.png")
    ]
    request = provider.chat_requests[0]
    player_body = request.messages[-1].body
    assert "Do you recognize this?" in player_body
    assert (
        "[Attached photo visible to recipient: "
        "A cracked blue key lies beside a torn paper map.]"
    ) in player_body
    assert "gate-key.png" not in player_body
    assert "fake-photo-bytes" not in player_body
    attachments = repositories.list_character_text_message_attachments(
        save_id=save_id,
        text_message_ids=(queued.player_message.id,),
    )
    assert [(row.kind, row.status, row.prompt) for row in attachments] == [
        (
            "uploaded_photo",
            "succeeded",
            "A cracked blue key lies beside a torn paper map.",
        )
    ]


def test_complete_queued_text_send_cleans_upload_when_attachment_insert_fails(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(response_body="That looks familiar.")
    media = RecordingCharacterTextMediaRunner(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    queued = service.queue_text_send(
        save_id=save_id,
        character_id=npc.id,
        body="Do you recognize this?",
    )

    def fail_attachment_insert(**_kwargs: object) -> NoReturn:
        raise RuntimeError("attachment insert failed")

    monkeypatch.setattr(
        repositories,
        "add_character_text_message_attachment",
        fail_attachment_insert,
    )

    with pytest.raises(RuntimeError, match="attachment insert failed"):
        asyncio.run(
            service.complete_queued_text_send(
                save_id=save_id,
                player_message_id=queued.player_message.id,
                uploaded_photo_bytes=b"fake-photo-bytes",
            )
        )

    assert media.upload_calls == [
        (queued.player_message.id, b"fake-photo-bytes", None)
    ]
    assert len(media.cleanup_calls) == 1
    assert repositories.list_media_assets(save_id) == []
    failed_message = repositories.get_character_text_message(
        save_id=save_id,
        message_id=queued.player_message.id,
    )
    assert failed_message is not None
    assert failed_message.delivery_status == "failed"
    assert failed_message.delivery_error == "attachment insert failed"
    assert provider.chat_requests == []


def test_send_thread_text_generates_capped_group_replies_from_willing_participants(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_group_text_save(repositories)
    npcs = [
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    ]
    for npc in npcs:
        _grant_player_has_number(repositories, save_id, npc.id)
    _configure_text_reply_model(repositories)
    _configure_group_response_assessment_model(repositories)
    thread = repositories.create_character_text_group_thread(
        save_id=save_id,
        title="Arcade Crew",
        character_ids=[npc.id for npc in npcs],
    )
    provider = QueuedStructuredTextProvider(
        chat_responses=[
            "Rowan: I can bring the spare tokens.",
            "Maya: I will cover the front desk.",
            "Toma: I will watch the alley.",
        ],
        structured_responses=[
            _group_response_decision(npc, should_respond=True, priority=index)
            for index, npc in enumerate(npcs, start=1)
        ],
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.send_thread_text(
            save_id=save_id,
            thread_id=thread.id,
            body="Can everyone help with the cabinet tonight?",
        )
    )

    assert result.player_message.delivery_status == "sent"
    assert [reply.body for reply in result.replies] == [
        "Rowan: I can bring the spare tokens.",
        "Maya: I will cover the front desk.",
        "Toma: I will watch the alley.",
    ]
    assert [reply.sender_character_id for reply in result.replies] == [
        npc.id for npc in npcs[:3]
    ]
    assert [
        request.schema_name
        for request in provider.structured_requests
        if request.schema_name == "character_text_group_response_assessment"
    ] == [
        "character_text_group_response_assessment",
        "character_text_group_response_assessment",
        "character_text_group_response_assessment",
        "character_text_group_response_assessment",
    ]
    assert len(provider.chat_requests) == 3
    for request, npc in zip(provider.chat_requests, npcs[:3], strict=True):
        system_body = chat_system_body(request)
        assert f"Target text character: {npc.name}" in system_body
        assert "Player character (do not portray): Mira" in system_body
        assert f"Only write as {npc.name}." in system_body
        assert "Do not write as Mira" in system_body
        other_names = ", ".join(
            other.name for other in npcs if other.id != npc.id
        )
        assert f"Other group participants (context only): {other_names}" in (
            system_body
        )
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=thread.id,
    )
    assert [(message.sender, message.sender_character_id) for message in messages] == [
        ("player", _player_character(repositories, save_id).id),
        ("character", npcs[0].id),
        ("character", npcs[1].id),
        ("character", npcs[2].id),
    ]


def test_complete_queued_thread_text_hides_group_reply_until_attachment_is_ready(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_group_text_save(repositories)
    npcs = [
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    ]
    for npc in npcs:
        _grant_player_has_number(repositories, save_id, npc.id)
    _configure_text_reply_model(repositories)
    _configure_group_response_assessment_model(repositories)
    thread = repositories.create_character_text_group_thread(
        save_id=save_id,
        title="Arcade Crew",
        character_ids=[npc.id for npc in npcs],
    )
    provider = QueuedStructuredTextProvider(
        chat_responses=["I can bring the spare tokens. Sending a photo."],
        structured_responses=[
            *(
                _group_response_decision(
                    npc,
                    should_respond=index == 0,
                    priority=index + 1,
                )
                for index, npc in enumerate(npcs)
            ),
            _attachment_decision(
                kind="object_context_image",
                visual_prompt="spare arcade tokens in a jacket pocket",
            ),
        ],
    )
    media = BlockingCharacterTextMediaRunner(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    queued = service.queue_thread_text_send(
        save_id=save_id,
        thread_id=thread.id,
        body="Can anyone bring spare tokens?",
    )

    async def complete_with_paused_attachment():
        task = asyncio.create_task(
            service.complete_queued_thread_text_send(
                save_id=save_id,
                player_message_id=queued.player_message.id,
            )
        )
        await asyncio.wait_for(media.started.wait(), timeout=1.0)
        messages = repositories.list_character_text_messages(
            save_id=save_id,
            thread_id=thread.id,
        )
        assert [
            (message.sender, message.body, message.delivery_status)
            for message in messages
        ] == [
            ("player", "Can anyone bring spare tokens?", "pending"),
            ("character", "", "pending"),
        ]
        media.release.set()
        return await asyncio.wait_for(task, timeout=1.0)

    result = asyncio.run(complete_with_paused_attachment())

    assert result.player_message.delivery_status == "sent"
    assert len(result.replies) == 1
    assert result.replies[0].body == "I can bring the spare tokens. Sending a photo."
    assert result.replies[0].attachments[0].status == "succeeded"


def test_send_thread_text_skips_group_replies_without_structured_assessment(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_group_text_save(repositories)
    npcs = [
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    ]
    for npc in npcs:
        _grant_player_has_number(repositories, save_id, npc.id)
    _configure_text_reply_model(repositories)
    _configure_group_response_assessment_model(repositories)
    thread = repositories.create_character_text_group_thread(
        save_id=save_id,
        title="Arcade Crew",
        character_ids=[npc.id for npc in npcs],
    )
    provider = QueuedStructuredTextProvider(
        chat_responses=["This should not be used."],
        structured_responses=RuntimeError("structured provider failed"),
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.send_thread_text(
            save_id=save_id,
            thread_id=thread.id,
            body="Can everyone help with the cabinet tonight?",
        )
    )

    assert result.player_message.delivery_status == "sent"
    assert result.replies == ()
    assert len(provider.structured_requests) == 2
    assert provider.chat_requests == []
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=thread.id,
    )
    assert [(message.sender, message.body) for message in messages] == [
        ("player", "Can everyone help with the cabinet tonight?"),
    ]


def test_group_thread_memory_preserves_actual_speakers_after_compaction(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_group_text_save(repositories)
    npcs = [
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    ]
    for npc in npcs:
        _grant_player_has_number(repositories, save_id, npc.id)
    _configure_text_reply_model(repositories)
    _configure_group_response_assessment_model(repositories)
    thread = repositories.create_character_text_group_thread(
        save_id=save_id,
        title="Arcade Crew",
        character_ids=[npc.id for npc in npcs],
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npcs[0].id,
        sender="character",
        sender_character_id=npcs[0].id,
        body="I promised to bring spare tokens.",
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npcs[1].id,
        sender="character",
        sender_character_id=npcs[1].id,
        body="I promised to bring the cabinet key.",
    )
    for index in range(31):
        repositories.append_character_text_message(
            save_id=save_id,
            thread_id=thread.id,
            character_id=None,
            sender="player",
            body=f"Filler group text {index}",
        )
    provider = QueuedStructuredTextProvider(
        chat_responses=["I still have the spare tokens."],
        structured_responses=[
            _group_response_decision(
                npc,
                should_respond=index == 0,
                priority=index + 1,
            )
            for index, npc in enumerate(npcs)
        ],
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(
        service.send_thread_text(
            save_id=save_id,
            thread_id=thread.id,
            body="Can everyone confirm what they have?",
        )
    )

    phone_context = "\n".join(provider.chat_requests[-1].phone_context)
    assert f"- {npcs[0].name}: I promised to bring spare tokens." in phone_context
    assert f"- {npcs[1].name}: I promised to bring the cabinet key." in phone_context
    assert "Arcade Crew: I promised" not in phone_context


def test_send_text_records_message_metadata(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Rowan and Mira stand beside the arcade prize counter.",
        objective="Choose whether to trade the brass token.",
        in_world_time="Friday evening after class",
        mood="soft competitive tension",
        present_character_ids=[npc.id],
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk after class?",
        )
    )

    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=result.thread.id,
    )
    assert messages[0].sender == "player"
    assert messages[0].in_world_sent_at == "Friday evening after class"
    assert messages[0].delivered_at is not None
    assert messages[1].sender == "character"
    assert messages[1].reply_to_message_id == messages[0].id
    assert messages[1].in_world_sent_at == "Friday evening after class"
    assert messages[1].delivered_at is not None
    assert result.player_message.delivered_at == messages[0].delivered_at
    assert result.reply.reply_to_message_id == messages[0].id
    assert all(
        "Friday evening after class" not in message.body
        for message in provider.chat_requests[-1].messages
    )
    assert "Friday evening after class" in "\n".join(
        provider.chat_requests[-1].current_scene_recap
    )


def test_send_text_keeps_in_world_timestamp_out_of_provider_history(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Rowan waits beside the arcade prize counter.",
        objective="Choose whether to meet after class.",
        in_world_time="Friday evening after class",
        present_character_ids=[npc.id],
    )
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(response_body="I can meet after class.")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Did you get the note?",
        in_world_sent_at="evening",
    )

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk after class?",
        )
    )

    request_bodies = [message.body for message in provider.chat_requests[-1].messages]
    assert "Did you get the note?" in request_bodies
    assert "Can we talk after class?" in request_bodies
    assert all(not body.startswith("Sent at ") for body in request_bodies)
    assert result.player_message.in_world_sent_at == "Friday evening after class"
    assert result.reply.in_world_sent_at == "Friday evening after class"


def test_send_text_strips_leaked_sent_at_prefix_from_character_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(
        response_body="Sent at evening: Sure, meet me after class."
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk after class?",
        )
    )

    assert result.reply.body == "Sure, meet me after class."
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=result.thread.id,
    )
    assert messages[-1].body == "Sure, meet me after class."


def test_send_text_keeps_normal_sent_at_reply_content(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(
        response_body="Sent at the old address: nobody answered."
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Did the courier reply?",
        )
    )

    assert result.reply.body == "Sent at the old address: nobody answered."


def test_send_text_sanitizes_stale_leaked_sent_at_prefix_in_thread_memory(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(response_body="I still remember.")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Sent at evening: I promised to bring the moon ladder code.",
    )
    for index in range(31):
        repositories.append_character_text_message(
            save_id=save_id,
            thread_id=thread.id,
            character_id=npc.id,
            sender="player" if index % 2 == 0 else "character",
            body=f"Small talk filler {index}",
        )

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="What did you promise?",
        )
    )

    phone_context = "\n".join(provider.chat_requests[-1].phone_context)
    refreshed_thread = repositories.get_character_text_thread(
        save_id=save_id,
        thread_id=thread.id,
    )
    assert refreshed_thread is not None
    assert "I promised to bring the moon ladder code." in phone_context
    assert "Sent at evening" not in phone_context
    assert "I promised to bring the moon ladder code." in refreshed_thread.memory_body
    assert "Sent at evening" not in refreshed_thread.memory_body


def test_send_text_keeps_reply_text_only_when_attachment_decision_is_none(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    _configure_text_attachment_decision_model(repositories)
    provider = StructuredTextWorldProvider(
        response_body="Sounds good. Meet me by the arcade after class?",
        structured_data=_attachment_decision(kind="none"),
    )
    media = RecordingCharacterTextMediaRunner(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk after class?",
        )
    )

    assert result.reply.attachments == ()
    assert media.character_calls == []
    assert media.object_calls == []
    assert any(
        request.schema_name == "character_text_image_attachment_decision"
        for request in provider.structured_requests
    )
    assert repositories.list_character_text_message_attachments(
        save_id=save_id,
        text_message_ids=[result.reply.id],
    ) == []


def test_send_text_generates_character_image_attachment_for_npc_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    _configure_text_attachment_decision_model(repositories)
    provider = StructuredTextWorldProvider(
        response_body="Found my old jacket. Sending proof.",
        structured_data=_attachment_decision(
            kind="character_image",
            visual_prompt="Rowan mirror selfie in a patched denim jacket",
            wearing="patched denim jacket over a faded arcade shirt",
            current_action="taking a mirror selfie",
            facial_expression="guarded half-smile",
        ),
    )
    media = RecordingCharacterTextMediaRunner(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="What jacket?",
        )
    )

    assert len(media.character_calls) == 1
    assert media.object_calls == []
    assert media.character_calls[0][1].startswith(
        "Rowan mirror selfie in a patched denim jacket"
    )
    assert "Character visual direction for Rowan" in media.character_calls[0][1]
    assert "Wearing: patched denim jacket over a faded arcade shirt" in (
        media.character_calls[0][1]
    )
    assert "Current action/pose: taking a mirror selfie" in media.character_calls[0][1]
    assert "Facial expression: guarded half-smile" in media.character_calls[0][1]
    assert "Found my old jacket" in media.character_calls[0][2]
    request = next(
        request
        for request in provider.structured_requests
        if request.schema_name == "character_text_image_attachment_decision"
    )
    properties = request.schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) >= {
        "wearing",
        "current_action",
        "facial_expression",
    }
    system_text = request.messages[0].body
    assert "what the character is wearing" in system_text
    assert "what the character is currently doing" in system_text
    assert "facial expression" in system_text
    attachment = result.reply.attachments[0]
    assert attachment.kind == "character_image"
    assert attachment.status == "succeeded"
    assert attachment.media_asset_id is not None
    assert attachment.mime_type == "image/png"
    assert attachment.provider == "fake-image"
    assert attachment.model == "fake-image-model"
    assert attachment.prompt_preview.startswith(
        "Rowan mirror selfie in a patched denim jacket"
    )
    rows = repositories.list_character_text_message_attachments(
        save_id=save_id,
        text_message_ids=[result.reply.id],
    )
    assert [(row.kind, row.status, row.media_asset_id) for row in rows] == [
        ("character_image", "succeeded", attachment.media_asset_id),
    ]
    thread = service.get_thread_model(save_id=save_id, thread_id=result.thread.id)
    assert thread.messages[-1].attachments[0].media_asset_id == (
        attachment.media_asset_id
    )


def test_complete_queued_text_send_hides_reply_until_attachment_is_ready(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    _configure_text_attachment_decision_model(repositories)
    provider = StructuredTextWorldProvider(
        response_body="Found my old jacket. Sending proof.",
        structured_data=_attachment_decision(
            kind="character_image",
            visual_prompt="Rowan mirror selfie in a patched denim jacket",
        ),
    )
    media = BlockingCharacterTextMediaRunner(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    queued = service.queue_text_send(
        save_id=save_id,
        character_id=npc.id,
        body="What jacket?",
    )

    async def complete_with_paused_attachment():
        task = asyncio.create_task(
            service.complete_queued_text_send(
                save_id=save_id,
                player_message_id=queued.player_message.id,
            )
        )
        await asyncio.wait_for(media.started.wait(), timeout=1.0)
        messages = repositories.list_character_text_messages(
            save_id=save_id,
            thread_id=queued.thread.id,
        )
        assert [
            (message.sender, message.body, message.delivery_status)
            for message in messages
        ] == [
            ("player", "What jacket?", "pending"),
            ("character", "", "pending"),
        ]
        assert repositories.list_character_text_message_attachments(
            save_id=save_id,
            text_message_ids=[messages[1].id],
        ) == []
        contact = next(
            contact
            for contact in service.build_model(save_id).contacts
            if contact.id == npc.id
        )
        assert contact.latest_message_sender == "player"
        assert contact.latest_message_body == "What jacket?"
        media.release.set()
        return await asyncio.wait_for(task, timeout=1.0)

    result = asyncio.run(complete_with_paused_attachment())

    assert result.reply.body == "Found my old jacket. Sending proof."
    assert result.player_message.delivery_status == "sent"
    assert result.reply.delivery_status == "sent"
    assert result.reply.attachments[0].status == "succeeded"
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=queued.thread.id,
    )
    assert [
        (message.sender, message.body, message.delivery_status)
        for message in messages
    ] == [
        ("player", "What jacket?", "sent"),
        ("character", "Found my old jacket. Sending proof.", "sent"),
    ]


def test_send_text_generates_object_context_attachment_for_npc_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    _configure_text_attachment_decision_model(repositories)
    provider = StructuredTextWorldProvider(
        response_body="I found the ticket stub under the cabinet.",
        structured_data=_attachment_decision(
            kind="object_context_image",
            visual_prompt="creased arcade ticket stub on a dusty cabinet",
        ),
    )
    media = RecordingCharacterTextMediaRunner(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Did you find anything?",
        )
    )

    assert media.character_calls == []
    assert len(media.object_calls) == 1
    attachment = result.reply.attachments[0]
    assert attachment.kind == "object_context_image"
    assert attachment.status == "succeeded"
    assert attachment.media_asset_id is not None
    assert attachment.prompt_preview == "creased arcade ticket stub on a dusty cabinet"


def test_send_text_records_failed_attachment_without_failing_text_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    _configure_text_attachment_decision_model(repositories)
    provider = StructuredTextWorldProvider(
        response_body="I found the ticket stub under the cabinet.",
        structured_data=_attachment_decision(
            kind="object_context_image",
            visual_prompt="creased arcade ticket stub on a dusty cabinet",
        ),
    )
    media = RecordingCharacterTextMediaRunner(repositories, fail=True)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Did you find anything?",
        )
    )

    assert result.reply.body == "I found the ticket stub under the cabinet."
    assert result.reply.delivery_status == "sent"
    attachment = result.reply.attachments[0]
    assert attachment.kind == "object_context_image"
    assert attachment.status == "failed"
    assert attachment.media_asset_id is None
    assert "image provider failed" in (attachment.error or "")
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=result.thread.id,
    )
    assert [(message.sender, message.delivery_status) for message in messages] == [
        ("player", "sent"),
        ("character", "sent"),
    ]


def test_complete_queued_text_send_marks_failed_when_provider_fails(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": FailingTextProvider()},
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    queued = service.queue_text_send(
        save_id=save_id,
        character_id=npc.id,
        body="Can we talk after class?",
    )

    with pytest.raises(RuntimeError, match="text provider failed"):
        asyncio.run(
            service.complete_queued_text_send(
                save_id=save_id,
                player_message_id=queued.player_message.id,
            )
        )

    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=queued.thread.id,
    )
    assert [(message.sender, message.delivery_status) for message in messages] == [
        ("player", "failed"),
    ]
    assert "text provider failed" in (messages[0].delivery_error or "")


def test_complete_queued_text_send_rejects_blank_provider_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = BlankTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    queued = service.queue_text_send(
        save_id=save_id,
        character_id=npc.id,
        body="Can we talk after class?",
    )

    with pytest.raises(ValueError, match="empty reply"):
        asyncio.run(
            service.complete_queued_text_send(
                save_id=save_id,
                player_message_id=queued.player_message.id,
            )
        )

    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=queued.thread.id,
    )
    assert [
        (message.sender, message.body, message.delivery_status)
        for message in messages
    ] == [
        ("player", "Can we talk after class?", "failed"),
    ]
    assert "empty reply" in (messages[0].delivery_error or "")
    assert len(provider.chat_requests) == 1


def test_failed_text_send_does_not_grant_contact_or_future_prompt_context(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": FailingTextProvider()},
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    failed = service.queue_text_send(
        save_id=save_id,
        character_id=npc.id,
        body="This undelivered text should stay private.",
    )

    with pytest.raises(RuntimeError, match="text provider failed"):
        asyncio.run(
            service.complete_queued_text_send(
                save_id=save_id,
                player_message_id=failed.player_message.id,
            )
        )

    assert service.can_character_proactively_text(
        save_id=save_id,
        character_id=npc.id,
    ) is False

    provider = RecordingTextProvider(response_body="I only saw this new one.")
    retry_service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    asyncio.run(
        retry_service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="This delivered text should be answered.",
        )
    )

    prompt_bodies = [message.body for message in provider.chat_requests[-1].messages]
    assert "This delivered text should be answered." in prompt_bodies
    assert "This undelivered text should stay private." not in prompt_bodies


def test_thread_memory_ignores_failed_text_messages(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(response_body="I remember the sent parts.")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Failed promise about the moon ladder.",
        delivery_status="failed",
        delivery_error="provider failed",
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Let's call the west arcade our ferris wheel spot.",
    )
    for index in range(31):
        repositories.append_character_text_message(
            save_id=save_id,
            thread_id=thread.id,
            character_id=npc.id,
            sender="player" if index % 2 == 0 else "character",
            body=f"Small talk filler {index}",
        )

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Same place?",
        )
    )

    refreshed_thread = repositories.get_character_text_thread(
        save_id=save_id,
        thread_id=thread.id,
    )
    assert refreshed_thread is not None
    assert "ferris wheel" in refreshed_thread.memory_body
    assert "moon ladder" not in refreshed_thread.memory_body


def test_send_text_models_include_markdown_blocks(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="**Absolutely.** Bring `notes` after class.",
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk about **algorithms**?",
        )
    )
    thread = service.get_thread_model(save_id=save_id, thread_id=result.thread.id)

    assert result.player_message.markdown_blocks[0].spans[1].kind == "strong"
    assert result.player_message.markdown_blocks[0].spans[1].text == "algorithms"
    assert result.reply.markdown_blocks[0].spans[0].kind == "strong"
    assert result.reply.markdown_blocks[0].spans[0].text == "Absolutely."
    assert thread.messages[1].markdown_blocks == result.reply.markdown_blocks


def test_contact_model_propagates_latest_message_markdown_blocks_and_body(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="**Absolutely.** Bring `notes` after class."
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk about **algorithms**?",
        )
    )

    model = service.build_model(save_id)
    contact = next(
        contact for contact in model.contacts if contact.id == npc.id
    )
    repair_contact = next(
        contact for contact in model.repair_contacts if contact.id == npc.id
    )

    assert contact.latest_message_body == "**Absolutely.** Bring `notes` after class."
    assert len(contact.latest_message_markdown_blocks) == 1
    assert contact.latest_message_markdown_blocks[0].spans[0].kind == "strong"
    assert contact.latest_message_markdown_blocks[0].spans[0].text == "Absolutely."
    assert repair_contact.latest_message_body == contact.latest_message_body
    assert repair_contact.latest_message_markdown_blocks == (
        contact.latest_message_markdown_blocks
    )


def test_contact_model_omits_latest_message_markdown_blocks_when_no_messages(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    model = CharacterTextService(
        repositories=repositories,
        providers={},
    ).build_model(save_id)
    contact = next(
        contact for contact in model.contacts if contact.id == npc.id
    )

    assert contact.latest_message_id is None
    assert contact.latest_message_body == ""
    assert contact.latest_message_markdown_blocks == ()


def test_send_text_builds_plain_chat_request_with_character_context(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="What are you thinking about?",
        )
    )

    request = provider.chat_requests[-1]
    assert request.provider == "fake"
    assert request.model_id == "fake-chat"
    assert request.messages[-1].role == "player"
    assert request.messages[-1].body == "What are you thinking about?"
    assert request.messages[-1].speaker_name == "Mira"
    assert request.response_style_section == CHARACTER_TEXT_RESPONSE_STYLE_SECTION
    system_body = chat_system_body(request)
    assert "Character text response style:" in system_body
    assert "- Send only the message body, like a normal phone text." in system_body
    assert "- Do not prefix the reply with >." in system_body
    assert "- Do not wrap the whole reply in quotation marks." in system_body
    assert "- Do not use Markdown, italics, action narration, or sender labels." in (
        system_body
    )
    assert "- Do not include timestamps or Sent at labels." in system_body
    assert "- Put dialogue in quotation marks." not in system_body
    assert "- Format text messages with > at the beginning of each message." not in (
        system_body
    )
    assert "Return one in-world text reply as Rowan." in request.scenario_instructions
    assert "Target text character: Rowan" in system_body
    assert "Player character (do not portray): Mira" in system_body
    assert "Only write as Rowan." in system_body
    assert "Do not write as Mira" in system_body
    assert request.turn_directive_purpose == CHAT_TURN_DIRECTIVE_PURPOSE_CHARACTER_TEXT
    assert "One-shot instruction for this character text message." in system_body
    assert "narrator response" not in system_body
    assert "explicit timeskip flow" not in system_body
    assert "Rowan" in "\n".join(request.current_scene_recap)
    assert "guarded but curious" in "\n".join(request.current_scene_recap)


def test_send_text_includes_visible_recent_chronicle_context(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan noticed Mira pocket the brass arcade token after practice.",
    )
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The prize counter sign buzzes awake while Rowan waits nearby.",
    )
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(response_body="I saw the token. Keep it safe.")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Did anything happen after practice?",
        )
    )

    chronicle_context = "\n".join(provider.chat_requests[-1].retrieved_recent_messages)
    assert "[message:" in chronicle_context
    assert "Rowan noticed Mira pocket the brass arcade token" in chronicle_context
    assert "The prize counter sign buzzes awake" in chronicle_context


def test_send_text_excludes_chronicle_context_hidden_from_character(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    visible = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan saw Mira leave the cabinet key on the public counter.",
    )
    hidden = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira secretly hid the spare arcade key in her locker.",
    )
    repositories.add_message_visibility(
        save_id=save_id,
        message_id=hidden.id,
        character_id=npc.id,
        visibility="not_visible",
        source="unit_test",
        evidence="Private player action.",
    )
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(response_body="I only saw the counter key.")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Where is the key?",
        )
    )

    chronicle_context = "\n".join(provider.chat_requests[-1].retrieved_recent_messages)
    assert visible.id in chronicle_context
    assert "public counter" in chronicle_context
    assert hidden.id not in chronicle_context
    assert "spare arcade key" not in chronicle_context


def test_send_text_retries_when_reply_uses_player_sender_label(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = SequenceTextProvider(
        (
            "Mira: I can meet after class.",
            "I can meet after class.",
        )
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can you meet after class?",
        )
    )

    assert result.reply.body == "I can meet after class."
    assert len(provider.chat_requests) == 2
    retry_feedback = provider.chat_requests[1].regeneration_feedback
    assert "Rowan" in retry_feedback
    assert "Mira" in retry_feedback
    assert "Do not write as Mira" in retry_feedback


def test_send_text_fails_when_identity_retry_still_speaks_as_player(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = SequenceTextProvider(
        (
            "Mira: I can meet after class.",
            "I\u2019m Mira and I can meet after class.",
        )
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    queued = service.queue_text_send(
        save_id=save_id,
        character_id=npc.id,
        body="Can you meet after class?",
    )

    with pytest.raises(ValueError, match="player character identity"):
        asyncio.run(
            service.complete_queued_text_send(
                save_id=save_id,
                player_message_id=queued.player_message.id,
            )
        )

    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=queued.thread.id,
    )
    assert [(message.sender, message.delivery_status) for message in messages] == [
        ("player", "failed"),
    ]
    assert "player character identity" in (messages[0].delivery_error or "")
    assert len(provider.chat_requests) == 2


def test_send_text_allows_typographic_player_possessive_without_identity_retry(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(
        response_body=(
            "This is Mira\u2019s umbrella. It\u2019s Mira\u2011adjacent chaos, "
            "but I found it."
        )
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Did you find the umbrella?",
        )
    )

    assert result.reply.body == (
        "This is Mira\u2019s umbrella. It\u2019s Mira\u2011adjacent chaos, "
        "but I found it."
    )
    assert len(provider.chat_requests) == 1


def test_send_text_includes_thread_memory_when_old_messages_drop_from_history(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Same place. I remember.")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Let's call the west arcade our moon ladder spot.",
    )
    for index in range(31):
        repositories.append_character_text_message(
            save_id=save_id,
            thread_id=thread.id,
            character_id=npc.id,
            sender="player" if index % 2 == 0 else "character",
            body=f"Small talk filler {index}",
        )

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Same place?",
        )
    )

    request = provider.chat_requests[-1]
    recent_history = "\n".join(message.body for message in request.messages)
    phone_context = "\n".join(request.phone_context)
    refreshed_thread = repositories.get_character_text_thread(
        save_id=save_id,
        thread_id=thread.id,
    )
    assert refreshed_thread is not None
    assert "moon ladder" not in recent_history
    assert "Phone thread memory" in phone_context
    assert "moon ladder" in phone_context
    assert "moon ladder" in refreshed_thread.memory_body


def test_send_text_omits_scene_snapshot_for_off_scene_character(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    player = next(
        character
        for character in repositories.list_characters(save_id)
        if character.is_player_character
    )
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    arcade = repositories.add_location(
        save_id=save_id,
        name="North Arcade",
        status="closed for cleanup",
    )
    npc = repositories.update_character(
        replace(
            npc,
            status="waiting for Mira's reply",
            current_intent="Buy a replacement cabinet token before practice.",
            location_id=arcade.id,
        )
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Mira and Toma discuss the hidden rooftop confession.",
        objective="Keep Rowan unaware of the rooftop confession.",
        in_world_time="Friday 9:41 PM after the festival",
        day_of_week="Friday",
        time_of_day="evening",
        world_day_index=12,
        mood="private panic under neon rain",
        present_character_ids=[player.id],
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can you meet later?",
        )
    )

    request = provider.chat_requests[-1]
    phone_context = "\n".join(request.phone_context)
    assert "Phone context contact: Rowan" in phone_context
    assert "Phone scene presence: off-scene from the active scene" in phone_context
    assert "Known character status: waiting for Mira's reply" in phone_context
    assert (
        "Known character current intent: Buy a replacement cabinet token before "
        "practice."
    ) in phone_context
    assert "Known character location: North Arcade; status=closed for cleanup" in (
        phone_context
    )
    assert (
        "Current world time: Friday evening at 21:41; world day index 12"
        in phone_context
    )
    assert "rooftop confession" not in phone_context
    assert "Friday 9:41 PM" not in phone_context
    assert "private panic" not in phone_context
    prompt_text = "\n".join(
        (
            *request.phone_context,
            *request.current_scene_recap,
            chat_system_body(request),
        )
    )
    assert "Rowan" in prompt_text
    assert "guarded but curious" in prompt_text
    assert "rooftop confession" not in prompt_text
    assert "Friday 9:41 PM" not in prompt_text
    assert "private panic" not in prompt_text


def test_send_text_includes_scene_snapshot_for_present_character(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Rowan and Mira stand beside the arcade prize counter.",
        objective="Choose whether to trade the brass token.",
        in_world_time="Friday morning",
        time_of_day="afternoon",
        day_of_week="saturday",
        world_day_index=3,
        world_time_clock_minutes=14 * 60 + 5,
        mood="soft competitive tension",
        present_character_ids=[npc.id],
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="What do you think?",
        )
    )

    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=result.thread.id,
    )
    assert messages[0].in_world_sent_at == (
        "Saturday afternoon at 14:05; world day index 3"
    )
    assert messages[1].in_world_sent_at == (
        "Saturday afternoon at 14:05; world day index 3"
    )
    request = provider.chat_requests[-1]
    assert "Phone scene presence: present in the active scene" in "\n".join(
        request.phone_context
    )
    prompt_text = "\n".join(
        (
            *request.phone_context,
            *request.current_scene_recap,
            chat_system_body(request),
        )
    )
    assert "Visible scene context" in prompt_text
    assert "arcade prize counter" in prompt_text
    assert "brass token" in prompt_text
    assert "Saturday afternoon at 14:05; world day index 3" in prompt_text
    assert "Friday morning" not in prompt_text
    assert "soft competitive tension" in prompt_text


def test_send_text_includes_only_character_scoped_knowledge(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    other = repositories.add_character(
        save_id=save_id,
        name="Toma",
        role="classmate",
        met=True,
    )
    known_memory = repositories.add_memory(
        save_id=save_id,
        body="Rowan knows Mira fixed the west arcade cabinet.",
        tags=["rowan", "arcade"],
    )
    private_memory = repositories.add_memory(
        save_id=save_id,
        body="Toma knows Mira hid the festival letter.",
        tags=["toma", "letter"],
    )
    repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=npc.id,
        target_type="memory",
        target_id=known_memory.id,
        knowledge_state="knows",
    )
    repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=other.id,
        target_type="memory",
        target_id=private_memory.id,
        knowledge_state="knows",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Do you remember the arcade?",
        )
    )

    prompt_text = "\n".join(provider.chat_requests[-1].current_scene_recap)
    assert "Character-scoped knowledge (Rowan knows) memory" in prompt_text
    assert "west arcade cabinet" in prompt_text
    assert "festival letter" not in prompt_text


def test_send_text_captures_character_text_prompt_when_inspection_enabled(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    store = PromptInspectionStore()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        prompt_inspection_store=store,
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk after class?",
        )
    )

    entries = store.entries_for_message(result.reply.id)
    prompt_text = store.prompt_for_message(result.reply.id) or ""
    assert [entry.kind for entry in entries] == ["character_text_prompt"]
    assert entries[0].title == "Character text prompt"
    assert "Character text prompt" in prompt_text
    assert "Character profile" in prompt_text
    assert "Current scene recap" in prompt_text
    assert "Can we talk after class?" in prompt_text


def test_send_text_includes_character_texting_style_when_present(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.update_character(
        replace(
            npc,
            texting_style=(
                "Short lowercase bursts, ellipses when nervous, rarely emojis."
            ),
        )
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Still awake?",
        )
    )

    prompt_text = "\n".join(provider.chat_requests[-1].current_scene_recap)
    assert (
        "Character texting style: Short lowercase bursts, ellipses when nervous, "
        "rarely emojis."
    ) in prompt_text


def test_send_text_includes_character_cooperation_conditions(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.update_character(
        replace(
            npc,
            attitude_toward_player="Hostile until Mira apologizes for the prank.",
            cooperation_conditions=(
                "Replies usefully only after Mira admits what happened."
            ),
        )
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can you just tell me where the key is?",
        )
    )

    request = provider.chat_requests[-1]
    current_scene_recap = "\n".join(request.current_scene_recap)
    phone_context = "\n".join(request.phone_context)
    assert (
        "Character profile attitude toward player: Hostile until Mira apologizes "
        "for the prank."
    ) in current_scene_recap
    assert (
        "Character profile cooperation conditions: Replies usefully only after "
        "Mira admits what happened."
    ) in current_scene_recap
    assert (
        "Known character cooperation conditions: Replies usefully only after "
        "Mira admits what happened."
    ) in phone_context
    assert (
        "Known character attitude toward player: Hostile until Mira apologizes "
        "for the prank."
    ) in phone_context


def test_send_text_applies_structured_text_world_updates(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    _save_fake_structured_model(repositories, model_id="fake-context")
    provider = StructuredTextWorldProvider(
        structured_data={
            "memories": [
                {
                    "body": "Rowan promised to bring the repair notes.",
                    "tags": ["promise", "rowan"],
                    "importance": 0.88,
                    "source_text_message_id": "reply",
                    "character_id": npc.id,
                    "knowledge_state": "knows",
                    "acquisition_method": "told",
                    "evidence_quote": "I promised I would bring the repair notes.",
                }
            ],
            "active_threads": [
                {
                    "title": "Repair note exchange",
                    "description": "Rowan needs to bring the repair notes.",
                    "priority": 3,
                    "visibility": "private",
                    "source_text_message_id": "reply",
                }
            ],
            "character_updates": [
                {
                    "character_id": npc.id,
                    "relationships": {"Mira": "trusts Mira with repair plans"},
                    "source_text_message_id": "reply",
                }
            ],
            "dating_route_updates": [
                {
                    "npc_character_id": npc.id,
                    "trust_level": "warming",
                    "next_reasonable_step": "Follow up about the repair notes.",
                    "source_text_message_id": "reply",
                }
            ],
        }
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can you bring the notes?",
        )
    )

    assert result.world_update is not None
    assert result.world_update.status == "applied"
    assert result.world_update.memory_count == 1
    assert result.world_update.active_thread_count == 1
    assert result.world_update.character_count == 1
    assert result.world_update.dating_route_count == 1
    assert result.world_update.knowledge_edge_count == 1
    assert any(
        request.schema_name == "character_text_world_update"
        for request in provider.structured_requests
    )
    details = repositories.load_save_details(save_id)
    assert details is not None
    assert details.messages == []
    memories = repositories.list_memories(save_id)
    assert [memory.body for memory in memories] == [
        "Rowan promised to bring the repair notes."
    ]
    source_ref = f"character_text_message:{result.reply.id}"
    assert memories[0].source_message_id is None
    assert memories[0].source_message_ids == [source_ref]
    edge = repositories.list_character_knowledge_edges(save_id)[0]
    assert edge.target_type == "memory"
    assert edge.target_id == memories[0].id
    assert edge.source_message_id is None
    assert edge.source_message_ids == [source_ref]
    assert repositories.list_active_threads(save_id)[0].source_message_id is None
    updated_npc = repositories.get_character(npc.id)
    assert updated_npc is not None
    assert updated_npc.relationships["Mira"] == "trusts Mira with repair plans"
    route = repositories.list_dating_route_states(save_id)[0]
    assert route.trust_level == "warming"
    assert route.next_reasonable_step == "Follow up about the repair notes."
    audit = repositories.list_context_update_audit(save_id)
    assert any(row.source_message_ids == [source_ref] for row in audit)
    provenance = repositories.list_character_text_provenance(
        save_id=save_id,
        text_message_id=result.reply.id,
    )
    assert {row.target_type for row in provenance} >= {
        "memory",
        "active_thread",
        "character",
        "dating_route_state",
        "character_knowledge_edge",
    }


def test_send_text_queues_retry_when_structured_text_world_update_fails(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    _save_fake_structured_model(repositories, model_id="fake-context")
    provider = StructuredTextWorldProvider(
        structured_data=RuntimeError("structured provider failed"),
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can you bring the notes?",
        )
    )

    assert result.reply.body == "I promised I would bring the repair notes."
    assert result.world_update is not None
    assert result.world_update.status == "retry_queued"
    assert repositories.list_memories(save_id) == []
    retry_jobs = [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "character_text_world_update_retry"
    ]
    assert len(retry_jobs) == 1
    assert retry_jobs[0].payload["text_message_ids"] == [
        result.player_message.id,
        result.reply.id,
    ]


def test_list_model_separates_visible_contacts_from_repair_candidates(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(
        repositories,
        scenario_type="dating_sim",
    )
    rowan = next(
        character
        for character in repositories.list_characters(save_id)
        if character.name == "Rowan"
    )
    maya = repositories.add_character(
        save_id=save_id,
        name="Maya",
        role="club president",
        met=True,
    )
    repositories.add_character(
        save_id=save_id,
        name="Toma",
        role="neighbor",
        met=False,
    )
    _grant_player_has_number(repositories, save_id, rowan.id)
    _grant_character_has_player_number(repositories, save_id, maya.id)

    model = CharacterTextService(
        repositories=repositories,
        providers={},
    ).build_model(save_id)

    contacts = {
        contact.name: (
            contact.is_player_character,
            contact.player_has_character_number,
            contact.character_has_player_number,
            contact.thread_id,
        )
        for contact in model.contacts
    }
    repair_contacts = {
        contact.name: (
            contact.is_player_character,
            contact.player_has_character_number,
            contact.character_has_player_number,
            contact.thread_id,
        )
        for contact in model.repair_contacts
    }

    assert model.enabled is True
    assert contacts == {
        "Maya": (False, False, True, None),
        "Rowan": (False, True, False, None),
    }
    assert repair_contacts == {
        "Maya": (False, False, True, None),
        "Rowan": (False, True, False, None),
        "Toma": (False, False, False, None),
    }
    assert repositories.list_character_text_threads(save_id) == []


def test_contact_model_exposes_directional_permission_provenance(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    player = next(
        character
        for character in repositories.list_characters(save_id)
        if character.is_player_character
    )
    npc = _npc_character(repositories, save_id)
    source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan gives Mira his number after class.",
    )
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        source_message_id=source.id,
    )

    service = CharacterTextService(
        repositories=repositories,
        providers={},
    )
    model = service.build_model(save_id)

    contact = next(contact for contact in model.contacts if contact.id == npc.id)
    assert contact.player_number_permission.allowed is True
    assert contact.player_number_permission.source == "chronicle"
    assert contact.player_number_permission.source_message_id == source.id
    assert "Chronicle" in contact.player_number_permission.reason
    assert contact.character_number_permission.allowed is False
    assert contact.character_number_permission.source == "none"
    repair_contact = next(
        contact for contact in model.repair_contacts if contact.id == npc.id
    )
    assert repair_contact.player_number_permission == contact.player_number_permission

    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    text_source = repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="I'll text you when I arrive.",
    )
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
        character_has_player_number=True,
        source_text_message_id=text_source.id,
    )

    contact = next(
        contact
        for contact in service.build_model(save_id).contacts
        if contact.id == npc.id
    )
    assert contact.player_number_permission.source == "chronicle"
    assert contact.character_number_permission.allowed is True
    assert contact.character_number_permission.source == "text_message"
    assert contact.character_number_permission.source_text_message_id == text_source.id


def test_update_contact_state_manually_sets_and_clears_text_permissions(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={},
    )

    granted = service.update_contact_state(
        save_id=save_id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )

    granted_contact = next(
        contact for contact in granted.contacts if contact.id == npc.id
    )
    assert granted_contact.player_has_character_number is True
    assert granted_contact.character_has_player_number is True
    assert repositories.character_text_outbound_allowed(
        save_id=save_id,
        character_id=npc.id,
    ) is True
    assert service.can_character_proactively_text(
        save_id=save_id,
        character_id=npc.id,
    ) is True

    corrected = service.update_contact_state(
        save_id=save_id,
        character_id=npc.id,
        player_has_character_number=False,
        character_has_player_number=False,
    )

    assert all(contact.id != npc.id for contact in corrected.contacts)
    corrected_contact = next(
        contact for contact in corrected.repair_contacts if contact.id == npc.id
    )
    assert corrected_contact.player_has_character_number is False
    assert corrected_contact.character_has_player_number is False
    assert repositories.character_text_outbound_allowed(
        save_id=save_id,
        character_id=npc.id,
    ) is False
    assert service.can_character_proactively_text(
        save_id=save_id,
        character_id=npc.id,
    ) is False
    assert repositories.list_character_text_threads(save_id) == []


def test_update_contact_state_rejects_disabled_or_invalid_contacts(
    repositories: PersistenceRepositories,
) -> None:
    disabled_save_id = _create_save_with_characters(
        repositories,
        scenario_type="full_roleplay",
    )
    disabled_npc = next(
        character
        for character in repositories.list_characters(disabled_save_id)
        if not character.is_player_character
    )
    enabled_save_id = _create_save_with_characters(
        repositories,
        scenario_type="dating_sim",
    )
    player = next(
        character
        for character in repositories.list_characters(enabled_save_id)
        if character.is_player_character
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={},
    )

    with pytest.raises(ValueError, match="not enabled"):
        service.update_contact_state(
            save_id=disabled_save_id,
            character_id=disabled_npc.id,
            player_has_character_number=True,
            character_has_player_number=True,
        )

    with pytest.raises(ValueError, match="Unknown textable character id"):
        service.update_contact_state(
            save_id=enabled_save_id,
            character_id=player.id,
            player_has_character_number=True,
            character_has_player_number=True,
        )


def test_list_model_exposes_reference_image_for_contacts_with_photo(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(
        repositories,
        scenario_type="dating_sim",
    )
    rowan = next(
        character
        for character in repositories.list_characters(save_id)
        if character.name == "Rowan"
    )
    _grant_player_has_number(repositories, save_id, rowan.id)
    asset = repositories.create_media_asset(
        save_id=save_id,
        type="image",
        path=f"{save_id}/uploads/rowan-reference.png",
        prompt="Rowan's portrait",
        provider="local",
        model="upload",
        status="succeeded",
        mime_type="image/png",
        metadata={
            "kind": "character_reference",
            "source": "uploaded",
            "character_id": rowan.id,
        },
    )
    repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=rowan.id,
        target_type="media_asset",
        target_id=asset.id,
        relation="reference_image",
    )

    model = CharacterTextService(
        repositories=repositories,
        providers={},
    ).build_model(save_id)

    rowan_contact = next(
        contact for contact in model.contacts if contact.id == rowan.id
    )
    assert rowan_contact.name == "Rowan"
    assert rowan_contact.reference_image is not None
    assert rowan_contact.reference_image.media_asset_id == asset.id
    assert rowan_contact.reference_image.mime_type == "image/png"


def test_list_model_leaves_reference_image_null_when_no_photo(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(
        repositories,
        scenario_type="dating_sim",
    )
    rowan = next(
        character
        for character in repositories.list_characters(save_id)
        if character.name == "Rowan"
    )
    _grant_player_has_number(repositories, save_id, rowan.id)

    model = CharacterTextService(
        repositories=repositories,
        providers={},
    ).build_model(save_id)

    rowan_contact = next(
        contact for contact in model.contacts if contact.id == rowan.id
    )
    assert rowan_contact.reference_image is None


def test_contact_model_exposes_character_contact_name_when_set(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(
        repositories,
        scenario_type="dating_sim",
    )
    rowan = next(
        character
        for character in repositories.list_characters(save_id)
        if character.name == "Rowan"
    )
    repositories.update_character(
        replace(rowan, contact_name="Row (schoolyard)")
    )
    _grant_player_has_number(repositories, save_id, rowan.id)

    model = CharacterTextService(
        repositories=repositories,
        providers={},
    ).build_model(save_id)
    rowan_contact = next(
        contact for contact in model.contacts if contact.id == rowan.id
    )

    assert rowan_contact.name == "Rowan"
    assert rowan_contact.contact_name == "Row (schoolyard)"
    repaired_rowan = next(
        contact for contact in model.repair_contacts if contact.id == rowan.id
    )
    assert repaired_rowan.contact_name == "Row (schoolyard)"


def test_contact_model_keeps_empty_contact_name_for_character_name_fallback(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(
        repositories,
        scenario_type="dating_sim",
    )
    rowan = next(
        character
        for character in repositories.list_characters(save_id)
        if character.name == "Rowan"
    )
    assert rowan.contact_name == ""
    _grant_player_has_number(repositories, save_id, rowan.id)

    model = CharacterTextService(
        repositories=repositories,
        providers={},
    ).build_model(save_id)
    rowan_contact = next(
        contact for contact in model.contacts if contact.id == rowan.id
    )

    assert rowan_contact.contact_name == ""
    assert rowan_contact.name == "Rowan"


def test_send_text_requires_player_to_have_character_number(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )

    with pytest.raises(ValueError, match="does not have this character's number"):
        asyncio.run(
            service.send_text(
                save_id=save_id,
                character_id=npc.id,
                body="Can we talk after class?",
            )
        )

    assert provider.chat_requests == []
    assert repositories.list_character_text_threads(save_id) == []


def test_send_text_grants_character_the_player_number(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    assert service.can_character_proactively_text(
        save_id=save_id,
        character_id=npc.id,
    ) is False

    asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk after class?",
        )
    )

    assert service.can_character_proactively_text(
        save_id=save_id,
        character_id=npc.id,
    ) is True


def test_send_text_consumes_matching_dating_route_proactive_trigger(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="I can talk about the arcade after class."
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_text(
            save_id=save_id,
            character_id=npc.id,
            body="Can we talk about your arcade plans?",
        )
    )

    route = repositories.list_dating_route_states(save_id)[0]
    triggers = repositories.list_character_text_proactive_triggers(save_id)
    assert len(triggers) == 1
    trigger = triggers[0]
    assert trigger.trigger_key.startswith(f"dating_route:{route.id}:")
    assert trigger.trigger_type == "dating_route"
    assert trigger.thread_id == result.reply.thread_id
    assert trigger.text_message_id == result.reply.id
    assert trigger.source_type == "dating_route_state"
    assert trigger.source_id == route.id
    assert trigger.reason == "Ask about Rowan's arcade plans."

    proactive = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert proactive.status == "skipped"
    assert proactive.reason == "duplicate_suppressed"
    assert proactive.candidate_count == 1
    assert len(provider.chat_requests) == 1
    assert len(repositories.list_character_text_messages(save_id=save_id)) == 2


def test_update_text_route_is_idempotent_for_same_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    player_message = repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Can we talk about your arcade plans?",
    )
    reply = repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="I can talk about the arcade after class.",
        provider="fake",
        model="fake-chat",
    )
    initial_route = repositories.list_dating_route_states(save_id)[0]

    _update_text_route(
        repositories=repositories,
        save_id=save_id,
        character=npc,
        player_message=player_message,
        reply=reply,
    )
    _update_text_route(
        repositories=repositories,
        save_id=save_id,
        character=npc,
        player_message=player_message,
        reply=reply,
    )

    route = repositories.list_dating_route_states(save_id)[0]
    assert route.completed_interactions == initial_route.completed_interactions + 1
    audits = [
        audit
        for audit in repositories.list_context_update_audit(save_id)
        if audit.entity_type == "dating_route_state"
        and audit.operation == "text_exchange"
    ]
    assert len(audits) == 1


def test_proactive_text_skips_when_character_lacks_player_number(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "skipped"
    assert result.reason == "no_candidate"
    assert provider.chat_requests == []
    assert repositories.list_character_text_messages(save_id=save_id) == []


def test_proactive_text_chance_zero_disables_route_trigger(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        value=0,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Still want to meet?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "skipped"
    assert result.reason == "proactive_texts_disabled"
    assert result.candidate_count == 0
    assert provider.chat_requests == []
    assert repositories.list_character_text_messages(save_id=save_id) == []
    assert repositories.list_character_text_proactive_triggers(save_id) == []


def test_proactive_text_persists_side_channel_message_without_chronicle_append(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="Still want to meet by the arcade?"
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "sent"
    assert result.message is not None
    assert result.thread is not None
    assert result.message.body == "Still want to meet by the arcade?"
    details = repositories.load_save_details(save_id)
    assert details is not None
    assert details.messages == []
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=result.thread.id,
    )
    assert [(message.sender, message.body) for message in messages] == [
        ("character", "Still want to meet by the arcade?"),
    ]
    request = provider.chat_requests[-1]
    system_body = chat_system_body(request)
    assert "Target text character: Rowan" in system_body
    assert "Player character (do not portray): Mira" in system_body
    assert "Only write as Rowan." in system_body
    assert "Do not write as Mira" in system_body
    assert request.response_style_section == CHARACTER_TEXT_RESPONSE_STYLE_SECTION
    assert "Character text response style:" in system_body
    assert "- Send only the message body, like a normal phone text." in system_body
    assert "- Do not prefix the reply with >." in system_body
    assert "- Do not wrap the whole reply in quotation marks." in system_body
    assert "- Do not use Markdown, italics, action narration, or sender labels." in (
        system_body
    )
    assert "- Do not include timestamps or Sent at labels." in system_body
    assert "- Put dialogue in quotation marks." not in system_body
    assert "- Format text messages with > at the beginning of each message." not in (
        system_body
    )
    prompt_text = "\n".join(
        [
            request.scenario_instructions,
            request.turn_directive,
            *request.current_scene_recap,
        ]
    ).casefold()
    assert "json" not in prompt_text
    model = service.build_model(save_id)
    contact_flags = [
        (
            contact.name,
            contact.player_has_character_number,
            contact.character_has_player_number,
        )
        for contact in model.contacts
    ]
    assert contact_flags == [("Rowan", True, True)]
    repair_contact = next(
        contact for contact in model.repair_contacts if contact.name == "Rowan"
    )
    assert repair_contact.player_has_character_number is True
    assert repair_contact.latest_message_id == result.message.id
    assert repair_contact.latest_message_sender == "character"


def test_proactive_text_rejects_blank_provider_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": BlankTextProvider()},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    with pytest.raises(ValueError, match="empty reply"):
        asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    thread = next(
        thread
        for thread in repositories.list_character_text_threads(save_id)
        if thread.character_id == npc.id
    )
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=thread.id,
    )
    assert [
        (message.sender, message.body, message.delivery_status)
        for message in messages
    ] == [
        ("character", "", "failed"),
    ]
    assert "empty reply" in (messages[0].delivery_error or "")


def test_spontaneous_text_persists_character_message_without_player_message(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="I just remembered the arcade machine code."
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(
        repositories,
        save_id,
        npc.id,
        player_has_character_number=True,
    )

    result = asyncio.run(
        service.send_spontaneous_text(
            save_id=save_id,
            character_id=npc.id,
        )
    )

    assert result.message.body == "I just remembered the arcade machine code."
    assert result.thread.messages[-1].id == result.message.id
    details = repositories.load_save_details(save_id)
    assert details is not None
    assert details.messages == []
    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=result.thread.id,
    )
    assert [
        (message.id, message.sender, message.body, message.delivery_status)
        for message in messages
    ] == [
        (
            result.message.id,
            "character",
            "I just remembered the arcade machine code.",
            "sent",
        ),
    ]
    request = provider.chat_requests[-1]
    assert "spontaneous phone text" in request.turn_directive
    assert "UI" not in request.turn_directive
    system_body = chat_system_body(request)
    assert "Target text character: Rowan" in system_body
    assert "Player character (do not portray): Mira" in system_body
    assert "Only write as Rowan." in system_body
    assert "Do not write as Mira" in system_body


def test_complete_queued_spontaneous_text_hides_reply_until_attachment_is_ready(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    _configure_text_attachment_decision_model(repositories)
    provider = StructuredTextWorldProvider(
        response_body="I found the ticket stub under the cabinet.",
        structured_data=_attachment_decision(
            kind="object_context_image",
            visual_prompt="creased arcade ticket stub on a dusty cabinet",
        ),
    )
    media = BlockingCharacterTextMediaRunner(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    _grant_character_has_player_number(
        repositories,
        save_id,
        npc.id,
        player_has_character_number=True,
    )
    queued = service.queue_spontaneous_text(
        save_id=save_id,
        character_id=npc.id,
    )

    async def complete_with_paused_attachment():
        task = asyncio.create_task(
            service.complete_queued_spontaneous_text(
                save_id=save_id,
                text_message_id=queued.message.id,
            )
        )
        await asyncio.wait_for(media.started.wait(), timeout=1.0)
        messages = repositories.list_character_text_messages(
            save_id=save_id,
            thread_id=queued.thread.id,
        )
        assert [
            (message.sender, message.body, message.delivery_status)
            for message in messages
        ] == [
            ("character", "", "pending"),
        ]
        contact = next(
            contact
            for contact in service.build_model(save_id).contacts
            if contact.id == npc.id
        )
        assert contact.latest_message_id is None
        assert contact.latest_message_body == ""
        media.release.set()
        return await asyncio.wait_for(task, timeout=1.0)

    result = asyncio.run(complete_with_paused_attachment())

    assert result.message.body == "I found the ticket stub under the cabinet."
    assert result.message.delivery_status == "sent"
    assert result.message.attachments[0].kind == "object_context_image"
    assert result.message.attachments[0].status == "succeeded"


def test_complete_queued_spontaneous_text_marks_failed_when_provider_fails(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": FailingTextProvider()},
    )
    _grant_character_has_player_number(
        repositories,
        save_id,
        npc.id,
        player_has_character_number=True,
    )
    queued = service.queue_spontaneous_text(
        save_id=save_id,
        character_id=npc.id,
    )

    with pytest.raises(RuntimeError, match="text provider failed"):
        asyncio.run(
            service.complete_queued_spontaneous_text(
                save_id=save_id,
                text_message_id=queued.message.id,
            )
        )

    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=queued.thread.id,
    )
    assert [
        (message.id, message.sender, message.delivery_status)
        for message in messages
    ] == [
        (queued.message.id, "character", "failed"),
    ]
    assert "text provider failed" in (messages[0].delivery_error or "")


def test_complete_queued_spontaneous_text_rejects_blank_provider_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": BlankTextProvider()},
    )
    _grant_character_has_player_number(
        repositories,
        save_id,
        npc.id,
        player_has_character_number=True,
    )
    queued = service.queue_spontaneous_text(
        save_id=save_id,
        character_id=npc.id,
    )

    with pytest.raises(ValueError, match="empty reply"):
        asyncio.run(
            service.complete_queued_spontaneous_text(
                save_id=save_id,
                text_message_id=queued.message.id,
            )
        )

    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=queued.thread.id,
    )
    assert [
        (message.sender, message.body, message.delivery_status)
        for message in messages
    ] == [
        ("character", "", "failed"),
    ]
    assert "empty reply" in (messages[0].delivery_error or "")


def test_spontaneous_text_rejects_thread_with_active_inbound_delivery(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": RecordingTextProvider()},
    )
    _grant_character_has_player_number(
        repositories,
        save_id,
        npc.id,
        player_has_character_number=True,
    )
    service.queue_spontaneous_text(
        save_id=save_id,
        character_id=npc.id,
    )

    with pytest.raises(ValueError, match="already pending"):
        service.queue_spontaneous_text(
            save_id=save_id,
            character_id=npc.id,
        )


def test_proactive_text_skips_thread_with_active_delivery(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(response_body="Still want to meet?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="",
        delivery_status="pending",
    )

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "skipped"
    assert result.reason == "thread_busy"
    assert provider.chat_requests == []


def test_spontaneous_text_includes_phone_context_when_scene_unknown(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Did you get home okay?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(
        repositories,
        save_id,
        npc.id,
        player_has_character_number=True,
    )

    result = asyncio.run(
        service.send_spontaneous_text(
            save_id=save_id,
            character_id=npc.id,
        )
    )

    assert result.message.body == "Did you get home okay?"
    phone_context = "\n".join(provider.chat_requests[-1].phone_context)
    assert "Target text character: Rowan" in phone_context
    assert "Player character (do not portray): Mira" in phone_context
    assert "Only write as Rowan." in phone_context
    assert "Do not write as Mira" in phone_context
    assert "Phone context contact: Rowan" in phone_context
    assert "Phone scene presence: active scene unknown" in phone_context
    assert "Active-scene details omitted from phone context are not known" in (
        phone_context
    )


def test_spontaneous_text_requires_character_to_have_player_number(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_player_has_number(repositories, save_id, npc.id)

    with pytest.raises(ValueError, match="Character does not have the player's number"):
        asyncio.run(
            service.send_spontaneous_text(
                save_id=save_id,
                character_id=npc.id,
            )
        )

    assert provider.chat_requests == []
    assert repositories.list_character_text_messages(save_id=save_id) == []


def test_proactive_text_generates_attachment_for_npc_message(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    _configure_text_attachment_decision_model(repositories)
    provider = StructuredTextWorldProvider(
        response_body="I found the ticket stub under the cabinet.",
        structured_data=_attachment_decision(
            kind="object_context_image",
            visual_prompt="creased arcade ticket stub on a dusty cabinet",
        ),
    )
    media = RecordingCharacterTextMediaRunner(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "sent"
    assert result.message is not None
    assert repositories.character_text_outbound_allowed(
        save_id=save_id,
        character_id=npc.id,
    ) is True
    assert len(media.object_calls) == 1
    attachment = result.message.attachments[0]
    assert attachment.kind == "object_context_image"
    assert attachment.status == "succeeded"
    assert attachment.media_asset_id is not None
    rows = repositories.list_character_text_message_attachments(
        save_id=save_id,
        text_message_ids=[result.message.id],
    )
    assert [(row.kind, row.status, row.media_asset_id) for row in rows] == [
        ("object_context_image", "succeeded", attachment.media_asset_id),
    ]


def test_proactive_text_hides_reply_until_attachment_is_ready(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    _configure_text_reply_model(repositories)
    _configure_text_attachment_decision_model(repositories)
    provider = StructuredTextWorldProvider(
        response_body="I found the ticket stub under the cabinet.",
        structured_data=_attachment_decision(
            kind="object_context_image",
            visual_prompt="creased arcade ticket stub on a dusty cabinet",
        ),
    )
    media = BlockingCharacterTextMediaRunner(repositories)
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
        media_service=media,
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    async def send_with_paused_attachment():
        task = asyncio.create_task(
            service.send_proactive_text_after_turn(save_id=save_id)
        )
        await asyncio.wait_for(media.started.wait(), timeout=1.0)
        messages = repositories.list_character_text_messages(save_id=save_id)
        assert [
            (message.sender, message.body, message.delivery_status)
            for message in messages
        ] == [
            ("character", "", "pending"),
        ]
        media.release.set()
        return await asyncio.wait_for(task, timeout=1.0)

    result = asyncio.run(send_with_paused_attachment())

    assert result.status == "sent"
    assert result.message is not None
    assert result.message.body == "I found the ticket stub under the cabinet."
    assert result.message.attachments[0].status == "succeeded"


def test_proactive_text_omits_scene_snapshot_for_off_scene_character(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    player = next(
        character
        for character in repositories.list_characters(save_id)
        if character.is_player_character
    )
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Mira privately accepts Toma's festival confession.",
        objective="Keep the confession secret from Rowan.",
        in_world_time="Sunday 11:12 PM",
        mood="breathless secrecy",
        present_character_ids=[player.id],
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Still want to meet tomorrow?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)
    route = repositories.list_dating_route_states(save_id)[0]
    repositories.upsert_dating_route_state(
        save_id=save_id,
        player_character_id=route.player_character_id,
        npc_character_id=route.npc_character_id,
        stage=route.stage,
        comfort_with_intimacy="comfortable with early physical closeness",
        pacing_preference="direct if chemistry is mutual",
        known_boundaries=["no public pressure"],
    )

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "sent"
    phone_context = "\n".join(provider.chat_requests[-1].phone_context)
    assert "Phone context contact: Rowan" in phone_context
    assert "Comfort with intimacy: comfortable with early physical closeness" in (
        phone_context
    )
    assert "Pacing: direct if chemistry is mutual" in phone_context
    assert "Known boundaries: no public pressure" in phone_context
    assert "Phone scene presence: off-scene from the active scene" in phone_context
    assert "festival confession" not in phone_context
    assert "Sunday 11:12 PM" not in phone_context
    assert "breathless secrecy" not in phone_context
    prompt_text = "\n".join(
        (
            *provider.chat_requests[-1].phone_context,
            *provider.chat_requests[-1].current_scene_recap,
            chat_system_body(provider.chat_requests[-1]),
        )
    )
    assert "Rowan" in prompt_text
    assert "festival confession" not in prompt_text
    assert "Sunday 11:12 PM" not in prompt_text
    assert "breathless secrecy" not in prompt_text


def test_proactive_text_suppresses_duplicate_recent_character_body(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Still want to meet by the arcade?",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    provider = StructuredTextWorldProvider(
        response_body="Still want to meet by the arcade?",
        structured_data={"memories": []},
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "skipped"
    assert result.reason == "duplicate_body_suppressed"
    assert result.trigger_key
    assert result.thread is not None
    assert result.message is None
    assert len(provider.chat_requests) == 1
    assert [request.schema_name for request in provider.structured_requests] == [
        "content_safety_review"
    ]
    assert [
        message.body
        for message in repositories.list_character_text_messages(
            save_id=save_id,
            thread_id=thread.id,
        )
    ] == ["Still want to meet by the arcade?"]
    triggers = repositories.list_character_text_proactive_triggers(save_id)
    assert len(triggers) == 1
    assert triggers[0].trigger_key == result.trigger_key
    assert triggers[0].thread_id == thread.id
    assert triggers[0].text_message_id is None


def test_proactive_text_suppresses_whitespace_normalized_duplicate_body(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Still want to meet by the arcade?",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="  Still  want to meet\nby the arcade?  "
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "skipped"
    assert result.reason == "duplicate_body_suppressed"
    assert len(provider.chat_requests) == 1
    assert len(
        repositories.list_character_text_messages(save_id=save_id, thread_id=thread.id)
    ) == 1


def test_proactive_text_suppresses_case_normalized_duplicate_body(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Still want to meet by the arcade?",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="still want to meet by the arcade?"
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "skipped"
    assert result.reason == "duplicate_body_suppressed"
    assert result.message is None
    assert [
        message.body
        for message in repositories.list_character_text_messages(
            save_id=save_id,
            thread_id=thread.id,
        )
    ] == ["Still want to meet by the arcade?"]


def test_proactive_text_suppresses_punctuation_normalized_duplicate_body(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Still want to meet by the arcade?",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="Still want to meet by the arcade."
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "skipped"
    assert result.reason == "duplicate_body_suppressed"
    assert len(
        repositories.list_character_text_messages(save_id=save_id, thread_id=thread.id)
    ) == 1


def test_proactive_text_suppresses_near_identical_recent_body(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Still want to meet by the arcade after practice?",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="Still want to meet by the arcade after practice tonight?"
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "skipped"
    assert result.reason == "duplicate_body_suppressed"
    assert len(
        repositories.list_character_text_messages(save_id=save_id, thread_id=thread.id)
    ) == 1


def test_proactive_text_sends_distinct_recent_body(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Still want to meet by the arcade?",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="I found the ticket stub under the cabinet."
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "sent"
    assert result.message is not None
    assert [
        message.body
        for message in repositories.list_character_text_messages(
            save_id=save_id,
            thread_id=thread.id,
        )
    ] == [
        "Still want to meet by the arcade?",
        "I found the ticket stub under the cabinet.",
    ]


def test_proactive_text_ignores_matching_recent_player_body(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Still want to meet by the arcade?",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="Still want to meet by the arcade?"
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "sent"
    assert result.message is not None
    assert [
        (message.sender, message.body)
        for message in repositories.list_character_text_messages(
            save_id=save_id,
            thread_id=thread.id,
        )
    ] == [
        ("player", "Still want to meet by the arcade?"),
        ("character", "Still want to meet by the arcade?"),
    ]


def test_proactive_text_suppresses_duplicate_trigger(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(
        response_body="Still want to meet by the arcade?"
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    first = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))
    second = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert first.status == "sent"
    assert second.status == "skipped"
    assert second.reason == "duplicate_suppressed"
    assert len(provider.chat_requests) == 1
    assert len(repositories.list_character_text_messages(save_id=save_id)) == 1
    triggers = repositories.list_character_text_proactive_triggers(save_id)
    assert len(triggers) == 1
    assert triggers[0].trigger_key == first.trigger_key


def test_proactive_text_respects_cooldown_for_active_thread_candidate(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    first_source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan texts from the station stairs.",
    )
    second_source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The station crowd gets louder.",
    )
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Rain starts tapping on the platform roof.",
    )
    fourth_source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The last train warning bell sounds.",
    )
    active_thread = repositories.add_active_thread(
        save_id=save_id,
        title="Reply to Rowan's station message",
        description="Rowan sent a time-sensitive text from the station.",
        status="waiting on phone reply",
        priority=2,
        related_entities=[npc.id],
        source_message_id=first_source.id,
        last_updated_message_id=first_source.id,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        value=2,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Can you still make the train?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    first = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(first_source.id,),
        )
    )
    repositories.update_active_thread(
        replace(
            active_thread,
            source_message_id=second_source.id,
            last_updated_message_id=second_source.id,
        )
    )
    second = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(second_source.id,),
        )
    )
    repositories.update_active_thread(
        replace(
            active_thread,
            source_message_id=fourth_source.id,
            last_updated_message_id=fourth_source.id,
        )
    )
    provider.response_body = "The train is about to leave."
    third = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(fourth_source.id,),
        )
    )

    assert first.status == "sent"
    assert second.status == "skipped"
    assert second.reason == "cooldown_active"
    assert third.status == "sent"
    assert len(provider.chat_requests) == 2
    assert [
        message.body
        for message in repositories.list_character_text_messages(save_id=save_id)
    ] == [
        "Can you still make the train?",
        "The train is about to leave.",
    ]


def test_proactive_text_respects_cooldown_for_character_intent_candidate(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    first_source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira walks home after the arcade closes.",
    )
    repositories.update_character(
        replace(
            npc,
            current_intent="Ask whether Mira got home safely.",
        )
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        value=2,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Did you get home okay?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    first = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira waits at a quiet crosswalk.",
    )
    updated_npc = repositories.get_character(npc.id)
    assert updated_npc is not None
    repositories.update_character(
        replace(
            updated_npc,
            current_intent="Ask if Mira got home safely.",
        )
    )
    second = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="A bus rolls past the school parking lot.",
    )
    fourth_source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira reaches the corner near home.",
    )
    updated_npc = repositories.get_character(npc.id)
    assert updated_npc is not None
    repositories.update_character(
        replace(
            updated_npc,
            current_intent="Ask whether Mira made it to her street.",
        )
    )
    provider.response_body = "I found your notebook."
    third = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert first.status == "sent"
    assert second.status == "skipped"
    assert second.reason == "cooldown_active"
    assert third.status == "sent"
    assert len(provider.chat_requests) == 2
    triggers = repositories.list_character_text_proactive_triggers(save_id)
    assert [trigger.source_message_id for trigger in triggers] == [
        first_source.id,
        fourth_source.id,
    ]
    assert [
        message.body
        for message in repositories.list_character_text_messages(save_id=save_id)
    ] == [
        "Did you get home okay?",
        "I found your notebook.",
    ]


def test_proactive_text_suppresses_present_dating_route_candidate(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Mira and Rowan are talking beside the arcade counter.",
        objective="Decide whether to play another round.",
        present_character_ids=[npc.id],
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Want to talk after practice?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "skipped"
    assert result.reason == "present_character_suppressed"
    assert result.candidate_count == 1
    assert provider.chat_requests == []
    assert repositories.list_character_text_messages(save_id=save_id) == []


def test_proactive_text_allows_present_text_native_active_thread_candidate(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan checks his phone while standing beside Mira.",
    )
    repositories.add_active_thread(
        save_id=save_id,
        title="Answer Rowan's urgent text",
        description="Rowan's phone message needs a clear response.",
        status="waiting on phone reply",
        priority=3,
        related_entities=[npc.id],
        source_message_id=source.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Mira and Rowan wait beside the locked arcade cabinet.",
        objective="Decide whether to split up.",
        present_character_ids=[npc.id],
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Check your phone for the code.")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(source.id,),
        )
    )

    assert result.status == "sent"
    assert result.trigger_key.startswith("active_thread:")
    assert result.message is not None
    assert result.message.body == "Check your phone for the code."


def test_proactive_text_prefers_active_thread_over_current_intent(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    repositories.update_character(
        replace(
            npc,
            current_intent="Ask whether Mira finished the cabinet repair.",
        )
    )
    source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan keeps checking the half-repaired cabinet.",
    )
    repositories.add_active_thread(
        save_id=save_id,
        title="Replace the arcade token",
        description="Rowan needs to follow up about the missing cabinet token.",
        status="active",
        priority=1,
        related_entities=[npc.id],
        source_message_id=source.id,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Did you find the token?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(source.id,),
        )
    )

    assert result.status == "sent"
    assert result.trigger_key.startswith("active_thread:")
    prompt = "\n".join(provider.chat_requests[-1].phone_context)
    assert "Proactive text trigger: active thread" in prompt
    assert "Proactive text trigger: character current intent" not in prompt


def test_proactive_text_includes_source_chronicle_context(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan sees Mira leave the arcade without her notebook.",
    )
    repositories.add_active_thread(
        save_id=save_id,
        title="Text Mira about Rowan's notebook",
        description="Rowan should follow up before Mira gets too far away.",
        status="waiting on phone reply",
        priority=3,
        related_entities=[npc.id],
        source_message_id=source.id,
    )
    _configure_text_reply_model(repositories)
    provider = RecordingTextProvider(response_body="You forgot your notebook.")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(source.id,),
        )
    )

    assert result.status == "sent"
    request = provider.chat_requests[-1]
    phone_context = "\n".join(request.phone_context)
    chronicle_context = "\n".join(request.retrieved_recent_messages)
    assert "Proactive text trigger: active thread" in phone_context
    assert source.id in chronicle_context
    assert "Mira leave the arcade without her notebook" in chronicle_context


def test_proactive_text_skips_private_active_thread(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira privately weighs whether to tell Rowan about the arcade key.",
    )
    repositories.add_active_thread(
        save_id=save_id,
        title="Decide whether to text Rowan about the arcade key",
        description=(
            "Player-only phone planning context about a secret Rowan has not learned."
        ),
        status="active",
        priority=7,
        visibility="private",
        related_entities=[npc.id],
        source_message_id=source.id,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Did you find the key?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_proactive_text_after_turn(save_id=save_id)
    )

    assert result.status == "skipped"
    assert result.reason == "no_candidate"
    assert provider.chat_requests == []
    assert repositories.list_character_text_messages(save_id=save_id) == []


def test_proactive_text_ignores_active_thread_that_mentions_another_character(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    rowan = _npc_character(repositories, save_id)
    maya = repositories.add_character(
        save_id=save_id,
        name="Maya",
        role="classmate",
        met=True,
    )
    source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Maya sends a careful follow-up after class.",
    )
    repositories.add_active_thread(
        save_id=save_id,
        title="Maya wants to follow up after class",
        description="Maya's message needs a thoughtful response.",
        status="active",
        priority=6,
        visibility="public",
        related_entities=[rowan.id, maya.id],
        source_message_id=source.id,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Still thinking about class?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, rowan.id)

    result = asyncio.run(
        service.send_proactive_text_after_turn(save_id=save_id)
    )

    assert result.status == "skipped"
    assert result.reason == "no_candidate"
    assert provider.chat_requests == []
    assert repositories.list_character_text_messages(save_id=save_id) == []


def test_proactive_text_suppresses_unavailable_character(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    repositories.update_character(
        replace(
            npc,
            status="asleep with phone off",
        )
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Want to meet after class?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "skipped"
    assert result.reason == "character_unavailable"
    assert provider.chat_requests == []
    assert repositories.list_character_text_messages(save_id=save_id) == []


def test_proactive_text_sends_ambient_random_text_when_probability_allows(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira finishes the quiet walk home from school.",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        value=100,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        value=0,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Did you get home okay?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(source.id,),
        )
    )

    assert result.status == "sent"
    assert result.candidate_count == 1
    assert result.message is not None
    assert result.message.body == "Did you get home okay?"
    triggers = repositories.list_character_text_proactive_triggers(save_id)
    assert len(triggers) == 1
    assert triggers[0].trigger_type == "ambient_random"
    assert triggers[0].source_message_id == source.id
    assert triggers[0].source_type == "message"
    assert triggers[0].source_id == source.id
    assert "thought of the player" in triggers[0].reason


def test_proactive_text_chance_zero_disables_ambient_random(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira checks her phone after the rain lets up.",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        value=0,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(source.id,),
        )
    )

    assert result.status == "skipped"
    assert result.reason == "proactive_texts_disabled"
    assert provider.chat_requests == []
    assert repositories.list_character_text_proactive_triggers(save_id) == []


def test_proactive_text_respects_ambient_random_cooldown(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    first_source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira leaves the arcade just before sunset.",
    )
    second_source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira waits for the next train.",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        value=100,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        value=2,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Thinking about the arcade.")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    first = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(first_source.id,),
        )
    )
    second = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(second_source.id,),
        )
    )

    assert first.status == "sent"
    assert second.status == "skipped"
    assert second.reason == "no_candidate"
    assert len(provider.chat_requests) == 1
    assert len(repositories.list_character_text_proactive_triggers(save_id)) == 1


def test_proactive_text_cooldown_blocks_ambient_after_intent_text(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    first_source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira leaves the arcade after the lights go out.",
    )
    repositories.update_character(
        replace(
            npc,
            current_intent="Ask whether Mira got home safely.",
        )
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        value=100,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        value=2,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider(response_body="Did you get home okay?")
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    first = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))
    second_source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mira waits for the next train.",
    )
    updated_npc = repositories.get_character(npc.id)
    assert updated_npc is not None
    repositories.update_character(replace(updated_npc, current_intent=""))
    provider.response_body = "Thinking about the arcade."
    second = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(second_source.id,),
        )
    )

    assert first.status == "sent"
    assert second.status == "skipped"
    assert second.reason == "no_candidate"
    assert len(provider.chat_requests) == 1
    triggers = repositories.list_character_text_proactive_triggers(save_id)
    assert len(triggers) == 1
    assert triggers[0].trigger_type == "character_intent"
    assert triggers[0].source_message_id == first_source.id


def test_proactive_text_does_not_pick_present_character_for_ambient_random(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_without_route(repositories, scenario_type="dating_sim")
    npc = _npc_character(repositories, save_id)
    source = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan and Mira stand beside the lockers in comfortable silence.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Rowan and Mira stand beside the lockers.",
        objective="Decide where to go next.",
        present_character_ids=[npc.id],
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        value=100,
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingTextProvider()
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(
        service.send_proactive_text_after_turn(
            save_id=save_id,
            source_message_ids=(source.id,),
        )
    )

    assert result.status == "skipped"
    assert result.reason == "no_candidate"
    assert provider.chat_requests == []
    assert repositories.list_character_text_proactive_triggers(save_id) == []


def test_proactive_text_applies_structured_world_update_from_npc_message(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save_with_characters(repositories, scenario_type="dating_sim")
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    _save_fake_structured_model(repositories, model_id="fake-context")
    provider = StructuredTextWorldProvider(
        response_body="I saved you a cabinet token.",
        structured_data={
            "memories": [
                {
                    "body": "Rowan saved Mira an arcade cabinet token.",
                    "tags": ["arcade", "rowan"],
                    "importance": 0.72,
                    "source_text_message_id": "message",
                    "character_id": npc.id,
                    "knowledge_state": "knows",
                    "acquisition_method": "told",
                    "evidence_quote": "I saved you a cabinet token.",
                }
            ],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
        },
    )
    service = CharacterTextService(
        repositories=repositories,
        providers={"fake": provider},
    )
    _grant_character_has_player_number(repositories, save_id, npc.id)

    result = asyncio.run(service.send_proactive_text_after_turn(save_id=save_id))

    assert result.status == "sent"
    assert result.message is not None
    assert result.world_update is not None
    assert result.world_update.status == "applied"
    assert result.world_update.memory_count == 1
    source_ref = f"character_text_message:{result.message.id}"
    memories = repositories.list_memories(save_id)
    assert [memory.body for memory in memories] == [
        "Rowan saved Mira an arcade cabinet token.",
    ]
    assert memories[0].source_message_ids == [source_ref]
    provenance = repositories.list_character_text_provenance(
        save_id=save_id,
        text_message_id=result.message.id,
    )
    assert {row.target_type for row in provenance} >= {
        "memory",
        "character_knowledge_edge",
    }


def _npc_character(
    repositories: PersistenceRepositories,
    save_id: str,
) -> CharacterRecord:
    return next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )


def _configure_text_reply_model(repositories: PersistenceRepositories) -> None:
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )


def _configure_text_attachment_decision_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="response_planning",
        provider="fake",
        model_id="fake-plan",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-plan",
        display_name="Fake Planner",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )


def _configure_group_response_assessment_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="response_planning",
        provider="fake",
        model_id="fake-plan",
    )
    _save_fake_structured_model(repositories, model_id="fake-plan")


def _save_fake_structured_model(
    repositories: PersistenceRepositories,
    *,
    model_id: str,
) -> None:
    repositories.save_provider_model(
        provider="fake",
        model_id=model_id,
        display_name=model_id.replace("-", " ").title(),
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )


def _attachment_decision(
    *,
    kind: str,
    visual_prompt: str = "",
    reason: str = "grounded in the current text",
    wearing: str = "",
    current_action: str = "",
    facial_expression: str = "",
) -> dict[str, object]:
    return {
        "attachment_kind": kind,
        "visual_prompt": visual_prompt,
        "wearing": wearing,
        "current_action": current_action,
        "facial_expression": facial_expression,
        "reason": reason,
    }


def _create_save_with_characters(
    repositories: PersistenceRepositories,
    *,
    scenario_type: str,
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY,
) -> str:
    scenario = repositories.create_scenario(
        type=scenario_type,
        title="After School",
        premise="A small town romance.",
        player_role="Mira",
        interaction_mode=interaction_mode,
        content={
            "player_character_name": "Mira",
            "characters": ["Rowan"],
            "opening_message": "The last bell rings.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="After School")
    player = repositories.add_character(
        save_id=save.id,
        name="Mira",
        role="player",
        is_player_character=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Rowan",
        role="classmate",
        personality="guarded but curious",
        voice="brief, dry, sincere",
        met=True,
    )
    repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="introduced",
        completed_interactions=1,
        interest_level="curious",
        next_reasonable_step="Ask about Rowan's arcade plans.",
    )
    return save.id


def _create_group_text_save(repositories: PersistenceRepositories) -> str:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Arcade Crew",
        premise="A small town arcade route.",
        player_role="Mira",
        content={
            "player_character_name": "Mira",
            "characters": ["Rowan", "Maya", "Toma", "Sera"],
            "opening_message": "The arcade lights hum after closing.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Arcade Crew")
    player = repositories.add_character(
        save_id=save.id,
        name="Mira",
        role="player",
        is_player_character=True,
    )
    for name in ("Rowan", "Maya", "Toma", "Sera"):
        npc = repositories.add_character(
            save_id=save.id,
            name=name,
            role="arcade regular",
            personality=f"{name} has distinct group chat habits.",
            voice=f"{name} texts in a concise style.",
            met=True,
        )
        repositories.upsert_dating_route_state(
            save_id=save.id,
            player_character_id=player.id,
            npc_character_id=npc.id,
            stage="introduced",
            completed_interactions=1,
            interest_level="curious",
            next_reasonable_step=f"Coordinate with {name} at the arcade.",
        )
    return save.id


def _create_save_without_route(
    repositories: PersistenceRepositories,
    *,
    scenario_type: str,
) -> str:
    scenario = repositories.create_scenario(
        type=scenario_type,
        title="After School",
        premise="A small town romance.",
        player_role="Mira",
        content={
            "player_character_name": "Mira",
            "characters": ["Rowan"],
            "opening_message": "The last bell rings.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="After School")
    repositories.add_character(
        save_id=save.id,
        name="Mira",
        role="player",
        is_player_character=True,
    )
    repositories.add_character(
        save_id=save.id,
        name="Rowan",
        role="classmate",
        personality="guarded but curious",
        voice="brief, dry, sincere",
        met=True,
    )
    return save.id


def _group_response_decision(
    character: CharacterRecord,
    *,
    should_respond: bool,
    priority: int,
) -> dict[str, object]:
    return {
        "character_id": character.id,
        "should_respond": should_respond,
        "response_intent": (
            f"{character.name} wants to answer the group request."
            if should_respond
            else ""
        ),
        "reason": "The player directly asked the group for help.",
        "confidence": 0.8,
        "priority": priority,
    }


def _player_character(
    repositories: PersistenceRepositories,
    save_id: str,
) -> CharacterRecord:
    return next(
        character
        for character in repositories.list_characters(save_id)
        if character.is_player_character
    )


def _grant_player_has_number(
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str,
) -> None:
    player = next(
        character
        for character in repositories.list_characters(save_id)
        if character.is_player_character
    )
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=character_id,
        player_has_character_number=True,
    )


def _grant_character_has_player_number(
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str,
    *,
    player_has_character_number: bool = False,
) -> None:
    player = next(
        character
        for character in repositories.list_characters(save_id)
        if character.is_player_character
    )
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=character_id,
        player_has_character_number=player_has_character_number,
        character_has_player_number=True,
    )
