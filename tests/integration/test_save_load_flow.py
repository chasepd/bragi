from __future__ import annotations

import asyncio
from pathlib import Path

from bragi.persistence.repositories import BragiRepository
from bragi.providers.contracts import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
)
from bragi.services.chat_service import ChatService
from bragi.services.context_search_service import ContextSearchResult
from bragi.services.save_service import SaveService


class ContinuingFakeProvider:
    provider_name = "fake"

    def __init__(self) -> None:
        self.chat_requests: list[ChatRequest] = []

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
                capabilities=frozenset({ProviderCapability.CHAT}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        return ChatResponse(
            body=f"fake narrator: {request.messages[-1].body}",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 8},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("save/load chat flow must not generate images")


class NoopContextSearchService:
    async def search(
        self,
        *,
        save_id: str,
        player_message_id: str,
    ) -> ContextSearchResult:
        return ContextSearchResult()


def test_save_load_and_continue_chat_flow_with_temp_sqlite(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    first_repository = BragiRepository(database_path)
    try:
        scenario = first_repository.create_scenario(
            type="full_roleplay",
            title="Frostglass Hall",
            premise="A sealed hall is thawing after a century.",
            player_role="Relic hunter",
            content={"starting_scene": "The mirror nave begins to thaw."},
        )
        save_service = SaveService(repositories=first_repository)
        save = save_service.create_save(
            scenario_id=scenario.id,
            title="First Thaw",
        )
        first_repository.set_model_preference(
            task="chat",
            provider="fake",
            model_id="fake-chat",
        )
        first_provider = ContinuingFakeProvider()
        chat_service = ChatService(
            repositories=first_repository,
            providers={"fake": first_provider},
            context_search_service=NoopContextSearchService(),
        )

        first_turn = asyncio.run(
            chat_service.submit_player_turn(
                save_id=save.id,
                body="I touch the mirror floor.",
                speaker_name="Mara",
            )
        )
    finally:
        first_repository.connection.close()

    second_repository = BragiRepository(database_path)
    try:
        loaded = SaveService(repositories=second_repository).load_save(save.id)

        assert loaded.save.id == save.id
        assert loaded.scenario.id == scenario.id
        assert loaded.messages == [
            first_turn.player_message,
            first_turn.narrator_message,
        ]

        second_repository.set_model_preference(
            task="chat",
            provider="fake",
            model_id="fake-chat",
        )
        second_provider = ContinuingFakeProvider()
        continued_turn = asyncio.run(
            ChatService(
                repositories=second_repository,
                providers={"fake": second_provider},
                context_search_service=NoopContextSearchService(),
            ).submit_player_turn(
                save_id=save.id,
                body="I step through the reflection.",
                speaker_name="Mara",
            )
        )

        reloaded = SaveService(repositories=second_repository).load_save(save.id)
        continued_request = second_provider.chat_requests[0]

        assert continued_turn.narrator_message.body == (
            "fake narrator: I step through the reflection."
        )
        assert [
            (message.role, message.body, message.provider, message.model)
            for message in reloaded.messages
        ] == [
            ("player", "I touch the mirror floor.", None, None),
            (
                "narrator",
                "fake narrator: I touch the mirror floor.",
                "fake",
                "fake-chat",
            ),
            ("player", "I step through the reflection.", None, None),
            (
                "narrator",
                "fake narrator: I step through the reflection.",
                "fake",
                "fake-chat",
            ),
        ]
        assert second_provider.chat_requests == [continued_request]
        assert continued_request.messages == (
            ChatMessage(
                role="player",
                body="I touch the mirror floor.",
                speaker_name="Mara",
            ),
            ChatMessage(
                role="narrator",
                body="fake narrator: I touch the mirror floor.",
                speaker_name="Narrator",
            ),
            ChatMessage(
                role="player",
                body="I step through the reflection.",
                speaker_name="Mara",
            ),
        )
        assert continued_request.retrieved_state == ()
        assert continued_request.retrieved_memories == ()
        assert continued_request.summary is None
    finally:
        second_repository.connection.close()
