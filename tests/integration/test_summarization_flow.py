from __future__ import annotations

import asyncio
from pathlib import Path

from bragi.persistence.repositories import BragiRepository
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.services.chat_service import ChatService
from bragi.services.context_search_service import ContextSearchService
from bragi.services.summary_service import SummaryService


class SummarizationContextNarratorProvider:
    provider_name = "fake"
    state_note = "The player is still acting on the ash bridge."
    memory_note = "Mara's distrust of windless bells colors the echo's meaning."
    echo_note = "The recent echo is what the player now follows."

    def __init__(self) -> None:
        self.events: list[str] = []
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
                model_id="fake-summary",
                display_name="Fake Summary",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=1024,
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-context",
                display_name="Fake Context",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
                context_window=8192,
            ),
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
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        if request.model_id == "fake-summary":
            self.events.append("summarization")
            prompt = "\n".join(message.body for message in request.messages)
            assert "I step onto the ash bridge." in prompt
            assert "A bell rings under the span." in prompt
            assert "scene.location" not in prompt
            assert "Mara distrusts bells" not in prompt
            return ChatResponse(
                body="Mara crossed the ash bridge and heard a windless bell.",
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 44},
            )

        if request.model_id == "fake-context":
            raise AssertionError("context search must use structured output")

        self.events.append("narrator_chat")
        assert request.summary is not None
        assert request.summary.startswith("[summary:")
        assert request.summary.endswith(
            "Mara crossed the ash bridge and heard a windless bell. "
            "(relevance: latest rolling summary.)"
        )
        assert "Legacy scene state: scene.location: name: Ash Bridge" in (
            "\n".join(request.current_scene_recap)
        )
        assert request.retrieved_state == ()
        assert request.context_breakdown["suppressed_duplicate_retrieval_keys"] == [
            "world_state:state-ash-bridge"
        ]
        assert request.retrieved_memories == (
            "[memory:memory-bell] Mara distrusts bells that ring without wind.",
        )
        assert [message.body for message in request.messages] == [
            "I step onto the ash bridge.",
            "A bell rings under the span.",
            "I ask who rang the bell.",
            "The echo answers from below.",
            "I lean over the stones and follow the bell's echo.",
        ]
        assert [message.body for message in request.messages].count(
            "The echo answers from below."
        ) == 1
        return ChatResponse(
            body="fake narrator: the bell rope trembles below the bridge stones.",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 31},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("summarization flow must not generate images")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return StructuredOutputResponse(
                data={
                    "action": "allow",
                    "category": "none",
                    "reason": "Integration fixture content is within the ceiling.",
                    "minimum_rating": "g",
                },
                provider=request.provider,
                model_id=request.model_id,
            )

        if request.schema_name == "context_search_selection":
            self.events.append("context_search")
            prompt = "\n".join(message.body for message in request.messages)
            assert "I lean over the stones and follow the bell's echo." in prompt
            assert "scene.location" in prompt
            assert "Mara distrusts bells" in prompt
            assert "Mara crossed the ash bridge" not in prompt
            assert "The echo answers from below." in prompt
            return StructuredOutputResponse(
                data={
                    "selections": [
                        {
                            "source_type": "world_state",
                            "source_id": "state-ash-bridge",
                            "relevance_note": self.state_note,
                        },
                        {
                            "source_type": "memory",
                            "source_id": "memory-bell",
                            "relevance_note": self.memory_note,
                        },
                        {
                            "source_type": "message",
                            "source_id": "message-echo",
                            "relevance_note": self.echo_note,
                        },
                    ],
                },
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 23},
            )

        if request.schema_name == "narrator_message_plan":
            self.events.append("narrator_message_plan")
            return StructuredOutputResponse(
                data={
                    "intent": "Continue the bridge scene.",
                    "thesis": "The bell rope trembles below.",
                    "must_say": [],
                    "avoid": [],
                    "tone": "eerie",
                    "uncertainties": [],
                    "evidence_source_ids": [],
                },
                provider=request.provider,
                model_id=request.model_id,
            )

        if request.schema_name == "narrator_message_verification":
            self.events.append("narrator_message_verification")
            return StructuredOutputResponse(
                data={
                    "passed": True,
                    "issues": [],
                    "retry_feedback": "",
                    "confidence": 1.0,
                },
                provider=request.provider,
                model_id=request.model_id,
            )

        if request.schema_name == "context_observation_extraction":
            self.events.append("context_observation_extraction")
            return StructuredOutputResponse(
                data={"observations": []},
                provider=request.provider,
                model_id=request.model_id,
            )

        raise AssertionError(f"unexpected structured schema: {request.schema_name}")


def test_summarization_persists_summary_before_future_narrator_context(
    tmp_path: Path,
) -> None:
    repository = BragiRepository(tmp_path / "bragi.sqlite3")
    try:
        scenario = repository.create_scenario(
            type="full_roleplay",
            title="Bridge of Cinders",
            premise="A bridge remembers every oath broken on it.",
            player_role="Oathkeeper",
            content={"starting_scene": "Cinders drift over the bridge stones."},
        )
        save = repository.create_save(scenario_id=scenario.id, title="Crossing")
        first_player = repository.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I step onto the ash bridge.",
            token_estimate=70,
        )
        first_narrator = repository.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="A bell rings under the span.",
            provider="fake",
            model="fake-chat",
            token_estimate=70,
        )
        repository.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I ask who rang the bell.",
            token_estimate=45,
        )
        repository.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="The echo answers from below.",
            provider="fake",
            model="fake-chat",
            token_estimate=45,
            message_id="message-echo",
        )
        repository.upsert_world_state(
            save_id=save.id,
            key="scene.location",
            value={"name": "Ash Bridge"},
            category="scene",
            source_message_id=first_player.id,
            state_id="state-ash-bridge",
        )
        repository.add_memory(
            save_id=save.id,
            body="Mara distrusts bells that ring without wind.",
            tags=["bells", "suspicion"],
            source_message_id=first_narrator.id,
            memory_id="memory-bell",
        )
        repository.set_model_preference(
            task="summarization",
            provider="fake",
            model_id="fake-summary",
        )
        repository.set_model_preference(
            task="context_search",
            provider="fake",
            model_id="fake-context",
        )
        repository.set_model_preference(
            task="chat",
            provider="fake",
            model_id="fake-chat",
        )
        repository.save_provider_model(
            provider="fake",
            model_id="fake-chat",
            display_name="Fake Chat",
            capabilities=["chat"],
            context_window=8192,
        )
        repository.save_provider_model(
            provider="fake",
            model_id="fake-context",
            display_name="Fake Context",
            capabilities=["structured_output"],
            context_window=8192,
        )
        provider = SummarizationContextNarratorProvider()
        summary_service = SummaryService(
            repositories=repository,
            providers={"fake": provider},
            threshold=0.01,
        )
        context_search = ContextSearchService(
            repositories=repository,
            providers={"fake": provider},
        )
        chat_service = ChatService(
            repositories=repository,
            providers={"fake": provider},
            context_search_service=context_search,
            summary_service=summary_service,
        )

        result = asyncio.run(
            chat_service.submit_player_turn(
                save_id=save.id,
                body="I lean over the stones and follow the bell's echo.",
                speaker_name="Mara",
            )
        )

        assert provider.events == [
            "summarization",
            "context_search",
            "narrator_message_plan",
            "narrator_chat",
            "narrator_message_verification",
            "context_observation_extraction",
        ]
        assert result.narrator_message.body == (
            "fake narrator: the bell rope trembles below the bridge stones."
        )
        summaries = repository.list_summaries(save.id)
        assert len(summaries) == 1
        assert summaries[0].covers_message_start_id == first_player.id
        assert summaries[0].covers_message_end_id == first_narrator.id
        assert summaries[0].body == (
            "Mara crossed the ash bridge and heard a windless bell."
        )
        assert summaries[0].provider == "fake"
        assert summaries[0].model == "fake-summary"
        assert [message.id for message in repository.list_messages(save.id)][:4] == [
            first_player.id,
            first_narrator.id,
            repository.list_messages(save.id)[2].id,
            "message-echo",
        ]
        assert repository.list_world_state(save.id)[0].key == "scene.location"
        assert repository.list_memories(save.id)[0].id == "memory-bell"
    finally:
        repository.connection.close()
