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


class ContextThenNarratorProvider:
    provider_name = "fake"
    state_note = "The beacon objective is directly relevant to the igniter."
    memory_note = "Mara's promise to Elian raises the stakes."
    message_note = "The lens message is the immediate setup."

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
                model_id="fake-context",
                display_name="Fake Context",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
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
            ),
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        if request.model_id == "fake-context":
            raise AssertionError("context search must use structured output")

        self.events.append("narrator_chat")
        assert (
            "Legacy scene state: scene.objective: goal: Relight the beacon"
            in "\n".join(request.current_scene_recap)
        )
        assert request.retrieved_state == ()
        assert request.context_breakdown["suppressed_duplicate_retrieval_keys"] == [
            "world_state:state-beacon",
            "message:message-lens",
        ]
        assert request.retrieved_memories == (
            "[memory:memory-elian] Mara promised Elian the beacon would burn.",
        )
        assert request.summary == (
            "[summary:summary-opening] The keep is cut off by the ash storm. "
            "(relevance: latest rolling summary.)"
        )
        assert [message.body for message in request.messages] == [
            "Ash claws at the beacon lens.",
            "I turn the beacon lens and strike the igniter.",
        ]
        return ChatResponse(
            body="fake narrator: the beacon catches and throws gold through the ash.",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 34},
        )

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
            self.events.append("context_search_selection")
            prompt = "\n".join(message.body for message in request.messages)
            assert "I turn the beacon lens and strike the igniter." in prompt
            assert "Ash claws at the beacon lens." in prompt
            assert "The keep is cut off by the ash storm." not in prompt
            return StructuredOutputResponse(
                data={
                    "selections": [
                        {
                            "source_type": "world_state",
                            "source_id": "state-beacon",
                            "relevance_note": self.state_note,
                        },
                        {
                            "source_type": "memory",
                            "source_id": "memory-elian",
                            "relevance_note": self.memory_note,
                        },
                        {
                            "source_type": "message",
                            "source_id": "message-lens",
                            "relevance_note": self.message_note,
                        },
                    ],
                },
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 21},
            )

        if request.schema_name == "narrator_message_plan":
            self.events.append("narrator_message_plan")
            return StructuredOutputResponse(
                data={
                    "intent": "Continue the beacon scene.",
                    "thesis": "The beacon catches.",
                    "must_say": [],
                    "avoid": [],
                    "tone": "urgent",
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

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("context search flow must not generate images")


def test_context_search_runs_before_narrator_and_persists_selected_context(
    tmp_path: Path,
) -> None:
    repository = BragiRepository(tmp_path / "bragi.sqlite3")
    try:
        scenario = repository.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A border keep is cut off by ash storms.",
            player_role="Signal warden",
            content={"starting_scene": "The beacon gutters in the tower."},
        )
        save = repository.create_save(scenario_id=scenario.id, title="Night Watch")
        source_message = repository.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="Ash claws at the beacon lens.",
            provider="fake",
            model="fake-chat",
            message_id="message-lens",
        )
        repository.upsert_world_state(
            save_id=save.id,
            key="scene.objective",
            value={"goal": "Relight the beacon"},
            category="scene",
            source_message_id=source_message.id,
            state_id="state-beacon",
        )
        repository.add_memory(
            save_id=save.id,
            body="Mara promised Elian the beacon would burn.",
            tags=["promise", "beacon"],
            source_message_id=source_message.id,
            memory_id="memory-elian",
        )
        repository.add_summary(
            save_id=save.id,
            covers_message_start_id=source_message.id,
            covers_message_end_id=source_message.id,
            body="The keep is cut off by the ash storm.",
            provider="fake",
            model="fake-summary",
            summary_id="summary-opening",
        )
        repository.set_model_preference(
            task="context_search",
            provider="fake",
            model_id="fake-context",
        )
        repository.save_provider_model(
            provider="fake",
            model_id="fake-context",
            display_name="Fake Context",
            capabilities=["structured_output"],
        )
        repository.set_model_preference(
            task="chat",
            provider="fake",
            model_id="fake-chat",
        )
        provider = ContextThenNarratorProvider()
        context_search = ContextSearchService(
            repositories=repository,
            providers={"fake": provider},
        )
        service = ChatService(
            repositories=repository,
            providers={"fake": provider},
            context_search_service=context_search,
        )

        result = asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I turn the beacon lens and strike the igniter.",
                speaker_name="Mara",
            )
        )

        assert provider.events == [
            "context_search_selection",
            "narrator_message_plan",
            "narrator_chat",
            "narrator_message_verification",
            "context_observation_extraction",
        ]
        assert result.narrator_message.body == (
            "fake narrator: the beacon catches and throws gold through the ash."
        )
        jobs = list(
            repository.connection.execute(
                """
                SELECT status, result_json
                FROM jobs
                WHERE save_id = ? AND type = 'context_search'
                ORDER BY created_at, rowid
                """,
                (save.id,),
            )
        )
        assert jobs[-1]["status"] == "succeeded"
        assert "scene.objective: goal: Relight the beacon" in jobs[-1]["result_json"]
        assert provider.state_note in jobs[-1]["result_json"]
    finally:
        repository.connection.close()
