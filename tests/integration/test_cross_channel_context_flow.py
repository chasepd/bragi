from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from bragi.persistence.models import CharacterRecord, SaveRecord, ScenarioRecord
from bragi.persistence.repositories import BragiRepository
from bragi.providers.contracts import (
    CHAT_TURN_DIRECTIVE_PURPOSE_CHARACTER_TEXT,
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
from bragi.services.character_text_service import CharacterTextService
from bragi.services.character_text_world_update_service import character_text_source_ref
from bragi.services.chat_service import ChatService
from bragi.services.context_search_service import ContextSearchService

VISIBLE_CHRONICLE_BEAT = "The blue flare blinks twice over the courtyard."
HIDDEN_CHRONICLE_BEAT = "Cass hides the obsidian-door passphrase in her notebook."
TEXT_DERIVED_PRIVATE_MEMORY = (
    "Rowan privately knows the circuit-lantern repair code from a phone text."
)


@dataclass(frozen=True)
class CrossChannelSave:
    scenario: ScenarioRecord
    save: SaveRecord
    player: CharacterRecord
    rowan: CharacterRecord
    cass: CharacterRecord


class CrossChannelProvider:
    provider_name = "fake"

    def __init__(self, *, rowan_id: str) -> None:
        self.rowan_id = rowan_id
        self.chat_requests: list[ChatRequest] = []
        self.structured_output_requests: list[StructuredOutputRequest] = []
        self.context_search_modes: list[str] = []
        self.context_search_mode = ""
        self.selected_memory_id = ""
        self.selected_text_thread_id = ""

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
                model_id="fake-context",
                display_name="Fake Context",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
                context_window=8192,
            ),
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        if request.turn_directive_purpose == CHAT_TURN_DIRECTIVE_PURPOSE_CHARACTER_TEXT:
            prompt_text = "\n".join(
                [
                    *request.retrieved_recent_messages,
                    *request.current_scene_recap,
                    *request.phone_context,
                ]
            )
            assert VISIBLE_CHRONICLE_BEAT in prompt_text
            assert HIDDEN_CHRONICLE_BEAT not in prompt_text
            body = "I saw the blue flare. I will keep the repair code ready."
        elif self.context_search_mode == "cass":
            prompt_text = _request_context_text(request)
            assert TEXT_DERIVED_PRIVATE_MEMORY not in prompt_text
            assert "repair code ready" not in prompt_text
            body = "fake narrator: Cass answers without the private text fact."
        elif self.context_search_mode == "rowan":
            prompt_text = _request_context_text(request)
            assert TEXT_DERIVED_PRIVATE_MEMORY in prompt_text
            assert any(
                "repair code ready" in line
                for line in request.retrieved_character_text_context
            )
            body = "fake narrator: Rowan can use the private text fact."
        else:
            body = "fake narrator: idle."
        return ChatResponse(
            body=body,
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

        self.structured_output_requests.append(request)
        if request.schema_name == "character_text_world_update":
            return StructuredOutputResponse(
                data={
                    "memories": [
                        {
                            "body": TEXT_DERIVED_PRIVATE_MEMORY,
                            "tags": ["rowan", "phone"],
                            "importance": 0.9,
                            "source_text_message_id": "reply",
                            "knowledge_state": "knows",
                            "acquisition_method": "told",
                            "evidence_quote": "repair code",
                            "reason": "Rowan sent the repair code privately.",
                        }
                    ],
                    "active_threads": [],
                    "character_updates": [],
                    "dating_route_updates": [],
                    "contact_permissions": [],
                },
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 21},
            )
        if request.schema_name == "context_search_selection":
            self.context_search_modes.append(self.context_search_mode)
            prompt = "\n".join(message.body for message in request.messages)
            if self.context_search_mode == "cass":
                assert TEXT_DERIVED_PRIVATE_MEMORY not in prompt
                assert "repair code ready" not in prompt
                selections: list[dict[str, str]] = []
            elif self.context_search_mode == "rowan":
                assert TEXT_DERIVED_PRIVATE_MEMORY in prompt
                assert "repair code ready" in prompt
                selections = [
                    {
                        "source_type": "character_text_thread",
                        "source_id": self.selected_text_thread_id,
                        "relevance_note": "Rowan's text thread has the repair code.",
                    }
                ]
            else:
                selections = []
            return StructuredOutputResponse(
                data={"selections": selections},
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 13},
            )
        if request.schema_name == "narrator_message_plan":
            return StructuredOutputResponse(
                data={
                    "intent": "Continue the scene.",
                    "thesis": "Keep character knowledge scoped.",
                    "must_say": [],
                    "avoid": [],
                    "tone": "grounded",
                    "uncertainties": [],
                    "evidence_source_ids": [],
                },
                provider=request.provider,
                model_id=request.model_id,
            )
        if request.schema_name == "narrator_message_verification":
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
            return StructuredOutputResponse(
                data={"observations": []},
                provider=request.provider,
                model_id=request.model_id,
            )
        raise AssertionError(f"unexpected structured schema: {request.schema_name}")

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("cross-channel context tests must not generate images")


def test_direct_text_creates_scoped_context_for_relevant_later_narration(
    tmp_path: Path,
) -> None:
    repository = BragiRepository(tmp_path / "bragi.sqlite3")
    try:
        fixture = _create_cross_channel_save(repository)
        provider = CrossChannelProvider(rowan_id=fixture.rowan.id)
        _configure_models(repository)

        hidden = repository.append_message(
            save_id=fixture.save.id,
            role="narrator",
            speaker_name="Narrator",
            body=HIDDEN_CHRONICLE_BEAT,
        )
        repository.add_message_visibility(
            save_id=fixture.save.id,
            message_id=hidden.id,
            character_id=fixture.rowan.id,
            visibility="not_visible",
        )
        repository.append_message(
            save_id=fixture.save.id,
            role="narrator",
            speaker_name="Narrator",
            body=VISIBLE_CHRONICLE_BEAT,
        )

        text_result = asyncio.run(
            CharacterTextService(
                repositories=repository,
                providers={"fake": provider},
            ).send_text(
                save_id=fixture.save.id,
                character_id=fixture.rowan.id,
                body="Did you see what happened in the courtyard?",
            )
        )

        assert text_result.world_update is not None
        assert text_result.world_update.status == "applied"
        memories = repository.list_memories(fixture.save.id)
        assert [memory.body for memory in memories] == [TEXT_DERIVED_PRIVATE_MEMORY]
        assert memories[0].source_message_ids == [
            character_text_source_ref(text_result.reply.id)
        ]
        edges = repository.list_character_knowledge_edges(fixture.save.id)
        assert {
            (edge.character_id, edge.target_type, edge.target_id)
            for edge in edges
        } == {
            (fixture.player.id, "memory", memories[0].id),
            (fixture.rowan.id, "memory", memories[0].id),
        }

        provider.selected_memory_id = memories[0].id
        provider.selected_text_thread_id = text_result.thread.id
        assert provider.selected_text_thread_id
        context_search = ContextSearchService(
            repositories=repository,
            providers={"fake": provider},
        )
        narrator_service = ChatService(
            repositories=repository,
            providers={"fake": provider},
            context_search_service=context_search,
        )

        repository.upsert_scene_snapshot(
            save_id=fixture.save.id,
            situation="Cass waits by the arcade counter while Rowan is away.",
            present_character_ids=[fixture.cass.id],
        )
        provider.context_search_mode = "cass"
        cass_turn = asyncio.run(
            narrator_service.submit_player_turn(
                save_id=fixture.save.id,
                body="I ask Cass whether the arcade has any update.",
                speaker_name=fixture.player.name,
                run_post_turn_jobs=False,
            )
        )
        assert cass_turn.narrator_message.body == (
            "fake narrator: Cass answers without the private text fact."
        )

        repository.upsert_scene_snapshot(
            save_id=fixture.save.id,
            situation="Rowan waits by the circuit-lantern with Mira.",
            present_character_ids=[fixture.rowan.id],
        )
        provider.context_search_mode = "rowan"
        rowan_turn = asyncio.run(
            narrator_service.submit_player_turn(
                save_id=fixture.save.id,
                body="I ask Rowan about the circuit-lantern repair.",
                speaker_name=fixture.player.name,
                run_post_turn_jobs=False,
            )
        )

        assert rowan_turn.narrator_message.body == (
            "fake narrator: Rowan can use the private text fact."
        )
        assert provider.context_search_modes == ["cass", "rowan"]
    finally:
        repository.connection.close()


def _create_cross_channel_save(repository: BragiRepository) -> CrossChannelSave:
    scenario = repository.create_scenario(
        type="dating_sim",
        title="Arcade Signals",
        premise="Students coordinate repairs around an old arcade.",
        player_role="Mira",
        content={
            "player_character_name": "Mira",
            "characters": ["Rowan", "Cass"],
            "opening_message": "The arcade lights hum after closing.",
        },
    )
    save = repository.create_save(scenario_id=scenario.id, title="Arcade Save")
    player = repository.add_character(
        save_id=save.id,
        name="Mira",
        role="player",
        is_player_character=True,
    )
    rowan = repository.add_character(
        save_id=save.id,
        name="Rowan",
        role="classmate",
        personality="careful and direct",
        voice="brief and sincere",
        met=True,
    )
    cass = repository.add_character(
        save_id=save.id,
        name="Cass",
        role="club president",
        personality="observant and guarded",
        voice="dry and precise",
        met=True,
    )
    for character in (rowan, cass):
        repository.upsert_character_contact_state(
            save_id=save.id,
            player_character_id=player.id,
            character_id=character.id,
            player_has_character_number=True,
            character_has_player_number=True,
        )
    return CrossChannelSave(
        scenario=scenario,
        save=save,
        player=player,
        rowan=rowan,
        cass=cass,
    )


def _request_context_text(request: ChatRequest) -> str:
    return "\n".join(
        [
            *request.phone_context,
            *request.current_scene_recap,
            *request.character_voice_profiles,
            *request.character_action_plans,
            *request.open_obligations,
            *request.pending_context_suggestions,
            *request.retrieved_scenario_sections,
            *request.retrieved_state,
            *request.retrieved_state_changes,
            *request.retrieved_recent_messages,
            *request.retrieved_media_assets,
            *request.retrieved_character_text_context,
            *request.retrieved_memories,
            *request.retrieved_observations,
            request.summary or "",
        ]
    )


def _configure_models(repository: BragiRepository) -> None:
    repository.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=[ProviderCapability.CHAT.value],
    )
    repository.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    repository.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repository.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    repository.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    repository.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
