from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import CharacterRecord, DatingRouteStateRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ProviderRetryProgressCallback,
)
from bragi.services.character_text_revision_service import CharacterTextRevisionService
from bragi.services.character_text_service import (
    CharacterTextService,
    _update_text_route,
)
from bragi.services.character_text_world_update_service import (
    CharacterTextWorldUpdateService,
    character_text_source_ref,
)


class RecordingTextProvider:
    def __init__(self, response_body: str = "I can meet after class.") -> None:
        self.response_body = response_body
        self.chat_requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        return ChatResponse(
            body=self.response_body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 11},
        )


class FailingTextProvider(RecordingTextProvider):
    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise RuntimeError("text provider failed")


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_edit_player_text_without_resubmit_updates_body_without_provider(
    repositories: PersistenceRepositories,
) -> None:
    save_id, thread_id, player_message_id, _reply_id = _seed_text_thread(
        repositories,
    )
    stale_memory = repositories.add_memory(
        save_id=save_id,
        body="Mira asked Mika to meet after class.",
        tags=["mika"],
        source_message_ids=[character_text_source_ref(player_message_id)],
    )
    provider = RecordingTextProvider()
    service = CharacterTextRevisionService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = service.edit_text_without_resubmit(
        save_id=save_id,
        text_message_id=player_message_id,
        body="Can we talk after class?",
    )

    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=thread_id,
    )
    revisions = repositories.list_character_text_message_revisions(
        save_id=save_id,
        text_message_id=player_message_id,
    )
    assert result.message.body == "Can we talk after class?"
    assert [(message.sender, message.body) for message in messages] == [
        ("player", "Can we talk after class?"),
        ("character", "Sure, meet me by the lockers."),
    ]
    assert provider.chat_requests == []
    assert revisions[0].previous_body == "Can we tak after class?"
    assert revisions[0].new_body == "Can we talk after class?"
    assert revisions[0].reconciliation_status == "succeeded"
    assert revisions[0].reconciliation_error is None
    assert stale_memory.id not in {
        memory.id for memory in repositories.list_memories(save_id)
    }


def test_correct_character_text_updates_body_and_revision_trail(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _thread_id, _player_message_id, reply_id = _seed_text_thread(
        repositories,
    )
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    source_ref = character_text_source_ref(reply_id)
    stale_memory = repositories.add_memory(
        save_id=save_id,
        body="Mika agreed to meet by the lockers.",
        tags=["mika"],
        source_message_ids=[source_ref],
    )
    edge = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=npc.id,
        target_type="memory",
        target_id=stale_memory.id,
        source_message_ids=[source_ref],
    )
    service = CharacterTextRevisionService(repositories=repositories, providers={})

    result = service.correct_character_text(
        save_id=save_id,
        text_message_id=reply_id,
        body="Sure, meet me by the science wing.",
    )

    revisions = repositories.list_character_text_message_revisions(
        save_id=save_id,
        text_message_id=reply_id,
    )
    metadata = repositories.character_text_message_revision_metadata(save_id)
    assert result.message.body == "Sure, meet me by the science wing."
    assert len(revisions) == 1
    assert revisions[0].previous_body == "Sure, meet me by the lockers."
    assert revisions[0].new_body == "Sure, meet me by the science wing."
    assert revisions[0].reconciliation_status == "succeeded"
    assert revisions[0].reconciliation_error is None
    assert metadata[reply_id].revision_count == 1
    assert stale_memory.id not in {
        memory.id for memory in repositories.list_memories(save_id)
    }
    assert edge.id not in {
        edge.id for edge in repositories.list_character_knowledge_edges(save_id)
    }


def test_edit_text_without_resubmit_recomputes_thread_memory(
    repositories: PersistenceRepositories,
) -> None:
    save_id, thread_id, player_message_id, _reply_id = _seed_text_thread(
        repositories,
    )
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    for index in range(31):
        repositories.append_character_text_message(
            save_id=save_id,
            thread_id=thread_id,
            character_id=npc.id,
            sender="player" if index % 2 == 0 else "character",
            body=f"Thread filler {index}",
        )
    repositories.update_character_text_thread_memory(
        save_id=save_id,
        thread_id=thread_id,
        body="Phone thread memory: Can we tak after class?",
        message_count=33,
    )
    service = CharacterTextRevisionService(repositories=repositories, providers={})

    service.edit_text_without_resubmit(
        save_id=save_id,
        text_message_id=player_message_id,
        body="Can we talk after club?",
    )

    thread = repositories.get_character_text_thread(
        save_id=save_id,
        thread_id=thread_id,
    )
    assert thread is not None
    assert "Can we talk after club?" in thread.memory_body
    assert "Can we tak after class?" not in thread.memory_body


def test_edit_player_text_and_resubmit_replays_downstream_reply(
    repositories: PersistenceRepositories,
) -> None:
    save_id, thread_id, player_message_id, old_reply_id = _seed_text_thread(
        repositories,
    )
    provider = RecordingTextProvider(response_body="Then meet me at the arcade.")
    service = CharacterTextRevisionService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.edit_text_and_resubmit(
            save_id=save_id,
            text_message_id=player_message_id,
            body="Can we meet at the arcade?",
        )
    )

    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=thread_id,
    )
    revisions = repositories.list_character_text_message_revisions(
        save_id=save_id,
        text_message_id=player_message_id,
    )
    assert old_reply_id not in {message.id for message in messages}
    assert [(message.sender, message.body) for message in messages] == [
        ("player", "Can we meet at the arcade?"),
        ("character", "Then meet me at the arcade."),
    ]
    assert result.reply.body == "Then meet me at the arcade."
    assert revisions[0].reconciliation_status == "succeeded"
    assert len(provider.chat_requests) == 1
    assert [
        (message.role, message.body)
        for message in provider.chat_requests[0].messages
    ] == [("player", "Can we meet at the arcade?")]


def test_edit_player_text_and_resubmit_forwards_actor_to_reply_completion(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    save_id, _thread_id, player_message_id, _old_reply_id = _seed_text_thread(
        repositories,
    )
    actor = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    provider = RecordingTextProvider(response_body="Then meet me at the arcade.")
    seen_user_ids: list[str | None] = []
    original = CharacterTextService.complete_queued_text_send

    async def recording_completion(
        service: CharacterTextService,
        *,
        save_id: str,
        player_message_id: str,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> object:
        seen_user_ids.append(current_user_id)
        return await original(
            service,
            save_id=save_id,
            player_message_id=player_message_id,
            retry_progress_callback=retry_progress_callback,
            current_user_id=current_user_id,
        )

    monkeypatch.setattr(
        CharacterTextService,
        "complete_queued_text_send",
        recording_completion,
    )
    service = CharacterTextRevisionService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(
        service.edit_text_and_resubmit(
            save_id=save_id,
            text_message_id=player_message_id,
            body="Can we meet at the arcade?",
            current_user_id=actor.id,
        )
    )

    assert seen_user_ids == [actor.id]


def test_edit_player_text_and_resubmit_archives_old_text_derived_memory(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _thread_id, player_message_id, old_reply_id = _seed_text_thread(
        repositories,
    )
    stale_memory = repositories.add_memory(
        save_id=save_id,
        body="Mika agreed to meet by the lockers.",
        tags=[],
        source_message_ids=[character_text_source_ref(old_reply_id)],
    )
    provider = RecordingTextProvider(response_body="Then meet me at the arcade.")
    service = CharacterTextRevisionService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(
        service.edit_text_and_resubmit(
            save_id=save_id,
            text_message_id=player_message_id,
            body="Can we meet at the arcade?",
        )
    )

    assert stale_memory.id not in {
        memory.id for memory in repositories.list_memories(save_id)
    }


def test_edit_player_text_and_resubmit_restores_prior_state_when_provider_fails(
    repositories: PersistenceRepositories,
) -> None:
    save_id, thread_id, player_message_id, old_reply_id = _seed_text_thread(
        repositories,
        create_contact_state=False,
    )
    source_ref = character_text_source_ref(old_reply_id)
    player, npc = _player_and_npc(repositories, save_id)
    stale_memory = repositories.add_memory(
        save_id=save_id,
        body="Mika agreed to meet by the lockers.",
        tags=["mika"],
        source_message_ids=[source_ref],
    )
    edge = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=npc.id,
        target_type="memory",
        target_id=stale_memory.id,
        source_message_ids=[source_ref],
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="upsert",
        entity_type="memory",
        entity_id=stale_memory.id,
        field_path="body",
        proposed_value="Mika agreed to meet by the lockers.",
        source_message_ids=[source_ref],
    )
    observation = repositories.add_context_observation(
        save_id=save_id,
        observation_type="memory",
        claim="Mika is meeting by the lockers.",
        source_message_ids=[source_ref],
    )
    route = repositories.upsert_dating_route_state(
        save_id=save_id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="introduced",
        completed_interactions=1,
        interest_level="curious",
        trust_level="guarded",
        next_reasonable_step="Ask about the lockers.",
    )
    CharacterTextWorldUpdateService(
        repositories=repositories,
        providers={},
    ).apply_structured_update(
        save_id=save_id,
        text_messages=tuple(
            repositories.list_character_text_messages(
                save_id=save_id,
                thread_id=thread_id,
            )
        ),
        data={
            "active_threads": [
                {
                    "title": "Locker meeting",
                    "description": "Mika agreed to meet by the lockers.",
                    "priority": 3,
                    "visibility": "private",
                    "source_text_message_id": "reply",
                }
            ],
            "character_updates": [
                {
                    "character_id": npc.id,
                    "status": "waiting by the lockers",
                    "source_text_message_id": "reply",
                }
            ],
            "dating_route_updates": [
                {
                    "npc_character_id": npc.id,
                    "trust_level": "warming",
                    "next_reasonable_step": "Meet by the lockers.",
                    "source_text_message_id": "reply",
                }
            ],
            "contact_permissions": [
                {
                    "character_id": npc.id,
                    "player_has_character_number": True,
                    "character_has_player_number": True,
                    "source_text_message_id": "reply",
                }
            ],
            "memories": [],
        },
    )
    service = CharacterTextRevisionService(
        repositories=repositories,
        providers={"fake": FailingTextProvider()},
    )

    with pytest.raises(RuntimeError, match="text provider failed"):
        asyncio.run(
            service.edit_text_and_resubmit(
                save_id=save_id,
                text_message_id=player_message_id,
                body="Can we meet at the arcade?",
            )
        )

    messages = repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=thread_id,
    )
    assert [
        (message.id, message.sender, message.body, message.delivery_status)
        for message in messages
    ] == [
        (player_message_id, "player", "Can we tak after class?", "sent"),
        (old_reply_id, "character", "Sure, meet me by the lockers.", "sent"),
    ]
    revisions = repositories.list_character_text_message_revisions(
        save_id=save_id,
        text_message_id=player_message_id,
    )
    assert revisions[0].reconciliation_status == "failed"
    assert revisions[0].reconciliation_error == "Text resubmit failed"
    assert stale_memory.id in {
        memory.id for memory in repositories.list_memories(save_id)
    }
    assert edge.id in {
        item.id for item in repositories.list_character_knowledge_edges(save_id)
    }
    assert suggestion.id in {
        item.id
        for item in repositories.list_context_update_suggestions(
            save_id,
            status="pending",
        )
    }
    assert observation.id in {
        item.id for item in repositories.list_context_observations(save_id)
    }
    assert [thread.title for thread in repositories.list_active_threads(save_id)] == [
        "Locker meeting"
    ]
    updated_npc = repositories.get_character(npc.id)
    assert updated_npc is not None
    assert updated_npc.status == "waiting by the lockers"
    restored_route = _route_for(repositories, save_id, npc.id)
    assert restored_route.id == route.id
    assert restored_route.trust_level == "warming"
    assert restored_route.next_reasonable_step == "Meet by the lockers."
    assert repositories.character_text_outbound_allowed(
        save_id=save_id,
        character_id=npc.id,
    )


def test_delete_text_messages_from_here_archives_suffix_and_text_context(
    repositories: PersistenceRepositories,
) -> None:
    save_id, thread_id, player_message_id, reply_id = _seed_text_thread(
        repositories,
    )
    source_ref = character_text_source_ref(reply_id)
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    stale_memory = repositories.add_memory(
        save_id=save_id,
        body="Mika agreed to meet by the lockers.",
        tags=["mika"],
        source_message_ids=[source_ref],
    )
    edge = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=npc.id,
        target_type="memory",
        target_id=stale_memory.id,
        source_message_ids=[source_ref],
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="upsert",
        entity_type="memory",
        entity_id=stale_memory.id,
        field_path="body",
        proposed_value="Mika agreed to meet by the lockers.",
        source_message_ids=[source_ref],
    )
    observation = repositories.add_context_observation(
        save_id=save_id,
        observation_type="memory",
        claim="Mika is meeting by the lockers.",
        source_message_ids=[source_ref],
    )
    repositories.update_character_text_thread_memory(
        save_id=save_id,
        thread_id=thread_id,
        body="Phone thread memory: Mika agreed to meet by the lockers.",
        message_count=2,
    )
    service = CharacterTextRevisionService(repositories=repositories, providers={})

    result = service.delete_text_messages_from_here(
        save_id=save_id,
        text_message_id=player_message_id,
    )

    assert [message.id for message in result.deleted_messages] == [
        player_message_id,
        reply_id,
    ]
    assert result.thread.messages == ()
    assert (
        repositories.list_character_text_messages(
            save_id=save_id,
            thread_id=thread_id,
        )
        == []
    )
    assert stale_memory.id not in {
        memory.id for memory in repositories.list_memories(save_id)
    }
    assert edge.id not in {
        edge.id for edge in repositories.list_character_knowledge_edges(save_id)
    }
    assert suggestion.id in {
        item.id
        for item in repositories.list_context_update_suggestions(
            save_id,
            status="expired",
        )
    }
    assert observation.id not in {
        item.id for item in repositories.list_context_observations(save_id)
    }
    thread = repositories.get_character_text_thread(
        save_id=save_id,
        thread_id=thread_id,
    )
    assert thread is not None
    assert thread.memory_body == ""
    assert thread.memory_message_count == 0


def test_delete_text_messages_from_here_reconciles_text_world_state(
    repositories: PersistenceRepositories,
) -> None:
    save_id, thread_id, player_message_id, reply_id = _seed_text_thread(
        repositories,
        create_contact_state=False,
    )
    player, npc = _player_and_npc(repositories, save_id)
    npc = repositories.update_character(replace(npc, status="curious"))
    route = repositories.upsert_dating_route_state(
        save_id=save_id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="introduced",
        completed_interactions=1,
        interest_level="curious",
        trust_level="guarded",
        next_reasonable_step="Ask about the arcade.",
    )
    CharacterTextWorldUpdateService(
        repositories=repositories,
        providers={},
    ).apply_structured_update(
        save_id=save_id,
        text_messages=tuple(
            repositories.list_character_text_messages(
                save_id=save_id,
                thread_id=thread_id,
            )
        ),
        data={
            "memories": [],
            "active_threads": [
                {
                    "title": "Arcade meeting",
                    "description": "Mika agreed to meet by the arcade.",
                    "priority": 3,
                    "visibility": "private",
                    "source_text_message_id": "reply",
                }
            ],
            "character_updates": [
                {
                    "character_id": npc.id,
                    "status": "waiting by the arcade",
                    "source_text_message_id": "reply",
                }
            ],
            "dating_route_updates": [
                {
                    "npc_character_id": npc.id,
                    "trust_level": "warming",
                    "next_reasonable_step": "Meet by the arcade.",
                    "source_text_message_id": "reply",
                }
            ],
            "contact_permissions": [
                {
                    "character_id": npc.id,
                    "player_has_character_number": True,
                    "character_has_player_number": True,
                    "source_text_message_id": "reply",
                }
            ],
        },
    )
    route_before_increment = _route_for(repositories, save_id, npc.id)
    incremented_route = repositories.upsert_dating_route_state(
        save_id=save_id,
        player_character_id=route_before_increment.player_character_id,
        npc_character_id=route_before_increment.npc_character_id,
        stage=route_before_increment.stage,
        completed_interactions=route_before_increment.completed_interactions + 1,
        interest_level=route_before_increment.interest_level,
        trust_level=route_before_increment.trust_level,
        next_reasonable_step=route_before_increment.next_reasonable_step,
    )
    _record_text_route_interaction(
        repositories,
        save_id=save_id,
        thread_id=thread_id,
        reply_id=reply_id,
        before=route_before_increment,
        after=incremented_route,
    )
    repositories.add_character_text_proactive_trigger(
        save_id=save_id,
        character_id=npc.id,
        trigger_key=f"dating_route:{route.id}:arcade",
        trigger_type="dating_route",
        thread_id=thread_id,
        text_message_id=reply_id,
        source_type="dating_route_state",
        source_id=route.id,
        reason="Consumed by the stale reply.",
    )
    service = CharacterTextRevisionService(repositories=repositories, providers={})

    service.delete_text_messages_from_here(
        save_id=save_id,
        text_message_id=player_message_id,
    )

    assert repositories.list_active_threads(save_id) == []
    updated_npc = repositories.get_character(npc.id)
    assert updated_npc is not None
    assert updated_npc.status == "curious"
    restored_route = _route_for(repositories, save_id, npc.id)
    assert restored_route.trust_level == "guarded"
    assert restored_route.next_reasonable_step == "Ask about the arcade."
    assert restored_route.completed_interactions == 1
    assert repositories.list_character_contact_states(save_id) == []
    assert repositories.list_character_text_proactive_triggers(save_id) == []


def test_delete_text_messages_from_here_restores_existing_contact_state(
    repositories: PersistenceRepositories,
) -> None:
    save_id, thread_id, player_message_id, _reply_id = _seed_text_thread(
        repositories,
        create_contact_state=False,
    )
    player, npc = _player_and_npc(repositories, save_id)
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=False,
        source_text_message_id=None,
    )
    CharacterTextWorldUpdateService(
        repositories=repositories,
        providers={},
    ).apply_structured_update(
        save_id=save_id,
        text_messages=tuple(
            repositories.list_character_text_messages(
                save_id=save_id,
                thread_id=thread_id,
            )
        ),
        data={
            "memories": [],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
            "contact_permissions": [
                {
                    "character_id": npc.id,
                    "player_has_character_number": False,
                    "character_has_player_number": True,
                    "source_text_message_id": "reply",
                }
            ],
        },
    )
    service = CharacterTextRevisionService(repositories=repositories, providers={})

    service.delete_text_messages_from_here(
        save_id=save_id,
        text_message_id=player_message_id,
    )

    restored = repositories.get_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
    )
    assert restored is not None
    assert restored.player_has_character_number is True
    assert restored.character_has_player_number is False
    assert restored.source_text_message_id is None


def test_edit_player_text_and_resubmit_replaces_text_route_interaction(
    repositories: PersistenceRepositories,
) -> None:
    save_id, thread_id, player_message_id, old_reply_id = _seed_text_thread(
        repositories,
    )
    player, npc = _player_and_npc(repositories, save_id)
    route = repositories.upsert_dating_route_state(
        save_id=save_id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="introduced",
        completed_interactions=0,
        interest_level="curious",
        next_reasonable_step="Ask about Mika's science project.",
    )
    incremented_route = repositories.upsert_dating_route_state(
        save_id=save_id,
        player_character_id=route.player_character_id,
        npc_character_id=route.npc_character_id,
        stage=route.stage,
        completed_interactions=1,
        interest_level=route.interest_level,
        next_reasonable_step=route.next_reasonable_step,
    )
    _record_text_route_interaction(
        repositories,
        save_id=save_id,
        thread_id=thread_id,
        reply_id=old_reply_id,
        before=route,
        after=incremented_route,
    )
    provider = RecordingTextProvider(response_body="Then meet me at the arcade.")
    service = CharacterTextRevisionService(
        repositories=repositories,
        providers={"fake": provider},
    )

    asyncio.run(
        service.edit_text_and_resubmit(
            save_id=save_id,
            text_message_id=player_message_id,
            body="Can we meet at the arcade?",
        )
    )

    restored_route = _route_for(repositories, save_id, npc.id)
    assert restored_route.completed_interactions == 1


def test_edit_player_text_and_resubmit_repoints_consumed_proactive_trigger(
    repositories: PersistenceRepositories,
) -> None:
    save_id, thread_id, player_message_id, old_reply_id = _seed_text_thread(
        repositories,
    )
    player, npc = _player_and_npc(repositories, save_id)
    repositories.upsert_dating_route_state(
        save_id=save_id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="introduced",
        completed_interactions=0,
        interest_level="curious",
        next_reasonable_step="Ask about Mika's science project.",
    )
    player_message = repositories.get_character_text_message(
        save_id=save_id,
        message_id=player_message_id,
    )
    old_reply = repositories.get_character_text_message(
        save_id=save_id,
        message_id=old_reply_id,
    )
    assert player_message is not None
    assert old_reply is not None
    _update_text_route(
        repositories=repositories,
        save_id=save_id,
        character=npc,
        player_message=player_message,
        reply=old_reply,
    )
    stale_triggers = repositories.list_character_text_proactive_triggers(save_id)
    assert len(stale_triggers) == 1
    assert stale_triggers[0].text_message_id == old_reply_id
    provider = RecordingTextProvider(response_body="Then meet me at the arcade.")
    service = CharacterTextRevisionService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(
        service.edit_text_and_resubmit(
            save_id=save_id,
            text_message_id=player_message_id,
            body="Can we meet at the arcade?",
        )
    )

    triggers = repositories.list_character_text_proactive_triggers(save_id)
    assert len(triggers) == 1
    assert triggers[0].trigger_key == stale_triggers[0].trigger_key
    assert triggers[0].text_message_id == result.reply.id
    assert triggers[0].thread_id == thread_id


def _seed_text_thread(
    repositories: PersistenceRepositories,
    *,
    create_contact_state: bool = True,
) -> tuple[str, str, str, str]:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="After School",
        premise="A small town romance.",
        player_role="Mira",
        content={"player_character_name": "Mira"},
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
        name="Mika",
        role="classmate",
        met=True,
    )
    if create_contact_state:
        repositories.upsert_character_contact_state(
            save_id=save.id,
            player_character_id=player.id,
            character_id=npc.id,
            player_has_character_number=True,
            character_has_player_number=True,
        )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=npc.id,
        title=npc.name,
    )
    player_message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Can we tak after class?",
    )
    reply = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Sure, meet me by the lockers.",
        provider="fake",
        model="fake-chat",
    )
    return save.id, thread.id, player_message.id, reply.id


def _player_and_npc(
    repositories: PersistenceRepositories,
    save_id: str,
) -> tuple[CharacterRecord, CharacterRecord]:
    characters = repositories.list_characters(save_id)
    player = next(
        character for character in characters if character.is_player_character
    )
    npc = next(
        character for character in characters if not character.is_player_character
    )
    return player, npc


def _route_for(
    repositories: PersistenceRepositories,
    save_id: str,
    npc_character_id: str,
) -> DatingRouteStateRecord:
    return next(
        route
        for route in repositories.list_dating_route_states(save_id)
        if route.npc_character_id == npc_character_id
    )


def _record_text_route_interaction(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    thread_id: str,
    reply_id: str,
    before: DatingRouteStateRecord,
    after: DatingRouteStateRecord,
) -> None:
    source_ref = character_text_source_ref(reply_id)
    repositories.add_context_update_audit(
        save_id=save_id,
        operation="text_exchange",
        entity_type="dating_route_state",
        entity_id=after.id,
        field_path="completed_interactions",
        before={
            "id": before.id,
            "completed_interactions": before.completed_interactions,
            "trust_level": before.trust_level,
            "next_reasonable_step": before.next_reasonable_step,
        },
        after={
            "id": after.id,
            "completed_interactions": after.completed_interactions,
            "trust_level": after.trust_level,
            "next_reasonable_step": after.next_reasonable_step,
        },
        reason="Character text exchange counted for dating route.",
        confidence=1.0,
        source_message_ids=[source_ref],
    )
    repositories.add_character_text_provenance(
        save_id=save_id,
        thread_id=thread_id,
        text_message_id=reply_id,
        target_type="dating_route_state",
        target_id=after.id,
        operation="text_exchange",
        field_path="completed_interactions",
    )
