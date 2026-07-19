from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import (
    CharacterRecord,
    CharacterTextMessageRecord,
    ContextUpdateAuditRecord,
    DatingRouteStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.services.character_text_world_update_service import (
    CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE,
    CharacterTextWorldUpdateService,
    character_text_source_ref,
)


class FakeStructuredTextWorldProvider:
    provider_name = "fake"

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.structured_requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        return StructuredOutputResponse(
            data=self.data,
            provider=request.provider,
            model_id=request.model_id,
        )


@dataclass(frozen=True)
class TextWorldFixture:
    save_id: str
    player_id: str
    thread_npc_id: str
    unrelated_npc_id: str
    player_message: CharacterTextMessageRecord
    reply: CharacterTextMessageRecord


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_update_after_text_messages_rejects_cross_character_targets(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    unrelated_route = _route_for(
        repositories,
        save_id=fixture.save_id,
        npc_character_id=fixture.unrelated_npc_id,
    )
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [
                {
                    "body": "Cass learned about Rowan's repair notes.",
                    "tags": ["cross-target"],
                    "importance": 0.6,
                    "source_text_message_id": "reply",
                    "character_id": fixture.unrelated_npc_id,
                    "knowledge_state": "knows",
                    "acquisition_method": "told",
                    "reason": "Wrong participant.",
                }
            ],
            "active_threads": [
                {
                    "title": "Repair note exchange",
                    "description": "Rowan and Mira need to follow up.",
                    "priority": 2,
                    "visibility": "private",
                    "related_entities": [
                        fixture.unrelated_npc_id,
                        f"character:{fixture.unrelated_npc_id}",
                        f"character:{fixture.thread_npc_id}",
                        "repair notes",
                    ],
                    "source_text_message_id": "reply",
                }
            ],
            "character_updates": [
                {
                    "character_id": fixture.unrelated_npc_id,
                    "status": "incorrectly changed",
                    "source_text_message_id": "reply",
                    "reason": "Wrong target.",
                    "confidence": 0.8,
                }
            ],
            "dating_route_updates": [
                {
                    "npc_character_id": fixture.unrelated_npc_id,
                    "trust_level": "incorrectly warmed",
                    "next_reasonable_step": "Follow up in Cass's thread.",
                    "source_text_message_id": "reply",
                    "reason": "Wrong route.",
                    "confidence": 0.7,
                }
            ],
        }
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(fixture.player_message, fixture.reply),
        )
    )

    assert result.status == "applied"
    assert result.memory_count == 1
    assert result.active_thread_count == 1
    assert result.character_count == 0
    assert result.dating_route_count == 0
    assert result.knowledge_edge_count == 0
    assert _character_for(repositories, fixture.unrelated_npc_id).status == "distant"
    assert (
        _route_for(
            repositories,
            save_id=fixture.save_id,
            npc_character_id=fixture.unrelated_npc_id,
        ).trust_level
        == unrelated_route.trust_level
    )
    assert repositories.list_character_knowledge_edges(fixture.save_id) == []
    active_thread = repositories.list_active_threads(fixture.save_id)[0]
    assert active_thread.related_entities == [
        f"character:{fixture.thread_npc_id}",
        "repair notes",
    ]
    rejected = _rejected_audits(repositories, fixture.save_id)
    assert {
        (audit.entity_type, audit.field_path)
        for audit in rejected
    } >= {
        ("character_knowledge_edge", "character_id"),
        ("active_thread", "related_entities"),
        ("character", "*"),
        ("dating_route_state", "*"),
    }
    assert any(
        _audit_after(audit).get("attempted_character_id")
        == fixture.unrelated_npc_id
        for audit in rejected
    )
    assert any(
        _audit_after(audit).get("attempted_npc_character_id")
        == fixture.unrelated_npc_id
        for audit in rejected
    )


def test_update_after_text_messages_allows_thread_npc_and_player_updates(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [],
            "active_threads": [],
            "character_updates": [
                {
                    "character_id": fixture.thread_npc_id,
                    "status": "waiting by the arcade",
                    "source_text_message_id": "reply",
                    "reason": "Rowan committed to the meeting.",
                    "confidence": 0.9,
                },
                {
                    "character_id": fixture.player_id,
                    "goals": "Meet Rowan by the arcade.",
                    "source_text_message_id": "player",
                    "reason": "Mira asked to meet.",
                    "confidence": 0.8,
                },
            ],
            "dating_route_updates": [
                {
                    "npc_character_id": fixture.thread_npc_id,
                    "trust_level": "warming",
                    "next_reasonable_step": "Meet by the arcade.",
                    "source_text_message_id": "reply",
                    "reason": "Rowan agreed.",
                    "confidence": 0.9,
                }
            ],
        }
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(fixture.player_message, fixture.reply),
        )
    )

    assert result.status == "applied"
    assert result.character_count == 2
    assert result.dating_route_count == 1
    assert _character_for(repositories, fixture.thread_npc_id).status == (
        "waiting by the arcade"
    )
    assert _character_for(repositories, fixture.player_id).goals == (
        "Meet Rowan by the arcade."
    )
    assert _character_for(repositories, fixture.unrelated_npc_id).status == "distant"
    route = _route_for(
        repositories,
        save_id=fixture.save_id,
        npc_character_id=fixture.thread_npc_id,
    )
    assert route.trust_level == "warming"
    assert route.next_reasonable_step == "Meet by the arcade."
    assert _rejected_audits(repositories, fixture.save_id) == []


def test_text_memory_without_character_id_defaults_to_text_audience(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [
                {
                    "body": "The repair-note exchange stays private to Mira and Rowan.",
                    "tags": ["repair", "phone"],
                    "importance": 0.82,
                    "source_text_message_id": "reply",
                    "knowledge_state": "knows",
                    "acquisition_method": "texted",
                    "evidence_quote": "repair notes",
                    "reason": "The fact came from a private text thread.",
                }
            ],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
        }
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(fixture.player_message, fixture.reply),
        )
    )

    assert result.status == "applied"
    assert result.memory_count == 1
    assert result.knowledge_edge_count == 2
    [memory] = repositories.list_memories(fixture.save_id)
    edges = repositories.list_character_knowledge_edges(fixture.save_id)
    assert {
        (edge.character_id, edge.target_type, edge.target_id)
        for edge in edges
    } == {
        (fixture.player_id, "memory", memory.id),
        (fixture.thread_npc_id, "memory", memory.id),
    }
    assert all(
        edge.source_message_ids == [character_text_source_ref(fixture.reply.id)]
        for edge in edges
    )
    assert _rejected_audits(repositories, fixture.save_id) == []


def test_update_after_text_messages_applies_contact_permission_update(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
            "contact_permissions": [
                {
                    "character_id": fixture.thread_npc_id,
                    "player_has_character_number": True,
                    "character_has_player_number": False,
                    "source_text_message_id": "reply",
                    "reason": "Rowan texted his direct number.",
                    "confidence": 0.94,
                }
            ],
        }
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(fixture.player_message, fixture.reply),
        )
    )

    assert result.status == "applied"
    assert result.contact_permission_count == 1
    state = repositories.get_character_contact_state(
        save_id=fixture.save_id,
        player_character_id=fixture.player_id,
        character_id=fixture.thread_npc_id,
    )
    assert state is not None
    assert state.player_has_character_number is True
    assert state.character_has_player_number is False
    assert state.source_text_message_id == fixture.reply.id
    audit = repositories.list_context_update_audit(fixture.save_id)[0]
    assert audit.entity_type == "character_contact_state"
    assert audit.source_message_ids == [character_text_source_ref(fixture.reply.id)]
    provenance = repositories.list_character_text_provenance(
        save_id=fixture.save_id,
        text_message_id=fixture.reply.id,
    )
    assert [(row.target_type, row.target_id) for row in provenance] == [
        ("character_contact_state", state.id),
    ]


def test_update_after_text_messages_rejects_cross_character_contact_permission(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
            "contact_permissions": [
                {
                    "character_id": fixture.unrelated_npc_id,
                    "player_has_character_number": True,
                    "source_text_message_id": "reply",
                    "reason": "Wrong participant.",
                    "confidence": 0.94,
                }
            ],
        }
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(fixture.player_message, fixture.reply),
        )
    )

    assert result.status == "applied"
    assert result.contact_permission_count == 0
    assert repositories.list_character_contact_states(fixture.save_id) == []
    rejected = _rejected_audits(repositories, fixture.save_id)
    assert [(audit.entity_type, audit.field_path) for audit in rejected] == [
        ("character_contact_state", "character_id"),
    ]
    assert _audit_after(rejected[0]).get("attempted_character_id") == (
        fixture.unrelated_npc_id
    )


def test_structured_request_limits_targets_to_text_participants(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
        }
    )
    service = _service(repositories, provider)

    asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(fixture.player_message, fixture.reply),
        )
    )

    request = provider.structured_requests[0]
    prompt_body = request.messages[-1].body
    assert fixture.thread_npc_id in prompt_body
    assert fixture.player_id in prompt_body
    assert fixture.unrelated_npc_id not in prompt_body
    assert f"npc={fixture.thread_npc_id}" in prompt_body
    assert f"npc={fixture.unrelated_npc_id}" not in prompt_body
    assert _schema_enum(request, "memories", "character_id") == {
        fixture.player_id,
        fixture.thread_npc_id,
    }
    assert _schema_enum(request, "character_updates", "character_id") == {
        fixture.player_id,
        fixture.thread_npc_id,
    }
    assert _schema_enum(request, "dating_route_updates", "npc_character_id") == {
        fixture.thread_npc_id,
    }
    assert _schema_enum(request, "contact_permissions", "character_id") == {
        fixture.thread_npc_id,
    }


def test_structured_request_limits_active_threads_to_text_participants(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    repositories.add_active_thread(
        save_id=fixture.save_id,
        title="Rowan repair notes follow-up",
        description="Rowan needs to bring the repair notes to the arcade.",
        status="active",
        priority=5,
        visibility="public",
        related_entities=[fixture.thread_npc_id],
    )
    repositories.add_active_thread(
        save_id=fixture.save_id,
        title="Cass private festival letter",
        description="Cass is deciding whether to send a private festival letter.",
        status="active",
        priority=6,
        visibility="private",
        related_entities=[fixture.unrelated_npc_id],
    )
    repositories.add_active_thread(
        save_id=fixture.save_id,
        title="Mira solo arcade plan",
        description="Mira is considering an unrelated solo plan.",
        status="active",
        priority=7,
        visibility="private",
        related_entities=[fixture.player_id],
    )
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
        }
    )
    service = _service(repositories, provider)

    asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(fixture.player_message, fixture.reply),
        )
    )

    prompt_body = provider.structured_requests[0].messages[-1].body
    assert "Rowan repair notes follow-up" in prompt_body
    assert "Cass private festival letter" not in prompt_body
    assert "Mira solo arcade plan" not in prompt_body


def test_update_after_text_messages_resolves_numbered_message_aliases(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    follow_up = repositories.append_character_text_message(
        save_id=fixture.save_id,
        thread_id=fixture.reply.thread_id,
        character_id=fixture.thread_npc_id,
        sender="character",
        body="I'll bring spare bolts too.",
        provider="fake",
        model="fake-chat",
    )
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [
                {
                    "body": "Mira asked Rowan for the repair notes.",
                    "tags": ["repair"],
                    "importance": 0.5,
                    "source_text_message_id": "message_1",
                },
                {
                    "body": "Rowan promised to bring the repair notes.",
                    "tags": ["promise"],
                    "importance": 0.7,
                    "source_text_message_id": "message_2",
                },
                {
                    "body": "Rowan also offered spare bolts.",
                    "tags": ["supplies"],
                    "importance": 0.6,
                    "source_text_message_id": "message_3",
                },
            ],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
        }
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(fixture.player_message, fixture.reply, follow_up),
        )
    )

    assert result.status == "applied"
    assert result.memory_count == 3
    memories_by_body = {
        memory.body: memory for memory in repositories.list_memories(fixture.save_id)
    }
    assert memories_by_body[
        "Mira asked Rowan for the repair notes."
    ].source_message_ids == [character_text_source_ref(fixture.player_message.id)]
    assert memories_by_body[
        "Rowan promised to bring the repair notes."
    ].source_message_ids == [character_text_source_ref(fixture.reply.id)]
    assert memories_by_body[
        "Rowan also offered spare bolts."
    ].source_message_ids == [character_text_source_ref(follow_up.id)]


def test_group_player_text_can_scope_memory_to_nonresponding_participant(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    thread = repositories.create_character_text_group_thread(
        save_id=fixture.save_id,
        title="Repair Crew",
        character_ids=[fixture.thread_npc_id, fixture.unrelated_npc_id],
    )
    player_message = repositories.append_character_text_message(
        save_id=fixture.save_id,
        thread_id=thread.id,
        character_id=None,
        sender="player",
        sender_character_id=fixture.player_id,
        body="Both of you should remember the lantern bypass code.",
    )
    reply = repositories.append_character_text_message(
        save_id=fixture.save_id,
        thread_id=thread.id,
        character_id=fixture.thread_npc_id,
        sender="character",
        sender_character_id=fixture.thread_npc_id,
        body="I have it written down.",
        provider="fake",
        model="fake-chat",
        reply_to_message_id=player_message.id,
    )
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [
                {
                    "body": "Cass knows the lantern bypass code from the group text.",
                    "tags": ["cass", "group-text"],
                    "importance": 0.9,
                    "source_text_message_id": "player",
                    "character_id": fixture.unrelated_npc_id,
                    "knowledge_state": "knows",
                    "acquisition_method": "told",
                    "evidence_quote": "Both of you should remember",
                }
            ],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
        }
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(player_message, reply),
        )
    )

    assert result.status == "applied"
    assert result.memory_count == 1
    assert result.knowledge_edge_count == 1
    memories = repositories.list_memories(fixture.save_id)
    assert [memory.body for memory in memories] == [
        "Cass knows the lantern bypass code from the group text."
    ]
    edges = repositories.list_character_knowledge_edges(fixture.save_id)
    assert [
        (edge.character_id, edge.target_type, edge.target_id)
        for edge in edges
    ] == [(fixture.unrelated_npc_id, "memory", memories[0].id)]
    request = provider.structured_requests[0]
    assert fixture.unrelated_npc_id in _schema_enum(
        request,
        "memories",
        "character_id",
    )
    prompt_body = request.messages[-1].body
    assert fixture.unrelated_npc_id in prompt_body
    assert _rejected_audits(repositories, fixture.save_id) == []


def test_group_text_memory_without_character_id_defaults_to_all_participants(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    thread = repositories.create_character_text_group_thread(
        save_id=fixture.save_id,
        title="Repair Crew",
        character_ids=[fixture.thread_npc_id, fixture.unrelated_npc_id],
    )
    player_message = repositories.append_character_text_message(
        save_id=fixture.save_id,
        thread_id=thread.id,
        character_id=None,
        sender="player",
        sender_character_id=fixture.player_id,
        body="Both of you should remember the lantern bypass code.",
    )
    reply = repositories.append_character_text_message(
        save_id=fixture.save_id,
        thread_id=thread.id,
        character_id=fixture.thread_npc_id,
        sender="character",
        sender_character_id=fixture.thread_npc_id,
        body="I have it written down.",
        provider="fake",
        model="fake-chat",
        reply_to_message_id=player_message.id,
    )
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [
                {
                    "body": "The repair crew knows the lantern bypass code.",
                    "tags": ["group-text"],
                    "importance": 0.9,
                    "source_text_message_id": "player",
                    "knowledge_state": "knows",
                    "acquisition_method": "texted",
                    "evidence_quote": "Both of you should remember",
                }
            ],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
        }
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(player_message, reply),
        )
    )

    assert result.status == "applied"
    assert result.memory_count == 1
    assert result.knowledge_edge_count == 3
    [memory] = repositories.list_memories(fixture.save_id)
    edges = repositories.list_character_knowledge_edges(fixture.save_id)
    assert {
        (edge.character_id, edge.target_type, edge.target_id)
        for edge in edges
    } == {
        (fixture.player_id, "memory", memory.id),
        (fixture.thread_npc_id, "memory", memory.id),
        (fixture.unrelated_npc_id, "memory", memory.id),
    }


def test_structured_request_includes_thread_memory_and_prior_thread_context(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    unrelated = repositories.add_character(
        save_id=fixture.save_id,
        name="Niko",
        role="stagehand",
        met=True,
    )
    unrelated_thread = repositories.get_or_create_character_text_thread(
        save_id=fixture.save_id,
        character_id=unrelated.id,
        title=unrelated.name,
    )
    repositories.append_character_text_message(
        save_id=fixture.save_id,
        thread_id=unrelated_thread.id,
        character_id=unrelated.id,
        sender="character",
        body="Private unrelated thread about the festival letter.",
    )
    repositories.update_character_text_thread_memory(
        save_id=fixture.save_id,
        thread_id=fixture.reply.thread_id,
        body="Phone thread memory: Rowan and Mira planned the west arcade meet.",
        message_count=2,
    )
    repositories.update_character_text_thread_memory(
        save_id=fixture.save_id,
        thread_id=unrelated_thread.id,
        body="Private unrelated memory about the festival letter.",
        message_count=1,
    )
    current_player = repositories.append_character_text_message(
        save_id=fixture.save_id,
        thread_id=fixture.reply.thread_id,
        character_id=fixture.thread_npc_id,
        sender="player",
        body="Same place after class?",
    )
    current_reply = repositories.append_character_text_message(
        save_id=fixture.save_id,
        thread_id=fixture.reply.thread_id,
        character_id=fixture.thread_npc_id,
        sender="character",
        body="Yes, the west arcade.",
        provider="fake",
        model="fake-chat",
    )
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
        }
    )
    service = _service(repositories, provider)

    asyncio.run(
        service.update_after_text_messages(
            save_id=fixture.save_id,
            text_messages=(current_player, current_reply),
        )
    )

    prompt_body = provider.structured_requests[0].messages[-1].body
    assert "Phone thread memory" in prompt_body
    assert "west arcade meet" in prompt_body
    assert "Prior phone thread context" in prompt_body
    assert "Can you bring the repair notes?" in prompt_body
    assert "I promised I would bring the repair notes." in prompt_body
    assert "festival letter" not in prompt_body
    assert unrelated.id not in prompt_body


def test_run_retries_processes_queued_character_text_world_update_retry(
    repositories: PersistenceRepositories,
) -> None:
    fixture = _create_text_world_fixture(repositories)
    retry_job = repositories.create_job(
        save_id=fixture.save_id,
        type=CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE,
        status="queued",
        payload={
            "text_message_ids": [fixture.player_message.id, fixture.reply.id],
            "retry_attempt": 1,
            "max_retry_attempts": 3,
            "reason": "character_text_world_update_failed",
        },
    )
    provider = FakeStructuredTextWorldProvider(
        {
            "memories": [
                {
                    "body": "Rowan promised to bring the repair notes.",
                    "tags": ["promise", "rowan"],
                    "importance": 0.88,
                    "source_text_message_id": "reply",
                    "character_id": fixture.thread_npc_id,
                    "knowledge_state": "knows",
                    "acquisition_method": "told",
                    "evidence_quote": "I promised I would bring the repair notes.",
                }
            ],
            "active_threads": [],
            "character_updates": [],
            "dating_route_updates": [],
        }
    )
    service = _service(repositories, provider)

    completed = asyncio.run(service.run_retries(save_id=fixture.save_id))

    assert completed == 1
    assert provider.structured_requests[0].schema_name == "character_text_world_update"
    succeeded_retry = next(
        job
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.id == retry_job.id
    )
    assert succeeded_retry.result == {
        "status": "applied",
        "memory_count": 1,
        "active_thread_count": 0,
        "character_count": 0,
        "dating_route_count": 0,
        "contact_permission_count": 0,
        "knowledge_edge_count": 1,
        "audit_count": 1,
        "text_message_ids": [fixture.player_message.id, fixture.reply.id],
    }
    memories = repositories.list_memories(fixture.save_id)
    assert [memory.body for memory in memories] == [
        "Rowan promised to bring the repair notes."
    ]
    source_ref = character_text_source_ref(fixture.reply.id)
    assert memories[0].source_message_ids == [source_ref]
    provenance = repositories.list_character_text_provenance(
        save_id=fixture.save_id,
        text_message_id=fixture.reply.id,
    )
    assert {row.target_type for row in provenance} >= {
        "memory",
        "character_knowledge_edge",
    }


def _create_text_world_fixture(
    repositories: PersistenceRepositories,
) -> TextWorldFixture:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="After School",
        premise="A small town romance.",
        player_role="Mira",
        content={
            "player_character_name": "Mira",
            "characters": ["Rowan", "Cass"],
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
    rowan = repositories.add_character(
        save_id=save.id,
        name="Rowan",
        role="classmate",
        status="curious",
        met=True,
    )
    cass = repositories.add_character(
        save_id=save.id,
        name="Cass",
        role="club president",
        status="distant",
        met=True,
    )
    repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=rowan.id,
        stage="introduced",
        interest_level="curious",
        trust_level="guarded",
        next_reasonable_step="Ask about Rowan's arcade plans.",
    )
    repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=cass.id,
        stage="introduced",
        interest_level="polite",
        trust_level="distant",
        next_reasonable_step="Talk in Cass's thread.",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=rowan.id,
        title="Rowan",
    )
    player_message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="player",
        body="Can you bring the repair notes?",
    )
    reply = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="character",
        body="I promised I would bring the repair notes.",
        provider="fake",
        model="fake-chat",
    )
    return TextWorldFixture(
        save_id=save.id,
        player_id=player.id,
        thread_npc_id=rowan.id,
        unrelated_npc_id=cass.id,
        player_message=player_message,
        reply=reply,
    )


def _service(
    repositories: PersistenceRepositories,
    provider: FakeStructuredTextWorldProvider,
) -> CharacterTextWorldUpdateService:
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=["structured_output"],
    )
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    return CharacterTextWorldUpdateService(
        repositories=repositories,
        providers={"fake": provider},
    )


def _route_for(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    npc_character_id: str,
) -> DatingRouteStateRecord:
    return next(
        route
        for route in repositories.list_dating_route_states(save_id)
        if route.npc_character_id == npc_character_id
    )


def _character_for(
    repositories: PersistenceRepositories,
    character_id: str,
) -> CharacterRecord:
    character = repositories.get_character(character_id)
    assert character is not None
    return character


def _rejected_audits(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[ContextUpdateAuditRecord]:
    return [
        audit
        for audit in repositories.list_context_update_audit(save_id)
        if audit.operation == "rejected"
    ]


def _audit_after(audit: ContextUpdateAuditRecord) -> dict[str, object]:
    assert isinstance(audit.after, dict)
    return audit.after


def _schema_enum(
    request: StructuredOutputRequest,
    collection: str,
    field: str,
) -> set[str]:
    field_schema = request.schema["properties"][collection]["items"]["properties"][
        field
    ]
    enum = field_schema["enum"]
    assert isinstance(enum, list)
    return {value for value in enum if isinstance(value, str)}
