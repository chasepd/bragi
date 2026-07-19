from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import JobRecord, MessageRecord, SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ProviderClient,
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from bragi.services.context_cleanup_service import ContextCleanupService
from bragi.services.world_data_service import (
    WorldDataEdits,
    WorldDataScenarioEdit,
    WorldDataService,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        repositories.save_provider_model(
            provider="fake",
            model_id="fake-structured",
            display_name="Fake Structured",
            capabilities=["structured_output"],
        )
        yield repositories


class FakeStructuredCleanupProvider:
    provider_name = "fake"

    def __init__(
        self,
        responses: list[dict[str, object]],
        *,
        fail_on_request: int | None = None,
        raw_metadata: dict[str, object] | None = None,
    ) -> None:
        self.responses = responses
        self.fail_on_request = fail_on_request
        self.raw_metadata = raw_metadata or {}
        self.requests: list[StructuredOutputRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("context cleanup must not use normal chat text")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.requests.append(request)
        if self.fail_on_request == len(self.requests):
            raise TimeoutError("The read operation timed out")
        if not self.responses:
            raise AssertionError(
                f"unexpected structured request: {request.schema_name}"
            )
        return StructuredOutputResponse(
            data=self.responses.pop(0),
            provider=request.provider,
            model_id=request.model_id,
            raw_metadata=self.raw_metadata,
        )


class FakeToolCleanupProvider:
    provider_name = "fake"

    def __init__(self, responses: list[tuple[ProviderToolCall, ...]]) -> None:
        self.responses = responses
        self.structured_requests: list[StructuredOutputRequest] = []
        self.tool_requests: list[ToolCallRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("context cleanup must not use normal chat text")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        raise AssertionError("tool-capable context cleanup should prefer tools")

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected tool-call request")
        return ToolCallResponse(
            tool_calls=self.responses.pop(0),
            body="",
            provider=request.provider,
            model_id=request.model_id,
        )


def test_analyze_and_apply_uses_structured_output_and_scans_every_message(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    messages = [
        repositories.append_message(
            save_id=save.id,
            role="player" if index % 2 == 0 else "narrator",
            speaker_name="Mara" if index % 2 == 0 else "Narrator",
            body=f"turn body {index}",
            message_id=f"msg-{index}",
        )
        for index in range(5)
    ]
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": [{"message_id": "msg-0", "note": "first chunk"}]},
            {"notes": [{"message_id": "msg-2", "note": "second chunk"}]},
            {"notes": [{"message_id": "msg-4", "note": "third chunk"}]},
            {"actions": []},
        ]
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        transcript_chunk_size=2,
    )

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert result.scanned_messages == 5
    assert result.proposed_actions == 0
    assert result.applied_actions == 0
    assert [request.schema_name for request in provider.requests] == [
        "context_cleanup_scan",
        "context_cleanup_scan",
        "context_cleanup_scan",
        "context_cleanup_actions",
    ]
    scan_bodies = [request.messages[-1].body for request in provider.requests[:3]]
    assert [_message_ids_from_request_body(body) for body in scan_bodies] == [
        ["msg-0", "msg-1"],
        ["msg-2", "msg-3"],
        ["msg-4"],
    ]
    action_body = provider.requests[-1].messages[-1].body
    assert "Transcript scan notes:" in action_body
    assert "msg-0: first chunk" in action_body
    assert "msg-2: second chunk" in action_body
    assert "msg-4: third chunk" in action_body
    assert "Transcript message ids: msg-0, msg-1, msg-2, msg-3, msg-4" in action_body
    assert "Transcript messages:" not in action_body
    for message in messages:
        assert message.body not in action_body
    evidence_items = _actions_evidence_items(provider.requests[-1].schema)
    assert evidence_items["type"] == "string"
    assert "enum" not in evidence_items
    assert repositories.list_context_update_audit(save.id) == []
    assert _cleanup_jobs(repositories)[0].status == "succeeded"


def test_analyze_and_apply_reports_phase_specific_structured_tasks(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The old alarm is resolved.",
        message_id="msg-cleanup",
    )
    provider = FakeStructuredCleanupProvider(
        [{"notes": []}, {"actions": []}],
        raw_metadata={
            "_bragi_retry": {
                "attempt_count": 1,
                "max_attempts": 3,
                "attempts": [],
            }
        },
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        providers=cast(dict[str, ProviderClient], {"fake": provider}),
    )
    asyncio.run(service.analyze_and_apply(save.id))

    job = _cleanup_jobs(repositories)[0]
    assert job.result is not None
    calls = job.result["provider_calls"]
    assert isinstance(calls, list)
    assert [call["task"] for call in calls] == [
        "context_cleanup_scan",
        "context_cleanup_actions",
    ]
    assert [call["schema_name"] for call in calls] == [
        "context_cleanup_scan",
        "context_cleanup_actions",
    ]


def test_analyze_and_apply_prefers_tool_calls(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The old alarm is explicitly resolved and no longer true.",
        message_id="msg-cleanup",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "active"},
        category="scene",
        source_message_id=message.id,
    )
    provider = FakeToolCleanupProvider(
        [
            (
                ProviderToolCall(
                    id="call-note",
                    name="note_cleanup_candidate",
                    arguments_json=json.dumps(
                        {
                            "message_id": message.id,
                            "note": "old alarm explicitly resolved",
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-action",
                    name="propose_cleanup_action",
                    arguments_json=json.dumps(
                        {
                            "operation": "archive",
                            "target_type": "world_state",
                            "target_id": state.id,
                            "field_path": "*",
                            "value": None,
                            "reason": (
                                "The fact is explicitly resolved and no longer true."
                            ),
                            "confidence": 0.91,
                            "evidence_message_ids": [message.id],
                        }
                    ),
                ),
            ),
        ]
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
        prefer_tool_calls=True,
    )

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert provider.structured_requests == []
    assert [request.tools[0].name for request in provider.tool_requests] == [
        "note_cleanup_candidate",
        "propose_cleanup_action",
    ]
    assert result.applied_actions == 1
    assert result.archives == 1
    assert repositories.list_world_state(save.id) == []


def test_analyze_and_apply_tool_calls_retry_malformed_action(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The old alarm is explicitly resolved and no longer true.",
        message_id="msg-cleanup",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "active"},
        category="scene",
        source_message_id=message.id,
    )
    provider = FakeToolCleanupProvider(
        [
            (),
            (
                ProviderToolCall(
                    id="call-bad",
                    name="propose_cleanup_action",
                    arguments_json=json.dumps(
                        {
                            "operation": "explode",
                            "target_type": "world_state",
                            "target_id": state.id,
                            "field_path": "*",
                            "value": None,
                            "reason": "Bad operation.",
                            "confidence": 0.91,
                            "evidence_message_ids": [message.id],
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-good",
                    name="propose_cleanup_action",
                    arguments_json=json.dumps(
                        {
                            "operation": "archive",
                            "target_type": "world_state",
                            "target_id": state.id,
                            "field_path": "*",
                            "value": None,
                            "reason": (
                                "The fact is explicitly resolved and no longer true."
                            ),
                            "confidence": 0.91,
                            "evidence_message_ids": [message.id],
                        }
                    ),
                ),
            ),
        ]
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
        prefer_tool_calls=True,
    )

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert result.applied_actions == 1
    assert len(provider.tool_requests) == 3
    assert "operation must be one of" in provider.tool_requests[2].messages[-1].body


def test_analyze_and_apply_applies_valid_archive_update_and_delete_actions(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="storm.level",
        value={"status": "red"},
        source_message_id=player.id,
        state_id="state-storm",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The east gate is sealed forever.",
        tags=["gate"],
        source_message_id=player.id,
        memory_id="memory-stale",
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=player.id,
        location_id="location-gallery",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        location_id=location.id,
        source_message_id=player.id,
        character_id="character-ilyra",
    )
    link = repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="location",
        target_id=location.id,
        relation="left",
        link_id="link-stale",
    )
    archived_target_link = repositories.add_entity_link(
        save_id=save.id,
        entity_type="memory",
        entity_id=memory.id,
        target_type="location",
        target_id=location.id,
        relation="mentions",
        link_id="link-archived-target",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="world_state",
                        target_id=state.id,
                        field_path="value",
                        value={"status": "green"},
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="archive",
                        target_type="memory",
                        target_id=memory.id,
                        field_path="*",
                        value=None,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="delete",
                        target_type="entity_link",
                        target_id=link.id,
                        field_path="*",
                        value=None,
                        evidence_message_ids=[narrator.id],
                    ),
                ]
            },
        ]
    )
    service = _service(repositories, provider)

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert result.applied_actions == 3
    assert result.rejected_actions == 0
    assert result.archives == 1
    assert result.updates == 1
    assert result.deleted_links == 2
    assert repositories.list_world_state(save.id)[0].value == {"status": "green"}
    assert repositories.list_memories(save.id) == []
    assert repositories.list_entity_links(save.id) == []
    remaining_link_ids = {link.id for link in repositories.list_entity_links(save.id)}
    assert link.id not in remaining_link_ids
    assert archived_target_link.id not in remaining_link_ids
    audit_rows = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type, row.entity_id) for row in audit_rows] == [
        ("updated", "world_state", state.id),
        ("archived", "memory", memory.id),
        ("deleted", "entity_link", link.id),
    ]
    assert all(row.source_message_ids == [narrator.id] for row in audit_rows)
    job = _cleanup_jobs(repositories)[0]
    assert job.status == "succeeded"
    assert job.result == {
        "scanned_messages": 2,
        "scan_batches": 1,
        "cleanup_target_count": 6,
        "action_batches": 1,
        "proposed_actions": 3,
        "applied_actions": 3,
        "rejected_actions": 0,
        "archives": 1,
        "updates": 1,
        "deleted_links": 2,
    }


def test_analyze_and_apply_syncs_canonical_scene_time_update(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    scene = repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="morning",
        time_of_day="morning",
        world_time_clock_minutes=9 * 60 + 30,
        world_time_period_label="festival week",
        source_message_id=player.id,
        snapshot_id="scene-time-cleanup",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": [{"message_id": narrator.id, "note": "time moved evening"}]},
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="scene_snapshot",
                        target_id=scene.id,
                        field_path="in_world_time",
                        value="evening",
                        evidence_message_ids=[narrator.id],
                        confidence=0.86,
                    )
                ]
            },
        ]
    )

    result = asyncio.run(_service(repositories, provider).analyze_and_apply(save.id))

    assert result.applied_actions == 1
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "festival week evening at 09:30"
    assert snapshot.world_time_phase == "evening"
    assert snapshot.world_time_clock_minutes == 9 * 60 + 30
    assert snapshot.world_time_period_label == "festival week"
    assert snapshot.world_time_source_message_id == narrator.id
    assert snapshot.world_time_confidence == 0.86


def test_analyze_and_apply_respects_canonical_scene_time_phase_lock(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    scene = repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="morning",
        time_of_day="morning",
        world_time_phase="morning",
        source_message_id=player.id,
        locked_fields=["world_time_phase"],
        snapshot_id="scene-time-cleanup",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": [{"message_id": narrator.id, "note": "time moved evening"}]},
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="scene_snapshot",
                        target_id=scene.id,
                        field_path="time_of_day",
                        value="evening",
                        evidence_message_ids=[narrator.id],
                        confidence=0.86,
                    )
                ]
            },
        ]
    )

    result = asyncio.run(_service(repositories, provider).analyze_and_apply(save.id))

    assert result.applied_actions == 0
    assert result.rejected_actions == 1
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.time_of_day == "morning"
    assert snapshot.world_time_phase == "morning"
    rejected = repositories.list_context_update_audit(save.id)[0]
    assert rejected.operation == "rejected"
    assert rejected.reason.startswith("Cleanup field is locked")


def test_analyze_and_apply_archives_supported_context_records_and_links(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="archive.anchor",
        value={"purpose": "link cleanup target"},
        source_message_id=player.id,
        state_id="state-link-anchor",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The loose tile was already catalogued elsewhere.",
        tags=["detail"],
        source_message_id=player.id,
        memory_id="memory-archive-target",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=player.id,
        covers_message_end_id=narrator.id,
        body="The watch discussed old tile details.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-archive-target",
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Old Storeroom",
        source_message_id=player.id,
        location_id="location-archive-target",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Archivist Pell",
        source_message_id=player.id,
        character_id="character-archive-target",
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Check the storeroom tile",
        description="The tile might hide a key.",
        status="active",
        visibility="public",
        source_message_id=player.id,
        thread_id="thread-archive-target",
    )
    for entity_type, entity_id in (
        ("memory", memory.id),
        ("summary", summary.id),
        ("location", location.id),
        ("character", character.id),
        ("active_thread", thread.id),
    ):
        repositories.add_entity_link(
            save_id=save.id,
            entity_type=entity_type,
            entity_id=entity_id,
            target_type="world_state",
            target_id=state.id,
            relation="mentions",
            link_id=f"link-{entity_type}",
        )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    _action(
                        operation="archive",
                        target_type="memory",
                        target_id=memory.id,
                        field_path="*",
                        value=None,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="archive",
                        target_type="summary",
                        target_id=summary.id,
                        field_path="*",
                        value=None,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="archive",
                        target_type="location",
                        target_id=location.id,
                        field_path="*",
                        value=None,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="archive",
                        target_type="character",
                        target_id=character.id,
                        field_path="*",
                        value=None,
                        reason="The narrator explicitly superseded this character.",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="archive",
                        target_type="active_thread",
                        target_id=thread.id,
                        field_path="*",
                        value=None,
                        reason="The narrator explicitly resolved this thread.",
                        evidence_message_ids=[narrator.id],
                    ),
                ]
            },
        ]
    )

    result = asyncio.run(_service(repositories, provider).analyze_and_apply(save.id))

    assert result.applied_actions == 5
    assert result.archives == 5
    assert result.deleted_links == 5
    assert repositories.list_memories(save.id) == []
    assert repositories.list_summaries(save.id) == []
    assert repositories.get_location(location.id) is None
    assert repositories.get_character(character.id) is None
    assert repositories.get_active_thread(thread.id) is None
    assert repositories.list_entity_links(save.id) == []
    assert repositories.list_world_state(save.id) == [state]
    audit_rows = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type, row.entity_id) for row in audit_rows] == [
        ("archived", "memory", memory.id),
        ("archived", "summary", summary.id),
        ("archived", "location", location.id),
        ("archived", "character", character.id),
        ("archived", "active_thread", thread.id),
    ]


def test_analyze_and_apply_rejects_protected_character_archive(
    repositories: PersistenceRepositories,
) -> None:
    save, _player, narrator = _create_save_with_messages(repositories)
    protected = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        protected_from_maintenance=True,
        character_id="character-protected",
    )
    link = repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=protected.id,
        target_type="memory",
        target_id="memory-important",
        relation="knows",
        link_id="link-protected-character",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    _action(
                        operation="archive",
                        target_type="character",
                        target_id=protected.id,
                        field_path="*",
                        value=None,
                        reason="The character seems stale but is protected.",
                        confidence=0.95,
                        evidence_message_ids=[narrator.id],
                    ),
                ],
            },
        ]
    )

    result = asyncio.run(_service(repositories, provider).analyze_and_apply(save.id))

    assert result.applied_actions == 0
    assert result.rejected_actions == 1
    assert repositories.get_character(protected.id) is not None
    stored_link_ids = [
        stored_link.id for stored_link in repositories.list_entity_links(save.id)
    ]
    assert stored_link_ids == [link.id]
    audit = repositories.list_context_update_audit(save.id)[0]
    assert audit.operation == "rejected"
    assert audit.entity_type == "character"
    assert audit.entity_id == protected.id
    assert "protected from maintenance" in audit.reason
    assert "protected=true" in provider.requests[1].messages[-1].body


def test_analyze_and_apply_rejects_invalid_reference_and_value_updates(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    other_save = _create_save(repositories, title="Other Save")
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        source_message_id=player.id,
        location_id="location-gallery",
    )
    other_location = repositories.add_location(
        save_id=other_save.id,
        name="Other Gallery",
        location_id="location-other",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        location_id=location.id,
        source_message_id=player.id,
        character_id="character-ilyra",
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Warn the east gate",
        description="The east gate needs a warning.",
        status="active",
        source_message_id=player.id,
        thread_id="thread-east-gate",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The east gate warning remains active.",
        tags=["gate"],
        importance=0.5,
        source_message_id=player.id,
        memory_id="memory-gate",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=player.id,
        covers_message_end_id=narrator.id,
        body="The watch still needs the gate warning.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-gate",
    )
    scene = repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="The gallery is quiet.",
        present_character_ids=[character.id],
        source_message_id=player.id,
        snapshot_id="scene-current",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="scene_snapshot",
                        target_id=scene.id,
                        field_path="current_location_id",
                        value=other_location.id,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="scene_snapshot",
                        target_id=scene.id,
                        field_path="present_character_ids",
                        value=[character.id, 3],
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="location",
                        target_id=location.id,
                        field_path="parent_location_id",
                        value=other_location.id,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="location",
                        target_id=location.id,
                        field_path="aliases",
                        value="gallery",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="character",
                        target_id=character.id,
                        field_path="met",
                        value="yes",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="character",
                        target_id=character.id,
                        field_path="relationships",
                        value=["ally"],
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="active_thread",
                        target_id=thread.id,
                        field_path="priority",
                        value=True,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="active_thread",
                        target_id=thread.id,
                        field_path="related_entities",
                        value=["location:location-gallery", 7],
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="memory",
                        target_id=memory.id,
                        field_path="tags",
                        value="gate, warning",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="memory",
                        target_id=memory.id,
                        field_path="importance",
                        value=True,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="summary",
                        target_id=summary.id,
                        field_path="provider",
                        value="other-provider",
                        evidence_message_ids=[narrator.id],
                    ),
                ]
            },
        ]
    )

    result = asyncio.run(_service(repositories, provider).analyze_and_apply(save.id))

    assert result.applied_actions == 0
    assert result.rejected_actions == 11
    assert repositories.get_scene_snapshot(save.id) == scene
    assert repositories.get_location(location.id) == location
    assert repositories.get_character(character.id) == character
    assert repositories.get_active_thread(thread.id) == thread
    assert repositories.list_memories(save.id) == [memory]
    assert repositories.list_summaries(save.id) == [summary]
    rejected_reasons = [
        row.reason
        for row in repositories.list_context_update_audit(save.id)
        if row.operation == "rejected"
    ]
    expected_reasons = (
        "Location reference must belong to the active save",
        "present_character_ids must be a string list",
        "aliases must be a string list",
        "Character met must be a boolean",
        "Character relationships must be an object",
        "Active-thread priority must be an integer",
        "related_entities must be a string list",
        "Memory tags must be a string list",
        "Memory importance must be a number",
        "Unsupported cleanup field: summary.provider",
    )
    missing_reasons = [
        expected
        for expected in expected_reasons
        if not any(expected in reason for reason in rejected_reasons)
    ]
    assert missing_reasons == []


def test_analyze_and_apply_batches_large_cleanup_target_registry(
    repositories: PersistenceRepositories,
) -> None:
    save, _player, _narrator = _create_save_with_messages(repositories)
    memories = [
        repositories.add_memory(
            save_id=save.id,
            body=f"Low-value memory {index}",
            tags=["detail"],
            memory_id=f"memory-large-{index:02d}",
        )
        for index in range(23)
    ]
    states = [
        repositories.upsert_world_state(
            save_id=save.id,
            key=f"scene.large_{index:02d}",
            value={"index": index},
            category="scene",
            state_id=f"state-large-{index:02d}",
        )
        for index in range(12)
    ]
    for index in range(220):
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id=f"context-source-{index:02d}",
            title=f"context source {index}",
            body="context source body",
            metadata={"indexed_by": "test"},
        )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {"actions": []},
            {"actions": []},
            {"actions": []},
            {"actions": []},
            {"actions": []},
        ]
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        target_batch_size=10,
    )

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert result.action_batches == 5
    assert result.cleanup_target_count == 35
    action_requests = [
        request
        for request in provider.requests
        if request.schema_name == "context_cleanup_actions"
    ]
    assert len(action_requests) == 5
    target_line_counts = [
        sum(
            line.startswith(("- memory ", "- world_state "))
            for line in request.messages[-1].body.splitlines()
        )
        for request in action_requests
    ]
    assert target_line_counts == [10, 2, 10, 10, 3]
    first_action_body = action_requests[0].messages[-1].body
    assert states[0].id in first_action_body
    assert states[9].id in first_action_body
    assert states[10].id not in first_action_body
    assert memories[0].id not in first_action_body
    all_action_text = "\n".join(
        request.messages[-1].body for request in action_requests
    )
    assert "context-source-219" not in all_action_text

    job = _cleanup_jobs(repositories)[0]
    assert job.result is not None
    assert job.result["action_batches"] == 5
    assert job.result["cleanup_target_count"] == 35


def test_analyze_and_apply_failed_later_action_batch_records_partial_progress(
    repositories: PersistenceRepositories,
) -> None:
    save, _player, narrator = _create_save_with_messages(repositories)
    memories = [
        repositories.add_memory(
            save_id=save.id,
            body=f"Over-specific scene detail {index}",
            tags=["detail"],
            memory_id=f"memory-batch-{index}",
        )
        for index in range(2)
    ]
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    _action(
                        operation="archive",
                        target_type="memory",
                        target_id=memories[0].id,
                        field_path="*",
                        value=None,
                        evidence_message_ids=[narrator.id],
                    )
                ]
            },
        ],
        fail_on_request=3,
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        target_batch_size=1,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(service.analyze_and_apply(save.id))

    assert [memory.id for memory in repositories.list_memories(save.id)] == [
        memories[1].id
    ]
    assert [request.schema_name for request in provider.requests] == [
        "context_cleanup_scan",
        "context_cleanup_actions",
        "context_cleanup_actions",
    ]
    job = _cleanup_jobs(repositories)[0]
    assert job.status == "failed"
    assert job.error is not None
    assert "timed out" in job.error
    assert job.result is not None
    assert job.result["cleanup_target_count"] == 2
    assert job.result["action_batches"] == 2
    assert job.result["completed_action_batches"] == 1
    assert job.result["proposed_actions"] == 1
    assert job.result["applied_actions"] == 1
    assert job.result["archives"] == 1


def test_analyze_and_apply_caps_cleanup_notes_and_message_id_reference(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    for index in range(12):
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            body=f"turn body {index}",
            message_id=f"msg-{index:02d}",
        )
    responses: list[dict[str, object]] = []
    for index in range(12):
        responses.append(
            {"notes": [{"message_id": f"msg-{index:02d}", "note": "x" * 120}]}
        )
    responses.append({"actions": []})
    provider = FakeStructuredCleanupProvider(responses)
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        transcript_chunk_size=1,
        action_scan_note_limit=3,
        action_scan_note_char_limit=20,
        action_message_reference_limit=4,
    )

    asyncio.run(service.analyze_and_apply(save.id))

    action_body = provider.requests[-1].messages[-1].body
    assert "9 older scan notes omitted" in action_body
    assert "msg-09" in action_body
    assert "msg-10" in action_body
    assert "msg-11" in action_body
    assert "msg-00" not in action_body
    assert "x" * 21 not in action_body
    assert "Transcript message ids (showing last 4 of 12):" in action_body


def test_analyze_and_apply_rejects_unsurfaced_structured_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="storm.warning",
        value={"level": "red"},
        category="weather",
        source_message_id=player.id,
        state_id="state-warning",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="world_state",
                        target_id=state.id,
                        field_path="value",
                        value={"level": "green"},
                        evidence_message_ids=[player.id],
                    )
                ]
            },
        ]
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        action_message_reference_limit=1,
    )

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert result.applied_actions == 0
    assert result.rejected_actions == 1
    assert repositories.list_world_state(save.id)[0].value == {"level": "red"}
    audit = repositories.list_context_update_audit(save.id)[0]
    assert audit.operation == "rejected"
    assert (
        f"Evidence message id was not surfaced to context cleanup: {player.id}"
        in audit.reason
    )
    action_body = provider.requests[-1].messages[-1].body
    assert f"Transcript message ids (showing last 1 of 2): {narrator.id}" in action_body


def test_analyze_and_apply_allows_evidence_from_visible_scan_notes(
    repositories: PersistenceRepositories,
) -> None:
    save, player, _narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="storm.warning",
        value={"level": "red"},
        category="weather",
        source_message_id=player.id,
        state_id="state-warning",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {
                "notes": [
                    {
                        "message_id": player.id,
                        "note": "player identified the stale storm warning",
                    }
                ]
            },
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="world_state",
                        target_id=state.id,
                        field_path="value",
                        value={"level": "green"},
                        evidence_message_ids=[player.id],
                    )
                ]
            },
        ]
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        action_message_reference_limit=1,
    )

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert result.applied_actions == 1
    assert result.rejected_actions == 0
    assert repositories.list_world_state(save.id)[0].value == {"level": "green"}


def test_analyze_and_apply_tool_calls_retry_unsurfaced_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="storm.warning",
        value={"level": "red"},
        category="weather",
        source_message_id=player.id,
        state_id="state-warning",
    )
    provider = FakeToolCleanupProvider(
        [
            (),
            (
                ProviderToolCall(
                    id="call-bad-evidence",
                    name="propose_cleanup_action",
                    arguments_json=json.dumps(
                        _action(
                            operation="update",
                            target_type="world_state",
                            target_id=state.id,
                            field_path="value",
                            value={"level": "green"},
                            evidence_message_ids=[player.id],
                        )
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-good-evidence",
                    name="propose_cleanup_action",
                    arguments_json=json.dumps(
                        _action(
                            operation="update",
                            target_type="world_state",
                            target_id=state.id,
                            field_path="value",
                            value={"level": "green"},
                            evidence_message_ids=[narrator.id],
                        )
                    ),
                ),
            ),
        ]
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
        prefer_tool_calls=True,
        action_message_reference_limit=1,
    )

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert result.applied_actions == 1
    assert result.rejected_actions == 0
    assert repositories.list_world_state(save.id)[0].value == {"level": "green"}
    assert len(provider.tool_requests) == 3
    assert (
        f"Evidence message id was not surfaced to context cleanup: {player.id}"
        in provider.tool_requests[2].messages[-1].body
    )


def test_cleanup_rejects_high_value_archive_without_contradiction_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save, _player, narrator = _create_save_with_messages(repositories)
    memory = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra promised to hold the east stair.",
        tags=["promise"],
        source_message_id=narrator.id,
        memory_id="memory-promise",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    _action(
                        operation="archive",
                        target_type="memory",
                        target_id=memory.id,
                        field_path="*",
                        value=None,
                        evidence_message_ids=[narrator.id],
                        reason="archive memory because it seems old",
                    ),
                ]
            },
        ]
    )

    result = asyncio.run(_service(repositories, provider).analyze_and_apply(save.id))

    assert result.applied_actions == 0
    assert result.rejected_actions == 1
    assert repositories.list_memories(save.id) == [memory]
    audit = repositories.list_context_update_audit(save.id)
    assert audit[0].operation == "rejected"
    assert "High-value continuity facts require explicit contradiction" in str(
        audit[0].reason
    )


def test_cleanup_rejects_negated_contradiction_reason_for_high_value_archive(
    repositories: PersistenceRepositories,
) -> None:
    save, _player, narrator = _create_save_with_messages(repositories)
    memory = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra promised to hold the east stair.",
        tags=["promise"],
        source_message_id=narrator.id,
        memory_id="memory-negated-contradiction",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    _action(
                        operation="archive",
                        target_type="memory",
                        target_id=memory.id,
                        field_path="*",
                        value=None,
                        evidence_message_ids=[narrator.id],
                        reason=(
                            "This does not contradict current facts, but is old."
                        ),
                    ),
                ]
            },
        ]
    )

    result = asyncio.run(_service(repositories, provider).analyze_and_apply(save.id))

    assert result.applied_actions == 0
    assert result.rejected_actions == 1
    assert repositories.list_memories(save.id) == [memory]


def test_analyze_and_apply_rejects_world_state_metadata_and_bad_importance_updates(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="storm.level",
        value={"status": "red"},
        category="weather",
        confidence=0.82,
        source_message_id=player.id,
        state_id="state-storm",
    )
    low_importance_memory = repositories.add_memory(
        save_id=save.id,
        body="The east gate was repaired.",
        tags=["gate"],
        importance=0.2,
        source_message_id=player.id,
        memory_id="memory-low-importance",
    )
    high_importance_memory = repositories.add_memory(
        save_id=save.id,
        body="The beacon lens was cracked.",
        tags=["beacon"],
        importance=0.8,
        source_message_id=player.id,
        memory_id="memory-high-importance",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="world_state",
                        target_id=state.id,
                        field_path="category",
                        value="threat",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="world_state",
                        target_id=state.id,
                        field_path="confidence",
                        value=0.3,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="memory",
                        target_id=low_importance_memory.id,
                        field_path="importance",
                        value=-0.01,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="memory",
                        target_id=high_importance_memory.id,
                        field_path="importance",
                        value=1.01,
                        evidence_message_ids=[narrator.id],
                    ),
                ]
            },
        ]
    )
    service = _service(repositories, provider)

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert result.applied_actions == 0
    assert result.rejected_actions == 4
    saved_state = repositories.list_world_state(save.id)[0]
    assert saved_state.category == "weather"
    assert saved_state.confidence == 0.82
    saved_memories = {
        memory.id: memory for memory in repositories.list_memories(save.id)
    }
    assert saved_memories[low_importance_memory.id].importance == 0.2
    assert saved_memories[high_importance_memory.id].importance == 0.8
    assert repositories.list_state_changes(save.id) == []
    audit_rows = repositories.list_context_update_audit(save.id)
    assert [row.operation for row in audit_rows] == ["rejected"] * 4
    rejected_reasons = [row.reason for row in audit_rows]
    assert any(
        "Unsupported cleanup field: world_state.category" in reason
        for reason in rejected_reasons
    )
    assert any(
        "Unsupported cleanup field: world_state.confidence" in reason
        for reason in rejected_reasons
    )
    assert (
        sum(
            "Memory importance must be between 0 and 1" in reason
            for reason in rejected_reasons
        )
        == 2
    )


def test_analyze_and_apply_rejects_invalid_or_unsafe_actions(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    other_save = _create_save(repositories, title="Other Save")
    memory = repositories.add_memory(
        save_id=save.id,
        body="The lens was cracked.",
        tags=["lens"],
        source_message_id=player.id,
        memory_id="memory-active",
    )
    other_memory = repositories.add_memory(
        save_id=other_save.id,
        body="Other save memory.",
        tags=["other"],
        memory_id="memory-other-save",
    )
    null_update_memory = repositories.add_memory(
        save_id=save.id,
        body="This body must stay intact.",
        tags=["null"],
        source_message_id=player.id,
        memory_id="memory-null-update",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        status="present",
        locked_fields=["status"],
        source_message_id=player.id,
        character_id="character-locked",
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Locked Gallery",
        locked_fields=["archive"],
        source_message_id=player.id,
        location_id="location-locked",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    _action(
                        operation="erase",
                        target_type="memory",
                        target_id=memory.id,
                        field_path="body",
                        value="ignored",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="quest",
                        target_id=memory.id,
                        field_path="body",
                        value="ignored",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="memory",
                        target_id="missing-memory",
                        field_path="body",
                        value="ignored",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="memory",
                        target_id=other_memory.id,
                        field_path="body",
                        value="wrong save",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="memory",
                        target_id=null_update_memory.id,
                        field_path="body",
                        value=None,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="memory",
                        target_id=memory.id,
                        field_path="body",
                        value="The lens is intact.",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="memory",
                        target_id=memory.id,
                        field_path="body",
                        value="Duplicate should be rejected.",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="memory",
                        target_id=memory.id,
                        field_path="tags",
                        value=["lens", "updated"],
                        confidence=0.64,
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="update",
                        target_type="character",
                        target_id=character.id,
                        field_path="status",
                        value="gone",
                        evidence_message_ids=[narrator.id],
                    ),
                    _action(
                        operation="archive",
                        target_type="location",
                        target_id=location.id,
                        field_path="*",
                        value=None,
                        evidence_message_ids=[narrator.id],
                    ),
                ]
            },
        ]
    )
    service = _service(repositories, provider)

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert result.applied_actions == 1
    assert result.rejected_actions == 9
    saved_memories = {
        memory.id: memory for memory in repositories.list_memories(save.id)
    }
    assert saved_memories[memory.id].body == "The lens is intact."
    assert saved_memories[null_update_memory.id].body == "This body must stay intact."
    saved_character = repositories.get_character(character.id)
    assert saved_character is not None
    assert saved_character.status == "present"
    assert repositories.get_location(location.id) is not None
    audit_rows = repositories.list_context_update_audit(save.id)
    assert [row.operation for row in audit_rows].count("updated") == 1
    rejected_reasons = [row.reason for row in audit_rows if row.operation == "rejected"]
    assert any(
        "Unsupported cleanup operation: erase" in reason
        for reason in rejected_reasons
    )
    assert any(
        "Unsupported cleanup target type: quest" in reason
        for reason in rejected_reasons
    )
    assert sum(
        "Cleanup action target is unknown or not in the active save" in reason
        for reason in rejected_reasons
    ) == 2
    assert any("Duplicate cleanup action" in reason for reason in rejected_reasons)
    assert any("body must be text" in reason for reason in rejected_reasons)
    assert any("confidence is below threshold" in reason for reason in rejected_reasons)
    assert any("Cleanup field is locked" in reason for reason in rejected_reasons)
    assert any(
        "Cleanup target archive is locked" in reason for reason in rejected_reasons
    )
    action_prompt = "\n".join(message.body for message in provider.requests[1].messages)
    assert "locked(read-only)=status" in action_prompt
    assert _cleanup_jobs(repositories)[0].status == "succeeded"


def test_analyze_and_apply_handles_empty_saves_and_empty_actions_cleanly(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    provider = FakeStructuredCleanupProvider([{"actions": []}])
    service = _service(repositories, provider)

    result = asyncio.run(service.analyze_and_apply(save.id))

    assert result.scanned_messages == 0
    assert result.proposed_actions == 0
    assert result.applied_actions == 0
    assert result.rejected_actions == 0
    assert [request.schema_name for request in provider.requests] == [
        "context_cleanup_actions"
    ]
    assert "Transcript message ids: none" in provider.requests[0].messages[-1].body
    assert repositories.list_context_update_audit(save.id) == []
    assert _cleanup_jobs(repositories)[0].status == "succeeded"


def test_guided_cleanup_queues_reviewable_world_state_fix_before_apply(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="storm.warning",
        value={"level": "red"},
        category="weather",
        source_message_id=player.id,
        state_id="state-warning",
    )
    instruction = "The storm warning should be green, not red."
    provider = FakeStructuredCleanupProvider(
        [
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="world_state",
                        target_id=state.id,
                        field_path="value",
                        value={"level": "green"},
                        reason="The user identified the stale storm warning value.",
                        confidence=0.86,
                        evidence_message_ids=[narrator.id],
                    )
                ]
            },
        ]
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.propose_guided_cleanup(save.id, instruction=instruction)
    )

    assert result.queued_suggestions == 1
    assert result.rejected_actions == 0
    assert repositories.list_world_state(save.id)[0].value == {"level": "red"}
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.update_type == "upsert"
    assert suggestion.entity_type == "world_state"
    assert suggestion.entity_id == state.id
    assert suggestion.field_path == "storm.warning"
    assert suggestion.status == "pending"
    assert suggestion.proposed_value == {
        "operation": "upsert",
        "key": "storm.warning",
        "value": {"level": "green"},
        "category": "weather",
        "confidence": 0.86,
        "source_message_id": narrator.id,
    }
    queued_audit = repositories.list_context_update_audit(save.id)[0]
    assert queued_audit.operation == "guided_cleanup_queued"
    assert queued_audit.entity_id == state.id
    assert instruction in queued_audit.reason

    _apply_first_pending_suggestion(repositories, save.id)

    assert repositories.list_world_state(save.id)[0].value == {"level": "green"}
    audit_operations = [
        row.operation for row in repositories.list_context_update_audit(save.id)
    ]
    assert audit_operations == [
        "guided_cleanup_queued",
        "manual_suggestion_apply",
    ]
    assert [request.schema_name for request in provider.requests] == [
        "guided_context_cleanup_actions"
    ]
    assert instruction in provider.requests[0].messages[-1].body


def test_guided_cleanup_rejects_protected_character_archive(
    repositories: PersistenceRepositories,
) -> None:
    save, _player, narrator = _create_save_with_messages(repositories)
    protected = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        protected_from_maintenance=True,
        character_id="character-protected",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {
                "actions": [
                    _action(
                        operation="archive",
                        target_type="character",
                        target_id=protected.id,
                        field_path="*",
                        value=None,
                        reason="The explicit cleanup request targets this character.",
                        confidence=0.95,
                        evidence_message_ids=[narrator.id],
                    ),
                ],
            },
        ]
    )

    result = asyncio.run(
        _service(repositories, provider).propose_guided_cleanup(
            save.id,
            instruction="Archive this character.",
        )
    )

    assert result.queued_suggestions == 0
    assert result.rejected_actions == 1
    assert repositories.list_context_update_suggestions(save.id) == []
    assert repositories.get_character(protected.id) is not None
    audit = repositories.list_context_update_audit(save.id)[0]
    assert audit.operation == "guided_cleanup_rejected"
    assert audit.entity_type == "character"
    assert audit.entity_id == protected.id
    assert "protected from maintenance" in audit.reason
    assert "protected=true" in provider.requests[0].messages[-1].body


def test_guided_cleanup_tool_call_queues_reviewable_world_state_fix(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="storm.warning",
        value={"level": "red"},
        category="weather",
        source_message_id=player.id,
        state_id="state-warning",
    )
    instruction = "The storm warning should be green, not red."
    provider = FakeToolCleanupProvider(
        [
            (
                ProviderToolCall(
                    id="call-guided-fix",
                    name="propose_cleanup_action",
                    arguments_json=json.dumps(
                        _action(
                            operation="update",
                            target_type="world_state",
                            target_id=state.id,
                            field_path="value",
                            value={"level": "green"},
                            reason=(
                                "The user identified the stale storm warning value."
                            ),
                            confidence=0.86,
                            evidence_message_ids=[narrator.id],
                        )
                    ),
                ),
            )
        ]
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
        prefer_tool_calls=True,
    )

    result = asyncio.run(
        service.propose_guided_cleanup(save.id, instruction=instruction)
    )

    assert provider.structured_requests == []
    assert result.queued_suggestions == 1
    assert result.rejected_actions == 0
    assert repositories.list_world_state(save.id)[0].value == {"level": "red"}
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.update_type == "upsert"
    assert suggestion.entity_type == "world_state"
    assert suggestion.entity_id == state.id
    assert suggestion.field_path == "storm.warning"
    assert suggestion.proposed_value == {
        "operation": "upsert",
        "key": "storm.warning",
        "value": {"level": "green"},
        "category": "weather",
        "confidence": 0.86,
        "source_message_id": narrator.id,
    }
    audit = repositories.list_context_update_audit(save.id)[0]
    assert audit.operation == "guided_cleanup_queued"
    assert audit.entity_id == state.id
    assert instruction in audit.reason
    assert [tool.name for tool in provider.tool_requests[0].tools] == [
        "propose_cleanup_action"
    ]

    _apply_first_pending_suggestion(repositories, save.id)

    assert repositories.list_world_state(save.id)[0].value == {"level": "green"}


def test_guided_cleanup_rejects_unsurfaced_structured_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="storm.warning",
        value={"level": "red"},
        category="weather",
        source_message_id=player.id,
        state_id="state-warning",
    )
    instruction = "The storm warning should be green, not red."
    provider = FakeStructuredCleanupProvider(
        [
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="world_state",
                        target_id=state.id,
                        field_path="value",
                        value={"level": "green"},
                        evidence_message_ids=[player.id],
                    )
                ]
            },
        ]
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        action_message_reference_limit=1,
    )

    result = asyncio.run(
        service.propose_guided_cleanup(save.id, instruction=instruction)
    )

    assert result.queued_suggestions == 0
    assert result.rejected_actions == 1
    assert repositories.list_context_update_suggestions(save.id) == []
    assert repositories.list_world_state(save.id)[0].value == {"level": "red"}
    audit = repositories.list_context_update_audit(save.id)[0]
    assert audit.operation == "guided_cleanup_rejected"
    assert (
        f"Evidence message id was not surfaced to context cleanup: {player.id}"
        in audit.reason
    )
    action_body = provider.requests[0].messages[-1].body
    assert f"Transcript message ids (showing last 1 of 2): {narrator.id}" in action_body


def test_guided_cleanup_tool_calls_retry_unsurfaced_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="storm.warning",
        value={"level": "red"},
        category="weather",
        source_message_id=player.id,
        state_id="state-warning",
    )
    instruction = "The storm warning should be green, not red."
    provider = FakeToolCleanupProvider(
        [
            (
                ProviderToolCall(
                    id="call-bad-evidence",
                    name="propose_cleanup_action",
                    arguments_json=json.dumps(
                        _action(
                            operation="update",
                            target_type="world_state",
                            target_id=state.id,
                            field_path="value",
                            value={"level": "green"},
                            evidence_message_ids=[player.id],
                        )
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-good-evidence",
                    name="propose_cleanup_action",
                    arguments_json=json.dumps(
                        _action(
                            operation="update",
                            target_type="world_state",
                            target_id=state.id,
                            field_path="value",
                            value={"level": "green"},
                            evidence_message_ids=[narrator.id],
                        )
                    ),
                ),
            ),
        ]
    )
    service = ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
        prefer_tool_calls=True,
        action_message_reference_limit=1,
    )

    result = asyncio.run(
        service.propose_guided_cleanup(save.id, instruction=instruction)
    )

    assert result.queued_suggestions == 1
    assert result.rejected_actions == 0
    assert repositories.list_world_state(save.id)[0].value == {"level": "red"}
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.source_message_ids == [narrator.id]
    assert len(provider.tool_requests) == 2
    assert (
        f"Evidence message id was not surfaced to context cleanup: {player.id}"
        in provider.tool_requests[1].messages[-1].body
    )


def test_guided_cleanup_suppresses_duplicate_pending_suggestion(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="storm.warning",
        value={"level": "red"},
        category="weather",
        source_message_id=player.id,
        state_id="state-warning",
    )
    existing = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="upsert",
        entity_type="world_state",
        entity_id=state.id,
        field_path="storm.warning",
        proposed_value={
            "operation": "upsert",
            "key": "storm.warning",
            "value": {"level": "green"},
            "category": "weather",
            "confidence": 0.86,
            "source_message_id": narrator.id,
        },
        reason="Previous guided cleanup suggested this.",
        confidence=0.86,
        source_message_ids=[narrator.id],
    )
    provider = FakeStructuredCleanupProvider(
        [
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="world_state",
                        target_id=state.id,
                        field_path="value",
                        value={"level": "green"},
                        reason="The user identified the stale storm warning value.",
                        confidence=0.86,
                        evidence_message_ids=[narrator.id],
                    )
                ]
            },
        ]
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.propose_guided_cleanup(
            save.id,
            instruction="The storm warning should be green, not red.",
        )
    )

    assert result.queued_suggestions == 0
    assert result.rejected_actions == 0
    suggestions = repositories.list_context_update_suggestions(save.id)
    assert [suggestion.id for suggestion in suggestions] == [existing.id]
    assert repositories.list_context_update_audit(save.id) == []


def test_guided_cleanup_queues_active_thread_archive_for_review(
    repositories: PersistenceRepositories,
) -> None:
    save, _player, narrator = _create_save_with_messages(repositories)
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Warn the east gate",
        description="The east gate still needs a warning.",
        status="open",
        source_message_id=narrator.id,
        thread_id="thread-east-gate",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {
                "actions": [
                    _action(
                        operation="archive",
                        target_type="active_thread",
                        target_id=thread.id,
                        field_path="*",
                        value=None,
                        reason="The user says this thread has already been resolved.",
                        confidence=0.82,
                        evidence_message_ids=[narrator.id],
                    )
                ]
            },
        ]
    )
    service = _service(repositories, provider)

    result = asyncio.run(
        service.propose_guided_cleanup(
            save.id,
            instruction="The east gate warning thread is already resolved.",
        )
    )

    assert result.queued_suggestions == 1
    assert repositories.list_active_threads(save.id) == [thread]
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.update_type == "archive"
    assert suggestion.entity_type == "active_thread"
    assert suggestion.entity_id == thread.id

    _apply_first_pending_suggestion(repositories, save.id)

    assert repositories.list_active_threads(save.id) == []
    audit_rows = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type, row.entity_id) for row in audit_rows] == [
        ("guided_cleanup_queued", "active_thread", thread.id),
        ("manual_suggestion_apply", "active_thread", thread.id),
    ]


def test_guided_cleanup_handles_contradictory_memory_as_reviewed_archive(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    stale_memory = repositories.add_memory(
        save_id=save.id,
        body="The beacon lens is cracked.",
        tags=["beacon"],
        source_message_id=player.id,
        memory_id="memory-cracked-lens",
    )
    current_memory = repositories.add_memory(
        save_id=save.id,
        body="The beacon lens is intact.",
        tags=["beacon"],
        source_message_id=narrator.id,
        memory_id="memory-intact-lens",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {
                "actions": [
                    _action(
                        operation="archive",
                        target_type="memory",
                        target_id=stale_memory.id,
                        field_path="*",
                        value=None,
                        reason="This memory contradicts the newer intact-lens memory.",
                        confidence=0.78,
                        evidence_message_ids=[narrator.id],
                    )
                ]
            },
        ]
    )

    result = asyncio.run(
        _service(repositories, provider).propose_guided_cleanup(
            save.id,
            instruction="Remove the stale cracked-lens memory.",
        )
    )

    assert result.queued_suggestions == 1
    assert [memory.id for memory in repositories.list_memories(save.id)] == [
        stale_memory.id,
        current_memory.id,
    ]

    _apply_first_pending_suggestion(repositories, save.id)

    assert [memory.id for memory in repositories.list_memories(save.id)] == [
        current_memory.id
    ]


def test_guided_cleanup_keeps_low_confidence_actions_review_only(
    repositories: PersistenceRepositories,
) -> None:
    save, _player, narrator = _create_save_with_messages(repositories)
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Check the tower bell",
        description="The bell may still be ringing.",
        status="open",
        source_message_id=narrator.id,
        thread_id="thread-tower-bell",
    )
    provider = FakeStructuredCleanupProvider(
        [
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="active_thread",
                        target_id=thread.id,
                        field_path="status",
                        value="resolved",
                        reason="The instruction may refer to this thread.",
                        confidence=0.22,
                        evidence_message_ids=[narrator.id],
                    )
                ]
            },
        ]
    )

    result = asyncio.run(
        _service(repositories, provider).propose_guided_cleanup(
            save.id,
            instruction="Maybe the tower bell thread is done.",
        )
    )

    assert result.queued_suggestions == 1
    assert result.rejected_actions == 0
    assert repositories.get_active_thread(thread.id) == thread
    suggestion = repositories.list_context_update_suggestions(save.id)[0]
    assert suggestion.status == "pending"
    assert suggestion.confidence == 0.22


def test_guided_cleanup_rejects_transcript_mutation_targets(
    repositories: PersistenceRepositories,
) -> None:
    save, player, narrator = _create_save_with_messages(repositories)
    original_messages = repositories.list_messages(save.id)
    provider = FakeStructuredCleanupProvider(
        [
            {
                "actions": [
                    _action(
                        operation="update",
                        target_type="message",
                        target_id=player.id,
                        field_path="body",
                        value="Edited transcript text",
                        reason="The instruction asked for a transcript rewrite.",
                        confidence=0.9,
                        evidence_message_ids=[narrator.id],
                    )
                ]
            },
        ]
    )

    result = asyncio.run(
        _service(repositories, provider).propose_guided_cleanup(
            save.id,
            instruction="Rewrite the first transcript message.",
        )
    )

    assert result.queued_suggestions == 0
    assert result.rejected_actions == 1
    assert repositories.list_context_update_suggestions(save.id) == []
    assert repositories.list_messages(save.id) == original_messages
    audit = repositories.list_context_update_audit(save.id)[0]
    assert audit.operation == "guided_cleanup_rejected"
    assert "Unsupported cleanup target type: message" in audit.reason
    assert "message" not in _actions_target_type_enum(provider.requests[0].schema)


def _service(
    repositories: PersistenceRepositories,
    provider: FakeStructuredCleanupProvider,
) -> ContextCleanupService:
    return ContextCleanupService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )


def _apply_first_pending_suggestion(
    repositories: PersistenceRepositories,
    save_id: str,
) -> None:
    service = WorldDataService(repositories=repositories, active_save_id=save_id)
    model = service.build_model()
    assert model.scenario is not None
    assert model.suggestions
    service.apply_edits(
        WorldDataEdits(
            scenario=WorldDataScenarioEdit(
                title=model.scenario.title,
                premise=model.scenario.premise,
                player_character_name=model.scenario.player_character_name,
                player_role=model.scenario.player_role,
                content_sections=model.scenario.content_sections,
            ),
            suggestions=(replace(model.suggestions[0], action="apply"),),
        )
    )


def _create_save(
    repositories: PersistenceRepositories,
    *,
    title: str = "Night Watch",
) -> SaveRecord:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    return repositories.create_save(scenario_id=scenario.id, title=title)


def _create_save_with_messages(
    repositories: PersistenceRepositories,
) -> tuple[SaveRecord, MessageRecord, MessageRecord]:
    save = _create_save(repositories)
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I inspect the cracked beacon lens.",
        message_id="message-player",
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Ilyra says the east gate was repaired hours ago.",
        provider="fake",
        model="fake-chat",
        message_id="message-narrator",
    )
    return save, player, narrator


def _action(
    *,
    operation: str,
    target_type: str,
    target_id: str,
    field_path: str,
    value: object,
    evidence_message_ids: list[str],
    confidence: float = 0.91,
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "operation": operation,
        "target_type": target_type,
        "target_id": target_id,
        "field_path": field_path,
        "value": value,
        "reason": reason or f"{operation} {target_type}",
        "confidence": confidence,
        "evidence_message_ids": evidence_message_ids,
    }


def _message_ids_from_request_body(body: str) -> list[str]:
    return [line.split()[1] for line in body.splitlines() if line.startswith("- msg-")]


def _actions_evidence_items(schema: dict[str, object]) -> dict[str, object]:
    properties = schema["properties"]
    assert isinstance(properties, dict)
    actions = properties["actions"]
    assert isinstance(actions, dict)
    action_items = actions["items"]
    assert isinstance(action_items, dict)
    action_properties = action_items["properties"]
    assert isinstance(action_properties, dict)
    evidence_message_ids = action_properties["evidence_message_ids"]
    assert isinstance(evidence_message_ids, dict)
    evidence_items = evidence_message_ids["items"]
    assert isinstance(evidence_items, dict)
    return evidence_items


def _actions_target_type_enum(schema: dict[str, object]) -> list[str]:
    properties = schema["properties"]
    assert isinstance(properties, dict)
    actions = properties["actions"]
    assert isinstance(actions, dict)
    action_items = actions["items"]
    assert isinstance(action_items, dict)
    action_properties = action_items["properties"]
    assert isinstance(action_properties, dict)
    target_type = action_properties["target_type"]
    assert isinstance(target_type, dict)
    enum = target_type["enum"]
    assert isinstance(enum, list)
    return [item for item in enum if isinstance(item, str)]


def _cleanup_jobs(repositories: PersistenceRepositories) -> list[JobRecord]:
    return [
        job
        for job in repositories.list_jobs_by_status(("succeeded", "failed"))
        if job.type == "context_cleanup"
    ]
