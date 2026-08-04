from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.memory_consolidation_service import MemoryConsolidationService
from bragi.services.prompt_inspection import PromptInspectionStore
from bragi.services.text_script_policy import (
    SCRIPT_GUARD_MODE_OFF,
    SCRIPT_GUARD_MODE_SETTING,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


class FakeStructuredProvider:
    provider_name = "fake"

    def __init__(self, data: dict[str, object]) -> None:
        self.data = data
        self.requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.requests.append(request)
        return StructuredOutputResponse(
            data=self.data,
            provider="fake",
            model_id=request.model_id,
        )


class FakeToolProvider:
    provider_name = "fake"

    def __init__(self, responses: list[tuple[ProviderToolCall, ...]]) -> None:
        self.responses = responses
        self.structured_requests: list[StructuredOutputRequest] = []
        self.tool_requests: list[ToolCallRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        raise AssertionError("tool-capable consolidation should prefer tools")

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


class ShapeSwitchConsolidationProvider(FakeToolProvider):
    """Tool-capable consolidation provider whose tool calls 404 but structured works."""

    def __init__(
        self,
        *,
        structured_data: dict[str, object] | None = None,
    ) -> None:
        super().__init__(responses=[])
        self.structured_data = structured_data or {"clusters": []}

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model not found",
            status_code=404,
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        return StructuredOutputResponse(
            data=self.structured_data,
            provider=request.provider,
            model_id=request.model_id,
        )


class ShapeFailingConsolidationProvider(ShapeSwitchConsolidationProvider):
    """Tool-capable consolidation provider whose tool and structured calls 404."""

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model not found",
            status_code=404,
        )


class RateLimitedConsolidationProvider(ShapeSwitchConsolidationProvider):
    """Tool-capable consolidation provider that rate-limits but structured works."""

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.RATE_LIMITED,
            "rate limited",
            status_code=429,
        )


class FailingConsolidationFallbackProvider(FakeToolProvider):
    provider_name = "fallback"

    def __init__(self, *, error: ProviderError) -> None:
        super().__init__(responses=[])
        self.error = error

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_requests.append(request)
        raise self.error


def test_consolidation_rewrites_canonical_archives_duplicates_and_unions_sources(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=messages[0].id,
        source_observation_ids=["observation-canonical"],
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara around the beacon.",
        tags=["relationship", "ilyra"],
        importance=0.8,
        source_message_id=messages[1].id,
        source_observation_ids=["observation-duplicate"],
    )
    provider = FakeStructuredProvider(
        {
            "clusters": [
                {
                    "canonical_memory_id": canonical.id,
                    "merged_memory_ids": [duplicate.id],
                    "body": (
                        "Captain Ilyra trusts Mara with the beacon lens and "
                        "its signal rites."
                    ),
                    "tags": ["dossier", "relationship", "character:ilyra"],
                    "importance": 0.92,
                    "confidence": 0.91,
                    "reason": "The two memories describe the same relationship.",
                }
            ]
        }
    )
    store = PromptInspectionStore()
    service = MemoryConsolidationService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        prompt_inspection_store=store,
        inspection_message_id=messages[1].id,
    )

    result = asyncio.run(
        service.consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert result.rewritten_count == 1
    assert result.archived_count == 1
    assert provider.requests[0].schema_name == "memory_consolidation"
    assert provider.requests[0].max_output_tokens == 2048
    assert [entry.kind for entry in store.entries_for_message(messages[1].id)] == [
        "memory_consolidation"
    ]
    assert "Memory consolidation" in (
        store.prompt_for_message(messages[1].id) or ""
    )
    memories = repositories.list_memories(save.id)
    assert [memory.id for memory in memories] == [canonical.id]
    assert memories[0].body == (
        "Captain Ilyra trusts Mara with the beacon lens and its signal rites."
    )
    assert memories[0].tags == ["dossier", "relationship", "character:ilyra"]
    assert memories[0].importance == 0.92
    assert memories[0].source_message_id == messages[0].id
    assert memories[0].source_message_ids == [messages[0].id, messages[1].id]
    assert memories[0].source_observation_ids == [
        "observation-canonical",
        "observation-duplicate",
    ]

    archived = repositories.connection.execute(
        "SELECT archived_at FROM memories WHERE id = ?",
        (duplicate.id,),
    ).fetchone()
    assert archived is not None
    assert archived["archived_at"] is not None
    audit = repositories.list_context_update_audit(save.id)
    assert [(row.operation, row.entity_type, row.entity_id) for row in audit] == [
        ("memory_consolidation_rewritten", "memory", canonical.id),
        ("memory_consolidation_archived", "memory", duplicate.id),
    ]
    assert audit[0].source_message_ids == [messages[0].id, messages[1].id]


def test_consolidation_uses_apply_guard_only_for_repository_writes(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara around the beacon.",
        tags=["relationship", "ilyra"],
        importance=0.8,
        source_message_id=messages[1].id,
    )
    inside_guard = False
    events: list[str] = []

    class GuardAwareProvider(FakeStructuredProvider):
        async def generate_structured_output(
            self,
            request: StructuredOutputRequest,
        ) -> StructuredOutputResponse:
            assert not inside_guard
            events.append("provider")
            return await super().generate_structured_output(request)

    @asynccontextmanager
    async def apply_guard() -> AsyncIterator[None]:
        nonlocal inside_guard
        events.append("enter")
        inside_guard = True
        try:
            yield
        finally:
            inside_guard = False
            events.append("exit")

    original_update_memory = repositories.update_memory
    original_begin_immediate_transaction = (
        repositories.begin_immediate_transaction
    )

    def update_memory_guarded(*args: Any, **kwargs: Any) -> Any:
        assert inside_guard
        events.append("write")
        return original_update_memory(*args, **kwargs)

    def begin_immediate_transaction() -> None:
        events.append(f"begin_immediate:{repositories._transaction_depth}")
        original_begin_immediate_transaction()

    monkeypatch.setattr(repositories, "update_memory", update_memory_guarded)
    monkeypatch.setattr(
        repositories,
        "begin_immediate_transaction",
        begin_immediate_transaction,
    )
    provider = GuardAwareProvider(
        {
            "clusters": [
                {
                    "canonical_memory_id": canonical.id,
                    "merged_memory_ids": [duplicate.id],
                    "body": "Ilyra trusts Mara with the beacon lens.",
                    "tags": ["relationship"],
                    "importance": 0.75,
                    "confidence": 0.91,
                    "reason": "Duplicate memories can be merged.",
                }
            ]
        }
    )
    service = MemoryConsolidationService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    result = asyncio.run(
        service.consolidate_if_needed(
            save.id,
            min_active_memories=1,
            apply_guard=apply_guard,
        )
    )

    assert result.rewritten_count == 1
    assert result.archived_count == 1
    assert events == [
        "provider",
        "enter",
        "begin_immediate:0",
        "write",
        "begin_immediate:1",
        "exit",
    ]


def test_consolidation_rejects_unexpected_generated_script(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara around the beacon.",
        tags=["relationship", "ilyra"],
        importance=0.8,
        source_message_id=messages[1].id,
    )
    provider = FakeStructuredProvider(
        {
            "clusters": [
                {
                    "canonical_memory_id": canonical.id,
                    "merged_memory_ids": [duplicate.id],
                    "body": "伊莉拉信任玛拉守护灯塔透镜。",
                    "tags": ["relationship"],
                    "importance": 0.92,
                    "confidence": 0.91,
                    "reason": "重复记忆。",
                }
            ]
        }
    )

    result = asyncio.run(
        MemoryConsolidationService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-structured",
        ).consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert result.rejected_count == 1
    memories = repositories.list_memories(save.id)
    assert [memory.id for memory in memories] == [canonical.id, duplicate.id]
    assert [memory.body for memory in memories] == [
        "Ilyra trusts Mara with the beacon lens.",
        "Captain Ilyra trusts Mara around the beacon.",
    ]
    assert repositories.list_context_update_audit(save.id) == []


def test_consolidation_prefers_tool_calls_when_enabled(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara around the beacon.",
        tags=["relationship", "ilyra"],
        importance=0.8,
        source_message_id=messages[1].id,
    )
    provider = FakeToolProvider(
        [
            (
                ProviderToolCall(
                    id="call-cluster",
                    name="merge_memory_cluster",
                    arguments_json=json.dumps(
                        {
                            "canonical_memory_id": canonical.id,
                            "merged_memory_ids": [duplicate.id],
                            "body": "Captain Ilyra trusts Mara with the beacon lens.",
                            "tags": ["relationship", "character:ilyra"],
                            "importance": 0.9,
                            "confidence": 0.93,
                            "reason": "The memories describe the same trust beat.",
                        }
                    ),
                ),
            )
        ]
    )

    result = asyncio.run(
        MemoryConsolidationService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
            prefer_tool_calls=True,
        ).consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert result.rewritten_count == 1
    assert result.archived_count == 1
    assert provider.structured_requests == []
    assert [tool.name for tool in provider.tool_requests[0].tools] == [
        "merge_memory_cluster"
    ]


def test_consolidation_tool_calls_switch_to_structured_route_on_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara around the beacon.",
        tags=["relationship", "ilyra"],
        importance=0.8,
        source_message_id=messages[1].id,
    )
    provider = ShapeSwitchConsolidationProvider(
        structured_data={
            "clusters": [
                {
                    "canonical_memory_id": canonical.id,
                    "merged_memory_ids": [duplicate.id],
                    "body": "Captain Ilyra trusts Mara with the beacon lens.",
                    "tags": ["relationship"],
                    "importance": 0.9,
                    "confidence": 0.95,
                    "reason": "The memories describe the same relationship.",
                }
            ]
        }
    )

    result = asyncio.run(
        MemoryConsolidationService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
            prefer_tool_calls=True,
        ).consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert len(provider.tool_requests) == 1
    assert len(provider.structured_requests) == 1
    assert result.rewritten_count == 1
    assert result.archived_count == 1

    jobs = _consolidation_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    result_json = json.loads(jobs[0]["result_json"])
    assert result_json["tool_diagnostics"] == {
        "shape_switch": "structured_output",
        "provider": "fake",
        "model": "fake-tools",
    }


def test_consolidation_tool_calls_keep_error_when_structured_route_also_fails(
    repositories: PersistenceRepositories,
) -> None:
    save, _messages = _save_with_messages(repositories)
    repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=repositories.list_messages(save.id)[0].id,
    )
    provider = ShapeFailingConsolidationProvider()

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            MemoryConsolidationService(
                repositories=repositories,
                provider=provider,
                provider_name="fake",
                model_id="fake-tools",
                prefer_tool_calls=True,
            ).consolidate_if_needed(save.id, min_active_memories=1)
        )

    assert exc_info.value.category == ProviderErrorCategory.MODEL_NOT_FOUND
    assert exc_info.value.fallback_attempted is True
    assert exc_info.value.fallback_provider == "fake"
    assert len(provider.structured_requests) == 1


def test_consolidation_recovers_when_tool_fallback_also_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara around the beacon.",
        tags=["relationship", "ilyra"],
        importance=0.8,
        source_message_id=messages[1].id,
    )
    _configure_consolidation_tool_fallback(repositories)
    primary = ShapeSwitchConsolidationProvider(
        structured_data={
            "clusters": [
                {
                    "canonical_memory_id": canonical.id,
                    "merged_memory_ids": [duplicate.id],
                    "body": "Captain Ilyra trusts Mara with the beacon lens.",
                    "tags": ["relationship"],
                    "importance": 0.9,
                    "confidence": 0.95,
                    "reason": "The memories describe the same relationship.",
                }
            ]
        }
    )
    fallback = FailingConsolidationFallbackProvider(
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "fallback model not found",
            status_code=404,
        )
    )

    result = asyncio.run(
        MemoryConsolidationService(
            repositories=repositories,
            provider=primary,
            provider_name="fake",
            model_id="fake-tools",
            prefer_tool_calls=True,
            providers={
                "fake": cast(Any, primary),
                "fallback": cast(Any, fallback),
            },
        ).consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert len(primary.tool_requests) == 1
    assert len(fallback.tool_requests) == 1
    assert len(primary.structured_requests) == 1
    assert result.rewritten_count == 1


def test_consolidation_recovers_when_tool_fallback_model_missing(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara around the beacon.",
        tags=["relationship", "ilyra"],
        importance=0.8,
        source_message_id=messages[1].id,
    )
    _configure_consolidation_tool_fallback(repositories)
    primary = RateLimitedConsolidationProvider(
        structured_data={
            "clusters": [
                {
                    "canonical_memory_id": canonical.id,
                    "merged_memory_ids": [duplicate.id],
                    "body": "Captain Ilyra trusts Mara with the beacon lens.",
                    "tags": ["relationship"],
                    "importance": 0.9,
                    "confidence": 0.95,
                    "reason": "The memories describe the same relationship.",
                }
            ]
        }
    )
    fallback = FailingConsolidationFallbackProvider(
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "fallback model missing",
            status_code=404,
        )
    )

    result = asyncio.run(
        MemoryConsolidationService(
            repositories=repositories,
            provider=primary,
            provider_name="fake",
            model_id="fake-tools",
            prefer_tool_calls=True,
            providers={
                "fake": cast(Any, primary),
                "fallback": cast(Any, fallback),
            },
        ).consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert len(primary.tool_requests) == 1
    assert len(fallback.tool_requests) == 1
    assert len(primary.structured_requests) == 1
    assert result.rewritten_count == 1


def test_consolidation_recovers_when_tool_fallback_rate_limited(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara around the beacon.",
        tags=["relationship", "ilyra"],
        importance=0.8,
        source_message_id=messages[1].id,
    )
    _configure_consolidation_tool_fallback(repositories)
    primary = ShapeSwitchConsolidationProvider(
        structured_data={
            "clusters": [
                {
                    "canonical_memory_id": canonical.id,
                    "merged_memory_ids": [duplicate.id],
                    "body": "Captain Ilyra trusts Mara with the beacon lens.",
                    "tags": ["relationship"],
                    "importance": 0.9,
                    "confidence": 0.95,
                    "reason": "The memories describe the same relationship.",
                }
            ]
        }
    )
    fallback = FailingConsolidationFallbackProvider(
        error=ProviderError(
            ProviderErrorCategory.RATE_LIMITED,
            "rate limited",
            status_code=429,
        )
    )

    result = asyncio.run(
        MemoryConsolidationService(
            repositories=repositories,
            provider=primary,
            provider_name="fake",
            model_id="fake-tools",
            prefer_tool_calls=True,
            providers={
                "fake": cast(Any, primary),
                "fallback": cast(Any, fallback),
            },
        ).consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert len(primary.tool_requests) == 1
    assert len(fallback.tool_requests) == 1
    assert len(primary.structured_requests) == 1
    assert result.rewritten_count == 1


def test_consolidation_keeps_fallback_result_when_tool_fallback_succeeds(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara around the beacon.",
        tags=["relationship", "ilyra"],
        importance=0.8,
        source_message_id=messages[1].id,
    )
    _configure_consolidation_tool_fallback(repositories)
    primary = ShapeSwitchConsolidationProvider()
    fallback = FakeToolProvider(
        responses=[
            (
                ProviderToolCall(
                    id="merge-call",
                    name="merge_memory_cluster",
                    arguments_json=json.dumps(
                        {
                            "canonical_memory_id": canonical.id,
                            "merged_memory_ids": [duplicate.id],
                            "body": "Captain Ilyra trusts Mara with the beacon lens.",
                            "tags": ["relationship"],
                            "importance": 0.9,
                            "confidence": 0.95,
                            "reason": "The memories describe the same relationship.",
                        }
                    ),
                ),
            )
        ]
    )

    result = asyncio.run(
        MemoryConsolidationService(
            repositories=repositories,
            provider=primary,
            provider_name="fake",
            model_id="fake-tools",
            prefer_tool_calls=True,
            providers={
                "fake": cast(Any, primary),
                "fallback": cast(Any, fallback),
            },
        ).consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert len(primary.tool_requests) == 1
    assert len(fallback.tool_requests) == 1
    assert primary.structured_requests == []
    assert result.rewritten_count == 1


def test_consolidation_tool_calls_retry_duplicate_merged_ids(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Mara remembers Ilyra's old warning.",
        tags=["ilyra"],
        importance=0.4,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Ilyra warned Mara about the old stairs.",
        tags=["ilyra"],
        importance=0.4,
        source_message_id=messages[1].id,
    )
    provider = FakeToolProvider(
        [
            (
                ProviderToolCall(
                    id="call-bad",
                    name="merge_memory_cluster",
                    arguments_json=json.dumps(
                        {
                            "canonical_memory_id": canonical.id,
                            "merged_memory_ids": [duplicate.id, duplicate.id],
                            "body": "Bad duplicate ids.",
                            "tags": ["ilyra"],
                            "importance": 0.7,
                            "confidence": 0.95,
                            "reason": "Bad cluster.",
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-good",
                    name="merge_memory_cluster",
                    arguments_json=json.dumps(
                        {
                            "canonical_memory_id": canonical.id,
                            "merged_memory_ids": [duplicate.id],
                            "body": "Ilyra warned Mara about the old stairs.",
                            "tags": ["ilyra"],
                            "importance": 0.7,
                            "confidence": 0.95,
                            "reason": "The memories describe the same warning.",
                        }
                    ),
                ),
            ),
        ]
    )

    result = asyncio.run(
        MemoryConsolidationService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
            prefer_tool_calls=True,
        ).consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert result.rewritten_count == 1
    assert len(provider.tool_requests) == 2
    retry_body = provider.tool_requests[1].messages[-1].body
    assert "merged_memory_ids must be unique" in retry_body


def test_consolidation_tool_calls_honor_save_script_guard_mode(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SCRIPT_GUARD_MODE_SETTING,
        value=SCRIPT_GUARD_MODE_OFF,
    )
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Ilyra trusts Mara with the beacon lens.",
        tags=["relationship"],
        importance=0.7,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara around the beacon.",
        tags=["relationship", "ilyra"],
        importance=0.8,
        source_message_id=messages[1].id,
    )
    provider = FakeToolProvider(
        [
            (
                ProviderToolCall(
                    id="call-cluster",
                    name="merge_memory_cluster",
                    arguments_json=json.dumps(
                        {
                            "canonical_memory_id": canonical.id,
                            "merged_memory_ids": [duplicate.id],
                            "body": "伊莉拉信任玛拉守护灯塔透镜。",
                            "tags": ["relationship"],
                            "importance": 0.9,
                            "confidence": 0.93,
                            "reason": "重复记忆。",
                        }
                    ),
                ),
            )
        ]
    )

    result = asyncio.run(
        MemoryConsolidationService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
            prefer_tool_calls=True,
        ).consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert result.rewritten_count == 1
    assert len(provider.tool_requests) == 1
    memories = {memory.id: memory for memory in repositories.list_memories(save.id)}
    assert memories[canonical.id].body == "伊莉拉信任玛拉守护灯塔透镜。"


def test_consolidation_tool_calls_retry_missing_and_self_referential_memory_ids(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Mara remembers Ilyra's old warning.",
        tags=["ilyra"],
        importance=0.4,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Ilyra warned Mara about the old stairs.",
        tags=["ilyra"],
        importance=0.4,
        source_message_id=messages[1].id,
    )
    provider = FakeToolProvider(
        [
            (
                ProviderToolCall(
                    id="call-missing",
                    name="merge_memory_cluster",
                    arguments_json=json.dumps(
                        {
                            "canonical_memory_id": "memory-missing",
                            "merged_memory_ids": [duplicate.id],
                            "body": "Bad missing canonical.",
                            "tags": ["ilyra"],
                            "importance": 0.7,
                            "confidence": 0.95,
                            "reason": "Bad cluster.",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="call-self",
                    name="merge_memory_cluster",
                    arguments_json=json.dumps(
                        {
                            "canonical_memory_id": canonical.id,
                            "merged_memory_ids": [canonical.id],
                            "body": "Bad self merge.",
                            "tags": ["ilyra"],
                            "importance": 0.7,
                            "confidence": 0.95,
                            "reason": "Bad self-referential cluster.",
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-good",
                    name="merge_memory_cluster",
                    arguments_json=json.dumps(
                        {
                            "canonical_memory_id": canonical.id,
                            "merged_memory_ids": [duplicate.id],
                            "body": "Ilyra warned Mara about the old stairs.",
                            "tags": ["ilyra"],
                            "importance": 0.7,
                            "confidence": 0.95,
                            "reason": "The memories describe the same warning.",
                        }
                    ),
                ),
            ),
        ]
    )

    result = asyncio.run(
        MemoryConsolidationService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-tools",
            prefer_tool_calls=True,
        ).consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert result.rewritten_count == 1
    assert result.archived_count == 1
    assert [memory.id for memory in repositories.list_memories(save.id)] == [
        canonical.id
    ]
    assert len(provider.tool_requests) == 2
    feedback = "\n".join(message.body for message in provider.tool_requests[1].messages)
    assert "canonical_memory_id must be one of" in feedback
    assert "canonical_memory_id must not appear in merged_memory_ids" in feedback


def test_consolidation_rejects_invalid_low_confidence_and_noop_clusters(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    canonical = repositories.add_memory(
        save_id=save.id,
        body="Mara remembers Ilyra's old warning.",
        tags=["ilyra"],
        importance=0.4,
        source_message_id=messages[0].id,
    )
    duplicate = repositories.add_memory(
        save_id=save.id,
        body="Ilyra warned Mara about the old stairs.",
        tags=["ilyra"],
        importance=0.4,
        source_message_id=messages[1].id,
    )
    provider = FakeStructuredProvider(
        {
            "clusters": [
                {
                    "canonical_memory_id": canonical.id,
                    "merged_memory_ids": [duplicate.id],
                    "body": "This low-confidence rewrite should be ignored.",
                    "tags": ["ilyra"],
                    "importance": 0.7,
                    "confidence": 0.84,
                    "reason": "Too uncertain.",
                },
                {
                    "canonical_memory_id": "missing-memory",
                    "merged_memory_ids": [duplicate.id],
                    "body": "Invalid target should be ignored.",
                    "tags": ["ilyra"],
                    "importance": 0.7,
                    "confidence": 0.95,
                    "reason": "Invalid id.",
                },
                {
                    "canonical_memory_id": canonical.id,
                    "merged_memory_ids": [duplicate.id],
                    "body": canonical.body,
                    "tags": canonical.tags,
                    "importance": canonical.importance,
                    "confidence": 0.95,
                    "reason": "No actual rewrite.",
                },
            ]
        }
    )
    service = MemoryConsolidationService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    result = asyncio.run(
        service.consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert result.rewritten_count == 0
    assert result.archived_count == 0
    assert repositories.list_memories(save.id) == [canonical, duplicate]
    assert repositories.list_context_update_audit(save.id) == []


def test_consolidation_skips_when_memory_count_is_below_threshold(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    repositories.add_memory(
        save_id=save.id,
        body="Mara trusts Ilyra.",
        tags=["relationship"],
        source_message_id=messages[0].id,
    )
    provider = FakeStructuredProvider({"clusters": []})
    service = MemoryConsolidationService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    result = asyncio.run(service.consolidate_if_needed(save.id))

    assert result.skipped_reason == "active memory count below threshold"
    assert provider.requests == []


def test_consolidation_batches_large_memory_sets(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_messages(repositories)
    for index in range(85):
        repositories.add_memory(
            save_id=save.id,
            body=f"Scene detail {index}.",
            tags=["detail"],
            source_message_id=messages[index % 2].id,
            memory_id=f"memory-{index:02d}",
        )
    provider = FakeStructuredProvider({"clusters": []})
    service = MemoryConsolidationService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    result = asyncio.run(
        service.consolidate_if_needed(save.id, min_active_memories=1)
    )

    assert result.batch_count == 2
    assert result.completed_batch_count == 2
    assert len(provider.requests) == 2
    first_schema_ids = _memory_id_enum(provider.requests[0])
    second_schema_ids = _memory_id_enum(provider.requests[1])
    assert set(first_schema_ids) == {
        f"memory-{index:02d}" for index in range(80)
    }
    assert set(second_schema_ids) == {
        f"memory-{index:02d}" for index in range(80, 85)
    }


def _memory_id_enum(request: StructuredOutputRequest) -> list[str]:
    properties = cast(dict[str, Any], request.schema["properties"])
    clusters = cast(dict[str, Any], properties["clusters"])
    items = cast(dict[str, Any], clusters["items"])
    item_properties = cast(dict[str, Any], items["properties"])
    memory_id_schema = cast(dict[str, Any], item_properties["canonical_memory_id"])
    return cast(list[str], memory_id_schema["enum"])


def _save_with_messages(
    repositories: PersistenceRepositories,
) -> tuple[Any, tuple[Any, ...]]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    messages = (
        repositories.append_message(
            save_id=save.id,
            role="player",
            body="I ask Ilyra about the beacon.",
        ),
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            body="Ilyra answers with a guarded warning.",
            provider="fake",
            model="fake-chat",
        ),
    )
    return save, messages


def _configure_consolidation_tool_fallback(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting("tool_call_fallback_enabled", True)
    repositories.set_model_preference(
        task="tool_call_fallback",
        provider="fallback",
        model_id="fallback-tools",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-tools",
        display_name="Fake Tools",
        capabilities=["tool_calling", "structured_output"],
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-tools",
        display_name="Fallback Tools",
        capabilities=["tool_calling"],
    )


def _consolidation_jobs(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[sqlite3.Row]:
    return list(
        repositories.connection.execute(
            """
            SELECT status, result_json, error
            FROM jobs
            WHERE save_id = ? AND type = 'memory_consolidation'
            ORDER BY created_at
            """,
            (save_id,),
        )
    )
