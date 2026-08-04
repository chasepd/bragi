from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.state_pruning_service import StatePruningService


class RecordingStructuredPruningProvider:
    provider_name = "fake"

    def __init__(
        self,
        response_data: dict[str, object] | None = None,
        *,
        response_data_by_request: list[dict[str, object]] | None = None,
        fail_on_request: int | None = None,
        on_generate: Callable[[StructuredOutputRequest], None] | None = None,
    ) -> None:
        self.response_data = response_data or {"archives": []}
        self.response_data_by_request = list(response_data_by_request or [])
        self.fail_on_request = fail_on_request
        self.on_generate = on_generate
        self.chat_requests: list[ChatRequest] = []
        self.structured_output_requests: list[StructuredOutputRequest] = []

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
                model_id="fake-pruner",
                display_name="Fake Pruner",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("state pruning must not call normal chat")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        if self.fail_on_request == len(self.structured_output_requests):
            raise TimeoutError("The read operation timed out")
        if self.on_generate is not None:
            self.on_generate(request)
        data = (
            self.response_data_by_request.pop(0)
            if self.response_data_by_request
            else self.response_data
        )
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("state pruning must not request image generation")


class RecordingToolPruningProvider:
    provider_name = "fake"

    def __init__(self, responses: list[tuple[ProviderToolCall, ...]]) -> None:
        self.responses = responses
        self.chat_requests: list[ChatRequest] = []
        self.structured_output_requests: list[StructuredOutputRequest] = []
        self.tool_call_requests: list[ToolCallRequest] = []

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
                model_id="fake-pruner",
                display_name="Fake Pruner",
                capabilities=frozenset({ProviderCapability.TOOL_CALLING}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("state pruning must not call normal chat")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        raise AssertionError("tool-capable state pruning should prefer tool calls")

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected tool-call request")
        return ToolCallResponse(
            tool_calls=self.responses.pop(0),
            body="",
            provider=request.provider,
            model_id=request.model_id,
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("state pruning must not request image generation")


class ShapeSwitchToolPruningProvider(RecordingToolPruningProvider):
    """Tool-capable pruner whose tool calls 404 but structured output works."""

    def __init__(
        self,
        *,
        structured_data: dict[str, object] | None = None,
    ) -> None:
        super().__init__(responses=[])
        self.structured_data = structured_data or {"archives": []}

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model not found",
            status_code=404,
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data=self.structured_data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )


class ShapeFailingToolPruningProvider(ShapeSwitchToolPruningProvider):
    """Tool-capable pruner whose tool and structured calls both 404."""

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model not found",
            status_code=404,
        )


class RateLimitedToolPruningProvider(ShapeSwitchToolPruningProvider):
    """Tool-capable pruner whose tool calls rate-limit but structured works."""

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        raise ProviderError(
            ProviderErrorCategory.RATE_LIMITED,
            "rate limited",
            status_code=429,
        )


class FailingToolPruningFallbackProvider(RecordingToolPruningProvider):
    provider_name = "fallback"

    def __init__(self, *, error: ProviderError) -> None:
        super().__init__(responses=[])
        self.error = error

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        raise self.error


class SlowStructuredPruningProvider(RecordingStructuredPruningProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        self.started.set()
        await asyncio.sleep(60)
        raise AssertionError("cancelled pruning should not complete")


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_state_pruning_archives_validated_active_rows_and_records_counts(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    kept_state = repositories.upsert_world_state(
        save_id=save.id,
        key="npc.elian.status",
        value={"status": "watching"},
        category="npc",
    )
    provider = RecordingStructuredPruningProvider(
        {
            "archives": [
                {
                    "world_state_id": stale_state.id,
                    "key": stale_state.key,
                    "reason": "The alarm was disabled and is no longer useful.",
                },
                {
                    "world_state_id": kept_state.id,
                    "key": "npc.elian.wrong_key",
                    "reason": "The key does not match the active row.",
                },
            ]
        }
    )
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.prune(save_id=save.id))

    assert provider.chat_requests == []
    assert len(provider.structured_output_requests) == 1
    request = provider.structured_output_requests[0]
    assert request.schema_name == "state_pruning_selection"
    request_body = "\n".join(message.body for message in request.messages)
    assert "starting_scene" not in request_body
    assert "The beacon gutters in the tower." not in request_body
    assert request.schema["properties"]["archives"]["items"]["properties"][
        "world_state_id"
    ]["enum"] == [kept_state.id, stale_state.id]
    assert [fact.world_state_id for fact in result.proposed] == [
        stale_state.id,
        kept_state.id,
    ]
    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]
    assert [fact.world_state_id for fact in result.rejected] == [kept_state.id]
    active_rows = [
        (state.id, state.key) for state in repositories.list_world_state(save.id)
    ]
    assert active_rows == [
        (kept_state.id, kept_state.key),
    ]
    assert _archived_at(repositories, stale_state.id) is not None

    jobs = _state_pruning_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["error"] is None
    result_json = json.loads(jobs[0]["result_json"])
    assert result_json["proposed_count"] == 2
    assert result_json["archived_count"] == 1
    assert result_json["rejected_count"] == 1
    assert result_json["review_only"] is False
    assert [proposal["archived"] for proposal in result_json["proposals"]] == [
        True,
        False,
    ]
    assert [proposal["rejected"] for proposal in result_json["proposals"]] == [
        False,
        True,
    ]
    assert result_json["active_state_count"] == 2
    assert result_json["batch_count"] == 1
    assert result_json["completed_batch_count"] == 1


def test_state_pruning_apply_guard_does_not_wrap_provider_calls(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    lock_depth = 0
    events: list[str] = []

    def on_generate(_request: StructuredOutputRequest) -> None:
        assert lock_depth == 0
        events.append("provider")

    provider = RecordingStructuredPruningProvider(
        {
            "archives": [
                {
                    "world_state_id": stale_state.id,
                    "key": stale_state.key,
                    "reason": "The old alarm is no longer relevant.",
                }
            ]
        },
        on_generate=on_generate,
    )
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
    )
    original_begin_transaction = repositories.begin_transaction
    original_update_job = repositories.update_job

    class InstrumentedApplyGuard:
        async def __aenter__(self) -> None:
            nonlocal lock_depth
            assert lock_depth == 0
            events.append("lock:enter")
            lock_depth += 1

        async def __aexit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            nonlocal lock_depth
            assert lock_depth == 1
            events.append("lock:exit")
            lock_depth -= 1

    def apply_guard() -> InstrumentedApplyGuard:
        events.append("lock:requested")
        return InstrumentedApplyGuard()

    def begin_transaction() -> None:
        assert lock_depth == 1
        events.append("apply:begin_transaction")
        original_begin_transaction()

    def update_job(
        job_id: str,
        *,
        status: str,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> object:
        if status == "succeeded":
            assert lock_depth == 1
            events.append("job:succeeded")
        return original_update_job(
            job_id,
            status=status,
            result=result,
            error=error,
        )

    monkeypatch.setattr(repositories, "begin_transaction", begin_transaction)
    monkeypatch.setattr(repositories, "update_job", update_job)

    result = asyncio.run(
        service.prune(
            save_id=save.id,
            apply_guard=apply_guard,
        )
    )

    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]
    assert events == [
        "lock:requested",
        "lock:enter",
        "lock:exit",
        "provider",
        "lock:requested",
        "lock:enter",
        "apply:begin_transaction",
        "lock:exit",
        "lock:requested",
        "lock:enter",
        "job:succeeded",
        "lock:exit",
    ]
    assert lock_depth == 0


def test_state_pruning_switches_to_structured_route_on_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    provider = ShapeSwitchToolPruningProvider(
        structured_data={
            "archives": [
                {
                    "world_state_id": stale_state.id,
                    "key": stale_state.key,
                    "reason": "The alarm was disabled and superseded.",
                }
            ]
        }
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": provider},
        ).prune(save_id=save.id)
    )

    assert len(provider.tool_call_requests) == 1
    assert len(provider.structured_output_requests) == 1
    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]
    assert _archived_at(repositories, stale_state.id) is not None

    jobs = _state_pruning_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    result_json = json.loads(jobs[0]["result_json"])
    assert result_json["tool_diagnostics"] == {
        "shape_switch": "structured_output",
        "provider": "fake",
        "model": "fake-pruner",
    }


def test_state_pruning_keeps_error_when_structured_route_also_fails(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    provider = ShapeFailingToolPruningProvider()

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            StatePruningService(
                repositories=repositories,
                providers={"fake": provider},
            ).prune(save_id=save.id)
        )

    assert exc_info.value.category == ProviderErrorCategory.MODEL_NOT_FOUND
    assert exc_info.value.fallback_attempted is True
    assert exc_info.value.fallback_provider == "fake"
    assert len(provider.structured_output_requests) == 1


def test_state_pruning_recovers_when_tool_fallback_also_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    _configure_state_pruning_tool_fallback(repositories)
    primary = ShapeSwitchToolPruningProvider(
        structured_data={
            "archives": [
                {
                    "world_state_id": stale_state.id,
                    "key": stale_state.key,
                    "reason": "The alarm was disabled and superseded.",
                }
            ]
        }
    )
    fallback = FailingToolPruningFallbackProvider(
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "fallback model not found",
            status_code=404,
        )
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": primary, "fallback": fallback},
        ).prune(save_id=save.id)
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert len(primary.structured_output_requests) == 1
    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]


def test_state_pruning_recovers_when_tool_fallback_model_missing(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    _configure_state_pruning_tool_fallback(repositories)
    primary = RateLimitedToolPruningProvider(
        structured_data={
            "archives": [
                {
                    "world_state_id": stale_state.id,
                    "key": stale_state.key,
                    "reason": "The alarm was disabled and superseded.",
                }
            ]
        }
    )
    fallback = FailingToolPruningFallbackProvider(
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "fallback model missing",
            status_code=404,
        )
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": primary, "fallback": fallback},
        ).prune(save_id=save.id)
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert len(primary.structured_output_requests) == 1
    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]


def test_state_pruning_recovers_when_tool_fallback_rate_limited(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    _configure_state_pruning_tool_fallback(repositories)
    primary = ShapeSwitchToolPruningProvider(
        structured_data={
            "archives": [
                {
                    "world_state_id": stale_state.id,
                    "key": stale_state.key,
                    "reason": "The alarm was disabled and superseded.",
                }
            ]
        }
    )
    fallback = FailingToolPruningFallbackProvider(
        error=ProviderError(
            ProviderErrorCategory.RATE_LIMITED,
            "rate limited",
            status_code=429,
        )
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": primary, "fallback": fallback},
        ).prune(save_id=save.id)
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert len(primary.structured_output_requests) == 1
    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]


def test_state_pruning_keeps_fallback_result_when_tool_fallback_succeeds(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[
            ProviderCapability.TOOL_CALLING.value,
            ProviderCapability.STRUCTURED_OUTPUT.value,
        ],
    )
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    _configure_state_pruning_tool_fallback(repositories)
    primary = ShapeSwitchToolPruningProvider()
    fallback = RecordingToolPruningProvider(
        responses=[
            (
                ProviderToolCall(
                    id="prune-call",
                    name="archive_world_state_fact",
                    arguments_json=json.dumps(
                        {
                            "world_state_id": stale_state.id,
                            "key": stale_state.key,
                            "reason": "The alarm was disabled and superseded.",
                        }
                    ),
                ),
            )
        ]
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": primary, "fallback": fallback},
        ).prune(save_id=save.id)
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert primary.structured_output_requests == []
    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]


def test_state_pruning_prefers_tool_calls_for_tool_capable_model(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    provider = RecordingToolPruningProvider(
        [
            (
                ProviderToolCall(
                    id="call-archive",
                    name="archive_world_state_fact",
                    arguments_json=json.dumps(
                        {
                            "world_state_id": stale_state.id,
                            "key": stale_state.key,
                            "reason": "The alarm was disabled and superseded.",
                        }
                    ),
                ),
            )
        ]
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": provider},
        ).prune(save_id=save.id)
    )

    assert provider.structured_output_requests == []
    assert len(provider.tool_call_requests) == 1
    assert [tool.name for tool in provider.tool_call_requests[0].tools] == [
        "archive_world_state_fact"
    ]
    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]
    assert _archived_at(repositories, stale_state.id) is not None


def test_state_pruning_tool_call_feedback_retries_invalid_key(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    provider = RecordingToolPruningProvider(
        [
            (
                ProviderToolCall(
                    id="call-bad",
                    name="archive_world_state_fact",
                    arguments_json=json.dumps(
                        {
                            "world_state_id": stale_state.id,
                            "key": "scene.wrong_key",
                            "reason": "Bad key.",
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-good",
                    name="archive_world_state_fact",
                    arguments_json=json.dumps(
                        {
                            "world_state_id": stale_state.id,
                            "key": stale_state.key,
                            "reason": "The alarm was disabled and superseded.",
                        }
                    ),
                ),
            ),
        ]
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": provider},
        ).prune(save_id=save.id)
    )

    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]
    assert len(provider.tool_call_requests) == 2
    assert "scene.wrong_key" in provider.tool_call_requests[1].messages[-1].body


def test_state_pruning_tool_call_feedback_retries_invalid_world_state_id(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    provider = RecordingToolPruningProvider(
        [
            (
                ProviderToolCall(
                    id="call-bad-id",
                    name="archive_world_state_fact",
                    arguments_json=json.dumps(
                        {
                            "world_state_id": "world-state-missing",
                            "key": stale_state.key,
                            "reason": "Bad id.",
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-good-id",
                    name="archive_world_state_fact",
                    arguments_json=json.dumps(
                        {
                            "world_state_id": stale_state.id,
                            "key": stale_state.key,
                            "reason": "The alarm was disabled and superseded.",
                        }
                    ),
                ),
            ),
        ]
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": provider},
        ).prune(save_id=save.id)
    )

    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]
    assert _archived_at(repositories, stale_state.id) is not None
    assert len(provider.tool_call_requests) == 2
    assert (
        "world_state_id must be one of"
        in provider.tool_call_requests[1].messages[-1].body
    )


def test_state_pruning_batches_large_active_state_payloads(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    rows = [
        repositories.upsert_world_state(
            save_id=save.id,
            key=f"scene.detail_{index:02d}",
            value={"index": index},
            category="scene",
        )
        for index in range(23)
    ]
    provider = RecordingStructuredPruningProvider()
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
        state_batch_size=7,
    )

    result = asyncio.run(service.prune(save_id=save.id))

    assert result.proposed == ()
    assert len(provider.structured_output_requests) == 4
    batch_sizes = [
        len(
            request.schema["properties"]["archives"]["items"]["properties"][
                "world_state_id"
            ]["enum"]
        )
        for request in provider.structured_output_requests
    ]
    assert batch_sizes == [7, 7, 7, 2]
    first_body = provider.structured_output_requests[0].messages[-1].body
    assert rows[0].id in first_body
    assert rows[6].id in first_body
    assert rows[7].id not in first_body

    jobs = _state_pruning_jobs(repositories, save.id)
    result_json = json.loads(jobs[0]["result_json"])
    assert result_json["active_state_count"] == 23
    assert result_json["batch_count"] == 4
    assert result_json["completed_batch_count"] == 4
    assert result_json["batch_size"] == 7


def test_state_pruning_failed_later_batch_keeps_prior_batch_progress(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    rows = [
        repositories.upsert_world_state(
            save_id=save.id,
            key=f"scene.expired_{index}",
            value={"index": index},
            category="scene",
        )
        for index in range(4)
    ]
    provider = RecordingStructuredPruningProvider(
        response_data_by_request=[
            {
                "archives": [
                    {
                        "world_state_id": rows[0].id,
                        "key": rows[0].key,
                        "reason": "The stale scene fact is no longer useful.",
                    }
                ]
            }
        ],
        fail_on_request=2,
    )
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
        state_batch_size=2,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(service.prune(save_id=save.id))

    assert _archived_at(repositories, rows[0].id) is not None
    assert _archived_at(repositories, rows[1].id) is None
    jobs = _state_pruning_jobs(repositories, save.id)
    assert jobs[0]["status"] == "failed"
    assert "timed out" in jobs[0]["error"]
    result_json = json.loads(jobs[0]["result_json"])
    assert result_json["active_state_count"] == 4
    assert result_json["batch_count"] == 2
    assert result_json["completed_batch_count"] == 1
    assert result_json["failed_batch_index"] == 1
    assert result_json["archived_count"] == 1


def test_state_pruning_rejects_high_value_fact_without_contradiction_reason(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    promise_state = repositories.upsert_world_state(
        save_id=save.id,
        key="promise.ilyra.east_stair",
        value={"status": "open"},
        category="promise",
    )
    provider = RecordingStructuredPruningProvider(
        {
            "archives": [
                {
                    "world_state_id": promise_state.id,
                    "key": promise_state.key,
                    "reason": "This old promise has not appeared recently.",
                }
            ]
        }
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": provider},
        ).prune(save_id=save.id)
    )

    assert [fact.world_state_id for fact in result.proposed] == [promise_state.id]
    assert result.archived == ()
    assert [fact.world_state_id for fact in result.rejected] == [promise_state.id]
    assert _archived_at(repositories, promise_state.id) is None


def test_state_pruning_rejects_open_thread_fact_without_contradiction_reason(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    open_thread_state = repositories.upsert_world_state(
        save_id=save.id,
        key="interaction.open_threads",
        value={
            "upcoming_weekend_plans": (
                "Jade has already told her grandmother about bringing the "
                "player character."
            )
        },
        category="open_threads",
    )
    provider = RecordingStructuredPruningProvider(
        {
            "archives": [
                {
                    "world_state_id": open_thread_state.id,
                    "key": open_thread_state.key,
                    "reason": "This thread seems stale after the latest exchange.",
                }
            ]
        }
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": provider},
        ).prune(save_id=save.id)
    )

    assert [fact.world_state_id for fact in result.proposed] == [
        open_thread_state.id
    ]
    assert result.archived == ()
    assert [fact.world_state_id for fact in result.rejected] == [
        open_thread_state.id
    ]
    assert _archived_at(repositories, open_thread_state.id) is None


def test_state_pruning_rejects_negated_contradiction_reason(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    promise_state = repositories.upsert_world_state(
        save_id=save.id,
        key="promise.ilyra.east_stair",
        value={"status": "open"},
        category="promise",
    )
    provider = RecordingStructuredPruningProvider(
        {
            "archives": [
                {
                    "world_state_id": promise_state.id,
                    "key": promise_state.key,
                    "reason": "This does not contradict current facts.",
                }
            ]
        }
    )

    result = asyncio.run(
        StatePruningService(
            repositories=repositories,
            providers={"fake": provider},
        ).prune(save_id=save.id)
    )

    assert result.archived == ()
    assert [fact.world_state_id for fact in result.rejected] == [promise_state.id]
    assert _archived_at(repositories, promise_state.id) is None


def test_state_pruning_rejects_stale_proposal_when_row_changes_after_selection(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )

    def update_state_after_provider_sees_snapshot(
        request: StructuredOutputRequest,
    ) -> None:
        assert request.schema_name == "state_pruning_selection"
        repositories.upsert_world_state(
            save_id=save.id,
            key=stale_state.key,
            value={"status": "fresh"},
            category="scene",
        )

    provider = RecordingStructuredPruningProvider(
        {
            "archives": [
                {
                    "world_state_id": stale_state.id,
                    "key": stale_state.key,
                    "reason": "The alarm looked disabled in the pruning snapshot.",
                }
            ]
        },
        on_generate=update_state_after_provider_sees_snapshot,
    )
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.prune(save_id=save.id))

    assert [fact.world_state_id for fact in result.proposed] == [stale_state.id]
    assert result.archived == ()
    assert [fact.world_state_id for fact in result.rejected] == [stale_state.id]
    active_rows = repositories.list_world_state(save.id)
    assert [(state.id, state.key, state.value) for state in active_rows] == [
        (
            stale_state.id,
            stale_state.key,
            {"status": "fresh"},
        )
    ]
    assert _archived_at(repositories, stale_state.id) is None

    jobs = _state_pruning_jobs(repositories, save.id)
    assert len(jobs) == 1
    result_json = json.loads(jobs[0]["result_json"])
    assert result_json["proposed_count"] == 1
    assert result_json["archived_count"] == 0
    assert result_json["rejected_count"] == 1
    assert [proposal["archived"] for proposal in result_json["proposals"]] == [False]
    assert [proposal["rejected"] for proposal in result_json["proposals"]] == [True]


def test_state_pruning_job_json_marks_duplicate_proposal_rejected(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    provider = RecordingStructuredPruningProvider(
        {
            "archives": [
                {
                    "world_state_id": stale_state.id,
                    "key": stale_state.key,
                    "reason": "The alarm was disabled and is no longer useful.",
                },
                {
                    "world_state_id": stale_state.id,
                    "key": stale_state.key,
                    "reason": "Duplicate selection for the same obsolete row.",
                },
            ]
        }
    )
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.prune(save_id=save.id))

    assert [fact.world_state_id for fact in result.proposed] == [
        stale_state.id,
        stale_state.id,
    ]
    assert [fact.world_state_id for fact in result.archived] == [stale_state.id]
    assert [fact.world_state_id for fact in result.rejected] == [stale_state.id]
    assert repositories.list_world_state(save.id) == []
    assert _archived_at(repositories, stale_state.id) is not None

    jobs = _state_pruning_jobs(repositories, save.id)
    assert len(jobs) == 1
    result_json = json.loads(jobs[0]["result_json"])
    assert result_json["proposed_count"] == 2
    assert result_json["archived_count"] == 1
    assert result_json["rejected_count"] == 1
    assert [proposal["archived"] for proposal in result_json["proposals"]] == [
        True,
        False,
    ]
    assert [proposal["rejected"] for proposal in result_json["proposals"]] == [
        False,
        True,
    ]


def test_state_pruning_review_only_proposes_without_archiving(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    stale_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled"},
        category="scene",
    )
    provider = RecordingStructuredPruningProvider(
        {
            "archives": [
                {
                    "world_state_id": stale_state.id,
                    "key": stale_state.key,
                    "reason": "Review this stale fact before archiving it.",
                }
            ]
        }
    )
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.prune(save_id=save.id, review_only=True))

    assert [fact.world_state_id for fact in result.proposed] == [stale_state.id]
    assert result.archived == ()
    assert result.rejected == ()
    assert repositories.list_world_state(save.id)[0].id == stale_state.id
    assert _archived_at(repositories, stale_state.id) is None

    jobs = _state_pruning_jobs(repositories, save.id)
    result_json = json.loads(jobs[0]["result_json"])
    assert jobs[0]["status"] == "succeeded"
    assert result_json["proposed_count"] == 1
    assert result_json["archived_count"] == 0
    assert result_json["rejected_count"] == 0
    assert result_json["review_only"] is True


def test_state_pruning_rejects_unknown_mismatched_and_other_save_rows(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    other_save = _create_save(repositories, title="Elsewhere")
    active_state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.current_alarm",
        value={"status": "active"},
        category="scene",
    )
    other_save_state = repositories.upsert_world_state(
        save_id=other_save.id,
        key="scene.old_alarm",
        value={"status": "disabled in another save"},
        category="scene",
    )
    provider = RecordingStructuredPruningProvider(
        {
            "archives": [
                {
                    "world_state_id": "world-state-that-was-not-offered",
                    "key": active_state.key,
                    "reason": "Unknown IDs are rejected.",
                },
                {
                    "world_state_id": active_state.id,
                    "key": "scene.mismatched_key",
                    "reason": "Mismatched keys are rejected.",
                },
                {
                    "world_state_id": other_save_state.id,
                    "key": other_save_state.key,
                    "reason": "Rows from another save are not candidates.",
                },
            ]
        }
    )
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.prune(save_id=save.id))

    assert result.archived == ()
    assert [fact.world_state_id for fact in result.rejected] == [
        "world-state-that-was-not-offered",
        active_state.id,
        other_save_state.id,
    ]
    active_rows = [
        (state.id, state.key) for state in repositories.list_world_state(save.id)
    ]
    assert active_rows == [
        (active_state.id, active_state.key),
    ]
    assert [state.id for state in repositories.list_world_state(other_save.id)] == [
        other_save_state.id,
    ]
    assert _archived_at(repositories, active_state.id) is None
    assert _archived_at(repositories, other_save_state.id) is None

    jobs = _state_pruning_jobs(repositories, save.id)
    result_json = json.loads(jobs[0]["result_json"])
    assert result_json["proposed_count"] == 3
    assert result_json["archived_count"] == 0
    assert result_json["rejected_count"] == 3


def test_state_pruning_rejects_missing_catalog_row_for_selected_model(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[],
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled"},
        category="scene",
    )
    provider = RecordingStructuredPruningProvider()
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
    )

    with pytest.raises(
        ValueError,
        match="does not advertise structured output or tool calling",
    ):
        asyncio.run(service.prune(save_id=save.id))

    assert provider.chat_requests == []
    assert provider.structured_output_requests == []
    jobs = _state_pruning_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert "does not advertise structured output or tool calling" in jobs[0]["error"]


def test_state_pruning_marks_job_failed_when_model_is_not_structured(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(
        repositories,
        model_capabilities=[ProviderCapability.CHAT.value],
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled"},
        category="scene",
    )
    provider = RecordingStructuredPruningProvider()
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
    )

    with pytest.raises(ValueError, match="does not advertise structured output"):
        asyncio.run(service.prune(save_id=save.id))

    assert provider.chat_requests == []
    assert provider.structured_output_requests == []
    jobs = _state_pruning_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert "does not advertise structured output" in jobs[0]["error"]
    result_json = json.loads(jobs[0]["result_json"])
    assert result_json["active_state_count"] == 1
    assert result_json["batch_count"] == 1
    assert result_json["completed_batch_count"] == 0
    assert len(repositories.list_world_state(save.id)) == 1


def test_state_pruning_marks_job_failed_when_model_is_unavailable(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled"},
        category="scene",
    )
    provider = RecordingStructuredPruningProvider()
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
    )

    with pytest.raises(ValueError, match="model is unavailable"):
        asyncio.run(service.prune(save_id=save.id))

    assert provider.chat_requests == []
    assert provider.structured_output_requests == []
    jobs = _state_pruning_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert "model is unavailable" in jobs[0]["error"]
    result_json = json.loads(jobs[0]["result_json"])
    assert result_json["active_state_count"] == 1
    assert result_json["batch_count"] == 1
    assert result_json["completed_batch_count"] == 0


def test_state_pruning_marks_lifecycle_job_cancelled(
    repositories: PersistenceRepositories,
) -> None:
    save = _save_with_state_pruning_preference(repositories)
    repositories.upsert_world_state(
        save_id=save.id,
        key="scene.old_alarm",
        value={"status": "disabled", "location": "north stair"},
        category="scene",
    )
    provider = SlowStructuredPruningProvider()
    service = StatePruningService(
        repositories=repositories,
        providers={"fake": provider},
    )

    async def run_and_cancel() -> None:
        task = asyncio.create_task(
            service.prune(save_id=save.id, review_only=False)
        )
        await provider.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())

    jobs = _state_pruning_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "cancelled"
    assert jobs[0]["error"] == "State pruning cancelled"


def _save_with_state_pruning_preference(
    repositories: PersistenceRepositories,
    *,
    model_capabilities: list[str] | None = None,
) -> SaveRecord:
    save = _create_save(repositories)
    repositories.set_model_preference(
        task="state_pruning",
        provider="fake",
        model_id="fake-pruner",
    )
    if model_capabilities is None:
        model_capabilities = [ProviderCapability.STRUCTURED_OUTPUT.value]
    if model_capabilities:
        repositories.save_provider_model(
            provider="fake",
            model_id="fake-pruner",
            display_name="Fake Pruner",
            capabilities=model_capabilities,
            context_window=64_000,
        )
    return save


def _configure_state_pruning_tool_fallback(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting("tool_call_fallback_enabled", True)
    repositories.set_model_preference(
        task="tool_call_fallback",
        provider="fallback",
        model_id="fallback-tools",
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-tools",
        display_name="Fallback Tools",
        capabilities=["tool_calling"],
    )


def _create_save(
    repositories: PersistenceRepositories,
    *,
    title: str = "Night Watch",
) -> SaveRecord:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title=f"{title} Scenario",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title=title)
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The north stair alarm is disabled after the danger passes.",
    )
    return save


def _state_pruning_jobs(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[sqlite3.Row]:
    return list(
        repositories.connection.execute(
            """
            SELECT status, result_json, error
            FROM jobs
            WHERE save_id = ? AND type = 'state_pruning'
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )
    )


def _archived_at(
    repositories: PersistenceRepositories,
    world_state_id: str,
) -> str | None:
    row = repositories.connection.execute(
        "SELECT archived_at FROM world_state WHERE id = ?",
        (world_state_id,),
    ).fetchone()
    assert row is not None
    return cast("str | None", row["archived_at"])
