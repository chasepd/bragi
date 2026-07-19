from __future__ import annotations

import asyncio
import sqlite3
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
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
from bragi.services import world_suggestion_review_service as module
from bragi.services.world_suggestion_review_service import (
    WorldSuggestionReviewService,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


class FakeSuggestionReviewer:
    provider_name = "fake"

    def __init__(
        self,
        response: dict[str, object] | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.response = response or {"decisions": []}
        self.failure = failure
        self.requests: list[StructuredOutputRequest] = []
        self.chat_requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("world-suggestion review must not use chat prose")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return StructuredOutputResponse(
            data=self.response,
            provider=request.provider,
            model_id=request.model_id,
        )


class FakeToolSuggestionReviewer:
    provider_name: str

    def __init__(
        self,
        *,
        provider_name: str = "fake",
        responses: tuple[tuple[ProviderToolCall, ...], ...],
    ) -> None:
        self.provider_name = provider_name
        self.responses = deque(responses)
        self.requests: list[ToolCallRequest] = []

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
                model_id=f"{self.provider_name}-model",
                display_name=f"{self.provider_name.title()} Model",
                capabilities=frozenset({ProviderCapability.TOOL_CALLING}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("world-suggestion review must not use chat prose")

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("world-suggestion review must not generate images")

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.requests.append(request)
        calls = self.responses.popleft() if self.responses else ()
        return ToolCallResponse(
            tool_calls=calls,
            body="tool response",
            provider=request.provider,
            model_id=request.model_id,
        )


def test_review_accepts_and_applies_pending_suggestion(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id, location_id, suggestion_id = _pending_location_suggestion(
        repositories
    )
    provider = FakeSuggestionReviewer(
        {
            "decisions": [
                {
                    "review_id": suggestion_id,
                    "action": "accept",
                    "reason": "The cited message directly supports the new text.",
                }
            ]
        }
    )

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
        ).review_pending(save_id)
    )

    assert result.reviewed_count == 1
    assert result.applied_count == 1
    assert result.rejected_count == 0
    assert result.deferred_count == 0
    location = repositories.get_location(location_id)
    assert location is not None
    assert location.description == "The beacon lens burns with red warning glyphs."
    suggestion = repositories.list_context_update_suggestions(save_id)[0]
    assert suggestion.status == "applied"
    audit = repositories.list_context_update_audit(save_id)[-1]
    assert audit.suggestion_id == suggestion_id
    assert audit.operation == "agent_suggestion_apply"
    assert "directly supports" in audit.reason
    assert provider.chat_requests == []
    assert provider.requests[0].schema_name == "world_suggestion_review"
    prompt = "\n".join(message.body for message in provider.requests[0].messages)
    assert suggestion_id in prompt
    assert "The narrator described the beacon lens change." in prompt
    assert message_id in prompt


def test_review_rejects_pending_suggestion_without_applying(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, location_id, suggestion_id = _pending_location_suggestion(
        repositories
    )
    provider = FakeSuggestionReviewer(
        {
            "decisions": [
                {
                    "review_id": suggestion_id,
                    "action": "reject",
                    "reason": (
                        "The cited reason does not justify overwriting the field."
                    ),
                }
            ]
        }
    )

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
        ).review_pending(save_id)
    )

    assert result.reviewed_count == 1
    assert result.applied_count == 0
    assert result.rejected_count == 1
    location = repositories.get_location(location_id)
    assert location is not None
    assert location.description == "The beacon lens overlooks the ash gate."
    suggestion = repositories.list_context_update_suggestions(save_id)[0]
    assert suggestion.status == "rejected"
    audit = repositories.list_context_update_audit(save_id)[-1]
    assert audit.suggestion_id == suggestion_id
    assert audit.operation == "agent_suggestion_reject"
    assert "does not justify" in audit.reason


def test_review_failure_leaves_suggestion_pending_for_future_pass(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, _location_id, suggestion_id = _pending_location_suggestion(
        repositories
    )
    provider = FakeSuggestionReviewer(failure=TimeoutError("provider timed out"))

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
        ).review_pending(save_id)
    )

    assert result.reviewed_count == 1
    assert result.applied_count == 0
    assert result.rejected_count == 0
    assert result.deferred_count == 1
    assert result.error == "provider timed out"
    suggestion = repositories.list_context_update_suggestions(save_id)[0]
    assert suggestion.status == "pending"
    audit = repositories.list_context_update_audit(save_id)[-1]
    assert audit.suggestion_id == suggestion_id
    assert audit.operation == "agent_suggestion_review_deferred"
    assert "provider timed out" in audit.reason
    assert suggestion.review_attempt_count == 1
    assert suggestion.next_review_at is not None
    assert suggestion.last_review_error == "provider timed out"


def test_review_failure_rejects_after_three_automated_attempts(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, _location_id, suggestion_id = _pending_location_suggestion(
        repositories
    )
    provider = FakeSuggestionReviewer(failure=TimeoutError("provider timed out"))
    service = WorldSuggestionReviewService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-reviewer",
    )

    for _ in range(3):
        asyncio.run(service.review_pending(save_id))

    suggestion = repositories.list_context_update_suggestions(save_id)[0]
    assert suggestion.id == suggestion_id
    assert suggestion.status == "rejected"
    assert suggestion.next_review_at is None
    audits = repositories.list_context_update_audit(save_id)
    assert [audit.operation for audit in audits] == [
        "agent_suggestion_review_deferred",
        "agent_suggestion_review_retry_exhausted",
    ]


def test_due_only_review_excludes_suggestions_in_backoff(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, location_id, deferred_id = _pending_location_suggestion(
        repositories
    )
    repositories.defer_context_update_suggestion_review(
        [deferred_id],
        error="temporary failure",
        retry_after_seconds=30 * 60,
    )
    due = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="location",
        entity_id=location_id,
        field_path="visual_description",
        proposed_value="A cracked lens rimmed in ash.",
        reason="A new observation supports the visual detail.",
        confidence=0.8,
    )
    provider = FakeSuggestionReviewer(
        {
            "decisions": [
                {
                    "review_id": due.id,
                    "action": "reject",
                    "reason": "Not enough evidence yet.",
                }
            ]
        }
    )

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
        ).review_pending(save_id, due_only=True)
    )

    assert result.reviewed_count == 1
    assert repositories.list_context_update_suggestions(save_id)[0].status == "pending"
    assert repositories.list_context_update_suggestions(save_id)[1].status == "rejected"
    prompt = "\n".join(message.body for message in provider.requests[0].messages)
    assert deferred_id not in prompt
    assert due.id in prompt


def test_review_preflight_rejects_missing_character_without_provider_call(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, _location_id, suggestion_id = _pending_location_suggestion(
        repositories
    )
    repositories.update_context_update_suggestion_status(
        suggestion_id,
        status="dismissed",
    )
    stale = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="character",
        entity_id="missing-character",
        field_path="goals",
        proposed_value="Leave the keep.",
        reason="Stale proposal.",
        confidence=0.6,
    )
    provider = FakeSuggestionReviewer()

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
        ).review_pending(save_id)
    )

    assert result.rejected_count == 1
    assert repositories.list_context_update_suggestions(save_id)[1].status == "rejected"
    assert provider.requests == []
    audit = repositories.list_context_update_audit(save_id)[-1]
    assert audit.suggestion_id == stale.id
    assert audit.operation == "agent_suggestion_preflight_reject"


def test_review_preflight_rejects_missing_source_without_provider_call(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, _location_id, suggestion_id = _pending_location_suggestion(
        repositories
    )
    repositories.connection.execute(
        """
        UPDATE context_update_suggestions
        SET source_message_ids_json = '["missing-message"]'
        WHERE id = ?
        """,
        (suggestion_id,),
    )
    repositories.commit()
    provider = FakeSuggestionReviewer()

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
        ).review_pending(save_id)
    )

    assert result.rejected_count == 1
    assert provider.requests == []
    assert repositories.list_context_update_suggestions(save_id)[0].status == "rejected"


def test_review_preflight_checks_sources_for_entityless_suggestions(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, _location_id, suggestion_id = _pending_location_suggestion(
        repositories
    )
    repositories.connection.execute(
        """
        UPDATE context_update_suggestions
        SET entity_type = 'world_state', entity_id = NULL,
            source_message_ids_json = '["missing-message"]'
        WHERE id = ?
        """,
        (suggestion_id,),
    )
    repositories.commit()
    provider = FakeSuggestionReviewer()

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
        ).review_pending(save_id)
    )

    assert result.rejected_count == 1
    assert provider.requests == []


def test_review_preflight_rejects_only_invalid_member_of_duplicate_group(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, location_id, valid_id = _pending_location_suggestion(
        repositories
    )
    invalid = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="location",
        entity_id=location_id,
        field_path="description",
        proposed_value="The beacon lens burns with red warning glyphs.",
        reason="Duplicate with a stale source.",
        confidence=0.8,
        source_message_ids=["missing-message"],
    )
    provider = FakeSuggestionReviewer(
        {
            "decisions": [
                {
                    "review_id": valid_id,
                    "action": "accept",
                    "reason": "The valid source supports the update.",
                }
            ]
        }
    )

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
        ).review_pending(save_id)
    )

    statuses = {
        suggestion.id: suggestion.status
        for suggestion in repositories.list_context_update_suggestions(save_id)
    }
    assert result.applied_count == 1
    assert result.rejected_count == 1
    assert statuses == {valid_id: "applied", invalid.id: "rejected"}


def test_review_reject_failure_defers_group_for_future_pass(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_id, _message_id, _location_id, suggestion_id = _pending_location_suggestion(
        repositories
    )
    provider = FakeSuggestionReviewer(
        {
            "decisions": [
                {
                    "review_id": suggestion_id,
                    "action": "reject",
                    "reason": "The update is unsupported.",
                }
            ]
        }
    )

    def reject_suggestions(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("audit write failed")

    monkeypatch.setattr(
        module.WorldDataService,
        "reject_suggestions",
        reject_suggestions,
    )

    result = asyncio.run(
        module.WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
        ).review_pending(save_id)
    )

    assert result.reviewed_count == 1
    assert result.applied_count == 0
    assert result.rejected_count == 0
    assert result.deferred_count == 1
    assert result.error == "reject failed: audit write failed"
    suggestion = repositories.list_context_update_suggestions(save_id)[0]
    assert suggestion.status == "pending"
    audit = repositories.list_context_update_audit(save_id)[-1]
    assert audit.suggestion_id == suggestion_id
    assert audit.operation == "agent_suggestion_review_deferred"
    assert "Automated reject deferred" in audit.reason
    assert "audit write failed" in audit.reason


def test_review_applies_duplicate_pending_group_once(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, location_id, first_id = _pending_location_suggestion(
        repositories
    )
    duplicate = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="location",
        entity_id=location_id,
        field_path="description",
        proposed_value="The beacon lens burns with red warning glyphs.",
        reason="Same update from another extraction pass.",
        confidence=0.81,
        source_message_ids=[],
    )
    provider = FakeSuggestionReviewer(
        {
            "decisions": [
                {
                    "review_id": first_id,
                    "action": "accept",
                    "reason": "The grouped suggestion is supported.",
                }
            ]
        }
    )

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
        ).review_pending(save_id)
    )

    assert result.reviewed_count == 1
    assert result.applied_count == 2
    statuses = {
        suggestion.id: suggestion.status
        for suggestion in repositories.list_context_update_suggestions(save_id)
    }
    assert statuses == {first_id: "applied", duplicate.id: "applied"}
    location = repositories.get_location(location_id)
    assert location is not None
    assert location.description == (
        "The beacon lens burns with red warning glyphs."
    )
    apply_audits = [
        audit
        for audit in repositories.list_context_update_audit(save_id)
        if audit.operation == "agent_suggestion_apply"
    ]
    assert [audit.suggestion_id for audit in apply_audits] == [first_id, duplicate.id]


def test_tool_review_retries_malformed_arguments_with_validation_feedback(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, location_id, _suggestion_id = _pending_location_suggestion(
        repositories
    )
    provider = FakeToolSuggestionReviewer(
        responses=(
            (
                ProviderToolCall(
                    id="call-bad-json",
                    name="review_world_suggestion",
                    arguments_json='{"review_id":',
                ),
            ),
            (
                ProviderToolCall(
                    id="call-accept",
                    name="review_world_suggestion",
                    arguments_json=(
                        '{"review_id":"suggestion-gallery-description",'
                        '"action":"accept","reason":"The cited source supports it."}'
                    ),
                ),
            ),
        )
    )

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-reviewer",
            prefer_tool_calls=True,
        ).review_pending(save_id)
    )

    assert result.applied_count == 1
    assert result.deferred_count == 0
    assert len(provider.requests) == 2
    feedback_messages = [
        message
        for message in provider.requests[1].messages
        if message.role == "tool"
    ]
    assert feedback_messages
    assert feedback_messages[0].tool_call_id == "call-bad-json"
    assert "Malformed JSON arguments" in feedback_messages[0].body
    location = repositories.get_location(location_id)
    assert location is not None
    assert location.description == "The beacon lens burns with red warning glyphs."


def test_tool_review_uses_fallback_after_argument_validation_failure(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _message_id, _location_id, suggestion_id = _pending_location_suggestion(
        repositories
    )
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
    invalid_call = ProviderToolCall(
        id="call-bad-json",
        name="review_world_suggestion",
        arguments_json='{"review_id":',
    )
    primary = FakeToolSuggestionReviewer(
        provider_name="primary",
        responses=((invalid_call,), (invalid_call,), (invalid_call,)),
    )
    fallback = FakeToolSuggestionReviewer(
        provider_name="fallback",
        responses=(
            (
                ProviderToolCall(
                    id="call-reject",
                    name="review_world_suggestion",
                    arguments_json=(
                        '{"review_id":"suggestion-gallery-description",'
                        '"action":"reject","reason":"Fallback rejected it."}'
                    ),
                ),
            ),
        ),
    )

    result = asyncio.run(
        WorldSuggestionReviewService(
            repositories=repositories,
            provider=primary,
            provider_name="primary",
            model_id="primary-tools",
            providers={"primary": primary, "fallback": fallback},
            prefer_tool_calls=True,
        ).review_pending(save_id)
    )

    assert result.rejected_count == 1
    assert result.deferred_count == 0
    assert len(primary.requests) == 3
    assert len(fallback.requests) == 1
    assert fallback.requests[0].provider == "fallback"
    assert fallback.requests[0].model_id == "fallback-tools"
    suggestion = repositories.list_context_update_suggestions(save_id)[0]
    assert suggestion.id == suggestion_id
    assert suggestion.status == "rejected"


def _pending_location_suggestion(
    repositories: PersistenceRepositories,
) -> tuple[str, str, str, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The narrator described the beacon lens change.",
        provider="fake",
        model="fake-chat",
        message_id="message-narrator",
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        description="The beacon lens overlooks the ash gate.",
        source_message_id=message.id,
        location_id="location-gallery",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="field_update",
        entity_type="location",
        entity_id=location.id,
        field_path="description",
        proposed_value="The beacon lens burns with red warning glyphs.",
        reason="The narrator described the beacon lens change.",
        confidence=0.91,
        source_message_ids=[message.id],
        suggestion_id="suggestion-gallery-description",
    )
    return save.id, message.id, location.id, suggestion.id
