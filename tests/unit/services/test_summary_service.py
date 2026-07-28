from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from bragi.interaction_mode import InteractionMode
from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import MessageRecord, SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.chat_rendering import chat_system_body
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
)
from bragi.safety import CONTENT_FILTER_TRANSITION
from bragi.services.content_safety_service import (
    ContentSafetyAction,
    ContentSafetyResult,
    ContentSafetyService,
)
from bragi.services.summary_service import PendingMessageEstimate, SummaryService


class BlockingContentSafetyService:
    def __init__(self) -> None:
        self.fade_settings: list[bool] = []

    async def review_narration(self, **kwargs: object) -> ContentSafetyResult:
        self.fade_settings.append(bool(kwargs["fade_to_black_enabled"]))
        return ContentSafetyResult(
            body=CONTENT_FILTER_TRANSITION,
            action=ContentSafetyAction.BLOCK,
            minimum_rating="r",
            transition_applied=True,
            agent_ran=True,
        )


class RecordingSummaryProvider:
    provider_name = "fake"

    def __init__(
        self,
        response_body: str = "Mara crossed the ash bridge and secured the bell.",
    ) -> None:
        self.response_body = response_body
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
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        return ChatResponse(
            body=self.response_body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"prompt": 37, "completion": 11, "total": 48},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("summarization must not request image generation")


class FailingSummaryProvider(RecordingSummaryProvider):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise self.error


class SequenceSummaryProvider(RecordingSummaryProvider):
    def __init__(self, response_bodies: list[str]) -> None:
        super().__init__()
        self.response_bodies = response_bodies

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        if not self.response_bodies:
            raise AssertionError("unexpected summary request")
        body = self.response_bodies.pop(0)
        return ChatResponse(
            body=body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"prompt": 37, "completion": 11, "total": 48},
        )


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        repositories.set_app_setting("content_filter_rating", "unrated")
        yield repositories


def test_context_budget_estimator_reports_pressure_and_threshold(
    repositories: PersistenceRepositories,
) -> None:
    messages = (
        _message("m1", token_estimate=90),
        _message("m2", token_estimate=120),
        _message("m3", token_estimate=90),
    )
    service = SummaryService(
        repositories=repositories,
        providers={},
        threshold=0.70,
    )

    pressure = service.estimate_context_budget(
        messages=list(messages),
        context_window=400,
    )

    assert pressure.token_estimate == 300
    assert pressure.context_window == 400
    assert pressure.pressure == pytest.approx(0.75)
    assert pressure.should_summarize is True

    roomy_service = SummaryService(
        repositories=repositories,
        providers={},
        threshold=0.80,
    )
    roomy_pressure = roomy_service.estimate_context_budget(
        messages=list(messages),
        context_window=500,
    )

    assert roomy_pressure.pressure == pytest.approx(0.60)
    assert roomy_pressure.should_summarize is False


def test_summary_service_disabled_skips_provider_summaries_and_jobs(
    repositories: PersistenceRepositories,
) -> None:
    save, _messages = _save_with_summary_preference(repositories)
    provider = RecordingSummaryProvider()
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        enabled=False,
        threshold=0.10,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=100,
        )
    )

    assert summary is None
    assert provider.chat_requests == []
    assert repositories.list_summaries(save.id) == []
    assert _summarization_jobs(repositories, save.id) == []


def test_summary_service_configured_threshold_controls_generation(
    repositories: PersistenceRepositories,
) -> None:
    save, _messages = _save_with_summary_preference(repositories)
    provider = RecordingSummaryProvider()
    high_threshold_service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.90,
    )

    skipped = asyncio.run(
        high_threshold_service.summarize_if_needed(
            save_id=save.id,
            context_window=300,
        )
    )

    assert skipped is None
    assert provider.chat_requests == []
    assert repositories.list_summaries(save.id) == []
    assert _summarization_jobs(repositories, save.id) == []

    low_threshold_service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.70,
    )

    summary = asyncio.run(
        low_threshold_service.summarize_if_needed(
            save_id=save.id,
            context_window=300,
        )
    )

    assert summary is not None
    assert len(provider.chat_requests) == 1
    jobs = _summarization_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    assert summary.id in jobs[0]["result_json"]


def test_summary_service_summarizes_messages_crossing_raw_history_frontier(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key="recent_player_message_window",
        value=1,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key="recent_narrator_message_window",
        value=1,
    )
    provider = RecordingSummaryProvider(
        response_body=(
            "Mara crossed the ash bridge, heard the bell, and asked who rang it."
        ),
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.99,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=1_000_000,
            pending_message=PendingMessageEstimate(
                body="I follow the answer into the cinders.",
                role="player",
            ),
        )
    )

    assert summary is not None
    assert summary.covers_message_start_id == messages[0].id
    assert summary.covers_message_end_id == messages[2].id
    assert summary.source_message_ids == tuple(message.id for message in messages[:3])
    assert len(provider.chat_requests) == 1
    prompt = _request_prompt(provider.chat_requests[0])
    assert messages[0].body in prompt
    assert messages[2].body in prompt
    assert messages[3].body not in prompt


def test_summary_service_includes_only_canonical_fade_transition_in_coverage(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    fade = repositories.update_message_body(
        save_id=save.id,
        message_id=messages[-1].id,
        body=(
            "The intimate moment is kept off-screen. Hours later, "
            "the next scene begins."
        ),
        safety_transition="fade_to_black",
    )
    latest_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I open my eyes at dawn.",
    )
    service = SummaryService(
        repositories=repositories,
        providers={},
        retain_recent_messages=1,
    )

    covered = service._messages_to_summarize(
        [*repositories.list_messages(save.id)]
    )

    assert fade.id in {message.id for message in covered}
    assert latest_player.id not in {message.id for message in covered}


def test_summary_service_preserves_safe_fade_continuity(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    fade = repositories.update_message_body(
        save_id=save.id,
        message_id=messages[-1].id,
        body=(
            "The intimate moment is kept off-screen. Hours later, "
            "the next scene begins."
        ),
        safety_transition="fade_to_black",
    )
    provider = RecordingSummaryProvider(
        response_body=(
            "Mara crossed the ash bridge; hours later, the next scene began "
            "after a private intimate moment remained off-screen."
        ),
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.10,
        retain_recent_messages=1,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=100,
        )
    )

    assert summary is not None
    request_prompt = _request_prompt(provider.chat_requests[0])
    assert "hands slid" not in request_prompt
    assert "hours later" in request_prompt
    assert "hands slid" not in summary.body
    assert fade.body == (
        "The intimate moment is kept off-screen. Hours later, the next scene begins."
    )


def test_summary_service_summarizes_older_messages_and_persists_metadata(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    provider = RecordingSummaryProvider(
        response_body="Mara crossed the ash bridge, heard the bell, and kept moving.",
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=180,
        )
    )

    assert summary is not None
    persisted = repositories.list_summaries(save.id)
    assert persisted == [summary]
    assert summary.body == (
        "Mara crossed the ash bridge, heard the bell, and kept moving."
    )
    assert summary.covers_message_start_id == messages[0].id
    assert summary.covers_message_end_id == messages[1].id
    assert summary.provider == "fake"
    assert summary.model == "fake-summary"

    assert len(provider.chat_requests) == 1
    request = provider.chat_requests[0]
    assert request.provider == "fake"
    assert request.model_id == "fake-summary"
    prompt = _request_prompt(request)
    assert "I step onto the ash bridge." in prompt
    assert "A bell rings under the span." in prompt
    assert "I ask who rang the bell." not in prompt
    assert "The echo answers from below." not in prompt

    jobs = _summarization_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"
    assert summary.id in jobs[-1]["result_json"]


def test_storyteller_summary_advances_over_directions_but_uses_narration_as_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(
        repositories,
        interaction_mode=InteractionMode.STORYTELLER,
    )
    provider = RecordingSummaryProvider(
        response_body="A bell rang under the ash bridge.",
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=180,
        )
    )

    assert summary is not None
    assert summary.covers_message_start_id == messages[0].id
    assert summary.covers_message_end_id == messages[1].id
    assert summary.source_message_ids == (messages[1].id,)
    prompt = _request_prompt(provider.chat_requests[0])
    assert "A bell rings under the span." in prompt
    assert "I step onto the ash bridge." not in prompt


def test_summary_service_prompts_for_factual_continuity_ledger(
    repositories: PersistenceRepositories,
) -> None:
    save, _messages = _save_with_summary_preference(repositories)
    provider = RecordingSummaryProvider(
        response_body="Mara crossed the ash bridge and heard the bell below.",
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
    )

    asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=180,
        )
    )

    instruction = provider.chat_requests[0].messages[0].body
    assert "third-person factual continuity ledger" in instruction
    assert "Do not write dialogue" in instruction
    assert "Do not ask direct questions" in instruction
    assert "Avoid first-person and second-person phrasing" in instruction


def test_summary_service_budgets_summary_output_before_provider_dispatch(
    repositories: PersistenceRepositories,
) -> None:
    save, _messages = _save_with_summary_preference(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-summary",
        display_name="Fake Summary",
        capabilities=["chat"],
        context_window=4096,
    )
    provider = RecordingSummaryProvider(
        response_body="Mara crossed the ash bridge and heard the bell below.",
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
    )

    asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=180,
        )
    )

    assert provider.chat_requests[0].max_output_tokens == 256


def test_summary_service_batches_summary_source_messages_by_model_window(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-summary",
        display_name="Fake Summary",
        capabilities=["chat"],
        context_window=2200,
    )
    for message in messages[:2]:
        repositories.connection.execute(
            "UPDATE messages SET token_estimate = ? WHERE id = ?",
            (1000, message.id),
        )
    repositories.connection.commit()
    provider = SequenceSummaryProvider(
        [
            "Mara stepped onto the ash bridge.",
            "Mara crossed the ash bridge and heard the hidden bell.",
        ],
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=180,
        )
    )

    assert summary is not None
    assert len(provider.chat_requests) == 2
    assert provider.chat_requests[1].summary is not None
    assert "Mara stepped onto the ash bridge." in provider.chat_requests[1].summary
    assert summary.covers_message_start_id == messages[0].id
    assert summary.covers_message_end_id == messages[1].id
    assert summary.source_message_ids == (messages[0].id, messages[1].id)


def test_summary_service_repairs_rejected_continuation_like_summary(
    repositories: PersistenceRepositories,
) -> None:
    save, _messages = _save_with_summary_preference(repositories)
    provider = SequenceSummaryProvider(
        [
            'Mara: "What do you do next?"',
            "Mara crossed the ash bridge and heard the bell below.",
        ]
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=180,
        )
    )

    assert summary is not None
    assert summary.body == "Mara crossed the ash bridge and heard the bell below."
    assert len(provider.chat_requests) == 2
    repair_prompt = provider.chat_requests[1].messages[-1].body
    assert "Previous summary attempt was rejected" in repair_prompt
    assert "summary rejected as continuation-risk" in repair_prompt
    assert repositories.list_summaries(save.id) == [summary]
    jobs = _summarization_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "succeeded"


def test_summary_service_ignores_already_summarized_messages_for_pressure(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I mark the bell rope with ash.",
        token_estimate=45,
    )
    prior_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[0].id,
        covers_message_end_id=messages[1].id,
        body="Mara crossed the ash bridge and heard a bell under the span.",
        provider="fake",
        model="fake-summary",
    )
    provider = RecordingSummaryProvider()
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.75,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=300,
        )
    )

    assert summary is None
    assert provider.chat_requests == []
    assert repositories.list_summaries(save.id) == [prior_summary]
    assert _summarization_jobs(repositories, save.id) == []


def test_summary_service_rolls_prior_summary_into_new_summary(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    fifth_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I tie a red thread around the bell rope.",
        token_estimate=70,
    )
    sixth_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The thread smokes but does not burn.",
        provider="fake",
        model="fake-chat",
        token_estimate=70,
    )
    prior_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[0].id,
        covers_message_end_id=messages[1].id,
        body="Mara crossed the ash bridge and heard a windless bell.",
        provider="fake",
        model="fake-summary",
    )
    provider = RecordingSummaryProvider(
        response_body=(
            "Mara crossed the ash bridge, heard the windless bell, and "
            "questioned the echo below."
        ),
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.75,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=300,
        )
    )

    assert summary is not None
    assert summary.covers_message_start_id == prior_summary.covers_message_start_id
    assert summary.covers_message_end_id == messages[3].id
    assert summary.source_message_ids == (messages[2].id, messages[3].id)
    assert summary.source_summary_ids == (prior_summary.id,)
    active_summaries = repositories.list_summaries(save.id)
    assert active_summaries == [summary]
    assert prior_summary not in active_summaries

    assert len(provider.chat_requests) == 1
    prompt = _request_prompt(provider.chat_requests[0])
    assert prior_summary.body in prompt
    assert messages[0].body not in prompt
    assert messages[1].body not in prompt
    assert messages[2].body in prompt
    assert messages[3].body in prompt
    assert fifth_message.body not in prompt
    assert sixth_message.body not in prompt


def test_summary_service_counts_single_large_active_summary_as_context_pressure(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    prior_summary_body = (
        "Mara crossed the ash bridge, heard a hidden bell below, and learned "
        "that the span remembers broken oaths. "
        * 12
    )
    prior_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[0].id,
        covers_message_end_id=messages[1].id,
        body=prior_summary_body,
        provider="fake",
        model="fake-summary",
    )
    provider = RecordingSummaryProvider(
        response_body=(
            "Mara's prior ash-bridge summary was compressed while her latest "
            "question and the echo remain in the recent window."
        ),
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
        retain_recent_messages=2,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=260,
        )
    )

    assert summary is not None
    active_summaries = repositories.list_summaries(save.id)
    assert active_summaries == [summary]
    assert prior_summary not in active_summaries
    assert summary.covers_message_start_id == prior_summary.covers_message_start_id
    assert summary.covers_message_end_id == prior_summary.covers_message_end_id
    assert summary.body == provider.response_body

    assert len(provider.chat_requests) == 1
    prompt = _request_prompt(provider.chat_requests[0])
    assert prior_summary_body in prompt
    assert messages[0].body not in prompt
    assert messages[1].body not in prompt
    assert messages[2].body not in prompt
    assert messages[3].body not in prompt


def test_summary_service_rejects_safety_placeholder_without_advancing_coverage(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting("content_filter_rating", "pg")
    save, _messages = _save_with_summary_preference(repositories)
    summary_provider = RecordingSummaryProvider(
        response_body="The generated summary crosses the selected ceiling."
    )
    safety = BlockingContentSafetyService()
    service = SummaryService(
        repositories=repositories,
        providers={"fake": summary_provider},
        threshold=0.10,
        content_safety_service=cast(ContentSafetyService, safety),
    )

    with pytest.raises(ValueError, match="content safety"):
        asyncio.run(
            service.summarize_if_needed(
                save_id=save.id,
                context_window=100,
            )
        )

    assert repositories.list_summaries(save.id) == []
    assert safety.fade_settings == [False]
    jobs = _summarization_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "failed"
    assert "content safety" in jobs[-1]["error"]


def test_summary_service_rolls_up_multiple_active_summaries_without_message_pressure(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    first_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[0].id,
        covers_message_end_id=messages[1].id,
        body="Mara stepped onto the ash bridge and heard a bell below.",
        provider="fake",
        model="fake-summary",
    )
    second_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[2].id,
        covers_message_end_id=messages[3].id,
        body="Mara questioned the bell and an echo answered from below.",
        provider="fake",
        model="fake-summary",
    )
    provider = RecordingSummaryProvider(
        response_body=(
            "Mara crossed the ash bridge, heard the hidden bell, and "
            "received an answer from the echo below."
        ),
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.95,
    )

    summary = asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=10_000,
        )
    )

    assert summary is not None
    assert len(provider.chat_requests) == 1
    prompt = _request_prompt(provider.chat_requests[0])
    assert first_summary.body in prompt
    assert second_summary.body in prompt
    assert messages[0].body not in prompt
    assert messages[3].body not in prompt

    assert repositories.list_summaries(save.id) == [summary]
    assert summary.body == (
        "Mara crossed the ash bridge, heard the hidden bell, and "
        "received an answer from the echo below."
    )
    assert summary.covers_message_start_id == first_summary.covers_message_start_id
    assert summary.covers_message_end_id == second_summary.covers_message_end_id


def test_summary_service_rolls_back_replacement_when_archiving_old_summary_fails(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    first_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[0].id,
        covers_message_end_id=messages[1].id,
        body="Mara stepped onto the ash bridge and heard a bell below.",
        provider="fake",
        model="fake-summary",
    )
    second_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[2].id,
        covers_message_end_id=messages[3].id,
        body="Mara questioned the bell and an echo answered from below.",
        provider="fake",
        model="fake-summary",
    )
    provider = RecordingSummaryProvider(
        response_body="DISTINCTIVE SUMMARY THAT MUST ROLL BACK",
    )
    archived_summary_ids: list[str] = []
    original_archive_summary = repositories.archive_summary

    def fail_after_archive(summary_id: str) -> None:
        archived_summary_ids.append(summary_id)
        active_bodies = [
            summary.body for summary in repositories.list_summaries(save.id)
        ]
        assert provider.response_body in active_bodies
        original_archive_summary(summary_id)
        raise RuntimeError("archive failed")

    monkeypatch.setattr(repositories, "archive_summary", fail_after_archive)
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.95,
    )

    with pytest.raises(RuntimeError, match="archive failed"):
        asyncio.run(
            service.summarize_if_needed(
                save_id=save.id,
                context_window=10_000,
            )
        )

    assert archived_summary_ids == [first_summary.id]
    assert len(provider.chat_requests) == 1
    active_summaries = repositories.list_summaries(save.id)
    assert active_summaries == [first_summary, second_summary]
    assert "DISTINCTIVE SUMMARY THAT MUST ROLL BACK" not in [
        summary.body for summary in active_summaries
    ]


def test_summary_provider_request_instructs_condensing_prior_chronicle(
    repositories: PersistenceRepositories,
) -> None:
    save, _messages = _save_with_summary_preference(repositories)
    provider = RecordingSummaryProvider()
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
    )

    asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=180,
        )
    )

    assert len(provider.chat_requests) == 1
    prompt = _request_prompt(provider.chat_requests[0])
    normalized_prompt = prompt.lower()
    assert "chronicle" in normalized_prompt
    assert (
        "summarize" in normalized_prompt
        or "condense" in normalized_prompt
    )
    assert "durable facts" in normalized_prompt


def test_summary_service_keeps_state_and_memories_separate_from_summaries(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Ash Bridge"},
        category="scene",
        source_message_id=messages[0].id,
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara distrusts bells that ring without wind.",
        tags=["bells", "suspicion"],
        source_message_id=messages[1].id,
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": RecordingSummaryProvider()},
        threshold=0.50,
    )

    asyncio.run(
        service.summarize_if_needed(
            save_id=save.id,
            context_window=180,
        )
    )

    assert repositories.list_world_state(save.id) == [state]
    assert repositories.list_memories(save.id) == [memory]
    assert len(repositories.list_summaries(save.id)) == 1


def test_summary_service_marks_job_failed_without_deleting_messages(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    provider = FailingSummaryProvider(RuntimeError("summary provider unavailable"))
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
    )

    with pytest.raises(RuntimeError, match="summary provider unavailable"):
        asyncio.run(
            service.summarize_if_needed(
                save_id=save.id,
                context_window=180,
            )
        )

    assert [message.id for message in repositories.list_messages(save.id)] == [
        message.id for message in messages
    ]
    assert repositories.list_summaries(save.id) == []
    jobs = _summarization_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "failed"
    assert jobs[-1]["error"] == "summary provider unavailable"
    assert jobs[-1]["result_json"] is None


def test_summary_service_rejects_blank_provider_output_without_archiving_prior_summary(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    prior_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[0].id,
        covers_message_end_id=messages[1].id,
        body="Mara crossed the ash bridge and heard a bell under the span.",
        provider="fake",
        model="fake-summary",
    )
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I chalk the bridge stones with a warning mark.",
        token_estimate=45,
    )
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The mark smokes as the bell rings twice.",
        provider="fake",
        model="fake-chat",
        token_estimate=45,
    )
    provider = RecordingSummaryProvider(response_body=" \n\t ")
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
    )

    with pytest.raises(ValueError, match="blank|empty"):
        asyncio.run(
            service.summarize_if_needed(
                save_id=save.id,
                context_window=180,
            )
        )

    assert len(provider.chat_requests) == 1
    assert repositories.list_summaries(save.id) == [prior_summary]
    assert all(summary.body.strip() for summary in repositories.list_summaries(save.id))
    jobs = _summarization_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "failed"
    assert jobs[-1]["result_json"] is None
    assert "blank" in jobs[-1]["error"].lower() or "empty" in jobs[-1]["error"].lower()


def test_summary_service_rejects_scene_continuation_without_archiving_prior_summary(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    prior_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=messages[0].id,
        covers_message_end_id=messages[1].id,
        body="Mara crossed the ash bridge and heard a bell under the span.",
        provider="fake",
        model="fake-summary",
    )
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I open the cracked cabinet beside the bridge shrine.",
        token_estimate=60,
    )
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A porcelain mug waits inside, still warm from hidden tea.",
        provider="fake",
        model="fake-chat",
        token_estimate=60,
    )
    provider = RecordingSummaryProvider(
        response_body=(
            'You open the cabinet and find the warm mug. "What do you do next?"'
        )
    )
    service = SummaryService(
        repositories=repositories,
        providers={"fake": provider},
        threshold=0.50,
    )

    with pytest.raises(ValueError, match="continuation"):
        asyncio.run(
            service.summarize_if_needed(
                save_id=save.id,
                context_window=180,
            )
        )

    assert len(provider.chat_requests) == 2
    assert "Previous summary attempt was rejected" in (
        provider.chat_requests[1].messages[-1].body
    )
    assert repositories.list_summaries(save.id) == [prior_summary]
    jobs = _summarization_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "failed"
    assert "continuation" in jobs[-1]["error"].lower()
    assert jobs[-1]["result_json"] is None


def test_summary_service_marks_job_failed_when_configured_provider_is_missing(
    repositories: PersistenceRepositories,
) -> None:
    save, messages = _save_with_summary_preference(repositories)
    service = SummaryService(
        repositories=repositories,
        providers={},
        threshold=0.50,
    )

    with pytest.raises(KeyError):
        asyncio.run(
            service.summarize_if_needed(
                save_id=save.id,
                context_window=180,
            )
        )

    assert [message.id for message in repositories.list_messages(save.id)] == [
        message.id for message in messages
    ]
    assert repositories.list_summaries(save.id) == []
    jobs = _summarization_jobs(repositories, save.id)
    assert jobs[-1]["status"] == "failed"
    assert "fake" in jobs[-1]["error"]
    assert jobs[-1]["result_json"] is None


def _save_with_summary_preference(
    repositories: PersistenceRepositories,
    *,
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY,
) -> tuple[SaveRecord, list[MessageRecord]]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Bridge of Cinders",
        premise="A bridge remembers every oath broken on it.",
        player_role="Oathkeeper",
        interaction_mode=interaction_mode,
        content={"starting_scene": "Cinders drift over the bridge stones."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Crossing")
    messages = [
        repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I step onto the ash bridge.",
            token_estimate=55,
        ),
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="A bell rings under the span.",
            provider="fake",
            model="fake-chat",
            token_estimate=65,
        ),
        repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I ask who rang the bell.",
            token_estimate=45,
        ),
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="The echo answers from below.",
            provider="fake",
            model="fake-chat",
            token_estimate=50,
        ),
    ]
    repositories.set_model_preference(
        task="summarization",
        provider="fake",
        model_id="fake-summary",
    )
    return save, messages


def _message(message_id: str, *, token_estimate: int) -> MessageRecord:
    return MessageRecord(
        id=message_id,
        save_id="save-1",
        role="player",
        body="x",
        speaker_name="Mara",
        provider=None,
        model=None,
        token_estimate=token_estimate,
        deleted_at=None,
    )


def _request_prompt(request: ChatRequest) -> str:
    return "\n".join(
        (
            chat_system_body(request),
            *(message.body for message in request.messages),
        )
    )


def _summarization_jobs(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[sqlite3.Row]:
    return list(
        repositories.connection.execute(
            """
            SELECT status, result_json, error
            FROM jobs
            WHERE save_id = ? AND type = 'summarization'
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )
    )
