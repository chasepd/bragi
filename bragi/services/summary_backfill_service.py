"""Manual long-save summary backfill and repair."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from bragi.app_logging import log_event
from bragi.persistence.models import MessageRecord, SummaryRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import ProviderClient
from bragi.services.chat_history_settings import (
    DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW,
    DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW,
    DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW,
    DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW,
    NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    chat_history_window_settings,
    narrator_planner_chat_history_window_settings,
)
from bragi.services.model_preferences import roleplay_model_preference
from bragi.services.summary_service import SummaryService

DEFAULT_BACKFILL_CONTEXT_WINDOW = 4096
DEFAULT_BACKFILL_RESERVED_TOKENS = 512


@dataclass(frozen=True)
class SummaryBackfillResult:
    save_id: str
    summary_id: str | None
    batch_count: int
    summarized_message_count: int
    retained_recent_message_count: int
    archived_summary_count: int
    repaired_batch_count: int
    current_player_window: int
    current_narrator_window: int
    recommended_player_window: int
    recommended_narrator_window: int
    applied_window_changes: dict[str, int]
    skipped_reason: str | None = None

    def to_result(self) -> dict[str, object]:
        result: dict[str, object] = {
            "summary_id": self.summary_id,
            "batch_count": self.batch_count,
            "summarized_message_count": self.summarized_message_count,
            "retained_recent_message_count": self.retained_recent_message_count,
            "archived_summary_count": self.archived_summary_count,
            "repaired_batch_count": self.repaired_batch_count,
            "current_player_window": self.current_player_window,
            "current_narrator_window": self.current_narrator_window,
            "recommended_player_window": self.recommended_player_window,
            "recommended_narrator_window": self.recommended_narrator_window,
            "applied_window_changes": dict(self.applied_window_changes),
        }
        if self.skipped_reason is not None:
            result["skipped_reason"] = self.skipped_reason
        return result


class SummaryBackfillService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
        retain_recent_messages: int = 2,
        batch_token_limit: int | None = None,
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.retain_recent_messages = max(0, retain_recent_messages)
        self.batch_token_limit = batch_token_limit
        self.summary_service = SummaryService(
            repositories=repositories,
            providers=providers,
            retain_recent_messages=self.retain_recent_messages,
        )

    async def backfill_save(
        self,
        save_id: str,
        *,
        apply_recommended_windows: bool = False,
    ) -> SummaryBackfillResult:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")

        window_plan = _window_plan(self.repositories, save_id=save_id)
        messages = self.repositories.list_messages(save_id)
        covered_messages = _messages_to_cover(
            messages,
            retain_recent_messages=self.retain_recent_messages,
        )
        retained_recent_messages = tuple(messages[len(covered_messages) :])
        if not covered_messages:
            return _skipped_result(
                save_id=save_id,
                reason="not_enough_messages",
                retained_recent_message_count=len(retained_recent_messages),
                window_plan=window_plan,
            )

        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="summarization",
        )
        if preference is None:
            return _skipped_result(
                save_id=save_id,
                reason="no_summarization_model",
                retained_recent_message_count=len(retained_recent_messages),
                window_plan=window_plan,
            )

        batch_token_limit = self.batch_token_limit or _batch_token_limit(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
        )
        batches = tuple(_message_batches(covered_messages, batch_token_limit))
        rolling_summary: SummaryRecord | None = None
        repaired_batch_count = 0
        for index, batch in enumerate(batches, start=1):
            prior_summaries = (rolling_summary,) if rolling_summary is not None else ()
            generated = await self.summary_service.generate_summary(
                save_id=save_id,
                preference=preference,
                covered_messages=batch,
                retained_recent_messages=retained_recent_messages,
                prior_summaries=prior_summaries,
            )
            if generated.repaired:
                repaired_batch_count += 1
            rolling_summary = SummaryRecord(
                id=f"summary-backfill-batch-{index}",
                save_id=save_id,
                covers_message_start_id=covered_messages[0].id,
                covers_message_end_id=batch[-1].id,
                body=generated.body,
                provider=generated.provider,
                model=generated.model,
            )

        if rolling_summary is None:
            return _skipped_result(
                save_id=save_id,
                reason="no_batches",
                retained_recent_message_count=len(retained_recent_messages),
                window_plan=window_plan,
            )

        active_summaries = self.repositories.list_summaries(save_id)
        applied_window_changes: dict[str, int] = {}
        self.repositories.begin_transaction()
        try:
            summary = self.repositories.add_summary(
                save_id=save_id,
                covers_message_start_id=covered_messages[0].id,
                covers_message_end_id=covered_messages[-1].id,
                body=rolling_summary.body,
                provider=rolling_summary.provider,
                model=rolling_summary.model,
            )
            for old_summary in active_summaries:
                self.repositories.archive_summary(old_summary.id)
            if apply_recommended_windows:
                applied_window_changes = _apply_recommended_windows(
                    self.repositories,
                    save_id=save_id,
                    window_plan=window_plan,
                )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise

        result = SummaryBackfillResult(
            save_id=save_id,
            summary_id=summary.id,
            batch_count=len(batches),
            summarized_message_count=len(covered_messages),
            retained_recent_message_count=len(retained_recent_messages),
            archived_summary_count=len(active_summaries),
            repaired_batch_count=repaired_batch_count,
            current_player_window=window_plan.current_player,
            current_narrator_window=window_plan.current_narrator,
            recommended_player_window=window_plan.recommended_player,
            recommended_narrator_window=window_plan.recommended_narrator,
            applied_window_changes=applied_window_changes,
        )
        log_event(
            "summary_backfill.succeeded",
            save_id=save_id,
            **result.to_result(),
        )
        return result


@dataclass(frozen=True)
class _WindowPlan:
    current_player: int
    current_narrator: int
    recommended_player: int
    recommended_narrator: int
    current_planner_player: int
    current_planner_narrator: int
    recommended_planner_player: int
    recommended_planner_narrator: int


def _messages_to_cover(
    messages: list[MessageRecord],
    *,
    retain_recent_messages: int,
) -> tuple[MessageRecord, ...]:
    if retain_recent_messages <= 0:
        return tuple(messages)
    if len(messages) <= retain_recent_messages:
        return ()
    return tuple(messages[:-retain_recent_messages])


def _message_batches(
    messages: tuple[MessageRecord, ...],
    token_limit: int,
) -> Iterator[tuple[MessageRecord, ...]]:
    batch: list[MessageRecord] = []
    batch_tokens = 0
    bounded_limit = max(1, token_limit)
    for message in messages:
        message_tokens = _message_tokens(message)
        if batch and batch_tokens + message_tokens > bounded_limit:
            yield tuple(batch)
            batch = []
            batch_tokens = 0
        batch.append(message)
        batch_tokens += message_tokens
    if batch:
        yield tuple(batch)


def _message_tokens(message: MessageRecord) -> int:
    if message.token_estimate is not None:
        return max(1, message.token_estimate)
    return max(1, (len(message.body) + 3) // 4)


def _batch_token_limit(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
) -> int:
    context_window = _model_context_window(
        repositories,
        provider=provider,
        model_id=model_id,
    )
    return max(1, context_window - DEFAULT_BACKFILL_RESERVED_TOKENS)


def _model_context_window(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
) -> int:
    for model in repositories.list_provider_models(provider):
        if model.model_id == model_id and model.context_window:
            return model.context_window
    return DEFAULT_BACKFILL_CONTEXT_WINDOW


def _window_plan(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> _WindowPlan:
    current = chat_history_window_settings(repositories, save_id=save_id)
    planner = narrator_planner_chat_history_window_settings(
        repositories,
        save_id=save_id,
    )
    return _WindowPlan(
        current_player=current.player_messages,
        current_narrator=current.narrator_messages,
        recommended_player=min(
            current.player_messages,
            DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW,
        ),
        recommended_narrator=min(
            current.narrator_messages,
            DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW,
        ),
        current_planner_player=planner.player_messages,
        current_planner_narrator=planner.narrator_messages,
        recommended_planner_player=min(
            planner.player_messages,
            DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW,
        ),
        recommended_planner_narrator=min(
            planner.narrator_messages,
            DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW,
        ),
    )


def _apply_recommended_windows(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    window_plan: _WindowPlan,
) -> dict[str, int]:
    changes: dict[str, int] = {}
    if window_plan.recommended_player != window_plan.current_player:
        repositories.set_scoped_setting(
            scope="save",
            scope_id=save_id,
            key=RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
            value=window_plan.recommended_player,
        )
        changes[RECENT_PLAYER_MESSAGE_WINDOW_SETTING] = (
            window_plan.recommended_player
        )
    if window_plan.recommended_narrator != window_plan.current_narrator:
        repositories.set_scoped_setting(
            scope="save",
            scope_id=save_id,
            key=RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
            value=window_plan.recommended_narrator,
        )
        changes[RECENT_NARRATOR_MESSAGE_WINDOW_SETTING] = (
            window_plan.recommended_narrator
        )
    if (
        window_plan.recommended_planner_player
        != window_plan.current_planner_player
    ):
        repositories.set_scoped_setting(
            scope="save",
            scope_id=save_id,
            key=NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
            value=window_plan.recommended_planner_player,
        )
        changes[NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING] = (
            window_plan.recommended_planner_player
        )
    if (
        window_plan.recommended_planner_narrator
        != window_plan.current_planner_narrator
    ):
        repositories.set_scoped_setting(
            scope="save",
            scope_id=save_id,
            key=NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
            value=window_plan.recommended_planner_narrator,
        )
        changes[NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING] = (
            window_plan.recommended_planner_narrator
        )
    return changes


def _skipped_result(
    *,
    save_id: str,
    reason: str,
    retained_recent_message_count: int,
    window_plan: _WindowPlan,
) -> SummaryBackfillResult:
    return SummaryBackfillResult(
        save_id=save_id,
        summary_id=None,
        batch_count=0,
        summarized_message_count=0,
        retained_recent_message_count=retained_recent_message_count,
        archived_summary_count=0,
        repaired_batch_count=0,
        current_player_window=window_plan.current_player,
        current_narrator_window=window_plan.current_narrator,
        recommended_player_window=window_plan.recommended_player,
        recommended_narrator_window=window_plan.recommended_narrator,
        applied_window_changes={},
        skipped_reason=reason,
    )
