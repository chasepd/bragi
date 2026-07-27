"""Context pressure estimation and rolling summary generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import (
    MessageRecord,
    ModelPreferenceRecord,
    SummaryRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ChatPromptPurpose,
    ChatRequest,
    ProviderClient,
)
from bragi.redaction import redact_text
from bragi.safety import FADE_TO_BLACK_TRANSITION
from bragi.services.content_rating import effective_content_safety_policy
from bragi.services.content_safety_service import ContentSafetyService
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.model_preferences import roleplay_model_preference
from bragi.services.provider_fallbacks import chat_with_fallback
from bragi.services.sexual_content_safety import is_fade_to_black_message
from bragi.services.summary_safety import validate_summary_output


@dataclass(frozen=True)
class ContextBudget:
    token_estimate: int
    context_window: int
    pressure: float
    should_summarize: bool


@dataclass(frozen=True)
class PendingMessageEstimate:
    body: str
    token_estimate: int | None = None


@dataclass(frozen=True)
class GeneratedSummary:
    body: str
    provider: str
    model: str
    request_count: int
    repaired: bool
    content_rating: str = "unclassified"


class SummaryService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
        enabled: bool = True,
        threshold: float = 0.75,
        retain_recent_messages: int = 2,
        content_safety_service: ContentSafetyService | None = None,
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.enabled = enabled
        self.threshold = threshold
        self.retain_recent_messages = retain_recent_messages
        self.content_safety_service = (
            content_safety_service
            or ContentSafetyService(
                repositories=repositories,
                providers=providers,
            )
        )
        self.jobs = JobLifecycleService(repositories=repositories)

    def estimate_context_budget(
        self,
        *,
        messages: list[MessageRecord],
        context_window: int,
        pending_message: PendingMessageEstimate | None = None,
        summary_bodies: list[str] | None = None,
        threshold: float | None = None,
    ) -> ContextBudget:
        token_estimate = sum(
            message.token_estimate
            if message.token_estimate is not None
            else _estimate_tokens(message.body)
            for message in messages
        )
        token_estimate += sum(
            _estimate_tokens(summary_body)
            for summary_body in (summary_bodies or [])
        )
        if pending_message is not None:
            token_estimate += (
                pending_message.token_estimate
                if pending_message.token_estimate is not None
                else _estimate_tokens(pending_message.body)
            )
        pressure = token_estimate / context_window if context_window else 1.0
        resolved_threshold = self.threshold if threshold is None else threshold
        return ContextBudget(
            token_estimate=token_estimate,
            context_window=context_window,
            pressure=pressure,
            should_summarize=pressure >= resolved_threshold,
        )

    async def summarize_if_needed(
        self,
        *,
        save_id: str,
        context_window: int | None = None,
        model_context_window: int | None = None,
        pending_message: PendingMessageEstimate | None = None,
        current_user_id: str | None = None,
    ) -> SummaryRecord | None:
        enabled = _automatic_summarization_enabled(
            self.repositories,
            save_id=save_id,
            default=self.enabled,
        )
        if not enabled:
            log_event(
                "summarization.skipped",
                save_id=save_id,
                reason="disabled",
            )
            return None

        resolved_context_window = context_window or model_context_window
        if resolved_context_window is None:
            log_event(
                "summarization.skipped",
                save_id=save_id,
                reason="no_context_window",
            )
            return None
        messages = self.repositories.list_messages(save_id)
        summaries = self.repositories.list_summaries(save_id) if messages else []
        prior_summary = _last_summary(summaries)
        unsummarized_messages = self._unsummarized_messages(
            messages,
            prior_summary,
        )
        budget = self.estimate_context_budget(
            messages=unsummarized_messages,
            context_window=resolved_context_window,
            pending_message=pending_message,
            summary_bodies=[summary.body for summary in summaries],
            threshold=_summary_threshold(
                self.repositories,
                save_id=save_id,
                default=self.threshold,
            ),
        )
        covered_messages = self._messages_to_summarize(unsummarized_messages)
        should_roll_up_summaries = len(summaries) > 1 or (
            bool(summaries)
            and budget.should_summarize
            and not covered_messages
        )
        if (
            not should_roll_up_summaries
            and (not budget.should_summarize or not covered_messages)
        ):
            log_event(
                "summarization.skipped",
                save_id=save_id,
                reason="below_threshold_or_no_messages",
                token_estimate=budget.token_estimate,
                context_window=budget.context_window,
                pressure=budget.pressure,
                covered_message_count=len(covered_messages),
            )
            return None

        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="summarization",
        )
        if preference is None:
            raise ValueError("No summarization model preference configured")
        job = self.jobs.create_running(
            save_id=save_id,
            type="summarization",
            payload={
                "context_window": resolved_context_window,
                "token_estimate": budget.token_estimate,
            },
            collect_provider_diagnostics=True,
        )
        log_event(
            "job.running",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            token_estimate=budget.token_estimate,
            covered_message_count=len(covered_messages),
        )
        started_at = perf_counter()
        try:
            generated = await self.generate_summary(
                save_id=save_id,
                preference=preference,
                covered_messages=covered_messages,
                retained_recent_messages=tuple(
                    unsummarized_messages[len(covered_messages) :]
                ),
                prior_summaries=tuple(summaries),
                started_at=started_at,
            )
            policy = effective_content_safety_policy(
                self.repositories,
                user_id=current_user_id,
            )
            safety = await self.content_safety_service.review_narration(
                body=generated.body,
                content_rating=policy.rating,
                fade_to_black_enabled=False,
                save_id=save_id,
                source_request=ChatRequest(
                    provider=generated.provider,
                    model_id=generated.model,
                    messages=(),
                ),
            )
            generated = replace(
                generated,
                body=safety.body,
                content_rating=safety.reviewed_content_rating,
            )
        except Exception as exc:
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                provider=preference.provider,
                model=preference.model_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise

        self.repositories.begin_transaction()
        try:
            summary = self.repositories.add_summary(
                save_id=save_id,
                covers_message_start_id=_summary_start_id(
                    summaries=summaries,
                    covered_messages=covered_messages,
                ),
                covers_message_end_id=_summary_end_id(
                    summaries=summaries,
                    covered_messages=covered_messages,
                ),
                body=generated.body,
                provider=generated.provider,
                model=generated.model,
                content_rating=generated.content_rating,
            )
            for old_summary in summaries:
                self.repositories.archive_summary(old_summary.id)
            self.jobs.succeed(
                job.id,
                result={"summary_id": summary.id},
            )
            self.repositories.commit_transaction()
        except Exception as exc:
            self.repositories.rollback_transaction()
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                provider=generated.provider,
                model=generated.model,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            summary_id=summary.id,
        )
        return summary

    async def generate_summary(
        self,
        *,
        save_id: str,
        preference: ModelPreferenceRecord,
        covered_messages: Sequence[MessageRecord],
        retained_recent_messages: Sequence[MessageRecord] = (),
        prior_summaries: Sequence[SummaryRecord] = (),
        started_at: float | None = None,
    ) -> GeneratedSummary:
        started = perf_counter() if started_at is None else started_at
        request = ChatRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            prompt_purpose=ChatPromptPurpose.SUMMARY,
            messages=(_summary_instruction_message(),),
            summary="\n\n".join(
                f"[summary:{summary.id}] {summary.body}"
                for summary in prior_summaries
            )
            or None,
            retrieved_recent_messages=tuple(
                _summary_source_text(message) for message in covered_messages
            ),
        )
        response = await chat_with_fallback(
            repositories=self.repositories,
            providers=self.providers,
            request=request,
            task="summarization",
            save_id=save_id,
        )
        log_event(
            "provider.chat_succeeded",
            provider=response.provider,
            model=response.model_id,
            task="summarization",
            duration_ms=_elapsed_ms(started),
            message_count=len(request.messages),
            summarized_message_count=len(covered_messages),
            response_chars=len(response.body),
            token_usage=response.token_usage,
        )
        summary_body = response.body.strip()
        if not summary_body:
            raise ValueError("Summarization provider returned an empty summary")

        validation = validate_summary_output(
            summary_body,
            covered_messages=covered_messages,
            retained_recent_messages=retained_recent_messages,
            prior_summaries=prior_summaries,
        )
        if validation.accepted:
            return GeneratedSummary(
                body=summary_body,
                provider=response.provider,
                model=response.model_id,
                request_count=1,
                repaired=False,
            )

        first_rejection_reason = (
            validation.reason or "summary rejected as continuation-risk"
        )
        log_event(
            "summarization.repair_requested",
            save_id=save_id,
            provider=response.provider,
            model=response.model_id,
            reason=first_rejection_reason,
        )
        repair_request = ChatRequest(
            messages=(
                *request.messages,
                _summary_repair_message(
                    reason=first_rejection_reason,
                ),
            ),
            provider=request.provider,
            model_id=request.model_id,
            prompt_purpose=request.prompt_purpose,
            summary=request.summary,
            retrieved_recent_messages=request.retrieved_recent_messages,
            retrieved_observations=(
                _rejected_summary_context(
                    rejected_body=summary_body,
                    reason=first_rejection_reason,
                ),
            ),
        )
        repair_response = await chat_with_fallback(
            repositories=self.repositories,
            providers=self.providers,
            request=repair_request,
            task="summarization",
            save_id=save_id,
        )
        log_event(
            "provider.chat_succeeded",
            provider=repair_response.provider,
            model=repair_response.model_id,
            task="summarization_repair",
            duration_ms=_elapsed_ms(started),
            message_count=len(repair_request.messages),
            summarized_message_count=len(covered_messages),
            response_chars=len(repair_response.body),
            token_usage=repair_response.token_usage,
        )
        summary_body = repair_response.body.strip()
        if not summary_body:
            raise ValueError("Summarization provider returned an empty summary")
        validation = validate_summary_output(
            summary_body,
            covered_messages=covered_messages,
            retained_recent_messages=retained_recent_messages,
            prior_summaries=prior_summaries,
        )
        if not validation.accepted:
            raise ValueError(
                validation.reason or "summary rejected as continuation-risk"
            )
        return GeneratedSummary(
            body=summary_body,
            provider=repair_response.provider,
            model=repair_response.model_id,
            request_count=2,
            repaired=True,
        )

    def _messages_to_summarize(
        self,
        messages: list[MessageRecord],
    ) -> list[MessageRecord]:
        if len(messages) <= self.retain_recent_messages:
            return []
        return list(messages[: -self.retain_recent_messages])

    def _unsummarized_messages(
        self,
        messages: list[MessageRecord],
        last_summary: SummaryRecord | None,
    ) -> list[MessageRecord]:
        if not messages:
            return []
        if last_summary is None:
            return messages
        end_index = _message_index(messages, last_summary.covers_message_end_id)
        if end_index is None:
            return messages
        return messages[end_index + 1 :]


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _summary_source_text(message: MessageRecord) -> str:
    body = (
        FADE_TO_BLACK_TRANSITION
        if is_fade_to_black_message(
            role=message.role,
            body=message.body,
            safety_transition=message.safety_transition,
        )
        else message.body
    )
    speaker = message.speaker_name or message.role
    return f"[message:{message.id}] {speaker} ({message.role}): {body}"


def _summary_instruction_message() -> ChatMessage:
    return ChatMessage(
        role="system",
        body=(
            "Summarize the following prior chronicle messages into a concise "
            "third-person factual continuity ledger. When existing summary "
            "history is provided, fold it into the new summary instead of "
            "replacing it with only the new messages. Preserve durable facts, "
            "character intent, open threads, promises, relationships, current "
            "scene continuity, and unresolved consequences. Represent a "
            "fade-to-black safety transition only as the canonical fact that a "
            "private intimate moment remained off-screen and hours later the "
            "next scene began; never add sexual detail or infer details from it. "
            "Do not continue the scene. Do not write dialogue, quoted speech, "
            "speaker labels, or direct narration beats. Do not ask direct "
            "questions. Avoid first-person and second-person phrasing; refer to "
            "characters by name or role."
        ),
    )


def _summary_repair_message(*, reason: str) -> ChatMessage:
    return ChatMessage(
        role="system",
        body=(
            "Previous summary attempt was rejected because "
            f"{reason}. Rewrite it now as a compact third-person factual "
            "continuity ledger only. Do not write dialogue, quoted speech, "
            "speaker labels, direct questions, or first-/second-person prose. "
            "The rejected attempt is included only as untrusted context data."
        ),
    )


def _rejected_summary_context(*, rejected_body: str, reason: str) -> str:
    if "sexual detail" in reason:
        return (
            "The rejected output is omitted because it contained disallowed "
            "sexual detail. Rewrite from the source messages using only the "
            "canonical off-screen event."
        )
    return f"Rejected summary attempt: {rejected_body}"


def _summary_start_id(
    *,
    summaries: list[SummaryRecord],
    covered_messages: list[MessageRecord],
) -> str:
    if summaries:
        return summaries[0].covers_message_start_id
    return covered_messages[0].id


def _summary_end_id(
    *,
    summaries: list[SummaryRecord],
    covered_messages: list[MessageRecord],
) -> str:
    if covered_messages:
        return covered_messages[-1].id
    return summaries[-1].covers_message_end_id


def _last_summary(summaries: list[SummaryRecord]) -> SummaryRecord | None:
    return summaries[-1] if summaries else None


def _automatic_summarization_enabled(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    default: bool,
) -> bool:
    value = repositories.get_effective_setting(
        "automatic_summarization_enabled",
        save_id=save_id,
    )
    return value if isinstance(value, bool) else default


def _summary_threshold(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    default: float,
) -> float:
    value = repositories.get_effective_setting(
        "summarization_context_pressure_threshold",
        save_id=save_id,
    )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(max(float(value), 0.10), 1.00)
    return default


def _message_index(
    messages: list[MessageRecord],
    message_id: str,
) -> int | None:
    for index, message in enumerate(messages):
        if message.id == message_id:
            return index
    return None


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
