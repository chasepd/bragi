"""Privacy-safe recent chat response timing summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.model_preferences import roleplay_model_preference
from bragi.services.turn_responsiveness import turn_responsiveness_mode

MIN_TIMING_SAMPLES = 5
MAX_TIMING_SAMPLES = 30


@dataclass(frozen=True)
class ChatTimingEstimate:
    p50_ms: int
    p95_ms: int


@dataclass(frozen=True)
class ChatTimingOutcomes:
    terminal_count: int
    success_count: int
    failed_count: int
    interrupted_count: int
    failure_rate: float | None
    route_sample_count: int
    fast_path_count: int
    combined_path_count: int
    standard_path_count: int
    unclassified_success_count: int


@dataclass(frozen=True)
class ChatTimingSummary:
    mode: str
    provider: str | None
    model: str | None
    sample_count: int
    estimate: ChatTimingEstimate | None
    outcomes: ChatTimingOutcomes


class ChatTimingService:
    def __init__(
        self,
        repositories: PersistenceRepositories,
    ) -> None:
        self.repositories = repositories

    def summary(self, save_id: str) -> ChatTimingSummary:
        mode = turn_responsiveness_mode(self.repositories, save_id=save_id)
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="chat",
        )
        if preference is None:
            return ChatTimingSummary(
                mode=mode,
                provider=None,
                model=None,
                sample_count=0,
                estimate=None,
                outcomes=_chat_timing_outcomes([]),
            )
        durations = self.repositories.list_chat_response_commit_durations(
            save_id=save_id,
            provider=preference.provider,
            model=preference.model_id,
            mode=mode,
            limit=MAX_TIMING_SAMPLES,
        )
        outcomes = self.repositories.list_chat_turn_outcomes(
            save_id=save_id,
            provider=preference.provider,
            model=preference.model_id,
            mode=mode,
            limit=MAX_TIMING_SAMPLES,
        )
        estimate = None
        if len(durations) >= MIN_TIMING_SAMPLES:
            ordered = sorted(durations)
            estimate = ChatTimingEstimate(
                p50_ms=_nearest_rank(ordered, 50),
                p95_ms=_nearest_rank(ordered, 95),
            )
        return ChatTimingSummary(
            mode=mode,
            provider=preference.provider,
            model=preference.model_id,
            sample_count=len(durations),
            estimate=estimate,
            outcomes=_chat_timing_outcomes(outcomes),
        )


def _nearest_rank(ordered_values: list[int], percentile: int) -> int:
    index = max(0, ((percentile * len(ordered_values) + 99) // 100) - 1)
    return ordered_values[index]


def _chat_timing_outcomes(records: Sequence[object]) -> ChatTimingOutcomes:
    statuses = [str(getattr(record, "status", "")) for record in records]
    success_count = statuses.count("succeeded")
    failed_count = statuses.count("failed")
    interrupted_count = statuses.count("cancelled")
    terminal_count = success_count + failed_count + interrupted_count
    fast_path_count = 0
    combined_path_count = 0
    standard_path_count = 0
    unclassified_success_count = 0
    for record in records:
        if getattr(record, "status", None) != "succeeded":
            continue
        fast_path = getattr(record, "fast_path_used", None)
        combined_path = getattr(record, "combined_path_used", None)
        if fast_path is None or combined_path is None:
            unclassified_success_count += 1
        elif fast_path:
            fast_path_count += 1
        elif combined_path:
            combined_path_count += 1
        else:
            standard_path_count += 1
    route_sample_count = (
        fast_path_count + combined_path_count + standard_path_count
    )
    return ChatTimingOutcomes(
        terminal_count=terminal_count,
        success_count=success_count,
        failed_count=failed_count,
        interrupted_count=interrupted_count,
        failure_rate=(
            (failed_count + interrupted_count) / terminal_count
            if terminal_count
            else None
        ),
        route_sample_count=route_sample_count,
        fast_path_count=fast_path_count,
        combined_path_count=combined_path_count,
        standard_path_count=standard_path_count,
        unclassified_success_count=unclassified_success_count,
    )
