"""Privacy-safe recent chat response timing summaries."""

from __future__ import annotations

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
class ChatTimingSummary:
    mode: str
    provider: str | None
    model: str | None
    sample_count: int
    estimate: ChatTimingEstimate | None


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
            )
        durations = self.repositories.list_chat_response_commit_durations(
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
        )


def _nearest_rank(ordered_values: list[int], percentile: int) -> int:
    index = max(0, ((percentile * len(ordered_values) + 99) // 100) - 1)
    return ordered_values[index]
