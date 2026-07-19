"""Cadence policy for automatic state pruning."""

from __future__ import annotations

from dataclasses import dataclass

from bragi.persistence.models import JobRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.model_preferences import roleplay_model_preference

STATE_PRUNING_TURN_INTERVAL = 5


@dataclass(frozen=True)
class StatePruningScheduleDecision:
    due: bool
    narrator_turn_count: int
    turn_interval: int = STATE_PRUNING_TURN_INTERVAL
    last_pruning_turn_count: int | None = None
    reason: str | None = None

    @property
    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "automatic": True,
            "narrator_turn_count": self.narrator_turn_count,
            "turn_interval": self.turn_interval,
        }
        if self.last_pruning_turn_count is not None:
            payload["last_pruning_turn_count"] = self.last_pruning_turn_count
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


def state_pruning_schedule_decision(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    turn_interval: int = STATE_PRUNING_TURN_INTERVAL,
) -> StatePruningScheduleDecision:
    narrator_turn_count = _completed_narrator_turn_count(repositories, save_id)
    if (
        roleplay_model_preference(
            repositories=repositories,
            save_id=save_id,
            purpose="state_pruning",
        )
        is None
    ):
        return StatePruningScheduleDecision(
            due=False,
            narrator_turn_count=narrator_turn_count,
            turn_interval=turn_interval,
            reason="missing_model_preference",
        )
    if narrator_turn_count < turn_interval:
        return StatePruningScheduleDecision(
            due=False,
            narrator_turn_count=narrator_turn_count,
            turn_interval=turn_interval,
            reason="below_interval",
        )

    latest = _latest_state_pruning_job(repositories, save_id)
    if latest is None:
        return StatePruningScheduleDecision(
            due=True,
            narrator_turn_count=narrator_turn_count,
            turn_interval=turn_interval,
            reason="first_interval_reached",
        )
    if latest.status in {"queued", "running"}:
        return StatePruningScheduleDecision(
            due=False,
            narrator_turn_count=narrator_turn_count,
            turn_interval=turn_interval,
            last_pruning_turn_count=_job_narrator_turn_count(latest),
            reason="state_pruning_job_active",
        )

    last_turn_count = _job_narrator_turn_count(latest)
    if last_turn_count is None:
        last_turn_count = _completed_narrator_turns_at_job_time(
            repositories,
            save_id,
            latest,
        )
    if narrator_turn_count - last_turn_count < turn_interval:
        return StatePruningScheduleDecision(
            due=False,
            narrator_turn_count=narrator_turn_count,
            turn_interval=turn_interval,
            last_pruning_turn_count=last_turn_count,
            reason="interval_not_reached",
        )
    return StatePruningScheduleDecision(
        due=True,
        narrator_turn_count=narrator_turn_count,
        turn_interval=turn_interval,
        last_pruning_turn_count=last_turn_count,
        reason="interval_reached",
    )


def _completed_narrator_turn_count(
    repositories: PersistenceRepositories,
    save_id: str,
) -> int:
    count_messages = getattr(repositories, "count_active_messages_by_role", None)
    if callable(count_messages):
        return int(count_messages(save_id, roles=("narrator",))["narrator"])
    return sum(
        1
        for message in repositories.list_messages(save_id)
        if message.role == "narrator"
    )


def _latest_state_pruning_job(
    repositories: PersistenceRepositories,
    save_id: str,
) -> JobRecord | None:
    jobs = repositories.list_recent_jobs(
        save_id=save_id,
        types=("state_pruning",),
        seconds=0,
        limit=1,
    )
    return jobs[0] if jobs else None


def _job_narrator_turn_count(job: JobRecord) -> int | None:
    for source in (job.payload, job.result or {}):
        value = source.get("narrator_turn_count")
        if isinstance(value, int):
            return value
    return None


def _completed_narrator_turns_at_job_time(
    repositories: PersistenceRepositories,
    save_id: str,
    job: JobRecord,
) -> int:
    cutoff = job.completed_at or job.started_at
    if cutoff is None:
        return 0
    count_messages = getattr(repositories, "count_active_messages_by_role", None)
    if callable(count_messages):
        return int(
            count_messages(
                save_id,
                roles=("narrator",),
                created_at_lte=cutoff,
            )["narrator"]
        )
    return sum(
        1
        for message in repositories.list_messages(save_id)
        if message.role == "narrator"
        and message.created_at is not None
        and message.created_at <= cutoff
    )
