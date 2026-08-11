from __future__ import annotations

from enum import StrEnum


class EpistemicStatus(StrEnum):
    OBJECTIVE_OUTCOME = "objective_outcome"
    CLAIM = "claim"
    BELIEF = "belief"
    REPORTED_SPEECH = "reported_speech"
    INTENTION = "intention"
    ATTEMPTED_ACTION = "attempted_action"
    LEGACY_UNCLASSIFIED = "legacy_unclassified"


EPISTEMIC_STATUSES = tuple(status.value for status in EpistemicStatus)


def normalize_epistemic_status(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in EPISTEMIC_STATUSES:
        raise ValueError(f"Unsupported epistemic status: {value}")
    return normalized


def format_epistemic_fact(
    body: str,
    *,
    status: str,
    actor_id: str | None = None,
    actor_name: str = "",
) -> str:
    actor = actor_name.strip() or (actor_id or "")
    qualifier = f"epistemic status: {status}"
    if actor:
        qualifier += f"; attributed actor: {actor}"
    return f"[{qualifier}] {body}"
