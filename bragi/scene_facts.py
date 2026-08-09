"""Typed, short-lived physical and causal scene facts."""

from __future__ import annotations

import re

SCENE_FACT_TYPES = frozenset(
    {
        "actor_position",
        "actor_pose",
        "object_possession",
        "object_location",
        "ongoing_action",
        "physical_constraint",
        "environment_state",
        "line_of_sight",
        "pending_reaction",
    }
)
SCENE_FACT_SUBJECT_TYPES = frozenset({"character", "object", "environment"})
SCENE_FACT_TARGET_TYPES = frozenset(
    {"", "character", "location", "object", "environment"}
)
TURN_LIVED_SCENE_FACT_TYPES = frozenset(
    {"ongoing_action", "line_of_sight", "pending_reaction"}
)
MAX_ACTIVE_SCENE_FACTS = 64
MAX_SCENE_FACT_MUTATIONS_PER_TURN = 24
MAX_SCENE_FACT_CONTEXT = 32


def normalize_scene_fact_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def scene_fact_lifetime(fact_type: str) -> str:
    if fact_type not in SCENE_FACT_TYPES:
        raise ValueError(f"Unsupported scene fact type: {fact_type}")
    return "turn" if fact_type in TURN_LIVED_SCENE_FACT_TYPES else "scene"


def validate_scene_fact_shape(
    *,
    fact_type: str,
    subject_type: str,
    subject_id: str | None,
    subject_label: str,
    target_type: str = "",
    target_id: str | None = None,
    target_label: str = "",
    aspect: str = "",
    value: str,
) -> None:
    if fact_type not in SCENE_FACT_TYPES:
        raise ValueError(f"Unsupported scene fact type: {fact_type}")
    if subject_type not in SCENE_FACT_SUBJECT_TYPES:
        raise ValueError(f"Unsupported scene fact subject type: {subject_type}")
    if target_type not in SCENE_FACT_TARGET_TYPES:
        raise ValueError(f"Unsupported scene fact target type: {target_type}")
    if not value.strip():
        raise ValueError("Scene fact value is required")
    if subject_type == "character" and not subject_id:
        raise ValueError("Character scene fact subjects require subject_id")
    if subject_type in {"object", "environment"} and subject_id is not None:
        raise ValueError("Scene-local object and environment subjects cannot use IDs")
    if subject_type != "character" and not subject_label.strip():
        raise ValueError("Object and environment subjects require subject_label")
    if target_type == "character" and not target_id:
        raise ValueError("Character scene fact targets require target_id")
    if target_type == "location" and not target_id:
        raise ValueError("Location scene fact targets require target_id")
    if target_type in {"", "object", "environment"} and target_id is not None:
        raise ValueError("Scene-local object and environment targets cannot use IDs")
    if target_type in {"object", "environment"} and not target_label.strip():
        raise ValueError("Object and environment targets require target_label")

    if fact_type in {
        "actor_position",
        "actor_pose",
        "ongoing_action",
        "pending_reaction",
    } and subject_type != "character":
        raise ValueError(f"{fact_type} requires a character subject")
    if fact_type in {"object_possession", "object_location"}:
        if subject_type != "object":
            raise ValueError(f"{fact_type} requires an object subject")
        if fact_type == "object_possession" and target_type != "character":
            raise ValueError("object_possession requires a character target")
    if fact_type == "line_of_sight":
        if subject_type != "character" or not target_type:
            raise ValueError("line_of_sight requires a character and target")
    if fact_type in {"physical_constraint", "environment_state"} and not aspect.strip():
        raise ValueError(f"{fact_type} requires an aspect")


def scene_fact_conflict_key(
    *,
    fact_type: str,
    subject_type: str,
    subject_id: str | None,
    subject_label: str,
    target_type: str = "",
    target_id: str | None = None,
    target_label: str = "",
    aspect: str = "",
) -> str:
    subject = _scene_fact_reference_key(subject_type, subject_id, subject_label)
    target = _scene_fact_reference_key(target_type, target_id, target_label)
    if fact_type in {"object_possession", "object_location"}:
        return f"object_placement:{subject}"
    if fact_type in {
        "actor_position",
        "actor_pose",
        "ongoing_action",
        "pending_reaction",
    }:
        return f"{fact_type}:{subject}"
    if fact_type == "line_of_sight":
        return f"{fact_type}:{subject}:{target}"
    return ":".join(
        part
        for part in (fact_type, subject, target, normalize_scene_fact_label(aspect))
        if part
    )


def _scene_fact_reference_key(
    reference_type: str,
    reference_id: str | None,
    label: str,
) -> str:
    if not reference_type:
        return ""
    value = (
        reference_id
        if reference_type in {"character", "location"}
        else normalize_scene_fact_label(label)
    )
    return f"{reference_type}:{value}"
