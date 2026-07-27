"""Typed interaction semantics shared by scenarios, saves, and runtime services."""

from __future__ import annotations

from enum import StrEnum


class InteractionMode(StrEnum):
    ROLEPLAY = "roleplay"
    STORYTELLER = "storyteller"


def normalize_interaction_mode(
    value: InteractionMode | str | None,
) -> InteractionMode:
    if value is None or value == "":
        return InteractionMode.ROLEPLAY
    try:
        return InteractionMode(value)
    except ValueError as exc:
        raise ValueError(f"Unknown interaction mode: {value}") from exc
