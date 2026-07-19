"""Shared model-task lifecycle helpers."""

from __future__ import annotations


def is_retired_model_task(task: object) -> bool:
    """Return whether a persisted model task belongs to a retired feature."""

    if not isinstance(task, str):
        return False
    normalized = task.strip()
    return normalized == "chat_character_interaction" or normalized.startswith(
        "character_interaction_"
    )
