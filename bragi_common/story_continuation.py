"""Shared marker and guidance for narrator-led Storyteller turns."""

from __future__ import annotations

STORY_CONTINUATION_SPEAKER_NAME = "Bragi Story Continuation"
STORY_CONTINUATION_DIRECTION = (
    "Continue the story naturally from the current moment. Choose the next "
    "logical beat from established canon and unresolved threads, keeping the "
    "current pace unless the story calls for a transition."
)


def is_story_continuation_message(message: object) -> bool:
    return (
        getattr(message, "role", None) == "player"
        and getattr(message, "speaker_name", None)
        == STORY_CONTINUATION_SPEAKER_NAME
    )
