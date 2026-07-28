from types import SimpleNamespace

from bragi_common.story_continuation import (
    STORY_CONTINUATION_DIRECTION,
    STORY_CONTINUATION_SPEAKER_NAME,
    is_story_continuation_message,
)


def test_story_continuation_marker_requires_player_role_and_internal_speaker() -> None:
    continuation = SimpleNamespace(
        role="player",
        speaker_name=STORY_CONTINUATION_SPEAKER_NAME,
        body=STORY_CONTINUATION_DIRECTION,
    )

    assert is_story_continuation_message(continuation) is True
    assert is_story_continuation_message(
        SimpleNamespace(role="narrator", speaker_name=continuation.speaker_name)
    ) is False
    assert is_story_continuation_message(
        SimpleNamespace(role="player", speaker_name="Player")
    ) is False
    assert is_story_continuation_message(
        SimpleNamespace(
            role="player",
            speaker_name=STORY_CONTINUATION_SPEAKER_NAME,
            body="A user-authored message",
        )
    ) is False
    assert "Continue the story naturally" in STORY_CONTINUATION_DIRECTION
