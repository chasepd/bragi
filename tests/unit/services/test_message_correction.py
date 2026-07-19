from __future__ import annotations

from bragi.services.message_correction import (
    MessageCorrectionContext,
    correction_context_text,
)


def test_correction_context_omits_full_previous_body_and_bounds_diff() -> None:
    text = correction_context_text(
        MessageCorrectionContext(
            message_id="message-1",
            previous_body="secret old narrator text that should stay local",
            new_body="The beacon stays steady.",
            diff_unified="-old\n+new\n" * 1000,
        )
    )

    assert "secret old narrator text" not in text
    assert "The beacon stays steady." in text
    assert "[diff truncated]" in text
    assert len(text) < 5000


def test_correction_context_describes_player_message_edits() -> None:
    text = correction_context_text(
        MessageCorrectionContext(
            message_id="player-1",
            previous_body="I open the sealed door.",
            new_body="I keep the sealed door shut.",
            diff_unified="-I open the sealed door.\n+I keep the sealed door shut.\n",
            message_role="player",
        )
    )

    assert "Player message correction:" in text
    assert "Edited player text:" in text
    assert "Previous player text is intentionally omitted" in text
    assert "I keep the sealed door shut." in text
    assert "edited player message" in text
