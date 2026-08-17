import pytest

from bragi.providers.system_prompt import (
    CHARACTER_TEXT_RESPONSE_STYLE_SECTION,
    DEFAULT_RESPONSE_STYLE_SECTION,
    prose_safety_section,
)


def test_default_response_style_uses_plain_prose_formatting_guidance() -> None:
    assert DEFAULT_RESPONSE_STYLE_SECTION == (
        "Response style:\n"
        "- Keep responses reasonably short.\n"
        "- Prefer plain, concrete, economical prose. State what happens and "
        "let events carry the weight.\n"
        "- Avoid purple prose, melodrama, and clichés. Do not stack adjectives "
        "or adverbs, and do not describe ordinary actions as if they were epic.\n"
        "- Put dialogue in quotation marks.\n"
        "- Put non-dialogue narration in italics.\n"
        "- Format text messages with > at the beginning of each message."
    )
    assert "JSON" not in DEFAULT_RESPONSE_STYLE_SECTION
    assert "schema" not in DEFAULT_RESPONSE_STYLE_SECTION.lower()


def test_character_text_response_style_uses_plain_message_guidance() -> None:
    assert CHARACTER_TEXT_RESPONSE_STYLE_SECTION == (
        "Character text response style:\n"
        "- Send only the message body, like a normal phone text.\n"
        "- Do not prefix the reply with >.\n"
        "- Do not wrap the whole reply in quotation marks.\n"
        "- Do not use Markdown, italics, action narration, or sender labels.\n"
        "- Do not include timestamps or Sent at labels."
    )
    assert "JSON" not in CHARACTER_TEXT_RESPONSE_STYLE_SECTION
    assert "schema" not in CHARACTER_TEXT_RESPONSE_STYLE_SECTION.lower()


@pytest.mark.parametrize(
    ("rating", "distinctive_instruction"),
    (
        ("g", "Any profanity beyond extremely mild exclamations"),
        ("pg", "Strong profanity or slurs"),
        ("pg-13", "Explicitly described sexual activity"),
        ("r", "Pornographic or explicitly erotic depictions"),
    ),
)
def test_prose_safety_section_uses_rating_specific_ceiling(
    rating: str,
    distinctive_instruction: str,
) -> None:
    section = prose_safety_section(
        content_rating=rating,
        fade_to_black_enabled=True,
    )

    assert distinctive_instruction.casefold() in section.casefold()
    assert "JSON" not in section
    assert "schema" not in section.lower()


def test_unrated_prose_has_no_safety_or_fade_instruction() -> None:
    section = prose_safety_section(
        content_rating="unrated",
        fade_to_black_enabled=True,
    )

    assert section == ""
