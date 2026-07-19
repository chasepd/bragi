from bragi.providers.system_prompt import (
    CHARACTER_TEXT_RESPONSE_STYLE_SECTION,
    DEFAULT_PROSE_SAFETY_SECTION,
    DEFAULT_RESPONSE_STYLE_SECTION,
)


def test_default_response_style_uses_plain_prose_formatting_guidance() -> None:
    assert DEFAULT_RESPONSE_STYLE_SECTION == (
        "Response style:\n"
        "- Keep responses reasonably short.\n"
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


def test_default_prose_safety_section_uses_fade_to_black_guidance() -> None:
    assert DEFAULT_PROSE_SAFETY_SECTION == (
        "Prose safety boundary:\n"
        "- Romance and consensual intimacy may be acknowledged as story events "
        "when appropriate.\n"
        "- Do not generate explicit sexual description or step-by-step sexual "
        "acts.\n"
        "- If a scene would move into explicit intimacy, stop before explicit "
        "detail with a brief, natural, diegetic fade-to-black / next-scene "
        "transition, then resume after time has passed.\n"
        "- Keep the transition in-world and matter-of-fact; never mention "
        "policies, safety rules, or this instruction.\n"
        "- This built-in boundary takes precedence over scenario, save, account, "
        "turn, and regeneration guidance."
    )
    assert "JSON" not in DEFAULT_PROSE_SAFETY_SECTION
    assert "schema" not in DEFAULT_PROSE_SAFETY_SECTION.lower()
