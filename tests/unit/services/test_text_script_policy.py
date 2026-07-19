from __future__ import annotations

from bragi.services.text_script_policy import (
    SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
    SCRIPT_GUARD_MODE_OFF,
    SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT,
    allowed_generated_scripts,
    sanitize_script_guard_mode,
    text_script_violations,
)


def test_source_aware_policy_rejects_generated_han_not_present_in_source() -> None:
    allowed = allowed_generated_scripts(
        ("Mara likes concise, grounded narration.",)
    )

    violations = text_script_violations(
        "玩家喜欢简洁叙事。",
        allowed_scripts=allowed,
        mode=SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT,
        field_name="memory_body",
    )

    assert [(violation.field_name, violation.script) for violation in violations] == [
        ("memory_body", "Han")
    ]


def test_source_aware_policy_allows_script_present_in_source() -> None:
    allowed = allowed_generated_scripts(("玩家说她喜欢简洁叙事。",))

    assert (
        text_script_violations(
            "玩家喜欢简洁叙事。",
            allowed_scripts=allowed,
            mode=SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT,
            field_name="memory_body",
        )
        == ()
    )


def test_latin_only_mode_rejects_non_latin_even_when_source_contains_it() -> None:
    allowed = allowed_generated_scripts(("玩家说她喜欢简洁叙事。",))

    violations = text_script_violations(
        "玩家喜欢简洁叙事。",
        allowed_scripts=allowed,
        mode=SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
        field_name="memory_body",
    )

    assert [violation.script for violation in violations] == ["Han"]


def test_off_mode_allows_unexpected_script() -> None:
    allowed = allowed_generated_scripts(())

    assert (
        text_script_violations(
            "玩家喜欢简洁叙事。",
            allowed_scripts=allowed,
            mode=SCRIPT_GUARD_MODE_OFF,
            field_name="memory_body",
        )
        == ()
    )


def test_script_guard_mode_sanitizer_defaults_to_source_aware() -> None:
    assert sanitize_script_guard_mode("latin_only_reject") == "latin_only_reject"
    assert sanitize_script_guard_mode("unexpected") == (
        SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT
    )
