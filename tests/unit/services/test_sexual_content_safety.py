from __future__ import annotations

import pytest

from bragi.safety import validate_safety_transition
from bragi.services.sexual_content_safety import (
    CONTENT_FILTER_TRANSITION,
    FADE_TO_BLACK_TRANSITION,
    SexualContentClassification,
    classify_sexual_content,
    sanitize_narrator_body,
    validate_image_prompt,
)


@pytest.mark.parametrize(
    "body",
    (
        "They kissed beneath the lanterns and held hands on the walk home.",
        "Their flirtation made the long train ride feel shorter.",
        "The scene discussed sexual health in calm, practical terms.",
        "The guide explains oral sex safety in calm, practical terms.",
    ),
)
def test_classifies_ordinary_romance_and_isolated_terms_as_acceptable(
    body: str,
) -> None:
    assert classify_sexual_content(body) is SexualContentClassification.ACCEPTABLE


@pytest.mark.parametrize(
    "body",
    (
        "Their hands slipped beneath each other's clothes.",
        "He undressed her slowly, and the room went quiet.",
        "They made love as rain tapped at the window.",
        "He slid inside her as the scene faded.",
        "He entered her slowly as the scene faded.",
    ),
)
def test_classifies_sexual_escalation_as_fade_needed(body: str) -> None:
    assert (
        classify_sexual_content(body)
        is SexualContentClassification.FADE_NEEDED
    )


@pytest.mark.parametrize(
    "body",
    (
        "He thrust into her as she cried out.",
        "The prose described oral sex in graphic detail.",
        "She reached orgasm as the scene faded.",
        "He thrusts into her as she cries out.",
        "She orgasms as the scene fades.",
        "She has sex with him.",
        "A sex act on a bed.",
        "They described sexual acts in detail.",
        "They had rough sex all night.",
        "They started to have passionate sex.",
        "They began sexual intercourse in full view.",
        "They started sex after the door closed.",
        "The adults were copulating in the bedroom.",
        "He forced his cock inside her and came inside her.",
        "He forced himself inside her.",
        "He came all over her.",
    ),
)
def test_classifies_graphic_sexual_acts_as_explicit(body: str) -> None:
    assert (
        classify_sexual_content(body)
        is SexualContentClassification.EXPLICIT
    )


def test_normalization_is_case_and_punctuation_insensitive_but_boundary_aware() -> None:
    assert (
        classify_sexual_content(
            "Their hands SLIPPED-BENEATH each other's clothes!"
        )
        is SexualContentClassification.FADE_NEEDED
    )
    assert (
        classify_sexual_content("The word undressingday is not a scene.")
        is SexualContentClassification.ACCEPTABLE
    )


def test_sanitization_replaces_rejected_body_without_retaining_draft() -> None:
    result = sanitize_narrator_body("He thrust into her as she cried out.")

    assert result.classification is SexualContentClassification.EXPLICIT
    assert result.transition_applied is True
    assert result.body == FADE_TO_BLACK_TRANSITION
    assert "thrust" not in result.body


def test_sanitization_replaces_modified_sex_phrase_for_pg_account() -> None:
    result = sanitize_narrator_body(
        "They had rough sex all night.",
        content_rating="pg",
        fade_to_black_enabled=True,
    )

    assert result.classification is SexualContentClassification.EXPLICIT
    assert result.transition_applied is True
    assert result.body == FADE_TO_BLACK_TRANSITION


def test_sanitization_is_idempotent_and_contains_hours_later_signal() -> None:
    first = sanitize_narrator_body("Their hands slipped beneath each other's clothes.")
    second = sanitize_narrator_body(first.body)

    assert first.body == FADE_TO_BLACK_TRANSITION
    assert "Hours later" in first.body
    assert second.body == first.body
    assert second.transition_applied is False
    assert second.classification is SexualContentClassification.ACCEPTABLE


def test_content_rating_and_fade_toggle_both_apply_to_narrator_prose() -> None:
    pg_result = sanitize_narrator_body(
        "They undressed before the fire.",
        content_rating="pg",
        fade_to_black_enabled=False,
    )
    adult_result = sanitize_narrator_body(
        "They had sex after returning to the inn.",
        content_rating="r",
        fade_to_black_enabled=False,
    )
    adult_fade_result = sanitize_narrator_body(
        "They had sex after returning to the inn.",
        content_rating="r",
        fade_to_black_enabled=True,
    )

    assert pg_result.transition_applied is True
    assert pg_result.body == FADE_TO_BLACK_TRANSITION
    assert adult_result.transition_applied is False
    assert adult_result.body == "They had sex after returning to the inn."
    assert adult_fade_result.transition_applied is True


def test_nonsexual_rating_violation_uses_neutral_content_transition() -> None:
    result = sanitize_narrator_body(
        "The blast dismembered the soldier in graphic detail.",
        content_rating="pg",
        fade_to_black_enabled=False,
    )

    assert result.transition_applied is True
    assert result.body == CONTENT_FILTER_TRANSITION
    assert "Hours later" not in result.body


def test_direct_graphic_violence_is_filtered_for_pg_accounts() -> None:
    result = sanitize_narrator_body(
        "He chopped off the prisoner's head and limbs, spraying the walls red.",
        content_rating="pg",
        fade_to_black_enabled=True,
    )

    assert result.transition_applied is True
    assert result.body == CONTENT_FILTER_TRANSITION


def test_exposed_brains_are_filtered_for_pg_accounts() -> None:
    result = sanitize_narrator_body(
        "He crushed her skull until her brains spilled across the floor.",
        content_rating="pg",
        fade_to_black_enabled=True,
    )

    assert result.transition_applied is True
    assert result.body == CONTENT_FILTER_TRANSITION


@pytest.mark.parametrize(
    "prompt",
    (
        "He thrust into her as the lanterns went dark.",
        "Their hands slipped beneath each other's clothes.",
    ),
)
def test_image_prompt_validation_rejects_intimate_content(prompt: str) -> None:
    with pytest.raises(
        ValueError,
        match="Image prompts cannot contain intimate sexual content",
    ):
        validate_image_prompt(prompt)


def test_image_prompt_validation_allows_ordinary_romance() -> None:
    prompt = "They kiss beneath the lanterns beside the quiet bridge."

    assert validate_image_prompt(prompt) == prompt


def test_image_prompt_validation_applies_the_selected_content_rating() -> None:
    romantic_prompt = "They kiss beneath the lanterns beside the quiet bridge."
    intimate_prompt = "They undressed before the bedroom fire."

    assert (
        validate_image_prompt(romantic_prompt, allowed_rating="pg") == romantic_prompt
    )
    with pytest.raises(ValueError, match="selected content rating"):
        validate_image_prompt(romantic_prompt, allowed_rating="g")
    with pytest.raises(ValueError, match="selected content rating"):
        validate_image_prompt(intimate_prompt, allowed_rating="pg")
    assert (
        validate_image_prompt(intimate_prompt, allowed_rating="r")
        == intimate_prompt
    )


@pytest.mark.parametrize(
    "prompt",
    (
        "Their hands slipped beneath each other's clothes.",
        "He removed her clothes beside the bedroom fire.",
        "They made love as rain tapped at the window.",
        "They were grinding against each other on the bed.",
        "She touched his groin beneath the blankets.",
        "A topless adult poses in a bedroom.",
        "An erotic portrait emphasizing cleavage.",
        "A close-up of nipples and buttocks.",
        "She has sex with him.",
        "A sex act on a bed.",
        "They described sexual acts in detail.",
        "He forced his cock inside her and came inside her.",
        "They had rough sex all night.",
        "They started to have passionate sex.",
        "They began sexual intercourse in full view.",
        "They started sex after the door closed.",
        "The adults were copulating in the bedroom.",
    ),
)
def test_rated_image_prompt_validation_preserves_intimate_safeguards(
    prompt: str,
) -> None:
    with pytest.raises(ValueError, match="selected content rating"):
        validate_image_prompt(prompt, allowed_rating="pg-13")

    assert validate_image_prompt(prompt, allowed_rating="r") == prompt


def test_legacy_canonical_body_is_recognized_as_safe_transition() -> None:
    from bragi.services.sexual_content_safety import is_fade_to_black_message

    assert is_fade_to_black_message(
        role="narrator",
        body=FADE_TO_BLACK_TRANSITION,
    )
    assert not is_fade_to_black_message(
        role="player",
        body=FADE_TO_BLACK_TRANSITION,
    )


def test_safety_transition_marker_is_allowlisted_for_narrators() -> None:
    assert validate_safety_transition("fade_to_black", role="narrator") == (
        "fade_to_black"
    )
    assert validate_safety_transition("content_filter", role="narrator") == (
        "content_filter"
    )
    with pytest.raises(ValueError):
        validate_safety_transition("fade_to_black", role="player")
    with pytest.raises(ValueError):
        validate_safety_transition("content_filter", role="player")
