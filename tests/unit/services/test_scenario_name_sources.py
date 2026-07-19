from __future__ import annotations

from bragi.services.scenario_name_sources import (
    ordinary_name_candidate_context,
    ordinary_name_candidates,
    repeated_first_names,
)


def test_ordinary_name_candidates_are_deterministic_and_exclude_used_names() -> None:
    first = ordinary_name_candidates(
        scenario_type="dating_sim",
        section_id="romance_options",
        seed="Emily and James are already central to this story.",
        sections={"premise": "Michael waits near the ticket counter."},
        per_bucket=1000,
    )
    second = ordinary_name_candidates(
        scenario_type="dating_sim",
        section_id="romance_options",
        seed="Emily and James are already central to this story.",
        sections={"premise": "Michael waits near the ticket counter."},
        per_bucket=1000,
    )

    assert first == second
    all_names = set(first.feminine + first.masculine + first.neutral)
    assert {"Emily", "James", "Michael"}.isdisjoint(all_names)
    assert first.feminine
    assert first.masculine
    assert first.neutral


def test_ordinary_name_candidate_context_skips_fantasy_roleplay() -> None:
    context = ordinary_name_candidate_context(
        scenario_type="fantasy_roleplay",
        section_id="characters",
        seed="A moonlit court with repeating names.",
        sections={},
    )

    assert context == ""


def test_ordinary_name_candidate_context_skips_multi_genre_with_fantasy() -> None:
    context = ordinary_name_candidate_context(
        scenario_type=("fantasy_roleplay", "dating_sim"),
        section_id="romance_options",
        seed="A moonlit court with repeating names.",
        sections={},
    )

    assert context == ""


def test_ordinary_name_candidate_context_formats_optional_prompt_guidance() -> None:
    context = ordinary_name_candidate_context(
        scenario_type="dating_sim",
        section_id="romance_options",
        seed="A contemporary speed dating night.",
        sections={},
    )

    assert "Ordinary contemporary name candidates" in context
    assert "Feminine:" in context
    assert "Masculine:" in context
    assert "Neutral:" in context
    assert "Do not repeat first names" in context


def test_repeated_first_names_detects_obvious_cast_duplicates() -> None:
    duplicates = repeated_first_names(
        "Emily Carter - a violinist with a guarded smile.\n"
        "Emily Brooks - a chef who knows the host.\n"
        "Lily Chen - a photographer watching the door."
    )

    assert duplicates == ("Emily",)


def test_repeated_first_names_do_not_treat_titles_as_first_names() -> None:
    duplicates = repeated_first_names(
        "Dr. Mara Voss - mission linguist and acting contact lead.\n"
        "Dr. Nia Sol - xenobiologist tracking contamination risk.\n"
        "Commander Reyes - cautious mission commander."
    )

    assert duplicates == ()
