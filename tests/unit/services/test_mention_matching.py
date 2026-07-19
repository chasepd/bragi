from __future__ import annotations

from bragi.services.mention_matching import character_name_is_mentioned


def test_character_name_mention_does_not_match_inside_unrelated_words() -> None:
    assert not character_name_is_mentioned(
        name="Archivist Ren",
        aliases=("Ren",),
        text="I check the rendezvous note and the glinting lantern.",
    )


def test_character_name_mention_matches_punctuation_possessive_and_case() -> None:
    assert character_name_is_mentioned(
        name="Archivist Ren",
        aliases=("Ren",),
        text="REN's ledger is open on the desk.",
    )
    assert character_name_is_mentioned(
        name="Archivist Ren",
        aliases=("Ren",),
        text="Where is Ren?",
    )


def test_character_name_mention_supports_multi_word_aliases() -> None:
    assert character_name_is_mentioned(
        name="Mael",
        aliases=("Sealed Archivist",),
        text="I ask the Sealed Archivist about the red index.",
    )


def test_character_name_mention_does_not_match_hyphenated_compounds() -> None:
    assert not character_name_is_mentioned(
        name="Archivist Ren",
        aliases=("Ren",),
        text="The Ren-like seal is probably a forgery.",
    )
