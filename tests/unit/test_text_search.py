from bragi.text_search import (
    cjk_lexical_anchors,
    identifier_filter_matches,
    structured_identifier_filter,
    structured_identifiers,
)


def test_cjk_lexical_anchors_split_script_runs_and_preserve_entities() -> None:
    anchors = set(cjk_lexical_anchors("月石の羅針盤はどこ 李"))

    assert {"月石", "羅針盤", "羅針", "針盤", "李"} <= anchors
    assert "の" not in anchors
    assert "は" not in anchors


def test_cjk_lexical_anchors_bound_oversized_runs() -> None:
    anchors = cjk_lexical_anchors("秘密" * 100_000)

    assert anchors
    assert max(map(len, anchors)) <= 64


def test_cjk_lexical_anchors_prioritize_middle_bigrams_before_trigrams() -> None:
    body = "".join(chr(0x4E00 + index) for index in range(220))

    anchors = cjk_lexical_anchors(body)

    assert body[200:202] in anchors[:255]


def test_structured_identifiers_preserve_full_normalized_boundaries() -> None:
    assert structured_identifiers("Use Ａ－７, not A-7.5 or A-7-B.") == (
        "a-7",
        "a-7.5",
        "a-7-b",
    )


def test_structured_identifier_filter_matches_middle_identifier() -> None:
    body = " ".join(f"CODE-{index:03d}" for index in range(257))
    identifier_filter = structured_identifier_filter("", body)

    assert identifier_filter_matches(identifier_filter, '["code-128"]') == 1
    assert identifier_filter_matches(identifier_filter, '["missing-999"]') == 0


def test_unicode_normalization_preserves_combining_sequence_at_old_chunk_edge() -> None:
    value = (" " * 1023) + "A\u030A-7"

    assert "å-7" in structured_identifiers(value)


def test_unicode_normalization_preserves_tail_after_compatibility_expansion() -> None:
    value = ("\ufdfa " * 4_000) + "TARGET-9999"

    assert "target-9999" in structured_identifiers(value)


def test_unicode_normalization_preserves_middle_after_compatibility_expansion() -> None:
    value = (
        ("\ufdfa " * 2_000)
        + " TARGET-9999 "
        + ("\ufdfa " * 2_000)
    )

    assert "target-9999" in structured_identifiers(value)


def test_structured_identifier_bounds_do_not_fabricate_partial_identifier() -> None:
    value = (" " * 65_520) + "SECRET-42XYZEXTRA"

    identifiers = structured_identifiers(value)

    assert "secret-42xyzextr" not in identifiers
    assert "secret-42xyzextra" in identifiers


def test_structured_identifier_bounds_preserve_combining_boundary_token() -> None:
    prefix = (" " * 32_761) + "CODE-A\u030A"
    value = prefix + (" " * (65_537 - len(prefix)))

    identifiers = structured_identifiers(value)

    assert "code-a" not in identifiers
    assert "code-å" in identifiers


def test_structured_identifier_bounds_drop_split_compatibility_token() -> None:
    circled_a_run = "\u24b6" * 1_000
    value = (" " * 37_000) + circled_a_run + "-7" + (" " * 32_000)

    assert structured_identifiers(value) == ()
