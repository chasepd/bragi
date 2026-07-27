from bragi.persistence.context_provenance import merge_context_source_metadata


def test_merge_context_source_metadata_keeps_legacy_derivations_separate() -> None:
    merged = merge_context_source_metadata(
        {"source_message_ids": ["message-hidden"]},
        {"source_message_ids": ["message-visible"]},
    )

    assert merged["source_message_ids"] == [
        "message-hidden",
        "message-visible",
    ]
    assert merged["source_provenance_groups"] == [
        ["message-hidden"],
        ["message-visible"],
    ]
    assert merged["source_provenance_mode"] == "any"


def test_merge_context_source_metadata_preserves_first_on_union_overflow() -> None:
    first_ids = [f"message-{index:02d}" for index in range(40)]
    second_ids = [f"message-{index:02d}" for index in range(40, 80)]

    merged = merge_context_source_metadata(
        {"source_message_ids": first_ids},
        {"source_message_ids": second_ids},
    )

    assert merged["source_message_ids"] == first_ids
    assert merged["source_provenance_groups"] == [first_ids]


def test_merge_context_source_metadata_keeps_duplicate_derivations_alternative(
) -> None:
    merged = merge_context_source_metadata(
        {
            "source_provenance_groups": [
                ["message-hidden-one"],
                ["message-hidden-two"],
            ],
            "source_provenance_mode": "all",
        },
        {
            "source_provenance_groups": [["message-visible"]],
            "source_provenance_mode": "any",
        },
    )

    assert merged["source_provenance_groups"] == [
        ["message-hidden-one", "message-hidden-two"],
        ["message-visible"],
    ]
    assert merged["source_provenance_mode"] == "any"


def test_merge_context_source_metadata_preserves_large_conjunctive_derivation(
) -> None:
    first_group = [f"message-a-{index:02d}" for index in range(40)]
    second_group = [f"message-b-{index:02d}" for index in range(40)]

    merged = merge_context_source_metadata(
        {
            "source_provenance_groups": [first_group, second_group],
            "source_provenance_mode": "all",
        },
        {
            "source_provenance_groups": [["message-alternative"]],
            "source_provenance_mode": "any",
        },
    )

    assert merged["source_provenance_groups"] == [first_group, second_group]
    assert merged["source_provenance_mode"] == "all"


def test_merge_context_source_metadata_does_not_broaden_first_audience() -> None:
    merged = merge_context_source_metadata(
        {"audience_character_ids": ["character-a"]},
        {"audience_character_ids": ["character-b"]},
    )

    assert merged["audience_character_ids"] == ["character-a"]
