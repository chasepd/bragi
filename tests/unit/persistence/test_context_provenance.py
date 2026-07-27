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
