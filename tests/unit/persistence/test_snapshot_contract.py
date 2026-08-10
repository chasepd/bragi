from __future__ import annotations

from bragi.persistence.snapshot_contract import (
    SNAPSHOT_TABLE_NAMES,
    SNAPSHOT_TABLES,
    SNAPSHOT_TABLES_BY_NAME,
)


def test_snapshot_table_registry_has_unique_names_and_row_keys() -> None:
    assert len(SNAPSHOT_TABLE_NAMES) == len(set(SNAPSHOT_TABLE_NAMES))
    assert set(SNAPSHOT_TABLES_BY_NAME) == set(SNAPSHOT_TABLE_NAMES)
    assert all(table.primary_key for table in SNAPSHOT_TABLES)
    assert (
        SNAPSHOT_TABLES_BY_NAME["context_observation_curation_state"].primary_key
        == "observation_id"
    )
    assert (
        SNAPSHOT_TABLES_BY_NAME["narrator_phone_activity_cursors"].primary_key
        == "narrator_message_id"
    )
