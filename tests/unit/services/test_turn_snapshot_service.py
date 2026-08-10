from __future__ import annotations

import base64
import json
import sqlite3
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.interaction_mode import InteractionMode
from bragi.persistence import repositories as repositories_module
from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services import turn_snapshot_service as turn_snapshot_module
from bragi.services.character_text_world_update_service import character_text_source_ref
from bragi.services.chat_bundle_service import ChatBundleService
from bragi.services.message_revision_service import MessageRevisionService
from bragi.services.save_fork_service import SaveForkService
from bragi.services.turn_snapshot_service import (
    TurnSnapshotService,
    _coalesce_remapped_snapshot_rows,
    _filter_character_text_snapshot_rows,
    _sanitize_snapshot_rows_for_safety,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_snapshot_import_coalesces_many_to_one_memory_remaps() -> None:
    rows = _coalesce_remapped_snapshot_rows(
        "memories",
        [
            {
                "id": "merged-memory",
                "save_id": "target-save",
                "body": "The moonstone opens the archive.",
                "tags_json": '["moonstone"]',
                "importance": 0.4,
                "source_message_ids_json": '["message-one"]',
                "source_observation_ids_json": '["observation-one"]',
                "archived_at": None,
            },
            {
                "id": "merged-memory",
                "save_id": "target-save",
                "body": "the moonstone opens the archive",
                "tags_json": '["archive"]',
                "importance": 0.9,
                "source_message_ids_json": '["message-two"]',
                "source_observation_ids_json": '["observation-two"]',
                "archived_at": None,
            },
        ],
    )

    assert len(rows) == 1
    assert rows[0]["importance"] == 0.9
    assert json.loads(str(rows[0]["tags_json"])) == ["moonstone", "archive"]
    assert json.loads(str(rows[0]["source_message_ids_json"])) == [
        "message-one",
        "message-two",
    ]
    assert json.loads(str(rows[0]["source_observation_ids_json"])) == [
        "observation-one",
        "observation-two",
    ]


def test_snapshot_object_decode_rejects_declared_size_bomb() -> None:
    payload = json.dumps("x" * (1024 * 1024)).encode("utf-8")

    with pytest.raises(ValueError, match="size mismatch"):
        turn_snapshot_module._decode_exported_snapshot_object(
            {
                "object_hash": "object-one",
                "kind": "table_rows",
                "encoding": turn_snapshot_module.SNAPSHOT_ENCODING,
                "payload_base64": base64.b64encode(
                    zlib.compress(payload)
                ).decode("ascii"),
                "uncompressed_size": 16,
            }
        )


def test_snapshot_object_decode_rejects_oversized_declared_object() -> None:
    with pytest.raises(ValueError, match="too large"):
        turn_snapshot_module._decode_exported_snapshot_object(
            {
                "object_hash": "object-one",
                "kind": "table_rows",
                "encoding": turn_snapshot_module.SNAPSHOT_ENCODING,
                "payload_base64": "",
                "uncompressed_size": (
                    turn_snapshot_module._MAX_SNAPSHOT_OBJECT_UNCOMPRESSED_BYTES
                    + 1
                ),
            }
        )


def test_snapshot_object_decode_rejects_json_node_bomb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        turn_snapshot_module,
        "_MAX_SNAPSHOT_OBJECT_JSON_NODES",
        4,
    )
    payload = b"[0,0,0,0]"
    kind = "row:messages"
    object_hash = turn_snapshot_module._snapshot_object_hash(
        kind=kind,
        payload=payload,
    )

    with pytest.raises(ValueError, match="Invalid snapshot object payload"):
        turn_snapshot_module._decode_exported_snapshot_object(
            {
                "object_hash": object_hash,
                "kind": kind,
                "encoding": turn_snapshot_module.SNAPSHOT_ENCODING,
                "payload_base64": base64.b64encode(
                    zlib.compress(payload)
                ).decode("ascii"),
                "uncompressed_size": len(payload),
            }
        )


def test_local_snapshot_object_rejects_oversized_declared_payload(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    snapshot = service.capture_baseline_snapshot(save.id)
    repositories.connection.execute(
        """
        UPDATE save_snapshot_objects
        SET uncompressed_size = ?
        WHERE object_hash = ?
        """,
        (
            turn_snapshot_module._MAX_SNAPSHOT_OBJECT_UNCOMPRESSED_BYTES + 1,
            snapshot.root_manifest_hash,
        ),
    )
    repositories.commit()

    with pytest.raises(ValueError, match="too large"):
        service._snapshot_manifest(snapshot)


def test_local_snapshot_tree_rejects_duplicate_node_reference(
    repositories: PersistenceRepositories,
) -> None:
    service = TurnSnapshotService(repositories)
    row_hash = service._store_object(
        kind="row:messages",
        value={"id": "message-one"},
    )
    child_hash = service._store_tree_node(
        table_name="messages",
        order_key="child",
        row_key="message-one",
        row_hash=row_hash,
        priority=2,
        left_hash=None,
        right_hash=None,
    )
    root_hash = service._store_tree_node(
        table_name="messages",
        order_key="root",
        row_key="message-root",
        row_hash=row_hash,
        priority=1,
        left_hash=child_hash,
        right_hash=child_hash,
    )

    with pytest.raises(ValueError, match="cycle"):
        service._tree_entries(table_name="messages", root_hash=root_hash)


def test_incremental_tree_mutation_rejects_excessive_depth(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TurnSnapshotService(repositories)
    row_hash = service._store_object(
        kind="row:messages",
        value={"id": "message-one", "save_id": "save-one"},
    )
    root_hash: str | None = None
    for index, order_key in enumerate(("a", "b", "c")):
        root_hash = service._store_tree_node(
            table_name="messages",
            order_key=order_key,
            row_key=f"message-{order_key}",
            row_hash=row_hash,
            priority=index,
            left_hash=root_hash,
            right_hash=None,
        )
    monkeypatch.setattr(
        turn_snapshot_module,
        "_MAX_SNAPSHOT_TREE_MUTATION_DEPTH",
        1,
    )

    with pytest.raises(ValueError, match="too deep"):
        service._tree_delete(
            table_name="messages",
            root_hash=root_hash,
            order_key="0",
        )


def test_snapshot_validation_bounds_snapshot_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(turn_snapshot_module, "_MAX_IMPORTED_SNAPSHOT_COUNT", 1)

    with pytest.raises(ValueError, match="too many snapshots"):
        turn_snapshot_module._validate_exported_snapshot_rows(
            [
                {"id": "snapshot-one", "root_manifest_hash": "manifest"},
                {"id": "snapshot-two", "root_manifest_hash": "manifest"},
            ],
            [],
        )


def test_snapshot_validation_bounds_unique_manifest_reference_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        turn_snapshot_module,
        "_MAX_SNAPSHOT_UNIQUE_ROW_OBJECTS",
        1,
    )
    objects: dict[str, dict[str, object]] = {}
    first_row_hash = turn_snapshot_module._add_snapshot_object_export(
        objects,
        kind="row:messages",
        value={"id": "message-one"},
        created_at=None,
    )
    second_row_hash = turn_snapshot_module._add_snapshot_object_export(
        objects,
        kind="row:messages",
        value={"id": "message-two"},
        created_at=None,
    )
    manifest_hash = turn_snapshot_module._add_snapshot_object_export(
        objects,
        kind="snapshot_manifest",
        value={
            "format": turn_snapshot_module.SNAPSHOT_FORMAT,
            "tables": {
                "messages": [
                    {"id": "message-one", "object_hash": first_row_hash},
                    {"id": "message-two", "object_hash": second_row_hash},
                ]
            },
        },
        created_at=None,
    )

    with pytest.raises(ValueError, match="too many unique rows"):
        turn_snapshot_module._validate_exported_snapshot_rows(
            [{"id": "snapshot-one", "root_manifest_hash": manifest_hash}],
            objects.values(),
        )


def test_snapshot_validation_bounds_aggregate_json_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        turn_snapshot_module,
        "_MAX_SNAPSHOT_TOTAL_JSON_NODES",
        30,
    )
    objects: dict[str, dict[str, object]] = {}
    row_hashes = [
        turn_snapshot_module._add_snapshot_object_export(
            objects,
            kind="row:messages",
            value={"id": f"message-{index}", "values": [0] * 10},
            created_at=None,
        )
        for index in range(2)
    ]
    manifest_hash = turn_snapshot_module._add_snapshot_object_export(
        objects,
        kind="snapshot_manifest",
        value={
            "format": turn_snapshot_module.SNAPSHOT_FORMAT,
            "tables": {
                "messages": [
                    {"id": f"message-{index}", "object_hash": row_hash}
                    for index, row_hash in enumerate(row_hashes)
                ]
            },
        },
        created_at=None,
    )

    with pytest.raises(ValueError, match="Invalid snapshot object payload"):
        turn_snapshot_module._validate_exported_snapshot_rows(
            [{"id": "snapshot-one", "root_manifest_hash": manifest_hash}],
            objects.values(),
        )


def test_snapshot_validation_allows_distinct_manifests_with_shared_rows() -> None:
    objects: dict[str, dict[str, object]] = {}
    row_hashes = [
        turn_snapshot_module._add_snapshot_object_export(
            objects,
            kind="row:messages",
            value={"id": f"message-{index}"},
            created_at=None,
        )
        for index in range(2)
    ]
    manifest_hashes = [
        turn_snapshot_module._add_snapshot_object_export(
            objects,
            kind="snapshot_manifest",
            value={
                "format": turn_snapshot_module.SNAPSHOT_FORMAT,
                "tables": {
                    "messages": [
                        {
                            "id": f"message-{index}",
                            "object_hash": row_hashes[index],
                        }
                        for index in order
                    ]
                },
            },
            created_at=None,
        )
        for order in ((0, 1), (1, 0))
    ]

    turn_snapshot_module._validate_exported_snapshot_rows(
        [
            {"id": f"snapshot-{index}", "root_manifest_hash": manifest_hash}
            for index, manifest_hash in enumerate(manifest_hashes)
        ],
        objects.values(),
    )


def test_snapshot_validation_bounds_nested_json_string_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        turn_snapshot_module,
        "_MAX_SNAPSHOT_TOTAL_JSON_NODES",
        80,
    )
    objects: dict[str, dict[str, object]] = {}
    row_hash = turn_snapshot_module._add_snapshot_object_export(
        objects,
        kind="row:state_changes",
        value={"id": "change-one", "before_json": json.dumps([0] * 100)},
        created_at=None,
    )
    manifest_hash = turn_snapshot_module._add_snapshot_object_export(
        objects,
        kind="snapshot_manifest",
        value={
            "format": turn_snapshot_module.SNAPSHOT_FORMAT,
            "tables": {
                "state_changes": [
                    {"id": "change-one", "object_hash": row_hash}
                ]
            },
        },
        created_at=None,
    )

    with pytest.raises(ValueError, match="Invalid snapshot nested JSON"):
        turn_snapshot_module._validate_exported_snapshot_rows(
            [{"id": "snapshot-one", "root_manifest_hash": manifest_hash}],
            objects.values(),
        )


def test_snapshot_validation_rejects_duplicate_row_object_references() -> None:
    objects: dict[str, dict[str, object]] = {}
    row_hash = turn_snapshot_module._add_snapshot_object_export(
        objects,
        kind="row:messages",
        value={"id": "message-one"},
        created_at=None,
    )
    manifest_hash = turn_snapshot_module._add_snapshot_object_export(
        objects,
        kind="snapshot_manifest",
        value={
            "format": turn_snapshot_module.SNAPSHOT_FORMAT,
            "tables": {
                "messages": [
                    {"id": "message-one", "object_hash": row_hash},
                    {"id": "message-one-copy", "object_hash": row_hash},
                ]
            },
        },
        created_at=None,
    )

    with pytest.raises(ValueError, match="Duplicate snapshot row object"):
        turn_snapshot_module._validate_exported_snapshot_rows(
            [{"id": "snapshot-one", "root_manifest_hash": manifest_hash}],
            objects.values(),
        )


def test_snapshot_table_signature_ignores_unused_manifest_entry_fields() -> None:
    first = {
        "tables": {
            "messages": [
                {"id": "message-one", "object_hash": "row-hash", "nonce": "one"}
            ]
        }
    }
    second = {
        "tables": {
            "messages": [
                {"id": "message-two", "object_hash": "row-hash", "nonce": "two"}
            ]
        }
    }

    assert turn_snapshot_module._snapshot_manifest_table_signature(
        first
    ) == turn_snapshot_module._snapshot_manifest_table_signature(second)


def test_snapshot_trigger_key_remapping_is_schema_aware() -> None:
    remapper = turn_snapshot_module._SnapshotRemapper(
        source_save_id="save-old",
        target_save_id="save-new",
        rows_by_table={},
        id_maps={
            "messages": {"message-old": "message-new"},
            "character_text_messages": {"text-old": "text-new"},
            "characters": {"character-old": "character-new"},
            "locations": {"location-old": "location-new"},
            "scenarios": {"scenario-old": "scenario-new"},
            "memories": {"shared-id": "memory-new"},
        },
    )

    assert remapper._remap_trigger_key(
        "ambient_random:message-old:character-old"
    ) == "ambient_random:message-new:character-new"
    assert remapper._remap_trigger_key(
        "character_intent:shared-id:basis"
    ) == "character_intent:shared-id:basis"


def test_snapshot_json_remapping_preserves_ordinary_text_equal_to_ids() -> None:
    remapper = turn_snapshot_module._SnapshotRemapper(
        source_save_id="save-old",
        target_save_id="save-new",
        rows_by_table={
            "characters": ({"id": "mara"},),
            "memories": ({"id": "memory-one"},),
        },
    )

    character = remapper.remap_row(
        "characters",
        {
            "id": "mara",
            "save_id": "save-old",
            "aliases_json": '["mara"]',
        },
    )
    memory = remapper.remap_row(
        "memories",
        {
            "id": "memory-one",
            "save_id": "save-old",
            "tags_json": '["mara"]',
        },
    )

    assert character["id"] != "mara"
    assert character["aliases_json"] == '["mara"]'
    assert memory["tags_json"] == '["mara"]'


def test_snapshot_json_remapping_updates_known_context_and_media_references() -> None:
    remapper = turn_snapshot_module._SnapshotRemapper(
        source_save_id="save-old",
        target_save_id="save-new",
        rows_by_table={},
        id_maps={
            "messages": {"message-old": "message-new"},
            "characters": {"character-old": "character-new"},
            "locations": {"location-old": "location-new"},
            "scenarios": {"scenario-old": "scenario-new"},
            "context_observations": {"observation-old": "observation-new"},
            "character_text_messages": {"text-old": "text-new"},
            "media_assets": {
                "media-old": "media-new",
                "reference-old": "reference-new",
            },
        },
    )

    context_source = remapper.remap_row(
        "context_sources",
        {
            "id": "context-old",
            "save_id": "save-old",
            "source_type": "observation",
            "source_id": "observation-old",
            "metadata_json": json.dumps({"observation_id": "observation-old"}),
        },
    )
    media = remapper.remap_row(
        "media_assets",
        {
            "id": "media-old",
            "save_id": "save-old",
            "metadata_json": json.dumps(
                {
                    "request_source_message_id": "text-old",
                    "sender_character_id": "character-old",
                    "source_media_asset_id": "reference-old",
                    "source_media_asset_ids": ["reference-old"],
                    "source_character_reference_asset_id": "reference-old",
                    "source_character_reference_asset_ids": ["reference-old"],
                    "source_character_reference_character_ids": ["character-old"],
                }
            ),
        },
    )
    character_profile = remapper.remap_row(
        "context_sources",
        {
            "id": "profile-context-old",
            "save_id": "save-old",
            "source_type": "memory",
            "source_id": "character_profile:character-old",
            "metadata_json": json.dumps({"entity_ids": ["character-old"]}),
        },
    )
    location_state = remapper.remap_row(
        "context_sources",
        {
            "id": "location-context-old",
            "save_id": "save-old",
            "source_type": "world_state",
            "source_id": "location:location-old",
            "metadata_json": json.dumps({"entity_ids": ["location-old"]}),
        },
    )
    suggestion = remapper.remap_row(
        "context_update_suggestions",
        {
            "id": "suggestion-old",
            "save_id": "save-old",
            "proposed_value_json": json.dumps(
                {
                    "source_message_id": "message-old",
                    "source_observation_id": "observation-old",
                    "location_id": "location-old",
                }
            ),
        },
    )
    scalar_location_suggestion = remapper.remap_row(
        "context_update_suggestions",
        {
            "id": "location-suggestion-old",
            "save_id": "save-old",
            "entity_type": "character",
            "field_path": "location_id",
            "proposed_value_json": json.dumps("location-old"),
        },
    )
    director_pressure = remapper.remap_row(
        "world_state",
        {
            "id": "pressure-old",
            "save_id": "save-old",
            "key": "story.director_pressure",
            "value_json": json.dumps(
                {
                    "escalation_history": [
                        {"source_message_id": "message-old"}
                    ]
                }
            ),
        },
    )
    scenario_context = remapper.remap_row(
        "context_sources",
        {
            "id": "scenario-context-old",
            "save_id": "save-old",
            "source_type": "scenario_section",
            "source_id": "scenario:scenario-old:section:lore",
            "metadata_json": json.dumps({"scenario_id": "scenario-old"}),
        },
    )
    message_context = remapper.remap_row(
        "context_sources",
        {
            "id": "message-context-old",
            "save_id": "save-old",
            "source_type": "message",
            "source_id": "message-old",
            "metadata_json": "{}",
        },
    )
    multi_message_context = remapper.remap_row(
        "context_sources",
        {
            "id": "messages-context-old",
            "save_id": "save-old",
            "source_type": "messages",
            "source_id": "message-old, character_text_message:text-old",
            "metadata_json": "{}",
        },
    )

    assert context_source["source_id"] == "observation-new"
    assert json.loads(str(context_source["metadata_json"]))["observation_id"] == (
        "observation-new"
    )
    assert json.loads(str(character_profile["metadata_json"]))["entity_ids"] == [
        "character-new"
    ]
    assert json.loads(str(location_state["metadata_json"]))["entity_ids"] == [
        "location-new"
    ]
    assert json.loads(str(suggestion["proposed_value_json"])) == {
        "source_message_id": "message-new",
        "source_observation_id": "observation-new",
        "location_id": "location-new",
    }
    assert json.loads(
        str(scalar_location_suggestion["proposed_value_json"])
    ) == "location-new"
    assert json.loads(str(director_pressure["value_json"]))[
        "escalation_history"
    ] == [{"source_message_id": "message-new"}]
    assert json.loads(str(scenario_context["metadata_json"]))["scenario_id"] == (
        "scenario-new"
    )
    assert message_context["source_id"] == "message-new"
    assert multi_message_context["source_id"] == (
        "message-new,character_text_message:text-new"
    )
    media_metadata = json.loads(str(media["metadata_json"]))
    assert media_metadata == {
        "request_source_message_id": "text-new",
        "sender_character_id": "character-new",
        "source_media_asset_id": "reference-new",
        "source_media_asset_ids": ["reference-new"],
        "source_character_reference_asset_id": "reference-new",
        "source_character_reference_asset_ids": ["reference-new"],
        "source_character_reference_character_ids": ["character-new"],
    }


def test_snapshot_knowledge_edge_merge_fails_closed_on_provenance_overflow() -> None:
    existing: dict[str, object] = {
        "knowledge_state": "knows",
        "acquisition_method": "observed",
        "confidence": 0.8,
        "source_message_ids_json": json.dumps(
            [f"message-{index:02d}" for index in range(40)]
        ),
        "archived_at": None,
    }
    incoming: dict[str, object] = {
        "knowledge_state": "knows",
        "acquisition_method": "told",
        "confidence": 0.9,
        "source_message_ids_json": json.dumps(
            [f"message-{index:02d}" for index in range(40, 80)]
        ),
        "archived_at": None,
    }

    turn_snapshot_module._merge_snapshot_knowledge_edge_rows(
        existing,
        incoming,
    )

    assert existing["knowledge_state"] == "does_not_know"
    assert existing["acquisition_method"] == "unknown"
    assert existing["source_message_ids_json"] == "[]"


def test_snapshot_context_merge_preserves_large_conjunctive_derivation() -> None:
    first_group = [f"message-a-{index:02d}" for index in range(40)]
    second_group = [f"message-b-{index:02d}" for index in range(40)]

    merged = json.loads(
        turn_snapshot_module._merged_context_source_metadata_json(
            json.dumps(
                {
                    "source_provenance_groups": [first_group, second_group],
                    "source_provenance_mode": "all",
                }
            ),
            json.dumps(
                {
                    "source_provenance_groups": [["message-alternative"]],
                    "source_provenance_mode": "any",
                }
            ),
        )
    )

    assert merged["source_provenance_groups"] == [first_group, second_group]
    assert merged["source_provenance_mode"] == "all"


def test_snapshot_context_merge_does_not_broaden_selected_body_audience() -> None:
    merged = json.loads(
        turn_snapshot_module._merged_context_source_metadata_json(
            json.dumps({"audience_character_ids": ["character-a"]}),
            json.dumps({"audience_character_ids": ["character-b"]}),
        )
    )

    assert merged["audience_character_ids"] == ["character-a"]


def test_snapshot_context_merge_rejects_conflicting_body_provenance() -> None:
    with pytest.raises(ValueError, match="Conflicting context sources"):
        turn_snapshot_module._coalesce_remapped_snapshot_rows(
            "context_sources",
            [
                {
                    "id": "source-hidden",
                    "save_id": "save-one",
                    "source_type": "memory",
                    "source_id": "memory-one",
                    "title": "Secret",
                    "body": "The hidden vault code is AMBER-77.",
                    "metadata_json": json.dumps(
                        {"source_message_ids": ["message-hidden"]}
                    ),
                    "archived_at": None,
                },
                {
                    "id": "source-visible",
                    "save_id": "save-one",
                    "source_type": "memory",
                    "source_id": "memory-one",
                    "title": "Harmless",
                    "body": "The lamps are lit.",
                    "metadata_json": json.dumps(
                        {"source_message_ids": ["message-visible"]}
                    ),
                    "archived_at": None,
                },
            ],
        )


def test_legacy_memory_normalization_preserves_other_id_namespaces() -> None:
    rows = turn_snapshot_module._normalize_legacy_snapshot_memories(
        {
            "memories": (
                {
                    "id": "memory-keeper",
                    "body": "Mara likes tea.",
                    "tags_json": "[]",
                    "importance": 0.4,
                    "source_message_ids_json": "[]",
                    "source_observation_ids_json": "[]",
                    "archived_at": None,
                },
                {
                    "id": "memory-duplicate",
                    "body": "mara likes tea",
                    "tags_json": '["memory-duplicate"]',
                    "importance": 0.9,
                    "source_message_ids_json": '["memory-duplicate"]',
                    "source_observation_ids_json": '["memory-duplicate"]',
                    "archived_at": None,
                },
            ),
            "context_observations": (
                {
                    "id": "observation-one",
                    "source_message_ids_json": '["memory-duplicate"]',
                },
            ),
            "active_threads": (
                {
                    "id": "thread-one",
                    "related_entities_json": (
                        '["memory:memory-duplicate",'
                        '"character:memory-duplicate"]'
                    ),
                },
            ),
            "character_text_proactive_triggers": (
                {
                    "id": "trigger-one",
                    "save_id": "save-one",
                    "character_id": "memory-duplicate",
                    "trigger_key": "character_intent:memory-duplicate:basis",
                    "source_type": "character",
                    "source_id": "memory-duplicate",
                },
            ),
        }
    )

    [memory] = rows["memories"]
    assert json.loads(str(memory["tags_json"])) == ["memory-duplicate"]
    assert json.loads(str(memory["source_message_ids_json"])) == [
        "memory-duplicate"
    ]
    assert json.loads(str(memory["source_observation_ids_json"])) == [
        "memory-duplicate"
    ]
    [observation] = rows["context_observations"]
    assert observation["source_message_ids_json"] == '["memory-duplicate"]'
    [thread] = rows["active_threads"]
    assert json.loads(str(thread["related_entities_json"])) == [
        "memory:memory-keeper",
        "character:memory-duplicate",
    ]
    [trigger] = rows["character_text_proactive_triggers"]
    assert trigger["trigger_key"] == "character_intent:memory-duplicate:basis"
    assert trigger["source_id"] == "memory-duplicate"


def test_snapshot_memory_normalization_preserves_epistemic_identity() -> None:
    rows = turn_snapshot_module._normalize_legacy_snapshot_memories(
        {
            "memories": (
                {
                    "id": "memory-first",
                    "body": "The north gate is unguarded.",
                    "tags_json": "[]",
                    "importance": 0.8,
                    "source_message_ids_json": "[]",
                    "source_observation_ids_json": "[]",
                    "epistemic_status": "reported_speech",
                    "epistemic_actor_id": "character-first",
                    "epistemic_actor_name": "Courier",
                    "archived_at": None,
                },
                {
                    "id": "memory-second",
                    "body": "The north gate is unguarded.",
                    "tags_json": "[]",
                    "importance": 0.8,
                    "source_message_ids_json": "[]",
                    "source_observation_ids_json": "[]",
                    "epistemic_status": "reported_speech",
                    "epistemic_actor_id": "character-second",
                    "epistemic_actor_name": "Courier",
                    "archived_at": None,
                },
            )
        }
    )

    assert [row["id"] for row in rows["memories"]] == [
        "memory-first",
        "memory-second",
    ]
    assert len({row["claim_fingerprint"] for row in rows["memories"]}) == 2


def test_legacy_memory_normalization_bounds_merged_provenance() -> None:
    rows = turn_snapshot_module._normalize_legacy_snapshot_memories(
        {
            "memories": (
                {
                    "id": "memory-keeper",
                    "body": "Mara likes tea.",
                    "tags_json": "[]",
                    "importance": 0.4,
                    "source_message_ids_json": json.dumps(
                        [f"message-{index:02d}" for index in range(40)]
                    ),
                    "source_observation_ids_json": json.dumps(
                        [f"observation-{index:02d}" for index in range(40)]
                    ),
                    "archived_at": None,
                },
                {
                    "id": "memory-duplicate",
                    "body": "mara likes tea",
                    "tags_json": "[]",
                    "importance": 0.9,
                    "source_message_ids_json": json.dumps(
                        [f"message-{index:02d}" for index in range(40, 80)]
                    ),
                    "source_observation_ids_json": json.dumps(
                        [f"observation-{index:02d}" for index in range(40, 80)]
                    ),
                    "archived_at": None,
                },
            )
        }
    )

    [memory] = rows["memories"]
    assert len(json.loads(str(memory["source_message_ids_json"]))) == 40
    assert len(json.loads(str(memory["source_observation_ids_json"]))) == 40


def test_legacy_memory_normalization_bounds_context_source_provenance() -> None:
    first_source_ids = [f"message-{index:02d}" for index in range(40)]
    second_source_ids = [f"message-{index:02d}" for index in range(40, 80)]
    rows = turn_snapshot_module._normalize_legacy_snapshot_memories(
        {
            "memories": (
                {
                    "id": "memory-keeper",
                    "body": "Mara likes tea.",
                    "tags_json": "[]",
                    "importance": 0.4,
                    "source_message_ids_json": json.dumps(first_source_ids),
                    "source_observation_ids_json": "[]",
                    "archived_at": None,
                },
                {
                    "id": "memory-duplicate",
                    "body": "mara likes tea",
                    "tags_json": "[]",
                    "importance": 0.9,
                    "source_message_ids_json": json.dumps(second_source_ids),
                    "source_observation_ids_json": "[]",
                    "archived_at": None,
                },
            ),
            "context_sources": (
                {
                    "id": "source-keeper",
                    "save_id": "save-one",
                    "source_type": "memory",
                    "source_id": "memory-keeper",
                    "metadata_json": json.dumps(
                        {"source_message_ids": first_source_ids}
                    ),
                    "archived_at": None,
                },
                {
                    "id": "source-duplicate",
                    "save_id": "save-one",
                    "source_type": "memory",
                    "source_id": "memory-duplicate",
                    "metadata_json": json.dumps(
                        {"source_message_ids": second_source_ids}
                    ),
                    "archived_at": None,
                },
            ),
        }
    )

    [source] = rows["context_sources"]
    metadata = json.loads(str(source["metadata_json"]))
    assert metadata["source_message_ids"] == first_source_ids
    assert metadata["source_provenance_groups"] == [first_source_ids]


def test_snapshot_context_source_merge_keeps_legacy_sources_in_one_group() -> None:
    merged = json.loads(
        turn_snapshot_module._merged_context_source_metadata_json(
            json.dumps(
                {
                    "source_message_ids": [
                        "message-visible",
                        "message-hidden",
                    ]
                }
            ),
            "{}",
        )
    )

    assert merged["source_provenance_groups"] == [
        ["message-visible", "message-hidden"]
    ]
    assert merged["source_provenance_mode"] == "any"


def test_legacy_duplicate_memories_restore_and_fork_after_unique_index_repair(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    first_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Mara likes tea.",
    )
    second_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Mara still likes tea.",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
    )
    repositories.add_context_observation(
        save_id=save.id,
        observation_type="preference",
        claim="Mara likes tea.",
        source_message_ids=[first_message.id],
        status="accepted",
        observation_id="observation-one",
    )
    repositories.add_context_observation(
        save_id=save.id,
        observation_type="preference",
        claim="Mara still likes tea.",
        source_message_ids=[second_message.id],
        status="accepted",
        observation_id="observation-two",
    )
    repositories.connection.execute(
        "DROP INDEX idx_memories_save_claim_fingerprint_active"
    )
    repositories.connection.executemany(
        """
        INSERT INTO memories(
            id, save_id, body, tags_json, importance,
            source_message_id, source_message_ids_json, claim_fingerprint,
            source_observation_ids_json
        )
        VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?)
        """,
        (
            (
                "memory-keeper",
                save.id,
                "Mara likes tea.",
                '["mara"]',
                0.4,
                first_message.id,
                "legacy-fingerprint-one",
                '["observation-one"]',
            ),
            (
                "memory-duplicate",
                save.id,
                "mara likes tea",
                '["tea"]',
                0.9,
                second_message.id,
                "legacy-fingerprint-two",
                '["observation-two"]',
            ),
        ),
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-keeper",
        title="Tea preference",
        body="Mara likes tea.",
        metadata={"source_message_ids": [first_message.id]},
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-duplicate",
        title="Tea preference duplicate",
        body="mara likes tea",
        metadata={"source_message_ids": [second_message.id]},
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id="memory-keeper",
        knowledge_state="knows",
        source_message_id=first_message.id,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id="memory-duplicate",
        knowledge_state="does_not_know",
        source_message_id=second_message.id,
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id="memory-keeper",
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id="memory-duplicate",
        relation="knows",
        source_message_id=second_message.id,
    )
    repositories.connection.executemany(
        """
        INSERT INTO character_text_proactive_triggers(
            id, save_id, character_id, trigger_key, trigger_type,
            source_type, source_id, source_message_id, reason
        )
        VALUES (?, ?, ?, ?, 'memory_changed', 'memory', ?, ?, ?)
        """,
        (
            (
                "trigger-keeper",
                save.id,
                character.id,
                "memory:memory-keeper",
                "memory-keeper",
                first_message.id,
                "",
            ),
            (
                "trigger-duplicate",
                save.id,
                character.id,
                "memory:memory-duplicate",
                "memory-duplicate",
                second_message.id,
                "Tea preference changed",
            ),
        ),
    )
    repositories.connection.commit()
    service = TurnSnapshotService(repositories)
    snapshot = service.capture_baseline_snapshot(save.id)
    repositories.connection.execute(
        "DELETE FROM memories WHERE save_id = ?",
        (save.id,),
    )
    repositories.connection.execute(
        """
        CREATE UNIQUE INDEX idx_memories_save_claim_fingerprint_active
        ON memories(save_id, claim_fingerprint)
        WHERE archived_at IS NULL AND claim_fingerprint != ''
        """
    )
    repositories.connection.commit()

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)
    assert repositories.continuity_index_requires_full_rebuild(save.id)
    restored = repositories.list_memories(save.id)
    fork = service.fork_snapshot_to_save(
        source_save_id=save.id,
        snapshot_id=snapshot.id,
        title="Forked legacy snapshot",
        media_dir=tmp_path / "media",
    )
    forked = repositories.list_memories(fork.save.id)

    for save_id, [memory] in ((save.id, restored), (fork.save.id, forked)):
        assert memory.body == "Mara likes tea."
        assert memory.tags == ["mara", "tea"]
        assert memory.importance == 0.9
        assert memory.claim_fingerprint
        assert len(memory.source_message_ids) == 2
        assert len(memory.source_observation_ids) == 2
        [source] = repositories.list_context_sources(
            save_id,
            source_type="memory",
        )
        assert source.source_id == memory.id
        [edge] = repositories.list_character_knowledge_edges(save_id)
        assert edge.target_id == memory.id
        assert edge.knowledge_state == "does_not_know"
        assert len(edge.source_message_ids) == 2
        [link] = repositories.list_entity_links(save_id)
        assert link.target_id == memory.id
        assert link.source_message_id is not None
        [trigger] = repositories.list_character_text_proactive_triggers(save_id)
        assert trigger.source_id == memory.id
        assert trigger.trigger_key == f"memory:{memory.id}"
        assert trigger.reason == "Tea preference changed"


def test_snapshot_coalesces_knowledge_aliases_and_scalar_provenance() -> None:
    rows = _coalesce_remapped_snapshot_rows(
        "character_knowledge_edges",
        [
            {
                "id": "edge-one",
                "save_id": "target-save",
                "character_id": "character-one",
                "target_type": "state",
                "target_id": "state-secret",
                "knowledge_state": "knows",
                "confidence": 0.9,
                "source_message_id": "message-visible",
                "source_message_ids_json": "[]",
                "archived_at": None,
            },
            {
                "id": "edge-two",
                "save_id": "target-save",
                "character_id": "character-one",
                "target_type": "world_state",
                "target_id": "state-secret",
                "knowledge_state": "does_not_know",
                "confidence": 0.7,
                "source_message_id": "message-hidden",
                "source_message_ids_json": "[]",
                "archived_at": None,
            },
        ],
    )

    assert len(rows) == 1
    assert rows[0]["target_type"] == "world_state"
    assert rows[0]["knowledge_state"] == "does_not_know"
    assert json.loads(str(rows[0]["source_message_ids_json"])) == [
        "message-visible",
        "message-hidden",
    ]


def test_snapshot_remaps_colliding_message_provenance_by_field_type() -> None:
    colliding_id = "save-and-message-collision"
    remapper = turn_snapshot_module._SnapshotRemapper(
        source_save_id=colliding_id,
        target_save_id="fork-save",
        rows_by_table={
            "messages": ({"id": colliding_id},),
            "memories": ({"id": "memory-one"}, {"id": "all"}),
            "context_sources": ({"id": "context-one"},),
        },
    )
    fork_message_id = remapper.id_maps["messages"][colliding_id]

    memory = remapper.remap_row(
        "memories",
        {
            "id": "memory-one",
            "save_id": colliding_id,
            "source_message_ids_json": json.dumps([colliding_id]),
        },
    )
    context_source = remapper.remap_row(
        "context_sources",
        {
            "id": "context-one",
            "save_id": colliding_id,
            "source_type": "memory",
            "source_id": "memory-one",
            "metadata_json": json.dumps(
                {
                    "source_message_id": colliding_id,
                    "source_message_ids": [colliding_id],
                    "source_provenance_groups": [[colliding_id]],
                    "source_provenance_mode": "all",
                }
            ),
        },
    )

    assert json.loads(str(memory["source_message_ids_json"])) == [fork_message_id]
    assert json.loads(str(context_source["metadata_json"]))[
        "source_provenance_mode"
    ] == "all"
    metadata = json.loads(str(context_source["metadata_json"]))
    assert metadata["source_message_id"] == fork_message_id
    assert metadata["source_message_ids"] == [fork_message_id]
    assert metadata["source_provenance_groups"] == [[fork_message_id]]
    assert fork_message_id != "fork-save"


def test_snapshot_remaps_summary_lineage_ids() -> None:
    remapper = turn_snapshot_module._SnapshotRemapper(
        source_save_id="source-save",
        target_save_id="fork-save",
        rows_by_table={
            "messages": ({"id": "message-one"},),
            "summaries": (
                {"id": "summary-prior"},
                {"id": "summary-active"},
            ),
        },
    )

    summary = remapper.remap_row(
        "summaries",
        {
            "id": "summary-active",
            "save_id": "source-save",
            "source_message_ids_json": json.dumps(["message-one"]),
            "source_summary_ids_json": json.dumps(["summary-prior"]),
        },
    )

    assert json.loads(str(summary["source_message_ids_json"])) == [
        remapper.id_maps["messages"]["message-one"]
    ]
    assert json.loads(str(summary["source_summary_ids_json"])) == [
        remapper.id_maps["summaries"]["summary-prior"]
    ]


def test_snapshot_rejects_unknown_context_source_provenance() -> None:
    remapper = turn_snapshot_module._SnapshotRemapper(
        source_save_id="source-save",
        target_save_id="fork-save",
        rows_by_table={
            "messages": ({"id": "known-message"},),
            "context_sources": ({"id": "context-one"},),
        },
    )

    with pytest.raises(
        ValueError,
        match="unknown (?:provenance source|world_state reference)",
    ):
        remapper.remap_row(
            "context_sources",
            {
                "id": "context-one",
                "save_id": "source-save",
                "source_type": "world_state",
                "source_id": "state-one",
                "metadata_json": json.dumps(
                    {"source_message_ids": ["missing-message"]}
                ),
            },
        )


def test_capture_dedupes_objects_and_restore_recovers_exact_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Mara sees the red lens.",
    )
    repositories.add_location(
        save_id=save.id,
        location_id="tower",
        name="Beacon Tower",
        source_message_id=message.id,
    )
    repositories.add_character(
        save_id=save.id,
        character_id="mara",
        name="Mara",
        location_id="tower",
        source_message_id=message.id,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id="mara",
        target_type="location",
        target_id="tower",
        source_message_id=message.id,
        evidence_quote="Mara sees the tower.",
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=message.id,
        character_id="mara",
        visibility="visible",
        source="scene",
    )
    repositories.replace_message_action_choices(
        save_id=save.id,
        message_id=message.id,
        choices=("Study the lens", "Signal east"),
        provider="fake",
        model="choices",
    )
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    media_dir.joinpath("lens.png").write_bytes(b"image bytes stay on disk")
    repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path="lens.png",
        prompt="red lens",
        provider="fake",
        model="image",
        status="succeeded",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
        source_message_id=message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=1,
        world_time_day_index=1,
        world_time_day_label="monday",
        world_time_phase="morning",
        world_time_clock_minutes=8 * 60 + 30,
        world_time_period_label="bell watch",
        world_time_source_message_id=message.id,
        world_time_confidence=0.93,
        source_message_id=message.id,
    )

    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    row_object_count = _row_object_count(repositories)
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)
    assert _row_object_count(repositories) == row_object_count
    _assert_snapshot_objects_do_not_store_media_bytes(repositories)

    later = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I repaint the lens blue.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "blue"},
        source_message_id=later.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Tuesday evening",
        time_of_day="evening",
        day_of_week="tuesday",
        world_day_index=2,
        world_time_day_index=2,
        world_time_day_label="tuesday",
        world_time_phase="evening",
        world_time_clock_minutes=20 * 60 + 15,
        world_time_period_label="moon watch",
        world_time_source_message_id=later.id,
        world_time_confidence=0.42,
        source_message_id=later.id,
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    assert [message.body for message in repositories.list_messages(save.id)] == [
        "Mara sees the red lens."
    ]
    assert repositories.list_world_state(save.id)[0].value == {"color": "red"}
    assert repositories.list_character_knowledge_edges(save.id)[0].target_id == "tower"
    assert repositories.list_message_visibility(save.id)[0].message_id == message.id
    choices = repositories.latest_message_action_choices(save.id)
    assert [choice.body for choice in choices] == [
        "Study the lens",
        "Signal east",
    ]
    assert repositories.list_media_assets(save.id)[0].prompt == "red lens"
    restored_snapshot = repositories.get_scene_snapshot(save.id)
    assert restored_snapshot is not None
    assert restored_snapshot.world_day_index == 1
    assert restored_snapshot.world_time_day_index == 1
    assert restored_snapshot.world_time_day_label == "monday"
    assert restored_snapshot.world_time_phase == "morning"
    assert restored_snapshot.world_time_clock_minutes == 8 * 60 + 30
    assert restored_snapshot.world_time_period_label == "bell watch"
    assert restored_snapshot.source_message_id == message.id
    assert restored_snapshot.world_time_source_message_id == message.id
    assert restored_snapshot.world_time_confidence == 0.93


def test_snapshot_restore_preserves_unrated_narration_without_resanitizing(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    adult_body = "They had sex after returning to the inn."
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=adult_body,
    )
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    repositories.append_message(
        save_id=save.id,
        role="player",
        body="I leave the inn.",
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    restored_messages = repositories.list_messages(save.id)
    assert [restored.body for restored in restored_messages] == [adult_body]
    assert restored_messages[0].safety_transition == ""


def test_snapshot_restores_curation_progress_without_reviving_worker_lease(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon was relit.",
    )
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="event",
        claim="The beacon was relit.",
        evidence_quote="The beacon was relit.",
        source_message_ids=[message.id],
        scope="durable",
        confidence=0.9,
    )
    claimed = repositories.claim_context_observations(
        (observation.id,),
        lease_token="snapshot-worker-secret",
        lease_seconds=600,
    )
    assert [row.id for row in claimed] == [observation.id]
    service = TurnSnapshotService(repositories)
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    repositories.defer_context_observation_curation(
        observation.id,
        lease_token="snapshot-worker-secret",
        error="temporary failure",
        retry_after_seconds=60,
        max_attempts=5,
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    state = repositories.get_context_observation_curation_state(observation.id)
    assert state is not None
    assert state.attempt_count == 0
    assert state.lease_token is None
    assert state.lease_until is None
    assert state.last_error is None
    assert state.terminal_outcome is None


def test_snapshot_rechecks_curation_lease_after_time_passes_without_table_write(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon was relit.",
    )
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="event",
        claim="The beacon was relit.",
        evidence_quote="The beacon was relit.",
        source_message_ids=[message.id],
        scope="durable",
        confidence=0.9,
    )
    repositories.claim_context_observations(
        (observation.id,),
        lease_token="snapshot-worker-secret",
        lease_seconds=600,
    )
    service = TurnSnapshotService(repositories)
    original = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    repositories.connection.execute(
        "DROP TRIGGER dirty_snapshot_context_observation_curation_state_after_update"
    )
    repositories.connection.execute(
        """
        UPDATE context_observation_curation_state
        SET lease_until = '2000-01-01 00:00:00'
        WHERE observation_id = ?
        """,
        (observation.id,),
    )
    repositories.connection.execute(
        """
        UPDATE save_snapshot_row_state
        SET recheck_at = '2000-01-01 00:00:00'
        WHERE save_id = ?
          AND table_name = 'context_observation_curation_state'
          AND row_key = ?
        """,
        (save.id, observation.id),
    )
    repositories.commit()

    changed = service.capture_current_head_if_dirty(save.id)

    assert changed.id != original.id
    rows = service._rows_from_manifest(service._snapshot_manifest(changed))
    [state] = rows["context_observation_curation_state"]
    assert state["attempt_count"] == 1
    assert state["lease_token"] is None
    assert state["lease_until"] is None


def test_imported_snapshot_rows_clear_curation_worker_lease() -> None:
    sanitized = _sanitize_snapshot_rows_for_safety(
        {
            "messages": (),
            "context_observations": (
                {"id": "observation-1"},
                {"id": "observation-2"},
            ),
            "context_observation_curation_state": (
                {
                    "observation_id": "observation-1",
                    "attempt_count": 5,
                    "lease_token": "crafted-worker-token",
                    "lease_until": "2999-01-01 00:00:00",
                },
                {
                    "observation_id": "observation-2",
                    "attempt_count": 5,
                    "lease_token": "expired-worker-token",
                    "lease_until": "2000-01-01 00:00:00",
                },
            ),
        }
    )

    live_state, expired_state = sanitized["context_observation_curation_state"]
    assert live_state["attempt_count"] == 4
    assert live_state["lease_token"] is None
    assert live_state["lease_until"] is None
    assert expired_state["attempt_count"] == 5
    assert expired_state["lease_token"] is None
    assert expired_state["lease_until"] is None


def test_restore_rolls_back_when_snapshot_row_insert_fails(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lens burns red.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
        source_message_id=first.id,
    )
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=first.id,
    )
    later = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I repaint the lens blue.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "blue"},
        source_message_id=later.id,
    )
    repositories.connection.execute(
        """
        CREATE TRIGGER fail_lens_restore
        BEFORE INSERT ON world_state
        WHEN NEW.key = 'lens'
        BEGIN
            SELECT RAISE(ABORT, 'forced restore failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="forced restore failure"):
        service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    assert [message.body for message in repositories.list_messages(save.id)] == [
        "The lens burns red.",
        "I repaint the lens blue.",
    ]
    assert repositories.list_world_state(save.id)[0].value == {"color": "blue"}


def test_delete_from_here_restores_snapshot_before_anchor(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    baseline = service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I light the lantern.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lantern",
        value={"status": "lit"},
        source_message_id=first.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)
    second = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I douse the lantern.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lantern",
        value={"status": "dark"},
        source_message_id=second.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=second.id)

    deletion = MessageRevisionService(repositories).delete_suffix_from_message(
        save_id=save.id,
        message_id=second.id,
    )

    assert [message.id for message in deletion.deleted_messages] == [second.id]
    assert [message.id for message in repositories.list_messages(save.id)] == [first.id]
    assert repositories.list_world_state(save.id)[0].value == {"status": "lit"}

    MessageRevisionService(repositories).delete_suffix_from_message(
        save_id=save.id,
        message_id=first.id,
    )

    assert repositories.list_messages(save.id) == []
    assert repositories.list_world_state(save.id) == []
    restored_baseline = service.latest_snapshot_for_message(
        save_id=save.id,
        message_id=None,
    )
    assert restored_baseline is not None
    assert restored_baseline.id == baseline.id


def test_restore_includes_character_text_state_and_removes_later_rows(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan sends the ticket stub.",
    )
    seeded = _seed_character_text_snapshot_state(
        repositories,
        save_id=save.id,
        source_message_id=message.id,
        media_dir=media_dir,
    )
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )

    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ignore the message.",
    )
    later_text = repositories.append_character_text_message(
        message_id="text-rowan-later",
        save_id=save.id,
        thread_id=seeded["thread_id"],
        character_id=seeded["rowan_id"],
        sender="player",
        body="Not now.",
    )
    repositories.set_character_contact_state(
        save_id=save.id,
        player_character_id=seeded["player_id"],
        character_id=seeded["rowan_id"],
        player_has_character_number=False,
        character_has_player_number=True,
        source_text_message_id=later_text.id,
    )
    repositories.add_character_text_proactive_trigger(
        save_id=save.id,
        character_id=seeded["rowan_id"],
        trigger_key=f"character_intent:{seeded['rowan_id']}:later",
        trigger_type="character_intent",
        thread_id=seeded["thread_id"],
        text_message_id=later_text.id,
        source_type="character",
        source_id=seeded["rowan_id"],
        source_message_id=message.id,
        reason="Later trigger.",
        trigger_id="trigger-rowan-later",
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    text_messages = repositories.list_character_text_messages(save_id=save.id)
    assert [text_message.id for text_message in text_messages] == [
        seeded["text_message_id"]
    ]
    revisions = repositories.list_character_text_message_revisions(save_id=save.id)
    assert [(revision.id, revision.text_message_id) for revision in revisions] == [
        ("revision-rowan-text", seeded["text_message_id"])
    ]
    attachments = repositories.list_character_text_message_attachments(save_id=save.id)
    assert [
        (attachment.id, attachment.media_asset_id) for attachment in attachments
    ] == [
        ("text-attachment-ticket", seeded["media_asset_id"])
    ]
    provenance = repositories.list_character_text_provenance(save_id=save.id)
    assert [
        (row.text_message_id, row.target_type, row.target_id) for row in provenance
    ] == [
        (seeded["text_message_id"], "memory", seeded["memory_id"])
    ]
    contacts = repositories.list_character_contact_states(save.id)
    assert len(contacts) == 1
    assert contacts[0].player_has_character_number is True
    assert contacts[0].character_has_player_number is False
    assert contacts[0].source_text_message_id == seeded["text_message_id"]
    triggers = repositories.list_character_text_proactive_triggers(save.id)
    assert [(trigger.id, trigger.text_message_id) for trigger in triggers] == [
        ("trigger-rowan", seeded["text_message_id"])
    ]
    memories = repositories.list_memories(save.id)
    assert memories[0].source_message_ids == [
        character_text_source_ref(seeded["text_message_id"])
    ]


def test_snapshot_backed_fork_remaps_rows_and_copies_media(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    media_dir.joinpath("old.png").write_bytes(b"old image")
    save = _create_save(
        repositories,
        interaction_mode=InteractionMode.STORYTELLER,
    )
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The tower lens glows red.",
    )
    repositories.add_location(
        save_id=save.id,
        location_id="tower",
        name="Beacon Tower",
        source_message_id=first.id,
    )
    repositories.add_character(
        save_id=save.id,
        character_id="mara",
        name="Mara",
        location_id="tower",
        source_message_id=first.id,
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=first.id,
        character_id="mara",
        visibility="visible",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id="mara",
        target_type="location",
        target_id="tower",
        source_message_id=first.id,
    )
    repositories.create_media_asset(
        save_id=save.id,
        source_message_id=first.id,
        type="image",
        path="old.png",
        prompt="red lens",
        provider="fake",
        model="image",
        status="succeeded",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
        source_message_id=first.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=1,
        world_time_day_index=1,
        world_time_day_label="monday",
        world_time_phase="morning",
        world_time_clock_minutes=8 * 60 + 30,
        world_time_period_label="bell watch",
        world_time_source_message_id=first.id,
        world_time_confidence=0.93,
        source_message_id=first.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)
    second = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The tower lens turns blue.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "blue"},
        source_message_id=second.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Tuesday evening",
        time_of_day="evening",
        day_of_week="tuesday",
        world_day_index=2,
        world_time_day_index=2,
        world_time_day_label="tuesday",
        world_time_phase="evening",
        world_time_clock_minutes=20 * 60 + 15,
        world_time_period_label="moon watch",
        world_time_source_message_id=second.id,
        world_time_confidence=0.42,
        source_message_id=second.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=second.id)

    result = SaveForkService(repositories).fork_from_message(
        save_id=save.id,
        message_id=first.id,
        media_dir=media_dir,
    )

    fork_messages = repositories.list_messages(result.save.id)
    assert result.save.interaction_mode is InteractionMode.STORYTELLER
    assert result.message_count == 1
    assert fork_messages[0].id != first.id
    assert fork_messages[0].body == "The tower lens glows red."
    assert repositories.list_world_state(result.save.id)[0].value == {"color": "red"}
    fork_location = repositories.list_locations(result.save.id)[0]
    fork_character = repositories.list_characters(result.save.id)[0]
    assert fork_location.id != "tower"
    assert fork_character.id != "mara"
    assert repositories.list_message_visibility(result.save.id)[0].message_id == (
        fork_messages[0].id
    )
    assert repositories.list_message_visibility(result.save.id)[0].character_id == (
        fork_character.id
    )
    assert repositories.list_character_knowledge_edges(result.save.id)[0].target_id == (
        fork_location.id
    )
    fork_media = repositories.list_media_assets(result.save.id)[0]
    assert fork_media.path != "old.png"
    assert (media_dir / fork_media.path).read_bytes() == b"old image"
    fork_snapshot = repositories.get_scene_snapshot(result.save.id)
    assert fork_snapshot is not None
    assert fork_snapshot.world_time_day_index == 1
    assert fork_snapshot.world_time_day_label == "monday"
    assert fork_snapshot.world_time_phase == "morning"
    assert fork_snapshot.world_time_clock_minutes == 8 * 60 + 30
    assert fork_snapshot.world_time_period_label == "bell watch"
    assert fork_snapshot.source_message_id == fork_messages[0].id
    assert fork_snapshot.world_time_source_message_id == fork_messages[0].id
    assert fork_snapshot.world_time_confidence == 0.93


def test_snapshot_message_ids_returns_active_message_ids_in_rowid_order(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The lantern waits in darkness.",
    )
    second = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Keeper",
        body="I touch the glass.",
    )
    snapshot = service.capture_baseline_snapshot(save.id)
    assert service.snapshot_message_ids(snapshot_id=snapshot.id) == (
        first.id,
        second.id,
    )

    repositories.archive_message(second.id)
    later_snapshot = service.capture_baseline_snapshot(save.id)
    assert service.snapshot_message_ids(snapshot_id=later_snapshot.id) == (
        first.id,
    )


def test_snapshot_sanitizer_drops_rows_referencing_missing_messages() -> None:
    rows = _sanitize_snapshot_rows_for_safety(
        {
            "messages": (
                {"id": "message-one", "role": "narrator", "body": "The lens glows."},
                {
                    "id": "message-two",
                    "role": "player",
                    "speaker_name": "Mara",
                    "body": "I look up.",
                },
            ),
            "characters": (
                {
                    "id": "mara",
                    "save_id": "save",
                    "name": "Mara",
                    "source_message_id": "message-one",
                    "first_seen_message_id": "message-one",
                },
            ),
            "state_changes": (
                {
                    "id": "change-kept",
                    "save_id": "save",
                    "source_message_id": "message-one",
                    "operation": "upsert",
                    "state_key": "lens",
                    "before_json": None,
                    "after_json": '{"color":"red"}',
                },
                {
                    "id": "change-stale",
                    "save_id": "save",
                    "source_message_id": "message-gone",
                    "operation": "upsert",
                    "state_key": "lens",
                    "before_json": None,
                    "after_json": '{"color":"blue"}',
                },
            ),
            "character_knowledge_edges": (
                {
                    "id": "edge-kept",
                    "save_id": "save",
                    "character_id": "mara",
                    "target_type": "world_state",
                    "target_id": "world-kept",
                    "knowledge_state": "knows",
                    "acquisition_method": "inferred",
                    "confidence": 1.0,
                    "source_message_id": "message-one",
                    "source_message_ids_json": '["message-one"]',
                    "evidence_quote": "lens",
                    "archived_at": None,
                },
                {
                    "id": "edge-stale",
                    "save_id": "save",
                    "character_id": "mara",
                    "target_type": "world_state",
                    "target_id": "world-kept",
                    "knowledge_state": "may_know",
                    "acquisition_method": "inferred",
                    "confidence": 0.5,
                    "source_message_id": "message-gone",
                    "source_message_ids_json": '["message-one","message-gone"]',
                    "evidence_quote": "lens",
                    "archived_at": None,
                },
            ),
            "message_scene_presence": (
                {
                    "id": "presence-kept",
                    "save_id": "save",
                    "message_id": "message-two",
                    "character_id": "mara",
                    "source": "post_turn_context",
                },
                {
                    "id": "presence-stale",
                    "save_id": "save",
                    "message_id": "message-gone",
                    "character_id": "mara",
                    "source": "post_turn_context",
                },
            ),
            "context_sources": (
                {
                    "id": "source-kept",
                    "save_id": "save",
                    "source_type": "message",
                    "source_id": "message-one",
                    "scope": "turn",
                    "status": "active",
                    "archived_at": None,
                },
                {
                    "id": "source-stale",
                    "save_id": "save",
                    "source_type": "message",
                    "source_id": "message-gone",
                    "scope": "turn",
                    "status": "active",
                    "archived_at": None,
                },
                {
                    "id": "source-metadata-stale",
                    "save_id": "save",
                    "source_type": "message",
                    "source_id": "message-one",
                    "metadata_json": json.dumps({"source_message_id": "message-gone"}),
                    "scope": "turn",
                    "status": "active",
                    "archived_at": None,
                },
            ),
            "world_state": (
                {
                    "id": "world-kept",
                    "save_id": "save",
                    "key": "lens",
                    "value_json": '{"color":"red"}',
                    "source_message_id": "message-one",
                    "category": "story",
                    "confidence": 1.0,
                    "archived_at": None,
                },
                {
                    "id": "world-stale",
                    "save_id": "save",
                    "key": "gate",
                    "value_json": '{"open":true}',
                    "source_message_id": "message-gone",
                    "category": "story",
                    "confidence": 1.0,
                    "archived_at": None,
                },
                {
                    "id": "world-pressure-stale",
                    "save_id": "save",
                    "key": "story.director_pressure",
                    "value_json": json.dumps(
                        {"escalation_history": [{"source_message_id": "message-gone"}]}
                    ),
                    "source_message_id": "message-one",
                    "category": "story",
                    "confidence": 1.0,
                    "archived_at": None,
                },
            ),
        }
    )

    assert [row["id"] for row in rows["state_changes"]] == ["change-kept"]
    assert [row["id"] for row in rows["character_knowledge_edges"]] == ["edge-kept"]
    assert [row["id"] for row in rows["message_scene_presence"]] == [
        "presence-kept"
    ]
    assert [row["id"] for row in rows["context_sources"]] == ["source-kept"]
    world_state_rows = rows["world_state"]
    assert {row["id"] for row in world_state_rows} == {
        "world-kept",
        "world-stale",
        "world-pressure-stale",
    }
    stale_world = next(row for row in world_state_rows if row["id"] == "world-stale")
    assert stale_world["source_message_id"] is None
    pressure_world = next(
        row for row in world_state_rows if row["id"] == "world-pressure-stale"
    )
    assert json.loads(str(pressure_world["value_json"])) == {
        "escalation_history": []
    }


def test_snapshot_sanitizer_drops_rows_referencing_archived_entities() -> None:
    rows = _sanitize_snapshot_rows_for_safety(
        {
            "messages": (
                {"id": "message-one", "role": "narrator", "body": "The lens glows."},
            ),
            "characters": (
                {
                    "id": "mara",
                    "save_id": "save",
                    "name": "Mara",
                    "source_message_id": "message-one",
                },
            ),
            "memories": (
                {
                    "id": "memory-kept",
                    "save_id": "save",
                    "body": "kept",
                    "source_message_ids_json": '["message-one"]',
                    "archived_at": None,
                },
            ),
            "active_threads": (
                {
                    "id": "thread-kept",
                    "save_id": "save",
                    "source_message_id": "message-one",
                },
            ),
            "entity_links": (
                {
                    "id": "link-kept",
                    "save_id": "save",
                    "entity_type": "character",
                    "entity_id": "mara",
                    "target_type": "memory",
                    "target_id": "memory-kept",
                    "source_message_id": "message-one",
                },
                {
                    "id": "link-stale",
                    "save_id": "save",
                    "entity_type": "character",
                    "entity_id": "mara",
                    "target_type": "memory",
                    "target_id": "memory-gone",
                    "source_message_id": "message-one",
                },
            ),
            "context_update_audit": (
                {
                    "id": "audit-stale",
                    "save_id": "save",
                    "entity_type": "active_thread",
                    "entity_id": "thread-gone",
                    "source_message_ids_json": '["message-one"]',
                },
            ),
            "context_sources": (
                {
                    "id": "source-thread-stale",
                    "save_id": "save",
                    "source_type": "open_obligation",
                    "source_id": "thread-gone",
                    "scope": "turn",
                    "status": "active",
                    "archived_at": None,
                },
                {
                    "id": "source-memory-stale",
                    "save_id": "save",
                    "source_type": "memory",
                    "source_id": "memory-gone",
                    "scope": "turn",
                    "status": "active",
                    "archived_at": None,
                },
                {
                    "id": "source-memory-kept",
                    "save_id": "save",
                    "source_type": "memory",
                    "source_id": "memory-kept",
                    "scope": "turn",
                    "status": "active",
                    "archived_at": None,
                },
            ),
            "save_loss_outcomes": (
                {
                    "id": "outcome-kept",
                    "save_id": "save",
                    "triggering_message_id": "message-one",
                },
                {
                    "id": "outcome-stale",
                    "save_id": "save",
                    "triggering_message_id": "message-gone",
                },
            ),
            "scene_snapshots": (
                {
                    "id": "scene-kept",
                    "save_id": "save",
                    "present_character_ids_json": '["mara","character-gone"]',
                    "source_message_id": "message-one",
                },
            ),
        }
    )

    assert [row["id"] for row in rows["entity_links"]] == ["link-kept"]
    assert [row["id"] for row in rows["context_update_audit"]] == []
    assert [row["id"] for row in rows["context_sources"]] == [
        "source-memory-kept"
    ]
    assert [row["id"] for row in rows["save_loss_outcomes"]] == ["outcome-kept"]
    scene = rows["scene_snapshots"][0]
    assert json.loads(str(scene["present_character_ids_json"])) == ["mara"]


def test_snapshot_sanitizer_cascades_drops_through_reference_chain() -> None:
    rows = _sanitize_snapshot_rows_for_safety(
        {
            "messages": (
                {"id": "message-one", "role": "narrator", "body": "The lens glows."},
            ),
            "characters": (
                {
                    "id": "mara",
                    "save_id": "save",
                    "name": "Mara",
                    "source_message_id": "message-one",
                },
            ),
            "context_update_suggestions": (
                {
                    "id": "suggestion-stale",
                    "save_id": "save",
                    "entity_type": "character",
                    "entity_id": "mara",
                    "field_path": "location_id",
                    "proposed_value_json": '"location-gone"',
                    "source_message_ids_json": '["message-one"]',
                },
            ),
            "context_update_audit": (
                {
                    "id": "audit-cascade",
                    "save_id": "save",
                    "entity_type": "character",
                    "entity_id": "mara",
                    "field_path": "location_id",
                    "suggestion_id": "suggestion-stale",
                    "source_message_ids_json": '["message-one"]',
                },
            ),
        }
    )

    assert [row["id"] for row in rows["context_update_suggestions"]] == []
    assert [row["id"] for row in rows["context_update_audit"]] == []


def test_snapshot_sanitizer_handles_summary_memory_and_metadata_references() -> None:
    rows = _sanitize_snapshot_rows_for_safety(
        {
            "messages": (
                {"id": "message-one", "role": "narrator", "body": "The lens glows."},
            ),
            "memories": (
                {
                    "id": "memory-kept",
                    "save_id": "save",
                    "body": "kept",
                    "source_message_ids_json": '["message-one"]',
                    "source_observation_ids_json": (
                        '["observation-one","observation-gone"]'
                    ),
                    "archived_at": None,
                },
            ),
            "context_observations": (
                {
                    "id": "observation-one",
                    "save_id": "save",
                    "source_message_ids_json": '["message-one"]',
                    "archived_at": None,
                },
            ),
            "summaries": (
                {
                    "id": "summary-kept",
                    "save_id": "save",
                    "covers_message_start_id": "message-one",
                    "covers_message_end_id": "message-one",
                    "source_message_ids_json": '["message-one"]',
                },
                {
                    "id": "summary-stale",
                    "save_id": "save",
                    "covers_message_start_id": "message-one",
                    "covers_message_end_id": "message-gone",
                    "source_message_ids_json": '["message-one"]',
                },
            ),
            "context_sources": (
                {
                    "id": "source-entity-stale",
                    "save_id": "save",
                    "source_type": "memory",
                    "source_id": "memory-kept",
                    "metadata_json": json.dumps(
                        {"entity_ids": ["character-gone"]}
                    ),
                    "scope": "turn",
                    "status": "active",
                    "archived_at": None,
                },
            ),
            "context_update_suggestions": (
                {
                    "id": "suggestion-location-stale",
                    "save_id": "save",
                    "entity_type": "character",
                    "entity_id": "mara",
                    "field_path": "location_id",
                    "proposed_value_json": json.dumps(
                        {
                            "location_id": "location-gone",
                            "source_message_id": "message-one",
                        }
                    ),
                    "source_message_ids_json": '["message-one"]',
                },
            ),
        }
    )

    memory = rows["memories"][0]
    assert json.loads(str(memory["source_observation_ids_json"])) == [
        "observation-one"
    ]
    assert [row["id"] for row in rows["summaries"]] == ["summary-kept"]
    assert [row["id"] for row in rows["context_sources"]] == []
    assert [row["id"] for row in rows["context_update_suggestions"]] == []


def test_snapshot_sanitizer_clears_nested_and_typed_references() -> None:
    rows = _sanitize_snapshot_rows_for_safety(
        {
            "messages": (
                {"id": "message-one", "role": "narrator", "body": "The lens glows."},
            ),
            "characters": (
                {
                    "id": "mara",
                    "save_id": "save",
                    "name": "Mara",
                    "source_message_id": "message-one",
                },
            ),
            "active_threads": (
                {
                    "id": "thread-kept",
                    "save_id": "save",
                    "source_message_id": "message-one",
                    "related_entities_json": (
                        '["character:mara","character:character-gone"]'
                    ),
                },
            ),
            "media_assets": (
                {
                    "id": "asset-kept",
                    "save_id": "save",
                    "path": "old.png",
                    "source_message_id": "message-one",
                    "metadata_json": json.dumps(
                        {
                            "decision_reason": "x",
                            "context": {
                                "character_id": "character-gone",
                                "nested": {"character_id": "mara"},
                            },
                        }
                    ),
                },
            ),
            "context_sources": (
                {
                    "id": "source-audience-stale",
                    "save_id": "save",
                    "source_type": "message",
                    "source_id": "message-one",
                    "metadata_json": json.dumps(
                        {"audience_character_ids": ["mara", "character-gone"]}
                    ),
                    "scope": "turn",
                    "status": "active",
                    "archived_at": None,
                },
            ),
            "dating_route_states": (
                {
                    "id": "route-stale",
                    "save_id": "save",
                    "player_character_id": "mara",
                    "npc_character_id": "npc-gone",
                    "stage": "met",
                    "source_message_id": "message-one",
                },
            ),
        }
    )

    thread = rows["active_threads"][0]
    assert json.loads(str(thread["related_entities_json"])) == ["character:mara"]
    asset = rows["media_assets"][0]
    metadata = json.loads(str(asset["metadata_json"]))
    assert metadata["context"]["character_id"] is None
    assert metadata["context"]["nested"]["character_id"] == "mara"
    assert [row["id"] for row in rows["context_sources"]] == []
    assert [row["id"] for row in rows["dating_route_states"]] == []


def test_snapshot_capture_excludes_rows_referencing_deleted_messages(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The tower lens glows red.",
    )
    repositories.add_state_change(
        save_id=save.id,
        operation="upsert",
        state_key="lens",
        after_json='{"color":"red"}',
        source_message_id=first.id,
    )
    repositories.add_character(
        character_id="mara",
        save_id=save.id,
        name="Mara",
        source_message_id=first.id,
    )
    lens_state = repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
        source_message_id=first.id,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id="mara",
        target_type="world_state",
        target_id=lens_state.id,
        source_message_id=first.id,
    )
    repositories.replace_message_scene_presence(
        save_id=save.id,
        message_id=first.id,
        character_ids=["mara"],
        source="post_turn_context",
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)

    second = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I touch the lens.",
    )
    repositories.add_state_change(
        save_id=save.id,
        operation="upsert",
        state_key="touch",
        after_json='{"touched":true}',
        source_message_id=second.id,
    )
    repositories.replace_message_scene_presence(
        save_id=save.id,
        message_id=second.id,
        character_ids=["mara"],
        source="post_turn_context",
    )
    repositories.archive_message(first.id)

    later_snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=second.id,
    )
    rows = _snapshot_rows_by_table(repositories, later_snapshot.id)
    assert service.snapshot_message_ids(snapshot_id=later_snapshot.id) == (
        second.id,
    )
    assert all(
        row["source_message_id"] != first.id
        for row in rows["state_changes"]
        if isinstance(row.get("source_message_id"), str)
    )
    assert all(
        row["source_message_id"] != first.id
        for row in rows["world_state"]
        if isinstance(row.get("source_message_id"), str)
    )
    assert all(
        row.get("source_message_id") != first.id
        for row in rows["character_knowledge_edges"]
    )
    assert all(
        row["message_id"] != first.id for row in rows["message_scene_presence"]
    )

    result = service.fork_snapshot_to_save(
        source_save_id=save.id,
        snapshot_id=later_snapshot.id,
        title="Fork after lens touch",
        media_dir=tmp_path / "media",
    )
    assert result.message_count == 1
    fork_rows = repositories.connection.execute(
        "SELECT source_message_id FROM state_changes WHERE save_id = ?",
        (result.save.id,),
    ).fetchall()
    assert all(
        row["source_message_id"] != first.id
        for row in fork_rows
        if row["source_message_id"] is not None
    )
    fork_presence = repositories.connection.execute(
        "SELECT message_id FROM message_scene_presence WHERE save_id = ?",
        (result.save.id,),
    ).fetchall()
    assert all(row["message_id"] != first.id for row in fork_presence)


def test_fork_heals_snapshot_with_unknown_message_references(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The tower lens glows red.",
    )
    message_hash = service._store_object(
        kind="row:messages",
        value={
            "id": first.id,
            "save_id": save.id,
            "role": "narrator",
            "speaker_name": "Narrator",
            "body": "The tower lens glows red.",
            "provider": None,
            "model": None,
            "token_estimate": 0,
            "deleted_at": None,
            "safety_transition": "",
            "content_rating": "unclassified",
        },
    )
    stale_change_hash = service._store_object(
        kind="row:state_changes",
        value={
            "id": "change-stale",
            "save_id": save.id,
            "source_message_id": "ghost-message",
            "operation": "upsert",
            "state_key": "lens",
            "before_json": None,
            "after_json": '{"color":"red"}',
        },
    )
    stale_presence_hash = service._store_object(
        kind="row:message_scene_presence",
        value={
            "id": "presence-stale",
            "save_id": save.id,
            "message_id": "ghost-message",
            "character_id": "mara",
            "source": "post_turn_context",
        },
    )
    manifest_hash = service._store_object(
        kind="snapshot_manifest",
        value={
            "format": turn_snapshot_module.SNAPSHOT_FORMAT,
            "save_id": save.id,
            "message_id": first.id,
            "active_message_ids": [first.id],
            "context_revision": 1,
            "tables": {
                "messages": [{"id": first.id, "object_hash": message_hash}],
                "state_changes": [
                    {"id": "change-stale", "object_hash": stale_change_hash}
                ],
                "message_scene_presence": [
                    {"id": "presence-stale", "object_hash": stale_presence_hash}
                ],
            },
        },
    )
    repositories.connection.execute(
        """
        INSERT INTO save_turn_snapshots(
            id, save_id, message_id, parent_snapshot_id, root_manifest_hash,
            context_revision, reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "broken-snapshot",
            save.id,
            first.id,
            None,
            manifest_hash,
            1,
            "test_broken",
        ),
    )
    repositories.commit()

    result = service.fork_snapshot_to_save(
        source_save_id=save.id,
        snapshot_id="broken-snapshot",
        title="Fork heals stale references",
        media_dir=tmp_path / "media",
    )

    assert result.message_count == 1
    stale_rows = repositories.connection.execute(
        """
        SELECT source_message_id
        FROM state_changes
        WHERE save_id = ?
        """,
        (result.save.id,),
    ).fetchall()
    assert stale_rows == []
    presence_rows = repositories.connection.execute(
        """
        SELECT message_id
        FROM message_scene_presence
        WHERE save_id = ?
        """,
        (result.save.id,),
    ).fetchall()
    assert presence_rows == []


def test_fork_snapshot_rejects_trailing_message_in_source_snapshot(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower lens glows red.",
    )
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=first.id,
    )

    with pytest.raises(
        ValueError,
        match="Trailing fork messages must follow the source snapshot",
    ):
        service.fork_snapshot_to_save(
            source_save_id=save.id,
            snapshot_id=snapshot.id,
            title="Forked from player message",
            media_dir=tmp_path / "media",
            trailing_messages=(first,),
        )
    save_count = repositories.connection.execute(
        "SELECT COUNT(*) FROM saves"
    ).fetchone()[0]
    assert save_count == 1


def test_fork_snapshot_rejects_trailing_message_from_other_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    other = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower lens glows red.",
    )
    other_message = repositories.append_message(
        save_id=other.id,
        role="player",
        speaker_name="Keeper",
        body="I turn the lens.",
    )
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=first.id,
    )

    with pytest.raises(
        ValueError,
        match="Trailing fork messages must be active source messages",
    ):
        service.fork_snapshot_to_save(
            source_save_id=save.id,
            snapshot_id=snapshot.id,
            title="Forked from player message",
            media_dir=tmp_path / "media",
            trailing_messages=(other_message,),
        )
    save_count = repositories.connection.execute(
        "SELECT COUNT(*) FROM saves"
    ).fetchone()[0]
    assert save_count == 2


def test_fork_snapshot_rejects_archived_trailing_message(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower lens glows red.",
    )
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=first.id,
    )
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Keeper",
        body="I turn the lens.",
    )
    repositories.archive_message(player.id)
    archived = repositories.get_message(
        save_id=save.id,
        message_id=player.id,
        include_deleted=True,
    )
    assert archived is not None and archived.deleted_at is not None

    with pytest.raises(
        ValueError,
        match="Trailing fork messages must be active source messages",
    ):
        service.fork_snapshot_to_save(
            source_save_id=save.id,
            snapshot_id=snapshot.id,
            title="Forked from player message",
            media_dir=tmp_path / "media",
            trailing_messages=(archived,),
        )
    save_count = repositories.connection.execute(
        "SELECT COUNT(*) FROM saves"
    ).fetchone()[0]
    assert save_count == 1


def test_fork_snapshot_appends_trailing_message_after_snapshot(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    first = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower lens glows red.",
    )
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=first.id,
    )
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Keeper",
        body="I turn the lens.",
    )

    result = service.fork_snapshot_to_save(
        source_save_id=save.id,
        snapshot_id=snapshot.id,
        title="Forked from player message",
        media_dir=tmp_path / "media",
        trailing_messages=(player,),
    )

    forked_messages = repositories.list_messages(result.save.id)
    assert [message.body for message in forked_messages] == [
        "The tower lens glows red.",
        "I turn the lens.",
    ]
    assert forked_messages[-1].role == "player"
    assert forked_messages[-1].id != player.id
    assert result.message_count == 2
    assert service.latest_snapshot_for_message(
        save_id=result.save.id,
        message_id=forked_messages[-1].id,
    ) is not None


def test_snapshot_fork_copies_legacy_normalized_budget_allowance(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = _create_save(repositories)
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="custom_note",
        source_id="legacy-budget-note",
        title="Legacy note",
        body="The moonstone opens the archive.",
    )
    normalized_text_bytes = repositories.connection.execute(
        """
        SELECT normalized_text_bytes
        FROM context_source_index_budget_state
        WHERE save_id = ?
        """,
        (save.id,),
    ).fetchone()[0]
    monkeypatch.setattr(
        repositories_module,
        "MAX_CONTEXT_SOURCE_NORMALIZED_BYTES_PER_REBUILD",
        1,
    )
    monkeypatch.setattr(
        repositories_module,
        "MAX_CONTEXT_SOURCE_NORMALIZED_BYTES_PER_RECORD",
        1,
    )
    repositories.ensure_context_source_legacy_budget_limit(
        save_id=save.id,
        normalized_text_bytes=normalized_text_bytes,
        normalized_record_bytes=normalized_text_bytes,
    )
    repositories.commit()
    service = TurnSnapshotService(repositories)
    snapshot = service.capture_baseline_snapshot(save.id)

    fork = service.fork_snapshot_to_save(
        source_save_id=save.id,
        snapshot_id=snapshot.id,
        title="Forked legacy save",
        media_dir=tmp_path / "media",
    )

    fork_limit = repositories.connection.execute(
        """
        SELECT normalized_text_bytes
        FROM context_source_legacy_budget_limits
        WHERE save_id = ?
        """,
        (fork.save.id,),
    ).fetchone()[0]
    fork_record_limit = repositories.connection.execute(
        """
        SELECT normalized_text_bytes
        FROM context_source_legacy_record_budget_limits
        WHERE save_id = ?
        """,
        (fork.save.id,),
    ).fetchone()[0]
    assert fork_limit == normalized_text_bytes
    assert fork_record_limit == normalized_text_bytes


def test_snapshot_backed_fork_rejects_media_paths_outside_media_root(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    outside_path = tmp_path / "outside.png"
    outside_path.write_bytes(b"outside image")
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower lens glows red.",
    )
    repositories.connection.execute(
        """
        INSERT INTO media_assets(
            id, save_id, source_message_id, type, path, thumbnail_path, prompt,
            provider, model, status, mime_type, metadata_json
        )
        VALUES (?, ?, ?, 'image', '../outside.png', NULL, 'escaped path',
                'fake', 'image', 'succeeded', 'image/png', '{}')
        """,
        ("escaped-media", save.id, message.id),
    )
    repositories.commit()
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)

    with pytest.raises(ValueError, match="escapes media directory"):
        SaveForkService(repositories).fork_from_message(
            save_id=save.id,
            message_id=message.id,
            media_dir=media_dir,
        )

    assert [listed_save.id for listed_save in repositories.list_saves()] == [save.id]
    assert not (media_dir / "forks").exists()


def test_snapshot_backed_fork_remaps_character_text_rows_and_attachment_media(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan sends the ticket stub.",
    )
    seeded = _seed_character_text_snapshot_state(
        repositories,
        save_id=save.id,
        source_message_id=message.id,
        media_dir=media_dir,
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="character_text_thread",
        source_id=seeded["thread_id"],
        title="Rowan text thread",
        body="Phone thread memory: Rowan found a ticket stub.",
        metadata={
            "source_message_ids": [
                character_text_source_ref(seeded["text_message_id"])
            ],
            "audience_character_ids": [
                seeded["player_id"],
                seeded["rowan_id"],
            ],
            "thread_id": seeded["thread_id"],
            "entity_ids": [seeded["thread_id"]],
        },
    )
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)

    fork = SaveForkService(repositories).fork_from_message(
        save_id=save.id,
        message_id=message.id,
        media_dir=media_dir,
    )

    fork_message = repositories.list_messages(fork.save.id)[0]
    fork_characters = {
        character.name: character
        for character in repositories.list_characters(fork.save.id)
    }
    fork_player = fork_characters["Mara"]
    fork_rowan = fork_characters["Rowan"]
    fork_thread = repositories.list_character_text_threads(fork.save.id)[0]
    fork_text = repositories.list_character_text_messages(save_id=fork.save.id)[0]
    fork_media = repositories.list_media_assets(fork.save.id)[0]
    fork_attachment = repositories.list_character_text_message_attachments(
        save_id=fork.save.id
    )[0]
    fork_memory = repositories.list_memories(fork.save.id)[0]

    assert fork_thread.id != seeded["thread_id"]
    assert fork_thread.character_id == fork_rowan.id
    assert fork_text.id != seeded["text_message_id"]
    assert fork_text.thread_id == fork_thread.id
    assert fork_text.character_id == fork_rowan.id
    assert fork_attachment.text_message_id == fork_text.id
    assert fork_attachment.character_id == fork_rowan.id
    assert fork_attachment.media_asset_id == fork_media.id
    assert fork_media.id != seeded["media_asset_id"]
    assert fork_media.path != seeded["media_path"]
    assert (media_dir / fork_media.path).read_bytes() == b"text attachment image"
    media_metadata = json.loads(fork_media.metadata_json)
    assert media_metadata["character_id"] == fork_rowan.id
    assert media_metadata["thread_id"] == fork_thread.id
    assert media_metadata["text_message_id"] == fork_text.id
    contacts = repositories.list_character_contact_states(fork.save.id)
    assert len(contacts) == 1
    assert contacts[0].player_character_id == fork_player.id
    assert contacts[0].character_id == fork_rowan.id
    assert contacts[0].source_message_id == fork_message.id
    assert contacts[0].source_text_message_id == fork_text.id
    triggers = repositories.list_character_text_proactive_triggers(fork.save.id)
    assert len(triggers) == 1
    assert triggers[0].character_id == fork_rowan.id
    assert triggers[0].thread_id == fork_thread.id
    assert triggers[0].text_message_id == fork_text.id
    assert triggers[0].source_id == fork_rowan.id
    assert triggers[0].source_message_id == fork_message.id
    assert triggers[0].trigger_key == (
        f"character_intent:{fork_rowan.id}:{fork_message.id}"
    )
    provenance = repositories.list_character_text_provenance(save_id=fork.save.id)
    assert len(provenance) == 1
    assert provenance[0].thread_id == fork_thread.id
    assert provenance[0].text_message_id == fork_text.id
    assert provenance[0].target_id == fork_memory.id
    assert fork_memory.source_message_ids == [
        character_text_source_ref(fork_text.id)
    ]
    [fork_thread_context] = repositories.list_context_sources(
        fork.save.id,
        source_type="character_text_thread",
    )
    assert fork_thread_context.source_id == fork_thread.id
    assert fork_thread_context.metadata["source_message_ids"] == [
        character_text_source_ref(fork_text.id)
    ]
    audience_character_ids = fork_thread_context.metadata["audience_character_ids"]
    assert isinstance(audience_character_ids, list)
    assert set(audience_character_ids) == {fork_player.id, fork_rowan.id}
    assert fork_thread_context.metadata["thread_id"] == fork_thread.id
    assert fork_thread_context.metadata["entity_ids"] == [fork_thread.id]
    revisions = repositories.list_character_text_message_revisions(
        save_id=fork.save.id
    )
    assert [
        (revision.text_message_id, revision.previous_body) for revision in revisions
    ] == [
        (fork_text.id, "I found a ticket stub.")
    ]


def test_restore_snapshot_replaces_scene_scratch_before_its_scene(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens stays warm.",
    )
    scene = repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The beacon lens stays warm.",
        source_message_id=message.id,
    )
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="scene_detail",
        claim="The beacon lens is warm.",
        evidence_quote="beacon lens stays warm",
        source_message_ids=[message.id],
        scope="scene",
        status="accepted",
    )
    scratch = repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id=observation.id,
        title="Warm lens",
        body=observation.claim,
        metadata={
            "observation_id": observation.id,
            "curation_action": "scene_scratch",
        },
        scene_snapshot_id=scene.id,
        scene_generation=scene.scene_generation,
        created_turn_number=1,
        expires_after_turn_number=13,
    )
    service = TurnSnapshotService(repositories)
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    restored_scene = repositories.get_scene_snapshot(save.id)
    restored_scratch = repositories.get_context_source(scratch.id)
    assert restored_scene is not None
    assert restored_scratch is not None
    assert restored_scratch.scene_snapshot_id == restored_scene.id

    fork = SaveForkService(repositories).fork_from_message(
        save_id=save.id,
        message_id=message.id,
        media_dir=tmp_path / "media",
    )
    [fork_observation] = repositories.list_context_observations(fork.save.id)
    [fork_scratch] = repositories.list_context_sources(
        fork.save.id,
        source_type="observation",
    )
    assert fork_scratch.source_id == fork_observation.id


def test_restore_snapshot_restores_scene_facts_and_provenance(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The brass key rests on the table.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="A brass key rests within reach.",
        source_message_id=message.id,
    )
    original, _, _ = repositories.upsert_scene_fact(
        save_id=save.id,
        fact_type="object_location",
        subject_type="object",
        subject_id=None,
        subject_label="brass key",
        value="on the table",
        source_message_id=message.id,
        evidence_quote="rests on the table",
    )
    service = TurnSnapshotService(repositories)
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    repositories.upsert_scene_fact(
        save_id=save.id,
        fact_type="object_location",
        subject_type="object",
        subject_id=None,
        subject_label="brass key",
        value="under the table",
        source_message_id=message.id,
        evidence_quote="brass key",
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    [restored] = repositories.list_scene_facts(save.id)
    assert restored.id == original.id
    assert restored.fact_type == "object_location"
    assert restored.value == "on the table"
    assert restored.provenance[0].source_message_id == message.id


def test_snapshot_backed_fork_remaps_character_text_reply_links(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Mara and Rowan exchange texts.",
    )
    repositories.add_character(
        character_id="mara",
        save_id=save.id,
        name="Mara",
        role="Signal warden",
        is_player_character=True,
        met=True,
        source_message_id=message.id,
    )
    rowan = repositories.add_character(
        character_id="rowan",
        save_id=save.id,
        name="Rowan",
        role="Courier",
        met=True,
        source_message_id=message.id,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=rowan.id,
        title="Rowan",
    )
    player_text = repositories.append_character_text_message(
        message_id="text-player-where",
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="player",
        body="Where are you?",
        in_world_sent_at="Friday evening after class",
        delivered_at="2026-07-01T12:05:00+00:00",
    )
    repositories.append_character_text_message(
        message_id="text-rowan-reply",
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="character",
        body="By the south gate.",
        in_world_sent_at="Friday evening after class",
        delivered_at="2026-07-01T12:06:00+00:00",
        read_at="2026-07-01T12:07:00+00:00",
        reply_to_message_id=player_text.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)

    fork = SaveForkService(repositories).fork_from_message(
        save_id=save.id,
        message_id=message.id,
        media_dir=tmp_path / "media",
    )

    fork_texts = repositories.list_character_text_messages(save_id=fork.save.id)
    fork_player_text = next(
        text for text in fork_texts if text.body == "Where are you?"
    )
    fork_reply = next(text for text in fork_texts if text.body == "By the south gate.")
    assert fork_player_text.id != player_text.id
    assert fork_reply.reply_to_message_id == fork_player_text.id
    assert fork_reply.in_world_sent_at == "Friday evening after class"
    assert fork_reply.delivered_at == "2026-07-01T12:06:00+00:00"
    assert fork_reply.read_at == "2026-07-01T12:07:00+00:00"


def test_character_text_snapshot_filter_prunes_stale_reply_chains() -> None:
    rows: dict[str, tuple[dict[str, object], ...]] = {
        "characters": ({"id": "character-rowan"},),
        "messages": (),
        "media_assets": (),
        "character_text_threads": (
            {"id": "thread-rowan", "character_id": "character-rowan"},
        ),
        "character_text_messages": (
            {
                "id": "text-keep",
                "thread_id": "thread-rowan",
                "character_id": "character-rowan",
                "reply_to_message_id": None,
            },
            {
                "id": "text-first-dependent",
                "thread_id": "thread-rowan",
                "character_id": "character-rowan",
                "reply_to_message_id": "text-pruned-target",
            },
            {
                "id": "text-second-dependent",
                "thread_id": "thread-rowan",
                "character_id": "character-rowan",
                "reply_to_message_id": "text-first-dependent",
            },
            {
                "id": "text-pruned-target",
                "thread_id": "thread-rowan",
                "character_id": "character-rowan",
                "reply_to_message_id": "text-missing",
            },
        ),
    }

    _filter_character_text_snapshot_rows(rows)

    assert [row["id"] for row in rows["character_text_messages"]] == ["text-keep"]


def test_manual_edits_are_removed_before_later_snapshot_and_included_at_head(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check the lens.",
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)
    repositories.upsert_world_state(
        save_id=save.id,
        key="manual.note",
        value={"body": "added after first message"},
        source_message_id=None,
    )
    second = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I keep walking.",
    )
    service.capture_message_snapshot(save_id=save.id, message_id=second.id)

    fork = SaveForkService(repositories).fork_from_message(
        save_id=save.id,
        message_id=second.id,
        media_dir=tmp_path / "media",
    )
    assert repositories.list_world_state(fork.save.id)[0].key == "manual.note"

    MessageRevisionService(repositories).delete_suffix_from_message(
        save_id=save.id,
        message_id=second.id,
    )

    assert repositories.list_world_state(save.id) == []


def test_dirty_head_capture_tracks_message_body_edits_without_context_change(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check the lens.",
    )
    original = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )

    repositories.update_message_body(
        save_id=save.id,
        message_id=message.id,
        body="I polish the lens.",
    )

    edited = service.capture_current_head_if_dirty(
        save.id,
        reason="message_edit",
    )

    assert edited.id != original.id
    repositories.update_message_body(
        save_id=save.id,
        message_id=message.id,
        body="I crack the lens.",
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=edited.id)

    assert repositories.list_messages(save.id)[0].body == "I polish the lens."


def test_clean_head_capture_does_not_prepare_active_snapshot_rows(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    baseline = service.capture_baseline_snapshot(save.id)

    def fail_active_scan(_save_id: str) -> object:
        raise AssertionError("clean snapshot capture scanned active tables")

    monkeypatch.setattr(service, "_active_rows_by_table", fail_active_scan)

    captured = service.capture_current_head_if_dirty(save.id)

    assert captured.id == baseline.id


def test_snapshot_v2_manifest_reuses_unchanged_table_roots(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    baseline = service.capture_baseline_snapshot(save.id)
    baseline_manifest = service._snapshot_manifest(baseline)
    message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check the lens.",
    )

    changed = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    changed_manifest = service._snapshot_manifest(changed)

    assert baseline_manifest["format"] == "bragi-turn-snapshot-v2"
    assert changed_manifest["format"] == "bragi-turn-snapshot-v2"
    baseline_tables = baseline_manifest["table_roots"]
    changed_tables = changed_manifest["table_roots"]
    assert isinstance(baseline_tables, dict)
    assert isinstance(changed_tables, dict)
    assert changed_tables["messages"] != baseline_tables["messages"]
    assert changed_tables["world_state"] == baseline_tables["world_state"]


def test_dirty_row_capture_updates_tree_without_full_active_scan(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check the lens.",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
        source_message_id=message.id,
    )
    service = TurnSnapshotService(repositories)
    original = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    original_manifest = service._snapshot_manifest(original)
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "blue"},
        source_message_id=message.id,
    )

    def fail_active_scan(_save_id: str) -> object:
        raise AssertionError("incremental snapshot capture scanned active tables")

    monkeypatch.setattr(service, "_active_rows_by_table", fail_active_scan)

    changed = service.capture_current_head_if_dirty(save.id)

    changed_manifest = service._snapshot_manifest(changed)
    original_roots = original_manifest["table_roots"]
    changed_roots = changed_manifest["table_roots"]
    assert isinstance(original_roots, dict)
    assert isinstance(changed_roots, dict)
    assert changed_roots["world_state"] != original_roots["world_state"]
    assert changed_roots["messages"] == original_roots["messages"]
    rows = service._rows_from_manifest(changed_manifest)
    [captured_state] = rows["world_state"]
    assert captured_state["id"] == state.id
    assert json.loads(str(captured_state["value_json"])) == {"color": "blue"}


def test_deleted_row_capture_updates_cached_graph_without_full_active_scan(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = _create_save(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
    )
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    repositories.connection.execute("DELETE FROM world_state WHERE id = ?", (state.id,))
    repositories.commit()

    def fail_active_scan(_save_id: str) -> object:
        raise AssertionError("incremental deletion scanned active tables")

    monkeypatch.setattr(service, "_active_rows_by_table", fail_active_scan)

    changed = service.capture_current_head_if_dirty(save.id)

    assert service._rows_from_manifest(service._snapshot_manifest(changed))[
        "world_state"
    ] == ()


def test_primary_key_update_removes_old_snapshot_identity(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = _create_save(repositories)
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
    )
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    repositories.connection.execute(
        "UPDATE world_state SET id = ? WHERE id = ?",
        ("replacement-state", state.id),
    )
    repositories.commit()

    def fail_active_scan(_save_id: str) -> object:
        raise AssertionError("primary-key update scanned active tables")

    monkeypatch.setattr(service, "_active_rows_by_table", fail_active_scan)
    changed = service.capture_current_head_if_dirty(save.id)

    rows = service._rows_from_manifest(service._snapshot_manifest(changed))
    assert [row["id"] for row in rows["world_state"]] == ["replacement-state"]


def test_incremental_reference_validation_is_scoped_to_save(
    repositories: PersistenceRepositories,
) -> None:
    first = _create_save(repositories)
    second = _create_save(repositories)
    own_message = repositories.append_message(
        save_id=first.id,
        role="player",
        body="I inspect the lens.",
    )
    foreign_message = repositories.append_message(
        save_id=second.id,
        role="player",
        body="I inspect another lens.",
    )
    state = repositories.upsert_world_state(
        save_id=first.id,
        key="lens",
        value={"color": "red"},
        source_message_id=own_message.id,
    )
    service = TurnSnapshotService(repositories)
    service.capture_message_snapshot(save_id=first.id, message_id=own_message.id)
    repositories.connection.execute(
        "UPDATE world_state SET source_message_id = ? WHERE id = ?",
        (foreign_message.id, state.id),
    )
    repositories.commit()

    changed = service.capture_current_head_if_dirty(first.id)

    [captured] = service._rows_from_manifest(service._snapshot_manifest(changed))[
        "world_state"
    ]
    assert captured["source_message_id"] is None


def test_incremental_reference_is_restored_when_target_reactivates(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    repositories.add_location(
        save_id=save.id,
        location_id="tower",
        name="Beacon Tower",
    )
    repositories.add_character(
        save_id=save.id,
        character_id="mara",
        name="Mara",
        location_id="tower",
    )
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    repositories.connection.execute(
        "UPDATE locations SET archived_at = CURRENT_TIMESTAMP WHERE id = 'tower'"
    )
    repositories.commit()

    archived = service.capture_current_head_if_dirty(save.id)
    [archived_character] = service._rows_from_manifest(
        service._snapshot_manifest(archived)
    )["characters"]
    assert archived_character["location_id"] is None

    repositories.connection.execute(
        "UPDATE locations SET archived_at = NULL WHERE id = 'tower'"
    )
    repositories.commit()
    restored = service.capture_current_head_if_dirty(save.id)

    [restored_character] = service._rows_from_manifest(
        service._snapshot_manifest(restored)
    )["characters"]
    assert restored_character["location_id"] == "tower"


def test_incremental_fade_transition_removes_dependents(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon is lit.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="beacon",
        value={"lit": True},
        source_message_id=message.id,
    )
    service = TurnSnapshotService(repositories)
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)
    repositories.connection.execute(
        """
        UPDATE messages
        SET body = ?, safety_transition = 'fade_to_black'
        WHERE id = ?
        """,
        ("The scene moves forward.", message.id),
    )
    repositories.commit()

    faded = service.capture_current_head_if_dirty(save.id)
    faded_rows = service._rows_from_manifest(service._snapshot_manifest(faded))
    assert faded_rows["messages"] == ()
    assert faded_rows["world_state"] == ()


def test_dirty_row_capture_serializes_only_changed_row(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I inventory the lenses.",
    )
    for index in range(32):
        repositories.upsert_world_state(
            save_id=save.id,
            key=f"lens.{index:02d}",
            value={"color": "red"},
            source_message_id=message.id,
        )
    service = TurnSnapshotService(repositories)
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens.17",
        value={"color": "blue"},
        source_message_id=message.id,
    )
    stored_row_kinds: list[str] = []
    original_store_object = service._store_object

    def record_store(*, kind: str, value: object) -> str:
        if kind.startswith("row:"):
            stored_row_kinds.append(kind)
        return original_store_object(kind=kind, value=value)

    monkeypatch.setattr(service, "_store_object", record_store)

    service.capture_current_head_if_dirty(save.id)

    assert stored_row_kinds == ["row:world_state"]


def test_character_text_edit_serializes_only_changed_row(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = _create_save(repositories)
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="A message arrives.",
    )
    character = repositories.add_character(
        character_id="rowan",
        save_id=save.id,
        name="Rowan",
        source_message_id=narrator.id,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=character.id,
        title="Rowan",
    )
    text_message = repositories.append_character_text_message(
        message_id="text-one",
        save_id=save.id,
        thread_id=thread.id,
        character_id=character.id,
        sender="character",
        body="Meet me by the south gate.",
        in_world_sent_at="Friday evening",
        delivered_at="2026-07-01T12:05:00+00:00",
    )
    service = TurnSnapshotService(repositories)
    service.capture_message_snapshot(save_id=save.id, message_id=narrator.id)
    repositories.update_character_text_message_body(
        save_id=save.id,
        message_id=text_message.id,
        body="Meet me by the north gate.",
    )
    stored_row_kinds: list[str] = []
    original_store_object = service._store_object

    def record_store(*, kind: str, value: object) -> str:
        if kind.startswith("row:"):
            stored_row_kinds.append(kind)
        return original_store_object(kind=kind, value=value)

    def fail_active_scan(_save_id: str) -> object:
        raise AssertionError("character-text edit scanned active tables")

    monkeypatch.setattr(service, "_store_object", record_store)
    monkeypatch.setattr(service, "_active_rows_by_table", fail_active_scan)

    service.capture_current_head_if_dirty(save.id)

    assert stored_row_kinds == ["row:character_text_messages"]


def test_snapshot_manifest_cannot_be_repointed_across_saves(
    repositories: PersistenceRepositories,
) -> None:
    first = _create_save(repositories)
    second = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    first_snapshot = service.capture_baseline_snapshot(first.id)
    second_snapshot = service.capture_baseline_snapshot(second.id)
    repositories.connection.execute(
        "UPDATE save_turn_snapshots SET root_manifest_hash = ? WHERE id = ?",
        (second_snapshot.root_manifest_hash, first_snapshot.id),
    )
    repositories.commit()

    with pytest.raises(ValueError, match="wrong save id"):
        service.restore_save_to_snapshot(
            save_id=first.id,
            snapshot_id=first_snapshot.id,
        )


def test_rolled_back_row_change_does_not_dirty_materialized_snapshot(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I inspect the lens.",
    )
    service = TurnSnapshotService(repositories)
    original = service.capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    repositories.begin_transaction()
    repositories.update_message_body(
        save_id=save.id,
        message_id=message.id,
        body="This edit is rolled back.",
    )
    repositories.rollback_transaction()

    clean = service.capture_current_head_if_dirty(save.id)

    assert clean.id == original.id


def test_restore_snapshot_strips_deprecated_scenario_update_sections(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    legacy_content: dict[str, object] = {
        "opening_message": "The red lens wakes.",
        "factions": "Beacon wardens",
        "rivals_and_factions": "Ash riders scout the ridge.",
        "reputation_and_contacts": "The old patrol owes Mara a warning.",
        "major_npcs": "Captain Rell guards the cracked stair.",
    }
    update = repositories.add_save_scenario_update(
        save_id=save.id,
        title="Ashfall Keep: Red Lens",
        premise="A red warning has reached the isolated border keep.",
        player_role="Signal warden",
        content=legacy_content,
        reason="Legacy snapshot fixture.",
        provider="fake-chat-provider",
        model="fake-chat-model",
    )
    service = TurnSnapshotService(repositories)
    snapshot = service.capture_baseline_snapshot(save.id)
    manifest = service._snapshot_manifest(snapshot)  # noqa: SLF001 - legacy fixture
    rows_by_table = service._rows_from_manifest(manifest)  # noqa: SLF001
    row = repositories.connection.execute(
        "SELECT * FROM save_scenario_updates WHERE id = ?",
        (update.id,),
    ).fetchone()
    legacy_row = dict(row)
    legacy_row["content_json"] = json.dumps(legacy_content, sort_keys=True)
    tables: dict[str, list[dict[str, str]]] = {}
    for table_name, table_rows in rows_by_table.items():
        entries: list[dict[str, str]] = []
        for table_row in table_rows:
            value = legacy_row if table_name == "save_scenario_updates" else table_row
            entries.append(
                {
                    "id": str(value.get("id", "")),
                    "object_hash": service._store_object(  # noqa: SLF001
                        kind=f"row:{table_name}",
                        value=value,
                    ),
                }
            )
        tables[table_name] = entries
    legacy_manifest_hash = service._store_object(  # noqa: SLF001 - fixture
        kind="snapshot_manifest",
        value={
            "format": "bragi-turn-snapshot-v1",
            "save_id": save.id,
            "message_id": None,
            "active_message_ids": [],
            "context_revision": snapshot.context_revision,
            "tables": tables,
        },
    )
    repositories.connection.execute(
        "UPDATE save_turn_snapshots SET root_manifest_hash = ? WHERE id = ?",
        (legacy_manifest_hash, snapshot.id),
    )
    repositories.connection.execute(
        "UPDATE save_scenario_updates SET content_json = ? WHERE id = ?",
        (json.dumps({"factions": "Current wardens"}, sort_keys=True), update.id),
    )
    repositories.commit()
    _snapshot_rows, snapshot_objects = service.export_snapshot_rows(
        save_id=save.id,
        active_message_ids=(),
    )
    exported_payloads = "\n".join(
        zlib.decompress(base64.b64decode(str(row["payload_base64"]))).decode("utf-8")
        for row in snapshot_objects
    )
    assert "rivals_and_factions" not in exported_payloads
    assert "reputation_and_contacts" not in exported_payloads
    assert "major_npcs" not in exported_payloads
    assert "Ash riders scout the ridge." in exported_payloads

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    [restored] = repositories.list_save_scenario_updates(save.id)
    restored_content = json.loads(restored.content_json)
    assert restored_content["factions"] == (
        "Beacon wardens\n\n"
        "Ash riders scout the ridge.\n\n"
        "The old patrol owes Mara a warning."
    )
    assert "rivals_and_factions" not in restored_content
    assert "reputation_and_contacts" not in restored_content
    assert "major_npcs" not in restored_content


def test_export_import_preserves_snapshot_backed_fork(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I set the lens red.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "red"},
        source_message_id=first.id,
    )
    repositories.add_save_scenario_update(
        save_id=save.id,
        title="Red Lens",
        premise="The beacon lens turns red.",
        player_role="Signal warden",
        content={
            "character_starters": [
                {
                    "name": "Mara",
                    "reference_image": {
                        "media_asset_id": "starter-reference",
                        "content_rating": "r",
                    },
                }
            ],
            "_source": {"content_rating": "r"},
        },
        reason="Snapshot rating quarantine fixture.",
        provider="fake",
        model="fake-chat",
        source_message_id=first.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)
    second = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I set the lens blue.",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="lens",
        value={"color": "blue"},
        source_message_id=second.id,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=second.id)
    bundle_path = tmp_path / "night-watch.bragi-chat"

    ChatBundleService(repositories=repositories, media_dir=media_dir).export_save(
        save.id,
        bundle_path,
    )

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["turn_snapshots"]
    assert data["snapshot_objects"]

    imported = ChatBundleService(
        repositories=repositories,
        media_dir=media_dir,
    ).import_save(bundle_path)
    imported_messages = repositories.list_messages(imported.save_id)
    imported_first = imported_messages[0]

    fork = SaveForkService(repositories).fork_from_message(
        save_id=imported.save_id,
        message_id=imported_first.id,
        media_dir=media_dir,
    )

    fork_messages = repositories.list_messages(fork.save.id)
    assert [message.body for message in fork_messages] == ["I set the lens red."]
    assert repositories.list_world_state(fork.save.id)[0].value == {"color": "red"}
    [fork_update] = repositories.list_save_scenario_updates(fork.save.id)
    fork_content = json.loads(fork_update.content_json)
    assert fork_content["_source"]["content_rating"] == "unclassified"
    assert fork_content["character_starters"][0]["reference_image"][
        "content_rating"
    ] == "unclassified"


def test_export_import_preserves_snapshot_only_media_for_fork(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    media_dir.joinpath("old.png").write_bytes(b"archived image bytes")
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    first = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I create the first image.",
    )
    media = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=first.id,
        type="image",
        path="old.png",
        prompt="old image",
        provider="fake",
        model="image",
        status="succeeded",
    )
    service.capture_message_snapshot(save_id=save.id, message_id=first.id)
    repositories.archive_media_asset(save_id=save.id, media_asset_id=media.id)
    second = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I archive that image.",
    )
    service.capture_message_snapshot(save_id=save.id, message_id=second.id)
    bundle_path = tmp_path / "snapshot-media.bragi-chat"

    ChatBundleService(repositories=repositories, media_dir=media_dir).export_save(
        save.id,
        bundle_path,
    )

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["media_assets"] == []
    assert data["snapshot_media_assets"][0]["id"] == media.id

    imported = ChatBundleService(
        repositories=repositories,
        media_dir=media_dir,
    ).import_save(bundle_path)
    imported_first = repositories.list_messages(imported.save_id)[0]

    fork = SaveForkService(repositories).fork_from_message(
        save_id=imported.save_id,
        message_id=imported_first.id,
        media_dir=media_dir,
    )

    fork_media = repositories.list_media_assets(fork.save.id)[0]
    assert fork_media.path != "old.png"
    assert (media_dir / fork_media.path).read_bytes() == b"archived image bytes"


def test_export_import_preserves_character_text_snapshot_backed_fork(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    save = _create_save(repositories)
    service = TurnSnapshotService(repositories)
    service.capture_baseline_snapshot(save.id)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Rowan sends the ticket stub.",
    )
    _seed_character_text_snapshot_state(
        repositories,
        save_id=save.id,
        source_message_id=message.id,
        media_dir=media_dir,
    )
    service.capture_message_snapshot(save_id=save.id, message_id=message.id)
    bundle_path = tmp_path / "character-text-snapshot.bragi-chat"

    ChatBundleService(repositories=repositories, media_dir=media_dir).export_save(
        save.id,
        bundle_path,
    )
    imported = ChatBundleService(
        repositories=repositories,
        media_dir=media_dir,
    ).import_save(bundle_path)
    imported_message = repositories.list_messages(imported.save_id)[0]

    fork = SaveForkService(repositories).fork_from_message(
        save_id=imported.save_id,
        message_id=imported_message.id,
        media_dir=media_dir,
    )

    fork_text = repositories.list_character_text_messages(save_id=fork.save.id)[0]
    fork_thread = repositories.list_character_text_threads(fork.save.id)[0]
    fork_media = repositories.list_media_assets(fork.save.id)[0]
    fork_attachment = repositories.list_character_text_message_attachments(
        save_id=fork.save.id
    )[0]
    fork_memory = repositories.list_memories(fork.save.id)[0]
    fork_rowan = next(
        character
        for character in repositories.list_characters(fork.save.id)
        if character.name == "Rowan"
    )

    assert fork_text.body == "I found a ticket stub."
    assert fork_text.thread_id == fork_thread.id
    assert fork_attachment.text_message_id == fork_text.id
    assert fork_attachment.media_asset_id == fork_media.id
    assert (media_dir / fork_media.path).read_bytes() == b"text attachment image"
    assert json.loads(fork_media.metadata_json)["text_message_id"] == fork_text.id
    assert fork_memory.source_message_ids == [
        character_text_source_ref(fork_text.id)
    ]
    assert repositories.list_character_text_provenance(
        save_id=fork.save.id,
        text_message_id=fork_text.id,
    )[0].target_id == fork_memory.id
    trigger = repositories.list_character_text_proactive_triggers(fork.save.id)[0]
    assert trigger.character_id == fork_rowan.id
    assert trigger.text_message_id == fork_text.id
    assert trigger.source_id == fork_rowan.id


def _create_save(
    repositories: PersistenceRepositories,
    *,
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY,
) -> SaveRecord:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A beacon tower.",
        player_role="Signal warden",
        content={"opening_message": "The lens hums."},
        interaction_mode=interaction_mode,
    )
    return repositories.create_save(scenario_id=scenario.id, title="Night Watch")


def _seed_character_text_snapshot_state(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    source_message_id: str,
    media_dir: Path,
) -> dict[str, str]:
    media_path = "texts/rowan-ticket.png"
    (media_dir / media_path).parent.mkdir(parents=True, exist_ok=True)
    (media_dir / media_path).write_bytes(b"text attachment image")
    player = repositories.add_character(
        character_id="mara",
        save_id=save_id,
        name="Mara",
        role="Signal warden",
        is_player_character=True,
        met=True,
        source_message_id=source_message_id,
    )
    rowan = repositories.add_character(
        character_id="rowan",
        save_id=save_id,
        name="Rowan",
        role="Courier",
        met=True,
        current_intent="Ask Mara to decode the ticket stub.",
        source_message_id=source_message_id,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=rowan.id,
        title="Rowan",
    )
    text_message = repositories.append_character_text_message(
        message_id="text-rowan-ticket",
        save_id=save_id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="character",
        body="I found a ticket stub.",
        provider="fake-text-provider",
        model="fake-text-model",
    )
    repositories.add_character_text_message_revision(
        revision_id="revision-rowan-text",
        save_id=save_id,
        text_message_id=text_message.id,
        previous_body="I found a ticket stub.",
        new_body="I found a ticket stub with blue ink.",
        diff_unified=(
            "--- previous\n"
            "+++ current\n"
            "@@ -1 +1 @@\n"
            "-I found a ticket stub.\n"
            "+I found a ticket stub with blue ink.\n"
        ),
        reconciliation_status="succeeded",
    )
    media_asset = repositories.create_media_asset(
        asset_id="media-rowan-ticket",
        save_id=save_id,
        source_message_id=None,
        type="image",
        path=media_path,
        prompt="ticket stub",
        provider="fake-image-provider",
        model="fake-image-model",
        status="succeeded",
        metadata={
            "kind": "character_text_object_context_image",
            "character_id": rowan.id,
            "thread_id": thread.id,
            "text_message_id": text_message.id,
        },
    )
    repositories.add_character_text_message_attachment(
        attachment_id="text-attachment-ticket",
        save_id=save_id,
        thread_id=thread.id,
        text_message_id=text_message.id,
        character_id=rowan.id,
        kind="object_context_image",
        status="succeeded",
        media_asset_id=media_asset.id,
        prompt="ticket stub",
        metadata={"decision_reason": "Rowan texted a concrete found object."},
    )
    memory = repositories.add_memory(
        memory_id="memory-rowan-ticket",
        save_id=save_id,
        body="Rowan found a ticket stub.",
        tags=["rowan"],
        source_message_ids=[character_text_source_ref(text_message.id)],
    )
    repositories.add_character_text_provenance(
        provenance_id="provenance-rowan-memory",
        save_id=save_id,
        thread_id=thread.id,
        text_message_id=text_message.id,
        target_type="memory",
        target_id=memory.id,
        operation="create",
        field_path="body",
    )
    repositories.upsert_character_contact_state(
        state_id="contact-rowan",
        save_id=save_id,
        player_character_id=player.id,
        character_id=rowan.id,
        player_has_character_number=True,
        character_has_player_number=False,
        source_message_id=source_message_id,
        source_text_message_id=text_message.id,
    )
    repositories.add_character_text_proactive_trigger(
        trigger_id="trigger-rowan",
        save_id=save_id,
        character_id=rowan.id,
        trigger_key=f"character_intent:{rowan.id}:{source_message_id}",
        trigger_type="character_intent",
        thread_id=thread.id,
        text_message_id=text_message.id,
        source_type="character",
        source_id=rowan.id,
        source_message_id=source_message_id,
        reason="Ask Mara to decode the ticket stub.",
    )
    return {
        "player_id": player.id,
        "rowan_id": rowan.id,
        "thread_id": thread.id,
        "text_message_id": text_message.id,
        "media_asset_id": media_asset.id,
        "media_path": media_path,
        "memory_id": memory.id,
    }


def _row_object_count(repositories: PersistenceRepositories) -> int:
    row = repositories.connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM save_snapshot_objects
        WHERE kind LIKE 'row:%'
        """
    ).fetchone()
    return int(row["count"])


def _snapshot_rows_by_table(
    repositories: PersistenceRepositories,
    snapshot_id: str,
) -> dict[str, list[dict[str, object]]]:
    service = TurnSnapshotService(repositories)
    snapshot = service._get_snapshot(snapshot_id)  # noqa: SLF001
    manifest = service._snapshot_manifest(snapshot)  # noqa: SLF001
    return {
        table_name: list(rows)
        for table_name, rows in service._rows_from_manifest(manifest).items()  # noqa: SLF001
    }


def _assert_snapshot_objects_do_not_store_media_bytes(
    repositories: PersistenceRepositories,
) -> None:
    rows = repositories.connection.execute(
        """
        SELECT payload
        FROM save_snapshot_objects
        """
    ).fetchall()
    payloads = b"\n".join(zlib.decompress(bytes(row["payload"])) for row in rows)
    assert b"image bytes stay on disk" not in payloads


def test_snapshot_captures_and_restores_turn_outcomes(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Evening comes.",
    )
    repositories.add_turn_outcome(
        save_id=save.id,
        message_id=narrator.id,
        payload={
            "save_id": save.id,
            "message_id": narrator.id,
            "attempt_resolution": "succeeded",
            "effects": [
                {
                    "candidate_id": "time:1",
                    "candidate_type": "world_time_change",
                    "domain": "time",
                    "operation": "update",
                    "state_key": "scene_snapshot.in_world_time",
                    "field_path": "",
                    "character_id": "",
                    "target_type": "",
                    "target_id": "",
                    "value": {"time_of_day": "evening"},
                    "confidence": 0.9,
                    "evidence_source_ids": [f"message:{narrator.id}"],
                    "evidence_quote": "Evening comes",
                    "verifier_status": "rendered",
                    "safe_to_commit": True,
                    "application_status": "committed",
                    "reason": "rendered",
                    "changed": True,
                }
            ],
            "applied_domains": ["time"],
            "queued_domains": [],
            "verification_passed": True,
            "verifier_available": True,
            "post_turn_update_needed": False,
            "committed_count": 1,
            "confirmation_queued_count": 0,
        },
    )
    service = TurnSnapshotService(repositories)
    snapshot = service.capture_message_snapshot(
        save_id=save.id,
        message_id=narrator.id,
    )
    repositories.add_turn_outcome(
        save_id=save.id,
        message_id=narrator.id,
        payload={"save_id": save.id, "message_id": narrator.id, "effects": []},
    )

    service.restore_save_to_snapshot(save_id=save.id, snapshot_id=snapshot.id)

    outcomes = repositories.list_turn_outcomes(save.id)
    assert len(outcomes) == 1
    assert outcomes[0].payload["attempt_resolution"] == "succeeded"
    assert outcomes[0].payload["applied_domains"] == ["time"]
    raw_effects = outcomes[0].payload["effects"]
    assert isinstance(raw_effects, list)
    effect = raw_effects[0]
    assert isinstance(effect, dict)
    assert effect["evidence_source_ids"] == [f"message:{narrator.id}"]
