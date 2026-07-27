"""Content-addressed save state snapshots for exact turn rollback and fork."""

from __future__ import annotations

import base64
import binascii
import json
import shutil
import sqlite3
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast
from uuid import uuid4

from bragi.app_logging import log_event
from bragi.persistence.models import MediaAssetRecord, MessageRecord, SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.safety import (
    CONTENT_FILTER_TRANSITION,
    CONTENT_FILTER_TRANSITION_KIND,
    FADE_TO_BLACK_TRANSITION,
    FADE_TO_BLACK_TRANSITION_KIND,
    normalize_message_safety,
)
from bragi.services.scenario_content_rating import (
    metadata_with_scenario_content_ratings,
)
from bragi.services.scenario_service import strip_deprecated_scenario_character_sections
from bragi.services.sexual_content_safety import (
    is_fade_to_black_message,
)

SNAPSHOT_FORMAT = "bragi-turn-snapshot-v1"
SNAPSHOT_ENCODING = "zlib-json-v1"


@dataclass(frozen=True)
class TurnSnapshotRecord:
    id: str
    save_id: str
    message_id: str | None
    parent_snapshot_id: str | None
    root_manifest_hash: str
    context_revision: int
    reason: str
    created_at: str | None = None


@dataclass(frozen=True)
class SnapshotDeletion:
    snapshot: TurnSnapshotRecord
    deleted_messages: tuple[MessageRecord, ...]


@dataclass(frozen=True)
class SnapshotForkResult:
    save: SaveRecord
    message_count: int
    media_count: int


@dataclass(frozen=True)
class _PreparedSnapshot:
    root_manifest_hash: str
    context_revision: int
    tables: dict[str, list[dict[str, str]]]


@dataclass(frozen=True)
class _SnapshotTable:
    name: str
    active_only: bool = False
    order_by: str = "rowid"


_SNAPSHOT_TABLES: tuple[_SnapshotTable, ...] = (
    _SnapshotTable("messages", order_by="rowid"),
    _SnapshotTable("world_state", active_only=True, order_by="key, rowid"),
    _SnapshotTable(
        "context_sources",
        active_only=True,
        order_by="source_type, source_id, rowid",
    ),
    _SnapshotTable("context_observations", active_only=True),
    _SnapshotTable("context_observation_curation_state"),
    _SnapshotTable("locations", active_only=True),
    _SnapshotTable("characters", active_only=True),
    _SnapshotTable("scene_snapshots"),
    _SnapshotTable("active_threads", active_only=True),
    _SnapshotTable("entity_links"),
    _SnapshotTable("context_update_suggestions"),
    _SnapshotTable("context_update_audit"),
    _SnapshotTable("state_changes"),
    _SnapshotTable("memories", active_only=True),
    _SnapshotTable("summaries", active_only=True),
    _SnapshotTable("save_scenario_updates", active_only=True),
    _SnapshotTable("save_loss_conditions", active_only=True),
    _SnapshotTable("save_loss_condition_changes", active_only=True),
    _SnapshotTable("save_loss_outcomes", active_only=True),
    _SnapshotTable("media_assets", active_only=True),
    _SnapshotTable("character_knowledge_edges", active_only=True),
    _SnapshotTable("message_visibility"),
    _SnapshotTable("message_scene_presence"),
    _SnapshotTable("message_action_choices"),
    _SnapshotTable("dating_route_states", active_only=True),
    _SnapshotTable("character_text_threads", active_only=True),
    _SnapshotTable("character_text_thread_participants", active_only=True),
    _SnapshotTable("character_text_messages"),
    _SnapshotTable("character_text_activity_events"),
    _SnapshotTable("narrator_phone_activity_cursors"),
    _SnapshotTable("character_text_message_revisions"),
    _SnapshotTable("character_text_message_attachments"),
    _SnapshotTable("character_text_provenance"),
    _SnapshotTable("character_contact_states", active_only=True),
    _SnapshotTable("character_text_proactive_triggers"),
)

_TABLES_BY_NAME = {table.name: table for table in _SNAPSHOT_TABLES}
_SNAPSHOT_TABLE_NAMES = tuple(table.name for table in _SNAPSHOT_TABLES)

_RESTORE_DELETE_ORDER = (
    "character_text_proactive_triggers",
    "character_contact_states",
    "character_text_provenance",
    "character_text_message_attachments",
    "character_text_message_revisions",
    "narrator_phone_activity_cursors",
    "character_text_activity_events",
    "character_text_messages",
    "character_text_thread_participants",
    "character_text_threads",
    "message_action_choices",
    "message_scene_presence",
    "message_visibility",
    "dating_route_states",
    "character_knowledge_edges",
    "entity_links",
    "context_update_audit",
    "context_update_suggestions",
    "scene_snapshots",
    "active_threads",
    "characters",
    "locations",
    "save_loss_outcomes",
    "save_loss_condition_changes",
    "save_loss_conditions",
    "media_assets",
    "save_scenario_updates",
    "summaries",
    "memories",
    "state_changes",
    "context_observation_curation_state",
    "context_observations",
    "context_sources",
    "world_state",
)

_RESTORE_INSERT_ORDER = (
    "locations",
    "characters",
    "character_text_threads",
    "character_text_thread_participants",
    "character_text_messages",
    "character_text_activity_events",
    "narrator_phone_activity_cursors",
    "scene_snapshots",
    "active_threads",
    "media_assets",
    "character_text_message_revisions",
    "character_text_message_attachments",
    "world_state",
    "state_changes",
    "memories",
    "summaries",
    "save_scenario_updates",
    "save_loss_conditions",
    "save_loss_condition_changes",
    "save_loss_outcomes",
    "context_sources",
    "context_observations",
    "context_observation_curation_state",
    "context_update_suggestions",
    "context_update_audit",
    "entity_links",
    "dating_route_states",
    "character_knowledge_edges",
    "character_text_provenance",
    "character_contact_states",
    "character_text_proactive_triggers",
    "message_visibility",
    "message_scene_presence",
    "message_action_choices",
)

_MESSAGE_REFERENCE_COLUMNS = {
    "source_message_id",
    "world_time_source_message_id",
    "first_seen_message_id",
    "last_updated_message_id",
    "covers_message_start_id",
    "covers_message_end_id",
    "triggering_message_id",
    "epilogue_message_id",
    "message_id",
    "narrator_message_id",
    "first_met_message_id",
    "last_interaction_message_id",
}

_TABLE_REFERENCE_COLUMNS: dict[str, dict[str, str]] = {
    "context_observation_curation_state": {
        "observation_id": "context_observations"
    },
    "locations": {"parent_location_id": "locations"},
    "characters": {"location_id": "locations"},
    "scene_snapshots": {"current_location_id": "locations"},
    "media_assets": {"source_media_asset_id": "media_assets"},
    "save_loss_condition_changes": {"condition_id": "save_loss_conditions"},
    "save_loss_outcomes": {"condition_id": "save_loss_conditions"},
    "context_update_audit": {"suggestion_id": "context_update_suggestions"},
    "message_visibility": {"character_id": "characters"},
    "message_scene_presence": {"character_id": "characters"},
    "character_knowledge_edges": {"character_id": "characters"},
    "dating_route_states": {
        "player_character_id": "characters",
        "npc_character_id": "characters",
    },
    "character_text_threads": {"character_id": "characters"},
    "character_text_thread_participants": {
        "thread_id": "character_text_threads",
        "character_id": "characters",
    },
    "character_text_messages": {
        "thread_id": "character_text_threads",
        "character_id": "characters",
        "sender_character_id": "characters",
        "reply_to_message_id": "character_text_messages",
    },
    "character_text_activity_events": {
        "thread_id": "character_text_threads",
        "text_message_id": "character_text_messages",
    },
    "narrator_phone_activity_cursors": {"narrator_message_id": "messages"},
    "character_text_message_revisions": {
        "text_message_id": "character_text_messages",
    },
    "character_text_message_attachments": {
        "thread_id": "character_text_threads",
        "text_message_id": "character_text_messages",
        "character_id": "characters",
        "media_asset_id": "media_assets",
    },
    "character_text_provenance": {
        "thread_id": "character_text_threads",
        "text_message_id": "character_text_messages",
    },
    "character_contact_states": {
        "player_character_id": "characters",
        "character_id": "characters",
        "source_text_message_id": "character_text_messages",
    },
    "character_text_proactive_triggers": {
        "character_id": "characters",
        "thread_id": "character_text_threads",
        "text_message_id": "character_text_messages",
    },
}

_JSON_COLUMNS_BY_TABLE: dict[str, frozenset[str]] = {
    "world_state": frozenset({"value_json"}),
    "context_sources": frozenset({"metadata_json"}),
    "context_observations": frozenset(
        {"source_message_ids_json", "tags_json", "metadata_json"}
    ),
    "locations": frozenset(
        {"aliases_json", "connections_json", "hazards_json", "locked_fields_json"}
    ),
    "characters": frozenset(
        {"aliases_json", "relationships_json", "locked_fields_json"}
    ),
    "scene_snapshots": frozenset(
        {
            "nearby_objects_json",
            "hazards_json",
            "present_character_ids_json",
            "locked_fields_json",
        }
    ),
    "active_threads": frozenset({"related_entities_json", "locked_fields_json"}),
    "context_update_suggestions": frozenset(
        {"proposed_value_json", "source_message_ids_json"}
    ),
    "context_update_audit": frozenset(
        {"before_json", "after_json", "source_message_ids_json"}
    ),
    "state_changes": frozenset({"before_json", "after_json"}),
    "memories": frozenset({"tags_json", "source_message_ids_json"}),
    "save_scenario_updates": frozenset({"content_json", "source_message_ids_json"}),
    "save_loss_condition_changes": frozenset({"before_json", "after_json"}),
    "save_loss_outcomes": frozenset({"evidence_json"}),
    "media_assets": frozenset({"metadata_json"}),
    "character_knowledge_edges": frozenset({"source_message_ids_json"}),
    "dating_route_states": frozenset(
        {"known_boundaries_json", "unresolved_questions_json"}
    ),
    "character_text_message_attachments": frozenset({"metadata_json"}),
}

_ENTITY_TABLES = {
    "location": "locations",
    "character": "characters",
    "thread": "active_threads",
    "active_thread": "active_threads",
    "scene_snapshot": "scene_snapshots",
    "loss_condition": "save_loss_conditions",
    "media_asset": "media_assets",
    "save": "saves",
    "state": "world_state",
    "world_state": "world_state",
    "memory": "memories",
    "summary": "summaries",
    "context_source": "context_sources",
    "dating_route_state": "dating_route_states",
    "character_text_thread": "character_text_threads",
    "character_text_message": "character_text_messages",
    "character_text_message_attachment": "character_text_message_attachments",
    "character_text_provenance": "character_text_provenance",
    "character_contact_state": "character_contact_states",
    "character_knowledge_edge": "character_knowledge_edges",
    "character_text_proactive_trigger": "character_text_proactive_triggers",
}


class TurnSnapshotService:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories

    def capture_baseline_snapshot(
        self,
        save_id: str,
        *,
        reason: str = "baseline",
    ) -> TurnSnapshotRecord:
        return self._capture_snapshot(
            save_id=save_id,
            message_id=None,
            reason=reason,
        )

    def capture_message_snapshot(
        self,
        *,
        save_id: str,
        message_id: str,
        reason: str = "message",
    ) -> TurnSnapshotRecord:
        row = self.repositories.connection.execute(
            """
            SELECT 1
            FROM messages
            WHERE save_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (save_id, message_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown active message id: {message_id}")
        return self._capture_snapshot(
            save_id=save_id,
            message_id=message_id,
            reason=reason,
        )

    def capture_current_head_if_dirty(
        self,
        save_id: str,
        *,
        reason: str = "dirty_head",
    ) -> TurnSnapshotRecord:
        message = self._latest_active_message(save_id)
        message_id = message.id if message is not None else None
        current_revision = self._context_revision(save_id)
        latest = self.latest_snapshot_for_message(
            save_id=save_id,
            message_id=message_id,
        )
        prepared = self._prepare_snapshot(
            save_id=save_id,
            message_id=message_id,
        )
        if (
            latest is not None
            and latest.context_revision == current_revision
            and latest.root_manifest_hash == prepared.root_manifest_hash
        ):
            self.repositories.commit()
            return latest
        return self._insert_prepared_snapshot(
            save_id=save_id,
            message_id=message_id,
            reason=reason,
            prepared=prepared,
        )

    def latest_snapshot_for_message(
        self,
        *,
        save_id: str,
        message_id: str | None,
    ) -> TurnSnapshotRecord | None:
        if message_id is None:
            row = self.repositories.connection.execute(
                """
                SELECT id, save_id, message_id, parent_snapshot_id,
                       root_manifest_hash, context_revision, reason, created_at
                FROM save_turn_snapshots
                WHERE save_id = ? AND message_id IS NULL
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (save_id,),
            ).fetchone()
        else:
            row = self.repositories.connection.execute(
                """
                SELECT id, save_id, message_id, parent_snapshot_id,
                       root_manifest_hash, context_revision, reason, created_at
                FROM save_turn_snapshots
                WHERE save_id = ? AND message_id = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (save_id, message_id),
            ).fetchone()
        return _snapshot_record_from_row(row) if row is not None else None

    def latest_snapshot_before_message(
        self,
        *,
        save_id: str,
        message_id: str,
    ) -> TurnSnapshotRecord | None:
        messages = self.repositories.list_messages(save_id)
        selected_index = _message_index(messages, message_id)
        if selected_index is None:
            raise ValueError(f"Unknown active message id: {message_id}")
        if selected_index == 0:
            return self.latest_snapshot_for_message(save_id=save_id, message_id=None)
        previous = messages[selected_index - 1]
        return self.latest_snapshot_for_message(
            save_id=save_id,
            message_id=previous.id,
        )

    def restore_delete_from_message(
        self,
        *,
        save_id: str,
        message_id: str,
    ) -> SnapshotDeletion | None:
        messages = self.repositories.list_messages(save_id)
        selected_index = _message_index(messages, message_id)
        if selected_index is None:
            raise ValueError(f"Unknown active message id: {message_id}")
        snapshot = self.latest_snapshot_before_message(
            save_id=save_id,
            message_id=message_id,
        )
        if snapshot is None:
            return None
        deleted_messages = tuple(messages[selected_index:])
        self.restore_save_to_snapshot(save_id=save_id, snapshot_id=snapshot.id)
        return SnapshotDeletion(snapshot=snapshot, deleted_messages=deleted_messages)

    def restore_save_to_snapshot(self, *, save_id: str, snapshot_id: str) -> None:
        snapshot = self._get_snapshot(snapshot_id)
        if snapshot.save_id != save_id:
            raise ValueError(
                f"Snapshot {snapshot_id} does not belong to save {save_id}"
            )
        manifest = self._snapshot_manifest(snapshot)
        rows_by_table = _sanitize_snapshot_rows_for_safety(
            self._rows_from_manifest(manifest)
        )
        raw_active_message_ids = manifest.get("active_message_ids", [])
        if not isinstance(raw_active_message_ids, list):
            raw_active_message_ids = []
        active_message_ids = tuple(
            str(message_id) for message_id in raw_active_message_ids
        )
        self.repositories.begin_transaction()
        try:
            self._restore_messages(
                save_id=save_id,
                rows=rows_by_table.get("messages", ()),
                active_message_ids=active_message_ids,
            )
            for table_name in _RESTORE_DELETE_ORDER:
                self.repositories.connection.execute(
                    f"DELETE FROM {table_name} WHERE save_id = ?",
                    (save_id,),
                )
            for table_name in _RESTORE_INSERT_ORDER:
                rows = rows_by_table.get(table_name, ())
                if table_name == "locations":
                    rows = _location_rows_parent_first(rows)
                elif table_name == "media_assets":
                    rows = _media_rows_parent_first(rows)
                for row in rows:
                    self._insert_row(table_name, row)
            self.repositories.ensure_context_observation_curation_states(save_id)
            _remove_snapshot_safety_transition_records(
                self.repositories.connection,
                save_id=save_id,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        log_event(
            "save.snapshot_restored",
            save_id=save_id,
            snapshot_id=snapshot.id,
            message_id=snapshot.message_id,
            active_message_count=len(active_message_ids),
        )

    def fork_snapshot_to_save(
        self,
        *,
        source_save_id: str,
        snapshot_id: str,
        title: str,
        media_dir: Path,
        owner_user_id: str | None = None,
    ) -> SnapshotForkResult:
        source_save = self.repositories.get_save(source_save_id)
        if source_save is None:
            raise ValueError(f"Unknown save id: {source_save_id}")
        snapshot = self._get_snapshot(snapshot_id)
        if snapshot.save_id != source_save_id:
            raise ValueError(
                f"Snapshot {snapshot_id} does not belong to save {source_save_id}"
            )
        manifest = self._snapshot_manifest(snapshot)
        rows_by_table = _sanitize_snapshot_rows_for_safety(
            self._rows_from_manifest(manifest)
        )
        remapper = _SnapshotRemapper(
            source_save_id=source_save_id,
            target_save_id=str(uuid4()),
            rows_by_table=rows_by_table,
        )
        copied_paths: list[Path] = []
        self.repositories.begin_transaction()
        try:
            fork_save = self.repositories.create_save(
                save_id=remapper.target_save_id,
                scenario_id=source_save.scenario_id,
                title=title,
                custom_instructions=source_save.custom_instructions,
                owner_user_id=owner_user_id or source_save.owner_user_id,
            )
            for row in rows_by_table.get("messages", ()):
                self._insert_row(
                    "messages",
                    _sanitize_snapshot_message_row(
                        remapper.remap_row("messages", row)
                    ),
                )
            self.repositories.copy_save_scoped_settings(
                source_save_id=source_save_id,
                target_save_id=fork_save.id,
            )
            for table_name in _RESTORE_INSERT_ORDER:
                rows = rows_by_table.get(table_name, ())
                if table_name == "locations":
                    rows = _location_rows_parent_first(rows)
                elif table_name == "media_assets":
                    rows = _media_rows_parent_first(rows)
                for row in rows:
                    remapped = remapper.remap_row(table_name, row)
                    if table_name == "media_assets":
                        remapped = _copy_snapshot_media_row(
                            media_dir=media_dir,
                            row=row,
                            remapped=remapped,
                            fork_save_id=fork_save.id,
                            copied_paths=copied_paths,
                        )
                    self._insert_row(table_name, remapped)
            self.repositories.ensure_context_observation_curation_states(fork_save.id)
            _remove_snapshot_safety_transition_records(
                self.repositories.connection,
                save_id=fork_save.id,
            )
            self._import_remapped_snapshots(
                source_save_id=source_save_id,
                target_save_id=fork_save.id,
                message_id_map=remapper.id_maps["messages"],
                id_maps=remapper.id_maps,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            _delete_copied_paths(copied_paths, media_dir=media_dir)
            raise

        media_count = len(rows_by_table.get("media_assets", ()))
        log_event(
            "save.snapshot_forked",
            source_save_id=source_save_id,
            fork_save_id=fork_save.id,
            snapshot_id=snapshot_id,
            message_count=len(rows_by_table.get("messages", ())),
            media_count=media_count,
        )
        return SnapshotForkResult(
            save=fork_save,
            message_count=len(rows_by_table.get("messages", ())),
            media_count=media_count,
        )

    def export_snapshot_rows(
        self,
        *,
        save_id: str,
        active_message_ids: Iterable[str],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        message_ids = frozenset(active_message_ids)
        snapshots = self._exported_snapshots(save_id=save_id, message_ids=message_ids)
        objects_by_hash: dict[str, dict[str, object]] = {}
        snapshot_rows: list[dict[str, object]] = []
        for snapshot in snapshots:
            manifest = self._snapshot_manifest(snapshot)
            rows_by_table = _sanitize_snapshot_rows_for_safety(
                self._rows_from_manifest(manifest)
            )
            tables: dict[str, list[dict[str, str]]] = {}
            for table_name, rows in rows_by_table.items():
                entries: list[dict[str, str]] = []
                for row in rows:
                    object_hash = _add_snapshot_object_export(
                        objects_by_hash,
                        kind=f"row:{table_name}",
                        value=row,
                        created_at=snapshot.created_at,
                    )
                    entries.append(
                        {
                            "id": str(row.get("id", "")),
                            "object_hash": object_hash,
                        }
                    )
                tables[table_name] = entries
            root_manifest_hash = _add_snapshot_object_export(
                objects_by_hash,
                kind="snapshot_manifest",
                value={
                    **manifest,
                    "tables": tables,
                },
                created_at=snapshot.created_at,
            )
            snapshot_rows.append(
                {
                    "id": snapshot.id,
                    "save_id": snapshot.save_id,
                    "message_id": snapshot.message_id,
                    "parent_snapshot_id": snapshot.parent_snapshot_id,
                    "root_manifest_hash": root_manifest_hash,
                    "context_revision": snapshot.context_revision,
                    "reason": snapshot.reason,
                    "created_at": snapshot.created_at,
                }
            )
        return snapshot_rows, list(objects_by_hash.values())

    def import_snapshot_rows(
        self,
        *,
        snapshot_rows: Iterable[Mapping[str, object]],
        object_rows: Iterable[Mapping[str, object]],
        source_save_id: str,
        target_save_id: str,
        message_id_map: Mapping[str, str],
        id_maps: dict[str, dict[str, str]] | None = None,
        media_path_map: Mapping[tuple[str, str], str] | None = None,
    ) -> int:
        snapshots = [dict(row) for row in snapshot_rows]
        objects = [dict(row) for row in object_rows]
        self.validate_exported_snapshot_rows(
            snapshot_rows=snapshots,
            object_rows=objects,
        )
        objects_by_hash = {
            _text(row, "object_hash"): dict(row) for row in objects
        }
        if not snapshots:
            return 0
        source_rows_by_manifest = {
            _text(row, "root_manifest_hash"): _sanitize_snapshot_rows_for_safety(
                self._rows_from_exported_manifest(
                    objects_by_hash,
                    _text(row, "root_manifest_hash"),
                ),
                quarantine_content_ratings=True,
            )
            for row in snapshots
        }
        remapper = _SnapshotRemapper(
            source_save_id=source_save_id,
            target_save_id=target_save_id,
            rows_by_table=_merge_rows_by_table(source_rows_by_manifest.values()),
            id_maps=id_maps,
            message_id_map=message_id_map,
            media_path_map=media_path_map,
        )
        snapshot_id_map = {
            _text(row, "id"): str(uuid4()) for row in snapshots
        }
        manifest_hash_map: dict[str, str] = {}
        for row in snapshots:
            old_manifest_hash = _text(row, "root_manifest_hash")
            rows_by_table = source_rows_by_manifest[old_manifest_hash]
            manifest_object = _decode_exported_object(
                objects_by_hash[old_manifest_hash]
            )
            if not isinstance(manifest_object, dict):
                raise ValueError("Snapshot manifest object is not a JSON object")
            manifest = cast(dict[str, object], manifest_object)
            remapped_tables: dict[str, list[dict[str, str]]] = {}
            for table_name, rows in rows_by_table.items():
                remapped_entries: list[dict[str, str]] = []
                for row_data in rows:
                    remapped_row = remapper.remap_row(table_name, row_data)
                    if table_name == "messages":
                        remapped_row = _sanitize_snapshot_message_row(remapped_row)
                    remapped_entries.append(
                        {
                            "id": str(remapped_row.get("id", "")),
                            "object_hash": self._store_object(
                                kind=f"row:{table_name}",
                                value=remapped_row,
                            ),
                        }
                    )
                remapped_tables[table_name] = remapped_entries
            raw_active_message_ids = manifest.get("active_message_ids", [])
            if not isinstance(raw_active_message_ids, list):
                raw_active_message_ids = []
            remapped_manifest = {
                **manifest,
                "save_id": target_save_id,
                "message_id": _remap_optional(
                    cast(str | None, manifest.get("message_id")),
                    dict(message_id_map),
                ),
                "active_message_ids": [
                    message_id_map.get(str(message_id), str(message_id))
                    for message_id in raw_active_message_ids
                ],
                "tables": remapped_tables,
            }
            manifest_hash_map[old_manifest_hash] = self._store_object(
                kind="snapshot_manifest",
                value=remapped_manifest,
            )
        for row in snapshots:
            original_id = _text(row, "id")
            original_parent_id = _optional_text(row, "parent_snapshot_id")
            self.repositories.connection.execute(
                """
                INSERT OR IGNORE INTO save_turn_snapshots(
                    id, save_id, message_id, parent_snapshot_id,
                    root_manifest_hash, context_revision, reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id_map[original_id],
                    target_save_id,
                    _remap_optional(
                        _optional_text(row, "message_id"),
                        dict(message_id_map),
                    ),
                    (
                        snapshot_id_map.get(original_parent_id)
                        if original_parent_id is not None
                        else None
                    ),
                    manifest_hash_map[_text(row, "root_manifest_hash")],
                    _int(row, "context_revision", default=0),
                    _optional_text(row, "reason") or "imported",
                    _optional_text(row, "created_at"),
                ),
            )
        self.repositories.commit()
        return len(snapshots)

    @staticmethod
    def validate_exported_snapshot_rows(
        *,
        snapshot_rows: Iterable[Mapping[str, object]],
        object_rows: Iterable[Mapping[str, object]],
    ) -> None:
        _validate_exported_snapshot_rows(snapshot_rows, object_rows)

    def media_asset_rows_from_snapshot_objects(
        self,
        object_rows: Iterable[Mapping[str, object]],
    ) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for row in object_rows:
            if _optional_text(row, "kind") != "row:media_assets":
                continue
            value = _decode_exported_object(row)
            if not isinstance(value, dict):
                continue
            media_row = cast(dict[str, object], value)
            row_id = media_row.get("id")
            if isinstance(row_id, str):
                if row_id in seen:
                    continue
                seen.add(row_id)
            rows.append(media_row)
        return tuple(rows)

    def media_asset_records_from_save_snapshots(
        self,
        save_id: str,
    ) -> tuple[MediaAssetRecord, ...]:
        records: dict[tuple[str, str, str | None], MediaAssetRecord] = {}
        for snapshot in self._snapshots_for_save(save_id):
            manifest = self._snapshot_manifest(snapshot)
            rows_by_table = _sanitize_snapshot_rows_for_safety(
                self._rows_from_manifest(manifest)
            )
            for row in rows_by_table.get("media_assets", ()):
                if row.get("save_id") != save_id:
                    continue
                record = _media_asset_record_from_snapshot_row(row)
                records.setdefault(
                    (record.id, record.path, record.thumbnail_path),
                    record,
                )
        return tuple(records.values())

    def prune_unreferenced_snapshot_objects(self) -> int:
        reachable_hashes = self._reachable_snapshot_object_hashes()
        rows = self.repositories.connection.execute(
            """
            SELECT object_hash
            FROM save_snapshot_objects
            ORDER BY rowid
            """
        ).fetchall()
        unreferenced_hashes = sorted(
            {str(row["object_hash"]) for row in rows} - reachable_hashes
        )
        deleted_count = 0
        for index in range(0, len(unreferenced_hashes), 500):
            chunk = unreferenced_hashes[index : index + 500]
            cursor = self.repositories.connection.execute(
                f"""
                DELETE FROM save_snapshot_objects
                WHERE object_hash IN ({_placeholders(len(chunk))})
                """,
                tuple(chunk),
            )
            deleted_count += int(cursor.rowcount or 0)
        self.repositories.commit()
        return deleted_count

    def _capture_snapshot(
        self,
        *,
        save_id: str,
        message_id: str | None,
        reason: str,
    ) -> TurnSnapshotRecord:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")
        prepared = self._prepare_snapshot(save_id=save_id, message_id=message_id)
        return self._insert_prepared_snapshot(
            save_id=save_id,
            message_id=message_id,
            reason=reason,
            prepared=prepared,
        )

    def _prepare_snapshot(
        self,
        *,
        save_id: str,
        message_id: str | None,
    ) -> _PreparedSnapshot:
        _remove_snapshot_safety_transition_records(
            self.repositories.connection,
            save_id=save_id,
        )
        rows_by_table = _sanitize_snapshot_rows_for_safety(
            self._active_rows_by_table(save_id)
        )
        active_message_ids = [
            str(row["id"]) for row in rows_by_table.get("messages", ())
        ]
        tables: dict[str, list[dict[str, str]]] = {}
        for table_name in _SNAPSHOT_TABLE_NAMES:
            table_entries: list[dict[str, str]] = []
            for row in rows_by_table.get(table_name, ()):
                row_id = row.get("id")
                object_hash = self._store_object(
                    kind=f"row:{table_name}",
                    value=row,
                )
                table_entries.append(
                    {
                        "id": str(row_id) if row_id is not None else "",
                        "object_hash": object_hash,
                    }
                )
            tables[table_name] = table_entries
        context_revision = self._context_revision(save_id)
        manifest = {
            "format": SNAPSHOT_FORMAT,
            "save_id": save_id,
            "message_id": message_id,
            "active_message_ids": active_message_ids,
            "context_revision": context_revision,
            "tables": tables,
        }
        root_manifest_hash = self._store_object(
            kind="snapshot_manifest",
            value=manifest,
        )
        return _PreparedSnapshot(
            root_manifest_hash=root_manifest_hash,
            context_revision=context_revision,
            tables=tables,
        )

    def _insert_prepared_snapshot(
        self,
        *,
        save_id: str,
        message_id: str | None,
        reason: str,
        prepared: _PreparedSnapshot,
    ) -> TurnSnapshotRecord:
        parent = self._latest_snapshot(save_id)
        snapshot_id = str(uuid4())
        self.repositories.connection.execute(
            """
            INSERT INTO save_turn_snapshots(
                id, save_id, message_id, parent_snapshot_id, root_manifest_hash,
                context_revision, reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                save_id,
                message_id,
                parent.id if parent is not None else None,
                prepared.root_manifest_hash,
                prepared.context_revision,
                reason,
            ),
        )
        self.repositories.commit()
        snapshot = self._get_snapshot(snapshot_id)
        log_event(
            "save.snapshot_captured",
            save_id=save_id,
            snapshot_id=snapshot.id,
            message_id=message_id,
            context_revision=prepared.context_revision,
            reason=reason,
            object_count=sum(len(entries) for entries in prepared.tables.values()),
        )
        return snapshot

    def _active_rows_by_table(
        self,
        save_id: str,
    ) -> dict[str, tuple[dict[str, object], ...]]:
        rows: dict[str, tuple[dict[str, object], ...]] = {}
        for table in _SNAPSHOT_TABLES:
            where = "save_id = ?"
            if table.name in {"messages", "character_text_messages"}:
                where += " AND deleted_at IS NULL"
            elif table.active_only and "archived_at" in self._column_names(table.name):
                where += " AND archived_at IS NULL"
            table_rows = [
                _row_dict(row)
                for row in self.repositories.connection.execute(
                    f"""
                    SELECT *
                    FROM {table.name}
                    WHERE {where}
                    ORDER BY {table.order_by}
                    """,
                    (save_id,),
                ).fetchall()
            ]
            if table.name == "locations":
                rows[table.name] = _location_rows_parent_first(table_rows)
            elif table.name == "media_assets":
                rows[table.name] = _media_rows_parent_first(table_rows)
            elif table.name == "context_observation_curation_state":
                rows[table.name] = tuple(
                    {
                        **row,
                        "lease_token": None,
                        "lease_until": None,
                    }
                    for row in table_rows
                )
            else:
                rows[table.name] = tuple(table_rows)
        _filter_character_text_snapshot_rows(rows)
        return rows

    def _store_object(self, *, kind: str, value: object) -> str:
        payload = _canonical_json_bytes(value)
        object_hash = _snapshot_object_hash(kind=kind, payload=payload)
        self.repositories.connection.execute(
            """
            INSERT OR IGNORE INTO save_snapshot_objects(
                object_hash, kind, encoding, payload, uncompressed_size
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                object_hash,
                kind,
                SNAPSHOT_ENCODING,
                zlib.compress(payload),
                len(payload),
            ),
        )
        return object_hash

    def _get_snapshot(self, snapshot_id: str) -> TurnSnapshotRecord:
        row = self.repositories.connection.execute(
            """
            SELECT id, save_id, message_id, parent_snapshot_id,
                   root_manifest_hash, context_revision, reason, created_at
            FROM save_turn_snapshots
            WHERE id = ?
            """,
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown snapshot id: {snapshot_id}")
        return _snapshot_record_from_row(row)

    def _snapshots_for_save(self, save_id: str) -> tuple[TurnSnapshotRecord, ...]:
        rows = self.repositories.connection.execute(
            """
            SELECT id, save_id, message_id, parent_snapshot_id,
                   root_manifest_hash, context_revision, reason, created_at
            FROM save_turn_snapshots
            WHERE save_id = ?
            ORDER BY rowid
            """,
            (save_id,),
        ).fetchall()
        return tuple(_snapshot_record_from_row(row) for row in rows)

    def _latest_snapshot(self, save_id: str) -> TurnSnapshotRecord | None:
        row = self.repositories.connection.execute(
            """
            SELECT id, save_id, message_id, parent_snapshot_id,
                   root_manifest_hash, context_revision, reason, created_at
            FROM save_turn_snapshots
            WHERE save_id = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (save_id,),
        ).fetchone()
        return _snapshot_record_from_row(row) if row is not None else None

    def _snapshot_manifest(self, snapshot: TurnSnapshotRecord) -> dict[str, object]:
        manifest = self._load_object(snapshot.root_manifest_hash)
        if not isinstance(manifest, dict) or manifest.get("format") != SNAPSHOT_FORMAT:
            raise ValueError(f"Invalid snapshot manifest: {snapshot.id}")
        return cast(dict[str, object], manifest)

    def _load_object(self, object_hash: str) -> object:
        row = self.repositories.connection.execute(
            """
            SELECT kind, encoding, payload, uncompressed_size
            FROM save_snapshot_objects
            WHERE object_hash = ?
            """,
            (object_hash,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing snapshot object: {object_hash}")
        if row["encoding"] != SNAPSHOT_ENCODING:
            raise ValueError(f"Unsupported snapshot object encoding: {row['encoding']}")
        payload = zlib.decompress(bytes(row["payload"]))
        if len(payload) != int(row["uncompressed_size"]):
            raise ValueError(f"Snapshot object size mismatch: {object_hash}")
        if _snapshot_object_hash(
            kind=str(row["kind"]),
            payload=payload,
        ) != object_hash:
            raise ValueError(f"Snapshot object hash mismatch: {object_hash}")
        return json.loads(payload.decode("utf-8"))

    def _reachable_snapshot_object_hashes(self) -> set[str]:
        rows = self.repositories.connection.execute(
            """
            SELECT root_manifest_hash
            FROM save_turn_snapshots
            ORDER BY rowid
            """
        ).fetchall()
        reachable = {str(row["root_manifest_hash"]) for row in rows}
        for object_hash in tuple(reachable):
            manifest = self._load_object(object_hash)
            if not isinstance(manifest, Mapping):
                raise ValueError(
                    f"Snapshot manifest object is not valid: {object_hash}"
                )
            for table_rows in _manifest_tables(manifest).values():
                for entry in table_rows:
                    reachable.add(_text(entry, "object_hash"))
        return reachable

    def _rows_from_manifest(
        self,
        manifest: Mapping[str, object],
    ) -> dict[str, tuple[dict[str, object], ...]]:
        rows_by_table: dict[str, tuple[dict[str, object], ...]] = {}
        for table_name, entries in _manifest_tables(manifest).items():
            rows: list[dict[str, object]] = []
            for entry in entries:
                object_hash = _text(entry, "object_hash")
                value = self._load_object(object_hash)
                if not isinstance(value, dict):
                    raise ValueError(f"Snapshot row object is not a row: {object_hash}")
                rows.append(cast(dict[str, object], value))
            rows_by_table[table_name] = tuple(rows)
        return rows_by_table

    def _rows_from_exported_manifest(
        self,
        objects_by_hash: Mapping[str, Mapping[str, object]],
        manifest_hash: str,
    ) -> dict[str, tuple[dict[str, object], ...]]:
        manifest_object = _decode_exported_object(objects_by_hash[manifest_hash])
        if not isinstance(manifest_object, Mapping):
            raise ValueError("Snapshot manifest object is not a JSON object")
        manifest = cast(Mapping[str, object], manifest_object)
        rows_by_table: dict[str, tuple[dict[str, object], ...]] = {}
        for table_name, entries in _manifest_tables(manifest).items():
            rows: list[dict[str, object]] = []
            for entry in entries:
                value = _decode_exported_object(
                    objects_by_hash[_text(entry, "object_hash")]
                )
                if isinstance(value, dict):
                    rows.append(cast(dict[str, object], value))
            rows_by_table[table_name] = tuple(rows)
        return rows_by_table

    def _restore_messages(
        self,
        *,
        save_id: str,
        rows: Iterable[Mapping[str, object]],
        active_message_ids: tuple[str, ...],
    ) -> None:
        for row in rows:
            self._upsert_row("messages", _sanitize_snapshot_message_row(row))
        if active_message_ids:
            self.repositories.connection.execute(
                f"""
                UPDATE messages
                SET deleted_at = CURRENT_TIMESTAMP
                WHERE save_id = ?
                  AND id NOT IN ({_placeholders(len(active_message_ids))})
                  AND deleted_at IS NULL
                """,
                (save_id, *active_message_ids),
            )
            self.repositories.connection.execute(
                f"""
                UPDATE messages
                SET deleted_at = NULL
                WHERE save_id = ?
                  AND id IN ({_placeholders(len(active_message_ids))})
                """,
                (save_id, *active_message_ids),
            )
            return
        self.repositories.connection.execute(
            """
            UPDATE messages
            SET deleted_at = CURRENT_TIMESTAMP
            WHERE save_id = ? AND deleted_at IS NULL
            """,
            (save_id,),
        )

    def _insert_row(self, table_name: str, row: Mapping[str, object]) -> None:
        columns = [
            column
            for column in row
            if column in self._column_names(table_name)
        ]
        if not columns:
            return
        self.repositories.connection.execute(
            f"""
            INSERT INTO {table_name}({", ".join(columns)})
            VALUES ({_placeholders(len(columns))})
            """,
            tuple(row[column] for column in columns),
        )

    def _upsert_row(self, table_name: str, row: Mapping[str, object]) -> None:
        columns = [
            column
            for column in row
            if column in self._column_names(table_name)
        ]
        if "id" not in columns:
            self._insert_row(table_name, row)
            return
        assignments = ", ".join(
            f"{column} = excluded.{column}" for column in columns if column != "id"
        )
        self.repositories.connection.execute(
            f"""
            INSERT INTO {table_name}({", ".join(columns)})
            VALUES ({_placeholders(len(columns))})
            ON CONFLICT(id) DO UPDATE SET {assignments}
            """,
            tuple(row[column] for column in columns),
        )

    def _context_revision(self, save_id: str) -> int:
        row = self.repositories.connection.execute(
            """
            SELECT revision
            FROM save_context_revisions
            WHERE save_id = ?
            """,
            (save_id,),
        ).fetchone()
        return int(row["revision"]) if row is not None else 0

    def _latest_active_message(self, save_id: str) -> MessageRecord | None:
        row = self.repositories.connection.execute(
            """
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition
            FROM messages
            WHERE save_id = ? AND deleted_at IS NULL
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (save_id,),
        ).fetchone()
        return MessageRecord(**dict(row)) if row is not None else None

    def _column_names(self, table_name: str) -> frozenset[str]:
        rows = self.repositories.connection.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()
        return frozenset(str(row["name"]) for row in rows)

    def _exported_snapshots(
        self,
        *,
        save_id: str,
        message_ids: frozenset[str],
    ) -> tuple[TurnSnapshotRecord, ...]:
        rows = self.repositories.connection.execute(
            """
            SELECT id, save_id, message_id, parent_snapshot_id,
                   root_manifest_hash, context_revision, reason, created_at
            FROM save_turn_snapshots
            WHERE save_id = ?
              AND (message_id IS NULL OR message_id IN (
                    SELECT id FROM messages WHERE save_id = ? AND deleted_at IS NULL
                  ))
            ORDER BY rowid
            """,
            (save_id, save_id),
        ).fetchall()
        return tuple(
            snapshot
            for snapshot in (_snapshot_record_from_row(row) for row in rows)
            if snapshot.message_id is None or snapshot.message_id in message_ids
        )

    def _import_remapped_snapshots(
        self,
        *,
        source_save_id: str,
        target_save_id: str,
        message_id_map: Mapping[str, str],
        id_maps: dict[str, dict[str, str]],
    ) -> None:
        active_message_ids = frozenset(message_id_map)
        snapshot_rows, object_rows = self.export_snapshot_rows(
            save_id=source_save_id,
            active_message_ids=active_message_ids,
        )
        self.import_snapshot_rows(
            snapshot_rows=snapshot_rows,
            object_rows=object_rows,
            source_save_id=source_save_id,
            target_save_id=target_save_id,
            message_id_map=message_id_map,
            id_maps=id_maps,
        )


class _SnapshotRemapper:
    def __init__(
        self,
        *,
        source_save_id: str,
        target_save_id: str,
        rows_by_table: Mapping[str, Iterable[Mapping[str, object]]],
        id_maps: dict[str, dict[str, str]] | None = None,
        message_id_map: Mapping[str, str] | None = None,
        media_path_map: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        self.source_save_id = source_save_id
        self.target_save_id = target_save_id
        self.id_maps = id_maps or {}
        self.media_path_map = dict(media_path_map or {})
        self.id_maps.setdefault("saves", {source_save_id: target_save_id})
        self.id_maps.setdefault("save", self.id_maps["saves"])
        self.id_maps.setdefault("messages", dict(message_id_map or {}))
        self.id_maps.setdefault("message", self.id_maps["messages"])
        for row in rows_by_table.get("messages", ()):
            row_id = row.get("id")
            if isinstance(row_id, str):
                self.id_maps["messages"].setdefault(row_id, str(uuid4()))
        self.id_maps["message"] = self.id_maps["messages"]
        for table_name, rows in rows_by_table.items():
            if table_name == "messages":
                continue
            table_map = self.id_maps.setdefault(table_name, {})
            for row in rows:
                row_id = row.get("id")
                if isinstance(row_id, str):
                    table_map.setdefault(row_id, str(uuid4()))

    def remap_row(
        self,
        table_name: str,
        row: Mapping[str, object],
    ) -> dict[str, object]:
        remapped: dict[str, object] = {}
        for column, value in row.items():
            if column == "id":
                remapped[column] = self._mapped_table_id(table_name, value)
            elif column == "save_id":
                remapped[column] = self.target_save_id
            elif column in _MESSAGE_REFERENCE_COLUMNS:
                remapped[column] = self._mapped_message_id(value)
            elif column in _TABLE_REFERENCE_COLUMNS.get(table_name, {}):
                target_table = _TABLE_REFERENCE_COLUMNS[table_name][column]
                remapped[column] = self._mapped_table_id(target_table, value)
            elif table_name == "media_assets" and column in {"path", "thumbnail_path"}:
                remapped[column] = self._mapped_media_path(
                    row.get("id"),
                    column,
                    value,
                )
            elif table_name == "entity_links" and column in {"entity_id", "target_id"}:
                type_column = "entity_type" if column == "entity_id" else "target_type"
                remapped[column] = self._mapped_entity_id(
                    cast(str | None, row.get(type_column)),
                    value,
                )
            elif table_name == "context_sources" and column == "source_id":
                remapped[column] = self._mapped_entity_id(
                    cast(str | None, row.get("source_type")),
                    value,
                )
            elif table_name in {
                "context_update_suggestions",
                "context_update_audit",
            } and column == "entity_id":
                remapped[column] = self._mapped_entity_id(
                    cast(str | None, row.get("entity_type")),
                    value,
                )
            elif table_name == "character_knowledge_edges" and column == "target_id":
                remapped[column] = self._mapped_entity_id(
                    cast(str | None, row.get("target_type")),
                    value,
                )
            elif table_name == "character_text_provenance" and column == "target_id":
                remapped[column] = self._mapped_entity_id(
                    cast(str | None, row.get("target_type")),
                    value,
                )
            elif (
                table_name == "character_text_proactive_triggers"
                and column == "source_id"
            ):
                remapped[column] = self._mapped_entity_id(
                    cast(str | None, row.get("source_type")),
                    value,
                )
            elif (
                table_name == "character_text_proactive_triggers"
                and column == "trigger_key"
            ):
                remapped[column] = self._remap_trigger_key(value)
            elif column in _JSON_COLUMNS_BY_TABLE.get(table_name, frozenset()):
                remapped[column] = self._remap_json_text(table_name, column, value)
            else:
                remapped[column] = value
        return remapped

    def _mapped_message_id(self, value: object) -> object:
        if isinstance(value, str):
            return self.id_maps["messages"].get(value, value)
        return value

    def _mapped_table_id(self, table_name: str, value: object) -> object:
        if not isinstance(value, str):
            return value
        return self.id_maps.get(table_name, {}).get(value, value)

    def _mapped_entity_id(self, entity_type: str | None, value: object) -> object:
        if not isinstance(value, str) or entity_type is None:
            return value
        table_name = _ENTITY_TABLES.get(entity_type)
        if table_name is None:
            return self.id_maps["messages"].get(value, value)
        return self._mapped_table_id(table_name, value)

    def _mapped_media_path(
        self,
        row_id: object,
        column: str,
        value: object,
    ) -> object:
        if not isinstance(row_id, str) or not isinstance(value, str):
            return value
        return self.media_path_map.get((row_id, column), value)

    def _remap_json_text(
        self,
        table_name: str,
        column: str,
        value: object,
    ) -> object:
        if not isinstance(value, str):
            return value
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return value
        if table_name == "scene_snapshots" and column == "present_character_ids_json":
            return _compact_json(self._remap_id_list(raw, "characters"))
        if table_name == "active_threads" and column == "related_entities_json":
            return _compact_json(self._remap_related_entities(raw))
        return _compact_json(self._remap_json_value(raw))

    def _remap_id_list(self, raw: object, table_name: str) -> object:
        if not isinstance(raw, list):
            return raw
        return [
            self._mapped_table_id(table_name, item) if isinstance(item, str) else item
            for item in raw
        ]

    def _remap_related_entities(self, raw: object) -> object:
        if not isinstance(raw, list):
            return raw
        remapped: list[object] = []
        for item in raw:
            if not isinstance(item, str):
                remapped.append(item)
                continue
            entity_type, separator, entity_id = item.partition(":")
            if not separator:
                remapped.append(self._remap_string_id(item))
                continue
            mapped_id = self._mapped_entity_id(entity_type, entity_id)
            if isinstance(mapped_id, str):
                remapped.append(f"{entity_type}:{mapped_id}")
        return remapped

    def _remap_json_value(self, value: object) -> object:
        if isinstance(value, str):
            return self._remap_string_id(value)
        if isinstance(value, list):
            return [self._remap_json_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._remap_json_value(item)
                for key, item in value.items()
            }
        return value

    def _remap_string_id(self, value: str) -> str:
        for table_map in self.id_maps.values():
            mapped = table_map.get(value)
            if mapped is not None:
                return mapped
        entity_type, separator, entity_id = value.partition(":")
        if separator:
            mapped_value = self._mapped_entity_id(entity_type, entity_id)
            if isinstance(mapped_value, str):
                return f"{entity_type}:{mapped_value}"
        return value

    def _remap_trigger_key(self, value: object) -> object:
        if not isinstance(value, str):
            return value
        parts = value.split(":")
        if len(parts) < 2:
            return self._remap_string_id(value)
        table_name = {
            "active_thread": "active_threads",
            "dating_route": "dating_route_states",
            "character_intent": "characters",
        }.get(parts[0])
        if table_name is None:
            return self._remap_string_id(value)
        remapped = list(parts)
        mapped_entity_id = self._mapped_table_id(table_name, remapped[1])
        if isinstance(mapped_entity_id, str):
            remapped[1] = mapped_entity_id
        remapped[2:] = [
            self._remap_string_id(part) if part else part for part in remapped[2:]
        ]
        return ":".join(remapped)


def _snapshot_record_from_row(row: sqlite3.Row) -> TurnSnapshotRecord:
    return TurnSnapshotRecord(
        id=str(row["id"]),
        save_id=str(row["save_id"]),
        message_id=cast(str | None, row["message_id"]),
        parent_snapshot_id=cast(str | None, row["parent_snapshot_id"]),
        root_manifest_hash=str(row["root_manifest_hash"]),
        context_revision=int(row["context_revision"]),
        reason=str(row["reason"]),
        created_at=cast(str | None, row["created_at"]),
    )


def _message_index(messages: Iterable[MessageRecord], message_id: str) -> int | None:
    for index, message in enumerate(messages):
        if message.id == message_id:
            return index
    return None


def _manifest_tables(
    manifest: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    raw = manifest.get("tables")
    if not isinstance(raw, dict):
        raise ValueError("Snapshot manifest is missing table entries")
    tables: dict[str, list[dict[str, object]]] = {}
    for table_name, entries in raw.items():
        if table_name not in _TABLES_BY_NAME or not isinstance(entries, list):
            continue
        tables[str(table_name)] = [
            cast(dict[str, object], entry)
            for entry in entries
            if isinstance(entry, dict)
        ]
    return tables


def _row_dict(row: sqlite3.Row) -> dict[str, object]:
    return {key: row[key] for key in row.keys()}


def _filter_character_text_snapshot_rows(
    rows: dict[str, tuple[dict[str, object], ...]],
) -> None:
    character_ids = _row_ids(rows.get("characters", ()))
    message_ids = _row_ids(rows.get("messages", ()))
    media_asset_ids = _row_ids(rows.get("media_assets", ()))

    participant_candidate_rows = tuple(
        row
        for row in rows.get("character_text_thread_participants", ())
        if _required_row_ref_active(row, "character_id", character_ids)
    )
    participant_count_by_thread: dict[object, int] = {}
    for row in participant_candidate_rows:
        thread_id = row.get("thread_id")
        participant_count_by_thread[thread_id] = (
            participant_count_by_thread.get(thread_id, 0) + 1
        )
    thread_rows = tuple(
        row
        for row in rows.get("character_text_threads", ())
        if (
            row.get("kind") == "group"
            and participant_count_by_thread.get(row.get("id"), 0) >= 2
        )
        or (
            row.get("kind") != "group"
            and _required_row_ref_active(row, "character_id", character_ids)
        )
    )
    rows["character_text_threads"] = thread_rows
    thread_ids = _row_ids(thread_rows)
    rows["character_text_thread_participants"] = tuple(
        row
        for row in participant_candidate_rows
        if _required_row_ref_active(row, "thread_id", thread_ids)
    )

    candidate_text_message_rows = tuple(
        row
        for row in rows.get("character_text_messages", ())
        if _required_row_ref_active(row, "thread_id", thread_ids)
        and _optional_row_ref_active(row, "character_id", character_ids)
        and _optional_row_ref_active(row, "sender_character_id", character_ids)
    )
    text_message_rows = _prune_character_text_reply_chains(candidate_text_message_rows)
    rows["character_text_messages"] = text_message_rows
    text_message_ids = _row_ids(text_message_rows)

    rows["character_text_message_revisions"] = tuple(
        row
        for row in rows.get("character_text_message_revisions", ())
        if _required_row_ref_active(row, "text_message_id", text_message_ids)
    )
    rows["character_text_message_attachments"] = tuple(
        row
        for row in rows.get("character_text_message_attachments", ())
        if _required_row_ref_active(row, "thread_id", thread_ids)
        and _required_row_ref_active(row, "text_message_id", text_message_ids)
        and _required_row_ref_active(row, "character_id", character_ids)
        and _optional_row_ref_active(row, "media_asset_id", media_asset_ids)
    )
    rows["character_text_provenance"] = tuple(
        row
        for row in rows.get("character_text_provenance", ())
        if _required_row_ref_active(row, "thread_id", thread_ids)
        and _required_row_ref_active(row, "text_message_id", text_message_ids)
        and _optional_typed_target_active(row, rows)
    )
    activity_events = tuple(
        row
        for row in rows.get("character_text_activity_events", ())
        if _required_row_ref_active(row, "thread_id", thread_ids)
        and _optional_row_ref_active(row, "text_message_id", text_message_ids)
    )
    rows["character_text_activity_events"] = activity_events
    max_activity_ordinal = max(
        (_snapshot_row_int(row, "ordinal") for row in activity_events), default=0
    )
    rows["narrator_phone_activity_cursors"] = tuple(
        {
            **row,
            "last_activity_ordinal": min(
                _snapshot_row_int(row, "last_activity_ordinal"),
                max_activity_ordinal,
            ),
        }
        for row in rows.get("narrator_phone_activity_cursors", ())
        if _required_row_ref_active(row, "narrator_message_id", message_ids)
    )
    rows["character_contact_states"] = tuple(
        row
        for row in rows.get("character_contact_states", ())
        if _required_row_ref_active(row, "player_character_id", character_ids)
        and _required_row_ref_active(row, "character_id", character_ids)
        and _optional_row_ref_active(row, "source_message_id", message_ids)
        and _optional_row_ref_active(row, "source_text_message_id", text_message_ids)
    )
    rows["character_text_proactive_triggers"] = tuple(
        row
        for row in rows.get("character_text_proactive_triggers", ())
        if _required_row_ref_active(row, "character_id", character_ids)
        and _optional_row_ref_active(row, "thread_id", thread_ids)
        and _optional_row_ref_active(row, "text_message_id", text_message_ids)
        and _optional_row_ref_active(row, "source_message_id", message_ids)
    )


def _snapshot_row_int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _optional_typed_target_active(
    row: Mapping[str, object],
    rows: Mapping[str, Iterable[Mapping[str, object]]],
) -> bool:
    target_type = row.get("target_type")
    target_id = row.get("target_id")
    table_name = _ENTITY_TABLES.get(str(target_type))
    if table_name is None:
        return True
    return isinstance(target_id, str) and target_id in _row_ids(
        rows.get(table_name, ())
    )


def _prune_character_text_reply_chains(
    rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    current = rows
    while True:
        active_ids = _row_ids(current)
        pruned = tuple(
            row
            for row in current
            if _optional_row_ref_active(row, "reply_to_message_id", active_ids)
        )
        if len(pruned) == len(current):
            return pruned
        current = pruned


def _row_ids(rows: Iterable[Mapping[str, object]]) -> frozenset[str]:
    return frozenset(str(row["id"]) for row in rows if isinstance(row.get("id"), str))


def _required_row_ref_active(
    row: Mapping[str, object],
    column: str,
    active_ids: frozenset[str],
) -> bool:
    value = row.get(column)
    return isinstance(value, str) and value in active_ids


def _optional_row_ref_active(
    row: Mapping[str, object],
    column: str,
    active_ids: frozenset[str],
) -> bool:
    value = row.get(column)
    return not isinstance(value, str) or not value or value in active_ids


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sanitize_snapshot_message_row(
    row: Mapping[str, object],
) -> dict[str, object]:
    copied = dict(row)
    if copied.get("role") != "narrator":
        copied["safety_transition"] = ""
        return copied
    body = copied.get("body")
    marker = copied.get("safety_transition")
    if not isinstance(body, str):
        return copied
    normalized_body, normalized_marker = normalize_message_safety(
        role="narrator",
        body=body,
        safety_transition=marker if isinstance(marker, str) else "",
    )
    copied["body"] = normalized_body
    copied["safety_transition"] = normalized_marker
    return copied


def _sanitize_snapshot_rows_for_safety(
    rows_by_table: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    quarantine_content_ratings: bool = False,
) -> dict[str, tuple[dict[str, object], ...]]:
    rows = {
        table_name: tuple(dict(row) for row in table_rows)
        for table_name, table_rows in rows_by_table.items()
    }
    rows["messages"] = tuple(
        _sanitize_snapshot_message_row(row)
        for row in rows.get("messages", ())
    )
    rows["save_scenario_updates"] = tuple(
        _sanitize_snapshot_scenario_update_row(row)
        for row in rows.get("save_scenario_updates", ())
    )
    if quarantine_content_ratings:
        for table_name in (
            "messages",
            "character_text_messages",
            "message_action_choices",
            "summaries",
            "characters",
        ):
            rows[table_name] = tuple(
                {**row, "content_rating": "unclassified"}
                for row in rows.get(table_name, ())
            )
        rows["media_assets"] = tuple(
            _snapshot_media_row_with_unclassified_rating(row)
            for row in rows.get("media_assets", ())
        )
        rows["save_scenario_updates"] = tuple(
            _snapshot_scenario_row_with_unclassified_rating(row)
            for row in rows.get("save_scenario_updates", ())
        )
    transition_ids = {
        str(row["id"])
        for row in rows["messages"]
        if isinstance(row.get("id"), str)
        and is_fade_to_black_message(
            role=str(row.get("role", "")),
            body=str(row.get("body", "")),
            safety_transition=str(row.get("safety_transition", "")),
        )
    }
    if not transition_ids:
        _filter_context_observation_curation_snapshot_rows(rows)
        return rows
    message_order = {
        str(row["id"]): index
        for index, row in enumerate(rows["messages"])
        if isinstance(row.get("id"), str)
    }
    for table_name, table_rows in tuple(rows.items()):
        if table_name == "messages":
            continue
        sanitized_rows: list[dict[str, object]] = []
        for row in table_rows:
            if table_name == "summaries" and _summary_row_covers_transition(
                row,
                transition_ids=transition_ids,
                message_order=message_order,
            ):
                continue
            if table_name == "scene_snapshots" and _snapshot_row_references_transition(
                table_name,
                row,
                transition_ids,
            ):
                sanitized_rows.append(_clear_snapshot_transition_fields(row))
                continue
            if _snapshot_row_references_transition(
                table_name,
                row,
                transition_ids,
            ):
                continue
            sanitized_rows.append(row)
        rows[table_name] = tuple(sanitized_rows)
    _filter_context_observation_curation_snapshot_rows(rows)
    return rows


def _filter_context_observation_curation_snapshot_rows(
    rows: dict[str, tuple[dict[str, object], ...]],
) -> None:
    observation_ids = _row_ids(rows.get("context_observations", ()))
    rows["context_observation_curation_state"] = tuple(
        row
        for row in rows.get("context_observation_curation_state", ())
        if _required_row_ref_active(row, "observation_id", observation_ids)
    )


def _snapshot_media_row_with_unclassified_rating(
    row: Mapping[str, object],
) -> dict[str, object]:
    copied = dict(row)
    metadata = copied.get("metadata_json")
    try:
        parsed = json.loads(metadata) if isinstance(metadata, str) else {}
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    parsed["content_rating"] = "unclassified"
    copied["metadata_json"] = _compact_json(parsed)
    return copied


def _snapshot_scenario_row_with_unclassified_rating(
    row: Mapping[str, object],
) -> dict[str, object]:
    copied = dict(row)
    content = copied.get("content_json")
    try:
        parsed = json.loads(content) if isinstance(content, str) else {}
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    source = parsed.get("_source")
    parsed["_source"] = metadata_with_scenario_content_ratings(
        source if isinstance(source, Mapping) else None,
        aggregate_rating="unclassified",
    )
    starters = parsed.get("character_starters")
    if isinstance(starters, list):
        for starter in starters:
            if not isinstance(starter, dict):
                continue
            reference = starter.get("reference_image")
            if isinstance(reference, dict):
                reference["content_rating"] = "unclassified"
    copied["content_json"] = _compact_json(parsed)
    return copied


def _sanitize_snapshot_scenario_update_row(
    row: dict[str, object],
) -> dict[str, object]:
    content_json = row.get("content_json")
    if not isinstance(content_json, str):
        return row
    try:
        content = json.loads(content_json)
    except json.JSONDecodeError:
        return row
    if not isinstance(content, dict):
        return row
    cleaned = dict(row)
    cleaned["content_json"] = json.dumps(
        strip_deprecated_scenario_character_sections(content),
        sort_keys=True,
        separators=(",", ":"),
    )
    return cleaned


def _snapshot_row_references_transition(
    table_name: str,
    row: Mapping[str, object],
    transition_ids: set[str],
) -> bool:
    if any(
        isinstance(row.get(column), str) and row.get(column) in transition_ids
        for column in _MESSAGE_REFERENCE_COLUMNS
    ):
        return True
    return any(
        _json_value_contains_message_id(row.get(column), transition_ids)
        for column in _JSON_COLUMNS_BY_TABLE.get(table_name, frozenset())
    )


def _json_value_contains_message_id(value: object, message_ids: set[str]) -> bool:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    if isinstance(value, str):
        return value in message_ids
    if isinstance(value, list):
        return any(_json_value_contains_message_id(item, message_ids) for item in value)
    if isinstance(value, dict):
        return any(
            _json_value_contains_message_id(item, message_ids)
            for item in value.values()
        )
    return False


def _summary_row_covers_transition(
    row: Mapping[str, object],
    *,
    transition_ids: set[str],
    message_order: Mapping[str, int],
) -> bool:
    start = message_order.get(str(row.get("covers_message_start_id", "")))
    end = message_order.get(str(row.get("covers_message_end_id", "")))
    if start is None or end is None:
        return False
    lower, upper = sorted((start, end))
    return any(
        lower <= message_order.get(message_id, -1) <= upper
        for message_id in transition_ids
    )


def _clear_snapshot_transition_fields(row: Mapping[str, object]) -> dict[str, object]:
    cleared = dict(row)
    for field, value in {
        "current_location_id": None,
        "situation": "",
        "objective": "",
        "in_world_time": "",
        "time_of_day": "",
        "day_of_week": "",
        "weather": "",
        "mood": "",
        "nearby_objects_json": "[]",
        "hazards_json": "[]",
        "present_character_ids_json": "[]",
        "world_day_index": None,
        "world_time_day_index": None,
        "world_time_day_label": "",
        "world_time_phase": "",
        "world_time_clock_minutes": None,
        "world_time_period_label": "",
        "world_time_source_message_id": None,
        "world_time_confidence": None,
        "source_message_id": None,
        "first_seen_message_id": None,
        "last_updated_message_id": None,
    }.items():
        if field in cleared:
            cleared[field] = value
    return cleared


def _remove_snapshot_safety_transition_records(
    connection: sqlite3.Connection,
    *,
    save_id: str,
) -> None:
    transition_ids = {
        str(row["id"])
        for row in connection.execute(
            """
            SELECT id
            FROM messages
            WHERE save_id = ?
              AND role = 'narrator'
              AND (
                  safety_transition = ?
                  OR (safety_transition = '' AND body = ?)
                  OR safety_transition = ?
                  OR (safety_transition = '' AND body = ?)
              )
            """,
            (
                save_id,
                FADE_TO_BLACK_TRANSITION_KIND,
                FADE_TO_BLACK_TRANSITION,
                CONTENT_FILTER_TRANSITION_KIND,
                CONTENT_FILTER_TRANSITION,
            ),
        ).fetchall()
    }
    if not transition_ids:
        return
    placeholders = ", ".join("?" for _ in transition_ids)
    parameters = (save_id, *sorted(transition_ids))
    for table_name in (
        "active_threads",
        "character_contact_states",
        "character_knowledge_edges",
        "character_text_proactive_triggers",
        "characters",
        "context_observations",
        "context_sources",
        "context_update_audit",
        "context_update_suggestions",
        "entity_links",
        "locations",
        "media_assets",
        "memories",
        "message_action_choices",
        "message_revisions",
        "message_scene_presence",
        "message_visibility",
        "save_scenario_updates",
        "narrator_phone_activity_cursors",
        "dating_route_states",
        "save_loss_condition_changes",
        "save_loss_conditions",
        "save_loss_outcomes",
        "state_changes",
        "world_state",
    ):
        columns = {
            str(row["name"])
            for row in connection.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
        }
        for column in (
            "source_message_id",
            "first_seen_message_id",
            "last_updated_message_id",
            "message_id",
            "narrator_message_id",
            "triggering_message_id",
            "epilogue_message_id",
        ):
            if column in columns:
                connection.execute(
                    f"DELETE FROM {table_name} "
                    f"WHERE save_id = ? AND {column} IN ({placeholders})",
                    parameters,
                )
        for column in (
            "source_message_ids_json",
            "metadata_json",
            "evidence_json",
            "proposed_value_json",
            "before_json",
            "after_json",
        ):
            if column in columns:
                for message_id in transition_ids:
                    connection.execute(
                        f"DELETE FROM {table_name} "
                        f"WHERE save_id = ? AND {column} LIKE ?",
                        (save_id, f"%{message_id}%"),
                    )
    connection.execute(
        "DELETE FROM context_sources "
        "WHERE save_id = ? AND source_type IN ('message', 'messages') "
        "AND ("
        + " OR ".join(
            "(source_id = ? OR source_id LIKE ? OR source_id LIKE ? "
            "OR source_id LIKE ?)"
            for _ in transition_ids
        )
        + ")",
        (
            save_id,
            *(
                value
                for message_id in sorted(transition_ids)
                for value in (
                    message_id,
                    f"{message_id},%",
                    f"%,{message_id}",
                    f"%,{message_id},%",
                )
            ),
        ),
    )
    connection.execute(
        f"""
        UPDATE scene_snapshots
        SET current_location_id = NULL,
            situation = '',
            objective = '',
            weather = '',
            mood = '',
            nearby_objects_json = '[]',
            hazards_json = '[]',
            present_character_ids_json = '[]',
            source_message_id = NULL,
            in_world_time = '',
            time_of_day = '',
            day_of_week = '',
            world_day_index = NULL,
            world_time_day_index = NULL,
            world_time_day_label = '',
            world_time_phase = '',
            world_time_clock_minutes = NULL,
            world_time_period_label = '',
            world_time_source_message_id = NULL,
            world_time_confidence = NULL,
            first_seen_message_id = NULL,
            last_updated_message_id = NULL
        WHERE save_id = ?
          AND (
              source_message_id IN ({placeholders})
              OR world_time_source_message_id IN ({placeholders})
              OR first_seen_message_id IN ({placeholders})
              OR last_updated_message_id IN ({placeholders})
          )
        """,
        (
            save_id,
            *sorted(transition_ids),
            *sorted(transition_ids),
            *sorted(transition_ids),
            *sorted(transition_ids),
        ),
    )
    message_order = {
        str(row["id"]): index
        for index, row in enumerate(
            connection.execute(
                "SELECT id FROM messages WHERE save_id = ? ORDER BY rowid",
                (save_id,),
            ).fetchall()
        )
    }
    for row in connection.execute(
        "SELECT id, covers_message_start_id, covers_message_end_id "
        "FROM summaries WHERE save_id = ?",
        (save_id,),
    ).fetchall():
        start = message_order.get(str(row["covers_message_start_id"]))
        end = message_order.get(str(row["covers_message_end_id"]))
        if start is None or end is None:
            continue
        lower, upper = sorted((start, end))
        if any(
            lower <= message_order.get(message_id, -1) <= upper
            for message_id in transition_ids
        ):
            connection.execute("DELETE FROM summaries WHERE id = ?", (row["id"],))
def _placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _location_rows_parent_first(
    rows: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    return _dependency_rows_first(rows, dependency_column="parent_location_id")


def _media_rows_parent_first(
    rows: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    return _dependency_rows_first(rows, dependency_column="source_media_asset_id")


def _dependency_rows_first(
    rows: Iterable[Mapping[str, object]],
    *,
    dependency_column: str,
) -> tuple[dict[str, object], ...]:
    pending = [dict(row) for row in rows]
    ordered: list[dict[str, object]] = []
    inserted: set[str] = set()
    pending_ids = {
        str(row["id"]) for row in pending if isinstance(row.get("id"), str)
    }
    while pending:
        progressed = False
        for row in tuple(pending):
            dependency = row.get(dependency_column)
            if (
                dependency is None
                or dependency not in pending_ids
                or dependency in inserted
            ):
                ordered.append(row)
                row_id = row.get("id")
                if isinstance(row_id, str):
                    inserted.add(row_id)
                pending.remove(row)
                progressed = True
        if not progressed:
            ordered.extend(pending)
            break
    return tuple(ordered)


def _copy_snapshot_media_row(
    *,
    media_dir: Path,
    row: Mapping[str, object],
    remapped: dict[str, object],
    fork_save_id: str,
    copied_paths: list[Path],
) -> dict[str, object]:
    asset_id = str(remapped["id"])
    path = row.get("path")
    if isinstance(path, str) and path:
        remapped["path"] = _copy_media_file(
            media_dir=media_dir,
            source_relative_path=path,
            fork_save_id=fork_save_id,
            asset_id=asset_id,
            copied_paths=copied_paths,
        )
    thumbnail_path = row.get("thumbnail_path")
    if isinstance(thumbnail_path, str) and thumbnail_path:
        remapped["thumbnail_path"] = _copy_media_file(
            media_dir=media_dir,
            source_relative_path=thumbnail_path,
            fork_save_id=fork_save_id,
            asset_id=asset_id,
            copied_paths=copied_paths,
            thumbnail=True,
        )
    return remapped


def _copy_media_file(
    *,
    media_dir: Path,
    source_relative_path: str,
    fork_save_id: str,
    asset_id: str,
    copied_paths: list[Path],
    thumbnail: bool = False,
) -> str:
    media_root = media_dir.resolve()
    source_path = (media_dir / source_relative_path).resolve()
    if not source_path.is_relative_to(media_root):
        raise ValueError(
            f"Media path escapes media directory: {source_relative_path}"
        )
    suffix = "".join(Path(source_relative_path).suffixes) or ".bin"
    name = f"{'thumb-' if thumbnail else ''}{asset_id}{suffix}"
    destination_relative = Path("forks") / fork_save_id / name
    destination_path = (media_dir / destination_relative).resolve()
    if not destination_path.is_relative_to(media_root):
        raise ValueError(
            f"Fork media destination escapes media directory: {destination_relative}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    copied_paths.append(destination_path)
    return destination_relative.as_posix()


def _delete_copied_paths(paths: list[Path], *, media_dir: Path) -> None:
    media_root = media_dir.resolve()
    for path in reversed(paths):
        try:
            if path.resolve().is_relative_to(media_root) and path.is_file():
                path.unlink()
        except OSError:
            continue


def _add_snapshot_object_export(
    objects_by_hash: dict[str, dict[str, object]],
    *,
    kind: str,
    value: object,
    created_at: str | None,
) -> str:
    payload = _canonical_json_bytes(value)
    object_hash = _snapshot_object_hash(kind=kind, payload=payload)
    objects_by_hash.setdefault(
        object_hash,
        {
            "object_hash": object_hash,
            "kind": kind,
            "encoding": SNAPSHOT_ENCODING,
            "payload_base64": base64.b64encode(
                zlib.compress(payload),
            ).decode("ascii"),
            "uncompressed_size": len(payload),
            "created_at": created_at,
        },
    )
    return object_hash


def _decode_exported_object(row: Mapping[str, object]) -> object:
    return _decode_exported_snapshot_object(row)[1]


def _decode_exported_snapshot_object(row: Mapping[str, object]) -> tuple[str, object]:
    object_hash = _text(row, "object_hash")
    kind = _text(row, "kind")
    encoding = _text(row, "encoding")
    if encoding != SNAPSHOT_ENCODING:
        raise ValueError(f"Unsupported snapshot object encoding: {encoding}")
    try:
        payload = zlib.decompress(base64.b64decode(_text(row, "payload_base64")))
        value = json.loads(payload.decode("utf-8"))
    except (
        binascii.Error,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zlib.error,
    ) as exc:
        raise ValueError(f"Invalid snapshot object payload: {object_hash}") from exc
    if len(payload) != _int(row, "uncompressed_size"):
        raise ValueError(f"Snapshot object size mismatch: {object_hash}")
    actual_hash = _snapshot_object_hash(kind=kind, payload=payload)
    if actual_hash != object_hash:
        raise ValueError(f"Snapshot object hash mismatch: {object_hash}")
    return kind, value


def _validate_exported_snapshot_rows(
    snapshot_rows: Iterable[Mapping[str, object]],
    object_rows: Iterable[Mapping[str, object]],
) -> None:
    snapshots = [dict(row) for row in snapshot_rows]
    seen_snapshot_ids: set[str] = set()
    for row in snapshots:
        snapshot_id = _text(row, "id")
        if snapshot_id in seen_snapshot_ids:
            raise ValueError(f"Duplicate snapshot id in bundle: {snapshot_id}")
        seen_snapshot_ids.add(snapshot_id)
        _text(row, "root_manifest_hash")

    objects_by_hash: dict[str, tuple[str, object]] = {}
    for object_row in object_rows:
        object_hash = _text(object_row, "object_hash")
        if object_hash in objects_by_hash:
            raise ValueError(f"Duplicate snapshot object in bundle: {object_hash}")
        objects_by_hash[object_hash] = _decode_exported_snapshot_object(object_row)

    for row in snapshots:
        root_hash = _text(row, "root_manifest_hash")
        manifest = _required_exported_snapshot_object(
            objects_by_hash,
            root_hash,
            expected_kind="snapshot_manifest",
        )
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("format") != SNAPSHOT_FORMAT
        ):
            raise ValueError(f"Invalid snapshot manifest object: {root_hash}")
        raw_tables = manifest.get("tables")
        if not isinstance(raw_tables, Mapping):
            raise ValueError(f"Snapshot manifest is missing table entries: {root_hash}")
        for table_name, entries in raw_tables.items():
            if table_name not in _TABLES_BY_NAME:
                raise ValueError(f"Unknown snapshot table in manifest: {table_name}")
            if not isinstance(entries, list):
                raise ValueError(f"Snapshot table entries must be a list: {table_name}")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError(f"Snapshot table entry is invalid: {table_name}")
                object_hash = _text(entry, "object_hash")
                value = _required_exported_snapshot_object(
                    objects_by_hash,
                    object_hash,
                    expected_kind=f"row:{table_name}",
                )
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"Snapshot row object is not a row: {object_hash}"
                    )


def _required_exported_snapshot_object(
    objects_by_hash: Mapping[str, tuple[str, object]],
    object_hash: str,
    *,
    expected_kind: str,
) -> object:
    try:
        kind, value = objects_by_hash[object_hash]
    except KeyError as exc:
        raise ValueError(f"Missing snapshot object: {object_hash}") from exc
    if kind != expected_kind:
        raise ValueError(
            f"Snapshot object kind mismatch for {object_hash}: "
            f"expected {expected_kind}, got {kind}"
        )
    return value


def _media_asset_record_from_snapshot_row(
    row: Mapping[str, object],
) -> MediaAssetRecord:
    return MediaAssetRecord(
        id=_text(row, "id"),
        save_id=_text(row, "save_id"),
        source_message_id=_optional_text(row, "source_message_id"),
        type=_text(row, "type"),
        path=_text(row, "path"),
        thumbnail_path=_optional_text(row, "thumbnail_path"),
        prompt=_optional_text(row, "prompt") or "",
        provider=_optional_text(row, "provider") or "",
        model=_optional_text(row, "model") or "",
        status=_optional_text(row, "status") or "",
        mime_type=_optional_text(row, "mime_type") or "image/png",
        metadata_json=_optional_text(row, "metadata_json") or "{}",
        source_media_asset_id=_optional_text(row, "source_media_asset_id"),
        created_at=_optional_text(row, "created_at"),
        archived_at=_optional_text(row, "archived_at"),
    )


def _snapshot_object_hash(*, kind: str, payload: bytes) -> str:
    return sha256(kind.encode("utf-8") + b"\0" + payload).hexdigest()


def _merge_rows_by_table(
    items: Iterable[Mapping[str, Iterable[Mapping[str, object]]]],
) -> dict[str, tuple[dict[str, object], ...]]:
    merged: dict[str, list[dict[str, object]]] = {}
    seen: set[tuple[str, str]] = set()
    for rows_by_table in items:
        for table_name, rows in rows_by_table.items():
            table_rows = merged.setdefault(table_name, [])
            for row in rows:
                row_id = row.get("id")
                key = (table_name, str(row_id))
                if isinstance(row_id, str) and key in seen:
                    continue
                if isinstance(row_id, str):
                    seen.add(key)
                table_rows.append(dict(row))
    return {table_name: tuple(rows) for table_name, rows in merged.items()}


def _text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Snapshot field is required: {key}")
    return value


def _optional_text(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    return value if isinstance(value, str) and value else None


def _int(row: Mapping[str, object], key: str, *, default: int | None = None) -> int:
    value = row.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    if default is not None:
        return default
    raise ValueError(f"Snapshot integer field is required: {key}")


def _remap_optional(value: str | None, mapping: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    return mapping.get(value, value)
