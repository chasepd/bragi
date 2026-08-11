"""Content-addressed save state snapshots for exact turn rollback and fork."""

from __future__ import annotations

import base64
import binascii
import json
import re
import shutil
import sqlite3
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import cast
from uuid import uuid4

from bragi.app_logging import log_event
from bragi.json_safety import JsonSafetyError, validate_json_structure
from bragi.persistence.models import MediaAssetRecord, MessageRecord, SaveRecord
from bragi.persistence.repositories import (
    PersistenceRepositories,
    _epistemic_claim_fingerprint,
    canonical_claim_fingerprint,
    validate_context_source_index_budget,
)
from bragi.persistence.snapshot_contract import (
    SNAPSHOT_TABLES,
    SNAPSHOT_TABLES_BY_NAME,
    SnapshotTable,
)
from bragi.safety import (
    CONTENT_FILTER_TRANSITION,
    CONTENT_FILTER_TRANSITION_KIND,
    FADE_TO_BLACK_TRANSITION,
    FADE_TO_BLACK_TRANSITION_KIND,
    normalize_message_safety,
)
from bragi.scene_facts import scene_fact_conflict_key
from bragi.services.character_text_world_update_service import (
    character_text_source_ref,
    parse_character_text_source_ref,
)
from bragi.services.knowledge_boundary import normalized_knowledge_target_type
from bragi.services.scenario_content_rating import (
    metadata_with_scenario_content_ratings,
)
from bragi.services.scenario_service import strip_deprecated_scenario_character_sections
from bragi.services.sexual_content_safety import (
    is_fade_to_black_message,
)

SNAPSHOT_FORMAT = "bragi-turn-snapshot-v1"
SNAPSHOT_FORMAT_V2 = "bragi-turn-snapshot-v2"
_SnapshotTable = SnapshotTable
_SNAPSHOT_TABLES = SNAPSHOT_TABLES
_TABLES_BY_NAME = SNAPSHOT_TABLES_BY_NAME
_SNAPSHOT_TABLE_NAMES = tuple(table.name for table in _SNAPSHOT_TABLES)
_MAX_SNAPSHOT_PROVENANCE_GROUPS = 64
_MAX_SNAPSHOT_PROVENANCE_GROUP_MEMBERS = 64
_MAX_SNAPSHOT_OBJECT_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
_MAX_SNAPSHOT_TOTAL_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_MAX_IMPORTED_SNAPSHOT_COUNT = 4_096
_MAX_SNAPSHOT_MANIFEST_ENTRIES = 10_000
_MAX_SNAPSHOT_UNIQUE_ROW_OBJECTS = 65_536
_MAX_SNAPSHOT_IMPORT_REFERENCED_BYTES = 128 * 1024 * 1024
_MAX_SNAPSHOT_OBJECT_JSON_NODES = 100_000
_MAX_SNAPSHOT_TOTAL_JSON_NODES = 2_000_000
_MAX_SNAPSHOT_OBJECT_JSON_DEPTH = 128
_MAX_SNAPSHOT_TREE_MUTATION_DEPTH = 256
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
    table_roots: dict[str, str | None]
    object_count: int


@dataclass
class _TreeMutationGuard:
    seen: set[str]
    byte_budget: list[int]
    json_node_budget: list[int]
    cache: dict[str, tuple[str, object]]

_RESTORE_DELETE_ORDER = (
    "character_text_proactive_triggers",
    "turn_outcomes",
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
    "context_sources",
    "scene_fact_sources",
    "scene_facts",
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
    "scene_facts",
    "scene_fact_sources",
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
    "turn_outcomes",
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
    "memories": {"epistemic_actor_id": "characters"},
    "context_observations": {"epistemic_actor_id": "characters"},
    "scene_snapshots": {"current_location_id": "locations"},
    "scene_facts": {"scene_snapshot_id": "scene_snapshots"},
    "scene_fact_sources": {"scene_fact_id": "scene_facts"},
    "context_sources": {"scene_snapshot_id": "scene_snapshots"},
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
    "memories": frozenset(
        {
            "tags_json",
            "source_message_ids_json",
            "source_observation_ids_json",
        }
    ),
    "summaries": frozenset(
        {"source_message_ids_json", "source_summary_ids_json"}
    ),
    "save_scenario_updates": frozenset({"content_json", "source_message_ids_json"}),
    "save_loss_condition_changes": frozenset({"before_json", "after_json"}),
    "save_loss_outcomes": frozenset({"evidence_json"}),
    "media_assets": frozenset({"metadata_json"}),
    "character_knowledge_edges": frozenset({"source_message_ids_json"}),
    "dating_route_states": frozenset(
        {"known_boundaries_json", "unresolved_questions_json"}
    ),
    "character_text_message_attachments": frozenset({"metadata_json"}),
    "turn_outcomes": frozenset({"payload_json"}),
}

_ENTITY_TABLES = {
    "location": "locations",
    "character": "characters",
    "thread": "active_threads",
    "active_thread": "active_threads",
    "scene_snapshot": "scene_snapshots",
    "scene_fact": "scene_facts",
    "loss_condition": "save_loss_conditions",
    "media_asset": "media_assets",
    "save": "saves",
    "state": "world_state",
    "world_state": "world_state",
    "memory": "memories",
    "observation": "context_observations",
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
        self.repositories.begin_immediate_transaction()
        try:
            clean = self._clean_materialized_snapshot(save_id)
            if clean is not None:
                self.repositories.commit_transaction()
                return clean
            message = self._latest_active_message(save_id)
            message_id = message.id if message is not None else None
            prepared = self._prepare_snapshot(
                save_id=save_id,
                message_id=message_id,
            )
            snapshot = self._insert_prepared_snapshot(
                save_id=save_id,
                message_id=message_id,
                reason=reason,
                prepared=prepared,
            )
            self.repositories.commit_transaction()
            return snapshot
        except BaseException:
            self.repositories.rollback_transaction()
            raise

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

    def snapshot_present_character_ids(self, *, snapshot_id: str) -> tuple[str, ...]:
        snapshot = self._get_snapshot(snapshot_id)
        manifest = self._snapshot_manifest(snapshot)
        rows_by_table = _sanitize_snapshot_rows_for_safety(
            self._rows_from_manifest(manifest)
        )
        scene_rows = rows_by_table.get("scene_snapshots", ())
        if not scene_rows:
            return ()
        raw_ids = scene_rows[-1].get("present_character_ids_json")
        if not isinstance(raw_ids, str) or not raw_ids.strip():
            return ()
        try:
            parsed = json.loads(raw_ids)
        except json.JSONDecodeError:
            return ()
        if not isinstance(parsed, list):
            return ()
        return tuple(item for item in parsed if isinstance(item, str) and item)

    def snapshot_message_ids(self, *, snapshot_id: str) -> tuple[str, ...]:
        snapshot = self._get_snapshot(snapshot_id)
        manifest = self._snapshot_manifest(snapshot)
        rows_by_table = _sanitize_snapshot_rows_for_safety(
            self._rows_from_manifest(manifest)
        )
        return _snapshot_message_ids_from_rows(rows_by_table)

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
        rows_by_table = _normalize_legacy_snapshot_memories(rows_by_table)
        rows_by_table = _normalize_legacy_snapshot_scene_facts(rows_by_table)
        raw_active_message_ids = manifest.get("active_message_ids", [])
        if not isinstance(raw_active_message_ids, list):
            raw_active_message_ids = []
        active_message_ids = (
            tuple(str(message_id) for message_id in raw_active_message_ids)
            if manifest.get("format") == SNAPSHOT_FORMAT
            else _snapshot_message_ids_from_rows(rows_by_table)
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
            self._normalize_memory_fingerprints(save_id)
            self.repositories.rebuild_context_source_search_terms(save_id)
            self.repositories.consolidate_active_memory_duplicates(
                save_id=save_id
            )
            self.repositories.ensure_context_observation_curation_states(save_id)
            self.repositories.require_continuity_index_full_rebuild(save_id)
            _remove_snapshot_safety_transition_records(
                self.repositories.connection,
                save_id=save_id,
            )
            self.repositories.rebuild_summary_pressure_state(save_id)
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
        trailing_messages: Iterable[MessageRecord] = (),
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
        rows_by_table = _normalize_legacy_snapshot_memories(rows_by_table)
        rows_by_table = _normalize_legacy_snapshot_scene_facts(rows_by_table)
        trailing = tuple(trailing_messages)
        source_snapshot_message_ids = frozenset(
            _snapshot_message_ids_from_rows(rows_by_table)
        )
        for message in trailing:
            if message.save_id != source_save_id or message.deleted_at is not None:
                raise ValueError(
                    "Trailing fork messages must be active source messages"
                )
            if message.id in source_snapshot_message_ids:
                raise ValueError(
                    "Trailing fork messages must follow the source snapshot"
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
                interaction_mode=source_save.interaction_mode,
            )
            for row in rows_by_table.get("messages", ()):
                self._insert_row(
                    "messages",
                    _sanitize_snapshot_message_row(
                        remapper.remap_row("messages", row)
                    ),
                )
            forked_trailing_messages = tuple(
                self.repositories.append_message(
                    save_id=fork_save.id,
                    role=message.role,
                    speaker_name=message.speaker_name,
                    body=message.body,
                    provider=message.provider,
                    model=message.model,
                    token_estimate=message.token_estimate,
                    created_at=message.created_at,
                    updated_at=message.updated_at,
                    safety_transition=message.safety_transition,
                    content_rating=message.content_rating,
                )
                for message in trailing
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
            self._normalize_memory_fingerprints(fork_save.id)
            self.repositories.copy_context_source_legacy_budget_limit(
                source_save_id=source_save_id,
                target_save_id=fork_save.id,
            )
            self.repositories.rebuild_context_source_search_terms(fork_save.id)
            self.repositories.consolidate_active_memory_duplicates(
                save_id=fork_save.id
            )
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
            if forked_trailing_messages:
                self.capture_message_snapshot(
                    save_id=fork_save.id,
                    message_id=forked_trailing_messages[-1].id,
                    reason="fork_trailing_message",
                )
            self.repositories.rebuild_summary_pressure_state(fork_save.id)
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            _delete_copied_paths(copied_paths, media_dir=media_dir)
            raise

        media_count = len(rows_by_table.get("media_assets", ()))
        message_count = len(rows_by_table.get("messages", ())) + len(trailing)
        log_event(
            "save.snapshot_forked",
            source_save_id=source_save_id,
            fork_save_id=fork_save.id,
            snapshot_id=snapshot_id,
            message_count=message_count,
            media_count=media_count,
        )
        return SnapshotForkResult(
            save=fork_save,
            message_count=message_count,
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
                    "format": SNAPSHOT_FORMAT,
                    "save_id": snapshot.save_id,
                    "message_id": snapshot.message_id,
                    "active_message_ids": _snapshot_message_ids_from_rows(
                        rows_by_table
                    ),
                    "context_revision": snapshot.context_revision,
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
        media_path_map: Mapping[tuple[str, str], str | None] | None = None,
        require_verified_media_paths: bool = False,
    ) -> int:
        snapshots = [dict(row) for row in snapshot_rows]
        objects = [dict(row) for row in object_rows]
        decoded_objects_by_hash = _validate_exported_snapshot_rows(
            snapshot_rows=snapshots,
            object_rows=objects,
        )
        if not snapshots:
            return 0
        manifests_by_hash = {
            root_hash: _required_exported_snapshot_object(
                decoded_objects_by_hash,
                root_hash,
                expected_kind="snapshot_manifest",
            )
            for root_hash in {
                _text(row, "root_manifest_hash") for row in snapshots
            }
        }
        table_signature_by_manifest = {
            root_hash: _snapshot_manifest_table_signature(manifest)
            for root_hash, manifest in manifests_by_hash.items()
            if isinstance(manifest, Mapping)
        }
        representative_manifest_by_signature = {
            signature: root_hash
            for root_hash, signature in table_signature_by_manifest.items()
        }
        source_rows_by_signature = {
            signature: _sanitize_snapshot_rows_for_safety(
                self._rows_from_exported_manifest(
                    decoded_objects_by_hash,
                    root_manifest_hash,
                ),
                quarantine_content_ratings=True,
            )
            for signature, root_manifest_hash
            in representative_manifest_by_signature.items()
        }
        remapper = _SnapshotRemapper(
            source_save_id=source_save_id,
            target_save_id=target_save_id,
            rows_by_table=_merge_rows_by_table(source_rows_by_signature.values()),
            id_maps=id_maps,
            message_id_map=message_id_map,
            media_path_map=media_path_map,
            require_media_path_map=require_verified_media_paths,
        )
        snapshot_id_map = {
            _text(row, "id"): str(uuid4()) for row in snapshots
        }
        manifest_hash_map: dict[str, str] = {}
        remapped_tables_by_signature: dict[
            str,
            dict[str, list[dict[str, str]]],
        ] = {}
        for row in snapshots:
            old_manifest_hash = _text(row, "root_manifest_hash")
            if old_manifest_hash in manifest_hash_map:
                continue
            signature = table_signature_by_manifest[old_manifest_hash]
            rows_by_table = source_rows_by_signature[signature]
            manifest_object = manifests_by_hash[old_manifest_hash]
            if not isinstance(manifest_object, dict):
                raise ValueError("Snapshot manifest object is not a JSON object")
            manifest = cast(dict[str, object], manifest_object)
            remapped_tables = remapped_tables_by_signature.get(signature)
            if remapped_tables is None:
                remapped_tables = {}
                for table_name, rows in rows_by_table.items():
                    remapped_entries: list[dict[str, str]] = []
                    remapped_rows: list[dict[str, object]] = []
                    for row_data in rows:
                        remapped_row = remapper.remap_row(table_name, row_data)
                        if table_name == "messages":
                            remapped_row = _sanitize_snapshot_message_row(
                                remapped_row
                            )
                        remapped_rows.append(remapped_row)
                    for remapped_row in _coalesce_remapped_snapshot_rows(
                        table_name,
                        remapped_rows,
                    ):
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
                remapped_tables_by_signature[signature] = remapped_tables
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

    @staticmethod
    def media_asset_rows_from_snapshot_objects(
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

    @staticmethod
    def context_source_budget_from_snapshot_objects(
        *,
        snapshot_rows: Iterable[Mapping[str, object]],
        object_rows: Iterable[Mapping[str, object]],
        max_normalized_text_bytes: int,
        max_normalized_record_bytes: int,
    ) -> tuple[int, int]:
        snapshots = [dict(row) for row in snapshot_rows]
        objects = [dict(row) for row in object_rows]
        decoded_objects = _validate_exported_snapshot_rows(
            snapshot_rows=snapshots,
            object_rows=objects,
        )
        max_total_bytes = 0
        max_record_bytes = 0
        seen_signatures: set[str] = set()
        for snapshot in snapshots:
            manifest_hash = _text(snapshot, "root_manifest_hash")
            manifest = _required_exported_snapshot_object(
                decoded_objects,
                manifest_hash,
                expected_kind="snapshot_manifest",
            )
            if not isinstance(manifest, Mapping):
                raise ValueError("Snapshot manifest object is not valid")
            signature = _snapshot_manifest_table_signature(manifest)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            context_rows: list[Mapping[str, object]] = []
            for entry in _manifest_tables(manifest).get("context_sources", ()):
                context_row = _required_exported_snapshot_object(
                    decoded_objects,
                    _text(entry, "object_hash"),
                    expected_kind="row:context_sources",
                )
                if not isinstance(context_row, Mapping):
                    raise ValueError("Snapshot context source is not valid")
                context_rows.append(context_row)
            total_bytes, record_bytes = validate_context_source_index_budget(
                context_rows,
                max_normalized_text_bytes=max_normalized_text_bytes,
                max_normalized_record_bytes=max_normalized_record_bytes,
            )
            max_total_bytes = max(max_total_bytes, total_bytes)
            max_record_bytes = max(max_record_bytes, record_bytes)
        return max_total_bytes, max_record_bytes

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
        self.repositories.begin_immediate_transaction()
        try:
            if self.repositories.get_save(save_id) is None:
                raise ValueError(f"Unknown save id: {save_id}")
            if message_id is not None:
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
            clean = self._clean_materialized_snapshot(save_id, message_id=message_id)
            if clean is not None:
                self.repositories.commit_transaction()
                return clean
            prepared = self._prepare_snapshot(save_id=save_id, message_id=message_id)
            snapshot = self._insert_prepared_snapshot(
                save_id=save_id,
                message_id=message_id,
                reason=reason,
                prepared=prepared,
            )
            self.repositories.commit_transaction()
            return snapshot
        except BaseException:
            self.repositories.rollback_transaction()
            raise

    def _prepare_snapshot(
        self,
        *,
        save_id: str,
        message_id: str | None,
    ) -> _PreparedSnapshot:
        self._queue_due_snapshot_rechecks(save_id)
        _remove_snapshot_safety_transition_records(
            self.repositories.connection,
            save_id=save_id,
        )
        incremental = self._prepare_incremental_snapshot(
            save_id=save_id,
            message_id=message_id,
        )
        if incremental is not None:
            return incremental
        recheck_deadlines = self._snapshot_recheck_deadlines(save_id)
        raw_rows_by_table = self._active_rows_by_table(save_id)
        rows_by_table = _sanitize_snapshot_rows_for_safety(raw_rows_by_table)
        self.repositories.connection.execute(
            "DELETE FROM save_snapshot_row_state WHERE save_id = ?",
            (save_id,),
        )
        self.repositories.connection.execute(
            "DELETE FROM save_snapshot_row_references WHERE save_id = ?",
            (save_id,),
        )
        table_roots: dict[str, str | None] = {}
        object_count = 0
        for table_name in _SNAPSHOT_TABLE_NAMES:
            root_hash: str | None = None
            table = _TABLES_BY_NAME[table_name]
            raw_rows = {
                _snapshot_row_key(table, row): row
                for row in raw_rows_by_table.get(table_name, ())
            }
            included_keys: set[str] = set()
            for ordinal, row in enumerate(rows_by_table.get(table_name, ())):
                row_key = _snapshot_row_key(table, row)
                included_keys.add(row_key)
                object_hash = self._store_object(
                    kind=f"row:{table_name}",
                    value=row,
                )
                order_key = _snapshot_tree_order_key(
                    table,
                    row=row,
                    row_key=row_key,
                    ordinal=ordinal,
                )
                root_hash = self._tree_insert(
                    table_name=table_name,
                    root_hash=root_hash,
                    order_key=order_key,
                    row_key=row_key,
                    row_hash=object_hash,
                )
                self.repositories.connection.execute(
                    """
                    INSERT INTO save_snapshot_row_state(
                        save_id, table_name, row_key, object_hash,
                        order_key, ordinal, included, recheck_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        save_id,
                        table_name,
                        row_key,
                        object_hash,
                        order_key,
                        ordinal,
                        recheck_deadlines.get((table_name, row_key)),
                    ),
                )
                self._replace_snapshot_row_references(
                    save_id=save_id,
                    table_name=table_name,
                    row_key=row_key,
                    row=raw_rows.get(row_key, row),
                )
                object_count += 1
            next_ordinal = len(included_keys)
            for row_key, raw_row in raw_rows.items():
                if row_key in included_keys:
                    continue
                raw_hash = self._store_object(
                    kind=f"row:{table_name}",
                    value=raw_row,
                )
                self.repositories.connection.execute(
                    """
                    INSERT INTO save_snapshot_row_state(
                        save_id, table_name, row_key, object_hash,
                        order_key, ordinal, included, recheck_at
                    ) VALUES (?, ?, ?, ?, NULL, ?, 0, NULL)
                    """,
                    (save_id, table_name, row_key, raw_hash, next_ordinal),
                )
                self._replace_snapshot_row_references(
                    save_id=save_id,
                    table_name=table_name,
                    row_key=row_key,
                    row=raw_row,
                )
                next_ordinal += 1
            table_roots[table_name] = root_hash
            self.repositories.connection.execute(
                """
                INSERT INTO save_snapshot_table_state(
                    save_id, table_name, current_generation,
                    captured_generation, root_hash, next_ordinal, needs_rebuild
                )
                VALUES (?, ?, 0, 0, ?, ?, 0)
                ON CONFLICT(save_id, table_name) DO UPDATE SET
                    root_hash = excluded.root_hash,
                    next_ordinal = excluded.next_ordinal,
                    needs_rebuild = 0
                """,
                (
                    save_id,
                    table_name,
                    root_hash,
                    next_ordinal,
                ),
            )
        activity_max_ordinal = max(
            (
                _snapshot_row_int(row, "ordinal")
                for row in rows_by_table.get("character_text_activity_events", ())
            ),
            default=0,
        )
        self.repositories.connection.execute(
            "DELETE FROM save_snapshot_included_activity_events WHERE save_id = ?",
            (save_id,),
        )
        self.repositories.connection.executemany(
            """
            INSERT INTO save_snapshot_included_activity_events(
                save_id, event_id, ordinal
            )
            VALUES (?, ?, ?)
            """,
            (
                (save_id, str(row["id"]), _snapshot_row_int(row, "ordinal"))
                for row in rows_by_table.get(
                    "character_text_activity_events", ()
                )
            ),
        )
        self.repositories.connection.execute(
            """
            INSERT INTO save_snapshot_activity_state(save_id, max_ordinal)
            VALUES (?, ?)
            ON CONFLICT(save_id) DO UPDATE SET
                max_ordinal = excluded.max_ordinal
            """,
            (save_id, activity_max_ordinal),
        )
        context_revision = self._context_revision(save_id)
        manifest = {
            "format": SNAPSHOT_FORMAT_V2,
            "save_id": save_id,
            "message_id": message_id,
            "context_revision": context_revision,
            "table_roots": table_roots,
        }
        root_manifest_hash = self._store_object(
            kind="snapshot_manifest",
            value=manifest,
        )
        return _PreparedSnapshot(
            root_manifest_hash=root_manifest_hash,
            context_revision=context_revision,
            table_roots=table_roots,
            object_count=object_count,
        )

    def _prepare_incremental_snapshot(
        self,
        *,
        save_id: str,
        message_id: str | None,
    ) -> _PreparedSnapshot | None:
        base_row = self.repositories.connection.execute(
            """
            SELECT state.base_snapshot_id
            FROM save_snapshot_state state
            WHERE state.save_id = ? AND state.base_snapshot_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM save_snapshot_table_state tables
                  WHERE tables.save_id = state.save_id
                    AND tables.needs_rebuild != 0
              )
            """,
            (save_id,),
        ).fetchone()
        if base_row is None:
            return None
        base_snapshot = self._get_snapshot(str(base_row["base_snapshot_id"]))
        if base_snapshot.save_id != save_id:
            raise ValueError("Incremental snapshot base belongs to another save")
        base_manifest = self._snapshot_manifest(base_snapshot)
        if base_manifest.get("format") != SNAPSHOT_FORMAT_V2:
            return None
        raw_roots = base_manifest.get("table_roots")
        if not isinstance(raw_roots, Mapping):
            return None
        dirty_rows = self.repositories.connection.execute(
            """
            SELECT dirty.table_name, dirty.row_key, dirty.generation
            FROM save_snapshot_dirty_rows dirty
            WHERE dirty.save_id = ?
            ORDER BY dirty.generation, dirty.table_name, dirty.row_key
            """,
            (save_id,),
        ).fetchall()
        if not dirty_rows:
            return None
        aggregate_dependents_queued: set[tuple[str, str]] = set()
        lifecycle_dependents = self._queue_snapshot_lifecycle_dependents(
            save_id,
            dirty_rows,
        )
        self._expand_snapshot_reverse_dependency_closure(
            save_id,
            lifecycle_dependents,
        )
        while True:
            closure_rows = self.repositories.connection.execute(
                """
                SELECT table_name, row_key, generation
                FROM save_snapshot_dirty_rows
                WHERE save_id = ?
                ORDER BY generation, table_name, row_key
                """,
                (save_id,),
            ).fetchall()
            aggregate_keys = self._queue_snapshot_aggregate_dependents(
                save_id,
                closure_rows,
                already_queued=aggregate_dependents_queued,
            )
            if not aggregate_keys:
                break
            self._expand_snapshot_reverse_dependency_closure(
                save_id,
                aggregate_keys,
            )
        dirty_rows = self.repositories.connection.execute(
            """
            SELECT table_name, row_key, generation
            FROM save_snapshot_dirty_rows
            WHERE save_id = ?
            ORDER BY generation, table_name, row_key
            """,
            (save_id,),
        ).fetchall()
        table_roots = {
            table_name: (
                str(raw_roots[table_name])
                if isinstance(raw_roots.get(table_name), str)
                else None
            )
            for table_name in _SNAPSHOT_TABLE_NAMES
        }
        pending: dict[
            tuple[str, str],
            tuple[
                _SnapshotTable,
                str,
                dict[str, object] | None,
                dict[str, object] | None,
                sqlite3.Row | None,
                str | None,
            ],
        ] = {}
        projected_inclusion: dict[tuple[str, str], bool] = {}
        initial_dirty_keys = {
            (str(row["table_name"]), str(row["row_key"])) for row in dirty_rows
        }
        queued_work_keys = set(initial_dirty_keys)

        def seed_projected_physical(table_name: str, row_key: str) -> None:
            key = (table_name, row_key)
            if key in projected_inclusion:
                return
            table = _TABLES_BY_NAME.get(table_name)
            if table is None:
                return
            live = self.repositories.connection.execute(
                f"SELECT * FROM {table.name} "
                f"WHERE save_id = ? AND {table.primary_key} = ?",
                (save_id, row_key),
            ).fetchone()
            eligible = live is not None
            if live is not None:
                live_value = _row_dict(live)
                if table.name in {"messages", "character_text_messages"}:
                    eligible = live_value.get("deleted_at") is None
                elif (
                    table.active_only
                    and "archived_at" in self._column_names(table.name)
                ):
                    eligible = live_value.get("archived_at") is None
                if table.name == "messages" and eligible:
                    sanitized = _sanitize_snapshot_message_row(live_value)
                    eligible = not is_fade_to_black_message(
                        role=str(sanitized.get("role", "")),
                        body=str(sanitized.get("body", "")),
                        safety_transition=str(sanitized.get("safety_transition", "")),
                    )
            projected_inclusion[key] = eligible

        for table_name, row_key in initial_dirty_keys:
            seed_projected_physical(table_name, row_key)
        dirty_index = 0
        group_participant_count_cache: dict[str, int] = {}
        activity_max_ordinal_cache: list[int | None] = [None]

        def add_pending(
            table: _SnapshotTable,
            row_key: str,
            snapshot_row: dict[str, object] | None,
            reference_row: dict[str, object] | None,
            previous: sqlite3.Row | None,
            recheck_at: str | None,
        ) -> None:
            key = (table.name, row_key)
            pending[key] = (
                table,
                row_key,
                snapshot_row,
                reference_row,
                previous,
                recheck_at,
            )
            included = snapshot_row is not None
            prior_included = projected_inclusion.get(
                key,
                bool(previous["included"]) if previous is not None else included,
            )
            projected_inclusion[key] = included
            if table.name == "character_text_thread_participants":
                thread_id = (
                    reference_row.get("thread_id")
                    if reference_row is not None
                    else None
                )
                if isinstance(thread_id, str):
                    group_participant_count_cache.pop(thread_id, None)
                dirty_source = self.repositories.connection.execute(
                    """
                    SELECT table_name, row_key, generation
                    FROM save_snapshot_dirty_rows
                    WHERE save_id = ? AND table_name = ? AND row_key = ?
                    """,
                    (save_id, table.name, row_key),
                ).fetchone()
                aggregate_keys = (
                    self._queue_snapshot_aggregate_dependents(
                        save_id,
                        [dirty_source],
                        already_queued=aggregate_dependents_queued,
                    )
                    if dirty_source is not None
                    else ()
                )
                for aggregate_key in aggregate_keys:
                    added = self.repositories.connection.execute(
                        """
                        SELECT table_name, row_key, generation
                        FROM save_snapshot_dirty_rows
                        WHERE save_id = ? AND table_name = ? AND row_key = ?
                        """,
                        (save_id, *aggregate_key),
                    ).fetchone()
                    if added is not None and aggregate_key not in queued_work_keys:
                        seed_projected_physical(*aggregate_key)
                        queued_work_keys.add(aggregate_key)
                        dirty_rows.append(added)
            elif table.name == "character_text_activity_events":
                if snapshot_row is None:
                    self.repositories.connection.execute(
                        """
                        DELETE FROM save_snapshot_included_activity_events
                        WHERE save_id = ? AND event_id = ?
                        """,
                        (save_id, row_key),
                    )
                else:
                    self.repositories.connection.execute(
                        """
                        INSERT INTO save_snapshot_included_activity_events(
                            save_id, event_id, ordinal
                        )
                        VALUES (?, ?, ?)
                        ON CONFLICT(save_id, event_id) DO UPDATE SET
                            ordinal = excluded.ordinal
                        """,
                        (save_id, row_key, _snapshot_row_int(snapshot_row, "ordinal")),
                    )
                activity_max_ordinal_cache[0] = None
                cursor_keys = self._queue_activity_cursors_if_max_changed(save_id)
                for cursor_key in cursor_keys:
                    added = self.repositories.connection.execute(
                        """
                        SELECT table_name, row_key, generation
                        FROM save_snapshot_dirty_rows
                        WHERE save_id = ? AND table_name = ? AND row_key = ?
                        """,
                        (save_id, *cursor_key),
                    ).fetchone()
                    if added is not None and cursor_key not in queued_work_keys:
                        seed_projected_physical(*cursor_key)
                        queued_work_keys.add(cursor_key)
                        dirty_rows.append(added)
            if prior_included != included:
                dependent_keys = self._queue_snapshot_reference_dependents(
                    save_id=save_id,
                    target_table=table.name,
                    target_key=row_key,
                )
                for dependent_key in dependent_keys:
                    added = self.repositories.connection.execute(
                    """
                    SELECT table_name, row_key, generation
                    FROM save_snapshot_dirty_rows
                    WHERE save_id = ? AND table_name = ? AND row_key = ?
                    """,
                        (save_id, *dependent_key),
                    ).fetchone()
                    if added is None:
                        continue
                    key = (str(added["table_name"]), str(added["row_key"]))
                    if key not in queued_work_keys:
                        seed_projected_physical(*key)
                        queued_work_keys.add(key)
                        dirty_rows.append(added)

        while dirty_index < len(dirty_rows):
            dirty = dirty_rows[dirty_index]
            dirty_index += 1
            dirty_key = (str(dirty["table_name"]), str(dirty["row_key"]))
            queued_work_keys.discard(dirty_key)
            aggregate_dependents_queued.discard(dirty_key)
            aggregate_keys = self._queue_snapshot_aggregate_dependents(
                save_id,
                [dirty],
                already_queued=aggregate_dependents_queued,
            )
            for aggregate_key in aggregate_keys:
                added = self.repositories.connection.execute(
                    """
                    SELECT table_name, row_key, generation
                    FROM save_snapshot_dirty_rows
                    WHERE save_id = ? AND table_name = ? AND row_key = ?
                    """,
                    (save_id, *aggregate_key),
                ).fetchone()
                if added is not None and aggregate_key not in queued_work_keys:
                    seed_projected_physical(*aggregate_key)
                    queued_work_keys.add(aggregate_key)
                    dirty_rows.append(added)
            table_name = str(dirty["table_name"])
            table = _TABLES_BY_NAME.get(table_name)
            if table is None:
                return None
            row_key = str(dirty["row_key"])
            previous = self.repositories.connection.execute(
                """
                SELECT object_hash, order_key, ordinal, included
                FROM save_snapshot_row_state
                WHERE save_id = ? AND table_name = ? AND row_key = ?
                """,
                (save_id, table_name, row_key),
            ).fetchone()
            live_row = self.repositories.connection.execute(
                f"""
                SELECT *
                FROM {table.name}
                WHERE save_id = ? AND {table.primary_key} = ?
                """,
                (save_id, row_key),
            ).fetchone()
            if live_row is None:
                add_pending(table, row_key, None, None, previous, None)
                continue
            raw_row = _row_dict(live_row)
            row = dict(raw_row)
            recheck_at = _snapshot_row_recheck_at(table.name, row)
            if table.name in {"messages", "character_text_messages"}:
                if row.get("deleted_at") is not None:
                    add_pending(table, row_key, None, raw_row, previous, None)
                    continue
            elif (
                table.active_only
                and "archived_at" in self._column_names(table.name)
                and row.get("archived_at") is not None
            ):
                add_pending(table, row_key, None, raw_row, previous, None)
                continue
            if table.name == "messages":
                row = _sanitize_snapshot_message_row(row)
                if is_fade_to_black_message(
                    role=str(row.get("role", "")),
                    body=str(row.get("body", "")),
                    safety_transition=str(row.get("safety_transition", "")),
                ):
                    add_pending(table, row_key, None, raw_row, previous, None)
                    continue
            elif table.name == "context_observation_curation_state":
                row = portable_context_observation_curation_state_row(row)
            elif table.name == "save_scenario_updates":
                row = _sanitize_snapshot_scenario_update_row(row)
            elif (
                table.name == "character_text_threads"
                and row.get("kind") == "group"
            ):
                participant_count = group_participant_count_cache.get(row_key)
                if participant_count is None:
                    participants = self.repositories.connection.execute(
                    """
                    SELECT participants.id, COALESCE(state.included, 1) AS included
                    FROM character_text_thread_participants participants
                    LEFT JOIN save_snapshot_row_state state
                      ON state.save_id = participants.save_id
                     AND state.table_name = 'character_text_thread_participants'
                     AND state.row_key = participants.id
                    WHERE participants.save_id = ? AND participants.thread_id = ?
                    """,
                    (save_id, row_key),
                    ).fetchall()
                    participant_count = sum(
                        projected_inclusion.get(
                            (
                                "character_text_thread_participants",
                                str(participant["id"]),
                            ),
                            bool(participant["included"]),
                        )
                        for participant in participants
                    )
                    group_participant_count_cache[row_key] = participant_count
                if participant_count < 2:
                    add_pending(table, row_key, None, raw_row, previous, None)
                    continue
            elif table.name == "narrator_phone_activity_cursors":
                max_ordinal = activity_max_ordinal_cache[0]
                if max_ordinal is None:
                    max_ordinal = self._projected_activity_max_ordinal(save_id)
                    activity_max_ordinal_cache[0] = max_ordinal
                row["last_activity_ordinal"] = min(
                    _snapshot_row_int(row, "last_activity_ordinal"),
                    int(max_ordinal),
                )
            if not self._incremental_row_references_are_active(
                save_id,
                table.name,
                row,
                projected_inclusion,
            ):
                if (
                    table.name not in _SNAPSHOT_KEEP_ENTITY_TABLES
                    or self._incremental_row_references_safety_transition(
                        save_id,
                        table.name,
                        row,
                    )
                ):
                    add_pending(table, row_key, None, raw_row, previous, None)
                    continue
                row = self._clear_incremental_scalar_references(
                    save_id,
                    table.name,
                    row,
                    projected_inclusion,
                )
            add_pending(table, row_key, row, raw_row, previous, recheck_at)

        for table, row_key, pending_row, reference_row, previous, recheck_at in (
            pending.values()
        ):
            table_name = table.name
            root_hash = table_roots[table_name]
            self._replace_snapshot_row_references(
                save_id=save_id,
                table_name=table_name,
                row_key=row_key,
                row=reference_row,
            )
            if pending_row is None:
                if previous is not None and bool(previous["included"]):
                    previous_order_key = cast(str | None, previous["order_key"])
                    if previous_order_key is not None:
                        root_hash = self._tree_delete(
                            table_name=table_name,
                            root_hash=root_hash,
                            order_key=previous_order_key,
                        )
                table_roots[table_name] = root_hash
                if previous is None:
                    ordinal_row = self.repositories.connection.execute(
                        """
                        SELECT next_ordinal FROM save_snapshot_table_state
                        WHERE save_id = ? AND table_name = ?
                        """,
                        (save_id, table_name),
                    ).fetchone()
                    ordinal = int(
                        ordinal_row["next_ordinal"] if ordinal_row else 0
                    )
                    previous_order_key = None
                else:
                    ordinal = int(previous["ordinal"])
                    previous_order_key = cast(str | None, previous["order_key"])
                raw_hash = (
                    self._store_object(
                        kind=f"row:{table_name}",
                        value=reference_row,
                    )
                    if reference_row is not None
                    else cast(str | None, previous["object_hash"])
                    if previous is not None
                    else None
                )
                self.repositories.connection.execute(
                    """
                    INSERT INTO save_snapshot_row_state(
                        save_id, table_name, row_key, object_hash,
                        order_key, ordinal, included, recheck_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, NULL)
                    ON CONFLICT(save_id, table_name, row_key) DO UPDATE SET
                        object_hash = excluded.object_hash,
                        order_key = excluded.order_key,
                        ordinal = excluded.ordinal,
                        included = 0,
                        recheck_at = NULL
                    """,
                    (
                        save_id,
                        table_name,
                        row_key,
                        raw_hash,
                        previous_order_key,
                        ordinal,
                    ),
                )
                self.repositories.connection.execute(
                    """
                    UPDATE save_snapshot_table_state
                    SET root_hash = ?, next_ordinal = MAX(next_ordinal, ?)
                    WHERE save_id = ? AND table_name = ?
                    """,
                    (root_hash, ordinal + 1, save_id, table_name),
                )
                continue
            row = pending_row
            if previous is None:
                ordinal_row = self.repositories.connection.execute(
                    """
                    SELECT next_ordinal
                    FROM save_snapshot_table_state
                    WHERE save_id = ? AND table_name = ?
                    """,
                    (save_id, table_name),
                ).fetchone()
                ordinal = int(ordinal_row["next_ordinal"] if ordinal_row else 0)
            else:
                ordinal = int(previous["ordinal"])
                previous_order_key = cast(str | None, previous["order_key"])
                if previous_order_key is not None and bool(previous["included"]):
                    root_hash = self._tree_delete(
                        table_name=table_name,
                        root_hash=root_hash,
                        order_key=previous_order_key,
                    )
            object_hash = self._store_object(
                kind=f"row:{table_name}",
                value=row,
            )
            order_key = _snapshot_tree_order_key(
                table,
                row=row,
                row_key=row_key,
                ordinal=ordinal,
            )
            root_hash = self._tree_insert(
                table_name=table_name,
                root_hash=root_hash,
                order_key=order_key,
                row_key=row_key,
                row_hash=object_hash,
            )
            table_roots[table_name] = root_hash
            self.repositories.connection.execute(
                """
                INSERT INTO save_snapshot_row_state(
                    save_id, table_name, row_key, object_hash,
                    order_key, ordinal, included, recheck_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(save_id, table_name, row_key) DO UPDATE SET
                    object_hash = excluded.object_hash,
                    order_key = excluded.order_key,
                    ordinal = excluded.ordinal,
                    included = 1,
                    recheck_at = excluded.recheck_at
                """,
                (
                    save_id,
                    table_name,
                    row_key,
                    object_hash,
                    order_key,
                    ordinal,
                    recheck_at,
                ),
            )
            self.repositories.connection.execute(
                """
                UPDATE save_snapshot_table_state
                SET root_hash = ?,
                    next_ordinal = MAX(next_ordinal, ?)
                WHERE save_id = ? AND table_name = ?
                """,
                (root_hash, ordinal + 1, save_id, table_name),
            )
        context_revision = self._context_revision(save_id)
        root_manifest_hash = self._store_object(
            kind="snapshot_manifest",
            value={
                "format": SNAPSHOT_FORMAT_V2,
                "save_id": save_id,
                "message_id": message_id,
                "context_revision": context_revision,
                "table_roots": table_roots,
            },
        )
        return _PreparedSnapshot(
            root_manifest_hash=root_manifest_hash,
            context_revision=context_revision,
            table_roots=table_roots,
            object_count=len(pending),
        )

    def _incremental_row_references_are_active(
        self,
        save_id: str,
        table_name: str,
        row: Mapping[str, object],
        projected_inclusion: Mapping[tuple[str, str], bool],
    ) -> bool:
        return not _snapshot_row_has_unresolved_references(
            table_name,
            row,
            self._incremental_active_ids(
                save_id,
                table_name,
                row,
                projected_inclusion,
            ),
        )


    def _queue_snapshot_reference_dependents(
        self,
        *,
        save_id: str,
        target_table: str,
        target_key: str,
    ) -> tuple[tuple[str, str], ...]:
        dependents = self.repositories.connection.execute(
            """
            SELECT source_table, source_key
            FROM save_snapshot_row_references
            WHERE save_id = ? AND target_table = ? AND target_key = ?
            """,
            (save_id, target_table, target_key),
        ).fetchall()
        for dependent in dependents:
            self._mark_snapshot_row_dirty(
                save_id=save_id,
                table_name=str(dependent["source_table"]),
                row_key=str(dependent["source_key"]),
            )
        return tuple(
            (str(dependent["source_table"]), str(dependent["source_key"]))
            for dependent in dependents
        )

    def _queue_snapshot_lifecycle_dependents(
        self,
        save_id: str,
        dirty_rows: list[sqlite3.Row],
    ) -> tuple[tuple[str, str], ...]:
        queued: set[tuple[str, str]] = set()
        for dirty in dirty_rows:
            table_name = str(dirty["table_name"])
            table = _TABLES_BY_NAME.get(table_name)
            if table is None:
                continue
            row_key = str(dirty["row_key"])
            previous = self.repositories.connection.execute(
                """
                SELECT included FROM save_snapshot_row_state
                WHERE save_id = ? AND table_name = ? AND row_key = ?
                """,
                (save_id, table_name, row_key),
            ).fetchone()
            if previous is None:
                live_exists = self.repositories.connection.execute(
                    f"SELECT 1 FROM {table.name} "
                    f"WHERE save_id = ? AND {table.primary_key} = ?",
                    (save_id, row_key),
                ).fetchone()
                if live_exists is not None:
                    queued.update(self._queue_snapshot_reference_dependents(
                        save_id=save_id,
                        target_table=table_name,
                        target_key=row_key,
                    ))
                continue
            live = self.repositories.connection.execute(
                f"SELECT * FROM {table.name} "
                f"WHERE save_id = ? AND {table.primary_key} = ?",
                (save_id, row_key),
            ).fetchone()
            is_included = live is not None
            if live is not None:
                row = _row_dict(live)
                if table.name in {"messages", "character_text_messages"}:
                    is_included = row.get("deleted_at") is None
                    if table.name == "messages" and is_included:
                        sanitized_message = _sanitize_snapshot_message_row(row)
                        is_included = not is_fade_to_black_message(
                            role=str(sanitized_message.get("role", "")),
                            body=str(sanitized_message.get("body", "")),
                            safety_transition=str(
                                sanitized_message.get("safety_transition", "")
                            ),
                        )
                elif (
                    table.active_only
                    and "archived_at" in self._column_names(table.name)
                ):
                    is_included = row.get("archived_at") is None
            if bool(previous["included"]) != is_included:
                queued.update(self._queue_snapshot_reference_dependents(
                    save_id=save_id,
                    target_table=table_name,
                    target_key=row_key,
                ))
        return tuple(sorted(queued))

    def _expand_snapshot_reverse_dependency_closure(
        self,
        save_id: str,
        roots: tuple[tuple[str, str], ...],
    ) -> None:
        pending = list(roots)
        seen = set(roots)
        while pending:
            target_table, target_key = pending.pop()
            dependents = self._queue_snapshot_reference_dependents(
                save_id=save_id,
                target_table=target_table,
                target_key=target_key,
            )
            for dependent in dependents:
                if dependent in seen:
                    continue
                seen.add(dependent)
                pending.append(dependent)

    def _queue_snapshot_aggregate_dependents(
        self,
        save_id: str,
        dirty_rows: list[sqlite3.Row],
        *,
        already_queued: set[tuple[str, str]],
    ) -> tuple[tuple[str, str], ...]:
        participant_thread_ids: set[str] = set()
        queued: list[tuple[str, str]] = []
        for dirty in dirty_rows:
            table_name = str(dirty["table_name"])
            row_key = str(dirty["row_key"])
            if table_name != "character_text_thread_participants":
                continue
            live = self.repositories.connection.execute(
                """
                SELECT thread_id FROM character_text_thread_participants
                WHERE save_id = ? AND id = ?
                """,
                (save_id, row_key),
            ).fetchone()
            if live is not None:
                participant_thread_ids.add(str(live["thread_id"]))
            state = self.repositories.connection.execute(
                """
                SELECT object_hash FROM save_snapshot_row_state
                WHERE save_id = ?
                  AND table_name = 'character_text_thread_participants'
                  AND row_key = ?
                """,
                (save_id, row_key),
            ).fetchone()
            if state is None or state["object_hash"] is None:
                continue
            previous = self._load_object(
                str(state["object_hash"]),
                expected_kind="row:character_text_thread_participants",
            )
            if isinstance(previous, Mapping) and isinstance(
                previous.get("thread_id"),
                str,
            ):
                participant_thread_ids.add(str(previous["thread_id"]))
        for thread_id in participant_thread_ids:
            dependent_key = ("character_text_threads", thread_id)
            if dependent_key in already_queued:
                continue
            self._mark_snapshot_row_dirty(
                save_id=save_id,
                table_name="character_text_threads",
                row_key=thread_id,
            )
            already_queued.add(dependent_key)
            queued.append(dependent_key)
        return tuple(queued)

    def _projected_activity_max_ordinal(
        self,
        save_id: str,
    ) -> int:
        activity = self.repositories.connection.execute(
            """
            SELECT ordinal
            FROM save_snapshot_included_activity_events
            WHERE save_id = ?
            ORDER BY ordinal DESC
            LIMIT 1
            """,
            (save_id,),
        ).fetchone()
        return int(activity["ordinal"]) if activity else 0

    def _queue_activity_cursors_if_max_changed(
        self,
        save_id: str,
    ) -> tuple[tuple[str, str], ...]:
        new_max = self._projected_activity_max_ordinal(save_id)
        state = self.repositories.connection.execute(
            """
            SELECT max_ordinal FROM save_snapshot_activity_state
            WHERE save_id = ?
            """,
            (save_id,),
        ).fetchone()
        old_max = int(state["max_ordinal"]) if state is not None else 0
        if new_max == old_max:
            return ()
        self.repositories.connection.execute(
            """
            INSERT INTO save_snapshot_activity_state(save_id, max_ordinal)
            VALUES (?, ?)
            ON CONFLICT(save_id) DO UPDATE SET
                max_ordinal = excluded.max_ordinal
            """,
            (save_id, new_max),
        )
        cursors = self.repositories.connection.execute(
            """
            SELECT narrator_message_id FROM narrator_phone_activity_cursors
            WHERE save_id = ?
            """,
            (save_id,),
        ).fetchall()
        keys: list[tuple[str, str]] = []
        for cursor in cursors:
            row_key = str(cursor["narrator_message_id"])
            self._mark_snapshot_row_dirty(
                save_id=save_id,
                table_name="narrator_phone_activity_cursors",
                row_key=row_key,
            )
            keys.append(("narrator_phone_activity_cursors", row_key))
        return tuple(keys)

    def _mark_snapshot_row_dirty(
        self,
        *,
        save_id: str,
        table_name: str,
        row_key: str,
    ) -> None:
        self.repositories.connection.execute(
            """
            UPDATE save_snapshot_table_state
            SET current_generation = current_generation + 1
            WHERE save_id = ? AND table_name = ?
            """,
            (save_id, table_name),
        )
        self.repositories.connection.execute(
            """
            INSERT INTO save_snapshot_dirty_rows(
                save_id, table_name, row_key, generation
            )
            SELECT ?, ?, ?, current_generation
            FROM save_snapshot_table_state
            WHERE save_id = ? AND table_name = ?
            ON CONFLICT(save_id, table_name, row_key) DO UPDATE SET
                generation = excluded.generation
            """,
            (save_id, table_name, row_key, save_id, table_name),
        )

    def _replace_snapshot_row_references(
        self,
        *,
        save_id: str,
        table_name: str,
        row_key: str,
        row: Mapping[str, object] | None,
    ) -> None:
        self.repositories.connection.execute(
            """
            DELETE FROM save_snapshot_row_references
            WHERE save_id = ? AND source_table = ? AND source_key = ?
            """,
            (save_id, table_name, row_key),
        )
        if row is None:
            return
        self.repositories.connection.executemany(
            """
            INSERT OR IGNORE INTO save_snapshot_row_references(
                save_id, source_table, source_key, target_table, target_key
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                (save_id, table_name, row_key, target_table, target_key)
                for target_table, target_key in self._snapshot_row_references(
                    save_id,
                    table_name,
                    row,
                )
            ),
        )

    def _snapshot_row_references(
        self,
        save_id: str,
        table_name: str,
        row: Mapping[str, object],
        *,
        active_only: bool = False,
    ) -> frozenset[tuple[str, str]]:
        candidates = _snapshot_reference_candidates(table_name, row)
        references = (
            set()
            if active_only
            else set(_declared_snapshot_row_references(table_name, row))
        )
        if not candidates:
            return frozenset()
        candidate_values = tuple(candidates)
        for target in _SNAPSHOT_TABLES:
            for offset in range(0, len(candidate_values), 500):
                chunk = candidate_values[offset : offset + 500]
                where = (
                    f"save_id = ? AND {target.primary_key} "
                    f"IN ({_placeholders(len(chunk))})"
                )
                if active_only:
                    if target.name in {"messages", "character_text_messages"}:
                        where += " AND deleted_at IS NULL"
                    elif target.active_only and "archived_at" in self._column_names(
                        target.name
                    ):
                        where += " AND archived_at IS NULL"
                select_columns = target.primary_key
                if active_only and target.name == "messages":
                    select_columns += ", role, body, safety_transition"
                matched = self.repositories.connection.execute(
                    f"SELECT {select_columns} FROM {target.name} WHERE {where}",
                    (save_id, *chunk),
                ).fetchall()
                references.update(
                    (target.name, str(match[target.primary_key]))
                    for match in matched
                    if target.name != "messages"
                    or not active_only
                    or not is_fade_to_black_message(
                        role=str(match["role"]),
                        body=str(match["body"]),
                        safety_transition=str(match["safety_transition"]),
                    )
                )
        return frozenset(references)

    def _incremental_active_ids(
        self,
        save_id: str,
        table_name: str,
        row: Mapping[str, object],
        projected_inclusion: Mapping[tuple[str, str], bool],
    ) -> dict[str, frozenset[str]]:
        references = self._snapshot_row_references(
            save_id,
            table_name,
            row,
            active_only=True,
        )
        active_ids: dict[str, set[str]] = {
            target.name: set() for target in _SNAPSHOT_TABLES
        }
        for target_table, target_key in references:
            key = (target_table, target_key)
            included = projected_inclusion.get(key)
            if included is None:
                state = self.repositories.connection.execute(
                    """
                    SELECT included FROM save_snapshot_row_state
                    WHERE save_id = ? AND table_name = ? AND row_key = ?
                    """,
                    (save_id, target_table, target_key),
                ).fetchone()
                included = state is None or bool(state["included"])
            if included:
                active_ids[target_table].add(target_key)
        return {
            target_table: frozenset(target_keys)
            for target_table, target_keys in active_ids.items()
        }

    def _incremental_row_references_safety_transition(
        self,
        save_id: str,
        table_name: str,
        row: Mapping[str, object],
    ) -> bool:
        message_candidates = _snapshot_reference_candidates(table_name, row)
        if not message_candidates:
            return False
        candidate_values = tuple(message_candidates)
        for offset in range(0, len(candidate_values), 500):
            chunk = candidate_values[offset : offset + 500]
            messages = self.repositories.connection.execute(
                f"""
                SELECT role, body, safety_transition
                FROM messages
                WHERE save_id = ? AND id IN ({_placeholders(len(chunk))})
                  AND deleted_at IS NULL
                """,
                (save_id, *chunk),
            ).fetchall()
            if any(
                is_fade_to_black_message(
                    role=str(message["role"]),
                    body=str(message["body"]),
                    safety_transition=str(message["safety_transition"]),
                )
                for message in messages
            ):
                return True
        return False


    def _clear_incremental_scalar_references(
        self,
        save_id: str,
        table_name: str,
        row: Mapping[str, object],
        projected_inclusion: Mapping[tuple[str, str], bool],
    ) -> dict[str, object]:
        return _clear_snapshot_row_unresolved_references(
            table_name,
            row,
            self._incremental_active_ids(
                save_id,
                table_name,
                row,
                projected_inclusion,
            ),
        )


    def _queue_due_snapshot_rechecks(self, save_id: str) -> None:
        due_rows = self.repositories.connection.execute(
            """
            SELECT table_name, row_key
            FROM save_snapshot_row_state
            WHERE save_id = ?
              AND recheck_at IS NOT NULL
              AND recheck_at <= CURRENT_TIMESTAMP
            """,
            (save_id,),
        ).fetchall()
        for row in due_rows:
            table_name = str(row["table_name"])
            row_key = str(row["row_key"])
            self.repositories.connection.execute(
                """
                UPDATE save_snapshot_table_state
                SET current_generation = current_generation + 1
                WHERE save_id = ? AND table_name = ?
                """,
                (save_id, table_name),
            )
            self.repositories.connection.execute(
                """
                INSERT INTO save_snapshot_dirty_rows(
                    save_id, table_name, row_key, generation
                )
                SELECT ?, ?, ?, current_generation
                FROM save_snapshot_table_state
                WHERE save_id = ? AND table_name = ?
                ON CONFLICT(save_id, table_name, row_key) DO UPDATE SET
                    generation = excluded.generation
                """,
                (save_id, table_name, row_key, save_id, table_name),
            )

    def _snapshot_recheck_deadlines(
        self,
        save_id: str,
    ) -> dict[tuple[str, str], str]:
        rows = self.repositories.connection.execute(
            """
            SELECT observation_id, lease_token, lease_until
            FROM context_observation_curation_state
            WHERE save_id = ?
            """,
            (save_id,),
        ).fetchall()
        deadlines: dict[tuple[str, str], str] = {}
        for row in rows:
            recheck_at = _snapshot_row_recheck_at(
                "context_observation_curation_state",
                _row_dict(row),
            )
            if recheck_at is not None:
                deadlines[
                    (
                        "context_observation_curation_state",
                        str(row["observation_id"]),
                    )
                ] = recheck_at
        return deadlines

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
        self.repositories.connection.execute(
            """
            INSERT INTO save_snapshot_state(
                save_id, base_snapshot_id, base_message_id, updated_at
            )
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(save_id) DO UPDATE SET
                base_snapshot_id = excluded.base_snapshot_id,
                base_message_id = excluded.base_message_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (save_id, snapshot_id, message_id),
        )
        self.repositories.connection.execute(
            """
            UPDATE save_snapshot_table_state
            SET captured_generation = current_generation,
                needs_rebuild = 0
            WHERE save_id = ?
            """,
            (save_id,),
        )
        self.repositories.connection.execute(
            "DELETE FROM save_snapshot_dirty_rows WHERE save_id = ?",
            (save_id,),
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
            object_count=prepared.object_count,
        )
        return snapshot

    def _clean_materialized_snapshot(
        self,
        save_id: str,
        *,
        message_id: str | None | object = ...,
    ) -> TurnSnapshotRecord | None:
        row = self.repositories.connection.execute(
            """
            SELECT state.base_snapshot_id, state.base_message_id
            FROM save_snapshot_state state
            WHERE state.save_id = ?
              AND state.base_snapshot_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM save_snapshot_table_state tables
                  WHERE tables.save_id = state.save_id
                    AND (
                        tables.needs_rebuild != 0
                        OR tables.current_generation != tables.captured_generation
                    )
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM save_snapshot_dirty_rows dirty
                  WHERE dirty.save_id = state.save_id
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM save_snapshot_row_state rows
                  WHERE rows.save_id = state.save_id
                    AND rows.recheck_at IS NOT NULL
                    AND rows.recheck_at <= CURRENT_TIMESTAMP
              )
            """,
            (save_id,),
        ).fetchone()
        if row is None:
            return None
        base_message_id = cast(str | None, row["base_message_id"])
        if message_id is not ... and base_message_id != message_id:
            return None
        return self._get_snapshot(str(row["base_snapshot_id"]))

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
                    portable_context_observation_curation_state_row(row)
                    for row in table_rows
                )
            else:
                rows[table.name] = tuple(table_rows)
        _filter_turn_outcomes_to_active_messages(rows)
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

    def _tree_insert(
        self,
        *,
        table_name: str,
        root_hash: str | None,
        order_key: str,
        row_key: str,
        row_hash: str,
        guard: _TreeMutationGuard | None = None,
        depth: int = 0,
    ) -> str:
        if depth > _MAX_SNAPSHOT_TREE_MUTATION_DEPTH:
            raise ValueError("Snapshot table tree is too deep")
        guard = guard or _new_tree_mutation_guard()
        priority = _snapshot_tree_priority(table_name, row_key)
        if root_hash is None:
            return self._store_tree_node(
                table_name=table_name,
                order_key=order_key,
                row_key=row_key,
                row_hash=row_hash,
                priority=priority,
                left_hash=None,
                right_hash=None,
            )
        node = self._load_guarded_tree_node(table_name, root_hash, guard)
        node_order_key = _text(node, "order_key")
        if order_key == node_order_key:
            return self._store_tree_node(
                table_name=table_name,
                order_key=order_key,
                row_key=row_key,
                row_hash=row_hash,
                priority=priority,
                left_hash=_optional_text(node, "left_hash"),
                right_hash=_optional_text(node, "right_hash"),
            )
        if priority < _int(node, "priority"):
            left_hash, right_hash = self._tree_split(
                table_name=table_name,
                root_hash=root_hash,
                order_key=order_key,
            )
            return self._store_tree_node(
                table_name=table_name,
                order_key=order_key,
                row_key=row_key,
                row_hash=row_hash,
                priority=priority,
                left_hash=left_hash,
                right_hash=right_hash,
            )
        if order_key < node_order_key:
            left_hash = self._tree_insert(
                table_name=table_name,
                root_hash=_optional_text(node, "left_hash"),
                order_key=order_key,
                row_key=row_key,
                row_hash=row_hash,
                guard=guard,
                depth=depth + 1,
            )
            return self._store_tree_node_from_value(
                table_name=table_name,
                node=node,
                left_hash=left_hash,
            )
        right_hash = self._tree_insert(
            table_name=table_name,
            root_hash=_optional_text(node, "right_hash"),
            order_key=order_key,
            row_key=row_key,
            row_hash=row_hash,
            guard=guard,
            depth=depth + 1,
        )
        return self._store_tree_node_from_value(
            table_name=table_name,
            node=node,
            right_hash=right_hash,
        )

    def _tree_split(
        self,
        *,
        table_name: str,
        root_hash: str | None,
        order_key: str,
        guard: _TreeMutationGuard | None = None,
        depth: int = 0,
    ) -> tuple[str | None, str | None]:
        if depth > _MAX_SNAPSHOT_TREE_MUTATION_DEPTH:
            raise ValueError("Snapshot table tree is too deep")
        guard = guard or _new_tree_mutation_guard()
        if root_hash is None:
            return None, None
        node = self._load_guarded_tree_node(table_name, root_hash, guard)
        if _text(node, "order_key") < order_key:
            left_of_right, right_hash = self._tree_split(
                table_name=table_name,
                root_hash=_optional_text(node, "right_hash"),
                order_key=order_key,
                guard=guard,
                depth=depth + 1,
            )
            return (
                self._store_tree_node_from_value(
                    table_name=table_name,
                    node=node,
                    right_hash=left_of_right,
                ),
                right_hash,
            )
        left_hash, right_of_left = self._tree_split(
            table_name=table_name,
            root_hash=_optional_text(node, "left_hash"),
            order_key=order_key,
            guard=guard,
            depth=depth + 1,
        )
        return (
            left_hash,
            self._store_tree_node_from_value(
                table_name=table_name,
                node=node,
                left_hash=right_of_left,
            ),
        )

    def _tree_delete(
        self,
        *,
        table_name: str,
        root_hash: str | None,
        order_key: str,
        guard: _TreeMutationGuard | None = None,
        depth: int = 0,
    ) -> str | None:
        if depth > _MAX_SNAPSHOT_TREE_MUTATION_DEPTH:
            raise ValueError("Snapshot table tree is too deep")
        guard = guard or _new_tree_mutation_guard()
        if root_hash is None:
            return None
        node = self._load_guarded_tree_node(table_name, root_hash, guard)
        node_order_key = _text(node, "order_key")
        if order_key == node_order_key:
            return self._tree_merge(
                table_name=table_name,
                left_hash=_optional_text(node, "left_hash"),
                right_hash=_optional_text(node, "right_hash"),
            )
        if order_key < node_order_key:
            left_hash = self._tree_delete(
                table_name=table_name,
                root_hash=_optional_text(node, "left_hash"),
                order_key=order_key,
                guard=guard,
                depth=depth + 1,
            )
            return self._store_tree_node_from_value(
                table_name=table_name,
                node=node,
                left_hash=left_hash,
            )
        right_hash = self._tree_delete(
            table_name=table_name,
            root_hash=_optional_text(node, "right_hash"),
            order_key=order_key,
            guard=guard,
            depth=depth + 1,
        )
        return self._store_tree_node_from_value(
            table_name=table_name,
            node=node,
            right_hash=right_hash,
        )

    def _tree_merge(
        self,
        *,
        table_name: str,
        left_hash: str | None,
        right_hash: str | None,
        guard: _TreeMutationGuard | None = None,
        depth: int = 0,
    ) -> str | None:
        if depth > _MAX_SNAPSHOT_TREE_MUTATION_DEPTH:
            raise ValueError("Snapshot table tree is too deep")
        guard = guard or _new_tree_mutation_guard()
        if left_hash is None:
            return right_hash
        if right_hash is None:
            return left_hash
        left = self._load_guarded_tree_node(table_name, left_hash, guard)
        right = self._load_guarded_tree_node(table_name, right_hash, guard)
        if _int(left, "priority") < _int(right, "priority"):
            guard.seen.discard(right_hash)
            merged_right = self._tree_merge(
                table_name=table_name,
                left_hash=_optional_text(left, "right_hash"),
                right_hash=right_hash,
                guard=guard,
                depth=depth + 1,
            )
            return self._store_tree_node_from_value(
                table_name=table_name,
                node=left,
                right_hash=merged_right,
            )
        guard.seen.discard(left_hash)
        merged_left = self._tree_merge(
            table_name=table_name,
            left_hash=left_hash,
            right_hash=_optional_text(right, "left_hash"),
            guard=guard,
            depth=depth + 1,
        )
        return self._store_tree_node_from_value(
            table_name=table_name,
            node=right,
            left_hash=merged_left,
        )

    def _store_tree_node_from_value(
        self,
        *,
        table_name: str,
        node: Mapping[str, object],
        left_hash: str | None | object = ...,
        right_hash: str | None | object = ...,
    ) -> str:
        return self._store_tree_node(
            table_name=table_name,
            order_key=_text(node, "order_key"),
            row_key=_text(node, "row_key"),
            row_hash=_text(node, "row_hash"),
            priority=_int(node, "priority"),
            left_hash=(
                _optional_text(node, "left_hash")
                if left_hash is ...
                else cast(str | None, left_hash)
            ),
            right_hash=(
                _optional_text(node, "right_hash")
                if right_hash is ...
                else cast(str | None, right_hash)
            ),
        )

    def _store_tree_node(
        self,
        *,
        table_name: str,
        order_key: str,
        row_key: str,
        row_hash: str,
        priority: int,
        left_hash: str | None,
        right_hash: str | None,
    ) -> str:
        return self._store_object(
            kind=f"snapshot_table_node:{table_name}",
            value={
                "table": table_name,
                "order_key": order_key,
                "row_key": row_key,
                "row_hash": row_hash,
                "priority": priority,
                "left_hash": left_hash,
                "right_hash": right_hash,
            },
        )

    def _load_tree_node(
        self,
        table_name: str,
        object_hash: str,
        *,
        byte_budget: list[int] | None = None,
        json_node_budget: list[int] | None = None,
        cache: dict[str, tuple[str, object]] | None = None,
    ) -> dict[str, object]:
        value = self._load_object(
            object_hash,
            expected_kind=f"snapshot_table_node:{table_name}",
            byte_budget=byte_budget,
            json_node_budget=json_node_budget,
            cache=cache,
        )
        if not isinstance(value, dict) or value.get("table") != table_name:
            raise ValueError(f"Invalid snapshot table node: {object_hash}")
        return cast(dict[str, object], value)

    def _load_guarded_tree_node(
        self,
        table_name: str,
        object_hash: str,
        guard: _TreeMutationGuard,
    ) -> dict[str, object]:
        if object_hash in guard.seen:
            raise ValueError("Snapshot table tree contains a cycle")
        guard.seen.add(object_hash)
        if len(guard.seen) > _MAX_SNAPSHOT_MANIFEST_ENTRIES:
            raise ValueError("Snapshot table tree contains too many entries")
        return self._load_tree_node(
            table_name,
            object_hash,
            byte_budget=guard.byte_budget,
            json_node_budget=guard.json_node_budget,
            cache=guard.cache,
        )

    def _tree_entries(
        self,
        *,
        table_name: str,
        root_hash: str | None,
        byte_budget: list[int] | None = None,
        json_node_budget: list[int] | None = None,
        cache: dict[str, tuple[str, object]] | None = None,
    ) -> tuple[tuple[str, str], ...]:
        if root_hash is None:
            return ()
        entries: list[tuple[str, str]] = []
        pending: list[tuple[str, bool]] = [(root_hash, False)]
        seen: set[str] = set()
        while pending:
            object_hash, expanded = pending.pop()
            if not expanded:
                if object_hash in seen:
                    raise ValueError("Snapshot table tree contains a cycle")
                seen.add(object_hash)
                if len(seen) > _MAX_SNAPSHOT_MANIFEST_ENTRIES:
                    raise ValueError("Snapshot table tree contains too many entries")
            node = self._load_tree_node(
                table_name,
                object_hash,
                byte_budget=byte_budget,
                json_node_budget=json_node_budget,
                cache=cache,
            )
            if expanded:
                entries.append((_text(node, "row_key"), _text(node, "row_hash")))
                right_hash = _optional_text(node, "right_hash")
                if right_hash is not None:
                    pending.append((right_hash, False))
                continue
            pending.append((object_hash, True))
            left_hash = _optional_text(node, "left_hash")
            if left_hash is not None:
                pending.append((left_hash, False))
        return tuple(entries)

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
        manifest = self._load_object(
            snapshot.root_manifest_hash,
            expected_kind="snapshot_manifest",
        )
        if not isinstance(manifest, dict) or manifest.get("format") not in {
            SNAPSHOT_FORMAT,
            SNAPSHOT_FORMAT_V2,
        }:
            raise ValueError(f"Invalid snapshot manifest: {snapshot.id}")
        if manifest.get("save_id") != snapshot.save_id:
            raise ValueError(f"Snapshot manifest has wrong save id: {snapshot.id}")
        return cast(dict[str, object], manifest)

    def _load_object(
        self,
        object_hash: str,
        *,
        expected_kind: str | None = None,
        byte_budget: list[int] | None = None,
        json_node_budget: list[int] | None = None,
        cache: dict[str, tuple[str, object]] | None = None,
    ) -> object:
        if cache is not None and object_hash in cache:
            cached_kind, cached_value = cache[object_hash]
            if expected_kind is not None and cached_kind != expected_kind:
                raise ValueError(f"Unexpected snapshot object kind: {object_hash}")
            return cached_value
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
        kind = str(row["kind"])
        if expected_kind is not None and kind != expected_kind:
            raise ValueError(f"Unexpected snapshot object kind: {object_hash}")
        declared_size = int(row["uncompressed_size"])
        if (
            declared_size < 0
            or declared_size > _MAX_SNAPSHOT_OBJECT_UNCOMPRESSED_BYTES
        ):
            raise ValueError(f"Snapshot object is too large: {object_hash}")
        if byte_budget is not None:
            if declared_size > byte_budget[0]:
                raise ValueError("Snapshot objects exceed total size limit")
            byte_budget[0] -= declared_size
        remaining_json_nodes = _MAX_SNAPSHOT_OBJECT_JSON_NODES
        if json_node_budget is not None:
            remaining_json_nodes = min(remaining_json_nodes, json_node_budget[0])
            if remaining_json_nodes <= 0:
                raise ValueError("Snapshot JSON contains too many values")
        try:
            decompressor = zlib.decompressobj()
            payload = decompressor.decompress(
                bytes(row["payload"]),
                declared_size + 1,
            )
            if (
                len(payload) > declared_size
                or decompressor.unconsumed_tail
                or not decompressor.eof
            ):
                raise ValueError(f"Snapshot object size mismatch: {object_hash}")
            payload += decompressor.flush()
            node_count = validate_json_structure(
                payload,
                max_nodes=remaining_json_nodes,
                max_depth=_MAX_SNAPSHOT_OBJECT_JSON_DEPTH,
            )
            value = json.loads(payload.decode("utf-8"))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            JsonSafetyError,
            zlib.error,
        ) as exc:
            raise ValueError(f"Invalid snapshot object payload: {object_hash}") from exc
        if len(payload) != declared_size:
            raise ValueError(f"Snapshot object size mismatch: {object_hash}")
        if _snapshot_object_hash(
            kind=kind,
            payload=payload,
        ) != object_hash:
            raise ValueError(f"Snapshot object hash mismatch: {object_hash}")
        if json_node_budget is not None:
            json_node_budget[0] -= node_count
        if cache is not None:
            cache[object_hash] = (kind, value)
        return value

    def _reachable_snapshot_object_hashes(self) -> set[str]:
        rows = self.repositories.connection.execute(
            """
            SELECT root_manifest_hash
            FROM save_turn_snapshots
            ORDER BY rowid
            """
        ).fetchall()
        reachable = {str(row["root_manifest_hash"]) for row in rows}
        byte_budget = [_MAX_SNAPSHOT_TOTAL_UNCOMPRESSED_BYTES]
        json_node_budget = [_MAX_SNAPSHOT_TOTAL_JSON_NODES]
        object_cache: dict[str, tuple[str, object]] = {}
        tree_node_count = 0
        for object_hash in tuple(reachable):
            manifest = self._load_object(
                object_hash,
                expected_kind="snapshot_manifest",
                byte_budget=byte_budget,
                json_node_budget=json_node_budget,
                cache=object_cache,
            )
            if not isinstance(manifest, Mapping):
                raise ValueError(
                    f"Snapshot manifest object is not valid: {object_hash}"
                )
            if manifest.get("format") == SNAPSHOT_FORMAT_V2:
                raw_roots = manifest.get("table_roots", {})
                if not isinstance(raw_roots, Mapping):
                    raise ValueError("Snapshot manifest is missing table roots")
                for table_name, raw_root_hash in raw_roots.items():
                    if table_name not in _TABLES_BY_NAME or not isinstance(
                        raw_root_hash, str
                    ):
                        continue
                    pending = [raw_root_hash]
                    while pending:
                        node_hash = pending.pop()
                        if node_hash in reachable:
                            continue
                        reachable.add(node_hash)
                        tree_node_count += 1
                        if tree_node_count > _MAX_SNAPSHOT_MANIFEST_ENTRIES:
                            raise ValueError(
                                "Snapshot table trees contain too many entries"
                            )
                        node = self._load_tree_node(
                            str(table_name),
                            node_hash,
                            byte_budget=byte_budget,
                            json_node_budget=json_node_budget,
                            cache=object_cache,
                        )
                        reachable.add(_text(node, "row_hash"))
                        for child_key in ("left_hash", "right_hash"):
                            child_hash = _optional_text(node, child_key)
                            if child_hash is not None:
                                pending.append(child_hash)
            else:
                for table_rows in _manifest_tables(manifest).values():
                    for entry in table_rows:
                        reachable.add(_text(entry, "object_hash"))
        return reachable

    def _rows_from_manifest(
        self,
        manifest: Mapping[str, object],
    ) -> dict[str, tuple[dict[str, object], ...]]:
        expected_save_id = _text(manifest, "save_id")
        total_entries = 0
        seen_row_hashes: set[str] = set()
        byte_budget = [_MAX_SNAPSHOT_TOTAL_UNCOMPRESSED_BYTES]
        json_node_budget = [_MAX_SNAPSHOT_TOTAL_JSON_NODES]
        object_cache: dict[str, tuple[str, object]] = {}
        rows_by_table: dict[str, tuple[dict[str, object], ...]] = {}
        if manifest.get("format") == SNAPSHOT_FORMAT_V2:
            raw_roots = manifest.get("table_roots")
            if not isinstance(raw_roots, Mapping):
                raise ValueError("Snapshot manifest is missing table roots")
            for table_name in _SNAPSHOT_TABLE_NAMES:
                raw_root_hash = raw_roots.get(table_name)
                root_hash = raw_root_hash if isinstance(raw_root_hash, str) else None
                table_rows: list[dict[str, object]] = []
                for _row_key, row_hash in self._tree_entries(
                    table_name=table_name,
                    root_hash=root_hash,
                    byte_budget=byte_budget,
                    json_node_budget=json_node_budget,
                    cache=object_cache,
                ):
                    total_entries += 1
                    if total_entries > _MAX_SNAPSHOT_MANIFEST_ENTRIES:
                        raise ValueError("Snapshot manifest contains too many entries")
                    if row_hash in seen_row_hashes:
                        raise ValueError("Duplicate snapshot row object")
                    seen_row_hashes.add(row_hash)
                    value = self._load_object(
                        row_hash,
                        expected_kind=f"row:{table_name}",
                        byte_budget=byte_budget,
                        json_node_budget=json_node_budget,
                        cache=object_cache,
                    )
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"Snapshot row object is not a row: {row_hash}"
                        )
                    if value.get("save_id") != expected_save_id:
                        raise ValueError("Snapshot row has wrong save id")
                    table_rows.append(cast(dict[str, object], value))
                rows_by_table[table_name] = tuple(table_rows)
            return rows_by_table
        for table_name, entries in _manifest_tables(manifest).items():
            rows: list[dict[str, object]] = []
            for entry in entries:
                object_hash = _text(entry, "object_hash")
                total_entries += 1
                if total_entries > _MAX_SNAPSHOT_MANIFEST_ENTRIES:
                    raise ValueError("Snapshot manifest contains too many entries")
                if object_hash in seen_row_hashes:
                    raise ValueError("Duplicate snapshot row object")
                seen_row_hashes.add(object_hash)
                value = self._load_object(
                    object_hash,
                    expected_kind=f"row:{table_name}",
                    byte_budget=byte_budget,
                    json_node_budget=json_node_budget,
                    cache=object_cache,
                )
                if not isinstance(value, dict):
                    raise ValueError(f"Snapshot row object is not a row: {object_hash}")
                if value.get("save_id") != expected_save_id:
                    raise ValueError("Snapshot row has wrong save id")
                rows.append(cast(dict[str, object], value))
            rows_by_table[table_name] = tuple(rows)
        return rows_by_table

    def _rows_from_exported_manifest(
        self,
        objects_by_hash: Mapping[str, tuple[str, object]],
        manifest_hash: str,
    ) -> dict[str, tuple[dict[str, object], ...]]:
        manifest_object = _required_exported_snapshot_object(
            objects_by_hash,
            manifest_hash,
            expected_kind="snapshot_manifest",
        )
        if not isinstance(manifest_object, Mapping):
            raise ValueError("Snapshot manifest object is not a JSON object")
        manifest = cast(Mapping[str, object], manifest_object)
        rows_by_table: dict[str, tuple[dict[str, object], ...]] = {}
        for table_name, entries in _manifest_tables(manifest).items():
            rows: list[dict[str, object]] = []
            for entry in entries:
                value = _required_exported_snapshot_object(
                    objects_by_hash,
                    _text(entry, "object_hash"),
                    expected_kind=f"row:{table_name}",
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

    def _normalize_memory_fingerprints(self, save_id: str) -> None:
        rows = self.repositories.connection.execute(
            """
            SELECT id, body
            FROM memories
            WHERE save_id = ?
              AND archived_at IS NULL
              AND (claim_fingerprint IS NULL OR claim_fingerprint = '')
            """,
            (save_id,),
        ).fetchall()
        for row in rows:
            self.repositories.connection.execute(
                """
                UPDATE memories
                SET claim_fingerprint = ?
                WHERE save_id = ? AND id = ?
                """,
                (
                    canonical_claim_fingerprint(row["body"]),
                    save_id,
                    row["id"],
                ),
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
        media_path_map: Mapping[tuple[str, str], str | None] | None = None,
        require_media_path_map: bool = False,
    ) -> None:
        self.source_save_id = source_save_id
        self.target_save_id = target_save_id
        self.id_maps = id_maps or {}
        self.media_path_map = dict(media_path_map or {})
        self.require_media_path_map = require_media_path_map
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
                remapped[column] = self._required_mapped_table_id(
                    target_table,
                    value,
                )
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
            elif table_name == "scene_facts" and column in {
                "subject_id",
                "target_id",
            }:
                type_column = (
                    "subject_type" if column == "subject_id" else "target_type"
                )
                remapped[column] = self._mapped_entity_id(
                    cast(str | None, row.get(type_column)),
                    value,
                )
            elif table_name == "scene_facts" and column == "conflict_key":
                continue
            elif table_name == "context_sources" and column == "source_id":
                remapped[column] = self._mapped_context_source_id(
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
            elif (
                table_name == "memories"
                and column == "source_observation_ids_json"
            ):
                remapped[column] = self._remap_json_id_list(
                    value,
                    "context_observations",
                )
            elif column in _JSON_COLUMNS_BY_TABLE.get(table_name, frozenset()):
                remapped[column] = self._remap_json_text(
                    table_name,
                    column,
                    value,
                    row=row,
                )
            else:
                remapped[column] = value
        if table_name == "scene_facts":
            remapped["conflict_key"] = scene_fact_conflict_key(
                fact_type=str(remapped.get("fact_type", "")),
                subject_type=str(remapped.get("subject_type", "")),
                subject_id=cast(str | None, remapped.get("subject_id")),
                subject_label=str(remapped.get("subject_label", "")),
                target_type=str(remapped.get("target_type", "")),
                target_id=cast(str | None, remapped.get("target_id")),
                target_label=str(remapped.get("target_label", "")),
                aspect=str(remapped.get("aspect", "")),
            )
        if table_name == "memories" and "epistemic_status" in remapped:
            actor_id = remapped.get("epistemic_actor_id")
            remapped["claim_fingerprint"] = _epistemic_claim_fingerprint(
                str(remapped.get("body", "") or ""),
                epistemic_status=str(remapped.get("epistemic_status", "")),
                epistemic_actor_id=(actor_id if isinstance(actor_id, str) else None),
                epistemic_actor_name=str(remapped.get("epistemic_actor_name", "")),
            )
        return remapped

    def _mapped_message_id(self, value: object) -> object:
        if isinstance(value, str):
            return self._required_mapped_table_id("messages", value)
        return value

    def _mapped_table_id(self, table_name: str, value: object) -> object:
        if not isinstance(value, str):
            return value
        return self.id_maps.get(table_name, {}).get(value, value)

    def _required_mapped_table_id(
        self,
        table_name: str,
        value: object,
    ) -> object:
        if not isinstance(value, str):
            return value
        mapped = self.id_maps.get(table_name, {}).get(value)
        if mapped is None:
            raise ValueError(
                f"Snapshot has unknown {table_name} reference: {value}"
            )
        return mapped

    def _mapped_entity_id(self, entity_type: str | None, value: object) -> object:
        if not isinstance(value, str) or entity_type is None:
            return value
        table_name = _ENTITY_TABLES.get(entity_type)
        if table_name is None:
            return value
        return self._required_mapped_table_id(table_name, value)

    def _mapped_context_source_id(
        self,
        source_type: str | None,
        value: object,
    ) -> object:
        if not isinstance(value, str) or source_type is None:
            return value
        if source_type in {"message", "messages"}:
            source_refs = [
                source_ref.strip()
                for source_ref in value.split(",")
                if source_ref.strip()
            ]
            return ",".join(
                str(self._mapped_source_ref(source_ref))
                for source_ref in source_refs
            )
        direct_tables = {
            "character_voice": "characters",
            "open_obligation": "active_threads",
            "character_text_thread": "character_text_threads",
        }
        if source_type in direct_tables:
            return self._required_mapped_table_id(
                direct_tables[source_type],
                value,
            )
        if source_type == "scenario_section":
            match = re.fullmatch(r"scenario:([^:]+):section:(.+)", value)
            if match is None:
                return value
            scenario_id = self._mapped_table_id("scenarios", match.group(1))
            return f"scenario:{scenario_id}:section:{match.group(2)}"
        if source_type == "world_state" and value.startswith("location:"):
            location_id = value.removeprefix("location:")
            return (
                "location:"
                f"{self._required_mapped_table_id('locations', location_id)}"
            )
        if source_type == "memory":
            mapped_memory_id = self._mapped_table_id("memories", value)
            if mapped_memory_id != value:
                return mapped_memory_id
            if value.startswith("character_profile:"):
                character_id = value.removeprefix("character_profile:")
                return (
                    "character_profile:"
                    f"{self._required_mapped_table_id('characters', character_id)}"
                )
            match = re.fullmatch(r"relationship:([^:]+):(.+)", value)
            if match is not None:
                character_id = str(
                    self._required_mapped_table_id(
                        "characters",
                        match.group(1),
                    )
                )
                return f"relationship:{character_id}:{match.group(2)}"
        return self._mapped_entity_id(source_type, value)

    def _mapped_media_path(
        self,
        row_id: object,
        column: str,
        value: object,
    ) -> object:
        if not isinstance(row_id, str) or not isinstance(value, str):
            return value
        key = (row_id, column)
        if key in self.media_path_map:
            return self.media_path_map[key]
        if self.require_media_path_map:
            if column == "thumbnail_path":
                return None
            raise ValueError(
                f"Snapshot media asset lacks verified imported path: {row_id}"
            )
        return value

    def _remap_json_text(
        self,
        table_name: str,
        column: str,
        value: object,
        *,
        row: Mapping[str, object],
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
        if table_name == "locations" and column == "connections_json":
            return _compact_json(self._remap_id_list(raw, "locations"))
        if (
            table_name == "world_state"
            and column == "value_json"
            and row.get("key") == "story.director_pressure"
        ):
            return _compact_json(self._remap_director_pressure_value(raw))
        if (
            table_name == "context_update_suggestions"
            and column == "proposed_value_json"
        ):
            return _compact_json(self._remap_suggestion_value(raw, row=row))
        if column == "source_message_ids_json":
            return _compact_json(self._remap_source_ref_list(raw))
        if table_name == "summaries" and column == "source_summary_ids_json":
            return _compact_json(self._remap_id_list(raw, "summaries"))
        if table_name == "context_sources" and column == "metadata_json":
            return _compact_json(
                self._remap_context_source_metadata(
                    raw,
                    source_type=cast(str | None, row.get("source_type")),
                    source_id=cast(str | None, row.get("source_id")),
                )
            )
        if table_name == "media_assets" and column == "metadata_json":
            return _compact_json(self._remap_known_metadata_ids(raw))
        if table_name == "turn_outcomes" and column == "payload_json":
            return _compact_json(self._remap_turn_outcome_payload(raw))
        return value

    def _remap_turn_outcome_payload(self, raw: object) -> object:
        if not isinstance(raw, dict):
            return raw
        remapped = dict(raw)
        remapped["save_id"] = self.target_save_id
        payload_message_id = raw.get("message_id")
        if isinstance(payload_message_id, str) and payload_message_id:
            remapped["message_id"] = self._mapped_table_id(
                "messages",
                payload_message_id,
            )

        def remap_refs(refs: object) -> object:
            if not isinstance(refs, list):
                return refs
            return [
                self._remapped_turn_outcome_source_ref(item)
                if isinstance(item, str)
                else item
                for item in refs
            ]

        remapped["source_message_ids"] = remap_refs(raw.get("source_message_ids"))
        remapped["attempt_evidence_source_ids"] = remap_refs(
            raw.get("attempt_evidence_source_ids")
        )
        raw_effects = raw.get("effects")
        if isinstance(raw_effects, list):
            remapped_effects: list[object] = []
            for item in raw_effects:
                if not isinstance(item, dict):
                    remapped_effects.append(item)
                    continue
                effect = dict(item)
                evidence = effect.get("evidence_source_ids")
                if isinstance(evidence, list):
                    effect["evidence_source_ids"] = [
                        self._remapped_turn_outcome_source_ref(ref)
                        if isinstance(ref, str)
                        else ref
                        for ref in evidence
                    ]
                remapped_effects.append(effect)
            remapped["effects"] = remapped_effects
        return remapped

    def _remapped_turn_outcome_source_ref(self, source_id: str) -> str:
        if source_id.startswith("message:"):
            message_id = source_id.removeprefix("message:")
            mapped = self._mapped_table_id("messages", message_id)
            if not isinstance(mapped, str) or mapped == message_id:
                return source_id
            return f"message:{mapped}"
        mapped = self._mapped_table_id("messages", source_id)
        if isinstance(mapped, str) and mapped != source_id:
            return mapped
        return source_id

    def _remap_json_id_list(self, value: object, table_name: str) -> object:
        if not isinstance(value, str):
            return value
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return value
        return _compact_json(self._remap_id_list(raw, table_name))

    def _remap_id_list(self, raw: object, table_name: str) -> object:
        if not isinstance(raw, list):
            return raw
        return [
            self._required_mapped_table_id(table_name, item)
            if isinstance(item, str)
            else item
            for item in raw
        ]

    def _remap_director_pressure_value(self, raw: object) -> object:
        if not isinstance(raw, dict):
            return raw
        remapped = dict(raw)
        history = raw.get("escalation_history")
        if not isinstance(history, list):
            return remapped
        remapped_history: list[object] = []
        for item in history:
            if not isinstance(item, dict):
                remapped_history.append(item)
                continue
            entry = dict(item)
            source_message_id = entry.get("source_message_id")
            if isinstance(source_message_id, str) and source_message_id:
                entry["source_message_id"] = self._required_mapped_table_id(
                    "messages",
                    source_message_id,
                )
            remapped_history.append(entry)
        remapped["escalation_history"] = remapped_history
        return remapped

    def _remap_suggestion_value(
        self,
        raw: object,
        *,
        row: Mapping[str, object],
    ) -> object:
        if isinstance(raw, str) and (
            row.get("entity_type"),
            row.get("field_path"),
        ) in {
            ("character", "location_id"),
            ("location", "parent_location_id"),
            ("scene_snapshot", "current_location_id"),
        }:
            return self._required_mapped_table_id("locations", raw)
        if not isinstance(raw, dict):
            return raw
        remapped = dict(raw)
        source_message_id = raw.get("source_message_id")
        if isinstance(source_message_id, str) and source_message_id:
            remapped["source_message_id"] = self._required_mapped_table_id(
                "messages",
                source_message_id,
            )
        source_message_ids = raw.get("source_message_ids")
        if isinstance(source_message_ids, list):
            remapped["source_message_ids"] = self._remap_source_ref_list(
                source_message_ids
            )
        for field in ("source_observation_id",):
            value = raw.get(field)
            if isinstance(value, str) and value:
                remapped[field] = self._required_mapped_table_id(
                    "context_observations",
                    value,
                )
        source_observation_ids = raw.get("source_observation_ids")
        if isinstance(source_observation_ids, list):
            remapped["source_observation_ids"] = self._remap_id_list(
                source_observation_ids,
                "context_observations",
            )
        location_id = raw.get("location_id")
        if isinstance(location_id, str) and location_id:
            remapped["location_id"] = self._required_mapped_table_id(
                "locations",
                location_id,
            )
        return remapped

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
                remapped.append(item)
                continue
            mapped_id = self._mapped_entity_id(entity_type, entity_id)
            if isinstance(mapped_id, str):
                remapped.append(f"{entity_type}:{mapped_id}")
        return remapped

    def _mapped_source_ref(self, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("Snapshot context source provenance is invalid")
        text_message_id = parse_character_text_source_ref(value)
        if text_message_id is not None:
            mapped = self.id_maps.get("character_text_messages", {}).get(
                text_message_id
            )
            if mapped is None:
                raise ValueError(
                    f"Snapshot context source has unknown provenance source: {value}"
                )
            return character_text_source_ref(mapped)
        mapped_message_id = self.id_maps["messages"].get(value)
        if mapped_message_id is None:
            raise ValueError(
                f"Snapshot context source has unknown provenance source: {value}"
            )
        return mapped_message_id

    def _remap_source_ref_list(self, raw: object) -> object:
        if not isinstance(raw, list):
            raise ValueError("Snapshot context source provenance is invalid")
        if (
            len(raw) > _MAX_SNAPSHOT_PROVENANCE_GROUP_MEMBERS
            or not all(isinstance(item, str) and item for item in raw)
        ):
            raise ValueError("Snapshot context source provenance is invalid")
        return [self._mapped_source_ref(item) for item in raw]

    def _remap_context_source_metadata(
        self,
        raw: object,
        *,
        source_type: str | None,
        source_id: str | None,
    ) -> object:
        if not isinstance(raw, dict):
            return raw
        remapped = dict(raw)
        if "source_provenance_mode" in raw:
            mode = raw["source_provenance_mode"]
            if mode not in {"all", "any"}:
                raise ValueError(
                    "Snapshot context source has invalid source_provenance_mode"
                )
            remapped["source_provenance_mode"] = mode
        for field in ("source_message_id", "last_seen_message_id"):
            if field in raw:
                remapped[field] = self._mapped_source_ref(raw[field])
        if "source_message_ids" in raw:
            remapped["source_message_ids"] = self._remap_source_ref_list(
                raw["source_message_ids"]
            )
        raw_groups = raw.get("source_provenance_groups")
        if raw_groups is not None:
            if not isinstance(raw_groups, list):
                raise ValueError("Snapshot context source provenance is invalid")
            if len(raw_groups) > _MAX_SNAPSHOT_PROVENANCE_GROUPS:
                raise ValueError("Snapshot context source provenance is too large")
            mapped_groups = [
                self._remap_source_ref_list(group)
                for group in raw_groups
            ]
            remapped["source_provenance_groups"] = mapped_groups
            grouped_source_ids = {
                source_id
                for group in mapped_groups
                if isinstance(group, list)
                for source_id in group
                if isinstance(source_id, str)
            }
            scalar_source_ids: set[str] = set()
            for field in ("source_message_id", "last_seen_message_id"):
                source_id = remapped.get(field)
                if isinstance(source_id, str):
                    scalar_source_ids.add(source_id)
            source_ids = remapped.get("source_message_ids")
            if isinstance(source_ids, list):
                scalar_source_ids.update(
                    source_id
                    for source_id in source_ids
                    if isinstance(source_id, str)
                )
            if not scalar_source_ids.issubset(grouped_source_ids):
                raise ValueError(
                    "Snapshot context source provenance groups omit sources"
                )
        audience_ids = raw.get("audience_character_ids")
        if isinstance(audience_ids, list):
            remapped["audience_character_ids"] = self._remap_id_list(
                audience_ids,
                "characters",
            )
        entity_ids = raw.get("entity_ids")
        if isinstance(entity_ids, list) and source_type is not None:
            remapped["entity_ids"] = [
                self._mapped_context_metadata_entity_id(
                    source_type,
                    source_id,
                    item,
                )
                if isinstance(item, str)
                else item
                for item in entity_ids
            ]
        thread_id = raw.get("thread_id")
        if isinstance(thread_id, str) and source_type == "character_text_thread":
            remapped["thread_id"] = self._required_mapped_table_id(
                "character_text_threads",
                thread_id,
            )
        observation_id = raw.get("observation_id")
        if isinstance(observation_id, str) and source_type == "observation":
            remapped["observation_id"] = self._mapped_table_id(
                "context_observations",
                observation_id,
            )
        scenario_id = raw.get("scenario_id")
        if isinstance(scenario_id, str):
            remapped["scenario_id"] = self._mapped_table_id(
                "scenarios",
                scenario_id,
            )
        return remapped

    def _mapped_context_metadata_entity_id(
        self,
        source_type: str,
        source_id: str | None,
        value: object,
    ) -> object:
        if source_type == "memory" and isinstance(source_id, str):
            if source_id.startswith(("character_profile:", "relationship:")):
                return self._required_mapped_table_id("characters", value)
        if (
            source_type == "world_state"
            and isinstance(source_id, str)
            and source_id.startswith("location:")
        ):
            return self._required_mapped_table_id("locations", value)
        direct_tables = {
            "character_voice": "characters",
            "open_obligation": "active_threads",
            "character_text_thread": "character_text_threads",
        }
        table_name = direct_tables.get(source_type)
        if table_name is not None:
            return self._required_mapped_table_id(table_name, value)
        return self._mapped_context_source_id(source_type, value)

    def _remap_known_metadata_ids(self, value: object) -> object:
        if isinstance(value, list):
            return [self._remap_known_metadata_ids(item) for item in value]
        if isinstance(value, dict):
            remapped: dict[str, object] = {}
            id_fields = {
                "character_id": "characters",
                "sender_character_id": "characters",
                "thread_id": "character_text_threads",
                "text_message_id": "character_text_messages",
                "source_message_id": "messages",
                "media_asset_id": "media_assets",
                "source_media_asset_id": "media_assets",
                "source_character_reference_asset_id": "media_assets",
                "source_character_reference_character_id": "characters",
            }
            id_list_fields = {
                "source_media_asset_ids": "media_assets",
                "source_character_reference_asset_ids": "media_assets",
                "source_character_reference_character_ids": "characters",
            }
            for key, item in value.items():
                table_name = id_fields.get(key)
                list_table_name = id_list_fields.get(key)
                if key == "request_source_message_id":
                    if isinstance(item, str):
                        mapped_message_id = self.id_maps.get("messages", {}).get(item)
                        mapped_text_message_id = self.id_maps.get(
                            "character_text_messages",
                            {},
                        ).get(item)
                        mapped = mapped_message_id or mapped_text_message_id
                        if mapped is None:
                            raise ValueError(
                                "Snapshot has unknown request source message "
                                f"reference: {item}"
                            )
                        remapped[key] = mapped
                    else:
                        remapped[key] = item
                elif table_name is not None:
                    remapped[key] = self._required_mapped_table_id(
                        table_name,
                        item,
                    )
                elif list_table_name is not None:
                    remapped[key] = self._remap_id_list(item, list_table_name)
                else:
                    remapped[key] = self._remap_known_metadata_ids(item)
            return remapped
        return value

    def _remap_trigger_key(self, value: object) -> object:
        if not isinstance(value, str):
            return value
        parts = value.split(":")
        if len(parts) < 2:
            return value
        if parts[0] == "ambient_random" and len(parts) >= 3:
            remapped = list(parts)
            mapped_message_id = self._mapped_table_id("messages", remapped[1])
            mapped_character_id = self._mapped_table_id("characters", remapped[2])
            if isinstance(mapped_message_id, str):
                remapped[1] = mapped_message_id
            if isinstance(mapped_character_id, str):
                remapped[2] = mapped_character_id
            return ":".join(remapped)
        table_name = {
            "active_thread": "active_threads",
            "dating_route": "dating_route_states",
            "character_intent": "characters",
            "memory": "memories",
            "memories": "memories",
        }.get(parts[0])
        if table_name is None:
            return value
        remapped = list(parts)
        mapped_entity_id = self._mapped_table_id(table_name, remapped[1])
        if isinstance(mapped_entity_id, str):
            remapped[1] = mapped_entity_id
        if parts[0] in {"active_thread", "character_intent"} and len(parts) >= 3:
            mapped_basis = self._mapped_table_id("messages", remapped[2])
            if isinstance(mapped_basis, str):
                remapped[2] = mapped_basis
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


def _declared_snapshot_row_references(
    table_name: str,
    row: Mapping[str, object],
) -> frozenset[tuple[str, str]]:
    references: set[tuple[str, str]] = set()
    for column in _MESSAGE_REFERENCE_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value:
            references.add(("messages", value))
    for column, target_table in _TABLE_REFERENCE_COLUMNS.get(table_name, {}).items():
        value = row.get(column)
        if isinstance(value, str) and value:
            references.add((target_table, value))

    key_targets = {
        "message_id": "messages",
        "source_message_id": "messages",
        "narrator_message_id": "messages",
        "character_id": "characters",
        "sender_character_id": "characters",
        "player_character_id": "characters",
        "location_id": "locations",
        "current_location_id": "locations",
        "connection": "locations",
        "present_character_id": "characters",
        "observation_id": "context_observations",
        "source_observation_id": "context_observations",
        "summary_id": "summaries",
        "source_summary_id": "summaries",
        "text_message_id": "character_text_messages",
        "source_text_message_id": "character_text_messages",
        "thread_id": "character_text_threads",
        "media_asset_id": "media_assets",
        "source_media_asset_id": "media_assets",
    }

    def collect(value: object, key: str | None = None) -> None:
        if isinstance(value, str):
            text_message_id = parse_character_text_source_ref(value)
            if text_message_id is not None:
                references.add(("character_text_messages", text_message_id))
            target_table = key_targets.get(key or "")
            if target_table is not None and value:
                references.add((target_table, value))
            entity_type, separator, entity_id = value.partition(":")
            entity_table = _ENTITY_TABLES.get(entity_type)
            if separator and entity_table is not None and entity_id:
                references.add((entity_table, entity_id))
            if value[:1] in {"[", "{"}:
                try:
                    collect(json.loads(value), key)
                except json.JSONDecodeError:
                    pass
            return
        if isinstance(value, Mapping):
            for nested_key, nested in value.items():
                collect(nested, str(nested_key))
            return
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            singular_key = key.removesuffix("s") if isinstance(key, str) else key
            for nested in value:
                collect(nested, singular_key)

    for column, value in row.items():
        collect(value, column.removesuffix("_json"))
    for id_column, type_column in (
        ("entity_id", "entity_type"),
        ("target_id", "target_type"),
        ("subject_id", "subject_type"),
        ("source_id", "source_type"),
    ):
        value = row.get(id_column)
        entity_type = row.get(type_column)
        typed_target_table = _ENTITY_TABLES.get(
            entity_type if isinstance(entity_type, str) else ""
        )
        if isinstance(value, str) and value and typed_target_table is not None:
            references.add((typed_target_table, value))
    if table_name == "context_sources":
        source_type = row.get("source_type")
        source_id = row.get("source_id")
        if isinstance(source_type, str) and isinstance(source_id, str):
            direct_table = {
                "character_voice": "characters",
                "open_obligation": "active_threads",
                "character_text_thread": "character_text_threads",
            }.get(source_type, _ENTITY_TABLES.get(source_type))
            if direct_table is not None and source_id:
                references.add((direct_table, source_id))
            if source_type == "world_state" and source_id.startswith("location:"):
                references.add(("locations", source_id.removeprefix("location:")))
            if source_type == "memory" and source_id.startswith(
                "character_profile:"
            ):
                references.add(
                    ("characters", source_id.removeprefix("character_profile:"))
                )
            relationship = re.fullmatch(r"relationship:([^:]+):(.+)", source_id)
            if source_type == "memory" and relationship is not None:
                references.add(("characters", relationship.group(1)))
    return frozenset(references)


def _snapshot_reference_candidates(
    table_name: str,
    row: Mapping[str, object],
) -> frozenset[str]:
    del table_name
    candidates: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, str):
            if not value:
                return
            candidates.add(value)
            text_message_id = parse_character_text_source_ref(value)
            if text_message_id is not None:
                candidates.add(text_message_id)
            for part in re.split(r"[:,]", value):
                normalized = part.strip()
                if normalized:
                    candidates.add(normalized)
            if value[:1] in {"[", "{"}:
                try:
                    collect(json.loads(value))
                except json.JSONDecodeError:
                    pass
            return
        if isinstance(value, Mapping):
            for nested in value.values():
                collect(nested)
            return
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            for nested in value:
                collect(nested)

    for value in row.values():
        collect(value)
    return frozenset(candidates)



def _filter_turn_outcomes_to_active_messages(
    rows: dict[str, tuple[dict[str, object], ...]],
) -> None:
    message_ids = _row_ids(rows.get("messages", ()))
    rows["turn_outcomes"] = tuple(
        row
        for row in rows.get("turn_outcomes", ())
        if row.get("message_id") is None
        or _optional_row_ref_active(row, "message_id", message_ids)
    )


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


def _snapshot_row_key(
    table: _SnapshotTable,
    row: Mapping[str, object],
) -> str:
    value = row.get(table.primary_key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Snapshot row is missing {table.name}.{table.primary_key}"
        )
    return value


def _snapshot_tree_order_key(
    table: _SnapshotTable,
    *,
    row: Mapping[str, object],
    row_key: str,
    ordinal: int,
) -> str:
    logical: tuple[str, ...]
    if table.name == "world_state":
        logical = (str(row.get("key", "")),)
    elif table.name == "context_sources":
        logical = (
            str(row.get("source_type", "")),
            str(row.get("source_id", "")),
        )
    elif table.name in {"scene_facts", "scene_fact_sources"}:
        logical = (str(row.get("created_at", "")),)
    else:
        logical = ()
    return _compact_json([*logical, f"{ordinal:020d}", row_key])


def _snapshot_tree_priority(table_name: str, row_key: str) -> int:
    digest = sha256(f"{table_name}\0{row_key}".encode()).hexdigest()
    return int(digest[:15], 16)


def _new_tree_mutation_guard() -> _TreeMutationGuard:
    return _TreeMutationGuard(
        seen=set(),
        byte_budget=[_MAX_SNAPSHOT_TOTAL_UNCOMPRESSED_BYTES],
        json_node_budget=[_MAX_SNAPSHOT_TOTAL_JSON_NODES],
        cache={},
    )


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _snapshot_message_ids_from_rows(
    rows_by_table: Mapping[str, Iterable[Mapping[str, object]]],
) -> tuple[str, ...]:
    return tuple(
        str(row["id"])
        for row in rows_by_table.get("messages", ())
        if isinstance(row.get("id"), str)
    )


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
    rows["context_observation_curation_state"] = tuple(
        portable_context_observation_curation_state_row(row)
        for row in rows.get("context_observation_curation_state", ())
    )
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
        _filter_snapshot_rows_for_unresolved_references(rows)
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
    _filter_snapshot_rows_for_unresolved_references(rows)
    _filter_context_observation_curation_snapshot_rows(rows)
    return rows


def _filter_snapshot_rows_for_unresolved_references(
    rows: dict[str, tuple[dict[str, object], ...]],
) -> None:
    total_rows = sum(len(table_rows) for table_rows in rows.values())
    for _ in range(max(total_rows + 1, 1)):
        active_ids = {
            table_name: _row_ids(rows.get(table_name, ()))
            for table_name in _SNAPSHOT_TABLE_NAMES
        }
        changed = False
        for table_name, table_rows in rows.items():
            if table_name == "messages":
                continue
            if table_name in _SNAPSHOT_MESSAGE_SCOPED_TABLES:
                kept = tuple(
                    row
                    for row in table_rows
                    if not _snapshot_row_has_unresolved_references(
                        table_name,
                        row,
                        active_ids,
                    )
                )
                if len(kept) != len(table_rows):
                    changed = True
                    rows[table_name] = kept
            elif table_name in _SNAPSHOT_KEEP_ENTITY_TABLES:
                if table_name == "dating_route_states":
                    kept = tuple(
                        row
                        for row in table_rows
                        if not _snapshot_row_has_unresolved_references(
                            table_name,
                            row,
                            active_ids,
                        )
                    )
                    if len(kept) != len(table_rows):
                        changed = True
                        rows[table_name] = kept
                    continue
                cleared_rows = tuple(
                    _clear_snapshot_row_unresolved_references(
                        table_name,
                        row,
                        active_ids,
                    )
                    for row in table_rows
                )
                if cleared_rows != table_rows:
                    changed = True
                    rows[table_name] = cleared_rows
        if not changed:
            return
    raise ValueError("Snapshot reference filtering did not converge")


_SNAPSHOT_MESSAGE_SCOPED_TABLES = frozenset(
    {
        "state_changes",
        "message_scene_presence",
        "message_visibility",
        "message_action_choices",
        "character_knowledge_edges",
        "entity_links",
        "context_update_suggestions",
        "context_update_audit",
        "context_observations",
        "summaries",
        "save_loss_outcomes",
        "context_sources",
        "scene_fact_sources",
        "narrator_phone_activity_cursors",
        "character_contact_states",
        "character_text_proactive_triggers",
    }
)

_SNAPSHOT_KEEP_ENTITY_TABLES = frozenset(
    {
        "characters",
        "locations",
        "active_threads",
        "scene_snapshots",
        "scene_facts",
        "world_state",
        "media_assets",
        "memories",
        "save_scenario_updates",
        "save_loss_conditions",
        "save_loss_condition_changes",
        "dating_route_states",
    }
)

_JSON_ENTITY_REFERENCE_FIELDS: dict[str, frozenset[str]] = {
    "scene_snapshots": frozenset({"present_character_ids_json"}),
    "locations": frozenset({"connections_json"}),
    "active_threads": frozenset({"related_entities_json"}),
    "memories": frozenset(
        {"source_message_ids_json", "source_observation_ids_json"}
    ),
    "summaries": frozenset(
        {"source_message_ids_json", "source_summary_ids_json"}
    ),
    "context_update_suggestions": frozenset(
        {"source_message_ids_json", "proposed_value_json"}
    ),
    "context_update_audit": frozenset({"source_message_ids_json"}),
    "context_observations": frozenset({"source_message_ids_json"}),
    "character_knowledge_edges": frozenset({"source_message_ids_json"}),
    "media_assets": frozenset({"metadata_json"}),
    "context_sources": frozenset({"metadata_json"}),
    "world_state": frozenset({"value_json"}),
    "save_scenario_updates": frozenset({"source_message_ids_json"}),
}


def _snapshot_row_has_unresolved_references(
    table_name: str,
    row: Mapping[str, object],
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    for column in _MESSAGE_REFERENCE_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value and value not in active_ids["messages"]:
            return True
    for column, target_table in _TABLE_REFERENCE_COLUMNS.get(table_name, {}).items():
        value = row.get(column)
        target_ids = active_ids.get(target_table)
        if target_ids is None:
            continue
        if (
            isinstance(value, str)
            and value
            and value not in target_ids
        ):
            return True
    if _snapshot_row_typed_entity_references_unresolved(
        table_name,
        row,
        active_ids,
    ):
        return True
    for column in _JSON_ENTITY_REFERENCE_FIELDS.get(table_name, frozenset()):
        raw = row.get(column)
        if isinstance(raw, str) and not _snapshot_json_reference_field_resolves(
            table_name,
            column,
            raw,
            row=row,
            active_ids=active_ids,
        ):
            return True
    return False


def _snapshot_row_typed_entity_references_unresolved(
    table_name: str,
    row: Mapping[str, object],
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    typed_columns: tuple[tuple[str, str], ...] = ()
    if table_name == "entity_links":
        typed_columns = (("entity_id", "entity_type"), ("target_id", "target_type"))
    elif table_name == "scene_facts":
        typed_columns = (("subject_id", "subject_type"), ("target_id", "target_type"))
    elif table_name in {"context_update_suggestions", "context_update_audit"}:
        typed_columns = (("entity_id", "entity_type"),)
    elif table_name == "character_knowledge_edges":
        typed_columns = (("target_id", "target_type"),)
    elif table_name == "character_text_proactive_triggers":
        typed_columns = (("source_id", "source_type"),)
    elif table_name == "context_sources":
        return _snapshot_context_source_references_unresolved(
            row,
            active_ids,
        )
    for id_column, type_column in typed_columns:
        entity_type = row.get(type_column)
        value = row.get(id_column)
        if not isinstance(value, str) or not value:
            continue
        target_table = _ENTITY_TABLES.get(
            entity_type if isinstance(entity_type, str) else ""
        )
        if target_table is None:
            continue
        target_ids = active_ids.get(target_table)
        if target_ids is None:
            continue
        if value not in target_ids:
            return True
    return False


def _snapshot_context_source_references_unresolved(
    row: Mapping[str, object],
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    source_type = row.get("source_type")
    source_id = row.get("source_id")
    if not isinstance(source_type, str) or not isinstance(source_id, str):
        return False
    if not source_id:
        return False
    if source_type in {"message", "messages"}:
        return any(
            ref and not _snapshot_source_ref_resolves(
                ref,
                message_ids=active_ids["messages"],
                text_message_ids=active_ids["character_text_messages"],
            )
            for ref in (part.strip() for part in source_id.split(","))
        )
    direct_tables = {
        "character_voice": "characters",
        "open_obligation": "active_threads",
        "character_text_thread": "character_text_threads",
    }
    if source_type in direct_tables:
        return source_id not in active_ids[direct_tables[source_type]]
    if source_type == "scenario_section":
        return False
    if source_type == "world_state" and source_id.startswith("location:"):
        return (
            source_id.removeprefix("location:")
            not in active_ids["locations"]
        )
    if source_type == "memory":
        if source_id in active_ids["memories"]:
            return False
        if source_id.startswith("character_profile:"):
            return (
                source_id.removeprefix("character_profile:")
                not in active_ids["characters"]
            )
        match = re.fullmatch(r"relationship:([^:]+):(.+)", source_id)
        if match is not None:
            return match.group(1) not in active_ids["characters"]
        return True
    target_table = _ENTITY_TABLES.get(source_type)
    if target_table is None:
        return False
    target_ids = active_ids.get(target_table)
    if target_ids is None:
        return False
    return source_id not in target_ids


def _snapshot_json_reference_field_resolves(
    table_name: str,
    column: str,
    raw: str,
    *,
    row: Mapping[str, object],
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    if column == "source_message_ids_json":
        return _snapshot_json_source_ref_list_resolves(
            raw,
            message_ids=active_ids["messages"],
            text_message_ids=active_ids["character_text_messages"],
        )
    if table_name == "world_state" and column == "value_json":
        if row.get("key") != "story.director_pressure":
            return True
        return _snapshot_director_pressure_value_resolves(
            raw,
            message_ids=active_ids["messages"],
            text_message_ids=active_ids["character_text_messages"],
        )
    if table_name == "context_sources" and column == "metadata_json":
        return _snapshot_context_source_metadata_resolves(
            raw,
            row=row,
            active_ids=active_ids,
        )
    if table_name == "context_update_suggestions" and column == "proposed_value_json":
        return _snapshot_suggestion_value_resolves(
            raw,
            row=row,
            active_ids=active_ids,
        )
    if table_name == "media_assets" and column == "metadata_json":
        return _snapshot_media_metadata_resolves(
            raw,
            active_ids=active_ids,
        )
    if column in {"present_character_ids_json", "connections_json"}:
        target_table = (
            "characters" if column == "present_character_ids_json" else "locations"
        )
        return _snapshot_json_id_list_resolves(raw, target_table, active_ids)
    if column == "related_entities_json":
        return _snapshot_related_entities_resolves(raw, active_ids)
    if column in {"source_observation_ids_json", "source_summary_ids_json"}:
        target_table = (
            "context_observations"
            if column == "source_observation_ids_json"
            else "summaries"
        )
        return _snapshot_json_id_list_resolves(raw, target_table, active_ids)
    return True


def _snapshot_json_id_list_resolves(
    raw: str,
    target_table: str,
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, list):
        return False
    target_ids = active_ids.get(target_table)
    if target_ids is None:
        return True
    return all(
        isinstance(item, str) and item in target_ids
        for item in parsed
    )


def _snapshot_related_entities_resolves(
    raw: str,
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, list):
        return False
    for item in parsed:
        if not isinstance(item, str):
            return False
        entity_type, separator, entity_id = item.partition(":")
        if not separator:
            continue
        target_table = _ENTITY_TABLES.get(entity_type)
        if target_table is None:
            continue
        target_ids = active_ids.get(target_table)
        if target_ids is None:
            continue
        if entity_id not in target_ids:
            return False
    return True


def _snapshot_json_source_ref_list_resolves(
    raw: str,
    *,
    message_ids: frozenset[str],
    text_message_ids: frozenset[str],
) -> bool:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, list):
        return False
    for item in parsed:
        if not isinstance(item, str) or not item:
            return False
        if not _snapshot_source_ref_resolves(
            item,
            message_ids=message_ids,
            text_message_ids=text_message_ids,
        ):
            return False
    return True


def _snapshot_source_ref_resolves(
    source_ref: str,
    *,
    message_ids: frozenset[str],
    text_message_ids: frozenset[str],
) -> bool:
    text_message_id = parse_character_text_source_ref(source_ref)
    if text_message_id is not None:
        return text_message_id in text_message_ids
    return source_ref in message_ids


def _snapshot_context_source_metadata_resolves(
    raw: str,
    *,
    row: Mapping[str, object],
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return True
    if not isinstance(parsed, dict):
        return True
    message_ids = active_ids["messages"]
    text_message_ids = active_ids["character_text_messages"]
    for field in ("source_message_id", "last_seen_message_id"):
        value = parsed.get(field)
        if isinstance(value, str) and value and not _snapshot_source_ref_resolves(
            value,
            message_ids=message_ids,
            text_message_ids=text_message_ids,
        ):
            return False
    if "source_message_ids" in parsed and not _snapshot_json_source_ref_list_resolves(
        json.dumps(parsed["source_message_ids"]),
        message_ids=message_ids,
        text_message_ids=text_message_ids,
    ):
        return False
    raw_groups = parsed.get("source_provenance_groups")
    if raw_groups is not None:
        if not isinstance(raw_groups, list):
            return False
        for group in raw_groups:
            if not _snapshot_json_source_ref_list_resolves(
                json.dumps(group),
                message_ids=message_ids,
                text_message_ids=text_message_ids,
            ):
                return False
    if "audience_character_ids" in parsed and not _snapshot_json_id_list_resolves(
        json.dumps(parsed["audience_character_ids"]),
        "characters",
        active_ids,
    ):
        return False
    entity_ids = parsed.get("entity_ids")
    if isinstance(entity_ids, list):
        source_type = row.get("source_type")
        source_id = row.get("source_id")
        for item in entity_ids:
            if not isinstance(item, str) or not item:
                return False
            if not _snapshot_context_metadata_entity_id_resolves(
                source_type,
                source_id,
                item,
                active_ids,
            ):
                return False
    thread_id = parsed.get("thread_id")
    if (
        isinstance(thread_id, str)
        and row.get("source_type") == "character_text_thread"
        and thread_id not in active_ids["character_text_threads"]
    ):
        return False
    return True


def _snapshot_context_metadata_entity_id_resolves(
    source_type: object,
    source_id: object,
    value: str,
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    if source_type == "memory" and isinstance(source_id, str):
        if source_id.startswith(("character_profile:", "relationship:")):
            return value in active_ids["characters"]
        return value in active_ids["memories"]
    if (
        source_type == "world_state"
        and isinstance(source_id, str)
        and source_id.startswith("location:")
    ):
        return value in active_ids["locations"]
    direct_tables = {
        "character_voice": "characters",
        "open_obligation": "active_threads",
        "character_text_thread": "character_text_threads",
    }
    table_name = (
        direct_tables.get(source_type)
        if isinstance(source_type, str)
        else None
    )
    if table_name is not None:
        return value in active_ids[table_name]
    return _snapshot_context_source_id_resolves_typed(
        source_type,
        source_id,
        value,
        active_ids,
    )


def _snapshot_context_source_id_resolves_typed(
    source_type: object,
    source_id: object,
    value: str,
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    if source_type in {"message", "messages"}:
        return value in active_ids["messages"]
    if source_type == "memory" and isinstance(source_id, str):
        if source_id.startswith(("character_profile:", "relationship:")):
            return value in active_ids["characters"]
        return value in active_ids["memories"]
    if source_type == "scenario_section":
        return True
    if (
        source_type == "world_state"
        and isinstance(source_id, str)
        and source_id.startswith("location:")
    ):
        return value in active_ids["locations"]
    if not isinstance(source_type, str):
        return True
    target_table = _ENTITY_TABLES.get(source_type)
    if target_table is None:
        return True
    target_ids = active_ids.get(target_table)
    if target_ids is None:
        return True
    return value in target_ids


def _snapshot_suggestion_value_resolves(
    raw: str,
    *,
    row: Mapping[str, object],
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return True
    if (
        isinstance(parsed, str)
        and (
            row.get("entity_type"),
            row.get("field_path"),
        )
        in {
            ("character", "location_id"),
            ("location", "parent_location_id"),
            ("scene_snapshot", "current_location_id"),
        }
    ):
        return parsed in active_ids["locations"]
    if not isinstance(parsed, dict):
        return True
    source_message_id = parsed.get("source_message_id")
    if isinstance(source_message_id, str) and source_message_id and not (
        _snapshot_source_ref_resolves(
            source_message_id,
            message_ids=active_ids["messages"],
            text_message_ids=active_ids["character_text_messages"],
        )
    ):
        return False
    raw_ids = parsed.get("source_message_ids")
    if raw_ids is not None and not _snapshot_json_source_ref_list_resolves(
        json.dumps(raw_ids),
        message_ids=active_ids["messages"],
        text_message_ids=active_ids["character_text_messages"],
    ):
        return False
    source_observation_id = parsed.get("source_observation_id")
    if (
        isinstance(source_observation_id, str)
        and source_observation_id
        and source_observation_id not in active_ids["context_observations"]
    ):
        return False
    raw_observation_ids = parsed.get("source_observation_ids")
    if raw_observation_ids is not None and not _snapshot_json_id_list_resolves(
        json.dumps(raw_observation_ids),
        "context_observations",
        active_ids,
    ):
        return False
    location_id = parsed.get("location_id")
    if (
        isinstance(location_id, str)
        and location_id
        and location_id not in active_ids["locations"]
    ):
        return False
    return True


def _snapshot_media_metadata_resolves(
    raw: str,
    active_ids: Mapping[str, frozenset[str]],
) -> bool:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return True
    if isinstance(parsed, list):
        return all(
            _snapshot_media_metadata_resolves(
                json.dumps(item),
                active_ids,
            )
            for item in parsed
            if isinstance(item, dict)
        )
    if not isinstance(parsed, dict):
        return True
    id_fields = {
        "character_id": "characters",
        "sender_character_id": "characters",
        "thread_id": "character_text_threads",
        "text_message_id": "character_text_messages",
        "source_message_id": "messages",
        "media_asset_id": "media_assets",
        "source_media_asset_id": "media_assets",
        "source_character_reference_asset_id": "media_assets",
        "source_character_reference_character_id": "characters",
    }
    id_list_fields = {
        "source_media_asset_ids": "media_assets",
        "source_character_reference_asset_ids": "media_assets",
        "source_character_reference_character_ids": "characters",
    }
    for key, item in parsed.items():
        target_table = id_fields.get(key)
        if key == "request_source_message_id":
            if (
                isinstance(item, str)
                and item
                and item not in active_ids["messages"]
                and item not in active_ids["character_text_messages"]
            ):
                return False
            continue
        if target_table is not None:
            if isinstance(item, str) and item and item not in active_ids[target_table]:
                return False
            continue
        list_target_table = id_list_fields.get(key)
        if list_target_table is not None and isinstance(item, list):
            if not _snapshot_json_id_list_resolves(
                json.dumps(item),
                list_target_table,
                active_ids,
            ):
                return False
            continue
        if isinstance(item, list):
            for list_item in item:
                if isinstance(list_item, dict) and not (
                    _snapshot_media_metadata_resolves(
                        json.dumps(list_item),
                        active_ids,
                    )
                ):
                    return False
            continue
        if isinstance(item, dict):
            if not _snapshot_media_metadata_resolves(
                json.dumps(item),
                active_ids,
            ):
                return False
    return True


def _snapshot_director_pressure_value_resolves(
    raw: str,
    *,
    message_ids: frozenset[str],
    text_message_ids: frozenset[str],
) -> bool:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return True
    if not isinstance(parsed, dict):
        return True
    history = parsed.get("escalation_history")
    if not isinstance(history, list):
        return True
    for item in history:
        if not isinstance(item, dict):
            continue
        source_message_id = item.get("source_message_id")
        if not isinstance(source_message_id, str) or not source_message_id:
            continue
        if source_message_id not in message_ids:
            return False
    return True


def _clear_snapshot_row_unresolved_references(
    table_name: str,
    row: Mapping[str, object],
    active_ids: Mapping[str, frozenset[str]],
) -> dict[str, object]:
    cleared = dict(row)
    for column in _MESSAGE_REFERENCE_COLUMNS:
        value = cleared.get(column)
        if isinstance(value, str) and value and value not in active_ids["messages"]:
            cleared[column] = None
    for column, target_table in _TABLE_REFERENCE_COLUMNS.get(table_name, {}).items():
        value = cleared.get(column)
        target_ids = active_ids.get(target_table)
        if target_ids is None:
            continue
        if (
            isinstance(value, str)
            and value
            and value not in target_ids
        ):
            cleared[column] = None
    for column in _JSON_ENTITY_REFERENCE_FIELDS.get(table_name, frozenset()):
        raw = cleared.get(column)
        if not isinstance(raw, str):
            continue
        if table_name == "world_state" and column == "value_json":
            if cleared.get("key") == "story.director_pressure":
                cleared[column] = _prune_snapshot_director_pressure_value(
                    raw,
                    message_ids=active_ids["messages"],
                    text_message_ids=active_ids["character_text_messages"],
                )
            continue
        if column == "source_message_ids_json":
            cleared[column] = _prune_snapshot_json_source_refs(
                raw,
                message_ids=active_ids["messages"],
                text_message_ids=active_ids["character_text_messages"],
            )
            continue
        if table_name == "media_assets" and column == "metadata_json":
            cleared[column] = _prune_snapshot_media_metadata(
                raw,
                active_ids,
            )
            continue
        if column in {"present_character_ids_json", "connections_json"}:
            target_table = (
                "characters"
                if column == "present_character_ids_json"
                else "locations"
            )
            cleared[column] = _prune_snapshot_json_id_list(
                raw,
                target_table,
                active_ids,
            )
            continue
        if column == "related_entities_json":
            cleared[column] = _prune_snapshot_related_entities(
                raw,
                active_ids,
            )
            continue
        if column in {"source_observation_ids_json", "source_summary_ids_json"}:
            target_table = (
                "context_observations"
                if column == "source_observation_ids_json"
                else "summaries"
            )
            cleared[column] = _prune_snapshot_json_id_list(
                raw,
                target_table,
                active_ids,
            )
    return cleared


def _prune_snapshot_json_source_refs(
    raw: str,
    *,
    message_ids: frozenset[str],
    text_message_ids: frozenset[str],
) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, list):
        return raw
    pruned = [
        item
        for item in parsed
        if isinstance(item, str)
        and item
        and _snapshot_source_ref_resolves(
            item,
            message_ids=message_ids,
            text_message_ids=text_message_ids,
        )
    ]
    return _compact_json(pruned)


def _prune_snapshot_json_id_list(
    raw: str,
    target_table: str,
    active_ids: Mapping[str, frozenset[str]],
) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, list):
        return raw
    target_ids = active_ids.get(target_table)
    if target_ids is None:
        return raw
    pruned = [
        item
        for item in parsed
        if isinstance(item, str) and item in target_ids
    ]
    return _compact_json(pruned)


def _prune_snapshot_related_entities(
    raw: str,
    active_ids: Mapping[str, frozenset[str]],
) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, list):
        return raw
    pruned: list[object] = []
    for item in parsed:
        if not isinstance(item, str):
            pruned.append(item)
            continue
        entity_type, separator, entity_id = item.partition(":")
        if not separator:
            pruned.append(item)
            continue
        target_table = _ENTITY_TABLES.get(entity_type)
        if target_table is None:
            pruned.append(item)
            continue
        target_ids = active_ids.get(target_table)
        if target_ids is None or entity_id in target_ids:
            pruned.append(item)
    return _compact_json(pruned)


def _prune_snapshot_media_metadata(
    raw: str,
    active_ids: Mapping[str, frozenset[str]],
) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, dict):
        return raw
    id_fields = {
        "character_id": "characters",
        "sender_character_id": "characters",
        "thread_id": "character_text_threads",
        "text_message_id": "character_text_messages",
        "source_message_id": "messages",
        "media_asset_id": "media_assets",
        "source_media_asset_id": "media_assets",
        "source_character_reference_asset_id": "media_assets",
        "source_character_reference_character_id": "characters",
    }
    id_list_fields = {
        "source_media_asset_ids": "media_assets",
        "source_character_reference_asset_ids": "media_assets",
        "source_character_reference_character_ids": "characters",
    }
    pruned = dict(parsed)
    for key, item in parsed.items():
        if key == "request_source_message_id":
            if (
                isinstance(item, str)
                and item
                and item not in active_ids["messages"]
                and item not in active_ids["character_text_messages"]
            ):
                pruned[key] = None
            continue
        target_table = id_fields.get(key)
        if target_table is not None:
            if (
                isinstance(item, str)
                and item
                and item not in active_ids[target_table]
            ):
                pruned[key] = None
            continue
        list_target_table = id_list_fields.get(key)
        if list_target_table is not None and isinstance(item, list):
            list_target_ids = active_ids[list_target_table]
            kept_items = [
                list_item
                for list_item in item
                if isinstance(list_item, str) and list_item in list_target_ids
            ]
            if kept_items != item:
                pruned[key] = kept_items
            continue
        if isinstance(item, list):
            pruned_items = [
                json.loads(
                    _prune_snapshot_media_metadata(
                        json.dumps(list_item),
                        active_ids,
                    )
                )
                if isinstance(list_item, dict)
                else list_item
                for list_item in item
            ]
            if pruned_items != item:
                pruned[key] = pruned_items
            continue
        if isinstance(item, dict):
            pruned[key] = json.loads(
                _prune_snapshot_media_metadata(
                    json.dumps(item),
                    active_ids,
                )
            )
    return _compact_json(pruned)


def _prune_snapshot_director_pressure_value(
    raw: str,
    *,
    message_ids: frozenset[str],
    text_message_ids: frozenset[str],
) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, dict):
        return raw
    history = parsed.get("escalation_history")
    if not isinstance(history, list):
        return raw
    pruned_history = [
        item
        for item in history
        if not (
            isinstance(item, dict)
            and isinstance(item.get("source_message_id"), str)
            and item["source_message_id"]
            and item["source_message_id"] not in message_ids
        )
    ]
    pruned = dict(parsed)
    pruned["escalation_history"] = pruned_history
    return _compact_json(pruned)


def portable_context_observation_curation_state_row(
    row: Mapping[str, object],
) -> dict[str, object]:
    copied = dict(row)
    lease_token = copied.get("lease_token")
    lease_until = copied.get("lease_until")
    attempt_count = copied.get("attempt_count")
    if (
        isinstance(lease_token, str)
        and lease_token
        and _timestamp_is_in_future(lease_until)
        and isinstance(attempt_count, int)
        and not isinstance(attempt_count, bool)
        and attempt_count > 0
    ):
        copied["attempt_count"] = attempt_count - 1
    copied["lease_token"] = None
    copied["lease_until"] = None
    return copied


def _snapshot_row_recheck_at(
    table_name: str,
    row: Mapping[str, object],
) -> str | None:
    if table_name != "context_observation_curation_state":
        return None
    lease_token = row.get("lease_token")
    lease_until = row.get("lease_until")
    if (
        isinstance(lease_token, str)
        and lease_token
        and isinstance(lease_until, str)
        and _timestamp_is_in_future(lease_until)
    ):
        return lease_until
    return None


def _timestamp_is_in_future(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC) > datetime.now(UTC)


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
    kind, value, _node_count = _decode_exported_snapshot_object_with_node_count(
        row,
    )
    return kind, value


def _decode_exported_snapshot_object_with_node_count(
    row: Mapping[str, object],
    *,
    max_json_nodes: int = _MAX_SNAPSHOT_OBJECT_JSON_NODES,
) -> tuple[str, object, int]:
    object_hash = _text(row, "object_hash")
    kind = _text(row, "kind")
    encoding = _text(row, "encoding")
    if encoding != SNAPSHOT_ENCODING:
        raise ValueError(f"Unsupported snapshot object encoding: {encoding}")
    declared_size = _int(row, "uncompressed_size")
    if (
        declared_size < 0
        or declared_size > _MAX_SNAPSHOT_OBJECT_UNCOMPRESSED_BYTES
    ):
        raise ValueError(f"Snapshot object is too large: {object_hash}")
    try:
        compressed = base64.b64decode(_text(row, "payload_base64"))
        decompressor = zlib.decompressobj()
        payload = decompressor.decompress(compressed, declared_size + 1)
        if (
            len(payload) > declared_size
            or decompressor.unconsumed_tail
            or not decompressor.eof
        ):
            raise ValueError(f"Snapshot object size mismatch: {object_hash}")
        payload += decompressor.flush()
        node_count = validate_json_structure(
            payload,
            max_nodes=min(_MAX_SNAPSHOT_OBJECT_JSON_NODES, max_json_nodes),
            max_depth=_MAX_SNAPSHOT_OBJECT_JSON_DEPTH,
        )
        value = json.loads(payload.decode("utf-8"))
    except (
        binascii.Error,
        UnicodeDecodeError,
        json.JSONDecodeError,
        JsonSafetyError,
        zlib.error,
    ) as exc:
        raise ValueError(f"Invalid snapshot object payload: {object_hash}") from exc
    if len(payload) != declared_size:
        raise ValueError(f"Snapshot object size mismatch: {object_hash}")
    actual_hash = _snapshot_object_hash(kind=kind, payload=payload)
    if actual_hash != object_hash:
        raise ValueError(f"Snapshot object hash mismatch: {object_hash}")
    return kind, value, node_count


def _validate_exported_snapshot_rows(
    snapshot_rows: Iterable[Mapping[str, object]],
    object_rows: Iterable[Mapping[str, object]],
) -> dict[str, tuple[str, object]]:
    snapshots = [dict(row) for row in snapshot_rows]
    if len(snapshots) > _MAX_IMPORTED_SNAPSHOT_COUNT:
        raise ValueError("Snapshot bundle contains too many snapshots")
    seen_snapshot_ids: set[str] = set()
    for row in snapshots:
        snapshot_id = _text(row, "id")
        if snapshot_id in seen_snapshot_ids:
            raise ValueError(f"Duplicate snapshot id in bundle: {snapshot_id}")
        seen_snapshot_ids.add(snapshot_id)
        _text(row, "root_manifest_hash")

    raw_objects_by_hash: dict[str, dict[str, object]] = {}
    objects_by_hash: dict[str, tuple[str, object]] = {}
    json_node_budget = [_MAX_SNAPSHOT_TOTAL_JSON_NODES]
    validated_nested_json: set[str] = set()
    object_sizes_by_hash: dict[str, int] = {}
    total_uncompressed_size = 0
    for object_row in object_rows:
        object_hash = _text(object_row, "object_hash")
        if object_hash in raw_objects_by_hash:
            raise ValueError(f"Duplicate snapshot object in bundle: {object_hash}")
        declared_size = _int(object_row, "uncompressed_size")
        if declared_size < 0:
            raise ValueError(f"Snapshot object size is invalid: {object_hash}")
        total_uncompressed_size += declared_size
        if total_uncompressed_size > _MAX_SNAPSHOT_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("Snapshot objects are too large")
        raw_objects_by_hash[object_hash] = dict(object_row)
        object_sizes_by_hash[object_hash] = declared_size

    total_referenced_bytes = 0
    validated_table_signatures: set[str] = set()
    referenced_object_hashes: set[str] = set()
    unique_referenced_row_hashes: set[str] = set()
    root_hashes = {
        _text(row, "root_manifest_hash") for row in snapshots
    }
    for root_hash in root_hashes:
        referenced_object_hashes.add(root_hash)
        manifest = _required_decoded_exported_snapshot_object(
            raw_objects_by_hash,
            objects_by_hash,
            root_hash,
            expected_kind="snapshot_manifest",
            json_node_budget=json_node_budget,
        )
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("format") != SNAPSHOT_FORMAT
        ):
            raise ValueError(f"Invalid snapshot manifest object: {root_hash}")
        raw_tables = manifest.get("tables")
        if not isinstance(raw_tables, Mapping):
            raise ValueError(f"Snapshot manifest is missing table entries: {root_hash}")
        table_signature = _snapshot_manifest_table_signature(manifest)
        if table_signature in validated_table_signatures:
            continue
        validated_table_signatures.add(table_signature)
        manifest_entry_count = 0
        seen_row_object_hashes: set[str] = set()
        for table_name, entries in raw_tables.items():
            if table_name not in _TABLES_BY_NAME:
                raise ValueError(f"Unknown snapshot table in manifest: {table_name}")
            if not isinstance(entries, list):
                raise ValueError(f"Snapshot table entries must be a list: {table_name}")
            manifest_entry_count += len(entries)
            if manifest_entry_count > _MAX_SNAPSHOT_MANIFEST_ENTRIES:
                raise ValueError("Snapshot manifest contains too many entries")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError(f"Snapshot table entry is invalid: {table_name}")
                object_hash = _text(entry, "object_hash")
                if object_hash in seen_row_object_hashes:
                    raise ValueError(
                        f"Duplicate snapshot row object reference: {object_hash}"
                    )
                seen_row_object_hashes.add(object_hash)
                referenced_object_hashes.add(object_hash)
                if object_hash not in unique_referenced_row_hashes:
                    unique_referenced_row_hashes.add(object_hash)
                    if (
                        len(unique_referenced_row_hashes)
                        > _MAX_SNAPSHOT_UNIQUE_ROW_OBJECTS
                    ):
                        raise ValueError(
                            "Snapshot manifests contain too many unique rows"
                        )
                    total_referenced_bytes += object_sizes_by_hash.get(
                        object_hash,
                        0,
                    )
                    if (
                        total_referenced_bytes
                        > _MAX_SNAPSHOT_IMPORT_REFERENCED_BYTES
                    ):
                        raise ValueError(
                            "Snapshot manifests reference too much row data"
                        )
                value = _required_decoded_exported_snapshot_object(
                    raw_objects_by_hash,
                    objects_by_hash,
                    object_hash,
                    expected_kind=f"row:{table_name}",
                    json_node_budget=json_node_budget,
                )
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"Snapshot row object is not a row: {object_hash}"
                    )
                _validate_snapshot_nested_json_columns(
                    table_name,
                    value,
                    json_node_budget=json_node_budget,
                    validated_nested_json=validated_nested_json,
                    object_hash=object_hash,
                )
    unreferenced_hashes = set(raw_objects_by_hash) - referenced_object_hashes
    if unreferenced_hashes:
        raise ValueError("Snapshot bundle contains unreferenced objects")
    return objects_by_hash


def _snapshot_manifest_table_signature(manifest: Mapping[str, object]) -> str:
    raw_tables = manifest.get("tables", {})
    effective_tables: dict[str, object] = {}
    if isinstance(raw_tables, Mapping):
        for table_name, entries in raw_tables.items():
            if isinstance(entries, list):
                effective_tables[str(table_name)] = [
                    (
                        entry.get("object_hash")
                        if isinstance(entry, Mapping)
                        else entry
                    )
                    for entry in entries
                ]
            else:
                effective_tables[str(table_name)] = entries
    else:
        effective_tables[""] = raw_tables
    return sha256(
        _canonical_json_bytes(effective_tables)
    ).hexdigest()


def _required_decoded_exported_snapshot_object(
    raw_objects_by_hash: Mapping[str, Mapping[str, object]],
    decoded_objects_by_hash: dict[str, tuple[str, object]],
    object_hash: str,
    *,
    expected_kind: str,
    json_node_budget: list[int] | None = None,
) -> object:
    if object_hash not in decoded_objects_by_hash:
        try:
            raw_row = raw_objects_by_hash[object_hash]
        except KeyError as exc:
            raise ValueError(f"Missing snapshot object: {object_hash}") from exc
        remaining_nodes = _MAX_SNAPSHOT_OBJECT_JSON_NODES
        if json_node_budget is not None:
            remaining_nodes = json_node_budget[0]
            if remaining_nodes <= 0:
                raise ValueError("Snapshot JSON contains too many values")
        kind, value, node_count = _decode_exported_snapshot_object_with_node_count(
            raw_row,
            max_json_nodes=remaining_nodes,
        )
        decoded_objects_by_hash[object_hash] = (kind, value)
        if json_node_budget is not None:
            json_node_budget[0] -= node_count
    return _required_exported_snapshot_object(
        decoded_objects_by_hash,
        object_hash,
        expected_kind=expected_kind,
    )


def _validate_snapshot_nested_json_columns(
    table_name: str,
    value: Mapping[str, object],
    *,
    json_node_budget: list[int],
    validated_nested_json: set[str],
    object_hash: str,
) -> None:
    for column in _JSON_COLUMNS_BY_TABLE.get(table_name, frozenset()):
        raw_json = value.get(column)
        if not isinstance(raw_json, str):
            continue
        budget_key = f"nested:{object_hash}:{column}"
        if budget_key in validated_nested_json:
            continue
        remaining_nodes = json_node_budget[0]
        if remaining_nodes <= 0:
            raise ValueError("Snapshot JSON contains too many values")
        try:
            node_count = validate_json_structure(
                raw_json.encode("utf-8"),
                max_nodes=remaining_nodes,
                max_depth=_MAX_SNAPSHOT_OBJECT_JSON_DEPTH,
            )
        except JsonSafetyError as exc:
            raise ValueError(
                f"Invalid snapshot nested JSON: {table_name}.{column}"
            ) from exc
        json_node_budget[0] -= node_count
        validated_nested_json.add(budget_key)


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


def _coalesce_remapped_snapshot_rows(
    table_name: str,
    rows: list[dict[str, object]],
    *,
    reject_context_conflicts: bool = True,
) -> list[dict[str, object]]:
    coalesced: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        if table_name == "character_knowledge_edges":
            row["target_type"] = normalized_knowledge_target_type(
                str(row.get("target_type", ""))
            )
            row["source_message_ids_json"] = (
                _merged_snapshot_json_string_lists(
                    row.get("source_message_ids_json"),
                    "[]",
                    limit=_MAX_SNAPSHOT_PROVENANCE_GROUP_MEMBERS,
                    extra_values=(row.get("source_message_id"),),
                )
            )
        key = _snapshot_row_unique_key(table_name, row)
        existing = coalesced.get(key)
        if existing is None:
            coalesced[key] = row
            continue
        if table_name == "memories":
            _merge_snapshot_memory_rows(existing, row)
        elif table_name == "context_sources":
            _merge_snapshot_context_source_rows(
                existing,
                row,
                reject_conflict=reject_context_conflicts,
            )
        elif table_name == "character_knowledge_edges":
            _merge_snapshot_knowledge_edge_rows(existing, row)
        elif table_name == "entity_links":
            _merge_snapshot_entity_link_rows(existing, row)
        elif table_name == "character_text_proactive_triggers":
            _merge_snapshot_proactive_trigger_rows(existing, row)
        elif (
            existing.get("archived_at") is not None
            and row.get("archived_at") is None
        ):
            coalesced[key] = row
    return list(coalesced.values())


def _normalize_legacy_snapshot_memories(
    rows_by_table: Mapping[str, Iterable[Mapping[str, object]]],
) -> dict[str, tuple[dict[str, object], ...]]:
    rows = {
        table_name: [dict(row) for row in table_rows]
        for table_name, table_rows in rows_by_table.items()
    }
    keepers_by_fingerprint: dict[str, dict[str, object]] = {}
    memory_id_map: dict[str, str] = {}
    normalized_memories: list[dict[str, object]] = []
    for row in rows.get("memories", []):
        if "epistemic_status" in row:
            actor_id = row.get("epistemic_actor_id")
            fingerprint = _epistemic_claim_fingerprint(
                str(row.get("body", "") or ""),
                epistemic_status=str(row.get("epistemic_status", "")),
                epistemic_actor_id=(actor_id if isinstance(actor_id, str) else None),
                epistemic_actor_name=str(row.get("epistemic_actor_name", "")),
            )
        else:
            fingerprint = canonical_claim_fingerprint(row.get("body", ""))
        if row.get("archived_at") is not None or not fingerprint:
            normalized_memories.append(row)
            continue
        row["claim_fingerprint"] = fingerprint
        keeper = keepers_by_fingerprint.get(fingerprint)
        if keeper is None:
            keepers_by_fingerprint[fingerprint] = row
            normalized_memories.append(row)
            continue
        duplicate_id = row.get("id")
        keeper_id = keeper.get("id")
        if isinstance(duplicate_id, str) and isinstance(keeper_id, str):
            memory_id_map[duplicate_id] = keeper_id
        _merge_snapshot_memory_rows(keeper, row)
    rows["memories"] = normalized_memories

    if memory_id_map:
        for table_name, table_rows in rows.items():
            for row in table_rows:
                _remap_snapshot_memory_reference(
                    table_name=table_name,
                    row=row,
                    memory_id_map=memory_id_map,
                )
            rows[table_name] = _coalesce_remapped_snapshot_rows(
                table_name,
                table_rows,
                reject_context_conflicts=False,
            )
    return {
        table_name: tuple(table_rows)
        for table_name, table_rows in rows.items()
    }


def _normalize_legacy_snapshot_scene_facts(
    rows_by_table: Mapping[str, Iterable[Mapping[str, object]]],
) -> dict[str, tuple[dict[str, object], ...]]:
    rows = {
        table_name: [dict(row) for row in table_rows]
        for table_name, table_rows in rows_by_table.items()
    }
    facts = rows.get("scene_facts", [])
    active_keys: set[tuple[object, ...]] = set()
    duplicate_ids: set[str] = set()
    normalized_reversed: list[dict[str, object]] = []
    for row in reversed(facts):
        canonical_key = scene_fact_conflict_key(
            fact_type=str(row.get("fact_type", "")),
            subject_type=str(row.get("subject_type", "")),
            subject_id=_optional_snapshot_string(row.get("subject_id")),
            subject_label=str(row.get("subject_label", "")),
            target_type=str(row.get("target_type", "")),
            target_id=_optional_snapshot_string(row.get("target_id")),
            target_label=str(row.get("target_label", "")),
            aspect=str(row.get("aspect", "")),
        )
        row["conflict_key"] = canonical_key
        if row.get("archived_at") is not None:
            normalized_reversed.append(row)
            continue
        fact_unique_key = (
            row.get("save_id"),
            row.get("scene_snapshot_id"),
            row.get("scene_generation"),
            canonical_key,
        )
        if fact_unique_key not in active_keys:
            active_keys.add(fact_unique_key)
            normalized_reversed.append(row)
            continue
        duplicate_id = row.get("id")
        if isinstance(duplicate_id, str):
            duplicate_ids.add(duplicate_id)
    rows["scene_facts"] = list(reversed(normalized_reversed))

    seen_sources: set[tuple[object, ...]] = set()
    normalized_sources: list[dict[str, object]] = []
    for row in rows.get("scene_fact_sources", []):
        fact_id = row.get("scene_fact_id")
        if isinstance(fact_id, str) and fact_id in duplicate_ids:
            continue
        source_unique_key = (
            row.get("scene_fact_id"),
            row.get("source_message_id"),
            row.get("evidence_quote"),
        )
        if source_unique_key in seen_sources:
            continue
        seen_sources.add(source_unique_key)
        normalized_sources.append(row)
    rows["scene_fact_sources"] = normalized_sources
    return {
        table_name: tuple(table_rows)
        for table_name, table_rows in rows.items()
    }


def _optional_snapshot_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _remap_snapshot_memory_reference(
    *,
    table_name: str,
    row: dict[str, object],
    memory_id_map: Mapping[str, str],
) -> None:
    if table_name == "context_sources" and row.get("source_type") == "memory":
        _remap_snapshot_row_id(row, "source_id", memory_id_map)
    elif table_name == "entity_links":
        if row.get("entity_type") in {"memory", "memories"}:
            _remap_snapshot_row_id(row, "entity_id", memory_id_map)
        if row.get("target_type") in {"memory", "memories"}:
            _remap_snapshot_row_id(row, "target_id", memory_id_map)
    elif table_name in {"context_update_suggestions", "context_update_audit"}:
        if row.get("entity_type") in {"memory", "memories"}:
            _remap_snapshot_row_id(row, "entity_id", memory_id_map)
    elif table_name in {
        "character_knowledge_edges",
        "character_text_provenance",
    }:
        if row.get("target_type") in {"memory", "memories"}:
            _remap_snapshot_row_id(row, "target_id", memory_id_map)
    if table_name == "character_text_proactive_triggers":
        if row.get("source_type") in {"memory", "memories"}:
            _remap_snapshot_row_id(row, "source_id", memory_id_map)
        trigger_key = row.get("trigger_key")
        if isinstance(trigger_key, str):
            row["trigger_key"] = _remap_snapshot_memory_text(
                trigger_key,
                memory_id_map,
            )
    if table_name == "active_threads":
        raw_related_entities = row.get("related_entities_json")
        if not isinstance(raw_related_entities, str):
            return
        try:
            related_entities = json.loads(raw_related_entities)
        except json.JSONDecodeError:
            return
        if isinstance(related_entities, list):
            row["related_entities_json"] = _compact_json(
                [
                    _remap_snapshot_typed_memory_reference(
                        value,
                        memory_id_map,
                    )
                    for value in related_entities
                ]
            )


def _remap_snapshot_row_id(
    row: dict[str, object],
    column: str,
    id_map: Mapping[str, str],
) -> None:
    value = row.get(column)
    if isinstance(value, str):
        row[column] = id_map.get(value, value)


def _remap_snapshot_memory_text(
    value: str,
    memory_id_map: Mapping[str, str],
) -> str:
    parts = value.split(":")
    if (
        len(parts) >= 2
        and parts[0] in {"memory", "memories"}
        and parts[1] in memory_id_map
    ):
        parts[1] = memory_id_map[parts[1]]
    return ":".join(parts)


def _remap_snapshot_typed_memory_reference(
    value: object,
    memory_id_map: Mapping[str, str],
) -> object:
    if not isinstance(value, str):
        return value
    entity_type, separator, entity_id = value.partition(":")
    if separator and entity_type in {"memory", "memories"}:
        return f"{entity_type}:{memory_id_map.get(entity_id, entity_id)}"
    return value


def _snapshot_row_unique_key(
    table_name: str,
    row: Mapping[str, object],
) -> tuple[object, ...]:
    if table_name == "context_sources":
        return (
            table_name,
            row.get("save_id"),
            row.get("source_type"),
            row.get("source_id"),
        )
    if table_name == "entity_links":
        return (
            table_name,
            row.get("save_id"),
            row.get("entity_type"),
            row.get("entity_id"),
            row.get("target_type"),
            row.get("target_id"),
            row.get("relation"),
        )
    if table_name == "character_knowledge_edges":
        return (
            table_name,
            row.get("save_id"),
            row.get("character_id"),
            normalized_knowledge_target_type(str(row.get("target_type", ""))),
            row.get("target_id"),
        )
    if table_name == "character_text_proactive_triggers":
        return (
            table_name,
            row.get("save_id"),
            row.get("character_id"),
            row.get("trigger_key"),
        )
    return (table_name, row.get("id"))


def _merge_snapshot_memory_rows(
    existing: dict[str, object],
    incoming: Mapping[str, object],
) -> None:
    for field in (
        "tags_json",
        "source_message_ids_json",
        "source_observation_ids_json",
    ):
        existing[field] = _merged_snapshot_json_string_lists(
            existing.get(field),
            incoming.get(field),
            limit=(
                None
                if field == "tags_json"
                else _MAX_SNAPSHOT_PROVENANCE_GROUP_MEMBERS
            ),
            extra_values=(
                (
                    existing.get("source_message_id"),
                    incoming.get("source_message_id"),
                )
                if field == "source_message_ids_json"
                else ()
            ),
            overflow_fallback_values=(
                (existing.get("source_message_id"),)
                if field == "source_message_ids_json"
                else ()
            ),
            preserve_first_on_overflow=field != "tags_json",
        )
    existing["importance"] = max(
        _snapshot_numeric_value(existing.get("importance")),
        _snapshot_numeric_value(incoming.get("importance")),
    )
    if (
        existing.get("archived_at") is not None
        and incoming.get("archived_at") is None
    ):
        existing["archived_at"] = None


def _merge_snapshot_entity_link_rows(
    existing: dict[str, object],
    incoming: Mapping[str, object],
) -> None:
    if (
        existing.get("source_message_id") is None
        and incoming.get("source_message_id") is not None
    ):
        existing["source_message_id"] = incoming["source_message_id"]


def _merge_snapshot_proactive_trigger_rows(
    existing: dict[str, object],
    incoming: Mapping[str, object],
) -> None:
    for field in ("thread_id", "text_message_id", "source_message_id"):
        if incoming.get(field) is not None:
            existing[field] = incoming[field]
    for field in ("source_type", "source_id", "reason"):
        value = incoming.get(field)
        if isinstance(value, str) and value:
            existing[field] = value


def _merge_snapshot_context_source_rows(
    existing: dict[str, object],
    incoming: Mapping[str, object],
    *,
    reject_conflict: bool = True,
) -> None:
    existing_active = existing.get("archived_at") is None
    incoming_active = incoming.get("archived_at") is None
    if incoming_active and not existing_active:
        replacement = dict(incoming)
        existing.clear()
        existing.update(replacement)
        return
    if existing_active and not incoming_active:
        return
    if not _snapshot_context_source_rows_have_same_content(existing, incoming):
        if reject_conflict:
            raise ValueError(
                "Conflicting context sources share one snapshot identity"
            )
        return
    existing["metadata_json"] = _merged_context_source_metadata_json(
        existing.get("metadata_json"),
        incoming.get("metadata_json"),
    )
    existing["token_estimate"] = max(
        int(_snapshot_numeric_value(existing.get("token_estimate"))),
        int(_snapshot_numeric_value(incoming.get("token_estimate"))),
    )


def _snapshot_context_source_rows_have_same_content(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> bool:
    return all(
        first.get(field) == second.get(field)
        for field in (
            "title",
            "body",
            "scene_snapshot_id",
            "scene_generation",
            "created_turn_number",
            "expires_after_turn_number",
        )
    )


def _merged_context_source_metadata_json(first: object, second: object) -> str:
    loaded_metadata: list[dict[str, object]] = []
    for raw in (first, second):
        try:
            loaded = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            loaded = {}
        if isinstance(loaded, dict):
            typed = cast(dict[str, object], loaded)
            loaded_metadata.append(typed)
    metadata: dict[str, object] = (
        dict(loaded_metadata[0]) if loaded_metadata else {}
    )
    provenance_overflow = False
    for field in (
        "source_message_ids",
        "tags",
    ):
        values: list[str] = []
        for item in loaded_metadata:
            raw_values = item.get(field)
            if not isinstance(raw_values, list):
                continue
            values.extend(
                str(value)
                for value in raw_values
                if isinstance(value, str) and value
            )
        merged_values = list(dict.fromkeys(values))
        if (
            field == "source_message_ids"
            and len(merged_values) > _MAX_SNAPSHOT_PROVENANCE_GROUP_MEMBERS
        ):
            provenance_overflow = True
        metadata[field] = merged_values
    groups = _snapshot_context_source_provenance_groups(loaded_metadata)
    provenance_mode = "any"
    if (
        len(groups) > _MAX_SNAPSHOT_PROVENANCE_GROUPS
        or any(
            len(group) > _MAX_SNAPSHOT_PROVENANCE_GROUP_MEMBERS
            for group in groups
        )
    ):
        provenance_overflow = True
    provenance_metadata = loaded_metadata
    if provenance_overflow:
        provenance_metadata = loaded_metadata[:1]
        first_metadata = provenance_metadata[0] if provenance_metadata else {}
        raw_source_ids = first_metadata.get("source_message_ids")
        source_ids = (
            [
                str(value)
                for value in raw_source_ids
                if isinstance(value, str) and value
            ]
            if isinstance(raw_source_ids, list)
            else []
        )
        groups = _snapshot_context_source_provenance_groups(
            provenance_metadata,
            collapse_all=False,
        )
        first_mode = first_metadata.get("source_provenance_mode")
        provenance_mode = (
            first_mode if first_mode in {"all", "any"} else "any"
        )
        if (
            len(source_ids) > _MAX_SNAPSHOT_PROVENANCE_GROUP_MEMBERS
            or len(groups) > _MAX_SNAPSHOT_PROVENANCE_GROUPS
            or any(
                len(group) > _MAX_SNAPSHOT_PROVENANCE_GROUP_MEMBERS
                for group in groups
            )
        ):
            raise ValueError("Snapshot provenance is too large")
        metadata["source_message_ids"] = source_ids
        for field in ("source_message_id", "last_seen_message_id"):
            value = first_metadata.get(field)
            if isinstance(value, str) and value:
                metadata[field] = value
            else:
                metadata.pop(field, None)
    metadata["source_provenance_groups"] = groups
    metadata["source_provenance_mode"] = provenance_mode
    if any(item.get("requires_audience") is True for item in loaded_metadata):
        metadata["requires_audience"] = True
    return _compact_json(metadata)


def _snapshot_context_source_provenance_groups(
    metadata_items: Iterable[Mapping[str, object]],
    *,
    collapse_all: bool = True,
) -> list[list[str]]:
    groups: list[list[str]] = []
    for item in metadata_items:
        raw_groups = item.get("source_provenance_groups")
        item_groups: list[list[str]] = []
        if isinstance(raw_groups, list):
            for raw_group in raw_groups:
                if not isinstance(raw_group, list):
                    continue
                group = [
                    str(value)
                    for value in raw_group
                    if isinstance(value, str) and value
                ]
                if group and group not in item_groups:
                    item_groups.append(group)
        raw_item_source_ids = item.get("source_message_ids")
        item_source_ids = [
            str(value)
            for value in (
                raw_item_source_ids
                if isinstance(raw_item_source_ids, list)
                else []
            )
            if isinstance(value, str) and value
        ]
        for field in ("source_message_id", "last_seen_message_id"):
            value = item.get(field)
            if isinstance(value, str) and value:
                item_source_ids.append(value)
        grouped_item_ids = {
            source_id
            for group in item_groups
            for source_id in group
        }
        ungrouped_item_ids = [
            source_id
            for source_id in dict.fromkeys(item_source_ids)
            if source_id not in grouped_item_ids
        ]
        if ungrouped_item_ids:
            item_groups.append(ungrouped_item_ids)
        if (
            collapse_all
            and item.get("source_provenance_mode") == "all"
            and item_groups
        ):
            item_groups = [
                list(
                    dict.fromkeys(
                        source_id
                        for group in item_groups
                        for source_id in group
                    )
                )
            ]
        for group in item_groups:
            if group not in groups:
                groups.append(group)
    return groups


def _merge_snapshot_knowledge_edge_rows(
    existing: dict[str, object],
    incoming: Mapping[str, object],
) -> None:
    existing_active = existing.get("archived_at") is None
    incoming_active = incoming.get("archived_at") is None
    if incoming_active and not existing_active:
        replacement = dict(incoming)
        existing.clear()
        existing.update(replacement)
        return
    if existing_active and not incoming_active:
        return
    source_message_ids = (
        existing.get("source_message_id"),
        incoming.get("source_message_id"),
    )
    state_rank = {"knows": 0, "may_know": 1, "does_not_know": 2}
    if state_rank.get(str(incoming.get("knowledge_state")), 1) > state_rank.get(
        str(existing.get("knowledge_state")),
        1,
    ):
        for field in (
            "knowledge_state",
            "acquisition_method",
            "source_message_id",
            "evidence_quote",
        ):
            existing[field] = incoming.get(field)
    existing["confidence"] = max(
        _snapshot_numeric_value(existing.get("confidence")),
        _snapshot_numeric_value(incoming.get("confidence")),
    )
    try:
        existing["source_message_ids_json"] = _merged_snapshot_json_string_lists(
            existing.get("source_message_ids_json"),
            incoming.get("source_message_ids_json"),
            limit=_MAX_SNAPSHOT_PROVENANCE_GROUP_MEMBERS,
            extra_values=source_message_ids,
        )
    except ValueError:
        existing["knowledge_state"] = "does_not_know"
        existing["acquisition_method"] = "unknown"
        existing["source_message_id"] = None
        existing["source_message_ids_json"] = "[]"
        existing["evidence_quote"] = None


def _merged_snapshot_json_string_lists(
    first: object,
    second: object,
    *,
    limit: int | None = None,
    extra_values: Iterable[object] = (),
    overflow_fallback_values: Iterable[object] = (),
    preserve_first_on_overflow: bool = False,
) -> str:
    values = [
        str(value)
        for value in extra_values
        if isinstance(value, str) and value
    ]
    loaded_lists: list[list[str]] = []
    for raw in (first, second):
        try:
            loaded = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            loaded = []
        if isinstance(loaded, list):
            loaded_values = [
                str(value)
                for value in loaded
                if isinstance(value, str) and value
            ]
            loaded_lists.append(loaded_values)
            values.extend(loaded_values)
        else:
            loaded_lists.append([])
    merged = list(dict.fromkeys(values))
    if limit is not None and len(merged) > limit:
        if preserve_first_on_overflow:
            fallback = [
                str(value)
                for value in overflow_fallback_values
                if isinstance(value, str) and value
            ]
            fallback.extend(loaded_lists[0])
            return _compact_json(list(dict.fromkeys(fallback))[:limit])
        raise ValueError("Merged snapshot provenance is too large")
    return _compact_json(merged)


def _snapshot_numeric_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


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
