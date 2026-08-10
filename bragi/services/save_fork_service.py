"""Create independent save forks from chronicle prefixes."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import (
    PersistenceRepositories,
    _epistemic_claim_fingerprint,
)
from bragi.services.turn_snapshot_service import TurnSnapshotService


@dataclass(frozen=True)
class SaveForkResult:
    save: SaveRecord
    message_count: int
    media_count: int


class SaveForkService:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories
        self._id_maps: dict[str, dict[str, str]] = {}

    def fork_from_message(
        self,
        *,
        save_id: str,
        message_id: str,
        media_dir: Path,
        owner_user_id: str | None = None,
    ) -> SaveForkResult:
        source_save = self.repositories.get_save(save_id)
        if source_save is None:
            raise ValueError(f"Unknown save id: {save_id}")

        messages = self.repositories.list_messages(save_id)
        selected_index = next(
            (
                index
                for index, message in enumerate(messages)
                if message.id == message_id
            ),
            None,
        )
        if selected_index is None:
            raise ValueError(f"Unknown active message id: {message_id}")

        prefix = messages[: selected_index + 1]
        snapshot_service = TurnSnapshotService(self.repositories)
        snapshot_service.capture_current_head_if_dirty(
            save_id,
            reason="pre_fork_head",
        )
        snapshot = snapshot_service.latest_snapshot_for_message(
            save_id=save_id,
            message_id=message_id,
        )
        if snapshot is not None:
            result = snapshot_service.fork_snapshot_to_save(
                source_save_id=save_id,
                snapshot_id=snapshot.id,
                title=self._fork_title(
                    source_save.title,
                    prefix[-1],
                    selected_index + 1,
                ),
                media_dir=media_dir,
                owner_user_id=owner_user_id,
            )
            return SaveForkResult(
                save=result.save,
                message_count=result.message_count,
                media_count=result.media_count,
            )
        if prefix[-1].role == "player":
            preceding_snapshot = snapshot_service.latest_snapshot_before_message(
                save_id=save_id,
                message_id=message_id,
            )
            expected_preceding_ids = tuple(message.id for message in prefix[:-1])
            if (
                preceding_snapshot is not None
                and snapshot_service.snapshot_message_ids(
                    snapshot_id=preceding_snapshot.id
                )
                == expected_preceding_ids
            ):
                result = snapshot_service.fork_snapshot_to_save(
                    source_save_id=save_id,
                    snapshot_id=preceding_snapshot.id,
                    title=self._fork_title(
                        source_save.title,
                        prefix[-1],
                        selected_index + 1,
                    ),
                    media_dir=media_dir,
                    owner_user_id=owner_user_id,
                    trailing_messages=(prefix[-1],),
                )
                return SaveForkResult(
                    save=result.save,
                    message_count=result.message_count,
                    media_count=result.media_count,
                )
        raise ValueError(
            "Forking from this message requires a turn snapshot. "
            "Open the save and advance the chronicle before forking older messages."
        )

    def _fork_title(self, source_title: str, message: Any, turn_number: int) -> str:
        speaker = (message.speaker_name or message.role or "message").strip()
        base = f"{source_title} - fork after {speaker} {turn_number}"
        existing_titles = {save.title for save in self.repositories.list_saves()}
        if base not in existing_titles:
            return base
        suffix = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        candidate = f"{base} {suffix}"
        if candidate not in existing_titles:
            return candidate
        return f"{candidate}-{_new_id()[:8]}"

    def _copy_save_app_settings(self, source_save_id: str, fork_save_id: str) -> None:
        self.repositories.copy_save_scoped_settings(
            source_save_id=source_save_id,
            target_save_id=fork_save_id,
        )

    def _copy_message_revisions(
        self,
        source_save_id: str,
        fork_save_id: str,
        message_id_map: dict[str, str],
    ) -> None:
        rows = self.repositories.connection.execute(
            """
            SELECT message_id, revision_number, previous_body, new_body,
                   diff_unified, reconciliation_status, reconciliation_error,
                   reconciled_at
            FROM message_revisions
            WHERE save_id = ?
            ORDER BY created_at, rowid
            """,
            (source_save_id,),
        ).fetchall()
        for row in rows:
            mapped_message_id = message_id_map.get(row["message_id"])
            if mapped_message_id is None:
                continue
            self.repositories.connection.execute(
                """
                INSERT INTO message_revisions(
                    id, save_id, message_id, revision_number, previous_body,
                    new_body, diff_unified, reconciliation_status,
                    reconciliation_error, reconciled_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _new_id(),
                    fork_save_id,
                    mapped_message_id,
                    row["revision_number"],
                    row["previous_body"],
                    row["new_body"],
                    row["diff_unified"],
                    row["reconciliation_status"],
                    row["reconciliation_error"],
                    row["reconciled_at"],
                ),
            )

    def _copy_state_changes(
        self,
        source_save_id: str,
        fork_save_id: str,
        message_id_map: dict[str, str],
        prefix_ids: frozenset[str],
    ) -> None:
        for row in self._rows("state_changes", source_save_id):
            if not _source_in_prefix(row["source_message_id"], prefix_ids):
                continue
            self.repositories.add_state_change(
                change_id=_new_id(),
                save_id=fork_save_id,
                source_message_id=_remap_optional(
                    row["source_message_id"],
                    message_id_map,
                ),
                operation=row["operation"],
                state_key=row["state_key"],
                before_json=_remap_json_text(row["before_json"], message_id_map),
                after_json=_remap_json_text(row["after_json"], message_id_map),
            )

    def _rebuild_world_state(
        self,
        source_save_id: str,
        fork_save_id: str,
        message_id_map: dict[str, str],
        prefix_ids: frozenset[str],
    ) -> None:
        manual_rows = [
            row
            for row in self._rows("world_state", source_save_id)
            if row["source_message_id"] is None and row["archived_at"] is None
        ]
        for row in manual_rows:
            self._insert_world_state(
                fork_save_id=fork_save_id,
                key=row["key"],
                value_json=_remap_json_text(row["value_json"], message_id_map) or "{}",
                category=row["category"],
                confidence=row["confidence"],
                source_message_id=None,
            )

        for change in self._rows("state_changes", source_save_id):
            if not _source_in_prefix(change["source_message_id"], prefix_ids):
                continue
            operation = str(change["operation"]).casefold()
            if operation in {"delete", "remove", "archive", "archived", "deleted"}:
                self.repositories.connection.execute(
                    "DELETE FROM world_state WHERE save_id = ? AND key = ?",
                    (fork_save_id, change["state_key"]),
                )
                continue
            if change["after_json"] is None:
                continue
            self._insert_world_state(
                fork_save_id=fork_save_id,
                key=change["state_key"],
                value_json=_remap_json_text(change["after_json"], message_id_map)
                or "{}",
                category="",
                confidence=1.0,
                source_message_id=_remap_optional(
                    change["source_message_id"], message_id_map
                ),
            )

        for row in self._rows("world_state", source_save_id):
            if row["archived_at"] is not None:
                continue
            if row["source_message_id"] is None:
                continue
            if not _source_in_prefix(row["source_message_id"], prefix_ids):
                continue
            exists = self.repositories.connection.execute(
                "SELECT 1 FROM world_state WHERE save_id = ? AND key = ?",
                (fork_save_id, row["key"]),
            ).fetchone()
            if exists is not None:
                continue
            self._insert_world_state(
                fork_save_id=fork_save_id,
                key=row["key"],
                value_json=_remap_json_text(row["value_json"], message_id_map) or "{}",
                category=row["category"],
                confidence=row["confidence"],
                source_message_id=_remap_optional(
                    row["source_message_id"],
                    message_id_map,
                ),
            )

    def _insert_world_state(
        self,
        *,
        fork_save_id: str,
        key: str,
        value_json: str,
        category: str,
        confidence: float,
        source_message_id: str | None,
    ) -> None:
        self.repositories.connection.execute(
            """
            INSERT INTO world_state(
                id, save_id, key, value_json, category, confidence,
                source_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(save_id, key) DO UPDATE SET
                value_json = excluded.value_json,
                category = excluded.category,
                confidence = excluded.confidence,
                source_message_id = excluded.source_message_id,
                updated_at = CURRENT_TIMESTAMP,
                archived_at = NULL
            """,
            (
                _new_id(),
                fork_save_id,
                key,
                value_json,
                category,
                confidence,
                source_message_id,
            ),
        )

    def _copy_save_scoped_tables(
        self,
        source_save_id: str,
        fork_save_id: str,
        message_id_map: dict[str, str],
        prefix_ids: frozenset[str],
    ) -> None:
        self._copy_table(
            "locations",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
            json_columns=tuple(
                column
                for column in self._column_names("locations")
                if column.endswith("_json")
            ),
            reference_columns={"parent_location_id": "locations"},
            message_columns=(
                "source_message_id",
                "first_seen_message_id",
                "last_updated_message_id",
            ),
        )
        self._copy_table(
            "characters",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
            json_columns=tuple(
                column
                for column in self._column_names("characters")
                if column.endswith("_json")
            ),
            reference_columns={"location_id": "locations"},
            message_columns=(
                "source_message_id",
                "first_seen_message_id",
                "last_updated_message_id",
            ),
        )
        self._copy_table(
            "dating_route_states",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
            json_columns=("known_boundaries_json", "unresolved_questions_json"),
            reference_columns={
                "player_character_id": "characters",
                "npc_character_id": "characters",
            },
            message_columns=(
                "source_message_id",
                "first_met_message_id",
                "last_interaction_message_id",
            ),
        )
        self._copy_table(
            "scene_snapshots",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            json_columns=tuple(
                column
                for column in self._column_names("scene_snapshots")
                if column.endswith("_json")
            ),
            reference_columns={"current_location_id": "locations"},
            message_columns=(
                "source_message_id",
                "first_seen_message_id",
                "last_updated_message_id",
            ),
        )
        self._copy_table(
            "context_sources",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
            json_columns=("metadata_json",),
            provenance_json_columns=("metadata_json",),
            source_id_message_types=frozenset({"message"}),
            reference_columns={"scene_snapshot_id": "scene_snapshots"},
        )
        self._copy_table(
            "message_scene_presence",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            message_columns=("message_id",),
            reference_columns={"character_id": "characters"},
        )
        self._copy_table(
            "message_action_choices",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            message_columns=("message_id",),
        )
        self._preallocate_table_ids(
            "active_threads",
            source_save_id,
            prefix_ids,
            exclude_archived=True,
            message_columns=(
                "source_message_id",
                "first_seen_message_id",
                "last_updated_message_id",
            ),
        )
        self._copy_table(
            "active_threads",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
            json_columns=("related_entities_json", "locked_fields_json"),
            message_columns=(
                "source_message_id",
                "first_seen_message_id",
                "last_updated_message_id",
            ),
        )
        self._copy_table(
            "context_update_suggestions",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            json_columns=("proposed_value_json", "source_message_ids_json"),
            message_list_columns=("source_message_ids_json",),
        )
        self._copy_table(
            "context_update_audit",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            json_columns=("before_json", "after_json", "source_message_ids_json"),
            message_list_columns=("source_message_ids_json",),
            reference_columns={"suggestion_id": "context_update_suggestions"},
        )
        self._copy_table(
            "context_observations",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
            json_columns=("tags_json", "metadata_json", "source_message_ids_json"),
            message_list_columns=("source_message_ids_json",),
            reference_columns={"epistemic_actor_id": "characters"},
        )
        self._copy_table(
            "memories",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
            json_columns=(
                "tags_json",
                "source_message_ids_json",
                "source_observation_ids_json",
            ),
            message_list_columns=("source_message_ids_json",),
            reference_columns={"epistemic_actor_id": "characters"},
            reference_list_columns={
                "source_observation_ids_json": "context_observations"
            },
        )
        for memory in self.repositories.list_memories(fork_save_id):
            self.repositories.connection.execute(
                "UPDATE memories SET claim_fingerprint = ? WHERE id = ?",
                (
                    _epistemic_claim_fingerprint(
                        memory.body,
                        epistemic_status=memory.epistemic_status,
                        epistemic_actor_id=memory.epistemic_actor_id,
                        epistemic_actor_name=memory.epistemic_actor_name,
                    ),
                    memory.id,
                ),
            )
        self._copy_table(
            "summaries",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            json_columns=("source_summary_ids_json",),
            message_columns=("covers_message_start_id", "covers_message_end_id"),
            message_list_columns=("source_message_ids_json",),
        )
        self._copy_table(
            "save_scenario_updates",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
            json_columns=("content_json", "source_message_ids_json"),
            message_list_columns=("source_message_ids_json",),
        )
        self._copy_table(
            "save_loss_conditions",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
        )
        self._copy_table(
            "save_loss_condition_changes",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
            json_columns=("before_json", "after_json"),
            reference_columns={"condition_id": "save_loss_conditions"},
        )
        self._copy_table(
            "save_loss_outcomes",
            source_save_id,
            fork_save_id,
            message_id_map,
            prefix_ids,
            exclude_archived=True,
            json_columns=("evidence_json",),
            provenance_json_columns=("evidence_json",),
            message_columns=("triggering_message_id", "epilogue_message_id"),
            reference_columns={"condition_id": "save_loss_conditions"},
        )

    def _copy_media_assets(
        self,
        source_save_id: str,
        fork_save_id: str,
        *,
        media_dir: Path,
        message_id_map: dict[str, str],
        prefix_ids: frozenset[str],
        copied_paths: list[Path],
    ) -> int:
        asset_id_map = self._id_maps.setdefault("media_assets", {})
        copied_count = 0
        for row in self._rows("media_assets", source_save_id):
            if row["archived_at"] is not None:
                continue
            if not _source_in_prefix(row["source_message_id"], prefix_ids):
                continue
            new_asset_id = asset_id_map.setdefault(row["id"], _new_id())
            copied_path = _copy_media_file(
                media_dir=media_dir,
                source_relative_path=row["path"],
                fork_save_id=fork_save_id,
                asset_id=new_asset_id,
                copied_paths=copied_paths,
            )
            copied_thumbnail = (
                _copy_media_file(
                    media_dir=media_dir,
                    source_relative_path=row["thumbnail_path"],
                    fork_save_id=fork_save_id,
                    asset_id=new_asset_id,
                    copied_paths=copied_paths,
                    thumbnail=True,
                )
                if row["thumbnail_path"] is not None
                else None
            )
            self.repositories.create_media_asset(
                asset_id=new_asset_id,
                save_id=fork_save_id,
                source_message_id=_remap_optional(
                    row["source_message_id"], message_id_map
                ),
                type=row["type"],
                path=copied_path,
                thumbnail_path=copied_thumbnail,
                prompt=row["prompt"],
                provider=row["provider"],
                model=row["model"],
                status=row["status"],
                mime_type=row["mime_type"],
                metadata=json.loads(
                    _remap_json_text(row["metadata_json"], message_id_map) or "{}"
                ),
                source_media_asset_id=asset_id_map.get(row["source_media_asset_id"]),
            )
            copied_count += 1
        return copied_count

    def _preallocate_media_asset_ids(
        self,
        source_save_id: str,
        prefix_ids: frozenset[str],
    ) -> None:
        asset_id_map = self._id_maps.setdefault("media_assets", {})
        for row in self._rows("media_assets", source_save_id):
            if row["archived_at"] is not None:
                continue
            if not _source_in_prefix(row["source_message_id"], prefix_ids):
                continue
            asset_id_map.setdefault(row["id"], _new_id())

    def _preallocate_table_ids(
        self,
        table_name: str,
        source_save_id: str,
        prefix_ids: frozenset[str],
        *,
        exclude_archived: bool = False,
        message_columns: tuple[str, ...] = ("source_message_id",),
    ) -> None:
        columns = self._column_names(table_name)
        for row in self._rows(table_name, source_save_id):
            if (
                exclude_archived
                and "archived_at" in columns
                and row["archived_at"] is not None
            ):
                continue
            if not _row_provenance_in_prefix(
                row,
                message_columns=tuple(
                    column for column in message_columns if column in columns
                ),
                message_list_columns=(),
                provenance_json_columns=(),
                prefix_ids=prefix_ids,
                source_id_message_types=frozenset(),
            ):
                continue
            self._mapped_table_id(table_name, row["id"])

    def _copy_table(
        self,
        table_name: str,
        source_save_id: str,
        fork_save_id: str,
        message_id_map: dict[str, str],
        prefix_ids: frozenset[str],
        *,
        exclude_archived: bool = False,
        json_columns: tuple[str, ...] = (),
        provenance_json_columns: tuple[str, ...] = (),
        message_columns: tuple[str, ...] = ("source_message_id",),
        message_list_columns: tuple[str, ...] = (),
        reference_columns: dict[str, str] | None = None,
        reference_list_columns: dict[str, str] | None = None,
        source_id_message_types: frozenset[str] = frozenset(),
    ) -> None:
        reference_columns = reference_columns or {}
        reference_list_columns = reference_list_columns or {}
        columns = self._column_names(table_name)
        insert_columns = tuple(
            column
            for column in columns
            if column not in {"created_at", "updated_at", "last_opened_at"}
        )
        for row in self._rows(table_name, source_save_id):
            if (
                exclude_archived
                and "archived_at" in columns
                and row["archived_at"] is not None
            ):
                continue
            if not _row_provenance_in_prefix(
                row,
                message_columns=tuple(
                    column for column in message_columns if column in columns
                ),
                message_list_columns=tuple(
                    column for column in message_list_columns if column in columns
                ),
                provenance_json_columns=tuple(
                    column for column in provenance_json_columns if column in columns
                ),
                prefix_ids=prefix_ids,
                source_id_message_types=source_id_message_types,
            ):
                continue

            values: list[Any] = []
            for column in insert_columns:
                value = row[column]
                if column == "id":
                    value = self._mapped_table_id(table_name, value)
                elif column == "save_id":
                    value = fork_save_id
                elif column in message_columns:
                    value = _remap_optional(value, message_id_map)
                elif column in reference_columns:
                    value = self._mapped_optional_table_id(
                        reference_columns[column],
                        value,
                    )
                elif column in reference_list_columns:
                    raw_ids = json.loads(value or "[]")
                    reference_map = self._id_maps.get(
                        reference_list_columns[column], {}
                    )
                    value = json.dumps(
                        [
                            reference_map[item]
                            for item in raw_ids
                            if isinstance(item, str) and item in reference_map
                        ],
                        sort_keys=True,
                    )
                elif table_name == "memories" and column == "claim_fingerprint":
                    actor_id = self._mapped_optional_table_id(
                        "characters", row["epistemic_actor_id"]
                    )
                    value = _epistemic_claim_fingerprint(
                        str(row["body"]),
                        epistemic_status=str(row["epistemic_status"]),
                        epistemic_actor_id=actor_id,
                        epistemic_actor_name=str(row["epistemic_actor_name"]),
                    )
                elif (
                    column == "source_id"
                    and row["source_type"] in source_id_message_types
                ):
                    value = _remap_optional(value, message_id_map)
                elif column in message_list_columns:
                    value = _remap_message_list_json(value, message_id_map)
                elif column in json_columns:
                    value = self._remap_json_column(
                        table_name,
                        column,
                        value,
                        message_id_map,
                    )
                values.append(value)
            placeholders = ", ".join("?" for _ in insert_columns)
            self.repositories.connection.execute(
                f"""
                INSERT INTO {table_name}({", ".join(insert_columns)})
                VALUES ({placeholders})
                """,
                tuple(values),
            )

    def _remap_json_column(
        self,
        table_name: str,
        column: str,
        value: str | None,
        message_id_map: dict[str, str],
    ) -> str | None:
        if (
            table_name == "scene_snapshots"
            and column == "present_character_ids_json"
        ):
            return self._remap_table_id_list_json(value, "characters")
        if table_name == "active_threads" and column == "related_entities_json":
            return self._remap_related_entities_json(value)
        if table_name == "summaries" and column == "source_summary_ids_json":
            return self._remap_table_id_list_json(value, "summaries")
        return _remap_json_text(value, message_id_map)

    def _remap_table_id_list_json(
        self,
        value: str | None,
        table_name: str,
    ) -> str | None:
        if value is None:
            return None
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return value
        if not isinstance(raw, list):
            return value
        table_map = self._id_maps.get(table_name, {})
        remapped: list[object] = []
        for item in raw:
            if isinstance(item, str):
                mapped = table_map.get(item)
                if mapped is not None:
                    remapped.append(mapped)
                continue
            remapped.append(item)
        return json.dumps(remapped, sort_keys=True)

    def _remap_related_entities_json(self, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return value
        if not isinstance(raw, list):
            return value
        remapped: list[object] = []
        for item in raw:
            if not isinstance(item, str):
                remapped.append(item)
                continue
            mapped = self._remap_related_entity(item)
            if mapped is not None:
                remapped.append(mapped)
        return json.dumps(remapped, sort_keys=True)

    def _remap_related_entity(self, value: str) -> str | None:
        entity_type, separator, entity_id = value.partition(":")
        if not separator:
            return value
        table_name = _entity_table_name(entity_type)
        if table_name is None:
            return value
        mapped = self._mapped_optional_table_id(table_name, entity_id)
        if mapped is None:
            return None
        return f"{entity_type}:{mapped}"

    def _copy_entity_links(
        self,
        source_save_id: str,
        fork_save_id: str,
        message_id_map: dict[str, str],
        prefix_ids: frozenset[str],
    ) -> None:
        columns = self._column_names("entity_links")
        insert_columns = tuple(
            column for column in columns if column not in {"created_at"}
        )
        for row in self._rows("entity_links", source_save_id):
            if not _row_provenance_in_prefix(
                row,
                message_columns=("source_message_id",),
                message_list_columns=(),
                provenance_json_columns=(),
                prefix_ids=prefix_ids,
                source_id_message_types=frozenset(),
            ):
                continue
            if _media_asset_link_endpoint_missing_from_fork(
                row,
                media_asset_id_map=self._id_maps.get("media_assets", {}),
            ):
                continue
            values: list[Any] = []
            for column in insert_columns:
                value = row[column]
                if column == "id":
                    value = self._mapped_table_id("entity_links", value)
                elif column == "save_id":
                    value = fork_save_id
                elif column == "source_message_id":
                    value = _remap_optional(value, message_id_map)
                elif column == "entity_id":
                    value = self._mapped_entity_id(row["entity_type"], value)
                elif column == "target_id":
                    value = self._mapped_entity_id(row["target_type"], value)
                values.append(value)
            self.repositories.connection.execute(
                f"""
                INSERT INTO entity_links({", ".join(insert_columns)})
                VALUES ({", ".join("?" for _ in insert_columns)})
                """,
                tuple(values),
            )

    def _mapped_entity_id(self, entity_type: str, value: str) -> str:
        table_name = _entity_table_name(entity_type)
        if table_name is None:
            return value
        return self._mapped_optional_table_id(table_name, value) or value

    def _mapped_table_id(self, table_name: str, value: str) -> str:
        table_map = self._id_maps.setdefault(table_name, {})
        mapped = table_map.get(value)
        if mapped is None:
            mapped = _new_id()
            table_map[value] = mapped
        return mapped

    def _mapped_optional_table_id(
        self,
        table_name: str,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        return self._id_maps.get(table_name, {}).get(value)

    def _column_names(self, table_name: str) -> tuple[str, ...]:
        rows = self.repositories.connection.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()
        return tuple(row["name"] for row in rows)

    def _rows(self, table_name: str, save_id: str) -> list[Any]:
        return list(
            self.repositories.connection.execute(
                f"SELECT * FROM {table_name} WHERE save_id = ? ORDER BY rowid",
                (save_id,),
            ).fetchall()
        )


def _new_id() -> str:
    return str(uuid4())


def _source_in_prefix(
    source_message_id: str | None,
    prefix_ids: frozenset[str],
) -> bool:
    return source_message_id is None or source_message_id in prefix_ids


def _media_asset_link_endpoint_missing_from_fork(
    row: Any,
    *,
    media_asset_id_map: dict[str, str],
) -> bool:
    return (
        (
            row["entity_type"] == "media_asset"
            and row["entity_id"] not in media_asset_id_map
        )
        or (
            row["target_type"] == "media_asset"
            and row["target_id"] not in media_asset_id_map
        )
    )


def _entity_table_name(entity_type: str) -> str | None:
    return {
        "location": "locations",
        "character": "characters",
        "thread": "active_threads",
        "active_thread": "active_threads",
        "scene_snapshot": "scene_snapshots",
        "dating_route_state": "dating_route_states",
        "loss_condition": "save_loss_conditions",
        "media_asset": "media_assets",
        "save": "saves",
    }.get(entity_type)


def _remap_optional(value: str | None, message_id_map: dict[str, str]) -> str | None:
    if value is None:
        return None
    return message_id_map.get(value, value)


def _remap_message_list_json(value: str | None, message_id_map: dict[str, str]) -> str:
    if value is None:
        return "[]"
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return value
    if not isinstance(raw, list):
        return value
    return json.dumps([message_id_map.get(item, item) for item in raw], sort_keys=True)


def _remap_json_text(value: str | None, message_id_map: dict[str, str]) -> str | None:
    if value is None:
        return None
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return value
    return json.dumps(_remap_json_value(raw, message_id_map), sort_keys=True)


def _remap_json_value(value: object, message_id_map: dict[str, str]) -> object:
    if isinstance(value, str):
        return message_id_map.get(value, value)
    if isinstance(value, list):
        return [_remap_json_value(item, message_id_map) for item in value]
    if isinstance(value, dict):
        return {
            key: _remap_json_value(item, message_id_map)
            for key, item in value.items()
        }
    return value


def _row_provenance_in_prefix(
    row: Any,
    *,
    message_columns: tuple[str, ...],
    message_list_columns: tuple[str, ...],
    provenance_json_columns: tuple[str, ...],
    prefix_ids: frozenset[str],
    source_id_message_types: frozenset[str],
) -> bool:
    if (
        source_id_message_types
        and row["source_type"] in source_id_message_types
        and row["source_id"] not in prefix_ids
    ):
        return False
    for column in message_columns:
        value = row[column]
        if value is not None and value not in prefix_ids:
            return False
    for column in message_list_columns:
        for message_id in _message_ids_from_json(row[column]):
            if message_id not in prefix_ids:
                return False
    for column in provenance_json_columns:
        for message_id in _message_ids_from_json_tree(row[column], prefix_ids):
            if message_id not in prefix_ids:
                return False
    return True


def _message_ids_from_json(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, str))


def _message_ids_from_json_tree(
    value: str | None,
    known_message_ids: frozenset[str],
) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return ()
    found: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, str):
            if item in known_message_ids or _looks_like_message_id(item):
                found.append(item)
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)

    visit(raw)
    return tuple(found)


def _looks_like_message_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F-]{32,36}", value))


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
