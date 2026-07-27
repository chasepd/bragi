"""Portable chat/save import and export bundles."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast
from urllib.parse import quote
from uuid import uuid4

from bragi import __version__
from bragi.app_logging import log_event
from bragi.json_safety import JsonSafetyError, validate_json_structure
from bragi.persistence.context_provenance import merge_context_source_metadata
from bragi.persistence.migrations import CURRENT_SCHEMA_VERSION
from bragi.persistence.repositories import (
    PersistenceRepositories,
    canonical_claim_fingerprint,
)
from bragi.private_files import write_private_bytes
from bragi.services.action_choice_flags import normalize_legacy_action_choice_scenario
from bragi.services.character_text_world_update_service import (
    character_text_source_ref,
    parse_character_text_source_ref,
)
from bragi.services.chat_history_settings import (
    DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW,
    DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW,
    NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    sanitize_recent_message_window,
)
from bragi.services.director_pressure_service import DIRECTOR_PRESSURE_STATE_KEY
from bragi.services.generation_settings import (
    MODEL_THINKING_PREFERENCES_SETTING,
    sanitize_model_thinking_preferences,
)
from bragi.services.image_style_settings import (
    IMAGE_STYLE_PRESET_SETTING,
    sanitize_image_style_preset,
)
from bragi.services.knowledge_boundary import normalized_knowledge_target_type
from bragi.services.model_preferences import (
    SAVE_MODEL_OVERRIDES_SETTING,
    sanitize_save_model_overrides,
)
from bragi.services.post_turn_inference import (
    POST_TURN_INFERENCE_MODE_SETTING,
    sanitize_post_turn_inference_mode,
)
from bragi.services.scenario_content_rating import (
    metadata_with_scenario_content_ratings,
)
from bragi.services.scenario_service import (
    RETIRED_SCENARIO_REASON,
    scenario_record_is_retired,
    strip_deprecated_scenario_character_sections,
)
from bragi.services.turn_snapshot_service import (
    TurnSnapshotService,
    portable_context_observation_curation_state_row,
)
from bragi.world_time_model import (
    canonical_world_time_from_values,
    legacy_world_time_fields,
)
from bragi.zip_safety import ZipSafetyError, validate_zip_directory
from bragi_common.media_mime import imported_media_mime_type

BUNDLE_FORMAT = "bragi-chat-bundle"
BUNDLE_VERSION = 1
MANIFEST_NAME = "manifest.json"
DATA_NAME = "data.json"
_MAX_BUNDLE_MEDIA_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_BUNDLE_MEDIA_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_BUNDLE_MANIFEST_JSON_BYTES = 1024 * 1024
_MAX_BUNDLE_DATA_JSON_BYTES = 128 * 1024 * 1024
_MAX_BUNDLE_TABLE_ROWS = 20_000
_MAX_BUNDLE_MESSAGE_ROWS = 5_000
_MAX_BUNDLE_TOTAL_ROWS = 50_000
_MAX_BUNDLE_JSON_OBJECTS = 150_000
_MAX_BUNDLE_JSON_NODES = 2_000_000
_MAX_BUNDLE_JSON_DEPTH = 128
_MAX_BUNDLE_JSON_TOTAL_BYTES = (
    _MAX_BUNDLE_MANIFEST_JSON_BYTES + _MAX_BUNDLE_DATA_JSON_BYTES
)
_MAX_BUNDLE_TOTAL_DECOMPRESSED_BYTES = (
    _MAX_BUNDLE_JSON_TOTAL_BYTES + _MAX_BUNDLE_MEDIA_TOTAL_BYTES
)
_MIN_BUNDLE_COMPRESSION_RATIO_CHECK_BYTES = 1024 * 1024
_MAX_BUNDLE_COMPRESSION_RATIO = 250.0
_BUNDLE_JSON_COPY_CHUNK_BYTES = 1024 * 1024
_BUNDLE_MEDIA_COPY_CHUNK_BYTES = 1024 * 1024
_CONTEXT_SOURCE_METADATA_MESSAGE_ID_FIELDS = (
    "source_message_id",
    "last_seen_message_id",
)
_CONTEXT_SOURCE_METADATA_MESSAGE_ID_LIST_FIELDS = ("source_message_ids",)
_MAX_CONTEXT_SOURCE_PROVENANCE_GROUPS = 64
_MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS = 64


class ChatBundleError(ValueError):
    """Raised when a chat bundle is invalid or unsupported."""


@dataclass(frozen=True)
class ChatBundleManifest:
    bundle_format: str
    bundle_version: int
    save_id: str
    title: str
    scenario_title: str
    message_count: int
    media_count: int
    created_at: str | None
    updated_at: str | None
    exported_at: str


@dataclass(frozen=True)
class _BundleMediaMember:
    bundle_name: str
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class _ExportMediaFile:
    path: Path
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class ChatBundlePreview:
    save_id: str
    title: str
    scenario_title: str
    message_count: int
    media_count: int
    bundle_version: int
    created_at: str | None = None
    updated_at: str | None = None
    exported_at: str | None = None


@dataclass(frozen=True)
class ImportedChatBundle:
    save_id: str
    scenario_id: str
    title: str
    message_count: int
    media_count: int
    skipped_media_count: int


class _BundleImportRepairTracker:
    def __init__(self) -> None:
        self._field_counts: dict[str, int] = {}

    def record(self, field_name: str) -> None:
        self._field_counts[field_name] = self._field_counts.get(field_name, 0) + 1

    @property
    def repaired_reference_count(self) -> int:
        return sum(self._field_counts.values())

    @property
    def repaired_fields(self) -> dict[str, int]:
        return dict(sorted(self._field_counts.items()))


class ChatBundleService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        media_dir: Path,
    ) -> None:
        self.repositories = repositories
        self.media_dir = media_dir

    def export_save(
        self,
        save_id: str,
        bundle_path: Path,
        *,
        include_message_revisions: bool = False,
    ) -> ChatBundleManifest:
        self.repositories.begin_transaction()
        try:
            (
                data,
                save,
                scenario,
                message_count,
                snapshot_media_asset_rows,
            ) = self._capture_export_save_data(
                save_id,
                include_message_revisions=include_message_revisions,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise

        _annotate_character_reference_image_asset_ids(data)
        _repair_export_media_asset_source_references(data)
        media_assets = cast(list[dict[str, object]], data["media_assets"])
        snapshot_media_assets = _snapshot_only_media_asset_rows(
            snapshot_media_asset_rows,
            active_media_asset_ids={_text(row, "id") for row in media_assets},
        )
        data["snapshot_media_assets"] = snapshot_media_assets
        _annotate_export_media_asset_files(
            [*media_assets, *snapshot_media_assets],
            self.media_dir,
        )
        media_files = _collect_media_files(
            [*media_assets, *snapshot_media_assets],
            self.media_dir,
        )
        exported_at = datetime.now(UTC).isoformat()
        manifest_payload: dict[str, object] = {
            "format": BUNDLE_FORMAT,
            "bundle_format": BUNDLE_FORMAT,
            "bundle_version": BUNDLE_VERSION,
            "title": save["title"],
            "save_title": save["title"],
            "scenario_title": scenario["title"],
            "message_count": message_count,
            "media_count": len(media_assets),
            "created_by": {
                "application": "Bragi",
                "version": __version__,
            },
            "bragi_schema_version": CURRENT_SCHEMA_VERSION,
            "schema_version": CURRENT_SCHEMA_VERSION,
            "exported_at": exported_at,
            "save": {
                "id": save["id"],
                "title": save["title"],
                "created_at": save["created_at"],
                "updated_at": save["updated_at"],
            },
            "scenario": {
                "id": scenario["id"],
                "title": scenario["title"],
            },
            "counts": {
                "messages": message_count,
                "media_assets": len(media_assets),
            },
        }
        manifest = _manifest_from_payload(manifest_payload)

        _write_bundle_atomically(
            bundle_path=bundle_path,
            manifest_payload=manifest_payload,
            data=data,
            media_files=media_files,
        )
        log_event(
            "chat_bundle.exported",
            save_id=save_id,
            bundle_path=str(bundle_path),
            message_count=manifest.message_count,
            media_count=manifest.media_count,
        )
        return manifest

    def _capture_export_save_data(
        self,
        save_id: str,
        *,
        include_message_revisions: bool,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        int,
        tuple[dict[str, object], ...],
    ]:
        details = self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")

        save = _require_row(
            self.repositories.connection.execute(
                """
                SELECT id, scenario_id, title, active, created_at, updated_at,
                       last_opened_at, custom_instructions
                FROM saves
                WHERE id = ?
                """,
                (save_id,),
            ).fetchone(),
            f"Unknown save id: {save_id}",
        )
        scenario = _require_row(
            self.repositories.connection.execute(
                """
                SELECT id, type, title, premise, player_role, content_json,
                       created_at, updated_at
                FROM scenarios
                WHERE id = ?
                """,
                (save["scenario_id"],),
            ).fetchone(),
            f"Unknown scenario id: {save['scenario_id']}",
        )
        scenario_payload = _row_dict(scenario)
        scenario_payload["content_json"] = json.dumps(
            strip_deprecated_scenario_character_sections(
                _json_object(scenario_payload, "content_json"),
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        data: dict[str, object] = {
            "scenario": scenario_payload,
            "save": _row_dict(save),
            "messages": self._rows(
                """
                SELECT id, save_id, role, speaker_name, body, provider, model,
                       token_estimate, created_at, updated_at, deleted_at,
                       safety_transition, content_rating
                FROM messages
                WHERE save_id = ? AND deleted_at IS NULL
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "message_revisions": (
                self._rows(
                    """
                    SELECT id, save_id, message_id, revision_number, previous_body,
                           new_body, diff_unified, reconciliation_status,
                           reconciliation_error, created_at, reconciled_at
                    FROM message_revisions
                    WHERE save_id = ?
                    ORDER BY message_id, revision_number, rowid
                    """,
                    (save_id,),
                )
                if include_message_revisions
                else []
            ),
            "character_text_message_revisions": (
                self._rows(
                    """
                    SELECT id, save_id, text_message_id, revision_number,
                           previous_body, new_body, diff_unified,
                           reconciliation_status, reconciliation_error,
                           created_at, reconciled_at
                    FROM character_text_message_revisions
                    WHERE save_id = ?
                    ORDER BY text_message_id, revision_number, rowid
                    """,
                    (save_id,),
                )
                if include_message_revisions
                else []
            ),
            "message_action_choices": self._rows(
                """
                SELECT id, save_id, message_id, ordinal, body, provider, model,
                       content_rating, created_at, updated_at
                FROM message_action_choices
                WHERE save_id = ?
                ORDER BY message_id, ordinal, rowid
                """,
                (save_id,),
            ),
            "world_state": self._rows(
                """
                SELECT id, save_id, key, value_json, category, confidence,
                       source_message_id, created_at, updated_at, archived_at
                FROM world_state
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY key
                """,
                (save_id,),
            ),
            "state_changes": self._rows(
                """
                SELECT id, save_id, source_message_id, operation, state_key,
                       before_json, after_json, created_at
                FROM state_changes
                WHERE save_id = ?
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "memories": self._rows(
                """
                SELECT id, save_id, body, tags_json, importance,
                       source_message_id, source_message_ids_json,
                       claim_fingerprint, source_observation_ids_json,
                       created_at, updated_at, archived_at
                FROM memories
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "summaries": self._rows(
                """
                SELECT id, save_id, covers_message_start_id,
                       covers_message_end_id, body, provider, model, created_at,
                       content_rating, archived_at
                FROM summaries
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "save_scenario_updates": self._rows(
                """
                SELECT id, save_id, source_message_id, title, premise,
                       player_role, content_json, source_message_ids_json,
                       reason, provider, model, created_at, archived_at
                FROM save_scenario_updates
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "save_loss_conditions": self._rows(
                """
                SELECT id, save_id, key, label, name, description, status,
                       severity, source, source_message_id, created_at, updated_at,
                       archived_at
                FROM save_loss_conditions
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "save_loss_condition_changes": self._rows(
                """
                SELECT id, save_id, condition_id, source_message_id, operation,
                       before_json, after_json, reason, provider, model,
                       created_at, archived_at
                FROM save_loss_condition_changes
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "save_loss_outcomes": self._rows(
                """
                SELECT id, save_id, condition_id, condition_name,
                       triggering_message_id, explanation, evidence_json,
                       confidence, provider, model, outcome_type, epilogue_provider,
                       epilogue_model, epilogue_message_id, epilogue_error,
                       created_at, archived_at
                FROM save_loss_outcomes
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "media_assets": self._media_asset_rows(save_id),
            "context_sources": self._rows(
                """
                SELECT id, save_id, source_type, source_id, title, body,
                       metadata_json, token_estimate, scene_snapshot_id,
                       scene_generation, created_turn_number,
                       expires_after_turn_number, created_at, updated_at, archived_at
                FROM context_sources
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY source_type, source_id, rowid
                """,
                (save_id,),
            ),
            "context_observations": self._rows(
                """
                SELECT id, save_id, observation_type, claim, evidence_quote,
                       source_message_ids_json, scope, status, confidence,
                       tags_json, metadata_json, created_at, updated_at,
                       archived_at
                FROM context_observations
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "context_observation_curation_states": [
                portable_context_observation_curation_state_row(row)
                for row in self._rows(
                    """
                    SELECT observation_id, save_id, attempt_count,
                           next_eligible_at, lease_token, lease_until,
                           last_error, terminal_outcome, completed_at,
                           created_at, updated_at
                    FROM context_observation_curation_state
                    WHERE save_id = ?
                    ORDER BY created_at, observation_id
                    """,
                    (save_id,),
                )
            ],
            "scene_snapshots": self._rows(
                """
                SELECT id, save_id, current_location_id, situation, objective,
                       in_world_time, time_of_day, day_of_week, world_day_index,
                       world_time_day_index, world_time_day_label,
                       world_time_phase, world_time_clock_minutes,
                       world_time_period_label, world_time_source_message_id,
                       world_time_confidence,
                       weather, mood,
                       nearby_objects_json, hazards_json,
                       present_character_ids_json, source_message_id,
                       locked_fields_json, first_seen_message_id,
                       last_updated_message_id, scene_generation,
                       created_at, updated_at
                FROM scene_snapshots
                WHERE save_id = ?
                ORDER BY rowid
                """,
                (save_id,),
            ),
            "dating_route_states": self._rows(
                """
                SELECT id, save_id, player_character_id, npc_character_id, stage,
                       first_met_message_id, first_met_world_day_index,
                       last_interaction_message_id,
                       last_interaction_world_day_index, completed_interactions,
                       dates_completed, interest_level, trust_level,
                       comfort_with_intimacy, pacing_preference,
                       known_boundaries_json, unresolved_questions_json,
                       next_reasonable_step, source_message_id, created_at,
                       updated_at, archived_at
                FROM dating_route_states
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY updated_at, created_at, rowid
                """,
                (save_id,),
            ),
            "character_text_threads": self._rows(
                """
                SELECT id, save_id, character_id, title, status, kind,
                       memory_body, memory_message_count, memory_updated_at,
                       created_at, updated_at, archived_at
                FROM character_text_threads
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY updated_at, created_at, rowid
                """,
                (save_id,),
            ),
            "character_text_thread_participants": self._rows(
                """
                SELECT id, save_id, thread_id, character_id, ordinal,
                       created_at, updated_at, archived_at
                FROM character_text_thread_participants
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY thread_id, ordinal, created_at, rowid
                """,
                (save_id,),
            ),
            "character_text_messages": self._rows(
                """
                SELECT id, save_id, thread_id, character_id, sender, body,
                       sender_character_id, provider, model, token_estimate,
                       content_rating,
                       created_at, updated_at, deleted_at, delivery_status,
                       delivery_error,
                       delivery_job_id, delivery_attempt, in_world_sent_at,
                       delivered_at, read_at, reply_to_message_id
                FROM character_text_messages
                WHERE save_id = ? AND deleted_at IS NULL
                ORDER BY thread_id, created_at, rowid
                """,
                (save_id,),
            ),
            "character_text_activity_events": self._rows(
                """
                SELECT id, save_id, ordinal, thread_id, activity_type,
                       text_message_id, read_count, delivery_status, created_at
                FROM character_text_activity_events
                WHERE save_id = ?
                ORDER BY ordinal
                """,
                (save_id,),
            ),
            "narrator_phone_activity_cursors": self._rows(
                """
                SELECT narrator_message_id, save_id, last_activity_ordinal
                FROM narrator_phone_activity_cursors
                WHERE save_id = ?
                ORDER BY narrator_message_id
                """,
                (save_id,),
            ),
            "character_text_message_attachments": self._rows(
                """
                SELECT id, save_id, thread_id, text_message_id, character_id,
                       ordinal, kind, status, media_asset_id, prompt, error,
                       metadata_json, created_at, updated_at
                FROM character_text_message_attachments
                WHERE save_id = ?
                ORDER BY text_message_id, ordinal, created_at, rowid
                """,
                (save_id,),
            ),
            "character_text_provenance": self._rows(
                """
                SELECT id, save_id, thread_id, text_message_id, target_type,
                       target_id, operation, field_path, created_at
                FROM character_text_provenance
                WHERE save_id = ?
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "character_text_proactive_triggers": self._rows(
                """
                SELECT id, save_id, character_id, trigger_key, trigger_type,
                       thread_id, text_message_id, source_type, source_id,
                       source_message_id, reason, created_at, updated_at
                FROM character_text_proactive_triggers
                WHERE save_id = ?
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "character_contact_states": self._rows(
                """
                SELECT id, save_id, player_character_id, character_id,
                       player_has_character_number, character_has_player_number,
                       source_message_id, source_text_message_id, created_at,
                       updated_at, archived_at
                FROM character_contact_states
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY updated_at, created_at, rowid
                """,
                (save_id,),
            ),
            "locations": self._rows(
                """
                SELECT id, save_id, name, aliases_json, description,
                       visual_description, parent_location_id, connections_json,
                       status, hazards_json, source_message_id, locked_fields_json,
                       first_seen_message_id, last_updated_message_id, created_at,
                       updated_at, archived_at
                FROM locations
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "characters": self._rows(
                """
                SELECT id, save_id, name, aliases_json, role, age, known_state,
                       history, met, appearance, visual_notes, current_clothing,
                       personality, voice, texting_style, relationships_json, goals,
                       motivations,
                       current_intent, boundaries, attitude_toward_player,
                       cooperation_conditions, status, location_id, private_notes,
                       source_message_id, locked_fields_json,
                       protected_from_maintenance, is_player_character,
                       contact_name,
                       first_seen_message_id,
                       last_updated_message_id, content_rating,
                       created_at, updated_at, archived_at
                FROM characters
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "active_threads": self._rows(
                """
                SELECT id, save_id, title, description, status, priority,
                       visibility, related_entities_json, source_message_id,
                       locked_fields_json, first_seen_message_id,
                       last_updated_message_id, created_at, updated_at, archived_at
                FROM active_threads
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY priority DESC, created_at, rowid
                """,
                (save_id,),
            ),
            "entity_links": self._rows(
                """
                SELECT id, save_id, entity_type, entity_id, target_type,
                       target_id, relation, source_message_id, created_at
                FROM entity_links
                WHERE save_id = ?
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "character_knowledge_edges": self._rows(
                """
                SELECT id, save_id, character_id, target_type, target_id,
                       knowledge_state, acquisition_method, confidence,
                       source_message_id, source_message_ids_json,
                       evidence_quote, created_at, updated_at, archived_at
                FROM character_knowledge_edges
                WHERE save_id = ? AND archived_at IS NULL
                ORDER BY character_id, target_type, target_id, created_at, rowid
                """,
                (save_id,),
            ),
            "message_visibility": self._rows(
                """
                SELECT id, save_id, message_id, character_id, visibility,
                       confidence, source, evidence, created_at, updated_at
                FROM message_visibility
                WHERE save_id = ?
                ORDER BY message_id, character_id, created_at, rowid
                """,
                (save_id,),
            ),
            "message_scene_presence": self._rows(
                """
                SELECT id, save_id, message_id, character_id, source,
                       created_at, updated_at
                FROM message_scene_presence
                WHERE save_id = ?
                ORDER BY message_id, character_id, created_at, rowid
                """,
                (save_id,),
            ),
            "context_update_suggestions": self._rows(
                """
                SELECT id, save_id, update_type, entity_type, entity_id,
                       field_path, proposed_value_json, status, reason,
                       confidence, source_message_ids_json, created_at, resolved_at,
                       review_attempt_count, next_review_at, last_review_error
                FROM context_update_suggestions
                WHERE save_id = ?
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "context_update_audit": self._rows(
                """
                SELECT id, save_id, suggestion_id, operation, entity_type,
                       entity_id, field_path, before_json, after_json, reason,
                       confidence, source_message_ids_json, created_at
                FROM context_update_audit
                WHERE save_id = ?
                ORDER BY created_at, rowid
                """,
                (save_id,),
            ),
            "save_app_settings": _save_app_setting_rows(
                self.repositories,
                save_id=save_id,
                scenario_id=str(save["scenario_id"]),
            ),
        }
        messages = cast(list[dict[str, object]], data["messages"])
        state_changes = cast(list[dict[str, object]], data["state_changes"])
        media_assets = cast(list[dict[str, object]], data["media_assets"])
        active_message_ids = {row["id"] for row in messages}
        snapshot_active_message_ids = {str(row["id"]) for row in messages}
        data["state_changes"] = [
            row
            for row in state_changes
            if row.get("source_message_id") is None
            or row.get("source_message_id") in active_message_ids
        ]
        data["message_revisions"] = [
            row
            for row in _list_of_objects(
                data["message_revisions"],
                "message_revisions",
            )
            if row.get("message_id") in active_message_ids
        ]
        data["message_action_choices"] = _filter_message_action_choice_rows(
            data["message_action_choices"],
            active_message_ids=active_message_ids,
        )
        data["world_state"] = _filter_optional_source_rows(
            data["world_state"], active_message_ids, "world_state"
        )
        data["summaries"] = [
            row
            for row in _list_of_objects(data["summaries"], "summaries")
            if row.get("covers_message_start_id") in active_message_ids
            and row.get("covers_message_end_id") in active_message_ids
        ]
        data["save_loss_conditions"] = _filter_optional_source_rows(
            data["save_loss_conditions"],
            active_message_ids,
            "save_loss_conditions",
        )
        data["save_loss_condition_changes"] = _filter_optional_source_rows(
            data["save_loss_condition_changes"],
            active_message_ids,
            "save_loss_condition_changes",
        )
        data["save_loss_outcomes"] = _filter_loss_outcomes(
            data["save_loss_outcomes"],
            active_message_ids,
        )
        data["media_assets"] = [
            row
            for row in media_assets
            if row.get("source_message_id") is None
            or row.get("source_message_id") in active_message_ids
        ]
        data["scene_snapshots"] = _filter_scene_snapshot_rows(
            data["scene_snapshots"],
            active_message_ids,
        )
        for key in (
            "locations",
            "characters",
            "active_threads",
            "entity_links",
        ):
            data[key] = _filter_optional_source_rows(data[key], active_message_ids, key)
        exported_character_ids = {
            row["id"] for row in _list_of_objects(data["characters"], "characters")
        }
        data["dating_route_states"] = _filter_dating_route_state_rows(
            data["dating_route_states"],
            active_message_ids=active_message_ids,
            exported_character_ids=exported_character_ids,
        )
        (
            text_threads,
            text_participants,
            text_messages,
            text_provenance,
        ) = _filter_character_text_rows(
            threads=data["character_text_threads"],
            participants=data["character_text_thread_participants"],
            messages=data["character_text_messages"],
            provenance=data["character_text_provenance"],
            exported_character_ids=exported_character_ids,
        )
        data["character_text_threads"] = text_threads
        data["character_text_thread_participants"] = text_participants
        data["character_text_messages"] = text_messages
        activity_rows = [
            row
            for row in _list_of_objects(
                data.get("character_text_activity_events"),
                "character_text_activity_events",
            )
            if row.get("thread_id") in {thread["id"] for thread in text_threads}
            and (
                row.get("text_message_id") is None
                or row.get("text_message_id")
                in {message["id"] for message in text_messages}
            )
        ]
        data["character_text_activity_events"] = activity_rows
        max_activity_ordinal = max(
            (_optional_int(row, "ordinal") or 0 for row in activity_rows),
            default=0,
        )
        data["narrator_phone_activity_cursors"] = [
            {
                **row,
                "last_activity_ordinal": min(
                    _optional_int(row, "last_activity_ordinal") or 0,
                    max_activity_ordinal,
                ),
            }
            for row in _list_of_objects(
                data.get("narrator_phone_activity_cursors"),
                "narrator_phone_activity_cursors",
            )
            if row.get("narrator_message_id") in active_message_ids
        ]
        data["character_text_provenance"] = text_provenance
        exported_text_message_ids = {row["id"] for row in text_messages}
        data["character_text_message_attachments"] = (
            _filter_character_text_attachment_rows(
                data.get("character_text_message_attachments"),
                exported_character_ids=exported_character_ids,
                exported_thread_ids={row["id"] for row in text_threads},
                exported_text_message_ids=exported_text_message_ids,
            )
        )
        exported_text_attachment_media_ids = {
            row["media_asset_id"]
            for row in _list_of_objects(
                data["character_text_message_attachments"],
                "character_text_message_attachments",
            )
            if isinstance(row.get("media_asset_id"), str)
        }
        data["media_assets"] = _filter_character_text_attachment_media_rows(
            data["media_assets"],
            exported_text_attachment_media_ids=exported_text_attachment_media_ids,
        )
        data["character_text_message_revisions"] = [
            row
            for row in _list_of_objects(
                data.get("character_text_message_revisions"),
                "character_text_message_revisions",
            )
            if row.get("text_message_id") in exported_text_message_ids
        ]
        data["character_text_proactive_triggers"] = (
            _filter_character_text_proactive_trigger_rows(
                data.get("character_text_proactive_triggers"),
                exported_character_ids=exported_character_ids,
                active_message_ids=active_message_ids,
                exported_thread_ids={row["id"] for row in text_threads},
                exported_text_message_ids=exported_text_message_ids,
            )
        )
        data["character_contact_states"] = _filter_character_contact_state_rows(
            data.get("character_contact_states"),
            exported_character_ids=exported_character_ids,
            active_message_ids=active_message_ids,
            exported_text_message_ids=exported_text_message_ids,
        )
        active_source_refs = _active_export_source_refs(
            active_message_ids=active_message_ids,
            exported_text_message_ids=exported_text_message_ids,
        )
        data["memories"] = _filter_memory_rows(
            data["memories"],
            active_source_refs,
        )
        data["save_scenario_updates"] = _filter_save_scenario_updates(
            data["save_scenario_updates"],
            active_source_refs,
        )
        data["context_sources"] = _filter_context_source_rows(
            data["context_sources"],
            active_source_refs,
        )
        data["context_observations"] = _filter_source_id_list_rows(
            data["context_observations"],
            active_source_refs,
            "context_observations",
        )
        data["context_update_suggestions"] = _filter_source_id_list_rows(
            data["context_update_suggestions"],
            active_source_refs,
            "context_update_suggestions",
        )
        data["context_update_audit"] = _filter_source_id_list_rows(
            data["context_update_audit"],
            active_source_refs,
            "context_update_audit",
        )
        exported_target_ids = _exported_knowledge_target_ids(data)
        data["character_knowledge_edges"] = _filter_character_knowledge_edge_rows(
            data["character_knowledge_edges"],
            active_source_refs=active_source_refs,
            exported_character_ids=exported_character_ids,
            exported_target_ids=exported_target_ids,
        )
        data["message_visibility"] = _filter_message_visibility_rows(
            data["message_visibility"],
            active_message_ids=active_message_ids,
            exported_character_ids=exported_character_ids,
        )
        data["message_scene_presence"] = _filter_message_presence_rows(
            data["message_scene_presence"],
            active_message_ids=active_message_ids,
            exported_character_ids=exported_character_ids,
        )
        data["locations"] = _location_rows_parent_first(
            _list_of_objects(data["locations"], "locations"),
            save_id,
        )
        snapshot_service = TurnSnapshotService(self.repositories)
        snapshot_rows, snapshot_objects = snapshot_service.export_snapshot_rows(
            save_id=save_id,
            active_message_ids=snapshot_active_message_ids,
        )
        data["turn_snapshots"] = snapshot_rows
        data["snapshot_objects"] = snapshot_objects
        return (
            data,
            cast(dict[str, object], data["save"]),
            cast(dict[str, object], data["scenario"]),
            len(messages),
            snapshot_service.media_asset_rows_from_snapshot_objects(snapshot_objects),
        )

    def preview_import(self, bundle_path: Path) -> ChatBundlePreview:
        manifest = self._read_manifest(bundle_path)
        return ChatBundlePreview(
            save_id=manifest.save_id,
            title=manifest.title,
            scenario_title=manifest.scenario_title,
            message_count=manifest.message_count,
            media_count=manifest.media_count,
            bundle_version=manifest.bundle_version,
            created_at=manifest.created_at,
            updated_at=manifest.updated_at,
            exported_at=manifest.exported_at,
        )

    def import_save(
        self,
        bundle_path: Path,
        *,
        owner_user_id: str | None = None,
    ) -> ImportedChatBundle:
        manifest_payload, data = self._read_bundle(bundle_path)
        _manifest_from_payload(manifest_payload)
        media_members = _load_media_members(bundle_path, data)
        media_backups: dict[Path, bytes | None] = {}
        repair_tracker = _BundleImportRepairTracker()
        try:
            self.repositories.begin_immediate_transaction()
            imported = self._import_data(
                data,
                bundle_path,
                media_members,
                media_backups,
                owner_user_id=owner_user_id,
                repair_tracker=repair_tracker,
            )
            self.repositories.rebuild_context_source_search_terms(imported.save_id)
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            _restore_media_backups(media_backups)
            raise
        if repair_tracker.repaired_reference_count:
            log_event(
                "chat_bundle.import_repaired",
                save_id=imported.save_id,
                repaired_reference_count=repair_tracker.repaired_reference_count,
                repaired_fields=repair_tracker.repaired_fields,
            )
        log_event(
            "chat_bundle.imported",
            save_id=imported.save_id,
            scenario_id=imported.scenario_id,
            message_count=imported.message_count,
            media_count=imported.media_count,
            skipped_media_count=imported.skipped_media_count,
        )
        return imported

    def _import_data(
        self,
        data: dict[str, object],
        bundle_path: Path,
        media_members: dict[tuple[str, str], _BundleMediaMember],
        media_backups: dict[Path, bytes | None],
        *,
        owner_user_id: str | None,
        repair_tracker: _BundleImportRepairTracker,
    ) -> ImportedChatBundle:
        scenario_data = _object(data.get("scenario"), "scenario")
        save_data = _object(data.get("save"), "save")
        messages_data = _list_of_objects(data.get("messages"), "messages")
        bundle_snapshot_rows = _list_of_objects(
            data.get("turn_snapshots"),
            "turn_snapshots",
        )
        bundle_snapshot_objects = _list_of_objects(
            data.get("snapshot_objects"),
            "snapshot_objects",
        )
        bundle_snapshot_media_rows = _list_of_objects(
            data.get("snapshot_media_assets"),
            "snapshot_media_assets",
        )
        scenario_type, scenario_content, _legacy_action_choices_enabled = (
            normalize_legacy_action_choice_scenario(
                scenario_type=_text(scenario_data, "type"),
                content=_json_object(scenario_data, "content_json"),
            )
        )
        scenario_content = strip_deprecated_scenario_character_sections(
            scenario_content,
        )
        scenario_content = _quarantine_imported_scenario_content(scenario_content)

        scenario = self.repositories.create_scenario(
            type=scenario_type,
            title=_text(scenario_data, "title"),
            premise=_text(scenario_data, "premise"),
            player_role=_text(scenario_data, "player_role"),
            content=scenario_content,
        )
        scenario_id_map = {_text(scenario_data, "id"): scenario.id}
        save = self.repositories.create_save(
            scenario_id=scenario.id,
            title=_text(save_data, "title"),
            custom_instructions=_optional_text(save_data, "custom_instructions") or "",
            owner_user_id=owner_user_id,
        )
        _apply_save_app_settings(
            self.repositories,
            data.get("save_app_settings"),
            save_id=save.id,
            scenario_id=scenario.id,
        )
        if not bundle_snapshot_rows:
            TurnSnapshotService(self.repositories).capture_baseline_snapshot(
                save.id,
                reason="legacy_import_baseline",
            )

        message_id_map: dict[str, str] = {}
        message_order: dict[str, int] = {}
        transitioned_original_message_ids: set[str] = set()
        transitioned_message_ids: set[str] = set()
        for message_data in messages_data:
            original_id = _text(message_data, "id")
            role = _text(message_data, "role")
            if role not in {"player", "narrator"}:
                raise ChatBundleError(
                    f"Unsupported message role in chat bundle: {role}"
                )
            body = _text(message_data, "body")
            safety_transition = _optional_text(message_data, "safety_transition") or ""
            content_rating = "unclassified"
            message = self.repositories.append_message(
                save_id=save.id,
                role=role,
                speaker_name=_optional_text(message_data, "speaker_name"),
                body=body,
                provider=_optional_text(message_data, "provider"),
                model=_optional_text(message_data, "model"),
                token_estimate=_optional_int(message_data, "token_estimate"),
                created_at=_optional_text(message_data, "created_at"),
                updated_at=_optional_text(message_data, "updated_at"),
                safety_transition=safety_transition,
                content_rating=content_rating,
                touch_save_updated_at=False,
            )
            safety_transition = message.safety_transition
            if safety_transition:
                transitioned_original_message_ids.add(original_id)
                transitioned_message_ids.add(message.id)
            message_id_map[original_id] = message.id
            message_order[original_id] = len(message_order)

        character_text_message_id_map = _new_id_map(
            data.get("character_text_messages"),
            "character_text_messages",
        )
        imported_id_maps: dict[str, dict[str, str]] = {
            "scenario": scenario_id_map,
            "scenarios": scenario_id_map,
            "message": message_id_map,
            "messages": message_id_map,
            "character_text_message": character_text_message_id_map,
            "character_text_messages": character_text_message_id_map,
        }
        message_action_choice_id_map: dict[str, str] = {}
        for row in _list_of_objects(
            data.get("message_action_choices"),
            "message_action_choices",
        ):
            original_id = _text(row, "id")
            choice_id = uuid4().hex
            self.repositories.connection.execute(
                """
                INSERT INTO message_action_choices(
                    id, save_id, message_id, ordinal, body, provider, model,
                    content_rating, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    choice_id,
                    save.id,
                    _mapped_required(message_id_map, _text(row, "message_id")),
                    _int(row, "ordinal"),
                    _text(row, "body"),
                    _optional_text(row, "provider") or "",
                    _optional_text(row, "model") or "",
                    "unclassified",
                    _optional_text(row, "created_at")
                    or datetime.now(UTC).isoformat(),
                    _optional_text(row, "updated_at")
                    or datetime.now(UTC).isoformat(),
                ),
            )
            message_action_choice_id_map[original_id] = choice_id
        imported_id_maps["message_action_choice"] = message_action_choice_id_map
        imported_id_maps["message_action_choices"] = message_action_choice_id_map
        for row in _list_of_objects(data.get("message_revisions"), "message_revisions"):
            self.repositories.connection.execute(
                """
                INSERT INTO message_revisions(
                    id, save_id, message_id, revision_number, previous_body,
                    new_body, diff_unified, reconciliation_status,
                    reconciliation_error, created_at, reconciled_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    save.id,
                    _mapped_required(message_id_map, _text(row, "message_id")),
                    _int(row, "revision_number"),
                    _text(row, "previous_body"),
                    _text(row, "new_body"),
                    _text(row, "diff_unified"),
                    _text(row, "reconciliation_status"),
                    _optional_text(row, "reconciliation_error"),
                    _text(row, "created_at"),
                    _optional_text(row, "reconciled_at"),
                ),
            )

        world_state_id_map: dict[str, str] = {}
        world_state_key_map: dict[str, str] = {}
        for row in _list_of_objects(data.get("world_state"), "world_state"):
            original_id = _text(row, "id")
            if _bundle_row_references_messages(
                row,
                transitioned_original_message_ids,
            ):
                continue
            key = _text(row, "key")
            value = _json_object(row, "value_json")
            if key == DIRECTOR_PRESSURE_STATE_KEY:
                value = _remap_director_pressure_state_value(
                    value,
                    message_id_map,
                    repair_tracker,
                )
            state = self.repositories.upsert_world_state(
                save_id=save.id,
                key=key,
                value=value,
                category=_text(row, "category"),
                confidence=_float(row, "confidence"),
                source_message_id=_mapped_optional_required(
                    message_id_map=message_id_map,
                    original_id=_optional_text(row, "source_message_id"),
                    field_name="world_state.source_message_id",
                    repair_tracker=repair_tracker,
                ),
            )
            world_state_id_map[original_id] = state.id
            world_state_key_map[key] = state.key
        imported_id_maps["world_state"] = world_state_id_map
        imported_id_maps["world_state_key"] = world_state_key_map

        memory_id_map: dict[str, str] = {}
        for row in _list_of_objects(data.get("memories"), "memories"):
            original_id = _text(row, "id")
            if _bundle_row_references_messages(
                row,
                transitioned_original_message_ids,
            ):
                continue
            memory = self.repositories.add_memory(
                save_id=save.id,
                body=_text(row, "body"),
                tags=_json_string_list(row, "tags_json"),
                importance=_float(row, "importance"),
                source_message_id=_mapped_optional_required(
                    message_id_map=message_id_map,
                    original_id=_optional_text(row, "source_message_id"),
                    field_name="memories.source_message_id",
                    repair_tracker=repair_tracker,
                ),
                source_message_ids=_mapped_memory_source_message_ids(
                    row=row,
                    message_id_map=message_id_map,
                    character_text_message_id_map=character_text_message_id_map,
                    message_order=message_order,
                    repair_tracker=repair_tracker,
                ),
                claim_fingerprint=canonical_claim_fingerprint(_text(row, "body")),
            )
            memory_id_map[original_id] = memory.id
        imported_id_maps["memory"] = memory_id_map
        imported_id_maps["memories"] = memory_id_map

        summary_id_map: dict[str, str] = {}
        for row in _list_of_objects(data.get("summaries"), "summaries"):
            original_id = _text(row, "id")
            if _bundle_summary_covers_transition(
                row,
                transition_message_ids=transitioned_original_message_ids,
                message_order=message_order,
            ):
                continue
            start_id = _mapped_required(
                message_id_map,
                _text(row, "covers_message_start_id"),
            )
            end_id = _mapped_required(
                message_id_map,
                _text(row, "covers_message_end_id"),
            )
            summary = self.repositories.add_summary(
                save_id=save.id,
                covers_message_start_id=start_id,
                covers_message_end_id=end_id,
                body=_text(row, "body"),
                provider=_text(row, "provider"),
                model=_text(row, "model"),
                content_rating="unclassified",
            )
            summary_id_map[original_id] = summary.id
        imported_id_maps["summary"] = summary_id_map
        imported_id_maps["summaries"] = summary_id_map

        state_change_id_map: dict[str, str] = {}
        for row in _list_of_objects(data.get("state_changes"), "state_changes"):
            original_id = _text(row, "id")
            if _bundle_row_references_messages(
                row,
                transitioned_original_message_ids,
            ):
                continue
            change = self.repositories.add_state_change(
                save_id=save.id,
                operation=_text(row, "operation"),
                state_key=_text(row, "state_key"),
                before_json=_validated_optional_json_object_text(row, "before_json"),
                after_json=_validated_optional_json_object_text(row, "after_json"),
                source_message_id=_mapped_optional_required(
                    message_id_map=message_id_map,
                    original_id=_optional_text(row, "source_message_id"),
                    field_name="state_changes.source_message_id",
                    repair_tracker=repair_tracker,
                ),
            )
            state_change_id_map[original_id] = change.id
        imported_id_maps["state_change"] = state_change_id_map
        imported_id_maps["state_changes"] = state_change_id_map

        observation_id_map: dict[str, str] = {}
        for row in _list_of_objects(
            data.get("context_observations"),
            "context_observations",
        ):
            original_id = _text(row, "id")
            if _bundle_row_references_messages(
                row,
                transitioned_original_message_ids,
            ):
                continue
            observation = self.repositories.add_context_observation(
                save_id=save.id,
                observation_type=_text(row, "observation_type"),
                claim=_text(row, "claim"),
                evidence_quote=_text(row, "evidence_quote"),
                source_message_ids=_mapped_observation_source_message_ids(
                    row=row,
                    message_id_map=message_id_map,
                    character_text_message_id_map=character_text_message_id_map,
                    message_order=message_order,
                    repair_tracker=repair_tracker,
                ),
                scope=_text(row, "scope"),
                status=_text(row, "status"),
                confidence=_float(row, "confidence"),
                tags=_json_string_list(row, "tags_json"),
                metadata=_optional_json_object(row, "metadata_json") or {},
            )
            observation_id_map[original_id] = observation.id
        imported_id_maps["observation"] = observation_id_map
        imported_id_maps["context_observations"] = observation_id_map
        imported_memories = {
            memory.id: memory for memory in self.repositories.list_memories(save.id)
        }
        for row in _list_of_objects(data.get("memories"), "memories"):
            memory_id = memory_id_map.get(_text(row, "id"))
            imported_memory = imported_memories.get(memory_id or "")
            if imported_memory is None:
                continue
            source_observation_ids = [
                observation_id_map[original_id]
                for original_id in (
                    _json_string_list(row, "source_observation_ids_json")
                    if "source_observation_ids_json" in row
                    else []
                )
                if original_id in observation_id_map
            ]
            if not source_observation_ids:
                continue
            current_memory = self.repositories.get_memory_by_claim_fingerprint(
                save_id=save.id,
                claim_fingerprint=imported_memory.claim_fingerprint,
            )
            if current_memory is None:
                continue
            self.repositories.update_memory(
                memory_id=current_memory.id,
                body=current_memory.body,
                tags=current_memory.tags,
                importance=current_memory.importance,
                source_message_ids=current_memory.source_message_ids,
                source_observation_ids=list(
                    dict.fromkeys(
                        (
                            *current_memory.source_observation_ids,
                            *source_observation_ids,
                        )
                    )
                ),
                claim_fingerprint=current_memory.claim_fingerprint,
            )
        self.repositories.consolidate_active_memory_duplicates(save_id=save.id)

        for row in _list_of_objects(
            data.get("context_observation_curation_states"),
            "context_observation_curation_states",
        ):
            mapped_observation_id = observation_id_map.get(
                _text(row, "observation_id")
            )
            if mapped_observation_id is None:
                continue
            self.repositories.restore_context_observation_curation_state(
                mapped_observation_id,
                attempt_count=_optional_int(row, "attempt_count") or 0,
                next_eligible_at=_optional_text(row, "next_eligible_at"),
                last_error=_optional_text(row, "last_error"),
                terminal_outcome=_optional_text(row, "terminal_outcome"),
                completed_at=_optional_text(row, "completed_at"),
            )

        scenario_update_id_map: dict[str, str] = {}
        for row in _list_of_objects(
            data.get("save_scenario_updates"),
            "save_scenario_updates",
        ):
            original_id = _text(row, "id")
            source_message_id = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "source_message_id"),
                field_name="save_scenario_updates.source_message_id",
                repair_tracker=repair_tracker,
            )
            source_message_ids = _mapped_source_message_ids(
                row=row,
                message_id_map=message_id_map,
                character_text_message_id_map=character_text_message_id_map,
                message_order=message_order,
                repair_tracker=repair_tracker,
            )
            update = self.repositories.add_save_scenario_update(
                save_id=save.id,
                title=_text(row, "title"),
                premise=_text(row, "premise"),
                player_role=_text(row, "player_role"),
                content=_quarantine_imported_scenario_content(
                    strip_deprecated_scenario_character_sections(
                        _json_object(row, "content_json"),
                    )
                ),
                reason=_text(row, "reason"),
                provider=_text(row, "provider"),
                model=_text(row, "model"),
                source_message_id=source_message_id,
                source_message_ids=source_message_ids,
            )
            self.repositories.connection.execute(
                """
                UPDATE save_scenario_updates
                SET source_message_ids_json = ?
                WHERE id = ?
                """,
                (_dump_json_compact(list(source_message_ids)), update.id),
            )
            scenario_update_id_map[original_id] = update.id
        imported_id_maps["save_scenario_update"] = scenario_update_id_map
        imported_id_maps["save_scenario_updates"] = scenario_update_id_map

        condition_id_map: dict[str, str] = {}
        for row in _list_of_objects(
            data.get("save_loss_conditions"),
            "save_loss_conditions",
        ):
            original_id = _text(row, "id")
            condition = self.repositories.add_loss_condition(
                save_id=save.id,
                key=_text(row, "key") if "key" in row else "",
                label=_text(row, "label") if "label" in row else _text(row, "name"),
                name=_text(row, "name"),
                description=_text(row, "description"),
                status=_text(row, "status"),
                severity=_text(row, "severity") if "severity" in row else "",
                source=_text(row, "source"),
                source_message_id=_mapped_optional_required(
                    message_id_map=message_id_map,
                    original_id=_optional_text(row, "source_message_id"),
                    field_name="save_loss_conditions.source_message_id",
                    repair_tracker=repair_tracker,
                ),
            )
            condition_id_map[original_id] = condition.id
        imported_id_maps["loss_condition"] = condition_id_map
        imported_id_maps["save_loss_conditions"] = condition_id_map

        condition_change_id_map: dict[str, str] = {}
        for row in _list_of_objects(
            data.get("save_loss_condition_changes"),
            "save_loss_condition_changes",
        ):
            original_id = _text(row, "id")
            original_condition_id = _optional_text(row, "condition_id")
            loss_change = self.repositories.add_loss_condition_change(
                save_id=save.id,
                condition_id=_mapped_optional_id(
                    condition_id_map,
                    original_condition_id,
                    field_name="save_loss_condition_changes.condition_id",
                    repair_tracker=repair_tracker,
                ),
                source_message_id=_mapped_optional_required(
                    message_id_map=message_id_map,
                    original_id=_optional_text(row, "source_message_id"),
                    field_name="save_loss_condition_changes.source_message_id",
                    repair_tracker=repair_tracker,
                ),
                operation=_text(row, "operation"),
                before=_optional_json_object(row, "before_json"),
                after=_optional_json_object(row, "after_json"),
                reason=_text(row, "reason"),
                provider=_text(row, "provider"),
                model=_text(row, "model"),
            )
            condition_change_id_map[original_id] = loss_change.id
        imported_id_maps["loss_condition_change"] = condition_change_id_map
        imported_id_maps["save_loss_condition_changes"] = condition_change_id_map

        outcome_id_map: dict[str, str] = {}
        for row in _list_of_objects(
            data.get("save_loss_outcomes"),
            "save_loss_outcomes",
        ):
            original_id = _text(row, "id")
            original_condition_id = _optional_text(row, "condition_id")
            outcome = self.repositories.create_loss_outcome(
                save_id=save.id,
                condition_id=_mapped_optional_id(
                    condition_id_map,
                    original_condition_id,
                    field_name="save_loss_outcomes.condition_id",
                    repair_tracker=repair_tracker,
                ),
                condition_name=_text(row, "condition_name"),
                triggering_message_id=_mapped_required(
                    message_id_map,
                    _text(row, "triggering_message_id"),
                ),
                explanation=_text(row, "explanation"),
                evidence=_remap_evidence_message_ids(
                    _json_object(row, "evidence_json"),
                    message_id_map,
                    repair_tracker,
                ),
                confidence=_float(row, "confidence"),
                provider=_text(row, "provider"),
                model=_text(row, "model"),
                epilogue_provider=_optional_text(row, "epilogue_provider"),
                epilogue_model=_optional_text(row, "epilogue_model"),
                epilogue_message_id=_mapped_optional_required(
                    message_id_map=message_id_map,
                    original_id=_optional_text(row, "epilogue_message_id"),
                    field_name="save_loss_outcomes.epilogue_message_id",
                    repair_tracker=repair_tracker,
                ),
                epilogue_error=_optional_text(row, "epilogue_error"),
                outcome_type=(
                    _text(row, "outcome_type")
                    if "outcome_type" in row
                    else (
                        "loss_condition"
                        if original_condition_id is not None
                        else "scenario_complete"
                    )
                ),
            )
            outcome_id_map[original_id] = outcome.id
        imported_id_maps["loss_outcome"] = outcome_id_map
        imported_id_maps["save_loss_outcomes"] = outcome_id_map

        imported_media_count = 0
        skipped_media_count = 0
        media_rows = _ordered_media_asset_import_rows(
            _list_of_objects(data.get("media_assets"), "media_assets")
        )
        snapshot_media_rows = _ordered_media_asset_import_rows(
            bundle_snapshot_media_rows
        )
        live_media_asset_id_map = {_text(row, "id"): uuid4().hex for row in media_rows}
        media_asset_id_map = dict(live_media_asset_id_map)
        for row in snapshot_media_rows:
            media_asset_id_map.setdefault(_text(row, "id"), uuid4().hex)
        imported_id_maps["media_asset"] = media_asset_id_map
        imported_id_maps["media_assets"] = media_asset_id_map
        media_path_map: dict[tuple[str, str], str] = {}
        for row in media_rows:
            original_media_asset_id = _text(row, "id")
            asset_id = live_media_asset_id_map[original_media_asset_id]
            file_member = media_members.get((original_media_asset_id, "path"))
            original_path = _text(row, "path")
            relative_path = _imported_media_relative_path(
                save_id=save.id,
                asset_id=asset_id,
                original_path=original_path,
            )
            if file_member is None:
                raise ChatBundleError(
                    "Missing primary media payload for media asset: "
                    f"{original_media_asset_id}"
                )
            output_path = self.media_dir / relative_path
            _remember_media_backup(media_backups, output_path)
            _copy_bundle_media_member(bundle_path, file_member, output_path)
            media_path_map[(original_media_asset_id, "path")] = relative_path.as_posix()

            thumbnail_relative_path: str | None = None
            thumbnail_member = media_members.get(
                (original_media_asset_id, "thumbnail_path")
            )
            if thumbnail_member is not None:
                thumbnail_relative_path = _imported_thumbnail_relative_path(
                    save_id=save.id,
                    asset_id=asset_id,
                    original_path=(
                        _optional_text(row, "thumbnail_path") or original_path
                    ),
                ).as_posix()
                thumbnail_path = self.media_dir / thumbnail_relative_path
                _remember_media_backup(media_backups, thumbnail_path)
                _copy_bundle_media_member(bundle_path, thumbnail_member, thumbnail_path)
                media_path_map[
                    (original_media_asset_id, "thumbnail_path")
                ] = thumbnail_relative_path

            self.repositories.create_media_asset(
                save_id=save.id,
                source_message_id=_mapped_optional_required(
                    message_id_map=message_id_map,
                    original_id=_optional_text(row, "source_message_id"),
                    field_name="media_assets.source_message_id",
                    repair_tracker=repair_tracker,
                ),
                type=_text(row, "type"),
                path=relative_path.as_posix(),
                thumbnail_path=thumbnail_relative_path,
                prompt=_text(row, "prompt"),
                provider=_text(row, "provider"),
                model=_text(row, "model"),
                status=_text(row, "status"),
                mime_type=imported_media_mime_type(
                    _optional_text(row, "mime_type"),
                    media_type=_text(row, "type"),
                ),
                metadata=_quarantined_imported_media_source_metadata(
                    row,
                    live_media_asset_id_map,
                    repair_tracker,
                ),
                source_media_asset_id=_mapped_optional_media_asset_id(
                    media_asset_id_map=live_media_asset_id_map,
                    original_id=_optional_text(row, "source_media_asset_id"),
                    repair_tracker=repair_tracker,
                ),
                asset_id=asset_id,
            )
            imported_media_count += 1
        for row in snapshot_media_rows:
            original_media_asset_id = _text(row, "id")
            asset_id = media_asset_id_map[original_media_asset_id]
            file_member = media_members.get((original_media_asset_id, "path"))
            original_path = _text(row, "path")
            relative_path = _imported_media_relative_path(
                save_id=save.id,
                asset_id=asset_id,
                original_path=original_path,
            )
            if file_member is None:
                raise ChatBundleError(
                    "Missing primary media payload for snapshot media asset: "
                    f"{original_media_asset_id}"
                )
            output_path = self.media_dir / relative_path
            _remember_media_backup(media_backups, output_path)
            _copy_bundle_media_member(bundle_path, file_member, output_path)
            media_path_map[(original_media_asset_id, "path")] = relative_path.as_posix()

            thumbnail_member = media_members.get(
                (original_media_asset_id, "thumbnail_path")
            )
            if thumbnail_member is not None:
                thumbnail_relative_path = _imported_thumbnail_relative_path(
                    save_id=save.id,
                    asset_id=asset_id,
                    original_path=(
                        _optional_text(row, "thumbnail_path") or original_path
                    ),
                ).as_posix()
                thumbnail_path = self.media_dir / thumbnail_relative_path
                _remember_media_backup(media_backups, thumbnail_path)
                _copy_bundle_media_member(bundle_path, thumbnail_member, thumbnail_path)
                media_path_map[
                    (original_media_asset_id, "thumbnail_path")
                ] = thumbnail_relative_path

        self._import_context_graph(
            data,
            save.id,
            message_id_map,
            imported_id_maps,
            live_media_asset_id_map=live_media_asset_id_map,
            repair_tracker=repair_tracker,
        )
        try:
            imported_snapshot_count = TurnSnapshotService(
                self.repositories
            ).import_snapshot_rows(
                snapshot_rows=bundle_snapshot_rows,
                object_rows=bundle_snapshot_objects,
                source_save_id=_text(save_data, "id"),
                target_save_id=save.id,
                message_id_map=message_id_map,
                id_maps=imported_id_maps,
                media_path_map=media_path_map,
            )
        except ValueError as exc:
            raise ChatBundleError(str(exc)) from exc
        if imported_snapshot_count == 0:
            TurnSnapshotService(self.repositories).capture_current_head_if_dirty(
                save.id,
                reason="legacy_import_head",
            )
        _remove_imported_safety_transition_records(
            self.repositories.connection,
            save_id=save.id,
            transition_message_ids=transitioned_message_ids,
        )

        return ImportedChatBundle(
            save_id=save.id,
            scenario_id=scenario.id,
            title=save.title,
            message_count=len(messages_data),
            media_count=imported_media_count,
            skipped_media_count=skipped_media_count,
        )

    def _import_context_graph(
        self,
        data: dict[str, object],
        save_id: str,
        message_id_map: dict[str, str],
        imported_id_maps: dict[str, dict[str, str]] | None = None,
        *,
        live_media_asset_id_map: dict[str, str],
        repair_tracker: _BundleImportRepairTracker,
    ) -> None:
        connection = self.repositories.connection
        location_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(data.get("locations"), "locations")
        }
        character_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(data.get("characters"), "characters")
        }
        thread_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(data.get("active_threads"), "active_threads")
        }
        suggestion_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("context_update_suggestions"),
                "context_update_suggestions",
            )
        }
        context_source_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(data.get("context_sources"), "context_sources")
        }
        entity_link_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(data.get("entity_links"), "entity_links")
        }
        knowledge_edge_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("character_knowledge_edges"),
                "character_knowledge_edges",
            )
        }
        message_visibility_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("message_visibility"),
                "message_visibility",
            )
        }
        message_scene_presence_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("message_scene_presence"),
                "message_scene_presence",
            )
        }
        context_update_audit_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("context_update_audit"),
                "context_update_audit",
            )
        }
        scene_snapshot_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(data.get("scene_snapshots"), "scene_snapshots")
        }
        dating_route_state_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("dating_route_states"),
                "dating_route_states",
            )
        }
        character_text_thread_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("character_text_threads"),
                "character_text_threads",
            )
        }
        character_text_thread_participant_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("character_text_thread_participants"),
                "character_text_thread_participants",
            )
        }
        imported_id_maps = imported_id_maps or {"message": message_id_map}
        character_text_message_id_map = imported_id_maps.get(
            "character_text_message",
            _new_id_map(data.get("character_text_messages"), "character_text_messages"),
        )
        character_text_attachment_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("character_text_message_attachments"),
                "character_text_message_attachments",
            )
        }
        character_text_provenance_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("character_text_provenance"),
                "character_text_provenance",
            )
        }
        character_text_proactive_trigger_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("character_text_proactive_triggers"),
                "character_text_proactive_triggers",
            )
        }
        character_contact_state_id_map = {
            _text(row, "id"): uuid4().hex
            for row in _list_of_objects(
                data.get("character_contact_states"),
                "character_contact_states",
            )
        }
        imported_id_maps["location"] = location_id_map
        imported_id_maps["locations"] = location_id_map
        imported_id_maps["character"] = character_id_map
        imported_id_maps["characters"] = character_id_map
        imported_id_maps["thread"] = thread_id_map
        imported_id_maps["active_thread"] = thread_id_map
        imported_id_maps["active_threads"] = thread_id_map
        imported_id_maps["context_source"] = context_source_id_map
        imported_id_maps["context_sources"] = context_source_id_map
        imported_id_maps["entity_link"] = entity_link_id_map
        imported_id_maps["entity_links"] = entity_link_id_map
        imported_id_maps["character_knowledge_edge"] = knowledge_edge_id_map
        imported_id_maps["character_knowledge_edges"] = knowledge_edge_id_map
        imported_id_maps["message_visibility"] = message_visibility_id_map
        imported_id_maps["message_scene_presence"] = message_scene_presence_id_map
        imported_id_maps["context_update_suggestion"] = suggestion_id_map
        imported_id_maps["context_update_suggestions"] = suggestion_id_map
        imported_id_maps["context_update_audit"] = context_update_audit_id_map
        imported_id_maps["scene_snapshot"] = scene_snapshot_id_map
        imported_id_maps["scene_snapshots"] = scene_snapshot_id_map
        imported_id_maps["dating_route_state"] = dating_route_state_id_map
        imported_id_maps["dating_route_states"] = dating_route_state_id_map
        imported_id_maps["character_text_thread"] = character_text_thread_id_map
        imported_id_maps["character_text_threads"] = character_text_thread_id_map
        imported_id_maps["character_text_thread_participant"] = (
            character_text_thread_participant_id_map
        )
        imported_id_maps["character_text_thread_participants"] = (
            character_text_thread_participant_id_map
        )
        imported_id_maps["character_text_message"] = character_text_message_id_map
        imported_id_maps["character_text_messages"] = character_text_message_id_map
        imported_id_maps["character_text_message_attachment"] = (
            character_text_attachment_id_map
        )
        imported_id_maps["character_text_message_attachments"] = (
            character_text_attachment_id_map
        )
        imported_id_maps["character_text_provenance"] = (
            character_text_provenance_id_map
        )
        imported_id_maps["character_text_proactive_trigger"] = (
            character_text_proactive_trigger_id_map
        )
        imported_id_maps["character_text_proactive_triggers"] = (
            character_text_proactive_trigger_id_map
        )
        imported_id_maps["character_contact_state"] = character_contact_state_id_map
        imported_id_maps["character_contact_states"] = character_contact_state_id_map
        save_id_map = {_text(_object(data.get("save"), "save"), "id"): save_id}

        entity_id_maps = {
            **imported_id_maps,
            "save": save_id_map,
            "location": location_id_map,
            "character": character_id_map,
            "character_voice": character_id_map,
            "media_asset": live_media_asset_id_map,
            "media_assets": live_media_asset_id_map,
            "open_obligation": thread_id_map,
            "state": imported_id_maps.get("world_state", {}),
            "thread": thread_id_map,
            "active_thread": thread_id_map,
            "character_text_thread": character_text_thread_id_map,
            "character_text_message": character_text_message_id_map,
            "scene_snapshot": scene_snapshot_id_map,
            "dating_route_state": dating_route_state_id_map,
        }
        _remap_imported_media_reference_metadata(
            connection,
            data=data,
            media_asset_id_map=live_media_asset_id_map,
            character_id_map=character_id_map,
            character_text_thread_id_map=character_text_thread_id_map,
            character_text_message_id_map=character_text_message_id_map,
        )

        locations: list[dict[str, object]] = []
        for row in _list_of_objects(data.get("locations"), "locations"):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=location_id_map[original_id],
            )
            copied["parent_location_id"] = _mapped_optional_value(
                location_id_map,
                _optional_text(row, "parent_location_id"),
                field_name="locations.parent_location_id",
                repair_tracker=repair_tracker,
            )
            copied["source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "source_message_id"),
                field_name="locations.source_message_id",
                repair_tracker=repair_tracker,
            )
            copied["first_seen_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "first_seen_message_id")
                or _optional_text(row, "source_message_id"),
                field_name="locations.first_seen_message_id",
                repair_tracker=repair_tracker,
            )
            copied["last_updated_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "last_updated_message_id")
                or _optional_text(row, "source_message_id"),
                field_name="locations.last_updated_message_id",
                repair_tracker=repair_tracker,
            )
            locations.append(copied)
        _insert_rows(connection, "locations", locations)

        scene_snapshots: list[dict[str, object]] = []
        for row in _list_of_objects(data.get("scene_snapshots"), "scene_snapshots"):
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=scene_snapshot_id_map[_text(row, "id")],
            )
            copied["current_location_id"] = _mapped_optional_value(
                location_id_map,
                _optional_text(row, "current_location_id"),
                field_name="scene_snapshots.current_location_id",
                repair_tracker=repair_tracker,
            )
            copied["present_character_ids_json"] = _dump_json_compact(
                [
                    character_id_map.get(character_id, character_id)
                    for character_id in _json_string_list(
                        row,
                        "present_character_ids_json",
                    )
                ]
            )
            copied["source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "source_message_id"),
                field_name="scene_snapshots.source_message_id",
                repair_tracker=repair_tracker,
            )
            copied["world_time_source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "world_time_source_message_id"),
                field_name="scene_snapshots.world_time_source_message_id",
                repair_tracker=repair_tracker,
            )
            copied["first_seen_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "first_seen_message_id")
                or _optional_text(row, "source_message_id"),
                field_name="scene_snapshots.first_seen_message_id",
                repair_tracker=repair_tracker,
            )
            copied["last_updated_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "last_updated_message_id")
                or _optional_text(row, "source_message_id"),
                field_name="scene_snapshots.last_updated_message_id",
                repair_tracker=repair_tracker,
            )
            scene_snapshots.append(copied)
        _insert_rows(connection, "scene_snapshots", scene_snapshots)
        _backfill_imported_scene_world_time(connection, save_id)

        context_sources: list[dict[str, object]] = []
        for row in _list_of_objects(data.get("context_sources"), "context_sources"):
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=context_source_id_map[_text(row, "id")],
            )
            source_type = _text(copied, "source_type")
            source_id = _text(copied, "source_id")
            mapped_source_id = _mapped_context_source_id(
                entity_id_maps,
                source_type,
                source_id,
                repair_tracker=repair_tracker,
            )
            if mapped_source_id is None:
                continue
            copied["source_id"] = mapped_source_id
            copied["scene_snapshot_id"] = _mapped_optional_value(
                entity_id_maps.get("scene_snapshot", {}),
                _optional_text(row, "scene_snapshot_id"),
                field_name="context_sources.scene_snapshot_id",
                repair_tracker=repair_tracker,
            )
            remapped_metadata_json = _remapped_context_source_metadata_json(
                copied,
                message_id_map,
                character_text_message_id_map,
                observation_id_map=entity_id_maps.get("observation", {}),
                scenario_id_map=entity_id_maps.get("scenario", {}),
                entity_id_maps=entity_id_maps,
                repair_tracker=repair_tracker,
            )
            if remapped_metadata_json is None:
                context_source_id_map.pop(_text(row, "id"), None)
                continue
            copied["metadata_json"] = remapped_metadata_json
            context_sources.append(copied)
        _insert_rows(
            connection,
            "context_sources",
            _coalesce_import_context_sources(context_sources),
        )

        characters: list[dict[str, object]] = []
        for row in _list_of_objects(data.get("characters"), "characters"):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=character_id_map[original_id],
            )
            copied["location_id"] = _mapped_optional_value(
                location_id_map,
                _optional_text(row, "location_id"),
                field_name="characters.location_id",
                repair_tracker=repair_tracker,
            )
            copied["source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "source_message_id"),
                field_name="characters.source_message_id",
                repair_tracker=repair_tracker,
            )
            copied["first_seen_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "first_seen_message_id")
                or _optional_text(row, "source_message_id"),
                field_name="characters.first_seen_message_id",
                repair_tracker=repair_tracker,
            )
            copied["last_updated_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "last_updated_message_id")
                or _optional_text(row, "source_message_id"),
                field_name="characters.last_updated_message_id",
                repair_tracker=repair_tracker,
            )
            copied["history"] = _optional_text(row, "history") or _text(
                row,
                "known_state",
            )
            copied["known_state"] = copied["history"]
            copied["content_rating"] = "unclassified"
            characters.append(copied)
        _insert_rows(connection, "characters", characters)

        dating_route_states: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("dating_route_states"),
            "dating_route_states",
        ):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=dating_route_state_id_map[original_id],
            )
            copied["player_character_id"] = _mapped_imported_id(
                character_id_map,
                _text(row, "player_character_id"),
                field_name="dating_route_states.player_character_id",
            )
            copied["npc_character_id"] = _mapped_imported_id(
                character_id_map,
                _text(row, "npc_character_id"),
                field_name="dating_route_states.npc_character_id",
            )
            copied["first_met_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "first_met_message_id"),
                field_name="dating_route_states.first_met_message_id",
                repair_tracker=repair_tracker,
            )
            copied["last_interaction_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "last_interaction_message_id"),
                field_name="dating_route_states.last_interaction_message_id",
                repair_tracker=repair_tracker,
            )
            copied["source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "source_message_id"),
                field_name="dating_route_states.source_message_id",
                repair_tracker=repair_tracker,
            )
            dating_route_states.append(copied)
        _insert_rows(connection, "dating_route_states", dating_route_states)

        character_text_threads: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("character_text_threads"),
            "character_text_threads",
        ):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=character_text_thread_id_map[original_id],
            )
            copied["character_id"] = _mapped_optional_id(
                character_id_map,
                _optional_text(row, "character_id"),
                field_name="character_text_threads.character_id",
                repair_tracker=repair_tracker,
            )
            character_text_threads.append(copied)
        _insert_rows(connection, "character_text_threads", character_text_threads)

        character_text_thread_participants: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("character_text_thread_participants"),
            "character_text_thread_participants",
        ):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=character_text_thread_participant_id_map[original_id],
            )
            copied["thread_id"] = _mapped_imported_id(
                character_text_thread_id_map,
                _text(row, "thread_id"),
                field_name="character_text_thread_participants.thread_id",
            )
            copied["character_id"] = _mapped_imported_id(
                character_id_map,
                _text(row, "character_id"),
                field_name="character_text_thread_participants.character_id",
            )
            character_text_thread_participants.append(copied)
        _insert_rows(
            connection,
            "character_text_thread_participants",
            character_text_thread_participants,
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO character_text_thread_participants(
                id, save_id, thread_id, character_id, ordinal
            )
            SELECT lower(hex(randomblob(16))), save_id, id, character_id, 0
            FROM character_text_threads
            WHERE save_id = ?
              AND kind = 'direct'
              AND character_id IS NOT NULL
            """,
            (save_id,),
        )

        character_text_messages: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("character_text_messages"),
            "character_text_messages",
        ):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=character_text_message_id_map[original_id],
            )
            copied["thread_id"] = _mapped_imported_id(
                character_text_thread_id_map,
                _text(row, "thread_id"),
                field_name="character_text_messages.thread_id",
            )
            copied["character_id"] = _mapped_optional_id(
                character_id_map,
                _optional_text(row, "character_id"),
                field_name="character_text_messages.character_id",
                repair_tracker=repair_tracker,
            )
            copied["sender_character_id"] = _mapped_optional_id(
                character_id_map,
                _optional_text(row, "sender_character_id"),
                field_name="character_text_messages.sender_character_id",
                repair_tracker=repair_tracker,
            )
            copied["reply_to_message_id"] = _mapped_optional_id(
                character_text_message_id_map,
                _optional_text(row, "reply_to_message_id"),
                field_name="character_text_messages.reply_to_message_id",
                repair_tracker=repair_tracker,
            )
            copied["content_rating"] = "unclassified"
            character_text_messages.append(copied)
        _insert_rows(connection, "character_text_messages", character_text_messages)

        activity_events: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("character_text_activity_events"),
            "character_text_activity_events",
        ):
            copied = _copy_row_for_save(row, save_id, new_id=uuid4().hex)
            copied["thread_id"] = _mapped_imported_id(
                character_text_thread_id_map,
                _text(row, "thread_id"),
                field_name="character_text_activity_events.thread_id",
            )
            copied["text_message_id"] = _mapped_optional_id(
                character_text_message_id_map,
                _optional_text(row, "text_message_id"),
                field_name="character_text_activity_events.text_message_id",
                repair_tracker=repair_tracker,
            )
            activity_events.append(copied)
        _insert_rows(connection, "character_text_activity_events", activity_events)

        cursors: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("narrator_phone_activity_cursors"),
            "narrator_phone_activity_cursors",
        ):
            copied = dict(row)
            copied["save_id"] = save_id
            copied["narrator_message_id"] = _mapped_imported_id(
                message_id_map,
                _text(row, "narrator_message_id"),
                field_name="narrator_phone_activity_cursors.narrator_message_id",
            )
            cursors.append(copied)
        _insert_rows(connection, "narrator_phone_activity_cursors", cursors)

        character_text_attachments: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("character_text_message_attachments"),
            "character_text_message_attachments",
        ):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=character_text_attachment_id_map[original_id],
            )
            copied["thread_id"] = _mapped_imported_id(
                character_text_thread_id_map,
                _text(row, "thread_id"),
                field_name="character_text_message_attachments.thread_id",
            )
            copied["text_message_id"] = _mapped_imported_id(
                character_text_message_id_map,
                _text(row, "text_message_id"),
                field_name="character_text_message_attachments.text_message_id",
            )
            copied["character_id"] = _mapped_imported_id(
                character_id_map,
                _text(row, "character_id"),
                field_name="character_text_message_attachments.character_id",
            )
            copied["media_asset_id"] = _mapped_optional_media_asset_id(
                media_asset_id_map=live_media_asset_id_map,
                original_id=_optional_text(row, "media_asset_id"),
                field_name="character_text_message_attachments.media_asset_id",
                repair_tracker=repair_tracker,
            )
            character_text_attachments.append(copied)
        _insert_rows(
            connection,
            "character_text_message_attachments",
            character_text_attachments,
        )

        character_text_message_revisions: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("character_text_message_revisions"),
            "character_text_message_revisions",
        ):
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=uuid4().hex,
            )
            copied["text_message_id"] = _mapped_imported_id(
                character_text_message_id_map,
                _text(row, "text_message_id"),
                field_name="character_text_message_revisions.text_message_id",
            )
            character_text_message_revisions.append(copied)
        _insert_rows(
            connection,
            "character_text_message_revisions",
            character_text_message_revisions,
        )

        character_text_provenance: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("character_text_provenance"),
            "character_text_provenance",
        ):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=character_text_provenance_id_map[original_id],
            )
            copied["thread_id"] = _mapped_imported_id(
                character_text_thread_id_map,
                _text(row, "thread_id"),
                field_name="character_text_provenance.thread_id",
            )
            copied["text_message_id"] = _mapped_imported_id(
                character_text_message_id_map,
                _text(row, "text_message_id"),
                field_name="character_text_provenance.text_message_id",
            )
            target_id = _mapped_optional_entity_id(
                entity_id_maps,
                _text(row, "target_type"),
                _text(row, "target_id"),
                field_name="character_text_provenance.target_id",
                repair_tracker=repair_tracker,
            )
            if target_id is None:
                continue
            copied["target_id"] = target_id
            character_text_provenance.append(copied)
        _insert_rows(
            connection,
            "character_text_provenance",
            character_text_provenance,
        )

        character_text_proactive_triggers: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("character_text_proactive_triggers"),
            "character_text_proactive_triggers",
        ):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=character_text_proactive_trigger_id_map[original_id],
            )
            copied["character_id"] = _mapped_imported_id(
                character_id_map,
                _text(row, "character_id"),
                field_name="character_text_proactive_triggers.character_id",
            )
            copied["thread_id"] = _mapped_optional_id(
                character_text_thread_id_map,
                _optional_text(row, "thread_id"),
                field_name="character_text_proactive_triggers.thread_id",
                repair_tracker=repair_tracker,
            )
            copied["text_message_id"] = _mapped_optional_id(
                character_text_message_id_map,
                _optional_text(row, "text_message_id"),
                field_name="character_text_proactive_triggers.text_message_id",
                repair_tracker=repair_tracker,
            )
            copied["source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "source_message_id"),
                field_name="character_text_proactive_triggers.source_message_id",
                repair_tracker=repair_tracker,
            )
            source_type = _optional_text(row, "source_type") or ""
            proactive_source_id = _optional_text(row, "source_id")
            mapped_proactive_source_id = _mapped_optional_entity_id(
                entity_id_maps,
                source_type,
                proactive_source_id,
                field_name="character_text_proactive_triggers.source_id",
                repair_tracker=repair_tracker,
            )
            if proactive_source_id is not None and mapped_proactive_source_id is None:
                copied["source_type"] = ""
            copied["source_id"] = mapped_proactive_source_id or ""
            copied["trigger_key"] = _remapped_character_text_trigger_key(
                _text(row, "trigger_key"),
                entity_id_maps,
            )
            character_text_proactive_triggers.append(copied)
        _insert_rows(
            connection,
            "character_text_proactive_triggers",
            _coalesce_import_proactive_triggers(
                character_text_proactive_triggers
            ),
        )

        character_contact_states: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("character_contact_states"),
            "character_contact_states",
        ):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=character_contact_state_id_map[original_id],
            )
            copied["player_character_id"] = _mapped_imported_id(
                character_id_map,
                _text(row, "player_character_id"),
                field_name="character_contact_states.player_character_id",
            )
            copied["character_id"] = _mapped_imported_id(
                character_id_map,
                _text(row, "character_id"),
                field_name="character_contact_states.character_id",
            )
            copied["source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "source_message_id"),
                field_name="character_contact_states.source_message_id",
                repair_tracker=repair_tracker,
            )
            copied["source_text_message_id"] = _mapped_optional_id(
                character_text_message_id_map,
                _optional_text(row, "source_text_message_id"),
                field_name="character_contact_states.source_text_message_id",
                repair_tracker=repair_tracker,
            )
            character_contact_states.append(copied)
        _insert_rows(
            connection,
            "character_contact_states",
            character_contact_states,
        )
        if not character_contact_states:
            _backfill_imported_character_contact_states(connection, save_id)

        threads: list[dict[str, object]] = []
        for row in _list_of_objects(data.get("active_threads"), "active_threads"):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(row, save_id, new_id=thread_id_map[original_id])
            copied["related_entities_json"] = _remap_related_entities_json(
                row,
                "related_entities_json",
                entity_id_maps,
                field_name="active_threads.related_entities_json",
                repair_tracker=repair_tracker,
            )
            copied["source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "source_message_id"),
                field_name="active_threads.source_message_id",
                repair_tracker=repair_tracker,
            )
            copied["first_seen_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "first_seen_message_id")
                or _optional_text(row, "source_message_id"),
                field_name="active_threads.first_seen_message_id",
                repair_tracker=repair_tracker,
            )
            copied["last_updated_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "last_updated_message_id")
                or _optional_text(row, "source_message_id"),
                field_name="active_threads.last_updated_message_id",
                repair_tracker=repair_tracker,
            )
            threads.append(copied)
        _insert_rows(connection, "active_threads", threads)

        links: list[dict[str, object]] = []
        for row in _list_of_objects(data.get("entity_links"), "entity_links"):
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=entity_link_id_map[_text(row, "id")],
            )
            entity_id = _mapped_optional_entity_id(
                entity_id_maps,
                _text(row, "entity_type"),
                _text(row, "entity_id"),
                field_name="entity_links.entity_id",
                repair_tracker=repair_tracker,
            )
            target_id = _mapped_optional_entity_id(
                entity_id_maps,
                _text(row, "target_type"),
                _text(row, "target_id"),
                field_name="entity_links.target_id",
                repair_tracker=repair_tracker,
            )
            if entity_id is None or target_id is None:
                continue
            copied["entity_id"] = entity_id
            copied["target_id"] = target_id
            copied["source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "source_message_id"),
                field_name="entity_links.source_message_id",
                repair_tracker=repair_tracker,
            )
            links.append(copied)
        links.extend(
            _advisory_character_reference_links(
                data=data,
                save_id=save_id,
                character_id_map=character_id_map,
                media_asset_id_map=live_media_asset_id_map,
                existing_links=links,
            )
        )
        _insert_rows(
            connection,
            "entity_links",
            _coalesce_import_entity_links(links),
        )

        knowledge_edges: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("character_knowledge_edges"),
            "character_knowledge_edges",
        ):
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=knowledge_edge_id_map[_text(row, "id")],
            )
            copied["character_id"] = _mapped_imported_id(
                character_id_map,
                _text(row, "character_id"),
                field_name="character_knowledge_edges.character_id",
            )
            target_id = _mapped_optional_entity_id(
                entity_id_maps,
                _text(row, "target_type"),
                _text(row, "target_id"),
                field_name="character_knowledge_edges.target_id",
                repair_tracker=repair_tracker,
            )
            if target_id is None:
                continue
            copied["target_id"] = target_id
            copied["source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=_optional_text(row, "source_message_id"),
                field_name="character_knowledge_edges.source_message_id",
                repair_tracker=repair_tracker,
            )
            copied["source_message_ids_json"] = _mapped_json_source_refs(
                row,
                "source_message_ids_json",
                message_id_map,
                character_text_message_id_map,
                "character_knowledge_edges.source_message_ids_json",
                repair_tracker,
            )
            knowledge_edges.append(copied)
        _insert_rows(
            connection,
            "character_knowledge_edges",
            _coalesce_import_knowledge_edges(knowledge_edges),
        )

        message_visibility: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("message_visibility"),
            "message_visibility",
        ):
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=message_visibility_id_map[_text(row, "id")],
            )
            copied["message_id"] = _mapped_imported_id(
                message_id_map,
                _text(row, "message_id"),
                field_name="message_visibility.message_id",
            )
            copied["character_id"] = _mapped_imported_id(
                character_id_map,
                _text(row, "character_id"),
                field_name="message_visibility.character_id",
            )
            message_visibility.append(copied)
        _insert_rows(connection, "message_visibility", message_visibility)

        message_scene_presence: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("message_scene_presence"),
            "message_scene_presence",
        ):
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=message_scene_presence_id_map[_text(row, "id")],
            )
            copied["message_id"] = _mapped_imported_id(
                message_id_map,
                _text(row, "message_id"),
                field_name="message_scene_presence.message_id",
            )
            copied["character_id"] = _mapped_imported_id(
                character_id_map,
                _text(row, "character_id"),
                field_name="message_scene_presence.character_id",
            )
            message_scene_presence.append(copied)
        _insert_rows(connection, "message_scene_presence", message_scene_presence)

        suggestions: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("context_update_suggestions"),
            "context_update_suggestions",
        ):
            original_id = _text(row, "id")
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=suggestion_id_map[original_id],
            )
            copied["entity_id"] = _mapped_optional_entity_id(
                entity_id_maps,
                _text(row, "entity_type"),
                _optional_text(row, "entity_id"),
                field_name="context_update_suggestions.entity_id",
                repair_tracker=repair_tracker,
            )
            copied["source_message_ids_json"] = _mapped_json_source_refs(
                row,
                "source_message_ids_json",
                message_id_map,
                character_text_message_id_map,
                "context_update_suggestions.source_message_ids_json",
                repair_tracker,
            )
            copied["proposed_value_json"] = (
                _remapped_context_update_suggestion_proposed_value_json(
                    row,
                    message_id_map,
                    character_text_message_id_map,
                    entity_id_maps,
                    repair_tracker,
                )
            )
            suggestions.append(copied)
        _insert_rows(connection, "context_update_suggestions", suggestions)

        audit_rows: list[dict[str, object]] = []
        for row in _list_of_objects(
            data.get("context_update_audit"),
            "context_update_audit",
        ):
            copied = _copy_row_for_save(
                row,
                save_id,
                new_id=context_update_audit_id_map[_text(row, "id")],
            )
            copied["suggestion_id"] = _mapped_optional_value(
                suggestion_id_map,
                _optional_text(row, "suggestion_id"),
                field_name="context_update_audit.suggestion_id",
                repair_tracker=repair_tracker,
            )
            copied["entity_id"] = _mapped_optional_entity_id(
                entity_id_maps,
                _text(row, "entity_type"),
                _optional_text(row, "entity_id"),
                field_name="context_update_audit.entity_id",
                repair_tracker=repair_tracker,
            )
            copied["source_message_ids_json"] = _mapped_json_source_refs(
                row,
                "source_message_ids_json",
                message_id_map,
                character_text_message_id_map,
                "context_update_audit.source_message_ids_json",
                repair_tracker,
            )
            audit_rows.append(copied)
        _insert_rows(connection, "context_update_audit", audit_rows)

    def _read_manifest(self, bundle_path: Path) -> ChatBundleManifest:
        manifest, _data = self._read_bundle(bundle_path, read_data=False)
        return _manifest_from_payload(manifest)

    def _read_bundle(
        self,
        bundle_path: Path,
        *,
        read_data: bool = True,
    ) -> tuple[dict[str, object], dict[str, object]]:
        try:
            validate_zip_directory(bundle_path)
            with zipfile.ZipFile(bundle_path) as bundle:
                _validate_no_duplicate_bundle_members(bundle)
                manifest = _json_object_from_bytes(
                    _read_limited_member(
                        bundle,
                        MANIFEST_NAME,
                        max_bytes=_MAX_BUNDLE_MANIFEST_JSON_BYTES,
                    ),
                    MANIFEST_NAME,
                )
                _validate_manifest_payload(manifest)
                if not read_data:
                    return manifest, {}
                data = _json_object_from_bytes(
                    _read_limited_member(
                        bundle,
                        DATA_NAME,
                        max_bytes=_MAX_BUNDLE_DATA_JSON_BYTES,
                    ),
                    DATA_NAME,
                )
                _validate_bundle_data(manifest, data)
                _validate_bundle_members(bundle, data)
                return manifest, data
        except (
            OSError,
            KeyError,
            ZipSafetyError,
            zipfile.BadZipFile,
            json.JSONDecodeError,
        ) as exc:
            raise ChatBundleError("Invalid Bragi chat bundle") from exc

    def _rows(self, query: str, params: tuple[object, ...]) -> list[dict[str, object]]:
        return [
            _row_dict(row)
            for row in self.repositories.connection.execute(query, params).fetchall()
        ]

    def _media_asset_rows(self, save_id: str) -> list[dict[str, object]]:
        return self._rows(
            """
            SELECT id, save_id, source_message_id, type, path, thumbnail_path,
                   prompt, provider, model, status, mime_type, metadata_json,
                   source_media_asset_id, created_at, archived_at
            FROM media_assets
            WHERE save_id = ? AND archived_at IS NULL
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )


def _snapshot_only_media_asset_rows(
    rows: Iterable[dict[str, object]],
    *,
    active_media_asset_ids: set[str],
) -> list[dict[str, object]]:
    snapshot_rows: list[dict[str, object]] = []
    seen = set(active_media_asset_ids)
    for row in rows:
        asset_id = _text(row, "id")
        if asset_id in seen:
            continue
        seen.add(asset_id)
        copied = dict(row)
        snapshot_rows.append(copied)
    return snapshot_rows


def _repair_export_media_asset_source_references(data: dict[str, object]) -> None:
    media_assets = _list_of_objects(data.get("media_assets"), "media_assets")
    active_media_asset_ids = {_text(row, "id") for row in media_assets}
    active_media_asset_id_map = {
        asset_id: asset_id for asset_id in active_media_asset_ids
    }
    for row in media_assets:
        source_id = _optional_text(row, "source_media_asset_id")
        if source_id is not None and source_id not in active_media_asset_ids:
            row["source_media_asset_id"] = None
        row["metadata_json"] = _dump_json_compact(
            _remapped_imported_media_source_metadata(
                row,
                active_media_asset_id_map,
            )
        )


def _new_id_map(rows: object, label: str) -> dict[str, str]:
    return {_text(row, "id"): uuid4().hex for row in _list_of_objects(rows, label)}


def _annotate_media_asset_files(row: dict[str, object], media_dir: Path) -> None:
    files: dict[str, dict[str, object]] = {}
    for field in ("path", "thumbnail_path"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            if field == "path":
                raise ChatBundleError(
                    f"Missing media file for chat bundle export: {value}"
                )
            continue
        local_path = _media_path(media_dir, value)
        if local_path is None or not local_path.is_file():
            if field == "path":
                raise ChatBundleError(
                    f"Missing media file for chat bundle export: {value}"
                )
            continue
        byte_count = local_path.stat().st_size
        if byte_count > _MAX_BUNDLE_MEDIA_FILE_BYTES:
            raise ChatBundleError(
                f"Media file is too large for chat bundle export: {value}"
            )
        bundle_path = _bundle_media_path(row["id"], field, value)
        files[field] = {
            "bundle_path": bundle_path,
            "sha256": _sha256(local_path),
            "byte_count": byte_count,
        }
    row["files"] = files


def _annotate_export_media_asset_files(
    rows: Iterable[dict[str, object]],
    media_dir: Path,
) -> None:
    for row in rows:
        _annotate_media_asset_files(row, media_dir)


def _annotate_character_reference_image_asset_ids(data: dict[str, object]) -> None:
    media_asset_ids = {
        _text(row, "id")
        for row in _list_of_objects(data.get("media_assets"), "media_assets")
    }
    character_rows = _list_of_objects(data.get("characters"), "characters")
    character_ids = {_text(row, "id") for row in character_rows}
    reference_by_character_id: dict[str, str] = {}
    for link in _list_of_objects(data.get("entity_links"), "entity_links"):
        if (
            _text(link, "target_type") != "media_asset"
            or _text(link, "relation") != "reference_image"
        ):
            continue
        target_id = _text(link, "target_id")
        if target_id not in media_asset_ids:
            continue
        entity_type = _text(link, "entity_type")
        entity_id = _text(link, "entity_id")
        if entity_type == "character" and entity_id in character_ids:
            reference_by_character_id.setdefault(entity_id, target_id)

    for row in character_rows:
        asset_id = reference_by_character_id.get(_text(row, "id"))
        if asset_id is not None:
            row["reference_image_asset_id"] = asset_id


def _advisory_character_reference_links(
    *,
    data: dict[str, object],
    save_id: str,
    character_id_map: dict[str, str],
    media_asset_id_map: dict[str, str],
    existing_links: list[dict[str, object]],
) -> list[dict[str, object]]:
    existing = {
        (
            _text(link, "entity_type"),
            _text(link, "entity_id"),
            _text(link, "target_type"),
            _text(link, "target_id"),
            _text(link, "relation"),
        )
        for link in existing_links
    }
    links: list[dict[str, object]] = []
    for row in _list_of_objects(data.get("characters"), "characters"):
        original_character_id = _text(row, "id")
        original_asset_id = _optional_text(row, "reference_image_asset_id")
        if original_asset_id is None:
            continue
        character_id = character_id_map.get(original_character_id)
        media_asset_id = media_asset_id_map.get(original_asset_id)
        if character_id is None or media_asset_id is None:
            continue
        key = (
            "character",
            character_id,
            "media_asset",
            media_asset_id,
            "reference_image",
        )
        if key in existing:
            continue
        existing.add(key)
        links.append(
            {
                "id": uuid4().hex,
                "save_id": save_id,
                "entity_type": "character",
                "entity_id": character_id,
                "target_type": "media_asset",
                "target_id": media_asset_id,
                "relation": "reference_image",
                "source_message_id": None,
            }
        )
    return links


def _collect_media_files(
    media_assets: object,
    media_dir: Path,
) -> dict[str, _ExportMediaFile]:
    files: dict[str, _ExportMediaFile] = {}
    total_byte_count = 0
    for row in _list_of_objects(media_assets, "media_assets"):
        asset_id = _text(row, "id")
        row_files = _object(row.get("files"), "media asset files")
        for field, metadata in row_files.items():
            if field not in {"path", "thumbnail_path"}:
                continue
            if not isinstance(metadata, dict):
                raise ChatBundleError(
                    f"Invalid media file metadata for {asset_id}:{field}"
                )
            source = row.get(field)
            bundle_path = metadata.get("bundle_path")
            expected_sha = _required_metadata_text(
                metadata,
                "sha256",
                str(bundle_path),
            )
            expected_byte_count = _required_metadata_int(
                metadata,
                "byte_count",
                str(bundle_path),
            )
            if not isinstance(source, str) or not isinstance(bundle_path, str):
                raise ChatBundleError(
                    f"Invalid media file metadata for {asset_id}:{field}"
                )
            local_path = _media_path(media_dir, source)
            if local_path is None or not local_path.is_file():
                raise ChatBundleError(f"Media file disappeared during export: {source}")
            actual_byte_count = local_path.stat().st_size
            if actual_byte_count != expected_byte_count:
                raise ChatBundleError(f"Media file changed during export: {source}")
            if _sha256(local_path) != expected_sha:
                raise ChatBundleError(f"Media file changed during export: {source}")
            total_byte_count += actual_byte_count
            if total_byte_count > _MAX_BUNDLE_MEDIA_TOTAL_BYTES:
                raise ChatBundleError("Chat bundle media is too large to export")
            files[bundle_path] = _ExportMediaFile(
                path=local_path,
                sha256=expected_sha,
                byte_count=expected_byte_count,
            )
    return files


def _bundle_media_asset_rows(data: dict[str, object]) -> list[dict[str, object]]:
    return [
        *_list_of_objects(data.get("media_assets"), "media_assets"),
        *_list_of_objects(data.get("snapshot_media_assets"), "snapshot_media_assets"),
    ]


def _filter_optional_source_rows(
    rows: object,
    active_message_ids: set[object],
    label: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in _list_of_objects(rows, label)
        if all(
            row.get(field_name) is None or row.get(field_name) in active_message_ids
            for field_name in (
                "source_message_id",
                "first_seen_message_id",
                "last_updated_message_id",
                "world_time_source_message_id",
            )
        )
    ]


def _filter_scene_snapshot_rows(
    rows: object,
    active_message_ids: set[object],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in _list_of_objects(rows, "scene_snapshots"):
        copied = dict(row)
        if (
            copied.get("world_time_source_message_id") is not None
            and copied.get("world_time_source_message_id") not in active_message_ids
        ):
            copied["world_time_source_message_id"] = None
        if all(
            copied.get(field_name) is None
            or copied.get(field_name) in active_message_ids
            for field_name in (
                "source_message_id",
                "first_seen_message_id",
                "last_updated_message_id",
            )
        ):
            filtered.append(copied)
    return filtered


def _filter_context_source_rows(
    rows: object,
    active_source_refs: set[str],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in _list_of_objects(rows, "context_sources"):
        source_ids = _export_context_source_message_ids(row)
        if source_ids is None:
            continue
        if any(source_id not in active_source_refs for source_id in source_ids):
            continue
        filtered.append(row)
    return filtered


def _active_export_source_refs(
    *,
    active_message_ids: set[object],
    exported_text_message_ids: set[object],
) -> set[str]:
    source_refs = {str(message_id) for message_id in active_message_ids}
    source_refs.update(
        character_text_source_ref(str(text_message_id))
        for text_message_id in exported_text_message_ids
    )
    return source_refs


def _export_context_source_message_ids(
    row: dict[str, object],
) -> tuple[str, ...] | None:
    source_ids: list[str] = []
    if row.get("source_type") == "message":
        source_id = row.get("source_id")
        if not isinstance(source_id, str):
            return None
        source_ids.extend(item.strip() for item in source_id.split(",") if item.strip())
    try:
        metadata = _optional_json_object(row, "metadata_json") or {}
    except ChatBundleError:
        return None
    for field in _CONTEXT_SOURCE_METADATA_MESSAGE_ID_FIELDS:
        if field not in metadata:
            continue
        value = metadata[field]
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            return None
        source_ids.append(value)
    for field in _CONTEXT_SOURCE_METADATA_MESSAGE_ID_LIST_FIELDS:
        if field not in metadata:
            continue
        value = metadata[field]
        if not isinstance(value, list):
            return None
        for item in value:
            if not isinstance(item, str) or not item:
                return None
            source_ids.append(item)
    raw_groups = metadata.get("source_provenance_groups")
    if raw_groups is not None:
        if (
            not isinstance(raw_groups, list)
            or len(raw_groups) > _MAX_CONTEXT_SOURCE_PROVENANCE_GROUPS
        ):
            return None
        for group in raw_groups:
            if (
                not isinstance(group, list)
                or not group
                or len(group) > _MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS
            ):
                return None
            for item in group:
                if not isinstance(item, str) or not item:
                    return None
                source_ids.append(item)
    return tuple(dict.fromkeys(source_ids))


def _filter_memory_rows(
    rows: object,
    active_source_refs: set[str],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in _list_of_objects(rows, "memories"):
        source_message_id = row.get("source_message_id")
        if (
            source_message_id is not None
            and str(source_message_id) not in active_source_refs
        ):
            continue
        try:
            source_message_ids = _json_string_list(row, "source_message_ids_json")
        except ChatBundleError:
            source_message_ids = []
        if any(source_id not in active_source_refs for source_id in source_message_ids):
            continue
        filtered.append(row)
    return filtered


def _save_app_setting_rows(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    scenario_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in repositories.list_scoped_settings(scope="save", scope_id=save_id):
        rows.append(
            {
                "scope": "save",
                "key": record.key,
                "value_json": record.value_json,
            }
        )
    for record in repositories.list_scoped_settings(
        scope="scenario",
        scope_id=scenario_id,
    ):
        rows.append(
            {
                "scope": "scenario",
                "key": record.key,
                "value_json": record.value_json,
            }
        )
    return rows


def _apply_save_app_settings(
    repositories: PersistenceRepositories,
    rows: object,
    *,
    save_id: str,
    scenario_id: str,
) -> None:
    for row in _list_of_objects(rows, "save_app_settings"):
        key = _text(row, "key")
        scope = _text(row, "scope")
        value = _json_value(
            _text(row, "value_json"), "save_app_settings.value_json"
        )
        if scope == "save":
            scope_id = save_id
        elif scope == "scenario":
            scope_id = scenario_id
        else:
            raise ChatBundleError(f"Unsupported save app setting scope: {scope}")
        if key == IMAGE_STYLE_PRESET_SETTING and scope == "save":
            value = sanitize_image_style_preset(value)
        if key == SAVE_MODEL_OVERRIDES_SETTING:
            value = sanitize_save_model_overrides(value)
        if key == MODEL_THINKING_PREFERENCES_SETTING:
            value = sanitize_model_thinking_preferences(value)
        if key == POST_TURN_INFERENCE_MODE_SETTING and scope == "save":
            value = sanitize_post_turn_inference_mode(value)
        if scope == "save" and key in {
            RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
            NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
        }:
            value = sanitize_recent_message_window(
                value,
                default=DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW,
            )
        if scope == "save" and key in {
            RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
            NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
        }:
            value = sanitize_recent_message_window(
                value,
                default=DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW,
            )
        repositories.set_scoped_setting(
            scope=scope,
            scope_id=scope_id,
            key=key,
            value=value,
        )


def _filter_save_scenario_updates(
    rows: object,
    active_source_refs: set[str],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in _list_of_objects(rows, "save_scenario_updates"):
        source_message_id = row.get("source_message_id")
        if (
            source_message_id is not None
            and str(source_message_id) not in active_source_refs
        ):
            continue
        source_message_ids = _json_string_list(row, "source_message_ids_json")
        if any(source_id not in active_source_refs for source_id in source_message_ids):
            continue
        filtered.append(_strip_deprecated_scenario_update_content(row))
    return filtered


def _strip_deprecated_scenario_update_content(
    row: dict[str, object],
) -> dict[str, object]:
    cleaned = dict(row)
    cleaned["content_json"] = _dump_json_compact(
        strip_deprecated_scenario_character_sections(
            _json_object(cleaned, "content_json"),
        ),
    )
    return cleaned


def _filter_source_id_list_rows(
    rows: object,
    active_source_refs: set[str],
    label: str,
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in _list_of_objects(rows, label):
        try:
            source_ids = _json_string_list(row, "source_message_ids_json")
        except ChatBundleError:
            continue
        if any(source_id not in active_source_refs for source_id in source_ids):
            continue
        filtered.append(row)
    return filtered


def _exported_knowledge_target_ids(
    data: dict[str, object],
) -> dict[str, set[object]]:
    return {
        "memory": {row["id"] for row in _list_of_objects(data["memories"], "memories")},
        "summary": {
            row["id"] for row in _list_of_objects(data["summaries"], "summaries")
        },
        "world_state": {
            row["id"] for row in _list_of_objects(data["world_state"], "world_state")
        },
    }


def _filter_character_knowledge_edge_rows(
    rows: object,
    *,
    active_source_refs: set[str],
    exported_character_ids: set[object],
    exported_target_ids: dict[str, set[object]],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in _list_of_objects(rows, "character_knowledge_edges"):
        if row.get("character_id") not in exported_character_ids:
            continue
        source_message_id = row.get("source_message_id")
        if (
            source_message_id is not None
            and str(source_message_id) not in active_source_refs
        ):
            continue
        try:
            source_ids = _json_string_list(row, "source_message_ids_json")
        except ChatBundleError:
            continue
        if any(source_id not in active_source_refs for source_id in source_ids):
            continue
        target_type = _export_knowledge_target_type(str(row.get("target_type", "")))
        allowed_target_ids = exported_target_ids.get(target_type)
        if (
            allowed_target_ids is not None
            and row.get("target_id") not in allowed_target_ids
        ):
            continue
        filtered.append(row)
    return filtered


def _filter_message_visibility_rows(
    rows: object,
    *,
    active_message_ids: set[object],
    exported_character_ids: set[object],
) -> list[dict[str, object]]:
    return [
        row
        for row in _list_of_objects(rows, "message_visibility")
        if row.get("message_id") in active_message_ids
        and row.get("character_id") in exported_character_ids
    ]


def _filter_message_presence_rows(
    rows: object,
    *,
    active_message_ids: set[object],
    exported_character_ids: set[object],
) -> list[dict[str, object]]:
    return [
        row
        for row in _list_of_objects(rows, "message_scene_presence")
        if row.get("message_id") in active_message_ids
        and row.get("character_id") in exported_character_ids
    ]


def _filter_dating_route_state_rows(
    rows: object,
    *,
    active_message_ids: set[object],
    exported_character_ids: set[object],
) -> list[dict[str, object]]:
    return [
        row
        for row in _list_of_objects(rows, "dating_route_states")
        if row.get("player_character_id") in exported_character_ids
        and row.get("npc_character_id") in exported_character_ids
        and all(
            row.get(field_name) is None or row.get(field_name) in active_message_ids
            for field_name in (
                "first_met_message_id",
                "last_interaction_message_id",
                "source_message_id",
            )
        )
    ]


def _filter_character_text_rows(
    *,
    threads: object,
    participants: object,
    messages: object,
    provenance: object,
    exported_character_ids: set[object],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    participant_rows = [
        row
        for row in _list_of_objects(
            participants,
            "character_text_thread_participants",
        )
        if row.get("character_id") in exported_character_ids
    ]
    participant_count_by_thread: dict[object, int] = {}
    for row in participant_rows:
        thread_id = row.get("thread_id")
        participant_count_by_thread[thread_id] = (
            participant_count_by_thread.get(thread_id, 0) + 1
        )
    filtered_threads = []
    for row in _list_of_objects(threads, "character_text_threads"):
        if row.get("kind") == "group":
            if participant_count_by_thread.get(row.get("id"), 0) >= 2:
                filtered_threads.append(row)
            continue
        if row.get("character_id") in exported_character_ids:
            filtered_threads.append(row)
    exported_thread_ids = {row["id"] for row in filtered_threads}
    filtered_participants = [
        row for row in participant_rows if row.get("thread_id") in exported_thread_ids
    ]
    filtered_messages = [
        row
        for row in _list_of_objects(messages, "character_text_messages")
        if row.get("thread_id") in exported_thread_ids
        and (
            row.get("character_id") is None
            or row.get("character_id") in exported_character_ids
        )
        and (
            row.get("sender_character_id") is None
            or row.get("sender_character_id") in exported_character_ids
        )
    ]
    exported_message_ids = {row["id"] for row in filtered_messages}
    filtered_provenance = [
        row
        for row in _list_of_objects(provenance, "character_text_provenance")
        if row.get("thread_id") in exported_thread_ids
        and row.get("text_message_id") in exported_message_ids
    ]
    return (
        filtered_threads,
        filtered_participants,
        filtered_messages,
        filtered_provenance,
    )


def _filter_character_text_attachment_rows(
    rows: object,
    *,
    exported_character_ids: set[object],
    exported_thread_ids: set[object],
    exported_text_message_ids: set[object],
) -> list[dict[str, object]]:
    return [
        row
        for row in _list_of_objects(rows, "character_text_message_attachments")
        if row.get("thread_id") in exported_thread_ids
        and row.get("text_message_id") in exported_text_message_ids
        and row.get("character_id") in exported_character_ids
    ]


def _filter_character_text_attachment_media_rows(
    rows: object,
    *,
    exported_text_attachment_media_ids: set[object],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in _list_of_objects(rows, "media_assets"):
        kind = _media_row_kind(row)
        if kind.startswith("character_text_"):
            if row.get("id") in exported_text_attachment_media_ids:
                filtered.append(row)
            continue
        filtered.append(row)
    return filtered


def _media_row_kind(row: dict[str, object]) -> str:
    try:
        metadata = _optional_json_object(row, "metadata_json") or {}
    except ChatBundleError:
        return ""
    kind = metadata.get("kind")
    return kind if isinstance(kind, str) else ""


def _filter_character_contact_state_rows(
    rows: object,
    *,
    exported_character_ids: set[object],
    active_message_ids: set[object],
    exported_text_message_ids: set[object],
) -> list[dict[str, object]]:
    return [
        row
        for row in _list_of_objects(rows, "character_contact_states")
        if row.get("player_character_id") in exported_character_ids
        and row.get("character_id") in exported_character_ids
        and (
            row.get("source_message_id") in active_message_ids
            or row.get("source_message_id") in {None, ""}
        )
        and (
            row.get("source_text_message_id") in exported_text_message_ids
            or row.get("source_text_message_id") in {None, ""}
        )
    ]


def _filter_character_text_proactive_trigger_rows(
    rows: object,
    *,
    exported_character_ids: set[object],
    active_message_ids: set[object],
    exported_thread_ids: set[object],
    exported_text_message_ids: set[object],
) -> list[dict[str, object]]:
    return [
        row
        for row in _list_of_objects(rows, "character_text_proactive_triggers")
        if row.get("character_id") in exported_character_ids
        and (
            row.get("thread_id") in exported_thread_ids
            or row.get("thread_id") in {None, ""}
        )
        and (
            row.get("text_message_id") in exported_text_message_ids
            or row.get("text_message_id") in {None, ""}
        )
        and (
            row.get("source_message_id") in active_message_ids
            or row.get("source_message_id") in {None, ""}
        )
    ]


def _filter_message_action_choice_rows(
    rows: object,
    *,
    active_message_ids: set[object],
) -> list[dict[str, object]]:
    return [
        row
        for row in _list_of_objects(rows, "message_action_choices")
        if row.get("message_id") in active_message_ids
    ]


def _export_knowledge_target_type(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"state", "world_state"}:
        return "world_state"
    return normalized


def _filter_loss_outcomes(
    rows: object,
    active_message_ids: set[object],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in _list_of_objects(rows, "save_loss_outcomes"):
        if row.get("triggering_message_id") not in active_message_ids:
            continue
        epilogue_message_id = row.get("epilogue_message_id")
        if (
            epilogue_message_id is not None
            and epilogue_message_id not in active_message_ids
        ):
            continue
        evidence = row.get("evidence_json")
        if not isinstance(evidence, str):
            continue
        try:
            evidence_source_ids = _evidence_source_message_ids(
                _json_value(evidence, "save_loss_outcomes.evidence_json")
            )
        except ChatBundleError:
            continue
        stale_evidence = any(
            source_id not in active_message_ids for source_id in evidence_source_ids
        )
        if stale_evidence:
            continue
        filtered.append(row)
    return filtered


def _evidence_source_message_ids(value: object) -> set[str]:
    if isinstance(value, dict):
        source_ids = {
            nested
            for key, nested in value.items()
            if key == "source_message_id" and isinstance(nested, str)
        }
        for nested in value.values():
            source_ids.update(_evidence_source_message_ids(nested))
        return source_ids
    if isinstance(value, list):
        list_source_ids: set[str] = set()
        for nested in value:
            list_source_ids.update(_evidence_source_message_ids(nested))
        return list_source_ids
    return set()


def _write_bundle_atomically(
    *,
    bundle_path: Path,
    manifest_payload: dict[str, object],
    data: dict[str, object],
    media_files: dict[str, _ExportMediaFile],
) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            prefix=f".{bundle_path.name}.",
            suffix=".tmp",
            dir=bundle_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as bundle:
            bundle.writestr(MANIFEST_NAME, _dump_json_pretty(manifest_payload))
            bundle.writestr(DATA_NAME, _dump_json_pretty(data))
            for bundle_name, media_file in media_files.items():
                _write_verified_media_member(bundle, bundle_name, media_file)
        temporary_path.chmod(0o600)
        os.replace(temporary_path, bundle_path)
        bundle_path.chmod(0o600)
    except Exception:
        if temporary_path is not None:
            try:
                if temporary_path.exists():
                    temporary_path.unlink()
            except OSError:
                pass
        raise


def _write_verified_media_member(
    bundle: zipfile.ZipFile,
    bundle_name: str,
    media_file: _ExportMediaFile,
) -> None:
    digest = hashlib.sha256()
    written = 0
    with media_file.path.open("rb") as source:
        with bundle.open(bundle_name, "w") as target:
            while chunk := source.read(_BUNDLE_MEDIA_COPY_CHUNK_BYTES):
                written += len(chunk)
                if written > media_file.byte_count:
                    raise ChatBundleError(
                        f"Media file changed during export: {media_file.path}"
                    )
                digest.update(chunk)
                target.write(chunk)
    if written != media_file.byte_count or digest.hexdigest() != media_file.sha256:
        raise ChatBundleError(f"Media file changed during export: {media_file.path}")


def _load_media_members(
    bundle_path: Path,
    data: dict[str, object],
) -> dict[tuple[str, str], _BundleMediaMember]:
    members: dict[tuple[str, str], _BundleMediaMember] = {}
    total_byte_count = 0
    try:
        with zipfile.ZipFile(bundle_path) as bundle:
            for row in _bundle_media_asset_rows(data):
                asset_id = _text(row, "id")
                files = _object(row.get("files"), "media asset files")
                if "path" not in files:
                    raise ChatBundleError(
                        "Missing primary media file metadata for media asset: "
                        f"{asset_id}"
                    )
                for field in ("path", "thumbnail_path"):
                    metadata = files.get(field)
                    if metadata is None:
                        continue
                    if not isinstance(metadata, dict):
                        raise ChatBundleError(
                            f"Invalid media file metadata for {asset_id}:{field}"
                        )
                    bundle_name = metadata.get("bundle_path")
                    if not isinstance(bundle_name, str) or not bundle_name:
                        raise ChatBundleError(
                            f"Missing media bundle_path metadata for {asset_id}:{field}"
                        )
                    _validate_bundle_member_name(bundle_name)
                    expected_sha = _required_metadata_text(
                        metadata,
                        "sha256",
                        bundle_name,
                    )
                    expected_byte_count = _required_metadata_int(
                        metadata,
                        "byte_count",
                        bundle_name,
                    )
                    if expected_byte_count > _MAX_BUNDLE_MEDIA_FILE_BYTES:
                        raise ChatBundleError(
                            f"Media file is too large in chat bundle: {bundle_name}"
                        )
                    total_byte_count += expected_byte_count
                    if total_byte_count > _MAX_BUNDLE_MEDIA_TOTAL_BYTES:
                        raise ChatBundleError("Chat bundle media is too large")
                    try:
                        info = bundle.getinfo(bundle_name)
                    except KeyError:
                        raise ChatBundleError(
                            f"Missing media file in chat bundle: {bundle_name}"
                        ) from None
                    if info.file_size != expected_byte_count:
                        raise ChatBundleError(
                            f"Media byte count mismatch for {bundle_name}"
                        )
                    members[(asset_id, field)] = _BundleMediaMember(
                        bundle_name=bundle_name,
                        sha256=expected_sha,
                        byte_count=expected_byte_count,
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ChatBundleError("Invalid Bragi chat bundle") from exc
    return members


def _copy_bundle_media_member(
    bundle_path: Path,
    member: _BundleMediaMember,
    destination_path: Path,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.is_symlink():
        raise OSError(
            f"Refusing to replace symlink as private file: {destination_path}"
        )

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            digest = hashlib.sha256()
            written = 0
            try:
                with zipfile.ZipFile(bundle_path) as bundle:
                    with bundle.open(member.bundle_name) as source:
                        while chunk := source.read(_BUNDLE_MEDIA_COPY_CHUNK_BYTES):
                            written += len(chunk)
                            if written > member.byte_count:
                                raise ChatBundleError(
                                    "Media byte count mismatch for "
                                    f"{member.bundle_name}"
                                )
                            digest.update(chunk)
                            destination.write(chunk)
            except KeyError:
                raise ChatBundleError(
                    f"Missing media file in chat bundle: {member.bundle_name}"
                ) from None
            except zipfile.BadZipFile as exc:
                raise ChatBundleError("Invalid Bragi chat bundle") from exc
            if written != member.byte_count:
                raise ChatBundleError(
                    f"Media byte count mismatch for {member.bundle_name}"
                )
            if digest.hexdigest() != member.sha256:
                raise ChatBundleError(
                    f"Media checksum mismatch for {member.bundle_name}"
                )
        temporary_path.chmod(0o600)
        os.replace(temporary_path, destination_path)
    except Exception:
        if temporary_path is not None:
            try:
                if temporary_path.exists():
                    temporary_path.unlink()
            except OSError:
                pass
        raise


def _read_limited_member(
    bundle: zipfile.ZipFile,
    name: str,
    *,
    max_bytes: int,
) -> bytes:
    info = bundle.getinfo(name)
    if info.file_size > max_bytes:
        raise ChatBundleError(f"Chat bundle {name} is too large")
    _validate_bundle_member_compression_ratio(info)
    payload = bytearray()
    with bundle.open(info) as source:
        while chunk := source.read(_BUNDLE_JSON_COPY_CHUNK_BYTES):
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise ChatBundleError(f"Chat bundle {name} is too large")
    return bytes(payload)


def _validate_no_duplicate_bundle_members(bundle: zipfile.ZipFile) -> None:
    seen: set[str] = set()
    for info in bundle.infolist():
        if info.filename in seen:
            raise ChatBundleError(f"Duplicate chat bundle member: {info.filename}")
        seen.add(info.filename)


def _validate_bundle_member_compression_ratio(info: zipfile.ZipInfo) -> None:
    if info.file_size < _MIN_BUNDLE_COMPRESSION_RATIO_CHECK_BYTES:
        return
    if info.compress_size <= 0:
        raise ChatBundleError(
            f"Chat bundle member is suspiciously compressed: {info.filename}"
        )
    ratio = info.file_size / info.compress_size
    if ratio > _MAX_BUNDLE_COMPRESSION_RATIO:
        raise ChatBundleError(
            f"Chat bundle member is suspiciously compressed: {info.filename}"
        )


def _validate_total_decompressed_size(total_byte_count: int) -> None:
    if total_byte_count > _MAX_BUNDLE_TOTAL_DECOMPRESSED_BYTES:
        raise ChatBundleError("Chat bundle is too large")


def _validate_total_json_size(total_byte_count: int) -> None:
    if total_byte_count > _MAX_BUNDLE_JSON_TOTAL_BYTES:
        raise ChatBundleError("Chat bundle JSON is too large")


def _validate_bundle_members(
    bundle: zipfile.ZipFile,
    data: dict[str, object],
) -> None:
    expected = {MANIFEST_NAME, DATA_NAME}
    total_decompressed_byte_count = (
        bundle.getinfo(MANIFEST_NAME).file_size + bundle.getinfo(DATA_NAME).file_size
    )
    _validate_total_json_size(total_decompressed_byte_count)
    _validate_total_decompressed_size(total_decompressed_byte_count)
    total_byte_count = 0
    for row in _bundle_media_asset_rows(data):
        asset_id = _text(row, "id")
        files = _object(row.get("files"), "media asset files")
        for field in ("path", "thumbnail_path"):
            metadata = files.get(field)
            if metadata is None:
                continue
            if not isinstance(metadata, dict):
                raise ChatBundleError(
                    f"Invalid media file metadata for {asset_id}:{field}"
                )
            bundle_name = metadata.get("bundle_path")
            if not isinstance(bundle_name, str) or not bundle_name:
                raise ChatBundleError(
                    f"Missing media bundle_path metadata for {asset_id}:{field}"
                )
            _validate_bundle_member_name(bundle_name)
            expected_byte_count = _required_metadata_int(
                metadata,
                "byte_count",
                bundle_name,
            )
            if expected_byte_count > _MAX_BUNDLE_MEDIA_FILE_BYTES:
                raise ChatBundleError(
                    f"Media file is too large in chat bundle: {bundle_name}"
                )
            total_byte_count += expected_byte_count
            if total_byte_count > _MAX_BUNDLE_MEDIA_TOTAL_BYTES:
                raise ChatBundleError("Chat bundle media is too large")
            try:
                info = bundle.getinfo(bundle_name)
            except KeyError:
                raise ChatBundleError(
                    f"Missing media file in chat bundle: {bundle_name}"
                ) from None
            if info.file_size != expected_byte_count:
                raise ChatBundleError(
                    f"Media byte count mismatch for {bundle_name}"
                )
            _validate_bundle_member_compression_ratio(info)
            total_decompressed_byte_count += info.file_size
            _validate_total_decompressed_size(total_decompressed_byte_count)
            expected.add(bundle_name)

    seen: set[str] = set()
    for info in bundle.infolist():
        if info.filename in seen:
            raise ChatBundleError(f"Duplicate chat bundle member: {info.filename}")
        seen.add(info.filename)
        if info.filename not in expected:
            raise ChatBundleError(f"Unexpected chat bundle member: {info.filename}")


def _validate_bundle_data(
    manifest: dict[str, object],
    data: dict[str, object],
) -> None:
    total_rows = 0
    for table_name, value in data.items():
        if not isinstance(value, list):
            continue
        row_count = len(value)
        if table_name == "messages" and row_count > _MAX_BUNDLE_MESSAGE_ROWS:
            raise ChatBundleError("Chat bundle contains too many messages")
        if row_count > _MAX_BUNDLE_TABLE_ROWS:
            raise ChatBundleError(
                f"Chat bundle table has too many rows: {table_name}"
            )
        total_rows += row_count
        if total_rows > _MAX_BUNDLE_TOTAL_ROWS:
            raise ChatBundleError("Chat bundle contains too many rows")
    save = _object(data.get("save"), "save")
    scenario = _object(data.get("scenario"), "scenario")
    messages = _list_of_objects(data.get("messages"), "messages")
    media_assets = _list_of_objects(data.get("media_assets"), "media_assets")
    text_attachments = _list_of_objects(
        data.get("character_text_message_attachments"),
        "character_text_message_attachments",
    )
    snapshot_media_assets = _list_of_objects(
        data.get("snapshot_media_assets"),
        "snapshot_media_assets",
    )
    turn_snapshots = _list_of_objects(
        data.get("turn_snapshots"),
        "turn_snapshots",
    )
    snapshot_objects = _list_of_objects(
        data.get("snapshot_objects"),
        "snapshot_objects",
    )
    manifest_save = _object(manifest.get("save"), "manifest save")
    manifest_scenario = _object(manifest.get("scenario"), "manifest scenario")
    manifest_counts = _object(manifest.get("counts"), "manifest counts")
    scenario_type = _text(scenario, "type")
    scenario_content = _json_object(scenario, "content_json")
    if scenario_record_is_retired(scenario_type, scenario_content):
        raise ChatBundleError(RETIRED_SCENARIO_REASON)
    if _text(save, "id") != _text(manifest_save, "id"):
        raise ChatBundleError("Chat bundle manifest does not match save data")
    if _text(scenario, "id") != _text(manifest_scenario, "id"):
        raise ChatBundleError("Chat bundle manifest does not match scenario data")
    if _text(save, "title") != _text(manifest_save, "title"):
        raise ChatBundleError("Chat bundle manifest does not match save data")
    if _text(scenario, "title") != _text(manifest_scenario, "title"):
        raise ChatBundleError("Chat bundle manifest does not match scenario data")
    if len(messages) != _int(manifest_counts, "messages"):
        raise ChatBundleError("Chat bundle manifest message count does not match data")
    if len(media_assets) != _int(manifest_counts, "media_assets"):
        raise ChatBundleError("Chat bundle manifest media count does not match data")
    _validate_unique_ids(messages, "messages")
    _validate_unique_ids(media_assets, "media_assets")
    try:
        TurnSnapshotService.validate_exported_snapshot_rows(
            snapshot_rows=turn_snapshots,
            object_rows=snapshot_objects,
        )
    except ValueError as exc:
        raise ChatBundleError(str(exc)) from exc
    _validate_unique_ids(
        text_attachments,
        "character_text_message_attachments",
    )
    _validate_unique_ids(snapshot_media_assets, "snapshot_media_assets")
    active_media_asset_ids = {_text(row, "id") for row in media_assets}
    for row in snapshot_media_assets:
        if _text(row, "id") in active_media_asset_ids:
            raise ChatBundleError(
                "Snapshot media asset duplicates active media asset id: "
                f"{_text(row, 'id')}"
            )


def _validate_unique_ids(rows: list[dict[str, object]], label: str) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = _text(row, "id")
        if row_id in seen:
            raise ChatBundleError(f"Duplicate {label} id in chat bundle: {row_id}")
        seen.add(row_id)


def _validate_bundle_member_name(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not name.startswith("media/"):
        raise ChatBundleError("Invalid media path in chat bundle")


def _required_metadata_text(
    metadata: dict[object, object],
    key: str,
    bundle_name: str,
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ChatBundleError(f"Missing media {key} metadata for {bundle_name}")
    return value


def _required_metadata_int(
    metadata: dict[object, object],
    key: str,
    bundle_name: str,
) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or value < 0:
        raise ChatBundleError(f"Missing media {key} metadata for {bundle_name}")
    return value


def _manifest_from_payload(payload: dict[str, object]) -> ChatBundleManifest:
    _validate_manifest_payload(payload)
    save = _object(payload.get("save"), "manifest save")
    scenario = _object(payload.get("scenario"), "manifest scenario")
    counts = _object(payload.get("counts"), "manifest counts")
    return ChatBundleManifest(
        bundle_format=BUNDLE_FORMAT,
        bundle_version=BUNDLE_VERSION,
        save_id=_text(save, "id"),
        title=_text(save, "title"),
        scenario_title=_text(scenario, "title"),
        message_count=_int(counts, "messages"),
        media_count=_int(counts, "media_assets"),
        created_at=_optional_text(save, "created_at"),
        updated_at=_optional_text(save, "updated_at"),
        exported_at=_text(payload, "exported_at"),
    )


def _validate_manifest_payload(payload: dict[str, object]) -> None:
    bundle_format = payload.get("format")
    if bundle_format != BUNDLE_FORMAT:
        raise ChatBundleError("Not a Bragi chat bundle")
    version = payload.get("bundle_version")
    if version != BUNDLE_VERSION:
        raise ChatBundleError(f"Unsupported Bragi chat bundle version: {version}")
    schema_version = payload.get("bragi_schema_version", payload.get("schema_version"))
    if (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version > CURRENT_SCHEMA_VERSION
    ):
        raise ChatBundleError(
            "Bragi chat bundle requires a newer database schema: "
            f"{schema_version}"
        )


def _json_object_from_bytes(payload: bytes, name: str) -> dict[str, object]:
    try:
        validate_json_structure(
            payload,
            max_nodes=_MAX_BUNDLE_JSON_NODES,
            max_depth=_MAX_BUNDLE_JSON_DEPTH,
        )
    except JsonSafetyError as exc:
        raise ChatBundleError(f"{name} {exc}") from exc
    object_count = 0

    def bounded_object(value: dict[str, object]) -> dict[str, object]:
        nonlocal object_count
        object_count += 1
        if object_count > _MAX_BUNDLE_JSON_OBJECTS:
            raise ChatBundleError(f"{name} contains too many objects")
        return value

    loaded = json.loads(
        payload.decode("utf-8"),
        object_hook=bounded_object,
    )
    if not isinstance(loaded, dict):
        raise ChatBundleError(f"{name} must contain a JSON object")
    return cast(dict[str, object], loaded)


def _row_dict(row: Any) -> dict[str, object]:
    return {key: row[key] for key in row.keys()}


def _require_row(row: Any | None, message: str) -> Any:
    if row is None:
        raise ValueError(message)
    return row


def _media_path(media_dir: Path, persisted_path: str) -> Path | None:
    path = Path(persisted_path)
    candidate = path if path.is_absolute() else media_dir / path
    try:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(media_dir.resolve()):
            return None
    except OSError:
        return None
    return resolved


def _bundle_media_path(asset_id: object, field: str, persisted_path: str) -> str:
    filename = _safe_filename(persisted_path)
    return f"media/{_safe_path_segment(str(asset_id))}/{field}/{filename}"


def _imported_media_relative_path(
    *,
    save_id: str,
    asset_id: str,
    original_path: str,
) -> Path:
    return (
        Path(_safe_path_segment(save_id))
        / "imports"
        / _safe_path_segment(asset_id)
        / _safe_filename(original_path)
    )


def _imported_thumbnail_relative_path(
    *,
    save_id: str,
    asset_id: str,
    original_path: str,
) -> Path:
    return (
        Path(_safe_path_segment(save_id))
        / "imports"
        / _safe_path_segment(asset_id)
        / "thumbnails"
        / _safe_filename(original_path)
    )


def _safe_path_segment(value: str) -> str:
    return quote(value, safe="").replace(".", "%2E")


def _safe_filename(value: str) -> str:
    name = Path(value).name
    if not name or name in {".", ".."}:
        return "asset.bin"
    return quote(name, safe="._-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapped_required(mapping: dict[str, str], original_id: str) -> str:
    try:
        return mapping[original_id]
    except KeyError as exc:
        raise ChatBundleError(
            f"Bundle references unknown message id: {original_id}"
        ) from exc


def _mapped_imported_id(
    mapping: dict[str, str],
    original_id: str,
    *,
    field_name: str,
) -> str:
    try:
        return mapping[original_id]
    except KeyError as exc:
        raise ChatBundleError(
            f"Bundle {field_name} references unknown id: {original_id}"
        ) from exc


def _mapped_optional_required(
    *,
    message_id_map: dict[str, str],
    original_id: str | None,
    field_name: str,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    if original_id is None:
        return None
    try:
        return message_id_map[original_id]
    except KeyError:
        if repair_tracker is not None:
            repair_tracker.record(field_name)
        return None


def _mapped_optional_id(
    mapping: dict[str, str],
    original_id: str | None,
    *,
    field_name: str,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    if original_id is None:
        return None
    try:
        return mapping[original_id]
    except KeyError:
        if repair_tracker is not None:
            repair_tracker.record(field_name)
        return None


def _ordered_media_asset_import_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows_by_id = {_text(row, "id"): row for row in rows}
    ordered: list[dict[str, object]] = []
    pending = list(rows)
    imported_ids: set[str] = set()
    while pending:
        next_pending: list[dict[str, object]] = []
        progressed = False
        for row in pending:
            original_id = _text(row, "id")
            source_id = _optional_text(row, "source_media_asset_id")
            if source_id is None or source_id in imported_ids:
                ordered.append(row)
                imported_ids.add(original_id)
                progressed = True
                continue
            if source_id not in rows_by_id:
                ordered.append(row)
                imported_ids.add(original_id)
                progressed = True
                continue
            next_pending.append(row)
        if not progressed:
            raise ChatBundleError("Bundle media asset source links contain a cycle")
        pending = next_pending
    return ordered


def _mapped_optional_media_asset_id(
    *,
    media_asset_id_map: dict[str, str],
    original_id: str | None,
    field_name: str = "media_assets.source_media_asset_id",
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    if original_id is None:
        return None
    try:
        return media_asset_id_map[original_id]
    except KeyError:
        if repair_tracker is not None:
            repair_tracker.record(field_name)
        return None


def _remap_imported_media_reference_metadata(
    connection: Any,
    *,
    data: dict[str, object],
    media_asset_id_map: dict[str, str],
    character_id_map: dict[str, str],
    character_text_thread_id_map: dict[str, str],
    character_text_message_id_map: dict[str, str],
) -> None:
    for row in _list_of_objects(data.get("media_assets"), "media_assets"):
        original_asset_id = _text(row, "id")
        imported_asset_id = media_asset_id_map.get(original_asset_id)
        if imported_asset_id is None:
            continue
        metadata = _quarantined_imported_media_source_metadata(
            row,
            media_asset_id_map,
        )
        remapped = dict(metadata)
        if metadata.get("kind") == "character_reference":
            character_id = metadata.get("character_id")
            if (
                not isinstance(character_id, str)
                or character_id not in character_id_map
            ):
                continue
            remapped["character_id"] = character_id_map[character_id]
        elif _media_row_kind(row).startswith("character_text_"):
            character_id = metadata.get("character_id")
            thread_id = metadata.get("thread_id")
            text_message_id = metadata.get("text_message_id")
            if isinstance(character_id, str) and character_id in character_id_map:
                remapped["character_id"] = character_id_map[character_id]
            if (
                isinstance(thread_id, str)
                and thread_id in character_text_thread_id_map
            ):
                remapped["thread_id"] = character_text_thread_id_map[thread_id]
            if (
                isinstance(text_message_id, str)
                and text_message_id in character_text_message_id_map
            ):
                remapped["text_message_id"] = character_text_message_id_map[
                    text_message_id
                ]
        else:
            continue
        connection.execute(
            """
            UPDATE media_assets
            SET metadata_json = ?
            WHERE id = ?
            """,
            (_dump_json_compact(remapped), imported_asset_id),
        )


_MEDIA_METADATA_SOURCE_ID_FIELDS = (
    "source_character_reference_asset_id",
    "source_media_asset_id",
)
_MEDIA_METADATA_SOURCE_ID_LIST_FIELDS = (
    "source_character_reference_asset_ids",
    "source_media_asset_ids",
)
_CONTEXT_SOURCE_ENTITY_ID_SOURCE_TYPES = frozenset(
    {
        "active_thread",
        "character",
        "character_voice",
        "character_text_message",
        "character_text_thread",
        "dating_route_state",
        "location",
        "media_asset",
        "memory",
        "observation",
        "open_obligation",
        "scene_snapshot",
        "state_change",
        "summary",
    }
)
_CONTEXT_UPDATE_SUGGESTION_LOCATION_FIELD_PATHS = frozenset(
    {
        ("character", "location_id"),
        ("location", "parent_location_id"),
        ("scene_snapshot", "current_location_id"),
    }
)


def _remapped_imported_media_source_metadata(
    row: dict[str, object],
    media_asset_id_map: dict[str, str],
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> dict[str, object]:
    metadata = _optional_json_object(row, "metadata_json") or {}
    remapped = dict(metadata)
    for field in _MEDIA_METADATA_SOURCE_ID_FIELDS:
        if field not in remapped:
            continue
        value = remapped[field]
        if value is None:
            continue
        if isinstance(value, str) and value:
            remapped[field] = _mapped_optional_media_asset_id(
                media_asset_id_map=media_asset_id_map,
                original_id=value,
                field_name=f"media_assets.metadata_json.{field}",
                repair_tracker=repair_tracker,
            )
    for field in _MEDIA_METADATA_SOURCE_ID_LIST_FIELDS:
        if field not in remapped:
            continue
        value = remapped[field]
        if not isinstance(value, list):
            continue
        mapped_items: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item:
                continue
            mapped = _mapped_optional_media_asset_id(
                media_asset_id_map=media_asset_id_map,
                original_id=item,
                field_name=f"media_assets.metadata_json.{field}",
                repair_tracker=repair_tracker,
            )
            if mapped is not None:
                mapped_items.append(mapped)
        remapped[field] = mapped_items
    return remapped


def _quarantined_imported_media_source_metadata(
    row: dict[str, object],
    media_asset_id_map: dict[str, str],
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> dict[str, object]:
    metadata = _remapped_imported_media_source_metadata(
        row,
        media_asset_id_map,
        repair_tracker,
    )
    metadata["content_rating"] = "unclassified"
    return metadata


def _quarantine_imported_scenario_content(
    content: Mapping[str, object],
) -> dict[str, object]:
    quarantined = dict(content)
    source = quarantined.get("_source")
    quarantined["_source"] = metadata_with_scenario_content_ratings(
        source if isinstance(source, Mapping) else None,
        aggregate_rating="unclassified",
    )
    starters = quarantined.get("character_starters")
    if isinstance(starters, list):
        for starter in starters:
            if not isinstance(starter, dict):
                continue
            reference = starter.get("reference_image")
            if isinstance(reference, dict):
                reference["content_rating"] = "unclassified"
    return quarantined


def _mapped_optional_value(
    mapping: dict[str, str],
    original_id: str | None,
    *,
    field_name: str,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    if original_id is None:
        return None
    mapped = mapping.get(original_id)
    if mapped is None:
        if repair_tracker is not None:
            repair_tracker.record(field_name)
        return None
    return mapped


def _mapped_entity_id(
    mappings: dict[str, dict[str, str]],
    entity_type: str,
    entity_id: str,
) -> str:
    return mappings.get(entity_type, {}).get(entity_id, entity_id)


def _remapped_character_text_trigger_key(
    trigger_key: str,
    mappings: dict[str, dict[str, str]],
) -> str:
    parts = trigger_key.split(":")
    if len(parts) < 2:
        return trigger_key
    if parts[0] == "ambient_random" and len(parts) >= 3:
        remapped = list(parts)
        remapped[1] = _mapped_entity_id(mappings, "message", remapped[1])
        remapped[2] = _mapped_entity_id(mappings, "character", remapped[2])
        return ":".join(remapped)
    source_type = {
        "active_thread": "active_thread",
        "dating_route": "dating_route_state",
        "character_intent": "character",
        "memory": "memory",
        "memories": "memory",
    }.get(parts[0])
    if source_type is None:
        return trigger_key
    remapped = list(parts)
    remapped[1] = _mapped_entity_id(mappings, source_type, remapped[1])
    if parts[0] in {"active_thread", "character_intent"} and len(parts) >= 3:
        remapped[2] = _mapped_entity_id(mappings, "message", remapped[2])
    return ":".join(remapped)


def _remapped_imported_id_fragment(
    value: str,
    mappings: dict[str, dict[str, str]],
) -> str:
    for mapping in mappings.values():
        mapped = mapping.get(value)
        if mapped is not None:
            return mapped
    source_type, separator, source_id = value.partition(":")
    if not separator:
        return value
    return f"{source_type}:{_mapped_entity_id(mappings, source_type, source_id)}"


def _mapped_context_source_id(
    mappings: dict[str, dict[str, str]],
    source_type: str,
    source_id: str,
    *,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    if source_type == "message":
        mapped_items: list[str] = []
        for item in source_id.split(","):
            item = item.strip()
            if not item:
                continue
            mapped = _mapped_optional_source_ref(
                item.strip(),
                mappings.get("message", {}),
                mappings.get("character_text_message", {}),
                "context_sources.source_id",
                repair_tracker,
            )
            if mapped is not None:
                mapped_items.append(mapped)
        if not mapped_items and source_id.strip():
            return None
        return ",".join(mapped_items)
    if source_type == "scenario_section":
        return _mapped_scenario_section_context_source_id(
            mappings,
            source_id,
            repair_tracker=repair_tracker,
        )
    mapping = mappings.get(source_type)
    if mapping is not None:
        mapped_source_id = mapping.get(source_id)
        if mapped_source_id is not None:
            return mapped_source_id
    if source_type == "memory":
        return _mapped_memory_context_source_id(
            mappings,
            source_id,
            repair_tracker=repair_tracker,
        )
    if source_type == "world_state":
        return _mapped_world_state_context_source_id(
            mappings,
            source_id,
            repair_tracker=repair_tracker,
        )
    if source_type not in _CONTEXT_SOURCE_ENTITY_ID_SOURCE_TYPES:
        return source_id
    mapped_items = []
    for item in source_id.split(","):
        item = item.strip()
        if not item:
            continue
        mapped = _mapped_optional_entity_id(
            mappings,
            source_type,
            item,
            field_name="context_sources.source_id",
            repair_tracker=repair_tracker,
        )
        if mapped is not None:
            mapped_items.append(mapped)
    if not mapped_items and source_id.strip():
        return None
    return ",".join(mapped_items)


def _mapped_scenario_section_context_source_id(
    mappings: dict[str, dict[str, str]],
    source_id: str,
    *,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    parts = source_id.split(":", 3)
    if len(parts) != 4 or parts[0] != "scenario" or parts[2] != "section":
        return source_id
    mapped_scenario_id = _mapped_optional_entity_id(
        mappings,
        "scenario",
        parts[1],
        field_name="context_sources.source_id",
        repair_tracker=repair_tracker,
    )
    if mapped_scenario_id is None:
        return None
    return f"scenario:{mapped_scenario_id}:section:{parts[3]}"


def _mapped_world_state_context_source_id(
    mappings: dict[str, dict[str, str]],
    source_id: str,
    *,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    mapped_items: list[str] = []
    for item in source_id.split(","):
        item = item.strip()
        if not item:
            continue
        mapped = _mapped_world_state_context_source_item(
            mappings,
            item,
            repair_tracker=repair_tracker,
        )
        if mapped is not None:
            mapped_items.append(mapped)
    if not mapped_items and source_id.strip():
        return None
    return ",".join(mapped_items)


def _mapped_world_state_context_source_item(
    mappings: dict[str, dict[str, str]],
    source_id: str,
    *,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    mapped_state_id = mappings.get("world_state", {}).get(source_id)
    if mapped_state_id is not None:
        return mapped_state_id
    mapped_key = mappings.get("world_state_key", {}).get(source_id)
    if mapped_key is not None:
        return mapped_key
    if source_id.startswith("location:"):
        location_id = source_id.removeprefix("location:")
        if not location_id:
            return source_id
        mapped_location_id = _mapped_optional_entity_id(
            mappings,
            "location",
            location_id,
            field_name="context_sources.source_id",
            repair_tracker=repair_tracker,
        )
        if mapped_location_id is None:
            return None
        return f"location:{mapped_location_id}"
    if ":" in source_id:
        return source_id
    if _looks_like_generated_import_id(source_id):
        if repair_tracker is not None:
            repair_tracker.record("context_sources.source_id")
        return None
    return source_id


def _mapped_memory_context_source_id(
    mappings: dict[str, dict[str, str]],
    source_id: str,
    *,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    mapped_items: list[str] = []
    for item in source_id.split(","):
        item = item.strip()
        if not item:
            continue
        mapped = _mapped_memory_context_source_item(
            mappings,
            item,
            repair_tracker=repair_tracker,
        )
        if mapped is not None:
            mapped_items.append(mapped)
    if not mapped_items and source_id.strip():
        return None
    return ",".join(mapped_items)


def _mapped_memory_context_source_item(
    mappings: dict[str, dict[str, str]],
    source_id: str,
    *,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    mapped_memory_id = mappings.get("memory", {}).get(source_id)
    if mapped_memory_id is not None:
        return mapped_memory_id
    if source_id.startswith("character_profile:"):
        character_id = source_id.removeprefix("character_profile:")
        if not character_id:
            return source_id
        return _mapped_memory_character_context_source_id(
            mappings,
            character_id,
            prefix="character_profile",
            suffix="",
            repair_tracker=repair_tracker,
        )
    if source_id.startswith("relationship:"):
        parts = source_id.split(":", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            return source_id
        return _mapped_memory_character_context_source_id(
            mappings,
            parts[1],
            prefix=parts[0],
            suffix=f":{parts[2]}",
            repair_tracker=repair_tracker,
        )
    if ":" in source_id:
        return source_id
    return _mapped_optional_entity_id(
        mappings,
        "memory",
        source_id,
        field_name="context_sources.source_id",
        repair_tracker=repair_tracker,
    )


def _mapped_memory_character_context_source_id(
    mappings: dict[str, dict[str, str]],
    character_id: str,
    *,
    prefix: str,
    suffix: str,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    mapped_character_id = _mapped_optional_entity_id(
        mappings,
        "character",
        character_id,
        field_name="context_sources.source_id",
        repair_tracker=repair_tracker,
    )
    if mapped_character_id is None:
        return None
    return f"{prefix}:{mapped_character_id}{suffix}"


def _looks_like_generated_import_id(value: str) -> bool:
    return len(value) == 32 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _remapped_context_source_metadata_json(
    row: dict[str, object],
    message_id_map: dict[str, str],
    character_text_message_id_map: dict[str, str],
    *,
    observation_id_map: dict[str, str],
    scenario_id_map: dict[str, str],
    entity_id_maps: dict[str, dict[str, str]],
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    metadata = _optional_json_object(row, "metadata_json") or {}
    remapped = dict(metadata)
    source_type = _text(row, "source_type")
    if source_type == "observation" and "observation_id" in remapped:
        observation_id = remapped["observation_id"]
        if observation_id is None or observation_id == "":
            pass
        elif not isinstance(observation_id, str):
            raise ChatBundleError(
                "Bundle context_sources.metadata_json.observation_id "
                "must be an observation id"
            )
        else:
            remapped["observation_id"] = _mapped_optional_id(
                observation_id_map,
                observation_id,
                field_name="context_sources.metadata_json.observation_id",
                repair_tracker=repair_tracker,
            )
    if "scenario_id" in remapped:
        scenario_id = remapped["scenario_id"]
        if scenario_id is None or scenario_id == "":
            pass
        elif not isinstance(scenario_id, str):
            raise ChatBundleError(
                "Bundle context_sources.metadata_json.scenario_id "
                "must be a scenario id"
            )
        else:
            remapped["scenario_id"] = _mapped_optional_id(
                scenario_id_map,
                scenario_id,
                field_name="context_sources.metadata_json.scenario_id",
                repair_tracker=repair_tracker,
            )
    if "audience_character_ids" in remapped:
        raw_audience = remapped["audience_character_ids"]
        mapped_audience = _mapped_context_source_metadata_id_list(
            raw_audience,
            entity_id_maps.get("character", {}),
            field_name="context_sources.metadata_json.audience_character_ids",
            repair_tracker=repair_tracker,
        )
        if isinstance(raw_audience, list) and raw_audience and not mapped_audience:
            return None
        remapped["audience_character_ids"] = mapped_audience
    if "source_provenance_mode" in remapped and remapped[
        "source_provenance_mode"
    ] not in {"all", "any"}:
        raise ChatBundleError(
            "Bundle context_sources.metadata_json.source_provenance_mode "
            "must be 'all' or 'any'"
        )
    if source_type == "character_text_thread" and "thread_id" in remapped:
        thread_id = remapped["thread_id"]
        if thread_id is None or thread_id == "":
            pass
        elif not isinstance(thread_id, str):
            raise ChatBundleError(
                "Bundle context_sources.metadata_json.thread_id "
                "must be a character text thread id"
            )
        else:
            remapped["thread_id"] = _mapped_optional_id(
                entity_id_maps.get("character_text_thread", {}),
                thread_id,
                field_name="context_sources.metadata_json.thread_id",
                repair_tracker=repair_tracker,
            )
    if source_type == "character_text_thread" and "entity_ids" in remapped:
        remapped["entity_ids"] = _mapped_context_source_metadata_id_list(
            remapped["entity_ids"],
            entity_id_maps.get("character_text_thread", {}),
            field_name="context_sources.metadata_json.entity_ids",
            repair_tracker=repair_tracker,
        )
    for field in _CONTEXT_SOURCE_METADATA_MESSAGE_ID_FIELDS:
        if field not in remapped:
            continue
        value = remapped[field]
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise ChatBundleError(
                f"Bundle context_sources.metadata_json.{field} "
                "must be a message id"
            )
        mapped = _mapped_optional_source_ref(
            value,
            message_id_map,
            character_text_message_id_map,
            f"context_sources.metadata_json.{field}",
            repair_tracker,
        )
        if mapped is None:
            return None
        remapped[field] = mapped
    for field in _CONTEXT_SOURCE_METADATA_MESSAGE_ID_LIST_FIELDS:
        if field not in remapped:
            continue
        value = remapped[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ChatBundleError(
                f"Bundle context_sources.metadata_json.{field} "
                "must be a list of message ids"
            )
        if len(value) > _MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS:
            return None
        mapped_items: list[str] = []
        for item in value:
            mapped = _mapped_optional_source_ref(
                item,
                message_id_map,
                character_text_message_id_map,
                f"context_sources.metadata_json.{field}",
                repair_tracker,
            )
            if mapped is None:
                return None
            mapped_items.append(mapped)
        remapped[field] = mapped_items
    if "source_provenance_groups" in remapped:
        raw_groups = remapped["source_provenance_groups"]
        if (
            not isinstance(raw_groups, list)
            or len(raw_groups) > _MAX_CONTEXT_SOURCE_PROVENANCE_GROUPS
            or not all(
                isinstance(group, list)
                and bool(group)
                and len(group) <= _MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS
                and all(isinstance(item, str) and item for item in group)
                for group in raw_groups
            )
        ):
            raise ChatBundleError(
                "Bundle context_sources.metadata_json."
                "source_provenance_groups must be a list of message-id lists"
            )
        mapped_groups: list[list[str]] = []
        for group in raw_groups:
            mapped_group: list[str] = []
            for item in group:
                mapped = _mapped_optional_source_ref(
                    item,
                    message_id_map,
                    character_text_message_id_map,
                    (
                        "context_sources.metadata_json."
                        "source_provenance_groups"
                    ),
                    repair_tracker,
                )
                if mapped is None:
                    return None
                mapped_group.append(mapped)
            mapped_groups.append(mapped_group)
        remapped["source_provenance_groups"] = mapped_groups
        scalar_source_ids = {
            source_id
            for field in _CONTEXT_SOURCE_METADATA_MESSAGE_ID_FIELDS
            if isinstance((source_id := remapped.get(field)), str)
        }
        for field in _CONTEXT_SOURCE_METADATA_MESSAGE_ID_LIST_FIELDS:
            value = remapped.get(field)
            if isinstance(value, list):
                scalar_source_ids.update(
                    item for item in value if isinstance(item, str)
                )
        grouped_source_ids = {
            source_id for group in mapped_groups for source_id in group
        }
        if not scalar_source_ids.issubset(grouped_source_ids):
            return None
    return _dump_json_compact(remapped)


def _mapped_context_source_metadata_id_list(
    value: object,
    mapping: dict[str, str],
    *,
    field_name: str,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ChatBundleError(f"Bundle {field_name} must be a list of ids")
    mapped_items: list[str] = []
    for item in value:
        mapped = _mapped_optional_id(
            mapping,
            item,
            field_name=field_name,
            repair_tracker=repair_tracker,
        )
        if mapped is not None:
            mapped_items.append(mapped)
    return mapped_items


def _mapped_source_ref(
    source_ref: str,
    message_id_map: dict[str, str],
    character_text_message_id_map: dict[str, str],
    field_name: str,
) -> str:
    text_message_id = parse_character_text_source_ref(source_ref)
    if text_message_id is not None:
        try:
            mapped_text_message_id = character_text_message_id_map[text_message_id]
            return character_text_source_ref(mapped_text_message_id)
        except KeyError as exc:
            raise ChatBundleError(
                f"Bundle {field_name} references unknown character text "
                f"message id: {text_message_id}"
            ) from exc
    try:
        return message_id_map[source_ref]
    except KeyError as exc:
        raise ChatBundleError(
            f"Bundle {field_name} references unknown message id: {source_ref}"
        ) from exc


def _mapped_optional_source_ref(
    source_ref: str,
    message_id_map: dict[str, str],
    character_text_message_id_map: dict[str, str],
    field_name: str,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    text_message_id = parse_character_text_source_ref(source_ref)
    if text_message_id is not None:
        mapped_text_message_id = character_text_message_id_map.get(text_message_id)
        if mapped_text_message_id is None:
            if repair_tracker is not None:
                repair_tracker.record(field_name)
            return None
        return character_text_source_ref(mapped_text_message_id)
    mapped_message_id = message_id_map.get(source_ref)
    if mapped_message_id is None:
        if repair_tracker is not None:
            repair_tracker.record(field_name)
        return None
    return mapped_message_id


def _remap_director_pressure_state_value(
    value: dict[str, object],
    message_id_map: dict[str, str],
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> dict[str, object]:
    copied = dict(value)
    history = copied.get("escalation_history")
    if not isinstance(history, list):
        return copied

    remapped_history: list[object] = []
    for item in history:
        if not isinstance(item, dict):
            remapped_history.append(item)
            continue
        entry = dict(item)
        source_message_id = entry.get("source_message_id")
        if source_message_id in (None, ""):
            remapped_history.append(entry)
            continue
        if not isinstance(source_message_id, str):
            raise ChatBundleError(
                "Bundle director_pressure.escalation_history.source_message_id "
                "must be a message id"
            )
        mapped_source_message_id = message_id_map.get(source_message_id)
        if mapped_source_message_id is None:
            field_name = "director_pressure.escalation_history.source_message_id"
            if repair_tracker is not None:
                repair_tracker.record(field_name)
            entry["source_message_id"] = None
        else:
            entry["source_message_id"] = mapped_source_message_id
        remapped_history.append(entry)
    copied["escalation_history"] = remapped_history
    return copied


def _mapped_optional_entity_id(
    mappings: dict[str, dict[str, str]],
    entity_type: str,
    entity_id: str | None,
    *,
    field_name: str,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    if entity_id is None:
        return None
    mapping = mappings.get(entity_type)
    if mapping is None:
        return entity_id
    mapped = mapping.get(entity_id)
    if mapped is None:
        if repair_tracker is not None:
            repair_tracker.record(field_name)
        return None
    return mapped


def _mapped_json_message_ids(
    row: dict[str, object],
    key: str,
    message_id_map: dict[str, str],
) -> str:
    mapped = [
        _mapped_required(message_id_map, value)
        for value in _json_string_list(row, key)
    ]
    return _dump_json_compact(mapped)


def _mapped_json_source_refs(
    row: dict[str, object],
    key: str,
    message_id_map: dict[str, str],
    character_text_message_id_map: dict[str, str],
    field_name: str,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str:
    mapped: list[str] = []
    for value in _json_string_list(row, key):
        mapped_value = _mapped_optional_source_ref(
            value,
            message_id_map,
            character_text_message_id_map,
            field_name,
            repair_tracker,
        )
        if mapped_value is not None:
            mapped.append(mapped_value)
    return _dump_json_compact(mapped)


def _remapped_context_update_suggestion_proposed_value_json(
    row: dict[str, object],
    message_id_map: dict[str, str],
    character_text_message_id_map: dict[str, str],
    entity_id_maps: dict[str, dict[str, str]],
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str:
    value = _json_value(
        _text(row, "proposed_value_json"),
        "context_update_suggestions.proposed_value_json",
    )
    if not isinstance(value, dict):
        if _context_update_suggestion_proposed_value_is_location_ref(row):
            return _dump_json_compact(
                _mapped_context_update_suggestion_scalar_location_id(
                    value,
                    entity_id_maps,
                    repair_tracker=repair_tracker,
                )
            )
        return _dump_json_compact(value)
    remapped = dict(value)
    if "source_message_id" in remapped:
        source_message_id = remapped["source_message_id"]
        if source_message_id is None or source_message_id == "":
            pass
        elif not isinstance(source_message_id, str):
            raise ChatBundleError(
                "Bundle context_update_suggestions.proposed_value_json."
                "source_message_id must be a message id"
            )
        else:
            remapped["source_message_id"] = _mapped_optional_required(
                message_id_map=message_id_map,
                original_id=source_message_id,
                field_name=(
                    "context_update_suggestions.proposed_value_json."
                    "source_message_id"
                ),
                repair_tracker=repair_tracker,
            )
    if "source_message_ids" in remapped:
        source_message_ids = remapped["source_message_ids"]
        if not isinstance(source_message_ids, list) or not all(
            isinstance(item, str) and item for item in source_message_ids
        ):
            raise ChatBundleError(
                "Bundle context_update_suggestions.proposed_value_json."
                "source_message_ids must be a list of message ids"
            )
        mapped_items: list[str] = []
        for item in source_message_ids:
            mapped = _mapped_optional_source_ref(
                item,
                message_id_map,
                character_text_message_id_map,
                "context_update_suggestions.proposed_value_json.source_message_ids",
                repair_tracker,
            )
            if mapped is not None:
                mapped_items.append(mapped)
        remapped["source_message_ids"] = mapped_items
    observation_id_map = entity_id_maps.get("observation", {})
    if "source_observation_id" in remapped:
        source_observation_id = remapped["source_observation_id"]
        if source_observation_id is None or source_observation_id == "":
            remapped["source_observation_id"] = None
        elif not isinstance(source_observation_id, str):
            raise ChatBundleError(
                "Bundle context_update_suggestions.proposed_value_json."
                "source_observation_id must be an observation id"
            )
        else:
            remapped["source_observation_id"] = _mapped_optional_id(
                observation_id_map,
                source_observation_id,
                field_name=(
                    "context_update_suggestions.proposed_value_json."
                    "source_observation_id"
                ),
                repair_tracker=repair_tracker,
            )
    if "source_observation_ids" in remapped:
        source_observation_ids = remapped["source_observation_ids"]
        remapped["source_observation_ids"] = (
            _mapped_context_source_metadata_id_list(
                source_observation_ids,
                observation_id_map,
                field_name=(
                    "context_update_suggestions.proposed_value_json."
                    "source_observation_ids"
                ),
                repair_tracker=repair_tracker,
            )
        )
    if "location_id" in remapped:
        location_id = remapped["location_id"]
        if location_id is None or location_id == "":
            remapped["location_id"] = None
        elif not isinstance(location_id, str):
            raise ChatBundleError(
                "Bundle context_update_suggestions.proposed_value_json."
                "location_id must be a location id"
            )
        else:
            remapped["location_id"] = _mapped_optional_id(
                entity_id_maps.get("location", {}),
                location_id,
                field_name=(
                    "context_update_suggestions.proposed_value_json.location_id"
                ),
                repair_tracker=repair_tracker,
            )
    return _dump_json_compact(remapped)


def _context_update_suggestion_proposed_value_is_location_ref(
    row: dict[str, object],
) -> bool:
    return (
        _text(row, "entity_type"),
        _text(row, "field_path"),
    ) in _CONTEXT_UPDATE_SUGGESTION_LOCATION_FIELD_PATHS


def _mapped_context_update_suggestion_scalar_location_id(
    value: object,
    entity_id_maps: dict[str, dict[str, str]],
    *,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ChatBundleError(
            "Bundle context_update_suggestions.proposed_value_json "
            "must be a location id"
        )
    return _mapped_optional_id(
        entity_id_maps.get("location", {}),
        value,
        field_name="context_update_suggestions.proposed_value_json",
        repair_tracker=repair_tracker,
    )


def _remap_related_entities_json(
    row: dict[str, object],
    key: str,
    mappings: dict[str, dict[str, str]],
    *,
    field_name: str,
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> str:
    remapped: list[str] = []
    for value in _json_string_list(row, key):
        entity_type, separator, entity_id = value.partition(":")
        if not separator:
            remapped.append(value)
            continue
        mapped_entity_id = _mapped_optional_entity_id(
            mappings,
            entity_type,
            entity_id,
            field_name=field_name,
            repair_tracker=repair_tracker,
        )
        if mapped_entity_id is None:
            continue
        remapped.append(f"{entity_type}:{mapped_entity_id}")
    return _dump_json_compact(remapped)


def _validated_optional_json_object_text(
    row: dict[str, object],
    key: str,
) -> str | None:
    value = _optional_text(row, key)
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ChatBundleError(f"Invalid state change {key} JSON") from exc
    if not isinstance(parsed, dict):
        raise ChatBundleError(f"State change {key} must be a JSON object")
    return value


def _mapped_source_message_ids(
    *,
    row: dict[str, object],
    message_id_map: dict[str, str],
    character_text_message_id_map: dict[str, str],
    message_order: dict[str, int],
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> tuple[str, ...]:
    source_ids = _json_string_list(row, "source_message_ids_json")
    source_message_id = _optional_text(row, "source_message_id")
    if source_message_id in message_id_map:
        source_ids.append(source_message_id)
    ordered_source_ids = _ordered_import_source_refs(
        source_ids,
        message_order,
    )
    mapped: list[str] = []
    for source_id in ordered_source_ids:
        mapped_source_id = _mapped_optional_source_ref(
            source_id,
            message_id_map,
            character_text_message_id_map,
            "save_scenario_updates.source_message_ids_json",
            repair_tracker,
        )
        if mapped_source_id is not None:
            mapped.append(mapped_source_id)
    return tuple(mapped)


def _mapped_memory_source_message_ids(
    *,
    row: dict[str, object],
    message_id_map: dict[str, str],
    character_text_message_id_map: dict[str, str],
    message_order: dict[str, int],
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> tuple[str, ...]:
    try:
        source_ids = _json_string_list(row, "source_message_ids_json")
    except ChatBundleError:
        source_ids = []
    source_message_id = _optional_text(row, "source_message_id")
    if source_message_id in message_id_map:
        source_ids.append(source_message_id)
    ordered_source_ids = _ordered_import_source_refs(
        source_ids,
        message_order,
    )
    mapped: list[str] = []
    for source_id in ordered_source_ids:
        mapped_source_id = _mapped_optional_source_ref(
            source_id,
            message_id_map,
            character_text_message_id_map,
            "memories.source_message_ids_json",
            repair_tracker,
        )
        if mapped_source_id is not None:
            mapped.append(mapped_source_id)
    return tuple(mapped)


def _mapped_observation_source_message_ids(
    *,
    row: dict[str, object],
    message_id_map: dict[str, str],
    character_text_message_id_map: dict[str, str],
    message_order: dict[str, int],
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> tuple[str, ...]:
    source_ids = _json_string_list(row, "source_message_ids_json")
    ordered_source_ids = _ordered_import_source_refs(
        source_ids,
        message_order,
    )
    mapped: list[str] = []
    for source_id in ordered_source_ids:
        mapped_source_id = _mapped_optional_source_ref(
            source_id,
            message_id_map,
            character_text_message_id_map,
            "context_observations.source_message_ids_json",
            repair_tracker,
        )
        if mapped_source_id is not None:
            mapped.append(mapped_source_id)
    return tuple(mapped)


def _ordered_import_source_refs(
    source_ids: list[str],
    message_order: dict[str, int],
) -> tuple[str, ...]:
    unique_source_ids = tuple(dict.fromkeys(source_ids))
    has_text_source = any(
        parse_character_text_source_ref(source_id) is not None
        for source_id in unique_source_ids
    )
    if not has_text_source:
        return tuple(
            sorted(
                unique_source_ids,
                key=lambda source_id: message_order.get(source_id, 10**9),
            )
        )
    return unique_source_ids


def _remap_evidence_message_ids(
    value: dict[str, object],
    message_id_map: dict[str, str],
    repair_tracker: _BundleImportRepairTracker | None = None,
) -> dict[str, object]:
    def remap(item: object) -> object:
        if isinstance(item, dict):
            result: dict[str, object] = {}
            for key, nested in item.items():
                if key == "source_message_id" and isinstance(nested, str):
                    mapped = message_id_map.get(nested)
                    if mapped is None:
                        if repair_tracker is not None:
                            repair_tracker.record(
                                "save_loss_outcomes.evidence.source_message_id"
                            )
                        result[key] = None
                    else:
                        result[key] = mapped
                else:
                    result[str(key)] = remap(nested)
            return result
        if isinstance(item, list):
            return [remap(nested) for nested in item]
        return item

    return cast(dict[str, object], remap(value))


def _text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ChatBundleError(f"Expected text field: {key}")
    return value


def _optional_text(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ChatBundleError(f"Expected optional text field: {key}")
    return value


def _bundle_row_references_messages(
    row: dict[str, object],
    message_ids: set[str],
) -> bool:
    if not message_ids:
        return False
    for key in (
        "source_message_id",
        "first_seen_message_id",
        "last_updated_message_id",
        "message_id",
    ):
        if _optional_text(row, key) in message_ids:
            return True
    source_ids = (
        _json_string_list(row, "source_message_ids_json")
        if "source_message_ids_json" in row
        else []
    )
    return bool(message_ids.intersection(source_ids))


def _bundle_summary_covers_transition(
    row: dict[str, object],
    *,
    transition_message_ids: set[str],
    message_order: dict[str, int],
) -> bool:
    if not transition_message_ids:
        return False
    start = message_order.get(_text(row, "covers_message_start_id"))
    end = message_order.get(_text(row, "covers_message_end_id"))
    if start is None or end is None:
        return False
    lower, upper = sorted((start, end))
    return any(
        lower <= message_order.get(message_id, -1) <= upper
        for message_id in transition_message_ids
    )


def _remove_imported_safety_transition_records(
    connection: sqlite3.Connection,
    *,
    save_id: str,
    transition_message_ids: set[str],
) -> None:
    if not transition_message_ids:
        return
    placeholders = ", ".join("?" for _ in transition_message_ids)
    parameters = (save_id, *sorted(transition_message_ids))
    table_names = (
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
    )
    for table_name in table_names:
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
                for message_id in transition_message_ids:
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
            for _ in transition_message_ids
        )
        + ")",
        (
            save_id,
            *(
                value
                for message_id in sorted(transition_message_ids)
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
            *sorted(transition_message_ids),
            *sorted(transition_message_ids),
            *sorted(transition_message_ids),
            *sorted(transition_message_ids),
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
            for message_id in transition_message_ids
        ):
            connection.execute("DELETE FROM summaries WHERE id = ?", (row["id"],))
def _int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int):
        raise ChatBundleError(f"Expected integer field: {key}")
    return value


def _optional_int(row: dict[str, object], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ChatBundleError(f"Expected optional integer field: {key}")
    return value


def _float(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, int | float):
        raise ChatBundleError(f"Expected numeric field: {key}")
    return float(value)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ChatBundleError(f"Expected object: {name}")
    return cast(dict[str, object], value)


def _list_of_objects(value: object, name: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ChatBundleError(f"Expected object list: {name}")
    return cast(list[dict[str, object]], value)


def _json_object(row: dict[str, object], key: str) -> dict[str, object]:
    loaded = _json_value(_text(row, key), key)
    if not isinstance(loaded, dict):
        raise ChatBundleError(f"Expected JSON object field: {key}")
    return cast(dict[str, object], loaded)


def _optional_json_object(
    row: dict[str, object],
    key: str,
) -> dict[str, object] | None:
    value = _optional_text(row, key)
    if value is None:
        return None
    loaded = _json_value(value, key)
    if not isinstance(loaded, dict):
        raise ChatBundleError(f"Expected optional JSON object field: {key}")
    return cast(dict[str, object], loaded)


def _json_string_list(row: dict[str, object], key: str) -> list[str]:
    loaded = _json_value(_text(row, key), key)
    if not isinstance(loaded, list) or not all(
        isinstance(item, str) for item in loaded
    ):
        raise ChatBundleError(f"Expected JSON string list field: {key}")
    return cast(list[str], loaded)


def _json_value(value: str, key: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ChatBundleError(f"Invalid JSON field: {key}") from exc


def _dump_json_pretty(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _dump_json_compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _copy_row_for_save(
    row: dict[str, object],
    save_id: str,
    *,
    new_id: str,
) -> dict[str, object]:
    copied = dict(row)
    copied["id"] = new_id
    copied["save_id"] = save_id
    return copied


def _backfill_imported_scene_world_time(connection: Any, save_id: str) -> None:
    rows = connection.execute(
        """
        SELECT id, in_world_time, time_of_day, day_of_week, world_day_index,
               source_message_id, world_time_day_index, world_time_day_label,
               world_time_phase, world_time_clock_minutes,
               world_time_period_label, world_time_source_message_id,
               world_time_confidence
        FROM scene_snapshots
        WHERE save_id = ?
        """,
        (save_id,),
    ).fetchall()
    for row in rows:
        has_canonical_fields = any(
            row[column] not in (None, "")
            for column in (
                "world_time_day_index",
                "world_time_day_label",
                "world_time_phase",
                "world_time_clock_minutes",
                "world_time_period_label",
            )
        )
        canonical = canonical_world_time_from_values(
            day_index=row["world_time_day_index"],
            day_label=row["world_time_day_label"],
            phase=row["world_time_phase"],
            clock_minutes=row["world_time_clock_minutes"],
            period_label=row["world_time_period_label"],
            source_message_id=row["world_time_source_message_id"],
            confidence=row["world_time_confidence"],
            legacy_in_world_time=row["in_world_time"],
            legacy_time_of_day=row["time_of_day"],
            legacy_day_of_week=row["day_of_week"],
            legacy_world_day_index=row["world_day_index"],
        )
        legacy_fields = {
            "in_world_time": row["in_world_time"],
            "time_of_day": row["time_of_day"],
            "day_of_week": row["day_of_week"],
            "world_day_index": row["world_day_index"],
        }
        if has_canonical_fields:
            synthesized_legacy_fields = legacy_world_time_fields(canonical)
            for field_name, field_value in synthesized_legacy_fields.items():
                if legacy_fields[field_name] in (None, ""):
                    legacy_fields[field_name] = field_value
        source_message_id = canonical.source_message_id
        if source_message_id is None and not has_canonical_fields:
            source_message_id = row["source_message_id"]
        connection.execute(
            """
            UPDATE scene_snapshots
            SET in_world_time = ?,
                time_of_day = ?,
                day_of_week = ?,
                world_day_index = ?,
                world_time_day_index = ?,
                world_time_day_label = ?,
                world_time_phase = ?,
                world_time_clock_minutes = ?,
                world_time_period_label = ?,
                world_time_source_message_id = ?,
                world_time_confidence = ?
            WHERE id = ?
            """,
            (
                legacy_fields["in_world_time"],
                legacy_fields["time_of_day"],
                legacy_fields["day_of_week"],
                legacy_fields["world_day_index"],
                canonical.day_index,
                canonical.day_label,
                canonical.phase,
                canonical.clock_minutes,
                canonical.period_label,
                source_message_id,
                canonical.confidence,
                row["id"],
            ),
        )


def _insert_rows(
    connection: Any,
    table_name: str,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    columns = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})")
    }
    if not columns:
        raise ChatBundleError(f"Unknown bundle table: {table_name}")
    insert_columns = [
        column for column in rows[0].keys() if column in columns and column != "files"
    ]
    if not insert_columns:
        return
    placeholders = ", ".join("?" for _ in insert_columns)
    column_sql = ", ".join(insert_columns)
    values = [
        tuple(_sqlite_value(row.get(column)) for column in insert_columns)
        for row in rows
    ]
    connection.executemany(
        f"INSERT INTO {table_name}({column_sql}) VALUES ({placeholders})",
        values,
    )


def _coalesce_import_context_sources(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    coalesced: dict[tuple[object, object, object], dict[str, object]] = {}
    for row in rows:
        key = (row.get("save_id"), row.get("source_type"), row.get("source_id"))
        existing = coalesced.get(key)
        if existing is None:
            coalesced[key] = row
        elif (
            existing.get("archived_at") is not None
            and row.get("archived_at") is None
        ):
            coalesced[key] = row
        elif (existing.get("archived_at") is None) == (
            row.get("archived_at") is None
        ):
            existing["metadata_json"] = _merged_import_context_source_metadata_json(
                existing.get("metadata_json"),
                row.get("metadata_json"),
            )
            existing["token_estimate"] = max(
                int(_numeric_import_value(existing.get("token_estimate"))),
                int(_numeric_import_value(row.get("token_estimate"))),
            )
    return list(coalesced.values())


def _coalesce_import_proactive_triggers(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    coalesced: dict[tuple[object, object, object], dict[str, object]] = {}
    for row in rows:
        key = (
            row.get("save_id"),
            row.get("character_id"),
            row.get("trigger_key"),
        )
        existing = coalesced.get(key)
        if existing is None:
            coalesced[key] = row
            continue
        for field in (
            "thread_id",
            "text_message_id",
            "source_message_id",
        ):
            if row.get(field) is not None:
                existing[field] = row[field]
        for field in (
            "trigger_type",
            "source_type",
            "source_id",
            "reason",
        ):
            value = row.get(field)
            if isinstance(value, str) and value:
                existing[field] = value
        if row.get("updated_at") is not None:
            existing["updated_at"] = row["updated_at"]
    return list(coalesced.values())


def _merged_import_context_source_metadata_json(
    first: object,
    second: object,
) -> str:
    try:
        metadata = merge_context_source_metadata(first, second)
    except ValueError as exc:
        raise ChatBundleError(str(exc)) from exc
    return _dump_json_compact(metadata)


def _coalesce_import_entity_links(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    coalesced: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            row.get("save_id"),
            row.get("entity_type"),
            row.get("entity_id"),
            row.get("target_type"),
            row.get("target_id"),
            row.get("relation"),
        )
        existing = coalesced.get(key)
        if existing is None:
            coalesced[key] = row
        elif existing.get("source_message_id") is None:
            existing["source_message_id"] = row.get("source_message_id")
    return list(coalesced.values())


def _coalesce_import_knowledge_edges(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    coalesced: dict[tuple[object, ...], dict[str, object]] = {}
    state_rank = {"knows": 0, "may_know": 1, "does_not_know": 2}
    for row in rows:
        row["target_type"] = normalized_knowledge_target_type(
            str(row.get("target_type", ""))
        )
        row["source_message_ids_json"] = _merge_json_string_lists(
            row.get("source_message_ids_json"),
            "[]",
            limit=_MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS,
            extra_values=(row.get("source_message_id"),),
        )
        key = (
            row.get("save_id"),
            row.get("character_id"),
            row.get("target_type"),
            row.get("target_id"),
        )
        existing = coalesced.get(key)
        if existing is None:
            coalesced[key] = row
            continue
        existing_active = existing.get("archived_at") is None
        row_active = row.get("archived_at") is None
        if row_active and not existing_active:
            coalesced[key] = row
            continue
        if existing_active and not row_active:
            continue
        source_message_ids = (
            existing.get("source_message_id"),
            row.get("source_message_id"),
        )
        if state_rank.get(str(row.get("knowledge_state")), 1) > state_rank.get(
            str(existing.get("knowledge_state")),
            1,
        ):
            for field in (
                "knowledge_state",
                "acquisition_method",
                "source_message_id",
                "evidence_quote",
            ):
                existing[field] = row.get(field)
        existing["confidence"] = max(
            _numeric_import_value(existing.get("confidence")),
            _numeric_import_value(row.get("confidence")),
        )
        try:
            existing["source_message_ids_json"] = _merge_json_string_lists(
                existing.get("source_message_ids_json"),
                row.get("source_message_ids_json"),
                limit=_MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS,
                extra_values=source_message_ids,
            )
        except ChatBundleError:
            existing["knowledge_state"] = "does_not_know"
            existing["acquisition_method"] = "unknown"
            existing["source_message_id"] = None
            existing["source_message_ids_json"] = "[]"
            existing["evidence_quote"] = None
    return list(coalesced.values())


def _merge_json_string_lists(
    first: object,
    second: object,
    *,
    limit: int | None = None,
    extra_values: Iterable[object] = (),
) -> str:
    values: list[str] = []
    for raw in (first, second):
        try:
            loaded = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            loaded = []
        if isinstance(loaded, list):
            values.extend(
                str(value)
                for value in loaded
                if isinstance(value, str) and value
            )
    values.extend(
        str(value) for value in extra_values if isinstance(value, str) and value
    )
    merged = list(dict.fromkeys(values))
    if limit is not None and len(merged) > limit:
        raise ChatBundleError("Merged provenance is too large")
    return _dump_json_compact(merged)


def _numeric_import_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _backfill_imported_character_contact_states(connection: Any, save_id: str) -> None:
    connection.execute(
        """
        INSERT INTO character_contact_states(
            id, save_id, player_character_id, character_id,
            player_has_character_number, character_has_player_number
        )
        SELECT lower(hex(randomblob(16))), thread.save_id, player.id,
               npc.id, 1, 1
        FROM character_text_threads AS thread
        JOIN characters AS npc
          ON npc.id = thread.character_id
         AND npc.save_id = thread.save_id
         AND npc.archived_at IS NULL
         AND npc.is_player_character = 0
        JOIN characters AS player
          ON player.save_id = thread.save_id
         AND player.archived_at IS NULL
         AND player.is_player_character = 1
        WHERE thread.save_id = ?
          AND thread.archived_at IS NULL
        ON CONFLICT(save_id, player_character_id, character_id) DO NOTHING
        """,
        (save_id,),
    )
    connection.execute(
        """
        INSERT INTO character_contact_states(
            id, save_id, player_character_id, character_id,
            player_has_character_number, character_has_player_number,
            source_message_id
        )
        SELECT lower(hex(randomblob(16))), route.save_id,
               route.player_character_id, route.npc_character_id, 1, 1,
               route.source_message_id
        FROM dating_route_states AS route
        WHERE route.save_id = ?
          AND route.archived_at IS NULL
          AND route.stage IN (
            'contact_exchanged',
            'first_date_planned',
            'first_date_in_progress',
            'early_dating',
            'exclusive',
            'committed'
          )
        ON CONFLICT(save_id, player_character_id, character_id) DO NOTHING
        """,
        (save_id,),
    )


def _location_rows_parent_first(
    rows: list[dict[str, object]],
    save_id: str,
) -> list[dict[str, object]]:
    row_ids = {_text(row, "id") for row in rows}
    pending = [
        (
            {**row, "parent_location_id": None}
            if _optional_text(row, "parent_location_id") not in row_ids
            else row
        )
        for row in rows
    ]
    inserted_ids: set[str] = set()
    ordered: list[dict[str, object]] = []
    while pending:
        next_pending: list[dict[str, object]] = []
        made_progress = False
        for row in pending:
            parent_id = _optional_text(row, "parent_location_id")
            parent_in_same_save = (
                parent_id is not None
                and parent_id != _text(row, "id")
                and _text(row, "save_id") == save_id
                and parent_id in row_ids
            )
            if parent_in_same_save and parent_id not in inserted_ids:
                next_pending.append(row)
                continue
            ordered.append(row)
            inserted_ids.add(_text(row, "id"))
            made_progress = True
        if not made_progress:
            ordered.extend(next_pending)
            break
        pending = next_pending
    return ordered


def _sqlite_value(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, dict | list):
        return _dump_json_compact(value)
    return value


def _remember_media_backup(
    backups: dict[Path, bytes | None],
    path: Path,
) -> None:
    if path in backups:
        return
    backups[path] = path.read_bytes() if path.is_file() else None


def _restore_media_backups(backups: dict[Path, bytes | None]) -> None:
    for path, payload in reversed(backups.items()):
        try:
            if payload is None:
                if path.is_file():
                    path.unlink()
            else:
                write_private_bytes(path, payload)
        except OSError:
            pass
