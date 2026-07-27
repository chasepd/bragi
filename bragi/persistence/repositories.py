"""Small SQLite repository facade for MVP data."""

from __future__ import annotations

import json
import math
import sqlite3
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast
from uuid import uuid4

from bragi.observation_types import normalize_observation_type
from bragi.persistence.context_provenance import merge_context_source_metadata
from bragi.persistence.migrations import (
    _remap_migrated_memory_proactive_triggers,
    _remap_migrated_memory_references,
    migrate_database,
)
from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterContactStateRecord,
    CharacterKnowledgeEdgeRecord,
    CharacterRecord,
    CharacterTextActivityEventRecord,
    CharacterTextMessageAttachmentRecord,
    CharacterTextMessageRecord,
    CharacterTextMessageRevisionRecord,
    CharacterTextProactiveTriggerRecord,
    CharacterTextProvenanceRecord,
    CharacterTextThreadParticipantRecord,
    CharacterTextThreadRecord,
    ContextObservationCurationHealthRecord,
    ContextObservationCurationStateRecord,
    ContextObservationRecord,
    ContextSourceRecord,
    ContextSourceSearchHit,
    ContextUpdateAuditRecord,
    ContextUpdateSuggestionRecord,
    DatingRouteStateRecord,
    EntityLinkRecord,
    JobRecord,
    JobStepRecord,
    LocationRecord,
    LossConditionChangeRecord,
    LossConditionRecord,
    LossOutcomeRecord,
    MediaAssetRecord,
    MemoryRecord,
    MessageActionChoiceRecord,
    MessagePageRecord,
    MessageRecord,
    MessageRevisionMetadataRecord,
    MessageRevisionRecord,
    MessageScenePresenceRecord,
    MessageVisibilityRecord,
    ModelPreferenceRecord,
    ProviderCatalogEntryRecord,
    ProviderConfigRecord,
    ProviderModelRecord,
    RuntimePerformanceRecord,
    RuntimeSlowOperationRecord,
    SaveDetailsRecord,
    SaveRecord,
    SaveScenarioUpdateRecord,
    ScenarioRecord,
    SceneSnapshotRecord,
    ScheduledTaskRecord,
    ScopedSettingRecord,
    StateChangeRecord,
    SummaryRecord,
    UserRecord,
    UserSessionRecord,
    WorldStateRecord,
)
from bragi.redaction import redact_text
from bragi.safety import normalize_message_safety
from bragi.text_search import cjk_lexical_anchors, unicode_word_terms
from bragi.world_time_model import (
    canonical_world_time_from_legacy,
    canonical_world_time_from_values,
    legacy_world_time_fields,
)
from bragi_common.media_mime import (
    INERT_MEDIA_MIME_TYPE,
    SUPPORTED_IMAGE_MIME_TYPES,
    SUPPORTED_VIDEO_MIME_TYPES,
    canonical_media_mime_type,
)

MAX_CONTEXT_SEARCH_TERMS = 64
MAX_UNICODE_SUBSTRING_TERMS = 32
MAX_CONTEXT_EXACT_PHRASES = 8
MAX_MEMORY_SOURCE_MESSAGE_IDS = 64
MAX_MEMORY_SOURCE_OBSERVATION_IDS = 64
MAX_KNOWLEDGE_EDGE_SOURCE_MESSAGE_IDS = 64
MAX_CONTEXT_SOURCE_PROVENANCE_GROUPS = 64
MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS = 64
MAX_CONTEXT_SOURCE_SEARCH_TEXT_CHARS = 65_536
MAX_CONTEXT_EXACT_IDENTIFIER_CHARS = 512
SCOPED_MAY_KNOW_CONFIDENCE_THRESHOLD = 0.7
MAX_NARRATION_GRAPH_CHARACTER_IDS = 64
JOB_STATUSES = frozenset({"queued", "running", "succeeded", "failed", "cancelled"})
_UNSET = object()
CHARACTER_TEXT_DELIVERY_STATUSES = frozenset(
    {"pending", "retrying", "sent", "failed"}
)
CHARACTER_TEXT_ATTACHMENT_KINDS = frozenset(
    {"character_image", "object_context_image", "uploaded_photo"}
)
CHARACTER_TEXT_ATTACHMENT_STATUSES = frozenset({"succeeded", "failed"})
JOB_STEP_STATUSES = frozenset(
    {
        "pending",
        "running",
        "succeeded",
        "applied",
        "narrowed",
        "failed",
        "cancelled",
        "skipped",
        "deferred",
        "blocked_dependency",
        "skipped_provider_pressure",
    }
)
_JOB_INITIAL_STATUSES = frozenset({"queued", "running"})
_JOB_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
_JOB_UPDATE_STATUSES = frozenset({"succeeded", "failed"})
_SAFE_TEXT_METADATA_KEYS = frozenset(
    {
        "evidence_quote",
        "evidence_source_id",
        "openrouter_selected_model",
        "openrouter_selected_provider",
        "skipped_reason",
        "status",
    }
)
_SAFE_TEXT_LIST_METADATA_KEYS = frozenset(
    {
        "openrouter_provider_attempts",
        "queued_suggestion_ids",
        "source_message_ids",
        "updated_fields",
    }
)
_SAFE_NUMBER_LIST_METADATA_KEYS = frozenset(
    {"openrouter_provider_attempt_statuses"}
)
_MAX_SAFE_METADATA_TEXT_LENGTH = 200
_MAX_SAFE_METADATA_LIST_ITEMS = 20
MESSAGE_REVISION_RECONCILIATION_STATUSES = frozenset(
    {"queued", "succeeded", "skipped", "failed"}
)
LOSS_CONDITION_STATUSES = frozenset(
    {"active", "mitigated", "retired", "triggered", "at_risk"}
)
_SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING = "scenario_evolution_turn_interval"
_IMAGE_STYLE_PRESET_SETTING = "image_style_preset"
_SETTING_SCOPES = frozenset({"global", "user", "save", "scenario"})
CHARACTER_KNOWLEDGE_STATES = frozenset({"knows", "may_know", "does_not_know"})
CHARACTER_KNOWLEDGE_ACQUISITION_METHODS = frozenset(
    {
        "witnessed",
        "overheard",
        "told",
        "inferred_from_visible_consequence",
        "background",
        "manual",
        "unknown",
    }
)
MESSAGE_VISIBILITY_STATES = frozenset({"visible", "not_visible", "unknown"})
USER_ROLES = frozenset({"admin", "user", "child"})
USER_STATUSES = frozenset({"active", "disabled"})
DATING_ROUTE_STAGES = frozenset(
    {
        "unmet",
        "introduced",
        "initial_interest",
        "contact_exchanged",
        "first_date_planned",
        "first_date_in_progress",
        "early_dating",
        "exclusive",
        "committed",
    }
)
_SCHEDULED_TASK_COLUMNS = (
    "id, task_type, save_id, enabled, interval_seconds, next_run_at, lease_until, "
    "last_started_at, last_completed_at, last_job_id, failure_count, payload_json, "
    "result_json, error, created_at, updated_at"
)
_CONTEXT_OBSERVATION_CURATION_STATE_COLUMNS = (
    "observation_id, save_id, attempt_count, next_eligible_at, lease_token, "
    "lease_until, last_error, terminal_outcome, completed_at, created_at, updated_at"
)
_JOB_COLUMNS = (
    "id, save_id, creator_user_id, type, status, payload_json, result_json, error, "
    "created_at, started_at, completed_at, duration_ms, diagnostics_json"
)
_DATING_ROUTE_STATE_COLUMNS = (
    "id, save_id, player_character_id, npc_character_id, stage, "
    "first_met_message_id, first_met_world_day_index, "
    "last_interaction_message_id, last_interaction_world_day_index, "
    "completed_interactions, dates_completed, interest_level, trust_level, "
    "comfort_with_intimacy, pacing_preference, known_boundaries_json, "
    "unresolved_questions_json, next_reasonable_step, source_message_id, "
    "created_at, updated_at"
)
_CHARACTER_TEXT_THREAD_COLUMNS = (
    "id, save_id, character_id, title, status, kind, memory_body, "
    "memory_message_count, memory_updated_at, created_at, updated_at, archived_at"
)
_CHARACTER_TEXT_THREAD_PARTICIPANT_COLUMNS = (
    "id, save_id, thread_id, character_id, ordinal, created_at, updated_at, "
    "archived_at"
)
_CHARACTER_TEXT_MESSAGE_COLUMNS = (
    "id, save_id, thread_id, character_id, sender, body, sender_character_id, "
    "provider, model, token_estimate, content_rating, delivery_status, delivery_error, "
    "delivery_job_id, delivery_attempt, in_world_sent_at, delivered_at, read_at, "
    "reply_to_message_id, created_at, updated_at, deleted_at"
)


class PersistenceRepositories:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.create_function(
            "bragi_normalize_text",
            1,
            _normalized_search_text,
            deterministic=True,
        )
        self.connection.create_function(
            "bragi_contains_exact_identifier",
            2,
            _contains_exact_structured_identifier,
            deterministic=True,
        )
        self._transaction_depth = 0

    def commit(self) -> None:
        if self._transaction_depth == 0:
            self.connection.commit()

    def begin_transaction(self) -> None:
        self._begin_transaction("BEGIN")

    def begin_immediate_transaction(self) -> None:
        self._begin_transaction("BEGIN IMMEDIATE")

    def _begin_transaction(self, statement: str) -> None:
        if self._transaction_depth == 0:
            self.connection.execute(statement)
        else:
            self.connection.execute(
                f"SAVEPOINT {_transaction_savepoint_name(self._transaction_depth)}"
            )
        self._transaction_depth += 1

    def commit_transaction(self) -> None:
        if self._transaction_depth == 0:
            raise RuntimeError("No active transaction to commit")

        self._transaction_depth -= 1
        if self._transaction_depth == 0:
            self.connection.commit()
        else:
            savepoint_name = _transaction_savepoint_name(self._transaction_depth)
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")

    def rollback_transaction(self) -> None:
        if self._transaction_depth == 0:
            self.connection.rollback()
            return

        self._transaction_depth -= 1
        if self._transaction_depth == 0:
            self.connection.rollback()
            return

        savepoint_name = _transaction_savepoint_name(self._transaction_depth)
        self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
        self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")

    def create_user(
        self,
        *,
        username: str,
        role: str,
        password_hash: str,
        user_id: str | None = None,
        status: str = "active",
    ) -> UserRecord:
        username = username.strip()
        username_normalized = _normalized_username(username)
        _validate_user_role(role)
        _validate_user_status(status)
        if not password_hash:
            raise ValueError("password_hash is required")
        record_id = user_id or _new_id()
        try:
            self.connection.execute(
                """
                INSERT INTO users(
                    id, username, username_normalized, role, password_hash, status
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    username,
                    username_normalized,
                    role,
                    password_hash,
                    status,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if _sqlite_error_mentions(exc, "users.username_normalized"):
                raise ValueError("username already exists") from exc
            raise
        self.commit()
        user = self.get_user(record_id)
        if user is None:
            raise ValueError(f"Unknown user id: {record_id}")
        return user

    def get_user(self, user_id: str) -> UserRecord | None:
        row = self._fetch_one(
            """
            SELECT id, username, username_normalized, role, password_hash, status,
                   created_at, updated_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        )
        return _user_from_row(row) if row is not None else None

    def get_user_by_username(self, username: str) -> UserRecord | None:
        row = self._fetch_one(
            """
            SELECT id, username, username_normalized, role, password_hash, status,
                   created_at, updated_at
            FROM users
            WHERE username_normalized = ?
            """,
            (_normalized_username(username),),
        )
        return _user_from_row(row) if row is not None else None

    def list_users(self) -> list[UserRecord]:
        rows = self._fetch_all(
            """
            SELECT id, username, username_normalized, role, password_hash, status,
                   created_at, updated_at
            FROM users
            ORDER BY username_normalized
            """,
            (),
        )
        return [_user_from_row(row) for row in rows]

    def update_user_role(self, user_id: str, role: str) -> UserRecord:
        _validate_user_role(role)
        self.connection.execute(
            """
            UPDATE users
            SET role = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (role, user_id),
        )
        self.commit()
        user = self.get_user(user_id)
        if user is None:
            raise ValueError(f"Unknown user id: {user_id}")
        return user

    def update_user_status(self, user_id: str, status: str) -> UserRecord:
        _validate_user_status(status)
        self.connection.execute(
            """
            UPDATE users
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, user_id),
        )
        self.commit()
        user = self.get_user(user_id)
        if user is None:
            raise ValueError(f"Unknown user id: {user_id}")
        return user

    def update_user_password_hash(
        self,
        user_id: str,
        password_hash: str,
    ) -> UserRecord:
        if not password_hash:
            raise ValueError("password_hash is required")
        self.connection.execute(
            """
            UPDATE users
            SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (password_hash, user_id),
        )
        self.commit()
        user = self.get_user(user_id)
        if user is None:
            raise ValueError(f"Unknown user id: {user_id}")
        return user

    def create_user_session(
        self,
        *,
        user_id: str,
        token_hash: str,
        expires_at: datetime | str,
        session_id: str | None = None,
    ) -> UserSessionRecord:
        if self.get_user(user_id) is None:
            raise ValueError(f"Unknown user id: {user_id}")
        if not token_hash:
            raise ValueError("token_hash is required")
        record_id = session_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO user_sessions(id, user_id, token_hash, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                record_id,
                user_id,
                token_hash,
                _timestamp_text(expires_at),
            ),
        )
        self.commit()
        session = self.get_user_session(record_id)
        if session is None:
            raise ValueError(f"Unknown session id: {record_id}")
        return session

    def get_user_session(self, session_id: str) -> UserSessionRecord | None:
        row = self._fetch_one(
            """
            SELECT id, user_id, token_hash, expires_at, revoked_at,
                   created_at, updated_at
            FROM user_sessions
            WHERE id = ?
            """,
            (session_id,),
        )
        return _user_session_from_row(row) if row is not None else None

    def get_active_user_session_by_token_hash(
        self,
        token_hash: str,
        *,
        now: datetime | str | None = None,
    ) -> UserSessionRecord | None:
        row = self._fetch_one(
            """
            SELECT id, user_id, token_hash, expires_at, revoked_at,
                   created_at, updated_at
            FROM user_sessions
            WHERE token_hash = ?
              AND revoked_at IS NULL
              AND expires_at > ?
            """,
            (
                token_hash,
                _timestamp_text(now or _utc_now()),
            ),
        )
        return _user_session_from_row(row) if row is not None else None

    def get_user_session_by_token_hash(
        self,
        token_hash: str,
    ) -> UserSessionRecord | None:
        row = self._fetch_one(
            """
            SELECT id, user_id, token_hash, expires_at, revoked_at,
                   created_at, updated_at
            FROM user_sessions
            WHERE token_hash = ?
            """,
            (token_hash,),
        )
        return _user_session_from_row(row) if row is not None else None

    def revoke_user_session(
        self,
        session_id: str,
        *,
        now: datetime | str | None = None,
    ) -> bool:
        cursor = self.connection.execute(
            """
            UPDATE user_sessions
            SET revoked_at = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND revoked_at IS NULL
            """,
            (_timestamp_text(now or _utc_now()), session_id),
        )
        self.commit()
        return cursor.rowcount > 0

    def revoke_user_sessions(
        self,
        user_id: str,
        *,
        except_token_hash: str | None = None,
        now: datetime | str | None = None,
    ) -> int:
        if self.get_user(user_id) is None:
            raise ValueError(f"Unknown user id: {user_id}")
        parameters: list[str] = [_timestamp_text(now or _utc_now()), user_id]
        token_filter = ""
        if except_token_hash is not None:
            token_filter = "AND token_hash != ?"
            parameters.append(except_token_hash)
        cursor = self.connection.execute(
            f"""
            UPDATE user_sessions
            SET revoked_at = ?, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
              AND revoked_at IS NULL
              {token_filter}
            """,
            tuple(parameters),
        )
        self.commit()
        return cursor.rowcount

    def create_scenario(
        self,
        *,
        type: str,
        title: str,
        premise: str,
        player_role: str,
        content: dict[str, object],
        scenario_id: str | None = None,
    ) -> ScenarioRecord:
        record = ScenarioRecord(
            id=scenario_id or _new_id(),
            type=type,
            title=title,
            premise=premise,
            player_role=player_role,
            content_json=_dump_json(content),
        )
        self.connection.execute(
            """
            INSERT INTO scenarios(id, type, title, premise, player_role, content_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.type,
                record.title,
                record.premise,
                record.player_role,
                record.content_json,
            ),
        )
        self.commit()
        created = self.get_scenario(record.id)
        if created is None:
            raise ValueError(f"Unknown scenario id: {record.id}")
        return created

    def create_save(
        self,
        *,
        scenario_id: str,
        title: str,
        save_id: str | None = None,
        custom_instructions: str = "",
        owner_user_id: str | None = None,
    ) -> SaveRecord:
        if owner_user_id is not None and self.get_user(owner_user_id) is None:
            raise ValueError(f"Unknown user id: {owner_user_id}")
        record = SaveRecord(
            id=save_id or _new_id(),
            scenario_id=scenario_id,
            title=title,
            active=True,
            custom_instructions=custom_instructions.strip(),
            owner_user_id=owner_user_id,
        )
        self.connection.execute(
            """
            INSERT INTO saves(
                id, scenario_id, title, active, custom_instructions, owner_user_id
            )
            VALUES (?, ?, ?, 1, ?, ?)
            """,
            (
                record.id,
                record.scenario_id,
                record.title,
                record.custom_instructions,
                record.owner_user_id,
            ),
        )
        self.commit()
        created = self.get_save(record.id)
        if created is None:
            raise ValueError(f"Unknown save id: {record.id}")
        return created

    def list_saves(self) -> list[SaveRecord]:
        rows = self._fetch_all(
            """
            SELECT
                saves.id,
                saves.scenario_id,
                scenarios.title AS scenario_title,
                saves.title,
                saves.active,
                saves.custom_instructions,
                saves.owner_user_id,
                saves.created_at,
                saves.updated_at,
                saves.last_opened_at
            FROM saves
            LEFT JOIN scenarios ON scenarios.id = saves.scenario_id
            ORDER BY
                julianday(COALESCE(saves.updated_at, saves.created_at)) DESC,
                julianday(saves.created_at) DESC,
                saves.rowid DESC
            """,
            (),
        )
        return [_save_from_row(row) for row in rows]

    def list_saves_for_user(self, user: UserRecord) -> list[SaveRecord]:
        if user.status != "active":
            return []
        if user.role == "admin":
            return self.list_saves()
        rows = self._fetch_all(
            """
            SELECT DISTINCT
                saves.id, saves.scenario_id, saves.title, saves.active,
                saves.custom_instructions, saves.owner_user_id,
                saves.created_at, saves.updated_at, saves.last_opened_at,
                scenarios.title AS scenario_title
            FROM saves
            LEFT JOIN scenarios ON scenarios.id = saves.scenario_id
            LEFT JOIN save_assignments
              ON save_assignments.save_id = saves.id
             AND save_assignments.user_id = ?
            WHERE saves.owner_user_id = ? OR save_assignments.user_id = ?
            ORDER BY
                julianday(COALESCE(saves.updated_at, saves.created_at)) DESC,
                julianday(saves.created_at) DESC,
                saves.rowid DESC
            """,
            (user.id, user.id, user.id),
        )
        return [_save_from_row(row) for row in rows]

    def get_save_for_user(
        self,
        user: UserRecord,
        save_id: str,
    ) -> SaveRecord | None:
        if not self.user_can_access_save(user, save_id):
            return None
        return self.get_save(save_id)

    def user_can_access_save(self, user: UserRecord, save_id: str) -> bool:
        if user.status != "active":
            return False
        if user.role == "admin":
            return self.get_save(save_id) is not None
        row = self._fetch_one(
            """
            SELECT 1
            FROM saves
            LEFT JOIN save_assignments
              ON save_assignments.save_id = saves.id
             AND save_assignments.user_id = ?
            WHERE saves.id = ?
              AND (
                    saves.owner_user_id = ?
                    OR save_assignments.user_id = ?
                  )
            """,
            (user.id, save_id, user.id, user.id),
        )
        return row is not None

    def grant_save_access(
        self,
        *,
        save_id: str,
        user_id: str,
    ) -> None:
        if self.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")
        if self.get_user(user_id) is None:
            raise ValueError(f"Unknown user id: {user_id}")
        self.connection.execute(
            """
            INSERT OR IGNORE INTO save_assignments(id, save_id, user_id)
            VALUES (?, ?, ?)
            """,
            (_new_id(), save_id, user_id),
        )
        self.commit()

    def claim_unowned_saves(self, owner_user_id: str) -> int:
        if self.get_user(owner_user_id) is None:
            raise ValueError(f"Unknown user id: {owner_user_id}")
        cursor = self.connection.execute(
            """
            UPDATE saves
            SET owner_user_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE owner_user_id IS NULL
            """,
            (owner_user_id,),
        )
        self.commit()
        return cursor.rowcount

    def get_user_active_save_id(self, user_id: str) -> str | None:
        row = self._fetch_one(
            """
            SELECT active_save_id
            FROM user_runtime_state
            WHERE user_id = ?
            """,
            (user_id,),
        )
        if row is None:
            return None
        value = row["active_save_id"]
        return value if isinstance(value, str) and value else None

    def list_user_active_save_ids(self) -> tuple[str, ...]:
        rows = self._fetch_all(
            """
            SELECT urs.active_save_id
            FROM user_runtime_state AS urs
            JOIN users
              ON users.id = urs.user_id
             AND users.status = 'active'
            JOIN saves
              ON saves.id = urs.active_save_id
            LEFT JOIN save_assignments
              ON save_assignments.save_id = saves.id
             AND save_assignments.user_id = users.id
            WHERE urs.active_save_id IS NOT NULL
              AND (
                    users.role = 'admin'
                    OR saves.owner_user_id = users.id
                    OR save_assignments.user_id IS NOT NULL
                  )
            GROUP BY urs.active_save_id
            ORDER BY MIN(urs.rowid)
            """,
            (),
        )
        return tuple(
            row["active_save_id"]
            for row in rows
            if isinstance(row["active_save_id"], str) and row["active_save_id"]
        )

    def set_user_active_save_id(
        self,
        *,
        user_id: str,
        save_id: str,
    ) -> None:
        if self.get_user(user_id) is None:
            raise ValueError(f"Unknown user id: {user_id}")
        if self.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")
        self.connection.execute(
            """
            INSERT INTO user_runtime_state(user_id, active_save_id)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                active_save_id = excluded.active_save_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, save_id),
        )
        self.commit()

    def clear_user_active_save_id(self, user_id: str) -> None:
        self.connection.execute(
            "DELETE FROM user_runtime_state WHERE user_id = ?",
            (user_id,),
        )
        self.commit()

    def get_scenario(self, scenario_id: str) -> ScenarioRecord | None:
        row = self._fetch_one(
            """
            SELECT id, type, title, premise, player_role, content_json,
                   created_at, updated_at
            FROM scenarios
            WHERE id = ?
            """,
            (scenario_id,),
        )
        return ScenarioRecord(**dict(row)) if row else None

    def list_scenarios(self) -> list[ScenarioRecord]:
        rows = self._fetch_all(
            """
            SELECT id, type, title, premise, player_role, content_json,
                   created_at, updated_at
            FROM scenarios
            ORDER BY
                julianday(updated_at) DESC,
                julianday(created_at) DESC,
                rowid DESC
            """,
            (),
        )
        return [ScenarioRecord(**dict(row)) for row in rows]

    def update_scenario(
        self,
        *,
        scenario_id: str,
        title: str,
        premise: str,
        player_role: str,
        content: dict[str, object],
    ) -> ScenarioRecord:
        self.connection.execute(
            """
            UPDATE scenarios
            SET title = ?, premise = ?, player_role = ?, content_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                title,
                premise,
                player_role,
                _dump_json(content),
                scenario_id,
            ),
        )
        self.commit()
        scenario = self.get_scenario(scenario_id)
        if scenario is None:
            raise ValueError(f"Unknown scenario id: {scenario_id}")
        return scenario

    def delete_scenario(self, scenario_id: str) -> bool:
        if self.count_saves_for_scenario(scenario_id) > 0:
            raise ValueError("Cannot delete a scenario with existing saves")
        self.delete_scoped_settings_for_scenario(scenario_id)
        cursor = self.connection.execute(
            "DELETE FROM scenarios WHERE id = ?",
            (scenario_id,),
        )
        self.commit()
        return cursor.rowcount > 0

    def count_saves_for_scenario(self, scenario_id: str) -> int:
        row = self._fetch_one(
            "SELECT COUNT(*) AS count FROM saves WHERE scenario_id = ?",
            (scenario_id,),
        )
        return int(row["count"]) if row else 0

    def count_saves_by_scenario(self) -> dict[str, int]:
        rows = self._fetch_all(
            """
            SELECT scenario_id, COUNT(*) AS count
            FROM saves
            GROUP BY scenario_id
            """,
            (),
        )
        return {str(row["scenario_id"]): int(row["count"]) for row in rows}

    def update_save_scenario(
        self,
        *,
        save_id: str,
        scenario_id: str,
    ) -> SaveRecord:
        self.connection.execute(
            """
            UPDATE saves
            SET scenario_id = ?
            WHERE id = ?
            """,
            (scenario_id, save_id),
        )
        self.commit()
        save = self.get_save(save_id)
        if save is None:
            raise ValueError(f"Unknown save id: {save_id}")
        return save

    def update_save_title(
        self,
        *,
        save_id: str,
        title: str,
    ) -> SaveRecord:
        self.connection.execute(
            """
            UPDATE saves
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (title.strip(), save_id),
        )
        self.commit()
        save = self.get_save(save_id)
        if save is None:
            raise ValueError(f"Unknown save id: {save_id}")
        return save

    def get_save(self, save_id: str) -> SaveRecord | None:
        row = self._fetch_one(
            """
            SELECT
                saves.id,
                saves.scenario_id,
                scenarios.title AS scenario_title,
                saves.title,
                saves.active,
                saves.custom_instructions,
                saves.owner_user_id,
                saves.created_at,
                saves.updated_at,
                saves.last_opened_at
            FROM saves
            LEFT JOIN scenarios ON scenarios.id = saves.scenario_id
            WHERE saves.id = ?
            """,
            (save_id,),
        )
        if row is None:
            return None
        return _save_from_row(row)

    def update_save_custom_instructions(
        self,
        *,
        save_id: str,
        custom_instructions: str,
    ) -> SaveRecord:
        self.connection.execute(
            """
            UPDATE saves
            SET custom_instructions = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (custom_instructions.strip(), save_id),
        )
        self.commit()
        save = self.get_save(save_id)
        if save is None:
            raise ValueError(f"Unknown save id: {save_id}")
        return save

    def touch_save_last_opened(self, save_id: str) -> None:
        self.connection.execute(
            """
            UPDATE saves
            SET last_opened_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
            WHERE id = ?
            """,
            (save_id,),
        )
        self.commit()

    def delete_save(self, save_id: str) -> bool:
        if self.get_save(save_id) is None:
            return False

        self.begin_transaction()
        try:
            for table_name in (
                "media_assets",
                "jobs",
                "save_loss_outcomes",
                "save_loss_condition_changes",
                "save_loss_conditions",
                "save_scenario_updates",
                "summaries",
                "memories",
                "state_changes",
                "world_state",
                "context_observations",
                "context_sources",
                "context_update_audit",
                "context_update_suggestions",
                "message_scene_presence",
                "message_visibility",
                "character_knowledge_edges",
                "character_text_message_revisions",
                "character_text_message_attachments",
                "character_text_provenance",
                "character_text_proactive_triggers",
                "character_contact_states",
                "character_text_messages",
                "character_text_thread_participants",
                "character_text_threads",
                "scene_snapshots",
                "entity_links",
                "active_threads",
                "characters",
                "locations",
                "save_turn_snapshots",
                "message_context_revisions",
                "message_revisions",
            ):
                self.connection.execute(
                    f"DELETE FROM {table_name} WHERE save_id = ?",
                    (save_id,),
                )
            self.connection.execute(
                "DELETE FROM messages WHERE save_id = ?",
                (save_id,),
            )
            cursor = self.connection.execute(
                "DELETE FROM saves WHERE id = ?",
                (save_id,),
            )
            self.delete_scoped_settings_for_save(save_id)
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise
        return cursor.rowcount > 0

    def load_save_details(
        self,
        save_id: str,
        *,
        message_limit: int | None = None,
        before_message_id: str | None = None,
    ) -> SaveDetailsRecord | None:
        save = self.get_save(save_id)
        if save is None:
            return None
        scenario = self.get_scenario(save.scenario_id)
        if scenario is None:
            return None
        scenario = self._effective_scenario_for_save(save_id, scenario)
        if message_limit is not None:
            message_page = self.list_message_page(
                save_id,
                before_message_id=before_message_id,
                limit=message_limit,
            )
            return SaveDetailsRecord(
                save=save,
                scenario=scenario,
                messages=message_page.messages,
                has_more_messages_before=message_page.has_more_before,
            )
        if before_message_id is not None:
            raise ValueError("before_message_id requires message_limit")
        return SaveDetailsRecord(
            save=save,
            scenario=scenario,
            messages=self.list_messages(save_id),
        )

    def context_candidate_revision_token(
        self,
        save_id: str,
        *,
        ignored_message_id: str | None = None,
    ) -> str:
        save_row = self._fetch_one(
            """
            SELECT id, scenario_id, title, active, custom_instructions
            FROM saves
            WHERE id = ?
            """,
            (save_id,),
        )
        scenario_row = None
        if save_row is not None:
            scenario_row = self._fetch_one(
                """
                SELECT id, type, title, premise, player_role, content_json, updated_at
                FROM scenarios
                WHERE id = ?
                """,
                (save_row["scenario_id"],),
            )
        revision_row = self._fetch_one(
            """
            SELECT revision
            FROM save_context_revisions
            WHERE save_id = ?
            """,
            (save_id,),
        )
        context_revision = 0 if revision_row is None else int(revision_row["revision"])
        ignored_message_revision = 0
        if ignored_message_id is not None:
            ignored_message_revision_row = self._fetch_one(
                """
                SELECT revision
                FROM message_context_revisions
                WHERE save_id = ? AND message_id = ?
                """,
                (save_id, ignored_message_id),
            )
            if ignored_message_revision_row is not None:
                ignored_message_revision = int(ignored_message_revision_row["revision"])
        payload = {
            "save": None if save_row is None else dict(save_row),
            "scenario": (
                None
                if scenario_row is None
                else {
                    "id": scenario_row["id"],
                    "type": scenario_row["type"],
                    "title_hash": _text_digest(scenario_row["title"]),
                    "premise_hash": _text_digest(scenario_row["premise"]),
                    "player_role_hash": _text_digest(scenario_row["player_role"]),
                    "content_hash": _text_digest(scenario_row["content_json"]),
                    "updated_at": scenario_row["updated_at"],
                }
            ),
            "context_revision": context_revision - ignored_message_revision,
        }
        return _json_digest(payload)

    def continuity_index_needs_sync(self, save_id: str) -> bool:
        row = self._fetch_one(
            """
            SELECT revision, indexed_revision
            FROM save_continuity_index_revisions
            WHERE save_id = ?
            """,
            (save_id,),
        )
        return self.continuity_index_dirty_source_count(save_id) > 0 or (
            row is None
            or int(row["revision"]) != int(row["indexed_revision"])
        )

    def continuity_index_requires_full_rebuild(self, save_id: str) -> bool:
        row = self._fetch_one(
            """
            SELECT indexed_revision
            FROM save_continuity_index_revisions
            WHERE save_id = ?
            """,
            (save_id,),
        )
        return row is None or int(row["indexed_revision"]) < 0

    def list_continuity_index_dirty_sources(
        self,
        save_id: str,
        *,
        limit: int,
    ) -> list[tuple[str, str, int]]:
        rows = self._fetch_all(
            """
            SELECT source_kind, source_id, dirty_generation
            FROM continuity_index_dirty_sources
            WHERE save_id = ?
            ORDER BY queued_at, source_kind, source_id
            LIMIT ?
            """,
            (save_id, max(0, limit)),
        )
        return [
            (
                str(row["source_kind"]),
                str(row["source_id"]),
                int(row["dirty_generation"]),
            )
            for row in rows
        ]

    def continuity_index_dirty_source_count(self, save_id: str) -> int:
        row = self._fetch_one(
            """
            SELECT COUNT(*) AS source_count
            FROM continuity_index_dirty_sources
            WHERE save_id = ?
            """,
            (save_id,),
        )
        return 0 if row is None else int(row["source_count"])

    def delete_continuity_index_dirty_source(
        self,
        save_id: str,
        *,
        source_kind: str,
        source_id: str,
        dirty_generation: int,
    ) -> None:
        self.connection.execute(
            """
            DELETE FROM continuity_index_dirty_sources
            WHERE save_id = ?
              AND source_kind = ?
              AND source_id = ?
              AND dirty_generation = ?
            """,
            (save_id, source_kind, source_id, dirty_generation),
        )

    def mark_continuity_index_synced(self, save_id: str) -> None:
        self.connection.execute(
            """
            INSERT INTO save_continuity_index_revisions(
                save_id,
                revision,
                indexed_revision,
                updated_at
            )
            VALUES (?, 0, 0, CURRENT_TIMESTAMP)
            ON CONFLICT(save_id) DO UPDATE SET
                indexed_revision = revision,
                updated_at = CURRENT_TIMESTAMP
            """,
            (save_id,),
        )
        self.connection.execute(
            "DELETE FROM continuity_index_dirty_sources WHERE save_id = ?",
            (save_id,),
        )

    def add_save_scenario_update(
        self,
        *,
        save_id: str,
        title: str,
        premise: str,
        player_role: str,
        content: dict[str, object],
        reason: str,
        provider: str,
        model: str,
        source_message_id: str | None = None,
        source_message_ids: tuple[str, ...] = (),
        update_id: str | None = None,
    ) -> SaveScenarioUpdateRecord:
        source_ids = _save_scenario_source_message_ids(
            source_message_id=source_message_id,
            source_message_ids=source_message_ids,
        )
        record = SaveScenarioUpdateRecord(
            id=update_id or _new_id(),
            save_id=save_id,
            source_message_id=source_message_id,
            title=title,
            premise=premise,
            player_role=player_role,
            content_json=_dump_json(content),
            source_message_ids_json=_dump_json(list(source_ids)),
            reason=reason,
            provider=provider,
            model=model,
        )
        self.connection.execute(
            """
            INSERT INTO save_scenario_updates(
                id, save_id, source_message_id, title, premise, player_role,
                content_json, source_message_ids_json, reason, provider, model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.source_message_id,
                record.title,
                record.premise,
                record.player_role,
                record.content_json,
                record.source_message_ids_json,
                record.reason,
                record.provider,
                record.model,
            ),
        )
        self.commit()
        return record

    def record_save_scenario_evolution(
        self,
        *,
        save_id: str,
        title: str,
        premise: str,
        player_role: str,
        content: dict[str, object],
        reason: str,
        provider: str,
        model: str,
        source_message_id: str | None = None,
        source_message_ids: tuple[str, ...] = (),
        update_id: str | None = None,
    ) -> SaveScenarioUpdateRecord:
        return self.add_save_scenario_update(
            save_id=save_id,
            title=title,
            premise=premise,
            player_role=player_role,
            content=content,
            reason=reason,
            provider=provider,
            model=model,
            source_message_id=source_message_id,
            source_message_ids=source_message_ids,
            update_id=update_id,
        )

    def get_active_save_scenario_update(
        self,
        save_id: str,
    ) -> SaveScenarioUpdateRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, source_message_id, title, premise, player_role,
                   content_json, source_message_ids_json, reason, provider, model,
                   created_at, archived_at
            FROM save_scenario_updates
            WHERE save_id = ? AND archived_at IS NULL
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (save_id,),
        )
        return SaveScenarioUpdateRecord(**dict(row)) if row else None

    def list_save_scenario_updates(
        self,
        save_id: str,
        *,
        include_archived: bool = False,
    ) -> list[SaveScenarioUpdateRecord]:
        archived_filter = "" if include_archived else "AND archived_at IS NULL"
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, source_message_id, title, premise, player_role,
                   content_json, source_message_ids_json, reason, provider, model,
                   created_at, archived_at
            FROM save_scenario_updates
            WHERE save_id = ? {archived_filter}
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )
        return [SaveScenarioUpdateRecord(**dict(row)) for row in rows]

    def list_save_scenario_evolution_audit(
        self,
        save_id: str,
        *,
        include_archived: bool = False,
    ) -> list[SaveScenarioUpdateRecord]:
        return self.list_save_scenario_updates(
            save_id,
            include_archived=include_archived,
        )

    def archive_save_scenario_updates_for_messages(
        self,
        *,
        save_id: str,
        message_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        if not message_ids:
            return frozenset()
        rows = self._fetch_all(
            """
            SELECT id, source_message_id, source_message_ids_json
            FROM save_scenario_updates
            WHERE save_id = ?
              AND archived_at IS NULL
            """,
            (save_id,),
        )
        archived_ids = frozenset(
            str(row["id"])
            for row in rows
            if _save_scenario_update_matches_messages(row, message_ids)
        )
        if not archived_ids:
            return frozenset()
        self.connection.execute(
            f"""
            UPDATE save_scenario_updates
            SET archived_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(archived_ids))})
            """,
            tuple(archived_ids),
        )
        self.commit()
        return archived_ids

    def restore_save_scenario_updates(
        self,
        update_ids: set[str] | frozenset[str],
    ) -> None:
        if not update_ids:
            return
        self.connection.execute(
            f"""
            UPDATE save_scenario_updates
            SET archived_at = NULL
            WHERE id IN ({_placeholders(len(update_ids))})
            """,
            tuple(update_ids),
        )
        self.commit()

    def _effective_scenario_for_save(
        self,
        save_id: str,
        base_scenario: ScenarioRecord,
    ) -> ScenarioRecord:
        update = self.get_active_save_scenario_update(save_id)
        if update is None:
            return base_scenario
        return ScenarioRecord(
            id=base_scenario.id,
            type=base_scenario.type,
            title=update.title,
            premise=update.premise,
            player_role=update.player_role,
            content_json=update.content_json,
            created_at=base_scenario.created_at,
            updated_at=base_scenario.updated_at,
        )

    def add_loss_condition(
        self,
        *,
        save_id: str,
        name: str,
        description: str,
        status: str = "active",
        source: str = "manual",
        key: str = "",
        label: str = "",
        severity: str = "",
        source_message_id: str | None = None,
        condition_id: str | None = None,
    ) -> LossConditionRecord:
        _validate_loss_condition_status(status)
        resolved_label = label or name
        resolved_key = key or _loss_condition_key(resolved_label)
        record = LossConditionRecord(
            id=condition_id or _new_id(),
            save_id=save_id,
            name=name,
            description=description,
            status=status,
            source=source,
            key=resolved_key,
            label=resolved_label,
            severity=severity,
            source_message_id=source_message_id,
        )
        self.connection.execute(
            """
            INSERT INTO save_loss_conditions(
                id, save_id, key, label, name, description, status, severity,
                source, source_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.key,
                record.label,
                record.name,
                record.description,
                record.status,
                record.severity,
                record.source,
                record.source_message_id,
            ),
        )
        self.commit()
        return self.get_loss_condition(record.id) or record

    def update_loss_condition(
        self,
        *,
        condition_id: str,
        name: str | None = None,
        description: str | None = None,
        status: str | None = None,
        source_message_id: str | None | object = ...,
    ) -> LossConditionRecord:
        current = self.get_loss_condition(condition_id)
        if current is None:
            raise ValueError(f"Unknown loss condition id: {condition_id}")
        next_status = current.status if status is None else status
        _validate_loss_condition_status(next_status)
        self.connection.execute(
            """
            UPDATE save_loss_conditions
            SET name = ?, label = ?, description = ?, status = ?,
                source_message_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                current.name if name is None else name,
                current.label if name is None else name,
                current.description if description is None else description,
                next_status,
                (
                    current.source_message_id
                    if source_message_id is ...
                    else cast(str | None, source_message_id)
                ),
                condition_id,
            ),
        )
        self.commit()
        updated = self.get_loss_condition(condition_id)
        if updated is None:
            raise ValueError(f"Unknown loss condition id: {condition_id}")
        return updated

    def get_loss_condition(self, condition_id: str) -> LossConditionRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, name, description, status, source, key, label,
                   severity, source_message_id, created_at, updated_at
            FROM save_loss_conditions
            WHERE id = ? AND archived_at IS NULL
            """,
            (condition_id,),
        )
        return LossConditionRecord(**dict(row)) if row else None

    def list_loss_conditions(
        self,
        save_id: str,
        *,
        include_archived: bool = False,
    ) -> list[LossConditionRecord]:
        archived_filter = "" if include_archived else "AND archived_at IS NULL"
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, name, description, status, source, key, label,
                   severity, source_message_id, created_at, updated_at
            FROM save_loss_conditions
            WHERE save_id = ? {archived_filter}
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )
        return [LossConditionRecord(**dict(row)) for row in rows]

    def archive_loss_condition(self, condition_id: str) -> None:
        self.connection.execute(
            """
            UPDATE save_loss_conditions
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (condition_id,),
        )
        self.commit()

    def upsert_save_loss_condition(
        self,
        *,
        save_id: str,
        key: str,
        label: str,
        description: str,
        status: str,
        severity: str,
        source_message_id: str | None = None,
        condition_id: str | None = None,
    ) -> LossConditionRecord:
        normalized_key = key.strip()
        existing = self._fetch_one(
            """
            SELECT id
            FROM save_loss_conditions
            WHERE save_id = ? AND key = ? AND archived_at IS NULL
            """,
            (save_id, normalized_key),
        )
        if existing is None:
            return self.add_loss_condition(
                save_id=save_id,
                key=normalized_key,
                label=label,
                name=label,
                description=description,
                status=_compat_condition_status(status),
                severity=severity,
                source="manual",
                source_message_id=source_message_id,
                condition_id=condition_id,
            )
        condition_id = str(existing["id"])
        self.connection.execute(
            """
            UPDATE save_loss_conditions
            SET label = ?, name = ?, description = ?, status = ?, severity = ?,
                source_message_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                label,
                label,
                description,
                _compat_condition_status(status),
                severity,
                source_message_id,
                condition_id,
            ),
        )
        self.commit()
        updated = self.get_loss_condition(condition_id)
        if updated is None:
            raise ValueError(f"Unknown loss condition id: {condition_id}")
        return updated

    def list_save_loss_conditions(
        self,
        save_id: str,
        *,
        include_archived: bool = False,
    ) -> list[LossConditionRecord]:
        return self.list_loss_conditions(
            save_id,
            include_archived=include_archived,
        )

    def add_loss_condition_change(
        self,
        *,
        save_id: str,
        operation: str,
        condition_id: str | None = None,
        source_message_id: str | None = None,
        before: dict[str, object] | None = None,
        after: dict[str, object] | None = None,
        reason: str = "",
        provider: str = "",
        model: str = "",
        change_id: str | None = None,
    ) -> LossConditionChangeRecord:
        record = LossConditionChangeRecord(
            id=change_id or _new_id(),
            save_id=save_id,
            condition_id=condition_id,
            source_message_id=source_message_id,
            operation=operation,
            before=before,
            after=after,
            reason=reason,
            provider=provider,
            model=model,
        )
        self.connection.execute(
            """
            INSERT INTO save_loss_condition_changes(
                id, save_id, condition_id, source_message_id, operation,
                before_json, after_json, reason, provider, model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.condition_id,
                record.source_message_id,
                record.operation,
                _dump_json(record.before) if record.before is not None else None,
                _dump_json(record.after) if record.after is not None else None,
                record.reason,
                record.provider,
                record.model,
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, condition_id, source_message_id, operation,
                   before_json, after_json, reason, provider, model, created_at,
                   archived_at
            FROM save_loss_condition_changes
            WHERE id = ?
            """,
            (record.id,),
        )
        return _loss_condition_change_from_row(row) if row else record

    def record_save_loss_condition_change(
        self,
        *,
        save_id: str,
        condition_id: str | None,
        operation: str,
        before_json: str | None = None,
        after_json: str | None = None,
        source_message_id: str | None = None,
        reason: str = "",
        provider: str = "",
        model: str = "",
        change_id: str | None = None,
    ) -> LossConditionChangeRecord:
        return self.add_loss_condition_change(
            save_id=save_id,
            condition_id=condition_id,
            source_message_id=source_message_id,
            operation=operation,
            before=_load_object(before_json) if before_json else None,
            after=_load_object(after_json) if after_json else None,
            reason=reason,
            provider=provider,
            model=model,
            change_id=change_id,
        )

    def list_loss_condition_changes(
        self,
        save_id: str,
        *,
        include_archived: bool = False,
    ) -> list[LossConditionChangeRecord]:
        archived_filter = "" if include_archived else "AND archived_at IS NULL"
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, condition_id, source_message_id, operation,
                   before_json, after_json, reason, provider, model, created_at,
                   archived_at
            FROM save_loss_condition_changes
            WHERE save_id = ? {archived_filter}
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )
        return [_loss_condition_change_from_row(row) for row in rows]

    def list_save_loss_condition_changes(
        self,
        save_id: str,
        *,
        include_archived: bool = False,
    ) -> list[LossConditionChangeRecord]:
        return self.list_loss_condition_changes(
            save_id,
            include_archived=include_archived,
        )

    def create_loss_outcome(
        self,
        *,
        save_id: str,
        condition_id: str | None,
        condition_name: str,
        triggering_message_id: str,
        explanation: str,
        evidence: dict[str, object],
        confidence: float,
        provider: str,
        model: str,
        epilogue_provider: str | None = None,
        epilogue_model: str | None = None,
        epilogue_message_id: str | None = None,
        epilogue_error: str | None = None,
        outcome_id: str | None = None,
        outcome_type: str = "loss_condition",
    ) -> LossOutcomeRecord:
        existing = self.get_active_loss_outcome(save_id)
        if existing is not None:
            raise ValueError("Save already has an active loss outcome")
        record = LossOutcomeRecord(
            id=outcome_id or _new_id(),
            save_id=save_id,
            condition_id=condition_id,
            condition_name=condition_name,
            triggering_message_id=triggering_message_id,
            explanation=explanation,
            evidence=evidence,
            confidence=confidence,
            provider=provider,
            model=model,
            epilogue_provider=epilogue_provider,
            epilogue_model=epilogue_model,
            epilogue_message_id=epilogue_message_id,
            epilogue_error=epilogue_error,
            outcome_type=outcome_type,
        )
        self.connection.execute(
            """
            INSERT INTO save_loss_outcomes(
                id, save_id, condition_id, condition_name, triggering_message_id,
                explanation, evidence_json, confidence, provider, model, outcome_type,
                epilogue_provider, epilogue_model, epilogue_message_id,
                epilogue_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.condition_id,
                record.condition_name,
                record.triggering_message_id,
                record.explanation,
                _dump_json(record.evidence),
                record.confidence,
                record.provider,
                record.model,
                record.outcome_type,
                record.epilogue_provider,
                record.epilogue_model,
                record.epilogue_message_id,
                record.epilogue_error,
            ),
        )
        self.commit()
        return self.get_loss_outcome(record.id) or record

    def trigger_save_loss_outcome(
        self,
        *,
        save_id: str,
        condition_id: str,
        source_message_id: str,
        title: str,
        body: str,
        epilogue: str,
        confidence: float,
        evidence: list[dict[str, object]],
        provider: str,
        model: str,
        outcome_id: str | None = None,
    ) -> LossOutcomeRecord:
        return self.create_loss_outcome(
            save_id=save_id,
            condition_id=condition_id,
            condition_name=title,
            triggering_message_id=source_message_id,
            explanation=body,
            evidence={"items": evidence, "epilogue": epilogue},
            confidence=confidence,
            provider=provider,
            model=model,
            outcome_id=outcome_id,
        )

    def update_loss_outcome_epilogue(
        self,
        *,
        outcome_id: str,
        epilogue_provider: str | None,
        epilogue_model: str | None,
        epilogue_message_id: str | None,
        epilogue_error: str | None = None,
    ) -> LossOutcomeRecord:
        self.connection.execute(
            """
            UPDATE save_loss_outcomes
            SET epilogue_provider = ?, epilogue_model = ?,
                epilogue_message_id = ?, epilogue_error = ?
            WHERE id = ?
            """,
            (
                epilogue_provider,
                epilogue_model,
                epilogue_message_id,
                epilogue_error,
                outcome_id,
            ),
        )
        self.commit()
        outcome = self.get_loss_outcome(outcome_id)
        if outcome is None:
            raise ValueError(f"Unknown loss outcome id: {outcome_id}")
        return outcome

    def get_loss_outcome(self, outcome_id: str) -> LossOutcomeRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, condition_id, condition_name, triggering_message_id,
                   explanation, evidence_json, confidence, provider, model,
                   outcome_type, epilogue_provider, epilogue_model,
                   epilogue_message_id, epilogue_error, created_at, archived_at
            FROM save_loss_outcomes
            WHERE id = ?
            """,
            (outcome_id,),
        )
        return _loss_outcome_from_row(row) if row else None

    def get_active_save_loss_outcome(self, save_id: str) -> LossOutcomeRecord | None:
        return self.get_active_loss_outcome(save_id)

    def get_active_loss_outcome(self, save_id: str) -> LossOutcomeRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, condition_id, condition_name, triggering_message_id,
                   explanation, evidence_json, confidence, provider, model,
                   outcome_type, epilogue_provider, epilogue_model,
                   epilogue_message_id, epilogue_error, created_at, archived_at
            FROM save_loss_outcomes
            WHERE save_id = ? AND archived_at IS NULL
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (save_id,),
        )
        return _loss_outcome_from_row(row) if row else None

    def list_loss_outcomes(
        self,
        save_id: str,
        *,
        include_archived: bool = False,
    ) -> list[LossOutcomeRecord]:
        archived_filter = "" if include_archived else "AND archived_at IS NULL"
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, condition_id, condition_name, triggering_message_id,
                   explanation, evidence_json, confidence, provider, model,
                   outcome_type, epilogue_provider, epilogue_model,
                   epilogue_message_id, epilogue_error, created_at, archived_at
            FROM save_loss_outcomes
            WHERE save_id = ? {archived_filter}
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )
        return [_loss_outcome_from_row(row) for row in rows]

    def list_save_loss_outcomes(
        self,
        save_id: str,
        *,
        include_archived: bool = False,
    ) -> list[LossOutcomeRecord]:
        return self.list_loss_outcomes(save_id, include_archived=include_archived)

    def archive_loss_outcomes_for_messages(
        self,
        *,
        save_id: str,
        message_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        if not message_ids:
            return frozenset()
        rows = self._fetch_all(
            f"""
            SELECT id
            FROM save_loss_outcomes
            WHERE save_id = ?
              AND archived_at IS NULL
              AND (
                  triggering_message_id IN ({_placeholders(len(message_ids))})
                  OR epilogue_message_id IN ({_placeholders(len(message_ids))})
              )
            """,
            (save_id, *tuple(message_ids), *tuple(message_ids)),
        )
        outcome_ids = frozenset(str(row["id"]) for row in rows)
        if not outcome_ids:
            return frozenset()
        self.connection.execute(
            f"""
            UPDATE save_loss_outcomes
            SET archived_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(outcome_ids))})
            """,
            tuple(outcome_ids),
        )
        self.commit()
        return outcome_ids

    def archive_save_loss_outcomes_for_messages(
        self,
        *,
        save_id: str,
        message_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        return self.archive_loss_outcomes_for_messages(
            save_id=save_id,
            message_ids=message_ids,
        )

    def archive_loss_outcomes_for_conditions(
        self,
        *,
        save_id: str,
        condition_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        if not condition_ids:
            return frozenset()
        rows = self._fetch_all(
            f"""
            SELECT id
            FROM save_loss_outcomes
            WHERE save_id = ?
              AND archived_at IS NULL
              AND condition_id IN ({_placeholders(len(condition_ids))})
            """,
            (save_id, *tuple(condition_ids)),
        )
        outcome_ids = frozenset(str(row["id"]) for row in rows)
        if not outcome_ids:
            return frozenset()
        self.connection.execute(
            f"""
            UPDATE save_loss_outcomes
            SET archived_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(outcome_ids))})
            """,
            tuple(outcome_ids),
        )
        self.commit()
        return outcome_ids

    def restore_loss_outcomes(self, outcome_ids: set[str] | frozenset[str]) -> None:
        if not outcome_ids:
            return
        self.connection.execute(
            f"""
            UPDATE save_loss_outcomes
            SET archived_at = NULL
            WHERE id IN ({_placeholders(len(outcome_ids))})
            """,
            tuple(outcome_ids),
        )
        self.commit()

    def archive_loss_condition_changes_for_messages(
        self,
        *,
        save_id: str,
        message_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        if not message_ids:
            return frozenset()
        rows = self._fetch_all(
            f"""
            SELECT id
            FROM save_loss_condition_changes
            WHERE save_id = ?
              AND archived_at IS NULL
              AND source_message_id IN ({_placeholders(len(message_ids))})
            """,
            (save_id, *tuple(message_ids)),
        )
        change_ids = frozenset(str(row["id"]) for row in rows)
        if not change_ids:
            return frozenset()
        self.connection.execute(
            f"""
            UPDATE save_loss_condition_changes
            SET archived_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(change_ids))})
            """,
            tuple(change_ids),
        )
        self.commit()
        return change_ids

    def archive_loss_condition_changes_for_conditions(
        self,
        *,
        save_id: str,
        condition_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        if not condition_ids:
            return frozenset()
        rows = self._fetch_all(
            f"""
            SELECT id
            FROM save_loss_condition_changes
            WHERE save_id = ?
              AND archived_at IS NULL
              AND condition_id IN ({_placeholders(len(condition_ids))})
            """,
            (save_id, *tuple(condition_ids)),
        )
        change_ids = frozenset(str(row["id"]) for row in rows)
        if not change_ids:
            return frozenset()
        self.connection.execute(
            f"""
            UPDATE save_loss_condition_changes
            SET archived_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(change_ids))})
            """,
            tuple(change_ids),
        )
        self.commit()
        return change_ids

    def archive_save_loss_condition_changes_for_messages(
        self,
        *,
        save_id: str,
        message_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        return self.archive_loss_condition_changes_for_messages(
            save_id=save_id,
            message_ids=message_ids,
        )

    def rebuild_save_loss_conditions_from_changes(self, save_id: str) -> None:
        changes = self.list_loss_condition_changes(save_id)
        snapshots: dict[str, dict[str, object]] = {}
        condition_ids = {
            str(change.condition_id)
            for change in self.list_loss_condition_changes(
                save_id,
                include_archived=True,
            )
            if change.condition_id is not None
        }
        for change in changes:
            if change.condition_id is None:
                continue
            if change.after is None:
                snapshots.pop(change.condition_id, None)
            else:
                snapshots[change.condition_id] = change.after
        for condition_id in condition_ids:
            snapshot = snapshots.get(condition_id)
            if snapshot is None:
                self.archive_loss_condition(condition_id)
                continue
            existing = self.get_loss_condition(condition_id)
            if existing is not None:
                self.update_loss_condition(
                    condition_id=condition_id,
                    name=str(snapshot.get("name") or snapshot.get("label") or ""),
                    description=str(snapshot.get("description") or ""),
                    status=str(snapshot.get("status") or "active"),
                    source_message_id=(
                        str(snapshot["source_message_id"])
                        if snapshot.get("source_message_id") is not None
                        else None
                    ),
                )
                continue
            if any(
                condition.id == condition_id
                for condition in self.list_loss_conditions(
                    save_id,
                    include_archived=True,
                )
            ):
                condition_ids_to_archive = frozenset({condition_id})
                self.archive_loss_condition_changes_for_conditions(
                    save_id=save_id,
                    condition_ids=condition_ids_to_archive,
                )
                self.archive_loss_outcomes_for_conditions(
                    save_id=save_id,
                    condition_ids=condition_ids_to_archive,
                )
                continue
            self.upsert_save_loss_condition(
                save_id=save_id,
                key=str(snapshot.get("key") or condition_id),
                label=str(snapshot.get("label") or snapshot.get("name") or ""),
                description=str(snapshot.get("description") or ""),
                status=str(snapshot.get("status") or "active"),
                severity=str(snapshot.get("severity") or ""),
                source_message_id=(
                    str(snapshot["source_message_id"])
                    if snapshot.get("source_message_id") is not None
                    else None
                ),
                condition_id=condition_id,
            )

    def restore_loss_condition_changes(
        self,
        change_ids: set[str] | frozenset[str],
    ) -> None:
        if not change_ids:
            return
        self.connection.execute(
            f"""
            UPDATE save_loss_condition_changes
            SET archived_at = NULL
            WHERE id IN ({_placeholders(len(change_ids))})
            """,
            tuple(change_ids),
        )
        self.commit()

    def append_message(
        self,
        *,
        save_id: str,
        role: str,
        body: str,
        speaker_name: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        token_estimate: int | None = None,
        message_id: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        safety_transition: str = "",
        content_rating: str = "unclassified",
        touch_save_updated_at: bool = True,
    ) -> MessageRecord:
        body, safety_transition = normalize_message_safety(
            body=body,
            role=role,
            safety_transition=safety_transition,
        )
        record = MessageRecord(
            id=message_id or _new_id(),
            save_id=save_id,
            role=role,
            body=body,
            speaker_name=speaker_name,
            provider=provider,
            model=model,
            token_estimate=token_estimate,
            deleted_at=None,
            safety_transition=safety_transition,
            content_rating=content_rating,
        )
        self.connection.execute(
            """
            INSERT INTO messages(
                id, save_id, role, speaker_name, body, provider, model,
                token_estimate, created_at, updated_at, safety_transition,
                content_rating
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE(?, CURRENT_TIMESTAMP),
                COALESCE(?, strftime('%Y-%m-%d %H:%M:%f', 'now')),
                ?, ?
            )
            """,
            (
                record.id,
                record.save_id,
                record.role,
                record.speaker_name,
                record.body,
                record.provider,
                record.model,
                record.token_estimate,
                created_at,
                updated_at,
                record.safety_transition,
                record.content_rating,
            ),
        )
        if touch_save_updated_at:
            self.connection.execute(
                """
                UPDATE saves
                SET updated_at = (
                    SELECT updated_at
                    FROM messages
                    WHERE id = ?
                )
                WHERE id = ?
                """,
                (record.id, record.save_id),
            )
        if role == "narrator":
            scene = self._fetch_one(
                """
                SELECT id, scene_generation
                FROM scene_snapshots
                WHERE save_id = ?
                """,
                (save_id,),
            )
            self.archive_stale_scene_scratch(
                save_id=save_id,
                current_scene_snapshot_id=(
                    str(scene["id"]) if scene is not None else None
                ),
                current_scene_generation=(
                    int(scene["scene_generation"])
                    if scene is not None
                    else None
                ),
                current_turn_number=self.count_active_messages_by_role(
                    save_id,
                    roles=("narrator",),
                )["narrator"],
            )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition, content_rating
            FROM messages
            WHERE id = ?
            """,
            (record.id,),
        )
        return MessageRecord(**dict(row)) if row else record

    def get_message(
        self,
        *,
        save_id: str,
        message_id: str,
        include_deleted: bool = False,
    ) -> MessageRecord | None:
        deleted_filter = "" if include_deleted else "AND deleted_at IS NULL"
        row = self._fetch_one(
            f"""
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition, content_rating
            FROM messages
            WHERE save_id = ? AND id = ? {deleted_filter}
            """,
            (save_id, message_id),
        )
        return MessageRecord(**dict(row)) if row else None

    def list_messages(
        self,
        save_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[MessageRecord]:
        deleted_filter = "" if include_deleted else "AND deleted_at IS NULL"
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition, content_rating
            FROM messages
            WHERE save_id = ? {deleted_filter}
            ORDER BY rowid
            """,
            (save_id,),
        )
        return [MessageRecord(**dict(row)) for row in rows]

    def list_recent_messages_visible_to_characters(
        self,
        save_id: str,
        *,
        character_ids: set[str] | frozenset[str] | tuple[str, ...],
        limit: int,
    ) -> list[MessageRecord]:
        if limit <= 0:
            return []
        scoped_character_ids = tuple(sorted(set(character_ids)))
        visibility_filter = ""
        params: list[object] = [save_id]
        if scoped_character_ids:
            visibility_filter = f"""
                AND NOT EXISTS (
                    SELECT 1
                    FROM message_visibility AS visibility
                    WHERE visibility.save_id = message.save_id
                      AND visibility.message_id = message.id
                      AND visibility.character_id IN (
                            {_placeholders(len(scoped_character_ids))}
                          )
                      AND visibility.visibility = 'not_visible'
                )
            """
            params.extend(scoped_character_ids)
        params.append(limit)
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition, content_rating
            FROM messages AS message
            WHERE save_id = ?
              AND deleted_at IS NULL
              {visibility_filter}
            ORDER BY message.rowid DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows.reverse()
        return [MessageRecord(**dict(row)) for row in rows]

    def list_messages_by_ids(
        self,
        save_id: str,
        message_ids: Iterable[str],
        *,
        include_deleted: bool = False,
    ) -> list[MessageRecord]:
        ids = list(dict.fromkeys(message_ids))
        if not ids:
            return []
        deleted_filter = "" if include_deleted else "AND deleted_at IS NULL"
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition, content_rating
            FROM messages
            WHERE save_id = ?
              AND id IN ({_placeholders(len(ids))})
              {deleted_filter}
            ORDER BY rowid
            """,
            (save_id, *ids),
        )
        return [MessageRecord(**dict(row)) for row in rows]

    def count_active_messages_by_role(
        self,
        save_id: str,
        *,
        roles: tuple[str, ...],
        created_at_lte: str | None = None,
    ) -> dict[str, int]:
        if not roles:
            return {}
        conditions = [
            "save_id = ?",
            "deleted_at IS NULL",
            f"role IN ({_placeholders(len(roles))})",
        ]
        params: list[object] = [save_id, *roles]
        if created_at_lte is not None:
            conditions.append("created_at <= ?")
            params.append(created_at_lte)
        rows = self._fetch_all(
            f"""
            SELECT role, COUNT(*) AS message_count
            FROM messages
            WHERE {' AND '.join(conditions)}
            GROUP BY role
            """,
            tuple(params),
        )
        counts = {role: 0 for role in roles}
        counts.update(
            {
                str(row["role"]): int(row["message_count"])
                for row in rows
                if row["role"] in counts
            }
        )
        return counts

    def latest_active_message_created_at(
        self,
        save_id: str,
        *,
        role: str | None = None,
    ) -> str | None:
        conditions = ["save_id = ?", "deleted_at IS NULL"]
        params: list[object] = [save_id]
        if role is not None:
            conditions.append("role = ?")
            params.append(role)
        row = self._fetch_one(
            f"""
            SELECT MAX(created_at) AS latest_created_at
            FROM messages
            WHERE {' AND '.join(conditions)}
            """,
            tuple(params),
        )
        if row is None:
            return None
        value = row["latest_created_at"]
        return value if isinstance(value, str) else None

    def latest_active_message_rowid(self, save_id: str) -> int | None:
        row = self._fetch_one(
            """
            SELECT rowid
            FROM messages
            WHERE save_id = ? AND deleted_at IS NULL
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (save_id,),
        )
        return int(row["rowid"]) if row is not None else None

    def find_active_message_after_rowid(
        self,
        save_id: str,
        *,
        after_rowid: int | None,
        role: str,
        body: str,
        speaker_name: str | None,
    ) -> MessageRecord | None:
        rowid_filter = "" if after_rowid is None else "AND rowid > ?"
        params: tuple[Any, ...]
        if after_rowid is None:
            params = (save_id, role, body, speaker_name, speaker_name)
        else:
            params = (save_id, after_rowid, role, body, speaker_name, speaker_name)
        row = self._fetch_one(
            f"""
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition, content_rating
            FROM messages
            WHERE save_id = ?
              {rowid_filter}
              AND deleted_at IS NULL
              AND role = ?
              AND body = ?
              AND ((speaker_name IS NULL AND ? IS NULL) OR speaker_name = ?)
            ORDER BY rowid
            LIMIT 1
            """,
            params,
        )
        return MessageRecord(**dict(row)) if row else None

    def list_message_page(
        self,
        save_id: str,
        *,
        before_message_id: str | None = None,
        limit: int = 80,
        include_deleted: bool = False,
    ) -> MessagePageRecord:
        if limit < 1:
            raise ValueError("Message page limit must be at least 1")
        deleted_filter = "" if include_deleted else "AND deleted_at IS NULL"
        before_filter = ""
        params: tuple[Any, ...] = (save_id,)
        if before_message_id is not None:
            before_row = self._fetch_one(
                f"""
                SELECT rowid
                FROM messages
                WHERE save_id = ? AND id = ? {deleted_filter}
                """,
                (save_id, before_message_id),
            )
            if before_row is None:
                raise ValueError(f"Unknown active message id: {before_message_id}")
            before_filter = "AND rowid < ?"
            params = (save_id, before_row["rowid"])
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition, content_rating
            FROM messages
            WHERE save_id = ? {deleted_filter} {before_filter}
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (*params, limit + 1),
        )
        page_rows = rows[:limit]
        return MessagePageRecord(
            messages=[MessageRecord(**dict(row)) for row in reversed(page_rows)],
            has_more_before=len(rows) > limit,
        )

    def list_chat_history_message_page(
        self,
        save_id: str,
        *,
        selected_filter: str = "all",
        before_message_id: str | None = None,
        limit: int = 80,
    ) -> MessagePageRecord:
        if limit < 1:
            raise ValueError("Message page limit must be at least 1")
        filter_clause = _chat_history_message_filter_clause(selected_filter)
        before_filter = ""
        params: tuple[Any, ...] = (save_id,)
        if before_message_id is not None:
            before_row = self._fetch_one(
                """
                SELECT rowid
                FROM messages
                WHERE save_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (save_id, before_message_id),
            )
            if before_row is None:
                raise ValueError(f"Unknown active message id: {before_message_id}")
            before_filter = "AND rowid < ?"
            params = (save_id, before_row["rowid"])
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition, content_rating
            FROM messages
            WHERE save_id = ? AND deleted_at IS NULL {before_filter}
              {filter_clause}
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (*params, limit + 1),
        )
        page_rows = rows[:limit]
        return MessagePageRecord(
            messages=[MessageRecord(**dict(row)) for row in reversed(page_rows)],
            has_more_before=len(rows) > limit,
        )

    def count_chat_history_messages(
        self,
        save_id: str,
        *,
        selected_filter: str = "all",
    ) -> int:
        filter_clause = _chat_history_message_filter_clause(selected_filter)
        row = self._fetch_one(
            f"""
            SELECT COUNT(*) AS message_count
            FROM messages
            WHERE save_id = ? AND deleted_at IS NULL
              {filter_clause}
            """,
            (save_id,),
        )
        return int(row["message_count"]) if row is not None else 0

    def image_counts_for_messages(
        self,
        *,
        save_id: str,
        message_ids: Iterable[str],
    ) -> dict[str, int]:
        ids = list(dict.fromkeys(message_ids))
        if not ids:
            return {}
        rows = self._fetch_all(
            f"""
            SELECT source_message_id, COUNT(*) AS image_count
            FROM media_assets
            WHERE save_id = ?
              AND archived_at IS NULL
              AND type = 'image'
              AND source_message_id IN ({_placeholders(len(ids))})
            GROUP BY source_message_id
            """,
            (save_id, *ids),
        )
        return {
            str(row["source_message_id"]): int(row["image_count"])
            for row in rows
            if row["source_message_id"] is not None
        }

    def update_message_body(
        self,
        *,
        save_id: str,
        message_id: str,
        body: str,
        content_rating: str = "unclassified",
        safety_transition: str = "",
    ) -> MessageRecord:
        existing = self._fetch_one(
            "SELECT role FROM messages "
            "WHERE save_id = ? AND id = ? AND deleted_at IS NULL",
            (save_id, message_id),
        )
        if existing is None:
            raise ValueError(f"Unknown active message id: {message_id}")
        body, safety_transition = normalize_message_safety(
            body=body,
            role=str(existing["role"]),
            safety_transition=safety_transition,
        )
        self.connection.execute(
            """
            UPDATE messages
            SET body = ?, token_estimate = NULL, safety_transition = ?,
                content_rating = ?,
                updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
                    || ':' || lower(hex(randomblob(4)))
            WHERE save_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (body, safety_transition, content_rating, save_id, message_id),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition, content_rating
            FROM messages
            WHERE save_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (save_id, message_id),
        )
        if row is None:
            raise ValueError(f"Unknown active message id: {message_id}")
        return MessageRecord(**dict(row))

    def add_message_revision(
        self,
        *,
        save_id: str,
        message_id: str,
        previous_body: str,
        new_body: str,
        diff_unified: str,
        reconciliation_status: str = "queued",
        reconciliation_error: str | None = None,
        revision_id: str | None = None,
        revision_number: int | None = None,
    ) -> MessageRevisionRecord:
        if reconciliation_status not in MESSAGE_REVISION_RECONCILIATION_STATUSES:
            raise ValueError(
                f"Unsupported message revision reconciliation status: "
                f"{reconciliation_status}"
            )
        resolved_revision_number = revision_number
        if resolved_revision_number is None:
            row = self._fetch_one(
                """
                SELECT COALESCE(MAX(revision_number), 0) + 1 AS revision_number
                FROM message_revisions
                WHERE message_id = ?
                """,
                (message_id,),
            )
            resolved_revision_number = int(row["revision_number"] if row else 1)
        reconciled_at_expression = (
            "CURRENT_TIMESTAMP"
            if reconciliation_status in {"succeeded", "skipped", "failed"}
            else "NULL"
        )
        record_id = revision_id or _new_id()
        self.connection.execute(
            f"""
            INSERT INTO message_revisions(
                id, save_id, message_id, revision_number, previous_body,
                new_body, diff_unified, reconciliation_status,
                reconciliation_error, reconciled_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {reconciled_at_expression})
            """,
            (
                record_id,
                save_id,
                message_id,
                resolved_revision_number,
                previous_body,
                new_body,
                diff_unified,
                reconciliation_status,
                reconciliation_error,
            ),
        )
        self.commit()
        revision = self.get_message_revision(record_id)
        if revision is None:
            raise ValueError(f"Unknown message revision id: {record_id}")
        return revision

    def get_message_revision(
        self,
        revision_id: str,
    ) -> MessageRevisionRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, message_id, revision_number, previous_body,
                   new_body, diff_unified, reconciliation_status,
                   reconciliation_error, created_at, reconciled_at
            FROM message_revisions
            WHERE id = ?
            """,
            (revision_id,),
        )
        return _message_revision_from_row(row) if row else None

    def list_message_revisions(
        self,
        *,
        save_id: str,
        message_id: str | None = None,
    ) -> list[MessageRevisionRecord]:
        message_filter = "" if message_id is None else "AND message_id = ?"
        params: tuple[Any, ...] = (
            (save_id,) if message_id is None else (save_id, message_id)
        )
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, message_id, revision_number, previous_body,
                   new_body, diff_unified, reconciliation_status,
                   reconciliation_error, created_at, reconciled_at
            FROM message_revisions
            WHERE save_id = ? {message_filter}
            ORDER BY message_id, revision_number, rowid
            """,
            params,
        )
        return [_message_revision_from_row(row) for row in rows]

    def message_revision_metadata(
        self,
        save_id: str,
    ) -> dict[str, MessageRevisionMetadataRecord]:
        rows = self._fetch_all(
            """
            SELECT message_id, COUNT(*) AS revision_count, MAX(created_at) AS edited_at
            FROM message_revisions
            WHERE save_id = ?
            GROUP BY message_id
            """,
            (save_id,),
        )
        return {
            str(row["message_id"]): MessageRevisionMetadataRecord(
                message_id=str(row["message_id"]),
                revision_count=int(row["revision_count"]),
                edited_at=cast(str | None, row["edited_at"]),
            )
            for row in rows
        }

    def message_revision_metadata_for_messages(
        self,
        save_id: str,
        message_ids: Iterable[str],
    ) -> dict[str, MessageRevisionMetadataRecord]:
        ids = list(dict.fromkeys(message_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._fetch_all(
            f"""
            SELECT message_id, COUNT(*) AS revision_count, MAX(created_at) AS edited_at
            FROM message_revisions
            WHERE save_id = ? AND message_id IN ({placeholders})
            GROUP BY message_id
            """,
            (save_id, *ids),
        )
        return {
            str(row["message_id"]): MessageRevisionMetadataRecord(
                message_id=str(row["message_id"]),
                revision_count=int(row["revision_count"]),
                edited_at=cast(str | None, row["edited_at"]),
            )
            for row in rows
        }

    def mark_message_revision_reconciled(
        self,
        revision_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> MessageRevisionRecord:
        if status not in MESSAGE_REVISION_RECONCILIATION_STATUSES:
            raise ValueError(
                f"Unsupported message revision reconciliation status: {status}"
            )
        self.connection.execute(
            """
            UPDATE message_revisions
            SET reconciliation_status = ?,
                reconciliation_error = ?,
                reconciled_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, error, revision_id),
        )
        self.commit()
        revision = self.get_message_revision(revision_id)
        if revision is None:
            raise ValueError(f"Unknown message revision id: {revision_id}")
        return revision

    def list_message_ids(self, save_id: str) -> set[str]:
        rows = self._fetch_all(
            """
            SELECT id
            FROM messages
            WHERE save_id = ?
            """,
            (save_id,),
        )
        return {str(row["id"]) for row in rows}

    def archive_message(self, message_id: str) -> None:
        row = self._fetch_one(
            "SELECT save_id FROM messages WHERE id = ?",
            (message_id,),
        )
        self.connection.execute(
            """
            UPDATE messages
            SET deleted_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (message_id,),
        )
        if row is not None:
            self.delete_message_scene_presence_for_messages(
                save_id=str(row["save_id"]),
                message_ids={message_id},
            )
        self.commit()

    def archive_messages_from(
        self,
        *,
        save_id: str,
        message_id: str,
    ) -> list[MessageRecord]:
        rows = self._fetch_all(
            """
            SELECT id, save_id, role, body, speaker_name, provider, model,
                   token_estimate, deleted_at, created_at, updated_at,
                   safety_transition, content_rating
            FROM messages
            WHERE save_id = ?
              AND deleted_at IS NULL
              AND rowid >= (
                  SELECT rowid
                  FROM messages
                  WHERE save_id = ? AND id = ? AND deleted_at IS NULL
              )
            ORDER BY rowid
            """,
            (save_id, save_id, message_id),
        )
        records = [MessageRecord(**dict(row)) for row in rows]
        if not records:
            return []
        self.connection.execute(
            """
            UPDATE messages
            SET deleted_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND deleted_at IS NULL
              AND rowid >= (
                  SELECT rowid
                  FROM messages
                  WHERE save_id = ? AND id = ?
              )
            """,
            (save_id, save_id, message_id),
        )
        self.delete_message_scene_presence_for_messages(
            save_id=save_id,
            message_ids={record.id for record in records},
        )
        self.commit()
        return records

    def restore_messages(self, message_ids: set[str] | frozenset[str]) -> None:
        if not message_ids:
            return
        self.connection.execute(
            f"""
            UPDATE messages
            SET deleted_at = NULL
            WHERE id IN ({_placeholders(len(message_ids))})
            """,
            tuple(message_ids),
        )
        self.commit()

    def upsert_world_state(
        self,
        *,
        save_id: str,
        key: str,
        value: dict[str, object],
        category: str = "",
        confidence: float = 1.0,
        source_message_id: str | None = None,
        state_id: str | None = None,
    ) -> WorldStateRecord:
        existing = self._fetch_one(
            "SELECT id FROM world_state WHERE save_id = ? AND key = ?",
            (save_id, key),
        )
        record = WorldStateRecord(
            id=existing["id"] if existing else state_id or _new_id(),
            save_id=save_id,
            key=key,
            value=value,
            category=category,
            confidence=confidence,
            source_message_id=source_message_id,
        )
        self.connection.execute(
            """
            INSERT INTO world_state(
                id, save_id, key, value_json, category, confidence, source_message_id
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
                record.id,
                record.save_id,
                record.key,
                _dump_json(record.value),
                record.category,
                record.confidence,
                record.source_message_id,
            ),
        )
        self.commit()
        return record

    def add_state_change(
        self,
        *,
        save_id: str,
        operation: str,
        state_key: str,
        before_json: str | None = None,
        after_json: str | None = None,
        source_message_id: str | None = None,
        change_id: str | None = None,
    ) -> StateChangeRecord:
        record = StateChangeRecord(
            id=change_id or _new_id(),
            save_id=save_id,
            source_message_id=source_message_id,
            operation=operation,
            state_key=state_key,
            before_json=before_json,
            after_json=after_json,
        )
        self.connection.execute(
            """
            INSERT INTO state_changes(
                id, save_id, source_message_id, operation, state_key,
                before_json, after_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.source_message_id,
                record.operation,
                record.state_key,
                record.before_json,
                record.after_json,
            ),
        )
        self.commit()
        return record

    def list_state_changes(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[StateChangeRecord]:
        order_sql = (
            "ORDER BY created_at DESC, rowid DESC"
            if limit is not None
            else "ORDER BY created_at, rowid"
        )
        limit_sql = "LIMIT ?" if limit is not None else ""
        params: tuple[object, ...] = (
            (save_id, max(0, limit))
            if limit is not None
            else (save_id,)
        )
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, source_message_id, operation, state_key,
                   before_json, after_json
            FROM state_changes
            WHERE save_id = ?
            {order_sql}
            {limit_sql}
            """,
            params,
        )
        if limit is not None:
            rows.reverse()
        return [StateChangeRecord(**dict(row)) for row in rows]

    def delete_world_state(
        self,
        *,
        save_id: str,
        key: str,
    ) -> None:
        self.connection.execute(
            "DELETE FROM world_state WHERE save_id = ? AND key = ?",
            (save_id, key),
        )
        self.commit()

    def archive_world_state(
        self,
        *,
        save_id: str,
        key: str,
    ) -> None:
        self.connection.execute(
            """
            UPDATE world_state
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ? AND key = ?
            """,
            (save_id, key),
        )
        self.commit()

    def archive_world_state_if_unchanged(
        self,
        *,
        save_id: str,
        world_state_id: str,
        key: str,
        value: dict[str, object],
    ) -> bool:
        cursor = self.connection.execute(
            """
            UPDATE world_state
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
                AND id = ?
                AND key = ?
                AND value_json = ?
                AND archived_at IS NULL
            """,
            (save_id, world_state_id, key, _dump_json(value)),
        )
        self.commit()
        return cursor.rowcount > 0

    def list_world_state(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[WorldStateRecord]:
        return self._list_world_state(
            save_id,
            include_archived=False,
            limit=limit,
        )

    def list_world_state_including_archived(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[WorldStateRecord]:
        return self._list_world_state(
            save_id,
            include_archived=True,
            limit=limit,
        )

    def _list_world_state(
        self,
        save_id: str,
        *,
        include_archived: bool,
        limit: int | None = None,
    ) -> list[WorldStateRecord]:
        archived_filter = "" if include_archived else "AND archived_at IS NULL"
        limit_sql = "LIMIT ?" if limit is not None else ""
        order_sql = (
            "ORDER BY updated_at DESC, rowid DESC"
            if limit is not None
            else "ORDER BY key"
        )
        params: tuple[object, ...] = (
            (save_id, max(0, limit))
            if limit is not None
            else (save_id,)
        )
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, key, value_json, category, confidence, source_message_id
            FROM world_state
            WHERE save_id = ? {archived_filter}
            {order_sql}
            {limit_sql}
            """,
            params,
        )
        if limit is not None:
            rows.reverse()
        return [
            WorldStateRecord(
                id=row["id"],
                save_id=row["save_id"],
                key=row["key"],
                value=_load_object(row["value_json"]),
                category=row["category"],
                confidence=row["confidence"],
                source_message_id=row["source_message_id"],
            )
            for row in rows
        ]

    def get_world_state_by_id(
        self,
        save_id: str,
        state_id: str,
    ) -> WorldStateRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, key, value_json, category, confidence,
                   source_message_id
            FROM world_state
            WHERE save_id = ? AND id = ? AND archived_at IS NULL
            """,
            (save_id, state_id),
        )
        if row is None:
            return None
        return WorldStateRecord(
            id=row["id"],
            save_id=row["save_id"],
            key=row["key"],
            value=_load_object(row["value_json"]),
            category=row["category"],
            confidence=row["confidence"],
            source_message_id=row["source_message_id"],
        )

    def list_world_state_by_keys(
        self,
        save_id: str,
        keys: set[str] | frozenset[str],
    ) -> list[WorldStateRecord]:
        if not keys:
            return []
        rows = self._fetch_all(
            """
            SELECT id, save_id, key, value_json, category, confidence,
                   source_message_id
            FROM world_state
            WHERE save_id = ?
              AND archived_at IS NULL
              AND key IN (SELECT CAST(value AS TEXT) FROM json_each(?))
            ORDER BY key
            """,
            (save_id, _dump_json(sorted(keys))),
        )
        return [
            WorldStateRecord(
                id=row["id"],
                save_id=row["save_id"],
                key=row["key"],
                value=_load_object(row["value_json"]),
                category=row["category"],
                confidence=row["confidence"],
                source_message_id=row["source_message_id"],
            )
            for row in rows
        ]

    def upsert_context_source(
        self,
        *,
        save_id: str,
        source_type: str,
        source_id: str,
        title: str,
        body: str,
        metadata: dict[str, object] | None = None,
        token_estimate: int | None = None,
        context_source_id: str | None = None,
        scene_snapshot_id: str | None = None,
        scene_generation: int | None = None,
        created_turn_number: int | None = None,
        expires_after_turn_number: int | None = None,
    ) -> ContextSourceRecord:
        resolved_metadata = dict(metadata or {})
        _validate_context_source_provenance_metadata(resolved_metadata)
        existing = self._fetch_one(
            """
            SELECT id, save_id, source_type, source_id, title, body, metadata_json,
                   token_estimate, scene_snapshot_id, scene_generation,
                   created_turn_number, expires_after_turn_number, archived_at
            FROM context_sources
            WHERE save_id = ? AND source_type = ? AND source_id = ?
            """,
            (save_id, source_type, source_id),
        )
        record = ContextSourceRecord(
            id=existing["id"] if existing else context_source_id or _new_id(),
            save_id=save_id,
            source_type=source_type,
            source_id=source_id,
            title=title,
            body=body,
            metadata=resolved_metadata,
            token_estimate=token_estimate,
            scene_snapshot_id=scene_snapshot_id,
            scene_generation=scene_generation,
            created_turn_number=created_turn_number,
            expires_after_turn_number=expires_after_turn_number,
        )
        if (
            existing is not None
            and existing["archived_at"] is None
            and existing["title"] == record.title
            and existing["body"] == record.body
            and existing["metadata_json"] == _dump_json(record.metadata)
            and existing["token_estimate"] == record.token_estimate
            and existing["scene_snapshot_id"] == record.scene_snapshot_id
            and existing["scene_generation"] == record.scene_generation
            and existing["created_turn_number"] == record.created_turn_number
            and existing["expires_after_turn_number"]
            == record.expires_after_turn_number
        ):
            return _context_source_from_row(existing)
        self.connection.execute(
            """
            INSERT INTO context_sources(
                id, save_id, source_type, source_id, title, body, metadata_json,
                token_estimate, scene_snapshot_id, scene_generation,
                created_turn_number, expires_after_turn_number
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(save_id, source_type, source_id) DO UPDATE SET
                title = excluded.title,
                body = excluded.body,
                metadata_json = excluded.metadata_json,
                token_estimate = excluded.token_estimate,
                scene_snapshot_id = excluded.scene_snapshot_id,
                scene_generation = excluded.scene_generation,
                created_turn_number = excluded.created_turn_number,
                expires_after_turn_number = excluded.expires_after_turn_number,
                updated_at = CURRENT_TIMESTAMP,
                archived_at = NULL
            """,
            (
                record.id,
                record.save_id,
                record.source_type,
                record.source_id,
                record.title,
                record.body,
                _dump_json(record.metadata),
                record.token_estimate,
                record.scene_snapshot_id,
                record.scene_generation,
                record.created_turn_number,
                record.expires_after_turn_number,
            ),
        )
        self._replace_context_source_search_terms(record)
        self.commit()
        return self.get_context_source(record.id) or record

    def _replace_context_source_search_terms(
        self,
        record: ContextSourceRecord,
    ) -> None:
        self.connection.execute(
            "DELETE FROM context_source_search_terms WHERE context_source_id = ?",
            (record.id,),
        )
        self.connection.executemany(
            """
            INSERT INTO context_source_search_terms(
                context_source_id,
                save_id,
                term
            )
            VALUES (?, ?, ?)
            """,
            (
                (record.id, record.save_id, term)
                for term in _context_source_search_terms(record.title, record.body)
            ),
        )

    def rebuild_context_source_search_terms(self, save_id: str) -> None:
        rows = self.list_context_sources(save_id)
        for record in rows:
            self._replace_context_source_search_terms(record)
        self.commit()

    def get_context_source(self, context_source_id: str) -> ContextSourceRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, source_type, source_id, title, body, metadata_json,
                   token_estimate, scene_snapshot_id, scene_generation,
                   created_turn_number, expires_after_turn_number
            FROM context_sources
            WHERE id = ? AND archived_at IS NULL
            """,
            (context_source_id,),
        )
        return _context_source_from_row(row) if row else None

    def archive_context_source_by_key(
        self,
        save_id: str,
        *,
        source_type: str,
        source_id: str,
    ) -> None:
        self.connection.execute(
            """
            UPDATE context_sources
            SET archived_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ? AND source_type = ? AND source_id = ?
            """,
            (save_id, source_type, source_id),
        )

    def archive_continuity_character_sources_except(
        self,
        save_id: str,
        *,
        character_id: str,
        active_keys: set[tuple[str, str]] | frozenset[tuple[str, str]],
    ) -> None:
        rows = self._fetch_all(
            """
            SELECT id, source_type, source_id
            FROM context_sources
            WHERE save_id = ?
              AND archived_at IS NULL
              AND json_extract(metadata_json, '$.indexed_by') =
                  'continuity_index'
              AND (
                    (source_type = 'character_voice' AND source_id = ?)
                    OR (
                        source_type = 'memory'
                        AND (
                            source_id = 'character_profile:' || ?
                            OR source_id LIKE 'relationship:' || ? || ':%'
                        )
                    )
              )
            """,
            (save_id, character_id, character_id, character_id),
        )
        for row in rows:
            key = (str(row["source_type"]), str(row["source_id"]))
            if key not in active_keys:
                self.archive_context_source(str(row["id"]))

    def archive_continuity_sources_by_type_except(
        self,
        save_id: str,
        *,
        source_type: str,
        active_source_ids: set[str] | frozenset[str],
    ) -> None:
        rows = self._fetch_all(
            """
            SELECT id, source_id
            FROM context_sources
            WHERE save_id = ?
              AND source_type = ?
              AND archived_at IS NULL
              AND json_extract(metadata_json, '$.indexed_by') =
                  'continuity_index'
            """,
            (save_id, source_type),
        )
        for row in rows:
            if str(row["source_id"]) not in active_source_ids:
                self.archive_context_source(str(row["id"]))

    def list_context_sources(
        self,
        save_id: str,
        *,
        source_type: str | None = None,
    ) -> list[ContextSourceRecord]:
        type_filter = "" if source_type is None else "AND source_type = ?"
        params: tuple[Any, ...] = (
            (save_id,) if source_type is None else (save_id, source_type)
        )
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, source_type, source_id, title, body, metadata_json,
                   token_estimate, scene_snapshot_id, scene_generation,
                   created_turn_number, expires_after_turn_number
            FROM context_sources
            WHERE save_id = ? AND archived_at IS NULL {type_filter}
            ORDER BY source_type, title, created_at, rowid
            """,
            params,
        )
        return [_context_source_from_row(row) for row in rows]

    def list_context_sources_by_keys(
        self,
        save_id: str,
        source_keys: set[tuple[str, str]] | frozenset[tuple[str, str]],
    ) -> list[ContextSourceRecord]:
        if not source_keys:
            return []
        rows = self._fetch_all(
            """
            SELECT source.id, source.save_id, source.source_type,
                   source.source_id, source.title, source.body,
                   source.metadata_json, source.token_estimate,
                   source.scene_snapshot_id, source.scene_generation,
                   source.created_turn_number, source.expires_after_turn_number
            FROM context_sources AS source
            JOIN json_each(?) AS selected
              ON source.source_type =
                 CAST(json_extract(selected.value, '$[0]') AS TEXT)
             AND source.source_id =
                 CAST(json_extract(selected.value, '$[1]') AS TEXT)
            WHERE source.save_id = ? AND source.archived_at IS NULL
            ORDER BY source.source_type, source.title,
                     source.created_at, source.rowid
            """,
            (_dump_json(sorted(source_keys)), save_id),
        )
        return [_context_source_from_row(row) for row in rows]

    def list_curated_observation_source_markers(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[ContextSourceRecord]:
        order_sql = (
            "ORDER BY created_at DESC, rowid DESC"
            if limit is not None
            else "ORDER BY created_at, rowid"
        )
        limit_sql = "LIMIT ?" if limit is not None else ""
        params: tuple[object, ...] = (
            (save_id, max(0, limit))
            if limit is not None
            else (save_id,)
        )
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, source_type, source_id, title, body, metadata_json,
                   token_estimate, scene_snapshot_id, scene_generation,
                   created_turn_number, expires_after_turn_number
            FROM context_sources
            WHERE save_id = ?
              AND source_type = 'observation'
              AND json_extract(metadata_json, '$.curation_action')
                  IN ('save_context', 'scene_scratch')
              AND (
                    archived_at IS NULL
                    OR json_extract(metadata_json, '$.curation_action')
                       = 'scene_scratch'
                  )
            {order_sql}
            {limit_sql}
            """,
            params,
        )
        if limit is not None:
            rows.reverse()
        return [_context_source_from_row(row) for row in rows]

    def search_context_sources(
        self,
        save_id: str,
        *,
        query_terms: set[str] | frozenset[str] | list[str] | tuple[str, ...],
        source_types: set[str] | frozenset[str] | list[str] | tuple[str, ...],
        limit: int,
        allowed_owner_names: set[str] | frozenset[str] | None = None,
        reference_character_ids: set[str] | frozenset[str] | None = None,
        visibility_character_ids: set[str] | frozenset[str] | None = None,
        current_scene_snapshot_id: str | None = None,
        current_scene_generation: int | None = None,
        current_turn_number: int | None = None,
        blocked_source_keys: set[tuple[str, str]] | frozenset[tuple[str, str]]
        | None = None,
        match_all: bool = False,
        exact_phrases: tuple[str, ...] = (),
        exact_identifiers: tuple[str, ...] = (),
    ) -> list[ContextSourceSearchHit]:
        if limit <= 0:
            return []
        bounded_query_terms = _bounded_repository_search_terms(
            {
                str(term)
                for term in query_terms
                if str(term).strip()
            },
            limit=MAX_CONTEXT_SEARCH_TERMS,
        )
        match_query = _fts_query_from_terms(
            bounded_query_terms,
            match_all=match_all,
        )
        phrase_match_query = _fts_query_from_exact_phrases(exact_phrases)
        exact_identifier_specs = _context_source_exact_identifier_specs(
            exact_identifiers
        )
        if (
            not match_query
            and not phrase_match_query
            and not exact_identifier_specs
        ):
            return []
        source_type_values = tuple(dict.fromkeys(str(item) for item in source_types))
        if not source_type_values:
            return []
        eligibility_sql, eligibility_params = _context_source_eligibility_sql(
            alias="context_sources",
            allowed_owner_names=allowed_owner_names,
            reference_character_ids=reference_character_ids,
            visibility_character_ids=visibility_character_ids,
            current_scene_snapshot_id=current_scene_snapshot_id,
            current_scene_generation=current_scene_generation,
            current_turn_number=current_turn_number,
            blocked_source_keys=blocked_source_keys,
        )
        normalized_query_terms = tuple(
            dict.fromkeys(
                term
                for value in bounded_query_terms
                for term in (
                    *_unicode_search_terms(str(value)),
                    *cjk_lexical_anchors(str(value)),
                )
            )
        )
        phrase_terms = tuple(
            term
            for phrase in exact_phrases[:MAX_CONTEXT_EXACT_PHRASES]
            for term in (
                *unicode_word_terms(phrase),
                *cjk_lexical_anchors(phrase),
            )
        )
        indexed_terms = tuple(
            dict.fromkeys((*normalized_query_terms, *phrase_terms))
        )[:MAX_UNICODE_SUBSTRING_TERMS]
        term_rows: list[sqlite3.Row] = []
        if indexed_terms:
            indexed_terms_json = _dump_json(indexed_terms)
            term_rows = self._fetch_all(
                f"""
                SELECT
                    context_sources.id,
                    context_sources.save_id,
                    context_sources.source_type,
                    context_sources.source_id,
                    context_sources.title,
                    context_sources.body,
                    context_sources.metadata_json,
                    context_sources.token_estimate,
                    context_sources.scene_snapshot_id,
                    context_sources.scene_generation,
                    context_sources.created_turn_number,
                    context_sources.expires_after_turn_number,
                    -COUNT(DISTINCT search_terms.term) AS bm25_rank
                FROM context_source_search_terms search_terms
                JOIN context_sources
                  ON context_sources.id = search_terms.context_source_id
                WHERE search_terms.save_id = ?
                  AND search_terms.term IN (
                      SELECT CAST(value AS TEXT) FROM json_each(?)
                  )
                  AND context_sources.archived_at IS NULL
                  AND context_sources.source_type IN (
                      {_placeholders(len(source_type_values))}
                  )
                  {eligibility_sql}
                GROUP BY context_sources.id
                HAVING ? = 0
                    OR COUNT(DISTINCT search_terms.term) = ?
                ORDER BY bm25_rank,
                         context_sources.created_at DESC,
                         context_sources.rowid DESC
                LIMIT ?
                """,
                (
                    save_id,
                    indexed_terms_json,
                    *source_type_values,
                    *eligibility_params,
                    int(match_all),
                    len(indexed_terms),
                    limit,
                ),
            )
        exact_identifier_rows = (
            self._fetch_all(
                f"""
                WITH identifier_specs(spec) AS MATERIALIZED (
                    SELECT value
                    FROM json_each(?)
                ),
                eligible_identifier_candidates AS MATERIALIZED (
                    SELECT
                        context_sources.id,
                        context_sources.save_id,
                        context_sources.source_type,
                        context_sources.source_id,
                        context_sources.title,
                        context_sources.body,
                        context_sources.metadata_json,
                        context_sources.token_estimate,
                        context_sources.scene_snapshot_id,
                        context_sources.scene_generation,
                        context_sources.created_turn_number,
                        context_sources.expires_after_turn_number,
                        context_sources.created_at,
                        context_sources.rowid AS source_rowid,
                        CAST(
                            json_extract(identifier_specs.spec, '$.identifier')
                            AS TEXT
                        ) AS identifier
                    FROM identifier_specs
                    JOIN context_source_search_terms anchor_terms
                      ON anchor_terms.save_id = ?
                     AND anchor_terms.term = CAST(
                            json_extract(identifier_specs.spec, '$.anchor')
                            AS TEXT
                         )
                    JOIN context_sources
                      ON context_sources.id = anchor_terms.context_source_id
                    WHERE context_sources.archived_at IS NULL
                      AND context_sources.source_type IN (
                          {_placeholders(len(source_type_values))}
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM json_each(
                              json_extract(identifier_specs.spec, '$.terms')
                          ) required_term
                          WHERE NOT EXISTS (
                              SELECT 1
                              FROM context_source_search_terms indexed_term
                              WHERE indexed_term.context_source_id =
                                    context_sources.id
                                AND indexed_term.term =
                                    CAST(required_term.value AS TEXT)
                          )
                      )
                      {eligibility_sql}
                )
                SELECT
                    id,
                    save_id,
                    source_type,
                    source_id,
                    title,
                    body,
                    metadata_json,
                    token_estimate,
                    scene_snapshot_id,
                    scene_generation,
                    created_turn_number,
                    expires_after_turn_number,
                    -1000.0 AS bm25_rank
                FROM eligible_identifier_candidates
                WHERE bragi_contains_exact_identifier(title, identifier) = 1
                   OR bragi_contains_exact_identifier(body, identifier) = 1
                GROUP BY id
                ORDER BY MAX(created_at) DESC, MAX(source_rowid) DESC
                LIMIT ?
                """,
                (
                    _dump_json(exact_identifier_specs),
                    save_id,
                    *source_type_values,
                    *eligibility_params,
                    limit,
                ),
            )
            if exact_identifier_specs
            else []
        )
        exact_phrase_rows = (
            self._fetch_all(
                f"""
            SELECT
                context_sources.id,
                context_sources.save_id,
                context_sources.source_type,
                context_sources.source_id,
                context_sources.title,
                context_sources.body,
                context_sources.metadata_json,
                context_sources.token_estimate,
                context_sources.scene_snapshot_id,
                context_sources.scene_generation,
                context_sources.created_turn_number,
                context_sources.expires_after_turn_number,
                bm25(context_source_fts, 1.2, 1.0) AS bm25_rank
            FROM context_source_fts
            JOIN context_sources
              ON context_sources.rowid = context_source_fts.rowid
            WHERE context_source_fts MATCH ?
              AND context_sources.save_id = ?
              AND context_sources.archived_at IS NULL
              AND context_sources.source_type IN (
                  {_placeholders(len(source_type_values))}
              )
              {eligibility_sql}
            ORDER BY bm25_rank, context_sources.created_at DESC,
                     context_sources.rowid DESC
            LIMIT ?
                """,
                (
                    phrase_match_query,
                    save_id,
                    *source_type_values,
                    *eligibility_params,
                    limit,
                ),
            )
            if phrase_match_query
            else []
        )
        fts_rows = (
            self._fetch_all(
                f"""
            SELECT
                context_sources.id,
                context_sources.save_id,
                context_sources.source_type,
                context_sources.source_id,
                context_sources.title,
                context_sources.body,
                context_sources.metadata_json,
                context_sources.token_estimate,
                context_sources.scene_snapshot_id,
                context_sources.scene_generation,
                context_sources.created_turn_number,
                context_sources.expires_after_turn_number,
                bm25(context_source_fts, 1.2, 1.0) AS bm25_rank
            FROM context_source_fts
            JOIN context_sources
              ON context_sources.rowid = context_source_fts.rowid
            WHERE context_source_fts MATCH ?
              AND context_sources.save_id = ?
              AND context_sources.archived_at IS NULL
              AND context_sources.source_type IN (
                  {_placeholders(len(source_type_values))}
              )
              {eligibility_sql}
            ORDER BY bm25_rank, context_sources.created_at DESC,
                     context_sources.rowid DESC
            LIMIT ?
                """,
                (
                    match_query,
                    save_id,
                    *source_type_values,
                    *eligibility_params,
                    limit,
                ),
            )
            if match_query
            else []
        )
        rows_by_id: dict[str, sqlite3.Row] = {}
        for row in exact_identifier_rows:
            rows_by_id.setdefault(str(row["id"]), row)
        for row in exact_phrase_rows:
            rows_by_id.setdefault(str(row["id"]), row)
        for index in range(max(len(term_rows), len(fts_rows))):
            for ranked_rows in (term_rows, fts_rows):
                if index >= len(ranked_rows):
                    continue
                row = ranked_rows[index]
                rows_by_id.setdefault(str(row["id"]), row)
        rows = list(rows_by_id.values())[:limit]
        return [
            ContextSourceSearchHit(
                record=_context_source_from_row(row),
                bm25_rank=float(row["bm25_rank"]),
            )
            for row in rows
        ]

    def list_protected_context_sources(
        self,
        save_id: str,
        *,
        limit: int,
        allowed_owner_names: set[str] | frozenset[str] | None = None,
        reference_character_ids: set[str] | frozenset[str] | None = None,
        visibility_character_ids: set[str] | frozenset[str] | None = None,
        current_scene_snapshot_id: str | None = None,
        current_scene_generation: int | None = None,
        current_turn_number: int | None = None,
        blocked_source_keys: set[tuple[str, str]] | frozenset[tuple[str, str]]
        | None = None,
    ) -> list[ContextSourceRecord]:
        if limit <= 0:
            return []
        eligibility_sql, eligibility_params = _context_source_eligibility_sql(
            alias="context_sources",
            allowed_owner_names=allowed_owner_names,
            reference_character_ids=reference_character_ids,
            visibility_character_ids=visibility_character_ids,
            current_scene_snapshot_id=current_scene_snapshot_id,
            current_scene_generation=current_scene_generation,
            current_turn_number=current_turn_number,
            blocked_source_keys=blocked_source_keys,
        )
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, source_type, source_id, title, body,
                   metadata_json, token_estimate, scene_snapshot_id,
                   scene_generation, created_turn_number, expires_after_turn_number
            FROM context_sources
            WHERE save_id = ?
              AND archived_at IS NULL
              AND (
                    source_type IN ('open_obligation', 'character_voice')
                    OR json_extract(metadata_json, '$.always_include_reason')
                       IS NOT NULL
                    OR (
                        source_type = 'observation'
                        AND json_extract(metadata_json, '$.curation_action')
                            IN ('save_context', 'scene_scratch')
                    )
                  )
              {eligibility_sql}
            ORDER BY
                CASE
                    WHEN source_type = 'open_obligation' THEN 0
                    WHEN source_type = 'character_voice' THEN 1
                    WHEN source_type = 'observation' THEN 3
                    ELSE 2
                END,
                CAST(COALESCE(json_extract(metadata_json, '$.importance'), 0.0)
                     AS REAL) DESC,
                created_at DESC,
                rowid DESC
            LIMIT ?
            """,
            (save_id, *eligibility_params, limit),
        )
        return [_context_source_from_row(row) for row in rows]

    def archive_context_source(self, context_source_id: str) -> None:
        self.connection.execute(
            """
            UPDATE context_sources
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (context_source_id,),
        )
        self.commit()

    def archive_stale_scene_scratch(
        self,
        *,
        save_id: str,
        current_scene_snapshot_id: str | None,
        current_scene_generation: int | None,
        current_turn_number: int,
    ) -> frozenset[str]:
        if current_scene_snapshot_id is None or current_scene_generation is None:
            stale_clause = "1"
            params: tuple[object, ...] = (save_id,)
        else:
            stale_clause = """
                (
                    scene_snapshot_id IS NULL
                    OR scene_snapshot_id != ?
                    OR scene_generation IS NULL
                    OR scene_generation != ?
                    OR (
                        expires_after_turn_number IS NOT NULL
                        AND expires_after_turn_number <= ?
                    )
                )
            """
            params = (
                save_id,
                current_scene_snapshot_id,
                current_scene_generation,
                current_turn_number,
            )
        rows = self._fetch_all(
            f"""
            SELECT id, source_id
            FROM context_sources
            WHERE save_id = ?
              AND archived_at IS NULL
              AND source_type = 'observation'
              AND json_extract(metadata_json, '$.curation_action')
                  = 'scene_scratch'
              AND {stale_clause}
            """,
            params,
        )
        stale_ids = frozenset(str(row["id"]) for row in rows)
        stale_observation_ids = frozenset(str(row["source_id"]) for row in rows)
        if not stale_ids:
            return stale_ids
        self.connection.execute(
            f"""
            UPDATE context_sources
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND id IN ({_placeholders(len(stale_ids))})
            """,
            (save_id, *stale_ids),
        )
        self.connection.execute(
            f"""
            UPDATE context_observations
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND id IN ({_placeholders(len(stale_observation_ids))})
            """,
            (save_id, *stale_observation_ids),
        )
        self.commit()
        return stale_ids

    def restore_context_sources(
        self,
        context_source_ids: set[str] | frozenset[str],
    ) -> None:
        if not context_source_ids:
            return
        self.connection.execute(
            f"""
            UPDATE context_sources
            SET archived_at = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(context_source_ids))})
            """,
            tuple(context_source_ids),
        )
        self.commit()

    def archive_context_sources_for_deleted_messages(
        self,
        *,
        save_id: str,
        message_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        if not message_ids:
            return frozenset()
        archived_ids: set[str] = set()
        for record in self.list_context_sources(save_id):
            if not _context_source_references_any_message(record, message_ids):
                continue
            self.archive_context_source(record.id)
            archived_ids.add(record.id)
        return frozenset(archived_ids)

    def add_context_observation(
        self,
        *,
        save_id: str,
        observation_type: str,
        claim: str,
        evidence_quote: str = "",
        source_message_ids: list[str] | tuple[str, ...] | None = None,
        scope: str = "turn",
        status: str = "pending",
        confidence: float = 0.0,
        tags: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, object] | None = None,
        observation_id: str | None = None,
    ) -> ContextObservationRecord:
        original_observation_type = observation_type.strip()
        normalized_observation_type = normalize_observation_type(
            original_observation_type
        )
        normalized_metadata = dict(metadata or {})
        if normalized_observation_type != original_observation_type:
            normalized_metadata.setdefault(
                "original_observation_type",
                original_observation_type,
            )
        record = ContextObservationRecord(
            id=observation_id or _new_id(),
            save_id=save_id,
            observation_type=normalized_observation_type,
            claim=claim.strip(),
            evidence_quote=evidence_quote.strip(),
            source_message_ids=_unique_strings(source_message_ids or ()),
            scope=scope.strip() or "turn",
            status=status.strip() or "pending",
            confidence=confidence,
            tags=_unique_strings(tags or ()),
            metadata=normalized_metadata,
        )
        self.connection.execute(
            """
            INSERT INTO context_observations(
                id, save_id, observation_type, claim, evidence_quote,
                source_message_ids_json, scope, status, confidence, tags_json,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.observation_type,
                record.claim,
                record.evidence_quote,
                _dump_json(record.source_message_ids),
                record.scope,
                record.status,
                record.confidence,
                _dump_json(record.tags),
                _dump_json(record.metadata),
            ),
        )
        self.connection.execute(
            """
            INSERT INTO context_observation_curation_state(
                observation_id, save_id, terminal_outcome, completed_at
            )
            VALUES (
                ?, ?,
                CASE WHEN ? = 'pending' THEN NULL ELSE ? END,
                CASE WHEN ? = 'pending' THEN NULL ELSE CURRENT_TIMESTAMP END
            )
            """,
            (
                record.id,
                record.save_id,
                record.status,
                record.status,
                record.status,
            ),
        )
        self.commit()
        return self.get_context_observation(record.id) or record

    def get_context_observation(
        self,
        observation_id: str,
    ) -> ContextObservationRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, observation_type, claim, evidence_quote,
                   source_message_ids_json, scope, status, confidence, tags_json,
                   metadata_json, created_at, updated_at, archived_at
            FROM context_observations
            WHERE id = ? AND archived_at IS NULL
            """,
            (observation_id,),
        )
        return _context_observation_from_row(row) if row else None

    def list_context_observations(
        self,
        save_id: str,
        *,
        statuses: set[str] | frozenset[str] | tuple[str, ...] | None = None,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> list[ContextObservationRecord]:
        filters = ["save_id = ?"]
        params: list[object] = [save_id]
        if statuses is not None:
            values = tuple(statuses)
            if not values:
                return []
            filters.append(f"status IN ({_placeholders(len(values))})")
            params.extend(values)
        if not include_archived:
            filters.append("archived_at IS NULL")
        limit_sql = "LIMIT ?" if limit is not None else ""
        order_sql = (
            "ORDER BY created_at DESC, rowid DESC"
            if limit is not None
            else "ORDER BY created_at, rowid"
        )
        if limit is not None:
            params.append(max(0, limit))
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, observation_type, claim, evidence_quote,
                   source_message_ids_json, scope, status, confidence, tags_json,
                   metadata_json, created_at, updated_at, archived_at
            FROM context_observations
            WHERE {' AND '.join(filters)}
            {order_sql}
            {limit_sql}
            """,
            tuple(params),
        )
        if limit is not None:
            rows.reverse()
        return [_context_observation_from_row(row) for row in rows]

    def get_context_observation_curation_state(
        self,
        observation_id: str,
    ) -> ContextObservationCurationStateRecord | None:
        row = self._fetch_one(
            f"""
            SELECT {_CONTEXT_OBSERVATION_CURATION_STATE_COLUMNS}
            FROM context_observation_curation_state
            WHERE observation_id = ?
            """,
            (observation_id,),
        )
        return _context_observation_curation_state_from_row(row) if row else None

    def ensure_context_observation_curation_states(self, save_id: str) -> int:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO context_observation_curation_state(
                observation_id, save_id, terminal_outcome, completed_at
            )
            SELECT id, save_id,
                   CASE WHEN status = 'pending' THEN NULL ELSE status END,
                   CASE WHEN status = 'pending' THEN NULL ELSE CURRENT_TIMESTAMP END
            FROM context_observations
            WHERE save_id = ?
            """,
            (save_id,),
        )
        self.commit()
        return cursor.rowcount

    def list_eligible_context_observations(
        self,
        save_id: str,
        *,
        limit: int,
    ) -> list[ContextObservationRecord]:
        if limit <= 0:
            return []
        rows = self._fetch_all(
            """
            SELECT observation.id, observation.save_id,
                   observation.observation_type, observation.claim,
                   observation.evidence_quote,
                   observation.source_message_ids_json, observation.scope,
                   observation.status, observation.confidence,
                   observation.tags_json, observation.metadata_json,
                   observation.created_at, observation.updated_at,
                   observation.archived_at
            FROM context_observations observation
            JOIN context_observation_curation_state curation
              ON curation.observation_id = observation.id
            WHERE observation.save_id = ?
              AND observation.status = 'pending'
              AND observation.archived_at IS NULL
              AND curation.terminal_outcome IS NULL
              AND (
                  curation.next_eligible_at IS NULL
                  OR curation.next_eligible_at <= CURRENT_TIMESTAMP
              )
              AND (
                  curation.lease_until IS NULL
                  OR curation.lease_until <= CURRENT_TIMESTAMP
              )
            ORDER BY observation.created_at, observation.rowid
            LIMIT ?
            """,
            (save_id, limit),
        )
        return [_context_observation_from_row(row) for row in rows]

    def list_save_ids_with_due_context_observation_curation(
        self,
        *,
        limit: int,
        offset: int = 0,
    ) -> list[str]:
        if limit <= 0:
            return []
        rows = self._fetch_all(
            """
            SELECT observation.save_id
            FROM context_observations observation
            JOIN context_observation_curation_state curation
              ON curation.observation_id = observation.id
            LEFT JOIN scheduled_tasks scheduled
              ON scheduled.task_type = 'observation_curation_drain'
             AND scheduled.save_id = observation.save_id
            WHERE observation.status = 'pending'
              AND observation.archived_at IS NULL
              AND curation.terminal_outcome IS NULL
              AND (
                  curation.next_eligible_at IS NULL
                  OR curation.next_eligible_at <= CURRENT_TIMESTAMP
              )
              AND (
                  curation.lease_until IS NULL
                  OR curation.lease_until <= CURRENT_TIMESTAMP
              )
              AND (
                  scheduled.id IS NULL
                  OR (
                      scheduled.enabled = 1
                      AND scheduled.next_run_at <= CURRENT_TIMESTAMP
                      AND (
                          scheduled.lease_until IS NULL
                          OR scheduled.lease_until <= CURRENT_TIMESTAMP
                      )
                  )
              )
            GROUP BY observation.save_id
            ORDER BY
                CASE WHEN MAX(scheduled.id) IS NULL THEN 0 ELSE 1 END,
                MAX(scheduled.updated_at),
                MIN(observation.created_at),
                MIN(observation.rowid)
            LIMIT ? OFFSET ?
            """,
            (limit, max(0, offset)),
        )
        return [str(row["save_id"]) for row in rows]

    def context_observation_curation_health(
        self,
        save_id: str,
    ) -> ContextObservationCurationHealthRecord:
        row = self._fetch_one(
            """
            SELECT
                SUM(CASE WHEN observation.status = 'pending'
                    AND observation.archived_at IS NULL
                    AND curation.terminal_outcome IS NULL THEN 1 ELSE 0 END)
                    AS pending_count,
                SUM(CASE WHEN observation.status = 'pending'
                    AND observation.archived_at IS NULL
                    AND curation.terminal_outcome IS NULL
                    AND (curation.next_eligible_at IS NULL
                         OR curation.next_eligible_at <= CURRENT_TIMESTAMP)
                    AND (curation.lease_until IS NULL
                         OR curation.lease_until <= CURRENT_TIMESTAMP)
                    THEN 1 ELSE 0 END) AS eligible_count,
                SUM(CASE WHEN observation.status = 'pending'
                    AND observation.archived_at IS NULL
                    AND curation.terminal_outcome IS NULL
                    AND curation.lease_until > CURRENT_TIMESTAMP
                    THEN 1 ELSE 0 END) AS leased_count,
                MIN(CASE WHEN observation.status = 'pending'
                    AND observation.archived_at IS NULL
                    AND curation.terminal_outcome IS NULL
                    THEN observation.created_at END) AS oldest_pending_at,
                SUM(CASE WHEN observation.status = 'pending'
                    AND observation.archived_at IS NULL
                    AND curation.terminal_outcome IS NULL
                    THEN curation.attempt_count ELSE 0 END) AS total_attempt_count,
                MAX(CASE WHEN observation.status = 'pending'
                    AND observation.archived_at IS NULL
                    AND curation.terminal_outcome IS NULL
                    THEN curation.attempt_count ELSE 0 END) AS max_attempt_count,
                SUM(CASE WHEN observation.status = 'curation_failed'
                    AND observation.archived_at IS NULL
                    THEN 1 ELSE 0 END) AS terminal_failure_count
            FROM context_observation_curation_state curation
            JOIN context_observations observation
              ON observation.id = curation.observation_id
            WHERE curation.save_id = ?
            """,
            (save_id,),
        )
        if row is None:
            raise RuntimeError("Curation health aggregate returned no row")
        values = row
        return ContextObservationCurationHealthRecord(
            pending_count=int(values["pending_count"] or 0),
            eligible_count=int(values["eligible_count"] or 0),
            leased_count=int(values["leased_count"] or 0),
            oldest_pending_at=(
                str(values["oldest_pending_at"])
                if values["oldest_pending_at"] is not None
                else None
            ),
            total_attempt_count=int(values["total_attempt_count"] or 0),
            max_attempt_count=int(values["max_attempt_count"] or 0),
            terminal_failure_count=int(values["terminal_failure_count"] or 0),
        )

    def claim_context_observations(
        self,
        observation_ids: Iterable[str],
        *,
        lease_token: str,
        lease_seconds: int,
        max_attempts: int = 5,
    ) -> list[ContextObservationRecord]:
        ids = tuple(dict.fromkeys(item for item in observation_ids if item))
        if not ids:
            return []
        if not lease_token:
            raise ValueError("lease_token is required")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be at least 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.begin_immediate_transaction()
        try:
            exhausted_rows = self._fetch_all(
                f"""
                SELECT curation.observation_id
                FROM context_observation_curation_state curation
                JOIN context_observations observation
                  ON observation.id = curation.observation_id
                WHERE curation.observation_id IN ({_placeholders(len(ids))})
                  AND curation.terminal_outcome IS NULL
                  AND curation.attempt_count >= ?
                  AND (
                      curation.next_eligible_at IS NULL
                      OR curation.next_eligible_at <= CURRENT_TIMESTAMP
                  )
                  AND (
                      curation.lease_until IS NULL
                      OR curation.lease_until <= CURRENT_TIMESTAMP
                  )
                  AND observation.status = 'pending'
                  AND observation.archived_at IS NULL
                """,
                (*ids, max_attempts),
            )
            exhausted_ids = tuple(
                str(row["observation_id"]) for row in exhausted_rows
            )
            if exhausted_ids:
                self.connection.execute(
                    f"""
                    UPDATE context_observation_curation_state
                    SET lease_token = NULL,
                        lease_until = NULL,
                        next_eligible_at = NULL,
                        last_error = 'retry_budget_exhausted',
                        terminal_outcome = 'retry_budget_exhausted',
                        completed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE observation_id IN ({_placeholders(len(exhausted_ids))})
                    """,
                    exhausted_ids,
                )
                self.connection.execute(
                    f"""
                    UPDATE context_observations
                    SET status = 'curation_failed',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id IN ({_placeholders(len(exhausted_ids))})
                    """,
                    exhausted_ids,
                )
            self.connection.execute(
                f"""
                UPDATE context_observation_curation_state
                SET attempt_count = attempt_count + 1,
                    next_eligible_at = NULL,
                    lease_token = ?,
                    lease_until = datetime('now', ?),
                    last_error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE observation_id IN ({_placeholders(len(ids))})
                  AND terminal_outcome IS NULL
                  AND attempt_count < ?
                  AND (
                      next_eligible_at IS NULL
                      OR next_eligible_at <= CURRENT_TIMESTAMP
                  )
                  AND (lease_until IS NULL OR lease_until <= CURRENT_TIMESTAMP)
                  AND EXISTS (
                      SELECT 1 FROM context_observations observation
                      WHERE observation.id = observation_id
                        AND observation.status = 'pending'
                        AND observation.archived_at IS NULL
                  )
                """,
                (lease_token, f"+{lease_seconds} seconds", *ids, max_attempts),
            )
            rows = self._fetch_all(
                f"""
                SELECT id, save_id, observation_type, claim, evidence_quote,
                       source_message_ids_json, scope, status, confidence,
                       tags_json, metadata_json, created_at, updated_at, archived_at
                FROM context_observations
                WHERE id IN (
                    SELECT observation_id
                    FROM context_observation_curation_state
                    WHERE lease_token = ?
                      AND observation_id IN ({_placeholders(len(ids))})
                )
                """,
                (lease_token, *ids),
            )
            records_by_id = {
                record.id: record
                for record in (_context_observation_from_row(row) for row in rows)
            }
            claimed = [records_by_id[item] for item in ids if item in records_by_id]
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise
        return claimed

    def release_context_observation_curation_claims(
        self,
        observation_ids: Iterable[str],
        *,
        lease_token: str,
        error: str,
    ) -> int:
        ids = tuple(dict.fromkeys(item for item in observation_ids if item))
        if not ids:
            return 0
        if not lease_token:
            raise ValueError("lease_token is required")
        safe_error = (redact_text(error) or "")[:240]
        self.begin_immediate_transaction()
        try:
            cursor = self.connection.execute(
                f"""
                UPDATE context_observation_curation_state
                SET lease_token = NULL,
                    lease_until = NULL,
                    next_eligible_at = CURRENT_TIMESTAMP,
                    last_error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE observation_id IN ({_placeholders(len(ids))})
                  AND lease_token = ?
                  AND terminal_outcome IS NULL
                """,
                (safe_error, *ids, lease_token),
            )
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise
        return cursor.rowcount

    def renew_context_observation_curation_claims(
        self,
        observation_ids: Iterable[str],
        *,
        lease_token: str,
        lease_seconds: int,
    ) -> int:
        ids = tuple(dict.fromkeys(item for item in observation_ids if item))
        if not ids:
            return 0
        if not lease_token:
            raise ValueError("lease_token is required")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be at least 1")
        self.begin_immediate_transaction()
        try:
            cursor = self.connection.execute(
                f"""
                UPDATE context_observation_curation_state
                SET lease_until = datetime('now', ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE observation_id IN ({_placeholders(len(ids))})
                  AND lease_token = ?
                  AND terminal_outcome IS NULL
                  AND lease_until > CURRENT_TIMESTAMP
                """,
                (f"+{lease_seconds + 1} seconds", *ids, lease_token),
            )
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise
        return cursor.rowcount

    def owns_context_observation_curation_lease(
        self,
        observation_id: str,
        *,
        lease_token: str,
    ) -> bool:
        return (
            self._fetch_one(
                """
                SELECT 1
                FROM context_observation_curation_state
                WHERE observation_id = ?
                  AND lease_token = ?
                  AND terminal_outcome IS NULL
                  AND lease_until > CURRENT_TIMESTAMP
                """,
                (observation_id, lease_token),
            )
            is not None
        )

    def complete_context_observation_curation(
        self,
        observation_id: str,
        *,
        lease_token: str,
        status: str,
        terminal_outcome: str,
        metadata: dict[str, object] | None = None,
    ) -> ContextObservationRecord | None:
        self.begin_immediate_transaction()
        try:
            cursor = self.connection.execute(
                """
                UPDATE context_observation_curation_state
                SET lease_token = NULL,
                    lease_until = NULL,
                    next_eligible_at = NULL,
                    last_error = NULL,
                    terminal_outcome = ?,
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE observation_id = ?
                  AND lease_token = ?
                  AND terminal_outcome IS NULL
                  AND lease_until > CURRENT_TIMESTAMP
                """,
                (terminal_outcome, observation_id, lease_token),
            )
            if cursor.rowcount == 0:
                self.rollback_transaction()
                return None
            updated = self.update_context_observation(
                observation_id,
                status=status,
                metadata=metadata,
            )
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise
        return updated

    def defer_context_observation_curation(
        self,
        observation_id: str,
        *,
        lease_token: str,
        error: str,
        retry_after_seconds: int,
        max_attempts: int,
    ) -> ContextObservationCurationStateRecord | None:
        if retry_after_seconds < 1:
            raise ValueError("retry_after_seconds must be at least 1")
        self.begin_immediate_transaction()
        try:
            row = self._fetch_one(
                """
                SELECT attempt_count
                FROM context_observation_curation_state
                WHERE observation_id = ?
                  AND lease_token = ?
                  AND terminal_outcome IS NULL
                  AND lease_until > CURRENT_TIMESTAMP
                """,
                (observation_id, lease_token),
            )
            if row is None:
                self.rollback_transaction()
                return None
            exhausted = int(row["attempt_count"]) >= max(1, max_attempts)
            safe_error = (redact_text(error) or "")[:240]
            cursor = self.connection.execute(
                """
                UPDATE context_observation_curation_state
                SET lease_token = NULL,
                    lease_until = NULL,
                    next_eligible_at = CASE
                        WHEN ? THEN NULL ELSE datetime('now', ?)
                    END,
                    last_error = ?,
                    terminal_outcome = CASE
                        WHEN ? THEN 'retry_budget_exhausted' ELSE NULL
                    END,
                    completed_at = CASE
                        WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL
                    END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE observation_id = ?
                  AND lease_token = ?
                  AND lease_until > CURRENT_TIMESTAMP
                """,
                (
                    int(exhausted),
                    f"+{retry_after_seconds} seconds",
                    safe_error,
                    int(exhausted),
                    int(exhausted),
                    observation_id,
                    lease_token,
                ),
            )
            if cursor.rowcount == 0:
                self.rollback_transaction()
                return None
            if exhausted:
                self.update_context_observation(
                    observation_id,
                    status="curation_failed",
                )
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise
        return self.get_context_observation_curation_state(observation_id)

    def restore_context_observation_curation_state(
        self,
        observation_id: str,
        *,
        attempt_count: int = 0,
        next_eligible_at: str | None = None,
        last_error: str | None = None,
        terminal_outcome: str | None = None,
        completed_at: str | None = None,
    ) -> ContextObservationCurationStateRecord:
        self.connection.execute(
            """
            UPDATE context_observation_curation_state
            SET attempt_count = ?,
                next_eligible_at = ?,
                lease_token = NULL,
                lease_until = NULL,
                last_error = ?,
                terminal_outcome = ?,
                completed_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE observation_id = ?
            """,
            (
                max(0, attempt_count),
                next_eligible_at,
                (redact_text(last_error) or "")[:240] if last_error else None,
                terminal_outcome,
                completed_at,
                observation_id,
            ),
        )
        self.commit()
        state = self.get_context_observation_curation_state(observation_id)
        if state is None:
            raise ValueError(f"Unknown context observation id: {observation_id}")
        return state

    def update_context_observation(
        self,
        observation_id: str,
        *,
        status: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ContextObservationRecord:
        current = self.get_context_observation(observation_id)
        if current is None:
            raise ValueError(f"Unknown context observation id: {observation_id}")
        next_status = status.strip() if status is not None else current.status
        next_metadata = dict(current.metadata)
        if metadata:
            next_metadata.update(metadata)
        self.connection.execute(
            """
            UPDATE context_observations
            SET status = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (next_status, _dump_json(next_metadata), observation_id),
        )
        self.commit()
        updated = self.get_context_observation(observation_id)
        if updated is None:
            raise ValueError(f"Unknown context observation id: {observation_id}")
        return updated

    def archive_context_observation(self, observation_id: str) -> None:
        self.connection.execute(
            """
            UPDATE context_observations
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (observation_id,),
        )
        self.commit()

    def restore_context_observations(
        self,
        context_observation_ids: set[str] | frozenset[str],
    ) -> None:
        if not context_observation_ids:
            return
        self.connection.execute(
            f"""
            UPDATE context_observations
            SET archived_at = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(context_observation_ids))})
            """,
            tuple(context_observation_ids),
        )
        self.commit()

    def archive_context_observations_for_deleted_messages(
        self,
        *,
        save_id: str,
        message_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        if not message_ids:
            return frozenset()
        archived_ids: set[str] = set()
        for observation in self.list_context_observations(save_id):
            if set(observation.source_message_ids) & set(message_ids):
                self.archive_context_observation(observation.id)
                archived_ids.add(observation.id)
        return frozenset(archived_ids)

    def get_scene_snapshot(self, save_id: str) -> SceneSnapshotRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, current_location_id, situation, objective,
                   in_world_time, time_of_day, day_of_week, world_day_index,
                   world_time_day_index, world_time_day_label,
                   world_time_phase, world_time_clock_minutes,
                   world_time_period_label, world_time_source_message_id,
                   world_time_confidence,
                   weather, mood,
                   nearby_objects_json, hazards_json,
                   present_character_ids_json, source_message_id, locked_fields_json,
                   first_seen_message_id, last_updated_message_id, scene_generation
            FROM scene_snapshots
            WHERE save_id = ?
            """,
            (save_id,),
        )
        return (
            _scene_snapshot_with_player_character(
                _scene_snapshot_from_row(row),
                self._player_character_id(save_id),
            )
            if row
            else None
        )

    def delete_scene_snapshot(self, save_id: str) -> str | None:
        row = self._fetch_one(
            """
            SELECT id
            FROM scene_snapshots
            WHERE save_id = ?
            """,
            (save_id,),
        )
        if row is None:
            return None
        self.connection.execute(
            """
            UPDATE context_observations
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND id IN (
                SELECT source_id
                FROM context_sources
                WHERE save_id = ?
                  AND scene_snapshot_id = ?
                  AND json_extract(metadata_json, '$.curation_action')
                      = 'scene_scratch'
            )
            """,
            (save_id, save_id, row["id"]),
        )
        self.connection.execute(
            """
            UPDATE context_sources
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND scene_snapshot_id = ?
              AND archived_at IS NULL
              AND json_extract(metadata_json, '$.curation_action')
                  = 'scene_scratch'
            """,
            (save_id, row["id"]),
        )
        self.connection.execute(
            "DELETE FROM scene_snapshots WHERE save_id = ?",
            (save_id,),
        )
        self.commit()
        return str(row["id"])

    def advance_scene_generation(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
    ) -> SceneSnapshotRecord:
        scene = self.get_scene_snapshot(save_id)
        if scene is None:
            raise ValueError(f"Unknown scene snapshot for save id: {save_id}")
        next_generation = scene.scene_generation + 1
        self.connection.execute(
            """
            UPDATE scene_snapshots
            SET scene_generation = ?,
                first_seen_message_id = COALESCE(?, first_seen_message_id),
                last_updated_message_id = COALESCE(?, last_updated_message_id),
                updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
            """,
            (
                next_generation,
                source_message_id,
                source_message_id,
                save_id,
            ),
        )
        self.archive_stale_scene_scratch(
            save_id=save_id,
            current_scene_snapshot_id=scene.id,
            current_scene_generation=next_generation,
            current_turn_number=self.count_active_messages_by_role(
                save_id,
                roles=("narrator",),
            )["narrator"],
        )
        saved = self.get_scene_snapshot(save_id)
        if saved is None:
            raise ValueError(f"Unknown scene snapshot for save id: {save_id}")
        return saved

    def upsert_scene_snapshot(
        self,
        *,
        save_id: str,
        current_location_id: str | None = None,
        situation: str = "",
        objective: str = "",
        in_world_time: str | object = _UNSET,
        time_of_day: str | object = _UNSET,
        day_of_week: str | object = _UNSET,
        world_day_index: int | None | object = _UNSET,
        world_time_day_index: int | None | object = _UNSET,
        world_time_day_label: str | object = _UNSET,
        world_time_phase: str | object = _UNSET,
        world_time_clock_minutes: int | None | object = _UNSET,
        world_time_period_label: str | object = _UNSET,
        world_time_source_message_id: str | None | object = _UNSET,
        world_time_confidence: float | None | object = _UNSET,
        weather: str = "",
        mood: str = "",
        nearby_objects: list[str] | None = None,
        hazards: list[str] | None = None,
        present_character_ids: list[str] | None = None,
        source_message_id: str | None = None,
        locked_fields: list[str] | None = None,
        snapshot_id: str | None = None,
        first_seen_message_id: str | None = None,
        last_updated_message_id: str | None = None,
        preserve_scene_generation: bool = False,
    ) -> SceneSnapshotRecord:
        self._validate_location_reference(
            save_id=save_id,
            location_id=current_location_id,
            field_name="current_location_id",
        )
        existing = self._fetch_one(
            """
            SELECT id, first_seen_message_id, current_location_id, scene_generation,
                   in_world_time, time_of_day, day_of_week, world_day_index,
                   world_time_day_index, world_time_day_label,
                   world_time_phase, world_time_clock_minutes,
                   world_time_period_label, world_time_source_message_id,
                   world_time_confidence
            FROM scene_snapshots
            WHERE save_id = ?
            """,
            (save_id,),
        )
        first_seen = (
            existing["first_seen_message_id"]
            if existing is not None and existing["first_seen_message_id"] is not None
            else first_seen_message_id
        )
        if first_seen is None:
            first_seen = source_message_id
        last_updated = (
            last_updated_message_id
            if last_updated_message_id is not None
            else source_message_id
        )
        scene_generation = 1
        if existing is not None:
            scene_generation = int(existing["scene_generation"])
            if (
                existing["current_location_id"] != current_location_id
                and not preserve_scene_generation
            ):
                scene_generation += 1
        provided_legacy_values = {
            "in_world_time": in_world_time,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "world_day_index": world_day_index,
        }
        in_world_time = cast(
            str,
            _scene_world_time_value(
                in_world_time,
                existing=existing,
                column="in_world_time",
                fallback="",
            ),
        )
        time_of_day = cast(
            str,
            _scene_world_time_value(
                time_of_day,
                existing=existing,
                column="time_of_day",
                fallback="",
            ),
        )
        day_of_week = cast(
            str,
            _scene_world_time_value(
                day_of_week,
                existing=existing,
                column="day_of_week",
                fallback="",
            ),
        )
        world_day_index = cast(
            int | None,
            _scene_world_time_value(
                world_day_index,
                existing=existing,
                column="world_day_index",
                fallback=None,
            ),
        )
        canonical_fields_provided = any(
            value is not _UNSET
            for value in (
                world_time_day_index,
                world_time_day_label,
                world_time_phase,
                world_time_clock_minutes,
                world_time_period_label,
                world_time_source_message_id,
                world_time_confidence,
            )
        )
        canonical_legacy_display_fields_provided = any(
            value is not _UNSET
            for value in (
                world_time_day_label,
                world_time_phase,
                world_time_clock_minutes,
                world_time_period_label,
            )
        )
        legacy_field_changes = {
            column: _scene_world_time_provided_value_changed(
                provided,
                existing=existing,
                column=column,
                effective=effective,
            )
            for column, provided, effective in (
                (
                    "in_world_time",
                    provided_legacy_values["in_world_time"],
                    in_world_time,
                ),
                ("time_of_day", provided_legacy_values["time_of_day"], time_of_day),
                ("day_of_week", provided_legacy_values["day_of_week"], day_of_week),
                (
                    "world_day_index",
                    provided_legacy_values["world_day_index"],
                    world_day_index,
                ),
            )
        }
        legacy_fields_changed = any(legacy_field_changes.values())
        legacy_display_fields_changed = any(
            legacy_field_changes[column]
            for column in ("in_world_time", "time_of_day", "day_of_week")
        )
        legacy_time_of_day_for_canonical = (
            ""
            if (
                legacy_field_changes["in_world_time"]
                and not legacy_field_changes["time_of_day"]
            )
            else time_of_day
        )
        legacy_world_time = canonical_world_time_from_legacy(
            in_world_time=in_world_time,
            time_of_day=legacy_time_of_day_for_canonical,
            day_of_week=day_of_week,
            world_day_index=world_day_index,
            source_message_id=source_message_id,
        )
        if not canonical_fields_provided and legacy_fields_changed:
            canonical_world_time = canonical_world_time_from_values(
                day_index=legacy_world_time.day_index,
                day_label=legacy_world_time.day_label,
                phase=legacy_world_time.phase,
                clock_minutes=(
                    legacy_world_time.clock_minutes
                    if legacy_world_time.clock_minutes is not None
                    else _scene_world_time_value(
                        _UNSET,
                        existing=existing,
                        column="world_time_clock_minutes",
                        fallback=None,
                    )
                ),
                period_label=_scene_world_time_value(
                    _UNSET,
                    existing=existing,
                    column="world_time_period_label",
                    fallback=legacy_world_time.period_label,
                ),
                source_message_id=(
                    legacy_world_time.source_message_id
                    or _scene_world_time_value(
                        _UNSET,
                        existing=existing,
                        column="world_time_source_message_id",
                        fallback=None,
                    )
                ),
                confidence=_scene_world_time_value(
                    _UNSET,
                    existing=existing,
                    column="world_time_confidence",
                    fallback=legacy_world_time.confidence,
                ),
                legacy_in_world_time=in_world_time,
                legacy_time_of_day=legacy_time_of_day_for_canonical,
                legacy_day_of_week=day_of_week,
                legacy_world_day_index=world_day_index,
            )
        else:
            canonical_world_time = canonical_world_time_from_values(
                day_index=_scene_world_time_value(
                    world_time_day_index,
                    existing=existing,
                    column="world_time_day_index",
                    fallback=legacy_world_time.day_index,
                ),
                day_label=_scene_world_time_value(
                    world_time_day_label,
                    existing=existing,
                    column="world_time_day_label",
                    fallback=legacy_world_time.day_label,
                ),
                phase=_scene_world_time_value(
                    world_time_phase,
                    existing=existing,
                    column="world_time_phase",
                    fallback=legacy_world_time.phase,
                ),
                clock_minutes=_scene_world_time_value(
                    world_time_clock_minutes,
                    existing=existing,
                    column="world_time_clock_minutes",
                    fallback=legacy_world_time.clock_minutes,
                ),
                period_label=_scene_world_time_value(
                    world_time_period_label,
                    existing=existing,
                    column="world_time_period_label",
                    fallback=legacy_world_time.period_label,
                ),
                source_message_id=_scene_world_time_value(
                    world_time_source_message_id,
                    existing=existing,
                    column="world_time_source_message_id",
                    fallback=legacy_world_time.source_message_id,
                ),
                confidence=_scene_world_time_value(
                    world_time_confidence,
                    existing=existing,
                    column="world_time_confidence",
                    fallback=legacy_world_time.confidence,
                ),
                legacy_in_world_time=in_world_time,
                legacy_time_of_day=time_of_day,
                legacy_day_of_week=day_of_week,
                legacy_world_day_index=world_day_index,
            )
        if legacy_field_changes["in_world_time"]:
            legacy_fields = legacy_world_time_fields(canonical_world_time)
            time_of_day = cast(str, legacy_fields["time_of_day"])
            day_of_week = cast(str, legacy_fields["day_of_week"])
            world_day_index = cast(int | None, legacy_fields["world_day_index"])
        elif canonical_legacy_display_fields_provided or (
            legacy_display_fields_changed and existing is not None
        ):
            legacy_fields = legacy_world_time_fields(canonical_world_time)
            in_world_time = cast(str, legacy_fields["in_world_time"])
            time_of_day = cast(str, legacy_fields["time_of_day"])
            day_of_week = cast(str, legacy_fields["day_of_week"])
            world_day_index = cast(int | None, legacy_fields["world_day_index"])
        record = SceneSnapshotRecord(
            id=existing["id"] if existing else snapshot_id or _new_id(),
            save_id=save_id,
            current_location_id=current_location_id,
            situation=situation,
            objective=objective,
            in_world_time=in_world_time,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            world_day_index=world_day_index,
            world_time_day_index=canonical_world_time.day_index,
            world_time_day_label=canonical_world_time.day_label,
            world_time_phase=canonical_world_time.phase,
            world_time_clock_minutes=canonical_world_time.clock_minutes,
            world_time_period_label=canonical_world_time.period_label,
            world_time_source_message_id=canonical_world_time.source_message_id,
            world_time_confidence=canonical_world_time.confidence,
            weather=weather,
            mood=mood,
            nearby_objects=list(nearby_objects or []),
            hazards=list(hazards or []),
            present_character_ids=_present_character_ids_with_player_character(
                present_character_ids or [],
                self._player_character_id(save_id),
            ),
            source_message_id=source_message_id,
            locked_fields=list(locked_fields or []),
            first_seen_message_id=first_seen,
            last_updated_message_id=last_updated,
            scene_generation=scene_generation,
        )
        self.connection.execute(
            """
            INSERT INTO scene_snapshots(
                id, save_id, current_location_id, situation, objective,
                in_world_time, time_of_day, day_of_week, world_day_index,
                world_time_day_index, world_time_day_label, world_time_phase,
                world_time_clock_minutes, world_time_period_label,
                world_time_source_message_id, world_time_confidence,
                weather, mood,
                nearby_objects_json, hazards_json,
                present_character_ids_json, source_message_id, locked_fields_json,
                first_seen_message_id, last_updated_message_id, scene_generation
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(save_id) DO UPDATE SET
                current_location_id = excluded.current_location_id,
                situation = excluded.situation,
                objective = excluded.objective,
                in_world_time = excluded.in_world_time,
                time_of_day = excluded.time_of_day,
                day_of_week = excluded.day_of_week,
                world_day_index = excluded.world_day_index,
                world_time_day_index = excluded.world_time_day_index,
                world_time_day_label = excluded.world_time_day_label,
                world_time_phase = excluded.world_time_phase,
                world_time_clock_minutes = excluded.world_time_clock_minutes,
                world_time_period_label = excluded.world_time_period_label,
                world_time_source_message_id = excluded.world_time_source_message_id,
                world_time_confidence = excluded.world_time_confidence,
                weather = excluded.weather,
                mood = excluded.mood,
                nearby_objects_json = excluded.nearby_objects_json,
                hazards_json = excluded.hazards_json,
                present_character_ids_json = excluded.present_character_ids_json,
                source_message_id = excluded.source_message_id,
                locked_fields_json = excluded.locked_fields_json,
                first_seen_message_id = COALESCE(
                    scene_snapshots.first_seen_message_id,
                    excluded.first_seen_message_id
                ),
                last_updated_message_id = excluded.last_updated_message_id,
                scene_generation = excluded.scene_generation,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                record.id,
                record.save_id,
                record.current_location_id,
                record.situation,
                record.objective,
                record.in_world_time,
                record.time_of_day,
                record.day_of_week,
                record.world_day_index,
                record.world_time_day_index,
                record.world_time_day_label,
                record.world_time_phase,
                record.world_time_clock_minutes,
                record.world_time_period_label,
                record.world_time_source_message_id,
                record.world_time_confidence,
                record.weather,
                record.mood,
                _dump_json(record.nearby_objects),
                _dump_json(record.hazards),
                _dump_json(record.present_character_ids),
                record.source_message_id,
                _dump_json(record.locked_fields),
                record.first_seen_message_id,
                record.last_updated_message_id,
                record.scene_generation,
            ),
        )
        location_changed = (
            existing is not None
            and existing["current_location_id"] != current_location_id
        )
        generation_changed = (
            existing is not None
            and scene_generation != int(existing["scene_generation"])
        )
        if location_changed or generation_changed:
            self.connection.execute(
                """
                UPDATE active_threads
                SET archived_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE save_id = ?
                  AND archived_at IS NULL
                  AND (
                        lower(trim(visibility)) IN (
                            'scene',
                            'scene local',
                            'scene only',
                            'current scene',
                            'local'
                        )
                        OR lower(visibility) LIKE '%scene%'
                        OR lower(visibility) LIKE '%local%'
                      )
                """,
                (save_id,),
            )
        if (
            generation_changed
        ):
            self.connection.execute(
                """
                UPDATE context_observations
                SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE save_id = ?
                  AND id IN (
                    SELECT source_id
                    FROM context_sources
                    WHERE save_id = ?
                      AND archived_at IS NULL
                      AND json_extract(metadata_json, '$.curation_action')
                          = 'scene_scratch'
                      AND (
                            scene_snapshot_id IS NOT ?
                            OR scene_generation IS NOT ?
                          )
                )
                """,
                (
                    save_id,
                    save_id,
                    record.id,
                    record.scene_generation,
                ),
            )
            self.connection.execute(
                """
                UPDATE context_sources
                SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE save_id = ?
                  AND archived_at IS NULL
                  AND json_extract(metadata_json, '$.curation_action')
                      = 'scene_scratch'
                  AND (
                        scene_snapshot_id IS NOT ?
                        OR scene_generation IS NOT ?
                      )
                """,
                (save_id, record.id, record.scene_generation),
            )
        self.commit()
        saved = self.get_scene_snapshot(save_id)
        if saved is None:
            raise ValueError(f"Unknown save id: {save_id}")
        return saved

    def add_location(
        self,
        *,
        save_id: str,
        name: str,
        aliases: list[str] | None = None,
        description: str = "",
        visual_description: str = "",
        parent_location_id: str | None = None,
        connections: list[str] | None = None,
        status: str = "",
        hazards: list[str] | None = None,
        source_message_id: str | None = None,
        locked_fields: list[str] | None = None,
        location_id: str | None = None,
        first_seen_message_id: str | None = None,
        last_updated_message_id: str | None = None,
    ) -> LocationRecord:
        self._validate_location_reference(
            save_id=save_id,
            location_id=parent_location_id,
            field_name="parent_location_id",
        )
        record = LocationRecord(
            id=location_id or _new_id(),
            save_id=save_id,
            name=name,
            aliases=list(aliases or []),
            description=description,
            visual_description=visual_description,
            parent_location_id=parent_location_id,
            connections=list(connections or []),
            status=status,
            hazards=list(hazards or []),
            source_message_id=source_message_id,
            locked_fields=list(locked_fields or []),
            first_seen_message_id=first_seen_message_id or source_message_id,
            last_updated_message_id=last_updated_message_id or source_message_id,
        )
        self.connection.execute(
            """
            INSERT INTO locations(
                id, save_id, name, aliases_json, description, visual_description,
                parent_location_id, connections_json, status, hazards_json,
                source_message_id, locked_fields_json, first_seen_message_id,
                last_updated_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _location_params(record),
        )
        self.commit()
        return record

    def update_location(self, location: LocationRecord) -> LocationRecord:
        self._validate_location_reference(
            save_id=location.save_id,
            location_id=location.parent_location_id,
            field_name="parent_location_id",
        )
        self.connection.execute(
            """
            UPDATE locations
            SET name = ?, aliases_json = ?, description = ?,
                visual_description = ?, parent_location_id = ?,
                connections_json = ?, status = ?, hazards_json = ?,
                source_message_id = ?, locked_fields_json = ?,
                first_seen_message_id = ?,
                last_updated_message_id = ?,
                updated_at = CURRENT_TIMESTAMP, archived_at = NULL
            WHERE id = ? AND save_id = ?
            """,
            (
                location.name,
                _dump_json(location.aliases),
                location.description,
                location.visual_description,
                location.parent_location_id,
                _dump_json(location.connections),
                location.status,
                _dump_json(location.hazards),
                location.source_message_id,
                _dump_json(location.locked_fields),
                location.first_seen_message_id or location.source_message_id,
                location.last_updated_message_id or location.source_message_id,
                location.id,
                location.save_id,
            ),
        )
        self.commit()
        saved = self.get_location(location.id)
        if saved is None:
            raise ValueError(f"Unknown location id: {location.id}")
        return saved

    def get_location(self, location_id: str) -> LocationRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, name, aliases_json, description, visual_description,
                   parent_location_id, connections_json, status, hazards_json,
                   source_message_id, locked_fields_json, first_seen_message_id,
                   last_updated_message_id
            FROM locations
            WHERE id = ? AND archived_at IS NULL
            """,
            (location_id,),
        )
        return _location_from_row(row) if row else None

    def list_locations(self, save_id: str) -> list[LocationRecord]:
        rows = self._fetch_all(
            """
            SELECT id, save_id, name, aliases_json, description, visual_description,
                   parent_location_id, connections_json, status, hazards_json,
                   source_message_id, locked_fields_json, first_seen_message_id,
                   last_updated_message_id
            FROM locations
            WHERE save_id = ? AND archived_at IS NULL
            ORDER BY name, created_at, rowid
            """,
            (save_id,),
        )
        return [_location_from_row(row) for row in rows]

    def archive_location(self, location_id: str) -> None:
        self.connection.execute(
            """
            UPDATE locations
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (location_id,),
        )
        self.commit()

    def restore_locations(self, location_ids: set[str] | frozenset[str]) -> None:
        if not location_ids:
            return
        self.connection.execute(
            f"""
            UPDATE locations
            SET archived_at = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(location_ids))})
            """,
            tuple(location_ids),
        )
        self.commit()

    def add_character(
        self,
        *,
        save_id: str,
        name: str,
        aliases: list[str] | None = None,
        role: str = "",
        age: str = "",
        known_state: str = "",
        history: str = "",
        met: bool = False,
        appearance: str = "",
        visual_notes: str = "",
        current_clothing: str = "",
        personality: str = "",
        voice: str = "",
        relationships: dict[str, object] | None = None,
        goals: str = "",
        motivations: str = "",
        current_intent: str = "",
        boundaries: str = "",
        attitude_toward_player: str = "",
        cooperation_conditions: str = "",
        status: str = "",
        location_id: str | None = None,
        private_notes: str = "",
        source_message_id: str | None = None,
        locked_fields: list[str] | None = None,
        protected_from_maintenance: bool = False,
        is_player_character: bool = False,
        texting_style: str = "",
        contact_name: str = "",
        character_id: str | None = None,
        first_seen_message_id: str | None = None,
        last_updated_message_id: str | None = None,
        content_rating: str = "unclassified",
    ) -> CharacterRecord:
        self._validate_location_reference(
            save_id=save_id,
            location_id=location_id,
            field_name="location_id",
        )
        if content_rating == "unclassified" and source_message_id is not None:
            source_message = self.get_message(
                save_id=save_id,
                message_id=source_message_id,
            )
            if source_message is not None:
                content_rating = source_message.content_rating
        resolved_history = history.strip() if history.strip() else known_state
        record = CharacterRecord(
            id=character_id or _new_id(),
            save_id=save_id,
            name=name,
            aliases=list(aliases or []),
            role=role,
            age=age,
            known_state=resolved_history,
            history=resolved_history,
            met=met,
            appearance=appearance,
            visual_notes=visual_notes,
            current_clothing=current_clothing,
            personality=personality,
            voice=voice,
            relationships=dict(relationships or {}),
            goals=goals,
            motivations=motivations,
            current_intent=current_intent,
            boundaries=boundaries,
            attitude_toward_player=attitude_toward_player,
            cooperation_conditions=cooperation_conditions,
            status=status,
            location_id=location_id,
            private_notes=private_notes,
            source_message_id=source_message_id,
            locked_fields=list(locked_fields or []),
            protected_from_maintenance=protected_from_maintenance,
            is_player_character=is_player_character,
            texting_style=texting_style,
            contact_name=contact_name,
            first_seen_message_id=first_seen_message_id or source_message_id,
            last_updated_message_id=last_updated_message_id or source_message_id,
            content_rating=content_rating,
        )
        if record.is_player_character:
            record = replace(record, protected_from_maintenance=True)
            self._clear_other_player_characters(record.save_id, record.id)
        self.connection.execute(
            """
            INSERT INTO characters(
                id, save_id, name, aliases_json, role, age, known_state, history, met,
                appearance, visual_notes, current_clothing, personality, voice,
                relationships_json,
                goals, motivations, current_intent, boundaries,
                attitude_toward_player, cooperation_conditions, status,
                location_id, private_notes, source_message_id,
                locked_fields_json, protected_from_maintenance,
                is_player_character, texting_style, contact_name,
                first_seen_message_id, last_updated_message_id, content_rating
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            _character_params(record),
        )
        if record.is_player_character:
            self._ensure_player_character_present(record.save_id, record.id)
        self.commit()
        return record

    def update_character(self, character: CharacterRecord) -> CharacterRecord:
        self._validate_location_reference(
            save_id=character.save_id,
            location_id=character.location_id,
            field_name="location_id",
        )
        if character.is_player_character:
            character = replace(character, protected_from_maintenance=True)
            self._clear_other_player_characters(character.save_id, character.id)
        self.connection.execute(
            """
            UPDATE characters
            SET name = ?, aliases_json = ?, role = ?, age = ?, known_state = ?,
                history = ?, met = ?,
                appearance = ?, visual_notes = ?, current_clothing = ?,
                personality = ?, voice = ?, relationships_json = ?, goals = ?,
                motivations = ?,
                current_intent = ?, boundaries = ?, attitude_toward_player = ?,
                cooperation_conditions = ?, status = ?, location_id = ?,
                private_notes = ?, source_message_id = ?, locked_fields_json = ?,
                protected_from_maintenance = ?, is_player_character = ?,
                texting_style = ?, contact_name = ?,
                first_seen_message_id = ?,
                last_updated_message_id = ?,
                content_rating = ?,
                updated_at = CURRENT_TIMESTAMP, archived_at = NULL
            WHERE id = ? AND save_id = ?
            """,
            (
                character.name,
                _dump_json(character.aliases),
                character.role,
                character.age,
                character.known_state,
                character.history,
                int(character.met),
                character.appearance,
                character.visual_notes,
                character.current_clothing,
                character.personality,
                character.voice,
                _dump_json(character.relationships),
                character.goals,
                character.motivations,
                character.current_intent,
                character.boundaries,
                character.attitude_toward_player,
                character.cooperation_conditions,
                character.status,
                character.location_id,
                character.private_notes,
                character.source_message_id,
                _dump_json(character.locked_fields),
                int(character.protected_from_maintenance),
                int(character.is_player_character),
                character.texting_style,
                character.contact_name,
                character.first_seen_message_id or character.source_message_id,
                character.last_updated_message_id or character.source_message_id,
                character.content_rating,
                character.id,
                character.save_id,
            ),
        )
        if character.is_player_character:
            self._ensure_player_character_present(character.save_id, character.id)
        self.commit()
        saved = self.get_character(character.id)
        if saved is None:
            raise ValueError(f"Unknown character id: {character.id}")
        return saved

    def get_character(self, character_id: str) -> CharacterRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, name, aliases_json, role, age, known_state, history,
                   met,
                   appearance, visual_notes, current_clothing, personality, voice,
                   relationships_json,
                   goals, motivations, current_intent, boundaries,
                   attitude_toward_player, cooperation_conditions, status,
                   location_id, private_notes, source_message_id,
                   locked_fields_json, protected_from_maintenance,
                   is_player_character, texting_style, contact_name,
                   first_seen_message_id, last_updated_message_id, content_rating
            FROM characters
            WHERE id = ? AND archived_at IS NULL
            """,
            (character_id,),
        )
        return _character_from_row(row) if row else None

    def list_characters(self, save_id: str) -> list[CharacterRecord]:
        rows = self._fetch_all(
            """
            SELECT id, save_id, name, aliases_json, role, age, known_state, history,
                   met,
                   appearance, visual_notes, current_clothing, personality, voice,
                   relationships_json,
                   goals, motivations, current_intent, boundaries,
                   attitude_toward_player, cooperation_conditions, status,
                   location_id, private_notes, source_message_id,
                   locked_fields_json, protected_from_maintenance,
                   is_player_character, texting_style, contact_name,
                   first_seen_message_id, last_updated_message_id, content_rating
            FROM characters
            WHERE save_id = ? AND archived_at IS NULL
            ORDER BY name, created_at, rowid
            """,
            (save_id,),
        )
        return [_character_from_row(row) for row in rows]

    def has_unprotected_character(self, save_id: str) -> bool:
        row = self._fetch_one(
            """
            SELECT 1
            FROM characters
            WHERE save_id = ?
              AND archived_at IS NULL
              AND protected_from_maintenance = 0
            LIMIT 1
            """,
            (save_id,),
        )
        return row is not None

    def upsert_character_contact_state(
        self,
        *,
        save_id: str,
        player_character_id: str,
        character_id: str,
        player_has_character_number: bool | None = None,
        character_has_player_number: bool | None = None,
        source_message_id: str | None = None,
        source_text_message_id: str | None = None,
        state_id: str | None = None,
    ) -> CharacterContactStateRecord:
        self._validate_character_reference(
            save_id=save_id,
            character_id=player_character_id,
            field_name="player_character_id",
        )
        self._validate_character_reference(
            save_id=save_id,
            character_id=character_id,
            field_name="character_id",
        )
        if player_character_id == character_id:
            raise ValueError("character contact state requires distinct characters")
        record_id = state_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO character_contact_states(
                id, save_id, player_character_id, character_id,
                player_has_character_number, character_has_player_number,
                source_message_id, source_text_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(save_id, player_character_id, character_id) DO UPDATE SET
                player_has_character_number = CASE
                    WHEN excluded.player_has_character_number = 1 THEN 1
                    ELSE character_contact_states.player_has_character_number
                END,
                character_has_player_number = CASE
                    WHEN excluded.character_has_player_number = 1 THEN 1
                    ELSE character_contact_states.character_has_player_number
                END,
                source_message_id = COALESCE(
                    character_contact_states.source_message_id,
                    excluded.source_message_id
                ),
                source_text_message_id = COALESCE(
                    character_contact_states.source_text_message_id,
                    excluded.source_text_message_id
                ),
                archived_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                record_id,
                save_id,
                player_character_id,
                character_id,
                int(bool(player_has_character_number)),
                int(bool(character_has_player_number)),
                source_message_id,
                source_text_message_id,
            ),
        )
        self.commit()
        saved = self.get_character_contact_state(
            save_id=save_id,
            player_character_id=player_character_id,
            character_id=character_id,
        )
        if saved is None:
            raise ValueError("Character contact state was not saved")
        return saved

    def set_character_contact_state(
        self,
        *,
        save_id: str,
        player_character_id: str,
        character_id: str,
        player_has_character_number: bool,
        character_has_player_number: bool,
        source_message_id: str | None = None,
        source_text_message_id: str | None = None,
        state_id: str | None = None,
    ) -> CharacterContactStateRecord:
        self._validate_character_reference(
            save_id=save_id,
            character_id=player_character_id,
            field_name="player_character_id",
        )
        self._validate_character_reference(
            save_id=save_id,
            character_id=character_id,
            field_name="character_id",
        )
        if player_character_id == character_id:
            raise ValueError("character contact state requires distinct characters")
        record_id = state_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO character_contact_states(
                id, save_id, player_character_id, character_id,
                player_has_character_number, character_has_player_number,
                source_message_id, source_text_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(save_id, player_character_id, character_id) DO UPDATE SET
                player_has_character_number = excluded.player_has_character_number,
                character_has_player_number = excluded.character_has_player_number,
                source_message_id = COALESCE(
                    excluded.source_message_id,
                    character_contact_states.source_message_id
                ),
                source_text_message_id = COALESCE(
                    excluded.source_text_message_id,
                    character_contact_states.source_text_message_id
                ),
                archived_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                record_id,
                save_id,
                player_character_id,
                character_id,
                int(player_has_character_number),
                int(character_has_player_number),
                source_message_id,
                source_text_message_id,
            ),
        )
        self.commit()
        saved = self.get_character_contact_state(
            save_id=save_id,
            player_character_id=player_character_id,
            character_id=character_id,
        )
        if saved is None:
            raise ValueError("Character contact state was not saved")
        return saved

    def replace_character_contact_state(
        self,
        *,
        save_id: str,
        player_character_id: str,
        character_id: str,
        player_has_character_number: bool,
        character_has_player_number: bool,
        source_message_id: str | None = None,
        source_text_message_id: str | None = None,
        state_id: str | None = None,
    ) -> CharacterContactStateRecord:
        self._validate_character_reference(
            save_id=save_id,
            character_id=player_character_id,
            field_name="player_character_id",
        )
        self._validate_character_reference(
            save_id=save_id,
            character_id=character_id,
            field_name="character_id",
        )
        if player_character_id == character_id:
            raise ValueError("character contact state requires distinct characters")
        record_id = state_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO character_contact_states(
                id, save_id, player_character_id, character_id,
                player_has_character_number, character_has_player_number,
                source_message_id, source_text_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(save_id, player_character_id, character_id) DO UPDATE SET
                player_has_character_number = excluded.player_has_character_number,
                character_has_player_number = excluded.character_has_player_number,
                source_message_id = excluded.source_message_id,
                source_text_message_id = excluded.source_text_message_id,
                archived_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                record_id,
                save_id,
                player_character_id,
                character_id,
                int(player_has_character_number),
                int(character_has_player_number),
                source_message_id,
                source_text_message_id,
            ),
        )
        self.commit()
        saved = self.get_character_contact_state(
            save_id=save_id,
            player_character_id=player_character_id,
            character_id=character_id,
        )
        if saved is None:
            raise ValueError("Character contact state was not saved")
        return saved

    def get_character_contact_state(
        self,
        *,
        save_id: str,
        player_character_id: str,
        character_id: str,
    ) -> CharacterContactStateRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, player_character_id, character_id,
                   player_has_character_number, character_has_player_number,
                   source_message_id, source_text_message_id, created_at,
                   updated_at, archived_at
            FROM character_contact_states
            WHERE save_id = ?
              AND player_character_id = ?
              AND character_id = ?
              AND archived_at IS NULL
            """,
            (save_id, player_character_id, character_id),
        )
        return _character_contact_state_from_row(row) if row is not None else None

    def list_character_contact_states(
        self,
        save_id: str,
    ) -> list[CharacterContactStateRecord]:
        rows = self._fetch_all(
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
        )
        return [_character_contact_state_from_row(row) for row in rows]

    def archive_character_contact_state(self, state_id: str) -> None:
        self.connection.execute(
            """
            UPDATE character_contact_states
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (state_id,),
        )
        self.commit()

    def character_text_outbound_allowed(
        self,
        *,
        save_id: str,
        character_id: str,
    ) -> bool:
        player_character_id = self._player_character_id(save_id)
        if player_character_id is None:
            return False
        state = self.get_character_contact_state(
            save_id=save_id,
            player_character_id=player_character_id,
            character_id=character_id,
        )
        return state is not None and state.player_has_character_number

    def can_character_proactively_text(
        self,
        *,
        save_id: str,
        character_id: str,
    ) -> bool:
        player_character_id = self._player_character_id(save_id)
        if player_character_id is None:
            return False
        state = self.get_character_contact_state(
            save_id=save_id,
            player_character_id=player_character_id,
            character_id=character_id,
        )
        return state is not None and state.character_has_player_number

    def get_or_create_character_text_thread(
        self,
        *,
        save_id: str,
        character_id: str,
        title: str = "",
    ) -> CharacterTextThreadRecord:
        self._validate_character_reference(
            save_id=save_id,
            character_id=character_id,
            field_name="character_id",
        )
        existing = self._fetch_one(
            f"""
            SELECT {_CHARACTER_TEXT_THREAD_COLUMNS}
            FROM character_text_threads
            WHERE save_id = ? AND character_id = ? AND kind = 'direct'
            """,
            (save_id, character_id),
        )
        if existing is not None:
            if existing["archived_at"] is not None:
                self.connection.execute(
                    """
                    UPDATE character_text_threads
                    SET archived_at = NULL, status = 'active',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (existing["id"],),
                )
                self.commit()
                revived = self.get_character_text_thread(
                    thread_id=str(existing["id"]),
                    save_id=save_id,
                    include_archived=True,
                )
                if revived is None:
                    raise ValueError("Character text thread was not restored")
                self._ensure_character_text_thread_participant(
                    save_id=save_id,
                    thread_id=revived.id,
                    character_id=character_id,
                    ordinal=0,
                )
                return revived
            thread = _character_text_thread_from_row(existing)
            self._ensure_character_text_thread_participant(
                save_id=save_id,
                thread_id=thread.id,
                character_id=character_id,
                ordinal=0,
            )
            return thread
        thread_id = _new_id()
        self.connection.execute(
            """
            INSERT INTO character_text_threads(
                id, save_id, character_id, title, kind
            )
            VALUES (?, ?, ?, ?, 'direct')
            """,
            (thread_id, save_id, character_id, title.strip()),
        )
        self._ensure_character_text_thread_participant(
            save_id=save_id,
            thread_id=thread_id,
            character_id=character_id,
            ordinal=0,
        )
        self.commit()
        created = self.get_character_text_thread(thread_id=thread_id, save_id=save_id)
        if created is None:
            raise ValueError("Character text thread was not saved")
        return created

    def create_character_text_group_thread(
        self,
        *,
        save_id: str,
        title: str,
        character_ids: Iterable[str],
    ) -> CharacterTextThreadRecord:
        ordered_character_ids = tuple(
            dict.fromkeys(character_id.strip() for character_id in character_ids)
        )
        if len(ordered_character_ids) < 2:
            raise ValueError("Group text threads require at least two characters")
        for character_id in ordered_character_ids:
            self._validate_character_reference(
                save_id=save_id,
                character_id=character_id,
                field_name="character_id",
            )
        thread_id = _new_id()
        normalized_title = title.strip()
        if not normalized_title:
            names = [
                character.name
                for character_id in ordered_character_ids
                if (character := self.get_character(character_id)) is not None
            ]
            normalized_title = ", ".join(names)
        self.connection.execute(
            """
            INSERT INTO character_text_threads(
                id, save_id, character_id, title, kind
            )
            VALUES (?, ?, NULL, ?, 'group')
            """,
            (thread_id, save_id, normalized_title),
        )
        for ordinal, character_id in enumerate(ordered_character_ids):
            self._ensure_character_text_thread_participant(
                save_id=save_id,
                thread_id=thread_id,
                character_id=character_id,
                ordinal=ordinal,
            )
        self.commit()
        created = self.get_character_text_thread(thread_id=thread_id, save_id=save_id)
        if created is None:
            raise ValueError("Character text group thread was not saved")
        return created

    def get_character_text_thread(
        self,
        *,
        thread_id: str,
        save_id: str | None = None,
        include_archived: bool = False,
    ) -> CharacterTextThreadRecord | None:
        clauses = ["id = ?"]
        params: list[object] = [thread_id]
        if save_id is not None:
            clauses.append("save_id = ?")
            params.append(save_id)
        if not include_archived:
            clauses.append("archived_at IS NULL")
        row = self._fetch_one(
            f"""
            SELECT {_CHARACTER_TEXT_THREAD_COLUMNS}
            FROM character_text_threads
            WHERE {' AND '.join(clauses)}
            """,
            tuple(params),
        )
        return _character_text_thread_from_row(row) if row is not None else None

    def list_character_text_threads(
        self,
        save_id: str,
        *,
        include_archived: bool = False,
    ) -> list[CharacterTextThreadRecord]:
        archive_filter = "" if include_archived else "AND archived_at IS NULL"
        rows = self._fetch_all(
            f"""
            SELECT {_CHARACTER_TEXT_THREAD_COLUMNS}
            FROM character_text_threads
            WHERE save_id = ? {archive_filter}
            ORDER BY updated_at DESC, created_at DESC, rowid DESC
            """,
            (save_id,),
        )
        return [_character_text_thread_from_row(row) for row in rows]

    def list_character_text_thread_participants(
        self,
        *,
        save_id: str,
        thread_id: str | None = None,
        include_archived: bool = False,
    ) -> list[CharacterTextThreadParticipantRecord]:
        clauses = ["save_id = ?"]
        params: list[object] = [save_id]
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if not include_archived:
            clauses.append("archived_at IS NULL")
        rows = self._fetch_all(
            f"""
            SELECT {_CHARACTER_TEXT_THREAD_PARTICIPANT_COLUMNS}
            FROM character_text_thread_participants
            WHERE {' AND '.join(clauses)}
            ORDER BY thread_id, ordinal, created_at, rowid
            """,
            tuple(params),
        )
        return [_character_text_thread_participant_from_row(row) for row in rows]

    def update_character_text_thread_memory(
        self,
        *,
        save_id: str,
        thread_id: str,
        body: str,
        message_count: int,
    ) -> CharacterTextThreadRecord:
        normalized_body = body.strip()
        normalized_count = max(0, int(message_count)) if normalized_body else 0
        self.connection.execute(
            """
            UPDATE character_text_threads
            SET memory_body = ?,
                memory_message_count = ?,
                memory_updated_at = CASE
                    WHEN ? = '' THEN NULL
                    ELSE CURRENT_TIMESTAMP
                END
            WHERE save_id = ? AND id = ? AND archived_at IS NULL
            """,
            (
                normalized_body,
                normalized_count,
                normalized_body,
                save_id,
                thread_id,
            ),
        )
        self.commit()
        updated = self.get_character_text_thread(
            save_id=save_id,
            thread_id=thread_id,
        )
        if updated is None:
            raise ValueError(f"Unknown character text thread id: {thread_id}")
        return updated

    def append_character_text_message(
        self,
        *,
        save_id: str,
        thread_id: str,
        character_id: str | None,
        sender: str,
        body: str,
        sender_character_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        token_estimate: int | None = None,
        message_id: str | None = None,
        delivery_status: str = "sent",
        delivery_error: str | None = None,
        delivery_job_id: str | None = None,
        delivery_attempt: int = 0,
        in_world_sent_at: str | None = None,
        delivered_at: str | None = None,
        read_at: str | None = None,
        reply_to_message_id: str | None = None,
        content_rating: str = "unclassified",
    ) -> CharacterTextMessageRecord:
        thread = self.get_character_text_thread(thread_id=thread_id, save_id=save_id)
        if thread is None:
            raise ValueError(f"Unknown character text thread id: {thread_id}")
        normalized_sender = sender.strip().lower()
        if normalized_sender not in {"player", "character"}:
            raise ValueError("character text sender must be player or character")
        resolved_sender_character_id = self._resolve_character_text_sender_character_id(
            save_id=save_id,
            thread=thread,
            character_id=character_id,
            sender=normalized_sender,
            sender_character_id=sender_character_id,
        )
        _validate_character_text_delivery_status(delivery_status)
        record_id = message_id or _new_id()
        normalized_reply_to_message_id = _blank_to_none(reply_to_message_id)
        if normalized_reply_to_message_id is not None:
            if normalized_reply_to_message_id == record_id:
                raise ValueError("character text cannot reply to itself")
            reply_to = self.get_character_text_message(
                save_id=save_id,
                message_id=normalized_reply_to_message_id,
            )
            if reply_to is None:
                raise ValueError(
                    f"Unknown character text reply id: {normalized_reply_to_message_id}"
                )
            if reply_to.thread_id != thread_id:
                raise ValueError("character text reply must be in the same thread")
        resolved_delivered_at = (
            delivered_at
            if delivered_at is not None
            else (_timestamp_text(_utc_now()) if delivery_status == "sent" else None)
        )
        self.connection.execute(
            """
            INSERT INTO character_text_messages(
                id, save_id, thread_id, character_id, sender, body,
                sender_character_id, provider, model, token_estimate,
                content_rating,
                delivery_status, delivery_error, delivery_job_id,
                delivery_attempt, in_world_sent_at, delivered_at, read_at,
                reply_to_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                save_id,
                thread_id,
                character_id,
                normalized_sender,
                body.strip(),
                resolved_sender_character_id,
                provider,
                model,
                token_estimate,
                content_rating,
                delivery_status,
                redact_text(delivery_error),
                delivery_job_id,
                max(0, int(delivery_attempt)),
                _blank_to_none(in_world_sent_at),
                resolved_delivered_at,
                read_at,
                normalized_reply_to_message_id,
            ),
        )
        self.connection.execute(
            """
            UPDATE character_text_threads
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (thread_id,),
        )
        if normalized_sender == "player" and delivery_status == "sent":
            self._record_character_text_activity(
                save_id=save_id,
                thread_id=thread_id,
                activity_type="player_sent",
                text_message_id=record_id,
                delivery_status="sent",
            )
        elif normalized_sender == "character" and delivery_status == "sent":
            self._record_character_text_activity(
                save_id=save_id,
                thread_id=thread_id,
                activity_type="character_received",
                text_message_id=record_id,
                delivery_status="sent",
            )
        self.commit()
        row = self._fetch_one(
            f"""
            SELECT {_CHARACTER_TEXT_MESSAGE_COLUMNS}
            FROM character_text_messages
            WHERE id = ?
            """,
            (record_id,),
        )
        if row is None:
            raise ValueError("Character text message was not saved")
        return _character_text_message_from_row(row)

    def get_character_text_message(
        self,
        *,
        save_id: str,
        message_id: str,
    ) -> CharacterTextMessageRecord | None:
        row = self._fetch_one(
            f"""
            SELECT {_CHARACTER_TEXT_MESSAGE_COLUMNS}
            FROM character_text_messages
            WHERE save_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (save_id, message_id),
        )
        return _character_text_message_from_row(row) if row is not None else None

    def mark_character_text_thread_read(
        self,
        *,
        save_id: str,
        thread_id: str,
        through_message_id: str | None = None,
    ) -> tuple[CharacterTextMessageRecord, ...]:
        thread = self.get_character_text_thread(thread_id=thread_id, save_id=save_id)
        if thread is None:
            raise ValueError(f"Unknown character text thread id: {thread_id}")
        boundary_rowid: int | None = None
        if through_message_id is not None:
            boundary = self._fetch_one(
                """
                SELECT rowid
                FROM character_text_messages
                WHERE save_id = ?
                  AND thread_id = ?
                  AND id = ?
                  AND deleted_at IS NULL
                """,
                (save_id, thread_id, through_message_id),
            )
            if boundary is None:
                raise ValueError(
                    f"Unknown character text message id: {through_message_id}"
                )
            boundary_rowid = int(boundary["rowid"])

        clauses = [
            "save_id = ?",
            "thread_id = ?",
            "deleted_at IS NULL",
            "sender = 'character'",
            "delivery_status = 'sent'",
            "read_at IS NULL",
        ]
        params: list[object] = [save_id, thread_id]
        if boundary_rowid is not None:
            clauses.append("rowid <= ?")
            params.append(boundary_rowid)
        rows = self._fetch_all(
            f"""
            SELECT id
            FROM character_text_messages
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at, rowid
            """,
            tuple(params),
        )
        message_ids = tuple(str(row["id"]) for row in rows)
        if not message_ids:
            self._record_character_text_activity(
                save_id=save_id,
                thread_id=thread_id,
                activity_type="thread_opened",
            )
            self.commit()
            return ()

        read_at = _timestamp_text(_utc_now())
        self.connection.execute(
            f"""
            UPDATE character_text_messages
            SET read_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND thread_id = ?
              AND id IN ({_placeholders(len(message_ids))})
              AND deleted_at IS NULL
              AND read_at IS NULL
            """,
            (read_at, save_id, thread_id, *message_ids),
        )
        self._record_character_text_activity(
            save_id=save_id,
            thread_id=thread_id,
            activity_type="thread_opened",
            read_count=len(message_ids),
        )
        self.commit()
        updated = self._fetch_all(
            f"""
            SELECT {_CHARACTER_TEXT_MESSAGE_COLUMNS}
            FROM character_text_messages
            WHERE save_id = ?
              AND thread_id = ?
              AND id IN ({_placeholders(len(message_ids))})
              AND deleted_at IS NULL
            ORDER BY created_at, rowid
            """,
            (save_id, thread_id, *message_ids),
        )
        return tuple(_character_text_message_from_row(row) for row in updated)

    def update_character_text_delivery(
        self,
        *,
        save_id: str,
        message_id: str,
        status: str,
        error: str | None = None,
        job_id: str | None | object = ...,
        attempt: int | None | object = ...,
        delivered_at: str | None | object = ...,
    ) -> CharacterTextMessageRecord:
        _validate_character_text_delivery_status(status)
        assignments = [
            "delivery_status = ?",
            "delivery_error = ?",
            "updated_at = CURRENT_TIMESTAMP",
        ]
        params: list[object] = [status, redact_text(error)]
        if job_id is not ...:
            assignments.append("delivery_job_id = ?")
            params.append(job_id)
        if attempt is not ...:
            assignments.append("delivery_attempt = ?")
            normalized_attempt = cast(int | None, attempt)
            params.append(max(0, int(normalized_attempt or 0)))
        if delivered_at is not ...:
            assignments.append("delivered_at = ?")
            params.append(delivered_at)
        elif status == "sent":
            assignments.append(
                "delivered_at = COALESCE(delivered_at, CURRENT_TIMESTAMP)"
            )
        params.extend([save_id, message_id])
        existing = self.get_character_text_message(
            save_id=save_id,
            message_id=message_id,
        )
        if existing is None:
            raise ValueError(f"Unknown character text message id: {message_id}")
        self.connection.execute(
            f"""
            UPDATE character_text_messages
            SET {', '.join(assignments)}
            WHERE save_id = ? AND id = ? AND deleted_at IS NULL
            """,
            tuple(params),
        )
        updated = self.get_character_text_message(
            save_id=save_id,
            message_id=message_id,
        )
        if updated is None:
            raise ValueError(f"Unknown character text message id: {message_id}")
        if status == "sent" and existing.delivery_status != "sent":
            self._record_character_text_activity(
                save_id=save_id,
                thread_id=updated.thread_id,
                activity_type=(
                    "player_sent"
                    if updated.sender == "player"
                    else "character_received"
                ),
                text_message_id=updated.id,
                delivery_status="sent",
            )
        self.commit()
        return updated

    def update_character_text_message_body(
        self,
        *,
        save_id: str,
        message_id: str,
        body: str,
        content_rating: str | None = None,
    ) -> CharacterTextMessageRecord:
        self.connection.execute(
            """
            UPDATE character_text_messages
            SET body = ?, content_rating = COALESCE(?, content_rating),
                updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (body.strip(), content_rating, save_id, message_id),
        )
        self.commit()
        updated = self.get_character_text_message(
            save_id=save_id,
            message_id=message_id,
        )
        if updated is None:
            raise ValueError(f"Unknown character text message id: {message_id}")
        return updated

    def complete_character_text_message_delivery(
        self,
        *,
        save_id: str,
        message_id: str,
        body: str,
        provider: str | None,
        model: str | None,
        token_estimate: int | None,
        in_world_sent_at: str | None = None,
        content_rating: str = "unclassified",
    ) -> CharacterTextMessageRecord:
        existing = self.get_character_text_message(
            save_id=save_id,
            message_id=message_id,
        )
        if existing is None:
            raise ValueError(f"Unknown character text message id: {message_id}")
        self.connection.execute(
            """
            UPDATE character_text_messages
            SET body = ?,
                provider = ?,
                model = ?,
                token_estimate = ?,
                content_rating = ?,
                delivery_status = 'sent',
                delivery_error = NULL,
                in_world_sent_at = COALESCE(?, in_world_sent_at),
                delivered_at = COALESCE(delivered_at, CURRENT_TIMESTAMP),
                updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (
                body.strip(),
                provider,
                model,
                token_estimate,
                content_rating,
                _blank_to_none(in_world_sent_at),
                save_id,
                message_id,
            ),
        )
        updated = self.get_character_text_message(
            save_id=save_id,
            message_id=message_id,
        )
        if updated is None:
            raise ValueError(f"Unknown character text message id: {message_id}")
        if existing.delivery_status != "sent":
            self._record_character_text_activity(
                save_id=save_id,
                thread_id=updated.thread_id,
                activity_type=(
                    "player_sent"
                    if updated.sender == "player"
                    else "character_received"
                ),
                text_message_id=updated.id,
                delivery_status="sent",
            )
        self.commit()
        return updated

    def _record_character_text_activity(
        self,
        *,
        save_id: str,
        thread_id: str,
        activity_type: str,
        text_message_id: str | None = None,
        read_count: int = 0,
        delivery_status: str = "",
    ) -> None:
        row = self._fetch_one(
            "SELECT COALESCE(MAX(ordinal), 0) + 1 AS next_ordinal "
            "FROM character_text_activity_events WHERE save_id = ?",
            (save_id,),
        )
        ordinal = int(row["next_ordinal"]) if row is not None else 1
        self.connection.execute(
            """
            INSERT INTO character_text_activity_events(
                id, save_id, ordinal, thread_id, activity_type,
                text_message_id, read_count, delivery_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _new_id(), save_id, ordinal, thread_id, activity_type,
                text_message_id, max(0, read_count), delivery_status,
            ),
        )

    def list_character_text_activity_events_after(
        self, *, save_id: str, ordinal: int, limit: int
    ) -> list[CharacterTextActivityEventRecord]:
        rows = self._fetch_all(
            """
            SELECT id, save_id, ordinal, thread_id, activity_type,
                   text_message_id, read_count, delivery_status, created_at
            FROM character_text_activity_events
            WHERE save_id = ? AND ordinal > ?
            ORDER BY ordinal
            LIMIT ?
            """,
            (save_id, max(0, ordinal), max(1, limit)),
        )
        return [_character_text_activity_event_from_row(row) for row in rows]

    def latest_character_text_activity_ordinal(self, *, save_id: str) -> int:
        row = self._fetch_one(
            "SELECT COALESCE(MAX(ordinal), 0) AS ordinal "
            "FROM character_text_activity_events WHERE save_id = ?",
            (save_id,),
        )
        return int(row["ordinal"]) if row is not None else 0

    def narrator_phone_activity_cursor(self, *, narrator_message_id: str) -> int | None:
        row = self._fetch_one(
            "SELECT last_activity_ordinal FROM narrator_phone_activity_cursors "
            "WHERE narrator_message_id = ?",
            (narrator_message_id,),
        )
        return int(row["last_activity_ordinal"]) if row is not None else None

    def set_narrator_phone_activity_cursor(
        self, *, save_id: str, narrator_message_id: str, last_activity_ordinal: int
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO narrator_phone_activity_cursors(
                narrator_message_id, save_id, last_activity_ordinal
            ) VALUES (?, ?, ?)
            ON CONFLICT(narrator_message_id) DO UPDATE SET
                save_id = excluded.save_id,
                last_activity_ordinal = excluded.last_activity_ordinal
            """,
            (narrator_message_id, save_id, max(0, last_activity_ordinal)),
        )
        self.commit()

    def add_character_text_message_revision(
        self,
        *,
        save_id: str,
        text_message_id: str,
        previous_body: str,
        new_body: str,
        diff_unified: str,
        reconciliation_status: str = "queued",
        reconciliation_error: str | None = None,
        revision_id: str | None = None,
        revision_number: int | None = None,
    ) -> CharacterTextMessageRevisionRecord:
        if reconciliation_status not in MESSAGE_REVISION_RECONCILIATION_STATUSES:
            raise ValueError(
                "Unsupported character text revision reconciliation status: "
                f"{reconciliation_status}"
            )
        if (
            self.get_character_text_message(
                save_id=save_id,
                message_id=text_message_id,
            )
            is None
        ):
            raise ValueError(f"Unknown character text message id: {text_message_id}")
        resolved_revision_number = revision_number
        if resolved_revision_number is None:
            row = self._fetch_one(
                """
                SELECT COALESCE(MAX(revision_number), 0) + 1 AS revision_number
                FROM character_text_message_revisions
                WHERE text_message_id = ?
                """,
                (text_message_id,),
            )
            resolved_revision_number = int(row["revision_number"] if row else 1)
        reconciled_at_expression = (
            "CURRENT_TIMESTAMP"
            if reconciliation_status in {"succeeded", "skipped", "failed"}
            else "NULL"
        )
        record_id = revision_id or _new_id()
        self.connection.execute(
            f"""
            INSERT INTO character_text_message_revisions(
                id, save_id, text_message_id, revision_number, previous_body,
                new_body, diff_unified, reconciliation_status,
                reconciliation_error, reconciled_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {reconciled_at_expression})
            """,
            (
                record_id,
                save_id,
                text_message_id,
                resolved_revision_number,
                previous_body,
                new_body,
                diff_unified,
                reconciliation_status,
                reconciliation_error,
            ),
        )
        self.commit()
        revision = self.get_character_text_message_revision(record_id)
        if revision is None:
            raise ValueError(f"Unknown character text revision id: {record_id}")
        return revision

    def get_character_text_message_revision(
        self,
        revision_id: str,
    ) -> CharacterTextMessageRevisionRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, text_message_id, revision_number, previous_body,
                   new_body, diff_unified, reconciliation_status,
                   reconciliation_error, created_at, reconciled_at
            FROM character_text_message_revisions
            WHERE id = ?
            """,
            (revision_id,),
        )
        return _character_text_message_revision_from_row(row) if row else None

    def list_character_text_message_revisions(
        self,
        *,
        save_id: str,
        text_message_id: str | None = None,
    ) -> list[CharacterTextMessageRevisionRecord]:
        message_filter = "" if text_message_id is None else "AND text_message_id = ?"
        params: tuple[Any, ...] = (
            (save_id,) if text_message_id is None else (save_id, text_message_id)
        )
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, text_message_id, revision_number, previous_body,
                   new_body, diff_unified, reconciliation_status,
                   reconciliation_error, created_at, reconciled_at
            FROM character_text_message_revisions
            WHERE save_id = ? {message_filter}
            ORDER BY text_message_id, revision_number, rowid
            """,
            params,
        )
        return [_character_text_message_revision_from_row(row) for row in rows]

    def character_text_message_revision_metadata(
        self,
        save_id: str,
    ) -> dict[str, MessageRevisionMetadataRecord]:
        rows = self._fetch_all(
            """
            SELECT text_message_id, COUNT(*) AS revision_count,
                   MAX(created_at) AS edited_at
            FROM character_text_message_revisions
            WHERE save_id = ?
            GROUP BY text_message_id
            """,
            (save_id,),
        )
        return {
            str(row["text_message_id"]): MessageRevisionMetadataRecord(
                message_id=str(row["text_message_id"]),
                revision_count=int(row["revision_count"]),
                edited_at=cast(str | None, row["edited_at"]),
            )
            for row in rows
        }

    def mark_character_text_message_revision_reconciled(
        self,
        revision_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> CharacterTextMessageRevisionRecord:
        if status not in MESSAGE_REVISION_RECONCILIATION_STATUSES:
            raise ValueError(
                "Unsupported character text revision reconciliation status: "
                f"{status}"
            )
        self.connection.execute(
            """
            UPDATE character_text_message_revisions
            SET reconciliation_status = ?,
                reconciliation_error = ?,
                reconciled_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, error, revision_id),
        )
        self.commit()
        revision = self.get_character_text_message_revision(revision_id)
        if revision is None:
            raise ValueError(f"Unknown character text revision id: {revision_id}")
        return revision

    def archive_character_text_messages_after(
        self,
        *,
        save_id: str,
        thread_id: str,
        message_id: str,
    ) -> tuple[CharacterTextMessageRecord, ...]:
        messages = self.list_character_text_messages(
            save_id=save_id,
            thread_id=thread_id,
        )
        anchor_index = next(
            (
                index
                for index, message in enumerate(messages)
                if message.id == message_id
            ),
            None,
        )
        if anchor_index is None:
            raise ValueError(f"Unknown character text message id: {message_id}")
        archived = tuple(messages[anchor_index + 1 :])
        if not archived:
            return ()
        message_ids = tuple(message.id for message in archived)
        self.connection.execute(
            f"""
            UPDATE character_text_messages
            SET deleted_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND thread_id = ?
              AND id IN ({_placeholders(len(message_ids))})
            """,
            (save_id, thread_id, *message_ids),
        )
        self.connection.execute(
            """
            UPDATE character_text_threads
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (thread_id,),
        )
        self.commit()
        return archived

    def archive_character_text_messages_from(
        self,
        *,
        save_id: str,
        thread_id: str,
        message_id: str,
    ) -> tuple[CharacterTextMessageRecord, ...]:
        messages = self.list_character_text_messages(
            save_id=save_id,
            thread_id=thread_id,
        )
        anchor_index = next(
            (
                index
                for index, message in enumerate(messages)
                if message.id == message_id
            ),
            None,
        )
        if anchor_index is None:
            raise ValueError(f"Unknown character text message id: {message_id}")
        archived = tuple(messages[anchor_index:])
        message_ids = tuple(message.id for message in archived)
        self.connection.execute(
            f"""
            UPDATE character_text_messages
            SET deleted_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND thread_id = ?
              AND id IN ({_placeholders(len(message_ids))})
            """,
            (save_id, thread_id, *message_ids),
        )
        self.connection.execute(
            """
            UPDATE character_text_threads
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (thread_id,),
        )
        self.commit()
        return archived

    def restore_character_text_messages(
        self,
        message_ids: set[str] | frozenset[str],
    ) -> None:
        if not message_ids:
            return
        self.connection.execute(
            f"""
            UPDATE character_text_messages
            SET deleted_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(message_ids))})
            """,
            tuple(message_ids),
        )
        self.commit()

    def has_active_character_text_delivery(
        self,
        *,
        save_id: str,
        thread_id: str,
    ) -> bool:
        row = self._fetch_one(
            """
            SELECT 1
            FROM character_text_messages
            WHERE save_id = ? AND thread_id = ?
              AND delivery_status IN ('pending', 'retrying')
              AND deleted_at IS NULL
            LIMIT 1
            """,
            (save_id, thread_id),
        )
        return row is not None

    def recover_interrupted_character_text_deliveries(
        self,
        *,
        error: str,
    ) -> list[CharacterTextMessageRecord]:
        rows = self._fetch_all(
            f"""
            SELECT {_CHARACTER_TEXT_MESSAGE_COLUMNS}
            FROM character_text_messages
            WHERE delivery_status IN ('pending', 'retrying')
              AND deleted_at IS NULL
            ORDER BY created_at, rowid
            """,
            (),
        )
        message_ids = [str(row["id"]) for row in rows]
        if not message_ids:
            return []
        self.connection.execute(
            f"""
            UPDATE character_text_messages
            SET delivery_status = 'failed', delivery_error = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(message_ids))})
            """,
            (redact_text(error), *message_ids),
        )
        self.commit()
        recovered: list[CharacterTextMessageRecord] = []
        for row in rows:
            updated = self.get_character_text_message(
                save_id=str(row["save_id"]),
                message_id=str(row["id"]),
            )
            if updated is not None:
                recovered.append(updated)
        return recovered

    def list_character_text_messages(
        self,
        *,
        save_id: str,
        thread_id: str | None = None,
    ) -> list[CharacterTextMessageRecord]:
        clauses = ["save_id = ?", "deleted_at IS NULL"]
        params: list[object] = [save_id]
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        rows = self._fetch_all(
            f"""
            SELECT {_CHARACTER_TEXT_MESSAGE_COLUMNS}
            FROM character_text_messages
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at, rowid
            """,
            tuple(params),
        )
        return [_character_text_message_from_row(row) for row in rows]

    def list_recent_sent_character_text_messages(
        self,
        *,
        save_id: str,
        thread_id: str,
        limit: int,
    ) -> list[CharacterTextMessageRecord]:
        if limit <= 0:
            return []
        rows = self._fetch_all(
            f"""
            SELECT {_CHARACTER_TEXT_MESSAGE_COLUMNS}
            FROM character_text_messages
            WHERE save_id = ?
              AND thread_id = ?
              AND deleted_at IS NULL
              AND delivery_status = 'sent'
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (save_id, thread_id, limit),
        )
        rows.reverse()
        return [_character_text_message_from_row(row) for row in rows]
    def add_character_text_message_attachment(
        self,
        *,
        save_id: str,
        thread_id: str,
        text_message_id: str,
        character_id: str,
        kind: str,
        status: str,
        media_asset_id: str | None = None,
        prompt: str = "",
        error: str | None = None,
        metadata: dict[str, object] | None = None,
        attachment_id: str | None = None,
        ordinal: int | None = None,
    ) -> CharacterTextMessageAttachmentRecord:
        _validate_character_text_attachment_kind(kind)
        _validate_character_text_attachment_status(status)
        if status == "succeeded" and media_asset_id is None:
            raise ValueError("Succeeded text attachment requires a media asset")
        if status == "failed" and media_asset_id is not None:
            raise ValueError("Failed text attachment must not reference media")
        message = self._fetch_one(
            """
            SELECT id, character_id, sender_character_id
            FROM character_text_messages
            WHERE id = ? AND save_id = ? AND thread_id = ?
              AND deleted_at IS NULL
            """,
            (text_message_id, save_id, thread_id),
        )
        if message is None:
            raise ValueError(f"Unknown character text message id: {text_message_id}")
        if character_id not in {
            message["character_id"],
            message["sender_character_id"],
        }:
            raise ValueError(
                "Character text attachment character does not match message"
            )
        if media_asset_id is not None:
            media = self._fetch_one(
                """
                SELECT id
                FROM media_assets
                WHERE id = ? AND save_id = ? AND archived_at IS NULL
                """,
                (media_asset_id, save_id),
            )
            if media is None:
                raise ValueError(f"Unknown media asset id: {media_asset_id}")
        resolved_ordinal = ordinal
        if resolved_ordinal is None:
            row = self._fetch_one(
                """
                SELECT COALESCE(MAX(ordinal), -1) + 1 AS ordinal
                FROM character_text_message_attachments
                WHERE text_message_id = ?
                """,
                (text_message_id,),
            )
            resolved_ordinal = int(row["ordinal"] if row else 0)
        record_id = attachment_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO character_text_message_attachments(
                id, save_id, thread_id, text_message_id, character_id, ordinal,
                kind, status, media_asset_id, prompt, error, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                save_id,
                thread_id,
                text_message_id,
                character_id,
                max(0, int(resolved_ordinal)),
                kind,
                status,
                media_asset_id,
                prompt.strip(),
                redact_text(error),
                _dump_json(metadata or {}),
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, thread_id, text_message_id, character_id,
                   ordinal, kind, status, media_asset_id, prompt, error,
                   metadata_json, created_at, updated_at
            FROM character_text_message_attachments
            WHERE id = ?
            """,
            (record_id,),
        )
        if row is None:
            raise ValueError("Character text attachment was not saved")
        return _character_text_message_attachment_from_row(row)

    def list_character_text_message_attachments(
        self,
        *,
        save_id: str,
        text_message_ids: tuple[str, ...] | list[str] | None = None,
    ) -> list[CharacterTextMessageAttachmentRecord]:
        clauses = ["save_id = ?"]
        params: list[object] = [save_id]
        if text_message_ids is not None:
            if not text_message_ids:
                return []
            placeholders = _placeholders(len(text_message_ids))
            clauses.append(f"text_message_id IN ({placeholders})")
            params.extend(text_message_ids)
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, thread_id, text_message_id, character_id,
                   ordinal, kind, status, media_asset_id, prompt, error,
                   metadata_json, created_at, updated_at
            FROM character_text_message_attachments
            WHERE {' AND '.join(clauses)}
            ORDER BY text_message_id, ordinal, created_at, rowid
            """,
            tuple(params),
        )
        return [_character_text_message_attachment_from_row(row) for row in rows]

    def replace_character_text_attachment_media_asset(
        self,
        *,
        save_id: str,
        old_media_asset_id: str,
        new_media_asset_id: str,
    ) -> int:
        media = self._fetch_one(
            """
            SELECT id
            FROM media_assets
            WHERE id = ? AND save_id = ? AND archived_at IS NULL
            """,
            (new_media_asset_id, save_id),
        )
        if media is None:
            raise ValueError(f"Unknown media asset id: {new_media_asset_id}")

        rows = self._fetch_all(
            """
            SELECT id, metadata_json
            FROM character_text_message_attachments
            WHERE save_id = ? AND media_asset_id = ?
            """,
            (save_id, old_media_asset_id),
        )
        for row in rows:
            metadata_json = row["metadata_json"]
            try:
                metadata = _load_object(metadata_json or "{}")
            except (TypeError, ValueError):
                metadata = {}
            if metadata.get("media_asset_id") == old_media_asset_id:
                metadata["media_asset_id"] = new_media_asset_id
            self.connection.execute(
                """
                UPDATE character_text_message_attachments
                SET media_asset_id = ?,
                    metadata_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (new_media_asset_id, _dump_json(metadata), row["id"]),
            )
        self.commit()
        return len(rows)

    def add_character_text_provenance(
        self,
        *,
        save_id: str,
        thread_id: str,
        text_message_id: str,
        target_type: str,
        target_id: str,
        operation: str = "",
        field_path: str = "",
        provenance_id: str | None = None,
    ) -> CharacterTextProvenanceRecord:
        thread = self.get_character_text_thread(thread_id=thread_id, save_id=save_id)
        if thread is None:
            raise ValueError(f"Unknown character text thread id: {thread_id}")
        message = self._fetch_one(
            """
            SELECT id
            FROM character_text_messages
            WHERE id = ? AND save_id = ? AND thread_id = ? AND deleted_at IS NULL
            """,
            (text_message_id, save_id, thread_id),
        )
        if message is None:
            raise ValueError(f"Unknown character text message id: {text_message_id}")
        record_id = provenance_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO character_text_provenance(
                id, save_id, thread_id, text_message_id, target_type, target_id,
                operation, field_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                save_id,
                thread_id,
                text_message_id,
                target_type.strip(),
                target_id.strip(),
                operation.strip(),
                field_path.strip(),
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, thread_id, text_message_id, target_type, target_id,
                   operation, field_path, created_at
            FROM character_text_provenance
            WHERE id = ?
            """,
            (record_id,),
        )
        if row is None:
            raise ValueError("Character text provenance was not saved")
        return _character_text_provenance_from_row(row)

    def list_character_text_provenance(
        self,
        *,
        save_id: str,
        thread_id: str | None = None,
        text_message_id: str | None = None,
    ) -> list[CharacterTextProvenanceRecord]:
        clauses = ["save_id = ?"]
        params: list[object] = [save_id]
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if text_message_id is not None:
            clauses.append("text_message_id = ?")
            params.append(text_message_id)
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, thread_id, text_message_id, target_type, target_id,
                   operation, field_path, created_at
            FROM character_text_provenance
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at, rowid
            """,
            tuple(params),
        )
        return [_character_text_provenance_from_row(row) for row in rows]

    def add_character_text_proactive_trigger(
        self,
        *,
        save_id: str,
        character_id: str,
        trigger_key: str,
        trigger_type: str,
        thread_id: str | None = None,
        text_message_id: str | None = None,
        source_type: str = "",
        source_id: str = "",
        source_message_id: str | None = None,
        reason: str = "",
        trigger_id: str | None = None,
    ) -> CharacterTextProactiveTriggerRecord:
        self._validate_character_reference(
            save_id=save_id,
            character_id=character_id,
            field_name="character_id",
        )
        normalized_trigger_key = trigger_key.strip()
        if not normalized_trigger_key:
            raise ValueError("character text proactive trigger key is required")
        if thread_id is not None:
            thread = self.get_character_text_thread(
                thread_id=thread_id,
                save_id=save_id,
            )
            if thread is None:
                raise ValueError(f"Unknown character text thread id: {thread_id}")
            if thread.character_id != character_id:
                raise ValueError(
                    "character text trigger character does not match thread"
                )
        if text_message_id is not None:
            message = self._fetch_one(
                """
                SELECT thread_id, character_id
                FROM character_text_messages
                WHERE id = ? AND save_id = ? AND deleted_at IS NULL
                """,
                (text_message_id, save_id),
            )
            if message is None:
                raise ValueError(
                    f"Unknown character text message id: {text_message_id}"
                )
            if message["character_id"] != character_id:
                raise ValueError(
                    "character text trigger character does not match text message"
                )
            if thread_id is not None and message["thread_id"] != thread_id:
                raise ValueError(
                    "character text trigger thread does not match text message"
                )
        if source_message_id is not None:
            source = self._fetch_one(
                """
                SELECT 1
                FROM messages
                WHERE id = ? AND save_id = ? AND deleted_at IS NULL
                """,
                (source_message_id, save_id),
            )
            if source is None:
                raise ValueError(
                    "source_message_id must reference a message in the same save"
                )
        record_id = trigger_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO character_text_proactive_triggers(
                id, save_id, character_id, trigger_key, trigger_type, thread_id,
                text_message_id, source_type, source_id, source_message_id, reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(save_id, character_id, trigger_key) DO UPDATE SET
                thread_id = COALESCE(
                    excluded.thread_id,
                    character_text_proactive_triggers.thread_id
                ),
                text_message_id = COALESCE(
                    excluded.text_message_id,
                    character_text_proactive_triggers.text_message_id
                ),
                source_type = COALESCE(
                    NULLIF(excluded.source_type, ''),
                    NULLIF(character_text_proactive_triggers.source_type, ''),
                    ''
                ),
                source_id = COALESCE(
                    NULLIF(excluded.source_id, ''),
                    NULLIF(character_text_proactive_triggers.source_id, ''),
                    ''
                ),
                source_message_id = COALESCE(
                    excluded.source_message_id,
                    character_text_proactive_triggers.source_message_id
                ),
                reason = COALESCE(
                    NULLIF(excluded.reason, ''),
                    NULLIF(character_text_proactive_triggers.reason, ''),
                    ''
                ),
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                record_id,
                save_id,
                character_id,
                normalized_trigger_key,
                trigger_type.strip(),
                thread_id,
                text_message_id,
                source_type.strip(),
                source_id.strip(),
                source_message_id,
                reason.strip(),
            ),
        )
        self.commit()
        saved = self.get_character_text_proactive_trigger(
            save_id=save_id,
            character_id=character_id,
            trigger_key=normalized_trigger_key,
        )
        if saved is None:
            raise ValueError("Character text proactive trigger was not saved")
        return saved

    def get_character_text_proactive_trigger(
        self,
        *,
        save_id: str,
        character_id: str,
        trigger_key: str,
    ) -> CharacterTextProactiveTriggerRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, character_id, trigger_key, trigger_type, thread_id,
                   text_message_id, source_type, source_id, source_message_id,
                   reason, created_at, updated_at
            FROM character_text_proactive_triggers
            WHERE save_id = ? AND character_id = ? AND trigger_key = ?
            """,
            (save_id, character_id, trigger_key.strip()),
        )
        return (
            _character_text_proactive_trigger_from_row(row)
            if row is not None
            else None
        )

    def list_character_text_proactive_triggers(
        self,
        save_id: str,
    ) -> list[CharacterTextProactiveTriggerRecord]:
        rows = self._fetch_all(
            """
            SELECT id, save_id, character_id, trigger_key, trigger_type, thread_id,
                   text_message_id, source_type, source_id, source_message_id,
                   reason, created_at, updated_at
            FROM character_text_proactive_triggers
            WHERE save_id = ?
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )
        return [_character_text_proactive_trigger_from_row(row) for row in rows]

    def delete_character_text_proactive_triggers_for_messages(
        self,
        *,
        save_id: str,
        text_message_ids: frozenset[str] | set[str],
    ) -> frozenset[str]:
        if not text_message_ids:
            return frozenset()
        ordered_ids = sorted(text_message_ids)
        placeholders = _placeholders(len(ordered_ids))
        rows = self._fetch_all(
            f"""
            SELECT id
            FROM character_text_proactive_triggers
            WHERE save_id = ? AND text_message_id IN ({placeholders})
            ORDER BY created_at, rowid
            """,
            (save_id, *ordered_ids),
        )
        trigger_ids = frozenset(str(row["id"]) for row in rows)
        if not trigger_ids:
            return frozenset()
        self.connection.execute(
            f"""
            DELETE FROM character_text_proactive_triggers
            WHERE id IN ({_placeholders(len(trigger_ids))})
            """,
            tuple(sorted(trigger_ids)),
        )
        self.commit()
        return trigger_ids

    def archive_character(self, character_id: str) -> None:
        self.connection.execute(
            """
            UPDATE characters
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (character_id,),
        )
        self.commit()

    def restore_characters(self, character_ids: set[str] | frozenset[str]) -> None:
        if not character_ids:
            return
        self.connection.execute(
            f"""
            UPDATE characters
            SET archived_at = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(character_ids))})
            """,
            tuple(character_ids),
        )
        self.commit()

    def upsert_dating_route_state(
        self,
        *,
        save_id: str,
        player_character_id: str,
        npc_character_id: str,
        stage: str | None = None,
        first_met_message_id: str | None = None,
        first_met_world_day_index: int | None = None,
        last_interaction_message_id: str | None = None,
        last_interaction_world_day_index: int | None = None,
        completed_interactions: int | None = None,
        dates_completed: int | None = None,
        interest_level: str | None = None,
        trust_level: str | None = None,
        comfort_with_intimacy: str | None = None,
        pacing_preference: str | None = None,
        known_boundaries: list[str] | None = None,
        unresolved_questions: list[str] | None = None,
        next_reasonable_step: str | None = None,
        source_message_id: str | None = None,
        route_id: str | None = None,
    ) -> DatingRouteStateRecord:
        self._validate_character_reference(
            save_id=save_id,
            character_id=player_character_id,
            field_name="player_character_id",
        )
        self._validate_character_reference(
            save_id=save_id,
            character_id=npc_character_id,
            field_name="npc_character_id",
        )
        if player_character_id == npc_character_id:
            raise ValueError("dating route player and NPC must be different")
        existing = self.get_dating_route_state_for_pair(
            save_id,
            player_character_id,
            npc_character_id,
            include_archived=True,
        )
        normalized_stage = stage or (existing.stage if existing else "unmet")
        _validate_dating_route_stage(normalized_stage)
        record = DatingRouteStateRecord(
            id=existing.id if existing is not None else route_id or _new_id(),
            save_id=save_id,
            player_character_id=player_character_id,
            npc_character_id=npc_character_id,
            stage=normalized_stage,
            first_met_message_id=(
                first_met_message_id
                if first_met_message_id is not None
                else existing.first_met_message_id if existing else None
            ),
            first_met_world_day_index=(
                first_met_world_day_index
                if first_met_world_day_index is not None
                else existing.first_met_world_day_index if existing else None
            ),
            last_interaction_message_id=(
                last_interaction_message_id
                if last_interaction_message_id is not None
                else existing.last_interaction_message_id if existing else None
            ),
            last_interaction_world_day_index=(
                last_interaction_world_day_index
                if last_interaction_world_day_index is not None
                else existing.last_interaction_world_day_index if existing else None
            ),
            completed_interactions=max(
                0,
                completed_interactions
                if completed_interactions is not None
                else existing.completed_interactions if existing else 0,
            ),
            dates_completed=max(
                0,
                dates_completed
                if dates_completed is not None
                else existing.dates_completed if existing else 0,
            ),
            interest_level=(
                interest_level.strip()
                if interest_level is not None
                else existing.interest_level if existing else ""
            ),
            trust_level=(
                trust_level.strip()
                if trust_level is not None
                else existing.trust_level if existing else ""
            ),
            comfort_with_intimacy=(
                comfort_with_intimacy.strip()
                if comfort_with_intimacy is not None
                else existing.comfort_with_intimacy if existing else ""
            ),
            pacing_preference=(
                pacing_preference.strip()
                if pacing_preference is not None
                else existing.pacing_preference if existing else ""
            ),
            known_boundaries=(
                _string_list(known_boundaries)
                if known_boundaries is not None
                else list(existing.known_boundaries) if existing else []
            ),
            unresolved_questions=(
                _string_list(unresolved_questions)
                if unresolved_questions is not None
                else list(existing.unresolved_questions) if existing else []
            ),
            next_reasonable_step=(
                next_reasonable_step.strip()
                if next_reasonable_step is not None
                else existing.next_reasonable_step if existing else ""
            ),
            source_message_id=(
                source_message_id
                if source_message_id is not None
                else existing.source_message_id if existing else None
            ),
            created_at=existing.created_at if existing else None,
            updated_at=existing.updated_at if existing else None,
        )
        self.connection.execute(
            """
            INSERT INTO dating_route_states(
                id, save_id, player_character_id, npc_character_id, stage,
                first_met_message_id, first_met_world_day_index,
                last_interaction_message_id, last_interaction_world_day_index,
                completed_interactions, dates_completed, interest_level,
                trust_level, comfort_with_intimacy, pacing_preference,
                known_boundaries_json, unresolved_questions_json,
                next_reasonable_step, source_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(save_id, player_character_id, npc_character_id)
            DO UPDATE SET
                stage = excluded.stage,
                first_met_message_id = excluded.first_met_message_id,
                first_met_world_day_index = excluded.first_met_world_day_index,
                last_interaction_message_id = excluded.last_interaction_message_id,
                last_interaction_world_day_index =
                    excluded.last_interaction_world_day_index,
                completed_interactions = excluded.completed_interactions,
                dates_completed = excluded.dates_completed,
                interest_level = excluded.interest_level,
                trust_level = excluded.trust_level,
                comfort_with_intimacy = excluded.comfort_with_intimacy,
                pacing_preference = excluded.pacing_preference,
                known_boundaries_json = excluded.known_boundaries_json,
                unresolved_questions_json = excluded.unresolved_questions_json,
                next_reasonable_step = excluded.next_reasonable_step,
                source_message_id = excluded.source_message_id,
                updated_at = CURRENT_TIMESTAMP,
                archived_at = NULL
            """,
            _dating_route_state_params(record),
        )
        self.commit()
        saved = self.get_dating_route_state_for_pair(
            save_id,
            player_character_id,
            npc_character_id,
        )
        if saved is None:
            raise ValueError("Dating route state was not saved")
        return saved

    def get_dating_route_state(
        self,
        route_id: str,
        *,
        include_archived: bool = False,
    ) -> DatingRouteStateRecord | None:
        archive_filter = "" if include_archived else "AND archived_at IS NULL"
        row = self._fetch_one(
            f"""
            SELECT {_DATING_ROUTE_STATE_COLUMNS}
            FROM dating_route_states
            WHERE id = ? {archive_filter}
            """,
            (route_id,),
        )
        return _dating_route_state_from_row(row) if row is not None else None

    def get_dating_route_state_for_pair(
        self,
        save_id: str,
        player_character_id: str,
        npc_character_id: str,
        *,
        include_archived: bool = False,
    ) -> DatingRouteStateRecord | None:
        archive_filter = "" if include_archived else "AND archived_at IS NULL"
        row = self._fetch_one(
            f"""
            SELECT {_DATING_ROUTE_STATE_COLUMNS}
            FROM dating_route_states
            WHERE save_id = ?
              AND player_character_id = ?
              AND npc_character_id = ?
              {archive_filter}
            """,
            (save_id, player_character_id, npc_character_id),
        )
        return _dating_route_state_from_row(row) if row is not None else None

    def list_dating_route_states(
        self,
        save_id: str,
        *,
        include_archived: bool = False,
    ) -> list[DatingRouteStateRecord]:
        archive_filter = "" if include_archived else "AND archived_at IS NULL"
        rows = self._fetch_all(
            f"""
            SELECT {_DATING_ROUTE_STATE_COLUMNS}
            FROM dating_route_states
            WHERE save_id = ? {archive_filter}
            ORDER BY updated_at, created_at, rowid
            """,
            (save_id,),
        )
        return [_dating_route_state_from_row(row) for row in rows]

    def archive_dating_route_state(self, route_id: str) -> None:
        self.connection.execute(
            """
            UPDATE dating_route_states
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (route_id,),
        )
        self.commit()

    def get_player_character_id(self, save_id: str) -> str | None:
        row = self._fetch_one(
            """
            SELECT id
            FROM characters
            WHERE save_id = ?
              AND archived_at IS NULL
              AND is_player_character = 1
            ORDER BY updated_at DESC, rowid DESC
            LIMIT 1
            """,
            (save_id,),
        )
        return str(row["id"]) if row else None

    def _player_character_id(self, save_id: str) -> str | None:
        return self.get_player_character_id(save_id)

    def _clear_other_player_characters(self, save_id: str, character_id: str) -> None:
        self.connection.execute(
            """
            UPDATE characters
            SET is_player_character = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND id != ?
              AND archived_at IS NULL
              AND is_player_character = 1
            """,
            (save_id, character_id),
        )

    def _ensure_player_character_present(
        self,
        save_id: str,
        character_id: str,
    ) -> None:
        row = self._fetch_one(
            """
            SELECT id, present_character_ids_json
            FROM scene_snapshots
            WHERE save_id = ?
            """,
            (save_id,),
        )
        if row is None:
            return
        present_ids = _present_character_ids_with_player_character(
            _load_list(row["present_character_ids_json"]),
            character_id,
        )
        self.connection.execute(
            """
            UPDATE scene_snapshots
            SET present_character_ids_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (_dump_json(present_ids), row["id"]),
        )
        self.commit()

    def add_active_thread(
        self,
        *,
        save_id: str,
        title: str,
        description: str = "",
        status: str = "active",
        priority: int = 0,
        visibility: str = "public",
        related_entities: list[str] | None = None,
        source_message_id: str | None = None,
        locked_fields: list[str] | None = None,
        thread_id: str | None = None,
        first_seen_message_id: str | None = None,
        last_updated_message_id: str | None = None,
    ) -> ActiveThreadRecord:
        record = ActiveThreadRecord(
            id=thread_id or _new_id(),
            save_id=save_id,
            title=title,
            description=description,
            status=status,
            priority=priority,
            visibility=visibility,
            related_entities=list(related_entities or []),
            source_message_id=source_message_id,
            locked_fields=list(locked_fields or []),
            first_seen_message_id=first_seen_message_id or source_message_id,
            last_updated_message_id=last_updated_message_id or source_message_id,
        )
        self.connection.execute(
            """
            INSERT INTO active_threads(
                id, save_id, title, description, status, priority, visibility,
                related_entities_json, source_message_id, locked_fields_json,
                first_seen_message_id, last_updated_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _active_thread_params(record),
        )
        self.commit()
        return record

    def update_active_thread(self, thread: ActiveThreadRecord) -> ActiveThreadRecord:
        self.connection.execute(
            """
            UPDATE active_threads
            SET title = ?, description = ?, status = ?, priority = ?,
                visibility = ?, related_entities_json = ?, source_message_id = ?,
                locked_fields_json = ?, first_seen_message_id = ?,
                last_updated_message_id = ?, updated_at = CURRENT_TIMESTAMP,
                archived_at = NULL
            WHERE id = ? AND save_id = ?
            """,
            (
                thread.title,
                thread.description,
                thread.status,
                thread.priority,
                thread.visibility,
                _dump_json(thread.related_entities),
                thread.source_message_id,
                _dump_json(thread.locked_fields),
                thread.first_seen_message_id or thread.source_message_id,
                thread.last_updated_message_id or thread.source_message_id,
                thread.id,
                thread.save_id,
            ),
        )
        self.commit()
        saved = self.get_active_thread(thread.id)
        if saved is None:
            raise ValueError(f"Unknown active thread id: {thread.id}")
        return saved

    def get_active_thread(self, thread_id: str) -> ActiveThreadRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, title, description, status, priority, visibility,
                   related_entities_json, source_message_id, locked_fields_json,
                   first_seen_message_id, last_updated_message_id
            FROM active_threads
            WHERE id = ? AND archived_at IS NULL
            """,
            (thread_id,),
        )
        return _active_thread_from_row(row) if row else None

    def list_active_threads(self, save_id: str) -> list[ActiveThreadRecord]:
        rows = self._fetch_all(
            """
            SELECT id, save_id, title, description, status, priority, visibility,
                   related_entities_json, source_message_id, locked_fields_json,
                   first_seen_message_id, last_updated_message_id
            FROM active_threads
            WHERE save_id = ? AND archived_at IS NULL
            ORDER BY priority DESC, created_at, rowid
            """,
            (save_id,),
        )
        return [_active_thread_from_row(row) for row in rows]

    def has_active_threads(self, save_id: str) -> bool:
        return (
            self._fetch_one(
                """
                SELECT 1
                FROM active_threads
                WHERE save_id = ? AND archived_at IS NULL
                LIMIT 1
                """,
                (save_id,),
            )
            is not None
        )

    def list_narration_active_threads(
        self,
        save_id: str,
        *,
        reference_character_ids: set[str] | frozenset[str],
        visibility_character_ids: set[str] | frozenset[str],
        limit: int,
    ) -> list[ActiveThreadRecord]:
        reference_ids_json = _dump_json(sorted(reference_character_ids))
        visibility_ids_json = _dump_json(sorted(visibility_character_ids))
        rows = self._fetch_all(
            """
            SELECT id, save_id, title, description, status, priority, visibility,
                   related_entities_json, source_message_id, locked_fields_json,
                   first_seen_message_id, last_updated_message_id
            FROM active_threads
            WHERE save_id = ?
              AND archived_at IS NULL
              AND lower(status) NOT IN ('closed', 'completed', 'resolved')
              AND lower(visibility) != 'hidden'
              AND (
                    lower(visibility) != 'private'
                    OR EXISTS (
                        SELECT 1
                        FROM json_each(related_entities_json) related
                        JOIN json_each(?) reference
                          ON CAST(related.value AS TEXT) =
                             CAST(reference.value AS TEXT)
                          OR CAST(related.value AS TEXT) =
                             'character:' || CAST(reference.value AS TEXT)
                    )
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM message_visibility hidden
                    JOIN json_each(?) present
                      ON hidden.character_id = CAST(present.value AS TEXT)
                    WHERE hidden.save_id = active_threads.save_id
                      AND hidden.visibility = 'not_visible'
                      AND hidden.message_id IN (
                            active_threads.source_message_id,
                            active_threads.first_seen_message_id,
                            active_threads.last_updated_message_id
                      )
              )
            ORDER BY priority DESC, updated_at DESC, rowid DESC
            LIMIT ?
            """,
            (
                save_id,
                reference_ids_json,
                visibility_ids_json,
                max(0, limit),
            ),
        )
        return [_active_thread_from_row(row) for row in rows]

    def archive_active_thread(self, thread_id: str) -> None:
        self.connection.execute(
            """
            UPDATE active_threads
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (thread_id,),
        )
        self.commit()

    def restore_active_threads(self, thread_ids: set[str] | frozenset[str]) -> None:
        if not thread_ids:
            return
        self.connection.execute(
            f"""
            UPDATE active_threads
            SET archived_at = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(thread_ids))})
            """,
            tuple(thread_ids),
        )
        self.commit()

    def add_entity_link(
        self,
        *,
        save_id: str,
        entity_type: str,
        entity_id: str,
        target_type: str,
        target_id: str,
        relation: str = "",
        source_message_id: str | None = None,
        overwrite_source: bool = False,
        link_id: str | None = None,
    ) -> EntityLinkRecord:
        record = EntityLinkRecord(
            id=link_id or _new_id(),
            save_id=save_id,
            entity_type=entity_type,
            entity_id=entity_id,
            target_type=target_type,
            target_id=target_id,
            relation=relation,
            source_message_id=source_message_id,
        )
        existing = self._fetch_one(
            """
            SELECT id, source_message_id
            FROM entity_links
            WHERE save_id = ? AND entity_type = ? AND entity_id = ?
              AND target_type = ? AND target_id = ? AND relation = ?
            """,
            (
                save_id,
                entity_type,
                entity_id,
                target_type,
                target_id,
                relation,
            ),
        )
        if existing is not None:
            if source_message_id is not None or overwrite_source:
                self.connection.execute(
                    """
                    UPDATE entity_links
                    SET source_message_id = ?
                    WHERE id = ?
                    """,
                    (source_message_id, existing["id"]),
                )
                self.commit()
            return EntityLinkRecord(
                id=existing["id"],
                save_id=save_id,
                entity_type=entity_type,
                entity_id=entity_id,
                target_type=target_type,
                target_id=target_id,
                relation=relation,
                source_message_id=(
                    source_message_id
                    if source_message_id is not None or overwrite_source
                    else existing["source_message_id"]
                ),
            )
        self.connection.execute(
            """
            INSERT INTO entity_links(
                id, save_id, entity_type, entity_id, target_type, target_id, relation,
                source_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.entity_type,
                record.entity_id,
                record.target_type,
                record.target_id,
                record.relation,
                record.source_message_id,
            ),
        )
        self.commit()
        return record

    def list_entity_links(self, save_id: str) -> list[EntityLinkRecord]:
        rows = self._fetch_all(
            """
            SELECT id, save_id, entity_type, entity_id, target_type, target_id,
                   relation, source_message_id
            FROM entity_links
            WHERE save_id = ?
            ORDER BY entity_type, entity_id, target_type, target_id, relation
            """,
            (save_id,),
        )
        return [EntityLinkRecord(**dict(row)) for row in rows]

    def list_narration_entity_links(
        self,
        save_id: str,
        *,
        target_keys: set[tuple[str, str]] | frozenset[tuple[str, str]],
        present_character_ids: set[str] | frozenset[str],
        visibility_character_ids: set[str] | frozenset[str],
    ) -> list[EntityLinkRecord]:
        if not target_keys:
            return []
        target_keys_json = _dump_json(
            sorted(
                [
                    [_normalized_graph_target_type(target_type), target_id]
                    for target_type, target_id in target_keys
                ]
            )
        )
        present_ids_json = _dump_json(
            sorted(present_character_ids)[:MAX_NARRATION_GRAPH_CHARACTER_IDS]
        )
        visibility_ids_json = _dump_json(
            sorted(visibility_character_ids)[:MAX_NARRATION_GRAPH_CHARACTER_IDS]
        )
        rows = self._fetch_all(
            """
            WITH target_keys(target_type, target_id) AS (
                SELECT
                    CAST(json_extract(value, '$[0]') AS TEXT),
                    CAST(json_extract(value, '$[1]') AS TEXT)
                FROM json_each(?)
            ),
            classified AS (
                SELECT links.*,
                       links.rowid AS source_rowid,
                       CASE
                           WHEN links.entity_type = 'character'
                            AND links.relation = 'knows'
                            AND links.entity_id IN (
                                SELECT CAST(value AS TEXT) FROM json_each(?)
                            )
                            AND NOT EXISTS (
                                SELECT 1
                                FROM message_visibility hidden
                                JOIN json_each(?) present
                                  ON hidden.character_id =
                                     CAST(present.value AS TEXT)
                                WHERE hidden.save_id = links.save_id
                                  AND hidden.message_id = links.source_message_id
                                  AND hidden.visibility = 'not_visible'
                           )
                           THEN 0
                           WHEN links.entity_type = 'character'
                            AND links.relation = 'knows'
                           THEN 1
                           ELSE 2
                       END AS scope_class
                FROM entity_links links
                JOIN target_keys targets
                  ON targets.target_type =
                     CASE
                         WHEN lower(links.target_type) IN ('state', 'world_state')
                         THEN 'world_state'
                         ELSE lower(links.target_type)
                     END
                 AND targets.target_id = links.target_id
                WHERE links.save_id = ?
                  AND (
                      links.entity_type != 'character'
                      OR links.relation != 'knows'
                      OR NOT EXISTS (
                        SELECT 1
                        FROM character_knowledge_edges edge
                        WHERE edge.save_id = links.save_id
                          AND edge.archived_at IS NULL
                          AND edge.character_id = links.entity_id
                          AND (
                              CASE
                                  WHEN lower(edge.target_type)
                                       IN ('state', 'world_state')
                                  THEN 'world_state'
                                  ELSE lower(edge.target_type)
                              END
                          ) = (
                              CASE
                                  WHEN lower(links.target_type)
                                       IN ('state', 'world_state')
                                  THEN 'world_state'
                                  ELSE lower(links.target_type)
                              END
                          )
                          AND edge.target_id = links.target_id
                      )
                  )
            ),
            ranked AS (
                SELECT classified.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY
                               CASE
                                   WHEN lower(target_type)
                                        IN ('state', 'world_state')
                                   THEN 'world_state'
                                   ELSE lower(target_type)
                               END,
                               target_id,
                               scope_class,
                               CASE
                                   WHEN scope_class IN (0, 2)
                                   THEN entity_type || ':' || entity_id || ':' ||
                                        relation
                                   ELSE ''
                               END
                           ORDER BY source_rowid DESC
                       ) AS scope_rank,
                       ROW_NUMBER() OVER (
                           PARTITION BY
                               CASE
                                   WHEN lower(target_type)
                                        IN ('state', 'world_state')
                                   THEN 'world_state'
                                   ELSE lower(target_type)
                               END,
                               target_id,
                               scope_class
                           ORDER BY entity_id, source_rowid DESC
                       ) AS owner_rank
                FROM classified
            )
            SELECT id, save_id, entity_type, entity_id, target_type, target_id,
                   relation, source_message_id
            FROM ranked
            WHERE scope_rank = 1
              AND (scope_class = 1 OR owner_rank <= 8)
            ORDER BY target_type, target_id, scope_class
            """,
            (
                target_keys_json,
                present_ids_json,
                visibility_ids_json,
                save_id,
            ),
        )
        return [EntityLinkRecord(**dict(row)) for row in rows]

    def delete_entity_link(self, link_id: str) -> None:
        self.connection.execute("DELETE FROM entity_links WHERE id = ?", (link_id,))
        self.commit()

    def restore_entity_links(self, links: tuple[EntityLinkRecord, ...]) -> None:
        for link in links:
            self.add_entity_link(
                save_id=link.save_id,
                entity_type=link.entity_type,
                entity_id=link.entity_id,
                target_type=link.target_type,
                target_id=link.target_id,
                relation=link.relation,
                source_message_id=link.source_message_id,
                overwrite_source=True,
                link_id=link.id,
            )

    def add_character_knowledge_edge(
        self,
        *,
        save_id: str,
        character_id: str,
        target_type: str,
        target_id: str,
        knowledge_state: str = "knows",
        acquisition_method: str = "unknown",
        confidence: float = 1.0,
        source_message_id: str | None = None,
        source_message_ids: list[str] | tuple[str, ...] | None = None,
        evidence_quote: str = "",
        edge_id: str | None = None,
    ) -> CharacterKnowledgeEdgeRecord:
        if knowledge_state not in CHARACTER_KNOWLEDGE_STATES:
            raise ValueError(f"Unknown character knowledge state: {knowledge_state}")
        if acquisition_method not in CHARACTER_KNOWLEDGE_ACQUISITION_METHODS:
            raise ValueError(
                f"Unknown character knowledge acquisition method: {acquisition_method}"
            )
        source_ids = list(source_message_ids or ())
        if source_message_id is not None and source_message_id not in source_ids:
            source_ids.insert(0, source_message_id)
        source_ids = list(dict.fromkeys(source_ids))
        if len(source_ids) > MAX_KNOWLEDGE_EDGE_SOURCE_MESSAGE_IDS:
            raise ValueError("knowledge edge provenance is too large")
        record_id = edge_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO character_knowledge_edges(
                id,
                save_id,
                character_id,
                target_type,
                target_id,
                knowledge_state,
                acquisition_method,
                confidence,
                source_message_id,
                source_message_ids_json,
                evidence_quote
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(save_id, character_id, target_type, target_id)
            DO UPDATE SET
                knowledge_state = excluded.knowledge_state,
                acquisition_method = excluded.acquisition_method,
                confidence = excluded.confidence,
                source_message_id = excluded.source_message_id,
                source_message_ids_json = excluded.source_message_ids_json,
                evidence_quote = excluded.evidence_quote,
                updated_at = CURRENT_TIMESTAMP,
                archived_at = NULL
            """,
            (
                record_id,
                save_id,
                character_id,
                target_type,
                target_id,
                knowledge_state,
                acquisition_method,
                confidence,
                source_message_id,
                _dump_json(source_ids),
                evidence_quote.strip(),
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, character_id, target_type, target_id,
                   knowledge_state, acquisition_method, confidence,
                   source_message_id, source_message_ids_json, evidence_quote,
                   created_at, updated_at, archived_at
            FROM character_knowledge_edges
            WHERE save_id = ? AND character_id = ? AND target_type = ?
              AND target_id = ?
            """,
            (save_id, character_id, target_type, target_id),
        )
        if row is None:
            raise ValueError("Failed to persist character knowledge edge")
        return _character_knowledge_edge_from_row(row)

    def list_character_knowledge_edges(
        self,
        save_id: str,
        *,
        character_ids: set[str] | frozenset[str] | tuple[str, ...] | None = None,
        include_archived: bool = False,
    ) -> list[CharacterKnowledgeEdgeRecord]:
        filters = ["save_id = ?"]
        params: list[object] = [save_id]
        if not include_archived:
            filters.append("archived_at IS NULL")
        if character_ids is not None:
            ids = tuple(character_ids)
            if not ids:
                return []
            filters.append(f"character_id IN ({_placeholders(len(ids))})")
            params.extend(ids)
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, character_id, target_type, target_id,
                   knowledge_state, acquisition_method, confidence,
                   source_message_id, source_message_ids_json, evidence_quote,
                   created_at, updated_at, archived_at
            FROM character_knowledge_edges
            WHERE {' AND '.join(filters)}
            ORDER BY character_id, target_type, target_id, created_at, rowid
            """,
            tuple(params),
        )
        return [_character_knowledge_edge_from_row(row) for row in rows]

    def list_narration_character_knowledge_edges(
        self,
        save_id: str,
        *,
        target_keys: set[tuple[str, str]] | frozenset[tuple[str, str]],
        present_character_ids: set[str] | frozenset[str],
        visibility_character_ids: set[str] | frozenset[str],
    ) -> list[CharacterKnowledgeEdgeRecord]:
        if not target_keys:
            return []
        target_keys_json = _dump_json(
            sorted(
                [
                    [_normalized_graph_target_type(target_type), target_id]
                    for target_type, target_id in target_keys
                ]
            )
        )
        present_ids_json = _dump_json(
            sorted(present_character_ids)[:MAX_NARRATION_GRAPH_CHARACTER_IDS]
        )
        visibility_ids_json = _dump_json(
            sorted(visibility_character_ids)[:MAX_NARRATION_GRAPH_CHARACTER_IDS]
        )
        rows = self._fetch_all(
            f"""
            WITH target_keys(target_type, target_id) AS (
                SELECT
                    CAST(json_extract(value, '$[0]') AS TEXT),
                    CAST(json_extract(value, '$[1]') AS TEXT)
                FROM json_each(?)
            ),
            classified AS (
                SELECT edges.*,
                       edges.rowid AS source_rowid,
                       CASE
                           WHEN edges.character_id IN (
                                SELECT CAST(value AS TEXT) FROM json_each(?)
                           )
                            AND (
                                edges.knowledge_state = 'knows'
                                OR (
                                    edges.knowledge_state = 'may_know'
                                    AND edges.confidence >=
                                        {SCOPED_MAY_KNOW_CONFIDENCE_THRESHOLD}
                                )
                            )
                            AND NOT EXISTS (
                                SELECT 1
                                FROM message_visibility hidden
                                JOIN json_each(?) present
                                  ON hidden.character_id =
                                     CAST(present.value AS TEXT)
                                WHERE hidden.save_id = edges.save_id
                                  AND hidden.visibility = 'not_visible'
                                  AND (
                                      hidden.message_id = edges.source_message_id
                                      OR hidden.message_id IN (
                                          SELECT CAST(value AS TEXT)
                                          FROM json_each(
                                              edges.source_message_ids_json
                                          )
                                      )
                                  )
                            )
                           THEN 0
                           ELSE 1
                       END AS scope_class
                FROM character_knowledge_edges edges
                JOIN target_keys targets
                  ON targets.target_type =
                     CASE
                         WHEN lower(edges.target_type)
                              IN ('state', 'world_state')
                         THEN 'world_state'
                         ELSE lower(edges.target_type)
                     END
                 AND targets.target_id = edges.target_id
                WHERE edges.save_id = ?
                  AND edges.archived_at IS NULL
            ),
            ranked AS (
                SELECT classified.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY
                               CASE
                                   WHEN lower(target_type)
                                        IN ('state', 'world_state')
                                   THEN 'world_state'
                                   ELSE lower(target_type)
                               END,
                               target_id,
                               scope_class,
                               CASE
                                   WHEN scope_class = 0
                                     OR character_id IN (
                                         SELECT CAST(value AS TEXT)
                                         FROM json_each(?)
                                     )
                                   THEN character_id
                                   ELSE ''
                               END
                           ORDER BY updated_at DESC, source_rowid DESC
                       ) AS scope_rank,
                       ROW_NUMBER() OVER (
                           PARTITION BY
                               CASE
                                   WHEN lower(target_type)
                                        IN ('state', 'world_state')
                                   THEN 'world_state'
                                   ELSE lower(target_type)
                               END,
                               target_id,
                               scope_class
                           ORDER BY character_id, updated_at DESC,
                                    source_rowid DESC
                       ) AS owner_rank
                FROM classified
            )
            SELECT id, save_id, character_id, target_type, target_id,
                   knowledge_state, acquisition_method, confidence,
                   source_message_id, source_message_ids_json, evidence_quote,
                   created_at, updated_at, archived_at
            FROM ranked
            WHERE scope_rank = 1
              AND (scope_class != 0 OR owner_rank <= 8)
            ORDER BY target_type, target_id, scope_class
            """,
            (
                target_keys_json,
                present_ids_json,
                visibility_ids_json,
                save_id,
                present_ids_json,
            ),
        )
        return [_character_knowledge_edge_from_row(row) for row in rows]

    def archive_character_knowledge_edge(self, edge_id: str) -> None:
        self.connection.execute(
            """
            UPDATE character_knowledge_edges
            SET archived_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (edge_id,),
        )
        self.commit()

    def restore_character_knowledge_edges(
        self,
        edge_ids: set[str] | frozenset[str],
    ) -> None:
        if not edge_ids:
            return
        self.connection.execute(
            f"""
            UPDATE character_knowledge_edges
            SET archived_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({_placeholders(len(edge_ids))})
            """,
            tuple(edge_ids),
        )
        self.commit()

    def add_message_visibility(
        self,
        *,
        save_id: str,
        message_id: str,
        character_id: str,
        visibility: str,
        confidence: float = 1.0,
        source: str = "unknown",
        evidence: str = "",
        visibility_id: str | None = None,
    ) -> MessageVisibilityRecord:
        if visibility not in MESSAGE_VISIBILITY_STATES:
            raise ValueError(f"Unknown message visibility state: {visibility}")
        record_id = visibility_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO message_visibility(
                id,
                save_id,
                message_id,
                character_id,
                visibility,
                confidence,
                source,
                evidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(save_id, message_id, character_id)
            DO UPDATE SET
                visibility = excluded.visibility,
                confidence = excluded.confidence,
                source = excluded.source,
                evidence = excluded.evidence,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                record_id,
                save_id,
                message_id,
                character_id,
                visibility,
                confidence,
                source.strip() or "unknown",
                evidence.strip(),
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, message_id, character_id, visibility, confidence,
                   source, evidence, created_at, updated_at
            FROM message_visibility
            WHERE save_id = ? AND message_id = ? AND character_id = ?
            """,
            (save_id, message_id, character_id),
        )
        if row is None:
            raise ValueError("Failed to persist message visibility")
        return _message_visibility_from_row(row)

    def list_message_visibility(
        self,
        save_id: str,
        *,
        message_id: str | None = None,
        character_ids: set[str] | frozenset[str] | tuple[str, ...] | None = None,
        message_ids: set[str] | frozenset[str] | tuple[str, ...] | None = None,
    ) -> list[MessageVisibilityRecord]:
        filters = ["save_id = ?"]
        params: list[object] = [save_id]
        if message_id is not None:
            filters.append("message_id = ?")
            params.append(message_id)
        if message_ids is not None:
            ids = tuple(sorted(set(message_ids)))
            if not ids:
                return []
            filters.append(
                "message_id IN ("
                "SELECT CAST(value AS TEXT) FROM json_each(?)"
                ")"
            )
            params.append(_dump_json(ids))
        if character_ids is not None:
            ids = tuple(character_ids)
            if not ids:
                return []
            filters.append(f"character_id IN ({_placeholders(len(ids))})")
            params.extend(ids)
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, message_id, character_id, visibility, confidence,
                   source, evidence, created_at, updated_at
            FROM message_visibility
            WHERE {' AND '.join(filters)}
            ORDER BY message_id, character_id, created_at, rowid
            """,
            tuple(params),
        )
        return [_message_visibility_from_row(row) for row in rows]

    def replace_message_scene_presence(
        self,
        save_id: str,
        message_id: str,
        character_ids: list[str] | tuple[str, ...] | set[str] | frozenset[str],
        source: str = "context_snapshot",
    ) -> list[MessageScenePresenceRecord]:
        normalized_ids = tuple(
            dict.fromkeys(
                character_id.strip()
                for character_id in character_ids
                if character_id.strip()
            )
        )
        self.begin_transaction()
        try:
            if normalized_ids:
                self.connection.execute(
                    f"""
                    DELETE FROM message_scene_presence
                    WHERE save_id = ?
                      AND message_id = ?
                      AND character_id NOT IN ({_placeholders(len(normalized_ids))})
                    """,
                    (save_id, message_id, *normalized_ids),
                )
            else:
                self.connection.execute(
                    """
                    DELETE FROM message_scene_presence
                    WHERE save_id = ? AND message_id = ?
                    """,
                    (save_id, message_id),
                )
            for character_id in normalized_ids:
                self.connection.execute(
                    """
                    INSERT INTO message_scene_presence(
                        id, save_id, message_id, character_id, source
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(save_id, message_id, character_id)
                    DO UPDATE SET
                        source = excluded.source,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        _new_id(),
                        save_id,
                        message_id,
                        character_id,
                        source.strip() or "context_snapshot",
                    ),
                )
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise
        return self.list_message_scene_presence(save_id, message_id=message_id)

    def list_message_scene_presence(
        self,
        save_id: str,
        *,
        message_id: str | None = None,
        character_ids: set[str] | frozenset[str] | tuple[str, ...] | None = None,
    ) -> list[MessageScenePresenceRecord]:
        filters = ["save_id = ?"]
        params: list[object] = [save_id]
        if message_id is not None:
            filters.append("message_id = ?")
            params.append(message_id)
        if character_ids is not None:
            ids = tuple(character_ids)
            if not ids:
                return []
            filters.append(f"character_id IN ({_placeholders(len(ids))})")
            params.extend(ids)
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, message_id, character_id, source,
                   created_at, updated_at
            FROM message_scene_presence
            WHERE {' AND '.join(filters)}
            ORDER BY message_id, character_id, created_at, rowid
            """,
            tuple(params),
        )
        return [_message_scene_presence_from_row(row) for row in rows]

    def delete_message_scene_presence(
        self,
        *,
        save_id: str,
        message_id: str,
    ) -> int:
        cursor = self.connection.execute(
            """
            DELETE FROM message_scene_presence
            WHERE save_id = ? AND message_id = ?
            """,
            (save_id, message_id),
        )
        self.commit()
        return cursor.rowcount

    def delete_message_scene_presence_for_messages(
        self,
        *,
        save_id: str,
        message_ids: set[str] | frozenset[str],
    ) -> int:
        if not message_ids:
            return 0
        cursor = self.connection.execute(
            f"""
            DELETE FROM message_scene_presence
            WHERE save_id = ?
              AND message_id IN ({_placeholders(len(message_ids))})
            """,
            (save_id, *tuple(message_ids)),
        )
        self.commit()
        return cursor.rowcount

    def replace_message_action_choices(
        self,
        *,
        save_id: str,
        message_id: str,
        choices: list[str] | tuple[str, ...],
        provider: str = "",
        model: str = "",
        content_ratings: list[str] | tuple[str, ...] | None = None,
    ) -> list[MessageActionChoiceRecord]:
        normalized_choices = tuple(
            choice.strip() for choice in choices if choice.strip()
        )
        self.begin_transaction()
        try:
            self.connection.execute(
                """
                DELETE FROM message_action_choices
                WHERE save_id = ? AND message_id = ?
                """,
                (save_id, message_id),
            )
            normalized_ratings = tuple(content_ratings or ())
            for ordinal, body in enumerate(normalized_choices, start=1):
                content_rating = (
                    normalized_ratings[ordinal - 1]
                    if ordinal <= len(normalized_ratings)
                    else "unclassified"
                )
                self.connection.execute(
                    """
                    INSERT INTO message_action_choices(
                        id, save_id, message_id, ordinal, body, provider, model,
                        content_rating, updated_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?,
                        strftime('%Y-%m-%d %H:%M:%f', 'now')
                    )
                    """,
                    (
                        _new_id(),
                        save_id,
                        message_id,
                        ordinal,
                        body,
                        provider,
                        model,
                        content_rating,
                    ),
                )
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise
        return self.list_message_action_choices(save_id, message_id=message_id)

    def list_message_action_choices(
        self,
        save_id: str,
        *,
        message_id: str | None = None,
    ) -> list[MessageActionChoiceRecord]:
        filters = ["save_id = ?"]
        params: list[object] = [save_id]
        if message_id is not None:
            filters.append("message_id = ?")
            params.append(message_id)
        rows = self._fetch_all(
            f"""
            SELECT c.id, c.save_id, c.message_id, c.ordinal, c.body, c.provider,
                   c.model, c.content_rating, c.created_at, c.updated_at
            FROM message_action_choices c
            JOIN messages m ON m.id = c.message_id
            WHERE {' AND '.join(f'c.{item}' for item in filters)}
            ORDER BY m.rowid, c.ordinal, c.rowid
            """,
            tuple(params),
        )
        return [_message_action_choice_from_row(row) for row in rows]

    def latest_message_action_choices(
        self,
        save_id: str,
    ) -> list[MessageActionChoiceRecord]:
        message_row = self._fetch_one(
            """
            SELECT m.id
            FROM messages m
            WHERE m.save_id = ?
              AND m.role = 'narrator'
              AND m.deleted_at IS NULL
              AND EXISTS (
                  SELECT 1
                  FROM message_action_choices c
                  WHERE c.save_id = m.save_id
                    AND c.message_id = m.id
              )
            ORDER BY m.rowid DESC
            LIMIT 1
            """,
            (save_id,),
        )
        if message_row is None:
            return []
        return self.list_message_action_choices(
            save_id,
            message_id=str(message_row["id"]),
        )

    def delete_entity_links_for_source_messages(
        self,
        *,
        save_id: str,
        source_message_ids: set[str] | frozenset[str],
    ) -> tuple[EntityLinkRecord, ...]:
        if not source_message_ids:
            return ()
        links = tuple(
            link
            for link in self.list_entity_links(save_id)
            if link.source_message_id in source_message_ids
        )
        if not links:
            return ()
        link_ids = tuple(link.id for link in links)
        self.connection.execute(
            f"""
            DELETE FROM entity_links
            WHERE id IN ({_placeholders(len(link_ids))})
            """,
            link_ids,
        )
        self.commit()
        return links

    def delete_entity_links_for_source_message(
        self,
        *,
        save_id: str,
        source_message_id: str,
    ) -> None:
        self.connection.execute(
            """
            DELETE FROM entity_links
            WHERE save_id = ? AND source_message_id = ?
            """,
            (save_id, source_message_id),
        )
        self.commit()

    def delete_entity_links_for_inactive_stored_endpoints(
        self,
        save_id: str,
    ) -> tuple[EntityLinkRecord, ...]:
        active_ids = {
            "location": {record.id for record in self.list_locations(save_id)},
            "character": {record.id for record in self.list_characters(save_id)},
            "active_thread": {
                record.id for record in self.list_active_threads(save_id)
            },
            "memory": {record.id for record in self.list_memories(save_id)},
            "summary": {record.id for record in self.list_summaries(save_id)},
            "world_state": {record.id for record in self.list_world_state(save_id)},
        }
        stale_links = tuple(
            link
            for link in self.list_entity_links(save_id)
            if (
                _entity_link_endpoint_is_inactive(
                    link.entity_type,
                    link.entity_id,
                    active_ids,
                )
                or _entity_link_endpoint_is_inactive(
                    link.target_type,
                    link.target_id,
                    active_ids,
                )
            )
        )
        if not stale_links:
            return ()
        link_ids = tuple(link.id for link in stale_links)
        self.connection.execute(
            f"""
            DELETE FROM entity_links
            WHERE id IN ({_placeholders(len(link_ids))})
            """,
            link_ids,
        )
        self.commit()
        return stale_links

    def delete_entity_links_for_entity(
        self,
        *,
        save_id: str,
        entity_type: str,
        entity_id: str,
    ) -> None:
        self.connection.execute(
            """
            DELETE FROM entity_links
            WHERE save_id = ? AND entity_type = ? AND entity_id = ?
            """,
            (save_id, entity_type, entity_id),
        )
        self.commit()

    def delete_entity_links_for_endpoint(
        self,
        *,
        save_id: str,
        entity_type: str,
        entity_id: str,
    ) -> None:
        self.connection.execute(
            """
            DELETE FROM entity_links
            WHERE save_id = ?
              AND (
                (entity_type = ? AND entity_id = ?)
                OR (target_type = ? AND target_id = ?)
              )
            """,
            (save_id, entity_type, entity_id, entity_type, entity_id),
        )
        self.commit()

    def add_context_update_suggestion(
        self,
        *,
        save_id: str,
        update_type: str,
        entity_type: str,
        field_path: str,
        proposed_value: object,
        reason: str = "",
        confidence: float = 0.0,
        source_message_ids: list[str] | None = None,
        entity_id: str | None = None,
        status: str = "pending",
        suggestion_id: str | None = None,
    ) -> ContextUpdateSuggestionRecord:
        record = ContextUpdateSuggestionRecord(
            id=suggestion_id or _new_id(),
            save_id=save_id,
            update_type=update_type,
            entity_type=entity_type,
            entity_id=entity_id,
            field_path=field_path,
            proposed_value=proposed_value,
            status=status,
            reason=reason,
            confidence=confidence,
            source_message_ids=list(source_message_ids or []),
        )
        self.connection.execute(
            """
            INSERT INTO context_update_suggestions(
                id, save_id, update_type, entity_type, entity_id, field_path,
                proposed_value_json, status, reason, confidence,
                source_message_ids_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.update_type,
                record.entity_type,
                record.entity_id,
                record.field_path,
                _dump_json(record.proposed_value),
                record.status,
                record.reason,
                record.confidence,
                _dump_json(record.source_message_ids),
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, update_type, entity_type, entity_id, field_path,
                   proposed_value_json, status, reason, confidence,
                   source_message_ids_json, created_at, resolved_at,
                   review_attempt_count, next_review_at, last_review_error
            FROM context_update_suggestions
            WHERE id = ?
            """,
            (record.id,),
        )
        if row is None:
            raise ValueError(f"Unknown context update suggestion id: {record.id}")
        return _context_update_suggestion_from_row(row)

    def list_context_update_suggestions(
        self,
        save_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[ContextUpdateSuggestionRecord]:
        status_filter = "" if status is None else "AND status = ?"
        params: list[object] = [save_id]
        if status is not None:
            params.append(status)
        order_sql = (
            "ORDER BY created_at DESC, rowid DESC"
            if limit is not None
            else "ORDER BY created_at, rowid"
        )
        limit_sql = "LIMIT ?" if limit is not None else ""
        if limit is not None:
            params.append(max(0, limit))
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, update_type, entity_type, entity_id, field_path,
                   proposed_value_json, status, reason, confidence,
                   source_message_ids_json, created_at, resolved_at,
                   review_attempt_count, next_review_at, last_review_error
            FROM context_update_suggestions
            WHERE save_id = ? {status_filter}
            {order_sql}
            {limit_sql}
            """,
            tuple(params),
        )
        if limit is not None:
            rows.reverse()
        return [_context_update_suggestion_from_row(row) for row in rows]

    def has_context_update_suggestions(
        self,
        save_id: str,
        *,
        status: str | None = "pending",
    ) -> bool:
        status_filter = "" if status is None else "AND status = ?"
        params: tuple[Any, ...] = (save_id,) if status is None else (save_id, status)
        row = self._fetch_one(
            f"""
            SELECT 1
            FROM context_update_suggestions
            WHERE save_id = ? {status_filter}
            LIMIT 1
            """,
            params,
        )
        return row is not None

    def has_due_context_update_suggestion_review(self, save_id: str) -> bool:
        row = self._fetch_one(
            """
            SELECT 1
            FROM context_update_suggestions
            WHERE save_id = ?
              AND status = 'pending'
              AND (next_review_at IS NULL OR next_review_at <= CURRENT_TIMESTAMP)
            LIMIT 1
            """,
            (save_id,),
        )
        return row is not None

    def list_save_ids_with_due_context_update_suggestion_reviews(
        self,
        *,
        limit: int = 10,
    ) -> tuple[str, ...]:
        rows = self._fetch_all(
            """
            SELECT DISTINCT save_id
            FROM context_update_suggestions
            WHERE status = 'pending'
              AND (next_review_at IS NULL OR next_review_at <= CURRENT_TIMESTAMP)
            ORDER BY save_id
            LIMIT ?
            """,
            (limit,),
        )
        return tuple(str(row["save_id"]) for row in rows)

    def has_world_context_retention_work(self, save_id: str) -> bool:
        stale_or_excess = self._fetch_one(
            """
            SELECT 1
            FROM context_update_suggestions
            WHERE save_id = ? AND status = 'pending'
            GROUP BY save_id
            HAVING SUM(
                CASE WHEN julianday(created_at) <= julianday('now', '-30 days')
                     THEN 1 ELSE 0 END
            ) > 0
               OR COUNT(*) > 200
            """,
            (save_id,),
        )
        if stale_or_excess is not None:
            return True
        audit_excess = self._fetch_one(
            """
            SELECT 1
            FROM context_update_audit AS audit
            WHERE audit.save_id = ?
              AND audit.rowid NOT IN (
                  SELECT rowid FROM context_update_audit
                  WHERE save_id = ?
                  ORDER BY created_at DESC, rowid DESC
                  LIMIT 500
              )
              AND (
                  audit.suggestion_id IS NULL
                  OR audit.suggestion_id NOT IN (
                      SELECT id FROM context_update_suggestions
                      WHERE save_id = ? AND status = 'pending'
                  )
              )
            LIMIT 1
            """,
            (save_id, save_id, save_id),
        )
        if audit_excess is not None:
            return True
        terminal_job_excess = self._fetch_one(
            """
            SELECT 1 FROM jobs
            WHERE save_id = ?
              AND type IN (
                  'character_registry_maintenance', 'context_cleanup',
                  'context_precompute', 'context_search', 'context_update',
                  'context_update_retry', 'context_update_retry_drain',
                  'guided_context_cleanup', 'memory_consolidation',
                  'observation_curation_drain',
                  'scenario_evolution', 'state_extraction_retry',
                  'state_extraction_retry_drain', 'state_pruning',
                  'web_maintenance_character_registry_maintenance',
                  'web_maintenance_memory_consolidation',
                  'web_maintenance_state_pruning',
                  'web_maintenance_world_context_retention',
                  'world_context_retention', 'world_suggestion_review'
              )
              AND status IN ('succeeded', 'failed', 'cancelled')
            GROUP BY type HAVING COUNT(*) > 50
            LIMIT 1
            """,
            (save_id,),
        )
        if terminal_job_excess is not None:
            return True
        for table_name in (
            "world_state",
            "memories",
            "summaries",
            "context_sources",
            "active_threads",
        ):
            archived = self._fetch_one(
                f"""
                SELECT 1 FROM {table_name}
                WHERE save_id = ? AND archived_at IS NOT NULL
                  AND julianday(archived_at) <= julianday('now', '-30 days')
                LIMIT 1
                """,
                (save_id,),
            )
            if archived is not None:
                return True
        return False

    def defer_context_update_suggestion_review(
        self,
        suggestion_ids: list[str] | tuple[str, ...],
        *,
        error: str,
        retry_after_seconds: int,
    ) -> list[ContextUpdateSuggestionRecord]:
        ids = tuple(dict.fromkeys(suggestion_ids))
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        self.connection.execute(
            f"""
            UPDATE context_update_suggestions
            SET review_attempt_count = review_attempt_count + 1,
                next_review_at = datetime('now', ?),
                last_review_error = ?,
                resolved_at = NULL
            WHERE id IN ({placeholders}) AND status = 'pending'
            """,
            (f"+{retry_after_seconds} seconds", error, *ids),
        )
        self.commit()
        return self._list_context_update_suggestions_by_ids(ids)

    def update_context_update_suggestion_status(
        self,
        suggestion_id: str,
        *,
        status: str,
    ) -> ContextUpdateSuggestionRecord:
        resolved_sql = (
            "resolved_at = CURRENT_TIMESTAMP"
            if status in {"applied", "rejected", "dismissed", "superseded", "expired"}
            else "resolved_at = NULL"
        )
        self.connection.execute(
            f"""
            UPDATE context_update_suggestions
            SET status = ?, {resolved_sql},
                next_review_at = CASE WHEN ? THEN NULL ELSE next_review_at END
            WHERE id = ?
            """,
            (
                status,
                status
                in {"applied", "rejected", "dismissed", "superseded", "expired"},
                suggestion_id,
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, update_type, entity_type, entity_id, field_path,
                   proposed_value_json, status, reason, confidence,
                   source_message_ids_json, created_at, resolved_at,
                   review_attempt_count, next_review_at, last_review_error
            FROM context_update_suggestions
            WHERE id = ?
            """,
            (suggestion_id,),
        )
        if row is None:
            raise ValueError(f"Unknown context update suggestion id: {suggestion_id}")
        return _context_update_suggestion_from_row(row)

    def update_context_update_suggestion_content(
        self,
        suggestion_id: str,
        *,
        proposed_value: object,
        confidence: float,
        source_message_ids: list[str] | tuple[str, ...],
    ) -> ContextUpdateSuggestionRecord:
        self.connection.execute(
            """
            UPDATE context_update_suggestions
            SET proposed_value_json = ?, confidence = ?,
                source_message_ids_json = ?
            WHERE id = ?
            """,
            (
                _dump_json(proposed_value),
                confidence,
                _dump_json(_unique_strings(source_message_ids)),
                suggestion_id,
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, update_type, entity_type, entity_id, field_path,
                   proposed_value_json, status, reason, confidence,
                   source_message_ids_json, created_at, resolved_at,
                   review_attempt_count, next_review_at, last_review_error
            FROM context_update_suggestions
            WHERE id = ?
            """,
            (suggestion_id,),
        )
        if row is None:
            raise ValueError(f"Unknown context update suggestion id: {suggestion_id}")
        return _context_update_suggestion_from_row(row)

    def update_context_update_suggestion_statuses(
        self,
        suggestion_ids: list[str] | tuple[str, ...],
        *,
        status: str,
    ) -> list[ContextUpdateSuggestionRecord]:
        ids = tuple(dict.fromkeys(suggestion_ids))
        if not ids:
            return []
        resolved_sql = (
            "resolved_at = CURRENT_TIMESTAMP"
            if status in {"applied", "rejected", "dismissed", "superseded", "expired"}
            else "resolved_at = NULL"
        )
        placeholders = ", ".join("?" for _ in ids)
        self.connection.execute(
            f"""
            UPDATE context_update_suggestions
            SET status = ?, {resolved_sql},
                next_review_at = CASE WHEN ? THEN NULL ELSE next_review_at END
            WHERE id IN ({placeholders})
            """,
            (
                status,
                status
                in {"applied", "rejected", "dismissed", "superseded", "expired"},
                *ids,
            ),
        )
        self.commit()
        return self._list_context_update_suggestions_by_ids(ids)

    def expire_stale_context_update_suggestions(
        self,
        save_id: str,
        *,
        older_than_days: int,
    ) -> list[ContextUpdateSuggestionRecord]:
        stale = self._fetch_all(
            """
            SELECT id
            FROM context_update_suggestions
            WHERE save_id = ?
              AND status = 'pending'
              AND julianday(created_at) <= julianday('now', ?)
            ORDER BY created_at, rowid
            """,
            (save_id, f"-{older_than_days} days"),
        )
        ids = [str(row["id"]) for row in stale]
        return self.update_context_update_suggestion_statuses(
            ids,
            status="expired",
        )

    def expire_context_update_suggestions_for_messages(
        self,
        *,
        save_id: str,
        message_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        if not message_ids:
            return frozenset()
        suggestion_ids = [
            suggestion.id
            for suggestion in self.list_context_update_suggestions(
                save_id,
                status="pending",
            )
            if set(suggestion.source_message_ids) & set(message_ids)
        ]
        self.update_context_update_suggestion_statuses(
            tuple(suggestion_ids),
            status="expired",
        )
        return frozenset(suggestion_ids)

    def restore_context_update_suggestions(
        self,
        suggestion_ids: set[str] | frozenset[str],
    ) -> None:
        if not suggestion_ids:
            return
        self.update_context_update_suggestion_statuses(
            tuple(suggestion_ids),
            status="pending",
        )

    def find_pending_context_update_suggestion(
        self,
        *,
        save_id: str,
        update_type: str,
        entity_type: str,
        entity_id: str | None,
        field_path: str,
        proposed_value: object,
    ) -> ContextUpdateSuggestionRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, update_type, entity_type, entity_id, field_path,
                   proposed_value_json, status, reason, confidence,
                   source_message_ids_json, created_at, resolved_at,
                   review_attempt_count, next_review_at, last_review_error
            FROM context_update_suggestions
            WHERE save_id = ?
              AND status = 'pending'
              AND update_type = ?
              AND entity_type = ?
              AND (entity_id = ? OR (entity_id IS NULL AND ? IS NULL))
              AND field_path = ?
              AND proposed_value_json = ?
            ORDER BY created_at, rowid
            LIMIT 1
            """,
            (
                save_id,
                update_type,
                entity_type,
                entity_id,
                entity_id,
                field_path,
                _dump_json(proposed_value),
            ),
        )
        if row is None:
            return None
        return _context_update_suggestion_from_row(row)

    def _list_context_update_suggestions_by_ids(
        self,
        suggestion_ids: tuple[str, ...],
    ) -> list[ContextUpdateSuggestionRecord]:
        if not suggestion_ids:
            return []
        placeholders = ", ".join("?" for _ in suggestion_ids)
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, update_type, entity_type, entity_id, field_path,
                   proposed_value_json, status, reason, confidence,
                   source_message_ids_json, created_at, resolved_at,
                   review_attempt_count, next_review_at, last_review_error
            FROM context_update_suggestions
            WHERE id IN ({placeholders})
            ORDER BY created_at, rowid
            """,
            suggestion_ids,
        )
        return [_context_update_suggestion_from_row(row) for row in rows]

    def add_context_update_audit(
        self,
        *,
        save_id: str,
        operation: str,
        entity_type: str,
        field_path: str,
        before: object | None,
        after: object | None,
        reason: str = "",
        confidence: float = 0.0,
        source_message_ids: list[str] | None = None,
        entity_id: str | None = None,
        suggestion_id: str | None = None,
        audit_id: str | None = None,
    ) -> ContextUpdateAuditRecord:
        record = ContextUpdateAuditRecord(
            id=audit_id or _new_id(),
            save_id=save_id,
            suggestion_id=suggestion_id,
            operation=operation,
            entity_type=entity_type,
            entity_id=entity_id,
            field_path=field_path,
            before=before,
            after=after,
            reason=reason,
            confidence=confidence,
            source_message_ids=list(source_message_ids or []),
        )
        self.connection.execute(
            """
            INSERT INTO context_update_audit(
                id, save_id, suggestion_id, operation, entity_type, entity_id,
                field_path, before_json, after_json, reason, confidence,
                source_message_ids_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.suggestion_id,
                record.operation,
                record.entity_type,
                record.entity_id,
                record.field_path,
                _dump_json(record.before) if record.before is not None else None,
                _dump_json(record.after) if record.after is not None else None,
                record.reason,
                record.confidence,
                _dump_json(record.source_message_ids),
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, suggestion_id, operation, entity_type, entity_id,
                   field_path, before_json, after_json, reason, confidence,
                   source_message_ids_json, created_at
            FROM context_update_audit
            WHERE id = ?
            """,
            (record.id,),
        )
        if row is None:
            raise ValueError(f"Unknown context update audit id: {record.id}")
        return _context_update_audit_from_row(row)

    def list_context_update_audit(
        self,
        save_id: str,
    ) -> list[ContextUpdateAuditRecord]:
        rows = self._fetch_all(
            """
            SELECT id, save_id, suggestion_id, operation, entity_type, entity_id,
                   field_path, before_json, after_json, reason, confidence,
                   source_message_ids_json, created_at
            FROM context_update_audit
            WHERE save_id = ?
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )
        return [_context_update_audit_from_row(row) for row in rows]

    def add_memory(
        self,
        *,
        save_id: str,
        body: str,
        tags: list[str],
        importance: float = 1.0,
        source_message_id: str | None = None,
        source_message_ids: list[str] | tuple[str, ...] | None = None,
        memory_id: str | None = None,
        claim_fingerprint: str | None = None,
        source_observation_ids: list[str] | tuple[str, ...] | None = None,
    ) -> MemoryRecord:
        resolved_source_message_ids = _memory_source_message_ids(
            source_message_id=source_message_id,
            source_message_ids=source_message_ids,
        )
        resolved_fingerprint = canonical_claim_fingerprint(body)
        self.begin_immediate_transaction()
        try:
            existing = self.get_memory_by_claim_fingerprint(
                save_id=save_id,
                claim_fingerprint=resolved_fingerprint,
            )
            if existing is not None:
                record = self.update_memory(
                    memory_id=existing.id,
                    body=existing.body,
                    tags=list(dict.fromkeys((*existing.tags, *tags))),
                    importance=max(existing.importance, importance),
                    source_message_ids=list(
                        dict.fromkeys(
                            (
                                *existing.source_message_ids,
                                *resolved_source_message_ids,
                            )
                        )
                    ),
                    source_observation_ids=list(
                        dict.fromkeys(
                            (
                                *existing.source_observation_ids,
                                *(source_observation_ids or ()),
                            )
                        )
                    ),
                    claim_fingerprint=resolved_fingerprint,
                )
                self.commit_transaction()
                return record
            record = MemoryRecord(
                id=memory_id or _new_id(),
                save_id=save_id,
                body=body,
                tags=tags,
                importance=importance,
                source_message_id=source_message_id,
                source_message_ids=resolved_source_message_ids,
                claim_fingerprint=resolved_fingerprint,
                source_observation_ids=_unique_strings(
                    source_observation_ids or ()
                )[:MAX_MEMORY_SOURCE_OBSERVATION_IDS],
            )
            self.connection.execute(
                """
                INSERT INTO memories(
                    id, save_id, body, tags_json, importance, source_message_id,
                    source_message_ids_json, claim_fingerprint,
                    source_observation_ids_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.save_id,
                    record.body,
                    _dump_json(record.tags),
                    record.importance,
                    record.source_message_id,
                    _dump_json(record.source_message_ids),
                    record.claim_fingerprint,
                    _dump_json(record.source_observation_ids),
                ),
            )
            self.commit_transaction()
            return record
        except BaseException:
            self.rollback_transaction()
            raise

    def update_memory(
        self,
        *,
        memory_id: str,
        body: str,
        tags: list[str],
        importance: float,
        source_message_ids: list[str] | tuple[str, ...] | None = None,
        source_observation_ids: list[str] | tuple[str, ...] | None = None,
        claim_fingerprint: str | None = None,
        clear_source: bool = False,
    ) -> MemoryRecord:
        self.begin_immediate_transaction()
        try:
            record = self._update_memory_in_transaction(
                memory_id=memory_id,
                body=body,
                tags=tags,
                importance=importance,
                source_message_ids=source_message_ids,
                source_observation_ids=source_observation_ids,
                claim_fingerprint=claim_fingerprint,
                clear_source=clear_source,
            )
            self.commit_transaction()
            return record
        except BaseException:
            self.rollback_transaction()
            raise

    def _update_memory_in_transaction(
        self,
        *,
        memory_id: str,
        body: str,
        tags: list[str],
        importance: float,
        source_message_ids: list[str] | tuple[str, ...] | None = None,
        source_observation_ids: list[str] | tuple[str, ...] | None = None,
        claim_fingerprint: str | None = None,
        clear_source: bool = False,
    ) -> MemoryRecord:
        current = self._fetch_one(
            """
            SELECT save_id, source_message_id, source_message_ids_json,
                   source_observation_ids_json, claim_fingerprint, archived_at
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )
        if current is None:
            raise ValueError(f"Unknown memory id: {memory_id}")
        if current["archived_at"] is not None:
            raise ValueError(f"Memory is archived: {memory_id}")
        source_message_id = None if clear_source else current["source_message_id"]
        resolved_source_message_ids = (
            []
            if clear_source
            else _memory_source_message_ids(
                source_message_id=source_message_id,
                source_message_ids=(
                    _load_list(current["source_message_ids_json"])
                    if source_message_ids is None
                    else list(source_message_ids)
                ),
            )
        )
        resolved_observation_ids = (
            []
            if clear_source
            else _unique_strings(
                _load_list(current["source_observation_ids_json"])
                if source_observation_ids is None
                else source_observation_ids
            )[:MAX_MEMORY_SOURCE_OBSERVATION_IDS]
        )
        resolved_fingerprint = canonical_claim_fingerprint(body)
        save_id = str(current["save_id"])
        collision = self.get_memory_by_claim_fingerprint(
            save_id=save_id,
            claim_fingerprint=resolved_fingerprint,
        )
        if collision is not None and collision.id != memory_id:
            return self._merge_colliding_memory_update(
                memory_id=memory_id,
                save_id=save_id,
                body=body,
                resolved_fingerprint=resolved_fingerprint,
                tags=tags,
                importance=importance,
                source_message_id=source_message_id,
                source_message_ids=resolved_source_message_ids,
                source_observation_ids=resolved_observation_ids,
            )
        self.connection.execute(
            """
            UPDATE memories
            SET body = ?, tags_json = ?, importance = ?, source_message_id = ?,
                source_message_ids_json = ?,
                claim_fingerprint = ?, source_observation_ids_json = ?,
                updated_at = CURRENT_TIMESTAMP, archived_at = NULL
            WHERE id = ?
            """,
            (
                body,
                _dump_json(tags),
                importance,
                source_message_id,
                _dump_json(resolved_source_message_ids),
                resolved_fingerprint,
                _dump_json(resolved_observation_ids),
                memory_id,
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, body, tags_json, importance, source_message_id,
                   source_message_ids_json, claim_fingerprint,
                   source_observation_ids_json
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )
        if row is None:
            raise ValueError(f"Unknown memory id: {memory_id}")
        return MemoryRecord(
            id=row["id"],
            save_id=row["save_id"],
            body=row["body"],
            tags=_load_list(row["tags_json"]),
            importance=row["importance"],
            source_message_id=row["source_message_id"],
            source_message_ids=_memory_source_message_ids(
                source_message_id=row["source_message_id"],
                source_message_ids=_load_list(row["source_message_ids_json"]),
            ),
            claim_fingerprint=row["claim_fingerprint"],
            source_observation_ids=_load_list(row["source_observation_ids_json"]),
        )

    def _merge_colliding_memory_update(
        self,
        *,
        memory_id: str,
        save_id: str,
        body: str,
        resolved_fingerprint: str,
        tags: list[str],
        importance: float,
        source_message_id: str | None,
        source_message_ids: list[str],
        source_observation_ids: list[str],
    ) -> MemoryRecord:
        self.begin_immediate_transaction()
        try:
            collision = self.get_memory_by_claim_fingerprint(
                save_id=save_id,
                claim_fingerprint=resolved_fingerprint,
            )
            if collision is None or collision.id == memory_id:
                raise RuntimeError("Memory fingerprint collision changed during update")
            merged_source_message_ids = _memory_source_message_ids(
                source_message_id=source_message_id or collision.source_message_id,
                source_message_ids=list(
                    dict.fromkeys(
                        (*source_message_ids, *collision.source_message_ids)
                    )
                ),
            )
            merged_observation_ids = _unique_strings(
                [
                    *source_observation_ids,
                    *collision.source_observation_ids,
                ]
            )[:MAX_MEMORY_SOURCE_OBSERVATION_IDS]
            archived = self.connection.execute(
                """
                UPDATE memories
                SET archived_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND save_id = ?
                """,
                (collision.id, save_id),
            )
            if archived.rowcount != 1:
                raise ValueError("Colliding memory is no longer active")
            updated = self.connection.execute(
                """
                UPDATE memories
                SET body = ?, tags_json = ?, importance = ?,
                    source_message_id = ?,
                    source_message_ids_json = ?,
                    claim_fingerprint = ?,
                    source_observation_ids_json = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    archived_at = NULL
                WHERE id = ? AND save_id = ? AND archived_at IS NULL
                """,
                (
                    body,
                    _dump_json(list(dict.fromkeys((*tags, *collision.tags)))),
                    max(collision.importance, importance),
                    source_message_id or collision.source_message_id,
                    _dump_json(merged_source_message_ids),
                    resolved_fingerprint,
                    _dump_json(merged_observation_ids),
                    memory_id,
                    save_id,
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("Edited memory is no longer active")
            _remap_migrated_memory_references(
                self.connection,
                save_id=save_id,
                duplicate_id=collision.id,
                keeper_id=memory_id,
            )
            self.connection.execute(
                """
                UPDATE context_sources
                SET archived_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE save_id = ? AND source_type = 'memory'
                  AND source_id = ?
                """,
                (save_id, collision.id),
            )
            self.commit_transaction()
        except BaseException:
            self.rollback_transaction()
            raise
        merged = self.get_memory(save_id, memory_id)
        if merged is None:
            raise ValueError("Failed to merge colliding memory update")
        return merged

    def archive_memory(self, memory_id: str) -> None:
        self.connection.execute(
            """
            UPDATE memories
            SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (memory_id,),
        )
        self.commit()

    def restore_memories(self, memory_ids: set[str] | frozenset[str]) -> None:
        if not memory_ids:
            return
        self.begin_immediate_transaction()
        try:
            rows = self._fetch_all(
                f"""
                SELECT id, save_id, body, tags_json, importance,
                       source_message_id, source_message_ids_json,
                       source_observation_ids_json, archived_at
                FROM memories
                WHERE id IN ({_placeholders(len(memory_ids))})
                ORDER BY created_at, rowid
                """,
                tuple(sorted(memory_ids)),
            )
            for row in rows:
                if row["archived_at"] is None:
                    continue
                memory_id = str(row["id"])
                save_id = str(row["save_id"])
                fingerprint = canonical_claim_fingerprint(row["body"])
                collision = self.get_memory_by_claim_fingerprint(
                    save_id=save_id,
                    claim_fingerprint=fingerprint,
                )
                if collision is None or collision.id == memory_id:
                    restored = self.connection.execute(
                        """
                        UPDATE memories
                        SET claim_fingerprint = ?, archived_at = NULL,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ? AND save_id = ?
                          AND archived_at IS NOT NULL
                        """,
                        (fingerprint, memory_id, save_id),
                    )
                    if restored.rowcount != 1:
                        raise ValueError("Memory changed while being restored")
                    continue
                source_message_id = (
                    row["source_message_id"] or collision.source_message_id
                )
                source_message_ids = _memory_source_message_ids(
                    source_message_id=source_message_id,
                    source_message_ids=list(
                        dict.fromkeys(
                            (
                                *_load_list(row["source_message_ids_json"]),
                                *collision.source_message_ids,
                            )
                        )
                    ),
                )
                observation_ids = _unique_strings(
                    (
                        *_load_list(row["source_observation_ids_json"]),
                        *collision.source_observation_ids,
                    )
                )[:MAX_MEMORY_SOURCE_OBSERVATION_IDS]
                archived = self.connection.execute(
                    """
                    UPDATE memories
                    SET archived_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND save_id = ? AND archived_at IS NULL
                    """,
                    (collision.id, save_id),
                )
                if archived.rowcount != 1:
                    raise ValueError("Colliding memory changed while restoring")
                restored = self.connection.execute(
                    """
                    UPDATE memories
                    SET tags_json = ?, importance = ?, source_message_id = ?,
                        source_message_ids_json = ?, claim_fingerprint = ?,
                        source_observation_ids_json = ?, archived_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND save_id = ? AND archived_at IS NOT NULL
                    """,
                    (
                        _dump_json(
                            list(
                                dict.fromkeys(
                                    (
                                        *_load_list(row["tags_json"]),
                                        *collision.tags,
                                    )
                                )
                            )
                        ),
                        max(float(row["importance"]), collision.importance),
                        source_message_id,
                        _dump_json(source_message_ids),
                        fingerprint,
                        _dump_json(observation_ids),
                        memory_id,
                        save_id,
                    ),
                )
                if restored.rowcount != 1:
                    raise ValueError("Memory changed while being restored")
                _remap_migrated_memory_references(
                    self.connection,
                    save_id=save_id,
                    duplicate_id=collision.id,
                    keeper_id=memory_id,
                )
                self.connection.execute(
                    """
                    UPDATE context_sources
                    SET archived_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE save_id = ? AND source_type = 'memory'
                      AND source_id = ?
                    """,
                    (save_id, collision.id),
                )
            self.commit_transaction()
        except BaseException:
            self.rollback_transaction()
            raise

    def list_memories(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        limit_sql = "LIMIT ?" if limit is not None else ""
        order_sql = (
            "ORDER BY created_at DESC, rowid DESC"
            if limit is not None
            else "ORDER BY created_at, rowid"
        )
        params: tuple[object, ...] = (
            (save_id, max(0, limit))
            if limit is not None
            else (save_id,)
        )
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, body, tags_json, importance, source_message_id,
                   source_message_ids_json, claim_fingerprint,
                   source_observation_ids_json
            FROM memories
            WHERE save_id = ? AND archived_at IS NULL
            {order_sql}
            {limit_sql}
            """,
            params,
        )
        if limit is not None:
            rows.reverse()
        return [
            MemoryRecord(
                id=row["id"],
                save_id=row["save_id"],
                body=row["body"],
                tags=_load_list(row["tags_json"]),
                importance=row["importance"],
                source_message_id=row["source_message_id"],
                source_message_ids=_memory_source_message_ids(
                    source_message_id=row["source_message_id"],
                    source_message_ids=_load_list(row["source_message_ids_json"]),
                ),
                claim_fingerprint=row["claim_fingerprint"],
                source_observation_ids=_load_list(
                    row["source_observation_ids_json"]
                ),
            )
            for row in rows
        ]

    def get_memory(self, save_id: str, memory_id: str) -> MemoryRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, body, tags_json, importance, source_message_id,
                   source_message_ids_json, claim_fingerprint,
                   source_observation_ids_json
            FROM memories
            WHERE save_id = ? AND id = ? AND archived_at IS NULL
            """,
            (save_id, memory_id),
        )
        if row is None:
            return None
        return MemoryRecord(
            id=row["id"],
            save_id=row["save_id"],
            body=row["body"],
            tags=_load_list(row["tags_json"]),
            importance=row["importance"],
            source_message_id=row["source_message_id"],
            source_message_ids=_memory_source_message_ids(
                source_message_id=row["source_message_id"],
                source_message_ids=_load_list(row["source_message_ids_json"]),
            ),
            claim_fingerprint=row["claim_fingerprint"],
            source_observation_ids=_load_list(
                row["source_observation_ids_json"]
            ),
        )

    def list_memories_for_continuity_index(
        self,
        save_id: str,
        *,
        limit: int,
    ) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        rows = self._fetch_all(
            """
            WITH classified AS (
                SELECT
                    id, save_id, body, tags_json, importance,
                    source_message_id, source_message_ids_json,
                    claim_fingerprint, source_observation_ids_json,
                    created_at, rowid AS source_rowid,
                    CASE
                        WHEN (
                            EXISTS (
                                SELECT 1 FROM json_each(memories.tags_json) tag
                                WHERE lower(CAST(tag.value AS TEXT)) = 'dossier'
                            )
                            AND EXISTS (
                                SELECT 1 FROM json_each(memories.tags_json) tag
                                WHERE lower(CAST(tag.value AS TEXT)) = 'relationship'
                                   OR lower(CAST(tag.value AS TEXT))
                                      LIKE 'character:%'
                            )
                        ) THEN 0.82
                        WHEN EXISTS (
                            SELECT 1 FROM json_each(memories.tags_json) tag
                            WHERE lower(CAST(tag.value AS TEXT))
                                  IN ('promise', 'oath', 'obligation', 'quest', 'task')
                        ) OR lower(memories.body) LIKE '%promised%'
                          OR lower(memories.body) LIKE '%swore%'
                          OR lower(memories.body) LIKE '%owes%'
                          OR lower(memories.body) LIKE '%must%'
                        THEN 0.9
                        WHEN EXISTS (
                            SELECT 1 FROM json_each(memories.tags_json) tag
                            WHERE lower(CAST(tag.value AS TEXT))
                                  IN ('relationship', 'trust', 'rivalry')
                        ) THEN 0.82
                        WHEN EXISTS (
                            SELECT 1 FROM json_each(memories.tags_json) tag
                            WHERE lower(CAST(tag.value AS TEXT))
                                  IN ('voice', 'speech', 'diction')
                        ) THEN 0.95
                        WHEN EXISTS (
                            SELECT 1 FROM json_each(memories.tags_json) tag
                            WHERE lower(CAST(tag.value AS TEXT))
                                  IN ('inventory', 'item', 'object')
                        ) THEN 0.85
                        ELSE 0.45
                    END AS fact_floor,
                    CASE
                        WHEN EXISTS (
                            SELECT 1 FROM json_each(memories.tags_json) tag
                            WHERE lower(CAST(tag.value AS TEXT)) IN (
                                'promise', 'oath', 'obligation', 'quest', 'task',
                                'relationship', 'trust', 'rivalry',
                                'voice', 'speech', 'diction',
                                'inventory', 'item', 'object'
                            )
                            OR lower(CAST(tag.value AS TEXT)) LIKE 'character:%'
                        ) OR lower(memories.body) LIKE '%promised%'
                          OR lower(memories.body) LIKE '%swore%'
                          OR lower(memories.body) LIKE '%owes%'
                          OR lower(memories.body) LIKE '%must%'
                        THEN 1
                        ELSE 0
                    END AS high_value
                FROM memories
                WHERE save_id = ? AND archived_at IS NULL
            ),
            selected AS (
                SELECT *
                FROM classified
                ORDER BY
                    high_value DESC,
                    CASE
                        WHEN importance > fact_floor THEN importance
                        ELSE fact_floor
                    END DESC,
                    created_at DESC,
                    source_rowid DESC
                LIMIT ?
            )
            SELECT
                id, save_id, body, tags_json, importance, source_message_id,
                source_message_ids_json, claim_fingerprint,
                source_observation_ids_json
            FROM selected
            ORDER BY created_at, source_rowid
            """,
            (save_id, limit),
        )
        return [
            MemoryRecord(
                id=row["id"],
                save_id=row["save_id"],
                body=row["body"],
                tags=_load_list(row["tags_json"]),
                importance=row["importance"],
                source_message_id=row["source_message_id"],
                source_message_ids=_memory_source_message_ids(
                    source_message_id=row["source_message_id"],
                    source_message_ids=_load_list(row["source_message_ids_json"]),
                ),
                claim_fingerprint=row["claim_fingerprint"],
                source_observation_ids=_load_list(
                    row["source_observation_ids_json"]
                ),
            )
            for row in rows
        ]

    def list_context_observations_by_ids(
        self,
        save_id: str,
        observation_ids: set[str] | frozenset[str],
    ) -> list[ContextObservationRecord]:
        if not observation_ids:
            return []
        rows = self._fetch_all(
            """
            SELECT
                observations.id, observations.save_id,
                observations.observation_type, observations.claim,
                observations.evidence_quote,
                observations.source_message_ids_json, observations.scope,
                observations.status, observations.confidence,
                observations.tags_json, observations.metadata_json,
                observations.created_at, observations.updated_at,
                observations.archived_at
            FROM context_observations AS observations
            JOIN json_each(?) selected
              ON observations.id = CAST(selected.value AS TEXT)
            WHERE observations.save_id = ?
              AND observations.archived_at IS NULL
            ORDER BY observations.created_at, observations.rowid
            """,
            (_dump_json(sorted(observation_ids)), save_id),
        )
        return [_context_observation_from_row(row) for row in rows]

    def get_memory_by_claim_fingerprint(
        self,
        *,
        save_id: str,
        claim_fingerprint: str,
    ) -> MemoryRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, body, tags_json, importance, source_message_id,
                   source_message_ids_json, claim_fingerprint,
                   source_observation_ids_json
            FROM memories
            WHERE save_id = ?
              AND claim_fingerprint = ?
              AND archived_at IS NULL
            ORDER BY created_at, rowid
            LIMIT 1
            """,
            (save_id, claim_fingerprint),
        )
        if row is None:
            return None
        return MemoryRecord(
            id=row["id"],
            save_id=row["save_id"],
            body=row["body"],
            tags=_load_list(row["tags_json"]),
            importance=row["importance"],
            source_message_id=row["source_message_id"],
            source_message_ids=_memory_source_message_ids(
                source_message_id=row["source_message_id"],
                source_message_ids=_load_list(row["source_message_ids_json"]),
            ),
            claim_fingerprint=row["claim_fingerprint"],
            source_observation_ids=_load_list(
                row["source_observation_ids_json"]
            ),
        )

    def consolidate_active_memory_duplicates(
        self,
        *,
        save_id: str,
    ) -> dict[str, str]:
        groups: dict[str, list[MemoryRecord]] = {}
        for memory in self.list_memories(save_id):
            fingerprint = (
                memory.claim_fingerprint
                or canonical_claim_fingerprint(memory.body)
            )
            if fingerprint:
                groups.setdefault(fingerprint, []).append(memory)
        remapped_ids: dict[str, str] = {}
        self.begin_immediate_transaction()
        try:
            for fingerprint, group in groups.items():
                if len(group) < 2:
                    continue
                keeper = group[0]
                self.update_memory(
                    memory_id=keeper.id,
                    body=keeper.body,
                    tags=list(
                        dict.fromkeys(
                            tag
                            for memory in group
                            for tag in memory.tags
                        )
                    ),
                    importance=max(memory.importance for memory in group),
                    source_message_ids=list(
                        dict.fromkeys(
                            source_id
                            for memory in group
                            for source_id in memory.source_message_ids
                        )
                    ),
                    source_observation_ids=list(
                        dict.fromkeys(
                            observation_id
                            for memory in group
                            for observation_id in memory.source_observation_ids
                        )
                    ),
                    claim_fingerprint=fingerprint,
                )
                for duplicate in group[1:]:
                    remapped_ids[duplicate.id] = keeper.id
                    self._merge_active_memory_context_source_conflicts(
                        save_id=save_id,
                        keeper_id=keeper.id,
                        duplicate_id=duplicate.id,
                    )
                    self._merge_active_memory_knowledge_edge_conflicts(
                        save_id=save_id,
                        keeper_id=keeper.id,
                        duplicate_id=duplicate.id,
                    )
                    self.connection.execute(
                        """
                        DELETE FROM character_knowledge_edges AS keeper_edge
                        WHERE keeper_edge.save_id = ?
                          AND keeper_edge.target_type IN ('memory', 'memories')
                          AND keeper_edge.target_id = ?
                          AND keeper_edge.archived_at IS NOT NULL
                          AND EXISTS (
                            SELECT 1
                            FROM character_knowledge_edges AS duplicate_edge
                            WHERE duplicate_edge.save_id = keeper_edge.save_id
                              AND duplicate_edge.character_id =
                                  keeper_edge.character_id
                              AND duplicate_edge.target_type =
                                  keeper_edge.target_type
                              AND duplicate_edge.target_id = ?
                              AND duplicate_edge.archived_at IS NULL
                          )
                        """,
                        (save_id, keeper.id, duplicate.id),
                    )
                    self.connection.execute(
                        """
                        DELETE FROM character_knowledge_edges AS duplicate_edge
                        WHERE duplicate_edge.save_id = ?
                          AND duplicate_edge.target_type IN ('memory', 'memories')
                          AND duplicate_edge.target_id = ?
                          AND EXISTS (
                            SELECT 1
                            FROM character_knowledge_edges AS keeper_edge
                            WHERE keeper_edge.save_id = duplicate_edge.save_id
                              AND keeper_edge.character_id =
                                  duplicate_edge.character_id
                              AND keeper_edge.target_type =
                                  duplicate_edge.target_type
                              AND keeper_edge.target_id = ?
                          )
                        """,
                        (save_id, duplicate.id, keeper.id),
                    )
                    self.connection.execute(
                        """
                        UPDATE character_knowledge_edges
                        SET target_id = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE save_id = ?
                          AND target_type IN ('memory', 'memories')
                          AND target_id = ?
                        """,
                        (keeper.id, save_id, duplicate.id),
                    )
                    self.connection.execute(
                        """
                        DELETE FROM context_sources AS keeper_source
                        WHERE keeper_source.save_id = ?
                          AND keeper_source.source_type = 'memory'
                          AND keeper_source.source_id = ?
                          AND keeper_source.archived_at IS NOT NULL
                          AND EXISTS (
                            SELECT 1
                            FROM context_sources AS duplicate_source
                            WHERE duplicate_source.save_id =
                                  keeper_source.save_id
                              AND duplicate_source.source_type = 'memory'
                              AND duplicate_source.source_id = ?
                              AND duplicate_source.archived_at IS NULL
                          )
                        """,
                        (save_id, keeper.id, duplicate.id),
                    )
                    self.connection.execute(
                        """
                        UPDATE OR IGNORE context_sources
                        SET source_id = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE save_id = ?
                          AND source_type = 'memory'
                          AND source_id = ?
                        """,
                        (keeper.id, save_id, duplicate.id),
                    )
                    self.connection.execute(
                        """
                        DELETE FROM entity_links AS duplicate_link
                        WHERE duplicate_link.save_id = ?
                          AND duplicate_link.target_type IN ('memory', 'memories')
                          AND duplicate_link.target_id = ?
                          AND EXISTS (
                            SELECT 1
                            FROM entity_links AS keeper_link
                            WHERE keeper_link.save_id = duplicate_link.save_id
                              AND keeper_link.entity_type =
                                  duplicate_link.entity_type
                              AND keeper_link.entity_id =
                                  duplicate_link.entity_id
                              AND keeper_link.target_type =
                                  duplicate_link.target_type
                              AND keeper_link.target_id = ?
                              AND keeper_link.relation =
                                  duplicate_link.relation
                          )
                        """,
                        (save_id, duplicate.id, keeper.id),
                    )
                    self.connection.execute(
                        """
                        UPDATE entity_links
                        SET target_id = ?
                        WHERE save_id = ?
                          AND target_type IN ('memory', 'memories')
                          AND target_id = ?
                        """,
                        (keeper.id, save_id, duplicate.id),
                    )
                    self.connection.execute(
                        """
                        UPDATE OR IGNORE entity_links
                        SET entity_id = ?
                        WHERE save_id = ?
                          AND entity_type IN ('memory', 'memories')
                          AND entity_id = ?
                        """,
                        (keeper.id, save_id, duplicate.id),
                    )
                    self.connection.execute(
                        """
                        DELETE FROM entity_links
                        WHERE save_id = ?
                          AND entity_type IN ('memory', 'memories')
                          AND entity_id = ?
                        """,
                        (save_id, duplicate.id),
                    )
                    self.connection.execute(
                        """
                        UPDATE context_update_suggestions
                        SET entity_id = ?
                        WHERE save_id = ?
                          AND entity_type = 'memory'
                          AND entity_id = ?
                        """,
                        (keeper.id, save_id, duplicate.id),
                    )
                    self.connection.execute(
                        """
                        UPDATE context_update_audit
                        SET entity_id = ?
                        WHERE save_id = ?
                          AND entity_type = 'memory'
                          AND entity_id = ?
                        """,
                        (keeper.id, save_id, duplicate.id),
                    )
                    self.connection.execute(
                        """
                        UPDATE character_text_provenance
                        SET target_id = ?
                        WHERE save_id = ?
                          AND target_type IN ('memory', 'memories')
                          AND target_id = ?
                        """,
                        (keeper.id, save_id, duplicate.id),
                    )
                    self.connection.execute(
                        """
                        UPDATE context_sources
                        SET archived_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE save_id = ?
                          AND source_type = 'memory'
                          AND source_id = ?
                        """,
                        (save_id, duplicate.id),
                    )
                    _remap_migrated_memory_proactive_triggers(
                        self.connection,
                        save_id=save_id,
                        duplicate_id=duplicate.id,
                        keeper_id=keeper.id,
                    )
                    self.archive_memory(duplicate.id)
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise
        return remapped_ids

    def _merge_active_memory_context_source_conflicts(
        self,
        *,
        save_id: str,
        keeper_id: str,
        duplicate_id: str,
    ) -> None:
        rows = self._fetch_all(
            """
            SELECT source_id, metadata_json, token_estimate
            FROM context_sources
            WHERE save_id = ? AND source_type = 'memory'
              AND source_id IN (?, ?) AND archived_at IS NULL
            """,
            (save_id, keeper_id, duplicate_id),
        )
        rows_by_source_id = {str(row["source_id"]): row for row in rows}
        keeper = rows_by_source_id.get(keeper_id)
        duplicate = rows_by_source_id.get(duplicate_id)
        if keeper is None or duplicate is None:
            return
        metadata = merge_context_source_metadata(
            keeper["metadata_json"],
            duplicate["metadata_json"],
        )
        _validate_context_source_provenance_metadata(metadata)
        self.connection.execute(
            """
            UPDATE context_sources
            SET metadata_json = ?, token_estimate = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ? AND source_type = 'memory'
              AND source_id = ? AND archived_at IS NULL
            """,
            (
                _dump_json(metadata),
                max(
                    int(keeper["token_estimate"] or 0),
                    int(duplicate["token_estimate"] or 0),
                ),
                save_id,
                keeper_id,
            ),
        )

    def _merge_active_memory_knowledge_edge_conflicts(
        self,
        *,
        save_id: str,
        keeper_id: str,
        duplicate_id: str,
    ) -> None:
        rows = self._fetch_all(
            """
            SELECT
                keeper_edge.id AS keeper_edge_id,
                keeper_edge.knowledge_state AS keeper_state,
                keeper_edge.acquisition_method AS keeper_method,
                keeper_edge.confidence AS keeper_confidence,
                keeper_edge.source_message_id AS keeper_source_message_id,
                keeper_edge.source_message_ids_json AS keeper_source_ids_json,
                keeper_edge.evidence_quote AS keeper_evidence,
                duplicate_edge.id AS duplicate_edge_id,
                duplicate_edge.knowledge_state AS duplicate_state,
                duplicate_edge.acquisition_method AS duplicate_method,
                duplicate_edge.confidence AS duplicate_confidence,
                duplicate_edge.source_message_id AS duplicate_source_message_id,
                duplicate_edge.source_message_ids_json AS duplicate_source_ids_json,
                duplicate_edge.evidence_quote AS duplicate_evidence
            FROM character_knowledge_edges AS keeper_edge
            JOIN character_knowledge_edges AS duplicate_edge
              ON duplicate_edge.save_id = keeper_edge.save_id
             AND duplicate_edge.character_id = keeper_edge.character_id
             AND duplicate_edge.target_type = keeper_edge.target_type
            WHERE keeper_edge.save_id = ?
              AND keeper_edge.target_type IN ('memory', 'memories')
              AND keeper_edge.target_id = ?
              AND duplicate_edge.target_id = ?
              AND keeper_edge.archived_at IS NULL
              AND duplicate_edge.archived_at IS NULL
            """,
            (save_id, keeper_id, duplicate_id),
        )
        state_rank = {"knows": 0, "may_know": 1, "does_not_know": 2}
        for row in rows:
            duplicate_dominates = state_rank.get(
                str(row["duplicate_state"]),
                1,
            ) > state_rank.get(str(row["keeper_state"]), 1)
            dominant_prefix = "duplicate" if duplicate_dominates else "keeper"
            source_ids = list(
                dict.fromkeys(
                    (
                        *_load_list(row["keeper_source_ids_json"]),
                        *_load_list(row["duplicate_source_ids_json"]),
                    )
                )
            )
            provenance_overflow = (
                len(source_ids) > MAX_KNOWLEDGE_EDGE_SOURCE_MESSAGE_IDS
            )
            self.connection.execute(
                """
                UPDATE character_knowledge_edges
                SET knowledge_state = ?, acquisition_method = ?,
                    confidence = ?, source_message_id = ?,
                    source_message_ids_json = ?, evidence_quote = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    (
                        "does_not_know"
                        if provenance_overflow
                        else row[f"{dominant_prefix}_state"]
                    ),
                    (
                        "unknown"
                        if provenance_overflow
                        else row[f"{dominant_prefix}_method"]
                    ),
                    max(
                        float(row["keeper_confidence"]),
                        float(row["duplicate_confidence"]),
                    ),
                    (
                        None
                        if provenance_overflow
                        else row[f"{dominant_prefix}_source_message_id"]
                    ),
                    _dump_json([] if provenance_overflow else source_ids),
                    (
                        "Provenance exceeded the safe bound."
                        if provenance_overflow
                        else row[f"{dominant_prefix}_evidence"]
                    ),
                    row["keeper_edge_id"],
                ),
            )
            self.connection.execute(
                "DELETE FROM character_knowledge_edges WHERE id = ?",
                (row["duplicate_edge_id"],),
            )

    def count_active_memories(self, save_id: str) -> int:
        row = self._fetch_one(
            """
            SELECT COUNT(*) AS memory_count
            FROM memories
            WHERE save_id = ? AND archived_at IS NULL
            """,
            (save_id,),
        )
        return int(row["memory_count"]) if row is not None else 0

    def add_summary(
        self,
        *,
        save_id: str,
        covers_message_start_id: str,
        covers_message_end_id: str,
        body: str,
        provider: str,
        model: str,
        content_rating: str = "unclassified",
        summary_id: str | None = None,
    ) -> SummaryRecord:
        record = SummaryRecord(
            id=summary_id or _new_id(),
            save_id=save_id,
            covers_message_start_id=covers_message_start_id,
            covers_message_end_id=covers_message_end_id,
            body=body,
            provider=provider,
            model=model,
            content_rating=content_rating,
        )
        self.connection.execute(
            """
            INSERT INTO summaries(
                id, save_id, covers_message_start_id, covers_message_end_id,
                body, provider, model, content_rating
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.covers_message_start_id,
                record.covers_message_end_id,
                record.body,
                record.provider,
                record.model,
                record.content_rating,
            ),
        )
        self.commit()
        return record

    def update_summary(
        self,
        *,
        summary_id: str,
        body: str,
        content_rating: str | None = None,
    ) -> SummaryRecord:
        self.connection.execute(
            """
            UPDATE summaries
            SET body = ?,
                content_rating = COALESCE(?, content_rating),
                archived_at = NULL
            WHERE id = ?
            """,
            (body, content_rating, summary_id),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, covers_message_start_id, covers_message_end_id,
                   body, provider, model, content_rating
            FROM summaries
            WHERE id = ?
            """,
            (summary_id,),
        )
        if row is None:
            raise ValueError(f"Unknown summary id: {summary_id}")
        return SummaryRecord(**dict(row))

    def archive_summary(self, summary_id: str) -> None:
        self.connection.execute(
            """
            UPDATE summaries
            SET archived_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (summary_id,),
        )
        self.commit()

    def restore_summaries(self, summary_ids: set[str] | frozenset[str]) -> None:
        if not summary_ids:
            return
        self.connection.execute(
            f"""
            UPDATE summaries
            SET archived_at = NULL
            WHERE id IN ({_placeholders(len(summary_ids))})
            """,
            tuple(summary_ids),
        )
        self.commit()

    def list_summaries(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[SummaryRecord]:
        order_sql = (
            "ORDER BY created_at DESC, rowid DESC"
            if limit is not None
            else "ORDER BY created_at, rowid"
        )
        limit_sql = "LIMIT ?" if limit is not None else ""
        params: tuple[object, ...] = (
            (save_id, max(0, limit))
            if limit is not None
            else (save_id,)
        )
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, covers_message_start_id, covers_message_end_id,
                   body, provider, model, content_rating
            FROM summaries
            WHERE save_id = ? AND archived_at IS NULL
            {order_sql}
            {limit_sql}
            """,
            params,
        )
        if limit is not None:
            rows.reverse()
        return [SummaryRecord(**dict(row)) for row in rows]

    def get_summary(self, save_id: str, summary_id: str) -> SummaryRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, covers_message_start_id, covers_message_end_id,
                   body, provider, model, content_rating
            FROM summaries
            WHERE save_id = ? AND id = ? AND archived_at IS NULL
            """,
            (save_id, summary_id),
        )
        return SummaryRecord(**dict(row)) if row is not None else None

    def summary_visible_to_characters(
        self,
        *,
        save_id: str,
        covers_message_start_id: str,
        covers_message_end_id: str,
        character_ids: set[str] | frozenset[str] | tuple[str, ...],
    ) -> bool:
        scoped_character_ids = tuple(sorted(set(character_ids)))
        if not scoped_character_ids:
            return True
        endpoints = self._fetch_all(
            """
            SELECT id, rowid AS message_rowid
            FROM messages
            WHERE save_id = ? AND id IN (?, ?)
            """,
            (save_id, covers_message_start_id, covers_message_end_id),
        )
        endpoint_rowids = {
            str(row["id"]): int(row["message_rowid"]) for row in endpoints
        }
        start_rowid = endpoint_rowids.get(covers_message_start_id)
        end_rowid = endpoint_rowids.get(covers_message_end_id)
        if start_rowid is None or end_rowid is None:
            return False
        hidden = self._fetch_one(
            f"""
            SELECT 1
            FROM message_visibility AS visibility
            JOIN messages AS message
              ON message.id = visibility.message_id
             AND message.save_id = visibility.save_id
            WHERE visibility.save_id = ?
              AND visibility.character_id IN (
                    {_placeholders(len(scoped_character_ids))}
                  )
              AND visibility.visibility = 'not_visible'
              AND message.rowid BETWEEN ? AND ?
            LIMIT 1
            """,
            (
                save_id,
                *scoped_character_ids,
                min(start_rowid, end_rowid),
                max(start_rowid, end_rowid),
            ),
        )
        return hidden is None

    def save_provider_model(
        self,
        *,
        provider: str,
        model_id: str,
        display_name: str,
        capabilities: list[str],
        supported_parameters: list[str] | None = None,
        context_window: int | None = None,
        pricing: dict[str, str] | None = None,
        thinking: Mapping[str, object] | None = None,
        refreshed_at: str | None = None,
        record_id: str | None = None,
    ) -> ProviderModelRecord:
        existing = self._fetch_one(
            "SELECT id FROM provider_models WHERE provider = ? AND model_id = ?",
            (provider, model_id),
        )
        record = ProviderModelRecord(
            id=existing["id"] if existing else record_id or _new_id(),
            provider=provider,
            model_id=model_id,
            display_name=display_name,
            capabilities=capabilities,
            context_window=context_window,
            available=True,
            supported_parameters=supported_parameters or [],
            pricing=pricing or {},
            thinking=dict(thinking or {}),
        )
        self.connection.execute(
            """
            INSERT INTO provider_models(
                id, provider, model_id, display_name, capabilities_json,
                supported_parameters_json, context_window, available, pricing_json,
                thinking_json, refreshed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
            ON CONFLICT(provider, model_id) DO UPDATE SET
                display_name = excluded.display_name,
                capabilities_json = excluded.capabilities_json,
                supported_parameters_json = excluded.supported_parameters_json,
                context_window = excluded.context_window,
                available = 1,
                pricing_json = excluded.pricing_json,
                thinking_json = excluded.thinking_json,
                refreshed_at = excluded.refreshed_at
            """,
            (
                record.id,
                record.provider,
                record.model_id,
                record.display_name,
                _dump_json(record.capabilities),
                _dump_json(record.supported_parameters),
                record.context_window,
                _dump_json(record.pricing),
                _dump_json(record.thinking),
                refreshed_at,
            ),
        )
        self.commit()
        return record

    def mark_missing_provider_models_unavailable(
        self,
        *,
        provider: str,
        available_model_ids: set[str],
    ) -> None:
        rows = self._fetch_all(
            "SELECT model_id FROM provider_models WHERE provider = ?",
            (provider,),
        )
        missing_model_ids = [
            row["model_id"]
            for row in rows
            if row["model_id"] not in available_model_ids
        ]
        if not missing_model_ids:
            return
        placeholders = ", ".join("?" for _ in missing_model_ids)
        self.connection.execute(
            f"""
            UPDATE provider_models
            SET available = 0, refreshed_at = CURRENT_TIMESTAMP
            WHERE provider = ? AND model_id IN ({placeholders})
            """,
            (provider, *missing_model_ids),
        )
        self.commit()

    def upsert_provider_config(
        self,
        *,
        provider: str,
        enabled: bool,
        has_api_key: bool,
        last_model_refresh_at: str | None = None,
        last_error: str | None = None,
        config_id: str | None = None,
    ) -> ProviderConfigRecord:
        existing = self._fetch_one(
            "SELECT id FROM provider_configs WHERE provider = ?",
            (provider,),
        )
        record = ProviderConfigRecord(
            id=existing["id"] if existing else config_id or _new_id(),
            provider=provider,
            enabled=enabled,
            has_api_key=has_api_key,
            last_model_refresh_at=last_model_refresh_at,
            last_error=last_error,
        )
        self.connection.execute(
            """
            INSERT INTO provider_configs(
                id, provider, enabled, has_api_key, last_model_refresh_at, last_error
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                enabled = excluded.enabled,
                has_api_key = excluded.has_api_key,
                last_model_refresh_at = excluded.last_model_refresh_at,
                last_error = excluded.last_error,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                record.id,
                record.provider,
                int(record.enabled),
                int(record.has_api_key),
                record.last_model_refresh_at,
                record.last_error,
            ),
        )
        self.commit()
        return record

    def get_provider_config(self, provider: str) -> ProviderConfigRecord | None:
        row = self._fetch_one(
            """
            SELECT id, provider, enabled, has_api_key, last_model_refresh_at, last_error
            FROM provider_configs
            WHERE provider = ?
            """,
            (provider,),
        )
        if row is None:
            return None
        return ProviderConfigRecord(
            id=row["id"],
            provider=row["provider"],
            enabled=bool(row["enabled"]),
            has_api_key=bool(row["has_api_key"]),
            last_model_refresh_at=row["last_model_refresh_at"],
            last_error=row["last_error"],
        )

    def list_provider_models(self, provider: str) -> list[ProviderModelRecord]:
        rows = self._fetch_all(
            """
            SELECT id, provider, model_id, display_name, capabilities_json,
                   supported_parameters_json, context_window, available,
                   pricing_json, thinking_json
            FROM provider_models
            WHERE provider = ?
            ORDER BY display_name
            """,
            (provider,),
        )
        return [
            ProviderModelRecord(
                id=row["id"],
                provider=row["provider"],
                model_id=row["model_id"],
                display_name=row["display_name"],
                capabilities=_load_list(row["capabilities_json"]),
                supported_parameters=_load_list(row["supported_parameters_json"]),
                context_window=row["context_window"],
                available=bool(row["available"]),
                pricing=_load_string_dict(row["pricing_json"]),
                thinking=_load_object(row["thinking_json"]),
            )
            for row in rows
        ]

    def count_provider_models(self, provider: str) -> int:
        row = self._fetch_one(
            """
            SELECT COUNT(*) AS count
            FROM provider_models
            WHERE provider = ?
            """,
            (provider,),
        )
        return int(row["count"]) if row is not None else 0

    def replace_provider_catalog_entries(
        self,
        *,
        provider: str,
        entries: list[Mapping[str, object]],
    ) -> None:
        self.connection.execute(
            "DELETE FROM provider_catalog_entries WHERE provider = ?",
            (provider,),
        )
        for entry in entries:
            self.connection.execute(
                """
                INSERT INTO provider_catalog_entries(
                    id, provider, slug, name, privacy_policy_url,
                    terms_of_service_url, status_page_url, headquarters,
                    datacenters_json, refreshed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(provider, slug) DO UPDATE SET
                    name = excluded.name,
                    privacy_policy_url = excluded.privacy_policy_url,
                    terms_of_service_url = excluded.terms_of_service_url,
                    status_page_url = excluded.status_page_url,
                    headquarters = excluded.headquarters,
                    datacenters_json = excluded.datacenters_json,
                    refreshed_at = excluded.refreshed_at,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    _new_id(),
                    provider,
                    str(entry["slug"]),
                    str(entry["name"]),
                    _optional_text(entry.get("privacy_policy_url")),
                    _optional_text(entry.get("terms_of_service_url")),
                    _optional_text(entry.get("status_page_url")),
                    _optional_text(entry.get("headquarters")),
                    _dump_json(_string_list(entry.get("datacenters"))),
                ),
            )
        self.commit()

    def list_provider_catalog_entries(
        self,
        provider: str,
    ) -> list[ProviderCatalogEntryRecord]:
        rows = self._fetch_all(
            """
            SELECT id, provider, slug, name, privacy_policy_url,
                   terms_of_service_url, status_page_url, headquarters,
                   datacenters_json, refreshed_at
            FROM provider_catalog_entries
            WHERE provider = ?
            ORDER BY lower(name), lower(slug)
            """,
            (provider,),
        )
        return [
            ProviderCatalogEntryRecord(
                id=row["id"],
                provider=row["provider"],
                slug=row["slug"],
                name=row["name"],
                privacy_policy_url=row["privacy_policy_url"],
                terms_of_service_url=row["terms_of_service_url"],
                status_page_url=row["status_page_url"],
                headquarters=row["headquarters"],
                datacenters=_load_list(row["datacenters_json"]),
                refreshed_at=row["refreshed_at"],
            )
            for row in rows
        ]

    def set_model_preference(
        self,
        *,
        task: str,
        provider: str,
        model_id: str,
        preference_id: str | None = None,
    ) -> ModelPreferenceRecord:
        existing = self._fetch_one(
            "SELECT id FROM model_preferences WHERE task = ?",
            (task,),
        )
        record = ModelPreferenceRecord(
            id=existing["id"] if existing else preference_id or _new_id(),
            task=task,
            provider=provider,
            model_id=model_id,
        )
        self.connection.execute(
            """
            INSERT INTO model_preferences(id, task, provider, model_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(task) DO UPDATE SET
                provider = excluded.provider,
                model_id = excluded.model_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (record.id, record.task, record.provider, record.model_id),
        )
        self.commit()
        return record

    def get_model_preference(self, task: str) -> ModelPreferenceRecord | None:
        row = self._fetch_one(
            """
            SELECT id, task, provider, model_id
            FROM model_preferences
            WHERE task = ?
            """,
            (task,),
        )
        return ModelPreferenceRecord(**dict(row)) if row else None

    def clear_model_preference(self, task: str) -> None:
        self.connection.execute(
            "DELETE FROM model_preferences WHERE task = ?",
            (task,),
        )
        self.commit()

    def list_model_preferences(self) -> list[ModelPreferenceRecord]:
        rows = self._fetch_all(
            """
            SELECT id, task, provider, model_id
            FROM model_preferences
            ORDER BY task
            """,
            (),
        )
        return [ModelPreferenceRecord(**dict(row)) for row in rows]

    def set_app_setting(self, key: str, value: object) -> None:
        scope, scope_id, scoped_key = _legacy_app_setting_parts(key)
        self._set_scoped_setting_json(
            scope=scope,
            scope_id=scope_id,
            key=scoped_key,
            value_json=_dump_json(value),
        )
        self.connection.execute(
            """
            INSERT INTO app_settings(key, value_json)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, _dump_json(value)),
        )
        self.commit()

    def get_app_setting(self, key: str) -> object | None:
        scope, scope_id, scoped_key = _legacy_app_setting_parts(key)
        value = self.get_scoped_setting(
            scope=scope,
            key=scoped_key,
            scope_id=scope_id or None,
        )
        if value is not None:
            return value
        row = self._fetch_one(
            "SELECT value_json FROM app_settings WHERE key = ?",
            (key,),
        )
        if row is None:
            return None
        return cast(object, json.loads(row["value_json"]))

    def set_scoped_setting(
        self,
        *,
        scope: str,
        key: str,
        value: object,
        scope_id: str | None = None,
    ) -> None:
        self._set_scoped_setting_json(
            scope=scope,
            key=key,
            scope_id=_setting_scope_id(scope, scope_id),
            value_json=_dump_json(value),
        )

    def _set_scoped_setting_json(
        self,
        *,
        scope: str,
        key: str,
        value_json: str,
        scope_id: str | None = None,
    ) -> None:
        scope_id_text = _setting_scope_id(scope, scope_id)
        self.connection.execute(
            """
            INSERT INTO scoped_settings(scope, scope_id, key, value_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scope, scope_id, key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (scope, scope_id_text, key, value_json),
        )
        self.commit()

    def get_scoped_setting(
        self,
        *,
        scope: str,
        key: str,
        scope_id: str | None = None,
    ) -> object | None:
        row = self._fetch_one(
            """
            SELECT value_json
            FROM scoped_settings
            WHERE scope = ? AND scope_id = ? AND key = ?
            """,
            (scope, _setting_scope_id(scope, scope_id), key),
        )
        if row is None:
            return None
        return cast(object, json.loads(row["value_json"]))

    def get_effective_setting(
        self,
        key: str,
        *,
        save_id: str | None = None,
        user_id: str | None = None,
        scenario_id: str | None = None,
    ) -> object | None:
        if save_id is not None:
            value = self.get_scoped_setting(scope="save", scope_id=save_id, key=key)
            if value is not None:
                return value
            save = self.get_save(save_id)
            if save is not None:
                scenario_id = scenario_id or save.scenario_id
        if scenario_id is not None:
            value = self.get_scoped_setting(
                scope="scenario",
                scope_id=scenario_id,
                key=key,
            )
            if value is not None:
                return value
        if user_id is not None:
            value = self.get_scoped_setting(scope="user", scope_id=user_id, key=key)
            if value is not None:
                return value
        return self.get_scoped_setting(scope="global", key=key)

    def list_scoped_settings(
        self,
        *,
        scope: str,
        scope_id: str | None = None,
    ) -> list[ScopedSettingRecord]:
        rows = self._fetch_all(
            """
            SELECT scope, scope_id, key, value_json, updated_at
            FROM scoped_settings
            WHERE scope = ? AND scope_id = ?
            ORDER BY key
            """,
            (scope, _setting_scope_id(scope, scope_id)),
        )
        return [_scoped_setting_from_row(row) for row in rows]

    def delete_scoped_setting(
        self,
        *,
        scope: str,
        key: str,
        scope_id: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            DELETE FROM scoped_settings
            WHERE scope = ? AND scope_id = ? AND key = ?
            """,
            (scope, _setting_scope_id(scope, scope_id), key),
        )
        self.commit()

    def delete_scoped_settings_for_save(self, save_id: str) -> None:
        self.connection.execute(
            "DELETE FROM scoped_settings WHERE scope = 'save' AND scope_id = ?",
            (save_id,),
        )
        for key in (
            _save_scenario_evolution_turn_interval_setting_key(save_id),
            _save_image_style_preset_setting_key(save_id),
        ):
            self.connection.execute("DELETE FROM app_settings WHERE key = ?", (key,))
        self.commit()

    def delete_scoped_settings_for_scenario(self, scenario_id: str) -> None:
        self.connection.execute(
            "DELETE FROM scoped_settings WHERE scope = 'scenario' AND scope_id = ?",
            (scenario_id,),
        )
        self.connection.execute(
            "DELETE FROM app_settings WHERE key = ?",
            (_scenario_template_evolution_turn_interval_setting_key(scenario_id),),
        )
        self.commit()

    def copy_save_scoped_settings(
        self,
        *,
        source_save_id: str,
        target_save_id: str,
    ) -> None:
        rows = self._fetch_all(
            """
            SELECT key, value_json
            FROM scoped_settings
            WHERE scope = 'save' AND scope_id = ?
            ORDER BY key
            """,
            (source_save_id,),
        )
        for row in rows:
            self._set_scoped_setting_json(
                scope="save",
                scope_id=target_save_id,
                key=row["key"],
                value_json=row["value_json"],
            )

    def upsert_scheduled_task(
        self,
        *,
        task_type: str,
        save_id: str | None = None,
        interval_seconds: int = 60,
        payload: dict[str, object] | None = None,
        due_now: bool = False,
        enabled: bool = True,
        task_id: str | None = None,
    ) -> ScheduledTaskRecord:
        if not task_type:
            raise ValueError("task_type is required")
        if interval_seconds < 1:
            raise ValueError("interval_seconds must be at least 1")
        payload_json = _dump_json(payload or {})
        next_run_modifier = (
            "+0 seconds" if due_now else f"+{interval_seconds} seconds"
        )
        if save_id is None:
            existing = self._fetch_one(
                """
                SELECT id FROM scheduled_tasks
                WHERE task_type = ? AND save_id IS NULL
                """,
                (task_type,),
            )
            if existing is None:
                record_id = task_id or _new_id()
                self.connection.execute(
                    """
                    INSERT INTO scheduled_tasks(
                        id, task_type, save_id, enabled, interval_seconds,
                        next_run_at, payload_json
                    )
                    VALUES (?, ?, NULL, ?, ?, datetime('now', ?), ?)
                    """,
                    (
                        record_id,
                        task_type,
                        int(enabled),
                        interval_seconds,
                        next_run_modifier,
                        payload_json,
                    ),
                )
            else:
                record_id = existing["id"]
                self.connection.execute(
                    """
                    UPDATE scheduled_tasks
                    SET enabled = ?,
                        interval_seconds = ?,
                        next_run_at = CASE
                            WHEN ? THEN CURRENT_TIMESTAMP
                            ELSE next_run_at
                        END,
                        payload_json = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        int(enabled),
                        interval_seconds,
                        int(due_now),
                        payload_json,
                        record_id,
                    ),
                )
        else:
            record_id = task_id or _new_id()
            self.connection.execute(
                """
                INSERT INTO scheduled_tasks(
                    id, task_type, save_id, enabled, interval_seconds,
                    next_run_at, payload_json
                )
                VALUES (?, ?, ?, ?, ?, datetime('now', ?), ?)
                ON CONFLICT(task_type, save_id) DO UPDATE SET
                    enabled = excluded.enabled,
                    interval_seconds = excluded.interval_seconds,
                    next_run_at = CASE
                        WHEN ? THEN CURRENT_TIMESTAMP
                        ELSE scheduled_tasks.next_run_at
                    END,
                    payload_json = excluded.payload_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    record_id,
                    task_type,
                    save_id,
                    int(enabled),
                    interval_seconds,
                    next_run_modifier,
                    payload_json,
                    int(due_now),
                ),
            )
            row = self._fetch_one(
                """
                SELECT id FROM scheduled_tasks
                WHERE task_type = ? AND save_id = ?
                """,
                (task_type, save_id),
            )
            if row is None:
                raise ValueError("Failed to upsert scheduled task")
            record_id = row["id"]
        self.commit()
        return self._get_scheduled_task(record_id)

    def list_due_scheduled_tasks(
        self,
        *,
        task_types: tuple[str, ...] = (),
        save_id: str | None | object = ...,
        limit: int = 10,
    ) -> list[ScheduledTaskRecord]:
        if limit <= 0:
            return []
        conditions = [
            "enabled = 1",
            "next_run_at <= CURRENT_TIMESTAMP",
            "(lease_until IS NULL OR lease_until <= CURRENT_TIMESTAMP)",
        ]
        params: list[object] = []
        if task_types:
            conditions.append(f"task_type IN ({_placeholders(len(task_types))})")
            params.extend(task_types)
        if save_id is not ...:
            if save_id is None:
                conditions.append("save_id IS NULL")
            else:
                conditions.append("save_id = ?")
                params.append(save_id)
        rows = self._fetch_all(
            f"""
            SELECT {_SCHEDULED_TASK_COLUMNS}
            FROM scheduled_tasks
            WHERE {' AND '.join(conditions)}
            ORDER BY next_run_at, rowid
            LIMIT ?
            """,
            (*params, limit),
        )
        return [_scheduled_task_from_row(row) for row in rows]

    def list_scheduled_tasks(
        self,
        *,
        task_types: tuple[str, ...] = (),
        save_id: str | None | object = ...,
        limit: int | None = None,
    ) -> list[ScheduledTaskRecord]:
        conditions: list[str] = []
        params: list[object] = []
        if task_types:
            conditions.append(f"task_type IN ({_placeholders(len(task_types))})")
            params.extend(task_types)
        if save_id is not ...:
            if save_id is None:
                conditions.append("save_id IS NULL")
            else:
                conditions.append("save_id = ?")
                params.append(save_id)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = ""
        if limit is not None:
            if limit <= 0:
                return []
            limit_clause = "LIMIT ?"
            params.append(limit)
        rows = self._fetch_all(
            f"""
            SELECT {_SCHEDULED_TASK_COLUMNS}
            FROM scheduled_tasks
            {where_clause}
            ORDER BY updated_at DESC, rowid DESC
            {limit_clause}
            """,
            tuple(params),
        )
        return [_scheduled_task_from_row(row) for row in rows]

    def get_scheduled_task(
        self,
        *,
        task_type: str,
        save_id: str | None = None,
    ) -> ScheduledTaskRecord | None:
        if save_id is None:
            row = self._fetch_one(
                f"""
                SELECT {_SCHEDULED_TASK_COLUMNS}
                FROM scheduled_tasks
                WHERE task_type = ? AND save_id IS NULL
                """,
                (task_type,),
            )
        else:
            row = self._fetch_one(
                f"""
                SELECT {_SCHEDULED_TASK_COLUMNS}
                FROM scheduled_tasks
                WHERE task_type = ? AND save_id = ?
                """,
                (task_type, save_id),
            )
        return _scheduled_task_from_row(row) if row is not None else None

    def lease_scheduled_task(
        self,
        task_id: str,
        *,
        lease_seconds: int = 300,
    ) -> ScheduledTaskRecord | None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be at least 1")
        cursor = self.connection.execute(
            """
            UPDATE scheduled_tasks
            SET lease_until = datetime('now', ?),
                last_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND enabled = 1
              AND next_run_at <= CURRENT_TIMESTAMP
              AND (lease_until IS NULL OR lease_until <= CURRENT_TIMESTAMP)
            """,
            (f"+{lease_seconds} seconds", task_id),
        )
        self.commit()
        if cursor.rowcount == 0:
            return None
        return self._get_scheduled_task(task_id)

    def complete_scheduled_task(
        self,
        task_id: str,
        *,
        succeeded: bool,
        result: dict[str, object] | None = None,
        error: str | None = None,
        last_job_id: str | None = None,
        next_run_after_seconds: int | None = None,
    ) -> ScheduledTaskRecord:
        if next_run_after_seconds is not None and next_run_after_seconds < 1:
            raise ValueError("next_run_after_seconds must be at least 1")
        self.connection.execute(
            """
            UPDATE scheduled_tasks
            SET lease_until = NULL,
                last_completed_at = CURRENT_TIMESTAMP,
                last_job_id = COALESCE(?, last_job_id),
                failure_count = CASE WHEN ? THEN 0 ELSE failure_count + 1 END,
                result_json = ?,
                error = ?,
                next_run_at = datetime(
                    'now',
                    '+' || CAST(COALESCE(?, interval_seconds) AS TEXT) || ' seconds'
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                last_job_id,
                int(succeeded),
                _dump_json(result) if result is not None else None,
                redact_text(error),
                next_run_after_seconds,
                task_id,
            ),
        )
        self.commit()
        return self._get_scheduled_task(task_id)

    def _get_scheduled_task(self, task_id: str) -> ScheduledTaskRecord:
        row = self._fetch_one(
            f"""
            SELECT {_SCHEDULED_TASK_COLUMNS}
            FROM scheduled_tasks
            WHERE id = ?
            """,
            (task_id,),
        )
        if row is None:
            raise ValueError(f"Unknown scheduled task: {task_id}")
        return _scheduled_task_from_row(row)

    def list_failed_jobs(self) -> list[JobRecord]:
        rows = self._fetch_all(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs
            WHERE status IN ('failed', 'cancelled')
            ORDER BY created_at, rowid
            """,
            (),
        )
        return [_job_from_row(row) for row in rows]

    def list_jobs_by_status(self, statuses: tuple[str, ...]) -> list[JobRecord]:
        if not statuses:
            return []
        for status in statuses:
            _validate_job_status(status)
        rows = self._fetch_all(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs
            WHERE status IN ({_placeholders(len(statuses))})
            ORDER BY rowid
            """,
            tuple(statuses),
        )
        return [_job_from_row(row) for row in rows]

    def has_matching_job(
        self,
        *,
        statuses: tuple[str, ...],
        types: tuple[str, ...] = (),
        save_id: str | None | object = ...,
    ) -> bool:
        if not statuses:
            return False
        for status in statuses:
            _validate_job_status(status)
        conditions = [f"status IN ({_placeholders(len(statuses))})"]
        params: list[object] = list(statuses)
        if types:
            conditions.append(f"type IN ({_placeholders(len(types))})")
            params.extend(types)
        if save_id is not ...:
            if save_id is None:
                conditions.append("save_id IS NULL")
            else:
                conditions.append("save_id = ?")
                params.append(save_id)
        row = self._fetch_one(
            f"""
            SELECT 1
            FROM jobs
            WHERE {' AND '.join(conditions)}
            LIMIT 1
            """,
            tuple(params),
        )
        return row is not None

    def list_job_save_ids(
        self,
        *,
        statuses: tuple[str, ...],
        types: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        if not statuses:
            return ()
        for status in statuses:
            _validate_job_status(status)
        conditions = [
            "save_id IS NOT NULL",
            f"status IN ({_placeholders(len(statuses))})",
        ]
        params: list[object] = list(statuses)
        if types:
            conditions.append(f"type IN ({_placeholders(len(types))})")
            params.extend(types)
        rows = self._fetch_all(
            f"""
            SELECT DISTINCT save_id
            FROM jobs
            WHERE {' AND '.join(conditions)}
            ORDER BY save_id
            """,
            tuple(params),
        )
        return tuple(str(row["save_id"]) for row in rows)

    def list_recent_jobs(
        self,
        *,
        save_id: str | None = None,
        types: tuple[str, ...] = (),
        statuses: tuple[str, ...] = (),
        seconds: int = 900,
        limit: int = 50,
    ) -> list[JobRecord]:
        if limit <= 0:
            return []
        for status in statuses:
            _validate_job_status(status)
        conditions: list[str] = []
        params: list[object] = []
        if save_id is not None:
            conditions.append("save_id = ?")
            params.append(save_id)
        if types:
            conditions.append(f"type IN ({_placeholders(len(types))})")
            params.extend(types)
        if statuses:
            conditions.append(f"status IN ({_placeholders(len(statuses))})")
            params.extend(statuses)
        if seconds > 0:
            conditions.append(
                "julianday('now') - "
                "julianday(COALESCE(completed_at, started_at, created_at)) <= ?"
            )
            params.append(seconds / 86400.0)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._fetch_all(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs
            {where_clause}
            ORDER BY COALESCE(completed_at, started_at, created_at) DESC, rowid DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return [_job_from_row(row) for row in rows]

    def list_terminal_jobs(
        self,
        *,
        statuses: tuple[str, ...] | None = None,
        save_id: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> list[JobRecord]:
        if limit <= 0:
            return []
        statuses = statuses or tuple(sorted(_JOB_TERMINAL_STATUSES))
        if not statuses:
            return []
        for status in statuses:
            _validate_job_status(status)
            if status not in _JOB_TERMINAL_STATUSES:
                raise ValueError(f"Unsupported terminal job status: {status}")
        conditions = [f"status IN ({_placeholders(len(statuses))})"]
        params: list[object] = list(statuses)
        if save_id is not None:
            conditions.append("save_id = ?")
            params.append(save_id)
        if since is not None:
            conditions.append(
                "julianday(COALESCE(completed_at, started_at, created_at)) "
                ">= julianday(?)"
            )
            params.append(since)
        rows = self._fetch_all(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs
            WHERE {' AND '.join(conditions)}
            ORDER BY COALESCE(completed_at, started_at, created_at) DESC, rowid DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return [_job_from_row(row) for row in rows]

    def get_persisted_job(self, job_id: str) -> JobRecord | None:
        row = self._fetch_one(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        )
        return _job_from_row(row) if row is not None else None

    def count_job_steps_by_job_id(
        self,
        job_ids: tuple[str, ...],
    ) -> dict[str, int]:
        if not job_ids:
            return {}
        rows = self._fetch_all(
            f"""
            SELECT job_id, COUNT(*) AS step_count
            FROM job_steps
            WHERE job_id IN ({_placeholders(len(job_ids))})
            GROUP BY job_id
            """,
            job_ids,
        )
        return {str(row["job_id"]): int(row["step_count"]) for row in rows}

    def find_chat_completion_job_for_narrator_message(
        self,
        narrator_message_id: str,
    ) -> JobRecord | None:
        rows = self._fetch_all(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs
            WHERE type = 'chat_completion'
              AND result_json IS NOT NULL
            ORDER BY completed_at DESC, rowid DESC
            """,
            (),
        )
        for row in rows:
            job = _job_from_row(row)
            if not isinstance(job.result, dict):
                continue
            if job.result.get("narrator_message_id") == narrator_message_id:
                return job
        return None

    def list_provider_configs(self) -> list[ProviderConfigRecord]:
        rows = self._fetch_all(
            """
            SELECT id, provider, enabled, has_api_key, last_model_refresh_at,
                   last_error
            FROM provider_configs
            ORDER BY provider
            """,
            (),
        )
        return [
            ProviderConfigRecord(
                id=row["id"],
                provider=row["provider"],
                enabled=bool(row["enabled"]),
                has_api_key=bool(row["has_api_key"]),
                last_model_refresh_at=row["last_model_refresh_at"],
                last_error=row["last_error"],
            )
            for row in rows
        ]

    def create_job(
        self,
        *,
        type: str,
        status: str,
        payload: dict[str, object],
        save_id: str | None = None,
        creator_user_id: str | None = None,
        job_id: str | None = None,
    ) -> JobRecord:
        if creator_user_id is not None and self.get_user(creator_user_id) is None:
            raise ValueError(f"Unknown user id: {creator_user_id}")
        _validate_job_initial_status(status)
        record = JobRecord(
            id=job_id or _new_id(),
            save_id=save_id,
            creator_user_id=creator_user_id,
            type=type,
            status=status,
            payload=payload,
            result=None,
            error=None,
            started_at=None,
            completed_at=None,
            duration_ms=None,
        )
        self.connection.execute(
            """
            INSERT INTO jobs(
                id, save_id, creator_user_id, type, status, payload_json, started_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?,
                CASE WHEN ? = 'running' THEN CURRENT_TIMESTAMP END
            )
            """,
            (
                record.id,
                record.save_id,
                record.creator_user_id,
                record.type,
                record.status,
                _dump_json(record.payload),
                record.status,
            ),
        )
        self.commit()
        return self._get_job(record.id)

    def start_job(self, job_id: str) -> JobRecord:
        existing = self._get_job(job_id)
        if existing.status != "queued":
            raise ValueError(
                f"Cannot start job {job_id} from status: {existing.status}"
            )
        self.connection.execute(
            """
            UPDATE jobs
            SET status = 'running',
                started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                completed_at = NULL,
                duration_ms = NULL,
                error = NULL
            WHERE id = ?
            """,
            (job_id,),
        )
        self.commit()
        return self._get_job(job_id)

    def cancel_job(
        self,
        job_id: str,
        *,
        error: str | None = None,
        result: dict[str, object] | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> JobRecord:
        existing = self._get_job(job_id)
        if existing.status in _JOB_TERMINAL_STATUSES:
            raise ValueError(
                f"Cannot cancel job {job_id} from status: {existing.status}"
            )
        self.connection.execute(
            """
            UPDATE jobs
            SET status = 'cancelled', result_json = ?, error = ?,
                diagnostics_json = COALESCE(?, diagnostics_json),
                completed_at = CURRENT_TIMESTAMP,
                duration_ms = CASE
                    WHEN started_at IS NULL THEN NULL
                    ELSE MAX(
                        0,
                        CAST(ROUND(
                            (julianday(CURRENT_TIMESTAMP) - julianday(started_at))
                            * 86400000
                        ) AS INTEGER)
                    )
                END
            WHERE id = ?
            """,
            (
                _dump_json(result) if result is not None else None,
                redact_text(error),
                _dump_json(diagnostics) if diagnostics is not None else None,
                job_id,
            ),
        )
        self.commit()
        return self._get_job(job_id)

    def cancel_stale_jobs(
        self,
        *,
        statuses: tuple[str, ...] = ("queued", "running"),
        error: str,
    ) -> list[JobRecord]:
        jobs = self.list_jobs_by_status(statuses)
        for job in jobs:
            self.cancel_job(
                job.id,
                error=error,
                result={
                    "previous_status": job.status,
                    "recovered_on_startup": True,
                },
            )
        return [self._get_job(job.id) for job in jobs]

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, object] | None = None,
        error: str | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> JobRecord:
        _validate_job_update_status(status)
        existing = self._get_job(job_id)
        if existing.status in _JOB_TERMINAL_STATUSES:
            raise ValueError(
                f"Cannot update terminal job {job_id} from status: {existing.status}"
            )
        self.connection.execute(
            """
            UPDATE jobs
            SET status = ?, result_json = ?, error = ?,
                diagnostics_json = COALESCE(?, diagnostics_json),
                completed_at = CURRENT_TIMESTAMP,
                duration_ms = CASE
                    WHEN started_at IS NULL THEN NULL
                    ELSE MAX(
                        0,
                        CAST(ROUND(
                            (julianday(CURRENT_TIMESTAMP) - julianday(started_at))
                            * 86400000
                        ) AS INTEGER)
                    )
                END
            WHERE id = ?
            """,
            (
                status,
                _dump_json(result) if result is not None else None,
                redact_text(error),
                _dump_json(diagnostics) if diagnostics is not None else None,
                job_id,
            ),
        )
        self.commit()
        return self._get_job(job_id)

    def set_job_diagnostics(
        self,
        job_id: str,
        diagnostics: dict[str, object] | None,
    ) -> JobRecord:
        self._get_job(job_id)
        self.connection.execute(
            """
            UPDATE jobs
            SET diagnostics_json = ?
            WHERE id = ?
            """,
            (
                _dump_json(diagnostics) if diagnostics is not None else None,
                job_id,
            ),
        )
        self.commit()
        return self._get_job(job_id)

    def update_queued_job_payload(
        self,
        job_id: str,
        *,
        payload: dict[str, object],
    ) -> JobRecord:
        existing = self._get_job(job_id)
        if existing.status != "queued":
            raise ValueError(
                f"Cannot update queued payload for job {job_id} "
                f"from status: {existing.status}"
            )
        self.connection.execute(
            """
            UPDATE jobs
            SET payload_json = ?
            WHERE id = ?
            """,
            (_dump_json(payload), job_id),
        )
        self.commit()
        return self._get_job(job_id)

    def record_job_step(
        self,
        *,
        job_id: str,
        name: str,
        status: str,
        provider: str | None = None,
        model: str | None = None,
        task: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
        step_id: str | None = None,
    ) -> JobStepRecord:
        _validate_job_step_status(status)
        normalized_duration_ms = (
            None if duration_ms is None else max(0, int(duration_ms))
        )
        record_id = step_id or _new_id()
        self.connection.execute(
            """
            INSERT INTO job_steps(
                id, job_id, name, status, provider, model, task, started_at,
                completed_at, duration_ms, error, metadata_json
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP),
                COALESCE(?, CURRENT_TIMESTAMP), ?, ?, ?
            )
            """,
            (
                record_id,
                job_id,
                name,
                status,
                provider,
                model,
                task,
                started_at,
                completed_at,
                normalized_duration_ms,
                redact_text(error),
                _dump_json(_safe_job_step_metadata(metadata)),
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, job_id, name, status, provider, model, task, started_at,
                   completed_at, duration_ms, error, metadata_json
            FROM job_steps
            WHERE id = ?
            """,
            (record_id,),
        )
        if row is None:
            raise ValueError(f"Unknown job step id: {record_id}")
        return _job_step_from_row(row)

    def list_job_steps(self, job_id: str | None = None) -> list[JobStepRecord]:
        if job_id is None:
            where_clause = ""
            params: tuple[object, ...] = ()
        else:
            where_clause = "WHERE job_id = ?"
            params = (job_id,)
        rows = self._fetch_all(
            f"""
            SELECT id, job_id, name, status, provider, model, task, started_at,
                   completed_at, duration_ms, error, metadata_json
            FROM job_steps
            {where_clause}
            ORDER BY completed_at, rowid
            """,
            params,
        )
        return [_job_step_from_row(row) for row in rows]

    def runtime_job_averages(
        self,
        *,
        save_id: str | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[RuntimePerformanceRecord]:
        conditions = ["status IN ('succeeded', 'failed', 'cancelled')"]
        params: list[object] = []
        if save_id is not None:
            conditions.append("save_id = ?")
            params.append(save_id)
        if since is not None:
            conditions.append("julianday(completed_at) >= julianday(?)")
            params.append(since)
        rows = self._fetch_all(
            f"""
            SELECT type AS job_type, status, duration_ms, completed_at,
                   CASE
                       WHEN started_at IS NULL OR created_at IS NULL THEN NULL
                       ELSE MAX(
                           0,
                           CAST(ROUND(
                               (julianday(started_at) - julianday(created_at))
                               * 86400000
                           ) AS INTEGER)
                       )
                   END AS queue_wait_ms
            FROM jobs
            WHERE {' AND '.join(conditions)}
            ORDER BY completed_at, rowid
            """,
            tuple(params),
        )
        return _runtime_performance_records(
            rows,
            key_fields=("job_type",),
            limit=limit,
        )

    def runtime_step_averages(
        self,
        *,
        save_id: str | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[RuntimePerformanceRecord]:
        join_clause = ""
        conditions: list[str] = []
        params: list[object] = []
        if save_id is not None:
            join_clause = "JOIN jobs j ON j.id = s.job_id"
            conditions.append("j.save_id = ?")
            params.append(save_id)
        if since is not None:
            conditions.append("julianday(s.completed_at) >= julianday(?)")
            params.append(since)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._fetch_all(
            f"""
            SELECT s.name AS step_name, s.status, s.duration_ms, s.completed_at
            FROM job_steps s
            {join_clause}
            {where_clause}
            ORDER BY s.completed_at, s.rowid
            """,
            tuple(params),
        )
        return _runtime_performance_records(
            rows,
            key_fields=("step_name",),
            limit=limit,
        )

    def runtime_model_averages(
        self,
        *,
        save_id: str | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[RuntimePerformanceRecord]:
        join_clause = ""
        conditions = ["(s.provider IS NOT NULL OR s.model IS NOT NULL)"]
        params: list[object] = []
        if save_id is not None:
            join_clause = "JOIN jobs j ON j.id = s.job_id"
            conditions.append("j.save_id = ?")
            params.append(save_id)
        if since is not None:
            conditions.append("julianday(s.completed_at) >= julianday(?)")
            params.append(since)
        rows = self._fetch_all(
            f"""
            SELECT s.provider, s.model, s.task, s.status, s.duration_ms, s.completed_at
            FROM job_steps s
            {join_clause}
            WHERE {' AND '.join(conditions)}
            ORDER BY s.completed_at, s.rowid
            """,
            tuple(params),
        )
        return _runtime_performance_records(
            rows,
            key_fields=("provider", "model", "task"),
            limit=limit,
        )

    def runtime_slowest_recent_operations(
        self,
        *,
        save_id: str | None = None,
        since: str | None = None,
        limit: int = 10,
    ) -> list[RuntimeSlowOperationRecord]:
        if limit <= 0:
            return []
        conditions = [
            "status IN ('succeeded', 'failed', 'cancelled')",
            "duration_ms IS NOT NULL",
        ]
        params: list[object] = []
        if save_id is not None:
            conditions.append("save_id = ?")
            params.append(save_id)
        if since is not None:
            conditions.append("julianday(completed_at) >= julianday(?)")
            params.append(since)
        rows = self._fetch_all(
            f"""
            SELECT {_JOB_COLUMNS},
                   CASE
                       WHEN started_at IS NULL OR created_at IS NULL THEN NULL
                       ELSE MAX(
                           0,
                           CAST(ROUND(
                               (julianday(started_at) - julianday(created_at))
                               * 86400000
                           ) AS INTEGER)
                       )
                   END AS queue_wait_ms
            FROM jobs
            WHERE {' AND '.join(conditions)}
            ORDER BY duration_ms DESC, completed_at DESC, rowid DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        records: list[RuntimeSlowOperationRecord] = []
        for row in rows:
            steps = self.list_job_steps(str(row["id"]))
            slowest_step = max(
                (step for step in steps if step.duration_ms is not None),
                key=lambda step: step.duration_ms or 0,
                default=None,
            )
            records.append(
                RuntimeSlowOperationRecord(
                    job_id=row["id"],
                    save_id=row["save_id"],
                    job_type=row["type"],
                    status=row["status"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    duration_ms=row["duration_ms"],
                    queue_wait_ms=row["queue_wait_ms"],
                    slowest_step_name=(
                        slowest_step.name if slowest_step is not None else None
                    ),
                    slowest_step_duration_ms=(
                        slowest_step.duration_ms
                        if slowest_step is not None
                        else None
                    ),
                    provider=(
                        slowest_step.provider if slowest_step is not None else None
                    ),
                    model=slowest_step.model if slowest_step is not None else None,
                    task=slowest_step.task if slowest_step is not None else None,
                )
            )
        return records

    def _get_job(self, job_id: str) -> JobRecord:
        row = self._fetch_one(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        )
        if row is None:
            raise ValueError(f"Unknown job id: {job_id}")
        return _job_from_row(row)

    def create_media_asset(
        self,
        *,
        save_id: str,
        type: str,
        path: str,
        prompt: str,
        provider: str,
        model: str,
        status: str,
        source_message_id: str | None = None,
        thumbnail_path: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, object] | None = None,
        source_media_asset_id: str | None = None,
        asset_id: str | None = None,
    ) -> MediaAssetRecord:
        normalized_mime_type = _media_mime_type(type=type, mime_type=mime_type)
        _validate_media_asset_path(path)
        if thumbnail_path is not None:
            _validate_media_asset_path(thumbnail_path)
        record = MediaAssetRecord(
            id=asset_id or _new_id(),
            save_id=save_id,
            source_message_id=source_message_id,
            type=type,
            path=path,
            thumbnail_path=thumbnail_path,
            prompt=prompt,
            provider=provider,
            model=model,
            status=status,
            mime_type=normalized_mime_type,
            metadata_json=_dump_json(metadata or {}),
            source_media_asset_id=source_media_asset_id,
        )
        self.connection.execute(
            """
            INSERT INTO media_assets(
                id, save_id, source_message_id, type, path, thumbnail_path,
                prompt, provider, model, status, mime_type, metadata_json,
                source_media_asset_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.save_id,
                record.source_message_id,
                record.type,
                record.path,
                record.thumbnail_path,
                record.prompt,
                record.provider,
                record.model,
                record.status,
                record.mime_type,
                record.metadata_json,
                record.source_media_asset_id,
            ),
        )
        self.commit()
        row = self._fetch_one(
            """
            SELECT id, save_id, source_message_id, type, path, thumbnail_path,
                   prompt, provider, model, status, mime_type, metadata_json,
                   source_media_asset_id, created_at, archived_at
            FROM media_assets
            WHERE id = ?
            """,
            (record.id,),
        )
        return MediaAssetRecord(**dict(row)) if row else record

    def list_media_assets(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[MediaAssetRecord]:
        order_sql = (
            "ORDER BY created_at DESC, rowid DESC"
            if limit is not None
            else "ORDER BY created_at, rowid"
        )
        limit_sql = "LIMIT ?" if limit is not None else ""
        params: tuple[object, ...] = (
            (save_id, max(0, limit))
            if limit is not None
            else (save_id,)
        )
        rows = self._fetch_all(
            f"""
            SELECT id, save_id, source_message_id, type, path, thumbnail_path,
                   prompt, provider, model, status, mime_type, metadata_json,
                   source_media_asset_id, created_at, archived_at
            FROM media_assets
            WHERE save_id = ? AND archived_at IS NULL
            {order_sql}
            {limit_sql}
            """,
            params,
        )
        if limit is not None:
            rows.reverse()
        return [MediaAssetRecord(**dict(row)) for row in rows]

    def get_media_asset(
        self,
        *,
        save_id: str,
        media_asset_id: str,
    ) -> MediaAssetRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, source_message_id, type, path, thumbnail_path,
                   prompt, provider, model, status, mime_type, metadata_json,
                   source_media_asset_id, created_at, archived_at
            FROM media_assets
            WHERE save_id = ? AND id = ? AND archived_at IS NULL
            """,
            (save_id, media_asset_id),
        )
        return MediaAssetRecord(**dict(row)) if row else None

    def replace_media_asset_source_references(
        self,
        *,
        save_id: str,
        old_media_asset_id: str,
        new_media_asset_id: str,
    ) -> int:
        media = self._fetch_one(
            """
            SELECT id
            FROM media_assets
            WHERE id = ? AND save_id = ? AND archived_at IS NULL
            """,
            (new_media_asset_id, save_id),
        )
        if media is None:
            raise ValueError(f"Unknown media asset id: {new_media_asset_id}")

        rows = self._fetch_all(
            """
            SELECT id, source_media_asset_id, metadata_json
            FROM media_assets
            WHERE save_id = ? AND archived_at IS NULL
            """,
            (save_id,),
        )
        updated_count = 0
        for row in rows:
            source_media_asset_id = row["source_media_asset_id"]
            next_source_media_asset_id = (
                new_media_asset_id
                if source_media_asset_id == old_media_asset_id
                else source_media_asset_id
            )
            try:
                metadata = _load_object(row["metadata_json"] or "{}")
            except (TypeError, ValueError):
                metadata = {}
            metadata_changed = _replace_media_source_metadata_refs(
                metadata,
                old_media_asset_id=old_media_asset_id,
                new_media_asset_id=new_media_asset_id,
            )
            if (
                next_source_media_asset_id == source_media_asset_id
                and not metadata_changed
            ):
                continue
            self.connection.execute(
                """
                UPDATE media_assets
                SET source_media_asset_id = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (next_source_media_asset_id, _dump_json(metadata), row["id"]),
            )
            updated_count += 1
        self.commit()
        return updated_count

    def list_all_media_assets(self, save_id: str) -> list[MediaAssetRecord]:
        rows = self._fetch_all(
            """
            SELECT id, save_id, source_message_id, type, path, thumbnail_path,
                   prompt, provider, model, status, mime_type, metadata_json,
                   source_media_asset_id, created_at, archived_at
            FROM media_assets
            WHERE save_id = ?
            ORDER BY created_at, rowid
            """,
            (save_id,),
        )
        return [MediaAssetRecord(**dict(row)) for row in rows]

    def archive_media_asset(
        self,
        *,
        save_id: str,
        media_asset_id: str,
    ) -> MediaAssetRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, source_message_id, type, path, thumbnail_path,
                   prompt, provider, model, status, mime_type, metadata_json,
                   source_media_asset_id, created_at, archived_at
            FROM media_assets
            WHERE id = ? AND save_id = ? AND archived_at IS NULL
            """,
            (media_asset_id, save_id),
        )
        if row is None:
            return None
        self.connection.execute(
            """
            WITH RECURSIVE archived_media(id) AS (
                SELECT id
                FROM media_assets
                WHERE id = ? AND save_id = ? AND archived_at IS NULL
                UNION
                SELECT child.id
                FROM media_assets child
                JOIN archived_media parent ON child.source_media_asset_id = parent.id
                WHERE child.save_id = ? AND child.archived_at IS NULL
            )
            UPDATE media_assets
            SET archived_at = CURRENT_TIMESTAMP
            WHERE id IN (SELECT id FROM archived_media)
            """,
            (media_asset_id, save_id, save_id),
        )
        self.commit()
        archived = self._fetch_one(
            """
            SELECT id, save_id, source_message_id, type, path, thumbnail_path,
                   prompt, provider, model, status, mime_type, metadata_json,
                   source_media_asset_id, created_at, archived_at
            FROM media_assets
            WHERE id = ? AND save_id = ?
            """,
            (media_asset_id, save_id),
        )
        return (
            MediaAssetRecord(**dict(archived))
            if archived
            else MediaAssetRecord(**dict(row))
        )

    def archive_media_asset_only(
        self,
        *,
        save_id: str,
        media_asset_id: str,
    ) -> MediaAssetRecord | None:
        row = self._fetch_one(
            """
            SELECT id, save_id, source_message_id, type, path, thumbnail_path,
                   prompt, provider, model, status, mime_type, metadata_json,
                   source_media_asset_id, created_at, archived_at
            FROM media_assets
            WHERE id = ? AND save_id = ? AND archived_at IS NULL
            """,
            (media_asset_id, save_id),
        )
        if row is None:
            return None
        self.connection.execute(
            """
            UPDATE media_assets
            SET archived_at = CURRENT_TIMESTAMP
            WHERE id = ? AND save_id = ? AND archived_at IS NULL
            """,
            (media_asset_id, save_id),
        )
        self.commit()
        archived = self._fetch_one(
            """
            SELECT id, save_id, source_message_id, type, path, thumbnail_path,
                   prompt, provider, model, status, mime_type, metadata_json,
                   source_media_asset_id, created_at, archived_at
            FROM media_assets
            WHERE id = ? AND save_id = ?
            """,
            (media_asset_id, save_id),
        )
        return (
            MediaAssetRecord(**dict(archived))
            if archived
            else MediaAssetRecord(**dict(row))
        )

    def archive_media_assets_for_messages(
        self,
        *,
        save_id: str,
        message_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        if not message_ids:
            return frozenset()
        ordered_message_ids = tuple(sorted(message_ids))
        roots_sql = _placeholders(len(ordered_message_ids))
        root_ids = [
            str(row["id"])
            for row in self._fetch_all(
                f"""
                WITH RECURSIVE archived_media(id) AS (
                    SELECT id
                    FROM media_assets
                    WHERE save_id = ?
                      AND archived_at IS NULL
                      AND source_message_id IN ({roots_sql})
                    UNION
                    SELECT child.id
                    FROM media_assets child
                    JOIN archived_media parent
                      ON child.source_media_asset_id = parent.id
                    WHERE child.save_id = ?
                      AND child.archived_at IS NULL
                )
                SELECT id
                FROM archived_media
                """,
                (save_id, *ordered_message_ids, save_id),
            )
        ]
        if not root_ids:
            return frozenset()
        self.connection.execute(
            f"""
            UPDATE media_assets
            SET archived_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND id IN ({_placeholders(len(root_ids))})
            """,
            (save_id, *root_ids),
        )
        self.commit()
        return frozenset(root_ids)

    def _fetch_one(
        self,
        query: str,
        params: tuple[Any, ...],
    ) -> sqlite3.Row | None:
        row = self.connection.execute(query, params).fetchone()
        return cast(sqlite3.Row | None, row)

    def _validate_location_reference(
        self,
        *,
        save_id: str,
        location_id: str | None,
        field_name: str,
    ) -> None:
        if location_id is None:
            return
        row = self._fetch_one(
            """
            SELECT 1
            FROM locations
            WHERE save_id = ? AND id = ?
            """,
            (save_id, location_id),
        )
        if row is None:
            raise ValueError(f"{field_name} must reference a location in the same save")

    def _validate_character_reference(
        self,
        *,
        save_id: str,
        character_id: str,
        field_name: str,
    ) -> None:
        row = self._fetch_one(
            """
            SELECT 1
            FROM characters
            WHERE save_id = ? AND id = ? AND archived_at IS NULL
            """,
            (save_id, character_id),
        )
        if row is None:
            raise ValueError(
                f"{field_name} must reference a character in the same save"
            )

    def _ensure_character_text_thread_participant(
        self,
        *,
        save_id: str,
        thread_id: str,
        character_id: str,
        ordinal: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO character_text_thread_participants(
                id, save_id, thread_id, character_id, ordinal
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(thread_id, character_id) DO UPDATE SET
                ordinal = excluded.ordinal,
                archived_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            """,
            (_new_id(), save_id, thread_id, character_id, max(0, int(ordinal))),
        )

    def _character_text_thread_participant_ids(
        self,
        *,
        save_id: str,
        thread_id: str,
    ) -> frozenset[str]:
        rows = self._fetch_all(
            """
            SELECT character_id
            FROM character_text_thread_participants
            WHERE save_id = ? AND thread_id = ? AND archived_at IS NULL
            """,
            (save_id, thread_id),
        )
        return frozenset(str(row["character_id"]) for row in rows)

    def _resolve_character_text_sender_character_id(
        self,
        *,
        save_id: str,
        thread: CharacterTextThreadRecord,
        character_id: str | None,
        sender: str,
        sender_character_id: str | None,
    ) -> str | None:
        normalized_sender_character_id = _blank_to_none(sender_character_id)
        if normalized_sender_character_id is not None:
            self._validate_character_reference(
                save_id=save_id,
                character_id=normalized_sender_character_id,
                field_name="sender_character_id",
            )

        if thread.kind == "direct":
            if thread.character_id is None or character_id != thread.character_id:
                raise ValueError(
                    "character text message character does not match thread"
                )
            if sender == "character":
                resolved = normalized_sender_character_id or thread.character_id
                if resolved != thread.character_id:
                    raise ValueError(
                        "direct character text sender must match thread character"
                    )
                return resolved
            if normalized_sender_character_id is not None:
                return normalized_sender_character_id
            return self._player_character_id(save_id)

        if thread.kind == "group":
            participant_ids = self._character_text_thread_participant_ids(
                save_id=save_id,
                thread_id=thread.id,
            )
            if sender == "character":
                group_resolved = normalized_sender_character_id or character_id
                if group_resolved is None:
                    raise ValueError("group character text sender is required")
                if group_resolved not in participant_ids:
                    raise ValueError(
                        "group character text sender must be a thread participant"
                    )
                if character_id is not None and character_id != group_resolved:
                    raise ValueError(
                        "group character text message character must match sender"
                    )
                return group_resolved
            if character_id is not None:
                raise ValueError("group player text messages must not target one NPC")
            if normalized_sender_character_id is not None:
                return normalized_sender_character_id
            return self._player_character_id(save_id)

        raise ValueError(f"Unsupported character text thread kind: {thread.kind}")

    def _fetch_all(
        self,
        query: str,
        params: tuple[Any, ...],
    ) -> list[sqlite3.Row]:
        return list(self.connection.execute(query, params))


class BragiRepository(PersistenceRepositories):
    def __init__(self, database_path: Path | str) -> None:
        path = Path(database_path)
        migrate_database(path)
        super().__init__(sqlite3.connect(path))


def _user_from_row(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        id=row["id"],
        username=row["username"],
        username_normalized=row["username_normalized"],
        role=row["role"],
        password_hash=row["password_hash"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _user_session_from_row(row: sqlite3.Row) -> UserSessionRecord:
    return UserSessionRecord(
        id=row["id"],
        user_id=row["user_id"],
        token_hash=row["token_hash"],
        expires_at=row["expires_at"],
        revoked_at=row["revoked_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _save_from_row(row: sqlite3.Row) -> SaveRecord:
    return SaveRecord(
        id=row["id"],
        scenario_id=row["scenario_id"],
        title=row["title"],
        active=bool(row["active"]),
        scenario_title=_row_value(row, "scenario_title"),
        custom_instructions=row["custom_instructions"],
        owner_user_id=row["owner_user_id"],
        created_at=_row_value(row, "created_at"),
        updated_at=_row_value(row, "updated_at"),
        last_opened_at=_row_value(row, "last_opened_at"),
    )


def _row_value(row: sqlite3.Row, key: str) -> str | None:
    if key not in row.keys():
        return None
    value = row[key]
    return value if isinstance(value, str) else None


def _normalized_username(username: str) -> str:
    normalized = username.strip().casefold()
    if not normalized:
        raise ValueError("username is required")
    return normalized


def _validate_user_role(role: str) -> None:
    if role not in USER_ROLES:
        raise ValueError(f"Unknown user role: {role}")


def _validate_user_status(status: str) -> None:
    if status not in USER_STATUSES:
        raise ValueError(f"Unknown user status: {status}")


def _timestamp_text(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sqlite_error_mentions(error: sqlite3.IntegrityError, text: str) -> bool:
    return text in str(error)


def _new_id() -> str:
    return uuid4().hex


def canonical_claim_fingerprint(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    canonical = "".join(
        character if character.isalnum() else " "
        for character in text
    )
    canonical = " ".join(canonical.split())
    return sha256(canonical.encode("utf-8")).hexdigest() if canonical else ""


def _transaction_savepoint_name(depth: int) -> str:
    return f"bragi_tx_{depth}"


def _chat_history_message_filter_clause(selected_filter: str) -> str:
    if selected_filter == "player":
        return "AND role = 'player'"
    if selected_filter == "narrator_character":
        return "AND role NOT IN ('player', 'system')"
    if selected_filter == "with_images":
        return """
        AND EXISTS (
            SELECT 1
            FROM media_assets
            WHERE media_assets.save_id = messages.save_id
              AND media_assets.source_message_id = messages.id
              AND media_assets.type = 'image'
              AND media_assets.archived_at IS NULL
        )
        """
    return ""


def _placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _normalized_graph_target_type(value: str) -> str:
    normalized = value.strip().casefold()
    return "world_state" if normalized in {"state", "world_state"} else normalized


def _fts_query_from_terms(
    terms: set[str] | frozenset[str] | list[str] | tuple[str, ...],
    *,
    match_all: bool = False,
) -> str:
    normalized: list[str] = []
    for value in terms:
        for term in _unicode_search_terms(str(value)):
            normalized.append(term)
            if term.endswith("'s") and len(term) > 3:
                normalized.append(term[:-2])
    unique_terms = sorted(dict.fromkeys(normalized))
    quoted_terms = []
    for term in unique_terms:
        escaped = term.replace('"', '""')
        quoted_terms.append(f'"{escaped}"*')
    return (" AND " if match_all else " OR ").join(quoted_terms)


def _fts_query_from_exact_phrases(phrases: tuple[str, ...]) -> str:
    quoted_phrases: list[str] = []
    for phrase in phrases[:MAX_CONTEXT_EXACT_PHRASES]:
        terms = unicode_word_terms(str(phrase)[:512])[:64]
        if len(terms) < 2:
            continue
        escaped = " ".join(terms).replace('"', '""')
        quoted_phrases.append(f'"{escaped}"')
    return " OR ".join(dict.fromkeys(quoted_phrases))


def _unicode_search_terms(value: str) -> tuple[str, ...]:
    return unicode_word_terms(value)


def _bounded_repository_search_terms(
    terms: set[str] | frozenset[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    if limit <= 0:
        return ()
    ordered = sorted(terms, key=lambda term: (-len(term), term))
    non_ascii = [
        term
        for term in ordered
        if any(ord(character) > 127 for character in term)
    ]
    non_ascii_set = set(non_ascii)
    ascii_terms = [term for term in ordered if term not in non_ascii_set]
    if not non_ascii or not ascii_terms:
        return tuple(ordered[:limit])
    reserve = limit // 2
    selected = [*non_ascii[:reserve], *ascii_terms[:reserve]]
    selected_set = set(selected)
    selected.extend(term for term in ordered if term not in selected_set)
    return tuple(selected[:limit])


def _normalized_search_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).casefold()


def _context_source_exact_identifier_specs(
    identifiers: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    specs: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in identifiers[:16]:
        identifier = _normalized_search_text(value).strip()
        if (
            not identifier
            or len(identifier) > MAX_CONTEXT_EXACT_IDENTIFIER_CHARS
            or identifier in seen
        ):
            continue
        terms = tuple(dict.fromkeys(unicode_word_terms(identifier)))
        if not terms:
            continue
        seen.add(identifier)
        anchor = max(
            enumerate(terms),
            key=lambda indexed_term: (
                len(indexed_term[1]),
                indexed_term[0],
            ),
        )[1]
        specs.append(
            {
                "identifier": identifier,
                "anchor": anchor,
                "terms": terms,
            }
        )
    return tuple(specs)


def _contains_exact_structured_identifier(
    value: object,
    identifier: object,
) -> int:
    text = _normalized_search_text(value)[
        :MAX_CONTEXT_SOURCE_SEARCH_TEXT_CHARS
    ]
    needle = _normalized_search_text(identifier).strip()
    if not needle:
        return 0
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return 0
        end = index + len(needle)
        before_is_word = index > 0 and (
            text[index - 1].isalnum() or text[index - 1] == "_"
        )
        after_is_word = end < len(text) and (
            text[end].isalnum() or text[end] == "_"
        )
        before_continues_identifier = (
            index > 1
            and text[index - 1] in {"-", ".", "_"}
            and text[index - 2].isalnum()
        )
        after_continues_identifier = (
            end + 1 < len(text)
            and text[end] in {"-", ".", "_"}
            and text[end + 1].isalnum()
        )
        if (
            not before_is_word
            and not after_is_word
            and not before_continues_identifier
            and not after_continues_identifier
        ):
            return 1
        start = index + 1


def _context_source_search_terms(title: str, body: str) -> tuple[str, ...]:
    bounded_title = title[:MAX_CONTEXT_SOURCE_SEARCH_TEXT_CHARS]
    bounded_body = body[:MAX_CONTEXT_SOURCE_SEARCH_TEXT_CHARS]
    terms = (
        *unicode_word_terms(bounded_title),
        *cjk_lexical_anchors(bounded_title),
        *unicode_word_terms(bounded_body),
        *cjk_lexical_anchors(bounded_body),
    )
    return tuple(dict.fromkeys(terms))[:4096]


def _validate_context_source_provenance_metadata(
    metadata: dict[str, object],
) -> None:
    scalar_source_ids: list[str] = []
    for field in ("source_message_id", "last_seen_message_id"):
        value = metadata.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError("context source scalar provenance is invalid")
        scalar_source_ids.append(value)
    raw_source_ids = metadata.get("source_message_ids")
    if raw_source_ids is not None and (
        not isinstance(raw_source_ids, list)
        or len(raw_source_ids) > MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS
        or not all(isinstance(item, str) and item for item in raw_source_ids)
    ):
        raise ValueError("context source message provenance is invalid or too large")
    if isinstance(raw_source_ids, list):
        scalar_source_ids.extend(raw_source_ids)
    raw_groups = metadata.get("source_provenance_groups")
    if raw_groups is not None and (
        not isinstance(raw_groups, list)
        or len(raw_groups) > MAX_CONTEXT_SOURCE_PROVENANCE_GROUPS
        or not all(
            isinstance(group, list)
            and bool(group)
            and len(group) <= MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS
            and all(isinstance(item, str) and item for item in group)
            for group in raw_groups
        )
    ):
        raise ValueError("context source provenance groups are invalid or too large")
    if isinstance(raw_groups, list) and raw_groups:
        grouped_source_ids = {
            item
            for group in raw_groups
            if isinstance(group, list)
            for item in group
            if isinstance(item, str)
        }
        if not set(scalar_source_ids).issubset(grouped_source_ids):
            raise ValueError(
                "context source provenance groups omit scalar provenance"
            )
    mode = metadata.get("source_provenance_mode")
    if mode is not None and mode not in {"all", "any"}:
        raise ValueError("context source provenance mode is invalid")


def _context_source_eligibility_sql(
    *,
    alias: str,
    allowed_owner_names: set[str] | frozenset[str] | None,
    reference_character_ids: set[str] | frozenset[str] | None,
    visibility_character_ids: set[str] | frozenset[str] | None,
    current_scene_snapshot_id: str | None,
    current_scene_generation: int | None,
    current_turn_number: int | None,
    blocked_source_keys: set[tuple[str, str]] | frozenset[tuple[str, str]]
    | None,
) -> tuple[str, tuple[object, ...]]:
    clauses: list[str] = []
    params: list[object] = []
    visibility_ids = tuple(sorted(visibility_character_ids or ()))
    if visibility_ids:
        clauses.append(
            "("
            "("
            f"json_array_length(json_extract({alias}.metadata_json, "
            "'$.source_provenance_groups')) > 0 "
            "AND ("
            "("
            f"json_extract({alias}.metadata_json, '$.source_provenance_mode') "
            "= 'all' "
            "AND NOT EXISTS ("
            f"SELECT 1 FROM json_each({alias}.metadata_json, "
            "'$.source_provenance_groups') provenance_group "
            "WHERE json_array_length(provenance_group.value) > 0 "
            "AND EXISTS ("
            "SELECT 1 FROM json_each(provenance_group.value) source_message "
            "JOIN message_visibility visibility "
            f"ON visibility.save_id = {alias}.save_id "
            "AND visibility.message_id = CAST(source_message.value AS TEXT) "
            "WHERE visibility.visibility = 'not_visible' "
            f"AND visibility.character_id IN "
            f"({_placeholders(len(visibility_ids))})"
            ")"
            ")"
            ")"
            " OR "
            "("
            f"COALESCE(json_extract({alias}.metadata_json, "
            "'$.source_provenance_mode'), 'any') != 'all' "
            "AND EXISTS ("
            f"SELECT 1 FROM json_each({alias}.metadata_json, "
            "'$.source_provenance_groups') provenance_group "
            "WHERE json_array_length(provenance_group.value) > 0 "
            "AND NOT EXISTS ("
            "SELECT 1 FROM json_each(provenance_group.value) source_message "
            "JOIN message_visibility visibility "
            f"ON visibility.save_id = {alias}.save_id "
            "AND visibility.message_id = CAST(source_message.value AS TEXT) "
            "WHERE visibility.visibility = 'not_visible' "
            f"AND visibility.character_id IN "
            f"({_placeholders(len(visibility_ids))})"
            ")"
            ")"
            ")"
            ")"
            ")"
            " OR "
            "("
            f"COALESCE(json_array_length(json_extract({alias}.metadata_json, "
            "'$.source_provenance_groups')), 0) = 0 "
            "AND NOT EXISTS ("
            "SELECT 1 FROM message_visibility visibility "
            f"WHERE visibility.save_id = {alias}.save_id "
            "AND visibility.visibility = 'not_visible' "
            f"AND visibility.character_id IN "
            f"({_placeholders(len(visibility_ids))}) "
            "AND visibility.message_id IN ("
            f"SELECT CAST(value AS TEXT) FROM json_each({alias}.metadata_json, "
            "'$.source_message_ids') "
            "UNION "
            f"SELECT CAST(json_extract({alias}.metadata_json, "
            "'$.source_message_id') AS TEXT) "
            f"WHERE json_extract({alias}.metadata_json, "
            "'$.source_message_id') IS NOT NULL "
            "UNION "
            f"SELECT CAST(json_extract({alias}.metadata_json, "
            "'$.last_seen_message_id') AS TEXT) "
            f"WHERE json_extract({alias}.metadata_json, "
            "'$.last_seen_message_id') IS NOT NULL"
            ")"
            ")"
            ")"
            ")"
        )
        params.extend(visibility_ids)
        params.extend(visibility_ids)
        params.extend(visibility_ids)
    if visibility_character_ids is not None:
        scope_ids_json = _dump_json(sorted(visibility_character_ids))
        target_matches_edge = (
            f"(edge.target_type = {alias}.source_type OR ("
            f"{alias}.source_type = 'world_state' AND edge.target_type = 'state')) "
            f"AND edge.target_id = {alias}.source_id"
        )
        target_matches_link = (
            f"(link.target_type = {alias}.source_type OR ("
            f"{alias}.source_type = 'world_state' AND link.target_type = 'state')) "
            f"AND link.target_id = {alias}.source_id"
        )
        clauses.append(
            "("
            f"{alias}.source_type NOT IN "
            "('memory', 'world_state', 'summary', 'scenario_section') "
            "OR ("
            "NOT EXISTS ("
            "SELECT 1 FROM character_knowledge_edges edge "
            f"WHERE edge.save_id = {alias}.save_id "
            "AND edge.archived_at IS NULL "
            f"AND {target_matches_edge}"
            ") "
            "AND NOT EXISTS ("
            "SELECT 1 FROM entity_links link "
            f"WHERE link.save_id = {alias}.save_id "
            "AND link.entity_type = 'character' "
            "AND link.relation = 'knows' "
            f"AND {target_matches_link}"
            ")"
            ") "
            "OR EXISTS ("
            "SELECT 1 FROM character_knowledge_edges edge "
            f"WHERE edge.save_id = {alias}.save_id "
            "AND edge.archived_at IS NULL "
            f"AND {target_matches_edge} "
            "AND edge.character_id IN ("
            "SELECT CAST(value AS TEXT) FROM json_each(?)"
            ") "
            "AND (edge.knowledge_state = 'knows' OR ("
            "edge.knowledge_state = 'may_know' AND edge.confidence >= 0.7"
            ")) "
            "AND NOT EXISTS ("
            "SELECT 1 FROM message_visibility hidden "
            f"WHERE hidden.save_id = {alias}.save_id "
            "AND hidden.visibility = 'not_visible' "
            "AND hidden.character_id IN ("
            "SELECT CAST(value AS TEXT) FROM json_each(?)"
            ") "
            "AND (hidden.message_id = edge.source_message_id OR "
            "hidden.message_id IN ("
            "SELECT CAST(value AS TEXT) "
            "FROM json_each(edge.source_message_ids_json)"
            "))"
            ")"
            ") "
            "OR EXISTS ("
            "SELECT 1 FROM entity_links link "
            f"WHERE link.save_id = {alias}.save_id "
            "AND link.entity_type = 'character' "
            "AND link.relation = 'knows' "
            f"AND {target_matches_link} "
            "AND link.entity_id IN ("
            "SELECT CAST(value AS TEXT) FROM json_each(?)"
            ") "
            "AND NOT EXISTS ("
            "SELECT 1 FROM character_knowledge_edges edge "
            "WHERE edge.save_id = link.save_id "
            "AND edge.archived_at IS NULL "
            "AND edge.character_id = link.entity_id "
            "AND edge.target_type = link.target_type "
            "AND edge.target_id = link.target_id"
            ") "
            "AND NOT EXISTS ("
            "SELECT 1 FROM message_visibility hidden "
            f"WHERE hidden.save_id = {alias}.save_id "
            "AND hidden.visibility = 'not_visible' "
            "AND hidden.character_id IN ("
            "SELECT CAST(value AS TEXT) FROM json_each(?)"
            ") "
            "AND hidden.message_id = link.source_message_id"
            ")"
            ")"
            ")"
        )
        params.extend(
            (
                scope_ids_json,
                scope_ids_json,
                scope_ids_json,
                scope_ids_json,
            )
        )
    if allowed_owner_names is not None or reference_character_ids is not None:
        owners = tuple(sorted(allowed_owner_names or ()))
        character_ids = tuple(sorted(reference_character_ids or ()))
        audience_match = "0"
        if reference_character_ids is None:
            audience_match = "1"
        elif character_ids:
            audience_match = (
                "EXISTS ("
                f"SELECT 1 FROM json_each({alias}.metadata_json, "
                "'$.audience_character_ids') audience "
                f"WHERE CAST(audience.value AS TEXT) IN "
                f"({_placeholders(len(character_ids))})"
                ")"
            )
            params.extend(character_ids)
        owner_match = "0"
        if allowed_owner_names is None:
            owner_match = "1"
        elif owners:
            owner_variants = tuple(
                sorted(
                    {
                        variant
                        for owner in owners
                        for variant in (owner, owner.casefold(), owner.upper())
                    }
                )
            )
            owner_match = (
                "EXISTS ("
                f"SELECT 1 FROM json_each({alias}.metadata_json, '$.known_by') owner "
                f"WHERE CAST(owner.value AS TEXT) COLLATE NOCASE IN "
                f"({_placeholders(len(owner_variants))})"
                ")"
            )
            params.extend(owner_variants)
        audience_count = (
            f"COALESCE(json_array_length(json_extract({alias}.metadata_json, "
            "'$.audience_character_ids')), 0)"
        )
        known_count = (
            f"COALESCE(json_array_length(json_extract({alias}.metadata_json, "
            "'$.known_by')), 0)"
        )
        requires_audience = (
            f"COALESCE(json_extract({alias}.metadata_json, "
            "'$.requires_audience'), 0)"
        )
        clauses.append(
            f"(({audience_count} > 0 AND {audience_match}) OR "
            f"({requires_audience} != 1 AND {audience_count} = 0 AND "
            f"({known_count} = 0 OR {owner_match})))"
        )
    if (
        current_scene_snapshot_id is not None
        and current_scene_generation is not None
        and current_turn_number is not None
    ):
        clauses.append(
            "("
            f"COALESCE(json_extract({alias}.metadata_json, "
            "'$.curation_action'), '') "
            "!= 'scene_scratch' OR "
            f"({alias}.scene_snapshot_id = ? AND {alias}.scene_generation = ? "
            f"AND ({alias}.expires_after_turn_number IS NULL "
            f"OR {alias}.expires_after_turn_number > ?))"
            ")"
        )
        params.extend(
            (
                current_scene_snapshot_id,
                current_scene_generation,
                current_turn_number,
            )
        )
    blocked_keys = tuple(sorted(blocked_source_keys or ()))
    if blocked_keys:
        clauses.append(
            f"({alias}.source_type, {alias}.source_id) NOT IN ("
            "SELECT "
            "CAST(json_extract(value, '$[0]') AS TEXT), "
            "CAST(json_extract(value, '$[1]') AS TEXT) "
            "FROM json_each(?)"
            ")"
        )
        params.append(_dump_json(blocked_keys))
    if not clauses:
        return "", ()
    return "AND " + " AND ".join(f"({clause})" for clause in clauses), tuple(params)


def _validate_job_status(status: str) -> None:
    if status not in JOB_STATUSES:
        raise ValueError(f"Unknown job status: {status}")


def _validate_character_text_delivery_status(status: str) -> None:
    if status not in CHARACTER_TEXT_DELIVERY_STATUSES:
        raise ValueError(f"Unknown character text delivery status: {status}")


def _validate_character_text_attachment_kind(kind: str) -> None:
    if kind not in CHARACTER_TEXT_ATTACHMENT_KINDS:
        raise ValueError(f"Unknown character text attachment kind: {kind}")


def _validate_character_text_attachment_status(status: str) -> None:
    if status not in CHARACTER_TEXT_ATTACHMENT_STATUSES:
        raise ValueError(f"Unknown character text attachment status: {status}")


def _validate_job_initial_status(status: str) -> None:
    _validate_job_status(status)
    if status not in _JOB_INITIAL_STATUSES:
        raise ValueError(f"Unsupported initial job status: {status}")


def _validate_job_update_status(status: str) -> None:
    if status not in _JOB_UPDATE_STATUSES:
        raise ValueError(f"Unsupported job terminal status: {status}")


def _validate_job_step_status(status: str) -> None:
    if status not in JOB_STEP_STATUSES:
        raise ValueError(f"Unknown job step status: {status}")


def _validate_loss_condition_status(status: str) -> None:
    if _compat_condition_status(status) not in LOSS_CONDITION_STATUSES:
        raise ValueError(f"Unknown loss condition status: {status}")


def _media_mime_type(*, type: str, mime_type: str | None) -> str:
    normalized = canonical_media_mime_type(mime_type)
    if type == "video":
        if (
            normalized in SUPPORTED_VIDEO_MIME_TYPES
            or normalized == INERT_MEDIA_MIME_TYPE
        ):
            return normalized
        raise ValueError(
            "Unsupported video mime type; use video/mp4, video/webm, "
            "or application/octet-stream"
        )
    if normalized is None:
        return "image/png"
    if (
        normalized in SUPPORTED_IMAGE_MIME_TYPES
        or normalized == INERT_MEDIA_MIME_TYPE
    ):
        return normalized
    raise ValueError(
        "Unsupported image mime type; use image/png, image/jpeg, image/webp, "
        "or application/octet-stream"
    )


def _validate_media_asset_path(value: str) -> None:
    path = Path(value)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in path.parts
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError("Media asset paths must be safe relative paths")


def _compat_condition_status(status: str) -> str:
    return status


def _loss_condition_key(label: str) -> str:
    return "-".join(label.casefold().strip().split())


def _dump_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _scene_world_time_value(
    provided: object,
    *,
    existing: sqlite3.Row | None,
    column: str,
    fallback: object,
) -> object:
    if provided is not _UNSET:
        return provided
    if existing is not None:
        existing_value = existing[column]
        if existing_value not in (None, ""):
            return existing_value
    return fallback


def _scene_world_time_provided_value_changed(
    provided: object,
    *,
    existing: sqlite3.Row | None,
    column: str,
    effective: object,
) -> bool:
    if provided is _UNSET:
        return False
    if existing is None:
        return effective not in (None, "")
    return bool(existing[column] != effective)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [text for item in value if isinstance(item, str) and (text := item.strip())]


def _safe_job_step_metadata(
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in (metadata or {}).items():
        normalized_key = str(key)
        if isinstance(value, bool):
            if normalized_key == "changed":
                safe[normalized_key] = value
            continue
        if isinstance(value, int | float):
            safe[normalized_key] = value
            continue
        if normalized_key in {"before", "proposed", "after"}:
            time_values = _safe_world_time_step_values(value)
            if time_values:
                safe[normalized_key] = time_values
            continue
        if normalized_key in _SAFE_TEXT_METADATA_KEYS:
            text = _safe_metadata_text(value)
            if text is not None:
                safe[normalized_key] = text
            continue
        if normalized_key in _SAFE_TEXT_LIST_METADATA_KEYS:
            text_values = _safe_metadata_text_list(value)
            if text_values:
                safe[normalized_key] = text_values
            continue
        if normalized_key in _SAFE_NUMBER_LIST_METADATA_KEYS:
            number_values = _safe_metadata_number_list(value)
            if number_values:
                safe[normalized_key] = number_values
    return safe


def _safe_world_time_step_values(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    safe: dict[str, object] = {}
    for key in ("in_world_time", "time_of_day", "day_of_week"):
        text = _safe_metadata_text(value.get(key))
        if text is not None:
            safe[key] = text
    world_day_index = value.get("world_day_index")
    if (
        isinstance(world_day_index, int)
        and not isinstance(world_day_index, bool)
    ) or world_day_index is None:
        safe["world_day_index"] = world_day_index
    return safe


def _safe_metadata_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > _MAX_SAFE_METADATA_TEXT_LENGTH:
        return None
    return text


def _safe_metadata_text_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    values: list[str] = []
    for item in value[:_MAX_SAFE_METADATA_LIST_ITEMS]:
        text = _safe_metadata_text(item)
        if text is not None:
            values.append(text)
    return values


def _safe_metadata_number_list(value: object) -> list[int | float]:
    if not isinstance(value, list | tuple):
        return []
    values: list[int | float] = []
    for item in value[:_MAX_SAFE_METADATA_LIST_ITEMS]:
        if isinstance(item, bool):
            continue
        if isinstance(item, int | float):
            values.append(item)
    return values


def _json_digest(value: object) -> str:
    return sha256(_dump_json(value).encode("utf-8")).hexdigest()


def _text_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _dedupe_strings(values: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _memory_source_message_ids(
    *,
    source_message_id: str | None,
    source_message_ids: list[str] | tuple[str, ...] | None,
) -> list[str]:
    values = [str(value) for value in source_message_ids or [] if value]
    if source_message_id:
        values.insert(0, source_message_id)
    return _dedupe_strings(tuple(values))[:MAX_MEMORY_SOURCE_MESSAGE_IDS]


def _save_scenario_evolution_turn_interval_setting_key(save_id: str) -> str:
    return f"{_SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING}:save:{save_id}"


def _save_image_style_preset_setting_key(save_id: str) -> str:
    return f"{_IMAGE_STYLE_PRESET_SETTING}:save:{save_id}"


def _scenario_template_evolution_turn_interval_setting_key(scenario_id: str) -> str:
    return f"{_SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING}:scenario:{scenario_id}"


def _legacy_app_setting_parts(key: str) -> tuple[str, str, str]:
    for prefix, scope, scoped_key in (
        (
            f"{_IMAGE_STYLE_PRESET_SETTING}:save:",
            "save",
            _IMAGE_STYLE_PRESET_SETTING,
        ),
        (
            f"{_SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING}:save:",
            "save",
            _SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
        ),
        (
            f"{_SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING}:scenario:",
            "scenario",
            _SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
        ),
    ):
        if key.startswith(prefix):
            return scope, key.removeprefix(prefix), scoped_key
    return "global", "", key


def _setting_scope_id(scope: str, scope_id: str | None) -> str:
    if scope not in _SETTING_SCOPES:
        raise ValueError(f"Unknown setting scope: {scope}")
    if scope == "global":
        return ""
    if not scope_id:
        raise ValueError(f"{scope} setting scope requires scope_id")
    return scope_id


def _scoped_setting_from_row(row: sqlite3.Row) -> ScopedSettingRecord:
    scope_id = row["scope_id"]
    return ScopedSettingRecord(
        scope=row["scope"],
        scope_id=scope_id if scope_id else None,
        key=row["key"],
        value=cast(object, json.loads(row["value_json"])),
        value_json=row["value_json"],
        updated_at=row["updated_at"],
    )


def _save_scenario_source_message_ids(
    *,
    source_message_id: str | None,
    source_message_ids: tuple[str, ...],
) -> tuple[str, ...]:
    ids: list[str] = []
    if source_message_id:
        ids.append(source_message_id)
    ids.extend(source_id for source_id in source_message_ids if source_id)
    return tuple(dict.fromkeys(ids))


def _save_scenario_update_matches_messages(
    row: sqlite3.Row,
    message_ids: set[str] | frozenset[str],
) -> bool:
    if row["source_message_id"] in message_ids:
        return True
    try:
        loaded = json.loads(str(row["source_message_ids_json"] or "[]"))
    except json.JSONDecodeError:
        return False
    if not isinstance(loaded, list):
        return False
    return any(isinstance(item, str) and item in message_ids for item in loaded)


def _context_source_references_any_message(
    record: ContextSourceRecord,
    message_ids: set[str] | frozenset[str],
) -> bool:
    if record.source_type == "message":
        source_ids = {
            item.strip() for item in record.source_id.split(",") if item.strip()
        }
        if source_ids & set(message_ids):
            return True
    metadata_source_ids = record.metadata.get("source_message_ids")
    if isinstance(metadata_source_ids, list):
        return any(
            isinstance(item, str) and item in message_ids
            for item in metadata_source_ids
        )
    metadata_source_id = record.metadata.get("source_message_id")
    return isinstance(metadata_source_id, str) and metadata_source_id in message_ids


def _context_source_from_row(row: sqlite3.Row) -> ContextSourceRecord:
    return ContextSourceRecord(
        id=row["id"],
        save_id=row["save_id"],
        source_type=row["source_type"],
        source_id=row["source_id"],
        title=row["title"],
        body=row["body"],
        metadata=_load_object(row["metadata_json"]),
        token_estimate=row["token_estimate"],
        scene_snapshot_id=row["scene_snapshot_id"],
        scene_generation=_optional_int(row["scene_generation"]),
        created_turn_number=_optional_int(row["created_turn_number"]),
        expires_after_turn_number=_optional_int(row["expires_after_turn_number"]),
    )


def _scene_snapshot_from_row(row: sqlite3.Row) -> SceneSnapshotRecord:
    first_seen_message_id = row["first_seen_message_id"] or row["source_message_id"]
    last_updated_message_id = row["last_updated_message_id"] or row["source_message_id"]
    return SceneSnapshotRecord(
        id=row["id"],
        save_id=row["save_id"],
        current_location_id=row["current_location_id"],
        situation=row["situation"],
        objective=row["objective"],
        in_world_time=row["in_world_time"],
        time_of_day=row["time_of_day"],
        day_of_week=row["day_of_week"],
        world_day_index=_optional_int(row["world_day_index"]),
        world_time_day_index=_optional_int(row["world_time_day_index"]),
        world_time_day_label=row["world_time_day_label"],
        world_time_phase=row["world_time_phase"],
        world_time_clock_minutes=_optional_int(row["world_time_clock_minutes"]),
        world_time_period_label=row["world_time_period_label"],
        world_time_source_message_id=row["world_time_source_message_id"],
        world_time_confidence=_optional_float(row["world_time_confidence"]),
        weather=row["weather"],
        mood=row["mood"],
        nearby_objects=_load_list(row["nearby_objects_json"]),
        hazards=_load_list(row["hazards_json"]),
        present_character_ids=_load_list(row["present_character_ids_json"]),
        source_message_id=row["source_message_id"],
        locked_fields=_load_list(row["locked_fields_json"]),
        first_seen_message_id=first_seen_message_id,
        last_updated_message_id=last_updated_message_id,
        scene_generation=int(row["scene_generation"]),
    )


def _location_params(record: LocationRecord) -> tuple[object, ...]:
    return (
        record.id,
        record.save_id,
        record.name,
        _dump_json(record.aliases),
        record.description,
        record.visual_description,
        record.parent_location_id,
        _dump_json(record.connections),
        record.status,
        _dump_json(record.hazards),
        record.source_message_id,
        _dump_json(record.locked_fields),
        record.first_seen_message_id or record.source_message_id,
        record.last_updated_message_id or record.source_message_id,
    )


def _location_from_row(row: sqlite3.Row) -> LocationRecord:
    first_seen_message_id = row["first_seen_message_id"] or row["source_message_id"]
    last_updated_message_id = row["last_updated_message_id"] or row["source_message_id"]
    return LocationRecord(
        id=row["id"],
        save_id=row["save_id"],
        name=row["name"],
        aliases=_load_list(row["aliases_json"]),
        description=row["description"],
        visual_description=row["visual_description"],
        parent_location_id=row["parent_location_id"],
        connections=_load_list(row["connections_json"]),
        status=row["status"],
        hazards=_load_list(row["hazards_json"]),
        source_message_id=row["source_message_id"],
        locked_fields=_load_list(row["locked_fields_json"]),
        first_seen_message_id=first_seen_message_id,
        last_updated_message_id=last_updated_message_id,
    )


def _character_params(record: CharacterRecord) -> tuple[object, ...]:
    return (
        record.id,
        record.save_id,
        record.name,
        _dump_json(record.aliases),
        record.role,
        record.age,
        record.known_state,
        record.history,
        int(record.met),
        record.appearance,
        record.visual_notes,
        record.current_clothing,
        record.personality,
        record.voice,
        _dump_json(record.relationships),
        record.goals,
        record.motivations,
        record.current_intent,
        record.boundaries,
        record.attitude_toward_player,
        record.cooperation_conditions,
        record.status,
        record.location_id,
        record.private_notes,
        record.source_message_id,
        _dump_json(record.locked_fields),
        int(record.protected_from_maintenance),
        int(record.is_player_character),
        record.texting_style,
        record.contact_name,
        record.first_seen_message_id or record.source_message_id,
        record.last_updated_message_id or record.source_message_id,
        record.content_rating,
    )


def _character_from_row(row: sqlite3.Row) -> CharacterRecord:
    first_seen_message_id = row["first_seen_message_id"] or row["source_message_id"]
    last_updated_message_id = row["last_updated_message_id"] or row["source_message_id"]
    return CharacterRecord(
        id=row["id"],
        save_id=row["save_id"],
        name=row["name"],
        aliases=_load_list(row["aliases_json"]),
        role=row["role"],
        age=row["age"],
        known_state=row["history"] or row["known_state"],
        history=row["history"] or row["known_state"],
        met=bool(row["met"]),
        appearance=row["appearance"],
        visual_notes=row["visual_notes"],
        current_clothing=row["current_clothing"],
        personality=row["personality"],
        voice=row["voice"],
        relationships=_load_object(row["relationships_json"]),
        goals=row["goals"],
        motivations=row["motivations"],
        current_intent=row["current_intent"],
        boundaries=row["boundaries"],
        attitude_toward_player=row["attitude_toward_player"],
        cooperation_conditions=row["cooperation_conditions"],
        status=row["status"],
        location_id=row["location_id"],
        private_notes=row["private_notes"],
        source_message_id=row["source_message_id"],
        locked_fields=_load_list(row["locked_fields_json"]),
        protected_from_maintenance=bool(row["protected_from_maintenance"]),
        is_player_character=bool(row["is_player_character"]),
        texting_style=row["texting_style"],
        contact_name=row["contact_name"],
        first_seen_message_id=first_seen_message_id,
        last_updated_message_id=last_updated_message_id,
        content_rating=row["content_rating"],
    )


def _character_text_thread_from_row(row: sqlite3.Row) -> CharacterTextThreadRecord:
    return CharacterTextThreadRecord(
        id=row["id"],
        save_id=row["save_id"],
        character_id=row["character_id"],
        title=row["title"],
        status=row["status"],
        kind=row["kind"],
        memory_body=row["memory_body"],
        memory_message_count=_optional_int(row["memory_message_count"]) or 0,
        memory_updated_at=row["memory_updated_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _character_text_thread_participant_from_row(
    row: sqlite3.Row,
) -> CharacterTextThreadParticipantRecord:
    return CharacterTextThreadParticipantRecord(
        id=row["id"],
        save_id=row["save_id"],
        thread_id=row["thread_id"],
        character_id=row["character_id"],
        ordinal=_optional_int(row["ordinal"]) or 0,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _character_text_message_from_row(row: sqlite3.Row) -> CharacterTextMessageRecord:
    return CharacterTextMessageRecord(
        id=row["id"],
        save_id=row["save_id"],
        thread_id=row["thread_id"],
        character_id=row["character_id"],
        sender=row["sender"],
        body=row["body"],
        sender_character_id=row["sender_character_id"],
        provider=row["provider"],
        model=row["model"],
        content_rating=row["content_rating"],
        token_estimate=_optional_int(row["token_estimate"]),
        delivery_status=row["delivery_status"],
        delivery_error=row["delivery_error"],
        delivery_job_id=row["delivery_job_id"],
        delivery_attempt=_optional_int(row["delivery_attempt"]) or 0,
        in_world_sent_at=row["in_world_sent_at"],
        delivered_at=row["delivered_at"],
        read_at=row["read_at"],
        reply_to_message_id=row["reply_to_message_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        deleted_at=row["deleted_at"],
    )


def _character_text_activity_event_from_row(
    row: sqlite3.Row,
) -> CharacterTextActivityEventRecord:
    return CharacterTextActivityEventRecord(
        id=row["id"],
        save_id=row["save_id"],
        ordinal=int(row["ordinal"]),
        thread_id=row["thread_id"],
        activity_type=row["activity_type"],
        text_message_id=row["text_message_id"],
        read_count=_optional_int(row["read_count"]) or 0,
        delivery_status=row["delivery_status"],
        created_at=row["created_at"],
    )


def _character_text_message_attachment_from_row(
    row: sqlite3.Row,
) -> CharacterTextMessageAttachmentRecord:
    return CharacterTextMessageAttachmentRecord(
        id=row["id"],
        save_id=row["save_id"],
        thread_id=row["thread_id"],
        text_message_id=row["text_message_id"],
        character_id=row["character_id"],
        ordinal=_optional_int(row["ordinal"]) or 0,
        kind=row["kind"],
        status=row["status"],
        media_asset_id=row["media_asset_id"],
        prompt=row["prompt"],
        error=row["error"],
        metadata_json=row["metadata_json"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _character_text_message_revision_from_row(
    row: sqlite3.Row,
) -> CharacterTextMessageRevisionRecord:
    return CharacterTextMessageRevisionRecord(
        id=row["id"],
        save_id=row["save_id"],
        text_message_id=row["text_message_id"],
        revision_number=row["revision_number"],
        previous_body=row["previous_body"],
        new_body=row["new_body"],
        diff_unified=row["diff_unified"],
        reconciliation_status=row["reconciliation_status"],
        reconciliation_error=row["reconciliation_error"],
        created_at=row["created_at"],
        reconciled_at=row["reconciled_at"],
    )


def _character_text_provenance_from_row(
    row: sqlite3.Row,
) -> CharacterTextProvenanceRecord:
    return CharacterTextProvenanceRecord(
        id=row["id"],
        save_id=row["save_id"],
        thread_id=row["thread_id"],
        text_message_id=row["text_message_id"],
        target_type=row["target_type"],
        target_id=row["target_id"],
        operation=row["operation"],
        field_path=row["field_path"],
        created_at=row["created_at"],
    )


def _character_text_proactive_trigger_from_row(
    row: sqlite3.Row,
) -> CharacterTextProactiveTriggerRecord:
    return CharacterTextProactiveTriggerRecord(
        id=row["id"],
        save_id=row["save_id"],
        character_id=row["character_id"],
        trigger_key=row["trigger_key"],
        trigger_type=row["trigger_type"],
        thread_id=row["thread_id"],
        text_message_id=row["text_message_id"],
        source_type=row["source_type"],
        source_id=row["source_id"],
        source_message_id=row["source_message_id"],
        reason=row["reason"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _character_contact_state_from_row(row: sqlite3.Row) -> CharacterContactStateRecord:
    return CharacterContactStateRecord(
        id=row["id"],
        save_id=row["save_id"],
        player_character_id=row["player_character_id"],
        character_id=row["character_id"],
        player_has_character_number=bool(row["player_has_character_number"]),
        character_has_player_number=bool(row["character_has_player_number"]),
        source_message_id=row["source_message_id"],
        source_text_message_id=row["source_text_message_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _dating_route_state_params(record: DatingRouteStateRecord) -> tuple[object, ...]:
    return (
        record.id,
        record.save_id,
        record.player_character_id,
        record.npc_character_id,
        record.stage,
        record.first_met_message_id,
        record.first_met_world_day_index,
        record.last_interaction_message_id,
        record.last_interaction_world_day_index,
        record.completed_interactions,
        record.dates_completed,
        record.interest_level,
        record.trust_level,
        record.comfort_with_intimacy,
        record.pacing_preference,
        _dump_json(record.known_boundaries),
        _dump_json(record.unresolved_questions),
        record.next_reasonable_step,
        record.source_message_id,
    )


def _dating_route_state_from_row(row: sqlite3.Row) -> DatingRouteStateRecord:
    return DatingRouteStateRecord(
        id=row["id"],
        save_id=row["save_id"],
        player_character_id=row["player_character_id"],
        npc_character_id=row["npc_character_id"],
        stage=row["stage"],
        first_met_message_id=row["first_met_message_id"],
        first_met_world_day_index=_optional_int(row["first_met_world_day_index"]),
        last_interaction_message_id=row["last_interaction_message_id"],
        last_interaction_world_day_index=_optional_int(
            row["last_interaction_world_day_index"]
        ),
        completed_interactions=max(0, int(row["completed_interactions"])),
        dates_completed=max(0, int(row["dates_completed"])),
        interest_level=row["interest_level"],
        trust_level=row["trust_level"],
        comfort_with_intimacy=row["comfort_with_intimacy"],
        pacing_preference=row["pacing_preference"],
        known_boundaries=_load_list(row["known_boundaries_json"]),
        unresolved_questions=_load_list(row["unresolved_questions_json"]),
        next_reasonable_step=row["next_reasonable_step"],
        source_message_id=row["source_message_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _validate_dating_route_stage(stage: str) -> None:
    if stage not in DATING_ROUTE_STAGES:
        raise ValueError(f"Unknown dating route stage: {stage}")


def _present_character_ids_with_player_character(
    present_character_ids: list[str],
    player_character_id: str | None,
) -> list[str]:
    normalized = list(dict.fromkeys(present_character_ids))
    if player_character_id and player_character_id not in normalized:
        normalized.append(player_character_id)
    return normalized


def _scene_snapshot_with_player_character(
    snapshot: SceneSnapshotRecord,
    player_character_id: str | None,
) -> SceneSnapshotRecord:
    present_ids = _present_character_ids_with_player_character(
        snapshot.present_character_ids,
        player_character_id,
    )
    if present_ids == snapshot.present_character_ids:
        return snapshot
    return replace(snapshot, present_character_ids=present_ids)


def _active_thread_params(record: ActiveThreadRecord) -> tuple[object, ...]:
    return (
        record.id,
        record.save_id,
        record.title,
        record.description,
        record.status,
        record.priority,
        record.visibility,
        _dump_json(record.related_entities),
        record.source_message_id,
        _dump_json(record.locked_fields),
        record.first_seen_message_id or record.source_message_id,
        record.last_updated_message_id or record.source_message_id,
    )


def _active_thread_from_row(row: sqlite3.Row) -> ActiveThreadRecord:
    first_seen_message_id = row["first_seen_message_id"] or row["source_message_id"]
    last_updated_message_id = row["last_updated_message_id"] or row["source_message_id"]
    return ActiveThreadRecord(
        id=row["id"],
        save_id=row["save_id"],
        title=row["title"],
        description=row["description"],
        status=row["status"],
        priority=row["priority"],
        visibility=row["visibility"],
        related_entities=_load_list(row["related_entities_json"]),
        source_message_id=row["source_message_id"],
        locked_fields=_load_list(row["locked_fields_json"]),
        first_seen_message_id=first_seen_message_id,
        last_updated_message_id=last_updated_message_id,
    )


def _context_update_suggestion_from_row(
    row: sqlite3.Row,
) -> ContextUpdateSuggestionRecord:
    return ContextUpdateSuggestionRecord(
        id=row["id"],
        save_id=row["save_id"],
        update_type=row["update_type"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        field_path=row["field_path"],
        proposed_value=_load_json(row["proposed_value_json"]),
        status=row["status"],
        reason=row["reason"],
        confidence=row["confidence"],
        source_message_ids=_load_list(row["source_message_ids_json"]),
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
        review_attempt_count=row["review_attempt_count"],
        next_review_at=row["next_review_at"],
        last_review_error=row["last_review_error"],
    )


def _context_update_audit_from_row(row: sqlite3.Row) -> ContextUpdateAuditRecord:
    return ContextUpdateAuditRecord(
        id=row["id"],
        save_id=row["save_id"],
        suggestion_id=row["suggestion_id"],
        operation=row["operation"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        field_path=row["field_path"],
        before=_load_json(row["before_json"]) if row["before_json"] else None,
        after=_load_json(row["after_json"]) if row["after_json"] else None,
        reason=row["reason"],
        confidence=row["confidence"],
        source_message_ids=_load_list(row["source_message_ids_json"]),
        created_at=row["created_at"],
    )


def _context_observation_from_row(row: sqlite3.Row) -> ContextObservationRecord:
    return ContextObservationRecord(
        id=row["id"],
        save_id=row["save_id"],
        observation_type=row["observation_type"],
        claim=row["claim"],
        evidence_quote=row["evidence_quote"],
        source_message_ids=_load_list(row["source_message_ids_json"]),
        scope=row["scope"],
        status=row["status"],
        confidence=row["confidence"],
        tags=_load_list(row["tags_json"]),
        metadata=_load_object(row["metadata_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _context_observation_curation_state_from_row(
    row: sqlite3.Row,
) -> ContextObservationCurationStateRecord:
    return ContextObservationCurationStateRecord(
        observation_id=row["observation_id"],
        save_id=row["save_id"],
        attempt_count=row["attempt_count"],
        next_eligible_at=row["next_eligible_at"],
        lease_token=row["lease_token"],
        lease_until=row["lease_until"],
        last_error=row["last_error"],
        terminal_outcome=row["terminal_outcome"],
        completed_at=row["completed_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _loss_condition_change_from_row(row: sqlite3.Row) -> LossConditionChangeRecord:
    return LossConditionChangeRecord(
        id=row["id"],
        save_id=row["save_id"],
        condition_id=row["condition_id"],
        source_message_id=row["source_message_id"],
        operation=row["operation"],
        before=_load_object(row["before_json"]) if row["before_json"] else None,
        after=_load_object(row["after_json"]) if row["after_json"] else None,
        reason=row["reason"],
        provider=row["provider"],
        model=row["model"],
        created_at=row["created_at"],
        archived_at=row["archived_at"],
    )


def _loss_outcome_from_row(row: sqlite3.Row) -> LossOutcomeRecord:
    return LossOutcomeRecord(
        id=row["id"],
        save_id=row["save_id"],
        condition_id=row["condition_id"],
        condition_name=row["condition_name"],
        triggering_message_id=row["triggering_message_id"],
        explanation=row["explanation"],
        evidence=_load_object(row["evidence_json"]),
        confidence=row["confidence"],
        provider=row["provider"],
        model=row["model"],
        outcome_type=row["outcome_type"],
        epilogue_provider=row["epilogue_provider"],
        epilogue_model=row["epilogue_model"],
        epilogue_message_id=row["epilogue_message_id"],
        epilogue_error=row["epilogue_error"],
        created_at=row["created_at"],
        archived_at=row["archived_at"],
    )


def _message_revision_from_row(row: sqlite3.Row) -> MessageRevisionRecord:
    return MessageRevisionRecord(
        id=row["id"],
        save_id=row["save_id"],
        message_id=row["message_id"],
        revision_number=row["revision_number"],
        previous_body=row["previous_body"],
        new_body=row["new_body"],
        diff_unified=row["diff_unified"],
        reconciliation_status=row["reconciliation_status"],
        reconciliation_error=row["reconciliation_error"],
        created_at=row["created_at"],
        reconciled_at=row["reconciled_at"],
    )


def _entity_link_endpoint_is_inactive(
    entity_type: str,
    entity_id: str,
    active_ids: dict[str, set[str]],
) -> bool:
    normalized_type = _normalized_entity_link_endpoint_type(entity_type)
    if normalized_type not in active_ids:
        return False
    return entity_id not in active_ids[normalized_type]


def _normalized_entity_link_endpoint_type(entity_type: str) -> str:
    normalized = entity_type.strip().casefold()
    if normalized == "state":
        return "world_state"
    return normalized


def _replace_media_source_metadata_refs(
    metadata: dict[str, object],
    *,
    old_media_asset_id: str,
    new_media_asset_id: str,
) -> bool:
    changed = False
    for key in ("source_character_reference_asset_id", "source_media_asset_id"):
        if metadata.get(key) == old_media_asset_id:
            metadata[key] = new_media_asset_id
            changed = True
    for key in ("source_character_reference_asset_ids", "source_media_asset_ids"):
        value = metadata.get(key)
        if not isinstance(value, list):
            continue
        replaced = [
            new_media_asset_id if item == old_media_asset_id else item
            for item in value
        ]
        if replaced != value:
            metadata[key] = replaced
            changed = True
    return changed


def _load_json(value: str) -> object:
    return json.loads(value)


def _load_object(value: str) -> dict[str, object]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("Expected JSON object")
    return cast(dict[str, object], loaded)


def _load_string_dict(value: str) -> dict[str, str]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in loaded.items()
    ):
        raise ValueError("Expected JSON string object")
    return cast(dict[str, str], loaded)


def _load_list(value: str) -> list[str]:
    loaded = json.loads(value)
    if not isinstance(loaded, list) or not all(
        isinstance(item, str) for item in loaded
    ):
        raise ValueError("Expected JSON string list")
    return cast(list[str], loaded)


def _unique_strings(values: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(item for value in values if (item := str(value).strip())))


def _character_knowledge_edge_from_row(
    row: sqlite3.Row,
) -> CharacterKnowledgeEdgeRecord:
    source_message_ids = _unique_strings(
        [
            *_load_list(row["source_message_ids_json"]),
            *([row["source_message_id"]] if row["source_message_id"] else []),
        ]
    )
    provenance_overflow = (
        len(source_message_ids) > MAX_KNOWLEDGE_EDGE_SOURCE_MESSAGE_IDS
    )
    return CharacterKnowledgeEdgeRecord(
        id=row["id"],
        save_id=row["save_id"],
        character_id=row["character_id"],
        target_type=row["target_type"],
        target_id=row["target_id"],
        knowledge_state=(
            "does_not_know" if provenance_overflow else row["knowledge_state"]
        ),
        acquisition_method=(
            "unknown" if provenance_overflow else row["acquisition_method"]
        ),
        confidence=row["confidence"],
        source_message_id=(
            None if provenance_overflow else row["source_message_id"]
        ),
        source_message_ids=[] if provenance_overflow else source_message_ids,
        evidence_quote=(
            "Provenance exceeded the safe bound."
            if provenance_overflow
            else row["evidence_quote"]
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _message_visibility_from_row(row: sqlite3.Row) -> MessageVisibilityRecord:
    return MessageVisibilityRecord(
        id=row["id"],
        save_id=row["save_id"],
        message_id=row["message_id"],
        character_id=row["character_id"],
        visibility=row["visibility"],
        confidence=row["confidence"],
        source=row["source"],
        evidence=row["evidence"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _message_scene_presence_from_row(row: sqlite3.Row) -> MessageScenePresenceRecord:
    return MessageScenePresenceRecord(
        id=row["id"],
        save_id=row["save_id"],
        message_id=row["message_id"],
        character_id=row["character_id"],
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _message_action_choice_from_row(row: sqlite3.Row) -> MessageActionChoiceRecord:
    return MessageActionChoiceRecord(
        id=row["id"],
        save_id=row["save_id"],
        message_id=row["message_id"],
        ordinal=row["ordinal"],
        body=row["body"],
        provider=row["provider"],
        model=row["model"],
        content_rating=row["content_rating"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _job_from_row(row: sqlite3.Row) -> JobRecord:
    result_json = row["result_json"]
    diagnostics_json = row["diagnostics_json"]
    return JobRecord(
        id=row["id"],
        save_id=row["save_id"],
        creator_user_id=row["creator_user_id"],
        type=row["type"],
        status=row["status"],
        payload=_load_object(row["payload_json"]),
        result=_load_object(result_json) if result_json is not None else None,
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        duration_ms=row["duration_ms"],
        diagnostics=(
            _load_object(diagnostics_json)
            if diagnostics_json is not None
            else None
        ),
    )


def _scheduled_task_from_row(row: sqlite3.Row) -> ScheduledTaskRecord:
    result_json = row["result_json"]
    return ScheduledTaskRecord(
        id=row["id"],
        task_type=row["task_type"],
        save_id=row["save_id"],
        enabled=bool(row["enabled"]),
        interval_seconds=row["interval_seconds"],
        next_run_at=row["next_run_at"],
        lease_until=row["lease_until"],
        last_started_at=row["last_started_at"],
        last_completed_at=row["last_completed_at"],
        last_job_id=row["last_job_id"],
        failure_count=row["failure_count"],
        payload=_load_object(row["payload_json"]),
        result=_load_object(result_json) if result_json is not None else None,
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _job_step_from_row(row: sqlite3.Row) -> JobStepRecord:
    return JobStepRecord(
        id=row["id"],
        job_id=row["job_id"],
        name=row["name"],
        status=row["status"],
        provider=row["provider"],
        model=row["model"],
        task=row["task"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        duration_ms=row["duration_ms"],
        error=row["error"],
        metadata=_load_object(row["metadata_json"]),
    )


def _runtime_performance_records(
    rows: list[sqlite3.Row],
    *,
    key_fields: tuple[str, ...],
    limit: int | None = None,
) -> list[RuntimePerformanceRecord]:
    groups: dict[tuple[object, ...], list[sqlite3.Row]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)

    records: list[RuntimePerformanceRecord] = []
    for key, group_rows in groups.items():
        successful_durations = [
            int(row["duration_ms"])
            for row in group_rows
            if _successful_performance_status(str(row["status"]))
            and row["duration_ms"] is not None
        ]
        queue_waits = [
            int(row["queue_wait_ms"])
            for row in group_rows
            if "queue_wait_ms" in row.keys() and row["queue_wait_ms"] is not None
        ]
        terminal_count = sum(
            1
            for row in group_rows
            if str(row["status"]) in _JOB_TERMINAL_STATUSES
        )
        failed_or_cancelled_count = sum(
            1
            for row in group_rows
            if str(row["status"]) in {"failed", "cancelled"}
        )
        latest = _latest_performance_row(group_rows)
        values = dict(zip(key_fields, key, strict=True))
        records.append(
            RuntimePerformanceRecord(
                job_type=cast(str | None, values.get("job_type")),
                step_name=cast(str | None, values.get("step_name")),
                provider=cast(str | None, values.get("provider")),
                model=cast(str | None, values.get("model")),
                task=cast(str | None, values.get("task")),
                sample_count=len(group_rows),
                success_count=sum(
                    1
                    for row in group_rows
                    if _successful_performance_status(str(row["status"]))
                ),
                failed_count=sum(1 for row in group_rows if row["status"] == "failed"),
                cancelled_count=sum(
                    1 for row in group_rows if row["status"] == "cancelled"
                ),
                skipped_count=sum(
                    1
                    for row in group_rows
                    if _skipped_performance_status(str(row["status"]))
                ),
                average_duration_ms=(
                    round(sum(successful_durations) / len(successful_durations))
                    if successful_durations
                    else None
                ),
                p50_duration_ms=_percentile_duration(successful_durations, 0.50),
                p95_duration_ms=_percentile_duration(successful_durations, 0.95),
                min_duration_ms=min(successful_durations)
                if successful_durations
                else None,
                max_duration_ms=max(successful_durations)
                if successful_durations
                else None,
                latest_duration_ms=(
                    cast(int | None, latest["duration_ms"]) if latest else None
                ),
                average_queue_wait_ms=(
                    round(sum(queue_waits) / len(queue_waits)) if queue_waits else None
                ),
                p95_queue_wait_ms=_percentile_duration(queue_waits, 0.95),
                failure_rate=(
                    failed_or_cancelled_count / terminal_count
                    if terminal_count
                    else 0.0
                ),
                latest_completed_at=(
                    cast(str | None, latest["completed_at"]) if latest else None
                ),
            )
        )
    sorted_records = sorted(
        records,
        key=lambda record: record.latest_completed_at or "",
        reverse=True,
    )
    if limit is not None:
        return sorted_records[: max(0, limit)]
    return sorted_records


def _percentile_duration(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, math.ceil(percentile * len(values)) - 1))
    return sorted_values[index]


def _latest_performance_row(rows: list[sqlite3.Row]) -> sqlite3.Row | None:
    completed = [row for row in rows if row["completed_at"] is not None]
    if not completed:
        return rows[-1] if rows else None
    return completed[-1]


def _successful_performance_status(status: str) -> bool:
    return status in {"succeeded", "applied"}


def _skipped_performance_status(status: str) -> bool:
    return status == "deferred" or status.startswith("skipped")
