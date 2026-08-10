from __future__ import annotations

import json
import os
import sqlite3
import stat
from hashlib import sha256
from pathlib import Path

import pytest

from bragi.persistence import migrations
from bragi.persistence.migrations import CURRENT_SCHEMA_VERSION, migrate_database
from bragi.persistence.repositories import (
    PersistenceRepositories,
    canonical_claim_fingerprint,
)
from bragi.text_search import cjk_lexical_anchors

EXPECTED_MIGRATION_VERSIONS = list(range(1, CURRENT_SCHEMA_VERSION + 1))

EXPECTED_TABLES = {
    "active_threads",
    "app_settings",
    "character_contact_states",
    "character_knowledge_edges",
    "character_text_message_attachments",
    "character_text_message_revisions",
    "character_text_activity_events",
    "character_text_messages",
    "character_text_proactive_triggers",
    "character_text_provenance",
    "character_text_threads",
    "characters",
    "context_observations",
    "context_observation_curation_state",
    "context_source_fts",
    "context_sources",
    "context_update_audit",
    "context_update_suggestions",
    "dating_route_states",
    "entity_links",
    "job_steps",
    "jobs",
    "locations",
    "media_assets",
    "memories",
    "message_action_choices",
    "message_context_revisions",
    "message_revisions",
    "message_scene_presence",
    "message_visibility",
    "messages",
    "narrator_phone_activity_cursors",
    "model_preferences",
    "provider_catalog_entries",
    "provider_configs",
    "provider_models",
    "save_assignments",
    "save_context_revisions",
    "save_loss_condition_changes",
    "save_loss_conditions",
    "save_loss_outcomes",
    "save_scenario_updates",
    "save_snapshot_objects",
    "save_turn_snapshots",
    "saves",
    "scenarios",
    "scene_snapshots",
    "scheduled_tasks",
    "schema_migrations",
    "scoped_settings",
    "state_changes",
    "summaries",
    "user_runtime_state",
    "user_sessions",
    "users",
    "world_state",
}

EXPECTED_COLUMNS = {
    "saves": {
        "id",
        "scenario_id",
        "title",
        "custom_instructions",
        "owner_user_id",
        "last_opened_at",
    },
    "messages": {
        "id",
        "save_id",
        "role",
        "speaker_name",
        "body",
        "content_rating",
        "safety_transition",
        "updated_at",
        "deleted_at",
    },
    "characters": {
        "id",
        "save_id",
        "name",
        "age",
        "current_clothing",
        "history",
        "goals",
        "motivations",
        "current_intent",
        "boundaries",
        "attitude_toward_player",
        "cooperation_conditions",
        "protected_from_maintenance",
        "is_player_character",
        "first_seen_message_id",
        "last_updated_message_id",
        "contact_name",
        "texting_style",
        "content_rating",
    },
    "scene_snapshots": {
        "id",
        "save_id",
        "time_of_day",
        "day_of_week",
        "world_day_index",
        "world_time_day_index",
        "world_time_day_label",
        "world_time_phase",
        "world_time_clock_minutes",
        "world_time_period_label",
        "world_time_source_message_id",
        "world_time_confidence",
        "first_seen_message_id",
        "last_updated_message_id",
    },
    "media_assets": {
        "id",
        "save_id",
        "source_message_id",
        "source_media_asset_id",
        "mime_type",
        "metadata_json",
        "archived_at",
    },
    "provider_configs": {
        "id",
        "provider",
        "enabled",
        "has_api_key",
        "last_model_refresh_at",
        "last_error",
    },
    "provider_models": {
        "id",
        "provider",
        "model_id",
        "capabilities_json",
        "supported_parameters_json",
        "available",
        "pricing_json",
        "thinking_json",
    },
    "jobs": {
        "id",
        "save_id",
        "creator_user_id",
        "type",
        "status",
        "duration_ms",
        "diagnostics_json",
    },
    "job_steps": {
        "id",
        "job_id",
        "name",
        "status",
        "provider",
        "model",
        "task",
        "metadata_json",
    },
    "users": {
        "id",
        "username",
        "username_normalized",
        "role",
        "password_hash",
        "status",
    },
    "save_assignments": {"id", "save_id", "user_id", "created_at"},
    "user_runtime_state": {"user_id", "active_save_id", "updated_at"},
    "save_context_revisions": {"save_id", "revision", "updated_at"},
    "message_context_revisions": {
        "message_id",
        "save_id",
        "revision",
        "updated_at",
    },
    "message_action_choices": {
        "id",
        "save_id",
        "message_id",
        "ordinal",
        "body",
        "provider",
        "model",
    },
    "character_contact_states": {
        "id",
        "save_id",
        "player_character_id",
        "character_id",
        "player_has_character_number",
        "character_has_player_number",
        "source_text_message_id",
        "archived_at",
    },
    "character_text_threads": {
        "id",
        "save_id",
        "character_id",
        "status",
        "memory_body",
        "memory_message_count",
        "memory_updated_at",
    },
    "character_text_messages": {
        "id",
        "save_id",
        "thread_id",
        "character_id",
        "sender",
        "body",
        "content_rating",
        "delivery_status",
        "delivery_job_id",
        "reply_to_message_id",
    },
    "character_text_message_revisions": {
        "id",
        "save_id",
        "text_message_id",
        "revision_number",
        "previous_body",
        "new_body",
        "reconciliation_status",
    },
    "character_text_message_attachments": {
        "id",
        "save_id",
        "thread_id",
        "text_message_id",
        "kind",
        "status",
        "media_asset_id",
        "metadata_json",
    },
    "character_text_proactive_triggers": {
        "id",
        "save_id",
        "character_id",
        "trigger_key",
        "trigger_type",
        "source_type",
        "source_id",
    },
}

EXPECTED_INDEXES = {
    "idx_active_threads_save_active_priority_created",
    "idx_character_contact_states_save_active",
    "idx_character_knowledge_edges_save_character",
    "idx_character_text_attachments_text_order",
    "idx_character_text_message_revisions_save_text",
    "idx_character_text_messages_thread_created",
    "idx_character_text_proactive_triggers_save_character",
    "idx_characters_save_active_name_created",
    "idx_characters_save_protected_active",
    "idx_characters_single_active_player_character",
    "idx_context_observations_save_active_created",
    "idx_context_sources_save_active_type_title_created",
    "idx_dating_route_states_save_active",
    "idx_jobs_save_status_completed",
    "idx_jobs_status_type_save_completed",
    "idx_job_steps_provider_model_task_status_completed",
    "idx_locations_save_active_name_created",
    "idx_media_assets_save_active_created",
    "idx_media_assets_save_source_type_active",
    "idx_message_action_choices_save_message_ordinal",
    "idx_message_context_revisions_save_id",
    "idx_messages_save_active_row_order",
    "idx_messages_save_role_active_created",
    "idx_provider_catalog_entries_provider_slug",
    "idx_save_assignments_save_user",
    "idx_save_turn_snapshots_root_manifest",
    "idx_scheduled_tasks_due",
    "idx_scheduled_tasks_type_due",
    "idx_scoped_settings_scope_key",
    "idx_user_sessions_token_hash",
    "idx_world_state_save_active_key",
}

EXPECTED_TRIGGERS = {
    "context_sources_fts_after_delete",
    "context_sources_fts_after_insert",
    "context_sources_fts_after_update",
    "init_save_context_revision_after_save_insert",
    "bump_messages_context_revision_after_insert",
    "bump_messages_context_revision_after_update",
    "bump_messages_context_revision_after_delete",
    "null_location_references_before_location_delete",
}


def test_migrate_database_from_empty_path_creates_current_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        table_names = _object_names(connection, "table")
        index_names = _object_names(connection, "index")
        trigger_names = _object_names(connection, "trigger")

        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS
        assert table_names >= EXPECTED_TABLES
        assert index_names >= EXPECTED_INDEXES
        assert trigger_names >= EXPECTED_TRIGGERS
        for table_name, expected_columns in EXPECTED_COLUMNS.items():
            assert _column_names(connection, table_name) >= expected_columns

        provider_config_columns = _column_names(connection, "provider_configs")
        assert "api_key" not in provider_config_columns
        assert "encrypted_api_key" not in provider_config_columns


def test_migrate_database_is_idempotent_for_current_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        _seed_save(connection)

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS
        assert connection.execute("SELECT title FROM saves").fetchone()[0] == (
            "Night Watch"
        )


def test_migrate_database_rebuilds_incomplete_exact_identifier_index(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A keep in the ash.",
            player_role="Warden",
            content={},
        )
        save = repositories.create_save(
            scenario_id=scenario.id,
            title="Night Watch",
        )
        source = repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id="memory-codes",
            title="Archive codes",
            body=" ".join(f"ARCHIVE-{index:03d}" for index in range(257)),
        )
        connection.execute(
            """
            DELETE FROM context_source_search_index_state
            WHERE key = 'exact_identifiers_complete_v2'
            """
        )
        connection.execute(
            """
            DELETE FROM context_source_exact_identifiers
            WHERE context_source_id = ?
            """,
            (source.id,),
        )
        connection.execute(
            """
            INSERT INTO context_source_exact_identifiers(
                context_source_id, save_id, identifier
            )
            VALUES (?, ?, 'archive-000')
            """,
            (source.id, save.id),
        )
        connection.execute(
            """
            UPDATE context_sources
            SET archived_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (source.id,),
        )
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM context_source_exact_identifiers
            WHERE context_source_id = ?
            """,
            (source.id,),
        ).fetchone()[0] == 0
        repositories = PersistenceRepositories(connection)
        repositories.restore_context_sources({source.id})
        hits = repositories.search_context_sources(
            save.id,
            query_terms={"archive", "128"},
            source_types={"memory"},
            limit=1,
            exact_identifiers=("ARCHIVE-128",),
        )
        assert [hit.record.id for hit in hits] == [source.id]


def test_migrate_database_keeps_outdated_lexical_index_searchable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    body = "".join(chr(0x4E00 + index) for index in range(220))
    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A keep in the ash.",
            player_role="Warden",
            content={},
        )
        save = repositories.create_save(
            scenario_id=scenario.id,
            title="Night Watch",
        )
        source = repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id="memory-long-han-run",
            title="長文",
            body=body,
        )
        connection.execute(
            """
            DELETE FROM context_source_search_terms
            WHERE context_source_id = ?
            """,
            (source.id,),
        )
        connection.executemany(
            """
            INSERT INTO context_source_search_terms(
                context_source_id, save_id, term
            )
            VALUES (?, ?, ?)
            """,
            (
                (source.id, save.id, body[index : index + 2])
                for index in range(len(body) - 1)
            ),
        )
        connection.commit()

    query = body[200:203]
    assert query not in {
        body[index : index + 2] for index in range(len(body) - 1)
    }

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        hits = repositories.search_context_sources(
            save.id,
            query_terms=set(cjk_lexical_anchors(query)),
            source_types={"memory"},
            limit=1,
            match_all=True,
        )
        assert [hit.record.id for hit in hits] == [source.id]


def test_migrate_database_preserves_legacy_per_record_normalized_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A keep in the ash.",
            player_role="Warden",
            content={},
        )
        save = repositories.create_save(
            scenario_id=scenario.id,
            title="Night Watch",
        )
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="custom_note",
            source_id="legacy-expansion",
            title="Legacy expansion",
            body="\ufdfa" * 32,
        )
        normalized_bytes = connection.execute(
            """
            SELECT normalized_text_bytes
            FROM context_source_normalized_budget_entries
            WHERE save_id = ?
            """,
            (save.id,),
        ).fetchone()[0]
        connection.commit()
    monkeypatch.setattr(
        migrations,
        "_MAX_CONTEXT_SOURCE_NORMALIZED_BYTES_PER_RECORD",
        1,
    )

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        stored_limit = connection.execute(
            """
            SELECT normalized_text_bytes
            FROM context_source_legacy_record_budget_limits
            WHERE save_id = ?
            """,
            (save.id,),
        ).fetchone()[0]
        assert stored_limit == normalized_bytes


def test_migrate_database_upgrades_main_schema_71_context_lifecycle(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A keep in the ash.",
            player_role="Warden",
            content={},
        )
        save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
        observation = repositories.add_context_observation(
            save_id=save.id,
            observation_type="character_fact",
            claim="Mara likes tea.",
            evidence_quote="Mara likes tea.",
            source_message_ids=[],
            scope="durable",
            status="accepted",
            confidence=0.95,
            tags=["preference"],
            metadata={
                "curation": {
                    "action": "durable_memory",
                    "memory_body": "Mara Likes Tea!",
                }
            },
        )
        memory = repositories.add_memory(
            save_id=save.id,
            body="Mara Likes Tea!",
            tags=["preference"],
            importance=0.4,
            source_observation_ids=[observation.id],
        )
        duplicate_id = "legacy-duplicate-memory"
        connection.execute(
            "DROP INDEX idx_memories_save_claim_fingerprint_active"
        )
        connection.execute(
            """
            INSERT INTO memories(
                id, save_id, body, tags_json, importance,
                source_message_ids_json, claim_fingerprint,
                source_observation_ids_json
            )
            VALUES (?, ?, ?, ?, ?, '[]', ?, '[]')
            """,
            (
                duplicate_id,
                save.id,
                "mara likes tea",
                json.dumps(["tea"], separators=(",", ":")),
                0.9,
                sha256(b"mara likes tea").hexdigest(),
            ),
        )
        character = repositories.add_character(
            save_id=save.id,
            name="Captain Ilyra",
        )
        archived_keeper_edge = repositories.add_character_knowledge_edge(
            save_id=save.id,
            character_id=character.id,
            target_type="memory",
            target_id=memory.id,
        )
        repositories.archive_character_knowledge_edge(archived_keeper_edge.id)
        repositories.add_character_knowledge_edge(
            save_id=save.id,
            character_id=character.id,
            target_type="memory",
            target_id=duplicate_id,
        )
        privacy_character = repositories.add_character(
            save_id=save.id,
            name="Archivist Ren",
        )
        repositories.add_character_knowledge_edge(
            save_id=save.id,
            character_id=privacy_character.id,
            target_type="memory",
            target_id=memory.id,
            knowledge_state="knows",
            source_message_ids=["keeper-proof"],
        )
        repositories.add_character_knowledge_edge(
            save_id=save.id,
            character_id=privacy_character.id,
            target_type="memory",
            target_id=duplicate_id,
            knowledge_state="does_not_know",
            source_message_ids=["duplicate-proof"],
        )
        archived_keeper_source = repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id=memory.id,
            title="Archived keeper source",
            body=memory.body,
        )
        repositories.archive_context_source(archived_keeper_source.id)
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="memory",
            source_id=duplicate_id,
            title="Active duplicate source",
            body="mara likes tea",
        )
        repositories.add_entity_link(
            save_id=save.id,
            entity_type="character",
            entity_id=character.id,
            target_type="memory",
            target_id=duplicate_id,
            relation="recalls",
        )
        suggestion = repositories.add_context_update_suggestion(
            save_id=save.id,
            update_type="update",
            entity_type="memory",
            entity_id=duplicate_id,
            field_path="tags",
            proposed_value=["tea"],
        )
        audit = repositories.add_context_update_audit(
            save_id=save.id,
            operation="legacy-memory-update",
            entity_type="memory",
            entity_id=duplicate_id,
            field_path="tags",
            before=[],
            after=["tea"],
        )
        connection.execute(
            "ALTER TABLE memories DROP COLUMN source_observation_ids_json"
        )
        connection.execute("ALTER TABLE memories DROP COLUMN claim_fingerprint")
        connection.execute(
            "ALTER TABLE context_sources DROP COLUMN expires_after_turn_number"
        )
        connection.execute(
            "ALTER TABLE context_sources DROP COLUMN created_turn_number"
        )
        connection.execute("ALTER TABLE context_sources DROP COLUMN scene_generation")
        connection.execute("ALTER TABLE context_sources DROP COLUMN scene_snapshot_id")
        connection.execute("ALTER TABLE scene_snapshots DROP COLUMN scene_generation")
        connection.execute("DELETE FROM schema_migrations WHERE version = 72")
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert {
            "scene_snapshot_id",
            "scene_generation",
            "created_turn_number",
            "expires_after_turn_number",
        } <= _column_names(connection, "context_sources")
        assert {
            "claim_fingerprint",
            "source_observation_ids_json",
        } <= _column_names(connection, "memories")
        assert "scene_generation" in _column_names(connection, "scene_snapshots")
        expected_fingerprint = sha256(b"mara likes tea").hexdigest()
        assert connection.execute(
            """
            SELECT claim_fingerprint, source_observation_ids_json
            FROM memories WHERE id = ?
            """,
            (memory.id,),
        ).fetchone() == (
            expected_fingerprint,
            json.dumps([observation.id], separators=(",", ":")),
        )
        active_memories = connection.execute(
            """
            SELECT tags_json, importance
            FROM memories
            WHERE save_id = ? AND archived_at IS NULL
            """,
            (save.id,),
        ).fetchall()
        assert active_memories == [
            (
                json.dumps(["preference", "tea"], separators=(",", ":")),
                0.9,
            )
        ]
        assert connection.execute(
            """
            SELECT target_id
            FROM character_knowledge_edges
            WHERE character_id = ?
            """,
            (character.id,),
        ).fetchone() == (memory.id,)
        assert connection.execute(
            """
            SELECT knowledge_state, source_message_ids_json
            FROM character_knowledge_edges
            WHERE character_id = ? AND archived_at IS NULL
            """,
            (privacy_character.id,),
        ).fetchone() == (
            "does_not_know",
            json.dumps(
                ["keeper-proof", "duplicate-proof"],
                separators=(",", ":"),
            ),
        )
        assert connection.execute(
            """
            SELECT source_id, title
            FROM context_sources
            WHERE save_id = ? AND source_type = 'memory'
              AND archived_at IS NULL
            """,
            (save.id,),
        ).fetchone() == (memory.id, "Active duplicate source")
        assert connection.execute(
            """
            SELECT target_id
            FROM entity_links
            WHERE entity_type = 'character' AND entity_id = ?
            """,
            (character.id,),
        ).fetchone() == (memory.id,)
        assert connection.execute(
            "SELECT entity_id FROM context_update_suggestions WHERE id = ?",
            (suggestion.id,),
        ).fetchone() == (memory.id,)
        assert connection.execute(
            "SELECT entity_id FROM context_update_audit WHERE id = ?",
            (audit.id,),
        ).fetchone() == (memory.id,)
        index_row = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'index'
              AND name = 'idx_memories_save_claim_fingerprint_active'
            """
        ).fetchone()
        assert index_row is not None
        assert "CREATE UNIQUE INDEX" in index_row[0]
        assert "WHERE archived_at IS NULL" in index_row[0]


def test_current_schema_repairs_nonunique_memory_fingerprint_index(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A keep in the ash.",
            player_role="Warden",
            content={},
        )
        save = repositories.create_save(
            scenario_id=scenario.id,
            title="Night Watch",
        )
        keeper = repositories.add_memory(
            save_id=save.id,
            body="Mara likes tea.",
            tags=["preference"],
        )
        connection.execute(
            "DROP INDEX idx_memories_save_claim_fingerprint_active"
        )
        connection.execute(
            """
            CREATE INDEX idx_memories_save_claim_fingerprint_active
            ON memories(save_id, claim_fingerprint)
            WHERE archived_at IS NULL AND claim_fingerprint != ''
            """
        )
        connection.execute(
            """
            INSERT INTO memories(
                id, save_id, body, tags_json, importance,
                source_message_ids_json, claim_fingerprint,
                source_observation_ids_json
            )
            VALUES (
                'duplicate-memory', ?, 'mara likes tea', '["dossier"]', 0.9,
                '["message-duplicate"]', ?, '[]'
            )
            """,
            (
                save.id,
                canonical_claim_fingerprint("mara likes tea"),
            ),
        )
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        active_ids = connection.execute(
            """
            SELECT id FROM memories
            WHERE save_id = ? AND archived_at IS NULL
            """,
            (save.id,),
        ).fetchall()
        assert active_ids == [(keeper.id,)]
        index_row = next(
            row
            for row in connection.execute("PRAGMA index_list('memories')")
            if row[1] == "idx_memories_save_claim_fingerprint_active"
        )
        assert index_row[2] == 1


def test_migration_72_to_73_adds_summary_lineage_and_repairs_message_estimates(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A keep in the ash.",
            player_role="Warden",
            content={},
        )
        save = repositories.create_save(
            scenario_id=scenario.id,
            title="Night Watch",
        )
        message = repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="The bell rings twice.",
            provider="fake",
            model="fake-chat",
            token_estimate=999,
        )
        repositories.add_summary(
            save_id=save.id,
            covers_message_start_id=message.id,
            covers_message_end_id=message.id,
            body="The bell rang twice.",
            provider="fake",
            model="fake-summary",
        )
        connection.execute("ALTER TABLE summaries DROP COLUMN source_summary_ids_json")
        connection.execute("ALTER TABLE summaries DROP COLUMN source_message_ids_json")
        connection.execute("DELETE FROM schema_migrations WHERE version = 73")
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert {
            "source_message_ids_json",
            "source_summary_ids_json",
        } <= _column_names(connection, "summaries")
        assert connection.execute(
            "SELECT token_estimate FROM messages WHERE id = ?",
            (message.id,),
        ).fetchone() == (6,)
        assert connection.execute(
            """
            SELECT source_message_ids_json, source_summary_ids_json
            FROM summaries
            """
        ).fetchone() == ("[]", "[]")
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS


def test_migration_73_to_74_adds_roleplay_interaction_mode_defaults(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("ALTER TABLE scenarios DROP COLUMN interaction_mode")
        connection.execute("ALTER TABLE saves DROP COLUMN interaction_mode")
        connection.execute("DELETE FROM schema_migrations WHERE version = 74")
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        scenario_columns = _column_names(connection, "scenarios")
        save_columns = _column_names(connection, "saves")
        assert "interaction_mode" in scenario_columns
        assert "interaction_mode" in save_columns
        scenario_id = "legacy-scenario"
        connection.execute(
            """
            INSERT INTO scenarios(
                id, type, title, premise, player_role, content_json
            ) VALUES (?, 'full_roleplay', 'Legacy', '', '', '{}')
            """,
            (scenario_id,),
        )
        connection.execute(
            """
            INSERT INTO saves(id, scenario_id, title)
            VALUES ('legacy-save', ?, 'Legacy Save')
            """,
            (scenario_id,),
        )
        assert connection.execute(
            "SELECT interaction_mode FROM scenarios WHERE id = ?",
            (scenario_id,),
        ).fetchone() == ("roleplay",)
        assert connection.execute(
            "SELECT interaction_mode FROM saves WHERE id = 'legacy-save'"
        ).fetchone() == ("roleplay",)
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS


def test_migration_74_to_75_adds_scene_fact_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE scene_fact_sources")
        connection.execute("DROP TABLE scene_facts")
        connection.execute("DELETE FROM schema_migrations WHERE version >= 75")
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"scene_facts", "scene_fact_sources"} <= tables
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS


def test_migration_75_to_76_marks_existing_knowledge_legacy_unclassified(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Legacy",
            premise="",
            player_role="Warden",
            content={},
        )
        save = repositories.create_save(scenario_id=scenario.id, title="Legacy")
        repositories.add_memory(save_id=save.id, body="An old fact.", tags=[])
        repositories.add_context_observation(
            save_id=save.id,
            observation_type="world_fact",
            claim="An old observation.",
        )
        for table_name in ("memories", "context_observations"):
            connection.execute(
                f"ALTER TABLE {table_name} DROP COLUMN epistemic_actor_name"
            )
            connection.execute(
                f"ALTER TABLE {table_name} DROP COLUMN epistemic_actor_id"
            )
            connection.execute(
                f"ALTER TABLE {table_name} DROP COLUMN epistemic_status"
            )
        connection.execute("DELETE FROM schema_migrations WHERE version = 76")
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT epistemic_status FROM memories"
        ).fetchone() == ("legacy_unclassified",)
        assert connection.execute(
            "SELECT epistemic_status FROM context_observations"
        ).fetchone() == ("legacy_unclassified",)
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS


def test_current_schema_repairs_missing_epistemic_columns(tmp_path: Path) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "ALTER TABLE context_observations DROP COLUMN epistemic_actor_name"
        )
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(context_observations)")
        }
        assert "epistemic_actor_name" in columns
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS


def test_current_schema_repair_bounds_memory_observation_backfill(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A keep in the ash.",
            player_role="Warden",
            content={},
        )
        save = repositories.create_save(
            scenario_id=scenario.id,
            title="Night Watch",
        )
        memory = repositories.add_memory(
            save_id=save.id,
            body="Mara likes tea.",
            tags=["preference"],
        )
        for index in range(65):
            repositories.add_context_observation(
                save_id=save.id,
                observation_type="character_fact",
                claim="Mara likes tea.",
                scope="durable",
                status="accepted",
                metadata={
                    "curation": {
                        "action": "durable_memory",
                        "memory_body": "Mara likes tea.",
                    }
                },
                observation_id=f"observation-{index:02d}",
            )
        connection.execute(
            "DROP INDEX idx_memories_save_claim_fingerprint_active"
        )
        connection.execute(
            """
            CREATE INDEX idx_memories_save_claim_fingerprint_active
            ON memories(save_id, claim_fingerprint)
            WHERE archived_at IS NULL AND claim_fingerprint != ''
            """
        )
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        source_ids_json = connection.execute(
            """
            SELECT source_observation_ids_json
            FROM memories
            WHERE id = ?
            """,
            (memory.id,),
        ).fetchone()[0]
        assert len(json.loads(source_ids_json)) == 64


def test_migration_rejects_orphaned_pending_review_suggestions(tmp_path: Path) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A keep in the ash.",
            player_role="Warden",
            content={},
        )
        save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
        suggestion = repositories.add_context_update_suggestion(
            save_id=save.id,
            update_type="field_update",
            entity_type="character",
            entity_id="missing-character",
            field_path="goals",
            proposed_value="Leave.",
        )
        connection.execute("DELETE FROM schema_migrations WHERE version >= 65")
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT status, next_review_at
            FROM context_update_suggestions WHERE id = ?
            """,
            (suggestion.id,),
        ).fetchone()
        assert row == ("rejected", None)
        assert connection.execute(
            """
            SELECT operation FROM context_update_audit
            WHERE suggestion_id = ?
            """,
            (suggestion.id,),
        ).fetchone() == ("agent_suggestion_preflight_reject",)


def test_migrate_database_upgrades_schema_61_world_time_columns(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version >= 62")
        connection.execute(
            "ALTER TABLE scene_snapshots DROP COLUMN world_time_day_index"
        )
        connection.execute(
            "ALTER TABLE scene_snapshots DROP COLUMN world_time_day_label"
        )
        connection.execute("ALTER TABLE scene_snapshots DROP COLUMN world_time_phase")
        connection.execute(
            "ALTER TABLE scene_snapshots DROP COLUMN world_time_clock_minutes"
        )
        connection.execute(
            "ALTER TABLE scene_snapshots DROP COLUMN world_time_period_label"
        )
        connection.execute(
            "ALTER TABLE scene_snapshots DROP COLUMN world_time_source_message_id"
        )
        connection.execute(
            "ALTER TABLE scene_snapshots DROP COLUMN world_time_confidence"
        )
        connection.execute(
            """
            INSERT INTO scenarios(id, type, title, content_json)
            VALUES ('scenario-legacy', 'full_roleplay', 'Legacy', '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO saves(id, scenario_id, title)
            VALUES ('save-legacy', 'scenario-legacy', 'Legacy Save')
            """
        )
        connection.execute(
            """
            INSERT INTO saves(id, scenario_id, title)
            VALUES ('save-sparse', 'scenario-legacy', 'Sparse Legacy Save')
            """
        )
        connection.execute(
            """
            INSERT INTO messages(id, save_id, role, body)
            VALUES ('message-legacy', 'save-legacy', 'narrator', 'Friday night.')
            """
        )
        connection.execute(
            """
            INSERT INTO scene_snapshots(
                id, save_id, in_world_time, time_of_day, day_of_week,
                world_day_index, source_message_id
            )
            VALUES (
                'scene-legacy', 'save-legacy',
                'Friday 9:41 PM after the festival', 'evening', 'friday',
                5, 'message-legacy'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO scene_snapshots(
                id, save_id, in_world_time, time_of_day, day_of_week,
                world_day_index, source_message_id
            )
            VALUES (
                'scene-sparse', 'save-sparse',
                'Friday late morning after the dance', '', '',
                6, 'message-legacy'
            )
            """
        )

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS
        row = connection.execute(
            """
            SELECT world_time_day_index, world_time_day_label,
                   world_time_phase, world_time_clock_minutes,
                   world_time_period_label, world_time_source_message_id,
                   world_time_confidence
            FROM scene_snapshots
            WHERE id = 'scene-legacy'
            """
        ).fetchone()
        assert row == (5, "friday", "evening", 21 * 60 + 41, "", "message-legacy", None)
        sparse = connection.execute(
            """
            SELECT world_time_day_index, world_time_day_label,
                   world_time_phase, world_time_clock_minutes,
                   world_time_period_label, world_time_source_message_id,
                   world_time_confidence
            FROM scene_snapshots
            WHERE id = 'scene-sparse'
            """
        ).fetchone()
        assert sparse == (
            6,
            "",
            "late_morning",
            None,
            "",
            "message-legacy",
            None,
        )


def test_character_current_clothing_schema_helper_adds_missing_column(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE characters (
                id TEXT PRIMARY KEY,
                save_id TEXT NOT NULL,
                name TEXT NOT NULL,
                visual_notes TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            """
            INSERT INTO characters(id, save_id, name, visual_notes)
            VALUES ('character-1', 'save-1', 'Mara', 'Warm lantern light.')
            """
        )
        migrations._ensure_character_current_clothing_schema(connection)
        connection.commit()

    with sqlite3.connect(database_path) as connection:
        assert "current_clothing" in _column_names(connection, "characters")
        assert connection.execute(
            "SELECT current_clothing FROM characters WHERE id = 'character-1'"
        ).fetchone()[0] == ""


def test_migrate_database_repairs_legacy_character_text_group_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    with sqlite3.connect(database_path) as connection:
        _create_schema_migrations(connection)
        connection.executemany(
            "INSERT INTO schema_migrations(version) VALUES (?)",
            [(version,) for version in EXPECTED_MIGRATION_VERSIONS],
        )
        connection.executescript(
            """
            CREATE TABLE saves (
                id TEXT PRIMARY KEY
            );

            CREATE TABLE characters (
                id TEXT PRIMARY KEY,
                save_id TEXT NOT NULL,
                name TEXT NOT NULL,
                is_player_character INTEGER NOT NULL DEFAULT 0,
                archived_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE character_text_threads (
                id TEXT PRIMARY KEY,
                save_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                archived_at TEXT,
                memory_body TEXT NOT NULL DEFAULT '',
                memory_message_count INTEGER NOT NULL DEFAULT 0,
                memory_updated_at TEXT
            );

            CREATE TABLE character_text_messages (
                id TEXT PRIMARY KEY,
                save_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                body TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                token_estimate INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                deleted_at TEXT,
                delivery_status TEXT NOT NULL DEFAULT 'sent',
                delivery_error TEXT,
                delivery_job_id TEXT,
                delivery_attempt INTEGER NOT NULL DEFAULT 0,
                in_world_sent_at TEXT,
                delivered_at TEXT,
                read_at TEXT,
                reply_to_message_id TEXT
            );
            """
        )
        connection.execute("INSERT INTO saves(id) VALUES (?)", ("save-1",))
        connection.execute(
            """
            INSERT INTO characters(id, save_id, name)
            VALUES (?, ?, ?)
            """,
            ("character-1", "save-1", "Mara"),
        )
        connection.execute(
            """
            INSERT INTO character_text_threads(
                id, save_id, character_id, title, status
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            ("thread-1", "save-1", "character-1", "Mara", "active"),
        )
        connection.execute(
            """
            INSERT INTO character_text_messages(
                id, save_id, thread_id, character_id, sender, body
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "text-message-1",
                "save-1",
                "thread-1",
                "character-1",
                "character",
                "On my way.",
            ),
        )

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert "kind" in _column_names(connection, "character_text_threads")
        assert "sender_character_id" in _column_names(
            connection,
            "character_text_messages",
        )
        assert _object_names(connection, "index") >= {
            "idx_character_text_threads_save_direct",
            "idx_character_text_participants_thread",
        }
        assert connection.execute(
            """
            SELECT kind, character_id, memory_body
            FROM character_text_threads
            WHERE id = ?
            """,
            ("thread-1",),
        ).fetchone() == ("direct", "character-1", "")
        assert connection.execute(
            """
            SELECT thread_id, character_id, ordinal
            FROM character_text_thread_participants
            WHERE thread_id = ?
            """,
            ("thread-1",),
        ).fetchone() == ("thread-1", "character-1", 0)
        assert connection.execute(
            """
            SELECT body, sender_character_id
            FROM character_text_messages
            WHERE id = ?
            """,
            ("text-message-1",),
        ).fetchone() == ("On my way.", "character-1")


def test_migrate_database_repairs_current_legacy_jobs_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    with sqlite3.connect(database_path) as connection:
        _create_schema_migrations(connection)
        connection.executemany(
            "INSERT INTO schema_migrations(version) VALUES (?)",
            [(version,) for version in EXPECTED_MIGRATION_VERSIONS],
        )
        connection.executescript(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                save_id TEXT,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                completed_at TEXT
            );
            """
        )
        connection.execute(
            """
            INSERT INTO jobs(id, type, status, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            ("job-1", "chat_completion", "queued", "{}"),
        )

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert _column_names(connection, "jobs") >= {
            "creator_user_id",
            "duration_ms",
            "diagnostics_json",
        }
        repositories = PersistenceRepositories(connection)
        jobs = repositories.list_jobs_by_status(("queued",))

    assert [job.id for job in jobs] == ["job-1"]
    assert jobs[0].diagnostics is None


def test_migrate_database_removes_retired_character_import_routing_state(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version >= 66")
        connection.executemany(
            """
            INSERT INTO model_preferences(id, task, provider, model_id)
            VALUES (?, ?, 'openrouter', 'openai/gpt-5-mini')
            """,
            (
                ("retired-chat", "chat_character_interaction"),
                ("retired-context", "character_interaction_context_update"),
                ("dating-context", "dating_sim_context_update"),
                ("image-description", "character_image_description"),
            ),
        )
        thinking = {
            task: {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "low",
            }
            for task in (
                "chat_character_interaction",
                "character_interaction_context_update",
                "dating_sim_context_update",
                "character_image_description",
            )
        }
        connection.execute(
            """
            INSERT INTO scoped_settings(scope, scope_id, key, value_json)
            VALUES ('global', '', 'model_thinking_preferences', ?)
            """,
            (json.dumps(thinking),),
        )
        overrides = {
            "preferences": {
                task: {
                    "provider": "openrouter",
                    "model_id": "openai/gpt-5-mini",
                }
                for task in thinking
            },
            "thinking": thinking,
        }
        connection.execute(
            """
            INSERT INTO scoped_settings(scope, scope_id, key, value_json)
            VALUES ('save', 'save-1', 'save_model_overrides', ?)
            """,
            (json.dumps(overrides),),
        )
        connection.executemany(
            """
            INSERT INTO jobs(id, type, status, payload_json)
            VALUES (?, ?, 'completed', '{}')
            """,
            (
                ("retired-job", "venice_character_import"),
                ("chat-job", "chat_completion"),
            ),
        )
        connection.executemany(
            """
            INSERT INTO job_steps(id, job_id, name, status)
            VALUES (?, ?, 'run', 'completed')
            """,
            (
                ("retired-step", "retired-job"),
                ("chat-step", "chat-job"),
            ),
        )
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        tasks = {
            row[0]
            for row in connection.execute(
                "SELECT task FROM model_preferences ORDER BY task"
            )
        }
        persisted_thinking = json.loads(
            connection.execute(
                """
                SELECT value_json FROM scoped_settings
                WHERE scope = 'global' AND scope_id = ''
                  AND key = 'model_thinking_preferences'
                """
            ).fetchone()[0]
        )
        persisted_overrides = json.loads(
            connection.execute(
                """
                SELECT value_json FROM scoped_settings
                WHERE scope = 'save' AND scope_id = 'save-1'
                  AND key = 'save_model_overrides'
                """
            ).fetchone()[0]
        )
        jobs = {
            row[0] for row in connection.execute("SELECT id FROM jobs ORDER BY id")
        }
        steps = {
            row[0]
            for row in connection.execute("SELECT id FROM job_steps ORDER BY id")
        }

    retained_tasks = {"dating_sim_context_update", "character_image_description"}
    assert tasks == retained_tasks
    assert set(persisted_thinking) == retained_tasks
    assert set(persisted_overrides["preferences"]) == retained_tasks
    assert set(persisted_overrides["thinking"]) == retained_tasks
    assert jobs == {"chat-job"}
    assert steps == {"chat-step"}


@pytest.mark.parametrize("current_schema_version", [66, 67, 68, 69])
def test_migrate_database_strips_deprecated_scenario_character_sections(
    tmp_path: Path,
    current_schema_version: int,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ledger Road",
            premise="A caravan must turn debt into profit.",
            player_role="Caravan factor",
            content={
                "title": "Ledger Road",
                "premise": "A caravan must turn debt into profit.",
                "player_role": "Caravan factor",
                "factions": "Kesh brokers",
                "characters": "Mara Voss and Ren the bell debtor.",
                "romance_options": "Mika Arai watches the station doors.",
                "rivals_and_factions": "A rival guild wants the contract.",
                "reputation_and_contacts": "Trusted by Red Harbor taxmen.",
                "character_starters": [
                    {
                        "name": "Mika Arai",
                        "role": "Station diplomat",
                        "known_state": "Mika knows the locked departure board.",
                    }
                ],
            },
        )
        save = repositories.create_save(
            scenario_id=scenario.id,
            title="Ledger Road Save",
        )
        scenario_update = repositories.add_save_scenario_update(
            save_id=save.id,
            title="Ledger Road: Bridge Debt",
            premise="The caravan owes a bridge toll.",
            player_role="Caravan factor",
            content={
                "title": "Ledger Road: Bridge Debt",
                "premise": "The caravan owes a bridge toll.",
                "player_role": "Caravan factor",
                "factions": "Bridge assessors",
                "major_npcs": "Orlen keeps the toll ledger.",
                "rivals_and_factions": "Kesh brokers contest the bridge debt.",
                "reputation_and_contacts": "Known to the Red Harbor tax office.",
            },
            reason="Legacy update fixture.",
            provider="fake-provider",
            model="fake-model",
        )
        null_only_scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Null Ledger",
            premise="A ledger contains blank legacy fields.",
            player_role="Auditor",
            content={"title": "Null Ledger"},
        )
        connection.execute(
            "UPDATE scenarios SET content_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "title": "Null Ledger",
                        "characters": None,
                        "reputation_and_contacts": None,
                    }
                ),
                null_only_scenario.id,
            ),
        )
        null_only_update = repositories.add_save_scenario_update(
            save_id=save.id,
            title="Null Ledger Update",
            premise="The blank fields remain blank.",
            player_role="Auditor",
            content={"title": "Null Ledger Update"},
            reason="Legacy null fixture.",
            provider="fake-provider",
            model="fake-model",
        )
        connection.execute(
            "UPDATE save_scenario_updates SET content_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "title": "Null Ledger Update",
                        "major_npcs": None,
                        "rivals_and_factions": None,
                    }
                ),
                null_only_update.id,
            ),
        )
        connection.execute(
            "DELETE FROM schema_migrations WHERE version > ?",
            (current_schema_version,),
        )
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT content_json FROM scenarios WHERE id = ?",
            (scenario.id,),
        ).fetchone()
        assert row is not None
        content = json.loads(row[0])
        assert content["factions"] == (
            "Kesh brokers\n\n"
            "A rival guild wants the contract.\n\n"
            "Trusted by Red Harbor taxmen."
        )
        assert content["character_starters"] == [
            {
                "name": "Mika Arai",
                "role": "Station diplomat",
                "known_state": "Mika knows the locked departure board.",
            }
        ]
        assert "characters" not in content
        assert "romance_options" not in content
        assert "rivals_and_factions" not in content
        assert "reputation_and_contacts" not in content
        row = connection.execute(
            "SELECT content_json FROM save_scenario_updates WHERE id = ?",
            (scenario_update.id,),
        ).fetchone()
        assert row is not None
        update_content = json.loads(row[0])
        assert update_content["factions"] == (
            "Bridge assessors\n\n"
            "Kesh brokers contest the bridge debt.\n\n"
            "Known to the Red Harbor tax office."
        )
        assert "major_npcs" not in update_content
        assert "rivals_and_factions" not in update_content
        assert "reputation_and_contacts" not in update_content
        row = connection.execute(
            "SELECT content_json FROM scenarios WHERE id = ?",
            (null_only_scenario.id,),
        ).fetchone()
        assert row is not None
        null_only_content = json.loads(row[0])
        assert null_only_content == {"title": "Null Ledger"}
        row = connection.execute(
            "SELECT content_json FROM save_scenario_updates WHERE id = ?",
            (null_only_update.id,),
        ).fetchone()
        assert row is not None
        null_only_update_content = json.loads(row[0])
        assert null_only_update_content == {"title": "Null Ledger Update"}


def test_migrate_database_recovers_empty_schema_migrations_table(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS
        assert _object_names(connection, "table") >= EXPECTED_TABLES


def test_migrate_database_rejects_empty_schema_migrations_with_app_tables(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE saves (
                id TEXT PRIMARY KEY
            );
            """
        )

    with pytest.raises(RuntimeError, match="schema_migrations has no applied versions"):
        migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert _migration_versions(connection) == []


def test_migrate_database_rejects_app_tables_without_schema_migrations(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE saves (
                id TEXT PRIMARY KEY
            )
            """
        )

    with pytest.raises(RuntimeError, match="existing tables are present: saves"):
        migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert "schema_migrations" not in _object_names(connection, "table")


def test_migrate_database_rolls_back_failed_baseline_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"

    def fail_late_baseline_step(connection: sqlite3.Connection) -> None:
        raise RuntimeError("late baseline failure")

    monkeypatch.setattr(
        migrations,
        "_ensure_context_revision_schema",
        fail_late_baseline_step,
    )

    with pytest.raises(RuntimeError, match="late baseline failure"):
        migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        table_names = _object_names(connection, "table")
        assert "schema_migrations" not in table_names
        assert "saves" not in table_names


def test_migrate_database_rejects_historical_schema_versions(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    with sqlite3.connect(database_path) as connection:
        _create_schema_migrations(connection)
        connection.execute("INSERT INTO schema_migrations(version) VALUES (60)")
        connection.execute("CREATE TABLE sentinel(value TEXT)")

    with pytest.raises(RuntimeError, match="version 60 is no longer supported"):
        migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert _migration_versions(connection) == [60]
        assert _object_names(connection, "table") == {"schema_migrations", "sentinel"}


def test_migrate_database_rejects_future_schema_versions(tmp_path: Path) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    future_version = CURRENT_SCHEMA_VERSION + 1
    with sqlite3.connect(database_path) as connection:
        _create_schema_migrations(connection)
        connection.execute(
            "INSERT INTO schema_migrations(version) VALUES (?)",
            (future_version,),
        )

    with pytest.raises(RuntimeError, match=f"version {future_version} is newer"):
        migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        assert _migration_versions(connection) == [future_version]


def test_context_source_fts_triggers_track_current_schema_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        _seed_save(connection)
        connection.execute(
            """
            INSERT INTO context_sources(
                id, save_id, source_type, source_id, title, body
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("context-1", "save-1", "memory", "memory-1", "Beacon", "red lens"),
        )

        assert _fts_context_source_ids(connection, "beacon") == ["memory-1"]

        connection.execute(
            """
            UPDATE context_sources
            SET title = ?, body = ?
            WHERE id = ?
            """,
            ("Gatehouse", "ashfall warning", "context-1"),
        )

        assert _fts_context_source_ids(connection, "beacon") == []
        assert _fts_context_source_ids(connection, "ashfall") == ["memory-1"]

        connection.execute("DELETE FROM context_sources WHERE id = ?", ("context-1",))

        assert _fts_context_source_ids(connection, "ashfall") == []


def test_context_revision_triggers_track_current_schema_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        _seed_save(connection)
        assert _save_context_revision(connection) == 0

        connection.execute(
            """
            INSERT INTO messages(id, save_id, role, body)
            VALUES (?, ?, ?, ?)
            """,
            ("message-1", "save-1", "narrator", "Ash falls."),
        )

        assert _save_context_revision(connection) == 1
        assert connection.execute(
            """
            SELECT revision
            FROM message_context_revisions
            WHERE message_id = ?
            """,
            ("message-1",),
        ).fetchone()[0] == 1


def test_migrate_database_creates_private_parent_and_database_file(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state" / "bragi" / "bragi.sqlite3"

    migrate_database(database_path)

    if os.name != "nt":
        assert stat.S_IMODE(database_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(database_path.stat().st_mode) == 0o600


def _create_schema_migrations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _seed_save(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        INSERT INTO scenarios(id, type, title, content_json)
        VALUES (?, ?, ?, ?)
        """,
        ("scenario-1", "full_roleplay", "Ashfall Keep", "{}"),
    )
    connection.execute(
        """
        INSERT INTO saves(id, scenario_id, title)
        VALUES (?, ?, ?)
        """,
        ("save-1", "scenario-1", "Night Watch"),
    )


def _object_names(connection: sqlite3.Connection, object_type: str) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = ?
              AND name NOT LIKE 'sqlite_autoindex_%'
            """,
            (object_type,),
        )
    }


def _column_names(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")}


def _migration_versions(connection: sqlite3.Connection) -> list[int]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        )
    ]


def _fts_context_source_ids(
    connection: sqlite3.Connection,
    query: str,
) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            """
            SELECT context_sources.source_id
            FROM context_source_fts
            JOIN context_sources
              ON context_sources.rowid = context_source_fts.rowid
            WHERE context_source_fts MATCH ?
              AND context_sources.archived_at IS NULL
            ORDER BY context_sources.source_id
            """,
            (query,),
        )
    ]


def _save_context_revision(connection: sqlite3.Connection) -> int:
    return int(
        connection.execute(
            """
            SELECT revision
            FROM save_context_revisions
            WHERE save_id = ?
            """,
            ("save-1",),
        ).fetchone()[0]
    )


def test_migration_76_to_77_adds_turn_outcomes_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE turn_outcomes")
        connection.execute("DELETE FROM schema_migrations WHERE version = 77")
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = _column_names(connection, "turn_outcomes")
        assert {"id", "save_id", "message_id", "payload_json", "created_at"} <= columns
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS


def test_migration_78_to_79_adds_action_choice_generation_claims(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE message_action_choice_generation_claims")
        connection.execute("DELETE FROM schema_migrations WHERE version = 79")
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = _column_names(
            connection,
            "message_action_choice_generation_claims",
        )
        assert {
            "message_id",
            "save_id",
            "narrator_updated_at",
            "generation_token",
            "created_at",
            "updated_at",
        } <= columns
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS


def test_migration_77_keeps_existing_turn_outcome_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO scenarios(id, type, title, content_json)
            VALUES ('s', 'full_roleplay', 'Legacy', '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO saves(id, scenario_id, title)
            VALUES ('save-1', 's', 'Keep')
            """
        )
        connection.execute(
            """
            INSERT INTO messages(save_id, role, speaker_name, body)
            VALUES ('save-1', 'narrator', 'Narrator', 'body')
            """
        )
        message_id = connection.execute(
            "SELECT id FROM messages WHERE save_id = 'save-1'"
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO turn_outcomes(id, save_id, message_id, payload_json)
            VALUES ('o-1', 'save-1', ?, '{"save_id": "save-1", "message_id": "m"}')
            """,
            (message_id,),
        )
        connection.execute("DELETE FROM schema_migrations WHERE version = 77")
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT id, payload_json FROM turn_outcomes WHERE id = 'o-1'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "o-1"


def test_migration_77_to_78_backfills_summary_pressure_state(tmp_path: Path) -> None:
    database_path = tmp_path / "summary-pressure.db"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version = 78")
        connection.execute("DROP TRIGGER init_summary_pressure_state_after_save_insert")
        connection.execute("DROP TABLE summary_pressure_state")
        connection.execute(
            "INSERT INTO scenarios(id, type, title, content_json) VALUES (?, ?, ?, ?)",
            ("scenario-1", "full_roleplay", "Bridge", "{}"),
        )
        connection.execute(
            "INSERT INTO saves(id, scenario_id, title) VALUES (?, ?, ?)",
            ("save-1", "scenario-1", "Crossing"),
        )
        connection.executemany(
            """
            INSERT INTO messages(
                id, save_id, role, body, token_estimate, deleted_at
            ) VALUES (?, 'save-1', ?, ?, ?, NULL)
            """,
            (
                ("player-1", "player", "Old player input", 10),
                ("narrator-1", "narrator", "Old narration", 20),
                ("player-2", "player", "Fresh player input", None),
            ),
        )
        connection.execute(
            """
            INSERT INTO summaries(
                id, save_id, covers_message_start_id, covers_message_end_id,
                body, provider, model
            ) VALUES (?, 'save-1', ?, ?, ?, ?, ?)
            """,
            (
                "summary-1",
                "player-1",
                "narrator-1",
                "The bridge was crossed.",
                "fake",
                "fake-summary",
            ),
        )
        connection.commit()

    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM summary_pressure_state WHERE save_id = 'save-1'"
        ).fetchone()
        assert row is not None
        assert row["summarized_through_message_id"] == "narrator-1"
        assert row["unsummarized_message_count"] == 1
        assert row["unsummarized_player_count"] == 1
        assert row["unsummarized_narrator_count"] == 0
        assert row["unsummarized_other_count"] == 0
        assert row["unsummarized_token_estimate"] == 5
        assert row["active_summary_count"] == 1
        assert row["active_summary_token_estimate"] == 6
        assert _migration_versions(connection) == EXPECTED_MIGRATION_VERSIONS
