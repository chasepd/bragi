from __future__ import annotations

import importlib
import json
import sqlite3
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from pytest import MonkeyPatch

from bragi.interaction_mode import InteractionMode
from bragi.persistence import repositories as repositories_module
from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import (
    PersistenceRepositories,
    canonical_claim_fingerprint,
)
from bragi.services import chat_bundle_service as chat_bundle_module
from bragi.services.chat_bundle_service import (
    _coalesce_import_context_sources,
    _coalesce_import_entity_links,
    _coalesce_import_knowledge_edges,
    _coalesce_import_proactive_triggers,
    _remapped_character_text_trigger_key,
)
from bragi.services.director_pressure_service import DIRECTOR_PRESSURE_STATE_KEY
from bragi.services.generation_settings import MODEL_THINKING_PREFERENCES_SETTING
from bragi.services.image_style_settings import (
    IMAGE_STYLE_PRESET_SETTING,
    save_image_style_preset_setting_key,
)
from bragi.services.model_preferences import SAVE_MODEL_OVERRIDES_SETTING
from bragi.services.post_turn_inference import (
    POST_TURN_INFERENCE_MODE_DEFAULT,
    POST_TURN_INFERENCE_MODE_SETTING,
)
from bragi.services.scenario_evolution_policy import (
    SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
    save_scenario_evolution_turn_interval_setting_key,
    scenario_template_evolution_turn_interval_setting_key,
)
from bragi.services.turn_snapshot_service import TurnSnapshotService

SCENARIO_ID = "scenario-ashfall"
SAVE_ID = "save-night-watch"
PLAYER_MESSAGE_ID = "message-player-1"
NARRATOR_MESSAGE_ID = "message-narrator-1"
SCENARIO_UPDATE_ID = "scenario-update-red-lens"
MEDIA_ASSET_ID = "media-beacon-image"
VIDEO_MEDIA_ASSET_ID = "media-beacon-video"
LOSS_CONDITION_ID = "loss-condition-beacon-collapse"
LOSS_CONDITION_CHANGE_ID = "loss-condition-change-beacon-collapse"
LOSS_OUTCOME_ID = "loss-outcome-beacon-collapse"
LOSS_EPILOGUE_MESSAGE_ID = "message-loss-epilogue"
OBSERVATION_ID = "observation-beacon-warning"
MEDIA_BYTES = b"red beacon lens image bytes"
MEDIA_PATH = "save-night-watch/images/beacon.png"
VIDEO_MEDIA_BYTES = b"red beacon lens video bytes"
VIDEO_MEDIA_PATH = "save-night-watch/videos/beacon.mp4"


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_legacy_import_rows_coalesce_after_memory_id_remapping() -> None:
    active_source: dict[str, object] = {
        "id": "source-active",
        "save_id": "target-save",
        "source_type": "memory",
        "source_id": "merged-memory",
        "archived_at": None,
    }
    sources = _coalesce_import_context_sources(
        [
            {
                **active_source,
                "id": "source-archived",
                "archived_at": "2026-01-01",
            },
            active_source,
        ]
    )
    links = _coalesce_import_entity_links(
        [
            {
                "id": "link-one",
                "save_id": "target-save",
                "entity_type": "character",
                "entity_id": "character-one",
                "target_type": "memory",
                "target_id": "merged-memory",
                "relation": "recalls",
                "source_message_id": None,
            },
            {
                "id": "link-two",
                "save_id": "target-save",
                "entity_type": "character",
                "entity_id": "character-one",
                "target_type": "memory",
                "target_id": "merged-memory",
                "relation": "recalls",
                "source_message_id": "message-two",
            },
        ]
    )
    edges = _coalesce_import_knowledge_edges(
        [
            {
                "id": "edge-knows",
                "save_id": "target-save",
                "character_id": "character-one",
                "target_type": "memory",
                "target_id": "merged-memory",
                "knowledge_state": "knows",
                "confidence": 0.9,
                "source_message_ids_json": '["message-one"]',
                "archived_at": None,
            },
            {
                "id": "edge-denial",
                "save_id": "target-save",
                "character_id": "character-one",
                "target_type": "memory",
                "target_id": "merged-memory",
                "knowledge_state": "does_not_know",
                "confidence": 0.7,
                "source_message_ids_json": '["message-two"]',
                "archived_at": None,
            },
        ]
    )

    assert sources == [active_source]
    assert len(links) == 1
    assert links[0]["source_message_id"] == "message-two"
    assert len(edges) == 1
    assert edges[0]["knowledge_state"] == "does_not_know"
    assert edges[0]["confidence"] == 0.9
    assert json.loads(cast(str, edges[0]["source_message_ids_json"])) == [
        "message-one",
        "message-two",
    ]


def test_bundle_validation_rejects_table_row_bomb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_bundle_module, "_MAX_BUNDLE_TABLE_ROWS", 1)

    with pytest.raises(
        chat_bundle_module.ChatBundleError,
        match="table has too many rows",
    ):
        chat_bundle_module._validate_bundle_data(
            {},
            {"message_action_choices": [{}, {}]},
        )


def test_bundle_json_decode_stops_at_object_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_bundle_module, "_MAX_BUNDLE_JSON_OBJECTS", 1)

    with pytest.raises(
        chat_bundle_module.ChatBundleError,
        match="contains too many objects",
    ):
        chat_bundle_module._json_object_from_bytes(
            b'{"rows":[{},{}]}',
            "data.json",
        )


def test_bundle_json_decode_stops_at_primitive_value_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_bundle_module, "_MAX_BUNDLE_JSON_NODES", 4)

    with pytest.raises(
        chat_bundle_module.ChatBundleError,
        match="too many values",
    ):
        chat_bundle_module._json_object_from_bytes(
            b'{"rows":[0,0,0,0]}',
            "data.json",
        )


def test_bundle_validation_stops_at_nested_json_value_budget() -> None:
    with pytest.raises(
        chat_bundle_module.ChatBundleError,
        match="nested JSON",
    ):
        chat_bundle_module._validate_bundle_nested_json(
            {
                "state_changes": [
                    {"before_json": json.dumps([0] * 20)}
                ]
            },
            json_node_budget=[10],
        )


def test_import_context_sources_keep_legacy_provenance_alternatives() -> None:
    [source] = _coalesce_import_context_sources(
        [
            {
                "id": "source-one",
                "save_id": "target-save",
                "source_type": "memory",
                "source_id": "merged-memory",
                "metadata_json": '{"source_message_ids":["message-hidden"]}',
                "token_estimate": 3,
                "archived_at": None,
            },
            {
                "id": "source-two",
                "save_id": "target-save",
                "source_type": "memory",
                "source_id": "merged-memory",
                "metadata_json": '{"source_message_ids":["message-visible"]}',
                "token_estimate": 4,
                "archived_at": None,
            },
        ]
    )

    metadata = json.loads(cast(str, source["metadata_json"]))
    assert metadata["source_provenance_groups"] == [
        ["message-hidden"],
        ["message-visible"],
    ]


def test_import_context_sources_reject_conflicting_body_provenance() -> None:
    with pytest.raises(
        chat_bundle_module.ChatBundleError,
        match="Conflicting context sources",
    ):
        _coalesce_import_context_sources(
            [
                {
                    "id": "source-hidden",
                    "save_id": "target-save",
                    "source_type": "memory",
                    "source_id": "merged-memory",
                    "title": "Secret",
                    "body": "The hidden vault code is AMBER-77.",
                    "metadata_json": (
                        '{"source_message_ids":["message-hidden"]}'
                    ),
                    "archived_at": None,
                },
                {
                    "id": "source-visible",
                    "save_id": "target-save",
                    "source_type": "memory",
                    "source_id": "merged-memory",
                    "title": "Harmless",
                    "body": "The lamps are lit.",
                    "metadata_json": (
                        '{"source_message_ids":["message-visible"]}'
                    ),
                    "archived_at": None,
                },
            ]
        )


def test_import_proactive_triggers_coalesce_and_remap_schema_keys() -> None:
    mappings = {
        "message": {"message-old": "message-new"},
        "character": {"character-old": "character-new"},
        "memory": {
            "memory-one": "memory-merged",
            "memory-two": "memory-merged",
        },
    }
    assert _remapped_character_text_trigger_key(
        "ambient_random:message-old:character-old",
        mappings,
    ) == "ambient_random:message-new:character-new"
    assert _remapped_character_text_trigger_key(
        "character_intent:memory-one:basis",
        mappings,
    ) == "character_intent:memory-one:basis"

    rows = _coalesce_import_proactive_triggers(
        [
            {
                "id": "trigger-one",
                "save_id": "target-save",
                "character_id": "character-new",
                "trigger_key": "memory:memory-merged",
                "trigger_type": "memory_changed",
                "source_type": "memory",
                "source_id": "memory-merged",
                "reason": "Original reason",
            },
            {
                "id": "trigger-two",
                "save_id": "target-save",
                "character_id": "character-new",
                "trigger_key": "memory:memory-merged",
                "trigger_type": "memory_changed",
                "source_type": "memory",
                "source_id": "memory-merged",
                "reason": "Replacement reason",
            },
        ]
    )

    assert len(rows) == 1
    assert rows[0]["reason"] == "Replacement reason"


def test_import_knowledge_edges_fail_closed_on_provenance_overflow() -> None:
    [edge] = _coalesce_import_knowledge_edges(
        [
            {
                "id": "edge-one",
                "save_id": "target-save",
                "character_id": "character-one",
                "target_type": "memory",
                "target_id": "memory-one",
                "knowledge_state": "knows",
                "acquisition_method": "observed",
                "confidence": 0.8,
                "source_message_ids_json": json.dumps(
                    [f"message-{index:02d}" for index in range(40)]
                ),
                "archived_at": None,
            },
            {
                "id": "edge-two",
                "save_id": "target-save",
                "character_id": "character-one",
                "target_type": "memory",
                "target_id": "memory-one",
                "knowledge_state": "knows",
                "acquisition_method": "told",
                "confidence": 0.9,
                "source_message_ids_json": json.dumps(
                    [f"message-{index:02d}" for index in range(40, 80)]
                ),
                "archived_at": None,
            },
        ]
    )

    assert edge["knowledge_state"] == "does_not_know"
    assert edge["acquisition_method"] == "unknown"
    assert edge["source_message_ids_json"] == "[]"


def test_import_knowledge_edges_coalesce_target_aliases_and_scalar_provenance() -> None:
    edges = _coalesce_import_knowledge_edges(
        [
            {
                "id": "edge-knows",
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
                "id": "edge-denial",
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
        ]
    )

    assert len(edges) == 1
    assert edges[0]["target_type"] == "world_state"
    assert edges[0]["knowledge_state"] == "does_not_know"
    assert json.loads(cast(str, edges[0]["source_message_ids_json"])) == [
        "message-visible",
        "message-hidden",
    ]


def test_chat_bundle_round_trips_storyteller_mode_and_defaults_legacy_mode(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(
        repositories,
        media_dir,
        interaction_mode=InteractionMode.STORYTELLER,
    )
    bundle_path = tmp_path / "storyteller.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))
    assert data["scenario"]["interaction_mode"] == "storyteller"
    assert data["save"]["interaction_mode"] == "storyteller"
    imported = service.import_save(bundle_path)
    imported_save = repositories.get_save(_imported_save_id(imported))
    assert imported_save is not None
    assert imported_save.interaction_mode is InteractionMode.STORYTELLER
    source_pressure = repositories.get_summary_pressure_state(save.id)
    imported_pressure = repositories.get_summary_pressure_state(imported_save.id)
    assert imported_pressure.unsummarized_message_count == (
        source_pressure.unsummarized_message_count
    )
    assert imported_pressure.unsummarized_token_estimate == (
        source_pressure.unsummarized_token_estimate
    )
    assert imported_pressure.active_summary_count == (
        source_pressure.active_summary_count
    )

    del data["scenario"]["interaction_mode"]
    del data["save"]["interaction_mode"]
    legacy_path = tmp_path / "legacy-roleplay.bragi-chat"
    with zipfile.ZipFile(bundle_path) as bundle:
        members = {
            name: bundle.read(name)
            for name in bundle.namelist()
            if name not in {"manifest.json", "data.json"}
        }
    with zipfile.ZipFile(
        legacy_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        bundle.writestr("data.json", json.dumps(data))
        for name, body in members.items():
            bundle.writestr(name, body)
    legacy_import = service.import_save(legacy_path)
    legacy_save = repositories.get_save(_imported_save_id(legacy_import))
    assert legacy_save is not None
    assert legacy_save.interaction_mode is InteractionMode.ROLEPLAY


def test_export_save_writes_manifest_data_and_referenced_media(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    _replace_seed_scenario_update_content(
        repositories,
        _legacy_character_list_update_content(),
    )
    bundle_path = tmp_path / "exports" / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    manifest = service.export_save(save.id, bundle_path)

    assert manifest.bundle_version == 1
    assert manifest.title == "Night Watch"
    assert manifest.scenario_title == "Ashfall Keep"
    assert manifest.message_count == 2
    assert manifest.media_count == 1

    with zipfile.ZipFile(bundle_path) as bundle:
        names = set(bundle.namelist())
        assert {"manifest.json", "data.json"}.issubset(names)
        media_names = [name for name in names if name.startswith("media/")]
        assert len(media_names) == 1
        assert bundle.read(media_names[0]) == MEDIA_BYTES

        manifest_payload = json.loads(bundle.read("manifest.json"))
        assert manifest_payload["bundle_version"] == 1
        assert _manifest_save_title(manifest_payload) == "Night Watch"
        assert manifest_payload["scenario_title"] == "Ashfall Keep"
        assert manifest_payload["message_count"] == 2
        assert manifest_payload["media_count"] == 1

        data = json.loads(bundle.read("data.json"))
        assert "provider_configs" not in data
        assert "model_preferences" not in data
        assert "sync_save_states" not in data
        assert "jobs" not in data
        assert data["save"]["id"] == SAVE_ID
        assert data["save"]["title"] == "Night Watch"
        assert data["save"]["custom_instructions"] == (
            "Keep choices brief and grounded."
        )
        assert "owner_user_id" not in data["save"]
        assert "save_assignments" not in data
        assert data["scenario"]["id"] == SCENARIO_ID
        assert data["scenario"]["title"] == "Ashfall Keep"
        scenario_updates = data["save_scenario_updates"]
        assert isinstance(scenario_updates, list)
        [scenario_update] = scenario_updates
        assert isinstance(scenario_update, dict)
        assert json.loads(cast(str, scenario_update["content_json"])) == (
            _cleaned_character_list_update_content()
        )
        assert [message["id"] for message in data["messages"]] == [
            PLAYER_MESSAGE_ID,
            NARRATOR_MESSAGE_ID,
        ]
        assert data["save_app_settings"] == [
            {
                "scope": "save",
                "key": IMAGE_STYLE_PRESET_SETTING,
                "value_json": '"watercolor"',
            },
            {
                "scope": "save",
                "key": SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
                "value_json": "3",
            },
            {
                "scope": "scenario",
                "key": SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
                "value_json": "5",
            },
        ]
        assert data["memories"][0]["source_message_ids_json"] == json.dumps(
            [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
            sort_keys=True,
            separators=(",", ":"),
        )
        assert data["context_observations"][0]["id"] == OBSERVATION_ID
        assert data["context_observations"][0]["source_message_ids_json"] == (
            json.dumps(
                [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        [curation_state] = data["context_observation_curation_states"]
        assert curation_state["observation_id"] == OBSERVATION_ID
        assert curation_state["terminal_outcome"] == "accepted"
        assert curation_state["lease_token"] is None
        assert curation_state["lease_until"] is None


def test_export_import_preserves_pending_retry_budgets(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.set_app_setting("retry_count", 2)
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="open_thread",
        claim="The red lens warning is still unresolved.",
        scope="save",
        status="pending",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="field_update",
        entity_type="save",
        field_path="custom_instructions",
        proposed_value="Keep the warning unresolved.",
        status="pending",
    )
    bundle_path = tmp_path / "exports" / "retry-budget.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    repositories.set_app_setting("retry_count", 0)
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_observation = next(
        item
        for item in repositories.list_context_observations(imported_save_id)
        if item.claim == observation.claim
    )
    imported_state = repositories.get_context_observation_curation_state(
        imported_observation.id
    )
    assert imported_state is not None
    assert imported_state.max_attempts == 3

    imported_suggestion = next(
        item
        for item in repositories.list_context_update_suggestions(imported_save_id)
        if item.field_path == suggestion.field_path
    )
    assert imported_suggestion.max_retry_count == 2


def test_export_import_preserves_incomplete_post_turn_outbox(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.ensure_post_turn_outbox_steps(
        save_id=save.id,
        player_message_id=PLAYER_MESSAGE_ID,
        narrator_message_id=NARRATOR_MESSAGE_ID,
        turn_revision="revision-1",
        steps=("context",),
        payload={
            "verified_plan_coverage": {
                "source_message_ids": [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID]
            }
        },
    )
    bundle_path = tmp_path / "exports" / "pending-continuity.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_rows = repositories.list_post_turn_outbox_steps(
        save_id=_imported_save_id(imported)
    )

    assert len(imported_rows) == 1
    assert imported_rows[0].step == "context"
    assert imported_rows[0].status == "pending"
    assert imported_rows[0].player_message_id != PLAYER_MESSAGE_ID
    assert imported_rows[0].narrator_message_id != NARRATOR_MESSAGE_ID
    imported_coverage = imported_rows[0].payload["verified_plan_coverage"]
    assert isinstance(imported_coverage, dict)
    assert imported_coverage["source_message_ids"] == [
        imported_rows[0].player_message_id,
        imported_rows[0].narrator_message_id,
    ]


def test_export_rejects_snapshot_that_cannot_be_imported(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "exports" / "oversized.bragi-chat"

    def reject_snapshot_rows(**_kwargs: object) -> None:
        raise ValueError("Snapshot manifest contains too many entries")

    monkeypatch.setattr(
        TurnSnapshotService,
        "validate_exported_snapshot_rows",
        staticmethod(reject_snapshot_rows),
    )

    with pytest.raises(
        chat_bundle_module.ChatBundleError,
        match="too many entries",
    ):
        _chat_bundle_service(repositories, media_dir).export_save(
            save.id,
            bundle_path,
        )

    assert not bundle_path.exists()


def test_import_preserves_exported_legacy_normalized_budget_allowance(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
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
    repositories.ensure_context_source_legacy_budget_limit(
        save_id=save.id,
        normalized_text_bytes=normalized_text_bytes,
    )
    repositories.commit()
    bundle_path = tmp_path / "exports" / "legacy-budget.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)

    imported_save_id = _imported_save_id(imported)
    imported_limit = repositories.connection.execute(
        """
        SELECT normalized_text_bytes
        FROM context_source_legacy_budget_limits
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()[0]
    assert imported_limit == normalized_text_bytes


def test_import_preserves_snapshot_legacy_normalized_budget_allowance(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="custom_note",
        source_id="snapshot-legacy-budget-note",
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
    repositories.ensure_context_source_legacy_budget_limit(
        save_id=save.id,
        normalized_text_bytes=normalized_text_bytes,
    )
    repositories.commit()
    TurnSnapshotService(repositories).capture_baseline_snapshot(save.id)
    repositories.archive_context_source(source.id)
    repositories.connection.execute(
        """
        DELETE FROM context_source_legacy_budget_limits
        WHERE save_id = ?
        """,
        (save.id,),
    )
    repositories.commit()
    bundle_path = tmp_path / "exports" / "snapshot-legacy-budget.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)

    imported_save_id = _imported_save_id(imported)
    imported_limit = repositories.connection.execute(
        """
        SELECT normalized_text_bytes
        FROM context_source_legacy_budget_limits
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()[0]
    snapshot_id = repositories.connection.execute(
        """
        SELECT id
        FROM save_turn_snapshots
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()[0]
    TurnSnapshotService(repositories).restore_save_to_snapshot(
        save_id=imported_save_id,
        snapshot_id=snapshot_id,
    )
    assert imported_limit == normalized_text_bytes
    assert [
        restored.source_id
        for restored in repositories.list_context_sources(imported_save_id)
    ] == ["snapshot-legacy-budget-note"]


def test_native_export_storage_satisfies_import_compression_guard(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _chat_bundle_module()
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "exports" / "import-safe-storage.bragi-chat"
    monkeypatch.setattr(module, "_MIN_BUNDLE_COMPRESSION_RATIO_CHECK_BYTES", 1)
    monkeypatch.setattr(module, "_MAX_BUNDLE_COMPRESSION_RATIO", 1.0)
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        assert all(
            info.compress_type == zipfile.ZIP_STORED
            for info in bundle.infolist()
        )
    service.import_save(bundle_path)


def test_import_rejects_uncovered_snapshot_media_object(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    TurnSnapshotService(repositories).capture_baseline_snapshot(save.id)
    exported_path = tmp_path / "covered-snapshot-media.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, exported_path)
    with zipfile.ZipFile(exported_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))
    _move_media_asset_to_snapshot_only(
        manifest,
        data,
        media_asset_id=MEDIA_ASSET_ID,
    )
    data["snapshot_media_assets"] = []
    bundle_path = tmp_path / "uncovered-snapshot-media.bragi-chat"
    _write_bundle(bundle_path, manifest=manifest, data=data)

    with pytest.raises(
        chat_bundle_module.ChatBundleError,
        match="do not cover",
    ):
        service.import_save(bundle_path)


def test_export_import_preserves_exact_identifier_after_long_token(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="custom_note",
        source_id="long-token-tail",
        title="Archive codes",
        body="KEEP-1 " + ("A" * 70_000) + " TAIL-2",
    )
    bundle_path = tmp_path / "long-token-tail.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)

    imported_save_id = _imported_save_id(imported)
    identifiers = {
        row[0]
        for row in repositories.connection.execute(
            """
            SELECT identifier
            FROM context_source_exact_identifiers
            WHERE save_id = ?
            """,
            (imported_save_id,),
        )
    }
    assert {"keep-1", "tail-2"} <= identifiers


def test_export_save_refunds_live_curation_attempt_when_clearing_lease(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="event",
        claim="The beacon was relit.",
        evidence_quote="The beacon was relit.",
        source_message_ids=[NARRATOR_MESSAGE_ID],
        scope="durable",
        confidence=0.9,
    )
    claimed = repositories.claim_context_observations(
        (observation.id,),
        lease_token="export-worker-secret",
        lease_seconds=600,
    )
    assert [row.id for row in claimed] == [observation.id]
    bundle_path = tmp_path / "exports" / "night-watch.bragi-chat"

    _chat_bundle_service(repositories, media_dir).export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    [state] = [
        row
        for row in data["context_observation_curation_states"]
        if row["observation_id"] == observation.id
    ]
    assert state["attempt_count"] == 0
    assert state["lease_token"] is None
    assert state["lease_until"] is None


def test_export_save_uses_consistent_read_snapshot(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    database_path = _repository_database_path(repositories)
    repositories.connection.execute("PRAGMA journal_mode=WAL")
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    writer_connection = sqlite3.connect(database_path, timeout=5)
    writer = PersistenceRepositories(writer_connection)
    inserted_concurrent_thread = False

    def insert_concurrent_thread() -> None:
        nonlocal inserted_concurrent_thread
        if inserted_concurrent_thread:
            return
        inserted_concurrent_thread = True
        writer.add_active_thread(
            thread_id="thread-concurrent-write",
            save_id=save.id,
            title="Concurrent warning",
            description="This thread was committed after export started.",
            priority=50,
        )

    def trace_export_query(statement: str) -> None:
        if "FROM active_threads" in statement:
            insert_concurrent_thread()

    bundle_path = tmp_path / "exports" / "night-watch-snapshot.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    repositories.connection.set_trace_callback(trace_export_query)
    try:
        service.export_save(save.id, bundle_path)
    finally:
        repositories.connection.set_trace_callback(None)
        writer_connection.close()

    assert inserted_concurrent_thread
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert isinstance(data, dict)
    active_threads = data["active_threads"]
    assert isinstance(active_threads, list)
    assert all(
        not isinstance(row, dict) or row.get("id") != "thread-concurrent-write"
        for row in active_threads
    )

    persisted_thread = repositories.get_active_thread("thread-concurrent-write")
    assert persisted_thread is not None


def test_export_import_preserves_message_safety_transition(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _seed_bundle_save(repositories, tmp_path / "media")
    repositories.connection.execute(
        "UPDATE messages SET safety_transition = ?, content_rating = ? WHERE id = ?",
        ("fade_to_black", "r", NARRATOR_MESSAGE_ID),
    )
    repositories.commit()
    service = _chat_bundle_service(repositories, tmp_path / "media")
    bundle_path = tmp_path / "exports" / "safety.bragi-chat"

    service.export_save(save.id, bundle_path)

    def corrupt_narrator_body(data: dict[str, object]) -> None:
        messages = data["messages"]
        assert isinstance(messages, list)
        narrator = messages[1]
        assert isinstance(narrator, dict)
        narrator["body"] = "Rejected narrator draft."

    _rewrite_bundle_data(bundle_path, corrupt_narrator_body)
    imported = service.import_save(bundle_path)
    imported_messages = repositories.list_messages(_imported_save_id(imported))

    assert imported_messages[1].body == (
        "The intimate moment is kept off-screen. Hours later, "
        "the next scene begins."
    )
    assert imported_messages[1].safety_transition == "fade_to_black"
    assert imported_messages[1].content_rating == "unclassified"


def test_export_import_preserves_unrated_narration_without_resanitizing(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    save = _seed_bundle_save(repositories, tmp_path / "media")
    adult_body = "They had sex after returning to the inn."
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=adult_body,
    )
    service = _chat_bundle_service(repositories, tmp_path / "media")
    bundle_path = tmp_path / "exports" / "unrated.bragi-chat"

    service.export_save(save.id, bundle_path)
    imported = service.import_save(bundle_path)
    imported_messages = repositories.list_messages(_imported_save_id(imported))

    assert imported_messages[-1].body == adult_body
    assert imported_messages[-1].safety_transition == ""


def test_export_save_includes_save_model_overrides_without_global_preferences(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/server-chat",
    )
    override_value = {
        "preferences": {
            "chat": {
                "provider": "venice",
                "model_id": "venice/save-chat",
            }
        }
    }
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_MODEL_OVERRIDES_SETTING,
        value=override_value,
    )
    bundle_path = tmp_path / "exports" / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert "model_preferences" not in data
    assert _save_app_setting_value(
        data,
        scope="save",
        key=SAVE_MODEL_OVERRIDES_SETTING,
    ) == json.dumps(override_value, sort_keys=True, separators=(",", ":"))


def test_import_save_sanitizes_retired_routing_tasks_in_all_scoped_settings(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    retired_preference = {
        "provider": "openrouter",
        "model_id": "openai/gpt-5-mini",
    }
    retained_preference = {
        "provider": "venice",
        "model_id": "venice/dating-chat",
    }
    repositories.set_scoped_setting(
        scope="scenario",
        scope_id=save.scenario_id,
        key=SAVE_MODEL_OVERRIDES_SETTING,
        value={
            "preferences": {
                "chat_character_interaction": retired_preference,
                "chat_dating_sim": retained_preference,
            }
        },
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=MODEL_THINKING_PREFERENCES_SETTING,
        value={
            "character_interaction_context_update": {
                **retired_preference,
                "level": "high",
            },
            "dating_sim_context_update": {
                **retained_preference,
                "level": "low",
            },
        },
    )
    bundle_path = tmp_path / "exports" / "retired-routing.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save = repositories.get_save(_imported_save_id(imported))
    assert imported_save is not None
    assert repositories.get_scoped_setting(
        scope="scenario",
        scope_id=imported_save.scenario_id,
        key=SAVE_MODEL_OVERRIDES_SETTING,
    ) == {"preferences": {"chat_dating_sim": retained_preference}}
    assert repositories.get_scoped_setting(
        scope="save",
        scope_id=imported_save.id,
        key=MODEL_THINKING_PREFERENCES_SETTING,
    ) == {
        "dating_sim_context_update": {
            **retained_preference,
            "level": "low",
        }
    }


def test_export_and_import_save_remaps_message_action_choices(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.replace_message_action_choices(
        save_id=save.id,
        message_id=NARRATOR_MESSAGE_ID,
        choices=(
            "Study the red lens.",
            "Signal the eastern tower.",
            "Wake the sleeping riders.",
            "Bar the ash gate.",
        ),
        provider="fake",
        model="choice-model",
    )
    deleted_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="This branch should not export.",
    )
    repositories.replace_message_action_choices(
        save_id=save.id,
        message_id=deleted_narrator.id,
        choices=("Deleted A", "Deleted B", "Deleted C", "Deleted D"),
        provider="fake",
        model="choice-model",
    )
    repositories.archive_messages_from(save_id=save.id, message_id=deleted_narrator.id)
    bundle_path = tmp_path / "exports" / "night-watch-choices.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert [row["body"] for row in data["message_action_choices"]] == [
        "Study the red lens.",
        "Signal the eastern tower.",
        "Wake the sleeping riders.",
        "Bar the ash gate.",
    ]
    assert {row["message_id"] for row in data["message_action_choices"]} == {
        NARRATOR_MESSAGE_ID
    }

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_choices = repositories.latest_message_action_choices(imported_save_id)

    assert [choice.body for choice in imported_choices] == [
        "Study the red lens.",
        "Signal the eastern tower.",
        "Wake the sleeping riders.",
        "Bar the ash gate.",
    ]
    assert {choice.message_id for choice in imported_choices} != {NARRATOR_MESSAGE_ID}


def test_export_and_import_save_remaps_director_pressure_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_world_state(
        state_id="world-state-director-pressure",
        save_id=save.id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={
            "dramatic_questions": ["Will Mara warn the lower village?"],
            "tension_level": 3,
            "tension_trend": "stalled",
            "stall_turns": 0,
            "cooldown_turns": 2,
            "active_clocks": [
                {
                    "title": "Guard search",
                    "status": "active",
                    "segments_total": 4,
                    "segments_filled": 1,
                }
            ],
            "escalation_history": [
                {
                    "kind": "external_complication",
                    "directive": "Raise stakes: guards search this floor.",
                    "source_message_id": NARRATOR_MESSAGE_ID,
                }
            ],
        },
        category="director_pressure",
        confidence=1.0,
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    repositories.add_active_thread(
        thread_id="thread-director-pressure",
        save_id=save.id,
        title="Guards search the tower floor",
        description="The guard sweep is moving toward the beacon.",
        priority=3,
        visibility="scene",
        related_entities=["director_pressure"],
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    bundle_path = tmp_path / "exports" / "night-watch-director.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_messages = repositories.list_messages(imported_save_id)
    message_id_map = {
        PLAYER_MESSAGE_ID: imported_messages[0].id,
        NARRATOR_MESSAGE_ID: imported_messages[1].id,
    }

    director_state = next(
        state
        for state in repositories.list_world_state(imported_save_id)
        if state.key == DIRECTOR_PRESSURE_STATE_KEY
    )
    assert director_state.source_message_id == message_id_map[NARRATOR_MESSAGE_ID]
    assert director_state.value["escalation_history"] == [
        {
            "kind": "external_complication",
            "directive": "Raise stakes: guards search this floor.",
            "source_message_id": message_id_map[NARRATOR_MESSAGE_ID],
        }
    ]
    director_thread = next(
        thread
        for thread in repositories.list_active_threads(imported_save_id)
        if thread.title == "Guards search the tower floor"
    )
    assert director_thread.source_message_id == message_id_map[NARRATOR_MESSAGE_ID]
    assert director_thread.related_entities == ["director_pressure"]


def test_export_import_preserves_political_intrigue_world_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        scenario_id="scenario-council-of-ash",
        type="political_intrigue",
        title="Council of Ash",
        premise="A city council vote will decide who controls the harbor.",
        player_role="Envoy holding the swing vote",
        content={
            "title": "Council of Ash",
            "premise": "A city council vote will decide who controls the harbor.",
            "player_character_name": "Mara Voss",
            "player_role": "Envoy holding the swing vote",
            "political_arena": "The harbor council chamber and public galleries.",
            "political_factions": "Guilds, Old Families, and dock unions.",
            "central_conflict": "A midnight no-confidence vote can replace the regent.",
            "secrets_and_leverage": "Only Mara knows Orro moved missing silver.",
            "reputation_and_standing": "Mara is trusted by reformers.",
            "obligations_and_favors": "Orro owes Mara one public endorsement.",
            "alliances_and_rivalries": "Reformers court Mara; old houses resist.",
            "event_calendar": "Dawn hearing; noon procession; midnight vote.",
            "political_pressure": "The midnight vote proceeds unless delayed.",
            "public_private_knowledge": (
                "The public knows the vote is close; only Mara knows the favor."
            ),
            "tone_genre": "Tense council intrigue.",
            "opening_message": "The council bell rings.",
        },
    )
    save = repositories.create_save(
        save_id="save-council-of-ash",
        scenario_id=scenario.id,
        title="Council of Ash",
    )
    message = repositories.append_message(
        message_id="message-council-opening",
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The council bell rings.",
    )
    repositories.upsert_world_state(
        state_id="world-state-intrigue-obligations",
        save_id=save.id,
        key="intrigue.obligations",
        value={"summary": "Orro owes Mara one public endorsement."},
        category="obligation",
        confidence=1.0,
        source_message_id=message.id,
    )
    repositories.upsert_world_state(
        state_id="world-state-intrigue-standing",
        save_id=save.id,
        key="intrigue.standing",
        value={"summary": "Mara is trusted by reformers."},
        category="reputation",
        confidence=1.0,
        source_message_id=message.id,
    )
    bundle_path = tmp_path / "exports" / "council-of-ash.bragi-chat"
    service = _chat_bundle_service(repositories, tmp_path / "media")

    service.export_save(save.id, bundle_path)
    imported = service.import_save(bundle_path)

    imported_save_id = _imported_save_id(imported)
    loaded = repositories.load_save_details(imported_save_id)
    assert loaded is not None
    assert loaded.scenario.type == "political_intrigue"
    imported_content = json.loads(loaded.scenario.content_json)
    assert imported_content["obligations_and_favors"] == (
        "Orro owes Mara one public endorsement."
    )
    imported_message_id = loaded.messages[0].id
    assert imported_message_id != message.id
    state_by_key = {
        state.key: state for state in repositories.list_world_state(imported_save_id)
    }
    assert state_by_key["intrigue.obligations"].value == {
        "summary": "Orro owes Mara one public endorsement."
    }
    assert state_by_key["intrigue.obligations"].category == "obligation"
    assert state_by_key["intrigue.obligations"].source_message_id == (
        imported_message_id
    )
    assert state_by_key["intrigue.standing"].value == {
        "summary": "Mara is trusted by reformers."
    }
    assert state_by_key["intrigue.standing"].category == "reputation"


def test_export_import_preserves_first_contact_world_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        scenario_id="scenario-songs-under-europa",
        type="first_contact_exploration",
        title="Songs Under Europa",
        premise="A survey crew finds patterned signals beneath the ice.",
        player_role="Mission linguist",
        content={
            "title": "Songs Under Europa",
            "premise": "A survey crew finds patterned signals beneath the ice.",
            "player_character_name": "Dr. Mara Voss",
            "player_role": "Mission linguist",
            "mission_profile": "Survey the hidden ocean.",
            "ship_or_base_status": "Habitat heat stable for 42 hours.",
            "exploration_target": "A black-water cavern beneath the ice.",
            "unknown_intelligence": "An unseen singer answers sonar.",
            "knowledge_state": "Observed songs; unknown intent.",
            "translation_progress": "Three descending pulses may mean open water.",
            "discoveries_and_samples": "Metallic spores remain quarantined.",
            "hazards_and_escalation": "Thermal fissures are spreading.",
            "tone_genre": "Hopeful first-contact science fiction.",
            "opening_message": "Blue light pulses beneath the ice.",
        },
    )
    save = repositories.create_save(
        save_id="save-songs-under-europa",
        scenario_id=scenario.id,
        title="Songs Under Europa",
    )
    message = repositories.append_message(
        message_id="message-contact-opening",
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Blue light pulses beneath the ice.",
    )
    repositories.upsert_world_state(
        state_id="world-state-contact-translation",
        save_id=save.id,
        key="contact.translation",
        value={"summary": "Three descending pulses may mean open water."},
        category="translation",
        confidence=1.0,
        source_message_id=message.id,
    )
    repositories.upsert_world_state(
        state_id="world-state-contact-hazards",
        save_id=save.id,
        key="contact.hazards",
        value={"summary": "Thermal fissures are spreading."},
        category="threat",
        confidence=1.0,
        source_message_id=message.id,
    )
    bundle_path = tmp_path / "exports" / "songs-under-europa.bragi-chat"
    service = _chat_bundle_service(repositories, tmp_path / "media")

    service.export_save(save.id, bundle_path)
    imported = service.import_save(bundle_path)

    imported_save_id = _imported_save_id(imported)
    loaded = repositories.load_save_details(imported_save_id)
    assert loaded is not None
    assert loaded.scenario.type == "first_contact_exploration"
    imported_message_id = loaded.messages[0].id
    assert imported_message_id != message.id
    state_by_key = {
        state.key: state for state in repositories.list_world_state(imported_save_id)
    }
    assert state_by_key["contact.translation"].value == {
        "summary": "Three descending pulses may mean open water."
    }
    assert state_by_key["contact.translation"].category == "translation"
    assert state_by_key["contact.translation"].source_message_id == (
        imported_message_id
    )
    assert state_by_key["contact.hazards"].value == {
        "summary": "Thermal fissures are spreading."
    }
    assert state_by_key["contact.hazards"].category == "threat"


def test_import_save_repairs_director_pressure_history_outside_bundle(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_world_state(
        state_id="world-state-director-pressure",
        save_id=save.id,
        key=DIRECTOR_PRESSURE_STATE_KEY,
        value={
            "dramatic_questions": ["Will Mara warn the lower village?"],
            "tension_level": 3,
            "tension_trend": "stalled",
            "stall_turns": 0,
            "cooldown_turns": 2,
            "active_clocks": [],
            "escalation_history": [
                {
                    "kind": "external_complication",
                    "directive": "Raise stakes: guards search this floor.",
                    "source_message_id": NARRATOR_MESSAGE_ID,
                }
            ],
        },
        category="director_pressure",
        confidence=1.0,
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    bundle_path = tmp_path / "exports" / "night-watch-director-stale.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)
    _rewrite_bundle_data(
        bundle_path,
        lambda data: _replace_director_pressure_history_source(
            data,
            source_message_id="foreign-message-id",
        ),
    )

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    director_state = next(
        state
        for state in repositories.list_world_state(imported_save_id)
        if state.key == DIRECTOR_PRESSURE_STATE_KEY
    )
    history = director_state.value["escalation_history"]
    assert isinstance(history, list)
    entry = history[0]
    assert isinstance(entry, dict)
    assert entry["source_message_id"] is None


def test_export_save_includes_complete_active_save_graph(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    _seed_context_graph_rows(repositories, save.id)
    bundle_path = tmp_path / "exports" / "night-watch-complete.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))

    assert manifest["bragi_schema_version"] >= 1
    assert [row["id"] for row in data["context_sources"]] == [
        "ctx-location",
        "ctx-media",
        "ctx-message",
    ]
    assert [row["id"] for row in data["scene_snapshots"]] == ["scene-main"]
    assert data["scene_snapshots"][0]["in_world_time"] == "Monday night"
    assert data["scene_snapshots"][0]["time_of_day"] == "night"
    assert data["scene_snapshots"][0]["day_of_week"] == "monday"
    assert data["scene_snapshots"][0]["world_day_index"] == 2
    assert data["scene_snapshots"][0]["world_time_day_index"] == 2
    assert data["scene_snapshots"][0]["world_time_day_label"] == "monday"
    assert data["scene_snapshots"][0]["world_time_phase"] == "night"
    assert data["scene_snapshots"][0]["world_time_clock_minutes"] is None
    assert data["scene_snapshots"][0]["world_time_period_label"] == ""
    assert data["scene_snapshots"][0]["world_time_source_message_id"] == (
        NARRATOR_MESSAGE_ID
    )
    assert data["scene_snapshots"][0]["world_time_confidence"] == 0.87
    assert [row["id"] for row in data["locations"]] == ["location-tower"]
    assert [row["id"] for row in data["characters"]] == ["character-mara"]
    assert data["characters"][0]["age"] == "late 30s"
    assert data["characters"][0]["current_clothing"] == (
        "Borrowed green raincoat over a linen shirt."
    )
    assert data["characters"][0]["is_player_character"] == 1
    assert data["characters"][0]["goals"] == "Keep the beacon lit."
    assert data["characters"][0]["cooperation_conditions"] == (
        "Helps after proof the lens can hold."
    )
    assert [row["id"] for row in data["active_threads"]] == ["thread-beacon"]
    assert [row["id"] for row in data["entity_links"]] == ["link-mara-tower"]
    assert data["entity_links"][0]["source_message_id"] == NARRATOR_MESSAGE_ID
    assert [row["id"] for row in data["character_knowledge_edges"]] == [
        "edge-mara-signal-code"
    ]
    assert data["character_knowledge_edges"][0]["source_message_ids_json"] == (
        json.dumps(
            [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    assert [row["id"] for row in data["message_visibility"]] == [
        "visibility-mara-narrator"
    ]
    assert [row["id"] for row in data["message_scene_presence"]] == [
        "presence-mara-narrator"
    ]
    assert [row["id"] for row in data["context_update_suggestions"]] == [
        "suggestion-mara-location"
    ]
    assert [row["id"] for row in data["context_update_audit"]] == [
        "audit-mara-location"
    ]
    assert "jobs" not in data


def test_import_world_time_backfill_preserves_missing_canonical_source(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Monday night",
        time_of_day="night",
        day_of_week="monday",
        world_day_index=2,
        world_time_day_index=2,
        world_time_day_label="monday",
        world_time_phase="night",
        world_time_source_message_id=None,
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    module = _chat_bundle_module()

    module._backfill_imported_scene_world_time(repositories.connection, save.id)

    row = repositories.connection.execute(
        """
        SELECT source_message_id, world_time_source_message_id
        FROM scene_snapshots
        WHERE save_id = ?
        """,
        (save.id,),
    ).fetchone()
    assert row is not None
    assert row["source_message_id"] == NARRATOR_MESSAGE_ID
    assert row["world_time_source_message_id"] is None


def test_import_world_time_backfill_preserves_legacy_text_without_canonical_fields(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Monday night",
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    repositories.connection.execute(
        """
        UPDATE scene_snapshots
        SET in_world_time = 'Friday night after the festival',
            time_of_day = '',
            day_of_week = '',
            world_day_index = NULL,
            world_time_day_index = NULL,
            world_time_day_label = '',
            world_time_phase = '',
            world_time_clock_minutes = NULL,
            world_time_period_label = '',
            world_time_source_message_id = NULL,
            world_time_confidence = NULL
        WHERE save_id = ?
        """,
        (save.id,),
    )
    module = _chat_bundle_module()

    module._backfill_imported_scene_world_time(repositories.connection, save.id)

    row = repositories.connection.execute(
        """
        SELECT in_world_time, time_of_day, day_of_week, world_day_index,
               world_time_phase
        FROM scene_snapshots
        WHERE save_id = ?
        """,
        (save.id,),
    ).fetchone()
    assert row is not None
    assert tuple(row) == (
        "Friday night after the festival",
        "",
        "",
        None,
        "night",
    )


def test_import_save_backfills_legacy_bundle_scene_world_time(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_scene_snapshot(
        save_id=original_save.id,
        in_world_time="Friday night after the festival",
        time_of_day="",
        day_of_week="",
        world_day_index=6,
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    bundle_path = tmp_path / "legacy-world-time.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(original_save.id, bundle_path)

    def remove_canonical_world_time(data: dict[str, object]) -> None:
        rows = data["scene_snapshots"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            for key in (
                "world_time_day_index",
                "world_time_day_label",
                "world_time_phase",
                "world_time_clock_minutes",
                "world_time_period_label",
                "world_time_source_message_id",
                "world_time_confidence",
            ):
                row.pop(key, None)

    _rewrite_bundle_data(bundle_path, remove_canonical_world_time)

    imported = service.import_save(bundle_path)
    snapshot = repositories.get_scene_snapshot(_imported_save_id(imported))

    assert snapshot is not None
    assert snapshot.in_world_time == "Friday night after the festival"
    assert snapshot.time_of_day == "night"
    assert snapshot.day_of_week == ""
    assert snapshot.world_day_index == 6
    assert snapshot.world_time_day_index == 6
    assert snapshot.world_time_phase == "night"
    assert snapshot.world_time_source_message_id == snapshot.source_message_id
    assert snapshot.world_time_source_message_id != NARRATOR_MESSAGE_ID


def test_import_save_preserves_current_bundle_legacy_world_time_detail(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_scene_snapshot(
        save_id=original_save.id,
        in_world_time="Friday evening",
        time_of_day="evening",
        day_of_week="friday",
        world_day_index=6,
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    repositories.connection.execute(
        """
        UPDATE scene_snapshots
        SET in_world_time = 'Friday evening after the festival'
        WHERE save_id = ?
        """,
        (original_save.id,),
    )
    repositories.connection.commit()
    bundle_path = tmp_path / "current-world-time.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(original_save.id, bundle_path)

    imported = service.import_save(bundle_path)
    snapshot = repositories.get_scene_snapshot(_imported_save_id(imported))

    assert snapshot is not None
    assert snapshot.in_world_time == "Friday evening after the festival"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "friday"
    assert snapshot.world_day_index == 6
    assert snapshot.world_time_day_index == 6
    assert snapshot.world_time_day_label == "friday"
    assert snapshot.world_time_phase == "evening"


def test_export_import_preserves_time_loop_clock_and_reset_baseline(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="time_loop",
        title="Bellwether Day",
        premise="A single day repeats.",
        player_role="Archivist",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Bell Loop")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Monday evening",
        time_of_day="evening",
        day_of_week="monday",
        world_day_index=0,
        world_time_clock_minutes=19 * 60,
        world_time_period_label="festival day",
    )
    loop_current = {
        "version": 1,
        "iteration": 3,
        "summary": "The bell has rung twice.",
        "current_time": {
            "day_index": 0,
            "day_label": "monday",
            "phase": "evening",
            "clock_minutes": 19 * 60,
            "period_label": "festival day",
        },
        "baseline_time": {
            "day_index": 0,
            "day_label": "monday",
            "phase": "morning",
            "clock_minutes": 8 * 60,
            "period_label": "festival day",
        },
        "resettable_baseline": {
            "loop.resettable.gate": {
                "value": {"open": False},
                "category": "loop_resettable",
                "confidence": 1.0,
            }
        },
    }
    repositories.upsert_world_state(
        save_id=save.id,
        key="loop.current",
        value=loop_current,
        category="loop_status",
    )
    bundle_path = tmp_path / "exports" / "bell-loop.bragi-chat"
    service = _chat_bundle_service(repositories, tmp_path / "media")

    service.export_save(save.id, bundle_path)
    imported = service.import_save(bundle_path)

    imported_save_id = _imported_save_id(imported)
    imported_snapshot = repositories.get_scene_snapshot(imported_save_id)
    state_by_key = {
        state.key: state for state in repositories.list_world_state(imported_save_id)
    }
    assert imported_snapshot is not None
    assert imported_snapshot.world_time_clock_minutes == 19 * 60
    assert imported_snapshot.world_time_period_label == "festival day"
    assert state_by_key["loop.current"].value == loop_current


def test_export_save_preserves_scene_snapshot_with_stale_world_time_source(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    deleted_message = repositories.append_message(
        message_id="message-deleted-world-time",
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="This world-time source should not remain portable.",
    )
    repositories.archive_messages_from(save_id=save.id, message_id=deleted_message.id)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Monday night",
        time_of_day="night",
        day_of_week="monday",
        world_day_index=2,
        world_time_source_message_id=deleted_message.id,
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    bundle_path = tmp_path / "exports" / "night-watch-stale-time.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert len(data["scene_snapshots"]) == 1
    scene = data["scene_snapshots"][0]
    assert scene["source_message_id"] == NARRATOR_MESSAGE_ID
    assert scene["world_time_source_message_id"] is None
    assert scene["world_time_phase"] == "night"


def test_export_save_drops_context_sources_with_stale_metadata_message_refs(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    deleted_message = repositories.append_message(
        message_id="message-deleted-context",
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="This context source should not remain portable.",
    )
    repositories.archive_messages_from(save_id=save.id, message_id=deleted_message.id)
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="world_state",
        source_id="portable-context",
        title="Portable context",
        body="This context source only references active messages.",
        metadata={
            "source_message_id": NARRATOR_MESSAGE_ID,
            "source_message_ids": [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
            "last_seen_message_id": NARRATOR_MESSAGE_ID,
        },
        context_source_id="ctx-portable",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="world_state",
        source_id="stale-metadata-context",
        title="Stale metadata context",
        body="This context source references a deleted message in metadata.",
        metadata={
            "source_message_ids": [deleted_message.id],
            "kept": "only if row survives",
        },
        context_source_id="ctx-stale-metadata",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="world_state",
        source_id="stale-provenance-context",
        title="Stale provenance context",
        body="This row has an inaccessible provenance alternative.",
        metadata={
            "source_message_ids": [NARRATOR_MESSAGE_ID],
            "source_provenance_groups": [
                [NARRATOR_MESSAGE_ID],
                [deleted_message.id],
            ],
            "source_provenance_mode": "all",
        },
        context_source_id="ctx-stale-provenance",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="message",
        source_id=f"{NARRATOR_MESSAGE_ID},{deleted_message.id}",
        title="Stale message context",
        body="This context source references a deleted message in source_id.",
        metadata={"source_message_id": NARRATOR_MESSAGE_ID},
        context_source_id="ctx-stale-source-id",
    )
    bundle_path = tmp_path / "exports" / "night-watch-context-metadata.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    context_source_ids = [row["id"] for row in data["context_sources"]]
    assert "ctx-portable" in context_source_ids
    assert "ctx-stale-metadata" not in context_source_ids
    assert "ctx-stale-provenance" not in context_source_ids
    assert "ctx-stale-source-id" not in context_source_ids


def test_export_save_clears_missing_location_parents(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    _seed_context_graph_rows(repositories, save.id)
    repositories.connection.execute(
        """
        INSERT INTO locations(
            id, save_id, name, aliases_json, description, visual_description,
            connections_json, status, hazards_json, source_message_id,
            locked_fields_json, archived_at
        )
        VALUES (?, ?, 'Old Kitchen', '[]', 'A quiet room.', '',
                '[]', 'archived', '[]', ?, '[]', CURRENT_TIMESTAMP)
        """,
        ("location-old-kitchen", save.id, NARRATOR_MESSAGE_ID),
    )
    repositories.connection.execute(
        """
        INSERT INTO locations(
            id, save_id, name, aliases_json, description, visual_description,
            parent_location_id, connections_json, status, hazards_json,
            source_message_id, locked_fields_json
        )
        VALUES (?, ?, 'Beacon Stair', '[]', 'A narrow ash-caked stair.',
                '', 'location-old-kitchen', '[]', 'stable', '[]', ?, '[]')
        """,
        ("location-stair", save.id, NARRATOR_MESSAGE_ID),
    )
    repositories.connection.commit()
    bundle_path = tmp_path / "exports" / "night-watch-missing-parent.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    location_rows = {
        row["id"]: row
        for row in data["locations"]
        if isinstance(row, dict)
    }
    assert "location-old-kitchen" not in location_rows
    assert location_rows["location-stair"]["parent_location_id"] is None


def test_import_save_remaps_complete_active_save_graph(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    _seed_context_graph_rows(repositories, save.id)
    bundle_path = tmp_path / "exports" / "night-watch-complete.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    assert imported_save_id != SAVE_ID
    location_context = repositories.connection.execute(
        """
        SELECT id, save_id, source_id
        FROM context_sources
        WHERE save_id = ? AND source_type = 'location'
        """,
        (imported_save_id,),
    ).fetchone()
    media_context = repositories.connection.execute(
        """
        SELECT source_id
        FROM context_sources
        WHERE save_id = ? AND source_type = 'media_asset'
        """,
        (imported_save_id,),
    ).fetchone()
    message_context = repositories.connection.execute(
        """
        SELECT source_id
        FROM context_sources
        WHERE save_id = ? AND source_type = 'message'
        """,
        (imported_save_id,),
    ).fetchone()
    location = repositories.connection.execute(
        """
        SELECT id, save_id, source_message_id, first_seen_message_id,
               last_updated_message_id
        FROM locations
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()
    character = repositories.connection.execute(
        """
        SELECT id, save_id, location_id, age, is_player_character,
               current_clothing, first_seen_message_id, last_updated_message_id
        FROM characters
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()
    scene = repositories.connection.execute(
        """
        SELECT id, save_id, current_location_id, present_character_ids_json,
               in_world_time, time_of_day, day_of_week, world_day_index,
               world_time_day_index, world_time_day_label, world_time_phase,
               world_time_clock_minutes, world_time_period_label,
               world_time_source_message_id, world_time_confidence,
               first_seen_message_id, last_updated_message_id
        FROM scene_snapshots
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()
    thread = repositories.connection.execute(
        """
        SELECT related_entities_json, first_seen_message_id, last_updated_message_id
        FROM active_threads
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()
    link = repositories.connection.execute(
        """
        SELECT source_message_id
        FROM entity_links
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()
    memory = repositories.connection.execute(
        """
        SELECT id
        FROM memories
        WHERE save_id = ? AND body = 'Mara knows the eastern signal code.'
        """,
        (imported_save_id,),
    ).fetchone()
    knowledge_edge = repositories.connection.execute(
        """
        SELECT character_id, target_type, target_id, source_message_id,
               source_message_ids_json
        FROM character_knowledge_edges
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()
    visibility = repositories.connection.execute(
        """
        SELECT message_id, character_id, visibility, source
        FROM message_visibility
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()
    presence = repositories.connection.execute(
        """
        SELECT message_id, character_id, source
        FROM message_scene_presence
        WHERE save_id = ?
        """,
        (imported_save_id,),
    ).fetchone()
    imported_media = repositories.list_media_assets(imported_save_id)[0]
    imported_message_ids = [
        message.id for message in repositories.list_messages(imported_save_id)
    ]
    job_count = repositories.connection.execute(
        "SELECT COUNT(*) FROM jobs WHERE save_id = ?",
        (imported_save_id,),
    ).fetchone()[0]

    assert location_context is not None
    assert media_context is not None
    assert message_context is not None
    assert location is not None
    assert character is not None
    assert scene is not None
    assert thread is not None
    assert link is not None
    assert memory is not None
    assert knowledge_edge is not None
    assert visibility is not None
    assert presence is not None
    assert location_context["id"] != "ctx-location"
    assert location_context["source_id"] == location["id"]
    assert media_context["source_id"] == imported_media.id
    assert message_context["source_id"] == ",".join(imported_message_ids)
    assert location["id"] != "location-tower"
    assert location["source_message_id"] != NARRATOR_MESSAGE_ID
    assert location["first_seen_message_id"] == imported_message_ids[1]
    assert location["last_updated_message_id"] == imported_message_ids[1]
    assert character["id"] != "character-mara"
    assert character["location_id"] == location["id"]
    assert character["age"] == "late 30s"
    assert character["current_clothing"] == (
        "Borrowed green raincoat over a linen shirt."
    )
    assert character["is_player_character"] == 1
    assert character["first_seen_message_id"] == imported_message_ids[0]
    assert character["last_updated_message_id"] == imported_message_ids[0]
    assert scene["current_location_id"] == location["id"]
    assert scene["in_world_time"] == "Monday night"
    assert scene["time_of_day"] == "night"
    assert scene["day_of_week"] == "monday"
    assert scene["world_day_index"] == 2
    assert scene["world_time_day_index"] == 2
    assert scene["world_time_day_label"] == "monday"
    assert scene["world_time_phase"] == "night"
    assert scene["world_time_clock_minutes"] is None
    assert scene["world_time_period_label"] == ""
    assert scene["world_time_source_message_id"] == imported_message_ids[1]
    assert scene["world_time_confidence"] == 0.87
    assert json.loads(scene["present_character_ids_json"]) == [character["id"]]
    assert scene["first_seen_message_id"] == imported_message_ids[1]
    assert scene["last_updated_message_id"] == imported_message_ids[1]
    assert json.loads(thread["related_entities_json"]) == [
        f"location:{location['id']}",
        f"character:{character['id']}",
    ]
    assert thread["first_seen_message_id"] == imported_message_ids[1]
    assert thread["last_updated_message_id"] == imported_message_ids[1]
    assert link["source_message_id"] == imported_message_ids[1]
    assert knowledge_edge["character_id"] == character["id"]
    assert knowledge_edge["target_type"] == "memory"
    assert knowledge_edge["target_id"] == memory["id"]
    assert knowledge_edge["source_message_id"] == imported_message_ids[1]
    assert json.loads(knowledge_edge["source_message_ids_json"]) == (
        imported_message_ids
    )
    assert visibility["message_id"] == imported_message_ids[1]
    assert visibility["character_id"] == character["id"]
    assert visibility["visibility"] == "visible"
    assert presence["message_id"] == imported_message_ids[1]
    assert presence["character_id"] == character["id"]
    assert presence["source"] == "context_snapshot"
    assert visibility["source"] == "scene_presence"
    assert job_count == 0


def test_import_save_remaps_dating_route_states(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    player = repositories.add_character(
        save_id=save.id,
        name="Ren Takahashi",
        character_id="character-ren",
        is_player_character=True,
        met=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        character_id="character-mika",
        relationships={player.name: "romance option for Ren Takahashi"},
        locked_fields=["name", "role", "appearance", "voice"],
        protected_from_maintenance=True,
        met=True,
    )
    repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="contact_exchanged",
        first_met_message_id=PLAYER_MESSAGE_ID,
        first_met_world_day_index=0,
        last_interaction_message_id=NARRATOR_MESSAGE_ID,
        last_interaction_world_day_index=2,
        completed_interactions=1,
        dates_completed=0,
        known_boundaries=["no instant commitment"],
        unresolved_questions=["what Mika wants after the festival"],
        next_reasonable_step="schedule a first date",
        route_id="route-mika",
    )
    bundle_path = tmp_path / "exports" / "route-state.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)
    imported = service.import_save(bundle_path)

    imported_save_id = _imported_save_id(imported)
    route = repositories.list_dating_route_states(imported_save_id)[0]
    characters_by_name = {
        character.name: character
        for character in repositories.list_characters(imported_save_id)
    }
    message_ids = [
        message.id for message in repositories.list_messages(imported_save_id)
    ]
    assert route.id != "route-mika"
    assert route.player_character_id == characters_by_name["Ren Takahashi"].id
    assert route.npc_character_id == characters_by_name["Mika Arai"].id
    assert route.first_met_message_id == message_ids[0]
    assert route.last_interaction_message_id == message_ids[1]
    assert route.stage == "contact_exchanged"
    assert route.known_boundaries == ["no instant commitment"]
    assert route.unresolved_questions == ["what Mika wants after the festival"]
    imported_mika = characters_by_name["Mika Arai"]
    assert imported_mika.protected_from_maintenance is True
    assert imported_mika.relationships == {
        "Ren Takahashi": "romance option for Ren Takahashi"
    }
    assert {"name", "role", "appearance", "voice"} <= set(imported_mika.locked_fields)
    assert "relationships" not in imported_mika.locked_fields


def test_import_save_remaps_context_source_metadata_message_refs(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="world_state",
        source_id="beacon_lens",
        title="Beacon lens",
        body="The beacon lens burns red.",
        metadata={
            "source_message_id": PLAYER_MESSAGE_ID,
            "source_message_ids": [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
            "last_seen_message_id": NARRATOR_MESSAGE_ID,
            "fact_type": "location",
        },
        context_source_id="ctx-metadata-remap",
    )
    bundle_path = tmp_path / "exports" / "night-watch-context-remap.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_messages = repositories.list_messages(imported_save_id)
    imported_source = repositories.list_context_sources(
        imported_save_id,
        source_type="world_state",
    )[0]

    assert imported_source.metadata == {
        "fact_type": "location",
        "last_seen_message_id": imported_messages[1].id,
        "source_message_id": imported_messages[0].id,
        "source_message_ids": [imported_messages[0].id, imported_messages[1].id],
    }
    archived_ids = repositories.archive_context_sources_for_deleted_messages(
        save_id=imported_save_id,
        message_ids={imported_messages[1].id},
    )
    assert archived_ids == frozenset({imported_source.id})


def test_import_save_remaps_context_source_row_id_source(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    prior_summary = repositories.add_summary(
        summary_id="summary-prior-source",
        save_id=save.id,
        covers_message_start_id=PLAYER_MESSAGE_ID,
        covers_message_end_id=PLAYER_MESSAGE_ID,
        body="Mara began the beacon approach.",
        provider="fake-summary-provider",
        model="fake-summary-model",
        source_message_ids=(PLAYER_MESSAGE_ID,),
    )
    repositories.archive_summary(prior_summary.id)
    summary = repositories.add_summary(
        summary_id="summary-context-source",
        save_id=save.id,
        covers_message_start_id=PLAYER_MESSAGE_ID,
        covers_message_end_id=NARRATOR_MESSAGE_ID,
        body="The beacon warning summary is indexed as context.",
        provider="fake-summary-provider",
        model="fake-summary-model",
        source_message_ids=(PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID),
        source_summary_ids=("summary-prior-source",),
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="summary",
        source_id=summary.id,
        title="Summary context",
        body=summary.body,
        metadata={"indexed_by": "test"},
        context_source_id="ctx-summary-source",
    )
    bundle_path = tmp_path / "exports" / "night-watch-summary-context.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_messages = repositories.list_messages(imported_save_id)

    imported_summary = next(
        summary
        for summary in repositories.list_summaries(imported_save_id)
        if summary.body == "The beacon warning summary is indexed as context."
    )
    [imported_source] = repositories.list_context_sources(
        imported_save_id,
        source_type="summary",
    )
    assert imported_summary.id != summary.id
    assert imported_summary.source_message_ids == (
        imported_messages[0].id,
        imported_messages[1].id,
    )
    imported_prior_summary = repositories.connection.execute(
        """
        SELECT id, archived_at
        FROM summaries
        WHERE save_id = ? AND body = ?
        """,
        (imported_save_id, prior_summary.body),
    ).fetchone()
    assert imported_prior_summary is not None
    assert imported_prior_summary["archived_at"] is not None
    assert imported_summary.source_summary_ids == (imported_prior_summary["id"],)
    assert imported_source.source_id == imported_summary.id


def test_import_save_remaps_plural_comma_joined_message_context_source(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="messages",
        source_id=f"{PLAYER_MESSAGE_ID},{NARRATOR_MESSAGE_ID}",
        title="Two-message context",
        body="The warning develops across two messages.",
        metadata={
            "source_message_ids": [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
        },
        context_source_id="ctx-plural-message-source",
    )
    bundle_path = tmp_path / "exports" / "plural-message-source.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)
    imported = service.import_save(bundle_path)

    imported_save_id = _imported_save_id(imported)
    imported_message_ids = {
        message.id for message in repositories.list_messages(imported_save_id)
    }
    [imported_source] = repositories.list_context_sources(
        imported_save_id,
        source_type="messages",
    )
    remapped_source_ids = set(imported_source.source_id.split(","))
    assert remapped_source_ids <= imported_message_ids
    assert PLAYER_MESSAGE_ID not in remapped_source_ids
    assert NARRATOR_MESSAGE_ID not in remapped_source_ids


def test_import_save_repairs_world_state_context_source_ids(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    _seed_context_graph_rows(repositories, save.id)
    state = next(
        state
        for state in repositories.list_world_state(save.id)
        if state.key == "beacon_lens"
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="world_state",
        source_id=state.id,
        title="Beacon direct state",
        body="The beacon lens state is indexed by row id.",
        metadata={},
        context_source_id="ctx-direct-world-state",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="world_state",
        source_id="beacon_lens",
        title="Beacon key state",
        body="The beacon lens state is indexed by key.",
        metadata={},
        context_source_id="ctx-key-world-state",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="world_state",
        source_id="location:location-tower",
        title="Tower location state",
        body="The beacon tower is indexed as world state.",
        metadata={},
        context_source_id="ctx-location-world-state",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="world_state",
        source_id="0a0cf9ab628e43148d51999b1e521b2e",
        title="Missing direct state",
        body="This context points at a state row that is not in the bundle.",
        metadata={},
        context_source_id="ctx-missing-world-state",
    )
    bundle_path = tmp_path / "exports" / "night-watch-world-state-context.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_state = next(
        state
        for state in repositories.list_world_state(imported_save_id)
        if state.key == "beacon_lens"
    )
    imported_location = next(
        location
        for location in repositories.list_locations(imported_save_id)
        if location.name == "Beacon Tower"
    )
    imported_world_state_sources = {
        source.title: source
        for source in repositories.list_context_sources(
            imported_save_id,
            source_type="world_state",
        )
    }
    assert imported_world_state_sources["Beacon direct state"].source_id == (
        imported_state.id
    )
    assert imported_world_state_sources["Beacon key state"].source_id == "beacon_lens"
    assert imported_world_state_sources["Tower location state"].source_id == (
        f"location:{imported_location.id}"
    )
    assert "Missing direct state" not in imported_world_state_sources


def test_import_save_remaps_scenario_section_context_source_ids(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    scenario_content = json.loads(scenario.content_json)
    assert isinstance(scenario_content, dict)
    scenario_content["factions"] = "The ash guild wants the beacon intact."
    repositories.update_scenario(
        scenario_id=scenario.id,
        title=scenario.title,
        premise=scenario.premise,
        player_role=scenario.player_role,
        content=scenario_content,
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="scenario_section",
        source_id=f"scenario:{scenario.id}:section:factions",
        title="factions",
        body="The ash guild wants the beacon intact.",
        metadata={
            "indexed_by": "continuity_index",
            "scenario_id": scenario.id,
            "fact_type": "scenario_section",
        },
        context_source_id="ctx-scenario-factions",
    )
    bundle_path = tmp_path / "exports" / "night-watch-scenario-context.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    [imported_source] = repositories.list_context_sources(
        imported_save_id,
        source_type="scenario_section",
    )
    assert imported.scenario_id != scenario.id
    assert imported_source.source_id == (
        f"scenario:{imported.scenario_id}:section:factions"
    )
    assert imported_source.metadata["scenario_id"] == imported.scenario_id


def test_import_save_repairs_memory_context_source_ids(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    _seed_context_graph_rows(repositories, save.id)
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="memory-not-exported",
        title="Missing memory context",
        body="This context points at a memory row that is not in the bundle.",
        metadata={},
        context_source_id="ctx-missing-memory",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="character_profile:character-mara",
        title="Mara profile context",
        body="Mara is a signal warden.",
        metadata={},
        context_source_id="ctx-mara-profile-memory",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="relationship:character-mara:signal-partner",
        title="Mara relationship context",
        body="Mara trusts her signal partner.",
        metadata={},
        context_source_id="ctx-mara-relationship-memory",
    )
    bundle_path = tmp_path / "exports" / "night-watch-memory-context.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_mara = next(
        character
        for character in repositories.list_characters(imported_save_id)
        if character.name == "Mara"
    )
    imported_memory_sources = {
        source.title: source
        for source in repositories.list_context_sources(
            imported_save_id,
            source_type="memory",
        )
    }
    assert "Missing memory context" not in imported_memory_sources
    assert (
        imported_memory_sources["Mara profile context"].source_id
        == f"character_profile:{imported_mara.id}"
    )
    assert (
        imported_memory_sources["Mara relationship context"].source_id
        == f"relationship:{imported_mara.id}:signal-partner"
    )


def test_import_save_remaps_observation_context_source_metadata_id(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id=OBSERVATION_ID,
        title="Curated observation",
        body="The red beacon warning may matter later.",
        metadata={
            "observation_id": OBSERVATION_ID,
            "observation_type": "open_thread",
            "curation_action": "save_context",
            "source_message_ids": [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
        },
        context_source_id="ctx-observation-source",
    )
    bundle_path = tmp_path / "exports" / "night-watch-observation-context.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    [imported_observation] = repositories.list_context_observations(
        imported_save_id,
    )
    [imported_source] = repositories.list_context_sources(
        imported_save_id,
        source_type="observation",
    )
    assert imported_observation.id != OBSERVATION_ID
    assert imported_source.source_id == imported_observation.id
    assert imported_source.metadata["observation_id"] == imported_observation.id


def test_import_save_preserves_memory_and_scene_scratch_provenance(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    [memory] = repositories.list_memories(save.id)
    repositories.update_memory(
        memory_id=memory.id,
        body=memory.body,
        tags=memory.tags,
        importance=memory.importance,
        source_message_ids=memory.source_message_ids,
        source_observation_ids=[OBSERVATION_ID],
        claim_fingerprint="forged-imported-fingerprint",
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id=memory.id,
        title="Beacon memory",
        body=memory.body,
        metadata={
            "source_message_ids": [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
            "source_provenance_groups": [
                [PLAYER_MESSAGE_ID],
                [NARRATOR_MESSAGE_ID],
            ],
            "source_provenance_mode": "any",
        },
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon gallery",
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    scene = repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="The lens is still warm.",
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="observation",
        source_id=OBSERVATION_ID,
        title="Warm lens",
        body="The beacon lens remains warm.",
        metadata={
            "observation_id": OBSERVATION_ID,
            "curation_action": "scene_scratch",
        },
        scene_snapshot_id=scene.id,
        scene_generation=scene.scene_generation,
        created_turn_number=1,
        expires_after_turn_number=13,
    )
    bundle_path = tmp_path / "exports" / "night-watch-scratch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    [imported_observation] = repositories.list_context_observations(imported_save_id)
    [imported_memory] = repositories.list_memories(imported_save_id)
    [imported_scratch] = repositories.list_context_sources(
        imported_save_id,
        source_type="observation",
    )
    imported_scene = repositories.get_scene_snapshot(imported_save_id)
    [imported_memory_source] = repositories.list_context_sources(
        imported_save_id,
        source_type="memory",
    )
    imported_messages = repositories.list_messages(imported_save_id)
    assert imported_scene is not None
    assert imported_memory.claim_fingerprint == canonical_claim_fingerprint(
        imported_memory.body
    )
    assert imported_memory.source_observation_ids == [imported_observation.id]
    assert imported_memory_source.source_id == imported_memory.id
    assert imported_memory_source.metadata["source_provenance_groups"] == [
        [imported_messages[0].id],
        [imported_messages[1].id],
    ]
    assert imported_scratch.source_id == imported_observation.id
    assert imported_scratch.scene_snapshot_id == imported_scene.id
    assert imported_scratch.scene_generation == scene.scene_generation
    assert imported_scratch.created_turn_number == 1
    assert imported_scratch.expires_after_turn_number == 13


def test_export_import_preserves_epistemic_fields_and_remaps_actor(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    courier = repositories.add_character(save_id=save.id, name="Courier")
    second_courier = repositories.add_character(save_id=save.id, name="Courier")
    repositories.add_memory(
        save_id=save.id,
        body="The courier reports that the north gate is unguarded.",
        tags=["hearsay"],
        source_message_id=PLAYER_MESSAGE_ID,
        epistemic_status="reported_speech",
        epistemic_actor_id=courier.id,
        epistemic_actor_name=courier.name,
    )
    repositories.add_memory(
        save_id=save.id,
        body="The courier reports that the north gate is unguarded.",
        tags=["hearsay"],
        source_message_id=PLAYER_MESSAGE_ID,
        epistemic_status="reported_speech",
        epistemic_actor_id=second_courier.id,
        epistemic_actor_name=second_courier.name,
    )
    repositories.add_context_observation(
        save_id=save.id,
        observation_type="character_fact",
        claim="The courier intends to leave before dawn.",
        evidence_quote="I climb toward the beacon lens.",
        source_message_ids=[PLAYER_MESSAGE_ID],
        epistemic_status="intention",
        epistemic_actor_id=courier.id,
        epistemic_actor_name=courier.name,
    )
    bundle_path = tmp_path / "exports" / "epistemic-round-trip.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_couriers = [
        character
        for character in repositories.list_characters(imported_save_id)
        if character.name == "Courier"
    ]
    imported_memories = [
        memory
        for memory in repositories.list_memories(imported_save_id)
        if "north gate" in memory.body
    ]
    imported_observation = next(
        observation
        for observation in repositories.list_context_observations(imported_save_id)
        if "before dawn" in observation.claim
    )

    assert len(imported_couriers) == 2
    assert len(imported_memories) == 2
    assert {memory.epistemic_actor_id for memory in imported_memories} == {
        character.id for character in imported_couriers
    }
    assert all(
        memory.epistemic_status == "reported_speech"
        and memory.epistemic_actor_name == "Courier"
        for memory in imported_memories
    )
    assert imported_observation.epistemic_status == "intention"
    assert imported_observation.epistemic_actor_id in {
        character.id for character in imported_couriers
    }


def test_import_save_accepts_legacy_memories_without_observation_provenance(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "exports" / "night-watch-schema-70.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    def remove_v71_memory_fields(data: dict[str, object]) -> None:
        rows = data["memories"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            row.pop("source_observation_ids_json", None)

    _rewrite_bundle_data(bundle_path, remove_v71_memory_fields)

    imported = service.import_save(bundle_path)
    [imported_memory] = repositories.list_memories(_imported_save_id(imported))

    assert imported_memory.source_observation_ids == []


def test_import_save_ignores_historical_job_diagnostics(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "exports" / "night-watch-historical-jobs.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)
    _rewrite_bundle_data(
        bundle_path,
        lambda data: data.__setitem__(
            "jobs",
            [
                {
                    "id": "historical-job",
                    "save_id": SAVE_ID,
                    "type": "provider_debug",
                    "status": "failed",
                    "payload_json": json.dumps({"api_key": "sk-secret"}),
                    "result_json": None,
                    "error": "request body contained a secret",
                    "created_at": "2026-05-01 00:00:00",
                    "started_at": None,
                    "completed_at": None,
                }
            ],
        ),
    )

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    job_count = repositories.connection.execute(
        "SELECT COUNT(*) FROM jobs WHERE save_id = ?",
        (imported_save_id,),
    ).fetchone()[0]

    assert job_count == 0


def test_import_save_preserves_chronicle_message_timestamps(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "exports" / "night-watch-timestamps.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)
    expected_timestamps = {
        PLAYER_MESSAGE_ID: {
            "created_at": "2001-02-03 04:05:06.007",
            "updated_at": "2001-02-03 04:05:07.008",
        },
        NARRATOR_MESSAGE_ID: {
            "created_at": "2001-02-03 04:06:06.007",
            "updated_at": "2001-02-03 04:06:07.008",
        },
    }

    def rewrite_timestamps(data: dict[str, object]) -> None:
        rows = data["messages"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            timestamp_values = expected_timestamps[row["id"]]
            row["created_at"] = timestamp_values["created_at"]
            row["updated_at"] = timestamp_values["updated_at"]

    _rewrite_bundle_data(bundle_path, rewrite_timestamps)

    imported = service.import_save(bundle_path)
    loaded = repositories.load_save_details(imported.save_id)
    assert loaded is not None

    imported_messages = loaded.messages
    assert [message.id for message in imported_messages] != [
        PLAYER_MESSAGE_ID,
        NARRATOR_MESSAGE_ID,
    ]
    assert [
        (message.created_at, message.updated_at) for message in imported_messages
    ] == [
        (
            expected_timestamps[PLAYER_MESSAGE_ID]["created_at"],
            expected_timestamps[PLAYER_MESSAGE_ID]["updated_at"],
        ),
        (
            expected_timestamps[NARRATOR_MESSAGE_ID]["created_at"],
            expected_timestamps[NARRATOR_MESSAGE_ID]["updated_at"],
        ),
    ]


def test_import_save_accepts_legacy_messages_without_timestamps(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "exports" / "night-watch-legacy.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    def remove_message_timestamps(data: dict[str, object]) -> None:
        rows = data["messages"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            row.pop("created_at", None)
            row.pop("updated_at", None)

    _rewrite_bundle_data(bundle_path, remove_message_timestamps)

    imported = service.import_save(bundle_path)
    loaded = repositories.load_save_details(imported.save_id)
    assert loaded is not None

    assert len(loaded.messages) == 2
    assert all(message.created_at is not None for message in loaded.messages)
    assert all(message.updated_at is not None for message in loaded.messages)


def test_preview_import_reads_manifest_without_adding_saves(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    preview = service.preview_import(bundle_path)

    assert preview.title == "Night Watch"
    assert preview.scenario_title == "Ashfall Keep"
    assert preview.message_count == 2
    assert preview.media_count == 1
    assert preview.bundle_version == 1
    assert [listed_save.id for listed_save in repositories.list_saves()] == [SAVE_ID]


def test_preview_import_uses_manifest_and_import_rejects_save_id_mismatch(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)
    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))
        members = {
            name: bundle.read(name)
            for name in bundle.namelist()
            if name not in {"manifest.json", "data.json"}
        }
    manifest["save"]["id"] = "manifest-save-id"
    broken_bundle_path = tmp_path / "night-watch-mismatched.bragi-chat"
    _write_bundle_with_members(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        members=members,
    )

    preview = service.preview_import(broken_bundle_path)
    assert preview.save_id == "manifest-save-id"
    with pytest.raises(module.ChatBundleError, match="manifest"):
        service.import_save(broken_bundle_path)


def test_import_save_rejects_missing_snapshot_object_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    TurnSnapshotService(repositories).capture_message_snapshot(
        save_id=save.id,
        message_id=NARRATOR_MESSAGE_ID,
    )
    bundle_path = tmp_path / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)
    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))
        members = {
            name: bundle.read(name)
            for name in bundle.namelist()
            if name not in {"manifest.json", "data.json"}
        }
    snapshots = data["turn_snapshots"]
    assert isinstance(snapshots, list)
    assert snapshots
    snapshot = snapshots[0]
    assert isinstance(snapshot, dict)
    missing_hash = snapshot["root_manifest_hash"]
    snapshot_objects = data["snapshot_objects"]
    assert isinstance(snapshot_objects, list)
    data["snapshot_objects"] = [
        row
        for row in snapshot_objects
        if not isinstance(row, dict) or row.get("object_hash") != missing_hash
    ]
    broken_bundle_path = tmp_path / "night-watch-missing-snapshot-object.bragi-chat"
    _write_bundle_with_members(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        members=members,
    )
    save_ids = [listed_save.id for listed_save in repositories.list_saves()]

    assert service.preview_import(broken_bundle_path).title == "Night Watch"
    with pytest.raises(module.ChatBundleError, match="snapshot object"):
        service.import_save(broken_bundle_path)

    assert [listed_save.id for listed_save in repositories.list_saves()] == save_ids


def test_preview_import_reads_manifest_without_loading_oversized_data_json(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    module = _chat_bundle_module()
    service, manifest, _data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    manifest_payload = json.dumps(manifest).encode("utf-8")
    data_payload = b"{invalid oversized data" + (b"x" * len(manifest_payload))
    monkeypatch.setattr(module, "_MAX_BUNDLE_DATA_JSON_BYTES", len(manifest_payload))
    broken_bundle_path = tmp_path / "night-watch-oversized-data-preview.bragi-chat"
    _write_bundle_member_bytes(
        broken_bundle_path,
        manifest_payload=manifest_payload,
        data_payload=data_payload,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )

    preview = service.preview_import(broken_bundle_path)

    assert preview.title == "Night Watch"
    with pytest.raises(module.ChatBundleError, match="data.json is too large"):
        service.import_save(broken_bundle_path)


def test_import_save_remaps_colliding_ids_and_preserves_bundle_data(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(original_save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    assert imported_save_id != SAVE_ID
    imported_save = _require_save(repositories, imported_save_id)
    assert imported_save.scenario_id != SCENARIO_ID
    assert imported_save.title == "Night Watch"
    assert imported_save.custom_instructions == "Keep choices brief and grounded."
    assert (
        repositories.get_app_setting(
            save_scenario_evolution_turn_interval_setting_key(imported_save_id)
        )
        == 3
    )

    assert (
        repositories.get_app_setting(
            save_image_style_preset_setting_key(imported_save_id)
        )
        == "watercolor"
    )
    assert (
        repositories.get_app_setting(
            scenario_template_evolution_turn_interval_setting_key(
                imported_save.scenario_id
            )
        )
        == 5
    )

    base_scenario = repositories.get_scenario(imported_save.scenario_id)
    assert base_scenario is not None
    assert base_scenario.title == "Ashfall Keep"
    loaded = repositories.load_save_details(imported_save_id)
    assert loaded is not None
    assert loaded.scenario.id == imported_save.scenario_id
    assert loaded.scenario.title == "Ashfall Keep: Red Lens"
    assert json.loads(loaded.scenario.content_json) == {
        "_source": {"content_rating": "unclassified"},
        "opening_message": "The red lens wakes.",
        "starting_scene": "The beacon burns crimson over the ash road.",
    }

    imported_messages = loaded.messages
    assert [message.body for message in imported_messages] == [
        "I climb toward the beacon lens.",
        "The lens flashes red and shows riders in the ash.",
    ]
    message_id_map = {
        PLAYER_MESSAGE_ID: imported_messages[0].id,
        NARRATOR_MESSAGE_ID: imported_messages[1].id,
    }
    assert message_id_map[PLAYER_MESSAGE_ID] != PLAYER_MESSAGE_ID
    assert message_id_map[NARRATOR_MESSAGE_ID] != NARRATOR_MESSAGE_ID
    assert all(message.save_id == imported_save_id for message in imported_messages)

    world_state = repositories.list_world_state(imported_save_id)
    assert [state.key for state in world_state] == ["beacon_lens"]
    assert world_state[0].value == {"color": "red", "lit": True}
    assert world_state[0].source_message_id == message_id_map[NARRATOR_MESSAGE_ID]

    memories = repositories.list_memories(imported_save_id)
    assert len(memories) == 1
    assert memories[0].body == "Mara knows the eastern signal code."
    assert memories[0].tags == ["mara", "signals"]
    assert memories[0].source_message_id == message_id_map[PLAYER_MESSAGE_ID]
    assert memories[0].source_message_ids == [
        message_id_map[PLAYER_MESSAGE_ID],
        message_id_map[NARRATOR_MESSAGE_ID],
    ]

    observations = repositories.list_context_observations(imported_save_id)
    assert len(observations) == 1
    assert observations[0].claim == "The red beacon warning may matter later."
    assert observations[0].source_message_ids == [
        message_id_map[PLAYER_MESSAGE_ID],
        message_id_map[NARRATOR_MESSAGE_ID],
    ]
    curation_state = repositories.get_context_observation_curation_state(
        observations[0].id
    )
    assert curation_state is not None
    assert curation_state.terminal_outcome == "accepted"
    assert curation_state.lease_token is None

    summaries = repositories.list_summaries(imported_save_id)
    assert len(summaries) == 1
    assert summaries[0].covers_message_start_id == message_id_map[PLAYER_MESSAGE_ID]
    assert summaries[0].covers_message_end_id == message_id_map[NARRATOR_MESSAGE_ID]
    assert summaries[0].body == "Mara reaches the beacon and sees a warning."

    state_changes = repositories.list_state_changes(imported_save_id)
    assert len(state_changes) == 1
    assert state_changes[0].source_message_id == message_id_map[NARRATOR_MESSAGE_ID]
    assert state_changes[0].state_key == "beacon_lens"
    assert json.loads(state_changes[0].after_json or "{}") == {
        "color": "red",
        "lit": True,
    }

    scenario_updates = repositories.list_save_scenario_updates(imported_save_id)
    assert len(scenario_updates) == 1
    assert scenario_updates[0].id != SCENARIO_UPDATE_ID
    assert scenario_updates[0].source_message_id == message_id_map[NARRATOR_MESSAGE_ID]
    assert json.loads(scenario_updates[0].source_message_ids_json) == [
        message_id_map[PLAYER_MESSAGE_ID],
        message_id_map[NARRATOR_MESSAGE_ID],
    ]

    media_assets = repositories.list_media_assets(imported_save_id)
    assert len(media_assets) == 1
    assert media_assets[0].id != MEDIA_ASSET_ID
    assert media_assets[0].source_message_id == message_id_map[NARRATOR_MESSAGE_ID]
    assert media_assets[0].prompt == "red beacon lens over ash"
    assert media_assets[0].provider == "fake-image-provider"
    assert media_assets[0].model == "fake-image-model"
    assert Path(media_assets[0].path).parts[0] == imported_save_id
    assert (media_dir / media_assets[0].path).read_bytes() == MEDIA_BYTES


def test_import_save_streams_media_members_without_zip_read(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    bundle_path = tmp_path / "night-watch-stream-media.bragi-chat"
    _write_bundle_with_member(
        bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    original_read = zipfile.ZipFile.read

    def fail_media_read(
        bundle: zipfile.ZipFile,
        name: str | zipfile.ZipInfo,
        pwd: bytes | None = None,
    ) -> bytes:
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        if filename == bundle_media_path:
            raise AssertionError("media members should be streamed")
        return original_read(bundle, name, pwd)

    monkeypatch.setattr(zipfile.ZipFile, "read", fail_media_read)

    imported = service.import_save(bundle_path)

    imported_save_id = _imported_save_id(imported)
    imported_media = repositories.list_media_assets(imported_save_id)
    assert len(imported_media) == 1
    assert (tmp_path / "media" / imported_media[0].path).read_bytes() == MEDIA_BYTES


def test_export_and_import_preserves_text_derived_source_refs(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    text_message = _seed_text_derived_world_rows(repositories, save.id)
    player = next(
        (
            character
            for character in repositories.list_characters(save.id)
            if character.is_player_character
        ),
        None,
    )
    if player is None:
        player = repositories.add_character(
            character_id="character-mara",
            save_id=save.id,
            name="Mara",
            role="Signal warden",
            is_player_character=True,
            met=True,
        )
    rowan = next(
        character
        for character in repositories.list_characters(save.id)
        if character.name == "Rowan"
    )
    repositories.upsert_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=rowan.id,
        player_has_character_number=True,
        character_has_player_number=False,
        source_text_message_id=text_message.id,
        state_id="contact-rowan",
    )
    repositories.add_character_text_proactive_trigger(
        save_id=save.id,
        character_id=rowan.id,
        trigger_key="dating_route:route-rowan:turn-1",
        trigger_type="dating_route",
        thread_id=text_message.thread_id,
        text_message_id=text_message.id,
        source_type="dating_route_state",
        source_id="route-rowan",
        reason="Ask about the repair notes.",
        trigger_id="trigger-rowan",
    )
    repositories.update_character_text_thread_memory(
        save_id=save.id,
        thread_id=text_message.thread_id,
        body=(
            "Phone thread memory: Rowan and Mara use the repair notes as their "
            "west arcade meetup cue."
        ),
        message_count=2,
    )
    source_ref = f"character_text_message:{text_message.id}"
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="character_text_thread",
        source_id=text_message.thread_id,
        title="Rowan phone thread",
        body="Phone thread memory: Rowan will bring repair notes.",
        metadata={
            "indexed_by": "continuity_index",
            "fact_type": "character_text_thread",
            "source_message_ids": [source_ref],
            "audience_character_ids": [player.id, rowan.id],
            "known_by": [player.name, rowan.name],
            "thread_id": text_message.thread_id,
            "entity_ids": [text_message.thread_id],
        },
        context_source_id="ctx-rowan-text-thread",
    )
    bundle_path = tmp_path / "exports" / "night-watch-text-world.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["character_text_threads"][0]["memory_body"] == (
        "Phone thread memory: Rowan and Mara use the repair notes as their west "
        "arcade meetup cue."
    )
    assert data["character_text_threads"][0]["memory_message_count"] == 2
    assert data["character_text_threads"][0]["memory_updated_at"] is not None
    text_memory_rows = [
        row
        for row in data["memories"]
        if row["body"] == "Rowan promised to bring repair notes."
    ]
    assert len(text_memory_rows) == 1
    assert json.loads(text_memory_rows[0]["source_message_ids_json"]) == [source_ref]
    assert any(
        row["source_message_ids_json"] == json.dumps([source_ref])
        for row in data["context_update_audit"]
    )
    assert any(
        row["source_message_ids_json"] == json.dumps([source_ref])
        for row in data["character_knowledge_edges"]
    )
    assert data["character_contact_states"] == [
        {
            "id": "contact-rowan",
            "save_id": save.id,
            "player_character_id": player.id,
            "character_id": rowan.id,
            "player_has_character_number": 1,
            "character_has_player_number": 0,
            "source_message_id": None,
            "source_text_message_id": text_message.id,
            "created_at": data["character_contact_states"][0]["created_at"],
            "updated_at": data["character_contact_states"][0]["updated_at"],
            "archived_at": None,
        }
    ]
    assert data["character_text_proactive_triggers"] == [
        {
            "id": "trigger-rowan",
            "save_id": save.id,
            "character_id": rowan.id,
            "trigger_key": "dating_route:route-rowan:turn-1",
            "trigger_type": "dating_route",
            "thread_id": text_message.thread_id,
            "text_message_id": text_message.id,
            "source_type": "dating_route_state",
            "source_id": "route-rowan",
            "source_message_id": None,
            "reason": "Ask about the repair notes.",
            "created_at": data["character_text_proactive_triggers"][0][
                "created_at"
            ],
            "updated_at": data["character_text_proactive_triggers"][0][
                "updated_at"
            ],
        }
    ]

    imported = service.import_save(bundle_path)
    imported_text_messages = repositories.list_character_text_messages(
        save_id=imported.save_id,
    )
    imported_player_text = next(
        message
        for message in imported_text_messages
        if message.body == "Can you bring the repair notes?"
    )
    imported_reply = next(
        message
        for message in imported_text_messages
        if message.body == text_message.body
    )
    imported_thread = repositories.get_character_text_thread(
        save_id=imported.save_id,
        thread_id=imported_reply.thread_id,
    )
    assert imported_thread is not None
    assert imported_thread.memory_body == (
        "Phone thread memory: Rowan and Mara use the repair notes as their west "
        "arcade meetup cue."
    )
    assert imported_thread.memory_message_count == 2
    assert imported_thread.memory_updated_at is not None
    imported_source_ref = f"character_text_message:{imported_reply.id}"
    assert imported_reply.reply_to_message_id == imported_player_text.id
    assert imported_reply.in_world_sent_at == "Friday evening after class"
    assert imported_reply.delivered_at == "2026-07-01T12:06:00+00:00"
    assert imported_reply.read_at == "2026-07-01T12:07:00+00:00"
    imported_memory = next(
        memory
        for memory in repositories.list_memories(imported.save_id)
        if memory.body == "Rowan promised to bring repair notes."
    )
    assert imported_memory.source_message_id is None
    assert imported_memory.source_message_ids == [imported_source_ref]
    imported_audit = repositories.list_context_update_audit(imported.save_id)
    assert any(
        row.source_message_ids == [imported_source_ref] for row in imported_audit
    )
    imported_edges = repositories.list_character_knowledge_edges(imported.save_id)
    assert any(
        edge.source_message_ids == [imported_source_ref] for edge in imported_edges
    )
    imported_provenance = repositories.list_character_text_provenance(
        save_id=imported.save_id,
        text_message_id=imported_reply.id,
    )
    assert any(row.target_type == "memory" for row in imported_provenance)
    imported_contacts = repositories.list_character_contact_states(imported.save_id)
    imported_characters = {
        character.name: character
        for character in repositories.list_characters(imported.save_id)
    }
    assert len(imported_contacts) == 1
    assert imported_contacts[0].player_character_id == imported_characters["Mara"].id
    assert imported_contacts[0].character_id == imported_characters["Rowan"].id
    assert imported_contacts[0].player_has_character_number is True
    assert imported_contacts[0].character_has_player_number is False
    assert imported_contacts[0].source_text_message_id == imported_reply.id
    [imported_thread_context] = repositories.list_context_sources(
        imported.save_id,
        source_type="character_text_thread",
    )
    assert imported_thread_context.source_id == imported_thread.id
    assert imported_thread_context.metadata["source_message_ids"] == [
        imported_source_ref
    ]
    audience_character_ids = imported_thread_context.metadata["audience_character_ids"]
    assert isinstance(audience_character_ids, list)
    assert set(audience_character_ids) == {
        imported_characters["Mara"].id,
        imported_characters["Rowan"].id,
    }
    assert imported_thread_context.metadata["thread_id"] == imported_thread.id
    assert imported_thread_context.metadata["entity_ids"] == [imported_thread.id]
    imported_triggers = repositories.list_character_text_proactive_triggers(
        imported.save_id
    )
    assert len(imported_triggers) == 1
    assert imported_triggers[0].character_id == imported_characters["Rowan"].id
    assert imported_triggers[0].trigger_key == "dating_route:route-rowan:turn-1"
    assert imported_triggers[0].thread_id == imported_reply.thread_id
    assert imported_triggers[0].text_message_id == imported_reply.id


def test_export_and_import_preserves_group_character_text_threads(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    player = repositories.add_character(
        character_id="character-mara",
        save_id=save.id,
        name="Mara",
        is_player_character=True,
        met=True,
    )
    rowan = repositories.add_character(
        character_id="character-rowan",
        save_id=save.id,
        name="Rowan",
        met=True,
    )
    lio = repositories.add_character(
        character_id="character-lio",
        save_id=save.id,
        name="Lio",
        met=True,
    )
    thread = repositories.create_character_text_group_thread(
        save_id=save.id,
        title="Beacon Crew",
        character_ids=[rowan.id, lio.id],
    )
    player_message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=None,
        sender="player",
        sender_character_id=player.id,
        body="Can both of you check the beacon lens?",
    )
    rowan_reply = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=rowan.id,
        sender="character",
        sender_character_id=rowan.id,
        body="I can take the lower stairs.",
        reply_to_message_id=player_message.id,
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=lio.id,
        sender="character",
        sender_character_id=lio.id,
        body="I will watch the east window.",
        reply_to_message_id=player_message.id,
    )
    bundle_path = tmp_path / "exports" / "beacon-crew.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["character_text_threads"][0]["kind"] == "group"
    assert data["character_text_threads"][0]["character_id"] is None
    assert [
        row["character_id"]
        for row in data["character_text_thread_participants"]
    ] == [rowan.id, lio.id]
    assert [
        row["sender_character_id"] for row in data["character_text_messages"]
    ] == [player.id, rowan.id, lio.id]

    imported = service.import_save(bundle_path)

    imported_characters = {
        character.name: character
        for character in repositories.list_characters(imported.save_id)
    }
    imported_thread = next(
        thread
        for thread in repositories.list_character_text_threads(imported.save_id)
        if thread.title == "Beacon Crew"
    )
    assert imported_thread.kind == "group"
    assert imported_thread.character_id is None
    imported_participants = repositories.list_character_text_thread_participants(
        save_id=imported.save_id,
        thread_id=imported_thread.id,
    )
    assert [participant.character_id for participant in imported_participants] == [
        imported_characters["Rowan"].id,
        imported_characters["Lio"].id,
    ]
    imported_messages = repositories.list_character_text_messages(
        save_id=imported.save_id,
        thread_id=imported_thread.id,
    )
    assert [
        (message.sender, message.sender_character_id)
        for message in imported_messages
    ] == [
        ("player", imported_characters["Mara"].id),
        ("character", imported_characters["Rowan"].id),
        ("character", imported_characters["Lio"].id),
    ]
    imported_rowan_reply = next(
        message for message in imported_messages if message.body == rowan_reply.body
    )
    assert imported_rowan_reply.reply_to_message_id == imported_messages[0].id


def test_export_and_import_preserves_character_text_attachments(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    text_message = _seed_text_derived_world_rows(repositories, save.id)
    rowan = next(
        character
        for character in repositories.list_characters(save.id)
        if character.name == "Rowan"
    )
    attachment_media_path = "save-night-watch/texts/ticket-stub.png"
    attachment_media_bytes = b"ticket stub image bytes"
    media_asset = repositories.create_media_asset(
        asset_id="media-text-ticket-stub",
        save_id=save.id,
        source_message_id=None,
        type="image",
        path=attachment_media_path,
        thumbnail_path=None,
        prompt="creased arcade ticket stub on a dusty cabinet",
        provider="fake-image-provider",
        model="fake-image-model",
        status="succeeded",
        metadata={
            "kind": "character_text_object_context_image",
            "thread_id": text_message.thread_id,
            "text_message_id": text_message.id,
            "character_id": rowan.id,
        },
    )
    media_path = media_dir / attachment_media_path
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(attachment_media_bytes)
    repositories.add_character_text_message_attachment(
        attachment_id="text-attachment-ticket-stub",
        save_id=save.id,
        thread_id=text_message.thread_id,
        text_message_id=text_message.id,
        character_id=rowan.id,
        kind="object_context_image",
        status="succeeded",
        media_asset_id=media_asset.id,
        prompt="creased arcade ticket stub on a dusty cabinet",
        metadata={"decision_reason": "Rowan texted a concrete found object."},
    )
    bundle_path = tmp_path / "night-watch-text-attachment.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    manifest = service.export_save(save.id, bundle_path)

    assert manifest.media_count == 2
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
        assert [row["id"] for row in data["character_text_message_attachments"]] == [
            "text-attachment-ticket-stub",
        ]
        assert _media_asset_by_id(
            data["media_assets"],
            "media-text-ticket-stub",
        )["source_message_id"] is None
        media_members = {
            name: bundle.read(name)
            for name in bundle.namelist()
            if name.startswith("media/")
        }
    assert attachment_media_bytes in media_members.values()

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_reply = next(
        message
        for message in repositories.list_character_text_messages(
            save_id=imported_save_id,
        )
        if message.body == text_message.body
    )
    imported_attachments = repositories.list_character_text_message_attachments(
        save_id=imported_save_id,
        text_message_ids=[imported_reply.id],
    )
    assert len(imported_attachments) == 1
    imported_attachment = imported_attachments[0]
    assert imported_attachment.id != "text-attachment-ticket-stub"
    assert imported_attachment.kind == "object_context_image"
    assert imported_attachment.status == "succeeded"
    assert imported_attachment.media_asset_id is not None
    imported_media = next(
        asset
        for asset in repositories.list_media_assets(imported_save_id)
        if asset.id == imported_attachment.media_asset_id
    )
    assert imported_media.id != "media-text-ticket-stub"
    assert imported_media.source_message_id is None
    assert imported_media.prompt == "creased arcade ticket stub on a dusty cabinet"
    assert (media_dir / imported_media.path).read_bytes() == attachment_media_bytes
    metadata = json.loads(imported_media.metadata_json)
    assert metadata["text_message_id"] == imported_reply.id
    assert metadata["thread_id"] == imported_reply.thread_id


def test_import_save_repairs_text_attachment_media_when_asset_is_snapshot_only(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    text_message = _seed_text_derived_world_rows(repositories, save.id)
    rowan = next(
        character
        for character in repositories.list_characters(save.id)
        if character.name == "Rowan"
    )
    attachment_media_path = "save-night-watch/texts/ticket-stub.png"
    attachment_media_bytes = b"ticket stub image bytes"
    media_asset = repositories.create_media_asset(
        asset_id="media-text-ticket-stub",
        save_id=save.id,
        source_message_id=None,
        type="image",
        path=attachment_media_path,
        thumbnail_path=None,
        prompt="creased arcade ticket stub on a dusty cabinet",
        provider="fake-image-provider",
        model="fake-image-model",
        status="succeeded",
        metadata={
            "kind": "character_text_object_context_image",
            "thread_id": text_message.thread_id,
            "text_message_id": text_message.id,
            "character_id": rowan.id,
        },
    )
    media_path = media_dir / attachment_media_path
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(attachment_media_bytes)
    repositories.add_character_text_message_attachment(
        attachment_id="text-attachment-ticket-stub",
        save_id=save.id,
        thread_id=text_message.thread_id,
        text_message_id=text_message.id,
        character_id=rowan.id,
        kind="object_context_image",
        status="succeeded",
        media_asset_id=media_asset.id,
        prompt="creased arcade ticket stub on a dusty cabinet",
        metadata={"decision_reason": "Rowan texted a concrete found object."},
    )
    bundle_path = tmp_path / "night-watch-text-attachment.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))
        media_members = {
            name: bundle.read(name)
            for name in bundle.namelist()
            if name.startswith("media/")
        }
    _move_media_asset_to_snapshot_only(
        manifest,
        data,
        media_asset_id="media-text-ticket-stub",
    )
    anomalous_bundle_path = tmp_path / "night-watch-text-snapshot-media.bragi-chat"
    _write_bundle_with_members(
        anomalous_bundle_path,
        manifest=manifest,
        data=data,
        members=media_members,
    )

    imported = service.import_save(anomalous_bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_reply = next(
        message
        for message in repositories.list_character_text_messages(
            save_id=imported_save_id,
        )
        if message.body == text_message.body
    )
    imported_attachments = repositories.list_character_text_message_attachments(
        save_id=imported_save_id,
        text_message_ids=[imported_reply.id],
    )
    assert len(imported_attachments) == 1
    assert imported_attachments[0].media_asset_id is None
    assert all(
        asset.prompt != "creased arcade ticket stub on a dusty cabinet"
        for asset in repositories.list_media_assets(imported_save_id)
    )


def test_import_save_repairs_live_graph_media_refs_when_asset_is_snapshot_only(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="media_asset",
        source_id=MEDIA_ASSET_ID,
        title="Beacon image context",
        body="The archived beacon frame is useful visual context.",
        metadata={},
        context_source_id="ctx-media-beacon",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="save",
        entity_id=save.id,
        target_type="media_asset",
        target_id=MEDIA_ASSET_ID,
        relation="visual_context",
    )
    bundle_path = tmp_path / "night-watch-media-graph.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))
        media_members = {
            name: bundle.read(name)
            for name in bundle.namelist()
            if name.startswith("media/")
        }
    _move_media_asset_to_snapshot_only(
        manifest,
        data,
        media_asset_id=MEDIA_ASSET_ID,
    )
    anomalous_bundle_path = tmp_path / "night-watch-media-graph-snapshot.bragi-chat"
    _write_bundle_with_members(
        anomalous_bundle_path,
        manifest=manifest,
        data=data,
        members=media_members,
    )

    imported = service.import_save(anomalous_bundle_path)
    imported_save_id = _imported_save_id(imported)

    assert repositories.list_media_assets(imported_save_id) == []
    assert (
        repositories.list_context_sources(
            imported_save_id,
            source_type="media_asset",
        )
        == []
    )
    assert [
        link
        for link in repositories.list_entity_links(imported_save_id)
        if link.relation == "visual_context"
    ] == []


def test_import_save_rejects_unreferenced_snapshot_media_payload(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    metadata = _media_file_metadata(data)
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    snapshot_media = dict(_media_asset_by_id(media_assets, MEDIA_ASSET_ID))
    snapshot_media["id"] = "media-unreferenced-snapshot"
    snapshot_media["path"] = "save-night-watch/images/unreferenced.png"
    snapshot_media["thumbnail_path"] = None
    snapshot_media["source_message_id"] = None
    snapshot_media["source_media_asset_id"] = None
    snapshot_media["files"] = {
        "path": {
            "bundle_path": "media/unreferenced.png",
            "sha256": metadata["sha256"],
            "byte_count": len(MEDIA_BYTES),
        }
    }
    snapshot_media_assets = data.setdefault("snapshot_media_assets", [])
    assert isinstance(snapshot_media_assets, list)
    snapshot_media_assets.append(snapshot_media)
    bundle_path = tmp_path / "night-watch-unreferenced-snapshot-media.bragi-chat"
    _write_bundle_with_members(
        bundle_path,
        manifest=manifest,
        data=data,
        members={
            bundle_media_path: MEDIA_BYTES,
            "media/unreferenced.png": MEDIA_BYTES,
        },
    )
    save_ids = [save.id for save in repositories.list_saves()]

    with pytest.raises(module.ChatBundleError, match="unreferenced snapshot media"):
        service.import_save(bundle_path)

    assert [save.id for save in repositories.list_saves()] == save_ids


def test_import_save_assigns_owner_from_import_context(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    owner = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    bundle_path = tmp_path / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(original_save.id, bundle_path)

    imported = service.import_save(bundle_path, owner_user_id=owner.id)

    imported_save = _require_save(repositories, _imported_save_id(imported))
    assert imported_save.owner_user_id == owner.id


def test_import_save_sanitizes_image_style_preset_before_reexport(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(original_save.id, bundle_path)
    _rewrite_bundle_data(
        bundle_path,
        lambda data: _replace_save_app_setting_value(
            data,
            scope="save",
            key=IMAGE_STYLE_PRESET_SETTING,
            value={"preset": "oil_painting", "private_note": "keep me"},
        ),
    )

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    assert (
        repositories.get_app_setting(
            save_image_style_preset_setting_key(imported_save_id)
        )
        == "realistic"
    )
    reexport_path = tmp_path / "reexported.bragi-chat"
    service.export_save(imported_save_id, reexport_path)
    with zipfile.ZipFile(reexport_path) as bundle:
        data = json.loads(bundle.read("data.json"))

    assert _save_app_setting_value(
        data,
        scope="save",
        key=IMAGE_STYLE_PRESET_SETTING,
    ) == '"realistic"'


def test_import_save_sanitizes_post_turn_inference_mode_before_reexport(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=original_save.id,
        key=POST_TURN_INFERENCE_MODE_SETTING,
        value="hybrid",
    )
    bundle_path = tmp_path / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(original_save.id, bundle_path)
    _rewrite_bundle_data(
        bundle_path,
        lambda data: _replace_save_app_setting_value(
            data,
            scope="save",
            key=POST_TURN_INFERENCE_MODE_SETTING,
            value="plan-owned",
        ),
    )

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    assert (
        repositories.get_scoped_setting(
            scope="save",
            scope_id=imported_save_id,
            key=POST_TURN_INFERENCE_MODE_SETTING,
        )
        == POST_TURN_INFERENCE_MODE_DEFAULT
    )
    reexport_path = tmp_path / "reexported.bragi-chat"
    service.export_save(imported_save_id, reexport_path)
    with zipfile.ZipFile(reexport_path) as bundle:
        data = json.loads(bundle.read("data.json"))

    assert _save_app_setting_value(
        data,
        scope="save",
        key=POST_TURN_INFERENCE_MODE_SETTING,
    ) == json.dumps(POST_TURN_INFERENCE_MODE_DEFAULT)


def test_export_omits_message_revision_history_by_default(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    repositories.update_message_body(
        save_id=original_save.id,
        message_id=NARRATOR_MESSAGE_ID,
        body="The lens flashes amber and shows riders in the ash.",
    )
    repositories.add_message_revision(
        revision_id="revision-narrator-correction",
        save_id=original_save.id,
        message_id=NARRATOR_MESSAGE_ID,
        previous_body="The lens flashes red and shows riders in the ash.",
        new_body="The lens flashes amber and shows riders in the ash.",
        diff_unified=(
            "--- previous\n"
            "+++ current\n"
            "@@ -1 +1 @@\n"
            "-The lens flashes red and shows riders in the ash.\n"
            "+The lens flashes amber and shows riders in the ash.\n"
        ),
        reconciliation_status="succeeded",
    )
    bundle_path = tmp_path / "night-watch-revision.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(original_save.id, bundle_path)
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))

    assert data["message_revisions"] == []
    assert "flashes red" not in json.dumps(data)


def test_export_import_preserves_message_revisions_when_explicitly_included(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    repositories.update_message_body(
        save_id=original_save.id,
        message_id=NARRATOR_MESSAGE_ID,
        body="The lens flashes amber and shows riders in the ash.",
    )
    repositories.add_message_revision(
        revision_id="revision-narrator-correction",
        save_id=original_save.id,
        message_id=NARRATOR_MESSAGE_ID,
        previous_body="The lens flashes red and shows riders in the ash.",
        new_body="The lens flashes amber and shows riders in the ash.",
        diff_unified=(
            "--- previous\n"
            "+++ current\n"
            "@@ -1 +1 @@\n"
            "-The lens flashes red and shows riders in the ash.\n"
            "+The lens flashes amber and shows riders in the ash.\n"
        ),
        reconciliation_status="succeeded",
    )
    bundle_path = tmp_path / "night-watch-revision-full.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(
        original_save.id,
        bundle_path,
        include_message_revisions=True,
    )
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_messages = repositories.list_messages(imported_save_id)
    imported_narrator_id = imported_messages[1].id

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
        assert [row["message_id"] for row in data["message_revisions"]] == [
            NARRATOR_MESSAGE_ID
        ]
        assert data["message_revisions"][0]["reconciliation_status"] == "succeeded"

    revisions = repositories.list_message_revisions(
        save_id=imported_save_id,
        message_id=imported_narrator_id,
    )
    assert len(revisions) == 1
    assert revisions[0].id != "revision-narrator-correction"
    assert revisions[0].message_id == imported_narrator_id
    assert revisions[0].previous_body == (
        "The lens flashes red and shows riders in the ash."
    )
    assert revisions[0].new_body == (
        "The lens flashes amber and shows riders in the ash."
    )
    assert revisions[0].reconciliation_status == "succeeded"


def test_export_import_preserves_character_text_revisions_when_explicitly_included(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    text_message = _seed_text_derived_world_rows(repositories, original_save.id)
    repositories.update_character_text_message_body(
        save_id=original_save.id,
        message_id=text_message.id,
        body="I promised I would bring the blue repair notes.",
    )
    repositories.add_character_text_message_revision(
        revision_id="revision-text-correction",
        save_id=original_save.id,
        text_message_id=text_message.id,
        previous_body="I promised I would bring repair notes.",
        new_body="I promised I would bring the blue repair notes.",
        diff_unified=(
            "--- previous\n"
            "+++ current\n"
            "@@ -1 +1 @@\n"
            "-I promised I would bring repair notes.\n"
            "+I promised I would bring the blue repair notes.\n"
        ),
        reconciliation_status="succeeded",
    )
    bundle_path = tmp_path / "night-watch-text-revision-full.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(
        original_save.id,
        bundle_path,
        include_message_revisions=True,
    )
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_text_message = next(
        message
        for message in repositories.list_character_text_messages(
            save_id=imported_save_id,
        )
        if message.body == "I promised I would bring the blue repair notes."
    )

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert [
        row["text_message_id"]
        for row in data["character_text_message_revisions"]
    ] == [text_message.id]

    revisions = repositories.list_character_text_message_revisions(
        save_id=imported_save_id,
        text_message_id=imported_text_message.id,
    )
    assert len(revisions) == 1
    assert revisions[0].id != "revision-text-correction"
    assert revisions[0].text_message_id == imported_text_message.id
    assert revisions[0].previous_body == "I promised I would bring repair notes."
    assert revisions[0].new_body == (
        "I promised I would bring the blue repair notes."
    )
    assert revisions[0].reconciliation_status == "succeeded"


def test_export_import_preserves_video_media_metadata_and_source_asset(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    repositories.create_media_asset(
        asset_id=VIDEO_MEDIA_ASSET_ID,
        save_id=original_save.id,
        source_message_id=NARRATOR_MESSAGE_ID,
        source_media_asset_id=MEDIA_ASSET_ID,
        type="video",
        mime_type="video/mp4",
        path=VIDEO_MEDIA_PATH,
        thumbnail_path=None,
        prompt="animate the red lens warning over the ash road",
        provider="fake-video-provider",
        model="fake-video-model",
        status="succeeded",
        metadata={"duration_seconds": 5, "flow": "image_plus_text_to_video"},
    )
    video_path = media_dir / VIDEO_MEDIA_PATH
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(VIDEO_MEDIA_BYTES)
    bundle_path = tmp_path / "night-watch-video.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(original_save.id, bundle_path)
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
        exported_assets = data["media_assets"]
        assert isinstance(exported_assets, list)
        exported_video = _media_asset_by_id(exported_assets, VIDEO_MEDIA_ASSET_ID)
        assert exported_video["type"] == "video"
        assert exported_video["mime_type"] == "video/mp4"
        assert exported_video["source_media_asset_id"] == MEDIA_ASSET_ID
        assert json.loads(str(exported_video["metadata_json"])) == {
            "duration_seconds": 5,
            "flow": "image_plus_text_to_video",
        }

    imported_assets = repositories.list_media_assets(imported_save_id)
    imported_image = next(asset for asset in imported_assets if asset.type == "image")
    imported_video = next(asset for asset in imported_assets if asset.type == "video")
    assert imported_video.source_message_id is not None
    assert imported_video.source_media_asset_id == imported_image.id
    assert imported_video.mime_type == "video/mp4"
    assert json.loads(imported_video.metadata_json) == {
        "content_rating": "unclassified",
        "duration_seconds": 5,
        "flow": "image_plus_text_to_video",
    }
    assert Path(imported_video.path).parts[0] == imported_save_id
    assert (media_dir / imported_video.path).read_bytes() == VIDEO_MEDIA_BYTES


@pytest.mark.parametrize("mime_type", ["text/html", "image/svg+xml", "video/mp4", None])
def test_import_save_stores_unsupported_media_mime_type_as_inert(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    mime_type: str | None,
) -> None:
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    media_asset = media_assets[0]
    assert isinstance(media_asset, dict)
    media_asset["mime_type"] = mime_type
    broken_bundle_path = tmp_path / "night-watch-active-mime.bragi-chat"
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )

    imported = service.import_save(broken_bundle_path)
    imported_save_id = _imported_save_id(imported)

    [asset] = repositories.list_media_assets(imported_save_id)
    assert asset.mime_type == "application/octet-stream"


@pytest.mark.parametrize("mime_type", ["image/png", "image/jpeg", "image/webp"])
def test_import_save_preserves_supported_image_mime_types(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    mime_type: str,
) -> None:
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    media_asset = media_assets[0]
    assert isinstance(media_asset, dict)
    media_asset["mime_type"] = mime_type
    bundle_path = tmp_path / "night-watch-image-mime.bragi-chat"
    _write_bundle_with_member(
        bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    [asset] = repositories.list_media_assets(imported_save_id)
    assert asset.mime_type == mime_type


@pytest.mark.parametrize("mime_type", ["video/mp4", "video/webm"])
def test_import_save_preserves_supported_video_mime_types(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    mime_type: str,
) -> None:
    service, manifest, data, media_members = _export_video_bundle_payloads(
        repositories,
        tmp_path,
    )
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    video_asset = _media_asset_by_id(media_assets, VIDEO_MEDIA_ASSET_ID)
    video_asset["mime_type"] = mime_type
    bundle_path = tmp_path / "night-watch-video-mime.bragi-chat"
    _write_bundle_with_members(
        bundle_path,
        manifest=manifest,
        data=data,
        members=media_members,
    )

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_video = next(
        asset
        for asset in repositories.list_media_assets(imported_save_id)
        if asset.type == "video"
    )
    assert imported_video.mime_type == mime_type


def test_export_import_remaps_character_reference_media_link(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    repositories.add_character(
        save_id=original_save.id,
        name="Mara",
        role="Signal warden",
        character_id="character-mara",
    )
    repositories.add_entity_link(
        save_id=original_save.id,
        entity_type="character",
        entity_id="character-mara",
        target_type="media_asset",
        target_id=MEDIA_ASSET_ID,
        relation="reference_image",
    )
    bundle_path = tmp_path / "night-watch-reference.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(original_save.id, bundle_path)
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
        exported_character = next(
            row for row in data["characters"] if row["id"] == "character-mara"
        )
        assert exported_character["reference_image_asset_id"] == MEDIA_ASSET_ID
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_assets = repositories.list_media_assets(imported_save_id)
    imported_characters = repositories.list_characters(imported_save_id)
    imported_links = [
        link
        for link in repositories.list_entity_links(imported_save_id)
        if link.relation == "reference_image"
    ]
    assert len(imported_links) == 1
    assert imported_links[0].entity_type == "character"
    assert imported_links[0].entity_id == imported_characters[0].id
    assert imported_links[0].target_type == "media_asset"
    assert imported_links[0].target_id == imported_assets[0].id


def test_export_import_remaps_save_level_character_reference_media_link(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    repositories.add_entity_link(
        save_id=original_save.id,
        entity_type="save",
        entity_id=original_save.id,
        target_type="media_asset",
        target_id=MEDIA_ASSET_ID,
        relation="reference_image",
    )
    bundle_path = tmp_path / "night-watch-save-reference.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(original_save.id, bundle_path)
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_assets = repositories.list_media_assets(imported_save_id)
    imported_links = [
        link
        for link in repositories.list_entity_links(imported_save_id)
        if link.relation == "reference_image"
    ]
    assert len(imported_links) == 1
    assert imported_links[0].entity_type == "save"
    assert imported_links[0].entity_id == imported_save_id
    assert imported_links[0].target_type == "media_asset"
    assert imported_links[0].target_id == imported_assets[0].id


def test_export_import_round_trips_character_contact_name(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    mara = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal warden",
        contact_name="Mar",
        character_id="character-mara",
    )
    bundle_path = tmp_path / "night-watch-contact-name.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    exported_mara = next(row for row in data["characters"] if row["id"] == mara.id)
    assert exported_mara["contact_name"] == "Mar"

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_characters = repositories.list_characters(imported_save_id)

    assert len(imported_characters) == 1
    assert imported_characters[0].name == "Mara"
    assert imported_characters[0].contact_name == "Mar"


def test_export_import_round_trips_character_texting_style(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    mara = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal warden",
        texting_style="Answers in short sentence fragments, late but warmly.",
        character_id="character-mara",
    )
    bundle_path = tmp_path / "night-watch-texting-style.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    exported_mara = next(row for row in data["characters"] if row["id"] == mara.id)
    assert exported_mara["texting_style"] == (
        "Answers in short sentence fragments, late but warmly."
    )

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)
    imported_characters = repositories.list_characters(imported_save_id)

    assert len(imported_characters) == 1
    assert imported_characters[0].name == "Mara"
    assert imported_characters[0].texting_style == (
        "Answers in short sentence fragments, late but warmly."
    )


def test_import_legacy_save_bundle_without_character_texting_style_defaults_blank(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal warden",
        texting_style="Answers in short sentence fragments, late but warmly.",
        character_id="character-mara",
    )
    bundle_path = tmp_path / "night-watch-legacy-texting-style.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    def remove_texting_style(data: dict[str, object]) -> None:
        characters = data["characters"]
        assert isinstance(characters, list)
        for row in characters:
            assert isinstance(row, dict)
            row.pop("texting_style", None)

    _rewrite_bundle_data(bundle_path, remove_texting_style)

    imported = service.import_save(bundle_path)
    imported_characters = repositories.list_characters(_imported_save_id(imported))

    assert len(imported_characters) == 1
    assert imported_characters[0].texting_style == ""





def test_export_import_preserves_uploaded_character_reference_metadata_and_files(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    character = repositories.add_character(
        save_id=original_save.id,
        name="Mara",
        role="Signal warden",
        character_id="character-mara",
    )
    reference = repositories.create_media_asset(
        asset_id="media-uploaded-reference",
        save_id=original_save.id,
        source_message_id=None,
        type="image",
        mime_type="image/webp",
        path="save-night-watch/uploads/reference.webp",
        thumbnail_path="save-night-watch/uploads/thumbnails/reference.webp",
        prompt="Uploaded character reference image",
        provider="local",
        model="upload",
        status="succeeded",
        metadata={
            "kind": "character_reference",
            "source": "uploaded",
            "character_id": character.id,
        },
    )
    (media_dir / reference.path).parent.mkdir(parents=True, exist_ok=True)
    (media_dir / reference.path).write_bytes(b"RIFF----WEBPuploaded-reference")
    assert reference.thumbnail_path is not None
    (media_dir / reference.thumbnail_path).parent.mkdir(parents=True, exist_ok=True)
    (media_dir / reference.thumbnail_path).write_bytes(b"thumbnail")
    repositories.add_entity_link(
        save_id=original_save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="media_asset",
        target_id=reference.id,
        relation="reference_image",
    )
    bundle_path = tmp_path / "night-watch-uploaded-reference.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(original_save.id, bundle_path)
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_reference = next(
        asset
        for asset in repositories.list_media_assets(imported_save_id)
        if json.loads(asset.metadata_json).get("source") == "uploaded"
    )
    imported_character = repositories.list_characters(imported_save_id)[0]
    assert imported_reference.source_message_id is None
    assert imported_reference.provider == "local"
    assert imported_reference.model == "upload"
    assert imported_reference.mime_type == "image/webp"
    imported_metadata = json.loads(imported_reference.metadata_json)
    assert imported_metadata["kind"] == "character_reference"
    assert imported_metadata["source"] == "uploaded"
    assert imported_metadata["character_id"] == imported_character.id
    assert (media_dir / imported_reference.path).read_bytes() == (
        b"RIFF----WEBPuploaded-reference"
    )
    assert imported_reference.thumbnail_path is not None
    assert (media_dir / imported_reference.thumbnail_path).read_bytes() == b"thumbnail"
    imported_links = [
        link
        for link in repositories.list_entity_links(imported_save_id)
        if link.relation == "reference_image"
    ]
    assert imported_links[0].target_id == imported_reference.id


def test_export_save_omits_archived_media_assets(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    repositories.archive_media_asset(
        save_id=original_save.id,
        media_asset_id=MEDIA_ASSET_ID,
    )
    bundle_path = tmp_path / "night-watch-archived-media.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(original_save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))

    assert manifest["counts"]["media_assets"] == 0
    assert data["media_assets"] == []


def test_export_import_clears_archived_source_media_asset_reference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    derived_asset_id = "media-regenerated-beacon"
    derived_media_path = "save-night-watch/images/regenerated-beacon.png"
    derived_media_bytes = b"regenerated beacon image bytes"
    repositories.create_media_asset(
        asset_id=derived_asset_id,
        save_id=original_save.id,
        source_message_id=NARRATOR_MESSAGE_ID,
        source_media_asset_id=MEDIA_ASSET_ID,
        type="image",
        path=derived_media_path,
        thumbnail_path=None,
        prompt="regenerated red beacon lens",
        provider="fake-image-provider",
        model="fake-image-model",
        status="succeeded",
        metadata={
            "source_media_asset_id": MEDIA_ASSET_ID,
            "source_media_asset_ids": [MEDIA_ASSET_ID],
            "source_character_reference_asset_id": MEDIA_ASSET_ID,
            "source_character_reference_asset_ids": [MEDIA_ASSET_ID],
        },
    )
    derived_path = media_dir / derived_media_path
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    derived_path.write_bytes(derived_media_bytes)
    repositories.archive_media_asset_only(
        save_id=original_save.id,
        media_asset_id=MEDIA_ASSET_ID,
    )
    bundle_path = tmp_path / "night-watch-archived-source-media.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(original_save.id, bundle_path)
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))

    exported_assets = data["media_assets"]
    assert isinstance(exported_assets, list)
    assert [asset["id"] for asset in exported_assets if isinstance(asset, dict)] == [
        derived_asset_id
    ]
    exported_derived = _media_asset_by_id(exported_assets, derived_asset_id)
    assert exported_derived["source_media_asset_id"] is None
    exported_metadata = json.loads(str(exported_derived["metadata_json"]))
    assert exported_metadata["source_media_asset_id"] is None
    assert exported_metadata["source_media_asset_ids"] == []
    assert exported_metadata["source_character_reference_asset_id"] is None
    assert exported_metadata["source_character_reference_asset_ids"] == []
    imported_assets = repositories.list_media_assets(imported_save_id)
    assert len(imported_assets) == 1
    assert imported_assets[0].source_media_asset_id is None
    imported_metadata = json.loads(imported_assets[0].metadata_json)
    assert imported_metadata["source_media_asset_id"] is None
    assert imported_metadata["source_media_asset_ids"] == []
    assert imported_metadata["source_character_reference_asset_id"] is None
    assert imported_metadata["source_character_reference_asset_ids"] == []
    assert (media_dir / imported_assets[0].path).read_bytes() == derived_media_bytes


def test_import_save_preserves_source_media_asset_when_video_row_precedes_image(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    service, manifest, data, media_members = _export_video_bundle_payloads(
        repositories,
        tmp_path,
    )
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    data["media_assets"] = sorted(
        media_assets,
        key=lambda row: 0
        if isinstance(row, dict) and row.get("id") == VIDEO_MEDIA_ASSET_ID
        else 1,
    )
    reordered_bundle_path = tmp_path / "night-watch-video-reordered.bragi-chat"
    _write_bundle_with_members(
        reordered_bundle_path,
        manifest=manifest,
        data=data,
        members=media_members,
    )

    imported = service.import_save(reordered_bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_assets = repositories.list_media_assets(imported_save_id)
    imported_image = next(asset for asset in imported_assets if asset.type == "image")
    imported_video = next(asset for asset in imported_assets if asset.type == "video")
    assert imported_video.source_media_asset_id == imported_image.id


def test_import_save_repairs_source_media_asset_when_source_is_snapshot_only(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    service, manifest, data, media_members = _export_video_bundle_payloads(
        repositories,
        tmp_path,
    )
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    exported_video = _media_asset_by_id(media_assets, VIDEO_MEDIA_ASSET_ID)
    exported_video["metadata_json"] = json.dumps(
        {
            "duration_seconds": 5,
            "source_media_asset_id": MEDIA_ASSET_ID,
            "source_media_asset_ids": [MEDIA_ASSET_ID],
            "source_character_reference_asset_id": MEDIA_ASSET_ID,
            "source_character_reference_asset_ids": [MEDIA_ASSET_ID],
        }
    )
    _move_media_asset_to_snapshot_only(
        manifest,
        data,
        media_asset_id=MEDIA_ASSET_ID,
    )
    anomalous_bundle_path = tmp_path / "night-watch-video-snapshot-source.bragi-chat"
    _write_bundle_with_members(
        anomalous_bundle_path,
        manifest=manifest,
        data=data,
        members=media_members,
    )

    imported = service.import_save(anomalous_bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_assets = repositories.list_media_assets(imported_save_id)
    assert len(imported_assets) == 1
    imported_video = imported_assets[0]
    assert imported_video.type == "video"
    assert imported_video.source_media_asset_id is None
    assert (tmp_path / "media" / imported_video.path).read_bytes() == VIDEO_MEDIA_BYTES
    imported_metadata = json.loads(imported_video.metadata_json)
    assert imported_metadata["duration_seconds"] == 5
    assert imported_metadata["source_media_asset_id"] is None
    assert imported_metadata["source_media_asset_ids"] == []
    assert imported_metadata["source_character_reference_asset_id"] is None
    assert imported_metadata["source_character_reference_asset_ids"] == []


def test_import_save_remaps_media_source_metadata(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    service, manifest, data, media_members = _export_video_bundle_payloads(
        repositories,
        tmp_path,
    )
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    exported_video = _media_asset_by_id(media_assets, VIDEO_MEDIA_ASSET_ID)
    exported_video["metadata_json"] = json.dumps(
        {
            "duration_seconds": 5,
            "source_media_asset_id": MEDIA_ASSET_ID,
            "source_media_asset_ids": [MEDIA_ASSET_ID, "media-missing-source"],
            "source_character_reference_asset_id": MEDIA_ASSET_ID,
            "source_character_reference_asset_ids": [
                MEDIA_ASSET_ID,
                "media-missing-reference",
            ],
        }
    )
    bundle_path = tmp_path / "night-watch-video-source-metadata.bragi-chat"
    _write_bundle_with_members(
        bundle_path,
        manifest=manifest,
        data=data,
        members=media_members,
    )

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_assets = repositories.list_media_assets(imported_save_id)
    imported_image = next(asset for asset in imported_assets if asset.type == "image")
    imported_video = next(asset for asset in imported_assets if asset.type == "video")
    imported_metadata = json.loads(imported_video.metadata_json)
    assert imported_video.source_media_asset_id == imported_image.id
    assert imported_metadata["duration_seconds"] == 5
    assert imported_metadata["source_media_asset_id"] == imported_image.id
    assert imported_metadata["source_media_asset_ids"] == [imported_image.id]
    assert imported_metadata["source_character_reference_asset_id"] == imported_image.id
    assert imported_metadata["source_character_reference_asset_ids"] == [
        imported_image.id
    ]


def test_import_save_repairs_unknown_source_media_asset(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, media_members = _export_video_bundle_payloads(
        repositories,
        tmp_path,
    )
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    exported_video = _media_asset_by_id(media_assets, VIDEO_MEDIA_ASSET_ID)
    exported_video["source_media_asset_id"] = "media-missing-source"
    broken_bundle_path = tmp_path / "night-watch-video-missing-source.bragi-chat"
    _write_bundle_with_members(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        members=media_members,
    )
    events: list[tuple[str, dict[str, object]]] = []

    def capture_log_event(event: str, /, **fields: object) -> None:
        events.append((event, fields))

    monkeypatch.setattr(module, "log_event", capture_log_event)

    imported = service.import_save(broken_bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_assets = repositories.list_media_assets(imported_save_id)
    imported_video = next(asset for asset in imported_assets if asset.type == "video")
    assert imported_video.source_media_asset_id is None
    repair_events = [
        fields for event, fields in events if event == "chat_bundle.import_repaired"
    ]
    assert repair_events == [
        {
            "save_id": imported_save_id,
            "repaired_reference_count": 1,
            "repaired_fields": {"media_assets.source_media_asset_id": 1},
        }
    ]
    assert all("red beacon" not in str(fields) for _event, fields in events)
    assert all(MEDIA_PATH not in str(fields) for _event, fields in events)


def test_export_import_preserves_loss_conditions_changes_and_outcomes(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_loss_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "night-watch-loss.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    manifest = service.export_save(original_save.id, bundle_path)
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    assert manifest.message_count == 3
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert [row["id"] for row in data["save_loss_conditions"]] == [
        LOSS_CONDITION_ID
    ]
    assert [row["id"] for row in data["save_loss_condition_changes"]] == [
        LOSS_CONDITION_CHANGE_ID
    ]
    assert [row["id"] for row in data["save_loss_outcomes"]] == [LOSS_OUTCOME_ID]
    assert data["save_loss_outcomes"][0]["epilogue_message_id"] == (
        LOSS_EPILOGUE_MESSAGE_ID
    )
    assert data["save_loss_outcomes"][0]["outcome_type"] == "loss_condition"

    imported_messages = repositories.list_messages(imported_save_id)
    imported_message_by_body = {
        message.body: message.id for message in imported_messages
    }
    message_id_map = {
        PLAYER_MESSAGE_ID: imported_message_by_body[
            "I climb toward the beacon lens."
        ],
        NARRATOR_MESSAGE_ID: imported_message_by_body[
            "The lens flashes red and shows riders in the ash."
        ],
        LOSS_EPILOGUE_MESSAGE_ID: imported_message_by_body[
            "The beacon goes dark, and Mara's watch ends in ash."
        ],
    }
    imported_conditions = repositories.list_loss_conditions(imported_save_id)
    imported_changes = repositories.list_loss_condition_changes(imported_save_id)
    imported_outcomes = repositories.list_loss_outcomes(imported_save_id)

    assert len(imported_conditions) == 1
    assert imported_conditions[0].id != LOSS_CONDITION_ID
    assert imported_conditions[0].name == "Beacon collapse"
    assert imported_conditions[0].status == "triggered"

    assert len(imported_changes) == 1
    assert imported_changes[0].id != LOSS_CONDITION_CHANGE_ID
    assert imported_changes[0].condition_id == imported_conditions[0].id
    assert imported_changes[0].source_message_id == (
        message_id_map[NARRATOR_MESSAGE_ID]
    )

    assert len(imported_outcomes) == 1
    assert imported_outcomes[0].id != LOSS_OUTCOME_ID
    assert imported_outcomes[0].condition_id == imported_conditions[0].id
    assert imported_outcomes[0].outcome_type == "loss_condition"
    assert imported_outcomes[0].triggering_message_id == (
        message_id_map[NARRATOR_MESSAGE_ID]
    )
    assert imported_outcomes[0].epilogue_message_id == (
        message_id_map[LOSS_EPILOGUE_MESSAGE_ID]
    )
    assert imported_outcomes[0].evidence == {
        "items": [
            {
                "source_message_id": message_id_map[NARRATOR_MESSAGE_ID],
                "quote": "riders in the ash",
            }
        ]
    }


def test_export_import_preserves_conditionless_terminal_outcomes(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_save = _seed_bundle_save(repositories, media_dir)
    repositories.create_loss_outcome(
        save_id=original_save.id,
        condition_id=None,
        condition_name="Mission complete",
        triggering_message_id=NARRATOR_MESSAGE_ID,
        explanation="Mara dies sealing the gate and the mission is complete.",
        confidence=0.96,
        evidence={
            "items": [
                {
                    "source_message_id": NARRATOR_MESSAGE_ID,
                    "quote": "riders in the ash",
                }
            ],
            "epilogue": "The gate holds.",
        },
        provider="fake-loss-provider",
        model="fake-loss-model",
        outcome_type="player_dead",
    )
    bundle_path = tmp_path / "night-watch-terminal.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(original_save.id, bundle_path)
    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["save_loss_conditions"] == []
    assert data["save_loss_outcomes"][0]["condition_id"] is None
    assert data["save_loss_outcomes"][0]["outcome_type"] == "player_dead"

    imported_messages = repositories.list_messages(imported_save_id)
    imported_message_by_body = {
        message.body: message.id for message in imported_messages
    }
    message_id_map = {
        PLAYER_MESSAGE_ID: imported_message_by_body[
            "I climb toward the beacon lens."
        ],
        NARRATOR_MESSAGE_ID: imported_message_by_body[
            "The lens flashes red and shows riders in the ash."
        ],
    }
    imported_outcomes = repositories.list_loss_outcomes(imported_save_id)

    assert repositories.list_loss_conditions(imported_save_id) == []
    assert len(imported_outcomes) == 1
    assert imported_outcomes[0].condition_id is None
    assert imported_outcomes[0].condition_name == "Mission complete"
    assert imported_outcomes[0].outcome_type == "player_dead"
    assert imported_outcomes[0].triggering_message_id == (
        message_id_map[NARRATOR_MESSAGE_ID]
    )
    assert imported_outcomes[0].evidence == {
        "items": [
            {
                "source_message_id": message_id_map[NARRATOR_MESSAGE_ID],
                "quote": "riders in the ash",
            }
        ],
        "epilogue": "The gate holds.",
    }


def test_import_save_removes_partial_media_file_after_write_failure(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    module = _chat_bundle_module()
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)
    save_ids = [listed_save.id for listed_save in repositories.list_saves()]
    partial_paths: list[Path] = []

    def fail_after_partial_copy(
        _bundle_path: Path,
        _member: object,
        path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial media")
        assert path.exists()
        assert path.read_bytes() == b"partial media"
        partial_paths.append(path)
        raise OSError("simulated media write failure")

    monkeypatch.setattr(module, "_copy_bundle_media_member", fail_after_partial_copy)

    with pytest.raises(OSError, match="simulated media write failure"):
        service.import_save(bundle_path)

    assert [listed_save.id for listed_save in repositories.list_saves()] == save_ids
    assert partial_paths
    assert all(not path.exists() for path in partial_paths)


def test_export_save_rejects_missing_primary_media_file_without_writing_bundle(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir, write_media_file=False)
    bundle_path = tmp_path / "exports" / "night-watch-missing-media.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    with pytest.raises(module.ChatBundleError, match="Missing media file"):
        service.export_save(save.id, bundle_path)

    assert not bundle_path.exists()


def test_export_filters_media_assets_for_archived_messages(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    save_level_asset_id = "media-save-level-map"
    save_level_media_path = "save-night-watch/images/save-map.png"
    save_level_media_bytes = b"save level map bytes"
    deleted_message_id = "message-deleted-media-source"
    deleted_media_path = "save-night-watch/images/deleted-message.png"
    repositories.append_message(
        message_id=deleted_message_id,
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The deleted vision should not export its image.",
    )
    repositories.create_media_asset(
        asset_id=save_level_asset_id,
        save_id=save.id,
        source_message_id=None,
        type="image",
        path=save_level_media_path,
        thumbnail_path=None,
        prompt="save level campaign map",
        provider="fake-image-provider",
        model="fake-image-model",
        status="succeeded",
    )
    repositories.create_media_asset(
        asset_id="media-deleted-message",
        save_id=save.id,
        source_message_id=deleted_message_id,
        type="image",
        path=deleted_media_path,
        thumbnail_path=None,
        prompt="deleted message vision",
        provider="fake-image-provider",
        model="fake-image-model",
        status="succeeded",
    )
    save_level_path = media_dir / save_level_media_path
    save_level_path.parent.mkdir(parents=True, exist_ok=True)
    save_level_path.write_bytes(save_level_media_bytes)
    deleted_path = media_dir / deleted_media_path
    deleted_path.parent.mkdir(parents=True, exist_ok=True)
    deleted_path.write_bytes(b"deleted message image bytes")
    repositories.archive_message(deleted_message_id)
    bundle_path = tmp_path / "exports" / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    manifest = service.export_save(save.id, bundle_path)

    assert manifest.media_count == 2
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
        exported_assets = data["media_assets"]
        assert isinstance(exported_assets, list)
        assert {
            (asset["id"], asset["source_message_id"])
            for asset in exported_assets
            if isinstance(asset, dict)
        } == {
            (MEDIA_ASSET_ID, NARRATOR_MESSAGE_ID),
            (save_level_asset_id, None),
        }
        media_names = [
            name for name in bundle.namelist() if name.startswith("media/")
        ]
        assert len(media_names) == 2
        assert {bundle.read(name) for name in media_names} == {
            MEDIA_BYTES,
            save_level_media_bytes,
        }


def test_export_filters_state_changes_for_archived_messages(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    deleted_message_id = "message-deleted-1"
    repositories.append_message(
        message_id=deleted_message_id,
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I whisper to a lens nobody will remember.",
    )
    repositories.add_state_change(
        change_id="state-change-deleted-message",
        save_id=save.id,
        operation="upsert",
        state_key="forgotten_lens",
        before_json=None,
        after_json=json.dumps({"forgotten": True}, sort_keys=True),
        source_message_id=deleted_message_id,
    )
    repositories.archive_message(deleted_message_id)
    bundle_path = tmp_path / "exports" / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))

    assert [message["id"] for message in data["messages"]] == [
        PLAYER_MESSAGE_ID,
        NARRATOR_MESSAGE_ID,
    ]
    assert [
        (
            state_change["id"],
            state_change["state_key"],
            state_change["source_message_id"],
        )
        for state_change in data["state_changes"]
    ] == [
        (
            "state-change-beacon-lens",
            "beacon_lens",
            NARRATOR_MESSAGE_ID,
        )
    ]


def test_export_import_filters_side_table_refs_for_archived_messages(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    _seed_context_graph_rows(repositories, save.id)
    deleted_message_id = "message-deleted-side-table-source"
    repositories.append_message(
        message_id=deleted_message_id,
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="This message will be archived before export.",
    )
    repositories.add_summary(
        summary_id="summary-deleted-endpoint",
        save_id=save.id,
        covers_message_start_id=PLAYER_MESSAGE_ID,
        covers_message_end_id=deleted_message_id,
        body="This summary should not survive export.",
        provider="fake-chat-provider",
        model="fake-chat-model",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id="character-mara",
        target_type="world_state",
        target_id="world-state-beacon-lens",
        knowledge_state="knows",
        acquisition_method="witnessed",
        source_message_id=deleted_message_id,
        source_message_ids=[deleted_message_id],
        edge_id="edge-deleted-source-message",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id="character-mara",
        target_type="summary",
        target_id="summary-deleted-endpoint",
        knowledge_state="knows",
        acquisition_method="witnessed",
        source_message_id=NARRATOR_MESSAGE_ID,
        source_message_ids=[NARRATOR_MESSAGE_ID],
        edge_id="edge-deleted-target-summary",
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=deleted_message_id,
        character_id="character-mara",
        visibility="visible",
        source="scene_presence",
        visibility_id="visibility-deleted-message",
    )
    repositories.replace_message_scene_presence(
        save.id,
        deleted_message_id,
        ["character-mara"],
        source="context_snapshot",
    )
    repositories.upsert_world_state(
        state_id="world-state-deleted-source",
        save_id=save.id,
        key="deleted_signal",
        value={"status": "gone"},
        category="location",
        confidence=0.7,
        source_message_id=deleted_message_id,
    )
    repositories.archive_message(deleted_message_id)
    bundle_path = tmp_path / "exports" / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)
    imported = service.import_save(bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))

    assert [row["id"] for row in data["summaries"]] == ["summary-beacon-warning"]
    assert [row["key"] for row in data["world_state"]] == ["beacon_lens"]
    assert [row["id"] for row in data["character_knowledge_edges"]] == [
        "edge-mara-signal-code"
    ]
    assert [row["id"] for row in data["message_visibility"]] == [
        "visibility-mara-narrator"
    ]
    assert [row["id"] for row in data["message_scene_presence"]] == [
        "presence-mara-narrator"
    ]

    imported_save_id = _imported_save_id(imported)
    imported_message_ids = {
        message.id for message in repositories.list_messages(imported_save_id)
    }
    imported_summaries = repositories.list_summaries(imported_save_id)
    assert [summary.body for summary in imported_summaries] == [
        "Mara reaches the beacon and sees a warning."
    ]
    assert all(
        summary.covers_message_start_id in imported_message_ids
        and summary.covers_message_end_id in imported_message_ids
        for summary in imported_summaries
    )
    assert [
        (state.key, state.source_message_id in imported_message_ids)
        for state in repositories.list_world_state(imported_save_id)
    ] == [("beacon_lens", True)]


def test_export_save_rejects_oversized_media_file_without_writing_bundle(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    module = _chat_bundle_module()
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "exports" / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    monkeypatch.setattr(module, "_MAX_BUNDLE_MEDIA_FILE_BYTES", len(MEDIA_BYTES) - 1)

    with pytest.raises(module.ChatBundleError, match="too large"):
        service.export_save(save.id, bundle_path)

    assert not bundle_path.exists()


@pytest.mark.parametrize(
    ("race_action", "message"),
    [
        ("delete", "disappeared"),
        ("rewrite", "changed"),
    ],
)
def test_export_save_rejects_media_race_without_writing_bundle(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    race_action: str,
    message: str,
) -> None:
    module = _chat_bundle_module()
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "exports" / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)
    original_bundle_bytes = bundle_path.read_bytes()
    original_annotate_media_files = module._annotate_export_media_asset_files

    def annotate_media_files_with_race(rows: Any, media_dir_arg: Path) -> None:
        original_annotate_media_files(rows, media_dir_arg)
        media_path = media_dir / MEDIA_PATH
        if race_action == "delete":
            media_path.unlink()
        elif race_action == "rewrite":
            changed_media_bytes = b"x" * len(MEDIA_BYTES)
            assert changed_media_bytes != MEDIA_BYTES
            media_path.write_bytes(changed_media_bytes)
        else:
            raise AssertionError(f"Unhandled race action: {race_action}")

    monkeypatch.setattr(
        module,
        "_annotate_export_media_asset_files",
        annotate_media_files_with_race,
    )

    with pytest.raises(module.ChatBundleError, match=message):
        service.export_save(save.id, bundle_path)

    assert bundle_path.read_bytes() == original_bundle_bytes


def test_import_save_rejects_declared_missing_media_member_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "night-watch.bragi-chat"
    broken_bundle_path = tmp_path / "night-watch-broken.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))

    metadata = _media_file_metadata(data)
    bundle_media_path = metadata["bundle_path"]
    assert isinstance(bundle_media_path, str)
    assert bundle_media_path
    _write_bundle(broken_bundle_path, manifest=manifest, data=data)
    save_count = len(repositories.list_saves())

    with pytest.raises(module.ChatBundleError):
        service.import_save(broken_bundle_path)

    assert len(repositories.list_saves()) == save_count


def test_import_save_rejects_media_asset_without_primary_file_payload(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, _bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    media_asset = media_assets[0]
    assert isinstance(media_asset, dict)
    files = media_asset["files"]
    assert isinstance(files, dict)
    files.pop("path")
    broken_bundle_path = tmp_path / "night-watch-no-primary-media.bragi-chat"
    _write_bundle(broken_bundle_path, manifest=manifest, data=data)
    save_count = len(repositories.list_saves())

    with pytest.raises(module.ChatBundleError, match="primary media"):
        service.import_save(broken_bundle_path)

    assert len(repositories.list_saves()) == save_count


def test_import_save_repairs_media_source_message_outside_bundle(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    media_asset = media_assets[0]
    assert isinstance(media_asset, dict)
    media_asset["source_message_id"] = "message-not-exported"
    broken_bundle_path = tmp_path / "night-watch-missing-media-source.bragi-chat"
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )

    imported = service.import_save(broken_bundle_path)
    imported_save_id = _imported_save_id(imported)

    [imported_media] = repositories.list_media_assets(imported_save_id)
    assert imported_media.source_message_id is None


@pytest.mark.parametrize("role", ["system", "developer"])
def test_import_save_rejects_unsupported_message_role_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    role: str,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    messages = data["messages"]
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, dict)
    message["role"] = role
    broken_bundle_path = tmp_path / f"night-watch-unsupported-{role}-role.bragi-chat"
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    save_ids = [save.id for save in repositories.list_saves()]

    with pytest.raises(module.ChatBundleError, match="Unsupported message role"):
        service.import_save(broken_bundle_path)

    assert [save.id for save in repositories.list_saves()] == save_ids


@pytest.mark.parametrize(
    "collection_name",
    [
        "world_state",
        "memories",
        "state_changes",
        "save_scenario_updates",
    ],
)
def test_import_save_repairs_unknown_source_message_references(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    collection_name: str,
) -> None:
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    records = data[collection_name]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    record["source_message_id"] = "message-not-exported"
    broken_bundle_path = (
        tmp_path / f"night-watch-unknown-{collection_name}-source.bragi-chat"
    )
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )

    imported = service.import_save(broken_bundle_path)
    imported_save_id = _imported_save_id(imported)

    if collection_name == "world_state":
        [record] = repositories.list_world_state(imported_save_id)
        assert record.source_message_id is None
    elif collection_name == "memories":
        [record] = repositories.list_memories(imported_save_id)
        assert record.source_message_id is None
    elif collection_name == "state_changes":
        [record] = repositories.list_state_changes(imported_save_id)
        assert record.source_message_id is None
    elif collection_name == "save_scenario_updates":
        [record] = repositories.list_save_scenario_updates(imported_save_id)
        assert record.source_message_id is None
    else:
        raise AssertionError(f"Unhandled collection: {collection_name}")


def test_import_save_repairs_unknown_scenario_update_sources(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    scenario_updates = data["save_scenario_updates"]
    assert isinstance(scenario_updates, list)
    scenario_update = scenario_updates[0]
    assert isinstance(scenario_update, dict)
    scenario_update["source_message_ids_json"] = json.dumps(
        [PLAYER_MESSAGE_ID, "message-not-exported"],
    )
    broken_bundle_path = (
        tmp_path / "night-watch-unknown-scenario-update-sources.bragi-chat"
    )
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )

    imported = service.import_save(broken_bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_messages = repositories.list_messages(imported_save_id)
    [scenario_update] = repositories.list_save_scenario_updates(imported_save_id)
    assert json.loads(scenario_update.source_message_ids_json) == [
        imported_messages[0].id,
        imported_messages[1].id,
    ]


def test_import_save_strips_deprecated_scenario_update_sections(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    scenario_updates = data["save_scenario_updates"]
    assert isinstance(scenario_updates, list)
    [scenario_update] = scenario_updates
    assert isinstance(scenario_update, dict)
    scenario_update["content_json"] = json.dumps(
        _legacy_character_list_update_content(),
    )
    bundle_path = tmp_path / "night-watch-legacy-scenario-update.bragi-chat"
    _write_bundle_with_member(
        bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    [imported_update] = repositories.list_save_scenario_updates(imported_save_id)
    assert json.loads(imported_update.content_json) == {
        **_cleaned_character_list_update_content(),
        "_source": {"content_rating": "unclassified"},
    }


def test_import_save_repairs_context_update_suggestion_proposed_sources(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    suggestions = data["context_update_suggestions"]
    assert isinstance(suggestions, list)
    suggestions.append(
        {
            "id": "suggestion-create-memory",
            "save_id": SAVE_ID,
            "update_type": "create",
            "entity_type": "memory",
            "entity_id": None,
            "field_path": "*",
            "proposed_value_json": json.dumps(
                {
                    "body": "Mara remembers the cracked beacon lens.",
                    "tags": ["mara", "beacon"],
                    "importance": 0.72,
                    "source_message_id": "message-not-exported",
                    "source_message_ids": [
                        PLAYER_MESSAGE_ID,
                        NARRATOR_MESSAGE_ID,
                        "message-not-exported",
                    ],
                    "source_observation_id": OBSERVATION_ID,
                    "source_observation_ids": [
                        OBSERVATION_ID,
                        "observation-not-exported",
                    ],
                }
            ),
            "status": "pending",
            "reason": "Imported pending memory suggestion.",
            "confidence": 0.72,
            "source_message_ids_json": json.dumps(
                [PLAYER_MESSAGE_ID, "message-not-exported"]
            ),
            "created_at": "2026-07-01T12:00:00+00:00",
            "resolved_at": None,
        }
    )
    broken_bundle_path = (
        tmp_path / "night-watch-suggestion-proposed-sources.bragi-chat"
    )
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )

    imported = service.import_save(broken_bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_messages = repositories.list_messages(imported_save_id)
    suggestion = next(
        suggestion
        for suggestion in repositories.list_context_update_suggestions(
            imported_save_id
        )
        if suggestion.reason == "Imported pending memory suggestion."
    )
    assert suggestion.source_message_ids == [imported_messages[0].id]
    assert isinstance(suggestion.proposed_value, dict)
    assert suggestion.proposed_value["source_message_id"] is None
    assert suggestion.proposed_value["source_message_ids"] == [
        imported_messages[0].id,
        imported_messages[1].id,
    ]
    [imported_observation] = repositories.list_context_observations(
        imported_save_id
    )
    assert suggestion.proposed_value["source_observation_id"] == (
        imported_observation.id
    )
    assert suggestion.proposed_value["source_observation_ids"] == [
        imported_observation.id
    ]


def test_import_save_remaps_context_update_suggestion_proposed_location_id(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    location = repositories.add_location(
        save_id=save.id,
        location_id="location-character-suggestion",
        name="Lower Arcade",
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="create",
        entity_type="character",
        field_path="*",
        proposed_value={
            "name": "Iris",
            "role": "scout",
            "location_id": location.id,
            "source_message_id": PLAYER_MESSAGE_ID,
        },
        reason="Imported pending character suggestion.",
        confidence=0.74,
        source_message_ids=[PLAYER_MESSAGE_ID],
        suggestion_id="suggestion-create-iris",
    )
    bundle_path = (
        tmp_path / "exports" / "night-watch-suggestion-location.bragi-chat"
    )
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_messages = repositories.list_messages(imported_save_id)
    imported_location = next(
        location
        for location in repositories.list_locations(imported_save_id)
        if location.name == "Lower Arcade"
    )
    suggestion = next(
        suggestion
        for suggestion in repositories.list_context_update_suggestions(
            imported_save_id
        )
        if suggestion.reason == "Imported pending character suggestion."
    )
    assert suggestion.source_message_ids == [imported_messages[0].id]
    assert isinstance(suggestion.proposed_value, dict)
    assert suggestion.proposed_value["location_id"] == imported_location.id
    assert suggestion.proposed_value["source_message_id"] == imported_messages[0].id


def test_import_save_remaps_context_update_suggestion_scalar_location_ids(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    upper_arcade = repositories.add_location(
        save_id=save.id,
        location_id="location-upper-arcade",
        name="Upper Arcade",
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    lower_arcade = repositories.add_location(
        save_id=save.id,
        location_id="location-lower-arcade",
        name="Lower Arcade",
        parent_location_id=upper_arcade.id,
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    iris = repositories.add_character(
        save_id=save.id,
        character_id="character-iris",
        name="Iris",
        role="scout",
        location_id=upper_arcade.id,
        source_message_id=PLAYER_MESSAGE_ID,
    )
    scene = repositories.upsert_scene_snapshot(
        save_id=save.id,
        snapshot_id="scene-scalar-location",
        current_location_id=upper_arcade.id,
        situation="The arcade is quiet.",
        objective="Find the cracked beacon lens.",
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="character",
        entity_id=iris.id,
        field_path="location_id",
        proposed_value=lower_arcade.id,
        reason="Move Iris to the lower arcade.",
        confidence=0.66,
        source_message_ids=[PLAYER_MESSAGE_ID],
        suggestion_id="suggestion-move-iris",
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="scene_snapshot",
        entity_id=scene.id,
        field_path="current_location_id",
        proposed_value=lower_arcade.id,
        reason="Move the scene to the lower arcade.",
        confidence=0.67,
        source_message_ids=[NARRATOR_MESSAGE_ID],
        suggestion_id="suggestion-move-scene",
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="location",
        entity_id=lower_arcade.id,
        field_path="parent_location_id",
        proposed_value=upper_arcade.id,
        reason="Set the lower arcade parent.",
        confidence=0.68,
        source_message_ids=[NARRATOR_MESSAGE_ID],
        suggestion_id="suggestion-location-parent",
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="character",
        entity_id=iris.id,
        field_path="location_id",
        proposed_value="location-not-exported",
        reason="Move Iris to a missing location.",
        confidence=0.69,
        source_message_ids=[PLAYER_MESSAGE_ID],
        suggestion_id="suggestion-missing-location",
    )
    bundle_path = (
        tmp_path / "exports" / "night-watch-suggestion-scalar-location.bragi-chat"
    )
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    imported = service.import_save(bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_locations = {
        location.name: location
        for location in repositories.list_locations(imported_save_id)
    }
    imported_iris = next(
        character
        for character in repositories.list_characters(imported_save_id)
        if character.name == "Iris"
    )
    imported_scene = repositories.get_scene_snapshot(imported_save_id)
    assert imported_scene is not None
    suggestions = {
        suggestion.reason: suggestion
        for suggestion in repositories.list_context_update_suggestions(
            imported_save_id
        )
    }
    assert suggestions["Move Iris to the lower arcade."].entity_id == imported_iris.id
    assert suggestions["Move Iris to the lower arcade."].proposed_value == (
        imported_locations["Lower Arcade"].id
    )
    assert suggestions["Move the scene to the lower arcade."].entity_id == (
        imported_scene.id
    )
    assert suggestions["Move the scene to the lower arcade."].proposed_value == (
        imported_locations["Lower Arcade"].id
    )
    assert suggestions["Set the lower arcade parent."].entity_id == (
        imported_locations["Lower Arcade"].id
    )
    assert suggestions["Set the lower arcade parent."].proposed_value == (
        imported_locations["Upper Arcade"].id
    )
    assert suggestions["Move Iris to a missing location."].proposed_value is None


def test_import_save_drops_unknown_context_source_metadata_refs(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    context_sources = data["context_sources"]
    assert isinstance(context_sources, list)
    context_sources.append(
        {
            "id": "ctx-unknown-metadata",
            "save_id": SAVE_ID,
            "source_type": "world_state",
            "source_id": "beacon_lens",
            "title": "Unknown metadata",
            "body": "This row references a message outside the bundle.",
            "metadata_json": json.dumps(
                {
                    "source_message_id": "message-not-exported",
                    "source_message_ids": [PLAYER_MESSAGE_ID],
                    "last_seen_message_id": NARRATOR_MESSAGE_ID,
                }
            ),
            "token_estimate": 8,
            "created_at": "2026-07-01T12:00:00+00:00",
            "updated_at": "2026-07-01T12:00:00+00:00",
            "archived_at": None,
        }
    )
    context_sources.append(
        {
            "id": "ctx-unsafe-provenance",
            "save_id": SAVE_ID,
            "source_type": "world_state",
            "source_id": "beacon_lens",
            "title": "Unsafe provenance",
            "body": "This row must be omitted instead of weakening provenance.",
            "metadata_json": json.dumps(
                {
                    "source_message_ids": [PLAYER_MESSAGE_ID],
                    "source_provenance_groups": [
                        [PLAYER_MESSAGE_ID],
                        ["message-not-exported"],
                    ],
                    "source_provenance_mode": "all",
                }
            ),
            "token_estimate": 8,
            "created_at": "2026-07-01T12:00:00+00:00",
            "updated_at": "2026-07-01T12:00:00+00:00",
            "archived_at": None,
        }
    )
    broken_bundle_path = (
        tmp_path / "night-watch-unknown-context-source-metadata.bragi-chat"
    )
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )

    imported = service.import_save(broken_bundle_path)
    imported_save_id = _imported_save_id(imported)

    imported_messages = repositories.list_messages(imported_save_id)
    assert all(
        source.title not in {"Unknown metadata", "Unsafe provenance"}
        for source in repositories.list_context_sources(imported_save_id)
    )
    assert imported_messages


@pytest.mark.parametrize(
    ("json_field", "json_value"),
    [
        ("before_json", "{not valid json"),
        ("after_json", "{not valid json"),
        ("before_json", "[]"),
        ("after_json", '"not an object"'),
    ],
)
def test_import_save_rejects_invalid_state_change_json_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    json_field: str,
    json_value: str,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    state_changes = data["state_changes"]
    assert isinstance(state_changes, list)
    state_change = state_changes[0]
    assert isinstance(state_change, dict)
    state_change[json_field] = json_value
    broken_bundle_path = tmp_path / f"night-watch-invalid-{json_field}.bragi-chat"
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    save_count = len(repositories.list_saves())

    with pytest.raises(module.ChatBundleError):
        service.import_save(broken_bundle_path)

    assert len(repositories.list_saves()) == save_count


def test_import_save_rejects_declared_media_without_sha256_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    metadata = _media_file_metadata(data)
    metadata.pop("sha256")
    broken_bundle_path = tmp_path / "night-watch-missing-sha.bragi-chat"
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    save_count = len(repositories.list_saves())

    with pytest.raises(module.ChatBundleError):
        service.import_save(broken_bundle_path)

    assert len(repositories.list_saves()) == save_count


def test_import_save_rejects_declared_media_byte_count_mismatch_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    metadata = _media_file_metadata(data)
    metadata["byte_count"] = len(MEDIA_BYTES) + 1
    broken_bundle_path = tmp_path / "night-watch-byte-count-mismatch.bragi-chat"
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    save_count = len(repositories.list_saves())

    with pytest.raises(module.ChatBundleError):
        service.import_save(broken_bundle_path)

    assert len(repositories.list_saves()) == save_count


def test_import_save_rejects_declared_media_byte_count_over_limit_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    metadata = _media_file_metadata(data)
    metadata["byte_count"] = module._MAX_BUNDLE_MEDIA_FILE_BYTES + 1
    broken_bundle_path = tmp_path / "night-watch-byte-count-over-limit.bragi-chat"
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=b"tiny",
    )
    save_count = len(repositories.list_saves())

    with pytest.raises(module.ChatBundleError):
        service.import_save(broken_bundle_path)

    assert len(repositories.list_saves()) == save_count


def test_chat_bundle_default_limits_support_large_save_exports() -> None:
    module = _chat_bundle_module()

    assert module._MAX_BUNDLE_MANIFEST_JSON_BYTES == 1024 * 1024
    assert module._MAX_BUNDLE_DATA_JSON_BYTES == 128 * 1024 * 1024
    assert module._MAX_BUNDLE_JSON_TOTAL_BYTES == 129 * 1024 * 1024
    assert module._MAX_BUNDLE_MEDIA_FILE_BYTES == 2 * 1024 * 1024 * 1024
    assert module._MAX_BUNDLE_MEDIA_TOTAL_BYTES == 2 * 1024 * 1024 * 1024


@pytest.mark.parametrize(
    "mismatch",
    ["save_title", "message_count", "media_count"],
)
def test_import_save_rejects_manifest_data_mismatch_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    mismatch: str,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    if mismatch == "save_title":
        save = data["save"]
        assert isinstance(save, dict)
        save["title"] = "Wrong Night Watch"
    elif mismatch == "message_count":
        counts = manifest["counts"]
        assert isinstance(counts, dict)
        counts["messages"] = 99
    elif mismatch == "media_count":
        counts = manifest["counts"]
        assert isinstance(counts, dict)
        counts["media_assets"] = 99
    else:
        raise AssertionError(f"Unhandled mismatch case: {mismatch}")
    broken_bundle_path = tmp_path / f"night-watch-{mismatch}-mismatch.bragi-chat"
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    save_ids = [save.id for save in repositories.list_saves()]

    with pytest.raises(module.ChatBundleError, match="manifest"):
        service.import_save(broken_bundle_path)

    assert [save.id for save in repositories.list_saves()] == save_ids


@pytest.mark.parametrize("member_name", ["manifest.json", "data.json"])
def test_import_save_rejects_oversized_json_member_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    member_name: str,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    manifest_payload = json.dumps(manifest).encode("utf-8")
    data_payload = json.dumps(data).encode("utf-8")
    if member_name == "manifest.json":
        monkeypatch.setattr(module, "_MAX_BUNDLE_MANIFEST_JSON_BYTES", 2)
        manifest_payload = b"{invalid oversized manifest"
        data_payload = b"{}"
    elif member_name == "data.json":
        monkeypatch.setattr(
            module,
            "_MAX_BUNDLE_DATA_JSON_BYTES",
            len(manifest_payload),
        )
        data_payload = b"{invalid oversized data" + (b"x" * len(manifest_payload))
    else:
        raise AssertionError(f"Unhandled JSON member: {member_name}")
    broken_bundle_path = tmp_path / f"night-watch-oversized-{member_name}.bragi-chat"
    _write_bundle_member_bytes(
        broken_bundle_path,
        manifest_payload=manifest_payload,
        data_payload=data_payload,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    save_ids = [save.id for save in repositories.list_saves()]

    with pytest.raises(module.ChatBundleError, match=f"{member_name} is too large"):
        service.import_save(broken_bundle_path)

    assert [save.id for save in repositories.list_saves()] == save_ids


def test_import_save_streams_json_members_without_unbounded_zipfile_read(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    service, _manifest, _data, _bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    bundle_path = tmp_path / "night-watch.bragi-chat"
    save_ids = [save.id for save in repositories.list_saves()]

    def reject_unbounded_read(
        _bundle: zipfile.ZipFile,
        _name: object,
        _pwd: object = None,
    ) -> bytes:
        raise AssertionError("ZipFile.read must not be used for bundle import")

    monkeypatch.setattr(zipfile.ZipFile, "read", reject_unbounded_read)

    imported = service.import_save(bundle_path)

    assert _imported_save_id(imported) not in save_ids


def test_import_save_rejects_unsafe_json_compression_ratio_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    messages = data["messages"]
    assert isinstance(messages, list)
    first_message = messages[0]
    assert isinstance(first_message, dict)
    first_message["body"] = "x" * (1024 * 1024)
    broken_bundle_path = tmp_path / "night-watch-compressed-json.bragi-chat"
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    save_ids = [save.id for save in repositories.list_saves()]

    with pytest.raises(module.ChatBundleError, match="suspiciously compressed"):
        service.import_save(broken_bundle_path)

    assert [save.id for save in repositories.list_saves()] == save_ids


def test_import_save_rejects_total_decompressed_payload_over_limit_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    manifest_payload = json.dumps(manifest).encode("utf-8")
    data_payload = json.dumps(data).encode("utf-8")
    monkeypatch.setattr(
        module,
        "_MAX_BUNDLE_TOTAL_DECOMPRESSED_BYTES",
        len(manifest_payload) + len(data_payload) + len(MEDIA_BYTES) - 1,
        raising=False,
    )
    broken_bundle_path = tmp_path / "night-watch-total-too-large.bragi-chat"
    _write_bundle_member_bytes(
        broken_bundle_path,
        manifest_payload=manifest_payload,
        data_payload=data_payload,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    save_ids = [save.id for save in repositories.list_saves()]

    with pytest.raises(module.ChatBundleError, match="Chat bundle is too large"):
        service.import_save(broken_bundle_path)

    assert [save.id for save in repositories.list_saves()] == save_ids


@pytest.mark.parametrize("duplicate_collection", ["messages", "media_assets"])
def test_import_save_rejects_duplicate_bundle_ids_without_new_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    duplicate_collection: str,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    rows = data[duplicate_collection]
    assert isinstance(rows, list)
    assert rows
    duplicate = dict(cast(dict[str, object], rows[0]))
    rows.append(duplicate)
    counts = manifest["counts"]
    assert isinstance(counts, dict)
    if duplicate_collection == "messages":
        counts["messages"] = len(rows)
    else:
        counts["media_assets"] = len(rows)
    broken_bundle_path = (
        tmp_path / f"night-watch-duplicate-{duplicate_collection}.bragi-chat"
    )
    _write_bundle_with_member(
        broken_bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    save_ids = [save.id for save in repositories.list_saves()]

    with pytest.raises(module.ChatBundleError, match="Duplicate"):
        service.import_save(broken_bundle_path)

    assert [save.id for save in repositories.list_saves()] == save_ids


def test_preview_and_import_reject_unsupported_bundle_version(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    bundle_path = tmp_path / "unsupported.bragi-chat"
    _write_bundle(
        bundle_path,
        manifest={
            "format": "bragi-chat-bundle",
            "bundle_format": "bragi-chat",
            "bundle_version": 999,
            "title": "Night Watch",
            "save_title": "Night Watch",
            "scenario_title": "Ashfall Keep",
            "message_count": 0,
            "media_count": 0,
        },
        data={},
    )
    service = _chat_bundle_service(repositories, tmp_path / "media")

    with pytest.raises(module.ChatBundleError):
        service.preview_import(bundle_path)
    with pytest.raises(module.ChatBundleError):
        service.import_save(bundle_path)


@pytest.mark.parametrize("hybrid", [False, True])
def test_import_rejects_retired_character_interaction_bundle(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    hybrid: bool,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    scenario = cast(dict[str, object], data["scenario"])
    content = json.loads(cast(str, scenario["content_json"]))
    if hybrid:
        content["_scenario_genres"] = ["full_roleplay", "character_interaction"]
    else:
        scenario["type"] = "character_interaction"
        manifest_scenario = cast(dict[str, object], manifest["scenario"])
        manifest_scenario["type"] = "character_interaction"
    scenario["content_json"] = json.dumps(content)
    bundle_path = tmp_path / f"retired-{hybrid}.bragi-chat"
    _write_bundle_with_member(
        bundle_path,
        manifest=manifest,
        data=data,
        bundle_name=bundle_media_path,
        payload=MEDIA_BYTES,
    )
    save_ids = [save.id for save in repositories.list_saves()]

    assert service.preview_import(bundle_path).title == "Night Watch"
    with pytest.raises(module.ChatBundleError, match="no longer supported"):
        service.import_save(bundle_path)

    assert [save.id for save in repositories.list_saves()] == save_ids


def test_preview_and_import_reject_malformed_bundle(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    bundle_path = tmp_path / "malformed.bragi-chat"
    bundle_path.write_text("not a zip bundle", encoding="utf-8")
    service = _chat_bundle_service(repositories, tmp_path / "media")

    with pytest.raises(module.ChatBundleError):
        service.preview_import(bundle_path)
    with pytest.raises(module.ChatBundleError):
        service.import_save(bundle_path)


def test_import_save_rejects_unexpected_zip_members(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _chat_bundle_module()
    service, manifest, data, bundle_media_path = _export_bundle_payloads(
        repositories,
        tmp_path,
    )
    bundle_path = tmp_path / "padded.bragi-chat"
    _write_bundle_with_members(
        bundle_path,
        manifest=manifest,
        data=data,
        members={
            bundle_media_path: MEDIA_BYTES,
            "ignored/padding.bin": b"x" * 128,
        },
    )
    save_ids = [save.id for save in repositories.list_saves()]

    assert service.preview_import(bundle_path).title == "Night Watch"
    with pytest.raises(module.ChatBundleError, match="Unexpected chat bundle member"):
        service.import_save(bundle_path)

    assert [save.id for save in repositories.list_saves()] == save_ids


def _seed_bundle_save(
    repositories: PersistenceRepositories,
    media_dir: Path,
    *,
    write_media_file: bool = True,
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY,
) -> SaveRecord:
    scenario = repositories.create_scenario(
        scenario_id=SCENARIO_ID,
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        interaction_mode=interaction_mode,
        content={
            "opening_message": "The tower bell cracks once.",
            "starting_scene": "The beacon gutters in the tower.",
        },
    )
    save = repositories.create_save(
        save_id=SAVE_ID,
        scenario_id=scenario.id,
        title="Night Watch",
        custom_instructions="Keep choices brief and grounded.",
        interaction_mode=interaction_mode,
    )
    repositories.set_app_setting(
        save_scenario_evolution_turn_interval_setting_key(save.id),
        3,
    )
    repositories.set_app_setting(
        save_image_style_preset_setting_key(save.id),
        "watercolor",
    )
    repositories.set_app_setting(
        scenario_template_evolution_turn_interval_setting_key(scenario.id),
        5,
    )
    repositories.append_message(
        message_id=PLAYER_MESSAGE_ID,
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    repositories.append_message(
        message_id=NARRATOR_MESSAGE_ID,
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lens flashes red and shows riders in the ash.",
        provider="fake-chat-provider",
        model="fake-chat-model",
        token_estimate=37,
    )
    repositories.upsert_world_state(
        state_id="world-state-old-patrol",
        save_id=save.id,
        key="old_patrol",
        value={"status": "departed"},
        category="faction",
        confidence=0.6,
        source_message_id=PLAYER_MESSAGE_ID,
    )
    repositories.archive_world_state(save_id=save.id, key="old_patrol")
    repositories.upsert_world_state(
        state_id="world-state-beacon-lens",
        save_id=save.id,
        key="beacon_lens",
        value={"color": "red", "lit": True},
        category="location",
        confidence=0.95,
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    repositories.add_memory(
        memory_id="memory-signal-code",
        save_id=save.id,
        body="Mara knows the eastern signal code.",
        tags=["mara", "signals"],
        importance=0.8,
        source_message_id=PLAYER_MESSAGE_ID,
        source_message_ids=[PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
    )
    repositories.add_context_observation(
        observation_id=OBSERVATION_ID,
        save_id=save.id,
        observation_type="open_thread",
        claim="The red beacon warning may matter later.",
        evidence_quote="The beacon shows a warning",
        source_message_ids=[PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
        scope="save",
        status="accepted",
        confidence=0.82,
        tags=["beacon", "warning"],
        metadata={"curation_action": "save_context"},
    )
    repositories.add_summary(
        summary_id="summary-beacon-warning",
        save_id=save.id,
        covers_message_start_id=PLAYER_MESSAGE_ID,
        covers_message_end_id=NARRATOR_MESSAGE_ID,
        body="Mara reaches the beacon and sees a warning.",
        provider="fake-chat-provider",
        model="fake-chat-model",
    )
    repositories.add_state_change(
        change_id="state-change-beacon-lens",
        save_id=save.id,
        operation="upsert",
        state_key="beacon_lens",
        before_json=json.dumps({"color": "dim", "lit": False}, sort_keys=True),
        after_json=json.dumps({"color": "red", "lit": True}, sort_keys=True),
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    repositories.add_save_scenario_update(
        update_id=SCENARIO_UPDATE_ID,
        save_id=save.id,
        title="Ashfall Keep: Red Lens",
        premise="A red warning has reached the isolated border keep.",
        player_role="Signal warden",
        content={
            "opening_message": "The red lens wakes.",
            "starting_scene": "The beacon burns crimson over the ash road.",
        },
        reason="The warning changes the current scene.",
        provider="fake-chat-provider",
        model="fake-chat-model",
        source_message_id=NARRATOR_MESSAGE_ID,
        source_message_ids=(PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID),
    )
    repositories.create_media_asset(
        asset_id=MEDIA_ASSET_ID,
        save_id=save.id,
        source_message_id=NARRATOR_MESSAGE_ID,
        type="image",
        path=MEDIA_PATH,
        thumbnail_path=None,
        prompt="red beacon lens over ash",
        provider="fake-image-provider",
        model="fake-image-model",
        status="succeeded",
    )
    if write_media_file:
        media_path = media_dir / MEDIA_PATH
        media_path.parent.mkdir(parents=True)
        media_path.write_bytes(MEDIA_BYTES)
    return save


def test_export_import_round_trips_active_scene_facts(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    save = _seed_bundle_save(repositories, media_dir)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara stands beside the red beacon lens.",
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    fact, _, _ = repositories.upsert_scene_fact(
        fact_id="scene-fact-beacon-lens",
        save_id=save.id,
        fact_type="object_location",
        subject_type="object",
        subject_id=None,
        subject_label="beacon lens",
        value="mounted above the tower stair",
        source_message_id=NARRATOR_MESSAGE_ID,
        evidence_quote="The lens flashes red",
    )
    bundle_path = tmp_path / "scene-facts.bragi-chat"

    chat_bundle_module.ChatBundleService(
        repositories=repositories,
        media_dir=media_dir,
    ).export_save(save.id, bundle_path)
    imported = chat_bundle_module.ChatBundleService(
        repositories=repositories,
        media_dir=media_dir,
    ).import_save(bundle_path)

    [imported_fact] = repositories.list_scene_facts(imported.save_id)
    imported_messages = repositories.list_messages(imported.save_id)
    imported_narrator = next(
        message for message in imported_messages if message.role == "narrator"
    )
    assert imported_fact.id != fact.id
    assert imported_fact.subject_label == "beacon lens"
    assert imported_fact.value == "mounted above the tower stair"
    assert imported_fact.scene_snapshot_id != fact.scene_snapshot_id
    assert imported_fact.provenance[0].source_message_id == imported_narrator.id


def _replace_seed_scenario_update_content(
    repositories: PersistenceRepositories,
    content: dict[str, object],
) -> None:
    repositories.connection.execute(
        "UPDATE save_scenario_updates SET content_json = ? WHERE id = ?",
        (json.dumps(content, sort_keys=True), SCENARIO_UPDATE_ID),
    )
    repositories.commit()


def _legacy_character_list_update_content() -> dict[str, object]:
    return {
        "opening_message": "The red lens wakes.",
        "starting_scene": "The beacon burns crimson over the ash road.",
        "factions": "Beacon wardens",
        "characters": "Captain Rell guards the cracked stair.",
        "rivals_and_factions": "Ash riders scout the ridge.",
        "reputation_and_contacts": "The old patrol owes Mara a warning.",
    }


def _cleaned_character_list_update_content() -> dict[str, object]:
    return {
        "opening_message": "The red lens wakes.",
        "starting_scene": "The beacon burns crimson over the ash road.",
        "factions": (
            "Beacon wardens\n\n"
            "Ash riders scout the ridge.\n\n"
            "The old patrol owes Mara a warning."
        ),
    }


def _seed_loss_bundle_save(
    repositories: PersistenceRepositories,
    media_dir: Path,
) -> SaveRecord:
    save = _seed_bundle_save(repositories, media_dir)
    epilogue_message = repositories.append_message(
        message_id=LOSS_EPILOGUE_MESSAGE_ID,
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon goes dark, and Mara's watch ends in ash.",
        provider="fake-loss-provider",
        model="fake-loss-model",
        token_estimate=18,
    )
    condition = repositories.add_loss_condition(
        condition_id=LOSS_CONDITION_ID,
        save_id=save.id,
        name="Beacon collapse",
        description="The beacon has collapsed after the red warning.",
        status="triggered",
        source="structured",
    )
    repositories.add_loss_condition_change(
        change_id=LOSS_CONDITION_CHANGE_ID,
        save_id=save.id,
        condition_id=condition.id,
        operation="update",
        before={"status": "active"},
        after={
            "id": condition.id,
            "name": condition.name,
            "description": "The beacon has collapsed after the red warning.",
            "status": "triggered",
            "source": "structured",
        },
        reason="The narrator described the terminal failure.",
        provider="fake-loss-provider",
        model="fake-loss-model",
        source_message_id=NARRATOR_MESSAGE_ID,
    )
    repositories.create_loss_outcome(
        outcome_id=LOSS_OUTCOME_ID,
        save_id=save.id,
        condition_id=condition.id,
        condition_name=condition.name,
        triggering_message_id=NARRATOR_MESSAGE_ID,
        explanation="The beacon falls and the watch is lost.",
        confidence=0.92,
        evidence={
            "items": [
                {
                    "source_message_id": NARRATOR_MESSAGE_ID,
                    "quote": "riders in the ash",
                }
            ]
        },
        provider="fake-loss-provider",
        model="fake-loss-model",
        epilogue_provider="fake-loss-provider",
        epilogue_model="fake-loss-model",
        epilogue_message_id=epilogue_message.id,
    )
    return save


def _seed_text_derived_world_rows(
    repositories: PersistenceRepositories,
    save_id: str,
) -> Any:
    character = repositories.add_character(
        character_id="character-rowan",
        save_id=save_id,
        name="Rowan",
        role="classmate",
        met=True,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=character.id,
        title=character.name,
    )
    player_message = repositories.append_character_text_message(
        message_id="text-player-notes",
        save_id=save_id,
        thread_id=thread.id,
        character_id=character.id,
        sender="player",
        body="Can you bring the repair notes?",
        in_world_sent_at="Friday evening after class",
        delivered_at="2026-07-01T12:05:00+00:00",
    )
    reply = repositories.append_character_text_message(
        message_id="text-reply-notes",
        save_id=save_id,
        thread_id=thread.id,
        character_id=character.id,
        sender="character",
        body="I promised I would bring repair notes.",
        provider="fake",
        model="fake-context",
        in_world_sent_at="Friday evening after class",
        delivered_at="2026-07-01T12:06:00+00:00",
        read_at="2026-07-01T12:07:00+00:00",
        reply_to_message_id=player_message.id,
    )
    source_ref = f"character_text_message:{reply.id}"
    memory = repositories.add_memory(
        memory_id="memory-rowan-repair-notes",
        save_id=save_id,
        body="Rowan promised to bring repair notes.",
        tags=["rowan", "promise"],
        importance=0.86,
        source_message_ids=[source_ref],
    )
    repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=character.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="knows",
        acquisition_method="told",
        confidence=0.86,
        source_message_ids=[source_ref],
        evidence_quote=reply.body,
        edge_id="edge-rowan-repair-notes",
    )
    repositories.add_context_update_audit(
        save_id=save_id,
        operation="created",
        entity_type="memory",
        entity_id=memory.id,
        field_path="*",
        before=None,
        after={"body": memory.body},
        reason="Text exchange established a promise.",
        confidence=0.86,
        source_message_ids=[source_ref],
        audit_id="audit-rowan-repair-notes",
    )
    repositories.add_character_text_provenance(
        save_id=save_id,
        thread_id=thread.id,
        text_message_id=reply.id,
        target_type="memory",
        target_id=memory.id,
        operation="created",
        field_path="*",
        provenance_id="provenance-rowan-memory",
    )
    return reply


def _seed_context_graph_rows(
    repositories: PersistenceRepositories,
    save_id: str,
) -> None:
    connection = repositories.connection
    connection.execute(
        """
        INSERT INTO context_sources(
            id, save_id, source_type, source_id, title, body, metadata_json,
            token_estimate
        )
        VALUES (?, ?, 'location', 'location-tower', 'Beacon Tower',
                'The beacon tower overlooks the ash road.', '{}', 12)
        """,
        ("ctx-location", save_id),
    )
    connection.execute(
        """
        INSERT INTO context_sources(
            id, save_id, source_type, source_id, title, body, metadata_json,
            token_estimate
        )
        VALUES (?, ?, 'media_asset', ?, 'Beacon image',
                'The prior beacon image.', '{}', 8)
        """,
        ("ctx-media", save_id, MEDIA_ASSET_ID),
    )
    connection.execute(
        """
        INSERT INTO context_sources(
            id, save_id, source_type, source_id, title, body, metadata_json,
            token_estimate
        )
        VALUES (?, ?, 'message', ?, 'Recent chronicle',
                'The recent chronicle.', '{}', 8)
        """,
        (
            "ctx-message",
            save_id,
            f"{PLAYER_MESSAGE_ID},{NARRATOR_MESSAGE_ID}",
        ),
    )
    connection.execute(
        """
        INSERT INTO locations(
            id, save_id, name, aliases_json, description, visual_description,
            connections_json, status, hazards_json, source_message_id,
            locked_fields_json
        )
        VALUES (?, ?, 'Beacon Tower', '[]', 'A cracked signal tower.',
                'Red glass and black ash.', '[]', 'lit', '[]', ?, '[]')
        """,
        ("location-tower", save_id, NARRATOR_MESSAGE_ID),
    )
    connection.execute(
        """
        INSERT INTO scene_snapshots(
            id, save_id, current_location_id, situation, objective,
            in_world_time, time_of_day, day_of_week, world_day_index,
            world_time_day_index, world_time_day_label, world_time_phase,
            world_time_clock_minutes, world_time_period_label,
            world_time_source_message_id, world_time_confidence, weather, mood,
            nearby_objects_json, hazards_json,
            present_character_ids_json, source_message_id, locked_fields_json
        )
        VALUES (?, ?, 'location-tower', 'The beacon burns red.',
                'Warn the keep.', 'Monday night', 'night', 'monday',
                2, 2, 'monday', 'night', NULL, '', ?, 0.87,
                'ash storm', 'urgent', '[]', '[]', '["character-mara"]',
                ?, '[]')
        """,
        ("scene-main", save_id, NARRATOR_MESSAGE_ID, NARRATOR_MESSAGE_ID),
    )
    connection.execute(
        """
        INSERT INTO characters(
            id, save_id, name, aliases_json, role, age, known_state, met,
            appearance, visual_notes, current_clothing, personality, voice,
            relationships_json,
            goals, motivations, current_intent, boundaries,
            attitude_toward_player, cooperation_conditions, status,
            location_id, private_notes, source_message_id,
            locked_fields_json, protected_from_maintenance, is_player_character
        )
        VALUES (?, ?, 'Mara', '[]', 'Signal warden', 'late 30s', 'At the tower.', 1,
                'Ash-covered cloak.', '',
                'Borrowed green raincoat over a linen shirt.',
                'Steady', 'Quiet', '{}',
                'Keep the beacon lit.', 'Protect the village.',
                'Guard the lens stair.', 'Will not leave the tower.',
                'Trusts the player under pressure.',
                'Helps after proof the lens can hold.',
                'alert', 'location-tower', '', ?, '[]', 1, 1)
        """,
        ("character-mara", save_id, PLAYER_MESSAGE_ID),
    )
    connection.execute(
        """
        INSERT INTO active_threads(
            id, save_id, title, description, status, priority, visibility,
            related_entities_json, source_message_id, locked_fields_json
        )
        VALUES (?, ?, 'Beacon warning', 'Warn the keep before dawn.',
                'active', 10, 'public', ?, ?, '[]')
        """,
        (
            "thread-beacon",
            save_id,
            json.dumps(["location:location-tower", "character:character-mara"]),
            NARRATOR_MESSAGE_ID,
        ),
    )
    connection.execute(
        """
        INSERT INTO entity_links(
            id, save_id, entity_type, entity_id, target_type, target_id, relation,
            source_message_id
        )
        VALUES (?, ?, 'character', 'character-mara', 'location',
                'location-tower', 'present_at', ?)
        """,
        ("link-mara-tower", save_id, NARRATOR_MESSAGE_ID),
    )
    connection.execute(
        """
        INSERT INTO character_knowledge_edges(
            id, save_id, character_id, target_type, target_id, knowledge_state,
            acquisition_method, confidence, source_message_id,
            source_message_ids_json, evidence_quote
        )
        VALUES (?, ?, 'character-mara', 'memory', 'memory-signal-code',
                'knows', 'witnessed', 1.0, ?, ?, 'Mara said the code aloud.')
        """,
        (
            "edge-mara-signal-code",
            save_id,
            NARRATOR_MESSAGE_ID,
            json.dumps(
                [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )
    connection.execute(
        """
        INSERT INTO message_visibility(
            id, save_id, message_id, character_id, visibility, confidence,
            source, evidence
        )
        VALUES (?, ?, ?, 'character-mara', 'visible', 1.0, 'scene_presence',
                'Mara was the active speaker.')
        """,
        ("visibility-mara-narrator", save_id, NARRATOR_MESSAGE_ID),
    )
    connection.execute(
        """
        INSERT INTO message_scene_presence(
            id, save_id, message_id, character_id, source
        )
        VALUES (?, ?, ?, 'character-mara', 'context_snapshot')
        """,
        ("presence-mara-narrator", save_id, NARRATOR_MESSAGE_ID),
    )
    connection.execute(
        """
        INSERT INTO context_update_suggestions(
            id, save_id, update_type, entity_type, entity_id, field_path,
            proposed_value_json, status, reason, confidence,
            source_message_ids_json
        )
        VALUES (?, ?, 'update', 'character', 'character-mara', 'location_id',
                '"location-tower"', 'accepted', 'Mara reached the tower.',
                0.9, ?)
        """,
        ("suggestion-mara-location", save_id, json.dumps([NARRATOR_MESSAGE_ID])),
    )
    connection.execute(
        """
        INSERT INTO context_update_audit(
            id, save_id, suggestion_id, operation, entity_type, entity_id,
            field_path, before_json, after_json, reason, confidence,
            source_message_ids_json
        )
        VALUES (?, ?, 'suggestion-mara-location', 'update', 'character',
                'character-mara', 'location_id', 'null', '"location-tower"',
                'Accepted update.', 0.9, ?)
        """,
        ("audit-mara-location", save_id, json.dumps([NARRATOR_MESSAGE_ID])),
    )
    connection.execute(
        """
        INSERT INTO jobs(
            id, save_id, type, status, payload_json, result_json, error,
            started_at, completed_at
        )
        VALUES (?, ?, 'image_generation', 'succeeded', '{}', '{}', NULL,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        ("job-terminal", save_id),
    )
    connection.execute(
        """
        INSERT INTO jobs(id, save_id, type, status, payload_json)
        VALUES (?, ?, 'image_generation', 'running', '{}')
        """,
        ("job-running", save_id),
    )
    connection.commit()


def _chat_bundle_service(
    repositories: PersistenceRepositories,
    media_dir: Path,
) -> Any:
    module = _chat_bundle_module()
    return module.ChatBundleService(repositories=repositories, media_dir=media_dir)


def _chat_bundle_module() -> Any:
    try:
        return importlib.import_module("bragi.services.chat_bundle_service")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing bragi.services.chat_bundle_service: {exc}")


def _repository_database_path(repositories: PersistenceRepositories) -> Path:
    row = repositories.connection.execute("PRAGMA database_list").fetchone()
    assert row is not None
    database_path = row["file"]
    assert isinstance(database_path, str)
    assert database_path
    return Path(database_path)


def _imported_save_id(imported: object) -> str:
    value = getattr(imported, "save_id", imported)
    assert isinstance(value, str)
    return value


def _manifest_save_title(manifest: dict[str, object]) -> object:
    return manifest.get("title", manifest.get("save_title"))


def _media_file_metadata(data: dict[str, object]) -> dict[str, object]:
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    assert media_assets
    media_asset = media_assets[0]
    assert isinstance(media_asset, dict)
    files = media_asset["files"]
    assert isinstance(files, dict)
    metadata = files["path"]
    assert isinstance(metadata, dict)
    return cast(dict[str, object], metadata)


def _reverse_bundle_rows(bundle_path: Path, table_name: str) -> None:
    _rewrite_bundle_data(
        bundle_path,
        lambda data: data.__setitem__(
            table_name,
            list(reversed(cast(list[object], data[table_name]))),
        ),
    )


def _remove_bundle_row(bundle_path: Path, table_name: str, row_id: str) -> None:
    def remove_row(data: dict[str, object]) -> None:
        rows = data[table_name]
        assert isinstance(rows, list)
        data[table_name] = [
            row
            for row in rows
            if not isinstance(row, dict) or row.get("id") != row_id
        ]

    _rewrite_bundle_data(bundle_path, remove_row)


def _rewrite_bundle_data(
    bundle_path: Path,
    rewrite: Callable[[dict[str, object]], None],
) -> None:
    rewritten_path = bundle_path.with_suffix(".rewritten.bragi-chat")
    with zipfile.ZipFile(bundle_path) as source:
        with zipfile.ZipFile(rewritten_path, "w") as target:
            data = json.loads(source.read("data.json"))
            assert isinstance(data, dict)
            rewrite(cast(dict[str, object], data))
            for item in source.infolist():
                payload = (
                    json.dumps(data).encode("utf-8")
                    if item.filename == "data.json"
                    else source.read(item.filename)
                )
                target.writestr(item, payload)
    rewritten_path.replace(bundle_path)


def _replace_director_pressure_history_source(
    data: dict[str, object],
    *,
    source_message_id: str,
) -> None:
    rows = data["world_state"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        if row.get("key") != DIRECTOR_PRESSURE_STATE_KEY:
            continue
        value_json = row["value_json"]
        assert isinstance(value_json, str)
        value = json.loads(value_json)
        assert isinstance(value, dict)
        history = value["escalation_history"]
        assert isinstance(history, list)
        entry = history[0]
        assert isinstance(entry, dict)
        entry["source_message_id"] = source_message_id
        row["value_json"] = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        return
    raise AssertionError("Director pressure world state row not found")


def _replace_save_app_setting_value(
    data: dict[str, object],
    *,
    scope: str,
    key: str,
    value: object,
) -> None:
    rows = data["save_app_settings"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        if row.get("scope") == scope and row.get("key") == key:
            row["value_json"] = json.dumps(value)
            return
    raise AssertionError(f"Missing save app setting: {scope}:{key}")


def _save_app_setting_value(
    data: dict[str, object],
    *,
    scope: str,
    key: str,
) -> object:
    rows = data["save_app_settings"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        if row.get("scope") == scope and row.get("key") == key:
            return row.get("value_json")
    raise AssertionError(f"Missing save app setting: {scope}:{key}")


def _media_asset_by_id(
    media_assets: list[object],
    media_asset_id: str,
) -> dict[str, object]:
    for media_asset in media_assets:
        assert isinstance(media_asset, dict)
        if media_asset.get("id") == media_asset_id:
            return cast(dict[str, object], media_asset)
    raise AssertionError(f"Missing media asset {media_asset_id}")


def _move_media_asset_to_snapshot_only(
    manifest: dict[str, object],
    data: dict[str, object],
    *,
    media_asset_id: str,
) -> None:
    media_assets = data["media_assets"]
    assert isinstance(media_assets, list)
    moved_media_asset = _media_asset_by_id(media_assets, media_asset_id)
    active_media_assets = [
        media_asset
        for media_asset in media_assets
        if not (
            isinstance(media_asset, dict)
            and media_asset.get("id") == media_asset_id
        )
    ]
    assert len(active_media_assets) == len(media_assets) - 1
    data["media_assets"] = active_media_assets
    snapshot_media_assets = data.setdefault("snapshot_media_assets", [])
    assert isinstance(snapshot_media_assets, list)
    snapshot_media_assets.append(moved_media_asset)
    counts = manifest["counts"]
    assert isinstance(counts, dict)
    counts["media_assets"] = len(active_media_assets)


def _require_save(
    repositories: PersistenceRepositories,
    save_id: str,
) -> SaveRecord:
    save = repositories.get_save(save_id)
    assert save is not None
    return save


def _export_bundle_payloads(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> tuple[Any, dict[str, object], dict[str, object], str]:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    bundle_path = tmp_path / "night-watch.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))

    metadata = _media_file_metadata(data)
    bundle_media_path = metadata["bundle_path"]
    assert isinstance(bundle_media_path, str)
    assert bundle_media_path
    return service, manifest, data, bundle_media_path


def _export_video_bundle_payloads(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> tuple[Any, dict[str, object], dict[str, object], dict[str, bytes]]:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.create_media_asset(
        asset_id=VIDEO_MEDIA_ASSET_ID,
        save_id=save.id,
        source_message_id=NARRATOR_MESSAGE_ID,
        source_media_asset_id=MEDIA_ASSET_ID,
        type="video",
        mime_type="video/mp4",
        path=VIDEO_MEDIA_PATH,
        thumbnail_path=None,
        prompt="animate the red lens warning over the ash road",
        provider="fake-video-provider",
        model="fake-video-model",
        status="succeeded",
        metadata={"duration_seconds": 5, "flow": "image_plus_text_to_video"},
    )
    video_path = media_dir / VIDEO_MEDIA_PATH
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(VIDEO_MEDIA_BYTES)
    bundle_path = tmp_path / "night-watch-video.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)
    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        data = json.loads(bundle.read("data.json"))
        media_members = {
            name: bundle.read(name)
            for name in bundle.namelist()
            if name.startswith("media/")
        }

    return service, manifest, data, media_members


def _write_bundle(
    bundle_path: Path,
    *,
    manifest: dict[str, object],
    data: dict[str, object],
) -> None:
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        bundle.writestr("data.json", json.dumps(data))


def _write_bundle_with_member(
    bundle_path: Path,
    *,
    manifest: dict[str, object],
    data: dict[str, object],
    bundle_name: str,
    payload: bytes,
) -> None:
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        bundle.writestr("data.json", json.dumps(data))
        bundle.writestr(bundle_name, payload)


def _write_bundle_with_members(
    bundle_path: Path,
    *,
    manifest: dict[str, object],
    data: dict[str, object],
    members: dict[str, bytes],
) -> None:
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        bundle.writestr("data.json", json.dumps(data))
        for bundle_name, payload in members.items():
            bundle.writestr(bundle_name, payload)


def _write_bundle_member_bytes(
    bundle_path: Path,
    *,
    manifest_payload: bytes,
    data_payload: bytes,
    bundle_name: str,
    payload: bytes,
) -> None:
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", manifest_payload)
        bundle.writestr("data.json", data_payload)
        bundle.writestr(bundle_name, payload)


def test_chat_bundle_round_trips_turn_outcomes_with_remapped_message_refs(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    save = _seed_bundle_save(repositories, media_dir)
    repositories.add_turn_outcome(
        save_id=save.id,
        message_id=NARRATOR_MESSAGE_ID,
        payload={
            "save_id": save.id,
            "message_id": NARRATOR_MESSAGE_ID,
            "source_message_ids": [PLAYER_MESSAGE_ID, NARRATOR_MESSAGE_ID],
            "attempted_action": "I climb toward the beacon lens.",
            "attempt_resolution": "succeeded",
            "attempt_evidence_source_ids": [f"message:{PLAYER_MESSAGE_ID}"],
            "effects": [
                {
                    "candidate_id": "scene:mood",
                    "candidate_type": "scene_snapshot_field",
                    "domain": "scene",
                    "operation": "update",
                    "state_key": "scene_snapshot.mood",
                    "field_path": "mood",
                    "character_id": "",
                    "target_type": "",
                    "target_id": "",
                    "value": {"mood": "uneasy"},
                    "confidence": 0.9,
                    "evidence_source_ids": [f"message:{NARRATOR_MESSAGE_ID}"],
                    "evidence_quote": "lens flashes red",
                    "verifier_status": "rendered",
                    "safe_to_commit": True,
                    "application_status": "committed",
                    "reason": "rendered",
                    "changed": True,
                }
            ],
            "applied_domains": ["scene"],
            "queued_domains": [],
            "verification_passed": True,
            "verifier_available": True,
            "post_turn_update_needed": False,
            "committed_count": 1,
            "confirmation_queued_count": 0,
        },
    )
    bundle_path = tmp_path / "outcomes.bragi-chat"
    service = _chat_bundle_service(repositories, media_dir)

    service.export_save(save.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert [row["message_id"] for row in data["turn_outcomes"]] == [
        NARRATOR_MESSAGE_ID
    ]

    imported = service.import_save(bundle_path)
    imported_save = repositories.get_save(_imported_save_id(imported))
    assert imported_save is not None
    imported_messages = [
        message
        for message in repositories.list_messages(imported_save.id)
        if message.role == "narrator"
    ]
    assert len(imported_messages) == 1
    imported_narrator_id = imported_messages[0].id
    imported_player_id = next(
        message.id
        for message in repositories.list_messages(imported_save.id)
        if message.role == "player"
    )
    outcome = repositories.get_turn_outcome_for_message(
        save_id=imported_save.id,
        message_id=imported_narrator_id,
    )
    assert outcome is not None
    assert outcome.payload["save_id"] == imported_save.id
    assert outcome.payload["attempt_resolution"] == "succeeded"
    assert outcome.payload["source_message_ids"] == [
        imported_player_id,
        imported_narrator_id,
    ]
    assert outcome.payload["attempt_evidence_source_ids"] == [
        f"message:{imported_player_id}"
    ]
    raw_effects = outcome.payload["effects"]
    assert isinstance(raw_effects, list)
    effect = raw_effects[0]
    assert isinstance(effect, dict)
    assert effect["evidence_source_ids"] == [f"message:{imported_narrator_id}"]
    assert effect["domain"] == "scene"
