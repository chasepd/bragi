from __future__ import annotations

import importlib
import json
import sqlite3
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from bragi.interaction_mode import InteractionMode
from bragi.persistence.migrations import CURRENT_SCHEMA_VERSION, migrate_database
from bragi.persistence.repositories import PersistenceRepositories

SCENARIO_ID = "scenario-ashfall"
VALID_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000b49444154789c6360000200000500017a5eab3f"
    "0000000049454e44ae426082"
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_export_scenario_writes_manifest_and_definition_only(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = _seed_scenario(repositories)
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="This chronicle message must not be in a scenario bundle.",
    )
    bundle_path = tmp_path / "exports" / "ashfall.bragi-scenario"
    service = _scenario_bundle_service(repositories)

    manifest = service.export_scenario(scenario.id, bundle_path)

    assert manifest.bundle_version == 2
    assert manifest.title == "Ashfall Keep"
    assert manifest.scenario_type == "full_roleplay"

    with zipfile.ZipFile(bundle_path) as bundle:
        assert set(bundle.namelist()) == {"manifest.json", "data.json"}
        manifest_payload = json.loads(bundle.read("manifest.json"))
        assert manifest_payload["format"] == "bragi-scenario-bundle"
        assert manifest_payload["bundle_version"] == 2
        assert manifest_payload["scenario"]["id"] == SCENARIO_ID
        assert manifest_payload["scenario"]["title"] == "Ashfall Keep"
        assert manifest_payload["scenario"]["type"] == "full_roleplay"

        data = json.loads(bundle.read("data.json"))
        assert set(data) == {"scenario"}
        assert data["scenario"]["id"] == SCENARIO_ID
        assert data["scenario"]["type"] == "full_roleplay"
        assert data["scenario"]["title"] == "Ashfall Keep"
        assert data["scenario"]["premise"] == "A border keep is cut off by ash."
        assert data["scenario"]["player_role"] == "Signal warden"
        assert data["scenario"]["content"]["opening_message"] == (
            "Ash falls over the gatehouse.\n\nThe beacon lens wakes red."
        )
        assert data["scenario"]["content"]["character_starters"][0]["name"] == (
            "Captain Ilyra"
        )
        assert data["scenario"]["content"]["_source"] == {
            "origin": "ai_draft",
            "generation_prompt": "A border keep cut off by ash storms.",
        }
        assert "_canon_claims" not in data["scenario"]["content"]
        assert "starting_scene" not in data["scenario"]["content"]
        assert "messages" not in data
        assert "saves" not in data


def test_preview_import_reads_scenario_manifest_without_mutating_database(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = _seed_scenario(repositories)
    service = _scenario_bundle_service(repositories)
    bundle_path = tmp_path / "ashfall.bragi-scenario"
    service.export_scenario(scenario.id, bundle_path)

    preview = service.preview_import(bundle_path)

    assert preview.scenario_id == SCENARIO_ID
    assert preview.title == "Ashfall Keep"
    assert preview.scenario_type == "full_roleplay"
    assert len(repositories.list_scenarios()) == 1


def test_import_scenario_creates_new_id_and_unique_duplicate_title(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = _seed_scenario(repositories)
    service = _scenario_bundle_service(repositories)
    bundle_path = tmp_path / "ashfall.bragi-scenario"
    service.export_scenario(scenario.id, bundle_path)

    first_import = service.import_scenario(bundle_path)
    second_import = service.import_scenario(bundle_path)

    assert first_import.scenario_id != SCENARIO_ID
    assert second_import.scenario_id not in {SCENARIO_ID, first_import.scenario_id}
    assert first_import.title == "Ashfall Keep (imported)"
    assert second_import.title == "Ashfall Keep (imported 2)"

    imported = repositories.get_scenario(first_import.scenario_id)
    assert imported is not None
    assert imported.type == "full_roleplay"
    assert imported.premise == "A border keep is cut off by ash."
    assert imported.player_role == "Signal warden"
    imported_content = json.loads(imported.content_json)
    assert "_canon_claims" not in imported_content
    assert "starting_scene" not in imported_content
    assert imported_content["opening_message"] == (
        "Ash falls over the gatehouse.\n\nThe beacon lens wakes red."
    )
    assert imported_content["character_starters"][0]["name"] == "Captain Ilyra"
    assert imported_content["_source"] == {
        "origin": "ai_draft",
        "generation_prompt": "A border keep cut off by ash storms.",
        "content_rating": "unclassified",
    }


def test_scenario_bundle_round_trips_storyteller_mode_and_defaults_legacy_mode(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="The Ceremony",
        premise="A rival waits in the wings.",
        player_role="",
        content={"opening_message": "The orchestra falls silent."},
        interaction_mode=InteractionMode.STORYTELLER,
    )
    service = _scenario_bundle_service(repositories)
    bundle_path = tmp_path / "ceremony.bragi-scenario"

    service.export_scenario(scenario.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["scenario"]["interaction_mode"] == "storyteller"
    imported = service.import_scenario(bundle_path)
    imported_record = repositories.get_scenario(imported.scenario_id)
    assert imported_record is not None
    assert imported_record.interaction_mode is InteractionMode.STORYTELLER

    legacy_path = tmp_path / "ceremony-legacy.bragi-scenario"
    del data["scenario"]["interaction_mode"]
    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    _write_bundle(legacy_path, manifest=manifest, data=data)
    legacy_import = service.import_scenario(legacy_path)
    legacy_record = repositories.get_scenario(legacy_import.scenario_id)
    assert legacy_record is not None
    assert legacy_record.interaction_mode is InteractionMode.ROLEPLAY


def test_export_import_scenario_bundle_preserves_starter_reference_images(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    source_path = media_dir / "scenario-starters" / SCENARIO_ID / "ilyra.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(VALID_PNG_BYTES)
    scenario = repositories.create_scenario(
        scenario_id=SCENARIO_ID,
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash.",
        player_role="Signal warden",
        content={
            "title": "Ashfall Keep",
            "premise": "A border keep is cut off by ash.",
            "player_role": "Signal warden",
            "opening_message": "The beacon lens wakes red.",
            "_canon_claims": {
                "version": 1,
                "source_digest": "derived-content-is-rebuilt-on-import",
                "claims": [],
            },
            "character_starters": [
                {
                    "starter_id": "starter-ilyra",
                    "name": "Captain Ilyra",
                    "reference_image": {
                        "id": "starter-ref-ilyra",
                        "path": f"scenario-starters/{SCENARIO_ID}/ilyra.png",
                        "thumbnail_path": None,
                        "mime_type": "image/png",
                        "prompt_preview": "Uploaded character reference image",
                        "source": "uploaded",
                    },
                }
            ],
        },
    )
    bundle_path = tmp_path / "ashfall.bragi-scenario"
    service = _scenario_bundle_service(repositories, media_dir=media_dir)

    service.export_scenario(scenario.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        media_members = [
            name for name in bundle.namelist() if name.startswith("media/")
        ]
        assert len(media_members) == 1
        data = json.loads(bundle.read("data.json"))
        reference = data["scenario"]["content"]["character_starters"][0][
            "reference_image"
        ]
        assert reference["bundle_path"] == media_members[0]
        assert bundle.read(media_members[0]) == source_path.read_bytes()

    imported = service.import_scenario(bundle_path)
    imported_record = repositories.get_scenario(imported.scenario_id)
    assert imported_record is not None
    imported_content = json.loads(imported_record.content_json)
    imported_reference = imported_content["character_starters"][0]["reference_image"]
    assert imported_reference["id"] != "starter-ref-ilyra"
    assert imported_reference["mime_type"] == "image/png"
    assert imported_reference["bundle_path"] is None
    assert (
        media_dir / imported_reference["path"]
    ).read_bytes() == source_path.read_bytes()
    assert imported_reference["thumbnail_path"]
    assert (media_dir / imported_reference["thumbnail_path"]).is_file()


def test_import_scenario_bundle_strips_unbundled_starter_reference_paths(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "path-only.bragi-scenario"
    manifest = _manifest_payload(scenario_id="scenario-1", title="Ashfall Keep")
    manifest["bundle_version"] = 2
    data = _data_payload(scenario_id="scenario-1", title="Ashfall Keep")
    scenario = cast(dict[str, object], data["scenario"])
    content = cast(dict[str, object], scenario["content"])
    content["character_starters"] = [
        {
            "name": "Captain Ilyra",
            "reference_image": {
                "id": "starter-ref-ilyra",
                "path": "scenario-starters/scenario-1/ilyra.png",
                "thumbnail_path": "scenario-starters/scenario-1/thumbnails/ilyra.png",
                "mime_type": "image/png",
                "prompt_preview": "Uploaded character reference image",
                "source": "uploaded",
            },
        }
    ]
    _write_bundle(bundle_path, manifest=manifest, data=data)
    service = _scenario_bundle_service(repositories, media_dir=tmp_path / "media")

    imported = service.import_scenario(bundle_path)

    imported_record = repositories.get_scenario(imported.scenario_id)
    assert imported_record is not None
    imported_content = json.loads(imported_record.content_json)
    assert imported_content["character_starters"][0]["reference_image"] is None


def test_export_scenario_rejects_starter_reference_paths_outside_namespace(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _scenario_bundle_module()
    media_dir = tmp_path / "media"
    scenario = repositories.create_scenario(
        scenario_id=SCENARIO_ID,
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash.",
        player_role="Signal warden",
        content={
            "opening_message": "The beacon lens wakes red.",
            "character_starters": [
                {
                    "name": "Captain Ilyra",
                    "reference_image": {
                        "id": "starter-ref-ilyra",
                        "path": "save-1/private.png",
                        "thumbnail_path": None,
                        "mime_type": "image/png",
                    },
                }
            ],
        },
    )
    service = _scenario_bundle_service(repositories, media_dir=media_dir)

    with pytest.raises(module.ScenarioBundleError, match="path is invalid"):
        service.export_scenario(scenario.id, tmp_path / "invalid.bragi-scenario")


def test_export_scenario_rejects_oversized_starter_reference_images(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _scenario_bundle_module()
    monkeypatch.setattr(module, "_MAX_SCENARIO_BUNDLE_MEDIA_BYTES", 8)
    media_dir = tmp_path / "media"
    source_path = media_dir / "scenario-starters" / SCENARIO_ID / "ilyra.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(VALID_PNG_BYTES)
    scenario = repositories.create_scenario(
        scenario_id=SCENARIO_ID,
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash.",
        player_role="Signal warden",
        content={
            "opening_message": "The beacon lens wakes red.",
            "character_starters": [
                {
                    "name": "Captain Ilyra",
                    "reference_image": {
                        "id": "starter-ref-ilyra",
                        "path": f"scenario-starters/{SCENARIO_ID}/ilyra.png",
                        "thumbnail_path": None,
                        "mime_type": "image/png",
                    },
                }
            ],
        },
    )
    bundle_path = tmp_path / "oversized.bragi-scenario"
    service = _scenario_bundle_service(repositories, media_dir=media_dir)

    with pytest.raises(module.ScenarioBundleError, match="too large"):
        service.export_scenario(scenario.id, bundle_path)

    assert not bundle_path.exists()


def test_export_scenario_rejects_too_many_starter_reference_images(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _scenario_bundle_module()
    monkeypatch.setattr(module, "_MAX_SCENARIO_BUNDLE_MEDIA_REFERENCES", 1)
    media_dir = tmp_path / "media"
    starters: list[dict[str, object]] = []
    for index in range(2):
        image_path = (
            media_dir / "scenario-starters" / SCENARIO_ID / f"ilyra-{index}.png"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(VALID_PNG_BYTES)
        starters.append(
            {
                "name": f"Captain Ilyra {index}",
                "reference_image": {
                    "id": f"starter-ref-{index}",
                    "path": f"scenario-starters/{SCENARIO_ID}/ilyra-{index}.png",
                    "thumbnail_path": None,
                    "mime_type": "image/png",
                },
            }
        )
    scenario = repositories.create_scenario(
        scenario_id=SCENARIO_ID,
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash.",
        player_role="Signal warden",
        content={
            "opening_message": "The beacon lens wakes red.",
            "character_starters": starters,
        },
    )
    service = _scenario_bundle_service(repositories, media_dir=media_dir)

    with pytest.raises(module.ScenarioBundleError, match="too many media files"):
        service.export_scenario(scenario.id, tmp_path / "too-many.bragi-scenario")


def test_export_scenario_rejects_total_starter_reference_media_over_limit(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _scenario_bundle_module()
    monkeypatch.setattr(module, "_MAX_SCENARIO_BUNDLE_TOTAL_MEDIA_BYTES", 1)
    media_dir = tmp_path / "media"
    source_path = media_dir / "scenario-starters" / SCENARIO_ID / "ilyra.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(VALID_PNG_BYTES)
    scenario = repositories.create_scenario(
        scenario_id=SCENARIO_ID,
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash.",
        player_role="Signal warden",
        content={
            "opening_message": "The beacon lens wakes red.",
            "character_starters": [
                {
                    "name": "Captain Ilyra",
                    "reference_image": {
                        "id": "starter-ref-ilyra",
                        "path": f"scenario-starters/{SCENARIO_ID}/ilyra.png",
                        "thumbnail_path": None,
                        "mime_type": "image/png",
                    },
                }
            ],
        },
    )
    service = _scenario_bundle_service(repositories, media_dir=media_dir)

    with pytest.raises(module.ScenarioBundleError, match="media is too large"):
        service.export_scenario(scenario.id, tmp_path / "total-large.bragi-scenario")


def test_export_scenario_rejects_duplicate_bundle_media_member_names(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _scenario_bundle_module()
    media_dir = tmp_path / "media"
    starters: list[dict[str, object]] = []
    for index in range(2):
        image_path = (
            media_dir / "scenario-starters" / SCENARIO_ID / f"ilyra-{index}.png"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(VALID_PNG_BYTES)
        starters.append(
            {
                "name": f"Captain Ilyra {index}",
                "reference_image": {
                    "id": "starter-ref-shared",
                    "path": f"scenario-starters/{SCENARIO_ID}/ilyra-{index}.png",
                    "thumbnail_path": None,
                    "mime_type": "image/png",
                },
            }
        )
    scenario = repositories.create_scenario(
        scenario_id=SCENARIO_ID,
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash.",
        player_role="Signal warden",
        content={
            "opening_message": "The beacon lens wakes red.",
            "character_starters": starters,
        },
    )
    service = _scenario_bundle_service(repositories, media_dir=media_dir)

    with pytest.raises(module.ScenarioBundleError, match="Duplicate"):
        service.export_scenario(scenario.id, tmp_path / "duplicate.bragi-scenario")


def test_import_scenario_bundle_rejects_duplicate_media_reference_fanout(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _scenario_bundle_module()
    media_dir = tmp_path / "media"
    bundle_path = tmp_path / "duplicate-media.bragi-scenario"
    manifest = _manifest_payload(scenario_id="scenario-1", title="Ashfall Keep")
    manifest["bundle_version"] = 2
    data = _data_payload(scenario_id="scenario-1", title="Ashfall Keep")
    scenario = cast(dict[str, object], data["scenario"])
    content = cast(dict[str, object], scenario["content"])
    content["character_starters"] = [
        {
            "name": "Captain Ilyra",
            "reference_image": {
                "id": "starter-ref-1",
                "bundle_path": "media/reference.png",
            },
        },
        {
            "name": "Signaler Nia",
            "reference_image": {
                "id": "starter-ref-2",
                "bundle_path": "media/reference.png",
            },
        },
    ]
    _write_bundle(
        bundle_path,
        manifest=manifest,
        data=data,
        media_members=[("media/reference.png", VALID_PNG_BYTES)],
    )
    service = _scenario_bundle_service(repositories, media_dir=media_dir)

    with pytest.raises(module.ScenarioBundleError, match="Duplicate"):
        service.import_scenario(bundle_path)

    assert not [path for path in media_dir.rglob("*") if path.is_file()]


def test_import_scenario_bundle_cleans_up_materialized_media_after_failure(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    bundle_path = tmp_path / "partial-media.bragi-scenario"
    manifest = _manifest_payload(scenario_id="scenario-1", title="Ashfall Keep")
    manifest["bundle_version"] = 2
    data = _data_payload(scenario_id="scenario-1", title="Ashfall Keep")
    scenario = cast(dict[str, object], data["scenario"])
    content = cast(dict[str, object], scenario["content"])
    content["character_starters"] = [
        {
            "name": "Captain Ilyra",
            "reference_image": {
                "id": "starter-ref-1",
                "bundle_path": "media/first.png",
            },
        },
        {
            "name": "Signaler Nia",
            "reference_image": {
                "id": "starter-ref-2",
                "bundle_path": "media/corrupt.png",
            },
        },
    ]
    _write_bundle(
        bundle_path,
        manifest=manifest,
        data=data,
        media_members=[
            ("media/first.png", VALID_PNG_BYTES),
            ("media/corrupt.png", b"\x89PNG\r\n\x1a\nnot a png"),
        ],
    )
    service = _scenario_bundle_service(repositories, media_dir=media_dir)

    with pytest.raises(ValueError, match="Unsupported image upload type"):
        service.import_scenario(bundle_path)

    assert repositories.list_scenarios() == []
    assert not [path for path in media_dir.rglob("*") if path.is_file()]


def test_export_import_preserves_investigation_mystery_sections(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        scenario_id="scenario-mystery",
        type="investigation_mystery",
        title="Broken Hours",
        premise="A curator disappears during a gala.",
        player_role="Lead investigator",
        content={
            "title": "Broken Hours",
            "premise": "A curator disappears during a gala.",
            "player_character_name": "Inspector Mara Voss",
            "player_role": "Lead investigator",
            "case_facts": "The east gallery was sealed during the disappearance.",
            "clues": "The watch log skips eight minutes.",
            "timeline": "Public alarm at 9:21; hidden lift movement at 9:12.",
            "red_herrings": "The bloody glove is from a mannequin repair.",
            "hidden_truth": "Sera hid the smuggling ledger in the lift.",
            "case_status": "Unresolved; public facts only.",
            "tone_genre": "Quiet investigative noir.",
            "opening_message": "Rain taps the museum glass.",
        },
    )
    bundle_path = tmp_path / "broken-hours.bragi-scenario"
    service = _scenario_bundle_service(repositories)

    manifest = service.export_scenario(scenario.id, bundle_path)
    imported = service.import_scenario(bundle_path)

    assert manifest.scenario_type == "investigation_mystery"
    assert imported.scenario_type == "investigation_mystery"
    imported_record = repositories.get_scenario(imported.scenario_id)
    assert imported_record is not None
    assert imported_record.type == "investigation_mystery"
    content = json.loads(imported_record.content_json)
    assert content["case_facts"] == (
        "The east gallery was sealed during the disappearance."
    )
    assert content["hidden_truth"] == "Sera hid the smuggling ledger in the lift."


def test_export_import_preserves_political_intrigue_sections(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        scenario_id="scenario-intrigue",
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
    bundle_path = tmp_path / "council-of-ash.bragi-scenario"
    service = _scenario_bundle_service(repositories)

    manifest = service.export_scenario(scenario.id, bundle_path)
    imported = service.import_scenario(bundle_path)

    assert manifest.scenario_type == "political_intrigue"
    assert imported.scenario_type == "political_intrigue"
    imported_record = repositories.get_scenario(imported.scenario_id)
    assert imported_record is not None
    assert imported_record.type == "political_intrigue"
    content = json.loads(imported_record.content_json)
    assert content["political_factions"] == "Guilds, Old Families, and dock unions."
    assert content["obligations_and_favors"] == (
        "Orro owes Mara one public endorsement."
    )
    assert content["public_private_knowledge"] == (
        "The public knows the vote is close; only Mara knows the favor."
    )


def test_export_import_preserves_first_contact_exploration_sections(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        scenario_id="scenario-contact",
        type="first_contact_exploration",
        title="Songs Under Europa",
        premise="A survey crew finds patterned signals under the ice.",
        player_role="Mission linguist",
        content={
            "title": "Songs Under Europa",
            "premise": "A survey crew finds patterned signals under the ice.",
            "player_character_name": "Dr. Mara Voss",
            "player_role": "Mission linguist",
            "mission_profile": "Survey the hidden ocean.",
            "ship_or_base_status": "Habitat heat is stable for 42 hours.",
            "exploration_target": "A black-water cavern beneath the ice.",
            "unknown_intelligence": "An unseen singer answers sonar.",
            "knowledge_state": "Observed songs; unknown intent.",
            "translation_progress": (
                "Three descending pulses may mean open water."
            ),
            "discoveries_and_samples": "Metallic spores remain quarantined.",
            "hazards_and_escalation": "Thermal fissures are spreading.",
            "tone_genre": "Hopeful, tense exploration science fiction.",
            "opening_message": "Blue light pulses beneath the ice.",
        },
    )
    bundle_path = tmp_path / "songs-under-europa.bragi-scenario"
    service = _scenario_bundle_service(repositories)

    manifest = service.export_scenario(scenario.id, bundle_path)
    imported = service.import_scenario(bundle_path)

    assert manifest.scenario_type == "first_contact_exploration"
    assert imported.scenario_type == "first_contact_exploration"
    imported_record = repositories.get_scenario(imported.scenario_id)
    assert imported_record is not None
    assert imported_record.type == "first_contact_exploration"
    content = json.loads(imported_record.content_json)
    assert content["mission_profile"] == "Survey the hidden ocean."
    assert content["translation_progress"] == (
        "Three descending pulses may mean open water."
    )
    assert content["hazards_and_escalation"] == "Thermal fissures are spreading."


def test_export_import_preserves_heist_infiltration_sections(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        scenario_id="scenario-heist",
        type="heist_infiltration",
        title="Skybank Treaty Job",
        premise="A crew must steal a treaty from a floating bank.",
        player_role="Crew planner",
        content={
            "title": "Skybank Treaty Job",
            "premise": "A crew must steal a treaty from a floating bank.",
            "player_character_name": "Mara Voss",
            "player_role": "Crew planner",
            "target_location": "Skybank vault above the storm moorings.",
            "objectives_and_stakes": "Recover the treaty and avoid war.",
            "intel_and_access": "Guard shift changes at bell three.",
            "security_model": "Clockwork cameras and a silent alarm.",
            "alert_and_heat": "Suspicion low; alarm inactive.",
            "loadout_and_tools": "Forged badges, lockpicks, smoke pellets.",
            "complications": "A rival crew shadows the job.",
            "extraction_routes": "Primary storm skiff; fallback service stairs.",
            "aftermath": "Clean success keeps heat low.",
            "tone_genre": "Tense caper with careful consequences.",
            "opening_message": "The skybank bell strikes three.",
        },
    )
    bundle_path = tmp_path / "skybank-treaty-job.bragi-scenario"
    service = _scenario_bundle_service(repositories)

    manifest = service.export_scenario(scenario.id, bundle_path)
    imported = service.import_scenario(bundle_path)

    assert manifest.scenario_type == "heist_infiltration"
    assert imported.scenario_type == "heist_infiltration"
    imported_record = repositories.get_scenario(imported.scenario_id)
    assert imported_record is not None
    assert imported_record.type == "heist_infiltration"
    content = json.loads(imported_record.content_json)
    assert content["security_model"] == "Clockwork cameras and a silent alarm."
    assert content["alert_and_heat"] == "Suspicion low; alarm inactive."
    assert content["extraction_routes"] == (
        "Primary storm skiff; fallback service stairs."
    )


@pytest.mark.parametrize(
    ("scenario_type", "content", "expected_field"),
    [
        (
            "settlement_builder",
            {
                "title": "Hearthstone Landing",
                "premise": "A flood-struck river town must survive its first year.",
                "player_character_name": "Mara Vale",
                "player_role": "Settlement steward",
                "settlement_profile": "A timber-and-stone river town.",
                "resources_and_indicators": "Food low; morale fragile.",
                "projects_and_facilities": "Repair the palisade and flood gate.",
                "threats_and_opportunities": "Spring floods and a grain compact.",
                "calendar_and_deadlines": "Flood season begins in sixteen days.",
                "tone_genre": "Grounded community survival.",
                "opening_message": "The river rises.",
            },
            "projects_and_facilities",
        ),
        (
            "monster_hunt_bounty",
            {
                "title": "The Thornback Contract",
                "premise": "A crew hunts a beast that learns from every trap.",
                "player_character_name": "Ira Voss",
                "player_role": "Monster tracker",
                "hunt_profile": "Find the Thornback before harvest road closes.",
                "target_profile": "The Thornback avoids firelight.",
                "leads_and_clues": "Blue sap marks its trail.",
                "hunt_locations": "Mill Creek and the old orchard.",
                "preparation_state": "Silver wire and oil snares.",
                "hunt_status": "Unresolved; target wounded but adapting.",
                "tone_genre": "Tense wilderness hunt.",
                "opening_message": "The newest tracks circle camp.",
            },
            "leads_and_clues",
        ),
        (
            "road_trip_pilgrimage",
            {
                "title": "Road to Saint Orra",
                "premise": "A divided party must reach the shrine before midsummer.",
                "player_character_name": "Nell Aran",
                "player_role": "Pilgrim guide",
                "journey_profile": "Carry a cracked bell relic to Saint Orra.",
                "route_and_stops": "Lantern Ford, Crow Market, hill shrine.",
                "transport_and_supplies": "One wagon, two mules, six days of oats.",
                "recurring_pressures": "Border patrols and summer storms.",
                "relationship_threads": "Tom doubts Sera.",
                "journey_progress": "Current leg: day one to Lantern Ford.",
                "tone_genre": "Warm travel drama.",
                "opening_message": "The shrine road begins.",
            },
            "journey_progress",
        ),
        (
            "merchant_trade_route",
            {
                "title": "Ledger Road",
                "premise": "A caravan must turn debt into profit.",
                "player_character_name": "Mara Den",
                "player_role": "Caravan factor",
                "trade_profile": "Run cedar oil from Kesh Gate to Red Harbor.",
                "cargo_inventory": "Cedar oil: 20 jars.",
                "markets_and_stops": "Red Harbor needs oil.",
                "contracts_and_debts": "Deliver ten jars in twelve days.",
                "route_hazards": "Tariff patrols and bridge bandits.",
                "profit_and_loss": "One lost crate erases profit.",
                "tone_genre": "Economy-lite caravan drama.",
                "opening_message": "The creditor stamps the contract.",
            },
            "contracts_and_debts",
        ),
    ],
)
def test_export_import_preserves_management_template_sections(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    scenario_type: str,
    content: dict[str, str],
    expected_field: str,
) -> None:
    scenario = repositories.create_scenario(
        scenario_id=f"scenario-{scenario_type}",
        type=scenario_type,
        title=content["title"],
        premise=content["premise"],
        player_role=content["player_role"],
        content=cast(dict[str, object], dict(content)),
    )
    bundle_path = tmp_path / f"{scenario_type}.bragi-scenario"
    service = _scenario_bundle_service(repositories)

    manifest = service.export_scenario(scenario.id, bundle_path)
    imported = service.import_scenario(bundle_path)

    assert manifest.scenario_type == scenario_type
    assert imported.scenario_type == scenario_type
    imported_record = repositories.get_scenario(imported.scenario_id)
    assert imported_record is not None
    assert imported_record.type == scenario_type
    imported_content = json.loads(imported_record.content_json)
    assert imported_content[expected_field] == content[expected_field]


def test_import_scenario_rejects_invalid_type_without_writing(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _scenario_bundle_module()
    bundle_path = tmp_path / "broken.bragi-scenario"
    manifest = _manifest_payload(
        scenario_id="scenario-broken",
        title="Broken",
        scenario_type="sandbox",
    )
    data = _data_payload(
        scenario_id="scenario-broken",
        title="Broken",
        scenario_type="sandbox",
    )
    _write_bundle(bundle_path, manifest=manifest, data=data)
    service = _scenario_bundle_service(repositories)

    with pytest.raises(module.ScenarioBundleError, match="Unsupported scenario type"):
        service.import_scenario(bundle_path)

    assert repositories.list_scenarios() == []


@pytest.mark.parametrize(
    ("scenario_type", "hybrid_types"),
    [
        ("character_interaction", None),
        ("dating_sim", ["dating_sim", "character_interaction"]),
    ],
)
def test_preview_and_import_reject_retired_character_interaction_without_writing(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    scenario_type: str,
    hybrid_types: list[str] | None,
) -> None:
    module = _scenario_bundle_module()
    bundle_path = tmp_path / "retired.bragi-scenario"
    manifest = _manifest_payload(
        scenario_id="scenario-retired",
        title="Retired",
        scenario_type=scenario_type,
    )
    data = _data_payload(
        scenario_id="scenario-retired",
        title="Retired",
        scenario_type=scenario_type,
    )
    if hybrid_types is not None:
        scenario = cast(dict[str, object], data["scenario"])
        content = cast(dict[str, object], scenario["content"])
        content["_scenario_genres"] = hybrid_types
    _write_bundle(bundle_path, manifest=manifest, data=data)
    service = _scenario_bundle_service(repositories)

    with pytest.raises(module.ScenarioBundleError, match="no longer supported"):
        service.preview_import(bundle_path)
    with pytest.raises(module.ScenarioBundleError, match="no longer supported"):
        service.import_scenario(bundle_path)

    assert repositories.list_scenarios() == []


def test_preview_and_import_reject_manifest_data_mismatch_without_writing(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _scenario_bundle_module()
    bundle_path = tmp_path / "mismatch.bragi-scenario"
    manifest = _manifest_payload(scenario_id="scenario-1", title="Ashfall Keep")
    data = _data_payload(scenario_id="scenario-2", title="Ashfall Keep")
    _write_bundle(bundle_path, manifest=manifest, data=data)
    service = _scenario_bundle_service(repositories)

    with pytest.raises(module.ScenarioBundleError, match="manifest does not match"):
        service.preview_import(bundle_path)
    with pytest.raises(module.ScenarioBundleError, match="manifest does not match"):
        service.import_scenario(bundle_path)

    assert repositories.list_scenarios() == []


def test_preview_and_import_reject_unsupported_version_and_future_schema(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _scenario_bundle_module()
    service = _scenario_bundle_service(repositories)
    unsupported_version = tmp_path / "unsupported.bragi-scenario"
    future_schema = tmp_path / "future.bragi-scenario"
    data = _data_payload(scenario_id="scenario-1", title="Ashfall Keep")
    version_manifest = _manifest_payload(
        scenario_id="scenario-1",
        title="Ashfall Keep",
    )
    version_manifest["bundle_version"] = 999
    schema_manifest = _manifest_payload(
        scenario_id="scenario-1",
        title="Ashfall Keep",
    )
    schema_manifest["bragi_schema_version"] = CURRENT_SCHEMA_VERSION + 1
    _write_bundle(unsupported_version, manifest=version_manifest, data=data)
    _write_bundle(future_schema, manifest=schema_manifest, data=data)

    with pytest.raises(module.ScenarioBundleError, match="Unsupported"):
        service.preview_import(unsupported_version)
    with pytest.raises(module.ScenarioBundleError, match="newer database schema"):
        service.import_scenario(future_schema)

    assert repositories.list_scenarios() == []


def test_preview_and_import_reject_unexpected_zip_members(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    module = _scenario_bundle_module()
    bundle_path = tmp_path / "padded.bragi-scenario"
    manifest = _manifest_payload(scenario_id="scenario-1", title="Ashfall Keep")
    data = _data_payload(scenario_id="scenario-1", title="Ashfall Keep")
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        bundle.writestr("data.json", json.dumps(data))
        bundle.writestr("ignored/padding.bin", b"x" * 128)
    service = _scenario_bundle_service(repositories)
    message = "Unexpected scenario bundle member"

    with pytest.raises(module.ScenarioBundleError, match=message):
        service.preview_import(bundle_path)
    with pytest.raises(module.ScenarioBundleError, match=message):
        service.import_scenario(bundle_path)

    assert repositories.list_scenarios() == []


def test_export_scenario_redacts_obvious_secret_values(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        scenario_id=SCENARIO_ID,
        type="full_roleplay",
        title="Oracle token=super-secret-token",
        premise="A meeting with api_key: sk-live-secret-value",
        player_role="Bearer live-secret-token investigator",
        content={
            "title": "Oracle token=super-secret-token",
            "setup_line": "A meeting with api_key: sk-live-secret-value",
            "player_role": "Bearer live-secret-token investigator",
            "api_key": "sk-live-secret-value",
            "opening_message": "Speak your token=super-secret-token.",
        },
    )
    bundle_path = tmp_path / "oracle.bragi-scenario"
    service = _scenario_bundle_service(repositories)

    service.export_scenario(scenario.id, bundle_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        payload = bundle.read("manifest.json") + bundle.read("data.json")
    assert b"super-secret-token" not in payload
    assert b"sk-live-secret-value" not in payload
    assert b"[redacted]" in payload


def _seed_scenario(repositories: PersistenceRepositories) -> Any:
    return repositories.create_scenario(
        scenario_id=SCENARIO_ID,
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash.",
        player_role="Signal warden",
        content={
            "title": "Ashfall Keep",
            "premise": "A border keep is cut off by ash.",
            "player_character_name": "Mara Voss",
            "player_role": "Signal warden",
            "worldbuilding": "Beacon keeps guard the ash road.",
            "lore": "The old signal code has been forgotten.",
            "locations": "Gatehouse, beacon tower, lower cistern.",
            "factions": "Wardens and ash smugglers.",
            "characters": "Captain Ilyra watches the wall.",
            "character_starters": [
                {
                    "name": "Captain Ilyra",
                    "aliases": ["Ilyra"],
                    "role": "Watch captain",
                    "known_state": "She watches the ash wall.",
                    "appearance": "Bronze cloak clasp and salt-stained boots.",
                    "visual_notes": "Straight silhouette in lighthouse glare.",
                    "personality": "Decisive and guarded.",
                    "voice": "Low clipped orders.",
                    "relationships": {"Mara Voss": "wary ally"},
                    "status": "waiting at the beacon",
                    "met": True,
                    "locked_fields": ["appearance"],
                }
            ],
            "tone_genre": "Tense frontier mystery.",
            "starting_scene": "Ash falls over the gatehouse.",
            "opening_message": "The beacon lens wakes red.",
            "_source": {
                "origin": "ai_draft",
                "generation_prompt": "A border keep cut off by ash storms.",
            },
        },
    )


def _scenario_bundle_service(
    repositories: PersistenceRepositories,
    *,
    media_dir: Path | None = None,
) -> Any:
    module = _scenario_bundle_module()
    return module.ScenarioBundleService(repositories=repositories, media_dir=media_dir)


def _scenario_bundle_module() -> Any:
    try:
        return importlib.import_module("bragi.services.scenario_bundle_service")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing bragi.services.scenario_bundle_service: {exc}")


def _manifest_payload(
    *,
    scenario_id: str,
    title: str,
    scenario_type: str = "full_roleplay",
) -> dict[str, object]:
    return {
        "format": "bragi-scenario-bundle",
        "bundle_format": "bragi-scenario-bundle",
        "bundle_version": 1,
        "title": title,
        "scenario_title": title,
        "scenario_type": scenario_type,
        "created_by": {"application": "Bragi", "version": "test"},
        "bragi_schema_version": CURRENT_SCHEMA_VERSION,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "exported_at": "2026-05-29T00:00:00+00:00",
        "scenario": {
            "id": scenario_id,
            "title": title,
            "type": scenario_type,
            "created_at": None,
            "updated_at": None,
        },
    }


def _data_payload(
    *,
    scenario_id: str,
    title: str,
    scenario_type: str = "full_roleplay",
) -> dict[str, object]:
    return {
        "scenario": {
            "id": scenario_id,
            "type": scenario_type,
            "title": title,
            "premise": "A border keep is cut off by ash.",
            "player_role": "Signal warden",
            "content": {
                "title": title,
                "premise": "A border keep is cut off by ash.",
                "player_role": "Signal warden",
                "opening_message": "The beacon lens wakes red.",
            },
            "created_at": None,
            "updated_at": None,
        }
    }


def _write_bundle(
    bundle_path: Path,
    *,
    manifest: dict[str, object],
    data: dict[str, object],
    media_members: list[tuple[str, bytes]] | None = None,
) -> None:
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        bundle.writestr("data.json", json.dumps(data))
        for name, member_bytes in media_members or []:
            bundle.writestr(name, member_bytes)
