from __future__ import annotations

import importlib
import json
import sqlite3
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories

SCENARIO_ID = "scenario-lantern"
SOURCE_SAVE_ID = "save-source"
TARGET_SAVE_ID = "save-target"
CHARACTER_ID = "character-mara"
REFERENCE_ASSET_ID = "media-mara-reference"
REFERENCE_PATH = "save-source/reference/mara.png"
THUMBNAIL_PATH = "save-source/reference/thumbnails/mara.png"
REFERENCE_BYTES = b"mara reference image bytes"
THUMBNAIL_BYTES = b"mara thumbnail bytes"


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_export_character_writes_profile_manifest_and_reference_media(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    bundle_path = tmp_path / "exports" / "mara.bragi-character"
    service = _character_bundle_service(repositories, media_dir)

    manifest = service.export_character(CHARACTER_ID, bundle_path)

    assert manifest.bundle_version == 1
    assert manifest.name == "Mara"
    assert manifest.media_count == 1
    with zipfile.ZipFile(bundle_path) as bundle:
        names = set(bundle.namelist())
        assert {"manifest.json", "data.json"}.issubset(names)
        media_names = sorted(name for name in names if name.startswith("media/"))
        assert len(media_names) == 2
        assert REFERENCE_BYTES in [bundle.read(name) for name in media_names]
        manifest_payload = json.loads(bundle.read("manifest.json"))
        assert manifest_payload["format"] == "bragi-character-bundle"
        assert manifest_payload["character"]["id"] == CHARACTER_ID
        assert manifest_payload["character"]["name"] == "Mara"
        assert manifest_payload["media_count"] == 1
        data = json.loads(bundle.read("data.json"))

    assert set(data) == {"character", "media_assets", "entity_links"}
    assert data["character"] == {
        "id": CHARACTER_ID,
        "name": "Mara",
        "aliases_json": '["Ember"]',
        "role": "Signal runner",
        "age": "early 30s",
        "known_state": "Carries the amber lens.",
        "history": "Carries the amber lens.",
        "met": 1,
        "appearance": "Ash-dusted cloak.",
        "visual_notes": "Warm lantern light.",
        "current_clothing": "Borrowed green raincoat over a linen shirt.",
        "personality": "Careful and dry-witted.",
        "voice": "Low and clipped.",
        "texting_style": "",
        "relationships_json": '{"Ren":"trusted rival"}',
        "goals": "Keep the amber lens out of Ren's hands.",
        "motivations": "Protect the signal villages.",
        "current_intent": "Negotiate passage without revealing the lens.",
        "boundaries": "Will not trade the amber lens.",
        "attitude_toward_player": "Trusts the player with caveats.",
        "cooperation_conditions": "Helps after the player names the north signal.",
        "status": "traveling",
        "location_id": None,
        "source_message_id": None,
        "locked_fields_json": '["name"]',
        "protected_from_maintenance": 1,
        "contact_name": "",
        "first_seen_message_id": None,
        "last_updated_message_id": None,
        "content_rating": "unclassified",
        "created_at": data["character"]["created_at"],
        "updated_at": data["character"]["updated_at"],
        "reference_image_asset_id": REFERENCE_ASSET_ID,
    }
    serialized_data = json.dumps(data, sort_keys=True)
    assert SOURCE_SAVE_ID not in serialized_data
    assert REFERENCE_PATH not in serialized_data
    assert THUMBNAIL_PATH not in serialized_data
    assert "source_save_id" not in serialized_data
    assert data["media_assets"][0]["id"] == REFERENCE_ASSET_ID
    assert "save_id" not in data["media_assets"][0]
    assert data["media_assets"][0]["path"] == "mara.png"
    assert data["media_assets"][0]["thumbnail_path"] == "mara.png"
    assert data["media_assets"][0]["source_message_id"] is None
    assert json.loads(data["media_assets"][0]["metadata_json"]) == {
        "kind": "character_reference",
        "character_id": CHARACTER_ID,
    }
    assert data["media_assets"][0]["files"]["path"]["byte_count"] == len(
        REFERENCE_BYTES
    )
    assert data["media_assets"][0]["files"]["thumbnail_path"]["byte_count"] == len(
        THUMBNAIL_BYTES
    )
    assert data["entity_links"] == [
        {
            "id": "link-mara-reference",
            "entity_type": "character",
            "entity_id": CHARACTER_ID,
            "target_type": "media_asset",
            "target_id": REFERENCE_ASSET_ID,
            "relation": "reference_image",
        }
    ]
    assert "private_notes" not in serialized_data


def test_export_character_includes_private_notes_when_explicitly_requested(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    bundle_path = tmp_path / "exports" / "mara-private.bragi-character"
    service = _character_bundle_service(repositories, media_dir)

    service.export_character(CHARACTER_ID, bundle_path, include_private_notes=True)

    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))

    assert data["character"]["private_notes"] == "Keep the lens secret."


def test_preview_import_suggests_unique_name_without_mutating_database(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    target_save = _seed_target_save(repositories)
    repositories.add_character(save_id=target_save.id, name="Mara")
    bundle_path = tmp_path / "mara.bragi-character"
    service = _character_bundle_service(repositories, media_dir)
    service.export_character(CHARACTER_ID, bundle_path)
    _inject_media_metadata(
        bundle_path,
        {
            "kind": "character_reference",
            "character_id": CHARACTER_ID,
            "source_save_id": SOURCE_SAVE_ID,
            "source_media_asset_id": REFERENCE_ASSET_ID,
        },
    )

    preview = service.preview_import(bundle_path, target_save_id=target_save.id)

    assert preview.name == "Mara"
    assert preview.suggested_name == "Mara (imported)"
    assert preview.name_conflict is True
    assert preview.media_count == 1
    assert preview.aliases == ("Ember",)
    assert preview.role == "Signal runner"
    assert preview.age == "early 30s"
    assert preview.known_state == "Carries the amber lens."
    assert preview.history == "Carries the amber lens."
    assert preview.appearance == "Ash-dusted cloak."
    assert preview.current_clothing == "Borrowed green raincoat over a linen shirt."
    assert preview.personality == "Careful and dry-witted."
    assert preview.voice == "Low and clipped."
    assert preview.status == "traveling"
    assert [
        character.name
        for character in repositories.list_characters(target_save.id)
    ] == ["Mara"]


def test_import_character_creates_new_profile_and_reference_image(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara.bragi-character"
    service = _character_bundle_service(repositories, media_dir)
    service.export_character(CHARACTER_ID, bundle_path)

    imported = service.import_character(
        bundle_path,
        target_save_id=target_save.id,
        name="Mara of the North",
    )

    assert imported.character_id != CHARACTER_ID
    assert imported.name == "Mara of the North"
    assert imported.media_count == 1
    assert imported.skipped_media_count == 0
    character = repositories.get_character(imported.character_id)
    assert character is not None
    assert character.age == "early 30s"
    assert character.save_id == target_save.id
    assert character.name == "Mara of the North"
    assert character.aliases == ["Ember"]
    assert character.known_state == "Carries the amber lens."
    assert character.history == "Carries the amber lens."
    assert character.relationships == {"Ren": "trusted rival"}
    assert character.goals == "Keep the amber lens out of Ren's hands."
    assert character.motivations == "Protect the signal villages."
    assert character.current_intent == "Negotiate passage without revealing the lens."
    assert character.boundaries == "Will not trade the amber lens."
    assert character.attitude_toward_player == "Trusts the player with caveats."
    assert character.cooperation_conditions == (
        "Helps after the player names the north signal."
    )
    assert character.location_id is None
    assert character.source_message_id is None
    assert character.first_seen_message_id is None
    assert character.last_updated_message_id is None
    assert character.locked_fields == ["name"]
    assert character.private_notes == ""
    assert character.protected_from_maintenance is True
    assert repositories.list_memories(target_save.id) == []
    assert repositories.list_world_state(target_save.id) == []

    [asset] = repositories.list_media_assets(target_save.id)
    assert asset.id != REFERENCE_ASSET_ID
    assert asset.source_message_id is None
    assert asset.source_media_asset_id is None
    assert asset.prompt == "Mara reference portrait"
    assert json.loads(asset.metadata_json) == {
        "kind": "character_reference",
        "character_id": imported.character_id,
    }
    assert (media_dir / asset.path).read_bytes() == REFERENCE_BYTES
    assert asset.thumbnail_path is not None
    assert (media_dir / asset.thumbnail_path).read_bytes() == THUMBNAIL_BYTES
    assert [
        (
            link.entity_type,
            link.entity_id,
            link.target_type,
            link.target_id,
            link.relation,
        )
        for link in repositories.list_entity_links(target_save.id)
    ] == [
        (
            "character",
            imported.character_id,
            "media_asset",
            asset.id,
            "reference_image",
        )
    ]


def test_import_character_preserves_legacy_private_notes_and_history(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara-legacy.bragi-character"
    service = _character_bundle_service(repositories, media_dir)
    service.export_character(CHARACTER_ID, bundle_path, include_private_notes=True)
    _rewrite_character_bundle_data(
        bundle_path=bundle_path,
        rewrite=lambda data: data["character"].pop("history", None),
    )

    imported = service.import_character(bundle_path, target_save_id=target_save.id)

    character = repositories.get_character(imported.character_id)
    assert character is not None
    assert character.private_notes == "Keep the lens secret."
    assert character.history == "Carries the amber lens."


def test_export_and_import_round_trips_character_contact_name(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    repositories.connection.execute(
        "UPDATE characters SET contact_name = ? WHERE id = ?",
        ("Em (signal runner)", CHARACTER_ID),
    )
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara-contact-name.bragi-character"
    service = _character_bundle_service(repositories, media_dir)

    service.export_character(CHARACTER_ID, bundle_path)
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["character"]["contact_name"] == "Em (signal runner)"

    imported = service.import_character(bundle_path, target_save_id=target_save.id)
    character = repositories.get_character(imported.character_id)
    assert character is not None
    assert character.contact_name == "Em (signal runner)"


def test_export_and_import_round_trips_character_texting_style(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    repositories.connection.execute(
        "UPDATE characters SET texting_style = ? WHERE id = ?",
        (
            "Fast double texts, lowercase, uses sun emoji only when relieved.",
            CHARACTER_ID,
        ),
    )
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara-texting-style.bragi-character"
    service = _character_bundle_service(repositories, media_dir)

    service.export_character(CHARACTER_ID, bundle_path)
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["character"]["texting_style"] == (
        "Fast double texts, lowercase, uses sun emoji only when relieved."
    )

    imported = service.import_character(bundle_path, target_save_id=target_save.id)
    character = repositories.get_character(imported.character_id)
    assert character is not None
    assert character.texting_style == (
        "Fast double texts, lowercase, uses sun emoji only when relieved."
    )


def test_export_and_import_round_trips_character_current_clothing(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    repositories.connection.execute(
        "UPDATE characters SET current_clothing = ? WHERE id = ?",
        ("Borrowed green raincoat over a linen shirt.", CHARACTER_ID),
    )
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara-current-clothing.bragi-character"
    service = _character_bundle_service(repositories, media_dir)

    service.export_character(CHARACTER_ID, bundle_path)
    with zipfile.ZipFile(bundle_path) as bundle:
        data = json.loads(bundle.read("data.json"))
    assert data["character"]["current_clothing"] == (
        "Borrowed green raincoat over a linen shirt."
    )

    imported = service.import_character(bundle_path, target_save_id=target_save.id)
    character = repositories.get_character(imported.character_id)
    assert character is not None
    assert character.current_clothing == "Borrowed green raincoat over a linen shirt."


def test_import_legacy_character_bundle_without_contact_name_defaults_blank(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara-legacy-contact.bragi-character"
    service = _character_bundle_service(repositories, media_dir)
    service.export_character(CHARACTER_ID, bundle_path)
    _rewrite_character_bundle_data(
        bundle_path=bundle_path,
        rewrite=lambda data: data["character"].pop("contact_name", None),
    )

    imported = service.import_character(bundle_path, target_save_id=target_save.id)
    character = repositories.get_character(imported.character_id)
    assert character is not None
    assert character.contact_name == ""


def test_import_legacy_character_bundle_without_texting_style_defaults_blank(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara-legacy-texting-style.bragi-character"
    service = _character_bundle_service(repositories, media_dir)
    service.export_character(CHARACTER_ID, bundle_path)
    _rewrite_character_bundle_data(
        bundle_path=bundle_path,
        rewrite=lambda data: data["character"].pop("texting_style", None),
    )

    imported = service.import_character(bundle_path, target_save_id=target_save.id)
    character = repositories.get_character(imported.character_id)
    assert character is not None
    assert character.texting_style == ""


def test_import_legacy_character_bundle_without_current_clothing_defaults_blank(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara-legacy-current-clothing.bragi-character"
    service = _character_bundle_service(repositories, media_dir)
    service.export_character(CHARACTER_ID, bundle_path)
    _rewrite_character_bundle_data(
        bundle_path=bundle_path,
        rewrite=lambda data: data["character"].pop("current_clothing", None),
    )

    imported = service.import_character(bundle_path, target_save_id=target_save.id)
    character = repositories.get_character(imported.character_id)
    assert character is not None
    assert character.current_clothing == ""


def test_import_legacy_character_bundle_without_agency_fields_defaults_empty(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara-no-agency.bragi-character"
    service = _character_bundle_service(repositories, media_dir)
    service.export_character(CHARACTER_ID, bundle_path)
    _rewrite_character_bundle_data(
        bundle_path,
        lambda data: [
            data["character"].pop(field, None)
            for field in (
                "goals",
                "motivations",
                "current_intent",
                "boundaries",
                "attitude_toward_player",
                "cooperation_conditions",
            )
        ],
    )

    imported = service.import_character(bundle_path, target_save_id=target_save.id)

    character = repositories.get_character(imported.character_id)
    assert character is not None
    assert character.goals == ""
    assert character.motivations == ""
    assert character.current_intent == ""
    assert character.boundaries == ""
    assert character.attitude_toward_player == ""
    assert character.cooperation_conditions == ""


@pytest.mark.parametrize("mime_type", ["text/html", "image/svg+xml", "video/mp4", None])
def test_import_character_stores_unsupported_media_mime_type_as_inert(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    mime_type: str | None,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara-active-mime.bragi-character"
    service = _character_bundle_service(repositories, media_dir)
    service.export_character(CHARACTER_ID, bundle_path)
    _rewrite_character_bundle_data(
        bundle_path,
        lambda data: data["media_assets"][0].__setitem__("mime_type", mime_type),
    )

    service.import_character(bundle_path, target_save_id=target_save.id)

    [asset] = repositories.list_media_assets(target_save.id)
    assert asset.mime_type == "application/octet-stream"


@pytest.mark.parametrize("mime_type", ["image/png", "image/jpeg", "image/webp"])
def test_import_character_preserves_supported_image_mime_types(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    mime_type: str,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara-image-mime.bragi-character"
    service = _character_bundle_service(repositories, media_dir)
    service.export_character(CHARACTER_ID, bundle_path)
    _rewrite_character_bundle_data(
        bundle_path,
        lambda data: data["media_assets"][0].__setitem__("mime_type", mime_type),
    )

    service.import_character(bundle_path, target_save_id=target_save.id)

    [asset] = repositories.list_media_assets(target_save.id)
    assert asset.mime_type == mime_type


def test_import_character_rejects_blank_name_without_mutating_database(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    _seed_source_character(repositories, media_dir)
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "mara.bragi-character"
    service = _character_bundle_service(repositories, media_dir)
    service.export_character(CHARACTER_ID, bundle_path)

    with pytest.raises(ValueError, match="Character name must not be blank"):
        service.import_character(bundle_path, target_save_id=target_save.id, name=" ")

    assert repositories.list_characters(target_save.id) == []
    assert repositories.list_media_assets(target_save.id) == []


def test_import_character_rejects_invalid_bundle_without_mutating_database(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    target_save = _seed_target_save(repositories)
    bundle_path = tmp_path / "broken.bragi-character"
    with zipfile.ZipFile(bundle_path, "w") as bundle:
        bundle.writestr("manifest.json", "{}")
        bundle.writestr("data.json", "{}")
    module = _character_bundle_module()
    service = module.CharacterBundleService(
        repositories=repositories,
        media_dir=media_dir,
    )

    with pytest.raises(module.CharacterBundleError):
        service.import_character(bundle_path, target_save_id=target_save.id)

    assert repositories.list_characters(target_save.id) == []


def _seed_source_character(
    repositories: PersistenceRepositories,
    media_dir: Path,
) -> None:
    scenario = repositories.create_scenario(
        scenario_id=SCENARIO_ID,
        type="character_interaction",
        title="Lantern Audience",
        premise="A watchtower audience.",
        player_role="Keeper",
        content={"character_name": "Mara"},
    )
    save = repositories.create_save(
        save_id=SOURCE_SAVE_ID,
        scenario_id=scenario.id,
        title="Source Save",
    )
    repositories.connection.execute(
        """
        INSERT INTO messages(id, save_id, role, speaker_name, body)
        VALUES ('message-source', ?, 'narrator', 'Narrator', 'Mara raises the lens.')
        """,
        (save.id,),
    )
    repositories.connection.execute(
        """
        INSERT INTO locations(
            id, save_id, name, aliases_json, description, visual_description,
            parent_location_id, connections_json, status, hazards_json,
            source_message_id, locked_fields_json
        )
        VALUES (
            'location-source', ?, 'Source Tower', '[]', '', '', NULL, '[]',
            '', '[]', 'message-source', '[]'
        )
        """,
        (save.id,),
    )
    repositories.connection.execute(
        """
        INSERT INTO characters(
            id, save_id, name, aliases_json, role, known_state, met,
            age, appearance, visual_notes, current_clothing, personality,
            voice, relationships_json,
            goals, motivations, current_intent, boundaries,
            attitude_toward_player, cooperation_conditions, status,
            location_id, private_notes, source_message_id,
            locked_fields_json, protected_from_maintenance,
            first_seen_message_id, last_updated_message_id
        )
        VALUES (
            ?, ?, 'Mara', '["Ember"]', 'Signal runner',
            'Carries the amber lens.', 1, 'early 30s', 'Ash-dusted cloak.',
            'Warm lantern light.', 'Borrowed green raincoat over a linen shirt.',
            'Careful and dry-witted.',
            'Low and clipped.', '{"Ren":"trusted rival"}',
            'Keep the amber lens out of Ren''s hands.',
            'Protect the signal villages.',
            'Negotiate passage without revealing the lens.',
            'Will not trade the amber lens.',
            'Trusts the player with caveats.',
            'Helps after the player names the north signal.', 'traveling',
            'location-source', 'Keep the lens secret.', 'message-source',
            '["name","location_id"]', 1, 'message-source', 'message-source'
        )
        """,
        (CHARACTER_ID, save.id),
    )
    repositories.create_media_asset(
        asset_id=REFERENCE_ASSET_ID,
        save_id=save.id,
        source_message_id=None,
        type="image",
        path=REFERENCE_PATH,
        thumbnail_path=THUMBNAIL_PATH,
        prompt="Mara reference portrait",
        provider="fake",
        model="fake-image",
        status="succeeded",
        mime_type="image/png",
        metadata={
            "kind": "character_reference",
            "character_id": CHARACTER_ID,
            "source": "continuation_clone",
            "source_save_id": SOURCE_SAVE_ID,
            "source_media_asset_id": "source-media-asset",
        },
    )
    reference_path = media_dir / REFERENCE_PATH
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_bytes(REFERENCE_BYTES)
    thumbnail_path = media_dir / THUMBNAIL_PATH
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_path.write_bytes(THUMBNAIL_BYTES)
    repositories.add_entity_link(
        link_id="link-mara-reference",
        save_id=save.id,
        entity_type="character",
        entity_id=CHARACTER_ID,
        target_type="media_asset",
        target_id=REFERENCE_ASSET_ID,
        relation="reference_image",
    )
    repositories.add_memory(
        save_id=save.id,
        body="This source memory should not be exported with the character.",
        tags=["private"],
        importance=0.9,
    )
    repositories.connection.commit()


def _seed_target_save(repositories: PersistenceRepositories) -> Any:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Target Keep",
        premise="A different tower.",
        player_role="Keeper",
        content={},
    )
    return repositories.create_save(
        save_id=TARGET_SAVE_ID,
        scenario_id=scenario.id,
        title="Target Save",
    )


def _character_bundle_service(
    repositories: PersistenceRepositories,
    media_dir: Path,
) -> Any:
    module = _character_bundle_module()
    return module.CharacterBundleService(repositories=repositories, media_dir=media_dir)


def _character_bundle_module() -> Any:
    try:
        return importlib.import_module("bragi.services.character_bundle_service")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing bragi.services.character_bundle_service: {exc}")


def _inject_media_metadata(
    bundle_path: Path,
    metadata: dict[str, object],
) -> None:
    _rewrite_character_bundle_data(
        bundle_path,
        lambda data: data["media_assets"][0].__setitem__(
            "metadata_json",
            json.dumps(metadata),
        ),
    )


def _rewrite_character_bundle_data(
    bundle_path: Path,
    rewrite: Any,
) -> None:
    with zipfile.ZipFile(bundle_path) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
    data = json.loads(members["data.json"])
    rewrite(data)
    members["data.json"] = json.dumps(data).encode("utf-8")
    with zipfile.ZipFile(bundle_path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
