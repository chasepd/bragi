"""Portable character import and export bundles."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any, cast
from uuid import uuid4

from bragi import __version__
from bragi.app_logging import log_event
from bragi.persistence.migrations import CURRENT_SCHEMA_VERSION
from bragi.persistence.repositories import PersistenceRepositories
from bragi.private_files import write_private_bytes
from bragi.services.character_locks import normalize_character_locked_fields
from bragi.zip_safety import ZipSafetyError, validate_zip_directory
from bragi_common.media_mime import imported_media_mime_type

CHARACTER_BUNDLE_FORMAT = "bragi-character-bundle"
CHARACTER_BUNDLE_VERSION = 1
MANIFEST_NAME = "manifest.json"
DATA_NAME = "data.json"
_MAX_CHARACTER_BUNDLE_JSON_BYTES = 5 * 1024 * 1024
_MAX_CHARACTER_BUNDLE_MEDIA_FILE_BYTES = 25 * 1024 * 1024
_MAX_CHARACTER_BUNDLE_MEDIA_TOTAL_BYTES = 250 * 1024 * 1024
_REFERENCE_RELATION = "reference_image"
_NON_PORTABLE_LOCKED_FIELDS = frozenset(
    {
        "location_id",
        "source_message_id",
        "first_seen_message_id",
        "last_updated_message_id",
    }
)


class CharacterBundleError(ValueError):
    """Raised when a character bundle is invalid or unsupported."""


@dataclass(frozen=True)
class CharacterBundleManifest:
    bundle_format: str
    bundle_version: int
    character_id: str
    name: str
    media_count: int
    created_at: str | None
    updated_at: str | None
    exported_at: str


@dataclass(frozen=True)
class CharacterBundlePreview:
    character_id: str
    name: str
    suggested_name: str
    name_conflict: bool
    media_count: int
    bundle_version: int
    aliases: tuple[str, ...] = ()
    role: str = ""
    age: str = ""
    known_state: str = ""
    history: str = ""
    appearance: str = ""
    current_clothing: str = ""
    personality: str = ""
    voice: str = ""
    texting_style: str = ""
    goals: str = ""
    motivations: str = ""
    current_intent: str = ""
    boundaries: str = ""
    attitude_toward_player: str = ""
    cooperation_conditions: str = ""
    status: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    exported_at: str | None = None
    skipped_media_count: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportedCharacterBundle:
    character_id: str
    name: str
    media_count: int
    skipped_media_count: int


class CharacterBundleService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        media_dir: Path,
    ) -> None:
        self.repositories = repositories
        self.media_dir = media_dir

    def export_character(
        self,
        character_id: str,
        bundle_path: Path,
        *,
        include_private_notes: bool = False,
    ) -> CharacterBundleManifest:
        row = _require_row(
            self.repositories.connection.execute(
                """
                SELECT id, save_id, name, aliases_json, role, age, known_state,
                       history, met, appearance, visual_notes, current_clothing,
                       personality, voice, texting_style, relationships_json, goals,
                       motivations,
                       current_intent, boundaries, attitude_toward_player,
                       cooperation_conditions, status, location_id, private_notes,
                       source_message_id, locked_fields_json,
                       protected_from_maintenance, contact_name,
                       first_seen_message_id, last_updated_message_id,
                       content_rating, created_at, updated_at
                FROM characters
                WHERE id = ? AND archived_at IS NULL
                """,
                (character_id,),
            ).fetchone(),
            f"Unknown character id: {character_id}",
        )
        save_id = str(row["save_id"])
        character = _character_payload(
            row,
            include_private_notes=include_private_notes,
        )
        media_assets, entity_links, media_files = self._reference_media_rows(
            save_id=save_id,
            character_id=character_id,
        )
        if entity_links:
            character["reference_image_asset_id"] = entity_links[0]["target_id"]
        exported_at = datetime.now(UTC).isoformat()
        manifest_payload: dict[str, object] = {
            "format": CHARACTER_BUNDLE_FORMAT,
            "bundle_format": CHARACTER_BUNDLE_FORMAT,
            "bundle_version": CHARACTER_BUNDLE_VERSION,
            "name": character["name"],
            "character_name": character["name"],
            "media_count": len(media_assets),
            "created_by": {
                "application": "Bragi",
                "version": __version__,
            },
            "bragi_schema_version": CURRENT_SCHEMA_VERSION,
            "schema_version": CURRENT_SCHEMA_VERSION,
            "exported_at": exported_at,
            "character": {
                "id": character["id"],
                "name": character["name"],
                "created_at": character["created_at"],
                "updated_at": character["updated_at"],
            },
        }
        data: dict[str, object] = {
            "character": character,
            "media_assets": media_assets,
            "entity_links": entity_links,
        }
        _verify_collected_media_files(data["media_assets"], media_files)
        manifest = _manifest_from_payload(manifest_payload)
        _write_bundle_atomically(
            bundle_path=bundle_path,
            manifest_payload=manifest_payload,
            data=data,
            media_files=media_files,
        )
        log_event(
            "character_bundle.exported",
            character_id=character_id,
            bundle_path=str(bundle_path),
            media_count=manifest.media_count,
        )
        return manifest

    def preview_import(
        self,
        bundle_path: Path,
        *,
        target_save_id: str | None = None,
    ) -> CharacterBundlePreview:
        manifest_payload, data = self._read_bundle(bundle_path)
        manifest = _manifest_from_payload(manifest_payload)
        if target_save_id is not None:
            _require_target_save(self.repositories, target_save_id)
        character_data = _object(data.get("character"), "character")
        suggested_name = _unique_character_name(
            self.repositories,
            target_save_id=target_save_id,
            desired_name=manifest.name,
        )
        warnings = _preview_warnings(data)
        return CharacterBundlePreview(
            character_id=manifest.character_id,
            name=manifest.name,
            suggested_name=suggested_name,
            name_conflict=suggested_name.casefold() != manifest.name.strip().casefold(),
            media_count=manifest.media_count,
            bundle_version=manifest.bundle_version,
            aliases=tuple(_json_string_list(character_data, "aliases_json")),
            role=_text(character_data, "role").strip(),
            age=(_optional_text(character_data, "age") or "").strip(),
            known_state=_character_history_payload(character_data),
            history=_character_history_payload(character_data),
            appearance=_text(character_data, "appearance").strip(),
            current_clothing=(
                _optional_text(character_data, "current_clothing") or ""
            ).strip(),
            personality=_text(character_data, "personality").strip(),
            voice=_text(character_data, "voice").strip(),
            texting_style=(
                _optional_text(character_data, "texting_style") or ""
            ).strip(),
            goals=_optional_text(character_data, "goals") or "",
            motivations=_optional_text(character_data, "motivations") or "",
            current_intent=_optional_text(character_data, "current_intent") or "",
            boundaries=_optional_text(character_data, "boundaries") or "",
            attitude_toward_player=(
                _optional_text(character_data, "attitude_toward_player") or ""
            ),
            cooperation_conditions=(
                _optional_text(character_data, "cooperation_conditions") or ""
            ),
            status=_text(character_data, "status").strip(),
            created_at=manifest.created_at,
            updated_at=manifest.updated_at,
            exported_at=manifest.exported_at,
            skipped_media_count=0,
            warnings=warnings,
        )

    def import_character(
        self,
        bundle_path: Path,
        *,
        target_save_id: str,
        name: str | None = None,
    ) -> ImportedCharacterBundle:
        _require_target_save(self.repositories, target_save_id)
        manifest_payload, data = self._read_bundle(bundle_path)
        manifest = _manifest_from_payload(manifest_payload)
        import_name = (
            name.strip()
            if name is not None
            else _unique_character_name(
                self.repositories,
                target_save_id=target_save_id,
                desired_name=manifest.name,
            )
        )
        if not import_name:
            raise ValueError("Character name must not be blank")
        media_payloads = _load_media_payloads(bundle_path, data)
        media_backups: dict[Path, bytes | None] = {}
        try:
            self.repositories.begin_transaction()
            imported = self._import_data(
                data,
                target_save_id=target_save_id,
                name=import_name,
                media_payloads=media_payloads,
                media_backups=media_backups,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            _restore_media_backups(media_backups)
            raise
        log_event(
            "character_bundle.imported",
            character_id=imported.character_id,
            target_save_id=target_save_id,
            media_count=imported.media_count,
            skipped_media_count=imported.skipped_media_count,
        )
        return imported

    def _import_data(
        self,
        data: dict[str, object],
        *,
        target_save_id: str,
        name: str,
        media_payloads: dict[tuple[str, str], bytes],
        media_backups: dict[Path, bytes | None],
    ) -> ImportedCharacterBundle:
        character_data = _object(data.get("character"), "character")
        history = _character_history_payload(character_data)
        character = self.repositories.add_character(
            save_id=target_save_id,
            name=name,
            aliases=_json_string_list(character_data, "aliases_json"),
            role=_text(character_data, "role").strip(),
            age=(_optional_text(character_data, "age") or "").strip(),
            known_state=history,
            history=history,
            met=_boolish(character_data, "met"),
            appearance=_text(character_data, "appearance").strip(),
            visual_notes=_text(character_data, "visual_notes").strip(),
            current_clothing=(
                _optional_text(character_data, "current_clothing") or ""
            ).strip(),
            personality=_text(character_data, "personality").strip(),
            voice=_text(character_data, "voice").strip(),
            texting_style=(
                _optional_text(character_data, "texting_style") or ""
            ).strip(),
            relationships=_json_object(character_data, "relationships_json"),
            goals=(_optional_text(character_data, "goals") or "").strip(),
            motivations=(
                _optional_text(character_data, "motivations") or ""
            ).strip(),
            current_intent=(
                _optional_text(character_data, "current_intent") or ""
            ).strip(),
            boundaries=(_optional_text(character_data, "boundaries") or "").strip(),
            attitude_toward_player=(
                _optional_text(character_data, "attitude_toward_player") or ""
            ).strip(),
            cooperation_conditions=(
                _optional_text(character_data, "cooperation_conditions") or ""
            ).strip(),
            status=_text(character_data, "status").strip(),
            location_id=None,
            private_notes=(
                _optional_text(character_data, "private_notes") or ""
            ).strip(),
            contact_name=(
                _optional_text(character_data, "contact_name") or ""
            ).strip(),
            source_message_id=None,
            locked_fields=[
                field
                for field in normalize_character_locked_fields(
                    _json_string_list(character_data, "locked_fields_json")
                )
                if field not in _NON_PORTABLE_LOCKED_FIELDS
            ],
            protected_from_maintenance=_boolish(
                character_data,
                "protected_from_maintenance",
            ),
            first_seen_message_id=None,
            last_updated_message_id=None,
            content_rating="unclassified",
        )
        imported_media_count = 0
        skipped_media_count = 0
        for row in _list_of_objects(data.get("media_assets"), "media_assets"):
            original_asset_id = _text(row, "id")
            payload = media_payloads.get((original_asset_id, "path"))
            if payload is None:
                skipped_media_count += 1
                continue
            asset_id = uuid4().hex
            relative_path = _imported_media_relative_path(
                save_id=target_save_id,
                asset_id=asset_id,
                original_path=_text(row, "path"),
            )
            output_path = self.media_dir / relative_path
            _remember_media_backup(media_backups, output_path)
            write_private_bytes(output_path, payload)

            thumbnail_relative_path: str | None = None
            thumbnail_payload = media_payloads.get(
                (original_asset_id, "thumbnail_path")
            )
            if thumbnail_payload is not None:
                thumbnail_relative_path = _imported_thumbnail_relative_path(
                    save_id=target_save_id,
                    asset_id=asset_id,
                    original_path=_optional_text(row, "thumbnail_path")
                    or _text(row, "path"),
                ).as_posix()
                thumbnail_path = self.media_dir / thumbnail_relative_path
                _remember_media_backup(media_backups, thumbnail_path)
                write_private_bytes(thumbnail_path, thumbnail_payload)

            asset = self.repositories.create_media_asset(
                save_id=target_save_id,
                source_message_id=None,
                type="image",
                path=relative_path.as_posix(),
                thumbnail_path=thumbnail_relative_path,
                prompt=_text(row, "prompt"),
                provider=_text(row, "provider"),
                model=_text(row, "model"),
                status=_text(row, "status"),
                mime_type=imported_media_mime_type(
                    _optional_text(row, "mime_type"),
                    media_type="image",
                ),
                metadata={
                    "kind": "character_reference",
                    "character_id": character.id,
                },
                source_media_asset_id=None,
                asset_id=asset_id,
            )
            self.repositories.add_entity_link(
                save_id=target_save_id,
                entity_type="character",
                entity_id=character.id,
                target_type="media_asset",
                target_id=asset.id,
                relation=_REFERENCE_RELATION,
            )
            imported_media_count += 1
        return ImportedCharacterBundle(
            character_id=character.id,
            name=character.name,
            media_count=imported_media_count,
            skipped_media_count=skipped_media_count,
        )

    def _read_bundle(
        self,
        bundle_path: Path,
    ) -> tuple[dict[str, object], dict[str, object]]:
        try:
            validate_zip_directory(bundle_path)
            with zipfile.ZipFile(bundle_path) as bundle:
                manifest = _json_object_from_bytes(
                    _read_limited_member(bundle, MANIFEST_NAME),
                    MANIFEST_NAME,
                )
                _validate_manifest_payload(manifest)
                data = _json_object_from_bytes(
                    _read_limited_member(bundle, DATA_NAME),
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
            raise CharacterBundleError("Invalid Bragi character bundle") from exc

    def _reference_media_rows(
        self,
        *,
        save_id: str,
        character_id: str,
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, Path],
    ]:
        character_links = [
            link
            for link in self.repositories.list_entity_links(save_id)
            if (
                link.entity_type == "character"
                and link.entity_id == character_id
                and link.target_type == "media_asset"
                and link.relation == _REFERENCE_RELATION
            )
        ]
        character_link_targets = {link.target_id for link in character_links}
        links_by_asset_id = {link.target_id: link for link in character_links}
        media_rows: list[dict[str, object]] = []
        entity_links: list[dict[str, object]] = []
        media_files: dict[str, Path] = {}
        for asset in self.repositories.list_media_assets(save_id):
            metadata = _media_asset_metadata(asset.metadata_json)
            linked_to_character = asset.id in character_link_targets
            tagged_for_character = (
                metadata.get("kind") == "character_reference"
                and metadata.get("character_id") == character_id
            )
            if not linked_to_character and not tagged_for_character:
                continue
            if not _is_portable_reference_asset(asset):
                continue
            row, row_media_files = _media_asset_row(
                asset,
                media_dir=self.media_dir,
                character_id=character_id,
            )
            if "path" not in _object(row.get("files"), "media asset files"):
                continue
            media_rows.append(row)
            media_files.update(row_media_files)
            link = links_by_asset_id.get(asset.id)
            entity_links.append(
                {
                    "id": (
                        link.id
                        if link is not None
                        else f"reference:{character_id}:{asset.id}"
                    ),
                    "entity_type": "character",
                    "entity_id": character_id,
                    "target_type": "media_asset",
                    "target_id": asset.id,
                    "relation": _REFERENCE_RELATION,
                }
            )
        return media_rows, entity_links, media_files


def _write_bundle_atomically(
    *,
    bundle_path: Path,
    manifest_payload: dict[str, object],
    data: dict[str, object],
    media_files: dict[str, Path],
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
            for bundle_name, local_path in media_files.items():
                bundle.write(local_path, bundle_name)
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


def _load_media_payloads(
    bundle_path: Path,
    data: dict[str, object],
) -> dict[tuple[str, str], bytes]:
    payloads: dict[tuple[str, str], bytes] = {}
    total_byte_count = 0
    try:
        with zipfile.ZipFile(bundle_path) as bundle:
            for row in _list_of_objects(data.get("media_assets"), "media_assets"):
                asset_id = _text(row, "id")
                files = _object(row.get("files"), "media asset files")
                for field in ("path", "thumbnail_path"):
                    metadata = files.get(field)
                    if metadata is None:
                        continue
                    if not isinstance(metadata, dict):
                        raise CharacterBundleError(
                            f"Invalid media file metadata for {asset_id}:{field}"
                        )
                    bundle_name = metadata.get("bundle_path")
                    if not isinstance(bundle_name, str) or not bundle_name:
                        raise CharacterBundleError(
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
                    if expected_byte_count > _MAX_CHARACTER_BUNDLE_MEDIA_FILE_BYTES:
                        raise CharacterBundleError(
                            "Media file is too large in character bundle: "
                            f"{bundle_name}"
                        )
                    total_byte_count += expected_byte_count
                    if total_byte_count > _MAX_CHARACTER_BUNDLE_MEDIA_TOTAL_BYTES:
                        raise CharacterBundleError(
                            "Character bundle media is too large"
                        )
                    try:
                        info = bundle.getinfo(bundle_name)
                    except KeyError:
                        raise CharacterBundleError(
                            f"Missing media file in character bundle: {bundle_name}"
                        ) from None
                    if info.file_size != expected_byte_count:
                        raise CharacterBundleError(
                            f"Media byte count mismatch for {bundle_name}"
                        )
                    payload = bundle.read(info)
                    if hashlib.sha256(payload).hexdigest() != expected_sha:
                        raise CharacterBundleError(
                            f"Media checksum mismatch for {bundle_name}"
                        )
                    payloads[(asset_id, field)] = payload
    except (OSError, zipfile.BadZipFile) as exc:
        raise CharacterBundleError("Invalid Bragi character bundle") from exc
    return payloads


def _verify_collected_media_files(
    media_assets: object,
    media_files: dict[str, Path],
) -> None:
    total_byte_count = 0
    for row in _list_of_objects(media_assets, "media_assets"):
        asset_id = _text(row, "id")
        row_files = _object(row.get("files"), "media asset files")
        for field, metadata in row_files.items():
            if field not in {"path", "thumbnail_path"}:
                continue
            if not isinstance(metadata, dict):
                raise CharacterBundleError(
                    f"Invalid media file metadata for {asset_id}:{field}"
                )
            bundle_path = metadata.get("bundle_path")
            if not isinstance(bundle_path, str):
                raise CharacterBundleError(
                    f"Invalid media file metadata for {asset_id}:{field}"
                )
            expected_sha = _required_metadata_text(
                metadata,
                "sha256",
                bundle_path,
            )
            expected_byte_count = _required_metadata_int(
                metadata,
                "byte_count",
                bundle_path,
            )
            local_path = media_files.get(bundle_path)
            if local_path is None or not local_path.is_file():
                raise CharacterBundleError(
                    f"Media file disappeared during export: {bundle_path}"
                )
            actual_byte_count = local_path.stat().st_size
            if actual_byte_count != expected_byte_count:
                raise CharacterBundleError(
                    f"Media file changed during export: {bundle_path}"
                )
            if _sha256(local_path) != expected_sha:
                raise CharacterBundleError(
                    f"Media file changed during export: {bundle_path}"
                )
            total_byte_count += actual_byte_count
            if total_byte_count > _MAX_CHARACTER_BUNDLE_MEDIA_TOTAL_BYTES:
                raise CharacterBundleError(
                    "Character bundle media is too large to export"
                )


def _read_limited_member(bundle: zipfile.ZipFile, name: str) -> bytes:
    info = bundle.getinfo(name)
    if info.file_size > _MAX_CHARACTER_BUNDLE_JSON_BYTES:
        raise CharacterBundleError(f"Character bundle {name} is too large")
    return bundle.read(info)


def _validate_bundle_members(
    bundle: zipfile.ZipFile,
    data: dict[str, object],
) -> None:
    expected = {MANIFEST_NAME, DATA_NAME}
    total_byte_count = 0
    for row in _list_of_objects(data.get("media_assets"), "media_assets"):
        asset_id = _text(row, "id")
        files = _object(row.get("files"), "media asset files")
        for field in ("path", "thumbnail_path"):
            metadata = files.get(field)
            if metadata is None:
                continue
            if not isinstance(metadata, dict):
                raise CharacterBundleError(
                    f"Invalid media file metadata for {asset_id}:{field}"
                )
            bundle_name = metadata.get("bundle_path")
            if not isinstance(bundle_name, str) or not bundle_name:
                raise CharacterBundleError(
                    f"Missing media bundle_path metadata for {asset_id}:{field}"
                )
            _validate_bundle_member_name(bundle_name)
            expected_byte_count = _required_metadata_int(
                metadata,
                "byte_count",
                bundle_name,
            )
            if expected_byte_count > _MAX_CHARACTER_BUNDLE_MEDIA_FILE_BYTES:
                raise CharacterBundleError(
                    f"Media file is too large in character bundle: {bundle_name}"
                )
            total_byte_count += expected_byte_count
            if total_byte_count > _MAX_CHARACTER_BUNDLE_MEDIA_TOTAL_BYTES:
                raise CharacterBundleError("Character bundle media is too large")
            try:
                info = bundle.getinfo(bundle_name)
            except KeyError:
                raise CharacterBundleError(
                    f"Missing media file in character bundle: {bundle_name}"
                ) from None
            if info.file_size != expected_byte_count:
                raise CharacterBundleError(
                    f"Media byte count mismatch for {bundle_name}"
                )
            expected.add(bundle_name)

    seen: set[str] = set()
    for info in bundle.infolist():
        if info.filename in seen:
            raise CharacterBundleError(
                f"Duplicate character bundle member: {info.filename}"
            )
        seen.add(info.filename)
        if info.filename not in expected:
            raise CharacterBundleError(
                f"Unexpected character bundle member: {info.filename}"
            )


def _validate_bundle_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name.startswith("media/")
        or path.is_absolute()
        or ".." in path.parts
        or "" in path.parts
    ):
        raise CharacterBundleError("Invalid media path in character bundle")


def _validate_bundle_data(
    manifest: dict[str, object],
    data: dict[str, object],
) -> None:
    character = _object(data.get("character"), "character")
    manifest_character = _object(manifest.get("character"), "manifest character")
    if _text(character, "id") != _text(manifest_character, "id"):
        raise CharacterBundleError("Character bundle manifest does not match data")
    if _text(character, "name") != _text(manifest_character, "name"):
        raise CharacterBundleError("Character bundle manifest does not match data")
    media_assets = _list_of_objects(data.get("media_assets"), "media_assets")
    if _int(manifest, "media_count") != len(media_assets):
        raise CharacterBundleError(
            "Character bundle manifest media count does not match data"
        )
    _json_string_list(character, "aliases_json")
    _json_object(character, "relationships_json")
    _json_string_list(character, "locked_fields_json")
    _validate_duplicate_ids(media_assets, "media asset")
    media_asset_ids = {_text(row, "id") for row in media_assets}
    reference_image_asset_id = _optional_text(character, "reference_image_asset_id")
    if (
        reference_image_asset_id is not None
        and reference_image_asset_id not in media_asset_ids
    ):
        raise CharacterBundleError("Character bundle reference image is missing")
    for link in _list_of_objects(data.get("entity_links"), "entity_links"):
        if (
            _text(link, "entity_type") != "character"
            or _text(link, "target_type") != "media_asset"
            or _text(link, "relation") != _REFERENCE_RELATION
            or _text(link, "entity_id") != _text(character, "id")
            or _text(link, "target_id") not in media_asset_ids
        ):
            raise CharacterBundleError("Invalid character bundle reference link")


def _validate_duplicate_ids(rows: list[dict[str, object]], label: str) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = _text(row, "id")
        if row_id in seen:
            raise CharacterBundleError(
                f"Duplicate {label} id in character bundle: {row_id}"
            )
        seen.add(row_id)


def _validate_manifest_payload(payload: dict[str, object]) -> None:
    if payload.get("format") != CHARACTER_BUNDLE_FORMAT:
        raise CharacterBundleError("Not a Bragi character bundle")
    version = payload.get("bundle_version")
    if version != CHARACTER_BUNDLE_VERSION:
        raise CharacterBundleError(
            f"Unsupported Bragi character bundle version: {version}"
        )
    schema_version = payload.get("bragi_schema_version", payload.get("schema_version"))
    if (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version > CURRENT_SCHEMA_VERSION
    ):
        raise CharacterBundleError(
            "Bragi character bundle requires a newer database schema: "
            f"{schema_version}"
        )
    _int(payload, "media_count")
    _object(payload.get("character"), "manifest character")


def _manifest_from_payload(payload: dict[str, object]) -> CharacterBundleManifest:
    _validate_manifest_payload(payload)
    character = _object(payload.get("character"), "manifest character")
    return CharacterBundleManifest(
        bundle_format=CHARACTER_BUNDLE_FORMAT,
        bundle_version=CHARACTER_BUNDLE_VERSION,
        character_id=_text(character, "id"),
        name=_text(character, "name"),
        media_count=_int(payload, "media_count"),
        created_at=_optional_text(character, "created_at"),
        updated_at=_optional_text(character, "updated_at"),
        exported_at=_text(payload, "exported_at"),
    )


def _character_payload(
    row: Any,
    *,
    include_private_notes: bool = False,
) -> dict[str, object]:
    locked_fields = [
        field
        for field in normalize_character_locked_fields(
            _json_string_list(dict(row), "locked_fields_json")
        )
        if field not in _NON_PORTABLE_LOCKED_FIELDS
    ]
    payload: dict[str, object] = {
        "id": row["id"],
        "name": row["name"],
        "aliases_json": row["aliases_json"],
        "role": row["role"],
        "age": row["age"],
        "known_state": row["history"] or row["known_state"],
        "history": row["history"] or row["known_state"],
        "met": row["met"],
        "appearance": row["appearance"],
        "visual_notes": row["visual_notes"],
        "current_clothing": row["current_clothing"],
        "personality": row["personality"],
        "voice": row["voice"],
        "texting_style": row["texting_style"],
        "relationships_json": row["relationships_json"],
        "goals": row["goals"],
        "motivations": row["motivations"],
        "current_intent": row["current_intent"],
        "boundaries": row["boundaries"],
        "attitude_toward_player": row["attitude_toward_player"],
        "cooperation_conditions": row["cooperation_conditions"],
        "status": row["status"],
        "location_id": None,
        "source_message_id": None,
        "locked_fields_json": json.dumps(locked_fields),
        "protected_from_maintenance": row["protected_from_maintenance"],
        "contact_name": row["contact_name"],
        "first_seen_message_id": None,
        "last_updated_message_id": None,
        "content_rating": row["content_rating"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if include_private_notes:
        payload["private_notes"] = row["private_notes"]
    _json_string_list(payload, "aliases_json")
    _json_object(payload, "relationships_json")
    return payload


def _media_asset_row(
    asset: Any,
    *,
    media_dir: Path,
    character_id: str,
) -> tuple[dict[str, object], dict[str, Path]]:
    row: dict[str, object] = {
        "id": asset.id,
        "source_message_id": None,
        "type": asset.type,
        "path": _safe_filename(asset.path),
        "thumbnail_path": (
            _safe_filename(asset.thumbnail_path) if asset.thumbnail_path else None
        ),
        "prompt": asset.prompt,
        "provider": asset.provider,
        "model": asset.model,
        "status": asset.status,
        "mime_type": asset.mime_type,
        "metadata_json": json.dumps(
            {
                "kind": "character_reference",
                "character_id": character_id,
            }
        ),
        "source_media_asset_id": None,
        "created_at": asset.created_at,
        "archived_at": asset.archived_at,
    }
    files: dict[str, dict[str, object]] = {}
    media_files: dict[str, Path] = {}
    total_byte_count = 0
    for field, persisted_path in (
        ("path", asset.path),
        ("thumbnail_path", asset.thumbnail_path),
    ):
        if not isinstance(persisted_path, str) or not persisted_path:
            continue
        local_path = _media_path(media_dir, persisted_path)
        if local_path is None or not local_path.is_file():
            continue
        byte_count = local_path.stat().st_size
        if byte_count > _MAX_CHARACTER_BUNDLE_MEDIA_FILE_BYTES:
            raise CharacterBundleError(
                "Media file is too large for character bundle export: "
                f"{persisted_path}"
            )
        total_byte_count += byte_count
        if total_byte_count > _MAX_CHARACTER_BUNDLE_MEDIA_TOTAL_BYTES:
            raise CharacterBundleError(
                "Character bundle media is too large to export"
            )
        checksum = _sha256(local_path)
        bundle_path = _bundle_media_path(row["id"], field, persisted_path)
        files[field] = {
            "bundle_path": bundle_path,
            "sha256": checksum,
            "byte_count": byte_count,
        }
        media_files[bundle_path] = local_path
    row["files"] = files
    return row, media_files


def _is_portable_reference_asset(asset: Any) -> bool:
    return (
        asset.type == "image"
        and asset.status == "succeeded"
        and str(asset.mime_type).startswith("image/")
    )


def _media_asset_metadata(metadata_json: str) -> dict[str, object]:
    try:
        loaded = json.loads(metadata_json)
    except ValueError:
        return {}
    return cast(dict[str, object], loaded) if isinstance(loaded, dict) else {}


def _preview_warnings(data: dict[str, object]) -> tuple[str, ...]:
    warnings: list[str] = []
    if _list_of_objects(data.get("media_assets"), "media_assets"):
        warnings.append("Reference images will be restored as character links.")
    return tuple(warnings)


def _unique_character_name(
    repositories: PersistenceRepositories,
    *,
    target_save_id: str | None,
    desired_name: str,
) -> str:
    base = desired_name.strip() or "Imported character"
    if target_save_id is None:
        return base
    existing_names = {
        character.name.casefold()
        for character in repositories.list_characters(target_save_id)
    }
    if base.casefold() not in existing_names:
        return base
    candidate = f"{base} (imported)"
    if candidate.casefold() not in existing_names:
        return candidate
    index = 2
    while True:
        candidate = f"{base} (imported {index})"
        if candidate.casefold() not in existing_names:
            return candidate
        index += 1


def _require_target_save(
    repositories: PersistenceRepositories,
    save_id: str,
) -> None:
    if repositories.get_save(save_id) is None:
        raise ValueError(f"Unknown target save id: {save_id}")


def _media_path(media_dir: Path, value: str) -> Path | None:
    path = Path(value)
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
        / "character-imports"
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
        / "character-imports"
        / _safe_path_segment(asset_id)
        / "thumbnails"
        / _safe_filename(original_path)
    )


def _safe_filename(value: str) -> str:
    name = Path(value).name.strip()
    return _safe_path_segment(name or "media.bin")


def _safe_path_segment(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-"
        for char in value.strip()
    ).strip(".")
    return cleaned or uuid4().hex


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _json_object_from_bytes(payload: bytes, name: str) -> dict[str, object]:
    loaded = json.loads(payload.decode("utf-8"))
    if not isinstance(loaded, dict):
        raise CharacterBundleError(f"{name} must contain a JSON object")
    return cast(dict[str, object], loaded)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CharacterBundleError(f"Expected object: {name}")
    return cast(dict[str, object], value)


def _list_of_objects(value: object, name: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise CharacterBundleError(f"Expected object list: {name}")
    return cast(list[dict[str, object]], value)


def _text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise CharacterBundleError(f"Expected text field: {key}")
    return value


def _character_history_payload(row: dict[str, object]) -> str:
    return (_optional_text(row, "history") or _text(row, "known_state")).strip()


def _optional_text(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CharacterBundleError(f"Expected optional text field: {key}")
    return value


def _int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CharacterBundleError(f"Expected integer field: {key}")
    return value


def _boolish(row: dict[str, object], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise CharacterBundleError(f"Expected boolean field: {key}")


def _json_object(row: dict[str, object], key: str) -> dict[str, object]:
    loaded = _json_value(_text(row, key), key)
    if not isinstance(loaded, dict):
        raise CharacterBundleError(f"Expected JSON object field: {key}")
    return cast(dict[str, object], loaded)


def _json_string_list(row: dict[str, object], key: str) -> list[str]:
    loaded = _json_value(_text(row, key), key)
    if not isinstance(loaded, list) or not all(
        isinstance(item, str) for item in loaded
    ):
        raise CharacterBundleError(f"Expected JSON string list field: {key}")
    return cast(list[str], loaded)


def _json_value(value: str, key: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise CharacterBundleError(f"Invalid JSON field: {key}") from exc


def _required_metadata_text(
    metadata: dict[str, object],
    key: str,
    bundle_name: str,
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise CharacterBundleError(f"Missing media {key} metadata for {bundle_name}")
    return value


def _required_metadata_int(
    metadata: dict[str, object],
    key: str,
    bundle_name: str,
) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CharacterBundleError(f"Missing media {key} metadata for {bundle_name}")
    return value


def _dump_json_pretty(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _require_row(row: Any | None, message: str) -> Any:
    if row is None:
        raise ValueError(message)
    return row
