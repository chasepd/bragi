"""Portable persistent-world import and export bundles."""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from bragi import __version__
from bragi.app_logging import log_event
from bragi.persistence.migrations import CURRENT_SCHEMA_VERSION
from bragi.persistence.repositories import PersistenceRepositories
from bragi.redaction import redact_log_value
from bragi.services.content_rating import sanitize_content_rating
from bragi.zip_safety import ZipSafetyError, validate_zip_directory

PERSISTENT_WORLD_BUNDLE_FORMAT = "bragi-persistent-world-bundle"
PERSISTENT_WORLD_BUNDLE_VERSION = 1
SUPPORTED_PERSISTENT_WORLD_BUNDLE_VERSIONS = frozenset(
    {PERSISTENT_WORLD_BUNDLE_VERSION}
)
MANIFEST_NAME = "manifest.json"
DATA_NAME = "data.json"
_MAX_PERSISTENT_WORLD_BUNDLE_JSON_BYTES = 5 * 1024 * 1024


class PersistentWorldBundleError(ValueError):
    """Raised when a persistent-world bundle is invalid or unsupported."""


@dataclass(frozen=True)
class PersistentWorldBundleManifest:
    bundle_format: str
    bundle_version: int
    world_id: str
    title: str
    created_at: str | None
    updated_at: str | None
    exported_at: str


@dataclass(frozen=True)
class PersistentWorldBundlePreview:
    world_id: str
    title: str
    bundle_version: int
    description: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    exported_at: str | None = None


@dataclass(frozen=True)
class ImportedPersistentWorldBundle:
    world_id: str
    title: str


class PersistentWorldBundleService:
    def __init__(self, *, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories

    def export_world(
        self,
        world_id: str,
        bundle_path: Path,
    ) -> PersistentWorldBundleManifest:
        world = self.repositories.get_persistent_world(world_id)
        if world is None:
            raise ValueError(f"Unknown persistent world id: {world_id}")
        sections = _json_object_from_text(world.content_json, "world content")
        source_metadata = _json_object_from_text(
            world.source_metadata_json,
            "world source metadata",
        )
        exported_at = datetime.now(UTC).isoformat()
        title = _text_value(world.title)
        world_payload = _redacted_mapping(
            {
                "id": _text_value(world.id),
                "title": title,
                "description": _text_value(world.description),
                "sections": sections,
                "source_metadata": source_metadata,
                "content_rating": _text_value(world.content_rating),
                "created_at": world.created_at,
                "updated_at": world.updated_at,
            }
        )
        manifest_payload = _redacted_mapping(
            {
                "format": PERSISTENT_WORLD_BUNDLE_FORMAT,
                "bundle_format": PERSISTENT_WORLD_BUNDLE_FORMAT,
                "bundle_version": PERSISTENT_WORLD_BUNDLE_VERSION,
                "title": title,
                "created_by": {"application": "Bragi", "version": __version__},
                "bragi_schema_version": CURRENT_SCHEMA_VERSION,
                "schema_version": CURRENT_SCHEMA_VERSION,
                "exported_at": exported_at,
                "world": {
                    "id": _text_value(world.id),
                    "title": title,
                    "created_at": world.created_at,
                    "updated_at": world.updated_at,
                },
            }
        )
        data: dict[str, object] = {"world": world_payload}
        manifest = _manifest_from_payload(manifest_payload)
        _write_bundle_atomically(
            bundle_path=bundle_path,
            manifest_payload=manifest_payload,
            data=data,
        )
        log_event(
            "persistent_world_bundle.exported",
            world_id=world_id,
            bundle_path=str(bundle_path),
        )
        return manifest

    def preview_import(
        self,
        bundle_path: Path,
    ) -> PersistentWorldBundlePreview:
        manifest_payload, data = self._read_bundle(bundle_path)
        manifest = _manifest_from_payload(manifest_payload)
        world = _object(data.get("world"), "world")
        return PersistentWorldBundlePreview(
            world_id=manifest.world_id,
            title=manifest.title,
            bundle_version=manifest.bundle_version,
            description=_optional_text(world, "description") or "",
            created_at=manifest.created_at,
            updated_at=manifest.updated_at,
            exported_at=manifest.exported_at,
        )

    def import_world(self, bundle_path: Path) -> ImportedPersistentWorldBundle:
        manifest_payload, data = self._read_bundle(bundle_path)
        _manifest_from_payload(manifest_payload)
        world = _object(data.get("world"), "world")
        title = _text(world, "title").strip() or "Imported persistent world"
        existing_titles = {
            candidate.title.casefold()
            for candidate in self.repositories.list_persistent_worlds()
        }
        title = _unique_title(title, existing_titles)
        source_metadata = _object_or_empty(world.get("source_metadata"))
        source_metadata["origin"] = "bundle_import"
        record = self.repositories.create_persistent_world(
            title=title,
            description=_optional_text(world, "description") or "",
            sections=_string_mapping(world.get("sections"), "sections"),
            source_metadata=source_metadata,
            content_rating=sanitize_content_rating(
                world.get("content_rating"),
                default="unclassified",
            ),
        )
        log_event(
            "persistent_world_bundle.imported",
            world_id=record.id,
            source_world_id=_text(world, "id"),
            bundle_path=str(bundle_path),
        )
        return ImportedPersistentWorldBundle(world_id=record.id, title=record.title)

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
                names = [info.filename for info in bundle.infolist()]
                if set(names) != {MANIFEST_NAME, DATA_NAME} or len(names) != 2:
                    raise PersistentWorldBundleError(
                        "Unexpected persistent world bundle member"
                    )
                return manifest, data
        except (
            OSError,
            KeyError,
            ZipSafetyError,
            zipfile.BadZipFile,
            json.JSONDecodeError,
        ) as exc:
            raise PersistentWorldBundleError(
                "Invalid Bragi persistent world bundle"
            ) from exc


def _write_bundle_atomically(
    *,
    bundle_path: Path,
    manifest_payload: dict[str, object],
    data: dict[str, object],
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
        temporary_path.chmod(0o600)
        os.replace(temporary_path, bundle_path)
        bundle_path.chmod(0o600)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _manifest_from_payload(
    payload: dict[str, object],
) -> PersistentWorldBundleManifest:
    _validate_manifest_payload(payload)
    world = _object(payload.get("world"), "manifest world")
    return PersistentWorldBundleManifest(
        bundle_format=PERSISTENT_WORLD_BUNDLE_FORMAT,
        bundle_version=_bundle_version(payload),
        world_id=_text(world, "id"),
        title=_text(world, "title"),
        created_at=_optional_text(world, "created_at"),
        updated_at=_optional_text(world, "updated_at"),
        exported_at=_text(payload, "exported_at"),
    )


def _validate_manifest_payload(payload: dict[str, object]) -> None:
    if payload.get("format") != PERSISTENT_WORLD_BUNDLE_FORMAT:
        raise PersistentWorldBundleError(
            "Not a Bragi persistent world bundle"
        )
    version = _bundle_version(payload)
    if version not in SUPPORTED_PERSISTENT_WORLD_BUNDLE_VERSIONS:
        raise PersistentWorldBundleError(
            f"Unsupported persistent world bundle version: {version}"
        )
    schema_version = payload.get("bragi_schema_version", payload.get("schema_version"))
    if (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version > CURRENT_SCHEMA_VERSION
    ):
        raise PersistentWorldBundleError(
            "Persistent world bundle requires a newer database schema"
        )


def _validate_bundle_data(
    manifest: dict[str, object],
    data: dict[str, object],
) -> None:
    world = _object(data.get("world"), "world")
    manifest_world = _object(manifest.get("world"), "manifest world")
    if _text(world, "id") != _text(manifest_world, "id"):
        raise PersistentWorldBundleError(
            "Persistent world bundle manifest does not match data"
        )
    if _text(world, "title") != _text(manifest_world, "title"):
        raise PersistentWorldBundleError(
            "Persistent world bundle manifest does not match data"
        )
    _optional_text(world, "description")
    _string_mapping(world.get("sections"), "sections")
    source_metadata = world.get("source_metadata")
    if source_metadata is not None:
        _object(source_metadata, "source_metadata")
    _optional_text(world, "content_rating")


def _bundle_version(payload: dict[str, object]) -> int:
    value = payload.get("bundle_version")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise PersistentWorldBundleError(
        f"Unsupported persistent world bundle version: {value}"
    )


def _read_limited_member(bundle: zipfile.ZipFile, name: str) -> bytes:
    info = bundle.getinfo(name)
    if info.file_size > _MAX_PERSISTENT_WORLD_BUNDLE_JSON_BYTES:
        raise PersistentWorldBundleError(f"Persistent world bundle {name} is too large")
    return bundle.read(info)


def _json_object_from_bytes(payload: bytes, name: str) -> dict[str, object]:
    loaded = json.loads(payload.decode("utf-8"))
    if not isinstance(loaded, dict):
        raise PersistentWorldBundleError(f"{name} must contain a JSON object")
    return cast(dict[str, object], loaded)


def _json_object_from_text(value: str, name: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PersistentWorldBundleError(f"Invalid persistent world {name}") from exc
    if not isinstance(loaded, dict):
        raise PersistentWorldBundleError(f"Persistent world {name} must be an object")
    return cast(dict[str, object], loaded)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PersistentWorldBundleError(f"Expected object: {name}")
    return cast(dict[str, object], value)


def _object_or_empty(value: object) -> dict[str, object]:
    if value is None:
        return {}
    return _object(value, "source_metadata")


def _string_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise PersistentWorldBundleError(f"Expected object: {name}")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise PersistentWorldBundleError(f"Expected text mapping: {name}")
        if key.strip() and item.strip():
            result[key.strip()] = item
    return result


def _text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise PersistentWorldBundleError(f"Expected text field: {key}")
    return value


def _optional_text(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise PersistentWorldBundleError(f"Expected optional text field: {key}")
    return value


def _text_value(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _redacted_mapping(value: dict[str, object]) -> dict[str, object]:
    redacted = redact_log_value(value)
    if not isinstance(redacted, dict):
        raise PersistentWorldBundleError(
            "Unable to redact persistent world bundle payload"
        )
    return cast(dict[str, object], redacted)


def _dump_json_pretty(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _unique_title(desired_title: str, existing_titles: set[str]) -> str:
    base = desired_title.strip() or "Imported persistent world"
    if base.casefold() not in existing_titles:
        return base
    candidate = f"{base} (imported)"
    if candidate.casefold() not in existing_titles:
        return candidate
    index = 2
    while True:
        candidate = f"{base} (imported {index})"
        if candidate.casefold() not in existing_titles:
            return candidate
        index += 1
