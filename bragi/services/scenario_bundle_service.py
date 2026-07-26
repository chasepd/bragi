"""Portable reusable scenario import and export bundles."""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast
from uuid import uuid4

from bragi import __version__
from bragi.app_logging import log_event
from bragi.persistence.migrations import CURRENT_SCHEMA_VERSION
from bragi.persistence.repositories import PersistenceRepositories
from bragi.private_files import write_private_bytes
from bragi.redaction import redact_log_value, redact_text
from bragi.services.action_choice_flags import normalize_legacy_action_choice_scenario
from bragi.services.media_service import (
    _assert_scenario_starter_reference_path,
    _assert_uploaded_image_size,
    _assert_within_media_dir,
    _persist_thumbnail,
    _safe_path_segment,
    _uploaded_image_mime_type,
)
from bragi.services.scenario_content_rating import (
    metadata_with_scenario_content_ratings,
)
from bragi.services.scenario_service import (
    RETIRED_SCENARIO_REASON,
    ScenarioType,
    normalize_scenario_definition,
    scenario_record_is_retired,
)

SCENARIO_BUNDLE_FORMAT = "bragi-scenario-bundle"
SCENARIO_BUNDLE_VERSION = 2
SUPPORTED_SCENARIO_BUNDLE_VERSIONS = frozenset({1, SCENARIO_BUNDLE_VERSION})
MANIFEST_NAME = "manifest.json"
DATA_NAME = "data.json"
MEDIA_PREFIX = "media/"
_MAX_SCENARIO_BUNDLE_JSON_BYTES = 5 * 1024 * 1024
_MAX_SCENARIO_BUNDLE_MEDIA_BYTES = 25 * 1024 * 1024
_MAX_SCENARIO_BUNDLE_MEDIA_REFERENCES = 32
_MAX_SCENARIO_BUNDLE_TOTAL_MEDIA_BYTES = 100 * 1024 * 1024


class ScenarioBundleError(ValueError):
    """Raised when a scenario bundle is invalid or unsupported."""


@dataclass(frozen=True)
class ScenarioBundleManifest:
    bundle_format: str
    bundle_version: int
    scenario_id: str
    title: str
    scenario_type: str
    created_at: str | None
    updated_at: str | None
    exported_at: str


@dataclass(frozen=True)
class ScenarioBundlePreview:
    scenario_id: str
    title: str
    scenario_type: str
    bundle_version: int
    created_at: str | None = None
    updated_at: str | None = None
    exported_at: str | None = None


@dataclass(frozen=True)
class ImportedScenarioBundle:
    scenario_id: str
    title: str
    scenario_type: str


class ScenarioBundleService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        media_dir: Path | None = None,
    ) -> None:
        self.repositories = repositories
        self.media_dir = media_dir

    def export_scenario(
        self,
        scenario_id: str,
        bundle_path: Path,
    ) -> ScenarioBundleManifest:
        row = _require_row(
            self.repositories.connection.execute(
                """
                SELECT id, type, title, premise, player_role, content_json,
                       created_at, updated_at
                FROM scenarios
                WHERE id = ?
                """,
                (scenario_id,),
            ).fetchone(),
            f"Unknown scenario id: {scenario_id}",
        )
        content = _json_object_from_text(row["content_json"], "content_json")
        safe_title = _redacted_text(row["title"])
        safe_type = _text_value(row["type"])
        if scenario_record_is_retired(safe_type, content):
            normalized_premise = _text_value(row["premise"])
            normalized_content = content
        else:
            safe_type, content, _legacy_action_choices_enabled = (
                normalize_legacy_action_choice_scenario(
                    scenario_type=safe_type,
                    content=content,
                )
            )
            normalized_premise, normalized_content = normalize_scenario_definition(
                scenario_type=safe_type,
                premise=_text_value(row["premise"]),
                content=content,
            )
        safe_created_at = _optional_text_value(row["created_at"])
        safe_updated_at = _optional_text_value(row["updated_at"])
        scenario_payload = _redacted_mapping(
            {
                "id": _text_value(row["id"]),
                "type": safe_type,
                "title": safe_title,
                "premise": _redacted_text(normalized_premise),
                "player_role": _redacted_text(row["player_role"]),
                "content": normalized_content,
                "created_at": safe_created_at,
                "updated_at": safe_updated_at,
            }
        )
        exported_at = datetime.now(UTC).isoformat()
        manifest_payload = _redacted_mapping(
            {
                "format": SCENARIO_BUNDLE_FORMAT,
                "bundle_format": SCENARIO_BUNDLE_FORMAT,
                "bundle_version": SCENARIO_BUNDLE_VERSION,
                "title": safe_title,
                "scenario_title": safe_title,
                "scenario_type": safe_type,
                "created_by": {
                    "application": "Bragi",
                    "version": __version__,
                },
                "bragi_schema_version": CURRENT_SCHEMA_VERSION,
                "schema_version": CURRENT_SCHEMA_VERSION,
                "exported_at": exported_at,
                "scenario": {
                    "id": _text_value(row["id"]),
                    "title": safe_title,
                    "type": safe_type,
                    "created_at": safe_created_at,
                    "updated_at": safe_updated_at,
                },
            }
        )
        data: dict[str, object] = {"scenario": scenario_payload}
        media_members = _attach_bundle_media_members(
            data=data,
            media_dir=self.media_dir,
        )
        manifest = _manifest_from_payload(
            manifest_payload,
            allow_retired=scenario_record_is_retired(safe_type, content),
        )
        _write_bundle_atomically(
            bundle_path=bundle_path,
            manifest_payload=manifest_payload,
            data=data,
            media_members=media_members,
        )
        log_event(
            "scenario_bundle.exported",
            scenario_id=scenario_id,
            bundle_path=str(bundle_path),
            scenario_type=manifest.scenario_type,
        )
        return manifest

    def preview_import(self, bundle_path: Path) -> ScenarioBundlePreview:
        manifest_payload, data = self._read_bundle(bundle_path)
        manifest = _manifest_from_payload(manifest_payload)
        preview_type = _normalized_import_scenario_type(manifest.scenario_type, data)
        return ScenarioBundlePreview(
            scenario_id=manifest.scenario_id,
            title=manifest.title,
            scenario_type=preview_type,
            bundle_version=manifest.bundle_version,
            created_at=manifest.created_at,
            updated_at=manifest.updated_at,
            exported_at=manifest.exported_at,
        )

    def import_scenario(self, bundle_path: Path) -> ImportedScenarioBundle:
        manifest_payload, data = self._read_bundle(bundle_path)
        _manifest_from_payload(manifest_payload)
        scenario = _object(data.get("scenario"), "scenario")
        scenario_type = _validated_scenario_type(_text(scenario, "type"))
        normalized_type, scenario_content, _legacy_action_choices_enabled = (
            normalize_legacy_action_choice_scenario(
                scenario_type=scenario_type.value,
                content=_object(scenario.get("content"), "scenario content"),
            )
        )
        original_title = _text(scenario, "title").strip() or "Imported scenario"
        title = _unique_scenario_title(self.repositories, original_title)
        premise, content = normalize_scenario_definition(
            scenario_type=normalized_type,
            premise=_text(scenario, "premise"),
            content=scenario_content,
        )
        content = _quarantine_imported_scenario_content(content)
        materialized_paths: list[str] = []
        try:
            content = _materialize_bundle_media_members(
                bundle_path=bundle_path,
                content=content,
                media_dir=self.media_dir,
                materialized_paths=materialized_paths,
            )
            record = self.repositories.create_scenario(
                type=normalized_type,
                title=title,
                premise=premise,
                player_role=_text(scenario, "player_role"),
                content=content,
            )
        except Exception:
            if self.media_dir is not None:
                for relative_path in reversed(materialized_paths):
                    _unlink_media_file(self.media_dir, relative_path)
            raise
        log_event(
            "scenario_bundle.imported",
            scenario_id=record.id,
            source_scenario_id=_text(scenario, "id"),
            scenario_type=record.type,
            bundle_path=str(bundle_path),
        )
        return ImportedScenarioBundle(
            scenario_id=record.id,
            title=record.title,
            scenario_type=record.type,
        )

    def _read_bundle(
        self,
        bundle_path: Path,
    ) -> tuple[dict[str, object], dict[str, object]]:
        try:
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
                _validate_bundle_members(manifest, data, bundle)
                return manifest, data
        except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            raise ScenarioBundleError("Invalid Bragi scenario bundle") from exc


def _write_bundle_atomically(
    *,
    bundle_path: Path,
    manifest_payload: dict[str, object],
    data: dict[str, object],
    media_members: list[tuple[str, bytes]] | None = None,
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
            for member_name, member_bytes in media_members or []:
                bundle.writestr(member_name, member_bytes)
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


def _read_limited_member(bundle: zipfile.ZipFile, name: str) -> bytes:
    info = bundle.getinfo(name)
    if info.file_size > _MAX_SCENARIO_BUNDLE_JSON_BYTES:
        raise ScenarioBundleError(f"Scenario bundle {name} is too large")
    return bundle.read(info)


def _validate_bundle_members(
    manifest: dict[str, object],
    data: dict[str, object],
    bundle: zipfile.ZipFile,
) -> None:
    expected = {MANIFEST_NAME, DATA_NAME}
    seen: set[str] = set()
    media_members: set[str] = set()
    version = _bundle_version(manifest)
    for info in bundle.infolist():
        if info.filename in seen:
            raise ScenarioBundleError(
                f"Duplicate scenario bundle member: {info.filename}"
            )
        seen.add(info.filename)
        if info.filename in expected:
            continue
        if version < 2 or not info.filename.startswith(MEDIA_PREFIX):
            raise ScenarioBundleError(
                f"Unexpected scenario bundle member: {info.filename}"
            )
        _validate_bundle_member_path(info.filename)
        if info.file_size > _MAX_SCENARIO_BUNDLE_MEDIA_BYTES:
            raise ScenarioBundleError(
                f"Scenario bundle media member is too large: {info.filename}"
            )
        media_members.add(info.filename)
    referenced_media = _referenced_bundle_media_members(data)
    missing = referenced_media - media_members
    if missing:
        raise ScenarioBundleError(
            f"Scenario bundle media member is missing: {sorted(missing)[0]}"
        )
    extra = media_members - referenced_media
    if extra:
        raise ScenarioBundleError(
            f"Unexpected scenario bundle media member: {sorted(extra)[0]}"
        )


def _validate_bundle_data(
    manifest: dict[str, object],
    data: dict[str, object],
) -> None:
    scenario = _object(data.get("scenario"), "scenario")
    manifest_scenario = _object(manifest.get("scenario"), "manifest scenario")
    if _text(scenario, "id") != _text(manifest_scenario, "id"):
        raise ScenarioBundleError("Scenario bundle manifest does not match data")
    if _text(scenario, "title") != _text(manifest_scenario, "title"):
        raise ScenarioBundleError("Scenario bundle manifest does not match data")
    if _text(scenario, "type") != _text(manifest_scenario, "type"):
        raise ScenarioBundleError("Scenario bundle manifest does not match data")
    scenario_type = _text(scenario, "type")
    _validated_scenario_type(scenario_type)
    content = _object(scenario.get("content"), "scenario content")
    if scenario_record_is_retired(scenario_type, content):
        raise ScenarioBundleError(RETIRED_SCENARIO_REASON)


def _attach_bundle_media_members(
    *,
    data: dict[str, object],
    media_dir: Path | None,
) -> list[tuple[str, bytes]]:
    members: list[tuple[str, bytes]] = []
    member_names: set[str] = set()
    source_paths: set[Path] = set()
    total_size = 0
    for reference in _scenario_starter_reference_images(data):
        if len(members) >= _MAX_SCENARIO_BUNDLE_MEDIA_REFERENCES:
            raise ScenarioBundleError("Scenario bundle references too many media files")
        path = reference.get("path")
        if not isinstance(path, str) or not path:
            continue
        try:
            _assert_scenario_starter_reference_path(path)
        except ValueError as exc:
            raise ScenarioBundleError(
                "Scenario starter reference image path is invalid"
            ) from exc
        if media_dir is None:
            raise ScenarioBundleError(
                "Scenario starter reference images require media storage"
            )
        source_path = media_dir / path
        _assert_within_media_dir(media_dir=media_dir, output_path=source_path)
        if not source_path.is_file():
            raise ScenarioBundleError("Scenario starter reference image is missing")
        resolved_source_path = source_path.resolve()
        if resolved_source_path in source_paths:
            raise ScenarioBundleError(
                f"Duplicate scenario bundle media reference: {path}"
            )
        source_paths.add(resolved_source_path)
        image_id = str(reference.get("id") or uuid4().hex)
        suffix = "".join(Path(path).suffixes) or ".bin"
        member_name = (
            f"{MEDIA_PREFIX}scenario-starter-reference-images/"
            f"{_safe_bundle_segment(image_id)}{suffix}"
        )
        _validate_bundle_member_path(member_name)
        if member_name in member_names:
            raise ScenarioBundleError(
                f"Duplicate scenario bundle media reference: {member_name}"
            )
        member_names.add(member_name)
        reference["bundle_path"] = member_name
        media_bytes = _read_limited_media_file(source_path)
        total_size += len(media_bytes)
        if total_size > _MAX_SCENARIO_BUNDLE_TOTAL_MEDIA_BYTES:
            raise ScenarioBundleError("Scenario bundle media is too large")
        members.append((member_name, media_bytes))
    return members


def _materialize_bundle_media_members(
    *,
    bundle_path: Path,
    content: dict[str, object],
    media_dir: Path | None,
    materialized_paths: list[str],
) -> dict[str, object]:
    media_references = _bundle_media_references_for_import(content)
    if not media_references:
        return content
    if media_dir is None:
        raise ScenarioBundleError(
            "Scenario starter reference images require media storage"
        )
    with zipfile.ZipFile(bundle_path) as bundle:
        _validate_materialized_media_budget(
            bundle=bundle,
            references=media_references,
        )
        for reference in media_references:
            member_name = cast(str, reference["bundle_path"])
            _validate_bundle_member_path(member_name)
            image_bytes = _read_limited_media_member(bundle, member_name)
            _assert_uploaded_image_size(len(image_bytes))
            mime_type, extension = _uploaded_image_mime_type(image_bytes)
            image_id = uuid4().hex
            relative_path = (
                Path("scenario-starters")
                / "imports"
                / f"{_safe_path_segment(image_id)}{extension}"
            )
            output_path = media_dir / relative_path
            _assert_within_media_dir(media_dir=media_dir, output_path=output_path)
            thumbnail_path: str | None = None
            try:
                write_private_bytes(output_path, image_bytes)
                materialized_paths.append(relative_path.as_posix())
                thumbnail_path = _persist_thumbnail(
                    media_dir=media_dir,
                    image_relative_path=relative_path,
                    image_path=output_path,
                )
                if thumbnail_path is not None:
                    materialized_paths.append(thumbnail_path)
            except Exception:
                _unlink_media_file(media_dir, relative_path.as_posix())
                _unlink_media_file(media_dir, thumbnail_path)
                raise
            reference["id"] = image_id
            reference["path"] = relative_path.as_posix()
            reference["thumbnail_path"] = thumbnail_path
            reference["mime_type"] = mime_type
            reference["prompt_preview"] = str(
                reference.get("prompt_preview")
                or "Uploaded character reference image"
            )
            reference["source"] = "uploaded"
            reference["content_rating"] = "unclassified"
            reference["created_at"] = datetime.now(UTC).isoformat()
            reference["bundle_path"] = None
    return content


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


def _validate_materialized_media_budget(
    *,
    bundle: zipfile.ZipFile,
    references: list[dict[str, object]],
) -> None:
    if len(references) > _MAX_SCENARIO_BUNDLE_MEDIA_REFERENCES:
        raise ScenarioBundleError("Scenario bundle references too many media files")
    seen_members: set[str] = set()
    total_size = 0
    for reference in references:
        member_name = cast(str, reference["bundle_path"])
        if member_name in seen_members:
            raise ScenarioBundleError(
                f"Duplicate scenario bundle media reference: {member_name}"
            )
        seen_members.add(member_name)
        info = bundle.getinfo(member_name)
        total_size += info.file_size
        if total_size > _MAX_SCENARIO_BUNDLE_TOTAL_MEDIA_BYTES:
            raise ScenarioBundleError("Scenario bundle media is too large")


def _bundle_media_references_for_import(
    content: dict[str, object],
) -> list[dict[str, object]]:
    starters = content.get("character_starters")
    if not isinstance(starters, list):
        return []
    references: list[dict[str, object]] = []
    for starter in starters:
        if not isinstance(starter, dict):
            continue
        reference = starter.get("reference_image")
        if not isinstance(reference, dict):
            continue
        bundle_path = reference.get("bundle_path")
        if isinstance(bundle_path, str) and bundle_path:
            references.append(cast(dict[str, object], reference))
            continue
        starter["reference_image"] = None
    return references


def _scenario_starter_reference_images(
    data: dict[str, object],
) -> list[dict[str, object]]:
    try:
        scenario = _object(data.get("scenario"), "scenario")
        content = _object(scenario.get("content"), "scenario content")
    except ScenarioBundleError:
        return []
    starters = content.get("character_starters")
    if not isinstance(starters, list):
        return []
    references: list[dict[str, object]] = []
    for starter in starters:
        if not isinstance(starter, dict):
            continue
        reference = starter.get("reference_image")
        if isinstance(reference, dict):
            references.append(cast(dict[str, object], reference))
    return references


def _referenced_bundle_media_members(data: dict[str, object]) -> set[str]:
    members: set[str] = set()
    for reference in _scenario_starter_reference_images(data):
        bundle_path = reference.get("bundle_path")
        if isinstance(bundle_path, str) and bundle_path:
            _validate_bundle_member_path(bundle_path)
            members.add(bundle_path)
    return members


def _validate_bundle_member_path(member_name: str) -> None:
    path = Path(member_name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not member_name.startswith(MEDIA_PREFIX)
    ):
        raise ScenarioBundleError(f"Invalid scenario bundle member: {member_name}")


def _safe_bundle_segment(value: str) -> str:
    return _safe_path_segment(value).replace("%", "_")


def _read_limited_media_member(bundle: zipfile.ZipFile, name: str) -> bytes:
    info = bundle.getinfo(name)
    if info.file_size > _MAX_SCENARIO_BUNDLE_MEDIA_BYTES:
        raise ScenarioBundleError(f"Scenario bundle media member is too large: {name}")
    return bundle.read(info)


def _read_limited_media_file(path: Path) -> bytes:
    if path.stat().st_size > _MAX_SCENARIO_BUNDLE_MEDIA_BYTES:
        raise ScenarioBundleError("Scenario starter reference image is too large")
    with path.open("rb") as file:
        data = file.read(_MAX_SCENARIO_BUNDLE_MEDIA_BYTES + 1)
    if len(data) > _MAX_SCENARIO_BUNDLE_MEDIA_BYTES:
        raise ScenarioBundleError("Scenario starter reference image is too large")
    return data


def _unlink_media_file(media_dir: Path, relative_path: str | None) -> None:
    if relative_path is None:
        return
    path = media_dir / relative_path
    try:
        _assert_within_media_dir(media_dir=media_dir, output_path=path)
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _manifest_from_payload(
    payload: dict[str, object],
    *,
    allow_retired: bool = False,
) -> ScenarioBundleManifest:
    _validate_manifest_payload(payload)
    scenario = _object(payload.get("scenario"), "manifest scenario")
    scenario_type = _text(scenario, "type")
    if not (allow_retired and scenario_record_is_retired(scenario_type)):
        _validated_scenario_type(scenario_type)
    return ScenarioBundleManifest(
        bundle_format=SCENARIO_BUNDLE_FORMAT,
        bundle_version=_bundle_version(payload),
        scenario_id=_text(scenario, "id"),
        title=_text(scenario, "title"),
        scenario_type=scenario_type,
        created_at=_optional_text(scenario, "created_at"),
        updated_at=_optional_text(scenario, "updated_at"),
        exported_at=_text(payload, "exported_at"),
    )


def _normalized_import_scenario_type(
    manifest_type: str,
    data: dict[str, object],
) -> str:
    try:
        scenario = _object(data.get("scenario"), "scenario")
        scenario_type = _validated_scenario_type(_text(scenario, "type")).value
        normalized_type, _content, _legacy_action_choices_enabled = (
            normalize_legacy_action_choice_scenario(
                scenario_type=scenario_type,
                content=_object(scenario.get("content"), "scenario content"),
            )
        )
        return normalized_type
    except ScenarioBundleError:
        return manifest_type


def _validate_manifest_payload(payload: dict[str, object]) -> None:
    bundle_format = payload.get("format")
    if bundle_format != SCENARIO_BUNDLE_FORMAT:
        raise ScenarioBundleError("Not a Bragi scenario bundle")
    version = _bundle_version(payload)
    if version not in SUPPORTED_SCENARIO_BUNDLE_VERSIONS:
        raise ScenarioBundleError(
            f"Unsupported Bragi scenario bundle version: {version}"
        )
    schema_version = payload.get("bragi_schema_version", payload.get("schema_version"))
    if (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version > CURRENT_SCHEMA_VERSION
    ):
        raise ScenarioBundleError(
            "Bragi scenario bundle requires a newer database schema: "
            f"{schema_version}"
        )


def _bundle_version(payload: dict[str, object]) -> int:
    version = payload.get("bundle_version")
    if isinstance(version, int) and not isinstance(version, bool):
        return version
    raise ScenarioBundleError(f"Unsupported Bragi scenario bundle version: {version}")


def _validated_scenario_type(value: str) -> ScenarioType:
    if scenario_record_is_retired(value):
        raise ScenarioBundleError(RETIRED_SCENARIO_REASON)
    try:
        return ScenarioType(value)
    except ValueError as exc:
        raise ScenarioBundleError(f"Unsupported scenario type: {value}") from exc


def _unique_scenario_title(
    repositories: PersistenceRepositories,
    desired_title: str,
) -> str:
    base = desired_title.strip() or "Imported scenario"
    existing_titles = {
        scenario.title.casefold() for scenario in repositories.list_scenarios()
    }
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


def _json_object_from_bytes(payload: bytes, name: str) -> dict[str, object]:
    loaded = json.loads(payload.decode("utf-8"))
    if not isinstance(loaded, dict):
        raise ScenarioBundleError(f"{name} must contain a JSON object")
    return cast(dict[str, object], loaded)


def _json_object_from_text(value: str, name: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ScenarioBundleError(f"Invalid scenario {name}") from exc
    if not isinstance(loaded, dict):
        raise ScenarioBundleError(f"Scenario {name} must contain a JSON object")
    return cast(dict[str, object], loaded)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScenarioBundleError(f"Expected object: {name}")
    return cast(dict[str, object], value)


def _text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ScenarioBundleError(f"Expected text field: {key}")
    return value


def _optional_text(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ScenarioBundleError(f"Expected optional text field: {key}")
    return value


def _text_value(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _optional_text_value(value: object) -> str | None:
    if value is None:
        return None
    return _text_value(value)


def _redacted_text(value: object) -> str:
    return redact_text(_text_value(value)) or ""


def _redacted_mapping(value: dict[str, object]) -> dict[str, object]:
    redacted = redact_log_value(value)
    if not isinstance(redacted, dict):
        raise ScenarioBundleError("Unable to redact scenario bundle payload")
    return cast(dict[str, object], redacted)


def _dump_json_pretty(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _require_row(row: Any | None, message: str) -> Any:
    if row is None:
        raise ValueError(message)
    return row
