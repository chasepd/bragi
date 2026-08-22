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


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_persistent_world_bundle_round_trips_world_definition(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    world = repositories.create_persistent_world(
        title="The Salt Marches",
        description="A setting of rivers, salt, and old oaths.",
        sections={"overview": "The river clans guard the wells."},
        source_metadata={"origin": "manual"},
        content_rating="pg-13",
    )
    service = _bundle_service(repositories)
    bundle_path = tmp_path / "salt-marches.bragi-world"

    manifest = service.export_world(world.id, bundle_path)

    assert manifest.bundle_version == 1
    preview = service.preview_import(bundle_path)
    assert preview.world_id == world.id
    imported = service.import_world(bundle_path)
    assert imported.title == "The Salt Marches (imported)"
    imported_world = repositories.get_persistent_world(imported.world_id)
    assert imported_world is not None
    assert imported_world.content_rating == "pg-13"
    assert json.loads(imported_world.content_json) == {
        "overview": "The river clans guard the wells."
    }
    assert json.loads(imported_world.source_metadata_json)["origin"] == (
        "bundle_import"
    )


def test_persistent_world_bundle_rejects_unexpected_members(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    world = repositories.create_persistent_world(
        title="The Salt Marches",
        sections={"overview": "The river clans guard the wells."},
    )
    service = _bundle_service(repositories)
    bundle_path = tmp_path / "salt-marches.bragi-world"
    service.export_world(world.id, bundle_path)
    with zipfile.ZipFile(bundle_path, mode="a") as bundle:
        bundle.writestr("unexpected.txt", "not allowed")

    module = _bundle_module()
    with pytest.raises(module.PersistentWorldBundleError):
        service.preview_import(bundle_path)


def _bundle_service(repositories: PersistenceRepositories) -> Any:
    return _bundle_module().PersistentWorldBundleService(
        repositories=repositories,
    )


def _bundle_module() -> Any:
    return importlib.import_module("bragi.services.persistent_world_bundle_service")
