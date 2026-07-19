from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.save_service import SaveService
from bragi.services.turn_snapshot_service import TurnSnapshotService


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_create_list_and_load_save_with_scenario_and_messages(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "opening_message": "The tower bell cracks once.",
        },
    )
    service = SaveService(repositories=repositories)

    save = service.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Ash scratches the glass as the stair shakes.",
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        token_estimate=42,
    )

    listed_saves = service.list_saves()
    loaded = service.load_save(save.id)

    assert [item.id for item in listed_saves] == [save.id]
    assert listed_saves[0].title == save.title
    assert loaded.save.id == save.id
    assert loaded.save.title == save.title
    assert loaded.scenario.id == scenario.id
    assert loaded.scenario.title == "Ashfall Keep"
    assert json.loads(loaded.scenario.content_json) == {
        "starting_scene": "The beacon gutters in the tower.",
        "opening_message": "The tower bell cracks once.",
    }
    assert loaded.messages == [player_message, narrator_message]


def test_rename_save_updates_trimmed_title(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    service = SaveService(repositories=repositories)
    save = service.create_save(scenario_id=scenario.id, title="Night Watch")

    updated = service.rename_save(save.id, "  Dawn Watch  ")

    assert updated.title == "Dawn Watch"
    fetched = repositories.get_save(save.id)
    assert fetched is not None
    assert fetched.title == "Dawn Watch"


def test_rename_save_rejects_blank_and_unknown_saves(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    service = SaveService(repositories=repositories)
    save = service.create_save(scenario_id=scenario.id, title="Night Watch")

    with pytest.raises(ValueError, match="Save title is required"):
        service.rename_save(save.id, "   ")

    with pytest.raises(ValueError, match="Unknown save id"):
        service.rename_save("missing-save", "Dawn Watch")


def test_load_save_marks_save_as_recently_opened_without_reordering_list(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    service = SaveService(repositories=repositories)
    old_save = service.create_save(scenario_id=scenario.id, title="Old Watch")
    recent_save = service.create_save(scenario_id=scenario.id, title="Recent Watch")
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, last_opened_at = ? WHERE id = ?",
        ("2026-05-01 00:00:00", "2026-05-01 00:00:00", old_save.id),
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, last_opened_at = ? WHERE id = ?",
        ("2026-05-02 00:00:00", "2026-05-02 00:00:00", recent_save.id),
    )
    repositories.commit()

    service.load_save(old_save.id)

    listed_saves = service.list_saves()
    touched_last_opened_at = repositories.connection.execute(
        "SELECT last_opened_at FROM saves WHERE id = ?",
        (old_save.id,),
    ).fetchone()[0]

    assert [save.id for save in listed_saves] == [recent_save.id, old_save.id]
    assert "T" not in touched_last_opened_at
    assert not touched_last_opened_at.endswith("Z")
    assert " " in touched_last_opened_at
    assert "." in touched_last_opened_at


def test_list_saves_orders_mixed_updated_timestamps_chronologically(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    service = SaveService(repositories=repositories)
    latest_sqlite = service.create_save(
        scenario_id=scenario.id,
        title="Latest SQLite",
    )
    legacy_fractional_later = service.create_save(
        scenario_id=scenario.id,
        title="Legacy Fractional Later",
    )
    sqlite_fractional_earlier = service.create_save(
        scenario_id=scenario.id,
        title="SQLite Fractional Earlier",
    )
    older_sqlite = service.create_save(
        scenario_id=scenario.id,
        title="Older SQLite",
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, created_at = ? WHERE id = ?",
        ("2026-05-02 00:00:01.000", "2026-05-02 00:00:00", latest_sqlite.id),
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, created_at = ? WHERE id = ?",
        (
            "2026-05-02T00:00:00.900Z",
            "2026-05-02 00:00:00",
            legacy_fractional_later.id,
        ),
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, created_at = ? WHERE id = ?",
        (
            "2026-05-02 00:00:00.100",
            "2026-05-02 00:00:00",
            sqlite_fractional_earlier.id,
        ),
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ?, created_at = ? WHERE id = ?",
        ("2026-05-01 23:59:59.999", "2026-05-02 00:00:00", older_sqlite.id),
    )
    repositories.commit()

    listed_saves = service.list_saves()

    assert [save.id for save in listed_saves] == [
        latest_sqlite.id,
        legacy_fractional_later.id,
        sqlite_fractional_earlier.id,
        older_sqlite.id,
    ]


def test_delete_save_removes_media_records_and_files(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    service = SaveService(repositories=repositories)
    save = service.create_save(scenario_id=scenario.id, title="Night Watch")
    media_dir = tmp_path / "media"
    media_dir.joinpath("generated").mkdir(parents=True)
    image_path = media_dir / "generated" / "beacon.png"
    thumbnail_path = media_dir / "generated" / "beacon.thumb.png"
    image_path.write_bytes(b"image")
    thumbnail_path.write_bytes(b"thumb")
    repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="generated/beacon.png",
        thumbnail_path="generated/beacon.thumb.png",
        prompt="cracked beacon",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )

    assert service.delete_save(save.id, media_dir=media_dir) is True

    assert repositories.get_save(save.id) is None
    assert repositories.list_media_assets(save.id) == []
    assert not image_path.exists()
    assert not thumbnail_path.exists()


def test_delete_save_removes_archived_media_files(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    service = SaveService(repositories=repositories)
    save = service.create_save(scenario_id=scenario.id, title="Night Watch")
    media_dir = tmp_path / "media"
    media_dir.joinpath("generated").mkdir(parents=True)
    image_path = media_dir / "generated" / "archived-beacon.png"
    thumbnail_path = media_dir / "generated" / "archived-beacon.thumb.png"
    image_path.write_bytes(b"image")
    thumbnail_path.write_bytes(b"thumb")
    archived_asset = repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="generated/archived-beacon.png",
        thumbnail_path="generated/archived-beacon.thumb.png",
        prompt="archived cracked beacon",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    repositories.archive_media_asset(
        save_id=save.id,
        media_asset_id=archived_asset.id,
    )

    assert repositories.list_media_assets(save.id) == []
    assert service.delete_save(save.id, media_dir=media_dir) is True

    assert repositories.get_save(save.id) is None
    assert not image_path.exists()
    assert not thumbnail_path.exists()


def test_delete_save_removes_snapshot_only_media_files_and_snapshot_objects(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    service = SaveService(repositories=repositories)
    save = service.create_save(scenario_id=scenario.id, title="Night Watch")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon throws a red warning over the ash road.",
    )
    media_dir = tmp_path / "media"
    media_dir.joinpath("generated").mkdir(parents=True)
    image_path = media_dir / "generated" / "snapshot-only.png"
    thumbnail_path = media_dir / "generated" / "snapshot-only.thumb.png"
    image_path.write_bytes(b"snapshot image")
    thumbnail_path.write_bytes(b"snapshot thumbnail")
    media = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path="generated/snapshot-only.png",
        thumbnail_path="generated/snapshot-only.thumb.png",
        prompt="red warning",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    TurnSnapshotService(repositories).capture_message_snapshot(
        save_id=save.id,
        message_id=message.id,
    )
    repositories.connection.execute(
        "DELETE FROM media_assets WHERE id = ?",
        (media.id,),
    )
    repositories.commit()

    assert repositories.list_all_media_assets(save.id) == []
    assert _snapshot_object_count(repositories) > 0
    assert service.delete_save(save.id, media_dir=media_dir) is True

    assert not image_path.exists()
    assert not thumbnail_path.exists()
    assert _snapshot_object_count(repositories) == 0


def test_delete_save_keeps_snapshot_objects_referenced_by_remaining_snapshots(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    service = SaveService(repositories=repositories)
    first_save = service.create_save(scenario_id=scenario.id, title="Night Watch")
    message = repositories.append_message(
        save_id=first_save.id,
        role="narrator",
        body="The lens flares red.",
    )
    snapshot_service = TurnSnapshotService(repositories)
    shared_snapshot = snapshot_service.capture_message_snapshot(
        save_id=first_save.id,
        message_id=message.id,
    )
    first_save_roots = _snapshot_root_hashes(repositories, first_save.id)
    first_save_only_roots = first_save_roots - {shared_snapshot.root_manifest_hash}
    assert first_save_only_roots
    second_save = service.create_save(scenario_id=scenario.id, title="Dawn Watch")
    repositories.connection.execute(
        """
        INSERT INTO save_turn_snapshots(
            id, save_id, message_id, parent_snapshot_id, root_manifest_hash,
            context_revision, reason
        )
        VALUES (?, ?, NULL, NULL, ?, 0, ?)
        """,
        (
            "shared-root-snapshot",
            second_save.id,
            shared_snapshot.root_manifest_hash,
            "shared-test-reference",
        ),
    )
    repositories.commit()

    assert service.delete_save(first_save.id, media_dir=tmp_path / "media") is True

    assert _snapshot_object_exists(repositories, shared_snapshot.root_manifest_hash)
    assert not any(
        _snapshot_object_exists(repositories, root_hash)
        for root_hash in first_save_only_roots
    )


def test_delete_save_keeps_media_files_when_database_delete_fails(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    service = SaveService(repositories=repositories)
    save = service.create_save(scenario_id=scenario.id, title="Night Watch")
    media_dir = tmp_path / "media"
    media_dir.joinpath("generated").mkdir(parents=True)
    image_path = media_dir / "generated" / "beacon.png"
    thumbnail_path = media_dir / "generated" / "beacon.thumb.png"
    image_path.write_bytes(b"image")
    thumbnail_path.write_bytes(b"thumb")
    repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="generated/beacon.png",
        thumbnail_path="generated/beacon.thumb.png",
        prompt="cracked beacon",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )

    def fail_database_delete(save_id: str) -> bool:
        assert save_id == save.id
        raise sqlite3.IntegrityError("simulated delete failure")

    monkeypatch.setattr(repositories, "delete_save", fail_database_delete)

    with pytest.raises(sqlite3.IntegrityError, match="simulated delete failure"):
        service.delete_save(save.id, media_dir=media_dir)

    assert repositories.get_save(save.id) == save
    assert [asset.save_id for asset in repositories.list_all_media_assets(save.id)] == [
        save.id
    ]
    assert image_path.exists()
    assert thumbnail_path.exists()


def test_delete_save_succeeds_when_post_commit_media_file_delete_fails(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    service = SaveService(repositories=repositories)
    save = service.create_save(scenario_id=scenario.id, title="Night Watch")
    media_dir = tmp_path / "media"
    media_dir.joinpath("generated").mkdir(parents=True)
    image_path = media_dir / "generated" / "beacon.png"
    thumbnail_path = media_dir / "generated" / "beacon.thumb.png"
    image_path.write_bytes(b"image")
    thumbnail_path.write_bytes(b"thumb")
    repositories.create_media_asset(
        save_id=save.id,
        type="image",
        path="generated/beacon.png",
        thumbnail_path="generated/beacon.thumb.png",
        prompt="cracked beacon",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    real_unlink = Path.unlink
    failed_once = False

    def fail_thumbnail_once(self: Path, missing_ok: bool = False) -> None:
        nonlocal failed_once
        if self == thumbnail_path and not failed_once:
            failed_once = True
            raise OSError("simulated unlink failure")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_thumbnail_once)

    assert service.delete_save(save.id, media_dir=media_dir) is True

    assert repositories.get_save(save.id) is None
    assert repositories.list_all_media_assets(save.id) == []
    assert not image_path.exists()
    assert thumbnail_path.exists()


def test_delete_save_rejects_unknown_save(
    repositories: PersistenceRepositories,
) -> None:
    service = SaveService(repositories=repositories)

    with pytest.raises(ValueError, match="Unknown save id: missing-save"):
        service.delete_save("missing-save")


def _snapshot_object_count(repositories: PersistenceRepositories) -> int:
    row = repositories.connection.execute(
        "SELECT COUNT(*) AS count FROM save_snapshot_objects"
    ).fetchone()
    return int(row["count"])


def _snapshot_root_hashes(
    repositories: PersistenceRepositories,
    save_id: str,
) -> set[str]:
    rows = repositories.connection.execute(
        """
        SELECT root_manifest_hash
        FROM save_turn_snapshots
        WHERE save_id = ?
        """,
        (save_id,),
    ).fetchall()
    return {str(row["root_manifest_hash"]) for row in rows}


def _snapshot_object_exists(
    repositories: PersistenceRepositories,
    object_hash: str,
) -> bool:
    row = repositories.connection.execute(
        """
        SELECT 1
        FROM save_snapshot_objects
        WHERE object_hash = ?
        """,
        (object_hash,),
    ).fetchone()
    return row is not None
