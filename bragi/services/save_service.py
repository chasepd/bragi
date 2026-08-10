"""Save creation and loading service."""

from __future__ import annotations

from pathlib import Path

from bragi.app_logging import log_error_event, log_event
from bragi.persistence.models import MediaAssetRecord, SaveDetailsRecord, SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.dating_route_profile_service import (
    enqueue_dating_route_profile_enrichment,
)
from bragi.services.dating_route_service import DatingRouteService
from bragi.services.turn_snapshot_service import TurnSnapshotService


class SaveService:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories

    def create_save(
        self,
        *,
        scenario_id: str,
        title: str,
        owner_user_id: str | None = None,
    ) -> SaveRecord:
        record = self.repositories.create_save(
            scenario_id=scenario_id,
            title=title,
            owner_user_id=owner_user_id,
        )
        log_event(
            "save.created",
            save_id=record.id,
            scenario_id=scenario_id,
            title_chars=len(title),
        )
        TurnSnapshotService(self.repositories).capture_baseline_snapshot(
            record.id,
            reason="save_created",
        )
        return record

    def list_saves(self) -> list[SaveRecord]:
        return self.repositories.list_saves()

    def load_save(self, save_id: str) -> SaveDetailsRecord:
        details = self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        DatingRouteService(self.repositories).seed_routes_for_save(save_id)
        enqueue_dating_route_profile_enrichment(
            self.repositories,
            save_id=save_id,
        )
        self.repositories.touch_save_last_opened(save_id)
        log_event(
            "save.loaded",
            save_id=save_id,
            scenario_id=details.scenario.id,
            message_count=len(details.messages),
        )
        return details

    def rename_save(self, save_id: str, title: str) -> SaveRecord:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")
        text = title.strip()
        if not text:
            raise ValueError("Save title is required")

        record = self.repositories.update_save_title(save_id=save_id, title=text)
        log_event(
            "save.renamed",
            save_id=record.id,
            title_chars=len(record.title),
        )
        return record

    def delete_save(self, save_id: str, *, media_dir: Path | None = None) -> bool:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")

        media_assets = tuple(self.repositories.list_all_media_assets(save_id))
        snapshot_media_assets: tuple[MediaAssetRecord, ...] = ()
        try:
            snapshot_media_assets = TurnSnapshotService(
                self.repositories
            ).media_asset_records_from_save_snapshots(save_id)
        except ValueError as exc:
            log_error_event(
                "save.snapshot_media_collection_failed_before_delete",
                save_id=save_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        media_assets_for_cleanup = _dedupe_media_assets(
            (*media_assets, *snapshot_media_assets)
        )
        deleted = self.repositories.delete_save(save_id)
        media_cleanup_failed_paths: tuple[str, ...] = ()
        snapshot_object_delete_count = 0
        if deleted and media_dir is not None:
            try:
                _delete_save_media_files(
                    media_assets_for_cleanup,
                    media_dir=media_dir,
                )
            except SaveMediaDeletionError as exc:
                media_cleanup_failed_paths = exc.failed_paths
                log_error_event(
                    "save.media_file_cleanup_failed_after_delete",
                    save_id=save_id,
                    failed_path_count=len(exc.failed_paths),
                    failed_paths=list(exc.failed_paths),
                )
        if deleted:
            try:
                snapshot_object_delete_count = TurnSnapshotService(
                    self.repositories
                ).prune_unreferenced_snapshot_objects()
            except ValueError as exc:
                log_error_event(
                    "save.snapshot_object_cleanup_failed_after_delete",
                    save_id=save_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
        log_event(
            "save.deleted",
            save_id=save_id,
            deleted=deleted,
            media_file_count=sum(
                1
                for asset in media_assets_for_cleanup
                for path in (asset.path, asset.thumbnail_path)
                if path
            ),
            media_cleanup_failed_path_count=len(media_cleanup_failed_paths),
            snapshot_media_file_count=sum(
                1
                for asset in snapshot_media_assets
                for path in (asset.path, asset.thumbnail_path)
                if path
            ),
            snapshot_object_delete_count=snapshot_object_delete_count,
        )
        return deleted


class SaveMediaDeletionError(RuntimeError):
    def __init__(self, failed_paths: tuple[str, ...]) -> None:
        self.failed_paths = failed_paths
        super().__init__("Could not delete save media files")


def _delete_save_media_files(
    media_assets: tuple[MediaAssetRecord, ...],
    *,
    media_dir: Path,
) -> None:
    media_root = media_dir.resolve()
    failed_paths: list[str] = []
    for asset in media_assets:
        for relative_path in (asset.thumbnail_path, asset.path):
            if not relative_path:
                continue
            path = media_dir / relative_path
            try:
                if not path.resolve().is_relative_to(media_root):
                    log_event(
                        "save.media_file_delete_skipped",
                        save_id=asset.save_id,
                        asset_id=asset.id,
                        path=relative_path,
                        reason="outside_media_dir",
                    )
                    failed_paths.append(relative_path)
                    continue
                if path.is_file():
                    path.unlink()
            except OSError as exc:
                log_event(
                    "save.media_file_delete_failed",
                    save_id=asset.save_id,
                    asset_id=asset.id,
                    path=relative_path,
                    error_type=type(exc).__name__,
                )
                failed_paths.append(relative_path)
    if failed_paths:
        raise SaveMediaDeletionError(tuple(failed_paths))


def _dedupe_media_assets(
    media_assets: tuple[MediaAssetRecord, ...],
) -> tuple[MediaAssetRecord, ...]:
    seen: set[tuple[str, str, str | None]] = set()
    deduped: list[MediaAssetRecord] = []
    for asset in media_assets:
        key = (asset.id, asset.path, asset.thumbnail_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(asset)
    return tuple(deduped)
