"""Import-safe media view models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bragi.persistence.models import MediaAssetRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.content_rating import content_exceeds_rating
from bragi.services.model_preferences import roleplay_model_preference

_IMAGE_TO_VIDEO_CAPABILITIES = frozenset(
    {
        "image_to_video",
        "image_plus_text_to_video",
        "image_text_to_video",
        "image_animation",
    }
)
_CHARACTER_NAME_MEDIA_KINDS = frozenset({"character_image", "character_reference"})
_CHARACTER_REFERENCE_RELATION = "reference_image"


@dataclass(frozen=True)
class MediaSourceModel:
    id: str
    type: str
    mime_type: str
    prompt_preview: str
    source_message_id: str | None
    created_at: str | None


@dataclass(frozen=True)
class MediaImageModel:
    id: str
    source_message_id: str | None
    source_media_asset_id: str | None
    type: str
    path: str
    thumbnail_path: str | None
    mime_type: str
    prompt: str
    provider: str
    model: str
    status: str
    created_at: str | None
    source_message: str | None
    prompt_preview: str = ""
    character_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_media: MediaSourceModel | None = None
    file_available: bool = True
    can_animate: bool = False
    is_character_reference: bool = False
    can_set_character_reference: bool = False


@dataclass(frozen=True)
class MediaModel:
    latest_scene_media: MediaImageModel | None
    latest_scene_image: MediaImageModel | None
    image_history: tuple[MediaImageModel, ...]
    media_history: tuple[MediaImageModel, ...]
    image_animation_available: bool = False
    character_reference_image: MediaImageModel | None = None


def build_media_model(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    providers: Mapping[str, object] | None = None,
    media_dir: Path | None = None,
    allowed_rating: str | None = None,
) -> MediaModel:
    source_messages = {
        message.id: message.body
        for message in repositories.list_messages(save_id)
    }
    image_animation_available = _image_animation_available(
        repositories=repositories,
        providers=providers,
        save_id=save_id,
    )
    all_media_assets = [
        asset
        for asset in repositories.list_media_assets(save_id)
        if (
            asset.source_message_id is None
            or asset.source_message_id in source_messages
        )
        and not _is_character_text_attachment_media(asset)
    ]
    all_media_assets_by_id = {asset.id: asset for asset in all_media_assets}
    media_assets = [
        asset
        for asset in all_media_assets
        if not _media_asset_exceeds_rating(
            asset,
            allowed_rating=allowed_rating,
            source_messages=source_messages,
            media_assets_by_id=all_media_assets_by_id,
        )
    ]
    media_asset_by_id = {asset.id: asset for asset in media_assets}
    character_names_by_id = {
        character.id: character.name
        for character in repositories.list_characters(save_id)
    }
    character_reference_asset_ids = _character_reference_asset_ids(
        repositories=repositories,
        save_id=save_id,
        media_asset_by_id=media_asset_by_id,
    )
    media: list[MediaImageModel] = []
    for asset in reversed(media_assets):
        metadata = _metadata(asset)
        file_available = _file_available(asset, media_dir=media_dir)
        is_character_reference = asset.id in character_reference_asset_ids
        media.append(
            _to_image_model(
                asset,
                source_messages=source_messages,
                image_animation_available=image_animation_available,
                source_asset=(
                    media_asset_by_id.get(asset.source_media_asset_id)
                    if asset.source_media_asset_id is not None
                    else None
                ),
                metadata=metadata,
                character_name=_character_media_name(
                    metadata,
                    character_names_by_id=character_names_by_id,
                ),
                file_available=file_available,
                is_character_reference=is_character_reference,
                can_set_character_reference=False,
            )
        )
    images = [asset for asset in media if asset.type == "image"]
    return MediaModel(
        latest_scene_media=media[0] if media else None,
        latest_scene_image=images[0] if images else None,
        image_history=tuple(images),
        media_history=tuple(media),
        image_animation_available=image_animation_available,
        character_reference_image=next(
            (asset for asset in media if asset.is_character_reference),
            None,
        ),
    )


def _media_asset_exceeds_rating(
    asset: MediaAssetRecord,
    *,
    allowed_rating: str | None,
    source_messages: Mapping[str, str],
    media_assets_by_id: Mapping[str, MediaAssetRecord],
    visited_asset_ids: frozenset[str] = frozenset(),
) -> bool:
    if allowed_rating is None:
        return False
    if content_exceeds_rating(asset.prompt, allowed_rating=allowed_rating):
        return True
    if asset.source_message_id is not None:
        source_message = source_messages.get(asset.source_message_id)
        if source_message is not None and content_exceeds_rating(
            source_message,
            allowed_rating=allowed_rating,
        ):
            return True
    source_asset_id = asset.source_media_asset_id
    if source_asset_id is None or source_asset_id in visited_asset_ids:
        return False
    source_asset = media_assets_by_id.get(source_asset_id)
    if source_asset is None:
        return False
    return _media_asset_exceeds_rating(
        source_asset,
        allowed_rating=allowed_rating,
        source_messages=source_messages,
        media_assets_by_id=media_assets_by_id,
        visited_asset_ids=visited_asset_ids | {asset.id},
    )


def _to_image_model(
    asset: MediaAssetRecord,
    *,
    source_messages: dict[str, str],
    image_animation_available: bool,
    source_asset: MediaAssetRecord | None,
    metadata: dict[str, Any],
    character_name: str | None,
    file_available: bool,
    is_character_reference: bool,
    can_set_character_reference: bool,
) -> MediaImageModel:
    source_message = (
        source_messages.get(asset.source_message_id)
        if asset.source_message_id is not None
        else None
    )
    prompt_preview = _prompt_preview(
        prompt=asset.prompt,
        source_message=source_message,
    )
    source_media = (
        _to_source_model(source_asset, source_messages=source_messages)
        if source_asset is not None
        else None
    )
    return MediaImageModel(
        id=asset.id,
        source_message_id=asset.source_message_id,
        source_media_asset_id=asset.source_media_asset_id,
        type=asset.type,
        path=asset.path,
        thumbnail_path=asset.thumbnail_path,
        mime_type=asset.mime_type,
        prompt=prompt_preview,
        provider=asset.provider,
        model=asset.model,
        status=asset.status,
        created_at=asset.created_at,
        source_message=source_message,
        prompt_preview=prompt_preview,
        character_name=character_name,
        metadata=metadata,
        source_media=source_media,
        file_available=file_available,
        can_animate=(
            image_animation_available
            and asset.type == "image"
            and asset.status == "succeeded"
            and asset.source_message_id is not None
        ),
        is_character_reference=is_character_reference,
        can_set_character_reference=can_set_character_reference,
    )


def _to_source_model(
    asset: MediaAssetRecord,
    *,
    source_messages: dict[str, str],
) -> MediaSourceModel:
    source_message = (
        source_messages.get(asset.source_message_id)
        if asset.source_message_id is not None
        else None
    )
    return MediaSourceModel(
        id=asset.id,
        type=asset.type,
        mime_type=asset.mime_type,
        prompt_preview=_prompt_preview(
            prompt=asset.prompt,
            source_message=source_message,
        ),
        source_message_id=asset.source_message_id,
        created_at=asset.created_at,
    )


def _metadata(asset: MediaAssetRecord) -> dict[str, Any]:
    try:
        value = json.loads(asset.metadata_json)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _is_character_text_attachment_media(asset: MediaAssetRecord) -> bool:
    kind = _metadata(asset).get("kind")
    return isinstance(kind, str) and kind.startswith("character_text_")


def _character_media_name(
    metadata: Mapping[str, Any],
    *,
    character_names_by_id: Mapping[str, str],
) -> str | None:
    if metadata.get("kind") not in _CHARACTER_NAME_MEDIA_KINDS:
        return None
    character_id = metadata.get("character_id")
    if isinstance(character_id, str):
        current_name = character_names_by_id.get(character_id)
        if current_name:
            return current_name
    stored_name = metadata.get("character_name")
    if isinstance(stored_name, str):
        normalized = stored_name.strip()
        return normalized or None
    return None


def _character_reference_asset_ids(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    media_asset_by_id: dict[str, MediaAssetRecord],
) -> frozenset[str]:
    asset_ids: set[str] = set()
    for link in repositories.list_entity_links(save_id):
        if (
            link.entity_type == "character"
            and link.target_type == "media_asset"
            and link.relation == _CHARACTER_REFERENCE_RELATION
        ):
            asset = media_asset_by_id.get(link.target_id)
            if (
                asset is not None
                and asset.type == "image"
                and asset.status == "succeeded"
            ):
                asset_ids.add(asset.id)
    return frozenset(asset_ids)


def _file_available(
    asset: MediaAssetRecord,
    *,
    media_dir: Path | None,
) -> bool:
    if media_dir is None:
        return True
    try:
        media_root = media_dir.resolve()
        path = (media_dir / asset.path).resolve()
    except OSError:
        return False
    return path.is_relative_to(media_root) and path.is_file()


def _image_animation_available(
    *,
    repositories: PersistenceRepositories,
    providers: Mapping[str, object] | None,
    save_id: str,
) -> bool:
    if not providers:
        return False
    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="image_animation",
    )
    if preference is None:
        return False
    provider = providers.get(preference.provider)
    if provider is None or not callable(getattr(provider, "generate_video", None)):
        return False
    for model in repositories.list_provider_models(preference.provider):
        if model.model_id != preference.model_id:
            continue
        if not model.available:
            return False
        return bool(
            _IMAGE_TO_VIDEO_CAPABILITIES
            & {_normalized_capability(value) for value in model.capabilities}
        )
    return False


def _normalized_capability(value: str) -> str:
    return value.lower().replace("-", "_")


def _prompt_preview(
    *,
    prompt: str,
    source_message: str | None,
    limit: int = 160,
) -> str:
    source = source_message.strip() if source_message is not None else ""
    text = source or prompt.strip()
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3].rstrip()}..."
