"""Content-rating provenance helpers for persisted media."""

from __future__ import annotations

import json
from collections.abc import Mapping

from bragi.content_rating_instructions import (
    CONTENT_RATING_UNCLASSIFIED,
    content_rating_exceeds,
    maximum_content_rating,
)

_SOURCE_ASSET_ID_KEYS = (
    "source_media_asset_id",
    "source_character_reference_asset_id",
)
_SOURCE_ASSET_IDS_KEYS = (
    "source_media_asset_ids",
    "source_character_reference_asset_ids",
)


def media_asset_content_rating(
    asset: object,
    *,
    media_assets_by_id: Mapping[str, object],
    source_messages: Mapping[str, object],
    character_text_messages: Mapping[str, object] | None = None,
    characters: Mapping[str, object] | None = None,
    visited_asset_ids: frozenset[str] = frozenset(),
) -> str:
    """Return the strictest rating across an asset's complete provenance."""

    metadata = _metadata(asset)
    ratings = [
        str(metadata.get("content_rating", CONTENT_RATING_UNCLASSIFIED)),
    ]
    source_message_id = getattr(asset, "source_message_id", None)
    if source_message_id is not None:
        source_message = source_messages.get(source_message_id)
        ratings.append(
            str(getattr(source_message, "content_rating", CONTENT_RATING_UNCLASSIFIED))
            if source_message is not None
            else CONTENT_RATING_UNCLASSIFIED
        )
    if character_text_messages is not None:
        ratings.extend(
            _metadata_source_ratings(
                metadata,
                keys=("text_message_id",),
                sources=character_text_messages,
            )
        )
    if characters is not None:
        ratings.extend(
            _metadata_source_ratings(
                metadata,
                keys=("character_id", "sender_character_id"),
                sources=characters,
            )
        )
    for source_asset_id in _source_asset_ids(asset, metadata=metadata):
        if source_asset_id in visited_asset_ids:
            continue
        source_asset = media_assets_by_id.get(source_asset_id)
        if source_asset is None:
            ratings.append(CONTENT_RATING_UNCLASSIFIED)
            continue
        ratings.append(
            media_asset_content_rating(
                source_asset,
                media_assets_by_id=media_assets_by_id,
                source_messages=source_messages,
                character_text_messages=character_text_messages,
                characters=characters,
                visited_asset_ids=visited_asset_ids
                | {str(getattr(asset, "id", ""))},
            )
        )
    return maximum_content_rating(
        tuple(ratings),
        default=CONTENT_RATING_UNCLASSIFIED,
    )


def media_asset_exceeds_rating(
    asset: object,
    *,
    allowed_rating: str,
    media_assets_by_id: Mapping[str, object],
    source_messages: Mapping[str, object],
    character_text_messages: Mapping[str, object] | None = None,
    characters: Mapping[str, object] | None = None,
) -> bool:
    """Return whether any media provenance exceeds a viewer's ceiling."""

    return content_rating_exceeds(
        minimum_rating=media_asset_content_rating(
            asset,
            media_assets_by_id=media_assets_by_id,
            source_messages=source_messages,
            character_text_messages=character_text_messages,
            characters=characters,
        ),
        allowed_rating=allowed_rating,
    )


def _metadata_source_ratings(
    metadata: Mapping[str, object],
    *,
    keys: tuple[str, ...],
    sources: Mapping[str, object],
) -> tuple[str, ...]:
    ratings: list[str] = []
    for key in keys:
        source_id = metadata.get(key)
        if not isinstance(source_id, str) or not source_id.strip():
            continue
        source = sources.get(source_id.strip())
        ratings.append(
            str(getattr(source, "content_rating", CONTENT_RATING_UNCLASSIFIED))
            if source is not None
            else CONTENT_RATING_UNCLASSIFIED
        )
    return tuple(ratings)


def _source_asset_ids(
    asset: object,
    *,
    metadata: Mapping[str, object],
) -> tuple[str, ...]:
    source_ids: list[str] = []
    source_media_asset_id = getattr(asset, "source_media_asset_id", None)
    if isinstance(source_media_asset_id, str) and source_media_asset_id:
        source_ids.append(source_media_asset_id)
    for key in _SOURCE_ASSET_ID_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            source_ids.append(value.strip())
    for key in _SOURCE_ASSET_IDS_KEYS:
        value = metadata.get(key)
        if not isinstance(value, list):
            continue
        source_ids.extend(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )
    return tuple(dict.fromkeys(source_ids))


def _metadata(asset: object) -> dict[str, object]:
    try:
        value = json.loads(str(getattr(asset, "metadata_json", "")))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
