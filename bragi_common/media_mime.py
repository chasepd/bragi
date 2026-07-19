from __future__ import annotations

INERT_MEDIA_MIME_TYPE = "application/octet-stream"
SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
SUPPORTED_VIDEO_MIME_TYPES = frozenset({"video/mp4", "video/webm"})
SUPPORTED_MEDIA_MIME_TYPES = SUPPORTED_IMAGE_MIME_TYPES | SUPPORTED_VIDEO_MIME_TYPES


def canonical_media_mime_type(mime_type: str | None) -> str | None:
    if mime_type is None:
        return None
    normalized = mime_type.split(";", 1)[0].strip().lower()
    return normalized or None


def imported_media_mime_type(
    mime_type: str | None,
    *,
    media_type: str | None = None,
) -> str:
    normalized = canonical_media_mime_type(mime_type)
    if media_type == "video":
        if normalized in SUPPORTED_VIDEO_MIME_TYPES:
            return normalized
        return INERT_MEDIA_MIME_TYPE
    if media_type is not None:
        if normalized in SUPPORTED_IMAGE_MIME_TYPES:
            return normalized
        return INERT_MEDIA_MIME_TYPE
    if normalized in SUPPORTED_MEDIA_MIME_TYPES:
        return normalized
    return INERT_MEDIA_MIME_TYPE


def safe_served_media_mime_type(mime_type: str | None) -> str:
    normalized = canonical_media_mime_type(mime_type)
    if (
        normalized in SUPPORTED_MEDIA_MIME_TYPES
        or normalized == INERT_MEDIA_MIME_TYPE
    ):
        return normalized
    return INERT_MEDIA_MIME_TYPE
