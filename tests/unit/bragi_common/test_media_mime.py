from __future__ import annotations

import pytest

from bragi_common.media_mime import (
    INERT_MEDIA_MIME_TYPE,
    canonical_media_mime_type,
    imported_media_mime_type,
    safe_served_media_mime_type,
)


@pytest.mark.parametrize(
    ("raw_mime_type", "expected"),
    [
        (" image/PNG ; charset=binary", "image/png"),
        ("", None),
        (None, None),
    ],
)
def test_canonical_media_mime_type_normalizes_values(
    raw_mime_type: str | None,
    expected: str | None,
) -> None:
    assert canonical_media_mime_type(raw_mime_type) == expected


@pytest.mark.parametrize("mime_type", ["image/png", "image/jpeg", "image/webp"])
def test_imported_media_mime_type_accepts_supported_images(mime_type: str) -> None:
    assert imported_media_mime_type(mime_type, media_type="image") == mime_type


@pytest.mark.parametrize("mime_type", ["video/mp4", "video/webm"])
def test_imported_media_mime_type_accepts_supported_videos(mime_type: str) -> None:
    assert imported_media_mime_type(mime_type, media_type="video") == mime_type


def test_imported_media_mime_type_inerts_mismatched_or_unsupported_values() -> None:
    assert imported_media_mime_type("video/mp4", media_type="image") == (
        INERT_MEDIA_MIME_TYPE
    )
    assert imported_media_mime_type("image/png", media_type="video") == (
        INERT_MEDIA_MIME_TYPE
    )
    assert imported_media_mime_type("text/html") == INERT_MEDIA_MIME_TYPE


@pytest.mark.parametrize(
    ("mime_type", "expected"),
    [
        ("image/png", "image/png"),
        ("video/webm", "video/webm"),
        (INERT_MEDIA_MIME_TYPE, INERT_MEDIA_MIME_TYPE),
        ("text/html", INERT_MEDIA_MIME_TYPE),
        (None, INERT_MEDIA_MIME_TYPE),
    ],
)
def test_safe_served_media_mime_type_only_serves_supported_or_inert_values(
    mime_type: str | None,
    expected: str,
) -> None:
    assert safe_served_media_mime_type(mime_type) == expected
