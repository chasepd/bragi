"""Low-allocation ZIP central-directory preflight checks."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, BinaryIO, cast

MAX_BUNDLE_ZIP_MEMBERS = 4_096
MAX_BUNDLE_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024


class ZipSafetyError(ValueError):
    """Raised when a ZIP directory exceeds configured resource bounds."""


def validate_zip_directory(
    path: Path,
    *,
    max_members: int = MAX_BUNDLE_ZIP_MEMBERS,
    max_central_directory_bytes: int = MAX_BUNDLE_CENTRAL_DIRECTORY_BYTES,
) -> None:
    """Reject oversized directories before ``ZipFile`` materializes members."""
    end_record_reader = cast(
        Any,
        getattr(zipfile, "_EndRecData", None),
    )
    entries_index = getattr(zipfile, "_ECD_ENTRIES_TOTAL", None)
    size_index = getattr(zipfile, "_ECD_SIZE", None)
    if (
        end_record_reader is None
        or not isinstance(entries_index, int)
        or not isinstance(size_index, int)
    ):
        raise zipfile.BadZipFile("ZIP preflight is unavailable")
    with path.open("rb") as source:
        end_record = end_record_reader(cast(BinaryIO, source))
    if end_record is None:
        raise zipfile.BadZipFile("ZIP end record is missing")
    member_count = int(end_record[entries_index])
    central_directory_size = int(end_record[size_index])
    if member_count < 0 or member_count > max_members:
        raise ZipSafetyError("ZIP archive has too many members")
    if (
        central_directory_size < 0
        or central_directory_size > max_central_directory_bytes
    ):
        raise ZipSafetyError("ZIP central directory is too large")
