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
    offset_index = getattr(zipfile, "_ECD_OFFSET", None)
    location_index = getattr(zipfile, "_ECD_LOCATION", None)
    if (
        end_record_reader is None
        or not isinstance(entries_index, int)
        or not isinstance(size_index, int)
        or not isinstance(offset_index, int)
        or not isinstance(location_index, int)
    ):
        raise zipfile.BadZipFile("ZIP preflight is unavailable")
    with path.open("rb") as source:
        end_record = end_record_reader(cast(BinaryIO, source))
        if end_record is None:
            raise zipfile.BadZipFile("ZIP end record is missing")
        declared_member_count = int(end_record[entries_index])
        central_directory_size = int(end_record[size_index])
        if declared_member_count < 0 or declared_member_count > max_members:
            raise ZipSafetyError("ZIP archive has too many members")
        if (
            central_directory_size < 0
            or central_directory_size > max_central_directory_bytes
        ):
            raise ZipSafetyError("ZIP central directory is too large")
        central_directory_offset = int(end_record[offset_index])
        end_record_location = int(end_record[location_index])
        concatenated_prefix_size = (
            end_record_location
            - central_directory_size
            - central_directory_offset
        )
        start_offset = central_directory_offset + concatenated_prefix_size
        if start_offset < 0:
            raise zipfile.BadZipFile("Bad offset for central directory")
        source.seek(start_offset)
        consumed = 0
        parsed_member_count = 0
        while consumed < central_directory_size:
            header = source.read(46)
            if len(header) != 46 or header[:4] != b"PK\x01\x02":
                raise zipfile.BadZipFile("Invalid ZIP central directory")
            name_length = int.from_bytes(header[28:30], "little")
            extra_length = int.from_bytes(header[30:32], "little")
            comment_length = int.from_bytes(header[32:34], "little")
            variable_size = name_length + extra_length + comment_length
            consumed += 46 + variable_size
            if consumed > central_directory_size:
                raise zipfile.BadZipFile("Truncated ZIP central directory")
            source.seek(variable_size, 1)
            parsed_member_count += 1
            if parsed_member_count > max_members:
                raise ZipSafetyError("ZIP archive has too many members")
        if (
            consumed != central_directory_size
            or parsed_member_count != declared_member_count
        ):
            raise zipfile.BadZipFile("ZIP central directory count mismatch")
