from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from bragi.zip_safety import validate_zip_directory


def test_validate_zip_directory_rejects_member_count_before_open(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "many-members.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for index in range(5):
            archive.writestr(f"entry-{index}.txt", b"")

    with pytest.raises(ValueError, match="too many members"):
        validate_zip_directory(archive_path, max_members=4)


def test_validate_zip_directory_accepts_bounded_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "bounded.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", b"{}")

    validate_zip_directory(archive_path)
