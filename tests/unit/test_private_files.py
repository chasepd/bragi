from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import bragi.private_files as private_files
from bragi.private_files import ensure_private_file, write_private_bytes


def test_ensure_private_file_repairs_read_only_file_without_changing_contents(
    tmp_path: Path,
) -> None:
    private_path = tmp_path / "private.txt"
    private_path.write_bytes(b"existing secret")
    private_path.chmod(0o400)

    ensure_private_file(private_path)

    assert private_path.read_bytes() == b"existing secret"
    _assert_private_file_mode(private_path)


def test_ensure_private_file_does_not_chmod_symlink_target(tmp_path: Path) -> None:
    target_path = tmp_path / "target.txt"
    private_path = tmp_path / "private.txt"
    target_path.write_bytes(b"public target")
    target_path.chmod(0o644)
    try:
        private_path.symlink_to(target_path)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    try:
        ensure_private_file(private_path)
    except OSError:
        pass
    else:
        assert not private_path.is_symlink()
        assert private_path.is_file()
        _assert_private_file_mode(private_path)

    assert target_path.read_bytes() == b"public target"
    _assert_mode(target_path, 0o644)


def test_write_private_bytes_does_not_truncate_symlink_target(tmp_path: Path) -> None:
    target_path = tmp_path / "target.txt"
    private_path = tmp_path / "private.txt"
    target_path.write_bytes(b"public target")
    target_path.chmod(0o644)
    try:
        private_path.symlink_to(target_path)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    try:
        write_private_bytes(private_path, b"secret")
    except OSError:
        pass
    else:
        assert not private_path.is_symlink()
        assert private_path.is_file()
        assert private_path.read_bytes() == b"secret"
        _assert_private_file_mode(private_path)

    assert target_path.read_bytes() == b"public target"
    _assert_mode(target_path, 0o644)


def test_private_file_helpers_tolerate_missing_posix_permission_apis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private_path = tmp_path / "private.txt"
    private_path.write_bytes(b"existing secret")

    monkeypatch.delattr(private_files.os, "fchmod", raising=False)

    ensure_private_file(private_path)

    assert private_path.read_bytes() == b"existing secret"


def _assert_private_file_mode(path: Path) -> None:
    _assert_mode(path, 0o600)


def _assert_mode(path: Path, expected: int) -> None:
    if os.name == "nt":
        return
    assert stat.S_IMODE(path.stat().st_mode) == expected
