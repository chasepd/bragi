"""Private local filesystem helpers for user-owned Bragi data."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _chmod_best_effort(path, 0o700)


def ensure_private_file(path: Path) -> None:
    ensure_private_dir(path.parent)
    if path.is_symlink():
        raise OSError(f"Refusing to use symlink as private file: {path}")
    nofollow_flag = os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    try:
        fd = os.open(
            path,
            os.O_WRONLY | nofollow_flag | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        fd = os.open(path, os.O_RDONLY | nofollow_flag)
    try:
        _fchmod_best_effort(fd, 0o600)
    finally:
        os.close(fd)


def write_private_bytes(path: Path, data: bytes) -> None:
    ensure_private_dir(path.parent)
    if path.is_symlink():
        raise OSError(f"Refusing to replace symlink as private file: {path}")

    temp_path: Path | None = None
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        _chmod_best_effort(temp_path, 0o600)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except (AttributeError, NotImplementedError, OSError):
        if os.name != "nt":
            raise


def _fchmod_best_effort(fd: int, mode: int) -> None:
    fchmod = getattr(os, "fchmod", None)
    if fchmod is None:
        return
    try:
        fchmod(fd, mode)
    except (AttributeError, NotImplementedError, OSError):
        if os.name != "nt":
            raise
