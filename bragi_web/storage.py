"""Storage path resolution for the web app."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WebStoragePaths:
    data_dir: Path
    database_path: Path
    media_dir: Path
    state_dir: Path
    cache_dir: Path
    temp_dir: Path


class StorageConfigurationError(ValueError):
    """Raised when web storage environment configuration is invalid."""


def resolve_web_storage_paths(
    *,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> WebStoragePaths:
    resolved_env = os.environ if env is None else env
    override = resolved_env.get("BRAGI_WEB_DATA_DIR", "").strip()
    if override:
        data_dir = Path(override).expanduser()
        if not data_dir.is_absolute():
            raise StorageConfigurationError(
                "BRAGI_WEB_DATA_DIR must be an absolute path: "
                f"{override}"
            )
        return WebStoragePaths(
            data_dir=data_dir,
            database_path=data_dir / "bragi.sqlite3",
            media_dir=data_dir / "media",
            state_dir=data_dir / "state",
            cache_dir=data_dir / "cache",
            temp_dir=data_dir / "tmp",
        )

    base_home = home or Path(resolved_env.get("HOME", str(Path.home())))
    data_home = _xdg_path(
        resolved_env,
        "XDG_DATA_HOME",
        base_home / ".local" / "share",
    )
    state_home = _xdg_path(
        resolved_env,
        "XDG_STATE_HOME",
        base_home / ".local" / "state",
    )
    cache_home = _xdg_path(resolved_env, "XDG_CACHE_HOME", base_home / ".cache")
    data_dir = data_home / "bragi-web"
    state_dir = state_home / "bragi-web"
    return WebStoragePaths(
        data_dir=data_dir,
        database_path=data_dir / "bragi.sqlite3",
        media_dir=data_dir / "media",
        state_dir=state_dir,
        cache_dir=cache_home / "bragi-web",
        temp_dir=state_dir / "tmp",
    )


def _xdg_path(env: Mapping[str, str], key: str, fallback: Path) -> Path:
    configured = env.get(key)
    if configured:
        path = Path(configured)
        if path.is_absolute():
            return path
    return fallback
