"""Platform storage path resolution."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoragePaths:
    data_dir: Path
    database_path: Path
    media_dir: Path
    state_dir: Path
    cache_dir: Path


def resolve_storage_paths(
    home: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> StoragePaths:
    resolved_env = os.environ if env is None else env
    resolved_platform = os.name if platform_name is None else platform_name
    if resolved_platform == "nt":
        return _resolve_windows_storage_paths(home=home, env=resolved_env)

    base_home: Path | None = None

    def fallback_home() -> Path:
        nonlocal base_home
        if base_home is None:
            home_value = resolved_env.get("HOME")
            base_home = home or (Path(home_value) if home_value else Path.home())
        return base_home

    data_home = _resolve_xdg_home_lazy(
        "XDG_DATA_HOME",
        lambda: fallback_home() / ".local" / "share",
        env=resolved_env,
    )
    state_home = _resolve_xdg_home_lazy(
        "XDG_STATE_HOME",
        lambda: fallback_home() / ".local" / "state",
        env=resolved_env,
    )
    cache_home = _resolve_xdg_home_lazy(
        "XDG_CACHE_HOME",
        lambda: fallback_home() / ".cache",
        env=resolved_env,
    )

    data_dir = data_home / "bragi"
    return StoragePaths(
        data_dir=data_dir,
        database_path=data_dir / "bragi.sqlite3",
        media_dir=data_dir / "media",
        state_dir=state_home / "bragi",
        cache_dir=cache_home / "bragi",
    )


def get_storage_paths(home: Path | None = None) -> StoragePaths:
    return resolve_storage_paths(home)


def _resolve_windows_storage_paths(
    *,
    home: Path | None,
    env: Mapping[str, str],
) -> StoragePaths:
    local_app_data = _absolute_env_path("LOCALAPPDATA", env)
    if local_app_data is None:
        base_home = home or _absolute_env_path("USERPROFILE", env) or Path.home()
        local_app_data = base_home / "AppData" / "Local"

    data_dir = local_app_data / "Bragi"
    return StoragePaths(
        data_dir=data_dir,
        database_path=data_dir / "bragi.sqlite3",
        media_dir=data_dir / "media",
        state_dir=data_dir / "state",
        cache_dir=data_dir / "cache",
    )


def _absolute_env_path(env_name: str, env: Mapping[str, str]) -> Path | None:
    configured = env.get(env_name)
    if configured:
        path = Path(configured)
        if path.is_absolute():
            return path
    return None


def resolve_xdg_home(
    env_name: str,
    fallback: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    resolved_env = os.environ if env is None else env
    configured = resolved_env.get(env_name)
    if configured:
        path = Path(configured)
        if path.is_absolute():
            return path
    return fallback


def _resolve_xdg_home_lazy(
    env_name: str,
    fallback_factory: Callable[[], Path],
    *,
    env: Mapping[str, str],
) -> Path:
    configured = env.get(env_name)
    if configured:
        path = Path(configured)
        if path.is_absolute():
            return path
    return fallback_factory()
