from __future__ import annotations

from pathlib import Path

import pytest

from bragi.persistence.paths import resolve_storage_paths


def test_resolve_storage_paths_uses_xdg_locations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_home = tmp_path / "xdg-data"
    state_home = tmp_path / "xdg-state"
    cache_home = tmp_path / "xdg-cache"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

    paths = resolve_storage_paths(platform_name="posix")

    assert paths.data_dir == data_home / "bragi"
    assert paths.database_path == data_home / "bragi" / "bragi.sqlite3"
    assert paths.media_dir == data_home / "bragi" / "media"
    assert paths.state_dir == state_home / "bragi"
    assert paths.cache_dir == cache_home / "bragi"


def test_resolve_storage_paths_uses_user_local_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    paths = resolve_storage_paths(platform_name="posix")

    assert paths.data_dir == home / ".local" / "share" / "bragi"
    assert (
        paths.database_path
        == home / ".local" / "share" / "bragi" / "bragi.sqlite3"
    )
    assert paths.media_dir == home / ".local" / "share" / "bragi" / "media"
    assert paths.state_dir == home / ".local" / "state" / "bragi"
    assert paths.cache_dir == home / ".cache" / "bragi"


def test_resolve_storage_paths_ignores_relative_xdg_locations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    first_cwd = tmp_path / "first-cwd"
    second_cwd = tmp_path / "second-cwd"
    first_cwd.mkdir()
    second_cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", "relative-data")
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")
    monkeypatch.setenv("XDG_CACHE_HOME", "relative-cache")

    monkeypatch.chdir(first_cwd)
    first_paths = resolve_storage_paths(platform_name="posix")
    monkeypatch.chdir(second_cwd)
    second_paths = resolve_storage_paths(platform_name="posix")

    assert first_paths == second_paths
    assert first_paths.data_dir == home / ".local" / "share" / "bragi"
    assert first_paths.database_path == (
        home / ".local" / "share" / "bragi" / "bragi.sqlite3"
    )
    assert first_paths.media_dir == home / ".local" / "share" / "bragi" / "media"
    assert first_paths.state_dir == home / ".local" / "state" / "bragi"
    assert first_paths.cache_dir == home / ".cache" / "bragi"
    assert first_paths.data_dir.is_absolute()
    assert first_paths.state_dir.is_absolute()
    assert first_paths.cache_dir.is_absolute()


def test_resolve_storage_paths_uses_local_app_data_on_windows(tmp_path: Path) -> None:
    local_app_data = tmp_path / "LocalAppData"

    paths = resolve_storage_paths(
        env={"LOCALAPPDATA": str(local_app_data), "HOME": str(tmp_path / "home")},
        platform_name="nt",
    )

    expected_base = local_app_data / "Bragi"
    assert paths.data_dir == expected_base
    assert paths.database_path == expected_base / "bragi.sqlite3"
    assert paths.media_dir == expected_base / "media"
    assert paths.state_dir == expected_base / "state"
    assert paths.cache_dir == expected_base / "cache"


@pytest.mark.parametrize("configured", ["relative", ""])
def test_resolve_storage_paths_ignores_unsafe_local_app_data_on_windows(
    configured: str,
    tmp_path: Path,
) -> None:
    home = tmp_path / "Users" / "Player"

    paths = resolve_storage_paths(
        env={"LOCALAPPDATA": configured, "USERPROFILE": str(home)},
        platform_name="nt",
    )

    expected_base = home / "AppData" / "Local" / "Bragi"
    assert paths.data_dir == expected_base
    assert paths.database_path == expected_base / "bragi.sqlite3"
    assert paths.media_dir == expected_base / "media"
    assert paths.state_dir == expected_base / "state"
    assert paths.cache_dir == expected_base / "cache"
    assert paths.data_dir.is_absolute()
