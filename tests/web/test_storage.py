from pathlib import Path

import pytest

from bragi_web.storage import StorageConfigurationError, resolve_web_storage_paths


def test_web_storage_uses_separate_xdg_app_name() -> None:
    paths = resolve_web_storage_paths(
        home=Path("/home/example"),
        env={},
    )

    assert paths.data_dir == Path("/home/example/.local/share/bragi-web")
    assert paths.database_path == paths.data_dir / "bragi.sqlite3"
    assert paths.media_dir == paths.data_dir / "media"
    assert paths.state_dir == Path("/home/example/.local/state/bragi-web")
    assert paths.cache_dir == Path("/home/example/.cache/bragi-web")


def test_web_storage_override_collocates_web_files() -> None:
    paths = resolve_web_storage_paths(
        env={"BRAGI_WEB_DATA_DIR": "/tmp/bragi-web-data"},
    )

    assert paths.data_dir == Path("/tmp/bragi-web-data")
    assert paths.database_path == Path("/tmp/bragi-web-data/bragi.sqlite3")
    assert paths.media_dir == Path("/tmp/bragi-web-data/media")
    assert paths.state_dir == Path("/tmp/bragi-web-data/state")
    assert paths.cache_dir == Path("/tmp/bragi-web-data/cache")


def test_web_storage_rejects_relative_override() -> None:
    with pytest.raises(
        StorageConfigurationError,
        match="BRAGI_WEB_DATA_DIR must be an absolute path",
    ):
        resolve_web_storage_paths(
            env={"BRAGI_WEB_DATA_DIR": "relative/bragi-web-data"},
        )
