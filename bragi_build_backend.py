"""PEP 517 backend wrapper that prepares Bragi Web SPA package assets."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, cast


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ensure_packaged_spa()
    return cast(
        str,
        _build_meta().build_wheel(
            wheel_directory,
            config_settings=config_settings,
            metadata_directory=metadata_directory,
        ),
    )


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    _ensure_packaged_spa()
    return cast(
        str,
        _build_meta().build_sdist(
            sdist_directory,
            config_settings=config_settings,
        ),
    )


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return cast(
        str,
        _build_meta().build_editable(
            wheel_directory,
            config_settings=config_settings,
            metadata_directory=metadata_directory,
        ),
    )


def get_requires_for_build_wheel(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return cast(
        list[str],
        _build_meta().get_requires_for_build_wheel(
            config_settings=config_settings,
        ),
    )


def get_requires_for_build_sdist(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return cast(
        list[str],
        _build_meta().get_requires_for_build_sdist(
            config_settings=config_settings,
        ),
    )


def get_requires_for_build_editable(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return cast(
        list[str],
        _build_meta().get_requires_for_build_editable(
            config_settings=config_settings,
        ),
    )


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return cast(
        str,
        _build_meta().prepare_metadata_for_build_wheel(
            metadata_directory,
            config_settings=config_settings,
        ),
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return cast(
        str,
        _build_meta().prepare_metadata_for_build_editable(
            metadata_directory,
            config_settings=config_settings,
        ),
    )


def _ensure_packaged_spa() -> None:
    root = _project_root()
    frontend_dir = root / "frontend"
    has_frontend_source = (frontend_dir / "package.json").is_file()

    if has_frontend_source:
        _build_frontend(frontend_dir)

    if _static_assets_available(root / "bragi_web" / "static"):
        return

    if has_frontend_source:
        raise RuntimeError(
            "Frontend build finished but did not produce packaged SPA assets "
            "under bragi_web/static."
        )
    raise RuntimeError(
        "Packaged SPA assets are missing. Build from a source checkout with "
        "frontend/package.json available, or build from an sdist that already "
        "contains bragi_web/static."
    )


def _build_frontend(frontend_dir: Path) -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError(
            "Cannot build packaged SPA assets because npm is not available. "
            "Install Node.js/npm and run `npm ci --prefix frontend` before "
            "`uv build`."
        )
    if not (frontend_dir / "node_modules").is_dir():
        raise RuntimeError(
            "Cannot build packaged SPA assets because frontend/node_modules is "
            "missing. Run `npm ci --prefix frontend` before `uv build`."
        )
    try:
        subprocess.run([npm, "run", "build"], cwd=frontend_dir, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Frontend build failed while preparing packaged SPA assets."
        ) from exc


def _static_assets_available(static_dir: Path) -> bool:
    if not (static_dir / "index.html").is_file():
        return False
    assets_dir = static_dir / "assets"
    return assets_dir.is_dir() and any(
        path.is_file() for path in assets_dir.rglob("*")
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _build_meta() -> Any:
    from setuptools import build_meta  # type: ignore[import-untyped]

    return build_meta
