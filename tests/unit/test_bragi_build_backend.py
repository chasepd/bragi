from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import bragi_build_backend


def test_ensure_packaged_spa_builds_frontend_from_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    (frontend / "node_modules").mkdir()
    commands: list[tuple[list[str], Path]] = []

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
        commands.append((command, cwd))
        assert check is True
        assets = tmp_path / "bragi_web" / "static" / "assets"
        assets.mkdir(parents=True)
        (tmp_path / "bragi_web" / "static" / "index.html").write_text(
            "INDEX",
            encoding="utf-8",
        )
        (assets / "index.js").write_text("console.log('ok')", encoding="utf-8")

    monkeypatch.setattr(bragi_build_backend, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(bragi_build_backend.shutil, "which", lambda name: "npm")
    monkeypatch.setattr(bragi_build_backend.subprocess, "run", fake_run)

    bragi_build_backend._ensure_packaged_spa()

    assert commands == [(["npm", "run", "build"], frontend)]


def test_ensure_packaged_spa_accepts_prebuilt_static_without_frontend_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    static = tmp_path / "bragi_web" / "static"
    assets = static / "assets"
    assets.mkdir(parents=True)
    (static / "index.html").write_text("INDEX", encoding="utf-8")
    (assets / "index.js").write_text("console.log('ok')", encoding="utf-8")

    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("frontend build should not run")

    monkeypatch.setattr(bragi_build_backend, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(bragi_build_backend.subprocess, "run", fail_run)

    bragi_build_backend._ensure_packaged_spa()


def test_ensure_packaged_spa_fails_when_static_and_frontend_source_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(bragi_build_backend, "_project_root", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="Packaged SPA assets are missing"):
        bragi_build_backend._ensure_packaged_spa()


def test_ensure_packaged_spa_fails_before_build_when_dependencies_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(bragi_build_backend, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(bragi_build_backend.shutil, "which", lambda name: "npm")

    with pytest.raises(RuntimeError, match="frontend/node_modules is missing"):
        bragi_build_backend._ensure_packaged_spa()


def test_ensure_packaged_spa_fails_when_frontend_build_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    (frontend / "node_modules").mkdir()

    def fail_run(command: list[str], *, cwd: Path, check: bool) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr(bragi_build_backend, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(bragi_build_backend.shutil, "which", lambda name: "npm")
    monkeypatch.setattr(bragi_build_backend.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="Frontend build failed"):
        bragi_build_backend._ensure_packaged_spa()
