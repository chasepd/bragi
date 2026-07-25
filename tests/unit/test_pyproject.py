from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_declares_keyring_runtime_dependency() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert any(
        dependency.split(">=", maxsplit=1)[0] == "keyring"
        for dependency in dependencies
    )


def test_pyproject_declares_mit_license() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["license"] == "MIT"
    assert "LICENSE" in pyproject["project"]["license-files"]


def test_pyproject_declares_web_runtime_dependencies_and_scripts() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = {
        dependency.split(">=", maxsplit=1)[0]
        for dependency in pyproject["project"]["dependencies"]
    }
    scripts = pyproject["project"]["scripts"]

    assert {"fastapi", "python-multipart", "uvicorn[standard]"}.issubset(
        dependencies
    )
    assert scripts["bragi"] == "bragi.app:cli_main"
    assert scripts["bragi-web"] == "bragi_web.main:main"


def test_pyproject_typechecks_validation_tooling() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    mypy_files = set(pyproject["tool"]["mypy"]["files"])

    assert ".codex/hooks" in mypy_files
    assert ".codex/tools" in mypy_files
