from __future__ import annotations

from pathlib import Path


def test_readme_uses_reproducible_frontend_dependency_install() -> None:
    body = Path("README.md").read_text(encoding="utf-8")

    assert "npm ci --prefix frontend" in body
    assert "npm install --prefix frontend" not in body
