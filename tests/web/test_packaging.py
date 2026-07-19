from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def test_wheel_includes_and_serves_packaged_spa(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    wheelhouse = tmp_path / "wheelhouse"
    target = tmp_path / "target"
    node_modules = repo / "frontend" / "node_modules"
    if not node_modules.is_dir():
        pytest.fail("frontend/node_modules is missing; run npm ci --prefix frontend")

    shutil.copytree(
        repo / "bragi",
        source / "bragi",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        repo / "bragi_common",
        source / "bragi_common",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        repo / "bragi_web",
        source / "bragi_web",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "static"),
    )
    shutil.copytree(
        repo / "frontend",
        source / "frontend",
        ignore=shutil.ignore_patterns("node_modules", ".vite", "dist"),
    )
    (source / "frontend" / "node_modules").symlink_to(
        node_modules,
        target_is_directory=True,
    )
    shutil.copy2(repo / "pyproject.toml", source / "pyproject.toml")
    shutil.copy2(repo / "README.md", source / "README.md")
    shutil.copy2(repo / "MANIFEST.in", source / "MANIFEST.in")
    shutil.copy2(repo / "bragi_build_backend.py", source / "bragi_build_backend.py")

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    subprocess.run(
        [
            "uv",
            "build",
            "--no-build-isolation",
            "--wheel",
            "--out-dir",
            str(wheelhouse),
            str(source),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
    )
    [wheel] = wheelhouse.glob("bragi-*.whl")
    with zipfile.ZipFile(wheel) as wheel_zip:
        names = set(wheel_zip.namelist())
    assert "bragi_common/__init__.py" in names
    assert "bragi_common/media_mime.py" in names
    assert "bragi/data/name_sources/README.md" in names
    assert "bragi/data/name_sources/ordinary_feminine.txt" in names
    assert "bragi/data/name_sources/ordinary_masculine.txt" in names
    assert "bragi/data/name_sources/ordinary_neutral.txt" in names
    assert "bragi_web/static/index.html" in names
    assert "bragi_web/static/manifest.webmanifest" in names
    assert "bragi_web/static/app-icon-192.png" in names
    js_assets = sorted(
        name
        for name in names
        if name.startswith("bragi_web/static/assets/") and name.endswith(".js")
    )
    css_assets = sorted(
        name
        for name in names
        if name.startswith("bragi_web/static/assets/") and name.endswith(".css")
    )
    assert js_assets
    assert css_assets
    assert {f"{name}.gz" for name in js_assets}.issubset(names)
    assert {f"{name}.gz" for name in css_assets}.issubset(names)

    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
    )
    asset_route = "/" + js_assets[0].removeprefix("bragi_web/static/")

    script = """
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

target = Path(os.environ["WHEEL_TARGET"]).resolve()
repo = Path(os.environ["REPO_ROOT"]).resolve()


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


sys.path = [
    str(target),
    *[
        str(path)
        for raw_path in sys.path
        if raw_path
        for path in [Path(raw_path).resolve()]
        if path not in {target, repo}
    ],
]


def import_from_target(module_name):
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    assert is_relative_to(module_path, target), (
        f"{module_name} imported from {module_path}, expected under {target}"
    )
    return module


import_from_target("bragi")
import_from_target("bragi_common")
import_from_target("bragi_common.media_mime")
import_from_target("bragi_web")
app_module = import_from_target("bragi_web.api.app")

from fastapi.testclient import TestClient

with TestClient(app_module.create_app(SimpleNamespace())) as client:
    root = client.get("/")
    deep_link = client.get("/deep/link")
    manifest = client.get("/manifest.webmanifest")
    asset = client.get(os.environ["ASSET_ROUTE"], headers={"Accept-Encoding": "gzip"})

assert root.status_code == 200, root.text
assert "<script" in root.text
assert root.headers["cache-control"] == "no-cache"
assert deep_link.status_code == 200, deep_link.text
assert "<script" in deep_link.text
assert deep_link.headers["cache-control"] == "no-cache"
assert manifest.status_code == 200, manifest.text
assert manifest.headers["content-type"].startswith("application/manifest+json")
assert manifest.headers["cache-control"] == "public, max-age=86400"
assert asset.status_code == 200, asset.text
assert asset.headers["content-encoding"] == "gzip"
assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
asset_type = asset.headers["content-type"]
assert asset_type.startswith(("application/javascript", "text/javascript"))
assert asset.text.strip()
"""
    env.pop("PYTHONPATH", None)
    env["ASSET_ROUTE"] = asset_route
    env["REPO_ROOT"] = str(repo)
    env["WHEEL_TARGET"] = str(target)
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        cwd=tmp_path,
        env=env,
    )
