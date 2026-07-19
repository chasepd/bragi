from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

from pytest import CaptureFixture, MonkeyPatch, raises

import bragi_web.bragi_adapter as adapter
from bragi import app as bragi_app


def test_cli_help_and_web_modules_import_without_loading_bragi() -> None:
    repo = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "bragi" or name.startswith("bragi."):
                raise AssertionError(f"unexpected bragi import: {name}")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = blocked_import

        import bragi_web.api.app
        import bragi_web.main

        try:
            bragi_web.main.main(["--help"])
        except SystemExit as exc:
            if exc.code != 0:
                raise

        try:
            bragi_web.main.main(["--version"])
        except SystemExit as exc:
            raise SystemExit(exc.code)
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: bragi-web" in result.stdout


def test_primary_bragi_cli_uses_bragi_program_name(
    capsys: CaptureFixture[str],
) -> None:
    with raises(SystemExit) as excinfo:
        bragi_app.cli_main(["--help"])

    assert excinfo.value.code == 0
    assert "usage: bragi" in capsys.readouterr().out


def test_missing_required_bragi_symbol_reports_compatibility_error(
    monkeypatch: MonkeyPatch,
) -> None:
    real_import_module = adapter.import_module

    def fake_import_module(module_name: str) -> object:
        if module_name == "bragi.application.controller":
            return ModuleType("bragi.application.controller")
        return real_import_module(module_name)

    monkeypatch.setattr(adapter, "import_module", fake_import_module)
    adapter.bragi_api_bindings.cache_clear()

    try:
        with raises(adapter.BragiCompatibilityError) as excinfo:
            adapter.bragi_api_bindings()
    finally:
        adapter.bragi_api_bindings.cache_clear()

    assert "ManualScenarioInput" in str(excinfo.value)
    assert "uv sync" in str(excinfo.value)
