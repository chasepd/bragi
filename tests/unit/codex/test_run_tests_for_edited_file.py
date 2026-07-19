from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

_HOOK_PATH = (
    Path(__file__).parents[3] / ".codex" / "hooks" / "run_tests_for_edited_file.py"
)
_SPEC = importlib.util.spec_from_file_location("run_tests_for_edited_file", _HOOK_PATH)
assert _SPEC is not None
hook = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(hook)


def _repo_with_mapped_test(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'bragi'\n",
        encoding="utf-8",
    )
    (repo / "bragi" / "providers").mkdir(parents=True)
    (repo / "bragi" / "providers" / "contracts.py").write_text("", encoding="utf-8")
    test_target = repo / "tests" / "unit" / "providers" / "test_contracts.py"
    test_target.parent.mkdir(parents=True)
    test_target.write_text("def test_contracts():\n    assert True\n", encoding="utf-8")
    return repo, test_target


def _repo_with_web_source_and_test(
    tmp_path: Path,
    source: str,
    test: str,
) -> tuple[Path, Path]:
    repo = tmp_path
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'bragi'\n",
        encoding="utf-8",
    )
    (repo / "bragi").mkdir()
    source_path = repo / source
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("", encoding="utf-8")
    test_target = repo / test
    test_target.parent.mkdir(parents=True, exist_ok=True)
    test_target.write_text(
        "def test_web_target():\n    assert True\n",
        encoding="utf-8",
    )
    return repo, test_target


def _invoke_hook(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    argv: list[str] | None = None,
    expected_status: int = 0,
) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        argv if argv is not None else ["run_tests_for_edited_file.py"],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    status = cast(int, hook.main())
    assert status == expected_status
    return status


def _read_queued_targets(queue_path: Path) -> list[str]:
    if not queue_path.exists():
        return []
    return [
        line.strip()
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_extract_file_candidates_includes_patch_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_tests_for_edited_file.py"])
    payload = {
        "tool_input": {
            "command": "\n".join(
                [
                    "*** Begin Patch",
                    "*** Update File: bragi/providers/old.py",
                    "*** Move to: bragi/providers/new.py",
                    "@@",
                    " pass",
                    "*** End Patch",
                ]
            )
        }
    }

    assert hook._extract_file_candidates(payload) == [
        "bragi/providers/old.py",
        "bragi/providers/new.py",
    ]


def test_test_env_excludes_secrets_validation_runner_and_preserves_uv_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/example",
            "OPENROUTER_API_KEY": "secret",
            "VENICE_API_KEY": "secret",
            "DATABASE_URL": "sqlite:///secret",
            "PYTHONPATH": "/tmp/injected",
            "BRAGI_VALIDATION_RUNNER": "1",
            "UV_CACHE_DIR": "/tmp/bragi-uv-cache",
        },
    )

    env = hook._test_env()

    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/home/example",
        "UV_CACHE_DIR": "/tmp/bragi-uv-cache",
    }


def test_unit_test_for_source_maps_bragi_module_to_unit_test(tmp_path: Path) -> None:
    target = hook._unit_test_for_source(
        tmp_path,
        Path("bragi/providers/contracts.py"),
    )

    assert target == tmp_path / "tests" / "unit" / "providers" / "test_contracts.py"


@pytest.mark.parametrize(
    ("source", "test_target"),
    [
        ("bragi_web/api/app.py", "tests/web/test_api.py"),
        ("bragi_web/auth_throttle.py", "tests/web/test_api.py"),
        ("bragi_web/bragi_adapter.py", "tests/web/test_bragi_adapter.py"),
        ("bragi_web/jobs.py", "tests/web/test_jobs.py"),
        ("bragi_web/main.py", "tests/web/test_cli.py"),
        (
            "bragi_web/maintenance_diagnostics.py",
            "tests/web/test_maintenance_diagnostics.py",
        ),
        ("bragi_web/observability.py", "tests/web/test_observability.py"),
        ("bragi_web/runtime.py", "tests/web/test_api.py"),
        ("bragi_web/serialization.py", "tests/web/test_api.py"),
        ("bragi_web/storage.py", "tests/web/test_storage.py"),
    ],
)
def test_web_test_for_source_maps_explicit_web_modules(
    source: str,
    test_target: str,
    tmp_path: Path,
) -> None:
    target = hook._web_test_for_source(
        tmp_path,
        Path(source),
    )

    assert target == tmp_path / test_target


def test_run_pytest_uses_sanitized_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_path = tmp_path / "tests" / "unit" / "test_app.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(hook.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/test-uv-cache")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        assert isinstance(env, dict)
        captured["env"] = env
        command = cast(list[str], args[0])
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(hook.subprocess, "run", fake_run)

    hook._run_pytest(tmp_path, test_path)

    env = cast(dict[str, str], captured["env"])
    assert env["PATH"] == "/usr/bin"
    assert "OPENROUTER_API_KEY" not in env
    assert captured["command"] == [
        "uv",
        "--cache-dir",
        "/tmp/test-uv-cache",
        "run",
        "--extra",
        "dev",
        "pytest",
        str(test_path),
        "-n",
        "auto",
    ]


def test_default_mode_queues_subsystem_dir_when_mirrored_test_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'bragi'\n",
        encoding="utf-8",
    )
    source_path = repo / "bragi" / "services" / "missing_tested_module.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("", encoding="utf-8")
    subsystem_dir = repo / "tests" / "unit" / "services"
    subsystem_dir.mkdir(parents=True)
    queue_path = tmp_path / "edit-test-queue.txt"
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))
    monkeypatch.delenv("BRAGI_EDIT_TEST_MODE", raising=False)
    monkeypatch.setattr(
        hook,
        "_run_pytest",
        lambda repo, target: calls.append((repo, target)),
    )

    _invoke_hook(
        monkeypatch,
        {
            "cwd": str(repo),
            "file_path": "bragi/services/missing_tested_module.py",
        },
    )

    assert calls == []
    assert _read_queued_targets(queue_path) == [str(subsystem_dir)]


def test_default_mode_queues_nothing_when_mirrored_and_subsystem_tests_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'bragi'\n",
        encoding="utf-8",
    )
    source_path = repo / "bragi" / "new_subsystem" / "module.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("", encoding="utf-8")
    queue_path = tmp_path / "edit-test-queue.txt"

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))
    monkeypatch.delenv("BRAGI_EDIT_TEST_MODE", raising=False)
    monkeypatch.setattr(
        hook,
        "_run_pytest",
        lambda repo, target: pytest.fail("_run_pytest should not run in queue mode"),
    )

    _invoke_hook(
        monkeypatch,
        {
            "cwd": str(repo),
            "file_path": "bragi/new_subsystem/module.py",
        },
    )

    assert _read_queued_targets(queue_path) == []


def test_default_mode_queues_mapped_test_target_instead_of_running_pytest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, test_target = _repo_with_mapped_test(tmp_path)
    queue_path = tmp_path / "edit-test-queue.txt"
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))
    monkeypatch.delenv("BRAGI_EDIT_TEST_MODE", raising=False)
    monkeypatch.setattr(
        hook,
        "_run_pytest",
        lambda repo, target: calls.append((repo, target)),
    )

    _invoke_hook(
        monkeypatch,
        {
            "cwd": str(repo),
            "file_path": "bragi/providers/contracts.py",
        },
    )

    assert calls == []
    assert _read_queued_targets(queue_path) == [str(test_target)]


def test_default_mode_queues_explicit_web_test_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, test_target = _repo_with_web_source_and_test(
        tmp_path,
        "bragi_web/main.py",
        "tests/web/test_cli.py",
    )
    queue_path = tmp_path / "edit-test-queue.txt"
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))
    monkeypatch.delenv("BRAGI_EDIT_TEST_MODE", raising=False)
    monkeypatch.setattr(
        hook,
        "_run_pytest",
        lambda repo, target: calls.append((repo, target)),
    )

    _invoke_hook(
        monkeypatch,
        {
            "cwd": str(repo),
            "file_path": "bragi_web/main.py",
        },
    )

    assert calls == []
    assert _read_queued_targets(queue_path) == [str(test_target)]


def test_default_mode_queues_nested_web_api_test_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, test_target = _repo_with_web_source_and_test(
        tmp_path,
        "bragi_web/api/app.py",
        "tests/web/test_api.py",
    )
    queue_path = tmp_path / "edit-test-queue.txt"
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))
    monkeypatch.delenv("BRAGI_EDIT_TEST_MODE", raising=False)
    monkeypatch.setattr(
        hook,
        "_run_pytest",
        lambda repo, target: calls.append((repo, target)),
    )

    _invoke_hook(
        monkeypatch,
        {
            "cwd": str(repo),
            "file_path": "bragi_web/api/app.py",
        },
    )

    assert calls == []
    assert _read_queued_targets(queue_path) == [str(test_target)]


def test_default_mode_queues_existing_generic_web_test_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, test_target = _repo_with_web_source_and_test(
        tmp_path,
        "bragi_web/example.py",
        "tests/web/test_example.py",
    )
    queue_path = tmp_path / "edit-test-queue.txt"
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))
    monkeypatch.delenv("BRAGI_EDIT_TEST_MODE", raising=False)
    monkeypatch.setattr(
        hook,
        "_run_pytest",
        lambda repo, target: calls.append((repo, target)),
    )

    _invoke_hook(
        monkeypatch,
        {
            "cwd": str(repo),
            "file_path": "bragi_web/example.py",
        },
    )

    assert calls == []
    assert _read_queued_targets(queue_path) == [str(test_target)]


def test_default_mode_queues_nothing_for_unmapped_web_source_without_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'bragi'\n",
        encoding="utf-8",
    )
    (repo / "bragi").mkdir()
    source_path = repo / "bragi_web" / "example.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("", encoding="utf-8")
    queue_path = tmp_path / "edit-test-queue.txt"

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))
    monkeypatch.delenv("BRAGI_EDIT_TEST_MODE", raising=False)
    monkeypatch.setattr(
        hook,
        "_run_pytest",
        lambda repo, target: pytest.fail("_run_pytest should not run in queue mode"),
    )

    _invoke_hook(
        monkeypatch,
        {
            "cwd": str(repo),
            "file_path": "bragi_web/example.py",
        },
    )

    assert _read_queued_targets(queue_path) == []


def test_default_mode_deduplicates_repeated_queued_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, test_target = _repo_with_mapped_test(tmp_path)
    queue_path = tmp_path / "edit-test-queue.txt"

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))
    monkeypatch.delenv("BRAGI_EDIT_TEST_MODE", raising=False)
    monkeypatch.setattr(
        hook,
        "_run_pytest",
        lambda repo, target: pytest.fail("_run_pytest should not run in queue mode"),
    )

    payload: dict[str, object] = {
        "cwd": str(repo),
        "file_path": "bragi/providers/contracts.py",
    }
    _invoke_hook(monkeypatch, payload)
    _invoke_hook(monkeypatch, payload)

    assert _read_queued_targets(queue_path) == [str(test_target)]


def test_run_mode_runs_pytest_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, test_target = _repo_with_mapped_test(tmp_path)
    queue_path = tmp_path / "edit-test-queue.txt"
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))
    monkeypatch.setenv("BRAGI_EDIT_TEST_MODE", "run")

    def fake_run_pytest(repo: Path, target: Path) -> bool:
        calls.append((repo, target))
        return True

    monkeypatch.setattr(
        hook,
        "_run_pytest",
        fake_run_pytest,
    )

    _invoke_hook(
        monkeypatch,
        {
            "cwd": str(repo),
            "file_path": "bragi/providers/contracts.py",
        },
    )

    assert calls == [(repo, test_target)]
    assert _read_queued_targets(queue_path) == []


def test_flush_runs_queued_targets_once_and_clears_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, test_target = _repo_with_mapped_test(tmp_path)
    queue_path = tmp_path / "edit-test-queue.txt"
    queue_path.write_text(f"{test_target}\n{test_target}\n", encoding="utf-8")
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))

    def fake_run_pytest(repo: Path, target: Path) -> bool:
        calls.append((repo, target))
        return True

    monkeypatch.setattr(
        hook,
        "_run_pytest",
        fake_run_pytest,
    )

    _invoke_hook(
        monkeypatch,
        {"cwd": str(repo)},
        argv=["run_tests_for_edited_file.py", "--flush"],
    )

    assert calls == [(repo, test_target)]
    assert _read_queued_targets(queue_path) == []


def test_flush_returns_nonzero_when_queued_target_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, test_target = _repo_with_mapped_test(tmp_path)
    queue_path = tmp_path / "edit-test-queue.txt"
    queue_path.write_text(f"{test_target}\n", encoding="utf-8")
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))

    def fake_run_pytest(repo: Path, target: Path) -> bool:
        calls.append((repo, target))
        return False

    monkeypatch.setattr(
        hook,
        "_run_pytest",
        fake_run_pytest,
    )

    _invoke_hook(
        monkeypatch,
        {"cwd": str(repo)},
        argv=["run_tests_for_edited_file.py", "--flush"],
        expected_status=1,
    )

    assert calls == [(repo, test_target)]
    assert _read_queued_targets(queue_path) == [str(test_target)]


def test_flush_removes_passed_targets_but_keeps_failed_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, first_target = _repo_with_mapped_test(tmp_path)
    failed_target = repo / "tests" / "unit" / "test_failed.py"
    failed_target.write_text("def test_failed():\n    assert False\n", encoding="utf-8")
    queue_path = tmp_path / "edit-test-queue.txt"
    queue_path.write_text(f"{first_target}\n{failed_target}\n", encoding="utf-8")
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))

    def fake_run_pytest(repo: Path, target: Path) -> bool:
        calls.append((repo, target))
        return target != failed_target

    monkeypatch.setattr(
        hook,
        "_run_pytest",
        fake_run_pytest,
    )

    _invoke_hook(
        monkeypatch,
        {"cwd": str(repo)},
        argv=["run_tests_for_edited_file.py", "--flush"],
        expected_status=1,
    )

    assert calls == [(repo, first_target), (repo, failed_target)]
    assert _read_queued_targets(queue_path) == [str(failed_target)]


def test_flush_interruption_keeps_unfinished_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, first_target = _repo_with_mapped_test(tmp_path)
    second_target = repo / "tests" / "unit" / "test_second.py"
    second_target.write_text("def test_second():\n    assert True\n", encoding="utf-8")
    queue_path = tmp_path / "edit-test-queue.txt"
    queue_path.write_text(f"{first_target}\n{second_target}\n", encoding="utf-8")

    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))

    def interrupted_run_pytest(_repo: Path, _target: Path) -> bool:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(
        hook,
        "_run_pytest",
        interrupted_run_pytest,
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        hook._flush_queued_tests(repo)

    assert _read_queued_targets(queue_path) == [str(first_target), str(second_target)]
