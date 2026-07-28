#!/usr/bin/env python3
"""Run deterministic pytest targets for files edited by Codex."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import validation_targets  # noqa: E402

_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$")
_PATCH_MOVE_RE = re.compile(r"^\*\*\* Move to: (.+)$")
_EDIT_TEST_MODE_ENV = "BRAGI_EDIT_TEST_MODE"
_EDIT_TEST_MODE_RUN = "run"
_DEFAULT_UV_CACHE_DIR = "/tmp/bragi-uv-cache"
_PYTEST_WORKERS_ENV = "BRAGI_PYTEST_WORKERS"
_PYTEST_WORKERS_DISABLED_VALUES = {"", "0", "1", "false", "no", "off"}
_TEST_ENV_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
    "UV_CACHE_DIR",
    "BRAGI_PYTEST_WORKERS",
    "WINDIR",
)
_WEB_TEST_TARGETS = validation_targets.WEB_TEST_TARGETS


def _emit_done() -> None:
    sys.stdout.write("{}\n")
    sys.stdout.flush()


def _add_candidate(candidates: list[str], value: object) -> None:
    if isinstance(value, str) and value.strip():
        candidates.append(value)


def _extract_file_candidates(payload: dict[str, object]) -> list[str]:
    candidates: list[str] = []

    _add_candidate(candidates, payload.get("file_path"))
    _add_candidate(candidates, payload.get("filePath"))
    _add_candidate(candidates, payload.get("path"))

    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        _add_candidate(candidates, tool_input.get("file_path"))
        _add_candidate(candidates, tool_input.get("filePath"))
        _add_candidate(candidates, tool_input.get("path"))
        command = tool_input.get("command")
        if isinstance(command, str):
            for line in command.splitlines():
                stripped = line.strip()
                for pattern in (_PATCH_FILE_RE, _PATCH_MOVE_RE):
                    match = pattern.match(stripped)
                    if match:
                        candidates.append(match.group(1).strip())
                        break

    if len(sys.argv) > 1:
        for raw in sys.argv[1:]:
            _add_candidate(candidates, raw)

    seen: set[str] = set()
    ordered: list[str] = []
    for raw in candidates:
        if raw in seen:
            continue
        seen.add(raw)
        ordered.append(raw)
    return ordered


def _test_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _TEST_ENV_ALLOWLIST
    }


def _cwd_path(cwd: str | None) -> Path:
    return Path(cwd).expanduser().resolve() if cwd else Path.cwd().resolve()


def _repo_root_from_cwd(cwd: str | None) -> Path | None:
    start = _cwd_path(cwd)

    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "bragi"
        ).is_dir():
            return directory
    return None


def _repo_root_containing(path: Path) -> Path | None:
    for directory in (path, *path.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "bragi"
        ).is_dir():
            return directory
    return None


def _queue_path(repo: Path | None = None) -> Path:
    return validation_targets.queue_path(repo)


def _read_queue(repo: Path | None = None) -> list[Path]:
    return validation_targets.read_queue(repo)


def _write_queue(test_targets: list[Path], repo: Path | None = None) -> None:
    try:
        validation_targets.write_queue(test_targets, repo)
    except OSError as exc:
        print(f"[bragi-tests-hook] could not write test queue: {exc}", file=sys.stderr)


def _target_text(repo: Path, target: Path) -> str:
    try:
        return str(target.relative_to(repo))
    except ValueError:
        return str(target)


def _queue_tests(repo: Path, test_targets: list[Path]) -> None:
    if not test_targets:
        return

    try:
        validation_targets.append_queue_targets(repo, test_targets)
    except OSError as exc:
        print(f"[bragi-tests-hook] could not write test queue: {exc}", file=sys.stderr)
        return

    queued = ", ".join(_target_text(repo, target) for target in test_targets)
    print(f"[bragi-tests-hook] queued mapped test targets: {queued}", file=sys.stderr)


def _flush_queued_tests(repo: Path) -> bool:
    targets = validation_targets.queued_targets_for_repo(repo)

    if not targets:
        print("[bragi-tests-hook] no queued test targets", file=sys.stderr)
        return True

    passed = True
    for target in targets:
        if _run_pytest(repo, target):
            validation_targets.remove_queue_targets(repo, [target])
        else:
            passed = False
    return passed


def _resolve_candidate(raw: str, cwd: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (cwd / path).resolve()


def _run_pytest(repo: Path, test_path: Path) -> bool:
    if not test_path.exists():
        print(
            f"[bragi-tests-hook] no test target yet: {test_path}",
            file=sys.stderr,
        )
        return True

    if shutil.which("uv") is not None:
        cache_dir = os.environ.get("UV_CACHE_DIR", _DEFAULT_UV_CACHE_DIR)
        cmd = [
            "uv",
            "--cache-dir",
            cache_dir,
            "run",
            "--extra",
            "dev",
            "pytest",
            str(test_path),
            *_pytest_parallel_args(),
        ]
    elif shutil.which("pytest") is not None:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            *_pytest_parallel_args(),
        ]
    else:
        print("[bragi-tests-hook] skip: pytest not available", file=sys.stderr)
        return True

    print(f"[bragi-tests-hook] {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(
        cmd,
        cwd=repo,
        env=_test_env(),
        stdout=sys.stderr,
        stderr=sys.stderr,
        check=False,
    )
    return result.returncode == 0


def _pytest_parallel_args() -> list[str]:
    workers = os.environ.get(_PYTEST_WORKERS_ENV, "auto").strip()
    if workers.lower() in _PYTEST_WORKERS_DISABLED_VALUES:
        return []
    return ["-n", workers]


def _unit_test_for_source(repo: Path, rel: Path) -> Path | None:
    return validation_targets.unit_test_for_source(repo, rel)


def _unit_subsystem_test_dir_for_source(repo: Path, rel: Path) -> Path | None:
    return validation_targets.unit_subsystem_test_dir_for_source(repo, rel)


def _web_test_for_source(repo: Path, rel: Path) -> Path | None:
    return validation_targets.web_test_for_source(repo, rel)


def _is_web_source(rel: Path) -> bool:
    return (
        rel.suffix == ".py"
        and rel.parts[0] == "bragi_web"
        and rel.name != "__init__.py"
    )


def _test_target_for_path(repo: Path, path: Path, rel: Path) -> Path | None:
    if not rel.parts:
        return None

    web_target = _web_test_for_source(repo, rel)
    target = validation_targets.test_target_for_path(repo, path, rel)
    if web_target is not None and not web_target.exists():
        print(
            f"[bragi-tests-hook] no web test target yet: "
            f"{_target_text(repo, web_target)}",
            file=sys.stderr,
        )
        return None
    if _is_web_source(rel) and target is None:
        print(
            f"[bragi-tests-hook] no web test target for {_target_text(repo, rel)}",
            file=sys.stderr,
        )
        return None

    direct_target = _unit_test_for_source(repo, rel)
    if (
        direct_target is not None
        and not direct_target.exists()
        and target is not None
        and target.is_dir()
    ):
        print(
            f"[bragi-tests-hook] no mirrored test target yet: "
            f"{_target_text(repo, direct_target)}; "
            f"falling back to {_target_text(repo, target)}",
            file=sys.stderr,
        )
        return target

    if direct_target is not None and target is None:
        print(
            f"[bragi-tests-hook] no mirrored/subsystem test target for "
            f"{_target_text(repo, rel)}",
            file=sys.stderr,
        )
    return target


def _payload_from_stdin() -> dict[str, object]:
    if sys.stdin.isatty():
        return {}

    try:
        raw_stdin = sys.stdin.read().strip()
        if not raw_stdin:
            return {}
        parsed = json.loads(raw_stdin)
    except json.JSONDecodeError as exc:
        print(f"[bragi-tests-hook] invalid stdin JSON: {exc}", file=sys.stderr)
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _run_or_queue_targets(repo: Path, test_targets: list[Path]) -> None:
    if os.environ.get(_EDIT_TEST_MODE_ENV) == _EDIT_TEST_MODE_RUN:
        for test_target in test_targets:
            _run_pytest(repo, test_target)
        return

    _queue_tests(repo, test_targets)


def main() -> int:
    payload = _payload_from_stdin()
    raw_cwd = str(payload.get("cwd")) if payload.get("cwd") else None
    cwd = _cwd_path(raw_cwd)
    repo_from_cwd = _repo_root_from_cwd(raw_cwd)

    if "--flush" in sys.argv[1:]:
        repo = repo_from_cwd or _repo_root_from_cwd(None)
        passed = True
        if repo is not None:
            passed = _flush_queued_tests(repo)
        _emit_done()
        return 0 if passed else 1

    files = _extract_file_candidates(payload)
    if not files:
        _emit_done()
        return 0

    targets_by_repo: dict[Path, list[Path]] = {}
    seen_by_repo: dict[Path, set[str]] = {}
    for raw in files:
        path = _resolve_candidate(raw, cwd)
        repo = _repo_root_containing(path) or repo_from_cwd
        if repo is None:
            continue

        try:
            rel = path.relative_to(repo)
        except ValueError:
            continue
        if not rel.parts:
            continue

        test_target = _test_target_for_path(repo, path, rel)
        if test_target is not None:
            target_text = _target_text(repo, test_target)
            seen = seen_by_repo.setdefault(repo, set())
            if target_text in seen:
                continue
            seen.add(target_text)
            targets_by_repo.setdefault(repo, []).append(test_target)

    for repo, test_targets in targets_by_repo.items():
        _run_or_queue_targets(repo, test_targets)

    _emit_done()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
