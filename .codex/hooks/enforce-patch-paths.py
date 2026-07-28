#!/usr/bin/env python3
"""Reject apply_patch paths outside this repository's registered worktrees.

This lets Codex edit a dedicated sibling worktree while retaining a boundary
against patches to unrelated filesystem paths.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$")
_PATCH_MOVE_RE = re.compile(r"^\*\*\* Move to: (.+)$")


def _command_from_payload(payload: dict[str, object]) -> str:
    for key in ("cmd", "command"):
        value = payload.get(key)
        if isinstance(value, str):
            return value

    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return ""

    for key in ("cmd", "command"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return ""


def _patch_paths(command: str) -> list[Path]:
    paths: list[Path] = []
    for line in command.splitlines():
        stripped = line.strip()
        for pattern in (_PATCH_FILE_RE, _PATCH_MOVE_RE):
            match = pattern.match(stripped)
            if match:
                paths.append(Path(match.group(1).strip()))
                break
    return paths


def _cwd_from_payload(payload: dict[str, object]) -> Path:
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return Path(cwd).expanduser().resolve()
    return Path.cwd().resolve()


def _registered_worktree_roots(repo_root: Path) -> tuple[Path, ...]:
    repo_root = repo_root.resolve()
    roots = {repo_root}
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "worktree",
                "list",
                "--porcelain",
                "-z",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return (repo_root,)

    if result.returncode != 0:
        return (repo_root,)

    for record in result.stdout.split("\0\0"):
        fields = record.split("\0")
        if any(field.startswith("prunable") for field in fields):
            continue
        root_field = next(
            (field for field in fields if field.startswith("worktree ")),
            None,
        )
        if root_field is None:
            continue
        roots.add(
            Path(root_field.removeprefix("worktree ")).expanduser().resolve()
        )
    return tuple(sorted(roots, key=str))


def _safe_patch_path(
    path: Path,
    cwd: Path,
    worktree_roots: tuple[Path, ...],
) -> Path | None:
    expanded = path.expanduser()
    resolved = (
        expanded.resolve()
        if expanded.is_absolute()
        else (cwd / expanded).resolve()
    )

    for root in worktree_roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    return None


def _unsafe_path_reason(paths: list[Path]) -> str:
    formatted_paths = ", ".join(str(path) for path in paths)
    return (
        "Patch paths must resolve inside a registered worktree for this "
        f"repository. Unsafe path(s): {formatted_paths}"
    )


def _emit_block(reason: str) -> None:
    print(
        json.dumps(
            {
                "decision": "block",
                "reason": reason,
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": reason,
                },
            }
        )
    )


def main() -> int:
    raw_stdin = sys.stdin.read().strip()
    if not raw_stdin:
        return 0

    try:
        payload = json.loads(raw_stdin)
    except json.JSONDecodeError:
        return 0

    if not isinstance(payload, dict):
        return 0

    cwd = _cwd_from_payload(payload)
    repo_root = Path(__file__).resolve().parents[2]
    worktree_roots = _registered_worktree_roots(repo_root)
    unsafe_paths = [
        path
        for path in _patch_paths(_command_from_payload(payload))
        if _safe_patch_path(path, cwd, worktree_roots) is None
    ]
    if unsafe_paths:
        _emit_block(_unsafe_path_reason(unsafe_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
