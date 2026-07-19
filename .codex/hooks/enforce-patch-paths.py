#!/usr/bin/env python3
"""Reject apply_patch paths outside the repository.

This keeps Codex patches using repository-relative paths without enforcing any
test-edit routing.
"""

from __future__ import annotations

import json
import re
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


def _safe_relative_patch_path(path: Path) -> Path | None:
    if path.is_absolute() or ".." in path.parts:
        return None

    repo_root = Path.cwd().resolve()
    resolved = (repo_root / path).resolve()
    try:
        return resolved.relative_to(repo_root)
    except ValueError:
        return None


def _unsafe_path_reason(paths: list[Path]) -> str:
    formatted_paths = ", ".join(str(path) for path in paths)
    return (
        "Patch paths must be relative to the repository and must not contain "
        f"`..` segments. Unsafe path(s): {formatted_paths}"
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

    unsafe_paths = [
        path
        for path in _patch_paths(_command_from_payload(payload))
        if _safe_relative_patch_path(path) is None
    ]
    if unsafe_paths:
        _emit_block(_unsafe_path_reason(unsafe_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
