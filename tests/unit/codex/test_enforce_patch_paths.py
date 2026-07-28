from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_HOOK_PATH = Path(__file__).parents[3] / ".codex" / "hooks" / "enforce-patch-paths.py"
_SPEC = importlib.util.spec_from_file_location("enforce_patch_paths", _HOOK_PATH)
assert _SPEC is not None
hook = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(hook)


def _payload(
    command: str,
    *,
    cwd: Path | None = None,
    tool_input_key: str = "command",
) -> str:
    payload: dict[str, object] = {"tool_input": {tool_input_key: command}}
    if cwd is not None:
        payload["cwd"] = str(cwd)
    return json.dumps(payload)


def _assert_block_reason(output: dict[str, object]) -> str:
    assert output["decision"] == "block"
    reason = output["reason"]
    assert isinstance(reason, str)
    hook_output = output["hookSpecificOutput"]
    assert isinstance(hook_output, dict)
    assert hook_output["hookEventName"] == "PreToolUse"
    assert hook_output["additionalContext"] == reason
    return reason


def test_allows_parent_relative_path_into_registered_sibling_worktree(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    sibling = tmp_path / "task"
    primary.mkdir()
    sibling.mkdir()
    monkeypatch.chdir(primary)
    monkeypatch.setattr(
        hook,
        "_registered_worktree_roots",
        lambda _repo: (primary.resolve(), sibling.resolve()),
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(
            _payload(
                _patch_command_for_path("../task/bragi/app.py"),
                cwd=primary,
            )
        ),
    )

    assert hook.main() == 0

    assert capsys.readouterr().out == ""


def test_allows_absolute_path_into_registered_sibling_worktree(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    sibling = tmp_path / "task"
    primary.mkdir()
    sibling.mkdir()
    monkeypatch.setattr(
        hook,
        "_registered_worktree_roots",
        lambda _repo: (primary.resolve(), sibling.resolve()),
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(
            _payload(
                _patch_command_for_path(str(sibling / "bragi" / "app.py")),
                cwd=primary,
            )
        ),
    )

    assert hook.main() == 0

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("patch_kind", ["absolute", "parent-relative"])
def test_blocks_path_outside_registered_worktrees(
    patch_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    sibling = tmp_path / "task"
    outside = tmp_path / "outside"
    primary.mkdir()
    sibling.mkdir()
    outside.mkdir()
    path = (
        str(outside / "app.py")
        if patch_kind == "absolute"
        else "../outside/app.py"
    )
    monkeypatch.setattr(
        hook,
        "_registered_worktree_roots",
        lambda _repo: (primary.resolve(), sibling.resolve()),
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(_payload(_patch_command_for_path(path), cwd=primary)),
    )

    assert hook.main() == 0

    output = json.loads(capsys.readouterr().out)
    reason = _assert_block_reason(output)
    assert "registered worktree" in reason


def test_blocks_symlink_escape_from_registered_worktree(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    outside = tmp_path / "outside"
    primary.mkdir()
    outside.mkdir()
    (primary / "linked").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        hook,
        "_registered_worktree_roots",
        lambda _repo: (primary.resolve(),),
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(
            _payload(
                _patch_command_for_path("linked/app.py"),
                cwd=primary,
            )
        ),
    )

    assert hook.main() == 0

    output = json.loads(capsys.readouterr().out)
    reason = _assert_block_reason(output)
    assert "registered worktree" in reason


def test_registered_worktree_roots_come_from_git_porcelain_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    sibling = tmp_path / "task"
    stale = tmp_path / "stale"
    primary.mkdir()
    sibling.mkdir()
    stale.mkdir()
    command: list[str] = []

    def fake_run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        command.extend(args)
        output = (
            f"worktree {primary}\0HEAD abc\0branch refs/heads/main\0\0"
            f"worktree {sibling}\0HEAD def\0branch refs/heads/task\0\0"
            f"worktree {stale}\0HEAD 123\0detached\0"
            "prunable gitdir file points to non-existent location\0\0"
        )
        return subprocess.CompletedProcess(args, 0, stdout=output)

    monkeypatch.setattr(hook.subprocess, "run", fake_run)

    roots = hook._registered_worktree_roots(primary)

    assert command == [
        "git",
        "-C",
        str(primary),
        "worktree",
        "list",
        "--porcelain",
        "-z",
    ]
    assert roots == (primary.resolve(), sibling.resolve())


def test_allows_repo_relative_test_patch_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        hook,
        "_registered_worktree_roots",
        lambda _repo: (repo.resolve(),),
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(
            _payload(
                _patch_command_for_path("tests/unit/test_example.py"),
                cwd=repo,
            )
        ),
    )

    assert hook.main() == 0

    assert capsys.readouterr().out == ""


def test_allows_repo_relative_source_patch_path_with_cmd_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        hook,
        "_registered_worktree_roots",
        lambda _repo: (repo.resolve(),),
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(
            _payload(
                _patch_command_for_path("bragi/app.py"),
                cwd=repo,
                tool_input_key="cmd",
            )
        ),
    )

    assert hook.main() == 0

    assert capsys.readouterr().out == ""


def _patch_command_for_path(path: str) -> str:
    return "\n".join(
        [
            "*** Begin Patch",
            f"*** Add File: {path}",
            "+def test_example() -> None:",
            "+    assert True",
            "*** End Patch",
        ]
    )


class _TextInput:
    def __init__(self, value: str) -> None:
        self._value = value

    def read(self) -> str:
        return self._value
