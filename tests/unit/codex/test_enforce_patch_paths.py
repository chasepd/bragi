from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HOOK_PATH = Path(__file__).parents[3] / ".codex" / "hooks" / "enforce-patch-paths.py"
_SPEC = importlib.util.spec_from_file_location("enforce_patch_paths", _HOOK_PATH)
assert _SPEC is not None
hook = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(hook)


def _payload(command: str, *, tool_input_key: str = "command") -> str:
    return json.dumps({"tool_input": {tool_input_key: command}})


def _assert_block_reason(output: dict[str, object]) -> str:
    assert output["decision"] == "block"
    reason = output["reason"]
    assert isinstance(reason, str)
    hook_output = output["hookSpecificOutput"]
    assert isinstance(hook_output, dict)
    assert hook_output["hookEventName"] == "PreToolUse"
    assert hook_output["additionalContext"] == reason
    return reason


def test_blocks_parent_segment_patch_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(_payload(_patch_command_for_path("tests/../bragi/app.py"))),
    )

    assert hook.main() == 0

    output = json.loads(capsys.readouterr().out)
    reason = _assert_block_reason(output)
    assert "must not contain `..` segments" in reason


def test_blocks_absolute_patch_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(_payload(_patch_command_for_path("/tmp/bragi/app.py"))),
    )

    assert hook.main() == 0

    output = json.loads(capsys.readouterr().out)
    reason = _assert_block_reason(output)
    assert "relative to the repository" in reason


def test_allows_repo_relative_test_patch_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(_payload(_patch_command_for_path("tests/unit/test_example.py"))),
    )

    assert hook.main() == 0

    assert capsys.readouterr().out == ""


def test_allows_repo_relative_source_patch_path_with_cmd_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(
            _payload(_patch_command_for_path("bragi/app.py"), tool_input_key="cmd")
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
