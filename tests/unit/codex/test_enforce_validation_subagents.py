from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HOOK_PATH = (
    Path(__file__).parents[3] / ".codex" / "hooks" / "enforce-validation-subagents.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "enforce_validation_subagents",
    _HOOK_PATH,
)
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


@pytest.mark.parametrize(
    "command",
    [
        "uv --cache-dir /tmp/bragi-uv-cache run --extra dev pytest tests/unit -q",
        "python -m pytest tests/unit -q",
        "python3.12 -m pytest tests/unit -q",
        "uv --cache-dir /tmp/bragi-uv-cache run --extra dev ruff check .",
        "python -m ruff check .",
        "python3.12 -m ruff check .",
        "uv --cache-dir /tmp/bragi-uv-cache run --extra dev mypy",
        "python -m mypy",
        "python3.12 -m mypy",
        "pytest tests/unit -q # BRAGI_VALIDATION_RUNNER=1",
        "python -m pytest BRAGI_VALIDATION_RUNNER=1",
        "true;pytest tests/unit -q",
        "true&&ruff check .",
        "false||python -m mypy",
        "printf ok|pytest tests/unit -q",
        "env pytest tests/unit -q",
        "env -u FOO pytest tests/unit -q",
        "/usr/bin/env ruff check .",
        "/usr/bin/env -i ruff check .",
        "command mypy",
        "time python -m pytest tests/unit -q",
        "time -p python -m mypy",
        "true\npytest tests/unit -q",
        "printf ok\nruff check .",
        "python -m json.tool .codex/hooks.json\npython -m mypy",
        "bash -lc 'pytest tests/unit -q'",
        "bash -lc \"ruff check .\"",
        "sh -c 'python -m mypy'",
        "zsh -lc 'uv run pytest tests/unit -q'",
    ],
)
def test_blocks_direct_validation_commands(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", _TextInput(_payload(command)))

    assert hook.main() == 0

    output = json.loads(capsys.readouterr().out)
    reason = _assert_block_reason(output)
    assert "python3 .codex/tools/validate.py" in reason


def test_blocks_codex_tool_input_cmd_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "stdin",
        _TextInput(_payload("python -m pytest tests/unit -q", tool_input_key="cmd")),
    )

    assert hook.main() == 0

    output = json.loads(capsys.readouterr().out)
    reason = _assert_block_reason(output)
    assert "python3 .codex/tools/validate.py" in reason


@pytest.mark.parametrize(
    "command",
    [
        (
            "BRAGI_VALIDATION_RUNNER=1 uv --cache-dir /tmp/bragi-uv-cache "
            "run --extra dev pytest tests/unit -q"
        ),
        (
            "BRAGI_VALIDATION_RUNNER=1 uv --cache-dir /tmp/bragi-uv-cache "
            "run --extra dev ruff check ."
        ),
        (
            "BRAGI_VALIDATION_RUNNER=1 uv --cache-dir /tmp/bragi-uv-cache "
            "run --extra dev mypy"
        ),
        "BRAGI_VALIDATION_RUNNER=1 env pytest tests/unit -q",
        "env -S 'BRAGI_VALIDATION_RUNNER=1 python3.12 -m pytest tests/unit -q'",
        "BRAGI_VALIDATION_RUNNER=1 /usr/bin/env ruff check .",
        "BRAGI_VALIDATION_RUNNER=1 command mypy",
        "BRAGI_VALIDATION_RUNNER=1 time python -m pytest tests/unit -q",
    ],
)
def test_blocks_old_runner_marked_direct_validation_commands(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", _TextInput(_payload(command)))

    assert hook.main() == 0

    output = json.loads(capsys.readouterr().out)
    reason = _assert_block_reason(output)
    assert "python3 .codex/tools/validate.py" in reason


@pytest.mark.parametrize(
    "command",
    [
        "python3 .codex/tools/validate.py",
        "python3 .codex/tools/validate.py --tests-only",
        "python3 .codex/tools/validate.py --typecheck-only",
        "python3 .codex/tools/validate.py --lint-only",
        "uv run python3 .codex/tools/validate.py --tests-only",
        "python -m json.tool .codex/hooks.json",
    ],
)
def test_allows_validation_script_and_non_validation_commands(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", _TextInput(_payload(command)))

    assert hook.main() == 0

    assert capsys.readouterr().out == ""


def test_blocks_later_unmarked_validation_segment_after_marked_segment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = (
        "BRAGI_VALIDATION_RUNNER=1 python -m pytest tests/unit -q;"
        "ruff check ."
    )
    monkeypatch.setattr(sys, "stdin", _TextInput(_payload(command)))

    assert hook.main() == 0

    output = json.loads(capsys.readouterr().out)
    reason = _assert_block_reason(output)
    assert "python3 .codex/tools/validate.py" in reason


class _TextInput:
    def __init__(self, value: str) -> None:
        self._value = value

    def read(self) -> str:
        return self._value
