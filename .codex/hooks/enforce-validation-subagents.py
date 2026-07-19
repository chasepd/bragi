#!/usr/bin/env python3
"""Block direct validation commands so Bragi's validation script handles them.

This hook is a best-effort guardrail for directing the model to the correct
validation flow most of the time. It is not a security boundary, and it is not
intended to be impossible for a model or user to bypass.
"""

from __future__ import annotations

import json
import re
import shlex
import sys

_VALIDATION_TOOLS = {"pytest", "ruff", "mypy"}
_ENV_WRAPPERS = {"env"}
_COMMAND_WRAPPERS = {"command", "time"}
_SHELL_WRAPPERS = {"bash", "dash", "sh", "zsh"}
_CHAIN_OPERATORS = {"&&", "||", ";", "|"}
_PYTHON_MODULE_FLAGS = {"-m"}
_ENV_OPTIONS_WITH_OPERAND = {
    "-C",
    "-u",
    "--chdir",
    "--unset",
}
_ENV_OPTIONS_WITHOUT_OPERAND = {
    "-0",
    "-i",
    "--debug",
    "--ignore-environment",
    "--null",
}
_COMMAND_OPTIONS_WITHOUT_OPERAND = {"-p", "-v", "-V"}
_TIME_OPTIONS_WITH_OPERAND = {"-f", "-o", "--format", "--output"}
_TIME_OPTIONS_WITHOUT_OPERAND = {
    "-a",
    "-p",
    "-q",
    "-v",
    "--append",
    "--portability",
    "--quiet",
    "--verbose",
}
_SHELL_OPTIONS_WITH_OPERAND = {
    "-D",
    "-O",
    "-o",
    "--init-file",
    "--rcfile",
}


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


def _tokens_for_command(command: str) -> list[str]:
    try:
        normalized_command = (
            command.replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\n", ";")
        )
        lexer = shlex.shlex(normalized_command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return command.split()


def _is_env_assignment(token: str) -> bool:
    name, separator, value = token.partition("=")
    return bool(
        separator
        and name
        and value
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
    )


def _strip_env_assignments(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    env_assignments: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if not _is_env_assignment(token):
            return tokens[index:], env_assignments
        name, _, value = token.partition("=")
        env_assignments[name] = value
    return [], env_assignments


def _validation_tool_for_segment(segment: list[str]) -> str | None:
    segment, _ = _strip_env_assignments(segment)
    return _validation_tool_for_prepared_segment(segment)


def _validation_tool_for_prepared_segment(segment: list[str]) -> str | None:
    if not segment:
        return None

    executable = segment[0].split("/")[-1]
    if executable in _ENV_WRAPPERS:
        return _validation_tool_for_segment(_strip_env_wrapper(segment[1:]))

    if executable == "command":
        return _validation_tool_for_segment(_strip_command_wrapper(segment[1:]))

    if executable == "time":
        return _validation_tool_for_segment(_strip_time_wrapper(segment[1:]))

    if executable in _SHELL_WRAPPERS:
        return _validation_tool_for_shell_wrapper(segment[1:])

    if executable in _VALIDATION_TOOLS:
        return executable

    if executable == "uv" and "run" in segment:
        nested_segment = segment[segment.index("run") + 1 :]
        for position in range(len(nested_segment)):
            validation_tool = _validation_tool_for_segment(nested_segment[position:])
            if validation_tool is not None:
                return validation_tool

    if _is_python_executable(executable):
        for position, value in enumerate(segment):
            if value in _PYTHON_MODULE_FLAGS and position + 1 < len(segment):
                module_name = segment[position + 1]
                if module_name in _VALIDATION_TOOLS:
                    return module_name

    return None


def _is_python_executable(executable: str) -> bool:
    return re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable) is not None


def _strip_env_wrapper(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if _is_env_assignment(token):
            index += 1
            continue
        if token == "--":
            return tokens[index + 1 :]
        if token in {"-S", "--split-string"}:
            if index + 1 >= len(tokens):
                return []
            return _strip_env_wrapper(
                _tokens_for_command(tokens[index + 1]) + tokens[index + 2 :]
            )
        if token in _ENV_OPTIONS_WITHOUT_OPERAND:
            index += 1
            continue
        if token in _ENV_OPTIONS_WITH_OPERAND:
            index += 2
            continue
        if any(
            token.startswith(f"{option}=")
            for option in ("--chdir", "--unset")
        ):
            index += 1
            continue
        if token.startswith("--split-string="):
            return _strip_env_wrapper(
                _tokens_for_command(token.split("=", 1)[1]) + tokens[index + 1 :]
            )
        if token.startswith("-S") and len(token) > 2:
            return _strip_env_wrapper(
                _tokens_for_command(token[2:]) + tokens[index + 1 :]
            )
        if (
            token.startswith("-u")
            or token.startswith("-C")
        ) and len(token) > 2:
            index += 1
            continue
        return tokens[index:]
    return []


def _strip_command_wrapper(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens) and tokens[index] in _COMMAND_OPTIONS_WITHOUT_OPERAND:
        index += 1
    return tokens[index:]


def _strip_time_wrapper(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return tokens[index + 1 :]
        if token in _TIME_OPTIONS_WITHOUT_OPERAND:
            index += 1
            continue
        if token in _TIME_OPTIONS_WITH_OPERAND:
            index += 2
            continue
        if any(token.startswith(f"{option}=") for option in ("--format", "--output")):
            index += 1
            continue
        return tokens[index:]
    return []


def _validation_tool_for_shell_wrapper(tokens: list[str]) -> str | None:
    command = _shell_command_string(tokens)
    if command is None:
        return None
    return _contains_validation_command(command)


def _shell_command_string(tokens: list[str]) -> str | None:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            continue
        if token == "-c" or (
            token.startswith("-")
            and not token.startswith("--")
            and "c" in token[1:]
        ):
            return tokens[index + 1] if index + 1 < len(tokens) else None
        if token in _SHELL_OPTIONS_WITH_OPERAND:
            index += 2
            continue
        if any(
            token.startswith(f"{option}=")
            for option in ("--init-file", "--rcfile")
        ):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return None
    return None


def _contains_validation_command(command: str) -> str | None:
    tokens = _tokens_for_command(command)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _CHAIN_OPERATORS:
            index += 1
            continue

        segment: list[str] = []
        while index < len(tokens) and tokens[index] not in _CHAIN_OPERATORS:
            segment.append(tokens[index])
            index += 1

        validation_tool = _validation_tool_for_segment(segment)
        if validation_tool is None:
            continue

        return validation_tool

    return None


def _block_reason(tool_name: str) -> str:
    return (
        f"Direct `{tool_name}` validation is blocked in Bragi. "
        "Run `python3 .codex/tools/validate.py` instead. Use "
        "`--full` for the exhaustive gate, or `--tests-only`, "
        "`--typecheck-only`, and `--lint-only` for focused reruns."
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
                }
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

    command = _command_from_payload(payload)
    if not command:
        return 0

    validation_tool = _contains_validation_command(command)
    if validation_tool is None:
        return 0

    _emit_block(_block_reason(validation_tool))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
