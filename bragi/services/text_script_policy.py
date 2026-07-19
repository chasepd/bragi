"""Deterministic script guard for provider-generated persistence text."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from bragi.persistence.repositories import PersistenceRepositories

SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT = "source_aware_reject"
SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT = "latin_only_reject"
SCRIPT_GUARD_MODE_OFF = "off"
SCRIPT_GUARD_MODE_OPTIONS = (
    SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT,
    SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
    SCRIPT_GUARD_MODE_OFF,
)
DEFAULT_SCRIPT_GUARD_MODE = SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT
SCRIPT_GUARD_MODE_SETTING = "generated_text_script_guard_mode"

_SIGNIFICANT_TOTAL_CHARS = 4
_SIGNIFICANT_RUN_CHARS = 2

_SCRIPT_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "Han": re.compile(
        r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
        r"\U00020000-\U0002ceaf]"
    ),
    "Kana": re.compile(r"[\u3040-\u30ff]"),
    "Hangul": re.compile(r"[\uac00-\ud7af]"),
    "Cyrillic": re.compile(r"[\u0400-\u052f]"),
    "Arabic": re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]"),
    "Hebrew": re.compile(r"[\u0590-\u05ff]"),
    "Thai": re.compile(r"[\u0e00-\u0e7f]"),
    "Devanagari": re.compile(r"[\u0900-\u097f]"),
    "Greek": re.compile(r"[\u0370-\u03ff]"),
}


@dataclass(frozen=True)
class ScriptPolicyViolation:
    field_name: str
    script: str
    character_count: int
    longest_run: int

    def diagnostic(self) -> dict[str, object]:
        return {
            "field": self.field_name,
            "script": self.script,
            "character_count": self.character_count,
            "longest_run": self.longest_run,
        }


def sanitize_script_guard_mode(value: object) -> str:
    if isinstance(value, str) and value in SCRIPT_GUARD_MODE_OPTIONS:
        return value
    return DEFAULT_SCRIPT_GUARD_MODE


def script_guard_mode(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> str:
    return sanitize_script_guard_mode(
        repositories.get_effective_setting(
            SCRIPT_GUARD_MODE_SETTING,
            save_id=save_id,
        )
    )


def allowed_generated_scripts(source_texts: Iterable[str]) -> frozenset[str]:
    allowed: set[str] = set()
    for text in source_texts:
        for script, count, longest_run in _script_counts(text):
            if _script_is_significant(count=count, longest_run=longest_run):
                allowed.add(script)
    return frozenset(allowed)


def text_script_violations(
    value: object,
    *,
    allowed_scripts: frozenset[str],
    mode: str,
    field_name: str,
) -> tuple[ScriptPolicyViolation, ...]:
    if mode == SCRIPT_GUARD_MODE_OFF or not isinstance(value, str):
        return ()
    allowed = (
        frozenset()
        if mode == SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT
        else allowed_scripts
    )
    violations: list[ScriptPolicyViolation] = []
    for script, count, longest_run in _script_counts(value):
        if script in allowed:
            continue
        if not _script_is_significant(count=count, longest_run=longest_run):
            continue
        violations.append(
            ScriptPolicyViolation(
                field_name=field_name,
                script=script,
                character_count=count,
                longest_run=longest_run,
            )
        )
    return tuple(violations)


def object_text_script_violations(
    value: object,
    *,
    allowed_scripts: frozenset[str],
    mode: str,
    field_name: str,
) -> tuple[ScriptPolicyViolation, ...]:
    violations: list[ScriptPolicyViolation] = []
    for path, text in _iter_text_values(value, field_name):
        violations.extend(
            text_script_violations(
                text,
                allowed_scripts=allowed_scripts,
                mode=mode,
                field_name=path,
            )
        )
    return tuple(violations)


def first_violation_diagnostic(
    violations: tuple[ScriptPolicyViolation, ...],
) -> dict[str, object]:
    if not violations:
        return {}
    return violations[0].diagnostic()


def summarize_script_policy_violations(
    violations: tuple[ScriptPolicyViolation, ...],
) -> str:
    if not violations:
        return "generated text script policy violation"
    first = violations[0]
    return (
        f"generated text script policy violation: field={first.field_name}, "
        f"script={first.script}"
    )


def _script_counts(text: str) -> tuple[tuple[str, int, int], ...]:
    counts: list[tuple[str, int, int]] = []
    for script, pattern in _SCRIPT_PATTERNS.items():
        total = 0
        longest_run = 0
        current_run = 0
        last_end = -1
        for match in pattern.finditer(text):
            total += 1
            if match.start() == last_end:
                current_run += 1
            else:
                current_run = 1
            last_end = match.end()
            longest_run = max(longest_run, current_run)
        if total:
            counts.append((script, total, longest_run))
    return tuple(counts)


def _script_is_significant(*, count: int, longest_run: int) -> bool:
    return count >= _SIGNIFICANT_TOTAL_CHARS or longest_run >= _SIGNIFICANT_RUN_CHARS


def _iter_text_values(value: object, path: str) -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).strip() or "field"
            yield from _iter_text_values(item, f"{path}.{key_text}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _iter_text_values(item, f"{path}[{index}]")
