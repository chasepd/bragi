"""Deterministic phrase denylist for generated prose and profile text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bragi.persistence.repositories import PersistenceRepositories
from bragi.retry_policy import MODEL_OUTPUT_MAX_ATTEMPTS

GENERATED_PHRASE_DENYLIST_SETTING = "generated_phrase_denylist"
SAVE_GENERATED_PHRASE_DENYLIST_SETTING = "save_generated_phrase_denylist"
DEFAULT_GENERATED_PHRASE_DENYLIST = (
    "That's not nothing",
    "that's actually everything",
)
GENERATED_PHRASE_GUARD_MAX_ATTEMPTS = MODEL_OUTPUT_MAX_ATTEMPTS

_WHITESPACE_RE = re.compile(r"\s+")
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "`": "'",
        "\u00b4": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)


@dataclass(frozen=True)
class PhraseDenylistViolation:
    field_name: str
    phrase: str
    match_count: int

    def diagnostic(self) -> dict[str, object]:
        return {
            "field": self.field_name,
            "phrase": self.phrase,
            "match_count": self.match_count,
        }


def sanitize_generated_phrase_denylist(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "\n".join(_dedupe_phrases(value.splitlines()))


def generated_phrase_denylist_text(
    repositories: PersistenceRepositories,
) -> str:
    value = repositories.get_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
    )
    if value is None:
        return "\n".join(DEFAULT_GENERATED_PHRASE_DENYLIST)
    return sanitize_generated_phrase_denylist(value)


def save_generated_phrase_denylist_text(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None,
) -> str:
    if save_id is None:
        return ""
    value = repositories.get_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
    )
    return sanitize_generated_phrase_denylist(value)


def effective_generated_phrase_denylist(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> tuple[str, ...]:
    return _dedupe_phrases(
        (
            *generated_phrase_denylist_text(repositories).splitlines(),
            *save_generated_phrase_denylist_text(
                repositories,
                save_id=save_id,
            ).splitlines(),
        )
    )


def denied_phrase_violations(
    value: object,
    *,
    phrases: tuple[str, ...],
    field_name: str,
) -> tuple[PhraseDenylistViolation, ...]:
    if not isinstance(value, str) or not value or not phrases:
        return ()
    normalized_value = _normalized_match_text(value)
    violations: list[PhraseDenylistViolation] = []
    for phrase in phrases:
        normalized_phrase = _normalized_match_text(phrase)
        if not normalized_phrase:
            continue
        count = normalized_value.count(normalized_phrase)
        if count:
            violations.append(
                PhraseDenylistViolation(
                    field_name=field_name,
                    phrase=phrase,
                    match_count=count,
                )
            )
    return tuple(violations)


def first_phrase_violation_diagnostic(
    violations: tuple[PhraseDenylistViolation, ...],
) -> dict[str, object]:
    if not violations:
        return {}
    return violations[0].diagnostic()


def summarize_phrase_policy_violations(
    violations: tuple[PhraseDenylistViolation, ...],
) -> str:
    if not violations:
        return "generated text phrase denylist violation"
    first = violations[0]
    return (
        "generated text phrase denylist violation: "
        f"field={first.field_name}, phrase={first.phrase!r}"
    )


def _dedupe_phrases(values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple | list):
        return ()
    phrases: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        phrase = value.strip()
        if not phrase:
            continue
        key = _normalized_match_text(phrase)
        if not key or key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
    return tuple(phrases)


def _normalized_match_text(value: str) -> str:
    return (
        _WHITESPACE_RE.sub(" ", value.translate(_QUOTE_TRANSLATION))
        .strip()
        .casefold()
    )
