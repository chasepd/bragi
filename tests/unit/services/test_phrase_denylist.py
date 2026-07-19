from __future__ import annotations

import sqlite3
from pathlib import Path

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.phrase_denylist import (
    GENERATED_PHRASE_DENYLIST_SETTING,
    SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
    denied_phrase_violations,
    effective_generated_phrase_denylist,
    sanitize_generated_phrase_denylist,
    summarize_phrase_policy_violations,
)


def test_phrase_denylist_sanitizes_lines_and_dedupes_case_insensitively() -> None:
    assert sanitize_generated_phrase_denylist(
        "  That's not nothing  \n\nthat's NOT nothing\nthat is everything "
    ) == "That's not nothing\nthat is everything"


def test_phrase_denylist_matches_case_quotes_and_whitespace() -> None:
    violations = denied_phrase_violations(
        "She says, “that’s   not\nnothing,” and waits.",
        phrases=("That's not nothing",),
        field_name="narrator_message",
    )

    assert len(violations) == 1
    assert violations[0].field_name == "narrator_message"
    assert violations[0].phrase == "That's not nothing"
    assert violations[0].match_count == 1
    assert "That's not nothing" in summarize_phrase_policy_violations(violations)


def test_effective_phrase_denylist_merges_defaults_global_and_save(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    connection = sqlite3.connect(database_path)
    repositories = PersistenceRepositories(connection)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A beacon tower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="save-only phrase\nthat's actually everything",
    )

    assert effective_generated_phrase_denylist(repositories, save_id=save.id) == (
        "That's not nothing",
        "that's actually everything",
        "save-only phrase",
    )

    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )

    assert effective_generated_phrase_denylist(repositories, save_id=save.id) == (
        "save-only phrase",
        "that's actually everything",
    )
