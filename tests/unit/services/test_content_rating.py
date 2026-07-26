from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.content_rating_instructions import content_rating_exceeds
from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.content_rating import (
    CONTENT_RATING_G,
    CONTENT_RATING_PG,
    CONTENT_RATING_PG_13,
    CONTENT_RATING_R,
    CONTENT_RATING_UNRATED,
    ContentSafetyPolicy,
    effective_content_safety_policy,
    sanitize_content_rating,
    set_user_content_rating,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_sanitize_content_rating_defaults_to_pg_13() -> None:
    assert sanitize_content_rating(None) == CONTENT_RATING_PG_13
    assert sanitize_content_rating(" PG-13 ") == CONTENT_RATING_PG_13
    assert sanitize_content_rating("unrated") == CONTENT_RATING_UNRATED
    assert sanitize_content_rating("unknown") == CONTENT_RATING_PG_13


def test_unclassified_content_is_hidden_from_every_rated_ceiling() -> None:
    for allowed_rating in ("g", "pg", "pg-13", "r"):
        assert content_rating_exceeds(
            minimum_rating="unclassified",
            allowed_rating=allowed_rating,
        )
    assert not content_rating_exceeds(
        minimum_rating="unclassified",
        allowed_rating="unrated",
    )


def test_effective_policy_defaults_by_account_role(
    repositories: PersistenceRepositories,
) -> None:
    adult = repositories.create_user(
        username="adult",
        role="user",
        password_hash="hash",
    )
    child = repositories.create_user(
        username="child",
        role="child",
        password_hash="hash",
    )

    assert effective_content_safety_policy(
        repositories,
        user_id=adult.id,
    ) == ContentSafetyPolicy(
        rating=CONTENT_RATING_PG_13,
        fade_to_black_enabled=True,
        force_venice_safe_mode=False,
    )
    assert effective_content_safety_policy(
        repositories,
        user_id=child.id,
    ) == ContentSafetyPolicy(
        rating=CONTENT_RATING_PG,
        fade_to_black_enabled=True,
        force_venice_safe_mode=True,
    )


def test_child_does_not_inherit_global_pg_13_without_account_grant(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_scoped_setting(
        scope="global",
        key="content_filter_rating",
        value=CONTENT_RATING_PG_13,
    )
    adult = repositories.create_user(
        username="adult",
        role="user",
        password_hash="hash",
    )
    child = repositories.create_user(
        username="child",
        role="child",
        password_hash="hash",
    )

    assert effective_content_safety_policy(
        repositories,
        user_id=adult.id,
    ).rating == CONTENT_RATING_PG_13
    assert effective_content_safety_policy(
        repositories,
        user_id=child.id,
    ).rating == CONTENT_RATING_PG


def test_child_may_self_select_g_or_pg_but_admin_may_grant_pg_13(
    repositories: PersistenceRepositories,
) -> None:
    child = repositories.create_user(
        username="child",
        role="child",
        password_hash="hash",
    )

    set_user_content_rating(
        repositories,
        user_id=child.id,
        rating=CONTENT_RATING_G,
    )
    assert effective_content_safety_policy(repositories, user_id=child.id).rating == (
        CONTENT_RATING_G
    )

    with pytest.raises(ValueError, match="Child accounts may select only G or PG"):
        set_user_content_rating(
            repositories,
            user_id=child.id,
            rating=CONTENT_RATING_PG_13,
        )

    set_user_content_rating(
        repositories,
        user_id=child.id,
        rating=CONTENT_RATING_PG_13,
        admin_grant=True,
    )
    assert effective_content_safety_policy(repositories, user_id=child.id).rating == (
        CONTENT_RATING_PG_13
    )

    with pytest.raises(ValueError, match="cannot exceed PG-13"):
        set_user_content_rating(
            repositories,
            user_id=child.id,
            rating=CONTENT_RATING_R,
            admin_grant=True,
        )


def test_child_policy_forces_safety_toggles_even_if_unsafe_values_are_stored(
    repositories: PersistenceRepositories,
) -> None:
    child = repositories.create_user(
        username="child",
        role="child",
        password_hash="hash",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=child.id,
        key="fade_to_black_enabled",
        value=False,
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=child.id,
        key="content_filter_rating",
        value=CONTENT_RATING_UNRATED,
    )

    assert effective_content_safety_policy(
        repositories,
        user_id=child.id,
    ) == ContentSafetyPolicy(
        rating=CONTENT_RATING_PG,
        fade_to_black_enabled=True,
        force_venice_safe_mode=True,
    )


def test_unrated_policy_always_disables_fade_to_black(
    repositories: PersistenceRepositories,
) -> None:
    adult = repositories.create_user(
        username="adult",
        role="user",
        password_hash="hash",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=adult.id,
        key="content_filter_rating",
        value=CONTENT_RATING_UNRATED,
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=adult.id,
        key="fade_to_black_enabled",
        value=True,
    )

    assert effective_content_safety_policy(
        repositories,
        user_id=adult.id,
    ).fade_to_black_enabled is False
