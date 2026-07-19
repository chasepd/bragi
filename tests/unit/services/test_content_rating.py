from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.content_rating import (
    CONTENT_RATING_G,
    CONTENT_RATING_PG,
    CONTENT_RATING_PG_13,
    CONTENT_RATING_R,
    CONTENT_RATING_UNRATED,
    ContentSafetyPolicy,
    classify_content_rating,
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


@pytest.mark.parametrize(
    ("body", "expected"),
    (
        ("The ranger studies the quiet road.", CONTENT_RATING_G),
        ("They share a brief kiss beneath the lantern.", CONTENT_RATING_PG),
        ("The rivals fight with wooden practice swords.", CONTENT_RATING_PG),
        ("They undressed before the scene moved on.", CONTENT_RATING_PG_13),
        ("Blood streaked the floor after the guard was stabbed.", CONTENT_RATING_PG_13),
        ("They had sex after returning to the inn.", CONTENT_RATING_R),
        ("She has sex with him.", CONTENT_RATING_R),
        ("A sex act on a bed.", CONTENT_RATING_R),
        ("They described sexual acts in detail.", CONTENT_RATING_R),
        ("They had rough sex all night.", CONTENT_RATING_R),
        ("They started to have passionate sex.", CONTENT_RATING_R),
        ("They began sexual intercourse in full view.", CONTENT_RATING_R),
        ("They started sex after the door closed.", CONTENT_RATING_R),
        ("The adults were copulating in the bedroom.", CONTENT_RATING_R),
        ("The blast dismembered the soldier in graphic detail.", CONTENT_RATING_R),
        (
            "He disemboweled the guard, spilling his organs across the floor.",
            CONTENT_RATING_R,
        ),
        (
            "He chopped off the prisoner's head and limbs, spraying the walls red.",
            CONTENT_RATING_R,
        ),
        (
            "He crushed her skull until her brains spilled across the floor.",
            CONTENT_RATING_R,
        ),
        ("She smashed his head and blood sprayed across the wall.", CONTENT_RATING_R),
        ("The attacker ripped out his victim's heart.", CONTENT_RATING_R),
        ("He slit the prisoner's throat.", CONTENT_RATING_R),
        ("He blew the prisoner's head off.", CONTENT_RATING_R),
        ("She tore the attacker's arm off.", CONTENT_RATING_R),
        ("He gouged out the guard's eyes.", CONTENT_RATING_R),
        ("His intestines spilled onto the floor.", CONTENT_RATING_R),
        ("He forced his cock inside her and came inside her.", CONTENT_RATING_R),
        ("He forced himself inside her.", CONTENT_RATING_R),
        ("He came all over her.", CONTENT_RATING_R),
        ("Explicit sex between adults.", CONTENT_RATING_R),
        ("Bare breasts and exposed genitals.", CONTENT_RATING_R),
        ("A topless adult poses in a bedroom.", CONTENT_RATING_R),
        ("An erotic portrait emphasizing cleavage.", CONTENT_RATING_R),
        ("A close-up of nipples and buttocks.", CONTENT_RATING_R),
        ("He raped and tortured the prisoner.", CONTENT_RATING_R),
        ("She points a rifle at him before murdering him.", CONTENT_RATING_R),
        ("He commits suicide in front of the children.", CONTENT_RATING_R),
        ("A r.a.p.e and torture scene.", CONTENT_RATING_R),
    ),
)
def test_classify_content_rating_assigns_minimum_supported_level(
    body: str,
    expected: str,
) -> None:
    assert classify_content_rating(body) == expected


def test_sanitize_content_rating_defaults_to_pg_13() -> None:
    assert sanitize_content_rating(None) == CONTENT_RATING_PG_13
    assert sanitize_content_rating(" PG-13 ") == CONTENT_RATING_PG_13
    assert sanitize_content_rating("unrated") == CONTENT_RATING_UNRATED
    assert sanitize_content_rating("unknown") == CONTENT_RATING_PG_13


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
