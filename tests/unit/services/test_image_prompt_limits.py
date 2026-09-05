from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.image_prompt_limits import (
    IMAGE_PROMPT_MODEL_LIMITS_SETTING,
    image_prompt_max_chars,
    refresh_image_prompt_model_limits,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


@pytest.mark.parametrize(
    ("provider", "has_references", "expected"),
    [("venice", False, 7500), ("venice", True, 32768), ("openrouter", False, None)],
)
def test_image_prompt_limit_without_model_metadata(
    repositories: PersistenceRepositories,
    provider: str,
    has_references: bool,
    expected: int | None,
) -> None:
    assert image_prompt_max_chars(
        repositories,
        provider=provider,
        model_id="unlisted-model",
        has_references=has_references,
    ) == expected


@pytest.mark.parametrize(
    ("provider", "model_limit", "has_references", "expected"),
    [
        ("venice", 1000, False, 1000),
        ("venice", 50000, False, 7500),
        ("venice", 10000, True, 10000),
        ("venice", 50000, True, 32768),
        ("openrouter", 12000, False, 12000),
    ],
)
def test_image_prompt_limit_uses_smallest_known_capacity(
    repositories: PersistenceRepositories,
    provider: str,
    model_limit: int,
    has_references: bool,
    expected: int,
) -> None:
    repositories.set_app_setting(
        IMAGE_PROMPT_MODEL_LIMITS_SETTING, {provider: {"image-model": model_limit}}
    )

    assert image_prompt_max_chars(
        repositories,
        provider=provider,
        model_id="image-model",
        has_references=has_references,
    ) == expected


@pytest.mark.parametrize("invalid_limit", [0, -1, True, "bad", None, 1.5, {}])
def test_image_prompt_limit_ignores_invalid_cached_values(
    repositories: PersistenceRepositories, invalid_limit: object
) -> None:
    repositories.set_app_setting(
        IMAGE_PROMPT_MODEL_LIMITS_SETTING,
        {"venice": {"image-model": invalid_limit}},
    )

    assert image_prompt_max_chars(
        repositories, provider="venice", model_id="image-model"
    ) == 7500


def test_refresh_limits_normalizes_official_metadata_without_modifying_it(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting(
        IMAGE_PROMPT_MODEL_LIMITS_SETTING,
        {"venice": {"removed-model": 500}, "openrouter": {"other-model": 9000}},
    )
    raw_metadata = {
        "data": [
            {
                "id": "detailed-image",
                "model_spec": {"constraints": {"promptCharacterLimit": 1500}},
            },
            {
                "id": "invalid-image",
                "model_spec": {"constraints": {"promptCharacterLimit": -1}},
            },
        ]
    }

    refresh_image_prompt_model_limits(
        repositories, provider="venice", raw_metadata=raw_metadata
    )

    assert repositories.get_app_setting(IMAGE_PROMPT_MODEL_LIMITS_SETTING) == {
        "venice": {"detailed-image": 1500},
        "openrouter": {"other-model": 9000},
    }
    assert raw_metadata["data"][0]["model_spec"] == {
        "constraints": {"promptCharacterLimit": 1500}
    }


def test_refresh_limits_accepts_normalized_model_metadata_and_clears_old_cache(
    repositories: PersistenceRepositories,
) -> None:
    refresh_image_prompt_model_limits(
        repositories,
        provider="openrouter",
        raw_metadata={
            "data": [{"id": "image-model", "prompt_character_limit": 12000}]
        },
    )

    assert image_prompt_max_chars(
        repositories, provider="openrouter", model_id="image-model"
    ) == 12000

    refresh_image_prompt_model_limits(
        repositories, provider="openrouter", raw_metadata={}
    )

    assert image_prompt_max_chars(
        repositories, provider="openrouter", model_id="image-model"
    ) is None
