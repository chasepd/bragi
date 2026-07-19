from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.model_capabilities import (
    CHAT_CAPABILITIES,
    IMAGE_TO_IMAGE_CAPABILITIES,
    check_model_capabilities,
    model_supports_any_capability,
    model_supports_any_capability_or_unknown,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_model_supports_any_capability_requires_available_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )

    assert (
        model_supports_any_capability(
            repositories,
            provider="fake",
            model_id="fake-chat",
            required=CHAT_CAPABILITIES,
        )
        is False
    )


def test_model_supports_any_capability_or_unknown_accepts_missing_catalog_row(
    repositories: PersistenceRepositories,
) -> None:
    assert (
        model_supports_any_capability_or_unknown(
            repositories,
            provider="fake",
            model_id="unsynced-chat",
            required=CHAT_CAPABILITIES,
        )
        is True
    )


def test_model_supports_any_capability_rejects_missing_catalog_row(
    repositories: PersistenceRepositories,
) -> None:
    assert (
        model_supports_any_capability(
            repositories,
            provider="fake",
            model_id="unsynced-chat",
            required=CHAT_CAPABILITIES,
        )
        is False
    )


def test_check_model_capabilities_reports_unavailable_separately(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image-edit"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )

    result = check_model_capabilities(
        repositories,
        provider="fake",
        model_id="fake-edit",
        required=IMAGE_TO_IMAGE_CAPABILITIES,
    )

    assert result.found is True
    assert result.available is False
    assert result.supported is False
    assert result.reason == "model_unavailable"


def test_check_model_capabilities_accepts_aliases_for_available_models(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image-edit"],
    )

    result = check_model_capabilities(
        repositories,
        provider="fake",
        model_id="fake-edit",
        required=IMAGE_TO_IMAGE_CAPABILITIES,
    )

    assert result.found is True
    assert result.available is True
    assert result.supported is True
    assert result.reason is None
