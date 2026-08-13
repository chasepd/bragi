from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.chat_timing_service import ChatTimingService


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def _save_with_chat_model(repositories: PersistenceRepositories) -> str:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Beacon Watch",
        premise="A beacon needs tending.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Beacon Watch")
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    return save.id


def test_timing_summary_suppresses_estimate_until_five_matching_turns(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _save_with_chat_model(repositories)
    repositories.list_chat_response_commit_durations = lambda **_kwargs: [  # type: ignore[method-assign]
        1_000,
        2_000,
        3_000,
        4_000,
    ]

    summary = ChatTimingService(repositories).summary(save_id)

    assert summary.mode == "quality"
    assert summary.provider == "fake"
    assert summary.model == "fake-chat"
    assert summary.sample_count == 4
    assert summary.estimate is None


def test_timing_summary_uses_latest_thirty_matching_successes_and_nearest_rank(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _save_with_chat_model(repositories)
    calls: list[dict[str, object]] = []

    def durations(**kwargs: object) -> list[int]:
        calls.append(kwargs)
        return [1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000]

    repositories.list_chat_response_commit_durations = durations  # type: ignore[method-assign]

    summary = ChatTimingService(repositories).summary(save_id)

    assert calls == [{
        "save_id": save_id,
        "provider": "fake",
        "model": "fake-chat",
        "limit": 30,
    }]
    assert summary.sample_count == 7
    assert summary.estimate is not None
    assert summary.estimate.p50_ms == 4_000
    assert summary.estimate.p95_ms == 7_000
