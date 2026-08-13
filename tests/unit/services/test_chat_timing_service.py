from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import ChatTurnOutcomeRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.chat_timing_service import ChatTimingService
from bragi.services.turn_responsiveness import (
    TURN_RESPONSIVENESS_MODE_RESPONSIVE,
    TURN_RESPONSIVENESS_MODE_SETTING,
)


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
        "mode": "quality",
        "limit": 30,
    }]
    assert summary.sample_count == 7
    assert summary.estimate is not None
    assert summary.estimate.p50_ms == 4_000
    assert summary.estimate.p95_ms == 7_000


def test_timing_summary_uses_effective_responsive_mode_for_samples(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _save_with_chat_model(repositories)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=TURN_RESPONSIVENESS_MODE_SETTING,
        value=TURN_RESPONSIVENESS_MODE_RESPONSIVE,
    )
    calls: list[dict[str, object]] = []

    def durations(**kwargs: object) -> list[int]:
        calls.append(kwargs)
        return []

    repositories.list_chat_response_commit_durations = durations  # type: ignore[method-assign]

    summary = ChatTimingService(repositories).summary(save_id)

    assert summary.mode == "responsive"
    assert calls[0]["mode"] == "responsive"


def test_timing_summary_reports_failure_rate_and_successful_route_usage(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _save_with_chat_model(repositories)
    repositories.list_chat_response_commit_durations = lambda **_kwargs: [  # type: ignore[method-assign]
        1_000,
        2_000,
    ]
    repositories.list_chat_turn_outcomes = lambda **_kwargs: [  # type: ignore[method-assign]
        ChatTurnOutcomeRecord(
            status="succeeded",
            fast_path_used=True,
            combined_path_used=False,
        ),
        ChatTurnOutcomeRecord(
            status="succeeded",
            fast_path_used=False,
            combined_path_used=True,
        ),
        ChatTurnOutcomeRecord(
            status="succeeded",
            fast_path_used=False,
            combined_path_used=False,
        ),
        ChatTurnOutcomeRecord(status="succeeded"),
        ChatTurnOutcomeRecord(status="failed"),
        ChatTurnOutcomeRecord(status="cancelled"),
    ]

    outcomes = ChatTimingService(repositories).summary(save_id).outcomes

    assert outcomes.terminal_count == 6
    assert outcomes.success_count == 4
    assert outcomes.failed_count == 1
    assert outcomes.interrupted_count == 1
    assert outcomes.failure_rate == pytest.approx(1 / 3)
    assert outcomes.route_sample_count == 3
    assert outcomes.fast_path_count == 1
    assert outcomes.combined_path_count == 1
    assert outcomes.standard_path_count == 1
    assert outcomes.unclassified_success_count == 1
