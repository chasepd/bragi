from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.engine_health_service import EngineHealthService
from bragi.services.job_lifecycle import JobLifecycleService


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_engine_health_does_not_expose_raw_job_errors(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    jobs = JobLifecycleService(repositories=repositories)
    failed = jobs.create_running(
        save_id=save.id,
        type="context_search",
        payload={},
    )
    jobs.fail(
        failed.id,
        error="provider leaked private endpoint https://example.invalid/secret",
    )

    snapshot = EngineHealthService(repositories).snapshot(save.id)

    assert snapshot.latest_context_search == {
        "status": "failed",
        "error_present": True,
        "result_counts": {},
        "diagnostics": {},
        "retrieval_degraded": False,
    }
    assert "secret" not in json.dumps(snapshot.latest_context_search)


def test_engine_health_warns_on_degraded_context_search(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    jobs = JobLifecycleService(repositories=repositories)
    context_search = jobs.create_running(
        save_id=save.id,
        type="context_search",
        payload={},
    )
    jobs.succeed(
        context_search.id,
        result={
            "selected_memories": [{"source_id": "memory-1"}],
            "retrieval_degraded": True,
            "retrieval_recovery": "deterministic_fallback",
            "provider": "fake",
            "model": "fake-context",
            "raw_prompt": "Secret chronicle phrase",
        },
    )

    snapshot = EngineHealthService(repositories).snapshot(save.id)

    assert snapshot.latest_context_search == {
        "status": "succeeded",
        "error_present": False,
        "result_counts": {"selected_memories": 1},
        "diagnostics": {},
        "retrieval_degraded": True,
        "retrieval_recovery": "deterministic_fallback",
    }
    warnings = {warning.code: warning for warning in snapshot.warnings}
    assert warnings["degraded_context_search"].severity == "warning"
    assert "Secret chronicle phrase" not in json.dumps(snapshot.latest_context_search)


def test_engine_health_reports_curation_backlog_without_worker_secrets(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="event",
        claim="The beacon was relit.",
        evidence_quote="The beacon was relit.",
        source_message_ids=[],
        scope="durable",
        confidence=0.9,
    )
    repositories.claim_context_observations(
        (observation.id,),
        lease_token="private-worker-token",
        lease_seconds=600,
    )

    snapshot = EngineHealthService(repositories).snapshot(save.id)

    assert snapshot.observation_curation.pending_count == 1
    assert snapshot.observation_curation.eligible_count == 0
    assert snapshot.observation_curation.leased_count == 1
    assert snapshot.observation_curation.total_attempt_count == 1
    warnings = {warning.code: warning for warning in snapshot.warnings}
    assert warnings["observation_curation_retries"].severity == "warning"
    assert "private-worker-token" not in json.dumps(snapshot, default=str)
