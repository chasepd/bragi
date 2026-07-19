from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories

_MISSING = object()
_SENTINEL_SECRET = "super-secret-token"
_SENTINEL_BEARER = f"Bearer {_SENTINEL_SECRET}"
_SENTINEL_UPPER_BEARER = f"BEARER {_SENTINEL_SECRET}"
_SENTINEL_TOKEN_PARAM = f"token={_SENTINEL_SECRET}"


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_diagnostics_lists_provider_and_job_failures_without_api_keys(
    repositories: PersistenceRepositories,
) -> None:
    from bragi.services.diagnostics_service import DiagnosticsService

    save = _save(repositories)
    failed_job = repositories.create_job(
        save_id=save.id,
        type="image_generation",
        status="running",
        payload={
            "provider": "openrouter",
            "model": "openrouter/image-model",
            "api_key": _SENTINEL_SECRET,
        },
    )
    repositories.update_job(
        failed_job.id,
        status="failed",
        error=f"image provider rejected {_SENTINEL_TOKEN_PARAM} during generation",
        result={
            "attempt_count": 3,
            "max_attempts": 3,
            "retry_attempts": [
                {
                    "attempt": 1,
                    "duration_ms": 10,
                    "error_category": "rate_limited",
                    "http_status": 429,
                    "raw": f"unsafe {_SENTINEL_SECRET}",
                },
                {
                    "attempt": 2,
                    "duration_ms": 15,
                    "error_category": "rate_limited",
                    "http_status": 429,
                },
                {
                    "attempt": 3,
                    "duration_ms": 20,
                    "error_category": "rate_limited",
                    "http_status": 429,
                },
            ],
        },
    )
    succeeded_job = repositories.create_job(
        save_id=save.id,
        type="summarization",
        status="queued",
        payload={"provider": "openrouter"},
    )
    repositories.start_job(succeeded_job.id)
    repositories.update_job(succeeded_job.id, status="succeeded")
    repositories.upsert_provider_config(
        provider="openrouter",
        enabled=True,
        has_api_key=True,
        last_error=f"authentication failed for {_SENTINEL_BEARER}",
    )
    repositories.upsert_provider_config(
        provider="venice",
        enabled=True,
        has_api_key=False,
        last_error=None,
    )
    repositories.upsert_provider_config(
        provider="api-key-provider",
        enabled=True,
        has_api_key=True,
        last_error=f"model refresh failed with api_key: {_SENTINEL_SECRET}",
    )
    repositories.upsert_provider_config(
        provider="jsonish-provider",
        enabled=True,
        has_api_key=True,
        last_error=(
            f'metadata sync failed with {{"api_key":"{_SENTINEL_SECRET}"}}'
        ),
    )
    bearer_job = repositories.create_job(
        save_id=save.id,
        type="chat",
        status="running",
        payload={"provider": "openrouter"},
    )
    repositories.update_job(
        bearer_job.id,
        status="failed",
        error=f"chat provider rejected {_SENTINEL_UPPER_BEARER} during reply",
    )
    cleanup_job = repositories.create_job(
        save_id=save.id,
        type="image_cleanup",
        status="running",
        payload={"provider": "openrouter"},
    )
    repositories.cancel_job(
        cleanup_job.id,
        error=f"user cancelled provider cleanup with {_SENTINEL_TOKEN_PARAM}",
    )

    diagnostics = DiagnosticsService(repositories=repositories).list_diagnostics()

    provider_entries = {
        _value(entry, "provider"): entry
        for entry in _list(_value(diagnostics, "provider_configs", "providers"))
    }
    assert provider_entries["openrouter"]
    assert _value(provider_entries["openrouter"], "last_error") == (
        "authentication failed for Bearer [redacted]"
    )
    assert "authentication failed" in _value(
        provider_entries["openrouter"],
        "last_error",
    )
    assert _SENTINEL_SECRET not in _value(
        provider_entries["openrouter"],
        "last_error",
    )
    assert _value(provider_entries["openrouter"], "has_api_key") is True
    assert _value(provider_entries["openrouter"], "api_key", default=None) is None
    assert _value(provider_entries["venice"], "last_error") is None
    assert _value(provider_entries["api-key-provider"], "last_error") == (
        "model refresh failed with api_key: [redacted]"
    )
    assert "model refresh failed" in _value(
        provider_entries["api-key-provider"],
        "last_error",
    )
    assert _SENTINEL_SECRET not in _value(
        provider_entries["api-key-provider"],
        "last_error",
    )
    jsonish_error = _value(provider_entries["jsonish-provider"], "last_error")
    assert "metadata sync failed" in jsonish_error
    assert '"api_key"' in jsonish_error
    assert "[redacted]" in jsonish_error
    assert _SENTINEL_SECRET not in jsonish_error

    failed_jobs = _list(_value(diagnostics, "failed_jobs", "jobs"))
    assert len(failed_jobs) == 3
    failed_jobs_by_type = {_value(job, "type"): job for job in failed_jobs}
    assert set(failed_jobs_by_type) == {"image_generation", "chat", "image_cleanup"}
    image_job = failed_jobs_by_type["image_generation"]
    assert _value(image_job, "save_id", "save") == save.id
    assert _value(image_job, "error") == (
        "image provider rejected token=[redacted] during generation"
    )
    assert "image provider rejected" in _value(image_job, "error")
    assert "during generation" in _value(image_job, "error")
    assert _SENTINEL_SECRET not in _value(image_job, "error")
    assert _value(image_job, "api_key", default=None) is None
    assert _value(image_job, "attempt_count") == 3
    assert _value(image_job, "max_attempts") == 3
    assert _value(image_job, "retry_attempts") == (
        {
            "attempt": 1,
            "duration_ms": 10,
            "error_category": "rate_limited",
            "http_status": 429,
        },
        {
            "attempt": 2,
            "duration_ms": 15,
            "error_category": "rate_limited",
            "http_status": 429,
        },
        {
            "attempt": 3,
            "duration_ms": 20,
            "error_category": "rate_limited",
            "http_status": 429,
        },
    )
    bearer_error = _value(failed_jobs_by_type["chat"], "error")
    assert bearer_error == "chat provider rejected Bearer [redacted] during reply"
    assert "chat provider rejected" in bearer_error
    assert "during reply" in bearer_error
    assert _SENTINEL_SECRET not in bearer_error
    cleanup_entry = failed_jobs_by_type["image_cleanup"]
    assert _value(cleanup_entry, "status") == "cancelled"
    assert _value(cleanup_entry, "error") == (
        "user cancelled provider cleanup with token=[redacted]"
    )
    assert _SENTINEL_SECRET not in _value(cleanup_entry, "error")
    assert _SENTINEL_SECRET not in repr(diagnostics)


def test_diagnostics_groups_repeated_maintenance_failures(
    repositories: PersistenceRepositories,
) -> None:
    from bragi.services.diagnostics_service import DiagnosticsService

    save = _save(repositories)
    for attempt in range(3):
        job = repositories.create_job(
            save_id=save.id,
            type="context_update",
            status="running",
            payload={"provider": "venice", "model": "venice/strict"},
        )
        repositories.update_job(
            job.id,
            status="failed",
            error=f"HTTP 400 schema rejected attempt {attempt}",
            result={
                "retry_attempts": [
                    {
                        "attempt": 1,
                        "duration_ms": 10,
                        "error_category": "provider_error",
                        "http_status": 400,
                    }
                ]
            },
        )

    diagnostics = DiagnosticsService(repositories=repositories).list_diagnostics()

    groups = _list(_value(diagnostics, "failed_job_groups", "job_groups"))
    assert len(groups) == 1
    group = groups[0]
    assert _value(group, "save_id", "save") == save.id
    assert _value(group, "type", "job_type") == "context_update"
    assert _value(group, "provider") == "venice"
    assert _value(group, "status") == "failed"
    assert _value(group, "count") == 3
    assert _value(group, "latest_http_status") == 400
    assert "HTTP 400 schema rejected attempt 2" in _value(group, "summary")


def test_diagnostics_runtime_performance_honors_window_and_save_scope(
    repositories: PersistenceRepositories,
) -> None:
    from bragi.services.diagnostics_service import DiagnosticsService

    save = _save(repositories)
    other_scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Other Keep",
        premise="A different watchtower.",
        player_role="Scout",
        content={},
    )
    other_save = repositories.create_save(
        scenario_id=other_scenario.id,
        title="Other Save",
    )
    fast = repositories.update_job(
        repositories.create_job(
            save_id=save.id,
            type="chat_turn",
            status="running",
            payload={"body": "private prompt text"},
        ).id,
        status="succeeded",
    )
    slow = repositories.update_job(
        repositories.create_job(
            save_id=save.id,
            type="chat_turn",
            status="running",
            payload={"body": "private prompt text"},
        ).id,
        status="succeeded",
    )
    failed = repositories.update_job(
        repositories.create_job(
            save_id=save.id,
            type="chat_turn",
            status="running",
            payload={"body": "private prompt text"},
        ).id,
        status="failed",
        error=f"provider failed with token={_SENTINEL_SECRET}",
    )
    other = repositories.update_job(
        repositories.create_job(
            save_id=other_save.id,
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="succeeded",
    )
    for job_id, duration_ms in (
        (fast.id, 100),
        (slow.id, 300),
        (failed.id, 900),
        (other.id, 50),
    ):
        repositories.connection.execute(
            """
            UPDATE jobs
            SET created_at = '2026-06-01 12:00:00',
                started_at = '2026-06-01 12:00:01',
                completed_at = '2026-06-01 12:00:02',
                duration_ms = ?
            WHERE id = ?
            """,
            (duration_ms, job_id),
        )
    repositories.record_job_step(
        job_id=slow.id,
        name="provider.chat",
        status="succeeded",
        provider="fake",
        model="fake-chat",
        task="chat",
        duration_ms=275,
        metadata={"token_total": 42, "prompt": "private prompt text"},
    )

    diagnostics = DiagnosticsService(repositories=repositories).list_diagnostics(
        save_id=save.id,
        since="2026-06-01T12:00:00+00:00",
        limit=10,
    )

    performance = _value(diagnostics, "runtime_performance")
    job_average = _list(_value(performance, "job_averages"))[0]
    assert _value(job_average, "sample_count") == 3
    assert _value(job_average, "success_count") == 2
    assert _value(job_average, "failed_count") == 1
    assert _value(job_average, "p50_duration_ms") == 100
    assert _value(job_average, "p95_duration_ms") == 300
    assert _value(job_average, "failure_rate") == 1 / 3
    slowest = _list(_value(performance, "slowest_recent"))[0]
    assert _value(slowest, "job_id") == failed.id
    assert _value(slowest, "job_type") == "chat_turn"
    assert "private prompt text" not in repr(performance)
    assert _SENTINEL_SECRET not in repr(performance)


def test_diagnostics_reports_generated_text_script_findings_without_content(
    repositories: PersistenceRepositories,
) -> None:
    from bragi.services.diagnostics_service import DiagnosticsService

    save = _save(repositories)
    memory = repositories.add_memory(
        save_id=save.id,
        body="玩家喜欢简洁、扎实的叙事。",
        tags=["tone"],
        importance=0.8,
    )

    diagnostics = DiagnosticsService(repositories=repositories).list_diagnostics()

    findings = _list(_value(diagnostics, "generated_text_script_findings"))
    assert len(findings) == 1
    finding = findings[0]
    assert _value(finding, "save_id") == save.id
    assert _value(finding, "table") == "memories"
    assert _value(finding, "field") == "body"
    assert _value(finding, "script") == "Han"
    assert _value(finding, "count") == 1
    assert _value(finding, "example_record_id") == memory.id
    assert "玩家" not in repr(diagnostics)


def test_diagnostics_report_exposes_resolved_log_file_path(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    from bragi.services.diagnostics_service import DiagnosticsService

    log_file_path = tmp_path / "state" / ".." / "state" / "logs" / "bragi.log"

    diagnostics = DiagnosticsService(
        repositories=repositories,
        log_file_path=log_file_path,
    ).list_diagnostics()

    assert Path(_value(diagnostics, "log_file_path")) == log_file_path.resolve()


def test_diagnostics_report_includes_runtime_performance_averages(
    repositories: PersistenceRepositories,
) -> None:
    from bragi.services.diagnostics_service import DiagnosticsService

    succeeded = repositories.update_job(
        repositories.create_job(
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="succeeded",
    )
    failed = repositories.update_job(
        repositories.create_job(
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="failed",
        error="provider failed",
    )
    repositories.connection.execute(
        "UPDATE jobs SET duration_ms = 100 WHERE id = ?",
        (succeeded.id,),
    )
    repositories.connection.execute(
        "UPDATE jobs SET duration_ms = 800 WHERE id = ?",
        (failed.id,),
    )
    repositories.record_job_step(
        job_id=succeeded.id,
        name="state",
        status="succeeded",
        duration_ms=40,
    )
    repositories.record_job_step(
        job_id=succeeded.id,
        name="state",
        status="skipped",
        duration_ms=400,
    )
    repositories.record_job_step(
        job_id=succeeded.id,
        name="provider.chat",
        status="succeeded",
        provider="fake",
        model="fake-chat",
        task="chat",
        duration_ms=70,
    )

    report = DiagnosticsService(repositories=repositories).list_diagnostics()
    performance = _value(report, "runtime_performance")
    job_row = _list(_value(performance, "job_averages"))[0]
    step_row = next(
        row for row in _list(_value(performance, "step_averages"))
        if _value(row, "step_name") == "state"
    )
    model_row = _list(_value(performance, "model_averages"))[0]

    assert _value(job_row, "job_type") == "chat_turn"
    assert _value(job_row, "success_count") == 1
    assert _value(job_row, "failed_count") == 1
    assert _value(job_row, "average_duration_ms") == 100
    assert _value(step_row, "success_count") == 1
    assert _value(step_row, "skipped_count") == 1
    assert _value(step_row, "average_duration_ms") == 40
    assert _value(model_row, "provider") == "fake"
    assert _value(model_row, "model") == "fake-chat"
    assert _value(model_row, "task") == "chat"
    assert _value(model_row, "average_duration_ms") == 70


def _save(repositories: PersistenceRepositories) -> Any:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    return repositories.create_save(scenario_id=scenario.id, title="Night Watch")


def _list(value: object) -> list[Any]:
    assert isinstance(value, list | tuple), f"Expected sequence, got {value!r}"
    return list(value)


def _value(
    item: object,
    *names: str,
    default: object = _MISSING,
) -> Any:
    for name in names:
        if isinstance(item, Mapping):
            if name in item:
                return item[name]
        elif hasattr(item, name):
            return getattr(item, name)

    if default is not _MISSING:
        return default

    raise AssertionError(f"{item!r} does not expose any of {names!r}")
