from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi_web.jobs import JobHandle, JobRegistry
from bragi_web.runtime import (
    WEB_SQLITE_BUSY_TIMEOUT_MS,
    ScopedPersistenceRepositories,
    open_web_sqlite_connection,
)


def test_open_web_sqlite_connection_configures_runtime_pragmas(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    connection = open_web_sqlite_connection(database_path)
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert (
            connection.execute("PRAGMA busy_timeout").fetchone()[0]
            == WEB_SQLITE_BUSY_TIMEOUT_MS
        )
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()


def test_scoped_repositories_use_distinct_connections_for_concurrent_scopes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = ScopedPersistenceRepositories(database_path, PersistenceRepositories)
    barrier = Barrier(2)

    def read_with_scoped_connection() -> int:
        with repositories.scope():
            connection_id = id(repositories.connection)
            barrier.wait(timeout=5)
            repositories.list_saves()
            return connection_id

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            connection_ids = list(
                executor.map(lambda _index: read_with_scoped_connection(), range(2))
            )
    finally:
        repositories.close()

    assert len(set(connection_ids)) == 2


def test_nested_repository_scopes_do_not_share_connection_state(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = ScopedPersistenceRepositories(database_path, PersistenceRepositories)

    try:
        with repositories.scope():
            outer_connection = repositories.connection
            repositories.connection.execute(
                "CREATE TEMP TABLE scoped_marker(value TEXT)"
            )

            with repositories.scope():
                assert repositories.connection is not outer_connection
                assert (
                    repositories.connection.execute(
                        """
                        SELECT name
                        FROM sqlite_temp_master
                        WHERE type = 'table' AND name = 'scoped_marker'
                        """
                    ).fetchone()
                    is None
                )
    finally:
        repositories.close()


def test_job_registry_runs_worker_in_fresh_repository_scope(
    tmp_path: Path,
) -> None:
    async def run_test() -> None:
        database_path = tmp_path / "bragi.sqlite3"
        migrate_database(database_path)
        repositories = ScopedPersistenceRepositories(
            database_path,
            PersistenceRepositories,
        )
        registry = JobRegistry(
            repositories=repositories,
            repository_scope=repositories.scope,
        )

        try:
            with repositories.scope():
                request_connection = repositories.connection

                async def worker(handle: JobHandle) -> dict[str, bool]:
                    return {
                        "reused_request_connection": (
                            repositories.connection is request_connection
                        )
                    }

                job = await registry.create("scope_probe", worker)

            assert job.task is not None
            await job.task
            completed = registry.get(job.id)
        finally:
            repositories.close()

        assert completed is not None
        assert completed.result == {"reused_request_connection": False}

    asyncio.run(run_test())
