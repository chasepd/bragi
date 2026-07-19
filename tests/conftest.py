from __future__ import annotations

from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database


@pytest.fixture(scope="session")
def migrated_database_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    template_dir = tmp_path_factory.mktemp("migrated-database-template")
    database_path = template_dir / "bragi.sqlite3"
    migrate_database(database_path)
    return database_path
