from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.auth_service import (
    MIN_LOCAL_PASSWORD_LENGTH,
    AuthService,
    CurrentUserDisableError,
    FirstAdminAlreadyExistsError,
    LastActiveAdminError,
    UnknownUserError,
)

VALID_PASSWORD = "correct horse"


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_auth_service_creates_users_with_password_hashes(
    repositories: PersistenceRepositories,
) -> None:
    service = AuthService(repositories=repositories, password_hasher=FakeHasher())

    user = service.create_user(username="Mira", password=VALID_PASSWORD, role="admin")

    assert user.password_hash == f"hash:{VALID_PASSWORD}"
    assert user.role == "admin"
    assert user.status == "active"


def test_auth_service_validates_credentials_and_blocks_disabled_users(
    repositories: PersistenceRepositories,
) -> None:
    service = AuthService(repositories=repositories, password_hasher=FakeHasher())
    user = service.create_user(username="Mira", password=VALID_PASSWORD, role="user")

    assert service.validate_credentials("mira", VALID_PASSWORD) == user
    assert service.validate_credentials("mira", "wrong") is None

    repositories.update_user_status(user.id, "disabled")

    assert service.validate_credentials("mira", VALID_PASSWORD) is None


def test_auth_service_rejects_invalid_stored_password_hash(
    repositories: PersistenceRepositories,
) -> None:
    service = AuthService(repositories=repositories)
    repositories.create_user(
        username="Mira",
        role="user",
        password_hash="not-an-argon2-hash",
    )

    assert service.validate_credentials("Mira", "correct") is None


def test_auth_service_rejects_empty_passwords(
    repositories: PersistenceRepositories,
) -> None:
    service = AuthService(repositories=repositories, password_hasher=FakeHasher())

    with pytest.raises(ValueError, match="password is required"):
        service.create_user(username="Mira", password="", role="user")


def test_auth_service_rejects_short_new_passwords(
    repositories: PersistenceRepositories,
) -> None:
    service = AuthService(repositories=repositories, password_hasher=FakeHasher())
    user = service.create_user(username="Mira", password=VALID_PASSWORD, role="user")

    with pytest.raises(ValueError, match=str(MIN_LOCAL_PASSWORD_LENGTH)):
        service.create_user(username="Rook", password="too short", role="user")

    with pytest.raises(ValueError, match=str(MIN_LOCAL_PASSWORD_LENGTH)):
        service.reset_user_password(user.id, "also bad")


def test_auth_service_updates_users_with_last_admin_protection(
    repositories: PersistenceRepositories,
) -> None:
    service = AuthService(repositories=repositories, password_hasher=FakeHasher())
    admin = service.create_user(username="Mira", password=VALID_PASSWORD, role="admin")

    with pytest.raises(LastActiveAdminError, match="last active admin"):
        service.update_user(admin.id, role="user", actor_user_id=admin.id)

    with pytest.raises(LastActiveAdminError, match="last active admin"):
        service.update_user(admin.id, status="disabled", actor_user_id=admin.id)

    service.create_user(
        username="Rook",
        password=VALID_PASSWORD,
        role="admin",
    )

    with pytest.raises(CurrentUserDisableError, match="current user"):
        service.update_user(admin.id, status="disabled", actor_user_id=admin.id)

    demoted = service.update_user(admin.id, role="user", actor_user_id=admin.id)
    normal_user = service.create_user(
        username="Ilyra",
        password=VALID_PASSWORD,
        role="user",
    )

    assert demoted.role == "user"
    assert service.update_user(normal_user.id, role="child").role == "child"
    assert service.update_user(normal_user.id, status="disabled").status == "disabled"


def test_auth_service_resets_password_and_revokes_other_sessions(
    repositories: PersistenceRepositories,
) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    service = AuthService(
        repositories=repositories,
        password_hasher=FakeHasher(),
        token_factory=lambda: "keep-session-token",
    )
    user = service.create_user(username="Mira", password="old password", role="admin")
    kept = service.login("Mira", "old password", now=now)
    assert kept is not None
    repositories.create_user_session(
        user_id=user.id,
        token_hash="other-token-hash",
        expires_at=now + timedelta(hours=1),
    )

    updated = service.reset_user_password(
        user.id,
        "new password",
        keep_session_token=kept.token,
        now=now,
    )

    assert updated.password_hash == "hash:new password"
    assert service.validate_credentials("Mira", "old password") is None
    assert service.validate_credentials("Mira", "new password") == updated
    assert service.load_current_user(kept.token, now=now) == updated
    assert (
        repositories.get_active_user_session_by_token_hash(
            "other-token-hash",
            now=now,
        )
        is None
    )

    with pytest.raises(UnknownUserError, match="Unknown user"):
        service.reset_user_password("missing-user", "new password")


def test_auth_service_bootstrap_first_admin_creates_session_and_claims_saves(
    repositories: PersistenceRepositories,
) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Warden",
        content={"opening_message": "The beacon snaps awake."},
    )
    legacy_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Legacy Lantern",
        owner_user_id=None,
    )
    service = AuthService(
        repositories=repositories,
        password_hasher=FakeHasher(),
        token_factory=lambda: "bootstrap-session-token",
    )

    login = service.bootstrap_first_admin(
        username="Mira",
        password=VALID_PASSWORD,
        now=now,
    )

    assert login.user.username == "Mira"
    assert login.user.role == "admin"
    assert login.token == "bootstrap-session-token"
    assert login.session.user_id == login.user.id
    claimed_save = repositories.get_save(legacy_save.id)
    assert claimed_save is not None
    assert claimed_save.owner_user_id == login.user.id
    assert service.load_current_user("bootstrap-session-token", now=now) == login.user


def test_auth_service_bootstrap_first_admin_rejects_duplicate_admin(
    repositories: PersistenceRepositories,
) -> None:
    service = AuthService(repositories=repositories, password_hasher=FakeHasher())
    service.bootstrap_first_admin(username="Mira", password=VALID_PASSWORD)

    with pytest.raises(FirstAdminAlreadyExistsError):
        service.bootstrap_first_admin(username="Rook", password=VALID_PASSWORD)

    active_admins = [
        user
        for user in repositories.list_users()
        if user.role == "admin" and user.status == "active"
    ]
    assert [user.username for user in active_admins] == ["Mira"]


def test_auth_service_login_creates_hashed_session_token(
    repositories: PersistenceRepositories,
) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    service = AuthService(
        repositories=repositories,
        password_hasher=FakeHasher(),
        token_factory=lambda: "raw-session-token",
        session_lifetime=timedelta(hours=2),
    )
    user = service.create_user(username="Mira", password=VALID_PASSWORD, role="user")

    login = service.login("Mira", VALID_PASSWORD, now=now)

    assert login is not None
    assert login.user == user
    assert login.token == "raw-session-token"
    assert login.session.token_hash != "raw-session-token"
    assert len(login.session.token_hash) == 64
    assert login.session.expires_at == "2026-01-02T02:00:00+00:00"
    assert service.load_current_user("raw-session-token", now=now) == user


def test_auth_service_rejects_expired_and_revoked_sessions(
    repositories: PersistenceRepositories,
) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    service = AuthService(
        repositories=repositories,
        password_hasher=FakeHasher(),
        token_factory=lambda: "raw-session-token",
        session_lifetime=timedelta(seconds=1),
    )
    service.create_user(username="Mira", password=VALID_PASSWORD, role="user")
    login = service.login("Mira", VALID_PASSWORD, now=now)

    assert login is not None
    assert service.load_current_user("raw-session-token", now=now) is not None
    assert (
        service.load_current_user(
            "raw-session-token",
            now=now + timedelta(seconds=2),
        )
        is None
    )

    assert service.revoke_session("raw-session-token") is True
    assert service.load_current_user("raw-session-token", now=now) is None
    assert service.revoke_session("missing-token") is False


def test_auth_service_rehashes_legacy_password_after_successful_login(
    repositories: PersistenceRepositories,
) -> None:
    service = AuthService(
        repositories=repositories,
        password_hasher=FakeHasher(),
        token_factory=lambda: "raw-session-token",
    )
    repositories.create_user(
        username="Mira",
        role="user",
        password_hash="legacy:correct",
    )

    assert service.login("Mira", "correct") is not None

    user = repositories.get_user_by_username("Mira")
    assert user is not None
    assert user.password_hash == "hash:correct"


def test_auth_service_uses_real_argon2_hasher(
    repositories: PersistenceRepositories,
) -> None:
    service = AuthService(repositories=repositories)
    user = service.create_user(username="Mira", password=VALID_PASSWORD, role="user")

    assert user.password_hash.startswith("$argon2")
    assert service.validate_credentials("Mira", VALID_PASSWORD) == user
    assert service.validate_credentials("Mira", "wrong") is None


class FakeHasher:
    def hash(self, password: str) -> str:
        return f"hash:{password}"

    def verify(self, password_hash: str, password: str) -> bool:
        return password_hash in {f"hash:{password}", f"legacy:{password}"}

    def check_needs_rehash(self, password_hash: str) -> bool:
        return password_hash.startswith("legacy:")
