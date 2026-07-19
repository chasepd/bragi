"""User authentication and session lifecycle services."""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from bragi.persistence.models import UserRecord, UserSessionRecord
from bragi.persistence.repositories import (
    USER_ROLES,
    USER_STATUSES,
    PersistenceRepositories,
)

DEFAULT_SESSION_LIFETIME = timedelta(days=30)
MIN_LOCAL_PASSWORD_LENGTH = 12


class PasswordHasherProtocol(Protocol):
    def hash(self, password: str) -> str: ...

    def verify(self, password_hash: str, password: str) -> bool: ...

    def check_needs_rehash(self, password_hash: str) -> bool: ...


@dataclass(frozen=True)
class AuthenticatedSession:
    user: UserRecord
    session: UserSessionRecord
    token: str


class AuthServiceError(ValueError):
    """Base exception for user-management policy failures."""


class UnknownUserError(AuthServiceError):
    pass


class LastActiveAdminError(AuthServiceError):
    pass


class CurrentUserDisableError(AuthServiceError):
    pass


class FirstAdminAlreadyExistsError(AuthServiceError):
    pass


class AuthService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        password_hasher: PasswordHasherProtocol | None = None,
        token_factory: Callable[[], str] | None = None,
        session_lifetime: timedelta = DEFAULT_SESSION_LIFETIME,
    ) -> None:
        self.repositories = repositories
        self.password_hasher = password_hasher or PasswordHasher()
        self.token_factory = token_factory or _new_session_token
        self.session_lifetime = session_lifetime

    def create_user(
        self,
        *,
        username: str,
        password: str,
        role: str,
    ) -> UserRecord:
        _validate_new_password(password)
        password_hash = self.password_hasher.hash(password)
        return self.repositories.create_user(
            username=username,
            role=role,
            password_hash=password_hash,
        )

    def bootstrap_first_admin(
        self,
        *,
        username: str,
        password: str,
        now: datetime | None = None,
    ) -> AuthenticatedSession:
        _validate_new_password(password)
        password_hash = self.password_hasher.hash(password)
        created_at = _utc_datetime(now)
        token = self.token_factory()
        token_hash = _session_token_hash(token)
        begin_immediate = getattr(
            self.repositories,
            "begin_immediate_transaction",
            self.repositories.begin_transaction,
        )
        begin_immediate()
        try:
            if _active_admin_count(self.repositories.list_users()) > 0:
                raise FirstAdminAlreadyExistsError(
                    "First admin has already been created"
                )
            user = self.repositories.create_user(
                username=username,
                role="admin",
                password_hash=password_hash,
            )
            self.repositories.claim_unowned_saves(user.id)
            session = self.repositories.create_user_session(
                user_id=user.id,
                token_hash=token_hash,
                expires_at=created_at + self.session_lifetime,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        return AuthenticatedSession(user=user, session=session, token=token)

    def list_users(self) -> list[UserRecord]:
        return self.repositories.list_users()

    def update_user(
        self,
        user_id: str,
        *,
        role: str | None = None,
        status: str | None = None,
        actor_user_id: str | None = None,
    ) -> UserRecord:
        user = self.repositories.get_user(user_id)
        if user is None:
            raise UnknownUserError(f"Unknown user id: {user_id}")
        next_role = user.role if role is None else role
        next_status = user.status if status is None else status
        _validate_role(next_role)
        _validate_status(next_status)
        if self._would_remove_last_active_admin(
            user,
            next_role=next_role,
            next_status=next_status,
        ):
            raise LastActiveAdminError("Cannot remove the last active admin")
        if (
            actor_user_id == user.id
            and user.status == "active"
            and next_status == "disabled"
        ):
            raise CurrentUserDisableError("Cannot disable your current user")

        self.repositories.begin_transaction()
        try:
            updated = user
            if role is not None and role != updated.role:
                updated = self.repositories.update_user_role(user_id, role)
            if status is not None and status != updated.status:
                updated = self.repositories.update_user_status(user_id, status)
            if status == "disabled":
                self.repositories.revoke_user_sessions(user_id)
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        return updated

    def reset_user_password(
        self,
        user_id: str,
        password: str,
        *,
        keep_session_token: str | None = None,
        now: datetime | None = None,
    ) -> UserRecord:
        _validate_new_password(password)
        if self.repositories.get_user(user_id) is None:
            raise UnknownUserError(f"Unknown user id: {user_id}")
        password_hash = self.password_hasher.hash(password)
        keep_token_hash = (
            _session_token_hash(keep_session_token)
            if keep_session_token is not None
            else None
        )
        self.repositories.begin_transaction()
        try:
            user = self.repositories.update_user_password_hash(user_id, password_hash)
            self.repositories.revoke_user_sessions(
                user_id,
                except_token_hash=keep_token_hash,
                now=_utc_datetime(now),
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        return user

    def validate_credentials(
        self,
        username: str,
        password: str,
    ) -> UserRecord | None:
        try:
            user = self.repositories.get_user_by_username(username)
        except ValueError:
            return None
        if user is None or user.status != "active":
            return None
        if not self._password_matches(user.password_hash, password):
            return None
        if self.password_hasher.check_needs_rehash(user.password_hash):
            user = self.repositories.update_user_password_hash(
                user.id,
                self.password_hasher.hash(password),
            )
        return user

    def login(
        self,
        username: str,
        password: str,
        *,
        now: datetime | None = None,
    ) -> AuthenticatedSession | None:
        user = self.validate_credentials(username, password)
        if user is None:
            return None
        created_at = _utc_datetime(now)
        token = self.token_factory()
        session = self.repositories.create_user_session(
            user_id=user.id,
            token_hash=_session_token_hash(token),
            expires_at=created_at + self.session_lifetime,
        )
        return AuthenticatedSession(user=user, session=session, token=token)

    def load_current_user(
        self,
        session_token: str,
        *,
        now: datetime | None = None,
    ) -> UserRecord | None:
        session = self.repositories.get_active_user_session_by_token_hash(
            _session_token_hash(session_token),
            now=_utc_datetime(now),
        )
        if session is None:
            return None
        user = self.repositories.get_user(session.user_id)
        if user is None or user.status != "active":
            return None
        return user

    def revoke_session(self, session_token: str) -> bool:
        session = self.repositories.get_user_session_by_token_hash(
            _session_token_hash(session_token),
        )
        if session is None:
            return False
        return self.repositories.revoke_user_session(session.id)

    def _would_remove_last_active_admin(
        self,
        user: UserRecord,
        *,
        next_role: str,
        next_status: str,
    ) -> bool:
        if user.role != "admin" or user.status != "active":
            return False
        if next_role == "admin" and next_status == "active":
            return False
        return _active_admin_count(self.repositories.list_users()) <= 1

    def _password_matches(self, password_hash: str, password: str) -> bool:
        try:
            return bool(self.password_hasher.verify(password_hash, password))
        except (InvalidHashError, VerificationError):
            return False


def _new_session_token() -> str:
    return secrets.token_urlsafe(32)


def _session_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _validate_role(role: str) -> None:
    if role not in USER_ROLES:
        raise ValueError(f"Unknown user role: {role}")


def _validate_status(status: str) -> None:
    if status not in USER_STATUSES:
        raise ValueError(f"Unknown user status: {status}")


def _validate_new_password(password: str) -> None:
    if not password:
        raise ValueError("password is required")
    if len(password) < MIN_LOCAL_PASSWORD_LENGTH:
        raise ValueError(
            f"password must be at least {MIN_LOCAL_PASSWORD_LENGTH} characters"
        )


def _active_admin_count(users: list[UserRecord]) -> int:
    return sum(1 for user in users if user.role == "admin" and user.status == "active")


def _utc_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
