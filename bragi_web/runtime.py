"""Application state wiring around the Bragi runtime."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, cast

from bragi_web.auth_throttle import AuthAttemptThrottle
from bragi_web.bragi_adapter import bragi_runtime_bindings
from bragi_web.jobs import JobRecord, JobRegistry
from bragi_web.storage import WebStoragePaths, resolve_web_storage_paths

WEB_SQLITE_BUSY_TIMEOUT_MS = 5000
_WEB_SQLITE_TIMEOUT_SECONDS = WEB_SQLITE_BUSY_TIMEOUT_MS / 1000.0
_WEB_SQLITE_CONFIGURATION_LOCK = threading.Lock()
WEB_RUNTIME_CHRONICLE_MESSAGE_LIMIT = 80


@dataclass
class BundlePreviewState:
    bundle_path: Path
    created_at: float = field(default_factory=time)
    owner_user_id: str | None = None
    target_save_id: str | None = None


class RuntimeAccessLock:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = _runtime_lock_executor()

    def __enter__(self) -> RuntimeAccessLock:
        if _running_in_event_loop():
            if not self._lock.acquire(blocking=False):
                raise RuntimeError(
                    "RuntimeAccessLock cannot block the event loop; use "
                    "async_access() or acquire it from a worker thread"
                )
        else:
            self._lock.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._lock.release()

    def close(self) -> None:
        with self._state_lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)

    @asynccontextmanager
    async def async_access(self) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        acquired: asyncio.Future[None] = loop.create_future()
        release = threading.Event()

        def hold_lock_until_released() -> None:
            with self._lock:
                loop.call_soon_threadsafe(_complete_future, acquired)
                release.wait()

        with self._state_lock:
            if self._executor is None:
                self._executor = _runtime_lock_executor()
            holder = loop.run_in_executor(self._executor, hold_lock_until_released)
        holder.add_done_callback(
            lambda future: _complete_acquisition_if_holder_exited(future, acquired)
        )
        lock_acquired = False
        try:
            await acquired
            lock_acquired = True
            yield
        finally:
            release.set()
            if lock_acquired:
                await asyncio.shield(holder)
            elif holder.done():
                _consume_future_exception(holder)
            else:
                holder.add_done_callback(_consume_future_exception)


def _complete_future(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    try:
        future.exception()
    except asyncio.CancelledError:
        pass


def _complete_acquisition_if_holder_exited(
    holder: asyncio.Future[Any],
    acquired: asyncio.Future[None],
) -> None:
    if acquired.done():
        return
    try:
        holder.result()
    except asyncio.CancelledError:
        acquired.set_exception(
            RuntimeError("RuntimeAccessLock async acquisition was cancelled")
        )
    except Exception as exc:
        acquired.set_exception(exc)
    else:
        acquired.set_exception(
            RuntimeError("RuntimeAccessLock holder exited before acquisition")
        )


def _runtime_lock_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="bragi-runtime-lock",
    )


def _running_in_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def open_web_sqlite_connection(database_path: Path | str) -> sqlite3.Connection:
    connection = sqlite3.connect(
        Path(database_path),
        timeout=_WEB_SQLITE_TIMEOUT_SECONDS,
        check_same_thread=False,
    )
    try:
        _configure_web_sqlite_connection(connection)
    except Exception:
        connection.close()
        raise
    return connection


def _configure_web_sqlite_connection(connection: sqlite3.Connection) -> None:
    with _WEB_SQLITE_CONFIGURATION_LOCK:
        connection.execute(f"PRAGMA busy_timeout = {WEB_SQLITE_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")


class _RepositoryScope:
    def __init__(self, owner: ScopedPersistenceRepositories) -> None:
        self._owner = owner
        self._repositories_by_thread: dict[int, Any] = {}
        self._lock = threading.RLock()
        self._closed = False

    def repository(self) -> Any:
        thread_id = threading.get_ident()
        with self._lock:
            if self._closed:
                raise RuntimeError("Repository scope is closed")
            repository = self._repositories_by_thread.get(thread_id)
            if repository is None:
                repository = self._owner._create_repository()
                self._repositories_by_thread[thread_id] = repository
            return repository

    def close(self) -> None:
        with self._lock:
            self._closed = True
            repositories = tuple(self._repositories_by_thread.values())
            self._repositories_by_thread.clear()
        for repository in repositories:
            _close_repository(repository)


class ScopedPersistenceRepositories:
    def __init__(
        self,
        database_path: Path | str,
        repository_factory: Callable[[sqlite3.Connection], Any],
        *,
        connection_factory: Callable[
            [Path | str],
            sqlite3.Connection,
        ] = open_web_sqlite_connection,
    ) -> None:
        self.database_path = Path(database_path)
        self._repository_factory = repository_factory
        self._connection_factory = connection_factory
        self._scope_var: ContextVar[_RepositoryScope | None] = ContextVar(
            "bragi_web_repository_scope",
            default=None,
        )
        self._default_scope = _RepositoryScope(self)

    @contextmanager
    def scope(self) -> Iterator[ScopedPersistenceRepositories]:
        scope = _RepositoryScope(self)
        token = self._scope_var.set(scope)
        try:
            yield self
        finally:
            self._scope_var.reset(token)
            scope.close()

    def close(self) -> None:
        self._default_scope.close()
        self._default_scope = _RepositoryScope(self)

    @property
    def connection(self) -> sqlite3.Connection:
        return cast(sqlite3.Connection, self.current_repository().connection)

    def current_repository(self) -> Any:
        scope = self._scope_var.get()
        if scope is None:
            scope = self._default_scope
        return scope.repository()

    def _create_repository(self) -> Any:
        return self._repository_factory(self._connection_factory(self.database_path))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current_repository(), name)


def _close_repository(repository: Any) -> None:
    connection = getattr(repository, "connection", None)
    close = getattr(connection, "close", None)
    if callable(close):
        close()


@dataclass(frozen=True)
class SaveEvent:
    event_id: int
    save_id: str | None
    event_type: str
    payload: Any = None
    owner_user_id: str | None = None


class SaveEventHub:
    def __init__(self, *, max_events: int = 1000) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self._max_events = max_events
        self._next_id = 1
        self._events: list[SaveEvent] = []
        self._condition = threading.Condition(threading.RLock())
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = (
            []
        )

    def publish(
        self,
        save_id: str | None,
        event_type: str,
        payload: Any = None,
        *,
        owner_user_id: str | None = None,
    ) -> SaveEvent:
        with self._condition:
            event = SaveEvent(
                event_id=self._next_id,
                save_id=save_id,
                event_type=event_type,
                payload=payload,
                owner_user_id=owner_user_id,
            )
            self._next_id += 1
            self._events.append(event)
            overflow = len(self._events) - self._max_events
            if overflow > 0:
                del self._events[:overflow]
            self._notify_waiters_locked()
            self._condition.notify_all()
            return event

    async def wait_for_event(
        self,
        save_id: str,
        last_event_id: int,
        *,
        owner_user_id: str | None = None,
        include_unowned_global: bool = True,
        include_all_global: bool = False,
    ) -> int:
        while True:
            with self._condition:
                if self._has_event_after_locked(
                    save_id,
                    last_event_id,
                    owner_user_id=owner_user_id,
                    include_unowned_global=include_unowned_global,
                    include_all_global=include_all_global,
                ):
                    return last_event_id
                loop = asyncio.get_running_loop()
                waiter: asyncio.Future[None] = loop.create_future()
                self._waiters.append((loop, waiter))
            try:
                await asyncio.wait_for(waiter, timeout=30)
            except TimeoutError:
                with self._condition:
                    self._remove_waiter_locked(waiter)
                return last_event_id
            except asyncio.CancelledError:
                with self._condition:
                    self._remove_waiter_locked(waiter)
                raise

    def events_after(
        self,
        save_id: str,
        last_event_id: int,
        *,
        owner_user_id: str | None = None,
        include_unowned_global: bool = True,
        include_all_global: bool = False,
    ) -> list[SaveEvent]:
        with self._condition:
            return [
                event
                for event in self._events
                if event.event_id > last_event_id
                and _save_event_visible_to_stream(
                    event,
                    save_id,
                    owner_user_id=owner_user_id,
                    include_unowned_global=include_unowned_global,
                    include_all_global=include_all_global,
                )
            ]

    def latest_event_id(self) -> int:
        with self._condition:
            return self._next_id - 1

    def _has_event_after_locked(
        self,
        save_id: str,
        last_event_id: int,
        *,
        owner_user_id: str | None,
        include_unowned_global: bool,
        include_all_global: bool,
    ) -> bool:
        return any(
            event.event_id > last_event_id
            and _save_event_visible_to_stream(
                event,
                save_id,
                owner_user_id=owner_user_id,
                include_unowned_global=include_unowned_global,
                include_all_global=include_all_global,
            )
            for event in self._events
        )

    def _notify_waiters_locked(self) -> None:
        waiters = self._waiters
        self._waiters = []
        for loop, waiter in waiters:
            if waiter.done():
                continue
            loop.call_soon_threadsafe(_complete_future, waiter)

    def _remove_waiter_locked(self, waiter: asyncio.Future[None]) -> None:
        self._waiters = [
            (loop, future)
            for loop, future in self._waiters
            if future is not waiter
        ]


@dataclass
class WebAppState:
    paths: WebStoragePaths
    repositories: Any
    secret_store: Any
    providers: dict[str, Any]
    runtime: Any
    jobs: JobRegistry = field(default_factory=JobRegistry)
    save_events: SaveEventHub = field(default_factory=SaveEventHub)
    lock: RuntimeAccessLock = field(default_factory=RuntimeAccessLock)
    bundle_previews: dict[str, BundlePreviewState] = field(default_factory=dict)
    scenario_bundle_previews: dict[str, BundlePreviewState] = field(
        default_factory=dict
    )
    character_bundle_previews: dict[str, BundlePreviewState] = field(
        default_factory=dict
    )
    auth_attempts: AuthAttemptThrottle = field(default_factory=AuthAttemptThrottle)
    log_file_path: Path | None = None
    auth_required: bool = True

    def settings_service(self) -> Any:
        return bragi_runtime_bindings().SettingsService(
            repositories=self.repositories,
            providers=self.providers,
            secret_store=self.secret_store,
            log_file_path=self.log_file_path,
        )

    def auth_service(self) -> Any:
        from bragi.services.auth_service import AuthService

        return AuthService(repositories=self.repositories)

    def repository_scope(self) -> Any:
        scope = getattr(self.repositories, "scope", None)
        if callable(scope):
            return scope()
        return nullcontext()

    def close(self) -> None:
        close = getattr(self.repositories, "close", None)
        if callable(close):
            close()
        self.lock.close()


def create_state() -> WebAppState:
    from bragi.services.character_text_world_update_service import (
        CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE,
    )
    from bragi.services.job_lifecycle import JobLifecycleService

    bindings = bragi_runtime_bindings()
    paths = resolve_web_storage_paths()
    for directory in (
        paths.data_dir,
        paths.media_dir,
        paths.state_dir,
        paths.cache_dir,
        paths.temp_dir,
    ):
        bindings.ensure_private_dir(directory)
    log_file_path = bindings.configure_logging(paths)
    bindings.migrate_database(paths.database_path)
    repositories = ScopedPersistenceRepositories(
        paths.database_path,
        bindings.PersistenceRepositories,
    )
    secret_store = bindings.SystemSecretStore(
        service_name="dev.chasepd.Bragi.Web",
        fallback_path=paths.state_dir / "api_keys.json",
    )
    if os.environ.get("BRAGI_WEB_USE_KEYRING") != "1":
        # Web servers are often headless. Prefer the private state-dir fallback
        # unless keyring use is explicitly requested.
        secret_store._use_keyring = False  # noqa: SLF001
    providers = _provider_clients(secret_store, repositories=repositories)
    with repositories.scope():
        JobLifecycleService(
            repositories=cast(Any, repositories),
        ).recover_stale_jobs(
            preserve_queued_types=(
                "state_extraction_retry",
                "context_update_retry",
                CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE,
            ),
        )
        recover_text_deliveries = getattr(
            repositories,
            "recover_interrupted_character_text_deliveries",
            None,
        )
        if callable(recover_text_deliveries):
            recover_text_deliveries(
                error="Text delivery was interrupted before completion",
            )
        _seed_fake_models_if_requested(repositories, providers)
    providers = bindings.wrap_provider_clients_for_telemetry(
        providers,
        repositories=repositories,
    )
    runtime = bindings.BragiRuntime(
        repositories=repositories,
        providers=providers,
        media_dir=paths.media_dir,
        chronicle_message_limit=WEB_RUNTIME_CHRONICLE_MESSAGE_LIMIT,
    )
    save_events = SaveEventHub()
    return WebAppState(
        paths=paths,
        repositories=repositories,
        secret_store=secret_store,
        providers=providers,
        runtime=runtime,
        jobs=JobRegistry(
            repositories=repositories,
            on_change=_publish_job_save_event(save_events),
            repository_scope=repositories.scope,
        ),
        save_events=save_events,
        log_file_path=log_file_path,
    )


def _publish_job_save_event(save_events: SaveEventHub) -> Any:
    def publish(record: JobRecord) -> None:
        if record.save_id is None:
            return
        save_events.publish(
            record.save_id,
            "job_changed",
            {"job": _save_event_job_summary(record)},
            owner_user_id=record.creator_user_id,
        )

    return publish


def _save_event_job_summary(record: JobRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "type": record.type,
        "save_id": record.save_id,
        "status": record.status,
        "error": record.error,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _save_event_visible_to_stream(
    event: SaveEvent,
    save_id: str,
    *,
    owner_user_id: str | None,
    include_unowned_global: bool,
    include_all_global: bool,
) -> bool:
    if event.save_id == save_id:
        return True
    if event.save_id is not None:
        return False
    if include_all_global:
        return True
    if owner_user_id is None:
        return include_unowned_global
    if event.owner_user_id == owner_user_id:
        return True
    return include_unowned_global and event.owner_user_id is None


def _provider_clients(
    secret_store: Any,
    *,
    repositories: Any | None = None,
) -> dict[str, Any]:
    from bragi.retry_policy import configured_max_attempts

    bindings = bragi_runtime_bindings()
    retry_max_attempts = (
        (lambda: configured_max_attempts(repositories))
        if repositories is not None
        else None
    )
    if os.environ.get("BRAGI_WEB_FAKE_PROVIDERS") == "1":
        fake = bindings.FakeProviderClient()
        return {"fake": fake, "openrouter": fake, "venice": fake}
    return {
        "openrouter": bindings.OpenRouterClient(
            secret_store=secret_store,
            retry_max_attempts=retry_max_attempts,
        ),
        "venice": bindings.VeniceClient(
            secret_store=secret_store,
            retry_max_attempts=retry_max_attempts,
        ),
    }


def _seed_fake_models_if_requested(
    repositories: Any,
    providers: dict[str, Any],
) -> None:
    if os.environ.get("BRAGI_WEB_FAKE_PROVIDERS") != "1":
        return
    repositories.upsert_provider_config(
        provider="fake",
        enabled=True,
        has_api_key=True,
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat", "structured_output", "vision"],
        context_window=8192,
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-image",
        display_name="Fake Image",
        capabilities=["image_generation"],
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-edit",
        display_name="Fake Edit",
        capabilities=["image_to_image"],
    )
    settings = bragi_runtime_bindings().build_settings_model(
        repositories=repositories,
        providers=tuple(providers.keys()),
    )
    for task, model_id in _fake_model_preferences_from_settings(settings):
        if repositories.get_model_preference(task) is None:
            repositories.set_model_preference(
                task=task,
                provider="fake",
                model_id=model_id,
            )


def _fake_model_preferences_from_settings(
    settings: object,
) -> tuple[tuple[str, str], ...]:
    preferences: dict[str, str] = {}
    for selector in _settings_selectors(settings):
        task = getattr(selector, "task", None)
        if not isinstance(task, str) or task in preferences:
            continue
        option = next(
            (
                option
                for option in getattr(selector, "options", ())
                if getattr(option, "provider", None) == "fake"
                and getattr(option, "available", False) is True
            ),
            None,
        )
        model_id = getattr(option, "model_id", None)
        if isinstance(model_id, str) and model_id:
            preferences[task] = model_id
    return tuple(preferences.items())


def _settings_selectors(settings: object) -> tuple[object, ...]:
    selectors = list(getattr(settings, "task_model_selectors", ()))
    selectors.extend(getattr(settings, "scenario_section_model_selectors", ()))
    for group in getattr(settings, "roleplay_model_groups", ()):
        selectors.extend(getattr(group, "selectors", ()))
    return tuple(selectors)
