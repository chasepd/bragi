from __future__ import annotations

import ast
import asyncio
import gzip
import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest
from fastapi.testclient import TestClient as FastAPITestClient
from pytest import MonkeyPatch
from starlette.requests import Request

import bragi_web.api.app as api_app
import bragi_web.runtime as runtime_module
from bragi.application.controller import BragiRuntime
from bragi.interaction_mode import InteractionMode
from bragi.persistence import migrate_database
from bragi.persistence.models import SaveRecord, UserRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ProviderClient,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.safety import CONTENT_FILTER_TRANSITION
from bragi.services.agentic_context import AGENTIC_CONTEXT_PIPELINE_SETTING
from bragi.services.auth_service import AuthService
from bragi.services.character_action_planning_service import (
    CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
    CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
)
from bragi.services.character_profile_completion import ScenarioCharacterStarter
from bragi.services.character_registry_service import (
    CharacterFieldEnhanceResult,
    CharacterRegistryEdits,
    CharacterRegistryRow,
    CharacterRegistryService,
)
from bragi.services.character_text_revision_service import (
    CharacterTextRevisionService,
)
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.model_preferences import SAVE_MODEL_OVERRIDES_SETTING
from bragi.services.prompt_inspection import PromptInspectionStore
from bragi.services.secrets import InMemorySecretStore
from bragi.services.settings_service import SettingsService
from bragi.services.world_data_service import ScenarioEdit
from bragi_web.api.app import create_app
from bragi_web.auth_throttle import AuthAttemptThrottle
from bragi_web.jobs import JobRecord, JobRegistry, JobRegistryLimits
from bragi_web.observability import clear_recent_events, recent_events
from bragi_web.runtime import (
    BundlePreviewState,
    RuntimeAccessLock,
    SaveEventHub,
    ScopedPersistenceRepositories,
    WebAppState,
)

EXPECTED_IMAGE_STYLE_PRESETS = [
    "none",
    "realistic",
    "anime",
    "cartoon",
    "cinematic",
    "concept_art",
    "digital_painting",
    "watercolor",
    "oil_painting",
    "comic_book",
    "colored_pencil",
    "sketch",
    "ink",
    "pixel_art",
    "three_d_render",
    "low_poly",
]
_BRAGI_WRITE_HEADER = "X-Bragi-Api-Request"
_SAVE_ID_REQUIRED_DETAIL = "save_id is required for this save-scoped operation"
SAFE_JOB_ERROR = "Background job failed. Check diagnostics for details."
VALID_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000b49444154789c6360000200000500017a5eab3f"
    "0000000049454e44ae426082"
)


class TestClient(FastAPITestClient):
    def __init__(
        self,
        *args: Any,
        authenticate: bool = True,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        merged_headers = {_BRAGI_WRITE_HEADER: "1"}
        if headers is not None:
            merged_headers.update(headers)
        kwargs["headers"] = merged_headers
        app = args[0] if args else kwargs.get("app")
        super().__init__(*args, **kwargs)
        if authenticate and _app_auth_required(app):
            self._bootstrap_test_admin()

    def _bootstrap_test_admin(self) -> None:
        status = self.get("/api/bootstrap/status")
        if status.status_code != 200:
            return
        if status.json().get("bootstrap_required") is not True:
            return
        response = self.post(
            "/api/bootstrap/admin",
            json={"username": "test-admin", "password": "correct horse battery"},
        )
        assert response.status_code == 200


def _app_auth_required(app: object) -> bool:
    state = getattr(getattr(app, "state", None), "bragi", None)
    return getattr(state, "auth_required", True) is True


def test_health_endpoint_reports_ok(tmp_path: Path) -> None:
    with TestClient(create_app(cast(WebAppState, _state_double(tmp_path)))) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_async_api_routes_do_not_use_blocking_state_lock() -> None:
    tree = ast.parse(Path(api_app.__file__).read_text())
    offenders: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.With):
                for item in child.items:
                    if ast.unparse(item.context_expr) == "state.lock":
                        offenders.append(f"{node.name}:{child.lineno}")
            elif (
                isinstance(child, ast.Call)
                and ast.unparse(child.func) == "_find_media_asset"
            ):
                offenders.append(f"{node.name}:{child.lineno}")

    assert offenders == []


def test_async_api_routes_wait_for_runtime_lock_contention(tmp_path: Path) -> None:
    class ActionChoiceRuntime(_RuntimeDouble):
        async def regenerate_action_choices(
            self,
            *,
            narrator_message_id: str,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            return {
                "active_save_id": active_save_id,
                "narrator_message_id": narrator_message_id,
                "action_choices": [],
            }

    state = _state_double(tmp_path, ActionChoiceRuntime())
    lock_entered = threading.Event()
    release_lock = threading.Event()
    request_finished = threading.Event()
    results: list[tuple[int, dict[str, Any]]] = []

    def hold_runtime_lock() -> None:
        with state.lock:
            lock_entered.set()
            release_lock.wait(1.0)

    def submit_request() -> None:
        with TestClient(
            create_app(cast(WebAppState, state)),
            raise_server_exceptions=False,
        ) as client:
            response = client.post(
                "/api/action-choices/regenerate",
                json={"message_id": "message-1", "save_id": "save-1"},
            )
            payload = response.json()
            if response.status_code == 200:
                payload = _wait_for_terminal_job(
                    client,
                    payload["id"],
                    save_id="save-1",
                )
            results.append((response.status_code, payload))
            request_finished.set()

    lock_thread = threading.Thread(target=hold_runtime_lock)
    request_thread: threading.Thread | None = None
    lock_thread.start()
    try:
        assert lock_entered.wait(1.0)
        request_thread = threading.Thread(target=submit_request)
        request_thread.start()
        try:
            assert not request_finished.wait(0.05)
            release_lock.set()
            request_thread.join(timeout=1.0)
        finally:
            release_lock.set()
            request_thread.join(timeout=1.0)
    finally:
        release_lock.set()
        lock_thread.join(timeout=1.0)

    assert not lock_thread.is_alive()
    assert request_thread is not None
    assert not request_thread.is_alive()
    assert results
    assert results[0][0] == 200
    assert results[0][1]["status"] == "succeeded"


def test_save_engine_health_reports_metadata_only_warnings(tmp_path: Path) -> None:
    state = _state_double(tmp_path)
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(database_path, check_same_thread=False)
    )
    state.repositories = repositories
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="Secret chronicle phrase that must not leak into health payload.",
    )
    repositories.set_app_setting("recent_player_message_window", 24)
    repositories.set_app_setting("recent_narrator_message_window", 24)
    suggestion = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="upsert",
        entity_type="scene_snapshot",
        field_path="situation",
        proposed_value="A stale, private scene description.",
        reason="review",
        source_message_ids=[message.id],
    )
    repositories.connection.execute(
        "UPDATE context_update_suggestions SET created_at = ? WHERE id = ?",
        ("2000-01-01 00:00:00", suggestion.id),
    )
    repositories.commit()
    jobs = JobLifecycleService(repositories=repositories)
    failed = jobs.create_running(
        save_id=save.id,
        type="world_suggestion_review",
        payload={},
    )
    jobs.fail(failed.id, error="review failed")
    empty_search = jobs.create_running(
        save_id=save.id,
        type="context_search",
        payload={"player_message_id": message.id},
    )
    jobs.succeed(
        empty_search.id,
        result={
            "selected_open_obligations": [],
            "selected_scenario_sections": [],
            "selected_state": [],
            "selected_state_changes": [],
            "selected_media_assets": [],
            "selected_memories": [],
            "selected_character_voice": [],
            "selected_summaries": [],
            "selected_recent_messages": [],
        },
    )
    chat = jobs.create_running(
        save_id=save.id,
        type="chat_completion",
        payload={"player_message_id": message.id},
    )
    jobs.succeed(
        chat.id,
        result={
            "prompt_context_diagnostics": {
                "baseline_recent_message_count": 48,
                "baseline_recent_message_chars": 40_000,
            }
        },
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get(f"/api/saves/{save.id}/engine-health")

    assert response.status_code == 200
    payload = response.json()
    warning_codes = {warning["code"] for warning in payload["warnings"]}
    assert {
        "high_recent_message_window",
        "stale_pending_suggestions",
        "failed_continuity_jobs",
        "empty_context_search",
        "large_recent_transcript",
    } <= warning_codes
    assert payload["pending_suggestion_count"] == 1
    assert payload["stale_pending_suggestion_count"] == 1
    assert payload["latest_context_search"]["result_counts"]["selected_memories"] == 0
    assert "Secret chronicle phrase" not in json.dumps(payload)


def test_bootstrap_creates_first_admin_and_unlocks_protected_api(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        bootstrap = client.get("/api/bootstrap/status")
        locked_runtime = client.get("/api/runtime")
        created = client.post(
            "/api/bootstrap/admin",
            json={"username": "Mira", "password": "correct horse"},
        )
        duplicate = client.post(
            "/api/bootstrap/admin",
            json={"username": "Other", "password": "correct horse"},
        )
        me = client.get("/api/auth/me")
        runtime = client.get("/api/runtime")

    assert bootstrap.status_code == 200
    assert bootstrap.json() == {
        "admin_exists": False,
        "bootstrap_required": True,
        "setup_token_required": False,
    }
    assert locked_runtime.status_code == 401
    assert created.status_code == 200
    created_user = created.json()["user"]
    assert created_user["id"]
    assert created_user == {
        "id": created_user["id"],
        "username": "Mira",
        "role": "admin",
        "status": "active",
    }
    assert "bragi_session=" in created.headers["set-cookie"]
    assert "HttpOnly" in created.headers["set-cookie"]
    assert "SameSite=lax" in created.headers["set-cookie"]
    assert duplicate.status_code == 409
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "Mira"
    assert runtime.status_code == 200


def test_auth_session_reports_bootstrap_login_and_current_user(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    app = create_app(cast(WebAppState, state))

    with TestClient(app, authenticate=False) as client:
        before_bootstrap = client.get("/api/auth/session")
        created = client.post(
            "/api/bootstrap/admin",
            json={"username": "Mira", "password": "correct horse"},
        )
        authenticated_session = client.get("/api/auth/session")

    with TestClient(app, authenticate=False) as anonymous_client:
        anonymous_session = anonymous_client.get("/api/auth/session")

    assert before_bootstrap.status_code == 200
    assert before_bootstrap.json() == {
        "bootstrap": {
            "admin_exists": False,
            "bootstrap_required": True,
            "setup_token_required": False,
        },
        "user": None,
    }
    assert created.status_code == 200
    assert authenticated_session.status_code == 200
    assert authenticated_session.json()["bootstrap"] == {
        "admin_exists": True,
        "bootstrap_required": False,
        "setup_token_required": False,
    }
    assert authenticated_session.json()["user"]["username"] == "Mira"
    assert anonymous_session.status_code == 200
    assert anonymous_session.json() == {
        "bootstrap": {
            "admin_exists": True,
            "bootstrap_required": False,
            "setup_token_required": False,
        },
        "user": None,
    }


def test_bootstrap_assigns_existing_unowned_saves_to_first_admin(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    legacy_save = _create_auth_save(
        state.repositories,
        title="Legacy Lantern",
        owner_user_id=None,
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        created = client.post(
            "/api/bootstrap/admin",
            json={"username": "Mira", "password": "correct horse"},
        )
        saves = client.get("/api/saves")

    assert created.status_code == 200
    admin_id = created.json()["user"]["id"]
    assert state.repositories.get_save(legacy_save.id).owner_user_id == admin_id
    assert saves.status_code == 200
    assert [item["save_id"] for item in saves.json()["saves"]] == [legacy_save.id]


def test_auth_requests_reject_oversized_credentials(tmp_path: Path) -> None:
    state = _auth_state(tmp_path)
    state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="admin",
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        oversized_username = client.post(
            "/api/auth/login",
            json={"username": "m" * 129, "password": "correct horse"},
        )
        oversized_password = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "p" * 1025},
        )
        valid_login = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        )

    assert oversized_username.status_code == 422
    assert oversized_password.status_code == 422
    assert valid_login.status_code == 200


def test_remote_bootstrap_rejects_oversized_setup_token(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_BOOTSTRAP_TOKEN", "setup-secret")
    state = _auth_state(tmp_path)

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
        client=("192.168.1.24", 50000),
    ) as client:
        response = client.post(
            "/api/bootstrap/admin",
            json={
                "username": "Mira",
                "password": "correct horse",
                "setup_token": "s" * 257,
            },
        )

    assert response.status_code == 422


def test_auth_attempt_key_bounds_normalized_username() -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/auth/login",
            "headers": [],
            "client": ("192.0.2.10", 50000),
            "server": ("testserver", 80),
        }
    )

    key = api_app._auth_attempt_key("login", request, f"  {'M' * 300}  ")

    assert key == ("login", "192.0.2.10", "m" * 128)


def test_remote_bootstrap_requires_setup_token(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_BOOTSTRAP_TOKEN", "setup-secret")
    state = _auth_state(tmp_path)

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
        client=("192.168.1.24", 50000),
    ) as client:
        status = client.get("/api/bootstrap/status")
        missing = client.post(
            "/api/bootstrap/admin",
            json={"username": "Mira", "password": "correct horse"},
        )
        wrong = client.post(
            "/api/bootstrap/admin",
            json={
                "username": "Mira",
                "password": "correct horse",
                "setup_token": "wrong-secret",
            },
        )
        created = client.post(
            "/api/bootstrap/admin",
            json={
                "username": "Mira",
                "password": "correct horse",
                "setup_token": "setup-secret",
            },
        )

    assert status.status_code == 200
    assert status.json() == {
        "admin_exists": False,
        "bootstrap_required": True,
        "setup_token_required": True,
    }
    assert missing.status_code == 403
    assert wrong.status_code == 403
    assert created.status_code == 200
    assert created.json()["user"]["username"] == "Mira"


def test_bootstrap_requires_setup_token_for_lan_host_via_loopback_proxy(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_BOOTSTRAP_TOKEN", "setup-secret")
    state = _auth_state(tmp_path)

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
        client=("127.0.0.1", 50000),
        headers={"host": "192.168.1.24:8787"},
    ) as client:
        status = client.get("/api/bootstrap/status")
        missing = client.post(
            "/api/bootstrap/admin",
            json={"username": "Mira", "password": "correct horse"},
        )
        created = client.post(
            "/api/bootstrap/admin",
            json={
                "username": "Mira",
                "password": "correct horse",
                "setup_token": "setup-secret",
            },
        )

    assert status.status_code == 200
    assert status.json()["setup_token_required"] is True
    assert missing.status_code == 403
    assert created.status_code == 200


def test_login_attempts_are_throttled_and_reset(
    tmp_path: Path,
) -> None:
    now = [1000.0]
    state = _auth_state(tmp_path)
    state.auth_attempts = AuthAttemptThrottle(clock=lambda: now[0])
    state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="admin",
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        failures = [
            client.post(
                "/api/auth/login",
                json={"username": "Mira", "password": "wrong password"},
            )
            for _ in range(5)
        ]
        throttled = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        )
        now[0] += 601.0
        success = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        )
        after_success_failure = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "wrong password"},
        )

    assert [response.status_code for response in failures] == [401] * 5
    assert throttled.status_code == 429
    assert int(throttled.headers["retry-after"]) > 0
    assert success.status_code == 200
    assert after_success_failure.status_code == 401


def test_remote_bootstrap_attempts_are_throttled_before_success(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    now = [1000.0]
    monkeypatch.setenv("BRAGI_WEB_BOOTSTRAP_TOKEN", "setup-secret")
    state = _auth_state(tmp_path)
    state.auth_attempts = AuthAttemptThrottle(clock=lambda: now[0])

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
        client=("192.168.1.24", 50000),
    ) as client:
        failures = [
            client.post(
                "/api/bootstrap/admin",
                json={
                    "username": "Mira",
                    "password": "correct horse",
                    "setup_token": "wrong-secret",
                },
            )
            for _ in range(5)
        ]
        throttled = client.post(
            "/api/bootstrap/admin",
            json={
                "username": "Mira",
                "password": "correct horse",
                "setup_token": "setup-secret",
            },
        )
        now[0] += 601.0
        created = client.post(
            "/api/bootstrap/admin",
            json={
                "username": "Mira",
                "password": "correct horse",
                "setup_token": "setup-secret",
            },
        )

    assert [response.status_code for response in failures] == [403] * 5
    assert throttled.status_code == 429
    assert created.status_code == 200
    assert created.json()["user"]["username"] == "Mira"


def test_remote_bootstrap_token_failures_are_throttled_across_usernames(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    now = [1000.0]
    monkeypatch.setenv("BRAGI_WEB_BOOTSTRAP_TOKEN", "setup-secret")
    state = _auth_state(tmp_path)
    state.auth_attempts = AuthAttemptThrottle(clock=lambda: now[0])

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
        client=("192.168.1.24", 50000),
    ) as client:
        failures = [
            client.post(
                "/api/bootstrap/admin",
                json={
                    "username": f"Mira{index}",
                    "password": "correct horse",
                    "setup_token": "wrong-secret",
                },
            )
            for index in range(5)
        ]
        throttled = client.post(
            "/api/bootstrap/admin",
            json={
                "username": "FinalAdmin",
                "password": "correct horse",
                "setup_token": "setup-secret",
            },
        )
        now[0] += 601.0
        created = client.post(
            "/api/bootstrap/admin",
            json={
                "username": "FinalAdmin",
                "password": "correct horse",
                "setup_token": "setup-secret",
            },
        )

    assert [response.status_code for response in failures] == [403] * 5
    assert throttled.status_code == 429
    assert created.status_code == 200
    assert created.json()["user"]["username"] == "FinalAdmin"


def test_concurrent_bootstrap_creates_exactly_one_admin_and_claims_saves(
    tmp_path: Path,
) -> None:
    state = _scoped_auth_state(tmp_path)
    legacy_save = _create_auth_save(
        state.repositories,
        title="Legacy Lantern",
        owner_user_id=None,
    )
    hash_barrier = threading.Barrier(2)
    state.auth_service = lambda: AuthService(
        repositories=state.repositories,
        password_hasher=PausingHasher(hash_barrier),
    )
    app = create_app(cast(WebAppState, state))
    results: dict[str, tuple[int, dict[str, Any]]] = {}
    errors: list[BaseException] = []

    def attempt(username: str) -> None:
        try:
            with TestClient(app, authenticate=False) as client:
                response = client.post(
                    "/api/bootstrap/admin",
                    json={"username": username, "password": "correct horse"},
                )
                results[username] = (response.status_code, response.json())
        except BaseException as exc:  # pragma: no cover - surfaced by assertion below
            errors.append(exc)

    threads = [
        threading.Thread(target=attempt, args=("Mira",)),
        threading.Thread(target=attempt, args=("Rook",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    try:
        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert sorted(status for status, _payload in results.values()) == [200, 409]
        active_admins = [
            user
            for user in state.repositories.list_users()
            if user.role == "admin" and user.status == "active"
        ]
        assert len(active_admins) == 1
        claimed_save = state.repositories.get_save(legacy_save.id)
        assert claimed_save is not None
        assert claimed_save.owner_user_id == active_admins[0].id
    finally:
        state.repositories.close()


def test_user_cannot_access_another_users_save_by_direct_id(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    rook = state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    mira_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    rook_save = _create_auth_save(
        state.repositories,
        title="Rook Save",
        owner_user_id=rook.id,
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        login = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        )
        visible = client.get("/api/saves")
        own_runtime = client.get(f"/api/runtime?save_id={mira_save.id}")
        other_runtime = client.get(f"/api/runtime?save_id={rook_save.id}")
        other_settings = client.get(f"/api/settings?save_id={rook_save.id}")
        other_load = client.post(f"/api/saves/{rook_save.id}/load")

    assert login.status_code == 200
    assert visible.status_code == 200
    visible_save = visible.json()["saves"][0]
    assert [item["save_id"] for item in visible.json()["saves"]] == [mira_save.id]
    assert visible_save["scenario_id"] == mira_save.scenario_id
    assert visible_save["scenario_title"] == "Mira Save Scenario"
    assert visible_save["created_at"]
    assert visible_save["updated_at"]
    assert visible_save["last_opened_at"]
    assert own_runtime.status_code == 200
    assert own_runtime.json()["active_save_id"] == mira_save.id
    assert other_runtime.status_code == 404
    assert other_settings.status_code == 404
    assert other_load.status_code == 404


def test_save_list_orders_visible_saves_by_latest_message_activity(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    older_created_stale_save = _create_auth_save(
        state.repositories,
        title="Lantern Save",
        owner_user_id=mira.id,
    )
    newer_created_save = _create_auth_save(
        state.repositories,
        title="Signal Save",
        owner_user_id=mira.id,
    )
    state.repositories.connection.execute(
        """
        UPDATE saves
        SET created_at = ?, updated_at = ?, last_opened_at = ?
        WHERE id = ?
        """,
        (
            "2026-05-01 00:00:00",
            "2000-01-01 00:00:00",
            "2026-05-01 00:00:00",
            older_created_stale_save.id,
        ),
    )
    state.repositories.connection.execute(
        """
        UPDATE saves
        SET created_at = ?, updated_at = ?, last_opened_at = ?
        WHERE id = ?
        """,
        (
            "2026-05-04 00:00:00",
            "2000-01-02 00:00:00",
            "2026-05-10 00:00:00",
            newer_created_save.id,
        ),
    )
    state.repositories.commit()
    state.repositories.append_message(
        save_id=older_created_stale_save.id,
        role="player",
        body="I check the beacon lens.",
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        login = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        )
        saves = client.get("/api/saves")
        runtime = client.get("/api/runtime")

    assert login.status_code == 200
    assert saves.status_code == 200
    assert [item["save_id"] for item in saves.json()["saves"]] == [
        older_created_stale_save.id,
        newer_created_save.id,
    ]
    assert runtime.status_code == 200
    assert runtime.json()["active_save_id"] == older_created_stale_save.id


def test_runtime_and_save_list_redact_restricted_scenario_metadata(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(sqlite3.connect(database_path))
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Restricted Scenario Title",
        premise="Restricted premise.",
        player_role="Restricted role.",
        content={"_source": {"content_rating": "r"}},
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Restricted Save Title",
        custom_instructions="Restricted custom instructions.",
    )
    repositories.set_app_setting("content_filter_rating", "g")
    state = _state_double(tmp_path)
    state.repositories = repositories

    payload = api_app._runtime_json_dict(  # noqa: SLF001
        cast(WebAppState, state),
        {
            "active_save_id": save.id,
            "active_save_title": save.title,
            "scenario_title": scenario.title,
            "scene_title": scenario.title,
            "custom_instructions": "Restricted custom instructions.",
            "saves": [],
            "chronicle": {"messages": []},
        },
    )

    assert payload["active_save_title"] == CONTENT_FILTER_TRANSITION
    assert payload["scenario_title"] == CONTENT_FILTER_TRANSITION
    assert payload["scene_title"] == CONTENT_FILTER_TRANSITION
    assert payload["custom_instructions"] == CONTENT_FILTER_TRANSITION
    assert payload["saves"][0]["title"] == CONTENT_FILTER_TRANSITION
    assert payload["saves"][0]["scenario_title"] == CONTENT_FILTER_TRANSITION


def test_user_with_no_accessible_saves_does_not_fall_back_to_global_save(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    admin_save = _create_auth_save(
        state.repositories,
        title="Admin Save",
        owner_user_id=admin.id,
    )
    state.repositories.append_message(
        save_id=admin_save.id,
        role="narrator",
        body="Admin-only chronicle text.",
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        login = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        )
        runtime = client.get("/api/runtime")
        world = client.get("/api/world-data")
        characters = client.get("/api/characters")

    assert login.status_code == 200
    assert runtime.status_code == 200
    assert runtime.json()["active_save_id"] is None
    assert runtime.json()["saves"] == []
    assert world.status_code == 200
    assert world.json()["active_save_id"] is None
    assert characters.status_code == 200
    assert characters.json()["active_save_id"] is None


def test_active_save_selection_is_per_authenticated_user(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    rook = state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    mira_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    rook_save = _create_auth_save(
        state.repositories,
        title="Rook Save",
        owner_user_id=rook.id,
    )
    app = create_app(cast(WebAppState, state))

    with (
        TestClient(app, authenticate=False) as mira_client,
        TestClient(app, authenticate=False) as rook_client,
    ):
        mira_login = mira_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        )
        rook_login = rook_client.post(
            "/api/auth/login",
            json={"username": "Rook", "password": "correct horse"},
        )
        mira_loaded = mira_client.post(f"/api/saves/{mira_save.id}/load")
        rook_loaded = rook_client.post(f"/api/saves/{rook_save.id}/load")
        mira_runtime = mira_client.get("/api/runtime")
        rook_runtime = rook_client.get("/api/runtime")

    assert mira_login.status_code == 200
    assert rook_login.status_code == 200
    assert mira_loaded.status_code == 200
    assert rook_loaded.status_code == 200
    assert mira_runtime.json()["active_save_id"] == mira_save.id
    assert rook_runtime.json()["active_save_id"] == rook_save.id
    assert state.repositories.get_user_active_save_id(mira.id) == mira_save.id
    assert state.repositories.get_user_active_save_id(rook.id) == rook_save.id


def test_load_save_api_refreshes_last_opened_timestamp(tmp_path: Path) -> None:
    class RuntimeDouble(_RuntimeDouble):
        def build_model(
            self,
            *,
            active_save_id: str | None | object = ...,
            status: str | None = None,
        ) -> dict[str, object]:
            save_id = (
                self.active_save_id
                if active_save_id is ...
                else cast(str | None, active_save_id)
            )
            model = _chat_model("The bell answers.")
            model["active_save_id"] = save_id
            model["status"] = status
            return model

    state = _auth_state(tmp_path, RuntimeDouble())
    state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    state.repositories.connection.execute(
        """
        UPDATE saves
        SET updated_at = ?, last_opened_at = ?
        WHERE id = ?
        """,
        ("2026-05-03 00:00:00", "2000-01-01 00:00:00", save.id),
    )
    state.repositories.commit()

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        loaded = client.post(f"/api/saves/{save.id}/load")

    assert loaded.status_code == 200
    persisted = state.repositories.get_save(save.id)
    assert persisted is not None
    assert persisted.updated_at == "2026-05-03 00:00:00"
    assert persisted.last_opened_at is not None
    assert persisted.last_opened_at > "2000-01-01 00:00:00"
    [save_item] = loaded.json()["saves"]
    assert save_item["save_id"] == save.id
    assert save_item["last_opened_at"] == persisted.last_opened_at


def test_rename_save_api_updates_title_without_switching_active_selection(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    first_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    second_save = _create_auth_save(
        state.repositories,
        title="Signal Tower",
        owner_user_id=mira.id,
    )
    state.repositories.set_user_active_save_id(
        user_id=mira.id,
        save_id=first_save.id,
    )

    class RenameRuntime(_RuntimeDouble):
        def __init__(self, repositories: PersistenceRepositories) -> None:
            super().__init__()
            self.repositories = repositories
            self.rename_calls: list[tuple[str, str, object]] = []

        def rename_save(
            self,
            *,
            save_id: str,
            title: str,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            self.rename_calls.append((save_id, title, active_save_id))
            save = self.repositories.update_save_title(
                save_id=save_id,
                title=title,
            )
            selected_save_id = (
                self.active_save_id
                if active_save_id is ...
                else cast(str | None, active_save_id)
            )
            active = (
                self.repositories.get_save(selected_save_id)
                if selected_save_id is not None
                else None
            )
            model = _chat_model("The bell answers.")
            model["active_save_id"] = active.id if active else None
            model["active_save_title"] = active.title if active else None
            model["status"] = f"Renamed save: {save.title}"
            return model

    runtime = RenameRuntime(state.repositories)
    state.runtime = runtime

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200

        renamed = client.post(
            f"/api/saves/{second_save.id}/rename",
            json={"title": "  Signal Tower Revised  "},
        )
        blank = client.post(
            f"/api/saves/{second_save.id}/rename",
            json={"title": "   "},
        )

    assert renamed.status_code == 200
    payload = renamed.json()
    save_items = {item["save_id"]: item for item in payload["saves"]}
    assert payload["active_save_id"] == first_save.id
    assert payload["active_save_title"] == "Mira Save"
    assert save_items[second_save.id]["title"] == "Signal Tower Revised"
    assert save_items[second_save.id]["active"] is False
    assert state.repositories.get_save(second_save.id).title == (
        "Signal Tower Revised"
    )
    assert runtime.rename_calls == [
        (second_save.id, "Signal Tower Revised", first_save.id),
    ]
    assert blank.status_code == 400
    assert blank.json() == {"detail": "Save title is required"}

    renamed_save_events = state.save_events.events_after(
        second_save.id,
        0,
        owner_user_id=mira.id,
        include_unowned_global=False,
    )
    assert [
        (event.save_id, event.event_type, event.payload)
        for event in renamed_save_events
    ] == [
        (second_save.id, "runtime_changed", {"reason": "save_renamed"}),
        (None, "saves_changed", {"reason": "save_renamed"}),
    ]
    active_save_events = state.save_events.events_after(
        first_save.id,
        0,
        owner_user_id=mira.id,
        include_unowned_global=False,
    )
    assert [
        (event.save_id, event.event_type, event.payload)
        for event in active_save_events
    ] == [(None, "saves_changed", {"reason": "save_renamed"})]


def test_child_role_can_read_chat_and_generate_media_but_cannot_mutate_save(
    tmp_path: Path,
) -> None:
    class ChildRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.chat_bodies: list[str] = []
            self.chat_user_ids: list[str | None] = []
            self.export_calls: list[str | None] = []
            self.guidance_calls: list[str] = []
            self.media_calls: list[str] = []
            self.media_user_ids: list[str | None] = []
            self.deleted_saves: list[str] = []
            self.rename_calls: list[tuple[str, str]] = []

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
            current_user_id: str | None = None,
        ) -> object:
            self.chat_bodies.append(body)
            self.chat_user_ids.append(current_user_id)
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=False,
                save_id=active_save_id if isinstance(active_save_id, str) else None,
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

        def export_active_save(
            self,
            bundle_path: Path,
            *,
            active_save_id: str | None = None,
            include_message_revisions: bool = False,
        ) -> SimpleNamespace:
            self.export_calls.append(active_save_id)
            bundle_path.write_bytes(b"save-bundle")
            return SimpleNamespace(error=None)

        def update_custom_instructions(
            self,
            *,
            custom_instructions: str,
            active_save_id: str | None,
        ) -> dict[str, object]:
            self.guidance_calls.append(custom_instructions)
            return {"active_save_id": active_save_id, "error": None}

        async def generate_image(
            self,
            *,
            source_message_id: str,
            active_save_id: str | None | object = ...,
            retry_progress_callback: Callable[[object], None] | None = None,
            current_user_id: str | None = None,
        ) -> dict[str, object]:
            self.media_calls.append(source_message_id)
            self.media_user_ids.append(current_user_id)
            return _chat_model("The image should not be generated.")

        async def generate_character_image(self, **kwargs: object) -> dict[str, object]:
            self.media_calls.append(str(kwargs["source_message_id"]))
            self.media_user_ids.append(cast(str | None, kwargs.get("current_user_id")))
            return _chat_model("Character image generated.")

        async def generate_character_registry_image(
            self,
            _character_id: str,
            **kwargs: object,
        ) -> dict[str, object]:
            self.media_calls.append("registry")
            self.media_user_ids.append(cast(str | None, kwargs.get("current_user_id")))
            return _chat_model("Registry image generated.")

        async def generate_character_reference_image(
            self,
            _character_id: str,
            **kwargs: object,
        ) -> dict[str, object]:
            self.media_calls.append("reference")
            self.media_user_ids.append(cast(str | None, kwargs.get("current_user_id")))
            return _chat_model("Reference image generated.")

        def delete_save(self, save_id: str) -> dict[str, object]:
            self.deleted_saves.append(save_id)
            return {"active_save_id": None, "saves": [], "status": "Deleted"}

        def rename_save(
            self,
            *,
            save_id: str,
            title: str,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            self.rename_calls.append((save_id, title))
            return {"active_save_id": active_save_id, "error": None}

    runtime = ChildRuntime()
    state = _auth_state(tmp_path, runtime)
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    assigned_save = _create_auth_save(
        state.repositories,
        title="Assigned Save",
        owner_user_id=admin.id,
    )
    child_owned_save = _create_auth_save(
        state.repositories,
        title="Legacy Child Save",
        owner_user_id=child.id,
    )
    state.repositories.grant_save_access(save_id=assigned_save.id, user_id=child.id)
    initial_reference_character = state.repositories.add_character(
        save_id=assigned_save.id,
        name="Mara",
        character_id="initial-reference-character",
    )
    existing_reference_character = state.repositories.add_character(
        save_id=assigned_save.id,
        name="Rook",
        character_id="existing-reference-character",
    )
    existing_reference = state.repositories.create_media_asset(
        save_id=assigned_save.id,
        source_message_id=None,
        type="image",
        path=f"{assigned_save.id}/existing-reference.png",
        prompt="A quiet portrait at the harbor.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    state.repositories.add_entity_link(
        save_id=assigned_save.id,
        entity_type="character",
        entity_id=existing_reference_character.id,
        target_type="media_asset",
        target_id=existing_reference.id,
        relation="reference_image",
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        readable = client.get(f"/api/runtime?save_id={assigned_save.id}")
        owned_readable = client.get(f"/api/runtime?save_id={child_owned_save.id}")
        visible_saves = client.get("/api/saves")
        engine_health = client.get(
            f"/api/saves/{assigned_save.id}/engine-health"
        )
        submitted = client.post(
            "/api/chat",
            json={"body": "I check the beacon.", "save_id": assigned_save.id},
        )
        assert submitted.status_code == 200
        chat_job = _wait_for_terminal_job(
            client,
            submitted.json()["id"],
            save_id=assigned_save.id,
        )
        exported = client.get(f"/api/bundles/export?save_id={assigned_save.id}")
        guidance = client.post(
            "/api/runtime/custom-instructions",
            json={
                "save_id": assigned_save.id,
                "custom_instructions": "Keep it cozy.",
            },
        )
        media = client.post(
            "/api/media/generate",
            json={"message_id": "message-1", "save_id": assigned_save.id},
        )
        character_media = client.post(
            "/api/media/generate-character-image",
            json={
                "message_id": "message-1",
                "character_id": "character-1",
                "save_id": assigned_save.id,
            },
        )
        registry_image = client.post(
            "/api/characters/character-1/image/generate",
            json={"save_id": assigned_save.id, "instructions": "moonlight"},
        )
        initial_reference_image = client.post(
            f"/api/characters/{initial_reference_character.id}/reference-image/generate",
            json={"save_id": assigned_save.id, "replace_existing": False},
        )
        reference_replacement = client.post(
            f"/api/characters/{existing_reference_character.id}/reference-image/generate",
            json={"save_id": assigned_save.id, "replace_existing": True},
        )
        reference_regeneration = client.post(
            f"/api/media/{existing_reference.id}/regenerate",
            json={
                "save_id": assigned_save.id,
                "prompt": "A quiet portrait beneath the beacon.",
            },
        )
        media_job = _wait_for_terminal_job(
            client,
            media.json()["id"],
            save_id=assigned_save.id,
        )
        character_media_job = _wait_for_terminal_job(
            client,
            character_media.json()["id"],
            save_id=assigned_save.id,
        )
        registry_image_job = _wait_for_terminal_job(
            client,
            registry_image.json()["id"],
            save_id=assigned_save.id,
        )
        initial_reference_image_job = _wait_for_terminal_job(
            client,
            initial_reference_image.json()["id"],
            save_id=assigned_save.id,
        )
        presence_edit = client.post(
            "/api/messages/message-1/scene-presence",
            json={"save_id": assigned_save.id, "character_ids": []},
        )
        suggestion_review = client.post(
            "/api/world-data/suggestion-review",
            json={"save_id": assigned_save.id},
        )
        context_retention = client.post(
            "/api/world-data/context-retention",
            json={"save_id": assigned_save.id},
        )
        summary_backfill = client.post(
            "/api/world-data/summary-backfill",
            json={"save_id": assigned_save.id},
        )
        renamed = client.post(
            f"/api/saves/{assigned_save.id}/rename",
            json={"title": "Child Rename"},
        )
        deleted = client.delete(f"/api/saves/{child_owned_save.id}")

    assert readable.status_code == 200
    assert readable.json()["active_save_id"] == assigned_save.id
    assert owned_readable.status_code == 200
    assert owned_readable.json()["active_save_id"] == child_owned_save.id
    assert visible_saves.status_code == 200
    assert {item["save_id"] for item in visible_saves.json()["saves"]} == {
        assigned_save.id,
        child_owned_save.id,
    }
    assert engine_health.status_code == 403
    assert chat_job["status"] == "succeeded"
    assert runtime.chat_bodies == ["I check the beacon."]
    assert runtime.chat_user_ids == [child.id]
    assert media_job["status"] == "succeeded"
    assert character_media_job["status"] == "succeeded"
    assert registry_image_job["status"] == "succeeded"
    assert initial_reference_image_job["status"] == "succeeded"
    assert reference_replacement.status_code == 403
    assert reference_regeneration.status_code == 403
    for response in (
        exported,
        guidance,
        presence_edit,
        suggestion_review,
        context_retention,
        summary_backfill,
        renamed,
        deleted,
    ):
        assert response.status_code == 403
    assert runtime.export_calls == []
    assert runtime.guidance_calls == []
    assert runtime.media_calls == ["message-1", "message-1", "registry", "reference"]
    assert runtime.media_user_ids == [child.id, child.id, child.id, child.id]
    assert runtime.rename_calls == []
    assert runtime.deleted_saves == []


def test_child_read_filters_existing_adult_chronicle_and_media(
    tmp_path: Path,
) -> None:
    graphic_body = (
        "He chopped off the prisoner's head and limbs, spraying the walls red."
    )
    explicit_media = {
        "id": "runtime-explicit-media",
        "source_message_id": "narrator-1",
        "source_media_asset_id": None,
        "type": "image",
        "path": "runtime-explicit.png",
        "thumbnail_path": None,
        "mime_type": "image/png",
        "prompt": graphic_body,
        "provider": "fake",
        "model": "fake-image",
        "status": "succeeded",
        "source_message": graphic_body,
        "metadata": {"content_rating": "r"},
    }

    class ExistingAdultContentRuntime(_RuntimeDouble):
        def build_model(
            self,
            *,
            active_save_id: str | None | object = ...,
            status: str | None = None,
        ) -> dict[str, object]:
            model = _chat_model(graphic_body)
            save_id = self.active_save_id if active_save_id is ... else active_save_id
            model["active_save_id"] = save_id
            message = cast(
                list[dict[str, Any]],
                cast(dict[str, Any], model["chronicle"])["messages"],
            )[0]
            message["markdown_blocks"] = [
                {"kind": "paragraph", "text": graphic_body}
            ]
            message["content_rating"] = "r"
            model["media"] = {
                "latest_scene_media": dict(explicit_media),
                "latest_scene_image": dict(explicit_media),
                "image_history": [dict(explicit_media)],
                "media_history": [dict(explicit_media)],
                "character_reference_image": None,
            }
            return model

    runtime = ExistingAdultContentRuntime()
    state = _auth_state(tmp_path, runtime)
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    save = _create_auth_save(
        state.repositories,
        title="Assigned Save",
        owner_user_id=admin.id,
    )
    runtime.active_save_id = save.id
    state.repositories.grant_save_access(save_id=save.id, user_id=child.id)
    message = state.repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=graphic_body,
        content_rating="r",
    )
    media_path = state.paths.media_dir / save.id / "explicit.png"
    thumbnail_path = state.paths.media_dir / save.id / "explicit-thumb.png"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(VALID_PNG_BYTES)
    thumbnail_path.write_bytes(VALID_PNG_BYTES)
    asset = state.repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path=f"{save.id}/explicit.png",
        thumbnail_path=f"{save.id}/explicit-thumb.png",
        prompt=graphic_body,
        provider="fake",
        model="fake-image",
        status="succeeded",
        mime_type="image/png",
        metadata={"content_rating": "r"},
    )
    character = state.repositories.add_character(
        save_id=save.id,
        name="Mara",
    )
    reference_asset = state.repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path=f"{save.id}/reference.png",
        prompt=graphic_body,
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={
            "kind": "character_reference",
            "character_id": character.id,
            "content_rating": "r",
        },
    )
    generated_asset = state.repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path=f"{save.id}/generated.png",
        prompt=graphic_body,
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={
            "kind": "character_image",
            "character_id": character.id,
            "content_rating": "r",
        },
    )
    state.repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="media_asset",
        target_id=reference_asset.id,
        relation="reference_image",
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        runtime_response = client.get(f"/api/runtime?save_id={save.id}")
        media_response = client.get(f"/api/saves/{save.id}/media")
        characters_response = client.get(f"/api/characters?save_id={save.id}")
        original_response = client.get(f"/api/media/{asset.id}?save_id={save.id}")
        thumbnail_response = client.get(
            f"/api/media/{asset.id}/thumbnail?save_id={save.id}"
        )

    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    assert runtime_payload["chronicle"]["messages"][0]["body"] == (
        CONTENT_FILTER_TRANSITION
    )
    assert graphic_body not in json.dumps(runtime_payload)
    assert runtime_payload["media"]["latest_scene_media"] is None
    assert runtime_payload["media"]["latest_scene_image"] is None
    assert runtime_payload["media"]["media_history"] == []
    assert runtime_payload["media"]["image_history"] == []
    assert media_response.status_code == 200
    assert media_response.json()["media_history"] == []
    assert characters_response.status_code == 200
    [character_payload] = characters_response.json()["characters"]
    assert character_payload["reference_image"] is None
    assert character_payload["generated_images"] == []
    assert reference_asset.id not in json.dumps(characters_response.json())
    assert generated_asset.id not in json.dumps(characters_response.json())
    assert original_response.status_code == 403
    assert thumbnail_response.status_code == 403


def test_child_read_filters_character_text_preview_and_attachment(
    tmp_path: Path,
) -> None:
    graphic_body = (
        "He chopped off the prisoner's head and limbs, spraying the walls red."
    )
    state = _auth_state(tmp_path)
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=admin.id,
    )
    state.repositories.grant_save_access(save_id=save_id, user_id=child.id)
    player = next(
        character
        for character in state.repositories.list_characters(save_id)
        if character.is_player_character
    )
    npc = next(
        character
        for character in state.repositories.list_characters(save_id)
        if not character.is_player_character
    )
    state.repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    thread = state.repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    message = state.repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body=graphic_body,
        content_rating="r",
    )
    media_path = state.paths.media_dir / save_id / "text-attachment.png"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(VALID_PNG_BYTES)
    media_asset = state.repositories.create_media_asset(
        save_id=save_id,
        source_message_id=None,
        type="image",
        path=f"{save_id}/text-attachment.png",
        prompt=graphic_body,
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={
            "kind": "character_text_object_context_image",
            "content_rating": "g",
            "text_message_id": message.id,
            "character_id": npc.id,
        },
    )
    state.repositories.add_character_text_message_attachment(
        save_id=save_id,
        thread_id=thread.id,
        text_message_id=message.id,
        character_id=npc.id,
        kind="object_context_image",
        status="succeeded",
        media_asset_id=media_asset.id,
        prompt=graphic_body,
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        contacts_response = client.get(f"/api/character-texts?save_id={save_id}")
        thread_response = client.get(
            f"/api/character-texts/threads/{thread.id}?save_id={save_id}"
        )
        media_response = client.get(
            f"/api/media/{media_asset.id}?save_id={save_id}"
        )

    assert contacts_response.status_code == 200
    contact = next(
        item for item in contacts_response.json()["contacts"] if item["id"] == npc.id
    )
    assert contact["latest_message_body"] == CONTENT_FILTER_TRANSITION
    assert graphic_body not in json.dumps(contact)
    assert thread_response.status_code == 200
    [thread_message] = thread_response.json()["messages"]
    assert thread_message["body"] == CONTENT_FILTER_TRANSITION
    assert thread_message["attachments"] == []
    assert media_asset.id not in json.dumps(thread_response.json())
    assert media_response.status_code == 403
    assert media_response.json() == {
        "detail": "Media exceeds your content rating",
    }


def test_character_text_contact_update_repairs_manual_contact_state(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    player = next(
        character
        for character in state.repositories.list_characters(save_id)
        if character.is_player_character
    )
    npc = next(
        character
        for character in state.repositories.list_characters(save_id)
        if not character.is_player_character
    )
    state.repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            f"/api/character-texts/contacts/{npc.id}",
            json={
                "save_id": save_id,
                "player_has_character_number": False,
                "character_has_player_number": True,
            },
        )
        player_response = client.post(
            f"/api/character-texts/contacts/{player.id}",
            json={
                "save_id": save_id,
                "player_has_character_number": True,
                "character_has_player_number": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    visible_contact = next(
        item for item in payload["contacts"] if item["id"] == npc.id
    )
    assert visible_contact["player_has_character_number"] is False
    assert visible_contact["character_has_player_number"] is True
    contact = next(
        item for item in payload["repair_contacts"] if item["id"] == npc.id
    )
    assert contact["player_has_character_number"] is False
    assert contact["character_has_player_number"] is True
    assert contact["player_number_permission"] == {
        "allowed": False,
        "source": "none",
        "reason": "You do not have this character's number.",
        "source_message_id": None,
        "source_text_message_id": None,
    }
    assert contact["character_number_permission"]["allowed"] is True
    assert contact["character_number_permission"]["source"] == "manual_or_legacy"
    assert "manually" in contact["character_number_permission"]["reason"].casefold()
    state_record = state.repositories.get_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
    )
    assert state_record is not None
    assert state_record.player_has_character_number is False
    assert state_record.character_has_player_number is True
    assert player_response.status_code == 404
    assert player_response.json()["detail"] == (
        f"Unknown textable character id: {player.id}"
    )
    events = state.save_events.events_after(
        save_id,
        0,
        owner_user_id=user.id,
        include_unowned_global=False,
    )
    assert [
        (event.save_id, event.event_type)
        for event in events
    ] == [(save_id, "character_texts_changed")]


def test_child_role_can_update_character_text_contacts_on_assigned_save(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=admin.id,
    )
    state.repositories.grant_save_access(save_id=save_id, user_id=child.id)
    npc = next(
        character
        for character in state.repositories.list_characters(save_id)
        if not character.is_player_character
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            f"/api/character-texts/contacts/{npc.id}",
            json={
                "save_id": save_id,
                "player_has_character_number": True,
                "character_has_player_number": False,
            },
        )

    assert response.status_code == 200
    contact = next(item for item in response.json()["contacts"] if item["id"] == npc.id)
    assert contact["player_has_character_number"] is True
    assert contact["character_has_player_number"] is False
    repair_contact = next(
        item for item in response.json()["repair_contacts"] if item["id"] == npc.id
    )
    assert repair_contact["player_has_character_number"] is True
    assert repair_contact["character_has_player_number"] is False


def test_child_role_cannot_upload_character_text_photo_on_assigned_save(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=admin.id,
    )
    state.repositories.grant_save_access(save_id=save_id, user_id=child.id)
    player = next(
        character
        for character in state.repositories.list_characters(save_id)
        if character.is_player_character
    )
    npc = next(
        character
        for character in state.repositories.list_characters(save_id)
        if not character.is_player_character
    )
    state.repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            "/api/character-texts/send-image",
            data={
                "save_id": save_id,
                "character_id": npc.id,
                "body": "What is this?",
            },
            files={"file": ("family-photo.png", VALID_PNG_BYTES, "image/png")},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Save action is not allowed"
    assert state.repositories.list_character_text_messages(save_id=save_id) == []


def test_character_text_send_is_not_blocked_by_active_chat_turn(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    player = next(
        character
        for character in state.repositories.list_characters(save_id)
        if character.is_player_character
    )
    npc = next(
        character
        for character in state.repositories.list_characters(save_id)
        if not character.is_player_character
    )
    state.repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    state.repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    state.repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Mira waits beside the arcade prize counter.",
        objective="Choose whether to text Rowan.",
        in_world_time="Friday evening after class",
        mood="soft competitive tension",
        present_character_ids=[npc.id],
    )
    provider = _BlockingCharacterTextProvider()
    state.providers = {"fake": provider}
    state.jobs._jobs = {  # noqa: SLF001 - active-job regression fixture
        "active-chat": JobRecord(
            id="active-chat",
            type="chat_turn",
            save_id=save_id,
            status="running",
        )
    }

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            "/api/character-texts/send",
            json={
                "save_id": save_id,
                "character_id": npc.id,
                "body": "Can we talk after class?",
            },
        )
        assert response.status_code == 200
        job_id = response.json()["id"]
        assert provider.wait_for_entered(1)
        pending = state.repositories.list_character_text_messages(save_id=save_id)
        assert [(message.sender, message.delivery_status) for message in pending] == [
            ("player", "pending")
        ]

        provider.release()
        job = _wait_for_terminal_job(client, job_id, save_id=save_id)
        thread_id = state.repositories.list_character_text_threads(save_id)[0].id
        thread_response = client.get(
            f"/api/character-texts/threads/{thread_id}?save_id={save_id}",
        )

    assert job["status"] == "succeeded"
    assert thread_response.status_code == 200
    api_messages = thread_response.json()["messages"]
    assert api_messages[0]["in_world_sent_at"] == "Friday evening after class"
    assert api_messages[0]["delivered_at"] is not None
    assert api_messages[1]["reply_to_message_id"] == api_messages[0]["id"]
    assert api_messages[1]["delivered_at"] is not None
    messages = state.repositories.list_character_text_messages(save_id=save_id)
    assert [(message.sender, message.delivery_status) for message in messages] == [
        ("player", "sent"),
        ("character", "sent"),
    ]


def test_character_text_spontaneous_send_queues_character_message(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    player = next(
        character
        for character in state.repositories.list_characters(save_id)
        if character.is_player_character
    )
    npc = next(
        character
        for character in state.repositories.list_characters(save_id)
        if not character.is_player_character
    )
    state.repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    state.repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    provider = _BlockingCharacterTextProvider()
    state.providers = {"fake": provider}

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            "/api/character-texts/spontaneous",
            json={
                "save_id": save_id,
                "character_id": npc.id,
            },
        )
        assert response.status_code == 200
        assert response.json()["type"] == "character_text_spontaneous"
        job_id = response.json()["id"]
        pending = state.repositories.list_character_text_messages(save_id=save_id)
        assert [
            (message.sender, message.body, message.delivery_status)
            for message in pending
        ] == [("character", "", "pending")]
        assert provider.wait_for_entered(1)

        blocked_send = client.post(
            "/api/character-texts/send",
            json={
                "save_id": save_id,
                "character_id": npc.id,
                "body": "Are you there?",
            },
        )
        assert blocked_send.status_code == 409

        provider.release()
        job = _wait_for_terminal_job(client, job_id, save_id=save_id)

    assert job["status"] == "succeeded"
    messages = state.repositories.list_character_text_messages(save_id=save_id)
    assert [
        (message.id, message.sender, message.body, message.delivery_status)
        for message in messages
    ] == [
        (
            pending[0].id,
            "character",
            "Meet me by the arcade after class.",
            "sent",
        ),
    ]
    events = state.save_events.events_after(save_id, 0)
    text_events = [
        event
        for event in events
        if event.event_type == "character_texts_changed"
    ]
    assert len(text_events) >= 3
    assert all(event.save_id == save_id for event in text_events)


def test_character_text_send_captures_prompt_inspection_when_debug_enabled(
    tmp_path: Path,
) -> None:
    class TextProvider:
        async def chat(self, request: object) -> SimpleNamespace:
            return SimpleNamespace(
                body="Meet me by the arcade after class.",
                provider=getattr(request, "provider", "fake"),
                model_id=getattr(request, "model_id", "fake-chat"),
                token_usage={"total": 12},
                raw_request_payload={"messages": ["character text prompt"]},
            )

    state = _auth_state(tmp_path)
    state.runtime.prompt_inspection_store = PromptInspectionStore()
    state.repositories.set_app_setting("debug_logging_enabled", True)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    player = next(
        character
        for character in state.repositories.list_characters(save_id)
        if character.is_player_character
    )
    npc = next(
        character
        for character in state.repositories.list_characters(save_id)
        if not character.is_player_character
    )
    state.repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    state.repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    state.providers = {"fake": TextProvider()}

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            "/api/character-texts/send",
            json={
                "save_id": save_id,
                "character_id": npc.id,
                "body": "Can we talk after class?",
            },
        )
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id=save_id)

    assert job["status"] == "succeeded"
    reply = next(
        message
        for message in state.repositories.list_character_text_messages(save_id=save_id)
        if message.sender == "character"
    )
    prompt_text = state.runtime.prompt_inspection_store.prompt_for_message(reply.id)
    assert prompt_text is not None
    assert "Character text prompt" in prompt_text
    assert "Can we talk after class?" in prompt_text
    assert (
        state.runtime.prompt_inspection_store.provider_payload_for_message(reply.id)
        is not None
    )


def test_character_text_sends_in_different_threads_run_concurrently(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    player = next(
        character
        for character in state.repositories.list_characters(save_id)
        if character.is_player_character
    )
    first_npc = next(
        character
        for character in state.repositories.list_characters(save_id)
        if not character.is_player_character
    )
    second_npc = state.repositories.add_character(
        save_id=save_id,
        name="Toma",
        met=True,
    )
    state.repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    for npc in (first_npc, second_npc):
        state.repositories.upsert_character_contact_state(
            save_id=save_id,
            player_character_id=player.id,
            character_id=npc.id,
            player_has_character_number=True,
            character_has_player_number=True,
        )
    provider = _BlockingCharacterTextProvider()
    state.providers = {"fake": provider}

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        first = client.post(
            "/api/character-texts/send",
            json={
                "save_id": save_id,
                "character_id": first_npc.id,
                "body": "Can we talk after class?",
            },
        )
        second = client.post(
            "/api/character-texts/send",
            json={
                "save_id": save_id,
                "character_id": second_npc.id,
                "body": "Are you free later?",
            },
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert provider.wait_for_entered(2)

        provider.release()
        first_job = _wait_for_terminal_job(
            client,
            first.json()["id"],
            save_id=save_id,
        )
        second_job = _wait_for_terminal_job(
            client,
            second.json()["id"],
            save_id=save_id,
        )

    assert first_job["status"] == "succeeded"
    assert second_job["status"] == "succeeded"


def test_character_text_thread_exposes_sender_specific_edit_actions(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    _player_message_id, _reply_id, thread_id = _seed_character_text_exchange(
        state.repositories,
        save_id,
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.get(
            f"/api/character-texts/threads/{thread_id}?save_id={save_id}",
        )

    assert response.status_code == 200
    messages = response.json()["messages"]
    assert _action_ids(messages[0]) == {
        "delete-text-messages-from-here",
        "edit-text-message",
        "edit-and-resubmit-text-message",
    }
    assert _action_ids(messages[1]) == {
        "correct-character-text-message",
        "delete-text-messages-from-here",
    }


def test_child_character_text_thread_hides_mutation_actions(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=admin.id,
    )
    state.repositories.grant_save_access(save_id=save_id, user_id=child.id)
    player_message_id, _reply_id, thread_id = _seed_character_text_exchange(
        state.repositories,
        save_id,
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        response = client.get(
            f"/api/character-texts/threads/{thread_id}?save_id={save_id}",
        )
        edit = client.post(
            "/api/character-texts/message-edit",
            json={
                "save_id": save_id,
                "text_message_id": player_message_id,
                "body": "Can we talk after class?",
            },
        )
        resubmit = client.post(
            "/api/character-texts/edit",
            json={
                "save_id": save_id,
                "text_message_id": player_message_id,
                "body": "Can we talk after class?",
            },
        )
        delete = client.post(
            "/api/character-texts/delete-from-here",
            json={
                "save_id": save_id,
                "text_message_id": player_message_id,
            },
        )

    assert response.status_code == 200
    assert [_action_ids(message) for message in response.json()["messages"]] == [
        set(),
        set(),
    ]
    assert edit.status_code == 403
    assert resubmit.status_code == 403
    assert delete.status_code == 403


def test_character_text_thread_read_endpoint_persists_read_state(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    _player_message_id, reply_id, thread_id = _seed_character_text_exchange(
        state.repositories,
        save_id,
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            f"/api/character-texts/threads/{thread_id}/read",
            json={
                "save_id": save_id,
                "through_message_id": reply_id,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["save_id"] == save_id
    assert payload["updated_message_ids"] == [reply_id]
    assert payload["thread"]["id"] == thread_id
    messages = payload["thread"]["messages"]
    assert messages[0]["sender"] == "player"
    assert messages[0]["read_at"] is None
    assert messages[1]["sender"] == "character"
    assert messages[1]["read_at"] is not None
    persisted = state.repositories.get_character_text_message(
        save_id=save_id,
        message_id=reply_id,
    )
    assert persisted is not None
    assert persisted.read_at is not None
    events = state.save_events.events_after(
        save_id,
        0,
        owner_user_id=user.id,
        include_unowned_global=False,
    )
    assert [
        (event.save_id, event.event_type)
        for event in events
    ] == [(save_id, "character_texts_changed")]


def test_child_can_mark_character_text_thread_read_on_assigned_save(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=admin.id,
    )
    state.repositories.grant_save_access(save_id=save_id, user_id=child.id)
    _player_message_id, reply_id, thread_id = _seed_character_text_exchange(
        state.repositories,
        save_id,
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            f"/api/character-texts/threads/{thread_id}/read",
            json={
                "save_id": save_id,
                "through_message_id": reply_id,
            },
        )

    assert response.status_code == 200
    assert response.json()["updated_message_ids"] == [reply_id]


def test_character_text_thread_read_rejects_other_save_thread(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    other_save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    _player_message_id, reply_id, thread_id = _seed_character_text_exchange(
        state.repositories,
        other_save_id,
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            f"/api/character-texts/threads/{thread_id}/read",
            json={
                "save_id": save_id,
                "through_message_id": reply_id,
            },
        )

    assert response.status_code == 404


def test_character_text_message_edit_updates_without_provider_request(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    player_message_id, _reply_id, thread_id = _seed_character_text_exchange(
        state.repositories,
        save_id,
    )
    provider = _RecordingCharacterTextProvider()
    state.providers = {"fake": provider}

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            "/api/character-texts/message-edit",
            json={
                "save_id": save_id,
                "text_message_id": player_message_id,
                "body": "Can we talk after class?",
            },
        )
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id=save_id)
        thread = client.get(
            f"/api/character-texts/threads/{thread_id}?save_id={save_id}",
        ).json()

    messages = state.repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=thread_id,
    )
    assert job["status"] == "succeeded"
    assert provider.requests == []
    assert [(message.sender, message.body) for message in messages] == [
        ("player", "Can we talk after class?"),
        ("character", "Sure, meet me by the lockers."),
    ]
    assert thread["messages"][0]["revision_count"] == 1


def test_character_text_edit_and_resubmit_replays_thread_reply(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    player_message_id, old_reply_id, thread_id = _seed_character_text_exchange(
        state.repositories,
        save_id,
    )
    provider = _RecordingCharacterTextProvider("Then meet me at the arcade.")
    state.providers = {"fake": provider}
    seen_user_ids: list[str | None] = []
    original = CharacterTextRevisionService.edit_text_and_resubmit

    async def recording_resubmit(
        service: CharacterTextRevisionService,
        *,
        save_id: str,
        text_message_id: str,
        body: str,
        current_user_id: str | None = None,
    ) -> object:
        seen_user_ids.append(current_user_id)
        return await original(
            service,
            save_id=save_id,
            text_message_id=text_message_id,
            body=body,
            current_user_id=current_user_id,
        )

    monkeypatch.setattr(
        CharacterTextRevisionService,
        "edit_text_and_resubmit",
        recording_resubmit,
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            "/api/character-texts/edit",
            json={
                "save_id": save_id,
                "text_message_id": player_message_id,
                "body": "Can we meet at the arcade?",
            },
        )
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id=save_id)

    messages = state.repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=thread_id,
    )
    assert job["status"] == "succeeded"
    assert old_reply_id not in {message.id for message in messages}
    assert [(message.sender, message.body) for message in messages] == [
        ("player", "Can we meet at the arcade?"),
        ("character", "Then meet me at the arcade."),
    ]
    assert len(provider.requests) == 1
    assert seen_user_ids == [user.id]


def test_character_text_unchanged_resubmit_rejects_before_job_creation(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    player_message_id, _reply_id, _thread_id = _seed_character_text_exchange(
        state.repositories,
        save_id,
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            "/api/character-texts/edit",
            json={
                "save_id": save_id,
                "text_message_id": player_message_id,
                "body": "  Can we tak after class?  ",
            },
        )
        active_jobs = client.get(f"/api/jobs?status=active&save_id={save_id}")

    assert response.status_code == 400
    assert response.json()["detail"] == "Text message was not changed"
    assert active_jobs.status_code == 200
    assert active_jobs.json()["jobs"] == []


def test_character_text_delete_from_here_archives_thread_suffix(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save_id = _create_dating_auth_save(
        state.repositories,
        owner_user_id=user.id,
    )
    player_message_id, reply_id, thread_id = _seed_character_text_exchange(
        state.repositories,
        save_id,
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        response = client.post(
            "/api/character-texts/delete-from-here",
            json={
                "save_id": save_id,
                "text_message_id": player_message_id,
            },
        )
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id=save_id)
        thread = client.get(
            f"/api/character-texts/threads/{thread_id}?save_id={save_id}",
        ).json()

    assert job["status"] == "succeeded"
    assert job["result"]["deleted_message_ids"] == [player_message_id, reply_id]
    assert job["result"]["deleted_count"] == 2
    assert thread["messages"] == []
    assert state.repositories.list_character_text_messages(
        save_id=save_id,
        thread_id=thread_id,
    ) == []


def test_child_role_blocks_unsafe_direct_routes_but_hides_unrelated_saves(
    tmp_path: Path,
) -> None:
    class GuardedRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.timeskips: list[str] = []
            self.chat_cancels: list[str | None] = []
            self.deleted_scenarios: list[str] = []
            self.exported_scenarios: list[str] = []

        async def submit_timeskip_for_initial_render(
            self,
            *,
            instruction: str,
            active_save_id: object,
        ) -> object:
            self.timeskips.append(instruction)
            return SimpleNamespace(
                model=_chat_model("Dawn catches on the city gates."),
                has_post_turn_jobs=False,
                save_id=active_save_id if isinstance(active_save_id, str) else None,
                player_message_id="timeskip-1",
                narrator_message_id="narrator-1",
            )

        def cancel_active_submit(self, *, save_id: str | None = None) -> bool:
            self.chat_cancels.append(save_id)
            return True

        def delete_saved_scenario(self, scenario_id: str) -> dict[str, object]:
            self.deleted_scenarios.append(scenario_id)
            return {"status": "Deleted"}

        def export_saved_scenario(
            self,
            scenario_id: str,
            bundle_path: Path,
        ) -> SimpleNamespace:
            self.exported_scenarios.append(scenario_id)
            bundle_path.write_bytes(b"scenario-bundle")
            return SimpleNamespace(error=None)

    class VeniceProvider:
        def __init__(self) -> None:
            self.searches: list[str] = []

        def list_characters(
            self,
            *,
            search: str,
            limit: int,
            offset: int,
        ) -> list[dict[str, str]]:
            self.searches.append(search)
            return [{"slug": "mara", "name": "Mara"}]

    runtime = GuardedRuntime()
    venice = VeniceProvider()
    state = _auth_state(tmp_path, runtime)
    state.providers = {"venice": venice}
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    other = state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    assigned_save = _create_auth_save(
        state.repositories,
        title="Assigned Save",
        owner_user_id=admin.id,
    )
    unrelated_save = _create_auth_save(
        state.repositories,
        title="Other Save",
        owner_user_id=other.id,
    )
    state.repositories.grant_save_access(save_id=assigned_save.id, user_id=child.id)
    state.jobs._jobs = {  # noqa: SLF001 - controlled auth regression fixture
        "admin-chat": JobRecord(
            id="admin-chat",
            type="chat_turn",
            save_id=assigned_save.id,
            creator_user_id=admin.id,
            status="running",
        )
    }
    scenario_id = assigned_save.scenario_id

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200

        unrelated_chat = client.post(
            "/api/chat",
            json={"body": "I peek.", "save_id": unrelated_save.id},
        )
        timeskip = client.post(
            "/api/chat/timeskip",
            json={
                "instruction": "Skip to dawn.",
                "save_id": assigned_save.id,
            },
        )
        chat_cancel = client.post(
            "/api/chat/cancel",
            json={"save_id": assigned_save.id},
        )
        job_cancel = client.post(
            f"/api/jobs/admin-chat/cancel?save_id={assigned_save.id}",
        )
        scenario_definition = client.post(
            f"/api/scenarios/{scenario_id}/definition",
            json={"edit": {"title": "Nope"}},
        )
        scenario_delete = client.delete(f"/api/scenarios/{scenario_id}")
        scenario_export = client.get(f"/api/scenario-bundles/export/{scenario_id}")
        venice_search = client.post(
            "/api/scenarios/venice/search",
            json={"search": "mara"},
        )
        venice_import = client.post(
            "/api/scenarios/venice/import",
            json={"slug": "mara"},
        )

    assert unrelated_chat.status_code == 404
    assert timeskip.status_code == 403
    assert chat_cancel.status_code == 403
    assert job_cancel.status_code == 403
    assert scenario_definition.status_code == 403
    assert scenario_delete.status_code == 403
    assert scenario_export.status_code == 403
    assert venice_search.status_code == 404
    assert venice_import.status_code == 404
    assert runtime.timeskips == []
    assert runtime.chat_cancels == []
    assert runtime.deleted_scenarios == []
    assert runtime.exported_scenarios == []
    assert venice.searches == []


def test_shared_scenario_mutations_require_admin(tmp_path: Path) -> None:
    class ScenarioRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.deleted: list[str] = []

        def delete_saved_scenario(self, scenario_id: str) -> dict[str, object]:
            self.deleted.append(scenario_id)
            return {"status": "Deleted"}

    runtime = ScenarioRuntime()
    state = _auth_state(tmp_path, runtime)
    service = state.auth_service()
    admin = service.create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    service.create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    scenario = state.repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A beacon is going dark.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    app = create_app(cast(WebAppState, state))

    with (
        TestClient(app, authenticate=False) as user_client,
        TestClient(app, authenticate=False) as admin_client,
    ):
        assert user_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        user_definition = user_client.post(
            f"/api/scenarios/{scenario.id}/definition",
            json={"edit": {"title": "User Edit"}},
        )
        user_delete = user_client.delete(f"/api/scenarios/{scenario.id}")

        assert admin_client.post(
            "/api/auth/login",
            json={"username": "Admin", "password": "correct horse"},
        ).status_code == 200
        admin_delete = admin_client.delete(f"/api/scenarios/{scenario.id}")

    assert admin.role == "admin"
    assert user_definition.status_code == 403
    assert user_delete.status_code == 403
    assert admin_delete.status_code == 200
    assert runtime.deleted == [scenario.id]


def test_runtime_debug_details_are_visible_only_to_admin(
    tmp_path: Path,
) -> None:
    class DebugRuntime(_RuntimeDouble):
        def build_model(
            self,
            *,
            active_save_id: str | None | object = ...,
            status: str | None = None,
        ) -> dict[str, object]:
            save_id = (
                self.active_save_id
                if active_save_id is ...
                else cast(str | None, active_save_id)
            )
            model = _debug_runtime_model(save_id or "save-1")
            model["status"] = status
            return model

    state = _auth_state(tmp_path, DebugRuntime())
    service = state.auth_service()
    admin = service.create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    user = service.create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    child = service.create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    admin_save = _create_auth_save(
        state.repositories,
        title="Admin Save",
        owner_user_id=admin.id,
    )
    user_save = _create_auth_save(
        state.repositories,
        title="User Save",
        owner_user_id=user.id,
    )
    child_save = _create_auth_save(
        state.repositories,
        title="Child Save",
        owner_user_id=child.id,
    )
    app = create_app(cast(WebAppState, state))

    with (
        TestClient(app, authenticate=False) as admin_client,
        TestClient(app, authenticate=False) as user_client,
        TestClient(app, authenticate=False) as child_client,
    ):
        assert admin_client.post(
            "/api/auth/login",
            json={"username": "Admin", "password": "correct horse"},
        ).status_code == 200
        assert user_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        assert child_client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200

        admin_runtime = admin_client.get(f"/api/runtime?save_id={admin_save.id}")
        user_runtime = user_client.get(f"/api/runtime?save_id={user_save.id}")
        child_runtime = child_client.get(f"/api/runtime?save_id={child_save.id}")

    assert admin_runtime.status_code == 200
    assert user_runtime.status_code == 200
    assert child_runtime.status_code == 200

    admin_message = admin_runtime.json()["chronicle"]["messages"][0]
    assert admin_message["debug_prompt"] == "secret prompt"
    assert admin_message["debug_provider_payload"] == {"messages": ["secret"]}
    assert {
        action["action_id"] for action in admin_message["actions"]
    } >= {"inspect-debug-prompt", "inspect-provider-payload"}

    user_message = user_runtime.json()["chronicle"]["messages"][0]
    assert "debug_prompt" not in user_message
    assert "debug_provider_payload" not in user_message
    assert "inspect-debug-prompt" not in _action_ids(user_message)
    assert "inspect-provider-payload" not in _action_ids(user_message)
    assert "regenerate-message" in _action_ids(user_message)

    child_message = child_runtime.json()["chronicle"]["messages"][0]
    assert "debug_prompt" not in child_message
    assert "debug_provider_payload" not in child_message
    assert _action_ids(child_message) == {
        "generate-character-image",
        "generate-scene-image",
    }


def test_job_runtime_results_and_events_scrub_debug_details_for_non_admin(
    tmp_path: Path,
) -> None:
    class DebugChatRuntime(_RuntimeDouble):
        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
        ) -> object:
            return SimpleNamespace(
                model=_debug_runtime_model(
                    active_save_id if isinstance(active_save_id, str) else "save-1"
                ),
                has_post_turn_jobs=False,
                save_id=active_save_id if isinstance(active_save_id, str) else None,
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

    state = _auth_state(tmp_path, DebugChatRuntime())
    service = state.auth_service()
    service.create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    user = service.create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save = _create_auth_save(
        state.repositories,
        title="User Save",
        owner_user_id=user.id,
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon.", "save_id": save.id},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        job = _wait_for_terminal_job(client, job_id, save_id=save.id)
        events = client.get(f"/api/jobs/{job_id}/events?save_id={save.id}")

    assert job["status"] == "succeeded"
    message = job["result"]["chronicle"]["messages"][0]
    assert "debug_prompt" not in message
    assert "debug_provider_payload" not in message
    assert "inspect-debug-prompt" not in _action_ids(message)
    assert "inspect-provider-payload" not in _action_ids(message)
    assert "regenerate-message" in _action_ids(message)
    assert events.status_code == 200
    assert "secret prompt" not in events.text
    assert "debug_provider_payload" not in events.text
    assert "inspect-debug-prompt" not in events.text
    assert "regenerate-message" in events.text


def test_job_save_events_omit_result_payloads() -> None:
    save_events = SaveEventHub()
    publish = runtime_module._publish_job_save_event(save_events)  # noqa: SLF001
    publish(
        JobRecord(
            id="job-debug",
            type="chat_turn",
            save_id="save-1",
            creator_user_id="user-1",
            status="succeeded",
            result={"debug_prompt": "secret prompt"},
        )
    )

    events = save_events.events_after(
        "save-1",
        0,
        owner_user_id="user-1",
        include_unowned_global=False,
    )

    assert len(events) == 1
    job = events[0].payload["job"]
    assert job["id"] == "job-debug"
    assert job["status"] == "succeeded"
    assert "result" not in job
    assert "debug_prompt" not in str(events[0].payload)


def test_character_bundle_export_requires_character_save_access(
    tmp_path: Path,
) -> None:
    class CharacterExportRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.exported: list[str] = []

        def export_character_bundle(
            self,
            character_id: str,
            bundle_path: Path,
            *,
            include_private_notes: bool = False,
        ) -> SimpleNamespace:
            self.exported.append(character_id)
            bundle_path.write_bytes(b"character-bundle")
            return SimpleNamespace(error=None)

    runtime = CharacterExportRuntime()
    state = _auth_state(tmp_path, runtime)
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    mira_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    character = state.repositories.add_character(
        save_id=mira_save.id,
        name="Mara",
        role="Signal runner",
    )

    app = create_app(cast(WebAppState, state))
    with (
        TestClient(app, authenticate=False) as mira_client,
        TestClient(app, authenticate=False) as rook_client,
    ):
        assert mira_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        assert rook_client.post(
            "/api/auth/login",
            json={"username": "Rook", "password": "correct horse"},
        ).status_code == 200

        blocked = rook_client.get(f"/api/character-bundles/export/{character.id}")
        allowed = mira_client.get(f"/api/character-bundles/export/{character.id}")

    assert blocked.status_code == 404
    assert allowed.status_code == 200
    assert allowed.content == b"character-bundle"
    assert runtime.exported == [character.id]


def test_character_bundle_private_notes_export_requires_admin(
    tmp_path: Path,
) -> None:
    class CharacterExportRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.exports: list[tuple[str, bool]] = []

        def export_character_bundle(
            self,
            character_id: str,
            bundle_path: Path,
            *,
            include_private_notes: bool = False,
        ) -> SimpleNamespace:
            self.exports.append((character_id, include_private_notes))
            bundle_path.write_bytes(b"character-bundle")
            return SimpleNamespace(error=None)

    runtime = CharacterExportRuntime()
    state = _auth_state(tmp_path, runtime)
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    owner = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    shared = state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    owner_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=owner.id,
    )
    state.repositories.grant_save_access(save_id=owner_save.id, user_id=shared.id)
    state.repositories.grant_save_access(save_id=owner_save.id, user_id=child.id)
    character = state.repositories.add_character(
        save_id=owner_save.id,
        name="Mara",
        role="Signal runner",
        private_notes="Keep the lens secret.",
    )

    app = create_app(cast(WebAppState, state))
    with (
        TestClient(app, authenticate=False) as admin_client,
        TestClient(app, authenticate=False) as owner_client,
        TestClient(app, authenticate=False) as shared_client,
        TestClient(app, authenticate=False) as child_client,
    ):
        assert admin_client.post(
            "/api/auth/login",
            json={"username": admin.username, "password": "correct horse"},
        ).status_code == 200
        assert owner_client.post(
            "/api/auth/login",
            json={"username": owner.username, "password": "correct horse"},
        ).status_code == 200
        assert shared_client.post(
            "/api/auth/login",
            json={"username": shared.username, "password": "correct horse"},
        ).status_code == 200
        assert child_client.post(
            "/api/auth/login",
            json={"username": child.username, "password": "correct horse"},
        ).status_code == 200

        admin_private = admin_client.get(
            f"/api/character-bundles/export/{character.id}"
            "?include_private_notes=true"
        )
        owner_private = owner_client.get(
            f"/api/character-bundles/export/{character.id}"
            "?include_private_notes=true"
        )
        shared_private = shared_client.get(
            f"/api/character-bundles/export/{character.id}"
            "?include_private_notes=true"
        )
        child_private = child_client.get(
            f"/api/character-bundles/export/{character.id}"
            "?include_private_notes=true"
        )
        owner_public = owner_client.get(
            f"/api/character-bundles/export/{character.id}"
        )

    assert admin_private.status_code == 200
    assert admin_private.content == b"character-bundle"
    assert owner_private.status_code == 403
    assert shared_private.status_code == 403
    assert child_private.status_code == 403
    assert owner_public.status_code == 200
    assert owner_public.content == b"character-bundle"
    assert runtime.exports == [(character.id, True), (character.id, False)]


def test_global_jobs_are_scoped_to_creating_user(tmp_path: Path) -> None:
    state = _auth_state(tmp_path)
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    rook = state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    state.jobs._jobs = {  # noqa: SLF001 - controlled auth regression fixture
        "mira-global": JobRecord(
            id="mira-global",
            type="model_refresh",
            status="running",
            creator_user_id=mira.id,
        ),
        "rook-global": JobRecord(
            id="rook-global",
            type="scenario_draft",
            status="succeeded",
            result={"ok": True},
            creator_user_id=rook.id,
        ),
    }

    app = create_app(cast(WebAppState, state))
    with TestClient(app, authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200

        jobs = client.get("/api/jobs")
        blocked_read = client.get("/api/jobs/rook-global")
        blocked_cancel = client.post("/api/jobs/rook-global/cancel")
        blocked_events = client.get("/api/jobs/rook-global/events")

    assert jobs.status_code == 200
    assert [job["id"] for job in jobs.json()["jobs"]] == ["mira-global"]
    assert blocked_read.status_code == 404
    assert blocked_cancel.status_code == 404
    assert blocked_events.status_code == 404


def test_save_event_hub_filters_global_events_by_owner(tmp_path: Path) -> None:
    state = _auth_state(tmp_path)
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    rook = state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    state.save_events.publish(
        None,
        "saves_changed",
        {"reason": "mira"},
        owner_user_id=mira.id,
    )
    state.save_events.publish(
        None,
        "saves_changed",
        {"reason": "rook"},
        owner_user_id=rook.id,
    )
    state.save_events.publish("save-1", "runtime_changed", {"reason": "chat"})

    events = state.save_events.events_after(
        "save-1",
        0,
        owner_user_id=mira.id,
        include_unowned_global=False,
    )

    assert [(event.save_id, event.event_type, event.payload) for event in events] == [
        (None, "saves_changed", {"reason": "mira"}),
        ("save-1", "runtime_changed", {"reason": "chat"}),
    ]


def test_authenticated_save_creation_does_not_mutate_process_active_save(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    app = create_app()
    state = cast(WebAppState, app.state.bragi)

    with TestClient(app) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower at the edge of a storm sea.",
                "player_role": "Keeper",
                "opening_message": "The beacon snaps awake.",
            },
        )
        runtime = client.get("/api/runtime")

    assert created.status_code == 200
    created_save_id = created.json()["active_save_id"]
    assert created_save_id
    assert runtime.status_code == 200
    assert runtime.json()["active_save_id"] == created_save_id
    assert state.runtime.active_save_id is None


def test_scenarios_api_exposes_library_metadata(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    app = create_app()

    with TestClient(app) as client:
        save_id = _create_manual_lantern_save(client)
        listed = client.get("/api/scenarios")

    assert save_id
    assert listed.status_code == 200
    scenario = listed.json()["scenarios"][0]
    assert scenario["title"] == "Lantern Keep"
    assert scenario["save_count"] == 1
    assert scenario["created_at"]
    assert scenario["updated_at"]
    assert scenario["supported"] is True
    assert scenario["unsupported_reason"] is None


def test_scenarios_api_tolerates_legacy_scenario_type(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    app = create_app()
    state = cast(WebAppState, app.state.bragi)
    state.repositories.create_scenario(
        type="legacy_roleplay",
        title="Old Harbor",
        premise="An older scenario template from a previous version.",
        player_role="Keeper",
        content={
            "opening_message": "The old harbor bell rings.",
            "_source": {
                "content_rating": "g",
                "section_content_ratings": {"opening_message": "g"},
            },
        },
    )

    with TestClient(app) as client:
        listed = client.get("/api/scenarios")

    assert listed.status_code == 200
    scenario = listed.json()["scenarios"][0]
    assert scenario["title"] == "Old Harbor"
    assert scenario["scenario_type"] == "legacy_roleplay"
    assert scenario["scenario_types"] == ["legacy_roleplay"]
    assert scenario["supported"] is True
    assert scenario["unsupported_reason"] is None


def test_retired_character_interaction_records_are_recovery_only(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    app = create_app()
    state = cast(WebAppState, app.state.bragi)
    scenario = state.repositories.create_scenario(
        type="character_interaction",
        title="Archived Conversation",
        premise="Historical recovery data.",
        player_role="Player",
        content={
            "opening_message": "A preserved greeting.",
            "_source": {
                "content_rating": "g",
                "section_content_ratings": {"opening_message": "g"},
            },
        },
    )
    save = state.repositories.create_save(
        scenario_id=scenario.id,
        title="Archived Conversation",
    )
    state.jobs._jobs["retired-job"] = JobRecord(  # noqa: SLF001
        id="retired-job",
        type="chat_turn",
        save_id=save.id,
        status="running",
    )

    with TestClient(app) as client:
        [admin] = [
            user
            for user in state.repositories.list_users()
            if user.role == "admin" and user.status == "active"
        ]
        staged_character_bundle = tmp_path / "retired-character.bragi-character"
        staged_character_bundle.write_bytes(b"staged character bundle")
        state.character_bundle_previews["retired-preview"] = BundlePreviewState(
            bundle_path=staged_character_bundle,
            owner_user_id=admin.id,
            target_save_id=save.id,
        )
        state.repositories.set_user_active_save_id(user_id=admin.id, save_id=save.id)
        listed_scenarios = client.get("/api/scenarios")
        listed_saves = client.get("/api/saves")
        blocked = [
            client.get("/api/runtime"),
            client.get("/api/runtime/shell"),
            client.post(
                f"/api/scenarios/{scenario.id}/start",
                json={"save_title": "Copy"},
            ),
            client.post(f"/api/saves/{save.id}/load"),
            client.post(
                f"/api/saves/{save.id}/rename",
                json={"title": "Nope"},
            ),
            client.post(
                "/api/chat",
                json={"save_id": save.id, "body": "Continue."},
            ),
            client.post(
                "/api/scenarios/continuation-draft",
                json={"save_id": save.id},
            ),
            client.get(f"/api/saves/{save.id}/media"),
            client.post(
                f"/api/scenarios/{scenario.id}/definition",
                json={"edit": {"title": "Nope"}},
            ),
            client.post(
                "/api/character-bundles/preview",
                data={"active_save_id": save.id},
                files={
                    "file": (
                        "character.bragi-character",
                        b"character bundle",
                        "application/octet-stream",
                    )
                },
            ),
            client.post(
                "/api/character-bundles/import/retired-preview",
                json={"active_save_id": save.id},
            ),
            client.post(
                "/api/admin/dating-sim-maintenance",
                json={"save_id": save.id, "apply": True},
            ),
            client.post(f"/api/jobs/retired-job/cancel?save_id={save.id}"),
            client.post(
                "/api/settings/model-preference",
                json={
                    "task": "chat",
                    "provider": "fake",
                    "model_id": "fake-chat",
                    "save_id": save.id,
                },
            ),
            client.delete(
                f"/api/settings/model-preference/chat?save_id={save.id}"
            ),
            client.post(
                "/api/settings/model-thinking",
                json={
                    "task": "chat",
                    "provider": "fake",
                    "model_id": "fake-chat",
                    "level": "low",
                    "save_id": save.id,
                },
            ),
            client.delete(
                f"/api/settings/model-thinking/chat?save_id={save.id}"
            ),
            client.post(
                "/api/settings/scoped",
                json={
                    "key": "image_style_preset",
                    "value": "pixel_art",
                    "save_id": save.id,
                },
            ),
        ]
        state.auth_required = False
        state.runtime.active_save_id = save.id
        blocked.extend(
            [
                client.get("/api/runtime"),
                client.get("/api/runtime/shell"),
            ]
        )
        state.auth_required = True
        save_export = client.get(f"/api/bundles/export?save_id={save.id}")
        save_delete = client.delete(f"/api/saves/{save.id}")
        scenario_export = client.get(
            f"/api/scenario-bundles/export/{scenario.id}"
        )
        scenario_delete = client.delete(f"/api/scenarios/{scenario.id}")

    listed_scenario = listed_scenarios.json()["scenarios"][0]
    assert listed_scenario["supported"] is False
    assert listed_scenario["unsupported_reason"] == (
        "The character_interaction scenario type is no longer supported"
    )
    listed_save = listed_saves.json()["saves"][0]
    assert listed_save["supported"] is False
    assert listed_save["unsupported_reason"] == listed_scenario["unsupported_reason"]
    assert [response.status_code for response in blocked] == [409] * len(blocked)
    assert all(
        response.json()["detail"] == listed_scenario["unsupported_reason"]
        for response in blocked
    )
    assert save_export.status_code == 200
    assert save_delete.status_code == 200
    assert scenario_export.status_code == 200
    assert scenario_delete.status_code == 200


def test_character_bundle_export_uses_authorized_save_without_process_active_save(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    app = create_app()
    state = cast(WebAppState, app.state.bragi)

    with TestClient(app) as client:
        save_id = _create_manual_lantern_save(client)
        with state.lock:
            character = state.repositories.add_character(
                save_id=save_id,
                name="Mara",
                role="Signal runner",
            )
            assert state.runtime.active_save_id is None

        exported = client.get(f"/api/character-bundles/export/{character.id}")

    assert exported.status_code == 200
    assert exported.content


def test_auth_login_logout_expired_and_revoked_sessions(tmp_path: Path) -> None:
    state = _auth_state(tmp_path)
    state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="admin",
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        bad_login = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "wrong"},
        )
        login = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        )
        me = client.get("/api/auth/me")
        logout = client.post("/api/auth/logout", json={})
        after_logout = client.get("/api/auth/me")

        expired_user = state.auth_service().create_user(
            username="Expired",
            password="correct horse",
            role="admin",
        )
        state.repositories.create_user_session(
            user_id=expired_user.id,
            token_hash=_test_session_token_hash("expired-token"),
            expires_at="2026-01-01T00:00:00+00:00",
        )
        client.cookies.set("bragi_session", "expired-token")
        expired_runtime = client.get("/api/runtime")

        revoked = state.auth_service().login("Mira", "correct horse")
        assert revoked is not None
        state.auth_service().revoke_session(revoked.token)
        client.cookies.set("bragi_session", revoked.token)
        revoked_runtime = client.get("/api/runtime")

    assert bad_login.status_code == 401
    assert login.status_code == 200
    assert "bragi_session=" in login.headers["set-cookie"]
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "Mira"
    assert logout.status_code == 200
    assert logout.json() == {"ok": True}
    assert "bragi_session=" in logout.headers["set-cookie"]
    assert after_logout.status_code == 401
    assert expired_runtime.status_code == 401
    assert revoked_runtime.status_code == 401


def test_admin_user_management_api_requires_admin_and_manages_users(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    service = state.auth_service()
    admin = service.create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    mira = service.create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    state.repositories.set_scoped_setting(
        scope="user",
        scope_id=mira.id,
        key="content_filter_rating",
        value="pg-13",
    )
    app = create_app(cast(WebAppState, state))

    with (
        TestClient(app, authenticate=False) as user_client,
        TestClient(app, authenticate=False) as admin_client,
        TestClient(app, authenticate=False) as child_client,
    ):
        user_login = user_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        )
        forbidden = user_client.get("/api/admin/users")

        admin_login = admin_client.post(
            "/api/auth/login",
            json={"username": "Admin", "password": "correct horse"},
        )
        users = admin_client.get("/api/admin/users")
        created = admin_client.post(
            "/api/admin/users",
            json={
                "username": "Ilyra",
                "password": "temporary pass",
                "role": "child",
            },
        )
        created_user = created.json()["user"]
        granted = admin_client.patch(
            f"/api/admin/users/{created_user['id']}",
            json={"content_rating": "pg-13"},
        )
        over_granted = admin_client.patch(
            f"/api/admin/users/{created_user['id']}",
            json={"content_rating": "r"},
        )
        invalid_combined_child_grant = admin_client.patch(
            f"/api/admin/users/{mira.id}",
            json={"role": "child", "content_rating": "r"},
        )
        converted_child = admin_client.patch(
            f"/api/admin/users/{mira.id}",
            json={"role": "child"},
        )
        patched = admin_client.patch(
            f"/api/admin/users/{created_user['id']}",
            json={"role": "user", "status": "active"},
        )
        explicitly_granted_on_conversion = admin_client.patch(
            f"/api/admin/users/{created_user['id']}",
            json={"role": "child", "content_rating": "pg-13"},
        )
        invalid = admin_client.patch(
            f"/api/admin/users/{created_user['id']}",
            json={"role": "owner"},
        )
        missing = admin_client.patch(
            "/api/admin/users/missing-user",
            json={"status": "disabled"},
        )

        child_login = child_client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "temporary pass"},
        )
        child_runtime_before_reset = child_client.get("/api/runtime")
        reset = admin_client.post(
            f"/api/admin/users/{created_user['id']}/password",
            json={"password": "new password"},
        )
        child_runtime_after_reset = child_client.get("/api/runtime")
        old_password_login = child_client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "temporary pass"},
        )
        new_password_login = child_client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "new password"},
        )

    assert user_login.status_code == 200
    assert forbidden.status_code == 403
    assert forbidden.json() == {"detail": "Admin access required"}
    assert admin_login.status_code == 200
    assert users.status_code == 200
    assert {
        item["username"]: set(item)
        for item in users.json()["users"]
    }["Admin"] == {
        "id",
        "username",
        "role",
        "status",
        "content_rating",
        "created_at",
        "updated_at",
    }
    assert created.status_code == 200
    assert created_user["username"] == "Ilyra"
    assert created_user["role"] == "child"
    assert created_user["content_rating"] == "pg"
    assert granted.status_code == 200
    assert granted.json()["user"]["content_rating"] == "pg-13"
    assert over_granted.status_code == 400
    assert over_granted.json() == {
        "detail": "Child account content rating cannot exceed PG-13",
    }
    assert invalid_combined_child_grant.status_code == 400
    assert converted_child.status_code == 200
    assert converted_child.json()["user"]["role"] == "child"
    assert converted_child.json()["user"]["content_rating"] == "pg"
    assert patched.status_code == 200
    assert patched.json()["user"]["role"] == "user"
    assert explicitly_granted_on_conversion.status_code == 200
    assert explicitly_granted_on_conversion.json()["user"]["role"] == "child"
    assert (
        explicitly_granted_on_conversion.json()["user"]["content_rating"]
        == "pg-13"
    )
    assert invalid.status_code == 400
    assert invalid.json() == {"detail": "Unknown user role: owner"}
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Unknown user id: missing-user"}
    assert child_login.status_code == 200
    assert child_runtime_before_reset.status_code == 200
    assert reset.status_code == 200
    assert reset.json()["user"]["id"] == created_user["id"]
    assert child_runtime_after_reset.status_code == 401
    assert old_password_login.status_code == 401
    assert new_password_login.status_code == 200
    assert state.repositories.get_user(admin.id) is not None


def test_admin_user_management_protects_last_admin_and_current_session(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    service = state.auth_service()
    admin = service.create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    app = create_app(cast(WebAppState, state))

    with TestClient(app, authenticate=False) as client:
        login = client.post(
            "/api/auth/login",
            json={"username": "Admin", "password": "correct horse"},
        )
        demote_last = client.patch(
            f"/api/admin/users/{admin.id}",
            json={"role": "user"},
        )
        disable_last = client.patch(
            f"/api/admin/users/{admin.id}",
            json={"status": "disabled"},
        )
        service.create_user(
            username="Other Admin",
            password="correct horse",
            role="admin",
        )
        disable_self = client.patch(
            f"/api/admin/users/{admin.id}",
            json={"status": "disabled"},
        )
        demote_self = client.patch(
            f"/api/admin/users/{admin.id}",
            json={"role": "user"},
        )
        after_demote = client.get("/api/admin/users")

    assert login.status_code == 200
    assert demote_last.status_code == 409
    assert demote_last.json() == {
        "detail": "Cannot remove the last active admin",
    }
    assert disable_last.status_code == 409
    assert disable_self.status_code == 409
    assert disable_self.json() == {
        "detail": "Cannot disable your current user",
    }
    assert demote_self.status_code == 200
    assert demote_self.json()["user"]["role"] == "user"
    assert after_demote.status_code == 403


def test_dating_sim_maintenance_admin_job_returns_redacted_report(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    save_id = _create_dating_auth_save(state.repositories, owner_user_id=None)
    state.repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika Arai meets Ren and they exchange numbers.",
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/admin/dating-sim-maintenance",
            json={"save_id": save_id},
        )
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id=save_id)

    assert job["status"] == "succeeded"
    result = job["result"]
    assert result["status"] == "ready"
    assert result["reviewable_repair_count"] == 1
    assert result["reviewable_repairs"][0]["stage"] == "contact_exchanged"
    assert "Mika Arai meets Ren" not in json.dumps(result)


def test_dating_sim_maintenance_requires_admin(tmp_path: Path) -> None:
    state = _auth_state(tmp_path)
    service = state.auth_service()
    user = service.create_user(
        username="regular-user",
        password="correct horse battery",
        role="user",
    )
    save_id = _create_dating_auth_save(state.repositories, owner_user_id=user.id)

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        login = client.post(
            "/api/auth/login",
            json={"username": user.username, "password": "correct horse battery"},
        )
        response = client.post(
            "/api/admin/dating-sim-maintenance",
            json={"save_id": save_id},
        )

    assert login.status_code == 200
    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


def test_auth_gate_keeps_cross_origin_write_guard_first(tmp_path: Path) -> None:
    state = _auth_state(tmp_path)

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
        headers={},
    ) as client:
        response = client.post(
            "/api/runtime/custom-instructions",
            headers={"Origin": "http://evil.example"},
            json={"save_id": "save-1", "custom_instructions": "secret"},
        )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Cross-origin state-changing requests are not allowed"
    }


def test_runtime_startup_uses_web_storage_and_returns_runtime(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        response = client.get("/api/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["scene_title"] == "No save loaded"
    assert (tmp_path / "bragi.sqlite3").is_file()
    assert (tmp_path / "media").is_dir()


def test_runtime_startup_cancels_stale_persisted_jobs(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(database_path, check_same_thread=False)
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    queued = repositories.create_job(
        save_id=save.id,
        type="context_update_retry",
        status="queued",
        payload={},
    )
    queued_text_retry = repositories.create_job(
        save_id=save.id,
        type="character_text_world_update_retry",
        status="queued",
        payload={"text_message_ids": ["text-message-1"]},
    )
    running = repositories.create_job(
        save_id=save.id,
        type="chat_completion",
        status="running",
        payload={},
    )
    running_context_retry = repositories.create_job(
        save_id=save.id,
        type="context_update_retry",
        status="running",
        payload={"source_message_ids": ["player-1", "narrator-1"]},
    )
    running_text_retry = repositories.create_job(
        save_id=save.id,
        type="character_text_world_update_retry",
        status="running",
        payload={"text_message_ids": ["text-message-2"]},
    )

    with TestClient(create_app()) as client:
        response = client.get("/api/runtime")
        active_jobs = client.get(f"/api/jobs?status=active&save_id={save.id}")

    assert response.status_code == 200
    assert active_jobs.status_code == 200
    assert active_jobs.json() == {"jobs": []}
    recovered = {
        job.id: job
        for job in repositories.list_jobs_by_status(("cancelled",))
        if job.id in {running.id, running_context_retry.id, running_text_retry.id}
    }
    preserved = {
        job.id: job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.id in {queued.id, queued_text_retry.id}
    }
    assert preserved[queued.id].payload == {}
    assert preserved[queued_text_retry.id].payload == {
        "text_message_ids": ["text-message-1"]
    }
    assert recovered[running.id].result == {
        "previous_status": "running",
        "recovered_on_startup": True,
    }
    assert recovered[running_context_retry.id].result == {
        "previous_status": "running",
        "recovered_on_startup": True,
    }
    assert recovered[running_text_retry.id].result == {
        "previous_status": "running",
        "recovered_on_startup": True,
    }
    assert recovered[running.id].error == "Job was cancelled during startup recovery"
    assert (
        recovered[running_context_retry.id].error
        == "Job was cancelled during startup recovery"
    )
    assert (
        recovered[running_text_retry.id].error
        == "Job was cancelled during startup recovery"
    )


def test_runtime_model_includes_active_scenario_type(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    app = create_app()
    with TestClient(app) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon snaps awake.",
            },
        )
        runtime = client.get("/api/runtime")

    assert created.status_code == 200
    assert runtime.status_code == 200
    payload = runtime.json()
    assert payload["active_scenario_type"] == "full_roleplay"
    action_ids = {
        action["action_id"]
        for action in payload["chronicle"]["messages"][0]["actions"]
    }
    assert "generate-scene-image" in action_ids
    assert "view-characters-present" in action_ids
    assert "generate-character-image" not in action_ids


def test_runtime_shell_omits_media_and_bounds_chronicle(
    tmp_path: Path,
) -> None:
    from bragi.application.controller import BragiRuntime

    state = _auth_state(tmp_path)
    state.runtime = BragiRuntime(
        repositories=state.repositories,
        providers={},
        media_dir=state.paths.media_dir,
    )
    save = _create_auth_save(
        state.repositories,
        title="Lantern Keep",
        owner_user_id=None,
    )
    messages = [
        state.repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Message {index}",
            content_rating="g",
        )
        for index in range(90)
    ]
    state.repositories.add_message_revision(
        save_id=save.id,
        message_id=messages[-1].id,
        previous_body="Message 89 draft",
        new_body=messages[-1].body,
        diff_unified="diff",
        reconciliation_status="succeeded",
    )
    state.repositories.create_media_asset(
        save_id=save.id,
        source_message_id=messages[-1].id,
        type="image",
        path=f"{save.id}/scene.png",
        prompt="storm beacon over the sea wall",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get(f"/api/runtime/shell?save_id={save.id}")

    assert response.status_code == 200
    payload = response.json()
    chronicle_messages = payload["chronicle"]["messages"]
    assert payload["active_save_id"] == save.id
    assert payload["active_save_title"] == "Lantern Keep"
    assert payload["media"] is None
    assert len(chronicle_messages) == 80
    assert chronicle_messages[0]["body"] == "Message 10"
    assert chronicle_messages[-1]["body"] == "Message 89"
    assert chronicle_messages[-1]["revision_count"] == 1
    assert payload["chronicle"]["has_more_before"] is True
    assert payload["chronicle"]["oldest_message_id"] == messages[10].id


def test_save_media_endpoint_is_save_scoped(
    tmp_path: Path,
) -> None:
    from bragi.application.controller import BragiRuntime

    state = _auth_state(tmp_path)
    state.runtime = BragiRuntime(
        repositories=state.repositories,
        providers={},
        media_dir=state.paths.media_dir,
    )
    service = state.auth_service()
    admin = service.create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    user = service.create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    admin_save = _create_auth_save(
        state.repositories,
        title="Admin Save",
        owner_user_id=admin.id,
    )
    user_save = _create_auth_save(
        state.repositories,
        title="User Save",
        owner_user_id=user.id,
    )
    message = state.repositories.append_message(
        save_id=user_save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon catches the fog.",
    )
    asset = state.repositories.create_media_asset(
        save_id=user_save.id,
        source_message_id=message.id,
        type="image",
        path=f"{user_save.id}/scene.png",
        prompt="storm beacon over the sea wall",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    app = create_app(cast(WebAppState, state))

    with TestClient(app, authenticate=False) as user_client:
        assert user_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        own_media = user_client.get(f"/api/saves/{user_save.id}/media")
        blocked_media = user_client.get(f"/api/saves/{admin_save.id}/media")

    assert own_media.status_code == 200
    assert own_media.json()["media_history"][0]["id"] == asset.id
    assert blocked_media.status_code == 404


def test_media_asset_file_routes_require_save_access(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    service = state.auth_service()
    owner = service.create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    service.create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    save = _create_auth_save(
        state.repositories,
        title="Private Save",
        owner_user_id=owner.id,
    )
    media_path = state.paths.media_dir / save.id / "uploaded.png"
    thumbnail_path = state.paths.media_dir / save.id / "thumbnails" / "uploaded.png"
    media_path.parent.mkdir(parents=True)
    thumbnail_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"private uploaded photo")
    thumbnail_path.write_bytes(b"private thumbnail")
    asset = state.repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path=f"{save.id}/uploaded.png",
        thumbnail_path=f"{save.id}/thumbnails/uploaded.png",
        prompt="private uploaded photo",
        provider="local",
        model="upload",
        status="succeeded",
        mime_type="image/png",
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Rook", "password": "correct horse"},
        ).status_code == 200
        media_model = client.get(f"/api/saves/{save.id}/media")
        original = client.get(f"/api/media/{asset.id}?save_id={save.id}")
        thumbnail = client.get(f"/api/media/{asset.id}/thumbnail?save_id={save.id}")

    assert media_model.status_code == 404
    assert original.status_code == 404
    assert thumbnail.status_code == 404


def test_runtime_model_includes_current_world_time(tmp_path: Path) -> None:
    class RuntimeWithChronicle(_RuntimeDouble):
        def build_model(
            self,
            *,
            active_save_id: str | None | object = ...,
            status: str | None = None,
        ) -> dict[str, object]:
            model = super().build_model(active_save_id=active_save_id, status=status)
            model["chronicle"] = {"messages": []}
            return model

    runtime = RuntimeWithChronicle()
    state = _auth_state(tmp_path, runtime)
    save = _create_auth_save(
        state.repositories,
        title="Lantern Keep",
        owner_user_id=None,
    )
    runtime.active_save_id = save.id
    snapshot = state.repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mira watches the beacon lens.",
        objective="Keep the light stable.",
        in_world_time="Friday evening after class",
        time_of_day="evening",
        day_of_week="friday",
        world_day_index=5,
        weather="clear",
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get(f"/api/runtime?save_id={save.id}")

    assert response.status_code == 200
    assert response.json()["world_time"] == {
        "snapshot_id": snapshot.id,
        "day_index": 5,
        "day_label": "friday",
        "phase": "evening",
        "clock_minutes": None,
        "period_label": "",
        "source_message_id": None,
        "confidence": None,
        "display": "Friday evening; world day index 5",
    }


def test_update_world_time_api_corrects_scene_snapshot(tmp_path: Path) -> None:
    class RuntimeWithChronicle(_RuntimeDouble):
        def build_model(
            self,
            *,
            active_save_id: str | None | object = ...,
            status: str | None = None,
        ) -> dict[str, object]:
            model = super().build_model(active_save_id=active_save_id, status=status)
            model["chronicle"] = {"messages": []}
            return model

    runtime = RuntimeWithChronicle()
    state = _auth_state(tmp_path, runtime)
    save = _create_auth_save(
        state.repositories,
        title="Lantern Keep",
        owner_user_id=None,
    )
    runtime.active_save_id = save.id
    first_message = state.repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name=None,
        body="The first lantern scene begins.",
    )
    source_message = state.repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name=None,
        body="The lantern scene continues.",
    )
    existing = state.repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=None,
        situation="Mira watches the beacon lens.",
        objective="Keep the light stable.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=1,
        world_time_clock_minutes=21 * 60 + 15,
        world_time_period_label="festival week",
        world_time_source_message_id=source_message.id,
        world_time_confidence=0.83,
        weather="rain",
        mood="focused",
        nearby_objects=["brass lens"],
        hazards=["dimming wick"],
        present_character_ids=[],
        source_message_id=source_message.id,
        locked_fields=["weather"],
        first_seen_message_id=first_message.id,
        last_updated_message_id=source_message.id,
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/runtime/world-time",
            json={
                "save_id": save.id,
                "day_label": "Friday",
                "phase": "Night",
                "day_index": 12,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_save_id"] == save.id
    assert payload["world_time"] == {
        "snapshot_id": existing.id,
        "day_index": 12,
        "day_label": "friday",
        "phase": "night",
        "clock_minutes": 21 * 60 + 15,
        "period_label": "festival week",
        "source_message_id": source_message.id,
        "confidence": 0.83,
        "display": "Friday festival week night at 21:15; world day index 12",
    }
    snapshot = state.repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.situation == "Mira watches the beacon lens."
    assert snapshot.objective == "Keep the light stable."
    assert snapshot.weather == "rain"
    assert snapshot.mood == "focused"
    assert snapshot.nearby_objects == ["brass lens"]
    assert snapshot.hazards == ["dimming wick"]
    assert snapshot.in_world_time == "Friday festival week night at 21:15"
    assert snapshot.time_of_day == "night"
    assert snapshot.day_of_week == "friday"
    assert snapshot.world_day_index == 12
    assert snapshot.world_time_day_index == 12
    assert snapshot.world_time_day_label == "friday"
    assert snapshot.world_time_phase == "night"
    assert snapshot.world_time_clock_minutes == 21 * 60 + 15
    assert snapshot.world_time_period_label == "festival week"
    assert snapshot.world_time_source_message_id == source_message.id
    assert snapshot.world_time_confidence == 0.83
    assert snapshot.source_message_id == source_message.id
    assert snapshot.locked_fields == ["weather"]
    assert snapshot.first_seen_message_id == first_message.id
    assert snapshot.last_updated_message_id == source_message.id
    events = state.save_events.events_after(save.id, 0)
    assert [(event.save_id, event.event_type) for event in events] == [
        (save.id, "runtime_changed")
    ]
    assert events[0].payload == {"reason": "world_time_corrected"}


def test_time_loop_baseline_and_reset_api_restore_selective_state(
    tmp_path: Path,
) -> None:
    runtime = _RuntimeDouble()
    state = _auth_state(tmp_path, runtime)
    scenario = state.repositories.create_scenario(
        type="time_loop",
        title="Bellwether Day",
        premise="A festival repeats.",
        player_role="Archivist",
        content={},
    )
    save = state.repositories.create_save(
        scenario_id=scenario.id,
        title="Bell Loop",
    )
    runtime.active_save_id = save.id
    state.repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Dawn at the archive.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=0,
        world_time_clock_minutes=8 * 60,
    )
    state.repositories.upsert_world_state(
        save_id=save.id,
        key="loop.resettable.note",
        value={"status": "dawn"},
        category="loop_resettable",
    )
    state.repositories.upsert_world_state(
        save_id=save.id,
        key="loop.knowledge",
        value={"summary": "The code persists."},
        category="loop_persistent",
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        captured = client.post(
            "/api/world-data/time-loop/baseline",
            json={"save_id": save.id},
        )
        state.repositories.upsert_scene_snapshot(
            save_id=save.id,
            situation="Midnight at the bell.",
            in_world_time="Monday night",
            time_of_day="night",
            day_of_week="monday",
            world_day_index=3,
        )
        state.repositories.upsert_world_state(
            save_id=save.id,
            key="loop.resettable.note",
            value={"status": "midnight"},
            category="loop_resettable",
        )
        reset = client.post(
            "/api/world-data/time-loop/reset",
            json={"save_id": save.id},
        )

    snapshot = state.repositories.get_scene_snapshot(save.id)
    world_state = {row.key: row for row in state.repositories.list_world_state(save.id)}
    assert captured.status_code == 200
    assert reset.status_code == 200
    assert snapshot is not None
    assert snapshot.time_of_day == "morning"
    assert snapshot.world_day_index == 0
    assert world_state["loop.resettable.note"].value == {"status": "dawn"}
    assert world_state["loop.knowledge"].value == {"summary": "The code persists."}
    assert world_state["loop.current"].value["iteration"] == 2
    state_changes = state.repositories.list_state_changes(save.id)
    assert any(
        change.state_key == "loop.resettable.note"
        and change.operation == "upsert"
        for change in state_changes
    )
    assert any(
        change.state_key == "loop.current" and change.operation == "upsert"
        for change in state_changes
    )


def test_update_world_time_api_preserves_legacy_text_for_day_index_only_edit(
    tmp_path: Path,
) -> None:
    class RuntimeWithChronicle(_RuntimeDouble):
        def build_model(
            self,
            *,
            active_save_id: str | None | object = ...,
            status: str | None = None,
        ) -> dict[str, object]:
            model = super().build_model(active_save_id=active_save_id, status=status)
            model["chronicle"] = {"messages": []}
            return model

    runtime = RuntimeWithChronicle()
    state = _auth_state(tmp_path, runtime)
    save = _create_auth_save(
        state.repositories,
        title="Lantern Keep",
        owner_user_id=None,
    )
    runtime.active_save_id = save.id
    state.repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Friday evening after class",
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/runtime/world-time",
            json={"save_id": save.id, "day_index": 9},
        )

        assert response.status_code == 200
        assert response.json()["world_time"]["display"] == (
            "evening; world day index 9"
        )
        legacy_response = client.post(
            "/api/runtime/world-time",
            json={"save_id": save.id, "world_day_index": 10},
        )

    assert legacy_response.status_code == 200
    assert legacy_response.json()["world_time"]["display"] == (
        "evening; world day index 10"
    )
    snapshot = state.repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "Friday evening after class"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == ""
    assert snapshot.world_day_index == 10
    assert snapshot.world_time_day_index == 10
    assert snapshot.world_time_phase == "evening"


def test_update_world_time_api_accepts_legacy_world_day_index_for_new_snapshot(
    tmp_path: Path,
) -> None:
    class RuntimeWithChronicle(_RuntimeDouble):
        def build_model(
            self,
            *,
            active_save_id: str | None | object = ...,
            status: str | None = None,
        ) -> dict[str, object]:
            model = super().build_model(active_save_id=active_save_id, status=status)
            model["chronicle"] = {"messages": []}
            return model

    runtime = RuntimeWithChronicle()
    state = _auth_state(tmp_path, runtime)
    save = _create_auth_save(
        state.repositories,
        title="Lantern Keep",
        owner_user_id=None,
    )
    runtime.active_save_id = save.id

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/runtime/world-time",
            json={"save_id": save.id, "world_day_index": 4},
        )

    assert response.status_code == 200
    assert response.json()["world_time"]["display"] == "world day index 4"
    snapshot = state.repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.world_day_index == 4
    assert snapshot.world_time_day_index == 4


def test_update_world_time_api_rejects_negative_legacy_world_day_index(
    tmp_path: Path,
) -> None:
    runtime = _RuntimeDouble()
    state = _auth_state(tmp_path, runtime)
    save = _create_auth_save(
        state.repositories,
        title="Lantern Keep",
        owner_user_id=None,
    )
    runtime.active_save_id = save.id

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/runtime/world-time",
            json={"save_id": save.id, "world_day_index": -1},
        )

    assert response.status_code == 400
    assert state.repositories.get_scene_snapshot(save.id) is None


def test_update_world_time_api_preserves_legacy_in_world_time_edit(
    tmp_path: Path,
) -> None:
    class RuntimeWithChronicle(_RuntimeDouble):
        def build_model(
            self,
            *,
            active_save_id: str | None | object = ...,
            status: str | None = None,
        ) -> dict[str, object]:
            model = super().build_model(active_save_id=active_save_id, status=status)
            model["chronicle"] = {"messages": []}
            return model

    runtime = RuntimeWithChronicle()
    state = _auth_state(tmp_path, runtime)
    save = _create_auth_save(
        state.repositories,
        title="Lantern Keep",
        owner_user_id=None,
    )
    runtime.active_save_id = save.id
    state.repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Friday morning",
        time_of_day="morning",
        day_of_week="friday",
        world_day_index=5,
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/runtime/world-time",
            json={
                "save_id": save.id,
                "in_world_time": "Friday night after the bell",
            },
        )

    assert response.status_code == 200
    snapshot = state.repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "Friday night after the bell"
    assert snapshot.time_of_day == "night"
    assert snapshot.day_of_week == "friday"
    assert snapshot.world_day_index == 5
    assert snapshot.world_time_phase == "night"


def test_runtime_model_includes_scene_presence_action_for_full_roleplay(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    app = create_app()
    with TestClient(app) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon snaps awake.",
            },
        )
        runtime = client.get("/api/runtime")

    assert created.status_code == 200
    assert runtime.status_code == 200
    payload = runtime.json()
    assert payload["active_scenario_type"] == "full_roleplay"
    action_ids = {
        action["action_id"]
        for action in payload["chronicle"]["messages"][0]["actions"]
    }
    assert "view-characters-present" in action_ids
    assert "generate-character-image" not in action_ids


def test_runtime_model_includes_character_image_action_for_full_roleplay(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    app = create_app()
    with TestClient(app) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon snaps awake.",
            },
        )
        assert created.status_code == 200
        initial_runtime = client.get("/api/runtime")
        assert initial_runtime.status_code == 200
        initial_payload = initial_runtime.json()
        save_id = initial_payload["active_save_id"]
        message_id = initial_payload["chronicle"]["messages"][0]["message_id"]

        repositories = PersistenceRepositories(
            sqlite3.connect(tmp_path / "bragi.sqlite3", check_same_thread=False)
        )
        character = repositories.add_character(
            save_id=save_id,
            name="Mara",
            met=True,
            status="present",
            visual_notes="A storm-cloaked keeper with a brass lantern.",
        )
        reference = repositories.create_media_asset(
            save_id=save_id,
            source_message_id=message_id,
            type="image",
            path=f"{save_id}/{message_id}/reference.png",
            thumbnail_path=None,
            prompt="Mara reference",
            provider="local",
            model="upload",
            status="succeeded",
            metadata={
                "kind": "character_reference",
                "character_id": character.id,
            },
        )
        repositories.add_entity_link(
            save_id=save_id,
            entity_type="character",
            entity_id=character.id,
            target_type="media_asset",
            target_id=reference.id,
            relation="reference_image",
        )
        repositories.replace_message_scene_presence(
            save_id,
            message_id,
            [character.id],
            source="manual",
        )

        runtime = client.get("/api/runtime")

    assert runtime.status_code == 200
    payload = runtime.json()
    assert payload["active_scenario_type"] == "full_roleplay"
    action_ids = {
        action["action_id"]
        for action in payload["chronicle"]["messages"][0]["actions"]
    }
    assert "view-characters-present" in action_ids
    assert "generate-character-image" in action_ids


def test_runtime_model_pages_large_chronicle_and_loads_older_page(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "Opening",
            },
        )
        assert created.status_code == 200
        save_id = created.json()["active_save_id"]
        state = cast(WebAppState, cast(Any, client.app).state.bragi)
        with state.repository_scope():
            for index in range(1, 85):
                state.repositories.append_message(
                    save_id=save_id,
                    role="narrator",
                    speaker_name="Narrator",
                    body=f"Turn {index}",
                    content_rating="g",
                )

        runtime = client.get("/api/runtime")
        assert runtime.status_code == 200
        chronicle = runtime.json()["chronicle"]
        first_page_bodies = [
            message["body"] for message in chronicle["messages"]
        ]
        oldest_message_id = chronicle["oldest_message_id"]

        older = client.get(
            f"/api/saves/{save_id}/chronicle"
            f"?before_message_id={oldest_message_id}"
        )

    assert len(first_page_bodies) == 80
    assert first_page_bodies[0] == "Turn 5"
    assert first_page_bodies[-1] == "Turn 84"
    assert chronicle["has_more_before"] is True
    assert oldest_message_id == chronicle["messages"][0]["message_id"]
    assert older.status_code == 200
    older_chronicle = older.json()
    assert [message["body"] for message in older_chronicle["messages"]] == [
        "Opening",
        "Turn 1",
        "Turn 2",
        "Turn 3",
        "Turn 4",
    ]
    assert older_chronicle["has_more_before"] is False
    assert (
        older_chronicle["oldest_message_id"]
        == older_chronicle["messages"][0]["message_id"]
    )


def test_manual_scenario_create_load_and_delete(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower at the edge of a storm sea.",
                "player_role": "Keeper",
                "opening_message": "The beacon snaps awake.",
            },
        )
        assert created.status_code == 200
        payload = created.json()
        assert payload["active_save_title"] == "Lantern Keep"
        assert payload["chronicle"]["messages"][0]["body"] == "The beacon snaps awake."

        save_id = payload["active_save_id"]
        loaded = client.post(f"/api/saves/{save_id}/load")
        assert loaded.status_code == 200
        assert loaded.json()["active_save_id"] == save_id

        deleted = client.delete(f"/api/saves/{save_id}")
    assert deleted.status_code == 200
    assert deleted.json()["active_save_id"] is None


def test_scoped_runtime_does_not_change_user_selected_save(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        first = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower at the edge of a storm sea.",
                "player_role": "Keeper",
                "opening_message": "The beacon snaps awake.",
            },
        )
        second = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Signal Tower",
                "premise": "A relay station above the clouds.",
                "player_role": "Operator",
                "opening_message": "The repeater hums.",
            },
        )
        assert first.status_code == 200
        assert second.status_code == 200
        first_save_id = first.json()["active_save_id"]
        second_save_id = second.json()["active_save_id"]

        scoped_first = client.get(f"/api/runtime?save_id={first_save_id}")
        loaded_first = client.post(f"/api/saves/{first_save_id}/load")
        scoped_second = client.get(f"/api/runtime?save_id={second_save_id}")
        unscoped = client.get("/api/runtime")

    assert scoped_first.status_code == 200
    assert scoped_first.json()["active_save_title"] == "Lantern Keep"
    assert loaded_first.status_code == 200
    assert loaded_first.json()["active_save_title"] == "Lantern Keep"
    assert scoped_second.status_code == 200
    assert scoped_second.json()["active_save_title"] == "Signal Tower"
    assert unscoped.status_code == 200
    assert unscoped.json()["active_save_title"] == "Lantern Keep"


def test_save_scoped_write_rejects_missing_save_id_after_presentation_load(
    tmp_path: Path,
) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        loaded = client.post("/api/saves/save-2/load")
        chat = client.post("/api/chat", json={"body": "Light the beacon"})
        cleanup = client.post("/api/world-data/context-cleanup", json={})

    assert loaded.status_code == 200
    assert state.runtime.load_save_calls == []
    assert chat.status_code == 400
    assert chat.json()["detail"] == _SAVE_ID_REQUIRED_DETAIL
    assert cleanup.status_code == 400
    assert cleanup.json()["detail"] == _SAVE_ID_REQUIRED_DETAIL


def test_settings_do_not_return_api_key_values(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    monkeypatch.delenv("BRAGI_WEB_USE_KEYRING", raising=False)

    with TestClient(create_app()) as client:
        saved = client.post(
            "/api/settings/provider-key",
            json={"provider": "fake", "api_key": "secret-value"},
        )
        assert saved.status_code == 200

        settings = client.get("/api/settings")

    assert settings.status_code == 200
    payload = settings.json()
    payload_text = settings.text
    assert "secret-value" not in payload_text
    assert '"has_api_key":true' in payload_text
    assert payload.get("secret_storage_warning") is None
    assert "diagnostics" not in payload


def test_provider_settings_endpoint_returns_only_provider_section(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    monkeypatch.delenv("BRAGI_WEB_USE_KEYRING", raising=False)

    with TestClient(create_app()) as client:
        saved = client.post(
            "/api/settings/provider-key",
            json={"provider": "fake", "api_key": "secret-value"},
        )
        providers = client.get("/api/settings/providers")

    assert saved.status_code == 200
    assert providers.status_code == 200
    payload = providers.json()
    fake_card = next(
        card for card in payload["provider_cards"] if card["provider"] == "fake"
    )
    assert fake_card["has_api_key"] is True
    assert payload.get("secret_storage_warning") is None
    assert "secret-value" not in providers.text
    assert "task_model_selectors" not in payload
    assert "pending_jobs_display_mode" not in payload
    assert "openrouter_routing" not in payload


def test_settings_omit_secret_storage_warning_when_service_has_none(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(database_path, check_same_thread=False)
    )

    class SettingsServiceDouble:
        def secret_storage_warning(self) -> None:
            return None

    state = _state_double(tmp_path)
    state.repositories = repositories
    state.providers = {}
    state.settings_service = lambda: SettingsServiceDouble()

    with TestClient(create_app(cast(WebAppState, state))) as client:
        settings = client.get("/api/settings")

    assert settings.status_code == 200
    assert settings.json().get("secret_storage_warning") is None
    assert "diagnostics" not in settings.json()


def test_local_settings_endpoint_returns_only_local_section(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        saved = client.post(
            "/api/settings/local",
            json={"key": "pending_jobs_display_mode", "value": "expanded"},
        )
        local = client.get("/api/settings/local")

    assert saved.status_code == 200
    assert local.status_code == 200
    payload = local.json()
    assert payload["pending_jobs_display_mode"] == {
        "setting_key": "pending_jobs_display_mode",
        "selected": "expanded",
        "options": ["compact", "expanded", "expanded_full"],
    }
    assert payload["user_narration_guidance"] == {
        "setting_key": "user_narration_guidance",
        "value": "",
    }
    assert payload["debug_logging"] == {
        "setting_key": "debug_logging_enabled",
        "enabled": False,
    }
    assert "provider_cards" not in payload
    assert "task_model_selectors" not in payload
    assert "openrouter_routing" not in payload


def test_settings_provider_key_can_be_cleared(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        saved = client.post(
            "/api/settings/provider-key",
            json={"provider": "fake", "api_key": "secret-value"},
        )
        cleared = client.delete("/api/settings/provider-key/fake")
        settings = client.get("/api/settings")

    assert saved.status_code == 200
    assert cleared.status_code == 200
    assert settings.status_code == 200
    fake_card = next(
        card for card in settings.json()["provider_cards"] if card["provider"] == "fake"
    )
    assert fake_card["enabled"] is False
    assert fake_card["has_api_key"] is False
    assert fake_card["refresh_status"] == "No API key"
    assert fake_card["last_error"] is None


def test_settings_provider_key_rejects_blank_and_unknown_provider(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        blank = client.post(
            "/api/settings/provider-key",
            json={"provider": "fake", "api_key": "  "},
        )
        unknown_save = client.post(
            "/api/settings/provider-key",
            json={"provider": "wat", "api_key": "secret-value"},
        )
        unknown_clear = client.delete("/api/settings/provider-key/wat")

    assert blank.status_code == 400
    assert blank.json()["detail"] == "API key must not be blank"
    assert unknown_save.status_code == 404
    assert unknown_clear.status_code == 404


def test_settings_model_refresh_dispatches_job(tmp_path: Path) -> None:
    class SettingsServiceDouble:
        def __init__(self) -> None:
            self.refreshed: list[str] = []

        async def refresh_provider_models(self, provider: str) -> dict[str, object]:
            self.refreshed.append(provider)
            return {
                "provider": provider,
                "configured": True,
                "authenticated": True,
                "model_count": 1,
                "error": None,
            }

    settings = SettingsServiceDouble()
    state = _state_double(tmp_path)
    state.providers = {"fake": object()}
    state.settings_service = lambda: settings

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post("/api/settings/model-refresh/fake", json={})
        assert created.status_code == 200
        job = _wait_for_terminal_job(client, created.json()["id"])

    assert job["status"] == "succeeded"
    assert job["type"] == "model_refresh"
    assert job["result"] == {
        "provider": "fake",
        "configured": True,
        "authenticated": True,
        "model_count": 1,
        "error": None,
    }
    assert settings.refreshed == ["fake"]


def test_settings_model_refresh_does_not_wait_for_runtime_lock(
    tmp_path: Path,
) -> None:
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    class SettingsServiceDouble:
        async def refresh_provider_models(self, provider: str) -> dict[str, object]:
            refresh_started.set()
            await asyncio.to_thread(release_refresh.wait)
            return {
                "provider": provider,
                "configured": True,
                "authenticated": True,
                "model_count": 1,
                "error": None,
            }

    state = _state_double(tmp_path)
    state.providers = {"fake": object()}
    state.settings_service = lambda: SettingsServiceDouble()

    with TestClient(create_app(cast(WebAppState, state))) as client:
        job_id: str | None = None
        try:
            with state.lock:
                created = client.post("/api/settings/model-refresh/fake", json={})
                assert created.status_code == 200
                job_id = cast(str, created.json()["id"])
                started_while_locked = refresh_started.wait(timeout=1.0)
        finally:
            release_refresh.set()

        assert job_id is not None
        job = _wait_for_terminal_job(client, job_id)

    assert started_while_locked is True
    assert job["status"] == "succeeded"


def test_settings_model_refresh_rejects_duplicate_provider_refresh(
    tmp_path: Path,
) -> None:
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    class SettingsServiceDouble:
        async def refresh_provider_models(self, provider: str) -> dict[str, object]:
            refresh_started.set()
            await asyncio.to_thread(release_refresh.wait)
            return {
                "provider": provider,
                "configured": True,
                "authenticated": True,
                "model_count": 1,
                "error": None,
            }

    state = _state_double(tmp_path)
    state.providers = {"fake": object()}
    state.settings_service = lambda: SettingsServiceDouble()

    with TestClient(create_app(cast(WebAppState, state))) as client:
        first = client.post("/api/settings/model-refresh/fake", json={})
        assert first.status_code == 200
        try:
            assert refresh_started.wait(timeout=1.0) is True
            duplicate = client.post("/api/settings/model-refresh/fake", json={})
        finally:
            release_refresh.set()
        job = _wait_for_terminal_job(client, first.json()["id"])

    assert duplicate.status_code == 429
    assert duplicate.json()["detail"] == (
        "An active job with this exclusivity key already exists."
    )
    assert job["status"] == "succeeded"


def test_settings_model_refresh_rejects_unknown_provider(tmp_path: Path) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post("/api/settings/model-refresh/missing", json={})

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown provider"


def test_fake_provider_seed_configures_structured_output_fallback_without_toggle(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        settings = client.get("/api/settings")

    assert settings.status_code == 200
    payload = settings.json()
    assert payload["structured_output_fallback"] == {
        "setting_key": "structured_output_fallback_enabled",
        "enabled": False,
    }
    assert "Structured output fallback model is configured" not in settings.text


def test_fake_provider_seed_configures_available_settings_selectors(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        settings = client.get("/api/settings")

    assert settings.status_code == 200
    payload = settings.json()
    selectors = list(payload["task_model_selectors"])
    selectors.extend(payload["scenario_section_model_selectors"])
    for group in payload["roleplay_model_groups"]:
        selectors.extend(group["selectors"])
    fake_selectors = [
        selector
        for selector in selectors
        if any(option["provider"] == "fake" for option in selector["options"])
    ]

    assert fake_selectors
    assert {
        (selector["task"], selector["selected_provider"], selector["selected_model_id"])
        for selector in fake_selectors
        if selector["selected_provider"] != "fake"
    } == set()
    assert all(selector["selected_available"] for selector in fake_selectors)


def test_fake_provider_seed_enables_image_to_image_generation(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        settings = client.get("/api/settings")

    assert settings.status_code == 200
    payload = settings.json()
    selectors = list(payload["task_model_selectors"])
    for group in payload["roleplay_model_groups"]:
        selectors.extend(group["selectors"])
    image_to_image_selectors = [
        selector
        for selector in selectors
        if selector["task"].endswith("image_to_image_generation")
    ]
    assert image_to_image_selectors
    assert {
        (selector["selected_provider"], selector["selected_model_id"])
        for selector in image_to_image_selectors
    } == {("fake", "fake-edit")}
    assert all(
        {
            option["model_id"]: option["capabilities"]
            for option in selector["options"]
        }
        == {"fake-edit": ["image_to_image"]}
        for selector in image_to_image_selectors
    )
    image_edit_fallback = next(
        selector
        for selector in selectors
        if selector["task"] == "image_edit_fallback"
    )
    assert (
        image_edit_fallback["selected_provider"],
        image_edit_fallback["selected_model_id"],
    ) == ("fake", "fake-edit")
    assert {
        option["model_id"]: option["capabilities"]
        for option in image_edit_fallback["options"]
    } == {"fake-edit": ["image_to_image"]}


def test_fake_provider_seed_preserves_existing_model_preferences(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    import bragi_web.runtime as runtime_module

    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    db_path = tmp_path / "bragi.sqlite3"
    migrate_database(db_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(db_path, check_same_thread=False)
    )
    repositories.upsert_provider_config(
        provider="custom",
        enabled=True,
        has_api_key=True,
    )
    repositories.save_provider_model(
        provider="custom",
        model_id="custom-chat",
        display_name="Custom Chat",
        capabilities=["chat", "structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="chat",
        provider="custom",
        model_id="custom-chat",
    )

    runtime_module._seed_fake_models_if_requested(  # noqa: SLF001
        repositories,
        {"fake": object()},
    )

    preference = repositories.get_model_preference("chat")
    assert preference is not None
    assert preference.provider == "custom"
    assert preference.model_id == "custom-chat"


def test_settings_model_preference_accepts_save_override(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    save = _create_auth_save(
        state.repositories,
        title="Night Watch",
        owner_user_id=None,
    )
    state.repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/server-chat",
    )
    state.settings_service = lambda: SettingsService(
        repositories=state.repositories,
        providers={},
        secret_store=InMemorySecretStore(),
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/settings/model-preference",
            json={
                "task": "chat",
                "provider": "venice",
                "model_id": "venice/save-chat",
                "save_id": save.id,
            },
        )

    assert response.status_code == 200
    assert state.repositories.get_model_preference("chat").model_id == (
        "openrouter/server-chat"
    )
    assert state.repositories.get_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_MODEL_OVERRIDES_SETTING,
    ) == {
        "preferences": {
            "chat": {
                "provider": "venice",
                "model_id": "venice/save-chat",
            }
        }
    }


def test_settings_expose_and_persist_save_scoped_image_style_preset(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        first = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower at the edge of a storm sea.",
                "player_role": "Keeper",
            },
        )
        second = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Signal Tower",
                "premise": "A relay station above the clouds.",
                "player_role": "Operator",
            },
        )
        assert first.status_code == 200
        assert second.status_code == 200
        first_save_id = first.json()["active_save_id"]
        second_save_id = second.json()["active_save_id"]

        settings = client.get(f"/api/settings?save_id={first_save_id}")
        saved = client.post(
            "/api/settings/scoped",
            json={
                "key": "image_style_preset",
                "value": "pixel-art",
                "save_id": first_save_id,
            },
        )
        first_updated = client.get(f"/api/settings?save_id={first_save_id}")
        second_updated = client.get(f"/api/settings?save_id={second_save_id}")

    assert settings.status_code == 200
    payload = settings.json()
    assert payload["image_style_preset"] == {
        "setting_key": "image_style_preset",
        "selected": "realistic",
        "options": EXPECTED_IMAGE_STYLE_PRESETS,
    }
    assert saved.status_code == 200
    assert first_updated.status_code == 200
    assert second_updated.status_code == 200
    assert first_updated.json()["image_style_preset"]["selected"] == "pixel_art"
    assert second_updated.json()["image_style_preset"]["selected"] == "realistic"


def test_settings_expose_and_persist_save_scoped_context_automation_toggles(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        first = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower at the edge of a storm sea.",
                "player_role": "Keeper",
            },
        )
        second = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Signal Tower",
                "premise": "A relay station above the clouds.",
                "player_role": "Operator",
            },
        )
        assert first.status_code == 200
        assert second.status_code == 200
        first_save_id = first.json()["active_save_id"]
        second_save_id = second.json()["active_save_id"]

        settings = client.get(f"/api/settings?save_id={first_save_id}")
        saved_agentic = client.post(
            "/api/settings/scoped",
            json={
                "key": AGENTIC_CONTEXT_PIPELINE_SETTING,
                "value": True,
                "save_id": first_save_id,
            },
        )
        saved_character_actions = client.post(
            "/api/settings/scoped",
            json={
                "key": CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
                "value": False,
                "save_id": first_save_id,
            },
        )
        saved_character_action_cap = client.post(
            "/api/settings/scoped",
            json={
                "key": CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
                "value": 99,
                "save_id": first_save_id,
            },
        )
        missing_save = client.post(
            "/api/settings/scoped",
            json={
                "key": CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
                "value": True,
            },
        )
        first_updated = client.get(f"/api/settings?save_id={first_save_id}")
        second_updated = client.get(f"/api/settings?save_id={second_save_id}")

    assert settings.status_code == 200
    assert settings.json()["agentic_context_pipeline"] == {
        "setting_key": AGENTIC_CONTEXT_PIPELINE_SETTING,
        "enabled": True,
    }
    assert settings.json()["character_action_planning"] == {
        "setting_key": CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        "enabled": True,
    }
    assert settings.json()["character_action_planning_max_concurrency"] == {
        "setting_key": CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
        "value": 20,
        "minimum": 1,
        "maximum": 20,
        "step": 1,
    }
    assert saved_agentic.status_code == 200
    assert saved_character_actions.status_code == 200
    assert saved_character_action_cap.status_code == 200
    assert missing_save.status_code == 400
    assert missing_save.json()["detail"] == _SAVE_ID_REQUIRED_DETAIL
    assert first_updated.status_code == 200
    assert second_updated.status_code == 200
    assert first_updated.json()["agentic_context_pipeline"]["enabled"] is True
    assert second_updated.json()["agentic_context_pipeline"]["enabled"] is True
    assert first_updated.json()["character_action_planning"]["enabled"] is False
    assert second_updated.json()["character_action_planning"]["enabled"] is True
    assert (
        first_updated.json()["character_action_planning_max_concurrency"]["value"]
        == 20
    )
    assert (
        second_updated.json()["character_action_planning_max_concurrency"]["value"]
        == 20
    )


def test_settings_expose_and_persist_pending_jobs_display_mode(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        settings = client.get("/api/settings")
        saved = client.post(
            "/api/settings/scoped",
            json={"key": "pending_jobs_display_mode", "value": "expanded"},
        )
        legacy_saved = client.post(
            "/api/settings/local",
            json={"key": "pending_jobs_display_mode", "value": "expanded_full"},
        )
        updated = client.get("/api/settings")

    assert settings.status_code == 200
    assert settings.json()["pending_jobs_display_mode"] == {
        "setting_key": "pending_jobs_display_mode",
        "selected": "compact",
        "options": ["compact", "expanded", "expanded_full"],
    }
    assert saved.status_code == 200
    assert legacy_saved.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["pending_jobs_display_mode"]["selected"] == "expanded_full"


def test_settings_shell_exposes_only_workbench_display_settings(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        initial = client.get("/api/settings/shell")
        saved = client.post(
            "/api/settings/local",
            json={"key": "pending_jobs_display_mode", "value": "expanded"},
        )
        updated = client.get("/api/settings/shell")

    assert initial.status_code == 200
    assert initial.json() == {
        "pending_jobs_display_mode": {
            "setting_key": "pending_jobs_display_mode",
            "selected": "compact",
            "options": ["compact", "expanded", "expanded_full"],
        }
    }
    assert saved.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["pending_jobs_display_mode"]["selected"] == "expanded"
    assert "provider_cards" not in updated.json()
    assert "diagnostics" not in updated.json()


def test_settings_api_scopes_settings_and_enforces_roles(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    state.providers = {"fake": object()}
    state.settings_service = lambda: SettingsService(
        repositories=state.repositories,
        providers={},
        secret_store=InMemorySecretStore(),
    )
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    rook = state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    mira_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    rook_save = _create_auth_save(
        state.repositories,
        title="Rook Save",
        owner_user_id=rook.id,
    )
    child_save = _create_auth_save(
        state.repositories,
        title="Assigned Save",
        owner_user_id=admin.id,
    )
    state.repositories.grant_save_access(save_id=child_save.id, user_id=child.id)
    app = create_app(cast(WebAppState, state))

    with (
        TestClient(app, authenticate=False) as admin_client,
        TestClient(app, authenticate=False) as user_client,
        TestClient(app, authenticate=False) as child_client,
    ):
        assert admin_client.post(
            "/api/auth/login",
            json={"username": "Admin", "password": "correct horse"},
        ).status_code == 200
        admin_settings = admin_client.get("/api/settings")
        admin_retry_saved = admin_client.post(
            "/api/settings/scoped",
            json={"key": "retry_count", "value": 99},
        )
        admin_updated_settings = admin_client.get("/api/settings")

        assert user_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        user_settings = user_client.get(f"/api/settings?save_id={mira_save.id}")
        user_saved = user_client.post(
            "/api/settings/scoped",
            json={
                "key": "image_generation_frequency",
                "value": 7,
                "save_id": mira_save.id,
            },
        )
        user_guidance = user_client.post(
            "/api/settings/scoped",
            json={
                "key": "user_narration_guidance",
                "value": "  Keep narrator responses to two paragraphs or less.  ",
            },
        )
        user_other_save = user_client.post(
            "/api/settings/scoped",
            json={
                "key": "image_generation_frequency",
                "value": 9,
                "save_id": rook_save.id,
            },
        )
        user_admin_setting = user_client.post(
            "/api/settings/scoped",
            json={"key": "chat_fallback_enabled", "value": True},
        )
        user_retry_setting = user_client.post(
            "/api/settings/scoped",
            json={"key": "retry_count", "value": 2},
        )
        user_provider_key = user_client.post(
            "/api/settings/provider-key",
            json={"provider": "fake", "api_key": "secret-value"},
        )

        assert child_client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        child_settings = child_client.get(f"/api/settings?save_id={child_save.id}")
        child_display_mode = child_client.post(
            "/api/settings/scoped",
            json={
                "key": "pending_jobs_display_mode",
                "value": "expanded_full",
            },
        )
        child_guidance = child_client.post(
            "/api/settings/scoped",
            json={
                "key": "user_narration_guidance",
                "value": "Keep narration cozy.",
            },
        )
        child_save_setting = child_client.post(
            "/api/settings/scoped",
            json={
                "key": "image_generation_frequency",
                "value": 5,
                "save_id": child_save.id,
            },
        )
        child_rating = child_client.post(
            "/api/settings/local",
            json={"key": "content_filter_rating", "value": "g"},
        )
        child_rating_too_high = child_client.post(
            "/api/settings/local",
            json={"key": "content_filter_rating", "value": "pg-13"},
        )
        child_local_settings = child_client.get("/api/settings/local")

    assert user_settings.status_code == 200
    assert admin_settings.status_code == 200
    assert admin_settings.json()["retry_count"] == {
        "setting_key": "retry_count",
        "value": 6,
        "minimum": 0,
        "maximum": 10,
        "step": 1,
    }
    assert admin_retry_saved.status_code == 200
    assert admin_updated_settings.status_code == 200
    assert admin_updated_settings.json()["retry_count"]["value"] == 10
    assert user_settings.json()["visible_sections"] == [
        "save",
        "local",
        "diagnostics",
    ]
    assert user_settings.json()["provider_cards"] == []
    assert "roleplay_shared_models" not in user_settings.json()
    assert "maintenance_jobs" not in user_settings.json()
    assert user_settings.json()["user_narration_guidance"] == {
        "setting_key": "user_narration_guidance",
        "value": "",
    }
    assert user_saved.status_code == 200
    assert user_guidance.status_code == 200
    assert user_other_save.status_code == 404
    assert user_admin_setting.status_code == 403
    assert user_retry_setting.status_code == 403
    assert user_provider_key.status_code == 403
    assert (
        state.repositories.get_effective_setting(
            "image_generation_frequency",
            save_id=mira_save.id,
        )
        == 7
    )
    assert (
        state.repositories.get_effective_setting(
            "image_generation_frequency",
            save_id=rook_save.id,
        )
        is None
    )
    assert (
        state.repositories.get_effective_setting(
            "user_narration_guidance",
            user_id=mira.id,
        )
        == "Keep narrator responses to two paragraphs or less."
    )
    assert (
        state.repositories.get_effective_setting(
            "user_narration_guidance",
            user_id=rook.id,
        )
        is None
    )
    assert state.repositories.get_effective_setting("retry_count") == 10
    assert child_settings.status_code == 200
    assert child_settings.json()["visible_sections"] == ["local"]
    assert child_settings.json()["user_narration_guidance"] == {
        "setting_key": "user_narration_guidance",
        "value": "",
    }
    assert "automatic_image_generation" not in child_settings.json()
    assert child_display_mode.status_code == 200
    assert child_guidance.status_code == 403
    assert child_save_setting.status_code == 403
    assert child_rating.status_code == 200
    assert child_rating_too_high.status_code == 400
    assert child_rating_too_high.json() == {
        "detail": "Child accounts may select only G or PG",
    }
    assert child_local_settings.json()["content_rating"] == {
        "setting_key": "content_filter_rating",
        "selected": "g",
        "options": ["g", "pg"],
        "admin_granted": False,
    }
    assert "fade_to_black" not in child_local_settings.json()
    assert (
        state.repositories.get_effective_setting(
            "pending_jobs_display_mode",
            user_id=child.id,
        )
        == "expanded_full"
    )
    assert (
        state.repositories.get_effective_setting(
            "user_narration_guidance",
            user_id=child.id,
        )
        is None
    )


def test_settings_expose_and_persist_openrouter_routing_profiles(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        settings = client.get("/api/settings")
        saved = client.post(
            "/api/settings/scoped",
            json={
                "key": "openrouter_routing_profiles",
                "value": {
                    "global": {
                        "order": ["deepinfra/turbo", "bad slug"],
                        "sort": "throughput",
                        "sort_partition": "none",
                    }
                },
            },
        )
        updated = client.get("/api/settings")

    assert settings.status_code == 200
    assert settings.json()["openrouter_routing"]["global_provider_payload"] == {}
    assert saved.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["openrouter_routing"]["global_profile"]["order"] == [
        "deepinfra/turbo"
    ]
    assert updated.json()["openrouter_routing"]["global_provider_payload"] == {
        "order": ["deepinfra/turbo"],
        "sort": {"by": "throughput", "partition": "none"},
    }


def test_request_logging_records_metadata_without_request_bodies(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    clear_recent_events()

    with TestClient(create_app()) as client:
        created = client.post(
            "/api/scenarios/manual?debug=secret-query",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "secret body text",
            },
        )
        missing = client.get("/api/jobs/unknown")

    assert created.status_code == 200
    assert missing.status_code == 404
    events = recent_events()
    success = next(
        event
        for event in events
        if event["event"] == "web.request.completed"
        and event["route"] == "/api/scenarios/manual"
    )
    failure = next(
        event
        for event in events
        if event["event"] == "web.request.completed"
        and event["route"] == "/api/jobs/{job_id}"
    )
    assert success["status_code"] == 200
    assert failure["status_code"] == 404
    assert "duration_ms" in success
    assert "secret body text" not in str(events)
    assert "secret-query" not in str(events)


def test_jobs_endpoint_lists_only_active_jobs(tmp_path: Path) -> None:
    state = _state_double(tmp_path)
    queued = JobRecord(id="queued-1", type="chat_turn", status="queued")
    running = JobRecord(id="running-1", type="image_generation", status="running")
    succeeded = JobRecord(id="done-1", type="model_refresh", status="succeeded")
    state.jobs._jobs = {  # noqa: SLF001 - controlled registry fixture
        queued.id: queued,
        running.id: running,
        succeeded.id: succeeded,
    }

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/jobs?status=active")

    assert response.status_code == 200
    payload = response.json()
    assert [job["id"] for job in payload["jobs"]] == ["queued-1", "running-1"]
    assert {job["status"] for job in payload["jobs"]} == {"queued", "running"}


def test_jobs_endpoint_includes_latest_progress_event(tmp_path: Path) -> None:
    state = _state_double(tmp_path)
    running = JobRecord(
        id="running-1",
        type="chat_turn",
        status="running",
        events=[
            {"event": "status", "payload": {"status": "running"}},
            {
                "event": "progress",
                "payload": {
                    "jobs": [
                        {"name": "state", "status": "complete"},
                        {"name": "context", "status": "running"},
                    ],
                },
            },
        ],
    )
    state.jobs._jobs = {running.id: running}  # noqa: SLF001 - controlled fixture

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/jobs?status=active")

    assert response.status_code == 200
    assert response.json()["jobs"][0]["latest_progress"] == {
        "jobs": [
            {"name": "state", "status": "complete"},
            {"name": "context", "status": "running"},
        ],
    }


# Regression: active-save polling must not surface jobs from another save.
# Save switching depends on this remaining scoped by the requested save id.
def test_jobs_endpoint_filters_active_jobs_by_save_id(tmp_path: Path) -> None:
    state = _state_double(tmp_path)
    state.jobs._jobs = {  # noqa: SLF001 - controlled registry fixture
        "chat-1": JobRecord(
            id="chat-1",
            type="chat_turn",
            status="running",
            save_id="save-1",
        ),
        "chat-2": JobRecord(
            id="chat-2",
            type="chat_turn",
            status="running",
            save_id="save-2",
        ),
        "done-1": JobRecord(
            id="done-1",
            type="chat_turn",
            status="succeeded",
            save_id="save-1",
        ),
    }

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/jobs?status=active&save_id=save-1")

    assert response.status_code == 200
    assert [job["id"] for job in response.json()["jobs"]] == ["chat-1"]


def test_jobs_endpoint_lists_terminal_jobs_with_safe_metadata(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(database_path, check_same_thread=False)
    )
    save = _create_auth_save(
        repositories,
        title="Lantern Save",
        owner_user_id=None,
    )
    failed = repositories.update_job(
        repositories.create_job(
            save_id=save.id,
            type="chat_turn",
            status="running",
            payload={"prompt": "secret prompt text", "provider": "fake"},
        ).id,
        status="failed",
        error="provider failed with token=super-secret",
        result={"body": "private result body"},
    )
    repositories.connection.execute(
        """
        UPDATE jobs
        SET created_at = '2026-06-01 12:00:00',
            started_at = '2026-06-01 12:00:03',
            completed_at = '2026-06-01 12:00:10',
            duration_ms = 7000
        WHERE id = ?
        """,
        (failed.id,),
    )
    repositories.record_job_step(
        job_id=failed.id,
        name="provider.chat",
        status="failed",
        provider="fake",
        model="fake-chat",
        task="chat",
        duration_ms=6500,
        error="step leaked token=super-secret",
        metadata={"token_total": 123, "prompt": "secret prompt text"},
    )
    state = _state_double(tmp_path)
    state.repositories = repositories

    with TestClient(create_app(cast(WebAppState, state))) as client:
        history = client.get(
            f"/api/jobs?status=failed&save_id={save.id}"
            "&since=2026-06-01T12:00:00Z&limit=5"
        )
        steps = client.get(f"/api/jobs/{failed.id}/steps?save_id={save.id}")

    assert history.status_code == 200
    payload = history.json()
    assert payload["jobs"] == [
        {
            "id": failed.id,
            "type": "chat_turn",
            "save_id": save.id,
            "status": "failed",
            "created_at": "2026-06-01 12:00:00",
            "started_at": "2026-06-01 12:00:03",
            "completed_at": "2026-06-01 12:00:10",
            "duration_ms": 7000,
            "queue_wait_ms": 3000,
            "step_count": 1,
            "error": SAFE_JOB_ERROR,
            "origin": {"kind": "chat_turn", "label": "Chat turn"},
            "provider": "fake",
            "model": None,
            "detail_available": False,
        }
    ]
    assert "prompt" not in repr(payload)
    assert "private result body" not in repr(payload)
    assert "super-secret" not in repr(payload)
    assert steps.status_code == 200
    assert steps.json() == {
        "job_id": failed.id,
        "steps": [
            {
                "id": steps.json()["steps"][0]["id"],
                "name": "provider.chat",
                "status": "failed",
                "provider": "fake",
                "model": "fake-chat",
                "task": "chat",
                "started_at": steps.json()["steps"][0]["started_at"],
                "completed_at": steps.json()["steps"][0]["completed_at"],
                "duration_ms": 6500,
                "metadata": {"token_total": 123},
            }
        ],
    }
    assert "super-secret" not in repr(steps.json())
    assert "secret prompt text" not in repr(steps.json())


def test_job_diagnostics_endpoint_returns_structured_admin_detail(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(database_path, check_same_thread=False)
    )
    job = repositories.create_job(
        type="character_reference_image",
        status="running",
        payload={
            "job_context": "manual_character_reference",
            "provider": "fake",
            "model": "fake-image",
            "api_key": "secret-key",
        },
    )
    failed = repositories.update_job(
        job.id,
        status="failed",
        error="provider failed token=secret-token",
        result={"provider_payload": {"api_key": "secret-key"}},
        diagnostics={
            "version": 1,
            "request": {
                "origin": {
                    "kind": "manual_character_reference",
                    "label": "Manual character reference image",
                },
                "prompt": "secret roleplay prompt",
                "provider": "fake",
            },
            "provider": {
                "provider": "fake",
                "error_category": "provider_error",
                "http_status": 500,
            },
            "bragi": {"status": "failed", "error": "provider failed"},
        },
    )
    state = _state_double(tmp_path)
    state.repositories = repositories

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get(f"/api/jobs/{failed.id}/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == failed.id
    assert payload["detail_level"] == "admin"
    assert payload["diagnostics"]["request"]["prompt"] == "secret roleplay prompt"
    assert payload["diagnostics"]["provider"] == {
        "provider": "fake",
        "error_category": "provider_error",
        "http_status": 500,
    }
    assert "provider_payload" not in response.text
    assert "secret-key" not in response.text


def test_job_diagnostics_endpoint_redacts_user_detail_and_blocks_children(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    child = state.auth_service().create_user(
        username="Kid",
        password="correct horse",
        role="child",
    )
    user_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    child_save = _create_auth_save(
        state.repositories,
        title="Child Save",
        owner_user_id=child.id,
    )
    user_job = state.repositories.update_job(
        state.repositories.create_job(
            save_id=user_save.id,
            creator_user_id=mira.id,
            type="image_generation",
            status="running",
            payload={"provider": "fake", "model": "fake-image"},
        ).id,
        status="failed",
        error="provider failed token=secret-token",
        diagnostics={
            "version": 1,
            "request": {
                "origin": {"kind": "manual_scene_image", "label": "Manual scene image"},
                "prompt": "private generated image prompt",
                "provider": "fake",
            },
            "provider": {
                "error_category": "provider_error",
                "final_error_message": "private provider response text",
            },
            "bragi": {"status": "failed", "error": "provider failed private text"},
        },
    )
    child_job = state.repositories.update_job(
        state.repositories.create_job(
            save_id=child_save.id,
            creator_user_id=child.id,
            type="image_generation",
            status="running",
            payload={},
        ).id,
        status="failed",
        diagnostics={"request": {"prompt": "child prompt"}},
    )

    app = create_app(cast(WebAppState, state))
    with TestClient(app, authenticate=False) as user_client:
        assert user_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        user_response = user_client.get(
            f"/api/jobs/{user_job.id}/diagnostics?save_id={user_save.id}"
        )

    with TestClient(app, authenticate=False) as child_client:
        assert child_client.post(
            "/api/auth/login",
            json={"username": "Kid", "password": "correct horse"},
        ).status_code == 200
        child_response = child_client.get(
            f"/api/jobs/{child_job.id}/diagnostics?save_id={child_save.id}"
        )

    assert user_response.status_code == 200
    user_payload = user_response.json()
    assert user_payload["detail_level"] == "metadata"
    assert user_payload["diagnostics"]["request"]["origin"] == {
        "kind": "manual_scene_image",
        "label": "Manual scene image",
    }
    assert user_payload["diagnostics"]["provider"] == {
        "error_category": "provider_error"
    }
    assert user_payload["diagnostics"]["bragi"] == {"status": "failed"}
    assert "prompt" not in user_payload["diagnostics"]["request"]
    assert "private generated image prompt" not in user_response.text
    assert "private provider response text" not in user_response.text
    assert "provider failed private text" not in user_response.text
    assert "secret-token" not in user_response.text
    assert child_response.status_code == 403


def test_terminal_job_history_respects_authenticated_job_access(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    rook = state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    mira_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    rook_save = _create_auth_save(
        state.repositories,
        title="Rook Save",
        owner_user_id=rook.id,
    )
    mira_job = state.repositories.update_job(
        state.repositories.create_job(
            save_id=mira_save.id,
            creator_user_id=mira.id,
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="succeeded",
    )
    rook_job = state.repositories.update_job(
        state.repositories.create_job(
            save_id=rook_save.id,
            creator_user_id=rook.id,
            type="chat_turn",
            status="running",
            payload={},
        ).id,
        status="failed",
        error="private failure",
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200

        history = client.get("/api/jobs?status=terminal")
        limited_history = client.get("/api/jobs?status=terminal&limit=1")
        blocked_steps = client.get(
            f"/api/jobs/{rook_job.id}/steps?save_id={rook_save.id}"
        )

    assert history.status_code == 200
    assert [job["id"] for job in history.json()["jobs"]] == [mira_job.id]
    assert limited_history.status_code == 200
    assert [job["id"] for job in limited_history.json()["jobs"]] == [mira_job.id]
    assert blocked_steps.status_code == 404


def test_jobs_endpoint_rejects_unsupported_status_filter(tmp_path: Path) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/jobs?status=stale")

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported job status filter"


def test_unknown_job_cancel_returns_not_found(tmp_path: Path) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post("/api/jobs/unknown/cancel")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown job"


def test_unknown_job_event_stream_returns_not_found(tmp_path: Path) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/jobs/unknown/events")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown job"


def test_job_api_summarizes_failed_jobs_with_public_error(tmp_path: Path) -> None:
    state = _state_double(tmp_path)
    failed = JobRecord(
        id="failed-1",
        type="chat_turn",
        status="failed",
        error=(
            "Provider request failed while echoing Mara's private scene and "
            "api_key=live-secret"
        ),
        save_id="save-1",
    )
    state.jobs._jobs = {failed.id: failed}  # noqa: SLF001 - controlled fixture

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/jobs/failed-1?save_id=save-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["error"] == SAFE_JOB_ERROR
    assert "Mara" not in repr(payload)
    assert "live-secret" not in repr(payload)
    assert "api_key" not in repr(payload)


def test_job_sse_summarizes_failed_job_error_events(tmp_path: Path) -> None:
    async def run_test() -> None:
        state = _state_double(tmp_path)
        failed = JobRecord(
            id="failed-1",
            type="chat_turn",
            status="failed",
            error="provider echoed Mara's private scene and api_key=live-secret",
            events=[
                {
                    "event": "error",
                    "payload": {
                        "error": (
                            "provider echoed Mara's private scene and "
                            "api_key=live-secret"
                        )
                    },
                }
            ],
        )
        state.jobs._jobs = {failed.id: failed}  # noqa: SLF001 - controlled fixture

        chunks = [
            chunk
            async for chunk in api_app._event_stream(  # noqa: SLF001
                cast(WebAppState, state),
                failed.id,
            )
        ]

        assert any("event: error" in chunk for chunk in chunks)
        assert any("event: done" in chunk for chunk in chunks)
        assert SAFE_JOB_ERROR in repr(chunks)
        assert "Mara" not in repr(chunks)
        assert "live-secret" not in repr(chunks)
        assert "api_key" not in repr(chunks)

    asyncio.run(run_test())


def test_job_runtime_payloads_use_fresh_visible_save_list(tmp_path: Path) -> None:
    async def collect_events(state: WebAppState, job_id: str) -> list[str]:
        return [
            chunk
            async for chunk in api_app._event_stream(  # noqa: SLF001
                state,
                job_id,
            )
        ]

    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(database_path, check_same_thread=False)
    )
    first_save = _create_auth_save(
        repositories,
        title="Lantern Save",
        owner_user_id=None,
    )
    second_save = _create_auth_save(
        repositories,
        title="Signal Save",
        owner_user_id=None,
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ? WHERE id = ?",
        ("2026-05-04 00:00:00", first_save.id),
    )
    repositories.connection.execute(
        "UPDATE saves SET updated_at = ? WHERE id = ?",
        ("2026-05-05 00:00:00", second_save.id),
    )
    repositories.commit()

    model = _chat_model("The bell answers.")
    model["active_save_id"] = first_save.id
    model["saves"] = [
        {
            "save_id": "stale-save",
            "title": "Stale Save",
            "active": True,
            "updated_at": "1999-01-01 00:00:00",
        }
    ]
    job = JobRecord(
        id="job-runtime",
        type="chat_turn",
        save_id=first_save.id,
        status="succeeded",
        result=model,
        events=[{"event": "runtime", "payload": model}],
    )
    state = _state_double(tmp_path)
    state.repositories = repositories
    state.jobs._jobs = {job.id: job}  # noqa: SLF001 - controlled fixture

    with TestClient(create_app(cast(WebAppState, state))) as client:
        summary = client.get(f"/api/jobs/{job.id}?save_id={first_save.id}")
    chunks = asyncio.run(collect_events(cast(WebAppState, state), job.id))

    assert summary.status_code == 200
    summary_save_ids = [
        item["save_id"] for item in summary.json()["result"]["saves"]
    ]
    assert summary_save_ids == [second_save.id, first_save.id]
    assert "stale-save" not in repr(chunks)
    assert first_save.id in repr(chunks)
    assert second_save.id in repr(chunks)


def test_job_sse_runtime_payload_uses_fresh_scope_after_request_scope_closes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = ScopedPersistenceRepositories(
        database_path,
        PersistenceRepositories,
    )
    try:
        with repositories.scope():
            scoped_repositories = cast(
                PersistenceRepositories,
                repositories.current_repository(),
            )
            first_save = _create_auth_save(
                scoped_repositories,
                title="Lantern Save",
                owner_user_id=None,
            )
            second_save = _create_auth_save(
                scoped_repositories,
                title="Signal Save",
                owner_user_id=None,
            )
            repositories.connection.execute(
                "UPDATE saves SET updated_at = ? WHERE id = ?",
                ("2026-05-04 00:00:00", first_save.id),
            )
            repositories.connection.execute(
                "UPDATE saves SET updated_at = ? WHERE id = ?",
                ("2026-05-05 00:00:00", second_save.id),
            )
            repositories.connection.commit()

        model = _chat_model("The bell answers.")
        model["active_save_id"] = first_save.id
        model["saves"] = [
            {
                "save_id": "stale-save",
                "title": "Stale Save",
                "active": True,
                "updated_at": "1999-01-01 00:00:00",
            }
        ]
        job = JobRecord(
            id="job-runtime-sse",
            type="chat_turn",
            save_id=first_save.id,
            status="succeeded",
            result=model,
            events=[{"event": "runtime", "payload": model}],
        )
        state = _state_double(tmp_path)
        state.repositories = repositories
        state.jobs = JobRegistry(
            repositories=repositories,
            repository_scope=repositories.scope,
        )
        state.jobs._jobs = {job.id: job}  # noqa: SLF001 - controlled fixture

        with TestClient(create_app(cast(WebAppState, state))) as client:
            with client.stream(
                "GET",
                f"/api/jobs/{job.id}/events?save_id={first_save.id}",
            ) as response:
                body = "".join(response.iter_text())

        assert response.status_code == 200
        assert "event: runtime" in body
        assert "event: done" in body
        assert "stale-save" not in body
        assert first_save.id in body
        assert second_save.id in body
    finally:
        repositories.close()


def test_save_sse_uses_fresh_scope_after_request_scope_closes(
    tmp_path: Path,
) -> None:
    state = _scoped_auth_state(tmp_path)
    repositories = cast(ScopedPersistenceRepositories, state.repositories)
    try:
        user = state.auth_service().create_user(
            username="Mira",
            password="correct horse",
            role="user",
        )
        save = _create_auth_save(
            cast(PersistenceRepositories, repositories),
            title="Lantern Save",
            owner_user_id=user.id,
        )
        state.save_events.publish(
            save.id,
            "runtime_changed",
            {"reason": "chat"},
            owner_user_id=user.id,
        )

        scope = repositories.scope()
        scope.__enter__()
        closed_scope = repositories._scope_var.get()  # noqa: SLF001 - regression setup
        scope.__exit__(None, None, None)
        token = repositories._scope_var.set(closed_scope)  # noqa: SLF001 - regression setup
        try:
            chunks = asyncio.run(
                _collect_save_event_chunks(
                    cast(WebAppState, state),
                    save.id,
                    last_event_id=0,
                    count=1,
                    current_user=user,
                )
            )
        finally:
            repositories._scope_var.reset(token)  # noqa: SLF001 - regression setup

        assert _save_event_ids(chunks) == [1]
        assert '"reason": "chat"' in chunks[0]
    finally:
        repositories.close()


def test_save_event_hub_filters_save_scoped_and_global_events(tmp_path: Path) -> None:
    state = _state_double(tmp_path)
    state.save_events.publish("save-2", "runtime_changed", {"ignored": True})
    state.save_events.publish(None, "saves_changed", {"reason": "library"})
    state.save_events.publish("save-1", "runtime_changed", {"reason": "chat"})

    events = state.save_events.events_after("save-1", 0)

    assert [(event.save_id, event.event_type, event.payload) for event in events] == [
        (None, "saves_changed", {"reason": "library"}),
        ("save-1", "runtime_changed", {"reason": "chat"}),
    ]


def test_save_event_stream_resumes_after_last_event_id(tmp_path: Path) -> None:
    async def run_test() -> None:
        state = _state_double(tmp_path)
        state.save_events.publish("save-1", "runtime_changed", {"step": 1})
        state.save_events.publish(None, "saves_changed", {"step": 2})
        state.save_events.publish("save-1", "runtime_changed", {"step": 3})

        chunks = await _collect_save_event_chunks(
            cast(WebAppState, state),
            "save-1",
            last_event_id=2,
            count=1,
        )

        assert _save_event_ids(chunks) == [3]
        assert '"step": 3' in chunks[0]
        assert '"step": 1' not in chunks[0]

    asyncio.run(run_test())


def test_save_event_stream_replays_retained_events_after_overflow(
    tmp_path: Path,
) -> None:
    async def run_test() -> None:
        state = _state_double(tmp_path)
        state.save_events = SaveEventHub(max_events=2)
        state.save_events.publish("save-1", "runtime_changed", {"step": 1})
        state.save_events.publish("save-1", "runtime_changed", {"step": 2})
        state.save_events.publish("save-1", "runtime_changed", {"step": 3})

        chunks = await _collect_save_event_chunks(
            cast(WebAppState, state),
            "save-1",
            last_event_id=0,
            count=2,
        )

        assert _save_event_ids(chunks) == [2, 3]
        assert '"step": 1' not in repr(chunks)

    asyncio.run(run_test())


def test_save_event_stream_filters_stored_content_for_current_viewer(
    tmp_path: Path,
) -> None:
    async def run_test() -> None:
        graphic_body = (
            "He chopped off the prisoner's head and limbs, spraying the walls red."
        )
        state = _auth_state(tmp_path)
        child = state.auth_service().create_user(
            username="Ilyra",
            password="correct horse",
            role="child",
        )
        state.save_events.publish(
            "save-1",
            "character_texts_changed",
            {
                "id": "text-message-1",
                "thread_id": "thread-1",
                "sender": "character",
                "body": graphic_body,
                "content_rating": "r",
                "attachments": [
                    {
                        "id": "attachment-1",
                        "media_asset_id": "media-1",
                        "prompt_preview": graphic_body,
                        "mime_type": "image/png",
                        "content_rating": "r",
                    }
                ],
            },
        )

        chunks = await _collect_save_event_chunks(
            cast(WebAppState, state),
            "save-1",
            last_event_id=0,
            count=1,
            current_user=child,
        )

        assert CONTENT_FILTER_TRANSITION in chunks[0]
        assert graphic_body not in chunks[0]
        assert "media-1" not in chunks[0]

    asyncio.run(run_test())


def test_save_events_route_reads_last_event_id_header(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _state_double(tmp_path)
    state.save_events.publish("save-1", "runtime_changed", {"step": 1})
    state.save_events.publish("save-1", "runtime_changed", {"step": 2})
    seen: list[tuple[WebAppState, str, int]] = []

    async def stream_double(
        stream_state: WebAppState,
        stream_save_id: str,
        last_event_id: int = 0,
        **_kwargs: object,
    ) -> AsyncGenerator[str, None]:
        seen.append((stream_state, stream_save_id, last_event_id))
        yield "event: done\ndata: {}\n\n"

    monkeypatch.setattr(api_app, "_save_event_stream", stream_double)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get(
            "/api/saves/save-1/events",
            headers={"Last-Event-ID": "1"},
        )

    assert response.status_code == 200
    assert seen == [(state, "save-1", 1)]


def test_save_event_last_event_id_header_parser_falls_back_safely() -> None:
    cursor_from_header = api_app._save_event_cursor_from_header  # noqa: SLF001

    assert cursor_from_header(None, latest_event_id=4) == 0
    assert cursor_from_header("", latest_event_id=4) == 0
    assert cursor_from_header("nope", latest_event_id=4) == 0
    assert cursor_from_header("-1", latest_event_id=4) == 0
    assert cursor_from_header("5", latest_event_id=4) == 0
    assert cursor_from_header("3", latest_event_id=4) == 3


def test_direct_job_routes_require_matching_save_id(tmp_path: Path) -> None:
    class BlockingRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()

        async def run_context_cleanup(
            self,
            *,
            active_save_id: str | None = None,
        ) -> dict[str, object]:
            self.started.set()
            await asyncio.sleep(60)
            return _chat_model(f"Cleaned {active_save_id}")

    runtime = BlockingRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/world-data/context-cleanup",
            json={"save_id": "save-1"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        assert runtime.started.wait(timeout=1.0)

        unscoped = client.get(f"/api/jobs/{job_id}")
        wrong_save = client.get(f"/api/jobs/{job_id}?save_id=save-2")
        scoped = client.get(f"/api/jobs/{job_id}?save_id=save-1")
        wrong_cancel = client.post(f"/api/jobs/{job_id}/cancel?save_id=save-2")
        scoped_cancel = client.post(f"/api/jobs/{job_id}/cancel?save_id=save-1")

    assert unscoped.status_code == 404
    assert wrong_save.status_code == 404
    assert scoped.status_code == 200
    assert scoped.json()["save_id"] == "save-1"
    assert wrong_cancel.status_code == 404
    assert scoped_cancel.status_code == 200
    assert scoped_cancel.json() == {"cancelled": True}


def test_chat_submission_status_blocks_same_save_active_chat_turn(
    tmp_path: Path,
) -> None:
    state = _state_double(tmp_path)
    jobs = _TrackingJobRegistry()
    jobs._jobs = {  # noqa: SLF001 - controlled registry fixture
        "chat-1": JobRecord(
            id="chat-1",
            type="chat_turn",
            status="running",
            save_id="save-1",
        ),
        "chat-2": JobRecord(
            id="chat-2",
            type="chat_turn",
            status="running",
            save_id="save-2",
        ),
        "image-1": JobRecord(
            id="image-1",
            type="image_generation",
            status="running",
            save_id="save-1",
        ),
    }
    state.jobs = jobs

    with TestClient(create_app(cast(WebAppState, state))) as client:
        blocked = client.get("/api/chat/submission-status?save_id=save-1")
        allowed = client.get("/api/chat/submission-status?save_id=save-3")

    assert blocked.status_code == 200
    assert blocked.json() == {
        "save_id": "save-1",
        "can_submit": False,
        "reason": "chat_turn_active",
        "blocking_job_id": "chat-1",
        "blocking_job_status": "running",
    }
    assert allowed.status_code == 200
    assert allowed.json() == {
        "save_id": "save-3",
        "can_submit": True,
        "reason": None,
        "blocking_job_id": None,
        "blocking_job_status": None,
    }
    assert jobs.list_active_save_ids == ["save-1", "save-3"]


def test_chat_submission_status_blocks_when_no_save_is_available(
    tmp_path: Path,
) -> None:
    runtime = _RuntimeDouble()
    runtime.active_save_id = None
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/chat/submission-status")

    assert response.status_code == 200
    assert response.json() == {
        "save_id": None,
        "can_submit": False,
        "reason": "no_save",
        "blocking_job_id": None,
        "blocking_job_status": None,
    }


def test_chat_post_rejects_same_save_active_chat_turn(tmp_path: Path) -> None:
    class RuntimeWithCalls(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.submissions: list[str] = []

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
        ) -> object:
            self.submissions.append(body)
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

    runtime = RuntimeWithCalls()
    state = _state_double(tmp_path, runtime)
    state.jobs._jobs = {  # noqa: SLF001 - controlled registry fixture
        "chat-1": JobRecord(
            id="chat-1",
            type="chat_turn",
            status="running",
            save_id="save-1",
        )
    }

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A chat turn is already being processed for this save."
    }
    assert runtime.submissions == []


def test_chat_regenerate_rejects_same_save_active_chat_turn(tmp_path: Path) -> None:
    class RuntimeWithRegenerate(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        async def regenerate_message(
            self,
            *,
            message_id: str,
            active_save_id: str | None | object = ...,
            regeneration_feedback: str = "",
        ) -> object:
            self.calls.append(message_id)
            return _chat_model("The bell answers.")

    runtime = RuntimeWithRegenerate()
    state = _state_double(tmp_path, runtime)
    state.jobs._jobs = {  # noqa: SLF001 - controlled registry fixture
        "chat-1": JobRecord(
            id="chat-1",
            type="chat_turn",
            status="running",
            save_id="save-1",
        )
    }

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/chat/regenerate",
            json={"message_id": "narrator-1", "save_id": "save-1"},
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A chat turn is already being processed for this save."
    }
    assert runtime.calls == []


def test_action_choice_regenerate_rejects_same_save_active_chat_turn(
    tmp_path: Path,
) -> None:
    class RuntimeWithActionChoiceRegenerate(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        async def regenerate_action_choices(
            self,
            *,
            narrator_message_id: str,
            active_save_id: str | None | object = ...,
        ) -> object:
            self.calls.append(narrator_message_id)
            return _chat_model("Options regenerated")

    runtime = RuntimeWithActionChoiceRegenerate()
    state = _state_double(tmp_path, runtime)
    state.jobs._jobs = {  # noqa: SLF001 - controlled registry fixture
        "chat-1": JobRecord(
            id="chat-1",
            type="chat_turn",
            status="running",
            save_id="save-1",
        )
    }

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/action-choices/regenerate",
            json={"message_id": "narrator-1", "save_id": "save-1"},
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A chat turn is already being processed for this save."
    }
    assert runtime.calls == []


def test_chat_revision_jobs_use_same_save_exclusivity(tmp_path: Path) -> None:
    class BlockingRevisionRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls: list[str] = []

        async def edit_and_resubmit_message(
            self,
            *,
            message_id: str,
            body: str,
            active_save_id: str | None | object = ...,
            retry_progress_callback: Callable[[object], None] | None = None,
        ) -> dict[str, object]:
            self.calls.append(message_id)
            self.entered.set()
            await asyncio.to_thread(self.release.wait)
            return _chat_model("The edit resolves.")

    runtime = BlockingRevisionRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        first = client.post(
            "/api/chat/edit",
            json={
                "message_id": "player-1",
                "body": "Hold the line",
                "save_id": "save-1",
            },
        )
        assert first.status_code == 200
        assert runtime.entered.wait(timeout=2.0)

        blocked = client.post(
            "/api/chat/regenerate",
            json={"message_id": "narrator-1", "save_id": "save-1"},
        )
        runtime.release.set()
        job = _wait_for_terminal_job(client, first.json()["id"], save_id="save-1")

    assert blocked.status_code == 409
    assert blocked.json() == {
        "detail": "A chat turn is already being processed for this save."
    }
    assert job["status"] == "succeeded"
    assert runtime.calls == ["player-1"]


@pytest.mark.parametrize(
    ("path", "payload", "expected_call"),
    [
        (
            "/api/chat/regenerate",
            {"message_id": "narrator-1", "save_id": "save-1"},
            "regenerate:narrator-1",
        ),
        (
            "/api/chat/edit",
            {
                "message_id": "player-1",
                "body": "Hold the line",
                "save_id": "save-1",
            },
            "edit:player-1",
        ),
        (
            "/api/chat/message-edit",
            {
                "message_id": "player-1",
                "body": "Hold the line",
                "save_id": "save-1",
            },
            "message-edit:player-1",
        ),
        (
            "/api/chat/narrator-edit",
            {
                "message_id": "narrator-1",
                "body": "The bell answers.",
                "save_id": "save-1",
            },
            "narrator-edit:narrator-1",
        ),
        (
            "/api/chat/delete-from-here",
            {"message_id": "narrator-1", "save_id": "save-1"},
            "delete:narrator-1",
        ),
        (
            "/api/chat/fork-from-here",
            {"message_id": "narrator-1", "save_id": "save-1"},
            "fork:narrator-1",
        ),
    ],
)
def test_chat_mutations_queue_behind_active_save_job_before_calling_runtime(
    tmp_path: Path,
    path: str,
    payload: dict[str, object],
    expected_call: str,
) -> None:
    class RuntimeWithChronicleMutations(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []
            self.cleanup_entered = threading.Event()
            self.release_cleanup = threading.Event()

        async def run_context_cleanup(
            self,
            *,
            active_save_id: str | None = None,
        ) -> dict[str, object]:
            self.cleanup_entered.set()
            await asyncio.to_thread(self.release_cleanup.wait)
            return _chat_model("Cleaned context.")

        async def regenerate_message(
            self,
            *,
            message_id: str,
            active_save_id: str | None | object = ...,
            regeneration_feedback: str = "",
        ) -> object:
            self.calls.append(f"regenerate:{message_id}")
            return _chat_model("Regenerated.")

        async def edit_and_resubmit_message(
            self,
            *,
            message_id: str,
            body: str,
            active_save_id: str | None | object = ...,
            retry_progress_callback: Callable[[object], None] | None = None,
        ) -> object:
            self.calls.append(f"edit:{message_id}")
            return _chat_model("Edited.")

        async def edit_narrator_message(
            self,
            *,
            message_id: str,
            body: str,
            active_save_id: str | None | object = ...,
            on_revision_committed: Callable[[object], None] | None = None,
        ) -> object:
            self.calls.append(f"narrator-edit:{message_id}")
            return _chat_model("Saved.")

        async def edit_message_without_resubmit(
            self,
            *,
            message_id: str,
            body: str,
            active_save_id: str | None | object = ...,
            on_revision_committed: Callable[[object], None] | None = None,
        ) -> object:
            self.calls.append(f"message-edit:{message_id}")
            return _chat_model("Saved.")

        def delete_messages_from_here(
            self,
            *,
            message_id: str,
            active_save_id: str | None | object = ...,
        ) -> object:
            self.calls.append(f"delete:{message_id}")
            return _chat_model("Deleted.")

        def fork_save_from_message(
            self,
            *,
            message_id: str,
            active_save_id: str | None | object = ...,
            **_kwargs: object,
        ) -> object:
            self.calls.append(f"fork:{message_id}")
            return _chat_model("Forked.")

    runtime = RuntimeWithChronicleMutations()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        blocker = client.post(
            "/api/world-data/context-cleanup",
            json={"save_id": "save-1"},
        )
        assert blocker.status_code == 200
        assert runtime.cleanup_entered.wait(timeout=1.0)

        queued = client.post(path, json=payload)
        assert queued.status_code == 200
        assert queued.json()["status"] == "queued"
        assert runtime.calls == []

        runtime.release_cleanup.set()
        job = _wait_for_terminal_job(client, queued.json()["id"], save_id="save-1")
        blocker_job = _wait_for_terminal_job(
            client,
            blocker.json()["id"],
            save_id="save-1",
        )

    assert blocker_job["status"] == "succeeded"
    assert job["status"] == "succeeded"
    assert runtime.calls == [expected_call]


# Regression: save-scoped job creation must persist the save id so the UI can
# hide old-save work after a save switch.
def test_chat_post_records_save_id_on_created_job(tmp_path: Path) -> None:
    class RuntimeWithCalls(_RuntimeDouble):
        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
        ) -> object:
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

    state = _state_double(tmp_path, RuntimeWithCalls())

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )

    assert created.status_code == 200
    assert created.json()["save_id"] == "save-1"


def test_chat_post_rejects_internal_story_continuation_speaker(
    tmp_path: Path,
) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/chat",
            json={
                "save_id": "save-1",
                "body": "Hide this user-authored message.",
                "speaker_name": "Bragi Story Continuation",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "speaker_name is reserved for internal Storyteller turns"
    )


def test_continue_story_submits_server_owned_storyteller_direction(
    tmp_path: Path,
) -> None:
    class RuntimeWithCalls(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.submissions: list[tuple[str, str | None, object]] = []

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
        ) -> object:
            self.submissions.append((body, speaker_name, active_save_id))
            return SimpleNamespace(
                model=_chat_model("The rival steps into the aisle."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="continuation-1",
                narrator_message_id="narrator-1",
            )

    runtime = RuntimeWithCalls()
    state = _state_double(tmp_path, runtime)
    state.repositories.get_save = lambda _save_id: SimpleNamespace(
        interaction_mode=InteractionMode.STORYTELLER
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/continue",
            json={"save_id": "save-1"},
        )
        assert created.status_code == 200
        job = _wait_for_terminal_job(
            client,
            created.json()["id"],
            save_id="save-1",
        )

    assert job["status"] == "succeeded"
    assert runtime.submissions == [
        (
            "Continue the story naturally from the current moment. Choose the "
            "next logical beat from established canon and unresolved threads, "
            "keeping the current pace unless the story calls for a transition.",
            "Bragi Story Continuation",
            "save-1",
        )
    ]


def test_continue_story_rejects_roleplay_save(tmp_path: Path) -> None:
    state = _state_double(tmp_path)
    state.repositories.get_save = lambda _save_id: SimpleNamespace(
        interaction_mode=InteractionMode.ROLEPLAY
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/chat/continue",
            json={"save_id": "save-1"},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Continue story is only available in Storyteller mode."
    )


def test_timeskip_post_records_save_id_on_created_job(tmp_path: Path) -> None:
    class RuntimeWithTimeskip(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.submissions: list[tuple[str, object]] = []

        async def submit_timeskip_for_initial_render(
            self,
            *,
            instruction: str,
            active_save_id: object,
        ) -> object:
            self.submissions.append((instruction, active_save_id))
            return SimpleNamespace(
                model=_chat_model("Dawn catches on the city gates."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="timeskip-1",
                narrator_message_id="narrator-1",
            )

    runtime = RuntimeWithTimeskip()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/timeskip",
            json={
                "instruction": "Skip to dawn at the city gates.",
                "save_id": "save-1",
            },
        )

    assert created.status_code == 200
    assert created.json()["save_id"] == "save-1"
    assert runtime.submissions == [("Skip to dawn at the city gates.", "save-1")]


@pytest.mark.parametrize("max_active_jobs", (None, 1), ids=("queued", "inline"))
def test_timeskip_post_turn_jobs_preserve_request_actor(
    tmp_path: Path,
    max_active_jobs: int | None,
) -> None:
    class RuntimeWithTimeskipPostTurn(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.submission_user_ids: list[str | None] = []
            self.post_turn_user_ids: list[str | None] = []
            self.post_turn_finished = threading.Event()

        async def submit_timeskip_for_initial_render(
            self,
            *,
            instruction: str,
            active_save_id: object,
            current_user_id: str | None = None,
        ) -> object:
            del instruction
            self.submission_user_ids.append(current_user_id)
            return SimpleNamespace(
                model=_chat_model("Dawn catches on the city gates."),
                has_post_turn_jobs=True,
                save_id=cast(str, active_save_id),
                player_message_id="timeskip-1",
                narrator_message_id="narrator-1",
            )

        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
            progress_callback: object | None = None,
            current_user_id: str | None = None,
        ) -> dict[str, object]:
            del save_id, player_message_id, narrator_message_id, progress_callback
            self.post_turn_user_ids.append(current_user_id)
            self.post_turn_finished.set()
            return _chat_model("The world settles.")

    runtime = RuntimeWithTimeskipPostTurn()
    state = _auth_state(tmp_path, runtime)
    if max_active_jobs is not None:
        state.jobs = JobRegistry(
            JobRegistryLimits(max_active_jobs=max_active_jobs),
        )
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    save = _create_auth_save(
        state.repositories,
        title="Lantern Keep",
        owner_user_id=user.id,
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        created = client.post(
            "/api/chat/timeskip",
            json={
                "instruction": "Skip to dawn at the city gates.",
                "save_id": save.id,
            },
        )
        assert created.status_code == 200
        job = _wait_for_terminal_job(client, created.json()["id"], save_id=save.id)
        assert runtime.post_turn_finished.wait(timeout=2.0)

    assert job["status"] == "succeeded"
    assert runtime.submission_user_ids == [user.id]
    assert runtime.post_turn_user_ids == [user.id]


def test_look_around_post_records_save_id_and_returns_answer_job(
    tmp_path: Path,
) -> None:
    class RuntimeWithLookAround(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.looks: list[tuple[str, object]] = []

        async def look_around(
            self,
            *,
            query: str,
            active_save_id: object,
        ) -> dict[str, object]:
            self.looks.append((query, active_save_id))
            return {
                "answer": "The brass lens hides a locked prism.",
                "save_id": "save-1",
                "latest_narrator_message_id": "narrator-1",
                "context_observation_id": "observation-1",
                "update_counts": {
                    "observations": 1,
                    "suggestions": 0,
                    "memories": 0,
                    "context_sources": 0,
                },
                "answer_markdown_blocks": [],
            }

    runtime = RuntimeWithLookAround()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/look-around",
            json={"query": "Inspect the brass lens.", "save_id": "save-1"},
        )
        assert created.status_code == 200
        job = _wait_for_terminal_job(
            client,
            created.json()["id"],
            save_id="save-1",
        )

    assert created.json()["type"] == "look_around"
    assert created.json()["save_id"] == "save-1"
    assert job["status"] == "succeeded"
    assert job["result"]["answer"] == "The brass lens hides a locked prism."
    assert job["result"]["answer_markdown_blocks"] == []
    assert runtime.looks == [("Inspect the brass lens.", "save-1")]


def test_chat_and_look_around_reject_oversized_text_inputs(tmp_path: Path) -> None:
    state = _state_double(tmp_path, _RuntimeDouble())
    oversized_chat = "x" * (api_app.MAX_CHAT_BODY_CHARS + 1)
    oversized_query = "x" * (api_app.MAX_LOOK_AROUND_QUERY_CHARS + 1)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        chat_response = client.post(
            "/api/chat",
            json={"body": oversized_chat, "save_id": "save-1"},
        )
        look_response = client.post(
            "/api/chat/look-around",
            json={"query": oversized_query, "save_id": "save-1"},
        )

    assert chat_response.status_code == 422
    assert look_response.status_code == 422


def test_json_request_body_limit_rejects_before_validation(tmp_path: Path) -> None:
    state = _state_double(tmp_path, _RuntimeDouble())
    oversized_body = b"x" * (api_app.MAX_JSON_REQUEST_BODY_BYTES + 1)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        json_response = client.post(
            "/api/chat/look-around",
            content=oversized_body,
            headers={"content-type": "application/json"},
        )
        text_response = client.post(
            "/api/chat/look-around",
            content=oversized_body,
            headers={"content-type": "text/plain"},
        )

    for response in (json_response, text_response):
        assert response.status_code == 413
        assert response.json() == {"detail": "Request body too large"}


def test_request_body_limit_counts_chunked_body_without_content_length() -> None:
    request_messages = iter(
        (
            {
                "type": "http.request",
                "body": b"x" * 64,
                "more_body": True,
            },
            {
                "type": "http.request",
                "body": b"x" * api_app.MAX_JSON_REQUEST_BODY_BYTES,
                "more_body": False,
            },
        )
    )
    sent_messages: list[dict[str, object]] = []

    async def receive() -> Any:
        return next(request_messages)

    async def send(message: Any) -> None:
        sent_messages.append(message)

    async def consume_body(
        _scope: Any,
        receive_body: Any,
        _send: Any,
    ) -> None:
        while True:
            message = await receive_body()
            if not message.get("more_body"):
                return

    middleware = api_app._JsonRequestBodyLimitMiddleware(
        consume_body,
        max_body_bytes=api_app.MAX_JSON_REQUEST_BODY_BYTES,
    )
    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/chat/look-around",
                "headers": [],
            },
            receive,
            send,
        )
    )

    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 413


def test_request_body_limit_counts_chunked_multipart_upload_body() -> None:
    chunk = b"x" * (
        (
            api_app.CHARACTER_REFERENCE_UPLOAD_MAX_BYTES
            + api_app.MULTIPART_REQUEST_OVERHEAD_BYTES
        )
        // 2
        + 1
    )
    request_messages = iter(
        (
            {
                "type": "http.request",
                "body": chunk,
                "more_body": True,
            },
            {
                "type": "http.request",
                "body": chunk,
                "more_body": False,
            },
        )
    )
    sent_messages: list[dict[str, object]] = []

    async def receive() -> Any:
        return next(request_messages)

    async def send(message: Any) -> None:
        sent_messages.append(message)

    async def consume_body(
        _scope: Any,
        receive_body: Any,
        _send: Any,
    ) -> None:
        while True:
            message = await receive_body()
            if not message.get("more_body"):
                return

    middleware = api_app._JsonRequestBodyLimitMiddleware(
        consume_body,
        max_body_bytes=api_app.MAX_JSON_REQUEST_BODY_BYTES,
    )
    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/media/character-reference/upload",
                "headers": [
                    (
                        b"content-type",
                        b"multipart/form-data; boundary=bragi-test-boundary",
                    )
                ],
            },
            receive,
            send,
        )
    )

    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 413


def test_request_body_limit_rejects_oversized_multipart_content_length() -> None:
    sent_messages: list[dict[str, object]] = []

    async def receive() -> Any:
        raise AssertionError("oversized request body should not be read")

    async def send(message: Any) -> None:
        sent_messages.append(message)

    async def consume_body(
        _scope: Any,
        _receive: Any,
        _send: Any,
    ) -> None:
        raise AssertionError("oversized request should not reach the application")

    middleware = api_app._JsonRequestBodyLimitMiddleware(
        consume_body,
        max_body_bytes=api_app.MAX_JSON_REQUEST_BODY_BYTES,
    )
    request_limit = (
        api_app.BUNDLE_UPLOAD_MAX_BYTES
        + api_app.MULTIPART_REQUEST_OVERHEAD_BYTES
    )
    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/bundles/preview",
                "headers": [
                    (b"content-type", b"multipart/form-data; boundary=test"),
                    (b"content-length", str(request_limit + 1).encode()),
                ],
            },
            receive,
            send,
        )
    )

    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 413


def test_look_around_post_returns_answer_markdown_blocks(
    tmp_path: Path,
) -> None:
    class RuntimeWithLookAround(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.looks: list[tuple[str, object]] = []

        async def look_around(
            self,
            *,
            query: str,
            active_save_id: object,
        ) -> dict[str, object]:
            self.looks.append((query, active_save_id))
            return {
                "answer": (
                    "The brass lens rests on a velvet pad.\n"
                    "- faintly etched runes\n"
                    "- a small keyhole"
                ),
                "save_id": "save-1",
                "latest_narrator_message_id": "narrator-1",
                "context_observation_id": "observation-1",
                "update_counts": {
                    "observations": 1,
                    "suggestions": 0,
                    "memories": 0,
                    "context_sources": 0,
                },
                "answer_markdown_blocks": [
                    {
                        "kind": "paragraph",
                        "spans": [
                            {
                                "kind": "text",
                                "text": "The brass lens rests on a velvet pad.",
                            }
                        ],
                    },
                    {
                        "kind": "bullet_item",
                        "spans": [{"kind": "text", "text": "faintly etched runes"}],
                        "list_kind": "bullet",
                        "marker": "•",
                    },
                    {
                        "kind": "bullet_item",
                        "spans": [{"kind": "text", "text": "a small keyhole"}],
                        "list_kind": "bullet",
                        "marker": "•",
                    },
                ],
            }

    runtime = RuntimeWithLookAround()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/look-around",
            json={"query": "Inspect the brass lens.", "save_id": "save-1"},
        )
        assert created.status_code == 200
        job = _wait_for_terminal_job(
            client,
            created.json()["id"],
            save_id="save-1",
        )

    blocks = job["result"]["answer_markdown_blocks"]
    assert [block["kind"] for block in blocks] == [
        "paragraph",
        "bullet_item",
        "bullet_item",
    ]
    assert blocks[1]["spans"][0]["text"] == "faintly etched runes"
    assert blocks[2]["spans"][0]["text"] == "a small keyhole"


def test_chat_turn_exposes_initial_pre_narrator_phase_progress(
    tmp_path: Path,
) -> None:
    class BlockingRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
            turn_progress_callback: Callable[[object], None] | None = None,
        ) -> object:
            self.entered.set()
            await asyncio.to_thread(self.release.wait)
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

    runtime = BlockingRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]

        assert runtime.entered.wait(timeout=2)
        job = client.get(_job_url(job_id, "save-1")).json()
        runtime.release.set()
        _wait_for_terminal_job(client, job_id, save_id="save-1")

    assert job["status"] == "running"
    assert job["latest_progress"] == _expected_chat_turn_progress(
        "Submitting turn",
        running="submission",
    )


def test_chat_turn_uses_runtime_pre_narrator_phase_progress(
    tmp_path: Path,
) -> None:
    class ProgressRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.progress_sent = threading.Event()
            self.release = threading.Event()

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
            turn_progress_callback: Callable[[object], None] | None = None,
        ) -> object:
            assert turn_progress_callback is not None
            turn_progress_callback(
                _expected_chat_turn_progress(
                    "Selecting context",
                    succeeded=("submission", "history", "input", "time_state"),
                    running="context_selection",
                )
            )
            self.progress_sent.set()
            await asyncio.to_thread(self.release.wait)
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

    runtime = ProgressRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]

        assert runtime.progress_sent.wait(timeout=2)
        job: dict[str, Any] = {}
        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job.get("latest_progress", {}).get("status_text") == "Selecting context":
                break
            time.sleep(0.01)
        runtime.release.set()
        _wait_for_terminal_job(client, job_id, save_id="save-1")

    assert job["status"] == "running"
    assert job["latest_progress"] == _expected_chat_turn_progress(
        "Selecting context",
        succeeded=("submission", "history", "input", "time_state"),
        running="context_selection",
    )


def test_timeskip_exposes_initial_pre_narrator_phase_progress(
    tmp_path: Path,
) -> None:
    class BlockingRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        async def submit_timeskip_for_initial_render(
            self,
            *,
            instruction: str,
            active_save_id: object,
            turn_progress_callback: Callable[[object], None] | None = None,
        ) -> object:
            self.entered.set()
            await asyncio.to_thread(self.release.wait)
            return SimpleNamespace(
                model=_chat_model("Dawn catches on the city gates."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="timeskip-1",
                narrator_message_id="narrator-1",
            )

    runtime = BlockingRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/timeskip",
            json={
                "instruction": "Skip to dawn at the city gates.",
                "save_id": "save-1",
            },
        )
        assert created.status_code == 200
        job_id = created.json()["id"]

        assert runtime.entered.wait(timeout=2)
        job = client.get(_job_url(job_id, "save-1")).json()
        runtime.release.set()
        _wait_for_terminal_job(client, job_id, save_id="save-1")

    assert job["status"] == "running"
    assert job["latest_progress"] == _expected_chat_turn_progress(
        "Submitting timeskip",
        running="submission",
    )


def test_timeskip_post_rejects_blank_instruction(tmp_path: Path) -> None:
    class RuntimeWithTimeskip(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        async def submit_timeskip_for_initial_render(
            self,
            *,
            instruction: str,
            active_save_id: object,
        ) -> object:
            self.called = True
            return SimpleNamespace(
                model=_chat_model("Dawn catches on the city gates."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="timeskip-1",
                narrator_message_id="narrator-1",
            )

    runtime = RuntimeWithTimeskip()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/chat/timeskip",
            json={"instruction": "   ", "save_id": "save-1"},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Timeskip instruction is required"}
    assert runtime.called is False


def test_timeskip_post_rejects_same_save_active_chat_turn(tmp_path: Path) -> None:
    class RuntimeWithTimeskip(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.submissions: list[str] = []

        async def submit_timeskip_for_initial_render(
            self,
            *,
            instruction: str,
            active_save_id: object,
        ) -> object:
            self.submissions.append(instruction)
            return SimpleNamespace(
                model=_chat_model("Dawn catches on the city gates."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="timeskip-1",
                narrator_message_id="narrator-1",
            )

    runtime = RuntimeWithTimeskip()
    state = _state_double(tmp_path, runtime)
    state.jobs._jobs = {  # noqa: SLF001 - controlled registry fixture
        "chat-1": JobRecord(
            id="chat-1",
            type="chat_turn",
            status="running",
            save_id="save-1",
        )
    }

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/chat/timeskip",
            json={
                "instruction": "Skip to dawn at the city gates.",
                "save_id": "save-1",
            },
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A chat turn is already being processed for this save."
    }
    assert runtime.submissions == []


def test_job_creating_endpoint_returns_429_when_registry_is_full(
    tmp_path: Path,
) -> None:
    state = _state_double(tmp_path)
    state.jobs = JobRegistry(JobRegistryLimits(max_active_jobs=1))
    active = JobRecord(id="busy-1", type="chat_turn", status="running")
    state.jobs._jobs = {active.id: active}  # noqa: SLF001 - controlled fixture

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/world-data/context-cleanup",
            json={"save_id": "save-1"},
        )

    assert response.status_code == 429
    assert response.json() == {
        "detail": (
            "Too many active jobs; wait for one to finish before starting another."
        )
    }


def test_save_scoped_jobs_do_not_block_other_save_jobs(tmp_path: Path) -> None:
    class BlockingRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.first_entered = threading.Event()
            self.release_first = threading.Event()
            self.calls: list[str] = []

        async def run_context_cleanup(
            self,
            *,
            active_save_id: str | None = None,
        ) -> dict[str, object]:
            self.calls.append(active_save_id or "default")
            if active_save_id == "first":
                self.first_entered.set()
                await asyncio.to_thread(self.release_first.wait)
            return _chat_model(f"Cleaned {active_save_id}")

    runtime = BlockingRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        first = client.post(
            "/api/world-data/context-cleanup",
            json={"save_id": "first"},
        )
        assert first.status_code == 200
        first_job_id = first.json()["id"]
        assert runtime.first_entered.wait(timeout=1.0)

        second = client.post(
            "/api/world-data/context-cleanup",
            json={"save_id": "second"},
        )
        assert second.status_code == 200
        second_job_id = second.json()["id"]

        time.sleep(0.05)
        second_while_blocked = client.get(_job_url(second_job_id, "second")).json()
        runtime.release_first.set()

        for _ in range(50):
            first_job = client.get(_job_url(first_job_id, "first")).json()
            second_job = client.get(_job_url(second_job_id, "second")).json()
            if (
                first_job["status"] == "succeeded"
                and second_job["status"] == "succeeded"
            ):
                break
            time.sleep(0.01)

    assert second_while_blocked["status"] == "succeeded"
    assert runtime.calls == ["first", "second"]
    assert first_job["status"] == "succeeded"
    assert second_job["status"] == "succeeded"


# Regression: loading Save B while Save A work runs is intentional. Do not
# reintroduce the broad runtime lock for save-scoped background jobs.
def test_save_load_returns_while_save_scoped_job_is_running(tmp_path: Path) -> None:
    class BlockingRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.cleanup_entered = threading.Event()
            self.release_cleanup = threading.Event()
            self.loaded: list[str] = []

        async def run_context_cleanup(
            self,
            *,
            active_save_id: str | None = None,
        ) -> dict[str, object]:
            self.cleanup_entered.set()
            await asyncio.to_thread(self.release_cleanup.wait)
            return _chat_model("Cleaned context.")

        def load_save(self, save_id: str) -> dict[str, object]:
            self.loaded.append(save_id)
            return {"active_save_id": save_id, "active_save_title": "Lantern Keep"}

    runtime = BlockingRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        cleanup = client.post(
            "/api/world-data/context-cleanup",
            json={"save_id": "save-1"},
        )
        assert cleanup.status_code == 200
        job_id = cleanup.json()["id"]
        assert runtime.cleanup_entered.wait(timeout=1.0)

        loaded = client.post("/api/saves/save-2/load")
        assert loaded.status_code == 200
        assert loaded.json()["active_save_id"] == "save-2"
        assert runtime.loaded == []
        runtime.release_cleanup.set()

        for _ in range(50):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"


def test_chat_cancel_is_not_blocked_by_runtime_access_lock(tmp_path: Path) -> None:
    class BlockingChatRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.chat_entered = threading.Event()
            self.release_chat = threading.Event()
            self.cancelled = False

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
        ) -> SimpleNamespace:
            self.chat_entered.set()
            await asyncio.to_thread(self.release_chat.wait)
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

        def cancel_active_submit(self, *, save_id: str | None = None) -> bool:
            self.cancelled = True
            return True

    runtime = BlockingChatRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        chat = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert chat.status_code == 200
        job_id = chat.json()["id"]
        assert runtime.chat_entered.wait(timeout=1.0)

        cancelled = client.post("/api/chat/cancel", json={"save_id": "save-1"})
        runtime.release_chat.set()

        for _ in range(50):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)

    assert cancelled.status_code == 200
    assert cancelled.json() == {"cancelled": True}
    assert runtime.cancelled is True
    assert job["status"] == "succeeded"


def test_waiting_sse_stream_emits_event_that_woke_wait(tmp_path: Path) -> None:
    async def run_test() -> None:
        state = _state_double(tmp_path)
        release = asyncio.Event()

        async def worker(handle: Any) -> dict[str, bool]:
            await handle.event("progress", {"label": "Already buffered"})
            await release.wait()
            await handle.event("progress", {"label": "Woke waiting stream"})
            return {"ok": True}

        record = await state.jobs.create("stream_wait", worker)
        assert record.task is not None
        for _ in range(100):
            snapshot = state.jobs.get(record.id)
            if snapshot is not None and any(
                event["payload"] == {"label": "Already buffered"}
                for event in snapshot.events
            ):
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("job did not emit the buffered progress event")

        stream = cast(
            AsyncGenerator[str, None],
            api_app._event_stream(  # noqa: SLF001 - SSE regression
                cast(WebAppState, state),
                record.id,
            ),
        )
        buffered_chunks: list[str] = []
        while not any("Already buffered" in chunk for chunk in buffered_chunks):
            buffered_chunks.append(await asyncio.wait_for(anext(stream), timeout=1.0))

        waiting_chunk: asyncio.Future[str] = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0)
        release.set()
        woke_chunk = await asyncio.wait_for(waiting_chunk, timeout=1.0)
        await record.task
        await stream.aclose()

        assert "event: progress" in woke_chunk
        assert "Woke waiting stream" in woke_chunk

    asyncio.run(run_test())


def test_generic_chat_job_cancel_records_cancelled_status(tmp_path: Path) -> None:
    class CancelledChatRuntime(_RuntimeDouble):
        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
        ) -> SimpleNamespace:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                model = _chat_model("Cancelled before narrator response.")
                model["error"] = "Chat turn cancelled"
                return SimpleNamespace(
                    model=model,
                    has_post_turn_jobs=False,
                    save_id="save-1",
                    player_message_id=None,
                    narrator_message_id=None,
                )
            raise AssertionError("chat job was not cancelled")

    state = _state_double(tmp_path, CancelledChatRuntime())
    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        for _ in range(20):
            running = client.get(_job_url(job_id, "save-1")).json()
            if running["status"] == "running":
                break
            time.sleep(0.01)
        else:
            raise AssertionError("chat job did not start running")

        cancelled = client.post(_job_cancel_url(job_id, "save-1"))
        assert cancelled.status_code == 200
        assert cancelled.json() == {"cancelled": True}

        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.01)

    assert job["status"] == "cancelled"
    assert job["error"] == "Cancelled"


def test_chat_turn_completes_after_initial_render_and_queues_post_turn_job(
    tmp_path: Path,
) -> None:
    class PostTurnRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.post_turn_started = threading.Event()
            self.release_post_turn = threading.Event()

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=True,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
            progress_callback: object | None = None,
        ) -> dict[str, object]:
            self.post_turn_started.set()
            callback = cast(Callable[[object], None], progress_callback)
            callback(
                {
                    "jobs": [
                        {"name": "state", "status": "running"},
                        {"name": "context", "status": "pending"},
                    ],
                }
            )
            callback(
                {
                    "jobs": [
                        {"name": "state", "status": "complete"},
                        {"name": "context", "status": "running"},
                    ],
                }
            )
            await asyncio.to_thread(self.release_post_turn.wait)
            return _chat_model("The world settles.")

    runtime = PostTurnRuntime()
    state = _state_double(tmp_path, runtime)
    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        assert runtime.post_turn_started.wait(timeout=2)
        for _ in range(50):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)
        active_jobs = client.get("/api/jobs?status=active").json()["jobs"]
        post_turn_active_jobs = [
            active_job
            for active_job in active_jobs
            if active_job["type"] == "post_turn_background"
        ]
        assert len(post_turn_active_jobs) == 1
        assert post_turn_active_jobs[0]["status"] == "running"
        post_turn_progress = post_turn_active_jobs[0]["latest_progress"]
        post_turn_job_id = post_turn_active_jobs[0]["id"]
        runtime.release_post_turn.set()
        for _ in range(50):
            post_turn_job = client.get(_job_url(post_turn_job_id, "save-1")).json()
            if post_turn_job["status"] == "succeeded":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert job["result"]["chronicle"]["messages"][0]["body"] == "The bell answers."
    assert post_turn_progress["jobs"][0] == {
        "name": "state",
        "status": "complete",
    }
    assert post_turn_progress["jobs"][1] == {
        "name": "context",
        "status": "running",
    }
    assert post_turn_job["status"] == "succeeded"


def test_chat_turn_job_returns_delta_for_initial_render(tmp_path: Path) -> None:
    class DeltaProvider:
        provider_name = "fake"

        async def chat(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(
                body="The bell answers.",
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 12},
            )

        async def generate_structured_output(
            self,
            request: StructuredOutputRequest,
        ) -> StructuredOutputResponse:
            if request.schema_name == "content_safety_review":
                return StructuredOutputResponse(
                    data={
                        "action": "allow",
                        "category": "none",
                        "reason": "Test fixture content is within the ceiling.",
                        "minimum_rating": "g",
                    },
                    provider=request.provider,
                    model_id=request.model_id,
                    token_usage={"total": 3},
                )
            return StructuredOutputResponse(
                data={"selections": []},
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 3},
            )

    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(database_path, check_same_thread=False)
    )
    save = _create_auth_save(repositories, title="Lantern Save", owner_user_id=None)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat", "structured_output"],
        context_window=32768,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    provider = cast(ProviderClient, DeltaProvider())
    runtime = BragiRuntime(
        repositories=repositories,
        providers={"fake": provider},
        media_dir=tmp_path / "media",
        active_save_id=save.id,
    )
    state = _state_double(tmp_path, runtime)
    state.repositories = repositories
    state.providers = runtime.providers

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": save.id},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        for _ in range(100):
            job_record = state.jobs.get(job_id)
            if job_record is not None and job_record.status in {
                "succeeded",
                "failed",
                "cancelled",
            }:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("chat turn job did not finish")

    assert job_record is not None
    assert job_record.status == "succeeded"
    result = cast(dict[str, Any], job_record.result)
    assert result["kind"] == "chat_turn_delta"
    assert result["version"] == 1
    assert result["save_id"] == save.id
    assert result["status"] == "Turn complete"
    assert "chronicle" not in result
    assert "saves" not in result
    assert [message["role"] for message in result["messages"]] == [
        "player",
        "narrator",
    ]
    assert [message["body"] for message in result["messages"]] == [
        "Light the beacon",
        "The bell answers.",
    ]
    assert result["player_message_id"] == result["messages"][0]["message_id"]
    assert result["narrator_message_id"] == result["messages"][1]["message_id"]
    assert result["save"]["save_id"] == save.id


def test_post_turn_jobs_expose_initial_phase_progress_before_runtime_callback(
    tmp_path: Path,
) -> None:
    class PostTurnRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.post_turn_started = threading.Event()
            self.release_post_turn = threading.Event()

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=True,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
            progress_callback: object | None = None,
        ) -> dict[str, object]:
            self.post_turn_started.set()
            await asyncio.to_thread(self.release_post_turn.wait)
            return _chat_model("The world settles.")

    runtime = PostTurnRuntime()
    state = _state_double(tmp_path, runtime)
    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert created.status_code == 200

        assert runtime.post_turn_started.wait(timeout=2)
        for _ in range(50):
            active_jobs = client.get("/api/jobs?status=active").json()["jobs"]
            post_turn_jobs = [
                active_job
                for active_job in active_jobs
                if active_job["type"] == "post_turn_background"
            ]
            if post_turn_jobs:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("post-turn job was not queued")
        job = client.get(_job_url(post_turn_jobs[0]["id"], "save-1")).json()
        runtime.release_post_turn.set()
        for _ in range(20):
            finished = client.get(_job_url(post_turn_jobs[0]["id"], "save-1")).json()
            if finished["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "running"
    assert job["latest_progress"] == {
        "status_text": "Updating world state",
        "jobs": [
            {"name": "state", "status": "pending"},
            {"name": "context", "status": "pending"},
            {"name": "proactive_text", "status": "pending"},
            {"name": "director", "status": "pending"},
            {"name": "scenario", "status": "pending"},
            {"name": "image", "status": "pending"},
        ],
    }
    assert finished["status"] == "succeeded"


def test_chat_turn_waits_for_background_post_turn_after_input_progress(
    tmp_path: Path,
) -> None:
    class PostTurnCatchupRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.submit_count = 0
            self.post_turn_started = threading.Event()
            self.release_post_turn = threading.Event()
            self.second_input_saved = threading.Event()
            self.second_catchup_done = threading.Event()

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
            turn_progress_callback: Callable[[object], None] | None = None,
            post_input_catchup: Callable[[], Awaitable[Any]] | None = None,
        ) -> SimpleNamespace:
            del speaker_name
            self.submit_count += 1
            if self.submit_count == 1:
                return SimpleNamespace(
                    model=_chat_model("The bell answers."),
                    has_post_turn_jobs=True,
                    save_id="save-1",
                    player_message_id="player-1",
                    narrator_message_id="narrator-1",
                )

            assert active_save_id == "save-1"
            assert body == "I check the lens while the world settles."
            assert turn_progress_callback is not None
            assert post_input_catchup is not None
            turn_progress_callback(
                _expected_chat_turn_progress(
                    "Player input saved",
                    succeeded=("submission", "history", "input"),
                )
            )
            self.second_input_saved.set()
            await post_input_catchup()
            self.second_catchup_done.set()
            return SimpleNamespace(
                model=_chat_model("The lens keeps humming."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="player-2",
                narrator_message_id="narrator-2",
            )

        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
            progress_callback: object | None = None,
        ) -> dict[str, object]:
            del save_id, player_message_id, narrator_message_id, progress_callback
            self.post_turn_started.set()
            await asyncio.to_thread(self.release_post_turn.wait)
            return _chat_model("The world settles.")

    runtime = PostTurnCatchupRuntime()
    state = _state_double(tmp_path, runtime)
    with TestClient(create_app(cast(WebAppState, state))) as client:
        first = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert first.status_code == 200
        assert runtime.post_turn_started.wait(timeout=2)
        first_job = _wait_for_terminal_job(
            client,
            first.json()["id"],
            save_id="save-1",
        )

        second = client.post(
            "/api/chat",
            json={
                "body": "I check the lens while the world settles.",
                "save_id": "save-1",
            },
        )
        assert second.status_code == 200
        second_job_id = second.json()["id"]
        assert runtime.second_input_saved.wait(timeout=2)
        assert not runtime.second_catchup_done.is_set()
        for _ in range(50):
            second_running = client.get(_job_url(second_job_id, "save-1")).json()
            if second_running.get("latest_progress", {}).get("status_text") == (
                "Player input saved"
            ):
                break
            time.sleep(0.01)
        runtime.release_post_turn.set()
        second_finished = _wait_for_terminal_job(
            client,
            second_job_id,
            save_id="save-1",
        )

    assert first_job["status"] == "succeeded"
    assert second_running["status"] == "running"
    assert second_running["latest_progress"] == _expected_chat_turn_progress(
        "Player input saved",
        succeeded=("submission", "history", "input"),
    )
    assert runtime.second_catchup_done.is_set()
    assert second_finished["status"] == "succeeded"


def test_chat_turn_leaves_state_pruning_for_scheduler(tmp_path: Path) -> None:
    class PostTurnRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            self.active_save_id = "save-1"
            self.state_pruning_calls: list[str] = []

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=True,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
            progress_callback: object | None = None,
        ) -> dict[str, object]:
            return _chat_model("The world settles.")

        async def run_state_pruning(
            self,
            *,
            active_save_id: str,
        ) -> dict[str, object]:
            self.state_pruning_calls.append(active_save_id)
            await asyncio.sleep(60)
            return _chat_model("World state cleaned.")

    runtime = PostTurnRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories = _repositories_with_state_pruning_due(tmp_path)
    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert created.status_code == 200
        chat_job_id = created.json()["id"]
        for _ in range(20):
            chat_job = client.get(_job_url(chat_job_id, "save-1")).json()
            if chat_job["status"] == "succeeded":
                break
            time.sleep(0.01)
        active_jobs = client.get("/api/jobs?status=active").json()["jobs"]

    assert chat_job["status"] == "succeeded"
    assert [job for job in active_jobs if job["type"] == "state_pruning"] == []
    assert runtime.state_pruning_calls == []


def test_chat_turn_skips_duplicate_active_state_pruning_job(tmp_path: Path) -> None:
    class PostTurnRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            self.active_save_id = "save-1"
            self.state_pruning_calls = 0

        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=True,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
            progress_callback: object | None = None,
        ) -> dict[str, object]:
            return _chat_model("The world settles.")

        async def run_state_pruning(
            self,
            *,
            active_save_id: str,
        ) -> dict[str, object]:
            self.state_pruning_calls += 1
            return _chat_model("Unexpected cleanup.")

    runtime = PostTurnRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories = _repositories_with_state_pruning_due(tmp_path)
    state.jobs._jobs = {  # noqa: SLF001 - controlled active-job fixture
        "cleanup-1": JobRecord(
            id="cleanup-1",
            type="state_pruning",
            save_id="save-1",
            exclusive_key="state_pruning:save-1",
            status="running",
        )
    }
    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert created.status_code == 200
        chat_job_id = created.json()["id"]
        for _ in range(20):
            chat_job = client.get(_job_url(chat_job_id, "save-1")).json()
            if chat_job["status"] == "succeeded":
                break
            time.sleep(0.01)
        active_jobs = client.get("/api/jobs?status=active").json()["jobs"]

    assert chat_job["status"] == "succeeded"
    assert [job["id"] for job in active_jobs if job["type"] == "state_pruning"] == [
        "cleanup-1"
    ]
    assert runtime.state_pruning_calls == 0


def test_provider_retry_status_text_renders_unlimited_attempts_without_max() -> None:
    from types import SimpleNamespace as ProgressNamespace

    assert api_app._provider_retry_status_text(  # noqa: SLF001
        ProgressNamespace(next_attempt=9, max_attempts=7, unlimited=True),
        "chat",
    ) == "Retrying chat request (attempt 9)..."

    assert api_app._provider_retry_status_text(  # noqa: SLF001
        ProgressNamespace(next_attempt=2, max_attempts=3, unlimited=False),
        "chat",
    ) == "Retrying chat request (attempt 2 of 3)..."

    assert api_app._provider_retry_status_text(  # noqa: SLF001
        ProgressNamespace(next_attempt=2, max_attempts=3),
        "chat",
    ) == "Retrying chat request (attempt 2 of 3)..."

    assert api_app._provider_retry_status_text(  # noqa: SLF001
        ProgressNamespace(next_attempt=None, max_attempts=None, unlimited=True),
        "chat",
    ) == "Retrying chat request..."


def test_provider_retry_progress_events_are_sent_before_sse_done(
    tmp_path: Path,
) -> None:
    class RetryRuntime(_RuntimeDouble):
        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
            retry_progress_callback: Callable[[object], None] | None = None,
        ) -> SimpleNamespace:
            assert body == "Light the beacon"
            if retry_progress_callback is not None:
                retry_progress_callback(
                    SimpleNamespace(
                        next_attempt=2,
                        max_attempts=3,
                    )
                )
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

    state = _state_double(tmp_path, RetryRuntime())
    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"

    async def collect_events() -> list[str]:
        return [
            chunk
            async for chunk in api_app._event_stream(  # noqa: SLF001 - SSE regression
                cast(WebAppState, state),
                job_id,
            )
        ]

    chunks = asyncio.run(collect_events())
    event_names = [chunk.split("\n", 1)[0] for chunk in chunks]
    retry_index = next(
        index
        for index, chunk in enumerate(chunks)
        if "Retrying chat request (attempt 2 of 3)..." in chunk
    )

    assert event_names[retry_index] == "event: progress"
    assert retry_index < event_names.index("event: done")


def test_chat_turn_emits_narrator_draft_before_runtime(tmp_path: Path) -> None:
    class StreamingRuntime(_RuntimeDouble):
        async def submit_player_message_for_initial_render(
            self,
            *,
            body: str,
            speaker_name: str | None,
            active_save_id: object,
            narrator_stream_callback: Callable[[str], None] | None = None,
        ) -> SimpleNamespace:
            assert body == "Light the beacon"
            if narrator_stream_callback is not None:
                narrator_stream_callback("The bell")
                narrator_stream_callback("The bell answers.")
            return SimpleNamespace(
                model=_chat_model("The bell answers."),
                has_post_turn_jobs=False,
                save_id="save-1",
                player_message_id="player-1",
                narrator_message_id="narrator-1",
            )

    state = _state_double(tmp_path, StreamingRuntime())
    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat",
            json={"body": "Light the beacon", "save_id": "save-1"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    snapshot = state.jobs.get(job_id)
    assert snapshot is not None
    event_names = [event["event"] for event in snapshot.events]
    drafts = [
        event["payload"]["message"]["body"]
        for event in snapshot.events
        if event["event"] == "narrator_draft"
    ]
    assert drafts == ["The bell", "The bell answers."]
    assert event_names.index("narrator_draft") < event_names.index("runtime")


def test_client_log_endpoint_sanitizes_sensitive_metadata(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    clear_recent_events()

    with TestClient(create_app()) as client:
        logged = client.post(
            "/api/log/client",
            json={
                "level": "error",
                "event": "client.api.failed",
                "fields": {
                    "component": "Composer",
                    "body": "secret chat body",
                    "api_key": "secret-key",
                    "error_message": "bad token=secret-token",
                },
            },
        )
        diagnostics = client.get("/api/diagnostics")
        settings = client.get("/api/settings")

    assert logged.status_code == 200
    assert logged.json() == {"ok": True}
    assert diagnostics.status_code == 200
    payload_text = diagnostics.text
    assert "web_events" in diagnostics.json()
    assert "web_events" not in settings.json()
    assert "secret chat body" not in payload_text
    assert "secret-key" not in payload_text
    assert "secret-token" not in payload_text
    assert "Composer" in payload_text


def test_client_log_endpoint_drops_reserved_client_metadata(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    clear_recent_events()

    with TestClient(create_app()) as client:
        logged = client.post(
            "/api/log/client",
            json={
                "level": "error",
                "event": "client.api.failed",
                "fields": {
                    "component": "Composer",
                    "event": "spoofed.event",
                    "level": "debug",
                    "source": "browser",
                    "timestamp": "1970-01-01T00:00:00Z",
                },
            },
        )
        diagnostics = client.get("/api/diagnostics")

    assert logged.status_code == 200
    assert logged.json() == {"ok": True}
    event = next(
        item
        for item in diagnostics.json()["web_events"]
        if item["event"] == "client.api.failed"
    )
    assert event["event"] == "client.api.failed"
    assert event["level"] == "error"
    assert event["source"] == "client"
    assert event["component"] == "Composer"
    assert event["timestamp"] != "1970-01-01T00:00:00Z"
    assert "spoofed.event" not in diagnostics.text


def test_settings_do_not_embed_diagnostics_payloads(tmp_path: Path) -> None:
    db_path = tmp_path / "bragi.sqlite3"
    migrate_database(db_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(db_path, check_same_thread=False)
    )
    failed = repositories.create_job(
        type="state_pruning",
        status="running",
        payload={},
    )
    repositories.update_job(
        failed.id,
        status="failed",
        error="provider timed out",
    )
    state = _state_double(tmp_path)
    state.repositories = repositories
    clear_recent_events()

    with TestClient(create_app(cast(WebAppState, state))) as client:
        client.post(
            "/api/log/client",
            json={
                "level": "error",
                "event": "client.api.failed",
                "fields": {"component": "Composer"},
            },
        )
        response = client.get("/api/settings")

    assert response.status_code == 200
    payload = response.json()
    assert "diagnostics" not in payload
    assert "maintenance_jobs" not in payload
    assert "runtime_performance" not in payload
    assert "web_events" not in payload


def test_diagnostics_endpoint_omits_deprecated_fallback_disabled_warning(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "bragi.sqlite3"
    migrate_database(db_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(db_path, check_same_thread=False)
    )
    repositories.set_model_preference(
        task="structured_output_fallback",
        provider="openrouter",
        model_id="openrouter/context-search",
    )
    repositories.set_app_setting("structured_output_fallback_enabled", False)
    state = _state_double(tmp_path)
    state.repositories = repositories

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/diagnostics?category=signals")

    assert response.status_code == 200
    assert not any(
        signal["kind"] == "configuration"
        and "Structured output fallback model is configured" in signal["error"]
        and "structured_output_fallback" in signal["error"]
        and "openrouter/openrouter/context-search" in signal["error"]
        and "fallback is disabled" in signal["error"]
        for signal in response.json()["signals"]
    )


def test_diagnostics_endpoint_includes_scheduler_and_active_save_health(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "bragi.sqlite3"
    migrate_database(db_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(db_path, check_same_thread=False)
    )
    save = _create_auth_save(
        repositories,
        title="Lantern Keep",
        owner_user_id=None,
    )
    task = repositories.upsert_scheduled_task(
        task_type="world_suggestion_review",
        save_id=save.id,
        interval_seconds=60,
        payload={"active_save_only": True},
        due_now=True,
    )
    repositories.complete_scheduled_task(
        task.id,
        succeeded=False,
        error="review provider unavailable with sk-secret",
        next_run_after_seconds=120,
    )
    jobs = JobLifecycleService(repositories=repositories)
    chat = jobs.create_running(
        save_id=save.id,
        type="chat_completion",
        payload={},
    )
    jobs.succeed(
        chat.id,
        result={
            "prompt_context_diagnostics": {
                "baseline_recent_message_count": 12,
                "baseline_recent_message_chars": 40_000,
                "retrieved_counts": {"state": 2, "memories": 1},
                "final_prompt_budget": {
                    "input_limit_tokens": 8_000,
                    "estimated_tokens_after": 5_400,
                    "trimmed": True,
                    "raw_prompt_text": "unshared prompt text",
                },
                "raw_notes": "unshared prompt text",
            }
        },
    )
    state = _state_double(tmp_path)
    state.repositories = repositories

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get(f"/api/diagnostics?save_id={save.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_save_health"]["save_id"] == save.id
    assert (
        payload["active_save_health"]["latest_chat_prompt"][
            "baseline_recent_message_chars"
        ]
        == 40_000
    )
    assert (
        payload["active_save_health"]["latest_chat_prompt"]["final_prompt_budget"][
            "input_limit_tokens"
        ]
        == 8_000
    )
    assert "raw_prompt_text" not in response.text
    assert "raw_notes" not in response.text
    assert "unshared prompt text" not in response.text
    scheduler = payload["scheduler_health"]
    assert scheduler["summary"]["failed"] == 1
    assert scheduler["tasks"] == [
        {
            "task_id": task.id,
            "task_type": "world_suggestion_review",
            "save_id": save.id,
            "status": "failed",
            "enabled": True,
            "interval_seconds": 60,
            "next_run_at": scheduler["tasks"][0]["next_run_at"],
            "lease_until": None,
            "last_started_at": None,
            "last_completed_at": scheduler["tasks"][0]["last_completed_at"],
            "last_job_id": None,
            "failure_count": 1,
            "error": "review provider unavailable with [redacted]",
            "skip_reason": None,
        }
    ]


def test_diagnostics_endpoint_suppresses_curation_scheduler_error_text(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "bragi.sqlite3"
    migrate_database(db_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(db_path, check_same_thread=False)
    )
    save = _create_auth_save(
        repositories,
        title="Lantern Keep",
        owner_user_id=None,
    )
    task = repositories.upsert_scheduled_task(
        task_type="observation_curation_drain",
        save_id=save.id,
        interval_seconds=60,
        payload={"active_save_only": False},
        due_now=True,
    )
    repositories.complete_scheduled_task(
        task.id,
        succeeded=False,
        error="Private chronicle detail escaped from a provider.",
        next_run_after_seconds=120,
    )
    state = _state_double(tmp_path)
    state.repositories = repositories

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get(f"/api/diagnostics?save_id={save.id}")

    assert response.status_code == 200
    [diagnostic] = response.json()["scheduler_health"]["tasks"]
    assert diagnostic["status"] == "failed"
    assert diagnostic["error"] is None
    assert "Private chronicle detail" not in response.text


def test_diagnostics_endpoint_scopes_non_admin_to_save_health(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    admin = state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    user = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    user_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=user.id,
    )
    other_save = _create_auth_save(
        state.repositories,
        title="Admin Save",
        owner_user_id=admin.id,
    )
    child_save = _create_auth_save(
        state.repositories,
        title="Child Save",
        owner_user_id=admin.id,
    )
    state.repositories.grant_save_access(save_id=child_save.id, user_id=child.id)
    failed = state.repositories.create_job(
        type="state_pruning",
        status="running",
        payload={},
    )
    state.repositories.update_job(
        failed.id,
        status="failed",
        error="admin-only diagnostic",
    )
    app = create_app(cast(WebAppState, state))

    with (
        TestClient(app, authenticate=False) as user_client,
        TestClient(app, authenticate=False) as child_client,
    ):
        assert user_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        user_response = user_client.get(f"/api/diagnostics?save_id={user_save.id}")
        user_other_save = user_client.get(
            f"/api/diagnostics?save_id={other_save.id}"
        )
        assert child_client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        child_response = child_client.get(
            f"/api/diagnostics?save_id={child_save.id}"
        )

    assert user_response.status_code == 200
    payload = user_response.json()
    assert payload["active_save_health"]["save_id"] == user_save.id
    assert payload["signals"] == []
    assert payload["maintenance_jobs"] == []
    assert payload["runtime_performance"] is None
    assert payload["scheduler_health"]["tasks"] == []
    assert payload["web_events"] == []
    assert "admin-only diagnostic" not in user_response.text
    assert user_other_save.status_code == 404
    assert child_response.status_code == 403


def test_diagnostics_endpoint_includes_maintenance_job_batch_diagnostics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "bragi.sqlite3"
    migrate_database(db_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(db_path, check_same_thread=False)
    )
    queued = repositories.create_job(
        type="state_pruning",
        status="queued",
        payload={},
    )
    running = repositories.start_job(queued.id)
    failed = repositories.update_job(
        running.id,
        status="failed",
        error="provider timed out",
        result={
            "active_state_count": 120,
            "batch_count": 3,
            "completed_batch_count": 1,
            "batch_size": 40,
            "failed_batch_index": 1,
            "archived_count": 8,
            "rejected_count": 2,
            "proposals": [{"reason": "secret state detail"}],
        },
    )
    state = _state_double(tmp_path)
    state.repositories = repositories
    state.log_file_path = None

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["maintenance_jobs"] == [
        {
            "job_id": failed.id,
            "job_type": "state_pruning",
            "status": "failed",
            "save_id": None,
            "error": "provider timed out",
            "started_at": failed.started_at,
            "completed_at": failed.completed_at,
            "summary": "1/3 batches, 8 archived, 2 rejected, failed at batch 2",
            "metrics": {
                "active_state_count": 120,
                "batch_count": 3,
                "completed_batch_count": 1,
                "batch_size": 40,
                "failed_batch_index": 1,
                "archived_count": 8,
                "rejected_count": 2,
            },
        }
    ]
    assert "secret state detail" not in response.text
    assert "proposals" not in response.text


def test_diagnostics_endpoint_includes_runtime_performance(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "bragi.sqlite3"
    migrate_database(db_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(db_path, check_same_thread=False)
    )
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
        error="provider timed out",
    )
    repositories.connection.execute(
        "UPDATE jobs SET duration_ms = 120 WHERE id = ?",
        (succeeded.id,),
    )
    repositories.connection.execute(
        "UPDATE jobs SET duration_ms = 999 WHERE id = ?",
        (failed.id,),
    )
    repositories.record_job_step(
        job_id=succeeded.id,
        name="provider.chat",
        status="succeeded",
        provider="fake",
        model="fake-chat",
        task="chat",
        duration_ms=80,
    )
    state = _state_double(tmp_path)
    state.repositories = repositories

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/diagnostics")

    assert response.status_code == 200
    performance = response.json()["runtime_performance"]
    assert performance["job_averages"][0] == {
        "job_type": "chat_turn",
        "step_name": None,
        "provider": None,
        "model": None,
        "task": None,
        "sample_count": 2,
        "success_count": 1,
        "failed_count": 1,
        "cancelled_count": 0,
        "skipped_count": 0,
        "average_duration_ms": 120,
        "p50_duration_ms": 120,
        "p95_duration_ms": 120,
        "min_duration_ms": 120,
        "max_duration_ms": 120,
        "latest_duration_ms": 999,
        "average_queue_wait_ms": 0,
        "p95_queue_wait_ms": 0,
        "failure_rate": 0.5,
        "latest_completed_at": failed.completed_at,
    }
    assert performance["model_averages"][0]["provider"] == "fake"
    assert performance["model_averages"][0]["model"] == "fake-chat"
    assert performance["model_averages"][0]["average_duration_ms"] == 80
    assert performance["slowest_recent"][0]["job_id"] == failed.id
    assert performance["window_started_at"]


def test_diagnostics_endpoint_includes_guided_cleanup_job_diagnostics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "bragi.sqlite3"
    migrate_database(db_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(db_path, check_same_thread=False)
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Lantern Keep",
    )
    queued = repositories.create_job(
        type="guided_context_cleanup",
        status="queued",
        save_id=save.id,
        payload={"instruction": "secret cleanup instruction"},
    )
    running = repositories.start_job(queued.id)
    failed = repositories.update_job(
        running.id,
        status="failed",
        error="provider timed out",
        result={
            "cleanup_target_count": 12,
            "action_batches": 2,
            "completed_action_batches": 1,
            "proposed_actions": 5,
            "queued_suggestions": 3,
            "rejected_actions": 1,
            "suggestion_ids": ["suggestion-secret"],
        },
    )
    state = _state_double(tmp_path)
    state.repositories = repositories
    state.log_file_path = None

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["maintenance_jobs"] == [
        {
            "job_id": failed.id,
            "job_type": "guided_context_cleanup",
            "status": "failed",
            "save_id": save.id,
            "error": "provider timed out",
            "started_at": failed.started_at,
            "completed_at": failed.completed_at,
            "summary": "1/2 action batches, 3 queued, 1 rejected",
            "metrics": {
                "cleanup_target_count": 12,
                "action_batches": 2,
                "completed_action_batches": 1,
                "proposed_actions": 5,
                "queued_suggestions": 3,
                "rejected_actions": 1,
            },
        }
    ]
    assert "secret cleanup instruction" not in response.text
    assert "suggestion-secret" not in response.text


def test_diagnostics_endpoint_includes_world_context_retention_job_diagnostics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "bragi.sqlite3"
    migrate_database(db_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(db_path, check_same_thread=False)
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Lantern Keep",
    )
    queued = repositories.create_job(
        type="world_context_retention",
        status="queued",
        save_id=save.id,
        payload={"save_title": "secret save context"},
    )
    running = repositories.start_job(queued.id)
    failed = repositories.update_job(
        running.id,
        status="failed",
        error="retention failed",
        result={
            "expired_stale_suggestions": 2,
            "expired_excess_suggestions": 1,
            "pruned_archived_rows": {"world_state": 4, "memories": 0},
            "pruned_audit_rows": 7,
            "pruned_terminal_jobs": 3,
            "suggestion_ids": ["secret-suggestion"],
        },
    )
    state = _state_double(tmp_path)
    state.repositories = repositories
    state.log_file_path = None

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["maintenance_jobs"] == [
        {
            "job_id": failed.id,
            "job_type": "world_context_retention",
            "status": "failed",
            "save_id": save.id,
            "error": "retention failed",
            "started_at": failed.started_at,
            "completed_at": failed.completed_at,
            "summary": (
                "2 stale suggestions expired, 1 excess suggestion expired, "
                "4 archived rows pruned, 7 audit rows pruned, "
                "3 terminal jobs pruned"
            ),
            "metrics": {
                "expired_stale_suggestions": 2,
                "expired_excess_suggestions": 1,
                "pruned_archived_rows": {"world_state": 4, "memories": 0},
                "pruned_audit_rows": 7,
                "pruned_terminal_jobs": 3,
            },
        }
    ]
    assert "secret save context" not in response.text
    assert "secret-suggestion" not in response.text


def test_diagnostics_endpoint_includes_post_turn_outcome_failure_diagnostics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "bragi.sqlite3"
    migrate_database(db_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(db_path, check_same_thread=False)
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Lantern Keep",
    )
    queued = repositories.create_job(
        type="post_turn_jobs",
        status="queued",
        save_id=save.id,
        payload={
            "player_message_id": "secret-player-message",
            "narrator_message_id": "secret-narrator-message",
        },
    )
    running = repositories.start_job(queued.id)
    failed = repositories.update_job(
        running.id,
        status="failed",
        error=(
            "Scenario outcome evaluation failed "
            "(provider=fake/fake-outcome, fallback_skipped_reason=disabled)"
        ),
        result={
            "jobs": [
                {"name": "state", "status": "succeeded"},
                {
                    "name": "outcome",
                    "status": "failed",
                    "result": {
                        "provider": "fake",
                        "model": "fake-outcome",
                        "error": "provider_error",
                        "error_category": "provider_error",
                        "fallback_attempted": False,
                        "fallback_skipped_reason": "disabled",
                    },
                },
                {"name": "characters", "status": "skipped"},
            ],
        },
    )
    state = _state_double(tmp_path)
    state.repositories = repositories
    state.log_file_path = None

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["maintenance_jobs"] == [
        {
            "job_id": failed.id,
            "job_type": "post_turn_jobs",
            "status": "failed",
            "save_id": save.id,
            "error": failed.error,
            "started_at": failed.started_at,
            "completed_at": failed.completed_at,
            "summary": (
                "outcome failed, characters skipped, "
                "outcome provider fake/fake-outcome, "
                "outcome error provider_error, outcome fallback skipped disabled"
            ),
            "metrics": {
                "failed_steps": ["outcome"],
                "skipped_steps": ["characters"],
                "outcome_provider": "fake",
                "outcome_model": "fake-outcome",
                "outcome_error": "provider_error",
                "outcome_error_category": "provider_error",
                "outcome_fallback_attempted": False,
                "outcome_fallback_skipped_reason": "disabled",
            },
        }
    ]
    assert "secret-player-message" not in response.text
    assert "secret-narrator-message" not in response.text


def test_job_registry_logs_lifecycle_metadata_only() -> None:
    async def run_jobs() -> None:
        clear_recent_events()
        registry = JobRegistry()

        async def success_worker(handle: object) -> dict[str, object]:
            return {"created_count": 2, "body": "secret result body"}

        async def failed_worker(handle: object) -> None:
            raise RuntimeError("failed with sk-secretvalue")

        success = await registry.create("success_job", success_worker)
        failed = await registry.create("failed_job", failed_worker)
        cancelled = await registry.create(
            "cancel_job",
            lambda handle: asyncio.sleep(60),
        )
        assert cancelled.task is not None
        await asyncio.sleep(0)
        assert await registry.cancel(cancelled.id) is True
        assert success.task is not None
        assert failed.task is not None
        await asyncio.gather(success.task, failed.task, cancelled.task)

    asyncio.run(run_jobs())

    events = recent_events()
    assert {event["event"] for event in events} >= {
        "web.job.queued",
        "web.job.started",
        "web.job.succeeded",
        "web.job.failed",
        "web.job.cancel_requested",
        "web.job.cancelled",
    }
    succeeded = next(event for event in events if event["event"] == "web.job.succeeded")
    failed = next(event for event in events if event["event"] == "web.job.failed")
    assert succeeded["job_type"] == "success_job"
    assert succeeded["created_count"] == 2
    assert failed["job_type"] == "failed_job"
    assert failed["error_class"] == "RuntimeError"
    assert "secret result body" not in str(events)
    assert "sk-secretvalue" not in str(events)


def test_chat_edit_job_fails_when_runtime_returns_no_narrator(
    tmp_path: Path,
) -> None:
    class FailedEditRuntime:
        async def edit_and_resubmit_message(
            self,
            *,
            message_id: str,
            body: str,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            return {
                "active_save_id": "save-1",
                "active_save_title": "Lantern Keep",
                "chronicle": {
                    "messages": [
                        {
                            "message_id": "edited-player",
                            "role": "player",
                            "speaker_name": "Keeper",
                            "body": body,
                            "actions": [],
                        }
                    ]
                },
                "composer_enabled": True,
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / broken-chat",
                "saves": [],
            }

    state = _state_double(tmp_path, FailedEditRuntime())

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/edit",
            json={
                "message_id": "player-1",
                "body": "Hold the line",
                "save_id": "save-1",
            },
        )
        job_id = created.json()["id"]
        assert created.status_code == 200

        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "failed"
    assert job["error"] == SAFE_JOB_ERROR


def test_chat_like_job_cancel_requests_runtime_chat_cancellation(
    tmp_path: Path,
) -> None:
    class BlockingEditRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.cancel_calls: list[str | None] = []

        async def edit_and_resubmit_message(
            self,
            *,
            message_id: str,
            body: str,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            self.entered.set()
            await asyncio.to_thread(self.release.wait)
            return _chat_model("The edited turn lands.")

        def cancel_active_submit(self, *, save_id: str | None = None) -> bool:
            self.cancel_calls.append(save_id)
            return True

    runtime = BlockingEditRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/edit",
            json={
                "message_id": "player-1",
                "body": "Hold the line",
                "save_id": "save-1",
            },
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        assert runtime.entered.wait(timeout=1.0)

        cancelled = client.post(_job_cancel_url(job_id, "save-1"))
        runtime.release.set()
        job = _wait_for_terminal_job(client, job_id, save_id="save-1")

    assert cancelled.status_code == 200
    assert cancelled.json() == {"cancelled": True}
    assert runtime.cancel_calls == ["save-1"]
    assert job["status"] == "cancelled"


def test_terminal_chat_like_job_cancel_does_not_request_runtime_cancellation(
    tmp_path: Path,
) -> None:
    class RecordingCancelRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_calls: list[str | None] = []

        def cancel_active_submit(self, *, save_id: str | None = None) -> bool:
            self.cancel_calls.append(save_id)
            return True

    runtime = RecordingCancelRuntime()
    state = _state_double(tmp_path, runtime)
    state.jobs._jobs = {  # noqa: SLF001 - controlled registry fixture
        "chat-edit-done": JobRecord(
            id="chat-edit-done",
            type="chat_edit",
            save_id="save-1",
            status="succeeded",
        )
    }

    with TestClient(create_app(cast(WebAppState, state))) as client:
        cancelled = client.post(_job_cancel_url("chat-edit-done", "save-1"))

    assert cancelled.status_code == 200
    assert cancelled.json() == {"cancelled": False}
    assert runtime.cancel_calls == []


def test_chat_regenerate_passes_feedback_to_runtime(tmp_path: Path) -> None:
    class RegenerateRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, object]] = []

        async def regenerate_message(
            self,
            *,
            message_id: str,
            active_save_id: str | None | object = ...,
            regeneration_feedback: str = "",
        ) -> dict[str, object]:
            self.calls.append((message_id, regeneration_feedback, active_save_id))
            return {
                "active_save_id": "save-1",
                "active_save_title": "Lantern Keep",
                "chronicle": {
                    "messages": [
                        {
                            "message_id": "narrator-2",
                            "role": "narrator",
                            "speaker_name": None,
                            "body": "The answer lands cleaner.",
                            "actions": [],
                        }
                    ]
                },
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
            }

    runtime = RegenerateRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/regenerate",
            json={
                "message_id": "narrator-1",
                "save_id": "save-1",
                "regeneration_feedback": "Make the response terser.",
            },
        )
        job_id = created.json()["id"]
        assert created.status_code == 200
        assert created.json()["save_id"] == "save-1"

        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert runtime.calls == [("narrator-1", "Make the response terser.", "save-1")]


def test_action_choice_regenerate_passes_message_and_save_to_runtime(
    tmp_path: Path,
) -> None:
    class ActionChoiceRegenerateRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, object]] = []

        async def regenerate_action_choices(
            self,
            *,
            narrator_message_id: str,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            self.calls.append((narrator_message_id, active_save_id))
            return {
                "active_save_id": "save-2",
                "active_save_title": "Lantern Keep",
                "action_choices_enabled": True,
                "action_choices": {
                    "narrator_message_id": narrator_message_id,
                    "choices": [
                        {
                            "choice_id": "choice-1",
                            "ordinal": 0,
                            "body": "Open the brass door.",
                        }
                    ],
                },
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
                "status": "Action choices regenerated",
            }

    runtime = ActionChoiceRegenerateRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/action-choices/regenerate",
            json={"message_id": "narrator-1", "save_id": "save-2"},
        )
        assert created.status_code == 200
        assert created.json()["type"] == "action_choice_regenerate"
        assert created.json()["save_id"] == "save-2"
        job = _wait_for_terminal_job(client, created.json()["id"], save_id="save-2")

    assert job["status"] == "succeeded"
    assert job["result"]["status"] == "Action choices regenerated"
    assert runtime.calls == [("narrator-1", "save-2")]


def test_chat_message_edit_passes_body_and_save_to_runtime(tmp_path: Path) -> None:
    class MessageEditRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, object, str | None]] = []

        async def edit_message_without_resubmit(
            self,
            *,
            message_id: str,
            body: str,
            active_save_id: str | None | object = ...,
            current_user_id: str | None = None,
            on_revision_committed: Callable[[object], object] | None = None,
        ) -> dict[str, object]:
            self.calls.append((message_id, body, active_save_id, current_user_id))
            if on_revision_committed is not None:
                on_revision_committed({"status": "Reconciling message edit..."})
            return {
                "active_save_id": "save-2",
                "active_save_title": "Lantern Keep",
                "chronicle": {
                    "messages": [
                        {
                            "message_id": message_id,
                            "role": "player",
                            "speaker_name": "Keeper",
                            "body": body,
                            "revision_count": 1,
                            "edited_at": "2026-06-02 15:30:00",
                            "actions": [],
                        }
                    ]
                },
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
            }

    runtime = MessageEditRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/message-edit",
            json={
                "message_id": "player-1",
                "body": "Hold the east line.",
                "save_id": "save-2",
            },
        )
        job_id = created.json()["id"]
        assert created.status_code == 200

        for _ in range(20):
            job = client.get(_job_url(job_id, "save-2")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert runtime.calls == [
        ("player-1", "Hold the east line.", "save-2", None),
    ]
    snapshot = state.jobs.get(job_id)
    assert snapshot is not None
    events = snapshot.events
    event_labels = [
        event["payload"].get("label")
        for event in events
        if event["event"] == "progress"
    ]
    event_names = [event["event"] for event in events]
    assert event_labels == ["Saving edit", "Reconciling world data"]
    assert event_names.index("runtime") < event_names.index("status", 2)


def test_chat_message_edit_job_fails_when_runtime_returns_error(
    tmp_path: Path,
) -> None:
    class MessageEditRuntime:
        async def edit_message_without_resubmit(
            self,
            *,
            message_id: str,
            body: str,
            active_save_id: str | None | object = ...,
            on_revision_committed: Callable[[object], object] | None = None,
        ) -> dict[str, object]:
            return {
                "active_save_id": active_save_id,
                "active_save_title": "Lantern Keep",
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "custom_instructions": "",
                "error": f"Message not found: {message_id}",
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
            }

    state = _state_double(tmp_path, MessageEditRuntime())

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/message-edit",
            json={
                "message_id": "missing-message",
                "body": "Hold the east line.",
                "save_id": "save-2",
            },
        )
        job_id = created.json()["id"]
        assert created.status_code == 200

        for _ in range(20):
            job = client.get(_job_url(job_id, "save-2")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "failed"
    assert job["error"] == SAFE_JOB_ERROR


def test_chat_narrator_edit_passes_body_and_save_to_runtime(tmp_path: Path) -> None:
    class NarratorEditRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, object, str | None]] = []

        async def edit_narrator_message(
            self,
            *,
            message_id: str,
            body: str,
            active_save_id: str | None | object = ...,
            current_user_id: str | None = None,
            on_revision_committed: Callable[[object], object] | None = None,
        ) -> dict[str, object]:
            self.calls.append((message_id, body, active_save_id, current_user_id))
            if on_revision_committed is not None:
                on_revision_committed({"status": "Reconciling message edit..."})
            return {
                "active_save_id": "save-2",
                "active_save_title": "Lantern Keep",
                "chronicle": {
                    "messages": [
                        {
                            "message_id": message_id,
                            "role": "narrator",
                            "speaker_name": None,
                            "body": body,
                            "revision_count": 1,
                            "edited_at": "2026-06-02 15:30:00",
                            "actions": [],
                        }
                    ]
                },
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
            }

    runtime = NarratorEditRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/narrator-edit",
            json={
                "message_id": "narrator-1",
                "body": "The lantern stays lit.",
                "save_id": "save-2",
            },
        )
        job_id = created.json()["id"]
        assert created.status_code == 200

        for _ in range(20):
            job = client.get(_job_url(job_id, "save-2")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert runtime.calls == [
        ("narrator-1", "The lantern stays lit.", "save-2", None)
    ]
    snapshot = state.jobs.get(job_id)
    assert snapshot is not None
    events = snapshot.events
    event_labels = [
        event["payload"].get("label")
        for event in events
        if event["event"] == "progress"
    ]
    event_names = [event["event"] for event in events]
    assert event_labels == ["Saving edit", "Reconciling world data"]
    assert event_names.index("runtime") < event_names.index("status", 2)


def test_chat_narrator_edit_job_fails_when_runtime_returns_error(
    tmp_path: Path,
) -> None:
    class NarratorEditRuntime:
        async def edit_narrator_message(
            self,
            *,
            message_id: str,
            body: str,
            active_save_id: str | None | object = ...,
            on_revision_committed: Callable[[object], object] | None = None,
        ) -> dict[str, object]:
            return {
                "active_save_id": active_save_id,
                "active_save_title": "Lantern Keep",
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "custom_instructions": "",
                "error": f"Message not found: {message_id}",
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
            }

    state = _state_double(tmp_path, NarratorEditRuntime())

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/chat/narrator-edit",
            json={
                "message_id": "missing-message",
                "body": "The lantern stays lit.",
                "save_id": "save-2",
            },
        )
        job_id = created.json()["id"]
        assert created.status_code == 200

        for _ in range(20):
            job = client.get(_job_url(job_id, "save-2")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "failed"
    assert job["error"] == SAFE_JOB_ERROR


def test_chat_delete_from_here_passes_message_and_save_to_runtime(
    tmp_path: Path,
) -> None:
    class DeleteRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, object]] = []

        def delete_messages_from_here(
            self,
            *,
            message_id: str,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            self.calls.append((message_id, active_save_id))
            return {
                "active_save_id": (
                    active_save_id if isinstance(active_save_id, str) else "save-1"
                ),
                "active_save_title": "Lantern Keep",
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
                "status": "Messages deleted",
            }

    runtime = DeleteRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        explicit = client.post(
            "/api/chat/delete-from-here",
            json={"message_id": "narrator-2", "save_id": "save-2"},
        )
        assert explicit.status_code == 200
        job = _wait_for_terminal_job(client, explicit.json()["id"], save_id="save-2")
        implicit = client.post(
            "/api/chat/delete-from-here",
            json={"message_id": "narrator-3"},
        )

    assert job["status"] == "succeeded"
    assert job["result"]["status"] == "Messages deleted"
    assert implicit.status_code == 400
    assert runtime.calls[0] == ("narrator-2", "save-2")
    assert implicit.json()["detail"] == _SAVE_ID_REQUIRED_DETAIL
    assert len(runtime.calls) == 1


def test_chat_delete_from_here_returns_runtime_model_error(
    tmp_path: Path,
) -> None:
    class FailingDeleteRuntime(_RuntimeDouble):
        def delete_messages_from_here(
            self,
            *,
            message_id: str,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            return {
                "active_save_id": "save-1",
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "error": f"Message not found: {message_id}",
            }

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, FailingDeleteRuntime())))
    ) as client:
        response = client.post(
            "/api/chat/delete-from-here",
            json={"message_id": "missing-message", "save_id": "save-1"},
        )
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id="save-1")

    assert job["status"] == "succeeded"
    assert job["result"]["error"] == "Message not found: missing-message"


def test_chat_fork_from_here_rejects_missing_save_id(
    tmp_path: Path,
) -> None:
    class ForkRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, object]] = []

        def fork_save_from_message(
            self,
            *,
            message_id: str,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            self.calls.append((message_id, active_save_id))
            return {
                "active_save_id": (
                    active_save_id if isinstance(active_save_id, str) else "fork-1"
                ),
                "active_save_title": "Lantern Keep - fork after narrator 2",
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
                "status": "Save forked",
            }

    runtime = ForkRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        explicit = client.post(
            "/api/chat/fork-from-here",
            json={"message_id": "narrator-2", "save_id": "save-2"},
        )
        assert explicit.status_code == 200
        job = _wait_for_terminal_job(client, explicit.json()["id"], save_id="save-2")
        implicit = client.post(
            "/api/chat/fork-from-here",
            json={"message_id": "narrator-3"},
        )

    assert job["status"] == "succeeded"
    assert job["result"]["status"] == "Save forked"
    assert implicit.status_code == 400
    assert runtime.calls[0] == ("narrator-2", "save-2")
    assert implicit.json()["detail"] == _SAVE_ID_REQUIRED_DETAIL
    assert len(runtime.calls) == 1


def test_save_scenario_draft_awaits_async_runtime(tmp_path: Path) -> None:
    class ScenarioDraftRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, object]] = []

        async def save_scenario_draft(
            self,
            *,
            scenario_type: str,
            scenario_types: list[str] | None,
            sections: dict[str, str],
            character_starters: list[dict[str, object]],
            action_choices_enabled: bool,
            save_title: str,
            source_metadata: dict[str, object] | None,
        ) -> dict[str, object]:
            await asyncio.sleep(0)
            self.calls.append(
                {
                    "scenario_type": scenario_type,
                    "scenario_types": scenario_types,
                    "sections": sections,
                    "character_starters": character_starters,
                    "action_choices_enabled": action_choices_enabled,
                    "save_title": save_title,
                    "source_metadata": source_metadata,
                }
            )
            return {
                "active_save_id": "save-1",
                "active_save_title": save_title,
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
                "status": "Created save: Lantern Keep",
            }

    runtime = ScenarioDraftRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/draft/save",
            json={
                "scenario_type": "full_roleplay",
                "scenario_types": ["full_roleplay", "dating_sim"],
                "sections": {"title": "Lantern Keep"},
                "character_starters": [
                    {
                        "name": "Mara Voss",
                        "concept": "Stormwarden scout",
                    }
                ],
                "action_choices_enabled": True,
                "save_title": "Lantern Keep",
                "source_metadata": {"origin": "generated"},
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "Created save: Lantern Keep"
    assert runtime.calls == [
        {
            "scenario_type": "full_roleplay",
            "scenario_types": ["full_roleplay", "dating_sim"],
            "sections": {"title": "Lantern Keep"},
            "character_starters": [
                {
                    "name": "Mara Voss",
                    "concept": "Stormwarden scout",
                }
            ],
            "action_choices_enabled": True,
            "save_title": "Lantern Keep",
            "source_metadata": {"origin": "generated"},
        }
    ]


def test_save_scenario_draft_defers_opening_choices_to_background_job(
    tmp_path: Path,
) -> None:
    class DeferredChoicesRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.deferred = False
            self.regenerate_calls: list[tuple[str, object, bool]] = []

        async def save_scenario_draft(
            self,
            *,
            scenario_type: str,
            scenario_types: list[str] | None,
            sections: dict[str, str],
            character_starters: list[dict[str, object]],
            action_choices_enabled: bool,
            save_title: str,
            source_metadata: dict[str, object] | None,
            defer_opening_action_choices: bool = False,
        ) -> dict[str, object]:
            del (
                scenario_type,
                scenario_types,
                sections,
                character_starters,
                action_choices_enabled,
                save_title,
                source_metadata,
            )
            self.deferred = defer_opening_action_choices
            return {
                **_chat_model("The beacon snaps awake."),
                "active_save_id": "save-opening",
                "action_choices_enabled": True,
                "action_choices": {
                    "narrator_message_id": "narrator-1",
                    "choices": [],
                },
            }

        async def regenerate_action_choices(
            self,
            *,
            narrator_message_id: str,
            active_save_id: str | None | object = ...,
            retry_progress_callback: object | None = None,
        ) -> dict[str, object]:
            self.regenerate_calls.append(
                (
                    narrator_message_id,
                    active_save_id,
                    retry_progress_callback is not None,
                )
            )
            return {
                **_chat_model("The beacon snaps awake."),
                "active_save_id": "save-opening",
                "action_choices_enabled": True,
                "action_choices": {
                    "narrator_message_id": narrator_message_id,
                    "choices": [
                        {
                            "choice_id": f"choice-{ordinal}",
                            "ordinal": ordinal,
                            "body": body,
                        }
                        for ordinal, body in enumerate(
                            (
                                "Climb the beacon stair.",
                                "Inspect the dark lens.",
                                "Signal the harbor watch.",
                                "Search the keeper's desk.",
                            ),
                            start=1,
                        )
                    ],
                },
                "status": "Action choices generated",
            }

    runtime = DeferredChoicesRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/draft/save",
            json={
                "scenario_type": "full_roleplay",
                "sections": {
                    "title": "Lantern Keep",
                    "opening_message": "The beacon snaps awake.",
                },
                "action_choices_enabled": True,
                "save_title": "Lantern Keep",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        generation_job = payload["action_choices"]["generation_job"]
        job = _wait_for_terminal_job(
            client,
            generation_job["id"],
            save_id="save-opening",
        )

    assert runtime.deferred is True
    assert generation_job["type"] == "action_choice_generate"
    assert generation_job["save_id"] == "save-opening"
    assert job["status"] == "succeeded"
    assert len(job["result"]["action_choices"]["choices"]) == 4
    assert runtime.regenerate_calls == [("narrator-1", "save-opening", True)]


def test_failed_scenario_draft_save_does_not_generate_choices_for_active_save(
    tmp_path: Path,
) -> None:
    class FailedDraftRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.regenerate_calls: list[str] = []

        async def save_scenario_draft(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                **_chat_model("An older chronicle remains active."),
                "active_save_id": "save-existing",
                "action_choices_enabled": True,
                "action_choices": {
                    "narrator_message_id": "narrator-existing",
                    "choices": [],
                },
                "error": "The new draft could not be saved.",
            }

        async def regenerate_action_choices(
            self,
            *,
            narrator_message_id: str,
            **kwargs: object,
        ) -> dict[str, object]:
            del kwargs
            self.regenerate_calls.append(narrator_message_id)
            return _chat_model("Unexpected generation.")

    runtime = FailedDraftRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/draft/save",
            json={
                "scenario_type": "full_roleplay",
                "sections": {
                    "title": "Lantern Keep",
                    "opening_message": "The beacon snaps awake.",
                },
                "action_choices_enabled": True,
                "save_title": "Lantern Keep",
            },
        )

    assert response.status_code == 200
    assert response.json()["error"] == "The new draft could not be saved."
    assert "generation_job" not in response.json()["action_choices"]
    assert runtime.regenerate_calls == []


def test_legacy_action_choice_draft_uses_normalized_result_to_queue_generation(
    tmp_path: Path,
) -> None:
    class LegacyDraftRuntime(_RuntimeDouble):
        async def save_scenario_draft(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                **_chat_model("The old road opens."),
                "active_save_id": "save-legacy",
                "action_choices_enabled": True,
                "action_choices": {
                    "narrator_message_id": "narrator-legacy",
                    "choices": [],
                },
            }

        async def regenerate_action_choices(
            self,
            *,
            narrator_message_id: str,
            **kwargs: object,
        ) -> dict[str, object]:
            del kwargs
            return {
                **_chat_model("The old road opens."),
                "active_save_id": "save-legacy",
                "action_choices_enabled": True,
                "action_choices": {
                    "narrator_message_id": narrator_message_id,
                    "choices": [],
                },
            }

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, LegacyDraftRuntime())))
    ) as client:
        response = client.post(
            "/api/scenarios/draft/save",
            json={
                "scenario_type": "choose_your_own_adventure",
                "sections": {
                    "title": "Old Road",
                    "opening_message": "The old road opens.",
                },
                "action_choices_enabled": False,
                "save_title": "Old Road",
            },
        )
        generation_job = response.json()["action_choices"]["generation_job"]
        job = _wait_for_terminal_job(
            client,
            generation_job["id"],
            save_id="save-legacy",
        )

    assert response.status_code == 200
    assert generation_job["type"] == "action_choice_generate"
    assert job["status"] == "succeeded"


def test_scenario_draft_character_starters_generation_uses_runtime_job(
    tmp_path: Path,
) -> None:
    class StarterGenerationRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, object]] = []

        async def generate_scenario_draft_character_starters(
            self,
            *,
            scenario_type: str,
            scenario_types: list[str] | None,
            sections: dict[str, str],
            character_starters: list[dict[str, object]],
            count: int | None,
            custom_description: str,
            action_choices_enabled: bool,
        ) -> dict[str, object]:
            await asyncio.sleep(0)
            self.calls.append(
                {
                    "scenario_type": scenario_type,
                    "scenario_types": scenario_types,
                    "sections": sections,
                    "character_starters": character_starters,
                    "count": count,
                    "custom_description": custom_description,
                    "action_choices_enabled": action_choices_enabled,
                }
            )
            return {
                "active_save_id": None,
                "active_save_title": None,
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
                "scenario_draft": {
                    "scenario_type": "full_roleplay",
                    "scenario_types": ["full_roleplay", "investigation_mystery"],
                    "action_choices_enabled": True,
                    "sections": [["title", "Lantern Keep"]],
                    "character_starters": [
                        {
                            "name": "Mara Voss",
                            "concept": "Stormwarden scout",
                        },
                        {
                            "name": "Ivo Hale",
                            "concept": "Archivist with a secret",
                        },
                    ],
                    "regeneration_seed": "",
                    "source_metadata": [],
                },
                "status": "Character starters generated",
            }

    runtime = StarterGenerationRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/draft/character-starters/generate",
            json={
                "scenario_type": "full_roleplay",
                "scenario_types": ["full_roleplay", "investigation_mystery"],
                "sections": {
                    "title": "Lantern Keep",
                    "opening_message": "The beacon wakes.",
                },
                "character_starters": [
                    {
                        "name": "Mara Voss",
                        "concept": "Stormwarden scout",
                    }
                ],
                "count": 1,
                "action_choices_enabled": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["type"] == "scenario_character_starters"
        job = _wait_for_terminal_job(client, response.json()["id"])

    assert job["status"] == "succeeded"
    assert job["result"]["status"] == "Character starters generated"
    assert job["result"]["scenario_draft"]["character_starters"] == [
        {
            "name": "Mara Voss",
            "concept": "Stormwarden scout",
        },
        {
            "name": "Ivo Hale",
            "concept": "Archivist with a secret",
        },
    ]
    assert runtime.calls == [
        {
            "scenario_type": "full_roleplay",
            "scenario_types": ["full_roleplay", "investigation_mystery"],
            "sections": {
                "title": "Lantern Keep",
                "opening_message": "The beacon wakes.",
            },
            "character_starters": [
                {
                    "name": "Mara Voss",
                    "concept": "Stormwarden scout",
                }
            ],
            "count": 1,
            "custom_description": "",
            "action_choices_enabled": True,
        }
    ]


@pytest.mark.parametrize(
    ("request_overrides", "detail"),
    [
        (
            {},
            "Number of characters or custom character description is required",
        ),
        (
            {"count": 0},
            "Number of characters must be between 1 and 12",
        ),
        (
            {"count": 13},
            "Number of characters must be between 1 and 12",
        ),
        (
            {"character_starters": [{"role": "Stormwarden scout"}], "count": 1},
            "character_starters[0].name is required",
        ),
    ],
)
def test_scenario_draft_character_starters_generation_rejects_invalid_before_job(
    tmp_path: Path,
    request_overrides: dict[str, object],
    detail: str,
) -> None:
    class StarterGenerationRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate_scenario_draft_character_starters(
            self,
            **_kwargs: object,
        ) -> dict[str, object]:
            self.calls += 1
            return _chat_model("Character starters generated.")

    runtime = StarterGenerationRuntime()
    payload = {
        "scenario_type": "full_roleplay",
        "sections": {"title": "Lantern Keep"},
        **request_overrides,
    }

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/draft/character-starters/generate",
            json=payload,
        )

    assert response.status_code == 400
    assert response.json() == {"detail": detail}
    assert runtime.calls == 0


@pytest.mark.parametrize("count", [True, "1", 1.0])
def test_scenario_draft_character_starters_generation_rejects_non_integer_count(
    tmp_path: Path,
    count: object,
) -> None:
    class StarterGenerationRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate_scenario_draft_character_starters(
            self,
            **_kwargs: object,
        ) -> dict[str, object]:
            self.calls += 1
            return _chat_model("Character starters generated.")

    runtime = StarterGenerationRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/draft/character-starters/generate",
            json={
                "scenario_type": "full_roleplay",
                "sections": {"title": "Lantern Keep"},
                "count": count,
            },
        )

    assert response.status_code == 422
    assert runtime.calls == 0


def test_scenario_draft_character_starters_generation_rejects_extra_fields(
    tmp_path: Path,
) -> None:
    class StarterGenerationRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate_scenario_draft_character_starters(
            self,
            **_kwargs: object,
        ) -> dict[str, object]:
            self.calls += 1
            return _chat_model("Character starters generated.")

    runtime = StarterGenerationRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/draft/character-starters/generate",
            json={
                "scenario_type": "full_roleplay",
                "sections": {"title": "Lantern Keep"},
                "count": 1,
                "ignored": "x" * 70_000,
            },
        )

    assert response.status_code == 422
    assert runtime.calls == 0


def test_scenario_draft_character_starters_generation_rejects_oversized_payload(
    tmp_path: Path,
) -> None:
    class StarterGenerationRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate_scenario_draft_character_starters(
            self,
            **_kwargs: object,
        ) -> dict[str, object]:
            self.calls += 1
            return _chat_model("Character starters generated.")

    runtime = StarterGenerationRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/draft/character-starters/generate",
            json={
                "scenario_type": "full_roleplay",
                "sections": {"title": "Lantern Keep"},
                "character_starters": [
                    {"name": f"Starter {index}"}
                    for index in range(25)
                ],
                "count": 1,
            },
        )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Character starter generation request is too large"
    }
    assert runtime.calls == 0


def test_scenario_draft_character_starters_generation_rejects_oversized_json(
    tmp_path: Path,
) -> None:
    class StarterGenerationRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate_scenario_draft_character_starters(
            self,
            **_kwargs: object,
        ) -> dict[str, object]:
            self.calls += 1
            return _chat_model("Character starters generated.")

    runtime = StarterGenerationRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/draft/character-starters/generate",
            json={
                "scenario_type": "full_roleplay",
                "sections": {"title": "Lantern Keep"},
                "character_starters": [
                    {
                        "name": "Mara Voss",
                        "padding": [0] * 35_000,
                    }
                ],
                "count": 1,
            },
        )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Character starter generation request is too large"
    }
    assert runtime.calls == 0


def test_continuation_scenario_draft_uses_runtime_job(tmp_path: Path) -> None:
    class ContinuationDraftRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str | None | object, str]] = []

        async def generate_continuation_scenario_draft(
            self,
            *,
            active_save_id: str | None | object,
            chapter_start_instructions: str,
            progress_callback: object,
        ) -> dict[str, object]:
            self.calls.append((active_save_id, chapter_start_instructions))
            return {
                "active_save_id": "save-2",
                "active_save_title": "Chapter Two",
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
                "scenario_draft": {
                    "scenario_type": "full_roleplay",
                    "sections": [["title", "Chapter Two"]],
                    "regeneration_seed": "seed",
                    "source_metadata": [["origin", "save_continuation"]],
                },
                "status": "Continuation draft generated",
            }

    runtime = ContinuationDraftRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/continuation-draft",
            json={
                "save_id": "save-1",
                "chapter_start_instructions": (
                    "Begin after the party wakes the next morning."
                ),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "scenario_draft"
    assert runtime.calls == [
        ("save-1", "Begin after the party wakes the next morning.")
    ]


def test_scenario_routes_forward_child_actor(
    tmp_path: Path,
) -> None:
    class ChildScenarioRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.actor_user_ids: list[tuple[str, str | None]] = []

        async def generate_scenario_draft(
            self,
            *,
            scenario_type: str,
            seed: str,
            action_choices_enabled: bool,
            progress_callback: object,
            current_user_id: str | None = None,
        ) -> dict[str, object]:
            del scenario_type, seed, action_choices_enabled, progress_callback
            self.actor_user_ids.append(("draft", current_user_id))
            return _chat_model("Draft generated.")

        async def generate_continuation_scenario_draft(
            self,
            *,
            active_save_id: str | None | object,
            chapter_start_instructions: str,
            progress_callback: object,
            current_user_id: str | None = None,
        ) -> dict[str, object]:
            del active_save_id, chapter_start_instructions, progress_callback
            self.actor_user_ids.append(("continuation", current_user_id))
            return _chat_model("Continuation generated.")

        async def regenerate_scenario_section(
            self,
            *,
            scenario_type: str,
            seed: str,
            section_id: str,
            sections: dict[str, str],
            action_choices_enabled: bool,
            current_user_id: str | None = None,
        ) -> dict[str, object]:
            del scenario_type, seed, section_id, sections, action_choices_enabled
            self.actor_user_ids.append(("section", current_user_id))
            return _chat_model("Section generated.")

        async def save_scenario_draft(
            self,
            *,
            scenario_type: str,
            scenario_types: list[str] | None,
            sections: dict[str, str],
            character_starters: list[dict[str, object]],
            action_choices_enabled: bool,
            save_title: str,
            source_metadata: dict[str, object] | None,
            current_user_id: str | None = None,
        ) -> dict[str, object]:
            del (
                scenario_type,
                scenario_types,
                sections,
                character_starters,
                action_choices_enabled,
                save_title,
                source_metadata,
            )
            self.actor_user_ids.append(("save", current_user_id))
            return _chat_model("Draft saved.")

        async def generate_scenario_draft_character_starters(
            self,
            *,
            scenario_type: str,
            sections: dict[str, str],
            character_starters: list[dict[str, object]],
            count: int | None,
            custom_description: str,
            scenario_types: list[str] | None = None,
            action_choices_enabled: bool = False,
            current_user_id: str | None = None,
        ) -> dict[str, object]:
            del (
                scenario_type,
                scenario_types,
                sections,
                character_starters,
                count,
                custom_description,
                action_choices_enabled,
            )
            self.actor_user_ids.append(("starters", current_user_id))
            return _chat_model("Character starters generated.")

        def start_saved_scenario(
            self,
            *,
            scenario_id: str,
            save_title: str,
            current_user_id: str | None = None,
        ) -> dict[str, object]:
            del scenario_id, save_title
            self.actor_user_ids.append(("start", current_user_id))
            return _chat_model("Scenario started.")

        def create_manual_scenario(
            self,
            scenario: object,
            *,
            current_user_id: str | None = None,
        ) -> dict[str, object]:
            del scenario
            self.actor_user_ids.append(("manual", current_user_id))
            return _chat_model("Manual scenario started.")

    runtime = ChildScenarioRuntime()
    state = _auth_state(tmp_path, runtime)
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    save = _create_auth_save(
        state.repositories,
        title="Lantern Keep",
        owner_user_id=child.id,
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        created_jobs = [
            (
                client.post(
                    "/api/scenarios/draft",
                    json={
                        "scenario_type": "full_roleplay",
                        "seed": "A quiet watchtower.",
                    },
                ),
                None,
            ),
            (
                client.post(
                    "/api/scenarios/continuation-draft",
                    json={"save_id": save.id},
                ),
                save.id,
            ),
            (
                client.post(
                    "/api/scenarios/draft/section",
                    json={
                        "scenario_type": "full_roleplay",
                        "seed": "A quiet watchtower.",
                        "section_id": "opening_message",
                        "sections": {"title": "Lantern Keep"},
                    },
                ),
                None,
            ),
        ]
        for created, job_save_id in created_jobs:
            assert created.status_code == 200
            job = _wait_for_terminal_job(
                client,
                created.json()["id"],
                save_id=job_save_id,
            )
            assert job["status"] == "succeeded"
        starter_generation = client.post(
            "/api/scenarios/draft/character-starters/generate",
            json={
                "scenario_type": "full_roleplay",
                "sections": {"title": "Lantern Keep"},
                "character_starters": [],
                "count": 1,
            },
        )
        saved = client.post(
            "/api/scenarios/draft/save",
            json={
                "scenario_type": "full_roleplay",
                "sections": {"title": "Lantern Keep"},
            },
        )
        started = client.post(
            f"/api/scenarios/{save.scenario_id}/start",
            json={},
        )
        manual = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A quiet watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes.",
            },
        )
        assert starter_generation.status_code == 403
        assert starter_generation.json() == {
            "detail": "Character starter generation is not allowed"
        }
        assert saved.status_code == 200
        assert started.status_code == 200
        assert manual.status_code == 200

    assert runtime.actor_user_ids == [
        ("draft", child.id),
        ("continuation", child.id),
        ("section", child.id),
        ("save", child.id),
        ("start", child.id),
        ("manual", child.id),
    ]


def test_scenario_draft_job_does_not_hold_global_runtime_lock(
    tmp_path: Path,
) -> None:
    class BlockingDraftRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        async def generate_scenario_draft(
            self,
            *,
            scenario_type: str,
            scenario_types: list[str] | None,
            seed: str,
            action_choices_enabled: bool,
            progress_callback: object,
        ) -> dict[str, object]:
            assert scenario_types == ["full_roleplay", "dating_sim"]
            self.started.set()
            await asyncio.to_thread(self.release.wait)
            return {
                "active_save_id": None,
                "active_save_title": None,
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
                "scenario_draft": {
                    "scenario_type": scenario_type,
                    "scenario_types": scenario_types,
                    "action_choices_enabled": action_choices_enabled,
                    "sections": [["title", seed]],
                    "regeneration_seed": seed,
                    "source_metadata": [],
                },
                "status": "Scenario draft generated",
            }

    runtime = BlockingDraftRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/scenarios/draft",
            json={
                "scenario_type": "full_roleplay",
                "scenario_types": ["full_roleplay", "dating_sim"],
                "seed": "mirror duel",
            },
        )
        assert response.status_code == 200
        job_id = response.json()["id"]
        assert runtime.started.wait(1.0)

        lock_entered = threading.Event()
        lock_thread = threading.Thread(
            target=lambda: _enter_runtime_lock(state.lock, lock_entered),
        )
        lock_thread.start()
        try:
            assert lock_entered.wait(1.0)
        finally:
            runtime.release.set()
            lock_thread.join(timeout=1.0)

        job = _wait_for_terminal_job(client, job_id)

    assert job["status"] == "succeeded"
    assert job["result"]["scenario_draft"]["scenario_type"] == "full_roleplay"
    assert job["result"]["scenario_draft"]["scenario_types"] == [
        "full_roleplay",
        "dating_sim",
    ]


def test_continuation_scenario_draft_defaults_blank_instructions(
    tmp_path: Path,
) -> None:
    class ContinuationDraftRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str | None | object, str]] = []

        async def generate_continuation_scenario_draft(
            self,
            *,
            active_save_id: str | None | object,
            chapter_start_instructions: str,
            progress_callback: object,
        ) -> dict[str, object]:
            self.calls.append((active_save_id, chapter_start_instructions))
            return {
                "active_save_id": "save-2",
                "active_save_title": "Chapter Two",
                "chronicle": {"messages": []},
                "composer_enabled": True,
                "custom_instructions": "",
                "error": None,
                "failed_save": False,
                "failure_text": None,
                "media": None,
                "model_indicator": "fake / chat",
                "saves": [],
                "scenario_draft": None,
                "status": "Continuation draft generated",
            }

    runtime = ContinuationDraftRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/scenarios/continuation-draft",
            json={"save_id": "save-1"},
        )

    assert response.status_code == 200
    assert response.json()["type"] == "scenario_draft"
    assert runtime.calls == [("save-1", "")]


def test_scenario_section_regeneration_creates_job_with_payload(
    tmp_path: Path,
) -> None:
    class ScenarioSectionRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, object]] = []

        async def regenerate_scenario_section(
            self,
            *,
            scenario_type: str,
            scenario_types: list[str] | None,
            seed: str,
            section_id: str,
            sections: dict[str, str],
            action_choices_enabled: bool,
        ) -> dict[str, object]:
            self.calls.append(
                {
                    "scenario_type": scenario_type,
                    "scenario_types": scenario_types,
                    "seed": seed,
                    "section_id": section_id,
                    "sections": sections,
                    "action_choices_enabled": action_choices_enabled,
                }
            )
            return {"section_id": section_id, "text": "A sharper opening."}

    runtime = ScenarioSectionRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        created = client.post(
            "/api/scenarios/draft/section",
            json={
                "scenario_type": "full_roleplay",
                "scenario_types": ["full_roleplay", "dating_sim"],
                "seed": "storm-keep",
                "section_id": "opening",
                "sections": {"title": "Lantern Keep"},
                "action_choices_enabled": True,
            },
        )
        assert created.status_code == 200
        job = _wait_for_terminal_job(client, created.json()["id"])

    assert job["status"] == "succeeded"
    assert job["type"] == "scenario_section"
    assert job["result"] == {"section_id": "opening", "text": "A sharper opening."}
    assert runtime.calls == [
        {
            "scenario_type": "full_roleplay",
            "scenario_types": ["full_roleplay", "dating_sim"],
            "seed": "storm-keep",
            "section_id": "opening",
            "sections": {"title": "Lantern Keep"},
            "action_choices_enabled": True,
        }
    ]


def test_venice_character_routes_are_absent(tmp_path: Path) -> None:
    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, _RuntimeDouble())))
    ) as client:
        search = client.post(
            "/api/scenarios/venice/search",
            json={"search": "mara", "limit": 10, "offset": 0},
        )
        imported = client.post(
            "/api/scenarios/venice/import",
            json={"slug": "mara"},
        )

    assert search.status_code == 404
    assert imported.status_code == 404


@pytest.mark.parametrize(
    "path,payload",
    [
        (
            "/api/scenarios/manual",
            {
                "scenario_type": "full_roleplay",
                "title": "Invalid",
                "premise": "Invalid",
                "opening_message": "Invalid",
            },
        ),
        (
            "/api/scenarios/draft",
            {"scenario_type": "full_roleplay", "seed": "Invalid"},
        ),
        (
            "/api/scenarios/draft/save",
            {"scenario_type": "full_roleplay", "sections": {"title": "Invalid"}},
        ),
        (
            "/api/scenarios/draft/section",
            {
                "scenario_type": "full_roleplay",
                "seed": "Invalid",
                "section_id": "title",
                "sections": {},
            },
        ),
    ],
)
def test_scenario_creation_rejects_invalid_interaction_mode(
    tmp_path: Path,
    path: str,
    payload: dict[str, object],
) -> None:
    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, _RuntimeDouble())))
    ) as client:
        response = client.post(
            path,
            json={**payload, "interaction_mode": "cinematic"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown interaction mode: cinematic"


def test_manual_scenario_rejects_non_string_interaction_mode(
    tmp_path: Path,
) -> None:
    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, _RuntimeDouble())))
    ) as client:
        response = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Invalid",
                "premise": "Invalid",
                "player_role": "",
                "interaction_mode": 123,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "interaction_mode must be 'roleplay' or 'storyteller'"
    )


def test_manual_scenario_rejects_empty_interaction_mode(
    tmp_path: Path,
) -> None:
    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, _RuntimeDouble())))
    ) as client:
        response = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Invalid",
                "premise": "Invalid",
                "player_role": "",
                "interaction_mode": "",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown interaction mode: "


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/api/scenarios/manual",
            {
                "scenario_type": "character_interaction",
                "title": "Retired",
                "premise": "Retired",
                "player_role": "Player",
                "opening_message": "Hello",
            },
        ),
        (
            "/api/scenarios/draft",
            {"scenario_type": "character_interaction", "seed": "Retired"},
        ),
        (
            "/api/scenarios/draft/save",
            {
                "scenario_type": "character_interaction",
                "sections": {"title": "Retired"},
            },
        ),
        (
            "/api/scenarios/draft/section",
            {
                "scenario_type": "character_interaction",
                "seed": "Retired",
                "section_id": "title",
                "sections": {},
            },
        ),
        (
            "/api/scenarios/draft/character-starters/generate",
            {
                "scenario_type": "character_interaction",
                "sections": {"title": "Retired"},
                "count": 1,
            },
        ),
        (
            "/api/scenarios/draft",
            {
                "scenario_type": "dating_sim",
                "scenario_types": ["dating_sim", "character_interaction"],
                "seed": "Retired hybrid",
            },
        ),
    ],
)
def test_scenario_creation_rejects_retired_character_interaction(
    tmp_path: Path,
    path: str,
    payload: dict[str, object],
) -> None:
    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, _RuntimeDouble())))
    ) as client:
        response = client.post(path, json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The character_interaction scenario type is no longer supported"
    )


def test_custom_instructions_update_uses_runtime(tmp_path: Path) -> None:
    class GuidanceRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, str | None]] = []

        def update_custom_instructions(
            self,
            *,
            custom_instructions: str,
            active_save_id: str | None,
        ) -> dict[str, object]:
            self.calls.append((custom_instructions, active_save_id))
            return {
                "active_save_id": active_save_id,
                "active_save_title": "Lantern Keep",
                "custom_instructions": custom_instructions,
                "status": "Response guidance saved",
            }

    runtime = GuidanceRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/runtime/custom-instructions",
            json={
                "save_id": "save-1",
                "custom_instructions": "Keep narration brief.",
            },
        )

    assert response.status_code == 200
    assert response.json()["custom_instructions"] == "Keep narration brief."
    assert runtime.calls == [("Keep narration brief.", "save-1")]


def test_guided_context_cleanup_passes_instruction_to_runtime(
    tmp_path: Path,
) -> None:
    class GuidedCleanupRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, str | None]] = []

        async def run_guided_context_cleanup(
            self,
            *,
            instruction: str,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            self.calls.append(
                (
                    instruction,
                    active_save_id if isinstance(active_save_id, str) else None,
                )
            )
            return {
                "active_save_id": "save-1",
                "active_save_title": "Lantern Keep",
                "status": (
                    "Guided cleanup queued: 1 suggestions ready for review, "
                    "0 rejected. They will be reviewed automatically."
                ),
                "error": None,
            }

    runtime = GuidedCleanupRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        created = client.post(
            "/api/world-data/guided-cleanup",
            json={
                "save_id": "save-1",
                "instruction": "Archive the resolved storm thread.",
            },
        )
        job_id = created.json()["id"]
        assert created.status_code == 200

        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert job["type"] == "guided_context_cleanup"
    assert job["result"]["status"].startswith("Guided cleanup queued")
    assert runtime.calls == [("Archive the resolved storm thread.", "save-1")]


def test_context_cleanup_passes_save_id_to_runtime(
    tmp_path: Path,
) -> None:
    class CleanupRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str | None] = []

        async def run_context_cleanup(
            self,
            *,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            self.calls.append(
                active_save_id if isinstance(active_save_id, str) else None,
            )
            return {
                "active_save_id": "save-2",
                "active_save_title": "Signal Tower",
                "status": "Context cleanup finished: 0 changes applied, 0 rejected.",
                "error": None,
            }

    runtime = CleanupRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        created = client.post(
            "/api/world-data/context-cleanup",
            json={"save_id": "save-2"},
        )
        job_id = created.json()["id"]
        assert created.status_code == 200
        assert created.json()["save_id"] == "save-2"

        for _ in range(20):
            job = client.get(_job_url(job_id, "save-2")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert job["type"] == "context_cleanup"
    assert runtime.calls == ["save-2"]


def test_summary_backfill_passes_save_id_and_window_flag_to_runtime(
    tmp_path: Path,
) -> None:
    class SummaryBackfillRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str | None, bool]] = []

        async def run_summary_backfill(
            self,
            *,
            active_save_id: str | None | object = ...,
            apply_recommended_windows: bool = False,
        ) -> dict[str, object]:
            self.calls.append(
                (
                    active_save_id if isinstance(active_save_id, str) else None,
                    apply_recommended_windows,
                )
            )
            return {
                "active_save_id": "save-2",
                "active_save_title": "Signal Tower",
                "status": "Summary backfill finished: 6 messages compacted.",
                "error": None,
            }

    runtime = SummaryBackfillRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        created = client.post(
            "/api/world-data/summary-backfill",
            json={"save_id": "save-2", "apply_recommended_windows": True},
        )
        job_id = created.json()["id"]
        assert created.status_code == 200
        assert created.json()["save_id"] == "save-2"

        job = _wait_for_terminal_job(client, job_id, save_id="save-2")

    assert job["status"] == "succeeded"
    assert job["type"] == "summary_backfill"
    assert job["result"]["status"].startswith("Summary backfill finished")
    assert runtime.calls == [("save-2", True)]


def test_summary_backfill_job_result_filters_runtime_model_for_request_user(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    rook = state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    mira_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    rook_save = _create_auth_save(
        state.repositories,
        title="Rook Save",
        owner_user_id=rook.id,
    )

    class SummaryBackfillRuntime(_RuntimeDouble):
        async def run_summary_backfill(
            self,
            *,
            active_save_id: str | None | object = ...,
            apply_recommended_windows: bool = False,
        ) -> dict[str, object]:
            save_id = active_save_id if isinstance(active_save_id, str) else None
            return {
                "active_save_id": save_id,
                "active_save_title": "Mira Save",
                "saves": [
                    {"save_id": mira_save.id, "title": "Mira Save", "active": True},
                    {"save_id": rook_save.id, "title": "Rook Save", "active": False},
                ],
                "chronicle": {"messages": []},
                "status": "Summary backfill finished: 4 messages compacted.",
                "error": None,
            }

    state.runtime = SummaryBackfillRuntime()

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        created = client.post(
            "/api/world-data/summary-backfill",
            json={"save_id": mira_save.id},
        )
        job = _wait_for_terminal_job(
            client,
            created.json()["id"],
            save_id=mira_save.id,
        )

    assert created.status_code == 200
    assert job["status"] == "succeeded"
    assert [save["save_id"] for save in job["result"]["saves"]] == [mira_save.id]
    assert rook_save.id not in repr(job["result"])


def test_summary_backfill_runtime_error_marks_job_failed(
    tmp_path: Path,
) -> None:
    class FailingSummaryBackfillRuntime(_RuntimeDouble):
        async def run_summary_backfill(
            self,
            *,
            active_save_id: str | None | object = ...,
            apply_recommended_windows: bool = False,
        ) -> dict[str, object]:
            save_id = active_save_id if isinstance(active_save_id, str) else None
            return {
                "active_save_id": save_id,
                "active_save_title": "Signal Tower",
                "saves": [],
                "chronicle": {"messages": []},
                "status": None,
                "error": "summary rejected as continuation-risk direct prompt",
            }

    runtime = FailingSummaryBackfillRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        created = client.post(
            "/api/world-data/summary-backfill",
            json={"save_id": "save-2"},
        )
        job = _wait_for_terminal_job(
            client,
            created.json()["id"],
            save_id="save-2",
        )

    assert created.status_code == 200
    assert job["status"] == "failed"
    assert job["type"] == "summary_backfill"
    assert job["error"] == SAFE_JOB_ERROR


def test_world_suggestion_review_and_retention_pass_save_id_to_runtime(
    tmp_path: Path,
) -> None:
    class MaintenanceRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.review_calls: list[str | None] = []
            self.retention_calls: list[str | None] = []

        async def run_world_suggestion_review(
            self,
            *,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            self.review_calls.append(
                active_save_id if isinstance(active_save_id, str) else None,
            )
            return {
                "active_save_id": "save-1",
                "active_save_title": "Lantern Keep",
                "status": "World suggestion review finished: 0 applied.",
                "error": None,
            }

        async def run_world_context_retention(
            self,
            *,
            active_save_id: str | None | object = ...,
        ) -> dict[str, object]:
            self.retention_calls.append(
                active_save_id if isinstance(active_save_id, str) else None,
            )
            return {
                "active_save_id": "save-1",
                "active_save_title": "Lantern Keep",
                "status": "World context retention finished: 0 suggestions expired.",
                "error": None,
            }

    runtime = MaintenanceRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        review = client.post(
            "/api/world-data/suggestion-review",
            json={"save_id": "save-1"},
        )
        retention = client.post(
            "/api/world-data/context-retention",
            json={"save_id": "save-1"},
        )

        review_job = _wait_for_terminal_job(
            client,
            review.json()["id"],
            save_id="save-1",
        )
        retention_job = _wait_for_terminal_job(
            client,
            retention.json()["id"],
            save_id="save-1",
        )

    assert review.status_code == 200
    assert review.json()["save_id"] == "save-1"
    assert review_job["status"] == "succeeded"
    assert review_job["type"] == "world_suggestion_review"
    assert retention.status_code == 200
    assert retention.json()["save_id"] == "save-1"
    assert retention_job["status"] == "succeeded"
    assert retention_job["type"] == "world_context_retention"
    assert runtime.review_calls == ["save-1"]
    assert runtime.retention_calls == ["save-1"]


def test_model_preference_update_is_reflected_in_settings(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        saved = client.post(
            "/api/settings/model-preference",
            json={
                "task": "image_prompt",
                "provider": "fake",
                "model_id": "fake-image",
            },
        )
        settings = client.get("/api/settings")

    assert saved.status_code == 200
    assert settings.status_code == 200
    selector = next(
        item
        for item in settings.json()["task_model_selectors"]
        if item["task"] == "image_prompt"
    )
    assert selector["selected_provider"] == "fake"
    assert selector["selected_model_id"] == "fake-image"
    assert selector["selected_available"] is False
    assert selector["warning"] == "Selected model is unavailable"


def test_content_safety_model_preference_rejects_unknown_model(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/settings/model-preference",
            json={
                "task": "content_safety",
                "provider": "fake",
                "model_id": "fake-unknown-safety-model",
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Safety Agent model must support structured output"
    }


def test_model_preference_clear_removes_task_override(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        saved = client.post(
            "/api/settings/model-preference",
            json={
                "task": "image_prompt",
                "provider": "fake",
                "model_id": "fake-image",
            },
        )
        cleared = client.delete("/api/settings/model-preference/image_prompt")
        settings = client.get("/api/settings")

    assert saved.status_code == 200
    assert cleared.status_code == 200
    assert cleared.json() == {"ok": True}
    selector = next(
        item
        for item in settings.json()["task_model_selectors"]
        if item["task"] == "image_prompt"
    )
    assert selector["selected_provider"] is None
    assert selector["selected_model_id"] is None
    assert selector["warning"] is None


def test_model_thinking_preference_update_and_clear_are_reflected_in_settings(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    app = create_app()
    state = cast(WebAppState, app.state.bragi)
    state.repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat"],
        thinking={
            "levels": ["high", "low"],
            "default_level": "low",
            "mandatory": False,
        },
    )
    state.repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )

    with TestClient(app) as client:
        saved = client.post(
            "/api/settings/model-thinking",
            json={
                "task": "chat",
                "provider": "fake",
                "model_id": "fake-chat",
                "level": "high",
            },
        )
        settings = client.get("/api/settings")
        rejected = client.post(
            "/api/settings/model-thinking",
            json={
                "task": "chat",
                "provider": "fake",
                "model_id": "fake-chat",
                "level": "medium",
            },
        )
        cleared = client.delete("/api/settings/model-thinking/chat")
        after_clear = client.get("/api/settings")

    assert saved.status_code == 200
    selector = next(
        item
        for item in settings.json()["task_model_selectors"]
        if item["task"] == "chat"
    )
    assert selector["thinking"]["supported"] is True
    assert selector["thinking"]["selected"] == "high"
    assert selector["thinking"]["options"] == ["provider_default", "off", "high", "low"]
    assert rejected.status_code == 400
    assert cleared.status_code == 200
    assert cleared.json() == {"ok": True}
    cleared_selector = next(
        item
        for item in after_clear.json()["task_model_selectors"]
        if item["task"] == "chat"
    )
    assert cleared_selector["thinking"]["selected"] == "provider_default"


def test_model_routing_profiles_save_apply_and_delete(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        saved_preference = client.post(
            "/api/settings/model-preference",
            json={
                "task": "chat",
                "provider": "fake",
                "model_id": "fake-chat",
            },
        )
        saved_profile = client.post(
            "/api/settings/model-routing-profiles",
            json={"name": "Fast Chat"},
        )
        profile_id = saved_profile.json()["profile"]["id"]
        changed_preference = client.post(
            "/api/settings/model-preference",
            json={
                "task": "chat",
                "provider": "fake",
                "model_id": "fake-other-chat",
            },
        )
        applied = client.post(
            f"/api/settings/model-routing-profiles/{profile_id}/apply",
        )
        settings = client.get("/api/settings")
        deleted = client.delete(f"/api/settings/model-routing-profiles/{profile_id}")
        after_delete = client.get("/api/settings")

    assert saved_preference.status_code == 200
    assert saved_profile.status_code == 200
    assert saved_profile.json()["profile"]["name"] == "Fast Chat"
    assert changed_preference.status_code == 200
    assert applied.status_code == 200
    selector = next(
        item
        for item in settings.json()["task_model_selectors"]
        if item["task"] == "chat"
    )
    assert selector["selected_model_id"] == "fake-chat"
    profiles = settings.json()["model_routing_profiles"]
    assert profiles["last_loaded_profile_id"] == profile_id
    saved_preferences = profiles["profiles"][0]["preferences"]
    assert {
        "task": "chat",
        "provider": "fake",
        "model_id": "fake-chat",
    } in saved_preferences
    assert deleted.status_code == 200
    assert after_delete.json()["model_routing_profiles"]["profiles"] == []
    assert (
        after_delete.json()["model_routing_profiles"]["last_loaded_profile_id"]
        is None
    )


def test_model_routing_profiles_require_admin(tmp_path: Path) -> None:
    state = _auth_state(tmp_path)
    state.settings_service = lambda: SettingsService(
        repositories=state.repositories,
        providers={},
        secret_store=InMemorySecretStore(),
    )
    state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        saved = client.post(
            "/api/settings/model-routing-profiles",
            json={"name": "Nope"},
        )
        applied = client.post(
            "/api/settings/model-routing-profiles/missing/apply",
        )
        deleted = client.delete("/api/settings/model-routing-profiles/missing")

    assert saved.status_code == 403
    assert applied.status_code == 403
    assert deleted.status_code == 403


def test_chat_history_endpoint_delegates_filter_to_runtime(tmp_path: Path) -> None:
    class HistoryRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, str | None, int]] = []

        def build_chat_history_model(
            self,
            *,
            selected_filter: str,
            before_message_id: str | None = None,
            limit: int = 80,
        ) -> dict[str, object]:
            self.calls.append((selected_filter, before_message_id, limit))
            return {
                "active_save_id": "save-1",
                "active_save_title": "Lantern Keep",
                "selected_filter": selected_filter,
                "filter_options": [
                    {"filter_id": "all", "label": "All", "active": False},
                    {
                        "filter_id": "with_images",
                        "label": "With images",
                        "active": True,
                    },
                ],
                "messages": [
                    {
                        "message_id": "message-1",
                        "role": "narrator",
                        "role_label": "Narrator",
                        "speaker_name": None,
                        "body": "The beacon throws an image across the fog.",
                        "markdown_blocks": [],
                        "style_class": "narrator",
                        "provider": "fake",
                        "model": "fake-chat",
                        "token_estimate": 42,
                        "created_at": "2026-05-29T12:00:00Z",
                        "image_count": 1,
                    }
                ],
                "total_message_count": 2,
                "matching_message_count": 1,
                "has_more_before": before_message_id is None,
                "oldest_message_id": "message-1",
                "empty_title": "",
                "empty_detail": "",
            }

    runtime = HistoryRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.get(
            "/api/chat-history?filter=with_images"
            "&before_message_id=message-0&limit=200"
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_filter"] == "with_images"
    assert payload["messages"][0]["provider"] == "fake"
    assert payload["messages"][0]["image_count"] == 1
    assert payload["matching_message_count"] == 1
    assert payload["has_more_before"] is False
    assert payload["oldest_message_id"] == "message-1"
    assert runtime.calls == [("with_images", "message-0", 80)]


def test_chat_history_endpoint_returns_404_for_unknown_page_anchor(
    tmp_path: Path,
) -> None:
    class HistoryRuntime(_RuntimeDouble):
        def build_chat_history_model(
            self,
            *,
            selected_filter: str,
            before_message_id: str | None = None,
            limit: int = 80,
        ) -> dict[str, object]:
            raise ValueError(f"Unknown active message id: {before_message_id}")

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, HistoryRuntime())))
    ) as client:
        response = client.get("/api/chat-history?before_message_id=missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown active message id: missing"


def test_world_data_apply_and_scenario_definition_edit(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    app = create_app()
    cast(WebAppState, app.state.bragi).repositories.set_app_setting(
        "content_filter_rating",
        "unrated",
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes.",
            },
        )
        assert created.status_code == 200
        world = client.get("/api/world-data").json()
        scenario_id = world["scenario"]["scenario_id"]

        applied = client.post(
            "/api/world-data/apply",
            json={
                "active_save_id": world["active_save_id"],
                "edits": {
                    "scenario": {
                        **world["scenario"],
                        "title": "Lantern Keep Revised",
                    },
                    "world_state": [
                        {
                            "row_id": "",
                            "key": "storm",
                            "category": "weather",
                            "confidence": 0.9,
                            "value_json": '{"intensity":"high"}',
                            "source_message_id": None,
                        }
                    ],
                },
            },
        )
        assert applied.status_code == 200
        assert applied.json()["model"]["scenario"]["title"] == "Lantern Keep Revised"
        assert applied.json()["model"]["world_state"][0]["key"] == "storm"

        definition = client.post(
            f"/api/scenarios/{scenario_id}/definition",
            json={
                "edit": {
                    "title": "Shared Lantern Keep",
                    "premise": "A shared watchtower.",
                    "player_role": "Keeper",
                    "content": {},
                }
            },
        )

    assert definition.status_code == 200
    assert definition.json()["model"]["scenario"]["title"] == "Shared Lantern Keep"


def test_world_data_apply_filters_new_unclassified_rows_from_response(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    app = create_app()

    with TestClient(app) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes.",
            },
        )
        assert created.status_code == 200
        world = client.get("/api/world-data").json()
        applied = client.post(
            "/api/world-data/apply",
            json={
                "active_save_id": world["active_save_id"],
                "edits": {
                    "scenario": world["scenario"],
                    "world_state": [
                        {
                            "row_id": "",
                            "key": "unreviewed",
                            "category": "note",
                            "confidence": 1.0,
                            "value_json": '{"body":"unclassified manual prose"}',
                            "source_message_id": None,
                        }
                    ]
                },
            },
        )

    assert applied.status_code == 200, applied.text
    assert applied.json()["model"]["world_state"] == []
    state = cast(WebAppState, app.state.bragi)
    assert any(
        row.key == "unreviewed"
        for row in state.repositories.list_world_state(world["active_save_id"])
    )


def test_scenario_character_starters_are_reviewed_by_safety_agent(
    tmp_path: Path,
) -> None:
    class BlockingSafetyProvider:
        provider_name = "fake"

        def __init__(self) -> None:
            self.requests: list[StructuredOutputRequest] = []

        async def generate_structured_output(
            self,
            request: StructuredOutputRequest,
        ) -> StructuredOutputResponse:
            self.requests.append(request)
            return StructuredOutputResponse(
                data={
                    "action": "block",
                    "category": "violence",
                    "reason": "The starter exceeds the configured ceiling.",
                    "minimum_rating": "r",
                },
                provider=request.provider,
                model_id=request.model_id,
            )

    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(sqlite3.connect(database_path))
    repositories.set_app_setting("content_filter_rating", "pg")
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-safety",
        display_name="Fake Safety",
        capabilities=["structured_output"],
    )
    repositories.set_model_preference(
        task="content_safety",
        provider="fake",
        model_id="fake-safety",
    )
    provider = BlockingSafetyProvider()
    state = cast(
        WebAppState,
        SimpleNamespace(
            repositories=repositories,
            providers={"fake": cast(ProviderClient, provider)},
        ),
    )
    starter = ScenarioCharacterStarter(
        name="The Ash Warden",
        aliases=("Warden",),
        role="Gaoler",
        appearance="A detailed visible injury.",
        relationships={"player": "A frightening threat."},
    )

    reviewed = asyncio.run(
        api_app._review_scenario_edit_for_request(  # noqa: SLF001
            state,
            ScenarioEdit(
                title="Lantern Keep",
                premise="",
                player_role="Keeper",
                content={},
                character_starters=(starter,),
            ),
            save_id=None,
            roleplay_type="full_roleplay",
            current_user_id=None,
        )
    )

    reviewed_starter = reviewed.character_starters[0]
    assert reviewed_starter.name == CONTENT_FILTER_TRANSITION
    assert reviewed_starter.role == CONTENT_FILTER_TRANSITION
    assert reviewed_starter.appearance == CONTENT_FILTER_TRANSITION
    assert reviewed_starter.aliases == ()
    assert reviewed_starter.relationships == {}
    assert dict(reviewed.section_content_ratings)["character_starters"] == "g"
    starter_request = provider.requests[0]
    submitted_starter = json.loads(starter_request.messages[1].body)
    assert submitted_starter["appearance"] == "A detailed visible injury."
    assert submitted_starter["relationships"] == {
        "player": "A frightening threat."
    }


def test_manual_character_profiles_are_reviewed_and_rated(
    tmp_path: Path,
) -> None:
    class BlockingSafetyProvider:
        provider_name = "fake"

        async def generate_structured_output(
            self,
            request: StructuredOutputRequest,
        ) -> StructuredOutputResponse:
            return StructuredOutputResponse(
                data={
                    "action": "block",
                    "category": "violence",
                    "reason": "The profile exceeds the configured ceiling.",
                    "minimum_rating": "r",
                },
                provider=request.provider,
                model_id=request.model_id,
            )

    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(sqlite3.connect(database_path))
    save = _create_auth_save(
        repositories,
        title="Lantern Save",
        owner_user_id=None,
    )
    repositories.set_app_setting("content_filter_rating", "pg")
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-safety",
        display_name="Fake Safety",
        capabilities=["structured_output"],
    )
    repositories.set_model_preference(
        task="content_safety",
        provider="fake",
        model_id="fake-safety",
    )
    state = cast(
        WebAppState,
        SimpleNamespace(
            repositories=repositories,
            providers={
                "fake": cast(ProviderClient, BlockingSafetyProvider())
            },
        ),
    )
    edits = CharacterRegistryEdits(
        characters=(
            CharacterRegistryRow(
                character_id="",
                name="The Ash Warden",
                appearance="A lingering graphic injury.",
                relationships_json='{"player":"A prolonged frightening threat."}',
            ),
        )
    )

    reviewed = asyncio.run(
        api_app._review_character_edits_for_request(  # noqa: SLF001
            state,
            edits,
            save_id=save.id,
            current_user_id=None,
        )
    )
    reviewed_row = reviewed.characters[0]
    assert reviewed_row.name == CONTENT_FILTER_TRANSITION
    assert reviewed_row.appearance == CONTENT_FILTER_TRANSITION
    assert reviewed_row.relationships_json == "{}"
    assert reviewed_row.content_rating == "g"


def test_untrusted_bundle_preview_text_is_replaced_when_safety_blocks(
    tmp_path: Path,
) -> None:
    class BlockingSafetyProvider:
        provider_name = "fake"

        async def generate_structured_output(
            self,
            request: StructuredOutputRequest,
        ) -> StructuredOutputResponse:
            return StructuredOutputResponse(
                data={
                    "action": "block",
                    "category": "sexual_content",
                    "reason": "Imported preview text exceeds the ceiling.",
                    "minimum_rating": "r",
                },
                provider=request.provider,
                model_id=request.model_id,
            )

    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(sqlite3.connect(database_path))
    repositories.set_app_setting("content_filter_rating", "pg")
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-safety",
        display_name="Fake Safety",
        capabilities=["structured_output"],
    )
    repositories.set_model_preference(
        task="content_safety",
        provider="fake",
        model_id="fake-safety",
    )
    state = _state_double(tmp_path)
    state.repositories = repositories
    state.providers = {
        "fake": cast(ProviderClient, BlockingSafetyProvider())
    }

    preview = asyncio.run(
        api_app._review_bundle_preview_for_request(  # noqa: SLF001
            cast(WebAppState, state),
            {
                "save_id": "external-save",
                "title": "Explicit imported title",
                "scenario_title": "Explicit imported scenario",
                "bundle_version": 1,
            },
        )
    )

    assert preview == {
        "save_id": "external-save",
        "title": CONTENT_FILTER_TRANSITION,
        "scenario_title": CONTENT_FILTER_TRANSITION,
        "bundle_version": 1,
    }


def test_scenario_starter_reference_image_upload_and_remove(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    image_bytes = VALID_PNG_BYTES

    app = create_app()
    with TestClient(app) as client:
        state = cast(WebAppState, app.state.bragi)
        state.repositories.set_app_setting("content_filter_rating", "unrated")
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes.",
            },
        )
        assert created.status_code == 200
        world = client.get("/api/world-data").json()
        scenario_id = world["scenario"]["scenario_id"]
        definition = client.post(
            f"/api/scenarios/{scenario_id}/definition",
            json={
                "edit": {
                    **world["scenario"],
                    "character_starters": [{"name": "Captain Ilyra"}],
                }
            },
        )
        assert definition.status_code == 200
        corrupt = client.post(
            f"/api/scenarios/{scenario_id}/character-starters/reference-image/upload",
            data={"starter_name": "Captain Ilyra"},
            files={"file": ("corrupt.png", b"\x89PNG\r\n\x1a\nnot a png", "image/png")},
        )

        uploaded = client.post(
            f"/api/scenarios/{scenario_id}/character-starters/reference-image/upload",
            data={"starter_name": "Captain Ilyra"},
            files={"file": ("ilyra.png", image_bytes, "image/png")},
        )
        assert uploaded.status_code == 200
        starter = uploaded.json()["scenario"]["character_starters"][0]
        reference = starter["reference_image"]
        reference_path = state.paths.media_dir / reference["path"]
        thumbnail_path = state.paths.media_dir / reference["thumbnail_path"]
        assert reference_path.is_file()
        assert thumbnail_path.is_file()
        sibling = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep Fork",
                "premise": "A forked watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes again.",
            },
        )
        assert sibling.status_code == 200
        sibling_world = client.get("/api/world-data").json()
        sibling_scenario_id = sibling_world["scenario"]["scenario_id"]
        sibling_definition = client.post(
            f"/api/scenarios/{sibling_scenario_id}/definition",
            json={
                "edit": {
                    **sibling_world["scenario"],
                    "character_starters": [
                        {
                            "name": "Captain Ilyra",
                            "reference_image": reference,
                        }
                    ],
                }
            },
        )
        assert sibling_definition.status_code == 200
        thumbnail = client.get(
            "/api/scenarios/"
            f"{scenario_id}/character-starters/reference-images/"
            f"{reference['id']}/thumbnail",
        )
        removed = client.post(
            f"/api/scenarios/{scenario_id}/character-starters/reference-image/remove",
            json={"starter_id": starter["starter_id"]},
        )

    assert corrupt.status_code == 400
    assert starter["starter_id"]
    assert reference["mime_type"] == "image/png"
    assert reference["source"] == "uploaded"
    assert thumbnail.status_code == 200
    assert removed.status_code == 200
    assert reference_path.is_file()
    assert thumbnail_path.is_file()
    assert (
        removed.json()["scenario"]["character_starters"][0]["reference_image"]
        is None
    )


def test_scenario_starter_reference_image_upload_requires_admin(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    state.auth_service().create_user(
        username="Admin",
        password="correct horse",
        role="admin",
    )
    state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    image_bytes = VALID_PNG_BYTES

    with TestClient(
        create_app(cast(WebAppState, state)),
        authenticate=False,
    ) as client:
        anonymous = client.post(
            "/api/scenarios/scenario-1/character-starters/reference-image/upload",
            data={"starter_name": "Captain Ilyra"},
            files={"file": ("ilyra.png", image_bytes, "image/png")},
        )
        login = client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        )
        non_admin = client.post(
            "/api/scenarios/scenario-1/character-starters/reference-image/upload",
            data={"starter_name": "Captain Ilyra"},
            files={"file": ("ilyra.png", image_bytes, "image/png")},
        )

    assert anonymous.status_code == 401
    assert login.status_code == 200
    assert non_admin.status_code == 403


def test_scenario_starter_reference_image_serving_requires_starter_media_path(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    app = create_app()
    with TestClient(app) as client:
        state = cast(WebAppState, app.state.bragi)
        private_path = state.paths.media_dir / "save-1" / "private.png"
        private_path.parent.mkdir(parents=True)
        private_path.write_bytes(b"\x89PNG\r\n\x1a\nprivate image bytes")
        legacy_path = (
            state.paths.media_dir / "scenario-starters" / "legacy-reference.png"
        )
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_bytes(b"\x89PNG\r\n\x1a\nlegacy image bytes")
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes.",
            },
        )
        assert created.status_code == 200
        world = client.get("/api/world-data").json()
        scenario_id = world["scenario"]["scenario_id"]
        definition = client.post(
            f"/api/scenarios/{scenario_id}/definition",
            json={
                "edit": {
                    **world["scenario"],
                    "character_starters": [
                        {
                            "name": "Legacy Captain",
                            "reference_image": {
                                "id": "starter-ref-legacy",
                                "path": "scenario-starters/legacy-reference.png",
                                "thumbnail_path": None,
                                "mime_type": "image/png",
                                "source": "uploaded",
                            },
                        },
                        {
                            "name": "Captain Ilyra",
                            "reference_image": {
                                "id": "starter-ref-unsafe",
                                "path": "save-1/private.png",
                                "thumbnail_path": None,
                                "mime_type": "image/png",
                                "source": "uploaded",
                                "content_rating": "g",
                            },
                        }
                    ],
                }
            },
        )
        unsafe = client.get(
            "/api/scenarios/"
            f"{scenario_id}/character-starters/reference-images/starter-ref-unsafe",
        )
        legacy = client.get(
            "/api/scenarios/"
            f"{scenario_id}/character-starters/reference-images/starter-ref-legacy",
        )

    assert definition.status_code == 200
    assert unsafe.status_code == 404
    assert legacy.status_code == 403


def test_world_data_apply_accepts_grouped_suggestion_actions(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    @dataclass(frozen=True)
    class FakeWorldModel:
        active_save_id: str

    @dataclass(frozen=True)
    class FakeApplyResult:
        model: FakeWorldModel
        group_actions: tuple[tuple[str, str], ...]

    captured: list[object] = []

    class FakeWorldDataService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def apply_edits(
            self,
            edits: object,
            *,
            active_save_id: str | object,
        ) -> FakeApplyResult:
            captured.append(edits)
            return FakeApplyResult(
                model=FakeWorldModel(
                    active_save_id=(
                        active_save_id
                        if isinstance(active_save_id, str)
                        else "save-1"
                    )
                ),
                group_actions=tuple(
                    (row.group_id, row.action)
                    for row in cast(Any, edits).suggestion_groups
                ),
            )

    monkeypatch.setattr(api_app, "WorldDataService", FakeWorldDataService)
    state = _state_double(tmp_path)
    state.runtime.active_save_id = "save-1"

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/world-data/apply",
            json={
                "active_save_id": "save-1",
                "edits": {
                    "suggestion_groups": [
                        {
                            "group_id": "group-1",
                            "suggestion_ids": ["suggestion-1", "suggestion-2"],
                            "update_type": "update",
                            "entity_type": "world_state",
                            "entity_id": "state-1",
                            "field_path": "storm",
                            "proposed_value_json": "{\"intensity\":\"low\"}",
                            "status": "pending",
                            "reason": "The storm eased.",
                            "confidence": 0.7,
                            "source_message_ids_text": "message-1",
                            "suggestion_count": 2,
                            "action": "dismiss",
                        }
                    ]
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["group_actions"] == [["group-1", "dismiss"]]
    assert len(captured) == 1
    captured_edits = cast(Any, captured[0])
    assert captured_edits.suggestion_groups[0].suggestion_ids == (
        "suggestion-1",
        "suggestion-2",
    )


def test_character_registry_apply_adds_manual_character_without_hidden_autofill(
    tmp_path: Path,
) -> None:
    class PoisoningRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.repositories: PersistenceRepositories | None = None
            self.completion_calls: list[tuple[str, tuple[str, ...]]] = []

        def complete_sparse_character_profiles(
            self,
            *,
            active_save_id: str,
            character_ids: tuple[str, ...],
        ) -> int:
            assert self.repositories is not None
            self.completion_calls.append((active_save_id, character_ids))
            updated = 0
            for character_id in character_ids:
                character = self.repositories.get_character(character_id)
                if character is None:
                    continue
                self.repositories.update_character(
                    replace(
                        character,
                        known_state=(
                            "Oracle scenario premise leaked into a manual character."
                        ),
                        appearance="Silver eyes, glass-dusted robes, and still hands.",
                        visual_notes="Scenario NPC visual notes leaked.",
                        personality="Curious, cryptic, and fiercely protective.",
                        voice="Soft riddles with sudden direct warnings.",
                        status="present as the scenario character",
                        locked_fields=[
                            "appearance",
                            "known_state",
                            "personality",
                            "visual_notes",
                            "voice",
                        ],
                    )
                )
                updated += 1
            return updated

    runtime = PoisoningRuntime()
    state = _auth_state(tmp_path, runtime)
    state.auth_required = False
    runtime.repositories = state.repositories
    scenario = state.repositories.create_scenario(
        type="full_roleplay",
        title="Oracle of Glass",
        premise="The player finds the oracle beneath a broken skylight.",
        player_role="A traveler carrying a cracked prophecy.",
        content={
            "opening_message": "The glass has been expecting you.",
            "character_name": "Oracle of Glass",
            "player_character_name": "Mara",
            "character_description": "A seer in the mirrored arcade.",
            "character_physical_description": (
                "Silver eyes, glass-dusted robes, and still hands."
            ),
            "character_personality": "Curious, cryptic, and fiercely protective.",
            "character_voice": "Soft riddles with sudden direct warnings.",
            "relationship_seed": "The oracle is wary of Mara but wants to help.",
        },
    )
    save = state.repositories.create_save(
        scenario_id=scenario.id,
        title="Oracle Visit",
    )
    runtime.active_save_id = save.id

    with TestClient(create_app(cast(WebAppState, state))) as client:
        applied = client.post(
            "/api/characters/apply",
            json={
                "active_save_id": save.id,
                "edits": {
                    "characters": [
                        {
                            "character_id": "",
                            "name": "Mara",
                            "role": "Signal runner",
                            "relationships_json": "{}",
                            "met": True,
                            "present": True,
                        }
                    ]
                },
            },
        )

    assert applied.status_code == 200
    payload = applied.json()
    assert payload["created_count"] == 1
    character = payload["model"]["characters"][0]
    assert character["name"] == "Mara"
    assert character["role"] == "Signal runner"
    assert character["known_state"] == ""
    assert character["appearance"] == ""
    assert character["visual_notes"] == ""
    assert character["personality"] == ""
    assert character["voice"] == ""
    assert character["status"] == ""
    assert character["relationships_json"] == "{}"
    assert character["locked_fields"] == []
    assert runtime.completion_calls == []


def test_character_registry_apply_flag_auto_enhances_created_character_agency(
    tmp_path: Path,
) -> None:
    class AgencyRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.repositories: PersistenceRepositories | None = None
            self.agency_calls: list[tuple[str, tuple[str, ...]]] = []

        def complete_new_character_agency(
            self,
            *,
            active_save_id: str,
            character_ids: tuple[str, ...],
        ) -> int:
            assert self.repositories is not None
            self.agency_calls.append((active_save_id, character_ids))
            updated = 0
            for character_id in character_ids:
                character = self.repositories.get_character(character_id)
                if character is None:
                    continue
                self.repositories.update_character(
                    replace(
                        character,
                        motivations="Protect the lower village from ash riders.",
                        boundaries="Will not leave the tower during a lens breach.",
                        locked_fields=[
                            *character.locked_fields,
                            "boundaries",
                            "motivations",
                        ],
                    )
                )
                updated += 1
            return updated

    runtime = AgencyRuntime()
    state = _auth_state(tmp_path, runtime)
    state.auth_required = False
    runtime.repositories = state.repositories
    scenario = state.repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={"opening_message": "The beacon wakes."},
    )
    save = state.repositories.create_save(
        scenario_id=scenario.id,
        title="Lantern Keep",
    )
    runtime.active_save_id = save.id

    with TestClient(create_app(cast(WebAppState, state))) as client:
        applied = client.post(
            "/api/characters/apply",
            json={
                "active_save_id": save.id,
                "auto_enhance_created_agency": True,
                "edits": {
                    "characters": [
                        {
                            "character_id": "",
                            "name": "Mara",
                            "role": "Signal runner",
                            "goals": "Keep the beacon lit.",
                            "relationships_json": "{}",
                            "met": True,
                            "present": True,
                        }
                    ]
                },
            },
        )

    assert applied.status_code == 200
    payload = applied.json()
    assert payload["created_count"] == 1
    assert payload["auto_enhanced_count"] == 1
    character = payload["model"]["characters"][0]
    assert character["name"] == "Mara"
    assert character["goals"] == "Keep the beacon lit."
    assert character["motivations"] == "Protect the lower village from ash riders."
    assert character["boundaries"] == "Will not leave the tower during a lens breach."
    assert character["locked_fields"] == ["boundaries", "motivations"]
    assert runtime.agency_calls == [(save.id, (character["character_id"],))]


def test_message_scene_presence_reads_snapshot_default_and_replaces_rows(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    state.auth_required = False
    scenario = state.repositories.create_scenario(
        type="full_roleplay",
        title="Oracle of Glass",
        premise="A mirrored chamber.",
        player_role="Petitioner",
        content={},
    )
    save = state.repositories.create_save(
        scenario_id=scenario.id,
        title="Oracle Visit",
    )
    message = state.repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The oracle waits beside the basin.",
    )
    character = state.repositories.add_character(
        save_id=save.id,
        name="Oracle of Glass",
        character_id="character-oracle",
    )
    state.repositories.upsert_scene_snapshot(
        save_id=save.id,
        in_world_time="Friday evening after class",
        time_of_day="evening",
        day_of_week="friday",
        world_day_index=5,
        present_character_ids=[character.id],
        source_message_id=message.id,
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        default = client.get(
            f"/api/messages/{message.id}/scene-presence?save_id={save.id}",
        )
        updated = client.post(
            f"/api/messages/{message.id}/scene-presence",
            json={"save_id": save.id, "character_ids": []},
        )

    assert default.status_code == 200
    assert default.json()["latest_message"] is True
    assert default.json()["characters"][0]["present"] is True
    assert updated.status_code == 200
    assert updated.json()["characters"][0]["present"] is False
    assert state.repositories.list_message_scene_presence(save.id) == []
    snapshot = state.repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "Friday evening after class"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "friday"
    assert snapshot.world_day_index == 5
    assert character.id not in snapshot.present_character_ids


def test_message_scene_presence_returns_404_for_unknown_message(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    state.auth_required = False
    scenario = state.repositories.create_scenario(
        type="full_roleplay",
        title="Oracle of Glass",
        premise="A mirrored chamber.",
        player_role="Petitioner",
        content={},
    )
    save = state.repositories.create_save(
        scenario_id=scenario.id,
        title="Oracle Visit",
    )
    state.repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The oracle waits beside the basin.",
    )
    character = state.repositories.add_character(
        save_id=save.id,
        name="Oracle of Glass",
        character_id="character-oracle",
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        fetched = client.get(
            f"/api/messages/missing-message/scene-presence?save_id={save.id}",
        )
        updated = client.post(
            "/api/messages/missing-message/scene-presence",
            json={"save_id": save.id, "character_ids": [character.id]},
        )

    assert fetched.status_code == 404
    assert fetched.json()["detail"] == "Unknown active message id: missing-message"
    assert updated.status_code == 404
    assert updated.json()["detail"] == "Unknown active message id: missing-message"
    assert state.repositories.list_message_scene_presence(save.id) == []


def test_character_registry_enhance_field_commits_current_row_and_returns_model(
    tmp_path: Path,
) -> None:
    class EnhancingRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.repositories: PersistenceRepositories | None = None
            self.enhance_calls: list[tuple[str, str, str, str]] = []

        def enhance_character_registry_field(
            self,
            *,
            active_save_id: str,
            character_id: str,
            field_name: str,
            row: Any,
        ) -> CharacterFieldEnhanceResult:
            assert self.repositories is not None
            self.enhance_calls.append(
                (active_save_id, character_id, field_name, row.role)
            )
            result = CharacterRegistryService(
                self.repositories,
                active_save_id=active_save_id,
            ).apply_edits(
                CharacterRegistryEdits(
                    characters=(
                        replace(
                            row,
                            appearance="A red signal cloak with salt-stained boots.",
                            locked_fields=("appearance",),
                        ),
                    )
                ),
                active_save_id=active_save_id,
            )
            return CharacterFieldEnhanceResult(
                model=result.model,
                character_id=character_id,
                field_name=field_name,
                updated_count=result.updated_count,
            )

    runtime = EnhancingRuntime()
    state = _auth_state(tmp_path, runtime)
    state.auth_required = False
    runtime.repositories = state.repositories
    scenario = state.repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={"opening_message": "The beacon wakes."},
    )
    save = state.repositories.create_save(
        scenario_id=scenario.id,
        title="Lantern Keep",
    )
    character = state.repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal runner",
        appearance="Red cloak.",
    )
    runtime.active_save_id = save.id
    row = CharacterRegistryService(
        state.repositories,
        active_save_id=save.id,
    ).build_model(active_save_id=save.id).characters[0]

    with TestClient(create_app(cast(WebAppState, state))) as client:
        enhanced = client.post(
            f"/api/characters/{character.id}/enhance-field",
            json={
                "active_save_id": save.id,
                "field_name": "appearance",
                "character": {
                    **api_app.to_jsonable(row),
                    "role": "Beacon courier",
                },
            },
        )

    assert enhanced.status_code == 200
    payload = enhanced.json()
    assert payload["character_id"] == character.id
    assert payload["field_name"] == "appearance"
    assert payload["updated_count"] == 1
    updated = payload["model"]["characters"][0]
    assert updated["role"] == "Beacon courier"
    assert updated["appearance"] == "A red signal cloak with salt-stained boots."
    assert updated["locked_fields"] == ["appearance"]
    assert runtime.enhance_calls == [
        (save.id, character.id, "appearance", "Beacon courier")
    ]


def test_character_registry_api_exposes_edge_only_linked_memories(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    state.auth_required = False
    scenario = state.repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower.",
        player_role="Keeper",
        content={"opening_message": "The beacon wakes."},
    )
    save = state.repositories.create_save(
        scenario_id=scenario.id,
        title="Lantern Keep",
    )
    character = state.repositories.add_character(
        save_id=save.id,
        name="Mara",
        character_id="character-mara",
    )
    memory = state.repositories.add_memory(
        save_id=save.id,
        body="Mara knows Ilyra keeps the copper lens key.",
        tags=["mara", "ilyra"],
        memory_id="memory-mara-lens",
    )
    state.repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        confidence=1.0,
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get(f"/api/characters?save_id={save.id}")

    assert response.status_code == 200
    payload = response.json()
    row = payload["characters"][0]
    assert row["linked_memory_ids"] == [memory.id]
    assert state.repositories.list_entity_links(save.id) == []
    target = payload["link_targets"][0]
    assert target["target_type"] == "memory"
    assert target["target_id"] == memory.id
    assert target["linked_character_ids"] == [character.id]


def test_character_knowledge_apply_creates_and_links_memory_and_fact(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes.",
            },
        )
        save_id = created.json()["active_save_id"]
        character = client.post(
            "/api/characters/apply",
            json={
                "active_save_id": save_id,
                "edits": {
                    "characters": [
                        {
                            "character_id": "",
                            "name": "Mara",
                            "role": "Signal runner",
                            "relationships_json": "{}",
                        }
                    ]
                },
            },
        ).json()["model"]["characters"][0]

        applied = client.post(
            f"/api/characters/{character['character_id']}/knowledge/apply",
            json={
                "active_save_id": save_id,
                "actions": [
                    {
                        "action": "create_memory",
                        "body": "Mara knows Ilyra keeps the copper lens key.",
                        "tags": ["mara", "ilyra"],
                        "importance": 0.75,
                    },
                    {
                        "action": "create_world_state",
                        "key": "character.mara.knowledge.lens_key",
                        "category": "character",
                        "confidence": 0.82,
                        "value": {"text": "Ilyra keeps the copper lens key."},
                    },
                ],
            },
        )

    assert applied.status_code == 200
    payload = applied.json()
    assert payload["created_count"] == 2
    row = payload["model"]["characters"][0]
    assert len(row["linked_memory_ids"]) == 1
    assert len(row["linked_state_ids"]) == 1
    targets = {
        (target["target_type"], target["target_id"]): target
        for target in payload["model"]["link_targets"]
    }
    memory = targets[("memory", row["linked_memory_ids"][0])]
    assert memory["body"] == "Mara knows Ilyra keeps the copper lens key."
    assert memory["tags"] == ["mara", "ilyra"]
    fact = targets[("world_state", row["linked_state_ids"][0])]
    assert fact["title"] == "character.mara.knowledge.lens_key"
    assert fact["value"] == {"text": "Ilyra keeps the copper lens key."}
    assert fact["linked_character_ids"] == [character["character_id"]]


def test_character_registry_apply_explicit_locked_fields_can_unlock_character(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes.",
            },
        )
        save_id = created.json()["active_save_id"]
        applied = client.post(
            "/api/characters/apply",
            json={
                "active_save_id": save_id,
                "edits": {
                    "characters": [
                        {
                            "character_id": "",
                            "name": "Mara",
                            "role": "Signal runner",
                            "relationships_json": "{}",
                            "locked_fields": ["voice", "appearance"],
                        }
                    ]
                },
            },
        )
        character = applied.json()["model"]["characters"][0]

        unlocked = client.post(
            "/api/characters/apply",
            json={
                "active_save_id": save_id,
                "edits": {
                    "characters": [
                        {
                            **character,
                            "status": "traveling",
                            "locked_fields": [],
                        }
                    ]
                },
            },
        )

    assert unlocked.status_code == 200
    updated = unlocked.json()["model"]["characters"][0]
    assert updated["status"] == "traveling"
    assert updated["locked_fields"] == []


def test_character_registry_apply_rejects_blank_existing_character_name(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes.",
            },
        )
        save_id = created.json()["active_save_id"]
        applied = client.post(
            "/api/characters/apply",
            json={
                "active_save_id": save_id,
                "edits": {
                    "characters": [
                        {
                            "character_id": "",
                            "name": "Mara",
                            "relationships_json": "{}",
                        }
                    ]
                },
            },
        )
        character_id = applied.json()["model"]["characters"][0]["character_id"]

        rejected = client.post(
            "/api/characters/apply",
            json={
                "active_save_id": save_id,
                "edits": {
                    "characters": [
                        {
                            "character_id": character_id,
                            "name": "   ",
                            "relationships_json": "{}",
                        }
                    ]
                },
            },
        )
        model = client.get("/api/characters").json()

    assert rejected.status_code == 400
    assert "Character name must not be blank" in rejected.json()["detail"]
    assert model["characters"][0]["name"] == "Mara"


def test_world_data_apply_rejects_invalid_edit_payloads(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes.",
            },
        )
        assert created.status_code == 200
        world = client.get("/api/world-data").json()

        invalid_cases = [
            (
                {"scenario": {**world["scenario"], "title": None}},
                "ScenarioEdit.title",
            ),
            (
                {
                    "scenario": {
                        **world["scenario"],
                        "character_starters": "Captain Ilyra",
                    }
                },
                "character_starters must be an array",
            ),
            (
                {
                    "scenario": {
                        **world["scenario"],
                        "character_starters": [
                            {"name": "Captain Ilyra", "met": "yes"}
                        ],
                    }
                },
                "character_starters[0].met",
            ),
            (
                {
                    "scenario": world["scenario"],
                    "world_state": [
                        {
                            "row_id": "",
                            "key": "storm",
                            "category": "weather",
                            "confidence": "0.9",
                            "value_json": "{}",
                            "source_message_id": None,
                        }
                    ],
                },
                "WorldDataStateRow.confidence",
            ),
            (
                {
                    "scenario": world["scenario"],
                    "scene": [],
                },
                "WorldDataSceneRow payload must be an object",
            ),
            (
                {
                    "scenario": world["scenario"],
                    "locations": [
                        {
                            "location_id": "",
                            "name": "Beacon",
                            "locked_fields": "name",
                        }
                    ],
                },
                "WorldDataLocationRow.locked_fields",
            ),
            (
                {
                    "scenario": world["scenario"],
                    "locations": [
                        {
                            "location_id": "",
                            "name": "Beacon",
                            "locked_fields": [7],
                        }
                    ],
                },
                "WorldDataLocationRow.locked_fields[0]",
            ),
        ]

        for edits, expected_detail in invalid_cases:
            response = client.post(
                "/api/world-data/apply",
                json={
                    "active_save_id": world["active_save_id"],
                    "edits": edits,
                },
            )

            assert response.status_code == 400
            assert expected_detail in response.json()["detail"]


def test_character_registry_apply_rejects_invalid_edit_payloads(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")

    with TestClient(create_app()) as client:
        created = client.post(
            "/api/scenarios/manual",
            json={
                "scenario_type": "full_roleplay",
                "title": "Lantern Keep",
                "premise": "A watchtower.",
                "player_role": "Keeper",
                "opening_message": "The beacon wakes.",
            },
        )
        save_id = created.json()["active_save_id"]

        invalid_cases = [
            (
                {
                    "character_id": "",
                    "name": "Mara",
                    "relationships_json": "{}",
                    "met": "false",
                },
                "CharacterRegistryRow.met",
            ),
            (
                {
                    "character_id": "",
                    "name": None,
                    "relationships_json": "{}",
                    "met": False,
                },
                "CharacterRegistryRow.name",
            ),
        ]

        for row, expected_detail in invalid_cases:
            response = client.post(
                "/api/characters/apply",
                json={
                    "active_save_id": save_id,
                    "edits": {"characters": [row]},
                },
            )

            assert response.status_code == 400
            assert expected_detail in response.json()["detail"]

        character = client.post(
            "/api/characters/apply",
            json={
                "active_save_id": save_id,
                "edits": {
                    "characters": [
                        {
                            "character_id": "",
                            "name": "Mara",
                            "relationships_json": "{}",
                        }
                    ]
                },
            },
        ).json()["model"]["characters"][0]
        blank_memory = client.post(
            f"/api/characters/{character['character_id']}/knowledge/apply",
            json={
                "active_save_id": save_id,
                "actions": [
                    {
                        "action": "create_memory",
                        "body": "   ",
                        "tags": [],
                        "importance": 0.5,
                    }
                ],
            },
        )

    assert blank_memory.status_code == 400
    assert "Memory body is required" in blank_memory.json()["detail"]


@dataclass
class _Preview:
    save_id: str = "save-1"
    title: str = "Lantern Keep"
    scenario_title: str = "Storm Sea"
    message_count: int = 2
    media_count: int = 1
    bundle_version: int = 1
    created_at: str | None = None
    updated_at: str | None = None
    exported_at: str | None = None


@dataclass
class _ScenarioPreview:
    scenario_id: str = "scenario-1"
    title: str = "Lantern Keep"
    scenario_type: str = "full_roleplay"
    bundle_version: int = 1
    created_at: str | None = None
    updated_at: str | None = None
    exported_at: str | None = None


@dataclass
class _CharacterPreview:
    character_id: str = "character-1"
    name: str = "Mara"
    suggested_name: str = "Mara"
    name_conflict: bool = False
    media_count: int = 1
    bundle_version: int = 1
    aliases: tuple[str, ...] = ("Ember",)
    role: str = "Signal runner"
    known_state: str = "Carries the amber lens."
    appearance: str = "Ash-dusted cloak."
    personality: str = "Careful and dry-witted."
    voice: str = "Low and clipped."
    status: str = "traveling"
    created_at: str | None = None
    updated_at: str | None = None
    exported_at: str | None = None
    skipped_media_count: int = 0
    warnings: tuple[str, ...] = ()


def _chat_model(body: str) -> dict[str, object]:
    return {
        "active_save_id": "save-1",
        "active_save_title": "Lantern Keep",
        "chronicle": {
            "messages": [
                {
                    "message_id": "narrator-1",
                    "role": "narrator",
                    "speaker_name": None,
                    "body": body,
                    "actions": [],
                }
            ]
        },
        "composer_enabled": True,
        "custom_instructions": "",
        "error": None,
        "failed_save": False,
        "failure_text": None,
        "media": None,
        "model_indicator": "fake / chat",
        "saves": [],
    }


_EXPECTED_CHAT_TURN_PHASES = (
    "submission",
    "history",
    "input",
    "time_state",
    "character_planning",
    "context_selection",
    "prompt",
    "narrator",
    "response_checks",
    "save_narration",
    "action_choices",
)


def _expected_chat_turn_progress(
    status_text: str,
    *,
    succeeded: tuple[str, ...] = (),
    running: str | None = None,
) -> dict[str, object]:
    jobs = []
    for name in _EXPECTED_CHAT_TURN_PHASES:
        if name in succeeded:
            status = "succeeded"
        elif name == running:
            status = "running"
        else:
            status = "pending"
        jobs.append({"name": name, "status": status})
    return {"status_text": status_text, "jobs": jobs}


def _debug_runtime_model(save_id: str) -> dict[str, object]:
    model = _chat_model("The bell answers.")
    model["active_save_id"] = save_id
    chronicle = cast(dict[str, Any], model["chronicle"])
    messages = cast(list[dict[str, Any]], chronicle["messages"])
    message = messages[0]
    message["debug_prompt"] = "secret prompt"
    message["debug_provider_payload"] = {"messages": ["secret"]}
    message["actions"] = [
        {
            "action_id": "inspect-debug-prompt",
            "label": "Inspect prompt",
            "detail_text": "secret prompt",
        },
        {
            "action_id": "inspect-provider-payload",
            "label": "Inspect provider payload",
            "detail_text": "secret provider payload",
        },
        {
            "action_id": "regenerate-message",
            "label": "Regenerate",
        },
        {
            "action_id": "generate-scene-image",
            "label": "Generate image",
        },
        {
            "action_id": "generate-character-image",
            "label": "Generate character image",
        },
    ]
    return model


def _action_ids(message: dict[str, Any]) -> set[str]:
    return {
        str(action.get("action_id"))
        for action in message.get("actions", [])
        if isinstance(action, dict)
    }


class _BlockingCharacterTextProvider:
    provider_name = "fake"

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self._release = threading.Event()
        self._condition = threading.Condition()

    async def chat(self, request: object) -> SimpleNamespace:
        with self._condition:
            self.requests.append(request)
            self._condition.notify_all()
        await asyncio.to_thread(self._release.wait)
        return SimpleNamespace(
            body="Meet me by the arcade after class.",
            provider=getattr(request, "provider", "fake"),
            model_id=getattr(request, "model_id", "fake-chat"),
            token_usage={"total": 12},
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        return _allow_safety_response(request)

    def wait_for_entered(self, count: int) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self.requests) >= count,
                timeout=2.0,
            )

    def release(self) -> None:
        self._release.set()


class _RecordingCharacterTextProvider:
    provider_name = "fake"

    def __init__(
        self,
        response_body: str = "Meet me by the arcade after class.",
    ) -> None:
        self.response_body = response_body
        self.requests: list[Any] = []

    async def chat(self, request: object) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(
            body=self.response_body,
            provider=getattr(request, "provider", "fake"),
            model_id=getattr(request, "model_id", "fake-chat"),
            token_usage={"total": 12},
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        return _allow_safety_response(request)


def _allow_safety_response(
    request: StructuredOutputRequest,
) -> StructuredOutputResponse:
    if request.schema_name != "content_safety_review":
        raise AssertionError(f"unexpected structured schema: {request.schema_name}")
    return StructuredOutputResponse(
        data={
            "action": "allow",
            "category": "none",
            "reason": "Test fixture content is within the ceiling.",
            "minimum_rating": "g",
        },
        provider=request.provider,
        model_id=request.model_id,
    )


def _seed_character_text_exchange(
    repositories: PersistenceRepositories,
    save_id: str,
) -> tuple[str, str, str]:
    player = next(
        character
        for character in repositories.list_characters(save_id)
        if character.is_player_character
    )
    npc = next(
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=npc.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save_id,
        character_id=npc.id,
        title=npc.name,
    )
    player_message = repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="player",
        body="Can we tak after class?",
        content_rating="g",
    )
    reply = repositories.append_character_text_message(
        save_id=save_id,
        thread_id=thread.id,
        character_id=npc.id,
        sender="character",
        body="Sure, meet me by the lockers.",
        provider="fake",
        model="fake-chat",
        content_rating="g",
    )
    return player_message.id, reply.id, thread.id


def _wait_for_terminal_job(
    client: FastAPITestClient,
    job_id: str,
    *,
    save_id: str | None = None,
) -> dict[str, Any]:
    job: dict[str, Any] = {}
    for _ in range(100):
        job = cast(dict[str, Any], client.get(_job_url(job_id, save_id)).json())
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.025)
    return job


def _enter_runtime_lock(lock: RuntimeAccessLock, entered: threading.Event) -> None:
    with lock:
        entered.set()


def _job_url(job_id: str, save_id: str | None) -> str:
    if save_id is None:
        return f"/api/jobs/{job_id}"
    return f"/api/jobs/{job_id}?save_id={save_id}"


def _job_cancel_url(job_id: str, save_id: str | None) -> str:
    if save_id is None:
        return f"/api/jobs/{job_id}/cancel"
    return f"/api/jobs/{job_id}/cancel?save_id={save_id}"


def _create_manual_lantern_save(client: FastAPITestClient) -> str:
    created = client.post(
        "/api/scenarios/manual",
        json={
            "scenario_type": "full_roleplay",
            "title": "Lantern Keep",
            "premise": "A watchtower at the edge of a storm sea.",
            "player_role": "Keeper",
            "opening_message": "The beacon snaps awake.",
        },
    )
    assert created.status_code == 200
    payload = created.json()
    assert payload["active_save_title"] == "Lantern Keep"
    save_id = payload["active_save_id"]
    assert isinstance(save_id, str)
    return save_id


def _submit_chat_and_wait(
    client: FastAPITestClient,
    *,
    save_id: str,
    body: str,
) -> dict[str, Any]:
    submitted = client.post(
        "/api/chat",
        json={"body": body, "speaker_name": "Mara", "save_id": save_id},
    )
    assert submitted.status_code == 200
    job = _wait_for_terminal_job(client, submitted.json()["id"], save_id=save_id)
    assert job["status"] == "succeeded"
    assert job["error"] is None
    assert isinstance(job["result"], dict)
    assert job["result"]["error"] is None
    return job


def _latest_narrator_message_id(state: WebAppState, save_id: str) -> str:
    with state.lock:
        messages = state.repositories.list_messages(save_id)
    narrator_message = next(
        message for message in reversed(messages) if message.role == "narrator"
    )
    return cast(str, narrator_message.id)


def _seed_portability_media_asset(
    state: WebAppState,
    *,
    save_id: str,
    source_message_id: str | None,
) -> tuple[Any, Path, Path]:
    image_relative_path = Path(save_id) / "scene.png"
    thumbnail_relative_path = Path(save_id) / "scene.thumb.png"
    image_path = state.paths.media_dir / image_relative_path
    thumbnail_path = state.paths.media_dir / thumbnail_relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"api portability image")
    thumbnail_path.write_bytes(b"api portability thumbnail")
    with state.lock:
        asset = state.repositories.create_media_asset(
            save_id=save_id,
            source_message_id=source_message_id,
            type="image",
            path=image_relative_path.as_posix(),
            thumbnail_path=thumbnail_relative_path.as_posix(),
            prompt="storm beacon over the sea wall",
            provider="fake",
            model="fake-image",
            status="succeeded",
            metadata={"content_rating": "g"},
        )
    return asset, image_path, thumbnail_path


class _RuntimeDouble:
    def __init__(self) -> None:
        self.active_save_id: str | None = "save-1"
        self.load_save_calls: list[str] = []
        self.character_export_include_private_notes: bool | None = None

    def load_save(self, save_id: str) -> dict[str, object]:
        self.load_save_calls.append(save_id)
        return {"active_save_id": save_id, "active_save_title": "Lantern Keep"}

    def build_model(
        self,
        *,
        active_save_id: str | None | object = ...,
        status: str | None = None,
    ) -> dict[str, object]:
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        return {
            "active_save_id": save_id,
            "active_save_title": "Lantern Keep" if save_id else None,
            "saves": [],
            "status": status,
        }

    def preview_import_bundle(self, bundle_path: Path) -> _Preview:
        assert bundle_path.is_file()
        return _Preview()

    def import_save_bundle(self, bundle_path: Path) -> dict[str, object]:
        assert bundle_path.is_file()
        return {"active_save_id": "save-1", "active_save_title": "Lantern Keep"}

    def preview_import_scenario_bundle(self, bundle_path: Path) -> _ScenarioPreview:
        assert bundle_path.is_file()
        return _ScenarioPreview()

    def import_scenario_bundle(self, bundle_path: Path) -> dict[str, object]:
        assert bundle_path.is_file()
        return {"status": "Imported scenario: Lantern Keep"}

    def preview_import_character_bundle(
        self,
        bundle_path: Path,
        *,
        target_save_id: str | None = None,
    ) -> _CharacterPreview:
        assert bundle_path.is_file()
        assert target_save_id == self.active_save_id
        return _CharacterPreview()

    def import_character_bundle(
        self,
        bundle_path: Path,
        *,
        target_save_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, object]:
        assert bundle_path.is_file()
        assert target_save_id == self.active_save_id
        assert name in {None, "Mara"}
        return {
            "active_save_id": "save-1",
            "characters": [{"character_id": "character-imported", "name": "Mara"}],
        }

    def export_character_bundle(
        self,
        character_id: str,
        bundle_path: Path,
        *,
        include_private_notes: bool = False,
    ) -> SimpleNamespace:
        assert character_id == "character-1"
        self.character_export_include_private_notes = include_private_notes
        bundle_path.write_bytes(b"character-bundle")
        return SimpleNamespace(error=None)

    def export_saved_scenario(
        self,
        scenario_id: str,
        bundle_path: Path,
    ) -> SimpleNamespace:
        assert scenario_id == "scenario-1"
        bundle_path.write_bytes(b"scenario-bundle")
        return SimpleNamespace(error=None)


def test_media_delete_archives_asset_through_runtime(tmp_path: Path) -> None:
    class MediaRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.deleted: list[tuple[str, str | None]] = []

        def delete_media_asset(
            self,
            media_asset_id: str,
            *,
            active_save_id: str | None,
        ) -> dict[str, object]:
            self.deleted.append((media_asset_id, active_save_id))
            return {
                "active_save_id": active_save_id,
                "media": {
                    "latest_scene_image": None,
                    "image_history": [],
                    "media_history": [],
                },
                "error": None,
                "status": "Media deleted",
            }

    runtime = MediaRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_media_assets=lambda _save_id: [SimpleNamespace(id="media-1")],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.delete("/api/media/media-1?save_id=save-1")
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id="save-1")

    assert job["status"] == "succeeded"
    assert job["result"]["status"] == "Media deleted"
    assert runtime.deleted == [("media-1", "save-1")]


def test_media_delete_queues_behind_active_save_job_before_calling_runtime(
    tmp_path: Path,
) -> None:
    class MediaRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.deleted: list[str] = []
            self.cleanup_entered = threading.Event()
            self.release_cleanup = threading.Event()

        async def run_context_cleanup(
            self,
            *,
            active_save_id: str | None = None,
        ) -> dict[str, object]:
            self.cleanup_entered.set()
            await asyncio.to_thread(self.release_cleanup.wait)
            return _chat_model("Cleaned context.")

        def delete_media_asset(
            self,
            media_asset_id: str,
            *,
            active_save_id: str | None,
        ) -> dict[str, object]:
            self.deleted.append(media_asset_id)
            return {"status": "Media deleted", "error": None}

    runtime = MediaRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_media_assets=lambda _save_id: [SimpleNamespace(id="media-1")],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        blocker = client.post(
            "/api/world-data/context-cleanup",
            json={"save_id": "save-1"},
        )
        assert blocker.status_code == 200
        assert runtime.cleanup_entered.wait(timeout=1.0)

        response = client.delete("/api/media/media-1?save_id=save-1")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert runtime.deleted == []

        runtime.release_cleanup.set()
        job = _wait_for_terminal_job(client, response.json()["id"], save_id="save-1")
        blocker_job = _wait_for_terminal_job(
            client,
            blocker.json()["id"],
            save_id="save-1",
        )

    assert blocker_job["status"] == "succeeded"
    assert job["status"] == "succeeded"
    assert runtime.deleted == ["media-1"]


def test_media_delete_returns_runtime_error(tmp_path: Path) -> None:
    class MissingMediaRuntime(_RuntimeDouble):
        def delete_media_asset(
            self,
            media_asset_id: str,
            *,
            active_save_id: str | None,
        ) -> dict[str, object]:
            return {"error": "Media asset not found"}

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, MissingMediaRuntime())))
    ) as client:
        response = client.delete("/api/media/missing?save_id=save-1")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown media asset"


def test_media_set_character_reference_passes_asset_and_save_to_runtime(
    tmp_path: Path,
) -> None:
    class MediaRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.references: list[tuple[str, str | None]] = []

        def set_character_reference_image(
            self,
            media_asset_id: str,
            *,
            active_save_id: str | None,
        ) -> dict[str, object]:
            self.references.append((media_asset_id, active_save_id))
            return {
                "active_save_id": active_save_id,
                "saves": [],
                "chronicle": {"messages": []},
                "media": {
                    "character_reference_image": {"id": media_asset_id},
                    "latest_scene_image": None,
                    "image_history": [],
                    "media_history": [],
                },
                "error": None,
                "status": "Character reference image updated",
            }

    runtime = MediaRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_media_assets=lambda _save_id: [SimpleNamespace(id="media-1")],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/media/media-1/set-character-reference",
            json={"save_id": "save-1"},
        )
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id="save-1")

    assert job["status"] == "succeeded"
    assert job["result"]["status"] == "Character reference image updated"
    assert runtime.references == [("media-1", "save-1")]


def test_media_set_character_reference_queues_behind_active_save_job(
    tmp_path: Path,
) -> None:
    class MediaRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.references: list[str] = []
            self.cleanup_entered = threading.Event()
            self.release_cleanup = threading.Event()

        async def run_context_cleanup(
            self,
            *,
            active_save_id: str | None = None,
        ) -> dict[str, object]:
            self.cleanup_entered.set()
            await asyncio.to_thread(self.release_cleanup.wait)
            return _chat_model("Cleaned context.")

        def set_character_reference_image(
            self,
            media_asset_id: str,
            *,
            active_save_id: str | None,
        ) -> dict[str, object]:
            self.references.append(media_asset_id)
            return {"status": "Character reference image updated", "error": None}

    runtime = MediaRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_media_assets=lambda _save_id: [SimpleNamespace(id="media-1")],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        blocker = client.post(
            "/api/world-data/context-cleanup",
            json={"save_id": "save-1"},
        )
        assert blocker.status_code == 200
        assert runtime.cleanup_entered.wait(timeout=1.0)

        response = client.post(
            "/api/media/media-1/set-character-reference",
            json={"save_id": "save-1"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert runtime.references == []

        runtime.release_cleanup.set()
        job = _wait_for_terminal_job(client, response.json()["id"], save_id="save-1")
        blocker_job = _wait_for_terminal_job(
            client,
            blocker.json()["id"],
            save_id="save-1",
        )

    assert blocker_job["status"] == "succeeded"
    assert job["status"] == "succeeded"
    assert runtime.references == ["media-1"]


def test_character_registry_set_reference_endpoint_queues_scoped_runtime_call(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    scenario = state.repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={},
    )
    save = state.repositories.create_save(
        scenario_id=scenario.id,
        title="Lantern Keep",
    )
    character = state.repositories.add_character(
        save_id=save.id,
        name="Mara",
        character_id="character-1",
    )
    previous_reference = state.repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path=f"{save.id}/reference.png",
        thumbnail_path=None,
        prompt="Mara old reference",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "character_reference", "character_id": character.id},
    )
    candidate = state.repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path=f"{save.id}/candidate.png",
        thumbnail_path=None,
        prompt="Mara in a storm cloak",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "character_image", "character_id": character.id},
    )
    state.repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="media_asset",
        target_id=previous_reference.id,
        relation="reference_image",
    )

    class CharacterReferenceRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.references: list[tuple[str, str | None, str | None]] = []

        def set_character_reference_image(
            self,
            media_asset_id: str,
            *,
            character_id: str | None = None,
            active_save_id: str | None,
        ) -> dict[str, object]:
            self.references.append((media_asset_id, character_id, active_save_id))
            assert active_save_id is not None
            assert character_id is not None
            for link in state.repositories.list_entity_links(active_save_id):
                if (
                    link.entity_type == "character"
                    and link.entity_id == character_id
                    and link.target_type == "media_asset"
                    and link.relation == "reference_image"
                ):
                    state.repositories.delete_entity_link(link.id)
            state.repositories.add_entity_link(
                save_id=active_save_id,
                entity_type="character",
                entity_id=character_id,
                target_type="media_asset",
                target_id=media_asset_id,
                relation="reference_image",
            )
            return {
                "active_save_id": active_save_id,
                "status": "Character reference image updated",
                "error": None,
            }

    runtime = CharacterReferenceRuntime()
    state.runtime = runtime

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/characters/character-1/reference-image/set",
            json={"save_id": save.id, "media_asset_id": candidate.id},
        )
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id=save.id)

    assert job["status"] == "succeeded"
    assert job["type"] == "character_reference_set"
    assert runtime.references == [(candidate.id, character.id, save.id)]
    [row] = job["result"]["characters"]
    assert row["reference_image"]["media_asset_id"] == candidate.id
    assert [image["media_asset_id"] for image in row["generated_images"]] == [
        previous_reference.id
    ]


def test_media_set_character_reference_returns_unknown_media_404(
    tmp_path: Path,
) -> None:
    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, _RuntimeDouble())))
    ) as client:
        response = client.post(
            "/api/media/missing/set-character-reference",
            json={"save_id": "save-1"},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown media asset"


def test_media_upload_character_reference_passes_file_and_save_to_runtime(
    tmp_path: Path,
) -> None:
    class MediaRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.uploads: list[tuple[bytes, str | None, bool, str | None]] = []

        def upload_character_reference_image(
            self,
            *,
            image_bytes: bytes,
            filename: str | None,
            replace_existing: bool,
            active_save_id: str | None,
        ) -> dict[str, object]:
            self.uploads.append(
                (image_bytes, filename, replace_existing, active_save_id)
            )
            return {
                "active_save_id": active_save_id,
                "saves": [],
                "chronicle": {"messages": []},
                "media": {
                    "character_reference_image": {"id": "media-upload"},
                    "latest_scene_image": None,
                    "image_history": [],
                    "media_history": [],
                },
                "error": None,
                "status": "Character reference image uploaded",
            }

    runtime = MediaRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories.get_effective_setting = (
        lambda key, **_kwargs: "unrated"
        if key == "content_filter_rating"
        else None
    )
    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/media/character-reference/upload",
            data={"save_id": "save-1", "replace_existing": "true"},
            files={"file": ("portrait.png", b"\x89PNG\r\n\x1a\nimage", "image/png")},
        )
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id="save-1")

    assert job["status"] == "succeeded"
    assert job["result"]["status"] == "Character reference image uploaded"
    assert runtime.uploads == [
        (b"\x89PNG\r\n\x1a\nimage", "portrait.png", True, "save-1")
    ]


def test_rated_reference_upload_is_rejected_before_persistence(
    tmp_path: Path,
) -> None:
    class NoUploadRuntime(_RuntimeDouble):
        def upload_character_reference_image(self, **_kwargs: object) -> NoReturn:
            raise AssertionError("rated upload must not reach persistence")

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, NoUploadRuntime())))
    ) as client:
        response = client.post(
            "/api/media/character-reference/upload",
            data={"save_id": "save-1"},
            files={
                "file": (
                    "portrait.png",
                    b"\x89PNG\r\n\x1a\nimage",
                    "image/png",
                )
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Uploaded images cannot be safety-reviewed. Set the content rating "
        "to Unrated before uploading a reference image."
    )


def test_media_upload_character_reference_maps_runtime_errors(
    tmp_path: Path,
) -> None:
    class FailingUploadRuntime(_RuntimeDouble):
        def upload_character_reference_image(
            self,
            *,
            image_bytes: bytes,
            filename: str | None,
            replace_existing: bool,
            active_save_id: str | None,
        ) -> dict[str, object]:
            return {"error": "Unsupported image upload type; use PNG, JPEG, or WebP"}

    state = _state_double(tmp_path, FailingUploadRuntime())
    state.repositories.get_effective_setting = (
        lambda key, **_kwargs: "unrated"
        if key == "content_filter_rating"
        else None
    )
    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/media/character-reference/upload",
            data={"save_id": "save-1"},
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 200
        job = _wait_for_terminal_job(client, response.json()["id"], save_id="save-1")

    assert job["status"] == "failed"
    assert job["error"] == SAFE_JOB_ERROR


def test_media_remove_character_reference_calls_runtime(tmp_path: Path) -> None:
    class MediaRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.removed_save_ids: list[str | None] = []

        def remove_character_reference_image(
            self,
            *,
            active_save_id: str | None,
        ) -> dict[str, object]:
            self.removed_save_ids.append(active_save_id)
            return {
                "active_save_id": active_save_id,
                "saves": [],
                "chronicle": {"messages": []},
                "media": {
                    "character_reference_image": None,
                    "latest_scene_image": None,
                    "image_history": [],
                    "media_history": [],
                },
                "error": None,
                "status": "Character reference image removed",
            }

    runtime = MediaRuntime()

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        response = client.post(
            "/api/media/character-reference/remove",
            json={"save_id": "save-1"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "Character reference image removed"
    assert runtime.removed_save_ids == ["save-1"]


def test_media_asset_serves_persisted_video_with_mime_type(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    video_path = media_dir / "save-1" / "motion.webm"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake webm payload")
    state = _state_double(tmp_path)
    def list_media_assets(save_id: str) -> list[object]:
        if save_id != "save-1":
            return []
        return [
            SimpleNamespace(
                id="media-video",
                path="save-1/motion.webm",
                mime_type="video/webm",
                metadata_json='{"content_rating":"g"}',
            )
        ]

    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_media_assets=list_media_assets,
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        unscoped = client.get("/api/media/media-video")
        wrong_save = client.get("/api/media/media-video?save_id=save-2")
        response = client.get("/api/media/media-video?save_id=save-1")

    assert unscoped.status_code == 400
    assert unscoped.json()["detail"] == _SAVE_ID_REQUIRED_DETAIL
    assert wrong_save.status_code == 404
    assert wrong_save.json()["detail"] == "Unknown media asset"
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/webm")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, max-age=31536000, immutable"
    assert response.content == b"fake webm payload"


def test_media_asset_serves_legacy_active_mime_type_as_inert(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_path = media_dir / "save-1" / "imported.html"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"<script>alert('x')</script>")
    state = _state_double(tmp_path)

    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_media_assets=lambda _save_id: [
            SimpleNamespace(
                id="media-active",
                path="save-1/imported.html",
                mime_type="text/html",
                metadata_json='{"content_rating":"g"}',
            )
        ],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/media/media-active?save_id=save-1")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.content == b"<script>alert('x')</script>"


def test_media_thumbnail_serves_persisted_thumbnail_with_cache_headers(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_path = media_dir / "save-1" / "scene.png"
    thumbnail_path = media_dir / "save-1" / "thumbnails" / "scene.png"
    original_path.parent.mkdir(parents=True)
    thumbnail_path.parent.mkdir(parents=True)
    original_path.write_bytes(b"original image payload")
    thumbnail_path.write_bytes(b"thumbnail image payload")
    state = _state_double(tmp_path)
    assets = {
        ("save-1", "media-image"): SimpleNamespace(
            id="media-image",
            path="save-1/scene.png",
            thumbnail_path="save-1/thumbnails/scene.png",
            mime_type="image/png",
            type="image",
            metadata_json='{"content_rating":"g"}',
        )
    }
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        get_media_asset=lambda save_id, media_asset_id: assets.get(
            (save_id, media_asset_id)
        ),
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        unscoped = client.get("/api/media/media-image/thumbnail")
        wrong_save = client.get("/api/media/media-image/thumbnail?save_id=save-2")
        response = client.get("/api/media/media-image/thumbnail?save_id=save-1")

    assert unscoped.status_code == 400
    assert unscoped.json()["detail"] == _SAVE_ID_REQUIRED_DETAIL
    assert wrong_save.status_code == 404
    assert wrong_save.json()["detail"] == "Unknown media asset"
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, max-age=31536000, immutable"
    assert response.content == b"thumbnail image payload"


def test_media_thumbnail_falls_back_to_original_image_when_thumbnail_is_missing(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_path = media_dir / "save-1" / "scene.jpg"
    original_path.parent.mkdir(parents=True)
    original_path.write_bytes(b"original jpeg payload")
    state = _state_double(tmp_path)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        get_media_asset=lambda save_id, media_asset_id: (
            SimpleNamespace(
                id="media-image",
                path="save-1/scene.jpg",
                thumbnail_path="save-1/thumbnails/missing.png",
                mime_type="image/jpeg",
                type="image",
                metadata_json='{"content_rating":"g"}',
            )
            if save_id == "save-1" and media_asset_id == "media-image"
            else None
        ),
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/media/media-image/thumbnail?save_id=save-1")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.content == b"original jpeg payload"


def test_media_thumbnail_falls_back_to_original_for_unusable_placeholder_thumbnail(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    original_path = media_dir / "save-1" / "scene.png"
    thumbnail_path = media_dir / "save-1" / "thumbnails" / "scene.png"
    original_path.parent.mkdir(parents=True)
    thumbnail_path.parent.mkdir(parents=True)
    original_path.write_bytes(b"original image payload")
    thumbnail_path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
            "1f15c4890000000a49444154789c636000000200015e027fea00000000"
            "49454e44ae426082"
        )
    )
    state = _state_double(tmp_path)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        get_media_asset=lambda save_id, media_asset_id: (
            SimpleNamespace(
                id="media-image",
                path="save-1/scene.png",
                thumbnail_path="save-1/thumbnails/scene.png",
                mime_type="image/png",
                type="image",
                metadata_json='{"content_rating":"g"}',
            )
            if save_id == "save-1" and media_asset_id == "media-image"
            else None
        ),
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/media/media-image/thumbnail?save_id=save-1")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == b"original image payload"


def test_media_thumbnail_rejects_paths_that_escape_media_dir(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    original_path = media_dir / "save-1" / "scene.png"
    original_path.parent.mkdir(parents=True)
    original_path.write_bytes(b"original image payload")
    outside_file = tmp_path / "outside.png"
    outside_file.write_bytes(b"outside thumbnail payload")
    state = _state_double(tmp_path)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        get_media_asset=lambda save_id, media_asset_id: (
            SimpleNamespace(
                id="media-image",
                path="save-1/scene.png",
                thumbnail_path="../outside.png",
                mime_type="image/png",
                type="image",
                metadata_json='{"content_rating":"g"}',
            )
            if save_id == "save-1" and media_asset_id == "media-image"
            else None
        ),
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/media/media-image/thumbnail?save_id=save-1")

    assert response.status_code == 404
    assert response.json()["detail"] == "Media file not found"
    assert b"outside thumbnail payload" not in response.content


def test_media_thumbnail_returns_not_found_for_video_without_thumbnail(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    video_path = media_dir / "save-1" / "motion.webm"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake webm payload")
    state = _state_double(tmp_path)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        get_media_asset=lambda save_id, media_asset_id: (
            SimpleNamespace(
                id="media-video",
                path="save-1/motion.webm",
                thumbnail_path=None,
                mime_type="video/webm",
                type="video",
                metadata_json='{"content_rating":"g"}',
            )
            if save_id == "save-1" and media_asset_id == "media-video"
            else None
        ),
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/media/media-video/thumbnail?save_id=save-1")

    assert response.status_code == 404
    assert response.json()["detail"] == "Media thumbnail not found"


def test_media_asset_returns_not_found_for_unknown_asset(tmp_path: Path) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/media/missing?save_id=save-1")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown media asset"


def test_media_asset_returns_not_found_for_missing_file(tmp_path: Path) -> None:
    state = _state_double(tmp_path)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_media_assets=lambda _save_id: [
            SimpleNamespace(
                id="media-missing",
                path="save-1/missing.png",
                mime_type="image/png",
                metadata_json='{"content_rating":"g"}',
            )
        ],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/media/media-missing?save_id=save-1")

    assert response.status_code == 404
    assert response.json()["detail"] == "Media file not found"


def test_media_asset_rejects_paths_that_escape_media_dir(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    outside_file = tmp_path / "outside.png"
    outside_file.write_bytes(b"outside media")
    state = _state_double(tmp_path)
    path_cases = ["../outside.png", str(outside_file)]

    with TestClient(create_app(cast(WebAppState, state))) as client:
        for index, asset_path in enumerate(path_cases):
            state.repositories = SimpleNamespace(
                list_saves=lambda: [SimpleNamespace(id="save-1")],
                list_media_assets=lambda _save_id, path=asset_path, i=index: [
                    SimpleNamespace(
                        id=f"media-escape-{i}",
                        path=path,
                        mime_type="image/png",
                        metadata_json='{"content_rating":"g"}',
                    )
                ],
            )

            response = client.get(f"/api/media/media-escape-{index}?save_id=save-1")

            assert response.status_code == 404
            assert response.json()["detail"] == "Media file not found"
            assert b"outside media" not in response.content


def test_media_prompt_endpoint_returns_raw_prompt(tmp_path: Path) -> None:
    state = _state_double(tmp_path)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        get_media_asset=lambda save_id, media_asset_id: (
            SimpleNamespace(
                id="media-1",
                type="image",
                provider="fake",
                status="succeeded",
                prompt="raw provider prompt with exact user-editable details",
                metadata_json='{"content_rating":"g"}',
            )
            if save_id == "save-1" and media_asset_id == "media-1"
            else None
        ),
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.get("/api/media/media-1/prompt?save_id=save-1")

    assert response.status_code == 200
    assert response.json() == {
        "media_asset_id": "media-1",
        "prompt": "raw provider prompt with exact user-editable details",
    }


def test_media_prompt_endpoint_hides_over_rating_prompt_from_child(
    tmp_path: Path,
) -> None:
    state = _auth_state(tmp_path)
    child = state.auth_service().create_user(
        username="Ilyra",
        password="correct horse",
        role="child",
    )
    save = _create_auth_save(
        state.repositories,
        title="Child Save",
        owner_user_id=child.id,
    )
    asset = state.repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="media/adult-scene.png",
        prompt="Explicit sex between adults.",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"content_rating": "r"},
    )

    with TestClient(create_app(cast(WebAppState, state)), authenticate=False) as client:
        assert client.post(
            "/api/auth/login",
            json={"username": "Ilyra", "password": "correct horse"},
        ).status_code == 200
        response = client.get(
            f"/api/media/{asset.id}/prompt?save_id={save.id}",
        )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Media exceeds your content rating",
    }


def test_media_regenerate_creates_job_for_source_message(tmp_path: Path) -> None:
    class RegenerateMediaRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.generate_calls: list[tuple[str, str | None]] = []

        async def generate_image(
            self,
            *,
            source_message_id: str,
            active_save_id: str | None | object = ...,
            retry_progress_callback: Callable[[object], None] | None = None,
        ) -> dict[str, object]:
            self.generate_calls.append(
                (
                    source_message_id,
                    active_save_id if isinstance(active_save_id, str) else None,
                )
            )
            if retry_progress_callback is not None:
                retry_progress_callback(SimpleNamespace(next_attempt=2, max_attempts=3))
            return _chat_model("The image returns sharper.")

    runtime = RegenerateMediaRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_messages=lambda _save_id: [
            SimpleNamespace(id="message-1", content_rating="g")
        ],
        list_media_assets=lambda _save_id: [
            SimpleNamespace(
                id="media-1",
                source_message_id="message-1",
                type="image",
                provider="fake",
                status="succeeded",
            )
        ],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/media/media-1/regenerate",
            json={"save_id": "save-1"},
        )
        assert created.status_code == 200
        job = _wait_for_terminal_job(client, created.json()["id"], save_id="save-1")

    assert job["status"] == "succeeded"
    assert job["type"] == "image_regeneration"
    assert job["save_id"] == "save-1"
    assert runtime.generate_calls == [("message-1", "save-1")]


def test_media_regenerate_with_prompt_replaces_asset_through_runtime(
    tmp_path: Path,
) -> None:
    class RegenerateMediaRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.regenerate_calls: list[tuple[str, str, str | None]] = []

        async def regenerate_media_asset(
            self,
            media_asset_id: str,
            *,
            prompt: str,
            active_save_id: str | None | object = ...,
            retry_progress_callback: Callable[[object], None] | None = None,
        ) -> dict[str, object]:
            self.regenerate_calls.append(
                (
                    media_asset_id,
                    prompt,
                    active_save_id if isinstance(active_save_id, str) else None,
                )
            )
            if retry_progress_callback is not None:
                retry_progress_callback(SimpleNamespace(next_attempt=2, max_attempts=3))
            return _chat_model("The edited image replaces the old one.")

    runtime = RegenerateMediaRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_messages=lambda _save_id: [
            SimpleNamespace(id="message-1", content_rating="g")
        ],
        list_media_assets=lambda _save_id: [
            SimpleNamespace(
                id="media-1",
                source_message_id=None,
                type="image",
                provider="fake",
                status="succeeded",
            )
        ],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/media/media-1/regenerate",
            json={
                "save_id": "save-1",
                "prompt": "edited replacement prompt",
            },
        )
        assert created.status_code == 200
        job = _wait_for_terminal_job(client, created.json()["id"], save_id="save-1")

    assert job["status"] == "succeeded"
    assert job["type"] == "image_regeneration"
    assert runtime.regenerate_calls == [
        ("media-1", "edited replacement prompt", "save-1")
    ]


def test_media_regenerate_rejects_unknown_asset(tmp_path: Path) -> None:
    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, _RuntimeDouble())))
    ) as client:
        response = client.post(
            "/api/media/missing/regenerate",
            json={"save_id": "save-1"},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown media asset"


def test_media_regenerate_rejects_asset_without_source_message(
    tmp_path: Path,
) -> None:
    state = _state_double(tmp_path)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_media_assets=lambda _save_id: [
            SimpleNamespace(
                id="media-1",
                source_message_id=None,
                type="image",
                provider="fake",
                status="succeeded",
            )
        ],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/media/media-1/regenerate",
            json={"save_id": "save-1"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Media asset has no source message"


def test_media_regenerate_job_fails_when_runtime_returns_error(
    tmp_path: Path,
) -> None:
    class FailingRegenerateRuntime(_RuntimeDouble):
        async def generate_image(
            self,
            *,
            source_message_id: str,
            active_save_id: str | None | object = ...,
            retry_progress_callback: Callable[[object], None] | None = None,
        ) -> dict[str, object]:
            model = _chat_model("The image failed.")
            model["error"] = "provider_error: image retry failed"
            return model

    state = _state_double(tmp_path, FailingRegenerateRuntime())
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_media_assets=lambda _save_id: [
            SimpleNamespace(
                id="media-1",
                source_message_id="message-1",
                type="image",
                provider="fake",
                status="succeeded",
            )
        ],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/media/media-1/regenerate",
            json={"save_id": "save-1"},
        )
        assert created.status_code == 200
        job = _wait_for_terminal_job(client, created.json()["id"], save_id="save-1")

    assert job["status"] == "failed"
    assert job["type"] == "image_regeneration"
    assert job["error"] == SAFE_JOB_ERROR


def test_media_animate_creates_job_through_runtime(tmp_path: Path) -> None:
    class AnimationRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.animation_calls: list[tuple[str, str, str | None]] = []

        async def animate_media_asset(
            self,
            media_asset_id: str,
            *,
            motion_prompt: str,
            active_save_id: str | None,
        ) -> dict[str, object]:
            self.animation_calls.append(
                (media_asset_id, motion_prompt, active_save_id)
            )
            model = _chat_model("The beacon catches motion.")
            model["active_save_id"] = active_save_id
            model["media"] = {
                "latest_scene_image": None,
                "image_history": [],
                "media_history": [
                    {
                        "id": "media-video",
                        "source_message_id": "message-1",
                        "source_media_asset_id": media_asset_id,
                        "type": "video",
                        "mime_type": "video/mp4",
                        "prompt_preview": "The beacon catches motion.",
                        "status": "succeeded",
                        "created_at": None,
                        "can_animate": False,
                    }
                ],
                "image_animation_available": True,
            }
            return model

    runtime = AnimationRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_messages=lambda _save_id: [
            SimpleNamespace(id="message-1", content_rating="g")
        ],
        list_media_assets=lambda _save_id: [
            SimpleNamespace(
                id="media-1",
                source_message_id="message-1",
                type="image",
                metadata_json='{"content_rating":"g"}',
            )
        ],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/media/media-1/animate",
            json={
                "save_id": "save-1",
                "motion_prompt": "make the flame gutter once",
            },
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert job["type"] == "image_animation"
    assert runtime.animation_calls == [
        ("media-1", "make the flame gutter once", "save-1")
    ]


def test_media_animate_job_fails_when_runtime_returns_error(
    tmp_path: Path,
) -> None:
    class FailingAnimationRuntime(_RuntimeDouble):
        async def animate_media_asset(
            self,
            media_asset_id: str,
            *,
            motion_prompt: str,
            active_save_id: str | None,
        ) -> dict[str, object]:
            model = _chat_model("The beacon catches motion.")
            model["active_save_id"] = active_save_id
            model["error"] = "provider_error: video timed out"
            return model

    state = _state_double(tmp_path, FailingAnimationRuntime())
    state.repositories = SimpleNamespace(
        list_saves=lambda: [SimpleNamespace(id="save-1")],
        list_messages=lambda _save_id: [
            SimpleNamespace(id="message-1", content_rating="g")
        ],
        list_media_assets=lambda _save_id: [
            SimpleNamespace(
                id="media-1",
                source_message_id="message-1",
                type="image",
                metadata_json='{"content_rating":"g"}',
            )
        ],
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/media/media-1/animate",
            json={"save_id": "save-1", "motion_prompt": "slow drift"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "failed"
    assert job["type"] == "image_animation"
    assert job["error"] == SAFE_JOB_ERROR


def test_media_generate_forwards_requested_save_id(tmp_path: Path) -> None:
    class MediaRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.generate_calls: list[tuple[str, str | None]] = []

        async def generate_image(
            self,
            *,
            source_message_id: str,
            active_save_id: str | None | object = ...,
            retry_progress_callback: Callable[[object], None] | None = None,
        ) -> dict[str, object]:
            self.generate_calls.append(
                (
                    source_message_id,
                    active_save_id if isinstance(active_save_id, str) else None,
                )
            )
            if retry_progress_callback is not None:
                retry_progress_callback(SimpleNamespace(next_attempt=2, max_attempts=3))
            return _chat_model("The scene catches the lamplight.")

    runtime = MediaRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/media/generate",
            json={"message_id": "message-1", "save_id": "save-1"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert job["type"] == "image_generation"
    assert job["save_id"] == "save-1"
    assert runtime.generate_calls == [("message-1", "save-1")]


def test_character_image_generation_endpoint_queues_runtime_call(
    tmp_path: Path,
) -> None:
    class CharacterImageRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.character_image_calls: list[tuple[str, str, str | None]] = []

        async def generate_character_image(
            self,
            *,
            source_message_id: str,
            character_id: str,
            active_save_id: str | None | object = ...,
            retry_progress_callback: Callable[[object], None] | None = None,
        ) -> dict[str, object]:
            self.character_image_calls.append(
                (
                    source_message_id,
                    character_id,
                    active_save_id if isinstance(active_save_id, str) else None,
                )
            )
            if retry_progress_callback is not None:
                retry_progress_callback(SimpleNamespace(next_attempt=2, max_attempts=3))
            return _chat_model("The character image catches the lamplight.")

    runtime = CharacterImageRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/media/generate-character-image",
            json={
                "message_id": "message-1",
                "character_id": "character-1",
                "save_id": "save-1",
            },
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert job["type"] == "character_image_generation"
    assert runtime.character_image_calls == [("message-1", "character-1", "save-1")]


def test_character_registry_image_endpoint_queues_runtime_call(
    tmp_path: Path,
) -> None:
    class CharacterImageRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.registry_image_calls: list[tuple[str, str, str | None]] = []

        async def generate_character_registry_image(
            self,
            character_id: str,
            *,
            instructions: str = "",
            active_save_id: str | None | object = ...,
            retry_progress_callback: Callable[[object], None] | None = None,
        ) -> dict[str, object]:
            self.registry_image_calls.append(
                (
                    character_id,
                    instructions,
                    active_save_id if isinstance(active_save_id, str) else None,
                )
            )
            if retry_progress_callback is not None:
                retry_progress_callback(SimpleNamespace(next_attempt=2, max_attempts=3))
            return _chat_model("The registry image catches the lamplight.")

    runtime = CharacterImageRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/characters/character-1/image/generate",
            json={"save_id": "save-1", "instructions": "moonlit profile"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert job["type"] == "character_image_generation"
    assert runtime.registry_image_calls == [
        ("character-1", "moonlit profile", "save-1")
    ]


def test_initial_media_uses_opening_image_progress_for_roleplay_saves(
    tmp_path: Path,
) -> None:
    class CharacterRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.initial_calls: list[tuple[str, str | None]] = []

        async def generate_initial_scenario_image(
            self,
            *,
            source_message_id: str,
            active_save_id: str | None | object = ...,
            retry_progress_callback: Callable[[object], None] | None = None,
        ) -> dict[str, object]:
            self.initial_calls.append(
                (
                    source_message_id,
                    active_save_id if isinstance(active_save_id, str) else None,
                )
            )
            if retry_progress_callback is not None:
                retry_progress_callback(SimpleNamespace(next_attempt=2, max_attempts=3))
            return _chat_model("The reference catches the lamplight.")

    runtime = CharacterRuntime()
    state = _state_double(tmp_path, runtime)
    state.repositories = SimpleNamespace(
        list_saves=lambda: [],
        list_media_assets=lambda _save_id: [],
        get_save=lambda save_id: SimpleNamespace(id=save_id, scenario_id="scenario-1"),
        get_scenario=lambda scenario_id: SimpleNamespace(
            id=scenario_id,
            type="full_roleplay",
        ),
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        created = client.post(
            "/api/media/initial",
            json={"message_id": "message-1", "save_id": "save-1"},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        for _ in range(20):
            job = client.get(_job_url(job_id, "save-1")).json()
            if job["status"] != "running":
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert runtime.initial_calls == [("message-1", "save-1")]

    async def collect_events() -> list[str]:
        return [
            chunk
            async for chunk in api_app._event_stream(  # noqa: SLF001 - SSE regression
                cast(WebAppState, state),
                job_id,
            )
        ]

    chunks = asyncio.run(collect_events())
    assert any("Generating opening image" in chunk for chunk in chunks)


def _state_double(tmp_path: Path, runtime: object | None = None) -> SimpleNamespace:
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    return SimpleNamespace(
        auth_required=False,
        runtime=runtime or _RuntimeDouble(),
        lock=RuntimeAccessLock(),
        jobs=JobRegistry(),
        save_events=SaveEventHub(),
        providers={},
        repositories=SimpleNamespace(
            list_saves=lambda: [],
            list_media_assets=lambda _save_id: [],
        ),
        settings_service=lambda: SimpleNamespace(secret_storage_warning=lambda: None),
        secret_store=SimpleNamespace(),
        paths=SimpleNamespace(temp_dir=temp_dir, media_dir=tmp_path / "media"),
        bundle_previews={},
        scenario_bundle_previews={},
        character_bundle_previews={},
        log_file_path=None,
    )


class _TrackingJobRegistry(JobRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.list_active_save_ids: list[str | None] = []

    def list_active(self, *, save_id: str | None = None) -> list[JobRecord]:
        self.list_active_save_ids.append(save_id)
        return super().list_active(save_id=save_id)


def _auth_state(tmp_path: Path, runtime: object | None = None) -> SimpleNamespace:
    state = _state_double(tmp_path, runtime)
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    state.repositories = PersistenceRepositories(
        sqlite3.connect(database_path, check_same_thread=False)
    )
    state.repositories.set_app_setting("content_filter_rating", "unrated")
    state.auth_required = True
    state.auth_attempts = AuthAttemptThrottle()
    state.auth_service = lambda: AuthService(repositories=state.repositories)
    return state


def _scoped_auth_state(
    tmp_path: Path,
    runtime: object | None = None,
) -> SimpleNamespace:
    state = _state_double(tmp_path, runtime)
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    state.repositories = runtime_module.ScopedPersistenceRepositories(
        database_path,
        PersistenceRepositories,
    )
    state.repositories.set_app_setting("content_filter_rating", "unrated")
    state.auth_required = True
    state.auth_attempts = AuthAttemptThrottle()
    state.auth_service = lambda: AuthService(repositories=state.repositories)
    return state


class PausingHasher:
    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier

    def hash(self, password: str) -> str:
        self._barrier.wait(timeout=5.0)
        return f"hash:{password}"

    def verify(self, password_hash: str, password: str) -> bool:
        return password_hash == f"hash:{password}"

    def check_needs_rehash(self, password_hash: str) -> bool:
        return False


def _test_session_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _create_auth_save(
    repositories: PersistenceRepositories,
    *,
    title: str,
    owner_user_id: str | None,
) -> SaveRecord:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title=f"{title} Scenario",
        premise="A beacon is going dark.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    return repositories.create_save(
        scenario_id=scenario.id,
        title=title,
        owner_user_id=owner_user_id,
    )


def _create_dating_auth_save(
    repositories: PersistenceRepositories,
    *,
    owner_user_id: str | None,
) -> str:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Summer Routes",
        premise="A school summer route story.",
        player_role="Transfer student",
        content={
            "player_character_name": "Ren Takahashi",
        },
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Summer Routes",
        owner_user_id=owner_user_id,
    )
    player = repositories.add_character(
        save_id=save.id,
        name="Ren Takahashi",
        aliases=["Ren"],
        is_player_character=True,
        met=True,
    )
    repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        relationships={player.name: "romance option for Ren Takahashi"},
        status="available romance option at scenario start",
        met=True,
    )
    return save.id


async def _collect_save_event_chunks(
    state: WebAppState,
    save_id: str,
    *,
    last_event_id: int,
    count: int,
    current_user: UserRecord | None = None,
) -> list[str]:
    stream = cast(
        AsyncGenerator[str, None],
        api_app._save_event_stream(  # noqa: SLF001 - SSE regression helper
            state,
            save_id,
            last_event_id,
            current_user=current_user,
            current_user_role=getattr(current_user, "role", None),
        ),
    )
    chunks: list[str] = []
    try:
        for _ in range(count):
            chunks.append(await asyncio.wait_for(anext(stream), timeout=1.0))
    finally:
        await stream.aclose()
    return chunks


def _save_event_ids(chunks: list[str]) -> list[int]:
    ids: list[int] = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("id: "):
                ids.append(int(line.removeprefix("id: ")))
    return ids


def _repositories_with_state_pruning_due(tmp_path: Path) -> PersistenceRepositories:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(database_path, check_same_thread=False)
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(
        save_id="save-1",
        scenario_id=scenario.id,
        title="Night Watch",
    )
    repositories.set_model_preference(
        task="state_pruning",
        provider="fake",
        model_id="fake-pruner",
    )
    for index in range(5):
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Narrator turn {index}",
            provider="fake",
            model="fake-chat",
        )
    return repositories


def test_spa_fallback_blocks_path_traversal_outside_static(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_package = tmp_path / "site-packages" / "bragi_web"
    fake_app_file = fake_package / "api" / "app.py"
    fake_app_file.parent.mkdir(parents=True)
    fake_app_file.write_text("# fake app module path\n")
    static = fake_package / "static"
    assets = static / "assets"
    assets.mkdir(parents=True)
    (static / "index.html").write_text("INDEX")
    (static / "manifest.webmanifest").write_text('{"name":"Bragi Web"}')
    (static / "app-icon-192.png").write_bytes(b"PNG192")
    (static / "apple-touch-icon.png").write_bytes(b"APPLEPNG")
    (static / "safe.txt").write_text("SAFE")
    (assets / "app.js").write_text("console.log('asset')")
    (assets / "app.js.gz").write_bytes(gzip.compress(b"console.log('asset')"))
    (fake_package / "private.txt").write_text("PRIVATE_PACKAGE_FILE")
    (tmp_path / "site-packages" / "pyproject.toml").write_text("PYPROJECT")

    monkeypatch.setattr(api_app, "__file__", str(fake_app_file))

    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path)))
    ) as client:
        root = client.get("/")
        safe = client.get("/safe.txt")
        asset = client.get("/assets/app.js", headers={"Accept-Encoding": "identity"})
        gzip_asset = client.get(
            "/assets/app.js",
            headers={"Accept-Encoding": "gzip"},
        )
        gzip_disabled_asset = client.get(
            "/assets/app.js",
            headers={"Accept-Encoding": "gzip;q=0, *;q=1"},
        )
        gzip_asset_head = client.head(
            "/assets/app.js",
            headers={"Accept-Encoding": "gzip"},
        )
        deep_link = client.get("/deep/link")
        manifest = client.get("/manifest.webmanifest")
        app_icon = client.get("/app-icon-192.png")
        apple_icon = client.get("/apple-touch-icon.png")
        missing_api = client.get("/api/sync/settings")

        package_private_file = client.get("/%2e%2e/private.txt")
        private_asset = client.get("/assets/%2e%2e/private.txt")
        raw_gzip_sidecar = client.get("/assets/app.js.gz")
        pyproject = client.get("/%2e%2e/%2e%2e/pyproject.toml")

    assert root.text == "INDEX"
    assert root.headers["cache-control"] == "no-cache"
    assert safe.text == "SAFE"
    assert safe.headers["cache-control"] == "public, max-age=86400"
    assert asset.text == "console.log('asset')"
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert asset.headers["x-content-type-options"] == "nosniff"
    assert "content-encoding" not in asset.headers
    assert gzip_asset.text == "console.log('asset')"
    assert gzip_asset.headers["content-encoding"] == "gzip"
    assert gzip_asset.headers["vary"] == "Accept-Encoding"
    assert gzip_asset.headers["cache-control"] == (
        "public, max-age=31536000, immutable"
    )
    assert gzip_asset.headers["content-type"].startswith("text/javascript")
    assert gzip_disabled_asset.text == "console.log('asset')"
    assert "content-encoding" not in gzip_disabled_asset.headers
    assert gzip_asset_head.status_code == 200
    assert gzip_asset_head.content == b""
    assert gzip_asset_head.headers["content-encoding"] == "gzip"
    assert gzip_asset_head.headers["cache-control"] == (
        "public, max-age=31536000, immutable"
    )
    assert deep_link.text == "INDEX"
    assert deep_link.headers["cache-control"] == "no-cache"
    assert manifest.status_code == 200
    assert manifest.text == '{"name":"Bragi Web"}'
    assert manifest.headers["content-type"].startswith("application/manifest+json")
    assert manifest.headers["cache-control"] == "public, max-age=86400"
    assert app_icon.status_code == 200
    assert app_icon.content == b"PNG192"
    assert app_icon.headers["content-type"].startswith("image/png")
    assert app_icon.headers["cache-control"] == "public, max-age=86400"
    assert apple_icon.status_code == 200
    assert apple_icon.content == b"APPLEPNG"
    assert apple_icon.headers["content-type"].startswith("image/png")
    assert apple_icon.headers["cache-control"] == "public, max-age=86400"
    assert missing_api.status_code == 404
    assert missing_api.json() == {"detail": "Not found"}
    assert "INDEX" not in missing_api.text
    assert package_private_file.status_code == 404
    assert "PRIVATE_PACKAGE_FILE" not in package_private_file.text
    assert private_asset.status_code == 404
    assert "PRIVATE_PACKAGE_FILE" not in private_asset.text
    assert raw_gzip_sidecar.status_code == 404
    assert pyproject.status_code == 404
    assert "PYPROJECT" not in pyproject.text


def test_removed_sync_api_routes_return_not_found(tmp_path: Path) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        responses = [
            client.get("/api/sync/settings"),
            client.post("/api/sync/settings", json={"server_url": "http://sync.test"}),
            client.get("/api/sync/health"),
            client.get("/api/sync/remote"),
            client.post("/api/sync/push", json={}),
            client.post("/api/sync/pull-preview/save-1", json={}),
            client.post("/api/sync/restore/preview-1", json={}),
        ]

    assert {response.status_code for response in responses} == {404}
    assert state.jobs.list_active() == []


def test_api_write_guard_rejects_untrusted_route_writes_before_route(
    tmp_path: Path,
) -> None:
    runtime = _RuntimeDouble()
    rejected_cases = [
        ({"Origin": "https://evil.test"}, 403),
        ({_BRAGI_WRITE_HEADER: "1", "Origin": "https://evil.test"}, 403),
        ({"Host": "evil.test", "Origin": "http://evil.test"}, 400),
        ({"Referer": "https://evil.test/page"}, 403),
        ({_BRAGI_WRITE_HEADER: "1", "Referer": "https://evil.test/page"}, 403),
        ({"Origin": "null"}, 403),
        ({"Origin": "not a url"}, 403),
        ({}, 403),
        (
            {
                "Host": "192.168.1.50:8787",
                "Origin": "http://192.168.1.50",
            },
            403,
        ),
        (
            {
                "Host": "192.168.1.50:8787",
                "Origin": "http://192.168.1.51:5173",
            },
            403,
        ),
        (
            {
                "Host": "localhost:not-a-port",
                "Origin": "http://localhost:8787",
            },
            400,
        ),
        ({"Origin": "http://localhost:not-a-port"}, 403),
        (
            {
                "Host": "keeper@localhost:8787",
                "Origin": "http://localhost:8787",
            },
            400,
        ),
        ({"Origin": "http://keeper@localhost:8787"}, 403),
        ({"Host": "[::1", "Origin": "http://[::1]:8787"}, 400),
        ({"Origin": "http://[::1"}, 403),
    ]

    with FastAPITestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        for headers, expected_status in rejected_cases:
            response = client.post("/api/saves/save-1/load", headers=headers)
            assert response.status_code == expected_status
            if expected_status == 403:
                assert response.json()["detail"] == (
                    "Cross-origin state-changing requests are not allowed"
                )

    assert runtime.load_save_calls == []


def test_api_write_guard_allows_trusted_route_writes(tmp_path: Path) -> None:
    runtime = _RuntimeDouble()
    allowed_cases = [
        {_BRAGI_WRITE_HEADER: "1"},
        {"Origin": "http://testserver:8787"},
        {
            "Host": "192.168.1.50:8787",
            "Origin": "http://192.168.1.50:8787",
        },
        {
            "Host": "192.168.1.50:8787",
            "Origin": "http://192.168.1.50:5173",
        },
        {"Host": "[::1]:8787", "Origin": "http://[::1]:8787"},
    ]

    with FastAPITestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        for headers in allowed_cases:
            response = client.post("/api/saves/save-1/load", headers=headers)
            assert response.status_code == 200
            assert response.json()["active_save_id"] == "save-1"

    assert runtime.load_save_calls == []


def test_configured_origin_api_writes_are_allowed(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_ALLOWED_HOSTS", "app.bragi.test")
    monkeypatch.setenv(
        "BRAGI_WEB_ALLOWED_ORIGINS",
        "https://app.bragi.test:9443,http://192.168.1.50",
    )
    runtime = _RuntimeDouble()
    allowed_cases = [
        {
            "Host": "app.bragi.test:9443",
            "Origin": "https://app.bragi.test:9443",
        },
        {
            "Host": "192.168.1.50:8787",
            "Origin": "http://192.168.1.50",
        },
    ]

    with FastAPITestClient(
        create_app(cast(WebAppState, _state_double(tmp_path, runtime)))
    ) as client:
        for headers in allowed_cases:
            response = client.post("/api/saves/save-1/load", headers=headers)
            assert response.status_code == 200
            assert response.json()["active_save_id"] == "save-1"

    assert runtime.load_save_calls == []


def test_write_guard_rejects_multipart_upload_without_bragi_header(
    tmp_path: Path,
) -> None:
    class UploadRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.preview_called = False

        def preview_import_bundle(self, bundle_path: Path) -> NoReturn:
            self.preview_called = True
            raise AssertionError("preview should not run for rejected upload")

    runtime = UploadRuntime()
    state = _state_double(tmp_path, runtime)

    with FastAPITestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/bundles/preview",
            files={"file": ("save.bragi-chat", b"bundle", "application/octet-stream")},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "Cross-origin state-changing requests are not allowed"
    )
    assert runtime.preview_called is False
    assert list(state.paths.temp_dir.iterdir()) == []


def test_write_guard_rejects_evil_origin_for_upload_routes(
    tmp_path: Path,
) -> None:
    class UploadRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.preview_called = False
            self.reference_called = False

        def preview_import_bundle(self, bundle_path: Path) -> NoReturn:
            self.preview_called = True
            raise AssertionError("preview should not run for rejected upload")

        def upload_character_reference_image(
            self,
            *,
            image_bytes: bytes,
            filename: str | None,
            replace_existing: bool,
            active_save_id: str | None,
        ) -> NoReturn:
            self.reference_called = True
            raise AssertionError("media upload should not run for rejected upload")

    runtime = UploadRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        bundle = client.post(
            "/api/bundles/preview",
            headers={"Origin": "https://evil.test"},
            files={"file": ("save.bragi-chat", b"bundle", "application/octet-stream")},
        )
        reference = client.post(
            "/api/media/character-reference/upload",
            headers={"Origin": "https://evil.test"},
            files={"file": ("portrait.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        )

    assert bundle.status_code == 403
    assert reference.status_code == 403
    assert runtime.preview_called is False
    assert runtime.reference_called is False
    assert list(state.paths.temp_dir.iterdir()) == []


def test_cross_origin_api_reads_are_allowed(tmp_path: Path) -> None:
    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path)))
    ) as client:
        response = client.get(
            "/api/saves",
            headers={"Origin": "https://evil.test"},
        )

    assert response.status_code == 200
    assert response.json() == {"saves": []}


def test_forged_host_api_reads_are_rejected(tmp_path: Path) -> None:
    with TestClient(
        create_app(cast(WebAppState, _state_double(tmp_path)))
    ) as client:
        response = client.get("/api/saves", headers={"Host": "evil.test"})

    assert response.status_code == 400


def test_bundle_api_round_trip_preserves_runtime_graph_and_media(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    app = create_app()
    state = cast(WebAppState, app.state.bragi)

    with TestClient(app) as client:
        original_save_id = _create_manual_lantern_save(client)
        _submit_chat_and_wait(
            client,
            save_id=original_save_id,
            body="I check the flame.",
        )
        source_message_id = _latest_narrator_message_id(state, original_save_id)
        with state.lock:
            state.repositories.upsert_scene_snapshot(
                save_id=original_save_id,
                in_world_time="Friday evening",
                time_of_day="evening",
                day_of_week="friday",
                world_day_index=5,
                world_time_day_index=5,
                world_time_day_label="friday",
                world_time_phase="evening",
                world_time_source_message_id=source_message_id,
                source_message_id=source_message_id,
            )
            state.repositories.connection.execute(
                """
                UPDATE scene_snapshots
                SET in_world_time = 'Friday evening after the bell'
                WHERE save_id = ?
                """,
                (original_save_id,),
            )
            state.repositories.connection.commit()
        original_asset, image_path, thumbnail_path = _seed_portability_media_asset(
            state,
            save_id=original_save_id,
            source_message_id=source_message_id,
        )

        exported = client.get("/api/bundles/export")
        assert exported.status_code == 200
        assert exported.content
        previewed = client.post(
            "/api/bundles/preview",
            files={
                "file": (
                    "lantern-keep.bragi-chat",
                    exported.content,
                    "application/octet-stream",
                )
            },
        )
        assert previewed.status_code == 200
        preview_payload = previewed.json()
        assert preview_payload["preview"]["title"] == "Lantern Keep"
        assert preview_payload["preview"]["message_count"] == 3
        assert preview_payload["preview"]["media_count"] == 1

        imported = client.post(
            f"/api/bundles/import/{preview_payload['preview_id']}",
            json={},
        )
        assert imported.status_code == 200
        imported_save_id = imported.json()["active_save_id"]
        assert imported_save_id != original_save_id

        with state.lock:
            imported_messages = state.repositories.list_messages(imported_save_id)
            imported_assets = state.repositories.list_media_assets(imported_save_id)
            imported_snapshot = state.repositories.get_scene_snapshot(
                imported_save_id
            )

        assert [message.body for message in imported_messages] == [
            "The beacon snaps awake.",
            "I check the flame.",
            "echo: I check the flame.",
        ]
        assert imported_snapshot is not None
        assert imported_snapshot.in_world_time == "Friday evening after the bell"
        assert imported_snapshot.world_time_day_index == 5
        assert imported_snapshot.world_time_day_label == "friday"
        assert imported_snapshot.world_time_phase == "evening"
        assert imported_snapshot.world_time_source_message_id is not None
        assert imported_snapshot.world_time_source_message_id != source_message_id
        assert len(imported_assets) == 1
        imported_asset = imported_assets[0]
        assert imported_asset.id != original_asset.id
        assert imported_asset.save_id == imported_save_id
        assert imported_asset.source_message_id is not None
        assert imported_asset.source_message_id != source_message_id
        assert Path(imported_asset.path).parts[0] == imported_save_id
        assert imported_asset.thumbnail_path is not None
        assert Path(imported_asset.thumbnail_path).parts[0] == imported_save_id
        assert (state.paths.media_dir / imported_asset.path).read_bytes() == (
            image_path.read_bytes()
        )
        assert (state.paths.media_dir / imported_asset.thumbnail_path).read_bytes() == (
            thumbnail_path.read_bytes()
        )

        _submit_chat_and_wait(
            client,
            save_id=imported_save_id,
            body="I answer the bell.",
        )
        runtime = client.get("/api/runtime")

    assert runtime.status_code == 200
    runtime_payload = runtime.json()
    assert runtime_payload["active_save_id"] == imported_save_id
    assert runtime_payload["world_time"] == {
        "snapshot_id": imported_snapshot.id,
        "day_index": 5,
        "day_label": "friday",
        "phase": "evening",
        "clock_minutes": None,
        "period_label": "",
        "source_message_id": imported_snapshot.world_time_source_message_id,
        "confidence": None,
        "display": "Friday evening; world day index 5",
    }
    assert [
        message["body"]
        for message in runtime_payload["chronicle"]["messages"][-2:]
    ] == [
        "I answer the bell.",
        "echo: I answer the bell.",
    ]


def test_delete_save_api_removes_real_media_files(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    app = create_app()
    state = cast(WebAppState, app.state.bragi)

    with TestClient(app) as client:
        save_id = _create_manual_lantern_save(client)
        _asset, image_path, thumbnail_path = _seed_portability_media_asset(
            state,
            save_id=save_id,
            source_message_id=None,
        )

        deleted = client.delete(f"/api/saves/{save_id}")

    assert deleted.status_code == 200
    assert deleted.json()["active_save_id"] is None
    with state.lock:
        assert state.repositories.get_save(save_id) is None
        assert state.repositories.list_all_media_assets(save_id) == []
    assert not image_path.exists()
    assert not thumbnail_path.exists()


def test_delete_save_api_deletes_save_and_events_when_media_cleanup_fails(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAGI_WEB_FAKE_PROVIDERS", "1")
    app = create_app()
    state = cast(WebAppState, app.state.bragi)

    with TestClient(app) as client:
        save_id = _create_manual_lantern_save(client)
        _asset, image_path, thumbnail_path = _seed_portability_media_asset(
            state,
            save_id=save_id,
            source_message_id=None,
        )
        latest_event_id = state.save_events.latest_event_id()
        real_unlink = Path.unlink

        def fail_thumbnail(self: Path, missing_ok: bool = False) -> None:
            if self == thumbnail_path:
                raise OSError("simulated unlink failure")
            real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", fail_thumbnail)

        deleted = client.delete(f"/api/saves/{save_id}")

    assert deleted.status_code == 200
    deleted_payload = deleted.json()
    assert not deleted_payload.get("error")
    assert deleted_payload["active_save_id"] is None
    with state.lock:
        assert state.repositories.get_save(save_id) is None
        assert state.repositories.list_all_media_assets(save_id) == []
    assert not image_path.exists()
    assert thumbnail_path.exists()
    assert state.save_events.latest_event_id() > latest_event_id


def test_bundle_preview_requires_confirming_import_and_cleans_temp_file(
    tmp_path: Path,
) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        previewed = client.post(
            "/api/bundles/preview",
            files={"file": ("save.bragi-chat", b"bundle", "application/octet-stream")},
        )
        assert previewed.status_code == 200
        preview_id = previewed.json()["preview_id"]
        assert previewed.json()["preview"]["message_count"] == 2
        assert preview_id in state.bundle_previews
        staged_path = state.bundle_previews[preview_id].bundle_path

        imported = client.post(f"/api/bundles/import/{preview_id}", json={})

    assert imported.status_code == 200
    assert imported.json()["active_save_title"] == "Lantern Keep"
    assert not staged_path.exists()
    assert state.bundle_previews == {}


def test_scenario_bundle_preview_import_and_export_cleanup(tmp_path: Path) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        previewed = client.post(
            "/api/scenario-bundles/preview",
            files={
                "file": (
                    "scenario.bragi-scenario",
                    b"scenario-bundle",
                    "application/octet-stream",
                )
            },
        )
        assert previewed.status_code == 200
        preview_id = previewed.json()["preview_id"]
        assert previewed.json()["preview"]["scenario_type"] == "full_roleplay"
        assert preview_id in state.scenario_bundle_previews
        staged_path = state.scenario_bundle_previews[preview_id].bundle_path
        assert staged_path.suffix == ".bragi-scenario"

        imported = client.post(f"/api/scenario-bundles/import/{preview_id}", json={})
        exported = client.get("/api/scenario-bundles/export/scenario-1")

    assert imported.status_code == 200
    assert imported.json()["status"] == "Imported scenario: Lantern Keep"
    assert not staged_path.exists()
    assert state.scenario_bundle_previews == {}
    assert exported.status_code == 200
    assert exported.content == b"scenario-bundle"
    assert not list(state.paths.temp_dir.glob("*.bragi-scenario"))


def test_character_bundle_preview_import_and_export_cleanup(tmp_path: Path) -> None:
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        previewed = client.post(
            "/api/character-bundles/preview",
            data={"active_save_id": "save-1"},
            files={
                "file": (
                    "character.bragi-character",
                    b"character-bundle",
                    "application/octet-stream",
                )
            },
        )
        assert previewed.status_code == 200
        preview_id = previewed.json()["preview_id"]
        assert previewed.json()["preview"]["name"] == "Mara"
        assert previewed.json()["preview"]["role"] == "Signal runner"
        assert preview_id in state.character_bundle_previews
        staged_path = state.character_bundle_previews[preview_id].bundle_path
        assert staged_path.suffix == ".bragi-character"

        imported = client.post(
            f"/api/character-bundles/import/{preview_id}",
            json={"active_save_id": "save-1", "name": "Mara"},
        )
        exported = client.get("/api/character-bundles/export/character-1")
        default_include_private_notes = (
            state.runtime.character_export_include_private_notes
        )
        exported_with_private_notes = client.get(
            "/api/character-bundles/export/character-1?include_private_notes=true"
        )
        explicit_include_private_notes = (
            state.runtime.character_export_include_private_notes
        )

    assert imported.status_code == 200
    assert imported.json()["characters"][0]["character_id"] == "character-imported"
    assert not staged_path.exists()
    assert state.character_bundle_previews == {}
    assert exported.status_code == 200
    assert exported.content == b"character-bundle"
    assert default_include_private_notes is False
    assert exported_with_private_notes.status_code == 200
    assert exported_with_private_notes.content == b"character-bundle"
    assert explicit_include_private_notes is True
    assert not list(state.paths.temp_dir.glob("*.bragi-character"))


def test_bundle_previews_are_scoped_to_creating_user(tmp_path: Path) -> None:
    state = _auth_state(tmp_path)
    state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    app = create_app(cast(WebAppState, state))

    with (
        TestClient(app, authenticate=False) as mira_client,
        TestClient(app, authenticate=False) as rook_client,
    ):
        assert mira_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        assert rook_client.post(
            "/api/auth/login",
            json={"username": "Rook", "password": "correct horse"},
        ).status_code == 200

        save_previewed = mira_client.post(
            "/api/bundles/preview",
            files={"file": ("save.bragi-chat", b"bundle", "application/octet-stream")},
        )
        assert save_previewed.status_code == 200
        save_preview_id = save_previewed.json()["preview_id"]
        save_staged_path = state.bundle_previews[save_preview_id].bundle_path

        blocked_save_import = rook_client.post(
            f"/api/bundles/import/{save_preview_id}",
            json={},
        )
        allowed_save_import = mira_client.post(
            f"/api/bundles/import/{save_preview_id}",
            json={},
        )

        scenario_previewed = mira_client.post(
            "/api/scenario-bundles/preview",
            files={
                "file": (
                    "scenario.bragi-scenario",
                    b"scenario-bundle",
                    "application/octet-stream",
                )
            },
        )
        assert scenario_previewed.status_code == 200
        scenario_preview_id = scenario_previewed.json()["preview_id"]
        scenario_staged_path = (
            state.scenario_bundle_previews[scenario_preview_id].bundle_path
        )

        blocked_scenario_import = rook_client.post(
            f"/api/scenario-bundles/import/{scenario_preview_id}",
            json={},
        )
        allowed_scenario_import = mira_client.post(
            f"/api/scenario-bundles/import/{scenario_preview_id}",
            json={},
        )

    assert blocked_save_import.status_code == 404
    assert blocked_save_import.json()["detail"] == "Unknown bundle preview"
    assert allowed_save_import.status_code == 200
    assert not save_staged_path.exists()
    assert state.bundle_previews == {}

    assert blocked_scenario_import.status_code == 404
    assert blocked_scenario_import.json()["detail"] == (
        "Unknown scenario bundle preview"
    )
    assert allowed_scenario_import.status_code == 200
    assert not scenario_staged_path.exists()
    assert state.scenario_bundle_previews == {}


def test_character_bundle_previews_are_scoped_to_user_and_target_save(
    tmp_path: Path,
) -> None:
    class CharacterRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.preview_targets: list[str | None] = []
            self.import_targets: list[str | None] = []

        def preview_import_character_bundle(
            self,
            bundle_path: Path,
            *,
            target_save_id: str | None = None,
        ) -> _CharacterPreview:
            assert bundle_path.is_file()
            self.preview_targets.append(target_save_id)
            return _CharacterPreview()

        def import_character_bundle(
            self,
            bundle_path: Path,
            *,
            target_save_id: str | None = None,
            name: str | None = None,
        ) -> dict[str, object]:
            assert bundle_path.is_file()
            assert name in {None, "Mara"}
            self.import_targets.append(target_save_id)
            return {
                "active_save_id": target_save_id,
                "characters": [
                    {"character_id": "character-imported", "name": "Mara"}
                ],
            }

    runtime = CharacterRuntime()
    state = _auth_state(tmp_path, runtime)
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    rook = state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    mira_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    mira_other_save = _create_auth_save(
        state.repositories,
        title="Mira Other Save",
        owner_user_id=mira.id,
    )
    rook_save = _create_auth_save(
        state.repositories,
        title="Rook Save",
        owner_user_id=rook.id,
    )
    app = create_app(cast(WebAppState, state))

    with (
        TestClient(app, authenticate=False) as mira_client,
        TestClient(app, authenticate=False) as rook_client,
    ):
        assert mira_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        assert rook_client.post(
            "/api/auth/login",
            json={"username": "Rook", "password": "correct horse"},
        ).status_code == 200

        inaccessible_preview = rook_client.post(
            "/api/character-bundles/preview",
            data={"active_save_id": mira_save.id},
            files={
                "file": (
                    "character.bragi-character",
                    b"character-bundle",
                    "application/octet-stream",
                )
            },
        )

        previewed = mira_client.post(
            "/api/character-bundles/preview",
            data={"active_save_id": mira_save.id},
            files={
                "file": (
                    "character.bragi-character",
                    b"character-bundle",
                    "application/octet-stream",
                )
            },
        )
        assert previewed.status_code == 200
        preview_id = previewed.json()["preview_id"]
        staged_path = state.character_bundle_previews[preview_id].bundle_path

        blocked_import = rook_client.post(
            f"/api/character-bundles/import/{preview_id}",
            json={"active_save_id": rook_save.id, "name": "Mara"},
        )
        mismatched_import = mira_client.post(
            f"/api/character-bundles/import/{preview_id}",
            json={"active_save_id": mira_other_save.id, "name": "Mara"},
        )
        allowed_import = mira_client.post(
            f"/api/character-bundles/import/{preview_id}",
            json={"active_save_id": mira_save.id, "name": "Mara"},
        )

    assert inaccessible_preview.status_code == 404
    assert len(runtime.preview_targets) == 1
    assert runtime.preview_targets == [mira_save.id]
    assert blocked_import.status_code == 404
    assert blocked_import.json()["detail"] == "Unknown character bundle preview"
    assert mismatched_import.status_code == 400
    assert mismatched_import.json()["detail"] == (
        "Character bundle preview target save does not match request"
    )
    assert allowed_import.status_code == 200
    assert allowed_import.json()["active_save_id"] == mira_save.id
    assert runtime.import_targets == [mira_save.id]
    assert not staged_path.exists()
    assert state.character_bundle_previews == {}


def test_character_bundle_preview_rejects_missing_target_before_store(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_store_upload(file: object, temp_dir: Path) -> Path:
        raise AssertionError("upload should not be stored before target validation")

    monkeypatch.setattr(api_app, "_store_upload", fail_store_upload)
    state = _state_double(tmp_path)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/character-bundles/preview",
            files={
                "file": (
                    "character.bragi-character",
                    b"character-bundle",
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == _SAVE_ID_REQUIRED_DETAIL
    assert state.character_bundle_previews == {}
    assert list(state.paths.temp_dir.iterdir()) == []


def test_character_bundle_preview_rejects_unknown_or_inaccessible_target_before_store(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    upload_attempts: list[str | None] = []

    async def fail_store_upload(file: object, temp_dir: Path) -> Path:
        upload_attempts.append(getattr(file, "filename", None))
        raise AssertionError("upload should not be stored before target validation")

    monkeypatch.setattr(api_app, "_store_upload", fail_store_upload)
    state = _auth_state(tmp_path)
    mira = state.auth_service().create_user(
        username="Mira",
        password="correct horse",
        role="user",
    )
    state.auth_service().create_user(
        username="Rook",
        password="correct horse",
        role="user",
    )
    mira_save = _create_auth_save(
        state.repositories,
        title="Mira Save",
        owner_user_id=mira.id,
    )
    app = create_app(cast(WebAppState, state))

    with (
        TestClient(app, authenticate=False) as mira_client,
        TestClient(app, authenticate=False) as rook_client,
    ):
        assert mira_client.post(
            "/api/auth/login",
            json={"username": "Mira", "password": "correct horse"},
        ).status_code == 200
        assert rook_client.post(
            "/api/auth/login",
            json={"username": "Rook", "password": "correct horse"},
        ).status_code == 200

        unknown_target = mira_client.post(
            "/api/character-bundles/preview",
            data={"active_save_id": "missing-save"},
            files={
                "file": (
                    "character.bragi-character",
                    b"character-bundle",
                    "application/octet-stream",
                )
            },
        )
        inaccessible_target = rook_client.post(
            "/api/character-bundles/preview",
            data={"active_save_id": mira_save.id},
            files={
                "file": (
                    "character.bragi-character",
                    b"character-bundle",
                    "application/octet-stream",
                )
            },
        )

    assert unknown_target.status_code == 404
    assert inaccessible_target.status_code == 404
    assert upload_attempts == []
    assert state.character_bundle_previews == {}
    assert list(state.paths.temp_dir.iterdir()) == []


def test_bundle_preview_invalid_upload_returns_400_and_cleans_temp_file(
    tmp_path: Path,
) -> None:
    class InvalidPreviewRuntime(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.preview_path: Path | None = None

        def preview_import_bundle(self, bundle_path: Path) -> NoReturn:
            self.preview_path = bundle_path
            assert bundle_path.is_file()
            raise ValueError("Not a Bragi bundle")

    runtime = InvalidPreviewRuntime()
    state = _state_double(tmp_path, runtime)

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/bundles/preview",
            files={"file": ("bad.bragi-chat", b"bad", "application/octet-stream")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Not a Bragi bundle"
    assert runtime.preview_path is not None
    assert not runtime.preview_path.exists()
    assert state.bundle_previews == {}
    assert list(state.paths.temp_dir.iterdir()) == []


def test_bundle_preview_rejects_oversized_upload_and_cleans_partial_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(api_app, "BUNDLE_UPLOAD_MAX_BYTES", 5)

    class NoPreviewRuntime(_RuntimeDouble):
        def preview_import_bundle(self, bundle_path: Path) -> NoReturn:
            raise AssertionError("preview should not run for oversized uploads")

    state = _state_double(tmp_path, NoPreviewRuntime())

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/bundles/preview",
            files={
                "file": (
                    "large.bragi-chat",
                    b"abcdef",
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Bundle upload exceeds 5 bytes"
    assert state.bundle_previews == {}
    assert list(state.paths.temp_dir.iterdir()) == []


def test_bundle_upload_defaults_support_large_save_exports() -> None:
    assert api_app.BUNDLE_UPLOAD_MAX_BYTES == 2 * 1024 * 1024 * 1024
    assert api_app.BUNDLE_PREVIEW_MAX_RETAINED_BYTES == 2 * 1024 * 1024 * 1024


def test_bundle_preview_prunes_expired_and_excess_previews(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(api_app, "BUNDLE_PREVIEW_TTL_SECONDS", 10.0)
    monkeypatch.setattr(api_app, "BUNDLE_PREVIEW_MAX_COUNT", 2)
    state = _state_double(tmp_path)
    now = time.time()
    expired_path = state.paths.temp_dir / "expired.bragi-chat"
    oldest_path = state.paths.temp_dir / "oldest.bragi-chat"
    kept_path = state.paths.temp_dir / "kept.bragi-chat"
    for path in (expired_path, oldest_path, kept_path):
        path.write_bytes(b"bundle")
    state.bundle_previews.update(
        {
            "expired": BundlePreviewState(
                bundle_path=expired_path,
                created_at=now - 20,
            ),
            "oldest": BundlePreviewState(
                bundle_path=oldest_path,
                created_at=now - 2,
            ),
            "kept": BundlePreviewState(bundle_path=kept_path, created_at=now - 1),
        }
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/bundles/preview",
            files={"file": ("save.bragi-chat", b"bundle", "application/octet-stream")},
        )

    assert response.status_code == 200
    new_preview_id = response.json()["preview_id"]
    assert set(state.bundle_previews) == {"kept", new_preview_id}
    assert not expired_path.exists()
    assert not oldest_path.exists()
    assert kept_path.exists()
    assert state.bundle_previews[new_preview_id].bundle_path.exists()


def test_bundle_preview_prunes_oldest_when_retained_bytes_exceed_cap(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(api_app, "BUNDLE_PREVIEW_MAX_RETAINED_BYTES", 20)
    state = _state_double(tmp_path)
    now = time.time()
    old_chat_path = state.paths.temp_dir / "old-chat.bragi-chat"
    old_scenario_path = state.paths.temp_dir / "old-scenario.bragi-scenario"
    old_chat_path.write_bytes(b"chat-keep")
    old_scenario_path.write_bytes(b"scenario!")
    state.bundle_previews["old-chat"] = BundlePreviewState(
        bundle_path=old_chat_path,
        created_at=now - 3,
    )
    state.scenario_bundle_previews["old-scenario"] = BundlePreviewState(
        bundle_path=old_scenario_path,
        created_at=now - 2,
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post(
            "/api/bundles/preview",
            files={"file": ("save.bragi-chat", b"bundle", "application/octet-stream")},
        )

    assert response.status_code == 200
    new_preview_id = response.json()["preview_id"]
    assert set(state.bundle_previews) == {new_preview_id}
    assert set(state.scenario_bundle_previews) == {"old-scenario"}
    assert not old_chat_path.exists()
    assert old_scenario_path.exists()
    assert state.bundle_previews[new_preview_id].bundle_path.exists()


def test_bundle_import_runtime_error_returns_400_and_cleans_preview(
    tmp_path: Path,
) -> None:
    class ImportErrorRuntime(_RuntimeDouble):
        def import_save_bundle(self, bundle_path: Path) -> dict[str, object]:
            assert bundle_path.is_file()
            return {"error": "Bundle import failed"}

    state = _state_double(tmp_path, ImportErrorRuntime())
    bundle_path = state.paths.temp_dir / "preview-error.bragi-chat"
    bundle_path.write_bytes(b"bundle")
    state.bundle_previews["preview-error"] = BundlePreviewState(
        bundle_path=bundle_path,
        created_at=time.time(),
    )

    with TestClient(create_app(cast(WebAppState, state))) as client:
        response = client.post("/api/bundles/import/preview-error", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == "Bundle import failed"
    assert not bundle_path.exists()
    assert state.bundle_previews == {}


def test_bundle_export_deletes_temp_file_after_success_error_and_exception(
    tmp_path: Path,
) -> None:
    class ExportRuntimeBase(_RuntimeDouble):
        def __init__(self) -> None:
            super().__init__()
            self.export_path: Path | None = None

    class ExportRuntime(ExportRuntimeBase):

        def export_active_save(self, bundle_path: Path) -> SimpleNamespace:
            self.export_path = bundle_path
            bundle_path.write_bytes(b"bundle")
            return SimpleNamespace(error=None)

    class ExportErrorRuntime(ExportRuntimeBase):
        def export_active_save(self, bundle_path: Path) -> SimpleNamespace:
            self.export_path = bundle_path
            bundle_path.write_bytes(b"partial")
            return SimpleNamespace(error="No active save")

    class ExplodingExportRuntime(ExportRuntimeBase):
        def export_active_save(self, bundle_path: Path) -> NoReturn:
            self.export_path = bundle_path
            bundle_path.write_bytes(b"partial")
            raise RuntimeError("export crashed")

    cases: list[tuple[ExportRuntimeBase, bool, int, bytes | None, str | None]] = [
        (ExportRuntime(), True, 200, b"bundle", None),
        (ExportErrorRuntime(), True, 400, None, "No active save"),
        (ExplodingExportRuntime(), False, 500, None, None),
    ]
    for index, (runtime, raise_server_exceptions, status_code, content, detail) in (
        enumerate(cases)
    ):
        case_path = tmp_path / f"case-{index}"
        case_path.mkdir()
        with TestClient(
            create_app(cast(WebAppState, _state_double(case_path, runtime))),
            raise_server_exceptions=raise_server_exceptions,
        ) as client:
            response = client.get("/api/bundles/export")

        assert response.status_code == status_code
        if content is not None:
            assert response.content == content
        if detail is not None:
            assert response.json()["detail"] == detail
        assert runtime.export_path is not None
        assert not runtime.export_path.exists()
