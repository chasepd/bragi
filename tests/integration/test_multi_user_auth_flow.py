from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient as FastAPITestClient

from bragi.persistence import migrate_database
from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.auth_service import AuthService
from bragi_web.api.app import create_app
from bragi_web.jobs import JobRegistry
from bragi_web.runtime import RuntimeAccessLock, SaveEventHub, WebAppState

_BRAGI_WRITE_HEADER = "X-Bragi-Api-Request"


class TestClient(FastAPITestClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        headers = {_BRAGI_WRITE_HEADER: "1"}
        headers.update(kwargs.pop("headers", {}))
        kwargs["headers"] = headers
        super().__init__(*args, **kwargs)


class AuthFlowRuntime:
    def __init__(self) -> None:
        self.active_save_id: str | None = None
        self.submitted: list[tuple[str | None, str]] = []

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
        return _runtime_model(
            save_id,
            body=f"Runtime for {save_id or 'none'}",
            status=status,
        )

    async def submit_player_message_for_initial_render(
        self,
        body: str,
        *,
        speaker_name: str | None = None,
        after_message_id: str | None = None,
        active_save_id: str | None | object = ...,
    ) -> SimpleNamespace:
        del speaker_name, after_message_id
        save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        self.submitted.append((save_id, body))
        return SimpleNamespace(
            model=_runtime_model(save_id, body=f"Narrator answers: {body}"),
            has_post_turn_jobs=False,
            save_id=save_id,
            player_message_id="player-1",
            narrator_message_id="narrator-1",
        )


def test_multi_user_auth_flow_scopes_saves_and_child_access(
    tmp_path: Path,
) -> None:
    runtime = AuthFlowRuntime()
    state = _auth_state(tmp_path, runtime)
    service = state.auth_service()
    service.create_user(
        username="admin",
        password="correct horse",
        role="admin",
    )
    mira = service.create_user(username="mira", password="correct horse", role="user")
    rook = service.create_user(username="rook", password="correct horse", role="user")
    child = service.create_user(
        username="ilyra",
        password="correct horse",
        role="child",
    )
    mira_save = _create_save(
        state.repositories,
        title="Mira Lantern",
        owner_user_id=mira.id,
    )
    rook_save = _create_save(
        state.repositories,
        title="Rook Tower",
        owner_user_id=rook.id,
    )
    child_save = _create_save(
        state.repositories,
        title="Ilyra Camp",
        owner_user_id=child.id,
    )
    state.repositories.grant_save_access(save_id=mira_save.id, user_id=child.id)
    state.repositories.grant_save_access(save_id=mira_save.id, user_id=rook.id)

    app = create_app(cast(WebAppState, state))
    with (
        TestClient(app) as admin_client,
        TestClient(app) as mira_client,
        TestClient(app) as rook_client,
        TestClient(app) as child_client,
    ):
        _login(admin_client, "admin")
        _login(mira_client, "mira")
        _login(rook_client, "rook")
        _login(child_client, "ilyra")

        admin_saves = admin_client.get("/api/saves")
        admin_loaded_rook = admin_client.post(f"/api/saves/{rook_save.id}/load")
        mira_saves = mira_client.get("/api/saves")
        rook_saves = rook_client.get("/api/saves")
        child_saves = child_client.get("/api/saves")
        mira_denied = mira_client.get(f"/api/runtime?save_id={rook_save.id}")
        rook_assigned_runtime = rook_client.get(
            f"/api/runtime?save_id={mira_save.id}"
        )
        rook_denied = rook_client.get(f"/api/runtime?save_id={child_save.id}")
        rook_delete_assigned = rook_client.delete(f"/api/saves/{mira_save.id}")
        child_assigned_runtime = child_client.get(
            f"/api/runtime?save_id={mira_save.id}"
        )
        child_owned_runtime = child_client.get(
            f"/api/runtime?save_id={child_save.id}"
        )
        child_denied = child_client.get(f"/api/runtime?save_id={rook_save.id}")
        child_chat = child_client.post(
            "/api/chat",
            json={"save_id": mira_save.id, "body": "I tend the lamp."},
        )
        chat_job = _wait_for_terminal_job(
            child_client,
            child_chat.json()["id"],
            save_id=mira_save.id,
        )

    assert admin_saves.status_code == 200
    assert set(_save_ids(admin_saves.json())) == {
        mira_save.id,
        rook_save.id,
        child_save.id,
    }
    assert admin_loaded_rook.status_code == 200
    assert admin_loaded_rook.json()["active_save_id"] == rook_save.id
    assert mira_saves.status_code == 200
    assert _save_ids(mira_saves.json()) == [mira_save.id]
    assert rook_saves.status_code == 200
    assert set(_save_ids(rook_saves.json())) == {mira_save.id, rook_save.id}
    assert child_saves.status_code == 200
    assert set(_save_ids(child_saves.json())) == {mira_save.id, child_save.id}
    assert mira_denied.status_code == 404
    assert rook_assigned_runtime.status_code == 200
    assert rook_assigned_runtime.json()["active_save_id"] == mira_save.id
    assert rook_denied.status_code == 404
    assert rook_delete_assigned.status_code == 403
    assert child_assigned_runtime.status_code == 200
    assert child_assigned_runtime.json()["active_save_id"] == mira_save.id
    assert child_owned_runtime.status_code == 200
    assert child_owned_runtime.json()["active_save_id"] == child_save.id
    assert child_denied.status_code == 404
    assert child_chat.status_code == 200
    assert chat_job["status"] == "succeeded"
    assert chat_job["result"]["active_save_id"] == mira_save.id
    assert runtime.submitted == [(mira_save.id, "I tend the lamp.")]


def _auth_state(tmp_path: Path, runtime: AuthFlowRuntime) -> SimpleNamespace:
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(
        sqlite3.connect(database_path, check_same_thread=False)
    )
    return SimpleNamespace(
        auth_required=True,
        runtime=runtime,
        lock=RuntimeAccessLock(),
        jobs=JobRegistry(),
        save_events=SaveEventHub(),
        providers={},
        repositories=repositories,
        settings_service=lambda: SimpleNamespace(secret_storage_warning=lambda: None),
        auth_service=lambda: AuthService(repositories=repositories),
        secret_store=SimpleNamespace(),
        paths=SimpleNamespace(temp_dir=temp_dir, media_dir=media_dir),
        bundle_previews={},
        scenario_bundle_previews={},
        character_bundle_previews={},
        log_file_path=None,
    )


def _create_save(
    repositories: PersistenceRepositories,
    *,
    title: str,
    owner_user_id: str,
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


def _login(client: TestClient, username: str) -> None:
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": "correct horse"},
    )
    assert response.status_code == 200


def _wait_for_terminal_job(
    client: TestClient,
    job_id: str,
    *,
    save_id: str,
) -> dict[str, Any]:
    job: dict[str, Any] = {}
    for _ in range(100):
        job = cast(dict[str, Any], client.get(_job_url(job_id, save_id)).json())
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.025)
    return job


def _job_url(job_id: str, save_id: str) -> str:
    return f"/api/jobs/{job_id}?save_id={save_id}"


def _save_ids(payload: dict[str, object]) -> list[str]:
    saves = cast(list[dict[str, object]], payload["saves"])
    return [cast(str, save["save_id"]) for save in saves]


def _runtime_model(
    save_id: str | None,
    *,
    body: str,
    status: str | None = None,
) -> dict[str, object]:
    return {
        "active_save_id": save_id,
        "active_save_title": "Lantern Keep" if save_id else None,
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
        "status": status,
    }
