from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ProviderClient,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.retry import call_with_provider_retries
from bragi_web.api.app import create_app
from bragi_web.runtime import create_state


@dataclass
class _VirtualLatencyClock:
    seconds: float = 0.0

    def now(self) -> float:
        return self.seconds

    def advance_ms(self, milliseconds: int) -> None:
        self.seconds += milliseconds / 1000

    @property
    def elapsed_ms(self) -> int:
        return round(self.seconds * 1000)


class _DelayedRetryProvider:
    provider_name = "fake"

    def __init__(
        self,
        clock: _VirtualLatencyClock,
        attempt_delays_ms: tuple[int, ...],
    ) -> None:
        self.clock = clock
        self.attempt_delays_ms = attempt_delays_ms
        self.chat_attempt_count = 0

    async def chat(self, request: ChatRequest) -> ChatResponse:
        async def request_transport() -> dict[str, object]:
            wave_index = self.chat_attempt_count
            self.chat_attempt_count += 1
            await asyncio.sleep(0)
            self.clock.advance_ms(self.attempt_delays_ms[wave_index])
            if wave_index == 0:
                raise ProviderError(
                    ProviderErrorCategory.NETWORK_ERROR,
                    "scripted transient failure",
                )
            return {"transport_status": 200}

        raw_metadata = await call_with_provider_retries(
            request_transport,
            provider=request.provider,
            task="chat",
            max_attempts=2,
            base_delay=0,
        )
        return ChatResponse(
            body="The bell answers.",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 12},
            raw_metadata=raw_metadata,
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        await asyncio.sleep(0)
        if request.schema_name == "content_safety_review":
            data: dict[str, Any] = {
                "action": "allow",
                "category": "none",
                "reason": "Integration fixture content is within the ceiling.",
                "minimum_rating": "g",
            }
        else:
            data = {"selections": []}
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 3},
        )


def test_actual_chat_job_orders_critical_spans_and_counts_retry_waves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _VirtualLatencyClock()
    provider = _DelayedRetryProvider(clock, (120, 80))
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", str(tmp_path / "web-data"))
    monkeypatch.setattr(
        "bragi_web.runtime._provider_clients",
        lambda *_args, **_kwargs: {"fake": cast(ProviderClient, provider)},
    )
    monkeypatch.setattr("bragi.providers.retry.perf_counter", clock.now)
    monkeypatch.setattr("bragi.services.runtime_telemetry.perf_counter", clock.now)
    monkeypatch.setattr("bragi_web.api.app.perf_counter", clock.now)

    state = create_state()
    state.auth_required = False
    try:
        scenario = state.repositories.create_scenario(
            type="full_roleplay",
            title="Lantern Keep",
            premise="A beacon is going dark.",
            player_role="Keeper",
            content={"opening_message": "The beacon snaps awake."},
        )
        save = state.repositories.create_save(
            scenario_id=scenario.id,
            title="Lantern Save",
        )
        state.repositories.save_provider_model(
            provider="fake",
            model_id="fake-chat",
            display_name="Fake Chat",
            capabilities=["chat", "structured_output"],
            context_window=32768,
        )
        for task in ("chat", "context_search"):
            state.repositories.set_model_preference(
                task=task,
                provider="fake",
                model_id="fake-chat",
            )
        state.runtime.active_save_id = save.id

        with TestClient(create_app(state)) as client:
            created = client.post(
                "/api/chat",
                headers={"X-Bragi-Api-Request": "1"},
                json={
                    "body": "Light the beacon",
                    "save_id": save.id,
                    "client_turn_id": "11111111-1111-4111-8111-111111111111",
                },
            )
            assert created.status_code == 200, created.text
            job_id = cast(str, created.json()["id"])
            for _ in range(100):
                job = client.get(f"/api/jobs/{job_id}?save_id={save.id}").json()
                if job["status"] in {"succeeded", "failed", "cancelled"}:
                    break
            else:
                raise AssertionError("chat job did not reach a terminal state")

        assert job["status"] == "succeeded", job
        assert provider.chat_attempt_count == 2
        assert clock.elapsed_ms == 200

        steps = state.repositories.list_job_steps(job_id)
        critical_names = {
            "chat.preflight",
            "chat.input_safety",
            "chat.history",
            "chat.context",
            "chat.character_planning",
            "chat.narrator_planning",
            "chat.narrator_generation",
            "chat.output_safety",
            "chat.verification",
            "chat.commit",
            "chat.response_committed",
        }
        critical_steps = [step for step in steps if step.name in critical_names]
        assert {step.name for step in critical_steps} == critical_names
        positions = {
            step.name: index for index, step in enumerate(critical_steps)
        }
        assert positions["chat.preflight"] == 0
        for concurrent_start in ("chat.input_safety", "chat.history"):
            assert positions[concurrent_start] < positions["chat.character_planning"]
            assert positions[concurrent_start] < positions["chat.context"]
        for concurrent_plan in ("chat.character_planning", "chat.context"):
            assert positions[concurrent_plan] < positions["chat.narrator_planning"]
        assert positions["chat.narrator_planning"] < positions[
            "chat.narrator_generation"
        ]
        generation_step = next(
            step for step in critical_steps if step.name == "chat.narrator_generation"
        )
        output_safety_step = next(
            step for step in critical_steps if step.name == "chat.output_safety"
        )
        assert generation_step.started_at is not None
        assert output_safety_step.started_at is not None
        assert generation_step.started_at <= output_safety_step.started_at
        assert positions["chat.output_safety"] < positions[
            "chat.narrator_generation"
        ] < positions["chat.verification"]
        assert positions["chat.verification"] < positions["chat.commit"]
        assert positions["chat.commit"] < positions["chat.response_committed"]
        assert generation_step.duration_ms == 200
        completion_job = next(
            candidate
            for candidate in state.repositories.list_jobs_by_status(("succeeded",))
            if candidate.type == "chat_completion"
        )
        provider_step = next(
            step
            for step in state.repositories.list_job_steps(completion_job.id)
            if step.name == "provider.chat"
        )
        assert provider_step.duration_ms == 200
        assert provider_step.metadata["attempt_count"] == 2
        assert provider_step.metadata["retry_count"] == 1
        assert generation_step.completed_at is not None
        assert provider_step.completed_at is not None
        assert provider_step.completed_at <= generation_step.completed_at
        response_committed_step = next(
            step for step in steps if step.name == "chat.response_committed"
        )
        assert response_committed_step.started_at is not None
        assert generation_step.completed_at <= response_committed_step.started_at
        persisted_telemetry = [*steps, provider_step]
        assert "Light the beacon" not in repr(persisted_telemetry)
        assert "The bell answers" not in repr(persisted_telemetry)
    finally:
        state.close()
        with sqlite3.connect(tmp_path / "web-data" / "bragi.sqlite3") as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
