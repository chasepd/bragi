from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import ChatMessage, ChatRequest, ChatResponse
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.retry import call_with_provider_retries
from bragi.services.runtime_telemetry import (
    runtime_telemetry_context,
    wrap_provider_clients_for_telemetry,
)


@dataclass
class _VirtualLatencyClock:
    seconds: float = 0.0

    def now(self) -> float:
        return self.seconds

    def advance_ms(self, milliseconds: int) -> None:
        self.seconds += milliseconds / 1000

    async def sleep(self, seconds: float) -> None:
        self.seconds += seconds


@dataclass
class _DelayedRetryProviderHarness:
    clock: _VirtualLatencyClock
    attempt_delays_ms: tuple[int, ...]
    milestones: list[tuple[str, int]] = field(default_factory=list)
    provider_call_wave_count: int = 0

    provider_name = "fake"

    @property
    def retry_count(self) -> int:
        return max(0, self.provider_call_wave_count - 1)

    @property
    def completion_ms(self) -> int:
        return round(self.clock.now() * 1000)

    def mark(self, name: str) -> None:
        self.milestones.append((name, self.completion_ms))

    async def chat(self, request: ChatRequest) -> ChatResponse:
        wave_index = self.provider_call_wave_count
        self.provider_call_wave_count += 1
        self.mark("provider_call_started")
        self.clock.advance_ms(self.attempt_delays_ms[wave_index])
        if wave_index == 0:
            self.mark("provider_call_failed")
            raise ProviderError(
                ProviderErrorCategory.NETWORK_ERROR,
                "scripted transient failure",
            )
        self.mark("provider_call_succeeded")
        return ChatResponse(
            body="The bell answers.",
            provider=request.provider,
            model_id=request.model_id,
        )


def test_delayed_fake_harness_orders_milestones_and_counts_retry_waves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    repositories = PersistenceRepositories(sqlite3.connect(database_path))
    clock = _VirtualLatencyClock()
    harness = _DelayedRetryProviderHarness(clock, (120, 80))
    provider = wrap_provider_clients_for_telemetry(
        {"fake": harness},
        repositories=repositories,
    )["fake"]
    monkeypatch.setattr("bragi.providers.retry.perf_counter", clock.now)
    monkeypatch.setattr("bragi.providers.retry.asyncio.sleep", clock.sleep)
    monkeypatch.setattr(
        "bragi.providers.retry._retry_delay",
        lambda **_kwargs: 0.0,
    )
    monkeypatch.setattr("bragi.services.runtime_telemetry.perf_counter", clock.now)

    async def run_test() -> None:
        job = repositories.create_job(
            type="chat_turn",
            status="running",
            payload={},
        )
        request = ChatRequest(
            provider="fake",
            model_id="fake-chat",
            messages=(ChatMessage(role="user", body="private player text"),),
        )
        harness.mark("optimistic_player_painted")
        with runtime_telemetry_context(
            repositories=repositories,
            job_id=job.id,
            task="chat",
        ):
            response = await call_with_provider_retries(
                lambda: provider.chat(request),
                provider="fake",
                task="chat",
                max_attempts=2,
            )
        harness.mark("response_committed")
        harness.mark("narrator_painted")

        assert response.body == "The bell answers."
        assert harness.provider_call_wave_count == 2
        assert harness.retry_count == 1
        assert harness.completion_ms == 200
        assert harness.milestones == [
            ("optimistic_player_painted", 0),
            ("provider_call_started", 0),
            ("provider_call_failed", 120),
            ("provider_call_started", 120),
            ("provider_call_succeeded", 200),
            ("response_committed", 200),
            ("narrator_painted", 200),
        ]
        steps = repositories.list_job_steps(job.id)
        assert [(step.status, step.duration_ms) for step in steps] == [
            ("failed", 120),
            ("succeeded", 80),
        ]
        assert "private player text" not in repr(steps)
        assert "The bell answers" not in repr(steps)

    asyncio.run(run_test())
