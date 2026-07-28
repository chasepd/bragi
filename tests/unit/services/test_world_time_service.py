from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.interaction_mode import InteractionMode
from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import MessageRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import StructuredOutputRequest, StructuredOutputResponse
from bragi.safety import FADE_TO_BLACK_TRANSITION_KIND
from bragi.services.sexual_content_safety import FADE_TO_BLACK_TRANSITION
from bragi.services.world_time_service import (
    StructuredProviderWorldTimeChecker,
    WorldTimeAssessment,
    WorldTimeService,
    _completed_turn_review_reason,
    latest_message_may_advance_time,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


class RecordingStructuredTimeProvider:
    provider_name = "fake"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.requests.append(request)
        return StructuredOutputResponse(
            data=self.payload,
            provider=request.provider,
            model_id=request.model_id,
        )


class ExplodingChecker:
    async def assess(self, **_: object) -> WorldTimeAssessment:
        raise AssertionError("time checker should not run")


def test_latest_message_may_advance_time_detects_explicit_waits() -> None:
    assert latest_message_may_advance_time("We wait until evening.")
    assert latest_message_may_advance_time("I sleep until tomorrow morning.")
    assert latest_message_may_advance_time(
        "We spend the afternoon traveling to the observatory.",
    )
    assert latest_message_may_advance_time("We wait half an hour.")
    assert latest_message_may_advance_time("I head home after the meeting.")
    assert latest_message_may_advance_time("The next day, I check the beacon.")
    assert latest_message_may_advance_time("We regroup at 14:30.")
    assert not latest_message_may_advance_time("I look at the evening lanterns.")
    assert not latest_message_may_advance_time(
        "Later, I ask whether anyone heard the bell.",
    )


def test_marked_fade_transition_can_supply_elapsed_time_evidence() -> None:
    player = MessageRecord(
        id="player",
        save_id="save",
        role="player",
        body="I close my eyes.",
        speaker_name="Mara",
        provider=None,
        model=None,
        token_estimate=None,
    )
    narrator = MessageRecord(
        id="narrator",
        save_id="save",
        role="narrator",
        body=FADE_TO_BLACK_TRANSITION,
        speaker_name="Narrator",
        provider=None,
        model=None,
        token_estimate=None,
        safety_transition=FADE_TO_BLACK_TRANSITION_KIND,
    )
    assessment = WorldTimeAssessment(
        changed=True,
        time_of_day="morning",
        confidence=0.75,
        evidence_quote=FADE_TO_BLACK_TRANSITION,
    )

    assert _completed_turn_review_reason(
        policy="completed_turn",
        assessment=assessment,
        source_message=narrator,
        candidate_messages=(player, narrator),
        proposed_values={"time_of_day": "morning"},
    ) == ""


def test_world_time_service_applies_canonical_fade_elapsed_time(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, narrator_message_id = _save_with_completed_turn(
        repositories,
        player_body="I close my eyes.",
        narrator_body=FADE_TO_BLACK_TRANSITION,
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "morning",
            "day_of_week": "tuesday",
            "days_elapsed": 1,
            "evidence_source_id": narrator_message_id,
            "evidence_quote": FADE_TO_BLACK_TRANSITION,
            "confidence": 0.75,
            "reason": "The canonical transition confirms that hours passed.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.reconcile_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "applied"
    assert snapshot.day_of_week == "tuesday"
    assert snapshot.world_day_index == 1
    assert snapshot.last_updated_message_id == narrator_message_id


def test_latest_message_may_advance_time_rejects_timer_readouts() -> None:
    assert not latest_message_may_advance_time(
        "I wait until 03:45 remains on the countdown timer.",
    )
    assert not latest_message_may_advance_time(
        "We wait until the timer hits 03:45 left.",
    )
    assert not latest_message_may_advance_time(
        "The round has 03:45 remaining.",
    )
    assert not latest_message_may_advance_time(
        "I wait until the game clock shows 03:45.",
    )
    assert not latest_message_may_advance_time(
        "The scoreboard reads 03:45.",
    )
    assert not latest_message_may_advance_time(
        "The game clock shows 3 p.m.",
    )
    assert not latest_message_may_advance_time(
        "I wait to see what happens.",
    )
    assert not latest_message_may_advance_time(
        "I wait until the shot clock hits 12 seconds.",
    )
    assert latest_message_may_advance_time("We wait until 3:45 p.m.")
    assert latest_message_may_advance_time("We arrive at 3:45 p.m.")
    assert latest_message_may_advance_time(
        "I wait until evening while the countdown timer shows 03:45 remaining.",
    )
    assert latest_message_may_advance_time(
        "We meet in the northern quarter at 14:30.",
    )
    assert latest_message_may_advance_time(
        "We meet at 14:30 in the half-light.",
    )


def test_world_time_service_skips_provider_without_advance_signal(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="I inspect the beacon lens.",
    )
    service = WorldTimeService(repositories=repositories, checker=ExplodingChecker())

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "no_time_advance_signal"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "monday"
    assert snapshot.in_world_time == "Monday morning"


def test_world_time_service_skips_countdown_timer_signal(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="I wait while the countdown timer shows 03:45 left in the round.",
    )
    service = WorldTimeService(repositories=repositories, checker=ExplodingChecker())

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "no_time_advance_signal"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "monday"
    assert snapshot.world_day_index == 0


def test_world_time_service_runs_checker_for_broadened_time_signal(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="We spend the afternoon traveling to the observatory.",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": False,
            "time_of_day": "",
            "day_of_week": "",
            "days_elapsed": 0,
            "evidence_source_id": "",
            "evidence_quote": "",
            "confidence": 0.0,
            "reason": "The structured checker declined the candidate.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "assessment_unchanged"
    assert len(provider.requests) == 1
    assert snapshot.in_world_time == "Monday morning"


def test_world_time_service_rejects_timer_evidence_from_provider_assessment(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body=(
            "The countdown timer shows 03:45 remaining, so I wait until evening "
            "before leaving the arena."
        ),
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "night",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "03:45 remaining",
            "confidence": 0.96,
            "reason": "The timer readout was mistaken for a clock time.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "timer_readout_not_clock"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "monday"
    assert snapshot.in_world_time == "Monday morning"
    assert snapshot.world_day_index == 0


def test_world_time_service_rejects_normalized_timer_evidence_quote(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body=(
            "The countdown timer has 03:45 remaining until   dawn, so I wait "
            "until evening before leaving the arena."
        ),
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "dawn",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "until dawn",
            "confidence": 0.96,
            "reason": "The timer target was mistaken for a clock advance.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "timer_readout_not_clock"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "monday"
    assert snapshot.in_world_time == "Monday morning"
    assert snapshot.world_day_index == 0


def test_world_time_service_rejects_game_clock_evidence_from_assessment(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body=(
            "The game clock shows 03:45, so I wait until evening before "
            "leaving the arena."
        ),
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "night",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "03:45",
            "confidence": 0.96,
            "reason": "The game clock readout was mistaken for a world time.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "timer_readout_not_clock"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "monday"
    assert snapshot.in_world_time == "Monday morning"
    assert snapshot.world_day_index == 0


def test_world_time_service_rejects_meridiem_game_clock_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body=(
            "The game clock shows 3:45 p.m., so I wait until evening before "
            "leaving the arena."
        ),
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "afternoon",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "3:45 p.m.",
            "confidence": 0.96,
            "reason": "The game clock readout was mistaken for a world time.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "timer_readout_not_clock"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "monday"
    assert snapshot.in_world_time == "Monday morning"
    assert snapshot.world_day_index == 0


def test_world_time_service_rejects_no_minute_meridiem_game_clock_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body=(
            "The game clock shows 3 p.m., so I wait until evening before "
            "leaving the arena."
        ),
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "afternoon",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "3 p.m.",
            "confidence": 0.96,
            "reason": "The game clock readout was mistaken for a world time.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "timer_readout_not_clock"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "monday"
    assert snapshot.in_world_time == "Monday morning"
    assert snapshot.world_day_index == 0


def test_world_time_service_rejects_seconds_only_shot_clock_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body=(
            "The shot clock hits 12 seconds, so I wait until evening before "
            "leaving the arena."
        ),
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "night",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "12 seconds",
            "confidence": 0.96,
            "reason": "The shot clock readout was mistaken for elapsed time.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "timer_readout_not_clock"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "monday"
    assert snapshot.in_world_time == "Monday morning"
    assert snapshot.world_day_index == 0


def test_world_time_service_applies_real_time_evidence_with_timer_elsewhere(
    repositories: PersistenceRepositories,
) -> None:
    body = (
        "The countdown timer shows 03:45 remaining, so I wait until evening "
        "before leaving the arena."
    )
    save_id, message_id = _save_with_time_message(repositories, body=body)
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": body,
            "confidence": 0.92,
            "reason": "The player explicitly waited until evening.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "applied"
    assert snapshot.time_of_day == "evening"
    assert snapshot.in_world_time == "Monday evening"
    assert snapshot.world_day_index == 0


def test_world_time_service_applies_structured_time_update(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="We wait until evening.",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "wait until evening",
            "confidence": 0.91,
            "reason": "The player explicitly waited until evening.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "applied"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "monday"
    assert snapshot.in_world_time == "Monday evening"
    assert snapshot.world_day_index == 0
    assert snapshot.last_updated_message_id == message_id
    assert provider.requests[0].schema_name == "world_time_advance"
    assert provider.requests[0].max_output_tokens == 1024
    assert "world_day_index: 0" in provider.requests[0].messages[1].body


def test_world_time_service_advances_weekday_from_elapsed_days(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="I sleep until tomorrow morning.",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "morning",
            "day_of_week": "",
            "days_elapsed": 1,
            "evidence_source_id": message_id,
            "evidence_quote": "sleep until tomorrow morning",
            "confidence": 0.93,
            "reason": "The player slept to the next morning.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "applied"
    assert snapshot.time_of_day == "morning"
    assert snapshot.day_of_week == "tuesday"
    assert snapshot.in_world_time == "Tuesday morning"
    assert snapshot.world_day_index == 1


def test_world_time_service_preserves_legacy_text_for_index_only_advance(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="I sleep until the next cycle.",
        world_day_index=4,
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        in_world_time="morning",
        time_of_day="morning",
        day_of_week="",
        world_day_index=4,
    )
    repositories.connection.execute(
        """
        UPDATE scene_snapshots
        SET in_world_time = 'Cycle 4, morning after the festival',
            time_of_day = 'morning',
            day_of_week = '',
            world_day_index = 4,
            world_time_day_index = 4,
            world_time_day_label = '',
            world_time_phase = 'morning'
        WHERE save_id = ?
        """,
        (save_id,),
    )
    repositories.connection.commit()
    seeded_snapshot = repositories.get_scene_snapshot(save_id)
    assert seeded_snapshot is not None
    assert seeded_snapshot.in_world_time == "Cycle 4, morning after the festival"
    assert seeded_snapshot.time_of_day == "morning"
    assert seeded_snapshot.day_of_week == ""
    assert seeded_snapshot.world_day_index == 4
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "morning",
            "day_of_week": "",
            "days_elapsed": 1,
            "evidence_source_id": message_id,
            "evidence_quote": "sleep until the next cycle",
            "confidence": 0.93,
            "reason": "The player slept into the next cycle.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "applied"
    assert snapshot.in_world_time == "Cycle 4, morning after the festival"
    assert snapshot.world_day_index == 5


def test_world_time_service_advances_world_day_index_from_explicit_weekday(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="I wait until Wednesday evening.",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "wednesday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "wait until Wednesday evening",
            "confidence": 0.94,
            "reason": "The player explicitly waited until Wednesday evening.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "applied"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "wednesday"
    assert snapshot.in_world_time == "Wednesday evening"
    assert snapshot.world_day_index == 2


def test_world_time_service_preserves_unknown_day_index_for_same_day_update(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="We wait until evening.",
        world_day_index=None,
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "wait until evening",
            "confidence": 0.91,
            "reason": "The player explicitly waited until evening.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "applied"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "monday"
    assert snapshot.in_world_time == "Monday evening"
    assert snapshot.world_day_index is None


def test_world_time_service_queues_locked_scene_time_suggestion(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="We wait until evening.",
        locked_fields=["time_of_day"],
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "wait until evening",
            "confidence": 0.91,
            "reason": "The player explicitly waited until evening.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "queued"
    assert snapshot.time_of_day == "morning"
    suggestions = repositories.list_context_update_suggestions(save_id)
    assert [
        (item.entity_type, item.field_path, item.proposed_value)
        for item in suggestions
    ] == [
        ("scene_snapshot", "time_of_day", "evening"),
        ("scene_snapshot", "in_world_time", "Monday evening"),
    ]


def test_world_time_service_queues_canonical_locked_phase_suggestion(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="We wait until evening.",
        locked_fields=["world_time_phase"],
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "wait until evening",
            "confidence": 0.91,
            "reason": "The player explicitly waited until evening.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "queued"
    assert snapshot.world_time_phase == "morning"
    assert repositories.list_context_update_suggestions(save_id)


def test_world_time_service_preserves_existing_clock_and_period_metadata(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="We wait until evening.",
    )
    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation=snapshot.situation,
        in_world_time=snapshot.in_world_time,
        time_of_day=snapshot.time_of_day,
        day_of_week=snapshot.day_of_week,
        world_day_index=snapshot.world_day_index,
        world_time_day_index=snapshot.world_day_index,
        world_time_day_label=snapshot.day_of_week,
        world_time_phase=snapshot.time_of_day,
        world_time_clock_minutes=9 * 60 + 30,
        world_time_period_label="festival week",
        source_message_id=snapshot.source_message_id,
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "wait until evening",
            "confidence": 0.91,
            "reason": "The player explicitly waited until evening.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "applied"
    assert snapshot.world_time_phase == "evening"
    assert snapshot.world_time_clock_minutes == 9 * 60 + 30
    assert snapshot.world_time_period_label == "festival week"


def test_world_time_service_preserves_arbitrary_canonical_day_label(
    repositories: PersistenceRepositories,
) -> None:
    save_id, message_id = _save_with_time_message(
        repositories,
        body="We wait until evening.",
    )
    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation=snapshot.situation,
        in_world_time="Cycle 4 morning",
        time_of_day="morning",
        day_of_week="",
        world_day_index=4,
        world_time_day_index=4,
        world_time_day_label="Cycle 4",
        world_time_phase="morning",
        source_message_id=snapshot.source_message_id,
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "",
            "days_elapsed": 0,
            "evidence_source_id": message_id,
            "evidence_quote": "wait until evening",
            "confidence": 0.91,
            "reason": "The player explicitly waited until evening.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.advance_time_if_supported(
            save_id=save_id,
            latest_message_id=message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "applied"
    assert snapshot.world_time_day_label == "Cycle 4"
    assert snapshot.world_time_phase == "evening"


def test_world_time_service_reconciles_player_authorized_narrator_time_jump(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, narrator_message_id = _save_with_completed_turn(
        repositories,
        player_body="I sleep until tomorrow morning.",
        narrator_body="Tuesday morning finds the tower quiet again.",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "morning",
            "day_of_week": "tuesday",
            "days_elapsed": 1,
            "evidence_source_id": narrator_message_id,
            "evidence_quote": "Tuesday morning",
            "confidence": 0.86,
            "reason": "The narrator confirmed the player's requested sleep.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.reconcile_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "applied"
    assert snapshot.day_of_week == "tuesday"
    assert snapshot.time_of_day == "morning"
    assert snapshot.world_day_index == 1
    assert snapshot.in_world_time == "Tuesday morning"
    assert snapshot.last_updated_message_id == narrator_message_id
    diagnostics = result.to_json()
    assert diagnostics["source_message_ids"] == [
        player_message_id,
        narrator_message_id,
    ]
    assert diagnostics["evidence_source_id"] == narrator_message_id
    assert diagnostics["updated_fields"] == [
        "day_of_week",
        "world_day_index",
        "in_world_time",
    ]
    assert diagnostics["before"] == {
        "in_world_time": "Monday morning",
        "time_of_day": "morning",
        "day_of_week": "monday",
        "world_day_index": 0,
    }
    assert diagnostics["after"] == {
        "in_world_time": "Tuesday morning",
        "time_of_day": "morning",
        "day_of_week": "tuesday",
        "world_day_index": 1,
    }
    assert provider.requests[0].schema_name == "world_time_reconciliation"
    assert f"[{player_message_id}] player" in provider.requests[0].messages[1].body
    assert f"[{narrator_message_id}] narrator" in provider.requests[0].messages[1].body


def test_world_time_service_queues_ambiguous_narrator_only_time_jump(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, narrator_message_id = _save_with_completed_turn(
        repositories,
        player_body="I inspect the beacon lens.",
        narrator_body="Evening shadows cross the tower windows.",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": narrator_message_id,
            "evidence_quote": "Evening shadows",
            "confidence": 0.72,
            "reason": "The narrator implied a time jump without player action.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.reconcile_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "queued"
    assert result.skipped_reason == "narrator_only_ambiguous"
    assert snapshot.time_of_day == "morning"
    assert snapshot.in_world_time == "Monday morning"
    suggestions = repositories.list_context_update_suggestions(save_id)
    assert [
        (item.field_path, item.proposed_value)
        for item in suggestions
    ] == [
        ("time_of_day", "evening"),
        ("in_world_time", "Monday evening"),
    ]
    diagnostics = result.to_json()
    assert diagnostics["queued_count"] == 2
    assert diagnostics["skipped_reason"] == "narrator_only_ambiguous"
    assert diagnostics["queued_suggestion_ids"] == [
        item.id for item in suggestions
    ]
    assert diagnostics["proposed"] == {
        "in_world_time": "Monday evening",
        "time_of_day": "evening",
        "day_of_week": "monday",
        "world_day_index": 0,
    }
    assert diagnostics["after"] == {
        "in_world_time": "Monday morning",
        "time_of_day": "morning",
        "day_of_week": "monday",
        "world_day_index": 0,
    }


def test_world_time_service_queues_conflicting_completed_turn_time_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, narrator_message_id = _save_with_completed_turn(
        repositories,
        player_body="We wait until evening.",
        narrator_body="Night has settled by the time the beacon is repaired.",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "night",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": narrator_message_id,
            "evidence_quote": "Night has settled",
            "confidence": 0.95,
            "reason": "The narrator contradicted the player's evening target.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.reconcile_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "queued"
    assert result.skipped_reason == "conflicting_time_evidence"
    assert snapshot.time_of_day == "morning"
    assert [
        (item.field_path, item.proposed_value)
        for item in repositories.list_context_update_suggestions(save_id)
    ] == [
        ("time_of_day", "night"),
        ("in_world_time", "Monday night"),
    ]


def test_world_time_service_queues_conflict_when_provider_cites_player(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, narrator_message_id = _save_with_completed_turn(
        repositories,
        player_body="We wait until evening.",
        narrator_body="Night has settled by the time the beacon is repaired.",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": player_message_id,
            "evidence_quote": "wait until evening",
            "confidence": 0.95,
            "reason": "The player requested an evening time jump.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.reconcile_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "queued"
    assert result.skipped_reason == "conflicting_time_evidence"
    assert snapshot.time_of_day == "morning"
    assert [
        (item.field_path, item.proposed_value)
        for item in repositories.list_context_update_suggestions(save_id)
    ] == [
        ("time_of_day", "evening"),
        ("in_world_time", "Monday evening"),
    ]


def test_storyteller_world_time_rejects_direction_as_completed_turn_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, narrator_message_id = _save_with_completed_turn(
        repositories,
        player_body="Advance to evening.",
        narrator_body="The beacon remains lit in the morning haze.",
        interaction_mode=InteractionMode.STORYTELLER,
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": player_message_id,
            "evidence_quote": "Advance to evening",
            "confidence": 0.95,
            "reason": "The direction requested an evening scene.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.reconcile_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status != "applied"
    assert snapshot.time_of_day == "morning"


def test_world_time_service_skips_reconciliation_without_checker(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, narrator_message_id = _save_with_completed_turn(
        repositories,
        player_body="I sleep until tomorrow morning.",
        narrator_body="Tuesday morning finds the tower quiet again.",
    )
    service = WorldTimeService(repositories=repositories, checker=None)

    result = asyncio.run(
        service.reconcile_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
        )
    )

    assert result.status == "skipped"
    assert result.skipped_reason == "checker_unavailable"


def test_world_time_service_reconciliation_rejects_timer_readout_evidence(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_message_id, narrator_message_id = _save_with_completed_turn(
        repositories,
        player_body="I check the arena door.",
        narrator_body="The game clock shows 03:45 over the arena gate.",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "afternoon",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "evidence_source_id": narrator_message_id,
            "evidence_quote": "03:45",
            "confidence": 0.96,
            "reason": "The game clock was mistaken for world time.",
        }
    )
    service = WorldTimeService(
        repositories=repositories,
        checker=StructuredProviderWorldTimeChecker(
            provider=provider,
            provider_name="fake",
            model_id="fake-time",
        ),
    )

    result = asyncio.run(
        service.reconcile_completed_turn(
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "timer_readout_not_clock"
    assert snapshot.time_of_day == "morning"
    assert snapshot.in_world_time == "Monday morning"


def test_world_time_service_time_loop_phase_advance_keeps_repeated_day_index(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="time_loop",
        title="Bellwether Day",
        premise="The harbor festival repeats.",
        player_role="Archivist",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Bell Loop")
    message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="We wait until evening.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The dawn bell rings.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=4,
        world_time_clock_minutes=8 * 60,
        source_message_id=message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="loop.current",
        value={"summary": "Loop 1, dawn phase."},
        category="loop_status",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "evening",
            "day_of_week": "monday",
            "days_elapsed": 1,
            "loop_transition": "phase_advance",
            "clock_minutes": 19 * 60,
            "period_label": "festival day",
            "evidence_source_id": message.id,
            "evidence_quote": "wait until evening",
            "confidence": 0.95,
            "reason": "The repeated day advances to evening.",
        }
    )

    result = asyncio.run(
        WorldTimeService(
            repositories=repositories,
            checker=StructuredProviderWorldTimeChecker(
                provider=provider,
                provider_name="fake",
                model_id="fake-time",
            ),
        ).advance_time_if_supported(save_id=save.id, latest_message_id=message.id)
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    current = next(
        row
        for row in repositories.list_world_state(save.id)
        if row.key == "loop.current"
    )
    assert result.status == "applied"
    assert snapshot is not None
    assert snapshot.world_day_index == 4
    assert snapshot.day_of_week == "monday"
    assert snapshot.in_world_time == "Monday evening"
    assert snapshot.time_of_day == "evening"
    assert snapshot.world_time_clock_minutes == 19 * 60
    assert current.value["iteration"] == 1
    assert current.value["last_transition"] == "phase_advance"
    assert current.value["baseline_time"] == {
        "day_index": 4,
        "day_label": "monday",
        "phase": "morning",
        "clock_minutes": 8 * 60,
        "period_label": "",
    }
    current_time = current.value["current_time"]
    assert isinstance(current_time, dict)
    assert current_time["phase"] == "evening"


def test_world_time_service_time_loop_reset_restores_baseline_and_preserves_knowledge(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="time_loop",
        title="Bellwether Day",
        premise="The harbor festival repeats.",
        player_role="Archivist",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Bell Loop")
    message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I wait until the bell rings and the loop resets.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The dawn bell rings.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=0,
        world_time_clock_minutes=8 * 60,
        source_message_id=message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="loop.resettable.note",
        value={"status": "changed"},
        category="loop_resettable",
        source_message_id=message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="loop.knowledge",
        value={"summary": "The tower code persists."},
        category="loop_persistent",
    )
    from bragi.services.time_loop_time_policy import TimeLoopTimePolicy

    policy = TimeLoopTimePolicy(repositories, save_id=save.id)
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    policy.capture_baseline(snapshot)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The bell has already returned to dawn.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=0,
        world_time_clock_minutes=8 * 60,
        source_message_id=message.id,
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="loop.resettable.note",
        value={"status": "changed again"},
        category="loop_resettable",
    )
    provider = RecordingStructuredTimeProvider(
        {
            "changed": True,
            "time_of_day": "morning",
            "day_of_week": "monday",
            "days_elapsed": 0,
            "loop_transition": "reset",
            "clock_minutes": None,
            "period_label": "",
            "evidence_source_id": message.id,
            "evidence_quote": "loop resets",
            "confidence": 0.95,
            "reason": "The bell reset the loop.",
        }
    )

    result = asyncio.run(
        WorldTimeService(
            repositories=repositories,
            checker=StructuredProviderWorldTimeChecker(
                provider=provider,
                provider_name="fake",
                model_id="fake-time",
            ),
        ).advance_time_if_supported(save_id=save.id, latest_message_id=message.id)
    )

    reset_snapshot = repositories.get_scene_snapshot(save.id)
    state = {row.key: row for row in repositories.list_world_state(save.id)}
    assert result.status == "applied"
    assert reset_snapshot is not None
    assert reset_snapshot.world_day_index == 0
    assert reset_snapshot.time_of_day == "morning"
    assert state["loop.resettable.note"].value == {"status": "changed"}
    assert state["loop.resettable.note"].source_message_id == message.id
    assert state["loop.knowledge"].value == {"summary": "The tower code persists."}
    assert state["loop.current"].value["iteration"] == 2
    assert state["loop.current"].value["last_transition"] == "reset"
    state_changes = repositories.list_state_changes(save.id)
    assert any(
        change.state_key == "loop.resettable.note"
        and change.operation == "upsert"
        and change.source_message_id == message.id
        for change in state_changes
    )
    assert any(
        change.state_key == "loop.current" and change.operation == "upsert"
        for change in state_changes
    )


def _save_with_time_message(
    repositories: PersistenceRepositories,
    *,
    body: str,
    locked_fields: list[str] | None = None,
    world_day_index: int | None = 0,
) -> tuple[str, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body=body,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The beacon lens waits in the tower.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=world_day_index,
        source_message_id=message.id,
        locked_fields=locked_fields or [],
    )
    return save.id, message.id


def _save_with_completed_turn(
    repositories: PersistenceRepositories,
    *,
    player_body: str,
    narrator_body: str,
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY,
) -> tuple[str, str, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        interaction_mode=interaction_mode,
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Ily",
        body=player_body,
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=narrator_body,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The beacon lens waits in the tower.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=0,
        source_message_id=player_message.id,
    )
    return save.id, player_message.id, narrator_message.id
