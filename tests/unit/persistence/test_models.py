from __future__ import annotations

import json

from bragi.persistence.models import (
    LossConditionChangeRecord,
    LossOutcomeRecord,
    SaveScenarioUpdateRecord,
)


def test_save_scenario_update_active_reflects_archive_timestamp() -> None:
    active = SaveScenarioUpdateRecord(
        id="update-1",
        save_id="save-1",
        source_message_id="message-1",
        title="Changed title",
        premise="Changed premise",
        player_role="Warden",
        content_json="{}",
        source_message_ids_json="[]",
        reason="new fact",
        provider="fake",
        model="fake-chat",
    )
    archived = SaveScenarioUpdateRecord(
        **{**active.__dict__, "id": "update-2", "archived_at": "2026-05-01T00:00:00"}
    )

    assert active.active is True
    assert archived.active is False


def test_loss_condition_change_json_properties_sort_keys_and_preserve_none() -> None:
    record = LossConditionChangeRecord(
        id="change-1",
        save_id="save-1",
        condition_id="condition-1",
        source_message_id="message-1",
        operation="update",
        before={"b": 1, "a": 2},
        after=None,
        reason="changed",
        provider="fake",
        model="fake-chat",
    )

    assert record.active is True
    assert record.before_json == json.dumps({"a": 2, "b": 1}, sort_keys=True)
    assert record.after_json is None


def test_loss_outcome_computed_properties_use_evidence_fallbacks() -> None:
    outcome = LossOutcomeRecord(
        id="outcome-1",
        save_id="save-1",
        condition_id="condition-1",
        condition_name="Doom clock",
        triggering_message_id="message-9",
        explanation="The doom clock struck midnight.",
        evidence={
            "epilogue": "The city fell silent.",
            "items": [{"quote": "midnight"}],
        },
        confidence=0.91,
        provider="fake",
        model="fake-chat",
        epilogue_provider=None,
        epilogue_model=None,
        epilogue_message_id=None,
        epilogue_error=None,
    )
    fallback = LossOutcomeRecord(
        **{
            **outcome.__dict__,
            "id": "outcome-2",
            "evidence": {"items": [{"quote": "bell"}]},
            "archived_at": "2026-05-01T00:00:00",
        }
    )

    assert outcome.active is True
    assert outcome.source_message_id == "message-9"
    assert outcome.title == "Doom clock"
    assert outcome.body == "The doom clock struck midnight."
    assert outcome.epilogue == "The city fell silent."
    assert outcome.evidence_json == json.dumps([{"quote": "midnight"}], sort_keys=True)
    assert fallback.active is False
    assert fallback.epilogue == "The doom clock struck midnight."
