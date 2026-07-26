from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from bragi.persistence.models import JobRecord
from bragi.services.job_diagnostics import (
    build_job_diagnostic_snapshot,
    redact_job_diagnostic_snapshot,
)


def test_snapshot_classifies_manual_character_reference_request() -> None:
    job = JobRecord(
        id="job-1",
        save_id="save-1",
        type="character_reference_image",
        status="failed",
        payload={
            "job_context": "manual_character_reference",
            "character_id": "character-1",
            "source_message_id": "message-1",
            "provider": "venice",
            "model": "venice/image",
        },
        result={
            "final_error_category": "content_blocked",
            "final_http_status": 400,
            "attempt_count": 2,
            "max_attempts": 3,
            "retry_attempts": [
                {
                    "attempt": 1,
                    "duration_ms": 20,
                    "error_category": "content_blocked",
                    "http_status": 400,
                }
            ],
        },
        error="content blocked by provider",
        created_at="2026-07-10T12:00:00Z",
        started_at="2026-07-10T12:00:01Z",
        completed_at="2026-07-10T12:00:02Z",
        duration_ms=1000,
    )

    snapshot = build_job_diagnostic_snapshot(job)

    assert snapshot["request"] == {
        "origin": {
            "kind": "manual_character_reference",
            "label": "Manual character reference image",
        },
        "job_type": "character_reference_image",
        "task": "image_generation",
        "provider": "venice",
        "model": "venice/image",
        "save_id": "save-1",
        "source_message_id": "message-1",
        "character_id": "character-1",
        "parameters": {},
    }
    assert snapshot["provider"] == {
        "provider": "venice",
        "model": "venice/image",
        "error_category": "content_blocked",
        "http_status": 400,
        "attempt_count": 2,
        "max_attempts": 3,
        "retry_attempts": [
            {
                "attempt": 1,
                "duration_ms": 20,
                "error_category": "content_blocked",
                "http_status": 400,
            }
        ],
    }
    assert snapshot["bragi"] == {
        "status": "failed",
        "error": "content blocked by provider",
    }
    assert snapshot["timing"] == {
        "created_at": "2026-07-10T12:00:00Z",
        "started_at": "2026-07-10T12:00:01Z",
        "completed_at": "2026-07-10T12:00:02Z",
        "duration_ms": 1000,
        "queue_wait_ms": 1000,
    }


def test_snapshot_includes_context_search_degradation_metadata() -> None:
    job = JobRecord(
        id="job-context",
        save_id="save-1",
        type="context_search",
        status="succeeded",
        payload={"provider": "fake", "model": "fake-context"},
        result={
            "retrieval_degraded": True,
            "retrieval_recovery": "provider_fallback",
            "final_provider": "fallback",
            "final_model": "fallback-tools",
            "fallback_used": True,
            "fallback_provider": "fallback",
            "fallback_model": "fallback-tools",
            "error_category": "model_not_found",
            "http_status": 404,
            "raw_prompt": "secret roleplay prompt",
        },
        error=None,
    )

    snapshot = build_job_diagnostic_snapshot(job)
    public = redact_job_diagnostic_snapshot(snapshot, include_prompt=False)

    assert snapshot["provider"] == {
        "provider": "fake",
        "model": "fake-context",
        "retrieval_degraded": True,
        "retrieval_recovery": "provider_fallback",
        "final_provider": "fallback",
        "final_model": "fallback-tools",
        "fallback_used": True,
        "fallback_provider": "fallback",
        "fallback_model": "fallback-tools",
        "error_category": "model_not_found",
        "http_status": 404,
    }
    assert public["provider"] == snapshot["provider"]
    assert "secret roleplay prompt" not in repr(snapshot)
    assert "secret roleplay prompt" not in repr(public)


def test_snapshot_preserves_explicit_text_message_origin_and_admin_prompt() -> None:
    job = JobRecord(
        id="job-2",
        save_id="save-1",
        type="character_text_image_generation",
        status="succeeded",
        payload={
            "job_context": "character_text_attachment",
            "provider": "openrouter",
            "model": "openrouter/image",
        },
        result={"media_asset_id": "asset-1"},
        error=None,
    )

    snapshot = build_job_diagnostic_snapshot(
        job,
        request_context={
            "kind": "character_text_attachment",
            "text_message_id": "text-1",
            "character_id": "character-1",
            "prompt": "A private but intentionally captured image prompt.",
            "prompt_chars": 48,
            "parameters": {"image_style_preset": "cinematic"},
        },
    )

    request = cast(Mapping[str, object], snapshot["request"])
    assert request["origin"] == {
        "kind": "character_text_attachment",
        "label": "Character text message image",
    }
    assert request["source_text_message_id"] == "text-1"
    assert request["prompt"] == (
        "A private but intentionally captured image prompt."
    )
    assert request["parameters"] == {
        "image_style_preset": "cinematic",
        "prompt_chars": 48,
    }


def test_non_admin_snapshot_removes_prompt_and_raw_unknown_fields() -> None:
    job = JobRecord(
        id="job-3",
        save_id="save-1",
        type="image_generation",
        status="failed",
        payload={"provider": "fake"},
        result=None,
        error="failed token=super-secret",
    )
    snapshot = build_job_diagnostic_snapshot(
        job,
        request_context={
            "origin": "manual_scene_image",
            "prompt": "secret roleplay prompt",
            "provider_payload": {"api_key": "secret"},
            "parameters": {"dimensions": [1024, 1024]},
        },
        result={
            "error_category": "provider_error",
            "final_error_message": "private provider response text",
        },
    )

    public = redact_job_diagnostic_snapshot(snapshot, include_prompt=False)

    public_request = cast(Mapping[str, object], public["request"])
    public_provider = cast(Mapping[str, object], public["provider"])
    public_bragi = cast(Mapping[str, object], public["bragi"])
    assert "prompt" not in public_request
    assert public_provider["error_category"] == "provider_error"
    assert "final_error_message" not in public_provider
    assert public_bragi == {"status": "failed"}
    assert "provider_payload" not in repr(public)
    assert "private provider response text" not in repr(public)
    assert "secret" not in repr(public)
    assert public_request["parameters"] == {"dimensions": [1024, 1024]}


def test_snapshot_includes_whitelisted_world_time_diagnostics() -> None:
    job = JobRecord(
        id="job-4",
        save_id="save-1",
        type="chat_completion",
        status="succeeded",
        payload={"provider": "fake", "model": "fake-chat"},
        result={
            "world_time": {
                "status": "queued",
                "skipped_reason": "narrator_only_ambiguous",
                "queued_count": 2,
                "source_message_ids": ["player-1", "narrator-1"],
                "evidence_source_id": "narrator-1",
                "evidence_quote": "Evening shadows cross the tower.",
                "reason": "Mara says a private phrase in the tower.",
                "before": {
                    "in_world_time": "Monday morning",
                    "time_of_day": "morning",
                    "day_of_week": "monday",
                    "world_day_index": 0,
                },
                "proposed": {
                    "in_world_time": "Monday evening",
                    "time_of_day": "evening",
                    "day_of_week": "monday",
                    "world_day_index": 0,
                },
                "after": {
                    "in_world_time": "Monday morning",
                    "time_of_day": "morning",
                    "day_of_week": "monday",
                    "world_day_index": 0,
                },
                "provider_payload": {"api_key": "secret"},
            },
            "raw_private_field": "should not be surfaced",
        },
        error=None,
    )

    snapshot = build_job_diagnostic_snapshot(job)
    public = redact_job_diagnostic_snapshot(snapshot, include_prompt=False)
    bragi = cast(Mapping[str, object], snapshot["bragi"])

    assert bragi["world_time"] == {
        "status": "queued",
        "skipped_reason": "narrator_only_ambiguous",
        "queued_count": 2,
        "source_message_ids": ["player-1", "narrator-1"],
        "evidence_source_id": "narrator-1",
        "evidence_quote": "Evening shadows cross the tower.",
        "before": {
            "in_world_time": "Monday morning",
            "time_of_day": "morning",
            "day_of_week": "monday",
            "world_day_index": 0,
        },
        "proposed": {
            "in_world_time": "Monday evening",
            "time_of_day": "evening",
            "day_of_week": "monday",
            "world_day_index": 0,
        },
        "after": {
            "in_world_time": "Monday morning",
            "time_of_day": "morning",
            "day_of_week": "monday",
            "world_day_index": 0,
        },
    }
    assert public["bragi"] == bragi
    assert "raw_private_field" not in repr(snapshot)
    assert "api_key" not in repr(snapshot)
    assert "private phrase" not in repr(snapshot)


def test_snapshot_includes_post_turn_world_time_child_diagnostics() -> None:
    job = JobRecord(
        id="job-5",
        save_id="save-1",
        type="post_turn_maintenance",
        status="succeeded",
        payload={},
        result={
            "jobs": [
                {"name": "context", "status": "succeeded"},
                {
                    "name": "time_reconciliation",
                    "status": "queued",
                    "result": {
                        "status": "queued",
                        "skipped_reason": "narrator_only_ambiguous",
                        "queued_count": 1,
                        "source_message_ids": ["player-1", "narrator-1"],
                        "reason": "Mara says a private phrase in the tower.",
                        "provider_payload": {"api_key": "secret"},
                    },
                },
            ],
            "raw_private_field": "should not be surfaced",
        },
        error=None,
    )

    snapshot = build_job_diagnostic_snapshot(job)
    bragi = cast(Mapping[str, object], snapshot["bragi"])

    assert bragi["world_time"] == {
        "status": "queued",
        "skipped_reason": "narrator_only_ambiguous",
        "queued_count": 1,
        "source_message_ids": ["player-1", "narrator-1"],
    }
    assert "raw_private_field" not in repr(snapshot)
    assert "api_key" not in repr(snapshot)
    assert "private phrase" not in repr(snapshot)


def test_snapshot_surfaces_only_safe_sexual_content_diagnostics() -> None:
    job = JobRecord(
        id="job-sexual-safety",
        save_id="save-1",
        type="chat_completion",
        status="succeeded",
        payload={"provider": "fake", "model": "fake-chat"},
        result={
            "sexual_content_safety": {
                "classification": "explicit_disallowed",
                "transition_applied": True,
                "rejected_draft": "private prose that must not escape",
            },
            "classification": "explicit_content_transition_applied",
        },
        error=None,
    )

    snapshot = build_job_diagnostic_snapshot(job)
    public = redact_job_diagnostic_snapshot(snapshot, include_prompt=False)
    provider = cast(Mapping[str, object], snapshot["provider"])
    public_provider = cast(Mapping[str, object], public["provider"])

    assert provider["sexual_content_safety"] == {
        "classification": "explicit_disallowed",
        "transition_applied": True,
    }
    assert public_provider["sexual_content_safety"] == {
        "classification": "explicit_disallowed",
        "transition_applied": True,
    }
    assert "rejected_draft" not in repr(snapshot)
