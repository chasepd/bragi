"""Safe, structured diagnostics for persisted terminal jobs."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from datetime import UTC, datetime

from bragi.persistence.models import JobRecord
from bragi.redaction import redact_text

_MAX_PROMPT_LENGTH = 12_000
_MAX_ERROR_LENGTH = 1_000
_MAX_LIST_ITEMS = 20
_JOB_REQUEST_CONTEXT: ContextVar[dict[str, object] | None] = ContextVar(
    "bragi_job_request_context",
    default=None,
)
_ORIGIN_LABELS = {
    "manual_character_reference": "Manual character reference image",
    "manual_character_image": "Manual scene character image",
    "manual_registry_character_image": "Manual character registry image",
    "manual_scene_image": "Manual scene image",
    "character_text_attachment": "Character text message image",
    "automatic_post_turn": "Automatic post-turn image",
    "initial_scenario_image": "Initial scenario image",
    "manual_image_animation": "Manual image animation",
    "scheduled_maintenance": "Scheduled maintenance",
    "chat_turn": "Chat turn",
    "scenario_wizard": "Scenario wizard",
}
_JOB_TYPE_ORIGINS = {
    "chat_completion": "chat_turn",
    "scenario_draft": "scenario_wizard",
    "scenario_generation": "scenario_wizard",
    "state_pruning": "scheduled_maintenance",
    "context_cleanup": "scheduled_maintenance",
    "guided_context_cleanup": "scheduled_maintenance",
    "memory_consolidation": "scheduled_maintenance",
    "world_context_retention": "scheduled_maintenance",
    "dating_sim_maintenance": "scheduled_maintenance",
    "character_registry_maintenance": "scheduled_maintenance",
}
_PARAMETER_KEYS = frozenset(
    {
        "dimensions",
        "image_style_preset",
        "prompt_chars",
        "scene_context_chars",
        "safe_mode",
        "source_media_asset_ids",
        "source_character_reference_asset_ids",
        "fallback_enabled",
        "replace_existing",
        "post_turn_inference_mode",
        "effective_post_turn_inference_mode",
    }
)
_PROVIDER_KEYS = frozenset(
    {
        "provider",
        "model",
        "original_provider",
        "original_model",
        "final_provider",
        "final_model",
        "fallback_provider",
        "fallback_model",
        "fallback_task",
        "fallback_skipped_reason",
        "retrieval_degraded",
        "retrieval_recovery",
        "classification",
        "error_category",
        "primary_error_category",
        "final_error_category",
        "primary_http_status",
        "final_http_status",
        "http_status",
        "primary_error_message",
        "final_error_message",
        "provider_headers",
        "finish_reason",
        "native_finish_reason",
        "fallback_used",
        "attempt_count",
        "max_attempts",
        "provider_call_count",
        "provider_calls",
        "retry_attempts",
    }
)
_SEXUAL_CONTENT_CLASSIFICATIONS = frozenset(
    {
        "acceptable_romance",
        "fade_needed_sexual_escalation",
        "explicit_disallowed",
    }
)


def build_job_diagnostic_snapshot(
    job: JobRecord,
    *,
    request_context: Mapping[str, object] | None = None,
    result: Mapping[str, object] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    """Build a whitelist-only terminal snapshot from job lifecycle data."""

    payload = job.payload
    terminal_result = result if result is not None else (job.result or {})
    context = request_context or {}
    origin_kind = _origin_kind(job, context)
    request = _request_detail(
        job,
        payload=payload,
        context=context,
        origin_kind=origin_kind,
    )
    provider = _provider_detail(
        payload=payload,
        result=terminal_result,
        context=context,
    )
    failure_text = error if error is not None else job.error
    snapshot: dict[str, object] = {
        "version": 1,
        "request": request,
        "provider": provider,
        "bragi": {
            "status": job.status,
            **(
                {
                    "error": _bounded_redacted_text(
                        failure_text,
                        limit=_MAX_ERROR_LENGTH,
                    )
                }
                if failure_text
                else {}
            ),
        },
        "timing": _timing_detail(job),
    }
    world_time = _world_time_detail(_world_time_result(terminal_result))
    if world_time and isinstance(snapshot["bragi"], dict):
        snapshot["bragi"]["world_time"] = world_time
    related = _related_detail(payload=payload, context=context)
    if related:
        snapshot["related"] = related
    return snapshot


def set_job_request_context(
    context: dict[str, object],
) -> Token[dict[str, object] | None]:
    return _JOB_REQUEST_CONTEXT.set(dict(context))


def reset_job_request_context(token: Token[dict[str, object] | None]) -> None:
    _JOB_REQUEST_CONTEXT.reset(token)


def current_job_request_context() -> dict[str, object]:
    return dict(_JOB_REQUEST_CONTEXT.get() or {})


def redact_job_diagnostic_snapshot(
    snapshot: Mapping[str, object],
    *,
    include_prompt: bool,
    include_failure_detail: bool | None = None,
) -> dict[str, object]:
    """Return a response-safe copy of a snapshot."""

    failure_detail_visible = (
        include_prompt if include_failure_detail is None else include_failure_detail
    )
    safe: dict[str, object] = {}
    version = snapshot.get("version")
    if isinstance(version, int) and not isinstance(version, bool):
        safe["version"] = version
    request = snapshot.get("request")
    if isinstance(request, Mapping):
        safe["request"] = _safe_stored_request(
            request,
            include_prompt=include_prompt,
        )
    provider = snapshot.get("provider")
    if isinstance(provider, Mapping):
        safe_provider = _provider_detail(
            payload={},
            result=provider,
            context={},
        )
        if not failure_detail_visible:
            safe_provider = {
                key: value
                for key, value in safe_provider.items()
                if not key.endswith("_message") and key != "error_message"
            }
        safe["provider"] = safe_provider
    bragi = snapshot.get("bragi")
    if isinstance(bragi, Mapping):
        safe_bragi: dict[str, object] = {}
        if isinstance(bragi.get("status"), str):
            safe_bragi["status"] = bragi["status"]
        if failure_detail_visible and bragi.get("error"):
            safe_bragi["error"] = _bounded_redacted_text(
                bragi["error"],
                limit=_MAX_ERROR_LENGTH,
            )
        world_time = _world_time_detail(bragi.get("world_time"))
        if world_time:
            safe_bragi["world_time"] = world_time
        if safe_bragi:
            safe["bragi"] = safe_bragi
    timing = snapshot.get("timing")
    if isinstance(timing, Mapping):
        safe_timing: dict[str, object] = {}
        for key in (
            "created_at",
            "started_at",
            "completed_at",
            "duration_ms",
            "queue_wait_ms",
        ):
            value = timing.get(key)
            if isinstance(value, str | int) or value is None:
                safe_timing[key] = value
        if safe_timing:
            safe["timing"] = safe_timing
    related = snapshot.get("related")
    if isinstance(related, Mapping):
        safe_related = {
            key: value
            for key in ("parent_job_id", "media_asset_id", "source_media_asset_id")
            if isinstance((value := related.get(key)), str) and value
        }
        if safe_related:
            safe["related"] = safe_related
    return safe


def job_origin_summary(job: JobRecord) -> dict[str, str]:
    kind = _origin_kind(job, {})
    return {"kind": kind, "label": _ORIGIN_LABELS.get(kind, _labelize(kind))}


def _request_detail(
    job: JobRecord,
    *,
    payload: Mapping[str, object],
    context: Mapping[str, object],
    origin_kind: str,
) -> dict[str, object]:
    merged: dict[str, object] = {**payload, **context}
    origin: dict[str, object] = {
        "kind": origin_kind,
        "label": _ORIGIN_LABELS.get(origin_kind, _labelize(origin_kind)),
    }
    if isinstance(context.get("route"), str) and context["route"]:
        origin["route"] = context["route"]
    if isinstance(context.get("request_id"), str) and context["request_id"]:
        origin["request_id"] = context["request_id"]
    request: dict[str, object] = {
        "origin": origin,
        "job_type": job.type,
        "task": _text_value(merged.get("task")) or _task_for_job(job.type),
        "provider": _text_value(merged.get("provider")),
        "model": _text_value(merged.get("model")),
        "save_id": job.save_id,
        "parameters": _parameters(merged),
    }
    for output_key, input_keys in (
        ("source_message_id", ("source_message_id", "request_source_message_id")),
        ("source_text_message_id", ("source_text_message_id", "text_message_id")),
        ("character_id", ("character_id",)),
        ("thread_id", ("thread_id",)),
    ):
        value = _first_text(merged, input_keys)
        if value:
            request[output_key] = value
    media_ids = _string_list(
        merged.get("source_media_asset_ids")
        or merged.get("source_character_reference_asset_ids")
    )
    if media_ids:
        request["source_media_asset_ids"] = media_ids
    prompt = context.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        request["prompt"] = _bounded_redacted_text(prompt)
    return request


def _safe_stored_request(
    request: Mapping[str, object],
    *,
    include_prompt: bool,
) -> dict[str, object]:
    safe: dict[str, object] = {}
    origin = request.get("origin")
    if isinstance(origin, Mapping):
        safe_origin = {
            key: value
            for key in ("kind", "label", "route", "request_id")
            if isinstance((value := origin.get(key)), str) and value
        }
        if safe_origin:
            safe["origin"] = safe_origin
    for key in (
        "job_type",
        "task",
        "provider",
        "model",
        "save_id",
        "source_message_id",
        "source_text_message_id",
        "character_id",
        "thread_id",
    ):
        value = request.get(key)
        if isinstance(value, str) and value:
            safe[key] = value
    media_ids = _string_list(request.get("source_media_asset_ids"))
    if media_ids:
        safe["source_media_asset_ids"] = media_ids
    parameters = request.get("parameters")
    if isinstance(parameters, Mapping):
        safe["parameters"] = _parameters(parameters)
    if include_prompt and isinstance(request.get("prompt"), str):
        safe["prompt"] = _bounded_redacted_text(request["prompt"])
    return safe


def _provider_detail(
    *,
    payload: Mapping[str, object],
    result: Mapping[str, object],
    context: Mapping[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = {**payload, **result, **context}
    provider: dict[str, object] = {}
    for key in _PROVIDER_KEYS:
        if key in {
            "primary_error_category",
            "final_error_category",
            "primary_http_status",
            "final_http_status",
        }:
            continue
        value = merged.get(key)
        if key in {"provider_calls", "retry_attempts"}:
            normalized = _safe_diagnostic_list(value)
            if normalized:
                provider[key] = normalized
        elif key == "provider_headers":
            normalized_headers = _safe_headers(value)
            if normalized_headers:
                provider[key] = normalized_headers
        elif isinstance(value, str) and value:
            provider[key] = (
                _bounded_redacted_text(value, limit=_MAX_ERROR_LENGTH)
                if key.endswith("message")
                else value
            )
        elif isinstance(value, int) and not isinstance(value, bool):
            provider[key] = value
        elif isinstance(value, bool):
            provider[key] = value
    if "provider" not in provider and isinstance(payload.get("provider"), str):
        provider["provider"] = payload["provider"]
    if "model" not in provider and isinstance(payload.get("model"), str):
        provider["model"] = payload["model"]
    error_category = _first_text(
        merged,
        ("error_category", "final_error_category", "primary_error_category"),
    )
    if error_category:
        provider["error_category"] = error_category
    http_status = _first_int(
        merged,
        ("http_status", "final_http_status", "primary_http_status"),
    )
    if http_status is not None:
        provider["http_status"] = http_status
    sexual_content_safety = _sexual_content_safety_detail(
        merged.get("sexual_content_safety")
    )
    if sexual_content_safety:
        provider["sexual_content_safety"] = sexual_content_safety
    return provider


def _world_time_detail(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    safe: dict[str, object] = {}
    for key in (
        "status",
        "skipped_reason",
        "evidence_source_id",
        "evidence_quote",
    ):
        item = value.get(key)
        if isinstance(item, str) and item:
            safe[key] = _bounded_redacted_text(
                item,
                limit=_MAX_ERROR_LENGTH if key == "evidence_quote" else 240,
            )
    changed = value.get("changed")
    if isinstance(changed, bool):
        safe["changed"] = changed
    queued_count = value.get("queued_count")
    if isinstance(queued_count, int) and not isinstance(queued_count, bool):
        safe["queued_count"] = queued_count
    confidence = value.get("confidence")
    if isinstance(confidence, int | float) and not isinstance(confidence, bool):
        safe["confidence"] = max(0.0, min(1.0, float(confidence)))
    for key in ("source_message_ids", "queued_suggestion_ids", "updated_fields"):
        strings = _string_list(value.get(key))
        if strings:
            safe[key] = strings
    for key in ("before", "proposed", "after"):
        detail = _world_time_values(value.get(key))
        if detail:
            safe[key] = detail
    return safe


def _sexual_content_safety_detail(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    classification = value.get("classification")
    transition_applied = value.get("transition_applied")
    safe: dict[str, object] = {}
    if classification in _SEXUAL_CONTENT_CLASSIFICATIONS:
        safe["classification"] = classification
    if isinstance(transition_applied, bool):
        safe["transition_applied"] = transition_applied
    return safe


def _world_time_result(result: Mapping[str, object]) -> object:
    direct = result.get("world_time")
    if isinstance(direct, Mapping):
        return direct
    jobs = result.get("jobs")
    if not isinstance(jobs, list | tuple):
        return None
    for item in jobs:
        if not isinstance(item, Mapping):
            continue
        if item.get("name") != "time_reconciliation":
            continue
        child_result = item.get("result")
        if isinstance(child_result, Mapping):
            return child_result
    return None


def _world_time_values(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    safe: dict[str, object] = {}
    for key in ("in_world_time", "time_of_day", "day_of_week"):
        item = value.get(key)
        if isinstance(item, str):
            safe[key] = _bounded_redacted_text(item, limit=240)
    world_day_index = value.get("world_day_index")
    if (
        isinstance(world_day_index, int)
        and not isinstance(world_day_index, bool)
    ) or world_day_index is None:
        safe["world_day_index"] = world_day_index
    return safe


def _related_detail(
    *,
    payload: Mapping[str, object],
    context: Mapping[str, object],
) -> dict[str, object]:
    merged = {**payload, **context}
    related: dict[str, object] = {}
    for output_key, input_keys in (
        ("parent_job_id", ("parent_job_id",)),
        ("media_asset_id", ("media_asset_id",)),
        ("source_media_asset_id", ("source_media_asset_id",)),
    ):
        value = _first_text(merged, input_keys)
        if value:
            related[output_key] = value
    return related


def _origin_kind(job: JobRecord, context: Mapping[str, object]) -> str:
    raw = context.get("kind") or context.get("origin") or payload_value(
        job.payload,
        "job_context",
    )
    if isinstance(raw, str) and raw.strip():
        normalized = raw.strip()
        if normalized in _ORIGIN_LABELS:
            return normalized
    return _JOB_TYPE_ORIGINS.get(job.type, _fallback_origin(job.type))


def _fallback_origin(job_type: str) -> str:
    if "image" in job_type or "media" in job_type:
        return "manual_scene_image"
    if "maintenance" in job_type or "cleanup" in job_type or "pruning" in job_type:
        return "scheduled_maintenance"
    return job_type or "unknown"


def _task_for_job(job_type: str) -> str:
    if "image" in job_type or "media" in job_type:
        return "image_generation"
    if job_type == "chat_completion":
        return "chat"
    return job_type


def _parameters(values: Mapping[str, object]) -> dict[str, object]:
    nested = values.get("parameters")
    if isinstance(nested, Mapping):
        values = {**values, **nested}
    parameters: dict[str, object] = {}
    for key in _PARAMETER_KEYS:
        value = values.get(key)
        if isinstance(value, bool | int | float | str):
            parameters[key] = value
        elif key in {"dimensions"} and isinstance(value, list | tuple):
            numbers = [item for item in value if isinstance(item, int)]
            if len(numbers) == 2:
                parameters[key] = numbers
        elif key in {"source_media_asset_ids", "source_character_reference_asset_ids"}:
            strings = _string_list(value)
            if strings:
                parameters[key] = strings
    return parameters


def _timing_detail(job: JobRecord) -> dict[str, object]:
    timing: dict[str, object] = {
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "duration_ms": job.duration_ms,
    }
    queue_wait_ms = _duration_ms(job.created_at, job.started_at)
    if queue_wait_ms is not None:
        timing["queue_wait_ms"] = queue_wait_ms
    return timing


def _duration_ms(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    try:
        start_dt = _parse_timestamp(start)
        end_dt = _parse_timestamp(end)
    except ValueError:
        return None
    return max(0, round((end_dt - start_dt).total_seconds() * 1000))


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _safe_headers(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    safe: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            continue
        normalized = key.strip().lower()
        if normalized in {
            "cf-ray",
            "content-type",
            "x-request-id",
            "x-retry-count",
            "x-venice-contains-minor",
            "x-venice-is-adult-model-content-violation",
            "x-venice-is-blurred",
            "x-venice-is-content-violation",
        }:
            safe[normalized] = item[:200]
    return safe


def _safe_diagnostic_list(value: object) -> list[object]:
    if not isinstance(value, list | tuple):
        return []
    safe: list[object] = []
    for item in value[:_MAX_LIST_ITEMS]:
        if isinstance(item, Mapping):
            safe_item: dict[str, object] = {}
            for key in (
                "attempt",
                "duration_ms",
                "error_category",
                "http_status",
                "task",
                "provider",
                "model",
                "operation",
            ):
                item_value = item.get(key)
                if isinstance(item_value, str | int) and not isinstance(
                    item_value, bool
                ):
                    safe_item[key] = (
                        _bounded_redacted_text(item_value)
                        if isinstance(item_value, str)
                        else item_value
                    )
                elif item_value is None and key == "error_category":
                    safe_item[key] = None
            if safe_item:
                safe.append(safe_item)
    return safe


def payload_value(payload: Mapping[str, object], key: str) -> object:
    return payload.get(key)


def _first_text(values: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _text_value(values.get(key))
        if value:
            return value
    return None


def _first_int(values: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _text_value(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str) and item][: _MAX_LIST_ITEMS]


def _bounded_redacted_text(
    value: object,
    *,
    limit: int = _MAX_PROMPT_LENGTH,
) -> str:
    text = redact_text(str(value).strip()) or ""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _labelize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title() or "Unknown"
