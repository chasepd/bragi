"""Metadata-only observability helpers for the web layer."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal, cast

from bragi_web.bragi_adapter import bragi_logging_bindings

LogLevel = Literal["debug", "info", "error"]

_RECENT_EVENTS: deque[dict[str, Any]] = deque(maxlen=200)
_MAX_EVENT_LENGTH = 80
_MAX_KEY_LENGTH = 60
_MAX_STRING_LENGTH = 240
_MAX_FIELDS = 24
_RESERVED_CLIENT_FIELD_KEYS = frozenset({"event", "level", "source", "timestamp"})
_BODY_LIKE_KEYS = frozenset(
    {
        "body",
        "prompt",
        "message",
        "messages",
        "content",
        "payload",
        "request",
        "response",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "token",
    }
)
_SENSITIVE_KEY_FRAGMENTS = frozenset(
    {
        "apikey",
        "authorization",
        "credential",
        "cookie",
        "password",
        "secret",
        "token",
    }
)


def observe(
    event: str,
    **fields: object,
) -> None:
    requested_level = fields.pop("level", "info")
    level: LogLevel = (
        cast(LogLevel, requested_level)
        if requested_level in {"debug", "info", "error"}
        else "info"
    )
    safe_event = _safe_text(event, max_length=_MAX_EVENT_LENGTH) or "web.event"
    safe_fields = sanitize_fields(fields)
    recent = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "event": safe_event,
        **safe_fields,
    }
    _RECENT_EVENTS.append(recent)
    bindings = bragi_logging_bindings()
    if level == "debug":
        bindings.log_debug_event(safe_event, **safe_fields)
    elif level == "error":
        bindings.log_error_event(safe_event, **safe_fields)
    else:
        bindings.log_event(safe_event, **safe_fields)


def recent_events() -> list[dict[str, Any]]:
    return list(reversed(_RECENT_EVENTS))


def clear_recent_events() -> None:
    _RECENT_EVENTS.clear()


def sanitize_fields(fields: Mapping[str, object] | object) -> dict[str, object]:
    if not isinstance(fields, Mapping):
        return {}
    safe: dict[str, object] = {}
    for key, value in list(fields.items())[:_MAX_FIELDS]:
        safe_key = _safe_key(key)
        if safe_key is None or _is_body_like_key(safe_key):
            continue
        safe[safe_key] = _sanitize_value(value, key=safe_key)
    return safe


def sanitize_client_fields(fields: Mapping[str, object] | object) -> dict[str, object]:
    return {
        key: value
        for key, value in sanitize_fields(fields).items()
        if _normalized_key(key) not in _RESERVED_CLIENT_FIELD_KEYS
    }


def result_shape(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        fields: dict[str, object] = {
            "result_type": "object",
            "result_keys": sorted(str(key)[:_MAX_KEY_LENGTH] for key in value.keys())[
                :12
            ],
        }
        for key in ("count", "created_count", "updated_count", "archived_count"):
            count = value.get(key)
            if isinstance(count, int | float | bool):
                fields[key] = count
        return fields
    if isinstance(value, list | tuple | set):
        return {"result_type": "array", "result_count": len(value)}
    if value is None:
        return {"result_type": "none"}
    return {"result_type": type(value).__name__}


def error_fields(exc: BaseException | str | None) -> dict[str, object]:
    if exc is None:
        return {}
    if isinstance(exc, BaseException):
        return {
            "error_class": type(exc).__name__,
            "error": _safe_text(str(exc) or type(exc).__name__),
        }
    return {
        "error_class": "Error",
        "error": _safe_text(exc),
    }


def _sanitize_value(value: object, *, key: str | None = None) -> object:
    redacted = bragi_logging_bindings().redact_log_value(value, key=key)
    if isinstance(redacted, str):
        if key is not None and key.lower().replace("-", "_") in {"path", "route"}:
            redacted = redacted.split("?", 1)[0].split("#", 1)[0]
        return _safe_text(redacted) or ""
    if isinstance(redacted, int | float | bool) or redacted is None:
        return redacted
    if isinstance(redacted, Mapping):
        return sanitize_fields(redacted)
    if isinstance(redacted, list | tuple):
        return [_sanitize_value(item) for item in redacted[:10]]
    return _safe_text(str(redacted)) or ""


def _safe_text(
    value: str | None,
    *,
    max_length: int = _MAX_STRING_LENGTH,
) -> str | None:
    redacted = cast(str | None, bragi_logging_bindings().redact_text(value))
    if redacted is None:
        return None
    redacted = redacted.replace("\r", " ").replace("\n", " ")
    if len(redacted) <= max_length:
        return redacted
    if max_length <= 3:
        return redacted[:max_length]
    return f"{redacted[: max_length - 3]}..."


def _safe_key(key: object) -> str | None:
    safe = str(key).strip()
    if not safe:
        return None
    return safe[:_MAX_KEY_LENGTH]


def _is_body_like_key(key: str) -> bool:
    normalized = _normalized_key(key)
    compact = normalized.replace("_", "")
    return (
        normalized in _BODY_LIKE_KEYS
        or normalized.endswith("_body")
        or any(fragment in compact for fragment in _SENSITIVE_KEY_FRAGMENTS)
    )


def _normalized_key(key: str) -> str:
    return key.lower().replace("-", "_")
