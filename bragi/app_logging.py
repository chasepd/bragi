"""Metadata-only local application logging."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Final, cast

from bragi.persistence.paths import StoragePaths
from bragi.private_files import ensure_private_dir, ensure_private_file
from bragi.redaction import redact_log_value, redact_text

LOGGER_NAME: Final = "bragi"
_HANDLER_MARKER: Final = "_bragi_rotating_file_handler"
_DEFAULT_LOG_LEVEL: Final = logging.INFO
_DEBUG_LOG_LEVEL: Final = logging.DEBUG


def configure_logging(paths: StoragePaths) -> Path:
    """Configure Bragi's metadata-only rotating file logger."""

    log_file_path = log_path(paths)
    ensure_private_file(log_file_path)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_log_level(os.environ.get("BRAGI_LOG_LEVEL")))
    logger.propagate = False

    existing = _configured_handlers(logger)
    for handler in existing[1:]:
        logger.removeHandler(handler)
        handler.close()

    if existing:
        handler = existing[0]
        if Path(handler.baseFilename) != log_file_path:
            logger.removeHandler(handler)
            handler.close()
            handler = _new_handler(log_file_path)
            logger.addHandler(handler)
    else:
        handler = _new_handler(log_file_path)
        logger.addHandler(handler)

    handler.setLevel(logger.level)
    handler.setFormatter(_JsonLineFormatter())
    log_file_path.chmod(0o600)
    return log_file_path


def log_path(paths: StoragePaths) -> Path:
    return paths.state_dir / "logs" / "bragi.log"


def log_event(event: str, /, **fields: object) -> None:
    logging.getLogger(LOGGER_NAME).info(
        event,
        extra={
            "bragi_event": event,
            "bragi_fields": fields,
        },
    )


def log_debug_event(event: str, /, **fields: object) -> None:
    logging.getLogger(LOGGER_NAME).debug(
        event,
        extra={
            "bragi_event": event,
            "bragi_fields": fields,
        },
    )


def log_error_event(event: str, /, **fields: object) -> None:
    logging.getLogger(LOGGER_NAME).error(
        event,
        extra={
            "bragi_event": event,
            "bragi_fields": fields,
        },
    )


def set_debug_logging_enabled(enabled: bool) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    level = (
        _DEBUG_LOG_LEVEL
        if enabled
        else _log_level(os.environ.get("BRAGI_LOG_LEVEL"))
    )
    logger.setLevel(level)
    for handler in _configured_handlers(logger):
        handler.setLevel(level)
    log_event("logging.debug_mode_changed", enabled=enabled)


def exception_log_fields(exc: Exception) -> dict[str, object]:
    category = getattr(exc, "category", None)
    category_value = getattr(category, "value", None)
    if isinstance(category_value, str):
        return {
            "error_category": category_value,
            "error": category_value,
            **_provider_fallback_log_fields(exc),
        }
    return {
        "error_category": type(exc).__name__,
        "error": str(exc),
        **_provider_fallback_log_fields(exc),
    }


def _provider_fallback_log_fields(exc: Exception) -> dict[str, object]:
    fields: dict[str, object] = {}
    fallback_attempted = getattr(exc, "fallback_attempted", None)
    if isinstance(fallback_attempted, bool):
        fields["fallback_attempted"] = fallback_attempted
    fallback_skipped_reason = getattr(exc, "fallback_skipped_reason", None)
    if isinstance(fallback_skipped_reason, str) and fallback_skipped_reason:
        fields["fallback_skipped_reason"] = fallback_skipped_reason
    fallback_provider = getattr(exc, "fallback_provider", None)
    if isinstance(fallback_provider, str) and fallback_provider:
        fields["fallback_provider"] = fallback_provider
    fallback_model_id = getattr(exc, "fallback_model_id", None)
    if isinstance(fallback_model_id, str) and fallback_model_id:
        fields["fallback_model_id"] = fallback_model_id
    return fields


def _configured_handlers(logger: logging.Logger) -> list[RotatingFileHandler]:
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, RotatingFileHandler)
        and getattr(handler, _HANDLER_MARKER, False)
    ]


class _PrivateRotatingFileHandler(RotatingFileHandler):
    def _open(self) -> TextIOWrapper[Any]:
        path = Path(self.baseFilename)
        ensure_private_dir(path.parent)
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        return cast(
            TextIOWrapper,
            os.fdopen(
                fd,
                self.mode,
                encoding=self.encoding,
                errors=self.errors,
            ),
        )


def _new_handler(log_file_path: Path) -> RotatingFileHandler:
    handler = _PrivateRotatingFileHandler(
        log_file_path,
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    setattr(handler, _HANDLER_MARKER, True)
    return handler


def _log_level(value: str | None) -> int:
    if not value:
        return _DEFAULT_LOG_LEVEL
    level = logging.getLevelName(value.upper())
    return level if isinstance(level, int) else _DEFAULT_LOG_LEVEL


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "bragi_fields", {})
        if not isinstance(fields, dict):
            fields = {}
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "event": redact_text(
                str(getattr(record, "bragi_event", record.getMessage()))
            ),
            **{
                str(key): redact_log_value(value, key=str(key))
                for key, value in fields.items()
            },
        }
        message = redact_text(record.getMessage())
        if message and message != payload["event"]:
            payload["message"] = message
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
