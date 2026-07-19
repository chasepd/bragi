from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pytest import MonkeyPatch

from bragi.persistence.paths import StoragePaths


def test_configure_logging_installs_one_state_rotating_file_handler_idempotently(
    tmp_path: Path,
) -> None:
    from bragi.app_logging import configure_logging

    paths = _storage_paths(tmp_path)
    expected_log_path = paths.state_dir / "logs" / "bragi.log"
    bragi_logger = logging.getLogger("bragi")
    original_handlers = tuple(bragi_logger.handlers)
    original_level = bragi_logger.level
    original_propagate = bragi_logger.propagate

    try:
        configure_logging(paths)
        configure_logging(paths)

        handlers = [
            handler
            for handler in bragi_logger.handlers
            if isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == expected_log_path
        ]
        assert handlers == [
            handler
            for handler in bragi_logger.handlers
            if isinstance(handler, RotatingFileHandler)
        ]
        assert len(handlers) == 1
        assert expected_log_path.parent.is_dir()

        logging.getLogger("bragi.tests.logging").warning("file handler smoke")
        _flush_handlers(bragi_logger.handlers)

        assert expected_log_path.read_text(encoding="utf-8")
    finally:
        _restore_logger(
            bragi_logger,
            original_handlers,
            original_level=original_level,
            original_propagate=original_propagate,
        )


def test_configure_logging_redacts_tokens_api_keys_and_bearer_values(
    tmp_path: Path,
) -> None:
    from bragi.app_logging import configure_logging

    paths = _storage_paths(tmp_path)
    log_path = paths.state_dir / "logs" / "bragi.log"
    bragi_logger = logging.getLogger("bragi")
    original_handlers = tuple(bragi_logger.handlers)
    original_level = bragi_logger.level
    original_propagate = bragi_logger.propagate

    token_secret = "plain-token-secret"
    api_key_secret = "plain-api-key-secret"
    bearer_secret = "plain-bearer-secret"
    sk_secret = "sk-live-secret-value"

    try:
        configure_logging(paths)

        logging.getLogger("bragi.tests.logging").error(
            "provider failed with token=%s api_key: %s Bearer %s %s",
            token_secret,
            api_key_secret,
            bearer_secret,
            sk_secret,
        )
        _flush_handlers(bragi_logger.handlers)

        log_text = log_path.read_text(encoding="utf-8")
        assert "provider failed" in log_text
        assert "[redacted]" in log_text
        assert token_secret not in log_text
        assert api_key_secret not in log_text
        assert bearer_secret not in log_text
        assert sk_secret not in log_text
    finally:
        _restore_logger(
            bragi_logger,
            original_handlers,
            original_level=original_level,
            original_propagate=original_propagate,
        )


def test_set_debug_logging_enabled_updates_runtime_levels_and_keeps_redaction(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    from bragi.app_logging import (
        configure_logging,
        log_debug_event,
        set_debug_logging_enabled,
    )

    monkeypatch.delenv("BRAGI_LOG_LEVEL", raising=False)
    paths = _storage_paths(tmp_path)
    log_path = paths.state_dir / "logs" / "bragi.log"
    bragi_logger = logging.getLogger("bragi")
    original_handlers = tuple(bragi_logger.handlers)
    original_level = bragi_logger.level
    original_propagate = bragi_logger.propagate

    enabled_secret = "sk-debug-enabled-secret"
    disabled_secret = "sk-debug-disabled-secret"

    try:
        configure_logging(paths)
        handlers = [
            handler
            for handler in bragi_logger.handlers
            if isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == log_path
        ]
        assert handlers
        assert bragi_logger.level == logging.INFO
        assert {handler.level for handler in handlers} == {logging.INFO}

        log_debug_event("logging.debug_disabled", token=disabled_secret)
        _flush_handlers(bragi_logger.handlers)
        log_text = log_path.read_text(encoding="utf-8")
        assert "logging.debug_disabled" not in log_text
        assert disabled_secret not in log_text

        set_debug_logging_enabled(True)
        assert bragi_logger.level == logging.DEBUG
        assert {handler.level for handler in handlers} == {logging.DEBUG}

        log_debug_event(
            "logging.debug_enabled",
            token=enabled_secret,
            api_key=enabled_secret,
        )
        _flush_handlers(bragi_logger.handlers)
        log_text = log_path.read_text(encoding="utf-8")
        assert "logging.debug_enabled" in log_text
        assert "[redacted]" in log_text
        assert enabled_secret not in log_text

        set_debug_logging_enabled(False)
        assert bragi_logger.level == logging.INFO
        assert {handler.level for handler in handlers} == {logging.INFO}

        log_debug_event("logging.debug_disabled_again")
        _flush_handlers(bragi_logger.handlers)
        assert "logging.debug_disabled_again" not in log_path.read_text(
            encoding="utf-8"
        )
    finally:
        _restore_logger(
            bragi_logger,
            original_handlers,
            original_level=original_level,
            original_propagate=original_propagate,
        )


def _storage_paths(tmp_path: Path) -> StoragePaths:
    data_dir = tmp_path / "data"
    return StoragePaths(
        data_dir=data_dir,
        database_path=data_dir / "bragi.sqlite3",
        media_dir=data_dir / "media",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
    )


def _flush_handlers(handlers: list[logging.Handler]) -> None:
    for handler in handlers:
        handler.flush()


def _restore_logger(
    logger: logging.Logger,
    original_handlers: tuple[logging.Handler, ...],
    *,
    original_level: int,
    original_propagate: bool,
) -> None:
    for handler in logger.handlers:
        if handler not in original_handlers:
            handler.close()
    logger.handlers[:] = list(original_handlers)
    logger.setLevel(original_level)
    logger.propagate = original_propagate
