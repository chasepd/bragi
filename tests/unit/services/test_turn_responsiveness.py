from __future__ import annotations

from typing import cast

import pytest

from bragi.persistence.repositories import PersistenceRepositories
from bragi.retry_policy import RetryExecutionClass
from bragi.services.turn_responsiveness import (
    TURN_RESPONSIVENESS_MODE_QUALITY,
    TURN_RESPONSIVENESS_MODE_RESPONSIVE,
    TURN_RESPONSIVENESS_MODE_SETTING,
    retry_execution_class_for_save,
    sanitize_turn_responsiveness_mode,
    turn_responsiveness_mode,
)


class FakeRepositories:
    def __init__(self, value: object = None) -> None:
        self.value = value

    def get_effective_setting(
        self,
        key: str,
        *,
        save_id: str | None = None,
        user_id: str | None = None,
        scenario_id: str | None = None,
    ) -> object:
        assert key == TURN_RESPONSIVENESS_MODE_SETTING
        return self.value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, TURN_RESPONSIVENESS_MODE_QUALITY),
        ("quality", TURN_RESPONSIVENESS_MODE_QUALITY),
        ("responsive", TURN_RESPONSIVENESS_MODE_RESPONSIVE),
        (" RESPONSIVE ", TURN_RESPONSIVENESS_MODE_QUALITY),
        ("fast", TURN_RESPONSIVENESS_MODE_QUALITY),
        (True, TURN_RESPONSIVENESS_MODE_QUALITY),
    ],
)
def test_turn_responsiveness_mode_is_strictly_sanitized(
    value: object,
    expected: str,
) -> None:
    assert sanitize_turn_responsiveness_mode(value) == expected


@pytest.mark.parametrize(
    ("value", "expected_mode", "expected_execution_class"),
    [
        (
            None,
            TURN_RESPONSIVENESS_MODE_QUALITY,
            RetryExecutionClass.QUALITY_FOREGROUND,
        ),
        (
            TURN_RESPONSIVENESS_MODE_RESPONSIVE,
            TURN_RESPONSIVENESS_MODE_RESPONSIVE,
            RetryExecutionClass.RESPONSIVE_FOREGROUND,
        ),
        (
            "invalid",
            TURN_RESPONSIVENESS_MODE_QUALITY,
            RetryExecutionClass.QUALITY_FOREGROUND,
        ),
    ],
)
def test_save_mode_selects_foreground_retry_execution_class(
    value: object,
    expected_mode: str,
    expected_execution_class: RetryExecutionClass,
) -> None:
    repositories = cast(PersistenceRepositories, FakeRepositories(value))

    assert turn_responsiveness_mode(repositories, save_id="save-1") == expected_mode
    assert (
        retry_execution_class_for_save(repositories, save_id="save-1")
        is expected_execution_class
    )
