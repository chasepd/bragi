from dataclasses import dataclass

import pytest

from bragi.retry_policy import (
    DEFAULT_RETRY_COUNT,
    DEFERRED_WORK_MAX_ATTEMPTS,
    MAX_RETRY_COUNT,
    MIN_RETRY_COUNT,
    MODEL_OUTPUT_MAX_ATTEMPTS,
    PROVIDER_MAX_ATTEMPTS,
    configured_max_attempts,
    configured_retry_count,
    sanitize_retry_count,
)


@dataclass
class RepositoryFake:
    value: object | None = None

    def get_effective_setting(self, key: str) -> object | None:
        assert key == "retry_count"
        return self.value


def test_quality_first_retry_policy_uses_seven_total_attempts() -> None:
    assert PROVIDER_MAX_ATTEMPTS == 7
    assert MODEL_OUTPUT_MAX_ATTEMPTS == 7
    assert DEFERRED_WORK_MAX_ATTEMPTS == 7


def test_retry_policy_default_counts_six_retries_after_initial_attempt() -> None:
    assert configured_retry_count() == DEFAULT_RETRY_COUNT == 6
    assert configured_max_attempts() == 7


def test_retry_policy_reads_repository_setting_and_adds_initial_attempt() -> None:
    repositories = RepositoryFake(value=2)

    assert configured_retry_count(repositories) == 2
    assert configured_max_attempts(repositories) == 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, DEFAULT_RETRY_COUNT),
        ("4", DEFAULT_RETRY_COUNT),
        (-1, MIN_RETRY_COUNT),
        (MAX_RETRY_COUNT + 4, MAX_RETRY_COUNT),
    ],
)
def test_retry_policy_sanitizes_persisted_values(value: object, expected: int) -> None:
    assert sanitize_retry_count(value) == expected


def test_retry_policy_falls_back_when_repository_lookup_fails() -> None:
    class BrokenRepository:
        def get_effective_setting(self, key: str) -> object:
            raise RuntimeError("database unavailable")

    assert configured_retry_count(BrokenRepository()) == DEFAULT_RETRY_COUNT
