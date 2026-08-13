from dataclasses import dataclass

import pytest

from bragi.retry_policy import (
    DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS,
    DEFAULT_RETRY_COUNT,
    DEFERRED_WORK_MAX_ATTEMPTS,
    MAX_PROVIDER_CALL_DEADLINE_SECONDS,
    MAX_RETRY_COUNT,
    MIN_PROVIDER_CALL_DEADLINE_SECONDS,
    MIN_RETRY_COUNT,
    MODEL_OUTPUT_MAX_ATTEMPTS,
    PROVIDER_CALL_DEADLINE_SETTING,
    PROVIDER_MAX_ATTEMPTS,
    RetryExecutionClass,
    claim_narrator_regeneration,
    configured_max_attempts,
    configured_provider_call_deadline_seconds,
    configured_retry_count,
    narrator_regeneration_budget,
    resolved_retry_budget,
    retry_execution_context,
    retry_execution_context_is_explicit,
    sanitize_provider_call_deadline_seconds,
    sanitize_retry_count,
)


@dataclass
class RepositoryFake:
    value: object | None = None
    deadline_value: object | None = None

    def get_effective_setting(self, key: str) -> object | None:
        if key == "retry_count":
            return self.value
        if key == PROVIDER_CALL_DEADLINE_SETTING:
            return self.deadline_value
        return None


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


def test_provider_call_deadline_default_is_one_hundred_twenty_seconds() -> None:
    assert DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS == 120.0
    assert configured_provider_call_deadline_seconds() == 120.0


def test_provider_call_deadline_reads_repository_setting() -> None:
    repositories = RepositoryFake(deadline_value=45)

    assert configured_provider_call_deadline_seconds(repositories) == 45.0


def test_provider_call_deadline_falls_back_when_repository_lookup_fails() -> None:
    class BrokenRepository:
        def get_effective_setting(self, key: str) -> object:
            raise RuntimeError("database unavailable")

    assert (
        configured_provider_call_deadline_seconds(BrokenRepository())
        == DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS),
        ("45", DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS),
        (float("nan"), DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS),
        (float("inf"), DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS),
        (float("-inf"), DEFAULT_PROVIDER_CALL_DEADLINE_SECONDS),
        (1.0, MIN_PROVIDER_CALL_DEADLINE_SECONDS),
        (
            MAX_PROVIDER_CALL_DEADLINE_SECONDS + 30.0,
            MAX_PROVIDER_CALL_DEADLINE_SECONDS,
        ),
        (90.0, 90.0),
    ],
)
def test_provider_call_deadline_sanitizes_persisted_values(
    value: object, expected: float
) -> None:
    assert sanitize_provider_call_deadline_seconds(value) == expected


def test_responsive_foreground_retry_budget_caps_provider_and_turn_retries() -> None:
    repositories = RepositoryFake(value=9, deadline_value=120)

    with retry_execution_context(RetryExecutionClass.RESPONSIVE_FOREGROUND):
        budget = resolved_retry_budget(repositories)

    assert budget.provider_max_attempts == 2
    assert budget.provider_call_deadline_seconds == 45.0
    assert budget.automatic_turn_retry_allowed is False
    assert budget.verification_max_attempts == 2


@pytest.mark.parametrize(
    "execution_class",
    [
        RetryExecutionClass.QUALITY_FOREGROUND,
        RetryExecutionClass.BACKGROUND,
    ],
)
def test_quality_and_background_retry_budgets_preserve_configured_behavior(
    execution_class: RetryExecutionClass,
) -> None:
    repositories = RepositoryFake(value=3, deadline_value=90)

    with retry_execution_context(execution_class):
        budget = resolved_retry_budget(repositories)

    assert budget.provider_max_attempts == 4
    assert budget.provider_call_deadline_seconds == 90.0
    assert budget.automatic_turn_retry_allowed is True
    assert budget.verification_max_attempts == 4


def test_narrator_regeneration_budget_is_shared_across_response_checks() -> None:
    with narrator_regeneration_budget(max_regenerations=1):
        assert claim_narrator_regeneration() is True
        assert claim_narrator_regeneration() is False

    assert claim_narrator_regeneration() is True


def test_retry_execution_context_reports_only_explicit_scopes() -> None:
    assert retry_execution_context_is_explicit() is False

    with retry_execution_context(RetryExecutionClass.BACKGROUND):
        assert retry_execution_context_is_explicit() is True

    assert retry_execution_context_is_explicit() is False
