from bragi.retry_policy import (
    DEFERRED_WORK_MAX_ATTEMPTS,
    MODEL_OUTPUT_MAX_ATTEMPTS,
    PROVIDER_MAX_ATTEMPTS,
)


def test_quality_first_retry_policy_uses_seven_total_attempts() -> None:
    assert PROVIDER_MAX_ATTEMPTS == 7
    assert MODEL_OUTPUT_MAX_ATTEMPTS == 7
    assert DEFERRED_WORK_MAX_ATTEMPTS == 7
