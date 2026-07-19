from __future__ import annotations

from bragi.providers.errors import ProviderError, ProviderErrorCategory


def test_provider_retry_diagnostics_extracts_only_safe_retry_fields() -> None:
    from bragi.services.provider_diagnostics import retry_diagnostics_from_metadata

    diagnostics = retry_diagnostics_from_metadata(
        {
            "_bragi_retry": {
                "attempt_count": 2,
                "max_attempts": 3,
                "api_key": "sk-should-not-leak",
                "attempts": [
                    {
                        "attempt": 1,
                        "duration_ms": 17,
                        "error_category": "rate_limited",
                        "http_status": 429,
                        "body": "unsafe provider body",
                    },
                    {
                        "attempt": 2,
                        "duration_ms": 5,
                        "error_category": None,
                    },
                    {"attempt": "bad", "duration_ms": 1},
                ],
            }
        }
    )

    assert diagnostics == {
        "attempt_count": 2,
        "max_attempts": 3,
        "retry_attempts": [
            {
                "attempt": 1,
                "duration_ms": 17,
                "error_category": "rate_limited",
                "http_status": 429,
            },
            {
                "attempt": 2,
                "duration_ms": 5,
                "error_category": None,
            },
        ],
    }
    assert "sk-should-not-leak" not in repr(diagnostics)
    assert "unsafe provider body" not in repr(diagnostics)


def test_result_with_provider_diagnostics_promotes_single_call_retry_fields() -> None:
    from bragi.services.provider_diagnostics import result_with_provider_diagnostics

    result = result_with_provider_diagnostics(
        {"field_count": 2},
        provider_calls=(
            {
                "task": "scenario_generation",
                "provider": "openrouter",
                "model": "openrouter/scenario",
                "attempt_count": 2,
                "max_attempts": 3,
                "retry_attempts": [
                    {
                        "attempt": 1,
                        "duration_ms": 100,
                        "error_category": "network_error",
                    },
                    {
                        "attempt": 2,
                        "duration_ms": 50,
                        "error_category": None,
                    },
                ],
            },
        ),
    )

    assert result is not None
    assert result["field_count"] == 2
    assert result["attempt_count"] == 2
    assert result["max_attempts"] == 3
    assert result["provider_call_count"] == 1
    assert result["provider_calls"] == [
        {
            "task": "scenario_generation",
            "provider": "openrouter",
            "model": "openrouter/scenario",
            "attempt_count": 2,
            "max_attempts": 3,
            "retry_attempts": [
                {
                    "attempt": 1,
                    "duration_ms": 100,
                    "error_category": "network_error",
                },
                {
                    "attempt": 2,
                    "duration_ms": 50,
                    "error_category": None,
                },
            ],
        }
    ]


def test_provider_error_diagnostics_extracts_exhausted_retry_fields() -> None:
    from bragi.services.provider_diagnostics import retry_diagnostics_from_error

    error = ProviderError(
        ProviderErrorCategory.RATE_LIMITED,
        "slow down",
        status_code=429,
        retry_attempt_count=3,
        max_retry_attempts=3,
        retry_attempts=(
            {
                "attempt": 1,
                "duration_ms": 10,
                "error_category": "rate_limited",
                "http_status": 429,
                "raw": "unsafe",
            },
            {
                "attempt": 2,
                "duration_ms": 20,
                "error_category": "rate_limited",
                "http_status": 429,
            },
            {
                "attempt": 3,
                "duration_ms": 30,
                "error_category": "rate_limited",
                "http_status": 429,
            },
        ),
    )

    assert retry_diagnostics_from_error(error) == {
        "attempt_count": 3,
        "max_attempts": 3,
        "retry_attempts": [
            {
                "attempt": 1,
                "duration_ms": 10,
                "error_category": "rate_limited",
                "http_status": 429,
            },
            {
                "attempt": 2,
                "duration_ms": 20,
                "error_category": "rate_limited",
                "http_status": 429,
            },
            {
                "attempt": 3,
                "duration_ms": 30,
                "error_category": "rate_limited",
                "http_status": 429,
            },
        ],
    }


def test_safe_provider_error_diagnostics_filters_reasoning_content() -> None:
    from bragi.services.provider_diagnostics import safe_provider_error_diagnostics

    diagnostics = safe_provider_error_diagnostics(
        {
            "finish_reason": "length",
            "native_finish_reason": "length",
            "reasoning_tokens": 20,
            "reasoning_detail_types": ["reasoning.text", "", 7],
            "reasoning": "private thinking text",
            "reasoning_details": [{"type": "reasoning.text", "text": "private"}],
            "encrypted_content": "secret",
        }
    )

    assert diagnostics == {
        "finish_reason": "length",
        "native_finish_reason": "length",
        "reasoning_tokens": 20,
        "reasoning_detail_types": ["reasoning.text"],
    }
    assert "private thinking text" not in repr(diagnostics)
    assert "private" not in repr(diagnostics)
    assert "secret" not in repr(diagnostics)
