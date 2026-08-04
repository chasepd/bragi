from __future__ import annotations

from bragi.providers.reasoning_diagnostics import (
    extract_reasoning_signals,
    is_reasoning_only_chat_response,
    is_reasoning_truncated_structured_response,
)


def test_extract_signals_pulls_finish_reason_and_tokens() -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": None,
                    "reasoning_details": [{"type": "thinking"}],
                },
            }
        ],
        "usage": {
            "completion_tokens": 3,
            "completion_tokens_details": {"reasoning_tokens": 220},
        },
    }

    signals = extract_reasoning_signals(payload)

    assert signals.finish_reason == "length"
    assert signals.reasoning_tokens == 220
    assert signals.detail_types == ("thinking",)
    assert signals.completion_tokens == 3


def test_extract_signals_handles_missing_optional_fields() -> None:
    payload = {
        "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
    }

    signals = extract_reasoning_signals(payload)

    assert signals.finish_reason == "stop"
    assert signals.reasoning_tokens is None
    assert signals.detail_types == ()
    assert signals.completion_tokens is None


def test_extract_signals_reads_legacy_and_output_token_keys() -> None:
    payload = {
        "choices": [{"finish_reason": "length", "message": {"content": ""}}],
        "usage": {"output_tokens_details": {"reasoning_tokens": 11}},
    }

    signals = extract_reasoning_signals(payload)

    assert signals.reasoning_tokens == 11
    assert signals.completion_tokens is None


def test_is_reasoning_only_chat_response_detects_truncated_thinking() -> None:
    signals = extract_reasoning_signals(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": ""},
                }
            ],
            "usage": {
                "completion_tokens": 1,
                "completion_tokens_details": {"reasoning_tokens": 1500},
            },
        }
    )

    assert is_reasoning_only_chat_response(signals)


def test_is_reasoning_only_chat_response_ignores_non_length_responses() -> None:
    signals = extract_reasoning_signals(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": ""},
                }
            ],
            "usage": {
                "completion_tokens": 1,
                "completion_tokens_details": {"reasoning_tokens": 1500},
            },
        }
    )

    assert not is_reasoning_only_chat_response(signals)


def test_is_reasoning_only_chat_response_ignores_long_visible_responses() -> None:
    signals = extract_reasoning_signals(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "a long response"},
                }
            ],
            "usage": {
                "completion_tokens": 500,
                "completion_tokens_details": {"reasoning_tokens": 100},
            },
        }
    )

    assert not is_reasoning_only_chat_response(signals)


def test_is_reasoning_only_chat_response_uses_detail_types_when_tokens_missing() -> (
    None
):
    signals = extract_reasoning_signals(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_details": [{"type": "thinking"}],
                    },
                }
            ],
            "usage": {"completion_tokens": 0},
        }
    )

    assert is_reasoning_only_chat_response(signals)


def test_is_reasoning_truncated_structured_response_detects_empty_payload() -> None:
    signals = extract_reasoning_signals(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": ""},
                }
            ],
            "usage": {
                "completion_tokens": 0,
                "completion_tokens_details": {"reasoning_tokens": 2000},
            },
        }
    )

    assert is_reasoning_truncated_structured_response(signals)


def test_is_reasoning_truncated_structured_response_ignores_long_outputs() -> None:
    signals = extract_reasoning_signals(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "a" * 50},
                }
            ],
            "usage": {
                "completion_tokens": 200,
                "completion_tokens_details": {"reasoning_tokens": 2000},
            },
        }
    )

    assert not is_reasoning_truncated_structured_response(signals)


def test_is_reasoning_truncated_structured_response_ignores_non_length_responses() -> (
    None
):
    signals = extract_reasoning_signals(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": ""},
                }
            ],
            "usage": {
                "completion_tokens": 0,
                "completion_tokens_details": {"reasoning_tokens": 2000},
            },
        }
    )

    assert not is_reasoning_truncated_structured_response(signals)
