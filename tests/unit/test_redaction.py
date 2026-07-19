from __future__ import annotations

from bragi.redaction import redact_log_value, redact_text


def test_redact_text_removes_common_secret_shapes() -> None:
    text = (
        "token=abc123 api_key: sk-live-secret "
        "Authorization: Bearer bearer-secret"
    )

    redacted = redact_text(text)

    assert redacted is not None
    assert "abc123" not in redacted
    assert "sk-live-secret" not in redacted
    assert "bearer-secret" not in redacted
    assert "[redacted]" in redacted


def test_redact_log_value_recurses_and_respects_sensitive_keys() -> None:
    value = {
        "Authorization": "Bearer secret-token",
        "safe": ["visible", {"api-key": "nested-secret"}],
        "count": 3,
    }

    assert redact_log_value(value) == {
        "Authorization": "[redacted]",
        "safe": ["visible", {"api-key": "[redacted]"}],
        "count": 3,
    }


def test_redact_log_value_keeps_none_for_sensitive_empty_values() -> None:
    assert redact_log_value(None, key="token") is None
