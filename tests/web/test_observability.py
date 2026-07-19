from __future__ import annotations

from bragi_web.observability import sanitize_fields


def test_sanitize_fields_drops_sensitive_key_variants() -> None:
    sanitized = sanitize_fields(
        {
            "component": "Composer",
            "access_token": "secret-access-token",
            "refresh-token": "secret-refresh-token",
            "session_cookie": "secret-session-cookie",
            "nested": {
                "Authorization": "Bearer secret-bearer",
                "safe": "kept",
                "userToken": "secret-user-token",
            },
            "items": [
                {
                    "api-key": "secret-api-key",
                    "label": "kept",
                },
            ],
        }
    )

    assert sanitized == {
        "component": "Composer",
        "nested": {"safe": "kept"},
        "items": [{"label": "kept"}],
    }
