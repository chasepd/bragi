"""User-scoped narrator guidance settings."""

from __future__ import annotations

USER_NARRATION_GUIDANCE_SETTING = "user_narration_guidance"
DEFAULT_USER_NARRATION_GUIDANCE = ""


def sanitize_user_narration_guidance(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return DEFAULT_USER_NARRATION_GUIDANCE
