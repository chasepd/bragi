"""Provider-safe chat message name helpers."""

from __future__ import annotations

import re

_PROVIDER_MESSAGE_NAME_MAX_CHARS = 64
_PROVIDER_MESSAGE_NAME_PART = re.compile(r"[A-Za-z0-9_-]+")


def provider_message_name(display_name: str | None) -> str | None:
    """Return an OpenAI-compatible message name, or None when none is safe."""
    if display_name is None:
        return None
    parts = _PROVIDER_MESSAGE_NAME_PART.findall(display_name.strip())
    if not parts:
        return None
    name = "_".join(parts)[:_PROVIDER_MESSAGE_NAME_MAX_CHARS]
    return name.strip("_-") or None
