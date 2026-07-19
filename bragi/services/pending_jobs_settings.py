"""Settings for pending job tray presentation."""

from __future__ import annotations

PENDING_JOBS_DISPLAY_MODE_SETTING = "pending_jobs_display_mode"
DEFAULT_PENDING_JOBS_DISPLAY_MODE = "compact"
PENDING_JOBS_DISPLAY_MODE_OPTIONS = ("compact", "expanded", "expanded_full")


def sanitize_pending_jobs_display_mode(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in PENDING_JOBS_DISPLAY_MODE_OPTIONS:
            return normalized
    return DEFAULT_PENDING_JOBS_DISPLAY_MODE
