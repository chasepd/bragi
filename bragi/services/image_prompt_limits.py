"""Image prompt capacities from endpoints and cached provider model listings."""

from __future__ import annotations

from typing import Any

from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.venice import (
    VENICE_IMAGE_EDIT_PROMPT_MAX_CHARS,
    VENICE_IMAGE_PROMPT_MAX_CHARS,
    venice_image_prompt_model_limits,
)

# Derived catalog cache, refreshed with models; not a user preference or save data.
IMAGE_PROMPT_MODEL_LIMITS_SETTING = "image_prompt_model_limits"


def image_prompt_max_chars(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
    has_references: bool = False,
) -> int | None:
    """Return the smallest known character cap for the actual image request."""
    limits: list[int] = []
    if provider == "venice":
        limits.append(
            VENICE_IMAGE_EDIT_PROMPT_MAX_CHARS
            if has_references
            else VENICE_IMAGE_PROMPT_MAX_CHARS
        )
    cached = repositories.get_app_setting(IMAGE_PROMPT_MODEL_LIMITS_SETTING)
    provider_limits = cached.get(provider) if isinstance(cached, dict) else None
    if isinstance(provider_limits, dict):
        model_limit = _positive_limit(provider_limits.get(model_id))
        if model_limit is not None:
            limits.append(model_limit)
    return min(limits) if limits else None


def refresh_image_prompt_model_limits(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    raw_metadata: dict[str, Any],
) -> None:
    """Replace one provider's derived limits after a successful model refresh."""
    provider_limits = (
        venice_image_prompt_model_limits(raw_metadata)
        if provider == "venice"
        else {}
    )
    records = raw_metadata.get("data") or raw_metadata.get("models") or []
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            model_id = record.get("id")
            limit = _positive_limit(record.get("prompt_character_limit"))
            if isinstance(model_id, str) and limit is not None:
                provider_limits[model_id] = min(
                    limit, provider_limits.get(model_id, limit)
                )
    cached = repositories.get_app_setting(IMAGE_PROMPT_MODEL_LIMITS_SETTING)
    limits = dict(cached) if isinstance(cached, dict) else {}
    limits[provider] = provider_limits
    repositories.set_app_setting(IMAGE_PROMPT_MODEL_LIMITS_SETTING, limits)


def _positive_limit(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None
