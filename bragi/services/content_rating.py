"""User-scoped content rating and safety policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bragi.persistence.repositories import PersistenceRepositories


CONTENT_FILTER_RATING_SETTING = "content_filter_rating"
FADE_TO_BLACK_ENABLED_SETTING = "fade_to_black_enabled"

CONTENT_RATING_G = "g"
CONTENT_RATING_PG = "pg"
CONTENT_RATING_PG_13 = "pg-13"
CONTENT_RATING_R = "r"
CONTENT_RATING_UNRATED = "unrated"
CONTENT_RATING_OPTIONS = (
    CONTENT_RATING_G,
    CONTENT_RATING_PG,
    CONTENT_RATING_PG_13,
    CONTENT_RATING_R,
    CONTENT_RATING_UNRATED,
)
CHILD_SELF_SERVICE_CONTENT_RATING_OPTIONS = (
    CONTENT_RATING_G,
    CONTENT_RATING_PG,
)
CHILD_ADMIN_CONTENT_RATING_OPTIONS = (
    *CHILD_SELF_SERVICE_CONTENT_RATING_OPTIONS,
    CONTENT_RATING_PG_13,
)

DEFAULT_ADULT_CONTENT_RATING = CONTENT_RATING_PG_13
DEFAULT_CHILD_CONTENT_RATING = CONTENT_RATING_PG
DEFAULT_FADE_TO_BLACK_ENABLED = True


@dataclass(frozen=True)
class ContentSafetyPolicy:
    """Effective user policy applied to generated prose and media prompts."""

    rating: str
    fade_to_black_enabled: bool
    force_venice_safe_mode: bool


def sanitize_content_rating(
    value: object,
    *,
    default: str = DEFAULT_ADULT_CONTENT_RATING,
) -> str:
    """Return a supported normalized rating or the supplied safe default."""

    normalized = str(value).strip().casefold().replace("_", "-") if value else ""
    if normalized == "pg13":
        normalized = CONTENT_RATING_PG_13
    if normalized in CONTENT_RATING_OPTIONS:
        return normalized
    return default


def effective_content_safety_policy(
    repositories: PersistenceRepositories,
    *,
    user_id: str | None,
) -> ContentSafetyPolicy:
    """Resolve defaults and non-bypassable child restrictions for a request actor."""

    user = repositories.get_user(user_id) if user_id is not None else None
    is_child = user is not None and user.role == "child"
    default_rating = (
        DEFAULT_CHILD_CONTENT_RATING if is_child else DEFAULT_ADULT_CONTENT_RATING
    )
    stored_rating = (
        repositories.get_scoped_setting(
            scope="user",
            scope_id=user.id,
            key=CONTENT_FILTER_RATING_SETTING,
        )
        if is_child and user is not None
        else repositories.get_effective_setting(
            CONTENT_FILTER_RATING_SETTING,
            user_id=user_id,
        )
    )
    rating = sanitize_content_rating(stored_rating, default=default_rating)
    if is_child and rating not in CHILD_ADMIN_CONTENT_RATING_OPTIONS:
        rating = default_rating

    stored_fade = repositories.get_effective_setting(
        FADE_TO_BLACK_ENABLED_SETTING,
        user_id=user_id,
    )
    fade_enabled = (
        stored_fade if isinstance(stored_fade, bool) else DEFAULT_FADE_TO_BLACK_ENABLED
    )
    if rating == CONTENT_RATING_UNRATED:
        fade_enabled = False
    return ContentSafetyPolicy(
        rating=rating,
        fade_to_black_enabled=True if is_child else fade_enabled,
        force_venice_safe_mode=is_child,
    )


def set_user_content_rating(
    repositories: PersistenceRepositories,
    *,
    user_id: str,
    rating: object,
    admin_grant: bool = False,
) -> str:
    """Persist a rating while enforcing child self-service and grant limits."""

    user = repositories.get_user(user_id)
    if user is None:
        raise ValueError(f"Unknown user id: {user_id}")
    normalized = sanitize_content_rating(rating, default="")
    if not normalized:
        raise ValueError("Unsupported content rating")
    if user.role == "child":
        if admin_grant and normalized not in CHILD_ADMIN_CONTENT_RATING_OPTIONS:
            raise ValueError("Child account content rating cannot exceed PG-13")
        if (
            not admin_grant
            and normalized not in CHILD_SELF_SERVICE_CONTENT_RATING_OPTIONS
        ):
            raise ValueError("Child accounts may select only G or PG")
    repositories.set_scoped_setting(
        scope="user",
        scope_id=user_id,
        key=CONTENT_FILTER_RATING_SETTING,
        value=normalized,
    )
    return normalized
