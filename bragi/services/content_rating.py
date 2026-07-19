"""User-scoped content rating and safety policy helpers."""

from __future__ import annotations

import re
import unicodedata
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

_RATING_RANK = {
    CONTENT_RATING_G: 0,
    CONTENT_RATING_PG: 1,
    CONTENT_RATING_PG_13: 2,
    CONTENT_RATING_R: 3,
    CONTENT_RATING_UNRATED: 4,
}
_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_SPACED_LETTERS_RE = re.compile(r"(?<!\w)(?:[a-z]\s+){2,}[a-z](?!\w)")
_PG_PATTERNS = (
    re.compile(r"\bkiss(?:es|ed|ing)?\b"),
    re.compile(r"\bflirt(?:s|ed|ing|ation)?\b"),
    re.compile(r"\bmake(?:s|ing)?\s+out\b"),
    re.compile(r"\bromantic\s+(?:embrace|moment|relationship)\b"),
    re.compile(
        r"\b(?:fight|fights|fought|fighting|battle|battles|weapon|weapons|"
        r"sword|swords|gun|guns|rifle|rifles|pistol|pistols|knife|knives|"
        r"dagger|daggers|danger|death|dead)\b"
    ),
)
_PG_13_PATTERNS = (
    re.compile(r"\b(?:undress(?:es|ed|ing)?|disrob(?:e|es|ed|ing))\b"),
    re.compile(r"\b(?:lingerie|sexual intimacy|sex scene|slept together)\b"),
    re.compile(
        r"\b(?:hands?|fingers?)\s+(?:slipped|slid|moved)\s+"
        r"(?:under|beneath|inside)\b"
    ),
    re.compile(
        r"\b(?:touched|caressed|stroked)\s+(?:his|her|their|the)?\s*"
        r"(?:breasts?|groin|crotch|inner thighs?)\b"
    ),
    re.compile(
        r"\b(?:blood|bloody|stab|stabs|stabbed|stabbing|shoot|shoots|shot|"
        r"kill|kills|killed|killing|wound|wounds|wounded|assault|drunk|drugs?)\b"
    ),
    re.compile(r"\b(?:damn|hell)\b"),
)
_R_PATTERNS = (
    re.compile(
        r"\b(?:had|has|have|having|began|begin|begins|beginning|started|starts|"
        r"starting|engaged|engages|engaging)\s+(?:in\s+)?"
        r"(?:\w+\s+){0,3}sex\b"
    ),
    re.compile(r"\b(?:sex|sexual)\s+acts?\b"),
    re.compile(r"\b(?:sexual\s+)?intercourse\b"),
    re.compile(r"\bcopulat(?:e|es|ed|ing|ion)\b"),
    re.compile(r"\b(?:explicit|graphic|hardcore)\s+(?:sex|sexual content|nudity)\b"),
    re.compile(r"\bsexually\s+explicit\b"),
    re.compile(r"\b(?:penetrat(?:e|ed|es|ing|ion)|ejaculat(?:e|ed|es|ing|ion))\b"),
    re.compile(r"\b(?:masturbat(?:e|ed|es|ing|ion)|orgasm(?:s)?|climax(?:es|ed|ing)?)\b"),
    re.compile(r"\b(?:oral sex|anal sex|blow ?job|fellatio|cunnilingus)\b"),
    re.compile(
        r"\bcame\s+(?:(?:inside|on)\s+|(?:all\s+)?over\s+)"
        r"(?:him|her|them)\b"
    ),
    re.compile(
        r"\bforc(?:e|es|ed|ing)\s+"
        r"(?:himself|herself|themself|themselves)\s+"
        r"(?:inside|into)\s+(?:him|her|them)\b"
    ),
    re.compile(r"\b(?:nude|nudity|naked|porn|pornographic)\b"),
    re.compile(
        r"\b(?:topless|bottomless|braless|erotic|fetish|cleavage|"
        r"nipples?|areolas?|breasts?|boobs?|genitals?|penis|cock|dick|"
        r"vagina|vulva|pussy|"
        r"testicles?|buttocks?)\b"
    ),
    re.compile(
        r"\b(?:bare|exposed|uncovered)\s+"
        r"(?:breasts?|genitals?|penis|vagina|vulva|testicles?)\b"
    ),
    re.compile(
        r"\b(?:rape|rapes|raped|raping|rapist|sexual assault|sexual violence|"
        r"molest(?:s|ed|ing|ation)?|incest)\b"
    ),
    re.compile(r"\bthrust(?:s|ed|ing)?\s+(?:into|inside)\b"),
    re.compile(
        r"\b(?:dismember(?:s|ed|ing|ment)?|disembowel(?:s|ed|ing|ment)?|"
        r"decapitat(?:e|es|ed|ing|ion)|"
        r"eviscerat(?:e|es|ed|ing|ion)|mutilat(?:e|es|ed|ing|ion)|guts?|"
        r"entrails|graphic gore|graphic violence|severed (?:head|limb|body))\b"
    ),
    re.compile(
        r"\b(?:chop(?:s|ped|ping)?|hack(?:s|ed|ing)?|cut(?:s|ting)?)\s+off\s+"
        r"(?:\w+\s+){0,3}(?:heads?|limbs?)\b"
    ),
    re.compile(
        r"\b(?:crush(?:es|ed|ing)?|smash(?:es|ed|ing)?|cav(?:e|es|ed|ing))\s+"
        r"(?:\w+\s+){0,3}(?:skulls?|heads?)\b"
    ),
    re.compile(
        r"\b(?:brains?|blood|organs?|intestines?|viscera)\s+"
        r"(?:spill(?:s|ed|ing)?|splatter(?:s|ed|ing)?|spray(?:s|ed|ing)?|"
        r"spurt(?:s|ed|ing)?)\b"
    ),
    re.compile(
        r"\b(?:rip(?:s|ped|ping)?|tear(?:s|ing)?|tore|pull(?:s|ed|ing)?)\s+out\s+"
        r"(?:\w+\s+){0,3}(?:hearts?|organs?|intestines?)\b"
    ),
    re.compile(
        r"\b(?:rip(?:s|ped|ping)?|tear(?:s|ing)?|tore|hack(?:s|ed|ing)?|"
        r"saw(?:s|ed|ing)?)\s+(?:\w+\s+){0,3}"
        r"(?:arms?|legs?|hands?|feet|limbs?)\s+(?:off|apart)\b"
    ),
    re.compile(
        r"\b(?:blow(?:s|ing)?|blew|blast(?:s|ed|ing)?)\s+"
        r"(?:\w+\s+){0,3}(?:heads?|skulls?)\s+(?:off|apart)\b"
    ),
    re.compile(
        r"\bgoug(?:e|es|ed|ing)\s+out\s+(?:\w+\s+){0,3}eyes?\b"
    ),
    re.compile(
        r"\b(?:slit(?:s|ting)?|cut(?:s|ting)?)\s+(?:\w+\s+){0,3}throats?\b"
    ),
    re.compile(r"\b(?:behead(?:s|ed|ing)?|flay(?:s|ed|ing)?)\b"),
    re.compile(
        r"\b(?:torture|tortures|tortured|torturing|murder|murders|murdered|"
        r"murdering|suicide|suicidal|self harm|self-harm|strangl(?:e|es|ed|ing))\b"
    ),
    re.compile(r"\b(?:cocaine|heroin|meth|methamphetamine|fentanyl)\b"),
    re.compile(r"\b(?:fuck|fucks|fucked|fucking|motherfucker|cunt)\b"),
)


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


def classify_content_rating(value: str) -> str:
    """Return the minimum supported rating for deterministic text patterns."""

    normalized = _normalize_for_matching(value)
    if any(pattern.search(normalized) for pattern in _R_PATTERNS):
        return CONTENT_RATING_R
    if any(pattern.search(normalized) for pattern in _PG_13_PATTERNS):
        return CONTENT_RATING_PG_13
    if any(pattern.search(normalized) for pattern in _PG_PATTERNS):
        return CONTENT_RATING_PG
    return CONTENT_RATING_G


def content_exceeds_rating(value: str, *, allowed_rating: str) -> bool:
    """Return whether deterministic content requires a higher rating."""

    allowed = sanitize_content_rating(allowed_rating)
    if allowed == CONTENT_RATING_UNRATED:
        return False
    return _RATING_RANK[classify_content_rating(value)] > _RATING_RANK[allowed]


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


def _normalize_for_matching(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _WHITESPACE_RE.sub(" ", _NON_WORD_RE.sub(" ", normalized)).strip()
    return _SPACED_LETTERS_RE.sub(
        lambda match: match.group(0).replace(" ", ""),
        normalized,
    )
