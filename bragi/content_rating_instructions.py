"""Dependency-neutral content-rating ceiling instructions."""

from __future__ import annotations

CONTENT_RATING_G = "g"
CONTENT_RATING_PG = "pg"
CONTENT_RATING_PG_13 = "pg-13"
CONTENT_RATING_R = "r"
CONTENT_RATING_UNRATED = "unrated"
CONTENT_RATING_UNCLASSIFIED = "unclassified"
CONTENT_RATING_PROHIBITED = "prohibited"

_CONTENT_RATING_RANK = {
    CONTENT_RATING_G: 0,
    CONTENT_RATING_PG: 1,
    CONTENT_RATING_PG_13: 2,
    CONTENT_RATING_R: 3,
    CONTENT_RATING_UNRATED: 4,
    CONTENT_RATING_PROHIBITED: 5,
    CONTENT_RATING_UNCLASSIFIED: 5,
}

_CONTENT_RATING_LABELS = {
    CONTENT_RATING_G: "G — General audiences",
    CONTENT_RATING_PG: "PG — Parental guidance suggested",
    CONTENT_RATING_PG_13: "PG-13 — Parents strongly cautioned",
    CONTENT_RATING_R: "R — Restricted",
}
_CONTENT_RATING_THINK_GUIDANCE = {
    CONTENT_RATING_G: "Safe for a young child without parental explanation.",
    CONTENT_RATING_PG: "A child may need a parent to contextualize it.",
    CONTENT_RATING_PG_13: (
        "Teen-oriented mainstream content, but not explicit adult material."
    ),
    CONTENT_RATING_R: (
        "Adult content is allowed; explicit pornography, extreme exploitation, "
        "and actionable harm remain blocked."
    ),
}
_CONTENT_RATING_BLOCKED = {
    CONTENT_RATING_G: (
        "Any profanity beyond extremely mild exclamations",
        "Sexual dialogue, sexual situations, or sexualized nudity",
        "Romantic content beyond innocent affection",
        "Drug or alcohol use presented meaningfully",
        "Realistic violence, visible injury, blood, or suffering",
        "Threats that are intense, prolonged, or frightening",
        "Horror imagery, disturbing descriptions, or psychologically dark material",
        (
            "Detailed discussion of abuse, suicide, sexual assault, or other "
            "mature themes"
        ),
    ),
    CONTENT_RATING_PG: (
        "Strong profanity or slurs",
        "Explicit sexual references or sexual activity",
        "Sexualized nudity",
        "Graphic violence, gore, torture, or realistic injury detail",
        "Detailed drug use or intoxication",
        "Intense horror or sustained terror",
        (
            "Detailed descriptions of suicide, abuse, sexual assault, or severe "
            "cruelty"
        ),
        (
            "Mature themes presented with significant detail or emotional "
            "intensity"
        ),
    ),
    CONTENT_RATING_PG_13: (
        "Explicitly described sexual activity",
        "Pornographic or strongly erotic content",
        "Graphic nudity focused on genitals or sexual stimulation",
        "Graphic gore, mutilation, dismemberment, or torture",
        "Sexual violence described in detail",
        "Detailed instructions for drug use",
        "Repeated extreme profanity or highly aggressive sexual profanity",
        "Extremely disturbing horror or cruelty",
        "Detailed suicide or self-harm methods",
    ),
    CONTENT_RATING_R: (
        "Pornographic or explicitly erotic depictions intended primarily for arousal",
        "Graphic, prolonged descriptions of sexual acts",
        "Any sexual content involving minors",
        "Sexual exploitation, coercion, or assault presented erotically",
        (
            "Extreme graphic gore or sadistic torture described in lingering "
            "detail"
        ),
        "Content celebrating real-world atrocities or targeted abuse",
        "Detailed, actionable self-harm or suicide instructions",
        "Detailed instructions for manufacturing or using illegal hard drugs",
        "Anything independently prohibited by the platform's safety rules",
    ),
}
_CONTENT_RATING_ALLOWED = {
    CONTENT_RATING_G: (
        "Cartoonish slapstick",
        "Mild peril with quick resolution",
        "Innocent romance or kissing",
        "Nonsexual anatomical or medical references",
        "Gentle discussion of death or sadness",
        "Fantasy conflict without injury detail",
    ),
    CONTENT_RATING_PG: (
        "Mild insults and occasional mild profanity",
        "Brief nonsexual nudity",
        "Mild romantic or suggestive references",
        "Moderate fantasy/action violence",
        "Limited bloodless realistic violence",
        "Alcohol use by adults without glamorization",
        "References to drugs without showing or describing their use",
        "Some frightening scenes and mature themes handled gently",
    ),
    CONTENT_RATING_PG_13: (
        "Moderate profanity, including limited strong profanity",
        "Non-explicit sexual references and implied sex",
        "Brief or non-graphic nudity",
        "Strong action violence",
        "Some blood or injury detail, provided it is not graphic",
        "Drug and alcohol use",
        "Dark themes, death, abuse, crime, and mental-health struggles",
        (
            "Frightening or disturbing material that stops short of graphic "
            "detail"
        ),
    ),
    CONTENT_RATING_R: (
        "Frequent strong profanity",
        "Adult sexual dialogue",
        "Implied or non-graphically described sex",
        "Nudity, depending on whether it is sexualized",
        "Strong and bloody violence",
        "Drug use and addiction themes",
        "Intense horror",
        (
            "Mature treatment of trauma, abuse, suicide, crime, and sexual "
            "violence, provided it is not instructional, exploitative, or "
            "pornographic"
        ),
    ),
}
_CONTENT_RATING_SUMMARIES = {
    CONTENT_RATING_G: (
        "G — General audiences. Block any profanity beyond extremely mild "
        "exclamations; sexual situations or sexualized nudity; romance beyond "
        "innocent affection; meaningful drug or alcohol use; realistic violence, "
        "injury, blood, or suffering; intense threats or horror; and detailed "
        "mature themes. Allow child-safe slapstick, mild peril, innocent kissing, "
        "medical references, gentle sadness, and bloodless fantasy conflict."
    ),
    CONTENT_RATING_PG: (
        "PG — Parental guidance suggested. Block strong profanity or slurs; "
        "explicit sex or sexualized nudity; graphic violence, injury, or torture; "
        "detailed drug use; sustained terror; and detailed or emotionally intense "
        "mature themes. Allow mild language, brief nonsexual nudity, mild "
        "suggestiveness, moderate action, adult alcohol use, and gently handled "
        "frightening or mature material."
    ),
    CONTENT_RATING_PG_13: (
        "PG-13 — Parents strongly cautioned. Block explicitly described sexual "
        "activity, pornography, graphic sexualized nudity, graphic gore or "
        "torture, detailed sexual violence or self-harm methods, drug-use "
        "instructions, repeated extreme profanity, and extreme horror. Allow "
        "limited strong language, implied sex, non-graphic nudity or injuries, "
        "strong action, substance use, and dark teen-oriented themes."
    ),
    CONTENT_RATING_R: (
        "R — Restricted. Block pornographic or explicitly erotic depictions, "
        "prolonged graphic sex, all sexual content involving minors, eroticized "
        "coercion or assault, lingering extreme gore or sadistic torture, "
        "celebration of atrocities, actionable self-harm, and hard-drug "
        "instructions. Allow non-pornographic adult dialogue, implied sex, "
        "nudity, bloody violence, substance themes, intense horror, and mature "
        "treatment of trauma or crime."
    ),
}


def content_rating_ceiling_instructions(value: str) -> str:
    """Render the complete content ceiling for a supported rated setting."""

    rating = value.strip().casefold().replace("_", "-")
    if rating == "pg13":
        rating = CONTENT_RATING_PG_13
    if rating == CONTENT_RATING_UNRATED:
        return ""
    if rating not in _CONTENT_RATING_LABELS:
        rating = CONTENT_RATING_PG_13
    blocked = "\n".join(f"- {item}" for item in _CONTENT_RATING_BLOCKED[rating])
    allowed = "\n".join(f"- {item}" for item in _CONTENT_RATING_ALLOWED[rating])
    return (
        f"{_CONTENT_RATING_LABELS[rating]}\n\n"
        f"Block:\n{blocked}\n\n"
        f"Generally allow:\n{allowed}\n\n"
        f"Interpretation: {_CONTENT_RATING_THINK_GUIDANCE[rating]}"
    )


def content_rating_ceiling_summary(value: str) -> str:
    """Render a compact narrator-facing summary of a rated ceiling."""

    rating = value.strip().casefold().replace("_", "-")
    if rating == "pg13":
        rating = CONTENT_RATING_PG_13
    if rating == CONTENT_RATING_UNRATED:
        return ""
    return _CONTENT_RATING_SUMMARIES.get(
        rating,
        _CONTENT_RATING_SUMMARIES[CONTENT_RATING_PG_13],
    )


def content_rating_exceeds(*, minimum_rating: str, allowed_rating: str) -> bool:
    """Compare an agent-produced minimum rating with a viewer's ceiling."""

    allowed = _normalize_rating(allowed_rating)
    if allowed == CONTENT_RATING_UNRATED:
        return False
    minimum = _normalize_rating(minimum_rating, unknown=CONTENT_RATING_PROHIBITED)
    return _CONTENT_RATING_RANK[minimum] > _CONTENT_RATING_RANK[allowed]


def maximum_content_rating(
    values: list[str] | tuple[str, ...],
    *,
    default: str = CONTENT_RATING_UNCLASSIFIED,
) -> str:
    """Return the strictest normalized rating in a set of provenance values."""

    normalized = [
        _normalize_rating(value, unknown=CONTENT_RATING_PROHIBITED)
        for value in values
    ]
    if not normalized:
        return _normalize_rating(default, unknown=CONTENT_RATING_PROHIBITED)
    return max(normalized, key=_CONTENT_RATING_RANK.__getitem__)


def _normalize_rating(value: str, *, unknown: str = CONTENT_RATING_PG_13) -> str:
    rating = value.strip().casefold().replace("_", "-")
    if rating == "pg13":
        rating = CONTENT_RATING_PG_13
    if rating not in _CONTENT_RATING_RANK:
        return unknown
    return rating
