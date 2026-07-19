"""Deterministic dating-route pacing and escalation policy."""

from __future__ import annotations

from dataclasses import dataclass

ROUTE_STAGE_ORDER = (
    "unmet",
    "introduced",
    "initial_interest",
    "contact_exchanged",
    "first_date_planned",
    "first_date_in_progress",
    "early_dating",
    "exclusive",
    "committed",
)
ROUTE_STAGE_RANK = {stage: index for index, stage in enumerate(ROUTE_STAGE_ORDER)}


@dataclass(frozen=True)
class DatingRouteEscalationPolicy:
    stage: str
    max_plausible_escalation: str
    allowed_progress: tuple[str, ...]
    needs_explicit_support: tuple[str, ...]
    premature_escalations: tuple[str, ...]


CHARACTER_SPECIFIC_INTIMACY_GUIDANCE = (
    "physical and sexual intimacy follow the character-specific route profile, "
    "known boundaries, consent, and established characterization"
)


_POLICIES: dict[str, DatingRouteEscalationPolicy] = {
    "unmet": DatingRouteEscalationPolicy(
        stage="unmet",
        max_plausible_escalation=(
            "introduction, first impression, and cautious curiosity"
        ),
        allowed_progress=(
            "noticing each other",
            "first introduction",
            "guarded curiosity",
        ),
        needs_explicit_support=("contact exchange",),
        premature_escalations=(
            "first date planning",
            "exclusivity or commitment language",
            "future-locking or domestic planning",
        ),
    ),
    "introduced": DatingRouteEscalationPolicy(
        stage="introduced",
        max_plausible_escalation=(
            "warmth, curiosity, light flirtation, and contact exchange"
        ),
        allowed_progress=(
            "emotional warmth",
            "curiosity",
            "light flirtation",
            "contact exchange",
        ),
        needs_explicit_support=("first date planning", "guarded vulnerability"),
        premature_escalations=(
            "exclusivity or commitment language",
            "future-locking or domestic planning",
        ),
    ),
    "initial_interest": DatingRouteEscalationPolicy(
        stage="initial_interest",
        max_plausible_escalation=(
            "warmth, curiosity, light flirtation, and contact exchange"
        ),
        allowed_progress=(
            "emotional warmth",
            "curiosity",
            "light flirtation",
            "contact exchange",
        ),
        needs_explicit_support=("first date planning", "guarded vulnerability"),
        premature_escalations=(
            "exclusivity or commitment language",
            "future-locking or domestic planning",
        ),
    ),
    "contact_exchanged": DatingRouteEscalationPolicy(
        stage="contact_exchanged",
        max_plausible_escalation=(
            "follow-up interaction, first-date planning, and first-date start"
        ),
        allowed_progress=(
            "follow-up interaction",
            "first date planning",
            "first date start",
        ),
        needs_explicit_support=(
            "guarded vulnerability",
            "character-specific physical intimacy from route profile",
        ),
        premature_escalations=(
            "exclusivity or commitment language",
            "future-locking or domestic planning",
        ),
    ),
    "first_date_planned": DatingRouteEscalationPolicy(
        stage="first_date_planned",
        max_plausible_escalation=(
            "follow-up interaction, first-date planning, and first-date start"
        ),
        allowed_progress=(
            "follow-up interaction",
            "first date planning",
            "first date start",
        ),
        needs_explicit_support=(
            "guarded vulnerability",
            "character-specific physical intimacy from route profile",
        ),
        premature_escalations=(
            "exclusivity or commitment language",
            "future-locking or domestic planning",
        ),
    ),
    "first_date_in_progress": DatingRouteEscalationPolicy(
        stage="first_date_in_progress",
        max_plausible_escalation=(
            "first-date warmth and guarded vulnerability"
        ),
        allowed_progress=(
            "first-date warmth",
            "guarded vulnerability",
        ),
        needs_explicit_support=("planning another date", "deeper personal disclosure"),
        premature_escalations=(
            "exclusivity or commitment language",
            "future-locking or domestic planning",
        ),
    ),
    "early_dating": DatingRouteEscalationPolicy(
        stage="early_dating",
        max_plausible_escalation=(
            "deeper affection, trust building, and future dates"
        ),
        allowed_progress=(
            "deeper affection",
            "trust building",
            "future dates",
        ),
        needs_explicit_support=("exclusivity",),
        premature_escalations=(
            "commitment or life-plan locking",
            "future domestic planning",
        ),
    ),
    "exclusive": DatingRouteEscalationPolicy(
        stage="exclusive",
        max_plausible_escalation=(
            "exclusive relationship language and relationship reassurance"
        ),
        allowed_progress=(
            "exclusive relationship affection",
            "relationship reassurance",
        ),
        needs_explicit_support=("long-term commitment", "future domestic planning"),
        premature_escalations=(
            "major life-locking changes without explicit support",
        ),
    ),
    "committed": DatingRouteEscalationPolicy(
        stage="committed",
        max_plausible_escalation=(
            "established commitment and canon-consistent long-term planning"
        ),
        allowed_progress=(
            "established commitment",
            "canon-consistent long-term planning",
        ),
        needs_explicit_support=("major life changes",),
        premature_escalations=(
            "major life-locking changes without explicit support",
        ),
    ),
}

_NEXT_REASONABLE_STEP = {
    "unmet": "introduce the characters",
    "introduced": "build early interest or exchange contact info",
    "initial_interest": "exchange contact info or plan another interaction",
    "contact_exchanged": "schedule a first date or follow-up interaction",
    "first_date_planned": "begin the planned first date",
    "first_date_in_progress": "complete the first date without overcommitting",
    "early_dating": "build trust through additional dates",
    "exclusive": "deepen the established exclusive relationship",
    "committed": "honor the established commitment",
}


def escalation_policy_for_stage(stage: str) -> DatingRouteEscalationPolicy:
    normalized = stage.strip().casefold()
    if normalized in _POLICIES:
        return _POLICIES[normalized]
    return DatingRouteEscalationPolicy(
        stage=stage.strip() or "unknown",
        max_plausible_escalation=(
            "proportionate relationship progress grounded in explicit route state"
        ),
        allowed_progress=("proportionate relationship progress",),
        needs_explicit_support=("major escalation",),
        premature_escalations=("unsupported commitment or domestic escalation",),
    )


def intimacy_profile_guidance(
    *,
    comfort_with_intimacy: str = "",
    pacing_preference: str = "",
    known_boundaries: tuple[str, ...] | list[str] = (),
) -> str:
    parts = []
    if comfort_with_intimacy.strip():
        parts.append(f"comfort with intimacy: {comfort_with_intimacy.strip()}")
    if pacing_preference.strip():
        parts.append(f"pacing: {pacing_preference.strip()}")
    if known_boundaries:
        parts.append("known boundaries: " + "; ".join(known_boundaries))
    if parts:
        return (
            CHARACTER_SPECIFIC_INTIMACY_GUIDANCE
            + " ("
            + "; ".join(parts)
            + ")"
        )
    return (
        "physical and sexual intimacy are character-specific and not globally "
        "blocked by early route stage, but still require consent, established "
        "characterization, and any known boundaries; no route-specific intimacy "
        "profile is established yet"
    )


def next_reasonable_step(stage: str) -> str:
    return _NEXT_REASONABLE_STEP.get(
        stage.strip().casefold(),
        "build proportionate relationship progress",
    )
