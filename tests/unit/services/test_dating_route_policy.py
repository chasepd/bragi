from __future__ import annotations

from bragi.services.dating_route_policy import (
    ROUTE_STAGE_RANK,
    escalation_policy_for_stage,
    intimacy_profile_guidance,
    next_reasonable_step,
)


def test_introduced_route_allows_warmth_but_blocks_commitment() -> None:
    policy = escalation_policy_for_stage("introduced")

    assert policy.max_plausible_escalation == (
        "warmth, curiosity, light flirtation, and contact exchange"
    )
    assert "contact exchange" in policy.allowed_progress
    assert "first date planning" in policy.needs_explicit_support
    assert "exclusivity or commitment language" in policy.premature_escalations
    assert "future-locking or domestic planning" in policy.premature_escalations
    assert "sexual escalation" not in policy.premature_escalations
    assert next_reasonable_step("introduced") == (
        "build early interest or exchange contact info"
    )


def test_first_date_route_does_not_globally_block_intimacy() -> None:
    policy = escalation_policy_for_stage("first_date_in_progress")

    assert policy.max_plausible_escalation == (
        "first-date warmth and guarded vulnerability"
    )
    assert "limited consensual closeness" not in policy.allowed_progress
    assert "sexual escalation" not in policy.premature_escalations
    assert "physical closeness" not in policy.premature_escalations
    assert "exclusivity or commitment language" in policy.premature_escalations


def test_early_dating_route_leaves_physical_pacing_to_profile() -> None:
    policy = escalation_policy_for_stage("early_dating")

    assert policy.max_plausible_escalation == (
        "deeper affection, trust building, and future dates"
    )
    assert "physical closeness" not in policy.allowed_progress
    assert "future dates" in policy.allowed_progress


def test_later_route_allows_exclusivity_without_life_plan_locking() -> None:
    policy = escalation_policy_for_stage("exclusive")

    assert policy.max_plausible_escalation == (
        "exclusive relationship language and relationship reassurance"
    )
    assert "exclusive relationship affection" in policy.allowed_progress
    assert "intimacy consistent with known boundaries" not in policy.allowed_progress
    assert "long-term commitment" in policy.needs_explicit_support
    assert (
        "major life-locking changes without explicit support"
        in policy.premature_escalations
    )
    assert ROUTE_STAGE_RANK["exclusive"] > ROUTE_STAGE_RANK["early_dating"]


def test_intimacy_profile_fallback_keeps_consent_and_characterization_guard() -> None:
    guidance = intimacy_profile_guidance()

    assert "consent" in guidance
    assert "established characterization" in guidance
    assert "known boundaries" in guidance


def test_unknown_route_stage_falls_back_to_proportionate_progress() -> None:
    policy = escalation_policy_for_stage("unexpected")

    assert policy.stage == "unexpected"
    assert policy.max_plausible_escalation == (
        "proportionate relationship progress grounded in explicit route state"
    )
    assert next_reasonable_step("unexpected") == (
        "build proportionate relationship progress"
    )
