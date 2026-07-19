from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.services.dating_route_profile_service import (
    DATING_ROUTE_PROFILE_TASK,
    DatingRouteProfileService,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


class ProfileProvider:
    provider_name = "fake"

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def validate_config(self) -> ProviderConfigStatus:
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
        )

    async def list_models(self) -> list[ProviderModel]:
        return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("dating route profiling must not call chat")

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("dating route profiling must not generate images")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data=self.response,
            provider=request.provider,
            model_id=request.model_id,
        )


def test_profile_service_applies_generated_character_route_profile(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    provider = ProfileProvider(
        {
            "profiles": [
                {
                    "npc_character_id": npc_id,
                    "comfort_with_intimacy": (
                        "open to physical and sexual intimacy early when "
                        "chemistry and consent are clear"
                    ),
                    "pacing_preference": "direct and chemistry-led",
                    "known_boundaries": [
                        "no public pressure",
                        "does not require a fixed number of dates first",
                    ],
                    "unresolved_questions": ["whether Ren can keep things discreet"],
                    "reason": "The character is direct and dislikes arbitrary rules.",
                    "confidence": 0.82,
                }
            ]
        }
    )
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "succeeded"
    assert result.updated_count == 1
    assert updated.comfort_with_intimacy.startswith("open to physical")
    assert updated.pacing_preference == "direct and chemistry-led"
    assert updated.known_boundaries == [
        "no public pressure",
        "does not require a fixed number of dates first",
    ]
    assert updated.unresolved_questions == [
        "whether Ren can keep things discreet"
    ]
    request = provider.structured_output_requests[0]
    assert request.schema_name == "dating_route_profile"
    assert request.schema["properties"]["profiles"]["minItems"] == 1
    assert request.schema["properties"]["profiles"]["maxItems"] == 1
    assert request.schema["properties"]["profiles"]["items"]["properties"][
        "npc_character_id"
    ]["enum"] == [npc_id]


def test_profile_service_preserves_explicit_route_profile_fields(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    repositories.upsert_dating_route_state(
        save_id=save_id,
        player_character_id=route.player_character_id,
        npc_character_id=route.npc_character_id,
        stage=route.stage,
        comfort_with_intimacy="does not kiss until trust is established",
        known_boundaries=["no public pressure"],
        route_id=route.id,
    )
    provider = ProfileProvider(
        {
            "profiles": [
                {
                    "npc_character_id": npc_id,
                    "comfort_with_intimacy": "comfortable with casual intimacy",
                    "pacing_preference": "fast if interested",
                    "known_boundaries": [
                        "no public pressure",
                        "needs private consent checks",
                    ],
                    "unresolved_questions": [],
                    "reason": "Generated fallback.",
                    "confidence": 0.8,
                }
            ]
        }
    )
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.updated_count == 1
    assert updated.comfort_with_intimacy == (
        "does not kiss until trust is established"
    )
    assert updated.pacing_preference == "fast if interested"
    assert updated.known_boundaries == [
        "no public pressure",
        "needs private consent checks",
    ]


def test_profile_service_rejects_out_of_scope_generated_preserved_field(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    repositories.upsert_dating_route_state(
        save_id=save_id,
        player_character_id=route.player_character_id,
        npc_character_id=route.npc_character_id,
        stage=route.stage,
        comfort_with_intimacy="does not kiss until trust is established",
        route_id=route.id,
    )
    provider = ProfileProvider(
        {
            "profiles": [
                {
                    **_generated_profile_item(npc_id),
                    "comfort_with_intimacy": (
                        "wants marriage by the end of the first date"
                    ),
                }
            ]
        }
    )
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "out_of_scope_profile_response"
    assert result.updated_count == 0
    assert updated.comfort_with_intimacy == (
        "does not kiss until trust is established"
    )
    assert updated.pacing_preference == ""


def test_profile_service_seeds_missing_route_before_generating_profiles(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, first_npc_id = _dating_route_save(repositories)
    second_npc = repositories.add_character(
        save_id=save_id,
        name="Yui Sato",
        personality="Careful, observant, slow to trust.",
        motivations="Wants consistency before emotional risk.",
        boundaries="no rushed public affection",
        relationships={"Ren Takahashi": "romance option for Ren Takahashi"},
        status="available romance option at scenario start",
        met=True,
    )
    assert {
        route.npc_character_id
        for route in repositories.list_dating_route_states(save_id)
    } == {first_npc_id}
    provider = ProfileProvider(
        {
            "profiles": [
                _generated_profile_item(first_npc_id),
                _generated_profile_item(second_npc.id),
            ]
        }
    )
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    routes = repositories.list_dating_route_states(save_id)
    assert result.status == "succeeded"
    assert result.updated_count == 2
    assert result.requested_count == 2
    assert {route.npc_character_id for route in routes} == {
        first_npc_id,
        second_npc.id,
    }
    assert all(route.comfort_with_intimacy for route in routes)
    assert all(route.pacing_preference for route in routes)
    request = provider.structured_output_requests[0]
    assert request.schema["properties"]["profiles"]["minItems"] == 2
    assert request.schema["properties"]["profiles"]["maxItems"] == 2
    assert set(
        request.schema["properties"]["profiles"]["items"]["properties"][
            "npc_character_id"
        ]["enum"]
    ) == {first_npc_id, second_npc.id}


def test_profile_service_skips_without_structured_model_preference(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, _npc_id = _dating_route_save(repositories)
    provider = ProfileProvider({"profiles": []})

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    assert result.status == "skipped"
    assert result.skipped_reason == "no_model_preference"
    assert provider.structured_output_requests == []


def test_profile_service_ignores_profile_for_out_of_scope_character(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, _npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    provider = ProfileProvider(
        {
            "profiles": [
                {
                    "npc_character_id": "other-character",
                    "comfort_with_intimacy": "wrong character profile",
                    "pacing_preference": "wrong character pacing",
                    "known_boundaries": ["wrong character boundary"],
                    "unresolved_questions": [],
                    "reason": "Wrong target.",
                    "confidence": 0.6,
                }
            ]
        }
    )
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "out_of_scope_profile_response"
    assert result.updated_count == 0
    assert result.requested_count == 1
    assert updated.comfort_with_intimacy == ""
    assert updated.pacing_preference == ""
    assert updated.known_boundaries == []


def test_profile_service_skips_incomplete_profile_response(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, _npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    provider = ProfileProvider({"profiles": []})
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "incomplete_profile_response"
    assert result.updated_count == 0
    assert result.requested_count == 1
    assert updated.comfort_with_intimacy == ""
    assert updated.pacing_preference == ""


def test_profile_service_skips_profile_response_with_non_object_item(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    provider = ProfileProvider(
        {
            "profiles": [
                _generated_profile_item(npc_id),
                "invalid profile item",
            ]
        }
    )
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "incomplete_profile_response"
    assert result.updated_count == 0
    assert result.requested_count == 1
    assert updated.comfort_with_intimacy == ""
    assert updated.pacing_preference == ""
    assert updated.known_boundaries == []


def test_profile_service_skips_profile_missing_required_schema_field(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    profile = _generated_profile_item(npc_id)
    del profile["known_boundaries"]
    provider = ProfileProvider({"profiles": [profile]})
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "incomplete_profile_response"
    assert result.updated_count == 0
    assert result.requested_count == 1
    assert updated.comfort_with_intimacy == ""
    assert updated.pacing_preference == ""
    assert updated.known_boundaries == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("known_boundaries", "not a list"),
        ("unresolved_questions", ["valid question", 3]),
        ("reason", None),
        ("confidence", "high"),
    ),
)
def test_profile_service_skips_profile_with_wrong_required_field_type(
    repositories: PersistenceRepositories,
    field: str,
    value: object,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    profile = _generated_profile_item(npc_id)
    profile[field] = value
    provider = ProfileProvider({"profiles": [profile]})
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "incomplete_profile_response"
    assert result.updated_count == 0
    assert result.requested_count == 1
    assert updated.comfort_with_intimacy == ""
    assert updated.pacing_preference == ""
    assert updated.known_boundaries == []


@pytest.mark.parametrize(
    "case",
    (
        "top_level_extra",
        "profile_extra",
        "too_many_boundaries",
        "too_many_questions",
    ),
)
def test_profile_service_skips_response_that_violates_declared_schema_shape(
    repositories: PersistenceRepositories,
    case: str,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    profile = _generated_profile_item(npc_id)
    if case == "top_level_extra":
        response: dict[str, object] = {
            "profiles": [profile],
            "extra": "unexpected",
        }
    elif case == "profile_extra":
        profile["extra"] = "unexpected"
        response = {"profiles": [profile]}
    elif case == "too_many_boundaries":
        profile["known_boundaries"] = [
            f"boundary {index}" for index in range(9)
        ]
        response = {"profiles": [profile]}
    else:
        profile["unresolved_questions"] = [
            f"question {index}" for index in range(7)
        ]
        response = {"profiles": [profile]}
    provider = ProfileProvider(response)
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "incomplete_profile_response"
    assert result.updated_count == 0
    assert result.requested_count == 1
    assert updated.comfort_with_intimacy == ""
    assert updated.pacing_preference == ""
    assert updated.known_boundaries == []


def test_profile_service_skips_duplicate_profile_response(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    provider = ProfileProvider(
        {
            "profiles": [
                {
                    "npc_character_id": npc_id,
                    "comfort_with_intimacy": "comfortable with direct chemistry",
                    "pacing_preference": "fast when attraction is mutual",
                    "known_boundaries": [],
                    "unresolved_questions": [],
                    "reason": "First response.",
                    "confidence": 0.7,
                },
                {
                    "npc_character_id": npc_id,
                    "comfort_with_intimacy": "conflicting duplicate",
                    "pacing_preference": "slow",
                    "known_boundaries": ["conflicting boundary"],
                    "unresolved_questions": [],
                    "reason": "Duplicate response.",
                    "confidence": 0.7,
                },
            ]
        }
    )
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "duplicate_profile_response"
    assert result.updated_count == 0
    assert result.requested_count == 1
    assert updated.comfort_with_intimacy == ""
    assert updated.pacing_preference == ""
    assert updated.known_boundaries == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "comfort_with_intimacy",
            "wants marriage by the end of the first date",
        ),
        ("pacing_preference", "plans living together soon"),
        ("known_boundaries", ["asks for major life plans right away"]),
        ("unresolved_questions", ["whether Ren wants immediate exclusivity"]),
        ("pacing_preference", "exclusivity after three dates"),
        ("known_boundaries", ["living together by winter"]),
        ("unresolved_questions", ["marriage timeline after graduation"]),
        ("pacing_preference", "wants monogamy soon"),
        ("known_boundaries", ["wants to be his girlfriend soon"]),
        ("unresolved_questions", ["wants a serious relationship now"]),
        ("pacing_preference", "wants a long-term relationship soon"),
        ("reason", "wants a long term relationship soon"),
        ("pacing_preference", "expects partner status within a month"),
        (
            "known_boundaries",
            ["prefers exclusivity before more emotional vulnerability"],
        ),
        ("pacing_preference", "desires monogamy"),
        ("known_boundaries", ["needs commitment"]),
        ("unresolved_questions", ["is looking for marriage"]),
        ("comfort_with_intimacy", "wants marriage before sex"),
        ("pacing_preference", "wants to be his wife soon"),
        ("known_boundaries", ["wants to be her husband soon"]),
        ("known_boundaries", ["plans a wedding soon"]),
        ("unresolved_questions", ["expects spouse status within a month"]),
        ("pacing_preference", "hopes to be her fiance soon"),
        ("known_boundaries", ["wants to be his fianc\u00e9e soon"]),
        ("reason", "wants marriage soon"),
        ("pacing_preference", "wants to make things official soon"),
        ("reason", "wants to make it official soon"),
        ("known_boundaries", ["expects relationship status within a month"]),
        ("unresolved_questions", ["whether Ren wants to label things"]),
        ("pacing_preference", "wants a baby soon"),
        ("known_boundaries", ["wants to share an apartment soon"]),
        ("unresolved_questions", ["wants to be roommates soon"]),
        ("reason", "expects a joint lease within a month"),
        ("pacing_preference", "plans to have a family by winter"),
        ("known_boundaries", ["hopes for parenthood after graduation"]),
    ),
)
def test_profile_service_skips_commitment_scope_profile_content(
    repositories: PersistenceRepositories,
    field: str,
    value: object,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    profile = _generated_profile_item(npc_id)
    profile[field] = value
    provider = ProfileProvider({"profiles": [profile]})
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "skipped"
    assert result.skipped_reason == "out_of_scope_profile_response"
    assert result.updated_count == 0
    assert result.requested_count == 1
    assert updated.comfort_with_intimacy == ""
    assert updated.pacing_preference == ""
    assert updated.known_boundaries == []


def test_profile_service_allows_intimacy_boundary_referencing_later_commitment(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    provider = ProfileProvider(
        {
            "profiles": [
                {
                    **_generated_profile_item(npc_id),
                    "comfort_with_intimacy": "does not kiss until marriage",
                    "known_boundaries": [
                        "sexual intimacy only in a committed relationship",
                        "sexual intimacy only after commitment",
                        "sexual intimacy requires commitment",
                    ],
                }
            ]
        }
    )
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "succeeded"
    assert result.updated_count == 1
    assert updated.comfort_with_intimacy == "does not kiss until marriage"
    assert updated.known_boundaries == [
        "sexual intimacy only in a committed relationship",
        "sexual intimacy only after commitment",
        "sexual intimacy requires commitment",
    ]


def test_profile_service_allows_benign_commitment_substrings(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_route_save(repositories)
    route = repositories.list_dating_route_states(save_id)[0]
    provider = ProfileProvider(
        {
            "profiles": [
                {
                    **_generated_profile_item(npc_id),
                    "reason": (
                        "They are childhood friends discussing the official "
                        "school festival schedule."
                    ),
                }
            ]
        }
    )
    _configure_profile_model(repositories)

    result = asyncio.run(
        DatingRouteProfileService(
            repositories=repositories,
            providers={"fake": provider},
        ).ensure_profiles_for_save(save_id=save_id)
    )

    updated = repositories.get_dating_route_state(route.id)
    assert updated is not None
    assert result.status == "succeeded"
    assert result.updated_count == 1
    assert updated.comfort_with_intimacy == "comfortable with direct chemistry"
    assert updated.pacing_preference == "fast when attraction is mutual"


def _dating_route_save(
    repositories: PersistenceRepositories,
) -> tuple[str, str, str]:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={"player_character_name": "Ren Takahashi"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Ren Takahashi",
        is_player_character=True,
        met=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        personality="Direct, playful, private about vulnerability.",
        motivations="Wants chemistry without being boxed into a timeline.",
        boundaries="no public pressure",
        relationships={player.name: "romance option for Ren Takahashi"},
        status="available romance option at scenario start",
        met=True,
    )
    repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="introduced",
        completed_interactions=0,
        dates_completed=0,
        next_reasonable_step="build early interest or exchange contact info",
    )
    return save.id, player.id, npc.id


def _generated_profile_item(npc_id: str) -> dict[str, object]:
    return {
        "npc_character_id": npc_id,
        "comfort_with_intimacy": "comfortable with direct chemistry",
        "pacing_preference": "fast when attraction is mutual",
        "known_boundaries": [],
        "unresolved_questions": [],
        "reason": "Generated fallback.",
        "confidence": 0.7,
    }


def _configure_profile_model(repositories: PersistenceRepositories) -> None:
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-profile",
        display_name="Fake Profile",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    repositories.set_model_preference(
        task=DATING_ROUTE_PROFILE_TASK,
        provider="fake",
        model_id="fake-profile",
    )
