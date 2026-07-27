"""Structured generation of action choices."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from bragi.persistence.models import (
    CharacterRecord,
    MessageActionChoiceRecord,
    MessageRecord,
    ModelPreferenceRecord,
    SaveDetailsRecord,
    SceneSnapshotRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ChatRequest,
    ProviderClient,
    ProviderRetryProgressCallback,
    StructuredOutputProvider,
    StructuredOutputRequest,
)
from bragi.services.action_choice_flags import scenario_action_choices_enabled
from bragi.services.character_action_planning_service import (
    CHARACTER_ACTION_PLANNING_TASK,
)
from bragi.services.content_rating import effective_content_safety_policy
from bragi.services.content_safety_service import ContentSafetyService
from bragi.services.model_capabilities import (
    MODEL_LACKS_CAPABILITY_REASON,
    MODEL_MISSING_REASON,
    MODEL_UNAVAILABLE_REASON,
    STRUCTURED_OUTPUT_CAPABILITIES,
    check_model_capabilities,
)
from bragi.services.model_preferences import (
    ACTION_CHOICE_GENERATION_PURPOSE,
    roleplay_model_preference_with_fallbacks,
)
from bragi.services.provider_fallbacks import structured_output_with_fallback
from bragi.world_time_model import format_world_time_from_snapshot

ACTION_CHOICE_GENERATION_TASK = ACTION_CHOICE_GENERATION_PURPOSE
ACTION_CHOICE_COUNT = 4


@dataclass(frozen=True)
class PreparedActionChoiceGeneration:
    save_id: str
    narrator_message_id: str
    request: StructuredOutputRequest
    current_user_id: str | None = None


class ActionChoiceService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
        content_safety_service: ContentSafetyService | None = None,
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.content_safety_service = (
            content_safety_service
            or ContentSafetyService(
                repositories=repositories,
                providers=providers,
            )
        )

    async def generate_for_message(
        self,
        *,
        save_id: str,
        narrator_message_id: str,
        save_details: SaveDetailsRecord | None = None,
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> list[MessageActionChoiceRecord]:
        prepared = self.prepare_for_message(
            save_id=save_id,
            narrator_message_id=narrator_message_id,
            save_details=save_details,
            current_user_id=current_user_id,
            retry_progress_callback=retry_progress_callback,
        )
        if prepared is None:
            return []
        return await self.generate_prepared(prepared)

    def prepare_for_message(
        self,
        *,
        save_id: str,
        narrator_message_id: str,
        save_details: SaveDetailsRecord | None = None,
        current_user_id: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> PreparedActionChoiceGeneration | None:
        details = save_details or self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        if not scenario_action_choices_enabled(details.scenario):
            return None
        narrator_message = next(
            (
                message
                for message in details.messages
                if message.id == narrator_message_id
                and message.role == "narrator"
                and message.deleted_at is None
            ),
            None,
        )
        if narrator_message is None:
            raise ValueError(
                f"Unknown active narrator message id: {narrator_message_id}"
            )
        preference = _action_choice_model_preference(
            repositories=self.repositories,
            save_id=save_id,
        )
        if preference is None:
            raise ValueError(
                "No action choice generation model preference configured"
            )
        provider = self.providers.get(preference.provider)
        if not isinstance(cast(object, provider), StructuredOutputProvider):
            raise ValueError(
                "Action choice generation provider does not support structured output"
            )
        requirement_error = _model_requirement_error(
            repositories=self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
        )
        if requirement_error is not None:
            raise ValueError(requirement_error)
        request = StructuredOutputRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            schema_name="action_choices",
            schema=_action_choice_schema(),
            messages=_action_choice_messages(
                scenario_type=details.scenario.type,
                scenario_title=details.scenario.title,
                scenario_premise=details.scenario.premise,
                player_role=details.scenario.player_role,
                scenario_content=_scenario_content(details.scenario.content_json),
                player_character=_player_character(
                    self.repositories,
                    save_id=save_id,
                ),
                scene_snapshot=self.repositories.get_scene_snapshot(save_id),
                present_characters=_present_characters(
                    self.repositories,
                    save_id=save_id,
                ),
                messages=tuple(details.messages),
                narrator_message=narrator_message,
            ),
            temperature=0.45,
            max_output_tokens=600,
            retry_progress_callback=retry_progress_callback,
        )
        return PreparedActionChoiceGeneration(
            save_id=save_id,
            narrator_message_id=narrator_message_id,
            request=request,
            current_user_id=current_user_id,
        )

    async def generate_prepared(
        self,
        prepared: PreparedActionChoiceGeneration,
    ) -> list[MessageActionChoiceRecord]:
        provider = self.providers.get(prepared.request.provider)
        if not isinstance(cast(object, provider), StructuredOutputProvider):
            raise ValueError(
                "Action choice generation provider does not support structured output"
            )
        requirement_error = _model_requirement_error(
            repositories=self.repositories,
            provider=prepared.request.provider,
            model_id=prepared.request.model_id,
        )
        if requirement_error is not None:
            raise ValueError(requirement_error)
        response = await structured_output_with_fallback(
            repositories=self.repositories,
            providers=self.providers,
            request=prepared.request,
            task=ACTION_CHOICE_GENERATION_TASK,
            save_id=prepared.save_id,
            diagnostic_context={
                "narrator_message_id": prepared.narrator_message_id
            },
        )
        choices = _choices_from_structured_data(response.data)
        policy = effective_content_safety_policy(
            self.repositories,
            user_id=prepared.current_user_id,
        )
        reviewed_choices: list[str] = []
        content_ratings: list[str] = []
        for choice in choices:
            safety = await self.content_safety_service.review_narration(
                body=choice,
                content_rating=policy.rating,
                fade_to_black_enabled=False,
                save_id=prepared.save_id,
                source_request=ChatRequest(
                    provider=response.provider,
                    model_id=response.model_id,
                    messages=(),
                ),
            )
            reviewed_choices.append(safety.body)
            content_ratings.append(safety.reviewed_content_rating)
        return self.repositories.replace_message_action_choices(
            save_id=prepared.save_id,
            message_id=prepared.narrator_message_id,
            choices=reviewed_choices,
            provider=response.provider,
            model=response.model_id,
            content_ratings=content_ratings,
        )


def _action_choice_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "choices": {
                "type": "array",
                "minItems": ACTION_CHOICE_COUNT,
                "maxItems": ACTION_CHOICE_COUNT,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "body": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                    "required": ["body"],
                },
            },
        },
        "required": ["choices"],
    }


def _model_requirement_error(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> str | None:
    check = check_model_capabilities(
        repositories,
        provider=provider,
        model_id=model_id,
        required=STRUCTURED_OUTPUT_CAPABILITIES,
    )
    if check.reason == MODEL_MISSING_REASON:
        return (
            "Action choice generation model is not in the provider model "
            f"catalog: {model_id}"
        )
    if check.reason == MODEL_UNAVAILABLE_REASON:
        return f"Action choice generation model is unavailable: {model_id}"
    if check.reason == MODEL_LACKS_CAPABILITY_REASON:
        return "Action choice generation model does not advertise structured output"
    return None


def _action_choice_model_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> ModelPreferenceRecord | None:
    return roleplay_model_preference_with_fallbacks(
        repositories=repositories,
        save_id=save_id,
        purposes=(ACTION_CHOICE_GENERATION_TASK, CHARACTER_ACTION_PLANNING_TASK),
    )


def _action_choice_messages(
    *,
    scenario_type: str,
    scenario_title: str,
    scenario_premise: str,
    player_role: str,
    scenario_content: Mapping[str, object],
    player_character: CharacterRecord | None,
    scene_snapshot: SceneSnapshotRecord | None,
    present_characters: tuple[CharacterRecord, ...],
    messages: tuple[MessageRecord, ...],
    narrator_message: MessageRecord,
) -> tuple[ChatMessage, ...]:
    choice_style = _content_text(scenario_content, "choice_style")
    tone_genre = _content_text(scenario_content, "tone_genre")
    recent_messages = messages[-12:]
    transcript = "\n".join(_transcript_line(message) for message in recent_messages)
    scenario_lines = [
        f"Scenario type: {scenario_type}",
        f"Title: {scenario_title}",
        f"Premise: {scenario_premise}",
        f"Player role: {player_role}",
        f"Tone/style: {tone_genre}",
        f"Choice style: {choice_style}",
    ]
    return (
        ChatMessage(
            role="system",
            body=(
                "You are Bragi's player-character action suggestion agent. "
                "Generate exactly four distinct next-action choices for the "
                "player character. Return only typed structured data matching "
                "the provided schema. Each choice body must be a complete "
                "player message Bragi can submit as-is. Make the choices "
                "concrete, mutually distinct, compatible with the latest "
                "narrator decision point, and grounded in what the player "
                "character plausibly knows. Do not write narrator prose, "
                "numbering, labels, markdown, or explanations. These choices "
                "are suggestions only and do not become canon unless the "
                "player chooses one."
            ),
        ),
        ChatMessage(
            role="user",
            body="\n".join(
                (
                    "Scenario:",
                    *[line for line in scenario_lines if not line.endswith(": ")],
                    "",
                    _player_character_text(
                        player_character=player_character,
                        scenario_content=scenario_content,
                        fallback_player_role=player_role,
                    ),
                    "",
                    _scene_text(scene_snapshot),
                    "",
                    _present_characters_text(present_characters),
                    "",
                    "Recent chronicle:",
                    transcript or "No prior chronicle messages.",
                    "",
                    "Latest narrator message:",
                    narrator_message.body,
                )
            ),
        ),
    )


def _player_character(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> CharacterRecord | None:
    return next(
        (
            character
            for character in repositories.list_characters(save_id)
            if character.is_player_character
        ),
        None,
    )


def _player_character_text(
    *,
    player_character: CharacterRecord | None,
    scenario_content: Mapping[str, object],
    fallback_player_role: str,
) -> str:
    if player_character is not None:
        return "\n".join(
            (
                "Player character:",
                *_character_public_lines(
                    player_character,
                    include_player_agency=True,
                ),
            )
        )
    name = _content_text(scenario_content, "player_character_name")
    profile = _content_text(scenario_content, "player_character_profile")
    role = _content_text(scenario_content, "player_role") or fallback_player_role
    parts = [
        f"name: {name}" if name else "name: unspecified",
        f"role: {role}" if role else "",
        f"profile: {profile}" if profile else "",
    ]
    return "\n".join(("Player character:", *(part for part in parts if part)))


def _scene_text(snapshot: SceneSnapshotRecord | None) -> str:
    if snapshot is None:
        return "Current scene: no scene snapshot"
    parts = [
        f"situation: {snapshot.situation}" if snapshot.situation else "",
        f"objective: {snapshot.objective}" if snapshot.objective else "",
        (
            f"time: {world_time}"
            if (world_time := format_world_time_from_snapshot(snapshot))
            else ""
        ),
        f"weather: {snapshot.weather}" if snapshot.weather else "",
        f"mood: {snapshot.mood}" if snapshot.mood else "",
        (
            "nearby objects: " + ", ".join(snapshot.nearby_objects)
            if snapshot.nearby_objects
            else ""
        ),
        "hazards: " + ", ".join(snapshot.hazards) if snapshot.hazards else "",
    ]
    visible_parts = [part for part in parts if part]
    if not visible_parts:
        return "Current scene: no public scene details"
    return "\n".join(("Current scene:", *visible_parts))


def _present_characters(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> tuple[CharacterRecord, ...]:
    snapshot = repositories.get_scene_snapshot(save_id)
    if snapshot is None or not snapshot.present_character_ids:
        return ()
    present_ids = set(snapshot.present_character_ids)
    return tuple(
        character
        for character in repositories.list_characters(save_id)
        if character.id in present_ids
    )


def _present_characters_text(characters: tuple[CharacterRecord, ...]) -> str:
    non_player_characters = tuple(
        character for character in characters if not character.is_player_character
    )
    if not non_player_characters:
        return "Present non-player characters: none known"
    return "\n".join(
        (
            "Present non-player characters:",
            *(
                " - " + " | ".join(_character_public_lines(character))
                for character in non_player_characters
            ),
        )
    )


def _character_public_lines(
    character: CharacterRecord,
    *,
    include_player_agency: bool = False,
) -> tuple[str, ...]:
    lines = [
        f"name: {character.name}",
        f"aliases: {', '.join(character.aliases)}" if character.aliases else "",
        f"role: {character.role}" if character.role else "",
        f"age: {character.age}" if character.age else "",
        f"known state: {character.known_state}" if character.known_state else "",
        f"status: {character.status}" if character.status else "",
        f"appearance: {character.appearance}" if character.appearance else "",
        f"personality: {character.personality}" if character.personality else "",
        f"voice: {character.voice}" if character.voice else "",
    ]
    if include_player_agency:
        lines.extend(
            (
                f"goals: {character.goals}" if character.goals else "",
                f"motivations: {character.motivations}"
                if character.motivations
                else "",
                f"current intent: {character.current_intent}"
                if character.current_intent
                else "",
                f"boundaries: {character.boundaries}" if character.boundaries else "",
            )
        )
    return tuple(line for line in lines if line)


def _choices_from_structured_data(data: Mapping[str, object]) -> tuple[str, ...]:
    raw_choices = data.get("choices")
    if not isinstance(raw_choices, list):
        raise ValueError("Action choice response missing choices array")
    if len(raw_choices) != ACTION_CHOICE_COUNT:
        raise ValueError("Action choice response must contain exactly four choices")
    choices: list[str] = []
    seen: set[str] = set()
    for raw_choice in raw_choices:
        if not isinstance(raw_choice, dict):
            raise ValueError("Action choice item must be an object")
        body = raw_choice.get("body")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("Action choice body must be non-empty text")
        normalized = body.strip()
        key = normalized.casefold()
        if key in seen:
            raise ValueError("Action choice bodies must be unique")
        seen.add(key)
        choices.append(normalized)
    return tuple(choices)


def _scenario_content(content_json: str) -> Mapping[str, object]:
    try:
        loaded = json.loads(content_json)
    except json.JSONDecodeError:
        return {}
    return cast(Mapping[str, object], loaded) if isinstance(loaded, dict) else {}


def _content_text(content: Mapping[str, object], key: str) -> str:
    value = content.get(key)
    return value.strip() if isinstance(value, str) else ""


def _transcript_line(message: MessageRecord) -> str:
    speaker = message.speaker_name or message.role
    return f"{speaker}: {message.body}"
