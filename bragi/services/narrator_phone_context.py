"""Deterministic phone context for narrator turns."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bragi.persistence.models import (
    CharacterRecord,
    CharacterTextActivityEventRecord,
    CharacterTextMessageRecord,
    CharacterTextThreadRecord,
    MessageRecord,
    ScenarioRecord,
    SceneSnapshotRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.character_text_context import (
    canonical_character_text_context_messages,
)
from bragi.services.mention_matching import character_name_is_mentioned

MAX_NARRATOR_PHONE_THREADS = 3
MAX_NARRATOR_PHONE_MESSAGES_PER_THREAD = 8
MAX_NARRATOR_PHONE_LINE_CHARS = 220
MAX_NARRATOR_PHONE_CONTEXT_CHARS = 3000
MAX_NARRATOR_PHONE_ACTIVITY_EVENTS = 8

_PHONE_CONTEXT_BOUNDARY = (
    "Narrator-only side-channel context: only the sender and recipient know a "
    "private phone message unless other context establishes broader knowledge."
)


@dataclass(frozen=True)
class NarratorPhoneContext:
    lines: tuple[str, ...]
    thread_count: int
    message_count: int
    chars: int


@dataclass(frozen=True)
class NarratorPhoneActivityContext:
    lines: tuple[str, ...]
    event_count: int
    thread_count: int
    chars: int
    prior_cursor: int
    next_cursor: int
    baseline: bool = False


@dataclass(frozen=True)
class _IncludedThread:
    thread: CharacterTextThreadRecord
    character: CharacterRecord | None
    messages: list[CharacterTextMessageRecord]
    mode: str
    unread_incoming_count: int = 0
    latest_unread_incoming: CharacterTextMessageRecord | None = None


@dataclass(frozen=True)
class _ThreadRelevance:
    mode: str
    unread_incoming_count: int = 0
    latest_unread_incoming: CharacterTextMessageRecord | None = None


_NO_THREAD_RELEVANCE = _ThreadRelevance(mode="none")

_PHONE_INTENT_PATTERN = re.compile(
    r"(?<![\w-])"
    r"(?:phone|text|texts|texted|texting|message|messages|messaged|"
    r"dm|dms|inbox)"
    r"(?![\w-])",
    re.IGNORECASE,
)
_GLOBAL_PHONE_CONTEXT_PATTERNS = (
    re.compile(
        r"(?<![\w-])"
        r"(?:check|read|open|scan|look(?:ing)?(?:\s+at)?)"
        r"(?:(?![.!?]).){0,40}"
        r"(?:phone|texts?|messages?|dms?|inbox)"
        r"(?![\w-])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![\w-])(?:any|new|unread)\s+(?:texts?|messages?|dms?)"
        r"(?![\w-])",
        re.IGNORECASE,
    ),
)


def build_narrator_phone_context(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario: ScenarioRecord,
    messages: list[MessageRecord],
    player_message: MessageRecord,
    scene_snapshot: SceneSnapshotRecord | None,
    characters: tuple[CharacterRecord, ...],
) -> NarratorPhoneContext:
    characters_by_id = {character.id: character for character in characters}
    player_name = _player_name(characters, scenario)
    previous_narrator = _previous_narrator(messages, player_message)
    included_threads: list[_IncludedThread] = []
    for thread in repositories.list_character_text_threads(save_id):
        thread_messages = canonical_character_text_context_messages(
            repositories=repositories,
            save_id=save_id,
            thread_id=thread.id,
        )
        if not thread_messages:
            continue
        character = (
            characters_by_id.get(thread.character_id)
            if thread.character_id is not None
            else None
        )
        relevance = _thread_relevance(
            thread=thread,
            character=character,
            messages=thread_messages,
            previous_narrator=previous_narrator,
            player_text=player_message.body,
        )
        if relevance.mode == "none":
            continue
        included_threads.append(
            _IncludedThread(
                thread=thread,
                character=character,
                messages=thread_messages,
                mode=relevance.mode,
                unread_incoming_count=relevance.unread_incoming_count,
                latest_unread_incoming=relevance.latest_unread_incoming,
            )
        )
    included_threads.sort(
        key=_included_thread_sort_key,
        reverse=True,
    )
    return _format_context(
        included_threads[:MAX_NARRATOR_PHONE_THREADS],
        player_name=player_name,
        characters_by_id=characters_by_id,
    )


def build_narrator_phone_activity_context(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    messages: list[MessageRecord],
    player_message: MessageRecord,
    characters: tuple[CharacterRecord, ...],
) -> NarratorPhoneActivityContext:
    previous = _previous_narrator(messages, player_message)
    if previous is None:
        cursor = repositories.latest_character_text_activity_ordinal(save_id=save_id)
        return NarratorPhoneActivityContext((), 0, 0, 0, cursor, cursor, True)
    prior_cursor = repositories.narrator_phone_activity_cursor(
        narrator_message_id=previous.id
    ) or 0
    events = repositories.list_character_text_activity_events_after(
        save_id=save_id,
        ordinal=prior_cursor,
        limit=MAX_NARRATOR_PHONE_ACTIVITY_EVENTS,
    )
    if not events:
        return NarratorPhoneActivityContext(
            (), 0, 0, 0, prior_cursor, prior_cursor
        )
    characters_by_id = {character.id: character for character in characters}
    threads = {
        thread.id: thread
        for thread in repositories.list_character_text_threads(save_id)
    }
    summaries: list[str] = []
    thread_ids: set[str] = set()
    for event in events:
        thread = threads.get(event.thread_id)
        if thread is None:
            continue
        thread_ids.add(event.thread_id)
        character = (
            characters_by_id.get(thread.character_id)
            if thread.character_id is not None
            else None
        )
        contact = _contact_name(thread, character)
        summaries.append(_activity_summary(event, contact))
    if not summaries:
        return NarratorPhoneActivityContext(
            (), 0, 0, 0, prior_cursor, events[-1].ordinal
        )
    digest = "\n".join(
        (
            "Narrator-only phone activity: metadata shows what the player visibly "
            "did with their phone. Do not infer or reveal private message contents.",
            *[f"- {summary}" for summary in summaries],
        )
    )
    return NarratorPhoneActivityContext(
        lines=(digest,),
        event_count=len(events),
        thread_count=len(thread_ids),
        chars=len(digest),
        prior_cursor=prior_cursor,
        next_cursor=events[-1].ordinal,
    )


def _activity_summary(event: CharacterTextActivityEventRecord, contact: str) -> str:
    if event.activity_type == "thread_opened":
        if event.read_count:
            return (
                f"Opened {contact}'s text thread and read {event.read_count} "
                "incoming text(s)."
            )
        return f"Opened {contact}'s text thread."
    if event.activity_type == "player_sent":
        status = event.delivery_status or "sent"
        return f"Sent a text to {contact}; delivery status={status}."
    if event.activity_type == "character_received":
        return f"Received a text notification from {contact}."
    return f"Phone activity with {contact}."


def _thread_relevance(
    *,
    thread: CharacterTextThreadRecord,
    character: CharacterRecord | None,
    messages: list[CharacterTextMessageRecord],
    previous_narrator: MessageRecord | None,
    player_text: str,
) -> _ThreadRelevance:
    unread_recent = _recent_unread_incoming_messages(
        messages,
        previous_narrator=previous_narrator,
    )
    phone_intent = _player_has_phone_context_intent(player_text)
    if phone_intent and (
        _thread_is_addressed_by_player(
            thread=thread,
            character=character,
            player_text=player_text,
        )
        or (
            unread_recent
            and _player_has_global_phone_context_intent(player_text)
        )
    ):
        return _ThreadRelevance(
            mode="full",
            unread_incoming_count=len(unread_recent),
            latest_unread_incoming=unread_recent[-1] if unread_recent else None,
        )
    return _NO_THREAD_RELEVANCE


def _thread_is_addressed_by_player(
    *,
    thread: CharacterTextThreadRecord,
    character: CharacterRecord | None,
    player_text: str,
) -> bool:
    if character is not None and character_name_is_mentioned(
        name=character.name,
        aliases=character.aliases,
        text=player_text,
    ):
        return True
    if not thread.title.strip():
        return False
    return character_name_is_mentioned(
        name=thread.title,
        aliases=(),
        text=player_text,
    )


def _recent_unread_incoming_messages(
    messages: list[CharacterTextMessageRecord],
    *,
    previous_narrator: MessageRecord | None,
) -> tuple[CharacterTextMessageRecord, ...]:
    return tuple(
        message
        for message in messages
        if message.sender == "character"
        and message.read_at is None
        and _message_is_since_previous_narrator(
            message,
            previous_narrator=previous_narrator,
        )
    )


def _message_is_since_previous_narrator(
    message: CharacterTextMessageRecord,
    *,
    previous_narrator: MessageRecord | None,
) -> bool:
    if previous_narrator is None:
        return False
    narrator_created_at = previous_narrator.created_at or ""
    if not narrator_created_at:
        return False
    message_time = (
        message.delivered_at or message.created_at or message.updated_at or ""
    )
    return bool(message_time and message_time >= narrator_created_at)


def _player_has_phone_context_intent(player_text: str) -> bool:
    return _PHONE_INTENT_PATTERN.search(player_text) is not None


def _player_has_global_phone_context_intent(player_text: str) -> bool:
    return any(
        pattern.search(player_text) is not None
        for pattern in _GLOBAL_PHONE_CONTEXT_PATTERNS
    )


def _format_context(
    threads: list[_IncludedThread],
    *,
    player_name: str,
    characters_by_id: dict[str, CharacterRecord],
) -> NarratorPhoneContext:
    if not threads:
        return NarratorPhoneContext(lines=(), thread_count=0, message_count=0, chars=0)
    lines: list[str] = [_PHONE_CONTEXT_BOUNDARY]
    message_count = 0
    thread_count = 0
    for included in threads:
        thread_lines, included_message_count = _format_thread_lines(
            included,
            player_name=player_name,
            characters_by_id=characters_by_id,
        )
        candidate_lines = lines + thread_lines
        if _line_chars(candidate_lines) > MAX_NARRATOR_PHONE_CONTEXT_CHARS:
            break
        lines.extend(thread_lines)
        thread_count += 1
        message_count += included_message_count
    if thread_count == 0:
        return NarratorPhoneContext(lines=(), thread_count=0, message_count=0, chars=0)
    return NarratorPhoneContext(
        lines=tuple(lines),
        thread_count=thread_count,
        message_count=message_count,
        chars=_line_chars(lines),
    )


def _format_thread_lines(
    included: _IncludedThread,
    *,
    player_name: str,
    characters_by_id: dict[str, CharacterRecord],
) -> tuple[list[str], int]:
    thread = included.thread
    character = included.character
    contact_name = _contact_name(thread, character)
    if included.mode == "notification":
        latest = included.latest_unread_incoming or included.messages[-1]
        return (
            [
                (
                    f"Phone notification: {contact_name}; latest="
                    f"{_message_time_label(latest)}; status={thread.status}; "
                    f"unread incoming messages={included.unread_incoming_count}"
                )
            ],
            0,
        )

    recent_messages = included.messages[-MAX_NARRATOR_PHONE_MESSAGES_PER_THREAD:]
    latest = recent_messages[-1]
    thread_lines = [
        (
            f"Phone thread: {contact_name}; latest="
            f"{_message_time_label(latest)}; status={thread.status}"
        )
    ]
    if thread.memory_body.strip():
        thread_lines.append(
            "Thread memory: "
            + _compact_line(thread.memory_body, MAX_NARRATOR_PHONE_LINE_CHARS)
        )
    thread_lines.append(f"Recent phone messages with {contact_name}:")
    for message in recent_messages:
        thread_lines.append(
            "  - "
            + _compact_line(
                (
                    f"{_message_time_label(message)} "
                    f"{_sender_label(
                        message,
                        character,
                        player_name,
                        characters_by_id,
                    )}: "
                    f"{message.body}"
                ),
                MAX_NARRATOR_PHONE_LINE_CHARS,
            )
        )
    return thread_lines, len(recent_messages)


def _previous_narrator(
    messages: list[MessageRecord],
    player_message: MessageRecord,
) -> MessageRecord | None:
    narrators = [
        message
        for message in messages
        if message.role == "narrator" and message.id != player_message.id
    ]
    return narrators[-1] if narrators else None


def _player_name(
    characters: tuple[CharacterRecord, ...],
    scenario: ScenarioRecord,
) -> str:
    for character in characters:
        if character.is_player_character and character.name.strip():
            return character.name.strip()
    if scenario.player_role.strip():
        return scenario.player_role.strip()
    return "Player"


def _contact_name(
    thread: CharacterTextThreadRecord,
    character: CharacterRecord | None,
) -> str:
    if character is not None:
        if character.contact_name.strip():
            return character.contact_name.strip()
        if character.name.strip():
            return character.name.strip()
    if thread.title.strip():
        return thread.title.strip()
    return "Unknown contact"


def _sender_label(
    message: CharacterTextMessageRecord,
    character: CharacterRecord | None,
    player_name: str,
    characters_by_id: dict[str, CharacterRecord],
) -> str:
    if message.sender == "player":
        return player_name
    if message.sender_character_id:
        sender = characters_by_id.get(message.sender_character_id)
        if sender is not None:
            return sender.contact_name.strip() or sender.name.strip() or "Character"
    if character is None:
        return "Character"
    return character.contact_name.strip() or character.name.strip() or "Character"


def _message_time_label(message: CharacterTextMessageRecord) -> str:
    return (
        message.in_world_sent_at
        or message.delivered_at
        or message.created_at
        or "time unknown"
    )


def _latest_message_sort_key(messages: list[CharacterTextMessageRecord]) -> str:
    latest = messages[-1]
    return latest.updated_at or latest.created_at or latest.delivered_at or ""


def _included_thread_sort_key(included: _IncludedThread) -> str:
    if included.mode == "notification" and included.latest_unread_incoming is not None:
        return _message_sort_key(included.latest_unread_incoming)
    return _latest_message_sort_key(included.messages)


def _message_sort_key(message: CharacterTextMessageRecord) -> str:
    return message.delivered_at or message.created_at or message.updated_at or ""


def _compact_line(value: str, limit: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: max(0, limit - 3)].rstrip() + "..."


def _line_chars(lines: list[str]) -> int:
    return sum(len(line) for line in lines)
