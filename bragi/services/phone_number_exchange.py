"""Deterministic phone-number exchange inference."""

from __future__ import annotations

from dataclasses import dataclass
from re import search, sub

from bragi.persistence.models import CharacterRecord, MessageRecord

PHONE_EXCHANGE_PLAYER_HAS_CHARACTER_NUMBER = "player_has_character_number"
PHONE_EXCHANGE_CHARACTER_HAS_PLAYER_NUMBER = "character_has_player_number"
PHONE_EXCHANGE_BOTH = "both"

_CHARACTER_TITLE_WORDS = frozenset(
    {
        "admiral",
        "brother",
        "captain",
        "commander",
        "dame",
        "doctor",
        "dr",
        "father",
        "general",
        "inspector",
        "king",
        "lady",
        "lieutenant",
        "lord",
        "mother",
        "prince",
        "princess",
        "prof",
        "professor",
        "queen",
        "sergeant",
        "sir",
        "sister",
        "warden",
    }
)


@dataclass(frozen=True)
class PhoneNumberExchange:
    character_id: str
    direction: str
    source_message_id: str
    reason: str = "Inferred from an explicit Chronicle phone number exchange."
    confidence: float = 0.85


def infer_phone_number_exchanges(
    *,
    completed_messages: tuple[MessageRecord, ...],
    player: CharacterRecord,
    npcs: tuple[CharacterRecord, ...],
) -> tuple[PhoneNumberExchange, ...]:
    inferred: list[PhoneNumberExchange] = []
    for message in completed_messages:
        exchange = infer_phone_number_exchange_from_message(
            message,
            player=player,
            npcs=npcs,
        )
        if exchange is not None:
            inferred.append(exchange)
    return tuple(inferred)


def infer_phone_number_exchange_from_message(
    message: MessageRecord,
    *,
    player: CharacterRecord,
    npcs: tuple[CharacterRecord, ...],
) -> PhoneNumberExchange | None:
    text_key = _phone_exchange_text_key(message.body)
    if not text_key:
        return None
    mentioned_npcs = tuple(
        character
        for character in npcs
        if _phone_exchange_mentions_character(
            text_key,
            character,
            speaker_name=message.speaker_name,
        )
    )
    if len(mentioned_npcs) != 1:
        return None
    npc = mentioned_npcs[0]
    player_speaker_name = None if message.role == "player" else message.speaker_name
    player_involved = _phone_exchange_mentions_character(
        text_key,
        player,
        speaker_name=player_speaker_name,
    ) or (
        message.role == "player"
        and _phone_exchange_has_player_speaker_pronoun(text_key)
    )
    if not player_involved:
        return None
    direction = ""
    if _phone_exchange_is_reciprocal(text_key):
        direction = PHONE_EXCHANGE_BOTH
    elif _phone_exchange_gives_number(
        text_key,
        giver=player,
        receiver=npc,
    ):
        direction = PHONE_EXCHANGE_CHARACTER_HAS_PLAYER_NUMBER
    elif _phone_exchange_gives_number(
        text_key,
        giver=npc,
        receiver=player,
    ):
        direction = PHONE_EXCHANGE_PLAYER_HAS_CHARACTER_NUMBER
    elif message.role == "player" and _phone_exchange_first_person_gives_number(
        text_key,
        receiver=npc,
    ):
        direction = PHONE_EXCHANGE_CHARACTER_HAS_PLAYER_NUMBER
    elif _phone_exchange_speaker_gives_number(
        text_key,
        message=message,
        speaker=npc,
        receiver=player,
    ):
        direction = PHONE_EXCHANGE_PLAYER_HAS_CHARACTER_NUMBER
    if not direction:
        return None
    return PhoneNumberExchange(
        character_id=npc.id,
        direction=direction,
        source_message_id=message.id,
    )


def _phone_exchange_text_key(value: str) -> str:
    text = value.casefold().replace("'", " ")
    text = sub(r"[_/\\-]+", " ", text)
    text = sub(r"[^\w\s]", " ", text)
    return sub(r"\s+", " ", text).strip()


def _phone_exchange_mentions_character(
    text_key: str,
    character: CharacterRecord,
    *,
    speaker_name: str | None,
) -> bool:
    if _phone_exchange_speaker_matches(speaker_name, character):
        return True
    return any(
        _phone_exchange_term_index(text_key, term) is not None
        for term in _phone_exchange_character_terms(character)
    )


def _phone_exchange_speaker_matches(
    speaker_name: str | None,
    character: CharacterRecord,
) -> bool:
    if not speaker_name:
        return False
    speaker_key = _phone_exchange_text_key(speaker_name)
    return any(
        speaker_key == _phone_exchange_text_key(term)
        for term in _phone_exchange_character_terms(character)
    )


def _phone_exchange_character_terms(character: CharacterRecord) -> tuple[str, ...]:
    terms = [character.name, *character.aliases]
    key = _character_name_key(character.name)
    parts = key.split()
    if len(parts) > 1:
        terms.append(parts[0])
    return tuple(dict.fromkeys(term for term in terms if term.strip()))


def _phone_exchange_term_index(text_key: str, term: str) -> int | None:
    term_key = _phone_exchange_text_key(term)
    if not term_key:
        return None
    index = f" {text_key} ".find(f" {term_key} ")
    return index if index >= 0 else None


def _phone_exchange_is_reciprocal(text_key: str) -> bool:
    if search(
        r"\b(?:ask|asks|asked|asking)\b.{0,40}\b(?:exchange|swap|trade|share)\b",
        text_key,
    ):
        return False
    return any(
        search(pattern, text_key) is not None
        for pattern in (
            (
                r"\b(?:exchange|exchanged|swap|swapped|trade|traded|"
                r"share|shared)\s+(?:phone\s+|cell\s+|mobile\s+)?numbers?\b"
            ),
            (
                r"\b(?:exchange|exchanged|swap|swapped|trade|traded|"
                r"share|shared)\s+contact\s+(?:info|details)\b"
            ),
        )
    )


def _phone_exchange_gives_number(
    text_key: str,
    *,
    giver: CharacterRecord,
    receiver: CharacterRecord,
) -> bool:
    giver_indices = [
        index
        for term in _phone_exchange_character_terms(giver)
        if (index := _phone_exchange_term_index(text_key, term)) is not None
    ]
    receiver_indices = [
        index
        for term in _phone_exchange_character_terms(receiver)
        if (index := _phone_exchange_term_index(text_key, term)) is not None
    ]
    if not giver_indices or not receiver_indices:
        return False
    for giver_index in giver_indices:
        verb_index = _phone_exchange_give_verb_index(text_key, start=giver_index)
        if verb_index is None:
            continue
        if not _phone_exchange_has_number_object(text_key[verb_index:]):
            continue
        if any(receiver_index > verb_index for receiver_index in receiver_indices):
            return True
    return False


def _phone_exchange_first_person_gives_number(
    text_key: str,
    *,
    receiver: CharacterRecord,
) -> bool:
    receiver_indices = [
        index
        for term in _phone_exchange_character_terms(receiver)
        if (index := _phone_exchange_term_index(text_key, term)) is not None
    ]
    if not receiver_indices:
        return False
    match = search(r"\b(?:i|i ll|ill|i will)\s+", text_key)
    if match is None:
        return False
    verb_index = _phone_exchange_give_verb_index(text_key, start=match.start())
    if verb_index is None:
        return False
    return _phone_exchange_has_number_object(text_key[verb_index:]) and any(
        receiver_index > verb_index for receiver_index in receiver_indices
    )


def _phone_exchange_has_player_speaker_pronoun(text_key: str) -> bool:
    return search(r"\b(?:i|me|my|mine|we|us|our|ours)\b", text_key) is not None


def _phone_exchange_speaker_gives_number(
    text_key: str,
    *,
    message: MessageRecord,
    speaker: CharacterRecord,
    receiver: CharacterRecord,
) -> bool:
    if not _phone_exchange_speaker_matches(message.speaker_name, speaker):
        return False
    return _phone_exchange_first_person_gives_number(
        text_key,
        receiver=receiver,
    )


def _phone_exchange_give_verb_index(text_key: str, *, start: int) -> int | None:
    match = search(
        (
            r"\b(?:give|gives|gave|giving|send|sends|sent|sending|share|"
            r"shares|shared|sharing|pass|passes|passed|passing|offer|"
            r"offers|offered|offering)\b"
        ),
        text_key[start:],
    )
    return start + match.start() if match is not None else None


def _phone_exchange_has_number_object(text_key: str) -> bool:
    return any(
        search(pattern, text_key) is not None
        for pattern in (
            r"\b(?:my|your|his|her|their|phone|cell|mobile)\s+numbers?\b",
            r"\bcontact\s+(?:info|details)\b",
            r"\b[a-z0-9]+\s+s\s+numbers?\b",
        )
    )


def _character_name_key(value: str) -> str:
    text = sub(r"\s+", " ", value.strip()).casefold()
    if not text:
        return ""
    parts = text.split()
    while parts and parts[0].rstrip(".") in _CHARACTER_TITLE_WORDS:
        parts = parts[1:]
    return " ".join(parts) or text
