"""Deterministic ordinary name pools for scenario generation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from types import MappingProxyType
from typing import Any

_RESOURCE_PACKAGE = "bragi.data.name_sources"
_RESOURCE_FILES = MappingProxyType(
    {
        "feminine": "ordinary_feminine.txt",
        "masculine": "ordinary_masculine.txt",
        "neutral": "ordinary_neutral.txt",
    }
)
_FANTASY_SCENARIO_TYPES = frozenset({"fantasy_roleplay"})
_NAME_CANDIDATE_SECTION_IDS = frozenset(
    {
        "player_character_name",
        "character_name",
        "character_starters",
    }
)
_CAST_DEDUPE_SECTION_IDS = frozenset({"character_starters"})
_PROSE_NAME_MARKERS = (
    " is ",
    " was ",
    " has ",
    " works ",
    " lives ",
    " goes by ",
    " stands ",
)
_OPTION_LABEL_RE = re.compile(
    r"^(?:romance\s+option|option|route|love\s+interest)\s+"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[:.)-]\s*",
    re.IGNORECASE,
)
_TITLE_WORDS = frozenset(
    {
        "admiral",
        "archivist",
        "brother",
        "captain",
        "commander",
        "dame",
        "doctor",
        "dr",
        "duchess",
        "duke",
        "father",
        "general",
        "guildmaster",
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
class OrdinaryNameCandidates:
    feminine: tuple[str, ...] = ()
    masculine: tuple[str, ...] = ()
    neutral: tuple[str, ...] = ()

    def any(self) -> bool:
        return bool(self.feminine or self.masculine or self.neutral)


def ordinary_name_candidate_context(
    *,
    scenario_type: Any,
    section_id: str,
    seed: str,
    sections: Mapping[str, str],
    per_bucket: int = 12,
) -> str:
    candidates = ordinary_name_candidates(
        scenario_type=scenario_type,
        section_id=section_id,
        seed=seed,
        sections=sections,
        per_bucket=per_bucket,
    )
    if not candidates.any():
        return ""
    lines = [
        "Ordinary contemporary name candidates (optional; use only when they fit):"
    ]
    if candidates.feminine:
        lines.append(f"Feminine: {', '.join(candidates.feminine)}")
    if candidates.masculine:
        lines.append(f"Masculine: {', '.join(candidates.masculine)}")
    if candidates.neutral:
        lines.append(f"Neutral: {', '.join(candidates.neutral)}")
    lines.append(
        "Do not repeat first names within this generated cast unless repetition "
        "is intentional and meaningful."
    )
    return "\n".join(lines)


def ordinary_character_name_candidate_context(
    *,
    scenario_type: Any,
    content: Mapping[str, object],
    per_bucket: int = 12,
) -> str:
    return ordinary_name_starter_generation_context(
        scenario_type=scenario_type,
        seed=_source_generation_prompt(content),
        sections=_string_content_sections(content),
        player_character_name=_content_text(content, "player_character_name"),
        existing_starter_names=_character_starter_names_from_content(content),
        per_bucket=per_bucket,
    )


def ordinary_name_starter_generation_context(
    *,
    scenario_type: Any,
    seed: str,
    sections: Mapping[str, str],
    player_character_name: str = "",
    existing_starter_names: Iterable[str] = (),
    per_bucket: int = 12,
) -> str:
    if _has_fantasy_scenario_type(scenario_type) or per_bucket <= 0:
        return ""
    excluded_names = tuple(
        name.strip()
        for name in (player_character_name, *tuple(existing_starter_names))
        if isinstance(name, str) and name.strip()
    )
    used_text = "\n".join(
        value.strip()
        for value in (
            seed,
            *(section_value for section_value in sections.values()),
            *excluded_names,
        )
        if value.strip()
    )
    salt = _candidate_salt(
        scenario_type=scenario_type,
        section_id="character_starters",
        seed=seed,
        sections={**sections, "_starter_exclusions": "\n".join(excluded_names)},
    )
    pools = _ordinary_name_pools()
    candidates = OrdinaryNameCandidates(
        feminine=_stable_name_sample(
            _exclude_used_names(pools["feminine"], used_text),
            per_bucket=per_bucket,
            salt=f"{salt}:feminine",
        ),
        masculine=_stable_name_sample(
            _exclude_used_names(pools["masculine"], used_text),
            per_bucket=per_bucket,
            salt=f"{salt}:masculine",
        ),
        neutral=_stable_name_sample(
            _exclude_used_names(pools["neutral"], used_text),
            per_bucket=per_bucket,
            salt=f"{salt}:neutral",
        ),
    )
    if not candidates.any():
        return ""
    lines = [
        "Ordinary contemporary name candidates for new character starters "
        "(optional; use only when they fit):"
    ]
    if candidates.feminine:
        lines.append(f"Feminine: {', '.join(candidates.feminine)}")
    if candidates.masculine:
        lines.append(f"Masculine: {', '.join(candidates.masculine)}")
    if candidates.neutral:
        lines.append(f"Neutral: {', '.join(candidates.neutral)}")
    lines.append(
        "Do not use the player character name or any existing starter name."
    )
    return "\n".join(lines)


def ordinary_name_candidates(
    *,
    scenario_type: Any,
    section_id: str,
    seed: str,
    sections: Mapping[str, str],
    per_bucket: int = 12,
) -> OrdinaryNameCandidates:
    if (
        _has_fantasy_scenario_type(scenario_type)
        or section_id not in _NAME_CANDIDATE_SECTION_IDS
        or per_bucket <= 0
    ):
        return OrdinaryNameCandidates()
    used_text = "\n".join(
        value.strip()
        for value in (
            seed,
            *(section_value for section_value in sections.values()),
        )
        if value.strip()
    )
    salt = _candidate_salt(
        scenario_type=scenario_type,
        section_id=section_id,
        seed=seed,
        sections=sections,
    )
    pools = _ordinary_name_pools()
    return OrdinaryNameCandidates(
        feminine=_stable_name_sample(
            _exclude_used_names(pools["feminine"], used_text),
            per_bucket=per_bucket,
            salt=f"{salt}:feminine",
        ),
        masculine=_stable_name_sample(
            _exclude_used_names(pools["masculine"], used_text),
            per_bucket=per_bucket,
            salt=f"{salt}:masculine",
        ),
        neutral=_stable_name_sample(
            _exclude_used_names(pools["neutral"], used_text),
            per_bucket=per_bucket,
            salt=f"{salt}:neutral",
        ),
    )


def repeated_first_names_for_section(
    *,
    scenario_type: Any,
    section_id: str,
    text: str,
) -> tuple[str, ...]:
    if (
        _has_fantasy_scenario_type(scenario_type)
        or section_id not in _CAST_DEDUPE_SECTION_IDS
    ):
        return ()
    return repeated_first_names(text)


def repeated_first_names(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    repeated: list[str] = []
    repeated_keys: set[str] = set()
    for name in _cast_name_candidates(text):
        first_name = _first_name(name)
        key = first_name.casefold()
        if not key:
            continue
        if key in seen and key not in repeated_keys:
            repeated.append(first_name)
            repeated_keys.add(key)
        seen.add(key)
    return tuple(repeated)


def _string_content_sections(content: Mapping[str, object]) -> dict[str, str]:
    return {
        key: value.strip()
        for key, value in content.items()
        if isinstance(value, str) and value.strip()
    }


def _source_generation_prompt(content: Mapping[str, object]) -> str:
    source = content.get("_source")
    if not isinstance(source, Mapping):
        return ""
    prompt = source.get("generation_prompt")
    return prompt.strip() if isinstance(prompt, str) else ""


def _content_text(content: Mapping[str, object], key: str) -> str:
    value = content.get(key)
    return value.strip() if isinstance(value, str) else ""


def _character_starter_names_from_content(
    content: Mapping[str, object],
) -> tuple[str, ...]:
    value = content.get("character_starters")
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
        aliases = item.get("aliases")
        if isinstance(aliases, list | tuple):
            names.extend(
                alias.strip()
                for alias in aliases
                if isinstance(alias, str)
            )
    return tuple(name for name in names if name)


@cache
def _ordinary_name_pools() -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType(
        {
            bucket: _load_name_resource(filename)
            for bucket, filename in _RESOURCE_FILES.items()
        }
    )


def _load_name_resource(filename: str) -> tuple[str, ...]:
    path = resources.files(_RESOURCE_PACKAGE).joinpath(filename)
    names = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        key = name.casefold()
        if not name or name.startswith("#") or key in seen:
            continue
        names.append(name)
        seen.add(key)
    return tuple(names)


def _exclude_used_names(names: tuple[str, ...], text: str) -> tuple[str, ...]:
    if not text:
        return names
    return tuple(
        name
        for name in names
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE) is None
    )


def _stable_name_sample(
    names: tuple[str, ...],
    *,
    per_bucket: int,
    salt: str,
) -> tuple[str, ...]:
    if len(names) <= per_bucket:
        return names
    return tuple(
        sorted(
            names,
            key=lambda name: hashlib.sha256(
                f"{salt}\0{name.casefold()}".encode()
            ).hexdigest(),
        )[:per_bucket]
    )


def _candidate_salt(
    *,
    scenario_type: Any,
    section_id: str,
    seed: str,
    sections: Mapping[str, str],
) -> str:
    context = "\n".join(
        f"{key.strip()}={value.strip()}" for key, value in sections.items()
    )
    normalized = "\n".join(
        (
            ",".join(_scenario_type_values(scenario_type)),
            section_id.strip(),
            _normalized_text(seed),
            _normalized_text(context),
        )
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cast_name_candidates(text: str) -> tuple[str, ...]:
    names: list[str] = []
    for fragment in _cast_fragments(text):
        name = _name_from_cast_fragment(fragment)
        if name:
            names.append(name)
    return tuple(names)


def _cast_fragments(text: str) -> tuple[str, ...]:
    fragments: list[str] = []
    for line in text.replace(";", "\n").splitlines():
        cleaned = _clean_fragment(line)
        if not cleaned:
            continue
        fragments.append(cleaned)
    return tuple(fragments)


def _name_from_cast_fragment(fragment: str) -> str:
    fragment = _OPTION_LABEL_RE.sub("", fragment, count=1).strip()
    if not fragment:
        return ""
    for separator in (" - ", " -- ", ":\t", ": ", "\u2013", "\u2014"):
        if separator in fragment:
            candidate = fragment.split(separator, 1)[0].strip()
            if _looks_like_name(candidate):
                return _clean_name(candidate)
    lowered = fragment.casefold()
    marker_index = len(fragment)
    for marker in _PROSE_NAME_MARKERS:
        index = lowered.find(marker)
        if 0 < index < marker_index:
            marker_index = index
    if marker_index == len(fragment):
        return ""
    candidate = fragment[:marker_index].strip()
    if _looks_like_name(candidate):
        return _clean_name(candidate)
    return ""


def _clean_fragment(value: str) -> str:
    return re.sub(
        r"^(?:[-*]|\d+[.)])\s*",
        "",
        value.strip(),
    ).strip()


def _clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip("\"'`()[]{}"))


def _looks_like_name(value: str) -> bool:
    name = _clean_name(value)
    if not name or len(name) > 80 or re.search(r"[.!?;,]", name):
        return False
    words = name.split()
    return bool(words and len(words) <= 6 and words[0][0].isupper())


def _first_name(name: str) -> str:
    clean = _clean_name(name)
    if not clean:
        return ""
    words = clean.split()
    first_word = re.sub(r"[^A-Za-z]", "", words[0]).casefold()
    if first_word in _TITLE_WORDS and len(words) > 1:
        return re.sub(r"[^A-Za-z'-]", "", words[1])
    return re.sub(r"[^A-Za-z'-]", "", words[0])


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _has_fantasy_scenario_type(scenario_type: Any) -> bool:
    return any(
        value in _FANTASY_SCENARIO_TYPES
        for value in _scenario_type_values(scenario_type)
    )


def _scenario_type_values(scenario_type: Any) -> tuple[str, ...]:
    if isinstance(scenario_type, str):
        return (_scenario_type_value(scenario_type),)
    if isinstance(scenario_type, Iterable):
        return tuple(_scenario_type_value(value) for value in scenario_type)
    return (_scenario_type_value(scenario_type),)


def _scenario_type_value(scenario_type: Any) -> str:
    value = getattr(scenario_type, "value", scenario_type)
    return str(value)
