"""Import-safe chronicle view model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from bragi.persistence.models import MessageRecord


class ChronicleMarkdownBlockKind(StrEnum):
    PARAGRAPH = "paragraph"
    BLOCKQUOTE = "blockquote"
    BULLET_ITEM = "bullet_item"
    NUMBERED_ITEM = "numbered_item"
    CODE_BLOCK = "code_block"
    THEMATIC_BREAK = "thematic_break"


class ChronicleMarkdownSpanKind(StrEnum):
    TEXT = "text"
    EMPHASIS = "emphasis"
    STRONG = "strong"
    INLINE_CODE = "inline_code"
    LINK = "link"


@dataclass(frozen=True)
class ChronicleMarkdownSpan:
    kind: ChronicleMarkdownSpanKind
    text: str
    target: str | None = None


@dataclass(frozen=True)
class ChronicleMarkdownBlock:
    kind: ChronicleMarkdownBlockKind
    spans: tuple[ChronicleMarkdownSpan, ...] = ()
    text: str = ""
    marker: str | None = None
    list_kind: str | None = None
    ordinal: int | None = None
    language: str | None = None

    @property
    def block_type(self) -> str:
        if self.kind in {
            ChronicleMarkdownBlockKind.BULLET_ITEM,
            ChronicleMarkdownBlockKind.NUMBERED_ITEM,
        }:
            return "list_item"
        return self.kind.value


@dataclass(frozen=True)
class ChronicleMessageAction:
    action_id: str
    label: str
    detail_text: str | None = None


@dataclass(frozen=True)
class MessageRevisionMetadata:
    revision_count: int
    edited_at: str | None


@dataclass(frozen=True)
class ChronicleMessageModel:
    message_id: str
    role: str
    speaker_name: str | None
    body: str
    markdown_blocks: tuple[ChronicleMarkdownBlock, ...]
    actions: tuple[ChronicleMessageAction, ...]
    revision_count: int = 0
    edited_at: str | None = None
    debug_prompt: str | None = None
    debug_provider_payload: str | None = None
    content_rating: str = "unclassified"

    @property
    def role_label(self) -> str:
        if self.role == "player" and self.speaker_name:
            return self.speaker_name
        return _message_role_label(self.role)

    @property
    def style_class(self) -> str:
        return _message_style_class(self.role)


@dataclass(frozen=True)
class ChronicleModel:
    messages: tuple[ChronicleMessageModel, ...]
    has_more_before: bool = False
    oldest_message_id: str | None = None


def build_chronicle_model(
    messages: list[MessageRecord],
    *,
    has_more_before: bool = False,
    player_speaker_name: str | None = None,
    character_image_actions_enabled: bool = False,
    character_image_message_ids: frozenset[str] | None = None,
    scene_presence_actions_enabled: bool = False,
    debug_prompt_text_by_message_id: dict[str, str] | None = None,
    debug_provider_payload_text_by_message_id: dict[str, str] | None = None,
    revision_metadata_by_message_id: dict[str, MessageRevisionMetadata] | None = None,
    debug_prompts_enabled: bool = False,
) -> ChronicleModel:
    debug_prompts = debug_prompt_text_by_message_id or {}
    debug_provider_payloads = debug_provider_payload_text_by_message_id or {}
    revision_metadata = revision_metadata_by_message_id or {}
    rendered_messages = tuple(
        ChronicleMessageModel(
            message_id=message.id,
            role=message.role,
            speaker_name=_display_speaker_name(
                message,
                player_speaker_name=player_speaker_name,
            ),
            body=message.body,
            markdown_blocks=parse_message_markdown(message.body),
            actions=_message_actions(
                message,
                character_image_actions_enabled=character_image_actions_enabled,
                character_image_eligible=(
                    character_image_message_ids is not None
                    and message.id in character_image_message_ids
                ),
                scene_presence_actions_enabled=scene_presence_actions_enabled,
                debug_prompt=(
                    debug_prompts.get(message.id)
                    if debug_prompts_enabled
                    else None
                ),
                debug_provider_payload=(
                    debug_provider_payloads.get(message.id)
                    if debug_prompts_enabled
                    else None
                ),
            ),
            revision_count=revision_metadata.get(
                message.id,
                MessageRevisionMetadata(revision_count=0, edited_at=None),
            ).revision_count,
            edited_at=revision_metadata.get(
                message.id,
                MessageRevisionMetadata(revision_count=0, edited_at=None),
            ).edited_at,
            debug_prompt=(
                debug_prompts.get(message.id) if debug_prompts_enabled else None
            ),
            debug_provider_payload=(
                debug_provider_payloads.get(message.id)
                if debug_prompts_enabled
                else None
            ),
            content_rating=message.content_rating,
        )
        for message in messages
    )
    return ChronicleModel(
        messages=rendered_messages,
        has_more_before=has_more_before,
        oldest_message_id=rendered_messages[0].message_id
        if rendered_messages
        else None,
    )


def _display_speaker_name(
    message: MessageRecord,
    *,
    player_speaker_name: str | None,
) -> str | None:
    configured_name = (player_speaker_name or "").strip()
    if message.role == "player" and configured_name:
        current_name = (message.speaker_name or "").strip()
        if not current_name or current_name.casefold() == "player":
            return configured_name
    return message.speaker_name


def parse_message_markdown(text: str) -> tuple[ChronicleMarkdownBlock, ...]:
    """Parse a safe Markdown subset for native presentation."""
    blocks: list[ChronicleMarkdownBlock] = []
    paragraph_lines: list[str] = []
    code_lines: list[str] = []
    code_language: str | None = None
    in_code_block = False

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = "\n".join(paragraph_lines).strip()
        paragraph_lines.clear()
        if paragraph:
            blocks.append(
                ChronicleMarkdownBlock(
                    kind=ChronicleMarkdownBlockKind.PARAGRAPH,
                    spans=_parse_inline_spans(paragraph),
                )
            )

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code_block:
                blocks.append(
                    ChronicleMarkdownBlock(
                        kind=ChronicleMarkdownBlockKind.CODE_BLOCK,
                        text="\n".join(code_lines),
                        language=code_language,
                    )
                )
                code_lines.clear()
                code_language = None
                in_code_block = False
            else:
                flush_paragraph()
                code_language = stripped.removeprefix("```").strip() or None
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(raw_line)
            continue

        if not stripped:
            flush_paragraph()
            continue

        if _is_thematic_break(stripped):
            flush_paragraph()
            blocks.append(
                ChronicleMarkdownBlock(kind=ChronicleMarkdownBlockKind.THEMATIC_BREAK)
            )
            continue

        quote_match = re.match(r"^\s*>\s?(.*)$", line)
        if quote_match is not None:
            flush_paragraph()
            blocks.append(
                ChronicleMarkdownBlock(
                    kind=ChronicleMarkdownBlockKind.BLOCKQUOTE,
                    spans=_parse_inline_spans(quote_match.group(1)),
                    marker=">",
                )
            )
            continue

        bullet_match = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if bullet_match is not None:
            flush_paragraph()
            blocks.append(
                ChronicleMarkdownBlock(
                    kind=ChronicleMarkdownBlockKind.BULLET_ITEM,
                    spans=_parse_inline_spans(bullet_match.group(1)),
                    marker="•",
                    list_kind="bullet",
                )
            )
            continue

        numbered_match = re.match(r"^\s*(\d+)[.)]\s+(.+)$", line)
        if numbered_match is not None:
            flush_paragraph()
            ordinal_text = numbered_match.group(1)
            blocks.append(
                ChronicleMarkdownBlock(
                    kind=ChronicleMarkdownBlockKind.NUMBERED_ITEM,
                    spans=_parse_inline_spans(numbered_match.group(2)),
                    marker=_safe_ordered_marker(ordinal_text),
                    list_kind="numbered",
                    ordinal=_safe_ordinal(ordinal_text),
                )
            )
            continue

        paragraph_lines.append(line)

    if in_code_block:
        paragraph_lines = ["```", *code_lines]
    flush_paragraph()
    return tuple(blocks)


def _parse_inline_spans(text: str) -> tuple[ChronicleMarkdownSpan, ...]:
    spans: list[ChronicleMarkdownSpan] = []
    position = 0
    while position < len(text):
        match = _next_inline_match(text, position)
        if match is None:
            _append_text_span(spans, text[position:])
            break
        start, end, kind, content, target = match
        _append_text_span(spans, text[position:start])
        spans.append(ChronicleMarkdownSpan(kind=kind, text=content, target=target))
        position = end
    return tuple(spans)


def _next_inline_match(
    text: str,
    position: int,
) -> tuple[int, int, ChronicleMarkdownSpanKind, str, str | None] | None:
    patterns = (
        (
            re.compile(r"`([^`\n]+)`"),
            ChronicleMarkdownSpanKind.INLINE_CODE,
        ),
        (
            re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)"),
            ChronicleMarkdownSpanKind.LINK,
        ),
        (
            re.compile(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__"),
            ChronicleMarkdownSpanKind.STRONG,
        ),
        (
            re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!_)_([^_\n]+)_(?!_)"),
            ChronicleMarkdownSpanKind.EMPHASIS,
        ),
    )
    candidates: list[tuple[int, int, ChronicleMarkdownSpanKind, str, str | None]] = []
    for pattern, kind in patterns:
        match = pattern.search(text, position)
        if match is None:
            continue
        if kind == ChronicleMarkdownSpanKind.LINK:
            content = match.group(1)
            target = match.group(2)
            if not _safe_link_target(target):
                target = None
                kind = ChronicleMarkdownSpanKind.TEXT
        else:
            content = next(group for group in match.groups() if group is not None)
            target = None
        candidates.append((match.start(), match.end(), kind, content, target))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])


def _append_text_span(
    spans: list[ChronicleMarkdownSpan],
    text: str,
) -> None:
    if text:
        spans.append(
            ChronicleMarkdownSpan(kind=ChronicleMarkdownSpanKind.TEXT, text=text)
        )


def _safe_link_target(target: str) -> bool:
    return target.startswith(("https://", "http://", "mailto:"))


def _safe_ordinal(text: str) -> int | None:
    if len(text) > 9:
        return None
    return int(text)


def _safe_ordered_marker(text: str) -> str:
    if len(text) > 9:
        return "#."
    return f"{text}."


def _is_thematic_break(text: str) -> bool:
    compact = text.replace(" ", "")
    return (
        len(compact) >= 3
        and len(set(compact)) == 1
        and compact[0] in {"-", "*", "_"}
    )


def _message_actions(
    message: MessageRecord,
    *,
    character_image_actions_enabled: bool = False,
    character_image_eligible: bool = False,
    scene_presence_actions_enabled: bool = False,
    debug_prompt: str | None = None,
    debug_provider_payload: str | None = None,
) -> tuple[ChronicleMessageAction, ...]:
    actions = [
        ChronicleMessageAction(
            action_id="generate-scene-image",
            label="Generate image of this scene",
        ),
    ]
    if scene_presence_actions_enabled:
        actions.append(
            ChronicleMessageAction(
                action_id="view-characters-present",
                label="Characters present",
                detail_text="View or edit who is in this scene.",
            )
        )
    if character_image_actions_enabled and character_image_eligible:
        actions.append(
            ChronicleMessageAction(
                action_id="generate-character-image",
                label="Generate image of a character",
            )
        )
    actions.append(
        ChronicleMessageAction(
            action_id="fork-from-here",
            label="Fork from here",
        )
    )
    actions.append(
        ChronicleMessageAction(
            action_id="delete-messages-from-here",
            label="Delete from here",
        )
    )
    if message.role != "player":
        actions.append(
            ChronicleMessageAction(
                action_id="edit-narrator-message",
                label="Edit this message",
            )
        )
        actions.append(
            ChronicleMessageAction(
                action_id="regenerate-message",
                label="Regenerate",
            )
        )
        actions.append(
            ChronicleMessageAction(
                action_id="regenerate-message-with-feedback",
                label="Regenerate with feedback",
            )
        )
    if message.role == "player":
        actions.append(
            ChronicleMessageAction(
                action_id="edit-and-resubmit-message",
                label="Edit this message",
            )
        )
    if message.role != "player" and message.provider is not None and debug_prompt:
        actions.append(
            ChronicleMessageAction(
                action_id="inspect-debug-prompt",
                label="Inspect prompt",
                detail_text=debug_prompt,
            )
        )
    if (
        message.role != "player"
        and message.provider is not None
        and debug_provider_payload
    ):
        actions.append(
            ChronicleMessageAction(
                action_id="inspect-provider-payload",
                label="Inspect sent payload",
                detail_text=debug_provider_payload,
            )
        )
    return tuple(actions)


def _message_role_label(role: str) -> str:
    labels = {
        "player": "Player",
        "narrator": "Narrator",
        "system": "System",
    }
    return labels.get(role, "Message")


def _message_style_class(role: str) -> str:
    classes = {
        "player": "message-player",
        "narrator": "message-narrator",
        "system": "message-system",
    }
    return classes.get(role, "message-other")
