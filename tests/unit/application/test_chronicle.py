from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from pytest import MonkeyPatch

from bragi.persistence.models import MessageRecord

_MISSING = object()


def test_chronicle_model_renders_persisted_message_fields(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                role="player",
                speaker_name="Mara",
                body="I climb toward the beacon lens.",
            ),
            _message(
                role="narrator",
                speaker_name="Narrator",
                body="Ash scratches the glass as the stair shakes.",
            ),
        )
    )

    rendered_messages = list(_value(model, "messages", "items"))
    assert [_value(message, "role") for message in rendered_messages] == [
        "player",
        "narrator",
    ]
    assert [_value(message, "role_label") for message in rendered_messages] == [
        "Mara",
        "Narrator",
    ]
    rendered_speakers = [
        _value(message, "speaker", "speaker_name") for message in rendered_messages
    ]
    assert rendered_speakers == [
        "Mara",
        "Narrator",
    ]
    assert [_value(message, "body", "text") for message in rendered_messages] == [
        "I climb toward the beacon lens.",
        "Ash scratches the glass as the stair shakes.",
    ]


def test_chronicle_model_marks_edited_narrator_messages_and_action(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                role="narrator",
                speaker_name="Narrator",
                body="The corridor holds steady.",
                message_id="narrator-1",
                provider="fake",
                model="fake-chat",
            ),
        ),
        revision_metadata_by_message_id={
            "narrator-1": chronicle.MessageRevisionMetadata(
                revision_count=1,
                edited_at="2026-06-02 15:30:00",
            )
        },
    )

    [message] = list(_value(model, "messages", "items"))
    assert _value(message, "revision_count") == 1
    assert _value(message, "edited_at") == "2026-06-02 15:30:00"
    assert _actions_by_id(_value(message, "actions"))["edit-narrator-message"] == (
        "Edit this message"
    )


def test_chronicle_model_uses_configured_player_speaker_for_legacy_labels(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                role="player",
                speaker_name="",
                body="I climb toward the beacon lens.",
                message_id="missing-speaker",
            ),
            _message(
                role="player",
                speaker_name="Player",
                body="I light the storm beacon.",
                message_id="legacy-speaker",
            ),
            _message(
                role="player",
                speaker_name="Mara",
                body="I keep my old name in the record.",
                message_id="explicit-speaker",
            ),
        ),
        player_speaker_name="Mara Voss",
    )

    rendered_speakers = [
        _value(message, "speaker", "speaker_name")
        for message in _value(model, "messages", "items")
    ]
    assert rendered_speakers == ["Mara Voss", "Mara Voss", "Mara"]
    rendered_role_labels = [
        _value(message, "role_label") for message in _value(model, "messages", "items")
    ]
    assert rendered_role_labels == ["Mara Voss", "Mara Voss", "Mara"]


def test_chronicle_message_model_exposes_safe_role_presentation_metadata(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)
    arbitrary_role = "gm admin<script>"

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                role="player",
                speaker_name="Mara",
                body="I climb toward the beacon lens.",
            ),
            _message(
                role="narrator",
                speaker_name="Narrator",
                body="Ash scratches the glass as the stair shakes.",
            ),
            _message(
                role="system",
                speaker_name="System",
                body="Autosave complete.",
            ),
            _message(
                role=arbitrary_role,
                speaker_name="GM",
                body="A private note crosses the table.",
            ),
        )
    )

    rendered_messages = list(_value(model, "messages", "items"))
    assert [
        (
            _value(message, "role"),
            _value(message, "role_label"),
            _value(message, "style_class"),
        )
        for message in rendered_messages
    ] == [
        ("player", "Mara", "message-player"),
        ("narrator", "Narrator", "message-narrator"),
        ("system", "System", "message-system"),
        (arbitrary_role, "Message", "message-other"),
    ]

    unknown_message = rendered_messages[-1]
    assert arbitrary_role not in _value(unknown_message, "role_label")
    assert arbitrary_role not in _value(unknown_message, "style_class")


def test_chronicle_model_preserves_raw_body_while_exposing_markdown_blocks(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)
    raw_body = (
        "I light *one* **bright** `flare` beside "
        "[the hatch](https://example.test/hatch)."
    )

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                role="narrator",
                speaker_name="Narrator",
                body=raw_body,
            ),
        )
    )

    [message] = list(_value(model, "messages", "items"))
    assert _value(message, "body", "text") == raw_body

    [paragraph] = _render_blocks(message)
    assert _block_kind(paragraph) == "paragraph"
    assert _display_text(paragraph) == "I light one bright flare beside the hatch."

    spans = _spans_by_text(paragraph)
    assert _marks(spans["one"]) == {"emphasis"}
    assert _marks(spans["bright"]) == {"strong"}
    assert _marks(spans["flare"]) == {"code"}
    assert _marks(spans["the hatch"]) == {"link"}
    assert _value(spans["the hatch"], "url", "href", "link_target", "target") == (
        "https://example.test/hatch"
    )


def test_chronicle_markdown_blocks_cover_lists_quotes_and_fenced_code(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                role="narrator",
                speaker_name="Narrator",
                body=(
                    "- Ash on the sill\n"
                    "- Copper at the lock\n"
                    "\n"
                    "1. Listen at the door\n"
                    "2. Open it carefully\n"
                    "\n"
                    "> The stair answers in whispers.\n"
                    "\n"
                    "```python\n"
                    "print('*not emphasis*')\n"
                    "```"
                ),
            ),
        )
    )

    [message] = list(_value(model, "messages", "items"))
    blocks = _render_blocks(message)

    assert [_block_kind(block) for block in blocks] == [
        "list_item",
        "list_item",
        "list_item",
        "list_item",
        "blockquote",
        "code_block",
    ]
    assert [
        (
            _value(block, "list_kind", "list_type", default=None),
            _value(block, "ordinal", "number", default=None),
            _display_text(block),
        )
        for block in blocks[:4]
    ] == [
        ("bullet", None, "Ash on the sill"),
        ("bullet", None, "Copper at the lock"),
        ("numbered", 1, "Listen at the door"),
        ("numbered", 2, "Open it carefully"),
    ]
    assert _display_text(blocks[4]) == "The stair answers in whispers."
    assert _value(blocks[5], "language", default=None) == "python"
    assert _value(blocks[5], "text", "display_text") == "print('*not emphasis*')"
    assert _value(blocks[5], "spans", "inline_spans", default=()) == ()


def test_chronicle_markdown_omits_unsafe_link_targets_but_keeps_link_text(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                role="narrator",
                speaker_name="Narrator",
                body="[the marked switch](javascript:alert-x) stays inert.",
            ),
        )
    )

    [message] = list(_value(model, "messages", "items"))
    [paragraph] = _render_blocks(message)

    assert _display_text(paragraph) == "the marked switch stays inert."
    assert "javascript:" not in _display_text(paragraph)
    spans = _value(paragraph, "spans", "inline_spans")
    assert all(
        _value(span, "url", "href", "link_target", "target", default=None) is None
        for span in spans
    )


def test_chronicle_markdown_malformed_input_falls_back_to_displayable_text(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)
    raw_body = (
        "A **broken warning and [half link](https://example.test\n"
        "\n"
        "```python\n"
        "print('still readable')"
    )

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                role="narrator",
                speaker_name="Narrator",
                body=raw_body,
            ),
        )
    )

    [message] = list(_value(model, "messages", "items"))
    assert _value(message, "body", "text") == raw_body

    blocks = _render_blocks(message)
    assert blocks
    assert all(_display_text(block) for block in blocks)
    assert "broken warning" in " ".join(_display_text(block) for block in blocks)


def test_chronicle_markdown_overlong_ordered_list_marker_stays_displayable(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)
    overlong_marker = "9" * 5000
    raw_body = f"{overlong_marker}. too many"

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                role="narrator",
                speaker_name="Narrator",
                body=raw_body,
            ),
        )
    )

    [message] = list(_value(model, "messages", "items"))
    assert _value(message, "body", "text") == raw_body

    [block] = _render_blocks(message)
    assert _block_kind(block) == "list_item"
    assert _value(block, "list_kind", "list_type", default=None) == "numbered"
    assert _value(block, "ordinal", "number", default=None) is None
    marker = str(_value(block, "marker", "list_marker"))
    assert 0 < len(marker) <= 16
    assert overlong_marker not in marker
    assert _display_text(block) == "too many"


def test_chronicle_message_actions_include_image_generation_revision_and_player_edit(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                role="player",
                speaker_name="Mara",
                body="I climb toward the beacon lens.",
            ),
            _message(
                role="narrator",
                speaker_name="Narrator",
                body="The sealed stacks breathe out candle smoke.",
            ),
        )
    )
    player_message, narrator_message = list(_value(model, "messages", "items"))

    player_actions = _actions_by_id(_value(player_message, "actions"))
    narrator_actions = _actions_by_id(_value(narrator_message, "actions"))
    for actions in (player_actions, narrator_actions):
        assert actions["generate-scene-image"] == "Generate image of this scene"
        assert actions["fork-from-here"] == "Fork from here"
        assert actions["delete-messages-from-here"] == "Delete from here"
        assert all("video" not in label.casefold() for label in actions.values())
    assert player_actions["edit-and-resubmit-message"] == "Edit this message"
    assert "regenerate-message" not in player_actions
    assert narrator_actions["regenerate-message"] == "Regenerate"
    assert narrator_actions["regenerate-message-with-feedback"] == (
        "Regenerate with feedback"
    )
    assert "edit-and-resubmit-message" not in narrator_actions


def test_chronicle_character_messages_include_character_image_action(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                message_id="message-oracle-window",
                role="narrator",
                speaker_name="Narrator",
                body="The oracle turns toward the moonlit window.",
            ),
        ),
        character_image_actions_enabled=True,
        character_image_message_ids=frozenset({"message-oracle-window"}),
        scene_presence_actions_enabled=True,
    )

    [message] = list(_value(model, "messages", "items"))
    actions = _actions_by_id(_value(message, "actions"))
    assert actions["generate-scene-image"] == "Generate image of this scene"
    assert actions["view-characters-present"] == "Characters present"
    assert actions["generate-character-image"] == "Generate image of a character"


def test_chronicle_hides_character_image_action_when_message_not_eligible(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                message_id="message-without-reference",
                role="narrator",
                speaker_name="Narrator",
                body="The oracle turns toward the moonlit window.",
            ),
        ),
        character_image_actions_enabled=True,
        character_image_message_ids=frozenset(),
        scene_presence_actions_enabled=True,
    )

    [message] = list(_value(model, "messages", "items"))
    actions = _actions_by_id(_value(message, "actions"))
    assert "view-characters-present" in actions
    assert "generate-character-image" not in actions


def test_chronicle_debug_prompt_action_requires_model_message_with_prompt(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)
    model_prompt = "The exact prompt sent to the narrator model."

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                message_id="player-with-prompt",
                role="player",
                speaker_name="Mara",
                body="I inspect the hatch.",
                provider="openrouter",
                model="narrator-model",
            ),
            _message(
                message_id="manual-narrator-with-prompt",
                role="narrator",
                speaker_name="Narrator",
                body="The hatch is already open.",
            ),
            _message(
                message_id="model-narrator-without-prompt",
                role="narrator",
                speaker_name="Narrator",
                body="A cold draft rises from below.",
                provider="openrouter",
                model="narrator-model",
            ),
            _message(
                message_id="model-narrator-with-prompt",
                role="narrator",
                speaker_name="Narrator",
                body="The lens hums with borrowed daylight.",
                provider="openrouter",
                model="narrator-model",
            ),
        ),
        debug_prompts_enabled=True,
        debug_prompt_text_by_message_id={
            "player-with-prompt": "Player prompt should not be inspectable.",
            "manual-narrator-with-prompt": (
                "Manual narrator prompt should not be inspectable."
            ),
            "model-narrator-with-prompt": model_prompt,
        },
    )

    messages_by_id = {
        _value(message, "message_id"): message
        for message in _value(model, "messages", "items")
    }

    assert _inspect_action_ids(messages_by_id["player-with-prompt"]) == []
    assert _inspect_action_ids(messages_by_id["manual-narrator-with-prompt"]) == []
    assert _inspect_action_ids(messages_by_id["model-narrator-without-prompt"]) == []

    action = _single_inspect_action(messages_by_id["model-narrator-with-prompt"])
    assert _value(action, "action_id", "id") == "inspect-debug-prompt"
    assert _value(action, "label") == "Inspect prompt"
    assert _value(action, "detail_text") == model_prompt


def test_chronicle_debug_prompt_action_hidden_when_debug_mode_disabled(
    monkeypatch: MonkeyPatch,
) -> None:
    chronicle = _import_chronicle_without_gtk(monkeypatch)

    model = chronicle.build_chronicle_model(
        messages=(
            _message(
                message_id="model-narrator-with-prompt",
                role="narrator",
                speaker_name="Narrator",
                body="The lens hums with borrowed daylight.",
                provider="openrouter",
                model="narrator-model",
            ),
        ),
        debug_prompts_enabled=False,
        debug_prompt_text_by_message_id={
            "model-narrator-with-prompt": "Prompt text exists but debug is off.",
        },
    )

    [message] = list(_value(model, "messages", "items"))
    assert _inspect_action_ids(message) == []


def _import_chronicle_without_gtk(monkeypatch: MonkeyPatch) -> Any:
    original_import = builtins.__import__

    def import_without_gtk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gi" or name.startswith("gi."):
            raise AssertionError(
                "bragi.application.chronicle must not import GTK/PyGObject"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gtk)
    sys.modules.pop("bragi.application.chronicle", None)
    return importlib.import_module("bragi.application.chronicle")


def _message(
    *,
    role: str,
    speaker_name: str | None,
    body: str,
    message_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> MessageRecord:
    return MessageRecord(
        id=message_id or f"{role}-message",
        save_id="save-1",
        role=role,
        body=body,
        speaker_name=speaker_name,
        provider=provider,
        model=model,
        token_estimate=None,
        deleted_at=None,
    )


def _actions_by_id(items: Iterable[object]) -> dict[str, str]:
    return {_value(item, "action_id", "id"): _value(item, "label") for item in items}


def _inspect_action_ids(message: object) -> list[str]:
    return [
        _value(action, "action_id", "id")
        for action in _value(message, "actions")
        if _value(action, "action_id", "id").startswith("inspect-")
    ]


def _single_inspect_action(message: object) -> object:
    actions = [
        action
        for action in _value(message, "actions")
        if _value(action, "action_id", "id").startswith("inspect-")
    ]
    assert len(actions) == 1
    return actions[0]


def _render_blocks(message: object) -> tuple[object, ...]:
    return tuple(_value(message, "render_blocks", "markdown_blocks"))


def _block_kind(block: object) -> str:
    return str(_value(block, "block_type", "kind", "type"))


def _spans_by_text(block: object) -> dict[str, object]:
    spans = _value(block, "spans", "inline_spans")
    return {_value(span, "text", "display_text"): span for span in spans}


def _display_text(block: object) -> str:
    text = _value(block, "text", "display_text", default="")
    if text:
        return str(text)

    spans = _value(block, "spans", "inline_spans", default=())
    return "".join(
        str(_value(span, "text", "display_text", default="")) for span in spans
    )


def _marks(span: object) -> set[str]:
    def normalized_mark(value: object) -> str | None:
        raw_value = getattr(value, "value", value)
        if raw_value == "inline_code":
            raw_value = "code"
        if raw_value in {"emphasis", "strong", "code", "link"}:
            return str(raw_value)
        return None

    direct_marks = _value(span, "marks", default=None)
    if direct_marks is not None:
        return {
            normalized
            for mark in direct_marks
            if (normalized := normalized_mark(mark)) is not None
        }

    marks: set[str] = set()
    kind = _value(span, "kind", "span_type", default=None)
    normalized = normalized_mark(kind)
    if normalized is not None:
        marks.add(normalized)

    for mark in ("emphasis", "strong", "code", "link"):
        if _value(span, mark, f"is_{mark}", default=False):
            marks.add(mark)
    return marks


def _value(
    item: object,
    *names: str,
    default: object = _MISSING,
) -> Any:
    for name in names:
        if isinstance(item, Mapping):
            if name in item:
                return item[name]
        elif hasattr(item, name):
            return getattr(item, name)

    if default is not _MISSING:
        return default

    raise AssertionError(f"{item!r} does not expose any of {names!r}")
