"""Shared message correction context for reconciliation jobs."""

from __future__ import annotations

from dataclasses import dataclass

_MAX_CORRECTION_DIFF_CHARS = 4000


@dataclass(frozen=True)
class MessageCorrectionContext:
    message_id: str
    previous_body: str
    new_body: str
    diff_unified: str
    message_role: str = "narrator"


def correction_context_text(context: MessageCorrectionContext | None) -> str:
    if context is None:
        return ""
    role = _correction_role(context.message_role)
    return "\n\n".join(
        (
            f"{role.title_label} message correction:",
            f"Message ID: {context.message_id}",
            f"Edited {role.text_label} text:\n" + context.new_body,
            (
                f"Previous {role.text_label} text is intentionally omitted; use the "
                "bounded diff below only to infer the narrow correction."
            ),
            "Unified diff:\n" + _bounded_text(context.diff_unified),
            (
                "Focused instruction: infer only the user's narrow correction "
                "from this edit. Update deterministic world, scene, character, "
                "memory, and thread data only where it should match the edited "
                f"{role.message_label}. Preserve unrelated facts and user edits."
            ),
        )
    )


@dataclass(frozen=True)
class _CorrectionRole:
    title_label: str
    text_label: str
    message_label: str


def _correction_role(message_role: str) -> _CorrectionRole:
    role = message_role.strip().casefold()
    if role == "player":
        return _CorrectionRole(
            title_label="Player",
            text_label="player",
            message_label="player message",
        )
    if role == "narrator":
        return _CorrectionRole(
            title_label="Narrator",
            text_label="narrator",
            message_label="narrator message",
        )
    return _CorrectionRole(
        title_label="Message",
        text_label="message",
        message_label="message",
    )


def _bounded_text(value: str) -> str:
    if len(value) <= _MAX_CORRECTION_DIFF_CHARS:
        return value
    return value[:_MAX_CORRECTION_DIFF_CHARS] + "\n[diff truncated]"
