"""Shared chat system prompt sections."""

from __future__ import annotations

DEFAULT_RESPONSE_STYLE_SECTION = (
    "Response style:\n"
    "- Keep responses reasonably short.\n"
    "- Put dialogue in quotation marks.\n"
    "- Put non-dialogue narration in italics.\n"
    "- Format text messages with > at the beginning of each message."
)

DEFAULT_PROSE_SAFETY_SECTION = (
    "Prose safety boundary:\n"
    "- Romance and consensual intimacy may be acknowledged as story events "
    "when appropriate.\n"
    "- Do not generate explicit sexual description or step-by-step sexual "
    "acts.\n"
    "- If a scene would move into explicit intimacy, stop before explicit "
    "detail with a brief, natural, diegetic fade-to-black / next-scene "
    "transition, then resume after time has passed.\n"
    "- Keep the transition in-world and matter-of-fact; never mention "
    "policies, safety rules, or this instruction.\n"
    "- This built-in boundary takes precedence over scenario, save, account, "
    "turn, and regeneration guidance."
)


def prose_safety_section(*, content_rating: str, fade_to_black_enabled: bool) -> str:
    """Render the narrator boundary for the actor's effective safety policy."""

    normalized = content_rating.strip().casefold()
    if normalized == "pg-13" and fade_to_black_enabled:
        return DEFAULT_PROSE_SAFETY_SECTION
    label = {
        "g": "G",
        "pg": "PG",
        "pg-13": "PG-13",
        "r": "R",
        "unrated": "Unrated",
    }.get(normalized, "PG-13")
    lines = [
        "Prose safety boundary:",
        f"- Keep all generated narration within the selected {label} content rating.",
    ]
    if fade_to_black_enabled:
        lines.append(
            "- If sexual intimacy would become explicit, use a brief, natural, "
            "diegetic fade-to-black transition and resume after time has passed."
        )
    lines.extend(
        (
            "- Never mention policies, safety rules, ratings, or this instruction "
            "in the story.",
            "- This built-in boundary takes precedence over scenario, save, account, "
            "turn, and regeneration guidance.",
        )
    )
    return "\n".join(lines)

CHARACTER_TEXT_RESPONSE_STYLE_SECTION = (
    "Character text response style:\n"
    "- Send only the message body, like a normal phone text.\n"
    "- Do not prefix the reply with >.\n"
    "- Do not wrap the whole reply in quotation marks.\n"
    "- Do not use Markdown, italics, action narration, or sender labels.\n"
    "- Do not include timestamps or Sent at labels."
)

DEFAULT_NPC_KNOWLEDGE_BOUNDARY_SECTION = (
    "NPC knowledge boundary:\n"
    "- Narrator knowledge and NPC knowledge are different. System context, "
    "summaries, memories, and player messages are omniscient narrator/audit "
    "context, not automatic NPC knowledge.\n"
    "- Before giving an NPC dialogue, behavior, or targeted reaction, check "
    "whether that NPC witnessed it, overheard it, was told, or can plainly "
    "infer it from visible consequences.\n"
    "- For any player-private or offscreen action, NPCs may react to the "
    "observed consequence, but must not state the hidden method, private "
    "intent, or unseen detail as known unless the story established how they "
    "learned it.\n"
    "- Player agency does not imply NPC compliance. NPCs may refuse, delay, "
    "mislead, negotiate, leave, escalate, or demand proof when consistent with "
    "their motives, boundaries, relationships, leverage, or recent events.\n"
    "- Avoid routine passive ending beats where NPCs wait, give the player "
    "space, or look to the player only to see what they do next. When a "
    "motivated NPC, faction, hazard, clock, or environmental pressure could "
    "act, give it concrete visible initiative."
)
