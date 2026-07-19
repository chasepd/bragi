# Context Assembly

Bragi assembles narrator and image prompts from ordered context tiers. The
deterministic tiers are always built by application code first; model-selected
retrieval is optional deep context.

## Tiers

1. Rules and compact scenario header: scenario title, player identity, current
   scenario scene, and the narrator control rule. Initial premise, tone, and
   player-role setup are included only while the opening narrator message remains
   in the configured recent narrator window.
2. Current scene: scene snapshot, current location, present characters, visible
   objects or hazards, active threads, and read-only pre-turn hints derived from
   the latest player message.
3. Active linked facts and participant continuity: entity links to memories,
   world state, summaries, or scenario sections when linked to the current scene
   entities.
   Relationship and revealed-knowledge world state for active participants is
   also included deterministically so characters do not miss what they already
   know about the player or other present characters.
4. Pending context review: compact, explicitly noncanonical suggestions queued
   for manual review. These narrator-only hints are framed as untrusted metadata,
   omit source message IDs and suggestion reasons from provider-facing text, and
   are excluded from image prompts.
5. Optional retrieval: selected scenario sections, older memories, older world
   state, observations, summaries, and non-baseline recent messages.
6. Baseline transcript: bounded recent player and narrator messages, plus the
   latest player message or selected image source message.

## Source IDs

Every prompt source has a stable source type and ID:

- `scenario:<scenario_id>`
- `scenario:<scenario_id>:section:<section_key>`
- `scene_snapshot:<snapshot_id>`
- `location:<location_id>`
- `character:<character_id>`
- `dating_route_state:<route_id>`
- `active_thread:<thread_id>`
- `pre_turn_scene_hint:<message_id>:...`
- `context_update_suggestion:<suggestion_id>[,<suggestion_id>...]`
- `world_state:<state_id>`
- `state_change:<state_change_id>`
- `memory:<memory_id>`
- `observation:<observation_id>`
- `summary:<summary_id>`
- `message:<message_id>`
- `media_asset:<media_asset_id>`

Diagnostics persist source type, source ID, tier, character count, inclusion
status, and reason. They do not persist prompt text.

Duplicate retrieval diagnostics report how many context-search selections were
covered by deterministic sources already in the prompt. Suppressed keys are
stored as `<source_type>:<source_id>` strings so operators can tell why an
indexed source was selected by search but omitted from retrieved prompt slots.

When the agentic context pipeline is enabled, a pre-narrator planner may add a
compact narration brief and evidence source IDs after retrieval. The narrator
still receives normal prose instructions; structured planning and verification
use provider-enforced structured-output requests.

## Pre-Turn Scene Hints

Pre-turn scene hints are deterministic narrator-only context. They are derived
from the current scene snapshot, known character registry, and the latest player
message before narrator generation. They can point out mentioned present
characters, mentioned known characters who are not currently present, and
mentioned current scene objects or hazards.

Hints are not persisted and do not update `world_state`, scene snapshots, audit
rows, suggestions, exports, or imports. Scene maintenance still persists only
after the narrator turn completes and validation accepts the update. Volatile
scene facts continue to use existing scene snapshot and `world_state` records;
a dedicated scene fact schema remains deferred until that storage shape becomes
awkward in practice.

## Dating Route Pacing

Dating-sim saves include deterministic `dating_route_state` anchors for present
romance-route participants, plus off-scene routes mentioned by the latest
player turn. These anchors are compact current-scene context built from typed
route state, not retrieval, memory, summary, or prompt-parsed model output.

## Deleted Messages

User-facing chronicle deletion is a soft delete: message rows keep `deleted_at`
for local audit/debugging, but normal chronicle, prompt assembly, context search,
and chat bundle export paths only use active messages. Deleting from a message
also archives directly sourced generated media metadata and derived context rows;
media files are not removed from disk by message deletion.

## Budget Modes

- `diagnostics_only`: include all assembled sources and report size metadata.
- `fixed_chars`: include ordered sources until `context_budget_fixed_total_chars`.
- `adaptive_tiers`: derive a limit from the fixed cap multiplied by
  `context_budget_adaptive_fraction`, preserving tier order.

The default is `diagnostics_only` so prompt behavior stays transparent while the
diagnostic layer exposes why prompts are large.

Final narrator requests still enforce the selected provider model's context
window when one is known. That hard guard trims optional prompt items before
core continuity, preserving current-scene context, open obligations, character
voice profiles, the latest source message, and the latest rolling summary ahead
of lower-priority retrieved snippets whenever possible.
