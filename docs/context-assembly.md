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
- `scene_fact:<fact_id>[,<fact_id>...]`
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
after the narrator turn completes and validation accepts the update.

After each narrator turn, provider-enforced structured context updates may also
record typed volatile scene facts such as positions, poses, object placement,
ongoing actions, physical constraints, environment state, line of sight, and
pending reactions. Physical facts remain active for the current scene; actions,
line of sight, and pending reactions expire after the following narrator turn.
Conflicting facts replace the prior value deterministically, while evidence and
source-message provenance remain auditable. Active scene facts are preserved by
snapshots, forking, and chat bundle export/import. Image context includes grounded
physical facts for the selected moment, excluding pending reactions, expired or
archived records, other scene generations, and facts with missing, unknown, or
future message provenance.

## Image Scene Grounding

Image context puts the selected message first, followed by confirmed participants,
physical scene facts, the eligible snapshot and location, and then older supporting
context. The selected message, complete relevant visual profiles, current scene
facts, and location remain protected when the context character budget is small.
Up to seven messages before the selected moment supply continuity; messages after
that moment never enter the transcript. If no source is specified, the latest
message becomes the selected moment.

`image_scene_characters(repositories, save_id, source_message_id=None)` supplies
one deterministic participant selection for image drafting, clothing, and
reference mappings. Per-message presence records take precedence over the scene
snapshot. Without presence records, the snapshot supplies participants only for
the latest message (or a save with no messages). A mentioned name never establishes
presence: other mentioned profiles are labeled as discussed/background context.
Empty presence and absent presence currently share the same persisted shape, so
both use that latest-message fallback.

Historical frames omit mutable character, location, and scene records with
updates after the source, or without reliable message provenance. They also omit
mutable linked lore and template world state rather than reconstructing old
versions. Current character appearance, visual notes, clothing, and age are omitted
for historical frames because registry edits and reference uploads can change
these fields without advancing their message provenance. The selected text still
supplies visible details. Historical reference images need eligible source-message
provenance and a creation timestamp at or before the selected moment; a solo image
reports an unavailable reference if the currently linked image cannot qualify.
The narrator's existing cutoff behavior is unchanged. Image profiles
keep appearance, visual notes, and current clothing in separate fields without
short-field clipping. Locations fall back to their general description when the
visual description is empty, explicitly asking the drafting model to extract
visible details only.

## Image prompt evaluation

Use this synthetic scenario to compare actual drafted prompts and resulting
images before and after changing prompt generation. The examples contain no
personal roleplay data. Live evaluation is optional and stays outside CI; unit
tests use fake providers returning ordinary prose.

### Fixture: the rain observatory

Create a disposable save with these adult characters:

| Character | Stable appearance | Current clothing |
| --- | --- | --- |
| Mira | Short adult human with copper curls, dark brown eyes, freckles, and a narrow crescent scar below her left eye. | Forest-green wool coat with brass toggles, dark trousers, and rain-spotted brown boots. |
| Oren | Tall adult humanoid with silver scales, a dark blue crest, amber eyes, and a chipped left horn. | Navy canvas coveralls with rolled sleeves and a leather tool belt. |
| Vale | Adult human with straight black hair and round glasses. Vale is away from the observatory and is only discussed. | Rust-red sweater. |

The observatory is a circular stone room. A barred east window overlooks rain.
A brass chest stands on a waist-high oak bench. A lit oil lamp sits on a shelf
behind the bench, leaving the foreground cooler than the background. A copper
door is closed on the west wall. Keep the visual description of this location
empty and put these details in its general description to exercise the fallback.

Use this selected narrator message:

> Mira kneels to the left of the bench and braces the open chest lid with her
> right palm. Oren stands to the right, lifting a blue glass key out of the chest
> with his left hand. His right hand rests flat on the bench. Mira watches the
> hinge with a concentrated frown; Oren looks down at the key with a small smile.
> Rain beads against the barred window. “Vale would recognize this,” Mira says.

Record matching physical scene facts: Mira's kneeling position, Oren standing
to the right, the key held in Oren's left hand, the closed west door, and the
lamp's shelf position. Add an expired fact about the chest being closed and a
pending reaction about Oren possibly turning toward the door. Neither belongs
in the selected image. A private note that Mira worries about a future visitor
must not create an additional figure.

### Cases

Run every case through the UI that normally invokes that image purpose. Keep
the same image model, image dimensions, style, and character references between
versions. Use a fixed image seed when the chosen provider supports one. Save
actual submitted prompts from the image prompt details alongside the images.

| Case | Input and setup | Required observations |
| --- | --- | --- |
| Manual scene | Generate an image from the selected narrator message. | Two subjects with distinct actions, correct key ownership, lid and bench interactions, and spatially coherent room details. Vale stays absent. |
| Automatic scene | Enable automatic images, queue the same moment, then continue narration so Oren leaves and Mira changes into a yellow raincoat before the queued job executes. | The prepared image retains Oren, Mira's green coat, original objects, selected style, and reference order. |
| Reference-assisted scene | Attach Oren's reference first and Mira's second; use references showing old outfits and different backgrounds. | Each input image anchors the correct identity; the new scene, actions, outfits, and lighting remain explicit. |
| Reusable character reference | Generate Mira's reference from her profile. | Copper curls, freckles, eye color and left-eye scar remain identifiable against a simple background; the chest-opening action is absent. |
| Solo character picture | Generate Oren's solo image while the scene contains both characters. | One visible character with silver scales, blue crest, chipped horn, and navy coveralls; Mira does not appear. |
| Selfie attachment | Mira sends “The copper door is behind me; the rain finally stopped” with a selfie. | Plausible selfie perspective and hands, recognizable Mira, a copper door behind her, and relevant surroundings. |
| Object attachment | Oren sends “Look at the blue glass key I found” with an object photo. | The key is the central subject with visible glass material and relevant supporting context; no replacement portrait or chat screenshot. |
| Historical scene | After changing the cast, outfits and location, generate an image from the earlier chest-opening message. | Earlier message evidence determines the moment; unproven current state and later events stay absent. |
| Long detail | Extend Oren's appearance with grounded descriptive prose and place the chipped horn detail at the end. | The distinctive detail reaches the drafting model and remains usable in the image prompt rather than disappearing at a short field cap. |
| Edited prompt | Edit the saved prompt to a precise short description and generate again. | The submitted text preserves the edit without an additional drafting or compression call. |

For a missing-outfit run, clear a character's clothing before preparing the
image. Check that a coherent inferred outfit appears in the prompt and is saved
only while the registry field remains blank and unlocked. Repeat after entering
or locking clothing during generation; that later registry edit must survive.

### Before and after review

1. On the baseline version, create and export the synthetic save. Run the cases
   and retain each actual submitted prompt and sample image in a local evaluation
   directory outside the repository. Record application revision, image provider
   and model, image-prompt model, dimensions, seed if available, and style.
2. Import a fresh copy of that save into the upgraded version. Use the same
   configuration and cases. Generate actual prompts with the configured Image
   Prompt text model and sample images with the configured image provider. Do
   not substitute fake-provider output or the prose in this document for the
   actual evaluation results.
3. Place baseline and upgraded prompts beside their corresponding images. For
   nondeterministic image models, inspect three samples per case to distinguish
   consistent prompt improvement from a single lucky image.
4. Score each prompt and each image separately using the rubric below. Record
   concrete evidence such as “Oren holds the key in his left hand” or “the image
   adds Vale despite the absent status.” Keep failures even when the image is
   visually attractive.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Scene fidelity | Wrong moment, setting, or event | Correct main event with omitted visible detail | Correct selected moment and relevant environment |
| Subject/action attribution | Missing, extra, or confused subjects and actions | Correct cast but ambiguous action ownership | Each subject's action, gaze, hands, and expression are distinct |
| Spatial clarity | Impossible or contradictory placement | Plausible but underspecified placement | Coherent relative positions, object placement, and interactions |
| Identity continuity | Wrong or inconsistent character identity | Some distinctive traits survive | Distinctive appearance and reference identities remain consistent |
| Unsupported invention | Consequential invented figures, actions, or facts | Minor distracting unsupported detail | Only conservative camera placement and incidental staging |

A successful prompt has no zero scores, no misplaced subject or owned object,
and preserves authoritative clothing and reference identity. Compare totals as
a secondary measure; extra words alone are not an improvement. Image failures
with a correct prompt should be recorded separately from drafting failures.

For provider-limit evaluation, choose a model with an advertised prompt limit
and a smaller configured fallback. Verify that the final submitted prompt stays
within the actual model's limit, authoritative instructions remain intact after
the single permitted compression retry, and the saved prompt equals the
successful request. Oversized manual prompts and failed compression should
produce an explicit length error. Save export/import must preserve prepared
briefs and the resulting prompt details. Never check in generated images,
provider keys, save databases, or live request logs.

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
