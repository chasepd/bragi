# Image prompt evaluation

Use this synthetic scenario to compare actual drafted prompts and resulting
images before and after changing prompt generation. The examples contain no
personal roleplay data. Live evaluation is optional and stays outside CI; unit
tests use fake providers returning ordinary prose.

## Fixture: the rain observatory

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

## Cases

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

## Before and after review

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
