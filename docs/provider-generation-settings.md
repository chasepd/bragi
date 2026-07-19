# Provider Generation Settings

Bragi exposes a small set of user-configurable generation and automation
settings for normal future narrator chat, scene image, and context jobs.

## User-Configurable Settings

- Narrator chat temperature: opt-in. Applied only when the selected narrator
  chat model advertises `temperature` support.
- Narrator chat max output tokens: opt-in. Applied only when the selected
  narrator chat model advertises `max_output_tokens` support.
- Scene image dimensions: optional preset. Applied only when the selected image
  generation model advertises `image_dimensions` support.
- Image style preset: save-specific provider-agnostic prompt guidance that
  remains separate from native provider payload parameters.
- Agentic context pipeline: save-specific opt-in for structured observation,
  memory curation, narrator planning, and response verification around turns.
- Content rating: user-specific G, PG, PG-13, R, or Unrated filtering for
  generated narrator prose and media prompts. Adult accounts default to PG-13.
  Child accounts default to PG, may choose G or PG, and may receive PG-13 only
  through an admin grant.
- Fade to black: user-specific narrator classifier toggle, enabled by default
  for adult accounts and always enabled for child accounts.
- Venice media safe mode: provider-specific safety setting for Venice image
  generation and Venice video models that advertise safe mode support. The
  stored setting key remains `venice_image_safe_mode` for compatibility. It is
  enabled by default and configurable for adult accounts, but forced on for
  every child-initiated media request. Because Venice is the configured media
  provider with enforceable safe mode, child image and video requests require a
  Venice model; non-Venice primary and fallback paths fail closed.

Fallback chat and image requests are checked against the fallback model before
settings are applied. Media prompts are checked against the initiating user's
effective rating before either the primary or fallback provider is called.
Child media fallbacks are additionally limited to Venice with forced safe mode.
Unsupported settings are skipped rather than sent to the provider.

## Fixed Application Settings

Maintenance and support jobs keep their existing fixed parameters. This includes
context selection, observation extraction, memory curation, response planning,
response verification, state and memory extraction, context updates, cleanup,
scenario evolution, summaries, image prompt drafting, and outcome checks. Those
jobs are tuned for deterministic local state maintenance, so v1 exposes only the
pipeline toggle, not per-job generation parameters.

Provider-specific image quality or native style payloads are not exposed in v1.
Only shared dimensions are configurable because OpenRouter and Venice both have
stable adapter behavior for that setting.
