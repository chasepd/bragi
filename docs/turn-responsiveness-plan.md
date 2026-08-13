# Turn Responsiveness Program

Umbrella issue: [#129](https://github.com/chasepd/bragi/issues/129)

This document is the durable source of truth for Bragi's multi-PR turn
responsiveness program. Update it in every program PR so the work can resume
without depending on chat context.

## Locked Decisions

- Existing quality behavior remains the default.
- Add an opt-in, save-scoped `responsive` mode.
- Responsive mode may skip or combine quality helpers only when deterministic
  eligibility checks say the local context is sufficient.
- Content-rating safety, phrase and script guards, authentication, and
  deterministic state integrity are never weakened.
- Show an in-chronicle narrator placeholder, but never expose unchecked prose.
- Preserve the prior-turn continuity barrier. This program does not introduce
  a stale-state narration path.
- Evaluate live performance using privacy-safe aggregates from organic use. CI
  remains deterministic, keyless, and free of live provider calls.
- Do not record prompts, narration, save identifiers, personal scenario data,
  API keys, media, or local databases in issues, PRs, benchmarks, telemetry
  metadata, or this document.

## Success Gates

- Optimistic player and narrator-placeholder paint p95 is below 250 ms.
- SSE fallback detects terminal job completion within five seconds.
- At least 20 organic samples exist for both modes in the same narrator
  provider/model stratum.
- Responsive mode improves response-committed median by at least 25 percent and
  p95 by at least 20 percent against the matched quality baseline.
- Failed or interrupted turns increase by no more than five percentage points.
- All mandatory safety and deterministic continuity tests remain green.

If a gate fails, keep #129 open, record the result below, and request product
direction instead of weakening safety or adaptive eligibility rules.

## Stable Interfaces And Telemetry

The program owns these names. Changing one requires an entry in the decision
log before implementation.

### Settings

- `turn_responsiveness_mode`: portable save setting with `quality` and
  `responsive` values. Missing or invalid values resolve to `quality`.
- `provider_call_deadline_seconds`: existing global admin setting, made writable
  through the supported settings API and UI.

### HTTP

- `GET /api/chat/timing-summary?save_id=...`: authenticated aggregate timing for
  the effective responsiveness mode and current narrator provider/model. It
  returns no estimate until five matching successful turns exist.

### Job steps

- `chat.preflight`
- `chat.input_safety`
- `chat.history`
- `chat.context`
- `chat.character_planning`
- `chat.narrator_planning`
- `chat.narrator_generation`
- `chat.output_safety`
- `chat.verification`
- `chat.commit`
- `chat.response_committed`
- `post_turn.continuity_ready`
- `post_turn.image_ready`

Step metadata may contain only safe categorical or numeric values: effective
mode, provider/model, retry count, cache/refinement/fast-path flags, token
counts, and bounded reasoning/routing enums. It must never contain model input
or output text.

### Client events

- `client.chat.optimistic_player_painted`
- `client.chat.placeholder_painted`
- `client.chat.narrator_painted`

Existing SSE event names and the committed `chat_turn_delta` contract remain
compatible. This program does not add provisional narration events.

## PR Ledger

Statuses are `pending`, `in_progress`, `blocked`, or `complete`. Each PR must
update its row before merge and append evidence to the measurement or decision
log when applicable.

| PR | Branch | Status | Issue/PR | Required outcome |
| --- | --- | --- | --- | --- |
| 0 | `docs/turn-responsiveness-plan` | `complete` | [#130](https://github.com/chasepd/bragi/pull/130) | Persisted this execution plan and update protocol. |
| 1 | `feat/turn-latency-telemetry` | `complete` | [#131](https://github.com/chasepd/bragi/pull/131) | Added critical-path spans, user-paint events, and deterministic latency harnesses. |
| 2 | `feat/turn-progress-ux` | `complete` | [#132](https://github.com/chasepd/bragi/pull/132) | Added narrator placeholder, timing summaries, and job-delivery cleanup. |
| 3 | `fix/foreground-retry-budgets` | `complete` | [#133](https://github.com/chasepd/bragi/pull/133) | Enforced hard deadlines and responsive foreground retry limits. |
| 4 | `feat/responsive-turn-mode` | `in_progress` | #129 | Add portable save-scoped mode and responsive routing/budget behavior. |
| 5 | `perf/adaptive-turn-pipeline` | `pending` | #129 | Add deterministic fast path and combined structured planning path. |
| 6 | `perf/post-turn-media-responsiveness` | `pending` | #129 | Start images earlier and improve media loading feedback. |
| 7 | `docs/turn-responsiveness-results` | `pending` | #129 | Evaluate organic aggregates, document results, and close the program only if gates pass. |

## PR Requirements

### PR 1: Critical-path telemetry and deterministic benchmarks

- Record the stable job-step spans above without a persistence migration.
- Extend provider-stream telemetry with time to first chunk and output rate when
  streaming transport is used. Do not enable browser prose streaming.
- Emit metadata-only client paint events.
- Add a delayed fake or loopback integration harness for milestone ordering,
  provider-call wave counts, retries, and completion timing. Avoid API keys and
  wall-clock sleeps.
- Record the initial quality-mode organic baseline below.

### PR 2: Immediate feedback and job delivery

- Insert a narrator placeholder directly after the optimistic player entry.
  Drive its copy from existing phases, show elapsed time after three seconds,
  expose cancellation, and replace it atomically with committed narration.
- Add the timing-summary endpoint. Use the latest 30 successful matching turns
  and show a broad p50-p95 range only after five samples.
- Poll every two seconds after successful nonterminal fallback polls. Back off
  only after errors and cap that delay at five seconds.
- Apply valid `job_changed` payloads directly. Refetch global job/submission
  state only on chat-job creation, terminal transition, SSE recovery, or
  malformed payloads.
- Avoid media and world invalidations for chat deltas that did not change them.

### PR 3: Provider deadlines and foreground retries

- Add the existing deadline setting to settings policy, sanitization, API, and
  admin UI.
- Wrap every provider attempt in its remaining hard deadline and normalize
  expiry to the existing transient timeout category.
- Add an internal retry budget resolved from execution context:
  - Quality and background work retain configured retry behavior.
  - Responsive foreground work allows at most two provider attempts and a
    45-second hard deadline.
  - Responsive work does not replay the entire turn after exhausting that
    budget.
  - Responsive verification allows at most one regeneration.

### PR 4: Save-scoped responsive mode

- Add the portable setting and expose it to administrators and adult users in
  Save settings. Child users cannot change it.
- Ensure save forks, snapshots, export, and import preserve it.
- Quality mode must preserve existing behavior.
- Responsive mode uses PR 3's foreground retry budget, disables optional helper
  thinking, caps foreground structured helper output at 2,048 tokens, uses an
  8-player/8-narrator planner window, and requests OpenRouter latency sorting
  only when no explicit routing profile overrides it.
- Do not change narrator model choice or its configured output limit.

### PR 5: Adaptive critical-path reduction

- A responsive new-player turn is fast-path eligible only when all are true:
  - A valid precomputed narration snapshot exists.
  - Initial indexed retrieval reports strong local recall.
  - No unresolved or absent named character is referenced.
  - Prior continuity is ready and retrieval is not degraded.
  - The operation is not regenerate, edit, timeskip, look-around, or recovery.
- Eligible turns use deterministic candidates and skip character action
  planning, narrator planning, response verification, and NPC knowledge audit.
  They retain input/output safety, prompt budgeting, deterministic guards,
  narration persistence, and all post-turn state work.
- Ineligible responsive turns use one provider-enforced
  `responsive_turn_plan` structured call combining validated context-source
  selection with the typed narrator plan.
- Validate every model-selected source ID against application-built candidates.
- If the combined route is unavailable or invalid, use the existing quality
  helper path under the responsive retry budget.
- Never ask ordinary chat prose to emit structured data.

### PR 6: Post-turn and media responsiveness

- Launch prepared automatic image generation as soon as preparation completes
  and run it concurrently with independent continuity work while preserving
  pre-post-turn image semantics.
- Preserve accurate `response_committed`, `continuity_ready`, and
  `optional_enrichments_complete` ordering. Images remain optional and never
  block chat submission.
- Show a source-linked scene-arriving tile.
- Use thumbnails and asynchronous decoding in the sidebar; fetch the full asset
  only for the modal.
- Refetch media only for media-job transitions or media-changing deltas.

### PR 7: Organic-results closeout

- Wait for at least 20 privacy-safe samples per mode in a matched narrator
  provider/model stratum.
- Record sample counts, median, p95, failure rate, and fast/combined-path usage
  below without identifiers or content.
- Update troubleshooting and operator guidance.
- Close #129 only when every success gate passes.

## Execution Protocol

For every program PR:

1. Fetch `origin` and create a fresh sibling worktree and branch from the latest
   `origin/main`. Never reuse a prior PR worktree for new changes.
2. Set that PR's ledger status to `in_progress` in its first commit.
3. For application code, follow TDD and record the failing test and failure
   reason in the PR description or decision log.
4. Use the project `test-engineering` guidance for tests.
5. Validate in fast-fail order with `.codex/tools/validate.py`: lint,
   typecheck, targeted tests, then smart validation. Restart at lint after a
   code fix. Run full validation for provider retry, persistence/portability,
   or broad turn-pipeline changes.
6. Use the project `code-review` workflow. Critical and Important findings are
   blockers until fixed or rejected with technical evidence.
7. Update this document before merge with the PR link, status, validation and
   review evidence, aggregate results, deviations, and exact next action.
8. Open the PR with `Refs #129`, monitor CI until green and mergeable, and merge
   before creating the next PR worktree from updated `origin/main`.

## Measurement Log

| Date | Commit/PR | Mode and stratum | Samples | Median | p95 | Failure/interruption rate | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-08-12 | audit at `940d585` | quality, code audit only | 0 | n/a | n/a | n/a | No personal runtime data inspected. |
| 2026-08-12 | PR 1 ([#131](https://github.com/chasepd/bragi/pull/131)) | quality, instrumentation baseline | 0 | n/a | n/a | n/a | Telemetry begins with this PR, so no eligible pre-instrumentation samples exist. No personal runtime data or live provider was inspected; matched organic aggregates begin after deployment. |
| 2026-08-12 | PR 2 ([#132](https://github.com/chasepd/bragi/pull/132)) | quality, progress UX | 0 | n/a | n/a | n/a | The endpoint reads local privacy-safe aggregates only; implementation and tests used synthetic data, and no personal runtime data was inspected. |
| 2026-08-13 | PR 3 ([#133](https://github.com/chasepd/bragi/pull/133)) | quality and internal responsive retry policy | 0 | n/a | n/a | n/a | Provider deadlines and execution budgets were verified with deterministic fakes. Responsive mode is not yet user-selectable, so no organic comparison samples exist; no personal runtime data or live provider was inspected. |

## PR Evidence

- PR 0 ([#130](https://github.com/chasepd/bragi/pull/130)): documentation-only;
  runtime validation was skipped per `AGENTS.md`, `git diff --check` passed, and
  an independent pinned-SHA review found one process blocker. This update fixes
  that blocker by recording the PR, completed status, evidence, and next action.
  No aggregate runtime measurements apply to this PR and there were no product
  or interface deviations.
- PR 1 ([#131](https://github.com/chasepd/bragi/pull/131)), pinned implementation
  commit `414ed60`: red tests first exposed the missing span API, production
  planner and output-safety call-site coverage, paint delivery edge cases, and
  the absence of persisted retry counts. The final harness drives the real API,
  job registry, runtime, and `ChatService` path under a virtual clock; it proves
  two provider-call waves, one retry, 200 ms completion, critical-span ordering,
  and metadata privacy without API keys or wall-clock sleeps. Changed-file
  validation escalated to the full nine-phase suite and passed unit/web tests,
  integration tests, typechecking, lint, frontend tests, build, and audit. A
  pinned-SHA review scored the implementation 9.5/10 with no Critical or
  Important findings. The only planned interface deferral is
  `client.chat.placeholder_painted`, which remains in PR 2 so it cannot report a
  UI milestone before the placeholder exists.
- PR 2 ([#132](https://github.com/chasepd/bragi/pull/132)), pinned implementation
  commit `6806433`: backend red tests first exposed the absent timing service and
  repository query; frontend red tests exposed the old fallback cadence and
  absent narrator placeholder. Review regressions then reproduced an early SSE
  event race and unsafe progress projection before their fixes. The completed
  implementation paints an in-chronicle narrator placeholder immediately,
  enriches it with bounded progress, timing, and cancellation, replaces it
  atomically with committed narration, applies valid job changes directly,
  polls successful fallbacks every two seconds, and narrows chat-delta
  invalidation. Full nine-phase validation passed. A pinned-SHA review scored
  the implementation 9.5/10 with no Critical or Important findings. The only
  implementation-budget change was a small frontend compressed-size allowance
  increase for the new placeholder and cache behavior; no stable interface or
  product requirement deviated from the plan.
- PR 3 ([#133](https://github.com/chasepd/bragi/pull/133)), pinned implementation
  commit `60adad3`: red tests first exposed the missing deadline policy/API/UI,
  unbounded attempts and stream reads, raw timeout expiry, whole-turn replay,
  and uncapped response checks. Review regressions then reproduced an async
  generator timeout scope that cancelled its consumer, streaming fallback that
  multiplied the foreground provider budget, non-finite deadlines that could
  fail calls immediately, and separate response guards multiplying responsive
  regenerations. The completed implementation enforces a remaining hard
  deadline around every provider attempt and stream read, normalizes expiry as
  a transient provider timeout, limits responsive foreground work to two
  provider attempts and 45 seconds without whole-turn replay, and shares one
  narrator regeneration across script, phrase, verifier, and legacy NPC checks.
  Quality and background execution retain configured behavior, and mandatory
  guards still reject invalid output. Full nine-phase validation passed. A
  fresh pinned-SHA review scored the complete implementation 9.2/10 with no
  Critical, runtime Important, or Minor findings after fixes. Responsive
  foreground deliberately uses final-only non-streaming transport so a failed
  stream cannot open a fresh non-streaming provider budget; this does not alter
  the browser contract because provisional prose remains disabled. No other
  stable interface or product requirement deviated from the plan.

## Decision And Change Log

- 2026-08-12: Preserve quality behavior as the default and add responsive mode
  per save.
- 2026-08-12: Responsive mode uses adaptive helpers rather than routing-only or
  an unconditional minimal pipeline.
- 2026-08-12: Ship a narrator placeholder first and do not expose provisional
  prose in this program.
- 2026-08-12: Use relative performance gates plus a user-feedback SLO rather
  than an absolute provider-dependent turn target.
- 2026-08-12: Use organic aggregate samples rather than paid canaries.
- 2026-08-12: Cover the full audited program, including transport and media
  responsiveness.
- 2026-08-12: PR 1 emits paint telemetry for the existing optimistic player and
  committed narrator UI. The stable `client.chat.placeholder_painted` event is
  implemented in PR 2 with the placeholder that it measures; emitting it before
  that UI exists would create misleading telemetry.
- 2026-08-12: Successful and exhausted provider calls persist only numeric
  `attempt_count`, `retry_count`, and `max_attempts` from provider retry
  metadata. This makes provider-call waves measurable without persisting
  transport payloads or prose.
- 2026-08-12: Direct `job_changed` progress is projected through a bounded
  metadata allowlist before SSE delivery. Generated sections, error text, and
  arbitrary nested progress data never enter the client event.
- 2026-08-12: Raise the frontend gzip and Brotli size budgets narrowly to cover
  narrator-placeholder, timing-summary, and direct job-cache behavior. The
  normal build-size check remains enforced.
- 2026-08-13: Treat responsive verification as one shared narrator-regeneration
  allowance across deterministic script and phrase guards, structured
  verification, and the legacy NPC audit. Exhausting that allowance never
  bypasses a guard; invalid output is still rejected or marked suspicious under
  the configured audit policy.
- 2026-08-13: Use non-streaming provider transport for responsive foreground
  narration. Bragi still delivers final-only prose, and this keeps streaming
  failure plus non-streaming fallback inside one two-attempt/45-second budget.
- 2026-08-13: Invalid non-finite provider deadlines resolve to the safe default,
  matching other malformed persisted setting values.

## Exact Next Action

After PR 3 merges, fetch the resulting `origin/main` and create a fresh sibling
worktree on `feat/responsive-turn-mode` for PR 4. Mark PR 4 in progress in the
first commit, then begin with red tests for the portable save-scoped
`turn_responsiveness_mode` setting, permissions, forks, snapshots, export, and
import. Add quality/responsive routing so quality remains unchanged while
responsive foreground work selects PR 3's retry budget, disables optional
helper thinking, caps structured helper output at 2,048 tokens, uses an
8-player/8-narrator planner window, and requests OpenRouter latency sorting only
when no explicit routing profile overrides it. Do not change the narrator model
or its configured output limit.
