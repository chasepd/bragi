# Troubleshooting

Use the Settings > Diagnostics panel first. It refreshes independently from
settings configuration and shows metadata-only health signals for the current
save, background maintenance, recent web events, provider status, and runtime
performance.

## Slow Load Or Slow Turns

- Bragi favors answer quality and completion over short latency. Provider calls,
  model-output repairs, and deferred AI work normally receive one initial
  attempt plus six retries. Admins can tune the shared retry count in Settings
  > Models > Retry policy; higher values increase provider usage and latency.
  Provider and background-work failures also use progressively longer delays, so
  a degraded provider can make a turn or maintenance task take substantially
  longer than a single request.
- Check Active Save Health for large recent transcript windows, stale pending
  suggestions, empty context search results, or failed continuity jobs.
- Check Performance for slow job, step, or model averages.
- Check Event Stream for repeated `web.request.completed` rows with long
  durations on the same route.
- For an adult or administrator-operated save, use Save settings > Turn
  Responsiveness Mode to compare the default `quality` pipeline with the
  bounded `responsive` pipeline. Responsive mode keeps all mandatory safety and
  continuity checks; turns that are not eligible for its fast or combined path
  use the standard helper route under the responsive retry budget.

## Measuring Turn Responsiveness Safely

The authenticated
`GET /api/chat/timing-summary?save_id=<save-id>` endpoint returns metadata-only
aggregates for the save's current responsiveness mode and configured narrator
provider/model. It never returns prompts, narration, scenario details, message
ids, job ids, or save ids. Use the response as the source for performance
comparisons instead of reading a local database or collecting chat logs.

The response fields use two independent, bounded windows:

- `sample_count` and `estimate` describe the latest 30 successful
  response-committed turns. `estimate` contains nearest-rank `p50_ms` and
  `p95_ms`, and remains `null` until five matching successes exist.
- `outcomes` describes the latest 30 terminal durable new-turn submissions:
  player turns, story continuations, and timeskips. Recovery retries,
  regenerations, and edit jobs are intentionally outside this comparison
  window. `failure_rate` is
  `(failed_count + interrupted_count) / terminal_count`.
- Route counts cover successful turns with route telemetry.
  `fast_path_count` is the deterministic fast path,
  `combined_path_count` is the provider-enforced combined structured planner,
  and `standard_path_count` is the normal quality path or the responsive
  fallback path. `unclassified_success_count` identifies older successful turns
  recorded before route telemetry existed.

For a trustworthy comparison:

1. Use the same executed narrator provider and model for both modes. Bragi
   records that stratum when the turn pipeline actually runs, so a settings
   change while a queued turn waits does not relabel the eventual execution.
   Do not combine different provider/model strata.
2. Let normal use accumulate at least 20 successful turns in each mode. Either
   use comparable saves or switch one save's mode; the endpoint reports only
   the currently effective mode when requested.
3. Recheck the endpoint in each mode and record only mode, provider/model,
   sample counts, p50, p95, failure rate, and route counts. Never copy the
   request's save id or add transcript, prompt, local database, log, or media
   data to a report.
4. Calculate improvement as `(quality - responsive) / quality`. The turn
   responsiveness program requires at least 25% median improvement, at least
   20% p95 improvement, and no more than a five-percentage-point increase in
   failed or interrupted turns.
5. In Settings > Diagnostics > Event Stream, filter the metadata-only client
   events for `client.chat.optimistic_player_painted` and
   `client.chat.placeholder_painted`. Record only the sample count and
   nearest-rank p95 of `duration_ms`; both p95 values must remain below 250 ms.
   Recent events are process-local, so collect them before a restart and do not
   export the raw rows.

If there are fewer than 20 successful samples in either mode, provider/model
does not match, or successful responsive turns are still unclassified, record
the comparison as pending. Do not infer a result from a small or mixed sample.
If latency improves but failures exceed the gate, keep quality mode as the
default and investigate provider and active-save health rather than weakening
safety or continuity rules.

## Provider Failures

- Check Signal Board for provider authentication or retry warnings.
- Check Job Failures for failed maintenance jobs and retry summaries.
- Repeated retries can increase provider usage and cost. Use diagnostic response
  verification only while investigating a verifier problem; it records findings
  without regenerating the narrator response and is not the normal quality-first
  mode.
- Re-enter provider keys from Settings > Providers when storage or
  authentication warnings are present.

## Maintenance Backlog

- Check Scheduler Health for failed, overdue, or leased tasks.
- Failed scheduler rows show only metadata such as task type, save id, last job
  id, failure count, redacted error, and next run time.
- If a task repeatedly fails, resolve the provider/settings warning first, then
  let the scheduler retry.

## Media Issues

- Expand a job card to see its origin, source records, provider/model, timing,
  retry state, and redacted failure details. Admins can open the full structured
  request detail when it was captured.
- Check Job Failures for image generation or media cleanup jobs.
- Check Event Stream for failed API calls from the media panel.
- Raw provider responses, API keys, and image bytes are not included in the
  diagnostics panel or support bundle. Admin job detail may include a bounded,
  redacted captured prompt and structured provider fields; older jobs may only
  have reconstructed metadata.

## Regenerate Turns Hang On A Reasoning Model

Some chat models are reasoning models that consume the configured
`max_output_tokens` budget on internal reasoning before emitting any visible
text. When the reasoning budget exceeds the output budget the model returns an
empty or near-empty response, the structured-output validation fails, and Bragi
retries the same call several times before surfacing an error. The result is a
regenerate turn that appears stuck for several minutes.

Symptom: `Chat Regenerate` stays in the "running" state for more than a minute
and the diagnostics panel shows repeated
`provider.generate_structured_output` or `provider.chat` steps for the same
model, often with `finish_reason: "length"` and `reasoning_tokens` close to the
output budget.

Verify in the running container or host process:

1. Inspect `job_steps` for the active regenerate job. Look for the
   `provider_error: Provider returned a reasoning-only response with no visible
   assistant text…` or `structured_output_invalid: Structured response was
   truncated; reasoning consumed the output budget before any visible JSON was
   emitted…` messages, and for completion token counts of one or two digits.
2. Inspect the open TCP connections from the Bragi process. A single
   `ESTABLISHED` HTTPS connection to a provider host with all other provider
   connections in `CLOSE_WAIT` means the process is blocked on a single
   in-flight call that cannot be cancelled until the upstream responds or the
   per-call deadline fires.

Mitigations, in order of safety:

1. Cancel the running job from the workbench. Bragi surfaces a
   "This is taking longer than usual" hint after sixty seconds and a
   "Cancelling…" hint after the cancel request. The provider call itself
   cannot be interrupted, but the global per-call deadline (default 120
   seconds) bounds the total wall time across all retry attempts.
2. Open Settings → Models and set `chat_fallback` and
   `structured_output_fallback` to a different model than the primary. If the
   primary and fallback resolve to the same provider and model, the fallback
   cannot help when the primary fails. The recommended extraction fallback is
   Venice `qwen3-5-9b`.
3. Verify that `model_thinking_preferences` is actually disabling reasoning
   for the model. The setting translates to `effort: "none"` on both
   providers, and some providers will silently ignore it if the model's
   capability discovery marks reasoning as mandatory. If reasoning cannot be
   disabled for the model, switch the task to a non-reasoning model.
4. Increase `chat_max_output_tokens` to at least the model's documented
   reasoning budget. Models that need thousands of reasoning tokens should
   have a `max_output_tokens` of 4096 or higher to leave room for a visible
   response.
5. Lower `provider_call_deadline_seconds` (default 120) to make misbehaving
   calls fail faster. The deadline is independent of `retry_count` and bounds
   the total wall time across all attempts, not the per-attempt timeout.

## Support Bundle Privacy

The Diagnostics panel can copy or download a support bundle. Review it before
sharing. It is intended to contain metadata only: request ids, routes, job ids,
save ids, task names, durations, statuses, redacted errors, and aggregate
counts. Do not add chat transcripts, prompts, private roleplay notes, provider
payloads, API keys, local database files, logs, or generated media when filing
issues.

## Backups

Before major maintenance or upgrades, back up the Bragi data directory or Docker
volume. For Docker Compose deployments, see `docs/docker-compose.md` for the
`bragi-data` backup command.
