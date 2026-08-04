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
