# Troubleshooting

Use the Settings > Diagnostics panel first. It refreshes independently from
settings configuration and shows metadata-only health signals for the current
save, background maintenance, recent web events, provider status, and runtime
performance.

## Slow Load Or Slow Turns

- Check Active Save Health for large recent transcript windows, stale pending
  suggestions, empty context search results, or failed continuity jobs.
- Check Performance for slow job, step, or model averages.
- Check Event Stream for repeated `web.request.completed` rows with long
  durations on the same route.

## Provider Failures

- Check Signal Board for provider authentication or retry warnings.
- Check Job Failures for failed maintenance jobs and retry summaries.
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
