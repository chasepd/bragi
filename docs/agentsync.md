# AgentSync Workflow

Bragi uses AgentSync to keep project-level agent instructions and portable
skills in sync across native agent formats. AgentSync is dry-run first; every
write path below has a matching preview command.

## Source Of Truth

| Resource type | Source of truth | Target policy |
| --- | --- | --- |
| Rules/context | `AGENTS.md` (Codex/`agents-md`) | Generate `CLAUDE.md` and `.cursor/rules/agentsync.md`. OpenCode reads `AGENTS.md` directly. |
| Skills | `.codex/skills/` | Generate Claude, Cursor, and OpenCode skill copies from Codex skills. |
| Subagents | None | Disabled until Bragi has portable subagents to sync. |
| Hooks | `.codex/hooks.json` and `.codex/hooks/` | Generate `.claude/settings.json`, `.cursor/hooks.json`, and `.opencode/plugins/agentsync-hooks.js`. |
| Commands | None | Disabled until Bragi has portable commands to sync. |

`.agentsync/config.toml` encodes this policy for project scope. It enables
rules, skills, and hooks; disables unsupported behavior-bearing resource types;
and targets non-Codex formats from the canonical Codex sources.

## Inspect

Run these before planning a sync:

```bash
agentsync doctor --check
agentsync scan --scope project
agentsync status --scope project
```

`scan` shows native files AgentSync can discover. `status` compares tracked
source and target checksums from `.agentsync/state.json`.

## Preview

Preview resource types separately so generated output is easy to review:

```bash
agentsync diff rules --from codex --to claude,cursor
agentsync sync rules --from codex --to claude,cursor --dry-run
agentsync sync skills --from codex --to claude,cursor,opencode --dry-run
agentsync sync hooks --from codex --to claude,cursor,opencode --dry-run
```

After the resource-specific previews are clean, the config-backed all-resource
preview should be boring:

```bash
agentsync sync --all --dry-run
```

Review the entire diff before writing. Hook previews should render all command
handlers. Cursor and OpenCode currently warn that `statusMessage` display
metadata is omitted; that is acceptable because the commands, matchers,
timeouts, and blocking behavior are still rendered. If AgentSync reports a
blocked conversion or omits executable behavior, stop and keep that resource
native-only.

## Write

After the dry-run diff is acceptable, write the generated files:

```bash
agentsync sync rules --from codex --to claude,cursor --write
agentsync sync skills --from codex --to claude,cursor,opencode --write
agentsync sync hooks --from codex --to claude,cursor,opencode --write
```

The equivalent config-backed write is:

```bash
agentsync sync --all --write
```

These commands update the generated native files and `.agentsync/state.json`.
Check in the source files, generated targets, and state together. Do not edit
generated target files by hand; update the source of truth and rerun AgentSync.

## Drift Check

Use this as the local drift gate:

```bash
agentsync status --scope project --check
```

It exits nonzero when a tracked source or generated target changes without a
matching sync. Current AgentSync may also list generated rule or hook targets as
untracked native resources; treat `ChangedSource`, `ChangedTarget`,
`MissingTarget`, blocked diagnostics, or a nonzero exit as the stale-config
signal.

For CI, install AgentSync first, then run the same check. Keep the normal Bragi
validation flow separate:

```bash
python3 .codex/tools/validate.py
```

## GitHub Workflows

Bragi has two AgentSync workflows. Both workflows are path-scoped so ordinary
application changes do not spend CI time on agent config drift checks.

- `.github/workflows/agentsync-drift.yml` runs `agentsync status --scope project
  --check` on pull requests and pushes to `main` when canonical agent config,
  generated agent targets, or AgentSync workflow files change.
- `.github/workflows/agentsync-auto-sync.yml` installs the pinned AgentSync CLI
  after pushes to `main` or manual dispatch. Push-triggered runs are limited to
  canonical agent config and AgentSync workflow changes. It uses `agentsync sync
  --all --write`, opens/updates an `agentsync/auto-sync` branch, and creates a
  PR when generated files drift. Generation runs with read-only repository
  permissions and uploads an allowlisted patch; a separate publish job gets
  write permissions only to apply that patch, push the sync branch, and manage
  the PR.

The auto-sync workflow runs with a write token, so keep it limited to trusted
events such as `push` to `main` and `workflow_dispatch`.

External action and install refs in these AgentSync workflows are pinned to
commit SHAs because the auto-sync job has write permissions. Update those pins
only in an intentional dependency-maintenance PR.

AgentSync may create temporary `*.bak` files while syncing. `.gitignore` keeps
those backups untracked, and Bragi validation fails if they remain under
agent-config paths because stale generated docs can mislead agents and tools.
