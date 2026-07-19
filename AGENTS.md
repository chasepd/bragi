# AGENTS.md

This file gives agents the repo-level context needed to work on Bragi.

## Project

Bragi is a trusted-LAN web app for AI-powered roleplaying. The app presents a
chat-style roleplay chronicle while maintaining deterministic world state,
memories, summaries, saves, and generated scene images.

Read these first:

- `README.md`

## Required Workflow

- Use test-driven development for application code.
- Use the project-local `code-review` skill in `.codex/skills/code-review/`
  to self-review code after an agent finishes code changes and before opening a PR.
- Start feature work from `main`.
- Use the main repository worktree for agent-owned code changes when it is clean
  and no other agent is active there.
- If the main worktree is dirty, has unrelated edits, or another agent is active
  there, create a separate git worktree for the task. Prefer a sibling directory
  named for the task or branch, for example `../bragi-issue-52`, created from
  `main`.
- Keep agent worktrees isolated: do not edit the same checkout from multiple
  agents at once, and do not share one worktree across unrelated tasks.
- Before starting in an existing checkout, check `git status --short`; if there
  are unrelated edits or another agent is active there, create or switch to a
  dedicated worktree instead of mixing changes.
- Worktrees may live under `/tmp` when that makes Codex hook/tool path handling
  cleaner, for example `/tmp/bragi-issue-52`, but still create them from `main`
  and keep them task-specific.
- When editing a sibling or `/tmp` worktree, apply manual patches from that
  worktree root and use repository-relative paths such as `bragi/app.py`.
  Do not pass absolute paths or `..` paths to `apply_patch`; the PreToolUse hook
  rejects patch paths like `/tmp/bragi-issue-52/bragi/app.py` and
  `../bragi-issue-52/bragi/app.py` even when the target worktree itself is valid.
- Open feature PRs into `main`.
- Prefer coherent larger PRs over many tiny PRs because the validation pipeline
  is long. Bundle related fixes or features that can be reviewed together, while
  still avoiding unrelated churn or risky scope creep.
- When work is completed, open a PR unless the user explicitly says not to.
- After submitting a PR, agents should monitor it until CI is green and the PR
  is mergeable, addressing failures or merge blockers that are in scope.
- Keep the project backlog in GitHub issues.
- When opening, creating, or editing GitHub issues, preserve user privacy:
  do not include personal details, scrub personal roleplay data, and genericize
  any sample data while keeping enough detail to communicate the issue intent.
- When a PR closes a GitHub issue, mention that issue in the PR description.
- Do not commit API keys, generated local databases, logs, or untracked media output.
- Preserve user changes in the worktree. Do not revert unrelated edits.
- When adding or changing persisted schemas, structured data, or filesystem data
  that belongs to a save, update the export/import functionality in the same
  change so saves remain portable and restorable.
- Use `docs/agentsync.md` for agent configuration sync work. `AGENTS.md` and
  `.codex/skills/` are the source of truth for synced project rules and skills;
  after changing them, preview AgentSync output before writing generated target
  files.

Do not run pytest, ruff, or mypy directly. The local PreToolUse hook blocks raw
validation commands and routes agents to the project validation script:
`python3 .codex/tools/validate.py`.

After making code changes, use the validation script in fast-fail order before
running expensive tests:

1. Run lint with `python3 .codex/tools/validate.py --lint-only`.
2. Run typechecks with `python3 .codex/tools/validate.py --typecheck-only`.
3. Run targeted tests with `python3 .codex/tools/validate.py --tests-only ...`
   or the smallest relevant validation target.

If any step requires a code change, restart at step 1 so cheap lint/typecheck
failures are caught before expensive tests are rerun. After that loop is clean,
run smart validation before PRs or handoff. By default the validation script runs
smart changed-file validation: queued edit tests plus checks inferred from the
current git diff. Pull request CI intentionally runs smart changed-file
validation with `python3 .codex/tools/validate.py --changed`. Full validation
with `python3 .codex/tools/validate.py --full` runs for manual workflow
dispatch, applicable non-PR push/fallback paths, and local full-validation
runs. Normal merge commits skip duplicate app validation. Use local `--full`
when changing broad-risk files, dependencies, CI/hooks,
persistence/schema portability, or when CI will not run before merge. Use
`--tests-only`, `--typecheck-only`, `--lint-only`, or `--frontend-only` for
focused reruns.
Skip validation entirely for tasks that do not affect repo files or runtime
behavior, such as GitHub issue/admin-only work.

## Project Skills

- `code-review`: Project-local copy lives at `.codex/skills/code-review/`.
  Use it to review completed code changes before publishing or updating a PR.
  Treat Critical and Important findings as blockers until fixed or explicitly
  rejected with technical reasoning.
- `test-engineering`: Project-local copy lives at
  `.codex/skills/test-engineering/`. Use it whenever writing, improving, or
  fixing tests. Agents may edit tests directly after loading this skill; no
  separate testing subagent or authorization hook is required.

## Current Architecture Decisions

- UI stack: FastAPI/uvicorn backend with a React/Vite/TypeScript frontend.
- Platform: Trusted LAN/private web deployment first.
- Storage: SQLite for structured data and filesystem storage for generated media,
  using the existing `bragi-web` XDG directory and `BRAGI_WEB_*` environment
  names.
- Auth: authentication is required by default. Public unauthenticated routes are
  limited to static assets, health, login/logout, current-session checks, and
  first-admin bootstrap; keep the trusted-LAN/private deployment assumption.
- Providers: Venice.ai and OpenRouter behind one provider abstraction.
- Media scope: image generation is supported; video generation is not exposed
  until a real provider strategy is chosen.
- Scenario creation: AI-assisted wizard with manual review and editing.
- Context retrieval: schema-enforced context selection before every narrator turn, with the configured model selecting the minimum relevant local context by source ID.

## Model Output Contract

- Never require a normal chat-text prompt to emit structured JSON, YAML, XML,
  delimiter protocols, machine-readable lists, or any other schema-like format.
- Treat "ask the model for JSON and parse it" as an architecture anti-pattern
  when structure is enforced only by prompt wording. Models may return prose,
  and Bragi should either use that prose directly or make deterministic
  application-side decisions before asking for plain text.
- Structured application data must come from application code, typed provider
  APIs, structured-output APIs, tool/function-call APIs with real schema
  enforcement, or deterministic local transforms. Do not fake structure by
  prompting a text model harder.
- Tests should fail when production code depends on prompt-only model-authored
  JSON or other structured text. Fake chat providers should return natural text;
  fake structured-output providers may return typed structured data because the
  provider API, not the prompt, guarantees structure.

## Testing Expectations

- Unit tests should use fake providers by default.
- CI must not require real provider API keys.
- Tests should be named deterministically so hooks can map edited files to related tests.
- Follow the hook-compatible naming guidance below.

### Hook-Compatible Test Naming

The Codex edit hook maps edited Python source files under `bragi/` to mirrored unit test files under `tests/unit/`.

Use this pattern:

```text
bragi/<subsystem>/<module>.py
tests/unit/<subsystem>/test_<module>.py
```

Examples:

```text
bragi/services/chat_service.py
tests/unit/services/test_chat_service.py

bragi/providers/contracts.py
tests/unit/providers/test_contracts.py

bragi/persistence/save_repository.py
tests/unit/persistence/test_save_repository.py
```

Rules:

- Keep test module paths mirrored to source module paths below `bragi/`.
- Name test files with the `test_<source_module>.py` prefix.
- Put shared unit fixtures in the nearest useful `conftest.py`, but do not rely on `conftest.py` as the only coverage for a source file.
- `__init__.py` files are not expected to have one-to-one mirrored tests.
- If a source file has no mirrored test yet, the hook falls back to `tests/unit/<subsystem>/` when that directory exists. If neither a mirrored test nor subsystem test directory exists, the hook queues nothing; add the mirrored test file when introducing behavior.
- Integration tests should remain under `tests/integration/` and use deterministic flow names such as `test_save_load_flow.py` or `test_scenario_wizard_flow.py`.

## Local Hook Expectation

This repo includes Codex-format hooks that queue associated tests after edits and route raw validation commands to `.codex/tools/validate.py`. Run smart validation before finishing a code change. Pull request CI repeats smart changed-file validation with `python3 .codex/tools/validate.py --changed`; manual workflow dispatch and applicable non-PR push/fallback paths run `python3 .codex/tools/validate.py --full`. Normal merge commits skip duplicate app validation. Run local `--full` for broad-risk changes, dependencies, CI/hooks, persistence/schema portability, or when CI will not run before merge.
