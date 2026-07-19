---
name: test-engineering
description: Use when writing, improving, or fixing Bragi tests. Provides project-specific testing expectations, fake-provider guidance, mirrored test naming, and validation workflow without requiring a separate testing agent.
---

# Bragi Test Engineering

Use this skill whenever you add, update, or debug tests in Bragi. You may edit
tests directly; no separate testing subagent or authorization hook is required.

## Core Rules

- Use test-driven development for application code when feasible: write or
  update the failing test first, then implement the production change.
- Prefer unit tests with fake providers. CI must not require real provider API
  keys, network calls, or live model behavior.
- Keep tests deterministic: stable names, stable ordering, no sleeps unless the
  behavior under test is explicitly timing-related.
- Test behavior and state changes, not incidental implementation details.
- Keep test fixtures as small as the behavior allows. If setup gets large, make
  the assertion especially clear.
- Preserve user and concurrent-agent edits. Do not revert unrelated changes in
  tests or production files.

## What Good Tests Look Like

A good Bragi test should make a real promise about user-visible behavior,
domain state, persistence, provider contracts, or UI/service coordination. The
test should fail for the bug it is meant to catch, not just prove that mocked
calls happened in the expected order.

Prefer tests that:

- Exercise the real production function or class under test with small fakes at
  the boundary.
- Assert observable outcomes: returned values, saved rows, emitted events,
  updated settings, generated prompts, queued images, or rendered UI state.
- Keep mocks at process edges: model providers, filesystem/media generation,
  clock/randomness, network, GTK widgets that are impractical to instantiate,
  and slow external services.
- Use fakes that model behavior, not scripts of expected method calls. A fake
  repository should store and return data; a fake provider should capture
  requests and return deterministic prose or typed structured data through the
  correct contract.
- Include negative or regression assertions when that is the point of the
  change, such as "does not call the provider without context" or "does not
  overwrite existing memories."

## Avoid Hollow Tests

Be suspicious when a test patches most of the unit under test, asserts only that
one mock was called, or duplicates the implementation line by line. That kind of
test can pass while the app is broken.

Avoid patterns like:

- Mocking the method whose behavior the test claims to verify.
- Mocking every collaborator when an in-memory fake or real value object would
  make the behavior observable.
- Asserting only `called_once_with(...)` for a complex workflow without checking
  the resulting state.
- Building expected values by copying the same transformation used in
  production; write the expected behavior independently.
- Using broad mocks that accept any attribute or return any type. Prefer typed
  fakes, small stub classes, or `spec_set` when a mock is the right tool.

If a unit test needs heavy mocking to cover an important path, add a narrower
unit test for the pure logic and consider an integration test for the workflow.

## Model Output Contract Tests

Bragi must not depend on prompt-only structured text from normal chat models.
Tests should fail production code that expects ordinary chat-text prompts to
emit JSON, YAML, XML, delimiter protocols, machine-readable lists, or other
schema-like formats.

- Fake chat providers should return natural prose by default.
- Only fake structured data when the production path uses a real typed
  structured-output/provider/tool API that enforces a schema outside ordinary
  model prose.
- If a test needs structured model output, make that explicit by using the
  structured-output provider contract, not a chat response string.

## Naming And Placement

Mirror source paths under `tests/unit/`:

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

Use the nearest useful `conftest.py` for shared unit fixtures, but do not rely on
`conftest.py` as the only coverage for a source file. Integration tests belong
under `tests/integration/` with deterministic flow names such as
`test_save_load_flow.py`.

## Validation

Do not run `pytest`, `ruff`, or `mypy` directly. Use the project validation
script:

```bash
python3 .codex/tools/validate.py
```

For iterative agent work, run validation in fast-fail order before running
expensive tests:

```bash
python3 .codex/tools/validate.py --lint-only
python3 .codex/tools/validate.py --typecheck-only
python3 .codex/tools/validate.py --tests-only
```

Use the smallest relevant `--tests-only` target when one is known. If any step
requires a code change, restart at lint so cheap lint/typecheck failures are
caught before expensive tests are rerun.

The default command runs smart changed-file validation for iterative agent work.
Run smart validation after the fast-fail loop passes and before PRs or handoff.
Pull request CI intentionally runs smart changed-file validation with
`python3 .codex/tools/validate.py --changed`; manual workflow dispatch and
applicable non-PR push/fallback paths run
`python3 .codex/tools/validate.py --full`. Normal merge commits skip duplicate
app validation. Run `--full` locally for broad-risk changes, dependencies,
CI/hooks, persistence/schema portability, or when CI will not run before merge:

```bash
python3 .codex/tools/validate.py --full
```

Useful focused variants:

```bash
python3 .codex/tools/validate.py --tests-only
python3 .codex/tools/validate.py --typecheck-only
python3 .codex/tools/validate.py --lint-only
```

Run smart validation near the end of code changes, after lint/typecheck/targeted
tests are clean, before PRs or handoff unless the task is purely
GitHub/admin-only.
