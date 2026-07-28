from __future__ import annotations

import re
from pathlib import Path


def test_agents_model_output_contract_allows_real_schema_enforcement() -> None:
    body = _agents_text()

    assert "normal chat-text prompt" in body
    assert "typed provider" in body
    assert "structured-output apis" in body
    assert "tool/function-call" in body
    assert "real schema enforcement" in body


def test_agents_model_output_contract_forbids_prompted_json_parsing() -> None:
    body = _agents_text()

    assert "ask the model for json and parse it" in body
    assert "architecture anti-pattern" in body
    assert "prompt-only model-authored json" in body
    assert "do not fake structure by prompting a text model harder" in body


def test_agents_describes_context_search_as_schema_enforced_selection() -> None:
    body = _agents_text()

    assert "schema-enforced context selection" in body
    assert "minimum relevant local context by source id" in body
    assert "llm-based context-search agent" not in body
    assert "normal chat-text prompt" in body


def test_agents_points_validation_to_script_without_mimir_worktree_reference() -> None:
    body = _agents_text()

    assert "python3 .codex/tools/validate.py" in body
    assert "../mimir" not in body


def test_agents_leads_with_worktree_and_pr_publication_guidance() -> None:
    body = _agents_text()

    tdd_index = body.index("use test-driven development")
    assert body.index("dedicated task worktree") < tdd_index
    assert body.index("origin/main") < tdd_index
    assert body.index("when work is completed, open a pr") < tdd_index
    assert body.index("closes #") < tdd_index
    assert body.index("refs #") < tdd_index


def test_agents_describes_smart_pr_validation_policy() -> None:
    body = _agents_text()

    assert "full gate" not in body
    assert "python3 .codex/tools/validate.py --changed" in body
    assert "pull request ci" in body
    assert "smart changed-file validation" in body
    assert "manual workflow dispatch" in body
    assert "non-pr push" in body
    assert "merge commits" in body
    assert "skip duplicate app validation" in body
    assert "python3 .codex/tools/validate.py --full" in body
    assert "broad-risk" in body
    assert "dependencies" in body
    assert "ci/hooks" in body
    assert "persistence/schema portability" in body
    assert "ci will not run before merge" in body


def test_agents_auth_decision_matches_authenticated_web_app() -> None:
    body = _agents_text()

    assert "authentication is required by default" in body
    assert "first-admin bootstrap" in body
    assert "public unauthenticated routes" in body
    assert "unauthenticated lan-only" not in body


def _agents_text() -> str:
    return re.sub(r"\s+", " ", Path("AGENTS.md").read_text(encoding="utf-8").lower())
