from __future__ import annotations

import re
from pathlib import Path


def test_readme_uses_reproducible_frontend_dependency_install() -> None:
    body = Path("README.md").read_text(encoding="utf-8")

    assert "npm ci --prefix frontend" in body
    assert "npm install --prefix frontend" not in body


def test_readme_explicitly_warns_that_default_lan_http_is_plaintext() -> None:
    body = Path("README.md").read_text(encoding="utf-8").casefold()

    assert "plaintext http" in body
    assert "session cookie" in body
    assert "trusted lan" in body


def test_brand_asset_docs_record_ai_only_portrait_provenance() -> None:
    body = Path("frontend/public/brand/README.md").read_text(encoding="utf-8")

    assert "AI-generated portrait reference" in body
    assert "no human-authored reference artwork" in body
    assert "MIT License" in body


def test_repository_has_open_source_policy_documents_and_issue_templates() -> None:
    required_paths = (
        Path("LICENSE"),
        Path("CONTRIBUTING.md"),
        Path("SECURITY.md"),
        Path("SUPPORT.md"),
        Path("CODE_OF_CONDUCT.md"),
        Path(".github/ISSUE_TEMPLATE/config.yml"),
        Path(".github/ISSUE_TEMPLATE/bug_report.yml"),
        Path(".github/ISSUE_TEMPLATE/feature_request.yml"),
    )

    assert all(path.is_file() for path in required_paths)
    assert "MIT License" in Path("LICENSE").read_text(encoding="utf-8")
    conduct = Path("CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    assert "https://github.com/chasepd/bragi/security/advisories/new" in conduct
    assert "contact options on their GitHub profile" not in conduct


def test_ci_actions_are_pinned_to_full_commit_shas() -> None:
    body = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    action_refs = re.findall(r"^\s*uses:\s*([^#\s]+)", body, flags=re.MULTILINE)

    assert action_refs
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", ref) for ref in action_refs)
