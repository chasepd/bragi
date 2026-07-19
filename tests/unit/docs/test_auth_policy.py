from __future__ import annotations

import re
from pathlib import Path


def test_auth_policy_defines_expected_roles_and_boundaries() -> None:
    body = _policy_text()

    for role in ("admin", "user", "child"):
        assert f"`{role}`" in body

    assert "authentication is required by default" in body
    assert "first active admin is created" in body
    assert "httponly cookie" in body
    assert "reverse proxies" in body
    assert "provider keys are global" in body
    assert "owned or assigned saves" in body
    assert "frontend-only enforcement is never sufficient" in body
    assert "bragiruntime.active_save_id" in body
    assert "bragi_web_bootstrap_token" in body
    assert "throttled by source address and username" in body
    assert "at least 12 characters" in body
    assert "character bundle private-note export is admin-only" in body


def test_auth_policy_categorizes_every_api_route() -> None:
    body = _policy_text()

    missing = sorted(route for route in _api_routes() if route not in body)

    assert missing == []


def test_deployment_docs_describe_authenticated_operation() -> None:
    for path in ("README.md", "docs/docker-compose.md"):
        body = _document_text(path)

        assert "authentication is required by default" in body
        assert "first admin" in body
        for role in ("admin", "user", "child"):
            assert role in body
        assert "bragi_session" in body
        assert "bragi_web_secure_cookies" in body
        assert "bragi_web_bootstrap_token" in body
        assert "bragi_web_allowed_hosts" in body
        assert "bragi_web_allowed_origins" in body
        assert "claimed by the first admin" in body
        assert "at least 12 characters" in body
        assert "docs/auth-policy.md" in body
        assert "do not expose" in body


def _policy_text() -> str:
    return _document_text("docs/auth-policy.md")


def _document_text(path: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        Path(path).read_text(encoding="utf-8").casefold(),
    )


def _api_routes() -> set[str]:
    source = Path("bragi_web/api/app.py").read_text(encoding="utf-8")
    return {
        match.group("path").casefold()
        for match in re.finditer(
            r"@app\.(?:get|post|put|patch|delete)\(\"(?P<path>/api/[^\"]+)\"",
            source,
        )
    }
