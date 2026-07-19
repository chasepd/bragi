from __future__ import annotations

import re
from pathlib import Path


def test_validation_runner_agent_config_is_removed() -> None:
    assert not (Path(".codex") / "agents" / "validation-runner.toml").exists()


def test_test_writer_agent_config_is_removed() -> None:
    assert not (Path(".codex") / "agents" / "test-writer.toml").exists()


def test_test_engineering_skill_replaces_test_writer_agent() -> None:
    skill_path = Path(".codex") / "skills" / "test-engineering" / "SKILL.md"
    skill_text = skill_path.read_text(encoding="utf-8")

    assert "name: test-engineering" in skill_text
    assert "Fake chat providers should return natural prose" in skill_text
    assert "Avoid Hollow Tests" in skill_text
    assert "called_once_with(...)" in skill_text
    assert "Use fakes that model behavior" in skill_text
    assert "tests/unit/<subsystem>/test_<module>.py" in skill_text
    assert "python3 .codex/tools/validate.py" in skill_text


def test_test_engineering_skill_describes_smart_pr_validation_policy() -> None:
    skill_path = Path(".codex") / "skills" / "test-engineering" / "SKILL.md"
    skill_text = re.sub(r"\s+", " ", skill_path.read_text(encoding="utf-8").lower())

    assert "full gate" not in skill_text
    assert "python3 .codex/tools/validate.py --changed" in skill_text
    assert "pull request ci" in skill_text
    assert "smart changed-file validation" in skill_text
    assert "manual workflow dispatch" in skill_text
    assert "non-pr push" in skill_text
    assert "merge commits" in skill_text
    assert "skip duplicate app validation" in skill_text
    assert "python3 .codex/tools/validate.py --full" in skill_text
    assert "broad-risk" in skill_text
    assert "dependencies" in skill_text
    assert "ci/hooks" in skill_text
    assert "persistence/schema portability" in skill_text
    assert "ci will not run before merge" in skill_text
