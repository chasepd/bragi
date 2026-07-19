from __future__ import annotations

import importlib.util
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

_TARGETS_PATH = (
    Path(__file__).parents[3] / ".codex" / "tools" / "validation_targets.py"
)
_SPEC = importlib.util.spec_from_file_location("validation_targets", _TARGETS_PATH)
assert _SPEC is not None
targets = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = targets
_SPEC.loader.exec_module(targets)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'bragi'\n")
    (repo / "bragi").mkdir()
    return repo


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_select_changed_validation_maps_backend_source_to_targeted_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    source = repo / "bragi" / "providers" / "contracts.py"
    test = repo / "tests" / "unit" / "providers" / "test_contracts.py"
    _write(source)
    _write(test)
    monkeypatch.setattr(
        targets,
        "changed_files",
        lambda _repo: [source.relative_to(repo)],
    )

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is False
    assert selection.test_targets == (test.resolve(),)
    assert selection.lint_targets == (source.resolve(),)
    assert selection.typecheck is True
    assert selection.frontend is False


def test_select_changed_validation_maps_service_helper_to_explicit_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    source = repo / "bragi" / "services" / "post_turn_inference.py"
    explicit_test = repo / "tests" / "unit" / "services" / "test_chat_service.py"
    subsystem_test = repo / "tests" / "unit" / "services" / "test_other.py"
    _write(source)
    _write(explicit_test)
    _write(subsystem_test)
    monkeypatch.setattr(
        targets,
        "changed_files",
        lambda _repo: [source.relative_to(repo)],
    )

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is False
    assert selection.test_targets == (explicit_test.resolve(),)
    assert selection.typecheck is True


def test_select_changed_validation_maps_message_reconciliation_to_controller_tests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    source = repo / "bragi" / "services" / "message_reconciliation_service.py"
    test = repo / "tests" / "unit" / "application" / "test_controller.py"
    _write(source)
    _write(test)
    monkeypatch.setattr(
        targets,
        "changed_files",
        lambda _repo: [source.relative_to(repo)],
    )

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is False
    assert selection.test_targets == (test.resolve(),)
    assert selection.typecheck is True


def test_select_changed_validation_maps_auth_throttle_to_api_tests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    source = repo / "bragi_web" / "auth_throttle.py"
    test = repo / "tests" / "web" / "test_api.py"
    _write(source)
    _write(test)
    monkeypatch.setattr(
        targets,
        "changed_files",
        lambda _repo: [source.relative_to(repo)],
    )

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is False
    assert selection.test_targets == (test.resolve(),)
    assert selection.typecheck is True


def test_select_changed_validation_maps_bragi_common_source_to_targeted_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    source = repo / "bragi_common" / "media_mime.py"
    test = repo / "tests" / "unit" / "bragi_common" / "test_media_mime.py"
    _write(source)
    _write(test)
    monkeypatch.setattr(
        targets,
        "changed_files",
        lambda _repo: [source.relative_to(repo)],
    )

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is False
    assert selection.test_targets == (test.resolve(),)
    assert selection.lint_targets == (source.resolve(),)
    assert selection.typecheck is True
    assert selection.frontend is False


def test_select_changed_validation_merges_and_dedupes_queued_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    source = repo / "bragi" / "providers" / "contracts.py"
    test = repo / "tests" / "unit" / "providers" / "test_contracts.py"
    _write(source)
    _write(test)
    monkeypatch.setattr(
        targets,
        "changed_files",
        lambda _repo: [source.relative_to(repo)],
    )

    selection = targets.select_changed_validation(repo, [test])

    assert selection.test_targets == (test.resolve(),)


def test_select_changed_validation_runs_frontend_for_frontend_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    frontend_source = repo / "frontend" / "src" / "main.tsx"
    _write(frontend_source)
    monkeypatch.setattr(
        targets,
        "changed_files",
        lambda _repo: [frontend_source.relative_to(repo)],
    )

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is False
    assert selection.frontend is True
    assert selection.test_targets == ()
    assert selection.typecheck is False


def test_select_changed_validation_escalates_for_broad_risk_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    lockfile = repo / "uv.lock"
    _write(lockfile)
    monkeypatch.setattr(targets, "changed_files", lambda _repo: [Path("uv.lock")])

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is True
    assert selection.reasons == ("uv.lock",)


@pytest.mark.parametrize(
    "changed_path",
    [
        Path(".dockerignore"),
        Path("Dockerfile"),
        Path("bragi_build_backend.py"),
        Path("compose.yaml"),
        Path("docker-compose.yml"),
        Path("MANIFEST.in"),
        Path("Makefile"),
        Path("scripts/restart-static"),
        Path("frontend/package.json"),
    ],
)
def test_select_changed_validation_escalates_for_packaging_and_lock_drift_files(
    changed_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setattr(targets, "changed_files", lambda _repo: [changed_path])

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is True
    assert selection.reasons == (changed_path.as_posix(),)


@pytest.mark.parametrize(
    "changed_path",
    [
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path(".agentsync/config.toml"),
        Path(".claude/settings.json"),
        Path(".codex/hooks/run_tests_for_edited_file.py"),
        Path(".codex/skills/code-review/SKILL.md"),
        Path(".cursor/hooks.json"),
        Path(".opencode/plugins/agentsync-hooks.js"),
        Path(".github/workflows/agentsync-drift.yml"),
    ],
)
def test_select_changed_validation_escalates_for_agentsync_paths(
    changed_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setattr(targets, "changed_files", lambda _repo: [changed_path])

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is True
    assert selection.reasons == (changed_path.as_posix(),)


def test_select_changed_validation_escalates_for_validation_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    source = repo / ".codex" / "tools" / "validation_targets.py"
    _write(source)
    monkeypatch.setattr(
        targets,
        "changed_files",
        lambda _repo: [source.relative_to(repo)],
    )

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is True
    assert selection.reasons == (".codex/tools/validation_targets.py",)


def test_changed_files_uses_validation_base_ref_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    calls: list[list[str]] = []

    def fake_git_lines(_repo: Path, args: list[str]) -> list[str]:
        calls.append(args)
        if args == ["merge-base", "HEAD", "abc123"]:
            return ["merge-base-sha"]
        if args == ["diff", "--name-only", "merge-base-sha", "HEAD"]:
            return ["bragi/app.py"]
        return []

    monkeypatch.setenv("BRAGI_VALIDATION_BASE_REF", "abc123")
    monkeypatch.setattr(targets, "_git_lines", fake_git_lines)

    assert targets.changed_files(repo) == [Path("bragi/app.py")]
    assert ["merge-base", "HEAD", "abc123"] in calls


@pytest.mark.parametrize(
    ("changed_path", "target_path"),
    [
        (Path("README.md"), Path("tests/unit/docs")),
        (Path("docs/auth-policy.md"), Path("tests/unit/docs/test_auth_policy.py")),
        (Path("docs/docker-compose.md"), Path("tests/unit/docs/test_auth_policy.py")),
        (Path("docs/troubleshooting.md"), Path("tests/unit/docs")),
    ],
)
def test_select_changed_validation_maps_deployment_docs_to_docs_tests(
    changed_path: Path,
    target_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    docs_target = repo / target_path
    _write(repo / changed_path)
    if target_path.suffix:
        _write(docs_target)
    else:
        docs_target.mkdir(parents=True)
    monkeypatch.setattr(targets, "changed_files", lambda _repo: [changed_path])

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is False
    assert selection.test_targets == (docs_target.resolve(),)
    assert selection.typecheck is False


@pytest.mark.parametrize(
    ("changed_path", "target_path"),
    [
        (
            Path("docs/context-assembly.md"),
            Path("tests/unit/services/test_context_assembly.py"),
        ),
        (
            Path("docs/provider-generation-settings.md"),
            Path("tests/unit/services/test_generation_settings.py"),
        ),
    ],
)
def test_select_changed_validation_maps_maintained_behavior_docs(
    changed_path: Path,
    target_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    target = repo / target_path
    _write(repo / changed_path)
    _write(target)
    monkeypatch.setattr(targets, "changed_files", lambda _repo: [changed_path])

    selection = targets.select_changed_validation(repo, [])

    assert selection.full is False
    assert selection.test_targets == (target.resolve(),)
    assert selection.typecheck is False


def test_all_maintained_docs_are_mapped_or_intentionally_unmapped() -> None:
    docs_dir = Path(__file__).parents[3] / "docs"
    maintained_docs = {
        Path("docs") / path.name
        for path in docs_dir.glob("*.md")
        if path.is_file()
    }
    tracked_docs = (
        set(targets.DOC_TEST_TARGETS)
        | set(targets.DOCS_WITHOUT_SMART_VALIDATION)
    )

    assert maintained_docs - tracked_docs == set()


def test_queued_targets_for_repo_reads_this_repo_without_removing_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "repo")
    test = repo / "tests" / "unit" / "test_app.py"
    other = tmp_path / "other" / "tests" / "unit" / "test_app.py"
    _write(test)
    _write(other)
    queue_path = tmp_path / "queue.txt"
    queue_path.write_text(f"{test}\n{other}\n", encoding="utf-8")
    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))

    queued = targets.queued_targets_for_repo(repo)

    assert queued == [test.resolve()]
    assert targets.read_queue(repo) == [test.resolve(), other.resolve()]


def test_default_queue_path_is_repo_specific(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("BRAGI_EDIT_TEST_QUEUE", raising=False)
    first_repo = _repo(tmp_path / "first")
    second_repo = _repo(tmp_path / "second")

    first_path = targets.queue_path(first_repo)
    second_path = targets.queue_path(second_repo)

    assert first_path != second_path
    assert first_path.name.startswith("bragi-edit-test-queue-")
    assert second_path.name.startswith("bragi-edit-test-queue-")


def test_queue_env_override_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override = tmp_path / "shared-queue.txt"
    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(override))

    assert targets.queue_path(_repo(tmp_path / "repo")) == override


def test_append_queue_targets_preserves_concurrent_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "repo")
    queue_path = tmp_path / "queue.txt"
    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))
    active_readers = 0
    overlapped = False
    active_lock = threading.Lock()
    real_read_queue_file = targets._read_queue_file

    def tracking_read_queue_file(path: Path) -> list[Path]:
        nonlocal active_readers, overlapped
        with active_lock:
            active_readers += 1
            if active_readers > 1:
                overlapped = True
        try:
            time.sleep(0.01)
            return cast(list[Path], real_read_queue_file(path))
        finally:
            with active_lock:
                active_readers -= 1

    monkeypatch.setattr(targets, "_read_queue_file", tracking_read_queue_file)
    test_targets = [
        repo / "tests" / "unit" / f"test_{index}.py"
        for index in range(12)
    ]

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(
            executor.map(
                lambda test_target: targets.append_queue_targets(repo, [test_target]),
                test_targets,
            )
        )

    queued_targets = targets.read_queue(repo)

    assert overlapped is False
    assert set(queued_targets) == {target.resolve() for target in test_targets}
    assert len(queued_targets) == len(test_targets)


def test_remove_queue_targets_keeps_unpassed_and_other_repo_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "repo")
    passed = repo / "tests" / "unit" / "test_passed.py"
    pending = repo / "tests" / "unit" / "test_pending.py"
    other = tmp_path / "other" / "tests" / "unit" / "test_app.py"
    queue_path = tmp_path / "queue.txt"
    queue_path.write_text(
        f"{passed}\n{pending}\n{passed}\n{other}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BRAGI_EDIT_TEST_QUEUE", str(queue_path))

    targets.remove_queue_targets(repo, [passed])

    assert targets.read_queue(repo) == [pending.resolve(), other.resolve()]
