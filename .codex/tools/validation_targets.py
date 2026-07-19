"""Shared validation target discovery for Bragi hooks and local checks."""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO, cast

_EDIT_TEST_QUEUE_ENV = "BRAGI_EDIT_TEST_QUEUE"
_VALIDATION_BASE_REF_ENV = "BRAGI_VALIDATION_BASE_REF"
_QUEUE_THREAD_LOCK = threading.Lock()

SERVICE_TEST_TARGETS = {
    Path("bragi/services/action_choice_flags.py"): Path(
        "tests/unit/services/test_action_choice_service.py"
    ),
    Path("bragi/services/character_locks.py"): Path(
        "tests/unit/services/test_character_registry_service.py"
    ),
    Path("bragi/services/character_text_world_update_service.py"): Path(
        "tests/unit/services/test_character_text_service.py"
    ),
    Path("bragi/services/message_reconciliation_service.py"): Path(
        "tests/unit/application/test_controller.py"
    ),
    Path("bragi/services/narration_context.py"): Path(
        "tests/unit/services/test_chat_service.py"
    ),
    Path("bragi/services/npc_knowledge_audit_service.py"): Path(
        "tests/unit/services/test_chat_service.py"
    ),
    Path("bragi/services/post_turn_inference.py"): Path(
        "tests/unit/services/test_chat_service.py"
    ),
    Path("bragi/services/redaction.py"): Path(
        "tests/unit/services/test_services_redaction.py"
    ),
    Path("bragi/services/settings_policy.py"): Path(
        "tests/unit/services/test_settings_service.py"
    ),
    Path("bragi/services/user_narration_guidance.py"): Path(
        "tests/unit/services/test_chat_service.py"
    ),
}

WEB_TEST_TARGETS = {
    Path("bragi_web/api/app.py"): Path("tests/web/test_api.py"),
    Path("bragi_web/auth_throttle.py"): Path("tests/web/test_api.py"),
    Path("bragi_web/bragi_adapter.py"): Path("tests/web/test_bragi_adapter.py"),
    Path("bragi_web/jobs.py"): Path("tests/web/test_jobs.py"),
    Path("bragi_web/main.py"): Path("tests/web/test_cli.py"),
    Path("bragi_web/maintenance_diagnostics.py"): Path(
        "tests/web/test_maintenance_diagnostics.py"
    ),
    Path("bragi_web/observability.py"): Path("tests/web/test_observability.py"),
    Path("bragi_web/runtime.py"): Path("tests/web/test_api.py"),
    Path("bragi_web/serialization.py"): Path("tests/web/test_api.py"),
    Path("bragi_web/storage.py"): Path("tests/web/test_storage.py"),
}

DOC_TEST_TARGETS = {
    Path("AGENTS.md"): Path("tests/unit/docs/test_agents_docs.py"),
    Path("README.md"): Path("tests/unit/docs"),
    Path("docs/auth-policy.md"): Path("tests/unit/docs/test_auth_policy.py"),
    Path("docs/context-assembly.md"): Path(
        "tests/unit/services/test_context_assembly.py"
    ),
    Path("docs/docker-compose.md"): Path("tests/unit/docs/test_auth_policy.py"),
    Path("docs/provider-generation-settings.md"): Path(
        "tests/unit/services/test_generation_settings.py"
    ),
    Path("docs/troubleshooting.md"): Path("tests/unit/docs"),
}

DOCS_WITHOUT_SMART_VALIDATION = {
    # Process-only AgentSync guidance; AgentSync source/config changes remain
    # broad-risk through their own paths.
    Path("docs/agentsync.md"),
    # Historical audit note; future behavior changes should update mapped docs
    # or add a dedicated validation target.
    Path("docs/narration-query-index-audit.md"),
    # Historical privacy review; implementation changes belong in mapped policy docs.
    Path("docs/privacy-review.md"),
}

BROAD_RISK_PATHS = {
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path(".dockerignore"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("Dockerfile"),
    Path("bragi_build_backend.py"),
    Path("compose.yaml"),
    Path("docker-compose.yml"),
    Path("MANIFEST.in"),
    Path("Makefile"),
    Path(".codex/hooks.json"),
    Path(".codex/config.toml"),
    Path(".github/workflows/ci.yml"),
    Path(".github/workflows/agentsync-auto-sync.yml"),
    Path(".github/workflows/agentsync-drift.yml"),
    Path("frontend/package.json"),
    Path("frontend/package-lock.json"),
}
BROAD_RISK_PREFIXES = (
    Path("scripts"),
    Path(".agentsync"),
    Path(".claude"),
    Path(".codex/hooks"),
    Path(".codex/tools"),
    Path(".codex/skills"),
    Path(".cursor"),
    Path(".opencode"),
)

FRONTEND_PREFIXES = (Path("frontend"),)
TYPECHECK_PREFIXES = (
    Path(".codex/hooks"),
    Path(".codex/tools"),
    Path("bragi"),
    Path("bragi_common"),
    Path("bragi_web"),
    Path("tests"),
)
PYTHON_LINT_PREFIXES = (
    *TYPECHECK_PREFIXES,
)


@dataclass(frozen=True)
class ValidationSelection:
    """Changed-file validation decisions."""

    full: bool = False
    test_targets: tuple[Path, ...] = ()
    lint_targets: tuple[Path, ...] = ()
    typecheck: bool = False
    frontend: bool = False
    changed_files: tuple[Path, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return (
            not self.full
            and not self.test_targets
            and not self.lint_targets
            and not self.typecheck
            and not self.frontend
        )


@dataclass
class _SelectionBuilder:
    repo: Path
    full: bool = False
    typecheck: bool = False
    frontend: bool = False
    changed_files: list[Path] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    _test_targets: list[Path] = field(default_factory=list)
    _lint_targets: list[Path] = field(default_factory=list)

    def add_test_target(self, target: Path) -> None:
        if target.exists():
            self._test_targets.append(target)

    def add_lint_target(self, target: Path) -> None:
        if target.exists():
            self._lint_targets.append(target)

    def build(self) -> ValidationSelection:
        return ValidationSelection(
            full=self.full,
            test_targets=tuple(_dedupe_paths(self._test_targets)),
            lint_targets=tuple(_dedupe_paths(self._lint_targets)),
            typecheck=self.typecheck,
            frontend=self.frontend,
            changed_files=tuple(_dedupe_relative(self.changed_files)),
            reasons=tuple(_dedupe_strings(self.reasons)),
        )


def queue_path(repo: Path | None = None) -> Path:
    raw = os.environ.get(_EDIT_TEST_QUEUE_ENV)
    if raw:
        return Path(raw).expanduser()
    if repo is not None:
        repo_key = hashlib.sha256(str(repo.resolve()).encode("utf-8")).hexdigest()[:16]
        return Path(tempfile.gettempdir()) / f"bragi-edit-test-queue-{repo_key}.txt"
    return Path(tempfile.gettempdir()) / "bragi-edit-test-queue.txt"


def read_queue(repo: Path | None = None) -> list[Path]:
    path = queue_path(repo)
    with _locked_queue(path):
        return _read_queue_file(path)


def write_queue(test_targets: list[Path], repo: Path | None = None) -> None:
    path = queue_path(repo)
    with _locked_queue(path):
        _write_queue_file(path, test_targets)


def append_queue_targets(repo: Path, test_targets: list[Path]) -> None:
    if not test_targets:
        return

    repo = repo.resolve()
    path = queue_path(repo)
    with _locked_queue(path):
        targets = _read_queue_file(path)
        seen = {str(target) for target in targets}
        for test_target in test_targets:
            resolved = test_target.expanduser().resolve()
            target_text = str(resolved)
            if target_text in seen:
                continue
            targets.append(resolved)
            seen.add(target_text)
        _write_queue_file(path, targets)


def queued_targets_for_repo(repo: Path) -> list[Path]:
    repo = repo.resolve()
    path = queue_path(repo)
    with _locked_queue(path):
        return _targets_for_repo(_read_queue_file(path), repo)


def remove_queue_targets(repo: Path, test_targets: list[Path]) -> None:
    if not test_targets:
        return

    repo = repo.resolve()
    path = queue_path(repo)
    passed = {str(target.expanduser().resolve()) for target in test_targets}
    with _locked_queue(path):
        remaining: list[Path] = []
        for target in _read_queue_file(path):
            try:
                target.relative_to(repo)
            except ValueError:
                remaining.append(target)
                continue
            if str(target.expanduser().resolve()) in passed:
                continue
            remaining.append(target)
        _write_queue_file(path, remaining)


@contextmanager
def _locked_queue(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with _QUEUE_THREAD_LOCK, lock_path.open("a+", encoding="utf-8") as lock_file:
        _lock_file(lock_file)
        try:
            yield
        finally:
            _unlock_file(lock_file)


def _lock_file(lock_file: TextIO) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return
    if os.name != "nt":
        return

    _ensure_windows_lock_byte(lock_file)
    msvcrt = cast(Any, importlib.import_module("msvcrt"))
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)


def _ensure_windows_lock_byte(lock_file: TextIO) -> None:
    lock_file.seek(0)
    if lock_file.read(1):
        lock_file.seek(0)
        return
    lock_file.write("\0")
    lock_file.flush()
    lock_file.seek(0)


def _unlock_file(lock_file: TextIO) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return
    if os.name != "nt":
        return

    lock_file.seek(0)
    msvcrt = cast(Any, importlib.import_module("msvcrt"))
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _read_queue_file(path: Path) -> list[Path]:
    if not path.exists():
        return []

    try:
        raw_targets = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    return [
        Path(raw).expanduser().resolve()
        for raw in raw_targets
        if raw.strip()
    ]


def _write_queue_file(path: Path, test_targets: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    contents = "\n".join(str(target) for target in test_targets)
    temp_path.write_text(f"{contents}\n" if contents else "", encoding="utf-8")
    temp_path.replace(path)


def _targets_for_repo(queued_targets: list[Path], repo: Path) -> list[Path]:
    repo = repo.resolve()
    targets: list[Path] = []

    for target in queued_targets:
        try:
            target.relative_to(repo)
        except ValueError:
            continue
        targets.append(target)
    return _dedupe_paths(targets)


def unit_test_for_source(repo: Path, rel: Path) -> Path | None:
    if rel.suffix != ".py" or not rel.parts:
        return None
    if rel.name == "__init__.py":
        return None

    if rel.parts[0] == "bragi":
        module_parts = rel.parts[1:-1]
    elif rel.parts[0] == "bragi_common":
        module_parts = rel.parts[:-1]
    else:
        return None

    return repo / "tests" / "unit" / Path(*module_parts) / f"test_{rel.stem}.py"


def unit_subsystem_test_dir_for_source(repo: Path, rel: Path) -> Path | None:
    if rel.suffix != ".py" or not rel.parts:
        return None
    if rel.parts[0] != "bragi" or rel.name == "__init__.py":
        return None
    if len(rel.parts) < 3:
        return None

    subsystem_dir = repo / "tests" / "unit" / rel.parts[1]
    return subsystem_dir if subsystem_dir.exists() else None


def web_test_for_source(repo: Path, rel: Path) -> Path | None:
    if rel.suffix != ".py" or not rel.parts:
        return None
    if rel.parts[0] != "bragi_web" or rel.name == "__init__.py":
        return None

    target = WEB_TEST_TARGETS.get(rel)
    if target is not None:
        return repo / target

    generic_target = repo / "tests" / "web" / f"test_{rel.stem}.py"
    return generic_target if generic_target.exists() else None


def codex_test_for_source(repo: Path, rel: Path) -> Path | None:
    if rel.suffix != ".py" or not rel.parts:
        return None
    if len(rel.parts) < 3 or rel.parts[:2] not in {
        (".codex", "hooks"),
        (".codex", "tools"),
    }:
        return None
    test_name = rel.stem.replace("-", "_")
    target = repo / "tests" / "unit" / "codex" / f"test_{test_name}.py"
    return target if target.exists() else None


def test_target_for_path(repo: Path, path: Path, rel: Path) -> Path | None:
    if not rel.parts:
        return None
    if rel.parts[0] == "tests" and rel.suffix == ".py":
        return path

    doc_target = DOC_TEST_TARGETS.get(rel)
    if doc_target is not None:
        target = repo / doc_target
        return target if target.exists() else None

    codex_target = codex_test_for_source(repo, rel)
    if codex_target is not None:
        return codex_target

    web_target = web_test_for_source(repo, rel)
    if web_target is not None and web_target.exists():
        return web_target

    service_target = SERVICE_TEST_TARGETS.get(rel)
    if service_target is not None:
        target = repo / service_target
        return target if target.exists() else None

    direct_target = unit_test_for_source(repo, rel)
    if direct_target is None:
        return None
    if direct_target.exists():
        return direct_target

    return unit_subsystem_test_dir_for_source(repo, rel)


def changed_files(repo: Path, base_ref: str | None = None) -> list[Path]:
    base_ref = base_ref or os.environ.get(_VALIDATION_BASE_REF_ENV) or "main"
    candidates: list[Path] = []
    merge_base = _git_lines(repo, ["merge-base", "HEAD", base_ref])
    if merge_base:
        candidates.extend(
            _paths_from_git_lines(
                _git_lines(repo, ["diff", "--name-only", merge_base[0], "HEAD"])
            )
        )
    else:
        candidates.extend(
            _paths_from_git_lines(
                _git_lines(repo, ["diff", "--name-only", f"{base_ref}...HEAD"])
            )
        )

    candidates.extend(_paths_from_git_lines(_git_lines(repo, ["diff", "--name-only"])))
    candidates.extend(
        _paths_from_git_lines(_git_lines(repo, ["diff", "--cached", "--name-only"]))
    )
    candidates.extend(
        _paths_from_git_lines(
            _git_lines(repo, ["ls-files", "--others", "--exclude-standard"])
        )
    )
    return _dedupe_relative(candidates)


def select_changed_validation(
    repo: Path,
    queued_targets: list[Path],
) -> ValidationSelection:
    builder = _SelectionBuilder(repo=repo)
    for target in queued_targets:
        builder.add_test_target(target)

    for rel in changed_files(repo):
        builder.changed_files.append(rel)
        abs_path = (repo / rel).resolve()

        if _is_broad_risk_file(rel):
            builder.full = True
            builder.reasons.append(str(rel))
            continue

        test_target = test_target_for_path(repo, abs_path, rel)
        if test_target is not None:
            builder.add_test_target(test_target)

        if _is_python_lint_file(rel):
            builder.add_lint_target(abs_path)

        if _is_typecheck_file(rel):
            builder.typecheck = True

        if _is_frontend_file(rel):
            builder.frontend = True

    return builder.build()


def _is_typecheck_file(rel: Path) -> bool:
    return rel.suffix == ".py" and any(
        _is_relative_to(rel, prefix) for prefix in TYPECHECK_PREFIXES
    )


def _is_broad_risk_file(rel: Path) -> bool:
    return rel in BROAD_RISK_PATHS or any(
        _is_relative_to(rel, prefix) for prefix in BROAD_RISK_PREFIXES
    )


def _is_python_lint_file(rel: Path) -> bool:
    return rel.suffix == ".py" and any(
        _is_relative_to(rel, prefix) for prefix in PYTHON_LINT_PREFIXES
    )


def _is_frontend_file(rel: Path) -> bool:
    return any(_is_relative_to(rel, prefix) for prefix in FRONTEND_PREFIXES)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _git_lines(repo: Path, args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _paths_from_git_lines(lines: list[str]) -> list[Path]:
    return [Path(line) for line in lines if line.strip()]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(resolved)
    return ordered


def _dedupe_relative(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
