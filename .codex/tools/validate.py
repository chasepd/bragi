#!/usr/bin/env python3
"""Run Bragi validation with concise output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import validation_targets  # noqa: E402

_DEFAULT_UV_CACHE_DIR = "/tmp/bragi-uv-cache"
_PYTEST_WORKERS_ENV = "BRAGI_PYTEST_WORKERS"
_PYTEST_WORKERS_DISABLED_VALUES = {"", "0", "1", "false", "no", "off"}
_LOG_PREFIX = "bragi-validation-"
_HEARTBEAT_SECONDS = 15.0
_PHASE_OVERHEAD_SECONDS = 8.0
_LOW_CONFIDENCE_SAMPLE_COUNT = 3
_MAX_TIMING_SAMPLES = 8
_DEFAULT_ESTIMATE_RANGE_SECONDS = (30.0, 120.0)
_DEFAULT_PHASE_ESTIMATE_RANGES_SECONDS = {
    "frontend dependencies": (120.0, 300.0),
    "queued tests": (30.0, 180.0),
    "tests": (45.0, 240.0),
    "changed tests": (30.0, 180.0),
    "integration tests": (60.0, 300.0),
    "typecheck": (45.0, 180.0),
    "linting": (20.0, 90.0),
    "changed linting": (15.0, 75.0),
    "frontend tests": (30.0, 120.0),
    "frontend build": (45.0, 180.0),
    "frontend audit": (30.0, 120.0),
    "coverage": (120.0, 360.0),
    "coverage report": (15.0, 60.0),
}
_DURATION_CACHE_PATH = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "bragi"
    / "validation-durations.json"
)
_ENV_ALLOWLIST = {
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "SHELL",
    "SSH_AUTH_SOCK",
    "SYSTEMROOT",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
    "UV_CACHE_DIR",
    "BRAGI_PYTEST_WORKERS",
    "WINDIR",
}
_PYTEST_FAILURE_RE = re.compile(r"^_{2,}\s+(.+?)\s+_{2,}$")
_MYPY_ERROR_RE = re.compile(r"^[^:\s][^:]*:\d+(?::\d+)?:\s+(?:error|note):\s+.+$")
_RUFF_ERROR_RE = re.compile(r"^[^:\s][^:]*:\d+:\d+:\s+[A-Z]\d+\s+.+$")
_AGENT_BACKUP_ROOT_PATHS = (Path("AGENTS.md.bak"), Path("CLAUDE.md.bak"))
_AGENT_BACKUP_DIRS = (
    Path(".agentsync"),
    Path(".claude"),
    Path(".codex"),
    Path(".cursor"),
    Path(".opencode"),
)


@dataclass(frozen=True)
class Phase:
    name: str
    command: list[str]
    parser: str


@dataclass(frozen=True)
class PhaseResult:
    name: str
    command: list[str]
    returncode: int
    output: str
    duration_seconds: float = 0.0
    log_path: Path | None = None
    timing_key: str | None = None

    @property
    def passed(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class PhaseTiming:
    samples: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean_seconds(self) -> float:
        return sum(self.samples) / self.count if self.samples else 0.0

    @property
    def max_seconds(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def last_seconds(self) -> float:
        return self.samples[-1] if self.samples else 0.0


@dataclass(frozen=True)
class PhaseEstimate:
    lower_seconds: float
    upper_seconds: float
    confidence: str
    sample_count: int = 0


class ValidationArgumentError(ValueError):
    """Invalid arguments that depend on repository state."""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = _repo_root()
    backup_paths = _forbidden_agent_backup_paths(repo)
    if backup_paths:
        _print_agent_backup_failure(backup_paths)
        return 1
    try:
        phases = _selected_phases(args, repo)
    except ValidationArgumentError as exc:
        print(f"validate.py: error: {exc}", file=sys.stderr)
        return 2
    if not phases:
        print("smart validation: no changed validation targets")
        return 0
    _warn_if_python_lock_enforcement_unavailable(phases)
    duration_cache = _load_duration_cache()
    repo_key = _repo_cache_key(repo)
    phase_timings = duration_cache.get(repo_key, {})
    phase_estimates = {
        _phase_timing_key(phase): _estimate_for_phase(
            phase,
            phase_timings.get(_phase_timing_key(phase)),
        )
        for phase in phases
    }

    failed = False
    for index, phase in enumerate(phases, start=1):
        estimate = phase_estimates.get(_phase_timing_key(phase))
        result = _run_phase(
            repo,
            phase,
            index=index,
            total=len(phases),
            estimate=estimate,
            overall_estimate=_overall_estimate(
                phases=phases,
                phase_estimates=phase_estimates,
                current_index=index - 1,
            ),
        )
        _print_summary(result, phase.parser)
        _record_phase_duration(duration_cache, repo_key, result)
        failed = failed or not result.passed
    _save_duration_cache(duration_cache)

    return 1 if failed else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    focus = parser.add_mutually_exclusive_group()
    focus.add_argument(
        "--tests-only",
        nargs="*",
        metavar="TEST_PATH",
        help="run unit tests only, or run only the supplied test files",
    )
    focus.add_argument("--typecheck-only", action="store_true", help="run mypy only")
    focus.add_argument("--lint-only", action="store_true", help="run ruff only")
    focus.add_argument(
        "--frontend-only",
        action="store_true",
        help="run frontend dependency, test, build, and audit checks only",
    )
    focus.add_argument(
        "--coverage",
        action="store_true",
        help="run unit and integration tests with coverage reporting",
    )
    focus.add_argument(
        "--changed",
        action="store_true",
        help="run smart changed-file validation (default)",
    )
    focus.add_argument(
        "--full",
        action="store_true",
        help="run the full local/CI validation gate",
    )
    return parser.parse_args(argv)


def _repo_root() -> Path:
    for directory in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (directory / "pyproject.toml").is_file() and (directory / "bragi").is_dir():
            return directory
    return Path.cwd().resolve()


def _forbidden_agent_backup_paths(repo: Path) -> tuple[Path, ...]:
    backup_paths: list[Path] = []
    for rel in _AGENT_BACKUP_ROOT_PATHS:
        path = repo / rel
        if path.is_file():
            backup_paths.append(rel)

    for rel_dir in _AGENT_BACKUP_DIRS:
        directory = repo / rel_dir
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.bak"):
            if path.is_file():
                backup_paths.append(path.relative_to(repo))

    return tuple(sorted(backup_paths, key=lambda path: path.as_posix()))


def _print_agent_backup_failure(backup_paths: Sequence[Path]) -> None:
    print("agent backup hygiene: failed")
    for path in backup_paths:
        print(f"  remove stale backup file: {path.as_posix()}")


def _selected_phases(args: argparse.Namespace, repo: Path | None = None) -> list[Phase]:
    if args.tests_only is not None and args.tests_only is not False:
        repo = repo or _repo_root()
        explicit_targets = [] if args.tests_only is True else args.tests_only
        test_args = (
            _explicit_test_args(repo, explicit_targets)
            if explicit_targets
            else ["tests/unit", "tests/web"]
        )
        return [
            _tool_phase(
                "tests",
                "pytest",
                _pytest_args(test_args),
                "pytest",
            )
        ]
    if args.typecheck_only:
        return [_tool_phase("typecheck", "mypy", [], "mypy")]
    if args.lint_only:
        return [_tool_phase("linting", "ruff", ["check", "."], "ruff")]
    if args.frontend_only:
        return [_frontend_dependency_phase(), *_frontend_phases()]
    if args.coverage:
        return [
            _tool_phase(
                "coverage",
                "pytest",
                _pytest_args(
                    ["tests/unit", "tests/web", "tests/integration"],
                    extra=[
                        "--cov=bragi",
                        "--cov=bragi_common",
                        "--cov=bragi_web",
                        "--cov-report=",
                    ],
                ),
                "pytest",
            ),
            _tool_phase(
                "coverage report",
                "coverage",
                ["report", "--skip-covered", "--sort=cover"],
                "coverage",
            ),
        ]
    if args.full:
        return _full_phases()

    repo = repo or _repo_root()
    return _changed_phases(repo)


def _explicit_test_args(repo: Path, raw_targets: Sequence[str]) -> list[str]:
    repo = repo.resolve()
    targets: list[str] = []
    for raw_target in raw_targets:
        target = _explicit_test_target(repo, raw_target)
        targets.append(_target_arg(repo, target))
    return targets


def _explicit_test_target(repo: Path, raw_target: str) -> Path:
    if "::" in raw_target:
        raise ValidationArgumentError(
            f"explicit test targets must be test file paths: {raw_target}"
        )

    raw_path = Path(raw_target).expanduser()
    target = (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (repo / raw_path).resolve()
    )
    try:
        rel = target.relative_to(repo)
    except ValueError as exc:
        raise ValidationArgumentError(
            f"explicit test target must be inside this repository: {raw_target}"
        ) from exc

    if not rel.parts or rel.parts[0] != "tests":
        raise ValidationArgumentError(
            f"explicit test target must be under tests/: {raw_target}"
        )
    if target.suffix != ".py":
        raise ValidationArgumentError(
            f"explicit test target must be a Python test file: {raw_target}"
        )
    if not target.name.startswith("test_"):
        raise ValidationArgumentError(
            f"explicit test target must be a test_*.py file: {raw_target}"
        )
    if not target.is_file():
        raise ValidationArgumentError(
            f"explicit test target does not exist: {raw_target}"
        )
    return target


def _full_phases() -> list[Phase]:
    return [
        _frontend_dependency_phase(),
        _queued_tests_phase(),
        _tool_phase(
            "tests",
            "pytest",
            _pytest_args(["tests/unit", "tests/web"]),
            "pytest",
        ),
        _tool_phase(
            "integration tests",
            "pytest",
            _pytest_args(["tests/integration"]),
            "pytest",
        ),
        _tool_phase("typecheck", "mypy", [], "mypy"),
        _tool_phase("linting", "ruff", ["check", "."], "ruff"),
        *_frontend_phases(),
    ]


def _changed_phases(repo: Path) -> list[Phase]:
    queued_targets = validation_targets.queued_targets_for_repo(repo)
    selection = validation_targets.select_changed_validation(repo, [])
    if selection.full:
        reasons = ", ".join(selection.reasons) or "broad-risk change"
        print(f"smart validation: escalating to full validation ({reasons})")
        return _full_phases()
    if selection.empty and not queued_targets:
        return []

    phases: list[Phase] = []
    if queued_targets:
        phases.append(_queued_tests_phase())
    if selection.test_targets:
        phases.append(_changed_tests_phase(repo, list(selection.test_targets)))
    if selection.typecheck:
        phases.append(_tool_phase("typecheck", "mypy", [], "mypy"))
    if selection.lint_targets:
        phases.append(_changed_lint_phase(repo, list(selection.lint_targets)))
    if selection.frontend:
        phases.append(_frontend_dependency_phase())
        phases.extend(_frontend_phases())
    return phases


def _queued_tests_phase() -> Phase:
    return Phase(
        "queued tests",
        [sys.executable, ".codex/hooks/run_tests_for_edited_file.py", "--flush"],
        "pytest",
    )


def _changed_tests_phase(repo: Path, targets: list[Path]) -> Phase:
    return _tool_phase(
        "changed tests",
        "pytest",
        _pytest_args([_target_arg(repo, target) for target in targets]),
        "pytest",
    )


def _pytest_args(targets: list[str], *, extra: list[str] | None = None) -> list[str]:
    return [*targets, "-q", *_pytest_parallel_args(), *(extra or [])]


def _pytest_parallel_args() -> list[str]:
    workers = os.environ.get(_PYTEST_WORKERS_ENV, "auto").strip()
    if workers.lower() in _PYTEST_WORKERS_DISABLED_VALUES:
        return []
    return ["-n", workers]


def _changed_lint_phase(repo: Path, targets: list[Path]) -> Phase:
    return _tool_phase(
        "changed linting",
        "ruff",
        ["check", *[_target_arg(repo, target) for target in targets]],
        "ruff",
    )


def _target_arg(repo: Path, target: Path) -> str:
    try:
        return str(target.relative_to(repo))
    except ValueError:
        return str(target)


def _tool_phase(name: str, tool: str, args: list[str], parser: str) -> Phase:
    return Phase(name, _tool_command(tool, args), parser)


def _frontend_test_phase() -> Phase:
    return Phase(
        "frontend tests",
        ["npm", "run", "test", "--prefix", "frontend", "--", "--run"],
        "frontend",
    )


def _frontend_dependency_phase() -> Phase:
    return Phase(
        "frontend dependencies",
        ["npm", "ci", "--prefix", "frontend"],
        "frontend",
    )


def _frontend_phase() -> Phase:
    return Phase(
        "frontend build",
        ["npm", "run", "build", "--prefix", "frontend"],
        "frontend",
    )


def _frontend_audit_phase() -> Phase:
    return Phase(
        "frontend audit",
        ["npm", "audit", "--prefix", "frontend", "--audit-level=moderate"],
        "frontend",
    )


def _frontend_phases() -> list[Phase]:
    return [_frontend_test_phase(), _frontend_phase(), _frontend_audit_phase()]


def _tool_command(tool: str, args: list[str]) -> list[str]:
    if shutil.which("uv") is not None:
        cache_dir = os.environ.get("UV_CACHE_DIR", _DEFAULT_UV_CACHE_DIR)
        return [
            "uv",
            "--cache-dir",
            cache_dir,
            "run",
            "--locked",
            "--extra",
            "dev",
            tool,
            *args,
        ]
    return [sys.executable, "-m", tool, *args]


def _warn_if_python_lock_enforcement_unavailable(phases: Sequence[Phase]) -> None:
    if shutil.which("uv") is not None:
        return
    if not any(_uses_python_module_fallback(phase.command) for phase in phases):
        return

    print(
        "dependency lock enforcement: uv is not available; Python validation "
        "is using python -m fallback, so uv.lock consistency is not checked. "
        "Install uv or run `uv sync --locked --extra dev` where uv is available "
        "to verify lock drift.",
        file=sys.stderr,
    )


def _uses_python_module_fallback(command: Sequence[str]) -> bool:
    return len(command) >= 3 and command[0] == sys.executable and command[1] == "-m"


def _validation_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _ENV_ALLOWLIST
    }


def _run_phase(
    repo: Path,
    phase: Phase,
    *,
    index: int,
    total: int,
    estimate: PhaseEstimate | None,
    overall_estimate: PhaseEstimate | None,
) -> PhaseResult:
    timing_key = _phase_timing_key(phase)
    print(f"[{index}/{total}] {phase.name}: running", flush=True)
    print(f"  command: {shlex.join(phase.command)}", flush=True)
    estimate_parts = [_format_phase_estimate(estimate, "for this phase")]
    if overall_estimate is not None:
        estimate_parts.append(
            _format_phase_estimate(overall_estimate, "remaining overall")
        )
    print(f"  estimate: {', '.join(estimate_parts)}", flush=True)
    returncode, output, duration_seconds = _run_command(
        repo,
        phase.command,
        phase.name,
        estimate_seconds=estimate.upper_seconds if estimate is not None else None,
    )
    log_path = (
        _write_failure_log(phase.name, phase.command, output)
        if returncode
        else None
    )
    return PhaseResult(
        name=phase.name,
        command=phase.command,
        returncode=returncode,
        output=output,
        duration_seconds=duration_seconds,
        log_path=log_path,
        timing_key=timing_key,
    )


def _run_command(
    repo: Path,
    command: list[str],
    phase_name: str,
    *,
    estimate_seconds: float | None = None,
) -> tuple[int, str, float]:
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=_validation_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_queue: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(
        target=_read_process_output,
        args=(process, output_queue),
        daemon=True,
    )
    reader.start()
    output_parts: list[str] = []
    started_at = time.monotonic()
    last_heartbeat_at = started_at
    reader_done = False
    try:
        while True:
            try:
                chunk = output_queue.get(timeout=0.25)
            except queue.Empty:
                chunk = ""
            if chunk is None:
                reader_done = True
            elif chunk:
                output_parts.append(chunk)

            now = time.monotonic()
            if (
                process.poll() is None
                and now - last_heartbeat_at >= _HEARTBEAT_SECONDS
            ):
                print(
                    _heartbeat_line(
                        phase_name,
                        elapsed_seconds=now - started_at,
                        estimate_seconds=estimate_seconds,
                    ),
                    flush=True,
                )
                last_heartbeat_at = now

            if process.poll() is not None and reader_done:
                break
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    finally:
        reader.join(timeout=1)

    duration_seconds = time.monotonic() - started_at
    return process.wait(), "".join(output_parts), duration_seconds


def _read_process_output(
    process: subprocess.Popen[str],
    output_queue: queue.Queue[str | None],
) -> None:
    if process.stdout is None:
        output_queue.put(None)
        return
    try:
        for line in process.stdout:
            output_queue.put(line)
    finally:
        output_queue.put(None)


def _format_elapsed(seconds: float) -> str:
    elapsed = max(0, int(seconds))
    minutes, remainder = divmod(elapsed, 60)
    if minutes:
        return f"{minutes}m {remainder}s"
    return f"{remainder}s"


def _format_estimate_range(estimate: PhaseEstimate) -> str:
    lower = _format_elapsed(estimate.lower_seconds)
    upper = _format_elapsed(estimate.upper_seconds)
    if lower == upper:
        return f"~{upper}"
    return f"~{lower}-{upper}"


def _format_phase_estimate(estimate: PhaseEstimate | None, suffix: str) -> str:
    if estimate is None:
        return (
            "low confidence "
            f"{_format_estimate_range(_default_estimate(None))} {suffix}"
        )
    detail = _format_estimate_range(estimate)
    if estimate.confidence == "low":
        return f"low confidence {detail} {suffix}"
    return f"{detail} {suffix} (based on {estimate.sample_count} runs)"


def _heartbeat_line(
    phase_name: str,
    *,
    elapsed_seconds: float,
    estimate_seconds: float | None,
) -> str:
    elapsed = _format_elapsed(elapsed_seconds)
    if estimate_seconds is None:
        return f"{phase_name}: still running ({elapsed} elapsed)"
    remaining = estimate_seconds - elapsed_seconds
    if remaining > 0:
        return (
            f"{phase_name}: still running "
            f"({elapsed} elapsed, ~{_format_elapsed(remaining)} left)"
        )
    return (
        f"{phase_name}: still running "
        f"({elapsed} elapsed, estimate exceeded by "
        f"{_format_elapsed(abs(remaining))})"
    )


def _overall_estimate_seconds(
    *,
    phases: list[Phase],
    phase_estimates: dict[str, PhaseEstimate],
    current_index: int,
) -> float | None:
    estimate = _overall_estimate(
        phases=phases,
        phase_estimates=phase_estimates,
        current_index=current_index,
    )
    return None if estimate is None else estimate.upper_seconds


def _overall_estimate(
    *,
    phases: list[Phase],
    phase_estimates: dict[str, PhaseEstimate],
    current_index: int,
) -> PhaseEstimate | None:
    estimates: list[PhaseEstimate] = []
    for phase in phases[current_index:]:
        estimate = phase_estimates.get(_phase_timing_key(phase))
        if estimate is None:
            return None
        estimates.append(estimate)
    if not estimates:
        return None
    return PhaseEstimate(
        lower_seconds=sum(estimate.lower_seconds for estimate in estimates),
        upper_seconds=sum(estimate.upper_seconds for estimate in estimates),
        confidence=(
            "low"
            if any(estimate.confidence == "low" for estimate in estimates)
            else "high"
        ),
        sample_count=min(estimate.sample_count for estimate in estimates),
    )


def _estimate_for_phase(
    phase: Phase,
    timing: PhaseTiming | None,
) -> PhaseEstimate:
    if timing is None or not timing.samples:
        return _default_estimate(phase)

    observed_floor = max(timing.mean_seconds, timing.last_seconds)
    if timing.count < _LOW_CONFIDENCE_SAMPLE_COUNT:
        lower = observed_floor + _PHASE_OVERHEAD_SECONDS
        upper = max(
            timing.max_seconds * 1.8,
            observed_floor + 30.0,
        ) + _PHASE_OVERHEAD_SECONDS
        return PhaseEstimate(
            lower_seconds=lower,
            upper_seconds=max(upper, lower),
            confidence="low",
            sample_count=timing.count,
        )

    lower = observed_floor + _PHASE_OVERHEAD_SECONDS
    upper = max(
        timing.max_seconds * 1.25,
        timing.mean_seconds * 1.4,
        timing.last_seconds * 1.2,
    ) + _PHASE_OVERHEAD_SECONDS
    return PhaseEstimate(
        lower_seconds=lower,
        upper_seconds=max(upper, lower),
        confidence="high",
        sample_count=timing.count,
    )


def _default_estimate(phase: Phase | None) -> PhaseEstimate:
    if phase is None:
        lower, upper = _DEFAULT_ESTIMATE_RANGE_SECONDS
    elif _is_full_unit_web_test_phase(phase):
        lower, upper = (240.0, 600.0)
    else:
        lower, upper = _DEFAULT_PHASE_ESTIMATE_RANGES_SECONDS.get(
            phase.name,
            _DEFAULT_ESTIMATE_RANGE_SECONDS,
        )
    return PhaseEstimate(
        lower_seconds=lower + _PHASE_OVERHEAD_SECONDS,
        upper_seconds=upper + _PHASE_OVERHEAD_SECONDS,
        confidence="low",
    )


def _is_full_unit_web_test_phase(phase: Phase) -> bool:
    return (
        phase.name == "tests"
        and "tests/unit" in phase.command
        and "tests/web" in phase.command
    )


def _phase_timing_key(phase: Phase) -> str:
    command_hash = hashlib.sha256(
        "\0".join(phase.command).encode("utf-8")
    ).hexdigest()[:12]
    return f"{phase.name}:{command_hash}"


def _record_phase_duration(
    duration_cache: dict[str, dict[str, PhaseTiming]],
    repo_key: str,
    result: PhaseResult,
) -> None:
    if not result.passed or result.duration_seconds <= 0:
        return
    repo_durations = duration_cache.setdefault(repo_key, {})
    timing_key = result.timing_key or result.name
    prior = repo_durations.get(timing_key)
    samples = (
        (result.duration_seconds,)
        if prior is None
        else (*prior.samples, result.duration_seconds)[-_MAX_TIMING_SAMPLES:]
    )
    repo_durations[timing_key] = PhaseTiming(samples=samples)


def _load_duration_cache() -> dict[str, dict[str, PhaseTiming]]:
    try:
        payload = json.loads(_DURATION_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    cache: dict[str, dict[str, PhaseTiming]] = {}
    for repo_key, phases in payload.items():
        if not isinstance(repo_key, str) or not isinstance(phases, dict):
            continue
        phase_durations: dict[str, PhaseTiming] = {}
        for phase_name, timing_payload in phases.items():
            if not isinstance(phase_name, str):
                continue
            timing = _parse_phase_timing(timing_payload)
            if timing is not None:
                phase_durations[phase_name] = timing
        cache[repo_key] = phase_durations
    return cache


def _parse_phase_timing(payload: object) -> PhaseTiming | None:
    if isinstance(payload, int | float):
        return PhaseTiming(samples=(max(0.0, float(payload)),))
    if not isinstance(payload, dict):
        return None
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        return None
    samples = tuple(
        max(0.0, float(sample))
        for sample in raw_samples
        if isinstance(sample, int | float)
    )
    if not samples:
        return None
    return PhaseTiming(samples=samples[-_MAX_TIMING_SAMPLES:])


def _save_duration_cache(duration_cache: dict[str, dict[str, PhaseTiming]]) -> None:
    try:
        _DURATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            repo_key: {
                phase_name: {
                    "samples": list(timing.samples),
                    "count": timing.count,
                    "mean_seconds": timing.mean_seconds,
                    "max_seconds": timing.max_seconds,
                    "last_seconds": timing.last_seconds,
                }
                for phase_name, timing in phases.items()
            }
            for repo_key, phases in duration_cache.items()
        }
        _DURATION_CACHE_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def _repo_cache_key(repo: Path) -> str:
    return hashlib.sha256(str(repo.resolve()).encode("utf-8")).hexdigest()[:16]


def _write_failure_log(name: str, command: list[str], output: str) -> Path:
    safe_name = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "phase"
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        prefix=f"{_LOG_PREFIX}{safe_name}-",
        suffix=".log",
    )
    with handle:
        handle.write(f"$ {' '.join(command)}\n\n")
        handle.write(output)
    return Path(handle.name)


def _print_summary(result: PhaseResult, parser: str) -> None:
    if result.passed:
        print(f"{result.name}: passed")
        if parser == "coverage":
            for line in _first_meaningful_lines(result.output.splitlines()):
                print(f"  {line}")
        return

    details = _failure_excerpt(result.output, parser)
    if details:
        print(f"{result.name}: failed")
        for line in details:
            print(f"  {line}")
    else:
        print(f"{result.name}: failed with exit code {result.returncode}")

    if result.log_path is not None:
        print(f"  full log: {result.log_path}")


def _failure_excerpt(output: str, parser: str) -> list[str]:
    lines = output.splitlines()
    if parser == "pytest":
        return _pytest_excerpt(lines)
    if parser == "mypy":
        return _matching_lines(lines, _MYPY_ERROR_RE)
    if parser == "ruff":
        return _matching_lines(lines, _RUFF_ERROR_RE)
    if parser == "coverage":
        return _first_meaningful_lines(lines)
    if parser == "frontend":
        return _frontend_excerpt(lines)
    return _first_meaningful_lines(lines)


def _pytest_excerpt(lines: list[str]) -> list[str]:
    excerpt: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _PYTEST_FAILURE_RE.match(stripped):
            excerpt.append(stripped.strip("_ "))
        elif stripped.startswith(("FAILED ", "E   ", ">")):
            excerpt.append(stripped)
        if len(excerpt) >= 12:
            break
    return excerpt or _first_meaningful_lines(lines)


def _matching_lines(lines: list[str], pattern: re.Pattern[str]) -> list[str]:
    matches = [line.strip() for line in lines if pattern.match(line.strip())]
    return matches[:12] if matches else _first_meaningful_lines(lines)


def _frontend_excerpt(lines: list[str]) -> list[str]:
    meaningful = [line.strip() for line in lines if line.strip()]
    failure_terms = (
        " FAIL ",
        " ❯ ",
        " × ",
        "AssertionError",
        "TestingLibraryElementError",
        "Error:",
        "Expected",
        "Received",
        "Unable to",
    )
    matches = [
        line
        for line in meaningful
        if any(term in line for term in failure_terms)
    ]
    tail = meaningful[-24:]
    excerpt = [*matches[:24], *tail]
    if not excerpt:
        return _first_meaningful_lines(lines)
    deduped: list[str] = []
    seen: set[str] = set()
    for line in excerpt:
        if line in seen:
            continue
        deduped.append(line)
        seen.add(line)
    return deduped[:36]


def _first_meaningful_lines(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip()][:12]


if __name__ == "__main__":
    raise SystemExit(main())
