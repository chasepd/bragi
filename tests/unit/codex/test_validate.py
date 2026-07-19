from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
from pathlib import Path

import pytest

_VALIDATE_PATH = Path(__file__).parents[3] / ".codex" / "tools" / "validate.py"
_SPEC = importlib.util.spec_from_file_location("bragi_validate", _VALIDATE_PATH)
assert _SPEC is not None
validate = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = validate
_SPEC.loader.exec_module(validate)


def test_tool_command_uses_uv_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validate.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/custom-uv-cache")

    assert validate._tool_command("pytest", ["tests/unit", "-q"]) == [
        "uv",
        "--cache-dir",
        "/tmp/custom-uv-cache",
        "run",
        "--locked",
        "--extra",
        "dev",
        "pytest",
        "tests/unit",
        "-q",
    ]


def test_tool_command_uses_python_module_when_uv_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)

    assert validate._tool_command("mypy", []) == [sys.executable, "-m", "mypy"]


def test_validation_env_excludes_provider_keys_and_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        os,
        "environ",
        {
            "PATH": "/usr/bin",
            "HOME": "/home/example",
            "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
            "UV_CACHE_DIR": "/tmp/bragi-uv-cache",
            "OPENROUTER_API_KEY": "secret",
            "VENICE_API_KEY": "secret",
            "GITHUB_TOKEN": "secret",
            "BRAGI_VALIDATION_RUNNER": "1",
            "DATABASE_URL": "sqlite:///secret",
            "PYTHONPATH": "/tmp/injected",
        },
    )

    env = validate._validation_env()

    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/home/example",
        "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
        "UV_CACHE_DIR": "/tmp/bragi-uv-cache",
    }


def test_main_warns_when_uv_lock_enforcement_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(validate, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)
    monkeypatch.setattr(validate, "_run_command", lambda *args, **kwargs: (0, "", 1.0))
    monkeypatch.setattr(validate, "_load_duration_cache", lambda: {})
    monkeypatch.setattr(validate, "_save_duration_cache", lambda _cache: None)

    assert validate.main(["--typecheck-only"]) == 0

    assert "uv is not available" in capsys.readouterr().err


def test_default_validation_passes_when_all_phases_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(validate, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)

    def fake_run_command(
        _repo: Path,
        _command: list[str],
        _phase_name: str,
        *,
        estimate_seconds: float | None = None,
    ) -> tuple[int, str, float]:
        return 0, "", 1.0

    monkeypatch.setattr(validate, "_run_command", fake_run_command)
    monkeypatch.setattr(validate, "_load_duration_cache", lambda: {})
    monkeypatch.setattr(validate, "_save_duration_cache", lambda _cache: None)

    assert validate.main(["--full"]) == 0

    lines = capsys.readouterr().out.splitlines()
    expected_phases = [
        "frontend dependencies",
        "queued tests",
        "tests",
        "integration tests",
        "typecheck",
        "linting",
        "frontend tests",
        "frontend build",
        "frontend audit",
    ]
    for phase_name in expected_phases:
        assert any(line.endswith(f"{phase_name}: running") for line in lines)
        assert f"{phase_name}: passed" in lines
    assert sum(line.startswith("  command: ") for line in lines) == len(
        expected_phases
    )
    assert sum(line.startswith("  estimate: low confidence ") for line in lines) == (
        len(expected_phases)
    )


def test_validation_returns_failure_when_any_phase_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(validate, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)
    monkeypatch.setattr(validate, "_write_failure_log", lambda *args: tmp_path / "log")

    def fake_run_command(
        _repo: Path,
        command: list[str],
        _phase_name: str,
        *,
        estimate_seconds: float | None = None,
    ) -> tuple[int, str, float]:
        if "mypy" in command:
            return (
                1,
                "bragi/app.py:10: error: Broken type\n",
                1.0,
            )
        return 0, "", 1.0

    monkeypatch.setattr(validate, "_run_command", fake_run_command)
    monkeypatch.setattr(validate, "_load_duration_cache", lambda: {})
    monkeypatch.setattr(validate, "_save_duration_cache", lambda _cache: None)

    assert validate.main(["--full"]) == 1

    output = capsys.readouterr().out
    assert "queued tests: passed" in output
    assert "tests: passed" in output
    assert "integration tests: passed" in output
    assert "typecheck: failed" in output
    assert "  bragi/app.py:10: error: Broken type" in output
    assert "linting: passed" in output
    assert "frontend dependencies: passed" in output
    assert "frontend tests: passed" in output
    assert "frontend build: passed" in output
    assert "frontend audit: passed" in output


def test_validation_phase_announces_command_before_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    phase = validate.Phase("tests", ["pytest", "-q"], "pytest")
    called = False

    def fake_run_command(
        _repo: Path,
        _command: list[str],
        _phase_name: str,
        *,
        estimate_seconds: float | None = None,
    ) -> tuple[int, str, float]:
        nonlocal called
        called = True
        return 0, "", 1.0

    monkeypatch.setattr(validate, "_run_command", fake_run_command)

    result = validate._run_phase(
        tmp_path,
        phase,
        index=1,
        total=1,
        estimate=validate.PhaseEstimate(
            lower_seconds=50.0,
            upper_seconds=70.0,
            confidence="high",
            sample_count=4,
        ),
        overall_estimate=validate.PhaseEstimate(
            lower_seconds=90.0,
            upper_seconds=120.0,
            confidence="high",
            sample_count=4,
        ),
    )

    assert called is True
    assert result.passed is True
    assert capsys.readouterr().out.splitlines() == [
        "[1/1] tests: running",
        "  command: pytest -q",
        "  estimate: ~50s-1m 10s for this phase (based on 4 runs), "
        "~1m 30s-2m 0s remaining overall (based on 4 runs)",
    ]


def test_validation_command_prints_heartbeat_for_long_silent_phase(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class SlowProcess:
        stdout = io.StringIO("")

        def __init__(self) -> None:
            self.polls: list[int | None] = [None, 0]

        def poll(self) -> int | None:
            return self.polls.pop(0) if self.polls else 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def terminate(self) -> None:
            raise AssertionError("test process should not be terminated")

        def kill(self) -> None:
            raise AssertionError("test process should not be killed")

    monkeypatch.setattr(
        validate.subprocess,
        "Popen",
        lambda *args, **kwargs: SlowProcess(),
    )
    monotonic_values = iter([0.0, 16.0, 16.0])
    monkeypatch.setattr(validate.time, "monotonic", lambda: next(monotonic_values))

    returncode, output, _duration = validate._run_command(
        tmp_path,
        ["pytest", "-q"],
        "tests",
        estimate_seconds=45.0,
    )

    assert returncode == 0
    assert output == ""
    assert capsys.readouterr().out.splitlines() == [
        "tests: still running (16s elapsed, ~29s left)",
    ]


def test_failed_validation_phase_does_not_update_duration_cache() -> None:
    cache = {"repo": {"tests": validate.PhaseTiming(samples=(120.0,))}}
    result = validate.PhaseResult(
        name="tests",
        command=["pytest"],
        returncode=1,
        output="failed early",
        duration_seconds=2.0,
    )

    validate._record_phase_duration(cache, "repo", result)

    assert cache == {"repo": {"tests": validate.PhaseTiming(samples=(120.0,))}}


def test_validation_uses_low_confidence_ranges_without_timing_data() -> None:
    phase = validate.Phase("typecheck", ["mypy"], "mypy")

    estimate = validate._estimate_for_phase(phase, None)

    assert estimate == validate.PhaseEstimate(
        lower_seconds=53.0,
        upper_seconds=188.0,
        confidence="low",
    )
    assert validate._format_phase_estimate(estimate, "for this phase") == (
        "low confidence ~53s-3m 8s for this phase"
    )


def test_validation_uses_wider_default_range_for_full_unit_web_tests() -> None:
    phase = validate.Phase(
        "tests",
        ["pytest", "tests/unit", "tests/web", "-q"],
        "pytest",
    )

    estimate = validate._estimate_for_phase(phase, None)

    assert estimate == validate.PhaseEstimate(
        lower_seconds=248.0,
        upper_seconds=608.0,
        confidence="low",
    )


def test_validation_estimate_uses_pessimistic_history_range() -> None:
    phase = validate.Phase("tests", ["pytest"], "pytest")
    timing = validate.PhaseTiming(samples=(80.0, 95.0, 140.0, 100.0))

    estimate = validate._estimate_for_phase(phase, timing)

    assert estimate == validate.PhaseEstimate(
        lower_seconds=111.75,
        upper_seconds=183.0,
        confidence="high",
        sample_count=4,
    )
    assert validate._format_phase_estimate(estimate, "for this phase") == (
        "~1m 51s-3m 3s for this phase (based on 4 runs)"
    )


def test_validation_estimate_marks_sparse_history_low_confidence() -> None:
    phase = validate.Phase("tests", ["pytest"], "pytest")
    timing = validate.PhaseTiming(samples=(60.0, 75.0))

    estimate = validate._estimate_for_phase(phase, timing)

    assert estimate == validate.PhaseEstimate(
        lower_seconds=83.0,
        upper_seconds=143.0,
        confidence="low",
        sample_count=2,
    )
    assert validate._format_phase_estimate(estimate, "for this phase") == (
        "low confidence ~1m 23s-2m 23s for this phase"
    )


def test_validation_overall_estimate_sums_remaining_ranges() -> None:
    phases = [
        validate.Phase("tests", ["pytest"], "pytest"),
        validate.Phase("typecheck", ["mypy"], "mypy"),
    ]
    estimates = {
        validate._phase_timing_key(phases[0]): validate.PhaseEstimate(
            30.0,
            60.0,
            "high",
            sample_count=4,
        ),
        validate._phase_timing_key(phases[1]): validate.PhaseEstimate(
            40.0,
            100.0,
            "low",
            sample_count=1,
        ),
    }

    assert validate._overall_estimate(
        phases=phases,
        phase_estimates=estimates,
        current_index=0,
    ) == validate.PhaseEstimate(
        lower_seconds=70.0,
        upper_seconds=160.0,
        confidence="low",
        sample_count=1,
    )


def test_validation_timing_key_includes_command_identity() -> None:
    targeted_tests = validate.Phase(
        "tests",
        ["pytest", "tests/unit/codex/test_validate.py", "-q"],
        "pytest",
    )
    full_tests = validate.Phase(
        "tests",
        ["pytest", "tests/unit", "tests/web", "-q"],
        "pytest",
    )

    assert validate._phase_timing_key(targeted_tests) != validate._phase_timing_key(
        full_tests
    )


def test_validation_duration_cache_loads_legacy_and_structured_timings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "validation-durations.json"
    cache_path.write_text(
        """
        {
          "repo": {
            "tests": 120.0,
            "typecheck": {
              "samples": [40.0, 50.0, "bad", -5.0],
              "count": 99,
              "mean_seconds": 999.0
            }
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(validate, "_DURATION_CACHE_PATH", cache_path)

    assert validate._load_duration_cache() == {
        "repo": {
            "tests": validate.PhaseTiming(samples=(120.0,)),
            "typecheck": validate.PhaseTiming(samples=(40.0, 50.0, 0.0)),
        }
    }


def test_successful_validation_phase_records_recent_timing_samples() -> None:
    phase = validate.Phase("tests", ["pytest", "tests/unit", "-q"], "pytest")
    timing_key = validate._phase_timing_key(phase)
    cache = {
        "repo": {
            timing_key: validate.PhaseTiming(
                samples=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
            )
        }
    }
    result = validate.PhaseResult(
        name=phase.name,
        command=phase.command,
        returncode=0,
        output="",
        duration_seconds=90.0,
        timing_key=timing_key,
    )

    validate._record_phase_duration(cache, "repo", result)

    assert cache["repo"][timing_key] == validate.PhaseTiming(
        samples=(20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)
    )


def test_failure_summaries_are_concise_for_pytest_mypy_and_ruff(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    cases = [
        (
            "tests",
            "pytest",
            "noise\n________ test_app_fails ________\nE   AssertionError: nope\n"
            "FAILED tests/unit/test_app.py::test_app_fails - AssertionError\n",
            [
                "tests: failed",
                "  test_app_fails",
                "  E   AssertionError: nope",
                "  FAILED tests/unit/test_app.py::test_app_fails - AssertionError",
                f"  full log: {tmp_path / 'pytest.log'}",
            ],
        ),
        (
            "typecheck",
            "mypy",
            "noise\nbragi/app.py:4:5: error: Bad type\nFound 1 error\n",
            [
                "typecheck: failed",
                "  bragi/app.py:4:5: error: Bad type",
                f"  full log: {tmp_path / 'mypy.log'}",
            ],
        ),
        (
            "linting",
            "ruff",
            "noise\nbragi/app.py:4:5: F401 unused import\n",
            [
                "linting: failed",
                "  bragi/app.py:4:5: F401 unused import",
                f"  full log: {tmp_path / 'ruff.log'}",
            ],
        ),
    ]

    for name, parser, output, expected in cases:
        result = validate.PhaseResult(
            name=name,
            command=["tool"],
            returncode=1,
            output=output,
            log_path=tmp_path / f"{parser}.log",
        )

        validate._print_summary(result, parser)

        assert capsys.readouterr().out.splitlines() == expected


@pytest.mark.parametrize(
    ("argv", "expected_phase_names"),
    [
        (["--tests-only"], ["tests"]),
        (["--typecheck-only"], ["typecheck"]),
        (["--lint-only"], ["linting"]),
        (
            ["--frontend-only"],
            [
                "frontend dependencies",
                "frontend tests",
                "frontend build",
                "frontend audit",
            ],
        ),
        (["--coverage"], ["coverage", "coverage report"]),
        (
            ["--full"],
            [
                "frontend dependencies",
                "queued tests",
                "tests",
                "integration tests",
                "typecheck",
                "linting",
                "frontend tests",
                "frontend build",
                "frontend audit",
            ],
        ),
    ],
)
def test_focused_flags_only_run_requested_phase(
    argv: list[str],
    expected_phase_names: list[str],
) -> None:
    args = validate._parse_args(argv)

    assert [phase.name for phase in validate._selected_phases(args)] == (
        expected_phase_names
    )


def test_full_mode_installs_frontend_dependencies_before_full_validation() -> None:
    args = argparse.Namespace(
        tests_only=None,
        typecheck_only=False,
        lint_only=False,
        frontend_only=False,
        coverage=False,
        changed=False,
        full=True,
    )

    assert [phase.name for phase in validate._selected_phases(args)] == [
        "frontend dependencies",
        "queued tests",
        "tests",
        "integration tests",
        "typecheck",
        "linting",
        "frontend tests",
        "frontend build",
        "frontend audit",
    ]


def test_frontend_validation_runs_moderate_audit_gate() -> None:
    phases = validate._selected_phases(validate._parse_args(["--frontend-only"]))
    audit_phase = phases[-1]

    assert audit_phase.name == "frontend audit"
    assert audit_phase.command == [
        "npm",
        "audit",
        "--prefix",
        "frontend",
        "--audit-level=moderate",
    ]
    assert audit_phase.parser == "frontend"


def test_frontend_only_checks_frontend_lockfile_consistency() -> None:
    phases = validate._selected_phases(validate._parse_args(["--frontend-only"]))
    dependency_phase = phases[0]

    assert dependency_phase.name == "frontend dependencies"
    assert dependency_phase.command == ["npm", "ci", "--prefix", "frontend"]
    assert dependency_phase.parser == "frontend"


def test_full_validation_checks_frontend_lockfile_consistency() -> None:
    phases = validate._selected_phases(validate._parse_args(["--full"]))
    dependency_phase = next(
        phase for phase in phases if phase.name == "frontend dependencies"
    )

    assert dependency_phase.command == ["npm", "ci", "--prefix", "frontend"]
    assert dependency_phase.parser == "frontend"


@pytest.mark.parametrize("value", ["0", "1", "false", "off"])
def test_pytest_workers_env_can_disable_parallel_args(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAGI_PYTEST_WORKERS", value)

    phases = validate._selected_phases(validate._parse_args(["--tests-only"]))

    assert "-n" not in phases[0].command


def test_default_mode_runs_integration_tests_but_tests_only_stays_unit_only() -> None:
    full_args = validate._parse_args(["--full"])
    tests_only_args = validate._parse_args(["--tests-only"])

    default_phases = validate._selected_phases(full_args)
    tests_only_phases = validate._selected_phases(tests_only_args)

    assert any(
        phase.name == "integration tests"
        and "tests/integration" in phase.command
        for phase in default_phases
    )
    assert [phase.name for phase in tests_only_phases] == ["tests"]
    assert "tests/unit" in tests_only_phases[0].command
    assert "tests/web" in tests_only_phases[0].command
    assert "tests/integration" not in tests_only_phases[0].command


def test_tests_only_accepts_explicit_test_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)
    test_file = tmp_path / "tests" / "unit" / "codex" / "test_validate.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    phases = validate._selected_phases(
        validate._parse_args(["--tests-only", "tests/unit/codex/test_validate.py"]),
        tmp_path,
    )

    assert [phase.name for phase in phases] == ["tests"]
    assert phases[0].command == [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/codex/test_validate.py",
        "-q",
        "-n",
        "auto",
    ]


def test_tests_only_accepts_multiple_explicit_test_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)
    test_paths = [
        Path("tests/unit/codex/test_validate.py"),
        Path("tests/web/test_api.py"),
    ]
    for test_path in test_paths:
        path = tmp_path / test_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    phases = validate._selected_phases(
        validate._parse_args(["--tests-only", *map(str, test_paths)]),
        tmp_path,
    )

    assert phases[0].command[-5:] == [
        "tests/unit/codex/test_validate.py",
        "tests/web/test_api.py",
        "-q",
        "-n",
        "auto",
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["tests/unit/codex/test_validate.py"],
        ["--full", "tests/unit/codex/test_validate.py"],
    ],
)
def test_test_paths_require_tests_only_flag(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        validate._parse_args(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "raw_path",
    [
        "tests/unit/codex",
        "tests/unit/codex/missing.py",
        "tests/unit/codex/helper.py",
        "bragi/app.py",
        "../outside/test_app.py",
        "tests/unit/codex/test_validate.py::test_specific",
    ],
)
def test_tests_only_rejects_invalid_explicit_test_paths(
    raw_path: str,
    tmp_path: Path,
) -> None:
    (tmp_path / "tests" / "unit" / "codex").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "codex" / "test_validate.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "unit" / "codex" / "helper.py").write_text(
        "",
        encoding="utf-8",
    )
    (tmp_path / "bragi").mkdir()
    (tmp_path / "bragi" / "app.py").write_text("", encoding="utf-8")

    args = validate._parse_args(["--tests-only", raw_path])

    with pytest.raises(validate.ValidationArgumentError):
        validate._selected_phases(args, tmp_path)


def test_default_mode_runs_changed_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_target = tmp_path / "tests" / "unit" / "test_app.py"
    lint_target = tmp_path / "bragi" / "app.py"
    test_target.parent.mkdir(parents=True)
    lint_target.parent.mkdir(parents=True)
    test_target.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    lint_target.write_text("", encoding="utf-8")
    selection = validate.validation_targets.ValidationSelection(
        test_targets=(test_target.resolve(),),
        lint_targets=(lint_target.resolve(),),
        typecheck=True,
    )
    monkeypatch.setattr(
        validate.validation_targets,
        "queued_targets_for_repo",
        lambda _repo: [],
    )
    monkeypatch.setattr(
        validate.validation_targets,
        "select_changed_validation",
        lambda _repo, _queued: selection,
    )

    phases = validate._selected_phases(validate._parse_args([]), tmp_path)

    assert [phase.name for phase in phases] == [
        "changed tests",
        "typecheck",
        "changed linting",
    ]
    assert phases[0].command[-4:] == ["tests/unit/test_app.py", "-q", "-n", "auto"]
    assert phases[2].command[-2:] == ["check", "bragi/app.py"]


def test_default_mode_flushes_queued_tests_when_nothing_else_changed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queued_target = tmp_path / "tests" / "unit" / "test_app.py"
    monkeypatch.setattr(
        validate.validation_targets,
        "queued_targets_for_repo",
        lambda _repo: [queued_target],
    )
    monkeypatch.setattr(
        validate.validation_targets,
        "select_changed_validation",
        lambda _repo, _queued: validate.validation_targets.ValidationSelection(),
    )

    phases = validate._selected_phases(validate._parse_args([]), tmp_path)

    assert [phase.name for phase in phases] == ["queued tests"]


def test_changed_flag_matches_default_changed_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection = validate.validation_targets.ValidationSelection(frontend=True)
    monkeypatch.setattr(
        validate.validation_targets,
        "queued_targets_for_repo",
        lambda _repo: [],
    )
    monkeypatch.setattr(
        validate.validation_targets,
        "select_changed_validation",
        lambda _repo, _queued: selection,
    )

    default_phases = validate._selected_phases(validate._parse_args([]), tmp_path)
    changed_phases = validate._selected_phases(
        validate._parse_args(["--changed"]),
        tmp_path,
    )

    assert [phase.name for phase in default_phases] == [
        "frontend dependencies",
        "frontend tests",
        "frontend build",
        "frontend audit",
    ]
    assert [phase.name for phase in changed_phases] == [
        "frontend dependencies",
        "frontend tests",
        "frontend build",
        "frontend audit",
    ]


def test_changed_validation_escalates_to_full_for_broad_risk_change(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    selection = validate.validation_targets.ValidationSelection(
        full=True,
        reasons=("pyproject.toml",),
    )
    monkeypatch.setattr(
        validate.validation_targets,
        "queued_targets_for_repo",
        lambda _repo: [],
    )
    monkeypatch.setattr(
        validate.validation_targets,
        "select_changed_validation",
        lambda _repo, _queued: selection,
    )

    phases = validate._selected_phases(validate._parse_args([]), tmp_path)

    assert [phase.name for phase in phases] == [
        "frontend dependencies",
        "queued tests",
        "tests",
        "integration tests",
        "typecheck",
        "linting",
        "frontend tests",
        "frontend build",
        "frontend audit",
    ]
    assert "escalating to full validation (pyproject.toml)" in capsys.readouterr().out


def test_ci_delegates_frontend_dependency_install_to_full_validation() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Install frontend dependencies" not in workflow
    assert "npm ci --prefix frontend" not in workflow
    assert validate._selected_phases(validate._parse_args(["--full"]))[0] == (
        validate._frontend_dependency_phase()
    )


def test_ci_uses_smart_pr_validation_and_skips_merge_duplicates() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert (
        "group: ci-${{ github.event_name == 'pull_request' && "
        "github.event.pull_request.number || github.run_id }}"
    ) in workflow
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in workflow
    assert "validation_mode=changed" in workflow
    assert "validation_mode=full" in workflow
    assert "validation_mode=skip" in workflow
    assert "Merge\\ pull\\ request\\ *" in workflow
    assert "python3 .codex/tools/validate.py --changed" in workflow
    assert "python3 .codex/tools/validate.py --full" in workflow
    assert "BRAGI_VALIDATION_BASE_REF: ${{ github.event.pull_request.base.sha }}" in (
        workflow
    )
    assert "docker_changed" in workflow
    assert "app_changed=true\n                break" not in workflow


def test_main_returns_success_when_smart_validation_has_no_targets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(validate, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(validate, "_selected_phases", lambda _args, _repo: [])

    assert validate.main([]) == 0

    assert capsys.readouterr().out == (
        "smart validation: no changed validation targets\n"
    )


def test_main_fails_when_agent_backup_files_exist(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    backup_paths = [
        tmp_path / "CLAUDE.md.bak",
        tmp_path / ".cursor" / "rules" / "agentsync.md.bak",
    ]
    for backup_path in backup_paths:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text("stale agent docs\n", encoding="utf-8")
    monkeypatch.setattr(validate, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        validate,
        "_selected_phases",
        lambda _args, _repo: pytest.fail("validation phases should not run"),
    )

    assert validate.main([]) == 1

    assert capsys.readouterr().out.splitlines() == [
        "agent backup hygiene: failed",
        "  remove stale backup file: .cursor/rules/agentsync.md.bak",
        "  remove stale backup file: CLAUDE.md.bak",
    ]


def test_coverage_mode_runs_tests_before_report() -> None:
    phases = validate._selected_phases(validate._parse_args(["--coverage"]))

    assert phases[0].name == "coverage"
    assert phases[0].command[-10:] == [
        "tests/unit",
        "tests/web",
        "tests/integration",
        "-q",
        "-n",
        "auto",
        "--cov=bragi",
        "--cov=bragi_common",
        "--cov=bragi_web",
        "--cov-report=",
    ]
    assert phases[0].parser == "pytest"
    assert phases[1].name == "coverage report"
    assert phases[1].command[-3:] == ["report", "--skip-covered", "--sort=cover"]
    assert phases[1].parser == "coverage"


def test_successful_coverage_report_prints_report_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = validate.PhaseResult(
        name="coverage report",
        command=["coverage", "report"],
        returncode=0,
        output="Name    Stmts   Miss  Cover\nbragi/app.py  10  2  80%\n",
    )

    validate._print_summary(result, "coverage")

    assert capsys.readouterr().out.splitlines() == [
        "coverage report: passed",
        "  Name    Stmts   Miss  Cover",
        "  bragi/app.py  10  2  80%",
    ]
