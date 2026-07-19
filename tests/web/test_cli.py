from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from typing import Any

from pytest import CaptureFixture, MonkeyPatch, raises

from bragi_web import __version__
from bragi_web import main as cli
from bragi_web.bragi_adapter import BragiCompatibilityError
from bragi_web.storage import StorageConfigurationError


class _FakeBindings:
    def ensure_private_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)


class _RunningProcess:
    def poll(self) -> None:
        return None


def test_compatibility_cli_preserves_version_flag(
    capsys: CaptureFixture[str],
) -> None:
    with raises(SystemExit) as excinfo:
        cli.main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out == f"Bragi {__version__}\n"


def test_cli_version_can_be_explicitly_disabled() -> None:
    with raises(SystemExit) as excinfo:
        cli.main(["--version"], version=None)

    assert excinfo.value.code == 2


def test_restart_stops_then_starts_with_start_options(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, argparse.Namespace | None]] = []

    def fake_stop() -> int:
        calls.append(("stop", None))
        return 0

    def fake_start(args: argparse.Namespace) -> int:
        calls.append(("start", args))
        return 0

    monkeypatch.setattr(cli, "_stop", fake_stop)
    monkeypatch.setattr(cli, "_start", fake_start)
    monkeypatch.setattr(cli, "_find_port_listeners", lambda port: ())
    monkeypatch.setattr(cli, "_repo_root", lambda: Path("/tmp/bragi"))

    assert cli.main(
        [
            "restart",
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--frontend-host",
            "127.0.0.1",
            "--frontend-port",
            "5174",
            "--frontend-dir",
            "/tmp/bragi-frontend",
            "--reload",
        ]
    ) == 0

    assert [name for name, _ in calls] == ["stop", "start"]
    start_args = calls[1][1]
    assert start_args is not None
    assert start_args.command == "restart"
    assert start_args.host == "127.0.0.1"
    assert start_args.port == 9001
    assert start_args.frontend_host == "127.0.0.1"
    assert start_args.frontend_port == 5174
    assert start_args.frontend_dir == "/tmp/bragi-frontend"
    assert start_args.reload is True


def test_restart_stops_unmanaged_bragi_web_port_listeners(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    (frontend_dir / "package.json").write_text(
        json.dumps({"name": "bragi-web-frontend"}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        port=9001,
        frontend_port=5174,
        frontend_dir=str(frontend_dir),
    )
    calls: list[str] = []
    signalled: list[tuple[int, signal.Signals]] = []
    running = {1234, 5678}

    listeners = {
        9001: (
            cli.PortListener(
                pid=1234,
                port=9001,
                command=("python", "-m", "uvicorn", "bragi_web.api.app:create_app"),
                cwd=tmp_path,
            ),
        ),
        5174: (
            cli.PortListener(
                pid=5678,
                port=5174,
                command=("node", str(frontend_dir / "node_modules/vite/bin/vite.js")),
                cwd=frontend_dir,
            ),
        ),
    }

    def fake_signal(pid: int, sig: signal.Signals) -> None:
        signalled.append((pid, sig))
        running.discard(pid)

    def fake_stop() -> int:
        calls.append("stop")
        return 0

    def fake_start(received: argparse.Namespace) -> int:
        calls.append("start")
        return 0

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "_stop", fake_stop)
    monkeypatch.setattr(cli, "_start", fake_start)
    monkeypatch.setattr(
        cli,
        "_find_port_listeners",
        lambda port: listeners.get(port, ()),
    )
    monkeypatch.setattr(cli, "_signal_pid", fake_signal)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: pid in running)
    monkeypatch.setattr(cli, "_wait_for_exit", lambda pid: None)

    assert cli._restart(args) == 0

    assert calls == ["stop", "start"]
    assert signalled == [(1234, signal.SIGTERM), (5678, signal.SIGTERM)]
    captured = capsys.readouterr()
    assert "Stopped unmanaged backend listener on port 9001 (1234)" in captured.out
    assert "Stopped unmanaged frontend listener on port 5174 (5678)" in captured.out
    assert captured.err == ""


def test_restart_stops_bragi_frontend_from_another_checkout(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    current_checkout = tmp_path / "current"
    current_frontend = current_checkout / "frontend"
    other_frontend = tmp_path / "other" / "frontend"
    other_frontend.mkdir(parents=True)
    (other_frontend / "package.json").write_text(
        json.dumps({"name": "bragi-web-frontend"}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        port=9001,
        frontend_port=5174,
        frontend_dir=str(current_frontend),
        frontend_mode="static",
    )
    calls: list[str] = []
    signalled: list[tuple[int, signal.Signals]] = []
    running = {5678}
    listener = cli.PortListener(
        pid=5678,
        port=5174,
        command=("node", str(other_frontend / "node_modules/.bin/vite")),
        cwd=other_frontend,
    )

    def fake_signal(pid: int, sig: signal.Signals) -> None:
        signalled.append((pid, sig))
        running.discard(pid)

    def fake_stop() -> int:
        calls.append("stop")
        return 0

    def fake_start(received: argparse.Namespace) -> int:
        calls.append("start")
        return 0

    monkeypatch.setattr(cli, "_repo_root", lambda: current_checkout)
    monkeypatch.setattr(cli, "_stop", fake_stop)
    monkeypatch.setattr(cli, "_start", fake_start)
    monkeypatch.setattr(
        cli,
        "_find_port_listeners",
        lambda port: (listener,) if port == 5174 else (),
    )
    monkeypatch.setattr(cli, "_signal_pid", fake_signal)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: pid in running)
    monkeypatch.setattr(cli, "_wait_for_exit", lambda pid: None)

    assert cli._restart(args) == 0

    assert calls == ["stop", "start"]
    assert signalled == [(5678, signal.SIGTERM)]
    captured = capsys.readouterr()
    assert "Stopped unmanaged frontend listener on port 5174 (5678)" in captured.out
    assert captured.err == ""


def test_restart_leaves_managed_listeners_for_stop(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    (frontend_dir / "package.json").write_text(
        json.dumps({"name": "bragi-web-frontend"}),
        encoding="utf-8",
    )
    backend_process = cli.ManagedProcess(
        "backend",
        tmp_path / "backend.pid",
        tmp_path / "backend.log",
    )
    frontend_process = cli.ManagedProcess(
        "frontend",
        tmp_path / "frontend.pid",
        tmp_path / "frontend.log",
    )
    backend_process.pid_path.write_text(
        json.dumps(
            {
                "pid": 1234,
                "name": "backend",
                "command": ["python", "-m", "uvicorn"],
                "cwd": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    frontend_process.pid_path.write_text(
        json.dumps(
            {
                "pid": 5678,
                "name": "frontend",
                "command": ["npm", "run", "dev"],
                "cwd": str(frontend_dir),
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        port=9001,
        frontend_port=5174,
        frontend_dir=str(frontend_dir),
    )
    calls: list[str] = []
    signalled: list[tuple[int, signal.Signals]] = []
    listeners = {
        9001: (
            cli.PortListener(
                pid=1234,
                port=9001,
                command=("python", "-m", "uvicorn"),
                cwd=tmp_path,
            ),
        ),
        5174: (
            cli.PortListener(
                pid=6789,
                port=5174,
                command=("node", str(frontend_dir / "node_modules/.bin/vite")),
                cwd=frontend_dir,
            ),
        ),
    }

    def fake_stop() -> int:
        calls.append("stop")
        return 0

    def fake_start(received: argparse.Namespace) -> int:
        calls.append("start")
        return 0

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        cli,
        "_managed_processes",
        lambda: (backend_process, frontend_process),
    )
    monkeypatch.setattr(cli, "_stop", fake_stop)
    monkeypatch.setattr(cli, "_start", fake_start)
    monkeypatch.setattr(
        cli,
        "_find_port_listeners",
        lambda port: listeners.get(port, ()),
    )
    monkeypatch.setattr(cli, "_pid_running", lambda pid: True)
    monkeypatch.setattr(
        cli,
        "_process_cmdline",
        lambda pid: ("python", "-m", "uvicorn")
        if pid == 1234
        else ("npm", "run", "dev"),
    )
    monkeypatch.setattr(
        cli,
        "_process_cwd",
        lambda pid: tmp_path if pid == 1234 else frontend_dir,
    )
    monkeypatch.setattr(cli, "_process_start_time", lambda pid: None)
    monkeypatch.setattr(
        cli,
        "_signal_pid",
        lambda pid, sig: signalled.append((pid, sig)),
    )

    assert cli._restart(args) == 0

    assert calls == ["stop", "start"]
    assert signalled == []


def test_restart_refuses_unrelated_port_listener(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    args = argparse.Namespace(
        port=9001,
        frontend_port=5174,
        frontend_dir=str(tmp_path / "frontend"),
    )
    calls: list[str] = []
    listener = cli.PortListener(
        pid=4321,
        port=9001,
        command=("python", "-m", "http.server", "9001"),
        cwd=tmp_path,
    )

    def fake_stop() -> int:
        calls.append("stop")
        return 0

    def fake_start(received: argparse.Namespace) -> int:
        calls.append("start")
        return 0

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path / "other")
    monkeypatch.setattr(cli, "_stop", fake_stop)
    monkeypatch.setattr(cli, "_start", fake_start)
    monkeypatch.setattr(
        cli,
        "_find_port_listeners",
        lambda port: (listener,) if port == 9001 else (),
    )

    assert cli._restart(args) == 1

    assert calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Backend port 9001 is already in use by pid 4321" in captured.err
    assert "python -m http.server 9001" in captured.err
    assert "only auto-stops Bragi Web listeners" in captured.err


def test_cli_reports_invalid_relative_data_dir(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    monkeypatch.setenv("BRAGI_WEB_DATA_DIR", "relative/bragi-web-data")

    assert cli.main(["status"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "BRAGI_WEB_DATA_DIR must be an absolute path" in captured.err
    assert "relative/bragi-web-data" in captured.err


def test_cli_reports_bragi_compatibility_errors(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    def fail_bindings() -> object:
        raise BragiCompatibilityError(
            "Bragi Web requires compatible Bragi application modules"
        )

    monkeypatch.setattr(cli, "bragi_runtime_bindings", fail_bindings)

    assert cli.main(["--host", "127.0.0.1", "--port", "9999"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Bragi Web requires compatible Bragi application modules" in captured.err


def test_frontend_mode_rejects_invalid_values() -> None:
    try:
        cli._frontend_mode(argparse.Namespace(frontend_mode="wat"))
    except StorageConfigurationError as exc:
        assert "BRAGI_WEB_FRONTEND_MODE must be 'dev' or 'static'" in str(exc)
    else:
        raise AssertionError("invalid frontend mode was accepted")


def test_start_missing_frontend_dir_explains_source_checkout(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    processes = (
        cli.ManagedProcess(
            "backend",
            tmp_path / "backend.pid",
            tmp_path / "backend.log",
        ),
        cli.ManagedProcess(
            "frontend",
            tmp_path / "frontend.pid",
            tmp_path / "frontend.log",
        ),
    )
    args = argparse.Namespace(
        host="127.0.0.1",
        port=8787,
        frontend_host="127.0.0.1",
        frontend_port=5173,
        frontend_dir=str(tmp_path / "missing-frontend"),
        frontend_mode="dev",
        build_frontend=False,
        reload=False,
    )
    monkeypatch.setattr(cli, "_managed_processes", lambda: processes)
    monkeypatch.setattr(cli, "_find_port_listeners", lambda port: ())

    assert cli._start(args) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Frontend source directory not found" in captured.err
    assert "`bragi-web start` is for source checkouts" in captured.err


def test_start_prints_bind_addresses_and_local_urls_for_wildcard_hosts(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    monkeypatch.delenv("BRAGI_WEB_BOOTSTRAP_TOKEN", raising=False)
    monkeypatch.setattr(cli.secrets, "token_urlsafe", lambda _bytes: "setup-token")
    args = _prepare_successful_start(monkeypatch, tmp_path)

    assert cli._start(args) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Backend bind:   0.0.0.0:8787" in captured.out
    assert "Frontend bind:  0.0.0.0:5173" in captured.out
    assert "Backend URL:    http://127.0.0.1:8787" in captured.out
    assert "Frontend URL:   http://127.0.0.1:5173" in captured.out
    assert "Remote bootstrap setup token: setup-token" in captured.out
    assert "http://0.0.0.0" not in captured.out


def test_start_prints_explicit_localhost_browser_urls(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    monkeypatch.delenv("BRAGI_WEB_BOOTSTRAP_TOKEN", raising=False)
    args = _prepare_successful_start(
        monkeypatch,
        tmp_path,
        host="127.0.0.1",
        port=9001,
        frontend_host="localhost",
        frontend_port=5174,
    )

    assert cli._start(args) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Backend bind:   127.0.0.1:9001" in captured.out
    assert "Frontend bind:  localhost:5174" in captured.out
    assert "Backend URL:    http://127.0.0.1:9001" in captured.out
    assert "Frontend URL:   http://localhost:5174" in captured.out
    assert "Remote bootstrap setup token" not in captured.out


def test_start_passes_frontend_bind_env_to_backend(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("BRAGI_WEB_BOOTSTRAP_TOKEN", raising=False)
    monkeypatch.setattr(cli.secrets, "token_urlsafe", lambda _bytes: "setup-token")
    args = _prepare_successful_start(
        monkeypatch,
        tmp_path,
        host="0.0.0.0",
        port=9001,
        frontend_host="192.168.1.20",
        frontend_port=5174,
    )
    spawned_envs: list[tuple[str, dict[str, str]]] = []

    def fake_spawn(
        process: cli.ManagedProcess,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> _RunningProcess:
        spawned_envs.append((process.name, dict(env)))
        return _RunningProcess()

    monkeypatch.setattr(cli, "_spawn", fake_spawn)

    assert cli._start(args) == 0

    backend_env = next(env for name, env in spawned_envs if name == "backend")
    assert backend_env["BRAGI_WEB_HOST"] == "0.0.0.0"
    assert backend_env["BRAGI_WEB_PORT"] == "9001"
    assert backend_env["BRAGI_WEB_FRONTEND_HOST"] == "192.168.1.20"
    assert backend_env["BRAGI_WEB_FRONTEND_PORT"] == "5174"
    assert backend_env["BRAGI_WEB_FRONTEND_MODE"] == "dev"
    assert backend_env["BRAGI_WEB_BOOTSTRAP_TOKEN"] == "setup-token"


def test_start_passes_configured_allowed_hosts_to_vite(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BRAGI_WEB_ALLOWED_HOSTS", "bragi.home, app.bragi.test")
    args = _prepare_successful_start(monkeypatch, tmp_path)
    spawned_envs: list[tuple[str, dict[str, str]]] = []

    def fake_spawn(
        process: cli.ManagedProcess,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> _RunningProcess:
        spawned_envs.append((process.name, dict(env)))
        return _RunningProcess()

    monkeypatch.setattr(cli, "_spawn", fake_spawn)

    assert cli._start(args) == 0

    frontend_env = next(env for name, env in spawned_envs if name == "frontend")
    assert (
        frontend_env["__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS"]
        == "bragi.home,app.bragi.test"
    )


def test_start_cleans_backend_when_frontend_spawn_raises(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    args = _prepare_successful_start(monkeypatch, tmp_path, host="127.0.0.1")
    backend_process, _frontend_process = cli._managed_processes()
    running = {2468}
    signalled: list[tuple[int, signal.Signals]] = []

    class FakePopen:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self) -> None:
            return None

    def fake_signal(pid: int, sig: signal.Signals) -> None:
        signalled.append((pid, sig))
        if sig == signal.SIGTERM:
            running.discard(pid)

    def fake_spawn(
        process: cli.ManagedProcess,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> FakePopen:
        if process.name == "frontend":
            raise OSError("frontend spawn failed")
        process.pid_path.write_text("backend pid metadata", encoding="utf-8")
        return FakePopen(2468)

    monkeypatch.setattr(cli, "_spawn", fake_spawn)
    monkeypatch.setattr(cli, "_signal_process", fake_signal)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: pid in running)
    monkeypatch.setattr(cli, "_wait_for_exit", lambda pid: None)

    assert cli._start(args) == 1

    assert signalled == [(2468, signal.SIGTERM)]
    assert not backend_process.pid_path.exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Failed to start frontend: frontend spawn failed" in captured.err


def test_start_cleans_spawned_processes_when_process_exits_quickly(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    args = _prepare_successful_start(monkeypatch, tmp_path, host="127.0.0.1")
    backend_process, frontend_process = cli._managed_processes()
    running = {1357}
    signalled: list[tuple[int, signal.Signals]] = []

    class FakePopen:
        def __init__(self, pid: int, status: int | None) -> None:
            self.pid = pid
            self._status = status

        def poll(self) -> int | None:
            return self._status

    def fake_signal(pid: int, sig: signal.Signals) -> None:
        if pid not in running:
            return
        signalled.append((pid, sig))
        if sig == signal.SIGTERM:
            running.discard(pid)

    def fake_spawn(
        process: cli.ManagedProcess,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> FakePopen:
        if process.name == "backend":
            process.pid_path.write_text("backend pid metadata", encoding="utf-8")
            return FakePopen(2468, 1)
        process.pid_path.write_text("frontend pid metadata", encoding="utf-8")
        return FakePopen(1357, None)

    monkeypatch.setattr(cli, "_spawn", fake_spawn)
    monkeypatch.setattr(cli, "_signal_process", fake_signal)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: pid in running)
    monkeypatch.setattr(cli, "_wait_for_exit", lambda pid: None)

    assert cli._start(args) == 1

    assert signalled == [(1357, signal.SIGTERM)]
    assert not backend_process.pid_path.exists()
    assert not frontend_process.pid_path.exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Failed to start: backend" in captured.err


def test_start_refuses_existing_port_listener(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    args = _prepare_successful_start(monkeypatch, tmp_path)
    listener = cli.PortListener(
        pid=2468,
        port=args.port,
        command=("python", "-m", "uvicorn", "other.app:create_app"),
        cwd=tmp_path,
    )
    spawned: list[cli.ManagedProcess] = []

    def fake_spawn(
        process: cli.ManagedProcess,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> _RunningProcess:
        spawned.append(process)
        return _RunningProcess()

    monkeypatch.setattr(
        cli,
        "_find_port_listeners",
        lambda port: (listener,) if port == args.port else (),
    )
    monkeypatch.setattr(cli, "_spawn", fake_spawn)

    assert cli._start(args) == 1

    assert spawned == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Backend port 8787 is already in use by pid 2468" in captured.err


def test_start_static_frontend_runs_backend_only(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    processes = (
        cli.ManagedProcess(
            "backend",
            tmp_path / "backend.pid",
            tmp_path / "backend.log",
        ),
        cli.ManagedProcess(
            "frontend",
            tmp_path / "frontend.pid",
            tmp_path / "frontend.log",
        ),
    )
    static_root = tmp_path / "bragi_web" / "static"
    static_root.mkdir(parents=True)
    (static_root / "index.html").write_text("INDEX", encoding="utf-8")
    spawned: list[cli.ManagedProcess] = []
    args = argparse.Namespace(
        host="0.0.0.0",
        port=8787,
        frontend_host="0.0.0.0",
        frontend_port=5173,
        frontend_dir=str(tmp_path / "missing-frontend"),
        frontend_mode="static",
        build_frontend=False,
        reload=False,
    )

    def fake_spawn(
        process: cli.ManagedProcess,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> _RunningProcess:
        spawned.append(process)
        return _RunningProcess()

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "_managed_processes", lambda: processes)
    monkeypatch.setattr(cli, "bragi_runtime_bindings", lambda: _FakeBindings())
    monkeypatch.setattr(cli, "_spawn", fake_spawn)
    monkeypatch.setattr(cli, "_find_port_listeners", lambda port: ())
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)

    assert cli._start(args) == 0

    assert spawned == [processes[0]]
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Backend bind:   0.0.0.0:8787" in captured.out
    assert "Frontend mode:  static assets served by backend" in captured.out
    assert "Frontend URL:   http://127.0.0.1:8787" in captured.out
    assert "Frontend bind:" not in captured.out


def test_start_static_frontend_builds_assets_before_backend(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    frontend_dir = tmp_path / "frontend"
    (frontend_dir / "node_modules").mkdir(parents=True)
    static_root = tmp_path / "bragi_web" / "static"
    processes = (
        cli.ManagedProcess(
            "backend",
            tmp_path / "backend.pid",
            tmp_path / "backend.log",
        ),
        cli.ManagedProcess(
            "frontend",
            tmp_path / "frontend.pid",
            tmp_path / "frontend.log",
        ),
    )
    run_calls: list[tuple[list[str], Path]] = []
    spawned: list[cli.ManagedProcess] = []
    args = argparse.Namespace(
        host="127.0.0.1",
        port=8787,
        frontend_host="127.0.0.1",
        frontend_port=5173,
        frontend_dir=str(frontend_dir),
        frontend_mode="static",
        build_frontend=True,
        reload=False,
    )

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
        run_calls.append((command, cwd))
        static_root.mkdir(parents=True)
        (static_root / "index.html").write_text("INDEX", encoding="utf-8")

    def fake_spawn(
        process: cli.ManagedProcess,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> _RunningProcess:
        spawned.append(process)
        return _RunningProcess()

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "_managed_processes", lambda: processes)
    monkeypatch.setattr(cli, "bragi_runtime_bindings", lambda: _FakeBindings())
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "_spawn", fake_spawn)
    monkeypatch.setattr(cli, "_find_port_listeners", lambda port: ())
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)

    assert cli._start(args) == 0

    assert run_calls == [(["npm", "run", "build"], frontend_dir)]
    assert spawned == [processes[0]]


def test_start_static_frontend_requires_built_assets(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    processes = (
        cli.ManagedProcess(
            "backend",
            tmp_path / "backend.pid",
            tmp_path / "backend.log",
        ),
        cli.ManagedProcess(
            "frontend",
            tmp_path / "frontend.pid",
            tmp_path / "frontend.log",
        ),
    )
    spawned: list[cli.ManagedProcess] = []
    args = argparse.Namespace(
        host="127.0.0.1",
        port=8787,
        frontend_host="127.0.0.1",
        frontend_port=5173,
        frontend_dir=str(tmp_path / "missing-frontend"),
        frontend_mode="static",
        build_frontend=False,
        reload=False,
    )

    def fake_spawn(
        process: cli.ManagedProcess,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> _RunningProcess:
        spawned.append(process)
        return _RunningProcess()

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "_managed_processes", lambda: processes)
    monkeypatch.setattr(cli, "bragi_runtime_bindings", lambda: _FakeBindings())
    monkeypatch.setattr(cli, "_spawn", fake_spawn)
    monkeypatch.setattr(cli, "_find_port_listeners", lambda port: ())

    assert cli._start(args) == 1

    assert spawned == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Built frontend assets not found" in captured.err
    assert "--frontend-mode static --build-frontend" in captured.err


def test_stop_skips_live_unmanaged_pid_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    process = cli.ManagedProcess(
        "backend",
        tmp_path / "backend.pid",
        tmp_path / "backend.log",
    )
    process.pid_path.write_text(
        json.dumps(
            {
                "pid": 4321,
                "name": "backend",
                "command": ["python", "-m", "uvicorn"],
                "cwd": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    signalled: list[tuple[int, signal.Signals]] = []

    monkeypatch.setattr(cli, "_managed_processes", lambda: (process,))
    monkeypatch.setattr(cli, "_pid_running", lambda pid: True)
    monkeypatch.setattr(cli, "_pid_metadata_matches_process", lambda metadata, p: False)
    monkeypatch.setattr(
        cli,
        "_signal_process",
        lambda pid, sig: signalled.append((pid, sig)),
    )

    assert cli._stop() == 0

    assert signalled == []
    assert not process.pid_path.exists()
    assert "Skipped backend unmanaged pid 4321" in capsys.readouterr().out


def test_stop_signals_verified_managed_pid_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    process = cli.ManagedProcess(
        "backend",
        tmp_path / "backend.pid",
        tmp_path / "backend.log",
    )
    process.pid_path.write_text(
        json.dumps(
            {
                "pid": 9876,
                "name": "backend",
                "command": ["python", "-m", "uvicorn"],
                "cwd": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    running = {9876}
    signalled: list[tuple[int, signal.Signals]] = []

    def fake_signal(pid: int, sig: signal.Signals) -> None:
        signalled.append((pid, sig))
        if sig == signal.SIGTERM:
            running.discard(pid)

    monkeypatch.setattr(cli, "_managed_processes", lambda: (process,))
    monkeypatch.setattr(cli, "_pid_running", lambda pid: pid in running)
    monkeypatch.setattr(cli, "_pid_metadata_matches_process", lambda metadata, p: True)
    monkeypatch.setattr(cli, "_signal_process", fake_signal)
    monkeypatch.setattr(cli, "_wait_for_exit", lambda pid: None)

    assert cli._stop() == 0

    assert signalled == [(9876, signal.SIGTERM)]
    assert not process.pid_path.exists()
    assert "Stopped backend (9876)" in capsys.readouterr().out


def test_stop_treats_live_legacy_pid_file_as_unmanaged(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = cli.ManagedProcess(
        "frontend",
        tmp_path / "frontend.pid",
        tmp_path / "frontend.log",
    )
    process.pid_path.write_text("1111", encoding="utf-8")
    signalled: list[tuple[int, signal.Signals]] = []

    monkeypatch.setattr(cli, "_managed_processes", lambda: (process,))
    monkeypatch.setattr(cli, "_pid_running", lambda pid: True)
    monkeypatch.setattr(cli, "_pid_metadata_matches_process", lambda metadata, p: False)
    monkeypatch.setattr(
        cli,
        "_signal_process",
        lambda pid, sig: signalled.append((pid, sig)),
    )

    assert cli._stop() == 0

    assert signalled == []
    assert not process.pid_path.exists()


def test_status_reports_live_unmanaged_pid_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    process = cli.ManagedProcess(
        "backend",
        tmp_path / "backend.pid",
        tmp_path / "backend.log",
    )
    process.pid_path.write_text(
        json.dumps(
            {
                "pid": 1234,
                "name": "backend",
                "command": ["python", "-m", "uvicorn"],
                "cwd": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "_managed_processes", lambda: (process,))
    monkeypatch.setattr(cli, "_pid_running", lambda pid: True)
    monkeypatch.setattr(cli, "_pid_metadata_matches_process", lambda metadata, p: False)

    assert cli._status() == 0

    assert capsys.readouterr().out == "backend: unmanaged/stale (1234)\n"


def test_spawn_writes_managed_pid_metadata(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = cli.ManagedProcess(
        "backend",
        tmp_path / "backend.pid",
        tmp_path / "backend.log",
    )

    class FakePopen:
        pid = 2468

    popen_kwargs: dict[str, Any] = {}

    def fake_popen(*args: Any, **kwargs: Any) -> FakePopen:
        popen_kwargs.update(kwargs)
        return FakePopen()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli, "_process_start_time", lambda pid: 13579)

    popen = cli._spawn(
        process,
        ["python", "-m", "uvicorn"],
        cwd=tmp_path,
        env={"BRAGI_WEB_HOST": "127.0.0.1"},
    )

    assert popen.pid == 2468
    assert popen_kwargs["stdin"] == cli.subprocess.DEVNULL
    assert json.loads(process.pid_path.read_text(encoding="utf-8")) == {
        "pid": 2468,
        "name": "backend",
        "command": ["python", "-m", "uvicorn"],
        "cwd": str(tmp_path),
        "start_time": 13579,
    }


def test_spawn_terminates_child_when_pid_metadata_write_fails(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = cli.ManagedProcess(
        "backend",
        tmp_path / "backend.pid",
        tmp_path / "backend.log",
    )
    running = {2468}
    signalled: list[tuple[int, signal.Signals]] = []

    class FakePopen:
        pid = 2468

    def fake_popen(*args: Any, **kwargs: Any) -> FakePopen:
        return FakePopen()

    def fail_write(path: Path, metadata: cli.ManagedPidMetadata) -> None:
        path.write_text("partial metadata", encoding="utf-8")
        raise OSError("metadata write failed")

    def fake_signal(pid: int, sig: signal.Signals) -> None:
        signalled.append((pid, sig))
        if sig == signal.SIGTERM:
            running.discard(pid)

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli, "_write_pid_metadata", fail_write)
    monkeypatch.setattr(cli, "_signal_process", fake_signal)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: pid in running)
    monkeypatch.setattr(cli, "_wait_for_exit", lambda pid: None)

    with raises(OSError, match="metadata write failed"):
        cli._spawn(
            process,
            ["python", "-m", "uvicorn"],
            cwd=tmp_path,
            env={"BRAGI_WEB_HOST": "127.0.0.1"},
        )

    assert signalled == [(2468, signal.SIGTERM)]
    assert not process.pid_path.exists()


def _prepare_successful_start(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8787,
    frontend_host: str = "0.0.0.0",
    frontend_port: int = 5173,
) -> argparse.Namespace:
    frontend_dir = tmp_path / "frontend"
    (frontend_dir / "node_modules").mkdir(parents=True)
    processes = (
        cli.ManagedProcess(
            "backend",
            tmp_path / "backend.pid",
            tmp_path / "backend.log",
        ),
        cli.ManagedProcess(
            "frontend",
            tmp_path / "frontend.pid",
            tmp_path / "frontend.log",
        ),
    )

    def fake_spawn(
        process: cli.ManagedProcess,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> _RunningProcess:
        return _RunningProcess()

    monkeypatch.setattr(cli, "_managed_processes", lambda: processes)
    monkeypatch.setattr(cli, "bragi_runtime_bindings", lambda: _FakeBindings())
    monkeypatch.setattr(cli, "_spawn", fake_spawn)
    monkeypatch.setattr(cli, "_find_port_listeners", lambda port: ())
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    return argparse.Namespace(
        host=host,
        port=port,
        frontend_host=frontend_host,
        frontend_port=frontend_port,
        frontend_dir=str(frontend_dir),
        frontend_mode="dev",
        build_frontend=False,
        reload=False,
    )
