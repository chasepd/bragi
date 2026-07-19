"""CLI entrypoint and local dev process manager for Bragi Web."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from bragi_web import __version__
from bragi_web.bragi_adapter import BragiCompatibilityError, bragi_runtime_bindings
from bragi_web.storage import StorageConfigurationError, resolve_web_storage_paths

_DEFAULT_BACKEND_PORT = 8787
_DEFAULT_FRONTEND_PORT = 5173
_DEFAULT_HOST = "0.0.0.0"
_FRONTEND_MODE_DEV = "dev"
_FRONTEND_MODE_STATIC = "static"
_BOOTSTRAP_TOKEN_ENV = "BRAGI_WEB_BOOTSTRAP_TOKEN"
_ALLOWED_HOSTS_ENV = "BRAGI_WEB_ALLOWED_HOSTS"
_VITE_ADDITIONAL_ALLOWED_HOSTS_ENV = "__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS"


@dataclass(frozen=True)
class ManagedProcess:
    name: str
    pid_path: Path
    log_path: Path


@dataclass(frozen=True)
class ManagedPidMetadata:
    pid: int
    name: str | None
    command: tuple[str, ...]
    cwd: Path | None
    start_time: int | None


@dataclass(frozen=True)
class PortBinding:
    name: str
    port: int
    cwd: Path


@dataclass(frozen=True)
class PortListener:
    pid: int
    port: int
    command: tuple[str, ...]
    cwd: Path | None


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "bragi-web",
    version: str | None = __version__,
) -> int:
    parser = argparse.ArgumentParser(prog=prog)
    if version is not None:
        parser.add_argument(
            "--version",
            action="version",
            version=f"Bragi {version}",
        )
    subparsers = parser.add_subparsers(dest="command")
    _add_serve_arguments(parser)

    start_parser = subparsers.add_parser(
        "start",
        help="start managed Bragi Web processes in the background",
    )
    _add_start_arguments(start_parser)

    restart_parser = subparsers.add_parser(
        "restart",
        help="restart managed Bragi Web processes in the background",
    )
    _add_start_arguments(restart_parser)

    subparsers.add_parser("stop", help="stop background Bragi Web processes")
    subparsers.add_parser("status", help="show background process status")

    args = parser.parse_args(argv)
    args.prog = prog
    try:
        if args.command == "start":
            return _start(args)
        if args.command == "restart":
            return _restart(args)
        if args.command == "stop":
            return _stop()
        if args.command == "status":
            return _status()
        return _serve(args)
    except (BragiCompatibilityError, StorageConfigurationError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host",
        default=os.environ.get("BRAGI_WEB_HOST", _DEFAULT_HOST),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("BRAGI_WEB_PORT", str(_DEFAULT_BACKEND_PORT))),
    )
    parser.add_argument("--reload", action="store_true")


def _add_start_arguments(parser: argparse.ArgumentParser) -> None:
    _add_serve_arguments(parser)
    parser.add_argument(
        "--frontend-host",
        default=os.environ.get("BRAGI_WEB_FRONTEND_HOST", _DEFAULT_HOST),
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=int(
            os.environ.get("BRAGI_WEB_FRONTEND_PORT", str(_DEFAULT_FRONTEND_PORT))
        ),
    )
    parser.add_argument(
        "--frontend-dir",
        default=str(_repo_root() / "frontend"),
        help="path to the source-checkout Vite frontend directory",
    )
    parser.add_argument(
        "--frontend-mode",
        choices=(_FRONTEND_MODE_DEV, _FRONTEND_MODE_STATIC),
        default=os.environ.get("BRAGI_WEB_FRONTEND_MODE", _FRONTEND_MODE_DEV),
        help=(
            "dev starts the Vite frontend; static serves built SPA assets from "
            "the backend only"
        ),
    )
    parser.add_argument(
        "--build-frontend",
        action="store_true",
        help="run `npm run build` before starting; useful with --frontend-mode static",
    )


def _serve(args: argparse.Namespace) -> int:
    resolve_web_storage_paths()
    bragi_runtime_bindings()
    bootstrap_token = _ensure_bootstrap_token_for_remote_bind(str(args.host))
    if bootstrap_token is not None:
        print(f"Remote bootstrap setup token: {bootstrap_token}")
    uvicorn.run(
        "bragi_web.api.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )
    return 0


def _start(args: argparse.Namespace) -> int:
    processes = _managed_processes()
    stale = [process for process in processes if _is_running_pid_file(process)]
    if stale:
        names = ", ".join(process.name for process in stale)
        print(f"Bragi Web is already running: {names}", file=sys.stderr)
        print(
            f"Run `{_prog(args)} status` or `{_prog(args)} stop`.",
            file=sys.stderr,
        )
        return 1

    bragi_runtime_bindings().ensure_private_dir(processes[0].pid_path.parent)
    frontend_mode = _frontend_mode(args)
    frontend_dir = Path(args.frontend_dir).expanduser().resolve()

    if frontend_mode == _FRONTEND_MODE_DEV or args.build_frontend:
        if not _ensure_frontend_source_dir(frontend_dir, prog=_prog(args)):
            return 1
        _ensure_frontend_dependencies(frontend_dir)
    if args.build_frontend:
        print("Building frontend static assets...")
        subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True)
    if frontend_mode == _FRONTEND_MODE_STATIC and not _static_frontend_available():
        print(
            "Built frontend assets not found. Run "
            f"`{_prog(args)} restart --frontend-mode static --build-frontend`.",
            file=sys.stderr,
        )
        return 1

    bindings = _port_bindings(
        args,
        frontend_dir,
        include_frontend=frontend_mode == _FRONTEND_MODE_DEV,
    )
    blockers = _port_listener_blockers(bindings)
    if blockers:
        _print_port_blockers(blockers, prog=_prog(args))
        return 1

    bootstrap_token = _ensure_bootstrap_token_for_remote_bind(str(args.host))
    env = os.environ.copy()
    env["BRAGI_WEB_HOST"] = str(args.host)
    env["BRAGI_WEB_PORT"] = str(args.port)
    env["BRAGI_WEB_FRONTEND_HOST"] = str(args.frontend_host)
    env["BRAGI_WEB_FRONTEND_PORT"] = str(args.frontend_port)
    env["BRAGI_WEB_FRONTEND_MODE"] = frontend_mode
    _configure_vite_allowed_hosts(env)
    spawned: list[tuple[ManagedProcess, subprocess.Popen[bytes]]] = []
    spawning_process = processes[0]
    try:
        backend = _spawn(
            spawning_process,
            [
                sys.executable,
                "-m",
                "uvicorn",
                "bragi_web.api.app:create_app",
                "--factory",
                "--host",
                str(args.host),
                "--port",
                str(args.port),
                *(("--reload",) if args.reload else ()),
            ],
            cwd=_repo_root(),
            env=env,
        )
        spawned.append((spawning_process, backend))
        if frontend_mode == _FRONTEND_MODE_DEV:
            spawning_process = processes[1]
            frontend = _spawn(
                spawning_process,
                [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--host",
                    str(args.frontend_host),
                    "--port",
                    str(args.frontend_port),
                ],
                cwd=frontend_dir,
                env=env,
            )
            spawned.append((spawning_process, frontend))
    except Exception as exc:  # noqa: BLE001 - report startup failures after cleanup.
        _cleanup_spawned_processes(spawned)
        print(f"Failed to start {spawning_process.name}: {exc}", file=sys.stderr)
        return 1

    time.sleep(0.4)
    failed = [
        process.name
        for process, popen in spawned
        if popen.poll() is not None
    ]
    if failed:
        print(f"Failed to start: {', '.join(failed)}", file=sys.stderr)
        _cleanup_spawned_processes(spawned)
        return 1

    print(f"Backend bind:   {_format_bind_address(str(args.host), args.port)}")
    print(f"Backend URL:    {_format_browser_url(str(args.host), args.port)}")
    if bootstrap_token is not None:
        print(f"Remote bootstrap setup token: {bootstrap_token}")
    if frontend_mode == _FRONTEND_MODE_DEV:
        print(
            f"Frontend bind:  "
            f"{_format_bind_address(str(args.frontend_host), args.frontend_port)}"
        )
        print(
            f"Frontend URL:   "
            f"{_format_browser_url(str(args.frontend_host), args.frontend_port)}"
        )
    else:
        print("Frontend mode:  static assets served by backend")
        print(f"Frontend URL:   {_format_browser_url(str(args.host), args.port)}")
    print(f"Logs:           {processes[0].log_path.parent}")
    return 0


def _frontend_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "frontend_mode", _FRONTEND_MODE_DEV))
    if mode not in {_FRONTEND_MODE_DEV, _FRONTEND_MODE_STATIC}:
        raise StorageConfigurationError(
            "BRAGI_WEB_FRONTEND_MODE must be 'dev' or 'static'"
        )
    return mode


def _ensure_frontend_source_dir(frontend_dir: Path, *, prog: str = "bragi-web") -> bool:
    if frontend_dir.is_dir():
        return True
    print(f"Frontend source directory not found: {frontend_dir}", file=sys.stderr)
    print(
        f"`{prog} start` is for source checkouts; run `{prog}` "
        "to serve packaged SPA assets.",
        file=sys.stderr,
    )
    return False


def _ensure_frontend_dependencies(frontend_dir: Path) -> None:
    if (frontend_dir / "node_modules").is_dir():
        return
    print("Installing frontend dependencies...")
    subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)


def _static_frontend_available() -> bool:
    return (_repo_root() / "bragi_web" / "static" / "index.html").is_file()


def _format_bind_address(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def _format_browser_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if _is_wildcard_host(host) else host
    if ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{port}"


def _is_wildcard_host(host: str) -> bool:
    return host in {"", "0.0.0.0", "::"}


def _ensure_bootstrap_token_for_remote_bind(host: str) -> str | None:
    if not _host_needs_remote_bootstrap_token(host):
        return None
    if os.environ.get(_BOOTSTRAP_TOKEN_ENV, "").strip():
        return None
    token = secrets.token_urlsafe(24)
    os.environ[_BOOTSTRAP_TOKEN_ENV] = token
    return token


def _host_needs_remote_bootstrap_token(host: str) -> bool:
    if _is_wildcard_host(host):
        return True
    if host in {"localhost", "testserver"}:
        return False
    normalized_host = host
    if normalized_host.startswith("[") and normalized_host.endswith("]"):
        normalized_host = normalized_host[1:-1]
    try:
        return not ipaddress.ip_address(normalized_host).is_loopback
    except ValueError:
        return True


def _restart(args: argparse.Namespace) -> int:
    frontend_dir = Path(args.frontend_dir).expanduser().resolve()
    if not _reclaim_owned_port_listeners(
        _port_bindings(
            args,
            frontend_dir,
            include_frontend=True,
        ),
        prog=_prog(args),
    ):
        return 1
    _stop()
    return _start(args)


def _stop() -> int:
    found_pid_file = False
    stopped = False
    for process in reversed(_managed_processes()):
        metadata = _read_pid_metadata(process.pid_path)
        if metadata is None:
            continue
        found_pid_file = True
        if not _pid_running(metadata.pid):
            process.pid_path.unlink(missing_ok=True)
            print(f"Removed stale {process.name} pid file ({metadata.pid})")
            continue
        if not _pid_metadata_matches_process(metadata, process):
            process.pid_path.unlink(missing_ok=True)
            print(f"Skipped {process.name} unmanaged pid {metadata.pid}")
            continue
        _signal_process(metadata.pid, signal.SIGTERM)
        stopped = True
        _wait_for_exit(metadata.pid)
        if _pid_running(metadata.pid) and _pid_metadata_matches_process(
            metadata,
            process,
        ):
            _signal_process(metadata.pid, signal.SIGKILL)
        process.pid_path.unlink(missing_ok=True)
        print(f"Stopped {process.name} ({metadata.pid})")
    if not found_pid_file and not stopped:
        print("Bragi Web is not running")
    return 0


def _cleanup_spawned_processes(
    spawned: list[tuple[ManagedProcess, subprocess.Popen[bytes]]],
) -> None:
    for process, popen in reversed(spawned):
        _terminate_spawned_process(process, popen)


def _terminate_spawned_process(
    process: ManagedProcess,
    popen: subprocess.Popen[bytes],
) -> None:
    try:
        _signal_process(popen.pid, signal.SIGTERM)
        _wait_for_exit(popen.pid)
        if _pid_running(popen.pid):
            _signal_process(popen.pid, signal.SIGKILL)
            _wait_for_exit(popen.pid)
    finally:
        process.pid_path.unlink(missing_ok=True)


def _status() -> int:
    for process in _managed_processes():
        metadata = _read_pid_metadata(process.pid_path)
        if metadata is None or not _pid_running(metadata.pid):
            print(f"{process.name}: stopped")
        elif _pid_metadata_matches_process(metadata, process):
            print(f"{process.name}: running ({metadata.pid})")
        else:
            print(f"{process.name}: unmanaged/stale ({metadata.pid})")
    return 0


def _port_bindings(
    args: argparse.Namespace,
    frontend_dir: Path,
    *,
    include_frontend: bool = True,
) -> tuple[PortBinding, ...]:
    bindings = [PortBinding("backend", int(args.port), _repo_root())]
    if include_frontend:
        bindings.append(PortBinding("frontend", int(args.frontend_port), frontend_dir))
    return tuple(bindings)


def _port_listener_blockers(
    bindings: tuple[PortBinding, ...],
) -> tuple[tuple[PortBinding, PortListener], ...]:
    blockers: list[tuple[PortBinding, PortListener]] = []
    seen: set[tuple[str, int]] = set()
    for binding in bindings:
        for listener in _find_port_listeners(binding.port):
            key = (binding.name, listener.pid)
            if key in seen:
                continue
            seen.add(key)
            blockers.append((binding, listener))
    return tuple(blockers)


def _reclaim_owned_port_listeners(
    bindings: tuple[PortBinding, ...],
    *,
    prog: str = "bragi-web",
) -> bool:
    blockers = _port_listener_blockers(bindings)
    if not blockers:
        return True
    unmanaged: list[tuple[PortBinding, PortListener]] = []
    reclaimed_pids: set[int] = set()
    for binding, listener in blockers:
        if _listener_belongs_to_managed_process(listener, binding):
            continue
        if not _listener_matches_binding(listener, binding):
            unmanaged.append((binding, listener))
            continue
        if listener.pid in reclaimed_pids:
            continue
        _signal_pid(listener.pid, signal.SIGTERM)
        _wait_for_exit(listener.pid)
        if _pid_running(listener.pid):
            _signal_pid(listener.pid, signal.SIGKILL)
            _wait_for_exit(listener.pid)
        reclaimed_pids.add(listener.pid)
        print(
            f"Stopped unmanaged {binding.name} listener "
            f"on port {binding.port} ({listener.pid})"
        )
    if unmanaged:
        _print_port_blockers(tuple(unmanaged), prog=prog)
        return False
    return True


def _listener_belongs_to_managed_process(
    listener: PortListener,
    binding: PortBinding,
) -> bool:
    for process in _managed_processes():
        if process.name != binding.name:
            continue
        metadata = _read_pid_metadata(process.pid_path)
        if metadata is None or not _pid_metadata_matches_process(metadata, process):
            continue
        if listener.pid == metadata.pid:
            return True
        if (
            binding.name == "frontend"
            and listener.cwd is not None
            and metadata.cwd is not None
            and _same_path(listener.cwd, metadata.cwd)
        ):
            return True
    return False


def _configure_vite_allowed_hosts(env: dict[str, str]) -> None:
    allowed_hosts = _csv_env_values(env.get(_ALLOWED_HOSTS_ENV, ""))
    if not allowed_hosts:
        return
    vite_hosts = _csv_env_values(env.get(_VITE_ADDITIONAL_ALLOWED_HOSTS_ENV, ""))
    env[_VITE_ADDITIONAL_ALLOWED_HOSTS_ENV] = ",".join(
        (*vite_hosts, *allowed_hosts)
    )


def _csv_env_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _print_port_blockers(
    blockers: tuple[tuple[PortBinding, PortListener], ...],
    *,
    prog: str = "bragi-web",
) -> None:
    for binding, listener in blockers:
        print(
            f"{binding.name.title()} port {binding.port} is already in use by "
            f"pid {listener.pid}: {_format_listener_command(listener)}",
            file=sys.stderr,
        )
    print(
        f"`{prog} restart` only auto-stops Bragi Web listeners. "
        "Stop the process above or choose another port.",
        file=sys.stderr,
    )


def _listener_matches_binding(listener: PortListener, binding: PortBinding) -> bool:
    if binding.name == "frontend":
        return _command_basename_contains(
            listener.command,
            "vite",
        ) and _is_bragi_frontend_cwd(listener.cwd)
    if listener.cwd is None or not _same_path(listener.cwd, binding.cwd):
        return False
    command_text = "\0".join(listener.command)
    if binding.name == "backend":
        return (
            "bragi_web.api.app:create_app" in command_text
            or _command_basename_contains(listener.command, "bragi-web")
            or _command_basename_contains(listener.command, "bragi")
        )
    return False


def _is_bragi_frontend_cwd(cwd: Path | None) -> bool:
    if cwd is None:
        return False
    package_path = cwd / "package.json"
    try:
        data = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and data.get("name") == "bragi-web-frontend"


def _command_basename_contains(command: tuple[str, ...], needle: str) -> bool:
    return any(needle in Path(part).name for part in command)


def _prog(args: argparse.Namespace) -> str:
    return str(getattr(args, "prog", "bragi-web"))


def _format_listener_command(listener: PortListener) -> str:
    if not listener.command:
        return "unknown command"
    return " ".join(shlex.quote(part) for part in listener.command[:8])


def _find_port_listeners(port: int) -> tuple[PortListener, ...]:
    inodes = _listening_socket_inodes(port)
    if not inodes:
        return ()
    listeners: list[PortListener] = []
    seen_pids: set[int] = set()
    proc_root = Path("/proc")
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in seen_pids:
            continue
        fd_dir = entry / "fd"
        try:
            fds = tuple(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            match = re.fullmatch(r"socket:\[(\d+)\]", target)
            if match is None or match.group(1) not in inodes:
                continue
            listeners.append(
                PortListener(
                    pid=pid,
                    port=port,
                    command=_process_cmdline(pid),
                    cwd=_process_cwd(pid),
                )
            )
            seen_pids.add(pid)
            break
    return tuple(listeners)


def _listening_socket_inodes(port: int) -> frozenset[str]:
    inodes: set[str] = set()
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                port_hex = fields[1].rsplit(":", 1)[1]
                local_port = int(port_hex, 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                inodes.add(fields[9])
    return frozenset(inodes)


def _spawn(
    process: ManagedProcess,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    log_handle = process.log_path.open("ab")
    try:
        popen = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    try:
        _write_pid_metadata(
            process.pid_path,
            ManagedPidMetadata(
                pid=popen.pid,
                name=process.name,
                command=tuple(command),
                cwd=cwd,
                start_time=_process_start_time(popen.pid),
            ),
        )
    except Exception:  # noqa: BLE001 - child must not outlive failed tracking.
        _terminate_spawned_process(process, popen)
        raise
    return popen


def _managed_processes() -> tuple[ManagedProcess, ManagedProcess]:
    run_dir = resolve_web_storage_paths().state_dir / "run"
    return (
        ManagedProcess("backend", run_dir / "backend.pid", run_dir / "backend.log"),
        ManagedProcess("frontend", run_dir / "frontend.pid", run_dir / "frontend.log"),
    )


def _is_running_pid_file(process: ManagedProcess) -> bool:
    metadata = _read_pid_metadata(process.pid_path)
    return (
        metadata is not None
        and _pid_running(metadata.pid)
        and _pid_metadata_matches_process(metadata, process)
    )


def _read_pid_metadata(path: Path) -> ManagedPidMetadata | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _legacy_pid_metadata(text)
    if isinstance(data, dict):
        pid = _positive_int(data.get("pid"))
        if pid is None:
            return None
        name = data.get("name")
        command = data.get("command")
        cwd = data.get("cwd")
        return ManagedPidMetadata(
            pid=pid,
            name=name if isinstance(name, str) and name else None,
            command=_string_tuple(command),
            cwd=Path(cwd) if isinstance(cwd, str) and cwd else None,
            start_time=_positive_int(data.get("start_time")),
        )
    if isinstance(data, int) and not isinstance(data, bool):
        return _legacy_pid_metadata(str(data))
    return _legacy_pid_metadata(text)


def _legacy_pid_metadata(text: str) -> ManagedPidMetadata | None:
    try:
        pid = int(text)
    except ValueError:
        return None
    if pid <= 0:
        return None
    return ManagedPidMetadata(
        pid=pid,
        name=None,
        command=(),
        cwd=None,
        start_time=None,
    )


def _write_pid_metadata(path: Path, metadata: ManagedPidMetadata) -> None:
    path.write_text(
        json.dumps(
            {
                "pid": metadata.pid,
                "name": metadata.name,
                "command": list(metadata.command),
                "cwd": str(metadata.cwd) if metadata.cwd is not None else None,
                "start_time": metadata.start_time,
            }
        ),
        encoding="utf-8",
    )


def _pid_metadata_matches_process(
    metadata: ManagedPidMetadata,
    process: ManagedProcess,
) -> bool:
    if metadata.name != process.name or metadata.cwd is None or not metadata.command:
        return False
    if not _pid_running(metadata.pid):
        return False
    actual_command = _process_cmdline(metadata.pid)
    if not actual_command or not _command_matches(metadata.command, actual_command):
        return False
    actual_cwd = _process_cwd(metadata.pid)
    if actual_cwd is None or not _same_path(metadata.cwd, actual_cwd):
        return False
    if metadata.start_time is not None:
        actual_start_time = _process_start_time(metadata.pid)
        if actual_start_time != metadata.start_time:
            return False
    return True


def _command_matches(
    expected: tuple[str, ...],
    actual: tuple[str, ...],
) -> bool:
    if expected == actual:
        return True
    expected_tail = expected[1:] if len(expected) > 1 else expected
    return bool(expected_tail) and _contains_contiguous_sequence(actual, expected_tail)


def _contains_contiguous_sequence(
    haystack: tuple[str, ...],
    needle: tuple[str, ...],
) -> bool:
    if len(needle) > len(haystack):
        return False
    return any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _process_cmdline(pid: int) -> tuple[str, ...]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ()
    return tuple(
        part.decode("utf-8", errors="replace")
        for part in raw.split(b"\0")
        if part
    )


def _process_cwd(pid: int) -> Path | None:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return None


def _process_start_time(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        fields_after_name = stat.rsplit(") ", 1)[1].split()
        return int(fields_after_name[19])
    except (IndexError, ValueError):
        return None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return ()
        items.append(item)
    return tuple(items)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return


def _signal_pid(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return


def _wait_for_exit(pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _pid_running(pid):
            return
        time.sleep(0.1)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(main())
