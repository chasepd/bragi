from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_restart_static_script_runs_static_frontend_restart(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "restart-static"
    bin_dir = tmp_path / "bin"
    output = tmp_path / "uv-call.txt"
    uv = bin_dir / "uv"

    bin_dir.mkdir()
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'cwd=%s\\nargs=%s\\n' \"$PWD\" \"$*\" > \"$UV_CALL_OUTPUT\"\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["UV_CALL_OUTPUT"] = str(output)

    subprocess.run([script], check=True, cwd=tmp_path, env=env)

    assert os.access(script, os.X_OK)
    assert output.read_text(encoding="utf-8") == (
        f"cwd={repo}\n"
        "args=run bragi restart --frontend-mode static --build-frontend\n"
    )


def test_makefile_exposes_docker_compose_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "COMPOSE_PROJECT_NAME := bragi-prod\n" in makefile
    assert (
        "COMPOSE := docker compose --project-name $(COMPOSE_PROJECT_NAME)\n"
        in makefile
    )
    assert "COMPOSE_SERVICE := bragi\n" in makefile
    assert "compose-build:\n\t$(COMPOSE) build $(COMPOSE_SERVICE)\n" in makefile
    assert (
        "compose-up:\n\t$(COMPOSE) up --build -d $(COMPOSE_SERVICE)\n"
        in makefile
    )
    assert "compose-down:\n\t$(COMPOSE) down\n" in makefile
    assert "compose-logs:\n\t$(COMPOSE) logs -f $(COMPOSE_SERVICE)\n" in makefile
    assert "compose-ps:\n\t$(COMPOSE) ps\n" in makefile
    assert "compose-restart: compose-down compose-up\n" in makefile


def test_make_default_target_shows_help() -> None:
    repo = Path(__file__).resolve().parents[2]

    default = subprocess.run(
        ["make"],
        check=True,
        cwd=repo,
        capture_output=True,
        text=True,
    )
    explicit = subprocess.run(
        ["make", "help"],
        check=True,
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert default.stderr == ""
    assert default.stdout == explicit.stdout
    assert "Available targets:\n" in default.stdout
    assert "  help             Show available targets.\n" in default.stdout
    assert (
        "  compose-up       Build and start the production Compose service.\n"
        in default.stdout
    )


def test_compose_passes_remote_bootstrap_token() -> None:
    compose = Path("compose.yaml").read_text(encoding="utf-8")

    assert "BRAGI_WEB_BOOTSTRAP_TOKEN: ${BRAGI_WEB_BOOTSTRAP_TOKEN:-}\n" in compose
