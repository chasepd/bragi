#!/usr/bin/env bash
# Codex PostToolUse hook entrypoint for file edits through apply_patch.
set -euo pipefail

hook_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:/snap/bin:/opt/homebrew/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"

exec python3 "${hook_dir}/run_tests_for_edited_file.py" "$@"
