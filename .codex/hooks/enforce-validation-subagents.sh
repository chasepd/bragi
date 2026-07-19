#!/usr/bin/env bash
# Codex PreToolUse hook entrypoint that routes validation commands to the script.
# Best-effort guardrail for directing the model to the correct flow most of the
# time; not a security boundary or an impossible-to-bypass mechanism.
set -euo pipefail

hook_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "${hook_dir}/enforce-validation-subagents.py"
