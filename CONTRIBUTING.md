# Contributing to Bragi

Thanks for helping improve Bragi.

## Before You Start

Use GitHub Issues for bugs and feature proposals. Keep reports generic: never
attach private roleplay transcripts, save exports, API keys, local databases, or
generated media containing personal information. Security vulnerabilities
belong in the private reporting channel described in `SECURITY.md`.

## Development

Bragi requires Python 3.12, `uv`, Node.js 22, and npm.

```bash
uv sync --locked --extra dev
npm ci --prefix frontend
python3 .codex/tools/validate.py --full
```

Application changes should be developed test-first. Unit tests use fake
providers and must not require real provider credentials or network access.
Follow the mirrored test naming and validation workflow in `AGENTS.md`.

Open pull requests into `main`, explain the user-visible impact, and keep each
pull request focused. By submitting a contribution, you agree that it may be
distributed under the repository's MIT License.
