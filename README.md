# Bragi

Bragi is a trusted-LAN web roleplaying workbench. It combines an AI-powered
chronicle with deterministic local state, memories, summaries, saves, and
generated scene images.

![Bragi roleplaying chronicle interface](docs/images/bragi-screenshot.png)

The Python backend serves the Bragi API with FastAPI/uvicorn, and the frontend
is a React/Vite app. The legacy GTK frontend has been retired; `bragi` now
starts the web experience by default, while `bragi-web` remains as a
compatibility alias.

## Development

```bash
uv sync --locked --extra dev
npm ci --prefix frontend
python3 .codex/tools/validate.py --full
```

If locked Python dependency setup fails after an intentional dependency change,
update `uv.lock` with `uv lock` or the relevant `uv add`/`uv remove` command,
then rerun validation.

When intentionally updating frontend dependencies, use the relevant npm update
or install command in `frontend/`, commit the resulting `package-lock.json`, and
return to `npm ci --prefix frontend` for normal setup.

Run the development web app:

```bash
uv run bragi start
```

The backend binds to `0.0.0.0:8787` and the Vite frontend binds to
`0.0.0.0:5173` by default. Authentication is required by default. On a new
install, open the app and create the first admin account through the bootstrap
screen; existing single-user saves are claimed by the first admin during
bootstrap. When first-admin bootstrap is opened from another LAN device, enter
the remote setup token printed by `bragi start`, or set
`BRAGI_WEB_BOOTSTRAP_TOKEN` before launch. Newly created local passwords must be
at least 12 characters.

Bragi has three local roles: `admin` can manage users, global settings, provider
keys, and all saves; `user` can use owned or assigned saves; `child` can read
and chat in owned or assigned saves and generate safeguarded media through a
provider with enforced safe mode. Destructive,
import/export, media-management, and provider controls remain blocked for child
accounts. The route-level authorization contract lives in `docs/auth-policy.md`.

Sessions use an HttpOnly `bragi_session` cookie with `SameSite=Lax`. The cookie
is marked secure automatically on HTTPS; set `BRAGI_WEB_SECURE_COOKIES=1` when
Bragi is behind a TLS-terminating reverse proxy.

The default trusted-LAN setup uses plaintext HTTP and does not encrypt the
session cookie or application traffic in transit. Anyone able to observe or
modify traffic on that LAN may be able to capture a session cookie or private
roleplay content. Use the default only on a network and devices you trust. For
stronger transport security, put Bragi behind an HTTPS reverse proxy, configure
the allowed host and origin, and enable secure cookies as described in
`docs/docker-compose.md`.

Bragi rejects unsafe API writes from untrusted browser origins. Localhost,
loopback, private/link-local IP hosts, and configured bind hosts are accepted as
request hosts, but browser write origins are only trusted by default when they
come from the same host on the configured backend/frontend ports. For a custom
LAN DNS name, reverse proxy, or different frontend origin, set
`BRAGI_WEB_ALLOWED_HOSTS` to comma-separated hostnames and
`BRAGI_WEB_ALLOWED_ORIGINS` to comma-separated full origins.

Bragi still assumes trusted LAN/private deployment. Do not expose it to the
public internet without TLS, backups, hardening, and a reverse proxy setup you
trust.

Provider API keys are stored in the private Bragi web state directory by
default, using `api_keys.json` with owner-only file permissions. This file is
part of the local or container data volume and should be covered by trusted
backups. For non-container installs where system keyring storage is preferred,
set `BRAGI_WEB_USE_KEYRING=1` before starting the server.

OpenRouter requests identify as `Bragi` with `X-OpenRouter-Title`. Requests use
`https://github.com/chasepd/bragi` as the default `HTTP-Referer` for app
attribution. Set `BRAGI_OPENROUTER_APP_URL` to override the referer with a
specific deployed Bragi app URL.

## Docker Compose

Bragi can also run as a single production Compose service. The image builds the
React/Vite frontend into the Python package, serves the SPA from FastAPI, and
stores runtime data in the `bragi-data` Docker volume.

```bash
docker compose up --build -d
```

Or use the Make target:

```bash
make compose-up
```

Open `http://127.0.0.1:8787` from the host, or use the host machine's LAN IP
from another trusted device. Create the first admin account on first launch; LAN
bootstrap requires `BRAGI_WEB_BOOTSTRAP_TOKEN` when the setup screen is reached
remotely.
For reverse proxies or custom LAN hostnames, configure allowed hosts/origins and
secure cookies as described in `docs/docker-compose.md`.

More deployment notes are in `docs/docker-compose.md`.
The full auth policy is in `docs/auth-policy.md`.
Troubleshooting guidance is in `docs/troubleshooting.md`.
Privacy review guidance for checked-in fixtures and documentation is in
`docs/privacy-review.md`.

Agent configuration sync and drift checks are documented in
`docs/agentsync.md`.

## Open Source

Bragi is available under the MIT License. See `LICENSE`.

Contributions are welcome under `CONTRIBUTING.md`. Security reports, support
requests, and community participation are covered by `SECURITY.md`,
`SUPPORT.md`, and `CODE_OF_CONDUCT.md`.
