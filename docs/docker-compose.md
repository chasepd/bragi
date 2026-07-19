# Docker Compose Deployment

Bragi's Compose deployment runs one production service:

- The Docker build compiles the React/Vite frontend into `bragi_web/static`.
- FastAPI serves both `/api/*` and the built SPA on port `8787`.
- Bragi data persists in the Docker named volume `bragi-data`.

## Start

```bash
docker compose up --build -d
```

The equivalent Make target is:

```bash
make compose-up
```

Open `http://127.0.0.1:8787` on the host, or use the host machine's LAN IP from
another trusted device.

On a fresh data volume, Bragi requires first-admin bootstrap before normal app
use. Open the app, create the admin account, then use that account to create
additional `admin`, `user`, or `child` accounts from Settings. Legacy
single-user data without an owner is claimed by the first admin during
bootstrap. If bootstrap is opened from another LAN device, set
`BRAGI_WEB_BOOTSTRAP_TOKEN` in your shell or local `.env` file before starting
Compose, then enter that token on the setup screen. The provided Compose file
passes this value through to the container. Newly created local passwords must
be at least 12 characters.

The full role and route authorization policy lives in `docs/auth-policy.md`.

## Operate

```bash
docker compose logs -f bragi
docker compose ps
docker compose down
```

The matching Make targets are:

```bash
make compose-logs
make compose-ps
make compose-down
```

The image includes a healthcheck that calls `GET /api/health`. A healthy
response is:

```json
{"status":"ok"}
```

## Data

The `bragi-data` volume is mounted at `/data/bragi` in the container. It holds:

- `bragi.sqlite3`
- `media/`
- `state/api_keys.json`
- `cache/`
- `tmp/`

Inspect the host-side volume location with:

```bash
docker volume inspect bragi-data
```

Create a portable backup archive with:

```bash
docker run --rm -v bragi-data:/data -v "$PWD:/backup" busybox \
  sh -c "tar czf /backup/bragi-data.tgz -C /data ."
```

## Auth And LAN Security

Authentication is required by default. Sessions are stored in an HttpOnly
`bragi_session` cookie with `SameSite=Lax`; cookies are marked secure when the
request is HTTPS, or when `BRAGI_WEB_SECURE_COOKIES=1` is set.

This Compose setup still assumes trusted LAN/private deployment and direct port
publishing on `8787`. Do not expose it to the public internet without TLS,
hardening, backups, and a reverse proxy configuration you trust.

For a custom LAN hostname or reverse proxy added outside this Compose file, set
the existing `BRAGI_WEB_ALLOWED_HOSTS` and `BRAGI_WEB_ALLOWED_ORIGINS`
environment variables on the `bragi` service. The Compose file passes these
values through from the shell or a local `.env` file. For example, Caddy serving
Bragi at `https://bragi.home` should use:

```dotenv
BRAGI_WEB_ALLOWED_HOSTS=bragi.home
BRAGI_WEB_ALLOWED_ORIGINS=https://bragi.home
BRAGI_WEB_SECURE_COOKIES=1
```

Reverse proxies should preserve the intended external host/origin, forward
HTTPS traffic consistently, and set `BRAGI_WEB_SECURE_COOKIES=1` when TLS
terminates before the Bragi container.
