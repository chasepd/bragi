# syntax=docker/dockerfile:1

FROM node:22-bookworm-slim AS frontend

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN npm ci --prefix frontend

COPY frontend ./frontend
RUN npm run build --prefix frontend


FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ENV BRAGI_WEB_DATA_DIR=/data/bragi \
    BRAGI_WEB_HOST=0.0.0.0 \
    BRAGI_WEB_PORT=8787 \
    HOME=/data/bragi \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md MANIFEST.in bragi_build_backend.py ./
COPY bragi ./bragi
COPY bragi_common ./bragi_common
COPY bragi_web ./bragi_web
COPY --from=frontend /app/bragi_web/static ./bragi_web/static

RUN uv sync --locked --no-dev --no-editable \
    && groupadd --system bragi \
    && useradd --system --gid bragi --home-dir /data/bragi --shell /usr/sbin/nologin bragi \
    && mkdir -p /data/bragi \
    && chown -R bragi:bragi /app /data/bragi

USER bragi

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import json, urllib.request; data = json.load(urllib.request.urlopen('http://127.0.0.1:8787/api/health', timeout=3)); raise SystemExit(0 if data.get('status') == 'ok' else 1)"

CMD ["bragi-web", "--host", "0.0.0.0", "--port", "8787"]
