FROM node:24-slim AS web-build
WORKDIR /app/web
RUN corepack enable && corepack prepare pnpm@11.25.0 --activate
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml web/.npmrc ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
ENV VITE_DATA_SOURCE=api VITE_API_BASE=/api
RUN pnpm build

FROM python:3.11-slim AS py-build
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
COPY pyproject.toml uv.lock .python-version ./
COPY src/ ./src/
RUN uv sync --locked --no-dev --no-editable

FROM python:3.11-slim AS runtime
WORKDIR /app
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin omnisim
COPY --from=py-build /app/.venv /app/.venv
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY --from=web-build /app/web/dist ./web/dist
RUN mkdir -p runs exports data && chown -R omnisim:omnisim runs exports data
ENV PATH="/app/.venv/bin:$PATH"
USER omnisim
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1
CMD ["uvicorn", "omnisim.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

FROM caddy:2-alpine AS caddy-runtime
COPY --from=web-build /app/web/dist /srv
COPY Caddyfile /etc/caddy/Caddyfile
