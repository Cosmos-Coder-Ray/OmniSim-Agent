# Deployment

## One Linux VM

Install Docker Engine and the Docker Compose plugin, clone the repository, then run:

```sh
cp .env.example .env
nano .env
mkdir -p data runs exports
sudo chown -R 10001:10001 data runs exports
docker compose up -d --build
docker compose ps
docker compose logs -f
```

Set `NEBIUS_API_KEY` and, when using documentation search, `TAVILY_API_KEY` in `.env` before starting. Open the VM's HTTP address. For HTTPS, configure Caddy with the VM's public DNS name and allow inbound ports 80 and 443.

## Separate frontend and API

Build the static frontend with these build-time variables:

```sh
cd web
VITE_DATA_SOURCE=api VITE_API_BASE=https://<api-host>/api pnpm build
```

Publish `web/dist` to Vercel or Cloudflare Pages. Build and run the backend image on a Nebius AI Cloud VM. Set `OMNISIM_API_CORS_ORIGINS` to the frontend origin and configure the API host's TLS reverse proxy to pass `/api/*` to port 8000. Do not expose the container port directly to the public internet.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `NEBIUS_API_KEY` | Token Factory credential. |
| `NEBIUS_BASE_URL` | Token Factory OpenAI-compatible endpoint. |
| `NEBIUS_FAST_MODEL`, `NEBIUS_DEEP_MODEL`, `NEBIUS_ULTRA_MODEL` | Model IDs for the configured Nemotron roles. |
| `TAVILY_API_KEY` | Tavily documentation search credential. |
| `OMNISIM_DEMO_MODE` | Serve the configured recording for new campaign requests when true. |
| `OMNISIM_API_DEMO_RUN` | Recorded campaign name, normally `hero`. |
| `OMNISIM_API_MAX_BUDGET` | Maximum campaign budget accepted by the API. |
| `OMNISIM_API_POST_RATE_LIMIT` | Per-client campaign submission limit. |
| `OMNISIM_API_OTHER_RATE_LIMIT` | Per-client limit for other API routes. |
| `OMNISIM_API_MAX_BODY_BYTES` | Maximum request body size, default 65536. |
| `OMNISIM_TRUST_PROXY_HEADERS` | Trust forwarded client IP headers from a configured reverse proxy. |
| `OMNISIM_DAILY_TOKEN_CAP` | Daily LLM token budget. |
| `OMNISIM_API_CORS_ORIGINS` | Comma-separated browser origins allowed to call the API. |
| Other `OMNISIM_API_*` settings | Queue limits, timeouts, model resolution timeout, CORS origins, and admission estimates. See `.env.example`. |

## Production modes

Guarded Live Mode uses `OMNISIM_DEMO_MODE=false`, `OMNISIM_API_MAX_BUDGET=10`, and `OMNISIM_API_POST_RATE_LIMIT=3/hour`. The limit applies per client IP. Rate-limited requests return HTTP 429 and include a replay link to `/api/campaigns/hero` when that recording is installed. Live campaigns require configured Nebius credentials.

Recorded Fallback Mode uses `OMNISIM_DEMO_MODE=true`. Campaign submissions return the configured saved campaign and do not enter the live job queue. The API identifies these results as recorded.

## Rotate credentials and preserve recordings

Replace provider keys in `.env`, then recreate the API container:

```sh
docker compose up -d --force-recreate api
```

Never commit `.env`. To save a completed run as the `hero` recording, run this in the repository environment before deploying or mount the saved recording under `data/demo_runs/`:

```sh
uv run python scripts/save_demo_run.py <run_id> --name hero --force
```

Back up `data/demo_runs/`, `data/urdf_limits.json`, and `data/tavily_cache/` as required. The compose file keeps these under the host `data/` directory. Restore them there before starting the API.
