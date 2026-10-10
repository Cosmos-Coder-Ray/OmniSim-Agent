.PHONY: setup check sim-smoke api web

setup:
	uv sync
	uv run pre-commit install

check:
	uv run ruff check
	uv run ruff format --check
	uv run pyright
	uv run pytest -q

sim-smoke:
	uv run python scripts/sim_smoke.py

api:
	uv run --env-file .env uvicorn omnisim.api.main:app --host 0.0.0.0 --port 8000

web:
	pnpm --dir web dev -- --host 0.0.0.0 --port 5173
