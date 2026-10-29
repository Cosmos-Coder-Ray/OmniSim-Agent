.PHONY: setup check test-slow sim-smoke api web web-setup web-check web-e2e web-screenshot e2e e2e-repeat

setup:
	uv sync
	uv run pre-commit install

check:
	uv run ruff check
	uv run ruff format --check
	uv run pyright
	uv run pytest -q -m "not slow"

test-slow:
	uv run pytest -q -m slow

e2e:
	uv run pytest -q tests/e2e -m "not slow"

e2e-repeat:
	uv run python scripts/e2e_repeat.py

sim-smoke:
	uv run python scripts/sim_smoke.py

api:
	uv run --env-file .env uvicorn omnisim.api.main:app --host 0.0.0.0 --port 8000

web:
	pnpm --dir web dev -- --host 0.0.0.0 --port 5173

web-setup:
	pnpm --dir web install --frozen-lockfile --strict-peer-dependencies

web-check:
	pnpm --dir web check
	pnpm --dir web exec playwright test --project=acceptance

web-e2e:
	pnpm --dir web build
	pnpm --dir web exec playwright test --project=acceptance

web-screenshot:
	pnpm --dir web build
	pnpm --dir web exec playwright test --project=screenshot
