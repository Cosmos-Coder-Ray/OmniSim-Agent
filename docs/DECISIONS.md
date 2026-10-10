# Decisions

- The six core data contracts are frozen v1. `FailureEvent.detail` uses `Field(default_factory=dict)` so instances do not share mutable state; this is the only deviation from the SPEC definitions.
- `Replay`, `ReplayFrame`, `CampaignResult`, and `BenchmarkResult` are additive contracts. The top-level replay, campaign, and benchmark models forbid unknown fields and label data with `source`.
- Replay frame coordinates use eight ordered xyz points. Torque and velocity ratio index `j` maps to link point `j + 1`; link point 0 is uncolored.
- Campaign strategy values are `nemotron`, `random`, and `cmaes`. Campaign event data uses `{type, data}` with the ten Phase 7 event names; payload details remain provisional until Phase 7.
- Benchmark means and standard deviations use population statistics. Missing per-seed first failures stay `null` in raw data and contribute `budget + 1` to the aggregate; the mean is marked as a lower bound when some seeds have no failure. Curves must have `budget` entries, per-seed results must have `seed_count` entries, and strategy names must be unique.
- Settings read process environment variables only. Keys, base URL, and model IDs default to empty strings; token cap defaults to 400000 and demo mode to false. Local API startup loads `.env` through `uv run --env-file .env`; if `.env` is absent, uv exits before starting the app. CI and `make check` do not use an env file.
- Hatchling is the only build-system dependency; it builds the `src/omnisim` package for installation.
- Mock fixtures use `source="mock"`, envelope version `mock-0`, `https://mock.invalid/` evidence URLs, and quotes prefixed with `[mock]`.
- The SPEC repo layout is used because `docs/ARCHITECTURE.md` is absent.
- LLM usage is stored in SQLite at `data/llm_usage.sqlite3`, keyed by UTC date and role. A SQLite write transaction holds request admission through usage recording, so a response can take usage over the cap by at most that response. Missing usage is recorded as zero and logged.
- LLM callers use `chat_json` with `fast`, `deep`, or `ultra` roles. Model resolution uses a configured ID when available, then the role-specific Nemotron preference order; it never selects another provider.
- Content candidates are validated from first to last; `reasoning_content` candidates are validated from last to first. Schema or JSON failures use the requested repair retries. Transport timeouts, connection errors, 429, and 5xx responses get one retry after 0.25 seconds. SDK retries are disabled and requests use a 30-second timeout.
- A response ending with `length` doubles `max_tokens` for the next output attempt, capped at 8192 tokens. Each HTTP request attempt, including model listing, is written to `data/llm_calls.jsonl` without prompt contents or credentials. Model listing is not charged against the token budget.
