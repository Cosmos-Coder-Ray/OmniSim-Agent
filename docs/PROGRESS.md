# Progress

## Phase 1

- Done: scaffold, frozen core contracts, settings, tests, CI, and deterministic mock fixtures.
- Next: Phase 2 LLM layer (completed below).
- Blockers: none.

## Phase 2

- Done: LLM JSON parsing, model selection, Nebius request layer, token budget, call log, tests, and smoke script.
- Next: Phase 3 per the Master Plan.
- Blockers: live Nebius smoke check has not been run.

## Phase 3

- Done: deterministic headless Panda environment, limit source, pose-sequence controller, bounded scenario perturbations, telemetry/replay downsampling, integration tests, and five-case smoke script.
- Next: Phase 4 detector and episode runner work.
- Blockers: none. Smoke torque ratios were 0.341 nominal, 0.798 for 3 kg plus speed 2, and 0.682 at torque scale 0.5. The stress values remain below the approximate 0.95 target and are recorded for Phase 4 calibration without policy tuning.

## Phase 4

- Implemented: deterministic failure detectors, ordered serial and spawn-based episode batches, cooperative and parent timeouts, optional shield hook, validated live replay output, and smoke reporting for risk and failure kinds.
- Smoke, seed 17: nominal risk 0.98 with no events and max torque ratio 0.3410; 3 kg plus speed 2 risk 1.75 with tracking and velocity events and max torque ratio 0.7975; torque scale 0.5 risk 0.99 with no events and max torque ratio 0.6821. The 12-tick latency case reported tracking error; the noise-plus-push case reported tracking error.
- Next: retune the Phase 3 gains and rerun the nominal tracking gate before broader campaign comparisons.
- Blockers: nominal maximum tracking error at t >= 0.5 s is 0.0489868 m, above the 0.03 m gate. The cutoff and torque threshold remain unchanged pending gain retuning. The 10.0 divergence ratio is provisional and uncalibrated.

## Phase 5

- Implemented: cached manufacturer search and per-URL extraction, quote-and-value verification, verified-only URDF comparison, advisory error mapping, usage records, and the spec demo.
- Next: live `spec_demo.py` run after the cache is populated and credentials are configured.
- Blockers: live Tavily and Nebius calls have not been made; no live specification evidence has been fetched.

## Phase 6: adversary and benchmark harness

- Implemented: schema-derived scenario bounds, deterministic random and CMA-ES baselines, the Nemotron adversary, atomic benchmark LLM caching, raw JSONL and result output, aggregation, and the seed-based dev/reporting path policy.
- Verified: `make check` passes; the offline random plus CMA-ES integration case passes with one seed, budget 8, and two workers.
- Next: review the reporting artifacts and run the Nebius benchmark when credentials and token budget are ready.
- Blockers: no live Nebius run was made; live Nemotron model resolution, cache replay against a real response, and the reporting benchmark remain unverified.

## Phase 7: campaign orchestration

- Implemented: deterministic failure clustering, bounded scenario shrinking, typed event streaming and persistence, campaign status/stop-reason reporting, evidence-grounded diagnosis, the campaign CLI, and recorded demo-run export.
- Verified: `make check` passes after each of the five commit groups. The separate budget-8 random campaign integration test passed with the real headless PyBullet simulator and injected spec/LLM functions. Demo-run tests validate source rewriting and overwrite protection.
- Next: run `python scripts/run_campaign.py --budget 40` when the user is ready to make the live campaign calls.
- Blockers: no live Nebius or Tavily call was made; the campaign has not been run against live LLM or knowledge services.

## Phase 8: mitigation, verification, and exports

- Implemented: deterministic shield decisions and runner-only rejection results; inverse-dynamics payload caps; evidence-grounded mitigation with a deterministic fallback; fixed-size tuning and confirmation verification; deterministic JSON, ROS 2 YAML, generated regression, and self-contained HTML exports; `--verify` and optional `--domain` CLI handling; demo export copying.
- Verified: shield, verification, mitigation, export, CLI skip, and demo-copy unit tests pass. The real-sim ON/OFF verification case with eight scenarios passes using PyBullet DIRECT. The generated regression file passes when run with pytest. `make check` passes with 274 tests and four slow tests deselected.
- Next: run `python scripts/run_campaign.py --budget 40 --verify` when ready for live LLM and campaign calls.
- Blockers: no live campaign or LLM call was made. The 5% unsafe, 40% blocked, and 90% completion thresholds remain provisional.

## Phase 9: FastAPI campaign backend

- Implemented: spawned campaign jobs with cooperative cancellation, per-profile admission estimates and timeouts, atomic stage state, resumable SSE, health and artifact routes, recorded-run isolation, CORS and rate limits, OpenAPI export, and the API secret-scan smoke script.
- Implemented: campaign state episode lists for terminal and recorded runs, with complete-line validation for live `episodes.jsonl` files and an all-or-nothing generic error on malformed records.
- Verified: `make check` passed after Groups 1–4 with 276, 286, 290, and 298 tests passed respectively; 5 slow tests were deselected each time. The nested-process shutdown test passed separately. The local API smoke returned the expected health, OpenAPI, benchmark, and not-found responses without reporting a secret leak. The OpenAPI drift check is current.
- Verified for the episode contract: `make check` passed with 310 tests and 5 slow tests deselected; `uv run python scripts/export_openapi.py --check` reported the export current. Both commands emitted the existing optional `cma` matplotlib warning.
- Next: replace the UNMEASURED token admission estimates after live runs with about 1.3 times measured per-run usage.
- Blockers: no live Nebius or Tavily call was made, and quick-mode time was not measured. The token estimates are conservative admission checks, not guarantees.

## Phase 10A: replay web shell

- Implemented: pinned React/Three toolchain, generated API types, mock/API client, responsive shell, URL state, replay validation, deterministic articulated mock replays, demand-rendered Panda scene, playback, seeking, event timeline, joint charts, Playwright acceptance and screenshot projects, and web CI job.
- Verified: final `make check` passed with 314 tests and 5 slow tests deselected; final `make web-check` passed with 30 Vitest tests and 4 Playwright tests. `make web-screenshot` passed and generated the paused 1440×900 `docs/img/replay.png`. Frozen strict pnpm install completed with no peer warnings. Playwright ran on Windows against installed Google Chrome using software SwiftShader.
- Next: merge the completed frontend branch.
- Blockers: CI runs typecheck, lint, unit tests, generated-type drift, and build; Playwright remains local because software WebGL can vary. The Three.js vendor chunk is 733.61 kB before gzip and triggers Vite's advisory. FPS was not measured.

## Phase 10B: hardening

- Implemented: shared provider retries and circuit breakers; caller fallback policies; worker/job recovery; structured JSON audit logs; health degradation and campaign warning fields; campaign evidence and verification routes; scenario-count injection and showcase reruns; OpenAPI/TypeScript updates; HTTP E2E and repeat commands.
- Verified: Group 1 `make check` passed (330 tests, 5 slow deselected); Group 2 focused tests passed (114) and `make check` passed (355 tests, 5 slow deselected); Group 4 focused tests passed (41); final `make check` passed (358 tests, 5 slow deselected); `make e2e` passed; `make e2e-repeat` passed all five runs; `make test-slow` passed all five tests; branch coverage passed (363 tests, 81% total branch coverage); OpenAPI export is current.
- Next: none for Phase 10B hardening.
- Blockers: no live Nebius or Tavily calls were made.
