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
- Next: retune the Phase 3 gains, rerun the nominal tracking gate, then proceed to Phase 5.
- Blockers: nominal maximum tracking error at t >= 0.5 s is 0.0489868 m, above the 0.03 m gate. The cutoff and torque threshold remain unchanged pending gain retuning. The 10.0 divergence ratio is provisional and uncalibrated.

## Phase 5

- Implemented: cached manufacturer search and per-URL extraction, quote-and-value verification, verified-only URDF comparison, advisory error mapping, usage records, and the spec demo.
- Next: live `spec_demo.py` run after the cache is populated and credentials are configured.
- Blockers: live Tavily and Nebius calls have not been made; no live specification evidence has been fetched.
