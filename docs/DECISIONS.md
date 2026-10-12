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
- LLM callers use `chat_json` with `fast`, `deep`, or `ultra` roles. Model resolution accepts a configured ID only when it is listed and is an NVIDIA Nemotron model, then uses the role-specific Nemotron preference order. Missing NVIDIA Nemotron models raise `LLMUnavailable`; no other model family is selected.
- Content candidates are validated from first to last; `reasoning_content` candidates are validated from last to first. Schema or JSON failures use the requested repair retries. Transport timeouts, connection errors, 429, and 5xx responses get one retry after 0.25 seconds. SDK retries are disabled and requests use a 30-second timeout.
- A response ending with `length` doubles `max_tokens` for the next output attempt, capped at 8192 tokens. Each HTTP request attempt, including model listing, is written to `data/llm_calls.jsonl` without prompt contents or credentials. Model listing is not charged against the token budget.

## Phase 3: simulation core

1. `PandaSim` owns one DIRECT client, uses `ScenarioParams.seed` for its local NumPy generator, supports context management, and closes idempotently. The frozen schemas remain unchanged.
2. Sensor noise affects measured arm position only. The controller uses `q_des = q_ref + 0.25 * (q_ref - q_meas)`; the gain is fixed for this phase.
3. IK and reference FK are planned once per episode on a separate DIRECT client. IK null-space inputs contain the seven arm joints and two open fingers; the arm uses the first seven results. The kinematics call needs nine values, and `getLinkState` after `resetJointState` needs `computeForwardKinematics=1`.
4. The pose sequence is approach `(0.4, -0.15, 0.28)`, pick `(0.4, -0.15, 0.2)`, lift to the approach height, and place at `(0.4, 0.15, 0.2)` metres, using a fixed tool-down orientation. Segments use min-jerk profiles with stop-and-go boundaries. `T0=2.5s`, so the slowest `speed_scale=0.5` reaches the place reference by 5 seconds and holds it for the remaining episode time.
5. The push acts in world +Y at the hand COM for `[push_time_s, push_time_s + 0.1s)`. Apply it at each 240 Hz substep because PyBullet clears external forces after every simulation step.
6. Scale mass and diagonal inertia together on links 0–8 before adding payload at the hand COM. Model payload inertia as a sphere with radius `0.05m`, added equally to each diagonal.
7. `jointDamping` is applied to arm joints 0–6. A fixed-seed comparison at 0 and 5 produced maximum differences of 0.000292 rad in position, 0.00145 rad/s in velocity, 8.29 N·m in applied torque, and 0.000114 m in end-effector position under POSITION_CONTROL.
8. Telemetry arrays are preallocated float64. Replay link point 0 is the base load position and points 1–7 are links 0–6; joint ratio index `j` maps to point `j+1`. The Jacobian metrics use the 3×7 arm slice from a nine-DOF calculation. Contact count is the number of distinct non-base robot links in contact.
9. Downsampling accepts only positive FPS values dividing 60, selects every `60/fps` tick without interpolation, and validates each frame as a `ReplayFrame` with empty flags. The Phase 4 runner adds replay metadata.
10. `deterministicOverlappingPairs=1` is enabled with the locked PyBullet 3.2.7. `render.py` remains deferred until a rendering dependency is approved. All simulation ranges and payload assumptions are not hardware tolerances.
