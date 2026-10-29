# Devpost submission

## Track

Physical AI

## Tagline

Find policy failure boundaries in simulation, ground the diagnosis in manufacturer evidence, and check a deterministic safety envelope.

## Inspiration

Robot policies can pass a nominal simulation and still fail when the real system differs in payload, latency, sensing, or actuator behavior. We wanted a repeatable way to search those conditions, understand the resulting failures, and test whether a safety envelope changes the outcome before moving toward hardware.

## What it does

OmniSim-Agent runs a scripted Panda policy in headless PyBullet. A Nemotron adversary proposes bounded scenarios from trial history. The system scores detector events, clusters and shrinks failures, searches manufacturer documentation through Tavily, and prepares a candidate envelope. Deterministic code applies the shield and the verifier checks held-out scenarios. A React-Three-Fiber dashboard presents campaign events, evidence, and trajectory replay.

The simulator smoke values in `docs/PROGRESS.md`, Phase 4 are not hardware measurements. No reporting benchmark or saved hero verification run is present in this checkout. Benchmark values in `web/public/mock/benchmark.json` are mock fixture data and are not presented as measured results.

## How we built it

The simulation, failure detectors, scenario shrinking, clustering, shield, and verification are Python modules under `src/omnisim/`. FastAPI exposes campaign state, artifacts, and server-sent event streams. The web dashboard uses React, TypeScript, and React-Three-Fiber. NVIDIA Nemotron is called through Nebius Token Factory for adversarial proposals, evidence extraction, diagnosis, and mitigation drafts. Tavily searches restricted manufacturer domains, extracts pages, and caches results locally. Pydantic validation and deterministic code own bounds, evidence checks, shield behavior, and held-out evaluation.

## Challenges we ran into

Nemotron responses can put text in `reasoning_content`, wrap JSON in `<think>` tags, or stop at the output limit. We parse and validate response candidates, retry repairs, and keep token admission checks on every completion attempt (`src/omnisim/llm/nebius.py`, `src/omnisim/llm/json_utils.py`, `src/omnisim/llm/budget.py`).

Manufacturer datasheets use formatting that can make exact string comparisons brittle. The verifier normalizes whitespace and Unicode punctuation, checks the quote against extracted text, and confirms that the claimed value occurs in the quote (`src/omnisim/knowledge/verifier.py`). That presence check does not prove semantic support, so evidence still needs review.

## Accomplishments that we're proud of

The Phase 4 seed-17 simulator smoke recorded nominal risk 0.98 with no events and maximum torque ratio 0.3410. Under the 3 kg payload plus speed-scale-2 condition it recorded risk 1.75, tracking and velocity events, and maximum torque ratio 0.7975 (`docs/PROGRESS.md`, Phase 4). These are simulation results only.

The Phase 8 ON/OFF verification test passed with eight scenarios under PyBullet DIRECT (`docs/PROGRESS.md`, Phase 8). The progress record does not include held-out failure, blocked, completion, or cycle-time rates, and we do not claim those as live benchmark results.

## What we learned

Risk scoring, failure shrinking, and evidence verification need explicit boundaries. Model output can suggest scenarios and constraints, but it cannot establish measured results or enforce safety limits. A quote/value presence check is useful for filtering evidence, but it is not a substitute for semantic review or hardware validation.

## What's next for OmniSim-Agent

Run and save a live benchmark with attributed token usage, generate and verify the `hero` recorded campaign, fit a Reality Anchor from CSV measurements, compare PyBullet with MuJoCo, evaluate pretrained SB3 policies, and add a Robot-CI GitHub Action. These are planned work; the current checkout does not include a saved hero campaign or live reporting benchmark (`docs/PROGRESS.md`, Phases 6–10B; `data/demo_runs/`).

## Built With

Python, PyBullet, FastAPI, React, TypeScript, React-Three-Fiber, Nebius Token Factory, NVIDIA Nemotron, Tavily, Pydantic, SQLite, Docker, Caddy

## Demo links

Public demo: `[PUBLIC_DEMO_URL]`  
YouTube video: `[YOUTUBE_VIDEO_URL]`
