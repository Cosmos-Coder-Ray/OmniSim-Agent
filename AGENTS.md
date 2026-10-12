# AGENTS.md: OmniSim-Agent

Rules for every AI coding tool working in this repo (Codex, Claude, Antigravity, OpenCode, Copilot). Read this file fully before touching code. If a rule here conflicts with a user request in chat, say so and ask before breaking it.

---

## 1. Project snapshot

OmniSim-Agent is an autonomous sim-to-real policy auditor for robots. It loads a robot (Franka Panda in headless PyBullet) and a control policy, attacks the policy with adversarial physics conditions, detects failures, grounds the diagnosis in manufacturer documentation, and outputs a verified safe operating envelope.

- Hackathon: Nebius x NVIDIA Global AI Hackathon, Physical AI track. Submission closes **30 Oct 2026, 10:00 AM PT**.
- Stack: Python 3.11, uv, pydantic v2, PyBullet (`p.DIRECT`), FastAPI, SQLite, React + TypeScript + react-three-fiber (in `web/`).
- LLM: **NVIDIA Nemotron on Nebius Token Factory** (OpenAI-compatible API). Search: **Tavily**.
- Pipeline: adversary proposes scenarios -> sim runs them -> detectors score risk -> failures get shrunk and clustered -> Tavily evidence -> diagnosis -> envelope -> deterministic shield -> held-out verification -> exports.

Source of truth for design: `docs/SPEC.md`. For contracts: `src/omnisim/schemas.py`. For progress: `docs/PROGRESS.md`.

---

## 2. Non-negotiables

1. **Nemotron via Nebius Token Factory is the LLM.** Base URL, key and model IDs come from env vars through `src/omnisim/config.py`. Never hardcode them. Never swap in another provider for core logic.
2. **Never print, log, commit or paste secrets.** `.env` is git-ignored. Use `.env.example` for names only.
3. **LLM output is untrusted.** Parse, validate with Pydantic, clamp to bounds in code, retry once with a repair message, then fall back to a deterministic option. Do not depend on native tool calling.
4. **The safety shield is plain deterministic code.** The LLM never enforces limits.
5. **Every hardware number shown to a user needs evidence** (`source_url`, `quote`, `verified`). Unverified numbers do not go into the envelope.
6. **Determinism.** Same params + same seed = identical result. Use a seeded `np.random.default_rng`, never the global RNG.
7. **Contracts are frozen.** Don't change `schemas.py` without adding a short note to `docs/DECISIONS.md` and telling the user.
8. **No new dependencies without asking.** Name the package and why.
9. **Don't fake results.** No invented benchmark numbers, no mocked output presented as real, no recorded run labelled as live.
10. **Stay in scope.** Do the task you were given. No drive-by refactors, no extra features, no renames of unrelated things.

---

## 3. How to work

1. **Plan first.** For anything bigger than a small fix, reply with a short plan: files to touch, function signatures, tests, risks. Wait for approval.
2. **Read before writing.** Look at neighbouring code and copy its conventions.
3. **Work in small steps.** One concern per change. If a change grows past roughly 300 lines, stop and split it.
4. **Run things.** Don't say "this should work". Run `make check` and the acceptance command for the task, and report what actually happened.
5. **If you're stuck or unsure, ask.** A one-line question is cheaper than a wrong 400-line change.
6. **Don't touch another agent's branch or worktree.** One agent per branch.
7. **Finish by reporting** in the format in section 12.

---

## 4. Code that reads like a person wrote it (compulsory)

All code, comments, docs, tests and commit messages must read like they came from a competent engineer on this team who is working to a deadline. This is a hard requirement, not a preference. Before you finish, reread your diff and remove anything that fails this section.

### 4.1 Comments

- Comment only when the reason isn't obvious from the code. Explain **why**, never what.
- No comment that restates the next line. No `# Loop through the items`, no `# Initialize variables`, no `# Return the result`.
- No section banners (`# ===== Helpers =====`), no `# Step 1`, `# Step 2` walkthroughs, no ASCII dividers.
- No commented-out code. Delete it; git remembers.
- `TODO` only with a reason and only if the user agreed to leave it. Don't scatter them.
- Leave a short comment for genuinely non-obvious things: units, physics assumptions, library quirks (for example, PyBullet's `calculateJacobian` needs lists for all 9 movable joints on the Panda).

### 4.2 Docstrings

- Skip docstrings on private helpers and obvious functions.
- Public functions with a non-obvious contract get **one or two plain lines**. No `Args:` / `Returns:` / `Raises:` blocks unless the function is part of the external API and the types don't say it all.
- Don't write module docstrings that just list what's inside the file.

### 4.3 Naming

- Use short, domain-normal names: `q`, `dq`, `tau`, `ee`, `tau_ratio`, `params`, `ep`. Robotics code reads better this way.
- Don't write names like `calculate_the_maximum_torque_ratio_for_all_joints`. Prefer `max_tau_ratio`.
- Avoid filler words in names: `Manager`, `Handler`, `Helper`, `Utils`, `Processor`, `Wrapper`, `Service`, unless the project already uses them.
- Keep the names the project already uses. Don't introduce a second word for the same idea.

### 4.4 Structure

- Write the simplest thing that works. No abstract base classes, factories, registries, plugin systems or strategy patterns unless there are already two real implementations.
- Don't wrap a function in another function that only forwards its arguments.
- Don't split code into many tiny single-use helpers. A 30-line function that reads top to bottom is fine.
- Don't add config options, flags or parameters nobody asked for.
- Don't build for hypothetical future needs. Build what the current phase needs.
- Functions can be different lengths. Not everything needs to be five tidy lines.

### 4.5 Error handling

- Catch specific exceptions, at boundaries only (network calls, LLM parsing, file IO, worker processes).
- No `except Exception: pass`. No blanket try/except around whole functions.
- No defensive checks for things the type system or the caller already guarantees.
- Error messages: short and specific. `raise ValueError(f"payload {kg} kg out of range")`, not a paragraph.

### 4.6 Types, constants, logging

- Type-hint function signatures. Don't annotate every obvious local.
- Name a constant only if it's reused or its meaning isn't obvious (put the unit in the name: `CTRL_HZ`, `TORQUE_SAT_RATIO`). Leave other numbers inline.
- Log messages are plain lowercase text with useful fields. No emoji, no banners, no "Successfully completed!".
- Use `logging`, not stray `print`, in library code. Scripts may print tables.

### 4.7 Wording in docs and text

- Write plainly and directly, the way you'd explain it to a teammate.
- Avoid marketing words in code comments, README and docs: *robust, seamless, comprehensive, cutting-edge, leverage, utilize, empower, streamline, delve, holistic*.
- Avoid stock phrasings like "It's not just X, it's Y", triple adjective lists, and closing summaries that repeat the section.
- Don't put headings, bullets and bold on everything. Use prose where prose is clearer.
- Go easy on em dashes. Use a comma or a period.
- No emoji in code, logs, commits, docs or test names (the dashboard UI may use them only if the user asks).

### 4.8 Leftovers that give it away (delete before finishing)

- Chat-style text in files: "Here's the updated code", "Updated implementation", "As requested".
- "Generated by ..." or "Created by AI" headers in source files.
- Placeholder values (`your_key_here` outside `.env.example`, `foo`, `example.com`) left in real code.
- Unused imports, dead parameters, unreachable branches, duplicated helpers.
- Over-explained `if __name__ == "__main__"` blocks with usage essays.
- Empty `__init__` classes, empty `pass` stubs, `NotImplementedError` placeholders in finished work.

Attribution lines and co-author trailers in commits are the team's decision, not yours. Don't add or strip them on your own.

### 4.9 Match the surroundings

- Follow the formatting and idioms already in the file. Let `ruff format` and `prettier` decide layout.
- Don't reformat or reorder code you aren't changing.
- Match the existing test style, fixture names and folder layout.

### 4.10 Example: what we want

Bad (reads as generated):

```python
def calculate_torque_ratio_for_joint(joint_torque: float, joint_torque_limit: float) -> float:
    """
    Calculate the torque ratio for a given joint.

    Args:
        joint_torque (float): The current torque applied to the joint.
        joint_torque_limit (float): The maximum allowable torque for the joint.

    Returns:
        float: The ratio of the current torque to the maximum torque limit.
    """
    # Check if the torque limit is valid
    if joint_torque_limit <= 0:
        raise ValueError("Joint torque limit must be greater than zero.")
    # Calculate the ratio
    ratio = abs(joint_torque) / joint_torque_limit
    # Return the ratio
    return ratio
```

Good:

```python
def tau_ratio(tau, limit):
    return np.abs(tau) / limit
```

Bad:

```python
try:
    result = run_episode(params)
except Exception as e:
    logger.error(f"An error occurred while running the episode: {e}")
    return None
```

Good:

```python
try:
    result = run_episode(params)
except TimeoutError:
    log.warning("episode %s timed out", params.seed)
    return EpisodeResult.errored(params)
```

---

## 5. Python conventions

- Python 3.11, `from __future__ import annotations` only if needed.
- Format and lint with `ruff` (line length 100). Types checked with `pyright`.
- Pydantic v2 (`model_validate`, `model_dump`). Use `Field(..., ge=, le=)` for bounds.
- Use `pathlib.Path`, f-strings, dataclasses for plain records, `numpy` arrays for telemetry.
- No wildcard imports. No mutable default arguments. No global state except the LLM client and the usage counters.
- One PyBullet client per process. Parallelism uses processes (`spawn`), not threads.
- Scripts live in `scripts/`, library code in `src/omnisim/`. Library code doesn't read `sys.argv`.

## 6. TypeScript / React conventions

- TypeScript strict mode. Function components and hooks. State in `zustand`; keep component-local state local.
- One data layer: `web/src/api/client.ts`. Components never call `fetch` directly.
- Mock data in `web/public/mock/` must match `src/omnisim/schemas.py`. Update both together.
- Tailwind utility classes; no inline style objects unless a value is dynamic (heat colours, transforms).
- R3F: keep geometry/material creation out of render loops; memoize arrays passed to `<Line>`.
- Every async view has loading, empty and error states.
- Keep components under about 200 lines. Split when it helps readability, not before.

---

## 7. LLM (Nebius Token Factory) rules

- Call through `src/omnisim/llm/nebius.py` only (`chat_json`). Don't create other clients.
- Roles: adversary uses the fast model, diagnosis and mitigation use the deep model. IDs come from env.
- Nemotron may return an empty `content` and put text elsewhere, wrap JSON in text, or burn tokens on reasoning. The parser handles this; don't work around it in callers.
- Always set `max_tokens` high enough for reasoning plus the JSON.
- Prompts live in `src/omnisim/llm/prompts/*.md`, not inline in code.
- Prompts for diagnosis and mitigation must say: use only the provided evidence and measured values.
- Every call is logged (role, model, latency, tokens) without prompt contents that include secrets.
- Respect the token budget guard. Never bypass it for convenience.
- Tests use recorded fixtures in `tests/fixtures/llm_cassettes/`. No live LLM calls in CI.

## 8. Tavily rules

- Use `search` (advanced depth, manufacturer domains), `extract` for full pages, and the error-taxonomy and known-issue searches described in `docs/SPEC.md`.
- Cache every result in `data/tavily_cache/`. A cache hit must be visible in the usage counters.
- The LLM fills specs only from the extracted text it is given. The verifier checks quote or number against that text.
- Report URDF-vs-documentation mismatches only when they are real.

## 9. Simulation rules

- Physics 240 Hz, control and telemetry 60 Hz, 360 ticks per episode, unless SPEC changes.
- `p.DIRECT` only in code and tests. Use `p.GUI` by hand when debugging and never commit it.
- Panda arm joints are 0 to 6, fingers 9 and 10, end effector link 11.
- `calculateJacobian` takes position, velocity and acceleration lists for all movable joints.
- Detectors and the risk score follow SPEC section 8. Thresholds live in `DetectorConfig`, not scattered through code.
- The nominal scenario must stay clean (risk well below 1). If a change breaks that, fix it before anything else.
- Held-out verification scenarios use a separate seed that the adversary never sees.

---

## 10. Testing

- Write tests first for parsers, detectors, shield, verifier and clustering.
- Unit tests make no network calls and don't launch the GUI.
- Integration tests for the sim keep episodes short and use fixed seeds.
- Add a `hypothesis` property test for any code whose job is to enforce a limit.
- Tests have plain names that say the behaviour: `test_latency_raises_tracking_error`. Parametrize repeated cases instead of copy-pasting. Don't mock what you can run for real in under a second.
- Fix the code, not the test, unless the test is wrong. If you change a test, say why.
- `make check` (ruff, pyright, pytest) must pass before you report done.

## 11. Git, branches, docs

- Branch per task: `feat/<thing>`, `fix/<thing>`. One agent per branch or worktree.
- Commit messages: short, lowercase, imperative, conventional prefix. Examples: `feat(sim): add payload inertia`, `fix(llm): handle empty content`. No body unless it helps a reviewer.
- Commit small and often so history shows steady work during the hackathon period.
- Don't commit: `.env`, keys, `runs/` output (except `data/demo_runs/`), caches other than `data/tavily_cache/`, large binaries, `node_modules`.
- Update `docs/PROGRESS.md` (done, next, blockers) and `docs/DECISIONS.md` when a decision changes something others rely on.
- Add to `docs/NEBIUS_FEEDBACK.md` when you hit a real quirk in Nebius, Nemotron, Tavily or NVIDIA tooling (what you did, what happened, what you expected).
- README text follows section 4.7.

---

## 12. Definition of done and how to report

A task is done when:
- the acceptance commands for the task pass and you ran them,
- `make check` passes,
- you reread the diff against section 4 and cleaned it up,
- docs and tests that depend on the change are updated.

Final message format (keep it short and plain):

```
Changed: <files and what changed, one line each>
Ran: <exact commands>  ->  <real output summary>
Not done / unsure: <anything skipped, assumptions, risks>
Next: <suggested next step, optional>
```

Don't praise your own work. Don't claim something works if you didn't run it.

---

## 13. Never do this

- Hardcode keys, endpoints or model IDs.
- Use a non-NVIDIA model for the core adversary, diagnosis or mitigation steps.
- Present cached or recorded data as live.
- Disable, loosen or skip a failing test, detector or shield check to get green.
- Write new code that depends on an unverified number from the LLM.
- Add telemetry, analytics or any network call not listed in SPEC.
- Leave the repo in a state where `make check` fails.
- Add comments, docs or wording that violate section 4.

---

## 14. Quick commands

```bash
make setup        # uv sync, pre-commit install
make check        # ruff + pyright + pytest -q
make sim-smoke    # 5 headless episodes
make api          # FastAPI on :8000
make web          # Vite on :5173
python scripts/smoke_nebius.py
python scripts/run_campaign.py --budget 40 --verify
python scripts/run_benchmark.py --budget 60 --seeds 5
```
