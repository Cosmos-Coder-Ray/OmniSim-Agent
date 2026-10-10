## 4. Architecture

<div class="mermaid">
flowchart TD
  A["Robot + Policy Ingestion<br/>Panda URDF + scripted policy"] --> B
  subgraph Brain["Nebius Token Factory — NVIDIA Nemotron"]
    B["Adversary Agent<br/>Nemotron Lightning"]
    D["Diagnostician + Mitigator<br/>Nemotron Super/Ultra"]
  end
  K["Knowledge Agent<br/>Tavily search + extract<br/>+ claim verifier"] --> D
  B -->|"ScenarioParams JSON"| S["Headless PyBullet workers<br/>240 Hz physics / 60 Hz telemetry"]
  S --> P["Failure Probe<br/>detectors + risk score"]
  P -->|"risk + failures (memory)"| B
  P -->|"failure signature"| C["Clustering + Shrinker"]
  C --> K
  C --> D
  D --> E["Safe Operating Envelope"]
  E --> V["Safety Shield +<br/>held-out verification"]
  V -->|"still failing"| D
  V -->|"pass"| X["Export pack<br/>JSON · ROS2 YAML · pytest · report"]
  S --> L["Trajectory logs"] --> W["React-Three-Fiber dashboard"]
  X --> W
</div>

### Components

| Component | Responsibility | Tech |
| --- | --- | --- |
| Sim core | Load Panda, run policy with perturbations, 240 Hz physics, 60 Hz telemetry | PyBullet (`p.DIRECT`) |
| Policy | Scripted pick-and-place via IK + min-jerk trajectories + PD gains | Python/NumPy |
| Perturbation layer | Payload, mass scale, joint damping, latency, sensor noise, torque degradation, speed scale, external push | PyBullet `changeDynamics` + custom |
| Failure probe | Detectors + continuous **risk score** (distance to failure) | NumPy |
| Adversary | Proposes next `ScenarioParams` given history | Nemotron Lightning via Token Factory |
| Baselines | Random + CMA-ES | NumPy, `cma` |
| Knowledge agent | Tavily search/extract → `HardwareSpec` with verified evidence | `tavily-python`, Nemotron |
| Diagnostician | Root-cause + real-error mapping | Nemotron Super/Ultra |
| Mitigator + Shield | Envelope JSON → deterministic clamps | Pydantic + Python |
| Verifier | Re-run held-out set with/without shield | Runner |
| API | Campaign control, SSE live events, artifacts | FastAPI + SQLite |
| Dashboard | 3D replay, charts, atlas, evidence, before/after | React + R3F + drei + recharts + zustand |

### Core data contracts (freeze these in Phase 1)

```python
# src/omnisim/schemas.py  (Pydantic v2)
class ScenarioParams(BaseModel):
    payload_kg: float = Field(0.0, ge=0.0, le=3.0)
    link_mass_scale: float = Field(1.0, ge=0.7, le=1.5)
    joint_damping: float = Field(0.0, ge=0.0, le=5.0)
    latency_ticks: int = Field(0, ge=0, le=12)          # at 60 Hz control
    sensor_noise_rad: float = Field(0.0, ge=0.0, le=0.02)
    torque_limit_scale: float = Field(1.0, ge=0.5, le=1.0)  # actuator degradation
    speed_scale: float = Field(1.0, ge=0.5, le=2.0)
    kp_scale: float = Field(1.0, ge=0.5, le=2.0)
    push_force_n: float = Field(0.0, ge=0.0, le=40.0)
    push_time_s: float = Field(2.0, ge=0.0, le=5.0)
    seed: int = 0

class FailureEvent(BaseModel):
    kind: Literal["torque_saturation","joint_limit","velocity_limit","tracking_error",
                  "singularity","collision","instability"]
    t: float; joint: int | None = None; severity: float; detail: dict = {}

class EpisodeResult(BaseModel):
    episode_id: str; params: ScenarioParams
    failures: list[FailureEvent]; risk: float          # 0..>1, >=1 means failure
    max_tau_ratio: float; max_vel_ratio: float; final_ee_error_m: float; min_manipulability: float
    trajectory_path: str | None = None

class Evidence(BaseModel):
    claim: str; value: float | str; unit: str | None = None
    source_url: str; quote: str; verified: bool        # verified = quote/number found in fetched text

class HardwareSpec(BaseModel):
    robot: str
    joint_torque_limits_nm: list[Evidence]
    joint_velocity_limits: list[Evidence]
    joint_position_limits: list[Evidence]
    error_taxonomy: list[Evidence]                     # real-robot error/reflex names

class SafeEnvelope(BaseModel):
    robot: str; version: str
    limits: dict            # max_payload_kg, speed_scale_max, max_latency_ms, torque_utilization_cap, ...
    controller: dict        # kp_scale/kd_scale clamps
    domain_randomization_bounds: dict
    evidence: list[Evidence]
    verification: dict      # n_scenarios, seed, fail_rate_before, fail_rate_after
```

### Repo layout

```
omnisim-agent/
├─ AGENTS.md  (+ CLAUDE.md symlink)     # shared rules for ALL AI tools
├─ LICENSE (Apache-2.0)  README.md  Makefile  pyproject.toml  .env.example
├─ docs/  SPEC.md  ARCHITECTURE.md  PROGRESS.md  DECISIONS.md  NEBIUS_FEEDBACK.md  prompts/
├─ src/omnisim/
│  ├─ config.py  schemas.py
│  ├─ llm/        nebius.py  json_utils.py  budget.py  prompts/*.md
│  ├─ knowledge/  tavily_agent.py  verifier.py  cache.py
│  ├─ sim/        panda_env.py  policy.py  perturbations.py  telemetry.py  detectors.py  runner.py  render.py
│  ├─ agents/     adversary.py  baselines.py  diagnostician.py  mitigator.py
│  ├─ core/       campaign.py  clustering.py  shrinker.py  shield.py  verify.py
│  ├─ export/     envelope.py  ros2.py  pytest_suite.py  report.py
│  └─ api/        main.py  sse.py  store.py
├─ web/           (Vite + React + TS + R3F)  public/mock/*.json
├─ tests/         unit/  integration/  e2e/  fixtures/llm_cassettes/
├─ data/          tavily_cache/  demo_runs/
└─ .github/workflows/ci.yml
```

---
