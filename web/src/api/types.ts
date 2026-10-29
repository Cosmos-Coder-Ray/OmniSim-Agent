import type { components } from "./openapi.generated";

export type ApiError = components["schemas"]["ApiError"];
export type CampaignState = components["schemas"]["CampaignState"];
export type EpisodeResult = components["schemas"]["EpisodeResult"];
export type FailureEvent = components["schemas"]["FailureEvent"];
export type FailureKind = FailureEvent["kind"];
export type HealthResponse = components["schemas"]["HealthResponse"];
export type ScenarioParams = components["schemas"]["ScenarioParams"];
export type Benchmark = components["schemas"]["BenchmarkResult"];

export interface EvidenceClaim {
  claim: string;
  value: number | string;
  unit: string | null;
  source_url: string;
  quote: string;
  verified: boolean;
}

export interface EvidenceView {
  robot: string;
  joint_torque_limits_nm: EvidenceClaim[];
  joint_velocity_limits: EvidenceClaim[];
  joint_position_limits: EvidenceClaim[];
  error_taxonomy: EvidenceClaim[];
  mismatches?: {
    joint: number;
    field: "torque" | "velocity" | "lower" | "upper";
    urdf: number;
    doc: number;
    source_url: string;
    sim_more_permissive: boolean;
  }[];
}

export interface EnvelopeView {
  robot: string;
  limits: {
    max_payload_kg: number;
    max_latency_ms: number;
    torque_utilization_cap: number;
  };
  verification?: {
    n_scenarios: number;
    fail_rate_before: number;
    fail_rate_after: number;
  } | null;
}

export type DataSource = "mock" | "recorded" | "live";

export type DataResult<T> =
  | { kind: "ok"; data: T }
  | { kind: "pending"; stage: string; source?: DataSource }
  | { kind: "unavailable"; reason: string }
  | { kind: "error"; message: string };

export type CampaignView = Omit<CampaignState, "source"> & { source: DataSource };

export interface ReplayFrame {
  t: number;
  links: [number, number, number][];
  ee: [number, number, number];
  tau_ratio: number[];
  vel_ratio: number[];
  flags: FailureKind[];
}

export type ReplayEvent = FailureEvent;

export interface Replay {
  episode_id: string;
  fps: number;
  frames: ReplayFrame[];
  events: ReplayEvent[];
  source: DataSource;
}

export interface DataClient {
  getCampaign(id: string): Promise<DataResult<CampaignView>>;
  getTrajectory(campaignId: string, trajectoryId: string): Promise<DataResult<Replay>>;
  getHealth(): Promise<DataResult<HealthResponse>>;
  getEnvelope(campaignId: string): Promise<DataResult<EnvelopeView>>;
  getEvidence(campaignId: string): Promise<DataResult<EvidenceView>>;
  getBenchmark(): Promise<DataResult<Benchmark>>;
}
