import type { components } from "./openapi.generated";

export type ApiError = components["schemas"]["ApiError"];
export type CampaignState = components["schemas"]["CampaignState"];
export type EpisodeResult = components["schemas"]["EpisodeResult"];
export type FailureEvent = components["schemas"]["FailureEvent"];
export type FailureKind = FailureEvent["kind"];
export type HealthResponse = components["schemas"]["HealthResponse"];
export type ScenarioParams = components["schemas"]["ScenarioParams"];

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
}
