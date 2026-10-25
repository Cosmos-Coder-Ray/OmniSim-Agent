import type {
  CampaignView,
  DataClient,
  DataResult,
  DataSource,
  EpisodeResult,
  HealthResponse,
  Replay,
} from "./types";

export const dataMode = import.meta.env.VITE_DATA_SOURCE === "api" ? "api" : "mock";
const apiBase = (import.meta.env.VITE_API_BASE || "/api").replace(/\/$/, "");

interface MockCampaign {
  robot: string;
  budget: number;
  strategy: string;
  seed: number;
  episodes: EpisodeResult[];
  source: DataSource;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed.";
}

async function request<T>(url: string): Promise<DataResult<T>> {
  try {
    const response = await fetch(url);
    if (response.status === 202) {
      const payload = (await response.json()) as { message?: string };
      return { kind: "pending", stage: payload.message ?? "Campaign is still running." };
    }
    if (response.status === 404) {
      return { kind: "unavailable", reason: "This campaign or replay is unavailable." };
    }
    if (!response.ok) {
      return { kind: "error", message: "The API request failed. Retry to try again." };
    }
    return { kind: "ok", data: (await response.json()) as T };
  } catch (error) {
    return { kind: "error", message: errorMessage(error) };
  }
}

async function mockJson<T>(path: string): Promise<DataResult<T>> {
  try {
    const response = await fetch(path);
    if (response.status === 404) {
      return { kind: "unavailable", reason: "This mock recording is unavailable." };
    }
    if (!response.ok) {
      return { kind: "error", message: "The mock recording could not be loaded." };
    }
    return { kind: "ok", data: (await response.json()) as T };
  } catch (error) {
    return { kind: "error", message: errorMessage(error) };
  }
}

function mockCampaignView(id: string, campaign: MockCampaign): CampaignView {
  const done = campaign.episodes.length > 0;
  return {
    campaign_id: id,
    status: done ? "completed" : "no_failures_found",
    stage: "done",
    stages: ["search", "shrink", "spec", "diagnosis", "done"],
    source: campaign.source,
    config: {
      robot: campaign.robot,
      strategy: campaign.strategy,
      seed: campaign.seed,
      budget: campaign.budget,
    },
    progress: { last_event_id: 0, sims_used: campaign.budget, clusters: 1 },
    episodes: campaign.episodes,
    created_at: null,
    started_at: null,
    finished_at: null,
    queue_position: null,
  };
}

export const dataClient: DataClient = {
  async getCampaign(id) {
    if (dataMode === "mock") {
      const result = await mockJson<MockCampaign>("/mock/campaign.json");
      return result.kind === "ok"
        ? { kind: "ok", data: mockCampaignView(id, result.data) }
        : result;
    }
    return request<CampaignView>(`${apiBase}/campaigns/${encodeURIComponent(id)}`);
  },

  getTrajectory(campaignId, trajectoryId) {
    if (dataMode === "mock") {
      const episodeId = trajectoryId === "nominal" ? "nominal" : trajectoryId;
      return mockJson<Replay>(`/mock/episode_${encodeURIComponent(episodeId)}.json`);
    }
    return request<Replay>(
      `${apiBase}/campaigns/${encodeURIComponent(campaignId)}/trajectories/${trajectoryId
        .split("/")
        .map(encodeURIComponent)
        .join("/")}`,
    );
  },

  getHealth() {
    if (dataMode === "mock") {
      return Promise.resolve({ kind: "unavailable", reason: "Health is unavailable in mock mode." });
    }
    return request<HealthResponse>(`${apiBase}/health`);
  },
};
