import { afterEach, describe, expect, it, vi } from "vitest";
import { createApiClient } from "../../src/api/client";

describe("API client results", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("maps accepted campaigns to pending", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response('{"message":"queued"}', { status: 202 })));
    await expect(createApiClient("/api").getCampaign("run-1"))
      .resolves.toEqual({ kind: "pending", stage: "queued" });
  });

  it("maps missing campaigns to unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 404 })));
    await expect(createApiClient().getCampaign("missing"))
      .resolves.toMatchObject({ kind: "unavailable" });
  });

  it("maps server failures to a generic error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("private detail", { status: 500 })));
    await expect(createApiClient().getCampaign("run-1"))
      .resolves.toEqual({ kind: "error", message: "The API request failed. Retry to try again." });
  });

  it("maps transport failures to an error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await expect(createApiClient().getCampaign("run-1"))
      .resolves.toEqual({ kind: "error", message: "offline" });
  });

  it("requests envelope, evidence, and benchmark data from the API", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const client = createApiClient("/api");
    await client.getEnvelope("run/1");
    await client.getEvidence("run/1");
    await client.getBenchmark();
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/api/exports/run%2F1/safe_envelope.json",
      "/api/exports/run%2F1/verification.json",
      "/api/campaigns/run%2F1/evidence",
      "/api/benchmark",
    ]);
  });
});
