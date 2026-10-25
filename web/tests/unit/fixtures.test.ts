import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { validateReplay } from "../../src/replay/validation";

describe("generated fixtures", () => {
  it("contains campaign episodes with valid replay fixtures", () => {
    const campaign = JSON.parse(readFileSync("public/mock/campaign.json", "utf8")) as {
      episodes: { episode_id: string }[];
    };
    expect(campaign.episodes.length).toBeGreaterThan(0);
    for (const episode of campaign.episodes) {
      const replay = JSON.parse(
        readFileSync(`public/mock/episode_${episode.episode_id}.json`, "utf8"),
      ) as unknown;
      expect(validateReplay(replay).ok, episode.episode_id).toBe(true);
    }
    const nominal = JSON.parse(readFileSync("public/mock/episode_nominal.json", "utf8")) as unknown;
    expect(validateReplay(nominal).ok).toBe(true);
  });

  it("keeps generated OpenAPI types in sync", () => {
    expect(() => execFileSync(process.execPath, ["scripts/check-generated-types.mjs"], { stdio: "pipe" }))
      .not.toThrow();
  }, 30_000);
});
