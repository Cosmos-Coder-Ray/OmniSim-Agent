import { describe, expect, it } from "vitest";
import { validateReplay } from "../../src/replay/validation";
import type { Replay } from "../../src/api/types";

function validReplay(): Replay {
  const frame = (t: number) => ({
    t,
    links: Array.from({ length: 8 }, (_, index) => [index * 0.1, 0, 0] as [number, number, number]),
    ee: [0.7, 0, 0] as [number, number, number],
    tau_ratio: Array(7).fill(0.4),
    vel_ratio: Array(7).fill(0.5),
    flags: [],
  });
  return { episode_id: "ep", fps: 30, frames: [frame(0), frame(1)], events: [], source: "mock" };
}

describe("replay validation", () => {
  it.each([
    ["non-finite frame rate", (replay: Replay) => { replay.fps = Number.NaN; }],
    ["empty frames", (replay: Replay) => { replay.frames = []; }],
    ["non-monotonic frame times", (replay: Replay) => { replay.frames[1].t = 0; }],
    ["wrong link count", (replay: Replay) => { replay.frames[0].links.pop(); }],
    ["wrong ratio count", (replay: Replay) => { replay.frames[0].tau_ratio.pop(); }],
    ["non-finite values", (replay: Replay) => { replay.frames[0].ee[0] = Number.NaN; }],
    ["event outside duration", (replay: Replay) => {
      replay.events = [{ kind: "tracking_error", t: 2, joint: null, severity: 0.5, detail: {} }];
    }],
    ["varying segment lengths", (replay: Replay) => { replay.frames[1].links[1][0] += 0.01; }],
  ])("rejects %s", (_name, mutate) => {
    const replay = validReplay();
    mutate(replay);
    expect(validateReplay(replay).ok).toBe(false);
  });

  it("uses the final sample plus one frame interval for duration", () => {
    expect(validateReplay(validReplay())).toEqual({ ok: true, duration: 1 + 1 / 30 });
  });
});
