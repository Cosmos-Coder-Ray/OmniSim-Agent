import { describe, expect, it } from "vitest";
import type { ReplayEvent, ReplayFrame } from "../../src/api/types";
import {
  eventMarkerPosition,
  heatColor,
  interpolateReplayFrame,
  jointLabel,
  layoutEventMarkers,
  sphereTorqueRatio,
} from "../../src/replay/geometry";

const frame = (t: number, x: number): ReplayFrame => ({
  t,
  links: Array.from({ length: 8 }, (_, index) => [x + index, 0, 0]),
  ee: [x, 1, 2],
  tau_ratio: Array(7).fill(x),
  vel_ratio: Array(7).fill(0),
  flags: [],
});

describe("replay geometry", () => {
  it("keeps the base sphere neutral and maps each arm sphere to its ratio", () => {
    const ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    expect(sphereTorqueRatio(0, ratios)).toBeNull();
    ratios.forEach((ratio, index) => expect(sphereTorqueRatio(index + 1, ratios)).toBe(ratio));
  });

  it("clamps heat values to the scale endpoints", () => {
    expect(heatColor(-1)).toBe(heatColor(0));
    expect(heatColor(2)).toBe(heatColor(1.2));
  });

  it("uses one-based joint labels", () => {
    expect(jointLabel(0)).toBe("J1");
    expect(jointLabel(6)).toBe("J7");
    expect(jointLabel(null)).toBe("Global");
  });

  it("interpolates frames and keeps event markers in duration bounds", () => {
    const interpolated = interpolateReplayFrame([frame(0, 0), frame(2, 2)], 1);
    expect(interpolated.ee).toEqual([1, 1, 2]);
    expect(interpolated.tau_ratio).toEqual(Array(7).fill(1));
    expect(eventMarkerPosition(3, 4)).toBe(75);
    expect(eventMarkerPosition(5, 4)).toBe(100);
  });

  it("stacks nearby event markers into separate rows", () => {
    const event = (t: number): ReplayEvent => ({
      kind: "tracking_error", t, joint: 0, severity: 0.5, detail: {},
    });
    expect(layoutEventMarkers([event(1), event(1.01), event(3)], 4).map((item) => item.row))
      .toEqual([0, 1, 0]);
  });
});
