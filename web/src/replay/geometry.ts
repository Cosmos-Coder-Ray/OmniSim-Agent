import type { ReplayEvent, ReplayFrame } from "../api/types";

const HEAT_STOPS = [
  { at: 0, rgb: [48, 154, 136] },
  { at: 0.55, rgb: [217, 188, 86] },
  { at: 1.2, rgb: [225, 83, 86] },
] as const;

export function sphereTorqueRatio(sphereIndex: number, tauRatio: readonly number[]): number | null {
  if (sphereIndex === 0) return null;
  return tauRatio[sphereIndex - 1] ?? 0;
}

export function heatColor(ratio: number): string {
  const value = Number.isFinite(ratio) ? Math.max(0, Math.min(1.2, ratio)) : 0;
  const upperIndex = HEAT_STOPS.findIndex((stop) => stop.at >= value);
  const lower = HEAT_STOPS[Math.max(0, upperIndex - 1)];
  const upper = HEAT_STOPS[upperIndex < 0 ? HEAT_STOPS.length - 1 : upperIndex];
  const portion = upper.at === lower.at ? 0 : (value - lower.at) / (upper.at - lower.at);
  const rgb = lower.rgb.map((channel, index) => Math.round(channel + (upper.rgb[index] - channel) * portion));
  return `#${rgb.map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
}

export function jointLabel(joint: number | null): string {
  return joint === null ? "Global" : `J${joint + 1}`;
}

export function interpolateReplayFrame(frames: readonly ReplayFrame[], t: number): ReplayFrame {
  if (frames.length === 0) throw new Error("Cannot interpolate an empty replay.");
  if (t <= frames[0].t) return frames[0];
  if (t >= frames[frames.length - 1].t) return frames[frames.length - 1];

  let low = 0;
  let high = frames.length - 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (frames[middle].t <= t) low = middle;
    else high = middle;
  }
  const first = frames[low];
  const second = frames[high];
  const portion = (t - first.t) / (second.t - first.t);
  const lerp = (a: number, b: number) => a + (b - a) * portion;
  const point = (a: [number, number, number], b: [number, number, number]) =>
    [lerp(a[0], b[0]), lerp(a[1], b[1]), lerp(a[2], b[2])] as [number, number, number];

  return {
    t,
    links: first.links.map((value, index) => point(value, second.links[index])),
    ee: point(first.ee, second.ee),
    tau_ratio: first.tau_ratio.map((value, index) => lerp(value, second.tau_ratio[index])),
    vel_ratio: first.vel_ratio.map((value, index) => lerp(value, second.vel_ratio[index])),
    flags: portion < 0.5 ? first.flags : second.flags,
  };
}

export interface PositionedEvent extends ReplayEvent {
  leftPercent: number;
  row: number;
}

export function eventMarkerPosition(time: number, duration: number): number {
  if (duration <= 0) return 0;
  return Math.max(0, Math.min(100, (time / duration) * 100));
}

export function layoutEventMarkers(events: readonly ReplayEvent[], duration: number): PositionedEvent[] {
  const lastByRow: number[] = [];
  const markers: PositionedEvent[] = [];
  for (const event of [...events].sort((a, b) => a.t - b.t)) {
    const leftPercent = eventMarkerPosition(event.t, duration);
    let row = 0;
    while (lastByRow[row] !== undefined && leftPercent - lastByRow[row] < 3) row += 1;
    lastByRow[row] = leftPercent;
    markers.push({ ...event, leftPercent, row });
  }
  return markers;
}
