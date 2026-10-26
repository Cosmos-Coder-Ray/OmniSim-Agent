import type { Replay, ReplayFrame } from "../api/types";

export const SEGMENT_LENGTH_TOLERANCE_M = 0.001;

export type ReplayValidation = { ok: true; duration: number } | { ok: false; error: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isPoint(value: unknown): value is [number, number, number] {
  return Array.isArray(value) && value.length === 3 && value.every(isFiniteNumber);
}

function distance(a: [number, number, number], b: [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function frameError(frame: unknown, previousTime: number): string | null {
  if (!isRecord(frame)) return "Replay frame is malformed.";
  if (!isFiniteNumber(frame.t) || frame.t < 0 || frame.t <= previousTime) {
    return "Replay frame times must be finite and strictly increasing.";
  }
  if (!Array.isArray(frame.links) || frame.links.length !== 8 || !frame.links.every(isPoint)) {
    return "Replay frames must contain eight finite link positions.";
  }
  if (!isPoint(frame.ee)) return "Replay end-effector position is malformed.";
  for (const key of ["tau_ratio", "vel_ratio"] as const) {
    const ratios = frame[key];
    if (!Array.isArray(ratios) || ratios.length !== 7 || !ratios.every(isFiniteNumber)) {
      return `Replay frames must contain seven finite ${key} values.`;
    }
  }
  if (!Array.isArray(frame.flags) || !frame.flags.every((flag) => typeof flag === "string")) {
    return "Replay frame flags are malformed.";
  }
  return null;
}

export function replayDuration(replay: Pick<Replay, "frames" | "fps">): number {
  const last = replay.frames.at(-1);
  return last ? last.t + 1 / replay.fps : 0;
}

export function validateReplay(value: unknown): ReplayValidation {
  if (!isRecord(value) || !isFiniteNumber(value.fps) || value.fps <= 0) {
    return { ok: false, error: "Replay frame rate must be finite and greater than zero." };
  }
  if (!Array.isArray(value.frames) || value.frames.length === 0) {
    return { ok: false, error: "Replay contains no frames." };
  }

  let previousTime = -1;
  for (const frame of value.frames) {
    const error = frameError(frame, previousTime);
    if (error) return { ok: false, error };
    previousTime = frame.t;
  }

  const frames = value.frames as ReplayFrame[];
  const firstLinks = frames[0].links;
  const lengths = firstLinks.slice(1).map((point, index) => distance(firstLinks[index], point));
  for (const frame of frames.slice(1)) {
    for (let index = 0; index < lengths.length; index += 1) {
      const actual = distance(frame.links[index], frame.links[index + 1]);
      if (Math.abs(actual - lengths[index]) > SEGMENT_LENGTH_TOLERANCE_M) {
        return { ok: false, error: "Replay link segment lengths vary by more than 1 mm." };
      }
    }
  }

  if (!Array.isArray(value.events)) return { ok: false, error: "Replay events are malformed." };
  const duration = replayDuration({ frames, fps: value.fps });
  for (const event of value.events) {
    if (!isRecord(event) || !isFiniteNumber(event.t) || !isFiniteNumber(event.severity)) {
      return { ok: false, error: "Replay event is malformed." };
    }
    if (event.t < 0 || event.t > duration) {
      return { ok: false, error: "Replay event time is outside the replay duration." };
    }
  }
  return { ok: true, duration };
}
