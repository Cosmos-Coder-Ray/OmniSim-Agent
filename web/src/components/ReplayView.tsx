import { lazy, Suspense, useCallback, useEffect, useMemo, useRef } from "react";
import type { CampaignView, DataResult, Replay } from "../api/types";
import { jointLabel } from "../replay/geometry";
import { replayDuration } from "../replay/validation";
import { useUrlStore } from "../state/urlStore";
import { EventTimeline } from "./EventTimeline";
import { PlaybackControls } from "./PlaybackControls";
import type { ReplayCanvasHandle } from "./ReplayCanvas";

const ReplayCanvas = lazy(() => import("./ReplayCanvas").then((module) => ({ default: module.ReplayCanvas })));
const ReplayCharts = lazy(() => import("./ReplayCharts").then((module) => ({ default: module.ReplayCharts })));

interface ReplayViewProps {
  campaign: CampaignView | null;
  replay: DataResult<Replay> | null;
  nominal: Replay | null;
  initialTime: number;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}

export function ReplayView({ campaign, replay, nominal, initialTime, loading, error, onRetry }: ReplayViewProps) {
  const canvas = useRef<ReplayCanvasHandle>(null);
  const timelineCursor = useRef<HTMLDivElement>(null);
  const torqueCursor = useRef<HTMLDivElement>(null);
  const velocityCursor = useRef<HTMLDivElement>(null);
  const currentTime = useUrlStore((state) => state.t);
  const seekTo = useUrlStore((state) => state.seekTo);
  const event = useMemo(
    () => replay?.kind === "ok"
      ? [...replay.data.events].sort((a, b) => b.severity - a.severity)[0]
      : undefined,
    [replay],
  );
  const duration = replay?.kind === "ok" ? replayDuration(replay.data) : 0;
  const updateTime = useCallback((time: number) => {
    const position = `${duration > 0 ? Math.max(0, Math.min(100, (time / duration) * 100)) : 0}%`;
    if (timelineCursor.current) timelineCursor.current.style.left = position;
    if (torqueCursor.current) torqueCursor.current.style.left = position;
    if (velocityCursor.current) velocityCursor.current.style.left = position;
    canvas.current?.setTime(time);
  }, [duration]);

  useEffect(() => updateTime(currentTime), [currentTime, updateTime]);
  if (loading) {
    return <section className="state-panel" aria-live="polite">Loading replay…</section>;
  }
  if (error) {
    return (
      <section className="state-panel state-error" role="alert">
        <p>{error}</p>
        <button className="button button-secondary" onClick={onRetry}>Retry</button>
      </section>
    );
  }
  if (!campaign || !campaign.episodes?.length) {
    return <section className="state-panel">No replay is available for this campaign.</section>;
  }
  if (replay?.kind === "pending") {
    return <section className="state-panel" aria-live="polite">{replay.stage}</section>;
  }
  if (replay?.kind === "unavailable") {
    return <section className="state-panel">{replay.reason}</section>;
  }
  if (replay?.kind === "error") {
    return (
      <section className="state-panel state-error" role="alert">
        <p>{replay.message}</p>
        <button className="button button-secondary" onClick={onRetry}>Retry</button>
      </section>
    );
  }
  if (!replay || replay.kind !== "ok") {
    return <section className="state-panel">Select an episode to open its replay.</section>;
  }

  return (
    <section className="replay-placeholder">
      <div className="replay-placeholder-heading">
        <div>
          <div className="section-kicker">EPISODE REPLAY</div>
          <h2>{replay.data.episode_id}</h2>
        </div>
        {event && <span className="status-pill status-failing">FAILING: {event.kind.replaceAll("_", " ")}</span>}
      </div>
      <Suspense fallback={<div className="canvas-frame canvas-loading" aria-live="polite">Loading 3D scene…</div>}>
        <ReplayCanvas ref={canvas} replay={replay.data} nominal={nominal} initialTime={initialTime} />
      </Suspense>
      <PlaybackControls canvas={canvas} duration={duration} onTime={updateTime} />
      <EventTimeline
        events={replay.data.events}
        duration={duration}
        cursorRef={timelineCursor}
        onSeek={(time) => {
          seekTo(time);
          updateTime(time);
        }}
      />
      <Suspense fallback={<div className="chart-loading" aria-live="polite">Loading charts…</div>}>
        <ReplayCharts replay={replay.data} torqueCursorRef={torqueCursor} velocityCursorRef={velocityCursor} />
      </Suspense>
      <div className="replay-placeholder-footer">
        <span>{replay.data.frames.length} frames</span>
        <span>{jointLabel(event?.joint ?? null)} · source sample {replay.data.fps} Hz</span>
      </div>
    </section>
  );
}
