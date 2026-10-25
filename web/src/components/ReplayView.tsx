import { lazy, Suspense, useMemo, useRef } from "react";
import type { CampaignView, DataResult, Replay } from "../api/types";
import { jointLabel } from "../replay/geometry";
import type { ReplayCanvasHandle } from "./ReplayCanvas";

const ReplayCanvas = lazy(() => import("./ReplayCanvas").then((module) => ({ default: module.ReplayCanvas })));

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
  const event = useMemo(
    () => replay?.kind === "ok"
      ? [...replay.data.events].sort((a, b) => b.severity - a.severity)[0]
      : undefined,
    [replay],
  );
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
      <div className="replay-placeholder-footer">
        <span>{replay.data.frames.length} frames</span>
        <span>{jointLabel(event?.joint ?? null)} · source sample {replay.data.fps} Hz</span>
      </div>
    </section>
  );
}
