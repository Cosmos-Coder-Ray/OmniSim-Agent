import type { CampaignView, DataResult, Replay } from "../api/types";

interface ReplayViewProps {
  campaign: CampaignView | null;
  replay: DataResult<Replay> | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}

export function ReplayView({ campaign, replay, loading, error, onRetry }: ReplayViewProps) {
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
        <span className="status-pill status-failing">FAILING: torque saturation</span>
      </div>
      <div className="canvas-placeholder" aria-label="Panda replay canvas">
        <span>Panda scene is loading in the next build step.</span>
      </div>
      <div className="replay-placeholder-footer">
        <span>{replay.data.frames.length} frames</span>
        <span>{replay.data.fps} fps source data</span>
      </div>
    </section>
  );
}
