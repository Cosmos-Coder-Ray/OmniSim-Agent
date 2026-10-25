import { useEffect, useMemo, useState } from "react";
import { dataClient, dataMode } from "../api/client";
import type { CampaignView, DataResult, Replay } from "../api/types";
import { DEFAULT_CAMPAIGN, tabs, useUrlStore } from "../state/urlStore";
import { PlaceholderTab } from "./PlaceholderTab";
import { ReplayView } from "./ReplayView";

export function AppShell() {
  const tab = useUrlStore((state) => state.tab);
  const campaignId = useUrlStore((state) => state.campaign);
  const episodeId = useUrlStore((state) => state.episode);
  const setTab = useUrlStore((state) => state.setTab);
  const setCampaign = useUrlStore((state) => state.setCampaign);
  const setEpisode = useUrlStore((state) => state.setEpisode);
  const syncFromUrl = useUrlStore((state) => state.syncFromUrl);
  const [campaignResponse, setCampaignResponse] = useState<{
    campaignId: string;
    result: DataResult<CampaignView>;
  } | null>(null);
  const [replayResponse, setReplayResponse] = useState<{
    campaignId: string;
    episodeId: string;
    attempt: number;
    result: DataResult<Replay>;
  } | null>(null);
  const [health, setHealth] = useState("checking");
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    const onPopState = () => syncFromUrl();
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [syncFromUrl]);

  useEffect(() => {
    let active = true;
    void dataClient.getHealth().then((result) => {
      if (!active) return;
      if (result.kind === "ok") {
        setHealth(result.data.status.toUpperCase());
        if (dataMode === "api" && campaignId === DEFAULT_CAMPAIGN) {
          setCampaign(result.data.recorded_runs[0] ?? "");
        }
      } else {
        setHealth(result.kind === "unavailable" ? "MOCK MODE" : "UNAVAILABLE");
      }
    });
    return () => {
      active = false;
    };
  }, [campaignId, retryCount, setCampaign]);

  useEffect(() => {
    let active = true;
    if (!campaignId) return;
    void dataClient.getCampaign(campaignId).then((result) => {
      if (!active) return;
      setCampaignResponse({ campaignId, result });
      if (result.kind === "ok") {
        const episodes = result.data.episodes ?? [];
        const currentEpisode = useUrlStore.getState().episode;
        if (episodes.length > 0 && !episodes.some((item) => item.episode_id === currentEpisode)) {
          const selected = [...episodes]
            .filter((item) => item.failures.length > 0)
            .sort((a, b) => b.risk - a.risk)[0] ?? episodes[0];
          setEpisode(selected.episode_id);
        }
      }
    });
    return () => {
      active = false;
    };
  }, [campaignId, retryCount, setEpisode]);

  const campaignResult = campaignResponse?.campaignId === campaignId ? campaignResponse.result : null;
  const campaign = campaignResult?.kind === "ok" ? campaignResult.data : null;
  const loadingCampaign = campaignId !== "" && campaignResult === null;
  const campaignPending = campaignResult?.kind === "pending" ? campaignResult.stage : null;
  const campaignError = campaignResult?.kind === "error" ? campaignResult.message : null;

  const selectedEpisode = useMemo(
    () => campaign?.episodes?.find((item) => item.episode_id === episodeId) ?? null,
    [campaign, episodeId],
  );

  useEffect(() => {
    if (!campaignId || !selectedEpisode) return;
    let active = true;
    const trajectoryId = selectedEpisode.trajectory_path ?? selectedEpisode.episode_id;
    void dataClient.getTrajectory(campaignId, trajectoryId).then((result) => {
      if (active) {
        setReplayResponse({
          campaignId,
          episodeId: selectedEpisode.episode_id,
          attempt: retryCount,
          result,
        });
      }
    });
    return () => {
      active = false;
    };
  }, [campaignId, selectedEpisode, retryCount]);

  const replay = replayResponse?.campaignId === campaignId &&
    replayResponse.episodeId === selectedEpisode?.episode_id &&
    replayResponse.attempt === retryCount
    ? replayResponse.result
    : null;

  const retry = () => setRetryCount((value) => value + 1);
  const source = replay?.kind === "ok" ? replay.data.source : campaign?.source ?? dataMode;

  return (
    <main className="app-shell">
      <header className="app-header">
        <a className="brand" href="#top" aria-label="OmniSim home">
          <span className="brand-mark" aria-hidden="true">O</span>
          <span>OmniSim</span>
        </a>
        <div className="header-context">
          <span className="source-badge" data-source={source}>SOURCE · {source.toUpperCase()}</span>
          <span className="health-badge"><span className="health-dot" /> API · {health}</span>
        </div>
      </header>

      <nav className="tab-strip" aria-label="Workspace tabs" role="tablist">
        {tabs.map((item) => (
          <button
            key={item.id}
            id={`tab-${item.id}`}
            className={`tab-button${tab === item.id ? " is-active" : ""}`}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            aria-controls="workspace-panel"
            tabIndex={tab === item.id ? 0 : -1}
            onClick={() => setTab(item.id)}
            onKeyDown={(event) => {
              if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
              event.preventDefault();
              const currentIndex = tabs.findIndex((candidate) => candidate.id === tab);
              const direction = event.key === "ArrowRight" ? 1 : -1;
              const next = tabs[(currentIndex + direction + tabs.length) % tabs.length];
              setTab(next.id);
              document.getElementById(`tab-${next.id}`)?.focus();
            }}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <section
        id="workspace-panel"
        className="workspace-panel"
        role="tabpanel"
        aria-labelledby={`tab-${tab}`}
        tabIndex={0}
      >
        {tab === "replay" ? (
          <>
            <div className="episode-toolbar">
              <div>
                <div className="section-kicker">CAMPAIGN</div>
                <strong>{campaignId || "No recorded campaign"}</strong>
              </div>
              <label className="episode-select">
                <span>Episode</span>
                <select
                  value={episodeId}
                  onChange={(event) => setEpisode(event.target.value)}
                  disabled={!campaign?.episodes?.length}
                  aria-label="Select episode"
                >
                  {campaign?.episodes?.map((item) => (
                    <option key={item.episode_id} value={item.episode_id}>
                      {item.episode_id}{item.failures.length ? " · failing" : " · nominal"}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {!campaignId && dataMode === "api" ? (
              <section className="state-panel">No recorded campaigns are available from the API.</section>
            ) : campaignPending ? (
              <section className="state-panel" aria-live="polite">{campaignPending}</section>
            ) : (
              <ReplayView
                campaign={campaign}
                replay={replay}
                loading={loadingCampaign || (selectedEpisode !== null && replay === null)}
                error={campaignError}
                onRetry={retry}
              />
            )}
          </>
        ) : (
          <PlaceholderTab title={tabs.find((item) => item.id === tab)?.label ?? "Workspace"}>
            {tab === "campaign" && "Live campaign controls and event streaming are not connected yet."}
            {tab === "atlas" && "Failure clusters and minimized repros are not connected yet."}
            {tab === "evidence" && "Evidence browsing is not connected yet."}
            {tab === "envelope" && "Verified operating limits are not connected yet."}
            {tab === "benchmarks" && "Benchmark comparisons are not connected yet."}
          </PlaceholderTab>
        )}
      </section>
    </main>
  );
}
