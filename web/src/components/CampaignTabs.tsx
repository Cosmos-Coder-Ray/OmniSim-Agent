import type { CampaignView, EpisodeResult } from "../api/types";

interface CampaignTabsProps {
  campaign: CampaignView | null;
  loading: boolean;
  error: string | null;
  atlas: boolean;
  onOpenEpisode: (episodeId: string) => void;
}

export function CampaignTabs({ campaign, loading, error, atlas, onOpenEpisode }: CampaignTabsProps) {
  if (loading) return <section className="data-panel state-panel" role="status">Loading campaign…</section>;
  if (error) return <section className="data-panel state-panel state-error">{error}</section>;
  if (!campaign?.episodes?.length) return <section className="data-panel state-panel">No campaign episodes are available.</section>;

  const episodes = campaign.episodes;
  const clusters = new Map<string, { count: number; maxRisk: number; episodes: Set<string> }>();
  for (const episode of episodes) {
    for (const failure of episode.failures) {
      const key = `${failure.kind}:${failure.joint === null ? "global" : failure.joint}`;
      const cluster = clusters.get(key) ?? { count: 0, maxRisk: 0, episodes: new Set<string>() };
      cluster.count += 1;
      cluster.maxRisk = Math.max(cluster.maxRisk, episode.risk);
      cluster.episodes.add(episode.episode_id);
      clusters.set(key, cluster);
    }
  }
  return <div className="data-panel"><div className="data-heading"><div className="section-kicker">{atlas ? "FAILURE ATLAS" : "CAMPAIGN"}</div><h2>{atlas ? "Failure Atlas" : "Live Campaign"}</h2><p>{campaign.campaign_id} · {campaign.status} · {episodes.length} episodes</p></div>
    <div className="metric-grid"><article className="metric-card"><span>Episodes</span><strong>{episodes.length}</strong></article><article className="metric-card"><span>Failures</span><strong>{episodes.filter((episode) => episode.failures.length).length}</strong></article><article className="metric-card"><span>Peak risk</span><strong>{Math.max(...episodes.map((episode) => episode.risk)).toFixed(2)}</strong></article></div>
    {atlas && <section className="data-section"><h3>Failure clusters</h3>{clusters.size === 0 ? <p>No failure clusters recorded.</p> : <div className="table-scroll"><table className="data-table"><thead><tr><th>Cluster</th><th>Events</th><th>Episodes</th><th>Peak risk</th></tr></thead><tbody>{[...clusters].map(([key, item]) => <tr key={key}><th scope="row">{key}</th><td>{item.count}</td><td>{item.episodes.size}</td><td>{item.maxRisk.toFixed(2)}</td></tr>)}</tbody></table></div>}</section>}
    <EpisodeTable episodes={episodes} onOpenEpisode={onOpenEpisode} />
  </div>;
}

function EpisodeTable({ episodes, onOpenEpisode }: { episodes: EpisodeResult[]; onOpenEpisode: (id: string) => void }) {
  return <section className="data-section"><h3>Episodes</h3><div className="table-scroll"><table className="data-table"><thead><tr><th>Episode</th><th>Status</th><th>Risk</th><th>Failures</th><th>Max torque</th></tr></thead><tbody>{episodes.map((episode) => <tr key={episode.episode_id}><th scope="row"><button className="table-link" onClick={() => onOpenEpisode(episode.episode_id)}>{episode.episode_id}</button></th><td>{episode.failures.length ? "Failing" : "Nominal"}</td><td>{episode.risk.toFixed(2)}</td><td>{episode.failures.length}</td><td>{(episode.max_tau_ratio * 100).toFixed(0)}%</td></tr>)}</tbody></table></div></section>;
}
