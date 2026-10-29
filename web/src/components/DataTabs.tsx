import { useEffect, useState } from "react";
import { dataClient } from "../api/client";
import type { DataResult, EvidenceClaim } from "../api/types";

const loadBenchmark = () => dataClient.getBenchmark();

function useData<T>(load: (key: string) => Promise<DataResult<T>>, key: string) {
  const [response, setResponse] = useState<{ key: string; result: DataResult<T> } | null>(null);
  useEffect(() => {
    let active = true;
    void load(key).then((result) => { if (active) setResponse({ key, result }); });
    return () => { active = false; };
  }, [key, load]);
  return response?.key === key ? response.result : null;
}

function DataState({ result }: { result: DataResult<unknown> | null }) {
  if (result === null || result.kind === "pending") return <p className="data-state" role="status">Loading data…</p>;
  if (result.kind === "unavailable") return <p className="data-state">{result.reason}</p>;
  if (result.kind === "error") return <p className="data-state state-error">{result.message}</p>;
  return null;
}

function EvidenceRows({ title, rows }: { title: string; rows: EvidenceClaim[] }) {
  if (!rows.length) return null;
  return <section className="data-section"><h3>{title}</h3>{rows.map((row) => (
    <article className="evidence-row" key={`${title}-${row.claim}`}>
      <div className="data-row-heading"><strong>{row.claim}</strong><span className={`verified-badge${row.verified ? " is-verified" : ""}`}>{row.verified ? "Verified" : "Unverified"}</span></div>
      <p className="evidence-value">{row.value}{row.unit ? ` ${row.unit}` : ""}</p>
      <blockquote>{row.quote}</blockquote><a href={row.source_url} target="_blank" rel="noreferrer">Source</a>
    </article>
  ))}</section>;
}

export function EvidenceTab({ campaignId }: { campaignId: string }) {
  const result = useData(dataClient.getEvidence, campaignId);
  const data = result?.kind === "ok" ? result.data : null;
  return <div className="data-panel"><div className="data-heading"><div className="section-kicker">HARDWARE RECORD</div><h2>Evidence</h2><p>{data?.robot ?? "Manufacturer claims and model comparison"}</p></div>
    <DataState result={result} />{data && <>
      <EvidenceRows title="Joint torque limits" rows={data.joint_torque_limits_nm} />
      <EvidenceRows title="Joint velocity limits" rows={data.joint_velocity_limits} />
      <EvidenceRows title="Joint position limits" rows={data.joint_position_limits} />
      <EvidenceRows title="Error taxonomy" rows={data.error_taxonomy} />
      <section className="data-section"><h3>URDF mismatches</h3>{data.mismatches?.length ? data.mismatches.map((item) => <p key={`${item.joint}-${item.field}`}>Joint {item.joint + 1} {item.field}: URDF {item.urdf}, documented {item.doc} · <a href={item.source_url} target="_blank" rel="noreferrer">Source</a></p>) : <p>No URDF mismatches recorded.</p>}</section>
    </>}</div>;
}

export function EnvelopeTab({ campaignId }: { campaignId: string }) {
  const result = useData(dataClient.getEnvelope, campaignId);
  const data = result?.kind === "ok" ? result.data : null;
  return <div className="data-panel"><div className="data-heading"><div className="section-kicker">VERIFIED LIMITS</div><h2>Safe Envelope</h2><p>{data?.robot ?? "Operating limits and verification"}</p></div>
    <DataState result={result} />{data && <>
      <div className="metric-grid"><Metric label="Maximum payload" value={`${data.limits.max_payload_kg} kg`} /><Metric label="Maximum latency" value={`${data.limits.max_latency_ms} ms`} /><Metric label="Torque utilization cap" value={`${(data.limits.torque_utilization_cap * 100).toFixed(0)}%`} /></div>
      <section className="data-section"><h3>Verification</h3>{data.verification ? <p>{(data.verification.fail_rate_before * 100).toFixed(1)}% before → {(data.verification.fail_rate_after * 100).toFixed(1)}% after · {data.verification.n_scenarios} scenarios</p> : <p>Verification results are unavailable.</p>}</section>
      <section className="data-section"><h3>Artifacts</h3><div className="artifact-links">{["safe_envelope.json", "ros2_parameters.yaml", "test_safe_envelope.py", "index.html"].map((name) => <a key={name} href={`/api/exports/${encodeURIComponent(campaignId)}/${name}`} download={name}>{name}</a>)}</div></section>
    </>}</div>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <article className="metric-card"><span>{label}</span><strong>{value}</strong></article>;
}

export function BenchmarkTab() {
  const result = useData(loadBenchmark, "benchmark");
  const data = result?.kind === "ok" ? result.data : null;
  return <div className="data-panel"><div className="data-heading"><div className="section-kicker">STRATEGY RESULTS</div><h2>Benchmarks</h2><p>{data ? `${data.budget} simulations · ${data.seed_count} seeds` : "Strategy comparison"}</p></div><DataState result={result} />
    {data && <div className="table-scroll"><table className="data-table"><thead><tr><th>Strategy</th><th>Sims to first failure</th><th>Unique failure kinds</th><th>Best risk</th></tr></thead><tbody>{data.strategies.map((item) => <tr key={item.strategy}><th scope="row">{item.strategy}</th><td>{item.sims_to_first_failure.mean.toFixed(2)}</td><td>{item.unique_failure_kinds.mean.toFixed(2)}</td><td>{item.best_risk.mean.toFixed(3)}</td></tr>)}</tbody></table></div>}
  </div>;
}
