import type { RefObject } from "react";
import { layoutEventMarkers } from "../replay/geometry";
import type { ReplayEvent } from "../api/types";

interface EventTimelineProps {
  events: ReplayEvent[];
  duration: number;
  cursorRef: RefObject<HTMLDivElement | null>;
  onSeek: (time: number) => void;
}

export function EventTimeline({ events, duration, cursorRef, onSeek }: EventTimelineProps) {
  const markers = layoutEventMarkers(events, duration);
  const rows = Math.max(1, ...markers.map((marker) => marker.row + 1));
  return (
    <section className="timeline-section" aria-label="Replay event timeline">
      <div className="section-heading">
        <div>
          <div className="section-kicker">EVENTS</div>
          <h3>Failure timeline</h3>
        </div>
        <span>{events.length} markers</span>
      </div>
      <div className="timeline-rail" style={{ minHeight: `${Math.max(40, rows * 24 + 16)}px` }}>
        <div className="timeline-track" />
        <div className="timeline-cursor" ref={cursorRef} />
        {markers.map((marker) => (
          <button
            key={`${marker.kind}-${marker.t}-${marker.joint ?? "global"}`}
            className="timeline-marker"
            style={{ left: `${marker.leftPercent}%`, top: `${8 + marker.row * 24}px` }}
            type="button"
            aria-label={`Seek to ${marker.kind.replaceAll("_", " ")} at ${marker.t.toFixed(2)} seconds`}
            title={`${marker.kind.replaceAll("_", " ")} · ${marker.t.toFixed(2)} s`}
            onClick={() => onSeek(marker.t)}
          >
            <span aria-hidden="true" />
          </button>
        ))}
      </div>
      {events.length === 0 && <p className="timeline-empty">No failure events in this replay.</p>}
    </section>
  );
}
