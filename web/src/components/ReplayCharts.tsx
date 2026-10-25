import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Replay } from "../api/types";

interface ReplayChartsProps {
  replay: Replay;
  torqueCursorRef: RefObject<HTMLDivElement | null>;
  velocityCursorRef: RefObject<HTMLDivElement | null>;
}

const jointColors = ["#71d6bd", "#64b8d2", "#8e9bea", "#c98fe0", "#e6a568", "#e17879", "#a9c76d"];

function ChartPanel({
  title,
  replay,
  dataKey,
  cursorRef,
}: {
  title: string;
  replay: Replay;
  dataKey: "tau_ratio" | "vel_ratio";
  cursorRef: RefObject<HTMLDivElement | null>;
}) {
  return (
    <section className="chart-panel">
      <div className="section-heading">
        <div>
          <div className="section-kicker">JOINT TELEMETRY</div>
          <h3>{title}</h3>
        </div>
        <span>J1–J7</span>
      </div>
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={replay.frames} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
            <CartesianGrid stroke="#27323e" strokeDasharray="3 5" />
            <XAxis
              dataKey="t"
              type="number"
              domain={[0, "dataMax"]}
              tickFormatter={(value: number) => `${value}s`}
              tick={{ fill: "#9ba9b7", fontSize: 10 }}
              stroke="#46515d"
            />
            <YAxis
              domain={[0, 1.2]}
              tick={{ fill: "#9ba9b7", fontSize: 10 }}
              stroke="#46515d"
              width={36}
            />
            <Tooltip
              labelFormatter={(value) => `${Number(value).toFixed(2)} s`}
              contentStyle={{ border: "1px solid #27323e", background: "#111820", color: "#e8edf2" }}
            />
            {jointColors.map((stroke, index) => (
              <Line
                key={stroke}
                type="monotone"
                dataKey={(frame: Replay["frames"][number]) => frame[dataKey][index]}
                stroke={stroke}
                strokeWidth={1.6}
                dot={false}
                activeDot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
        <div className="chart-cursor" ref={cursorRef} />
      </div>
    </section>
  );
}

export function ReplayCharts({ replay, torqueCursorRef, velocityCursorRef }: ReplayChartsProps) {
  return (
    <div className="replay-charts">
      <ChartPanel title="Torque ratio" replay={replay} dataKey="tau_ratio" cursorRef={torqueCursorRef} />
      <ChartPanel title="Velocity ratio" replay={replay} dataKey="vel_ratio" cursorRef={velocityCursorRef} />
    </div>
  );
}
import type { RefObject } from "react";
