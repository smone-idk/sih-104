import {
  CartesianGrid,
  Legend,
  Line,
  ComposedChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { WindowMsg } from "../api";

/**
 * Streaming risk chart: raw per-window dots AND the EMA line, together, so the
 * smoothing is visibly honest (§4). Silence windows carry no score and are
 * simply absent — they are not interpolated over.
 */
export function StreamChart({
  windows,
  bands,
  alpha,
}: {
  windows: WindowMsg[];
  bands: { low_max: number; high_min: number };
  alpha: number;
}) {
  const data = windows.map((w) => ({
    t: Number(w.t_start.toFixed(1)),
    raw: w.raw_score,
    ema: w.ema_score,
    speech: w.is_speech,
  }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-zinc-200" />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={{ fontSize: 11 }}
            label={{ value: "seconds", position: "insideBottomRight", fontSize: 10, dy: 6 }}
          />
          <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            formatter={(v, n) => [
              typeof v === "number" ? v.toFixed(1) : "—",
              String(n),
            ]}
            labelFormatter={(l) => `t = ${l}s`}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <ReferenceLine y={bands.low_max} strokeDasharray="4 4" className="stroke-amber-400" />
          <ReferenceLine y={bands.high_min} strokeDasharray="4 4" className="stroke-red-400" />
          <Scatter name="raw (per window)" dataKey="raw" fill="#a1a1aa" />
          <Line
            name={`EMA α=${alpha}`}
            type="monotone"
            dataKey="ema"
            stroke="#0284c7"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/** RMS envelope + a scrolling spectrogram, both from streamed measurements. */
export function WaveformSpectrogram({ windows }: { windows: WindowMsg[] }) {
  const recent = windows.slice(-60);
  const maxRms = Math.max(0.01, ...recent.map((w) => w.rms_energy));

  // dBFS -> 0..1 for colour. Fixed -85..-35 dBFS window: covers speech energy
  // while staying comparable across frames (no per-frame auto-gain, which would
  // make a quiet window look as loud as a shout).
  const norm = (db: number) => Math.max(0, Math.min(1, (db + 85) / 50));

  return (
    <div className="space-y-3">
      <div>
        <p className="mb-1 text-[10px] uppercase tracking-wide text-zinc-500">
          RMS envelope (one bar per 1 s hop)
        </p>
        <div className="flex h-14 items-end gap-[2px]">
          {recent.map((w) => (
            <div
              key={w.index}
              title={`t=${w.t_start}s rms=${w.rms_energy}`}
              className={`flex-1 rounded-sm ${w.is_speech ? "bg-sky-500" : "bg-zinc-300"}`}
              style={{ height: `${Math.max(3, (w.rms_energy / maxRms) * 100)}%` }}
            />
          ))}
          {recent.length === 0 && (
            <p className="text-xs text-zinc-400">waiting for audio…</p>
          )}
        </div>
      </div>
      <div>
        <p className="mb-1 text-[10px] uppercase tracking-wide text-zinc-500">
          Spectrogram — 24 log bands, 80 Hz–8 kHz, −85…−35 dBFS (FFT per window)
        </p>
        <div className="flex h-24 gap-[2px] overflow-hidden rounded bg-zinc-900 p-[2px]">
          {recent.map((w) => (
            <div key={w.index} className="flex flex-1 flex-col-reverse">
              {w.spectrum.map((db, i) => {
                const v = norm(db);
                return (
                  <div
                    key={i}
                    className="flex-1"
                    style={{
                      backgroundColor: `hsl(${220 - v * 200}, 85%, ${8 + v * 55}%)`,
                    }}
                  />
                );
              })}
            </div>
          ))}
          {recent.length === 0 && (
            <p className="self-center px-2 text-xs text-zinc-500">waiting for audio…</p>
          )}
        </div>
      </div>
    </div>
  );
}
