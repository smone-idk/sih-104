import type { Band } from "../api";
import { BAND_STYLE } from "./common";

/**
 * Big risk gauge. The number is whatever the fusion layer computed — this
 * component never derives, clamps for effect, or animates toward a target.
 */
export function RiskGauge({
  score,
  band,
  verdict,
  bands,
  live,
  floored,
}: {
  score: number | null;
  band: Band | null;
  verdict?: string;
  bands: { low_max: number; high_min: number };
  live?: boolean;
  /** set when a policy rule raised the band above what the score implies */
  floored?: { from_band: string; description: string } | null;
}) {
  const R = 70;
  const C = Math.PI * R; // semicircle length
  const pct = score == null ? 0 : Math.max(0, Math.min(100, score)) / 100;
  const style = band ? BAND_STYLE[band] : BAND_STYLE.LOW;

  const tick = (v: number) => {
    const a = Math.PI * (1 - v / 100);
    return { x: 90 + R * Math.cos(a), y: 84 - R * Math.sin(a) };
  };
  const t1 = tick(bands.low_max);
  const t2 = tick(bands.high_min);

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 180 100" className="w-full max-w-[260px]">
        <path
          d={`M 20 84 A ${R} ${R} 0 0 1 160 84`}
          className="fill-none stroke-zinc-200"
          strokeWidth="12"
          strokeLinecap="round"
        />
        <path
          d={`M 20 84 A ${R} ${R} 0 0 1 160 84`}
          className={`fill-none ${style.ring} transition-[stroke-dashoffset] duration-500`}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={C}
          strokeDashoffset={C * (1 - pct)}
        />
        {/* band boundaries — drawn from the server's thresholds, not hardcoded */}
        <line x1={t1.x} y1={t1.y} x2={t1.x} y2={t1.y - 7} className="stroke-zinc-400" strokeWidth="1.5" />
        <line x1={t2.x} y1={t2.y} x2={t2.x} y2={t2.y - 7} className="stroke-zinc-400" strokeWidth="1.5" />
        <text x="90" y="72" textAnchor="middle" className="fill-zinc-900 text-[30px] font-bold">
          {score == null ? "—" : score.toFixed(1)}
        </text>
        <text x="90" y="92" textAnchor="middle" className="fill-zinc-500 text-[9px]">
          estimated impersonation risk
        </text>
      </svg>
      <div className="-mt-1 flex items-center gap-2">
        <span className={`rounded px-2 py-0.5 text-sm font-bold ${style.bg} ${style.text}`}>
          {band ?? "—"}
        </span>
        {live && (
          <span className="flex items-center gap-1 text-[10px] font-medium text-red-600">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-600" /> LIVE
          </span>
        )}
      </div>
      {verdict && (
        <p className="mt-2 text-center text-xs font-semibold tracking-wide text-zinc-700">
          {verdict.replace(/_/g, " ")}
        </p>
      )}
      {floored && (
        <p className="mt-2 rounded border border-red-300 bg-red-50 px-2 py-1.5 text-[10px] leading-snug text-red-900">
          <span className="font-semibold">
            Band raised from {floored.from_band} by policy rule.
          </span>{" "}
          {floored.description}
        </p>
      )}
    </div>
  );
}
