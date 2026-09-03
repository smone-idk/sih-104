import type { Component, DetectorInfo, DetectorResult, Finding, WindowMsg } from "../api";
import { Card, KindBadge } from "./common";

const COMPONENT_LABEL: Record<string, string> = {
  voice_authenticity: "Voice authenticity",
  speaker_consistency: "Speaker consistency",
  prosody_anomaly: "Prosody anomaly",
  caller_trust: "Caller trust",
  transaction_context: "Transaction context",
  behavioural_risk: "Behavioural risk",
};

/** One detector card. Badge and weight come straight from the registry. */
export function DetectorCard({
  info,
  latest,
}: {
  info: DetectorInfo;
  latest?: DetectorResult;
}) {
  const score = latest?.score ?? null;
  const unavailable = !info.available || latest?.available === false;
  return (
    <div
      className={`rounded-lg border p-3 ${
        unavailable ? "border-zinc-200 bg-zinc-50" : "border-zinc-200 bg-white"
      }`}
    >
      <div className="mb-1 flex items-start justify-between gap-2">
        <h3 className="text-xs font-semibold text-zinc-800">
          {info.name.replace(/_/g, " ")}
        </h3>
        <KindBadge kind={info.kind} />
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold tabular-nums text-zinc-900">
          {score == null ? "—" : score.toFixed(3)}
        </span>
        <span className="text-[10px] text-zinc-500">
          {info.contributes ? `weight ${info.fusion_weight.toFixed(2)}` : "weight 0.00"}
        </span>
      </div>
      {!info.contributes && (
        <p className="mt-1 rounded bg-amber-50 px-1.5 py-1 text-[10px] leading-snug text-amber-900">
          {info.note || "zero fusion weight — displayed as a measured baseline"}
        </p>
      )}
      {unavailable && (
        <p className="mt-1 text-[10px] leading-snug text-zinc-500">
          {latest?.note || info.load_error || "detector unavailable"}
        </p>
      )}
      {info.licence && (
        <p className="mt-1 truncate text-[10px] text-zinc-400" title={info.training_data}>
          {info.licence}
        </p>
      )}
    </div>
  );
}

export function FindingsPanel({ findings }: { findings: Finding[] }) {
  const style = {
    critical: "border-red-300 bg-red-50 text-red-900",
    warning: "border-amber-300 bg-amber-50 text-amber-900",
    info: "border-sky-300 bg-sky-50 text-sky-900",
  } as const;
  return (
    <Card
      title="Findings"
      subtitle="What fired — distinct from how much it moved the score"
    >
      {findings.length === 0 ? (
        <p className="text-xs text-zinc-500">No findings yet.</p>
      ) : (
        <ul className="space-y-2">
          {findings.map((f) => (
            <li key={f.code} className={`rounded border px-2 py-1.5 ${style[f.severity]}`}>
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold">{f.code.replace(/_/g, " ")}</span>
                <KindBadge kind={f.kind} />
              </div>
              <p className="mt-0.5 text-[11px] leading-snug">{f.label}</p>
              {Object.keys(f.evidence).length > 0 && (
                <dl className="mt-1 flex flex-wrap gap-x-3 text-[10px] opacity-80">
                  {Object.entries(f.evidence)
                    .filter(([, v]) => typeof v === "number" || typeof v === "string")
                    .slice(0, 4)
                    .map(([k, v]) => (
                      <span key={k}>
                        <span className="font-medium">{k}</span>:{" "}
                        {typeof v === "number" ? v.toFixed(3) : String(v).slice(0, 60)}
                      </span>
                    ))}
                </dl>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export function ExplainPanel({
  components,
  disclaimer,
  redistributed,
}: {
  components: Component[];
  disclaimer: string;
  redistributed: boolean;
}) {
  const max = Math.max(1, ...components.map((c) => c.contribution_points));
  return (
    <Card
      title="Explainability"
      subtitle="Each component's raw value, weight and contribution in points"
    >
      {redistributed && (
        <p className="mb-2 rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-900">
          Some layers are unavailable — their weight was redistributed across the
          rest, not replaced with a substitute value.
        </p>
      )}
      <table className="w-full text-xs">
        <thead className="text-[10px] uppercase tracking-wide text-zinc-500">
          <tr>
            <th className="py-1 text-left font-medium">component</th>
            <th className="text-right font-medium">value</th>
            <th className="text-right font-medium">wt</th>
            <th className="text-right font-medium">eff</th>
            <th className="text-right font-medium">points</th>
          </tr>
        </thead>
        <tbody>
          {components.map((c) => (
            <tr key={c.name} className="border-t border-zinc-100">
              <td className="py-1">
                <div className="font-medium text-zinc-800">
                  {COMPONENT_LABEL[c.name] ?? c.name}
                </div>
                {!c.available && (
                  <div className="text-[10px] leading-snug text-zinc-500">{c.note}</div>
                )}
                {c.available && (
                  <div className="mt-0.5 h-1 rounded bg-zinc-100">
                    <div
                      className="h-1 rounded bg-sky-500"
                      style={{ width: `${(c.contribution_points / max) * 100}%` }}
                    />
                  </div>
                )}
              </td>
              <td className="text-right tabular-nums">
                {c.raw_value == null ? "—" : c.raw_value.toFixed(3)}
              </td>
              <td className="text-right tabular-nums text-zinc-500">{c.weight.toFixed(2)}</td>
              <td className="text-right tabular-nums text-zinc-500">
                {c.effective_weight.toFixed(3)}
              </td>
              <td className="text-right font-semibold tabular-nums">
                {c.contribution_points.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[10px] italic leading-snug text-zinc-500">{disclaimer}</p>
    </Card>
  );
}

export function LatencyStrip({
  windows,
  device,
}: {
  windows: WindowMsg[];
  device: string;
}) {
  const recent = windows.filter((w) => w.is_speech).slice(-10);
  const names = Array.from(new Set(recent.flatMap((w) => Object.keys(w.latency_ms))));
  const mean = (n: string) => {
    const vals = recent.map((w) => w.latency_ms[n]).filter((v) => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };
  const total = names.reduce((a, n) => a + (mean(n) ?? 0), 0);
  const budget = device === "cuda" ? 200 : 500;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-[11px]">
      <span className="font-semibold text-zinc-700">Latency / 4 s window</span>
      {names.map((n) => (
        <span key={n} className="text-zinc-600">
          {n.replace(/_/g, " ")}:{" "}
          <span className="font-medium tabular-nums text-zinc-900">
            {(mean(n) ?? 0).toFixed(0)} ms
          </span>
        </span>
      ))}
      <span
        className={`ml-auto rounded px-1.5 py-0.5 font-semibold tabular-nums ${
          total <= budget ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"
        }`}
      >
        total {total.toFixed(0)} ms / {budget} ms budget ({device})
      </span>
    </div>
  );
}
