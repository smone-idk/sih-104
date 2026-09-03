import type { Band, DetectorKind } from "../api";

export const KIND_STYLE: Record<DetectorKind, string> = {
  trained: "bg-emerald-100 text-emerald-800 border-emerald-300",
  pretrained: "bg-sky-100 text-sky-800 border-sky-300",
  heuristic: "bg-amber-100 text-amber-900 border-amber-300",
  simulated: "bg-zinc-200 text-zinc-700 border-zinc-400",
};

export function KindBadge({ kind }: { kind: DetectorKind | "" }) {
  if (!kind) return null;
  return (
    <span
      className={`inline-block rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${KIND_STYLE[kind as DetectorKind]}`}
    >
      {kind}
    </span>
  );
}

export const BAND_STYLE: Record<Band, { text: string; bg: string; ring: string }> = {
  LOW: { text: "text-emerald-700", bg: "bg-emerald-50", ring: "stroke-emerald-500" },
  MEDIUM: { text: "text-amber-700", bg: "bg-amber-50", ring: "stroke-amber-500" },
  HIGH: { text: "text-red-700", bg: "bg-red-50", ring: "stroke-red-500" },
};

export function Card({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border border-zinc-200 bg-white p-4 shadow-sm ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex items-start justify-between gap-2">
          <div>
            {title && <h2 className="text-sm font-semibold text-zinc-800">{title}</h2>}
            {subtitle && <p className="text-xs text-zinc-500">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

/** Small label/value row used throughout the session header. */
export function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</dt>
      <dd className="text-sm font-medium text-zinc-800">{value}</dd>
    </div>
  );
}
