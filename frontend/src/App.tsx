import { useEffect, useState } from "react";
import {
  getHealth,
  getInventory,
  type Health,
  type InventoryReport,
  type DetectorKind,
} from "./api";

const KIND_STYLE: Record<DetectorKind, string> = {
  trained: "bg-emerald-100 text-emerald-800 border-emerald-300",
  pretrained: "bg-sky-100 text-sky-800 border-sky-300",
  heuristic: "bg-amber-100 text-amber-900 border-amber-300",
  simulated: "bg-zinc-200 text-zinc-700 border-zinc-400",
};

function KindBadge({ kind }: { kind: DetectorKind }) {
  return (
    <span
      className={`inline-block rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${KIND_STYLE[kind]}`}
    >
      {kind}
    </span>
  );
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [inv, setInv] = useState<InventoryReport | null>(null);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    Promise.all([getHealth(), getInventory()])
      .then(([h, i]) => {
        setHealth(h);
        setInv(i);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900">
      <header className="border-b bg-white px-6 py-4">
        <h1 className="text-xl font-bold">VoiceShield</h1>
        <p className="text-sm text-zinc-500">
          Voice Integrity &amp; Impersonation Risk Engine (Prototype)
        </p>
      </header>

      <main className="mx-auto max-w-3xl space-y-6 p-6">
        <div className="rounded-lg border bg-blue-50 px-4 py-3 text-sm text-blue-900">
          Authorized channels only: file upload, bundled demo clips, the
          authorized enterprise-stream API, or an explicit push-to-record button.
          VoiceShield never records, taps, or monitors calls or a background
          microphone.
        </div>

        {err && (
          <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
            Backend not reachable: {err}
          </div>
        )}

        {health && inv && (
          <>
            <section className="rounded-lg border bg-white p-4">
              <h2 className="mb-2 font-semibold">Runtime</h2>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                <dt className="text-zinc-500">Version</dt>
                <dd>{health.version}</dd>
                <dt className="text-zinc-500">Device</dt>
                <dd className="font-mono">{inv.device}</dd>
                <dt className="text-zinc-500">Torch / CUDA</dt>
                <dd className="font-mono">
                  {inv.torch.torch} · cuda={inv.torch.cuda_available}
                  {inv.torch.cuda_device !== "-" && ` (${inv.torch.cuda_device})`}
                </dd>
                <dt className="text-zinc-500">Offline</dt>
                <dd>{String(inv.offline)}</dd>
                <dt className="text-zinc-500">Detectors loaded</dt>
                <dd>
                  {health.detectors_loaded}/{health.detectors_total}
                </dd>
                <dt className="text-zinc-500">Tier-A OK</dt>
                <dd className={inv.tier_a_ok ? "text-emerald-700" : "text-red-700"}>
                  {String(inv.tier_a_ok)}
                </dd>
              </dl>
            </section>

            <section className="rounded-lg border bg-white p-4">
              <h2 className="mb-3 font-semibold">Detector inventory</h2>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-zinc-500">
                    <th className="py-1">Detector</th>
                    <th className="py-1">Kind</th>
                    <th className="py-1">Feeds</th>
                    <th className="py-1">Status</th>
                    <th className="py-1">Device</th>
                    <th className="py-1">Warmup</th>
                  </tr>
                </thead>
                <tbody>
                  {inv.detectors.map((d) => (
                    <tr key={d.name} className="border-b last:border-0">
                      <td className="py-1.5 font-mono">{d.name}</td>
                      <td className="py-1.5">
                        <KindBadge kind={d.kind} />
                      </td>
                      <td className="py-1.5 text-zinc-500">{d.feeds}</td>
                      <td className="py-1.5">
                        {d.available ? (
                          <span className="text-emerald-700">loaded</span>
                        ) : (
                          <span className="text-red-700" title={d.load_error}>
                            unavailable
                          </span>
                        )}
                      </td>
                      <td className="py-1.5 font-mono">{d.device}</td>
                      <td className="py-1.5 font-mono text-zinc-500">
                        {d.warmup_ms != null ? `${Math.round(d.warmup_ms)} ms` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-3 text-xs text-zinc-500">
                Phase 0 scaffold. Live Analysis, context, explainability,
                prevention and evaluation screens land in later phases.
              </p>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
