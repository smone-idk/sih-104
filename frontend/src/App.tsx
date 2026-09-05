import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { getHealth, getInventory, type Health, type InventoryReport } from "./api";
import { KindBadge } from "./components/common";
import LiveAnalysis from "./pages/LiveAnalysis";
import Approvals from "./pages/Approvals";
import Incidents from "./pages/Incidents";
import Profiles from "./pages/Profiles";
import Upload from "./pages/Upload";

function SystemPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [inv, setInv] = useState<InventoryReport | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    Promise.all([getHealth(), getInventory()])
      .then(([h, i]) => {
        setHealth(h);
        setInv(i);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p className="rounded bg-red-50 p-3 text-sm text-red-700">{err}</p>;
  if (!health || !inv) return <p className="text-sm text-zinc-500">Loading…</p>;

  return (
    <div className="space-y-4">
      <section className="rounded-lg border border-zinc-200 bg-white p-4 shadow-sm">
        <h2 className="mb-2 text-sm font-semibold text-zinc-800">Runtime</h2>
        <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
          <div><dt className="text-[10px] uppercase text-zinc-500">device</dt><dd>{health.device}</dd></div>
          <div><dt className="text-[10px] uppercase text-zinc-500">offline</dt><dd>{String(health.offline)}</dd></div>
          <div><dt className="text-[10px] uppercase text-zinc-500">detectors</dt><dd>{health.detectors_loaded}/{health.detectors_total}</dd></div>
          <div><dt className="text-[10px] uppercase text-zinc-500">torch</dt><dd>{inv.torch.torch}</dd></div>
        </dl>
      </section>
      <section className="rounded-lg border border-zinc-200 bg-white p-4 shadow-sm">
        <h2 className="mb-2 text-sm font-semibold text-zinc-800">Detector inventory</h2>
        <table className="w-full text-xs">
          <thead className="text-[10px] uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="py-1 text-left font-medium">detector</th>
              <th className="text-left font-medium">kind</th>
              <th className="text-left font-medium">feeds</th>
              <th className="text-right font-medium">weight</th>
              <th className="text-left font-medium">device</th>
              <th className="text-left font-medium">status</th>
            </tr>
          </thead>
          <tbody>
            {inv.detectors.map((d) => (
              <tr key={d.name} className="border-t border-zinc-100">
                <td className="py-1.5 font-medium text-zinc-800">{d.name.replace(/_/g, " ")}</td>
                <td><KindBadge kind={d.kind} /></td>
                <td className="text-zinc-600">{d.feeds ?? "—"}</td>
                <td className="text-right tabular-nums">{d.fusion_weight.toFixed(2)}</td>
                <td className="text-zinc-600">{d.device}</td>
                <td className={d.available ? "text-emerald-700" : "text-red-700"}>
                  {d.available ? "loaded" : d.load_error || "unavailable"}
                  {d.note && <div className="text-[10px] leading-snug text-amber-800">{d.note}</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

const NAV = [
  { to: "/live", label: "Live Analysis" },
  { to: "/upload", label: "Upload" },
  { to: "/approvals", label: "Approvals" },
  { to: "/incidents", label: "Incidents" },
  { to: "/profiles", label: "Voice Profiles" },
  { to: "/system", label: "System" },
];

export default function App() {
  return (
    <div className="min-h-screen bg-zinc-50">
      <header className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <div>
            <h1 className="text-base font-bold tracking-tight text-zinc-900">VoiceShield</h1>
            <p className="text-[11px] text-zinc-500">
              Voice Integrity &amp; Impersonation Risk Engine (Prototype)
            </p>
          </div>
          <nav className="flex gap-1">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) =>
                  `rounded px-3 py-1.5 text-sm font-medium ${
                    isActive ? "bg-sky-100 text-sky-800" : "text-zinc-600 hover:bg-zinc-100"
                  }`
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <p className="ml-auto max-w-md text-right text-[10px] leading-snug text-zinc-500">
            Analyses only audio supplied through an authorized channel. Never taps
            phone or messaging calls.
          </p>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-5">
        <Routes>
          <Route path="/" element={<Navigate to="/live" replace />} />
          <Route path="/live" element={<LiveAnalysis />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/profiles" element={<Profiles />} />
          <Route path="/approvals" element={<Approvals />} />
          <Route path="/incidents" element={<Incidents />} />
          <Route path="/system" element={<SystemPage />} />
        </Routes>
      </main>
    </div>
  );
}
