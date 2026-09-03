import { useCallback, useEffect, useRef, useState } from "react";
import {
  getScenarios,
  openStream,
  type FinalMsg,
  type Scenario,
  type SessionMsg,
  type StreamMsg,
  type WindowMsg,
} from "../api";
import { Card, Stat } from "../components/common";
import { RiskGauge } from "../components/RiskGauge";
import { StreamChart, WaveformSpectrogram } from "../components/StreamChart";
import {
  DetectorCard,
  ExplainPanel,
  FindingsPanel,
  LatencyStrip,
} from "../components/Panels";

const TIER_STYLE = {
  genuine: "bg-emerald-100 text-emerald-800",
  synthetic: "bg-amber-100 text-amber-900",
  cloned: "bg-red-100 text-red-800",
} as const;

const DEFAULT_BANDS = { low_max: 40, high_min: 70 };

export default function LiveAnalysis() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [telephony, setTelephony] = useState(false);
  const [useProfile, setUseProfile] = useState(true);
  const [realtime, setRealtime] = useState(true);

  const [session, setSession] = useState<SessionMsg | null>(null);
  const [windows, setWindows] = useState<WindowMsg[]>([]);
  const [final, setFinal] = useState<FinalMsg | null>(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState("");
  const closer = useRef<null | (() => void)>(null);

  useEffect(() => {
    getScenarios()
      .then((d) => {
        setScenarios(d.scenarios);
        const first = d.scenarios.find((s) => s.available);
        if (first) setSelected(first.id);
      })
      .catch((e) => setErr(String(e)));
    return () => closer.current?.();
  }, []);

  const onMsg = useCallback((m: StreamMsg) => {
    if (m.type === "session") setSession(m);
    else if (m.type === "window") setWindows((w) => [...w, m]);
    else if (m.type === "final") {
      setFinal(m);
      setRunning(false);
    } else if (m.type === "error") {
      setErr(m.error);
      setRunning(false);
    }
  }, []);

  function start() {
    closer.current?.();
    setSession(null);
    setWindows([]);
    setFinal(null);
    setErr("");
    setRunning(true);
    closer.current = openStream(
      { scenario: selected, telephony, use_profile: useProfile, realtime },
      onMsg,
      () => setRunning(false),
    );
  }

  function stop() {
    closer.current?.();
    setRunning(false);
  }

  const meta = scenarios.find((s) => s.id === selected);
  const bands = session?.bands ?? DEFAULT_BANDS;
  const live = windows.filter((w) => w.ema_score != null).at(-1);
  const shownScore = final ? final.score : live?.ema_score ?? null;
  const floor = final?.fusion.floors_applied?.[0] ?? null;
  const shownBand = final
    ? final.band
    : shownScore == null
      ? null
      : shownScore < bands.low_max
        ? "LOW"
        : shownScore > bands.high_min
          ? "HIGH"
          : "MEDIUM";
  const latestWindow = windows.filter((w) => w.is_speech).at(-1);

  return (
    <div className="space-y-4">
      {/* --- controls ------------------------------------------------ */}
      <Card title="Session" subtitle="Authorized channel: bundled demo clip (simulated transport, real analysis)">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-wide text-zinc-500">Scenario</span>
            <select
              className="min-w-[22rem] rounded border border-zinc-300 px-2 py-1.5 text-sm"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              disabled={running}
            >
              {scenarios.map((s) => (
                <option key={s.id} value={s.id} disabled={!s.available}>
                  {s.title}
                  {s.available ? "" : "  (asset missing)"}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-xs text-zinc-700">
            <input type="checkbox" checked={useProfile} onChange={(e) => setUseProfile(e.target.checked)} disabled={running} />
            use enrolled profile
          </label>
          <label className="flex items-center gap-2 text-xs text-zinc-700">
            <input type="checkbox" checked={telephony} onChange={(e) => setTelephony(e.target.checked)} disabled={running} />
            telephony 8 kHz + µ-law
          </label>
          <label className="flex items-center gap-2 text-xs text-zinc-700">
            <input type="checkbox" checked={realtime} onChange={(e) => setRealtime(e.target.checked)} disabled={running} />
            real-time pacing
          </label>
          <div className="ml-auto flex gap-2">
            <button
              onClick={start}
              disabled={running || !selected}
              className="rounded bg-sky-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-sky-700 disabled:opacity-40"
            >
              {running ? "Streaming…" : "Start simulation"}
            </button>
            <button
              onClick={stop}
              disabled={!running}
              className="rounded border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-50 disabled:opacity-40"
            >
              Stop
            </button>
          </div>
        </div>

        {meta && (
          <div className="mt-3 rounded border border-zinc-200 bg-zinc-50 p-2.5">
            <div className="flex items-center gap-2">
              <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${TIER_STYLE[meta.tier]}`}>
                {meta.tier}
              </span>
              <span className="text-xs text-zinc-600">
                caller claims to be <strong>{meta.caller_claims}</strong>
                {meta.duration_s != null && ` · ${meta.duration_s}s`}
              </span>
            </div>
            <p className="mt-1 text-xs text-zinc-600">{meta.description}</p>
            <p className="mt-1 text-[11px] text-zinc-500">
              <span className="font-medium">Expected behaviour:</span> {meta.expected}
            </p>
          </div>
        )}

        {session && (
          <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-5">
            <Stat label="session" value={<code className="text-[11px]">{session.session_id.slice(0, 8)}</code>} />
            <Stat label="channel" value={<span className="text-xs">{session.channel}</span>} />
            <Stat label="enrolled profile" value={<span className="text-xs">{session.profile_name ?? "none"}</span>} />
            <Stat label="device" value={session.device} />
            <Stat label="window / hop" value={`${session.window_seconds}s / ${session.hop_seconds}s`} />
          </dl>
        )}
        {err && <p className="mt-2 rounded bg-red-50 px-2 py-1 text-xs text-red-700">{err}</p>}
      </Card>

      {/* --- gauge + chart ------------------------------------------ */}
      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <Card title="Estimated impersonation risk">
          <RiskGauge
            score={shownScore}
            band={shownBand}
            verdict={final?.voice_verdict}
            bands={bands}
            live={running}
            floored={floor}
          />
          {final && (
            <dl className="mt-3 grid grid-cols-2 gap-2 border-t border-zinc-100 pt-2">
              <Stat label="windows" value={`${final.n_speech_windows}/${final.n_windows} speech`} />
              <Stat label="duration" value={`${final.duration_s}s`} />
            </dl>
          )}
          {final?.warnings.map((w) => (
            <p key={w} className="mt-2 rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-900">{w}</p>
          ))}
        </Card>

        <Card
          title="Streaming risk"
          subtitle="Raw per-window scores and the EMA line are shown together, so the smoothing is visible"
        >
          <StreamChart windows={windows} bands={bands} alpha={session?.ema_alpha ?? 0.3} />
        </Card>
      </div>

      <LatencyStrip windows={windows} device={session?.device ?? "cpu"} />

      {/* --- detectors ---------------------------------------------- */}
      <Card title="Detectors" subtitle="Badges and weights are read from the registry — no detector may misreport its kind">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {(session?.detectors ?? []).map((d) => (
            <DetectorCard key={d.name} info={d} latest={latestWindow?.detectors[d.name]} />
          ))}
          {!session && <p className="text-xs text-zinc-500">Start a session to see live detector output.</p>}
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Signal" subtitle="Measured from the stream, not decorative">
          <WaveformSpectrogram windows={windows} />
        </Card>
        {final ? (
          <FindingsPanel findings={final.findings} />
        ) : (
          <Card title="Findings" subtitle="Populated when the session completes">
            <p className="text-xs text-zinc-500">
              {running ? "Analysing…" : "No session yet."}
            </p>
          </Card>
        )}
      </div>

      {final && (
        <ExplainPanel
          components={final.fusion.components}
          disclaimer={final.fusion.disclaimer}
          redistributed={final.fusion.redistributed}
        />
      )}
    </div>
  );
}
