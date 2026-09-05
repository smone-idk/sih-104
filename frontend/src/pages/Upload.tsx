import { useState } from "react";
import { analyzeUpload, type Band } from "../api";
import { Card, Stat } from "../components/common";
import { ContextPanel, TranscriptPanel } from "../components/ContextPanel";
import { ExplainPanel, FindingsPanel } from "../components/Panels";
import { PushToRecord } from "../components/PushToRecord";
import { RiskGauge } from "../components/RiskGauge";

const BANDS = { low_max: 40, high_min: 70 };

const PILL: Record<Band, string> = {
  LOW: "bg-emerald-100 text-emerald-800",
  MEDIUM: "bg-amber-100 text-amber-900",
  HIGH: "bg-red-100 text-red-800",
};

/** Upload / record analysis (§11 screen 7) with the §8 telephony comparison. */
export default function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [useProfile, setUseProfile] = useState(true);
  const [compare, setCompare] = useState(true);
  const [snr, setSnr] = useState<number | null>(null);
  const [res, setRes] = useState<Awaited<ReturnType<typeof analyzeUpload>> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function run() {
    if (!file) return;
    setBusy(true);
    setErr("");
    setRes(null);
    try {
      setRes(await analyzeUpload(file, {
        useProfile, compareTelephony: compare, snrDb: snr,
      }));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  const clean = res?.clean;
  const tele = res?.telephony;
  const floor = clean?.fusion.floors_applied?.[0] ?? null;

  return (
    <div className="space-y-4">
      <Card
        title="Analyse a clip"
        subtitle="Authorized channel: a file you choose, or an explicit recording. Same pipeline as the live stream."
      >
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-wide text-zinc-500">
              audio file
            </span>
            <input
              type="file"
              accept="audio/*"
              className="text-xs file:mr-2 file:rounded file:border-0 file:bg-zinc-200 file:px-2 file:py-1 file:text-xs"
              onChange={(e) => {
                setFile(e.target.files?.[0] ?? null);
                setRes(null);
              }}
            />
          </label>
          <label className="flex items-center gap-2 text-xs text-zinc-700">
            <input type="checkbox" checked={useProfile}
                   onChange={(e) => setUseProfile(e.target.checked)} />
            use enrolled profile
          </label>
          <label className="flex items-center gap-2 text-xs text-zinc-700">
            <input type="checkbox" checked={compare}
                   onChange={(e) => setCompare(e.target.checked)} />
            compare against telephony (8&nbsp;kHz + µ-law)
          </label>
          <label className="flex items-center gap-2 text-xs text-zinc-700">
            <input type="checkbox" checked={snr !== null}
                   onChange={(e) => setSnr(e.target.checked ? 20 : null)} />
            add noise
          </label>
          {snr !== null && (
            <label className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wide text-zinc-500">
                SNR {snr} dB
              </span>
              <input type="range" min={5} max={30} step={5} value={snr}
                     onChange={(e) => setSnr(Number(e.target.value))} />
            </label>
          )}
          <button
            onClick={run}
            disabled={!file || busy}
            className="ml-auto rounded bg-sky-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-sky-700 disabled:opacity-40"
          >
            {busy ? "Analysing…" : "Analyse"}
          </button>
        </div>

        <div className="mt-3">
          <PushToRecord
            label="Push to record"
            disabled={busy}
            onClip={(f) => {
              setFile(f);
              setRes(null);
            }}
          />
        </div>

        {file && (
          <p className="mt-2 text-xs text-zinc-600">
            Ready: <strong>{file.name}</strong> ({Math.round(file.size / 1024)} KB)
          </p>
        )}
        {err && (
          <p className="mt-2 rounded bg-red-50 px-2 py-1.5 text-xs text-red-700">{err}</p>
        )}
        {res && (
          <p className="mt-2 text-[11px] italic text-zinc-500">{res.privacy_note}</p>
        )}
      </Card>

      {clean && (
        <>
          {tele && (
            <Card
              title="Telephony robustness (§8)"
              subtitle="The same clip, before and after a real 8 kHz + G.711 µ-law chain. Shown side by side, not hidden."
            >
              <div className="grid gap-3 sm:grid-cols-3">
                {[
                  { label: "Clean (as supplied)", r: clean },
                  { label: "After telephony chain", r: tele },
                ].map((c) => (
                  <div key={c.label} className="rounded border border-zinc-200 p-3">
                    <p className="text-[10px] uppercase tracking-wide text-zinc-500">
                      {c.label}
                    </p>
                    <div className="mt-1 flex items-baseline gap-2">
                      <span className="text-2xl font-bold tabular-nums">
                        {c.r.score.toFixed(1)}
                      </span>
                      <span className={`rounded px-1.5 py-0.5 text-xs font-semibold ${PILL[c.r.band]}`}>
                        {c.r.band}
                      </span>
                    </div>
                    <p className="mt-1 text-[11px] text-zinc-600">{c.r.voice_verdict}</p>
                    <p className="text-[11px] text-zinc-500">
                      synthetic prob{" "}
                      {c.r.detector_means.synthetic_speech_ssl?.toFixed(3) ?? "—"}
                    </p>
                  </div>
                ))}
                <div className="rounded border border-zinc-200 bg-zinc-50 p-3">
                  <p className="text-[10px] uppercase tracking-wide text-zinc-500">
                    Difference
                  </p>
                  <div className={`text-2xl font-bold tabular-nums ${
                    (res!.delta ?? 0) > 0 ? "text-red-700" : "text-emerald-700"}`}>
                    {(res!.delta ?? 0) > 0 ? "+" : ""}
                    {res!.delta?.toFixed(1)}
                  </div>
                  <p className="mt-1 text-[11px] leading-snug text-zinc-600">
                    Real fraud calls arrive narrowband and compressed. Measuring
                    this gap is worth more than hiding it.
                  </p>
                </div>
              </div>
            </Card>
          )}

          <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
            <Card title="Estimated impersonation risk">
              <RiskGauge
                score={clean.score}
                band={clean.band}
                verdict={clean.voice_verdict}
                bands={BANDS}
                floored={floor}
              />
              <dl className="mt-3 grid grid-cols-2 gap-2 border-t border-zinc-100 pt-2">
                <Stat label="duration" value={`${clean.duration_s}s`} />
                <Stat
                  label="windows"
                  value={`${clean.n_speech_windows}/${clean.n_windows} speech`}
                />
              </dl>
              {clean.warnings.map((w) => (
                <p key={w} className="mt-2 rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-900">
                  {w}
                </p>
              ))}
            </Card>
            <FindingsPanel findings={clean.findings} />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <ContextPanel context={clean.context} />
            <TranscriptPanel
              transcript={clean.transcript}
              quotes={clean.context_quotes}
            />
          </div>

          <ExplainPanel
            components={clean.fusion.components}
            disclaimer={clean.fusion.disclaimer}
            redistributed={clean.fusion.redistributed}
          />
        </>
      )}
    </div>
  );
}
