import { useCallback, useEffect, useRef, useState } from "react";
import {
  approve,
  getApprovals,
  getVerifications,
  issueChallenge,
  resetApproval,
  respondToChallenge,
  simulatedVerification,
  type Approval,
  type Verification,
} from "../api";
import { Card, Stat } from "../components/common";

const SIMULATED = [
  { method: "callback", label: "Registered callback" },
  { method: "mfa", label: "Push MFA" },
  { method: "supervisor", label: "Supervisor approval" },
];

function inr(v: number) {
  return `₹${v.toLocaleString("en-IN")}`;
}

const BAND_PILL: Record<string, string> = {
  LOW: "bg-emerald-100 text-emerald-800",
  MEDIUM: "bg-amber-100 text-amber-900",
  HIGH: "bg-red-100 text-red-800",
};

export default function Approvals() {
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [vers, setVers] = useState<Verification[]>([]);
  const [challenge, setChallenge] = useState<any>(null);
  const [lastResponse, setLastResponse] = useState<any>(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const d = await getApprovals();
      setApprovals(d.approvals);
      const sid = d.approvals[0]?.session_id ?? undefined;
      setVers((await getVerifications(sid)).verifications);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const a = approvals[0];

  async function tryApprove(forged = false) {
    if (!a) return;
    setBusy("approve");
    const payload = forged
      ? {
          band: "LOW",
          score: 1.0,
          verified: true,
          policy: { allowed: true },
          override: true,
        }
      : {};
    const r = await approve(a.id, payload);
    setLastResponse({ ...r, forged });
    setBusy("");
    refresh();
  }

  async function doChallenge() {
    if (!a) return;
    setBusy("challenge");
    setLastResponse(null);
    setChallenge(await issueChallenge(a.session_id, a.id));
    setBusy("");
    refresh();
  }

  async function submitResponse(f: File) {
    if (!challenge) return;
    setBusy("respond");
    const r = await respondToChallenge(challenge.verification_id, f);
    setChallenge({ ...challenge, outcome: r });
    setBusy("");
    refresh();
  }

  async function doReset() {
    if (!a) return;
    setBusy("reset");
    await resetApproval(a.id);
    setChallenge(null);
    setLastResponse(null);
    setBusy("");
    refresh();
  }

  async function doSimulated(method: string) {
    if (!a) return;
    setBusy(method);
    await simulatedVerification(a.session_id, method);
    setBusy("");
    refresh();
  }

  if (!a) {
    return (
      <Card title="Mock approval system">
        <p className="text-xs text-zinc-500">
          {err || "No pending approvals. Run scripts/seed.py."}
        </p>
      </Card>
    );
  }

  const blocked = !a.policy.allowed;

  return (
    <div className="space-y-4">
      {blocked && a.policy.band === "HIGH" && (
        <div className="rounded-lg border-2 border-red-400 bg-red-50 p-4">
          <h2 className="text-base font-bold text-red-900">
            Possible AI voice impersonation
          </h2>
          <p className="mt-1 text-sm text-red-800">
            Do not approve the requested action until identity is independently
            verified.
          </p>
        </div>
      )}

      <Card
        title="Pending approval"
        subtitle="A mock transfer. The block below is enforced by the API, not the UI."
      >
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-2xl font-bold text-zinc-900">{inr(a.amount)}</div>
            <p className="text-sm text-zinc-600">{a.description}</p>
          </div>
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat
              label="state"
              value={
                <span
                  className={`rounded px-1.5 py-0.5 text-xs font-semibold uppercase ${
                    a.state === "approved"
                      ? "bg-emerald-100 text-emerald-800"
                      : a.state === "blocked"
                        ? "bg-red-100 text-red-800"
                        : "bg-zinc-200 text-zinc-700"
                  }`}
                >
                  {a.state}
                </span>
              }
            />
            <Stat
              label="linked call risk"
              value={
                a.session ? (
                  <span
                    className={`rounded px-1.5 py-0.5 text-xs font-semibold ${BAND_PILL[a.session.final_band] ?? ""}`}
                  >
                    {a.session.final_band} · {a.session.final_score?.toFixed(1)}
                  </span>
                ) : (
                  <span className="text-xs text-zinc-500">no session linked</span>
                )
              }
            />
            <Stat label="policy action" value={<span className="text-sm">{a.policy.action}</span>} />
            <Stat
              label="session"
              value={
                <code className="text-[11px]">
                  {a.session_id ? a.session_id.slice(0, 8) : "—"}
                </code>
              }
            />
          </dl>
        </div>

        <p className="mt-3 rounded bg-zinc-50 px-2 py-1.5 text-xs text-zinc-700">
          <span className="font-semibold">Server policy:</span> {a.policy.reason}
        </p>

        <div className="mt-3 flex flex-wrap gap-2">
          <button
            onClick={() => tryApprove(false)}
            disabled={blocked || busy === "approve"}
            title={blocked ? a.policy.reason : "Approve the transfer"}
            className="rounded bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-zinc-300 disabled:text-zinc-500"
          >
            Approve {inr(a.amount)}
          </button>
          <button
            onClick={() => tryApprove(true)}
            disabled={busy === "approve"}
            className="rounded border border-zinc-400 px-3 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-50"
          >
            Try to bypass (send forged low-risk payload)
          </button>
          <button
            onClick={doReset}
            disabled={busy === "reset"}
            className="ml-auto rounded border border-zinc-300 px-3 py-1.5 text-xs font-medium text-zinc-600 hover:bg-zinc-50 disabled:opacity-40"
            title="Demo affordance — clears the linked session; the policy then fails closed"
          >
            Reset demo
          </button>
        </div>
        <p className="mt-1.5 text-[11px] leading-snug text-zinc-500">
          The second button POSTs <code>{`{band:"LOW", verified:true, override:true}`}</code>{" "}
          — the same thing a judge could send from devtools. The server ignores
          every one of those fields and reads the band from its own session
          record, so the answer does not change.
        </p>

        {lastResponse && (
          <div
            className={`mt-3 rounded border p-2.5 ${
              lastResponse.status === 403
                ? "border-red-300 bg-red-50"
                : "border-emerald-300 bg-emerald-50"
            }`}
          >
            <div className="flex items-center gap-2">
              <span
                className={`rounded px-1.5 py-0.5 text-xs font-bold ${
                  lastResponse.status === 403
                    ? "bg-red-200 text-red-900"
                    : "bg-emerald-200 text-emerald-900"
                }`}
              >
                HTTP {lastResponse.status}
              </span>
              {lastResponse.forged && (
                <span className="text-[11px] font-semibold uppercase text-zinc-600">
                  forged payload
                </span>
              )}
              {lastResponse.body.error && (
                <code className="text-xs text-red-900">{lastResponse.body.error}</code>
              )}
            </div>
            <p className="mt-1 text-xs text-zinc-800">{lastResponse.body.reason}</p>
            <p className="mt-1 text-[11px] text-zinc-600">
              server-side band: <strong>{String(lastResponse.body.band)}</strong>
            </p>
          </div>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card
          title="Challenge–response verification"
          subtitle="Real: a fresh random phrase, and the response audio is re-analysed"
          right={
            <span className="rounded border border-sky-300 bg-sky-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-sky-800">
              real
            </span>
          }
        >
          <button
            onClick={doChallenge}
            disabled={busy === "challenge"}
            className="rounded bg-sky-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-sky-700 disabled:opacity-40"
          >
            Generate challenge phrase
          </button>
          {challenge && (
            <div className="mt-3 space-y-2">
              <div className="rounded border border-sky-300 bg-sky-50 px-3 py-2">
                <p className="text-[10px] uppercase tracking-wide text-sky-800">
                  Ask the caller to repeat
                </p>
                <p className="text-lg font-bold tracking-wide text-sky-900">
                  “{challenge.phrase}”
                </p>
              </div>
              <div>
                <input
                  ref={fileRef}
                  type="file"
                  accept="audio/*"
                  className="block w-full text-xs file:mr-2 file:rounded file:border-0 file:bg-zinc-200 file:px-2 file:py-1 file:text-xs"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) submitResponse(f);
                  }}
                />
                <p className="mt-1 text-[10px] text-zinc-500">
                  Upload the caller's response. It runs through the same analysis
                  pipeline; both the phrase match and the voice must pass.
                </p>
              </div>
              {busy === "respond" && (
                <p className="text-xs text-zinc-500">Re-analysing response…</p>
              )}
              {challenge.outcome && (
                <div
                  className={`rounded border p-2 text-xs ${
                    challenge.outcome.result === "passed"
                      ? "border-emerald-300 bg-emerald-50 text-emerald-900"
                      : "border-red-300 bg-red-50 text-red-900"
                  }`}
                >
                  <p className="font-semibold uppercase">{challenge.outcome.result}</p>
                  <p>{challenge.outcome.why}</p>
                  <p className="mt-1 text-[11px] opacity-80">
                    phrase match {(challenge.outcome.phrase_check?.ratio ?? 0) * 100}% ·
                    re-analysis {challenge.outcome.reanalysis?.band} (
                    {challenge.outcome.reanalysis?.score})
                  </p>
                </div>
              )}
            </div>
          )}
          <p className="mt-2 text-[10px] italic leading-snug text-zinc-500">
            Defeats replay of pre-recorded audio — a recording made before the
            phrase existed cannot contain it. It does <strong>not</strong> defeat
            a live real-time voice-conversion attacker who can simply repeat the
            phrase in the cloned voice.
          </p>
        </Card>

        <Card
          title="Other verification methods"
          subtitle="State machine only — nothing is actually contacted"
          right={
            <span className="rounded border border-zinc-400 bg-zinc-200 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-zinc-700">
              simulated
            </span>
          }
        >
          <div className="flex flex-wrap gap-2">
            {SIMULATED.map((m) => (
              <button
                key={m.method}
                onClick={() => doSimulated(m.method)}
                disabled={busy === m.method}
                className="rounded border border-zinc-300 px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-50 disabled:opacity-40"
              >
                {m.label}
              </button>
            ))}
          </div>

          <h3 className="mt-4 mb-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
            Attempts for this session
          </h3>
          {vers.length === 0 ? (
            <p className="text-xs text-zinc-500">No verification attempts yet.</p>
          ) : (
            <ul className="max-h-56 space-y-1 overflow-y-auto">
              {vers.map((v) => (
                <li
                  key={v.id}
                  className="flex items-center justify-between gap-2 rounded border border-zinc-200 px-2 py-1 text-[11px]"
                >
                  <span className="font-medium text-zinc-700">
                    {v.method.replace(/_/g, " ")}
                    {v.challenge_phrase && (
                      <span className="ml-1 text-zinc-400">“{v.challenge_phrase}”</span>
                    )}
                  </span>
                  <span
                    className={`rounded px-1.5 py-0.5 font-semibold uppercase ${
                      v.result === "passed"
                        ? "bg-emerald-100 text-emerald-800"
                        : v.result === "failed"
                          ? "bg-red-100 text-red-800"
                          : "bg-zinc-200 text-zinc-600"
                    }`}
                  >
                    {v.result}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}
