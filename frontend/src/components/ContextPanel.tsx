import type { ContextMsg, ContextQuote, TranscriptData } from "../api";
import { Card, KindBadge } from "./common";

const SIGNAL_LABEL: Record<string, string> = {
  urgency: "Urgency",
  secrecy: "Secrecy",
  authority_claim: "Authority claim",
  transaction_intent: "Transaction + amount",
  out_of_workflow: "Out of workflow",
  credential_solicitation: "Credential / PII solicitation",
};

const SIGNAL_ORDER = [
  "transaction_intent",
  "credential_solicitation",
  "out_of_workflow",
  "secrecy",
  "authority_claim",
  "urgency",
];

function inr(v: number): string {
  if (v >= 1e7) return `₹${(v / 1e7).toFixed(2).replace(/\.00$/, "")} crore`;
  if (v >= 1e5) return `₹${(v / 1e5).toFixed(2).replace(/\.00$/, "")} lakh`;
  return `₹${v.toLocaleString("en-IN")}`;
}

/**
 * Context signals with the transcript quote that triggered each one (§5).
 * A signal with no quote cannot have a value — the backend refuses to build
 * one — so every bar here is traceable to something that was actually said.
 */
export function ContextPanel({
  context,
}: {
  context: ContextMsg["context"] | null;
}) {
  const signals = context?.signals ?? {};
  const present = SIGNAL_ORDER.filter((n) => n in signals);
  const amount = signals.transaction_intent?.detail?.amount_inr as number | undefined;

  return (
    <Card
      title="Context signals"
      subtitle="Derived from the transcript — every flag shows the quote that triggered it"
      right={<KindBadge kind="heuristic" />}
    >
      {present.length === 0 ? (
        <p className="text-xs text-zinc-500">
          No transcript yet. Context components report unavailable and their
          fusion weight is redistributed — never replaced with a neutral value.
        </p>
      ) : (
        <div className="space-y-2.5">
          {amount != null && (
            <div className="rounded border border-amber-300 bg-amber-50 px-2 py-1.5">
              <span className="text-[10px] uppercase tracking-wide text-amber-800">
                Amount detected
              </span>
              <div className="text-lg font-bold text-amber-900">{inr(amount)}</div>
              <span className="text-[10px] text-amber-800">
                normalised from “{String(signals.transaction_intent?.detail?.amount_text ?? "")}”
              </span>
            </div>
          )}
          {present.map((name) => {
            const sig = signals[name];
            const on = sig.value > 0;
            return (
              <div key={name} className={on ? "" : "opacity-45"}>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-xs font-medium text-zinc-800">
                    {SIGNAL_LABEL[name] ?? name}
                  </span>
                  <span className="text-xs font-semibold tabular-nums text-zinc-700">
                    {sig.value.toFixed(2)}
                  </span>
                </div>
                <div className="mt-0.5 h-1.5 rounded bg-zinc-100">
                  <div
                    className={`h-1.5 rounded ${on ? "bg-amber-500" : "bg-zinc-200"}`}
                    style={{ width: `${sig.value * 100}%` }}
                  />
                </div>
                {sig.matches.length > 0 ? (
                  <ul className="mt-1 space-y-0.5">
                    {sig.matches.slice(0, 3).map((m, i) => (
                      <li key={i} className="text-[11px] leading-snug text-zinc-600">
                        <span className="text-zinc-400">
                          {m.t_start != null ? `${m.t_start.toFixed(1)}s` : "—"}
                        </span>{" "}
                        <span className="rounded bg-amber-100 px-1 font-medium text-amber-900">
                          “{m.quote}”
                        </span>{" "}
                        <span className="text-zinc-400">({m.rule})</span>
                      </li>
                    ))}
                    {sig.matches.length > 3 && (
                      <li className="text-[10px] text-zinc-400">
                        +{sig.matches.length - 3} more
                      </li>
                    )}
                  </ul>
                ) : (
                  <p className="mt-0.5 text-[10px] text-zinc-400">no match in transcript</p>
                )}
              </div>
            );
          })}
        </div>
      )}

      {context?.directory && (
        <div className="mt-3 rounded border border-zinc-200 bg-zinc-50 p-2">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">
              Enterprise directory
            </span>
            <span className="rounded bg-zinc-200 px-1.5 py-0.5 text-[9px] font-semibold uppercase text-zinc-700">
              demo data
            </span>
          </div>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px]">
            <dt className="text-zinc-500">caller</dt>
            <dd className="text-zinc-800">{String(context.directory.display_name ?? "—")}</dd>
            <dt className="text-zinc-500">known contact</dt>
            <dd className="text-zinc-800">{context.directory.known_contact ? "yes" : "no"}</dd>
            <dt className="text-zinc-500">identity verified</dt>
            <dd className="text-zinc-800">{context.directory.verified_identity ? "yes" : "no"}</dd>
            <dt className="text-zinc-500">prior interactions</dt>
            <dd className="text-zinc-800">{String(context.directory.prior_interactions ?? 0)}</dd>
          </dl>
        </div>
      )}
      <p className="mt-2 text-[10px] italic leading-snug text-zinc-500">
        Rules and regex over real ASR output — not a trained classifier. Quotes
        are timestamped by interpolation within the utterance, not word-level
        alignment.
      </p>
    </Card>
  );
}

/** Rolling transcript. Context updates at this cadence (3–6 s utterances),
 *  while the acoustic score ticks every second — §4 asks for both to be visible. */
export function TranscriptPanel({
  transcript,
  quotes,
}: {
  transcript: TranscriptData | null;
  quotes: ContextQuote[];
}) {
  const segs = transcript?.segments ?? [];
  const hit = (i: number) => quotes.some((q) => q.utterance_index === i);
  return (
    <Card
      title="Transcript"
      subtitle={`Whisper over VAD-segmented utterances — ${segs.length} utterance(s); acoustic score ticks every 1 s independently`}
      right={<KindBadge kind="pretrained" />}
    >
      {segs.length === 0 ? (
        <p className="text-xs text-zinc-500">No speech transcribed yet.</p>
      ) : (
        <ol className="max-h-64 space-y-1 overflow-y-auto pr-1">
          {segs.map((sg) => (
            <li
              key={sg.index}
              className={`rounded px-1.5 py-1 text-[11px] leading-snug ${
                hit(sg.index) ? "bg-amber-50 text-zinc-900" : "text-zinc-600"
              }`}
            >
              <span className="mr-1.5 tabular-nums text-zinc-400">
                {sg.t_start.toFixed(1)}s
              </span>
              {sg.text}
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}
