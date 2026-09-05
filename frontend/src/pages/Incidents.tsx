import { Fragment, useEffect, useMemo, useState } from "react";
import { getIncidents, type Incident } from "../api";
import { Card } from "../components/common";

const BAND_PILL: Record<string, string> = {
  LOW: "bg-emerald-100 text-emerald-800",
  MEDIUM: "bg-amber-100 text-amber-900",
  HIGH: "bg-red-100 text-red-800",
  UNKNOWN: "bg-zinc-200 text-zinc-600",
};

const ACTION_PILL: Record<string, string> = {
  ALLOW: "text-emerald-700",
  VERIFY: "text-amber-700",
  ESCALATE: "text-red-700",
};

type SortKey = "created_at" | "score" | "band";

export default function Incidents() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [band, setBand] = useState("");
  const [sort, setSort] = useState<SortKey>("created_at");
  const [desc, setDesc] = useState(true);
  const [open, setOpen] = useState<string | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    getIncidents(band || undefined)
      .then((d) => setIncidents(d.incidents))
      .catch((e) => setErr(String(e)));
  }, [band]);

  const rows = useMemo(() => {
    const order = { LOW: 0, MEDIUM: 1, HIGH: 2, UNKNOWN: -1 } as Record<string, number>;
    const r = [...incidents].sort((a, b) => {
      let d = 0;
      if (sort === "score") d = a.score - b.score;
      else if (sort === "band") d = (order[a.band] ?? -1) - (order[b.band] ?? -1);
      else d = a.created_at.localeCompare(b.created_at);
      return desc ? -d : d;
    });
    return r;
  }, [incidents, sort, desc]);

  function th(key: SortKey, label: string) {
    return (
      <th
        className="cursor-pointer py-1 text-left font-medium hover:text-zinc-900"
        onClick={() => {
          if (sort === key) setDesc(!desc);
          else {
            setSort(key);
            setDesc(true);
          }
        }}
      >
        {label} {sort === key ? (desc ? "▾" : "▴") : ""}
      </th>
    );
  }

  return (
    <Card
      title="Incident log"
      subtitle="Every analysed MEDIUM/HIGH call, approval attempt and verification attempt is recorded"
      right={
        <select
          className="rounded border border-zinc-300 px-2 py-1 text-xs"
          value={band}
          onChange={(e) => setBand(e.target.value)}
        >
          <option value="">all bands</option>
          <option value="HIGH">HIGH</option>
          <option value="MEDIUM">MEDIUM</option>
          <option value="LOW">LOW</option>
          <option value="UNKNOWN">UNKNOWN</option>
        </select>
      }
    >
      {err && <p className="rounded bg-red-50 px-2 py-1 text-xs text-red-700">{err}</p>}
      {rows.length === 0 ? (
        <p className="text-xs text-zinc-500">
          No incidents yet. Run a MEDIUM/HIGH scenario on Live Analysis, or try
          an approval.
        </p>
      ) : (
        <table className="w-full text-xs">
          <thead className="text-[10px] uppercase tracking-wide text-zinc-500">
            <tr>
              {th("created_at", "time")}
              {th("band", "band")}
              {th("score", "score")}
              <th className="py-1 text-left font-medium">action</th>
              <th className="py-1 text-left font-medium">summary</th>
              <th className="py-1 text-left font-medium">session</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((i) => (
              <Fragment key={i.id}>
                <tr
                  className="cursor-pointer border-t border-zinc-100 hover:bg-zinc-50"
                  onClick={() => setOpen(open === i.id ? null : i.id)}
                >
                  <td className="py-1.5 tabular-nums text-zinc-600">{i.created_at}</td>
                  <td>
                    <span
                      className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${BAND_PILL[i.band] ?? BAND_PILL.UNKNOWN}`}
                    >
                      {i.band}
                    </span>
                  </td>
                  <td className="tabular-nums">{i.score.toFixed(1)}</td>
                  <td className={`font-semibold ${ACTION_PILL[i.action] ?? ""}`}>
                    {i.action}
                  </td>
                  <td className="text-zinc-700">{i.summary}</td>
                  <td>
                    <code className="text-[10px] text-zinc-500">
                      {i.session_id ? i.session_id.slice(0, 8) : "—"}
                    </code>
                  </td>
                </tr>
                {open === i.id && (
                  <tr className="bg-zinc-50">
                    <td colSpan={6} className="px-2 py-2">
                      <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all text-[10px] leading-snug text-zinc-700">
                        {JSON.stringify(i.payload, null, 2)}
                      </pre>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}
