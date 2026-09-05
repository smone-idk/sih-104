import { useCallback, useEffect, useState } from "react";
import {
  deleteProfile,
  enrolProfile,
  getProfiles,
  type VoiceProfile,
} from "../api";
import { Card } from "../components/common";
import { PushToRecord } from "../components/PushToRecord";

/**
 * Voice profiles (§11 screen 6). Enrolment stores the ECAPA embedding, never
 * the audio — the backend has no column for it.
 *
 * This is also the Phase 5 gate: deleting the last profile must visibly disable
 * the speaker-consistency layer and redistribute its weight, which the banner
 * below states and the Live Analysis explainability table then shows.
 */
export default function Profiles() {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [name, setName] = useState("");
  const [role, setRole] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    try {
      setProfiles((await getProfiles()).profiles);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function submit() {
    if (!name.trim() || files.length === 0) return;
    setBusy(true);
    setErr("");
    setMsg("");
    try {
      const r = await enrolProfile(name.trim(), role.trim(), files);
      setMsg(
        `Enrolled “${r.display_name}” from ${r.n_enroll_clips} clip(s), ` +
          `${r.total_audio_s}s of audio → ${r.embedding_dim}-dim embedding. ${r.stored}.`,
      );
      setName("");
      setRole("");
      setFiles([]);
      refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove(p: VoiceProfile) {
    setBusy(true);
    setErr("");
    try {
      const r = await deleteProfile(p.id);
      setMsg(`Deleted “${p.display_name}”. ${r.effect}`);
      refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      {profiles.length === 0 && (
        <div className="rounded-lg border-2 border-amber-400 bg-amber-50 p-3">
          <h2 className="text-sm font-bold text-amber-900">
            No enrolled voice profile
          </h2>
          <p className="mt-0.5 text-xs text-amber-800">
            The <strong>speaker-consistency</strong> layer is unavailable. Fusion
            redistributes its 0.20 weight across the remaining layers rather than
            substituting a value — you can see this on Live Analysis, where the
            component reads “layer unavailable — no enrolled profile” with an
            effective weight of 0.000.
          </p>
        </div>
      )}

      <Card
        title="Enrolled voice profiles"
        subtitle="A profile is a 192-dim ECAPA embedding. The enrolment audio is never stored."
      >
        {profiles.length === 0 ? (
          <p className="text-xs text-zinc-500">No profiles enrolled.</p>
        ) : (
          <table className="w-full text-xs">
            <thead className="text-[10px] uppercase tracking-wide text-zinc-500">
              <tr>
                <th className="py-1 text-left font-medium">name</th>
                <th className="text-left font-medium">role</th>
                <th className="text-right font-medium">clips</th>
                <th className="text-right font-medium">dim</th>
                <th className="text-left font-medium">enrolled</th>
                <th className="text-left font-medium">stored</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {profiles.map((p) => (
                <tr key={p.id} className="border-t border-zinc-100">
                  <td className="py-1.5 font-medium text-zinc-800">{p.display_name}</td>
                  <td className="text-zinc-600">{p.role || "—"}</td>
                  <td className="text-right tabular-nums">{p.n_enroll_clips}</td>
                  <td className="text-right tabular-nums">{p.embedding_dim}</td>
                  <td className="text-zinc-500">{p.created_at}</td>
                  <td className="text-[10px] text-zinc-500">embedding only</td>
                  <td className="text-right">
                    <button
                      onClick={() => remove(p)}
                      disabled={busy}
                      className="rounded border border-red-300 px-2 py-0.5 text-[11px] font-medium text-red-700 hover:bg-red-50 disabled:opacity-40"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Card
        title="Enrol a new profile"
        subtitle="Upload one or more clips, or record directly. Embeddings are averaged."
      >
        <div className="flex flex-wrap gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-wide text-zinc-500">name</span>
            <input
              className="rounded border border-zinc-300 px-2 py-1.5 text-sm"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Rajesh Sharma"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-wide text-zinc-500">role</span>
            <input
              className="rounded border border-zinc-300 px-2 py-1.5 text-sm"
              value={role}
              onChange={(e) => setRole(e.target.value)}
              placeholder="Chief Financial Officer"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-wide text-zinc-500">
              clips ({files.length} selected)
            </span>
            <input
              type="file"
              accept="audio/*"
              multiple
              className="text-xs file:mr-2 file:rounded file:border-0 file:bg-zinc-200 file:px-2 file:py-1 file:text-xs"
              onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
            />
          </label>
        </div>

        <div className="mt-3">
          <PushToRecord
            label="Record enrolment clip"
            disabled={busy}
            onClip={(f) => setFiles((prev) => [...prev, f])}
          />
        </div>

        <button
          onClick={submit}
          disabled={busy || !name.trim() || files.length === 0}
          className="mt-3 rounded bg-sky-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-sky-700 disabled:opacity-40"
        >
          Enrol profile
        </button>

        {msg && (
          <p className="mt-2 rounded bg-emerald-50 px-2 py-1.5 text-xs text-emerald-900">
            {msg}
          </p>
        )}
        {err && (
          <p className="mt-2 rounded bg-red-50 px-2 py-1.5 text-xs text-red-700">{err}</p>
        )}
      </Card>
    </div>
  );
}
