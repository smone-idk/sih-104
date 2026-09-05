import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Push-to-record (§1 authorized channels, §12 Privacy Center).
 *
 * The Privacy Center claims this prototype never opens a background microphone.
 * This component is written so that claim is true by construction:
 *
 *   1. `getUserMedia` is called ONLY inside the press handler. There is no call
 *      on mount, in an effect, or on hover — nothing opens the mic until the
 *      user asks for it. Grep the file: there is exactly one call site.
 *   2. While recording, an unmistakable indicator is shown (pulsing red dot,
 *      "RECORDING", elapsed time) and the browser's own tab indicator is lit.
 *   3. Every track is stopped on stop, on unmount, and on `pagehide`, so
 *      navigating away releases the microphone rather than leaving it open.
 *   4. Nothing is uploaded automatically. The clip stays local until the user
 *      presses Analyse.
 */
export function PushToRecord({
  onClip,
  disabled,
  label = "Push to record",
}: {
  onClip: (file: File) => void;
  disabled?: boolean;
  label?: string;
}) {
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [err, setErr] = useState("");
  const [clipUrl, setClipUrl] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<number | null>(null);

  /** Release the microphone. Safe to call repeatedly. */
  const releaseMic = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    try {
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
    } catch {
      /* already stopped */
    }
    releaseMic();
    setRecording(false);
  }, [releaseMic]);

  // Stop on unmount (navigation away) and on tab hide. The mic must never
  // outlive the screen that opened it.
  useEffect(() => {
    const onHide = () => stop();
    window.addEventListener("pagehide", onHide);
    return () => {
      window.removeEventListener("pagehide", onHide);
      stop();
    };
  }, [stop]);

  useEffect(() => () => {
    if (clipUrl) URL.revokeObjectURL(clipUrl);
  }, [clipUrl]);

  /** The ONLY getUserMedia call site in the app, and it is inside a click. */
  async function startRecording() {
    setErr("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setErr("This browser does not expose microphone capture.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      const rec = new MediaRecorder(stream);
      recorderRef.current = rec;
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: rec.mimeType || "audio/webm",
        });
        if (clipUrl) URL.revokeObjectURL(clipUrl);
        setClipUrl(URL.createObjectURL(blob));
        const ext = (rec.mimeType || "audio/webm").includes("ogg") ? "ogg" : "webm";
        onClip(new File([blob], `recording.${ext}`, { type: blob.type }));
      };
      rec.start();
      setRecording(true);
      setElapsed(0);
      timerRef.current = window.setInterval(() => setElapsed((s) => s + 1), 1000);
    } catch (e) {
      releaseMic();
      setRecording(false);
      setErr(
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Microphone permission denied. Nothing was captured."
          : `Could not start recording: ${String(e)}`,
      );
    }
  }

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-3">
      <div className="flex flex-wrap items-center gap-3">
        {!recording ? (
          <button
            onClick={startRecording}
            disabled={disabled}
            className="flex items-center gap-2 rounded bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 disabled:opacity-40"
          >
            <span className="h-2.5 w-2.5 rounded-full bg-white" />
            {label}
          </button>
        ) : (
          <button
            onClick={stop}
            className="flex items-center gap-2 rounded bg-zinc-800 px-4 py-2 text-sm font-semibold text-white hover:bg-zinc-900"
          >
            <span className="h-2.5 w-2.5 bg-white" />
            Stop recording
          </button>
        )}

        {recording && (
          <span
            role="status"
            aria-live="assertive"
            className="flex items-center gap-2 rounded-full border-2 border-red-500 bg-red-50 px-3 py-1"
          >
            <span className="h-3 w-3 animate-pulse rounded-full bg-red-600" />
            <span className="text-sm font-bold tracking-wide text-red-800">
              RECORDING
            </span>
            <span className="tabular-nums text-sm text-red-700">
              {String(Math.floor(elapsed / 60)).padStart(2, "0")}:
              {String(elapsed % 60).padStart(2, "0")}
            </span>
          </span>
        )}

        {clipUrl && !recording && (
          <audio controls src={clipUrl} className="h-8" />
        )}
      </div>

      <p className="mt-2 text-[11px] leading-snug text-zinc-500">
        The microphone is opened <strong>only</strong> when you press record, and
        released when you stop or leave this page. Nothing is captured in the
        background and nothing is uploaded until you choose to analyse it.
      </p>
      {err && (
        <p className="mt-1 rounded bg-red-50 px-2 py-1 text-xs text-red-700">{err}</p>
      )}
    </div>
  );
}
