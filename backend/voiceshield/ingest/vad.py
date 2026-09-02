"""Voice-activity detection — backend-dispatching front end.

Two backends, selected by `settings.vad_backend`:

  "silero"  (default) — Silero VAD, a small trained model (MIT, ~2 MB, bundled
                        in the wheel). Used to skip non-speech windows (§4) and
                        to segment utterances for the ASR worker.
  "energy"            — the original relative-noise-floor + absolute-floor gate,
                        kept behind the flag so the two can be compared.

VAD is not a scored layer; it only decides which windows are worth scoring.
`analyze()` runs it ONCE over the whole clip — callers then ask `ratio_in()` per
window instead of re-running the model per window.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import get_settings

log = logging.getLogger("voiceshield.vad")

FRAME_MS = 30.0
ABS_FLOOR_DB = -45.0   # energy backend only


@dataclass
class SpeechSegment:
    start: float          # seconds
    end: float
    duration: float


@dataclass
class VadResult:
    backend: str                       # backend actually used
    segments: list[SpeechSegment]
    frame_mask: np.ndarray             # bool per frame
    frame_times: np.ndarray            # frame start times (s)
    frame_s: float
    total_s: float
    fallback_reason: str = ""          # set if the requested backend failed
    detail: dict = field(default_factory=dict)

    def ratio_in(self, t0: float, t1: float) -> float:
        """Fraction of the [t0, t1) span marked as speech."""
        if self.frame_mask.size == 0 or t1 <= t0:
            return 0.0
        i0 = int(np.floor(t0 / self.frame_s))
        i1 = int(np.ceil(t1 / self.frame_s))
        i0, i1 = max(0, i0), min(len(self.frame_mask), i1)
        if i1 <= i0:
            return 0.0
        return float(np.mean(self.frame_mask[i0:i1]))

    @property
    def overall_ratio(self) -> float:
        return float(np.mean(self.frame_mask)) if self.frame_mask.size else 0.0


# --------------------------------------------------------------- entry point
def analyze(audio: np.ndarray, sr: int, backend: str | None = None) -> VadResult:
    s = get_settings()
    backend = (backend or s.vad_backend).lower()
    total = len(audio) / sr if sr else 0.0

    if backend == "silero":
        try:
            from . import vad_silero

            spans = vad_silero.speech_timestamps(
                audio, sr,
                threshold=s.vad_threshold,
                min_speech_ms=s.vad_min_speech_ms,
                min_silence_ms=s.vad_min_silence_ms,
                speech_pad_ms=s.vad_speech_pad_ms,
            )
            segs = [SpeechSegment(a, b, b - a) for a, b in spans]
            mask, times = _mask_from_segments(segs, total)
            return VadResult("silero", segs, mask, times, FRAME_MS / 1000.0, total,
                             detail={"threshold": s.vad_threshold,
                                     "n_segments": len(segs)})
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            log.warning("Silero VAD failed (%s) — falling back to the energy gate", reason)
            res = _energy_analyze(audio, sr, total)
            res.fallback_reason = reason
            return res

    if backend != "energy":
        log.warning("unknown vad_backend %r — using energy gate", backend)
    return _energy_analyze(audio, sr, total)


def _mask_from_segments(segs: list[SpeechSegment], total_s: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    fs = FRAME_MS / 1000.0
    n = max(1, int(np.ceil(total_s / fs))) if total_s > 0 else 0
    mask = np.zeros(n, dtype=bool)
    times = np.arange(n) * fs
    for sg in segs:
        i0 = max(0, int(np.floor(sg.start / fs)))
        i1 = min(n, int(np.ceil(sg.end / fs)))
        if i1 > i0:
            mask[i0:i1] = True
    return mask, times


# ------------------------------------------------------------- energy gate
def frame_energy_db(audio: np.ndarray, sr: int, frame_ms: float = FRAME_MS
                    ) -> tuple[np.ndarray, np.ndarray]:
    n = max(1, int(sr * frame_ms / 1000))
    pad = (-len(audio)) % n
    if pad:
        audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])
    frames = audio.reshape(-1, n)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    times = np.arange(len(frames)) * (n / sr)
    return db, times


def _energy_mask(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    db, times = frame_energy_db(audio, sr)
    if db.size == 0:
        return np.array([], dtype=bool), times
    noise_floor = np.percentile(db, 15)
    peak = np.percentile(db, 95)
    thr = max(noise_floor + 6.0, peak - 35.0)
    mask = (db > thr) | (db > ABS_FLOOR_DB)
    return _smooth(mask), times


def _energy_analyze(audio: np.ndarray, sr: int, total: float) -> VadResult:
    mask, times = _energy_mask(audio, sr)
    fs = FRAME_MS / 1000.0
    segs: list[SpeechSegment] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            start = max(0.0, times[i] - 0.1)
            end = min(total, times[j - 1] + fs + 0.1)
            if end - start >= 0.3:
                segs.append(SpeechSegment(start, end, end - start))
            i = j
        else:
            i += 1
    return VadResult("energy", segs, mask, times, fs, total,
                     detail={"abs_floor_db": ABS_FLOOR_DB})


def _smooth(mask: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Fill short gaps and drop isolated blips (~<90 ms at 30 ms frames)."""
    m = mask.copy()
    i = 0
    while i < len(m):
        if not m[i]:
            j = i
            while j < len(m) and not m[j]:
                j += 1
            if 0 < i and j < len(m) and (j - i) < min_run:
                m[i:j] = True
            i = j
        else:
            i += 1
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if (j - i) < min_run:
                m[i:j] = False
            i = j
        else:
            i += 1
    return m


# ------------------------------------------------- backwards-compatible API
def speech_mask(audio: np.ndarray, sr: int, frame_ms: float = FRAME_MS
                ) -> tuple[np.ndarray, np.ndarray]:
    r = analyze(audio, sr)
    return r.frame_mask, r.frame_times


def segments(audio: np.ndarray, sr: int, **_kw) -> list[SpeechSegment]:
    return analyze(audio, sr).segments


def speech_ratio(audio: np.ndarray, sr: int) -> float:
    return analyze(audio, sr).overall_ratio
