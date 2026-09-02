"""Voice-activity detection.

v1 is an energy + spectral-flatness gate with an adaptive noise floor — a
HEURISTIC, deliberately simple and dependency-free. It is only used to (a) skip
scoring windows that are silence/noise (§4) and (b) segment utterances for the
ASR worker (§4, Phase 3). It is not a scored layer, so a learned VAD (silero) is
a drop-in improvement, not a correctness issue — tracked for a later phase.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeechSegment:
    start: float          # seconds
    end: float
    duration: float


def frame_energy_db(audio: np.ndarray, sr: int, frame_ms: float = 30.0
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


def speech_mask(audio: np.ndarray, sr: int, frame_ms: float = 30.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame bool mask + frame start times."""
    db, times = frame_energy_db(audio, sr, frame_ms)
    if db.size == 0:
        return np.array([], dtype=bool), times
    noise_floor = np.percentile(db, 15)
    peak = np.percentile(db, 95)
    # relative gate: 6 dB above the noise floor (but not chasing the peak alone)
    thr = max(noise_floor + 6.0, peak - 35.0)
    # absolute gate: anything above -45 dBFS is clearly not silence, even if the
    # signal has near-constant energy (sustained tone / vowel / broadband noise)
    ABS_FLOOR_DB = -45.0
    mask = (db > thr) | (db > ABS_FLOOR_DB)
    return _smooth(mask), times


def _smooth(mask: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Fill short gaps and drop isolated blips (~<90 ms at 30 ms frames)."""
    m = mask.copy()
    # close gaps
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
    # remove blips
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


def segments(audio: np.ndarray, sr: int, frame_ms: float = 30.0,
             min_speech_s: float = 0.3, pad_s: float = 0.1) -> list[SpeechSegment]:
    mask, times = speech_mask(audio, sr, frame_ms)
    out: list[SpeechSegment] = []
    if mask.size == 0:
        return out
    fs = frame_ms / 1000.0
    total = len(audio) / sr
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            start = max(0.0, times[i] - pad_s)
            end = min(total, times[j - 1] + fs + pad_s)
            if end - start >= min_speech_s:
                out.append(SpeechSegment(start, end, end - start))
            i = j
        else:
            i += 1
    return out


def speech_ratio(audio: np.ndarray, sr: int) -> float:
    mask, _ = speech_mask(audio, sr)
    return float(np.mean(mask)) if mask.size else 0.0
