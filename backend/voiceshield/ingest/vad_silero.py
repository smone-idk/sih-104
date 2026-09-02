"""Silero VAD backend (MIT, ~2 MB).

The model ships inside the `silero-vad` wheel, so there is nothing extra to
download and the offline path is automatic. Loaded once and kept warm, like
every other model (§1).

Silero is a small trained classifier, so unlike the energy gate this is a
`pretrained` component — but it is not a *scored* layer, it only decides which
windows are worth scoring and where utterances start/end for ASR.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger("voiceshield.vad.silero")

_LOCK = threading.Lock()
_MODEL = None
_UTILS = None
_LOAD_ERROR = ""

# Silero v5 consumes fixed 512-sample chunks at 16 kHz (32 ms).
CHUNK = 512
SILERO_SR = 16000


def available() -> bool:
    _ensure_loaded()
    return _MODEL is not None


def load_error() -> str:
    _ensure_loaded()
    return _LOAD_ERROR


def _ensure_loaded() -> None:
    global _MODEL, _UTILS, _LOAD_ERROR
    if _MODEL is not None or _LOAD_ERROR:
        return
    with _LOCK:
        if _MODEL is not None or _LOAD_ERROR:
            return
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps

            _MODEL = load_silero_vad()          # bundled weights, no network
            _UTILS = {"get_speech_timestamps": get_speech_timestamps}
            log.info("Silero VAD loaded (bundled weights)")
        except Exception as exc:                # stay usable via the energy gate
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            log.warning("Silero VAD unavailable: %s", _LOAD_ERROR)


def speech_timestamps(audio: np.ndarray, sr: int, *, threshold: float = 0.5,
                      min_speech_ms: int = 250, min_silence_ms: int = 100,
                      speech_pad_ms: int = 30) -> list[tuple[float, float]]:
    """Return [(start_s, end_s), ...]. Raises if the model is unavailable."""
    _ensure_loaded()
    if _MODEL is None:
        raise RuntimeError(f"Silero VAD unavailable: {_LOAD_ERROR}")
    import torch

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SILERO_SR:
        from .audio import resample

        x = resample(x, sr, SILERO_SR)

    with torch.no_grad():
        ts = _UTILS["get_speech_timestamps"](
            torch.from_numpy(np.ascontiguousarray(x)),
            _MODEL,
            sampling_rate=SILERO_SR,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
    return [(t["start"] / SILERO_SR, t["end"] / SILERO_SR) for t in ts]


def reset() -> None:
    """Drop the loaded model — used by tests that force the energy backend."""
    global _MODEL, _UTILS, _LOAD_ERROR
    with _LOCK:
        _MODEL, _UTILS, _LOAD_ERROR = None, None, ""
