"""Audio loading + resampling.

WAV/FLAC/OGG go through soundfile; MP3/M4A/AAC and anything soundfile rejects
fall back to PyAV (`av`), whose wheel bundles ffmpeg — no system ffmpeg needed.
Everything downstream works on float32 mono at `settings.sample_rate` (16 kHz).

Privacy (§12): this module never writes the decoded audio to disk. The pipeline
holds it in memory for the analysis and drops it unless RETAIN_AUDIO is set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import get_settings


class AudioLoadError(RuntimeError):
    pass


def load_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Return (mono float32 in [-1, 1], sample_rate)."""
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"no such file: {path}")
    target_sr = target_sr or get_settings().sample_rate

    audio, sr = _try_soundfile(path)
    if audio is None:
        audio, sr = _try_pyav(path)
    if audio is None:
        raise AudioLoadError(f"could not decode {path} (tried soundfile + pyav)")

    audio = _to_mono_float32(audio)
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
        sr = target_sr
    # guard against denormals / clipping from decode
    audio = np.nan_to_num(audio, copy=False)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32), sr


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)
    import resampy

    return resampy.resample(audio.astype(np.float32), sr, target_sr, filter="kaiser_fast")


# --- backends ------------------------------------------------------------
def _try_soundfile(path: Path):
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return data, sr
    except Exception:
        return None, 0


def _try_pyav(path: Path):
    try:
        import av

        with av.open(str(path)) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            sr = stream.rate or 16000
            chunks: list[np.ndarray] = []
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sr)
            for frame in container.decode(stream):
                for rframe in resampler.resample(frame):
                    chunks.append(rframe.to_ndarray().reshape(-1))
            if not chunks:
                return None, 0
            return np.concatenate(chunks).astype(np.float32), sr
    except Exception:
        return None, 0


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # soundfile gives (frames, channels)
        ch_axis = 1 if audio.shape[1] <= audio.shape[0] else 0
        audio = audio.mean(axis=ch_axis)
    return np.ascontiguousarray(audio, dtype=np.float32)
