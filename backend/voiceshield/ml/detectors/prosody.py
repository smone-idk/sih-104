"""Prosody anomaly — real DSP, badged HEURISTIC.

Extracts F0 contour + variance, jitter, shimmer, speaking rate, pause
statistics, energy variation, spectral flatness with librosa +
praat-parselmouth. The anomaly score is the distance of this feature vector
from a human-baseline distribution (mean/std) estimated over the bundled
genuine reference clips by scripts/build_baseline.py.

If the baseline file is missing we fall back to built-in priors and say so on
the card — the score is still a real measurement, just against a weaker
reference.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..detectors.base import Detector, DetectorResult
from ...config import get_settings

# Built-in fallback baseline (mean, std) for adult conversational speech,
# 16 kHz, clean. Replaced by models/prosody_baseline.json when present.
_FALLBACK_BASELINE: dict[str, tuple[float, float]] = {
    "f0_mean": (165.0, 55.0),
    "f0_std": (42.0, 18.0),
    "jitter_local": (0.018, 0.012),
    "shimmer_local": (0.065, 0.03),
    "speaking_rate_sylps": (4.1, 1.1),
    "pause_ratio": (0.22, 0.12),
    "energy_cv": (0.55, 0.22),
    "spectral_flatness": (0.19, 0.09),
}


class ProsodyAnomalyDetector(Detector):
    name = "prosody_anomaly"
    kind = "heuristic"
    feeds = "prosody_anomaly"

    def __init__(self) -> None:
        super().__init__()
        self._baseline = _FALLBACK_BASELINE
        self._baseline_source = "built-in priors (no baseline file)"

    def load(self) -> None:
        self.device = "cpu"
        try:
            import librosa  # noqa: F401
            import parselmouth  # noqa: F401
        except Exception as exc:
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            return
        s = get_settings()
        bpath = s.model_cache_dir / "prosody_baseline.json"
        if bpath.exists():
            raw = json.loads(bpath.read_text())
            self._baseline = {k: tuple(v) for k, v in raw["features"].items()}
            self._baseline_source = (
                f"{bpath.name} (n={raw.get('n_clips', '?')} genuine clips)"
            )
        self.available = True

    # --- feature extraction ------------------------------------------
    def extract_features(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        import librosa
        import parselmouth
        from parselmouth.praat import call

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float64)
        dur = len(audio) / sr

        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
        f0 = pitch.selected_array["frequency"]
        f0v = f0[f0 > 0]
        f0_mean = float(np.mean(f0v)) if f0v.size else 0.0
        f0_std = float(np.std(f0v)) if f0v.size else 0.0

        try:
            pp = call(snd, "To PointProcess (periodic, cc)", 75.0, 500.0)
            jitter = float(call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3))
            shimmer = float(
                call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6)
            )
        except Exception:
            jitter, shimmer = 0.0, 0.0
        jitter = 0.0 if np.isnan(jitter) else jitter
        shimmer = 0.0 if np.isnan(shimmer) else shimmer

        # energy / pauses
        rms = librosa.feature.rms(y=audio.astype(np.float32), frame_length=1024,
                                  hop_length=256)[0]
        energy_cv = float(np.std(rms) / (np.mean(rms) + 1e-8))
        thr = 0.15 * float(np.mean(rms) + 1e-8)
        silent = rms < thr
        pause_ratio = float(np.mean(silent))
        # crude syllable rate via onset envelope peaks
        onset_env = librosa.onset.onset_strength(y=audio.astype(np.float32), sr=sr)
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3,
                                       post_avg=5, delta=0.2, wait=5)
        speaking_rate = float(len(peaks) / dur) if dur > 0 else 0.0
        flatness = float(np.mean(librosa.feature.spectral_flatness(
            y=audio.astype(np.float32))))

        return {
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "jitter_local": jitter,
            "shimmer_local": shimmer,
            "speaking_rate_sylps": speaking_rate,
            "pause_ratio": pause_ratio,
            "energy_cv": energy_cv,
            "spectral_flatness": flatness,
        }

    def _analyze(self, audio: np.ndarray, sr: int, ctx: dict[str, Any]) -> DetectorResult:
        feats = self.extract_features(audio, sr)
        z: dict[str, float] = {}
        for k, val in feats.items():
            mean, std = self._baseline.get(k, (val, 1.0))
            z[k] = abs(val - mean) / (std + 1e-8)
        # robust aggregate: mean of clipped z-scores mapped to 0..1
        zc = np.clip(np.array(list(z.values())), 0, 4)
        anomaly = float(np.clip(zc.mean() / 3.0, 0.0, 1.0))
        return DetectorResult(
            name=self.name, kind=self.kind, score=anomaly, available=True,
            detail={
                "features": {k: round(v, 4) for k, v in feats.items()},
                "z_scores": {k: round(v, 3) for k, v in z.items()},
                "baseline": self._baseline_source,
            },
            note="" if "json" in self._baseline_source else
                 "using built-in priors — run scripts/build_baseline.py for a real baseline",
        )
