"""DSP anti-spoofing heuristic — the documented fallback for AASIST (§2 Tier B).

This is NOT a trained model. It looks at three cues that vocoded / TTS speech
tends to leave behind:

  * phase linearity      — neural vocoders produce unusually smooth group delay
  * spectral-rolloff regularity — synthetic speech has a very stable rolloff
  * vocoder-band energy ratio   — excess energy in the 4–8 kHz band relative to
                                  the speech-formant band

Each cue is squashed to 0..1 and averaged. The UI and README must state that
this is a placeholder for a trained anti-spoofing model.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import SyntheticSpeechDetector
from .base import DetectorResult


class HeuristicSyntheticDetector(SyntheticSpeechDetector):
    kind = "heuristic"

    def load(self) -> None:
        self.device = "cpu"
        try:
            import scipy.signal  # noqa: F401
            self.available = True
        except Exception as exc:
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _analyze(self, audio: np.ndarray, sr: int, ctx: dict[str, Any]) -> DetectorResult:
        import librosa
        from scipy.signal import stft

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        x = audio.astype(np.float32)
        x = x / (np.max(np.abs(x)) + 1e-8)

        f, _, Z = stft(x, fs=sr, nperseg=512, noverlap=384)
        mag = np.abs(Z) + 1e-9
        phase = np.angle(Z)

        # 1. phase linearity: variance of unwrapped group delay across frames
        gd = np.diff(np.unwrap(phase, axis=0), axis=0)
        gd_var = float(np.mean(np.var(gd, axis=1)))
        phase_cue = float(np.clip(1.0 - gd_var / 2.5, 0.0, 1.0))

        # 2. spectral-rolloff regularity: low CV of rolloff => synthetic
        rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.9)[0]
        rolloff_cv = float(np.std(rolloff) / (np.mean(rolloff) + 1e-8))
        rolloff_cue = float(np.clip(1.0 - rolloff_cv / 0.25, 0.0, 1.0))

        # 3. vocoder-band energy ratio: (4-8kHz) / (300-3400Hz)
        def band(lo: float, hi: float) -> float:
            m = (f >= lo) & (f < hi)
            return float(np.mean(mag[m, :] ** 2)) if m.any() else 0.0

        voc_ratio = band(4000, min(8000, sr / 2)) / (band(300, 3400) + 1e-9)
        voc_cue = float(np.clip((voc_ratio - 0.05) / 0.35, 0.0, 1.0))

        score = float(np.mean([phase_cue, rolloff_cue, voc_cue]))
        return DetectorResult(
            name=self.name, kind=self.kind, score=score, available=True,
            detail={
                "phase_linearity_cue": round(phase_cue, 4),
                "rolloff_regularity_cue": round(rolloff_cue, 4),
                "vocoder_band_ratio_cue": round(voc_cue, 4),
                "raw": {
                    "group_delay_var": round(gd_var, 5),
                    "rolloff_cv": round(rolloff_cv, 5),
                    "vocoder_band_ratio": round(voc_ratio, 5),
                },
            },
            note="placeholder for a trained anti-spoofing model (AASIST) — DSP heuristic only",
        )
