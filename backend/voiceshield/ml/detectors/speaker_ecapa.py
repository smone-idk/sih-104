"""Speaker consistency via SpeechBrain ECAPA-TDNN (pretrained speaker verification).

Real speaker verification: embed the window, cosine-compare against an enrolled
profile embedding. Runs fine on CPU. If no profile is enrolled for the session
the layer reports itself unavailable so fusion redistributes its weight
(§6) — it never substitutes a fake similarity.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..detectors.base import Detector, DetectorResult
from ...config import get_settings


class SpeakerConsistencyDetector(Detector):
    name = "speaker_consistency"
    kind = "pretrained"
    feeds = "speaker_consistency"

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def load(self) -> None:
        s = get_settings()
        self.device = s.device
        try:
            import torch  # noqa: F401
            from speechbrain.inference.speaker import EncoderClassifier

            if not s.ecapa_dir.exists():
                raise FileNotFoundError(
                    f"ECAPA weights not found at {s.ecapa_dir} — run scripts/fetch_models.py"
                )
            self._model = EncoderClassifier.from_hparams(
                source=str(s.ecapa_dir),
                savedir=str(s.ecapa_dir),
                run_opts={"device": self.device},
            )
            self.available = True
        except Exception as exc:
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Public helper used by voice-profile enrollment."""
        import torch

        s = get_settings()
        wav = _to_16k_mono(audio, sr, s.sample_rate)
        with torch.no_grad():
            t = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
            emb = self._model.encode_batch(t).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _analyze(self, audio: np.ndarray, sr: int, ctx: dict[str, Any]) -> DetectorResult:
        enrolled = ctx.get("enrolled_embedding")
        if enrolled is None:
            return DetectorResult(
                name=self.name, kind=self.kind, score=None, available=False,
                note="layer unavailable — no enrolled profile",
            )
        emb = self.embed(audio, sr)
        enrolled = np.asarray(enrolled, dtype=np.float32)
        enrolled = enrolled / (np.linalg.norm(enrolled) + 1e-8)
        cos = float(np.dot(emb, enrolled))
        # suspicion score: low similarity => high suspicion
        score = float(np.clip((0.55 - cos) / 0.55, 0.0, 1.0))
        return DetectorResult(
            name=self.name, kind=self.kind, score=score, available=True,
            detail={
                "cosine_similarity": round(cos, 4),
                "enrolled_speaker": ctx.get("enrolled_speaker_name", "unknown"),
            },
        )


def _to_16k_mono(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target:
        import resampy

        audio = resampy.resample(audio.astype(np.float32), sr, target)
    return audio.astype(np.float32)
