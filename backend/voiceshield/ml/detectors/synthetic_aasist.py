"""Pretrained AASIST anti-spoofing (§2 Tier B).

AASIST (clovaai, MIT) trained on ASVspoof2019 LA. We do NOT re-implement the
architecture from memory — scripts/fetch_models.py vendors the exact model
definition from the MIT-licensed upstream repo into models/aasist/ alongside
AASIST.pth, so the graph always matches the weights.

Run as plain PyTorch (no ONNX — graph-attention ops export badly and the model
is tiny). If the weights or vendored code are missing, this detector stays
unavailable and the registry falls back to HeuristicSyntheticDetector.
"""
from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

from .base import SyntheticSpeechDetector
from .base import DetectorResult
from ...config import get_settings


class AasistSyntheticDetector(SyntheticSpeechDetector):
    kind = "pretrained"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._torch = None

    def load(self) -> None:
        s = get_settings()
        self.device = s.device
        aasist_dir = s.model_cache_dir / "aasist"
        weights = s.aasist_weights_path
        try:
            import torch

            self._torch = torch
            if not weights.exists():
                raise FileNotFoundError(f"missing {weights}")
            model_py = aasist_dir / "models" / "AASIST.py"
            config_json = aasist_dir / "config" / "AASIST.conf"
            if not model_py.exists():
                raise FileNotFoundError(f"missing vendored model code {model_py}")

            if str(aasist_dir) not in sys.path:
                sys.path.insert(0, str(aasist_dir))
            from models.AASIST import Model as AasistModel  # type: ignore

            model_config = {
                "architecture": "AASIST",
                "nb_samp": 64600,
                "first_conv": 128,
                "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                "gat_dims": [64, 32],
                "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                "temperatures": [2.0, 2.0, 100.0, 100.0],
            }
            if config_json.exists():
                try:
                    model_config = json.loads(config_json.read_text())["model_config"]
                except Exception:
                    pass

            model = AasistModel(model_config)
            # clovaai's own checkpoint (MIT); plain state_dict pickle.
            state = torch.load(weights, map_location=self.device, weights_only=False)
            model.load_state_dict(state, strict=True)
            model.eval().to(self.device)
            self._model = model
            self.available = True
        except Exception as exc:
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"

    def warmup(self) -> None:
        if self.available:
            self._analyze(np.zeros(64600, dtype=np.float32), 16000, {})

    def _analyze(self, audio: np.ndarray, sr: int, ctx: dict[str, Any]) -> DetectorResult:
        torch = self._torch
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        x = audio.astype(np.float32)
        if sr != 16000:
            import resampy

            x = resampy.resample(x, sr, 16000)
        nb = 64600  # AASIST fixed input (~4.03 s @ 16 kHz)
        if len(x) < nb:
            reps = int(np.ceil(nb / len(x))) if len(x) else 1
            x = np.tile(x, reps)[:nb]
        else:
            x = x[:nb]
        with torch.no_grad():
            t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            _, out = self._model(t)
            # AASIST head: index 1 = bonafide logit, 0 = spoof
            prob = torch.softmax(out, dim=1)[0]
            spoof_prob = float(prob[0].cpu())
        return DetectorResult(
            name=self.name, kind=self.kind, score=spoof_prob, available=True,
            detail={
                "spoof_probability": round(spoof_prob, 4),
                "bonafide_probability": round(1.0 - spoof_prob, 4),
                "model": "AASIST / ASVspoof2019-LA (pretrained, clovaai)",
            },
        )
