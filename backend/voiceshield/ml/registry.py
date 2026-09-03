"""DetectorRegistry — the single source of truth for which analysis layers
exist, what kind each is, and whether it loaded.

The API renders detector badges straight from `Detector.kind`. Fusion asks the
registry for results and redistributes weight for any layer that reports
`available == False`.
"""
from __future__ import annotations

import logging
from typing import Any

from .detectors.base import Detector
from .detectors.prosody import ProsodyAnomalyDetector
from .detectors.speaker_ecapa import SpeakerConsistencyDetector
from .detectors.synthetic_aasist import AasistSyntheticDetector
from .detectors.synthetic_antideepfake import AntiDeepfakeDetector
from .detectors.synthetic_heuristic import HeuristicSyntheticDetector

log = logging.getLogger("voiceshield.registry")


class DetectorRegistry:
    def __init__(self) -> None:
        self._detectors: dict[str, Detector] = {}
        self._loaded = False

    # --- construction ------------------------------------------------
    def build(self) -> "DetectorRegistry":
        """Instantiate + load every detector once. Idempotent."""
        if self._loaded:
            return self

        # --- anti-spoofing slot -------------------------------------
        # Primary: AntiDeepfake (SSL, post-trained on 74k h multi-corpus).
        # It carries the voice_authenticity fusion weight.
        primary = AntiDeepfakeDetector()
        primary.load()
        if primary.available:
            self._add(primary)
        else:
            log.warning("AntiDeepfake unavailable (%s) — falling back to DSP heuristic",
                        primary.load_error)
            heur = HeuristicSyntheticDetector()
            heur.load()
            self._add(heur)

        # AASIST stays LOADED and DISPLAYED as a second badged detector with
        # ZERO fusion weight. It is measurably broken on modern TTS
        # (LIMITATIONS.md §3); blending it back in would reintroduce exactly
        # the noise we diagnosed out. Kept so the Evaluation page can show
        # both models on the same clips.
        aasist = AasistSyntheticDetector()
        aasist.load()
        if primary.available:
            aasist.contributes = False
            aasist.display_note = (
                "zero fusion weight — retained as a measured baseline; fails to "
                "separate modern TTS on our corpus (LIMITATIONS.md §3)"
            )
        self._add(aasist)

        for cls in (SpeakerConsistencyDetector, ProsodyAnomalyDetector):
            det = cls()
            det.load()
            self._add(det)

        self._loaded = True
        self._log_vram()
        return self

    # --- VRAM accounting -------------------------------------------
    def vram(self) -> dict[str, float]:
        """Measured CUDA memory, MB. Zeros on CPU."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0}
            return {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
                "total_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
            }
        except Exception:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0}

    def _log_vram(self) -> None:
        v = self.vram()
        if v["total_mb"]:
            log.info("VRAM after model load: %.0f MB allocated / %.0f MB reserved "
                     "(device total %.0f MB, budget 5000 MB)",
                     v["allocated_mb"], v["reserved_mb"], v["total_mb"])
            if v["reserved_mb"] > 5000:
                log.warning("VRAM reserved %.0f MB exceeds the 5 GB ceiling",
                            v["reserved_mb"])

    def warmup(self) -> None:
        for d in self._detectors.values():
            if d.available:
                try:
                    d.warmup()
                except Exception as exc:  # pragma: no cover
                    log.warning("warmup failed for %s: %s", d.name, exc)

    def _add(self, det: Detector) -> None:
        self._detectors[det.name] = det

    # --- access ----------------------------------------------------
    def get(self, name: str) -> Detector | None:
        return self._detectors.get(name)

    def all(self) -> list[Detector]:
        return list(self._detectors.values())

    def available(self) -> list[Detector]:
        return [d for d in self._detectors.values() if d.available]

    # --- introspection --------------------------------------------
    def inventory(self) -> list[dict[str, Any]]:
        return [d.info() for d in self._detectors.values()]


_REGISTRY: DetectorRegistry | None = None


def get_registry() -> DetectorRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DetectorRegistry().build()
    return _REGISTRY
