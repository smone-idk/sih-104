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

        # Anti-spoofing slot: prefer the pretrained model, fall back to DSP.
        aasist = AasistSyntheticDetector()
        aasist.load()
        if aasist.available:
            self._add(aasist)
        else:
            log.warning("AASIST unavailable (%s) — falling back to DSP heuristic",
                        aasist.load_error)
            heur = HeuristicSyntheticDetector()
            heur.load()
            self._add(heur)

        for cls in (SpeakerConsistencyDetector, ProsodyAnomalyDetector):
            det = cls()
            det.load()
            self._add(det)

        self._loaded = True
        return self

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
