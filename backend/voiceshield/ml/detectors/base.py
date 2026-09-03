"""Detector abstraction.

Every analysis layer implements `Detector`. The important field is `kind`:
the UI renders a badge straight from it and no detector is allowed to
misreport it.

  trained     — a model we trained/fine-tuned on task data (none in v1)
  pretrained  — a third-party trained model used as-is (ECAPA, Whisper, AASIST)
  heuristic   — hand-built DSP / rules, NOT a learned model
  simulated   — transport or data is faked (analysis on top may still be real)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

DetectorKind = Literal["trained", "pretrained", "heuristic", "simulated"]


@dataclass
class DetectorResult:
    name: str
    kind: DetectorKind
    score: float | None            # 0..1, higher = more suspicious / more anomalous
    available: bool                # False => fusion redistributes this weight
    detail: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    note: str = ""                 # shown on the card, e.g. "no enrolled profile"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "score": self.score,
            "available": self.available,
            "detail": self.detail,
            "latency_ms": round(self.latency_ms, 2),
            "note": self.note,
        }


class Detector(ABC):
    #: stable identifier used by the registry, fusion and the API
    name: str = "detector"
    #: honest declaration — see module docstring
    kind: DetectorKind = "heuristic"
    #: which fusion component this detector feeds (see config.fusion_weights)
    feeds: str = ""
    #: False => the detector still runs and is displayed with its badge, but
    #: carries ZERO fusion weight. Used to keep a known-broken layer visible
    #: and measurable without letting it move the score (see AASIST,
    #: LIMITATIONS.md §3). Never blend a detector we cannot explain.
    contributes: bool = True

    def __init__(self) -> None:
        self.available: bool = False
        self.load_error: str = ""
        self.device: str = "cpu"

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        """Load weights once at startup. Must set self.available.

        Implementations should catch their own errors, store a message on
        self.load_error and leave self.available = False rather than raising,
        so one broken detector never takes down the process.
        """
        self.available = True

    def warmup(self) -> None:
        """Optional: run one dummy inference so the first real call is fast."""

    # --- inference -----------------------------------------------------
    @abstractmethod
    def _analyze(self, audio: np.ndarray, sr: int, ctx: dict[str, Any]) -> DetectorResult:
        ...

    def analyze(self, audio: np.ndarray, sr: int, ctx: dict[str, Any] | None = None) -> DetectorResult:
        ctx = ctx or {}
        if not self.available:
            return DetectorResult(
                name=self.name, kind=self.kind, score=None, available=False,
                note=self.load_error or "detector unavailable",
            )
        t0 = time.perf_counter()
        try:
            res = self._analyze(audio, sr, ctx)
        except Exception as exc:  # never let a detector crash the pipeline
            return DetectorResult(
                name=self.name, kind=self.kind, score=None, available=False,
                note=f"runtime error: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        res.latency_ms = (time.perf_counter() - t0) * 1000
        res.name, res.kind = self.name, self.kind
        return res

    # --- introspection ----------------------------------------------
    def info(self) -> dict[str, Any]:
        from ...config import get_settings

        w = get_settings().fusion_weights().get(self.feeds, 0.0) if self.feeds else 0.0
        return {
            "name": self.name,
            "kind": self.kind,
            "feeds": self.feeds if self.contributes else None,
            "contributes": self.contributes,
            "fusion_weight": round(w if self.contributes else 0.0, 4),
            "available": self.available,
            "device": self.device,
            "load_error": self.load_error,
            "model_id": getattr(self, "model_id", ""),
            "training_data": getattr(self, "training_data", ""),
            "licence": getattr(self, "licence", ""),
            "note": getattr(self, "display_note", ""),
        }


class SyntheticSpeechDetector(Detector):
    """Marker base class for the anti-spoofing slot (§2 Tier B).

    Swapping AASIST for a trained model later is a one-line registry change:
    register a different subclass of this under the same `name`.
    """

    name = "synthetic_speech"
    feeds = "voice_authenticity"
