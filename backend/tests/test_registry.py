"""Phase 0 tests: the registry builds, kinds are honest, unavailable detectors
degrade instead of crashing. Detector known-input/known-output fixtures land in
Phase 1 alongside the batch pipeline.
"""
from __future__ import annotations

import numpy as np

from voiceshield.ml.registry import DetectorRegistry
from voiceshield.ml.detectors.synthetic_heuristic import HeuristicSyntheticDetector

VALID_KINDS = {"trained", "pretrained", "heuristic", "simulated"}


def test_registry_builds_and_has_synthetic_slot():
    reg = DetectorRegistry().build()
    names = {d.name for d in reg.all()}
    assert "synthetic_speech" in names
    assert "speaker_consistency" in names
    assert "prosody_anomaly" in names


def test_all_kinds_are_declared_and_valid():
    reg = DetectorRegistry().build()
    for d in reg.all():
        assert d.kind in VALID_KINDS, d.name


def test_unavailable_detector_returns_none_not_crash():
    class Broken(HeuristicSyntheticDetector):
        def load(self) -> None:  # noqa: D401
            self.available = False
            self.load_error = "simulated failure"

    d = Broken()
    d.load()
    res = d.analyze(np.zeros(16000, dtype=np.float32), 16000)
    assert res.available is False
    assert res.score is None
    assert "simulated failure" in res.note


def test_heuristic_synthetic_detector_produces_bounded_score():
    d = HeuristicSyntheticDetector()
    d.load()
    if not d.available:  # scipy missing in a bare env
        return
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16000 * 3).astype(np.float32) * 0.1
    res = d.analyze(audio, 16000)
    assert res.available is True
    assert 0.0 <= res.score <= 1.0
    assert res.kind == "heuristic"
