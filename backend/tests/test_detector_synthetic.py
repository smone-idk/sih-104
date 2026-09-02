"""Known-input/known-output fixtures for the synthetic-speech detectors (§14)."""
from __future__ import annotations

import numpy as np
import pytest

from voiceshield.ml.detectors.synthetic_heuristic import HeuristicSyntheticDetector
from voiceshield.ml.detectors.synthetic_aasist import AasistSyntheticDetector


@pytest.fixture(scope="module")
def heur() -> HeuristicSyntheticDetector:
    d = HeuristicSyntheticDetector()
    d.load()
    return d


def test_heuristic_kind_and_bounds(heur, white_noise, sr):
    assert heur.kind == "heuristic"
    r = heur.analyze(white_noise, sr)
    assert r.available and 0.0 <= r.score <= 1.0
    assert set(r.detail) >= {"phase_linearity_cue", "rolloff_regularity_cue",
                             "vocoder_band_ratio_cue"}


def test_heuristic_deterministic(heur, voiced_like, sr):
    a = heur.analyze(voiced_like, sr).score
    b = heur.analyze(voiced_like.copy(), sr).score
    assert a == pytest.approx(b, abs=1e-9)


def test_heuristic_distinguishes_different_inputs(heur, pure_tone, white_noise, sr):
    # contract only: this layer is a documented placeholder for a trained
    # anti-spoofing model, so we assert it responds to the signal, not a
    # particular semantic ordering it cannot reliably deliver.
    tone = heur.analyze(pure_tone, sr).score
    noise = heur.analyze(white_noise, sr).score
    assert abs(tone - noise) > 0.05
    assert "placeholder" in heur.analyze(pure_tone, sr).note


def test_heuristic_regression_guard_fixed_tone(heur, pure_tone, sr):
    # exact closed-form input -> stable score across runs
    scores = {round(heur.analyze(pure_tone.copy(), sr).score, 6) for _ in range(3)}
    assert len(scores) == 1
    assert 0.0 <= scores.pop() <= 1.0


@pytest.fixture(scope="module")
def aasist() -> AasistSyntheticDetector:
    d = AasistSyntheticDetector()
    d.load()
    return d


def test_aasist_bounds_and_determinism(aasist, pure_tone, white_noise, sr):
    if not aasist.available:
        pytest.skip(f"AASIST weights unavailable: {aasist.load_error}")
    assert aasist.kind == "pretrained"
    r1 = aasist.analyze(pure_tone, sr)
    r2 = aasist.analyze(pure_tone.copy(), sr)
    assert 0.0 <= r1.score <= 1.0
    assert r1.score == pytest.approx(r2.score, abs=1e-4)
    rn = aasist.analyze(white_noise, sr)
    assert 0.0 <= rn.score <= 1.0
    assert "spoof_probability" in r1.detail
