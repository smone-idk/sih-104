"""Prosody detector fixtures (§14). The DSP is deterministic, so feature values
are asserted against known closed-form inputs within tolerance."""
from __future__ import annotations

import pytest

from voiceshield.ml.detectors.prosody import ProsodyAnomalyDetector


@pytest.fixture(scope="module")
def prosody() -> ProsodyAnomalyDetector:
    d = ProsodyAnomalyDetector()
    d.load()
    return d


def test_kind_and_score_bounds(prosody, voiced_like, sr):
    assert prosody.kind == "heuristic"
    r = prosody.analyze(voiced_like, sr)
    assert r.available and 0.0 <= r.score <= 1.0
    assert "features" in r.detail and "z_scores" in r.detail


def test_f0_estimate_matches_input(prosody, pure_tone, voiced_like, sr):
    f_tone = prosody.extract_features(pure_tone, sr)["f0_mean"]
    f_voiced = prosody.extract_features(voiced_like, sr)["f0_mean"]
    assert f_tone == pytest.approx(180.0, abs=6.0)      # 180 Hz tone
    assert f_voiced == pytest.approx(140.0, abs=12.0)   # 140 Hz carrier


def test_deterministic(prosody, voiced_like, sr):
    a = prosody.analyze(voiced_like, sr).score
    b = prosody.analyze(voiced_like.copy(), sr).score
    assert a == pytest.approx(b, abs=1e-9)


def test_steady_tone_has_low_f0_variance(prosody, pure_tone, voiced_like, sr):
    tone = prosody.extract_features(pure_tone, sr)
    voiced = prosody.extract_features(voiced_like, sr)
    assert tone["f0_std"] < voiced["f0_std"]            # vibrato widens F0 spread
