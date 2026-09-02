"""Speaker-consistency detector fixtures (§14)."""
from __future__ import annotations

import numpy as np
import pytest

from voiceshield.ml.detectors.speaker_ecapa import SpeakerConsistencyDetector


@pytest.fixture(scope="module")
def ecapa() -> SpeakerConsistencyDetector:
    d = SpeakerConsistencyDetector()
    d.load()
    return d


def test_unavailable_without_enrolled_profile(ecapa, voiced_like, sr):
    if not ecapa.available:
        pytest.skip(f"ECAPA unavailable: {ecapa.load_error}")
    r = ecapa.analyze(voiced_like, sr, ctx={})       # no enrolled_embedding
    assert r.available is False
    assert "no enrolled profile" in r.note


def test_same_signal_high_similarity(ecapa, voiced_like, sr):
    if not ecapa.available:
        pytest.skip("ECAPA unavailable")
    half = len(voiced_like) // 2
    enrolled = ecapa.embed(voiced_like[:half], sr)
    r = ecapa.analyze(voiced_like[half:], sr, ctx={"enrolled_embedding": enrolled})
    assert r.available
    assert r.detail["cosine_similarity"] > 0.6   # same source
    assert r.score < 0.4                          # low suspicion


def test_different_signal_lower_similarity(ecapa, voiced_like, white_noise, sr):
    if not ecapa.available:
        pytest.skip("ECAPA unavailable")
    half = len(voiced_like) // 2
    enrolled = ecapa.embed(voiced_like[:half], sr)
    same = ecapa.analyze(voiced_like[half:], sr,
                         ctx={"enrolled_embedding": enrolled})
    other = ecapa.analyze(white_noise, sr,
                          ctx={"enrolled_embedding": enrolled})
    assert other.detail["cosine_similarity"] < same.detail["cosine_similarity"]


def test_embedding_is_unit_norm(ecapa, voiced_like, sr):
    if not ecapa.available:
        pytest.skip("ECAPA unavailable")
    e = ecapa.embed(voiced_like, sr)
    assert np.linalg.norm(e) == pytest.approx(1.0, abs=1e-4)
