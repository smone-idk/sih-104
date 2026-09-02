"""Fusion math + weight redistribution + bands (§6)."""
from __future__ import annotations

import pytest

from voiceshield.config import get_settings
from voiceshield.fusion.scorer import ComponentInput, band_for, fuse


def _all(v: float) -> dict[str, ComponentInput]:
    return {n: ComponentInput(value=v) for n in get_settings().fusion_weights()}


def test_all_components_equal_gives_that_score():
    r = fuse(_all(0.5))
    assert r.score == pytest.approx(50.0, abs=1e-6)
    assert r.redistributed is False
    assert r.band == "MEDIUM"


def test_known_weighted_combination():
    s = get_settings()
    w = s.fusion_weights()  # 0.30/0.20/0.10/0.10/0.20/0.10
    inp = {
        "voice_authenticity": ComponentInput(0.9),
        "speaker_consistency": ComponentInput(0.1),
        "prosody_anomaly": ComponentInput(0.5),
        "caller_trust": ComponentInput(0.0),
        "transaction_context": ComponentInput(1.0),
        "behavioural_risk": ComponentInput(0.2),
    }
    expected = 100 * (0.30*0.9 + 0.20*0.1 + 0.10*0.5 + 0.10*0.0 + 0.20*1.0 + 0.10*0.2)
    assert fuse(inp, w).score == pytest.approx(expected, abs=1e-6)


def test_redistribution_when_component_unavailable():
    inp = _all(0.4)
    inp["speaker_consistency"] = ComponentInput(value=None, available=False,
                                                note="no enrolled profile")
    r = fuse(inp)
    assert r.redistributed is True
    # remaining 5 components still all 0.4 -> score stays 40 after renormalisation
    assert r.score == pytest.approx(40.0, abs=1e-6)
    sp = next(c for c in r.components if c.name == "speaker_consistency")
    assert sp.effective_weight == 0.0 and sp.available is False
    assert sum(c.effective_weight for c in r.components) == pytest.approx(1.0, abs=1e-6)


def test_contributions_sorted_and_sum_to_score():
    r = fuse({
        "voice_authenticity": ComponentInput(0.8),
        "speaker_consistency": ComponentInput(0.2),
        "prosody_anomaly": ComponentInput(0.6),
        "caller_trust": ComponentInput(0.1),
        "transaction_context": ComponentInput(0.9),
        "behavioural_risk": ComponentInput(0.3),
    })
    pts = [c.contribution_points for c in r.components]
    assert pts == sorted(pts, reverse=True)
    assert sum(pts) == pytest.approx(r.score, abs=1e-6)


def test_bands():
    assert band_for(0) == "LOW"
    assert band_for(39.9) == "LOW"
    assert band_for(55) == "MEDIUM"
    assert band_for(70.1) == "HIGH"
    assert band_for(100) == "HIGH"
