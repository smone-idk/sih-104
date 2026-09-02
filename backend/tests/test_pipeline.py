"""Pipeline / Phase 1 gate: two different inputs -> two different, explainable
scores; one shared analysis path."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voiceshield.config import get_settings
from voiceshield.pipeline import analyze_audio, analyze_file

SR = 16000
GENUINE = get_settings().demo_assets_dir / "genuine"


def _clip(prefix: str) -> Path | None:
    if not GENUINE.exists():
        return None
    hits = sorted(GENUINE.glob(f"{prefix}*.wav"))
    return hits[0] if hits else None


def test_result_shape_and_explainability(voiced_like):
    res = analyze_audio(voiced_like, SR, source="test").as_dict()
    assert 0.0 <= res["score"] <= 100.0
    assert res["band"] in {"LOW", "MEDIUM", "HIGH"}
    comps = res["fusion"]["components"]
    assert {c["name"] for c in comps} == {
        "voice_authenticity", "speaker_consistency", "prosody_anomaly",
        "caller_trust", "transaction_context", "behavioural_risk",
    }
    for c in comps:
        assert {"raw_value", "weight", "effective_weight",
                "contribution_points", "available", "note"} <= set(c)
    assert res["fusion"]["disclaimer"]
    assert res["latency_ms"]["total"] > 0
    # context components are not wired in Phase 1 -> weights redistributed
    assert res["fusion"]["redistributed"] is True
    assert "transaction_context" in res["fusion"]["unavailable_components"]


def test_two_different_clips_give_two_different_scores():
    """Phase 1 gate — real demo audio, two different LibriSpeech speakers."""
    a_path, b_path = _clip("enrolled_1272"), _clip("genuine_1462")
    if not (a_path and b_path):
        pytest.skip("demo assets not built — run scripts/build_demo_assets.py --tier genuine")
    a = analyze_file(a_path, ctx={"enrolled_embedding": None}).as_dict()
    b = analyze_file(b_path, ctx={"enrolled_embedding": None}).as_dict()
    assert a["score"] != b["score"], (a["score"], b["score"])
    assert abs(a["score"] - b["score"]) > 1.0
    # explainable: contributions present and summing to the score
    for r in (a, b):
        pts = [c["contribution_points"] for c in r["fusion"]["components"]]
        assert sum(pts) == pytest.approx(r["score"], abs=0.05)


def test_synthetic_fixtures_reproducible(voiced_like, white_noise):
    a1 = analyze_audio(voiced_like, SR).as_dict()["score"]
    a2 = analyze_audio(voiced_like.copy(), SR).as_dict()["score"]
    assert a1 == pytest.approx(a2, abs=0.5)


def test_same_clip_is_reproducible(voiced_like):
    a = analyze_audio(voiced_like, SR).as_dict()["score"]
    b = analyze_audio(voiced_like.copy(), SR).as_dict()["score"]
    assert a == pytest.approx(b, abs=0.5)


def test_silence_flagged_not_meaningful(silence):
    res = analyze_audio(silence, SR).as_dict()
    assert res["n_speech_windows"] == 0
    assert any("no speech" in w for w in res["warnings"])


def test_windows_timeline_present_for_long_clip(voiced_like):
    clip = np.tile(voiced_like, 2)  # 8 s -> multiple hops
    res = analyze_audio(clip, SR).as_dict()
    assert res["n_windows"] >= 5
    speech = [w for w in res["windows"] if w["is_speech"]]
    assert speech and all(w["raw_score"] is not None for w in speech)
    assert all(w["ema_score"] is not None for w in speech)
