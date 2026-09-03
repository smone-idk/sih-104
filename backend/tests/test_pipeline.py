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


def test_result_shape_and_explainability(real_speech):
    res = analyze_audio(real_speech, SR, source="test").as_dict()
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
    # context IS wired from Phase 3, so transcript + quote fields must be present
    assert "transcript" in res and "context_quotes" in res
    # caller_trust has no directory record here -> still unavailable, and its
    # weight must be redistributed rather than substituted
    assert "caller_trust" in res["fusion"]["unavailable_components"]
    assert res["fusion"]["redistributed"] is True
    eff = sum(c["effective_weight"] for c in comps)
    assert eff == pytest.approx(1.0, abs=1e-3)


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


def test_reproducible_on_real_audio(real_speech):
    a1 = analyze_audio(real_speech, SR).as_dict()["score"]
    a2 = analyze_audio(real_speech.copy(), SR).as_dict()["score"]
    assert a1 == pytest.approx(a2, abs=0.5)


def test_findings_and_verdict_present(real_speech):
    res = analyze_audio(real_speech, SR).as_dict()
    assert res["voice_verdict"] in {
        "CLONED_VOICE", "SYNTHETIC_OTHER", "SPEAKER_MISMATCH", "CONSISTENT",
        "SYNTHETIC_SUSPECTED", "NO_SYNTHETIC_MARKERS", "SPEAKER_CONSISTENT",
        "INDETERMINATE"}
    assert res["findings"]
    for f in res["findings"]:
        assert {"code", "label", "severity", "kind", "evidence"} <= set(f)
    assert res["vad_backend"] in {"silero", "energy"}


def test_speaker_mismatch_is_a_distinct_finding(enrolled_clip_path, other_clip_path):
    """A different human speaker must surface as speaker_mismatch, NOT as
    synthetic_speech — the two findings are different claims."""
    from voiceshield.ml.detectors.speaker_ecapa import SpeakerConsistencyDetector
    from voiceshield.ingest.audio import load_audio

    det = SpeakerConsistencyDetector()
    det.load()
    if not det.available:
        pytest.skip("ECAPA unavailable")
    enrolled_audio, sr = load_audio(enrolled_clip_path)
    emb = det.embed(enrolled_audio, sr)

    other, _ = load_audio(other_clip_path)
    res = analyze_audio(other, SR, ctx={"enrolled_embedding": emb}).as_dict()
    codes = {f["code"] for f in res["findings"]}
    assert "speaker_mismatch" in codes
    assert res["voice_verdict"] in {"SPEAKER_MISMATCH", "SYNTHETIC_OTHER"}


def test_same_clip_is_reproducible(voiced_like):
    a = analyze_audio(voiced_like, SR).as_dict()["score"]
    b = analyze_audio(voiced_like.copy(), SR).as_dict()["score"]
    assert a == pytest.approx(b, abs=0.5)


def test_silence_flagged_not_meaningful(silence):
    res = analyze_audio(silence, SR).as_dict()
    assert res["n_speech_windows"] == 0
    assert any("no speech" in w for w in res["warnings"])


def test_windows_timeline_present_for_long_clip(real_speech):
    res = analyze_audio(real_speech, SR).as_dict()
    assert res["n_windows"] >= 5
    speech = [w for w in res["windows"] if w["is_speech"]]
    assert speech and all(w["raw_score"] is not None for w in speech)
    assert all(w["ema_score"] is not None for w in speech)
