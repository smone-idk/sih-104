"""Band floors — the rule that lets fusion express the clone case (§6 + policy).

A weighted linear blend is additive and cannot say "synthetic AND matches the
enrolled speaker is worse than either alone". Without a floor, a successful clone
of the target scores BELOW crude TTS in a stranger's voice, because matching the
profile correctly lowers the speaker component. These tests pin the fix.
"""
from __future__ import annotations

import glob

import numpy as np
import pytest

from voiceshield.config import get_settings
from voiceshield.fusion.scorer import ComponentInput, fuse
from voiceshield.ingest.audio import load_audio
from voiceshield.pipeline import analyze_file
from voiceshield.policy.rules import apply_band_floors, rules_for
from voiceshield.stream import scenarios as scen


# --- unit: the rule itself --------------------------------------------
def test_cloned_voice_floors_band_to_high():
    band, applied = apply_band_floors("MEDIUM", "CLONED_VOICE")
    assert band == "HIGH"
    assert [a.code for a in applied] == ["cloned_voice_floor"]
    assert applied[0].from_band == "MEDIUM" and applied[0].to_band == "HIGH"
    assert "cannot represent this interaction" in applied[0].description


def test_floor_never_lowers_a_band():
    band, applied = apply_band_floors("HIGH", "CLONED_VOICE")
    assert band == "HIGH"
    assert applied == []          # already at/above the floor: no rule recorded


def test_other_verdicts_are_untouched():
    for verdict in ("CONSISTENT", "SPEAKER_MISMATCH", "SYNTHETIC_OTHER",
                    "SYNTHETIC_SUSPECTED", "INDETERMINATE"):
        for band in ("LOW", "MEDIUM", "HIGH"):
            out, applied = apply_band_floors(band, verdict)
            assert out == band and applied == []


def test_floors_can_be_disabled_and_thresholds_are_config():
    s = get_settings().model_copy(update={"enable_band_floors": False})
    band, applied = apply_band_floors("MEDIUM", "CLONED_VOICE", s)
    assert band == "MEDIUM" and applied == []
    # floor band is a config value, not a literal
    s2 = get_settings().model_copy(update={"cloned_voice_band_floor": "MEDIUM"})
    assert rules_for(s2)[0].floor_band == "MEDIUM"
    assert apply_band_floors("LOW", "CLONED_VOICE", s2)[0] == "MEDIUM"


def test_score_is_not_modified_by_the_floor():
    """The floor raises the BAND only. The number stays what fusion computed."""
    r = fuse({
        "voice_authenticity": ComponentInput(1.0),
        "speaker_consistency": ComponentInput(0.05),
        "prosody_anomaly": ComponentInput(0.4),
    })
    before = r.score
    r.band, r.floors_applied = apply_band_floors(r.band_from_score, "CLONED_VOICE")
    assert r.score == before
    assert r.band == "HIGH" and r.band_from_score != "HIGH"
    d = r.as_dict()
    assert d["score"] == pytest.approx(before, abs=0.005)   # as_dict rounds to 2dp
    assert d["band"] == "HIGH" and d["floors_applied"]
    assert d["band_from_score"] == "MEDIUM"                 # the un-floored band survives


# --- integration: the gate --------------------------------------------
@pytest.fixture(scope="module")
def enrolled_ctx():
    from voiceshield.ml.detectors.speaker_ecapa import SpeakerConsistencyDetector

    det = SpeakerConsistencyDetector()
    det.load()
    if not det.available:
        pytest.skip("ECAPA unavailable")
    refs = sorted((get_settings().demo_assets_dir / "genuine").glob("enrolled_1272*.wav"))[:3]
    if not refs:
        pytest.skip("demo corpus not built")
    emb = np.mean([det.embed(*load_audio(r)) for r in refs], axis=0)
    return {"enrolled_embedding": (emb / np.linalg.norm(emb)).astype("float32")}


def test_both_cloned_scenarios_land_high_without_context(enrolled_ctx):
    """THE gate: a judge uploading a clone with no scam script must still get
    HIGH. Context components are absent here — Phase 3 is not wired."""
    for sid in ("ceo_transfer_cloned", "bank_otp_cloned"):
        sc = scen.get(sid)
        if not sc.path().exists():
            pytest.skip("demo corpus not built")
        d = analyze_file(sc.path(), ctx=dict(enrolled_ctx)).as_dict()
        assert d["voice_verdict"] == "CLONED_VOICE", sid
        assert d["band"] == "HIGH", (sid, d["score"], d["band"])
        assert "transaction_context" in d["fusion"]["unavailable_components"]
        assert [f["code"] for f in d["fusion"]["floors_applied"]] == ["cloned_voice_floor"]


def test_every_cloned_clip_lands_high(enrolled_ctx):
    clips = sorted(glob.glob(str(get_settings().demo_assets_dir / "cloned" / "*.wav")))
    if not clips:
        pytest.skip("cloned tier not built")
    bad = []
    for f in clips:
        d = analyze_file(f, ctx=dict(enrolled_ctx)).as_dict()
        if d["band"] != "HIGH":
            bad.append((f.split("/")[-1], d["score"], d["band"], d["voice_verdict"]))
    assert not bad, f"cloned clips below HIGH: {bad}"


def test_genuine_control_stays_low(enrolled_ctx):
    """The floor must not drag honest calls upward."""
    sc = scen.get("genuine_control")
    if not sc.path().exists():
        pytest.skip("demo corpus not built")
    d = analyze_file(sc.path(), ctx=dict(enrolled_ctx)).as_dict()
    assert d["band"] == "LOW"
    assert d["fusion"]["floors_applied"] == []


def test_synthetic_speaker_deadband_still_reports_synthesis():
    """Speaker similarity between the thresholds must not discard the synthetic
    signal — that would report INDETERMINATE on an obvious TTS clip."""
    from voiceshield.fusion.findings import derive_findings

    s = get_settings()
    mid = (s.speaker_match_threshold + s.speaker_mismatch_threshold) / 2
    findings, verdict = derive_findings({
        "voice_authenticity": ComponentInput(0.99, kind="pretrained"),
        "speaker_consistency": ComponentInput(mid, kind="pretrained"),
    })
    assert verdict == "SYNTHETIC_SUSPECTED"
    assert "synthetic_speech" in {f.code for f in findings}
