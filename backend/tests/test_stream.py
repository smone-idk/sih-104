"""Streaming session + scenario tests (Phase 2).

The important property: streaming and batch are the SAME analysis (§14). If
these drift apart, the demo is lying about what it measures.
"""
from __future__ import annotations

import numpy as np
import pytest

from voiceshield.config import get_settings
from voiceshield.ingest.audio import load_audio
from voiceshield.pipeline import analyze_audio
from voiceshield.stream import scenarios as scen
from voiceshield.stream.session import StreamSession

SR = 16000


# --- scenarios ---------------------------------------------------------
def test_scenarios_include_a_genuine_control():
    """§13: a demo where a real call correctly scores LOW is the persuasive one."""
    tiers = {s.tier for s in scen.SCENARIOS}
    assert "genuine" in tiers and "cloned" in tiers and "synthetic" in tiers
    assert any(s.id == "genuine_control" for s in scen.SCENARIOS)


def test_every_scenario_is_backed_by_a_real_file():
    missing = [s.id for s in scen.SCENARIOS if not s.path().exists()]
    if missing:
        pytest.skip(f"demo corpus not fully built: {missing}")
    for s in scen.SCENARIOS:
        assert s.path().stat().st_size > 1000


def test_listing_reports_availability():
    for row in scen.listing():
        assert {"id", "title", "tier", "expected", "available"} <= set(row)


# --- session mechanics -------------------------------------------------
def test_windows_emitted_on_the_hop(real_speech):
    s = get_settings()
    sess = StreamSession(ctx={"enrolled_embedding": None})
    chunk = int(s.hop_seconds * SR)
    got = []
    for i in range(0, len(real_speech), chunk):
        got += sess.feed(real_speech[i:i + chunk])
    got += sess.flush()
    assert got, "expected at least one scored window"
    # windows start on the hop grid
    for w in got[:-1]:
        assert w.t_start == pytest.approx(w.index * s.hop_seconds, abs=1e-6)
    assert all(len(w.spectrum) == 24 for w in got)


def test_ema_is_monotonic_in_index_and_bounded(real_speech):
    sess = StreamSession(ctx={"enrolled_embedding": None})
    got = sess.feed(real_speech) + sess.flush()
    scored = [w for w in got if w.ema_score is not None]
    assert scored
    assert all(0 <= w.ema_score <= 100 for w in scored)
    assert [w.index for w in scored] == sorted(w.index for w in scored)


def test_silence_produces_no_score():
    sess = StreamSession(ctx={"enrolled_embedding": None})
    got = sess.feed(np.zeros(SR * 8, dtype=np.float32)) + sess.flush()
    assert got
    assert all(w.raw_score is None for w in got)
    assert sess.final_payload()["n_speech_windows"] == 0


def test_close_drops_the_buffer(real_speech):
    """§12 — audio is not retained after the session ends."""
    sess = StreamSession(ctx={"enrolled_embedding": None})
    sess.feed(real_speech)
    sess.close()
    assert sess._buf.size == 0
    assert sess.feed(real_speech) == []


def test_payload_shapes(real_speech):
    sess = StreamSession(ctx={"enrolled_embedding": None}, scenario="x")
    sp = sess.session_payload()
    assert {"type", "session_id", "channel", "detectors", "bands",
            "fusion_weights", "window_seconds", "hop_seconds"} <= set(sp)
    wins = sess.feed(real_speech)
    wp = StreamSession.window_payload(wins[0])
    assert {"type", "index", "t_start", "is_speech", "raw_score", "ema_score",
            "rms_energy", "spectrum", "detectors", "latency_ms"} <= set(wp)
    fp = sess.final_payload()
    assert {"type", "score", "band", "voice_verdict", "findings", "fusion",
            "detector_means", "baseline_detectors"} <= set(fp)


# --- the thing that matters: one pipeline -----------------------------
def test_streaming_agrees_with_batch(enrolled_clip_path):
    """Same clip, both paths, same aggregated verdict and a close score.

    They are not bit-identical by design: batch runs VAD once over the whole
    clip, streaming runs it per window. The scores must still land together.
    """
    audio, sr = load_audio(enrolled_clip_path)
    batch = analyze_audio(audio, sr, ctx={"enrolled_embedding": None}).as_dict()

    sess = StreamSession(ctx={"enrolled_embedding": None})
    chunk = int(get_settings().hop_seconds * sr)
    for i in range(0, len(audio), chunk):
        sess.feed(audio[i:i + chunk])
    sess.flush()
    stream = sess.final_payload()

    assert stream["voice_verdict"] == batch["voice_verdict"]
    assert stream["band"] == batch["band"]
    assert stream["score"] == pytest.approx(batch["score"], abs=5.0)


# --- Phase 2 gate ------------------------------------------------------
def test_gate_genuine_low_cloned_higher():
    """Genuine clip scores LOW; the cloned attack scores higher. No controls
    touched — both run with identical settings and the enrolled profile."""
    from voiceshield.ml.detectors.speaker_ecapa import SpeakerConsistencyDetector

    genuine = scen.get("genuine_control")
    cloned = scen.get("ceo_transfer_cloned")
    if not (genuine.path().exists() and cloned.path().exists()):
        pytest.skip("demo corpus not built")

    det = SpeakerConsistencyDetector()
    det.load()
    if not det.available:
        pytest.skip("ECAPA unavailable")
    ea, esr = load_audio(genuine.path())
    ctx = {"enrolled_embedding": det.embed(ea, esr)}

    def run(path):
        audio, sr = load_audio(path)
        sess = StreamSession(ctx=dict(ctx))
        chunk = int(get_settings().hop_seconds * sr)
        for i in range(0, len(audio), chunk):
            sess.feed(audio[i:i + chunk])
        sess.flush()
        return sess.final_payload()

    g, c = run(genuine.path()), run(cloned.path())
    assert g["band"] == "LOW", g["score"]
    assert c["score"] > g["score"]
    assert c["voice_verdict"] == "CLONED_VOICE"
