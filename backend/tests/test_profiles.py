"""Voice profiles, upload analysis, telephony toggle (§8, §11–12, Phase 5).

THE gate: removing an enrolled profile disables the speaker-consistency layer
and redistributes its weight — visibly, in the explainability payload.

Also pins the privacy claims the Privacy Center makes, so they are properties of
the code rather than promises in a document.
"""
from __future__ import annotations

import io
import uuid
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voiceshield.api.app import app
from voiceshield.config import get_settings
from voiceshield.store import repo


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def clip_bytes(enrolled_clip_path):
    return enrolled_clip_path.read_bytes()


def _wav_bytes(x: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


# --- enrolment ---------------------------------------------------------
def test_enrol_stores_an_embedding_and_no_audio(client, clip_bytes, enrolled_clip_path):
    r = client.post("/api/v1/profiles",
                    data={"display_name": "Test Speaker", "role": "QA"},
                    files={"audio": (enrolled_clip_path.name, clip_bytes, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["embedding_dim"] == 192
    assert body["n_enroll_clips"] == 1
    assert "audio was discarded" in body["stored"]

    # the row itself must carry no audio, only the embedding
    prof = repo.get_profile(body["id"])
    assert prof["embedding"].shape == (192,)
    assert np.linalg.norm(prof["embedding"]) == pytest.approx(1.0, abs=1e-4)
    assert not any("audio" in k.lower() for k in prof if k != "source_note")
    repo.delete_profile(body["id"])


def test_enrol_rejects_too_short_audio(client):
    tiny = _wav_bytes(np.zeros(1600, dtype=np.float32))     # 0.1 s
    r = client.post("/api/v1/profiles",
                    data={"display_name": "Too Short"},
                    files={"audio": ("t.wav", tiny, "audio/wav")})
    assert r.status_code == 400
    assert "at least 1 s" in r.json()["detail"]


def test_enrol_rejects_empty_upload(client):
    r = client.post("/api/v1/profiles",
                    data={"display_name": "Nothing"},
                    files={"audio": ("t.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_listing_never_exposes_the_embedding(client):
    r = client.get("/api/v1/profiles")
    assert r.status_code == 200
    for p in r.json()["profiles"]:
        assert "embedding" not in p
        assert {"id", "display_name", "n_enroll_clips", "embedding_dim"} <= set(p)


def test_delete_unknown_profile_404s(client):
    assert client.delete(f"/api/v1/profiles/{uuid.uuid4()}").status_code == 404


# --- THE GATE ----------------------------------------------------------
def test_removing_the_profile_disables_the_speaker_layer(client, clip_bytes,
                                                         enrolled_clip_path):
    """Enrol -> speaker layer carries weight. Delete -> layer unavailable and
    its 0.20 weight is redistributed, never substituted."""
    existing = [p["id"] for p in repo.list_profiles()]
    for pid in existing:
        repo.delete_profile(pid)

    enrolled = client.post(
        "/api/v1/profiles", data={"display_name": "Gate Speaker", "role": "CFO"},
        files={"audio": (enrolled_clip_path.name, clip_bytes, "audio/wav")}).json()

    def analyse():
        return client.post("/api/v1/analyze", data={"use_profile": "true"},
                           files={"audio": (enrolled_clip_path.name, clip_bytes,
                                            "audio/wav")}).json()["clean"]

    with_profile = analyse()
    sp = next(c for c in with_profile["fusion"]["components"]
              if c["name"] == "speaker_consistency")
    assert sp["available"] is True
    assert sp["effective_weight"] > 0
    assert sp["raw_value"] is not None

    d = client.delete(f"/api/v1/profiles/{enrolled['id']}")
    assert d.status_code == 200
    assert "redistribute" in d.json()["effect"]

    without = analyse()
    sp2 = next(c for c in without["fusion"]["components"]
               if c["name"] == "speaker_consistency")
    assert sp2["available"] is False, "speaker layer must go unavailable"
    assert sp2["effective_weight"] == 0.0, "its weight must not still be applied"
    assert sp2["raw_value"] is None, "no substituted value"
    assert "no enrolled profile" in sp2["note"]
    assert "speaker_consistency" in without["fusion"]["unavailable_components"]

    # the weight went somewhere: effective weights still sum to 1
    total = sum(c["effective_weight"] for c in without["fusion"]["components"])
    assert total == pytest.approx(1.0, abs=1e-3)
    # and the layers that remain now carry MORE than their nominal weight
    va = next(c for c in without["fusion"]["components"]
              if c["name"] == "voice_authenticity")
    assert va["effective_weight"] > va["weight"]


# --- upload analysis + telephony (§8) ----------------------------------
def test_analyze_upload_runs_the_same_pipeline(client, clip_bytes, enrolled_clip_path):
    r = client.post("/api/v1/analyze", data={"use_profile": "false"},
                    files={"audio": (enrolled_clip_path.name, clip_bytes, "audio/wav")})
    assert r.status_code == 200
    c = r.json()["clean"]
    assert {"score", "band", "voice_verdict", "fusion", "findings",
            "transcript", "context_quotes", "latency_ms"} <= set(c)
    assert 0 <= c["score"] <= 100


def test_telephony_comparison_returns_both_and_a_delta(client, clip_bytes,
                                                       enrolled_clip_path):
    r = client.post("/api/v1/analyze",
                    data={"use_profile": "false", "compare_telephony": "true"},
                    files={"audio": (enrolled_clip_path.name, clip_bytes, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert "clean" in body and "telephony" in body
    assert body["telephony"]["telephony_degraded"] is True
    assert body["clean"]["telephony_degraded"] is False
    assert body["delta"] == pytest.approx(
        body["telephony"]["score"] - body["clean"]["score"], abs=0.02)
    assert body["telephony_config"]["narrowband_hz"] == 8000


def test_analyze_rejects_empty_upload(client):
    r = client.post("/api/v1/analyze", files={"audio": ("x.wav", b"", "audio/wav")})
    assert r.status_code == 400


# --- privacy claims are code, not promises (§12) -----------------------
def test_no_audio_file_is_left_behind(client, clip_bytes, enrolled_clip_path,
                                      tmp_path_factory):
    """The endpoint writes a temp file to decode, then deletes it. Assert no
    voiceshield temp dirs survive the request."""
    import tempfile
    from pathlib import Path

    before = set(Path(tempfile.gettempdir()).glob("voiceshield-*"))
    client.post("/api/v1/analyze", data={"use_profile": "false"},
                files={"audio": (enrolled_clip_path.name, clip_bytes, "audio/wav")})
    after = set(Path(tempfile.gettempdir()).glob("voiceshield-*"))
    assert after <= before, f"leaked temp audio dirs: {after - before}"


def test_retain_audio_defaults_to_false():
    assert get_settings().retain_audio is False


def test_db_holds_no_audio_columns():
    """The Privacy Center says the DB contains no audio. Check the schema."""
    from voiceshield.store.db import connect

    conn = connect()
    try:
        cols = []
        for (table,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall():
                cols.append((table, row[1], (row[2] or "").upper()))
    finally:
        conn.close()
    blobs = [(t, c) for t, c, ty in cols if ty == "BLOB"]
    # exactly one BLOB in the whole schema: the ECAPA embedding
    assert blobs == [("voice_profiles", "embedding")], blobs
