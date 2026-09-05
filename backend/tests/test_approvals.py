"""Approval gate, verification workflow, incident log (§7, Phase 4).

THE gate: `POST /approve` returns 403 while the linked session is HIGH risk.

THE security property: that 403 derives from server-side state ONLY. A client
with devtools open must not be able to unlock the approval by sending a
low-risk band, a high score, or a `verified: true` flag.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from voiceshield.api.app import app
from voiceshield.policy.challenge import check_phrase, generate_challenge
from voiceshield.policy.engine import ALLOW, ESCALATE, VERIFY, decide_approval
from voiceshield.store import repo


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _session(band: str, score: float) -> str:
    sid = str(uuid.uuid4())
    repo.save_session_result(sid, {
        "source": "test", "channel": "test", "final_band": band,
        "final_score": score, "duration_s": 10.0, "device": "cpu"})
    return sid


def _approval(session_id: str | None = None) -> str:
    aid = str(uuid.uuid4())
    from voiceshield.store.db import connect
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO approvals (id, description, amount, currency, state, session_id)"
            " VALUES (?,?,?,?,?,?)",
            (aid, "Vendor payment — test", 2_500_000.0, "INR", "pending", session_id))
        conn.commit()
    finally:
        conn.close()
    return aid


# --- policy unit -------------------------------------------------------
def test_band_to_action():
    assert decide_approval(session={"final_band": "LOW", "final_score": 5},
                           verification=None).action == ALLOW
    assert decide_approval(session={"final_band": "MEDIUM", "final_score": 55},
                           verification=None).action == VERIFY
    assert decide_approval(session={"final_band": "HIGH", "final_score": 80},
                           verification=None).action == ESCALATE


def test_no_session_is_refused_not_allowed():
    """Absence of analysis must fail closed."""
    d = decide_approval(session=None, verification=None)
    assert d.allowed is False and d.http_status == 403


def test_unanalysed_session_fails_closed():
    d = decide_approval(session={"id": "x", "final_band": None, "final_score": None},
                        verification=None)
    assert d.allowed is False and d.http_status == 403


def test_passed_verification_unlocks():
    d = decide_approval(session={"final_band": "HIGH", "final_score": 80},
                        verification={"id": "v", "method": "callback",
                                      "result": "passed"})
    assert d.allowed is True


def test_failed_or_pending_verification_does_not_unlock():
    for result in ("pending", "failed"):
        d = decide_approval(session={"final_band": "HIGH", "final_score": 80},
                            verification={"id": "v", "method": "callback",
                                          "result": result})
        assert d.allowed is False, result


# --- THE gate ----------------------------------------------------------
def test_approve_returns_403_while_high(client):
    aid = _approval(_session("HIGH", 82.0))
    r = client.post(f"/api/v1/approvals/{aid}/approve", json={})
    assert r.status_code == 403
    body = r.json()
    assert body["error"] == "verification_required"
    assert body["approved"] is False
    assert body["action"] == ESCALATE
    assert "do not approve" in body["reason"].lower()
    assert repo.get_approval(aid)["state"] == "blocked"


def test_approve_allowed_when_low(client):
    aid = _approval(_session("LOW", 8.0))
    r = client.post(f"/api/v1/approvals/{aid}/approve", json={})
    assert r.status_code == 200
    assert r.json()["approved"] is True
    assert repo.get_approval(aid)["state"] == "approved"


def test_medium_requires_verification(client):
    aid = _approval(_session("MEDIUM", 55.0))
    assert client.post(f"/api/v1/approvals/{aid}/approve", json={}).status_code == 403


# --- THE security property --------------------------------------------
@pytest.mark.parametrize("forged", [
    {"band": "LOW", "score": 1.0},
    {"final_band": "LOW", "final_score": 0.0},
    {"verified": True, "verification_passed": True},
    {"policy": {"allowed": True}, "action": "ALLOW"},
    {"override": True, "admin": True, "allowed": True},
    {"session": {"final_band": "LOW"}},
])
def test_forged_low_risk_payload_is_ignored(client, forged):
    """A judge with devtools open must not be able to unlock the approval.

    Every one of these payloads asserts the call is safe. None of them is
    consulted: the band comes from the sessions row, the verification state from
    the verifications table.
    """
    aid = _approval(_session("HIGH", 88.0))
    r = client.post(f"/api/v1/approvals/{aid}/approve", json=forged)
    assert r.status_code == 403, f"forged payload unlocked the approval: {forged}"
    assert r.json()["error"] == "verification_required"
    assert r.json()["band"] == "HIGH"          # server's band, not the forged one
    assert repo.get_approval(aid)["state"] == "blocked"


def test_forged_session_id_pointing_at_a_low_session_does_not_unlock(client):
    """The linked session wins. Naming a different, low-risk session in the body
    must not redirect the lookup."""
    high = _session("HIGH", 90.0)
    low = _session("LOW", 3.0)
    aid = _approval(high)                       # linked server-side to HIGH
    r = client.post(f"/api/v1/approvals/{aid}/approve", json={"session_id": low})
    assert r.status_code == 403
    assert r.json()["band"] == "HIGH"


def test_client_cannot_forge_a_verification_via_approve(client):
    sid = _session("HIGH", 80.0)
    aid = _approval(sid)
    client.post(f"/api/v1/approvals/{aid}/approve",
                json={"verification": {"result": "passed", "method": "callback"}})
    assert repo.latest_passed_verification(sid) is None


# --- verification workflow --------------------------------------------
def test_challenge_is_random_and_recorded(client):
    sid = _session("HIGH", 80.0)
    phrases = set()
    for _ in range(5):
        r = client.post("/api/v1/verification/challenge", json={"session_id": sid})
        assert r.status_code == 200
        b = r.json()
        phrases.add(b["phrase"])
        assert b["result"] == "pending"
        assert repo.get_verification(b["verification_id"])["challenge_phrase"] == b["phrase"]
    assert len(phrases) > 1, "challenge phrases must not repeat"


def test_pending_challenge_does_not_unlock(client):
    sid = _session("HIGH", 80.0)
    aid = _approval(sid)
    client.post("/api/v1/verification/challenge", json={"session_id": sid})
    assert client.post(f"/api/v1/approvals/{aid}/approve", json={}).status_code == 403


def test_simulated_verification_unlocks_and_is_labelled(client):
    sid = _session("HIGH", 80.0)
    aid = _approval(sid)
    assert client.post(f"/api/v1/approvals/{aid}/approve", json={}).status_code == 403
    r = client.post("/api/v1/verification/simulated",
                    json={"session_id": sid, "method": "callback"})
    assert r.json()["kind"] == "simulated"
    assert client.post(f"/api/v1/approvals/{aid}/approve", json={}).status_code == 200


def test_simulated_rejects_unknown_method(client):
    r = client.post("/api/v1/verification/simulated",
                    json={"session_id": None, "method": "telepathy"})
    assert r.status_code == 400


def test_failed_simulated_verification_does_not_unlock(client):
    sid = _session("HIGH", 80.0)
    aid = _approval(sid)
    client.post("/api/v1/verification/simulated",
                json={"session_id": sid, "method": "mfa", "outcome": "failed"})
    assert client.post(f"/api/v1/approvals/{aid}/approve", json={}).status_code == 403


# --- challenge phrase matching ----------------------------------------
def test_phrase_check_matches_and_rejects():
    c = generate_challenge(4)
    assert check_phrase(c.phrase, f"sure, {c.phrase} okay")["matched"]
    assert not check_phrase(c.phrase, "I am not going to repeat that")["matched"]


def test_phrase_check_reports_missing_words():
    got = check_phrase("amber seven willow granite", "amber seven")
    assert got["matched"] is False
    assert set(got["missing"]) == {"willow", "granite"}
    assert got["ratio"] == pytest.approx(0.5)


# --- incident log ------------------------------------------------------
def test_every_approval_attempt_writes_an_incident(client):
    sid = _session("HIGH", 80.0)
    aid = _approval(sid)
    client.post(f"/api/v1/approvals/{aid}/approve", json={})
    newest = repo.list_incidents(limit=5)[0]
    assert newest["action"] == ESCALATE
    assert "BLOCKED" in newest["summary"]
    assert newest["payload"]["approval_id"] == aid


def test_incident_log_filters_by_band(client):
    r = client.get("/api/v1/incidents", params={"band": "HIGH"})
    assert r.status_code == 200
    assert all(i["band"] == "HIGH" for i in r.json()["incidents"])


def test_incident_detail_404s_for_unknown(client):
    assert client.get(f"/api/v1/incidents/{uuid.uuid4()}").status_code == 404


def test_approvals_listing_exposes_policy(client):
    r = client.get("/api/v1/approvals")
    assert r.status_code == 200
    for a in r.json()["approvals"]:
        assert "policy" in a and "allowed" in a["policy"]


# --- demo reset (must not become a bypass) -----------------------------
def test_reset_returns_to_pending_and_fails_closed(client):
    sid = _session("HIGH", 80.0)
    aid = _approval(sid)
    client.post("/api/v1/verification/simulated",
                json={"session_id": sid, "method": "callback"})
    assert client.post(f"/api/v1/approvals/{aid}/approve", json={}).status_code == 200

    r = client.post(f"/api/v1/approvals/{aid}/reset")
    assert r.status_code == 200
    a = repo.get_approval(aid)
    assert a["state"] == "pending" and a["session_id"] is None

    # With no session linked the policy must FAIL CLOSED, not open.
    r2 = client.post(f"/api/v1/approvals/{aid}/approve", json={})
    assert r2.status_code == 403
    assert r2.json()["detail"]["cause"] == "no_session"


def test_reset_is_recorded_in_the_incident_log(client):
    """Count-based assertions break once the log exceeds list_incidents()'s
    limit, so assert on the newest row instead."""
    aid = _approval(_session("HIGH", 80.0))
    client.post(f"/api/v1/approvals/{aid}/reset")
    newest = repo.list_incidents(limit=5)[0]
    assert "reset" in newest["summary"].lower()
    assert newest["payload"]["approval_id"] == aid


def test_reset_404s_for_unknown_approval(client):
    assert client.post(f"/api/v1/approvals/{uuid.uuid4()}/reset").status_code == 404


# --- challenge-response endpoint --------------------------------------
def test_respond_rejects_unknown_verification(client):
    r = client.post(f"/api/v1/verification/{uuid.uuid4()}/respond",
                    files={"audio": ("x.wav", b"RIFF0000", "audio/wav")})
    assert r.status_code == 404


def test_respond_rejects_a_simulated_attempt(client):
    sid = _session("HIGH", 80.0)
    v = client.post("/api/v1/verification/simulated",
                    json={"session_id": sid, "method": "mfa"}).json()
    r = client.post(f"/api/v1/verification/{v['verification_id']}/respond",
                    files={"audio": ("x.wav", b"RIFF0000", "audio/wav")})
    assert r.status_code == 400


def test_respond_with_the_genuine_clip_and_no_phrase_fails(client):
    """The enrolled speaker's own audio, which does not contain the challenge
    phrase, must FAIL — saying the wrong words never passes, however genuine
    the voice is."""
    from voiceshield.config import get_settings
    clips = sorted((get_settings().demo_assets_dir / "genuine").glob("enrolled_1272*.wav"))
    if not clips:
        pytest.skip("demo corpus not built")
    sid = _session("HIGH", 80.0)
    v = client.post("/api/v1/verification/challenge", json={"session_id": sid}).json()
    r = client.post(f"/api/v1/verification/{v['verification_id']}/respond",
                    files={"audio": (clips[0].name, clips[0].read_bytes(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["result"] == "failed"
    assert body["phrase_check"]["matched"] is False
    assert repo.get_verification(v["verification_id"])["result"] == "failed"
