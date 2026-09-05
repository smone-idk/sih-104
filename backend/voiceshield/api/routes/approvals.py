"""Mock approval system, verification workflow, incident log (§7).

This turns the dashboard into an actual security control: while the linked call
session is HIGH risk, `POST /approve` returns **403 verification_required** and
the transfer cannot proceed.

  GET  /api/v1/approvals                       list pending approvals
  POST /api/v1/approvals/{id}/approve          the gate — 403 while blocked
  POST /api/v1/approvals/{id}/link             attach an analysed session
  POST /api/v1/verification/challenge          issue a challenge phrase
  POST /api/v1/verification/{id}/respond       submit response audio (re-analysed)
  POST /api/v1/verification/simulated          callback / MFA / supervisor
  GET  /api/v1/verification                    list attempts
  GET  /api/v1/incidents                       incident log
  GET  /api/v1/incidents/{id}

SECURITY: the approve decision reads the risk band from the `sessions` row and
the verification state from the `verifications` table. **Nothing in the request
body influences it.** Extra fields a client invents — `band`, `score`,
`verified`, `override` — are accepted by the parser and then ignored, so a
forged low-risk payload changes nothing. See `policy/engine.decide_approval`
and `tests/test_approvals.py::test_forged_low_risk_payload_is_ignored`.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Body, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict

from ...policy import challenge as ch
from ...policy.engine import ALLOW, decide_approval
from ...store import repo

log = logging.getLogger("voiceshield.api.approvals")
router = APIRouter()


class ApproveRequest(BaseModel):
    """What a client MAY send. Anything else is ignored rather than rejected,
    so a forged payload is provably inert instead of merely refused."""
    model_config = ConfigDict(extra="ignore")
    session_id: str | None = None
    note: str | None = None


class LinkRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    session_id: str


class ChallengeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    session_id: str | None = None
    approval_id: str | None = None
    n_words: int = 4


class SimulatedRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    session_id: str | None = None
    method: str
    outcome: str = "passed"          # demo operator chooses; labelled SIMULATED


# ---------------------------------------------------------------- approvals
@router.get("/api/v1/approvals")
def list_approvals() -> dict:
    out = []
    for a in repo.list_approvals():
        sess = repo.get_session(a.get("session_id"))
        ver = repo.latest_passed_verification(a.get("session_id"))
        d = decide_approval(session=sess, verification=ver)
        out.append({**a, "policy": d.as_dict(),
                    "session": {"id": sess["id"], "final_band": sess["final_band"],
                                "final_score": sess["final_score"],
                                "scenario": sess.get("scenario")} if sess else None})
    return {"approvals": out,
            "note": "Approval state is server-side. The Approve button is "
                    "disabled in the UI, and the API independently returns 403 "
                    "while the linked session is HIGH risk."}


@router.post("/api/v1/approvals/{approval_id}/link")
def link_session(approval_id: str, req: LinkRequest) -> dict:
    if repo.get_approval(approval_id) is None:
        raise HTTPException(status_code=404, detail="unknown approval")
    if repo.get_session(req.session_id) is None:
        raise HTTPException(status_code=404, detail="unknown session")
    repo.link_approval_session(approval_id, req.session_id)
    return {"ok": True, "approval_id": approval_id, "session_id": req.session_id}


@router.post("/api/v1/approvals/{approval_id}/approve")
def approve(approval_id: str, response: Response,
            req: ApproveRequest = Body(default=ApproveRequest())) -> dict:
    """THE gate. 403 while the linked session is HIGH and unverified.

    The only thing taken from the request is which session to look up, and even
    that is a lookup key — the risk band and verification state are read from
    the database.
    """
    approval = repo.get_approval(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="unknown approval")

    # Prefer the session already linked server-side; fall back to the one named
    # in the request purely as a lookup key.
    session_id = approval.get("session_id") or req.session_id
    session = repo.get_session(session_id)
    verification = repo.latest_passed_verification(session_id)

    decision = decide_approval(session=session, verification=verification)

    repo.create_incident({
        "session_id": session_id,
        "band": decision.band or "UNKNOWN",
        "score": decision.score or 0.0,
        "action": decision.action,
        "summary": ("Approval permitted" if decision.allowed
                    else "Approval BLOCKED at the API"),
        "payload": {"approval_id": approval_id, "amount": approval.get("amount"),
                    "decision": decision.as_dict()},
    })

    if not decision.allowed:
        repo.set_approval_state(approval_id, "blocked", session_id=session_id)
        response.status_code = decision.http_status
        return {"error": decision.error_code, "approved": False,
                **decision.as_dict()}

    repo.set_approval_state(approval_id, "approved", session_id=session_id,
                            unlocked_by=(verification or {}).get("id"))
    return {"approved": True, **decision.as_dict()}


@router.post("/api/v1/approvals/{approval_id}/reset")
def reset_approval(approval_id: str) -> dict:
    """DEMO AFFORDANCE. Return an approval to `pending` and detach its session
    so the block can be demonstrated again.

    This does not bypass anything: it clears the linked session, and with no
    session the policy fails closed (403, cause `no_session`). Every reset is
    written to the incident log, so the audit trail still shows what happened.
    """
    if repo.get_approval(approval_id) is None:
        raise HTTPException(status_code=404, detail="unknown approval")
    repo.reset_approval(approval_id)
    repo.create_incident({
        "session_id": None, "band": "UNKNOWN", "score": 0.0, "action": "VERIFY",
        "summary": "Approval reset to pending (demo affordance)",
        "payload": {"approval_id": approval_id,
                    "note": "clears the linked session; policy then fails closed"},
    })
    return {"ok": True, "approval_id": approval_id, "state": "pending"}


# ------------------------------------------------------------ verification
@router.post("/api/v1/verification/challenge")
def issue_challenge(req: ChallengeRequest) -> dict:
    c = ch.generate_challenge(req.n_words)
    vid = repo.create_verification({
        "session_id": req.session_id, "method": "challenge_response",
        "challenge_phrase": c.phrase, "result": "pending",
        "detail": {"approval_id": req.approval_id, "issued_words": c.words},
    })
    return {"verification_id": vid, "phrase": c.phrase, "result": "pending",
            "instruction": "Ask the caller to repeat this phrase, then submit "
                           "their response audio for re-analysis.",
            "kind": "real",
            "limitation": "Defeats replay of pre-recorded audio. A live "
                          "real-time voice-conversion attacker could still "
                          "repeat the phrase in the cloned voice."}


@router.post("/api/v1/verification/{verification_id}/respond")
async def challenge_respond(verification_id: str,
                            audio: UploadFile = File(...),
                            session_id: str | None = Form(default=None)) -> dict:
    """Re-analyse the caller's response to a challenge phrase.

    Real: the audio goes through the SAME `analyze_audio` pipeline as everything
    else (§14), and the ASR transcript is compared against the phrase we issued.
    BOTH must pass — saying the right words in a synthetic voice fails.
    """
    import tempfile
    from pathlib import Path as _P

    from ...pipeline import analyze_file
    from ...store.repo import get_profile

    ver = repo.get_verification(verification_id)
    if ver is None:
        raise HTTPException(status_code=404, detail="unknown verification")
    if ver["method"] != "challenge_response":
        raise HTTPException(status_code=400, detail="not a challenge-response attempt")

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio")

    suffix = _P(audio.filename or "response.wav").suffix or ".wav"
    tmp = _P(tempfile.mkdtemp()) / f"challenge{suffix}"
    try:
        tmp.write_bytes(data)
        ctx: dict = {}
        prof = get_profile(None)
        if prof is not None:
            ctx["enrolled_embedding"] = prof["embedding"]
            ctx["enrolled_speaker_name"] = prof["display_name"]
        res = analyze_file(tmp, source="challenge", ctx=ctx).as_dict()
    finally:
        # §12 — the response audio is never retained
        try:
            tmp.unlink(missing_ok=True)
            tmp.parent.rmdir()
        except Exception:
            pass

    phrase = ch.check_phrase(ver["challenge_phrase"] or "",
                             res.get("transcript", {}).get("text", ""))
    acoustic_ok = res["band"] == "LOW" and res["voice_verdict"] in {
        "CONSISTENT", "NO_SYNTHETIC_MARKERS", "SPEAKER_CONSISTENT"}
    passed = bool(phrase["matched"] and acoustic_ok)

    detail = {
        "phrase_check": phrase,
        "reanalysis": {"score": res["score"], "band": res["band"],
                       "voice_verdict": res["voice_verdict"],
                       "findings": [f["code"] for f in res["findings"]]},
        "acoustic_ok": acoustic_ok,
        "why": ("phrase repeated and voice consistent" if passed else
                "phrase not repeated" if not phrase["matched"] else
                "phrase repeated but the responding voice failed analysis"),
    }
    repo.update_verification(verification_id, "passed" if passed else "failed", detail)
    sid = session_id or ver.get("session_id")
    repo.create_incident({
        "session_id": sid, "band": res["band"], "score": res["score"],
        "action": "VERIFY",
        "summary": f"Challenge-response {'PASSED' if passed else 'FAILED'}",
        "payload": {"verification_id": verification_id, **detail},
    })
    return {"verification_id": verification_id,
            "result": "passed" if passed else "failed", **detail}


@router.post("/api/v1/verification/simulated")
def simulated_verification(req: SimulatedRequest) -> dict:
    if req.method not in ch.SIMULATED_METHODS:
        raise HTTPException(status_code=400,
                            detail=f"unknown method; expected one of "
                                   f"{sorted(ch.SIMULATED_METHODS)}")
    result = "passed" if req.outcome == "passed" else "failed"
    vid = repo.create_verification({
        "session_id": req.session_id, "method": req.method,
        "result": result,
        "detail": {"simulated": True,
                   "description": ch.SIMULATED_METHODS[req.method],
                   "note": "SIMULATED state machine — nothing was contacted."},
    })
    repo.create_incident({
        "session_id": req.session_id, "band": "UNKNOWN", "score": 0.0,
        "action": "VERIFY",
        "summary": f"Simulated verification ({req.method}) -> {result}",
        "payload": {"verification_id": vid, "simulated": True},
    })
    return {"verification_id": vid, "method": req.method, "result": result,
            "kind": "simulated",
            "description": ch.SIMULATED_METHODS[req.method]}


@router.get("/api/v1/verification")
def list_verifications(session_id: str | None = None) -> dict:
    return {"verifications": repo.list_verifications(session_id),
            "methods": {"challenge_response": "real — response audio re-analysed",
                        **{k: f"simulated — {v}" for k, v in ch.SIMULATED_METHODS.items()}}}


# ---------------------------------------------------------------- incidents
@router.get("/api/v1/incidents")
def list_incidents(band: str | None = None, limit: int = 200) -> dict:
    return {"incidents": repo.list_incidents(band, limit)}


@router.get("/api/v1/incidents/{incident_id}")
def get_incident(incident_id: str) -> dict:
    inc = repo.get_incident(incident_id)
    if inc is None:
        raise HTTPException(status_code=404, detail="unknown incident")
    sess = repo.get_session(inc.get("session_id"))
    return {"incident": inc, "session": sess}
