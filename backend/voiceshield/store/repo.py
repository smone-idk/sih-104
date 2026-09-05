"""Small query helpers over the SQLite store. Kept separate from db.py (schema
/ connection) so the pipeline can pull fixtures without importing FastAPI.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np

from .db import connect


# --- voice profiles --------------------------------------------------
def list_profiles() -> list[dict[str, Any]]:
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, display_name, role, embedding_dim, n_enroll_clips, "
            "created_at, source_note FROM voice_profiles ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_profile(profile_id: str | None = None) -> dict[str, Any] | None:
    conn = connect()
    try:
        if profile_id:
            row = conn.execute("SELECT * FROM voice_profiles WHERE id=?",
                               (profile_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM voice_profiles ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["embedding"] = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        return d
    finally:
        conn.close()


def add_profile(display_name: str, role: str, embedding: np.ndarray,
                n_clips: int, source_note: str) -> str:
    emb = np.asarray(embedding, dtype=np.float32).ravel()
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    pid = str(uuid.uuid4())
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO voice_profiles (id, display_name, role, embedding, "
            "embedding_dim, n_enroll_clips, source_note) VALUES (?,?,?,?,?,?,?)",
            (pid, display_name, role, emb.tobytes(), int(emb.shape[0]),
             int(n_clips), source_note),
        )
        conn.commit()
    finally:
        conn.close()
    return pid


def delete_profile(profile_id: str) -> bool:
    conn = connect()
    try:
        cur = conn.execute("DELETE FROM voice_profiles WHERE id=?", (profile_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# --- enterprise directory (demo data) ------------------------------
def match_directory(name_hint: str | None) -> dict[str, Any] | None:
    conn = connect()
    try:
        if name_hint:
            row = conn.execute(
                "SELECT * FROM directory_contacts WHERE lower(display_name) LIKE ?",
                (f"%{name_hint.lower()}%",),
            ).fetchone()
            if row:
                return dict(row)
        return None
    finally:
        conn.close()


def get_contact(display_name: str | None) -> dict[str, Any] | None:
    """Exact directory lookup. DEMO DATA (§2 Tier C)."""
    if not display_name:
        return None
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM directory_contacts WHERE display_name = ?",
            (display_name,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def unknown_caller() -> dict[str, Any]:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM directory_contacts WHERE known_contact=0 ORDER BY trust_score LIMIT 1"
        ).fetchone()
        return dict(row) if row else {
            "display_name": "Unknown Caller", "known_contact": 0,
            "verified_identity": 0, "prior_interactions": 0, "trust_score": 0.0,
        }
    finally:
        conn.close()


# --- sessions ----------------------------------------------------
def create_session(rec: dict[str, Any]) -> str:
    sid = rec.get("id") or str(uuid.uuid4())
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO sessions (id, source, channel, language, scenario, "
            "profile_id, telephony_degraded, duration_s, final_score, final_band, "
            "device, weights_json, retain_audio) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, rec.get("source", "upload"), rec.get("channel", "file upload"),
             rec.get("language"), rec.get("scenario"), rec.get("profile_id"),
             int(rec.get("telephony_degraded", 0)), rec.get("duration_s"),
             rec.get("final_score"), rec.get("final_band"), rec.get("device"),
             json.dumps(rec.get("weights")), int(rec.get("retain_audio", 0))),
        )
        conn.commit()
    finally:
        conn.close()
    return sid


# --- session risk (server-side source of truth for the approval gate) ----
def save_session_result(session_id: str, rec: dict[str, Any]) -> None:
    """Persist a finished analysis. The approval gate reads `final_band` from
    HERE — never from a client request (see policy/engine.py)."""
    conn = connect()
    try:
        row = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO sessions (id, source, channel, language, scenario, "
                "telephony_degraded, duration_s, final_score, final_band, device, "
                "weights_json, retain_audio) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (session_id, rec.get("source", "stream"),
                 rec.get("channel", "authorized channel"), rec.get("language"),
                 rec.get("scenario"), int(rec.get("telephony_degraded", 0)),
                 rec.get("duration_s"), rec.get("final_score"), rec.get("final_band"),
                 rec.get("device"), json.dumps(rec.get("weights")), 0),
            )
        else:
            conn.execute(
                "UPDATE sessions SET final_score=?, final_band=?, duration_s=?, "
                "scenario=COALESCE(?, scenario), device=COALESCE(?, device) WHERE id=?",
                (rec.get("final_score"), rec.get("final_band"), rec.get("duration_s"),
                 rec.get("scenario"), rec.get("device"), session_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_session(session_id: str | None) -> dict[str, Any] | None:
    if not session_id:
        return None
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# --- approvals -------------------------------------------------------
def list_approvals() -> list[dict[str, Any]]:
    conn = connect()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM approvals ORDER BY created_at DESC").fetchall()]
    finally:
        conn.close()


def get_approval(approval_id: str) -> dict[str, Any] | None:
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def set_approval_state(approval_id: str, state: str, *,
                       session_id: str | None = None,
                       unlocked_by: str | None = None) -> None:
    conn = connect()
    try:
        conn.execute(
            "UPDATE approvals SET state=?, session_id=COALESCE(?, session_id), "
            "unlocked_by=COALESCE(?, unlocked_by) WHERE id=?",
            (state, session_id, unlocked_by, approval_id))
        conn.commit()
    finally:
        conn.close()


def reset_approval(approval_id: str) -> None:
    """Back to pending, session and unlock cleared. Demo affordance — with no
    session linked, the policy engine fails closed."""
    conn = connect()
    try:
        conn.execute(
            "UPDATE approvals SET state='pending', session_id=NULL, "
            "unlocked_by=NULL WHERE id=?", (approval_id,))
        conn.commit()
    finally:
        conn.close()


def link_approval_session(approval_id: str, session_id: str) -> None:
    conn = connect()
    try:
        conn.execute("UPDATE approvals SET session_id=? WHERE id=?",
                     (session_id, approval_id))
        conn.commit()
    finally:
        conn.close()


# --- verifications ---------------------------------------------------
def create_verification(rec: dict[str, Any]) -> str:
    vid = rec.get("id") or str(uuid.uuid4())
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO verifications (id, session_id, incident_id, method, "
            "challenge_phrase, result, detail_json) VALUES (?,?,?,?,?,?,?)",
            (vid, rec.get("session_id"), rec.get("incident_id"),
             rec["method"], rec.get("challenge_phrase"),
             rec.get("result", "pending"), json.dumps(rec.get("detail", {}))),
        )
        conn.commit()
    finally:
        conn.close()
    return vid


def update_verification(vid: str, result: str, detail: dict[str, Any]) -> None:
    conn = connect()
    try:
        conn.execute("UPDATE verifications SET result=?, detail_json=? WHERE id=?",
                     (result, json.dumps(detail), vid))
        conn.commit()
    finally:
        conn.close()


def get_verification(vid: str | None) -> dict[str, Any] | None:
    if not vid:
        return None
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM verifications WHERE id=?", (vid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["detail"] = json.loads(d.pop("detail_json") or "{}")
        return d
    finally:
        conn.close()


def latest_passed_verification(session_id: str | None) -> dict[str, Any] | None:
    """Most recent PASSED verification for a session. The approval gate reads
    this from the DB; a client cannot assert it."""
    if not session_id:
        return None
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM verifications WHERE session_id=? AND result='passed' "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1", (session_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["detail"] = json.loads(d.pop("detail_json") or "{}")
        return d
    finally:
        conn.close()


def list_verifications(session_id: str | None = None) -> list[dict[str, Any]]:
    conn = connect()
    try:
        q = "SELECT * FROM verifications"
        args: tuple = ()
        if session_id:
            q += " WHERE session_id=?"
            args = (session_id,)
        q += " ORDER BY created_at DESC, rowid DESC"
        out = []
        for r in conn.execute(q, args).fetchall():
            d = dict(r)
            d["detail"] = json.loads(d.pop("detail_json") or "{}")
            out.append(d)
        return out
    finally:
        conn.close()


# --- incidents -------------------------------------------------------
def create_incident(rec: dict[str, Any]) -> str:
    iid = rec.get("id") or str(uuid.uuid4())
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO incidents (id, session_id, band, score, action, summary, "
            "payload_json) VALUES (?,?,?,?,?,?,?)",
            (iid, rec.get("session_id"), rec.get("band", "UNKNOWN"),
             float(rec.get("score") or 0.0), rec.get("action", "VERIFY"),
             rec.get("summary", ""), json.dumps(rec.get("payload", {}))),
        )
        conn.commit()
    finally:
        conn.close()
    return iid


def list_incidents(band: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    conn = connect()
    try:
        # created_at is second-resolution, so it cannot order events that land
        # in the same second. rowid is monotonic per insert and breaks the tie —
        # an audit log has to be deterministically ordered.
        q = "SELECT * FROM incidents"
        args: list = []
        if band:
            q += " WHERE band=?"
            args.append(band)
        q += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(int(limit))
        out = []
        for r in conn.execute(q, tuple(args)).fetchall():
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json") or "{}")
            out.append(d)
        return out
    finally:
        conn.close()


def get_incident(incident_id: str) -> dict[str, Any] | None:
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["payload"] = json.loads(d.pop("payload_json") or "{}")
        return d
    finally:
        conn.close()
