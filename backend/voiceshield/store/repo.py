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
