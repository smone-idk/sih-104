#!/usr/bin/env python3
"""Seed the SQLite DB: schema + enterprise directory fixture + (if demo assets
exist) the enrolled voice profile and the mock pending approval (§7.2, §14).

  python scripts/seed.py           # schema + directory + approval + profile if possible
  python scripts/seed.py --reset   # drop the DB file first
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from voiceshield.config import get_settings          # noqa: E402
from voiceshield.store.db import connect, init_db     # noqa: E402

# Enterprise directory — DEMO DATA (§2 Tier C). Panel is labelled as such in UI.
DIRECTORY = [
    # name, role, phone, known, verified, prior_interactions, trust
    ("Rajesh Sharma", "Chief Financial Officer", "+91-98xxxxxx01", 1, 1, 214, 0.92),
    ("Priya Nair", "Finance Manager", "+91-98xxxxxx02", 1, 1, 88, 0.85),
    ("IT Service Desk", "Internal", "+91-1800-xxxxx", 1, 1, 640, 0.80),
    ("Unknown Caller", "-", "+91-70xxxxxx77", 0, 0, 0, 0.05),
    ("HDFC Fraud Dept (claimed)", "claimed bank official", "+91-22xxxxxx00", 0, 0, 0, 0.03),
    ("Inspector Verma (claimed)", "claimed law enforcement", "withheld", 0, 0, 0, 0.02),
]


def seed_directory(conn) -> None:
    conn.execute("DELETE FROM directory_contacts")
    for name, role, phone, known, verified, prior, trust in DIRECTORY:
        conn.execute(
            "INSERT INTO directory_contacts (id, display_name, role, phone, "
            "known_contact, verified_identity, prior_interactions, trust_score) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), name, role, phone, known, verified, prior, trust),
        )
    print(f"[ok] seeded {len(DIRECTORY)} directory contacts (demo data)")


def seed_approval(conn) -> None:
    row = conn.execute("SELECT id FROM approvals WHERE state='pending'").fetchone()
    if row:
        print("[skip] pending approval already exists")
        return
    conn.execute(
        "INSERT INTO approvals (id, description, amount, currency, state) "
        "VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), "Vendor payment — new beneficiary", 2500000.0, "INR", "pending"),
    )
    print("[ok] seeded mock pending approval: ₹25,00,000 vendor payment")


def seed_voice_profile(conn) -> None:
    s = get_settings()
    enrolled = sorted((s.demo_assets_dir / "genuine").glob("enrolled_*.wav"))[:3]
    if not enrolled:
        print("[skip] no enrolled_*.wav in demo_assets/genuine — run build_demo_assets.py --tier genuine")
        return
    if conn.execute("SELECT count(*) FROM voice_profiles").fetchone()[0]:
        print("[skip] voice profile already enrolled")
        return
    try:
        import numpy as np
        import soundfile as sf
        from voiceshield.ml.detectors.speaker_ecapa import SpeakerConsistencyDetector

        det = SpeakerConsistencyDetector()
        det.load()
        if not det.available:
            print(f"[skip] ECAPA not available ({det.load_error}) — run fetch_models.py")
            return
        embs = []
        for wav in enrolled:
            data, sr = sf.read(wav)
            embs.append(det.embed(np.asarray(data), sr))
        emb = np.mean(embs, axis=0).astype("float32")
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        label_file = s.demo_assets_dir / "genuine" / "ENROLLED_SPEAKER.txt"
        label = "Rajesh Sharma — CFO"
        if label_file.exists():
            label = label_file.read_text(encoding="utf-8").splitlines()[1].strip()
        conn.execute(
            "INSERT INTO voice_profiles (id, display_name, role, embedding, "
            "embedding_dim, n_enroll_clips, source_note) VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), label, "Chief Financial Officer", emb.tobytes(),
             int(emb.shape[0]), len(enrolled),
             "LibriSpeech dev-clean speaker; embedding only, no audio stored"),
        )
        print(f"[ok] enrolled voice profile '{label}' (dim={emb.shape[0]}, {len(enrolled)} clips, embedding only)")
    except Exception as exc:
        print(f"[skip] voice profile enrollment failed: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    if args.reset and s.db_path.exists():
        s.db_path.unlink()
        for ext in ("-wal", "-shm"):
            p = Path(str(s.db_path) + ext)
            p.unlink(missing_ok=True)
        print(f"[ok] removed {s.db_path}")

    init_db()
    print(f"[ok] schema at {s.db_path}")
    conn = connect()
    try:
        seed_directory(conn)
        seed_approval(conn)
        seed_voice_profile(conn)
        conn.commit()
        from voiceshield.store.db import table_summary
        print("\nDB contents:", json.dumps(table_summary(), indent=2))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
