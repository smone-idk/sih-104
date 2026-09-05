"""Voice profiles + upload analysis + push-to-record intake (§11 screens 6–7, §8).

  GET    /api/v1/profiles              list enrolled profiles (metadata only)
  POST   /api/v1/profiles              enrol from uploaded audio
  DELETE /api/v1/profiles/{id}         remove a profile
  POST   /api/v1/analyze               batch-analyse an uploaded/recorded clip

PRIVACY (§12), enforced here rather than promised:
  * Enrolment stores the **ECAPA embedding only**. The uploaded audio is written
    to a temp file so the decoder can read it, then deleted in a `finally`. No
    audio column exists in `voice_profiles` — see `store/schema.sql`.
  * `/analyze` does the same: decode, analyse, delete. Nothing is retained
    unless `RETAIN_AUDIO` is set, which defaults to false.
  * The recorder in the UI never opens the microphone until the user presses
    record; this endpoint only ever sees a finished clip the user chose to send.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ...config import get_settings
from ...ingest.telephony import TelephonyConfig
from ...store import repo

log = logging.getLogger("voiceshield.api.profiles")
router = APIRouter()

MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def _tempfile(upload: UploadFile, data: bytes) -> Path:
    suffix = Path(upload.filename or "clip.wav").suffix or ".wav"
    return Path(tempfile.mkdtemp(prefix="voiceshield-")) / f"clip{suffix}"


def _shred(tmp: Path) -> None:
    """§12 — the decoded audio does not outlive the request."""
    try:
        tmp.unlink(missing_ok=True)
        tmp.parent.rmdir()
    except Exception:
        pass


# ----------------------------------------------------------------- profiles
@router.get("/api/v1/profiles")
def list_profiles() -> dict:
    return {
        "profiles": repo.list_profiles(),
        "note": "Metadata only. A profile stores a 192-dim ECAPA embedding, "
                "never the enrolment audio — see the Privacy Center.",
    }


@router.post("/api/v1/profiles")
async def enrol_profile(
    display_name: str = Form(...),
    role: str = Form(default=""),
    audio: list[UploadFile] = File(...),
) -> dict:
    """Enrol from one or more clips. Embeddings are averaged and L2-normalised."""
    from ...ingest.audio import load_audio
    from ...ml.detectors.speaker_ecapa import SpeakerConsistencyDetector
    from ...ml.registry import get_registry

    det = get_registry().get("speaker_consistency")
    if det is None or not det.available:
        det = SpeakerConsistencyDetector()
        det.load()
    if not det.available:
        raise HTTPException(status_code=503,
                            detail=f"speaker model unavailable: {det.load_error}")

    embeddings: list[np.ndarray] = []
    durations: list[float] = []
    for up in audio:
        data = await up.read()
        if not data:
            continue
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="clip too large")
        tmp = _tempfile(up, data)
        try:
            tmp.write_bytes(data)
            x, sr = load_audio(tmp)
            if len(x) < sr * 1.0:
                raise HTTPException(status_code=400,
                                    detail=f"{up.filename}: need at least 1 s of audio")
            embeddings.append(det.embed(x, sr))
            durations.append(len(x) / sr)
        finally:
            _shred(tmp)

    if not embeddings:
        raise HTTPException(status_code=400, detail="no usable audio supplied")

    emb = np.mean(embeddings, axis=0).astype("float32")
    pid = repo.add_profile(
        display_name=display_name, role=role, embedding=emb,
        n_clips=len(embeddings),
        source_note="enrolled via upload; embedding only, no audio stored")
    return {
        "id": pid, "display_name": display_name, "role": role,
        "n_enroll_clips": len(embeddings),
        "embedding_dim": int(emb.shape[0]),
        "total_audio_s": round(sum(durations), 2),
        "stored": "192-dim ECAPA embedding only — the audio was discarded",
    }


@router.delete("/api/v1/profiles/{profile_id}")
def delete_profile(profile_id: str) -> dict:
    if not repo.delete_profile(profile_id):
        raise HTTPException(status_code=404, detail="unknown profile")
    remaining = repo.list_profiles()
    return {
        "deleted": profile_id,
        "remaining": len(remaining),
        "effect": ("No profiles remain: the speaker-consistency layer will "
                   "report unavailable and fusion will redistribute its weight."
                   if not remaining else
                   f"{len(remaining)} profile(s) remain."),
    }


# ------------------------------------------------------------------ analyse
@router.post("/api/v1/analyze")
async def analyze_upload(
    audio: UploadFile = File(...),
    use_profile: bool = Form(default=True),
    profile_id: str | None = Form(default=None),
    telephony: bool = Form(default=False),
    snr_db: float | None = Form(default=None),
    compare_telephony: bool = Form(default=False),
    source: str = Form(default="upload"),
) -> dict:
    """Batch analysis of an uploaded or recorded clip — the SAME pipeline (§14).

    `compare_telephony` runs the clip twice, clean and through the 8 kHz + µ-law
    chain, and returns both so the §8 before/after can be shown side by side
    rather than hidden.
    """
    from ...pipeline import analyze_file

    s = get_settings()
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="clip too large")

    ctx: dict = {}
    if not use_profile:
        ctx["enrolled_embedding"] = None
    elif profile_id:
        ctx["profile_id"] = profile_id
    try:
        ctx["directory"] = repo.unknown_caller()
    except Exception:
        pass

    tmp = _tempfile(audio, data)
    try:
        tmp.write_bytes(data)
        clean = analyze_file(tmp, source=source, ctx=dict(ctx)).as_dict()
        out: dict = {"clean": clean, "filename": audio.filename,
                     "retain_audio": s.retain_audio,
                     "privacy_note": "The uploaded audio was decoded in memory "
                                     "and deleted; only the analysis is kept."}
        if compare_telephony or telephony:
            tele = TelephonyConfig(enabled=True, mu_law=True,
                                   add_noise=snr_db is not None,
                                   snr_db=snr_db if snr_db is not None else 20.0)
            degraded = analyze_file(tmp, source=source, ctx=dict(ctx),
                                    telephony=tele).as_dict()
            out["telephony"] = degraded
            out["delta"] = round(degraded["score"] - clean["score"], 2)
            out["telephony_config"] = {
                "narrowband_hz": tele.narrowband_hz, "mu_law": tele.mu_law,
                "add_noise": tele.add_noise,
                "snr_db": tele.snr_db if tele.add_noise else None,
            }
        return out
    finally:
        _shred(tmp)
