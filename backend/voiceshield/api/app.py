"""FastAPI application.

Phase 0: boots offline, loads every model once, prints the detector inventory,
exposes health + inventory + config. Later phases add /analyze, /stream, etc.
Models are loaded in the startup hook and kept warm for the process lifetime —
never per request (§1, §3).
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..config import get_settings
from ..inventory import build_report, print_inventory
from ..ml.registry import get_registry
from ..store.db import init_db, table_summary
from .routes import approvals as approval_routes
from .routes import stream as stream_routes

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voiceshield.api")

app = FastAPI(title="VoiceShield", version=__version__,
              description="Voice Integrity & Impersonation Risk Engine (Prototype)")

app.include_router(stream_routes.router)
app.include_router(approval_routes.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    s = get_settings()
    log.info("VoiceShield %s starting — device=%s offline=%s", __version__, s.device, s.offline)
    init_db()
    reg = get_registry()          # loads all models once
    reg.warmup()
    tier_a_ok = print_inventory()
    if not tier_a_ok:
        log.error("Tier-A detectors missing. Run: python scripts/fetch_models.py")
    app.state.registry = reg


@app.get("/api/v1/health")
def health() -> dict:
    s = get_settings()
    reg = get_registry()
    return {
        "status": "ok",
        "version": __version__,
        "device": s.device,
        "offline": s.offline,
        "detectors_loaded": len(reg.available()),
        "detectors_total": len(reg.all()),
    }


@app.get("/api/v1/inventory")
def inventory() -> dict:
    return build_report()


@app.get("/api/v1/config")
def config() -> dict:
    s = get_settings()
    return {
        "device": s.device,
        "window_seconds": s.window_seconds,
        "hop_seconds": s.hop_seconds,
        "sample_rate": s.sample_rate,
        "ema_alpha": s.ema_alpha,
        "fusion_weights": s.fusion_weights(),
        "bands": {"low_max": s.band_low_max, "high_min": s.band_high_min},
        "retain_audio": s.retain_audio,
        "languages": ["en", "hi", "pa"],
    }


@app.get("/api/v1/privacy/db-contents")
def db_contents() -> dict:
    """Backs the Privacy Center — exactly what the SQLite DB holds (§12)."""
    return {"tables": table_summary(), "note": "no audio blobs; voice_profiles store ECAPA embeddings only"}
