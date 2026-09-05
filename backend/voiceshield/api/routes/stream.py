"""WebSocket streaming + scenario listing (§4, §13, §11 screen 1).

  GET  /api/v1/scenarios          bundled scenarios, each backed by a real file
  WS   /api/v1/stream             live analysis

Protocol (server -> client), newline-delimited JSON frames:
  {"type":"session", ...}   once on open: ids, config, detector inventory
  {"type":"window",  ...}   per 4 s window, on the 1 s hop
  {"type":"final",   ...}   on completion: fusion, findings, verdict, latencies
  {"type":"error",   ...}

Client -> server: one JSON "start" frame, then optionally {"type":"stop"}.
Audio is streamed server-side from a bundled clip (simulated transport, real
analysis). Binary client audio (push-to-record) lands in Phase 5 on the same
session object.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...config import get_settings
from ...ingest.audio import load_audio
from ...ingest.telephony import TelephonyConfig
from ...asr.worker import AsrWorker
from ...store.repo import (create_incident, get_contact, get_profile,
                           save_session_result, unknown_caller)
from ...stream import scenarios as scen
from ...stream.session import StreamSession

log = logging.getLogger("voiceshield.api.stream")
router = APIRouter()


@router.get("/api/v1/scenarios")
def list_scenarios() -> dict:
    return {
        "scenarios": scen.listing(),
        "note": "Transport is simulated (Tier C); the analysis is real and runs "
                "the same pipeline as upload and live capture (§13, §14).",
    }


def _resolve_ctx(use_profile: bool, profile_id: str | None) -> dict:
    ctx: dict = {}
    if not use_profile:
        ctx["enrolled_embedding"] = None
        return ctx
    try:
        prof = get_profile(profile_id)
        if prof is not None:
            ctx["enrolled_embedding"] = prof["embedding"]
            ctx["enrolled_speaker_name"] = prof["display_name"]
    except Exception as exc:                       # pragma: no cover
        log.warning("profile lookup failed: %s", exc)
    return ctx


@router.websocket("/api/v1/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    s = get_settings()
    session: StreamSession | None = None
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=30)
        req = json.loads(raw)

        scenario_id = req.get("scenario")
        sc = scen.get(scenario_id) if scenario_id else None
        if scenario_id and sc is None:
            await ws.send_json({"type": "error", "error": f"unknown scenario {scenario_id!r}"})
            return
        if sc is not None and not sc.path().exists():
            await ws.send_json({
                "type": "error",
                "error": f"scenario asset missing: {sc.relpath} — "
                         "run scripts/build_demo_assets.py --all"})
            return

        tele = TelephonyConfig(
            enabled=bool(req.get("telephony")),
            add_noise=req.get("snr_db") is not None,
            snr_db=float(req.get("snr_db") or 20.0),
        )
        ctx = _resolve_ctx(req.get("use_profile", True), req.get("profile_id"))
        # Enterprise directory record — DEMO DATA (§2 Tier C), labelled in the UI.
        # Resolved from the scenario's caller number, NOT hardcoded: attaching
        # "Unknown Caller" to every session would penalise the genuine control
        # for something never measured from its audio.
        if req.get("directory", True) and sc is not None:
            try:
                ctx["directory"] = get_contact(sc.directory_contact) or unknown_caller()
            except Exception:
                pass
        speed = float(req.get("speed", 1.0))       # 1.0 = real time
        realtime = bool(req.get("realtime", True))

        session = StreamSession(
            source="simulation",
            channel=f"bundled demo clip — {sc.title}" if sc else "bundled demo clip",
            scenario=scenario_id,
            ctx=ctx,
            telephony=tele,
        )
        payload = session.session_payload()
        if sc is not None:
            payload["scenario_meta"] = sc.as_dict()
        await ws.send_json(payload)

        if sc is None:
            await ws.send_json({"type": "error", "error": "no scenario given"})
            return

        audio, sr = load_audio(sc.path())
        asr = AsrWorker.get() if (s.asr_enabled and req.get("asr", True)) else None
        loop = asyncio.get_running_loop()

        async def drain_asr(force: bool = False) -> None:
            """Transcribe any closed utterances and push a context frame.

            Whisper runs in a thread executor so the 1 Hz acoustic loop is never
            blocked by it — the two cadences stay independent (§4).
            """
            if asr is None or not asr.available or session is None:
                return
            for t_start, seg in session.pending_utterances(force=force):
                segs = await loop.run_in_executor(
                    None, lambda a=seg, t=t_start: asr.transcribe(
                        a, sr, t_offset=t, index=len(session.transcript.segments)))
                if segs:
                    session.apply_transcript(segs)
                    await ws.send_json(session.context_payload())

        # feed in hop-sized chunks so windows complete on the 1 s cadence (§4)
        chunk = int(round(s.hop_seconds * sr))
        t_wall = time.perf_counter()
        for i in range(0, len(audio), chunk):
            piece = audio[i:i + chunk]
            for wsr in session.feed(piece):
                await ws.send_json(StreamSession.window_payload(wsr))
            await drain_asr()
            if realtime:
                # pace to wall clock so the chart moves like a real call
                target = (i + len(piece)) / sr / max(speed, 0.01)
                lag = target - (time.perf_counter() - t_wall)
                if lag > 0:
                    await asyncio.sleep(lag)
        for wsr in session.flush():
            await ws.send_json(StreamSession.window_payload(wsr))
        await drain_asr(force=True)

        final = session.final_payload()
        # Persist the result SERVER-SIDE. This row is the only thing the
        # approval gate reads for risk — a client cannot assert its own band.
        try:
            save_session_result(session.meta.session_id, {
                "source": session.meta.source,
                "channel": session.meta.channel,
                "scenario": session.meta.scenario,
                "telephony_degraded": session.meta.telephony_degraded,
                "duration_s": final.get("duration_s"),
                "final_score": final.get("score"),
                "final_band": final.get("band"),
                "device": s.device,
                "weights": s.fusion_weights(),
            })
            if final.get("band") in ("MEDIUM", "HIGH"):
                create_incident({
                    "session_id": session.meta.session_id,
                    "band": final["band"], "score": final["score"],
                    "action": "ESCALATE" if final["band"] == "HIGH" else "VERIFY",
                    "summary": f"{final['band']} risk call analysed "
                               f"({final.get('voice_verdict')})",
                    "payload": {"scenario": session.meta.scenario,
                                "findings": [f["code"] for f in final.get("findings", [])],
                                "quotes": final.get("context_quotes", [])[:10]},
                })
        except Exception as exc:                    # never break the stream
            log.warning("failed to persist session: %s", exc)
        await ws.send_json(final)
    except WebSocketDisconnect:
        log.info("stream client disconnected")
    except asyncio.TimeoutError:
        try:
            await ws.send_json({"type": "error", "error": "no start frame within 30s"})
        except Exception:
            pass
    except Exception as exc:                        # pragma: no cover
        log.exception("stream failed")
        try:
            await ws.send_json({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        if session is not None:
            session.close()
        try:
            await ws.close()
        except Exception:
            pass
