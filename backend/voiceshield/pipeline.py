"""The single analysis pipeline (§14).

Batch (whole file), simulation and streaming all call `analyze_audio`. Batch =
run every 4 s / 1 s-hop window, average each detector's score over the speech
windows, fuse once, and also keep the per-window timeline for the chart.

Context-derived components (caller_trust, transaction_context, behavioural_risk)
are wired in Phase 3 with the ASR + context engine. Until then they report
`available=False` and fusion redistributes their weight — no fake values.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import get_settings
from .fusion.scorer import ComponentInput, FusionResult, fuse
from .fusion.smoothing import EMA
from .ingest.audio import load_audio
from .ingest.chunker import iter_windows
from .ingest.telephony import TelephonyConfig, degrade
from .ingest.vad import speech_mask, speech_ratio
from .ml.registry import get_registry

# detector name -> fusion component it feeds
_DETECTOR_COMPONENT = {
    "synthetic_speech": "voice_authenticity",
    "speaker_consistency": "speaker_consistency",
    "prosody_anomaly": "prosody_anomaly",
}


@dataclass
class WindowScore:
    index: int
    t_start: float
    t_end: float
    is_speech: bool
    raw_score: float | None
    ema_score: float | None
    rms_energy: float
    detectors: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    source: str
    duration_s: float
    sample_rate: int
    device: str
    n_windows: int
    n_speech_windows: int
    speech_ratio: float
    telephony_degraded: bool
    fusion: FusionResult
    windows: list[WindowScore]
    detector_means: dict[str, float | None]
    latency_ms: dict[str, float]
    context: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "device": self.device,
            "n_windows": self.n_windows,
            "n_speech_windows": self.n_speech_windows,
            "speech_ratio": round(self.speech_ratio, 3),
            "telephony_degraded": self.telephony_degraded,
            "score": round(self.fusion.score, 2),
            "band": self.fusion.band,
            "fusion": self.fusion.as_dict(),
            "detector_means": {
                k: (None if v is None else round(v, 4))
                for k, v in self.detector_means.items()
            },
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
            "context": self.context,
            "warnings": self.warnings,
            "windows": [
                {
                    "index": w.index,
                    "t_start": round(w.t_start, 3),
                    "t_end": round(w.t_end, 3),
                    "is_speech": w.is_speech,
                    "raw_score": None if w.raw_score is None else round(w.raw_score, 2),
                    "ema_score": None if w.ema_score is None else round(w.ema_score, 2),
                    "rms_energy": round(w.rms_energy, 5),
                    "detectors": w.detectors,
                    "latency_ms": {k: round(v, 2) for k, v in w.latency_ms.items()},
                }
                for w in self.windows
            ],
        }


def _window_is_speech(seg: np.ndarray, sr: int) -> tuple[bool, float]:
    mask, _ = speech_mask(seg, sr)
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))
    ratio = float(np.mean(mask)) if mask.size else 0.0
    return ratio >= 0.25, rms


def _context_inputs(ctx: dict[str, Any]) -> dict[str, ComponentInput]:
    """Phase 1: nothing to derive without ASR. Emit explicit 'unavailable'
    for the three context components so the explainability panel shows why and
    fusion redistributes their weight."""
    provided = ctx.get("context_components") or {}
    out: dict[str, ComponentInput] = {}
    for name in ("caller_trust", "transaction_context", "behavioural_risk"):
        if name in provided:
            c = provided[name]
            out[name] = ComponentInput(value=c.get("value"),
                                       available=c.get("available", True),
                                       note=c.get("note", ""),
                                       kind=c.get("kind", "heuristic"),
                                       detail=c.get("detail", {}))
        else:
            out[name] = ComponentInput(
                value=None, available=False, kind="heuristic",
                note="context engine not wired yet (Phase 3)")
    return out


def analyze_audio(audio: np.ndarray, sr: int, *,
                  source: str = "upload",
                  ctx: dict[str, Any] | None = None,
                  telephony: TelephonyConfig | None = None) -> AnalysisResult:
    s = get_settings()
    ctx = dict(ctx or {})
    reg = get_registry()
    warnings: list[str] = []
    lat: dict[str, float] = {}

    t0 = time.perf_counter()
    telephony = telephony or TelephonyConfig()
    if telephony.enabled:
        audio, sr = degrade(audio, sr, telephony)
    lat["degrade"] = (time.perf_counter() - t0) * 1000

    duration = len(audio) / sr if sr else 0.0
    sratio = speech_ratio(audio, sr)

    # per-detector accumulation over speech windows
    acc: dict[str, list[float]] = {n: [] for n in _DETECTOR_COMPONENT}
    det_latency: dict[str, list[float]] = {n: [] for n in _DETECTOR_COMPONENT}
    ema = EMA(s.ema_alpha)
    windows: list[WindowScore] = []
    n_speech = 0

    ctx_inputs = _context_inputs(ctx)

    for win in iter_windows(audio, sr):
        is_speech, rms = _window_is_speech(win.samples, sr)
        ws = WindowScore(win.index, win.t_start, win.t_end, is_speech, None, None, rms)
        if not is_speech:
            windows.append(ws)
            continue
        n_speech += 1

        comp_inputs: dict[str, ComponentInput] = {}
        for det in reg.available():
            comp = _DETECTOR_COMPONENT.get(det.name)
            if comp is None:
                continue
            res = det.analyze(win.samples, sr, ctx)
            ws.detectors[det.name] = res.as_dict()
            ws.latency_ms[det.name] = res.latency_ms
            det_latency[det.name].append(res.latency_ms)
            if res.available and res.score is not None:
                acc[det.name].append(res.score)
                comp_inputs[comp] = ComponentInput(
                    value=res.score, available=True, note=res.note,
                    kind=res.kind, detail=res.detail)
            else:
                comp_inputs[comp] = ComponentInput(
                    value=None, available=False, note=res.note, kind=res.kind)

        comp_inputs.update({k: v for k, v in ctx_inputs.items()})
        wfused = fuse(comp_inputs, ctx.get("weights"), s)
        ws.raw_score = wfused.score
        ws.ema_score = ema.update(wfused.score)
        windows.append(ws)

    # aggregate: mean of each detector over speech windows -> single fusion
    detector_means: dict[str, float | None] = {}
    agg_inputs: dict[str, ComponentInput] = {}
    for name, comp in _DETECTOR_COMPONENT.items():
        vals = acc[name]
        det = reg.get(name)
        if vals:
            mean_v = float(np.mean(vals))
            detector_means[name] = mean_v
            # carry a representative detail from the last window for the panel
            last_detail = {}
            for w in reversed(windows):
                if name in w.detectors and w.detectors[name].get("available"):
                    last_detail = w.detectors[name].get("detail", {})
                    break
            agg_inputs[comp] = ComponentInput(
                value=mean_v, available=True, kind=det.kind if det else "",
                note=f"mean over {len(vals)} speech window(s)", detail=last_detail)
        else:
            detector_means[name] = None
            note = "no speech windows to score"
            if det and not det.available:
                note = det.load_error or "detector unavailable"
            elif name == "speaker_consistency" and ctx.get("enrolled_embedding") is None:
                note = "layer unavailable — no enrolled profile"
            agg_inputs[comp] = ComponentInput(
                value=None, available=False, kind=det.kind if det else "", note=note)

    agg_inputs.update(ctx_inputs)
    if n_speech == 0:
        warnings.append("no speech detected — score is not meaningful")

    final = fuse(agg_inputs, ctx.get("weights"), s)

    for name, lst in det_latency.items():
        if lst:
            lat[f"detector.{name}"] = float(np.mean(lst))
    lat["total"] = (time.perf_counter() - t0) * 1000

    return AnalysisResult(
        source=source,
        duration_s=duration,
        sample_rate=sr,
        device=s.device,
        n_windows=len(windows),
        n_speech_windows=n_speech,
        speech_ratio=sratio,
        telephony_degraded=telephony.enabled,
        fusion=final,
        windows=windows,
        detector_means=detector_means,
        latency_ms=lat,
        context={"components_wired": False, "phase": 1,
                 **{k: v for k, v in ctx.items() if k in ("scenario", "channel")}},
        warnings=warnings,
    )


def analyze_file(path: str | Path, *, source: str = "upload",
                 ctx: dict[str, Any] | None = None,
                 telephony: TelephonyConfig | None = None) -> AnalysisResult:
    s = get_settings()
    ctx = dict(ctx or {})
    t0 = time.perf_counter()
    audio, sr = load_audio(path)
    load_ms = (time.perf_counter() - t0) * 1000

    # resolve an enrolled speaker profile unless one was passed in
    if "enrolled_embedding" not in ctx:
        try:
            from .store.repo import get_profile

            prof = get_profile(ctx.get("profile_id"))
            if prof is not None:
                ctx["enrolled_embedding"] = prof["embedding"]
                ctx["enrolled_speaker_name"] = prof["display_name"]
        except Exception:
            pass

    res = analyze_audio(audio, sr, source=source, ctx=ctx, telephony=telephony)
    res.latency_ms["load_audio"] = load_ms
    res.source = str(path)

    if not s.retain_audio:
        del audio  # §12 — not written to disk, dropped from memory here
    return res
