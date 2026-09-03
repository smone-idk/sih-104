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
from .fusion.findings import Finding, derive_findings
from .fusion.scorer import ComponentInput, FusionResult, fuse
from .policy.rules import apply_band_floors
from .fusion.smoothing import EMA
from .ingest.audio import load_audio
from .ingest.chunker import iter_windows
from .ingest.telephony import TelephonyConfig, degrade
from .ingest import vad as vad_mod
from .ml.registry import get_registry


def _scoring_map() -> dict[str, str]:
    """detector name -> fusion component, for detectors that carry weight.

    Read from the registry rather than hardcoded, so a zero-weight detector
    (AASIST, kept as a measured baseline) still runs and is reported but never
    reaches fusion. See DetectorRegistry.build and base.Detector.contributes.
    """
    return {d.name: d.feeds for d in get_registry().all()
            if d.feeds and d.contributes}


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
    #: 24 log-spaced band energies in dBFS — real FFT of this window, used for
    #: the streaming spectrogram. Measured, not decorative.
    spectrum: list[float] = field(default_factory=list)


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
    #: detectors that ran but carry ZERO fusion weight (measured baselines)
    baseline_detectors: dict[str, float] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    voice_verdict: str = "INDETERMINATE"
    vad_backend: str = ""
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
            "voice_verdict": self.voice_verdict,
            "findings": [f.as_dict() for f in self.findings],
            "vad_backend": self.vad_backend,
            "fusion": self.fusion.as_dict(),
            "detector_means": {
                k: (None if v is None else round(v, 4))
                for k, v in self.detector_means.items()
            },
            "baseline_detectors": {k: round(v, 4)
                                   for k, v in self.baseline_detectors.items()},
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
                    "spectrum": w.spectrum,
                    "detectors": w.detectors,
                    "latency_ms": {k: round(v, 2) for k, v in w.latency_ms.items()},
                }
                for w in self.windows
            ],
        }


def _rms(seg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))


_SPEC_BANDS = 24


def _spectrum(seg: np.ndarray, sr: int, n_bands: int = _SPEC_BANDS) -> list[float]:
    """Log-spaced band energies in dBFS for the streaming spectrogram."""
    if seg.size == 0:
        return [-90.0] * n_bands
    w = np.hanning(len(seg))
    # normalise by the window's coherent gain so magnitudes are in signal units
    # and the dB values are true dBFS (negative), not raw FFT bin magnitudes.
    mag = np.abs(np.fft.rfft(seg.astype(np.float64) * w)) / (np.sum(w) / 2.0)
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    edges = np.logspace(np.log10(80.0), np.log10(min(sr / 2, 8000.0)), n_bands + 1)
    out: list[float] = []
    for i in range(n_bands):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        e = float(np.mean(mag[m] ** 2)) if m.any() else 0.0
        out.append(round(float(10.0 * np.log10(e + 1e-12)), 2))
    return out


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


class WindowScorer:
    """Scores one 4 s window and accumulates session state.

    THE shared analysis core (§14). Batch (`analyze_audio`) and the WebSocket
    stream (`voiceshield.stream.session`) both drive this — there is no second
    scoring path for the demo. The caller decides whether a window is speech
    (batch runs VAD once over the whole clip; streaming runs it per window,
    since the clip does not exist yet), and the scorer stays agnostic.
    """

    def __init__(self, ctx: dict[str, Any] | None = None,
                 settings=None) -> None:
        self.s = settings or get_settings()
        self.ctx = dict(ctx or {})
        self.reg = get_registry()
        self.scoring = _scoring_map()
        self.ctx_inputs = _context_inputs(self.ctx)
        self.ema = EMA(self.s.ema_alpha)
        self.acc: dict[str, list[float]] = {n: [] for n in self.scoring}
        self.det_latency: dict[str, list[float]] = {n: [] for n in self.scoring}
        self.baseline_means: dict[str, list[float]] = {}
        self.windows: list[WindowScore] = []
        self.n_speech = 0

    def score(self, index: int, t_start: float, t_end: float,
              samples: np.ndarray, sr: int, is_speech: bool) -> WindowScore:
        ws = WindowScore(index, t_start, t_end, is_speech, None, None, _rms(samples))
        ws.spectrum = _spectrum(samples, sr)
        if not is_speech:
            self.windows.append(ws)
            return ws
        self.n_speech += 1

        comp_inputs: dict[str, ComponentInput] = {}
        for det in self.reg.available():
            res = det.analyze(samples, sr, self.ctx)
            ws.detectors[det.name] = res.as_dict()
            ws.latency_ms[det.name] = res.latency_ms
            comp = self.scoring.get(det.name)
            if comp is None:
                # zero-weight baseline detector: record it, never fuse it
                if res.available and res.score is not None:
                    self.baseline_means.setdefault(det.name, []).append(res.score)
                continue
            self.det_latency[det.name].append(res.latency_ms)
            if res.available and res.score is not None:
                self.acc[det.name].append(res.score)
                comp_inputs[comp] = ComponentInput(
                    value=res.score, available=True, note=res.note,
                    kind=res.kind, detail=res.detail)
            else:
                comp_inputs[comp] = ComponentInput(
                    value=None, available=False, note=res.note, kind=res.kind)

        comp_inputs.update(self.ctx_inputs)
        wfused = fuse(comp_inputs, self.ctx.get("weights"), self.s)
        ws.raw_score = wfused.score
        ws.ema_score = self.ema.update(wfused.score)
        self.windows.append(ws)
        return ws

    def aggregate(self):
        """Mean each weighted detector over its speech windows, fuse once.

        Returns (fusion, findings, verdict, detector_means, agg_inputs).
        """
        detector_means: dict[str, float | None] = {}
        agg_inputs: dict[str, ComponentInput] = {}
        for name, comp in self.scoring.items():
            vals = self.acc[name]
            det = self.reg.get(name)
            if vals:
                mean_v = float(np.mean(vals))
                detector_means[name] = mean_v
                last_detail = {}
                for w in reversed(self.windows):
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
                elif name == "speaker_consistency" and self.ctx.get("enrolled_embedding") is None:
                    note = "layer unavailable — no enrolled profile"
                agg_inputs[comp] = ComponentInput(
                    value=None, available=False,
                    kind=det.kind if det else "", note=note)

        agg_inputs.update(self.ctx_inputs)
        final = fuse(agg_inputs, self.ctx.get("weights"), self.s)
        findings, verdict = derive_findings(agg_inputs, self.s)
        # A linear blend cannot express "synthetic AND matches the target", so a
        # named rule floors the band. Applied HERE, in the one place batch and
        # streaming converge, so both paths get it (§14). See policy/rules.py.
        final.band, final.floors_applied = apply_band_floors(
            final.band_from_score, verdict, self.s)
        return final, findings, verdict, detector_means, agg_inputs

    def detector_latencies(self) -> dict[str, float]:
        return {f"detector.{n}": float(np.mean(v))
                for n, v in self.det_latency.items() if v}

    def baseline_detector_means(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.baseline_means.items() if v}


def analyze_audio(audio: np.ndarray, sr: int, *,
                  source: str = "upload",
                  ctx: dict[str, Any] | None = None,
                  telephony: TelephonyConfig | None = None) -> AnalysisResult:
    s = get_settings()
    ctx = dict(ctx or {})
    warnings: list[str] = []
    lat: dict[str, float] = {}

    t0 = time.perf_counter()
    telephony = telephony or TelephonyConfig()
    if telephony.enabled:
        audio, sr = degrade(audio, sr, telephony)
    lat["degrade"] = (time.perf_counter() - t0) * 1000

    duration = len(audio) / sr if sr else 0.0

    # VAD runs ONCE over the whole clip; windows then ask ratio_in() (§Phase 1.5 C)
    t_vad = time.perf_counter()
    vad = vad_mod.analyze(audio, sr)
    lat["vad"] = (time.perf_counter() - t_vad) * 1000
    sratio = vad.overall_ratio
    if vad.fallback_reason:
        warnings.append(f"VAD fell back to the energy gate: {vad.fallback_reason}")

    scorer = WindowScorer(ctx, s)
    for win in iter_windows(audio, sr):
        is_speech = vad.ratio_in(win.t_start, win.t_end) >= s.window_speech_ratio
        scorer.score(win.index, win.t_start, win.t_end, win.samples, sr, is_speech)

    windows = scorer.windows
    n_speech = scorer.n_speech
    if n_speech == 0:
        warnings.append("no speech detected — score is not meaningful")

    final, findings, verdict, detector_means, _ = scorer.aggregate()
    baseline_means = scorer.baseline_means
    lat.update(scorer.detector_latencies())
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
        baseline_detectors={k: float(np.mean(v)) for k, v in baseline_means.items() if v},
        findings=findings,
        voice_verdict=verdict,
        vad_backend=vad.backend,
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
