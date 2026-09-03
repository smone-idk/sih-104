"""Streaming analysis session (§4).

Audio arrives in chunks; this maintains the rolling buffer, cuts 4 s windows on
a 1 s hop, and hands each one to the SAME `WindowScorer` the batch path uses
(§14). There is no separate scoring code for the demo.

Difference from batch, stated plainly: batch runs VAD once over the whole clip,
which is impossible here because the clip does not exist yet. Streaming runs VAD
per window instead. Same backend, same threshold — slightly less context.

The ASR/context cadence (§4: transcripts land every 3-6 s, acoustic ticks at
1 Hz) is wired in Phase 3; `set_context_components()` is the hook.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import get_settings
from ..ingest import vad as vad_mod
from ..ingest.telephony import TelephonyConfig, degrade
from ..pipeline import WindowScore, WindowScorer


@dataclass
class SessionMeta:
    session_id: str
    source: str                  # "simulation" | "microphone" | "enterprise_stream"
    channel: str                 # human-readable authorized channel (§1)
    scenario: str | None = None
    profile_name: str | None = None
    telephony_degraded: bool = False
    started_at: float = field(default_factory=time.time)


class StreamSession:
    """Feed it audio, get scored windows back."""

    def __init__(self, *, source: str = "simulation", channel: str = "",
                 scenario: str | None = None,
                 ctx: dict[str, Any] | None = None,
                 telephony: TelephonyConfig | None = None,
                 sample_rate: int | None = None) -> None:
        self.s = get_settings()
        self.sr = sample_rate or self.s.sample_rate
        self.ctx = dict(ctx or {})
        self.telephony = telephony or TelephonyConfig()
        self.scorer = WindowScorer(self.ctx, self.s)

        self.meta = SessionMeta(
            session_id=str(uuid.uuid4()),
            source=source,
            channel=channel or "bundled demo clip (simulated transport)",
            scenario=scenario,
            profile_name=self.ctx.get("enrolled_speaker_name"),
            telephony_degraded=self.telephony.enabled,
        )

        self._win_n = int(round(self.s.window_seconds * self.sr))
        self._hop_n = int(round(self.s.hop_seconds * self.sr))
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_origin = 0      # absolute sample index of _buf[0]
        self._next_start = 0      # absolute sample index of the next window
        self._index = 0
        self._total_samples = 0
        self.closed = False

    # --- input ------------------------------------------------------
    def feed(self, samples: np.ndarray) -> list[WindowScore]:
        """Append audio; return every window that became complete."""
        if self.closed:
            return []
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self.telephony.enabled:
            x, _ = degrade(x, self.sr, self.telephony, seed=self._index)
        self._buf = np.concatenate([self._buf, x])
        self._total_samples += len(x)

        out: list[WindowScore] = []
        while True:
            off = self._next_start - self._buf_origin
            if off + self._win_n > len(self._buf):
                break
            seg = self._buf[off:off + self._win_n]
            out.append(self._score_segment(seg))
            self._next_start += self._hop_n
            # drop everything before the next window start
            drop = self._next_start - self._buf_origin
            if drop > 0:
                self._buf = self._buf[drop:]
                self._buf_origin = self._next_start
        return out

    def flush(self) -> list[WindowScore]:
        """Score the trailing remainder (zero-padded), as batch mode does."""
        if self.closed:
            return []
        off = self._next_start - self._buf_origin
        tail = self._buf[off:] if off < len(self._buf) else np.zeros(0, dtype=np.float32)
        # only if there is real audio left that no full window covered
        if len(tail) == 0 or self._next_start >= self._total_samples:
            return []
        seg = np.concatenate([tail, np.zeros(self._win_n - len(tail), dtype=np.float32)])
        ws = self._score_segment(seg, real_len=len(tail))
        self._next_start += self._hop_n
        return [ws]

    def _score_segment(self, seg: np.ndarray, real_len: int | None = None) -> WindowScore:
        t_start = self._next_start / self.sr
        t_end = t_start + (real_len or self._win_n) / self.sr
        vr = vad_mod.analyze(seg, self.sr)
        is_speech = vr.overall_ratio >= self.s.window_speech_ratio
        ws = self.scorer.score(self._index, t_start, t_end, seg, self.sr, is_speech)
        self._index += 1
        return ws

    # --- Phase 3 hook -----------------------------------------------
    def set_context_components(self, components: dict[str, Any]) -> None:
        """Called by the ASR/context worker when a transcript segment lands.
        Acoustic windows tick at 1 Hz; context updates at its own cadence (§4)."""
        self.ctx["context_components"] = components
        from ..pipeline import _context_inputs

        self.scorer.ctx_inputs = _context_inputs(self.ctx)

    # --- output -----------------------------------------------------
    def session_payload(self) -> dict:
        from ..ml.registry import get_registry

        return {
            "type": "session",
            "session_id": self.meta.session_id,
            "source": self.meta.source,
            "channel": self.meta.channel,
            "scenario": self.meta.scenario,
            "profile_name": self.meta.profile_name,
            "telephony_degraded": self.meta.telephony_degraded,
            "sample_rate": self.sr,
            "window_seconds": self.s.window_seconds,
            "hop_seconds": self.s.hop_seconds,
            "ema_alpha": self.s.ema_alpha,
            "bands": {"low_max": self.s.band_low_max, "high_min": self.s.band_high_min},
            "fusion_weights": self.s.fusion_weights(),
            "detectors": get_registry().inventory(),
            "device": self.s.device,
        }

    @staticmethod
    def window_payload(ws: WindowScore) -> dict:
        return {
            "type": "window",
            "index": ws.index,
            "t_start": round(ws.t_start, 3),
            "t_end": round(ws.t_end, 3),
            "is_speech": ws.is_speech,
            "raw_score": None if ws.raw_score is None else round(ws.raw_score, 2),
            "ema_score": None if ws.ema_score is None else round(ws.ema_score, 2),
            "rms_energy": round(ws.rms_energy, 5),
            "spectrum": ws.spectrum,
            "detectors": ws.detectors,
            "latency_ms": {k: round(v, 2) for k, v in ws.latency_ms.items()},
        }

    def final_payload(self) -> dict:
        final, findings, verdict, det_means, _ = self.scorer.aggregate()
        warnings: list[str] = []
        if self.scorer.n_speech == 0:
            warnings.append("no speech detected — score is not meaningful")
        return {
            "type": "final",
            "session_id": self.meta.session_id,
            "score": round(final.score, 2),
            "band": final.band,
            "voice_verdict": verdict,
            "findings": [f.as_dict() for f in findings],
            "fusion": final.as_dict(),
            "detector_means": {k: (None if v is None else round(v, 4))
                               for k, v in det_means.items()},
            "baseline_detectors": {k: round(v, 4) for k, v in
                                   self.scorer.baseline_detector_means().items()},
            "latency_ms": {k: round(v, 2)
                           for k, v in self.scorer.detector_latencies().items()},
            "n_windows": len(self.scorer.windows),
            "n_speech_windows": self.scorer.n_speech,
            "duration_s": round(self._total_samples / self.sr, 3),
            "warnings": warnings,
        }

    def close(self) -> None:
        self.closed = True
        self._buf = np.zeros(0, dtype=np.float32)   # §12 — audio not retained
