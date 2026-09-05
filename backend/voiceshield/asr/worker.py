"""faster-whisper wrapper — real ASR, decoupled from the acoustic loop (§4).

The acoustic layers tick once per 1 s hop. Whisper does NOT: it runs over
VAD-segmented utterances (roughly every 3-6 s), because transcribing a 4 s
window every second would transcribe the same words up to four times and make
the context engine count one "transfer 25 lakh" as four separate hits.

Segmentation source of truth is the Silero VAD pass the acoustic loop already
runs — see docs/PHASES.md "Phase 3 — segmentation decision". This module never
segments audio itself; it transcribes whatever span it is handed.

Model is loaded once and kept warm, like every other model (§1).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import get_settings

log = logging.getLogger("voiceshield.asr")


@dataclass
class TranscriptSegment:
    """One transcribed utterance, positioned in the SESSION timeline."""
    index: int
    t_start: float                # seconds from session start
    t_end: float
    text: str
    language: str = ""
    language_probability: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    latency_ms: float = 0.0
    #: character offset of this segment's text inside the rolling transcript,
    #: so a context match can be traced back to the exact utterance and time.
    char_start: int = 0
    char_end: int = 0
    #: False => below the ASR confidence gate. The segment is still transcribed,
    #: still shown in the UI and still stored — it just does not reach the
    #: context engine, so it can never produce a quote. Dropping it silently
    #: would hide why a signal is missing.
    confident: bool = True
    gate_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "t_start": round(self.t_start, 2),
            "t_end": round(self.t_end, 2),
            "text": self.text,
            "language": self.language,
            "language_probability": round(self.language_probability, 3),
            "avg_logprob": round(self.avg_logprob, 3),
            "no_speech_prob": round(self.no_speech_prob, 3),
            "latency_ms": round(self.latency_ms, 1),
            "char_start": self.char_start,
            "char_end": self.char_end,
            "confident": self.confident,
            "gate_reason": self.gate_reason,
        }


class AsrWorker:
    """Blocking transcription. Callers wanting concurrency should run
    `transcribe()` in a thread executor — faster-whisper releases the GIL
    during inference."""

    _instance: "AsrWorker | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self.available = False
        self.load_error = ""
        self.device = "cpu"
        self.model_id = ""

    # --- lifecycle ---------------------------------------------------
    def load(self) -> None:
        s = get_settings()
        try:
            from faster_whisper import WhisperModel

            path = s.whisper_dir
            if not (path / "model.bin").exists():
                raise FileNotFoundError(
                    f"{path} missing — run scripts/fetch_models.py")
            self._model = WhisperModel(
                str(path), device=s.device,
                compute_type=s.whisper_compute_type,
                local_files_only=True,
            )
            self.device = s.device
            self.model_id = f"faster-whisper-{s.whisper_model} ({s.whisper_compute_type})"
            self.available = True
            log.info("Whisper loaded: %s on %s", self.model_id, s.device)
        except Exception as exc:
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.warning("Whisper unavailable: %s", self.load_error)

    @classmethod
    def get(cls) -> "AsrWorker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    w = cls()
                    w.load()
                    cls._instance = w
        return cls._instance

    def info(self) -> dict:
        return {
            "name": "asr_whisper",
            "kind": "pretrained",
            "available": self.available,
            "device": self.device,
            "model_id": self.model_id,
            "load_error": self.load_error,
            "licence": "MIT (faster-whisper / CTranslate2)",
        }

    # --- inference ---------------------------------------------------
    def transcribe(self, audio: np.ndarray, sr: int, *,
                   t_offset: float = 0.0,
                   index: int = 0,
                   language: str | None = None) -> list[TranscriptSegment]:
        """Transcribe one utterance span. `t_offset` positions it in the
        session timeline so context matches can cite a real timestamp."""
        if not self.available:
            return []
        s = get_settings()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sr != 16000:
            from ..ingest.audio import resample
            x = resample(x, sr, 16000)

        t0 = time.perf_counter()
        try:
            segs, info = self._model.transcribe(
                x,
                language=language or (s.asr_language or None),
                beam_size=s.asr_beam_size,
                vad_filter=False,           # Silero already segmented this span
                condition_on_previous_text=False,   # avoid runaway hallucination
            )
            out: list[TranscriptSegment] = []
            for i, sg in enumerate(segs):
                text = (sg.text or "").strip()
                if not text:
                    continue
                out.append(TranscriptSegment(
                    index=index + i,
                    t_start=t_offset + float(sg.start),
                    t_end=t_offset + float(sg.end),
                    text=text,
                    language=getattr(info, "language", "") or "",
                    language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
                    avg_logprob=float(getattr(sg, "avg_logprob", 0.0) or 0.0),
                    no_speech_prob=float(getattr(sg, "no_speech_prob", 0.0) or 0.0),
                ))
            ms = (time.perf_counter() - t0) * 1000
            for sg in out:
                sg.latency_ms = ms / max(1, len(out))
            return out
        except Exception as exc:                    # never take down the session
            log.warning("transcription failed: %s", exc)
            return []


def gate_segment(sg: TranscriptSegment, settings=None) -> tuple[bool, str]:
    """Should this segment be trusted enough to feed the context engine?

    Whisper invents words on XTTS-cloned audio (see LIMITATIONS.md §7), and an
    invented word matching a context rule produces a flag quoting speech nobody
    made. Thresholds are Settings values, chosen by sweeping against ground
    truth — not literals.
    """
    s = settings or get_settings()
    if not s.asr_confidence_gate:
        return True, ""
    if sg.avg_logprob < s.asr_min_avg_logprob:
        return False, (f"low ASR confidence (avg_logprob {sg.avg_logprob:.2f} "
                       f"< {s.asr_min_avg_logprob})")
    if sg.no_speech_prob > s.asr_max_no_speech_prob:
        return False, (f"likely non-speech (no_speech_prob {sg.no_speech_prob:.2f} "
                       f"> {s.asr_max_no_speech_prob})")
    return True, ""


@dataclass
class Transcript:
    """Rolling transcript with char offsets, so every context match can point
    at the exact quote AND the utterance it came from.

    `text` contains ONLY confident segments — that is what the context engine
    reads. Low-confidence segments stay in `segments` flagged `confident=False`
    so the UI can show them as discarded rather than making them vanish.
    """
    segments: list[TranscriptSegment] = field(default_factory=list)
    text: str = ""

    def add(self, segs: list[TranscriptSegment], settings=None) -> None:
        for sg in segs:
            sg.confident, sg.gate_reason = gate_segment(sg, settings)
            if not sg.confident:
                # not part of `text`, so it can never yield a context quote
                sg.char_start = sg.char_end = len(self.text)
                self.segments.append(sg)
                continue
            if self.text:
                self.text += " "
            sg.char_start = len(self.text)
            self.text += sg.text
            sg.char_end = len(self.text)
            self.segments.append(sg)

    @property
    def discarded(self) -> list[TranscriptSegment]:
        return [s for s in self.segments if not s.confident]

    def segment_at(self, char_pos: int) -> TranscriptSegment | None:
        for sg in self.segments:
            if sg.confident and sg.char_start <= char_pos < sg.char_end:
                return sg
        return None

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": [s.as_dict() for s in self.segments],
            "n_segments": len(self.segments),
            "n_discarded": len(self.discarded),
            "duration_s": round(self.segments[-1].t_end, 2) if self.segments else 0.0,
            "gate": {
                "enabled": get_settings().asr_confidence_gate,
                "min_avg_logprob": get_settings().asr_min_avg_logprob,
                "max_no_speech_prob": get_settings().asr_max_no_speech_prob,
            },
        }
