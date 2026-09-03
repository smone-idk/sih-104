"""Utterance segmentation for the ASR worker (§4).

Implements the Phase 3 decision recorded in docs/PHASES.md: the ASR worker gets
utterance-shaped input but does NOT run its own VAD. It consumes the speech
segments from the SAME Silero pass the acoustic loop already ran, then merges
and splits them into ASR-sized utterances.

One segmentation source of truth means "the acoustic layer scored this window
but the transcript has no words there" can never happen from two VADs
disagreeing.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import get_settings


@dataclass
class Utterance:
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def utterances_from_segments(segments, settings=None) -> list[Utterance]:
    """Merge VAD speech segments up to `asr_min_segment_seconds` and split
    anything longer than `asr_max_segment_seconds`.

    Whisper degrades on very short fragments (it has too little context and
    hallucinates), and long spans delay the context update, so we aim for the
    3-6 s band §4 specifies.
    """
    s = settings or get_settings()
    lo, hi = s.asr_min_segment_seconds, s.asr_max_segment_seconds
    out: list[Utterance] = []
    cur: Utterance | None = None

    for seg in segments:
        if cur is None:
            cur = Utterance(seg.start, seg.end)
        elif seg.end - cur.t_start <= hi:
            # extend across the pause — keeps a sentence together
            cur.t_end = seg.end
        else:
            out.append(cur)
            cur = Utterance(seg.start, seg.end)

        if cur.duration >= lo and cur.duration >= hi * 0.75:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)

    # split anything still over the max (one very long unbroken utterance)
    split: list[Utterance] = []
    for u in out:
        if u.duration <= hi * 1.5:
            split.append(u)
            continue
        n = int(u.duration // hi) + 1
        step = u.duration / n
        for i in range(n):
            split.append(Utterance(u.t_start + i * step,
                                   min(u.t_end, u.t_start + (i + 1) * step)))
    return [u for u in split if u.duration > 0.2]


def transcribe_utterances(audio, sr, segments, *, worker=None,
                          settings=None, language: str | None = None):
    """Run the ASR worker over VAD-derived utterances. Returns a Transcript."""
    import numpy as np

    from .worker import AsrWorker, Transcript

    s = settings or get_settings()
    worker = worker or AsrWorker.get()
    t = Transcript()
    if not worker.available:
        return t
    for u in utterances_from_segments(segments, s):
        a = int(u.t_start * sr)
        b = min(len(audio), int(u.t_end * sr))
        if b - a < int(0.2 * sr):
            continue
        segs = worker.transcribe(np.asarray(audio[a:b]), sr,
                                 t_offset=u.t_start,
                                 index=len(t.segments),
                                 language=language)
        t.add(segs)
    return t
