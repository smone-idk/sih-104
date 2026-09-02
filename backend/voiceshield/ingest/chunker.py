"""Sliding-window chunking — 4 s window, 1 s hop (§4).

The same generator feeds batch analysis (iterate the whole file) and streaming
(feed it a growing buffer). One code path (§14).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..config import get_settings


@dataclass
class Window:
    index: int
    t_start: float
    t_end: float
    samples: np.ndarray


def iter_windows(audio: np.ndarray, sr: int,
                 window_s: float | None = None,
                 hop_s: float | None = None,
                 pad_last: bool = True) -> Iterator[Window]:
    s = get_settings()
    window_s = window_s or s.window_seconds
    hop_s = hop_s or s.hop_seconds
    w = int(round(window_s * sr))
    h = int(round(hop_s * sr))
    if len(audio) == 0:
        return
    if len(audio) <= w:
        seg = audio
        if pad_last and len(seg) < w:
            seg = np.concatenate([seg, np.zeros(w - len(seg), dtype=audio.dtype)])
        yield Window(0, 0.0, len(audio) / sr, seg)
        return
    idx = 0
    start = 0
    prev_end = 0
    while start + w <= len(audio):
        yield Window(idx, start / sr, (start + w) / sr, audio[start:start + w])
        prev_end = start + w
        idx += 1
        start += h
    # trailing remainder — only if the last full window didn't already cover the tail
    if start < len(audio) and prev_end < len(audio):
        seg = audio[start:]
        if pad_last:
            seg = np.concatenate([seg, np.zeros(w - len(seg), dtype=audio.dtype)])
        yield Window(idx, start / sr, len(audio) / sr, seg)
