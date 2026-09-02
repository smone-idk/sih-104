"""Shared fixtures. Synthetic signals are deterministic (fixed seed / closed
form) so detector tests have known-input/known-output behaviour without shipping
audio blobs (§14)."""
from __future__ import annotations

import numpy as np
import pytest

SR = 16000


def _tone(freq: float, dur: float, sr: int = SR, amp: float = 0.3,
          vibrato_hz: float = 0.0, vibrato_depth: float = 0.0) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    inst = freq + vibrato_depth * np.sin(2 * np.pi * vibrato_hz * t)
    phase = 2 * np.pi * np.cumsum(inst) / sr
    return (amp * np.sin(phase)).astype(np.float32)


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def pure_tone() -> np.ndarray:
    """4 s steady 180 Hz tone — maximally 'unnatural' prosody, near-linear phase."""
    return _tone(180.0, 4.0)


@pytest.fixture
def voiced_like() -> np.ndarray:
    """4 s tone with vibrato + formant-ish shaping + light noise — closer to
    natural voiced speech than a pure tone."""
    rng = np.random.default_rng(42)
    base = _tone(140.0, 4.0, amp=0.25, vibrato_hz=5.0, vibrato_depth=8.0)
    h2 = _tone(280.0, 4.0, amp=0.10, vibrato_hz=5.0, vibrato_depth=16.0)
    h3 = _tone(2100.0, 4.0, amp=0.04)
    env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3.0 * np.arange(len(base)) / SR))
    sig = (base + h2 + h3) * env + 0.005 * rng.standard_normal(len(base)).astype(np.float32)
    return (sig / (np.max(np.abs(sig)) + 1e-8) * 0.5).astype(np.float32)


@pytest.fixture
def white_noise() -> np.ndarray:
    rng = np.random.default_rng(7)
    return (0.1 * rng.standard_normal(SR * 4)).astype(np.float32)


@pytest.fixture
def silence() -> np.ndarray:
    return np.zeros(SR * 4, dtype=np.float32)
