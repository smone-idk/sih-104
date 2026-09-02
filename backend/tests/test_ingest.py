"""Ingest: chunking math, VAD, telephony degradation."""
from __future__ import annotations

import numpy as np
import pytest

from voiceshield.ingest.chunker import iter_windows
from voiceshield.ingest.telephony import TelephonyConfig, degrade
from voiceshield.ingest.vad import speech_mask, segments


def test_window_count_4s_1s_hop(sr):
    audio = np.zeros(int(10 * sr), dtype=np.float32)   # 10 s
    wins = list(iter_windows(audio, sr, 4.0, 1.0))
    # windows start at 0,1,2,3,4,5,6 -> 7 full windows, no remainder
    assert [w.index for w in wins] == list(range(7))
    assert all(len(w.samples) == 4 * sr for w in wins)
    assert wins[0].t_start == 0.0 and wins[-1].t_end == pytest.approx(10.0)


def test_short_audio_single_padded_window(sr):
    audio = np.ones(int(1.5 * sr), dtype=np.float32)
    wins = list(iter_windows(audio, sr, 4.0, 1.0))
    assert len(wins) == 1 and len(wins[0].samples) == 4 * sr


def test_vad_marks_silence_and_tone(sr, voiced_like):
    silence = np.zeros(2 * sr, dtype=np.float32)
    clip = np.concatenate([silence, voiced_like, silence])
    mask, _ = speech_mask(clip, sr)
    assert mask.any()
    segs = segments(clip, sr)
    assert segs, "should find at least one speech segment"
    s0 = segs[0]
    assert 1.0 < s0.start < 3.0        # speech starts ~2 s in
    assert s0.end > s0.start


def test_telephony_degrade_deterministic_and_changes_signal(sr, voiced_like):
    cfg = TelephonyConfig(enabled=True, mu_law=True, add_noise=True, snr_db=15.0)
    a, sra = degrade(voiced_like, sr, cfg, seed=1)
    b, srb = degrade(voiced_like, sr, cfg, seed=1)
    assert sra == srb == sr
    assert np.allclose(a, b)                                   # deterministic
    assert not np.allclose(a[: len(voiced_like)], voiced_like)  # actually degraded
    # narrowband: energy above 4 kHz should drop vs the original
    def hf_energy(x):
        X = np.abs(np.fft.rfft(x))
        f = np.fft.rfftfreq(len(x), 1 / sr)
        return float(np.sum(X[f > 4000] ** 2))
    assert hf_energy(a) < hf_energy(voiced_like)


def test_telephony_disabled_is_passthrough(sr, voiced_like):
    a, _ = degrade(voiced_like, sr, TelephonyConfig(enabled=False))
    assert a is voiced_like
