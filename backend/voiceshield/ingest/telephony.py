"""Telephony degradation chain (§8).

Real narrowband path: resample to 8 kHz -> G.711 mu-law encode/decode ->
optional additive noise at a target SNR -> back to 16 kHz for the models.
Used by the before/after comparison. Not applied unless the caller asks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import resample


@dataclass
class TelephonyConfig:
    enabled: bool = False
    narrowband_hz: int = 8000
    mu_law: bool = True
    add_noise: bool = False
    snr_db: float = 20.0


def _mu_law_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * mu).astype(np.int16)  # 0..255
    return q


def _mu_law_decode(q: np.ndarray, mu: int = 255) -> np.ndarray:
    y = q.astype(np.float32) / mu * 2 - 1
    x = np.sign(y) * (1.0 / mu) * (np.power(1 + mu, np.abs(y)) - 1.0)
    return x.astype(np.float32)


def _add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = float(np.mean(x.astype(np.float64) ** 2)) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
    return x + noise


def degrade(audio: np.ndarray, sr: int, cfg: TelephonyConfig,
            seed: int = 0) -> tuple[np.ndarray, int]:
    """Return (degraded audio at the SAME sr as input, sr). Deterministic given seed."""
    if not cfg.enabled:
        return audio, sr
    rng = np.random.default_rng(seed)
    nb = resample(audio, sr, cfg.narrowband_hz)
    if cfg.mu_law:
        nb = _mu_law_decode(_mu_law_encode(nb))
    if cfg.add_noise:
        nb = _add_noise(nb, cfg.snr_db, rng)
    out = resample(nb, cfg.narrowband_hz, sr)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32), sr
