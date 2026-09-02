"""EMA smoothing for the streamed risk score (§4).

Raw per-window scores jitter on real audio; the UI shows both the raw dots and
this smoothed line so the smoothing is visibly honest. alpha ~= 0.3.
"""
from __future__ import annotations

from ..config import get_settings


class EMA:
    def __init__(self, alpha: float | None = None):
        self.alpha = alpha if alpha is not None else get_settings().ema_alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value
