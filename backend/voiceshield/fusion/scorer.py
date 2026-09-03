"""Fusion: weighted linear combination of the risk components (§6).

  score = 100 * sum(effective_weight_i * value_i)   over AVAILABLE components

Rules:
  * weights are runtime-configurable (settings / API).
  * a component that reports `available=False` is dropped and its weight is
    redistributed proportionally across the remaining components — never
    replaced with a fake value.
  * bands: LOW < 40, MEDIUM 40..70, HIGH > 70 (thresholds in settings).
  * explainability = per-component raw value, nominal weight, effective weight
    and contribution in points, sorted by contribution.

EMA smoothing for the streamed score lives in `smoothing.py`; batch mode
aggregates per-window scores directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings, get_settings

DISCLAIMER = (
    "Prototype risk model. Weights are expert-elicited priors, not fitted on "
    "labelled fraud data. Not a calibrated fraud probability."
)

# component -> the detector/signal that feeds it
COMPONENTS = (
    "voice_authenticity",
    "speaker_consistency",
    "prosody_anomaly",
    "caller_trust",
    "transaction_context",
    "behavioural_risk",
)


@dataclass
class ComponentInput:
    value: float | None          # 0..1, higher = more risk
    available: bool = True
    note: str = ""
    kind: str = ""               # detector kind badge, passed through for the UI
    detail: dict = field(default_factory=dict)


@dataclass
class ComponentContribution:
    name: str
    raw_value: float | None
    weight: float                # nominal weight from settings
    effective_weight: float      # after redistribution (0 if unavailable)
    contribution_points: float   # effective_weight * value * 100
    available: bool
    note: str
    kind: str
    detail: dict


@dataclass
class FusionResult:
    score: float                 # 0..100
    band: str                    # LOW | MEDIUM | HIGH (after any band floor)
    components: list[ComponentContribution]
    redistributed: bool
    available_components: list[str]
    unavailable_components: list[str]
    #: band implied by the score alone, before policy floors were applied
    band_from_score: str = ""
    #: named rules that raised the band (policy/rules.py). Empty in the normal case.
    floors_applied: list = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "band": self.band,
            "band_from_score": self.band_from_score or self.band,
            "floors_applied": [f.as_dict() for f in self.floors_applied],
            "redistributed": self.redistributed,
            "available_components": self.available_components,
            "unavailable_components": self.unavailable_components,
            "components": [
                {
                    "name": c.name,
                    "raw_value": None if c.raw_value is None else round(c.raw_value, 4),
                    "weight": round(c.weight, 4),
                    "effective_weight": round(c.effective_weight, 4),
                    "contribution_points": round(c.contribution_points, 2),
                    "available": c.available,
                    "note": c.note,
                    "kind": c.kind,
                    "detail": c.detail,
                }
                for c in self.components
            ],
            "disclaimer": self.disclaimer,
        }


def band_for(score: float, s: Settings | None = None) -> str:
    s = s or get_settings()
    if score < s.band_low_max:
        return "LOW"
    if score > s.band_high_min:
        return "HIGH"
    return "MEDIUM"


def fuse(inputs: dict[str, ComponentInput],
         weights: dict[str, float] | None = None,
         settings: Settings | None = None) -> FusionResult:
    s = settings or get_settings()
    weights = weights or s.fusion_weights()

    avail = {
        name: inp for name, inp in inputs.items()
        if inp.available and inp.value is not None
    }
    avail_weight = sum(weights.get(n, 0.0) for n in avail) or 1e-9
    redistributed = len(avail) != len([i for i in inputs.values()
                                       if i.available and i.value is not None]) or \
        any(not inp.available or inp.value is None for inp in inputs.values())

    contribs: list[ComponentContribution] = []
    score = 0.0
    for name in COMPONENTS:
        inp = inputs.get(name, ComponentInput(value=None, available=False,
                                              note="not provided"))
        w = weights.get(name, 0.0)
        if name in avail:
            eff = w / avail_weight
            pts = eff * float(inp.value) * 100.0
            score += pts
            contribs.append(ComponentContribution(
                name, float(inp.value), w, eff, pts, True, inp.note, inp.kind, inp.detail))
        else:
            contribs.append(ComponentContribution(
                name, inp.value, w, 0.0, 0.0, False,
                inp.note or "layer unavailable", inp.kind, inp.detail))

    contribs.sort(key=lambda c: c.contribution_points, reverse=True)
    score = max(0.0, min(100.0, score))
    return FusionResult(
        score=score,
        band=band_for(score, s),
        band_from_score=band_for(score, s),
        components=contribs,
        redistributed=bool(redistributed),
        available_components=[n for n in COMPONENTS if n in avail],
        unavailable_components=[n for n in COMPONENTS if n not in avail],
    )
