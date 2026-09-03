"""Band floors — rule-level overrides that a linear blend cannot express.

Why this exists
---------------
Fusion (§6) is a weighted linear combination. That is honest and explainable,
but it is *additive*: it has no interaction term, so it cannot say
"synthetic AND matching the enrolled speaker is worse than either alone".

That limitation bites in exactly the case PS26104 is named after. A successful
voice clone of the enrolled person is SUPPOSED to match the profile, so
`speaker_consistency` correctly reports low suspicion and correctly contributes
few points. The result, measured in Phase 2:

    XTTS clone of the enrolled CFO   61.0  MEDIUM   <- the headline attack
    Piper TTS in an unrelated voice  78.9  HIGH     <- the crude, easy case

The clone lands *below* off-the-shelf TTS. Every component was individually
right and the aggregate was still backwards.

The fix, and why it is this one
-------------------------------
We do NOT reweight. Reweighting to force the clone above the TTS case would
distort component semantics that are currently correct and honest — and it would
still be additive, so it would only paper over the ordering for these particular
clips.

We also do not multiply in an interaction term. A magic coefficient is hard to
justify to a judge and hard to defend in LIMITATIONS.

Instead: a small set of NAMED rules that floor the risk BAND when a verdict
fires. The score is left exactly as fusion computed it, the band is raised, and
the reason is carried through to the UI and the incident record. A judge can
read the rule in one sentence, and the numbers underneath stay untouched.

Thresholds and the floor band are Settings values, not literals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings, get_settings

_BAND_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


@dataclass
class AppliedFloor:
    code: str
    description: str
    from_band: str
    to_band: str

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "description": self.description,
            "from_band": self.from_band,
            "to_band": self.to_band,
        }


@dataclass
class FloorRule:
    code: str
    verdict: str
    floor_band: str
    description: str


def rules_for(s: Settings) -> list[FloorRule]:
    return [
        FloorRule(
            code="cloned_voice_floor",
            verdict="CLONED_VOICE",
            floor_band=s.cloned_voice_band_floor,
            description=(
                "Speech is machine-generated AND matches the enrolled speaker "
                f"(synthetic ≥ {s.synthetic_high_threshold:g}, speaker suspicion "
                f"≤ {s.speaker_match_threshold:g}). A weighted linear score cannot "
                "represent this interaction — high speaker similarity correctly "
                "lowers the speaker component — so the band is floored at "
                f"{s.cloned_voice_band_floor}. The score itself is unchanged."
            ),
        ),
    ]


def apply_band_floors(band: str, verdict: str,
                      settings: Settings | None = None
                      ) -> tuple[str, list[AppliedFloor]]:
    """Return (possibly raised band, list of rules that fired)."""
    s = settings or get_settings()
    if not s.enable_band_floors:
        return band, []
    applied: list[AppliedFloor] = []
    out = band
    for rule in rules_for(s):
        if verdict != rule.verdict:
            continue
        if _BAND_ORDER.get(rule.floor_band, 0) > _BAND_ORDER.get(out, 0):
            applied.append(AppliedFloor(rule.code, rule.description, out, rule.floor_band))
            out = rule.floor_band
    return out, applied
