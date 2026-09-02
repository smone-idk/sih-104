"""Findings — what actually fired, as distinct from how much it moved the score.

"This is not who they claim to be" and "this voice is machine-generated" are
different findings with different responses, even though both raise the same
0-100 number. The UI needs to show which one fired, so the analysis result
carries them explicitly.

The interesting part is the 2x2 between the two independent layers:

                     speaker similarity HIGH        speaker similarity LOW
  synthetic HIGH     CLONED_VOICE                   SYNTHETIC_OTHER
                     (sounds like the enrolled      (machine voice, and not
                      person AND machine-made —      the enrolled person)
                      the headline attack)
  synthetic LOW      CONSISTENT                     SPEAKER_MISMATCH
                     (consistent with the           (a different human — wrong
                      enrolled speaker)              person, or an impersonator)

That separation is the argument for keeping the layers independent, and it is
only derivable because they are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# thresholds for calling a layer "high" — deliberately conservative, and
# exposed here rather than buried, because they shape what the UI announces.
SYNTHETIC_HIGH = 0.65
SYNTHETIC_LOW = 0.35
SPEAKER_MISMATCH_HIGH = 0.60   # speaker_consistency score (higher = less similar)
SPEAKER_MATCH_LOW = 0.40


@dataclass
class Finding:
    code: str
    label: str
    severity: str                     # "info" | "warning" | "critical"
    value: float | None
    kind: str                         # detector kind badge (pretrained/heuristic/...)
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "label": self.label,
            "severity": self.severity,
            "value": None if self.value is None else round(self.value, 4),
            "kind": self.kind,
            "evidence": self.evidence,
        }


def derive_findings(components: dict[str, Any]) -> tuple[list[Finding], str]:
    """`components` maps component name -> ComponentInput-like object with
    .value / .available / .kind / .detail. Returns (findings, voice_verdict)."""
    syn = components.get("voice_authenticity")
    spk = components.get("speaker_consistency")

    syn_v = syn.value if (syn and syn.available and syn.value is not None) else None
    spk_v = spk.value if (spk and spk.available and spk.value is not None) else None
    syn_kind = getattr(syn, "kind", "") if syn else ""
    spk_kind = getattr(spk, "kind", "") if spk else ""
    cos = (getattr(spk, "detail", {}) or {}).get("cosine_similarity")

    findings: list[Finding] = []

    # --- independent layer findings ---
    if syn_v is not None and syn_v >= SYNTHETIC_HIGH:
        findings.append(Finding(
            "synthetic_speech",
            "Acoustic markers consistent with machine-generated speech",
            "critical", syn_v, syn_kind,
            {"synthetic_probability": round(syn_v, 4),
             "threshold": SYNTHETIC_HIGH,
             **(getattr(syn, "detail", {}) or {})}))

    if spk_v is not None and spk_v >= SPEAKER_MISMATCH_HIGH:
        findings.append(Finding(
            "speaker_mismatch",
            "Voice does not match the enrolled speaker profile",
            "critical", spk_v, spk_kind,
            {"cosine_similarity": cos, "threshold": SPEAKER_MISMATCH_HIGH}))
    elif spk_v is not None and spk_v <= SPEAKER_MATCH_LOW:
        findings.append(Finding(
            "speaker_match",
            "Voice is consistent with the enrolled speaker profile",
            "info", spk_v, spk_kind,
            {"cosine_similarity": cos}))

    # --- the 2x2 verdict ---
    verdict = "INDETERMINATE"
    if syn_v is None and spk_v is None:
        verdict = "INDETERMINATE"
    elif syn_v is not None and spk_v is None:
        verdict = "SYNTHETIC_SUSPECTED" if syn_v >= SYNTHETIC_HIGH else "NO_SYNTHETIC_MARKERS"
    elif syn_v is None and spk_v is not None:
        verdict = "SPEAKER_MISMATCH" if spk_v >= SPEAKER_MISMATCH_HIGH else "SPEAKER_CONSISTENT"
    else:
        syn_high = syn_v >= SYNTHETIC_HIGH
        spk_match = spk_v <= SPEAKER_MATCH_LOW
        spk_mismatch = spk_v >= SPEAKER_MISMATCH_HIGH
        if syn_high and spk_match:
            verdict = "CLONED_VOICE"
            findings.append(Finding(
                "cloned_voice",
                "Machine-generated speech that matches the enrolled speaker — "
                "consistent with a voice clone of this person",
                "critical", max(syn_v, 1 - spk_v), syn_kind,
                {"synthetic_probability": round(syn_v, 4),
                 "cosine_similarity": cos,
                 "why": "both layers fired in the same direction; this is the "
                        "case a single combined score would hide"}))
        elif syn_high and spk_mismatch:
            verdict = "SYNTHETIC_OTHER"
        elif not syn_high and spk_mismatch:
            verdict = "SPEAKER_MISMATCH"
        elif not syn_high and spk_match:
            verdict = "CONSISTENT"

    if not findings:
        findings.append(Finding(
            "no_findings", "No individual layer crossed its reporting threshold",
            "info", None, "", {}))
    return findings, verdict
