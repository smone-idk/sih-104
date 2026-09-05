"""Context engine — transcript + directory metadata -> fusion components (§5).

Produces the three context-derived fusion components:

  transaction_context  <- transaction intent + normalised amount
  behavioural_risk     <- urgency, secrecy, authority claim, out-of-workflow,
                          credential/PII solicitation
  caller_trust         <- enterprise directory (DEMO DATA, labelled as such)

AVAILABILITY RULE (§7). A transcript-derived component that matched nothing
reports `available=False` and its weight redistributes, exactly as the speaker
layer does with no enrolled profile. It never reports 0.0. The consequence is
deliberate and correct for a triage tool: **context can only raise risk, never
lower it.** Absence of evidence is not evidence of safety.

`caller_trust` is the exception, and it is not an exception to the rule: a
directory record is *presence* of evidence about the caller, so it stays
available (and a known, verified contact legitimately reports low risk). With no
directory record attached it is unavailable, like the others.

Every signal carries the transcript span that produced it, and each match is
resolved back to the utterance and timestamp it came from, so the UI can show
the judge exactly why a flag fired.

Badged `HEURISTIC` — rules and regex over real ASR output, not a trained
classifier.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..asr.worker import Transcript
from .signals import SignalResult, extract_all

log = logging.getLogger("voiceshield.context")

#: How conclusive each behavioural signal is ON ITS OWN, at full strength.
#: These are NOT mixing weights — see `_behavioural_risk` for why that
#: distinction matters. Still expert-elicited, but now documented posteriors:
#: revised after measuring six scenarios (LIMITATIONS.md §7), not untouched
#: priors.
BEHAVIOURAL_STRENGTH = {
    # "read me the OTP" is not 25% of a fraud — on its own it is close to
    # conclusive. No legitimate caller asks for a one-time password.
    "credential_solicitation": 0.85,
    # "a warrant will be issued unless you pay" — agencies do not cold-call
    # with arrest threats.
    "threat_coercion": 0.70,
    # "skip the second approval" is a request to disable the control that
    # exists precisely to stop this.
    "out_of_workflow": 0.70,
    # "don't loop in the finance team" — strong, but has benign uses.
    "secrecy": 0.60,
    # urgency and authority claims are common in legitimate calls too, so
    # neither is close to sufficient alone.
    "urgency": 0.40,
    "authority_claim": 0.35,
}


@dataclass
class ContextResult:
    signals: dict[str, SignalResult] = field(default_factory=dict)
    components: dict[str, dict] = field(default_factory=dict)
    directory: dict[str, Any] | None = None
    transcript_chars: int = 0

    def as_dict(self) -> dict:
        return {
            "signals": {k: v.as_dict() for k, v in self.signals.items()},
            "components": self.components,
            "directory": self.directory,
            "transcript_chars": self.transcript_chars,
            "kind": "heuristic",
            "note": "rules + regex over real ASR output; every value traces to a "
                    "transcript span",
        }

    def quotes(self) -> list[dict]:
        """Flat list of every triggering quote, for the UI panel."""
        out = []
        for name, sig in self.signals.items():
            for m in sig.matches:
                d = m.as_dict()
                d["signal"] = name
                out.append(d)
        return sorted(out, key=lambda d: (d["t_start"] is None, d["t_start"] or 0,
                                          d["start"]))


def _locate(sig: SignalResult, transcript: Transcript) -> None:
    """Resolve each match back to the utterance and timestamp it came from."""
    for m in sig.matches:
        seg = transcript.segment_at(m.start)
        if seg is None:
            continue
        m.utterance_index = seg.index
        # linear interpolation within the utterance is enough to point a judge
        # at the right moment; we do not claim word-level alignment.
        span = max(1, seg.char_end - seg.char_start)
        frac0 = (m.start - seg.char_start) / span
        frac1 = (m.end - seg.char_start) / span
        dur = max(0.0, seg.t_end - seg.t_start)
        m.t_start = seg.t_start + dur * max(0.0, min(1.0, frac0))
        m.t_end = seg.t_start + dur * max(0.0, min(1.0, frac1))


def _component(value: float, note: str, detail: dict,
               available: bool = True) -> dict:
    return {"value": value, "available": available, "kind": "heuristic",
            "note": note, "detail": detail}


def analyze_context(transcript: Transcript,
                    directory: dict[str, Any] | None = None) -> ContextResult:
    text = transcript.text
    res = ContextResult(transcript_chars=len(text), directory=directory)
    if not text.strip():
        # No transcript yet -> components stay unavailable and fusion
        # redistributes their weight. Never substitute a neutral value.
        res.components = {
            "transaction_context": _component(
                0.0, "no transcript yet", {}, available=False),
            "behavioural_risk": _component(
                0.0, "no transcript yet", {}, available=False),
        }
        res.components.update(_caller_trust(directory))
        return res

    res.signals = extract_all(text)
    for sig in res.signals.values():
        _locate(sig, transcript)

    # --- transaction_context ---
    # A layer that looked and found nothing is UNAVAILABLE, not 0.0. Reporting
    # 0.0 would hand a fifth of the score budget to "no evidence" and dilute the
    # acoustic layers — which measurably pushed a fraud call's score DOWN (§7).
    # Absence of evidence is not evidence of safety.
    tx = res.signals["transaction_intent"]
    res.components["transaction_context"] = _component(
        tx.value,
        f"{len(tx.matches)} transcript match(es)",
        {"rules_fired": tx.detail.get("rules_fired", []),
         "amount_inr": tx.detail.get("amount_inr"),
         "amount_text": tx.detail.get("amount_text"),
         "quotes": [m.as_dict() for m in tx.matches]},
    ) if tx.matches else _component(
        0.0, "no transaction discussed — layer not applicable", {},
        available=False)

    # --- behavioural_risk ---
    total, parts, fired = _behavioural_risk(res.signals)
    res.components["behavioural_risk"] = _component(
        total,
        f"{len(fired)} behavioural signal(s) fired",
        {"parts": parts, "fired": fired, "combination": "noisy-OR"},
    ) if fired else _component(
        0.0, "no behavioural signals in the transcript — layer not applicable",
        {"parts": parts, "combination": "noisy-OR"}, available=False)

    res.components.update(_caller_trust(directory))
    return res


def _behavioural_risk(signals: dict[str, SignalResult]):
    """Combine behavioural signals as independent evidence (noisy-OR), not as a
    weighted mean.

    WHY THIS CHANGED (decision recorded in LIMITATIONS.md §7). A weighted mean
    treats each signal as a fractional contribution to one latent quantity, so a
    call has to fire nearly everything to score high. Measured consequence: the
    bank-OTP scenario, where the caller asks for a one-time password outright,
    produced behavioural_risk 0.43 — because credential_solicitation carried
    only 0.25 of the mix. But asking for an OTP is not 25% of a fraud; it is on
    its own close to conclusive.

    These signals are better modelled as independent evidence, any one of which
    can be sufficient. Noisy-OR does that:

        risk = 1 - Π (1 - strength_i * value_i)

    It is monotone (evidence never lowers risk), saturates at 1, and reduces to
    `strength_i * value_i` when only one signal fires. On the demo corpus it
    lifts the OTP case 0.43 -> 0.93 and the govt-summons case 0.43 -> 0.91,
    while leaving the benign control at exactly 0.0.
    """
    parts: dict[str, dict] = {}
    fired: list[str] = []
    product = 1.0
    for name, strength in BEHAVIOURAL_STRENGTH.items():
        sig = signals.get(name)
        v = sig.value if sig else 0.0
        parts[name] = {
            "value": round(v, 4),
            "strength": strength,
            "evidence": round(strength * v, 4),
            "quotes": [m.as_dict() for m in (sig.matches if sig else [])],
        }
        product *= (1.0 - strength * v)
        if v > 0:
            fired.append(name)
    return min(1.0, 1.0 - product), parts, fired


def _caller_trust(directory: dict[str, Any] | None) -> dict[str, dict]:
    """Directory metadata -> caller_trust risk. DEMO DATA (§2 Tier C).

    Unavailable when no directory record is attached, rather than assuming a
    trusted or untrusted caller.
    """
    if not directory:
        return {"caller_trust": _component(
            0.0, "no directory record attached — layer unavailable", {},
            available=False)}
    trust = float(directory.get("trust_score", 0.0) or 0.0)
    known = bool(directory.get("known_contact", 0))
    verified = bool(directory.get("verified_identity", 0))
    prior = int(directory.get("prior_interactions", 0) or 0)
    risk = 1.0 - trust
    reasons = []
    if not known:
        reasons.append("caller not in the enterprise directory")
    if not verified:
        reasons.append("identity not verified")
    if prior == 0:
        reasons.append("no prior interactions")
    return {"caller_trust": _component(
        min(1.0, max(0.0, risk)),
        "; ".join(reasons) or "known, verified contact",
        {"display_name": directory.get("display_name"),
         "trust_score": trust, "known_contact": known,
         "verified_identity": verified, "prior_interactions": prior,
         "source": "enterprise directory (demo data)"},
    )}
