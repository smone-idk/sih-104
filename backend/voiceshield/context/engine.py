"""Context engine — transcript + directory metadata -> fusion components (§5).

Produces the three context-derived fusion components:

  transaction_context  <- transaction intent + normalised amount
  behavioural_risk     <- urgency, secrecy, authority claim, out-of-workflow,
                          credential/PII solicitation
  caller_trust         <- enterprise directory (DEMO DATA, labelled as such)

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

#: behavioural_risk is a weighted mix of the behavioural signals. Weights are
#: expert-elicited priors, like the fusion weights — not fitted.
BEHAVIOURAL_WEIGHTS = {
    "credential_solicitation": 0.30,
    "out_of_workflow": 0.25,
    "secrecy": 0.20,
    "urgency": 0.15,
    "authority_claim": 0.10,
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
    tx = res.signals["transaction_intent"]
    res.components["transaction_context"] = _component(
        tx.value,
        f"{len(tx.matches)} transcript match(es)" if tx.matches
        else "no transaction language found",
        {"rules_fired": tx.detail.get("rules_fired", []),
         "amount_inr": tx.detail.get("amount_inr"),
         "amount_text": tx.detail.get("amount_text"),
         "quotes": [m.as_dict() for m in tx.matches]},
    )

    # --- behavioural_risk ---
    parts, fired = {}, []
    total = 0.0
    for name, w in BEHAVIOURAL_WEIGHTS.items():
        sig = res.signals.get(name)
        v = sig.value if sig else 0.0
        parts[name] = {"value": round(v, 4), "weight": w,
                       "contribution": round(v * w, 4),
                       "quotes": [m.as_dict() for m in (sig.matches if sig else [])]}
        total += v * w
        if v > 0:
            fired.append(name)
    res.components["behavioural_risk"] = _component(
        min(1.0, total),
        f"{len(fired)} behavioural signal(s) fired" if fired
        else "no behavioural signals found",
        {"parts": parts, "fired": fired},
    )

    res.components.update(_caller_trust(directory))
    return res


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
