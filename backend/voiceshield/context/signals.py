"""Context signal extraction from the ASR transcript (§5).

This is the differentiator, so it is derived from what was actually said — not
from presenter-operated toggles. Every signal returns a 0-1 value AND the
matched spans, so the UI can show the judge the quote that triggered it.

Design rules:
  * A signal with no matched span is not allowed to have a non-zero value.
    `SignalResult.__post_init__` enforces that — it is the Phase 3 gate.
  * Patterns are per-language lists including Hinglish code-switching (§10).
    English is tested; Hindi/Punjabi lists exist but there is no Hindi/Punjabi
    audio in the corpus to test them against (see LIMITATIONS.md).
  * This is `HEURISTIC` — rules and regex, not a trained classifier. It is
    badged as such in the UI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- amounts
_MULT = {
    "k": 1e3, "thousand": 1e3, "hazaar": 1e3, "hazar": 1e3,
    "l": 1e5, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "lakhon": 1e5,
    "cr": 1e7, "crore": 1e7, "crores": 1e7, "karod": 1e7,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
}
_UNIT_RE = "|".join(sorted(_MULT, key=len, reverse=True))

# ₹25,00,000 / rs 25 lakh / 25L / 2.5 million / twenty five lakh
_AMOUNT_PATTERNS = [
    rf"(?:₹|rs\.?|inr|rupees?)\s*([\d][\d,]*(?:\.\d+)?)\s*({_UNIT_RE})?\b",
    rf"\b([\d][\d,]*(?:\.\d+)?)\s*({_UNIT_RE})\b",
    r"(?:₹|rs\.?|inr|rupees?)\s*([\d][\d,]*(?:\.\d+)?)",
]

_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}

_ONES_RE = "|".join(sorted(_ONES, key=len, reverse=True))
_TENS_RE = "|".join(sorted(_TENS, key=len, reverse=True))

# "twenty-five lakh", "twenty five lakh", "fifty thousand", "two crore".
# The hyphen matters: without it "twenty-five lakh" parses as "five lakh"
# and understates the amount by 5x.
_WORD_AMOUNT_RE = re.compile(
    rf"\b((?:{_TENS_RE})(?:[-\s]+(?:{_ONES_RE}))?|(?:{_ONES_RE})|hundred)"
    rf"\s+({_UNIT_RE})\b",
    re.I)


def _word_to_num(phrase: str) -> float | None:
    total = 0.0
    seen = False
    for tok in re.split(r"[-\s]+", phrase.strip().lower()):
        if tok in _TENS:
            total += _TENS[tok]
            seen = True
        elif tok in _ONES:
            total += _ONES[tok]
            seen = True
        elif tok == "hundred":
            total = (total or 1) * 100
            seen = True
    return total if seen else None


def parse_amount(text: str) -> tuple[float | None, tuple[int, int] | None]:
    """Return (normalised rupees, (start, end)) for the LARGEST amount found."""
    best: tuple[float, tuple[int, int]] | None = None
    for pat in _AMOUNT_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
            except ValueError:
                continue
            unit = (m.group(2) or "").lower() if m.lastindex and m.lastindex >= 2 else ""
            if unit:
                val *= _MULT.get(unit, 1.0)
            if best is None or val > best[0]:
                best = (val, (m.start(), m.end()))
    for m in _WORD_AMOUNT_RE.finditer(text):
        n = _word_to_num(m.group(1))
        if n is None:
            continue
        val = n * _MULT.get(m.group(2).lower(), 1.0)
        if best is None or val > best[0]:
            best = (val, (m.start(), m.end()))
    return (best[0], best[1]) if best else (None, None)


# ------------------------------------------------------------- data types
@dataclass
class SignalMatch:
    quote: str                  # the exact text that triggered the signal
    start: int                  # char offset in the rolling transcript
    end: int
    rule: str                   # which pattern fired — explainability
    t_start: float | None = None
    t_end: float | None = None
    utterance_index: int | None = None

    def as_dict(self) -> dict:
        return {
            "quote": self.quote,
            "start": self.start,
            "end": self.end,
            "rule": self.rule,
            "t_start": None if self.t_start is None else round(self.t_start, 2),
            "t_end": None if self.t_end is None else round(self.t_end, 2),
            "utterance_index": self.utterance_index,
        }


@dataclass
class SignalResult:
    name: str
    value: float
    matches: list[SignalMatch] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    kind: str = "heuristic"

    def __post_init__(self) -> None:
        # THE Phase 3 gate, enforced in code: no quote, no signal.
        if self.value > 0 and not self.matches:
            raise ValueError(
                f"context signal {self.name!r} has value {self.value} but no "
                "matched transcript span — every flag must trace to a quote")
        self.value = max(0.0, min(1.0, float(self.value)))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "kind": self.kind,
            "matches": [m.as_dict() for m in self.matches],
            "detail": self.detail,
        }


# ------------------------------------------------------------- lexicons
# Each entry: (rule name, regex). Kept explicit rather than a bag of words so
# the UI can name WHICH rule fired, not just that something did.
URGENCY = [
    ("immediacy", r"\b(right now|immediately|at once|this instant|straight away|urgent(?:ly)?|asap|abhi|turant)\b"),
    ("deadline", r"\b(before (?:the )?end of (?:the )?day|by (?:close of business|cob|eod|noon|\d{1,2}\s*(?:am|pm))|within (?:the next )?\d+\s*(?:minutes?|hours?)|within the hour|in the next \d+)\b"),
    ("no_time", r"\b(no time|running out of time|can'?t wait|cannot wait|there'?s no time|last chance|final (?:notice|warning|reminder))\b"),
    ("pressure", r"\b(hurry|quickly|quick|fast|jaldi|do it now|need this done now|i'?ll be quick)\b"),
]

SECRECY = [
    ("dont_tell", r"\b(don'?t tell|do not tell|not a word to|kisi ko mat bata)\b"),
    ("between_us", r"\b(between (?:you and me|us)|keep (?:this|it) (?:quiet|between|to yourself)|off the record)\b"),
    ("dont_loop_in", r"\b(don'?t (?:loop|bring|copy|cc|involve|inform)\b[^.]{0,40}|do not (?:loop|involve|inform)\b[^.]{0,40})"),
    ("confidential", r"\b(confidential for now|strictly confidential|keep this confidential|confidential (?:investigation|matter|case)|hush|discreet(?:ly)?)\b"),
    ("must_not_discuss", r"\b((?:you )?must not discuss|cannot discuss this|do not discuss this)\b"),
]

AUTHORITY = [
    # allow a name between the self-identification and the role:
    # "this is Rajesh Sharma, your CFO"
    ("exec_claim", r"\b(?:this is|i am|i'?m|it'?s|speaking[, ]+)\s*(?:[a-z]+,?\s+){0,3}(?:the\s+|your\s+)?(ceo|cfo|coo|cto|managing director|chairman|director|manager|head of [a-z ]{3,20}|boss|senior partner)\b"),
    ("institution_claim", r"\b(?:calling |speaking )?from (?:the )?(rbi|reserve bank|income tax|it department|cbi|police|cyber cell|customs|trai|sebi|enforcement directorate|court|bank'?s? (?:fraud|security) (?:desk|department|team))\b"),
    ("bank_official", r"\b(?:this is|i am|i'?m)\s*[a-z ]{0,20}from\s+(?:your\s+)?(bank|hdfc|icici|sbi|axis|kotak)\b"),
    ("law_enforcement", r"\b(inspector|sub-?inspector|officer|constable|magistrate|advocate)\s+[a-z]{3,}\b"),
]

TRANSACTION = [
    ("transfer_verb", r"\b(transfer|wire|remit|send|release|process|make|move|push|pay)\s+(?:the\s+|a\s+|this\s+|out\s+)?(?:payment|amount|funds?|money|sum|₹|rs\.?|\d)"),
    ("payment_noun", r"\b(fund transfer|wire transfer|payment|remittance|beneficiary|neft|rtgs|imps|upi)\b"),
    ("account_change", r"\b(new (?:account|beneficiary|vendor)|different account|updated (?:bank )?details|change (?:the )?account)\b"),
]

OUT_OF_WORKFLOW = [
    ("skip_approval", r"\b(skip (?:the )?(?:second |dual |maker.?checker )?approvals?|bypass (?:the )?(?:approval|process|protocol|checks?)|without (?:the )?(?:usual |normal |standard )?approvals?|no need for (?:the )?approvals?)\b"),
    ("bypass_control", r"\b(don'?t (?:follow|use) the (?:usual|normal|standard) (?:process|procedure|channel)|outside (?:the )?(?:usual |normal )?(?:process|channel)|make an exception|override)\b"),
    ("avoid_verification", r"\b(no need to verify|skip (?:the )?verification|don'?t (?:need to )?(?:call|check|confirm) back|no callback)\b"),
    ("different_account", r"\b(use a different account|pay(?: it)? to (?:a|this) (?:new|different) account)\b"),
]

CREDENTIAL = [
    ("otp", r"\b(otp|one[- ]time (?:password|pin|code)|verification code|security code|read (?:me|out) the code)\b"),
    ("password", r"\b(password|passcode|pin(?: number)?|credentials|login details)\b"),
    ("account_number", r"\b(account number|card number|cvv|expiry date|ifsc|debit card|credit card)\b"),
    ("national_id", r"\b(aadhaar|aadhar|pan (?:card|number)|passport number|voter id)\b"),
]

LEXICONS = {
    "urgency": URGENCY,
    "secrecy": SECRECY,
    "authority_claim": AUTHORITY,
    "transaction_intent": TRANSACTION,
    "out_of_workflow": OUT_OF_WORKFLOW,
    "credential_solicitation": CREDENTIAL,
}

#: how many distinct rules must fire for a signal to reach 1.0
_SATURATE_AT = {
    "urgency": 3, "secrecy": 2, "authority_claim": 2,
    "transaction_intent": 3, "out_of_workflow": 2, "credential_solicitation": 2,
}


#: A negator immediately before a match flips its meaning: "nothing urgent" is
#: the OPPOSITE of urgency. Without this the genuine control clip fires an
#: urgency flag on "Nothing urgent, give me a call back whenever" — the worst
#: possible false positive for this demo.
_NEGATOR_RE = re.compile(r"\b(no|not|nothing|never|n'?t|hardly|without)\s+(?:\w+\s+){0,1}$", re.I)


def _negated(text: str, start: int) -> bool:
    return bool(_NEGATOR_RE.search(text[max(0, start - 25):start]))


def _find(text: str, rules: list[tuple[str, str]]) -> list[SignalMatch]:
    out: list[SignalMatch] = []
    seen: set[tuple[int, int]] = set()
    for rule, pat in rules:
        for m in re.finditer(pat, text, re.I):
            span = (m.start(), m.end())
            if span in seen:
                continue
            seen.add(span)
            if _negated(text, m.start()):
                continue
            out.append(SignalMatch(quote=text[m.start():m.end()].strip(),
                                   start=m.start(), end=m.end(), rule=rule))
    return sorted(out, key=lambda m: m.start)


def extract_signal(name: str, text: str) -> SignalResult:
    """Generic lexicon signal: value scales with the number of DISTINCT rules
    that fired, saturating at a per-signal count. Repeating one phrase five
    times does not make it five times more suspicious."""
    rules = LEXICONS[name]
    matches = _find(text, rules)
    distinct = {m.rule for m in matches}
    sat = _SATURATE_AT.get(name, 2)
    value = min(1.0, len(distinct) / sat) if matches else 0.0
    return SignalResult(name=name, value=value, matches=matches,
                        detail={"rules_fired": sorted(distinct),
                                "n_matches": len(matches),
                                "saturates_at_rules": sat})


def extract_transaction(text: str) -> SignalResult:
    """Transaction intent + amount. Amount magnitude scales the signal, but an
    amount alone (with no transfer intent) is not a transaction request."""
    base = extract_signal("transaction_intent", text)
    amount, span = parse_amount(text)
    matches = list(base.matches)
    detail = dict(base.detail)

    if amount is not None and span is not None:
        matches.append(SignalMatch(quote=text[span[0]:span[1]].strip(),
                                   start=span[0], end=span[1], rule="amount"))
        detail["amount_inr"] = amount
        detail["amount_text"] = text[span[0]:span[1]].strip()

    value = base.value
    if amount is not None:
        # log-scaled: ₹10k ~ 0.2, ₹1L ~ 0.5, ₹25L ~ 0.75, ₹1cr+ ~ 1.0
        import math
        mag = min(1.0, max(0.0, (math.log10(max(amount, 1.0)) - 3.0) / 4.0))
        detail["amount_magnitude"] = round(mag, 3)
        value = max(value, mag) if base.matches else mag * 0.6
        value = min(1.0, 0.5 * base.value + 0.5 * mag + 0.15 * bool(base.matches))

    return SignalResult(name="transaction_intent", value=min(1.0, value),
                        matches=matches, detail=detail)


def extract_all(text: str) -> dict[str, SignalResult]:
    if not text.strip():
        return {}
    out = {n: extract_signal(n, text) for n in LEXICONS if n != "transaction_intent"}
    out["transaction_intent"] = extract_transaction(text)
    return out
