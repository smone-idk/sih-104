"""Context engine tests (§5, Phase 3).

THE gate: every context flag traces to a transcript quote. That is enforced in
code (`SignalResult.__post_init__` raises), and asserted end-to-end here.
"""
from __future__ import annotations

import pathlib

import pytest

from voiceshield.asr.segmenter import Utterance, utterances_from_segments
from voiceshield.asr.worker import Transcript, TranscriptSegment
from voiceshield.config import get_settings
from voiceshield.context.engine import analyze_context
from voiceshield.context.signals import (
    SignalMatch,
    SignalResult,
    extract_all,
    parse_amount,
)
from voiceshield.ingest.vad import SpeechSegment

CEO = ("Hi, this is Rajesh Sharma, your CFO. I'm between meetings so I'll be quick. "
       "We are closing an acquisition today and I need you to move twenty-five lakh "
       "rupees to a new vendor account before end of day. Please don't loop in the "
       "wider finance team yet, keep this between us. Skip the second approval this "
       "once. I'll message you the account number now.")

BENIGN = ("Hi, it's me. Just calling to check about the team lunch on Friday. "
          "No rush on those, sometime next week is fine. Nothing urgent, give me "
          "a call back whenever you get a chance.")


def _transcript(text: str, t0: float = 0.0, dur: float = 10.0) -> Transcript:
    t = Transcript()
    t.add([TranscriptSegment(index=0, t_start=t0, t_end=t0 + dur, text=text)])
    return t


# --- THE GATE ----------------------------------------------------------
def test_every_nonzero_signal_has_a_quote():
    for text in (CEO, BENIGN, "transfer 5 crore immediately, don't tell anyone"):
        for name, sig in extract_all(text).items():
            if sig.value > 0:
                assert sig.matches, f"{name} has value but no quote"
                for m in sig.matches:
                    assert m.quote.strip(), f"{name} produced an empty quote"
                    assert text[m.start:m.end].strip().lower() == m.quote.lower()


def test_signal_without_quote_is_rejected_at_construction():
    """The gate is enforced in code, not just by convention."""
    with pytest.raises(ValueError, match="must trace to a quote"):
        SignalResult(name="urgency", value=0.8, matches=[])
    # zero-valued signals with no match are fine
    SignalResult(name="urgency", value=0.0, matches=[])


def test_quotes_resolve_to_utterance_and_timestamp():
    t = Transcript()
    t.add([
        TranscriptSegment(0, 0.0, 5.0, "Hi, this is Rajesh Sharma, your CFO."),
        TranscriptSegment(1, 5.0, 12.0,
                          "Move twenty-five lakh rupees before end of day."),
    ])
    res = analyze_context(t, None)
    quotes = res.quotes()
    assert quotes
    for q in quotes:
        assert q["utterance_index"] in (0, 1)
        assert q["t_start"] is not None
        seg = t.segments[q["utterance_index"]]
        assert seg.t_start - 0.01 <= q["t_start"] <= seg.t_end + 0.01


# --- amount parsing ----------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("transfer ₹25,00,000 today", 2_500_000),
    ("move 25 lakh rupees", 2_500_000),
    ("send 25L now", 2_500_000),
    ("pay 2.5 million", 2_500_000),
    ("two lakh rupees", 200_000),
    ("rs 9,000 debited", 9_000),
    ("1.5 crore", 15_000_000),
    ("twenty five lakh", 2_500_000),
])
def test_indian_amount_formats(text, expected):
    amount, span = parse_amount(text)
    assert amount == pytest.approx(expected, rel=0.01), text
    assert span is not None


def test_no_amount_returns_none():
    assert parse_amount("just calling to say hello")[0] is None


def test_largest_amount_wins():
    amount, _ = parse_amount("a fee of 500 rupees on a 25 lakh transfer")
    assert amount == pytest.approx(2_500_000)


# --- negation ----------------------------------------------------------
def test_benign_call_fires_no_signals():
    """The genuine control must not trip urgency on 'Nothing urgent'."""
    sigs = extract_all(BENIGN)
    fired = {n: s.value for n, s in sigs.items() if s.value > 0}
    assert not fired, f"false positives on a benign call: {fired}"


def test_negation_suppresses_a_match():
    assert extract_all("this is urgent")["urgency"].value > 0
    assert extract_all("nothing urgent here")["urgency"].value == 0


# --- signal semantics --------------------------------------------------
def test_scam_script_fires_the_expected_signals():
    sigs = extract_all(CEO)
    for name in ("urgency", "secrecy", "authority_claim",
                 "transaction_intent", "out_of_workflow"):
        assert sigs[name].value > 0, f"{name} should fire on the CEO script"
    assert sigs["transaction_intent"].detail["amount_inr"] == pytest.approx(2_500_000)


def test_repeating_one_phrase_does_not_saturate():
    once = extract_all("do it immediately")["urgency"].value
    many = extract_all("immediately immediately immediately immediately")["urgency"].value
    assert many == once, "value should scale with distinct rules, not repetitions"


# --- components --------------------------------------------------------
def test_components_unavailable_without_transcript():
    res = analyze_context(Transcript(), None)
    for name in ("transaction_context", "behavioural_risk", "caller_trust"):
        assert res.components[name]["available"] is False
        assert res.components[name]["value"] == 0.0


def test_caller_trust_from_directory():
    res = analyze_context(_transcript(CEO), {
        "display_name": "Unknown Caller", "known_contact": 0,
        "verified_identity": 0, "prior_interactions": 0, "trust_score": 0.05})
    ct = res.components["caller_trust"]
    assert ct["available"] and ct["value"] == pytest.approx(0.95)
    assert "not in the enterprise directory" in ct["note"]

    res2 = analyze_context(_transcript(CEO), {
        "display_name": "Priya Nair", "known_contact": 1,
        "verified_identity": 1, "prior_interactions": 88, "trust_score": 0.85})
    assert res2.components["caller_trust"]["value"] == pytest.approx(0.15)


def test_behavioural_risk_carries_its_parts():
    res = analyze_context(_transcript(CEO), None)
    br = res.components["behavioural_risk"]
    assert br["available"] and br["value"] > 0
    parts = br["detail"]["parts"]
    assert set(parts) == {"credential_solicitation", "out_of_workflow", "secrecy",
                          "urgency", "authority_claim"}
    for p in parts.values():
        if p["value"] > 0:
            assert p["quotes"], "a contributing part must carry its quotes"


# --- segmentation (the Phase 3 decision) -------------------------------
def test_utterances_merge_short_vad_segments():
    s = get_settings()
    segs = [SpeechSegment(0.0, 1.0, 1.0), SpeechSegment(1.4, 2.6, 1.2),
            SpeechSegment(3.0, 4.8, 1.8)]
    utts = utterances_from_segments(segs, s)
    assert utts
    assert all(u.duration <= s.asr_max_segment_seconds * 1.5 for u in utts)
    # merged rather than one Whisper call per tiny fragment
    assert len(utts) < len(segs)


def test_long_segment_is_split_under_the_max():
    s = get_settings()
    utts = utterances_from_segments([SpeechSegment(0.0, 30.0, 30.0)], s)
    assert len(utts) > 1
    assert all(u.duration <= s.asr_max_segment_seconds * 1.5 for u in utts)


def test_no_segments_gives_no_utterances():
    assert utterances_from_segments([], get_settings()) == []
