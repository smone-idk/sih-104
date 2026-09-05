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
    from voiceshield.context.engine import BEHAVIOURAL_STRENGTH

    res = analyze_context(_transcript(CEO), None)
    br = res.components["behavioural_risk"]
    assert br["available"] and br["value"] > 0
    parts = br["detail"]["parts"]
    assert set(parts) == set(BEHAVIOURAL_STRENGTH)
    for p in parts.values():
        if p["value"] > 0:
            assert p["quotes"], "a contributing part must carry its quotes"


# --- availability rule (§7): absence of evidence is not evidence of safety ---
def test_empty_context_layers_are_unavailable_not_zero():
    """A transcript with no fraud language must leave the context layers
    UNAVAILABLE so their weight redistributes — reporting 0.0 would hand a
    fifth of the score budget to 'no evidence' and dilute the acoustic layers."""
    res = analyze_context(_transcript(BENIGN), None)
    for name in ("transaction_context", "behavioural_risk"):
        c = res.components[name]
        assert c["available"] is False, name
        assert "not applicable" in c["note"], c["note"]


def test_context_can_only_raise_risk_never_lower_it():
    """The consequence of the availability rule, asserted directly: adding a
    benign transcript must not reduce a score computed without one."""
    from voiceshield.fusion.scorer import ComponentInput, fuse

    acoustic = {
        "voice_authenticity": ComponentInput(1.0),
        "speaker_consistency": ComponentInput(0.05),
        "prosody_anomaly": ComponentInput(0.4),
    }
    without = fuse(dict(acoustic)).score

    benign = analyze_context(_transcript(BENIGN), None)
    with_ctx = dict(acoustic)
    for name, c in benign.components.items():
        with_ctx[name] = ComponentInput(value=c["value"], available=c["available"],
                                        note=c["note"])
    assert fuse(with_ctx).score == pytest.approx(without, abs=1e-6)


def test_populated_context_layers_are_available():
    res = analyze_context(_transcript(CEO), None)
    assert res.components["transaction_context"]["available"] is True
    assert res.components["behavioural_risk"]["available"] is True


# --- threat/coercion family ------------------------------------------
@pytest.mark.parametrize("text,rule", [
    ("a case has been registered against you", "legal_process"),
    ("a warrant will be issued", "legal_process"),
    ("this is a non-bailable offence", "legal_process"),
    ("you will be arrested today", "arrest_threat"),
    ("a penalty of two lakh rupees", "penalty_threat"),
    ("your account will be frozen", "penalty_threat"),
    ("this is from the income tax department", "agency_process"),
    ("stay on the line, do not hang up", "stay_on_line"),
])
def test_threat_coercion_fires_with_a_quote(text, rule):
    sig = extract_all(text)["threat_coercion"]
    assert sig.value > 0, text
    assert rule in {m.rule for m in sig.matches}
    assert sig.matches[0].quote


@pytest.mark.parametrize("text", [
    "that's fine, talk soon",
    "I'm fine thanks",
    "the sales department will call you",
    "we booked the tennis court",
    "read the fine print",
    "I'll send you the notice board photo",
])
def test_threat_coercion_does_not_fire_on_benign_speech(text):
    """Bare tokens (fine, court, department, notice, police) are deliberately
    NOT matched — only their fraud-bearing forms are."""
    assert extract_all(text)["threat_coercion"].value == 0, text


def test_threat_coercion_respects_negation():
    assert extract_all("you will be arrested")["threat_coercion"].value > 0
    assert extract_all("no arrest is being threatened")["threat_coercion"].value == 0


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


# --- ASR confidence gate (§5) ------------------------------------------
def test_low_confidence_segment_never_reaches_the_context_engine():
    """Whisper invents words on cloned audio; an invented word matching a rule
    would quote speech nobody made. A gated segment is excluded from the text
    the context engine reads."""
    s = get_settings()
    t = Transcript()
    t.add([
        TranscriptSegment(0, 0.0, 4.0, "Nothing urgent, call me back whenever.",
                          avg_logprob=-0.2),
        TranscriptSegment(1, 4.0, 6.0, "transfer 25 lakh right now immediately",
                          avg_logprob=s.asr_min_avg_logprob - 0.5),
    ])
    assert "25 lakh" not in t.text
    assert len(t.discarded) == 1
    for name, sig in extract_all(t.text).items():
        assert sig.value == 0, f"{name} fired on gated text"


def test_discarded_segments_stay_visible_with_a_reason():
    """Silently dropping them would hide why a signal is missing."""
    s = get_settings()
    t = Transcript()
    t.add([TranscriptSegment(0, 0.0, 2.0, "mumble",
                             avg_logprob=s.asr_min_avg_logprob - 1.0)])
    d = t.as_dict()
    assert d["n_segments"] == 1 and d["n_discarded"] == 1
    seg = d["segments"][0]
    assert seg["confident"] is False
    assert "low ASR confidence" in seg["gate_reason"]
    assert d["gate"]["enabled"] is True


def test_high_no_speech_prob_is_gated():
    s = get_settings()
    t = Transcript()
    t.add([TranscriptSegment(0, 0.0, 2.0, "transfer money now",
                             avg_logprob=-0.1,
                             no_speech_prob=s.asr_max_no_speech_prob + 0.1)])
    assert t.text == ""
    assert "non-speech" in t.discarded[0].gate_reason


def test_gate_can_be_disabled_and_thresholds_are_config():
    s = get_settings().model_copy(update={"asr_confidence_gate": False})
    t = Transcript()
    t.add([TranscriptSegment(0, 0.0, 2.0, "transfer 25 lakh", avg_logprob=-9.0)],
          settings=s)
    assert "25 lakh" in t.text and not t.discarded


def test_gated_segment_cannot_be_cited_as_a_quote():
    """segment_at() must never resolve into a discarded segment."""
    s = get_settings()
    t = Transcript()
    t.add([
        TranscriptSegment(0, 0.0, 3.0, "please transfer 25 lakh rupees today",
                          avg_logprob=-0.1),
        TranscriptSegment(1, 3.0, 5.0, "garbled", avg_logprob=-9.0),
    ])
    res = analyze_context(t, None)
    for q in res.quotes():
        assert q["utterance_index"] != 1
    assert all(sg.confident for sg in t.segments if sg.char_end > sg.char_start)


def test_transfer_verb_accepts_spelled_out_numbers():
    """'move 25 lakh' and 'move twenty-five lakh' are the same sentence; the
    signal must not depend on how the ASR chose to write the number."""
    for text in ("move 25 lakh rupees", "move twenty-five lakh rupees"):
        rules = {m.rule for m in extract_all(text)["transaction_intent"].matches}
        assert "transfer_verb" in rules, text


# --- behavioural_risk combination (decision recorded in LIMITATIONS §7) ---
def test_one_conclusive_signal_produces_high_behavioural_risk():
    """A weighted mean gave 0.43 when the caller asks for an OTP outright.
    Asking for a one-time password is not 25% of a fraud."""
    res = analyze_context(_transcript(
        "Please read me the one time password that was just sent to you."), None)
    br = res.components["behavioural_risk"]
    assert br["available"] and br["value"] > 0.8, br["value"]
    assert br["detail"]["combination"] == "noisy-OR"


def test_behavioural_risk_is_monotone_in_evidence():
    """Adding a signal can never lower the risk — the point of noisy-OR."""
    from voiceshield.context.engine import _behavioural_risk
    from voiceshield.context.signals import SignalMatch, SignalResult

    def sig(name, v):
        return SignalResult(name=name, value=v,
                            matches=[SignalMatch("q", 0, 1, "r")] if v else [])

    one, _, _ = _behavioural_risk({"secrecy": sig("secrecy", 0.5)})
    two, _, _ = _behavioural_risk({"secrecy": sig("secrecy", 0.5),
                                   "urgency": sig("urgency", 0.5)})
    assert two >= one


def test_behavioural_risk_stays_zero_on_a_benign_call():
    """The change must not manufacture risk where there is no evidence."""
    res = analyze_context(_transcript(BENIGN), None)
    assert res.components["behavioural_risk"]["available"] is False
    assert res.components["behavioural_risk"]["value"] == 0.0


def test_behavioural_parts_carry_strength_and_quotes():
    res = analyze_context(_transcript(CEO), None)
    parts = res.components["behavioural_risk"]["detail"]["parts"]
    for name, p in parts.items():
        assert {"value", "strength", "evidence", "quotes"} <= set(p)
        if p["value"] > 0:
            assert p["quotes"]
