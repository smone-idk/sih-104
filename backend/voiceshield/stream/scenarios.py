"""Bundled demo scenarios (§13).

Each scenario is backed by a REAL audio file in demo_assets/. "Start simulation"
streams that file through the same WebSocket pipeline as any other input — the
transport is simulated, the analysis is not, and there is no scripted score.

The genuine control is deliberately included: a demo where a real human call
correctly scores LOW is more persuasive than five that score HIGH.

Attack scenarios point at `scenarios/*_full.wav` — the WHOLE script rendered as
one call. They used to point at `_p1` chunks, which contain only the call
opening, so the OTP ask and the money demand sat in files the demo never
analysed and the context engine had nothing to find. The chunked `synthetic/`
and `cloned/` tiers remain the measurement corpus.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from ..config import get_settings


@dataclass
class Scenario:
    id: str
    title: str
    description: str
    tier: str                # genuine | synthetic | cloned
    expected: str            # what a working system SHOULD do — not a hardcoded score
    relpath: str
    #: who the caller SAYS they are
    caller_claims: str = ""
    #: which enterprise-directory record the caller's number resolves to.
    #: DEMO DATA (§2 Tier C) — the attacks come from unknown numbers claiming a
    #: identity they cannot substantiate; the genuine calls come from the real
    #: contact's number. This is metadata, not anything derived from the audio.
    directory_contact: str = "Unknown Caller"
    language: str = "en"

    def path(self) -> Path:
        return get_settings().demo_assets_dir / self.relpath

    def as_dict(self) -> dict:
        d = asdict(self)
        p = self.path()
        d["available"] = p.exists()
        d["duration_s"] = None
        if p.exists():
            try:
                import soundfile as sf
                d["duration_s"] = round(sf.info(str(p)).duration, 2)
            except Exception:
                pass
        return d


SCENARIOS: list[Scenario] = [
    Scenario(
        id="ceo_transfer_cloned",
        title="CEO/CFO fund-transfer impersonation (voice clone)",
        description="Caller uses a cloned voice of the enrolled CFO to push an "
                    "urgent transfer to a new beneficiary.",
        tier="cloned",
        expected="synthetic markers present AND speaker matches the enrolled "
                 "profile -> CLONED_VOICE",
        relpath="scenarios/cloned_ceo_transfer_en_full.wav",
        caller_claims="Rajesh Sharma — CFO",
    ),
    Scenario(
        id="bank_otp_cloned",
        title="Bank official OTP solicitation (voice clone)",
        description="Cloned voice claiming to be from the bank's fraud desk, "
                    "asking for an OTP.",
        tier="cloned",
        expected="CLONED_VOICE; credential solicitation once context lands (Phase 3)",
        relpath="scenarios/cloned_bank_otp_en_full.wav",
        caller_claims="HDFC Fraud Dept (claimed)",
        directory_contact="HDFC Fraud Dept (claimed)",
    ),
    Scenario(
        id="govt_summons_synthetic",
        title="Government official — fake summons/penalty (TTS)",
        description="Synthetic voice, unrelated speaker, threatening legal action.",
        tier="synthetic",
        expected="synthetic markers present AND speaker mismatch -> SYNTHETIC_OTHER",
        relpath="scenarios/synthetic_govt_summons_en_full.wav",
        caller_claims="Inspector Verma (claimed)",
        directory_contact="Inspector Verma (claimed)",
    ),
    Scenario(
        id="family_emergency_synthetic",
        title="Family-member emergency (TTS)",
        description="Synthetic voice claiming a relative is in trouble and needs money.",
        tier="synthetic",
        expected="SYNTHETIC_OTHER",
        relpath="scenarios/synthetic_family_emergency_en_full.wav",
        caller_claims="Unknown Caller",
    ),
    Scenario(
        id="genuine_control",
        title="Genuine call — control",
        description="A real human recording of the enrolled speaker. This is the "
                    "control: it MUST score LOW.",
        tier="genuine",
        expected="no synthetic markers, speaker matches -> CONSISTENT, band LOW",
        relpath="genuine/enrolled_1272_1272-128104-0002.wav",
        caller_claims="Rajesh Sharma — CFO",
        directory_contact="Rajesh Sharma",
    ),
    Scenario(
        id="benign_script_cloned",
        title="Benign call — but in a cloned voice",
        description="The harmless team-lunch script, rendered in a clone of the "
                    "enrolled CFO's voice. Nothing fraudulent is said.",
        tier="cloned",
        expected="acoustic layers detect the clone (CLONED_VOICE); context finds "
                 "no fraud language, so the context layers report not-applicable "
                 "rather than lowering the score",
        relpath="scenarios/cloned_genuine_control_en_full.wav",
        caller_claims="Rajesh Sharma — CFO",
        directory_contact="Rajesh Sharma",
    ),
    Scenario(
        id="genuine_other_speaker",
        title="Genuine call — different human speaker",
        description="A real human, but not the enrolled CFO. Tests that a speaker "
                    "mismatch is reported as mismatch, not as a deepfake.",
        tier="genuine",
        expected="no synthetic markers, speaker mismatch -> SPEAKER_MISMATCH",
        relpath="genuine/genuine_1462_1462-170138-0000.wav",
        caller_claims="Rajesh Sharma — CFO",
        directory_contact="Priya Nair",
    ),
]

_BY_ID = {s.id: s for s in SCENARIOS}


def get(scenario_id: str) -> Scenario | None:
    return _BY_ID.get(scenario_id)


def listing() -> list[dict]:
    return [s.as_dict() for s in SCENARIOS]
