"""Challenge–response verification (§7.3).

The system generates a random phrase, the caller is asked to repeat it, and the
response audio is re-analysed. This defeats pre-recorded/replay attacks: a
recording made before the challenge existed cannot contain the phrase.

What is REAL here:
  * the phrase is generated server-side from a CSPRNG, per attempt;
  * the response audio is run through the full analysis pipeline (same
    `analyze_audio` as everything else — §14);
  * the spoken words are compared against the issued phrase using the ASR
    transcript, with a word-overlap threshold;
  * both the acoustic verdict and the phrase match must pass.

What is NOT real: nothing about the analysis. The limitation is that a live
attacker with a real-time voice-conversion model could repeat the phrase in the
cloned voice — challenge–response defeats *replay*, not live conversion. That is
stated in the UI and in LIMITATIONS.
"""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

# Deliberately concrete, unrelated nouns and colours: easy to say, hard to
# stitch from a pre-recorded clip, and unlikely to appear in normal speech.
_WORDS = [
    "amber", "anchor", "basket", "bridge", "cactus", "candle", "copper",
    "crimson", "diamond", "engine", "falcon", "garden", "granite", "harbour",
    "indigo", "island", "jasmine", "kettle", "ladder", "lantern", "marble",
    "meadow", "orchid", "pepper", "pilot", "quartz", "ribbon", "saffron",
    "silver", "sparrow", "temple", "thunder", "tiger", "velvet", "walnut",
    "willow", "yellow", "zebra",
]
_NUMBERS = ["three", "seven", "nine", "twelve", "twenty", "forty"]

#: fraction of challenge words that must appear in the response transcript
PHRASE_MATCH_THRESHOLD = 0.6


@dataclass
class Challenge:
    phrase: str
    words: list[str]

    def as_dict(self) -> dict:
        return {"phrase": self.phrase, "n_words": len(self.words)}


def generate_challenge(n_words: int = 4) -> Challenge:
    """CSPRNG-backed. A fresh phrase per attempt, so a recording captured before
    the challenge was issued cannot satisfy it."""
    rng = secrets.SystemRandom()
    words = rng.sample(_WORDS, max(2, n_words - 1))
    words.insert(rng.randrange(len(words) + 1), rng.choice(_NUMBERS))
    return Challenge(phrase=" ".join(words), words=words)


def _norm(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def check_phrase(challenge_phrase: str, transcript: str) -> dict:
    """Did the caller actually say the phrase we issued?"""
    want = _norm(challenge_phrase)
    got = set(_norm(transcript))
    if not want:
        return {"matched": False, "ratio": 0.0, "missing": [], "threshold": PHRASE_MATCH_THRESHOLD}
    hit = [w for w in want if w in got]
    missing = [w for w in want if w not in got]
    ratio = len(hit) / len(want)
    return {
        "matched": ratio >= PHRASE_MATCH_THRESHOLD,
        "ratio": round(ratio, 3),
        "matched_words": hit,
        "missing": missing,
        "threshold": PHRASE_MATCH_THRESHOLD,
        "transcript": transcript,
    }


#: Simulated verification methods (§7.4) — a state machine, clearly labelled.
#: These do not contact anything; they exist so the workflow is complete and the
#: incident log is realistic. The UI badges them `SIMULATED`.
SIMULATED_METHODS = {
    "callback": "Registered callback to the number on file",
    "mfa": "Push MFA to the enrolled device",
    "supervisor": "Second-line supervisor approval",
}
