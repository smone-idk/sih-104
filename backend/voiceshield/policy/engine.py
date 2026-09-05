"""Policy engine — band -> action, and the approval gate (§7).

SECURITY MODEL, and it is the point of this module:

    The decision to block an approval is derived ONLY from server-side state.

The client sends an approval id and a session id. Nothing else it sends is
consulted — not a risk band, not a score, not a verification flag. The risk band
is read from the `sessions` row that the analysis pipeline wrote, and the
verification state from the `verifications` table. A judge with devtools open can
forge any payload they like and the answer does not change, because none of the
inputs to the decision come from the request body.

Actions (§7):
    ALLOW     LOW risk        — proceed
    VERIFY    MEDIUM risk     — proceed only after identity verification
    ESCALATE  HIGH risk       — blocked; verification required, supervisor notified
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings, get_settings

ALLOW, VERIFY, ESCALATE = "ALLOW", "VERIFY", "ESCALATE"

#: bands that block an approval outright until verification passes
BLOCKING_BANDS = {"HIGH"}
#: bands that require verification but are not treated as an active attack
VERIFY_BANDS = {"MEDIUM"}


@dataclass
class PolicyDecision:
    action: str                  # ALLOW | VERIFY | ESCALATE
    allowed: bool                # may the approval proceed right now?
    reason: str                  # human-readable, shown in the UI and logged
    band: str | None             # the band the decision was made on
    score: float | None
    http_status: int = 200
    error_code: str = ""
    detail: dict[str, Any] = None  # type: ignore[assignment]

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "allowed": self.allowed,
            "reason": self.reason,
            "band": self.band,
            "score": None if self.score is None else round(self.score, 2),
            "error_code": self.error_code,
            "detail": self.detail or {},
        }


def action_for_band(band: str | None, s: Settings | None = None) -> str:
    if band in BLOCKING_BANDS:
        return ESCALATE
    if band in VERIFY_BANDS:
        return VERIFY
    return ALLOW


def decide_approval(*, session: dict[str, Any] | None,
                    verification: dict[str, Any] | None,
                    settings: Settings | None = None) -> PolicyDecision:
    """Should this approval proceed?

    `session` and `verification` MUST come from the database. Callers must not
    pass anything derived from the request body — see the module docstring.
    """
    get_settings() if settings is None else settings

    if session is None:
        return PolicyDecision(
            action=ESCALATE, allowed=False,
            reason="No analysed call session is linked to this approval. A "
                   "high-value transfer cannot be approved without one.",
            band=None, score=None,
            http_status=403, error_code="verification_required",
            detail={"cause": "no_session"})

    band = session.get("final_band")
    score = session.get("final_score")

    if band is None:
        return PolicyDecision(
            action=ESCALATE, allowed=False,
            reason="The linked call session has not finished analysis yet.",
            band=None, score=score,
            http_status=403, error_code="verification_required",
            detail={"cause": "analysis_incomplete", "session_id": session.get("id")})

    action = action_for_band(band)
    passed = bool(verification and verification.get("result") == "passed")

    if action == ALLOW:
        return PolicyDecision(
            action=ALLOW, allowed=True,
            reason=f"Call risk is {band}; no verification required.",
            band=band, score=score)

    if passed:
        return PolicyDecision(
            action=action, allowed=True,
            reason=f"Call risk is {band}, but identity verification "
                   f"({verification.get('method')}) passed.",
            band=band, score=score,
            detail={"verification_id": verification.get("id"),
                    "method": verification.get("method")})

    if action == ESCALATE:
        reason = ("Possible AI voice impersonation — do not approve the "
                  "requested action until identity is independently verified.")
    else:
        reason = (f"Call risk is {band}. Identity verification is required "
                  "before this approval can proceed.")

    return PolicyDecision(
        action=action, allowed=False, reason=reason, band=band, score=score,
        http_status=403, error_code="verification_required",
        detail={"cause": "unverified",
                "session_id": session.get("id"),
                "verification_state": (verification or {}).get("result", "none")})
