#!/usr/bin/env python3
"""Batch analysis CLI (§17 Phase 1).

    python analyze.py clip.wav
    python analyze.py clip.mp3 --telephony --snr 15
    python analyze.py clip.wav --no-profile        # ignore any enrolled profile
    python analyze.py clip.wav --summary           # short human-readable table

Prints the full analysis JSON on stdout. Same pipeline as streaming/simulation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voiceshield.ingest.telephony import TelephonyConfig      # noqa: E402
from voiceshield.pipeline import analyze_file                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="VoiceShield batch analysis")
    ap.add_argument("audio", help="WAV / MP3 / M4A / FLAC / OGG")
    ap.add_argument("--telephony", action="store_true",
                    help="pass through the 8 kHz + mu-law degradation chain (§8)")
    ap.add_argument("--snr", type=float, default=None,
                    help="add noise at this SNR (dB) in the telephony chain")
    ap.add_argument("--profile-id", default=None, help="specific voice profile id")
    ap.add_argument("--no-profile", action="store_true",
                    help="do not use any enrolled speaker profile")
    ap.add_argument("--summary", action="store_true", help="short table instead of JSON")
    args = ap.parse_args()

    tele = TelephonyConfig(
        enabled=args.telephony,
        add_noise=args.snr is not None,
        snr_db=args.snr if args.snr is not None else 20.0,
    )
    ctx: dict = {}
    if args.no_profile:
        ctx["enrolled_embedding"] = None  # explicit: skip profile resolution
    if args.profile_id:
        ctx["profile_id"] = args.profile_id

    try:
        res = analyze_file(args.audio, source="cli", ctx=ctx, telephony=tele)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2

    d = res.as_dict()
    if not args.summary:
        print(json.dumps(d, indent=2))
        return 0

    print(f"file        : {d['source']}")
    print(f"duration    : {d['duration_s']}s   speech windows: {d['n_speech_windows']}/{d['n_windows']}")
    print(f"device      : {d['device']}   telephony: {d['telephony_degraded']}")
    print(f"SCORE       : {d['score']}  ({d['band']})")
    if d["fusion"]["redistributed"]:
        print("              [weights redistributed — some layers unavailable]")
    print("\ncomponent              value    weight  eff.wt   points  status")
    for c in d["fusion"]["components"]:
        rv = "  n/a " if c["raw_value"] is None else f"{c['raw_value']:.3f}"
        print(f"  {c['name']:<20} {rv:>6}  {c['weight']:.2f}   {c['effective_weight']:.3f}   "
              f"{c['contribution_points']:>6.2f}  {'ok' if c['available'] else c['note']}")
    print(f"\nlatency (ms): {d['latency_ms']}")
    print(f"\n{d['fusion']['disclaimer']}")
    for w in d["warnings"]:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
