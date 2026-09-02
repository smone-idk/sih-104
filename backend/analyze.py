#!/usr/bin/env python3
"""Batch analysis CLI.  `python analyze.py clip.wav` -> JSON on stdout.

Phase 1 fills in the real implementation: one pipeline shared with streaming and
simulation (§14). Right now it only proves the registry loads and reports which
detectors would run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voiceshield.config import get_settings
from voiceshield.ml.registry import get_registry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", help="path to a WAV/MP3/M4A file")
    args = ap.parse_args()

    s = get_settings()
    reg = get_registry()
    out = {
        "phase": 0,
        "note": "batch pipeline lands in Phase 1",
        "device": s.device,
        "audio": args.audio,
        "detectors": reg.inventory(),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
