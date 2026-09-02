#!/usr/bin/env python3
"""Held-out evaluation -> JSON for the Evaluation page (§9).

Computes EER, ROC/DET points, confusion matrix at the chosen threshold, and a
per-condition breakdown (clean vs 8 kHz telephony) over the small bundled demo
set. States the sample size next to every metric. These are NOT generalization
claims — see LIMITATIONS.md §6.

Phase 6 fills in the real implementation once the batch pipeline (Phase 1) and
telephony degradation (Phase 5) exist.

  python scripts/run_eval.py --out data/eval/latest.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "eval" / "latest.json"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.parse_args()
    print("run_eval.py is a Phase 6 deliverable — not implemented yet.")
    print("It needs the Phase 1 batch pipeline and the Phase 5 telephony chain.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
