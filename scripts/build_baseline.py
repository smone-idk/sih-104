#!/usr/bin/env python3
"""Estimate the human-baseline prosody distribution from the genuine demo clips
and write it to models/prosody_baseline.json.

The ProsodyAnomalyDetector scores anomaly as distance from this distribution.
Until this runs it uses built-in priors and says so on the card.

  python scripts/build_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from voiceshield.config import get_settings                       # noqa: E402
from voiceshield.ml.detectors.prosody import ProsodyAnomalyDetector  # noqa: E402


def main() -> int:
    s = get_settings()
    clips = sorted((s.demo_assets_dir / "genuine").glob("*.wav"))
    clips = [c for c in clips if not c.name.startswith("cloned")]
    if len(clips) < 3:
        print("Need >= 3 genuine clips in demo_assets/genuine — run build_demo_assets.py --tier genuine")
        return 1

    det = ProsodyAnomalyDetector()
    det.load()
    if not det.available:
        print(f"prosody deps unavailable: {det.load_error}")
        return 1

    import soundfile as sf

    rows: dict[str, list[float]] = {}
    for c in clips:
        data, sr = sf.read(c)
        feats = det.extract_features(np.asarray(data), sr)
        for k, v in feats.items():
            rows.setdefault(k, []).append(float(v))
        print(f"  {c.name}: {', '.join(f'{k}={v:.3f}' for k, v in feats.items())}")

    baseline = {
        k: [float(np.mean(v)), float(np.std(v) or 1.0)] for k, v in rows.items()
    }
    out = s.model_cache_dir / "prosody_baseline.json"
    out.write_text(json.dumps(
        {"n_clips": len(clips), "features": baseline,
         "source": "demo_assets/genuine (LibriSpeech dev-clean / Common Voice)"},
        indent=2), encoding="utf-8")
    print(f"\n[ok] wrote {out} from {len(clips)} genuine clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
