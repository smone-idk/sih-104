#!/usr/bin/env python3
"""Validate a synthetic-speech detector BEFORE trusting it (Phase 1.5b Task B).

This is the same diagnosis AASIST failed. A newer model gets no benefit of the
doubt: it must (1) prove its wrapper is correct on labelled home-turf data,
(2) state its polarity explicitly, and (3) separate genuine from XTTS-cloned
speech on OUR corpus — Piper alone does not count, Piper is the easy target.

  python scripts/validate_antideepfake.py                 # full report
  python scripts/validate_antideepfake.py --skip-telephony

Outputs a model x tier x clean/telephony table of means and separation, and
writes JSON to data/eval/detector_validation.json for the Phase 6 Evaluation page.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from voiceshield.ingest import vad as vad_mod                      # noqa: E402
from voiceshield.ingest.audio import load_audio                    # noqa: E402
from voiceshield.ingest.chunker import iter_windows                # noqa: E402
from voiceshield.ingest.telephony import TelephonyConfig, degrade  # noqa: E402
from voiceshield.ml.detectors.synthetic_aasist import AasistSyntheticDetector    # noqa: E402
from voiceshield.ml.detectors.synthetic_antideepfake import AntiDeepfakeDetector  # noqa: E402

ASSETS = REPO / "demo_assets"
TIERS = {
    "genuine": str(ASSETS / "genuine" / "*.wav"),
    "piper": str(ASSETS / "synthetic" / "*.wav"),
    "cloned_xtts": str(ASSETS / "cloned" / "*.wav"),
}
TELE = TelephonyConfig(enabled=True, mu_law=True, add_noise=False)


def noise_cfg(snr_db: float) -> TelephonyConfig:
    """Additive white noise at a target SNR, WITHOUT narrowbanding or mu-law —
    isolates the noise-floor variable."""
    return TelephonyConfig(enabled=True, narrowband_hz=16000, mu_law=False,
                           add_noise=True, snr_db=snr_db)


def clip_score(det, path: str, telephony: bool,
               noise_snr: float | None = None) -> float:
    """Mean detector score over the clip's speech windows — the same
    aggregation the pipeline uses, so these numbers transfer."""
    audio, sr = load_audio(path)
    if telephony:
        audio, sr = degrade(audio, sr, TELE)
    if noise_snr is not None:
        audio, sr = degrade(audio, sr, noise_cfg(noise_snr))
    vr = vad_mod.analyze(audio, sr)
    vals = [det.analyze(w.samples, sr).score
            for w in iter_windows(audio, sr)
            if vr.ratio_in(w.t_start, w.t_end) >= 0.25]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def polarity_check(det, base: Path) -> dict:
    """Home-turf wrapper control: labelled ASVspoof2019 LA dev clips.

    A correct wrapper puts bonafide LOW and spoof HIGH on this set. If it
    cannot, the bug is ours and no result on our own corpus means anything.
    """
    labels_file = base / "labels.json"
    if not labels_file.exists():
        return {"status": "unavailable",
                "note": "ASVspoof dev subset not present — diagnosis incomplete"}
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    sc = {"bonafide": [], "spoof": []}
    for fn, lab in labels.items():
        audio, sr = load_audio(base / fn)
        r = det.analyze(audio, sr)
        if r.score is not None:
            sc[lab].append(r.score)
    b, s = np.array(sc["bonafide"]), np.array(sc["spoof"])
    if not len(b) or not len(s):
        return {"status": "unavailable"}
    allv = np.concatenate([b, s])
    best = (1.0, 1.0, 0.0)
    for t in np.unique(allv):
        far, frr = float((b >= t).mean()), float((s < t).mean())
        if abs(far - frr) < abs(best[0]):
            best = (far - frr, (far + frr) / 2, float(t))
    return {
        "status": "ok",
        "n_bonafide": len(b), "n_spoof": len(s),
        "bonafide_mean": round(float(b.mean()), 4),
        "bonafide_median": round(float(np.median(b)), 4),
        "spoof_mean": round(float(s.mean()), 4),
        "spoof_median": round(float(np.median(s)), 4),
        "eer_pct": round(best[1] * 100, 2),
        "accuracy_at_0.5_pct": round(
            float(((b < 0.5).sum() + (s >= 0.5).sum()) / (len(b) + len(s)) * 100), 1),
        "polarity_correct": bool(np.median(s) > np.median(b)),
    }


def separation(a: np.ndarray, b: np.ndarray) -> dict:
    """How cleanly does `b` (attack) sit above `a` (genuine)?"""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return {"separable": False}
    # AUC via Mann-Whitney U — threshold-free, so no tuning is possible
    from itertools import product
    wins = sum(1.0 if y > x else 0.5 if y == x else 0.0 for x, y in product(a, b))
    return {
        "genuine_mean": round(float(a.mean()), 4),
        "attack_mean": round(float(b.mean()), 4),
        "delta": round(float(b.mean() - a.mean()), 4),
        "genuine_max": round(float(a.max()), 4),
        "attack_min": round(float(b.min()), 4),
        "auc": round(wins / (len(a) * len(b)), 4),
        "separable": bool(b.min() > a.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-telephony", action="store_true")
    ap.add_argument("--noise-sweep", action="store_true",
                    help="also score every tier at 20/10/5 dB additive SNR. Tests "
                         "whether separation depends on our synthetic tiers having "
                         "no room tone (i.e. keying on digital silence).")
    args = ap.parse_args()

    dets = {}
    adf = AntiDeepfakeDetector(); adf.load()
    dets["AntiDeepfake(wav2vec-large)"] = adf
    aas = AasistSyntheticDetector(); aas.load()
    dets["AASIST(ASVspoof2019LA)"] = aas

    conditions = ["clean"] if args.skip_telephony else ["clean", "telephony_8k_mulaw"]
    report: dict = {"detectors": {}}

    for dname, det in dets.items():
        print(f"\n{'='*78}\n### {dname}   available={det.available}")
        if not det.available:
            print(f"  UNAVAILABLE: {det.load_error}")
            report["detectors"][dname] = {"available": False, "error": det.load_error}
            continue

        pol = polarity_check(det, ASSETS / "_asvspoof_dev")
        print(f"  polarity/home-turf (ASVspoof2019 LA dev): {pol.get('status')}")
        if pol.get("status") == "ok":
            print(f"    bonafide median={pol['bonafide_median']}  "
                  f"spoof median={pol['spoof_median']}  "
                  f"EER={pol['eer_pct']}%  acc@0.5={pol['accuracy_at_0.5_pct']}%  "
                  f"polarity_correct={pol['polarity_correct']}")

        entry = {"available": True, "polarity": pol, "conditions": {}}
        for cond in conditions:
            tel = cond != "clean"
            scores = {}
            for tier, pat in TIERS.items():
                files = sorted(glob.glob(pat))
                scores[tier] = np.array([clip_score(det, f, tel) for f in files])
                print(f"    [{cond}] {tier:<12} n={len(files):<3} "
                      f"mean={np.nanmean(scores[tier]):.4f}")
            entry["conditions"][cond] = {
                "n": {k: int(len(v)) for k, v in scores.items()},
                "means": {k: round(float(np.nanmean(v)), 4) for k, v in scores.items()},
                "vs_piper": separation(scores["genuine"], scores["piper"]),
                "vs_cloned_xtts": separation(scores["genuine"], scores["cloned_xtts"]),
            }
        report["detectors"][dname] = entry

    # --- summary table -------------------------------------------------
    print(f"\n\n{'='*100}\nSUMMARY — mean detector score by model x tier x condition")
    print(f"{'model':<28} {'condition':<20} {'genuine':>9} {'piper':>9} {'cloned':>9} "
          f"{'AUC piper':>10} {'AUC cloned':>11}")
    print("-" * 100)
    for dname, e in report["detectors"].items():
        if not e.get("available"):
            print(f"{dname:<28} UNAVAILABLE"); continue
        for cond, c in e["conditions"].items():
            m = c["means"]
            print(f"{dname:<28} {cond:<20} {m['genuine']:>9.4f} {m['piper']:>9.4f} "
                  f"{m['cloned_xtts']:>9.4f} {c['vs_piper']['auc']:>10.3f} "
                  f"{c['vs_cloned_xtts']['auc']:>11.3f}")
    print("\nAUC = P(attack clip scores above a genuine clip). 0.5 = no separation, "
          "1.0 = perfect.\nThreshold-free, so it cannot be tuned.")

    # --- noise sweep ---------------------------------------------------
    if args.noise_sweep:
        print(f"\n\n{'='*100}\nNOISE SWEEP — identical additive white noise applied to EVERY tier.")
        print("If separation depended on our synthetic tiers lacking room tone, "
              "equalising the\nnoise floor would collapse it.\n")
        print(f"{'model':<28} {'SNR':>8} {'genuine':>9} {'piper':>9} {'cloned':>9} "
              f"{'AUC piper':>10} {'AUC cloned':>11}")
        print("-" * 100)
        sweep: dict = {}
        for dname, det in dets.items():
            if not det.available:
                continue
            sweep[dname] = {}
            for snr in (None, 20.0, 10.0, 5.0):
                sc = {t: np.array([clip_score(det, f, False, snr)
                                   for f in sorted(glob.glob(p))])
                      for t, p in TIERS.items()}
                lbl = "clean" if snr is None else f"{snr:g} dB"
                vp = separation(sc["genuine"], sc["piper"])
                vc = separation(sc["genuine"], sc["cloned_xtts"])
                sweep[dname][lbl] = {
                    "means": {k: round(float(np.nanmean(v)), 4) for k, v in sc.items()},
                    "vs_piper": vp, "vs_cloned_xtts": vc,
                }
                print(f"{dname:<28} {lbl:>8} {np.nanmean(sc['genuine']):>9.4f} "
                      f"{np.nanmean(sc['piper']):>9.4f} {np.nanmean(sc['cloned_xtts']):>9.4f} "
                      f"{vp['auc']:>10.3f} {vc['auc']:>11.3f}")
        report["noise_sweep"] = sweep

    out = REPO / "data" / "eval" / "detector_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
