#!/usr/bin/env python3
"""V1 validation measurements over data/eval/manifest.json.

Measures, with n reported everywhere:
  - context engine precision/recall/F1 per signal family, vs reference text
  - anti-spoofing ROC-AUC / EER / FAR / FRR / confusion, AntiDeepfake vs AASIST,
    keeping "synthetic non-enrolled" and "cloned enrolled" as separate cases
  - speaker verification genuine vs impostor distributions, EER/FAR/FRR,
    cloned-enrolled kept separate from genuine-enrolled
  - fusion + policy per clip: components, availability, redistribution, floors

Writes data/eval/v1_eval.json. Changes no behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from voiceshield.asr.segmenter import transcribe_utterances   # noqa: E402
from voiceshield.config import get_settings                   # noqa: E402
from voiceshield.context.engine import analyze_context        # noqa: E402
from voiceshield.context.signals import LEXICONS, extract_all  # noqa: E402
from voiceshield.ingest import vad as vad_mod                 # noqa: E402
from voiceshield.ingest.audio import load_audio               # noqa: E402
from voiceshield.ingest.telephony import TelephonyConfig, degrade  # noqa: E402
from voiceshield.ml.detectors.speaker_ecapa import SpeakerConsistencyDetector  # noqa: E402
from voiceshield.ml.detectors.synthetic_aasist import AasistSyntheticDetector  # noqa: E402
from voiceshield.ml.detectors.synthetic_antideepfake import AntiDeepfakeDetector  # noqa: E402
from voiceshield.pipeline import analyze_file                 # noqa: E402

MANIFEST = REPO / "data" / "eval" / "manifest.json"
OUT = REPO / "data" / "eval" / "v1_eval.json"


# ----------------------------------------------------------- statistics
def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(a positive scores above a negative). Threshold-free."""
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    if not len(neg) or not len(pos):
        return float("nan")
    from itertools import product
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for n, p in product(neg, pos))
    return wins / (len(neg) * len(pos))


def eer(neg: np.ndarray, pos: np.ndarray) -> dict:
    """Equal error rate plus the FAR/FRR at that operating point."""
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    if not len(neg) or not len(pos):
        return {"eer_pct": None, "threshold": None, "far_pct": None, "frr_pct": None}
    best = None
    for t in np.unique(np.concatenate([neg, pos])):
        far = float((neg >= t).mean())   # negatives accepted as positive
        frr = float((pos < t).mean())    # positives rejected
        if best is None or abs(far - frr) < abs(best[0] - best[1]):
            best = (far, frr, float(t))
    far, frr, t = best
    return {"eer_pct": round((far + frr) / 2 * 100, 2), "threshold": round(t, 4),
            "far_pct": round(far * 100, 2), "frr_pct": round(frr * 100, 2)}


def confusion(neg: np.ndarray, pos: np.ndarray, thr: float) -> dict:
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    return {"threshold": thr,
            "tp": int((pos >= thr).sum()), "fn": int((pos < thr).sum()),
            "fp": int((neg >= thr).sum()), "tn": int((neg < thr).sum()),
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def dist(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    if not len(x):
        return {}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 4),
            "median": round(float(np.median(x)), 4),
            "min": round(float(x.min()), 4), "max": round(float(x.max()), 4),
            "p10": round(float(np.percentile(x, 10)), 4),
            "p90": round(float(np.percentile(x, 90)), 4)}


# ------------------------------------------------------------ measures
def clip_detector_scores(det, path: Path, tele: TelephonyConfig | None = None) -> float:
    from voiceshield.ingest.chunker import iter_windows
    audio, sr = load_audio(path)
    if tele is not None:
        audio, sr = degrade(audio, sr, tele, seed=0)
    vr = vad_mod.analyze(audio, sr)
    vals = [det.analyze(w.samples, sr).score for w in iter_windows(audio, sr)
            if vr.ratio_in(w.t_start, w.t_end) >= get_settings().window_speech_ratio]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def context_prf(rows: list[dict]) -> dict:
    """Per-family precision/recall/F1 against the reference transcript.

    Ground truth = the family fires on the reference text. Hypothesis = it fires
    on what the context engine actually received. Only clips WITH a reference.
    """
    fams = sorted(LEXICONS)
    tp = {f: 0 for f in fams}
    fp = {f: [] for f in fams}
    fn = {f: [] for f in fams}
    quote_bugs = []
    for r in rows:
        if not r["has_reference"]:
            continue
        audio, sr = load_audio(REPO / r["file_path"])
        vr = vad_mod.analyze(audio, sr)
        tr = transcribe_utterances(audio, sr, vr.segments)
        ctx = analyze_context(tr, None)
        ref = {f for f, s in extract_all(r["reference_transcript"]).items() if s.value > 0}
        hyp = {f for f, s in ctx.signals.items() if s.value > 0}
        for f in fams:
            if f in hyp and f in ref:
                tp[f] += 1
            elif f in hyp:
                q = [m.quote for m in ctx.signals[f].matches][:2]
                fp[f].append({"clip": r["clip_id"], "quotes": q})
            elif f in ref:
                fn[f].append({"clip": r["clip_id"],
                              "ref_quotes": [m.quote for m in
                                             extract_all(r["reference_transcript"])[f].matches][:2]})
        # every fired signal must carry quote + timestamp + rule
        for f, sig in ctx.signals.items():
            for m in sig.matches:
                if not m.quote.strip() or m.rule == "" or m.t_start is None:
                    quote_bugs.append({"clip": r["clip_id"], "family": f,
                                       "quote": m.quote, "rule": m.rule,
                                       "t_start": m.t_start})
    out = {}
    for f in fams:
        p = tp[f] / max(1, tp[f] + len(fp[f]))
        rc = tp[f] / max(1, tp[f] + len(fn[f]))
        out[f] = {
            "tp": tp[f], "fp": len(fp[f]), "fn": len(fn[f]),
            "precision": round(p, 3) if (tp[f] + len(fp[f])) else None,
            "recall": round(rc, 3) if (tp[f] + len(fn[f])) else None,
            "f1": round(2 * p * rc / (p + rc), 3) if (p + rc) else None,
            "false_positives": fp[f][:5],
            "false_negatives": fn[f][:5],
        }
    return {"per_family": out, "signals_missing_quote_or_timestamp": quote_bugs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-telephony", action="store_true")
    args = ap.parse_args()

    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    s = get_settings()
    report: dict = {"n_clips": len(rows), "settings": {
        "asr_min_avg_logprob": s.asr_min_avg_logprob,
        "asr_max_no_speech_prob": s.asr_max_no_speech_prob,
        "synthetic_high_threshold": s.synthetic_high_threshold,
        "speaker_match_threshold": s.speaker_match_threshold,
        "fusion_weights": s.fusion_weights(),
    }}

    def by(cat):
        return [r for r in rows if r["category"] == cat]

    # ---------------- anti-spoofing ----------------
    adf = AntiDeepfakeDetector(); adf.load()
    aas = AasistSyntheticDetector(); aas.load()
    groups = {c: by(c) for c in ("genuine_enrolled", "genuine_non_enrolled",
                                 "synthetic_non_enrolled", "cloned_enrolled",
                                 "asvspoof_control")}
    spoof: dict = {}
    for dname, det in (("AntiDeepfake", adf), ("AASIST", aas)):
        if not det.available:
            spoof[dname] = {"available": False, "error": det.load_error}
            continue
        sc = {}
        for cat, rs in groups.items():
            if cat == "asvspoof_control":
                continue
            sc[cat] = np.array([clip_detector_scores(det, REPO / r["file_path"])
                                for r in rs])
        genuine = np.concatenate([sc["genuine_enrolled"], sc["genuine_non_enrolled"]])
        spoof[dname] = {
            "available": True,
            "distributions": {k: dist(v) for k, v in sc.items()},
            "genuine_all": dist(genuine),
            "vs_synthetic_non_enrolled": {
                "auc": round(auc(genuine, sc["synthetic_non_enrolled"]), 4),
                **eer(genuine, sc["synthetic_non_enrolled"]),
                "confusion_at_operating_threshold": confusion(
                    genuine, sc["synthetic_non_enrolled"], s.synthetic_high_threshold)},
            "vs_cloned_enrolled": {
                "auc": round(auc(genuine, sc["cloned_enrolled"]), 4),
                **eer(genuine, sc["cloned_enrolled"]),
                "confusion_at_operating_threshold": confusion(
                    genuine, sc["cloned_enrolled"], s.synthetic_high_threshold)},
        }
        # ASVspoof home-turf control
        ctrl = groups["asvspoof_control"]
        if ctrl:
            b = np.array([clip_detector_scores(det, REPO / r["file_path"])
                          for r in ctrl if r["expected_label"] == "bonafide"])
            p = np.array([clip_detector_scores(det, REPO / r["file_path"])
                          for r in ctrl if r["expected_label"] == "spoof"])
            spoof[dname]["asvspoof_control"] = {
                "bonafide": dist(b), "spoof": dist(p),
                "auc": round(auc(b, p), 4), **eer(b, p)}
    report["anti_spoofing"] = spoof

    # ---------------- speaker verification ----------------
    spk = SpeakerConsistencyDetector(); spk.load()
    if spk.available:
        refs = sorted((REPO / "demo_assets" / "genuine").glob("enrolled_1272*.wav"))[:3]
        emb = np.mean([spk.embed(*load_audio(r)) for r in refs], axis=0)
        emb = (emb / np.linalg.norm(emb)).astype("float32")

        def cosines(rs):
            from voiceshield.ingest.chunker import iter_windows
            out = []
            for r in rs:
                audio, sr = load_audio(REPO / r["file_path"])
                vr = vad_mod.analyze(audio, sr)
                cs = []
                for w in iter_windows(audio, sr):
                    if vr.ratio_in(w.t_start, w.t_end) < s.window_speech_ratio:
                        continue
                    res = spk.analyze(w.samples, sr, {"enrolled_embedding": emb})
                    if res.available:
                        cs.append(res.detail["cosine_similarity"])
                out.append(float(np.mean(cs)) if cs else float("nan"))
            return np.array(out)

        gen_enr = cosines(by("genuine_enrolled"))
        imposter = cosines(by("genuine_non_enrolled"))
        cloned = cosines(by("cloned_enrolled"))
        synth = cosines(by("synthetic_non_enrolled"))
        report["speaker_verification"] = {
            "note": "cosine similarity vs the enrolled profile; higher = more "
                    "like the enrolled speaker",
            "genuine_enrolled": dist(gen_enr),
            "genuine_non_enrolled_impostor": dist(imposter),
            "cloned_enrolled": dist(cloned),
            "synthetic_non_enrolled": dist(synth),
            "genuine_vs_impostor": {"auc": round(auc(imposter, gen_enr), 4),
                                    **eer(imposter, gen_enr)},
            "cloned_scores_like_enrolled": {
                "auc_cloned_vs_impostor": round(auc(imposter, cloned), 4)},
        }

    # ---------------- context engine ----------------
    report["context_engine"] = context_prf(rows)

    # ---------------- fusion + policy per clip ----------------
    per_clip = []
    for r in rows:
        if r["category"] == "asvspoof_control":
            continue
        ctx = {}
        d = analyze_file(REPO / r["file_path"], ctx=ctx).as_dict()
        comps = {c["name"]: c for c in d["fusion"]["components"]}
        per_clip.append({
            "clip_id": r["clip_id"], "category": r["category"],
            "content": r["content"],
            "score": d["score"], "band": d["band"],
            "band_from_score": d["fusion"]["band_from_score"],
            "verdict": d["voice_verdict"],
            "floors_applied": [f["code"] for f in d["fusion"]["floors_applied"]],
            "redistributed": d["fusion"]["redistributed"],
            "unavailable": d["fusion"]["unavailable_components"],
            "components": {k: {"value": v["raw_value"],
                               "available": v["available"],
                               "eff_weight": v["effective_weight"]}
                           for k, v in comps.items()},
            "n_quotes": len(d["context_quotes"]),
            "transcript_discarded": d["transcript"].get("n_discarded", 0),
        })
    report["fusion_policy_per_clip"] = per_clip

    # invariants
    zero_valued_unavailable = [
        c["clip_id"] for c in per_clip
        for k, v in c["components"].items()
        if not v["available"] and v["value"] is not None]
    weights_sum = [round(sum(v["eff_weight"] for v in c["components"].values()), 4)
                   for c in per_clip]
    report["invariants"] = {
        "unavailable_components_that_carry_a_value": zero_valued_unavailable,
        "effective_weights_sum_min": min(weights_sum) if weights_sum else None,
        "effective_weights_sum_max": max(weights_sum) if weights_sum else None,
    }

    # ---------------- telephony sweep ----------------
    if not args.skip_telephony and adf.available:
        sweep = {}
        for label, cfg in (("clean", None),
                           ("20dB", TelephonyConfig(enabled=True, narrowband_hz=16000,
                                                    mu_law=False, add_noise=True, snr_db=20)),
                           ("10dB", TelephonyConfig(enabled=True, narrowband_hz=16000,
                                                    mu_law=False, add_noise=True, snr_db=10))):
            sweep[label] = {}
            for cat in ("genuine_enrolled", "genuine_non_enrolled",
                        "synthetic_non_enrolled", "cloned_enrolled"):
                v = np.array([clip_detector_scores(adf, REPO / r["file_path"], cfg)
                              for r in by(cat)])
                sweep[label][cat] = dist(v)
        report["telephony_sweep"] = {
            "seed": 0, "reproducible": True,
            "note": "additive white noise only (narrowband/mu-law disabled) so the "
                    "SNR variable is isolated",
            "results": sweep}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ok] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
