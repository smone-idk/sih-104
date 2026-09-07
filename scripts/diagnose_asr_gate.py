#!/usr/bin/env python3
"""THROWAWAY DIAGNOSTIC for the V1 validation pass. Changes no behaviour.

Answers, per clip:
  - how many ASR segments exist, how many pass the confidence gate
  - what Transcript.text actually contains
  - what analyze_context therefore received
  - which of avg_logprob / no_speech_prob is the binding constraint
  - the six-way transcription comparison from §1.3

Usage:
  python scripts/diagnose_asr_gate.py                 # whole corpus summary
  python scripts/diagnose_asr_gate.py --sixway CLIP   # one clip, six ways
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

from voiceshield.asr.segmenter import (                    # noqa: E402
    transcribe_utterances, utterances_from_segments)
from voiceshield.asr.worker import AsrWorker, Transcript   # noqa: E402
from voiceshield.config import get_settings                # noqa: E402
from voiceshield.context.engine import analyze_context     # noqa: E402
from voiceshield.ingest import vad as vad_mod              # noqa: E402
from voiceshield.ingest.audio import load_audio            # noqa: E402

ASSETS = REPO / "demo_assets"


def tiers() -> dict[str, list[str]]:
    return {
        "genuine": sorted(glob.glob(str(ASSETS / "genuine" / "*.wav"))),
        "piper": sorted(glob.glob(str(ASSETS / "synthetic" / "*.wav"))),
        "cloned": sorted(glob.glob(str(ASSETS / "cloned" / "*.wav"))),
        "scenarios": sorted(glob.glob(str(ASSETS / "scenarios" / "*.wav"))),
    }


def probe(path: str) -> dict:
    """One clip: segments, gate outcome, what context received."""
    s = get_settings()
    audio, sr = load_audio(path)
    vr = vad_mod.analyze(audio, sr)
    utts = utterances_from_segments(vr.segments, s)
    tr = transcribe_utterances(audio, sr, vr.segments)
    ctx = analyze_context(tr, None)

    passed = [x for x in tr.segments if x.confident]
    dropped = [x for x in tr.segments if not x.confident]
    # which threshold bound each drop?
    by_logprob = [x for x in dropped if x.avg_logprob < s.asr_min_avg_logprob]
    by_nospeech = [x for x in dropped
                   if x.no_speech_prob > s.asr_max_no_speech_prob]
    return {
        "clip": Path(path).name,
        "duration_s": round(len(audio) / sr, 2),
        "vad_segments": len(vr.segments),
        "vad_speech_ratio": round(vr.overall_ratio, 3),
        "asr_utterances_requested": len(utts),
        "utt_durations": [round(u.duration, 2) for u in utts],
        "n_segments": len(tr.segments),
        "n_passed": len(passed),
        "n_dropped": len(dropped),
        "all_dropped": bool(tr.segments) and not passed,
        "no_segments_at_all": not tr.segments,
        "text_len": len(tr.text),
        "text": tr.text[:160],
        "context_got_empty_text": not tr.text.strip(),
        "context_signal_families": len(ctx.signals),
        "bound_by_logprob": len(by_logprob),
        "bound_by_no_speech": len(by_nospeech),
        "logprobs": [round(x.avg_logprob, 3) for x in tr.segments],
        "no_speech": [round(x.no_speech_prob, 3) for x in tr.segments],
    }


def sixway(path: str) -> None:
    """§1.3: the six-way comparison for one clip."""
    s = get_settings()
    w = AsrWorker.get()
    audio, sr = load_audio(path)
    vr = vad_mod.analyze(audio, sr)

    print(f"\n{'='*78}\nSIX-WAY: {Path(path).name}  ({len(audio)/sr:.2f}s @ {sr} Hz)")
    print(f"{'='*78}")

    print("\n[1] FULL-AUDIO transcription (no VAD segmentation, one call)")
    full = w.transcribe(audio, sr)
    for x in full:
        print(f"    lp={x.avg_logprob:>7.3f} ns={x.no_speech_prob:>6.3f}  {x.text!r}")

    print(f"\n[2] VAD segments as found ({len(vr.segments)})")
    for sg in vr.segments:
        print(f"    {sg.start:>6.2f}-{sg.end:>6.2f}s  ({sg.duration:.2f}s)")

    utts = utterances_from_segments(vr.segments, s)
    print(f"\n[3] UTTERANCE segmentation handed to ASR ({len(utts)})")
    for u in utts:
        print(f"    {u.t_start:>6.2f}-{u.t_end:>6.2f}s  ({u.duration:.2f}s)")

    print("\n[4] RAW ASR per utterance, BEFORE gating")
    raw = []
    for u in utts:
        a, b = int(u.t_start * sr), min(len(audio), int(u.t_end * sr))
        if b - a < int(0.2 * sr):
            continue
        segs = w.transcribe(np.asarray(audio[a:b]), sr, t_offset=u.t_start,
                            index=len(raw))
        for x in segs:
            raw.append(x)
            print(f"    lp={x.avg_logprob:>7.3f} ns={x.no_speech_prob:>6.3f}  {x.text!r}")

    tr = transcribe_utterances(audio, sr, vr.segments)
    print(f"\n[5] AFTER gating (min_logprob={s.asr_min_avg_logprob}, "
          f"max_no_speech={s.asr_max_no_speech_prob})")
    for x in tr.segments:
        mark = "KEPT   " if x.confident else "DROPPED"
        print(f"    {mark} lp={x.avg_logprob:>7.3f}  {x.text!r}")

    print("\n[6] WHAT THE CONTEXT ENGINE RECEIVED")
    print(f"    Transcript.text = {tr.text!r}")
    ctx = analyze_context(tr, None)
    print(f"    signal families computed: {len(ctx.signals)}")
    for n, sig in sorted(ctx.signals.items()):
        if sig.value > 0:
            print(f"      {n}: {sig.value:.2f} {[m.quote for m in sig.matches][:3]}")
    for n, c in ctx.components.items():
        print(f"    component {n:<22} available={c['available']} value={c['value']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sixway", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.sixway:
        matches = [p for ps in tiers().values() for p in ps
                   if args.sixway in Path(p).name]
        if not matches:
            print(f"no clip matching {args.sixway!r}")
            return 2
        for m in matches[:3]:
            sixway(m)
        return 0

    s = get_settings()
    print(f"gate: enabled={s.asr_confidence_gate} "
          f"min_avg_logprob={s.asr_min_avg_logprob} "
          f"max_no_speech_prob={s.asr_max_no_speech_prob}\n")
    rows = []
    print(f"{'tier':<10} {'clip':<42} {'segs':>4} {'pass':>4} {'drop':>4} "
          f"{'txtlen':>6} {'ctxfam':>6}  flag")
    for tier, paths in tiers().items():
        for p in paths:
            r = probe(p)
            r["tier"] = tier
            rows.append(r)
            flag = ("ALL-DROPPED" if r["all_dropped"] else
                    "NO-SEGMENTS" if r["no_segments_at_all"] else "")
            print(f"{tier:<10} {r['clip'][:42]:<42} {r['n_segments']:>4} "
                  f"{r['n_passed']:>4} {r['n_dropped']:>4} {r['text_len']:>6} "
                  f"{r['context_signal_families']:>6}  {flag}")

    print("\n--- summary ---")
    tot = len(rows)
    empty = [r for r in rows if r["context_got_empty_text"]]
    alldrop = [r for r in rows if r["all_dropped"]]
    nosegs = [r for r in rows if r["no_segments_at_all"]]
    print(f"clips: {tot}")
    print(f"context received EMPTY text: {len(empty)}  "
          f"({', '.join(r['clip'] for r in empty) or 'none'})")
    print(f"  of which ALL segments dropped by the gate: {len(alldrop)}")
    print(f"  of which ASR returned NO segments at all : {len(nosegs)}")
    seg_tot = sum(r["n_segments"] for r in rows)
    drop_tot = sum(r["n_dropped"] for r in rows)
    lp_tot = sum(r["bound_by_logprob"] for r in rows)
    ns_tot = sum(r["bound_by_no_speech"] for r in rows)
    print(f"segments: {seg_tot} total, {drop_tot} dropped "
          f"({drop_tot/max(1,seg_tot)*100:.1f}%)")
    print(f"  bound by avg_logprob   : {lp_tot}")
    print(f"  bound by no_speech_prob: {ns_tot}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n[ok] wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
