#!/usr/bin/env python3
"""Measure Whisper word error rate per corpus tier against real ground truth.

We have exact ground truth for every tier:
  genuine  — LibriSpeech ships `*.trans.txt` alongside the audio.
  piper    — the scam script text Piper was given, chunk for chunk.
  cloned   — the same script text XTTS was given.

This is a MEASUREMENT, not an attempt to improve Whisper. Context signals are
extracted from ASR output, so ASR error is an upper bound on context recall:
a signal whose trigger phrase was mis-transcribed cannot fire.

  python scripts/measure_wer.py
  python scripts/measure_wer.py --json data/eval/asr_wer.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from voiceshield.asr.worker import AsrWorker            # noqa: E402
from voiceshield.asr.segmenter import transcribe_utterances  # noqa: E402
from voiceshield.config import get_settings             # noqa: E402
from voiceshield.ingest import vad as vad_mod           # noqa: E402
from voiceshield.ingest.audio import load_audio         # noqa: E402

ASSETS = REPO / "demo_assets"

_NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
}


def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split hyphenated compounds.

    Deliberately lenient about formatting ("twenty-five" -> "twenty five") so we
    measure recognition error, not transcription-style differences.
    """
    t = text.lower()
    t = t.replace("₹", " rupees ")
    t = re.sub(r"[-–—/]", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    toks = t.split()
    return [_NUM_WORDS.get(w, w) for w in toks]


def wer(ref: list[str], hyp: list[str]) -> tuple[float, dict]:
    """Standard Levenshtein word error rate with S/D/I breakdown."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return (0.0 if m == 0 else 1.0), {"S": 0, "D": 0, "I": m, "N": 0}
    d = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
        bt[i][0] = "D"
    for j in range(m + 1):
        d[0][j] = j
        bt[0][j] = "I"
    bt[0][0] = ""
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j], bt[i][j] = d[i - 1][j - 1], "="
            else:
                sub, dele, ins = d[i - 1][j - 1] + 1, d[i - 1][j] + 1, d[i][j - 1] + 1
                best = min(sub, dele, ins)
                d[i][j] = best
                bt[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    i, j, counts = n, m, {"S": 0, "D": 0, "I": 0, "N": n}
    inserted: list[str] = []
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "=":
            i, j = i - 1, j - 1
        elif op == "S":
            counts["S"] += 1
            i, j = i - 1, j - 1
        elif op == "D":
            counts["D"] += 1
            i -= 1
        else:
            counts["I"] += 1
            inserted.append(hyp[j - 1])
            j -= 1
    counts["inserted_words"] = list(reversed(inserted))
    return d[n][m] / n, counts


def script_chunk_text(stem: str, chunk: int, n_chunks: int) -> str | None:
    """Reconstruct the exact text a TTS tier was given for one chunk, using the
    same splitter build_demo_assets.py used."""
    sys.path.insert(0, str(REPO / "scripts"))
    from build_demo_assets import _script_chunks

    p = ASSETS / "scripts" / f"{stem}.txt"
    if not p.exists():
        return None
    chunks = _script_chunks(p, n_chunks)
    return chunks[chunk - 1] if 0 < chunk <= len(chunks) else None


def librispeech_refs() -> dict[str, str]:
    """utt-id -> reference transcript, straight from the dev-clean tarball."""
    tar = ASSETS / "_cache" / "dev-clean.tar.gz"
    if not tar.exists():
        return {}
    out: dict[str, str] = {}
    with tarfile.open(tar) as t:
        for mem in t.getmembers():
            if mem.name.endswith(".trans.txt"):
                for line in t.extractfile(mem).read().decode().splitlines():
                    if " " in line:
                        uid, text = line.split(" ", 1)
                        out[uid] = text
    return out


def _lexicon_check(per_clip_insertions: list[tuple[str, list[str]]]) -> dict:
    """Do hallucinated insertions ever match a context lexicon pattern?

    The Phase 3 WER finding left this open: Whisper invents words on XTTS
    output, and an invented word matching a rule would produce a context flag
    quoting something nobody said. This closes it with a measurement.

    Insertions are checked PER CLIP. Concatenating them across clips is wrong —
    it manufactures spans that never existed in any transcript (our first
    attempt did exactly that and reported a bogus "right now" hit stitched from
    two different clips).
    """
    sys.path.insert(0, str(REPO / "backend"))
    from voiceshield.context.signals import LEXICONS, parse_amount

    import re as _re
    pats = [(sig, rule, pat) for sig, rules in LEXICONS.items() for rule, pat in rules]
    total = 0
    distinct: set[str] = set()
    word_hits, span_hits, amounts = [], [], []
    for clip, words in per_clip_insertions:
        total += len(words)
        distinct |= set(words)
        for w in set(words):
            for sig, rule, pat in pats:
                if _re.search(pat, w, _re.I):
                    word_hits.append({"clip": clip, "word": w,
                                      "signal": sig, "rule": rule})
        blob = " ".join(words)          # within ONE clip only
        for sig, rule, pat in pats:
            for mm in _re.finditer(pat, blob, _re.I):
                span_hits.append({"clip": clip, "span": mm.group(0)[:60],
                                  "signal": sig, "rule": rule})
        amt, _ = parse_amount(blob)
        if amt is not None:
            amounts.append({"clip": clip, "amount": amt})
    clean = not word_hits and not span_hits and not amounts
    return {
        "method": "per-clip; insertions are never concatenated across clips",
        "n_inserted_words": total,
        "n_distinct": len(distinct),
        "single_word_matches": word_hits,
        "within_clip_span_matches": span_hits,
        "amounts_parsed_from_insertions": amounts,
        "verdict": ("no hallucinated word, span or amount matches any lexicon "
                    "pattern — the false-positive risk flagged in Phase 3 is not "
                    "realised on this corpus"
                    if clean else
                    "SOME hallucinated text matches a lexicon — false-positive "
                    "risk is real"),
    }


def transcribe(path: Path) -> str:
    audio, sr = load_audio(path)
    vr = vad_mod.analyze(audio, sr)
    return transcribe_utterances(audio, sr, vr.segments).text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "data" / "eval" / "asr_wer.json"))
    ap.add_argument("--limit", type=int, default=0, help="cap clips per tier")
    args = ap.parse_args()

    w = AsrWorker.get()
    if not w.available:
        print(f"Whisper unavailable: {w.load_error}")
        return 2
    print(f"model: {w.model_id} on {w.device}\n")

    tiers: dict[str, list[tuple[Path, str]]] = {"genuine": [], "piper": [], "cloned": []}

    refs = librispeech_refs()
    for p in sorted((ASSETS / "genuine").glob("*.wav")):
        uid = p.stem.split("_", 2)[-1]
        if uid in refs:
            tiers["genuine"].append((p, refs[uid]))

    for tier, sub, pref in (("piper", "synthetic", "synthetic_"),
                            ("cloned", "cloned", "cloned_")):
        for p in sorted((ASSETS / sub).glob("*.wav")):
            m = re.match(rf"{pref}(.+)_p(\d+)$", p.stem)
            if not m:
                continue
            ref = script_chunk_text(m.group(1), int(m.group(2)), 3)
            if ref:
                tiers[tier].append((p, ref))

    report: dict = {"model": w.model_id, "device": w.device, "tiers": {}}
    print(f"{'tier':<10} {'n':>3} {'ref words':>10} {'WER':>8} {'sub':>6} {'del':>6} {'ins':>6}")
    print("-" * 56)
    for tier, items in tiers.items():
        if args.limit:
            items = items[: args.limit]
        if not items:
            print(f"{tier:<10}   0  (no clips / no ground truth)")
            continue
        tot = {"S": 0, "D": 0, "I": 0, "N": 0}
        clip_insertions: list[tuple[str, list[str]]] = []
        per_clip = []
        for path, ref in items:
            r, h = normalise(ref), normalise(transcribe(path))
            rate, c = wer(r, h)
            for k in ("S", "D", "I", "N"):
                tot[k] += c[k]
            ins = c.get("inserted_words", [])
            clip_insertions.append((path.name, ins))
            per_clip.append({"clip": path.name, "wer": round(rate, 4),
                             "ref_words": c["N"], **{k: c[k] for k in "SDI"},
                             "inserted_words": ins})
        overall = (tot["S"] + tot["D"] + tot["I"]) / max(1, tot["N"])
        report["tiers"][tier] = {
            "hallucinated_insertions": _lexicon_check(clip_insertions),
            "n_clips": len(items), "ref_words": tot["N"],
            "wer": round(overall, 4),
            "substitutions": tot["S"], "deletions": tot["D"], "insertions": tot["I"],
            "per_clip": per_clip,
        }
        print(f"{tier:<10} {len(items):>3} {tot['N']:>10} {overall*100:>7.1f}% "
              f"{tot['S']:>6} {tot['D']:>6} {tot['I']:>6}")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[ok] wrote {out}")

    g = report["tiers"].get("genuine", {}).get("wer")
    c = report["tiers"].get("cloned", {}).get("wer")
    if g is not None and c is not None and g > 0:
        print(f"\ncloned speech is {c/g:.1f}x the genuine WER "
              f"({c*100:.1f}% vs {g*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
