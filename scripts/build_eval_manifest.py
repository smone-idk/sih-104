#!/usr/bin/env python3
"""Build a reproducible evaluation manifest over the existing English corpus.

Columns: clip_id, file_path, category, scenario, expected_speaker,
         expected_label, reference_transcript

Categories are derived from where a clip came from and how it was generated —
NOT invented. A clip with no reference transcript says so explicitly
(`reference_transcript = null`, `has_reference = false`).

  python scripts/build_eval_manifest.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "scripts"))

ASSETS = REPO / "demo_assets"
OUT_JSON = REPO / "data" / "eval" / "manifest.json"
OUT_CSV = REPO / "data" / "eval" / "manifest.csv"

#: LibriSpeech speaker designated as the enrolled CFO (build_demo_assets.py)
ENROLLED_SPEAKER = "1272"

#: which scripts are fraud vs benign — from demo_assets/scripts/
BENIGN_SCRIPTS = {"genuine_control_en"}


def librispeech_refs() -> dict[str, str]:
    import tarfile
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


def script_text(stem: str, chunk: int | None, n_chunks: int = 3) -> str | None:
    from build_demo_assets import _clean_script, _script_chunks

    p = ASSETS / "scripts" / f"{stem}.txt"
    if not p.exists():
        return None
    if chunk is None:
        return _clean_script(p)
    chunks = _script_chunks(p, n_chunks)
    return chunks[chunk - 1] if 0 < chunk <= len(chunks) else None


def rows() -> list[dict]:
    out: list[dict] = []
    refs = librispeech_refs()

    # --- genuine (LibriSpeech) ---------------------------------------
    for p in sorted((ASSETS / "genuine").glob("*.wav")):
        m = re.match(r"(enrolled|genuine)_(\d+)_(.+)$", p.stem)
        if not m:
            continue
        tag, spk, uid = m.groups()
        enrolled = spk == ENROLLED_SPEAKER
        out.append({
            "clip_id": p.stem,
            "file_path": str(p.relative_to(REPO)),
            "category": "genuine_enrolled" if enrolled else "genuine_non_enrolled",
            "scenario": None,
            "expected_speaker": spk,
            "expected_label": "bonafide",
            "is_enrolled_speaker": enrolled,
            "content": "benign",          # audiobook prose, no fraud language
            "reference_transcript": refs.get(uid),
            "has_reference": uid in refs,
            "source_tier": "genuine",
        })

    # --- generated attack tiers --------------------------------------
    for tier, sub, pref, cat in (
        ("piper", "synthetic", "synthetic_", "synthetic_non_enrolled"),
        ("cloned", "cloned", "cloned_", "cloned_enrolled"),
    ):
        for p in sorted((ASSETS / sub).glob("*.wav")):
            m = re.match(rf"{pref}(.+)_p(\d+)$", p.stem)
            if not m:
                continue
            stem, chunk = m.group(1), int(m.group(2))
            out.append({
                "clip_id": p.stem,
                "file_path": str(p.relative_to(REPO)),
                "category": cat,
                "scenario": stem,
                "expected_speaker": ENROLLED_SPEAKER if tier == "cloned" else "piper_lessac",
                "expected_label": "spoof",
                "is_enrolled_speaker": tier == "cloned",
                "content": "benign" if stem in BENIGN_SCRIPTS else "fraud_script",
                "reference_transcript": script_text(stem, chunk),
                "has_reference": script_text(stem, chunk) is not None,
                "source_tier": tier,
            })

    # --- full-length scenario renders --------------------------------
    for p in sorted((ASSETS / "scenarios").glob("*.wav")):
        m = re.match(r"(synthetic|cloned)_(.+)_full$", p.stem)
        if not m:
            continue
        kind, stem = m.groups()
        out.append({
            "clip_id": p.stem,
            "file_path": str(p.relative_to(REPO)),
            "category": ("cloned_enrolled" if kind == "cloned"
                         else "synthetic_non_enrolled"),
            "scenario": stem,
            "expected_speaker": ENROLLED_SPEAKER if kind == "cloned" else "piper_lessac",
            "expected_label": "spoof",
            "is_enrolled_speaker": kind == "cloned",
            "content": "benign" if stem in BENIGN_SCRIPTS else "fraud_script",
            "reference_transcript": script_text(stem, None),
            "has_reference": script_text(stem, None) is not None,
            "source_tier": "scenario",
        })

    # --- ASVspoof control set ----------------------------------------
    lab = ASSETS / "_asvspoof_dev" / "labels.json"
    if lab.exists():
        labels = json.loads(lab.read_text(encoding="utf-8"))
        for fn, label in sorted(labels.items()):
            out.append({
                "clip_id": Path(fn).stem,
                "file_path": str((ASSETS / "_asvspoof_dev" / fn).relative_to(REPO)),
                "category": "asvspoof_control",
                "scenario": None,
                "expected_speaker": None,
                "expected_label": label,
                "is_enrolled_speaker": False,
                "content": "unknown",
                "reference_transcript": None,
                "has_reference": False,
                "source_tier": "asvspoof_dev",
            })
    return out


def main() -> int:
    r = rows()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(r[0].keys()))
        w.writeheader()
        w.writerows(r)

    from collections import Counter
    print(f"{'category':<26} {'n':>4} {'with reference':>15}")
    for cat, n in sorted(Counter(x["category"] for x in r).items()):
        ref = sum(1 for x in r if x["category"] == cat and x["has_reference"])
        print(f"{cat:<26} {n:>4} {ref:>15}")
    print(f"{'TOTAL':<26} {len(r):>4} {sum(1 for x in r if x['has_reference']):>15}")
    noref = [x["clip_id"] for x in r if not x["has_reference"]]
    print(f"\nclips WITHOUT a reference transcript: {len(noref)}")
    print(f"  (all genuine LibriSpeech clips have refs from the tarball;"
          f" ASVspoof control clips have none)")
    print(f"\n[ok] {OUT_JSON}\n[ok] {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
