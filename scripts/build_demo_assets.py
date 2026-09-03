#!/usr/bin/env python3
"""Build the entire demo corpus locally — nobody records anything (§13).

Tiers produced under demo_assets/:
  genuine/    — LibriSpeech dev-clean (English) + Common Voice Hindi clips.
                One LibriSpeech speaker is designated "Rajesh Sharma — CFO";
                3 of their clips become the enrolled voice profile.
  synthetic/  — Piper TTS (MIT) reading the scam scripts in an UNRELATED voice.
                Expect: high synthetic prob AND low speaker similarity.
  cloned/     — Coqui XTTS-v2 cloning the enrolled speaker, same scripts.
                Expect: high synthetic prob AND high speaker similarity.
                (Coqui Public Model License — non-commercial; see LIMITATIONS.md)

Every clip is recorded in demo_assets/manifest.json with its source, licence,
model, generation parameters and SHA256. PROVENANCE.md is regenerated from that
manifest (idempotent — re-running never duplicates lines).

The synthetic/cloned tiers need the ISOLATED TTS venv (tools/.venv-tts), because
coqui-tts pulls transformers 5.x / librosa 0.11 which would silently change the
analysis stack. Run them with that interpreter:

  backend/.venv/bin/python   scripts/build_demo_assets.py --tier genuine
  tools/.venv-tts/bin/python scripts/build_demo_assets.py --tier synthetic
  tools/.venv-tts/bin/python scripts/build_demo_assets.py --tier cloned
  python scripts/build_demo_assets.py --provenance      # regenerate the doc only
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "demo_assets"
SCRIPTS = ASSETS / "scripts"
MANIFEST = ASSETS / "manifest.json"
PROV = REPO / "PROVENANCE.md"

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
ENROLLED_SPEAKER_LABEL = "Rajesh Sharma — CFO"
PIPER_VOICE = "en_US-lessac-medium"
PIPER_VOICE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
)
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# corpus size targets (§ Phase 1.5 Task A)
N_SPEAKERS = 8
N_CLIPS_PER_SPEAKER = 4


# ------------------------------------------------------------------ manifest
def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"clips": {}}


def save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def record(m: dict, path: Path, **fields) -> None:
    """Add/replace one clip's provenance entry, keyed by its repo-relative path."""
    import soundfile as sf

    rel = str(path.relative_to(ASSETS))
    try:
        info = sf.info(str(path))
        dur, sr = round(info.duration, 3), info.samplerate
    except Exception:
        dur, sr = None, None
    m["clips"][rel] = {
        "sha256": sha256(path),
        "duration_s": dur,
        "sample_rate": sr,
        "built": dt.date.today().isoformat(),
        **fields,
    }


# ------------------------------------------------------------------- genuine
def build_genuine(quick: bool, m: dict, variants: int = 1) -> None:
    out = ASSETS / "genuine"
    out.mkdir(parents=True, exist_ok=True)
    tar = ASSETS / "_cache" / "dev-clean.tar.gz"
    tar.parent.mkdir(parents=True, exist_ok=True)

    if not tar.exists():
        print(f"Downloading LibriSpeech dev-clean (~340 MB) -> {tar}")
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar)

    n_spk = 2 if quick else N_SPEAKERS
    n_clip = 3 if quick else N_CLIPS_PER_SPEAKER
    print(f"Extracting {n_spk} speakers x {n_clip} clips ...")

    import soundfile as sf

    with tarfile.open(tar) as t:
        members = [x for x in t.getmembers() if x.name.endswith(".flac")]
        members.sort(key=lambda x: x.name)
        by_spk: dict[str, list] = {}
        for mem in members:
            by_spk.setdefault(mem.name.split("/")[2], []).append(mem)

        speakers = sorted(by_spk)
        enrolled_spk = speakers[0]                      # 1272 — kept stable
        take = speakers[:n_spk]
        written = 0
        for spk in take:
            for mem in by_spk[spk][:n_clip]:
                data, sr = sf.read(t.extractfile(mem))
                tag = "enrolled" if spk == enrolled_spk else "genuine"
                dest = out / f"{tag}_{spk}_{Path(mem.name).stem}.wav"
                sf.write(dest, data, sr)
                record(m, dest, tier="genuine", language="en",
                       source="LibriSpeech dev-clean", licence="CC BY 4.0",
                       speaker_id=spk, upstream_path=mem.name,
                       model=None, params={"format": "flac->wav, unmodified"},
                       note=(ENROLLED_SPEAKER_LABEL if tag == "enrolled" else ""))
                written += 1

    (out / "ENROLLED_SPEAKER.txt").write_text(
        f"{enrolled_spk}\n{ENROLLED_SPEAKER_LABEL}\n"
        "First 3 'enrolled_*' clips are used by scripts/seed.py for the voice profile.\n",
        encoding="utf-8",
    )
    print(f"[ok] genuine: {written} clips from {len(take)} speakers -> {out}")
    print(f"     enrolled speaker: {enrolled_spk} ('{ENROLLED_SPEAKER_LABEL}')")
    print("     Hindi: fetch Common Voice hi manually (CC0) into demo_assets/genuine/ "
          "as hi_*.wav — see README (dataset requires a click-through).")


# ----------------------------------------------------------------- synthetic
def _piper_voice_files() -> tuple[Path, Path]:
    vdir = REPO / "models" / "piper"
    vdir.mkdir(parents=True, exist_ok=True)
    onnx = vdir / f"{PIPER_VOICE}.onnx"
    cfg = vdir / f"{PIPER_VOICE}.onnx.json"
    for f, url in ((onnx, PIPER_VOICE_URL + f"{PIPER_VOICE}.onnx"),
                   (cfg, PIPER_VOICE_URL + f"{PIPER_VOICE}.onnx.json")):
        if not f.exists():
            print(f"  downloading {f.name}")
            urllib.request.urlretrieve(url, f)
    return onnx, cfg


def build_synthetic(quick: bool, m: dict, variants: int = 1) -> None:
    out = ASSETS / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    try:
        from piper import PiperVoice, SynthesisConfig
    except Exception as exc:
        print(f"piper not importable ({exc}).")
        print("Run this tier with the TTS venv: tools/.venv-tts/bin/python "
              "scripts/build_demo_assets.py --tier synthetic")
        return
    import wave

    onnx, _cfg = _piper_voice_files()
    voice = PiperVoice.load(str(onnx))
    syn = SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8)

    scripts = sorted(SCRIPTS.glob("*_en.txt"))[: (2 if quick else None)]
    for sp in scripts:
        for i, text in enumerate(_script_chunks(sp, variants)):
            suffix = "" if variants <= 1 else f"_p{i + 1}"
            dest = out / f"synthetic_{sp.stem}{suffix}.wav"
            with wave.open(str(dest), "wb") as wf:
                voice.synthesize_wav(text, wf, syn_config=syn)
            record(m, dest, tier="synthetic", language="en",
                   source=f"Piper TTS voice {PIPER_VOICE}", licence="MIT",
                   speaker_id=PIPER_VOICE,
                   upstream_path=f"{sp.relative_to(ASSETS)}#chunk{i + 1}/{variants}",
                   model=f"piper/{PIPER_VOICE}",
                   params={"length_scale": 1.0, "noise_scale": 0.667,
                           "noise_w_scale": 0.8},
                   note="unrelated speaker — expect high synthetic prob, LOW speaker sim")
            print(f"  [ok] {dest.name}")
    print(f"[ok] synthetic -> {out}")


# -------------------------------------------------------------------- cloned
def build_cloned(quick: bool, m: dict, variants: int = 1) -> None:
    out = ASSETS / "cloned"
    out.mkdir(parents=True, exist_ok=True)
    refs = sorted((ASSETS / "genuine").glob("enrolled_*.wav"))[:3]
    if not refs:
        print("Run --tier genuine first (need enrolled_*.wav reference clips).")
        return
    try:
        from TTS.api import TTS
    except Exception as exc:
        print(f"coqui-tts not importable ({exc}).")
        print("Run this tier with the TTS venv: tools/.venv-tts/bin/python "
              "scripts/build_demo_assets.py --tier cloned")
        return
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 on {dev} (first run downloads ~1.8 GB) ...")
    tts = TTS(XTTS_MODEL).to(dev)

    scripts = sorted(SCRIPTS.glob("*_en.txt"))[: (2 if quick else None)]
    for sp in scripts:
        for i, text in enumerate(_script_chunks(sp, variants)):
            suffix = "" if variants <= 1 else f"_p{i + 1}"
            dest = out / f"cloned_{sp.stem}{suffix}.wav"
            tts.tts_to_file(text=text, speaker_wav=[str(r) for r in refs],
                            language="en", file_path=str(dest))
            record(m, dest, tier="cloned", language="en",
                   source="Coqui XTTS-v2 voice clone of LibriSpeech spk 1272",
                   licence="Coqui Public Model License (non-commercial)",
                   speaker_id="1272 (cloned)",
                   upstream_path=f"{sp.relative_to(ASSETS)}#chunk{i + 1}/{variants}",
                   model=XTTS_MODEL,
                   params={"language": "en", "device": dev,
                           "speaker_wav": [r.name for r in refs]},
                   note="clone of the enrolled speaker — expect high synthetic prob "
                        "AND high speaker sim (the key demo case)")
            print(f"  [ok] {dest.name}")
    print(f"[ok] cloned -> {out}")
    print("     Key demo tier: HIGH synthetic prob AND HIGH speaker similarity.")


def _clean_script(path: Path) -> str:
    return " ".join(
        ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    )


def _script_chunks(path: Path, n: int) -> list[str]:
    """Split a script into n roughly equal chunks on sentence boundaries.

    Used to reach the >=10 clips per attack tier that the detector validation
    needs, without inventing new scenarios: same scripts, more audio.
    """
    text = _clean_script(path)
    if n <= 1:
        return [text]
    import re

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sents) < n:
        return [text]
    per = len(sents) / n
    chunks, i = [], 0.0
    for k in range(n):
        j = len(sents) if k == n - 1 else int(round((k + 1) * per))
        chunk = " ".join(sents[int(round(i)):j])
        if chunk:
            chunks.append(chunk)
        i = j
    return chunks or [text]


# --------------------------------------------------------------- provenance
PROV_HEADER = """# PROVENANCE — every demo clip's origin, licence, parameters

`scripts/build_demo_assets.py` regenerates this file from
`demo_assets/manifest.json`. Nothing is recorded from a person: the corpus is
built entirely from public datasets and open TTS models.

## Sources & licences

| Tier | Source | Licence | Notes |
|---|---|---|---|
| genuine (English) | LibriSpeech `dev-clean` | CC BY 4.0 | speaker 1272 designated "Rajesh Sharma — CFO"; 3 clips enroll the voice profile |
| genuine (Hindi) | Mozilla Common Voice — Hindi | CC0 1.0 | fetched manually (click-through) into `demo_assets/genuine/` as `hi_*.wav` |
| synthetic (non-cloned) | Piper TTS `en_US-lessac-medium` | MIT | unrelated speaker reading the scam scripts |
| cloned | Coqui XTTS-v2 | Coqui Public Model License — **non-commercial** | clones the enrolled speaker; prototype-only, see LIMITATIONS.md §5 |
| ASVspoof LA (control) | ASVspoof 2019 LA dev, via `Nemez1z/asvspoof-2019-la` mirror | ASVspoof EULA | 80-clip balanced subset in `demo_assets/_asvspoof_dev/`; used as the detector wrapper-correctness control (LIMITATIONS.md §3) |

## Model licences

| Model | Role | Licence |
|---|---|---|
| `nii-yamagishilab/wav2vec-large-anti-deepfake` | **primary** anti-spoofing (weight 0.30) | **CC-BY-NC-SA-4.0 — non-commercial** |
| `clovaai` AASIST | anti-spoofing baseline (weight 0.00) | MIT |
| `speechbrain/spkrec-ecapa-voxceleb` | speaker verification | Apache-2.0 |
| `silero-vad` | voice activity detection | MIT |
| `faster-whisper small` | ASR (Phase 3) | MIT |
| Piper `en_US-lessac-medium` | demo asset generation | MIT |
| Coqui XTTS-v2 | demo asset generation | Coqui CPML — non-commercial |

## Scam scripts

`demo_assets/scripts/*.txt` — written for this project, fictional, benign in
intent. They exist as text so they can be regenerated in any voice or language.

| File | Scenario |
|---|---|
| `ceo_transfer_en.txt` | CEO/CFO fund-transfer impersonation |
| `bank_otp_en.txt` | Bank official OTP solicitation |
| `govt_summons_en.txt` | Government official — fake summons/penalty |
| `family_emergency_en.txt` | Family-member emergency |
| `genuine_control_en.txt` | Genuine benign call — **must score LOW** |
"""


def prune_manifest(m: dict) -> int:
    """Drop entries whose file no longer exists, so PROVENANCE never lists a
    clip that is not in the corpus."""
    gone = [rel for rel in m.get("clips", {}) if not (ASSETS / rel).exists()]
    for rel in gone:
        del m["clips"][rel]
    if gone:
        print(f"[ok] pruned {len(gone)} manifest entries for deleted clips")
    return len(gone)


def write_provenance(m: dict) -> None:
    prune_manifest(m)
    save_manifest(m)
    clips = m.get("clips", {})
    lines = [PROV_HEADER, "\n## Generated clips\n"]
    if not clips:
        lines.append("_No clips built yet — run `build_demo_assets.py --all`._\n")
    for tier in ("genuine", "synthetic", "cloned"):
        rows = {k: v for k, v in sorted(clips.items()) if v.get("tier") == tier}
        if not rows:
            continue
        lines.append(f"\n### {tier} ({len(rows)} clips)\n")
        lines.append("| clip | dur (s) | sr | source | licence | model | params | sha256 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for rel, c in rows.items():
            params = c.get("params") or {}
            pstr = ", ".join(f"{k}={v}" for k, v in params.items()) or "—"
            lines.append(
                f"| `{rel}` | {c.get('duration_s', '—')} | {c.get('sample_rate', '—')} | "
                f"{c.get('source', '—')} | {c.get('licence', '—')} | "
                f"{c.get('model') or '—'} | {pstr} | `{c['sha256'][:16]}…` |"
            )
    lines.append(
        f"\n_Regenerated {dt.date.today().isoformat()} from `demo_assets/manifest.json` "
        f"({len(clips)} clips). Full SHA256 digests are in the manifest._\n"
    )
    PROV.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] PROVENANCE.md regenerated ({len(clips)} clips)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["genuine", "synthetic", "cloned"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variants", type=int, default=1,
                    help="split each script into N chunks -> N clips per script "
                         "(used to reach >=10 clips per attack tier)")
    ap.add_argument("--provenance", action="store_true",
                    help="regenerate PROVENANCE.md from the manifest and exit")
    args = ap.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    m = load_manifest()

    if args.provenance:
        write_provenance(m)
        return 0

    tiers = ["genuine", "synthetic", "cloned"] if args.all else (
        [args.tier] if args.tier else [])
    if not tiers:
        ap.error("pass --tier {genuine|synthetic|cloned}, --all, or --provenance")

    for tier in tiers:
        print(f"\n=== build tier: {tier} ===")
        {"genuine": build_genuine, "synthetic": build_synthetic,
         "cloned": build_cloned}[tier](args.quick, m, args.variants)
        save_manifest(m)

    write_provenance(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
