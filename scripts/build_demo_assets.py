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

Every clip's origin / licence / params is appended to PROVENANCE.md.

  python scripts/build_demo_assets.py --tier genuine   # LibriSpeech + CV Hindi
  python scripts/build_demo_assets.py --tier synthetic # needs piper-tts
  python scripts/build_demo_assets.py --tier cloned    # needs TTS (coqui), GPU
  python scripts/build_demo_assets.py --all
  python scripts/build_demo_assets.py --quick          # tiny subset for a smoke test
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "demo_assets"
SCRIPTS = ASSETS / "scripts"
PROV = REPO / "PROVENANCE.md"
sys.path.insert(0, str(REPO / "backend"))

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
ENROLLED_SPEAKER_LABEL = "Rajesh Sharma — CFO"


def prov(line: str) -> None:
    with open(PROV, "a") as f:
        f.write(f"- {dt.date.today().isoformat()}  {line}\n")


# --------------------------------------------------------------------- genuine
def build_genuine(quick: bool) -> None:
    out = ASSETS / "genuine"
    out.mkdir(parents=True, exist_ok=True)
    tar = ASSETS / "_cache" / "dev-clean.tar.gz"
    tar.parent.mkdir(parents=True, exist_ok=True)

    if not tar.exists():
        print(f"Downloading LibriSpeech dev-clean (~340 MB) -> {tar}")
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar)
    print("Extracting a subset ...")
    with tarfile.open(tar) as t:
        members = [m for m in t.getmembers() if m.name.endswith(".flac")]
        members.sort(key=lambda m: m.name)
        # group by speaker id: LibriSpeech/dev-clean/<spk>/<chap>/<utt>.flac
        by_spk: dict[str, list] = {}
        for m in members:
            spk = m.name.split("/")[2]
            by_spk.setdefault(spk, []).append(m)
        speakers = sorted(by_spk)
        enrolled_spk = speakers[0]
        take_speakers = speakers[: (2 if quick else 6)]
        import soundfile as sf

        for spk in take_speakers:
            clips = by_spk[spk][: (3 if quick else 8)]
            for m in clips:
                f = t.extractfile(m)
                data, sr = sf.read(f)
                tag = "enrolled" if spk == enrolled_spk else "genuine"
                name = f"{tag}_{spk}_{Path(m.name).stem}.wav"
                sf.write(out / name, data, sr)
        prov(f"genuine/  LibriSpeech dev-clean (CC BY 4.0), {len(take_speakers)} speakers. "
             f"Enrolled speaker id={enrolled_spk} labelled '{ENROLLED_SPEAKER_LABEL}'.")
    (out / "ENROLLED_SPEAKER.txt").write_text(
        f"{enrolled_spk}\n{ENROLLED_SPEAKER_LABEL}\n"
        "First 3 'enrolled_*' clips are used by scripts/seed.py for the voice profile.\n"
    )
    print(f"[ok] genuine set -> {out}  (enrolled speaker: {enrolled_spk})")
    print("     Hindi: fetch Common Voice hi manually (CC0) into demo_assets/genuine/ "
          "as hi_*.wav — see README (dataset requires a click-through).")


# ------------------------------------------------------------------- synthetic
def build_synthetic(quick: bool) -> None:
    out = ASSETS / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    try:
        from piper import PiperVoice  # type: ignore
    except Exception:
        print("piper-tts not installed. Install with:  pip install piper-tts")
        print("Then download a voice, e.g. en_US-lessac-medium, into models/piper/.")
        return
    voice_path = REPO / "models" / "piper" / "en_US-lessac-medium.onnx"
    if not voice_path.exists():
        print(f"Piper voice not found at {voice_path} — see README.")
        return
    import wave

    voice = PiperVoice.load(str(voice_path))
    scripts = sorted(SCRIPTS.glob("*_en.txt"))[: (2 if quick else None)]
    for sp in scripts:
        text = _clean_script(sp)
        wav_path = out / f"synthetic_{sp.stem}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            voice.synthesize(text, wf)
        prov(f"synthetic/{wav_path.name}  Piper TTS (MIT) voice=en_US-lessac-medium, "
             f"script={sp.name}, unrelated speaker.")
    print(f"[ok] synthetic set -> {out}")


# ---------------------------------------------------------------------- cloned
def build_cloned(quick: bool) -> None:
    out = ASSETS / "cloned"
    out.mkdir(parents=True, exist_ok=True)
    enrolled_dir = ASSETS / "genuine"
    refs = sorted(enrolled_dir.glob("enrolled_*.wav"))[:3]
    if not refs:
        print("Run --tier genuine first (need enrolled_*.wav reference clips).")
        return
    try:
        from TTS.api import TTS  # type: ignore
    except Exception:
        print("Coqui TTS not installed. Install with:  pip install coqui-tts")
        print("Model: tts_models/multilingual/multi-dataset/xtts_v2 (non-commercial licence).")
        return
    from voiceshield.config import get_settings

    dev = get_settings().device
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    scripts = sorted(SCRIPTS.glob("*_en.txt"))[: (2 if quick else None)]
    for sp in scripts:
        text = _clean_script(sp)
        wav_path = out / f"cloned_{sp.stem}.wav"
        tts.tts_to_file(text=text, speaker_wav=[str(r) for r in refs],
                        language="en", file_path=str(wav_path))
        prov(f"cloned/{wav_path.name}  Coqui XTTS-v2 (Coqui Public Model License, "
             f"non-commercial), cloned from enrolled speaker, script={sp.name}.")
    print(f"[ok] cloned set -> {out}")
    print("     This tier is the key demo: expect HIGH synthetic prob AND HIGH speaker similarity.")


def _clean_script(path: Path) -> str:
    return " ".join(
        ln.strip() for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["genuine", "synthetic", "cloned"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    if not PROV.exists():
        PROV.write_text("# PROVENANCE — every demo clip's origin, licence, parameters\n\n")

    tiers = ["genuine", "synthetic", "cloned"] if args.all else (
        [args.tier] if args.tier else [])
    if not tiers:
        ap.error("pass --tier {genuine|synthetic|cloned} or --all")
    for tier in tiers:
        print(f"\n=== build tier: {tier} ===")
        {"genuine": build_genuine, "synthetic": build_synthetic,
         "cloned": build_cloned}[tier](args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
