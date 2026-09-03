#!/usr/bin/env python3
"""Download and cache every model weight VoiceShield needs, into ./models.

Run once, online, during setup. After this the system runs fully offline.
Fails loudly (non-zero exit + explicit list) if anything is missing at the end.

  python scripts/fetch_models.py            # fetch everything
  python scripts/fetch_models.py --check    # verify cache only, download nothing
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
sys.path.insert(0, str(REPO / "backend"))

from voiceshield.config import get_settings  # noqa: E402

# AASIST (MIT, clovaai) — weights + the exact model definition so the graph
# always matches the checkpoint. Raw GitHub URLs.
AASIST_BASE = "https://raw.githubusercontent.com/clovaai/aasist/main"
AASIST_FILES = {
    "AASIST.pth": f"{AASIST_BASE}/models/weights/AASIST.pth",
    "models/AASIST.py": f"{AASIST_BASE}/models/AASIST.py",
    "config/AASIST.conf": f"{AASIST_BASE}/config/AASIST.conf",
}


def _get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "voiceshield-setup"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def fetch_ecapa(check: bool) -> bool:
    s = get_settings()
    target = s.ecapa_dir
    needed = ["embedding_model.ckpt", "hyperparams.yaml", "mean_var_norm_emb.ckpt",
              "classifier.ckpt", "label_encoder.txt"]
    if all((target / n).exists() for n in needed):
        _patch_ecapa_hyperparams(target)
        print(f"[ok] ECAPA cached at {target}")
        return True
    if check:
        print(f"[MISSING] ECAPA at {target}")
        return False
    print("Fetching SpeechBrain ECAPA-TDNN ...")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=s.ecapa_source, local_dir=str(target),
                      local_dir_use_symlinks=False)
    _patch_ecapa_hyperparams(target)
    ok = all((target / n).exists() for n in needed)
    print(f"[{'ok' if ok else 'FAIL'}] ECAPA -> {target}")
    return ok


def _patch_ecapa_hyperparams(target: Path) -> None:
    """Rewrite the pretrainer paths to bare filenames so SpeechBrain loads them
    from the local dir instead of trying to fetch from the HF Hub — required for
    fully-offline startup."""
    y = target / "hyperparams.yaml"
    txt = y.read_text(encoding="utf-8")
    patched = txt.replace("!ref <pretrained_path>/", "")
    if patched != txt:
        y.write_text(patched, encoding="utf-8")
        print("  patched hyperparams.yaml for offline load")


def fetch_whisper(check: bool) -> bool:
    s = get_settings()
    target = s.whisper_dir
    needed = ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"]
    if any((target / n).exists() for n in ("model.bin",)) and (target / "config.json").exists():
        print(f"[ok] faster-whisper '{s.whisper_model}' cached at {target}")
        return True
    if check:
        print(f"[MISSING] faster-whisper '{s.whisper_model}' at {target}")
        return False
    print(f"Fetching faster-whisper '{s.whisper_model}' ...")
    from huggingface_hub import snapshot_download

    repo = f"Systran/faster-whisper-{s.whisper_model}"
    snapshot_download(repo_id=repo, local_dir=str(target),
                      local_dir_use_symlinks=False)
    ok = (target / "model.bin").exists() and (target / "config.json").exists()
    print(f"[{'ok' if ok else 'FAIL'}] whisper -> {target}")
    return ok


def fetch_aasist(check: bool) -> bool:
    target = MODELS / "aasist"
    weights = target / "AASIST.pth"
    model_py = target / "models" / "AASIST.py"
    if weights.exists() and model_py.exists():
        print(f"[ok] AASIST cached at {target}")
        _ensure_pkg(target)
        return True
    if check:
        print(f"[MISSING] AASIST at {target} — anti-spoofing will fall back to DSP heuristic")
        return False
    print("Fetching pretrained AASIST (MIT, clovaai) ...")
    all_ok = True
    for rel, url in AASIST_FILES.items():
        try:
            _get(url, target / rel)
        except Exception as exc:
            print(f"  [warn] could not fetch {rel}: {exc}")
            all_ok = False
    _ensure_pkg(target)
    ok = weights.exists() and model_py.exists()
    if not ok:
        print("[warn] AASIST not fully available — HeuristicSyntheticDetector will be used. "
              "This is a documented, badged fallback (see LIMITATIONS.md).")
    else:
        print(f"[ok] AASIST -> {target}")
    return ok or all_ok  # non-fatal: heuristic fallback exists


def fetch_antideepfake(check: bool) -> bool:
    """AntiDeepfake wav2vec2-large — the PRIMARY anti-spoofing layer.

    nii-yamagishilab/wav2vec-large-anti-deepfake (arXiv 2506.21090), post-trained
    on 18k h fake + 56k h real multi-corpus speech. CC-BY-NC-SA-4.0.
    """
    target = MODELS / "antideepfake-wav2vec-large"
    weights = target / "model.safetensors"
    cfg = target / "config.json"
    if weights.exists() and cfg.exists():
        print(f"[ok] AntiDeepfake cached at {target}")
        return True
    if check:
        print(f"[MISSING] AntiDeepfake at {target} — "
              "voice_authenticity falls back to the DSP heuristic")
        return False
    print("Fetching AntiDeepfake wav2vec2-large (~1.2 GB, CC-BY-NC-SA-4.0) ...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("nii-yamagishilab/wav2vec-large-anti-deepfake",
                          local_dir=str(target))
        print(f"[ok] AntiDeepfake -> {target}")
        return True
    except Exception as exc:
        print(f"[warn] could not fetch AntiDeepfake: {exc}")
        return False


def check_silero() -> bool:
    """Silero VAD weights ship inside the `silero-vad` wheel — nothing to fetch,
    but verify they are importable so offline startup cannot surprise us."""
    try:
        from pathlib import Path as _P

        import silero_vad

        data = _P(silero_vad.__file__).parent / "data"
        ok = any(data.glob("*.jit")) or any(data.glob("*.onnx"))
        print(f"[{'ok' if ok else 'FAIL'}] Silero VAD bundled weights at {data}")
        return ok
    except Exception as exc:
        print(f"[MISSING] silero-vad not installed ({exc}) — "
              "VAD will fall back to the energy gate")
        return False


def _ensure_pkg(target: Path) -> None:
    (target / "models" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (target / "models" / "__init__.py").touch(exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify cache only")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"model cache: {MODELS}\n")

    results = {
        "ECAPA-TDNN (speaker, Tier A)": fetch_ecapa(args.check),
        "faster-whisper (ASR, Tier A)": fetch_whisper(args.check),
        "AntiDeepfake wav2vec2-large (anti-spoofing PRIMARY)": fetch_antideepfake(args.check),
        "AASIST (anti-spoofing baseline, zero fusion weight)": fetch_aasist(args.check),
        "Silero VAD (bundled in the wheel)": check_silero(),
    }
    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {'OK   ' if ok else 'MISS '} {name}")

    tier_a_ok = results["ECAPA-TDNN (speaker, Tier A)"] and results["faster-whisper (ASR, Tier A)"]
    if not tier_a_ok:
        print("\nFATAL: a Tier-A model is missing. VoiceShield will refuse to start clean.")
        return 1
    print("\nTier-A models present. VoiceShield can run offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
