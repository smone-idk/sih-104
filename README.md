# VoiceShield

**Voice Integrity & Impersonation Risk Engine (Prototype)**

AI-powered, real-time *estimated impersonation risk* for voice calls — built for
Smart India Hackathon 2026, Problem Statement **26104** (AICTE Cyber Security
Cell): *AI-Powered Real-Time Detection and Prevention of Voice Cloning
Impersonation Attacks*.

This is **prototype v1**. Design bar: **every number on screen is computed from
the audio that was actually supplied.** No hardcoded score timelines, no
fabricated accuracy claims. Layers that are heuristic rather than trained say so
in the UI with a `HEURISTIC` badge.

> The prototype analyzes only audio supplied through an **authorized channel**
> (file upload, bundled demo clips, the authorized enterprise-stream API, or an
> explicit push-to-record button). It never records, taps, or monitors calls or
> a background microphone. See the Privacy Center page.

---

## What is real vs simulated

| Layer | Status |
|---|---|
| Speaker consistency (ECAPA-TDNN, cosine vs enrolled profile) | **real** — pretrained model |
| Prosody anomaly (F0/jitter/shimmer/rate/pauses/flatness vs human baseline) | **real DSP**, badged `HEURISTIC` |
| Transcript (faster-whisper `small`) | **real** — pretrained ASR |
| Context signals (urgency/secrecy/authority/amount/out-of-workflow/PII) | **real** — rules+regex+span extraction from the transcript |
| Latency | **real** — measured per stage, displayed |
| Synthetic-speech detection (AASIST, ASVspoof2019-LA) | **real if weights fetched**, else a documented DSP `HEURISTIC` fallback |
| Enterprise stream | **simulated transport** — bundled WAV fed through the real pipeline over WebSocket |
| Caller metadata / directory | **demo data** — SQLite fixture, labelled in the UI |

Full detail: [`LIMITATIONS.md`](LIMITATIONS.md).

---

## Requirements

- **Python 3.11** (the ML stack has no 3.12+/3.14 wheels yet). `python3.11` must be on PATH.
- **Node 18+** for the frontend.
- Target GPU: RTX 4060 (8 GB). **CUDA-first with a working CPU fallback** — set
  `VOICESHIELD_DEVICE=cuda|cpu`. Both paths run.
- No system `ffmpeg` required: `av` (PyAV) and `soundfile` ship the codecs in
  their wheels. Install system `ffmpeg` only if you want the CLI tool.
- ~4 GB disk for model weights; keeps under ~5 GB VRAM at runtime. Models load
  **once at startup**, never per request.

---

## Setup (do this online, once)

```bash
make install                 # venv + backend deps  (Python 3.11)
make install-torch-cuda      # or: make install-torch-cpu
make fetch-models            # download + cache ECAPA, faster-whisper, AASIST -> ./models
make demo-assets             # LibriSpeech dev-clean genuine set + enrolled speaker
make seed                    # SQLite schema + enterprise directory + voice profile
make baseline                # prosody human-baseline from the genuine clips (optional but recommended)

cd frontend && npm install
```

Optional richer demo corpus (needs `piper-tts` / `coqui-tts`, GPU recommended):

```bash
backend/.venv/bin/python scripts/build_demo_assets.py --all
```

### Offline check

```bash
make fetch-models            # (once, online)
VOICESHIELD_OFFLINE=true make inventory
```

`make inventory` loads every model and prints the detector table with device
(`cuda`/`cpu`). It exits non-zero if a **Tier-A** model is missing — startup
fails loudly rather than silently degrading.

---

## Run

```bash
make dev                     # backend (:8000) + frontend (:5173)
# or separately:
make api
make frontend
```

Health: `curl localhost:8000/api/v1/health`
Inventory: `curl localhost:8000/api/v1/inventory`

---

## Project layout

```
backend/
  voiceshield/
    config.py          single Settings object — DEVICE is the one hardware flag
    inventory.py        startup detector inventory (Phase 0 gate)
    api/app.py          FastAPI; loads models once at startup, keeps them warm
    ingest/             chunking, resampling, VAD, telephony degradation (§8)
    ml/registry.py      DetectorRegistry — kind badges + weight redistribution
    ml/detectors/       synthetic (AASIST | DSP heuristic) · speaker · prosody
    asr/                faster-whisper worker (decoupled from the acoustic loop)
    context/            transcript -> behavioural + transactional signals (§5)
    fusion/             weighted score + EMA + calibration notes (§6)
    policy/             thresholds -> ALLOW / VERIFY / ESCALATE (§7)
    store/              SQLite: sessions, incidents, voice profiles, eval runs
scripts/
  fetch_models.py       download + cache all weights; fail loudly if missing
  build_demo_assets.py  build genuine / synthetic / cloned corpus locally (§13)
  build_baseline.py     prosody human-baseline from genuine clips
  seed.py               DB schema + directory fixture + enrolled profile
  run_eval.py           held-out eval -> JSON for the Evaluation page (§9)
demo_assets/scripts/    scam scripts as text — regenerate in any voice
frontend/               React + Vite + Tailwind + Recharts
```

---

## Build phases

Built in strict order; a phase does not start until the previous gate passes
(§17). See [`docs/PHASES.md`](docs/PHASES.md) for the gate status log.

| Phase | Scope | Gate |
|---|---|---|
| 0 | Scaffold, fetch_models, build_demo_assets, SQLite schema, startup inventory | Boots offline; prints accurate detector list with device |
| 1 | Batch pipeline as a CLI (`python analyze.py clip.wav` → JSON) | Two clips → two different explainable scores; detector tests pass |
| 2 | WebSocket streaming + Live Analysis screen | Genuine clip LOW, cloned clip higher, no controls touched |
| 3 | Whisper worker + context engine + explainability | Every context flag traces to a transcript quote |
| 4 | Policy engine, mock approval blocked at API, challenge-response, incident log | `curl POST /approve` → 403 while risk HIGH |
| 5 | Voice profiles, upload UI, push-to-record, telephony toggle | Removing a profile disables speaker layer, redistributes weight |
| 6 | Eval harness + page, Privacy Center, API page, architecture, docs | §16 checklist clean |

---

## Troubleshooting

- **`make inventory` says a Tier-A model is missing** — run `make fetch-models`
  with network access, then retry.
- **CUDA not used** — check `VOICESHIELD_DEVICE`, and
  `backend/.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`.
  Driver must support the cu124 wheel (CUDA 12.4 runtime is bundled).
- **`piper`/`TTS` import errors in build_demo_assets** — those tiers are
  optional; the `genuine` tier only needs LibriSpeech and always works.
- **Whisper is slow on first call** — warmup runs at startup; the first real
  request after a cold process still pays tokenizer init.

---

## Honesty notes

- "estimated impersonation risk", "prototype detection engine" — never
  "confirmed deepfake".
- The fusion weights are expert-elicited priors, **not fitted on labelled fraud
  data**. The score is not a calibrated fraud probability.
- Accuracy numbers appear **only** on the Evaluation page, always with the
  sample size, and only describe a small demo set.
