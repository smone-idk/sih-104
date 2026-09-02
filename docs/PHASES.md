# Build phase log

Each phase ends with a written report: what works, measured latencies, and any
compromise made. A phase does not begin until the previous gate passes (§17).

---

## Phase 0 — Scaffold & inventory

**Scope:** repo scaffold, `scripts/fetch_models.py`, `scripts/build_demo_assets.py`,
SQLite schema, startup detector inventory.

**Gate:** Boots offline; prints an accurate detector list with device (cuda/cpu).

**Status:** ✅ **PASSED** (2026-09-02)

Verified on the target box (RTX 4060 Laptop, i7 13th gen, WSL2):
- `VOICESHIELD_OFFLINE=true python -m voiceshield.inventory` → exit 0, all three
  detectors `loaded`, Tier-A OK, no network access.
- `uvicorn voiceshield.api.app:app` boots offline; startup hook loads models
  once and prints the inventory; `/health`, `/inventory`, `/config`,
  `/privacy/db-contents` all 200.
- `scripts/fetch_models.py` fetched ECAPA + faster-whisper-small + AASIST
  (weights + vendored model code); `--check` mode confirms cache.
- `scripts/seed.py` created the schema and seeded 6 directory contacts + the
  ₹25,00,000 mock approval.
- `pytest` — 4 passed.

| detector | kind | device | status |
|---|---|---|---|
| `synthetic_speech` (AASIST, ASVspoof2019-LA) | PRETRAINED | cuda | loaded |
| `speaker_consistency` (ECAPA-TDNN) | PRETRAINED | cuda | loaded |
| `prosody_anomaly` (DSP) | HEURISTIC | cpu | loaded |

**What works:**
- `backend/voiceshield/` package: `config.py` (single `DEVICE` flag),
  `DetectorRegistry` + three detectors (`synthetic_speech` [AASIST → DSP
  heuristic fallback], `speaker_consistency` [ECAPA], `prosody_anomaly` [DSP]),
  each declaring an honest `kind`.
- `python -m voiceshield.inventory` / `make inventory` — loads models once,
  prints name / kind / status / device, exits non-zero if a Tier-A detector is
  missing.
- FastAPI app: startup hook loads models once and prints the inventory;
  `/api/v1/health`, `/api/v1/inventory`, `/api/v1/config`,
  `/api/v1/privacy/db-contents`.
- SQLite schema (`store/schema.sql`) — sessions, windows, transcript_segments,
  context_signals, voice_profiles (embedding only), incidents, verifications,
  approvals, directory_contacts, eval_runs.
- `scripts/fetch_models.py` — ECAPA + faster-whisper + AASIST (weights + vendored
  model code) into `./models`; `--check` mode; fails loudly on missing Tier-A.
- `scripts/build_demo_assets.py` — genuine (LibriSpeech) / synthetic (Piper) /
  cloned (XTTS-v2) tiers; `scripts/seed.py`; `scripts/build_baseline.py`.
- `Makefile` (`make install`, `install-torch-cuda|cpu`, `fetch-models`,
  `demo-assets`, `seed`, `inventory`, `api`, `dev`, `test`, `grep-honesty`).

**Measured latencies** (cold single-shot on a 4 s zero-signal clip — *not*
representative, real numbers come with real audio in Phase 1):
- `synthetic_speech` (AASIST, CUDA): ~640 ms first call (CUDA kernel autotune),
  **~15 ms** once warm — comfortably under the 200 ms GPU target.
- `prosody_anomaly` (CPU): **~600 ms** — over the 500 ms CPU target. The
  `librosa` onset-strength + `peak_pick` + full-clip `spectral_flatness` are the
  cost. Flagged for optimization in Phase 1 (decimate for onset detection,
  shorter FFT hop).
- `speaker_consistency`: 0 ms in warmup because the dummy ctx has no enrolled
  embedding so it short-circuits to "layer unavailable". Real timing in Phase 1.

**Compromises:**
- Python 3.11 pinned (target machine's default `python3` is 3.14, which has no
  torch/speechbrain/librosa wheels). `python3.11` is present on the box.
- No system `ffmpeg` on the box and no sudo; relying on the `av` + `soundfile`
  wheels bundling codecs. To be verified when the ingest layer lands (Phase 1).
- AASIST is fetched from GitHub raw URLs; if the venue blocks that during setup
  the anti-spoofing layer runs as the badged DSP heuristic.

---

## Phase 1 — Batch pipeline (CLI)

**Gate:** Two different clips give two different, explainable scores; detector
unit tests pass. — _not started_

## Phase 2 — WebSocket streaming + Live Analysis

**Gate:** Genuine LibriSpeech clip scores LOW, cloned attack clip scores higher,
no controls touched. — _not started_

## Phase 3 — Whisper worker + context engine + explainability

**Gate:** Every context flag traces to a transcript quote. — _not started_

## Phase 4 — Policy, mock approval, challenge-response, incident log

**Gate:** `curl POST /approve` returns 403 while risk is HIGH. — _not started_

## Phase 5 — Voice profiles, upload UI, push-to-record, telephony toggle

**Gate:** Removing a profile disables the speaker layer and visibly
redistributes weight. — _not started_

## Phase 6 — Eval harness + page, Privacy Center, API page, architecture, docs

**Gate:** §16 acceptance checklist runs clean. — _not started_
