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
unit tests pass. — ✅ **PASSED** (2026-09-02)

**What works:**
- `backend/analyze.py clip.{wav,mp3,m4a,flac,ogg}` → full analysis JSON (or
  `--summary`). WAV/FLAC/OGG via `soundfile`; MP3/M4A/AAC via PyAV — **no system
  ffmpeg** (verified decoding transcoded `.m4a` and `.ogg`).
- `voiceshield/pipeline.py` — the single path (`analyze_audio`) that streaming
  and simulation will reuse (§14). Batch = 4 s / 1 s-hop windows → mean each
  detector over speech windows → fuse once → explainability + per-window
  timeline (raw + EMA) for the chart.
- `ingest/`: `audio.py` (load/resample), `vad.py` (energy + abs-floor gate,
  heuristic), `chunker.py` (sliding window), `telephony.py` (8 kHz + G.711
  µ-law + optional noise, deterministic).
- `fusion/scorer.py` — weighted linear blend, **weight redistribution** for
  unavailable layers (no fake values), bands LOW/MED/HIGH, per-component
  raw value / weight / effective weight / contribution points sorted by
  contribution, disclaimer string. `fusion/smoothing.py` — EMA (α=0.3).
- `store/repo.py` — voice-profile CRUD (embedding only), directory lookup,
  session insert.
- **33 unit tests pass** — detector known-input/known-output fixtures
  (synthetic heuristic, AASIST, ECAPA, prosody), fusion math + redistribution +
  bands, ingest (window math, VAD, telephony), pipeline gate.

**Gate evidence** (real LibriSpeech clips, `VOICESHIELD_OFFLINE=true`):

| clip | speaker vs enrolled profile (1272) | score | band | top contributor |
|---|---|---|---|---|
| `enrolled_1272_…-0002.wav` | same speaker | **15.1** | LOW | voice_authenticity 7.96 pts (speaker sim ≈ 1.0 → 0 suspicion) |
| `genuine_1462_…-0000.wav` | different speaker | **55.7** | MEDIUM | speaker_consistency 32.9 pts (cos ≈ 0.01 → 0.99 suspicion) |

Two different clips → two different, fully explainable scores. ✔

**Measured latencies** (per 4 s window, real audio, models warm):
- `synthetic_speech` (AASIST, CUDA): **~45 ms** (target <200 ms) ✔
- `speaker_consistency` (ECAPA, CUDA): **~27–37 ms** ✔
- `prosody_anomaly` (CPU): **~80–230 ms** (target <500 ms) ✔ — the Phase 0
  ~600 ms was cold-start on a zero clip; warm on real audio it is fine, no
  optimization needed.
- `load_audio`: 3–5 ms. `telephony degrade`: ~520 ms one-off per clip.

**Compromises / observations:**
- Without the speaker profile **and** context layers (Phase 3), absolute scores
  on lone genuine clips land MEDIUM (~55–60): AASIST alone reports spoof-prob
  ~0.3–0.65 on clean genuine LibriSpeech and, with only 2 acoustic layers live,
  carries 0.75 effective weight. The **discriminative** behaviour is correct
  (matches profile → LOW; different speaker → elevated); absolute calibration is
  a known gap the Evaluation page (Phase 6) will quantify. Not papered over.
- Telephony 8 kHz + µ-law pushed the same genuine clip 60 → **85 (HIGH)**:
  AASIST spoof-prob 0.65 → 0.96. This is the ASVspoof-2019-trained-model
  narrowband gap (§8) — shown, not hidden.
- VAD is an energy gate with a −45 dBFS absolute floor. Noisy room recordings
  above that floor will be treated as speech; a learned VAD (silero) is a
  drop-in upgrade, tracked for a later phase.
- Prosody baseline is estimated from only 6 clips / 2 speakers (`--quick` demo
  set). Wider baseline needs the full `build_demo_assets.py` run.

## Phase 1.5 — Corpus, AASIST diagnosis, VAD

**Status:** ⚠️ **6 of 7 gate items pass; item 2 fails on a measured finding, not a bug.**

**Task A — corpus.** 42 clips, all three tiers, manifest-driven:
32 genuine (LibriSpeech dev-clean, 8 speakers, spk 1272 = "Rajesh Sharma — CFO"),
5 synthetic (Piper `en_US-lessac-medium`), 5 cloned (XTTS-v2 of spk 1272).
`PROVENANCE.md` regenerates from `demo_assets/manifest.json` with source,
licence, model, generation params and SHA256 per clip. Prosody baseline rebuilt
on 32 clips / 8 speakers. TTS toolchain isolated in `tools/.venv-tts`.

**Task B — AASIST diagnosis: measured domain gap.** Polarity verified against
clovaai's eval convention (correct). Preprocessing verified (exact 64,600
samples, tiled not zero-padded, no normalisation, 16 kHz — contiguous vs tiled
differs by ≤0.03). Home turf clean: **ASVspoof2019 LA dev, n=80 balanced,
EER 0.00 %, accuracy 100 %**, bonafide median 0.0000 / spoof median 1.0000.
On our corpus it separates nothing. Left as-is and badged, per the diagnosis.
Full numbers in `LIMITATIONS.md §3`.

**Task C — Silero VAD** in place (MIT, weights bundled in the wheel, offline-safe);
energy gate retained behind `VOICESHIELD_VAD_BACKEND=energy`. VAD now runs once
per clip rather than once per window. Silero gives 7 clean utterance boundaries
where the energy gate produced 10 fragments on the same clip.

**Also — findings separated from score.** `fusion/findings.py` emits typed
findings (`speaker_mismatch`, `synthetic_speech`, `cloned_voice`,
`speaker_match`) plus a `voice_verdict` resolving the 2×2 of
synthetic × speaker-similarity. Carried on every analysis result for the Phase 2 UI.

**Blocker resolved in Phase 1.5b** (below). Measured end-to-end *before* that fix:

| clip | score | band | verdict |
|---|---|---|---|
| cloned XTTS of the CFO, reading the fraud script | **38.4** | **LOW** | CONSISTENT |
| Piper synthetic, unrelated voice | 52.4 | MEDIUM | SPEAKER_MISMATCH |
| genuine, different real speaker | 52.6 | MEDIUM | SPEAKER_MISMATCH |

The cloned attack scored **lowest of the three**, because the clone matches the
enrolled profile (ECAPA cosine +0.531) and AASIST is blind to XTTS (0.379,
indistinguishable from genuine 0.324–0.383).

---

## Phase 1.5b — SSL anti-spoofing detector

**Status:** ✅ **all gate items pass.**

**Selected:** `nii-yamagishilab/wav2vec-large-anti-deepfake` (AntiDeepfake,
arXiv 2506.21090) — wav2vec2-large SSL, 317.4M params, post-trained on
**18k h fake + 56k h real** multi-corpus speech. CC-BY-NC-SA-4.0 (non-commercial).
Chosen over higher-download alternatives specifically because its training set is
documented and is *not* ASVspoof2019 LA; candidates with `audiofolder` / `None
dataset` cards were rejected on that basis.

**Validated before adoption** (`scripts/validate_antideepfake.py`):

| model | condition | genuine (32) | Piper (15) | XTTS-cloned (15) | AUC Piper | **AUC cloned** |
|---|---|---|---|---|---|---|
| **AntiDeepfake** | clean | **0.0003** | 1.0000 | 1.0000 | 1.000 | **1.000** |
| **AntiDeepfake** | 8 kHz + µ-law | **0.0006** | 0.9999 | 0.9998 | 1.000 | **1.000** |
| AASIST | clean | 0.3314 | 0.5073 | 0.3502 | 0.675 | **0.554** |
| AASIST | 8 kHz + µ-law | 0.4963 | 0.9512 | 0.7257 | 0.935 | 0.688 |

- **Polarity:** head emits `<fake, real>`; we read `softmax(logits)[0]`.
  `FAKE_INDEX = 0` is a named constant, verified on labelled data.
- **Wrapper control:** the checkpoint is fairseq-named and `fairseq` will not
  install on Py3.11/torch 2.5, so we remap onto HF `Wav2Vec2Model`, guarded by
  `load_state_dict(strict=True)` **and** ASVspoof2019 LA dev: bonafide median
  0.0005 / spoof median 1.0000, **EER 0.00 %, acc 100 %**.
- **Confound ruled out:** tiers have different native rates (16/22.05/24 kHz).
  Genuine clips pushed through the Piper and XTTS resample paths stay at
  **0.00009** — the detector responds to synthesis, not resampling.
- **Telephony holds:** unlike AASIST, separation does not collapse under µ-law.

**Wired in:** AntiDeepfake carries `voice_authenticity` (0.30). **AASIST stays
loaded and badged at zero weight** (`Detector.contributes = False`) as a measured
baseline — never averaged or ensembled. The pipeline reads the scoring map from
the registry rather than a hardcoded dict, so a zero-weight detector runs and
reports but cannot reach fusion.

**VRAM measured at startup:** 730 MB allocated / **1512 MB reserved** on an
8.6 GB card, logged by `DetectorRegistry._log_vram()` with a warning above the
5 GB ceiling. Whisper `small` (Phase 3) will add ~1 GB. AntiDeepfake warm
latency **22.3 ms** per 4 s window (fp16, CUDA).

**End-to-end, after the fix:**

| clip | before | after |
|---|---|---|
| XTTS clone of the CFO | 38.4 LOW `CONSISTENT` | **60.9 MEDIUM `CLONED_VOICE`** |
| Piper synthetic | 52.4 MEDIUM `SPEAKER_MISMATCH` | 80.6 HIGH `SYNTHETIC_OTHER` |
| genuine, other speaker | 52.6 MEDIUM | 36.9 LOW `SPEAKER_MISMATCH` |
| genuine, enrolled speaker | 15.1 LOW | 5.4 LOW `CONSISTENT` |

47 tests pass; `make grep-honesty` clean.

---

## Phase 2 — WebSocket streaming + Live Analysis ✅

**Gate:** genuine LibriSpeech clip scores LOW, cloned attack clip scores higher,
with no controls touched — **PASSED**.

Measured over the WebSocket, identical settings, enrolled profile active:

| scenario | score | band | verdict |
|---|---|---|---|
| Genuine call — control | **5.4** | **LOW** | CONSISTENT |
| Genuine, different human | 36.9 | LOW | SPEAKER_MISMATCH |
| **CEO transfer (XTTS clone)** | **61.0** | MEDIUM | **CLONED_VOICE** |
| Bank OTP (XTTS clone) | 58.8 | MEDIUM | CLONED_VOICE |
| Govt summons (Piper TTS) | 78.9 | HIGH | SYNTHETIC_OTHER |

**Backend**
- `pipeline.WindowScorer` extracted as THE shared analysis core. Batch and
  streaming both drive it — `test_streaming_agrees_with_batch` asserts the two
  paths produce the same verdict and band on the same clip.
- `stream/session.py` — rolling buffer, 4 s windows on a 1 s hop, per-window
  VAD (batch runs VAD once over the whole clip; streaming cannot, since the
  clip does not exist yet — stated in the module docstring, asserted in tests).
  `close()` drops the buffer (§12). `set_context_components()` is the Phase 3 hook.
- `stream/scenarios.py` — 6 bundled scenarios, each backed by a real file,
  including **two genuine controls**. Each declares `expected` behaviour as
  prose, not an expected number.
- `WS /api/v1/stream`, `GET /api/v1/scenarios`. Server-side pacing to wall clock
  so the chart moves like a real call (`realtime: false` for fast tests).
- Window payloads carry a real 24-band FFT spectrum (dBFS, coherent-gain
  normalised) so the spectrogram is a measurement, not decoration.

**Frontend** (`/live`) — session header, risk gauge with server-supplied band
thresholds, streaming chart showing **raw dots and the EMA line together**,
RMS envelope + spectrogram, four detector cards with kind badges and their
fusion weight (AASIST visibly at 0.00 with its reason), latency strip against
the device budget, findings with evidence, and the explainability table.
Verified in a real browser (Playwright): no JS errors, gauge/verdict/chart all
render from streamed data.

**Measured latency** (per 4 s window, CUDA, from the live strip):
AntiDeepfake 27 ms · AASIST 17 ms · ECAPA 11 ms · prosody 21 ms —
**total 75 ms against the 200 ms budget**.

57 tests pass; `make grep-honesty` clean.

**Not yet done here:** context/ASR components still report unavailable and their
weight is redistributed (Phase 3); push-to-record and upload use the same
session object but have no UI yet (Phase 5).

---

## Phase 2.5 — fusion band floor ✅

**Problem.** The Phase 2 gate table showed the headline failure mode of PS26104:

| | score | band |
|---|---|---|
| XTTS clone of the enrolled CFO | 61.0 | MEDIUM |
| Piper TTS, unrelated voice | 78.9 | HIGH |

A clone of the target scoring *below* crude off-the-shelf TTS. Root cause is
structural, not weights: a successful clone is **supposed** to match the enrolled
profile, so `speaker_consistency` correctly reports low suspicion and correctly
contributes few points. A weighted linear blend is additive — it has no
interaction term for "synthetic **AND** matches the target".

**Fix — a named band-floor rule, not reweighting.** `policy/rules.py`:
`CLONED_VOICE` verdict → band floors at HIGH. The **score is left untouched**;
only the band is raised, and the rule's reason travels with it into the payload,
the UI and (later) the incident record. Thresholds and the floor band are
Settings values (`synthetic_high_threshold`, `speaker_match_threshold`,
`cloned_voice_band_floor`, `enable_band_floors`), not literals.

Reweighting was rejected: it would distort per-component semantics that are
individually correct, and being additive it would only reorder these particular
clips. An interaction-term multiplier was rejected as harder to justify to a
judge than one sentence of rule.

**Result** — applied in `WindowScorer.aggregate()`, the single point where batch
and streaming converge, so both paths get it:

| scenario | score | band from score | final band | rule |
|---|---|---|---|---|
| genuine control | 5.4 | LOW | **LOW** | — |
| genuine, other speaker | 36.6 | LOW | **LOW** | — |
| CEO transfer (clone) | 60.8 | MEDIUM | **HIGH** | `cloned_voice_floor` |
| Bank OTP (clone) | 59.0 | MEDIUM | **HIGH** | `cloned_voice_floor` |
| Govt summons (Piper) | 78.8 | HIGH | HIGH | — |

**All 15 XTTS clips** now land HIGH; all 15 Piper clips land HIGH on score alone;
genuine stays LOW with no floor applied.

Also fixed a real gap this surfaced: speaker similarity landing between the match
and mismatch thresholds discarded the synthetic signal and reported
`INDETERMINATE` (two Piper clips did). Such clips now report
`SYNTHETIC_SUSPECTED`.

66 tests pass, including `test_every_cloned_clip_lands_high` and
`test_both_cloned_scenarios_land_high_without_context` — the latter asserts the
context components are absent, so the floor is proven to work *before* Phase 3
can mask the problem.

---

## Phase 3 — Whisper worker + context engine ✅

**Gate:** every context flag traces to a transcript quote — **PASSED**, and
enforced in code: `SignalResult.__post_init__` raises if a signal has a non-zero
value with no matched span, so a quote-less flag cannot be constructed.

Measured over the WebSocket (identical settings, enrolled profile, per-scenario
directory record):

| scenario | score | band | verdict | quotes | caller_trust | floor |
|---|---|---|---|---|---|---|
| Genuine call — control | **4.0** | **LOW** | CONSISTENT | 0 | 0.08 | — |
| Genuine, different human | 23.6 | LOW | SPEAKER_MISMATCH | 0 | 0.15 | — |
| CEO transfer (clone) | 62.4 | **HIGH** | CLONED_VOICE | **5** | 0.95 | yes |
| Bank OTP (clone) | 45.5 | **HIGH** | CLONED_VOICE | 1 | 0.97 | yes |
| Govt summons (Piper) | 81.6 | HIGH | SYNTHETIC_OTHER | 0 | 0.98 | — |
| Family emergency (Piper) | 56.3 | MEDIUM | SYNTHETIC_OTHER | 0 | 0.95 | — |

Example traced flags on the cloned CEO call — each is a real Whisper transcript
span with an interpolated timestamp:

| t | signal | rule | quote |
|---|---|---|---|
| 0.3 s | authority_claim | exec_claim | “this is Rajesh Sharma, your CFO” |
| 10.6 s | urgency | pressure | “I'll be quick” |
| 14.7 s | transaction_intent | amount | “25 lakh” → **₹25,00,000** |
| 16.6 s | transaction_intent | account_change | “new vendor” |
| 18.4 s | urgency | deadline | “before end of day” |

**Two cadences, visibly independent (§4):** 18 acoustic windows at 1 Hz against
5 context frames as utterances closed, with quotes accumulating 1 → 2 → 4 → 5.
Whisper runs in a thread executor so it never blocks the 1 Hz loop.

**Built:** `asr/worker.py` (faster-whisper, loaded once, `Transcript` with char
offsets), `asr/segmenter.py` (utterances from the shared VAD pass),
`context/signals.py` (6 signal families, Indian amount parser, negation guard),
`context/engine.py` (components + quote resolution), plus the Context and
Transcript panels.

**Things this surfaced and fixed:**
- **“twenty-five lakh” parsed as ₹5,00,000** — the hyphen broke compound-number
  matching so only “five lakh” matched, understating the amount 5×. Replaced the
  word-number lookup with a real tens+units parser.
- **False positive on the genuine control**: “Nothing urgent” fired the urgency
  flag. Added a negation guard; the benign script now fires **0/6** signals, and
  a test pins it.
- **`caller_trust` was gated behind transcript arrival.** Directory metadata is
  known at call setup and has nothing to do with ASR, so on a clip too short to
  transcribe it would have stayed unavailable forever. Now seeded at session start.
- **The directory record was hardcoded to “Unknown Caller” for every session**,
  which charged the honest control 9.8 points for something never measured from
  its audio. Each scenario now declares which directory record its number
  resolves to; the genuine control resolves to the real Rajesh Sharma (0.08).

89 tests pass; `make grep-honesty` clean. Verified in a real browser, no JS errors.

**Not done here:** policy engine / mock approval (Phase 4), and the amount is
extracted but nothing is blocked on it yet.

---

## Phase 3 — segmentation decision (made before wiring)

Whisper wants utterance-shaped input, which would be a third granularity beside
batch-whole-clip and stream-per-window.

**Decision: the ASR worker gets utterance segmentation, but NOT its own
segmenter — it reads utterance spans from the same Silero VAD pass the acoustic
loop already runs.** Three granularities, two consumers, **one segmentation
source of truth.**

Why not reuse the 4 s / 1 s-hop window boundaries:
- They cut mid-word. Whisper hallucinates at truncated boundaries.
- 75 % overlap means the same words transcribe up to 4×. The context engine
  would count one "transfer 25 lakh" as four urgency/amount hits.
- §4 explicitly requires the decoupling: ASR runs "over VAD-segmented
  utterances (roughly every 3–6 s), not on every hop".
- §5 requires each context signal to carry its **matched transcript span**.
  Spans must map to real utterance times, not arbitrary window cuts.

Why not a second, independent segmenter: two VADs disagreeing about where speech
is would make "the acoustic layer scored this window but the transcript has no
words there" unexplainable. `vad.analyze()` already returns `segments`; the
streaming path currently uses only `overall_ratio`.

**Implementation shape.** `StreamSession` keeps a rolling utterance buffer and
runs the same VAD over the incoming stream (separate from the per-window ratio
call, because utterances cross window boundaries). When a segment closes and
exceeds `asr_min_segment_seconds`, or hits `asr_max_segment_seconds`, it is
queued to an async Whisper worker. The worker emits `transcript` frames at its
own cadence; the context engine consumes the rolling transcript and calls the
existing `StreamSession.set_context_components()` hook. Acoustic score keeps
ticking at 1 Hz throughout — the UI shows both cadences, as §4 requires.

---

## Phase 2 — original notes

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
