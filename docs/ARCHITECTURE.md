# VoiceShield — Architecture

**Voice Integrity & Impersonation Risk Engine (Prototype)** — SIH 2026, PS 26104.

This document explains how the system actually works, for someone who did not
write it. It assumes no knowledge of FastAPI, React, WebSockets, or machine
learning. Every concept is introduced before it is used.

**Every factual claim carries a `file::function` reference so you can check it.**
Where I reconstructed something rather than read it, it is marked
`UNVERIFIED — requires confirmation`. A list of everything I reconstructed is at
the end, along with the three claims I am least sure of.

This document does not repeat [`LIMITATIONS.md`](../LIMITATIONS.md) (what the
system cannot do, and every measurement behind that) or
[`PROVENANCE.md`](../PROVENANCE.md) (where every audio clip came from). Read
those separately; they are the honest core of the project.

---

## Table of contents

1. [What this is, four ways](#1-what-this-is-four-ways)
2. [Component map](#2-component-map)
3. [The trace — one clip, end to end](#3-the-trace--one-clip-end-to-end)
4. [The scoring model](#4-the-scoring-model)
5. [Data model](#5-data-model)
6. [Decision log](#6-decision-log)
7. [What to understand before changing anything](#7-what-to-understand-before-changing-anything)
8. [Where I reconstructed rather than read](#8-where-i-reconstructed-rather-than-read)

---

## 1. What this is, four ways

### To a 10-year-old

Computers can now copy a person's voice. Someone can record a few seconds of
your headmaster speaking, feed it to a program, and make that program say
anything in the headmaster's voice. Criminals use this to phone a company, sound
exactly like the boss, and say "send the money now".

VoiceShield listens to a phone call and tries to work out two different things.
First: *is this a real human voice, or one a computer made?* Second: *is this
actually the person it claims to be?* Then it listens to the **words** too — is
the caller rushing you, telling you to keep it secret, asking for a password?

It puts all of that together into one number from 0 to 100. If the number is
high, it stops the money transfer from going through until someone checks.

### To a first-year engineering student

VoiceShield is a Python backend and a React web page. You give it a piece of
audio. It chops the audio into overlapping 4-second slices and runs four
independent analyses on each slice:

- a neural network that estimates whether the audio was machine-generated;
- a neural network that turns a voice into a 192-number "fingerprint" and
  compares it to a fingerprint we enrolled earlier;
- classical signal processing that measures pitch, wobble, and pauses, and
  compares them to a human baseline;
- speech-to-text, whose output is scanned by regular expressions looking for
  fraud language ("transfer 25 lakh", "don't tell anyone", "read me the OTP").

Each analysis produces a number between 0 and 1. Those are combined into a
weighted score out of 100, plus a LOW/MEDIUM/HIGH band. If the band is HIGH, a
web endpoint that approves a money transfer returns HTTP 403 and refuses.

The interesting engineering is not any single model — they are all pretrained
and downloaded. It is in the **plumbing**: making sure a layer that cannot answer
says so rather than guessing, making sure every claim traces back to evidence,
and making sure the number on the screen was computed from the audio supplied.

### To a software engineer

FastAPI backend, React/Vite frontend, SQLite for state. One analysis core
(`pipeline.py::WindowScorer`) is driven by three entry points — a CLI
(`analyze.py`), a REST upload (`api/routes/profiles.py::analyze_upload`), and a
WebSocket stream (`api/routes/stream.py::stream`) — so there is no second
scoring path that could drift from the demonstrated one.

Models load once at process start into a registry
(`ml/registry.py::DetectorRegistry.build`) and stay warm. Each detector declares
a `kind` (`pretrained`/`heuristic`) and which fusion component it feeds; the UI
renders those badges directly from the registry, so a detector cannot misreport
what it is.

Fusion is a weighted linear blend with **weight redistribution**: a component
that cannot answer is dropped and its weight is reallocated proportionally,
never replaced with a neutral value (`fusion/scorer.py::fuse`). On top sits a
small rule layer (`policy/rules.py::apply_band_floors`) that can raise the
*band* without touching the *score*, because a linear blend provably cannot
express the interaction that matters most here.

"Real-time" means specifically: a 4.0 s analysis window advancing on a 1.0 s hop
(`config.py` `window_seconds`, `hop_seconds`), with a **measured mean of
112.8 ms of detector work per window on CUDA** — so the 1 Hz loop has roughly 9×
headroom. Speech-to-text runs on a separate, slower cadence over whole
utterances.

### To an SIH judge

Every number displayed is computed from the audio supplied. There is no scripted
score timeline; `make grep-honesty` fails the build if one appears.

The strongest result is a negative one we then fixed. The standard academic
anti-spoofing model (AASIST, trained on ASVspoof2019) scores **AUC 0.554** — a
coin flip — against XTTS-v2 voice clones on our corpus. We diagnosed that to the
training data rather than our code, by proving the wrapper achieves **0.00 % EER
on the model's own distribution**. We replaced it with an SSL model post-trained
on modern data (AUC 1.000), and kept AASIST loaded at **zero fusion weight** so
the comparison stays visible rather than being a claim in a document.

We also publish where it breaks: at **10 dB SNR** — an ordinary phone-line
condition — the clone detector's score collapses below threshold and a cloned
call reads as LOW risk. That is on the Evaluation page next to the working
numbers, with the remedy named.

The prevention side is a real control, not a banner: the approval endpoint
derives its refusal purely from server-side state, and a forged client payload
claiming low risk still gets 403 (`policy/engine.py::decide_approval`).

---

## 2. Component map

```
                        ┌────────────────────────────────┐
   AUTHORIZED INPUT     │  React UI  (frontend/src)      │
   ────────────────     │  Live · Upload · Approvals ·   │
   • bundled clip       │  Incidents · Profiles · System │
   • file upload        └───────────┬────────────────────┘
   • push-to-record                 │ HTTP + WebSocket
   • CLI                            ▼
                        ┌────────────────────────────────┐
                        │  FastAPI  (api/app.py)         │
                        │  loads models ONCE at startup  │
                        └───────────┬────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  ┌───────────┐            ┌─────────────────┐         ┌────────────────┐
  │  INGEST   │            │  ANALYSIS CORE  │         │  POLICY/STORE  │
  │ ingest/   │            │  pipeline.py    │         │ policy/ store/ │
  │           │            │                 │         │                │
  │ audio.py  │──audio──▶  │  WindowScorer   │──score─▶│ rules.py       │
  │ vad.py    │──speech─▶  │   .score()      │         │  band floor    │
  │ chunker.py│──windows▶  │   .aggregate()  │         │ engine.py      │
  │telephony.py           │        │        │         │  403 gate      │
  └───────────┘            └────────┼────────┘         │ repo.py→SQLite │
                                    │                   └────────────────┘
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
         ┌──────────────┐  ┌────────────────┐  ┌──────────────┐
         │  DETECTORS   │  │  ASR + CONTEXT │  │   FUSION     │
         │ ml/registry  │  │ asr/ context/  │  │ fusion/      │
         │              │  │                │  │              │
         │ AntiDeepfake │  │ worker.py      │  │ scorer.py    │
         │ AASIST (wt 0)│  │ segmenter.py   │  │ findings.py  │
         │ ECAPA        │  │ signals.py     │  │ smoothing.py │
         │ prosody      │  │ engine.py      │  │              │
         └──────────────┘  └────────────────┘  └──────────────┘
```

### Ingest — `backend/voiceshield/ingest/`

| | |
|---|---|
| **What** | Turns any audio file into 16 kHz mono float32, decides which parts are speech, and cuts it into overlapping windows. |
| **Why** | Every model downstream expects one specific format. Doing this once, in one place, means the CLI and the WebSocket cannot disagree about what "the audio" is. |
| **Files** | `audio.py::load_audio`, `vad.py::analyze`, `chunker.py::iter_windows`, `telephony.py::degrade` |
| **In → Out** | file path → `(np.ndarray float32, sample_rate)`; audio → `VadResult`; audio → `Window` objects |
| **On failure** | `audio.py::load_audio` raises `AudioLoadError` if neither backend can decode. `vad.py::analyze` catches a Silero failure and falls back to the energy gate, recording `fallback_reason`, which `pipeline.py::analyze_audio` surfaces as a user-visible warning. |

Two decoder backends: `audio.py::_try_soundfile` handles WAV/FLAC/OGG, and
`audio.py::_try_pyav` handles MP3/M4A/AAC. PyAV bundles its own codecs, so **no
system ffmpeg is required**.

### Detectors — `backend/voiceshield/ml/`

| | |
|---|---|
| **What** | Four independent analyses, each producing one number in 0–1 where higher = more suspicious. |
| **Why** | Separate layers can disagree, and their disagreement is informative. A voice that is *both* machine-made *and* matches the enrolled speaker is a clone — which you can only detect if the two questions are asked separately. |
| **Files** | `registry.py::DetectorRegistry`, `detectors/base.py::Detector`, `detectors/synthetic_antideepfake.py`, `detectors/synthetic_aasist.py`, `detectors/speaker_ecapa.py`, `detectors/prosody.py` |
| **In → Out** | `(window samples, sr, ctx)` → `DetectorResult(score, available, detail, latency_ms)` |
| **On failure** | `base.py::Detector.analyze` wraps every call in try/except and returns `available=False` with the error as a note. A detector cannot crash the pipeline; fusion then redistributes its weight. |

The registry is the single source of truth for *what exists and what it is*:

| detector | `name` | `kind` | `feeds` | weight |
|---|---|---|---|---|
| AntiDeepfake (wav2vec2-large SSL) | `synthetic_speech_ssl` | `pretrained` | `voice_authenticity` | 0.30 |
| AASIST | `synthetic_speech` | `pretrained` | *(none — see below)* | **0.00** |
| ECAPA-TDNN | `speaker_consistency` | `pretrained` | `speaker_consistency` | 0.20 |
| Prosody DSP | `prosody_anomaly` | `heuristic` | `prosody_anomaly` | 0.10 |

AASIST inherits `name`/`feeds` from `base.py::SyntheticSpeechDetector`, but
`registry.py::DetectorRegistry.build` sets `aasist.contributes = False`. Because
`pipeline.py::_scoring_map` builds its map from
`{d.name: d.feeds for d in registry.all() if d.feeds and d.contributes}`, a
zero-weight detector is **physically absent from the fusion inputs** — it is not
merely multiplied by zero. It still runs, and its score is recorded separately in
`WindowScorer.baseline_means`.

### ASR + context — `backend/voiceshield/asr/`, `backend/voiceshield/context/`

| | |
|---|---|
| **What** | Transcribes speech, then scans the transcript for fraud language, attaching the exact quote to every signal. |
| **Why** | Acoustics tell you *how* someone sounds; only the words tell you *what they are asking for*. And a flag a judge cannot trace to a quote is not evidence. |
| **Files** | `asr/worker.py::AsrWorker`, `asr/worker.py::Transcript`, `asr/segmenter.py::utterances_from_segments`, `context/signals.py::extract_all`, `context/engine.py::analyze_context` |
| **In → Out** | audio + VAD segments → `Transcript`; transcript + directory record → `ContextResult` with fusion components and timestamped quotes |
| **On failure** | `worker.py::AsrWorker.transcribe` catches exceptions and returns `[]`. `pipeline.py::analyze_audio` wraps the whole ASR+context block in try/except and appends a warning — acoustic analysis continues regardless. |

The rule that makes this trustworthy is enforced in code, not by convention:
`signals.py::SignalResult.__post_init__` **raises `ValueError` if a signal has a
non-zero value but no matched span.** A quote-less flag cannot be constructed.

### Fusion — `backend/voiceshield/fusion/`

| | |
|---|---|
| **What** | Combines six components into one 0–100 score and a band, with a full explanation. |
| **Why** | A single number is what an operator acts on; the explanation is what makes it checkable. |
| **Files** | `scorer.py::fuse`, `findings.py::derive_findings`, `smoothing.py::EMA` |
| **In → Out** | `{component: ComponentInput}` → `FusionResult(score, band, components[], redistributed, …)` |
| **On failure** | No failure mode: unavailable inputs are the normal case and are handled by redistribution. |

### Policy + store — `backend/voiceshield/policy/`, `backend/voiceshield/store/`

| | |
|---|---|
| **What** | Raises the band for cases the linear model cannot express; decides whether a money transfer may proceed; persists sessions, incidents, verifications, profiles. |
| **Why** | This is the part that makes it a security control rather than a dashboard. |
| **Files** | `policy/rules.py::apply_band_floors`, `policy/engine.py::decide_approval`, `policy/challenge.py::generate_challenge`, `store/repo.py`, `store/schema.sql` |
| **In → Out** | `(band, verdict)` → possibly-raised band + reasons; `(session row, verification row)` → `PolicyDecision` |
| **On failure** | `engine.py::decide_approval` **fails closed**: no session, incomplete analysis, or a pending/failed verification all return `allowed=False`, HTTP 403. |

### Frontend — `frontend/src/`

Six pages (`App.tsx`): Live Analysis, Upload, Approvals, Incidents, Voice
Profiles, System. The typed API boundary is `frontend/src/api.ts`. The one
security-sensitive component is `components/PushToRecord.tsx` — see §6.

---

## 3. The trace — one clip, end to end

This section follows `python analyze.py clip.wav` literally through the code.
Open each file as you read; every step names its function.

### Step 0 — the command

```bash
cd backend
.venv/bin/python analyze.py ../demo_assets/genuine/enrolled_1272_1272-128104-0002.wav
```

### Step 1 — `analyze.py::main`

Parses arguments (`argparse`), builds a `TelephonyConfig` from `--telephony` and
`--snr`, and builds a `ctx` dictionary. `--no-profile` sets
`ctx["enrolled_embedding"] = None`, which later suppresses profile lookup.

It then calls `pipeline.py::analyze_file(args.audio, source="cli", ctx=ctx,
telephony=tele)` and prints the returned object's `.as_dict()` as JSON.

> **What "ctx" is.** A plain dict carrying per-request context — the enrolled
> voice fingerprint, an optional directory record, optional weight overrides. It
> is passed down to every detector so they can use it or ignore it.

### Step 2 — `pipeline.py::analyze_file`

1. `load_audio(path)` → `(audio, sr)`. Timed into `load_ms`.
2. **Profile lookup.** If `"enrolled_embedding"` is *not already a key* in `ctx`,
   it calls `store/repo.py::get_profile(None)`, which returns the most recently
   created voice profile, and copies its `embedding` into `ctx`.
   Note the check is `not in ctx` — so `--no-profile`, which sets the key to
   `None`, correctly prevents the lookup.
3. Calls `analyze_audio(...)`.
4. Records `load_audio` latency, sets `res.source` to the file path.
5. If `settings.retain_audio` is false (the default), `del audio` drops the
   decoded samples from memory. **Nothing was ever written to disk.**

### Step 3 — `ingest/audio.py::load_audio`

- Tries `_try_soundfile` (WAV/FLAC/OGG). On any exception returns `(None, 0)`.
- Falls back to `_try_pyav` (MP3/M4A/AAC), decoding with PyAV's bundled codecs.
- If both fail → `AudioLoadError`.
- `_to_mono_float32` averages channels.
- If the sample rate differs from `settings.sample_rate` (16000), calls
  `resample()` → `resampy.resample(..., filter="kaiser_fast")`.
- `np.nan_to_num` scrubs NaNs; if the peak exceeds 1.0 the signal is divided by
  the peak.

**Out:** `(float32 array, 16000)`.

### Step 4 — `pipeline.py::analyze_audio`, first half

```python
t0 = time.perf_counter()
telephony = telephony or TelephonyConfig()
if telephony.enabled:
    audio, sr = degrade(audio, sr, telephony)
lat["degrade"] = ...
```

`ingest/telephony.py::degrade` is a **real** narrowband chain, not a filter
approximation: resample to 8 kHz → `_mu_law_encode`/`_mu_law_decode` (G.711,
8-bit companding) → optional additive Gaussian noise at a target SNR → resample
back to 16 kHz. It is deterministic given its `seed`.

### Step 5 — Voice activity detection, **once for the whole clip**

```python
vad = vad_mod.analyze(audio, sr)
sratio = vad.overall_ratio
```

`ingest/vad.py::analyze` dispatches on `settings.vad_backend`:

- `"silero"` (default) → `vad_silero.py::speech_timestamps`, a small pretrained
  neural VAD whose weights ship *inside the pip wheel* (so it works offline with
  nothing to download). Returns `(start, end)` spans, converted to
  `SpeechSegment` objects and a per-frame boolean mask via `_mask_from_segments`.
- `"energy"` → `_energy_analyze`, the original hand-written gate, kept behind the
  flag for comparison.
- If Silero raises, the exception is caught, the energy gate runs instead, and
  `fallback_reason` is set — which becomes a warning in the output.

The returned `VadResult` exposes `ratio_in(t0, t1)`, so each window can later ask
"what fraction of me is speech?" without re-running the model.

### Step 6 — Windowing

```python
scorer = WindowScorer(ctx, s)
for win in iter_windows(audio, sr):
    is_speech = vad.ratio_in(win.t_start, win.t_end) >= s.window_speech_ratio
    scorer.score(win.index, win.t_start, win.t_end, win.samples, sr, is_speech)
```

`ingest/chunker.py::iter_windows` yields `Window(index, t_start, t_end, samples)`
with `window_seconds=4.0` and `hop_seconds=1.0`, so consecutive windows overlap
by 75 %. A trailing remainder is zero-padded, but **only if the last full window
did not already cover it** — the `prev_end < len(audio)` guard.

`window_speech_ratio` defaults to 0.25: a window needs a quarter of its frames
marked speech to be scored at all.

*Our 12.485 s example yields 10 windows, all 10 classified as speech.*

### Step 7 — `pipeline.py::WindowScorer.score`, per window

1. Build a `WindowScore`, compute `_rms(samples)` and `_spectrum(samples, sr)`
   (a 24-band log-spaced FFT in dBFS, normalised by the Hann window's coherent
   gain — this feeds the UI's spectrogram and is a real measurement).
2. If `is_speech` is false: append and return. **No detector runs, no score.**
3. Otherwise, for every detector in `registry.available()`:
   - `det.analyze(samples, sr, ctx)` — the wrapper in
     `detectors/base.py::Detector.analyze` times it and catches exceptions.
   - `comp = self.scoring.get(det.name)`. **If `comp is None`** — which is how
     AASIST arrives, since `_scoring_map` excluded it — the score is appended to
     `baseline_means` and the loop `continue`s. It never becomes a fusion input.
   - Otherwise the score becomes a `ComponentInput` for its component.
4. `comp_inputs.update(self.ctx_inputs)` folds in the three context components.
5. `fuse(comp_inputs, ...)` → this window's raw score.
6. `self.ema.update(...)` → the smoothed score.

**What each detector actually does:**

- **AntiDeepfake** (`synthetic_antideepfake.py::_analyze`): mono, resample to
  16 kHz, `F.layer_norm(t, t.shape)` (zero-mean/unit-variance over the window,
  matching the model card exactly), forward pass, mean-pool over time, 2-way
  head, `softmax(...)[FAKE_INDEX=0]` → spoof probability.
- **ECAPA** (`speaker_ecapa.py`): if `ctx["enrolled_embedding"]` is absent,
  returns `available=False` with note "no enrolled profile". Otherwise embeds
  the window and computes cosine similarity, then converts it to a *suspicion*
  score: `score = clip((0.55 - cos) / 0.55, 0, 1)`. **High similarity → low
  score.** That inversion is the reason the band floor in §4 exists.
- **Prosody** (`prosody.py`): librosa + praat-parselmouth extract F0 mean/std,
  jitter, shimmer, speaking rate, pause ratio, energy CV, spectral flatness;
  each is turned into a z-score against a baseline built from the genuine
  corpus, clipped to 4, and `anomaly = clip(mean(z)/3, 0, 1)`.

### Step 8 — ASR and context, on a different cadence

After the window loop (`pipeline.py::analyze_audio`):

```python
run_asr = s.asr_enabled and ctx.get("asr", True) and bool(n_speech)
if run_asr or ctx.get("directory"):
    tr = transcribe_utterances(audio, sr, vad.segments, language=...)
    cres = analyze_context(tr, ctx.get("directory"))
    scorer.set_context(cres.components)
```

- `asr/segmenter.py::utterances_from_segments` merges the **VAD's own speech
  segments** into 3–6 s utterances (`asr_min_segment_seconds` /
  `asr_max_segment_seconds`), splitting anything much longer. It does not run a
  second VAD.
- `asr/worker.py::AsrWorker.transcribe` runs faster-whisper `small` per
  utterance with `vad_filter=False` (Silero already segmented it) and
  `condition_on_previous_text=False` (limits runaway hallucination).
- `asr/worker.py::Transcript.add` calls `gate_segment` on each segment. Below
  `asr_min_avg_logprob` (−0.6) or above `asr_max_no_speech_prob` (0.6), the
  segment is marked `confident=False`, **excluded from `Transcript.text`**, but
  kept in `Transcript.segments` so the UI can show it struck through.
- `context/signals.py::extract_all` runs seven lexicon families over
  `Transcript.text` — urgency, threat_coercion, secrecy, authority_claim,
  transaction_intent, out_of_workflow, credential_solicitation — plus
  `parse_amount` for Indian formats (₹25,00,000 / 25 lakh / 25L / twenty-five
  lakh → 2 500 000). `_negated()` suppresses a match preceded within 25
  characters by a negator, which is what stops "Nothing urgent" firing urgency.
- `context/engine.py::_locate` maps each match back to its utterance and
  interpolates a timestamp within it.
- `context/engine.py::analyze_context` produces the three context components.

### Step 9 — `WindowScorer.aggregate`

For each weighted detector, take the **mean of its per-window scores over speech
windows**, and build one `ComponentInput` from it. If a detector produced no
values, emit `available=False` with a reason — and specifically, if the detector
is `speaker_consistency` and `ctx["enrolled_embedding"]` is None, the note is
`"layer unavailable — no enrolled profile"`.

Then:

```python
final = fuse(agg_inputs, self.ctx.get("weights"), self.s)
findings, verdict = derive_findings(agg_inputs, self.s)
final.band, final.floors_applied = apply_band_floors(
    final.band_from_score, verdict, self.s)
```

This is **the single convergence point** — batch and streaming both end here, so
the band floor cannot apply to one path and not the other.

### Step 10 — Output

`AnalysisResult.as_dict()` serialises score, band, verdict, findings, the full
fusion breakdown, per-detector means, `baseline_detectors` (AASIST), latencies,
the transcript, the context quotes, warnings, and the per-window timeline.
`analyze.py::main` prints it.

**Measured on the example clip** (`enrolled_1272_1272-128104-0002.wav`,
12.485 s, 10/10 speech windows): `score 5.4, band LOW, verdict CONSISTENT,
vad_backend silero`.

> **Cold vs warm — an important distinction.** The CLI loads every model from
> scratch, so a one-shot run reports `total ≈ 17 850 ms`, of which ASR+context is
> ~3 183 ms and VAD ~309 ms. Those are **cold-start numbers and are not the
> streaming performance.** Measured on the warm server over the same clip, mean
> per-4 s-window detector cost is: AntiDeepfake 33.1 ms, AASIST 25.4 ms, ECAPA
> 15.6 ms, prosody 38.6 ms — **112.8 ms total against a 1000 ms hop.** Measured
> VRAM after all models load: **722 MB allocated / 1332 MB reserved** on an
> 8585 MB device, against the project's 5 GB budget
> (`registry.py::_log_vram`).

### The streaming path — only where it differs

`api/routes/stream.py::stream` handles `WS /api/v1/stream`.

> **What a WebSocket is.** Normal HTTP is one request, one response. A WebSocket
> keeps the connection open so the server can keep pushing messages. That is what
> lets the risk chart update every second.

The differences from batch, and why:

1. **VAD runs per window, not once.** `stream/session.py::_score_segment` calls
   `vad_mod.analyze(seg, self.sr)` on each 4 s window, because when audio is
   arriving there is no "whole clip" to analyse yet. Same backend, same
   threshold, less context. Batch uses `vad.ratio_in()` against a single pass.
2. **A rolling buffer replaces the chunker.** `session.py::feed` appends samples,
   emits every window that has become complete, and trims consumed audio.
   `session.py::flush` handles the zero-padded remainder.
3. **Two independent cadences.** The acoustic loop emits a `window` message every
   1 s hop. Separately, `session.py::pending_utterances` accumulates a *second*
   buffer and closes an utterance when it hits `asr_max_segment_seconds` or a
   trailing pause is detected. `routes/stream.py::drain_asr` then runs Whisper
   **in a thread executor** (`loop.run_in_executor`) so the 1 Hz loop is never
   blocked, and emits a `context` message. Measured on the current build, the
   `ceo_transfer_cloned` scenario produces **33 `window` messages against 8
   `context` messages** — the two cadences are visibly independent.
4. **Everything else is shared.** `StreamSession.__init__` constructs a
   `WindowScorer`; `_score_segment` calls `scorer.score(...)`; `final_payload`
   calls `scorer.aggregate()`. Same code, same fusion, same band floor.
5. **Session results are persisted server-side** via
   `repo.py::save_session_result`, which is what the approval gate later reads.

---

## 4. The scoring model

### The six components

`fusion/scorer.py::COMPONENTS`, weights from `config.py::Settings.fusion_weights`:

| component | weight | source |
|---|---|---|
| `voice_authenticity` | 0.30 | AntiDeepfake detector |
| `speaker_consistency` | 0.20 | ECAPA detector |
| `prosody_anomaly` | 0.10 | prosody detector |
| `caller_trust` | 0.10 | enterprise directory (demo data) |
| `transaction_context` | 0.20 | context engine |
| `behavioural_risk` | 0.10 | context engine |

### The blend, and why redistribution matters

`fusion/scorer.py::fuse`:

```
score = 100 × Σ (effective_weightᵢ × valueᵢ)   over AVAILABLE components only
effective_weightᵢ = weightᵢ / Σ(weights of available components)
```

An unavailable component contributes nothing **and its weight is removed from
the denominator**, so the remaining components' effective weights rise and still
sum to 1.0.

*Why this and not a default value.* If a missing layer were scored 0.0, the
system would be asserting "this layer found no risk", which is a claim it has no
evidence for. With no enrolled profile, treating `speaker_consistency = 0.0`
would actively *lower* the score — absence of a profile would look like safety.
Redistribution instead says "I cannot answer this; judge on what I can".

Bands come from `scorer.py::band_for`: LOW < 40, MEDIUM 40–70, HIGH > 70.

### Availability semantics for context

`context/engine.py::analyze_context` applies the same rule to itself. A
transcript-derived component that matched nothing reports `available=False` with
the note *"no transaction discussed — layer not applicable"* — **not 0.0**.

The consequence is deliberate: **context can only raise risk, never lower it.**
`caller_trust` is the one component that stays available with a low value, and
that is not an inconsistency — a directory record is *presence* of evidence about
the caller, not absence of evidence.

### Noisy-OR inside `behavioural_risk`

`context/engine.py::_behavioural_risk`:

```
risk = 1 − Π (1 − strengthᵢ × valueᵢ)
```

with per-signal *evidential strength* in `BEHAVIOURAL_STRENGTH`:
credential 0.85, threat_coercion 0.70, out_of_workflow 0.70, secrecy 0.60,
urgency 0.40, authority_claim 0.35.

*Why not a weighted mean.* A mean treats each signal as a fraction of one
quantity, so a call must fire nearly everything to score high. Measured
consequence: a scenario where the caller asks for a one-time password outright
produced `behavioural_risk = 0.43`. But asking for an OTP is not 25 % of a fraud
— on its own it is close to conclusive. Noisy-OR models them as independent
evidence sources: it is monotone (more evidence never lowers risk), saturates at
1, and reduces to a single signal's strength when only one fires. It moved that
case to 0.93 while leaving a benign call at exactly 0.00.

### EMA smoothing (streaming only)

`fusion/smoothing.py::EMA`, α = 0.3: `value = α·x + (1−α)·value`. Raw per-window
scores jitter on real audio. The UI plots **both** the raw dots and the smoothed
line (`frontend/src/components/StreamChart.tsx`) so the smoothing is visible
rather than hidden.

### The band floor — the part a linear model cannot do

`policy/rules.py::apply_band_floors` + `rules_for`.

A successful voice clone of the enrolled person *matches* the profile. ECAPA
therefore correctly reports high similarity → low suspicion → few points. The
component is right. But "sounds exactly like the target" is precisely what makes
a clone dangerous, and a weighted **sum** has no interaction term: it cannot say
"synthetic AND matching is worse than either alone".

Measured consequence before the fix: an XTTS clone of the CFO scored **61.0
MEDIUM** while crude TTS in a stranger's voice scored **78.9 HIGH** — the
headline attack ranked below the easy case.

The fix is a named rule, not a coefficient. When
`fusion/findings.py::derive_findings` returns verdict `CLONED_VOICE`, the band
is floored at HIGH. **The score is not modified** — `FusionResult` keeps both
`band_from_score` and the final `band`, plus `floors_applied` carrying the
rule's human-readable reason, which the UI displays under the gauge. A judge can
read the rule in one sentence and disagree with it; that is harder with a magic
multiplier.

### The verdict 2×2

`fusion/findings.py::derive_findings` crosses the two independent acoustic
layers, using thresholds from `config.py` (`synthetic_high_threshold` 0.65,
`speaker_match_threshold` 0.40, `speaker_mismatch_threshold` 0.60):

| | speaker matches | speaker mismatch |
|---|---|---|
| **synthetic high** | `CLONED_VOICE` | `SYNTHETIC_OTHER` |
| **synthetic low** | `CONSISTENT` | `SPEAKER_MISMATCH` |

Similarity between the two speaker thresholds is a deliberate dead band. A clip
landing there with high synthetic probability reports `SYNTHETIC_SUSPECTED`
rather than discarding what is known.

### The approval gate

`policy/engine.py::decide_approval` maps band → action (LOW → `ALLOW`, MEDIUM →
`VERIFY`, HIGH → `ESCALATE`) and decides `allowed`.

Its inputs are a **session row** and a **verification row**, both read from
SQLite by `api/routes/approvals.py::approve`. The request body is parsed by a
Pydantic model with `extra="ignore"`, so invented fields are accepted and then
never consulted. A client sending `{"band":"LOW","verified":true,"override":true}`
still receives 403, and the response reports the *server's* band.

---

## 5. Data model

`backend/voiceshield/store/schema.sql`, opened by `store/db.py::connect`.

| table | holds |
|---|---|
| `sessions` | one row per analysis: source, channel, scenario, duration, `final_score`, `final_band`, device, weights used. **The approval gate reads `final_band` from here.** |
| `windows` | **schema only — nothing writes to it.** See the note below. |
| `transcript_segments` | **schema only — nothing writes to it.** |
| `context_signals` | **schema only — nothing writes to it.** |
| `voice_profiles` | `display_name`, `role`, `embedding` (BLOB), `embedding_dim`, `n_enroll_clips`, `source_note` |
| `incidents` | band, score, action, summary, JSON payload — written on every approval attempt, verification and reset |
| `verifications` | method, `challenge_phrase`, result (`pending`/`passed`/`failed`), JSON detail |
| `approvals` | the mock transfer: description, amount, currency, state, linked `session_id`, `unlocked_by` |
| `directory_contacts` | enterprise directory **demo fixture** — known/verified flags, prior interactions, trust score |
| `eval_runs` | index of evaluation runs |

> **Three tables are defined but unpopulated.** `windows`,
> `transcript_segments` and `context_signals` exist in `schema.sql` but have no
> writer anywhere in the codebase — verified by grepping every `INSERT INTO` in
> the Python source, which targets exactly five tables: `approvals`,
> `incidents`, `sessions`, `verifications`, `voice_profiles`.
>
> This is a real gap, not a design choice. Per-window results, transcript
> segments and context signals currently live only in the API response and the
> browser; nothing persists them, so a session cannot be re-opened after the
> fact and the Incident Log can only link back to a session's *summary*
> (`final_score`, `final_band`), not its detail. Anyone extending this should
> write those three tables in `stream/session.py::final_payload` and
> `pipeline.py::analyze_audio`, or delete them from the schema so it stops
> implying a capability that does not exist.

### What is deliberately **not** stored

**No audio.** The whole schema contains exactly **one** BLOB column —
`voice_profiles.embedding`, a 192-float ECAPA vector. This is asserted by a test
that walks every table via `PRAGMA table_info` and fails if any other BLOB
appears (`tests/test_profiles.py::test_db_holds_no_audio_columns`).

Uploads are written to a temp file only so a decoder can read them, then deleted
in a `finally` block (`api/routes/profiles.py::_shred`), asserted by
`tests/test_profiles.py::test_no_audio_file_is_left_behind`. `retain_audio`
defaults to `False`.

**A voice embedding is still biometric data.** Not storing the waveform reduces
the exposure; it does not remove it. See `LIMITATIONS.md`.

Ordering note: `repo.py::list_incidents` orders by `created_at DESC, rowid DESC`
— `created_at` has one-second resolution, so rows landing in the same second
would otherwise sort arbitrarily, which is unacceptable in an audit log.

---

## 6. Decision log

*This section records reasoning that is **not** recoverable by reading the code.*

### 6.1 Anti-spoofing model: replaced AASIST

**Decided:** use `nii-yamagishilab/wav2vec-large-anti-deepfake` (AntiDeepfake,
arXiv 2506.21090) as the scoring anti-spoofing layer.

**Rejected:** AASIST (ASVspoof2019 LA); models whose cards list training data as
`audiofolder` or `None`; and newer architectures trained on the same 2019 LA data.

**Why:** AASIST measured **AUC 0.554** against XTTS clones on our corpus — a coin
flip. We rejected undocumented-training-data models on that basis alone, because
a model you cannot characterise cannot be defended in judging. We rejected newer
architectures on 2019 data because the failure is in the *training distribution*,
not the architecture.

**Evidence that forced it:** we first proved our *wrapper* was correct, by
scoring AASIST on its own distribution (ASVspoof2019 LA dev, balanced, n=80):
**0.00 % EER, 100 % accuracy**. Same code, near-random on our clips. That
isolates the cause to the data. AntiDeepfake then measured AUC 1.000 on the same
clips.

### 6.2 AASIST kept loaded at fusion weight 0.00

**Decided:** keep it running and displayed, contributing nothing.

**Rejected:** deleting it; or blending it in at a reduced weight.

**Why:** it is not dead code — it is the evidence for our strongest finding, "the
2019 model fails on 2023 TTS". Blending a detector we have measured as
near-random would reintroduce exactly the noise we diagnosed out.

**How it is enforced:** `registry.py::build` sets `contributes = False`;
`pipeline.py::_scoring_map` filters on that, so the detector is *absent from the
fusion inputs* rather than weighted by zero.

### 6.3 Band floor for `CLONED_VOICE`

**Decided:** a rule layer that raises the band, leaving the score untouched.

**Rejected:** reweighting the components; adding a multiplicative interaction
term.

**Why:** reweighting would distort component semantics that are individually
*correct*, and being additive would only reorder these particular clips. An
interaction coefficient is hard to justify to a judge; a one-sentence named rule
is not.

**Evidence:** XTTS clone of the CFO 61.0 MEDIUM vs Piper TTS in a stranger's
voice 78.9 HIGH — the headline attack below the easy case.

### 6.4 Noisy-OR inside `behavioural_risk`

**Decided:** combine behavioural signals as independent evidence.

**Rejected:** the weighted mean it replaced; also raising the six §6 fusion
weights at the same time.

**Why:** a mean models each signal as a fraction of one latent quantity. "Read me
the OTP" is not 25 % of a fraud — no legitimate caller asks for a one-time
password. We deliberately left the six fusion weights alone so we were not
tuning in two places at once and could attribute the change.

**Evidence:** OTP scenario 0.43 → 0.93; govt-summons scenario 66.5 MEDIUM →
71.3 HIGH on score alone; benign control unchanged at exactly 0.00.

### 6.5 Context components report UNAVAILABLE, not 0.0

**Decided:** a transcript-derived layer that matched nothing is unavailable.

**Rejected:** reporting 0.0.

**Why:** absence of evidence is not evidence of safety. The consequence — context
can only raise risk, never lower it — is the correct asymmetry for a triage tool.

**Evidence:** with 0.0-and-full-weight, turning context *on* moved a fraud call
from 58.8 down to 45.5, because three context components held 40 % of the weight
while contributing nothing and diluting the acoustic evidence.

### 6.6 ASR reuses the acoustic loop's Silero VAD spans

**Decided:** one segmentation source of truth; the ASR worker consumes VAD spans.

**Rejected:** a second VAD for ASR; reusing the 4 s window boundaries.

**Why:** window boundaries cut mid-word, and with 75 % overlap the same utterance
would be transcribed up to four times — the context engine would count one
"transfer 25 lakh" as four hits. A second VAD was rejected because two VADs
disagreeing would make "the acoustic layer scored this window but the transcript
has no words there" unexplainable.

### 6.7 ASR confidence gate at `avg_logprob ≥ −0.6`

**Decided:** gate segments before the context engine, threshold in Settings.

**Rejected:** leaving it ungated; a stricter threshold.

**Why and evidence:** swept against ground truth. At −0.6 the one measured
spurious flag disappears at a cost of **zero** true signals (10 of 96 segments
discarded). At −0.5 it starts costing real evidence for no further benefit.

**Caveat we did not hide:** Whisper assigns `avg_logprob` per decoding segment,
so the hallucinated span and the legitimate sentence after it shared an identical
−0.669. The gate works here because the innocuous neighbour carried no signals,
not because it separates hallucination from speech.

### 6.8 Hindi and Punjabi cut from v1 — not deferred

**Decided:** English only, stated as a cut.

**Why:** we have no Hindi attack data, so there would be nothing to evaluate
against. The pipeline would *run* on Hindi audio, but running is not evidence. A
language dropdown backed by an untested path is exactly the thing the project
brief warns against.

### 6.9 No classifier in the context engine

**Decided:** span-traced regex rules only.

**Rejected:** the "small classifier" the brief also allowed.

**Why:** the only labelled data we have is our own five scam scripts. A
classifier fitted to those would learn those scripts, emit confident
probabilities with no basis for generalisation, and could not trace its output to
a span. A rule that fires produces the exact quote, its timestamp and the rule
name; a judge can read the pattern and disagree with it.

### 6.10 Telephony toggle defaults to 20 dB SNR, not clean

**Decided:** the demo's default condition is degraded, not studio.

**Why:** a demo that only works on clean audio is not demonstrating the
deployment case. Real fraud calls arrive narrowband and compressed.

**Status: DECIDED BUT NOT YET IMPLEMENTED.** Verified in the current code — the
defaults are still clean: `Upload.tsx` has `useState<number | null>(null)` for
SNR and `LiveAnalysis.tsx` has `useState(false)` for the telephony toggle. This
is a Phase 6 task that has not been done at the time of writing. Do not cite
this section as describing current behaviour.

### 6.11 Push-to-record built to the privacy claim

**Decided:** `getUserMedia` appears at exactly one call site, inside the press
handler (`frontend/src/components/PushToRecord.tsx::startRecording`).

**Why:** the Privacy Center claims no background microphone access. That claim
should be a property of the code, not a promise. Tracks are stopped on stop, on
unmount, and on `pagehide`.

**Evidence:** verified in a real browser with **no microphone permission
granted**, instrumenting `getUserMedia` from an init script and visiting all six
pages — zero calls on every page.

---

## 7. What to understand before changing anything

1. **`backend/voiceshield/pipeline.py`** — `WindowScorer` is the analysis core.
   Batch and streaming both drive it. Change it and you change everything at
   once; that is the point, and the risk.
2. **`backend/voiceshield/fusion/scorer.py`** — `fuse()` and its redistribution
   rule. Almost every "why is the score that?" question ends here.
3. **`backend/voiceshield/ml/registry.py`** — decides which detectors exist,
   which carry weight, and what badge the UI shows. The `contributes` flag is
   how AASIST stays visible without influencing anything.
4. **`backend/voiceshield/config.py`** — every threshold and weight in one
   `Settings` object, overridable by `VOICESHIELD_*` environment variables.
   There are no tuning literals scattered through the code.
5. **`backend/voiceshield/policy/engine.py`** — the 403. Its security property is
   that it reads only server-side state; anything you add here must preserve
   that, and `tests/test_approvals.py` will tell you if it does not.

---

## 8. Where I reconstructed rather than read

To write this I read: `pipeline.py`, `ml/registry.py`, `ml/detectors/base.py`,
`analyze.py`, `ingest/audio.py` (signatures), `ingest/chunker.py`,
`ingest/vad.py` (public API), `store/schema.sql` (table list), the route
decorators across `api/`, `stream/session.py` (method list), plus targeted greps
for detector declarations and score formulas. I also ran the system to obtain the
latency, VRAM and score figures quoted.

**Reconstructed from memory of having written them, not re-read line by line for
this document:**

- `ingest/telephony.py::degrade` — the µ-law chain order (resample → companding →
  noise → resample back). I read its function list, not its body, this session.
- `context/signals.py` lexicon internals — I verified the family names and
  `parse_amount` behaviour by running them, but did not re-read every regex.
- `policy/challenge.py` — described only in passing; not re-read here.
- The frontend components beyond `PushToRecord.tsx`'s `getUserMedia` call site.
- `store/db.py::connect` — referenced but not re-read.

**I flagged three claims as low-confidence, then checked all three before
publishing. All three were wrong or stale, and are now corrected in place:**

1. **`windows` / `transcript_segments` / `context_signals` — CONFIRMED
   UNPOPULATED.** Grepping every `INSERT INTO` in the Python source shows writers
   for exactly five tables; these three have none. §5 now states this as a
   known gap rather than describing them as working storage. This was the error
   I most suspected, and it was real.
2. **The window-vs-context message counts were stale.** The figure quoted was
   taken before the full-length scenario clips landed. Re-measured on the
   current build, `ceo_transfer_cloned` produces **33 `window` messages against
   8 `context` messages** (plus one `session` and one `final`). §3 now carries
   the current numbers.
3. **The telephony 20 dB default was a decision I had not yet implemented.**
   Verified: `Upload.tsx` still defaults SNR to `null` and `LiveAnalysis.tsx`
   defaults the telephony toggle to `false`. §6.10 now says so explicitly
   instead of describing intent as behaviour.

**What remains genuinely unverified** is the material listed under
"Reconstructed" above — principally the body of `ingest/telephony.py::degrade`
and the individual regexes in `context/signals.py`. Both were exercised by
running them, but not re-read line by line for this document.

---

*Document generated 2026-09-05 against commit `652a781`. If the code has moved,
the `file::function` references are the thing to trust — re-verify anything that
does not resolve.*
