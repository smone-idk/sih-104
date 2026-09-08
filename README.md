# VoiceShield

**Voice Integrity & Impersonation Risk Engine (Prototype)**

Real-time *estimated impersonation risk* for voice calls — built for Smart India
Hackathon 2026, Problem Statement **26104** (AICTE Cyber Security Cell):
*AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation
Attacks*.

**Design bar: every number on screen is computed from the audio that was actually
supplied.** No hardcoded score timelines, no fabricated accuracy claims. A build
check (`make grep-honesty`) fails if a scripted score appears anywhere in the
source.

> The prototype analyses only audio supplied through an **authorized channel** —
> a file you choose, a bundled demo clip, the enterprise-stream API, or an
> explicit push-to-record button. It never records, taps, or monitors calls, and
> never opens a background microphone. That claim is enforced in code, not
> promised: `getUserMedia` has exactly one call site in the entire frontend,
> inside the press handler.

---

## Table of contents

- [The problem, and why it is hard](#the-problem-and-why-it-is-hard)
- [How it works, in one page](#how-it-works-in-one-page)
- [The five decisions that matter](#the-five-decisions-that-matter)
- [Measured results](#measured-results)
- [What is real vs simulated](#what-is-real-vs-simulated)
- [Setup](#setup)
- [Run it](#run-it)
- [Project layout](#project-layout)
- [Status: what works, what is broken, what is not built](#status-what-works-what-is-broken-what-is-not-built)
- [Documentation map](#documentation-map)
- [Honesty notes](#honesty-notes)

---

## The problem, and why it is hard

A criminal records a few seconds of a company's CFO, clones the voice with
freely available software, phones the finance team, and says *"move twenty-five
lakh to this account before end of day, and don't loop in the team yet."*

The naive product is "a deepfake detector": one model, one number, done. That
does not work, for four reasons this project ran into and had to solve.

**1. Off-the-shelf anti-spoofing models are blind to modern cloning.**
The standard academic model (AASIST, trained on ASVspoof2019) scores **AUC 0.561**
against XTTS-v2 clones on our corpus — a coin flip. Not because the code is
wrong: the exact same wrapper scores **AUC 1.000, EER 0.00 %** on AASIST's own
2019 test set. The model is fine; its training data is five years stale. You
cannot discover this without building the measurement first.

**2. "Is it synthetic?" and "is it the right person?" are different questions,
and a successful clone inverts one of them.**
A good clone *matches* the enrolled speaker — that is the whole point of it. So a
speaker-verification layer correctly reports "yes, this is the CFO", which in any
additive scoring model *lowers* the risk. Measured before we fixed it: a cloned
CFO scored **61.0 (MEDIUM)** while crude text-to-speech in a stranger's voice
scored **78.9 (HIGH)**. The headline attack ranked below the easy case.

**3. Acoustics alone cannot tell you the caller is committing fraud.**
A clone reading a shopping list is not an attack. You need the *words* — the
urgency, the secrecy, the ₹25,00,000, the request to skip an approval. That means
speech-to-text, which brings its own failure mode: Whisper **hallucinates** on
cloned audio. We measured it inventing *"7 at the year right now and I only get
it"* on a benign clip, and the invented phrase *"right now"* fired a real urgency
flag on a call where nobody said anything urgent.

**4. A dashboard is not a control.**
Showing a red banner stops nothing. The prevention side has to actually refuse —
and refuse in a way a user with browser devtools cannot talk it out of.

---

## How it works, in one page

Audio is cut into **4-second windows advancing on a 1-second hop** (so each
window overlaps the last by 75 %). Four independent analyses run on every window
that contains speech:

| layer | what it asks | how |
|---|---|---|
| **Voice authenticity** | Was this made by a machine? | AntiDeepfake — a wav2vec2-large neural net post-trained on 74 000 hours of real + fake speech |
| **Speaker consistency** | Is this the person we enrolled? | ECAPA-TDNN turns a voice into a 192-number fingerprint; cosine-compared to the enrolled one |
| **Prosody anomaly** | Does this sound humanly natural? | Classical DSP — pitch, jitter, shimmer, speaking rate, pauses — scored against a human baseline |
| **Context** | What is the caller actually asking for? | Whisper transcribes; regex families detect urgency, secrecy, authority claims, transaction+amount, out-of-workflow, credential requests, threats |

Each returns a number in 0–1. They combine into one score out of 100 and a band
(**LOW** < 40, **MEDIUM** 40–70, **HIGH** > 70). If the band is HIGH, an API
endpoint that approves a money transfer returns **HTTP 403** and refuses.

```
  audio ──▶ decode 16 kHz mono ──▶ VAD (Silero) ──▶ 4 s windows / 1 s hop
                                        │                    │
                                        │                    ▼
                                        │          ┌──── 4 detectors ────┐
                                        │          │  authenticity       │
                                        │          │  speaker            │──▶ weighted
                                        │          │  prosody            │    fusion
                                        │          └─────────────────────┘      │
                                        ▼                                       │
                              utterance spans ──▶ Whisper ──▶ context ──────────┤
                              (3–6 s, same VAD)    + gate      signals          │
                                                                                ▼
                                                              score + band + explanation
                                                                                │
                                                                                ▼
                                                          policy ──▶ ALLOW / VERIFY / 403
```

**One analysis core, three entry points.** A CLI (`analyze.py`), a REST upload,
and a WebSocket stream all drive the same `WindowScorer`. There is deliberately
no second scoring path that could drift from the one being demonstrated.

**Two cadences.** The acoustic score ticks every second. Speech-to-text runs
separately over whole utterances (3–6 s) in a thread executor, so it never blocks
the 1 Hz loop. A typical run emits ~33 window updates against ~8 context updates.

**Speed.** Measured warm on CUDA: **112.8 ms of detector work per 4-second
window** (AntiDeepfake 33 ms · AASIST 25 ms · ECAPA 16 ms · prosody 39 ms) —
roughly 9× headroom against the 1-second hop. **722 MB VRAM allocated**
(1332 MB reserved) against a 5 GB budget.

---

## The five decisions that matter

These are the parts that are not obvious, each forced by a measurement. Full
reasoning with rejected alternatives is in
[`docs/ARCHITECTURE.md` §6](docs/ARCHITECTURE.md).

### 1. A layer that cannot answer says so — it never guesses

If there is no enrolled voice profile, the speaker layer reports **unavailable**
and its 0.20 weight is **redistributed** across the remaining layers, which then
sum to 1.0 again. It does not report 0.0.

*Why it matters:* scoring a missing layer as 0.0 would mean "this layer found no
risk" — a claim with no evidence behind it. Worse, it would make *the absence of
a profile look like safety*.

The same rule applies to context: a transcript with no fraud language leaves
those layers **not applicable**, not zero. The consequence is deliberate and
correct for a triage tool — **context can only raise risk, never lower it.
Absence of evidence is not evidence of safety.**

### 2. A rule layer, because the maths provably cannot express the clone case

A weighted **sum** has no interaction term. It cannot say "synthetic AND matching
the target is worse than either alone" — which is exactly what a voice clone is.

The fix is a named rule, not a fudged coefficient: when the verdict is
`CLONED_VOICE`, the **band** is floored at HIGH. **The score is never modified.**
The payload carries both the un-floored band and the rule's plain-English reason,
which the UI prints under the gauge. A judge can read the rule in one sentence
and disagree with it — much harder with a magic multiplier.

### 3. Evidence combines by noisy-OR, not by averaging

Behavioural signals were originally a weighted mean, which meant a call had to
fire nearly everything to score high. Measured result: a caller asking for a
one-time password outright produced a behavioural risk of **0.43**.

But asking for an OTP is not 25 % of a fraud — no legitimate caller ever asks for
one. These are independent evidence sources, any one of which can be nearly
conclusive, so they now combine as `1 − Π(1 − strength × value)`. That case moved
to **0.93**, and a benign call still measures exactly **0.00**.

### 4. The broken model is kept, running, at zero weight

AASIST stays loaded and displayed with a `PRETRAINED` badge and a fusion weight
of **0.00**, because "the 2019 model fails on 2023 TTS" is the strongest finding
in the project and it should be visible, not a claim in a footnote.

It is not merely multiplied by zero — the pipeline builds its scoring map from
the registry filtered on `contributes`, so a zero-weight detector is *physically
absent* from the fusion inputs.

### 5. Every context flag must carry the quote that produced it

`SignalResult.__post_init__` **raises** if a signal has a non-zero value with no
matched transcript span. A quote-less flag cannot be constructed. Every flag in
the UI shows the exact words, the timestamp, and the rule name.

That is necessary but, as we measured, **not sufficient** — Whisper hallucination
can put words in the transcript that were never spoken. Hence an ASR confidence
gate: segments below `avg_logprob −0.6` are transcribed and displayed (struck
through, labelled *"discarded — low ASR confidence"*) but never reach the context
engine. The threshold was chosen by sweeping against ground truth: at −0.6 it
removes the measured false positive at a cost of **zero** true signals; at −0.5
it starts costing real evidence.

---

## Measured results

Every number below has its sample size. Reproduce with the scripts in
[Documentation map](#documentation-map). Full detail:
[`docs/V1_VALIDATION.md`](docs/V1_VALIDATION.md).

Corpus: **152 clips** — 32 genuine (LibriSpeech, 8 speakers), 20 Piper TTS,
20 XTTS-v2 clones of the enrolled speaker, 80 ASVspoof2019 control clips.

### Anti-spoofing — the headline comparison

| model | vs Piper TTS (n=20) | vs **XTTS clones** (n=20) | on its own 2019 test set (n=80) |
|---|---|---|---|
| **AntiDeepfake** (74k h multi-corpus) | **AUC 1.000**, EER 0.00 % | **AUC 1.000**, EER 0.00 % | AUC 1.000 |
| AASIST (ASVspoof2019 only) | AUC 0.670 | **AUC 0.561**, EER 40.3 % | **AUC 1.000, EER 0.00 %** |

Genuine baseline n=32. That last column is the control that proves the wrapper is
correct and the failure is distributional, not a bug.

### Speaker verification

| group | n | mean cosine vs enrolled |
|---|---|---|
| genuine, enrolled speaker | 4 | **0.854** |
| genuine, other humans (impostors) | 28 | **0.016** |
| **XTTS clones of the enrolled speaker** | 20 | **0.534** |

Genuine vs impostor: **AUC 1.000, EER 0.00 %, FAR 0.00 %, FRR 0.00 %.**

Every clone scores more like the enrolled speaker than every real impostor does.
That is the attack succeeding at what it is designed to do — and precisely why
the two layers must stay separate.

### Context engine (n=72 clips with reference transcripts)

| family | precision | recall | F1 |
|---|---|---|---|
| urgency | 1.000 | 1.000 | **1.000** |
| secrecy | 1.000 | 1.000 | **1.000** |
| threat / coercion | 1.000 | 1.000 | **1.000** |
| authority claim | 1.000 | 0.875 | 0.933 |
| out of workflow | 1.000 | 0.750 | 0.857 |
| credential solicitation | 1.000 | 0.667 | 0.800 |
| transaction + amount | 1.000 | 0.667 | 0.800 |

**Zero false positives in every family.** All remaining error is recall, mostly
spelled-out numbers ("forty-nine thousand") the ASR wrote differently from the
reference. **Zero signals missing a quote, rule or timestamp.**

### Speech-to-text (post-gate, as the context engine receives it)

| tier | clips | WER | sub | del | ins |
|---|---|---|---|---|---|
| genuine | 32 | 5.1 % | 23 | 11 | 3 |
| Piper | 15 | 4.6 % | 12 | 5 | 3 |
| XTTS cloned | 15 | 6.7 % | 10 | 6 | 13 |

Before the confidence gate, cloned-tier WER was **23.4 %** with 88 hallucinated
insertions. The gate cut that to 6.7 % and 13, and the hallucination-driven false
positive now measures **zero**. Cost: +0.6 pp WER on genuine speech.

### End-to-end decisions (n=72)

| category | n | LOW | MEDIUM | HIGH |
|---|---|---|---|---|
| XTTS clones of the enrolled speaker | 20 | 0 | 0 | **20** |
| Piper TTS, unrelated voice | 20 | 0 | 1 | 19 |
| genuine, enrolled speaker | 4 | **4** | 0 | 0 |
| genuine, other humans | 28 | 27 | 1 | 0 |

Effective weights sum to 1.0000 ± 0.0001 on all 72 clips. A forged client payload
claiming low risk still gets **403**.

### Where it breaks — published, not hidden

| condition | genuine | Piper | **cloned** |
|---|---|---|---|
| clean | 0.000 | 1.000 | **1.000** |
| 20 dB SNR | 0.003 | 0.785 | **0.775** |
| **10 dB SNR** | 0.023 | 0.314 | **0.305** |

At **10 dB SNR — an ordinary phone-line condition — both attack tiers fall below
the 0.65 detection threshold.** The ranking survives (genuine stays two orders of
magnitude lower) but the verdict does not fire, so the clone reads as low risk.
This is a *calibration* failure, not a detection failure. The remedy —
SNR-conditioned thresholds — is named and not implemented.

**One further honesty note on the good numbers:** the AntiDeepfake model's
training data includes LibriTTS and Multilingual LibriSpeech, which are the same
LibriVox family as our genuine tier. So the *genuine* half of AUC 1.000 is
in-distribution and flattered. The attack tiers (XTTS, Piper) are **not** in its
training data, so detecting them is a real generalization result. The hard half
is real; the easy half is easy.

---

## What is real vs simulated

| Layer | Status |
|---|---|
| Synthetic-speech detection (**AntiDeepfake** wav2vec2-large) | **real** — pretrained, carries the `voice_authenticity` weight |
| Synthetic-speech baseline (AASIST) | **real, zero fusion weight** — kept and measured as the comparison |
| Speaker consistency (ECAPA-TDNN) | **real** — pretrained |
| Prosody anomaly | **real DSP**, badged `HEURISTIC` — not a trained model |
| Voice activity detection (Silero VAD) | **real** — pretrained; energy gate via `VOICESHIELD_VAD_BACKEND=energy` |
| Transcript (faster-whisper `small`) | **real** — over VAD-segmented utterances; WER 5.1 % genuine / 6.7 % cloned |
| Context signals | **real** — regex over the transcript, badged `HEURISTIC`; every flag carries quote + timestamp |
| Approval block (403 while HIGH) | **real control** — server-side state only; a forged payload cannot unlock it |
| Challenge–response verification | **real** — CSPRNG phrase, response audio re-analysed by the same pipeline |
| Telephony degradation (8 kHz + G.711 µ-law + noise) | **real** signal chain, deterministic under a seed |
| Latency + VRAM | **real** — measured per stage, displayed |
| Enterprise stream | **simulated transport** — a bundled WAV fed through the real pipeline over WebSocket |
| Callback / MFA / supervisor verification | **simulated** state machine, badged `SIMULATED` |
| Caller directory metadata | **demo data** — SQLite fixture, labelled `DEMO DATA` in the UI |

Full detail and every caveat: [`LIMITATIONS.md`](LIMITATIONS.md).

---

## Requirements

- **Python 3.11** — the ML stack has no 3.12+ wheels yet. `python3.11` on PATH.
- **Node 18+** for the frontend.
- GPU optional. **CUDA-first with a working CPU fallback** — one flag,
  `VOICESHIELD_DEVICE=cuda|cpu`. Developed on an RTX 4060 (8 GB).
- **No system `ffmpeg` needed** — PyAV and `soundfile` ship codecs in their
  wheels, so MP3/M4A/WebM all decode out of the box.
- ~4 GB disk for model weights. Models load **once at startup**, never per
  request, and stay under the 5 GB VRAM budget.

---

## Setup

Do this once, online.

```bash
make install                 # venv + backend deps (Python 3.11)
make install-torch-cuda      # or: make install-torch-cpu
make fetch-models            # ECAPA, faster-whisper, AASIST, AntiDeepfake -> ./models
make demo-assets             # LibriSpeech genuine set + enrolled speaker
make seed                    # SQLite schema + directory fixture + voice profile
make baseline                # prosody human-baseline from the genuine clips

cd frontend && npm install
```

### The full attack corpus (optional but this is the interesting part)

Generating Piper TTS and XTTS-v2 clones needs a **separate virtualenv**, because
`coqui-tts` pulls numpy 2.x and transformers 5.x, which would silently change
librosa's behaviour and invalidate the prosody baseline in the analysis venv.

```bash
python3.11 -m venv tools/.venv-tts
tools/.venv-tts/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
tools/.venv-tts/bin/pip install coqui-tts "transformers>=4.57,<5" piper-tts soundfile

backend/.venv/bin/python   scripts/build_demo_assets.py --tier genuine
tools/.venv-tts/bin/python scripts/build_demo_assets.py --tier synthetic --variants 3
COQUI_TOS_AGREED=1 tools/.venv-tts/bin/python scripts/build_demo_assets.py --tier cloned --variants 3
COQUI_TOS_AGREED=1 tools/.venv-tts/bin/python scripts/build_demo_assets.py --tier scenarios
```

Nobody records anything: the corpus is built from public datasets and open TTS
models. Every clip's source, licence, model, generation parameters and SHA256 is
recorded in [`PROVENANCE.md`](PROVENANCE.md).

### Verify it runs offline

```bash
VOICESHIELD_OFFLINE=true make inventory
```

Loads every model and prints the detector table with its device. **Exits
non-zero if a required model is missing** — it fails loudly rather than silently
degrading at the demo venue.

---

## Run it

```bash
make dev          # backend :8000 + frontend :5173
```

Then open **http://localhost:5173**.

### A five-minute tour

1. **Live Analysis** → pick *Genuine call — control* → **Start simulation**.
   Scores **5.8 LOW / CONSISTENT**, no context quotes. This is the control: a
   demo where a real call correctly scores LOW is more persuasive than five that
   score HIGH.
2. Switch to *CEO/CFO fund-transfer impersonation (voice clone)*. Scores
   **HIGH / CLONED_VOICE** with ~10 timestamped quotes — *"this is Rajesh
   Sharma, your CFO"* at 0.3 s, *"25 lakh"* normalised to **₹25,00,000**,
   *"before end of day"*. Watch the raw dots and the smoothed EMA line move
   together, and the context panel fill in on its own slower cadence.
3. Try *Benign call — but in a cloned voice*: the clone is detected, context
   correctly finds **nothing**, and the transcript shows segments struck through
   as *discarded — low ASR confidence*. That is the hallucination gate working.
4. **Approvals** → the ₹25,00,000 transfer is blocked. Press **"Try to bypass
   (send forged low-risk payload)"** — the server returns **HTTP 403** and
   reports its own band, ignoring everything the client claimed.
5. **Voice Profiles** → delete the enrolled profile, then re-run Live Analysis.
   Speaker consistency reads *"layer unavailable — no enrolled profile"* with
   effective weight **0.000**, and the other layers visibly take up the slack.
   `make seed` restores it.

> The Approvals page needs a session linked before it can block — run a Live
> Analysis scenario first, or the policy fails closed with `no_session`.

### Command line

```bash
cd backend
.venv/bin/python analyze.py ../demo_assets/scenarios/cloned_ceo_transfer_en_full.wav --summary
.venv/bin/python analyze.py clip.mp3                       # full JSON on stdout
.venv/bin/python analyze.py clip.wav --telephony --snr 15  # degradation chain
.venv/bin/python analyze.py clip.wav --no-profile          # ignore enrolled profile
```

### The approval gate, from curl

```bash
curl -X POST localhost:8000/api/v1/approvals/$AID/approve -d '{}'
#   HTTP 403  {"error":"verification_required","band":"HIGH","action":"ESCALATE"}

curl -X POST localhost:8000/api/v1/approvals/$AID/approve \
     -d '{"band":"LOW","verified":true,"override":true}'
#   HTTP 403  — band reported back is HIGH; the server never read those fields
```

---

## Project layout

```
backend/
  analyze.py            CLI entry point
  voiceshield/
    config.py           ONE Settings object — every threshold and weight lives
                        here, overridable by VOICESHIELD_* env vars
    pipeline.py         WindowScorer — THE shared analysis core (§14)
    inventory.py        startup detector table; fails loudly on missing models
    api/                FastAPI app + routes (stream, approvals, profiles)
    ingest/             decode · Silero VAD · 4 s/1 s chunker · telephony chain
    ml/registry.py      DetectorRegistry — kind badges, weights, `contributes`
    ml/detectors/       antideepfake · aasist(@0) · ecapa · prosody · dsp
    asr/                faster-whisper worker + utterance segmenter + gate
    context/            transcript -> signals with span-traced quotes
    fusion/             weighted blend, redistribution, findings, EMA
    policy/             band floors, approval decision, challenge-response
    store/              SQLite schema + repo
  tests/                15 files, 162 tests
frontend/src/
  pages/                LiveAnalysis · Upload · Approvals · Incidents · Profiles
  components/           RiskGauge · StreamChart · ContextPanel · PushToRecord …
scripts/
  fetch_models.py           download + cache all weights
  build_demo_assets.py      build genuine / synthetic / cloned / scenario tiers
  build_baseline.py         prosody human-baseline
  seed.py                   DB schema + directory fixture + enrolled profile
  validate_antideepfake.py  detector comparison + SNR sweep
  measure_wer.py            WER per tier + hallucination lexicon check
  build_eval_manifest.py    labelled evaluation manifest
  run_v1_eval.py            full validation metrics
  diagnose_asr_gate.py      ASR gate diagnostics, six-way transcript comparison
demo_assets/scripts/    the scam scripts as text — regenerable in any voice
docs/                   ARCHITECTURE · PHASES · V1_VALIDATION
```

---

## Status: what works, what is broken, what is not built

**162 tests pass, zero skipped.** `make grep-honesty` clean.

### Built and verified (Phases 0–5)

| Phase | Scope | Gate |
|---|---|---|
| 0 ✅ | Scaffold, model fetch, SQLite schema, startup inventory | Boots offline with an accurate detector list |
| 1 ✅ | Batch pipeline as a CLI | Two clips → two different explainable scores |
| 2 ✅ | WebSocket streaming + Live Analysis | Genuine LOW, cloned higher, no controls touched |
| 3 ✅ | Whisper worker + context engine | Every context flag traces to a transcript quote |
| 4 ✅ | Policy engine, API-level approval block | `POST /approve` → 403 while HIGH |
| 5 ✅ | Voice profiles, upload, push-to-record, telephony toggle | Deleting a profile visibly disables the speaker layer |

### Known bugs

- **The Context panel is blank on the Upload and push-to-record pages.**
  `Upload.tsx` reads `clean.context`, which the backend fills with *metadata*;
  the real context result sits at `clean.transcript.context`. The panel then
  claims the context layers are "unavailable and redistributed" while those same
  layers contributed 46.5 of 89.2 points on that response. **The streaming page
  is unaffected.** One-line fix, diagnosed in
  [`docs/V1_VALIDATION.md`](docs/V1_VALIDATION.md) §A.
- An unavailable *context* component reports `raw_value = 0.0` instead of `null`,
  so the explainability table prints `0.000` where the speaker layer correctly
  prints `—`. Cosmetic: the score is unaffected (its weight is 0.0).
- The ASR confidence gate drops **correct** transcriptions on short, unusual
  utterances — measured on 1 clip of 72, where `"illustration, Long Pepper."` was
  transcribed accurately at −0.627 and rejected.

### Not built

- **Phase 6** — the Evaluation page, Privacy Center page, API page and
  architecture diagram screens do not exist. The measurements they would display
  are real and reproducible today via the scripts above; they simply have no UI.
- `windows`, `transcript_segments` and `context_signals` exist in the SQLite
  schema but **nothing writes to them**. Consequence: an incident links to a
  session's score and band but cannot reconstruct *why* — no per-window timeline
  or transcript is persisted, so post-hoc audit of a blocked transfer is not
  possible.
- **Hindi and Punjabi are cut, not deferred.** We have no Hindi attack data, so
  there would be nothing to evaluate against. The pipeline would *run* on Hindi
  audio, but running is not evidence.
- No authentication on the API. The demonstrated property is that a client
  cannot forge its *risk state* — not that the endpoint is access-controlled.

---

## Documentation map

| document | what it is for |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the system works, for someone who did not write it. Explains the system four ways, then traces one clip literally through the code with a `file::function` reference at every step. Includes the full decision log. |
| [`LIMITATIONS.md`](LIMITATIONS.md) | Everything the system cannot do, and every measurement behind that. Written honestly on purpose — in judging this is an asset. |
| [`docs/V1_VALIDATION.md`](docs/V1_VALIDATION.md) | Independent validation pass: root-cause of the known bug, what is verified, what is broken, all metrics with sample sizes, and ranked fixes. |
| [`PROVENANCE.md`](PROVENANCE.md) | Every demo clip's source, licence, model, generation parameters and SHA256. |
| [`docs/PHASES.md`](docs/PHASES.md) | The build log — each phase's gate, evidence it passed, and compromises made. |

Reproduce every number in this README:

```bash
python scripts/build_eval_manifest.py     # labelled 152-clip manifest
python scripts/run_v1_eval.py             # detectors, speaker, context, fusion, SNR
python scripts/measure_wer.py             # WER + hallucination lexicon check
python scripts/validate_antideepfake.py --noise-sweep
python scripts/diagnose_asr_gate.py       # per-clip ASR gate behaviour
```

---

## Honesty notes

- The wording is always *"estimated impersonation risk"* and *"prototype
  detection engine"* — **never "confirmed deepfake"**.
- The fusion weights are **expert-elicited priors, not fitted on labelled fraud
  data**. The score is not a calibrated fraud probability. Every explainability
  panel says so.
- Accuracy figures always carry their sample size and describe a small demo set.
  They are **not** generalization claims.
- Layers that are hand-written rules rather than trained models are badged
  `HEURISTIC` in the UI. Simulated components are badged `SIMULATED`. No detector
  is allowed to misreport what it is — the badges are rendered straight from the
  registry.
- Where a model fails, we measured the failure and published it rather than
  quietly reweighting around it. The AASIST comparison and the 10 dB SNR collapse
  are both in this README on purpose.

### Licences worth knowing

- **AntiDeepfake** (`nii-yamagishilab/wav2vec-large-anti-deepfake`) —
  CC-BY-NC-SA-4.0, **non-commercial**.
- **XTTS-v2** (corpus generation only) — Coqui Public Model License,
  **non-commercial**.
- ECAPA-TDNN Apache-2.0 · faster-whisper MIT · Silero VAD MIT · AASIST MIT ·
  Piper MIT · LibriSpeech CC BY 4.0.

A shipped product could not use the two non-commercial components as they stand.
