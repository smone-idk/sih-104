# LIMITATIONS

Written honestly, on purpose. In judging this is an asset, not a liability.
It is the reference for what VoiceShield v1 does and does not claim.

---

## 1. What is genuinely computed

| Layer | How | Model / method | Runs on |
|---|---|---|---|
| Speaker consistency | ECAPA-TDNN embedding, cosine similarity vs an enrolled profile | `speechbrain/spkrec-ecapa-voxceleb` (pretrained, Apache-2.0) | CPU or GPU |
| Prosody anomaly | F0 mean/std, jitter, shimmer, speaking rate, pause ratio, energy CV, spectral flatness; z-distance from a human baseline | librosa + praat-parselmouth DSP — **not a learned model** | CPU |
| Transcript | streaming ASR over VAD-segmented utterances | `faster-whisper small` (MIT), float16 on CUDA / int8 on CPU | CPU or GPU |
| Context signals | urgency, secrecy, authority claim, transaction + amount, out-of-workflow, PII solicitation — each with the matched transcript span | regex + keyword lexicons + small rules; **derived from the transcript, not presenter toggles** | CPU |
| Latency | `time.perf_counter()` around each stage | — | — |
| Fusion | weighted linear combination + EMA smoothing | weights are **expert-elicited priors** | — |

## 2. What is heuristic (badged `HEURISTIC` in the UI)

- **Prosody anomaly** is real DSP but not a trained detector. The baseline is
  estimated from a handful of genuine demo clips (`scripts/build_baseline.py`).
  Small baseline ⇒ wide variance ⇒ treat the absolute number with caution.
- **DSP anti-spoofing fallback** (`HeuristicSyntheticDetector`) — used only when
  the AASIST weights could not be fetched. Three cues: phase linearity,
  spectral-rolloff regularity, vocoder-band energy ratio. It is a *placeholder
  for a trained anti-spoofing model*, not a substitute for one.
- **Behavioural / caller-trust** components draw partly on the demo directory
  fixture (see §4).

## 3. What is a pretrained model used as-is (badged `PRETRAINED`)

- **AASIST** trained on **ASVspoof2019 LA**. The generalization gap is not a
  caveat we inherited from the literature — we measured it. See the boxed
  finding below.

> #### Measured finding — AASIST domain gap (Phase 1.5, 2026-09-02)
>
> **The wrapper is correct; the model does not transfer to our corpus.**
>
> Diagnosis followed polarity → preprocessing → home-turf, in that order:
>
> 1. **Polarity — correct.** `out_layer` emits 2 logits; clovaai's eval reads
>    `batch_out[:, 1]` as *bonafide*. Our wrapper reports `softmax(out)[0]` as
>    spoof-probability, which is the same convention.
> 2. **Preprocessing — correct.** Input is exactly 64,600 samples, 16 kHz mono,
>    tiled (not zero-padded) when short, with **no amplitude normalization** —
>    matching `data_utils.py` upstream. Feeding contiguous 64,600 real samples
>    instead of a tiled 64,000-sample window changes the score by **≤0.03**
>    across every offset tested, so the tile splice is not responsible.
> 3. **Home turf — clean.** On **ASVspoof2019 LA dev** (the model's own
>    distribution; balanced subset, n=40 bonafide / 40 spoof, original FLAC):
>
>    | class | median spoof-prob | mean | extremum |
>    |---|---|---|---|
>    | bonafide | **0.0000** | 0.0001 | max 0.0033 |
>    | spoof | **1.0000** | 0.9992 | min 0.9704 |
>
>    **EER 0.00 %, accuracy 100 % @ 0.5.** Perfect separation.
>
> On **our corpus** the same wrapper is near-random. Measured against the
> enrolled profile (speaker 1272) on all four tiers:
>
> | tier | n | AASIST spoof-prob | ECAPA cosine vs enrolled |
> |---|---|---|---|
> | genuine — spk 1272 (the enrolled speaker) | 4 | 0.383 | **+0.854** |
> | genuine — 7 other speakers | 28 | 0.324 | +0.016 |
> | synthetic — Piper TTS, unrelated voice | 5 | **0.529** | +0.156 |
> | cloned — XTTS-v2 of spk 1272 | 5 | **0.379** | **+0.531** |
>
> Three things fall out of that table:
>
> - **AASIST does not separate anything here.** Genuine 0.32–0.38 vs Piper 0.53
>   vs cloned 0.38 — the distributions overlap completely (genuine max 0.91 >
>   Piper min 0.27). Genuine speaker 1673 scores *higher* than most Piper clips.
>   Per-window scores on continuous genuine speech swing between 0.0001 and 0.87
>   on windows that overlap by 75 %.
> - **AASIST is blind to XTTS-v2 clones specifically** (0.379, statistically
>   indistinguishable from genuine 0.324–0.383). XTTS-v2 is a 2023 model; the
>   19 ASVspoof2019 attacks are 2019-era vocoders.
> - **ECAPA, by contrast, works exactly as intended** — and the clone fools it
>   on purpose: cosine **+0.531** against the enrolled speaker, versus +0.016
>   for other real humans. That is the attack succeeding at the thing it is
>   designed to do.
>
> **Conclusion: a measured domain gap, not a bug.** ASVspoof2019 LA bonafide is
> VCTK-derived studio speech; neither LibriSpeech's channel nor Piper's/XTTS's
> modern neural vocoders are represented. The detector is left **as-is and
> honestly badged** rather than reweighted — downweighting it would hide the
> finding. Not fixable by tuning; it needs a model trained on modern TTS.
>
> The DSP `HEURISTIC` fallback does not rescue this: on the same sets it scores
> genuine 0.025 vs Piper 0.002 — it separates in the *wrong direction*.
>
> **This gap has since been closed** by adding an SSL detector — see the
> side-by-side below. AASIST remains loaded and badged at **zero fusion
> weight** as a measured baseline.

### The fix, and the side-by-side that makes the point

The registry now runs **two** anti-spoofing detectors. Only the first carries
weight.

| | **AntiDeepfake** (primary) | **AASIST** (baseline) |
|---|---|---|
| model | `nii-yamagishilab/wav2vec-large-anti-deepfake` | `clovaai/aasist` AASIST.pth |
| architecture | wav2vec2-large SSL, 317.4M params + 2-way head | graph-attention, ~300k params |
| **training data** | **18k h fake + 56k h real, multi-corpus** post-training (arXiv [2506.21090](https://arxiv.org/abs/2506.21090)) | **ASVspoof2019 LA only** (19 attacks, 2019-era vocoders) |
| licence | CC-BY-NC-SA-4.0 (non-commercial) | MIT |
| fusion weight | **0.30** (`voice_authenticity`) | **0.00** — displayed, never fused |
| badge | `PRETRAINED` | `PRETRAINED` |

**Measured on our own corpus** (mean spoof-probability per clip, aggregated over
speech windows exactly as the pipeline does). AUC is `P(attack clip scores above
a genuine clip)` — threshold-free, so it cannot be tuned into looking good:

| model | condition | genuine (n=32) | Piper (n=15) | XTTS-cloned (n=15) | AUC vs Piper | **AUC vs cloned** |
|---|---|---|---|---|---|---|
| **AntiDeepfake** | clean | **0.0003** | **1.0000** | **1.0000** | **1.000** | **1.000** |
| **AntiDeepfake** | 8 kHz + µ-law | **0.0006** | 0.9999 | 0.9998 | **1.000** | **1.000** |
| AASIST | clean | 0.3314 | 0.5073 | 0.3502 | 0.675 | **0.554** |
| AASIST | 8 kHz + µ-law | 0.4963 | 0.9512 | 0.7257 | 0.935 | 0.688 |

AASIST's AUC of **0.554** against XTTS clones is a coin flip. Its apparently
better telephony numbers are an artefact: µ-law inflates *everything*, genuine
included (0.33 → 0.50), so the ranking improves without the detector gaining any
real ability.

**The finding, in one line:** a 2019-trained anti-spoofing model is blind to
2023 voice cloning (AUC 0.554), while the same task is solved by an SSL model
post-trained on modern multi-corpus data (AUC 1.000) — and the gap is in the
*training data*, not the architecture or our wrapper.

#### Why we believe the new numbers

The new model got no benefit of the doubt; it passed the same diagnosis AASIST
failed, plus a confound check:

1. **Polarity stated explicitly.** The head emits `<fake, real>`; we read
   `softmax(logits)[0]` as spoof-probability. `FAKE_INDEX = 0` is a named
   constant in the detector, not an implicit assumption.
2. **Wrapper control on labelled home turf.** The checkpoint ships fairseq-style
   parameter names and its recipe needs `fairseq`, which will not install on
   Python 3.11 / torch 2.5 — so we remap onto HF's `Wav2Vec2Model`. A silent
   mis-map would have produced a confident but meaningless detector. Guards:
   `load_state_dict(strict=True)` (any missing/unexpected key makes the detector
   report unavailable rather than score), and scoring the remapped model on the
   80 labelled ASVspoof2019 LA dev clips: **bonafide median 0.0005, spoof median
   1.0000, EER 0.00 %, accuracy 100 %.**
3. **Sample-rate confound ruled out.** Our tiers have different native rates
   (genuine 16 kHz, Piper 22.05 kHz, XTTS 24 kHz), so the detector could have
   been keying on resampling artefacts rather than synthesis. Pushing *genuine*
   clips through the exact Piper and XTTS resample paths leaves them at
   **0.00009** (vs 0.00012 as-is) — no effect. It is responding to synthesis.

Reproduce with `python scripts/validate_antideepfake.py`; results are written to
`data/eval/detector_validation.json` and are what the Phase 6 Evaluation page
reads.

#### How much of AUC 1.000 is in-distribution

We checked the AntiDeepfake paper's own corpus table (arXiv 2506.21090, Table I
and §IV-A). The two sides of our test are **not** symmetric:

| our tier | in the model's training data? |
|---|---|
| genuine (LibriSpeech dev-clean) | **effectively yes.** The bonafide set includes **LibriTTS**, **LibriTTS-R** and **Multilingual LibriSpeech (MLS)** — all LibriVox-derived, the same source corpus family and recording conditions as LibriSpeech. Speaker-level overlap is not verifiable from the paper. |
| Piper TTS | **no.** Piper is not mentioned anywhere in the paper. |
| XTTS-v2 clones | **no.** XTTS/Coqui are not mentioned anywhere in the paper. |

So the near-zero genuine scores (0.0003) are **flattered by in-distribution
data** — this model has seen a great deal of LibriVox-style audiobook speech and
is confident it is real. The attack side is genuinely out-of-distribution, so
detecting Piper and XTTS at AUC 1.000 *is* a real generalization result. The
honest summary: **the hard half of the result is real; the easy half is easy.**
A genuine-speech corpus outside the LibriVox family (spontaneous, telephony,
Indian-accent) is needed before the false-positive rate means anything.

#### Noise robustness — the ranking survives, the calibration does not

Our synthetic tiers are rendered TTS with no room tone, so the detector could
have been separating on *digital silence* rather than synthesis. Test: apply
identical additive white noise to **every** tier, equalising the noise floor.

| SNR | genuine | Piper | XTTS-cloned | AUC Piper | AUC cloned |
|---|---|---|---|---|---|
| clean | 0.0003 | 1.0000 | 1.0000 | 1.000 | 1.000 |
| 20 dB | 0.0025 | 0.8080 | 0.7947 | 1.000 | 1.000 |
| 10 dB | 0.0202 | 0.3607 | 0.3306 | 0.967 | 0.960 |
| 5 dB | 0.0708 | 0.3813 | 0.1874 | 0.902 | 0.823 |

**Good news:** the confound is ruled out. Separation does not collapse when the
noise floor is equalised — AUC stays 1.000 at 20 dB and above 0.82 even at 5 dB.
The detector responds to synthesis, not to silence.

> **⚠ Bad news, and this is the most operationally serious limitation we have
> measured. Absolute scores collapse with noise even though ranking holds.**
> Attack means fall from 1.00 (clean) to ~0.33 (10 dB) — **below our
> `synthetic_high_threshold` of 0.65**. The verdict therefore flips, and with it
> the `CLONED_VOICE` band floor. Measured end-to-end on the cloned CEO clip:
>
> | SNR | score | band | verdict | synthetic prob |
> |---|---|---|---|---|
> | clean | 61.0 | **HIGH** | CLONED_VOICE | 1.000 |
> | 20 dB | 58.9 | **HIGH** | CLONED_VOICE | 0.929 |
> | 10 dB | 25.9 | **LOW** | CONSISTENT | 0.201 |
> | 5 dB | 24.2 | **LOW** | CONSISTENT | 0.107 |
>
> **At 10 dB SNR the clone evades the system completely.** Real phone calls
> routinely sit at 10–20 dB. The ranking is still good at 10 dB (AUC 0.960), so
> this is a *calibration* failure, not a detection failure: a fixed threshold
> tuned on clean audio is wrong under noise. The fix is an SNR estimate feeding
> a noise-conditioned threshold (or score normalisation against a per-condition
> reference), which is real work and is **not** in the prototype. We have not
> lowered the threshold to hide this — doing so would raise false positives on
> clean genuine audio, trading a measured weakness for an unmeasured one.
>
> Reproduce: `python scripts/validate_antideepfake.py --noise-sweep`.
- **ECAPA-TDNN** trained on VoxCeleb (English, mostly celebrity interview
  audio). Accent and channel mismatch with Indian telephony speech will move
  the operating point; not compensated for in v1.
- **Whisper small** — word error rate rises on heavy Indian-accent code-switched
  Hinglish and on narrowband 8 kHz audio.

## 4. What is simulated (badged `SIMULATED` / labelled "demo data")

- **Enterprise stream** — a background task reads a bundled WAV and feeds it
  through the pipeline in chunks over a WebSocket, exactly as a live stream
  would arrive. **Transport is simulated; the analysis on top is real.**
- **Enterprise directory** (known contact, prior interaction count, verified
  identity) — seeded from a SQLite fixture in `scripts/seed.py`. Panel is
  labelled "Enterprise directory (demo data)".
- **Verification methods** other than challenge-response (registered callback,
  MFA, supervisor approval) — a clearly-labelled state machine, no real
  integration.

## 5. Demo corpus

- **Genuine**: LibriSpeech `dev-clean` (CC BY 4.0), Common Voice Hindi (CC0).
  Read speech, mostly clean 16 kHz — not conversational telephony.
- **Synthetic (non-cloned)**: Piper TTS (MIT), unrelated voice.
- **Cloned**: Coqui **XTTS-v2** — **Coqui Public Model License, non-commercial**.
  Acceptable for a prototype; recorded here explicitly. A production system
  cannot ship model outputs under that licence.
- Every clip's origin, licence and generation parameters are in
  [`PROVENANCE.md`](PROVENANCE.md).
- The cloned tier is the important case: a good clone should score **high on
  synthetic probability and high on speaker similarity** — that is the argument
  for keeping the layers separate.

## 6. What the evaluation numbers do and do not show

- EER / ROC-DET / confusion matrix / per-condition (clean vs 8 kHz telephony)
  are computed by `scripts/run_eval.py` on a **small held-out demo set**.
- Sample size is printed next to every metric. These are **not generalization
  claims**. They characterise this prototype on this tiny set, nothing more.
- No "detection rate" figure appears anywhere outside that page.

## 7. Scoring model

- The 0–100 risk score is a weighted linear blend of six components with
  **expert-elicited weights**, smoothed with an EMA (α ≈ 0.3). It is **not a
  calibrated fraud probability** and has **not been fitted on labelled fraud
  data**. Treat band (LOW / MEDIUM / HIGH) as a triage signal, not a verdict.
- If the speaker layer has no enrolled reference, its weight is **redistributed
  across the remaining layers**; no placeholder similarity is invented.

## 8. Privacy scope

- No audio file is written to disk by default (`RETAIN_AUDIO=false`, enforced in
  code). Voice profiles store the **ECAPA embedding only**, never audio.
- An embedding is still biometric data under the DPDP Act 2023. A compromised
  voice template cannot be reissued like a password — discussed in the Privacy
  Center.

## 9. Languages

- ASR: English, Hindi, Punjabi (Whisper). Context lexicons maintained per
  language incl. Hinglish code-switching.
- Language-agnostic acoustic features generalize only partially. Language-
  specific prosody models and Indian-accent spoof data are **future work**, not
  in v1. There is deliberately no non-functional ten-language dropdown.

## 10. Known engineering compromises

_(updated at the end of each build phase)_

- **Phase 0:**
  - Python is pinned to **3.11** — the target box's default `python3` is 3.14,
    which has no wheels for torch / speechbrain / librosa yet. `python3.11` is
    installed on the box; the Makefile and README use it explicitly.
  - No system `ffmpeg` and no sudo on the box. Audio decode relies on the `av`
    (PyAV) and `soundfile` wheels, which bundle their codecs. MP3/M4A decode is
    to be verified when the ingest layer lands (Phase 1).
  - The SpeechBrain ECAPA `hyperparams.yaml` ships pretrainer paths pointing at
    the HuggingFace repo id, which breaks fully-offline load.
    `scripts/fetch_models.py` rewrites them to bare local filenames after
    download (`_patch_ecapa_hyperparams`). Idempotent, re-applied on `--check`.
  - AASIST weights + model definition are pulled from `raw.githubusercontent.com`
    during setup. If the venue blocks GitHub during setup, the anti-spoofing
    layer runs as the badged `HeuristicSyntheticDetector` (DSP).
  - `prosody_anomaly` cold latency (~600 ms CPU on a 4 s window) is above the
    500 ms target — optimization deferred to Phase 1.
  - `AASIST.pth` is loaded with `torch.load(weights_only=False)` (it is
    clovaai's own MIT checkpoint, a plain state-dict pickle).
- **Phase 1:**
  - VAD is an energy gate (relative noise-floor + −45 dBFS absolute floor), not
    a learned model. Room noise above the floor counts as speech; silero-VAD is
    a drop-in upgrade tracked for a later phase.
  - Batch aggregation = **mean of each detector over speech windows**, then one
    fusion. Simple and explainable; a max/percentile aggregate would be more
    alarmist. Streaming (Phase 2) keeps the per-window EMA.
  - Absolute scores are only meaningful once the speaker profile **and** context
    layers are active. On a lone genuine clip with neither, AASIST's ~0.3–0.65
    spoof-prob on clean genuine speech (a known ASVspoof-2019 calibration gap)
    dominates the redistributed weight and lands the clip in MEDIUM. The
    Evaluation page (Phase 6) measures this; the demo always runs with a profile.
  - Telephony µ-law degradation inflates AASIST spoof-probability markedly
    (0.65 → 0.96 on one genuine clip) — the narrowband generalization gap. This
    is surfaced in the §8 before/after view, not corrected for.
- **Phase 1.5:**
  - **The AASIST anti-spoofing layer does not work on this corpus.** Diagnosed
    to a measured domain gap, not a bug (§3). Resolved in Phase 1.5b by adding
    the AntiDeepfake SSL detector; AASIST stays loaded at zero fusion weight.
- **Phase 1.5b:**
  - **AntiDeepfake is licensed CC-BY-NC-SA-4.0 — non-commercial.** Same posture
    as XTTS-v2: fine for a prototype, blocking for a shipped product. A
    commercial deployment needs either a licence from NII or an equivalently
    trained model. This is now the licence constraint on the *core* detector,
    not just on demo asset generation.
  - **The remap is ours, not upstream's.** We convert fairseq-style parameter
    names onto HF `Wav2Vec2Model` because `fairseq` will not install on
    Python 3.11 / torch 2.5. It is guarded by `strict=True` plus the ASVspoof
    control (§3), but it is still a reimplementation of someone else's loading
    path and should be re-verified if the checkpoint is ever updated.
  - **Perfect scores deserve suspicion.** AntiDeepfake returns ~0.0000 on
    genuine and ~1.0000 on both attack tiers. We ruled out the sample-rate
    confound (§3), but a corpus of 32 genuine / 30 attack clips from *two*
    generators is small and homogeneous. These are not generalization claims —
    they say this model separates *these* attacks, which AASIST could not.
  - **A cloned clip currently scores lower than a Piper clip** (60.9 vs 80.6)
    because the clone matches the enrolled speaker, so `speaker_consistency`
    contributes ~0 while Piper's mismatch adds points. Component-wise that is
    correct, and the `CLONED_VOICE` verdict plus the `cloned_voice` finding make
    the distinction explicit rather than hiding it in one number. We have
    deliberately **not** retuned fusion weights to reorder them; the context
    engine (Phase 3) is the layer that should lift a fraud-script clone.
  - **VRAM measured at 1.5 GB reserved** with AntiDeepfake (fp16) + AASIST +
    ECAPA loaded. Whisper `small` is not loaded until Phase 3 and will add
    roughly 1 GB; the 5 GB ceiling still holds but must be re-measured then.
  - The corpus grew to 15 Piper + 15 XTTS clips by splitting each of the 5 scam
    scripts into 3 chunks — **same 5 scenarios, more audio**, not 30 independent
    scenarios. Clip counts overstate scenario diversity.
- **Phase 2.5 (fusion band floor):**
  - **The score and the band can now disagree, by design.** A `CLONED_VOICE`
    verdict floors the band at HIGH while leaving the score untouched, so the UI
    can read "61.0 / HIGH". That is deliberate: the number stays an honest
    report of what the linear blend computed, and the band carries a named,
    stated rule. The alternative — reweighting until the clone outranks Piper —
    would have distorted component semantics that are individually correct.
  - **The floor is a rule, not a model.** It encodes an expert judgement ("a
    clone of the target is at least HIGH risk"), not anything fitted to data.
    It is exactly as unvalidated as the fusion weights are, and is listed here
    rather than presented as a detection capability.
  - **The floor inherits the detector's noise fragility.** It only fires when
    `CLONED_VOICE` fires, which needs synthetic probability ≥ 0.65 — see the
    noise table in §3. At 10 dB SNR the verdict does not fire, so the floor does
    not either. The floor fixes an *ordering* problem, not a *sensitivity* one.
  - **Speaker dead band.** Similarity between `speaker_match_threshold` (0.40)
    and `speaker_mismatch_threshold` (0.60) is deliberately "cannot say". Such
    clips now report `SYNTHETIC_SUSPECTED` when synthesis is detected, rather
    than discarding the synthetic signal as `INDETERMINATE` (which two Piper
    clips previously did).
- **Phase 3 (ASR + context engine):**
  - **The context engine is `HEURISTIC` — regex and rules, not a classifier.**
    It will miss paraphrases it has no pattern for. §5 asked for "rule + regex +
    small classifier"; only the rule/regex half is built. Coverage on our five
    scam scripts is 6/6 signals on the CEO script, 5/6 on the govt summons, 4/6
    on the bank OTP, 3/6 on the family emergency, and **0/6 on the benign
    control** — but those are the scripts the lexicons were written against, so
    that is a sanity check, not a generalisation measurement.
  - **Whisper mis-transcribes the XTTS clones.** Cloned audio produces garbled
    spans ("your CFO Dharuddin Naimukh"). The fraud-bearing content survives, but
    a context signal can be missed because ASR mangled the phrase carrying it.
    Word error rate is not measured — we have no reference transcripts.
  - **Quote timestamps are interpolated within the utterance**, not word-level
    forced alignment. Good enough to point a judge at the right moment; not
    evidence-grade timing.
  - **Negation handling is a 25-character lookback** for a small negator list.
    It fixes "Nothing urgent" but will not catch distant or complex negation
    ("I would not say this is urgent, but...").
  - **The enterprise directory is demo data and the mapping is authored.** Each
    scenario declares which directory record its number resolves to. That is
    legitimate call-setup metadata (§2 Tier C) and is labelled DEMO DATA in the
    UI — but it is authored, not measured, and it does move the score
    (`caller_trust`, weight 0.10). On the genuine control it is 0.08; on the
    attack scenarios ~0.95.
  - **Short clips get little context.** Utterances shorter than
    `asr_min_segment_seconds` produce no transcript, so a ~6 s clip can score on
    acoustics alone (the family-emergency scenario lands MEDIUM with 0 quotes).
  - **Measured: context can push a fraud call's score DOWN.** Bank OTP (clone)
    scores 58.8 with context off and **45.5** with it on. The cause is weight
    redistribution, not a weak signal averaging things down:

    | component | context OFF | context ON |
    |---|---|---|
    | voice_authenticity | eff. weight 0.500 → **50.0 pts** | eff. weight 0.300 → **30.0 pts** |
    | caller_trust | unavailable (weight redistributed) | 0.970 → +9.7 pts |
    | behavioural_risk | unavailable | 0.050 → +0.5 pts |
    | transaction_context | unavailable | **0.000 → +0.0 pts, holding 0.20 weight** |

    With three context components unavailable, the acoustic layers renormalise
    to 1.0 and voice_authenticity carries 0.50. Turning context on dilutes it to
    0.30 — a 20-point loss — while context returns only 10.2. The specific
    offender is `transaction_context`: **20 % of the score budget assigned to a
    component that found nothing.** Fusion currently cannot distinguish "this
    layer looked and found nothing" from "this layer found an absence of risk",
    and treats an available-but-zero component as evidence of safety.
    The `CLONED_VOICE` band floor masks the outcome here, but a judge reading
    the explainability table on an OTP scam sees context contributing nothing
    while diluting real evidence. **Not yet fixed — diagnosed only.**
  - **Two coverage gaps compound that.** (a) Scenarios point at `_p1` chunks,
    which hold only the call opening; the fraud payload (the OTP ask, the money
    demand) is in `_p2`/`_p3`. Per chunk only 1–2 signals fire, against 4–6 on
    the full script. (b) There is **no government-impersonation vocabulary at
    all**: *summons, penalty, case number, warrant, arrest, notice, legal
    action, court, fine, investigation, FIR, police* — 14/14 terms match no
    pattern in any lexicon. The govt-summons scenario fires only via
    `law_enforcement` ("Inspector Verma") and `national_id` ("Aadhaar"); there
    is no threat/coercion signal family. **Not yet fixed — measured only.**
  - **Hindi/Punjabi lexicons exist but are untested** — a handful of Hinglish
    terms are in the patterns, with no Hindi/Punjabi audio in the corpus to
    exercise them. Do not read their presence as language support.
- **Measured: Whisper ASR error by tier** (`scripts/measure_wer.py`, ground
  truth is LibriSpeech's own `*.trans.txt` for genuine and the exact script text
  given to Piper/XTTS for the attack tiers; `faster-whisper small`, float16, CUDA):

  | tier | clips | ref words | WER | content error (S+D) | insertions |
  |---|---|---|---|---|---|
  | genuine (LibriSpeech) | 32 | 732 | **4.5 %** | 4.1 % | 0.4 % |
  | Piper synthetic | 15 | 431 | **4.6 %** | 3.9 % | 0.7 % |
  | XTTS cloned | 15 | 431 | **23.4 %** | **3.0 %** | **20.4 %** |

  Cloned speech has 5.2× the headline WER — but the breakdown inverts the naive
  reading. **Substitution + deletion error on clones (3.0 %) is LOWER than on
  genuine speech (4.1 %)**: Whisper hears the scripted words *more* accurately on
  XTTS output than on real audiobook audio. The entire gap is **88 hallucinated
  insertions**, concentrated on short clips (`cloned_genuine_control_en_p3`:
  15 reference words, 17 insertions, 113 % WER) — consistent with XTTS emitting
  babble artifacts after short generations, which Whisper dutifully transcribes.

  Consequence for the context engine, stated precisely: **signal recall is
  largely intact** (trigger phrases survive, which is why the cloned CEO call
  still produces 5 correct quotes), but hallucinated text is a **false-positive
  surface** — invented words could match a lexicon pattern and produce a quote
  that was never spoken. We have not observed that yet and have not measured its
  rate. This is an upstream Whisper/XTTS interaction, not a defect we introduced.

- **Decision: the context engine is regex-only, deliberately.** §5 called for
  "rule + regex + small classifier". We built the rules and regex and
  **chose not to add the classifier**, rather than leaving it as an unmet item.
  A small classifier would have to be trained on the only labelled data we
  have — our own five scam scripts — which means it would learn those scripts
  and report confident probabilities with no basis for generalisation, and its
  output could not be traced to a span. A regex that fires produces the exact
  quote, its timestamp, and the named rule; a judge can read the pattern and
  disagree with it. Transparent and auditable beats confident and ungrounded at
  this scale. The classifier becomes worth building when there is labelled
  fraud-call data to fit it on — that is a data problem, not a code one, and it
  is on the production roadmap.

- **Corpus, as built (v1):**

  | tier | clips | total | min / median / max | native sr |
  |---|---|---|---|---|
  | genuine (LibriSpeech dev-clean, 8 speakers) | 32 | 274 s | 2.3 / 8.3 / 15.9 s | 16 kHz |
  | Piper synthetic | 15 | 121 s | 3.9 / 8.0 / 11.6 s | 22.05 kHz |
  | XTTS cloned | 15 | 192 s | 8.7 / 12.7 / 20.5 s | 24 kHz |
  | ASVspoof2019 LA dev (wrapper control) | 86 | 308 s | 1.3 / 3.4 / 7.8 s | 16 kHz |

  Note the duration spread differs by tier (cloned clips are the longest), and
  each tier has a single native sample rate — a nuisance variable we tested for
  and ruled out (§3), but one a larger corpus should balance rather than rely on
  a control for.
- **Languages: English only in v1. Hindi and Punjabi are CUT, not pending.**
  Zero `hi_*`/`pa_*` clips exist. Mozilla Common Voice Hindi needs a manual
  click-through download, and we have no Hindi/Punjabi synthetic or cloned tier
  at all, so there would be nothing to evaluate against. Whisper's ASR is
  multilingual and the acoustic layers are largely language-agnostic, so the
  pipeline would *run* on Hindi audio — but running is not evidence. Claiming
  three languages on the strength of an untested code path is exactly the kind
  of ten-language-dropdown-that-does-nothing §10 warns against. Listed as future
  work with its actual cost: Indian-language spoof data collection and
  per-language prosody baselines.
  - **TTS toolchain is a second, isolated venv** (`tools/.venv-tts`).
    `coqui-tts` requires `transformers>=4.57,<5` (5.x removed
    `isin_mps_friendly`, which XTTS's GPT layer imports) and pulls numpy 2.x /
    librosa 0.11, which would have silently changed prosody features and
    invalidated the baseline in the analysis venv. Corpus generation is a build
    step, so the split costs nothing at runtime — but it is ~5 GB of extra disk.
  - **XTTS-v2 is non-commercial** (Coqui Public Model License). Generation runs
    with `COQUI_TOS_AGREED=1`. Fine for a prototype; a shipped product cannot
    use its outputs.
  - Hindi/Punjabi tiers are still **not built** — Common Voice needs a manual
    click-through download, and no cloned/synthetic clips exist in those
    languages. The corpus is English-only so far.
  - Prosody baseline is now 32 clips / 8 speakers, all LibriSpeech read speech —
    still not conversational telephony, so the "human baseline" is narrow.
  - Silero VAD replaces the energy gate. Consequence: synthetic test tones are
    now correctly rejected as non-speech, so pipeline tests run on real audio
    and skip when the corpus is absent. The energy gate stays available via
    `VOICESHIELD_VAD_BACKEND=energy`.
  - All text file IO now pins `encoding="utf-8"` — the TTS venv runs under an
    ASCII locale and crashed on the em-dash in the scam scripts.
