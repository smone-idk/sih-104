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
> **Consequence for the product story:** on this corpus the voice-clone claim
> cannot rest on the anti-spoofing layer. What does work today is speaker
> verification (ECAPA) plus, from Phase 3, the transcript-derived context
> engine. Closing the synthetic-detection gap requires swapping in an
> SSL-based anti-spoofing model trained on ASVspoof2021-DF / In-the-Wild — a
> one-line `DetectorRegistry` change by design, and the top item on the
> production roadmap.
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
  - **The anti-spoofing layer does not work on this corpus.** Diagnosed to a
    measured domain gap, not a bug (§3). Left as-is and badged, per the
    diagnosis. The practical consequence is that a cloned clip does **not**
    currently score higher than a genuine one on the synthetic axis — it scores
    *lower* than a different-speaker genuine clip, because the clone matches the
    enrolled profile. This blocks the Phase 2 gate as originally written and
    needs a decision before Phase 2 starts.
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
