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

- **AASIST** trained on **ASVspoof2019 LA**. Known generalization gap: ASVspoof
  2019-era training data does **not** cover modern neural TTS/VC (XTTS-v2,
  StyleTTS2, VALL-E-class). Expect degraded, sometimes inverted, scores on
  current cloning tools. The Evaluation page measures this on our own clips
  rather than hiding it.
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
