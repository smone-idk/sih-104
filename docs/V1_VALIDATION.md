# VoiceShield V1 — Diagnosis and Validation Report

**Diagnosis only. No behaviour was changed in this pass.** Two throwaway
diagnostic scripts and two evaluation scripts were added
(`scripts/diagnose_asr_gate.py`, `scripts/build_eval_manifest.py`,
`scripts/run_v1_eval.py`, plus an insertion check inside the existing
`scripts/measure_wer.py`). No detector, threshold, weight, lexicon or rule was
touched.

Measured against the working tree at commit `07d7a7f` + the uncommitted eval
scripts. `docs/ARCHITECTURE.md` was treated as a claim and verified, not trusted.

---

## A. Root cause of the reported bug

### A.1 The symptom, and where it actually occurs

> Transcript lines render, several marked `DISCARDED — LOW ASR CONFIDENCE`,
> while the Context Signals panel says *"No transcript yet. Context components
> report unavailable and their fusion weight is redistributed."*

**It occurs on the Upload path only.** This is the most important fact in the
report, and §1.1 was right to ask for it.

| input path | reproduces? | evidence |
|---|---|---|
| Bundled scenario (WebSocket) | **No** | all 7 scenarios return a populated `context` with 7 signal families |
| File upload (`POST /api/v1/analyze`) | **YES** | see A.2 |
| Push-to-record | **YES** — same endpoint | `Upload.tsx` posts recorded clips to the same `/api/v1/analyze` |

### A.2 Root cause: a field-name mismatch, not the confidence gate

`frontend/src/pages/Upload.tsx:200` renders:

```tsx
<ContextPanel context={clean.context} />
```

But `pipeline.py::analyze_audio` populates `AnalysisResult.context` with a
**metadata dict**, not the context result:

```python
context={"components_wired": bool(transcript_d), "phase": 3,
         **{k: v for k, v in ctx.items() if k in ("scenario", "channel")}},
```
`pipeline.py:381`

The real `ContextResult` is stashed one level down, at
`transcript_d["context"] = cres.as_dict()` (`pipeline.py:348`).

`ContextPanel` then does `context?.signals ?? {}` → `{}` → `present.length === 0`
→ renders the "No transcript yet" branch
(`frontend/src/components/ContextPanel.tsx:48-52`).

Measured on `cloned_ceo_transfer_en_full.wav` via `POST /api/v1/analyze`:

```
clean.context               = {"components_wired": true, "phase": 3}
clean.context has .signals? = False
clean.transcript.n_segments = 13   n_discarded = 0
clean.context_quotes        = 11 quotes
transcript.context.signals  = 7 families   <-- the real result, unread by the UI
```

**The panel's message is not merely unhelpful, it is false.** The same response
reports:

| component | available | value | eff. weight | points |
|---|---|---|---|---|
| `transaction_context` | **True** | 0.908 | 0.250 | **22.70** |
| `caller_trust` | **True** | 0.980 | 0.125 | **12.25** |
| `behavioural_risk` | **True** | 0.926 | 0.125 | **11.57** |

Score 89.17, band HIGH. The context layers contributed **46.5 of 89.2 points**
while the panel told the operator they were unavailable and redistributed.

The streaming path is unaffected because `stream/session.py:169,267` emits
`"context": self.context.as_dict()` — the correct object.

### A.3 §1.3 answered explicitly: is the gate rejecting bad ASR, or is upstream producing bad ASR?

**Neither — and the gate is not the cause of the reported bug at all.** But the
question had to be answered, and the answer is a separate finding:

> **The confidence gate is, in at least one measured case, rejecting *correct*
> ASR.**

`scripts/diagnose_asr_gate.py --sixway genuine_1919_1919-142785-0000`:

```
[1] FULL-AUDIO  lp=-0.691  'Illustration, Long Pepper.'
[2] VAD segments (1)      0.55–2.30s  (1.76s)
[3] utterances handed to ASR (1)   0.55–2.30s
[4] RAW ASR    lp=-0.627  'illustration, Long Pepper.'
[5] AFTER GATE DROPPED lp=-0.627
[6] context received  Transcript.text = ''   → 0 signal families
```

The transcription is **correct** — that is genuine LibriSpeech audiobook prose.
It was dropped because a 1.76 s, two-word, lexically unusual fragment scores
−0.627, just under the −0.6 threshold. `avg_logprob` is a proxy for lexical
predictability and utterance length, not for correctness.

This is the only clip of 72 where *every* segment is dropped, so it is the only
clip where the gate alone could produce an empty context.

**Upstream causes were checked and eliminated:**

| candidate | finding |
|---|---|
| Sample rate to Whisper | 16 kHz in every path; `audio.py::load_audio` resamples before ASR. |
| Browser codec (push-to-record) | **Not a factor.** A 48 kHz WebM/Opus re-encode of the same clip produced logprobs differing by ≤0.01 and an identical transcript and identical fired signals. |
| Channel handling | `audio.py::_to_mono_float32` downmixes; decoded duration and peak matched the source exactly. |
| Amplitude normalisation | Applied only if peak > 1.0. AntiDeepfake applies its own `F.layer_norm` per the model card. |
| VAD boundaries | Handed to `utterances_from_segments` unchanged; verified in the six-way output. |
| Short segments | **Not the driver.** A duration sweep from 2 s to 35.6 s showed short clips score *better* (−0.32 at 2 s vs −0.26 at 35 s) and produced **zero** drops at every length. |
| `language` parameter | `asr_language` is `""` → passed as `None` → auto-detect. |
| Binding constraint | **`avg_logprob` in 14/14 drops. `no_speech_prob` bound zero drops** (max observed 0.218 vs a 0.6 threshold). |

Corpus-wide: **14 of 257 segments dropped (5.4 %)**, across 72 clips.

### A.4 Other candidates eliminated (§1.4)

- **Does streaming call `set_context`?** Yes, but by a different route than
  `ARCHITECTURE.md` §3 step 8 describes. Batch calls
  `scorer.set_context(cres.components)` inside `pipeline.py::analyze_audio`.
  Streaming calls `stream/session.py::apply_transcript`, which calls
  `analyze_context` then `self.scorer.set_context(...)`, driven by
  `api/routes/stream.py::drain_asr`. Both converge on the same
  `WindowScorer.set_context`. **Verified.**
- **WebSocket schema match?** Yes. Backend emits `context` at the top level of
  both the `context` and `final` frames; `LiveAnalysis.tsx` reads
  `final?.context ?? ctx?.context`. Consistent.
- **Stale state across sessions?** No leak found. `LiveAnalysis.tsx::start`
  resets `session`, `windows`, `final` and `ctx` before opening a new socket;
  each `StreamSession` constructs its own `Transcript` and `WindowScorer`.

---

## B. What is actually working (verified)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Frontend upload → REST | **Verified** | `Upload.tsx` → `api.ts::analyzeUpload` → `POST /api/v1/analyze` (`routes/profiles.py::analyze_upload`) |
| 2 | CLI entry | **Verified** | `analyze.py::main` → `pipeline.py::analyze_file`; ran end to end |
| 3 | Shared `WindowScorer` | **Verified** | batch `pipeline.py::analyze_audio` and `stream/session.py::_score_segment` both call `WindowScorer.score`; both end at `WindowScorer.aggregate` |
| 4 | ASR config / segmenter / gate | **Verified** | faster-whisper `small` float16 CUDA; `vad_filter=False`, `condition_on_previous_text=False`; gate in `asr/worker.py::gate_segment` |
| 5 | Seven families, `parse_amount`, `_negated` | **Verified** | 7 families in `LEXICONS`; amount parsing and negation exercised (§E) |
| 6 | `telephony.py::degrade` order | **Verified** | resample→µ-law→noise→resample at `telephony.py:51-57`; deterministic via `np.random.default_rng(seed)` |
| 6b | Telephony **reachable from the frontend** | **Verified** | `LiveAnalysis.tsx:79` passes `telephony` into `openStream`; `Upload.tsx` exposes `compare_telephony` + SNR slider |
| 7 | AASIST at 0.00, excluded | **Verified** | `/inventory`: `contributes=False, feeds=None, wt=0.0`. `_scoring_map` filters on `contributes`, so it is *absent from* fusion inputs, not multiplied by zero |
| 8 | Redistribution / UNAVAILABLE / noisy-OR | **Verified with one defect** | effective weights sum to 1.0000±0.0001 across all 72 clips; see C-3 |
| 9 | Band floor + server-side approval | **Verified** | forged `{"band":"LOW","verified":true,"override":true}` → **403**, server band HIGH |
| 10 | Test quality | **Verified** | 162 pass, **0 skipped** — no vacuous skips; but see C-4 |

**Confirmed known gap:** `windows`, `transcript_segments` and `context_signals`
have **no writers** — every `INSERT INTO` in the Python source targets exactly
`approvals`, `incidents`, `sessions`, `verifications`, `voice_profiles`.

*Consequence for the Incident Log:* an incident row can link to a `session_id`
and show that session's `final_score`/`final_band`, but **cannot reconstruct
why** — no per-window timeline, no transcript, no context signals are persisted.
Clicking through to a session detail can only ever show the summary. Post-hoc
audit of a blocked transfer is therefore not possible, which is a meaningful
limitation for something positioned as a security control.

---

## C. What is broken

| # | Issue | Severity | Smallest fix |
|---|---|---|---|
| **C-1** | Upload/push-to-record Context panel reads `clean.context` (metadata) instead of `clean.transcript.context` (the real result). Panel actively reports "unavailable and redistributed" while those components contributed 46.5 of 89.2 points. | **High** — misinforms the operator on 2 of 3 input paths | One line in `Upload.tsx:200`: `context={clean.transcript?.context ?? null}`. Better: make `pipeline.py` put the ContextResult at `AnalysisResult.context` and move the metadata elsewhere, so both paths use the same field name. |
| **C-2** | Confidence gate drops **correct** ASR on short, lexically unusual utterances (`'illustration, Long Pepper.'`, lp −0.627). 1 of 72 clips loses its entire transcript. | **Medium** | Do **not** special-case. Either accept it (cost measured below and it is small), or re-sweep with a length-aware threshold and show the tradeoff table. |
| **C-3** | An unavailable **context** component carries `raw_value = 0.0` instead of `None`, so the explainability table renders **`0.000`** where the speaker layer correctly renders `—`. 174 occurrences across 72 clips (caller_trust 72, transaction_context 60, behavioural_risk 42). | **Low — cosmetic, score unaffected** (`effective_weight` is 0.0 in every case) | `context/engine.py::_component` should pass `value=None` when `available=False`, matching `speaker_ecapa`'s convention. |
| **C-4** | No test covers the frontend↔backend field contract for `context`. This is exactly why C-1 shipped. `test_analyze_upload_runs_the_same_pipeline` asserts `transcript` and `context_quotes` are present but never that `context` carries `signals`. | **Medium** | Add one assertion that the `/analyze` response exposes context signals wherever the UI reads them. |
| **C-5** | `pipeline.py` docstrings are stale — the module docstring and `_context_inputs` still say context is "not wired yet (Phase 3)", and that string is emitted as a user-visible `note` when no context components are supplied. | **Low** | Update the three strings. |

---

## D. What is unverified

- **Push-to-record with a real microphone.** Verified by *simulation* — a 48 kHz
  WebM/Opus re-encode through the same endpoint. No physical microphone was
  used; browser-specific codec variants (Firefox Ogg/Opus, Safari) are
  `UNVERIFIED`.
- **`policy/challenge.py` end-to-end with real response audio.** Unit-tested;
  not exercised with a genuine spoken challenge response in this pass.
- **Frontend rendering of the discarded-segment state.** The backend contract is
  verified (`confident`, `gate_reason`, `n_discarded`); the visual rendering was
  not re-screenshotted this pass.
- **ASVspoof control clips have no reference transcripts**, so they contribute to
  anti-spoofing metrics only, never to ASR or context metrics.
- **Whether the reported symptom was observed on Upload or push-to-record.** Both
  share the endpoint and both reproduce; which one the reporter saw is
  `UNVERIFIED` and does not change the fix.

---

## E. Metrics

Manifest: `data/eval/manifest.json` (152 clips). Full results:
`data/eval/v1_eval.json`, `data/eval/asr_wer.json`.

```
category                  n    with reference
asvspoof_control         80         0
cloned_enrolled          20        20
genuine_enrolled          4         4
genuine_non_enrolled     28        28
synthetic_non_enrolled   20        20
TOTAL                   152        72
```

### E.1 ASR — word error rate (n=62 clips with references)

Measured on the transcript **as the context engine receives it** (post-gate).

| tier | clips | ref words | WER | sub | del | ins |
|---|---|---|---|---|---|---|
| genuine | 32 | 732 | **5.1 %** | 23 | 11 | 3 |
| Piper | 15 | 431 | **4.6 %** | 12 | 5 | 3 |
| XTTS cloned | 15 | 431 | **6.7 %** | 10 | 6 | 13 |

**The gate materially improved this.** Before gating, cloned-tier WER was 23.4 %
with 88 hallucinated insertions; post-gate it is 6.7 % with 13. The cost is
visible too: genuine deletions rose 6 → 11 and WER 4.5 % → 5.1 %, i.e. **+0.6 pp
on genuine speech**, which is the price of C-2.

**Hallucinated-insertion lexicon check, per clip (never concatenated across
clips):**

| tier | inserted words | single-word matches | within-clip span matches | amounts parsed |
|---|---|---|---|---|
| genuine | 3 | 0 | 0 | 0 |
| Piper | 3 | 0 | 0 | 0 |
| cloned | 13 | **0** | **0** | **0** |

The previously documented false positive — Whisper hallucinating *"7 at the year
right now and I only get it"* on a benign clip and firing urgency — **no longer
occurs.** The gate eliminated it.

**Do fraud-bearing phrases survive?** Yes, in every case measured. The context
false negatives in E.3 are lexicon gaps, not ASR losses.

### E.2 Anti-spoofing (genuine n=32; each attack tier n=20)

| model | condition | genuine mean | attack mean | AUC | EER | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|
| **AntiDeepfake** | vs synthetic non-enrolled | 0.0003 | 1.0000 | **1.000** | **0.00 %** | 20 | 0 | 0 | 32 |
| **AntiDeepfake** | vs **cloned enrolled** | 0.0003 | 1.0000 | **1.000** | **0.00 %** | 20 | 0 | 0 | 32 |
| AASIST | vs synthetic non-enrolled | 0.3314 | 0.4986 | 0.670 | 36.25 % | 6 | 14 | 7 | 25 |
| AASIST | vs **cloned enrolled** | 0.3314 | 0.3569 | **0.561** | 40.31 % | 2 | 18 | 7 | 25 |

Confusion matrices at the operating threshold (`synthetic_high_threshold` 0.65).

**Home-turf control (ASVspoof2019 LA dev, n=40 bonafide / 40 spoof):** *both*
models score AUC 1.000, EER 0.00 % — AntiDeepfake bonafide 0.002 / spoof 0.9999;
AASIST bonafide 0.0013 / spoof 0.9773. This is the control that proves the AASIST
wrapper is correct and the failure is distributional.

### E.3 Context engine — per family (n=72 clips with references)

| family | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| urgency | 20 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| secrecy | 16 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| threat_coercion | 12 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| authority_claim | 7 | 0 | 1 | 1.000 | 0.875 | 0.933 |
| out_of_workflow | 3 | 0 | 1 | 1.000 | 0.750 | 0.857 |
| credential_solicitation | 8 | 0 | 4 | 1.000 | 0.667 | 0.800 |
| transaction_intent | 12 | 0 | 6 | 1.000 | 0.667 | 0.800 |

**Zero false positives in every family.** All error is recall.

**Signals missing a quote, rule or timestamp: 0.** The Phase 3 invariant holds
across the corpus.

### E.4 Speaker verification (cosine vs the enrolled profile)

| group | n | mean | min | max |
|---|---|---|---|---|
| genuine **enrolled** | 4 | **0.854** | 0.807 | 0.883 |
| genuine non-enrolled (impostor) | 28 | **0.016** | −0.129 | 0.138 |
| **cloned enrolled** | 20 | **0.534** | 0.498 | 0.578 |
| synthetic non-enrolled | 20 | 0.170 | 0.130 | 0.228 |

- genuine vs impostor: **AUC 1.000, EER 0.00 %, FAR 0.00 %, FRR 0.00 %**
- cloned vs impostor: **AUC 1.000** — every clone scores more like the enrolled
  speaker than every real impostor does. That is the attack succeeding at what it
  is designed to do, and it is why the two layers must stay separate.

### E.5 Fusion and policy (n=72; ASVspoof control excluded)

| category | n | LOW | MEDIUM | HIGH | floored |
|---|---|---|---|---|---|
| cloned_enrolled | 20 | 0 | 0 | **20** | **0** |
| synthetic_non_enrolled | 20 | 0 | 1 | 19 | 0 |
| genuine_enrolled | 4 | **4** | 0 | 0 | 0 |
| genuine_non_enrolled | 28 | 27 | 1 | 0 | 0 |

**The band floor fired zero times on the full corpus.** Every cloned clip now
reaches HIGH on score alone. The floor remains a correct safety net but is no
longer load-bearing under these conditions (no directory record attached in this
run, so `caller_trust` was unavailable throughout).

**Invariants:**

| invariant | result |
|---|---|
| Effective weights sum to 1.0 | ✅ 0.9999 – 1.0001 across all 72 |
| Missing profile does not read as safety | ✅ `speaker_consistency` → `available=False`, `raw_value=None`, `eff_weight=0.0` |
| Context cannot lower risk | ✅ unavailable context contributes 0 points at 0 weight |
| Unavailable components do not become zero **in scoring** | ✅ |
| Unavailable components do not become zero **in the payload** | ❌ **174 violations** — see C-3 (cosmetic) |
| Forged client payload cannot bypass approval | ✅ 403, server band HIGH |

### E.6 Telephony sweep (AntiDeepfake mean spoof-prob, seed 0, reproducible)

Additive white noise only, so SNR is the isolated variable.

| condition | genuine enrolled | genuine non-enrolled | synthetic | cloned |
|---|---|---|---|---|
| clean | 0.0002 | 0.0003 | **1.000** | **1.000** |
| 20 dB | 0.0002 | 0.0028 | 0.785 | 0.775 |
| **10 dB** | 0.002 | 0.023 | **0.314** | **0.305** |

At 10 dB both attack tiers fall **below the 0.65 verdict threshold**. Ranking
survives (genuine stays ~0.02, two orders of magnitude lower) but the *verdict*
does not fire, so `CLONED_VOICE` never triggers and the band floor never applies.
This is the previously documented calibration failure, now measured across the
full corpus rather than one clip.

---

## F. Representative failures

**F-1 — the gate drops correct ASR (C-2).**
`genuine_1919_1919-142785-0000.wav`, 2.66 s, ref `"ILLUSTRATION LONG PEPPER"`.
Whisper: `'illustration, Long Pepper.'` — correct. lp −0.627 → dropped → WER
100 % (3 deletions) and empty context. The transcription was right; the
confidence proxy was not.

**F-2 — XTTS artifacts, now contained.**
`cloned_ceo_transfer_en_p1.wav`, WER 39.0 %, 12 insertions:
`"dharuddin naimukh tell them what they need and adapt at our help"`. The
scripted content transcribes correctly; XTTS appends babble that Whisper
faithfully renders. Post-gate this produced **zero** context signals — the
failure is contained.

**F-3 — spelled-out amounts are the biggest recall gap.**
`transaction_intent` missed 6 of 18. Every miss is a written-out number the
reference contains but the hypothesis phrased differently:
`'forty-nine thousand'` (bank OTP ×2), `'forty thousand'` (family emergency).
Same root as `credential_solicitation` missing `'Aadhaar'` ×3 — ASR renders it
differently from the reference.

**F-4 — the one attack clip that did not reach HIGH.**
`synthetic_govt_summons_en_p3` → **66.8 MEDIUM**, verdict `SYNTHETIC_OTHER`.
Reported as a finding, not scripted around.

**F-5 — the one genuine clip above LOW.**
`genuine_1462_1462-170138-0003` → **41.08 MEDIUM**, verdict `SPEAKER_MISMATCH`.
A different real human measured against the enrolled profile. Arguably correct
behaviour rather than a false positive: the system is saying "this is not who the
profile says", which is true.

---

## G. Recommended fixes, ranked by impact ÷ effort

| rank | fix | impact | effort | before the competition? |
|---|---|---|---|---|
| **1** | **C-1** — point the Upload panel at the real context object (one line), then unify the field name across both paths | High. Removes a panel that actively contradicts the score on 2 of 3 input paths | Minutes | **Yes. Fix this.** A judge uploading a clip currently sees the system deny its own working output. |
| **2** | **C-4** — assert the `/analyze` context contract in a test | Medium. Prevents recurrence | Minutes | **Yes**, with fix 1 |
| **3** | **C-3** — `value=None` for unavailable context components | Low, but directly undercuts the "not applicable" messaging already committed to | Minutes | **Yes** — cheap and visible in the explainability table |
| **4** | **C-5** — stale "not wired yet (Phase 3)" strings, one of which is user-visible | Low | Minutes | Yes |
| **5** | Persist `windows` / `transcript_segments` / `context_signals` | Medium. Makes the Incident Log auditable | Hours | **No — disclose.** Already documented; too large to land safely now |
| **6** | **C-2** — length-aware confidence gate | Low on this corpus (one clip, no fraud content lost) | Hours, plus a re-sweep | **No — disclose.** The measured cost is +0.6 pp genuine WER against elimination of the hallucination FP surface. That trade is currently favourable. |
| **7** | SNR-conditioned thresholds for the 10 dB collapse | High if deployed, none for the demo | Days | **No — disclose prominently.** Belongs on the Evaluation page as the headline limitation. |
| **8** | Recall gaps in `transaction_intent` / `credential_solicitation` | Low — precision is 1.000, only recall suffers | Hours | **No.** Rules get added when a measurement demands them; these are measured and documented as F-3. |

---

## H. Deferred to V2

Nothing here is built until the English baseline is reproducible.

- **Hindi and Punjabi.** Cut, not deferred-with-intent. No attack data exists in
  those languages, so there would be nothing to evaluate against.
- **Additional anti-spoofing models.** AntiDeepfake is at AUC 1.000 / EER 0.00 %
  on this corpus; adding models cannot be justified by a measurement.
- **Real telephony integration.** The µ-law chain is a faithful simulation;
  actual SIP/PSTN capture is out of scope.
- **SNR-conditioned threshold calibration.** Requires a labelled multi-SNR
  corpus, which we do not have.
- **A trained context classifier.** Requires labelled fraud-call data. The
  regex-only decision is recorded in `ARCHITECTURE.md` §6.9.

---

## Instructions considered and refused (Part 4)

Recorded as requested:

- **Did not loosen the ASR confidence gate**, and **did not add a keyword bypass**
  for low-confidence segments. The measurements in this report actively support
  keeping it: it cut cloned-tier WER 23.4 % → 6.7 % and reduced hallucinated
  insertions 88 → 13, with the previously documented urgency false positive now
  measuring **zero**. A keyword bypass would have re-admitted exactly the
  hallucinated text the gate removes. Its one measured cost (C-2) is reported
  with its tradeoff rather than special-cased away.
- **Did not add new signal families.** Coverage gaps are reported as measured
  false negatives in E.3 and F-3 instead. Note that precision is **1.000 in all
  seven families** — the argument for new families is not supported by the data;
  the gap is recall on existing families.
- **Did not add a bounded additive context accumulator.** Noisy-OR is already in
  place for a documented reason (`ARCHITECTURE.md` §6.4).
- **Did not design a demo script to make the score rise.** F-4 reports the one
  attack clip that did not reach HIGH as a finding.

---

*Generated 2026-09-05. Reproduce with:*
```bash
python scripts/build_eval_manifest.py
python scripts/run_v1_eval.py
python scripts/measure_wer.py
python scripts/diagnose_asr_gate.py
python scripts/diagnose_asr_gate.py --sixway genuine_1919_1919-142785-0000
```
