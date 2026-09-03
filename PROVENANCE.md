# PROVENANCE — every demo clip's origin, licence, parameters

`scripts/build_demo_assets.py` regenerates this file from
`demo_assets/manifest.json`. Nothing is recorded from a person: the corpus is
built entirely from public datasets and open TTS models.

## Sources & licences

| Tier | Source | Licence | Notes |
|---|---|---|---|
| genuine (English) | LibriSpeech `dev-clean` | CC BY 4.0 | speaker 1272 designated "Rajesh Sharma — CFO"; 3 clips enroll the voice profile |
| genuine (Hindi) | Mozilla Common Voice — Hindi | CC0 1.0 | fetched manually (click-through) into `demo_assets/genuine/` as `hi_*.wav` |
| synthetic (non-cloned) | Piper TTS `en_US-lessac-medium` | MIT | unrelated speaker reading the scam scripts |
| cloned | Coqui XTTS-v2 | Coqui Public Model License — **non-commercial** | clones the enrolled speaker; prototype-only, see LIMITATIONS.md §5 |
| ASVspoof LA (control) | ASVspoof 2019 LA dev, via `Nemez1z/asvspoof-2019-la` mirror | ASVspoof EULA | 80-clip balanced subset in `demo_assets/_asvspoof_dev/`; used as the detector wrapper-correctness control (LIMITATIONS.md §3) |

## Model licences

| Model | Role | Licence |
|---|---|---|
| `nii-yamagishilab/wav2vec-large-anti-deepfake` | **primary** anti-spoofing (weight 0.30) | **CC-BY-NC-SA-4.0 — non-commercial** |
| `clovaai` AASIST | anti-spoofing baseline (weight 0.00) | MIT |
| `speechbrain/spkrec-ecapa-voxceleb` | speaker verification | Apache-2.0 |
| `silero-vad` | voice activity detection | MIT |
| `faster-whisper small` | ASR (Phase 3) | MIT |
| Piper `en_US-lessac-medium` | demo asset generation | MIT |
| Coqui XTTS-v2 | demo asset generation | Coqui CPML — non-commercial |

## Scam scripts

`demo_assets/scripts/*.txt` — written for this project, fictional, benign in
intent. They exist as text so they can be regenerated in any voice or language.

| File | Scenario |
|---|---|
| `ceo_transfer_en.txt` | CEO/CFO fund-transfer impersonation |
| `bank_otp_en.txt` | Bank official OTP solicitation |
| `govt_summons_en.txt` | Government official — fake summons/penalty |
| `family_emergency_en.txt` | Family-member emergency |
| `genuine_control_en.txt` | Genuine benign call — **must score LOW** |


## Generated clips


### genuine (32 clips)

| clip | dur (s) | sr | source | licence | model | params | sha256 |
|---|---|---|---|---|---|---|---|
| `genuine/enrolled_1272_1272-128104-0000.wav` | 5.855 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `799f78ed4beb4de7…` |
| `genuine/enrolled_1272_1272-128104-0001.wav` | 4.815 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `473b1bb573e8f7cd…` |
| `genuine/enrolled_1272_1272-128104-0002.wav` | 12.485 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `5f4d8f7331acffc3…` |
| `genuine/enrolled_1272_1272-128104-0003.wav` | 9.9 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `fc23353b884061a5…` |
| `genuine/genuine_1462_1462-170138-0000.wav` | 14.55 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `ba4ee8747cef66c2…` |
| `genuine/genuine_1462_1462-170138-0001.wav` | 3.985 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `ae632195406f7cbe…` |
| `genuine/genuine_1462_1462-170138-0002.wav` | 4.645 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `3c3e7214375c2da3…` |
| `genuine/genuine_1462_1462-170138-0003.wav` | 2.32 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `4f5f74f7966e58a0…` |
| `genuine/genuine_1673_1673-143396-0000.wav` | 14.675 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `d6caba430847ce6c…` |
| `genuine/genuine_1673_1673-143396-0001.wav` | 14.05 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `e2d8c648942c1b31…` |
| `genuine/genuine_1673_1673-143396-0002.wav` | 7.85 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `dc352314816d7c24…` |
| `genuine/genuine_1673_1673-143396-0003.wav` | 11.355 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `9fefc23140e94ae8…` |
| `genuine/genuine_174_174-168635-0000.wav` | 4.53 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `6cc2d2aa5a57b9f5…` |
| `genuine/genuine_174_174-168635-0001.wav` | 4.65 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `25c9454a8b3411d9…` |
| `genuine/genuine_174_174-168635-0002.wav` | 15.86 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `50733bcfbd300b3e…` |
| `genuine/genuine_174_174-168635-0003.wav` | 13.18 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `fb60dafd56135a7c…` |
| `genuine/genuine_1919_1919-142785-0000.wav` | 2.66 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `5ee4500b59e981fa…` |
| `genuine/genuine_1919_1919-142785-0001.wav` | 11.05 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `17546c7ee0d935c9…` |
| `genuine/genuine_1919_1919-142785-0002.wav` | 10.26 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `7bf4bb928140b5b9…` |
| `genuine/genuine_1919_1919-142785-0003.wav` | 5.405 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `274d11f95dbf434e…` |
| `genuine/genuine_1988_1988-147956-0000.wav` | 14.95 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `afc03ae866ffac66…` |
| `genuine/genuine_1988_1988-147956-0001.wav` | 14.21 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `98a8754e281e12c5…` |
| `genuine/genuine_1988_1988-147956-0002.wav` | 4.395 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `d7464f9d082fccfb…` |
| `genuine/genuine_1988_1988-147956-0003.wav` | 2.495 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `480487d9e13b6b1a…` |
| `genuine/genuine_1993_1993-147149-0000.wav` | 6.715 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `56d518b1ba271363…` |
| `genuine/genuine_1993_1993-147149-0001.wav` | 9.535 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `83df2b6894754013…` |
| `genuine/genuine_1993_1993-147149-0002.wav` | 5.76 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `84022bcf89ee4506…` |
| `genuine/genuine_1993_1993-147149-0003.wav` | 14.3 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `e846f8d2c65274dc…` |
| `genuine/genuine_2035_2035-147960-0000.wav` | 9.02 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `b0bef79e65f00b31…` |
| `genuine/genuine_2035_2035-147960-0001.wav` | 3.925 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `1ed72e29c4de0770…` |
| `genuine/genuine_2035_2035-147960-0002.wav` | 8.84 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `ca7cd13377adfa22…` |
| `genuine/genuine_2035_2035-147960-0003.wav` | 5.84 | 16000 | LibriSpeech dev-clean | CC BY 4.0 | — | format=flac->wav, unmodified | `6f47c18d7e2515e4…` |

### synthetic (15 clips)

| clip | dur (s) | sr | source | licence | model | params | sha256 |
|---|---|---|---|---|---|---|---|
| `synthetic/synthetic_bank_otp_en_p1.wav` | 10.101 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `0e9a042610de2e08…` |
| `synthetic/synthetic_bank_otp_en_p2.wav` | 9.369 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `f216a34ba3db4123…` |
| `synthetic/synthetic_bank_otp_en_p3.wav` | 8.382 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `259a042896cf71e1…` |
| `synthetic/synthetic_ceo_transfer_en_p1.wav` | 11.622 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `5fe1d5f5332ac934…` |
| `synthetic/synthetic_ceo_transfer_en_p2.wav` | 7.616 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `38fdb02b707af3b8…` |
| `synthetic/synthetic_ceo_transfer_en_p3.wav` | 7.964 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `a2eff802c7dde045…` |
| `synthetic/synthetic_family_emergency_en_p1.wav` | 5.991 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `892e2759fcfcf364…` |
| `synthetic/synthetic_family_emergency_en_p2.wav` | 6.966 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `77737398b28eb59a…` |
| `synthetic/synthetic_family_emergency_en_p3.wav` | 7.338 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `bd349a34dabe5178…` |
| `synthetic/synthetic_genuine_control_en_p1.wav` | 3.866 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `431d59665e49aa44…` |
| `synthetic/synthetic_genuine_control_en_p2.wav` | 11.215 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `f63e75549a85825f…` |
| `synthetic/synthetic_genuine_control_en_p3.wav` | 4.888 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `1ed9cfd61b93b603…` |
| `synthetic/synthetic_govt_summons_en_p1.wav` | 5.793 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `d07732e72d681825…` |
| `synthetic/synthetic_govt_summons_en_p2.wav` | 11.505 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `db52ad761349a378…` |
| `synthetic/synthetic_govt_summons_en_p3.wav` | 8.011 | 22050 | Piper TTS voice en_US-lessac-medium | MIT | piper/en_US-lessac-medium | length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8 | `3eeff7414cf0a0ab…` |

### cloned (15 clips)

| clip | dur (s) | sr | source | licence | model | params | sha256 |
|---|---|---|---|---|---|---|---|
| `cloned/cloned_bank_otp_en_p1.wav` | 14.625 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `58d41430e62d7d46…` |
| `cloned/cloned_bank_otp_en_p2.wav` | 17.495 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `a69fdce076ef0bf9…` |
| `cloned/cloned_bank_otp_en_p3.wav` | 12.663 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `53bca093222f51b8…` |
| `cloned/cloned_ceo_transfer_en_p1.wav` | 20.503 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `a002f92ff675bc56…` |
| `cloned/cloned_ceo_transfer_en_p2.wav` | 9.644 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `d13f0f9de6eec23d…` |
| `cloned/cloned_ceo_transfer_en_p3.wav` | 14.978 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `3bc4c64e11b8a879…` |
| `cloned/cloned_family_emergency_en_p1.wav` | 10.487 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `d5fad2d738e0638b…` |
| `cloned/cloned_family_emergency_en_p2.wav` | 11.596 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `ea36f0fb2d716369…` |
| `cloned/cloned_family_emergency_en_p3.wav` | 8.716 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `e4a06e0e787efef9…` |
| `cloned/cloned_genuine_control_en_p1.wav` | 8.865 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `274c3c88bd70b2c9…` |
| `cloned/cloned_genuine_control_en_p2.wav` | 12.759 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `08ade8d50adfb9a4…` |
| `cloned/cloned_genuine_control_en_p3.wav` | 12.759 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `d5cab40e4eb4179a…` |
| `cloned/cloned_govt_summons_en_p1.wav` | 10.209 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `cefd53652c4d6e25…` |
| `cloned/cloned_govt_summons_en_p2.wav` | 15.543 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `7b8e05d7ad0aeed2…` |
| `cloned/cloned_govt_summons_en_p3.wav` | 10.903 | 24000 | Coqui XTTS-v2 voice clone of LibriSpeech spk 1272 | Coqui Public Model License (non-commercial) | tts_models/multilingual/multi-dataset/xtts_v2 | device=cuda, language=en, speaker_wav=['enrolled_1272_1272-128104-0000.wav', 'enrolled_1272_1272-128104-0001.wav', 'enrolled_1272_1272-128104-0002.wav'] | `490b554e1658a8ce…` |

_Regenerated 2026-09-03 from `demo_assets/manifest.json` (62 clips). Full SHA256 digests are in the manifest._
