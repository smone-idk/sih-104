# PROVENANCE — every demo clip's origin, licence, parameters

`scripts/build_demo_assets.py` appends one line per generated clip below.
Nothing is recorded from a person for v1; the corpus is built from public
datasets and open TTS models.

## Sources & licences

| Tier | Source | Licence | Notes |
|---|---|---|---|
| genuine (English) | LibriSpeech `dev-clean` | CC BY 4.0 | one speaker designated "Rajesh Sharma — CFO"; 3 clips → enrolled profile |
| genuine (Hindi) | Mozilla Common Voice — Hindi | CC0 1.0 | fetched manually (dataset has a click-through) into `demo_assets/genuine/` as `hi_*.wav` |
| synthetic (non-cloned) | Piper TTS voice `en_US-lessac-medium` | MIT | unrelated speaker; reads the scam scripts |
| cloned | Coqui **XTTS-v2** | **Coqui Public Model License — non-commercial** | clones the enrolled LibriSpeech speaker; prototype-only, see LIMITATIONS.md §5 |
| ASVspoof LA samples (optional) | ASVspoof 2019 LA | ASVspoof EULA | added to the eval set only if downloadable at the venue |

## Scam scripts

`demo_assets/scripts/*.txt` — written for this project, fictional, benign in
intent. They exist as text so they can be regenerated in any voice / language.

| File | Scenario |
|---|---|
| `ceo_transfer_en.txt` | CEO/CFO fund-transfer impersonation |
| `bank_otp_en.txt` | Bank official OTP solicitation |
| `govt_summons_en.txt` | Government official — fake summons/penalty |
| `family_emergency_en.txt` | Family-member emergency |
| `genuine_control_en.txt` | Genuine benign call — **must score LOW** |

## Generated clips

<!-- build_demo_assets.py appends dated lines here -->
