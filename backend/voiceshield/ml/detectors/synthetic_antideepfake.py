"""AntiDeepfake synthetic-speech detector (Wang & Yamagishi et al., 2025).

Model: `nii-yamagishilab/wav2vec-large-anti-deepfake`
  arXiv 2506.21090 — "Post-training for Deepfake Speech Detection"
  wav2vec2-large backbone (317.4M params) + a 2-way `proj_fc` head, post-trained
  on **18k hours of fake and 56k hours of real speech** aggregated across many
  corpora — deliberately NOT the ASVspoof2019-LA-only recipe that our AASIST
  layer fails on (see LIMITATIONS.md §3).
  Licence: CC-BY-NC-SA-4.0 (non-commercial — prototype use, documented).

Why there is a key remap here
-----------------------------
The published checkpoint stores fairseq-style parameter names
(`m_ssl.model.encoder.layers.N.fc1`, `post_extract_proj`, ...) and the authors'
recipe imports `fairseq`, which does not install on Python 3.11 / torch 2.5.
We therefore remap onto HuggingFace's `Wav2Vec2Model`, whose config the repo
already ships in HF form (24 layers, 1024 hidden, 16 heads, 4096 FFN,
do_stable_layer_norm=True, feat_extract_norm="layer").

A silent mis-map would produce a confident, meaningless detector — the exact
failure mode we spent Phase 1.5 Task B ruling out for AASIST. Two guards:

  1. `load_state_dict(strict=True)` — every target parameter must be filled by
     a remapped source tensor, with matching shape. Any missing/unexpected key
     raises and the detector reports unavailable rather than scoring.
  2. `scripts/validate_antideepfake.py` scores the model on labelled
     ASVspoof2019 LA dev clips as a wrapper-correctness control before the
     model is trusted on our own corpus.

Preprocessing follows the model card exactly: mono, 16 kHz, then
`F.layer_norm(wav, wav.shape)` over the whole window (zero-mean/unit-variance).
Note this DIFFERS from AASIST, which takes the raw un-normalised waveform.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ...config import get_settings
from .base import Detector, DetectorResult

log = logging.getLogger("voiceshield.detector.antideepfake")

HF_REPO = "nii-yamagishilab/wav2vec-large-anti-deepfake"
LOCAL_DIR = "antideepfake-wav2vec-large"

#: Model card states the head emits <Fake score, Real score>, so index 0 is the
#: spoof/fake logit. Verified empirically — see scripts/validate_antideepfake.py.
FAKE_INDEX = 0
REAL_INDEX = 1


def remap_fairseq_to_hf(sd: dict, prefix: str = "m_ssl.model.") -> dict:
    """fairseq wav2vec2 parameter names -> HuggingFace Wav2Vec2Model names.

    Pretraining-only tensors (quantizer, project_q, final_proj, mask_emb) are
    dropped: they take no part in `features_only` inference.
    """
    import re

    out: dict = {}
    drop = ("quantizer.", "project_q.", "final_proj.")
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        s = k[len(prefix):]
        if s.startswith(drop):
            continue

        # fairseq's SpecAugment mask embedding == HF's masked_spec_embed.
        # Unused at inference (we never mask) but mapped so strict=True holds.
        if s == "mask_emb":
            out["masked_spec_embed"] = v
            continue

        # conv feature extractor: <i>.0.* = conv, <i>.2.1.* = layer norm
        m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)$", s)
        if m:
            out[f"feature_extractor.conv_layers.{m[1]}.conv.{m[2]}"] = v
            continue
        m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)$", s)
        if m:
            out[f"feature_extractor.conv_layers.{m[1]}.layer_norm.{m[2]}"] = v
            continue

        # feature projection
        if s.startswith("post_extract_proj."):
            out["feature_projection.projection." + s.split(".", 1)[1]] = v
            continue
        if re.match(r"layer_norm\.(weight|bias)$", s):
            out["feature_projection.layer_norm." + s.split(".", 1)[1]] = v
            continue

        # positional conv embedding (weight-norm parametrised)
        m = re.match(r"encoder\.pos_conv\.0\.(bias|weight_g|weight_v)$", s)
        if m:
            out[f"encoder.pos_conv_embed.conv.{m[1]}"] = v
            continue

        # transformer layers
        m = re.match(r"encoder\.layers\.(\d+)\.(.+)$", s)
        if m:
            i, rest = m[1], m[2]
            if rest.startswith("self_attn."):
                rest = "attention." + rest[len("self_attn."):]
            elif rest.startswith("self_attn_layer_norm."):
                rest = "layer_norm." + rest[len("self_attn_layer_norm."):]
            elif rest.startswith("fc1."):
                rest = "feed_forward.intermediate_dense." + rest[len("fc1."):]
            elif rest.startswith("fc2."):
                rest = "feed_forward.output_dense." + rest[len("fc2."):]
            # final_layer_norm keeps its name
            out[f"encoder.layers.{i}.{rest}"] = v
            continue

        # encoder final layer norm (stable-layer-norm variant)
        if s.startswith("encoder.layer_norm."):
            out[s] = v
            continue

        log.debug("antideepfake remap: unhandled source key %s", s)
    return out


class AntiDeepfakeDetector(Detector):
    """Primary voice-authenticity layer. Carries the fusion weight."""

    name = "synthetic_speech_ssl"
    kind = "pretrained"
    feeds = "voice_authenticity"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._head = None
        self._dtype = None
        self.model_id = HF_REPO
        self.licence = "CC-BY-NC-SA-4.0 (non-commercial)"
        self.training_data = "18k h fake + 56k h real, multi-corpus (arXiv 2506.21090)"

    # --- loading -----------------------------------------------------
    def load(self) -> None:
        s = get_settings()
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import Wav2Vec2Config, Wav2Vec2Model

            path = s.model_cache_dir / LOCAL_DIR
            weights = path / "model.safetensors"
            cfg_file = path / "config.json"
            if not weights.exists() or not cfg_file.exists():
                raise FileNotFoundError(
                    f"{path} missing model.safetensors/config.json — "
                    "run scripts/fetch_models.py")

            cfg = Wav2Vec2Config.from_pretrained(str(cfg_file))
            model = Wav2Vec2Model(cfg)

            raw = load_file(str(weights))
            mapped = remap_fairseq_to_hf(raw)

            # torch >= 2.1 may expose weight-norm as parametrizations; match
            # whichever naming this build of transformers/torch produced.
            tgt = set(model.state_dict().keys())
            if "encoder.pos_conv_embed.conv.weight_g" not in tgt:
                g = mapped.pop("encoder.pos_conv_embed.conv.weight_g", None)
                v = mapped.pop("encoder.pos_conv_embed.conv.weight_v", None)
                if g is not None:
                    mapped["encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = g
                    mapped["encoder.pos_conv_embed.conv.parametrizations.weight.original1"] = v

            # GUARD 1: strict load — any missing/unexpected key is a mapping bug
            model.load_state_dict(mapped, strict=True)

            self._dtype = torch.float16 if (s.device == "cuda" and s.use_fp16) else torch.float32
            model = model.to(s.device).to(self._dtype).eval()

            head = torch.nn.Linear(cfg.hidden_size, 2)
            head.load_state_dict({"weight": raw["proj_fc.weight"],
                                  "bias": raw["proj_fc.bias"]}, strict=True)
            head = head.to(s.device).to(self._dtype).eval()

            self._model, self._head = model, head
            self.device = s.device
            self.available = True
            log.info("AntiDeepfake loaded on %s (%s, %d params remapped)",
                     s.device, self._dtype, len(mapped))
        except Exception as exc:
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.warning("AntiDeepfake unavailable: %s", self.load_error)

    # --- inference ---------------------------------------------------
    def _analyze(self, audio: np.ndarray, sr: int, ctx: dict[str, Any]) -> DetectorResult:
        import torch
        import torch.nn.functional as F

        x = np.asarray(audio, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            from ...ingest.audio import resample
            x = resample(x, sr, 16000)

        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
            # model-card preprocessing: zero-mean / unit-variance over the window
            t = F.layer_norm(t, t.shape)
            t = t.unsqueeze(0).to(self._dtype)
            feats = self._model(t).last_hidden_state          # [1, T, 1024]
            pooled = feats.transpose(1, 2).mean(dim=-1)       # [1, 1024]
            logits = self._head(pooled).float()[0]            # [2]
            probs = torch.softmax(logits, dim=0)
            spoof = float(probs[FAKE_INDEX])

        return DetectorResult(
            name=self.name, kind=self.kind, score=spoof, available=True,
            detail={
                "spoof_probability": round(spoof, 4),
                "fake_logit": round(float(logits[FAKE_INDEX]), 4),
                "real_logit": round(float(logits[REAL_INDEX]), 4),
                "model": self.model_id,
                "training_data": self.training_data,
                "licence": self.licence,
            },
            note="AntiDeepfake wav2vec2-large, post-trained on 74k h multi-corpus",
        )
