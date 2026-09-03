"""AntiDeepfake detector fixtures — with emphasis on the fairseq->HF key remap,
which is the part most likely to fail silently (§14, Phase 1.5b Task B)."""
from __future__ import annotations

import numpy as np
import pytest

from voiceshield.ml.detectors.synthetic_antideepfake import (
    FAKE_INDEX,
    REAL_INDEX,
    AntiDeepfakeDetector,
    remap_fairseq_to_hf,
)


@pytest.fixture(scope="module")
def adf() -> AntiDeepfakeDetector:
    d = AntiDeepfakeDetector()
    d.load()
    return d


# --- remap unit tests (no weights needed) --------------------------------
def test_remap_key_names():
    src = {
        "m_ssl.model.feature_extractor.conv_layers.0.0.weight": 1,
        "m_ssl.model.feature_extractor.conv_layers.6.2.1.bias": 2,
        "m_ssl.model.post_extract_proj.weight": 3,
        "m_ssl.model.layer_norm.weight": 4,
        "m_ssl.model.encoder.pos_conv.0.weight_v": 5,
        "m_ssl.model.encoder.layers.3.self_attn.k_proj.weight": 6,
        "m_ssl.model.encoder.layers.3.self_attn_layer_norm.bias": 7,
        "m_ssl.model.encoder.layers.3.fc1.weight": 8,
        "m_ssl.model.encoder.layers.3.fc2.bias": 9,
        "m_ssl.model.encoder.layers.3.final_layer_norm.weight": 10,
        "m_ssl.model.encoder.layer_norm.weight": 11,
        "m_ssl.model.mask_emb": 12,
        "proj_fc.weight": 99,          # head — not part of the backbone map
    }
    out = remap_fairseq_to_hf(src)
    assert out["feature_extractor.conv_layers.0.conv.weight"] == 1
    assert out["feature_extractor.conv_layers.6.layer_norm.bias"] == 2
    assert out["feature_projection.projection.weight"] == 3
    assert out["feature_projection.layer_norm.weight"] == 4
    assert out["encoder.pos_conv_embed.conv.weight_v"] == 5
    assert out["encoder.layers.3.attention.k_proj.weight"] == 6
    assert out["encoder.layers.3.layer_norm.bias"] == 7
    assert out["encoder.layers.3.feed_forward.intermediate_dense.weight"] == 8
    assert out["encoder.layers.3.feed_forward.output_dense.bias"] == 9
    assert out["encoder.layers.3.final_layer_norm.weight"] == 10
    assert out["encoder.layer_norm.weight"] == 11
    assert out["masked_spec_embed"] == 12
    assert "proj_fc.weight" not in out


def test_remap_drops_pretraining_only_tensors():
    src = {
        "m_ssl.model.quantizer.vars": 1,
        "m_ssl.model.project_q.weight": 2,
        "m_ssl.model.final_proj.bias": 3,
        "m_ssl.model.encoder.layer_norm.bias": 4,
    }
    out = remap_fairseq_to_hf(src)
    assert set(out) == {"encoder.layer_norm.bias"}


# --- loaded-model tests --------------------------------------------------
def test_loads_strictly(adf):
    """Guard 1: the detector only reports available if strict=True succeeded."""
    if not adf.available:
        pytest.skip(f"AntiDeepfake weights unavailable: {adf.load_error}")
    assert adf.kind == "pretrained"
    assert adf.feeds == "voice_authenticity"
    assert adf.contributes is True


def test_polarity_indices_are_explicit():
    # index 0 = fake/spoof, index 1 = real/bonafide (model card + verified on
    # ASVspoof2019 LA dev by scripts/validate_antideepfake.py)
    assert FAKE_INDEX == 0
    assert REAL_INDEX == 1


def test_bounds_and_determinism(adf, white_noise, sr):
    if not adf.available:
        pytest.skip("AntiDeepfake unavailable")
    a = adf.analyze(white_noise, sr)
    b = adf.analyze(white_noise.copy(), sr)
    assert 0.0 <= a.score <= 1.0
    assert a.score == pytest.approx(b.score, abs=1e-3)
    assert {"spoof_probability", "fake_logit", "real_logit"} <= set(a.detail)


def test_genuine_speech_scores_low(adf, real_speech, sr):
    """Real human speech must not be flagged as synthetic."""
    if not adf.available:
        pytest.skip("AntiDeepfake unavailable")
    assert adf.analyze(real_speech[: 4 * sr], sr).score < 0.2


def test_cloned_speech_scores_high(adf, sr):
    """The XTTS clone tier is the case AASIST cannot see (LIMITATIONS.md §3)."""
    from voiceshield.config import get_settings
    from voiceshield.ingest.audio import load_audio

    if not adf.available:
        pytest.skip("AntiDeepfake unavailable")
    clips = sorted((get_settings().demo_assets_dir / "cloned").glob("*.wav"))
    if not clips:
        pytest.skip("cloned tier not built")
    audio, asr = load_audio(clips[0])
    assert adf.analyze(audio[: 4 * asr], asr).score > 0.8


def test_separates_genuine_from_cloned(adf, real_speech, sr):
    from voiceshield.config import get_settings
    from voiceshield.ingest.audio import load_audio

    if not adf.available:
        pytest.skip("AntiDeepfake unavailable")
    clips = sorted((get_settings().demo_assets_dir / "cloned").glob("*.wav"))
    if not clips:
        pytest.skip("cloned tier not built")
    cloned, casr = load_audio(clips[0])
    genuine_score = adf.analyze(real_speech[: 4 * sr], sr).score
    cloned_score = adf.analyze(cloned[: 4 * casr], casr).score
    assert cloned_score > genuine_score
    assert cloned_score - genuine_score > 0.5
