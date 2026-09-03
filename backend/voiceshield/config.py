"""Central configuration for VoiceShield.

Every tunable lives here. `DEVICE` is the single flag that switches the whole
stack between CUDA and CPU — nothing else in the codebase should branch on
hardware directly; it should read `settings.device`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_device() -> str:
    """cuda if a usable GPU is visible, else cpu. Import torch lazily."""
    if os.environ.get("VOICESHIELD_DEVICE"):
        return os.environ["VOICESHIELD_DEVICE"]
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch not installed yet / broken CUDA
        return "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOICESHIELD_",
        env_file=os.environ.get("VOICESHIELD_ENV_FILE", str(REPO_ROOT / ".env")),
        extra="ignore",
    )

    # --- runtime ---
    device: str = Field(default_factory=_default_device)  # "cuda" | "cpu"
    host: str = "127.0.0.1"
    port: int = 8000
    offline: bool = True  # if True, model loaders never hit the network

    # --- paths ---
    model_cache_dir: Path = REPO_ROOT / "models"
    demo_assets_dir: Path = REPO_ROOT / "demo_assets"
    data_dir: Path = REPO_ROOT / "data"
    db_path: Path = REPO_ROOT / "data" / "voiceshield.sqlite"

    # --- privacy ---
    retain_audio: bool = False  # §12 — genuinely enforced in ingest/pipeline

    # --- streaming pipeline (§4) ---
    window_seconds: float = 4.0
    hop_seconds: float = 1.0
    sample_rate: int = 16000
    ema_alpha: float = 0.3
    asr_min_segment_seconds: float = 3.0
    asr_max_segment_seconds: float = 6.0

    #: fp16 for the SSL anti-spoofing backbone on CUDA (halves its VRAM)
    use_fp16: bool = True

    # --- VAD (§Phase 1.5 Task C) — "silero" (trained, default) | "energy" ---
    vad_backend: str = "silero"
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 100
    vad_speech_pad_ms: int = 30
    #: a window needs at least this fraction of speech frames to be scored
    window_speech_ratio: float = 0.25

    # --- fusion weights (§6) — expert-elicited priors, editable at runtime ---
    weight_voice_authenticity: float = 0.30
    weight_speaker_consistency: float = 0.20
    weight_prosody_anomaly: float = 0.10
    weight_caller_trust: float = 0.10
    weight_transaction_context: float = 0.20
    weight_behavioural_risk: float = 0.10

    # --- policy bands (§6) ---
    band_low_max: float = 40.0
    band_high_min: float = 70.0

    # --- model ids / weights ---
    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    whisper_model: str = "small"
    aasist_weights_url: str = (
        "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth"
    )

    @property
    def whisper_compute_type(self) -> str:
        return "float16" if self.device == "cuda" else "int8"

    @property
    def aasist_weights_path(self) -> Path:
        return self.model_cache_dir / "aasist" / "AASIST.pth"

    @property
    def ecapa_dir(self) -> Path:
        return self.model_cache_dir / "ecapa-voxceleb"

    @property
    def whisper_dir(self) -> Path:
        return self.model_cache_dir / f"faster-whisper-{self.whisper_model}"

    def fusion_weights(self) -> dict[str, float]:
        return {
            "voice_authenticity": self.weight_voice_authenticity,
            "speaker_consistency": self.weight_speaker_consistency,
            "prosody_anomaly": self.weight_prosody_anomaly,
            "caller_trust": self.weight_caller_trust,
            "transaction_context": self.weight_transaction_context,
            "behavioural_risk": self.weight_behavioural_risk,
        }

    def ensure_dirs(self) -> None:
        for p in (self.model_cache_dir, self.demo_assets_dir, self.data_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    if s.offline:
        # Belt-and-braces: no model loader should reach the network at the venue.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    return s
