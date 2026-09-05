"""Shared fixtures. Synthetic signals are deterministic (fixed seed / closed
form) so detector tests have known-input/known-output behaviour without shipping
audio blobs (§14).

IMPORTANT: this module points the DB at a throwaway file BEFORE anything imports
`voiceshield.config`, whose Settings is lru_cached. Without this the API tests
write approvals, verifications and incidents straight into the demo database and
quietly corrupt the demo state.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path as _Path

_TEST_DB = _Path(tempfile.mkdtemp(prefix="voiceshield-test-")) / "test.sqlite"
os.environ["VOICESHIELD_DB_PATH"] = str(_TEST_DB)

import numpy as np      # noqa: E402
import pytest           # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _isolated_db():
    """Fail loudly if the real database ever gets wired into a test run."""
    from voiceshield.config import get_settings
    from voiceshield.store.db import init_db

    s = get_settings()
    assert str(s.db_path) == str(_TEST_DB), (
        f"tests must not touch the real DB (got {s.db_path})")
    init_db()
    yield
    for suffix in ("", "-wal", "-shm"):
        _Path(str(_TEST_DB) + suffix).unlink(missing_ok=True)

SR = 16000


def _tone(freq: float, dur: float, sr: int = SR, amp: float = 0.3,
          vibrato_hz: float = 0.0, vibrato_depth: float = 0.0) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    inst = freq + vibrato_depth * np.sin(2 * np.pi * vibrato_hz * t)
    phase = 2 * np.pi * np.cumsum(inst) / sr
    return (amp * np.sin(phase)).astype(np.float32)


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def pure_tone() -> np.ndarray:
    """4 s steady 180 Hz tone — maximally 'unnatural' prosody, near-linear phase."""
    return _tone(180.0, 4.0)


@pytest.fixture
def voiced_like() -> np.ndarray:
    """4 s tone with vibrato + formant-ish shaping + light noise — closer to
    natural voiced speech than a pure tone."""
    rng = np.random.default_rng(42)
    base = _tone(140.0, 4.0, amp=0.25, vibrato_hz=5.0, vibrato_depth=8.0)
    h2 = _tone(280.0, 4.0, amp=0.10, vibrato_hz=5.0, vibrato_depth=16.0)
    h3 = _tone(2100.0, 4.0, amp=0.04)
    env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3.0 * np.arange(len(base)) / SR))
    sig = (base + h2 + h3) * env + 0.005 * rng.standard_normal(len(base)).astype(np.float32)
    return (sig / (np.max(np.abs(sig)) + 1e-8) * 0.5).astype(np.float32)


@pytest.fixture
def white_noise() -> np.ndarray:
    rng = np.random.default_rng(7)
    return (0.1 * rng.standard_normal(SR * 4)).astype(np.float32)


@pytest.fixture
def silence() -> np.ndarray:
    return np.zeros(SR * 4, dtype=np.float32)


# --- real demo audio (skipped when the corpus has not been built) ----------
def _genuine_dir():
    from voiceshield.config import get_settings
    return get_settings().demo_assets_dir / "genuine"


def _find(prefix: str):
    d = _genuine_dir()
    if not d.exists():
        return None
    hits = sorted(d.glob(f"{prefix}*.wav"))
    return hits[0] if hits else None


@pytest.fixture
def enrolled_clip_path():
    """Longest enrolled clip, so window-timeline tests have several hops."""
    d = _genuine_dir()
    if not d.exists():
        pytest.skip("demo corpus not built — scripts/build_demo_assets.py --tier genuine")
    hits = sorted(d.glob("enrolled_1272*.wav"), key=lambda p: p.stat().st_size)
    if not hits:
        pytest.skip("demo corpus not built")
    return hits[-1]


@pytest.fixture
def other_clip_path():
    p = _find("genuine_1462")
    if p is None:
        pytest.skip("demo corpus not built")
    return p


@pytest.fixture
def real_speech(enrolled_clip_path):
    from voiceshield.ingest.audio import load_audio
    audio, _sr = load_audio(enrolled_clip_path)
    return audio
