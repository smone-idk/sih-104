-- VoiceShield SQLite schema (§3 store/).
-- Privacy: no audio blobs live here. Voice profiles store the ECAPA embedding
-- only. See the Privacy Center page for the authoritative contents list.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Analysis sessions: one per upload / simulation / stream.
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,               -- uuid4
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    source          TEXT NOT NULL,                  -- 'upload' | 'simulation' | 'stream' | 'challenge'
    channel         TEXT NOT NULL,                  -- authorized channel label (§1)
    language        TEXT,                           -- 'en' | 'hi' | 'pa' | null(auto)
    scenario        TEXT,                           -- demo scenario key, if any
    profile_id      TEXT REFERENCES voice_profiles(id) ON DELETE SET NULL,
    telephony_degraded INTEGER NOT NULL DEFAULT 0,  -- §8 toggle state
    duration_s      REAL,
    final_score     REAL,                           -- 0..100
    final_band      TEXT,                           -- 'LOW' | 'MEDIUM' | 'HIGH'
    device          TEXT,                           -- 'cuda' | 'cpu'
    weights_json    TEXT,                           -- fusion weights used
    retain_audio    INTEGER NOT NULL DEFAULT 0
);

-- Per-window acoustic results (streaming, §4).
CREATE TABLE IF NOT EXISTS windows (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_start         REAL NOT NULL,                  -- seconds into the session
    t_end           REAL NOT NULL,
    is_speech       INTEGER NOT NULL DEFAULT 1,
    raw_score       REAL,                           -- per-window fused 0..100
    ema_score       REAL,                           -- smoothed 0..100
    synthetic_prob  REAL,
    speaker_sim     REAL,
    prosody_anomaly REAL,
    rms_energy      REAL,
    latencies_json  TEXT,                           -- {stage: ms}
    detectors_json  TEXT                            -- full DetectorResult list
);

-- ASR transcript segments (decoupled cadence, §4).
CREATE TABLE IF NOT EXISTS transcript_segments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_start         REAL NOT NULL,
    t_end           REAL NOT NULL,
    language        TEXT,
    text            TEXT NOT NULL,
    avg_logprob     REAL,
    no_speech_prob  REAL
);

-- Context-engine signals with the transcript span that triggered them (§5).
CREATE TABLE IF NOT EXISTS context_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_at            REAL NOT NULL,
    signal          TEXT NOT NULL,                  -- 'urgency' | 'secrecy' | 'authority' | 'transaction' | 'out_of_workflow' | 'pii_solicit'
    value           REAL NOT NULL,                  -- 0..1
    matched_span    TEXT,                           -- the exact quote
    extra_json      TEXT                            -- e.g. normalized amount
);

-- Enrolled voice profiles — EMBEDDING ONLY, never audio (§11.6, §12).
CREATE TABLE IF NOT EXISTS voice_profiles (
    id              TEXT PRIMARY KEY,               -- uuid4
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    display_name    TEXT NOT NULL,                  -- 'Rajesh Sharma — CFO'
    role            TEXT,
    embedding       BLOB NOT NULL,                  -- float32 ECAPA vector
    embedding_dim   INTEGER NOT NULL,
    n_enroll_clips  INTEGER NOT NULL DEFAULT 0,
    source_note     TEXT
);

-- Incident log (§7, §11.5). Every verification attempt writes a row.
CREATE TABLE IF NOT EXISTS incidents (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    session_id      TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    band            TEXT NOT NULL,
    score           REAL NOT NULL,
    action          TEXT NOT NULL,                  -- 'ALLOW' | 'VERIFY' | 'ESCALATE'
    summary         TEXT,
    payload_json    TEXT
);

-- Verification attempts (challenge-response + simulated methods, §7).
CREATE TABLE IF NOT EXISTS verifications (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    session_id      TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    incident_id     TEXT REFERENCES incidents(id) ON DELETE SET NULL,
    method          TEXT NOT NULL,                  -- 'challenge_response' | 'callback' | 'mfa' | 'supervisor'
    challenge_phrase TEXT,
    result          TEXT NOT NULL,                  -- 'pending' | 'passed' | 'failed'
    detail_json     TEXT
);

-- Mock approval workflow (§7.2).
CREATE TABLE IF NOT EXISTS approvals (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    description     TEXT NOT NULL,                  -- 'Vendor payment — ₹25,00,000'
    amount          REAL,
    currency        TEXT DEFAULT 'INR',
    state           TEXT NOT NULL DEFAULT 'pending',-- 'pending' | 'blocked' | 'approved' | 'rejected'
    session_id      TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    unlocked_by     TEXT REFERENCES verifications(id) ON DELETE SET NULL
);

-- Enterprise directory — DEMO DATA (§2 Tier C). Seeded from a fixture.
CREATE TABLE IF NOT EXISTS directory_contacts (
    id              TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    role            TEXT,
    phone           TEXT,
    known_contact   INTEGER NOT NULL DEFAULT 0,
    verified_identity INTEGER NOT NULL DEFAULT 0,
    prior_interactions INTEGER NOT NULL DEFAULT 0,
    trust_score     REAL NOT NULL DEFAULT 0.0       -- 0..1, higher = more trusted
);

-- Evaluation runs (§9). run_eval.py writes JSON; this indexes them.
CREATE TABLE IF NOT EXISTS eval_runs (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    n_samples       INTEGER NOT NULL,
    threshold       REAL NOT NULL,
    eer             REAL,
    condition       TEXT,                           -- 'clean' | 'telephony_8k' | 'all'
    metrics_json    TEXT NOT NULL,
    report_path     TEXT
);

CREATE INDEX IF NOT EXISTS ix_windows_session ON windows(session_id);
CREATE INDEX IF NOT EXISTS ix_segments_session ON transcript_segments(session_id);
CREATE INDEX IF NOT EXISTS ix_signals_session ON context_signals(session_id);
CREATE INDEX IF NOT EXISTS ix_incidents_band ON incidents(band);
