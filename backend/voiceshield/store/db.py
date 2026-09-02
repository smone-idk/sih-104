"""Thin SQLite access layer. No ORM — the schema is small and the queries are
explicit. `init_db()` is idempotent and safe to call at startup.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ..config import get_settings

_SCHEMA = Path(__file__).with_name("schema.sql")


def connect() -> sqlite3.Connection:
    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(s.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> Path:
    s = get_settings()
    conn = connect()
    try:
        conn.executescript(_SCHEMA.read_text())
        conn.commit()
    finally:
        conn.close()
    return s.db_path


def table_summary() -> dict[str, int]:
    """Row counts per table — backs the Privacy Center 'what the DB contains' page."""
    conn = connect()
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )]
        return {n: conn.execute(f"SELECT count(*) FROM {n}").fetchone()[0] for n in names}
    finally:
        conn.close()
