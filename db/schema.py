import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pipeline.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            filename      TEXT NOT NULL,
            target        TEXT NOT NULL,
            problem_type  TEXT NOT NULL,
            holdout_score REAL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS model_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      INTEGER NOT NULL REFERENCES jobs(id),
            model_name  TEXT NOT NULL,
            score       REAL NOT NULL,
            std         REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      INTEGER NOT NULL REFERENCES jobs(id),
            row_index   INTEGER NOT NULL,
            prediction  REAL NOT NULL
        );
    """)
    try:
        conn.execute("ALTER TABLE jobs ADD COLUMN holdout_score REAL")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
