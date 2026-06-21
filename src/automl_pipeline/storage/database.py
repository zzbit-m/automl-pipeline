from __future__ import annotations
from typing import Any
import sqlite3

from automl_pipeline.config import DB_PATH


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


def insert_job(filename: str, target: str, problem_type: str, holdout_score: float | None = None) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO jobs (filename, target, problem_type, holdout_score) VALUES (?, ?, ?, ?)",
        (filename, target, problem_type, holdout_score),
    )
    assert cur.lastrowid is not None
    job_id = cur.lastrowid
    conn.commit()
    conn.close()
    return job_id


def insert_model_result(job_id: int, model_name: str, score: float, std: float) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO model_results (job_id, model_name, score, std) VALUES (?, ?, ?, ?)",
        (job_id, model_name, score, std),
    )
    conn.commit()
    conn.close()


def insert_predictions(job_id: int, preds: list[dict[str, Any]]) -> None:
    conn = get_connection()
    conn.executemany(
        "INSERT INTO predictions (job_id, row_index, prediction) VALUES (?, ?, ?)",
        [(job_id, p["row_index"], p["prediction"]) for p in preds],
    )
    conn.commit()
    conn.close()


def get_jobs(limit: int = 10) -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_best_model(job_id: int) -> dict[str, Any] | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM model_results WHERE job_id = ? ORDER BY score DESC LIMIT 1",
        (job_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_predictions(job_id: int) -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT row_index, prediction FROM predictions WHERE job_id = ? ORDER BY row_index",
        (job_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_model_results(job_id: int) -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT model_name, score, std FROM model_results WHERE job_id = ? ORDER BY score DESC",
        (job_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
