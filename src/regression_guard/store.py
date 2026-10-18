"""SQLite persistence for eval runs.

One file, no server, git-ignorable. Every run is immutable once written; the
comparison layer reads history out of here rather than re-running anything.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .models import CaseResult, RunSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                  TEXT PRIMARY KEY,
    created_at              TEXT NOT NULL,
    prompt_version          TEXT NOT NULL,
    dataset_version         TEXT NOT NULL,
    model                   TEXT NOT NULL,
    provider                TEXT NOT NULL,
    git_sha                 TEXT NOT NULL DEFAULT '',
    total_cases             INTEGER NOT NULL,
    passed_cases            INTEGER NOT NULL,
    pass_rate               REAL NOT NULL,
    category_accuracy       REAL NOT NULL,
    mean_judge_score        REAL NOT NULL,
    composite_score         REAL NOT NULL,
    p50_latency_ms          REAL NOT NULL,
    p95_latency_ms          REAL NOT NULL,
    total_tokens            INTEGER NOT NULL,
    error_count             INTEGER NOT NULL,
    per_category_accuracy   TEXT NOT NULL,
    per_difficulty_pass_rate TEXT NOT NULL,
    schema_version          INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS case_results (
    run_id              TEXT NOT NULL,
    case_id             TEXT NOT NULL,
    expected_category   TEXT NOT NULL,
    predicted_category  TEXT,
    predicted_summary   TEXT NOT NULL DEFAULT '',
    category_match      INTEGER NOT NULL,
    judge_score         INTEGER,
    judge_reason        TEXT NOT NULL DEFAULT '',
    passed              INTEGER NOT NULL,
    latency_ms          REAL NOT NULL DEFAULT 0,
    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
    completion_tokens   INTEGER NOT NULL DEFAULT 0,
    error               TEXT,
    difficulty          TEXT NOT NULL DEFAULT 'easy',
    assertion_failures  TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (run_id, case_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_prompt_version ON runs(prompt_version);
CREATE INDEX IF NOT EXISTS idx_case_results_run ON case_results(run_id);
"""


class RunStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Bring a database written by an older schema up to date in place.

        Run history is the baseline, so an upgrade must never require
        deleting it. Columns are only ever added, with a default that reads
        correctly for the old rows.
        """
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(case_results)")}
        if "assertion_failures" not in columns:
            conn.execute("ALTER TABLE case_results ADD COLUMN assertion_failures TEXT NOT NULL DEFAULT '[]'")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_run(self, summary: RunSummary, results: list[CaseResult]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs VALUES
                (:run_id, :created_at, :prompt_version, :dataset_version, :model, :provider,
                 :git_sha, :total_cases, :passed_cases, :pass_rate, :category_accuracy,
                 :mean_judge_score, :composite_score, :p50_latency_ms, :p95_latency_ms,
                 :total_tokens, :error_count, :per_category_accuracy, :per_difficulty_pass_rate,
                 :schema_version)
                """,
                {
                    **summary.model_dump(mode="json"),
                    "per_category_accuracy": json.dumps(summary.per_category_accuracy),
                    "per_difficulty_pass_rate": json.dumps(summary.per_difficulty_pass_rate),
                },
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO case_results
                (run_id, case_id, expected_category, predicted_category, predicted_summary,
                 category_match, judge_score, judge_reason, passed, latency_ms,
                 prompt_tokens, completion_tokens, error, difficulty, assertion_failures)
                VALUES
                (:run_id, :case_id, :expected_category, :predicted_category, :predicted_summary,
                 :category_match, :judge_score, :judge_reason, :passed, :latency_ms,
                 :prompt_tokens, :completion_tokens, :error, :difficulty, :assertion_failures)
                """,
                [
                    {
                        **r.model_dump(mode="json"),
                        "run_id": summary.run_id,
                        "assertion_failures": json.dumps(
                            [a.model_dump(mode="json") for a in r.assertion_failures]
                        ),
                    }
                    for r in results
                ],
            )

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> RunSummary:
        data = dict(row)
        data["per_category_accuracy"] = json.loads(data["per_category_accuracy"])
        data["per_difficulty_pass_rate"] = json.loads(data["per_difficulty_pass_rate"])
        return RunSummary.model_validate(data)

    def get_run(self, run_id: str) -> RunSummary | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_summary(row) if row else None

    def get_case_results(self, run_id: str) -> list[CaseResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM case_results WHERE run_id = ? ORDER BY case_id", (run_id,)
            ).fetchall()
        out: list[CaseResult] = []
        for row in rows:
            data = dict(row)
            data.pop("run_id", None)
            data["category_match"] = bool(data["category_match"])
            data["passed"] = bool(data["passed"])
            data["assertion_failures"] = json.loads(data["assertion_failures"])
            out.append(CaseResult.model_validate(data))
        return out

    def recent_runs(self, limit: int = 20, prompt_version: str | None = None) -> list[RunSummary]:
        query = "SELECT * FROM runs"
        params: list[object] = []
        if prompt_version:
            query += " WHERE prompt_version = ?"
            params.append(prompt_version)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def latest_baseline(
        self,
        exclude_run_id: str | None = None,
        prompt_version: str | None = None,
        model: str | None = None,
    ) -> RunSummary | None:
        """Pick the run the current run should be diffed against.

        Not simply 'the previous run'. If it were, re-running a failing PR
        would make the failing run its own baseline, the delta would be zero,
        and the second attempt would go green - the pipeline would reward
        pressing retry. So the baseline is the most recent run of a *different*
        configuration (prompt version + model), which is the last thing that
        was actually accepted. Re-runs of the same config are therefore
        idempotent: they keep comparing against the same prior config.

        Falls back to the most recent other run when no different configuration
        has ever been recorded, and to None on an empty database.
        """
        with self._connect() as conn:
            if prompt_version is not None:
                row = conn.execute(
                    """
                    SELECT * FROM runs
                    WHERE run_id IS NOT ?
                      AND NOT (prompt_version = ? AND model IS ?)
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (exclude_run_id, prompt_version, model),
                ).fetchone()
                if row:
                    return self._row_to_summary(row)
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id IS NOT ? ORDER BY created_at DESC LIMIT 1",
                (exclude_run_id,),
            ).fetchone()
        return self._row_to_summary(row) if row else None
