from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class Storage:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._migrate()

    def _columns(self, table: str) -> set[str]:
        rows = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row["name"]) for row in rows}

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                linked_prediction_id INTEGER,
                FOREIGN KEY(linked_prediction_id) REFERENCES predictions(id)
            );
            """
        )
        prediction_columns = self._columns("predictions")
        additions = {
            "positive_probability": "REAL NOT NULL DEFAULT 0",
            "neutral_probability": "REAL NOT NULL DEFAULT 0",
            "negative_probability": "REAL NOT NULL DEFAULT 0",
            "confidence_level": "TEXT NOT NULL DEFAULT 'LOW'",
        }
        for name, definition in additions.items():
            if name not in prediction_columns:
                self.connection.execute(f"ALTER TABLE predictions ADD COLUMN {name} {definition}")
        self.connection.commit()

    def add_prediction(
        self,
        model: str,
        text: str,
        sentiment: str,
        confidence: float,
        probabilities: Mapping[str, float] | None = None,
        confidence_level: str = "LOW",
    ) -> int:
        probabilities = probabilities or {}
        cursor = self.connection.execute(
            """
            INSERT INTO predictions(
                timestamp, model, text, sentiment, confidence,
                positive_probability, neutral_probability, negative_probability,
                confidence_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                model,
                text,
                sentiment,
                confidence,
                float(probabilities.get("positive", 0.0)),
                float(probabilities.get("neutral", 0.0)),
                float(probabilities.get("negative", 0.0)),
                confidence_level,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def add_note(self, title: str, body: str, linked_prediction_id: int | None = None) -> int:
        cursor = self.connection.execute(
            "INSERT INTO notes(timestamp, title, body, linked_prediction_id) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), title, body, linked_prediction_id),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def recent_predictions(self, limit: int = 250):
        return self.connection.execute(
            """
            SELECT id, timestamp, model, text, sentiment, confidence,
                   positive_probability, neutral_probability, negative_probability,
                   confidence_level
            FROM predictions ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def recent_notes(self, limit: int = 250):
        return self.connection.execute(
            "SELECT id, timestamp, title, body, linked_prediction_id FROM notes ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def execute_read_query(self, query: str):
        statement = query.strip().lower()
        if not (statement.startswith("select") or statement.startswith("pragma") or statement.startswith("with")):
            raise ValueError("Database tab is read-only. Use SELECT, WITH, or PRAGMA queries.")
        cursor = self.connection.execute(query)
        columns = [item[0] for item in cursor.description or []]
        rows = cursor.fetchall()
        return columns, rows


def export_records(records: Iterable[Mapping], path: str | Path, fmt: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in records]
    fmt = fmt.lower()
    if fmt == "json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    if fmt == "csv":
        fields = list(rows[0].keys()) if rows else []
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if fields:
                writer.writeheader()
                writer.writerows(rows)
        return path
    raise ValueError(f"Unsupported export format: {fmt}")


def export_table(columns: Sequence[str], rows: Sequence[Sequence], path: str | Path, fmt: str) -> Path:
    dictionaries = [dict(zip(columns, row)) for row in rows]
    return export_records(dictionaries, path, fmt)
