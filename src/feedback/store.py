"""SQLite-backed feedback store for prediction corrections.

In production this would be PostgreSQL. Locally we use SQLite
for zero-dependency persistence.
"""

import sqlite3
import time
from pathlib import Path

from src.config import settings


class FeedbackStore:
    """Stores user feedback on predictions for future model improvement."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.feedback_db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    predicted_is_claim BOOLEAN NOT NULL,
                    correct_is_claim BOOLEAN NOT NULL,
                    comment TEXT DEFAULT '',
                    created_at REAL NOT NULL
                )
            """)

    def add(self, text: str, predicted: bool, correct: bool, comment: str = "") -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO feedback (text, predicted_is_claim, correct_is_claim, comment, created_at) VALUES (?, ?, ?, ?, ?)",
                (text, predicted, correct, comment, time.time()),
            )
            return cursor.lastrowid

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

    def recent(self, limit: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
