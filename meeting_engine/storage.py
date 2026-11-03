
"""Persistent storage for completed meeting records."""

from __future__ import annotations

import json
import sqlite3

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MeetingStore:
    """SQLite-backed storage for completed meeting records."""

    def __init__(self, db_path: str = "data/meetings.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meetings (
                    meeting_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    topic TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_meetings_project
                ON meetings(project_id)
            """)

    @staticmethod
    def _serialize(record: dict[str, Any]) -> str:
        return json.dumps(
            record,
            ensure_ascii=False,
            default=lambda value: (
                value.isoformat()
                if isinstance(value, datetime)
                else str(value)
            ),
        )

    def save(self, record: dict[str, Any]) -> None:
        """Insert or update a meeting record."""

        meeting_id = record.get("meeting_id")
        if not meeting_id:
            raise ValueError("Meeting record must contain meeting_id.")

        topic = record.get("topic", "")
        project_id = record.get("project_id") or None

        now = datetime.now(timezone.utc).isoformat()
        record_json = self._serialize(record)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meetings (
                    meeting_id,
                    project_id,
                    topic,
                    created_at,
                    updated_at,
                    record_json
                )
                VALUES (?, ?, ?, ?, ?, ?)

                ON CONFLICT(meeting_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    topic = excluded.topic,
                    updated_at = excluded.updated_at,
                    record_json = excluded.record_json
                """,
                (
                    meeting_id,
                    project_id,
                    topic,
                    now,
                    now,
                    record_json,
                ),
            )

    def get(self, meeting_id: str) -> dict[str, Any] | None:
        """Retrieve one meeting record by ID."""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT record_json
                FROM meetings
                WHERE meeting_id = ?
                """,
                (meeting_id,),
            ).fetchone()

        if row is None:
            return None

        return json.loads(row["record_json"])

    def list_meetings(
        self,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List meeting records, optionally filtered by project."""

        with self._connect() as conn:
            if project_id is None:
                rows = conn.execute("""
                    SELECT record_json
                    FROM meetings
                    ORDER BY created_at DESC
                """).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT record_json
                    FROM meetings
                    WHERE project_id = ?
                    ORDER BY created_at DESC
                    """,
                    (project_id,),
                ).fetchall()

        return [
            json.loads(row["record_json"])
            for row in rows
        ]

    def delete(self, meeting_id: str) -> bool:
        """Delete a meeting record. Return whether it existed."""

        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM meetings WHERE meeting_id = ?",
                (meeting_id,),
            )

        return cursor.rowcount > 0
