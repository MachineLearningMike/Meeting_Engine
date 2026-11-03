"""Persistent storage for meeting records and project state.

Two tables, kept deliberately separate:

- meetings: one row per completed meeting. These are historical events and
  are never rewritten by later meetings.
- artifacts: the current state of each project, one row per artifact. A
  meeting's new artifacts and status updates are applied here when the
  meeting is first saved.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .schemas import ARTIFACT_STATUS_FIELDS


class MeetingStorage(Protocol):
    """Storage interface the meeting engine depends on."""

    def save(self, record: dict[str, Any]) -> None: ...

    def get(self, meeting_id: str) -> dict[str, Any] | None: ...

    def list_meetings(
        self,
        project_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_project_state(self, project_id: str) -> dict[str, list[dict]]: ...


class MeetingStore:
    """SQLite-backed implementation of MeetingStorage."""

    def __init__(self, db_path: str = "data/meetings.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        """Open a connection, commit on success, roll back on error, close."""

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._transaction() as conn:
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

            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    project_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    status TEXT,
                    created_in_meeting TEXT,
                    updated_in_meeting TEXT,
                    updated_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (project_id, artifact_id)
                )
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
        """Insert or update a meeting record.

        The first time a meeting with a project_id is saved, its artifacts and
        status updates are applied to the project's current state. Saving the
        same meeting again only updates the meeting row, so it can neither
        duplicate artifacts nor undo changes made by later meetings.
        """

        meeting_id = record.get("meeting_id")
        if not meeting_id:
            raise ValueError("Meeting record must contain meeting_id.")

        topic = record.get("topic", "")
        project_id = record.get("project_id") or None

        now = datetime.now(timezone.utc).isoformat()
        record_json = self._serialize(record)

        with self._transaction() as conn:
            is_new = conn.execute(
                "SELECT 1 FROM meetings WHERE meeting_id = ?",
                (meeting_id,),
            ).fetchone() is None

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

            if is_new and project_id:
                self._apply_artifacts(conn, project_id, meeting_id, record, now)

    def _apply_artifacts(
        self,
        conn: sqlite3.Connection,
        project_id: str,
        meeting_id: str,
        record: dict[str, Any],
        now: str,
    ) -> None:
        for artifact_type, status_field in ARTIFACT_STATUS_FIELDS.items():
            for item in record.get(artifact_type, []):
                item = {
                    "created_in_meeting": meeting_id,
                    "updated_in_meeting": meeting_id,
                    **item,
                }

                conn.execute(
                    """
                    INSERT INTO artifacts (
                        project_id,
                        artifact_id,
                        artifact_type,
                        status,
                        created_in_meeting,
                        updated_in_meeting,
                        updated_at,
                        data_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, artifact_id) DO NOTHING
                    """,
                    (
                        project_id,
                        item["id"],
                        artifact_type,
                        item.get(status_field),
                        item["created_in_meeting"],
                        item["updated_in_meeting"],
                        now,
                        self._serialize(item),
                    ),
                )

        # Updates run after inserts so they may refer to this meeting's items.
        for update in record.get("artifact_updates", []):
            row = conn.execute(
                """
                SELECT artifact_type, data_json
                FROM artifacts
                WHERE project_id = ? AND artifact_id = ?
                """,
                (project_id, update["id"]),
            ).fetchone()

            if row is None:
                raise ValueError(
                    f"Update refers to unknown artifact '{update['id']}'."
                )

            if row["artifact_type"] != update["artifact_type"]:
                raise ValueError(
                    f"Artifact '{update['id']}' is a {row['artifact_type']} "
                    f"entry, not {update['artifact_type']}."
                )

            item = json.loads(row["data_json"])
            item[ARTIFACT_STATUS_FIELDS[row["artifact_type"]]] = update["status"]
            item["updated_in_meeting"] = meeting_id

            if update.get("superseded_by"):
                item["superseded_by"] = update["superseded_by"]

            conn.execute(
                """
                UPDATE artifacts
                SET status = ?,
                    updated_in_meeting = ?,
                    updated_at = ?,
                    data_json = ?
                WHERE project_id = ? AND artifact_id = ?
                """,
                (
                    update["status"],
                    meeting_id,
                    now,
                    self._serialize(item),
                    project_id,
                    update["id"],
                ),
            )

    def get(self, meeting_id: str) -> dict[str, Any] | None:
        """Retrieve one meeting record by ID."""

        with self._transaction() as conn:
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
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List meeting records newest first, optionally filtered by project."""

        query = "SELECT record_json FROM meetings"
        params: list[Any] = []

        if project_id is not None:
            query += " WHERE project_id = ?"
            params.append(project_id)

        query += " ORDER BY created_at DESC, rowid DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._transaction() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            json.loads(row["record_json"])
            for row in rows
        ]

    def get_project_state(self, project_id: str) -> dict[str, list[dict]]:
        """Return the project's artifacts with their latest status, by type."""

        state: dict[str, list[dict]] = {
            artifact_type: [] for artifact_type in ARTIFACT_STATUS_FIELDS
        }

        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT artifact_type, data_json
                FROM artifacts
                WHERE project_id = ?
                ORDER BY rowid
                """,
                (project_id,),
            ).fetchall()

        for row in rows:
            state[row["artifact_type"]].append(json.loads(row["data_json"]))

        return state

    def delete(self, meeting_id: str) -> bool:
        """Delete a meeting record. Return whether it existed.

        The project's current state is not rolled back.
        """

        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM meetings WHERE meeting_id = ?",
                (meeting_id,),
            )

        return cursor.rowcount > 0
