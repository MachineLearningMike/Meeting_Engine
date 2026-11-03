"""Project memory: what earlier meetings established.

Two kinds of information are kept apart:

- current_state: every artifact of the project with its latest status. This is
  authoritative; a decision superseded in a later meeting shows "superseded".
- history: summaries of past meetings, oldest first. This is background only
  and may describe things that have since changed.
"""

from __future__ import annotations

from .schemas import ARTIFACT_STATUS_FIELDS, MeetingSummary, ProjectContext
from .storage import MeetingStorage


ARTIFACT_TEXT_FIELDS = {
    "proposals": "description",
    "arguments": "content",
    "decisions": "description",
    "action_items": "description",
    "unresolved_issues": "description",
}


def load_project_context(
    store: MeetingStorage,
    project_id: str,
    recent: int = 3,
) -> ProjectContext:
    """Load a project's current state and its most recent meeting summaries."""

    records = store.list_meetings(project_id, limit=recent)

    history: list[MeetingSummary] = [
        {
            "meeting_id": record.get("meeting_id", ""),
            "topic": record.get("topic", ""),
            "agenda": record.get("agenda", ""),
            "meeting_summary": record.get("meeting_summary", ""),
        }
        for record in reversed(records)
    ]

    return {
        "project_id": project_id,
        "current_state": store.get_project_state(project_id),
        "history": history,
    }


def compact_state(current_state: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Reduce artifacts to id, text and status, for use in prompts."""

    compact = {}

    for field, text_field in ARTIFACT_TEXT_FIELDS.items():
        status_field = ARTIFACT_STATUS_FIELDS[field]

        compact[field] = [
            {
                "id": item["id"],
                "text": item.get(text_field, ""),
                "status": item.get(status_field, ""),
            }
            for item in current_state.get(field, [])
        ]

    return compact
