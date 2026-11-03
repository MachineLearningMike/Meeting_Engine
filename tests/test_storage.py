import sqlite3

import pytest

from meeting_engine.storage import MeetingStore


@pytest.fixture
def store(tmp_path):
    return MeetingStore(str(tmp_path / "meetings.db"))


def make_record(meeting_id="m-1", project_id="trading-ai", **fields):
    return {
        "meeting_id": meeting_id,
        "project_id": project_id,
        "topic": "Adaptive trading frequency",
        "agenda": "Reduce excessive trading frequency",
        "proposals": [
            {
                "id": "proposal-001",
                "description": "Develop an adaptive trading policy",
                "status": "proposed",
            }
        ],
        "arguments": [],
        "decisions": [],
        "action_items": [],
        "unresolved_issues": [],
        **fields,
    }


def test_save_and_get_round_trip(store):
    record = make_record()

    store.save(record)

    assert store.get("m-1") == record


def test_get_unknown_meeting_returns_none(store):
    assert store.get("missing") is None


def test_save_requires_meeting_id(store):
    with pytest.raises(ValueError, match="meeting_id"):
        store.save(make_record(meeting_id=""))


def test_list_meetings_filters_by_project(store):
    store.save(make_record("m-1", "trading-ai"))
    store.save(make_record("m-2", "trading-ai"))
    store.save(make_record("m-3", "other-project"))

    trading = {r["meeting_id"] for r in store.list_meetings("trading-ai")}
    everything = {r["meeting_id"] for r in store.list_meetings()}

    assert trading == {"m-1", "m-2"}
    assert everything == {"m-1", "m-2", "m-3"}


def test_saving_again_updates_instead_of_duplicating(store, tmp_path):
    store.save(make_record(topic="Old topic"))
    store.save(make_record(topic="New topic"))

    meetings = store.list_meetings("trading-ai")

    assert len(meetings) == 1
    assert meetings[0]["topic"] == "New topic"

    with sqlite3.connect(tmp_path / "meetings.db") as conn:
        created, updated = conn.execute(
            "SELECT created_at, updated_at FROM meetings WHERE meeting_id = 'm-1'"
        ).fetchone()

    assert created <= updated


def test_empty_project_id_is_stored_as_null(store, tmp_path):
    store.save(make_record(project_id=""))

    with sqlite3.connect(tmp_path / "meetings.db") as conn:
        (project_id,) = conn.execute(
            "SELECT project_id FROM meetings WHERE meeting_id = 'm-1'"
        ).fetchone()

    assert project_id is None
    assert len(store.list_meetings()) == 1


def test_delete(store):
    store.save(make_record())

    assert store.delete("m-1") is True
    assert store.delete("m-1") is False
    assert store.get("m-1") is None
