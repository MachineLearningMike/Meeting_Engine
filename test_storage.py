from meeting_engine.storage import MeetingStore


def main():
    store = MeetingStore("data/test_meetings.db")

    record = {
        "meeting_id": "test-001",
        "project_id": "trading-ai",
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
    }

    # Save
    store.save(record)
    print("Saved.")

    # Retrieve
    loaded = store.get("test-001")
    print("\nRetrieved:")
    print(loaded)

    assert loaded == record

    # List by project
    meetings = store.list_meetings("trading-ai")
    print("\nProject meetings:", len(meetings))

    assert len(meetings) == 1

    print("\nStorage test passed.")


if __name__ == "__main__":
    main()