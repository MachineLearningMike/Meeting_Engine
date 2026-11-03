import json

import pytest

from fakes import ScriptedModel
from meeting_engine import build_meeting_graph
from meeting_engine.history import compact_state, load_project_context
from meeting_engine.nodes.extract_artifacts import resolve_extraction
from meeting_engine.storage import MeetingStore


@pytest.fixture
def store(tmp_path):
    return MeetingStore(str(tmp_path / "meetings.db"))


def empty_state():
    return {
        "proposals": [],
        "arguments": [],
        "decisions": [],
        "action_items": [],
        "unresolved_issues": [],
    }


def record(meeting_id, project_id="trading-ai", **fields):
    return {
        "meeting_id": meeting_id,
        "project_id": project_id,
        "topic": "Adaptive trading frequency",
        "meeting_summary": f"Summary of {meeting_id}.",
        **empty_state(),
        "artifact_updates": [],
        **fields,
    }


def save_extraction(store, meeting_id, extraction, project_id="trading-ai"):
    """Resolve an extractor output against the project and save the meeting."""

    current = store.get_project_state(project_id)
    artifacts, updates, warnings = resolve_extraction(extraction, current, meeting_id)
    store.save(record(meeting_id, project_id, **artifacts, artifact_updates=updates))
    return artifacts, updates, warnings


# --- resolve_extraction ------------------------------------------------------


def test_new_items_get_provenance():
    artifacts, updates, warnings = resolve_extraction(
        {"proposals": [{"description": "Adaptive policy"}]}, empty_state(), "m-1"
    )

    proposal = artifacts["proposals"][0]
    assert proposal["created_in_meeting"] == "m-1"
    assert proposal["updated_in_meeting"] == "m-1"
    assert updates == [] and warnings == []


def test_repeated_item_is_dropped_and_its_ref_points_at_the_original():
    current = empty_state()
    current["proposals"] = [
        {"id": "p-old", "description": "Adaptive  POLICY", "status": "proposed"}
    ]

    artifacts, _, warnings = resolve_extraction(
        {
            "proposals": [{"ref": "p1", "description": "adaptive policy"}],
            "arguments": [{"content": "Fewer trades", "target_id": "p1"}],
        },
        current,
        "m-2",
    )

    assert artifacts["proposals"] == []
    assert artifacts["arguments"][0]["target_id"] == "p-old"
    assert "repeats existing artifact p-old" in warnings[0]


def test_item_reusing_an_existing_id_is_dropped():
    current = empty_state()
    current["proposals"] = [{"id": "p-old", "description": "A", "status": "proposed"}]

    artifacts, _, warnings = resolve_extraction(
        {"proposals": [{"id": "p-old", "description": "Different text"}]},
        current,
        "m-2",
    )

    assert artifacts["proposals"] == []
    assert len(warnings) == 1


def test_argument_targets_resolve_refs_and_drop_unknown_ids():
    artifacts, _, warnings = resolve_extraction(
        {
            "proposals": [{"ref": "p1", "description": "Adaptive policy"}],
            "arguments": [
                {"content": "Fewer trades", "target_id": "p1"},
                {"content": "Untargeted"},
                {"content": "Bad link", "target_id": "nope"},
            ],
        },
        empty_state(),
        "m-1",
    )

    proposal_id = artifacts["proposals"][0]["id"]
    targets = [a.get("target_id") for a in artifacts["arguments"]]

    assert targets == [proposal_id, None, ""]
    assert "ref" not in artifacts["proposals"][0]
    assert "unknown target_id 'nope'" in warnings[0]


def test_valid_update_is_kept_and_normalized():
    current = empty_state()
    current["proposals"] = [{"id": "p-1", "description": "A", "status": "proposed"}]

    _, updates, warnings = resolve_extraction(
        {"updates": [{"id": "p-1", "status": "accepted", "note": " Agreed "}]},
        current,
        "m-2",
    )

    assert updates == [
        {"id": "p-1", "artifact_type": "proposals", "status": "accepted", "note": "Agreed"}
    ]
    assert warnings == []


def test_update_that_changes_nothing_is_skipped():
    current = empty_state()
    current["proposals"] = [{"id": "p-1", "description": "A", "status": "proposed"}]

    _, updates, warnings = resolve_extraction(
        {"updates": [{"id": "p-1", "status": "proposed"}]}, current, "m-2"
    )

    assert updates == [] and warnings == []


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ("text", "must be a dictionary"),
        ({"id": "missing", "status": "accepted"}, "unknown artifact id 'missing'"),
        (
            {"id": "p-1", "artifact_type": "decisions", "status": "approved"},
            "'p-1' is in proposals, not decisions",
        ),
        ({"id": "p-1", "status": "approved"}, "invalid status 'approved' for proposals"),
        (
            {"id": "d-1", "status": "rejected", "superseded_by": "d-2"},
            "superseded_by requires status 'superseded'",
        ),
        (
            {"id": "d-1", "status": "superseded", "superseded_by": "zzz"},
            "superseded_by refers to unknown 'zzz'",
        ),
        (
            {"id": "d-1", "status": "superseded", "superseded_by": "p-1"},
            "must name a different decision",
        ),
    ],
)
def test_invalid_updates_are_dropped_with_a_warning(update, message):
    current = empty_state()
    current["proposals"] = [{"id": "p-1", "description": "A", "status": "proposed"}]
    current["decisions"] = [
        {"id": "d-1", "description": "D", "status": "approved"},
        {"id": "d-2", "description": "E", "status": "approved"},
    ]

    _, updates, warnings = resolve_extraction({"updates": [update]}, current, "m-2")

    assert updates == []
    assert message in warnings[0]


def test_updates_must_be_a_list():
    _, updates, warnings = resolve_extraction({"updates": {}}, empty_state(), "m-1")

    assert updates == []
    assert "'updates' must be a list" in warnings[0]


# --- storage: current state --------------------------------------------------


def test_project_state_starts_empty(store):
    assert store.get_project_state("trading-ai") == empty_state()


def test_saved_artifacts_become_current_state(store):
    save_extraction(store, "m-1", {"proposals": [{"description": "Adaptive policy"}]})

    state = store.get_project_state("trading-ai")

    assert [p["description"] for p in state["proposals"]] == ["Adaptive policy"]
    assert state["proposals"][0]["created_in_meeting"] == "m-1"


def test_later_meeting_updates_state_but_not_history(store):
    artifacts, _, _ = save_extraction(
        store, "m-1", {"proposals": [{"description": "Adaptive policy"}]}
    )
    proposal_id = artifacts["proposals"][0]["id"]

    save_extraction(
        store, "m-2", {"updates": [{"id": proposal_id, "status": "accepted"}]}
    )

    current = store.get_project_state("trading-ai")["proposals"]
    assert len(current) == 1
    assert current[0]["status"] == "accepted"
    assert current[0]["created_in_meeting"] == "m-1"
    assert current[0]["updated_in_meeting"] == "m-2"

    # Meeting 1's record still shows what was true at the time.
    assert store.get("m-1")["proposals"][0]["status"] == "proposed"


def test_superseded_decision_links_to_its_replacement(store):
    first, _, _ = save_extraction(
        store, "m-1",
        {"decisions": [{"description": "Trade hourly", "status": "approved"}]},
    )
    old_id = first["decisions"][0]["id"]

    second, _, warnings = save_extraction(
        store, "m-2",
        {
            "decisions": [
                {"ref": "d1", "description": "Trade adaptively", "status": "approved"}
            ],
            "updates": [
                {"id": old_id, "status": "superseded", "superseded_by": "d1"}
            ],
        },
    )
    new_id = second["decisions"][0]["id"]

    decisions = {d["id"]: d for d in store.get_project_state("trading-ai")["decisions"]}

    assert warnings == []
    assert decisions[old_id]["status"] == "superseded"
    assert decisions[old_id]["superseded_by"] == new_id
    assert decisions[new_id]["status"] == "approved"


def test_projects_are_isolated(store):
    save_extraction(store, "m-1", {"proposals": [{"description": "A"}]}, "trading-ai")
    save_extraction(store, "m-2", {"proposals": [{"description": "B"}]}, "other")

    trading = store.get_project_state("trading-ai")["proposals"]
    other = store.get_project_state("other")["proposals"]

    assert [p["description"] for p in trading] == ["A"]
    assert [p["description"] for p in other] == ["B"]


def test_resaving_a_meeting_neither_duplicates_nor_reverts(store):
    first = record("m-1", proposals=[{"id": "p-1", "description": "A", "status": "proposed"}])
    second = record("m-2", artifact_updates=[
        {"id": "p-1", "artifact_type": "proposals", "status": "accepted"}
    ])
    store.save(first)
    store.save(second)
    store.save(record("m-3", artifact_updates=[
        {"id": "p-1", "artifact_type": "proposals", "status": "rejected"}
    ]))

    store.save(first)
    store.save(second)

    proposals = store.get_project_state("trading-ai")["proposals"]
    assert len(proposals) == 1
    assert proposals[0]["status"] == "rejected"
    assert proposals[0]["updated_in_meeting"] == "m-3"


def test_meeting_without_project_does_not_touch_project_state(store):
    store.save(record("m-1", project_id="", proposals=[{"id": "p-1", "description": "A"}]))

    assert store.get("m-1") is not None
    assert store.get_project_state("") == empty_state()


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"id": "missing", "artifact_type": "proposals", "status": "accepted"}, "unknown artifact"),
        ({"id": "p-1", "artifact_type": "decisions", "status": "approved"}, "not decisions"),
    ],
)
def test_bad_update_rolls_back_the_whole_save(store, update, message):
    store.save(record("m-1", proposals=[{"id": "p-1", "description": "A", "status": "proposed"}]))

    with pytest.raises(ValueError, match=message):
        store.save(record(
            "m-2",
            proposals=[{"id": "p-2", "description": "B"}],
            artifact_updates=[update],
        ))

    assert store.get("m-2") is None
    assert [p["id"] for p in store.get_project_state("trading-ai")["proposals"]] == ["p-1"]


def test_list_meetings_limit_returns_newest(store):
    for meeting_id in ("m-1", "m-2", "m-3"):
        store.save(record(meeting_id))

    newest = store.list_meetings("trading-ai", limit=2)

    assert [r["meeting_id"] for r in newest] == ["m-3", "m-2"]


# --- history -----------------------------------------------------------------


def test_load_project_context_separates_state_and_history(store):
    for meeting_id in ("m-1", "m-2", "m-3", "m-4"):
        store.save(record(meeting_id))
    save_extraction(store, "m-5", {"unresolved_issues": [{"description": "Costs?"}]})

    context = load_project_context(store, "trading-ai", recent=3)

    assert context["project_id"] == "trading-ai"
    assert [h["meeting_id"] for h in context["history"]] == ["m-3", "m-4", "m-5"]
    assert context["history"][-1]["meeting_summary"] == "Summary of m-5."
    assert [i["description"] for i in context["current_state"]["unresolved_issues"]] == ["Costs?"]


def test_load_project_context_for_new_project(store):
    context = load_project_context(store, "new-project")

    assert context["history"] == []
    assert context["current_state"] == empty_state()


def test_compact_state_keeps_id_text_and_status():
    current = empty_state()
    current["proposals"] = [{"id": "p-1", "description": "A", "status": "proposed", "evidence": ["x"]}]
    current["arguments"] = [{"id": "a-1", "content": "B", "position": "support"}]

    compact = compact_state(current)

    assert compact["proposals"] == [{"id": "p-1", "text": "A", "status": "proposed"}]
    assert compact["arguments"] == [{"id": "a-1", "text": "B", "status": "support"}]
    assert compact["decisions"] == []


# --- across meetings, through the graph --------------------------------------


def run_meeting(store, extraction):
    """Run one meeting with the project's context and save its record."""

    model = ScriptedModel(replies={"extraction": [json.dumps(extraction)]})
    graph = build_meeting_graph()
    config = {"configurable": {"thread_id": "t", "meeting_model": model}}

    result = graph.invoke(
        {
            "project_id": "trading-ai",
            "topic": "Adaptive trading frequency",
            "project_context": load_project_context(store, "trading-ai"),
        },
        config,
    )
    store.save(result["meeting_record"])
    return result, model


def test_three_meetings_continue_the_same_work(store):
    # Meeting 1: a proposal and two open issues.
    first, _ = run_meeting(store, {
        "proposals": [{"ref": "p1", "description": "Adaptive trading frequency"}],
        "arguments": [{"content": "Fewer unnecessary trades", "target_id": "p1",
                       "position": "support"}],
        "unresolved_issues": [
            {"description": "Optimal policy is unknown"},
            {"description": "Transaction costs are unevaluated"},
        ],
    })
    proposal_id = first["meeting_record"]["proposals"][0]["id"]
    issue_ids = [i["id"] for i in first["meeting_record"]["unresolved_issues"]]
    assert first["meeting_record"]["arguments"][0]["target_id"] == proposal_id

    # Meeting 2 sees Meeting 1's work and moves it forward.
    second, model = run_meeting(store, {
        "proposals": [{"description": "adaptive trading frequency"}],
        "updates": [
            {"id": proposal_id, "status": "under_discussion"},
            {"id": issue_ids[1], "status": "resolved", "note": "Costs measured"},
        ],
    })
    shown = json.loads(model.calls_for("extraction")[0]["human"])["existing_artifacts"]
    assert [p["id"] for p in shown["proposals"]] == [proposal_id]
    assert len(shown["unresolved_issues"]) == 2
    assert second["meeting_record"]["proposals"] == []
    assert len(second["extraction_warnings"]) == 1

    # Meeting 3 decides.
    run_meeting(store, {
        "decisions": [{"description": "Adopt adaptive frequency", "status": "approved"}],
        "updates": [{"id": proposal_id, "status": "accepted"}],
    })

    state = store.get_project_state("trading-ai")
    assert [(p["id"], p["status"]) for p in state["proposals"]] == [(proposal_id, "accepted")]
    assert {i["id"]: i["status"] for i in state["unresolved_issues"]} == {
        issue_ids[0]: "open",
        issue_ids[1]: "resolved",
    }
    assert [d["description"] for d in state["decisions"]] == ["Adopt adaptive frequency"]
    assert len(store.list_meetings("trading-ai")) == 3
