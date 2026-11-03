"""End-to-end behavior: several meetings, restarts and concurrent projects."""

import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from fakes import ASK, CONCLUDE, ScriptedModel, invoke, moderator_reply, new_graph
from meeting_engine import MeetingStore, build_meeting_graph


PARTICIPANTS = [
    {"id": "researcher", "name": "Researcher", "role": "develops strategies"},
    {"id": "risk", "name": "Risk analyst", "role": "identifies risks"},
    {"id": "owner", "name": "Haruto", "role": "project owner", "is_human": True},
]


@pytest.fixture
def store(tmp_path):
    return MeetingStore(str(tmp_path / "meetings.db"))


def next_item(summary):
    return moderator_reply("conclude_agenda", agenda_summary=summary)


def run(graph, store, thread_id, model, meeting_input, resumes=()):
    """Run a meeting to the end, answering each feedback question in turn."""

    config = {"configurable": {
        "thread_id": thread_id, "meeting_model": model, "meeting_store": store,
    }}
    result = invoke(graph, meeting_input, config)

    for answer in resumes:
        assert result.get("__interrupt__"), "expected a feedback question"
        result = invoke(graph, Command(resume={"feedback": answer}), config)

    assert not result.get("__interrupt__")
    return result


def test_trading_project_across_three_meetings(store):
    graph = new_graph()

    # Meeting 1: two agenda items, a question to the owner, open issues.
    ask_owner = moderator_reply(
        "request_user_feedback", question="Fixed or adaptive?", ask_participant_id="owner"
    )
    first = run(graph, store, "m1", ScriptedModel(replies={
        "moderator": [ask_owner, next_item("Adaptive preferred."), next_item("Costs open.")],
        "extraction": [json.dumps({
            "proposals": [{"ref": "p1", "description": "Adaptive trading frequency",
                           "created_by": "researcher", "agenda_id": "approach"}],
            "arguments": [{"content": "Drawdown may rise", "position": "oppose",
                           "target_id": "p1", "author_id": "risk", "agenda_id": "approach"}],
            "unresolved_issues": [
                {"description": "Which market variables drive frequency?", "agenda_id": "approach"},
                {"description": "Transaction costs are unevaluated", "agenda_id": "costs"},
            ],
        })],
    }), {
        "project_id": "trading-ai",
        "topic": "Reduce excessive trading frequency",
        "participants": PARTICIPANTS,
        "agenda": [
            {"id": "approach", "description": "Fixed versus adaptive frequency"},
            {"id": "costs", "description": "Cost impact"},
        ],
    }, resumes=["Adaptive, if drawdown stays under 5%."])

    record = first["meeting_record"]
    proposal_id = record["proposals"][0]["id"]
    variables_id, costs_id = [i["id"] for i in record["unresolved_issues"]]

    assert first["extraction_warnings"] == []
    assert record["arguments"][0]["target_id"] == proposal_id
    assert [i["status"] for i in record["agenda_items"]] == ["concluded", "concluded"]
    assert any(m.name == "owner" for m in first["messages"] if m.type == "human")

    # Meeting 2 starts from Meeting 1 and refines the policy.
    model = ScriptedModel(replies={
        "moderator": [CONCLUDE],
        "extraction": [json.dumps({
            "proposals": [{"description": "adaptive trading frequency"}],
            "decisions": [{"ref": "d1", "description": "Use volatility as the driver",
                           "status": "approved", "approved_by": "owner"}],
            "action_items": [{"description": "Backtest the volatility policy",
                              "owner_id": "researcher"}],
            "updates": [
                {"id": proposal_id, "status": "under_discussion"},
                {"id": variables_id, "status": "resolved", "note": "Volatility chosen"},
            ],
        })],
    })
    second = run(graph, store, "m2", model, {
        "project_id": "trading-ai",
        "topic": "Design the adaptive policy",
        "participants": PARTICIPANTS,
    })

    analysis = json.loads(model.calls_for("analysis")[0]["human"])["project_context"]
    assert [p["id"] for p in analysis["current_state"]["proposals"]] == [proposal_id]
    assert analysis["history"][0]["meeting_id"] == first["meeting_id"]
    assert second["meeting_record"]["proposals"] == []  # repeat was dropped
    decision_id = second["meeting_record"]["decisions"][0]["id"]

    # Meeting 3 reviews the backtest, replaces the decision and accepts.
    third = run(graph, store, "m3", ScriptedModel(replies={
        "moderator": [CONCLUDE],
        "extraction": [json.dumps({
            "decisions": [{"ref": "d2", "description": "Use volatility and spread",
                           "status": "approved", "approved_by": "owner"}],
            "updates": [
                {"id": decision_id, "status": "superseded", "superseded_by": "d2"},
                {"id": proposal_id, "status": "accepted"},
                {"id": costs_id, "status": "resolved"},
            ],
        })],
    }), {
        "project_id": "trading-ai",
        "topic": "Review backtest results",
        "participants": PARTICIPANTS,
    })

    assert third["extraction_warnings"] == []

    state = store.get_project_state("trading-ai")
    decisions = {d["id"]: d for d in state["decisions"]}
    new_decision_id = third["meeting_record"]["decisions"][0]["id"]

    assert [(p["id"], p["status"]) for p in state["proposals"]] == [(proposal_id, "accepted")]
    assert decisions[decision_id]["status"] == "superseded"
    assert decisions[decision_id]["superseded_by"] == new_decision_id
    assert decisions[new_decision_id]["approved_by"] == "owner"
    assert {i["status"] for i in state["unresolved_issues"]} == {"resolved"}
    assert state["action_items"][0]["owner_id"] == "researcher"
    assert len(store.list_meetings("trading-ai")) == 3

    # History keeps what each meeting saw at the time.
    assert store.get(second["meeting_id"])["decisions"][0]["status"] == "approved"


def test_paused_meeting_resumes_after_restart(tmp_path, store):
    checkpoints = tmp_path / "checkpoints.db"
    meeting_input = {"project_id": "trading-ai", "topic": "Adaptive trading frequency"}

    # First process: the meeting pauses for feedback.
    conn = sqlite3.connect(checkpoints, check_same_thread=False)
    graph = build_meeting_graph(SqliteSaver(conn))
    first_model = ScriptedModel(replies={"moderator": [ASK]})
    config = {"configurable": {
        "thread_id": "meeting-1", "meeting_model": first_model, "meeting_store": store,
    }}
    paused = graph.invoke(meeting_input, config)
    conn.close()

    assert paused["__interrupt__"][0].value["question"] == "Which approach?"
    assert store.list_meetings() == []

    # Second process: a new graph and model pick the meeting up.
    conn = sqlite3.connect(checkpoints, check_same_thread=False)
    graph = build_meeting_graph(SqliteSaver(conn))
    second_model = ScriptedModel(replies={"moderator": [CONCLUDE]})
    config["configurable"]["meeting_model"] = second_model
    result = graph.invoke(Command(resume={"feedback": "Adaptive."}), config)
    conn.close()

    assert result["meeting_saved"] is True
    assert result["meeting_id"] == paused["meeting_id"]
    assert [c["node"] for c in second_model.calls] == [
        "analysis", "deliberation", "moderator", "finalize", "extraction",
    ]
    feedback = json.loads(second_model.calls_for("analysis")[0]["human"])["user_feedback"]
    assert feedback == "Adaptive."
    assert store.get(result["meeting_id"]) is not None


def test_interleaved_projects_stay_isolated(store):
    graph = new_graph()

    def config(thread_id, model):
        return {"configurable": {
            "thread_id": thread_id, "meeting_model": model, "meeting_store": store,
        }}

    def extraction(text):
        return [json.dumps({"proposals": [{"description": text}]})]

    model_a = ScriptedModel(replies={"moderator": [ASK, CONCLUDE], "extraction": extraction("A idea")})
    model_b = ScriptedModel(replies={"extraction": extraction("B idea")})

    # A pauses; B runs to the end in between; then A resumes.
    paused = invoke(graph, {"project_id": "alpha", "topic": "A"}, config("a", model_a))
    done_b = invoke(graph, {"project_id": "beta", "topic": "B"}, config("b", model_b))
    done_a = invoke(graph, Command(resume={"feedback": "go"}), config("a", model_a))

    assert paused["__interrupt__"]
    assert done_a["topic"] == "A" and done_b["topic"] == "B"
    assert done_a["meeting_id"] != done_b["meeting_id"]

    def proposals(project_id):
        return [p["description"] for p in store.get_project_state(project_id)["proposals"]]

    assert proposals("alpha") == ["A idea"]
    assert proposals("beta") == ["B idea"]
    assert [r["topic"] for r in store.list_meetings("alpha")] == ["A"]


def test_empty_feedback_lets_the_meeting_continue():
    graph = new_graph()
    model = ScriptedModel(replies={"moderator": [ASK, CONCLUDE]})
    config = {"configurable": {"thread_id": "t", "meeting_model": model}}

    invoke(graph, {"topic": "Trading"}, config)
    result = invoke(graph, Command(resume={"feedback": ""}), config)

    assert result["user_feedback"] == ""
    assert result["current_phase"] == "completed"
    assert result["meeting_summary"] == "Report."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
