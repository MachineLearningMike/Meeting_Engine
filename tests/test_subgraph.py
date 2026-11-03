"""The meeting engine as a node inside a parent LangGraph application."""

import operator
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from fakes import ASK, CONCLUDE, ScriptedModel, new_graph
from meeting_engine import MeetingOutput, build_meeting_graph
from meeting_engine.storage import MeetingStore


class ParentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    # Shared with the meeting: inputs it reads, outputs it writes.
    project_id: str
    topic: str
    meeting_record: dict
    meeting_summary: str


def build_parent(meeting):
    def prepare(state):
        return {"log": ["prepare"], "topic": "Adaptive trading frequency"}

    def act_on_meeting(state):
        return {"log": [f"act on {state['meeting_record']['topic']}"]}

    builder = StateGraph(ParentState)
    builder.add_node("prepare", prepare)
    builder.add_node("meeting", meeting)
    builder.add_node("act", act_on_meeting)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "meeting")
    builder.add_edge("meeting", "act")
    builder.add_edge("act", END)
    return builder.compile(checkpointer=InMemorySaver())


def config(model, **configurable):
    return {"configurable": {"thread_id": "parent", "meeting_model": model, **configurable}}


def test_standalone_output_is_the_meeting_output_contract():
    graph = new_graph()
    cfg = config(ScriptedModel())

    result = graph.invoke({"topic": "Trading"}, cfg)

    assert set(result) <= set(MeetingOutput.__annotations__)
    assert {"meeting_id", "meeting_record", "meeting_summary"} <= set(result)
    assert "messages" not in result
    # The full internal state is still available.
    assert graph.get_state(cfg).values["messages"]


def test_input_outside_the_contract_is_ignored():
    graph = new_graph()
    cfg = config(ScriptedModel())

    graph.invoke({"topic": "Trading", "round_count": 99, "current_phase": "x"}, cfg)

    assert graph.get_state(cfg).values["round_count"] == 1


def test_meeting_without_interrupts_runs_without_a_checkpointer():
    graph = build_meeting_graph()

    result = graph.invoke({"topic": "Trading"}, config(ScriptedModel()))

    assert result["meeting_summary"] == "Report."


def test_resuming_feedback_needs_a_checkpointer_when_standalone():
    graph = build_meeting_graph()
    cfg = config(ScriptedModel(replies={"moderator": [ASK, CONCLUDE]}))

    assert graph.invoke({"topic": "Trading"}, cfg)["__interrupt__"]

    with pytest.raises(RuntimeError, match="without checkpointer"):
        graph.invoke(Command(resume={"feedback": "x"}), cfg)


def test_parent_runs_the_meeting_and_receives_its_record():
    parent = build_parent(build_meeting_graph())

    result = parent.invoke(
        {"messages": [HumanMessage("Parent conversation")]},
        config(ScriptedModel()),
    )

    assert result["log"] == ["prepare", "act on Adaptive trading frequency"]
    assert result["meeting_summary"] == "Report."
    assert result["meeting_record"]["topic"] == "Adaptive trading frequency"


def test_meeting_transcript_does_not_leak_into_parent_messages():
    parent = build_parent(build_meeting_graph())

    result = parent.invoke(
        {"messages": [HumanMessage("Parent conversation")]},
        config(ScriptedModel()),
    )

    assert [m.content for m in result["messages"]] == ["Parent conversation"]


def test_human_feedback_interrupts_and_resumes_through_the_parent():
    parent = build_parent(build_meeting_graph())
    model = ScriptedModel(replies={"moderator": [ASK, CONCLUDE]})
    cfg = config(model)

    paused = parent.invoke({}, cfg)

    assert paused["__interrupt__"][0].value["question"] == "Which approach?"
    assert paused["log"] == ["prepare"]

    result = parent.invoke(Command(resume={"feedback": "Adaptive."}), cfg)

    assert result["log"] == ["prepare", "act on Adaptive trading frequency"]
    assert len(model.calls_for("analysis")) == 2


def test_parent_config_reaches_the_meeting_store(tmp_path):
    store = MeetingStore(str(tmp_path / "meetings.db"))
    parent = build_parent(build_meeting_graph())

    result = parent.invoke(
        {"project_id": "trading-ai"},
        config(ScriptedModel(), meeting_store=store),
    )

    saved = store.list_meetings("trading-ai")
    assert [r["meeting_id"] for r in saved] == [result["meeting_record"]["meeting_id"]]


def test_parent_with_different_state_can_wrap_the_meeting():
    meeting = build_meeting_graph()

    class OtherState(TypedDict, total=False):
        question: str
        answer: str

    def run_meeting(state, config):
        output = meeting.invoke({"topic": state["question"]}, config)
        return {"answer": output["meeting_summary"]}

    builder = StateGraph(OtherState)
    builder.add_node("meeting", run_meeting)
    builder.add_edge(START, "meeting")
    builder.add_edge("meeting", END)
    parent = builder.compile(checkpointer=InMemorySaver())

    result = parent.invoke({"question": "Trade less?"}, config(ScriptedModel()))

    assert result == {"question": "Trade less?", "answer": "Report."}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
