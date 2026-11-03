import json

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from fakes import ASK, CONCLUDE, CONTINUE, ScriptedModel
from meeting_engine import build_meeting_graph


INITIAL_STATE = {
    "project_id": "trading-ai",
    "topic": "Adaptive trading frequency",
    "background": "The trading system generates too many trades.",
    "problems": [
        {"id": "problem-001", "description": "Trading frequency is too high."}
    ],
    "objectives": [],
    "max_rounds_per_agenda": 3,
}


def start(model, **overrides):
    """Start a meeting on a fresh graph and return (graph, config, result)."""

    graph = build_meeting_graph()
    config = {"configurable": {"thread_id": "test", "meeting_model": model}}
    result = graph.invoke({**INITIAL_STATE, **overrides}, config)
    return graph, config, result


def test_normal_meeting_completes():
    model = ScriptedModel()

    _, _, result = start(model)

    assert "__interrupt__" not in result
    assert result["current_phase"] == "completed"
    assert result["meeting_summary"] == "Report."
    assert result["round_count"] == 1
    assert result["current_agenda"] == "Trading frequency is too high."

    record = result["meeting_record"]
    assert record["meeting_id"] == result["meeting_id"]
    assert record["project_id"] == "trading-ai"
    assert record["topic"] == "Adaptive trading frequency"

    assert [call["node"] for call in model.calls] == [
        "analysis", "deliberation", "moderator", "finalize", "extraction",
    ]


def test_feedback_interrupt_and_resume():
    model = ScriptedModel(replies={"moderator": [ASK, CONCLUDE]})

    graph, config, result = start(model)

    assert result["__interrupt__"][0].value["question"] == "Which approach?"

    result = graph.invoke(
        Command(resume={"feedback": "I prefer adaptive frequency."}), config
    )

    assert "__interrupt__" not in result
    assert result["user_feedback"] == "I prefer adaptive frequency."
    assert result["round_count"] == 2
    assert any(
        isinstance(message, HumanMessage)
        and message.content == "I prefer adaptive frequency."
        for message in result["messages"]
    )

    # Feedback reaches the analysis and deliberation that follow it.
    second_analysis = json.loads(model.calls_for("analysis")[1]["human"])
    assert second_analysis["user_feedback"] == "I prefer adaptive frequency."


def test_resume_accepts_plain_string():
    model = ScriptedModel(replies={"moderator": [ASK, CONCLUDE]})

    graph, config, _ = start(model)
    result = graph.invoke(Command(resume="Plain answer"), config)

    assert result["user_feedback"] == "Plain answer"
    assert result["current_phase"] == "completed"


def test_declining_feedback_stops_questions():
    model = ScriptedModel(replies={"moderator": [ASK] * 10})

    graph, config, result = start(model, max_rounds_per_agenda=5)
    interrupts = 0

    while result.get("__interrupt__"):
        interrupts += 1
        result = graph.invoke(
            Command(resume={"feedback": "", "decline_future_questions": True}),
            config,
        )

    assert interrupts == 1
    assert result["current_phase"] == "completed"
    assert result["round_count"] == 5


def test_round_cap_bounds_feedback_requests():
    model = ScriptedModel(replies={"moderator": [ASK] * 10})

    graph, config, result = start(model, max_rounds_per_agenda=2)
    interrupts = 0

    while result.get("__interrupt__"):
        interrupts += 1
        result = graph.invoke(Command(resume={"feedback": "ok"}), config)

    assert interrupts == 1
    assert result["round_count"] == 2
    assert result["current_phase"] == "completed"


def test_round_cap_bounds_continued_discussion():
    model = ScriptedModel(replies={"moderator": [CONTINUE] * 10})

    _, _, result = start(model, max_rounds_per_agenda=3)

    assert result["round_count"] == 3
    assert result["moderator_action"] == "conclude_agenda"
    assert result["agenda"][0]["status"] == "concluded"


def test_invalid_moderator_output_continues_until_cap():
    model = ScriptedModel(replies={"moderator": ["not json"] * 10})

    _, _, result = start(model, max_rounds_per_agenda=2)

    assert result["round_count"] == 2
    assert result["current_phase"] == "completed"
    assert result["moderator_reasoning"] == "Invalid moderator response."


def test_extracted_artifacts_are_recorded():
    extraction = json.dumps({
        "proposals": [
            {"description": "Develop an adaptive policy.", "status": "proposed"}
        ],
        "arguments": [
            {"content": "Reduces unnecessary trades.", "position": "support"}
        ],
        "decisions": [],
        "action_items": [],
        "unresolved_issues": [
            {"description": "Evaluate transaction costs.", "status": "open"}
        ],
    })
    model = ScriptedModel(replies={"extraction": [extraction]})

    _, _, result = start(model)
    record = result["meeting_record"]

    assert not result["extraction_error"]
    assert [p["description"] for p in record["proposals"]] == [
        "Develop an adaptive policy."
    ]
    assert record["arguments"][0]["position"] == "support"
    assert record["unresolved_issues"][0]["status"] == "open"
    assert all(item["id"] for item in record["proposals"] + record["arguments"])
    assert result["proposals"] == record["proposals"]


def test_invalid_extraction_keeps_meeting():
    model = ScriptedModel(replies={"extraction": ["not json"]})

    _, _, result = start(model)
    record = result["meeting_record"]

    assert result["current_phase"] == "completed"
    assert result["meeting_summary"] == "Report."
    assert result["extraction_error"] == "No JSON object found in model response."
    assert record["extraction_error"] == result["extraction_error"]
    assert record["proposals"] == []


def test_missing_model_raises():
    graph = build_meeting_graph()

    with pytest.raises(ValueError, match="No meeting model supplied"):
        graph.invoke(INITIAL_STATE, {"configurable": {"thread_id": "test"}})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
