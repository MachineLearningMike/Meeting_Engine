import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from fakes import CONCLUDE, CONTINUE, ScriptedModel, moderator_reply
from meeting_engine import build_meeting_graph
from meeting_engine.nodes.extract_artifacts import resolve_extraction


PARTICIPANTS = [
    {"id": "researcher", "name": "Researcher", "role": "researcher",
     "description": "Develops alternative strategies."},
    {"id": "risk", "name": "Risk analyst", "role": "risk analyst",
     "instructions": "Always quantify downside risk."},
    {"id": "owner", "name": "Haruto", "role": "project owner", "is_human": True},
]


def start(replies=None, participants=PARTICIPANTS, configurable=None, **state):
    model = ScriptedModel(replies=replies or {})
    graph = build_meeting_graph()
    config = {"configurable": {
        "thread_id": "t", "meeting_model": model, **(configurable or {})
    }}
    result = graph.invoke(
        {"topic": "Trading", "participants": participants, **state}, config
    )
    return graph, config, result, model


def test_each_ai_participant_speaks_once_per_round_in_order():
    _, _, result, model = start({"moderator": [CONTINUE, CONCLUDE]})

    speakers = [c["participant"] for c in model.calls_for("deliberation")]

    assert speakers == ["researcher", "risk", "researcher", "risk"]
    assert result["round_count"] == 2


def test_later_speakers_see_this_round_and_earlier_ones_do_not():
    _, _, _, model = start({
        "deliberation:researcher": ["Use adaptive frequency."],
        "deliberation:risk": ["Drawdown risk is high."],
    })

    first, second = [json.loads(c["human"]) for c in model.calls_for("deliberation")]

    assert first["this_round"] == []
    assert second["this_round"] == [{
        "participant_id": "researcher",
        "name": "Researcher",
        "contribution": "Use adaptive frequency.",
    }]
    assert [p["id"] for p in second["participants"]] == ["researcher", "risk", "owner"]


def test_role_description_and_instructions_reach_the_prompt():
    _, _, _, model = start()

    researcher, risk = [c["system"] for c in model.calls_for("deliberation")]

    assert "You are Researcher, the researcher." in researcher
    assert "Develops alternative strategies." in researcher
    assert "Always quantify downside risk." in risk
    assert "Always quantify" not in researcher


def test_contributions_are_recorded_per_participant():
    _, _, result, _ = start({
        "deliberation:researcher": ["Idea."],
        "deliberation:risk": ["Risk."],
    })

    named = [m for m in result["messages"] if isinstance(m, AIMessage) and m.name]

    assert [(m.name, m.content) for m in named] == [
        ("researcher", "[Researcher, round 1]\n\nIdea."),
        ("risk", "[Risk analyst, round 1]\n\nRisk."),
    ]
    assert result["latest_deliberation"] == "Researcher:\nIdea.\n\nRisk analyst:\nRisk."


def test_humans_only_and_no_participants_use_a_single_reasoner():
    for participants in ([PARTICIPANTS[2]], []):
        _, _, result, model = start(participants=participants)

        calls = model.calls_for("deliberation")
        assert len(calls) == 1 and calls[0]["participant"] is None
        assert "You are a reasoning participant" in calls[0]["system"]
        assert "this_round" not in calls[0]["human"]


def test_participants_can_use_their_own_models():
    risk_model = ScriptedModel(replies={"deliberation": ["From the risk model."]})
    participants = [PARTICIPANTS[0], {**PARTICIPANTS[1], "model": "careful"}]

    _, _, result, main_model = start(
        participants=participants,
        configurable={"participant_models": {"careful": risk_model}},
    )

    assert [c["participant"] for c in main_model.calls_for("deliberation")] == ["researcher"]
    assert [c["participant"] for c in risk_model.calls_for("deliberation")] == ["risk"]
    assert "From the risk model." in result["latest_deliberation"]


def test_unknown_participant_model_is_an_error():
    participants = [{**PARTICIPANTS[0], "model": "missing"}]

    with pytest.raises(ValueError, match="'missing', which is not in"):
        start(participants=participants)


def test_moderator_can_address_a_human_participant():
    ask = moderator_reply(
        "request_user_feedback", question="Is the drawdown acceptable?",
        ask_participant_id="owner",
    )
    graph, config, result, _ = start({"moderator": [ask, CONCLUDE]})

    payload = result["__interrupt__"][0].value
    assert payload["participant"] == {"id": "owner", "name": "Haruto"}

    result = graph.invoke(Command(resume={"feedback": "Yes, up to 5%."}), config)

    human = [m for m in result["messages"] if isinstance(m, HumanMessage) and m.name]
    assert [(m.name, m.content) for m in human] == [("owner", "[Haruto] Yes, up to 5%.")]
    assert result["user_feedback"] == "Yes, up to 5%."


@pytest.mark.parametrize("target", ["risk", "nobody"])
def test_questions_can_only_be_addressed_to_humans(target):
    ask = moderator_reply("request_user_feedback", question="?", ask_participant_id=target)

    _, _, result, _ = start({"moderator": [ask, CONCLUDE]})

    assert result["moderator_ask_participant_id"] == ""
    assert result["__interrupt__"][0].value["participant"] is None


def empty_state():
    return {k: [] for k in ("proposals", "arguments", "decisions", "action_items", "unresolved_issues")}


def test_extraction_checks_participant_references():
    artifacts, _, warnings = resolve_extraction(
        {
            "proposals": [
                {"description": "A", "created_by": "researcher"},
                {"description": "B", "created_by": "ghost"},
            ],
            "arguments": [{"content": "C", "author_id": "risk"}],
            "decisions": [
                {"description": "D", "approved_by": "owner"},
                {"description": "E", "approved_by": "risk"},
            ],
            "action_items": [{"description": "F", "owner_id": "owner"}],
        },
        empty_state(),
        "m-1",
        participants=PARTICIPANTS,
    )

    assert [p.get("created_by") for p in artifacts["proposals"]] == ["researcher", None]
    assert artifacts["arguments"][0]["author_id"] == "risk"
    assert [d.get("approved_by") for d in artifacts["decisions"]] == ["owner", None]
    assert artifacts["action_items"][0]["owner_id"] == "owner"
    assert len(warnings) == 2
    assert "not a participant" in warnings[0]
    assert "not a human participant" in warnings[1]


def test_without_participants_references_are_not_checked():
    artifacts, _, warnings = resolve_extraction(
        {"proposals": [{"description": "A", "created_by": "AI"}]}, empty_state(), "m-1"
    )

    assert artifacts["proposals"][0]["created_by"] == "AI"
    assert warnings == []


def test_extractor_and_moderator_see_participants():
    _, _, _, model = start()

    for node in ("moderator", "extraction", "analysis"):
        shown = json.loads(model.calls_for(node)[0]["human"])["participants"]
        assert [p["id"] for p in shown] == ["researcher", "risk", "owner"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
