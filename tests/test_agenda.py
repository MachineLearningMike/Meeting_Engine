import json

import pytest
from langgraph.types import Command

from fakes import ASK, CONCLUDE, CONTINUE, ScriptedModel, moderator_reply
from meeting_engine import build_meeting_graph
from meeting_engine.nodes.extract_artifacts import resolve_extraction
from meeting_engine.nodes.initialization import build_agenda


AGENDA = [
    {"id": "a1", "description": "Fixed or adaptive frequency?"},
    {"id": "a2", "description": "Which market variables?"},
    {"id": "a3", "description": "How to evaluate?"},
]


def next_item(summary=""):
    return moderator_reply("conclude_agenda", agenda_summary=summary)


DEFER = moderator_reply("defer_agenda", agenda_summary="Needs data.")


def revisit(target):
    return moderator_reply("revisit_agenda", target_agenda_id=target)


def start(moderator, agenda=AGENDA, **state):
    model = ScriptedModel(replies={"moderator": list(moderator)})
    graph = build_meeting_graph()
    config = {"configurable": {"thread_id": "t", "meeting_model": model}}
    result = graph.invoke({"topic": "Trading", "agenda": agenda, **state}, config)
    return graph, config, result, model


def discussed(model):
    """Agenda item ids in the order deliberation rounds discussed them."""
    return [
        json.loads(call["human"])["current_agenda"]
        for call in model.calls_for("deliberation")
    ]


def statuses(result):
    return {item["id"]: item["status"] for item in result["agenda"]}


# --- building the agenda -----------------------------------------------------


def test_agenda_from_problems_then_objectives():
    agenda = build_agenda({
        "problems": [{"id": "p", "description": "Too many trades"}],
        "objectives": [{"description": "Lower costs"}],
    })

    assert [(i["description"], i["source"], i["status"]) for i in agenda] == [
        ("Too many trades", "problem", "active"),
        ("Lower costs", "objective", "pending"),
    ]
    assert agenda[0]["id"] == "p" and agenda[1]["id"]


def test_explicit_agenda_takes_precedence():
    agenda = build_agenda({"agenda": AGENDA, "problems": [{"description": "x"}]})

    assert [i["id"] for i in agenda] == ["a1", "a2", "a3"]
    assert {i["source"] for i in agenda} == {"agenda"}


def test_topic_is_the_agenda_when_nothing_else_is_given():
    agenda = build_agenda({"topic": "Trading"})

    assert [(i["description"], i["source"]) for i in agenda] == [("Trading", "topic")]


# --- moving through the agenda -----------------------------------------------


def test_items_are_discussed_in_order():
    _, _, result, model = start(
        [next_item("A done."), next_item("B done."), next_item("C done.")]
    )

    assert discussed(model) == [i["description"] for i in AGENDA]
    assert len(model.calls_for("analysis")) == 3
    assert statuses(result) == {"a1": "concluded", "a2": "concluded", "a3": "concluded"}
    assert [i["summary"] for i in result["agenda"]] == ["A done.", "B done.", "C done."]
    assert result["current_phase"] == "completed"

    finalize = json.loads(model.calls_for("finalize")[0]["human"])
    assert [i["summary"] for i in finalize["agenda"]] == ["A done.", "B done.", "C done."]


def test_round_count_is_per_item():
    _, _, result, model = start([CONTINUE, next_item(), next_item(), next_item()])

    rounds = [json.loads(c["human"])["round_count"] for c in model.calls_for("deliberation")]

    assert rounds == [0, 1, 0, 0]
    assert [i["round_count"] for i in result["agenda"]] == [2, 1, 1]
    assert result["total_rounds"] == 4


def test_deferred_item_is_skipped():
    _, _, result, _ = start([DEFER, next_item(), next_item()])

    assert statuses(result) == {"a1": "deferred", "a2": "concluded", "a3": "concluded"}
    assert result["agenda"][0]["summary"] == "Needs data."


def test_revisit_returns_to_an_earlier_item_then_resumes():
    _, _, result, model = start(
        [next_item(), revisit("a1"), next_item(), next_item(), next_item()]
    )

    descriptions = {i["id"]: i["description"] for i in AGENDA}
    assert discussed(model) == [descriptions[x] for x in ("a1", "a2", "a1", "a2", "a3")]
    assert set(statuses(result).values()) == {"concluded"}
    # A revisited item starts its round count again.
    assert json.loads(model.calls_for("deliberation")[2]["human"])["round_count"] == 0
    assert result["agenda"][0]["round_count"] == 1


@pytest.mark.parametrize("target", ["a3", "unknown", ""])
def test_revisit_needs_a_closed_item(target):
    # a3 is still pending, so it cannot be revisited.
    _, _, result, model = start([revisit(target), next_item(), next_item(), next_item()])

    assert discussed(model)[:2] == [AGENDA[0]["description"]] * 2
    assert result["moderator_target_agenda_id"] == ""


def test_conclude_meeting_ends_early_and_leaves_items_pending():
    _, _, result, model = start(
        [moderator_reply("conclude_meeting", agenda_summary="Settled.")]
    )

    assert statuses(result) == {"a1": "concluded", "a2": "pending", "a3": "pending"}
    assert result["agenda"][0]["summary"] == "Settled."
    assert "not reached" in model.calls_for("finalize")[0]["system"]


def test_round_cap_moves_to_the_next_item():
    _, _, result, model = start([CONTINUE] * 20, max_rounds_per_agenda=2)

    assert [i["round_count"] for i in result["agenda"]] == [2, 2, 2]
    assert result["total_rounds"] == 6
    assert len(discussed(model)) == 6


def test_total_cap_stops_endless_revisits():
    _, _, result, _ = start(
        [next_item(), revisit("a1")] * 50,
        agenda=AGENDA[:2],
        max_rounds_per_agenda=3,
    )

    # Default meeting-wide cap: 3 rounds x (2 items + 1).
    assert result["total_rounds"] == 9
    assert result["moderator_action"] == "conclude_meeting"
    assert result["current_phase"] == "completed"


def test_feedback_does_not_carry_over_to_the_next_item():
    graph, config, result, model = start([ASK, next_item(), next_item(), next_item()])

    result = graph.invoke(Command(resume={"feedback": "Adaptive."}), config)

    analyses = [json.loads(c["human"])["user_feedback"] for c in model.calls_for("analysis")]
    assert analyses == ["", "Adaptive.", "", ""]
    assert result["current_phase"] == "completed"


def test_single_item_meeting_still_ends_with_conclude_meeting():
    _, _, result, _ = start([CONCLUDE], agenda=AGENDA[:1])

    assert statuses(result) == {"a1": "concluded"}
    assert result["current_phase"] == "completed"


# --- artifacts per agenda item -----------------------------------------------


def empty_state():
    return {k: [] for k in ("proposals", "arguments", "decisions", "action_items", "unresolved_issues")}


def test_artifacts_keep_valid_agenda_ids_and_drop_unknown_ones():
    artifacts, _, warnings = resolve_extraction(
        {"proposals": [
            {"description": "A", "agenda_id": "a2"},
            {"description": "B", "agenda_id": "zz"},
            {"description": "C"},
        ]},
        empty_state(),
        "m-1",
        agenda_ids=["a1", "a2"],
    )

    assert [p.get("agenda_id") for p in artifacts["proposals"]] == ["a2", None, None]
    assert "unknown agenda_id 'zz'" in warnings[0]


def test_single_agenda_item_is_assigned_automatically():
    artifacts, _, _ = resolve_extraction(
        {"proposals": [{"description": "A"}]}, empty_state(), "m-1", agenda_ids=["a1"]
    )

    assert artifacts["proposals"][0]["agenda_id"] == "a1"


def test_record_holds_agenda_items_and_tagged_artifacts():
    extraction = json.dumps({"proposals": [{"description": "Adaptive", "agenda_id": "a1"}]})
    model = ScriptedModel(replies={
        "moderator": [next_item(), next_item(), next_item()],
        "extraction": [extraction],
    })
    graph = build_meeting_graph()
    config = {"configurable": {"thread_id": "t", "meeting_model": model}}

    result = graph.invoke({"topic": "Trading", "agenda": AGENDA}, config)
    record = result["meeting_record"]

    assert [i["id"] for i in record["agenda_items"]] == ["a1", "a2", "a3"]
    assert record["agenda"] == "; ".join(i["description"] for i in AGENDA)
    assert record["proposals"][0]["agenda_id"] == "a1"

    shown = json.loads(model.calls_for("extraction")[0]["human"])["agenda_items"]
    assert [i["id"] for i in shown] == ["a1", "a2", "a3"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
