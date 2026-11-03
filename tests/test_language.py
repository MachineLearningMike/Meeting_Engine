import json

import pytest

from fakes import ScriptedModel, invoke, new_graph


NODES = ("analysis", "deliberation", "moderator", "finalize", "extraction")
STRUCTURED = {"moderator", "extraction"}


def run(**state):
    model = ScriptedModel()
    graph = new_graph()
    config = {"configurable": {"thread_id": "t", "meeting_model": model}}
    result = invoke(graph, {"topic": "取引頻度を減らす", **state}, config)
    return result, model


def system_prompt(model, node):
    return model.calls_for(node)[0]["system"]


def test_without_a_language_every_node_follows_the_meeting():
    _, model = run()

    for node in NODES:
        prompt = system_prompt(model, node)
        assert "LANGUAGE: Write in the language the meeting is held in" in prompt
        assert ("keep JSON keys" in prompt) == (node in STRUCTURED)


def test_a_meeting_language_applies_to_every_node():
    result, model = run(language="日本語")

    assert result["language"] == "日本語"
    for node in NODES:
        assert "LANGUAGE: Write all your output in 日本語." in system_prompt(model, node)


def test_blank_language_means_automatic():
    _, model = run(language="  ")

    assert "the language the meeting is held in" in system_prompt(model, "analysis")


def test_participants_with_a_language_are_shown_to_the_moderator():
    participants = [
        {"id": "risk", "name": "Risk analyst", "role": "risk"},
        {"id": "anna", "name": "Anna", "is_human": True, "language": "English"},
        {"id": "kenji", "name": "Kenji", "is_human": True},
    ]

    _, model = run(participants=participants)
    shown = json.loads(model.calls_for("moderator")[0]["human"])["participants"]

    assert [p.get("language") for p in shown] == [None, "English", None]
    assert "write question_for_user in their" in system_prompt(model, "moderator")


def test_non_ascii_text_reaches_the_model_unescaped():
    _, model = run(background="手数料が利益を減らしている。")

    human = model.calls_for("analysis")[0]["human"]
    assert "取引頻度を減らす" in human
    assert "手数料が利益を減らしている。" in human


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
