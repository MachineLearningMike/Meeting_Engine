import pytest

from meeting_engine.nodes.common import parse_json_object
from meeting_engine.nodes.extract_artifacts import validate_artifacts
from meeting_engine.schemas import make_objective, make_problem, make_proposal


def test_make_helpers_assign_ids():
    problem = make_problem("Trading frequency is too high", priority="high")
    objective = make_objective("Reduce trades")
    proposal = make_proposal(problem["id"], "Investigate adaptive frequency")

    assert problem["id"] and objective["id"] and proposal["id"]
    assert len({problem["id"], objective["id"], proposal["id"]}) == 3
    assert problem["priority"] == "high"
    assert proposal["problem_id"] == problem["id"]


def test_missing_fields_become_empty_lists():
    result = validate_artifacts({})

    assert result == {
        "proposals": [],
        "arguments": [],
        "decisions": [],
        "action_items": [],
        "unresolved_issues": [],
    }


def test_defaults_and_ids_are_filled_in():
    result = validate_artifacts({
        "proposals": [{"description": "A"}],
        "decisions": [{"description": "B"}],
        "action_items": [{"description": "C"}],
        "unresolved_issues": [{"description": "D"}],
        "arguments": [{"content": "E"}],
    })

    assert result["proposals"][0]["status"] == "proposed"
    assert result["decisions"][0]["status"] == "provisional"
    assert result["action_items"][0]["status"] == "pending"
    assert result["unresolved_issues"][0]["status"] == "open"
    assert result["arguments"][0]["position"] == "neutral"
    assert result["arguments"][0]["target_type"] == "proposal"
    assert all(items[0]["id"] for items in result.values())


def test_existing_id_is_kept():
    result = validate_artifacts({"proposals": [{"id": "p-1", "description": "A"}]})

    assert result["proposals"][0]["id"] == "p-1"


def test_input_is_not_mutated():
    item = {"description": "A"}

    validate_artifacts({"proposals": [item]})

    assert item == {"description": "A"}


def test_arguments_drop_status_field():
    result = validate_artifacts({
        "arguments": [{"content": "E", "position": "oppose", "status": "x"}]
    })

    assert "status" not in result["arguments"][0]
    assert result["arguments"][0]["position"] == "oppose"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"proposals": {}}, "'proposals' must be a list"),
        ({"proposals": ["text"]}, r"'proposals\[0\]' must be a dictionary"),
        ({"proposals": [{"description": "  "}]}, "non-empty 'description'"),
        ({"arguments": [{"position": "support"}]}, "non-empty 'content'"),
        (
            {"proposals": [{"description": "A", "status": "approved"}]},
            "Invalid status 'approved' in 'proposals'",
        ),
        (
            {"arguments": [{"content": "A", "position": "maybe"}]},
            "Invalid position 'maybe' in 'arguments'",
        ),
        (
            {"arguments": [{"content": "A", "target_type": "agenda"}]},
            "Invalid argument target_type: agenda",
        ),
    ],
)
def test_invalid_artifacts_are_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        validate_artifacts(data)


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Here you go: {"a": 1} Thanks.',
    ],
)
def test_parse_json_object_accepts_common_wrappings(text):
    assert parse_json_object(text) == {"a": 1}


def test_parse_json_object_rejects_missing_object():
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_json_object("no json here")
