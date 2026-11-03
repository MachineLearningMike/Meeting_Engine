import uuid

from langchain_core.messages import SystemMessage, HumanMessage


def build_agenda(state: dict) -> list[dict]:
    """Build the meeting agenda.

    An explicit "agenda" list in the input is used as given. Otherwise every
    problem and then every objective becomes an agenda item, and a meeting
    with neither discusses its topic.
    """

    if state.get("agenda"):
        sources = [(item, "agenda") for item in state["agenda"]]
    else:
        sources = [(item, "problem") for item in state.get("problems", [])]
        sources += [(item, "objective") for item in state.get("objectives", [])]

    if not sources:
        sources = [
            ({"description": state.get("topic", "General discussion")}, "topic")
        ]

    return [
        {
            "id": item.get("id") or str(uuid.uuid4()),
            "description": item.get("description", ""),
            "source": source,
            "status": "active" if index == 0 else "pending",
            "round_count": 0,
            "summary": "",
        }
        for index, (item, source) in enumerate(sources)
    ]


def initialize_meeting(state: dict) -> dict:
    meeting_id = state.get("meeting_id") or str(uuid.uuid4())

    agenda = build_agenda(state)
    first = agenda[0]

    max_rounds = state.get("max_rounds_per_agenda", 3)
    # Allows one extra item's worth of rounds for revisiting earlier items.
    max_total_rounds = state.get(
        "max_total_rounds", max_rounds * (len(agenda) + 1)
    )

    agenda_lines = "\n".join(
        f"{index}. {item['description']}"
        for index, item in enumerate(agenda, start=1)
    )

    return {
        "meeting_id": meeting_id,
        "agenda": agenda,
        "agenda_complete": False,
        "current_agenda_id": first["id"],
        "current_agenda": first["description"],
        "current_phase": "orientation",
        "round_count": 0,
        "total_rounds": 0,
        "max_total_rounds": max_total_rounds,
        "moderator_action": "continue_discussion",
        "messages": [
            SystemMessage(
                content=(
                    f"Meeting started with {len(agenda)} agenda item(s).\n"
                    f"{agenda_lines}"
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {state.get('topic', '')}\n"
                    f"Background: {state.get('background', '')}\n"
                    f"Prior state: {state.get('prior_state', '')}\n"
                    f"Agenda:\n{agenda_lines}"
                )
            ),
        ],
    }
