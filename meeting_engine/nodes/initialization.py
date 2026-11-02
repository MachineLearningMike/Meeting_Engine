import uuid

from langchain_core.messages import SystemMessage, HumanMessage


def initialize_meeting(state: dict) -> dict:
    import uuid

    from langchain_core.messages import SystemMessage, HumanMessage

    meeting_id = state.get("meeting_id") or str(uuid.uuid4())

    problems = state.get("problems", [])
    objectives = state.get("objectives", [])

    if problems:
        agenda = problems[0]
        agenda_type = "problem"
    elif objectives:
        agenda = objectives[0]
        agenda_type = "objective"
    else:
        agenda = {
            "id": str(uuid.uuid4()),
            "description": state.get("topic", "General discussion"),
        }
        agenda_type = "general"

    agenda_id = agenda.get("id") or str(uuid.uuid4())
    agenda_description = agenda.get("description", "")

    return {
        "meeting_id": meeting_id,
        "current_agenda_id": agenda_id,
        "current_agenda": agenda_description,
        "current_phase": "orientation",
        "round_count": 0,
        "moderator_action": "continue_discussion",
        "messages": [
            SystemMessage(
                content=(
                    f"Meeting started. Agenda type: {agenda_type}. "
                    f"Agenda: {agenda_description}"
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {state.get('topic', '')}\n"
                    f"Background: {state.get('background', '')}\n"
                    f"Prior state: {state.get('prior_state', '')}\n"
                    f"Agenda: {agenda_description}"
                )
            ),
        ],
    }