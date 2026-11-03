from langgraph.types import interrupt
from langchain_core.messages import HumanMessage


def find_human(state: dict, participant_id: str | None) -> dict | None:
    return next(
        (
            participant
            for participant in state.get("participants", [])
            if participant["id"] == participant_id and participant.get("is_human")
        ),
        None,
    )


def collect_human_feedback(state: dict) -> dict:

    question = state.get(
        "moderator_question",
        "What is your assessment of the discussion?",
    )

    asked = find_human(state, state.get("moderator_ask_participant_id"))

    response = interrupt({
        "type": "human_feedback",
        "question": question,
        "participant": (
            {"id": asked["id"], "name": asked.get("name", asked["id"])}
            if asked
            else None
        ),
        "agenda": state.get("current_agenda", ""),
        "latest_deliberation": state.get("latest_deliberation", ""),
    })

    if isinstance(response, dict):
        feedback = str(response.get("feedback", "")).strip()
        decline_future = bool(
            response.get("decline_future_questions", False)
        )
        # Whoever answered a general question may identify themselves.
        speaker = find_human(state, response.get("participant_id")) or asked
    else:
        feedback = str(response).strip()
        decline_future = False
        speaker = asked

    return {
        "user_feedback": feedback,
        "user_declined_feedback": decline_future,
        "pending_user_input": False,
        "current_phase": "analysis",
        "messages": [
            HumanMessage(
                content=(
                    f"[{speaker.get('name', speaker['id'])}] {feedback}"
                    if speaker
                    else feedback
                ),
                name=speaker["id"] if speaker else None,
            )
        ],
    }
