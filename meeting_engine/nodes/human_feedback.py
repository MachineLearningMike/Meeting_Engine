from langgraph.types import interrupt
from langchain_core.messages import HumanMessage


def collect_human_feedback(state: dict) -> dict:

    question = state.get(
        "moderator_question",
        "What is your assessment of the discussion?",
    )

    asked = next(
        (
            participant
            for participant in state.get("participants", [])
            if participant["id"] == state.get("moderator_ask_participant_id")
        ),
        None,
    )

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
    else:
        feedback = str(response).strip()
        decline_future = False

    return {
        "user_feedback": feedback,
        "user_declined_feedback": decline_future,
        "pending_user_input": False,
        "current_phase": "analysis",
        "messages": [
            HumanMessage(
                content=(
                    f"[{asked.get('name', asked['id'])}] {feedback}"
                    if asked
                    else feedback
                ),
                name=asked["id"] if asked else None,
            )
        ],
    }