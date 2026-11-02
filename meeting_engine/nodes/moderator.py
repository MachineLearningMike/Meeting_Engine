import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from .common import get_model, invoke_text, parse_json_object


VALID_ACTIONS = {
    "continue_discussion",
    "request_user_feedback",
    "conclude_meeting",
}


def moderate(
    state: dict,
    config: RunnableConfig,
) -> dict:
    model = get_model(config)

    max_rounds = state.get("max_rounds_per_agenda", 3)
    round_count = state.get("round_count", 0)

    context = {
        "topic": state.get("topic", ""),
        "current_agenda": state.get("current_agenda", ""),
        "context_analysis": state.get("context_analysis", ""),
        "latest_deliberation": state.get("latest_deliberation", ""),
        "user_feedback": state.get("user_feedback", ""),
        "round_count": round_count,
        "max_rounds": max_rounds,
    }

    system_prompt = """
You are the moderator of a structured human-AI meeting.

Choose exactly one action:

1. continue_discussion
2. request_user_feedback
3. conclude_meeting

Request human feedback when human judgment or missing information
is important.

Continue when further reasoning is useful.

Conclude when sufficient progress has been made.

Never invent user approval or consensus.

If the maximum discussion rounds have been reached, do not continue.

Return ONLY a JSON object:

{
    "action": "continue_discussion",
    "reason": "Explanation",
    "question_for_user": ""
}
"""

    raw_response = invoke_text(
        model=model,
        system_prompt=system_prompt,
        human_prompt=json.dumps(
            context,
            ensure_ascii=False,
            default=str,
            indent=2,
        ),
    )

    try:
        result = parse_json_object(raw_response)

        action = result.get("action")

        if action not in VALID_ACTIONS:
            raise ValueError("Invalid moderator action")

    except (ValueError, json.JSONDecodeError):
        action = (
            "conclude_meeting"
            if round_count >= max_rounds
            else "continue_discussion"
        )

        result = {
            "reason": "Invalid moderator response.",
            "question_for_user": "",
        }

    if round_count >= max_rounds and action == "continue_discussion":
        action = "conclude_meeting"

    reason = result.get("reason", "")
    question = result.get("question_for_user", "")

    if action == "request_user_feedback" and not question:
        question = "What is your assessment of the discussion?"

    phase = {
        "continue_discussion": "deliberation",
        "request_user_feedback": "awaiting_user",
        "conclude_meeting": "finalization",
    }[action]

    return {
        "moderator_action": action,
        "moderator_reasoning": reason,
        "moderator_question": question,
        "pending_user_input": action == "request_user_feedback",
        "current_phase": phase,
        "messages": [
            AIMessage(
                content=(
                    f"[Moderator]\n"
                    f"Action: {action}\n"
                    f"Reason: {reason}\n"
                    f"Question: {question}"
                )
            )
        ],
    }