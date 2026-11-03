import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..history import PROJECT_CONTEXT_RULES, prompt_context
from .common import agenda_overview, get_model, invoke_text, parse_json_object


VALID_ACTIONS = {
    "continue_discussion",
    "request_user_feedback",
    "conclude_agenda",
    "defer_agenda",
    "revisit_agenda",
    "conclude_meeting",
}

PHASES = {
    "continue_discussion": "deliberation",
    "request_user_feedback": "awaiting_user",
    "conclude_agenda": "agenda_transition",
    "defer_agenda": "agenda_transition",
    "revisit_agenda": "agenda_transition",
    "conclude_meeting": "finalization",
}


def moderate(
    state: dict,
    config: RunnableConfig,
) -> dict:
    model = get_model(config)

    max_rounds = state.get("max_rounds_per_agenda", 3)
    round_count = state.get("round_count", 0)
    max_total_rounds = state.get("max_total_rounds", max_rounds)
    total_rounds = state.get("total_rounds", round_count)
    current_id = state.get("current_agenda_id", "")

    context = {
        "topic": state.get("topic", ""),
        "current_agenda_id": current_id,
        "current_agenda": state.get("current_agenda", ""),
        "agenda": agenda_overview(state),
        "context_analysis": state.get("context_analysis", ""),
        "latest_deliberation": state.get("latest_deliberation", ""),
        "user_feedback": state.get("user_feedback", ""),
        "round_count": round_count,
        "max_rounds": max_rounds,
    }

    system_prompt = """
You are the moderator of a structured human-AI meeting.

The meeting works through the agenda one item at a time. Choose exactly one
action:

1. continue_discussion: further reasoning on the current item is useful.
2. request_user_feedback: human judgment or missing information is needed.
3. conclude_agenda: the current item has made sufficient progress; move on.
4. defer_agenda: the current item cannot progress in this meeting; move on.
5. revisit_agenda: return to an earlier concluded or deferred item, named by
   target_agenda_id, because new information affects it.
6. conclude_meeting: end the whole meeting now.

Never invent user approval or consensus.

round_count and max_rounds apply to the current agenda item. When the
maximum has been reached, do not continue the current item.

For conclude_agenda, defer_agenda and conclude_meeting, give agenda_summary:
one or two sentences on where the current item stands.

Return ONLY a JSON object:

{
    "action": "continue_discussion",
    "reason": "Explanation",
    "question_for_user": "",
    "agenda_summary": "",
    "target_agenda_id": ""
}
"""

    project_context = prompt_context(state.get("project_context"))

    if project_context:
        context["project_context"] = project_context
        system_prompt += PROJECT_CONTEXT_RULES

    raw_response = invoke_text(
        model=model,
        system_prompt=system_prompt,
        node="moderator",
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
        action = "continue_discussion"
        result = {"reason": "Invalid moderator response."}

    target_id = str(result.get("target_agenda_id") or "")

    if action == "revisit_agenda":
        revisitable = {
            item["id"]
            for item in state.get("agenda", [])
            if item.get("status") in ("concluded", "deferred")
            and item["id"] != current_id
        }
        if target_id not in revisitable:
            action = "continue_discussion"

    if (
        action == "request_user_feedback"
        and state.get("user_declined_feedback", False)
    ):
        action = "continue_discussion"

    # The per-item limit also bounds feedback requests, which re-enter
    # deliberation of the same item.
    if round_count >= max_rounds and action in (
        "continue_discussion",
        "request_user_feedback",
    ):
        action = "conclude_agenda"

    # The meeting-wide limit bounds revisits.
    if total_rounds >= max_total_rounds:
        action = "conclude_meeting"

    reason = result.get("reason", "")
    question = result.get("question_for_user", "")
    agenda_summary = result.get("agenda_summary", "")

    if action == "request_user_feedback" and not question:
        question = "What is your assessment of the discussion?"

    if action != "revisit_agenda":
        target_id = ""

    return {
        "moderator_action": action,
        "moderator_reasoning": reason,
        "moderator_question": question,
        "moderator_target_agenda_id": target_id,
        "agenda_summary": agenda_summary,
        "pending_user_input": action == "request_user_feedback",
        "current_phase": PHASES[action],
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
