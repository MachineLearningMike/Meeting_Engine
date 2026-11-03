import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..history import PROJECT_CONTEXT_RULES, prompt_context
from .common import agenda_overview, get_model, invoke_text


def deliberate(
    state: dict,
    config: RunnableConfig,
) -> dict:
    model = get_model(config)

    context = {
        "topic": state.get("topic", ""),
        "current_agenda": state.get("current_agenda", ""),
        "agenda": agenda_overview(state),
        "context_analysis": state.get("context_analysis", ""),
        "user_feedback": state.get("user_feedback", ""),
        "round_count": state.get("round_count", 0),
        "recent_messages": [
            message.content
            for message in state.get("messages", [])[-12:]
            if hasattr(message, "content")
        ],
    }

    system_prompt = """
You are a reasoning participant in a structured meeting.

Analyze the current agenda item and contribute substantive reasoning.
Focus on the current item; use summaries of other agenda items only as context.

Identify:
1. Main analysis.
2. Possible approaches.
3. Advantages and disadvantages.
4. Risks and uncertainties.
5. Questions requiring further discussion.
6. Suggested next step.

Incorporate human feedback.
Distinguish facts from assumptions.
Do not claim that a proposal has been accepted.
Do not invent decisions.

Be analytical and concise.
"""

    project_context = prompt_context(state.get("project_context"))

    if project_context:
        context["project_context"] = project_context
        system_prompt += PROJECT_CONTEXT_RULES

    contribution = invoke_text(
        model=model,
        system_prompt=system_prompt,
        node="deliberation",
        human_prompt=json.dumps(
            context,
            ensure_ascii=False,
            default=str,
            indent=2,
        ),
    )

    round_count = state.get("round_count", 0) + 1

    return {
        "latest_deliberation": contribution,
        "round_count": round_count,
        "total_rounds": state.get("total_rounds", 0) + 1,
        "current_phase": "deliberation",
        "messages": [
            AIMessage(
                content=(
                    f"[Deliberation, round {round_count}]\n\n"
                    f"{contribution}"
                )
            )
        ],
    }