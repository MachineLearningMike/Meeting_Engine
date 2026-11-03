from langchain_core.runnables import RunnableConfig

import json

from ..history import PROJECT_CONTEXT_RULES, prompt_context
from .common import get_model, invoke_text


def analyze_context(
    state: dict,
    config: RunnableConfig,
) -> dict:
    model = get_model(config)

    context = {
        "topic": state.get("topic", ""),
        "background": state.get("background", ""),
        "prior_state": state.get("prior_state", ""),
        "objectives": state.get("objectives", []),
        "problems": state.get("problems", []),
        "current_agenda": state.get("current_agenda", ""),
        "user_feedback": state.get("user_feedback", ""),
        "recent_messages": [
            message.content
            for message in state.get("messages", [])[-12:]
            if hasattr(message, "content")
        ],
    }

    system_prompt = """
You are an analytical assistant participating in a structured meeting.

Analyze the provided meeting context.

Identify:
1. Established facts.
2. Assumptions that have not been verified.
3. Constraints.
4. Important missing information.
5. The central question that the meeting should resolve.

Do not invent facts.
Do not make decisions on behalf of participants.
Be precise and concise.
"""

    project_context = prompt_context(state.get("project_context"))

    if project_context:
        context["project_context"] = project_context
        system_prompt += PROJECT_CONTEXT_RULES

    analysis = invoke_text(
        model=model,
        system_prompt=system_prompt,
        node="analysis",
        human_prompt=json.dumps(context, ensure_ascii=False, default=str),
    )

    return {
        "context_analysis": analysis,
        "current_phase": "analysis",
    }