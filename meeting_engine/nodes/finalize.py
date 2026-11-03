import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from .common import agenda_overview, get_model, invoke_text


def finalize_meeting(
    state: dict,
    config: RunnableConfig,
) -> dict:
    model = get_model(config)

    # Close the item under discussion when the meeting ends early.
    agenda = [dict(item) for item in state.get("agenda", [])]
    for item in agenda:
        if item.get("status") == "active":
            item["status"] = "concluded"
            item["round_count"] = state.get("round_count", 0)
            if state.get("agenda_summary"):
                item["summary"] = state["agenda_summary"]

    context = {
        "topic": state.get("topic", ""),
        "agenda": agenda_overview({"agenda": agenda}),
        "context_analysis": state.get("context_analysis", ""),
        "messages": [
            message.content
            for message in state.get("messages", [])
            if hasattr(message, "content")
        ],
    }

    system_prompt = """
You are the secretary of a structured human-AI meeting.

Produce a concise, accurate meeting report.

Use these sections:

1. Meeting objective
2. Background and established facts
3. Agenda items: for each, the discussion and its outcome
4. Proposals discussed
5. Decisions actually agreed upon
6. Action items
7. Unresolved issues
8. Recommended next steps

Important rules:
- Never invent decisions or agreements.
- Distinguish suggestions from accepted decisions.
- If no decision was reached, explicitly say so.
- Do not invent owners or deadlines for action items.
- Preserve important disagreements.
- Say which agenda items were deferred or not reached.
"""

    summary = invoke_text(
        model=model,
        system_prompt=system_prompt,
        node="finalize",
        human_prompt=json.dumps(
            context,
            ensure_ascii=False,
            default=str,
            indent=2,
        ),
    )

    return {
        "meeting_summary": summary,
        "agenda": agenda,
        "current_phase": "completed",
        "pending_user_input": False,
        "messages": [
            AIMessage(content=f"[Meeting Report]\n\n{summary}")
        ],
    }