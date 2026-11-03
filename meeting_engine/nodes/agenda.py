"""Move between agenda items after the moderator closes, defers or revisits one."""

from langchain_core.messages import AIMessage


def advance_agenda(state: dict) -> dict:
    agenda = [dict(item) for item in state.get("agenda", [])]
    by_id = {item["id"]: item for item in agenda}

    action = state.get("moderator_action")
    current = by_id.get(state.get("current_agenda_id", ""))
    summary = state.get("agenda_summary", "")

    if current is not None:
        current["round_count"] = state.get("round_count", 0)
        if summary:
            current["summary"] = summary

    if action == "revisit_agenda":
        if current is not None:
            # Resume this item after the revisit.
            current["status"] = "pending"
        next_item = by_id[state["moderator_target_agenda_id"]]
        next_item["round_count"] = 0
    else:
        if current is not None:
            current["status"] = (
                "deferred" if action == "defer_agenda" else "concluded"
            )
        next_item = next(
            (item for item in agenda if item["status"] == "pending"), None
        )

    if next_item is None:
        return {
            "agenda": agenda,
            "agenda_complete": True,
        }

    next_item["status"] = "active"

    return {
        "agenda": agenda,
        "agenda_complete": False,
        "current_agenda_id": next_item["id"],
        "current_agenda": next_item["description"],
        "round_count": next_item["round_count"],
        "user_feedback": "",
        "messages": [
            AIMessage(
                content=f"[Agenda] Now discussing: {next_item['description']}"
            )
        ],
    }
