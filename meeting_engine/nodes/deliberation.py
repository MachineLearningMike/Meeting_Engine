import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..history import PROJECT_CONTEXT_RULES, prompt_context
from .common import (
    agenda_overview,
    get_model,
    invoke_text,
    participants_overview,
)


BASE_PROMPT = """
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


def role_prompt(participant: dict) -> str:
    lines = [
        "",
        f"You are {participant.get('name', participant['id'])}"
        + (f", the {participant['role']}." if participant.get("role") else "."),
    ]

    if participant.get("description"):
        lines.append(participant["description"])

    if participant.get("instructions"):
        lines.append(participant["instructions"])

    lines += [
        "Contribute from your role's perspective; do not try to cover every",
        "role. this_round holds what other participants said earlier in this",
        "round: build on it, challenge it, or add what it misses rather than",
        "repeating it.",
        "",
    ]

    return "\n".join(lines)


def deliberate(
    state: dict,
    config: RunnableConfig,
) -> dict:
    speakers = [
        participant
        for participant in state.get("participants", [])
        if not participant.get("is_human")
    ]

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

    system_prompt = BASE_PROMPT

    project_context = prompt_context(state.get("project_context"))

    if project_context:
        context["project_context"] = project_context
        system_prompt += PROJECT_CONTEXT_RULES

    round_count = state.get("round_count", 0) + 1
    progress = {
        "round_count": round_count,
        "total_rounds": state.get("total_rounds", 0) + 1,
        "current_phase": "deliberation",
    }

    # Without participants, a single unnamed reasoner deliberates.
    if not speakers:
        contribution = invoke_text(
            model=get_model(config),
            system_prompt=system_prompt,
            node="deliberation",
            human_prompt=json.dumps(
                context, ensure_ascii=False, default=str, indent=2
            ),
        )

        return {
            **progress,
            "latest_deliberation": contribution,
            "messages": [
                AIMessage(
                    content=(
                        f"[Deliberation, round {round_count}]\n\n"
                        f"{contribution}"
                    )
                )
            ],
        }

    context["participants"] = participants_overview(state)
    this_round: list[dict] = []
    messages = []

    for speaker in speakers:
        name = speaker.get("name", speaker["id"])

        contribution = invoke_text(
            model=get_model(config, speaker),
            system_prompt=system_prompt + role_prompt(speaker),
            node="deliberation",
            participant=speaker["id"],
            human_prompt=json.dumps(
                {**context, "this_round": this_round},
                ensure_ascii=False,
                default=str,
                indent=2,
            ),
        )

        this_round = this_round + [
            {"participant_id": speaker["id"], "name": name, "contribution": contribution}
        ]
        messages.append(
            AIMessage(
                content=f"[{name}, round {round_count}]\n\n{contribution}",
                name=speaker["id"],
            )
        )

    return {
        **progress,
        "latest_deliberation": "\n\n".join(
            f"{entry['name']}:\n{entry['contribution']}" for entry in this_round
        ),
        "messages": messages,
    }
