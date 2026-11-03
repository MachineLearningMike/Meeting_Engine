import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage


def get_model(config: dict, participant: dict | None = None):
    """Retrieve the chat model injected into the graph configuration.

    A participant with a "model" key uses
    configurable["participant_models"][key]; everything else uses
    configurable["meeting_model"].
    """
    configurable = config.get("configurable", {})

    model_key = (participant or {}).get("model")

    if model_key:
        model = configurable.get("participant_models", {}).get(model_key)

        if model is None:
            raise ValueError(
                f"Participant {participant.get('id')!r} uses model "
                f"{model_key!r}, which is not in "
                "configurable['participant_models']."
            )

        return model

    model = configurable.get("meeting_model")

    if model is None:
        raise ValueError(
            "No meeting model supplied. "
            "Pass configurable={'meeting_model': model}."
        )

    return model


def participants_overview(state: dict) -> list[dict]:
    """The participants in the compact form used in prompts."""
    return [
        {
            "id": participant["id"],
            "name": participant.get("name", participant["id"]),
            "role": participant.get("role", ""),
            "is_human": participant.get("is_human", False),
        }
        for participant in state.get("participants", [])
    ]


def agenda_overview(state: dict) -> list[dict]:
    """The agenda in the compact form used in prompts."""
    return [
        {
            "id": item["id"],
            "description": item.get("description", ""),
            "status": item.get("status", ""),
            "summary": item.get("summary", ""),
        }
        for item in state.get("agenda", [])
    ]


def get_store(config: dict):
    """Return the meeting store injected into the configuration, if any."""
    return config.get("configurable", {}).get("meeting_store")


def invoke_text(
    model: Any,
    system_prompt: str,
    human_prompt: str,
    node: str,
    participant: str | None = None,
) -> str:
    """Invoke any LangChain-compatible chat model and return text.

    The call is tagged "meeting:<node>" (and "participant:<id>" for a
    participant's turn) so callbacks, tracing and test models can tell which
    meeting node made it.
    """

    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ],
        config={"tags": [f"meeting:{node}"] + (
            [f"participant:{participant}"] if participant else []
        )},
    )

    content = response.content

    if isinstance(content, str):
        return content

    # Handle models that return content as a list of blocks.
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return str(content)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object, including one wrapped in a Markdown code block."""

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]

        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")

    result = json.loads(text[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    return result