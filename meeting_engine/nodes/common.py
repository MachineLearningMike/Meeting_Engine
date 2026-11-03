import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage


def get_model(config: dict):
    """Retrieve the chat model injected into the graph configuration."""
    configurable = config.get("configurable", {})
    model = configurable.get("meeting_model")

    if model is None:
        raise ValueError(
            "No meeting model supplied. "
            "Pass configurable={'meeting_model': model}."
        )

    return model


def get_store(config: dict):
    """Return the meeting store injected into the configuration, if any."""
    return config.get("configurable", {}).get("meeting_store")


def invoke_text(
    model: Any,
    system_prompt: str,
    human_prompt: str,
    node: str,
) -> str:
    """Invoke any LangChain-compatible chat model and return text.

    The call is tagged "meeting:<node>" so callbacks, tracing and test models
    can tell which meeting node made it.
    """

    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ],
        config={"tags": [f"meeting:{node}"]},
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