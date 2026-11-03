from __future__ import annotations

import json
from typing import Any, get_args
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..schemas import (
    ActionItemStatus,
    ArgumentPosition,
    ArgumentTargetType,
    DecisionStatus,
    ProposalStatus,
    UnresolvedIssueStatus,
)
from .common import get_model, invoke_text, parse_json_object


ARTIFACT_FIELDS = {
    "proposals": {
        "required": ["description"],
        "status_field": "status",
        "statuses": set(get_args(ProposalStatus)),
        "default_status": "proposed",
    },
    "arguments": {
        "required": ["content"],
        "status_field": "position",
        "statuses": set(get_args(ArgumentPosition)),
        "default_status": "neutral",
    },
    "decisions": {
        "required": ["description"],
        "status_field": "status",
        "statuses": set(get_args(DecisionStatus)),
        "default_status": "provisional",
    },
    "action_items": {
        "required": ["description"],
        "status_field": "status",
        "statuses": set(get_args(ActionItemStatus)),
        "default_status": "pending",
    },
    "unresolved_issues": {
        "required": ["description"],
        "status_field": "status",
        "statuses": set(get_args(UnresolvedIssueStatus)),
        "default_status": "open",
    },
}

ARGUMENT_TARGET_TYPES = set(get_args(ArgumentTargetType))


def validate_artifacts(data: dict[str, Any]) -> dict[str, list[dict]]:
    """Validate and normalize extracted meeting artifacts."""

    result = {}

    for field, specification in ARTIFACT_FIELDS.items():
        items = data.get(field, [])

        if not isinstance(items, list):
            raise ValueError(f"'{field}' must be a list.")

        validated_items = []

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(
                    f"'{field}[{index}]' must be a dictionary."
                )

            for required_field in specification["required"]:
                value = item.get(required_field)

                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"'{field}[{index}]' requires "
                        f"a non-empty '{required_field}'."
                    )

            item = dict(item)

            item["id"] = item.get("id") or str(uuid4())

            status_field = specification["status_field"]
            status = item.get(status_field, specification["default_status"])

            if status not in specification["statuses"]:
                raise ValueError(
                    f"Invalid {status_field} '{status}' in '{field}'."
                )

            item[status_field] = status

            if field == "arguments":
                target_type = item.get("target_type", "proposal")

                if target_type not in ARGUMENT_TARGET_TYPES:
                    raise ValueError(
                        f"Invalid argument target_type: {target_type}"
                    )

                item["target_type"] = target_type
                item.pop("status", None)

            validated_items.append(item)

        result[field] = validated_items

    return result


def extract_meeting_artifacts(
    state: dict,
    config: RunnableConfig,
) -> dict:
    model = get_model(config)

    context = {
        "meeting_id": state.get("meeting_id", ""),
        "topic": state.get("topic", ""),
        "agenda": state.get("current_agenda", ""),
        "meeting_summary": state.get("meeting_summary", ""),
        "conversation": [
            message.content
            for message in state.get("messages", [])
            if hasattr(message, "content")
        ],
    }

    system_prompt = """
You are extracting structured records from a completed human-AI meeting.

Extract only information explicitly supported by the meeting.

Return exactly one JSON object with these five list fields:

{
  "proposals": [],
  "arguments": [],
  "decisions": [],
  "action_items": [],
  "unresolved_issues": []
}

SCHEMAS:

proposals:
{
  "description": "Proposal description",
  "status": "proposed",
  "created_by": "AI or human"
}

arguments:
{
  "target_id": "",
  "target_type": "proposal",
  "position": "support",
  "content": "Argument",
  "reasoning": ""
}

Valid target_type:
proposal, problem, decision

Valid position:
support, oppose, neutral

decisions:
{
  "description": "Decision description",
  "basis": "Reason for decision",
  "status": "provisional"
}

Valid status:
provisional, approved, rejected, superseded

action_items:
{
  "description": "Action description",
  "status": "pending",
  "owner_id": null
}

Valid status:
pending, in_progress, completed, cancelled

unresolved_issues:
{
  "description": "Issue description",
  "reason_unresolved": "",
  "next_step": "",
  "status": "open"
}

Valid status:
open, deferred, resolved

IMPORTANT RULES:

1. Never invent proposals, arguments, decisions, or action items.
2. A recommendation is NOT a decision.
3. A proposal is NOT an accepted proposal.
4. Only explicitly agreed decisions may be marked approved.
5. Do not invent owners or deadlines.
6. If no items exist in a category, return an empty list.
7. Do not include Markdown or explanations outside the JSON object.
"""

    raw = invoke_text(
        model=model,
        system_prompt=system_prompt,
        node="extraction",
        human_prompt=json.dumps(
            context,
            ensure_ascii=False,
            default=str,
            indent=2,
        ),
    )

    try:
        artifacts = validate_artifacts(parse_json_object(raw))
        extraction_error = ""
    except (ValueError, json.JSONDecodeError) as error:
        # Keep the completed meeting; report the failure instead of raising.
        artifacts = {field: [] for field in ARTIFACT_FIELDS}
        extraction_error = str(error)

    meeting_record = {
        "meeting_id": state.get("meeting_id", ""),
        "project_id": state.get("project_id", ""),
        "topic": state.get("topic", ""),
        "agenda": state.get("current_agenda", ""),
        **artifacts,
    }

    if extraction_error:
        meeting_record["extraction_error"] = extraction_error
        status_message = (
            f"[Artifact extraction failed: {extraction_error}]"
        )
    else:
        status_message = "[Structured meeting artifacts extracted and validated.]"

    return {
        **artifacts,
        "meeting_record": meeting_record,
        "extraction_error": extraction_error,
        "messages": [AIMessage(content=status_message)],
    }
