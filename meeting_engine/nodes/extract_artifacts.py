from __future__ import annotations

import json
from typing import Any, get_args
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..history import compact_state
from ..schemas import (
    ALLOWED_TRANSITIONS,
    ARTIFACT_STATUS_FIELDS,
    ActionItemStatus,
    ArgumentPosition,
    ArgumentTargetType,
    ArtifactUpdate,
    DecisionStatus,
    ProposalStatus,
    UnresolvedIssueStatus,
    check_transition,
)
from .common import (
    get_model,
    invoke_text,
    parse_json_object,
    participants_overview,
)


ARTIFACT_FIELDS = {
    "proposals": {
        "required": ["description"],
        "statuses": set(get_args(ProposalStatus)),
        "default_status": "proposed",
    },
    "arguments": {
        "required": ["content"],
        "statuses": set(get_args(ArgumentPosition)),
        "default_status": "neutral",
    },
    "decisions": {
        "required": ["description"],
        "statuses": set(get_args(DecisionStatus)),
        "default_status": "provisional",
    },
    "action_items": {
        "required": ["description"],
        "statuses": set(get_args(ActionItemStatus)),
        "default_status": "pending",
    },
    "unresolved_issues": {
        "required": ["description"],
        "statuses": set(get_args(UnresolvedIssueStatus)),
        "default_status": "open",
    },
}

ARGUMENT_TARGET_TYPES = set(get_args(ArgumentTargetType))

# One retry, told what was wrong, before extraction is reported as failed.
MAX_EXTRACTION_ATTEMPTS = 2

# Fields naming a participant. Decisions may only be approved by a human.
PARTICIPANT_FIELDS = {
    "proposals": "created_by",
    "arguments": "author_id",
    "decisions": "approved_by",
    "action_items": "owner_id",
}


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

            status_field = ARTIFACT_STATUS_FIELDS[field]
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


LIFECYCLE_RULES = "\nALLOWED STATUS CHANGES FOR EXISTING ITEMS:\n\n" + "\n".join(
    f"{field}: {current} -> {', '.join(sorted(targets))}"
    for field, transitions in ALLOWED_TRANSITIONS.items()
    for current, targets in transitions.items()
) + "\n\nAny other status is final and must not change.\n"


def _normalize(text: Any) -> str:
    return " ".join(str(text).lower().split())


def _text_field(field: str) -> str:
    return ARTIFACT_FIELDS[field]["required"][0]


def _validate_update(
    update: Any,
    existing: dict[str, tuple[str, dict]],
    refs: dict[str, tuple[str, str]],
) -> ArtifactUpdate | None:
    """Validate one status update. Return None when it changes nothing."""

    if not isinstance(update, dict):
        raise ValueError("must be a dictionary")

    artifact_id = update.get("id")

    if artifact_id not in existing:
        raise ValueError(f"unknown artifact id {artifact_id!r}")

    field, item = existing[artifact_id]
    given_type = update.get("artifact_type", field)

    if given_type != field:
        raise ValueError(f"{artifact_id!r} is in {field}, not {given_type}")

    status = update.get("status")

    if status not in ARTIFACT_FIELDS[field]["statuses"]:
        raise ValueError(f"invalid status {status!r} for {field}")

    result: ArtifactUpdate = {
        "id": artifact_id,
        "artifact_type": field,
        "status": status,
    }

    note = update.get("note")
    if isinstance(note, str) and note.strip():
        result["note"] = note.strip()

    check_transition(field, item.get(ARTIFACT_STATUS_FIELDS[field]), status)

    superseded_by = update.get("superseded_by")

    if superseded_by:
        if status != "superseded":
            raise ValueError("superseded_by requires status 'superseded'")

        if superseded_by in refs:
            target_field, target_id = refs[superseded_by]
        elif superseded_by in existing:
            target_field, target_id = existing[superseded_by][0], superseded_by
        else:
            raise ValueError(f"superseded_by refers to unknown {superseded_by!r}")

        if target_field != "decisions" or target_id == artifact_id:
            raise ValueError("superseded_by must name a different decision")

        result["superseded_by"] = target_id

    if (
        status == item.get(ARTIFACT_STATUS_FIELDS[field])
        and "superseded_by" not in result
    ):
        return None

    return result


def resolve_extraction(
    data: dict[str, Any],
    current_state: dict[str, list[dict]],
    meeting_id: str,
    agenda_ids: list[str] | None = None,
    participants: list[dict] | None = None,
) -> tuple[dict[str, list[dict]], list[ArtifactUpdate], list[str]]:
    """Turn the extractor's output into new artifacts, updates and warnings.

    New items may carry a temporary "ref" so that updates and arguments in the
    same output can point at them. New items that repeat an existing artifact
    are dropped, and their ref points at the existing artifact instead.
    Invalid updates and unknown argument targets are dropped with a warning
    rather than failing the whole extraction.

    With agenda_ids, each new item's agenda_id must be one of them; with a
    single agenda item, items without one are assigned to it.

    With participants, authors and owners must be participant ids, and
    approved_by must be a human participant. Other values are removed.
    """

    artifacts = validate_artifacts(data)
    warnings: list[str] = []

    existing = {
        item["id"]: (field, item)
        for field, items in current_state.items()
        for item in items
    }
    existing_by_text = {
        (field, _normalize(item.get(_text_field(field), ""))): item["id"]
        for field, items in current_state.items()
        for item in items
    }

    refs: dict[str, tuple[str, str]] = {}

    for field, items in artifacts.items():
        kept = []

        for item in items:
            ref = item.pop("ref", None)
            text = item[_text_field(field)]
            existing_id = (
                item["id"]
                if item["id"] in existing
                else existing_by_text.get((field, _normalize(text)))
            )

            if existing_id:
                warnings.append(
                    f"Dropped new {field} item that repeats existing "
                    f"artifact {existing_id}: {text!r}"
                )
                if ref:
                    refs[str(ref)] = (field, existing_id)
                continue

            if agenda_ids:
                agenda_id = item.get("agenda_id")

                if not agenda_id and len(agenda_ids) == 1:
                    item["agenda_id"] = agenda_ids[0]
                elif agenda_id and agenda_id not in agenda_ids:
                    warnings.append(
                        f"New {field} item {text!r} has unknown agenda_id "
                        f"{agenda_id!r}; it was removed."
                    )
                    item.pop("agenda_id")

            person_field = PARTICIPANT_FIELDS.get(field)
            person = item.get(person_field) if person_field else None

            if participants and person:
                allowed = {
                    p["id"]
                    for p in participants
                    if field != "decisions" or p.get("is_human")
                }
                if person not in allowed:
                    warnings.append(
                        f"New {field} item {text!r} has {person_field} "
                        f"{person!r}, which is not "
                        + ("a human participant" if field == "decisions" else "a participant")
                        + "; it was removed."
                    )
                    item.pop(person_field)

            item["created_in_meeting"] = meeting_id
            item["updated_in_meeting"] = meeting_id

            if ref:
                refs[str(ref)] = (field, item["id"])

            kept.append(item)

        artifacts[field] = kept

    for argument in artifacts["arguments"]:
        target = argument.get("target_id")

        if not target or target in existing:
            continue

        if target in refs:
            argument["target_id"] = refs[target][1]
        else:
            warnings.append(
                f"Argument {argument['id']} has unknown target_id {target!r}; "
                "the link was removed."
            )
            argument["target_id"] = ""

    raw_updates = data.get("updates", [])

    if not isinstance(raw_updates, list):
        warnings.append("'updates' must be a list; it was ignored.")
        raw_updates = []

    updates = []

    for index, raw_update in enumerate(raw_updates):
        try:
            update = _validate_update(raw_update, existing, refs)
        except ValueError as error:
            warnings.append(f"Dropped updates[{index}]: {error}.")
            continue

        if update is not None:
            updates.append(update)

    return artifacts, updates, warnings


def extract_meeting_artifacts(
    state: dict,
    config: RunnableConfig,
) -> dict:
    model = get_model(config)

    meeting_id = state.get("meeting_id", "")
    current_state = state.get("project_context", {}).get("current_state", {})
    agenda = state.get("agenda", [])

    context = {
        "meeting_id": meeting_id,
        "topic": state.get("topic", ""),
        "agenda_items": [
            {"id": item["id"], "description": item.get("description", "")}
            for item in agenda
        ],
        "participants": participants_overview(state),
        "meeting_summary": state.get("meeting_summary", ""),
        "existing_artifacts": compact_state(current_state),
        "conversation": [
            message.content
            for message in state.get("messages", [])
            if hasattr(message, "content")
        ],
    }

    system_prompt = """
You are extracting structured records from a completed human-AI meeting.

Extract only information explicitly supported by the meeting.

Return exactly one JSON object with these six list fields:

{
  "proposals": [],
  "arguments": [],
  "decisions": [],
  "action_items": [],
  "unresolved_issues": [],
  "updates": []
}

The first five lists hold NEW items only. "updates" holds status changes to
items listed in existing_artifacts, which were recorded in earlier meetings.

SCHEMAS:

Any new item may include "ref": a short temporary name such as "p1" or "d1".
Use it to refer to that new item from an argument or an update.

Every new item should include "agenda_id": the id of the agenda item in
agenda_items that it belongs to.

created_by, author_id, owner_id and approved_by hold a participant id from
participants, or null if unknown. Only a human participant can approve a
decision.

proposals:
{
  "ref": "p1",
  "description": "Proposal description",
  "status": "proposed",
  "created_by": null
}

Valid status:
proposed, under_discussion, accepted, rejected, deferred

arguments:
{
  "target_id": "id of an existing item, or ref of a new one",
  "target_type": "proposal",
  "position": "support",
  "content": "Argument",
  "reasoning": "",
  "author_id": null
}

Valid target_type:
proposal, problem, decision

Valid position:
support, oppose, neutral

decisions:
{
  "ref": "d1",
  "description": "Decision description",
  "basis": "Reason for decision",
  "status": "provisional",
  "approved_by": null
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

updates:
{
  "id": "id from existing_artifacts",
  "artifact_type": "proposals",
  "status": "accepted",
  "note": "Why the status changed",
  "superseded_by": null
}

artifact_type is the list the item belongs to: proposals, arguments,
decisions, action_items or unresolved_issues. For arguments, "status" is the
new position. When a new decision replaces an existing one, set the existing
decision's status to "superseded" and superseded_by to the new decision's ref.

IMPORTANT RULES:

1. Never invent proposals, arguments, decisions, or action items.
2. A recommendation is NOT a decision.
3. A proposal is NOT an accepted proposal.
4. Only explicitly agreed decisions may be marked approved.
5. Do not invent owners or deadlines.
6. Never re-create an item that is already in existing_artifacts. If the
   meeting discussed it again, refer to it by id, and add an update only if
   its status actually changed.
7. If no items exist in a category, return an empty list.
8. Do not include Markdown or explanations outside the JSON object.
""" + LIFECYCLE_RULES

    for _ in range(MAX_EXTRACTION_ATTEMPTS):
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
            artifacts, updates, warnings = resolve_extraction(
                parse_json_object(raw),
                current_state,
                meeting_id,
                agenda_ids=[item["id"] for item in agenda],
                participants=state.get("participants", []),
            )
            extraction_error = ""
            break
        except (ValueError, json.JSONDecodeError) as error:
            extraction_error = str(error)
            context["previous_attempt_error"] = (
                f"Your previous response was rejected: {error} "
                "Return exactly one valid JSON object."
            )

    if extraction_error:
        # Keep the completed meeting; report the failure instead of raising.
        artifacts = {field: [] for field in ARTIFACT_FIELDS}
        updates, warnings = [], []

    meeting_record = {
        "meeting_id": meeting_id,
        "project_id": state.get("project_id", ""),
        "topic": state.get("topic", ""),
        "agenda": "; ".join(item.get("description", "") for item in agenda),
        "agenda_items": agenda,
        "meeting_summary": state.get("meeting_summary", ""),
        **artifacts,
        "artifact_updates": updates,
        "extraction_warnings": warnings,
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
        "extraction_warnings": warnings,
        "messages": [AIMessage(content=status_message)],
    }
