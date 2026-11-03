"""Native-Python TypedDict definitions for meeting artifacts.

These types provide static type hints. They do not perform runtime validation.
That is intentional: the core meeting engine does not require Pydantic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, TypedDict
from uuid import uuid4


def new_id() -> str:
    return str(uuid4())


# Allowed values, shared with runtime validation in nodes/extract_artifacts.py.
Priority = Literal["low", "medium", "high", "critical"]
ObjectiveStatus = Literal["active", "achieved", "deferred", "cancelled"]
ProblemStatus = Literal["open", "investigating", "blocked", "resolved", "deferred"]
ProposalStatus = Literal[
    "proposed", "under_discussion", "accepted", "rejected", "deferred"
]
ArgumentTargetType = Literal["proposal", "problem", "decision"]
ArgumentPosition = Literal["support", "oppose", "neutral"]
DecisionStatus = Literal["provisional", "approved", "rejected", "superseded"]
ActionItemStatus = Literal["pending", "in_progress", "completed", "cancelled"]
UnresolvedIssueStatus = Literal["open", "deferred", "resolved"]

# Record fields that hold project artifacts.
ArtifactType = Literal[
    "proposals", "arguments", "decisions", "action_items", "unresolved_issues"
]

# The field holding each artifact type's status.
ARTIFACT_STATUS_FIELDS: dict[str, str] = {
    "proposals": "status",
    "arguments": "position",
    "decisions": "status",
    "action_items": "status",
    "unresolved_issues": "status",
}

# Allowed status changes for artifacts recorded in earlier meetings. A status
# missing from a mapping is final. New items may start in any valid status.
# Argument positions are not a lifecycle and may change freely.
ALLOWED_TRANSITIONS: dict[str, dict[str, set[str]]] = {
    "proposals": {
        "proposed": {"under_discussion", "accepted", "rejected", "deferred"},
        "under_discussion": {"accepted", "rejected", "deferred"},
        "deferred": {"under_discussion", "accepted", "rejected"},
    },
    "decisions": {
        "provisional": {"approved", "rejected", "superseded"},
        "approved": {"superseded"},
    },
    "action_items": {
        "pending": {"in_progress", "completed", "cancelled"},
        "in_progress": {"completed", "cancelled"},
    },
    "unresolved_issues": {
        "open": {"deferred", "resolved"},
        "deferred": {"open", "resolved"},
    },
}


def check_transition(artifact_type: str, current: str | None, new: str) -> None:
    """Raise ValueError if an existing artifact may not move to the new status."""

    transitions = ALLOWED_TRANSITIONS.get(artifact_type)

    if transitions is None or current is None or current == new:
        return

    if new not in transitions.get(current, set()):
        allowed = sorted(transitions.get(current, set()))
        raise ValueError(
            f"{artifact_type} cannot change from '{current}' to '{new}' "
            f"(allowed: {', '.join(allowed) or 'none, it is final'})"
        )


class Participant(TypedDict, total=False):
    id: str
    name: str
    role: str
    description: str
    is_human: bool


class MeetingObjective(TypedDict, total=False):
    id: str
    description: str
    priority: Priority
    status: ObjectiveStatus
    success_criteria: list[str]


class MeetingProblem(TypedDict, total=False):
    id: str
    description: str
    priority: Priority
    status: ProblemStatus
    related_objective_ids: list[str]
    constraints: list[str]
    hypotheses: list[str]


class Proposal(TypedDict, total=False):
    id: str
    problem_id: str
    description: str
    status: ProposalStatus
    supporting_argument_ids: list[str]
    opposing_argument_ids: list[str]
    evidence: list[str]
    created_by: str
    created_in_meeting: str
    updated_in_meeting: str


class Argument(TypedDict, total=False):
    id: str
    target_id: str
    target_type: ArgumentTargetType
    position: ArgumentPosition
    content: str
    reasoning: str
    evidence: list[str]
    author_id: str
    created_in_meeting: str
    updated_in_meeting: str


class Decision(TypedDict, total=False):
    id: str
    description: str
    basis: str
    status: DecisionStatus
    related_problem_ids: list[str]
    related_proposal_ids: list[str]
    approved_by: Optional[str]
    created_at: datetime
    superseded_by: Optional[str]
    created_in_meeting: str
    updated_in_meeting: str


class ActionItem(TypedDict, total=False):
    id: str
    description: str
    owner_id: Optional[str]
    due_date: Optional[datetime]
    status: ActionItemStatus
    related_decision_id: Optional[str]
    created_in_meeting: str
    updated_in_meeting: str


class UnresolvedIssue(TypedDict, total=False):
    id: str
    description: str
    reason_unresolved: str
    next_step: str
    related_problem_id: Optional[str]
    status: UnresolvedIssueStatus
    created_in_meeting: str
    updated_in_meeting: str


def make_problem(description: str, **fields: object) -> MeetingProblem:
    return {"id": new_id(), "description": description, **fields}


def make_objective(description: str, **fields: object) -> MeetingObjective:
    return {"id": new_id(), "description": description, **fields}


def make_proposal(problem_id: str, description: str, **fields: object) -> Proposal:
    return {
        "id": new_id(),
        "problem_id": problem_id,
        "description": description,
        **fields,
    }


class ArtifactUpdate(TypedDict, total=False):
    """A status change to an artifact recorded in an earlier meeting."""

    id: str
    artifact_type: ArtifactType
    status: str
    note: str
    superseded_by: Optional[str]


class MeetingSummary(TypedDict, total=False):
    meeting_id: str
    topic: str
    agenda: str
    meeting_summary: str


class ProjectContext(TypedDict, total=False):
    """What a new meeting knows about its project.

    current_state holds each artifact with its latest status and is
    authoritative. history holds summaries of past meetings, oldest first,
    and is background only.
    """

    project_id: str
    current_state: dict[str, list[dict]]
    history: list[MeetingSummary]


class MeetingRecord(TypedDict, total=False):
    meeting_id: str
    project_id: str
    topic: str
    agenda: str
    meeting_summary: str
    proposals: list[Proposal]
    arguments: list[Argument]
    decisions: list[Decision]
    action_items: list[ActionItem]
    unresolved_issues: list[UnresolvedIssue]
    artifact_updates: list[ArtifactUpdate]
    extraction_error: str
    extraction_warnings: list[str]
