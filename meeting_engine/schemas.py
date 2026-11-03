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


class Argument(TypedDict, total=False):
    id: str
    target_id: str
    target_type: ArgumentTargetType
    position: ArgumentPosition
    content: str
    reasoning: str
    evidence: list[str]
    author_id: str


class Decision(TypedDict, total=False):
    id: str
    description: str
    basis: str
    status: DecisionStatus
    related_problem_ids: list[str]
    related_proposal_ids: list[str]
    approved_by: Optional[str]
    created_at: datetime


class ActionItem(TypedDict, total=False):
    id: str
    description: str
    owner_id: Optional[str]
    due_date: Optional[datetime]
    status: ActionItemStatus
    related_decision_id: Optional[str]


class UnresolvedIssue(TypedDict, total=False):
    id: str
    description: str
    reason_unresolved: str
    next_step: str
    related_problem_id: Optional[str]
    status: UnresolvedIssueStatus


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


class MeetingRecord(TypedDict, total=False):
    meeting_id: str
    project_id: str
    topic: str
    agenda: str
    proposals: list[Proposal]
    arguments: list[Argument]
    decisions: list[Decision]
    action_items: list[ActionItem]
    unresolved_issues: list[UnresolvedIssue]
    extraction_error: str
