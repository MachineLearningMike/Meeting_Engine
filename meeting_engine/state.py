"""LangGraph state definition for the meeting engine."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .schemas import (
    ActionItem,
    Argument,
    Decision,
    MeetingObjective,
    MeetingProblem,
    Participant,
    Proposal,
    UnresolvedIssue,
)


class MeetingState(TypedDict, total=False):
    # Identity and context
    meeting_id: str
    project_id: str
    topic: str
    background: str
    prior_state: dict[str, Any]

    # Meeting inputs
    objectives: list[MeetingObjective]
    problems: list[MeetingProblem]
    participants: list[Participant]

    # Conversation history
    messages: Annotated[list[AnyMessage], add_messages]

    # Structured meeting artifacts.
    # With operator.add, nodes return only newly-created items.
    proposals: Annotated[list[Proposal], operator.add]
    arguments: Annotated[list[Argument], operator.add]
    decisions: Annotated[list[Decision], operator.add]
    action_items: Annotated[list[ActionItem], operator.add]
    unresolved_issues: Annotated[list[UnresolvedIssue], operator.add]

    # Deliberation control
    current_phase: Literal[
        "preparation",
        "orientation",
        "deliberation",
        "human_feedback",
        "convergence",
        "conclusion",
        "completed",
    ]
    current_agenda_id: Optional[str]
    current_agenda: str
    round_count: int
    max_rounds_per_agenda: int
    pending_user_input: bool
    next_action: str
    user_feedback: Optional[str]

    # Output
    meeting_summary: str

    # additional fields for moderator and human feedback nodes
    moderator_question: str
    moderator_reasoning: str
    pending_user_input: bool

    # additional fields for human feedback node
    user_feedback: str
    user_declined_feedback: bool

    # additional fields for analysis node
    moderator_question: str
    moderator_reasoning: str
    moderator_action: str
    pending_user_input: bool

    user_feedback: str
    user_declined_feedback: bool

    context_analysis: str
    latest_deliberation: str
    meeting_summary: str

    round_count: int
    max_rounds_per_agenda: int
    current_phase: str
    current_agenda_id: str
    current_agenda: str
