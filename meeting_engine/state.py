"""LangGraph state definition for the meeting engine."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .schemas import (
    ActionItem,
    Argument,
    Decision,
    MeetingObjective,
    MeetingProblem,
    MeetingRecord,
    Participant,
    ProjectContext,
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
    project_context: ProjectContext

    # Meeting inputs
    objectives: list[MeetingObjective]
    problems: list[MeetingProblem]
    participants: list[Participant]

    # Conversation history
    messages: Annotated[list[AnyMessage], add_messages]

    # Structured artifacts
    proposals: Annotated[list[Proposal], operator.add]
    arguments: Annotated[list[Argument], operator.add]
    decisions: Annotated[list[Decision], operator.add]
    action_items: Annotated[list[ActionItem], operator.add]
    unresolved_issues: Annotated[list[UnresolvedIssue], operator.add]

    # Aggregate structured record
    meeting_record: MeetingRecord
    extraction_error: str
    extraction_warnings: list[str]
    meeting_saved: bool

    # Deliberation control
    current_phase: str
    current_agenda_id: str
    current_agenda: str
    round_count: int
    max_rounds_per_agenda: int
    pending_user_input: bool
    moderator_action: str
    moderator_question: str
    moderator_reasoning: str
    user_feedback: str
    user_declined_feedback: bool

    # Analysis
    context_analysis: str
    latest_deliberation: str

    # Final output
    meeting_summary: str