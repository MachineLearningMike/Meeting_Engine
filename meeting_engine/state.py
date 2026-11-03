"""LangGraph state definition for the meeting engine."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .schemas import (
    ActionItem,
    AgendaItem,
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
    language: str
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

    # Agenda
    agenda: list[AgendaItem]
    agenda_complete: bool

    # Deliberation control
    current_phase: str
    current_agenda_id: str
    current_agenda: str
    round_count: int
    total_rounds: int
    max_rounds_per_agenda: int
    max_total_rounds: int
    pending_user_input: bool
    moderator_action: str
    moderator_question: str
    moderator_reasoning: str
    moderator_target_agenda_id: str
    moderator_ask_participant_id: str
    agenda_summary: str
    user_feedback: str
    user_declined_feedback: bool

    # Analysis
    context_analysis: str
    latest_deliberation: str

    # Final output
    meeting_summary: str


class MeetingInput(TypedDict, total=False):
    """What a caller or parent graph provides to start a meeting.

    meeting_id is optional; a new one is generated when it is missing. Reusing
    an id saves the meeting over the earlier record with that id.
    """

    meeting_id: str
    project_id: str
    topic: str
    background: str
    language: str
    prior_state: dict[str, Any]
    project_context: ProjectContext
    objectives: list[MeetingObjective]
    problems: list[MeetingProblem]
    participants: list[Participant]
    agenda: list[AgendaItem]
    max_rounds_per_agenda: int
    max_total_rounds: int


class MeetingOutput(TypedDict, total=False):
    """What a finished meeting returns to its caller or parent graph.

    The transcript and deliberation control stay inside the meeting; read
    them with graph.get_state(config) when needed.
    """

    meeting_id: str
    project_id: str
    meeting_record: MeetingRecord
    meeting_summary: str
    agenda: list[AgendaItem]
    extraction_error: str
    extraction_warnings: list[str]
    meeting_saved: bool
