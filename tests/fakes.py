"""Test helpers for the meeting engine."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from meeting_engine import build_meeting_graph
from meeting_engine.testing import (  # noqa: F401  (re-exported for tests)
    ASK,
    CONCLUDE,
    CONTINUE,
    DEFAULT_REPLIES,
    EMPTY_ARTIFACTS,
    ScriptedModel,
    moderator_reply,
)


def new_graph():
    """A standalone meeting graph with its own in-memory checkpointer."""
    return build_meeting_graph(InMemorySaver())


def invoke(graph, graph_input, config):
    """Invoke the graph and return its full internal state plus any interrupt.

    The graph itself only returns MeetingOutput; tests also inspect the
    transcript and deliberation control.
    """
    result = graph.invoke(graph_input, config)
    return {**graph.get_state(config).values, **result}
