from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import MeetingState

from .nodes.initialization import initialize_meeting
from .nodes.agenda import advance_agenda
from .nodes.analysis import analyze_context
from .nodes.deliberation import deliberate
from .nodes.moderator import moderate
from .nodes.human_feedback import collect_human_feedback
from .nodes.finalize import finalize_meeting

from .nodes.extract_artifacts import extract_meeting_artifacts
from .nodes.project_memory import load_project_memory, save_meeting


def route_after_moderator(state: MeetingState) -> str:
    action = state.get("moderator_action")

    if action == "continue_discussion":
        return "deliberate"

    if action == "request_user_feedback":
        return "human_feedback"

    if action in ("conclude_agenda", "defer_agenda", "revisit_agenda"):
        return "advance_agenda"

    if action == "conclude_meeting":
        return "finalize"

    raise ValueError(f"Unknown moderator action: {action}")


def route_after_agenda(state: MeetingState) -> str:
    return "finalize" if state.get("agenda_complete") else "analyze_context"


def build_meeting_graph(checkpointer=None):
    """
    Build and compile the reusable meeting graph.

    The model is supplied at invocation time through:
        config["configurable"]["meeting_model"]

    Optionally, a MeetingStorage in config["configurable"]["meeting_store"]
    loads the project's memory at the start and saves the record at the end.
    """

    builder = StateGraph(MeetingState)

    # Register nodes.
    builder.add_node("initialize", initialize_meeting)
    builder.add_node("load_project_memory", load_project_memory)
    builder.add_node("analyze_context", analyze_context)
    builder.add_node("deliberate", deliberate)
    builder.add_node("moderate", moderate)
    builder.add_node("human_feedback", collect_human_feedback)
    builder.add_node("advance_agenda", advance_agenda)

    # Define execution flow.
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "load_project_memory")
    builder.add_edge("load_project_memory", "analyze_context")
    builder.add_edge("analyze_context", "deliberate")
    builder.add_edge("deliberate", "moderate")

    builder.add_node("finalize", finalize_meeting)
    builder.add_node("extract_artifacts", extract_meeting_artifacts)
    builder.add_node("save_meeting", save_meeting)

    builder.add_conditional_edges(
        "moderate",
        route_after_moderator,
        {
            "deliberate": "deliberate",
            "human_feedback": "human_feedback",
            "advance_agenda": "advance_agenda",
            "finalize": "finalize",
        },
    )

    builder.add_conditional_edges(
        "advance_agenda",
        route_after_agenda,
        {
            "analyze_context": "analyze_context",
            "finalize": "finalize",
        },
    )

    builder.add_edge("human_feedback", "analyze_context")
    builder.add_edge("finalize", "extract_artifacts")
    builder.add_edge("extract_artifacts", "save_meeting")
    builder.add_edge("save_meeting", END)

    # Use in-memory checkpointing by default.
    if checkpointer is None:
        checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)

