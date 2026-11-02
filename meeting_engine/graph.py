from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import MeetingState

from .nodes.initialization import initialize_meeting
from .nodes.analysis import analyze_context
from .nodes.deliberation import deliberate
from .nodes.moderator import moderate
from .nodes.human_feedback import collect_human_feedback
from .nodes.finalize import finalize_meeting


def route_after_moderator(state: MeetingState) -> str:
    action = state.get("moderator_action")

    if action == "continue_discussion":
        return "deliberate"

    if action == "request_user_feedback":
        return "human_feedback"

    if action == "conclude_meeting":
        return "finalize"

    raise ValueError(f"Unknown moderator action: {action}")


def build_meeting_graph(checkpointer=None):
    """
    Build and compile the reusable meeting graph.

    The model is supplied at invocation time through:
        config["configurable"]["meeting_model"]
    """

    builder = StateGraph(MeetingState)

    # Register nodes.
    builder.add_node("initialize", initialize_meeting)
    builder.add_node("analyze_context", analyze_context)
    builder.add_node("deliberate", deliberate)
    builder.add_node("moderate", moderate)
    builder.add_node("human_feedback", collect_human_feedback)
    builder.add_node("finalize", finalize_meeting)

    # Define execution flow.
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "analyze_context")
    builder.add_edge("analyze_context", "deliberate")
    builder.add_edge("deliberate", "moderate")

    builder.add_conditional_edges(
        "moderate",
        route_after_moderator,
        {
            "deliberate": "deliberate",
            "human_feedback": "human_feedback",
            "finalize": "finalize",
        },
    )

    builder.add_edge("human_feedback", "analyze_context")
    builder.add_edge("finalize", END)

    # Use in-memory checkpointing by default.
    if checkpointer is None:
        checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)