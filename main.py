import argparse
import os

from langchain.chat_models import init_chat_model
from langgraph.types import Command

from meeting_engine.graph import build_meeting_graph
from meeting_engine.storage import MeetingStore


DEFAULT_MODEL = "openai:gpt-4o"


def main():
    parser = argparse.ArgumentParser(description="Run a meeting.")
    parser.add_argument(
        "--model",
        default=os.environ.get("MEETING_MODEL", DEFAULT_MODEL),
        help=(
            "Chat model as 'provider:model', e.g. openai:gpt-4o, "
            "anthropic:claude-sonnet-5-5, ollama:llama3.1. "
            "Defaults to $MEETING_MODEL or " + DEFAULT_MODEL + "."
        ),
    )
    parser.add_argument(
        "--project",
        default="",
        help="Project ID. Meetings of the same project share their memory.",
    )
    args = parser.parse_args()

    # The provider package (langchain-openai, langchain-anthropic, ...)
    # is imported only for the provider actually selected.
    model = init_chat_model(args.model, temperature=0)

    graph = build_meeting_graph()

    config = {
        "configurable": {
            "thread_id": "meeting-001",
            "meeting_model": model,
            "meeting_store": MeetingStore(),
        }
    }

    initial_state = {
        "project_id": args.project,
        "topic": "Adaptive trading frequency",
        "background": "Our trading system generates too many trades.",
        "problems": [
            {
                "id": "problem-001",
                "description": "Trading frequency is too high.",
            }
        ],
        "objectives": [],
        "max_rounds_per_agenda": 3,
    }

    # Start meeting.
    result = graph.invoke(initial_state, config=config)

    # Meeting interaction loop.
    while result.get("__interrupt__"):

        for interruption in result["__interrupt__"]:
            print("\nModerator asks:")
            print(interruption.value["question"])

        feedback = input("\nYour feedback: ")

        result = graph.invoke(
            Command(resume={"feedback": feedback}),
            config=config,
        )

    print("\nMeeting completed.")
    print(result.get("meeting_summary", ""))


if __name__ == "__main__":
    main()