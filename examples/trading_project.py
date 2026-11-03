"""The trading-frequency project as three meetings that build on each other.

Run one meeting at a time with a real model, for example:

    python examples/trading_project.py 1 --model openai:gpt-4o
    python examples/trading_project.py 2 --model openai:gpt-4o
    python examples/trading_project.py 3 --model openai:gpt-4o

Use --model fake to try it offline with a scripted model.

Project memory is kept in data/meetings.db and checkpoints in
data/checkpoints.db, so a meeting paused for your answer can be resumed later
by running the same command again.
"""

import argparse
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from meeting_engine import MeetingStore, build_meeting_graph
from meeting_engine.testing import demo_model


PROJECT_ID = "trading-ai"
DATA = Path("data")

PARTICIPANTS = [
    {"id": "researcher", "name": "Researcher", "role": "researcher",
     "description": "Develops alternative trading strategies."},
    {"id": "quant", "name": "Quant analyst", "role": "quant analyst",
     "description": "Evaluates mathematical and statistical assumptions."},
    {"id": "risk", "name": "Risk analyst", "role": "risk analyst",
     "description": "Identifies risks, drawdowns and failure modes."},
    {"id": "engineer", "name": "Engineer", "role": "engineer",
     "description": "Assesses implementation feasibility and cost."},
    {"id": "owner", "name": "Project owner", "role": "project owner",
     "is_human": True},
]

MEETINGS = {
    1: {
        "topic": "Reduce excessive trading frequency",
        "background": "The trading system generates too many trades; fees "
                      "and slippage reduce profitability.",
        "agenda": [
            {"description": "Fixed versus adaptive trading frequency"},
            {"description": "What is still unknown"},
        ],
    },
    2: {
        "topic": "Design the adaptive trading-frequency policy",
        "agenda": [
            {"description": "Which market-state variables control frequency?"},
            {"description": "How should the policy be evaluated?"},
        ],
    },
    3: {
        "topic": "Review the policy evaluation and decide",
        "background": "Paste backtest results here before running.",
        "agenda": [
            {"description": "Review evaluation results"},
            {"description": "Decide whether to adopt the policy"},
        ],
    },
}


def print_project_state(store: MeetingStore) -> None:
    state = store.get_project_state(PROJECT_ID)

    print("\n===== Current project state =====")
    for field, items in state.items():
        print(f"\n{field} ({len(items)})")
        for item in items:
            text = item.get("description") or item.get("content", "")
            status = item.get("status") or item.get("position", "")
            print(f"  - [{status}] {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("meeting", type=int, choices=sorted(MEETINGS))
    parser.add_argument(
        "--model", default=os.environ.get("MEETING_MODEL", "openai:gpt-4o")
    )
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    store = MeetingStore(str(DATA / "meetings.db"))

    checkpoints = sqlite3.connect(DATA / "checkpoints.db", check_same_thread=False)

    with closing(checkpoints) as conn:
        graph = build_meeting_graph(SqliteSaver(conn))
        config = {"configurable": {
            "thread_id": f"{PROJECT_ID}-meeting-{args.meeting}",
            "meeting_model": (
                demo_model()
                if args.model == "fake"
                else init_chat_model(args.model, temperature=0)
            ),
            "meeting_store": store,
        }}

        # Resume a paused meeting, or start it.
        snapshot = graph.get_state(config)
        if snapshot.interrupts:
            result = {"__interrupt__": snapshot.interrupts}
        elif snapshot.values.get("meeting_saved"):
            print(f"Meeting {args.meeting} is already finished.")
            print_project_state(store)
            return
        else:
            result = graph.invoke({
                "project_id": PROJECT_ID,
                "participants": PARTICIPANTS,
                "max_rounds_per_agenda": 2,
                **MEETINGS[args.meeting],
            }, config)

        while result.get("__interrupt__"):
            question = result["__interrupt__"][0].value
            asked = question.get("participant") or {}
            print(f"\nModerator asks {asked.get('name', 'you')}: {question['question']}")
            answer = input("Your answer (empty to skip): ")
            result = graph.invoke(Command(resume={"feedback": answer}), config)

    print("\n===== Meeting report =====\n")
    print(result.get("meeting_summary", ""))

    for warning in result.get("extraction_warnings", []):
        print(f"Warning: {warning}")
    if result.get("extraction_error"):
        print(f"Extraction failed: {result['extraction_error']}")

    print_project_state(store)


if __name__ == "__main__":
    main()
