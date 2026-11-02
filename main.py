from langgraph.types import Command
from langchain_openai import ChatOpenAI

from meeting_engine.graph import build_meeting_graph


def main():
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    graph = build_meeting_graph()

    config = {
        "configurable": {
            "thread_id": "meeting-001",
            "meeting_model": model,
        }
    }

    initial_state = {
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