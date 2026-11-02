from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.types import Command

from meeting_engine import build_meeting_graph


def main():

    # Seven deterministic model responses:
    # 1. Context analysis
    # 2. First deliberation
    # 3. Moderator requests feedback
    # 4. Context analysis after feedback
    # 5. Second deliberation
    # 6. Moderator concludes
    # 7. Final meeting report

    responses = [
        "The trading system generates excessive trades. "
        "Transaction fees may be reducing profitability.",

        "Approach A: fixed trading intervals. "
        "Approach B: adaptive trading frequency based on market state.",

        """
        {
            "action": "request_user_feedback",
            "reason": "The choice depends on the user's operational priorities.",
            "question_for_user": "Which approach do you prefer?"
        }
        """,

        "The user prefers adaptive frequency. "
        "The practical objective is to reduce unnecessary trades.",

        "An adaptive policy could reduce trading frequency "
        "during low-opportunity market conditions.",

        """
        {
            "action": "conclude_meeting",
            "reason": "The discussion has established a useful direction.",
            "question_for_user": ""
        }
        """,

        """
        Meeting objective:
        Reduce excessive trading frequency.

        Main analysis:
        Fixed intervals and adaptive frequency were considered.

        Decisions:
        No formal implementation decision was made.

        Recommended next steps:
        Develop and backtest an adaptive trading-frequency policy.

        Unresolved issues:
        Determine the optimal policy and evaluate transaction costs.
        """,
    ]

    model = FakeListChatModel(responses=responses)

    graph = build_meeting_graph()

    config = {
        "configurable": {
            "thread_id": "test-meeting-001",
            "meeting_model": model,
        }
    }

    initial_state = {
        "topic": "Adaptive trading frequency",
        "background": "The trading system generates too many trades.",
        "problems": [
            {
                "id": "problem-001",
                "description": "Trading frequency is too high.",
            }
        ],
        "objectives": [],
        "max_rounds_per_agenda": 3,
    }

    print("Starting meeting...")

    result = graph.invoke(initial_state, config=config)

    while result.get("__interrupt__"):

        for interruption in result["__interrupt__"]:
            print("\nModerator:")
            print(interruption.value["question"])

        feedback = input("\nYour feedback: ")

        result = graph.invoke(
            Command(
                resume={"feedback": feedback}
            ),
            config=config,
        )

    print("\nMeeting completed.")
    print(result.get("meeting_summary", ""))


if __name__ == "__main__":
    main()