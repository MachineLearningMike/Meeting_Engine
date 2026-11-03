from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.types import Command

from meeting_engine import build_meeting_graph

from meeting_engine.storage import MeetingStore


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

        """
        {
            "proposals": [
                {
                    "description": "Develop an adaptive trading-frequency policy.",
                    "status": "proposed",
                    "created_by": "AI"
                }
            ],
            "arguments": [
                {
                    "target_type": "proposal",
                    "position": "support",
                    "content": "Adaptive frequency may reduce unnecessary trades.",
                    "reasoning": "Trading frequency can respond to market opportunities."
                }
            ],
            "decisions": [],
            "action_items": [],
            "unresolved_issues": [
                {
                    "description": "Determine the optimal adaptive trading-frequency policy.",
                    "reason_unresolved": "The policy has not yet been developed or evaluated.",
                    "next_step": "Develop and backtest candidate policies.",
                    "status": "open"
                },
                {
                    "description": "Evaluate transaction costs.",
                    "reason_unresolved": "The transaction-cost impact has not yet been quantified.",
                    "next_step": "Include transaction costs in backtesting.",
                    "status": "open"
                }
            ]
        }
        """
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
        "project_id": "trading-ai",
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

    # Start meeting
    result = graph.invoke(initial_state, config)

    # Display moderator question and collect feedback
    # ...

    # Resume meeting

    user_feedback = input("\nYour feedback: ")

    result = graph.invoke(
        Command(resume=user_feedback),
        config
    )

    # Persist completed meeting
    store = MeetingStore()

    record = result["meeting_record"]

    assert record["project_id"] == "trading-ai"
    assert not result.get("extraction_error")

    store.save(record)

    print("\nMeeting saved to SQLite.")


    loaded_record = store.get(record["meeting_id"])

    assert loaded_record is not None
    assert loaded_record["meeting_id"] == record["meeting_id"]

    print("Persistence verification passed.")

    project_meetings = store.list_meetings("trading-ai")

    print("\nMeetings in trading-ai:", len(project_meetings))

    for meeting in project_meetings:
        print(meeting["meeting_id"], meeting["topic"])


    print("\nMeeting completed.")
    print(result.get("meeting_summary", ""))

    # Inspect structured artifacts AFTER completion
    print("\n===== STRUCTURED MEETING RECORD =====")

    record = result.get("meeting_record", {})

    for key, value in record.items():
        print(f"\n{key}:")
        print(value)

    print("\n===== ARTIFACT COUNTS =====")

    for key in (
        "proposals",
        "arguments",
        "decisions",
        "action_items",
        "unresolved_issues",
    ):
        print(f"{key}: {len(result.get(key, []))}")


if __name__ == "__main__":
    main()
