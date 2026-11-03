"""Scripted chat models for testing and for trying the engine without an API.

ScriptedModel answers according to the meeting node that calls it, so tests
do not depend on the order or number of model calls. demo_model() returns one
with plausible replies for a quick offline run.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def moderator_reply(
    action: str,
    question: str = "",
    reason: str = "test",
    **fields: str,
) -> str:
    return json.dumps({
        "action": action,
        "reason": reason,
        "question_for_user": question,
        **fields,
    })


ASK = moderator_reply("request_user_feedback", question="Which approach?")
CONTINUE = moderator_reply("continue_discussion")
CONCLUDE = moderator_reply("conclude_meeting")

EMPTY_ARTIFACTS = json.dumps({
    "proposals": [],
    "arguments": [],
    "decisions": [],
    "action_items": [],
    "unresolved_issues": [],
})

DEFAULT_REPLIES = {
    "analysis": "Analysis.",
    "deliberation": "Deliberation.",
    "moderator": CONCLUDE,
    "finalize": "Report.",
    "extraction": EMPTY_ARTIFACTS,
}


class ScriptedModel(BaseChatModel):
    """Chat model that answers according to the meeting node calling it.

    Each node is identified by the "meeting:<node>" tag set in invoke_text,
    and a participant's turn by "participant:<id>". Scripted replies for a
    node (or "node:participant") are used in order; once they run out, the
    node's default reply is returned. Every call is recorded in ``calls``.
    """

    replies: dict[str, list[str]] = {}
    defaults: dict[str, str] = DEFAULT_REPLIES
    calls: list[dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-test-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        tags = run_manager.tags if run_manager else []
        nodes = [tag.split(":", 1)[1] for tag in tags if tag.startswith("meeting:")]

        if len(nodes) != 1:
            raise AssertionError(f"Expected one meeting:<node> tag, got {tags}")

        node = nodes[0]
        participant = next(
            (tag.split(":", 1)[1] for tag in tags if tag.startswith("participant:")),
            None,
        )

        # Replies keyed "node:participant" take precedence over "node".
        queue = self.replies.get(f"{node}:{participant}") or self.replies.get(node, [])
        content = queue.pop(0) if queue else self.defaults[node]

        self.calls.append({
            "node": node,
            "participant": participant,
            "system": messages[0].content,
            "human": messages[-1].content,
        })

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    def calls_for(self, node: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["node"] == node]


def demo_model() -> ScriptedModel:
    """A scripted model for an offline demo meeting.

    The moderator asks one question, then concludes each agenda item in turn.
    Extraction records one proposal and one open issue.
    """

    def next_item(summary: str) -> str:
        return moderator_reply(
            "conclude_agenda",
            reason="The item has made enough progress for now.",
            agenda_summary=summary,
        )

    return ScriptedModel(
        replies={
            "moderator": [
                moderator_reply(
                    "request_user_feedback",
                    reason="The choice depends on your priorities.",
                    question="Should we prefer fixed intervals or adaptive frequency?",
                ),
                *[next_item("Adaptive frequency is the preferred direction.")] * 10,
            ],
        },
        defaults={
            "analysis": (
                "Established: the system trades too often and fees reduce "
                "profit. Unverified: that fewer trades keep returns. Missing: "
                "which market conditions justify trading."
            ),
            "deliberation": (
                "Option A: fixed trading intervals, simple but blind to market "
                "conditions. Option B: adaptive frequency tied to volatility, "
                "fewer pointless trades but more parameters to tune. Risk: "
                "missing large moves. Next step: backtest both."
            ),
            "moderator": CONCLUDE,
            "finalize": (
                "Meeting objective: reduce excessive trading frequency.\n"
                "Discussion: fixed intervals and adaptive frequency were "
                "compared; adaptive frequency is preferred.\n"
                "Decisions: none yet.\n"
                "Unresolved: transaction-cost impact.\n"
                "Next step: backtest an adaptive policy.\n\n"
                "(This report was produced by the offline demo model.)"
            ),
            "extraction": json.dumps({
                "proposals": [{
                    "description": "Adopt adaptive trading frequency tied to volatility",
                    "status": "proposed",
                }],
                "arguments": [],
                "decisions": [],
                "action_items": [],
                "unresolved_issues": [{
                    "description": "Transaction-cost impact is not yet quantified",
                    "next_step": "Include costs in the backtest",
                    "status": "open",
                }],
            }),
        },
    )
