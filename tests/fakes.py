"""Test doubles for the meeting engine."""

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

    Each node is identified by the "meeting:<node>" tag set in invoke_text.
    Scripted replies for a node are used in order; once they run out, the
    node's default reply is returned. Every call is recorded in ``calls``.
    """

    replies: dict[str, list[str]] = {}
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
        queue = self.replies.get(node, [])
        content = queue.pop(0) if queue else DEFAULT_REPLIES[node]

        self.calls.append({
            "node": node,
            "system": messages[0].content,
            "human": messages[-1].content,
        })

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    def calls_for(self, node: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["node"] == node]
