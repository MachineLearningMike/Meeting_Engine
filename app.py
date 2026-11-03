"""Chainlit chat UI for the meeting engine.

Run with:
    chainlit run app.py

Each graph node is shown as a collapsible step. When the moderator asks for
input, the question appears as a chat message and your reply resumes the
meeting. Model, project and round limit are set in the chat settings panel.
"""

import os
import uuid

import chainlit as cl
from chainlit.input_widget import Slider, TextInput
from langchain.chat_models import init_chat_model
from langgraph.types import Command

from meeting_engine import build_meeting_graph
from meeting_engine.schemas import new_id
from meeting_engine.storage import MeetingStore


DEFAULT_MODEL = os.environ.get("MEETING_MODEL", "openai:gpt-4o")

STEP_NAMES = {
    "initialize": "Agenda",
    "analyze_context": "Context analysis",
    "deliberate": "Deliberation",
    "moderate": "Moderator",
    "finalize": "Meeting report",
    "extract_artifacts": "Meeting records",
}

ARTIFACT_TITLES = {
    "proposals": "Proposals",
    "arguments": "Arguments",
    "decisions": "Decisions",
    "action_items": "Action items",
    "unresolved_issues": "Unresolved issues",
}

# One graph and checkpointer for the server; each meeting is its own thread.
graph = build_meeting_graph()
store = MeetingStore()


def format_step(node: str, update: dict) -> str:
    if node == "initialize":
        return f"Agenda: {update.get('current_agenda', '')}"

    if node == "analyze_context":
        return update.get("context_analysis", "")

    if node == "deliberate":
        return (
            f"**Round {update.get('round_count', '')}**\n\n"
            f"{update.get('latest_deliberation', '')}"
        )

    if node == "moderate":
        text = (
            f"**Action:** `{update.get('moderator_action', '')}`\n\n"
            f"{update.get('moderator_reasoning', '')}"
        )
        if update.get("moderator_question"):
            text += f"\n\n**Question:** {update['moderator_question']}"
        return text

    if node == "finalize":
        return "Report written."

    if node == "extract_artifacts":
        return format_artifacts(update)

    return ""


def format_artifacts(update: dict) -> str:
    if update.get("extraction_error"):
        return f"Extraction failed: {update['extraction_error']}"

    lines = []

    for field, title in ARTIFACT_TITLES.items():
        items = update.get(field, [])
        lines.append(f"**{title}** ({len(items)})")

        for item in items:
            text = item.get("description") or item.get("content", "")
            label = item.get("status") or item.get("position", "")
            lines.append(f"- {text} _({label})_")

        lines.append("")

    return "\n".join(lines).strip()


def get_model():
    """Build the chat model for the current settings, reusing it if unchanged."""

    name = cl.user_session.get("settings")["model"]
    cached = cl.user_session.get("model")

    if cached and cached[0] == name:
        return cached[1]

    model = init_chat_model(name, temperature=0)
    cl.user_session.set("model", (name, model))
    return model


async def run_meeting(graph_input) -> None:
    """Stream the graph until it pauses for feedback or finishes."""

    try:
        model = get_model()
    except Exception as error:
        await cl.Message(content=f"Could not load the model: {error}").send()
        return

    config = {
        "configurable": {
            "thread_id": cl.user_session.get("thread_id"),
            "meeting_model": model,
        }
    }

    cl.user_session.set("phase", "running")
    steps = {}

    try:
        async for event in graph.astream(
            graph_input, config, stream_mode="tasks"
        ):
            node = event["name"]

            if node not in STEP_NAMES:
                continue

            if "input" in event:
                step = cl.Step(name=STEP_NAMES[node], type="llm")
                await step.send()
                steps[event["id"]] = step
                continue

            step = steps.pop(event["id"], None)

            if step is not None:
                step.output = (
                    f"Error: {event['error']}"
                    if event["error"]
                    else format_step(node, event["result"])
                )
                await step.update()

    except Exception as error:
        cl.user_session.set("phase", "done")
        await cl.Message(
            content=(
                f"The meeting stopped with an error: {error}\n\n"
                "Send a new message to start another meeting."
            )
        ).send()
        return

    snapshot = await graph.aget_state(config)

    if snapshot.interrupts:
        cl.user_session.set("phase", "awaiting_feedback")
        question = snapshot.interrupts[0].value.get("question", "")

        await cl.Message(
            content=f"**Moderator asks:** {question}",
            actions=[
                cl.Action(
                    name="decline_feedback",
                    payload={},
                    label="Continue without me",
                ),
            ],
        ).send()
        return

    await finish_meeting(snapshot.values)


async def finish_meeting(state: dict) -> None:
    cl.user_session.set("phase", "done")

    await cl.Message(
        content=f"## Meeting report\n\n{state.get('meeting_summary', '')}"
    ).send()

    record = state.get("meeting_record")

    if record:
        store.save(record)
        note = f"Saved as meeting `{record['meeting_id']}`."
    else:
        note = "No meeting record was produced."

    await cl.Message(
        content=f"{note}\n\nSend a new message to start another meeting."
    ).send()


@cl.on_chat_start
async def start() -> None:
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="model",
                label="Model (provider:model)",
                initial=DEFAULT_MODEL,
            ),
            TextInput(
                id="project_id",
                label="Project ID (optional)",
                initial="",
            ),
            Slider(
                id="max_rounds",
                label="Max deliberation rounds",
                initial=3,
                min=1,
                max=10,
                step=1,
            ),
        ]
    ).send()

    cl.user_session.set("settings", settings)
    cl.user_session.set("phase", "idle")

    await cl.Message(
        content=(
            "What should this meeting be about?\n\n"
            "Put the topic on the first line. Anything below it is used "
            "as background."
        )
    ).send()


@cl.on_settings_update
async def update_settings(settings: dict) -> None:
    cl.user_session.set("settings", settings)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    phase = cl.user_session.get("phase")

    if phase == "awaiting_feedback":
        await run_meeting(Command(resume={"feedback": message.content}))
        return

    if phase == "running":
        await cl.Message(content="The meeting is still running.").send()
        return

    topic, _, background = message.content.strip().partition("\n")
    settings = cl.user_session.get("settings")

    cl.user_session.set("thread_id", str(uuid.uuid4()))

    await run_meeting({
        "project_id": settings["project_id"].strip(),
        "topic": topic.strip(),
        "background": background.strip(),
        "problems": [{"id": new_id(), "description": topic.strip()}],
        "objectives": [],
        "max_rounds_per_agenda": int(settings["max_rounds"]),
    })


@cl.action_callback("decline_feedback")
async def decline_feedback(action: cl.Action) -> None:
    if cl.user_session.get("phase") != "awaiting_feedback":
        return

    await action.remove()
    await run_meeting(
        Command(resume={"feedback": "", "decline_future_questions": True})
    )
