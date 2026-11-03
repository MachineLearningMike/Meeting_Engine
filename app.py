"""Chainlit chat UI for the meeting engine.

Run with:
    chainlit run app.py

Each graph node is shown as a collapsible step. When the moderator asks for
input, the question appears as a chat message and a reply resumes the
meeting. Model, project, round limit and AI participants are set in the chat
settings panel.

Several people can share a meeting: one sends /room and shares the code, the
others send /join CODE. Everyone in the room sees the meeting as it runs and
takes part as a human participant.
"""

import asyncio
import os
import re
import secrets
import string
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import chainlit as cl
from chainlit.context import init_ws_context
from chainlit.input_widget import Slider, TextInput
from chainlit.session import WebsocketSession
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from meeting_engine import build_meeting_graph
from meeting_engine.schemas import new_id
from meeting_engine.storage import MeetingStore
from meeting_engine.testing import demo_model


DEFAULT_MODEL = os.environ.get("MEETING_MODEL", "openai:gpt-4o")

STEP_NAMES = {
    "initialize": "Agenda",
    "load_project_memory": "Project memory",
    "advance_agenda": "Next agenda item",
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
graph = build_meeting_graph(InMemorySaver())
store = MeetingStore()


# --- rooms and pushing to several browser sessions ---------------------------


@dataclass
class Room:
    """People sharing one meeting. Keys of ``members`` are Chainlit session ids."""

    code: str
    host: str
    members: dict[str, str] = field(default_factory=dict)  # session id -> name
    languages: dict[str, str] = field(default_factory=dict)  # session id -> language
    phase: str = "lobby"  # lobby | running | awaiting_feedback | done
    thread_id: str = ""
    model: Any = None
    humans: dict[str, str] = field(default_factory=dict)  # participant id -> session id
    asked: str | None = None  # session that must answer; None means anyone


ROOMS: dict[str, Room] = {}


def current_session() -> str:
    return cl.context.session.id


def room_of(session_id: str) -> Room | None:
    return next((r for r in ROOMS.values() if session_id in r.members), None)


def new_code() -> str:
    alphabet = string.ascii_uppercase + string.digits
    while True:
        code = "".join(secrets.choice(alphabet) for _ in range(6))
        if code not in ROOMS:
            return code


async def push_to(session_id: str, send: Callable[[], Awaitable[Any]]) -> None:
    """Run ``send`` as if in another browser session.

    init_ws_context points Chainlit's context at that session. It runs in its
    own task, which has its own copy of the context, so the session that is
    running this code is not affected. Sessions are looked up by session id
    (stable across reconnects); a plain string would be taken as a socket id.
    """

    session = WebsocketSession.get_by_id(session_id)

    if session is None:
        raise LookupError(f"Session {session_id} is gone.")

    async def run() -> None:
        init_ws_context(session)
        await send()

    await asyncio.create_task(run())


async def push(room: Room, session_id: str, send: Callable[[], Awaitable[Any]]) -> None:
    try:
        await push_to(session_id, send)
    except Exception:
        # The tab was closed without on_chat_end running.
        room.members.pop(session_id, None)


async def say(
    room: Room,
    content: str,
    actions: list[tuple[str, str]] = (),
    only: str | None = None,
    exclude: set[str] = frozenset(),
) -> None:
    """Send a message to everyone in the room, to one member, or to all but some.

    ``actions`` are (name, label) pairs; messages and actions are built inside
    each session's context.
    """

    targets = [only] if only else [s for s in room.members if s not in exclude]

    for session_id in targets:
        await push(room, session_id, lambda: cl.Message(
            content=content,
            actions=[cl.Action(name=name, payload={}, label=label) for name, label in actions],
        ).send())


class SharedStep:
    """A collapsible step shown in every session of a room."""

    def __init__(self, room: Room, name: str):
        self.room = room
        self.name = name
        self.steps: dict[str, cl.Step] = {}

    async def open(self) -> None:
        for session_id in list(self.room.members):
            async def create(session_id=session_id) -> None:
                step = cl.Step(name=self.name, type="llm")
                await step.send()
                self.steps[session_id] = step

            await push(self.room, session_id, create)

    async def close(self, output: str) -> None:
        for session_id, step in self.steps.items():
            async def update(step=step) -> None:
                step.output = output
                await step.update()

            await push(self.room, session_id, update)


# --- formatting --------------------------------------------------------------


def format_step(node: str, update: dict) -> str:
    if node == "initialize":
        return "\n".join(
            f"{index}. {item['description']}"
            for index, item in enumerate(update.get("agenda", []), start=1)
        )

    if node == "advance_agenda":
        if update.get("agenda_complete"):
            return "All agenda items handled."
        return f"Now discussing: {update.get('current_agenda', '')}"

    if node == "load_project_memory":
        context = update.get("project_context")
        if not context:
            return "No project memory used."
        counts = ", ".join(
            f"{len(items)} {field.replace('_', ' ')}"
            for field, items in context.get("current_state", {}).items()
        )
        return f"{len(context.get('history', []))} earlier meetings; {counts}."

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

    for field_name, title in ARTIFACT_TITLES.items():
        items = update.get(field_name, [])
        lines.append(f"**{title}** ({len(items)})")

        for item in items:
            text = item.get("description") or item.get("content", "")
            label = item.get("status") or item.get("position", "")
            lines.append(f"- {text} _({label})_")

        lines.append("")

    updates = update.get("meeting_record", {}).get("artifact_updates", [])
    if updates:
        lines.append(f"**Status updates** ({len(updates)})")
        lines.extend(f"- `{u['id'][:8]}` → {u['status']}" for u in updates)
        lines.append("")

    for warning in update.get("extraction_warnings", []):
        lines.append(f"_Warning: {warning}_")

    return "\n".join(lines).strip()


# --- participants ------------------------------------------------------------


def slug(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "participant"
    candidate, suffix = base, 2
    while candidate in taken:
        candidate, suffix = f"{base}-{suffix}", suffix + 1
    taken.add(candidate)
    return candidate


def parse_participants(text: str, taken: set[str] | None = None) -> list[dict]:
    """Read AI participants from "Name: role | instructions" lines.

    The role and the instructions are optional.
    """

    taken = set() if taken is None else taken
    participants = []

    for line in (text or "").splitlines():
        line, _, instructions = line.partition("|")
        name, _, role = line.partition(":")
        name = name.strip()

        if not name:
            continue

        participant = {
            "id": slug(name, taken),
            "name": name,
            "role": role.strip() or name.lower(),
        }

        if instructions.strip():
            participant["instructions"] = instructions.strip()

        participants.append(participant)

    return participants


def meeting_participants(room: Room, ai_text: str) -> list[dict]:
    """AI participants from settings plus everyone in the room as a human.

    A person alone in a room without AI participants is left out, so a solo
    meeting works as before.
    """

    taken: set[str] = set()
    ai = parse_participants(ai_text, taken)
    room.humans = {}

    if not ai and len(room.members) < 2:
        return []

    humans = []
    for session_id, name in room.members.items():
        participant_id = slug(name, taken)
        room.humans[participant_id] = session_id
        human = {
            "id": participant_id,
            "name": name,
            "role": "host" if session_id == room.host else "human participant",
            "is_human": True,
        }
        if room.languages.get(session_id):
            human["language"] = room.languages[session_id]
        humans.append(human)

    return ai + humans


def display_name(session_id: str, room: Room | None = None) -> str:
    name = (cl.user_session.get("settings") or {}).get("name", "").strip()
    if name:
        return name
    count = len(room.members) + 1 if room else 1
    return f"Guest {count}"


def own_language() -> str:
    return (cl.user_session.get("settings") or {}).get("my_language", "").strip()


def make_model(name: str):
    # The scripted demo model keeps its place in its script, so each meeting
    # gets a fresh one.
    return demo_model() if name == "fake" else init_chat_model(name, temperature=0)


# --- running a meeting -------------------------------------------------------


async def run_meeting(room: Room, graph_input) -> None:
    """Stream the graph to everyone in the room until it pauses or finishes."""

    config = {
        "configurable": {
            "thread_id": room.thread_id,
            "meeting_model": room.model,
            "meeting_store": store,
        }
    }

    room.phase = "running"
    steps: dict[str, SharedStep] = {}

    try:
        async for event in graph.astream(graph_input, config, stream_mode="tasks"):
            node = event["name"]

            if node not in STEP_NAMES:
                continue

            if "input" in event:
                step = SharedStep(room, STEP_NAMES[node])
                await step.open()
                steps[event["id"]] = step
                continue

            step = steps.pop(event["id"], None)

            if step is not None:
                await step.close(
                    f"Error: {event['error']}"
                    if event["error"]
                    else format_step(node, event["result"])
                )

    except Exception as error:
        room.phase = "done"
        await say(room, f"The meeting stopped with an error: {error}")
        return

    snapshot = await graph.aget_state(config)

    if snapshot.interrupts:
        await ask(room, snapshot.interrupts[0].value)
        return

    await finish_meeting(room, snapshot.values)


async def ask(room: Room, payload: dict) -> None:
    room.phase = "awaiting_feedback"
    question = payload.get("question", "")
    asked = payload.get("participant")
    room.asked = room.humans.get(asked["id"]) if asked else None
    decline = [("decline_feedback", "Continue without me")]

    if room.asked and room.asked in room.members:
        name = room.members[room.asked]
        await say(room, f"**Moderator asks you:** {question}", decline, only=room.asked)
        await say(
            room,
            f"**Moderator asks {name}:** {question}\n\n_Waiting for {name} to answer._",
            exclude={room.asked},
        )
        return

    room.asked = None
    hint = "\n\n_Anyone in the room can answer._" if len(room.members) > 1 else ""
    await say(room, f"**Moderator asks:** {question}{hint}", decline)


async def answer(room: Room, session_id: str, text: str, decline: bool = False) -> None:
    name = room.members.get(session_id, "Someone")
    participant_id = next(
        (pid for pid, sid in room.humans.items() if sid == session_id), None
    )

    # Set before any await so a second, simultaneous answer is refused.
    room.phase = "running"
    room.asked = None

    if decline:
        await say(room, f"_{name} let the meeting continue without more questions._")
    elif len(room.members) > 1:
        await say(room, f"**{name}:** {text}", exclude={session_id})

    resume = {"feedback": "" if decline else text, "decline_future_questions": decline}
    if participant_id:
        resume["participant_id"] = participant_id

    await run_meeting(room, Command(resume=resume))


async def finish_meeting(room: Room, state: dict) -> None:
    room.phase = "done"

    await say(room, f"## Meeting report\n\n{state.get('meeting_summary', '')}")

    if state.get("meeting_saved"):
        note = f"Saved as meeting `{state['meeting_id']}`."
    else:
        note = "No meeting record was saved."

    host = room.members.get(room.host, "the host")
    await say(
        room,
        f"{note}\n\nSend a new message to start another meeting.",
        [("show_meetings", "Past meetings")],
        only=room.host,
    )
    await say(
        room,
        f"{note}\n\nWaiting for {host} to start another meeting.",
        [("show_meetings", "Past meetings")],
        exclude={room.host},
    )


async def start_meeting(room: Room, text: str) -> None:
    settings = cl.user_session.get("settings")

    try:
        room.model = make_model(settings["model"])
    except Exception as error:
        await cl.Message(content=f"Could not load the model: {error}").send()
        return

    topic, _, rest = text.partition("\n")
    topic = topic.strip()
    lines = [line.strip() for line in rest.splitlines()]
    items = [line[2:].strip() for line in lines if line.startswith("- ")]
    background = "\n".join(line for line in lines if not line.startswith("- "))

    room.thread_id = str(uuid.uuid4())

    if len(room.members) > 1:
        await say(room, f"**{room.members[room.host]}** started the meeting: {topic}",
                  exclude={room.host})

    await run_meeting(room, {
        "project_id": settings["project_id"].strip(),
        "language": settings.get("language", "").strip(),
        "topic": topic,
        "background": background.strip(),
        "agenda": [
            {"id": new_id(), "description": description}
            for description in (items or [topic])
        ],
        "participants": meeting_participants(room, settings.get("participants", "")),
        "max_rounds_per_agenda": int(settings["max_rounds"]),
    })


# --- room commands -----------------------------------------------------------


def create_room(session_id: str) -> Room:
    room = Room(code=new_code(), host=session_id)
    room.members[session_id] = display_name(session_id)
    room.languages[session_id] = own_language()
    ROOMS[room.code] = room
    return room


async def leave_room(session_id: str) -> None:
    room = room_of(session_id)
    if room is None:
        return

    name = room.members.pop(session_id, "Someone")
    room.languages.pop(session_id, None)

    if not room.members:
        ROOMS.pop(room.code, None)
        return

    await say(room, f"_{name} left the meeting._")

    if room.host == session_id:
        room.host = next(iter(room.members))
        await say(room, "You are now the host.", only=room.host)

    if room.asked == session_id:
        # Nobody else was asked, so let anyone answer.
        room.asked = None
        await say(room, f"_{name} was asked a question; anyone can answer now._")


async def join_room(session_id: str, code: str) -> None:
    room = ROOMS.get(code.strip().upper())

    if room is None:
        await cl.Message(content=f"No meeting with code `{code.strip()}`.").send()
        return

    if session_id in room.members:
        await cl.Message(content=f"You are already in meeting `{room.code}`.").send()
        return

    await leave_room(session_id)
    name = display_name(session_id, room)
    room.members[session_id] = name
    room.languages[session_id] = own_language()

    await say(room, f"_{name} joined._", exclude={session_id})

    host = room.members.get(room.host, "the host")
    status = {
        "lobby": f"Waiting for {host} to start the meeting.",
        "running": "The meeting is running; you will see it from here on.",
        "awaiting_feedback": "The moderator is waiting for an answer.",
        "done": f"The last meeting has finished. Waiting for {host} to start another.",
    }[room.phase]

    others = ", ".join(n for s, n in room.members.items() if s != session_id)
    await cl.Message(
        content=f"Joined meeting `{room.code}` as **{name}** with {others}.\n\n{status}"
    ).send()


# --- Chainlit handlers -------------------------------------------------------


WELCOME = (
    "What should this meeting be about?\n\n"
    "Put the topic on the first line. Lines starting with `- ` become agenda "
    "items; anything else below the topic is background.\n\n"
    "To hold the meeting with others, send `/room` first and share the code. "
    "They join with `/join CODE`. Set your name in the settings panel."
)


@cl.on_chat_start
async def start() -> None:
    settings = await cl.ChatSettings(
        [
            TextInput(id="name", label="Your name", initial=""),
            TextInput(
                id="my_language",
                label="Your language (optional; questions to you use it)",
                initial="",
            ),
            TextInput(
                id="language",
                label="Meeting language (blank: the language you write in)",
                initial="",
            ),
            TextInput(
                id="model",
                label="Model (provider:model, or 'fake' for an offline demo)",
                initial=DEFAULT_MODEL,
            ),
            TextInput(
                id="project_id",
                label="Project ID (optional)",
                initial="",
            ),
            TextInput(
                id="participants",
                label="AI participants (one per line, 'Name: role | instructions')",
                initial="",
                placeholder=(
                    "Researcher: develops alternative strategies\n"
                    "Risk analyst: identifies risks | Always quantify the downside."
                ),
                multiline=True,
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

    await cl.Message(content=WELCOME, actions=[past_meetings_action()]).send()


@cl.on_settings_update
async def update_settings(settings: dict) -> None:
    cl.user_session.set("settings", settings)

    session_id = current_session()
    room = room_of(session_id)
    name = settings.get("name", "").strip()

    if room:
        room.languages[session_id] = own_language()

    if room and name and room.members.get(session_id) != name:
        old = room.members[session_id]
        room.members[session_id] = name
        await say(room, f"_{old} is now {name}._", exclude={session_id})


@cl.on_chat_end
async def end() -> None:
    await leave_room(current_session())


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session_id = current_session()
    room = room_of(session_id)
    text = message.content.strip()
    command = text.lower()

    if command == "/room":
        if room is None:
            room = create_room(session_id)
        await cl.Message(content=(
            f"Meeting code: **{room.code}**\n\n"
            f"Others join with `/join {room.code}`. Send the topic when everyone "
            "is here."
        )).send()
        return

    if command.startswith("/join"):
        await join_room(session_id, text[len("/join"):])
        return

    if command == "/leave":
        await leave_room(session_id)
        await cl.Message(content="You left the meeting.").send()
        return

    if room is None:
        room = create_room(session_id)

    if room.phase == "awaiting_feedback":
        if room.asked and room.asked != session_id:
            name = room.members.get(room.asked, "someone else")
            await cl.Message(content=f"The moderator asked {name}; please wait.").send()
            return
        await answer(room, session_id, text)
        return

    if room.phase == "running":
        await cl.Message(content="The meeting is still running.").send()
        return

    if session_id != room.host:
        host = room.members.get(room.host, "the host")
        await cl.Message(content=f"Only {host} can start the meeting.").send()
        return

    await start_meeting(room, text)


@cl.action_callback("decline_feedback")
async def decline_feedback(action: cl.Action) -> None:
    session_id = current_session()
    room = room_of(session_id)

    if room is None or room.phase != "awaiting_feedback":
        return
    if room.asked and room.asked != session_id:
        return

    await action.remove()
    await answer(room, session_id, "", decline=True)


def past_meetings_action() -> cl.Action:
    return cl.Action(name="show_meetings", payload={}, label="Past meetings")


@cl.action_callback("show_meetings")
async def show_meetings(action: cl.Action) -> None:
    project_id = cl.user_session.get("settings", {}).get("project_id", "").strip()
    records = store.list_meetings(project_id or None, limit=20)

    if not records:
        scope = f" for project `{project_id}`" if project_id else ""
        await cl.Message(content=f"No saved meetings{scope} yet.").send()
        return

    scope = f"project `{project_id}`" if project_id else "all projects"
    await cl.Message(
        content=f"**Past meetings** ({scope}, newest first). Pick one to open its report.",
        actions=[
            cl.Action(
                name="show_report",
                payload={"meeting_id": record["meeting_id"]},
                label=(record.get("topic") or record["meeting_id"])[:60],
            )
            for record in records
        ],
    ).send()


@cl.action_callback("show_report")
async def show_report(action: cl.Action) -> None:
    record = store.get(action.payload.get("meeting_id", ""))

    if record is None:
        await cl.Message(content="That meeting is no longer saved.").send()
        return

    project = f" · project `{record['project_id']}`" if record.get("project_id") else ""
    agenda = "\n".join(
        f"- {item.get('description', '')} _({item.get('status', '')})_"
        for item in record.get("agenda_items", [])
    )

    await cl.Message(
        content=(
            f"## {record.get('topic', 'Meeting')}\n"
            f"`{record['meeting_id']}`{project}\n\n"
            + (f"**Agenda**\n{agenda}\n\n" if agenda else "")
            + f"{record.get('meeting_summary') or '_No report saved._'}\n\n"
            "---\n\n"
            + format_artifacts({**record, "meeting_record": record})
        ),
        actions=[past_meetings_action()],
    ).send()
