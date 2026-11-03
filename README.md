# Meeting Engine

A reusable, model-independent LangGraph engine for structured human-AI
meetings. It keeps the state of a project across meetings: proposals,
arguments, decisions, action items and open issues, each with a tracked
status, so every meeting continues from where the last one stopped.

## Quick start

```bash
pip install -e .                 # once, in your environment
pip install langchain-openai     # or another provider package
pytest                           # no API key needed
```

Run the example project, one meeting at a time:

```bash
python examples/trading_project.py 1 --model openai:gpt-4o
python examples/trading_project.py 2 --model openai:gpt-4o
```

## Using the engine

On its own, give the graph a checkpointer so human feedback can pause and
resume:

```python
from langgraph.checkpoint.memory import InMemorySaver
from meeting_engine import MeetingStore, build_meeting_graph

graph = build_meeting_graph(InMemorySaver())
config = {"configurable": {
    "thread_id": "meeting-1",
    "meeting_model": model,          # any LangChain chat model
    "meeting_store": MeetingStore(), # optional: project memory
}}
result = graph.invoke({"project_id": "trading-ai", "topic": "..."}, config)
```

When the moderator asks a question, `result["__interrupt__"]` holds it.
Answer with `graph.invoke(Command(resume={"feedback": "..."}), config)`, or
pass `"decline_future_questions": True` to let the meeting go on alone.

Inside a parent graph, leave the checkpointer out; the parent's is used, and
interrupts pause the parent:

```python
parent_builder.add_node("meeting", build_meeting_graph())
```

The transcript stays inside the meeting, so it never mixes with a parent's
`messages`. A parent with differently named state can call the graph from its
own node instead.

### Input (`MeetingInput`)

| Key | Meaning |
| --- | --- |
| `topic`, `background` | What the meeting is about |
| `project_id` | Meetings with the same id share project memory |
| `agenda` | Items discussed one at a time; if missing, built from `problems` and `objectives`, else the topic |
| `participants` | AI participants speak in every round from their role; human ones answer questions |
| `max_rounds_per_agenda` | Round limit per agenda item (default 3) |
| `max_total_rounds` | Limit for the whole meeting (default: per-item limit × (items + 1)) |
| `project_context` | Supply project memory yourself instead of loading it from the store |

### Output (`MeetingOutput`)

`meeting_id`, `meeting_record` (all artifacts, status updates and agenda
items), `meeting_summary`, `agenda`, `extraction_warnings`,
`extraction_error`, `meeting_saved`. The full internal state is available
through `graph.get_state(config)`.

### Configuration (`config["configurable"]`)

| Key | Meaning |
| --- | --- |
| `meeting_model` | Required. Any LangChain chat model |
| `participant_models` | `{key: model}` for participants with `"model": key` |
| `meeting_store` | A `MeetingStorage`; loads project memory at the start and saves the record at the end |
| `thread_id` | Identifies the meeting for checkpointing |

## Concepts

**Artifacts.** Each meeting extracts proposals, arguments, decisions, action
items and unresolved issues. Items get stable ids; later meetings refer to
them and update their status instead of creating copies. Arguments point at
the proposal they support or oppose, and a superseded decision points at its
replacement. Only a human participant can approve a decision.

**Lifecycle.** Status changes follow `ALLOWED_TRANSITIONS` in
`meeting_engine/schemas.py`, for example proposed → under_discussion →
accepted / rejected / deferred. Final statuses never change.

**Two kinds of persistence**, kept separate:

- *Project memory* (`MeetingStore`, SQLite): each meeting's record as history,
  plus each project's current state. A new meeting sees both, clearly
  labelled: current state is authoritative, history is background.
- *Checkpoints* (any LangGraph checkpointer, e.g. `SqliteSaver`): the running
  meeting, so a meeting paused for feedback can resume, even after a restart.

## Meeting flow

```text
START
  |
initialize               (agenda from input, or from problems + objectives)
  |
load_project_memory      (current state + past meetings, if a store is given)
  |
analyze_context  <-------------------------------+
  |                                              |
deliberate  <-----------+   (each AI participant |
  |                     |    speaks in turn)     |
moderate                |                        |
  |                     |                        |
  +-- continue ---------+                        |
  |                                              |
  +-- request input --> interrupt --> resume ----+
  |                                              |
  +-- conclude / defer / revisit agenda item     |
  |       --> advance_agenda --- next item ------+
  |                 |
  |            no items left
  |                 |
  +-- conclude meeting --> finalize
                              |
                       extract_artifacts   (one retry on invalid output)
                              |
                         save_meeting     (if a store is given)
                              |
                             END
```

## Other ways to run it

```bash
python main.py --model anthropic:claude-sonnet-5-5 --project trading-ai
chainlit run app.py
```

`main.py` is a console meeting. `app.py` is a browser chat: each stage
appears as a collapsible step, and the settings panel holds the model,
project id, round limit and AI participants. Both default to
`MEETING_MODEL` or `openai:gpt-4o`.

## Project layout

```text
Meeting_Engine/
├── app.py                     # Chainlit web chat UI
├── main.py                    # Console meeting
├── examples/
│   └── trading_project.py     # Three meetings that build on each other
├── meeting_engine/
│   ├── graph.py               # Builds the meeting graph
│   ├── state.py               # MeetingState, MeetingInput, MeetingOutput
│   ├── schemas.py             # Artifact types and lifecycle rules
│   ├── storage.py             # SQLite: meeting records + project state
│   ├── history.py             # Loads project memory for a meeting
│   └── nodes/
│       ├── initialization.py
│       ├── project_memory.py
│       ├── analysis.py
│       ├── deliberation.py
│       ├── moderator.py
│       ├── human_feedback.py
│       ├── agenda.py
│       ├── finalize.py
│       └── extract_artifacts.py
└── tests/
    ├── fakes.py               # Scripted model that answers per node
    ├── test_graph.py
    ├── test_artifacts.py
    ├── test_storage.py
    ├── test_history.py
    ├── test_agenda.py
    ├── test_participants.py
    ├── test_subgraph.py
    └── test_integration.py
```

The tests use a scripted model and temporary databases, so they need no API
key or input. A single file can also be run directly, e.g.
`python tests/test_graph.py`.
