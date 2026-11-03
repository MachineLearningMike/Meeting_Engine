# Meeting Engine

A reusable, model-independent LangGraph engine for structured human-AI meetings.

## Project layout

```text
Meeting_Engine/
├── app.py                   # Chainlit web chat UI
├── main.py                  # Console demo (choose a model with --model)
├── meeting_engine/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py             # Builds and compiles the meeting graph
│   ├── history.py           # Loads a project's current state and past meetings
│   ├── schemas.py           # TypedDict artifact schemas
│   ├── state.py             # LangGraph state
│   ├── storage.py           # SQLite: meeting records + current project state
│   └── nodes/
│       ├── initialization.py
│       ├── agenda.py
│       ├── analysis.py
│       ├── deliberation.py
│       ├── moderator.py
│       ├── human_feedback.py
│       ├── finalize.py
│       ├── project_memory.py
│       └── extract_artifacts.py
└── tests/
    ├── fakes.py             # Scripted test model, answers per node
    ├── test_agenda.py
    ├── test_participants.py
    ├── test_graph.py
    ├── test_artifacts.py
    ├── test_history.py
    └── test_storage.py
```

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
deliberate  <-----------+                        |
  |                     |                        |
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
                       extract_artifacts
                              |
                         save_meeting     (if a store is given)
                              |
                             END
```

## Running

The model is chosen at run time as `provider:model`. Install the matching
provider package (see `requirements.txt`) and set its API key.

```bash
python main.py --model openai:gpt-4o
python main.py --model anthropic:claude-sonnet-5-5
python main.py --model ollama:llama3.1
```

`MEETING_MODEL` can be set instead of `--model`.

### Setup

Install the package in editable mode once, so `meeting_engine` can be
imported from anywhere:

```bash
pip install -e .
```

### Tests

```bash
pytest
```

A single test file can also be run directly, e.g. `python tests/test_graph.py`.

The tests use a scripted model and temporary databases, so they need no API
key or input.

### Web chat UI

```bash
chainlit run app.py
```

This opens a chat in the browser. Each meeting stage appears as a collapsible
step, and the moderator's questions are answered in the chat. The model,
project ID and round limit are set in the settings panel; the model defaults
to `MEETING_MODEL` or `openai:gpt-4o`.
