# Meeting Engine

A reusable, model-independent LangGraph engine for structured human-AI meetings.

## Project layout

```text
Meeting_Engine/
├── main.py                  # Interactive demo (choose a model with --model)
├── meeting_engine/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py             # Builds and compiles the meeting graph
│   ├── schemas.py           # TypedDict artifact schemas
│   ├── state.py             # LangGraph state
│   ├── storage.py           # SQLite storage for meeting records
│   └── nodes/
│       ├── initialization.py
│       ├── analysis.py
│       ├── deliberation.py
│       ├── moderator.py
│       ├── human_feedback.py
│       ├── finalize.py
│       └── extract_artifacts.py
├── test_graph.py
├── test_state.py
└── test_storage.py
```

## Meeting flow

```text
START
  |
initialize
  |
analyze_context
  |
deliberate
  |
moderate
  |
  +---- continue ------> deliberate
  |
  +---- request input -> interrupt
  |                         |
  |                       resume
  |                         |
  |                    analyze_context
  |
  +---- conclude ------> finalize
                            |
                     extract_artifacts
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
