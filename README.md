

Meeting_Engine/
│
├── meeting_engine/
│   ├── __init__.py
│   ├── schemas.py
│   ├── state.py
│   └── config.py
│
└── test_state.py



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
                           END



                           