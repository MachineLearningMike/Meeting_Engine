# Meeting Engine

A structured human-AI meeting. Start by sending the topic on the first line.
Lines below it starting with `- ` become agenda items, discussed one at a
time; other lines are background.

Each stage (context analysis, deliberation, moderator) appears as a
collapsible step. When the moderator needs your judgment, answer in the chat,
or press **Continue without me** to let the meeting proceed on its own.

Use the settings panel to choose the model (`provider:model`, for example
`openai:gpt-4o` or `anthropic:claude-sonnet-5-5`), a project ID, and the
maximum number of deliberation rounds. Finished meetings are saved to
`data/meetings.db`.
