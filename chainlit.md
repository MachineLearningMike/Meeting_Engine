# Meeting Engine

A structured human-AI meeting. Start by sending the topic on the first line.
Lines below it starting with `- ` become agenda items, discussed one at a
time; other lines are background.

Each stage (context analysis, deliberation, moderator) appears as a
collapsible step. When the moderator needs your judgment, answer in the chat,
or press **Continue without me** to let the meeting proceed on its own.

Use the settings panel to choose the model (`provider:model`, for example
`openai:gpt-4o` or `anthropic:claude-sonnet-5-5`), a project ID, and the
maximum number of deliberation rounds, and AI participants: one per line as
`Name: role`, optionally followed by `| instructions` (for example
`Risk analyst: identifies risks | Always quantify the downside.`). Each speaks
in every round from its role; without any, a single
analyst deliberates. Finished meetings are saved to
`data/meetings.db`.

## Meeting with others

Set your name in the settings panel. One person sends `/room` and shares the
code; the others send `/join CODE`. When everyone is there, the host sends the
topic. Everyone sees the meeting as it runs. A question addressed to someone
goes to that person only; anyone can answer a general question. `/leave`
leaves the room.

## Languages

Write in any language; the meeting follows it. To fix the language, set
**Meeting language** (for example `日本語`). **Your language** makes the
moderator ask you in that language, which helps in mixed-language rooms.
