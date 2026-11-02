"""Provider-independent configuration for the meeting engine.

The graph core accepts a model object through dependency injection. Import a
provider integration (for example, ChatOpenAI or ChatOllama) in the application
that constructs the graph, not inside the core state/schema modules.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MeetingConfig:
    model: Any
    max_rounds_per_agenda: int = 5
    max_proposals_per_problem: int = 5
    require_human_approval_for_decisions: bool = True
    moderator_temperature_note: Optional[str] = None
