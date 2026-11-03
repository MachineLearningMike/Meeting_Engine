"""Load project memory at the start of a meeting and save the record at the end.

Both nodes use the store supplied through config["configurable"]["meeting_store"]
and do nothing when no store is given.
"""

from langchain_core.runnables import RunnableConfig

from ..history import load_project_context
from .common import get_store


def load_project_memory(
    state: dict,
    config: RunnableConfig,
) -> dict:
    store = get_store(config)
    project_id = state.get("project_id")

    # A caller may supply project_context directly; keep it.
    if store is None or not project_id or state.get("project_context"):
        return {}

    return {"project_context": load_project_context(store, project_id)}


def save_meeting(
    state: dict,
    config: RunnableConfig,
) -> dict:
    store = get_store(config)
    record = state.get("meeting_record")

    if store is None or not record:
        return {}

    store.save(record)
    return {"meeting_saved": True}
