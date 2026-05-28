from __future__ import annotations

from .state import SoxxGraphState


def continue_or_stop(state: SoxxGraphState) -> str:
    if state.get("status") == "failed":
        return "stop"
    return "continue"
