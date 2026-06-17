from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RuntimeEventType = Literal[
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_approval_required",
    "tool_approval_resolved",
    "tool_execution_end",
    "file_diff",
    "error",
]


@dataclass
class EventEnvelope:
    type: RuntimeEventType
    payload: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    turn_id: str = ""
    event_id: str = ""
    timestamp: int = 0
    session_id: str = ""


RuntimeEvent = EventEnvelope
AgentEvent = EventEnvelope


__all__ = [
    "AgentEvent",
    "EventEnvelope",
    "RuntimeEvent",
    "RuntimeEventType",
]
