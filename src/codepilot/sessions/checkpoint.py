from __future__ import annotations

"""Session checkpoint helpers.

Checkpoints are intentionally event-based for now. Later phases can attach
file snapshots and diff records without changing the AgentSession facade.
"""

from dataclasses import dataclass, field
from typing import Any

from .store import SessionStore


@dataclass(frozen=True)
class SessionCheckpoint:
    session_id: str
    label: str
    details: dict[str, Any] = field(default_factory=dict)


def record_checkpoint(
    store: SessionStore,
    *,
    session_id: str,
    label: str,
    details: dict[str, Any] | None = None,
) -> SessionCheckpoint:
    checkpoint = SessionCheckpoint(
        session_id=session_id,
        label=label,
        details=dict(details or {}),
    )
    store.append_event(
        {
            "type": "session_checkpoint",
            "session_id": checkpoint.session_id,
            "label": checkpoint.label,
            "details": checkpoint.details,
        }
    )
    return checkpoint
