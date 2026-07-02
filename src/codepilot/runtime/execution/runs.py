from __future__ import annotations

"""Active run tracking for RuntimeService."""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, cast


ActiveRunStatus = Literal["running", "completed", "failed", "aborted"]
_ACTIVE_RUN_STATUSES = frozenset({"running", "completed", "failed", "aborted"})


@dataclass
class ActiveRun:
    """One in-flight RuntimeService run."""

    run_id: str
    session_id: str
    task: asyncio.Task[Any] | None = None
    started_at: float = 0
    status: ActiveRunStatus = "running"

    def __post_init__(self) -> None:
        self.status = ensure_active_run_status(self.status)


def ensure_active_run_status(value: object) -> ActiveRunStatus:
    if value not in _ACTIVE_RUN_STATUSES:
        raise ValueError(f"Unknown active run status: {value}")
    return cast(ActiveRunStatus, value)


class ActiveRunTracker:
    """Owns single-active-run bookkeeping per session."""

    def __init__(self) -> None:
        self._runs: dict[str, ActiveRun] = {}

    def get(self, session_id: str) -> ActiveRun | None:
        active_run = self._runs.get(session_id)
        if active_run is None:
            return None
        if active_run.task is not None and active_run.task.done():
            self._runs.pop(session_id, None)
            return None
        return active_run

    def create(self, session_id: str) -> ActiveRun:
        active_run = ActiveRun(
            run_id=uuid.uuid4().hex,
            session_id=session_id,
            started_at=time.time(),
        )
        self._runs[session_id] = active_run
        return active_run

    def discard(self, session_id: str) -> None:
        self._runs.pop(session_id, None)

    def session_ids(self) -> list[str]:
        return list(self._runs)

    async def cancel(self, session_id: str) -> bool:
        active_run = self.get(session_id)
        if active_run is None:
            return False
        if active_run.task is None:
            active_run.status = "aborted"
            self.discard(session_id)
            return True
        active_run.task.cancel()
        try:
            await active_run.task
        except asyncio.CancelledError:
            pass
        finally:
            active_run.status = "aborted"
            self.discard(session_id)
        return True


__all__ = [
    "ActiveRun",
    "ActiveRunStatus",
    "ActiveRunTracker",
    "ensure_active_run_status",
]
