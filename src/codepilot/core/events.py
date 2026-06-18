from __future__ import annotations

"""Agent event envelope helpers."""

import asyncio
import time
from typing import Any, cast

from codepilot.protocols import AgentEvent, AgentEventSink


def now_ms() -> int:
    """Return the current timestamp in milliseconds."""

    return int(time.time() * 1000)


async def maybe_await(value: Any) -> Any:
    """Await coroutine/future values and pass through plain values."""

    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


class AgentEventEmitter:
    """Adds stable run/turn/event metadata before forwarding events."""

    def __init__(
        self,
        sink: AgentEventSink,
        *,
        run_id: str,
        session_id: str | None = None,
    ) -> None:
        self._sink = sink
        self._session_id = session_id
        self.run_id = run_id
        self.turn_id = 0
        self._event_seq = 0

    async def emit(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "turn_start":
            self.turn_id += 1

        self._event_seq += 1
        enriched = {
            **event,
            "runId": self.run_id,
            "turnId": self.turn_id,
            "eventId": f"{self.run_id}:{self._event_seq}",
            "timestamp": now_ms(),
            "sessionId": self._session_id,
        }
        await maybe_await(self._sink(cast(AgentEvent, enriched)))
