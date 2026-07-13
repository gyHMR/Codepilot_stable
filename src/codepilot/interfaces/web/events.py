from __future__ import annotations

import asyncio
from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field


class WebEvent(BaseModel):
    event_id: str
    session_id: str
    run_id: str | None = None
    type: str
    sequence: int = Field(ge=1)
    timestamp: str
    data: dict[str, Any]


@dataclass(frozen=True)
class ReplayResult:
    events: tuple[WebEvent, ...]
    expired: bool = False


def runtime_frame_to_event(
    frame: Any,
    *,
    session_id: str,
    sequence: int,
    event_id_factory: Callable[[], str] = lambda: uuid4().hex,
    clock: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat(),
) -> WebEvent:
    kind = str(getattr(frame, "kind", "progress"))
    run_id = _run_id(frame)
    if kind == "progress":
        payload = _json_dict(getattr(frame, "event", {}))
        progress_type = str(payload.get("type", ""))
        event_type = (
            "message_delta"
            if progress_type == "text_delta"
            else "tool_activity"
            if progress_type.startswith("tool_")
            else "progress"
        )
    elif kind == "approval_required":
        payload = _json_dict(getattr(frame, "approval", {}))
        event_type = kind
    elif kind in {"run_paused", "run_finished", "command_finished"}:
        payload = _json_dict(getattr(frame, "record", {}))
        if kind == "run_paused":
            payload["checkpoint"] = _json_value(getattr(frame, "checkpoint", {}))
        event_type = kind
    elif kind == "cancelled":
        payload = {
            "cancelled": bool(getattr(frame, "cancelled", False)),
            "reason": str(getattr(frame, "reason", "user")),
        }
        event_type = kind
    elif kind == "failed":
        error = getattr(frame, "error", "Runtime failed")
        payload = _json_dict(error) if isinstance(error, dict) else {"message": str(error)}
        event_type = kind
    else:
        payload = _json_dict(frame)
        event_type = kind
    return WebEvent(
        event_id=event_id_factory(),
        session_id=session_id,
        run_id=run_id,
        type=event_type,
        sequence=sequence,
        timestamp=clock(),
        data=payload,
    )


class EventHub:
    def __init__(self, capacity: int = 256, subscriber_capacity: int = 256) -> None:
        if capacity < 1 or subscriber_capacity < 1:
            raise ValueError("Event capacities must be positive")
        self._events: deque[WebEvent] = deque(maxlen=capacity)
        self._subscribers: set[asyncio.Queue[WebEvent]] = set()
        self._subscriber_capacity = subscriber_capacity

    async def publish(self, event: WebEvent) -> None:
        self._events.append(event)
        stale: list[asyncio.Queue[WebEvent]] = []
        for queue in tuple(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            self.unsubscribe(queue)

    def subscribe(self) -> asyncio.Queue[WebEvent]:
        queue: asyncio.Queue[WebEvent] = asyncio.Queue(self._subscriber_capacity)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[WebEvent]) -> None:
        self._subscribers.discard(queue)

    def replay_after(self, event_id: str | None) -> ReplayResult:
        events = tuple(self._events)
        if event_id is None:
            return ReplayResult(events=())
        for index, event in enumerate(events):
            if event.event_id == event_id:
                return ReplayResult(events=events[index + 1 :])
        return ReplayResult(events=(), expired=bool(events))

    @property
    def latest_sequence(self) -> int:
        return self._events[-1].sequence if self._events else 0


def _run_id(frame: Any) -> str | None:
    record = getattr(frame, "record", None)
    value = getattr(record, "run_id", None)
    return str(value) if value else None


def _json_dict(value: Any) -> dict[str, Any]:
    converted = _json_value(value)
    return converted if isinstance(converted, dict) else {"value": converted}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump())
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


__all__ = ["EventHub", "ReplayResult", "WebEvent", "runtime_frame_to_event"]
