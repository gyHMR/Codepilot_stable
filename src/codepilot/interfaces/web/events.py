"""把 Runtime 帧投影为 Web 事件，并支持 SSE 有界重放。"""

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
    """Web/SSE 对 Runtime 帧的稳定事件投影。"""
    event_id: str
    session_id: str
    run_id: str | None = None
    type: str
    sequence: int = Field(ge=1)
    timestamp: str
    data: dict[str, Any]


@dataclass(frozen=True)
class ReplayResult:
    """按事件 ID 重放得到的事件集合及游标过期标记。"""
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
        event_type, payload = _project_progress(payload)
    elif kind == "approval_required":
        payload = _json_dict(getattr(frame, "approval", {}))
        if "risk_level" not in payload and "risk" in payload:
            payload["risk_level"] = payload["risk"]
        event_type = "approval.requested"
    elif kind in {"run_paused", "run_finished", "command_finished"}:
        payload = _json_dict(getattr(frame, "record", {}))
        if kind == "run_paused":
            payload["checkpoint"] = _json_value(getattr(frame, "checkpoint", {}))
            waiting = payload["checkpoint"].get("waiting", {}) if isinstance(payload["checkpoint"], dict) else {}
            wait_kind = str(waiting.get("kind", "")) if isinstance(waiting, dict) else ""
            wait_payload = waiting.get("payload", {}) if isinstance(waiting, dict) else {}
            payload.update({
                "status": {
                    "tool_approval": "waiting_approval",
                    "user_input": "waiting_user",
                    "plan_confirmation": "waiting_plan",
                    "continuation": "waiting_continuation",
                }.get(wait_kind, "paused"),
                "kind": wait_kind,
                "request_id": str(waiting.get("request_id", "")) if isinstance(waiting, dict) else "",
                "payload": wait_payload if isinstance(wait_payload, dict) else {},
            })
            event_type = {
                "user_input": "interaction.requested",
                "plan_confirmation": "plan.confirmation_requested",
                "continuation": "continuation.requested",
            }.get(wait_kind, "run.status_changed")
        else:
            payload["status"] = "completed"
            event_type = "run.completed"
    elif kind == "cancelled":
        payload = {
            "cancelled": bool(getattr(frame, "cancelled", False)),
            "reason": str(getattr(frame, "reason", "user")),
        }
        payload["status"] = "cancelled"
        event_type = "run.completed"
    elif kind == "failed":
        error = getattr(frame, "error", "Runtime failed")
        payload = _json_dict(error) if isinstance(error, dict) else {"message": str(error)}
        event_type = "error"
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
    """维护有界事件缓存和实时订阅队列，不持久化业务状态。"""
    def __init__(self, capacity: int = 256, subscriber_capacity: int = 256) -> None:
        if capacity < 1 or subscriber_capacity < 1:
            raise ValueError("Event capacities must be positive")
        self._events: deque[WebEvent] = deque(maxlen=capacity)
        self._subscribers: set[asyncio.Queue[WebEvent]] = set()
        self._subscriber_capacity = subscriber_capacity
        self._overflowed: set[asyncio.Queue[WebEvent]] = set()

    async def publish(self, event: WebEvent) -> None:
        self._events.append(event)
        for queue in tuple(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Never leave an SSE client waiting forever after dropping deltas.
                # Replace the backlog with an explicit resync barrier; the route
                # closes the stream after delivering it and the browser reconnects.
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                sync = WebEvent(
                    event_id=uuid4().hex,
                    session_id=event.session_id,
                    run_id=event.run_id,
                    type="sync_required",
                    sequence=event.sequence,
                    timestamp=event.timestamp,
                    data={"reason": "subscriber_queue_overflow"},
                )
                queue.put_nowait(sync)
                self._overflowed.add(queue)

    def subscribe(self) -> asyncio.Queue[WebEvent]:
        queue: asyncio.Queue[WebEvent] = asyncio.Queue(self._subscriber_capacity)
        self._subscribers.add(queue)
        return queue

    def subscribe_after(
        self,
        event_id: str | None,
    ) -> tuple[ReplayResult, asyncio.Queue[WebEvent]]:
        """Atomically capture replay state and register the live subscriber."""

        queue = self.subscribe()
        return self.replay_after(event_id), queue

    def unsubscribe(self, queue: asyncio.Queue[WebEvent]) -> None:
        self._subscribers.discard(queue)
        self._overflowed.discard(queue)

    def should_close(self, queue: asyncio.Queue[WebEvent]) -> bool:
        return queue in self._overflowed

    def replay_after(self, event_id: str | None) -> ReplayResult:
        events = tuple(self._events)
        if event_id is None:
            return ReplayResult(events=())
        for index, event in enumerate(events):
            if event.event_id == event_id:
                return ReplayResult(events=events[index + 1 :])
        return ReplayResult(events=(), expired=True)

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


def _project_progress(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Project Runtime's internal progress envelope to the stable Web protocol."""

    progress_type = str(payload.get("type", ""))
    if progress_type == "message_update":
        assistant_event = payload.get("assistant_message_event")
        if isinstance(assistant_event, Mapping):
            projected = _json_dict(assistant_event)
            assistant_type = str(projected.get("type", ""))
            if assistant_type == "text_delta":
                return "assistant.delta", {
                    "delta": str(projected.get("delta", "")),
                }
            if assistant_type in {"thinking_delta", "reasoning_delta"}:
                return "activity.updated", {
                    "activity_id": "model-thinking",
                    "type": "thinking",
                    "name": "思考",
                    "status": "running",
                    "append_summary": str(projected.get("delta", "")),
                }
            if assistant_type.startswith("tool_call"):
                raw_call = projected.get("toolCall")
                call = _json_dict(raw_call) if raw_call is not None else {}
                call_id = str(call.get("id") or "").strip()
                index = call.get("index", projected.get("contentIndex"))
                activity_id = call_id or (
                    f"tool-call-index-{index}" if index is not None else "tool-call-current"
                )
                activity = {
                    "activity_id": activity_id,
                    "type": "tool_call",
                    "status": "running",
                }
                if call.get("name"):
                    activity["name"] = str(call["name"])
                if call.get("arguments"):
                    activity["arguments"] = call["arguments"]
                if call.get("raw_arguments"):
                    activity["raw_arguments"] = str(call["raw_arguments"])
                return "activity.updated", activity
    if progress_type == "text_delta":
        return "assistant.delta", {"delta": str(payload.get("delta", ""))}
    if progress_type.startswith("tool_"):
        return "activity.updated", payload
    payload.setdefault("status", progress_type or "running")
    return "run.status_changed", payload


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
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
