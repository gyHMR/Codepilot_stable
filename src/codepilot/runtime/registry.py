"""登记活动 Session 和 Run，用于取消、查询与资源清理。"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Callable, TYPE_CHECKING

from codepilot.protocols import Message
from codepilot.core.plan import RunMode, ensure_run_mode
from codepilot.protocols import (
    AgentEvent,
    AgentEventSink,
    AgentRunResult,
    Message,
    Model,
    ThinkingLevel,
)

from .environment import RunResourceScope
from .config import RuntimePermissionMode

if TYPE_CHECKING:
    from .coordinator import SessionController


@dataclass
class SessionConversationState:
    """Live conversation projection for one opened Runtime session."""

    model: Model
    system_prompt: str = ""
    messages: list[Message] = field(default_factory=list)
    thinking_level: ThinkingLevel | str = "off"
    current_mode: RunMode = "build"
    stream_message: Message | None = None
    error: str | None = None
    active_tool_call_ids: set[str] = field(default_factory=set)
    last_run_result: AgentRunResult | None = None
    _steering_messages: list[Message] = field(default_factory=list)
    _listeners: list[AgentEventSink] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.system_prompt = str(self.system_prompt or "")
        self.messages = list(self.messages)
        self.current_mode = ensure_run_mode(self.current_mode)

    def set_messages(self, messages: list[Message]) -> None:
        self.messages = list(messages)

    def append_messages(self, messages: list[Message]) -> None:
        self.messages.extend(messages)

    def remember_result(self, result: AgentRunResult) -> None:
        self.last_run_result = result
        self.error = None if result.status == "completed" else self.error

    def set_current_mode(self, mode: RunMode | str) -> RunMode:
        self.current_mode = ensure_run_mode(mode)
        return self.current_mode

    def add_steering_message(self, message: Message) -> None:
        self._steering_messages.append(message)

    def drain_steering_messages(self) -> list[Message]:
        messages = list(self._steering_messages)
        self._steering_messages.clear()
        return messages

    def subscribe(self, listener: AgentEventSink) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def clear_listeners(self) -> None:
        self._listeners.clear()

    async def dispatch_event(self, event: AgentEvent) -> None:
        self._apply_event(event)
        for listener in list(self._listeners):
            value = listener(event)
            if inspect.isawaitable(value):
                await value

    def _apply_event(self, event: AgentEvent) -> None:
        event_type = event.get("type")
        if event_type in {"message_start", "message_update"}:
            self.stream_message = event.get("message")
        elif event_type == "message_end":
            self.stream_message = None
        elif event_type == "tool_started":
            tool_call_id = event.get("tool_call_id")
            if tool_call_id:
                self.active_tool_call_ids.add(str(tool_call_id))
        elif event_type in {"tool_completed", "tool_failed", "tool_interrupted"}:
            tool_call_id = event.get("tool_call_id")
            if tool_call_id:
                self.active_tool_call_ids.discard(str(tool_call_id))
        elif event_type == "error":
            self.error = str(event.get("error", "unknown error"))


@dataclass
class RuntimeStatusInfo:
    """Small status payload kept with an opened runtime session."""

    session_id: str
    model_id: str
    workspace: str
    permission_mode: RuntimePermissionMode
    credential_source: str = "unknown"
    warnings: tuple[str, ...] = ()


@dataclass
class RuntimeSession:
    """The live runtime objects needed to dispatch actions for one session."""

    controller: SessionController
    model_port: Any | None = None
    tool_port: Any | None = None
    status: RuntimeStatusInfo | None = None
    commands: dict[str, Any] | None = None
    mcp_manager: Any | None = None

    @property
    def session_id(self) -> str:
        return self.controller.session_id


class RuntimeSessionRegistry:
    """登记活动 Session 和 Run 资源的运行时注册中心。"""
    def __init__(self) -> None:
        self._items: dict[str, RuntimeSession] = {}

    def add(self, session: RuntimeSession) -> None:
        self._items[session.session_id] = session

    def require(self, session_id: str) -> RuntimeSession:
        try:
            return self._items[session_id]
        except KeyError as exc:
            raise KeyError(f"Session not found: {session_id}") from exc

    def close(self, session_id: str) -> RuntimeSession | None:
        entry = self._items.pop(session_id, None)
        if entry is not None:
            entry.controller.close()
        return entry

    def detach(self, session_id: str) -> RuntimeSession | None:
        """Remove a session without closing resources still used by an active Run."""

        return self._items.pop(session_id, None)

    def close_all(self) -> None:
        for session_id in list(self._items):
            self.close(session_id)

    def values(self) -> tuple[RuntimeSession, ...]:
        return tuple(self._items.values())

    def derive(
        self,
        *,
        source_session_id: str,
        controller: SessionController,
    ) -> RuntimeSession | None:
        source = self.require(source_session_id)
        status = source.status
        if status is not None:
            status = replace(
                status,
                session_id=controller.session_id,
            )
        tool_port = source.tool_port
        clone_for_session = getattr(tool_port, "for_session", None)
        if callable(clone_for_session):
            tool_port = clone_for_session(controller.session_id)
        session = RuntimeSession(
            controller=controller,
            model_port=source.model_port,
            tool_port=tool_port,
            status=status,
            commands=dict(source.commands or {}),
            mcp_manager=source.mcp_manager,
        )
        self.add(session)
        return session


@dataclass
class ActiveRun:
    """可取消、可查询的活动 Run 句柄。"""
    run_id: str
    resources: RunResourceScope


class ActiveRunRegistry:
    """Index active runs while RunResourceScope owns their live resources."""

    def __init__(self) -> None:
        self._items: dict[str, ActiveRun] = {}

    def start(self, session_id: str, run_id: str, resources: RunResourceScope) -> None:
        if session_id in self._items:
            raise RuntimeError(f"Session already has an active run: {session_id}")
        self._items[session_id] = ActiveRun(run_id=run_id, resources=resources)

    def finish(self, session_id: str, *, run_id: str | None = None) -> str | None:
        active = self._items.pop(session_id, None)
        if active is not None and run_id is not None and active.run_id != run_id:
            self._items[session_id] = active
            return None
        return active.run_id if active is not None else None

    def cancel(self, session_id: str, reason: str = "cancelled") -> str | None:
        active = self._items.get(session_id)
        if active is None or not active.resources.accepts_cancellation:
            return None
        active.resources.cancel(reason)
        return active.run_id

    def is_running(self, session_id: str) -> bool:
        return session_id in self._items

    def has_active_tasks(self, session_id: str) -> bool:
        active = self._items.get(session_id)
        return active is not None and active.resources.has_active_tasks

    def resources(self, session_id: str) -> RunResourceScope | None:
        active = self._items.get(session_id)
        return active.resources if active is not None else None

    def clear(self) -> None:
        self._items.clear()

    def cancel_all(self, reason: str = "runtime_closed") -> tuple[RunResourceScope, ...]:
        scopes: list[RunResourceScope] = []
        for session_id in tuple(self._items):
            active = self._items[session_id]
            active.resources.cancel(reason)
            scopes.append(active.resources)
        return tuple(scopes)


__all__ = [
    "ActiveRunRegistry",
    "RuntimeSession",
    "RuntimeSessionRegistry",
    "RuntimeStatusInfo",
    "SessionConversationState",
]
