from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

from codepilot.core.plan import RunMode, ensure_run_mode
from codepilot.core.contracts import AgentMessage
from codepilot.protocols import (
    AgentEvent,
    AgentEventSink,
    AgentRunResult,
    Message,
    Model,
    ThinkingLevel,
)


@dataclass
class SessionConversationState:
    """Session-owned conversation state used by the V2 spine.

    The old core ``Agent`` used to own model, tools, transcript and listeners.
    V2 keeps that state in sessions so core can stay a stateless loop over
    prepared input and ports.
    """

    model: Model
    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: ThinkingLevel | str = "off"
    current_mode: RunMode = "build"
    stream_message: Message | None = None
    error: str | None = None
    active_tool_call_ids: set[str] = field(default_factory=set)
    last_run_result: AgentRunResult | None = None
    _steering_messages: list[AgentMessage] = field(default_factory=list)
    _listeners: list[AgentEventSink] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.system_prompt = str(self.system_prompt or "")
        self.messages = list(self.messages)
        self.current_mode = ensure_run_mode(self.current_mode)

    def set_messages(self, messages: list[AgentMessage]) -> None:
        self.messages = list(messages)

    def append_messages(self, messages: list[AgentMessage]) -> None:
        self.messages.extend(messages)

    def remember_result(self, result: AgentRunResult) -> None:
        self.last_run_result = result
        self.error = None if result.status == "completed" else self.error

    def set_current_mode(self, mode: RunMode | str) -> RunMode:
        self.current_mode = ensure_run_mode(mode)
        return self.current_mode

    def add_steering_message(self, message: AgentMessage) -> None:
        self._steering_messages.append(message)

    def drain_steering_messages(self) -> list[AgentMessage]:
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
        if event_type == "message_start":
            self.stream_message = event.get("message")
        elif event_type == "message_update":
            self.stream_message = event.get("message")
        elif event_type == "message_end":
            self.stream_message = None
        elif event_type == "tool_started":
            tool_call_id = event.get("toolCallId")
            if tool_call_id:
                self.active_tool_call_ids.add(str(tool_call_id))
        elif event_type in {"tool_completed", "tool_failed", "tool_interrupted"}:
            tool_call_id = event.get("toolCallId")
            if tool_call_id:
                self.active_tool_call_ids.discard(str(tool_call_id))
        elif event_type == "error":
            self.error = str(event.get("error", "unknown error"))


__all__ = ["SessionConversationState"]
