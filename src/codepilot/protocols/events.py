from __future__ import annotations

from typing import Any, Awaitable, Callable, Literal, TypedDict

from .errors import ErrorInfo
from .llm import LLMStreamEvent
from .messages import AssistantMessage, Message, ToolResultMessage
from .runs import AgentRunCounters, AgentRunResult, AgentRunStatus, AgentRunStopReason
from .tools import ToolResult, ToolResultStatus


RuntimeEventType = Literal[
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_start",
    "message_update",
    "message_end",
    "model_retry_start",
    "tool_execution_start",
    "tool_execution_update",
    "tool_approval_required",
    "tool_approval_resolved",
    "tool_execution_end",
    "file_diff",
    "error",
]


class AgentEventBase(TypedDict):
    """Stable event envelope emitted by the core loop."""

    type: RuntimeEventType
    runId: str
    turnId: int
    eventId: str
    timestamp: int
    sessionId: str | None


EventEnvelope = AgentEventBase


class AgentStartEvent(AgentEventBase):
    type: Literal["agent_start"]


class AgentEndEvent(AgentEventBase):
    type: Literal["agent_end"]
    messages: list[Message]
    status: AgentRunStatus
    stopReason: AgentRunStopReason
    counters: AgentRunCounters
    result: AgentRunResult


class TurnStartEvent(AgentEventBase):
    type: Literal["turn_start"]


class TurnEndEvent(AgentEventBase):
    type: Literal["turn_end"]
    message: AssistantMessage
    toolResults: list[ToolResultMessage]


class MessageStartEvent(AgentEventBase):
    type: Literal["message_start"]
    message: Message


class MessageUpdateEvent(AgentEventBase):
    type: Literal["message_update"]
    message: Message
    assistantMessageEvent: LLMStreamEvent


class MessageEndEvent(AgentEventBase):
    type: Literal["message_end"]
    message: Message


class ModelRetryStartEvent(AgentEventBase):
    type: Literal["model_retry_start"]
    attempt: int
    maxAttempts: int
    delayMs: int
    error: ErrorInfo


class ToolExecutionStartEvent(AgentEventBase):
    type: Literal["tool_execution_start"]
    toolCallId: str
    toolName: str
    args: dict[str, Any]


class ToolExecutionUpdateEvent(AgentEventBase):
    type: Literal["tool_execution_update"]
    toolCallId: str
    toolName: str
    args: dict[str, Any]
    partialResult: ToolResult


class ToolExecutionEndEvent(AgentEventBase):
    type: Literal["tool_execution_end"]
    toolCallId: str
    toolName: str
    result: ToolResult
    status: ToolResultStatus
    isError: bool
    approved: bool
    approvalId: str | None
    errorReason: str | None


class ErrorEvent(AgentEventBase, total=False):
    type: Literal["error"]
    error: str
    message: str
    source: str
    code: str
    retryable: bool
    provider: str
    model: str
    statusCode: int | None
    errorInfo: ErrorInfo


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ModelRetryStartEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ErrorEvent
)
RuntimeEvent = AgentEvent
AgentEventSink = Callable[[AgentEvent], None | Awaitable[None]]


__all__ = [
    "AgentEndEvent",
    "AgentEvent",
    "AgentEventBase",
    "AgentEventSink",
    "AgentStartEvent",
    "ErrorEvent",
    "EventEnvelope",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "ModelRetryStartEvent",
    "RuntimeEvent",
    "RuntimeEventType",
    "ToolExecutionEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
