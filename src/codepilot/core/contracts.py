from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol

from codepilot.llm.ports import ModelDescriptor, ModelPort
from codepilot.protocols import (
    AgentEvent,
    AgentRunCounters,
    AssistantMessage,
    Message,
    RunVerification,
    TaskSummary,
    TextContent,
    Usage,
)
from codepilot.tools.ports import ToolInterruption, ToolPort


AgentLoopStatus = Literal[
    "completed",
    "waiting_approval",
    "waiting_user",
    "failed",
    "cancelled",
    "aborted",
]


@dataclass(frozen=True)
class RunCorrelation:
    session_id: str | None = None
    turn_id: str | None = None


@dataclass(frozen=True)
class AgentLoopLimits:
    max_model_turns: int = 12
    max_tool_iterations: int = 12
    max_tool_calls_per_turn: int | None = 8
    max_tool_calls: int | None = None
    repeated_tool_call_limit: int = 3


@dataclass(frozen=True)
class AgentLoopInput:
    run_id: str
    correlation: RunCorrelation
    messages: list[Message] = field(default_factory=list)
    user_prompt: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    model: ModelDescriptor = field(default_factory=lambda: ModelDescriptor(provider="unknown", model_id="unknown"))
    tools: list[Any] = field(default_factory=list)
    task_strategy: dict[str, Any] = field(default_factory=dict)
    limits: AgentLoopLimits = field(default_factory=AgentLoopLimits)
    retry_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResumeInput:
    run_id: str
    correlation: RunCorrelation
    messages: list[Message] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    model: ModelDescriptor = field(default_factory=lambda: ModelDescriptor(provider="unknown", model_id="unknown"))
    tools: list[Any] = field(default_factory=list)
    approval_id: str | None = None
    decision: str | None = None
    reason: str = ""
    task_strategy: dict[str, Any] = field(default_factory=dict)
    limits: AgentLoopLimits = field(default_factory=AgentLoopLimits)
    retry_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentLoopPorts:
    model: ModelPort | None
    tools: ToolPort | None
    context: ContextPort | None = None
    events: Callable[[dict[str, Any]], None] | None = None


@dataclass(frozen=True)
class WorkspaceEffects:
    affected_paths: tuple[str, ...] = ()
    changed: bool = False


@dataclass(frozen=True)
class AgentLoopOutcome:
    run_id: str
    status: AgentLoopStatus
    stop_reason: str
    new_messages: list[Message] = field(default_factory=list)
    final_message: AssistantMessage | None = None
    interruptions: list[ToolInterruption] = field(default_factory=list)
    counters: AgentRunCounters = field(default_factory=AgentRunCounters)
    usage: Usage | None = None
    verification: list[RunVerification] = field(default_factory=list)
    workspace_effects: WorkspaceEffects = field(default_factory=WorkspaceEffects)
    events: list[AgentEvent] = field(default_factory=list)
    task: TaskSummary | None = None
    error: Any = None

    @property
    def final_text(self) -> str:
        if self.final_message is None:
            return ""
        if isinstance(self.final_message.content, str):
            return self.final_message.content
        chunks: list[str] = []
        for block in self.final_message.content:
            if isinstance(block, TextContent):
                chunks.append(block.text)
        return "".join(chunks)


class ContextPort(Protocol):
    def prepare(self, request: Any) -> Any | Awaitable[Any]:
        ...


__all__ = [
    "AgentLoopInput",
    "AgentLoopLimits",
    "AgentLoopOutcome",
    "AgentLoopPorts",
    "AgentLoopStatus",
    "AgentResumeInput",
    "ContextPort",
    "RunCorrelation",
    "WorkspaceEffects",
]
