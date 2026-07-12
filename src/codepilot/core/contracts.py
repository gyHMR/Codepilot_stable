from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Literal, Protocol

from codepilot.llm.ports import ModelDescriptor, ModelPort
from codepilot.protocols import (
    AgentEvent,
    AgentRunCounters,
    AssistantMessage,
    ContextReport,
    Message,
    PlanSummary,
    RunSignalsSummary,
    RunVerification,
    TextContent,
    Tool,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from codepilot.tools.contracts import ToolPort
from codepilot.tools.security import ApprovalChallenge
from .plan import RunMode, ensure_run_mode, plan_state_to_dict


ToolExecutionMode = Literal["sequential", "parallel"]
AgentMessage = Message


@dataclass
class AgentContext:
    """Model-facing run context prepared by sessions and consumed by core."""

    system_prompt: str
    messages: list[AgentMessage]
    tools: list[Tool] = field(default_factory=list)
    mode: RunMode = "build"
    plan_state: dict[str, object] | None = None
    run_signals: dict[str, object] | None = None
    runtime_state: dict[str, object] | None = None

    def __post_init__(self) -> None:
        self.system_prompt = _clean_core_text(self.system_prompt)
        self.messages = _copy_messages(self.messages, field_name="messages")
        self.tools = _copy_tools(self.tools, field_name="tools")
        self.mode = ensure_run_mode(self.mode)
        self.plan_state = _copy_optional_dict(
            self.plan_state,
            field_name="plan_state",
        )
        self.run_signals = _copy_optional_dict(
            self.run_signals,
            field_name="run_signals",
        )
        self.runtime_state = _copy_optional_dict(
            self.runtime_state,
            field_name="runtime_state",
        )


@dataclass(frozen=True)
class ContextPreparationRequest:
    session_id: str | None
    model_context_window: int
    model_max_output_tokens: int
    signal: Any | None = None


@dataclass
class PreparedAgentContext:
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[Tool]
    report: ContextReport


PrepareContextFn = Callable[
    [AgentContext, ContextPreparationRequest],
    PreparedAgentContext | Awaitable[PreparedAgentContext],
]


AgentLoopStatus = Literal[
    "completed",
    "waiting_approval",
    "waiting_user",
    "failed",
    "cancelled",
    "aborted",
]


EventSink = Callable[[AgentEvent], None]


@dataclass(frozen=True)
class PreparedContext(Mapping[str, object]):
    values: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            MappingProxyType(deepcopy(dict(self.values))),
        )

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def system_prompt(self) -> str:
        return str(self.values.get("system_prompt", ""))

    @property
    def session_id(self) -> str | None:
        return _optional_text(self.values.get("session_id"))


@dataclass(frozen=True)
class RunCorrelation:
    session_id: str | None = None
    turn_id: str | None = None


@dataclass(frozen=True)
class AgentLoopLimits:
    max_model_turns: int = 256
    max_tool_iterations: int = 240
    max_tool_calls_per_turn: int | None = 16
    max_tool_calls: int | None = None
    repeated_tool_call_limit: int = 6


@dataclass(frozen=True)
class RetryPolicy:
    enabled: bool = False
    max_retries: int = 0
    base_delay_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(
            self,
            "max_retries",
            _non_negative_int(self.max_retries, default=0),
        )
        object.__setattr__(
            self,
            "base_delay_ms",
            _non_negative_int(self.base_delay_ms, default=0),
        )


@dataclass(frozen=True)
class AgentLoopInput:
    run_id: str
    correlation: RunCorrelation
    messages: list[Message] = field(default_factory=list)
    user_prompt: str | None = None
    context: PreparedContext = field(default_factory=PreparedContext)
    model: ModelDescriptor = field(default_factory=lambda: ModelDescriptor(provider="unknown", model_id="unknown"))
    tools: list[Any] = field(default_factory=list)
    mode: RunMode = "build"
    plan_state: dict[str, object] | None = None
    limits: AgentLoopLimits = field(default_factory=AgentLoopLimits)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    event_start_seq: int = 0
    turn_start_seq: int = 0
    run_state: dict[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _prepared_context(self.context))
        object.__setattr__(self, "mode", ensure_run_mode(self.mode))
        object.__setattr__(self, "plan_state", plan_state_to_dict(self.plan_state))
        object.__setattr__(
            self,
            "run_state",
            _copy_optional_dict(self.run_state, field_name="run_state"),
        )
        object.__setattr__(
            self,
            "event_start_seq",
            _non_negative_int(self.event_start_seq, default=0),
        )
        object.__setattr__(
            self,
            "turn_start_seq",
            _non_negative_int(self.turn_start_seq, default=0),
        )


@dataclass(frozen=True)
class AgentResumeInput:
    run_id: str
    correlation: RunCorrelation
    messages: list[Message] = field(default_factory=list)
    context: PreparedContext = field(default_factory=PreparedContext)
    model: ModelDescriptor = field(default_factory=lambda: ModelDescriptor(provider="unknown", model_id="unknown"))
    tools: list[Any] = field(default_factory=list)
    approval_id: str | None = None
    decision: str | None = None
    reason: str = ""
    mode: RunMode = "build"
    plan_state: dict[str, object] | None = None
    limits: AgentLoopLimits = field(default_factory=AgentLoopLimits)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    event_start_seq: int = 0
    turn_start_seq: int = 0
    run_state: dict[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _prepared_context(self.context))
        object.__setattr__(self, "mode", ensure_run_mode(self.mode))
        object.__setattr__(self, "plan_state", plan_state_to_dict(self.plan_state))
        object.__setattr__(
            self,
            "run_state",
            _copy_optional_dict(self.run_state, field_name="run_state"),
        )
        object.__setattr__(
            self,
            "event_start_seq",
            _non_negative_int(self.event_start_seq, default=0),
        )
        object.__setattr__(
            self,
            "turn_start_seq",
            _non_negative_int(self.turn_start_seq, default=0),
        )


@dataclass(frozen=True)
class AgentLoopPorts:
    model: ModelPort | None
    tools: ToolPort | None
    context: ContextPort | None = None
    events: EventSink | None = None


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
    interruptions: list[ApprovalChallenge] = field(default_factory=list)
    counters: AgentRunCounters = field(default_factory=AgentRunCounters)
    usage: Usage | None = None
    verification: list[RunVerification] = field(default_factory=list)
    workspace_effects: WorkspaceEffects = field(default_factory=WorkspaceEffects)
    events: list[AgentEvent] = field(default_factory=list)
    plan: PlanSummary | None = None
    signals: RunSignalsSummary = field(default_factory=RunSignalsSummary)
    run_state: dict[str, object] = field(default_factory=dict)
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


def _prepared_context(value: object) -> PreparedContext:
    if isinstance(value, PreparedContext):
        return value
    if value is None:
        return PreparedContext()
    if not isinstance(value, Mapping):
        raise TypeError("prepared context must be a mapping")
    return PreparedContext(value)


def _non_negative_int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(0, value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_core_text(value: object) -> str:
    return str(value) if value is not None else ""


def _optional_core_text(value: object) -> str | None:
    text = _clean_core_text(value).strip()
    return text or None


def _copy_messages(value: object, *, field_name: str) -> list[AgentMessage]:
    if not isinstance(value, list):
        raise TypeError(f"AgentContext {field_name} must be a list")
    messages: list[AgentMessage] = []
    for message in value:
        if not isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
            raise TypeError(f"AgentContext {field_name} entries must be AgentMessage")
        messages.append(message)
    return messages


def _copy_tools(value: object, *, field_name: str) -> list[Tool]:
    if not isinstance(value, list):
        raise TypeError(f"AgentContext {field_name} must be a list")
    tools: list[Tool] = []
    for tool in value:
        if not isinstance(tool, Tool):
            raise TypeError(f"AgentContext {field_name} entries must be Tool")
        tools.append(tool)
    return tools


def _copy_optional_dict(
    value: object,
    *,
    field_name: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"AgentContext {field_name} must be a dict or None")
    return deepcopy(value)


__all__ = [
    "AgentLoopInput",
    "AgentLoopLimits",
    "AgentLoopOutcome",
    "AgentLoopPorts",
    "AgentLoopStatus",
    "AgentResumeInput",
    "AgentContext",
    "AgentMessage",
    "ContextPreparationRequest",
    "ContextPort",
    "EventSink",
    "PreparedAgentContext",
    "PreparedContext",
    "PrepareContextFn",
    "RetryPolicy",
    "RunCorrelation",
    "ToolExecutionMode",
    "WorkspaceEffects",
]
