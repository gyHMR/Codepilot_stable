from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional

from codepilot.core.contracts import (
    AgentMessage,
    AgentLoopInput,
    AgentLoopOutcome,
    AgentResumeInput,
    AgentLoopStatus,
    ContextPort,
    PrepareContextFn,
    ToolExecutionMode,
)
from codepilot.core.plan import PlanningBudgetProfile, RunMode
from codepilot.llm.provider_types import ProviderSimpleStreamFn
from codepilot.protocols import AgentEvent, Message
from codepilot.protocols import Model
from codepilot.protocols.commands import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    LifecycleHook,
    RegisteredCommand,
)


ConvertToLlmFn = Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]
SystemPromptBuilder = Callable[[RunMode], str]
SessionContinuationKind = Literal[
    "plan_approved",
    "plan_rejected",
    "plan_feedback",
    "plan_clarification",
    "tool_approved",
    "tool_denied",
    "mode_changed",
    "automatic_continuation",
]
_CONTINUATION_KINDS = {
    "plan_approved",
    "plan_rejected",
    "plan_feedback",
    "plan_clarification",
    "tool_approved",
    "tool_denied",
    "mode_changed",
    "automatic_continuation",
}


@dataclass
class SessionOptions:
    """Runtime-supplied configuration for opening a session controller."""

    model: Model
    workspace_dir: str | Path
    system_prompt: str = ""
    system_prompt_builder: Optional[SystemPromptBuilder] = None
    session_id: Optional[str] = None
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: str = "off"
    tool_execution: ToolExecutionMode = "parallel"
    max_tool_calls_per_turn: int = 16
    memory_enabled: bool = True
    current_mode: RunMode = "build"
    planning_budget_profile: PlanningBudgetProfile = "balanced"
    convert_to_llm: Optional[ConvertToLlmFn] = None
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None
    retry_enabled: bool = True
    max_retries: int = 2
    retry_base_delay_ms: int = 1200
    extension_commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    after_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    before_tool_call: Optional[
        Callable[
            [BeforeToolCallContext, Any | None],
            BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
        ]
    ] = None
    after_tool_call: Optional[
        Callable[
            [AfterToolCallContext, Any | None],
            AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
        ]
    ] = None
    stream_fn: ProviderSimpleStreamFn | None = None
    prepare_context: PrepareContextFn | None = None


@dataclass(frozen=True)
class SessionRunIntent:
    text: str
    images: tuple[str, ...] = ()
    mode_hint: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "run text"))
        object.__setattr__(self, "mode_hint", _optional_text(self.mode_hint))
        object.__setattr__(self, "run_id", _optional_text(self.run_id))


@dataclass(frozen=True)
class SessionResumeIntent:
    approval_id: str
    decision: str
    reason: str = ""
    run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(self, "decision", _approval_decision(self.decision))
        object.__setattr__(self, "reason", _optional_text(self.reason) or "")
        object.__setattr__(self, "run_id", _optional_text(self.run_id))


@dataclass(frozen=True)
class SessionContinuationIntent:
    kind: SessionContinuationKind
    run_id: str | None = None
    text: str = ""
    approval_id: str = ""
    decision: str = ""
    reason: str = ""
    target_mode: str | None = None

    def __post_init__(self) -> None:
        kind = _require_text(self.kind, "continuation kind")
        if kind not in _CONTINUATION_KINDS:
            raise ValueError(f"Unknown continuation kind: {self.kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(self, "text", _optional_text(self.text) or "")
        object.__setattr__(self, "approval_id", _optional_text(self.approval_id) or "")
        object.__setattr__(self, "decision", _optional_text(self.decision) or "")
        object.__setattr__(self, "reason", _optional_text(self.reason) or "")
        object.__setattr__(self, "target_mode", _optional_text(self.target_mode))
        if kind == "plan_feedback" and not self.text:
            raise ValueError("plan feedback text is required")
        if kind in {"tool_approved", "tool_denied"} and not self.approval_id:
            raise ValueError("tool continuation approval_id is required")


@dataclass(frozen=True)
class SessionCommandIntent:
    text: str
    tool_catalog: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "command text"))
        object.__setattr__(self, "tool_catalog", tuple(self.tool_catalog))


@dataclass(frozen=True)
class CancelRunIntent:
    run_id: str | None = None
    reason: str = "user"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(self, "reason", _optional_text(self.reason) or "user")


SessionIntent = (
    SessionRunIntent
    | SessionResumeIntent
    | SessionContinuationIntent
    | SessionCommandIntent
    | CancelRunIntent
)


@dataclass(frozen=True)
class SessionView:
    session_id: str
    message_count: int = 0
    last_run_id: str | None = None
    current_mode: str = "build"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackBaselineRef:
    session_id: str
    run_id: str
    kind: Literal["rollback_baseline_ref"] = field(
        default="rollback_baseline_ref",
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))


@dataclass(frozen=True)
class PreparedAgentRun:
    run_id: str
    session_id: str
    loop_input: AgentLoopInput
    resume_input: AgentResumeInput | None = None
    context_port: ContextPort | None = None
    input_messages: list[Message] = field(default_factory=list)
    rollback_baseline: RollbackBaselineRef | None = None
    context_refs: dict[str, Any] = field(default_factory=dict)
    memory_refs: dict[str, Any] = field(default_factory=dict)
    plan_refs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionRunRecord:
    run_id: str
    session_id: str
    status: AgentLoopStatus
    stop_reason: str
    new_messages: list[Message] = field(default_factory=list)
    final_text: str = ""
    events: list[AgentEvent] = field(default_factory=list)
    outcome: AgentLoopOutcome | None = None
    snapshots: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionCommandRecord:
    session_id: str
    command: str
    handled: bool
    output_lines: tuple[str, ...] = ()
    switched_session_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text must be a string or None")
    return value.strip() or None


def _approval_decision(value: object) -> str:
    text = _require_text(value, "approval decision").lower()
    if text not in {"approve", "deny"}:
        raise ValueError(f"Unknown approval decision: {value}")
    return text


__all__ = [
    "CancelRunIntent",
    "ConvertToLlmFn",
    "PreparedAgentRun",
    "RollbackBaselineRef",
    "SessionCommandIntent",
    "SessionCommandRecord",
    "SessionContinuationIntent",
    "SessionContinuationKind",
    "SessionIntent",
    "SessionResumeIntent",
    "SessionRunIntent",
    "SessionRunRecord",
    "SessionOptions",
    "SessionView",
    "SystemPromptBuilder",
]
