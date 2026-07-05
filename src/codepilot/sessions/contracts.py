from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codepilot.core.contracts import AgentLoopInput, AgentLoopOutcome, AgentResumeInput
from codepilot.protocols import AgentEvent, Message


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


SessionIntent = SessionRunIntent | SessionResumeIntent | SessionCommandIntent | CancelRunIntent


@dataclass(frozen=True)
class SessionView:
    session_id: str
    message_count: int = 0
    last_run_id: str | None = None
    task_mode: str = "edit"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedAgentRun:
    run_id: str
    session_id: str
    loop_input: AgentLoopInput
    resume_input: AgentResumeInput | None = None
    context_port: Any | None = None
    input_messages: list[Message] = field(default_factory=list)
    rollback_baseline: Any = None
    context_refs: dict[str, Any] = field(default_factory=dict)
    memory_refs: dict[str, Any] = field(default_factory=dict)
    recovery_refs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionRunRecord:
    run_id: str
    session_id: str
    status: str
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
    "PreparedAgentRun",
    "SessionCommandIntent",
    "SessionCommandRecord",
    "SessionIntent",
    "SessionResumeIntent",
    "SessionRunIntent",
    "SessionRunRecord",
    "SessionView",
]
