from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .errors import ErrorInfo
from .messages import AssistantMessage, Message


AgentRunStatus = Literal[
    "running",
    "completed",
    "failed",
    "aborted",
    "waiting_approval",
]
AgentRunStopReason = Literal[
    "final_answer",
    "max_iterations",
    "model_error",
    "aborted",
    "approval_required",
    "repeated_tool_call",
    "internal_error",
]
RunVerificationStatus = Literal["passed", "failed", "cancelled", "unknown"]


@dataclass
class AgentRunCounters:
    model_attempts: int = 0
    tool_iterations: int = 0
    tool_calls: int = 0


@dataclass
class RunVerification:
    tool_call_id: str
    tool_name: str
    status: RunVerificationStatus
    command: str | None = None
    exit_code: int | None = None
    summary: str = ""


@dataclass
class AgentRunResult:
    run_id: str
    session_id: str | None
    status: AgentRunStatus
    stop_reason: AgentRunStopReason
    counters: AgentRunCounters = field(default_factory=AgentRunCounters)
    messages: list[Message] = field(default_factory=list)
    final_message: AssistantMessage | None = None
    error: ErrorInfo | None = None
    affected_paths: list[str] = field(default_factory=list)
    workspace_changed: bool = False
    verification: list[RunVerification] = field(default_factory=list)


__all__ = [
    "AgentRunCounters",
    "AgentRunResult",
    "AgentRunStatus",
    "AgentRunStopReason",
    "RunVerification",
    "RunVerificationStatus",
]
