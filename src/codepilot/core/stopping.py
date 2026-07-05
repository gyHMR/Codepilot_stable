from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codepilot.protocols import (
    AgentEvent,
    AgentRunCounters,
    AssistantMessage,
    AgentRunStatus,
    AgentRunStopReason,
    ErrorInfo,
    TaskSummary,
    ToolCall,
    ToolResultMessage,
)
from codepilot.tools.ports import ToolObservation

from .contracts import AgentLoopOutcome
from .task_control import CompletionCheck, ExecutionDecision
from .tool_turn import verification, workspace_effects


ToolCallSignature = tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


@dataclass(frozen=True)
class ToolExecutionGate:
    should_execute: bool
    reason: str = "allowed"
    code: str | None = None
    message: str | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class PostToolDecision:
    should_stop: bool
    reason: str
    status: AgentRunStatus | None = None
    stop_reason: AgentRunStopReason | None = None
    error: ErrorInfo | None = None
    force_completion_check: bool = False


@dataclass(frozen=True)
class CompletionDecision:
    should_stop: bool
    should_continue: bool
    reason: str
    status: AgentRunStatus | None = None
    stop_reason: AgentRunStopReason | None = None


def completed_outcome(
    run_id: str,
    messages: list[AssistantMessage | ToolResultMessage],
    final_message: AssistantMessage,
    events: list[AgentEvent],
    *,
    model_attempts: int,
    tool_iterations: int = 0,
    tool_calls: int = 0,
    usage: Any = None,
    observations: list[ToolObservation] | None = None,
    task: TaskSummary | None = None,
) -> AgentLoopOutcome:
    return AgentLoopOutcome(
        run_id=run_id,
        status="completed",
        stop_reason="final_answer",
        new_messages=list(messages),
        final_message=final_message,
        counters=AgentRunCounters(
            model_attempts=model_attempts,
            tool_iterations=tool_iterations,
            tool_calls=tool_calls,
        ),
        usage=usage,
        verification=verification(observations or []),
        workspace_effects=workspace_effects(observations or []),
        events=events,
        task=task,
    )


def last_assistant(
    messages: list[AssistantMessage | ToolResultMessage],
) -> AssistantMessage | None:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            return message
    return None


def tool_call_signature(tool_calls: list[ToolCall]) -> ToolCallSignature:
    return tuple(
        (
            call.name,
            tuple(sorted((key, repr(value)) for key, value in call.arguments.items())),
        )
        for call in tool_calls
    )


def tool_execution_gate(
    *,
    tool_iterations: int,
    tool_calls: list[ToolCall],
    current_signature: ToolCallSignature | None,
    repeated_count: int,
    max_tool_iterations: int,
    repeated_tool_call_limit: int,
) -> ToolExecutionGate:
    if max_tool_iterations >= 0 and tool_iterations >= max_tool_iterations:
        return ToolExecutionGate(
            should_execute=False,
            reason="max_iterations",
            code="run.max_iterations",
            message=f"Stopped after reaching max_tool_iterations={max_tool_iterations}",
            stop_reason="max_iterations",
        )
    if (
        current_signature is not None
        and repeated_tool_call_limit >= 0
        and repeated_count > repeated_tool_call_limit
    ):
        return ToolExecutionGate(
            should_execute=False,
            reason="repeated_tool_call",
            code="run.repeated_tool_call",
            message="Stopped after repeated identical tool calls",
            stop_reason="repeated_tool_call",
        )
    return ToolExecutionGate(should_execute=True)


def post_tool_decision(
    tool_results: list[ToolResultMessage],
    task_decision: ExecutionDecision | None,
) -> PostToolDecision:
    if any(result.status == "approval_required" for result in tool_results):
        return PostToolDecision(
            should_stop=True,
            reason="approval_required",
            status="waiting_approval",
            stop_reason="approval_required",
        )
    if any(result.status == "cancelled" for result in tool_results):
        return PostToolDecision(
            should_stop=True,
            reason="cancelled",
            status="aborted",
            stop_reason="aborted",
        )
    if task_decision is None:
        return PostToolDecision(should_stop=False, reason="continue")
    if task_decision.action == "propose_revert":
        return PostToolDecision(
            should_stop=True,
            reason=task_decision.reason,
            status="waiting_user",
            stop_reason="task_blocked",
        )
    if task_decision.action == "finish":
        return PostToolDecision(
            should_stop=False,
            reason="finish",
            force_completion_check=True,
        )
    if task_decision.action != "stop":
        return PostToolDecision(should_stop=False, reason=task_decision.reason)
    if task_decision.reason == "replan_limit_exceeded":
        return PostToolDecision(
            should_stop=True,
            reason=task_decision.reason,
            status="failed",
            stop_reason="replan_limit",
            error=ErrorInfo(
                code="run.replan_limit",
                message=task_decision.reason,
                retryable=False,
                source="runtime",
            ),
        )
    return PostToolDecision(
        should_stop=True,
        reason=task_decision.reason,
        status="waiting_user",
        stop_reason="task_blocked",
    )


def completion_decision(check: CompletionCheck) -> CompletionDecision:
    if check.satisfied:
        return CompletionDecision(
            should_stop=False,
            should_continue=False,
            reason=check.reason,
        )
    if check.can_continue:
        return CompletionDecision(
            should_stop=False,
            should_continue=True,
            reason=check.reason,
        )
    return CompletionDecision(
        should_stop=True,
        should_continue=False,
        reason=check.reason,
        status="waiting_user",
        stop_reason="task_blocked" if check.reason == "blocked_steps" else "task_incomplete",
    )
