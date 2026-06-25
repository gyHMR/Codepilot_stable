"""Pure run-level decisions used by the Agent loop.

The Agent loop coordinates I/O-heavy work: model streaming, tool execution,
task updates, and event emission. This module keeps the small policy decisions
that translate lower-level facts into run-level outcomes in one testable place.
"""

from __future__ import annotations

from dataclasses import dataclass

from codepilot.protocols import (
    AgentRunStatus,
    AgentRunStopReason,
    ErrorInfo,
    ToolCall,
    ToolResultMessage,
)

from .run_state import RunState
from .task_state import CompletionCheck, ExecutionDecision
from .types import AgentLoopConfig


@dataclass(frozen=True)
class ModelRetryDecision:
    """Decision after a model error.

    The loop only needs to know whether to retry, which retry number comes next,
    and how long to wait. The detailed policy stays here instead of being spread
    across the control flow.
    """

    should_retry: bool
    reason: str
    next_retry_count: int
    delay_ms: int


@dataclass(frozen=True)
class ToolExecutionGate:
    """Pre-execution gate for a batch of tool calls."""

    should_execute: bool
    reason: str
    error_code: str | None = None
    message: str | None = None
    stop_reason: AgentRunStopReason | None = None
    assistant_stop_reason: str | None = None


@dataclass(frozen=True)
class PostToolRunDecision:
    """Run-level decision after tool execution.

    ``TaskController`` describes what should happen at the task layer. The loop
    still needs a run status, stop reason, and optional error details. This
    object is the explicit boundary between those two concepts.
    """

    should_stop: bool
    reason: str
    status: AgentRunStatus | None = None
    stop_reason: AgentRunStopReason | None = None
    error_code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class CompletionRunDecision:
    """Run-level decision after checking task completion."""

    action: str
    should_stop: bool
    reason: str
    status: AgentRunStatus | None = None
    stop_reason: AgentRunStopReason | None = None


def decide_model_retry(
    error: ErrorInfo,
    *,
    retries_so_far: int,
    config: AgentLoopConfig,
) -> ModelRetryDecision:
    """Decide whether a model error should be retried."""

    if not config.retry_enabled:
        return ModelRetryDecision(
            should_retry=False,
            reason="retry_disabled",
            next_retry_count=retries_so_far,
            delay_ms=0,
        )
    if not error.retryable:
        return ModelRetryDecision(
            should_retry=False,
            reason="error_not_retryable",
            next_retry_count=retries_so_far,
            delay_ms=0,
        )
    if retries_so_far >= config.max_model_retries:
        return ModelRetryDecision(
            should_retry=False,
            reason="retry_limit_exhausted",
            next_retry_count=retries_so_far,
            delay_ms=0,
        )
    next_retry_count = retries_so_far + 1
    return ModelRetryDecision(
        should_retry=True,
        reason="retry",
        next_retry_count=next_retry_count,
        delay_ms=int(config.retry_base_delay_ms * (2 ** (next_retry_count - 1))),
    )


def decide_tool_execution_gate(
    tool_calls: list[ToolCall],
    state: RunState,
    config: AgentLoopConfig,
) -> ToolExecutionGate:
    """Decide whether this batch of tool calls may execute."""

    if state.has_repeated_call(
        tool_calls,
        limit=config.repeated_tool_call_limit,
    ):
        return ToolExecutionGate(
            should_execute=False,
            reason="repeated_tool_call",
            error_code="run.repeated_tool_call",
            message="Stopped after repeated identical tool calls",
            stop_reason="repeated_tool_call",
        )
    if state.counters.tool_iterations >= config.max_tool_iterations:
        return ToolExecutionGate(
            should_execute=False,
            reason="max_iterations",
            error_code="run.max_iterations",
            message=(
                "Stopped after reaching "
                f"max_tool_iterations={config.max_tool_iterations}"
            ),
            stop_reason="max_iterations",
            assistant_stop_reason="max_iterations",
        )
    return ToolExecutionGate(should_execute=True, reason="allowed")


def decide_post_tool_run(
    tool_results: list[ToolResultMessage],
    *,
    task_decision: ExecutionDecision | None,
) -> PostToolRunDecision:
    """Decide whether tool execution should pause or stop the run."""

    if any(result.status == "approval_required" for result in tool_results):
        return PostToolRunDecision(
            should_stop=True,
            reason="approval_required",
            status="waiting_approval",
            stop_reason="approval_required",
        )
    if any(result.status == "cancelled" for result in tool_results):
        return PostToolRunDecision(
            should_stop=True,
            reason="cancelled",
            status="aborted",
            stop_reason="aborted",
        )
    if task_decision is None:
        return PostToolRunDecision(should_stop=False, reason="continue")
    if task_decision.action == "propose_revert":
        return PostToolRunDecision(
            should_stop=True,
            reason=task_decision.reason,
            status="waiting_user",
            stop_reason="task_blocked",
        )
    if task_decision.action != "stop":
        return PostToolRunDecision(should_stop=False, reason="continue")
    if task_decision.reason == "replan_limit_exceeded":
        return PostToolRunDecision(
            should_stop=True,
            reason=task_decision.reason,
            status="failed",
            stop_reason="replan_limit",
            error_code="run.replan_limit",
            message=task_decision.reason,
        )
    return PostToolRunDecision(
        should_stop=True,
        reason=task_decision.reason,
        status="waiting_user",
        stop_reason="task_blocked",
    )


def decide_completion_run(check: CompletionCheck) -> CompletionRunDecision:
    """Decide whether the run can finish after task completion checking."""

    if check.satisfied:
        return CompletionRunDecision(
            action="satisfied",
            should_stop=False,
            reason=check.reason,
        )
    if check.can_continue:
        return CompletionRunDecision(
            action="continue_with_steering",
            should_stop=False,
            reason=check.reason,
        )
    return CompletionRunDecision(
        action="stop",
        should_stop=True,
        reason=check.reason,
        status="waiting_user",
        stop_reason=(
            "task_blocked" if check.reason == "blocked_steps" else "task_incomplete"
        ),
    )
