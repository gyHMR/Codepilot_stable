from __future__ import annotations

"""
Agent 运行结果与状态类型定义。

定义了一次 Agent 运行（run）的完整结果结构：
- 运行状态和停止原因
- 执行计数器（模型调用次数、工具迭代次数等）
- 运行验证结果
- 最终的运行结果汇总
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import ErrorInfo
from .messages import AssistantMessage, Message


# ── 枚举类型 ────────────────────────────────────────────────────

# Agent 运行状态
AgentRunStatus = Literal[
    "running",           # 正在运行
    "completed",         # 正常完成
    "failed",            # 运行失败
    "aborted",           # 被用户中止
    "waiting_approval",  # 等待用户审批（如危险工具调用）
    "waiting_user",      # 等待用户确认或补充指示（如任务阻塞/回退确认）
]

# Agent 运行停止原因
AgentRunStopReason = Literal[
    "final_answer",          # 模型给出了最终回答
    "max_iterations",        # 达到最大迭代次数限制
    "model_error",           # 模型调用出错
    "aborted",               # 被用户中止
    "approval_required",     # 需要用户审批
    "repeated_tool_call",    # 检测到重复的工具调用（可能陷入循环）
    "replan_limit",          # 连续失败后达到重新规划上限
    "task_blocked",          # 任务控制器判断当前任务阻塞，需要用户确认/指示
    "task_incomplete",       # 任务控制器判断完成条件未满足
    "internal_error",        # 内部错误
]

# 运行验证状态
RunVerificationStatus = Literal["passed", "failed", "cancelled", "unknown"]


@dataclass
class AgentRunCounters:
    """Agent 运行计数器。

    记录一次运行中的各类调用次数，用于监控和限制。

    Attributes:
        model_attempts: 模型调用次数（含重试）。
        tool_iterations: 工具执行迭代轮次。
        tool_calls: 工具调用总次数。
    """

    model_attempts: int = 0
    tool_iterations: int = 0
    tool_calls: int = 0


@dataclass
class RunVerification:
    """单次工具调用的验证结果。

    Attributes:
        tool_call_id: 工具调用 ID。
        tool_name: 工具名称。
        status: 验证状态（passed/failed/cancelled/unknown）。
        command: 执行的命令（可选，适用于命令行工具）。
        exit_code: 进程退出码（可选）。
        summary: 验证摘要说明。
    """

    tool_call_id: str
    tool_name: str
    status: RunVerificationStatus
    command: str | None = None
    exit_code: int | None = None
    summary: str = ""


@dataclass
class TaskSummary:
    """Lightweight summary of the active task plan for one run."""

    task_id: str
    goal: str
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    blocked_steps: list[str] = field(default_factory=list)
    next_action: str | None = None
    completion_satisfied: bool = False
    completion_reason: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)
    change_sets: list[dict[str, Any]] = field(default_factory=list)
    replans: list[dict[str, Any]] = field(default_factory=list)
    control_signal: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunResult:
    """Agent 运行的完整结果。

    汇总一次 Agent 运行的所有信息：状态、消息、计数器、错误、影响范围等。

    Attributes:
        run_id: 本次运行的唯一 ID。
        session_id: 所属会话 ID（可选）。
        status: 运行最终状态。
        stop_reason: 停止原因。
        counters: 执行计数器。
        messages: 本次运行产生的所有消息。
        final_message: 最终的助手回复消息（可选）。
        error: 错误信息（运行失败时设置）。
        affected_paths: 受影响的文件路径列表。
        workspace_changed: 是否修改了工作区。
        verification: 工具调用验证结果列表。
    """

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
    task: TaskSummary | None = None


__all__ = [
    "AgentRunCounters",
    "AgentRunResult",
    "AgentRunStatus",
    "AgentRunStopReason",
    "RunVerification",
    "RunVerificationStatus",
    "TaskSummary",
]
