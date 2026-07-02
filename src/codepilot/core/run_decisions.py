"""
Agent 循环的纯决策函数模块。

本模块将 Agent 循环中的小型策略决策集中在一个可测试的位置。
Agent 循环负责协调 I/O 密集型工作（模型流式调用、工具执行、任务更新、事件发射），
而本模块负责将底层事实（错误信息、工具结果等）转换为运行级别的决策结果。

设计原则：
    - 所有决策函数都是纯函数：输入相同则输出相同，没有副作用
    - 决策结果使用 frozen dataclass 表示，不可修改
    - 决策逻辑与 I/O 操作分离，便于单元测试

决策类型：
    - ModelRetryDecision: LLM 调用失败后是否重试
    - ToolExecutionGate: 工具调用前的执行门禁（是否允许执行）
    - PostToolRunDecision: 工具执行后是否需要暂停或停止运行
    - CompletionRunDecision: 任务完成度检查后的运行决策
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
from .task_control import CompletionCheck, ExecutionDecision
from .types import AgentLoopConfig


@dataclass(frozen=True)
class ModelRetryDecision:
    """LLM 调用失败后的重试决策。

    Agent 循环只需要知道：是否重试、下次重试计数、等待多久。
    详细的重试策略封装在此处，避免散布在控制流中。

    Attributes:
        should_retry: 是否应该重试。
        reason: 决策原因（如 "retry_disabled"、"error_not_retryable"、"retry_limit_exhausted"）。
        next_retry_count: 下次重试的计数（从 1 开始）。
        delay_ms: 重试前的等待时间（毫秒），按指数退避计算。
    """
    should_retry: bool          # 是否重试
    reason: str                 # 决策原因
    next_retry_count: int       # 下次重试计数
    delay_ms: int               # 等待时间（毫秒）


@dataclass(frozen=True)
class ToolExecutionGate:
    """工具调用前的执行门禁：决定一批工具调用是否允许执行。

    在工具执行前检查两个条件：
    1. 是否存在连续重复的工具调用（可能陷入死循环）
    2. 是否达到最大工具迭代次数

    Attributes:
        should_execute: 是否允许执行。
        reason: 决策原因（如 "allowed"、"repeated_tool_call"、"max_iterations"）。
        error_code: 错误代码（仅 should_execute=False 时有值）。
        message: 错误描述信息。
        stop_reason: 运行停止原因。
        assistant_stop_reason: 设置到助手消息上的停止原因。
    """
    should_execute: bool        # 是否允许执行
    reason: str                 # 决策原因
    error_code: str | None = None           # 错误代码
    message: str | None = None              # 错误描述
    stop_reason: AgentRunStopReason | None = None       # 运行停止原因
    assistant_stop_reason: str | None = None             # 助手消息停止原因


@dataclass(frozen=True)
class PostToolRunDecision:
    """工具执行后的运行级决策。

    TaskController 描述任务层面应该发生什么（继续/修复/重新规划/停止），
    而 Agent 循环还需要知道运行状态、停止原因和错误详情。
    此对象是这两个概念之间的显式边界。

    Attributes:
        should_stop: 是否应该停止运行。
        reason: 决策原因（如 "continue"、"approval_required"、"cancelled"）。
        status: 运行状态（如 "waiting_approval"、"aborted"、"waiting_user"）。
        stop_reason: 停止原因（如 "approval_required"、"aborted"、"task_blocked"）。
        error_code: 错误代码（仅在需要报告错误时有值）。
        message: 错误描述信息。
        force_completion_check: 是否跳出工具循环并立即进入任务完成检查。
    """
    should_stop: bool           # 是否停止
    reason: str                 # 决策原因
    status: AgentRunStatus | None = None        # 运行状态
    stop_reason: AgentRunStopReason | None = None  # 停止原因
    error_code: str | None = None               # 错误代码
    message: str | None = None                  # 错误描述
    force_completion_check: bool = False        # 是否立即进入完成检查


@dataclass(frozen=True)
class CompletionRunDecision:
    """任务完成度检查后的运行级决策。

    Attributes:
        action: 决策动作（如 "satisfied"、"continue_with_steering"、"stop"）。
        should_stop: 是否应该停止运行。
        reason: 决策原因。
        status: 运行状态（仅 should_stop=True 时有值）。
        stop_reason: 停止原因（仅 should_stop=True 时有值）。
    """
    action: str                 # 决策动作
    should_stop: bool           # 是否停止
    reason: str                 # 决策原因
    status: AgentRunStatus | None = None        # 运行状态
    stop_reason: AgentRunStopReason | None = None  # 停止原因


def decide_model_retry(
    error: ErrorInfo,
    *,
    retries_so_far: int,
    config: AgentLoopConfig,
) -> ModelRetryDecision:
    """决定 LLM 调用失败后是否应该重试。

    重试策略：
    1. 如果重试功能被禁用 → 不重试
    2. 如果错误不可重试（如认证失败）→ 不重试
    3. 如果已达到最大重试次数 → 不重试
    4. 否则 → 重试，延迟按指数退避计算（base_delay * 2^(n-1)）

    Args:
        error: LLM 错误信息。
        retries_so_far: 已重试次数。
        config: Agent 循环配置（包含重试策略参数）。

    Returns:
        ModelRetryDecision: 重试决策。
    """
    # 检查重试功能是否启用
    if not config.retry_enabled:
        return ModelRetryDecision(
            should_retry=False,
            reason="retry_disabled",
            next_retry_count=retries_so_far,
            delay_ms=0,
        )
    # 检查错误是否可重试
    if not error.retryable:
        return ModelRetryDecision(
            should_retry=False,
            reason="error_not_retryable",
            next_retry_count=retries_so_far,
            delay_ms=0,
        )
    # 检查是否达到最大重试次数
    if retries_so_far >= config.max_model_retries:
        return ModelRetryDecision(
            should_retry=False,
            reason="retry_limit_exhausted",
            next_retry_count=retries_so_far,
            delay_ms=0,
        )
    # 计算下次重试的延迟（指数退避：base * 2^(n-1)）
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
    """决定一批工具调用是否允许执行（执行门禁）。

    检查两个安全条件：
    1. 重复调用检测：如果连续多次调用相同的工具（参数也相同），可能陷入死循环
    2. 迭代次数限制：如果已达到最大工具迭代次数，停止执行防止无限循环

    Args:
        tool_calls: 待执行的工具调用列表。
        state: 运行状态（包含重复调用检测和迭代计数）。
        config: Agent 循环配置（包含限制参数）。

    Returns:
        ToolExecutionGate: 执行门禁决策。
    """
    # 检查是否存在连续重复的工具调用
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
    # 检查是否达到最大工具迭代次数
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
    # 通过门禁检查，允许执行
    return ToolExecutionGate(should_execute=True, reason="allowed")


def decide_post_tool_run(
    tool_results: list[ToolResultMessage],
    *,
    task_decision: ExecutionDecision | None,
) -> PostToolRunDecision:
    """决定工具执行后是否需要暂停或停止运行。

    检查优先级（从高到低）：
    1. 工具需要审批 → 暂停等待用户审批
    2. 工具被取消 → 中止运行
    3. 无任务决策 → 继续运行
    4. 任务决策为"建议回滚" → 暂停等待用户确认
    5. 任务决策非"停止" → 继续运行
    6. 任务决策为"重新规划次数超限" → 失败停止
    7. 其他停止决策 → 暂停等待用户

    Args:
        tool_results: 工具执行结果列表。
        task_decision: 任务控制器的执行决策（可选）。

    Returns:
        PostToolRunDecision: 运行级决策。
    """
    # 检查是否有工具需要用户审批
    if any(result.status == "approval_required" for result in tool_results):
        return PostToolRunDecision(
            should_stop=True,
            reason="approval_required",
            status="waiting_approval",
            stop_reason="approval_required",
        )
    # 检查是否有工具被取消
    if any(result.status == "cancelled" for result in tool_results):
        return PostToolRunDecision(
            should_stop=True,
            reason="cancelled",
            status="aborted",
            stop_reason="aborted",
        )
    # 无任务决策时继续运行
    if task_decision is None:
        return PostToolRunDecision(should_stop=False, reason="continue")
    # 任务决策为"建议回滚"时暂停等待用户确认
    if task_decision.action == "propose_revert":
        return PostToolRunDecision(
            should_stop=True,
            reason=task_decision.reason,
            status="waiting_user",
            stop_reason="task_blocked",
        )
    # 任务决策为"完成"时，不再继续工具循环，交给完成检查生成闭环证据
    if task_decision.action == "finish":
        return PostToolRunDecision(
            should_stop=False,
            reason="finish",
            force_completion_check=True,
        )
    # 任务决策非"停止"时继续运行
    if task_decision.action != "stop":
        return PostToolRunDecision(should_stop=False, reason="continue")
    # 重新规划次数超限则失败停止
    if task_decision.reason == "replan_limit_exceeded":
        return PostToolRunDecision(
            should_stop=True,
            reason=task_decision.reason,
            status="failed",
            stop_reason="replan_limit",
            error_code="run.replan_limit",
            message=task_decision.reason,
        )
    # 其他停止决策：暂停等待用户指示
    return PostToolRunDecision(
        should_stop=True,
        reason=task_decision.reason,
        status="waiting_user",
        stop_reason="task_blocked",
    )


def decide_completion_run(check: CompletionCheck) -> CompletionRunDecision:
    """任务完成度检查后决定运行是否可以结束。

    三种决策：
    1. 任务已满足完成条件 → 不停止，继续运行（正常完成）
    2. 任务未满足但可以继续 → 不停止，注入引导消息继续尝试
    3. 任务未满足且无法继续 → 停止运行，等待用户指示

    Args:
        check: 任务完成度检查结果。

    Returns:
        CompletionRunDecision: 运行级决策。
    """
    # 任务已满足完成条件
    if check.satisfied:
        return CompletionRunDecision(
            action="satisfied",
            should_stop=False,
            reason=check.reason,
        )
    # 任务未满足但可以继续尝试（如需要运行验证）
    if check.can_continue:
        return CompletionRunDecision(
            action="continue_with_steering",
            should_stop=False,
            reason=check.reason,
        )
    # 任务未满足且无法继续 → 停止等待用户
    return CompletionRunDecision(
        action="stop",
        should_stop=True,
        reason=check.reason,
        status="waiting_user",
        stop_reason=(
            "task_blocked" if check.reason == "blocked_steps" else "task_incomplete"
        ),
    )
