from __future__ import annotations

"""
运行时事件类型定义。

定义了 Agent 运行过程中产生的所有事件类型，用于：
- 实时通知外部监听器（UI 更新、日志记录等）
- 流式渲染助手回复
- 追踪工具调用生命周期
- 错误传播和重试通知

事件层次结构：
- AgentEventBase: 所有事件的基类（TypedDict），包含 runId、turnId 等公共字段
- 各具体事件类型继承自 AgentEventBase，通过 type 字段区分
"""

from typing import Any, Awaitable, Callable, Literal, TypedDict, cast

from .errors import ErrorInfo
from .llm import LLMStreamEvent
from .messages import AssistantMessage, Message, ToolResultMessage
from .runs import AgentRunCounters, AgentRunResult, AgentRunStatus, AgentRunStopReason
from .tools import ToolResult, ToolResultStatus


# 运行时事件类型枚举：覆盖 Agent 运行全生命周期的所有事件
RuntimeEventType = Literal[
    "agent_start",               # Agent 运行开始
    "agent_end",                 # Agent 运行结束
    "turn_start",                # 一轮对话开始
    "turn_end",                  # 一轮对话结束
    "message_start",             # 消息开始生成
    "message_update",            # 消息增量更新（流式）
    "message_end",               # 消息生成完成
    "model_retry_start",         # 模型调用重试开始
    "tool_execution_start",      # 工具执行开始
    "tool_execution_update",     # 工具执行增量更新
    "tool_approval_required",    # 工具调用需要审批
    "tool_approval_resolved",    # 工具调用审批已解决
    "tool_approval_decision",    # 用户提交了工具审批决策
    "tool_approval_result_replaced",  # 审批恢复替换了 pending 工具结果
    "tool_execution_end",        # 工具执行结束
    "context_prepared",          # 模型调用前上下文已编译
    "context_freshness_checked", # 上下文新鲜度已检查
    "context_compacted",         # 会话上下文已压缩
    "memory_retrieved",          # 长期记忆已检索
    "memory_created",            # 用户命令创建了记忆
    "memory_promoted",           # 用户命令提升了记忆
    "memory_invalidated",        # 用户命令废弃了记忆
    "memory_updated",            # 运行收尾写入了记忆
    "memory_warning",            # 记忆流程产生警告
    "task_plan_created",         # 任务计划已创建
    "task_step_updated",         # 任务步骤状态已更新
    "task_decision",             # 任务控制器做出执行决策
    "task_recovery_updated",     # 会话任务恢复投影已更新
    "task_recovery_warning",     # 任务恢复流程产生警告
    "completion_checked",        # 完成门槛检查结果
    "file_diff",                 # 文件变更差异
    "error",                     # 错误事件
]
_RUNTIME_EVENT_TYPES = frozenset(
    {
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
        "tool_approval_decision",
        "tool_approval_result_replaced",
        "tool_execution_end",
        "context_prepared",
        "context_freshness_checked",
        "context_compacted",
        "memory_retrieved",
        "memory_created",
        "memory_promoted",
        "memory_invalidated",
        "memory_updated",
        "memory_warning",
        "task_plan_created",
        "task_step_updated",
        "task_decision",
        "task_recovery_updated",
        "task_recovery_warning",
        "completion_checked",
        "file_diff",
        "error",
    }
)


def ensure_runtime_event_type(value: object) -> RuntimeEventType:
    """校验并清洗运行时事件类型。

    事件对象仍然使用 TypedDict 表达开放载荷；这里保护的是最基础的
    分发字段，确保 core/runtime/interface 都能按同一份事件词表处理。
    """

    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError("runtime event type cannot be empty")
    if text not in _RUNTIME_EVENT_TYPES:
        raise ValueError(f"Unknown runtime event type: {value}")
    return cast(RuntimeEventType, text)


class AgentEventBase(TypedDict):
    """所有 Agent 事件的基类（稳定信封）。

    每个事件都携带这些公共字段，用于事件的标识、排序和关联。

    Attributes:
        type: 事件类型（见 RuntimeEventType）。
        runId: 本次运行的唯一 ID。
        turnId: 当前轮次编号（从 1 开始）。
        eventId: 事件唯一 ID（格式 "runId:seq"）。
        timestamp: 事件时间戳（毫秒）。
        sessionId: 所属会话 ID（可选）。
    """

    type: RuntimeEventType
    runId: str
    turnId: int
    eventId: str
    timestamp: int
    sessionId: str | None


# EventEnvelope 是 AgentEventBase 的别名
EventEnvelope = AgentEventBase


# ── 具体事件类型 ────────────────────────────────────────────────

class AgentStartEvent(AgentEventBase):
    """Agent 运行开始事件。"""
    type: Literal["agent_start"]


class AgentEndEvent(AgentEventBase):
    """Agent 运行结束事件。

    Attributes:
        messages: 本次运行产生的所有消息。
        status: 运行最终状态。
        stopReason: 停止原因。
        counters: 执行计数器。
        result: 完整的运行结果。
    """

    type: Literal["agent_end"]
    messages: list[Message]
    status: AgentRunStatus
    stopReason: AgentRunStopReason
    counters: AgentRunCounters
    result: AgentRunResult


class TurnStartEvent(AgentEventBase):
    """一轮对话开始事件。"""
    type: Literal["turn_start"]


class TurnEndEvent(AgentEventBase):
    """一轮对话结束事件。

    Attributes:
        message: 本轮的助手回复消息。
        toolResults: 本轮执行的工具结果列表。
    """

    type: Literal["turn_end"]
    message: AssistantMessage
    toolResults: list[ToolResultMessage]


class MessageStartEvent(AgentEventBase):
    """消息开始生成事件（流式响应的第一个事件）。

    Attributes:
        message: 开始生成的消息对象（初始状态）。
    """

    type: Literal["message_start"]
    message: Message


class MessageUpdateEvent(AgentEventBase):
    """消息增量更新事件（流式响应的中间事件）。

    Attributes:
        message: 更新后的消息对象（包含最新状态）。
        assistantMessageEvent: 触发此更新的原始 LLM 流式事件。
    """

    type: Literal["message_update"]
    message: Message
    assistantMessageEvent: LLMStreamEvent


class MessageEndEvent(AgentEventBase):
    """消息生成完成事件（流式响应的最后一个事件）。

    Attributes:
        message: 最终完成的消息对象。
    """

    type: Literal["message_end"]
    message: Message


class ModelRetryStartEvent(AgentEventBase):
    """模型调用重试开始事件。

    当模型调用失败且满足重试条件时触发。

    Attributes:
        attempt: 当前重试次数（从 1 开始）。
        maxAttempts: 最大重试次数。
        delayMs: 本次重试的延迟时间（毫秒）。
        error: 触发重试的错误信息。
    """

    type: Literal["model_retry_start"]
    attempt: int
    maxAttempts: int
    delayMs: int
    error: ErrorInfo


class ToolExecutionStartEvent(AgentEventBase):
    """工具执行开始事件。

    Attributes:
        toolCallId: 工具调用 ID。
        toolName: 工具名称。
        args: 工具调用参数。
    """

    type: Literal["tool_execution_start"]
    toolCallId: str
    toolName: str
    args: dict[str, Any]


class ToolExecutionUpdateEvent(AgentEventBase):
    """工具执行增量更新事件（长运行工具的进度通知）。

    Attributes:
        toolCallId: 工具调用 ID。
        toolName: 工具名称。
        args: 工具调用参数。
        partialResult: 部分执行结果。
    """

    type: Literal["tool_execution_update"]
    toolCallId: str
    toolName: str
    args: dict[str, Any]
    partialResult: ToolResult


class ToolExecutionEndEvent(AgentEventBase):
    """工具执行结束事件。

    Attributes:
        toolCallId: 工具调用 ID。
        toolName: 工具名称。
        result: 最终执行结果。
        status: 执行状态。
        isError: 是否为错误。
        approved: 是否已通过审批。
        approvalId: 审批记录 ID（可选）。
        errorReason: 错误原因（可选）。
    """

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
    """错误事件。

    使用 total=False 使所有字段都是可选的，
    不同来源的错误可能填充不同的字段。

    Attributes:
        error: 错误描述文本。
        message: 详细错误消息。
        source: 错误来源。
        code: 错误代码。
        retryable: 是否可重试。
        provider: 出错的 provider。
        model: 出错的模型。
        statusCode: HTTP 状态码（可选）。
        errorInfo: 结构化错误信息。
    """

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


# ── 联合类型与回调类型 ──────────────────────────────────────────

# Agent 事件联合类型：所有可能的事件类型的集合
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

# RuntimeEvent 是 AgentEvent 的别名
RuntimeEvent = AgentEvent

# 事件接收器类型：可以是同步或异步回调函数
AgentEventSink = Callable[[AgentEvent], None | Awaitable[None]]


__all__ = [
    "AgentEndEvent",
    "AgentEvent",
    "AgentEventBase",
    "AgentEventSink",
    "AgentStartEvent",
    "ErrorEvent",
    "EventEnvelope",
    "ensure_runtime_event_type",
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
