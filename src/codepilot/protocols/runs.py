from __future__ import annotations

# 新手导读：runs.py 定义 AgentRunResult、停止原因和任务摘要。
# 关注点：一次 run 的最终输出会沿着这些类型返回给上层。

"""
Agent 运行结果与状态类型定义。

定义了一次 Agent 运行（run）的完整结果结构：
- 运行状态和停止原因
- 执行计数器（模型调用次数、工具迭代次数等）
- 运行验证结果
- 最终的运行结果汇总
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .errors import ErrorInfo
from .messages import AssistantMessage, Message, ToolResultMessage, UserMessage


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

_RUN_STATUSES = frozenset(
    {
        "running",
        "completed",
        "failed",
        "aborted",
        "waiting_approval",
        "waiting_user",
    }
)
_STOP_REASONS = frozenset(
    {
        "final_answer",
        "max_iterations",
        "model_error",
        "aborted",
        "approval_required",
        "repeated_tool_call",
        "replan_limit",
        "task_blocked",
        "task_incomplete",
        "internal_error",
    }
)
_VERIFICATION_STATUSES = frozenset({"passed", "failed", "cancelled", "unknown"})


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_attempts",
            _ensure_non_negative_int(self.model_attempts, field_name="model_attempts"),
        )
        object.__setattr__(
            self,
            "tool_iterations",
            _ensure_non_negative_int(self.tool_iterations, field_name="tool_iterations"),
        )
        object.__setattr__(
            self,
            "tool_calls",
            _ensure_non_negative_int(self.tool_calls, field_name="tool_calls"),
        )


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_call_id",
            _require_text(self.tool_call_id, field_name="tool_call_id"),
        )
        object.__setattr__(
            self,
            "tool_name",
            _require_text(self.tool_name, field_name="tool_name"),
        )
        object.__setattr__(
            self,
            "status",
            _ensure_verification_status(self.status),
        )
        object.__setattr__(
            self,
            "command",
            _optional_text(self.command),
        )
        object.__setattr__(
            self,
            "exit_code",
            _ensure_optional_int(self.exit_code, field_name="exit_code"),
        )
        object.__setattr__(
            self,
            "summary",
            _clean_text(self.summary),
        )


@dataclass
class TaskSummary:
    """任务摘要：一次运行中活跃任务计划的轻量级快照。

    由 TaskController.summarize() 生成，包含任务的完整状态信息。
    用于事件上报（task_plan_created 等）和 AgentRunResult.task 字段。

    Attributes:
        task_id: 任务唯一标识。
        goal: 任务目标描述。
        completed_steps: 已完成步骤的标题列表。
        pending_steps: 待处理步骤的标题列表。
        blocked_steps: 被阻塞步骤的标题列表。
        next_action: 下一步动作描述。
        completion_satisfied: 任务是否满足完成条件。
        completion_reason: 完成/未完成原因。
        attempts: 工具执行尝试记录列表（字典格式）。
        change_sets: 文件变更证据集合列表（字典格式）。
        replans: 重新规划记录列表（字典格式）。
        control_signal: 轻量级任务控制信号。
        step_details: 步骤详情字典（步骤标题 → 详情）。
    """

    task_id: str                                                                 # 任务唯一标识
    goal: str                                                                    # 任务目标
    completed_steps: list[str] = field(default_factory=list)                     # 已完成步骤
    pending_steps: list[str] = field(default_factory=list)                       # 待处理步骤
    blocked_steps: list[str] = field(default_factory=list)                       # 阻塞步骤
    next_action: str | None = None                                               # 下一步动作
    completion_satisfied: bool = False                                           # 是否满足完成条件
    completion_reason: str = ""                                                  # 完成原因
    attempts: list[dict[str, Any]] = field(default_factory=list)                 # 尝试记录
    change_sets: list[dict[str, Any]] = field(default_factory=list)              # 变更集合
    replans: list[dict[str, Any]] = field(default_factory=list)                  # 重新规划记录
    control_signal: dict[str, Any] = field(default_factory=dict)                 # 控制信号
    step_details: dict[str, dict[str, Any]] = field(default_factory=dict)        # 步骤详情

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_id",
            _require_text(self.task_id, field_name="task_id"),
        )
        object.__setattr__(
            self,
            "goal",
            _require_text(self.goal, field_name="goal"),
        )
        object.__setattr__(
            self,
            "completed_steps",
            _clean_text_list(self.completed_steps, field_name="completed_steps"),
        )
        object.__setattr__(
            self,
            "pending_steps",
            _clean_text_list(self.pending_steps, field_name="pending_steps"),
        )
        object.__setattr__(
            self,
            "blocked_steps",
            _clean_text_list(self.blocked_steps, field_name="blocked_steps"),
        )
        object.__setattr__(
            self,
            "next_action",
            _optional_text(self.next_action),
        )
        object.__setattr__(
            self,
            "completion_satisfied",
            _ensure_bool(self.completion_satisfied, field_name="completion_satisfied"),
        )
        object.__setattr__(
            self,
            "completion_reason",
            _clean_text(self.completion_reason),
        )
        object.__setattr__(
            self,
            "attempts",
            _copy_dict_list(self.attempts, field_name="attempts"),
        )
        object.__setattr__(
            self,
            "change_sets",
            _copy_dict_list(self.change_sets, field_name="change_sets"),
        )
        object.__setattr__(
            self,
            "replans",
            _copy_dict_list(self.replans, field_name="replans"),
        )
        object.__setattr__(
            self,
            "control_signal",
            _copy_dict(self.control_signal, field_name="control_signal"),
        )
        object.__setattr__(
            self,
            "step_details",
            _copy_nested_dict(self.step_details, field_name="step_details"),
        )


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            _require_text(self.run_id, field_name="run_id"),
        )
        object.__setattr__(
            self,
            "session_id",
            _optional_text(self.session_id),
        )
        object.__setattr__(
            self,
            "status",
            _ensure_run_status(self.status),
        )
        object.__setattr__(
            self,
            "stop_reason",
            _ensure_stop_reason(self.stop_reason),
        )
        if not isinstance(self.counters, AgentRunCounters):
            raise TypeError("AgentRunResult counters must be AgentRunCounters")
        object.__setattr__(
            self,
            "messages",
            _copy_messages(self.messages),
        )
        if self.final_message is not None and not isinstance(self.final_message, AssistantMessage):
            raise TypeError("AgentRunResult final_message must be AssistantMessage or None")
        if self.error is not None and not isinstance(self.error, ErrorInfo):
            raise TypeError("AgentRunResult error must be ErrorInfo or None")
        object.__setattr__(
            self,
            "affected_paths",
            _clean_unique_text_list(self.affected_paths, field_name="affected_paths"),
        )
        object.__setattr__(
            self,
            "workspace_changed",
            _ensure_bool(self.workspace_changed, field_name="workspace_changed"),
        )
        object.__setattr__(
            self,
            "verification",
            _copy_verification(self.verification),
        )
        if self.task is not None and not isinstance(self.task, TaskSummary):
            raise TypeError("AgentRunResult task must be TaskSummary or None")


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _optional_text(value: object) -> str | None:
    text = _clean_text(value)
    return text or None


def _require_text(value: object, *, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _ensure_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _ensure_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int or None")
    return value


def _ensure_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


def _ensure_run_status(value: object) -> AgentRunStatus:
    text = _clean_text(value)
    if text not in _RUN_STATUSES:
        raise ValueError(f"Unknown run status: {value}")
    return cast(AgentRunStatus, text)


def _ensure_stop_reason(value: object) -> AgentRunStopReason:
    text = _clean_text(value)
    if text not in _STOP_REASONS:
        raise ValueError(f"Unknown stop reason: {value}")
    return cast(AgentRunStopReason, text)


def _ensure_verification_status(value: object) -> RunVerificationStatus:
    text = _clean_text(value)
    if text not in _VERIFICATION_STATUSES:
        raise ValueError(f"Unknown verification status: {value}")
    return cast(RunVerificationStatus, text)


def _clean_text_list(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return [text for item in value if (text := _clean_text(item))]


def _clean_unique_text_list(value: object, *, field_name: str) -> list[str]:
    items = _clean_text_list(value, field_name=field_name)
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _copy_dict(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict")
    return deepcopy(value)


def _copy_nested_dict(value: object, *, field_name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict")
    copied: dict[str, dict[str, Any]] = {}
    for key, details in value.items():
        if not isinstance(details, dict):
            raise TypeError(f"{field_name} values must be dicts")
        copied[_require_text(key, field_name=f"{field_name} key")] = deepcopy(details)
    return copied


def _copy_dict_list(value: object, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    copied: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError(f"{field_name} entries must be dicts")
        copied.append(deepcopy(item))
    return copied


def _copy_messages(value: object) -> list[Message]:
    if not isinstance(value, list):
        raise TypeError("AgentRunResult messages must be a list")
    messages: list[Message] = []
    for message in value:
        if not isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
            raise TypeError("AgentRunResult messages entries must be Message objects")
        messages.append(message)
    return messages


def _copy_verification(value: object) -> list[RunVerification]:
    if not isinstance(value, list):
        raise TypeError("AgentRunResult verification must be a list")
    verification: list[RunVerification] = []
    for item in value:
        if not isinstance(item, RunVerification):
            raise TypeError("AgentRunResult verification entries must be RunVerification")
        verification.append(item)
    return verification


__all__ = [
    "AgentRunCounters",
    "AgentRunResult",
    "AgentRunStatus",
    "AgentRunStopReason",
    "RunVerification",
    "RunVerificationStatus",
    "TaskSummary",
]
