from __future__ import annotations

# 新手导读：runtime.py 定义 run 状态、run 结果和运行时事件。
# 关注点：这里描述跨层可观察的运行事实，不管理 session，也不分发事件。

"""
Agent 运行结果与事件类型定义。

定义了一次 Agent 运行（run）的完整结果结构：
- 运行状态和停止原因
- 执行计数器（模型调用次数、工具迭代次数等）
- 运行验证结果
- 最终的运行结果汇总
- 运行过程中的事件信封和事件 payload
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, TypedDict, cast

from .errors import ErrorInfo
from .conversation import AssistantMessage, Message, ToolResultMessage, UserMessage
from .tools import ToolResult, ToolResultStatus


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
    "approval_denied",       # 用户拒绝工具审批
    "plan_approval_required",  # plan 模式提出计划后等待用户批准
    "repeated_tool_call",    # 检测到重复的工具调用（可能陷入循环）
    "tool_call_limit",       # 工具调用数量超出限制
    "tool_unavailable",      # 模型请求了不可用工具
    "run_guard",             # 运行护栏要求继续处理或等待用户
    "missing_tool_port",     # 内部工具端口缺失
    "missing_approval_decision",  # approval resume 缺少审批决定
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
        "approval_denied",
        "plan_approval_required",
        "repeated_tool_call",
        "tool_call_limit",
        "tool_unavailable",
        "run_guard",
        "missing_tool_port",
        "missing_approval_decision",
        "internal_error",
    }
)
_VERIFICATION_STATUSES = frozenset({"passed", "failed", "cancelled", "unknown"})
_SIGNAL_VERIFICATION_STATUSES = frozenset(
    {"unknown", "passed", "failed", "cancelled", "stale"}
)


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


RunSignalsVerificationStatus = Literal["unknown", "passed", "failed", "cancelled", "stale"]


@dataclass
class PlanSummary:
    """Soft plan board snapshot saved with a run result."""

    schema_version: int
    plan_id: str
    status: str
    approval_state: str
    origin_mode: str
    objective: str
    items: list[dict[str, str]] = field(default_factory=list)
    explanation: str = ""
    created_at: str = ""
    updated_at: str = ""
    last_update_run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _ensure_non_negative_int(self.schema_version, field_name="schema_version"),
        )
        object.__setattr__(self, "plan_id", _require_text(self.plan_id, field_name="plan_id"))
        object.__setattr__(self, "status", _require_text(self.status, field_name="status"))
        object.__setattr__(
            self,
            "approval_state",
            _require_text(self.approval_state, field_name="approval_state"),
        )
        object.__setattr__(
            self,
            "origin_mode",
            _require_text(self.origin_mode, field_name="origin_mode"),
        )
        object.__setattr__(self, "objective", _clean_text(self.objective))
        object.__setattr__(self, "items", _copy_plan_items(self.items))
        object.__setattr__(self, "explanation", _clean_text(self.explanation))
        object.__setattr__(self, "created_at", _clean_text(self.created_at))
        object.__setattr__(self, "updated_at", _clean_text(self.updated_at))
        object.__setattr__(self, "last_update_run_id", _optional_text(self.last_update_run_id))


@dataclass
class RunSignalsSummary:
    """Observable run facts used by RunGuard and context reporting."""

    workspace_changed: bool = False
    affected_paths: list[str] = field(default_factory=list)
    verification_status: RunSignalsVerificationStatus = "unknown"
    last_error: dict[str, Any] | None = None
    approval_required: bool = False
    tool_unavailable: bool = False
    cancelled: bool = False
    counters: AgentRunCounters = field(default_factory=AgentRunCounters)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workspace_changed",
            _ensure_bool(self.workspace_changed, field_name="workspace_changed"),
        )
        object.__setattr__(
            self,
            "affected_paths",
            _clean_unique_text_list(self.affected_paths, field_name="affected_paths"),
        )
        object.__setattr__(
            self,
            "verification_status",
            _ensure_signal_verification_status(self.verification_status),
        )
        if self.last_error is not None and not isinstance(self.last_error, dict):
            raise TypeError("RunSignalsSummary last_error must be a dict or None")
        object.__setattr__(
            self,
            "last_error",
            deepcopy(self.last_error) if self.last_error is not None else None,
        )
        object.__setattr__(
            self,
            "approval_required",
            _ensure_bool(self.approval_required, field_name="approval_required"),
        )
        object.__setattr__(
            self,
            "tool_unavailable",
            _ensure_bool(self.tool_unavailable, field_name="tool_unavailable"),
        )
        object.__setattr__(
            self,
            "cancelled",
            _ensure_bool(self.cancelled, field_name="cancelled"),
        )
        if not isinstance(self.counters, AgentRunCounters):
            raise TypeError("RunSignalsSummary counters must be AgentRunCounters")


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
    plan: PlanSummary | None = None
    signals: RunSignalsSummary = field(default_factory=RunSignalsSummary)

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
        if self.plan is not None and not isinstance(self.plan, PlanSummary):
            raise TypeError("AgentRunResult plan must be PlanSummary or None")
        if not isinstance(self.signals, RunSignalsSummary):
            raise TypeError("AgentRunResult signals must be RunSignalsSummary")


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


def _ensure_signal_verification_status(value: object) -> RunSignalsVerificationStatus:
    text = _clean_text(value)
    if text not in _SIGNAL_VERIFICATION_STATUSES:
        raise ValueError(f"Unknown run signal verification status: {value}")
    return cast(RunSignalsVerificationStatus, text)


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


def _copy_plan_items(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError("PlanSummary items must be a list")
    items: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"PlanSummary items[{index}] must be a dict")
        step = _require_text(item.get("step"), field_name=f"items[{index}].step")
        status = _require_text(item.get("status"), field_name=f"items[{index}].status")
        item_id = _require_text(item.get("id"), field_name=f"items[{index}].id")
        items.append({"id": item_id, "step": step, "status": status})
    return items


# 运行时事件类型枚举：覆盖 Agent 运行全生命周期的公共事件。
RuntimeEventType = Literal[
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_start",
    "message_update",
    "message_end",
    "model_retry_start",
    "tool_started",
    "tool_completed",
    "tool_failed",
    "tool_interrupted",
    "context_projected",
    "context_preflight",
    "context_compacted",
    "context_projection_failed",
    "context_freshness_checked",
    "checkpoint_saved",
    "checkpoint_restored",
    "checkpoint_cleared",
    "memory_retrieved",
    "memory_warning",
    "memory_candidate_created",
    "memory_record_created",
    "memory_record_approved",
    "memory_record_edited",
    "memory_record_disabled",
    "memory_record_deleted",
    "memory_record_superseded",
    "plan_proposed",
    "plan_approval_required",
    "plan_approved",
    "plan_rejected",
    "plan_updated",
    "plan_completed",
    "plan_abandoned",
    "plan_state_warning",
    "run_guard_checked",
    "file_diff",
    "error",
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
        "tool_started",
        "tool_completed",
        "tool_failed",
        "tool_interrupted",
        "context_projected",
        "context_preflight",
        "context_compacted",
        "context_projection_failed",
        "context_freshness_checked",
        "checkpoint_saved",
        "checkpoint_restored",
        "checkpoint_cleared",
        "memory_retrieved",
        "memory_warning",
        "memory_candidate_created",
        "memory_record_created",
        "memory_record_approved",
        "memory_record_edited",
        "memory_record_disabled",
        "memory_record_deleted",
        "memory_record_superseded",
        "plan_proposed",
        "plan_approval_required",
        "plan_approved",
        "plan_rejected",
        "plan_updated",
        "plan_completed",
        "plan_abandoned",
        "plan_state_warning",
        "run_guard_checked",
        "file_diff",
        "error",
    }
)


def ensure_runtime_event_type(value: object) -> RuntimeEventType:
    """Validate a runtime event type shared by core/runtime/interfaces."""

    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError("runtime event type cannot be empty")
    if text not in _RUNTIME_EVENT_TYPES:
        raise ValueError(f"Unknown runtime event type: {value}")
    return cast(RuntimeEventType, text)


class AgentEventBase(TypedDict):
    """Stable envelope shared by all runtime events."""

    type: RuntimeEventType
    runId: str
    turnId: int
    eventId: str
    timestamp: int
    sessionId: str | None


EventEnvelope = AgentEventBase


class AgentStartEvent(AgentEventBase):
    type: Literal["agent_start"]


class AgentEndEvent(AgentEventBase):
    type: Literal["agent_end"]
    messages: list[Message]
    status: AgentRunStatus
    stopReason: AgentRunStopReason
    counters: AgentRunCounters
    result: AgentRunResult


class TurnStartEvent(AgentEventBase):
    type: Literal["turn_start"]


class TurnEndEvent(AgentEventBase):
    type: Literal["turn_end"]
    message: AssistantMessage
    toolResults: list[ToolResultMessage]


class MessageStartEvent(AgentEventBase):
    type: Literal["message_start"]
    message: Message


class MessageUpdateEvent(AgentEventBase):
    type: Literal["message_update"]
    message: Message
    assistantMessageEvent: dict[str, Any]


class MessageEndEvent(AgentEventBase):
    type: Literal["message_end"]
    message: Message


class ModelRetryStartEvent(AgentEventBase):
    type: Literal["model_retry_start"]
    attempt: int
    maxAttempts: int
    delayMs: int
    error: ErrorInfo


class ToolStartedEvent(AgentEventBase):
    type: Literal["tool_started"]
    toolCallId: str
    toolName: str
    args: dict[str, Any]


class ToolFinishedEvent(AgentEventBase):
    type: Literal["tool_completed", "tool_failed", "tool_interrupted"]
    toolCallId: str
    toolName: str
    result: Any
    status: ToolResultStatus
    isError: bool
    approved: bool
    approvalId: str | None
    errorReason: str | None


class ErrorEvent(AgentEventBase, total=False):
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


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ModelRetryStartEvent
    | ToolStartedEvent
    | ToolFinishedEvent
    | ErrorEvent
)
RuntimeEvent = AgentEvent
AgentEventSink = Callable[[AgentEvent], None | Awaitable[None]]


__all__ = [
    "AgentEndEvent",
    "AgentEvent",
    "AgentEventBase",
    "AgentEventSink",
    "AgentStartEvent",
    "AgentRunCounters",
    "AgentRunResult",
    "AgentRunStatus",
    "AgentRunStopReason",
    "ErrorEvent",
    "EventEnvelope",
    "ensure_runtime_event_type",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "ModelRetryStartEvent",
    "PlanSummary",
    "RunVerification",
    "RunVerificationStatus",
    "RunSignalsSummary",
    "RunSignalsVerificationStatus",
    "RuntimeEvent",
    "RuntimeEventType",
    "ToolFinishedEvent",
    "ToolStartedEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
