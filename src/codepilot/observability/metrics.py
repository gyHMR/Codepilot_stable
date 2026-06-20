from __future__ import annotations

"""单次 Agent Run 的轻量级可观测性投影。

本模块只读取 Run 结果和已发射的事件，不调用模型、不执行工具、
不写入状态，也不影响正常的运行决策。
"""

from dataclasses import dataclass, field
from typing import Any

from codepilot.protocols import AgentRunResult

from .events import normalize_event_value


@dataclass(frozen=True)
class ModelCallRecord:
    """单次模型调用记录。"""
    index: int                            # 调用序号
    provider: str                         # Provider 名称
    model: str                            # 模型 ID
    api: str                              # API 协议
    stop_reason: str                      # 停止原因
    token_usage: dict[str, int] = field(default_factory=dict)  # token 用量
    cost: dict[str, float] = field(default_factory=dict)       # 费用
    timestamp: int | None = None          # 时间戳
    latency_ms: int | None = None         # 延迟（毫秒）
    error: dict[str, Any] | None = None   # 错误信息

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "provider": self.provider,
            "model": self.model,
            "api": self.api,
            "stop_reason": self.stop_reason,
            "token_usage": dict(self.token_usage),
            "cost": dict(self.cost),
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass(frozen=True)
class ToolCallRecord:
    """单次工具调用记录。"""
    tool_call_id: str                     # 调用 ID
    tool_name: str                        # 工具名称
    status: str                           # 执行状态
    is_error: bool = False                # 是否出错
    approved: bool = True                 # 是否已审批
    approval_id: str | None = None        # 审批 ID
    error_reason: str | None = None       # 错误原因
    duration_ms: int | None = None        # 耗时（毫秒）
    affected_paths: list[str] = field(default_factory=list)  # 受影响文件
    workspace_changed: bool | None = None # 是否修改了工作区
    verification_status: str | None = None  # 验证状态
    output_truncated: bool = False        # 输出是否被截断

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "is_error": self.is_error,
            "approved": self.approved,
            "approval_id": self.approval_id,
            "error_reason": self.error_reason,
            "duration_ms": self.duration_ms,
            "affected_paths": list(self.affected_paths),
            "workspace_changed": self.workspace_changed,
            "verification_status": self.verification_status,
            "output_truncated": self.output_truncated,
        }


@dataclass(frozen=True)
class RunMetrics:
    """Run 聚合指标。"""
    duration_ms: int | None = None        # 总耗时（毫秒）
    model_attempts: int = 0               # 模型调用尝试次数
    model_calls: int = 0                  # 实际模型调用次数
    model_errors: int = 0                 # 模型错误次数
    tool_iterations: int = 0              # 工具迭代次数
    tool_calls: int = 0                   # 工具调用次数
    tool_errors: int = 0                  # 工具错误次数
    input_tokens: int = 0                 # 输入 token 总数
    output_tokens: int = 0                # 输出 token 总数
    total_tokens: int = 0                 # token 总数
    total_cost: float = 0.0               # 总费用
    verification_count: int = 0           # 验证次数
    verification_passed: int = 0          # 验证通过次数
    approval_count: int = 0               # 审批请求次数
    denied_count: int = 0                 # 拒绝次数

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "model_attempts": self.model_attempts,
            "model_calls": self.model_calls,
            "model_errors": self.model_errors,
            "tool_iterations": self.tool_iterations,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "verification_count": self.verification_count,
            "verification_passed": self.verification_passed,
            "approval_count": self.approval_count,
            "denied_count": self.denied_count,
        }


def build_model_call_records(
    result: AgentRunResult | dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
) -> list[ModelCallRecord]:
    """从 Run 结果和事件提取模型调用记录列表。"""
    record = _run_record(result)
    messages = [
        message
        for message in _list_of_dicts(record.get("messages"))
        if message.get("role") == "assistant"
    ]
    latencies = _assistant_message_latencies(_normalize_events(events or []))
    records: list[ModelCallRecord] = []

    for message in messages:
        usage = _dict(message.get("usage"))
        cost = _dict(usage.get("cost"))
        index = len(records) + 1
        error = _dict_or_none(message.get("error_info"))
        if error is None and message.get("error_message"):
            error = {"message": str(message.get("error_message"))}
        records.append(
            ModelCallRecord(
                index=index,
                provider=str(message.get("provider", "")),
                model=str(message.get("model", "")),
                api=str(message.get("api", "")),
                stop_reason=str(message.get("stop_reason", "")),
                token_usage={
                    "input_tokens": _int(usage.get("input")),
                    "output_tokens": _int(usage.get("output")),
                    "cache_read_tokens": _int(usage.get("cache_read")),
                    "cache_write_tokens": _int(usage.get("cache_write")),
                    "total_tokens": _int(usage.get("total_tokens")),
                },
                cost={
                    "input": _float(cost.get("input")),
                    "output": _float(cost.get("output")),
                    "cache_read": _float(cost.get("cache_read")),
                    "cache_write": _float(cost.get("cache_write")),
                    "total": _float(cost.get("total")),
                },
                timestamp=_optional_int(message.get("timestamp")),
                latency_ms=latencies[index - 1] if index - 1 < len(latencies) else None,
                error=error,
            )
        )
    return records


def build_tool_call_records(
    result: AgentRunResult | dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
) -> list[ToolCallRecord]:
    """从 Run 结果和事件提取工具调用记录列表。"""
    normalized_events = _normalize_events(events or [])
    records = [_tool_record_from_event(event) for event in normalized_events if event.get("type") == "tool_execution_end"]
    if records:
        return records

    record = _run_record(result)
    messages = _list_of_dicts(record.get("messages"))
    return [
        _tool_record_from_message(message)
        for message in messages
        if message.get("role") == "toolResult"
    ]


def build_run_metrics(
    result: AgentRunResult | dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
) -> RunMetrics:
    """从 Run 结果和事件构建聚合指标。"""
    record = _run_record(result)
    counters = _dict(record.get("counters"))
    model_calls = build_model_call_records(record, events=events)
    tool_calls = build_tool_call_records(record, events=events)
    verification = _list_of_dicts(record.get("verification"))

    counted_tool_calls = _int(counters.get("tool_calls"))

    return RunMetrics(
        duration_ms=_duration_ms(_normalize_events(events or [])),
        model_attempts=_int(counters.get("model_attempts")) or len(model_calls),
        model_calls=len(model_calls),
        model_errors=sum(1 for item in model_calls if item.stop_reason == "error" or item.error),
        tool_iterations=_int(counters.get("tool_iterations")),
        tool_calls=counted_tool_calls or len(tool_calls),
        tool_errors=sum(1 for item in tool_calls if item.is_error),
        input_tokens=sum(item.token_usage.get("input_tokens", 0) for item in model_calls),
        output_tokens=sum(item.token_usage.get("output_tokens", 0) for item in model_calls),
        total_tokens=sum(item.token_usage.get("total_tokens", 0) for item in model_calls),
        total_cost=sum(item.cost.get("total", 0.0) for item in model_calls),
        verification_count=len(verification),
        verification_passed=sum(1 for item in verification if item.get("status") == "passed"),
        approval_count=sum(1 for item in tool_calls if item.status == "approval_required"),
        denied_count=sum(1 for item in tool_calls if item.status == "denied"),
    )


def _tool_record_from_event(event: dict[str, Any]) -> ToolCallRecord:
    result = _dict(event.get("result"))
    verification = _dict(result.get("verification"))
    affected_paths = event.get("affectedPaths")
    if not isinstance(affected_paths, list):
        affected_paths = result.get("affected_paths")
    return ToolCallRecord(
        tool_call_id=str(event.get("toolCallId", result.get("tool_call_id", ""))),
        tool_name=str(event.get("toolName", result.get("tool_name", ""))),
        status=str(event.get("status", result.get("status", ""))),
        is_error=bool(event.get("isError", result.get("is_error", False))),
        approved=bool(event.get("approved", result.get("approved", True))),
        approval_id=_optional_str(event.get("approvalId", result.get("approval_id"))),
        error_reason=_optional_str(event.get("errorReason")),
        duration_ms=_optional_int(event.get("durationMs")),
        affected_paths=[str(path) for path in affected_paths or [] if isinstance(path, str)],
        workspace_changed=_optional_bool(event.get("workspaceChanged", result.get("workspace_changed"))),
        verification_status=_optional_str(verification.get("status")),
        output_truncated=bool(event.get("outputTruncated", False)),
    )


def _tool_record_from_message(message: dict[str, Any]) -> ToolCallRecord:
    verification = _dict(message.get("verification"))
    return ToolCallRecord(
        tool_call_id=str(message.get("tool_call_id", "")),
        tool_name=str(message.get("tool_name", "")),
        status=str(message.get("status", "")),
        is_error=bool(message.get("is_error", False)),
        approved=bool(message.get("approved", True)),
        approval_id=_optional_str(message.get("approval_id")),
        error_reason=_optional_str(message.get("error_code")),
        affected_paths=[
            str(path)
            for path in message.get("affected_paths", [])
            if isinstance(path, str)
        ],
        workspace_changed=_optional_bool(message.get("workspace_changed")),
        verification_status=_optional_str(verification.get("status")),
    )


def _assistant_message_latencies(events: list[dict[str, Any]]) -> list[int | None]:
    starts: list[int] = []
    latencies: list[int | None] = []
    for event in events:
        event_type = event.get("type")
        message = _dict(event.get("message"))
        if message.get("role") != "assistant":
            continue
        timestamp = _optional_int(event.get("timestamp"))
        if event_type == "message_start" and timestamp is not None:
            starts.append(timestamp)
        elif event_type == "message_end":
            if starts and timestamp is not None:
                latencies.append(max(0, timestamp - starts.pop(0)))
            else:
                latencies.append(None)
    return latencies


def _duration_ms(events: list[dict[str, Any]]) -> int | None:
    timestamps = [event.get("timestamp") for event in events]
    numeric = [int(item) for item in timestamps if isinstance(item, int) and not isinstance(item, bool)]
    if len(numeric) < 2:
        return None
    return max(numeric) - min(numeric)


def _normalize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for event in events:
        value = normalize_event_value(event)
        if isinstance(value, dict):
            normalized.append(value)
    return normalized


def _run_record(result: AgentRunResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    record = normalize_event_value(result)
    return record if isinstance(record, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


__all__ = [
    "ModelCallRecord",
    "RunMetrics",
    "ToolCallRecord",
    "build_model_call_records",
    "build_run_metrics",
    "build_tool_call_records",
]
