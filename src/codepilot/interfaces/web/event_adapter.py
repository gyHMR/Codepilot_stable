from __future__ import annotations

# 新手导读：event_adapter.py 把内部 AgentEvent 转换成 Web 前端友好的事件格式。
# 关注点：新增前端事件字段时，先在这里找映射关系。

"""运行时/核心事件到 Web Console 事件信封的适配器。"""

from dataclasses import asdict, is_dataclass
from typing import Any

from codepilot.protocols.events import AgentEvent

from .schemas import WebErrorPayload, WebEventEnvelope, WebEventKind


def agent_event_to_web(event: AgentEvent) -> WebEventEnvelope:
    """将 Agent 事件转换为 Web 事件信封。

    Web Console 需要区分“普通运行事件”和“需要用户操作的事件”。
    这里保留原始事件载荷，避免接口层重新解释工具参数或模型输出。
    """

    event_kind: WebEventKind = "agent_event"
    if event.get("type") == "tool_approval_required":
        event_kind = "tool_approval_required"

    return WebEventEnvelope(
        type=event_kind,
        session_id=event.get("sessionId"),
        payload=_jsonable(event),
    )


def error_to_web(message: str, *, session_id: str | None = None, code: str = "error") -> WebEventEnvelope:
    """将错误信息转换为 Web 错误事件信封。"""
    error = WebErrorPayload(code=code, message=message)
    return WebEventEnvelope(
        type="error",
        session_id=session_id,
        payload=asdict(error),
    )


def _jsonable(value: Any) -> Any:
    """递归将值转换为 JSON 可序列化格式（处理 dataclass、dict、list、set 等）。"""
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return value
