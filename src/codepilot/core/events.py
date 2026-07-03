from __future__ import annotations

# 新手导读：AgentEventEmitter 统一给 core 层事件补齐 run/turn/sequence 等观测字段。
# 关注点：事件从这里发出后，会被 CLI 渲染、session 持久化和 observability 消费。

"""Agent 事件信封辅助工具：为原始事件注入 run/turn/event 元数据后转发。"""

import asyncio
import time
from typing import Any, cast

from codepilot.protocols import AgentEvent, AgentEventSink, ensure_runtime_event_type


def now_ms() -> int:
    """返回当前时间戳（毫秒）。"""
    return int(time.time() * 1000)


async def maybe_await(value: Any) -> Any:
    """统一处理同步/异步值：可等待对象则 await，否则直接返回。"""
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


class AgentEventEmitter:
    """Agent 事件发射器：在转发事件前注入稳定的 run/turn/event 元数据。"""

    def __init__(
        self,
        sink: AgentEventSink,
        *,
        run_id: str,
        session_id: str | None = None,
    ) -> None:
        self._sink = sink          # 事件接收端回调
        self._session_id = _optional_event_text(session_id)
        self.run_id = _require_event_text(run_id, field_name="run_id")
        self.turn_id = 0           # 当前轮次计数
        self._event_seq = 0        # 事件序列号（自增）

    async def emit(self, event: dict[str, Any]) -> None:
        """发射一个事件：注入 runId、turnId、eventId、timestamp 后转发给 sink。"""
        if not isinstance(event, dict):
            raise TypeError("event must be a dict")
        event_type = ensure_runtime_event_type(event.get("type"))
        if event_type == "turn_start":
            self.turn_id += 1

        self._event_seq += 1
        enriched = {
            **event,
            "type": event_type,
            "runId": self.run_id,
            "turnId": self.turn_id,
            "eventId": f"{self.run_id}:{self._event_seq}",
            "timestamp": now_ms(),
            "sessionId": self._session_id,
        }
        await maybe_await(self._sink(cast(AgentEvent, enriched)))


def _clean_event_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _optional_event_text(value: object) -> str | None:
    text = _clean_event_text(value)
    return text or None


def _require_event_text(value: object, *, field_name: str) -> str:
    text = _clean_event_text(value)
    if not text:
        raise ValueError(f"event {field_name} cannot be empty")
    return text
