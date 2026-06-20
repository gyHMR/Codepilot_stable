from __future__ import annotations

"""Agent 事件信封辅助工具：为原始事件注入 run/turn/event 元数据后转发。"""

import asyncio
import time
from typing import Any, cast

from codepilot.protocols import AgentEvent, AgentEventSink


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
        self._session_id = session_id
        self.run_id = run_id
        self.turn_id = 0           # 当前轮次计数
        self._event_seq = 0        # 事件序列号（自增）

    async def emit(self, event: dict[str, Any]) -> None:
        """发射一个事件：注入 runId、turnId、eventId、timestamp 后转发给 sink。"""
        event_type = event.get("type")
        if event_type == "turn_start":
            self.turn_id += 1

        self._event_seq += 1
        enriched = {
            **event,
            "runId": self.run_id,
            "turnId": self.turn_id,
            "eventId": f"{self.run_id}:{self._event_seq}",
            "timestamp": now_ms(),
            "sessionId": self._session_id,
        }
        await maybe_await(self._sink(cast(AgentEvent, enriched)))
