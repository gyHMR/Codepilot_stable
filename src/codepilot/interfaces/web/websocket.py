from __future__ import annotations

"""框架无关的 WebSocket 事件流辅助工具。"""

from dataclasses import dataclass
from typing import AsyncIterator

from codepilot.runtime.service import RuntimeService

from .event_adapter import agent_event_to_web
from .schemas import WebEventEnvelope


@dataclass
class WebSocketSessionStream:
    """WebSocket 会话事件流：将 RuntimeService 的事件转换为 WebEventEnvelope 异步迭代器。"""
    runtime: RuntimeService
    session_id: str

    def __post_init__(self) -> None:
        session_id = str(self.session_id).strip() if self.session_id is not None else ""
        if not session_id:
            raise ValueError("Web session_id cannot be empty")
        self.session_id = session_id

    async def continue_events(self) -> AsyncIterator[WebEventEnvelope]:
        """继续会话并将 Agent 事件转换为 Web 事件信封逐个 yield。"""
        async for event in self.runtime.continue_session(self.session_id):
            yield agent_event_to_web(event)
