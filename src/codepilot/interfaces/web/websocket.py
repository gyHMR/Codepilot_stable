from __future__ import annotations

"""Framework-neutral WebSocket event streaming helpers."""

from dataclasses import dataclass
from typing import AsyncIterator

from codepilot.runtime.service import RuntimeService

from .event_adapter import agent_event_to_web
from .schemas import WebEventEnvelope


@dataclass
class WebSocketSessionStream:
    runtime: RuntimeService
    session_id: str

    async def continue_events(self) -> AsyncIterator[WebEventEnvelope]:
        async for event in self.runtime.continue_session(self.session_id):
            yield agent_event_to_web(event)
