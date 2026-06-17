from __future__ import annotations

"""Application service shared by CLI, Web, and IM interfaces."""

from dataclasses import dataclass
from typing import AsyncIterator

from codepilot.core import AgentEvent, AgentMessage
from codepilot.sessions.session import AgentSession

from .factory import create_agent_session
from .types import CreateAgentSessionOptions


@dataclass(frozen=True)
class CreateSessionRequest:
    options: CreateAgentSessionOptions


@dataclass(frozen=True)
class SessionHandle:
    session_id: str
    session: AgentSession


@dataclass(frozen=True)
class UserInput:
    text: str
    images: list[str] | None = None


class RuntimeService:
    """High-level runtime facade for user-facing interfaces."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}

    def create_session(self, request: CreateSessionRequest | CreateAgentSessionOptions) -> SessionHandle:
        options = request.options if isinstance(request, CreateSessionRequest) else request
        session = create_agent_session(options)
        self._sessions[session.session_id] = session
        return SessionHandle(session_id=session.session_id, session=session)

    def get_session(self, session_id: str) -> AgentSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise ValueError(f"Session not found: {session_id}") from exc

    def list_sessions(self) -> list[dict[str, str]]:
        return [{"session_id": session_id} for session_id in sorted(self._sessions)]

    async def send_message(self, session_id: str, message: UserInput) -> AsyncIterator[AgentEvent]:
        session = self.get_session(session_id)
        queue: list[AgentEvent] = []

        def _listener(event: AgentEvent) -> None:
            queue.append(event)

        unsubscribe = session.subscribe(_listener)
        try:
            await session.prompt(message.text, images=message.images)
        finally:
            unsubscribe()

        for event in queue:
            yield event

    async def continue_session(self, session_id: str) -> AsyncIterator[AgentEvent]:
        session = self.get_session(session_id)
        queue: list[AgentEvent] = []

        def _listener(event: AgentEvent) -> None:
            queue.append(event)

        unsubscribe = session.subscribe(_listener)
        try:
            await session.continue_run()
        finally:
            unsubscribe()

        for event in queue:
            yield event

    async def approve_tool_call(self, approval_id: str, decision: str) -> None:
        _ = approval_id, decision
        raise NotImplementedError("Tool approval flow is not implemented yet")

    def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def close_all(self) -> None:
        for session_id in list(self._sessions):
            self.close_session(session_id)
