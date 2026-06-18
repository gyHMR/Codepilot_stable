from __future__ import annotations

"""Application service shared by CLI, Web, and IM interfaces."""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, cast

from codepilot.core import AgentEvent
from codepilot.observability import build_run_report
from codepilot.protocols import AgentRunResult
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

    async def run_message(self, session_id: str, message: UserInput) -> AgentRunResult:
        session = self.get_session(session_id)
        return await session.run(message.text, images=message.images)

    async def send_message(self, session_id: str, message: UserInput) -> AsyncIterator[AgentEvent]:
        session = self.get_session(session_id)
        async for event in self._stream_session_events(
            session,
            lambda: session.run(message.text, images=message.images),
        ):
            yield event

    async def continue_session(self, session_id: str) -> AsyncIterator[AgentEvent]:
        session = self.get_session(session_id)
        async for event in self._stream_session_events(session, session.continue_run):
            yield event

    def list_runs(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        session = self.get_session(session_id)
        return session.store.load_run_results(limit=limit)

    def get_run_result(self, session_id: str, run_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        return session.store.run_store.load_run_result(run_id)

    def get_run_events(self, session_id: str, run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        session = self.get_session(session_id)
        return session.store.run_store.load_events(run_id, limit=limit)

    def get_run_report(self, session_id: str, run_id: str) -> dict[str, Any]:
        result = self.get_run_result(session_id, run_id)
        events = self.get_run_events(session_id, run_id)
        return build_run_report(result, events=events)

    async def _stream_session_events(
        self,
        session: AgentSession,
        run: Callable[[], Awaitable[object]],
    ) -> AsyncIterator[AgentEvent]:
        done_marker = object()
        queue: asyncio.Queue[AgentEvent | object] = asyncio.Queue()

        def _listener(event: AgentEvent) -> None:
            queue.put_nowait(event)

        unsubscribe = session.subscribe(_listener)
        task = asyncio.create_task(run())
        task.add_done_callback(lambda _task: queue.put_nowait(done_marker))

        try:
            while True:
                item = await queue.get()
                if item is done_marker:
                    break
                yield cast(AgentEvent, item)
            await task
        finally:
            unsubscribe()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

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
