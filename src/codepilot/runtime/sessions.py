from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

from codepilot.sessions.controller import SessionController

from .config import RuntimePermissionMode

@dataclass
class RuntimeStatusInfo:
    """Small status payload kept with an opened runtime session."""

    session_id: str
    model_id: str
    workspace: str
    permission_mode: RuntimePermissionMode
    credential_source: str = "unknown"
    warnings: tuple[str, ...] = ()


@dataclass
class RuntimeSession:
    """The live runtime objects needed to dispatch actions for one session."""

    controller: SessionController
    model_port: Any | None = None
    tool_port: Any | None = None
    status: RuntimeStatusInfo | None = None
    commands: dict[str, Any] | None = None

    @property
    def session_id(self) -> str:
        return self.controller.session_id


class RuntimeSessionStore:
    def __init__(self) -> None:
        self._items: dict[str, RuntimeSession] = {}

    def add(self, session: RuntimeSession) -> None:
        self._items[session.session_id] = session

    def require(self, session_id: str) -> RuntimeSession:
        try:
            return self._items[session_id]
        except KeyError as exc:
            raise KeyError(f"Session not found: {session_id}") from exc

    def close(self, session_id: str) -> None:
        entry = self._items.pop(session_id, None)
        if entry is not None:
            entry.controller.close()

    def close_all(self) -> None:
        for session_id in list(self._items):
            self.close(session_id)

    def derive(
        self,
        *,
        source_session_id: str,
        controller: SessionController,
    ) -> RuntimeSession | None:
        source = self.require(source_session_id)
        status = source.status
        if status is not None:
            status = replace(
                status,
                session_id=controller.session_id,
            )
        session = RuntimeSession(
            controller=controller,
            model_port=source.model_port,
            tool_port=source.tool_port,
            status=status,
            commands=dict(source.commands or {}),
        )
        self.add(session)
        return session


@dataclass
class ActiveRun:
    run_id: str
    task: asyncio.Task[Any] | None = None


class ActiveRunRegistry:
    """Track active run ids and the asyncio task that drives them."""

    def __init__(self) -> None:
        self._items: dict[str, ActiveRun] = {}

    def start(self, session_id: str, run_id: str) -> None:
        self._items[session_id] = ActiveRun(run_id=run_id)

    def attach_task(self, session_id: str, task: asyncio.Task[Any]) -> None:
        active = self._items.get(session_id)
        if active is not None:
            active.task = task

    def finish(self, session_id: str, *, run_id: str | None = None) -> str | None:
        active = self._items.pop(session_id, None)
        if active is not None and run_id is not None and active.run_id != run_id:
            self._items[session_id] = active
            return None
        return active.run_id if active is not None else None

    def cancel(self, session_id: str) -> str | None:
        active = self._items.get(session_id)
        if active is None:
            return None
        if active.task is not None and not active.task.done():
            active.task.cancel()
        return active.run_id

    def is_running(self, session_id: str) -> bool:
        return session_id in self._items

    def has_attached_task(self, session_id: str) -> bool:
        active = self._items.get(session_id)
        return active is not None and active.task is not None

    def clear(self) -> None:
        self._items.clear()

    def cancel_all(self) -> None:
        for session_id in tuple(self._items):
            self.cancel(session_id)


__all__ = [
    "ActiveRunRegistry",
    "RuntimeSession",
    "RuntimeSessionStore",
    "RuntimeStatusInfo",
]
