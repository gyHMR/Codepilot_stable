from __future__ import annotations

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


class ActiveRunRegistry:
    """Track active run ids without exposing task objects."""

    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def start(self, session_id: str, run_id: str) -> None:
        self._items[session_id] = run_id

    def finish(self, session_id: str) -> str | None:
        return self._items.pop(session_id, None)

    def is_running(self, session_id: str) -> bool:
        return session_id in self._items

    def clear(self) -> None:
        self._items.clear()


__all__ = [
    "ActiveRunRegistry",
    "RuntimeSession",
    "RuntimeSessionStore",
    "RuntimeStatusInfo",
]
