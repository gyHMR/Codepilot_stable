from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codepilot.sessions.controller import SessionController


@dataclass
class RuntimeSessionEntry:
    controller: SessionController
    assembly: Any | None = None
    model_port: Any | None = None
    tool_port: Any | None = None


class RuntimeSessionRegistry:
    def __init__(self) -> None:
        self._items: dict[str, RuntimeSessionEntry] = {}

    def add(
        self,
        session_id: str,
        controller: SessionController,
        *,
        assembly: Any | None = None,
        model_port: Any | None = None,
        tool_port: Any | None = None,
    ) -> None:
        self._items[session_id] = RuntimeSessionEntry(
            controller=controller,
            assembly=assembly,
            model_port=model_port,
            tool_port=tool_port,
        )

    def require(self, session_id: str) -> RuntimeSessionEntry:
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
    "RuntimeSessionEntry",
    "RuntimeSessionRegistry",
]
