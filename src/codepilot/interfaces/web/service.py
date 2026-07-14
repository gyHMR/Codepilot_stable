"""把 HTTP/SSE 用例映射到 Runtime Gateway，不持有业务权威状态。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from codepilot.runtime import RuntimeGateway, SessionOpenIntent
from codepilot.runtime.actions import (
    ApprovalDecided,
    CommandSubmitted,
    PromptSubmitted,
    RunCancelled,
)
from codepilot.sessions import SessionStateService

from .events import EventHub, runtime_frame_to_event
from .schemas import AcceptedAction


class WebServiceError(RuntimeError):
    """Web 用例映射失败的基础异常。"""
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class WebConflict(WebServiceError):
    """请求与当前 Session/Run 状态冲突。"""
    pass


class WebNotFound(WebServiceError):
    """请求的 Session、Run 或审批不存在。"""
    pass


class WebService:
    """面向路由的 Runtime Gateway 用例门面。"""
    def __init__(
        self,
        *,
        runtime: RuntimeGateway,
        workspace: Path,
        event_capacity: int = 256,
        session_open_options: dict[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.workspace = Path(workspace).resolve()
        self._opened: set[str] = set()
        self._hubs: dict[str, EventHub] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancellations: dict[str, asyncio.Task[None]] = {}
        self._sequences: dict[str, int] = {}
        self._event_capacity = event_capacity
        self._session_open_options = dict(session_open_options or {})
        self._session_states = SessionStateService(self.workspace)

    async def ensure_open(self, session_id: str) -> str:
        if session_id in self._opened:
            return session_id
        ref = self.runtime.open_session(
            SessionOpenIntent(
                workspace_dir=self.workspace,
                session_id=session_id,
                **self._session_open_options,
            )
        )
        actual = ref.session_id
        view = self.runtime.describe(actual)
        actual_workspace = Path(view.status.workspace).resolve()
        if actual_workspace != self.workspace:
            self.runtime.close(actual)
            raise WebConflict(
                "web.workspace_mismatch",
                "Session belongs to a different workspace",
            )
        self._opened.add(actual)
        self.events_for(actual)
        return actual

    async def create_session(self) -> dict[str, Any]:
        ref = self.runtime.open_session(
            SessionOpenIntent(workspace_dir=self.workspace, **self._session_open_options)
        )
        self._opened.add(ref.session_id)
        self.events_for(ref.session_id)
        return self.session_detail(ref.session_id)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        await self.ensure_open(session_id)
        return self.session_detail(session_id)

    async def list_sessions(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for state in self._session_states.list_sessions():
            session_id = state.session_id
            seen.add(session_id)
            if session_id in self._opened:
                summaries.append(self.session_detail(session_id))
                continue
            summaries.append({
                "session_id": session_id,
                "workspace": str(self.workspace),
                "model_id": state.model.model,
                "permission_mode": "workspace-write",
                "current_mode": state.current_mode,
                "is_running": False,
                "message_count": len(self._session_states.load_messages(session_id)),
                "pending_approvals": [],
                "created_at": state.created_at,
                "updated_at": state.updated_at,
            })
        summaries.extend(
            self.session_detail(session_id)
            for session_id in sorted(self._opened - seen)
        )
        return summaries

    def events_for(self, session_id: str) -> EventHub:
        return self._hubs.setdefault(
            session_id, EventHub(capacity=self._event_capacity)
        )

    def session_detail(self, session_id: str) -> dict[str, Any]:
        view = self.runtime.describe(session_id)
        status = view.status
        return {
            "session_id": status.session_id,
            "workspace": status.workspace,
            "model_id": status.model_id,
            "permission_mode": status.permission_mode,
            "current_mode": status.current_mode,
            "is_running": status.is_running or session_id in self._tasks,
            "message_count": status.message_count,
            "pending_approvals": [
                _public_dict(item) for item in view.pending_approvals
            ],
        }

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        records = self._session_states.load_messages(session_id)
        return [_public_dict(record.message) for record in records]

    async def submit_prompt(
        self, session_id: str, text: str, mode_hint: str | None = None
    ) -> AcceptedAction:
        return self._start_dispatch(
            session_id, PromptSubmitted(text=text, mode_hint=mode_hint)
        )

    async def submit_command(self, session_id: str, text: str) -> AcceptedAction:
        return self._start_dispatch(session_id, CommandSubmitted(text=text))

    async def decide_approval(
        self,
        session_id: str,
        approval_id: str,
        decision: str,
        reason: str = "",
    ) -> AcceptedAction:
        await self.ensure_open(session_id)
        pending = self.runtime.describe(session_id).pending_approvals
        if approval_id not in {item.approval_id for item in pending}:
            raise WebNotFound("web.approval_not_found", "Approval is not pending")
        return self._start_dispatch(
            session_id,
            ApprovalDecided(
                approval_id=approval_id, decision=decision, reason=reason
            ),
        )

    async def cancel(self, session_id: str) -> AcceptedAction:
        await self.ensure_open(session_id)
        pending = self._cancellations.get(session_id)
        if pending is not None and not pending.done():
            return AcceptedAction(session_id=session_id)
        task = asyncio.create_task(
            self._consume_action(session_id, RunCancelled(reason="user"), cancellation=True)
        )
        self._cancellations[session_id] = task
        return AcceptedAction(session_id=session_id)

    async def delete_session(self, session_id: str) -> None:
        if session_id in self._tasks:
            raise WebConflict("runtime.run_active", "Cancel the active run before deletion")
        if session_id in self._opened:
            self.runtime.close(session_id)
            self._opened.discard(session_id)
        self._hubs.pop(session_id, None)
        if not self._session_states.delete_session(session_id):
            raise WebNotFound("web.session_not_found", "Session not found")

    async def wait_for_idle(self, session_id: str) -> None:
        task = self._tasks.get(session_id)
        if task is not None:
            await task
        cancellation = self._cancellations.get(session_id)
        if cancellation is not None:
            await cancellation

    async def shutdown(self) -> None:
        tasks = tuple(self._tasks.values())
        tasks += tuple(self._cancellations.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._cancellations.clear()
        await self.runtime.close_all()

    def _start_dispatch(
        self, session_id: str, action: Any, *, allow_active: bool = False
    ) -> AcceptedAction:
        active = self._tasks.get(session_id)
        if active is not None and not active.done():
            if not allow_active:
                raise WebConflict(
                    "runtime.run_active", "A run is already active for this session"
                )
            raise WebConflict(
                "runtime.cancellation_pending", "Cancellation is already pending"
            )
        task = asyncio.create_task(self._consume_dispatch(session_id, action))
        self._tasks[session_id] = task
        return AcceptedAction(session_id=session_id)

    async def _consume_dispatch(self, session_id: str, action: Any) -> None:
        await self._consume_action(session_id, action)

    async def _consume_action(
        self, session_id: str, action: Any, *, cancellation: bool = False
    ) -> None:
        try:
            await self.ensure_open(session_id)
            async for frame in self.runtime.dispatch(session_id, action):
                sequence = self._sequences.get(session_id, 0) + 1
                self._sequences[session_id] = sequence
                await self.events_for(session_id).publish(
                    runtime_frame_to_event(
                        frame, session_id=session_id, sequence=sequence
                    )
                )
        finally:
            current = asyncio.current_task()
            registry = self._cancellations if cancellation else self._tasks
            if registry.get(session_id) is current:
                registry.pop(session_id, None)


def _public_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {"value": str(value)}


__all__ = ["WebConflict", "WebNotFound", "WebService", "WebServiceError"]
