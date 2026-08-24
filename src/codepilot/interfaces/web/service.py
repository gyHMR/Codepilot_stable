"""把 HTTP/SSE 用例映射到 Runtime Gateway，不持有业务权威状态。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from codepilot.runtime import RuntimeGateway, SessionOpenIntent
from codepilot.runtime.errors import runtime_error_payload
from codepilot.runtime.actions import (
    ApprovalDecided,
    CommandSubmitted,
    ContinuationRequested,
    PromptSubmitted,
    RunCancelled,
    InteractionResponded,
)
from codepilot.protocols import TextContent, UserMessage
from codepilot.sessions import SessionStateService

from .events import EventHub, WebEvent, runtime_frame_to_event
from .metadata import WebMetadataStore
from .schemas import AcceptedAction
from .workspace import workspace_summary as build_workspace_summary


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
        self._interaction_resolutions: dict[str, str] = {}
        self._approval_resolutions: dict[str, str] = {}
        self._event_capacity = event_capacity
        self._session_open_options = dict(session_open_options or {})
        self._session_states = SessionStateService(self.workspace)
        self._metadata = WebMetadataStore(self.workspace)

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

    async def create_session(self, title: str | None = None) -> dict[str, Any]:
        ref = self.runtime.open_session(
            SessionOpenIntent(workspace_dir=self.workspace, **self._session_open_options)
        )
        self._opened.add(ref.session_id)
        self.events_for(ref.session_id)
        self._metadata.ensure(ref.session_id)
        if title is not None:
            self._metadata.set_title(ref.session_id, title)
        return self.session_detail(ref.session_id)

    async def update_session(self, session_id: str, title: str) -> dict[str, Any]:
        await self.ensure_open(session_id)
        self._metadata.set_title(session_id, title)
        return self.session_detail(session_id)

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
            metadata = self._session_metadata(session_id, state=state)
            wait = self._wait_state(session_id, state=state)
            summaries.append({
                "session_id": session_id,
                "title": self._session_title(session_id, metadata),
                "workspace": str(self.workspace),
                "model_id": state.model.model,
                "permission_mode": "workspace-write",
                "current_mode": state.current_mode,
                "is_running": False,
                "message_count": len(self._session_states.load_messages(session_id)),
                "pending_approvals": [],
                "created_at": state.created_at,
                "updated_at": state.updated_at,
                "status": "waiting" if wait is not None else "idle",
                "wait": wait,
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

    def projection(self, session_id: str) -> dict[str, Any]:
        """Return the authoritative Web state used to recover from lost SSE events."""
        detail = self.session_detail(session_id)
        state = self._state_for(session_id)
        current_run_id = getattr(state, "current_run_id", None) if state is not None else None
        wait = self._wait_state(session_id, state=state)
        if wait is None and not detail.get("is_running"):
            wait = self._recovery_wait_state(session_id, state=state)
        pending = detail.get("pending_approvals", [])
        if wait is not None:
            execution_status = {
                "tool_approval": "waiting_approval",
                "user_input": "waiting_user",
                "plan_confirmation": "waiting_plan",
                "continuation": "waiting_continuation",
            }.get(str(wait.get("kind")), "paused")
        elif pending:
            execution_status = "waiting_approval"
        elif detail.get("is_running"):
            execution_status = "running"
        else:
            execution_status = "idle"
        return {
            "session_revision": int(getattr(state, "revision", 0) or 0),
            "execution": {
                "run_id": current_run_id,
                "status": execution_status,
            },
            "pending_approvals": pending,
            "pending_interaction": (
                wait if wait is not None and wait.get("kind") == "user_input" else None
            ),
            "wait": wait,
            "plan": detail.get("plan"),
        }

    def session_detail(self, session_id: str) -> dict[str, Any]:
        view = self.runtime.describe(session_id)
        status = view.status
        view_state = getattr(view, "state", None) or {}
        state = self._state_for(session_id)
        metadata = self._session_metadata(session_id, state=state)
        pending_approvals = [
            _public_dict(item) for item in view.pending_approvals
        ]
        wait = self._wait_state(session_id, state=state)
        is_running = status.is_running or session_id in self._tasks
        if wait is None and not is_running:
            wait = self._recovery_wait_state(session_id, state=state)
        return {
            "session_id": status.session_id,
            "title": self._session_title(session_id, metadata),
            "workspace": status.workspace,
            "model_id": status.model_id,
            "permission_mode": status.permission_mode,
            "current_mode": status.current_mode,
            "is_running": is_running,
            "message_count": status.message_count,
            "pending_approvals": pending_approvals,
            "wait": wait,
            "plan": view_state.get("current_plan"),
            "created_at": metadata["created_at"],
            "updated_at": metadata["updated_at"],
            "status": (
                "waiting"
                if wait is not None
                else "approval"
                if pending_approvals
                else "running"
                if is_running
                else "idle"
            ),
        }

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        records = self._session_states.load_messages(session_id)
        return [_public_dict(record.message) for record in records]

    def timeline(self, session_id: str) -> list[dict[str, Any]]:
        records = self._session_states.load_messages(session_id)
        items: list[dict[str, Any]] = []
        for record in records:
            message = record.message
            class_name = type(message).__name__.lower()
            public_message = _public_dict(message)
            base = {
                "session_id": record.session_id,
                "run_id": record.run_id,
                "timestamp": record.created_at,
            }
            if isinstance(message, UserMessage):
                items.append({**base, "item_id": record.message_id,
                              "type": "user_message",
                              "data": {"message": public_message}})
                continue
            if "toolresult" in class_name:
                items.append({**base, "item_id": record.message_id,
                              "type": "activity_group",
                              "data": _tool_result_activity(record.message_id, public_message)})
                continue

            activities, final_message = _assistant_timeline_parts(
                record.message_id, public_message
            )
            if activities:
                items.append({**base, "item_id": f"{record.message_id}:activity",
                              "type": "activity_group",
                              "data": {"activities": activities}})
            if final_message is not None:
                items.append({**base, "item_id": record.message_id,
                              "type": "assistant_message",
                              "data": {"message": final_message}})
        return items

    def workspace_summary(self) -> dict[str, Any]:
        return build_workspace_summary(self.workspace)

    async def submit_prompt(
        self, session_id: str, text: str, mode_hint: str | None = None
    ) -> AcceptedAction:
        self._metadata.touch(session_id)
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
        self._approval_resolutions[session_id] = approval_id
        return self._start_dispatch(
            session_id,
            ApprovalDecided(
                approval_id=approval_id, decision=decision, reason=reason
            ),
        )

    async def respond_interaction(
        self, session_id: str, request_id: str, answer: str
    ) -> AcceptedAction:
        await self.ensure_open(session_id)
        wait = self._wait_state(session_id)
        if wait is None or wait.get("kind") != "user_input":
            raise WebNotFound("web.interaction_not_found", "Interaction is not pending")
        if str(wait.get("request_id")) != request_id:
            raise WebConflict(
                "web.interaction_mismatch", "Interaction does not match the active checkpoint"
            )
        self._interaction_resolutions[session_id] = request_id
        return self._start_dispatch(
            session_id,
            InteractionResponded(request_id=request_id, answer=answer.strip()),
        )

    async def continue_run(self, session_id: str, request_id: str) -> AcceptedAction:
        await self.ensure_open(session_id)
        wait = self._wait_state(session_id) or self._recovery_wait_state(session_id)
        if wait is None or wait.get("kind") != "continuation":
            raise WebNotFound("web.continuation_not_found", "Continuation is not pending")
        if str(wait.get("request_id")) != request_id:
            raise WebConflict("web.continuation_mismatch", "Continuation does not match the active checkpoint")
        return self._start_dispatch(session_id, ContinuationRequested(request_id=request_id))

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
        self._metadata.delete(session_id)

    def _state_for(self, session_id: str) -> Any | None:
        return next(
            (
                state
                for state in self._session_states.list_sessions()
                if state.session_id == session_id
            ),
            None,
        )

    def _wait_state(self, session_id: str, *, state: Any | None = None) -> dict[str, Any] | None:
        state = state or self._state_for(session_id)
        run_id = getattr(state, "current_run_id", None) if state is not None else None
        if not run_id:
            return None
        run = self._session_states.get_run(str(run_id))
        checkpoint = getattr(run, "checkpoint", None) if run is not None else None
        waiting = getattr(checkpoint, "waiting", None) if checkpoint is not None else None
        if waiting is None:
            return None
        payload = _public_value(getattr(waiting, "payload", {}) or {})
        return {
            "run_id": str(run_id),
            "kind": str(getattr(waiting, "kind", "")),
            "request_id": str(getattr(waiting, "request_id", "")),
            "payload": payload if isinstance(payload, dict) else {},
        }

    def _recovery_wait_state(
        self,
        session_id: str,
        *,
        state: Any | None = None,
    ) -> dict[str, Any] | None:
        state = state or self._state_for(session_id)
        run_id = getattr(state, "current_run_id", None) if state is not None else None
        if not run_id:
            return None
        run = self._session_states.get_run(str(run_id))
        checkpoint = getattr(run, "checkpoint", None) if run is not None else None
        if checkpoint is None or getattr(checkpoint, "waiting", None) is not None:
            return None
        return {
            "run_id": str(run_id),
            "kind": "continuation",
            "request_id": f"recovery:{run_id}",
            "payload": {"reason": "runtime.recovery_required"},
        }

    def _session_metadata(self, session_id: str, *, state: Any | None) -> dict[str, str]:
        metadata = self._metadata.ensure(session_id)
        if state is not None:
            metadata["created_at"] = state.created_at
            metadata["updated_at"] = state.updated_at
        return metadata

    def _session_title(self, session_id: str, metadata: dict[str, str]) -> str:
        explicit = metadata.get("title", "").strip()
        if explicit:
            return explicit
        for record in self._session_states.load_messages(session_id):
            if not isinstance(record.message, UserMessage):
                continue
            content = record.message.content
            if isinstance(content, str):
                text = content
            else:
                text = "".join(
                    block.text for block in content if isinstance(block, TextContent)
                )
            title = " ".join(text.split())
            if title:
                return title[:42]
        return "新建会话"

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
        registry = self._cancellations if cancellation else self._tasks
        try:
            await self.ensure_open(session_id)
            async for frame in self.runtime.dispatch(session_id, action):
                pending_approval = self._approval_resolutions.get(session_id)
                if pending_approval and str(getattr(frame, "kind", "")) != "failed":
                    sequence = self._sequences.get(session_id, 0) + 1
                    self._sequences[session_id] = sequence
                    await self.events_for(session_id).publish(
                        WebEvent(
                            event_id=uuid4().hex,
                            session_id=session_id,
                            run_id=None,
                            type="approval.resolved",
                            sequence=sequence,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            data={"approval_id": pending_approval},
                        )
                    )
                    self._approval_resolutions.pop(session_id, None)
                pending_interaction = self._interaction_resolutions.get(session_id)
                if pending_interaction and str(getattr(frame, "kind", "")) != "failed":
                    sequence = self._sequences.get(session_id, 0) + 1
                    self._sequences[session_id] = sequence
                    await self.events_for(session_id).publish(
                        WebEvent(
                            event_id=uuid4().hex,
                            session_id=session_id,
                            run_id=None,
                            type="interaction.resolved",
                            sequence=sequence,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            data={"request_id": pending_interaction},
                        )
                    )
                    self._interaction_resolutions.pop(session_id, None)
                sequence = self._sequences.get(session_id, 0) + 1
                self._sequences[session_id] = sequence
                await self.events_for(session_id).publish(
                    runtime_frame_to_event(
                        frame, session_id=session_id, sequence=sequence
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            current = asyncio.current_task()
            if registry.get(session_id) is current:
                registry.pop(session_id, None)
            self._approval_resolutions.pop(session_id, None)
            self._interaction_resolutions.pop(session_id, None)
            try:
                sequence = self._sequences.get(session_id, 0) + 1
                self._sequences[session_id] = sequence
                await self.events_for(session_id).publish(
                    WebEvent(
                        event_id=uuid4().hex,
                        session_id=session_id,
                        run_id=None,
                        type="error",
                        sequence=sequence,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        data=runtime_error_payload(exc),
                    )
                )
                sequence += 1
                self._sequences[session_id] = sequence
                await self.events_for(session_id).publish(
                    WebEvent(
                        event_id=uuid4().hex,
                        session_id=session_id,
                        run_id=None,
                        type="session.snapshot",
                        sequence=sequence,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        data={
                            **self.projection(session_id),
                            "reason": "dispatch_failed",
                        },
                    )
                )
            except Exception:
                # The background task must never leak a second exception to waiters.
                pass
        finally:
            current = asyncio.current_task()
            if registry.get(session_id) is current:
                registry.pop(session_id, None)


def _public_dict(value: Any) -> dict[str, Any]:
    converted = _public_value(value)
    return converted if isinstance(converted, dict) else {"value": converted}


def _public_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _public_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_public_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _public_value(value.model_dump())
    if hasattr(value, "to_dict"):
        return _public_value(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            key: _public_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _message_summary(content: Any) -> str:
    if isinstance(content, str):
        return content[:240]
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return " ".join(parts)[:240]
    return ""


def _assistant_timeline_parts(
    message_id: str, message: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    content = message.get("content")
    if not isinstance(content, (list, tuple)):
        return [], message
    blocks = [item for item in content if isinstance(item, dict)]
    tool_calls = [item for item in blocks if _block_type(item) == "toolcall"]
    thinking = [str(item.get("thinking", "")).strip()
                for item in blocks if _block_type(item) == "thinking"]
    text_blocks = [item for item in blocks if _block_type(item) == "text"]
    activities: list[dict[str, Any]] = []
    if thinking:
        activities.append({
            "activity_id": f"{message_id}:thinking",
            "type": "thinking",
            "name": "思考",
            "status": "completed",
            "summary": "\n\n".join(item for item in thinking if item),
        })
    for index, call in enumerate(tool_calls):
        if str(call.get("name") or "") == "request_user_input":
            continue
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        activities.append({
            "activity_id": str(call.get("id") or f"{message_id}:tool:{index}"),
            "type": "tool_call",
            "name": str(call.get("name") or "工具操作"),
            "status": "running",
            "target": _tool_target(str(call.get("name") or ""), arguments),
            "arguments": arguments,
        })
    if tool_calls and text_blocks:
        activities.insert(0, {
            "activity_id": f"{message_id}:narration",
            "type": "narration",
            "name": "执行说明",
            "status": "completed",
            "summary": _message_summary(text_blocks),
        })
    if tool_calls:
        return activities, None
    visible = dict(message)
    visible["content"] = text_blocks
    return activities, visible if text_blocks else None


def _tool_result_activity(message_id: str, message: dict[str, Any]) -> dict[str, Any]:
    details = message.get("details") if isinstance(message.get("details"), dict) else {}
    name = str(message.get("tool_name") or "工具操作")
    return {
        "activities": [{
            "activity_id": message_id,
            "type": "tool_result",
            "name": name,
            "status": str(message.get("status") or "completed"),
            "target": _tool_target(name, details),
            "summary": _message_summary(message.get("content")),
            "details": details,
        }]
    }


def _block_type(block: dict[str, Any]) -> str:
    return str(block.get("type", "")).replace("_", "").replace("-", "").lower()


def _tool_target(name: str, values: dict[str, Any]) -> str:
    lowered = name.lower()
    keys = ("command", "cmd") if lowered in {"bash", "shell", "command", "exec"} else (
        "path", "file_path", "directory", "workspace", "query", "pattern"
    )
    for key in keys:
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


__all__ = ["WebConflict", "WebNotFound", "WebService", "WebServiceError"]
