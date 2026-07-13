from __future__ import annotations

"""Runtime gateway: receive interface actions and stream runtime frames."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from typing import TYPE_CHECKING

from codepilot.core.contracts import AgentLoopOutcome
from codepilot.sessions.contracts import (
    PreparedAgentRun,
    SessionCommandRecord,
    SessionCommandIntent,
    SessionContinuationIntent,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionView,
)
from codepilot.runtime.session_controller import SessionController

from .actions import (
    ApprovalDecided,
    ApprovalRequiredFrame,
    CancelledFrame,
    CommandFinishedFrame,
    CommandSubmitted,
    FailedFrame,
    ProgressFrame,
    PromptSubmitted,
    RunCancelled,
    RunFinishedFrame,
    RunPausedFrame,
    RuntimeFrame,
    UserAction,
)
from .approvals import ApprovalView
from .builder import build_runtime_session
from .opening import AppSessionView, SessionRef
from .sessions import ActiveRunRegistry, RuntimeSession, RuntimeSessionRegistry
from .views import CommandDescriptor, SessionStatus, builtin_commands
from .executor import RunExecutionCompleted, RunExecutionEvent

if TYPE_CHECKING:
    from .opening import SessionOpenIntent


__all__ = ["RuntimeGateway"]


class RuntimeGateway:
    """Application boundary used by CLI, DingTalk, evaluation, and tests."""

    def __init__(
        self,
        *,
        model_port: Any | None = None,
        tool_port: Any | None = None,
    ) -> None:
        self._sessions = RuntimeSessionRegistry()
        self._model_port = model_port
        self._tool_port = tool_port
        self._active_runs = ActiveRunRegistry()
        self._session_locks: dict[str, asyncio.Lock] = {}

    def open_session(self, intent: "SessionOpenIntent") -> SessionRef:
        session = build_runtime_session(intent)
        if self._active_runs.is_running(session.session_id):
            session.controller.close()
            raise RuntimeError(
                f"Session {session.session_id} still stopping its previous run; reopen it after cancellation completes."
            )
        if self._model_port is not None:
            session.model_port = self._model_port
        if self._tool_port is not None:
            session.tool_port = self._tool_port
        self._sessions.add(session)
        return SessionRef(session_id=session.session_id)

    async def dispatch(
        self,
        session_id: str,
        action: UserAction,
    ) -> AsyncIterator[RuntimeFrame]:
        session = self._sessions.require(session_id)
        if isinstance(action, RunCancelled):
            yield self._cancel_run(session_id, action)
            return
        if self._active_runs.is_running(session_id):
            yield self._run_active_frame(session_id)
            return
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            if self._active_runs.is_running(session_id):
                yield self._run_active_frame(session_id)
                return
            try:
                async for frame in self._dispatch_action(session, action):
                    yield frame
            except Exception as exc:
                yield FailedFrame(error=_runtime_error_payload(exc))

    async def _dispatch_action(
        self,
        session: RuntimeSession,
        action: UserAction,
    ) -> AsyncIterator[RuntimeFrame]:
        if isinstance(action, PromptSubmitted):
            await self._ensure_mcp_ready(session)
            checkpoint = session.controller.runtime_checkpoint()
            checkpoint_run_id = (
                _optional_text(checkpoint.get("run_id"))
                if isinstance(checkpoint, dict)
                else None
            )
            if checkpoint_run_id is not None:
                waiting_kind = (
                    _optional_text(checkpoint.get("waiting_kind"))
                    if isinstance(checkpoint, dict)
                    else None
                )
                if waiting_kind == "tool_approval":
                    yield FailedFrame(
                        error={
                            "code": "runtime.waiting_approval",
                            "message": "Resolve the pending tool approval before continuing.",
                            "run_id": checkpoint_run_id,
                        }
                    )
                    return
                if _is_plan_wait_checkpoint(checkpoint):
                    plan_command = _explicit_plan_command(session, action.text)
                    if plan_command is not None:
                        record = await self._run_command(
                            session,
                            CommandSubmitted(plan_command),
                        )
                        yield CommandFinishedFrame(record=record)
                        async for frame in self._follow_up_from_command(session, record):
                            yield frame
                        return
                phase = _optional_text(checkpoint.get("phase"))
                continuation_kind = (
                    "plan_clarification"
                    if phase == "plan_clarification"
                    else "automatic_continuation"
                    if phase == "plan_incomplete"
                    else "plan_feedback"
                    if _is_plan_wait_checkpoint(checkpoint)
                    else "automatic_continuation"
                )
                async for frame in self._run_continuation(
                    session,
                    SessionContinuationIntent(
                        kind=continuation_kind,
                        run_id=checkpoint_run_id,
                        text=action.text if waiting_kind is not None else "",
                    ),
                ):
                    yield frame
                return
            async for frame in self._run_prompt(session, action):
                yield frame
            return
        if isinstance(action, CommandSubmitted):
            if action.text.strip().partition(" ")[0] == "/tools":
                await self._ensure_mcp_ready(session)
            plan_command = _explicit_plan_command(session, action.text)
            record = await self._run_command(
                session,
                CommandSubmitted(plan_command or action.text),
            )
            yield CommandFinishedFrame(record=record)
            async for frame in self._follow_up_from_command(session, record):
                yield frame
            return
        if isinstance(action, ApprovalDecided):
            async for frame in self._resume_after_approval(session, action):
                yield frame
            return
        yield FailedFrame(error={"code": "runtime.unknown_action", "action": type(action).__name__})

    @staticmethod
    def _run_active_frame(session_id: str) -> FailedFrame:
        return FailedFrame(
            error={
                "code": "runtime.run_active",
                "message": "Wait for the active run to pause or cancel it before sending another action.",
                "session_id": session_id,
            }
        )

    def describe(self, session_id: str) -> AppSessionView:
        session = self._sessions.require(session_id)
        view = session.controller.describe()
        return AppSessionView(
            session=view,
            status=self._status_for(session, view),
            state=dict(view.context),
            commands=tuple(self._commands_for(session)),
            pending_approvals=tuple(self._pending_approvals_for(session)),
        )

    def close(self, session_id: str) -> None:
        active_run_id = self._active_runs.cancel(session_id, "session_closed")
        closed = self._sessions.close(session_id)
        if (
            closed is not None
            and closed.mcp_manager is not None
            and all(
                item.mcp_manager is not closed.mcp_manager
                for item in self._sessions.values()
            )
        ):
            _schedule_close(closed.mcp_manager)
        if active_run_id is None or not self._active_runs.has_active_tasks(session_id):
            self._active_runs.finish(session_id)
        self._session_locks.pop(session_id, None)

    async def close_all(self) -> None:
        scopes = self._active_runs.cancel_all()
        if scopes:
            await asyncio.gather(*(scope.release() for scope in scopes))
            self._active_runs.clear()
        managers = {
            id(session.mcp_manager): session.mcp_manager
            for session in self._sessions.values()
            if session.mcp_manager is not None
        }
        self._sessions.close_all()
        if managers:
            await asyncio.gather(
                *(manager.aclose() for manager in managers.values()),
                return_exceptions=True,
            )
        self._session_locks.clear()

    def _require_session(self, session_id: str) -> SessionController:
        return self._sessions.require(session_id).controller

    async def _run_prompt(
        self,
        session: RuntimeSession,
        action: PromptSubmitted,
    ) -> AsyncIterator[RuntimeFrame]:
        prepared = await session.controller.runs.prepare(
            SessionRunIntent(
                text=action.text,
                images=action.images,
                mode_hint=action.mode_hint,
            ),
            model=session.controller.model,
        )
        async for frame in self._run_agent_loop(
            session,
            prepared,
        ):
            yield frame

    async def _run_command(
        self,
        session: RuntimeSession,
        action: CommandSubmitted,
    ) -> SessionCommandRecord:
        record = await session.controller.apply_command(
            SessionCommandIntent(
                text=action.text,
                tool_catalog=tuple(self._tool_catalog_for(session.session_id)),
            )
        )
        if record.switched_session_id:
            self._register_derived_controller(
                source_session_id=session.session_id,
                new_session_id=record.switched_session_id,
                controller=session.controller,
            )
        return record

    async def _follow_up_from_command(
        self,
        session: RuntimeSession,
        record: SessionCommandRecord,
    ) -> AsyncIterator[RuntimeFrame]:
        prompt = _optional_text(record.data.get("prompt"))
        if prompt is not None:
            async for frame in self._run_prompt(session, PromptSubmitted(prompt)):
                yield frame
            return
        kind = _optional_text(record.data.get("continuation_kind"))
        run_id = _optional_text(record.data.get("continuation_run_id"))
        if kind is None or run_id is None:
            return
        async for frame in self._run_continuation(
            session,
            SessionContinuationIntent(
                kind=kind,  # type: ignore[arg-type]
                run_id=run_id,
                target_mode=_optional_text(record.data.get("current_mode")),
            ),
        ):
            yield frame

    async def _run_continuation(
        self,
        session: RuntimeSession,
        intent: SessionContinuationIntent,
    ) -> AsyncIterator[RuntimeFrame]:
        prepared = await session.controller.runs.prepare(
            intent,
            model=session.controller.model,
        )
        async for frame in self._run_agent_loop(session, prepared):
            yield frame

    async def _resume_after_approval(
        self,
        session: RuntimeSession,
        action: ApprovalDecided,
    ) -> AsyncIterator[RuntimeFrame]:
        tool_port = session.tool_port or self._tool_port
        challenge = (
            tool_port.approval_challenge(action.approval_id)
            if tool_port is not None
            else None
        )
        if challenge is None:
            yield FailedFrame(
                error={
                    "code": "runtime.approval_not_found",
                    "approval_id": action.approval_id,
                }
            )
            return
        if challenge.session_id != session.session_id:
            yield FailedFrame(
                error={
                    "code": "runtime.approval_session_mismatch",
                    "approval_id": action.approval_id,
                    "session_id": session.session_id,
                }
            )
            return

        prepared = await session.controller.runs.prepare(
            SessionResumeIntent(
                approval_id=action.approval_id,
                decision=action.decision,
                reason=action.reason,
                run_id=challenge.run_id,
            ),
            model=session.controller.model,
        )
        async for frame in self._run_agent_loop(
            session,
            prepared,
        ):
            yield frame

    def _cancel_run(self, session_id: str, action: RunCancelled) -> CancelledFrame:
        active = self._active_runs.cancel(session_id, action.reason)
        return CancelledFrame(
            session_id=session_id,
            cancelled=active is not None,
            reason=action.reason,
        )

    async def _run_agent_loop(
        self,
        session: RuntimeSession,
        prepared: PreparedAgentRun,
    ) -> AsyncIterator[RuntimeFrame]:
        environment = session.controller.runs.create_environment(
            prepared,
            model=session.model_port or self._model_port,
            tools=session.tool_port or self._tool_port,
        )
        self._active_runs.start(
            session.session_id,
            prepared.run_id,
            environment.resources,
        )
        outcome: AgentLoopOutcome | None = None
        try:
            async for update in session.controller.runs.execute(environment, prepared):
                if isinstance(update, RunExecutionEvent):
                    yield ProgressFrame(event=update.event)
                elif isinstance(update, RunExecutionCompleted):
                    outcome = update.outcome
        finally:
            self._active_runs.finish(session.session_id, run_id=prepared.run_id)

        if outcome is None:
            return

        try:
            record = await session.controller.runs.commit(prepared, outcome)
        except Exception as exc:
            yield FailedFrame(error=_runtime_error_payload(exc))
            return
        if outcome.status == "failed":
            yield FailedFrame(error=_runtime_error_payload(outcome.error))
            return

        async for frame in self._frames_from_outcome(session.controller, outcome, record):
            yield frame

    async def _frames_from_outcome(
        self,
        controller: SessionController,
        outcome: AgentLoopOutcome,
        record: SessionRunRecord,
    ) -> AsyncIterator[RuntimeFrame]:
        if outcome.status == "waiting_approval":
            for interruption in outcome.interruptions:
                yield ApprovalRequiredFrame(approval=interruption)
            yield RunPausedFrame(
                record=record,
                checkpoint=controller.runtime_checkpoint() or {},
            )
            return
        if outcome.status == "waiting_user":
            yield RunPausedFrame(
                record=record,
                checkpoint=controller.runtime_checkpoint() or {},
            )
            return
        if outcome.status == "failed":
            yield FailedFrame(error=_runtime_error_payload(outcome.error))
            return
        if outcome.status == "cancelled":
            yield CancelledFrame(
                session_id=controller.session_id,
                cancelled=True,
                reason=outcome.stop_reason,
            )
            return
        yield RunFinishedFrame(record=record)

    def _tool_catalog_for(self, session_id: str) -> list[Any]:
        session = self._sessions.require(session_id)
        tool_port = session.tool_port or self._tool_port
        if tool_port is None:
            return []
        mode = session.controller.describe().current_mode
        snapshot = tool_port.catalog_snapshot(
            mode="plan" if mode in {"read", "plan"} else "execute"
        )
        return [entry.spec for entry in snapshot.entries]

    def _pending_approvals_for(self, session: RuntimeSession) -> list[ApprovalView]:
        tool_port = session.tool_port or self._tool_port
        challenges = tuple(tool_port.pending_challenges()) if tool_port is not None else ()
        views = [
            ApprovalView(
                approval_id=challenge.approval_id,
                session_id=challenge.session_id,
                run_id=challenge.run_id,
                tool_call_id=challenge.tool_call_id,
                tool_name=challenge.tool_name,
                reason=challenge.reason,
                risk_level=challenge.risk,
            )
            for challenge in challenges
            if challenge.session_id == session.session_id
        ]
        return sorted(views, key=lambda item: item.approval_id)

    async def _ensure_mcp_ready(self, session: RuntimeSession) -> None:
        manager = session.mcp_manager
        if manager is None:
            return
        tool_port = session.tool_port or self._tool_port
        registry = getattr(tool_port, "registry", None)
        if registry is None:
            raise RuntimeError("MCP discovery requires the canonical ToolRuntime registry")
        await manager.ensure_ready(registry)

    def _register_derived_controller(
        self,
        *,
        source_session_id: str,
        new_session_id: str,
        controller: SessionController,
    ) -> None:
        derived = controller.claim_derived_controller(new_session_id)
        if derived is None:
            return
        self._sessions.derive(
            source_session_id=source_session_id,
            controller=derived,
        )

    def _status_for(self, session: RuntimeSession, view: SessionView) -> SessionStatus:
        info = session.status
        leaf_id = str((view.context or {}).get("leaf_id") or "N/A")
        if info is None:
            model = session.controller.model
            model_id = f"{model.provider}/{model.model_id}"
            return SessionStatus(
                session_id=view.session_id,
                model_id=model_id,
                workspace=".",
                permission_mode="workspace-write",
                message_count=view.message_count,
                leaf_id=leaf_id,
                current_mode=view.current_mode,  # type: ignore[arg-type]
                is_running=self._active_runs.is_running(session.session_id),
                credential_source="unknown",
                plan_summary=_plan_summary_from_view(view),
            )
        return SessionStatus(
            session_id=view.session_id,
            model_id=info.model_id,
            workspace=info.workspace,
            permission_mode=info.permission_mode,
            message_count=view.message_count,
            leaf_id=leaf_id,
            current_mode=view.current_mode,  # type: ignore[arg-type]
            is_running=self._active_runs.is_running(session.session_id),
            credential_source=info.credential_source,
            warnings=tuple(
                dict.fromkeys(
                    [
                        *info.warnings,
                        *(
                            session.mcp_manager.diagnostics
                            if session.mcp_manager is not None
                            else ()
                        ),
                    ]
                )
            ),
            plan_summary=_plan_summary_from_view(view),
        )

    def _commands_for(self, session: RuntimeSession) -> list[CommandDescriptor]:
        commands = list(builtin_commands())
        for command in (session.commands or {}).values():
            name = getattr(command, "name", "")
            description = getattr(command, "description", None) or ""
            source = getattr(command, "source", "extension")
            if not name or not description:
                continue
            commands.append(
                CommandDescriptor(
                    name=str(name),
                    description=str(description),
                    source=source,  # type: ignore[arg-type]
                )
            )
        return commands


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _schedule_close(manager: object) -> None:
    async def close() -> None:
        await manager.aclose()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(close())
    else:
        loop.create_task(close())


def _is_plan_wait_checkpoint(checkpoint: object) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    phase = _optional_text(checkpoint.get("phase"))
    return phase in {
        "plan_approval",
        "plan_feedback",
        "plan_clarification",
        "plan_incomplete",
    }


def _explicit_plan_command(session: RuntimeSession, text: str) -> str | None:
    normalized = _normalize_plan_command_text(text)
    plan = (session.controller.describe().context or {}).get("plan_summary")
    if not isinstance(plan, dict):
        return None
    if plan.get("status") != "proposed":
        return None
    if normalized in {"/approve", "批准", "同意执行", "执行这个方案"}:
        return "/plan approve"
    if normalized in {"/reject", "拒绝", "不要这个方案"}:
        return "/plan reject"
    if normalized in {"清除计划", "取消计划"}:
        return "/plan clear"
    return None


def _normalize_plan_command_text(text: str) -> str:
    return " ".join(text.strip().lower().strip("。.!！?？").split())


def _plan_summary_from_view(view: SessionView) -> dict[str, object] | None:
    value = (view.context or {}).get("plan_summary")
    return dict(value) if isinstance(value, dict) else None


def _runtime_error_payload(error: Any) -> dict[str, Any]:
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        details = error.get("details")
        return {
            "code": code if isinstance(code, str) and code else "runtime.dispatch_failed",
            "message": message if isinstance(message, str) and message else "Runtime dispatch failed",
            "details": details if isinstance(details, dict) else {},
        }
    message = str(error)
    details: dict[str, Any] = {}
    error_info = getattr(error, "error", None)
    if error_info is not None:
        message = getattr(error_info, "message", message)
        details["cause_code"] = getattr(error_info, "code", "")
    if hasattr(error, "run_id"):
        details["run_id"] = getattr(error, "run_id")
    if hasattr(error, "status"):
        details["status"] = getattr(error, "status")
    return {
        "code": "runtime.dispatch_failed",
        "message": message,
        "details": details,
    }
