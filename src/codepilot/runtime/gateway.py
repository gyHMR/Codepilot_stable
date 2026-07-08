from __future__ import annotations

"""Runtime gateway: receive interface actions and stream runtime frames."""

import asyncio
from collections.abc import AsyncIterator, Awaitable
from typing import Any, Callable
from typing import TYPE_CHECKING

from codepilot.core.contracts import AgentLoopOutcome, AgentLoopPorts, ContextPort
from codepilot.core.runner import resume_agent_loop, run_agent_loop
from codepilot.sessions.contracts import (
    PreparedAgentRun,
    SessionCommandIntent,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionView,
)
from codepilot.sessions.controller import SessionController
from codepilot.protocols import RunSignalsSummary
from codepilot.tools.contracts import ToolCatalogView

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
    RuntimeFrame,
    UserAction,
)
from .approvals import ApprovalRegistry, ApprovalView
from .builder import build_runtime_session
from .opening import AppSessionView, SessionRef
from .sessions import ActiveRunRegistry, RuntimeSession, RuntimeSessionStore
from .views import CommandDescriptor, SessionStatus, builtin_commands

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
        self._sessions = RuntimeSessionStore()
        self._model_port = model_port
        self._tool_port = tool_port
        self._approvals = ApprovalRegistry()
        self._active_runs = ActiveRunRegistry()

    def open_session(self, intent: "SessionOpenIntent") -> SessionRef:
        session = build_runtime_session(intent)
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
        if isinstance(action, PromptSubmitted):
            plan_command = self._plan_command_from_prompt(session, action)
            if plan_command is not None:
                record = await self._run_command(session, CommandSubmitted(plan_command))
                yield CommandFinishedFrame(record=record)
                async for frame in self._follow_up_from_command(session, record):
                    yield frame
                return
            async for frame in self._run_prompt(session, action):
                yield frame
            return
        if isinstance(action, CommandSubmitted):
            record = await self._run_command(session, action)
            yield CommandFinishedFrame(record=record)
            async for frame in self._follow_up_from_command(session, record):
                yield frame
            return
        if isinstance(action, ApprovalDecided):
            async for frame in self._resume_after_approval(session, action):
                yield frame
            return
        if isinstance(action, RunCancelled):
            yield self._cancel_run(session_id, action)
            return
        yield FailedFrame(error={"code": "runtime.unknown_action", "action": type(action).__name__})

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
        self._sessions.close(session_id)
        self._approvals.remove_session(session_id)
        self._active_runs.finish(session_id)

    async def close_all(self) -> None:
        self._sessions.close_all()
        self._approvals.clear()
        self._active_runs.clear()

    def _require_session(self, session_id: str) -> SessionController:
        return self._sessions.require(session_id).controller

    async def _run_prompt(
        self,
        session: RuntimeSession,
        action: PromptSubmitted,
    ) -> AsyncIterator[RuntimeFrame]:
        prepared = await session.controller.prepare_run(
            SessionRunIntent(
                text=action.text,
                images=action.images,
                mode_hint=action.mode_hint,
            )
        )
        async for frame in self._run_agent_loop(
            session,
            prepared,
            lambda ports: run_agent_loop(prepared.loop_input, ports),
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

    def _plan_command_from_prompt(
        self,
        session: RuntimeSession,
        action: PromptSubmitted,
    ) -> str | None:
        if session.controller.pending_approvals():
            return None
        view = session.controller.describe()
        plan = (view.context or {}).get("plan_summary")
        decision = _plan_decision_alias(action.text)
        if decision is None or not isinstance(plan, dict):
            return None
        if plan.get("status") != "proposed":
            return None
        if plan.get("approval_state") != "proposed":
            return None
        return "/plan approve" if decision == "approve" else "/plan reject"

    async def _follow_up_from_command(
        self,
        session: RuntimeSession,
        record: SessionCommandRecord,
    ) -> AsyncIterator[RuntimeFrame]:
        prompt = _optional_text(record.data.get("followup_prompt"))
        if prompt is None:
            return
        mode_hint = _optional_text(record.data.get("followup_mode"))
        async for frame in self._run_prompt(
            session,
            PromptSubmitted(text=prompt, mode_hint=mode_hint),
        ):
            yield frame

    async def _resume_after_approval(
        self,
        session: RuntimeSession,
        action: ApprovalDecided,
    ) -> AsyncIterator[RuntimeFrame]:
        transaction = self._approvals.get(action.approval_id)
        checkpoint_approval = session.controller.pending_approval(action.approval_id)
        if transaction is None and checkpoint_approval is None:
            yield FailedFrame(
                error={
                    "code": "runtime.approval_not_found",
                    "approval_id": action.approval_id,
                }
            )
            return
        if transaction is not None and transaction.session_id != session.session_id:
            yield FailedFrame(
                error={
                    "code": "runtime.approval_session_mismatch",
                    "approval_id": action.approval_id,
                    "session_id": session.session_id,
                }
            )
            return

        prepared = await session.controller.prepare_resume(
            SessionResumeIntent(
                approval_id=action.approval_id,
                decision=action.decision,
                reason=action.reason,
            )
        )
        if prepared.resume_input is None:
            yield FailedFrame(
                error={
                    "code": "runtime.missing_resume_input",
                    "approval_id": action.approval_id,
                }
            )
            return

        async for frame in self._run_agent_loop(
            session,
            prepared,
            lambda ports: resume_agent_loop(prepared.resume_input, ports),
        ):
            yield frame
        self._approvals.pop(action.approval_id)

    def _cancel_run(self, session_id: str, action: RunCancelled) -> CancelledFrame:
        active = self._active_runs.cancel(session_id)
        return CancelledFrame(
            session_id=session_id,
            cancelled=active is not None,
            reason=action.reason,
        )

    async def _run_agent_loop(
        self,
        session: RuntimeSession,
        prepared: PreparedAgentRun,
        run_loop: Callable[[AgentLoopPorts], Awaitable[AgentLoopOutcome]],
    ) -> AsyncIterator[RuntimeFrame]:
        self._active_runs.start(session.session_id, prepared.run_id)
        event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def event_sink(event: dict[str, Any]) -> None:
            payload = dict(event)
            session.controller.record_event(payload)
            event_queue.put_nowait(payload)

        try:
            ports = self._ports_for(
                session.session_id,
                context_port=prepared.context_port,
                event_sink=event_sink,
            )
            task = asyncio.create_task(run_loop(ports))
            self._active_runs.attach_task(session.session_id, task)
            while not task.done() or not event_queue.empty():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                yield ProgressFrame(event=event)
            outcome = await task
            record = await session.controller.commit_run(prepared, outcome)
        except asyncio.CancelledError:
            outcome = AgentLoopOutcome(
                run_id=prepared.run_id,
                status="aborted",
                stop_reason="aborted",
                signals=RunSignalsSummary(cancelled=True),
                error={
                    "code": "runtime.cancelled",
                    "message": "Run cancelled by user",
                },
            )
            record = await session.controller.commit_run(prepared, outcome)
        except Exception as exc:
            error = _runtime_error_payload(exc)
            outcome = AgentLoopOutcome(
                run_id=prepared.run_id,
                status="failed",
                stop_reason="internal_error",
                error=error,
            )
            try:
                await session.controller.commit_run(prepared, outcome)
            except Exception as commit_exc:
                details = dict(error.get("details") or {})
                details["commit_error"] = str(commit_exc)
                error["details"] = details
            yield FailedFrame(error=error)
            return
        finally:
            self._active_runs.finish(session.session_id)

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
                self._approvals.add(controller.session_id, interruption)
                yield ApprovalRequiredFrame(approval=interruption)
            return
        if outcome.status == "failed":
            yield FailedFrame(error=_runtime_error_payload(outcome.error))
            return
        yield RunFinishedFrame(record=record)

    def _ports_for(
        self,
        session_id: str,
        *,
        context_port: ContextPort | None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentLoopPorts:
        session = self._sessions.require(session_id)
        return AgentLoopPorts(
            model=session.model_port or self._model_port,
            tools=session.tool_port or self._tool_port,
            context=context_port,
            events=event_sink,
        )

    def _tool_catalog_for(self, session_id: str) -> list[Any]:
        session = self._sessions.require(session_id)
        tool_port = session.tool_port or self._tool_port
        if tool_port is None:
            return []
        mode = session.controller.describe().current_mode
        try:
            catalog = tool_port.catalog(mode)
        except TypeError:
            catalog = tool_port.catalog()
        if isinstance(catalog, ToolCatalogView):
            return list(catalog.tools)
        if isinstance(catalog, dict):
            value = catalog.get("tools")
            return list(value) if isinstance(value, (list, tuple)) else [catalog]
        if isinstance(catalog, (list, tuple)):
            return list(catalog)
        return [catalog]

    def _pending_approvals_for(self, session: RuntimeSession) -> list[ApprovalView]:
        by_id = {
            view.approval_id: view
            for view in self._approvals.list(session.session_id)
        }
        for item in session.controller.pending_approvals():
            approval_id = _optional_text(item.get("approval_id"))
            if approval_id is None or approval_id in by_id:
                continue
            by_id[approval_id] = ApprovalView(
                approval_id=approval_id,
                session_id=session.session_id,
                run_id=_optional_text(item.get("run_id")) or "",
                tool_call_id=_optional_text(item.get("id")) or "",
                tool_name=_optional_text(item.get("name")) or "",
                reason=_optional_text(item.get("reason")) or "",
                risk_level=_optional_text(item.get("risk_level")) or "unknown",
            )
        return sorted(by_id.values(), key=lambda item: item.approval_id)

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
            warnings=info.warnings,
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


def _plan_decision_alias(value: object) -> str | None:
    text = str(value).strip().lower().lstrip("/") if value is not None else ""
    if text in {"approve", "yes", "y", "ok", "同意", "批准", "可以", "确认"}:
        return "approve"
    if text in {"reject", "deny", "no", "n", "拒绝", "不同意", "不行"}:
        return "reject"
    return None


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
