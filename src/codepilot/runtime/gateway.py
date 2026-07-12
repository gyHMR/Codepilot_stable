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
    SessionContinuationIntent,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionView,
)
from codepilot.sessions.controller import SessionController
from codepilot.protocols import RunSignalsSummary

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
            async for frame in self._dispatch_action(session, action):
                yield frame

    async def _dispatch_action(
        self,
        session: RuntimeSession,
        action: UserAction,
    ) -> AsyncIterator[RuntimeFrame]:
        if isinstance(action, PromptSubmitted):
            checkpoint = session.controller.runtime_checkpoint()
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
                async for frame in self._run_continuation(
                    session,
                    SessionContinuationIntent(
                        kind=(
                            "plan_clarification"
                            if phase == "plan_clarification"
                            else "automatic_continuation"
                            if phase == "plan_incomplete"
                            else "plan_feedback"
                        ),
                        run_id=_optional_text(checkpoint.get("run_id")),
                        text=action.text,
                    ),
                ):
                    yield frame
                return
            async for frame in self._run_prompt(session, action):
                yield frame
            return
        if isinstance(action, CommandSubmitted):
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

    def messages(self, session_id: str) -> tuple[Any, ...]:
        """Return persisted messages through the runtime application boundary."""
        return tuple(self._sessions.require(session_id).controller.messages())

    def close(self, session_id: str) -> None:
        active_run_id = self._active_runs.cancel(session_id)
        self._sessions.close(session_id)
        self._approvals.remove_session(session_id)
        if active_run_id is None or not self._active_runs.has_attached_task(session_id):
            self._active_runs.finish(session_id)
        self._session_locks.pop(session_id, None)

    async def close_all(self) -> None:
        self._active_runs.cancel_all()
        self._sessions.close_all()
        self._approvals.clear()
        self._session_locks.clear()

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

    async def _follow_up_from_command(
        self,
        session: RuntimeSession,
        record: SessionCommandRecord,
    ) -> AsyncIterator[RuntimeFrame]:
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
        prepared = await session.controller.prepare_continuation(intent)
        run_loop = (
            (lambda ports: resume_agent_loop(prepared.resume_input, ports))
            if prepared.resume_input is not None
            else (lambda ports: run_agent_loop(prepared.loop_input, ports))
        )
        async for frame in self._run_agent_loop(session, prepared, run_loop):
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

        prepared = await session.controller.prepare_resume(
            SessionResumeIntent(
                approval_id=action.approval_id,
                decision=action.decision,
                reason=action.reason,
                run_id=challenge.run_id,
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
            self._active_runs.finish(session.session_id, run_id=prepared.run_id)

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
        snapshot = tool_port.catalog_snapshot(
            mode="plan" if mode in {"read", "plan"} else "execute"
        )
        return [entry.spec for entry in snapshot.entries]

    def _pending_approvals_for(self, session: RuntimeSession) -> list[ApprovalView]:
        by_id = {
            view.approval_id: view
            for view in self._approvals.list(session.session_id)
        }
        tool_port = session.tool_port or self._tool_port
        challenges = tuple(tool_port.pending_challenges()) if tool_port is not None else ()
        for challenge in challenges:
            if challenge.approval_id in by_id:
                continue
            by_id[challenge.approval_id] = ApprovalView(
                approval_id=challenge.approval_id,
                session_id=session.session_id,
                run_id=challenge.run_id,
                tool_call_id=challenge.tool_call_id,
                tool_name=challenge.tool_name,
                reason=challenge.reason,
                risk_level=challenge.risk,
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
