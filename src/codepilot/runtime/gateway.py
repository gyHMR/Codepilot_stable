"""提供 Interface 调用 Runtime Session 能力的统一门面。"""

from __future__ import annotations

"""Runtime gateway: receive interface actions and stream runtime frames."""

import asyncio
from collections.abc import AsyncIterator, Awaitable
from typing import Any, TypeVar
from typing import TYPE_CHECKING

from codepilot.core.contracts import CoreOutcome, CoreReason
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
from .actions import (
    AppSessionView,
    ApprovalDecided,
    ApprovalRequiredFrame,
    ApprovalView,
    CancelledFrame,
    CommandDescriptor,
    CommandFinishedFrame,
    CommandSubmitted,
    ContinuationRequested,
    FailedFrame,
    ProgressFrame,
    PromptSubmitted,
    InteractionResponded,
    RunCancelled,
    RunFinishedFrame,
    RunPausedFrame,
    RuntimeFrame,
    SessionOpenIntent,
    SessionRef,
    SessionStatus,
    UserAction,
)
from .builder import build_runtime_session
from .commands import builtin_commands
from .coordinator import SessionController
from .contracts import external_stop_reason, terminal_outcome_for_status
from .registry import ActiveRunRegistry, RuntimeSession, RuntimeSessionRegistry
from .executor import RunExecutionCompleted, RunExecutionEvent
from .errors import runtime_error_payload

__all__ = ["RuntimeGateway"]


T = TypeVar("T")


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
        self._background_tasks: set[asyncio.Task[Any]] = set()

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
                yield FailedFrame(error=runtime_error_payload(exc))

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
                if waiting_kind is None:
                    request_id = f"recovery:{checkpoint_run_id}"
                    yield FailedFrame(
                        error={
                            "code": "runtime.recovery_required",
                            "message": (
                                "Recover the interrupted run before sending a new prompt."
                            ),
                            "retryable": True,
                            "run_id": checkpoint_run_id,
                            "request_id": request_id,
                        }
                    )
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
        if isinstance(action, InteractionResponded):
            checkpoint = session.controller.runtime_checkpoint() or {}
            if str(checkpoint.get("waiting_kind") or "") != "user_input":
                yield FailedFrame(error={"code": "runtime.interaction_not_pending", "request_id": action.request_id})
                return
            if str(checkpoint.get("request_id") or "") != action.request_id:
                yield FailedFrame(error={"code": "runtime.interaction_mismatch", "request_id": action.request_id})
                return
            async for frame in self._run_continuation(
                session,
                SessionContinuationIntent(
                    kind="user_input_response",
                    run_id=_optional_text(checkpoint.get("run_id")),
                    text=action.answer,
                ),
            ):
                yield frame
            return
        if isinstance(action, ContinuationRequested):
            checkpoint = session.controller.runtime_checkpoint() or {}
            checkpoint_run_id = _optional_text(checkpoint.get("run_id"))
            recovery_request_id = (
                f"recovery:{checkpoint_run_id}" if checkpoint_run_id is not None else None
            )
            if (
                checkpoint_run_id is not None
                and checkpoint.get("waiting_kind") is None
                and action.request_id == recovery_request_id
            ):
                async for frame in self._run_continuation(
                    session,
                    SessionContinuationIntent(
                        kind="automatic_continuation",
                        run_id=checkpoint_run_id,
                    ),
                ):
                    yield frame
                return
            if str(checkpoint.get("waiting_kind") or "") != "continuation":
                yield FailedFrame(error={"code": "runtime.continuation_not_pending", "request_id": action.request_id})
                return
            if str(checkpoint.get("request_id") or "") != action.request_id:
                yield FailedFrame(error={"code": "runtime.continuation_mismatch", "request_id": action.request_id})
                return
            async for frame in self._run_continuation(
                session,
                SessionContinuationIntent(
                    kind="automatic_continuation",
                    run_id=_optional_text(checkpoint.get("run_id")),
                ),
            ):
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
        active_scope = self._active_runs.resources(session_id)
        active_run_id = self._active_runs.cancel(session_id, "session_closed")
        closed = self._sessions.detach(session_id)
        if closed is not None:
            self._schedule_controller_close(closed.controller, after=active_scope)
        if (
            closed is not None
            and closed.mcp_manager is not None
            and all(
                item.mcp_manager is not closed.mcp_manager
                for item in self._sessions.values()
            )
        ):
            self._schedule_manager_close(closed.mcp_manager, after=active_scope)
        if active_run_id is None or not self._active_runs.has_active_tasks(session_id):
            self._active_runs.finish(session_id)
        self._session_locks.pop(session_id, None)

    async def close_all(self) -> None:
        scopes = self._active_runs.cancel_all()
        if scopes:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(scope.wait_released() for scope in scopes)),
                    timeout=5,
                )
            except asyncio.TimeoutError:
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
        if self._background_tasks:
            await asyncio.gather(*tuple(self._background_tasks), return_exceptions=True)
        self._session_locks.clear()

    def _schedule_manager_close(
        self,
        manager: object,
        *,
        after: object | None = None,
    ) -> None:
        async def close() -> None:
            if after is not None:
                await after.wait_released()
            await manager.aclose()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(close())
            return
        task = loop.create_task(close(), name="runtime-manager-close")
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _schedule_controller_close(
        self,
        controller: SessionController,
        *,
        after: object | None = None,
    ) -> None:
        if after is None:
            controller.close()
            return

        async def close() -> None:
            await after.wait_released()
            controller.close()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(close())
            return
        task = loop.create_task(close(), name="runtime-session-close")
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

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
                request_id=action.request_id,
                images=action.images,
                mode_hint=action.mode_hint,
            ),
            model=session.controller.model,
            tools=session.tool_port or self._tool_port,
        )
        async for frame in self._execute_prepared_run(
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
            tools=session.tool_port or self._tool_port,
        )
        async for frame in self._execute_prepared_run(session, prepared):
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
            tools=tool_port,
        )
        async for frame in self._execute_prepared_run(
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

    async def _execute_prepared_run(
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
        outcome: CoreOutcome | None = None
        outcome_events: tuple[dict[str, object], ...] = ()
        released = False
        lifecycle = environment.lifecycle
        if lifecycle is None:  # pragma: no cover - RunEnvironment guarantees it
            raise RuntimeError("RunEnvironment lifecycle is required")
        try:
            try:
                async for update in session.controller.runs.execute(environment, prepared):
                    if isinstance(update, RunExecutionEvent):
                        yield ProgressFrame(event=update.event)
                    elif isinstance(update, RunExecutionCompleted):
                        outcome = update.outcome
                        outcome_events = update.events
            except asyncio.CancelledError:
                environment.resources.cancel("runtime_stream_cancelled")
                lifecycle.transition("cancelling")
                await environment.resources.quiesce()
                cancelled = CoreOutcome(
                    status="cancelled",
                    reason=CoreReason(
                        "runtime.stream_cancelled",
                        message="Runtime stream consumer cancelled",
                        source="runtime",
                    ),
                    state=prepared.loop_input.state,
                    error={
                        "code": "runtime.cancelled",
                        "message": "Runtime stream consumer cancelled",
                    },
                )
                lifecycle.transition("finalizing")
                try:
                    await asyncio.shield(
                        session.controller.runs.commit(
                            prepared,
                            cancelled,
                            events=outcome_events,
                        )
                    )
                except Exception:
                    lifecycle.transition("terminal", terminal_outcome="failed")
                    raise
                lifecycle.transition("terminal", terminal_outcome="cancelled")
                await asyncio.shield(environment.resources.release())
                lifecycle.mark_released()
                released = True
                self._active_runs.finish(session.session_id, run_id=prepared.run_id)
                raise

            if outcome is None:
                return

            waiting = outcome.status == "waiting"
            environment.resources.seal_cancellation()
            if not waiting:
                lifecycle.transition("finalizing")
            stream_cancellation: asyncio.CancelledError | None = None
            try:
                record, stream_cancellation = await _complete_critical(
                    session.controller.runs.commit(
                        prepared,
                        outcome,
                        events=outcome_events,
                    )
                )
            except Exception as exc:
                if lifecycle.state == "executing":
                    lifecycle.transition("finalizing")
                lifecycle.transition("terminal", terminal_outcome="failed")
                yield FailedFrame(error=runtime_error_payload(exc))
                return

            if waiting:
                lifecycle.transition("waiting")
            else:
                lifecycle.transition(
                    "terminal",
                    terminal_outcome=terminal_outcome_for_status(outcome.status),
                )

            _, release_cancellation = await _complete_critical(
                environment.resources.release()
            )
            stream_cancellation = stream_cancellation or release_cancellation
            lifecycle.mark_released()
            released = True
            self._active_runs.finish(session.session_id, run_id=prepared.run_id)

            if stream_cancellation is not None:
                raise stream_cancellation

            async for frame in self._frames_from_outcome(session, outcome, record):
                yield frame
        finally:
            cleanup_cancellation: asyncio.CancelledError | None = None
            if not released:
                _, cleanup_cancellation = await _complete_critical(
                    environment.resources.release()
                )
                if lifecycle.state in {"terminal", "waiting"}:
                    lifecycle.mark_released()
            self._active_runs.finish(session.session_id, run_id=prepared.run_id)
            if cleanup_cancellation is not None:
                raise cleanup_cancellation

    async def _frames_from_outcome(
        self,
        session: RuntimeSession,
        outcome: CoreOutcome,
        record: SessionRunRecord,
    ) -> AsyncIterator[RuntimeFrame]:
        controller = session.controller
        if outcome.status == "waiting" and outcome.wait is not None:
            if outcome.wait.kind == "tool_approval":
                tool_port = session.tool_port or self._tool_port
                challenge = (
                    tool_port.approval_challenge(outcome.wait.request_id)
                    if tool_port is not None
                    else None
                )
                if challenge is not None:
                    yield ApprovalRequiredFrame(approval=challenge)
            yield RunPausedFrame(
                record=record,
                checkpoint=controller.runtime_checkpoint() or {},
            )
            return
        if outcome.status == "failed":
            yield FailedFrame(
                error=runtime_error_payload(outcome.error or outcome.reason)
            )
            return
        if outcome.status == "cancelled":
            yield CancelledFrame(
                session_id=controller.session_id,
                cancelled=True,
                reason=external_stop_reason(outcome.reason),
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
                effects=tuple(sorted(challenge.effects)),
                safe_preview=dict(challenge.safe_preview),
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


async def _complete_critical(
    awaitable: Awaitable[T],
) -> tuple[T, asyncio.CancelledError | None]:
    """Finish a commit or release before propagating transport cancellation."""

    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task), None
    except asyncio.CancelledError as exc:
        return await asyncio.shield(task), exc


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
