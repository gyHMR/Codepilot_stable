from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from codepilot.core.loop import resume_agent_loop, run_agent_loop
from codepilot.sessions.contracts import (
    SessionCommandIntent,
    SessionResumeIntent,
    SessionRunIntent,
    SessionView,
)
from codepilot.sessions.controller import SessionController

from .approvals import ApprovalRegistry, ApprovalView
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
from .session_opening import (
    AppSessionView,
    SessionOpenIntent as _SessionOpenIntent,
    SessionRef,
    _to_runtime_assembly_intent,
)
from .sessions import ActiveRunRegistry, RuntimeSessionRegistry


__all__ = ["RuntimeGateway"]


class RuntimeGateway:
    def __init__(
        self,
        *,
        model_port: Any | None = None,
        tool_port: Any | None = None,
    ) -> None:
        self._sessions = RuntimeSessionRegistry()
        self._model_port = model_port
        self._tool_port = tool_port
        self._approvals = ApprovalRegistry()
        self._active_runs = ActiveRunRegistry()

    def open_session(self, intent: _SessionOpenIntent) -> SessionRef:
        from .assembly import assemble_runtime
        from codepilot.tools.ports import ToolRuntimePort

        controller, assembly = assemble_runtime(_to_runtime_assembly_intent(intent))
        session_id = controller.session_id
        from codepilot.llm.ports import ProviderModelPort

        self._sessions.add(
            session_id,
            controller,
            assembly=assembly,
            model_port=self._model_port
            or ProviderModelPort(
                model=assembly.session_options.model,
                stream_fn=assembly.session_options.stream_fn,
                convert_messages=assembly.session_options.convert_to_llm,
                get_api_key=assembly.session_options.get_api_key,
            ),
            tool_port=self._tool_port
            or ToolRuntimePort(
                assembly.tool_runtime,
                before_tool_call=assembly.session_options.before_tool_call,
                after_tool_call=assembly.session_options.after_tool_call,
            ),
        )
        return SessionRef(session_id=session_id)

    async def dispatch(
        self,
        session_id: str,
        action: UserAction,
    ) -> AsyncIterator[RuntimeFrame]:
        controller = self._require_session(session_id)
        if isinstance(action, PromptSubmitted):
            async for frame in self._dispatch_prompt(controller, action):
                yield frame
            return
        if isinstance(action, CommandSubmitted):
            record = await controller.apply_command(
                SessionCommandIntent(
                    text=action.text,
                    tool_catalog=tuple(self._tool_catalog_for(session_id)),
                )
            )
            if record.switched_session_id:
                self._register_derived_controller(
                    source_session_id=session_id,
                    new_session_id=record.switched_session_id,
                    controller=controller,
                )
            yield CommandFinishedFrame(record=record)
            return
        if isinstance(action, ApprovalDecided):
            async for frame in self._dispatch_approval(controller, action):
                yield frame
            return
        if isinstance(action, RunCancelled):
            active = self._active_runs.finish(session_id)
            yield CancelledFrame(
                session_id=session_id,
                cancelled=active is not None,
                reason=action.reason,
            )
            return
        yield FailedFrame(error={"code": "runtime.unknown_action", "action": type(action).__name__})

    def describe(self, session_id: str) -> AppSessionView:
        entry = self._sessions.require(session_id)
        session = entry.controller.describe()
        return AppSessionView(
            session=session,
            status=_session_status(
                entry,
                session,
                running=self._active_runs.is_running(session_id),
            ),
            state=dict(session.context),
            commands=tuple(_builtin_commands()),
            pending_approvals=tuple(self._approvals.list(session_id)),
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

    async def _dispatch_prompt(
        self,
        controller: SessionController,
        action: PromptSubmitted,
    ) -> AsyncIterator[RuntimeFrame]:
        prepared = await controller.prepare_run(
            SessionRunIntent(
                text=action.text,
                images=action.images,
                mode_hint=action.mode_hint,
            )
        )
        self._active_runs.start(controller.session_id, prepared.run_id)
        try:
            outcome = await run_agent_loop(
                prepared.loop_input,
                ports=self._ports_for(
                    controller.session_id,
                    context_port=prepared.context_port,
                ),
            )
            record = await controller.commit_run(prepared, outcome)
        except Exception as exc:
            yield FailedFrame(error=_runtime_error_payload(exc))
            return
        finally:
            self._active_runs.finish(controller.session_id)

        async for frame in self._frames_from_outcome(controller, outcome, record):
            yield frame

    async def _dispatch_approval(
        self,
        controller: SessionController,
        action: ApprovalDecided,
    ) -> AsyncIterator[RuntimeFrame]:
        transaction = self._approvals.get(action.approval_id)
        if transaction is None:
            yield FailedFrame(
                error={
                    "code": "runtime.approval_not_found",
                    "approval_id": action.approval_id,
                }
            )
            return
        if transaction.session_id != controller.session_id:
            yield FailedFrame(
                error={
                    "code": "runtime.approval_session_mismatch",
                    "approval_id": action.approval_id,
                    "session_id": controller.session_id,
                }
            )
            return
        try:
            prepared = await controller.prepare_resume(
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
            self._active_runs.start(controller.session_id, prepared.run_id)
            outcome = await resume_agent_loop(
                prepared.resume_input,
                self._ports_for(
                    controller.session_id,
                    context_port=prepared.context_port,
                ),
            )
            record = await controller.commit_run(prepared, outcome)
        except Exception as exc:
            yield FailedFrame(error=_runtime_error_payload(exc))
            return
        finally:
            self._active_runs.finish(controller.session_id)
        self._approvals.pop(action.approval_id)
        async for frame in self._frames_from_outcome(controller, outcome, record):
            yield frame

    async def _frames_from_outcome(
        self,
        controller: SessionController,
        outcome: Any,
        record: Any,
    ) -> AsyncIterator[RuntimeFrame]:
        for event in outcome.events:
            yield ProgressFrame(event=dict(event))
        if outcome.status == "waiting_approval":
            for interruption in outcome.interruptions:
                self._approvals.add(controller.session_id, interruption)
                yield ApprovalRequiredFrame(approval=interruption)
            return
        if outcome.status == "failed":
            yield FailedFrame(error=_runtime_error_payload(outcome.error))
            return
        yield RunFinishedFrame(record=record)

    def _ports_for(self, session_id: str, *, context_port: Any | None):
        entry = self._sessions.require(session_id)
        return _ports(
            entry.model_port or self._model_port,
            entry.tool_port or self._tool_port,
            context_port,
        )

    def _tool_catalog_for(self, session_id: str) -> list[Any]:
        entry = self._sessions.require(session_id)
        tool_port = entry.tool_port or self._tool_port
        if tool_port is None:
            return []
        catalog = tool_port.catalog()
        if isinstance(catalog, dict):
            value = catalog.get("tools")
            return list(value) if isinstance(value, (list, tuple)) else [catalog]
        if isinstance(catalog, (list, tuple)):
            return list(catalog)
        return [catalog]

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
        source = self._sessions.require(source_session_id)
        assembly = source.assembly
        if assembly is not None:
            session_options = replace(
                assembly.session_options,
                session_id=new_session_id,
                messages=[],
                task_mode=derived.task_mode,  # type: ignore[arg-type]
            )
            assembly = replace(
                assembly,
                session_options=session_options,
                profile=replace(
                    assembly.profile,
                    task_mode=derived.task_mode,  # type: ignore[arg-type]
                ),
            )
        self._sessions.add(
            new_session_id,
            derived,
            assembly=assembly,
            model_port=source.model_port,
            tool_port=source.tool_port,
        )


def _ports(model_port: Any, tool_port: Any | None = None, context_port: Any | None = None):
    from codepilot.core.contracts import AgentLoopPorts

    return AgentLoopPorts(model=model_port, tools=tool_port, context=context_port)


def _builtin_commands():
    from .command_catalog import builtin_commands

    return builtin_commands()


def _runtime_error_payload(error: Any) -> dict[str, Any]:
    code = "runtime.dispatch_failed"
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
        "code": code,
        "message": message,
        "details": details,
    }


def _session_status(entry: Any, session: SessionView, *, running: bool):
    from .views import SessionStatus

    assembly = entry.assembly
    if assembly is not None:
        model = assembly.profile.model
        model_id = f"{model.provider}/{model.id}" if model.provider else model.id
        warnings = tuple(
            diagnostic.message
            for diagnostic in assembly.diagnostics
            if diagnostic.severity == "warning"
        )
        workspace = str(assembly.session_options.workspace_dir)
        permission_mode = assembly.profile.permission_mode
        credential_source = assembly.profile.credential_source
        leaf_id = str((session.context or {}).get("leaf_id") or "N/A")
    else:
        model_id = f"{entry.controller.model.provider}/{entry.controller.model.model_id}"
        workspace = "."
        permission_mode = "workspace-write"
        credential_source = "test"
        warnings = ()
        leaf_id = "N/A"
    return SessionStatus(
        session_id=session.session_id,
        model_id=model_id,
        workspace=workspace,
        permission_mode=permission_mode,
        message_count=session.message_count,
        leaf_id=leaf_id,
        task_mode=session.task_mode,  # type: ignore[arg-type]
        is_running=running,
        credential_source=credential_source,
        warnings=warnings,
    )
