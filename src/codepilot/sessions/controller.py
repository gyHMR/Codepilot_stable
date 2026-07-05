from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codepilot.core.contracts import (
    AgentLoopInput,
    AgentLoopOutcome,
    AgentLoopLimits,
    AgentResumeInput,
    RunCorrelation,
)
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import AgentEvent, AgentRunResult, ErrorInfo, Message, UserMessage

from .commands import apply_session_command
from .contracts import (
    PreparedAgentRun,
    SessionCommandIntent,
    SessionCommandRecord,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionView,
)
from .lifecycle import (
    capture_rollback_baseline_ref,
    close_runtime_session,
    commit_runtime_run,
    describe_runtime_session,
    new_v2_run_id,
    prepare_runtime_resume,
    prepare_runtime_run,
)


if TYPE_CHECKING:
    from .session import SessionRuntime


SessionEventListener = Callable[[AgentEvent], None]
Unsubscribe = Callable[[], None]


def create_session_controller(options: Any) -> "SessionController":
    """Create a controller from session-owned runtime options."""

    from .session import SessionRuntime

    return _bind_session_runtime(SessionRuntime(options))


def _bind_session_runtime(session: "SessionRuntime") -> "SessionController":
    model_value = getattr(getattr(session, "conversation", None), "model", None)
    descriptor = ModelDescriptor(
        provider=getattr(model_value, "provider", "unknown"),
        model_id=getattr(model_value, "id", "unknown"),
    )
    return SessionController(
        session_id=session.session_id,
        model=descriptor,
        task_mode=session.task_mode,
        _session=session,
    )


@dataclass
class SessionController:
    session_id: str
    model: ModelDescriptor = field(default_factory=lambda: ModelDescriptor(provider="local", model_id="v2-test"))
    task_mode: str = "edit"
    _session: Any | None = None
    _messages: list[Message] = field(default_factory=list)
    _events: list[AgentEvent] = field(default_factory=list)
    _last_run_id: str | None = None
    _listeners: list[SessionEventListener] = field(default_factory=list)
    _derived_controllers: dict[str, "SessionController"] = field(default_factory=dict)

    def describe(self) -> SessionView:
        if self._session is not None:
            return describe_runtime_session(
                self._session,
                last_run_id=self._last_run_id,
            )
        return SessionView(
            session_id=self.session_id,
            message_count=len(self._messages),
            last_run_id=self._last_run_id,
            task_mode=self.task_mode,
            context=self._snapshot_state(),
        )

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "message_count": len(self._messages),
            "entry_ids": [],
            "entries": [],
            "tree": [],
            "leaf_id": self._last_run_id or "N/A",
            "task_mode": self.task_mode,
            "planning_budget_profile": "balanced",
        }

    async def prepare_run(self, intent: SessionRunIntent) -> PreparedAgentRun:
        run_id = intent.run_id or new_v2_run_id()
        if self._session is not None:
            return await prepare_runtime_run(
                self._session,
                intent,
                run_id=run_id,
                model=self.model,
            )
        user_message = UserMessage(content=intent.text)
        loop_input = AgentLoopInput(
            run_id=run_id,
            correlation=RunCorrelation(session_id=self.session_id),
            messages=list(self._messages),
            user_prompt=intent.text,
            context={
                "system_prompt": "You are Codepilot.",
                "session_id": self.session_id,
            },
                model=self.model,
                task_strategy=_controller_task_strategy(self, mode_hint=intent.mode_hint),
                limits=_controller_loop_limits(self),
            )
        return PreparedAgentRun(
            run_id=run_id,
            session_id=self.session_id,
            loop_input=loop_input,
            input_messages=[user_message],
            rollback_baseline=capture_rollback_baseline_ref(self.session_id, run_id),
            context_refs={"prepared": True},
            memory_refs={"admitted": bool(intent.text)},
            recovery_refs={"run_id": run_id},
        )

    async def prepare_resume(self, intent: SessionResumeIntent) -> PreparedAgentRun:
        run_id = intent.run_id or new_v2_run_id()
        if self._session is not None:
            return await prepare_runtime_resume(
                self._session,
                intent,
                run_id=run_id,
                model=self.model,
            )
        resume_input = AgentResumeInput(
            run_id=run_id,
            correlation=RunCorrelation(session_id=self.session_id),
            messages=list(self._messages),
            context={
                "system_prompt": "You are Codepilot.",
                "session_id": self.session_id,
            },
            model=self.model,
            approval_id=intent.approval_id,
            decision=intent.decision,
            reason=intent.reason,
            task_strategy=_controller_task_strategy(self),
        )
        return PreparedAgentRun(
            run_id=run_id,
            session_id=self.session_id,
            loop_input=AgentLoopInput(
                run_id=run_id,
                correlation=RunCorrelation(session_id=self.session_id),
                messages=list(self._messages),
                context={
                    "system_prompt": "You are Codepilot.",
                    "session_id": self.session_id,
                },
                model=self.model,
                limits=_controller_loop_limits(self),
            ),
            resume_input=resume_input,
            rollback_baseline=capture_rollback_baseline_ref(self.session_id, run_id),
            context_refs={"prepared": True},
            recovery_refs={"run_id": run_id},
        )

    async def commit_run(
        self,
        prepared: PreparedAgentRun,
        outcome: AgentLoopOutcome,
    ) -> SessionRunRecord:
        if self._session is not None:
            structured_result = isinstance(outcome.error, AgentRunResult)
            result = outcome.error if structured_result else self._agent_result_from_outcome(prepared, outcome)
            record = await commit_runtime_run(
                self._session,
                prepared,
                outcome,
                result,
                store_outcome=not structured_result,
            )
            self._last_run_id = prepared.run_id
            return record
        self._messages.extend(prepared.input_messages)
        self._messages.extend(outcome.new_messages)
        self._events.extend(outcome.events)
        self._last_run_id = prepared.run_id
        for event in outcome.events:
            for listener in list(self._listeners):
                listener(event)
        return SessionRunRecord(
            run_id=prepared.run_id,
            session_id=self.session_id,
            status=outcome.status,
            stop_reason=outcome.stop_reason,
            new_messages=list(outcome.new_messages),
            final_text=outcome.final_text,
            events=list(outcome.events),
            outcome=outcome,
            snapshots={
                "context": prepared.context_refs,
                "memory": prepared.memory_refs,
                "recovery": prepared.recovery_refs,
                "rollback": prepared.rollback_baseline,
            },
        )

    async def apply_command(self, intent: SessionCommandIntent) -> SessionCommandRecord:
        record = await apply_session_command(
            self.session_id,
            intent,
            session=self._session,
            controller=self,
        )
        if "task_mode" in record.data:
            self.task_mode = str(record.data["task_mode"])
        return record

    def stage_derived_session(self, session: Any) -> None:
        """Stage a session created by a session command until runtime registers it."""

        self._derived_controllers[session.session_id] = _bind_session_runtime(session)

    def claim_derived_controller(self, session_id: str) -> "SessionController" | None:
        """Return the staged controller for a command-created session."""

        return self._derived_controllers.pop(session_id, None)

    def subscribe(self, listener: SessionEventListener) -> Unsubscribe:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def close(self) -> None:
        self._listeners.clear()
        if self._session is not None:
            close_runtime_session(self._session)

    def _agent_result_from_outcome(
        self,
        prepared: PreparedAgentRun,
        outcome: AgentLoopOutcome,
    ) -> AgentRunResult:
        return AgentRunResult(
            run_id=outcome.run_id,
            session_id=self.session_id,
            status=_agent_status(outcome.status),
            stop_reason=_agent_stop_reason(outcome.stop_reason),
            counters=outcome.counters,
            messages=[*prepared.input_messages, *outcome.new_messages],
            final_message=outcome.final_message,
            error=_error_info(outcome.error) if outcome.status == "failed" else None,
            affected_paths=list(outcome.workspace_effects.affected_paths),
            workspace_changed=outcome.workspace_effects.changed,
            verification=list(outcome.verification),
            task=outcome.task,
        )


def _controller_loop_limits(_controller: SessionController) -> AgentLoopLimits:
    return AgentLoopLimits()


def _controller_task_strategy(
    controller: SessionController,
    *,
    mode_hint: str | None = None,
) -> dict[str, Any]:
    return {
        "enabled": False,
        "mode": mode_hint or controller.task_mode,
    }


def _agent_status(status: str) -> str:
    if status == "cancelled":
        return "aborted"
    return status


def _agent_stop_reason(reason: str) -> str:
    mapping = {
        "max_model_turns": "max_iterations",
        "tool_call_limit": "max_iterations",
        "missing_tool_port": "internal_error",
        "missing_approval_decision": "internal_error",
    }
    return mapping.get(reason, reason)


def _error_info(error: Any) -> ErrorInfo | None:
    if isinstance(error, ErrorInfo):
        return error
    if error is None:
        return None
    return ErrorInfo(
        code="run.internal_error",
        message=str(error),
        retryable=False,
        source="runtime",
        details={"error_type": type(error).__name__},
    )
