from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codepilot.core.contracts import (
    AgentLoopOutcome,
)
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import AgentRunResult, ErrorInfo

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
from .commit import commit_runtime_run
from .prepare import (
    close_runtime_session,
    describe_runtime_session,
    new_v2_run_id,
    prepare_runtime_resume,
    prepare_runtime_run,
)


if TYPE_CHECKING:
    from .prepare import SessionRuntime


def create_session_controller(options: Any) -> "SessionController":
    """Create a controller from session-owned runtime options."""

    from .prepare import SessionRuntime

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
    task_mode: str = "build"
    _session: Any | None = None
    _last_run_id: str | None = None
    _derived_controllers: dict[str, "SessionController"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self._session is None:
            raise ValueError("SessionController requires SessionRuntime")

    def describe(self) -> SessionView:
        return describe_runtime_session(
            self._session,
            last_run_id=self._last_run_id,
        )

    async def prepare_run(self, intent: SessionRunIntent) -> PreparedAgentRun:
        run_id = intent.run_id or new_v2_run_id()
        return await prepare_runtime_run(
            self._session,
            intent,
            run_id=run_id,
            model=self.model,
        )

    async def prepare_resume(self, intent: SessionResumeIntent) -> PreparedAgentRun:
        run_id = intent.run_id or new_v2_run_id()
        return await prepare_runtime_resume(
            self._session,
            intent,
            run_id=run_id,
            model=self.model,
        )

    async def commit_run(
        self,
        prepared: PreparedAgentRun,
        outcome: AgentLoopOutcome,
    ) -> SessionRunRecord:
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

    def close(self) -> None:
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
