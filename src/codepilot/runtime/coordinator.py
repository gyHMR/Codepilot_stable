"""创建面向 Interface 的 SessionController 并连接运行时依赖。"""

from __future__ import annotations

"""Runtime-owned orchestration for one Agent Run.

The session object owns durable session facts and component state.  This
coordinator owns the live Run sequence: prepare/recover, execute, normalize
the outcome, and commit the terminal or waiting result.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import time
from typing import Any

from codepilot.core.contracts import CoreOutcome
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import AgentRunResult
from codepilot.sessions.contracts import (
    PreparedAgentRun,
    SessionContinuationIntent,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionCommandIntent,
    SessionCommandRecord,
    SessionView,
)

from .commands import apply_session_command
from .contracts import (
    external_status,
    external_stop_reason,
    project_core_counters,
    project_core_signals,
    project_core_verification,
)
from .environment import RunEnvironment, RunEnvironmentFactory
from .errors import runtime_error_info
from .executor import RunExecutionUpdate, RunExecutor
from .model import RetryingModelPort, RuntimeContextSummarizer
from .session_coordinator import RuntimeSessionCoordinator, new_run_id


RunRequest = SessionRunIntent | SessionResumeIntent | SessionContinuationIntent


class RunCoordinator:
    """Prepare, execute, and commit one run through a single runtime entry."""

    def __init__(
        self,
        session: RuntimeSessionCoordinator,
        *,
        environment_factory: RunEnvironmentFactory | None = None,
        executor: RunExecutor | None = None,
    ) -> None:
        self.session = session
        self._environment_factory = environment_factory or RunEnvironmentFactory()
        self._executor = executor or RunExecutor()

    async def prepare(
        self,
        request: RunRequest,
        *,
        model: ModelDescriptor,
        tools: Any | None = None,
    ) -> PreparedAgentRun:
        """Open or recover a run and build the unified Core input."""

        if isinstance(request, SessionRunIntent):
            return await self.session._prepare_run(  # noqa: SLF001
                request,
                run_id=request.run_id or new_run_id(),
                model=model,
            )
        if isinstance(request, SessionResumeIntent):
            return await self.session._prepare_resume(  # noqa: SLF001
                request,
                run_id=request.run_id or self.session.resume_run_id(request.approval_id),
                model=model,
                tools=tools,
            )
        return await self.session._prepare_continuation(  # noqa: SLF001
            request,
            run_id=request.run_id or self.session.continuation_run_id(),
            model=model,
            tools=tools,
        )

    def create_environment(
        self,
        prepared: PreparedAgentRun,
        *,
        model: Any | None,
        tools: Any | None,
    ) -> RunEnvironment:
        runtime_model = (
            RetryingModelPort(
                model,
                enabled=bool(self.session.retry_enabled),
                max_retries=max(0, int(self.session.max_retries)),
                base_delay_ms=max(0, int(self.session.retry_base_delay_ms)),
                finalization_sink=self.session.capture_memory_proposals,
            )
            if model is not None
            else None
        )
        self.session.context_service.set_summarizer(
            RuntimeContextSummarizer(runtime_model, prepared.loop_input.model)
            if runtime_model is not None
            else None
        )
        return self._environment_factory.create(
            prepared,
            model=runtime_model,
            tools=tools,
            # Core events are streamed by RunExecutor and become durable only
            # when the next Sessions boundary is committed.
            event_sink=None,
            deadline_at_ms=self._deadline_at_ms(),
        )

    def _deadline_at_ms(self) -> int | None:
        seconds = self.session.run_timeout_seconds
        if seconds is None:
            return None
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
            raise ValueError("run_timeout_seconds must be a positive integer or None")
        return int(time.time() * 1000) + seconds * 1000

    async def execute(
        self,
        environment: RunEnvironment,
        prepared: PreparedAgentRun,
    ) -> AsyncIterator[RunExecutionUpdate]:
        async for update in self._executor.execute(environment, prepared):
            yield update

    async def commit(
        self,
        prepared: PreparedAgentRun,
        outcome: CoreOutcome,
        *,
        events: tuple[dict[str, object], ...] = (),
    ) -> SessionRunRecord:
        """Normalize the Core outcome and commit it through the session store."""

        result = self._agent_result_from_outcome(prepared, outcome)
        record = await self.session._commit_run(  # noqa: SLF001
            prepared,
            outcome,
            result,
            events=events,
        )
        return record

    def _agent_result_from_outcome(
        self,
        prepared: PreparedAgentRun,
        outcome: CoreOutcome,
    ) -> AgentRunResult:
        state = outcome.state
        return AgentRunResult(
            run_id=prepared.run_id,
            session_id=self.session.session_id,
            status=external_status(outcome),
            stop_reason=external_stop_reason(outcome.reason),
            counters=project_core_counters(outcome),
            messages=[*prepared.input_messages, *outcome.new_messages],
            final_message=outcome.final_message,
            error=(
                runtime_error_info(outcome.error or outcome.reason)
                if outcome.status == "failed"
                else None
            ),
            affected_paths=list(state.facts.workspace.affected_paths),
            workspace_changed=state.facts.workspace.changed,
            verification=project_core_verification(outcome),
            plan=(state.task.plan.to_summary() if state.task.plan is not None else None),
            signals=project_core_signals(outcome),
        )


def create_session_controller(options: Any) -> "SessionController":
    return _bind_session_runtime(RuntimeSessionCoordinator(options))


def _bind_session_runtime(session: RuntimeSessionCoordinator) -> "SessionController":
    model = session.conversation.model
    return SessionController(
        session_id=session.session_id,
        model=ModelDescriptor(
            provider=getattr(model, "provider", "unknown"),
            model_id=getattr(model, "id", "unknown"),
        ),
        current_mode=session.current_mode,
        _session=session,
    )


@dataclass
class SessionController:
    """Session-facing facade exposing facts, commands, and one Run coordinator."""

    session_id: str
    model: ModelDescriptor = field(
        default_factory=lambda: ModelDescriptor(provider="local", model_id="v2-test")
    )
    current_mode: str = "build"
    _session: RuntimeSessionCoordinator | None = None
    _derived_controllers: dict[str, "SessionController"] = field(default_factory=dict)
    _runs: RunCoordinator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self._session is None:
            raise ValueError("SessionController requires RuntimeSessionCoordinator")
        self._runs = RunCoordinator(self._session)

    @property
    def runs(self) -> RunCoordinator:
        return self._runs

    def describe(self) -> SessionView:
        return self._session.describe(last_run_id=self._session.session_state.last_run_id)

    async def apply_command(self, intent: SessionCommandIntent) -> SessionCommandRecord:
        record = await apply_session_command(
            self.session_id,
            intent,
            session=self._session,
            controller=self,
        )
        if "current_mode" in record.data:
            self.current_mode = str(record.data["current_mode"])
        return record

    def runtime_checkpoint(self) -> dict[str, Any] | None:
        return self._session.runtime_checkpoint()

    def component_checkpoint_state(self, owner: str) -> dict[str, object] | None:
        return self._session.component_checkpoint_state(owner)

    def current_plan_state(self) -> dict[str, Any] | None:
        return self._session.current_plan_state()

    def stage_derived_session(self, session: RuntimeSessionCoordinator) -> None:
        self._derived_controllers[session.session_id] = _bind_session_runtime(session)

    def claim_derived_controller(self, session_id: str) -> "SessionController" | None:
        return self._derived_controllers.pop(session_id, None)

    def close(self) -> None:
        self._session.close()


__all__ = [
    "RunCoordinator",
    "RunRequest",
    "SessionController",
    "create_session_controller",
]
