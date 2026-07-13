from __future__ import annotations

"""Runtime-owned orchestration for one Agent Run.

The session object owns durable session facts and component state.  This
coordinator owns the live Run sequence: prepare/recover, execute, normalize
the outcome, and commit the terminal or waiting result.
"""

from collections.abc import AsyncIterator
import time
from typing import Any

from codepilot.core.contracts import AgentLoopOutcome
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import AgentRunResult, ErrorInfo
from codepilot.sessions.contracts import (
    PreparedAgentRun,
    SessionContinuationIntent,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
)

from .environment import RunEnvironment, RunEnvironmentFactory
from .executor import RunExecutionUpdate, RunExecutor
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
            )
        return await self.session._prepare_continuation(  # noqa: SLF001
            request,
            run_id=request.run_id or self.session.continuation_run_id(),
            model=model,
        )

    def create_environment(
        self,
        prepared: PreparedAgentRun,
        *,
        model: Any | None,
        tools: Any | None,
    ) -> RunEnvironment:
        return self._environment_factory.create(
            prepared,
            model=model,
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
        outcome: AgentLoopOutcome,
    ) -> SessionRunRecord:
        """Normalize the Core outcome and commit it through the session store."""

        structured_result = isinstance(outcome.error, AgentRunResult)
        result = (
            outcome.error
            if structured_result
            else self._agent_result_from_outcome(prepared, outcome)
        )
        record = await self.session._commit_run(  # noqa: SLF001
            prepared,
            outcome,
            result,
            store_outcome=not structured_result,
        )
        return record

    def _agent_result_from_outcome(
        self,
        prepared: PreparedAgentRun,
        outcome: AgentLoopOutcome,
    ) -> AgentRunResult:
        return AgentRunResult(
            run_id=outcome.run_id,
            session_id=self.session.session_id,
            status=_agent_status(outcome.status),
            stop_reason=_agent_stop_reason(outcome.stop_reason),
            counters=outcome.counters,
            messages=[*prepared.input_messages, *outcome.new_messages],
            final_message=outcome.final_message,
            error=_error_info(outcome.error) if outcome.status == "failed" else None,
            affected_paths=list(outcome.workspace_effects.affected_paths),
            workspace_changed=outcome.workspace_effects.changed,
            verification=list(outcome.verification),
            plan=outcome.plan,
            signals=outcome.signals,
        )


def _agent_status(status: str) -> str:
    return "aborted" if status == "cancelled" else status


def _agent_stop_reason(reason: str) -> str:
    return {
        "max_model_turns": "max_iterations",
        "tool_call_limit": "max_iterations",
        "missing_tool_port": "internal_error",
        "missing_approval_decision": "internal_error",
        "cancelled": "aborted",
        "user_cancelled": "aborted",
        "runtime_stream_cancelled": "aborted",
        "session_closed": "aborted",
    }.get(reason, reason)


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


__all__ = ["RunCoordinator", "RunRequest"]
