from __future__ import annotations

"""Execute one prepared Core run inside a Runtime-owned task."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any, Literal

from codepilot.core.contracts import AgentLoopOutcome
from codepilot.core.runner import run_agent_loop
from codepilot.protocols import RunSignalsSummary
from codepilot.sessions.contracts import PreparedAgentRun

from .environment import RunEnvironment
from .lifecycle import RuntimeLifecycle


class RunExecutor:
    """Own the in-process execution of a prepared run.

    The executor invokes Core's single loop entry and normalizes task cancellation
    and unexpected exceptions into ``AgentLoopOutcome``.  It does not create runs,
    persist Sessions state, or decide task semantics.
    """

    async def execute(
        self,
        environment: RunEnvironment,
        prepared: PreparedAgentRun,
    ) -> AsyncIterator["RunExecutionUpdate"]:
        event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        downstream = environment.event_sink
        lifecycle = RuntimeLifecycle(environment.run_id)
        lifecycle.transition("preparing")

        def event_sink(event: dict[str, Any]) -> None:
            payload = dict(event)
            if downstream is not None:
                downstream(payload)
            event_queue.put_nowait(payload)

        execution_environment = replace(environment, event_sink=event_sink)
        lifecycle.transition("executing")
        task = environment.resources.create_task(
            self._execute_core(execution_environment, prepared),
            name=f"core:{environment.run_id}",
        )
        try:
            while not task.done() or not event_queue.empty():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                yield RunExecutionEvent(event)
            outcome = await task
            if outcome.status in {"waiting_approval", "waiting_user"}:
                lifecycle.transition("waiting")
            else:
                lifecycle.transition("finalizing")
                lifecycle.transition(
                    "terminal",
                    terminal_outcome=(
                        "cancelled"
                        if outcome.status == "cancelled"
                        else outcome.status
                    ),
                )
        except asyncio.CancelledError:
            environment.resources.cancel("runtime_stream_cancelled")
            raise
        finally:
            await environment.resources.release()
            if lifecycle.state in {"terminal", "waiting"}:
                lifecycle.mark_released()
        yield RunExecutionCompleted(outcome)

    async def _execute_core(
        self,
        environment: RunEnvironment,
        prepared: PreparedAgentRun,
    ) -> AgentLoopOutcome:
        try:
            environment.cancellation.raise_if_cancelled()
            return await run_agent_loop(prepared.loop_input, environment.ports())
        except asyncio.CancelledError:
            reason = environment.cancellation.reason or "cancelled"
            timed_out = reason == "deadline_exceeded"
            return AgentLoopOutcome(
                run_id=environment.run_id,
                status="failed" if timed_out else "cancelled",
                stop_reason=reason,
                signals=RunSignalsSummary(cancelled=True),
                error={
                    "code": (
                        "runtime.deadline_exceeded"
                        if timed_out
                        else "runtime.cancelled"
                    ),
                    "message": (
                        f"Run deadline exceeded: {reason}"
                        if timed_out
                        else f"Run cancelled: {reason}"
                    ),
                },
            )
        except Exception as exc:
            return AgentLoopOutcome(
                run_id=environment.run_id,
                status="failed",
                stop_reason="internal_error",
                error=_runtime_error_payload(exc),
            )


@dataclass(frozen=True)
class RunExecutionEvent:
    event: dict[str, Any]
    kind: Literal["event"] = "event"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event", dict(self.event))


@dataclass(frozen=True)
class RunExecutionCompleted:
    outcome: AgentLoopOutcome
    kind: Literal["completed"] = "completed"


RunExecutionUpdate = RunExecutionEvent | RunExecutionCompleted


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
    details: dict[str, Any] = {}
    error_info = getattr(error, "error", None)
    message = str(error)
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


__all__ = [
    "RunExecutionCompleted",
    "RunExecutionEvent",
    "RunExecutionUpdate",
    "RunExecutor",
]
