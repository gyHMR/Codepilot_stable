from __future__ import annotations

"""Execute one prepared Core run inside a Runtime-owned task."""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from codepilot.core.contracts import (
    AgentLoopOutcome,
    AgentLoopPorts,
    ContextPort,
    RunStatePort,
)
from codepilot.core.runner import resume_agent_loop, run_agent_loop
from codepilot.protocols import RunSignalsSummary
from codepilot.sessions.contracts import PreparedAgentRun


@dataclass(frozen=True)
class RunEnvironment:
    """The run-scoped ports needed by Core.

    Session preparation owns the durable state and the prepared inputs.  This
    object only carries the live ports into the executor; it is intentionally
    not a second state store.
    """

    run_id: str
    session_id: str
    model: Any | None
    tools: Any | None
    context: ContextPort | None = None
    state: RunStatePort | None = None
    event_sink: Callable[[dict[str, Any]], None] | None = None

    def ports(self) -> AgentLoopPorts:
        return AgentLoopPorts(
            model=self.model,
            tools=self.tools,
            context=self.context,
            state=self.state,
            events=self.event_sink,
        )


class RunExecutor:
    """Own the in-process execution of a prepared run.

    The executor selects prompt versus resume execution and normalizes task
    cancellation and unexpected exceptions into ``AgentLoopOutcome``.  It does
    not create runs, persist Sessions state, or decide task semantics.
    """

    async def execute(
        self,
        environment: RunEnvironment,
        prepared: PreparedAgentRun,
    ) -> AgentLoopOutcome:
        ports = environment.ports()
        try:
            if prepared.resume_input is not None:
                return await resume_agent_loop(prepared.resume_input, ports)
            return await run_agent_loop(prepared.loop_input, ports)
        except asyncio.CancelledError:
            return AgentLoopOutcome(
                run_id=environment.run_id,
                status="aborted",
                stop_reason="aborted",
                signals=RunSignalsSummary(cancelled=True),
                error={
                    "code": "runtime.cancelled",
                    "message": "Run cancelled by user",
                },
            )
        except Exception as exc:
            return AgentLoopOutcome(
                run_id=environment.run_id,
                status="failed",
                stop_reason="internal_error",
                error=_runtime_error_payload(exc),
            )


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


__all__ = ["RunEnvironment", "RunExecutor"]
