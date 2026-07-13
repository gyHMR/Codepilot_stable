from __future__ import annotations

"""Execute one prepared Core run inside a Runtime-owned task."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any, Literal

from codepilot.core.contracts import CoreOutcome, CoreReason
from codepilot.core.driver import run_core
from codepilot.sessions.contracts import PreparedAgentRun

from .environment import RunEnvironment
from .errors import runtime_error_payload


_LIVE_EVENT_QUEUE_LIMIT = 2048


class RunExecutor:
    """Own the in-process execution of a prepared run.

    The executor invokes Core's single loop entry and normalizes task cancellation
    and unexpected exceptions into ``CoreOutcome``. It does not create runs,
    persist Sessions state, or decide task semantics.
    """

    async def execute(
        self,
        environment: RunEnvironment,
        prepared: PreparedAgentRun,
    ) -> AsyncIterator["RunExecutionUpdate"]:
        event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=_LIVE_EVENT_QUEUE_LIMIT
        )
        downstream = environment.event_sink
        live_events: list[dict[str, object]] = []
        lifecycle = environment.lifecycle
        if lifecycle is None:  # pragma: no cover - RunEnvironment guarantees it
            raise RuntimeError("RunEnvironment lifecycle is required")
        lifecycle.transition("preparing")

        def event_sink(event: dict[str, Any]) -> None:
            payload = dict(event)
            live_events.append(payload)
            if downstream is not None:
                try:
                    downstream(payload)
                except Exception:
                    # Live progress is non-authoritative; boundary events remain durable.
                    pass
            try:
                event_queue.put_nowait(payload)
            except asyncio.QueueFull:
                # Durable events remain in Core's boundary recorder. A slow
                # interface may lose live progress, but cannot grow memory without bound.
                pass

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
        except asyncio.CancelledError:
            environment.resources.cancel("runtime_stream_cancelled")
            raise
        yield RunExecutionCompleted(outcome, events=tuple(live_events))

    async def _execute_core(
        self,
        environment: RunEnvironment,
        prepared: PreparedAgentRun,
    ) -> CoreOutcome:
        try:
            _bind_tool_checkpoint_reader(environment)
            environment.cancellation.raise_if_cancelled()
            outcome = await run_core(prepared.loop_input, environment.ports())
            environment.cancellation.raise_if_cancelled()
            return outcome
        except asyncio.CancelledError:
            reason = environment.cancellation.reason or "cancelled"
            timed_out = reason == "deadline_exceeded"
            error = {
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
            }
            return CoreOutcome(
                status="failed" if timed_out else "cancelled",
                reason=CoreReason(
                    "runtime.deadline_exceeded" if timed_out else "runtime.cancelled",
                    message=error["message"],
                    source="runtime",
                    details={"cancellation_reason": reason},
                ),
                state=prepared.loop_input.state,
                error=error,
            )
        except Exception as exc:
            error = runtime_error_payload(exc)
            return CoreOutcome(
                status="failed",
                reason=CoreReason(
                    str(error.get("code") or "runtime.internal_error"),
                    message=str(error.get("message") or "Runtime execution failed"),
                    source="runtime",
                ),
                state=prepared.loop_input.state,
                error=error,
            )


@dataclass(frozen=True)
class RunExecutionEvent:
    event: dict[str, Any]
    kind: Literal["event"] = "event"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event", dict(self.event))


@dataclass(frozen=True)
class RunExecutionCompleted:
    outcome: CoreOutcome
    events: tuple[dict[str, object], ...] = ()
    kind: Literal["completed"] = "completed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(dict(event) for event in self.events))


RunExecutionUpdate = RunExecutionEvent | RunExecutionCompleted


def _bind_tool_checkpoint_reader(environment: RunEnvironment) -> None:
    bind = getattr(environment.state, "bind_tool_state", None)
    if not callable(bind):
        return
    checkpoint = getattr(environment.tools, "checkpoint_state", None)
    bind((lambda: checkpoint()) if callable(checkpoint) else None)


__all__ = [
    "RunExecutionCompleted",
    "RunExecutionEvent",
    "RunExecutionUpdate",
    "RunExecutor",
]
