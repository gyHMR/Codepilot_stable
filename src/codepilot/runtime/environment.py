from __future__ import annotations

"""Run-scoped capabilities and resource ownership."""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from codepilot.core.contracts import AgentLoopPorts, ContextPort, RunStatePort
from codepilot.sessions.contracts import PreparedAgentRun


RunTrigger = Literal["prompt", "resume", "continuation"]
CleanupCallback = Callable[[], Awaitable[None] | None]


class RunCancellationToken:
    def __init__(self) -> None:
        self._cancelled = False
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> bool:
        if self._cancelled:
            return False
        self._cancelled = True
        self._reason = _required_text(reason, "cancellation reason")
        return True

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError(self._reason)


class RunResourceScope:
    """Own tasks and cleanup callbacks created for one execution attempt."""

    def __init__(
        self,
        *,
        deadline_at_ms: int | None = None,
        cancellation: RunCancellationToken | None = None,
        cancel_grace_ms: int = 250,
    ) -> None:
        if deadline_at_ms is not None and (
            isinstance(deadline_at_ms, bool) or not isinstance(deadline_at_ms, int)
        ):
            raise TypeError("deadline_at_ms must be an int or None")
        if isinstance(cancel_grace_ms, bool) or not isinstance(cancel_grace_ms, int):
            raise TypeError("cancel_grace_ms must be an int")
        if cancel_grace_ms < 0:
            raise ValueError("cancel_grace_ms cannot be negative")
        self.deadline_at_ms = deadline_at_ms
        self.cancel_grace_ms = cancel_grace_ms
        self.cancellation = cancellation or RunCancellationToken()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._cleanups: list[CleanupCallback] = []
        self._deadline_task: asyncio.Task[None] | None = None
        self._force_cancel_task: asyncio.Task[None] | None = None
        self._release_lock = asyncio.Lock()
        self._released = False
        self._release_errors: list[str] = []

    @property
    def released(self) -> bool:
        return self._released

    @property
    def release_errors(self) -> tuple[str, ...]:
        return tuple(self._release_errors)

    @property
    def has_active_tasks(self) -> bool:
        return any(not task.done() for task in self._tasks)

    def add_cleanup(self, callback: CleanupCallback) -> None:
        if self._released:
            raise RuntimeError("RunResourceScope is already released")
        if not callable(callback):
            raise TypeError("cleanup callback must be callable")
        self._cleanups.append(callback)

    def track_task(self, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        if self._released:
            task.cancel()
            raise RuntimeError("RunResourceScope is already released")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def create_task(
        self,
        awaitable: Awaitable[Any],
        *,
        name: str | None = None,
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(awaitable, name=name)
        self.track_task(task)
        self.arm_deadline()
        return task

    def arm_deadline(self) -> None:
        if self.deadline_at_ms is None or self._deadline_task is not None or self._released:
            return
        task = asyncio.create_task(self._wait_for_deadline(), name="run-deadline")
        self._deadline_task = task
        self.track_task(task)

    def cancel(self, reason: str = "cancelled") -> bool:
        changed = self.cancellation.cancel(reason)
        if changed and self.cancel_grace_ms == 0:
            self._force_cancel_tasks()
        elif changed and self._force_cancel_task is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._force_cancel_tasks()
            else:
                self._force_cancel_task = loop.create_task(
                    self._force_cancel_after_grace(),
                    name="run-cancel-grace",
                )
        return changed

    async def release(self) -> tuple[str, ...]:
        async with self._release_lock:
            if self._released:
                return self.release_errors
            current = asyncio.current_task()
            if (
                self._force_cancel_task is not None
                and self._force_cancel_task is not current
                and not self._force_cancel_task.done()
            ):
                self._force_cancel_task.cancel()
                await asyncio.gather(self._force_cancel_task, return_exceptions=True)
            tasks = [
                task
                for task in self._tasks
                if task is not current and not task.done()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for callback in reversed(self._cleanups):
                try:
                    value = callback()
                    if inspect.isawaitable(value):
                        await value
                except Exception as exc:
                    self._release_errors.append(f"{type(exc).__name__}: {exc}")
            self._tasks.clear()
            self._cleanups.clear()
            self._released = True
            return self.release_errors

    async def _wait_for_deadline(self) -> None:
        if self.deadline_at_ms is None:
            return
        remaining = max(0.0, (self.deadline_at_ms - _now_ms()) / 1000.0)
        await asyncio.sleep(remaining)
        self.cancel("deadline_exceeded")

    async def _force_cancel_after_grace(self) -> None:
        await asyncio.sleep(self.cancel_grace_ms / 1000)
        self._force_cancel_tasks()

    def _force_cancel_tasks(self) -> None:
        current = _current_task()
        for task in tuple(self._tasks):
            if task is not current and not task.done():
                task.cancel()


@dataclass(frozen=True)
class RunEnvironment:
    run_id: str
    session_id: str
    trigger: RunTrigger
    model: Any | None
    tools: Any | None
    context: ContextPort | None
    state: RunStatePort | None
    cancellation: RunCancellationToken
    deadline_at_ms: int | None
    event_sink: Callable[[dict[str, Any]], None] | None
    resources: RunResourceScope
    memory: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _required_text(self.run_id, "run_id"))
        object.__setattr__(self, "session_id", _required_text(self.session_id, "session_id"))
        if self.trigger not in {"prompt", "resume", "continuation"}:
            raise ValueError(f"Unknown run trigger: {self.trigger}")
        if self.cancellation is not self.resources.cancellation:
            raise ValueError("RunEnvironment cancellation must belong to resources")
        if self.deadline_at_ms != self.resources.deadline_at_ms:
            raise ValueError("RunEnvironment deadline must match resources")

    def ports(self) -> AgentLoopPorts:
        return AgentLoopPorts(
            model=self.model,
            tools=self.tools,
            context=self.context,
            state=self.state,
            events=self.event_sink,
            cancellation=self.cancellation,
            deadline_at_ms=self.deadline_at_ms,
        )


class RunEnvironmentFactory:
    def create(
        self,
        prepared: PreparedAgentRun,
        *,
        model: Any | None,
        tools: Any | None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        memory: Any | None = None,
        deadline_at_ms: int | None = None,
    ) -> RunEnvironment:
        trigger: RunTrigger = "resume" if prepared.loop_input.entry == "resume" else "prompt"
        if trigger == "prompt" and prepared.context_refs.get("continuation"):
            trigger = "continuation"
        resources = RunResourceScope(deadline_at_ms=deadline_at_ms)
        return RunEnvironment(
            run_id=prepared.run_id,
            session_id=prepared.session_id,
            trigger=trigger,
            model=model,
            tools=tools,
            context=prepared.context_port,
            state=prepared.state_port,
            cancellation=resources.cancellation,
            deadline_at_ms=deadline_at_ms,
            event_sink=event_sink,
            resources=resources,
            memory=memory,
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _current_task() -> asyncio.Task[Any] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


__all__ = [
    "RunCancellationToken",
    "RunEnvironment",
    "RunEnvironmentFactory",
    "RunResourceScope",
    "RunTrigger",
]
