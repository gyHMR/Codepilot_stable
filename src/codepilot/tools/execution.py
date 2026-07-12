from __future__ import annotations

"""Canonical timeout, cancellation, cleanup and basic scheduling primitives."""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .security import ConcurrencyPolicy, TimeoutPolicy


class ToolExecutionControlError(RuntimeError):
    pass


class ToolQueueFullError(ToolExecutionControlError):
    pass


class ToolQueueTimeoutError(ToolExecutionControlError):
    pass


class ToolExecutionTimeoutError(ToolExecutionControlError):
    pass


class ToolExecutionCancelledError(ToolExecutionControlError):
    pass


@dataclass(frozen=True)
class ToolRuntimeLimits:
    max_parallel_per_session: int = 4
    max_pending_per_session: int = 32

    def __post_init__(self) -> None:
        for name in ("max_parallel_per_session", "max_pending_per_session"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError


@dataclass(frozen=True)
class ToolProgressEvent:
    attempt_id: str
    kind: str
    message: str
    data: Mapping[str, object]
    timestamp_ms: int


class ProgressReporter:
    def __init__(
        self,
        attempt_id: str,
        callback: Callable[[ToolProgressEvent], Awaitable[None] | None] | None = None,
    ) -> None:
        self._attempt_id = attempt_id
        self._callback = callback
        self._events: list[ToolProgressEvent] = []

    @property
    def events(self) -> tuple[ToolProgressEvent, ...]:
        return tuple(self._events)

    async def report(
        self,
        kind: str,
        *,
        message: str = "",
        data: Mapping[str, object] | None = None,
    ) -> None:
        event = ToolProgressEvent(
            attempt_id=self._attempt_id,
            kind=str(kind).strip() or "progress",
            message=str(message),
            data=dict(data or {}),
            timestamp_ms=int(time.time() * 1000),
        )
        self._events.append(event)
        if self._callback is None:
            return
        value = self._callback(event)
        if inspect.isawaitable(value):
            await value


class CleanupStack:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[], Awaitable[None] | None]] = []
        self._errors: list[str] = []

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    def push(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        if not callable(callback):
            raise TypeError("cleanup callback must be callable")
        self._callbacks.append(callback)

    async def close(self) -> None:
        while self._callbacks:
            callback = self._callbacks.pop()
            try:
                value = callback()
                if inspect.isawaitable(value):
                    await value
            except Exception as exc:
                self._errors.append(type(exc).__name__)


TOutput = TypeVar("TOutput")
ExecutionOperation = Callable[
    [CancellationToken, ProgressReporter, CleanupStack],
    Awaitable[TOutput],
]


@dataclass
class _ActiveExecution:
    token: CancellationToken
    task: asyncio.Task
    cleanup_grace_ms: int


@dataclass
class _Lease:
    semaphore: asyncio.Semaphore
    serial_lock: asyncio.Lock | None = None

    def release(self) -> None:
        if self.serial_lock is not None and self.serial_lock.locked():
            self.serial_lock.release()
        self.semaphore.release()


class ExecutionController:
    def __init__(self, limits: ToolRuntimeLimits | None = None) -> None:
        self.limits = limits or ToolRuntimeLimits()
        self._session_semaphores: dict[str, asyncio.Semaphore] = {}
        self._serial_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._pending: dict[str, int] = {}
        self._active: dict[str, _ActiveExecution] = {}
        self._cleanup_errors: dict[str, tuple[str, ...]] = {}

    async def run(
        self,
        *,
        attempt_id: str,
        session_id: str,
        timeout: TimeoutPolicy,
        concurrency: ConcurrencyPolicy,
        request_deadline_at_ms: int | None,
        operation: ExecutionOperation[TOutput],
    ) -> TOutput:
        deadline_at_ms = _effective_deadline(timeout, request_deadline_at_ms)
        lease = await self._acquire(session_id, concurrency, deadline_at_ms)
        token = CancellationToken()
        progress = ProgressReporter(attempt_id)
        cleanup = CleanupStack()
        task: asyncio.Task | None = None
        try:
            remaining_ms = _remaining_ms(deadline_at_ms)
            if remaining_ms <= 0:
                raise ToolExecutionTimeoutError("Tool execution deadline expired")
            task = asyncio.create_task(operation(token, progress, cleanup))
            self._active[attempt_id] = _ActiveExecution(
                token=token,
                task=task,
                cleanup_grace_ms=timeout.cleanup_grace_ms,
            )
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=remaining_ms / 1_000,
                )
            except asyncio.TimeoutError as exc:
                token.cancel()
                await _stop_task(task, timeout.cleanup_grace_ms)
                raise ToolExecutionTimeoutError("Tool execution timed out") from exc
            except asyncio.CancelledError as exc:
                token.cancel()
                await _stop_task(task, timeout.cleanup_grace_ms)
                raise ToolExecutionCancelledError("Tool execution cancelled") from exc
        finally:
            self._active.pop(attempt_id, None)
            try:
                await asyncio.wait_for(
                    cleanup.close(),
                    timeout=max(0.001, timeout.cleanup_grace_ms / 1_000),
                )
            except asyncio.TimeoutError:
                pass
            if cleanup.errors:
                self._cleanup_errors[attempt_id] = cleanup.errors
            lease.release()

    def take_cleanup_errors(self, attempt_id: str) -> tuple[str, ...]:
        return self._cleanup_errors.pop(attempt_id, ())

    async def cancel(self, attempt_id: str) -> bool:
        active = self._active.get(attempt_id)
        if active is None:
            return False
        active.token.cancel()
        await _stop_task(active.task, active.cleanup_grace_ms)
        return True

    async def _acquire(
        self,
        session_id: str,
        concurrency: ConcurrencyPolicy,
        deadline_at_ms: int,
    ) -> _Lease:
        pending = self._pending.get(session_id, 0)
        if pending >= self.limits.max_pending_per_session:
            raise ToolQueueFullError("Tool queue is full")
        self._pending[session_id] = pending + 1
        semaphore = self._session_semaphores.setdefault(
            session_id,
            asyncio.Semaphore(self.limits.max_parallel_per_session),
        )
        serial_lock = None
        semaphore_acquired = False
        try:
            await _wait_acquire(semaphore, deadline_at_ms)
            semaphore_acquired = True
            if concurrency.mode == "serial":
                group = concurrency.group or "serial"
                serial_lock = self._serial_locks.setdefault((session_id, group), asyncio.Lock())
                await _wait_acquire(serial_lock, deadline_at_ms)
            return _Lease(semaphore=semaphore, serial_lock=serial_lock)
        except asyncio.TimeoutError as exc:
            if semaphore_acquired:
                semaphore.release()
            raise ToolQueueTimeoutError("Tool queue wait timed out") from exc
        finally:
            self._pending[session_id] = max(0, self._pending.get(session_id, 1) - 1)


def _effective_deadline(timeout: TimeoutPolicy, request_deadline_at_ms: int | None) -> int:
    now_ms = int(time.time() * 1000)
    policy_deadline = now_ms + min(timeout.default_execution_ms, timeout.max_execution_ms)
    if request_deadline_at_ms is None:
        return policy_deadline
    return min(policy_deadline, request_deadline_at_ms)


def _remaining_ms(deadline_at_ms: int) -> int:
    return max(0, deadline_at_ms - int(time.time() * 1000))


async def _wait_acquire(lock, deadline_at_ms: int) -> None:
    remaining_ms = _remaining_ms(deadline_at_ms)
    if remaining_ms <= 0:
        raise asyncio.TimeoutError
    await asyncio.wait_for(lock.acquire(), timeout=remaining_ms / 1_000)


async def _stop_task(task: asyncio.Task, grace_ms: int) -> None:
    if task.done():
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=max(0.001, grace_ms / 1_000))
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    except (asyncio.CancelledError, Exception):
        pass


__all__ = [
    "CancellationToken",
    "CleanupStack",
    "ExecutionController",
    "ProgressReporter",
    "ToolExecutionCancelledError",
    "ToolExecutionControlError",
    "ToolExecutionTimeoutError",
    "ToolProgressEvent",
    "ToolQueueFullError",
    "ToolQueueTimeoutError",
    "ToolRuntimeLimits",
]
