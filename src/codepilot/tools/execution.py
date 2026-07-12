"""规范的超时、取消、清理和基本调度原语。

本文件提供工具执行的运行时基础设施：
1. CancellationToken    — 取消令牌（传播取消信号到工具处理器）
2. ProgressReporter    — 进度报告器（工具处理器发送中间状态）
3. CleanupStack        — 清理栈（注册和运行清理回调）
4. ExecutionController — 执行控制器（管理并发、超时、取消）
5. 各类控制异常        — 超时、取消、队列满、队列超时

ExecutionController 是核心：协调 asyncio 任务、信号量、锁
来实现工具执行的并发控制策略（并行、串行、限流）。
"""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from .security import ConcurrencyPolicy, TimeoutPolicy


# ── 异常类 ────────────────────────────────────────────────────────────────────


class ToolExecutionControlError(RuntimeError):
    """工具执行控制错误 —— 执行控制相关的异常基类。"""


class ToolQueueFullError(ToolExecutionControlError):
    """队列已满 —— 超过 max_pending_per_session 限制时抛出。"""


class ToolQueueTimeoutError(ToolExecutionControlError):
    """队列超时 —— 在队列中等待执行的时间超过截止时间时抛出。"""


class ToolExecutionTimeoutError(ToolExecutionControlError):
    """执行超时 —— 工具执行时间超过超时策略限制时抛出。"""


class ToolExecutionCancelledError(ToolExecutionControlError):
    """执行取消 —— 工具在执行过程中被取消时抛出。"""


# ── 运行时限制 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolRuntimeLimits:
    """工具运行时限制 —— 控制并行度和队列深度的全局限制。

    参数:
        max_parallel_per_session: 每个会话的最大并行执行数（默认 4）
        max_pending_per_session: 每个会话的最大待处理数（默认 32）
    """

    max_parallel_per_session: int = 4
    max_pending_per_session: int = 32

    def __post_init__(self) -> None:
        for name in ("max_parallel_per_session", "max_pending_per_session"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


# ── 取消令牌 ──────────────────────────────────────────────────────────────────


class CancellationToken:
    """取消令牌 —— 工具处理器检查是否被取消的通信对象。

    由 ExecutionController 在工具执行开始时创建并传递给工具处理器。
    工具处理器在长操作中应定期检查 cancelled 属性或调用 raise_if_cancelled()。
    """

    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        """检查是否已被取消。"""
        return self._cancelled

    def cancel(self) -> None:
        """标记取消状态。"""
        self._cancelled = True

    def raise_if_cancelled(self) -> None:
        """如果已被取消，抛出 asyncio.CancelledError。

        工具处理器应在每个重要的检查点调用此方法。
        """
        if self._cancelled:
            raise asyncio.CancelledError


# ── 进度报告器 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolProgressEvent:
    """工具进度事件 —— 工具执行过程中的中间状态通知。

    参数:
        attempt_id: 所属工具尝试的 ID
        kind: 事件类型（如 "reading"、"writing"、"searching"）
        message: 人类可读的消息
        data: 事件相关的结构化数据
        timestamp_ms: 事件发生时间戳
    """

    attempt_id: str
    kind: str
    message: str
    data: Mapping[str, object]
    timestamp_ms: int


class ProgressReporter:
    """进度报告器 —— 工具处理器发送进度事件的通信对象。

    工具处理器可以通过 report() 发送中间状态事件。
    这些事件会被 ExecutionController 转发到 progress_callback，
    再由上层（如 model_step）发送给用户显示进度。
    """

    def __init__(
        self,
        attempt_id: str,
        callback: Callable[[ToolProgressEvent], Awaitable[None] | None] | None = None,
    ) -> None:
        self._attempt_id = attempt_id
        self._callback = callback

    async def report(
        self,
        kind: str,
        *,
        message: str = "",
        data: Mapping[str, object] | None = None,
    ) -> None:
        """发送一个进度事件。

        参数:
            kind: 事件类型标签（如 "reading", "writing"）
            message: 文本描述
            data: 结构化数据（可选）
        """
        event = ToolProgressEvent(
            attempt_id=self._attempt_id,
            kind=str(kind).strip() or "progress",
            message=str(message),
            data=dict(data or {}),
            timestamp_ms=int(time.time() * 1000),
        )
        if self._callback is None:
            return
        value = self._callback(event)
        if inspect.isawaitable(value):
            await value


# ── 清理栈 ────────────────────────────────────────────────────────────────────


class CleanupStack:
    """清理栈 —— 注册和运行工具清理回调。

    工具处理器可以通过 push() 注册清理回调（如关闭临时文件句柄），
    工具执行完成后这些回调会被逆序执行（后进先出）。

    即使工具因错误而终止，清理回调也会被执行。
    """

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], Awaitable[None] | None]] = []
        self._errors: list[str] = []

    @property
    def errors(self) -> tuple[str, ...]:
        """返回清理过程中发生的错误列表。"""
        return tuple(self._errors)

    def push(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        """注册一个清理回调。

        回调会在工具执行完成后（包括异常和取消时）被调用。
        多个回调会按逆序执行（后注册的先执行）。

        参数:
            callback: 无参的可调用对象，可以是同步或异步
        """
        if not callable(callback):
            raise TypeError("cleanup callback must be callable")
        self._callbacks.append(callback)

    async def close(self) -> None:
        """执行所有已注册的清理回调（逆序）。

        即使部分回调失败，也会继续执行剩余的回调。
        失败信息收集在 errors 属性中。
        """
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


# ── 内部执行管理 ──────────────────────────────────────────────────────────────


@dataclass
class _ActiveExecution:
    """内部数据结构 —— 记录一个正在执行的操作。

    用于在需要时取消正在运行的工具。
    """
    token: CancellationToken
    task: asyncio.Task
    cleanup_grace_ms: int


@dataclass
class _Lease:
    """内部数据结构 —— 执行权限租赁。

    管理一组信号量/锁的释放，确保所有获取的并发资源
    在工具执行完成后被正确归还。
    """
    semaphore: asyncio.Semaphore
    group_semaphore: asyncio.Semaphore | None = None
    serial_lock: asyncio.Lock | None = None

    def release(self) -> None:
        """释放所有已获取的并发资源。

        按逆序释放：serial_lock → group_semaphore → semaphore
        """
        if self.serial_lock is not None and self.serial_lock.locked():
            self.serial_lock.release()
        if self.group_semaphore is not None:
            self.group_semaphore.release()
        self.semaphore.release()


# ── 执行控制器 ────────────────────────────────────────────────────────────────


class ExecutionController:
    """执行控制器 —— 管理工具执行的并发、超时和取消。

    这是工具执行的核心调度器，负责：
    1. 并发控制 —— 每个会话限制并行数，支持串行/并行模式
    2. 组级限流 —— 同一组内限制最大并行度
    3. 超时管理 —— 软超时（asyncio.wait_for）+ 硬超时（清理宽限）
    4. 取消支持 —— 通过 CancellationToken 传播取消信号
    5. 队列管理 —— 待处理队列深度限制

    ExecutionController 与 ToolRuntime 协作：
    - ToolRuntime 负责"策略层"：权限检查、状态管理
    - ExecutionController 负责"调度层"：并发、超时、取消

    参数:
        limits: 运行时限制（并行度、队列深度）
    """

    def __init__(self, limits: ToolRuntimeLimits | None = None) -> None:
        self.limits = limits or ToolRuntimeLimits()
        # 每个会话的信号量（控制 max_parallel_per_session）
        self._session_semaphores: dict[str, asyncio.Semaphore] = {}
        # 串行锁（同组内的工具串行执行）
        self._serial_locks: dict[tuple[str, str], asyncio.Lock] = {}
        # 组级限流信号量
        self._group_semaphores: dict[tuple[str, str, int], asyncio.Semaphore] = {}
        # 每个会话的待处理计数
        self._pending: dict[str, int] = {}
        # 当前活跃的执行
        self._active: dict[str, _ActiveExecution] = {}
        # 清理错误
        self._cleanup_errors: dict[str, tuple[str, ...]] = {}

    async def run(
        self,
        *,
        attempt_id: str,
        session_id: str,
        timeout: TimeoutPolicy,
        requested_execution_ms: int | None,
        concurrency: ConcurrencyPolicy,
        request_deadline_at_ms: int | None,
        progress_callback: Callable[[ToolProgressEvent], Awaitable[None] | None] | None,
        operation: ExecutionOperation[TOutput],
    ) -> TOutput:
        """执行一个工具操作（带并发控制、超时和取消支持）。

        执行流程：
        1. 计算有效截止时间（结合策略超时和请求级别截止时间）
        2. 获取并发租赁（信号量/锁）
        3. 创建 CancellationToken、ProgressReporter、CleanupStack
        4. 创建 asyncio 任务执行操作
        5. 在超时/取消时执行清理

        参数:
            attempt_id: 工具尝试的 ID
            session_id: 会话 ID（用于并发控制）
            timeout: 超时策略
            requested_execution_ms: 请求的执行超时（None 使用默认值）
            concurrency: 并发策略
            request_deadline_at_ms: 请求级别的截止时间
            progress_callback: 进度事件回调
            operation: 实际执行的操作

        返回:
            操作的结果

        抛出:
            ToolQueueFullError: 队列已满
            ToolQueueTimeoutError: 队列等待超时
            ToolExecutionTimeoutError: 执行超时
            ToolExecutionCancelledError: 执行取消
        """
        deadline_at_ms = _effective_deadline(
            timeout,
            requested_execution_ms,
            request_deadline_at_ms,
        )
        lease = await self._acquire(session_id, concurrency, deadline_at_ms)
        token = CancellationToken()
        progress = ProgressReporter(attempt_id, progress_callback)
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
        """取出并清除指定 attempt 的清理错误。

        参数:
            attempt_id: 工具尝试的 ID

        返回:
            清理错误名称的元组
        """
        return self._cleanup_errors.pop(attempt_id, ())

    async def cancel(self, attempt_id: str) -> bool:
        """取消正在执行的工具。

        设置 CancellationToken 并等待任务终止。

        参数:
            attempt_id: 要取消的工具尝试 ID

        返回:
            True 表示成功取消，False 表示未找到对应的运行
        """
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
        """获取执行租赁 —— 获取所有必要的并发资源。

        获取顺序：
        1. 检查待处理队列是否已满
        2. 递增待处理计数
        3. 获取会话级信号量（限制每个会话的并行数）
        4. 如果需要，获取组级限流信号量（max_parallel）
        5. 如果需要，获取串行锁

        参数:
            session_id: 会话 ID
            concurrency: 并发策略
            deadline_at_ms: 截止时间

        返回:
            _Lease 包含所有需要释放的资源
        """
        pending = self._pending.get(session_id, 0)
        if pending >= self.limits.max_pending_per_session:
            raise ToolQueueFullError("Tool queue is full")
        self._pending[session_id] = pending + 1
        semaphore = self._session_semaphores.setdefault(
            session_id,
            asyncio.Semaphore(self.limits.max_parallel_per_session),
        )
        serial_lock = None
        group_semaphore = None
        semaphore_acquired = False
        group_acquired = False
        try:
            await _wait_acquire(semaphore, deadline_at_ms)
            semaphore_acquired = True
            if concurrency.max_parallel is not None:
                group = concurrency.group or "group"
                group_semaphore = self._group_semaphores.setdefault(
                    (session_id, group, concurrency.max_parallel),
                    asyncio.Semaphore(concurrency.max_parallel),
                )
                await _wait_acquire(group_semaphore, deadline_at_ms)
                group_acquired = True
            if concurrency.mode == "serial":
                group = concurrency.group or "serial"
                serial_lock = self._serial_locks.setdefault((session_id, group), asyncio.Lock())
                await _wait_acquire(serial_lock, deadline_at_ms)
            return _Lease(
                semaphore=semaphore,
                group_semaphore=group_semaphore,
                serial_lock=serial_lock,
            )
        except asyncio.TimeoutError as exc:
            if group_acquired and group_semaphore is not None:
                group_semaphore.release()
            if semaphore_acquired:
                semaphore.release()
            raise ToolQueueTimeoutError("Tool queue wait timed out") from exc
        finally:
            self._pending[session_id] = max(0, self._pending.get(session_id, 1) - 1)


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────


def _effective_deadline(
    timeout: TimeoutPolicy,
    requested_execution_ms: int | None,
    request_deadline_at_ms: int | None,
) -> int:
    """计算有效的执行截止时间。

    结合三个因素：
    1. 策略的默认/最大超时
    2. 用户请求的执行超时
    3. 请求级别的截止时间

    取最严格的（最小的）作为最终截止时间。
    """
    now_ms = int(time.time() * 1000)
    execution_ms = (
        timeout.default_execution_ms
        if requested_execution_ms is None
        else requested_execution_ms
    )
    policy_deadline = now_ms + min(execution_ms, timeout.max_execution_ms)
    if request_deadline_at_ms is None:
        return policy_deadline
    return min(policy_deadline, request_deadline_at_ms)


def _remaining_ms(deadline_at_ms: int) -> int:
    """计算距离截止时间还有多少毫秒。"""
    return max(0, deadline_at_ms - int(time.time() * 1000))


async def _wait_acquire(lock, deadline_at_ms: int) -> None:
    """带超时的锁/信号量获取。"""
    remaining_ms = _remaining_ms(deadline_at_ms)
    if remaining_ms <= 0:
        raise asyncio.TimeoutError
    await asyncio.wait_for(lock.acquire(), timeout=remaining_ms / 1_000)


async def _stop_task(task: asyncio.Task, grace_ms: int) -> None:
    """停止一个 asyncio 任务（带宽限时间）。

    策略：
    1. 如果任务已完成，直接等待（收集可能的异常）
    2. 如果任务还在运行，给 grace_ms 宽限时间让其正常结束
    3. 宽限时间到仍未结束，调用 task.cancel()
    """
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