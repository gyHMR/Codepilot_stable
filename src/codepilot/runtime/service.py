from __future__ import annotations

"""
运行时应用服务层。

RuntimeService 是面向用户接口（CLI、Web、IM）的统一门面，
提供会话管理、消息发送、运行结果查询等高层操作。

内部通过 asyncio.Queue 实现事件流式转发，
将 Agent 事件实时推送给调用方。
"""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, cast

from codepilot.core import AgentEvent
from codepilot.observability import build_run_report
from codepilot.protocols import AgentRunResult
from codepilot.sessions.session import AgentSession

from .factory import create_agent_session
from .types import CreateAgentSessionOptions


@dataclass(frozen=True)
class CreateSessionRequest:
    """创建会话的请求封装。"""
    options: CreateAgentSessionOptions


@dataclass(frozen=True)
class SessionHandle:
    """会话句柄（创建会话后返回给调用方）。

    Attributes:
        session_id: 会话唯一 ID。
        session: AgentSession 实例。
    """

    session_id: str
    session: AgentSession


@dataclass(frozen=True)
class UserInput:
    """用户输入封装。

    Attributes:
        text: 用户输入的文本。
        images: 可选的图片列表（base64 编码）。
    """

    text: str
    images: list[str] | None = None


class RuntimeService:
    """运行时应用服务（面向用户接口的高层门面）。

    管理多个 AgentSession 的生命周期，提供：
    - 会话创建、获取、列表、关闭
    - 消息发送（同步等待结果 / 异步流式事件）
    - 运行结果和事件查询
    - 运行报告生成

    使用示例::

        service = RuntimeService()
        handle = service.create_session(CreateAgentSessionRequest(options=...))

        # 流式发送消息
        async for event in service.send_message(handle.session_id, UserInput(text="hello")):
            print(event["type"])

        # 查询运行结果
        runs = service.list_runs(handle.session_id)
    """

    def __init__(self) -> None:
        # 会话注册表：session_id -> AgentSession
        self._sessions: dict[str, AgentSession] = {}

    def create_session(self, request: CreateSessionRequest | CreateAgentSessionOptions) -> SessionHandle:
        """创建一个新的 Agent 会话。

        Args:
            request: 创建会话的请求（CreateSessionRequest 或直接传 CreateAgentSessionOptions）。

        Returns:
            SessionHandle，包含 session_id 和 session 实例。
        """
        options = request.options if isinstance(request, CreateSessionRequest) else request
        session = create_agent_session(options)
        self._sessions[session.session_id] = session
        return SessionHandle(session_id=session.session_id, session=session)

    def get_session(self, session_id: str) -> AgentSession:
        """按 session_id 获取已注册的会话。

        Args:
            session_id: 会话 ID。

        Returns:
            AgentSession 实例。

        Raises:
            ValueError: 会话不存在时抛出。
        """
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise ValueError(f"Session not found: {session_id}") from exc

    def list_sessions(self) -> list[dict[str, str]]:
        """列出所有已注册的会话 ID。"""
        return [{"session_id": session_id} for session_id in sorted(self._sessions)]

    async def run_message(self, session_id: str, message: UserInput) -> AgentRunResult:
        """发送消息并等待完整结果返回（非流式）。

        Args:
            session_id: 目标会话 ID。
            message: 用户输入。

        Returns:
            AgentRunResult 运行结果。
        """
        session = self.get_session(session_id)
        return await session.run(message.text, images=message.images)

    async def send_message(self, session_id: str, message: UserInput) -> AsyncIterator[AgentEvent]:
        """发送消息并以异步迭代器方式流式返回事件。

        Args:
            session_id: 目标会话 ID。
            message: 用户输入。

        Yields:
            AgentEvent 事件对象。
        """
        session = self.get_session(session_id)
        async for event in self._stream_session_events(
            session,
            lambda: session.run(message.text, images=message.images),
        ):
            yield event

    async def continue_session(self, session_id: str) -> AsyncIterator[AgentEvent]:
        """继续上一次未完成的会话运行，流式返回事件。

        Args:
            session_id: 目标会话 ID。

        Yields:
            AgentEvent 事件对象。
        """
        session = self.get_session(session_id)
        async for event in self._stream_session_events(session, session.continue_run):
            yield event

    def list_runs(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """列出会话的运行记录。

        Args:
            session_id: 目标会话 ID。
            limit: 返回的最大记录数。

        Returns:
            运行记录字典列表。
        """
        session = self.get_session(session_id)
        return session.store.load_run_results(limit=limit)

    def get_run_result(self, session_id: str, run_id: str) -> dict[str, Any]:
        """获取指定运行的完整结果。"""
        session = self.get_session(session_id)
        return session.store.run_store.load_run_result(run_id)

    def get_run_events(self, session_id: str, run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """获取指定运行的事件列表。"""
        session = self.get_session(session_id)
        return session.store.run_store.load_events(run_id, limit=limit)

    def get_run_report(self, session_id: str, run_id: str) -> dict[str, Any]:
        """生成指定运行的报告（结果 + 事件的汇总）。"""
        result = self.get_run_result(session_id, run_id)
        events = self.get_run_events(session_id, run_id)
        return build_run_report(result, events=events)

    async def _stream_session_events(
        self,
        session: AgentSession,
        run: Callable[[], Awaitable[object]],
    ) -> AsyncIterator[AgentEvent]:
        """内部方法：将 Agent 事件通过 asyncio.Queue 流式转发给调用方。

        流程：
        1. 订阅 session 的事件监听器。
        2. 在后台 Task 中执行 run()。
        3. 监听器将事件推入队列。
        4. 从队列中取出事件并 yield 给调用方。
        5. run() 完成后推送结束标记，退出循环。

        Args:
            session: AgentSession 实例。
            run: 要执行的异步操作（通常是 session.run 或 session.continue_run）。

        Yields:
            AgentEvent 事件对象。
        """
        done_marker = object()
        queue: asyncio.Queue[AgentEvent | object] = asyncio.Queue()

        def _listener(event: AgentEvent) -> None:
            queue.put_nowait(event)

        unsubscribe = session.subscribe(_listener)
        task = asyncio.create_task(run())
        task.add_done_callback(lambda _task: queue.put_nowait(done_marker))

        try:
            while True:
                item = await queue.get()
                if item is done_marker:
                    break
                yield cast(AgentEvent, item)
            await task
        finally:
            unsubscribe()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def approve_tool_call(self, approval_id: str, decision: str) -> None:
        """审批工具调用（尚未实现）。"""
        _ = approval_id, decision
        raise NotImplementedError("Tool approval flow is not implemented yet")

    def close_session(self, session_id: str) -> None:
        """关闭并移除指定会话。"""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def close_all(self) -> None:
        """关闭并移除所有会话。"""
        for session_id in list(self._sessions):
            self.close_session(session_id)
