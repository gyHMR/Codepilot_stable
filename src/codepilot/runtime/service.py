from __future__ import annotations

"""
运行时应用服务层。

RuntimeService 是面向用户接口（CLI、Web）的统一门面，
提供会话管理、消息发送、运行结果查询等高层操作。
"""

import asyncio
import time
import uuid
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, cast

from codepilot.core import AgentEvent
from codepilot.observability import build_run_report
from codepilot.protocols import AgentRunResult, AssistantMessage
from codepilot.sessions.session import AgentSession

from .command_registry import (
    RuntimeCommandResult,
    handle_runtime_command,
    list_runtime_commands,
)
from .factory import assemble_runtime
from .types import (
    ActiveRun,
    CreateAgentSessionOptions,
    RuntimeAssembly,
    SessionHandle,
    SessionStatus,
    UserInput,
)


class RuntimeServiceError(Exception):
    """RuntimeService 错误基类。"""
    code: str = "runtime.internal_error"


class SessionNotFoundError(RuntimeServiceError):
    """会话不存在。"""
    code = "runtime.session_not_found"


class SessionBusyError(RuntimeServiceError):
    """会话正在运行中。"""
    code = "runtime.session_busy"


class EmptyInputError(RuntimeServiceError):
    """输入为空。"""
    code = "runtime.empty_input"


# ── RuntimeService ────────────────────────────────────────────────

class RuntimeService:
    """运行时应用服务（面向用户接口的高层门面）。

    管理多个 AgentSession 的生命周期，提供：
    - 会话创建、获取、列表、关闭
    - 消息发送（同步等待结果 / 异步流式事件）
    - 运行结果和事件查询
    - 会话状态查询（从 RuntimeAssembly 获取）
    - 斜杠命令执行
    - 运行取消
    """

    def __init__(self) -> None:
        # 会话注册表：session_id -> AgentSession
        self._sessions: dict[str, AgentSession] = {}
        # 装配产物注册表：session_id -> RuntimeAssembly
        self._assemblies: dict[str, RuntimeAssembly] = {}
        # 活跃运行：session_id -> ActiveRun（单 Session 单 Run）
        self._active_runs: dict[str, ActiveRun] = {}

    def create_session(self, options: CreateAgentSessionOptions) -> SessionHandle:
        """创建一个新的 Agent 会话。

        Args:
            options: 创建会话选项。

        Returns:
            SessionHandle，包含 session_id、session 实例和装配产物。
        """
        session, assembly = assemble_runtime(options)
        self._sessions[session.session_id] = session
        self._assemblies[session.session_id] = assembly

        return SessionHandle(
            session_id=session.session_id,
            session=session,
            assembly=assembly,
        )

    def _register_derived_session(
        self,
        session: AgentSession,
        *,
        source_session_id: str,
    ) -> SessionHandle:
        """注册由现有会话分支得到的新会话。"""

        source = self.get_assembly(source_session_id)
        session_options = replace(
            source.session_options,
            session_id=session.session_id,
            messages=[],
        )
        assembly = replace(source, session_options=session_options)
        self._sessions[session.session_id] = session
        self._assemblies[session.session_id] = assembly
        return SessionHandle(
            session_id=session.session_id,
            session=session,
            assembly=assembly,
        )

    def get_session(self, session_id: str) -> AgentSession:
        """按 session_id 获取已注册的会话。

        Args:
            session_id: 会话 ID。

        Returns:
            AgentSession 实例。

        Raises:
            SessionNotFoundError: 会话不存在时抛出。
        """
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(f"Session not found: {session_id}") from exc

    def get_assembly(self, session_id: str) -> RuntimeAssembly:
        """获取会话的装配产物。

        Args:
            session_id: 会话 ID。

        Returns:
            RuntimeAssembly 实例。
        """
        self.get_session(session_id)
        try:
            return self._assemblies[session_id]
        except KeyError as exc:
            raise RuntimeServiceError(
                f"Runtime assembly not found for session: {session_id}"
            ) from exc

    def get_session_status(self, session_id: str) -> SessionStatus:
        """获取会话状态信息。

        从 RuntimeAssembly 获取准确的配置信息，而不是从 Session 内部推断。

        Args:
            session_id: 会话 ID。

        Returns:
            SessionStatus 对象，包含模型、工作区、权限等信息。
        """
        session = self.get_session(session_id)
        assembly = self.get_assembly(session_id)
        model = assembly.profile.model
        model_id = f"{model.provider}/{model.id}" if model.provider else model.id
        warnings = [
            diagnostic.message
            for diagnostic in assembly.diagnostics
            if diagnostic.severity == "warning"
        ]

        return SessionStatus(
            session_id=session.session_id,
            model_id=model_id,
            workspace=str(session.workspace_dir),
            permission_mode=assembly.profile.permission_mode,
            message_count=len(session.messages),
            leaf_id=session.get_leaf_id() or "N/A",
            is_running=session_id in self._active_runs,
            credential_source=assembly.profile.credential_source,
            warnings=warnings,
        )

    def get_workspace(self, session_id: str) -> Path:
        """返回会话工作区。"""

        return Path(self.get_assembly(session_id).repository.workspace_root)

    def get_latest_assistant_message(
        self,
        session_id: str,
    ) -> AssistantMessage | None:
        """返回会话最近一条助手消息。"""

        session = self.get_session(session_id)
        return next(
            (
                message
                for message in reversed(session.messages)
                if isinstance(message, AssistantMessage)
            ),
            None,
        )

    def list_sessions(self) -> list[dict[str, str]]:
        """列出所有已注册的会话 ID。"""
        return [{"session_id": session_id} for session_id in sorted(self._sessions)]

    def get_session_state(self, session_id: str) -> dict[str, Any]:
        """返回接口层需要的会话状态。"""

        session = self.get_session(session_id)
        return {
            "session_id": session.session_id,
            "message_count": len(session.messages),
            "entry_ids": session.list_entry_ids(),
            "leaf_id": session.get_leaf_id(),
        }

    def list_session_entries(self, session_id: str) -> list[dict[str, Any]]:
        return self.get_session(session_id).list_entries()

    def get_session_tree(self, session_id: str) -> list[dict[str, Any]]:
        return self.get_session(session_id).get_session_tree()

    def get_entry_path(self, session_id: str, entry_id: str) -> list[str]:
        return self.get_session(session_id).get_entry_path(entry_id)

    def switch_entry(self, session_id: str, entry_id: str) -> None:
        self.get_session(session_id).switch_to_entry(entry_id)

    def fork_session(self, session_id: str, entry_id: str) -> str:
        forked = self.get_session(session_id).fork_from_entry(entry_id)
        self._register_derived_session(
            forked,
            source_session_id=session_id,
        )
        return forked.session_id

    def list_commands(self, session_id: str) -> list[dict[str, str]]:
        return [
            {
                "name": command.name,
                "description": command.description,
                "source": command.source,
            }
            for command in list_runtime_commands(self.get_session(session_id))
        ]

    async def run_message(self, session_id: str, message: UserInput) -> AgentRunResult:
        """发送消息并等待完整结果返回（非流式）。

        Args:
            session_id: 目标会话 ID。
            message: 用户输入。

        Returns:
            AgentRunResult 运行结果。

        Raises:
            SessionNotFoundError: 会话不存在。
            SessionBusyError: 会话正在运行中。
            EmptyInputError: 输入为空。
        """
        # 请求校验
        self._validate_request(session_id, message)

        session = self.get_session(session_id)
        active_run = self._create_active_run(session_id)
        active_run.task = asyncio.current_task()

        try:
            result = await session.run(message.text, images=message.images)
            active_run.status = "completed"
            return result
        except asyncio.CancelledError:
            active_run.status = "aborted"
            raise
        except Exception:
            active_run.status = "failed"
            raise
        finally:
            self._active_runs.pop(session_id, None)

    async def send_message(self, session_id: str, message: UserInput) -> AsyncIterator[AgentEvent]:
        """发送消息并以异步迭代器方式流式返回事件。

        Args:
            session_id: 目标会话 ID。
            message: 用户输入。

        Yields:
            AgentEvent 事件对象。

        Raises:
            SessionNotFoundError: 会话不存在。
            SessionBusyError: 会话正在运行中。
            EmptyInputError: 输入为空。
        """
        # 请求校验
        self._validate_request(session_id, message)

        session = self.get_session(session_id)
        active_run = self._create_active_run(session_id)

        try:
            async for event in self._stream_session_events(
                session,
                lambda: session.run(message.text, images=message.images),
                active_run=active_run,
            ):
                yield event

            active_run.status = "completed"
        except asyncio.CancelledError:
            active_run.status = "aborted"
            raise
        except Exception:
            active_run.status = "failed"
            raise
        finally:
            self._active_runs.pop(session_id, None)

    async def continue_session(self, session_id: str) -> AsyncIterator[AgentEvent]:
        """继续上一次未完成的会话运行，流式返回事件。

        Args:
            session_id: 目标会话 ID。

        Yields:
            AgentEvent 事件对象。
        """
        self.get_session(session_id)
        active = self._active_runs.get(session_id)
        if active is not None and active.task is not None and active.task.done():
            self._active_runs.pop(session_id, None)
            active = None
        if active is not None:
            raise SessionBusyError(f"Session {session_id} is already running")

        session = self.get_session(session_id)
        active_run = self._create_active_run(session_id)

        try:
            async for event in self._stream_session_events(
                session,
                session.continue_run,
                active_run=active_run,
            ):
                yield event
            active_run.status = "completed"
        except asyncio.CancelledError:
            active_run.status = "aborted"
            raise
        except Exception:
            active_run.status = "failed"
            raise
        finally:
            self._active_runs.pop(session_id, None)

    async def execute_command(self, session_id: str, text: str) -> RuntimeCommandResult:
        """执行斜杠命令。

        Args:
            session_id: 目标会话 ID。
            text: 命令文本（如 "/status"）。

        Returns:
            RuntimeCommandResult 命令执行结果。
        """
        if text.strip() == "/status":
            status = self.get_session_status(session_id)
            return RuntimeCommandResult(
                handled=True,
                output_lines=[
                    "=== Status ===",
                    f"  Model      : {status.model_id}",
                    f"  Workspace  : {status.workspace}",
                    f"  Session    : {status.session_id}",
                    f"  Leaf       : {status.leaf_id}",
                    f"  Messages   : {status.message_count}",
                    f"  Permission : {status.permission_mode}",
                ],
            )

        session = self.get_session(session_id)
        result = await handle_runtime_command(session, text)
        if result.switched_session is not None:
            replacement = result.switched_session
            self._register_derived_session(
                replacement,
                source_session_id=session_id,
            )
            result.switched_session_id = replacement.session_id
            result.switched_session = None
        return result

    async def cancel_run(self, session_id: str) -> bool:
        """取消当前正在运行的任务。

        Args:
            session_id: 目标会话 ID。

        Returns:
            是否成功取消（True 表示取消，False 表示没有运行中的任务）。
        """
        active_run = self._active_runs.get(session_id)
        if active_run is None or (active_run.task is not None and active_run.task.done()):
            self._active_runs.pop(session_id, None)
            return False

        if active_run.task is not None:
            active_run.task.cancel()
            try:
                await active_run.task
            except asyncio.CancelledError:
                pass
            finally:
                active_run.status = "aborted"
                self._active_runs.pop(session_id, None)
        return True

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

    def _validate_request(self, session_id: str, message: UserInput) -> None:
        """校验请求。

        Args:
            session_id: 会话 ID。
            message: 用户输入。

        Raises:
            SessionNotFoundError: 会话不存在。
            SessionBusyError: 会话正在运行中。
            EmptyInputError: 输入为空。
        """
        # 检查会话是否存在
        self.get_session(session_id)

        # 检查输入是否为空
        if not message.text or not message.text.strip():
            raise EmptyInputError("Input text is empty")

        # 检查是否正在运行
        active_run = self._active_runs.get(session_id)
        if active_run is not None and active_run.task is not None and active_run.task.done():
            self._active_runs.pop(session_id, None)
            active_run = None
        if active_run is not None:
            raise SessionBusyError(f"Session {session_id} is already running")

    def _create_active_run(self, session_id: str) -> ActiveRun:
        """创建 ActiveRun。

        Args:
            session_id: 会话 ID。

        Returns:
            ActiveRun 实例。
        """
        run_id = uuid.uuid4().hex
        active_run = ActiveRun(
            run_id=run_id,
            session_id=session_id,
            started_at=time.time(),
        )
        self._active_runs[session_id] = active_run
        return active_run

    async def _stream_session_events(
        self,
        session: AgentSession,
        run: Callable[[], Awaitable[object]],
        active_run: ActiveRun | None = None,
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
            active_run: 活跃运行实例（可选）。

        Yields:
            AgentEvent 事件对象。
        """
        done_marker = object()
        queue: asyncio.Queue[AgentEvent | object] = asyncio.Queue()

        def _listener(event: AgentEvent) -> None:
            queue.put_nowait(event)

        unsubscribe = session.subscribe(_listener)
        task = asyncio.create_task(run())

        # 更新 ActiveRun 的 task
        if active_run:
            active_run.task = task

        task.add_done_callback(lambda _: queue.put_nowait(done_marker))

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
        """同步关闭空闲会话；运行中的会话应使用 aclose_session。"""
        active_run = self._active_runs.get(session_id)
        if active_run is not None and (
            active_run.task is None or not active_run.task.done()
        ):
            raise SessionBusyError(
                f"Cannot close running session: {session_id}"
            )
        self._active_runs.pop(session_id, None)

        # 移除装配产物
        self._assemblies.pop(session_id, None)

        # 关闭会话
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    async def aclose_session(self, session_id: str) -> None:
        """取消并等待当前运行结束，然后关闭指定会话。"""

        await self.cancel_run(session_id)
        self.close_session(session_id)

    def close_all(self) -> None:
        """同步关闭所有空闲会话；存在运行任务时拒绝关闭。"""

        running_session_ids = [
            session_id
            for session_id, active_run in self._active_runs.items()
            if active_run.task is None or not active_run.task.done()
        ]
        if running_session_ids:
            raise SessionBusyError(
                "Cannot close RuntimeService with running sessions: "
                + ", ".join(sorted(running_session_ids))
            )
        for session_id in list(self._sessions):
            self.close_session(session_id)

    async def aclose_all(self) -> None:
        """取消并等待全部运行任务结束，然后关闭所有会话。"""

        for session_id in list(self._active_runs):
            await self.cancel_run(session_id)
        for session_id in list(self._sessions):
            self.close_session(session_id)
