from __future__ import annotations

"""
运行时应用服务层。

RuntimeService 是面向用户接口（CLI、Web）的统一门面，
提供会话管理、消息发送、运行结果查询等高层操作。
"""

import asyncio
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, cast

from codepilot.observability import (
    AuditBundle,
    build_run_report,
    load_audit_bundle,
)
from codepilot.protocols import (
    AgentEvent,
    AgentRunResult,
    AssistantMessage,
    TextContent,
    ToolResultMessage,
)
from codepilot.sessions.memory import load_global_memory
from codepilot.sessions.session import AgentSession
from codepilot.tools import AgentTool, AgentToolResult

from .execution.approval import (
    PendingApproval,
    build_pending_approvals,
    denied_tool_result,
    normalize_approval_decision,
    to_tool_result_message,
)
from .execution.runs import ActiveRun as _ActiveRun
from .execution.runs import ActiveRunTracker
from .assembly import assemble_runtime
from .contracts import (
    CreateAgentSessionOptions as _CreateAgentSessionOptions,
    RuntimeAssembly as _RuntimeAssembly,
    SessionHandle as _SessionHandle,
    SessionStatus as _SessionStatus,
    UserInput as _UserInput,
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


class ApprovalNotFoundError(RuntimeServiceError):
    """找不到待审批工具调用。"""
    code = "runtime.approval_not_found"


class InvalidApprovalDecisionError(RuntimeServiceError):
    """审批决策值无效。"""
    code = "runtime.invalid_approval_decision"


def _normalize_approval_decision(decision: str) -> str:
    normalized = normalize_approval_decision(decision)
    if normalized is not None:
        return normalized
    raise InvalidApprovalDecisionError(f"Invalid approval decision: {decision}")


def _session_images(message: _UserInput) -> list[str] | None:
    """Convert immutable runtime input images into the Session API shape."""

    return list(message.images) if message.images is not None else None


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
        self._assemblies: dict[str, _RuntimeAssembly] = {}
        self._active_runs = ActiveRunTracker()
        # 本 Runtime 实例中由持久化数据恢复的 Session。
        self._restored_sessions: set[str] = set()
        # 轻量 MVP：同进程内保存等待用户审批的工具调用。
        self._pending_approvals: dict[str, PendingApproval] = {}

    def create_session(self, options: _CreateAgentSessionOptions) -> _SessionHandle:
        """创建一个新的 Agent 会话。

        Args:
            options: 创建会话选项。

        Returns:
            SessionHandle，包含 session_id、session 实例和装配产物。
        """
        persisted_meta = (
            Path(options.workspace_dir)
            / ".codepilot"
            / "sessions"
            / str(options.session_id)
            / "session.json"
        )
        is_restored = options.session_id is not None and persisted_meta.is_file()
        session, assembly = assemble_runtime(options)
        self._sessions[session.session_id] = session
        self._assemblies[session.session_id] = assembly
        if is_restored:
            self._restored_sessions.add(session.session_id)
        else:
            self._restored_sessions.discard(session.session_id)

        return _SessionHandle(
            session_id=session.session_id,
            session=session,
            assembly=assembly,
        )

    def _register_derived_session(
        self,
        session: AgentSession,
        *,
        source_session_id: str,
    ) -> _SessionHandle:
        """注册由现有会话分支得到的新会话。"""

        source = self.get_assembly(source_session_id)
        session_options = replace(
            source.session_options,
            session_id=session.session_id,
            messages=[],
            task_mode=session.task_mode,
            planning_budget_profile=session.planning_budget_profile,
        )
        assembly = replace(
            source,
            session_options=session_options,
            profile=replace(
                source.profile,
                task_mode=session.task_mode,
                planning_budget_profile=session.planning_budget_profile,
            ),
        )
        self._sessions[session.session_id] = session
        self._assemblies[session.session_id] = assembly
        self._restored_sessions.discard(session.session_id)
        return _SessionHandle(
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

    def get_assembly(self, session_id: str) -> _RuntimeAssembly:
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

    def get_session_status(self, session_id: str) -> _SessionStatus:
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

        return _SessionStatus(
            session_id=session.session_id,
            model_id=model_id,
            workspace=str(session.workspace_dir),
            permission_mode=assembly.profile.permission_mode,
            task_mode=session.task_mode,
            message_count=len(session.messages),
            leaf_id=session.get_leaf_id() or "N/A",
            is_running=self._current_active_run(session_id) is not None,
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
            "task_mode": session.task_mode,
            "planning_budget_profile": session.planning_budget_profile,
        }

    def get_session_freshness(self, session_id: str) -> dict[str, Any]:
        """返回最近 Run Artifact 相对当前工作区的新鲜度。"""

        session = self.get_session(session_id)
        return session.store.run_store.evaluate_freshness().to_event_payload()

    def get_context_report(self, session_id: str) -> dict[str, Any] | None:
        """返回 Session 最近一次模型调用的上下文编译报告。"""

        report = self.get_session(session_id).latest_context_report
        return dict(report) if report is not None else None

    def get_memory_state(self, session_id: str) -> dict[str, Any]:
        """Return a read-only summary of structured session/project memory."""

        session = self.get_session(session_id)
        session_records = session.memory_store.load_session()
        project_records = session.memory_store.load_project()
        records = [*session_records, *project_records]
        pinned = load_global_memory(session.workspace_dir)
        return {
            "session_id": session_id,
            "pinned": {
                "path": str(Path(session.workspace_dir) / ".codepilot" / "MEMORY.md"),
                "chars": len(pinned),
                "preview": pinned[:400],
            },
            "session": [record.to_dict() for record in session_records],
            "project": [record.to_dict() for record in project_records],
            "counts": {
                "session_active": sum(
                    record.status == "active" for record in session_records
                ),
                "project_active": sum(
                    record.status == "active" for record in project_records
                ),
                "deleted": sum(record.status == "deleted" for record in records),
                "superseded": sum(
                    record.status == "superseded" for record in records
                ),
            },
        }

    def get_session_recovery_state(self, session_id: str) -> dict[str, Any]:
        """返回 Eval/接口层需要的只读恢复状态。"""

        runs = self.list_runs(session_id)
        return {
            "session_id": session_id,
            "restored": session_id in self._restored_sessions,
            "run_ids": [
                run_id
                for result in runs
                if isinstance((run_id := result.get("run_id")), str)
            ],
            "freshness": self.get_session_freshness(session_id),
        }

    def list_session_entries(self, session_id: str) -> list[dict[str, Any]]:
        return self.get_session(session_id).list_entries()

    def get_session_tree(self, session_id: str) -> list[dict[str, Any]]:
        return self.get_session(session_id).get_session_tree()

    def get_entry_path(self, session_id: str, entry_id: str) -> list[str]:
        return self.get_session(session_id).get_entry_path(entry_id)

    def switch_entry(self, session_id: str, entry_id: str) -> None:
        self.get_session(session_id).switch_to_entry(entry_id)

    def set_task_mode(self, session_id: str, mode: str) -> str:
        """Set the user-facing task mode for future runs in a session."""
        return self.get_session(session_id).set_task_mode(mode)

    def fork_session(self, session_id: str, entry_id: str) -> str:
        forked = self.get_session(session_id).fork_from_entry(entry_id)
        self._register_derived_session(
            forked,
            source_session_id=session_id,
        )
        return forked.session_id

    def clear_session(self, session_id: str) -> str:
        """Create and register a fresh session derived from the current one."""

        from codepilot.sessions.history.branching import create_fresh_session

        fresh = create_fresh_session(self.get_session(session_id))
        self._register_derived_session(
            fresh,
            source_session_id=session_id,
        )
        return fresh.session_id

    async def run_message(self, session_id: str, message: _UserInput) -> AgentRunResult:
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

        return await self._run_active_session_call(
            session_id,
            lambda session, active_run: self._run_prompt(
                session,
                active_run,
                message,
            ),
        )

    async def send_message(self, session_id: str, message: _UserInput) -> AsyncIterator[AgentEvent]:
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

        async for event in self._stream_active_session_call(
            session_id,
            lambda session, active_run: self._run_prompt(
                session,
                active_run,
                message,
            ),
        ):
            yield event

    async def _run_prompt(
        self,
        session: AgentSession,
        active_run: _ActiveRun,
        message: _UserInput,
    ) -> AgentRunResult:
        run_kwargs: dict[str, Any] = {
            "images": _session_images(message),
            "run_id": active_run.run_id,
        }
        if message.task_mode is not None:
            run_kwargs["task_mode"] = message.task_mode
        return await session.run(message.text, **run_kwargs)

    async def continue_session(self, session_id: str) -> AsyncIterator[AgentEvent]:
        """继续上一次未完成的会话运行，流式返回事件。

        Args:
            session_id: 目标会话 ID。

        Yields:
            AgentEvent 事件对象。
        """
        self.get_session(session_id)
        self._require_session_idle(session_id)

        async for event in self._stream_active_session_call(
            session_id,
            lambda session, active_run: session.continue_run(run_id=active_run.run_id),
        ):
            yield event

    async def cancel_run(self, session_id: str) -> bool:
        """取消当前正在运行的任务。

        Args:
            session_id: 目标会话 ID。

        Returns:
            是否成功取消（True 表示取消，False 表示没有运行中的任务）。
        """
        return await self._active_runs.cancel(session_id)

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

    def get_run_audit_bundle(
        self,
        session_id: str,
        run_id: str,
    ) -> AuditBundle:
        """读取指定 Run 的统一审计证据。"""

        session = self.get_session(session_id)
        return load_audit_bundle(
            session.store.run_store.root / run_id,
            workspace=session.workspace_dir,
        )

    def list_pending_approvals(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """列出当前 Runtime 内存中等待用户决策的审批项。"""

        approvals = [
            approval
            for approval in self._pending_approvals.values()
            if session_id is None or approval.session_id == session_id
        ]
        return [
            {
                "approval_id": approval.approval_id,
                "session_id": approval.session_id,
                "run_id": approval.run_id,
                "tool_call_id": approval.tool_call.id,
                "tool_name": approval.tool_call.name,
                "reason": approval.reason,
            }
            for approval in sorted(approvals, key=lambda item: item.approval_id)
        ]

    def _validate_request(self, session_id: str, message: _UserInput) -> None:
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

        self._require_session_idle(session_id)

    def _current_active_run(self, session_id: str) -> _ActiveRun | None:
        """Return the still-running ActiveRun and discard completed leftovers."""

        return self._active_runs.get(session_id)

    def _require_session_idle(self, session_id: str) -> None:
        """Raise when a session already has an active run in this Runtime."""

        if self._current_active_run(session_id) is not None:
            raise SessionBusyError(f"Session {session_id} is already running")

    def _create_active_run(self, session_id: str) -> _ActiveRun:
        """创建 ActiveRun。

        Args:
            session_id: 会话 ID。

        Returns:
            ActiveRun 实例。
        """
        return self._active_runs.create(session_id)

    async def _run_active_session_call(
        self,
        session_id: str,
        run: Callable[[AgentSession, _ActiveRun], Awaitable[AgentRunResult]],
    ) -> AgentRunResult:
        """执行一次非流式会话调用，并维护 active run 生命周期。"""

        session = self.get_session(session_id)
        active_run = self._create_active_run(session_id)
        active_run.task = asyncio.current_task()

        try:
            result = await run(session, active_run)
            self._record_pending_approvals(session_id, result)
            active_run.status = "completed"
            return result
        except asyncio.CancelledError:
            active_run.status = "aborted"
            raise
        except Exception:
            active_run.status = "failed"
            raise
        finally:
            self._active_runs.discard(session_id)

    async def _stream_active_session_call(
        self,
        session_id: str,
        run: Callable[[AgentSession, _ActiveRun], Awaitable[object]],
    ) -> AsyncIterator[AgentEvent]:
        """以流式事件方式执行一次会话 run，并维护 active run 生命周期。"""

        session = self.get_session(session_id)
        active_run = self._create_active_run(session_id)

        try:
            async for event in self._stream_session_events(
                session,
                lambda: run(session, active_run),
                active_run=active_run,
            ):
                yield event
            result = getattr(session, "last_run_result", None)
            if result is not None:
                self._record_pending_approvals(session_id, result)
            active_run.status = "completed"
        except asyncio.CancelledError:
            active_run.status = "aborted"
            raise
        except Exception:
            active_run.status = "failed"
            raise
        finally:
            self._active_runs.discard(session_id)

    async def _stream_session_events(
        self,
        session: AgentSession,
        run: Callable[[], Awaitable[object]],
        active_run: _ActiveRun | None = None,
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

    async def approve_tool_call(
        self,
        approval_id: str,
        decision: str,
        *,
        session_id: str | None = None,
    ) -> AgentRunResult | None:
        """审批待恢复的工具调用，并用审批后的结果继续会话。"""

        normalized = _normalize_approval_decision(decision)
        approval = self._resolve_pending_approval(approval_id, session_id=session_id)
        resolved_session_id = approval.session_id
        self._require_session_idle(resolved_session_id)

        session = self.get_session(resolved_session_id)
        if not self._session_has_pending_approval_result(session, approval):
            raise ApprovalNotFoundError(
                f"Pending approval result not found in session context: {approval.approval_id}"
            )

        return await self._run_active_session_call(
            resolved_session_id,
            lambda session, active_run: self._resume_after_tool_approval(
                session,
                active_run,
                approval=approval,
                decision=normalized,
            ),
        )

    def close_session(self, session_id: str) -> None:
        """同步关闭空闲会话；运行中的会话应使用 aclose_session。"""
        active_run = self._current_active_run(session_id)
        if active_run is not None:
            raise SessionBusyError(
                f"Cannot close running session: {session_id}"
            )
        self._active_runs.discard(session_id)
        self._restored_sessions.discard(session_id)

        # 移除装配产物
        self._assemblies.pop(session_id, None)

        # 关闭会话
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()
        self._pending_approvals = {
            approval_id: approval
            for approval_id, approval in self._pending_approvals.items()
            if approval.session_id != session_id
        }

    async def aclose_session(self, session_id: str) -> None:
        """取消并等待当前运行结束，然后关闭指定会话。"""

        await self.cancel_run(session_id)
        self.close_session(session_id)

    def close_all(self) -> None:
        """同步关闭所有空闲会话；存在运行任务时拒绝关闭。"""

        running_session_ids = [
            session_id
            for session_id in self._active_runs.session_ids()
            if self._current_active_run(session_id) is not None
        ]
        if running_session_ids:
            raise SessionBusyError(
                "Cannot close RuntimeService with running sessions: "
                + ", ".join(sorted(running_session_ids))
            )
        for session_id in list(self._sessions):
            self.close_session(session_id)

    def _record_pending_approvals(
        self,
        session_id: str,
        result: AgentRunResult,
    ) -> None:
        """从 waiting_approval run 中提取可恢复的工具调用。"""

        for approval in build_pending_approvals(session_id, result):
            self._pending_approvals[approval.approval_id] = approval

    def _resolve_pending_approval(
        self,
        approval_id: str,
        *,
        session_id: str | None = None,
    ) -> PendingApproval:
        approval = self._pending_approvals.get(approval_id)
        if approval is not None:
            if session_id is not None and approval.session_id != session_id:
                raise ApprovalNotFoundError(
                    f"Approval not found for session {session_id}: {approval_id}"
                )
            return approval
        candidates = [
            candidate
            for candidate in self._pending_approvals.values()
            if candidate.tool_call.id == approval_id
            and (session_id is None or candidate.session_id == session_id)
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ApprovalNotFoundError(
                f"Approval id is ambiguous across sessions: {approval_id}"
            )
        raise ApprovalNotFoundError(f"Approval not found: {approval_id}")

    async def _resume_after_tool_approval(
        self,
        session: AgentSession,
        active_run: _ActiveRun,
        *,
        approval: PendingApproval,
        decision: str,
    ) -> AgentRunResult:
        """Apply a tool approval decision and continue the interrupted run.

        The rollback baseline is captured before executing the approved tool so
        the persisted run can describe both the approved tool side effect and
        the follow-up model/tool work triggered by ``continue_run``.
        """

        approval_baseline = session.capture_run_rollback_baseline()
        if decision == "approve":
            replacement = await self._execute_approved_tool(
                approval,
                run_id=active_run.run_id,
            )
        else:
            replacement = denied_tool_result(approval)
        replaced = session.replace_tool_result_message(
            replacement,
            approval_id=approval.approval_id,
        )
        if not replaced:
            raise ApprovalNotFoundError(
                f"Pending approval result not found in session context: {approval.approval_id}"
            )
        self._pending_approvals.pop(approval.approval_id, None)
        session.store.append_event(
            {
                "type": "tool_approval_decision",
                "sessionId": session.session_id,
                "runId": active_run.run_id,
                "approvalId": approval.approval_id,
                "toolCallId": approval.tool_call.id,
                "toolName": approval.tool_call.name,
                "decision": decision,
            }
        )
        return await session.continue_after_tool_approval(
            run_id=active_run.run_id,
            approved_tool_result=replacement,
            rollback_baseline=approval_baseline,
        )

    @staticmethod
    def _session_has_pending_approval_result(
        session: AgentSession,
        approval: PendingApproval,
    ) -> bool:
        for message in session.messages:
            if not isinstance(message, ToolResultMessage):
                continue
            approval_matches = (
                bool(approval.approval_id)
                and message.approval_id == approval.approval_id
            )
            call_matches = (
                message.tool_call_id == approval.tool_call.id
                and message.status == "approval_required"
            )
            if approval_matches or call_matches:
                return True
        return False

    async def _execute_approved_tool(
        self,
        approval: PendingApproval,
        *,
        run_id: str,
    ) -> ToolResultMessage:
        registered = next(
            (
                item
                for item in self.get_assembly(approval.session_id).capabilities.tools
                if item.name == approval.tool_call.name
            ),
            None,
        )
        if registered is None:
            result = AgentToolResult(
                content=[TextContent(text=f"Tool {approval.tool_call.name} not found")],
                status="error",
                is_error=True,
                approved=True,
                approval_id=approval.approval_id,
                error_code="tool_not_found",
            )
            result.tool_call_id = approval.tool_call.id
            result.tool_name = approval.tool_call.name
            result.approved = True
            result.approval_id = approval.approval_id
            result.metadata.setdefault(
                "approval_resume",
                {
                    "approval_id": approval.approval_id,
                    "decision": "approved",
                },
            )
            return to_tool_result_message(result)

        session = self.get_session(approval.session_id)
        direct_tool = AgentTool(
            name=registered.tool.name,
            label=registered.tool.label,
            description=registered.tool.description,
            parameters=registered.tool.parameters,
            execute=registered.tool.execute,
            runtime_managed=True,
            metadata=registered.metadata,
        )
        message = await session.execute_approved_tool_call(
            tool=direct_tool,
            tool_call=approval.tool_call,
            run_id=run_id,
        )
        if message is None:
            result = AgentToolResult(
                content=[TextContent(text="Approved tool execution produced no result")],
                status="error",
                is_error=True,
                approved=True,
                approval_id=approval.approval_id,
                error_code="tool_no_result",
            )
            result.tool_call_id = approval.tool_call.id
            result.tool_name = approval.tool_call.name
            return to_tool_result_message(result)
        message.approval_id = approval.approval_id
        message.metadata.setdefault(
            "approval_resume",
            {
                "approval_id": approval.approval_id,
                "decision": "approved",
            },
        )
        if message.status == "success":
            message.approved = True
            message.is_error = False
        if isinstance(message.details, dict):
            message.details.setdefault("approval_id", approval.approval_id)
            message.details.setdefault("approval_decision", "approved")
        return message

    async def aclose_all(self) -> None:
        """取消并等待全部运行任务结束，然后关闭所有会话。"""

        for session_id in self._active_runs.session_ids():
            await self.cancel_run(session_id)
        for session_id in list(self._sessions):
            self.close_session(session_id)
