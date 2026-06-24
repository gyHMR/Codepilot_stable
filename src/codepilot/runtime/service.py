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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, cast

from codepilot.core import AgentEvent
from codepilot.core import AgentContext, AgentEventEmitter, AgentLoopConfig
from codepilot.core import ToolCallCoordinator
from codepilot.observability import (
    AuditBundle,
    build_run_report,
    load_audit_bundle,
)
from codepilot.protocols import (
    AgentRunResult,
    AssistantMessage,
    RunVerification,
    RunVerificationStatus,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from codepilot.sessions.session import AgentSession
from codepilot.sessions.history.git_rollback import capture_git_baseline
from codepilot.tools import AgentTool, AgentToolResult

from .command_registry import (
    RuntimeCommandResult,
    handle_runtime_command,
    list_runtime_commands,
)
from .assembly import assemble_runtime
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


class ApprovalNotFoundError(RuntimeServiceError):
    """找不到待审批工具调用。"""
    code = "runtime.approval_not_found"


class InvalidApprovalDecisionError(RuntimeServiceError):
    """审批决策值无效。"""
    code = "runtime.invalid_approval_decision"


@dataclass(frozen=True)
class PendingApproval:
    """Runtime 内存中的待审批工具调用。"""

    approval_id: str
    session_id: str
    run_id: str
    assistant_message: AssistantMessage
    tool_call: ToolCall
    reason: str = ""


def _normalize_approval_decision(decision: str) -> str:
    value = decision.strip().lower()
    if value in {"approve", "approved", "allow", "yes", "y"}:
        return "approve"
    if value in {"deny", "denied", "reject", "rejected", "no", "n"}:
        return "deny"
    raise InvalidApprovalDecisionError(f"Invalid approval decision: {decision}")


def _denied_tool_result(approval: PendingApproval) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=approval.tool_call.id,
        tool_name=approval.tool_call.name,
        content=[TextContent(text="Tool execution denied by user")],
        details={
            "status": "denied",
            "reason": "user_denied",
            "approval_id": approval.approval_id,
        },
        is_error=True,
        status="denied",
        approved=False,
        approval_id=approval.approval_id,
        error_code="user_denied",
        timestamp=int(time.time() * 1000),
        metadata={
            "approval_resume": {
                "approval_id": approval.approval_id,
                "decision": "denied",
            }
        },
    )


def _to_tool_result_message(result: AgentToolResult) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        content=list(result.content),
        details=result.details,
        is_error=result.is_error,
        status=result.status,
        approved=result.approved,
        approval_id=result.approval_id,
        error_code=result.error_code,
        exit_code=result.exit_code,
        affected_paths=list(result.affected_paths),
        workspace_changed=result.workspace_changed,
        diff_summary=result.diff_summary,
        verification=dict(result.verification) if result.verification else None,
        timestamp=int(time.time() * 1000),
        metadata=dict(result.metadata),
    )


def _verification_from_tool_result(
    result: ToolResultMessage,
) -> RunVerification | None:
    if not result.verification:
        return None
    raw_status = result.verification.get("status")
    status: RunVerificationStatus = (
        raw_status
        if raw_status in {"passed", "failed", "cancelled", "unknown"}
        else "unknown"
    )
    raw_command = result.verification.get("command")
    raw_exit_code = result.verification.get("exit_code")
    return RunVerification(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        status=status,
        command=raw_command if isinstance(raw_command, str) else None,
        exit_code=(
            raw_exit_code
            if isinstance(raw_exit_code, int) and not isinstance(raw_exit_code, bool)
            else None
        ),
        summary=str(result.verification.get("summary", "")),
    )


def _merge_approval_tool_result(
    result: AgentRunResult,
    approved_tool_result: ToolResultMessage,
) -> AgentRunResult:
    """把审批恢复阶段的工具结果并入后续 continue run 的结构化证据。"""

    messages = list(result.messages)
    already_present = any(
        isinstance(message, ToolResultMessage)
        and message.tool_call_id == approved_tool_result.tool_call_id
        and message.approval_id == approved_tool_result.approval_id
        for message in messages
    )
    if not already_present:
        messages.insert(0, approved_tool_result)

    counters = replace(
        result.counters,
        tool_calls=result.counters.tool_calls + (0 if already_present else 1),
    )
    affected_paths = sorted(
        {
            *result.affected_paths,
            *[
                str(path)
                for path in approved_tool_result.affected_paths
                if str(path)
            ],
        }
    )
    verification = list(result.verification)
    approved_verification = _verification_from_tool_result(approved_tool_result)
    if approved_verification is not None:
        duplicate_verification = any(
            item.tool_call_id == approved_verification.tool_call_id
            and item.tool_name == approved_verification.tool_name
            and item.command == approved_verification.command
            for item in verification
        )
        if not duplicate_verification:
            verification.append(approved_verification)

    return AgentRunResult(
        run_id=result.run_id,
        session_id=result.session_id,
        status=result.status,
        stop_reason=result.stop_reason,
        counters=counters,
        messages=messages,
        final_message=result.final_message,
        error=result.error,
        affected_paths=affected_paths,
        workspace_changed=bool(
            result.workspace_changed or approved_tool_result.workspace_changed is True
        ),
        verification=verification,
        task=result.task,
    )


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
        # 本 Runtime 实例中由持久化数据恢复的 Session。
        self._restored_sessions: set[str] = set()
        # 轻量 MVP：同进程内保存等待用户审批的工具调用。
        self._pending_approvals: dict[str, PendingApproval] = {}

    def create_session(self, options: CreateAgentSessionOptions) -> SessionHandle:
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
            / "meta.json"
        )
        is_restored = options.session_id is not None and persisted_meta.is_file()
        session, assembly = assemble_runtime(options)
        self._sessions[session.session_id] = session
        self._assemblies[session.session_id] = assembly
        if is_restored:
            self._restored_sessions.add(session.session_id)
        else:
            self._restored_sessions.discard(session.session_id)

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
        self._restored_sessions.discard(session.session_id)
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
        return {
            "session_id": session_id,
            "session": [record.to_dict() for record in session_records],
            "project": [record.to_dict() for record in project_records],
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
            result = await session.run(
                message.text,
                images=message.images,
                run_id=active_run.run_id,
            )
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
                lambda: session.run(
                    message.text,
                    images=message.images,
                    run_id=active_run.run_id,
                ),
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
                lambda: session.continue_run(run_id=active_run.run_id),
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
        active = self._active_runs.get(resolved_session_id)
        if active is not None and active.task is not None and active.task.done():
            self._active_runs.pop(resolved_session_id, None)
            active = None
        if active is not None:
            raise SessionBusyError(f"Session {resolved_session_id} is already running")

        session = self.get_session(resolved_session_id)
        if not self._session_has_pending_approval_result(session, approval):
            raise ApprovalNotFoundError(
                f"Pending approval result not found in session context: {approval.approval_id}"
            )

        active_run = self._create_active_run(resolved_session_id)
        active_run.task = asyncio.current_task()
        approval_baseline = capture_git_baseline(session.workspace_dir)
        try:
            if normalized == "approve":
                replacement = await self._execute_approved_tool(
                    approval,
                    run_id=active_run.run_id,
                )
            else:
                replacement = _denied_tool_result(approval)
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
                    "sessionId": resolved_session_id,
                    "runId": active_run.run_id,
                    "approvalId": approval.approval_id,
                    "toolCallId": approval.tool_call.id,
                    "toolName": approval.tool_call.name,
                    "decision": normalized,
                }
            )
            result = await session.continue_run(run_id=active_run.run_id)
            result = _merge_approval_tool_result(result, replacement)
            session.agent._last_run_result = result
            session.store.append_run_result(result)
            session._write_rollback_metadata(result, approval_baseline)
            self._record_pending_approvals(resolved_session_id, result)
            active_run.status = "completed"
            return result
        except asyncio.CancelledError:
            active_run.status = "aborted"
            raise
        except Exception:
            active_run.status = "failed"
            raise
        finally:
            self._active_runs.pop(resolved_session_id, None)

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

    def _record_pending_approvals(
        self,
        session_id: str,
        result: AgentRunResult,
    ) -> None:
        """从 waiting_approval run 中提取可恢复的工具调用。"""

        if result.status != "waiting_approval":
            return
        assistant_by_tool_call: dict[str, AssistantMessage] = {}
        tool_calls: dict[str, ToolCall] = {}
        for message in result.messages:
            if not isinstance(message, AssistantMessage):
                continue
            for block in message.content:
                if isinstance(block, ToolCall) and block.id:
                    assistant_by_tool_call[block.id] = message
                    tool_calls[block.id] = block
        for message in result.messages:
            if (
                not isinstance(message, ToolResultMessage)
                or message.status != "approval_required"
                or not message.approval_id
            ):
                continue
            tool_call = tool_calls.get(message.tool_call_id)
            assistant = assistant_by_tool_call.get(message.tool_call_id)
            if tool_call is None or assistant is None:
                continue
            reason = ""
            if isinstance(message.details, dict):
                raw_reason = message.details.get("reason") or message.details.get("policy_reason")
                reason = raw_reason if isinstance(raw_reason, str) else ""
            self._pending_approvals[message.approval_id] = PendingApproval(
                approval_id=message.approval_id,
                session_id=session_id,
                run_id=result.run_id,
                assistant_message=assistant,
                tool_call=tool_call,
                reason=reason,
            )

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
            return _to_tool_result_message(result)

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
        agent_options = session.agent._options
        emitter = AgentEventEmitter(
            session.agent._dispatch_event,
            run_id=run_id,
            session_id=session.session_id,
        )
        coordinator = ToolCallCoordinator(
            config=AgentLoopConfig(
                model=session.agent.state.model,
                convert_to_llm=agent_options.convert_to_llm,
                transform_context=agent_options.transform_context,
                prepare_context=agent_options.prepare_context,
                get_api_key=agent_options.get_api_key,
                get_steering_messages=None,
                get_follow_up_messages=None,
                tool_execution="sequential",
                before_tool_call=session.before_tool_call,
                after_tool_call=session.after_tool_call,
                reasoning=None,
                session_id=session.session_id,
                max_tool_iterations=agent_options.max_tool_iterations,
                max_tool_calls_per_turn=1,
                allow_unmanaged_tools=True,
                repeated_tool_call_limit=agent_options.repeated_tool_call_limit,
                retry_enabled=agent_options.retry_enabled,
                max_model_retries=agent_options.max_model_retries,
                retry_base_delay_ms=agent_options.retry_base_delay_ms,
                task_control_enabled=agent_options.task_control_enabled,
            ),
            emitter=emitter,
        )
        tool_results = await coordinator.execute_batch(
            AgentContext(
                system_prompt=session.agent.state.system_prompt,
                messages=list(session.messages),
                tools=[direct_tool],
            ),
            AssistantMessage(content=[approval.tool_call]),
        )
        if not tool_results:
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
            return _to_tool_result_message(result)
        message = tool_results[0]
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

        for session_id in list(self._active_runs):
            await self.cancel_run(session_id)
        for session_id in list(self._sessions):
            self.close_session(session_id)
