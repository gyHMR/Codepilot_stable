from __future__ import annotations

"""AgentSession 负责编排一次应用级别的 Agent 对话。

主要职责：
1) 管理工作区会话目录。
2) 持久化 Agent 事件和消息。
3) 提供稳定的 run（发送任务）和 continue（继续运行）入口。
4) 通过 ContextGovernor 为每轮模型调用投影上下文。
"""

from pathlib import Path
import inspect
import logging
from typing import Callable

from codepilot.protocols import (
    AgentEvent,
    AgentRunResult,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
)
from codepilot.core import (
    Agent,
    AgentMessage,
    AgentOptions,
    TaskMode,
    ensure_planning_budget_profile,
    ensure_task_mode,
    new_run_id,
)

from codepilot.extensions.types import ExtensionLifecycleContext
from codepilot.tools import AgentTool
from .context.freshness import build_context_freshness_notice
from .context.governor import ContextGovernor
from .context.state import SessionContextState
from .history.branching import fork_session as branch_fork_session
from .history.branching import switch_session as branch_switch_session
from .history.branching import switch_to_entry as branch_switch_to_entry
from .history.checkpoint import SessionCheckpoint, record_checkpoint
from .history.git_rollback import (
    GitRollbackBaseline,
    GitRollbackResult,
    build_rollback_metadata,
    capture_git_baseline,
    revert_run_changes,
)
from .history.task_recovery import TaskRecoveryStore
from .memory import MemoryRetriever, MemoryStore, MemoryWriter
from .persistence.store import SessionStore, new_session_id
from .run_reconciliation import (
    merge_approved_tool_result,
    replace_pending_tool_result,
)
from .types import AgentSessionOptions

logger = logging.getLogger("codepilot.sessions.session")


class AgentSession:
    """Agent 会话管理类。

    封装了一个完整的 Agent 对话生命周期，包括：
    - 消息的收发与持久化存储
    - 每轮通过 ContextGovernor 投影上下文视图
    - 会话分支（fork）与切换
    - 生命周期钩子的执行
    - 请求失败时的自动重试
    """

    def __init__(self, options: AgentSessionOptions) -> None:
        """初始化 Agent 会话。

        Args:
            options: 会话配置选项，包含工作区目录、模型信息、系统提示词、工具列表等。
        """
        workspace_dir = Path(options.workspace_dir)
        self.workspace_dir = workspace_dir
        self.get_api_key = options.get_api_key
        # 如果未提供 session_id，则自动生成一个新的
        self.session_id = options.session_id or new_session_id()

        # 初始化会话持久化存储，并确保目录结构已创建
        self.store = SessionStore(workspace_dir=workspace_dir, session_id=self.session_id)
        self.store.ensure_initialized(
            model_id=options.model.id,
            provider=options.model.provider,
            system_prompt=options.system_prompt,
        )
        self.memory_store = MemoryStore(self.store)
        self.memory_writer = MemoryWriter(
            store=self.memory_store,
            workspace_dir=self.workspace_dir,
        )
        self.memory_retriever = MemoryRetriever(
            store=self.memory_store,
            workspace_dir=self.workspace_dir,
        )
        self.task_recovery = TaskRecoveryStore(self.store)
        self.memory_enabled = options.memory_enabled
        self.task_mode = ensure_task_mode(options.task_mode)
        self.planning_budget_profile = ensure_planning_budget_profile(
            options.planning_budget_profile
        )
        self.context_governor: ContextGovernor | None = None
        prepare_context = self._build_context_preparer()

        # 加载已持久化的 canonical transcript。
        persisted_messages = self.store.load_session_messages()
        # 将历史消息与本次传入的新消息合并
        merged_messages = [*persisted_messages, *options.messages]

        # 构建 Agent 配置项
        agent_opts = AgentOptions(
            model=options.model,
            system_prompt=options.system_prompt,
            tools=options.tools,
            messages=merged_messages,
            thinking_level=options.thinking_level,
            tool_execution=options.tool_execution,
            max_tool_calls_per_turn=options.max_tool_calls_per_turn,
            get_api_key=options.get_api_key,
            before_tool_call=options.before_tool_call,
            after_tool_call=options.after_tool_call,
            retry_enabled=options.retry_enabled,
            max_model_retries=options.max_retries,
            retry_base_delay_ms=options.retry_base_delay_ms,
            session_id=self.session_id,
            stream_fn=options.stream_fn,
            prepare_context=prepare_context,
            task_control_enabled=options.task_control_enabled,
            task_mode=self.task_mode,
            planning_budget_profile=self.planning_budget_profile,
            max_task_replans_per_run=options.max_task_replans_per_run,
        )
        if options.convert_to_llm is not None:
            agent_opts.convert_to_llm = options.convert_to_llm

        # 创建核心 Agent 实例
        self.agent = Agent(agent_opts)

        self.latest_context_report: dict | None = None
        self.prepare_context = prepare_context
        self.stream_fn = options.stream_fn

        self.tool_execution = options.tool_execution
        self.max_tool_calls_per_turn = options.max_tool_calls_per_turn
        # 重试机制配置
        self.retry_enabled = options.retry_enabled
        self.max_retries = options.max_retries
        self.retry_base_delay_ms = options.retry_base_delay_ms

        # 扩展命令注册表
        self.extension_commands = dict(options.extension_commands)
        # 提示词执行前后的生命周期钩子
        self.before_prompt_hooks = list(options.before_prompt_hooks)
        self.after_prompt_hooks = list(options.after_prompt_hooks)
        # 工具调用前后的回调
        self.before_tool_call = options.before_tool_call
        self.after_tool_call = options.after_tool_call

        # 订阅 Agent 事件，用于持久化事件和消息
        self._unsubscribe = self.agent.subscribe(self._on_agent_event)

    @property
    def messages(self) -> list[AgentMessage]:
        """获取当前会话的全部消息列表。"""
        return self.agent.state.messages

    @property
    def last_run_result(self) -> AgentRunResult | None:
        """返回最近一次 Run 的结构化结果（含状态、停止原因、消息等）。"""
        return self.agent.last_run_result

    @property
    def last_usage(self) -> dict | None:
        """返回最近一条助手消息的 token 用量和费用信息。

        Returns:
            包含 input_tokens、output_tokens、total_tokens、cache_read、
            cache_write 和 cost 的字典；若无助手消息则返回 None。
        """
        for msg in reversed(self.agent.state.messages):
            if isinstance(msg, AssistantMessage):
                u = msg.usage
                return {
                    "input_tokens": u.input,
                    "output_tokens": u.output,
                    "total_tokens": u.total_tokens,
                    "cache_read": u.cache_read,
                    "cache_write": u.cache_write,
                    "cost": {
                        "input": u.cost.input,
                        "output": u.cost.output,
                        "total": u.cost.total,
                    },
                }
        return None

    @property
    def cumulative_usage(self) -> dict:
        """返回本次会话的累计 token 用量和总费用。

        Returns:
            包含 input_tokens、output_tokens、total_tokens 和 total_cost 的字典。
        """
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_cost = 0.0
        for msg in self.agent.state.messages:
            if isinstance(msg, AssistantMessage):
                total_input += msg.usage.input
                total_output += msg.usage.output
                total_tokens += msg.usage.total_tokens
                total_cost += msg.usage.cost.total
        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }

    def set_task_mode(self, mode: TaskMode | str) -> TaskMode:
        """Set the task mode used by future runs in this session."""
        normalized = ensure_task_mode(mode)
        changed = normalized != self.task_mode
        if changed:
            self.task_mode = normalized
            self.agent.set_task_mode(normalized)
            self.store.append_event(
                {
                    "type": "task_mode_changed",
                    "sessionId": self.session_id,
                    "taskMode": normalized,
                }
            )
        projection = self.task_recovery.load_projection()
        if projection is not None and projection.get("task_mode") != normalized:
            projection["task_mode"] = normalized
            self.task_recovery.save_projection(projection)
        return normalized

    async def continue_run(
        self,
        *,
        run_id: str | None = None,
    ) -> AgentRunResult:
        """继续上一次未完成的 Agent 运行（例如工具调用后的延续）。

        Returns:
            继续运行产生的结构化 Run 结果。
        """
        run_id = run_id or new_run_id()
        rollback_baseline = await self._start_run_lifecycle(
            text="",
            run_id=run_id,
            is_continue=True,
        )
        result = await self.agent.continue_run(run_id=run_id)
        return await self._complete_run_lifecycle(
            result,
            rollback_baseline=rollback_baseline,
            hook_text="",
            is_continue=True,
        )

    async def continue_after_tool_approval(
        self,
        *,
        run_id: str,
        approved_tool_result: ToolResultMessage,
        rollback_baseline: GitRollbackBaseline,
    ) -> AgentRunResult:
        """Continue a run after Runtime has resolved a pending tool approval.

        Runtime owns the user-facing approval transaction and executes (or
        denies) the pending tool call.  Session owns the durable run record, so
        it also owns reconciling that approval tool result into the follow-up
        ``AgentRunResult`` before the result is persisted and exposed as
        ``last_run_result``.
        """

        await self._start_run_lifecycle(
            text="",
            run_id=run_id,
            is_continue=True,
            rollback_baseline=rollback_baseline,
        )
        result = await self.agent.continue_run(run_id=run_id)
        result = merge_approved_tool_result(result, approved_tool_result)
        self.agent.replace_last_run_result(result)
        return await self._complete_run_lifecycle(
            result,
            rollback_baseline=rollback_baseline,
            hook_text="",
            is_continue=True,
        )

    async def execute_approved_tool_call(
        self,
        *,
        tool: AgentTool,
        tool_call: ToolCall,
        run_id: str,
    ) -> ToolResultMessage | None:
        """Execute a previously deferred tool call during approval recovery.

        Runtime decides that a pending approval has been accepted and selects
        the concrete tool implementation.  Session delegates execution to Agent
        so the normal before/after tool hooks and tool events still apply.
        """

        results = await self.agent.execute_tool_call_once(
            tool=tool,
            tool_call=tool_call,
            run_id=run_id,
        )
        return results[0] if results else None

    async def run(
        self,
        text: str,
        *,
        images: list[str] | None = None,
        run_id: str | None = None,
        task_mode: TaskMode | str | None = None,
    ) -> AgentRunResult:
        """执行一次用户任务并持久化结构化结果。

        完整流程：
        1. 执行 before_prompt 生命周期钩子
        2. 将任务写入 task recovery，并按策略提取 durable memory
        3. 恢复未完成任务投影（供 Agent 决策参考）
        4. 检查上下文新鲜度（文件是否被外部修改）
        5. 检测并压缩溢出的上下文
        6. 调用 Agent.run() 执行 LLM 推理和工具调用
        7. 持久化 Run 结果
        8. 将 Run 结果写入记忆（_finalize_memory）
        9. 根据需要压缩上下文
        10. 执行 after_prompt 生命周期钩子

        Args:
            text: 用户输入文本。
            images: 可选的图片列表（base64 编码）。
            run_id: 可选的 Run ID（不提供则自动生成）。

        Returns:
            结构化的 Run 结果。
        """
        if task_mode is not None:
            self.set_task_mode(task_mode)
        run_id = run_id or new_run_id()
        rollback_baseline = await self._start_run_lifecycle(
            text=text,
            run_id=run_id,
            is_continue=False,
        )
        result = await self.agent.run(
            text,
            images=images,
            run_id=run_id,
        )
        return await self._complete_run_lifecycle(
            result,
            rollback_baseline=rollback_baseline,
            hook_text=text,
            is_continue=False,
        )

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """订阅 Agent 事件流。

        Args:
            listener: 事件回调函数，接收 AgentEvent 参数。

        Returns:
            取消订阅的函数，调用后停止接收事件。
        """
        return self.agent.subscribe(listener)

    def close(self) -> None:
        """关闭会话，取消事件订阅以释放资源。"""
        self._unsubscribe()

    # ── 会话分支与历史导航 ──────────────────────────────────────

    def list_entry_ids(self) -> list[str]:
        """列出当前会话所有条目的 ID 列表。"""
        return self.store.list_entry_ids()

    def list_entries(self) -> list[dict]:
        """列出当前会话所有条目的详细信息。"""
        return self.store.list_entries()

    def get_leaf_id(self) -> str | None:
        """获取当前会话分支的末端（叶子）条目 ID。"""
        return self.store.get_leaf_id()

    def get_entry_path(self, entry_id: str) -> list[str]:
        """获取从根条目到指定条目的路径（ID 列表）。"""
        return self.store.get_entry_path(entry_id)

    def get_session_tree(self) -> list[dict]:
        """获取会话的树形结构，用于可视化分支历史。"""
        return self.store.get_session_tree()

    def fork_session(self, from_entry_id: str | None = None) -> "AgentSession":
        """从指定条目分叉（fork）一个新的会话。

        Args:
            from_entry_id: 分叉起点的条目 ID，None 表示从当前末端分叉。

        Returns:
            新创建的 AgentSession 实例。
        """
        return branch_fork_session(self, from_entry_id=from_entry_id)

    def fork_from_entry(self, entry_id: str) -> "AgentSession":
        """从指定条目分叉会话（fork_session 的便捷别名）。"""
        return self.fork_session(from_entry_id=entry_id)

    def switch_to_entry(self, entry_id: str) -> None:
        """切换当前会话到指定的历史条目。"""
        branch_switch_to_entry(self, entry_id)

    def switch_session(self, session_id: str) -> None:
        """切换到另一个会话。"""
        branch_switch_session(self, session_id)

    def rebind_store(self, store: SessionStore) -> None:
        """会话切换后重新绑定持久化存储、记忆和上下文编译器。

        当 switch_session 切换到另一个会话时，需要将所有内部状态
        指向新的 SessionStore，否则后续操作会写入旧会话目录。

        Args:
            store: 新的 SessionStore 实例。
        """
        self.store = store
        self.session_id = store.session_id
        self.agent.set_session_id(self.session_id)
        # 重建记忆子系统（它们依赖 SessionStore 的 memory_file 路径）
        self.memory_store = MemoryStore(store)
        self.memory_writer = MemoryWriter(
            store=self.memory_store,
            workspace_dir=self.workspace_dir,
        )
        self.memory_retriever = MemoryRetriever(
            store=self.memory_store,
            workspace_dir=self.workspace_dir,
        )
        self.task_recovery = TaskRecoveryStore(store)
        # 重新编译上下文准备函数（绑定新的 session_id / memory_retriever）
        self.prepare_context = self._build_context_preparer()
        self.agent.set_prepare_context(self.prepare_context)

    def record_checkpoint(self, label: str, details: dict | None = None) -> SessionCheckpoint:
        """在当前会话中记录一个检查点（checkpoint）。

        Args:
            label: 检查点的标签名称。
            details: 可选的附加详情字典。

        Returns:
            创建的 SessionCheckpoint 实例。
        """
        return record_checkpoint(self.store, session_id=self.session_id, label=label, details=details)

    def capture_run_rollback_baseline(self) -> GitRollbackBaseline:
        """Capture the workspace rollback baseline for a run-owned transaction."""

        return capture_git_baseline(self.workspace_dir)

    def revert_last_run(self) -> GitRollbackResult:
        """撤销当前会话最近一次支持 Git clean-worktree 回退的 Run。"""

        runs = self.store.load_run_results(limit=1)
        if not runs:
            return GitRollbackResult(
                status="not_eligible",
                run_id="",
                reason="no_run_results",
            )
        run_id = runs[-1].get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return GitRollbackResult(
                status="not_eligible",
                run_id="",
                reason="missing_run_id",
            )
        return self.revert_run(run_id)

    def revert_run(self, run_id: str) -> GitRollbackResult:
        """按 run_id 撤销该 Run 记录的工作区文件修改。"""

        try:
            state = self.store.run_store.load_run_state(run_id)
        except FileNotFoundError:
            return GitRollbackResult(
                status="not_eligible",
                run_id=run_id,
                reason="missing_run_state",
            )
        result = revert_run_changes(self.workspace_dir, state)
        self.store.append_event(
            {
                "type": "run_reverted",
                "sessionId": self.session_id,
                "targetRunId": run_id,
                "status": result.status,
                "reason": result.reason,
                "restoredPaths": list(result.restored_paths),
                "removedPaths": list(result.removed_paths),
                "conflictedPaths": list(result.conflicted_paths),
            }
        )
        return result

    # ── 内部方法 ────────────────────────────────────────────────

    def _write_rollback_metadata(
        self,
        result: AgentRunResult,
        baseline: GitRollbackBaseline,
    ) -> None:
        self.store.write_rollback_metadata(
            result.run_id,
            build_rollback_metadata(
                baseline,
                affected_paths=list(result.affected_paths),
                workspace_changed=bool(result.workspace_changed),
            ),
        )

    async def _start_run_lifecycle(
        self,
        *,
        text: str,
        run_id: str,
        is_continue: bool,
        rollback_baseline: GitRollbackBaseline | None = None,
    ) -> GitRollbackBaseline:
        """Prepare session-owned state before delegating to the core Agent.

        This is the boundary between application session concerns and the core
        run loop: hooks, durable memory admission, task recovery projection,
        and context freshness are prepared here. The ContextGovernor projection
        runs later through AgentOptions.prepare_context, immediately before the
        Agent asks the model for the next response.
        """

        rollback_baseline = rollback_baseline or capture_git_baseline(self.workspace_dir)
        await self._run_lifecycle_hooks(
            text=text,
            is_continue=is_continue,
            hooks=self.before_prompt_hooks,
        )
        if not is_continue:
            if self.memory_enabled:
                self._admit_prompt_memory(text, run_id=run_id)
            self._begin_task_recovery(text, run_id=run_id)
        self.agent.set_task_recovery_projection(self._active_task_recovery_projection())
        self._check_context_freshness()
        return rollback_baseline

    async def _complete_run_lifecycle(
        self,
        result: AgentRunResult,
        *,
        rollback_baseline: GitRollbackBaseline,
        hook_text: str,
        is_continue: bool,
    ) -> AgentRunResult:
        """完成一次 run 的会话侧收尾。

        Agent 负责推理和工具执行；Session 负责把结果落盘、更新恢复状态、
        写入可检索记忆，并在 after hooks 之前保证这些状态已经可观察。
        """

        self.store.append_run_result(result)
        self._write_rollback_metadata(result, rollback_baseline)
        self._finalize_task_recovery(result)
        if self.memory_enabled:
            self._finalize_memory(result)
        self.context_governor.finalize_run(result)
        await self._run_lifecycle_hooks(
            text=hook_text,
            is_continue=is_continue,
            hooks=self.after_prompt_hooks,
        )
        return result

    def replace_tool_result_message(
        self,
        replacement: ToolResultMessage,
        *,
        approval_id: str | None = None,
    ) -> bool:
        """Replace a pending approval ToolResultMessage in live and persisted context."""

        messages, replaced = replace_pending_tool_result(
            list(self.agent.state.messages),
            replacement,
            approval_id=approval_id,
        )
        if not replaced:
            return False
        self.agent.set_messages(messages)
        self.store.rewrite_session_messages(messages)
        self.store.append_event(
            {
                "type": "tool_approval_result_replaced",
                "sessionId": self.session_id,
                "approvalId": approval_id or replacement.approval_id,
                "toolCallId": replacement.tool_call_id,
                "toolName": replacement.tool_name,
                "status": replacement.status,
            }
        )
        return True

    async def _on_agent_event(self, event: AgentEvent) -> None:
        """Agent 事件回调：将事件持久化到存储，并在消息结束时保存上下文消息。"""
        self.store.append_event(event)
        if event["type"] == "context_prepared":
            report = event.get("report")
            if isinstance(report, dict):
                self.latest_context_report = report
                memory_ids = report.get("retrieved_memory_ids")
                if self.memory_enabled and isinstance(memory_ids, list) and memory_ids:
                    self.store.append_event(
                        {
                            "type": "memory_retrieved",
                            "sessionId": self.session_id,
                            "runId": event.get("runId"),
                            "memoryIds": memory_ids,
                            "reasons": report.get("memory_retrieval_reasons", {}),
                        }
                    )
        if event["type"] == "message_end":
            message = event["message"]
            self.store.append_message(message)
            if self.memory_enabled and isinstance(message, ToolResultMessage):
                self._observe_tool_memory(
                    message,
                    run_id=event.get("runId"),
                )

    def _build_context_preparer(self):
        """构建当前会话的上下文准备入口。"""
        self.context_governor = ContextGovernor(
            workspace_dir=self.workspace_dir,
            session_id=self.session_id,
            state=SessionContextState(workspace_dir=self.workspace_dir),
            memory_retriever=(
                self.memory_retriever if self.memory_enabled else None
            ),
        )
        return self.context_governor.prepare

    def _admit_prompt_memory(self, text: str, *, run_id: str | None) -> None:
        """Admit durable project memory from the user prompt when policy allows it."""
        try:
            record = self.memory_writer.admit_prompt_memory(text, run_id=run_id)
            if record is None:
                return
            self.store.append_event(
                {
                    "type": "memory_updated",
                    "sessionId": self.session_id,
                    "memoryId": record.id,
                    "kind": record.kind,
                }
            )
        except Exception as exc:
            logger.warning("failed to admit prompt memory: %s", exc)
            self.store.append_event(
                {
                    "type": "memory_warning",
                    "sessionId": self.session_id,
                    "operation": "prompt_memory_admission",
                    "message": str(exc),
                }
            )

    def _begin_task_recovery(self, text: str, *, run_id: str | None) -> None:
        """Persist the current task projection outside durable memory."""
        try:
            projection = self.task_recovery.begin_task(text, run_id=run_id)
            self.store.append_event(
                {
                    "type": "task_recovery_updated",
                    "sessionId": self.session_id,
                    "runId": run_id,
                    "goal": projection.get("goal"),
                }
            )
        except Exception as exc:
            logger.warning("failed to write task recovery: %s", exc)
            self.store.append_event(
                {
                    "type": "task_recovery_warning",
                    "sessionId": self.session_id,
                    "operation": "task_recovery_begin",
                    "message": str(exc),
                }
            )

    def _active_task_recovery_projection(self) -> dict[str, object] | None:
        """Return unfinished task recovery state for the next Agent run."""
        projection = self.task_recovery.active_projection()
        return dict(projection) if projection is not None else None

    def _observe_tool_memory(
        self,
        message: ToolResultMessage,
        *,
        run_id: str | None,
    ) -> None:
        """观察工具结果，并让记忆写入器处理持久记忆副作用。

        临时文件摘要、工具输出和验证证据属于上下文状态；未完成任务进度
        属于 TaskRecoveryStore。MemoryWriter 在这里不沉淀长期记忆，只保留
        将来需要观察工具事件时的统一扩展点。

        Args:
            message: 工具结果消息。
            run_id: 当前 Run ID。
        """
        try:
            records = self.memory_writer.observe_tool_result(message, run_id=run_id)
            for record in records:
                self.store.append_event(
                    {
                        "type": "memory_updated",
                        "sessionId": self.session_id,
                        "memoryId": record.id,
                        "kind": record.kind,
                    }
                )
        except Exception as exc:
            logger.warning("failed to observe tool memory: %s", exc)
            self.store.append_event(
                {
                    "type": "memory_warning",
                    "sessionId": self.session_id,
                    "operation": "observe_tool_result",
                    "message": str(exc),
                }
            )

    def _finalize_memory(self, result: AgentRunResult) -> None:
        """Run 结束后提取可复用的 durable memory。

        Task progress 不写入 durable memory；它由 TaskRecoveryStore 维护。
        MemoryWriter.finalize_run() 只从已验证的失败-修复-验证闭环中提取
        可复用经验，避免把一次性的过程状态污染后续检索。

        Args:
            result: Agent Run 结果。
        """
        try:
            records = self.memory_writer.finalize_run(result)
            for record in records:
                self.store.append_event(
                    {
                        "type": "memory_updated",
                        "sessionId": self.session_id,
                        "memoryId": record.id,
                        "kind": record.kind,
                    }
                )
        except Exception as exc:
            logger.warning("failed to finalize memory: %s", exc)
            self.store.append_event(
                {
                    "type": "memory_warning",
                    "sessionId": self.session_id,
                    "operation": "finalize_run",
                    "message": str(exc),
                }
            )

    def _finalize_task_recovery(self, result: AgentRunResult) -> None:
        """Update session task recovery from the structured run result."""
        try:
            projection = self.task_recovery.update_from_result(result)
            if projection is None:
                return
            self.store.append_event(
                {
                    "type": "task_recovery_updated",
                    "sessionId": self.session_id,
                    "runId": result.run_id,
                    "goal": projection.get("goal"),
                    "completionSatisfied": (
                        projection.get("task_progress", {}) or {}
                    ).get("completion_satisfied")
                    if isinstance(projection.get("task_progress"), dict)
                    else None,
                }
            )
        except Exception as exc:
            logger.warning("failed to finalize task recovery: %s", exc)
            self.store.append_event(
                {
                    "type": "task_recovery_warning",
                    "sessionId": self.session_id,
                    "operation": "task_recovery_finalize",
                    "message": str(exc),
                }
            )

    def _check_context_freshness(self) -> None:
        """检查上下文新鲜度：对比上次 Run 跟踪的文件与当前工作区状态。

        如果文件被外部修改（changed）或删除（missing），则注入一条
        steering message 提醒 Agent 重新读取相关文件，避免基于过时信息推理。
        """
        freshness = self.store.run_store.evaluate_freshness()
        if not freshness.should_record_event():
            return
        payload = freshness.to_event_payload()
        self.store.append_event(
            {
                "type": "context_freshness_checked",
                "sessionId": self.session_id,
                "freshness": payload,
            }
        )
        if not freshness.requires_steering():
            return
        notice = build_context_freshness_notice(freshness)
        if notice is not None:
            self.agent.add_steering_message(notice)

    async def _run_lifecycle_hooks(
        self,
        *,
        text: str,
        is_continue: bool,
        hooks: list,
    ) -> None:
        """执行生命周期钩子列表。

        钩子可以是同步或异步函数，统一通过 inspect.isawaitable 处理。

        Args:
            text: 当前用户输入文本。
            is_continue: 是否为继续运行（而非新提示）。
            hooks: 要执行的钩子函数列表。
        """
        if not hooks:
            return
        ctx = ExtensionLifecycleContext(
            session=self,
            text=text,
            is_continue=is_continue,
            message_count=len(self.agent.state.messages),
        )
        for hook in hooks:
            value = hook(ctx)
            if inspect.isawaitable(value):
                await value

