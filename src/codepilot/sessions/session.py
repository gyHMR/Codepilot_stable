from __future__ import annotations

# 新手导读：SessionRuntime 是会话层内部运行时，串联消息持久化、生命周期 hook、上下文投影、记忆和任务恢复。
# 关注点：对外主入口是 SessionController；本文件只承载持久化会话状态和生命周期副作用。

"""SessionRuntime 承载一次应用级别会话的持久化状态。

主要职责：
1) 管理工作区会话目录。
2) 持久化 Agent 事件和消息。
3) 承载会话生命周期副作用；run 入口由 SessionController 驱动。
4) 通过 ContextGovernor 为每轮模型调用投影上下文。
"""

from pathlib import Path
from typing import Callable

from codepilot.protocols import AgentEvent
from codepilot.core.task_control import (
    ensure_planning_budget_profile,
    ensure_task_mode,
)
from codepilot.core.types import AgentMessage

from .context.governor import ContextGovernor
from .context.state import SessionContextState
from .conversation_state import SessionConversationState
from .history.task_recovery import TaskRecoveryStore
from .memory import MemoryRetriever, MemoryStore, MemoryWriter
from .persistence.store import SessionStore, new_session_id

from .types import SessionOptions
from .contracts import SessionRunRecord


class SessionRuntime:
    """会话层内部运行时状态。

    封装了一个完整的 Agent 对话生命周期，包括：
    - 消息的收发与持久化存储
    - 每轮通过 ContextGovernor 投影上下文视图
    - 会话分支（fork）与切换
    - 生命周期钩子的执行
    - 请求失败时的自动重试
    """

    def __init__(self, options: SessionOptions) -> None:
        """初始化会话运行时。

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
        self.task_control_enabled = options.task_control_enabled
        self.task_mode = ensure_task_mode(options.task_mode)
        self.max_task_replans_per_run = options.max_task_replans_per_run
        self.planning_budget_profile = ensure_planning_budget_profile(
            options.planning_budget_profile
        )
        self.context_governor: ContextGovernor | None = None
        prepare_context = self._build_context_preparer()

        # 加载已持久化的 canonical transcript。
        persisted_messages = self.store.load_session_messages()
        # 将历史消息与本次传入的新消息合并
        merged_messages = [*persisted_messages, *options.messages]

        # Session owns conversation state; core receives snapshots through
        # V2 contracts and never holds this live object.
        self.conversation = SessionConversationState(
            model=options.model,
            system_prompt=options.system_prompt,
            messages=merged_messages,
            thinking_level=options.thinking_level,
            task_mode=self.task_mode,
        )

        self.latest_context_report: dict | None = None
        self.prepare_context = prepare_context
        self.stream_fn = options.stream_fn
        self.convert_to_llm = options.convert_to_llm

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

        self._last_session_run_record: SessionRunRecord | None = None

    def _subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """订阅 Agent 事件流。

        Args:
            listener: 事件回调函数，接收 AgentEvent 参数。

        Returns:
            取消订阅的函数，调用后停止接收事件。
        """
        return self.conversation.subscribe(listener)

    def _close(self) -> None:
        """关闭会话，取消事件订阅以释放资源。"""
        self.conversation.clear_listeners()

    def _rebind_store(self, store: SessionStore) -> None:
        """会话切换后重新绑定持久化存储、记忆和上下文编译器。

        当 switch_session 切换到另一个会话时，需要将所有内部状态
        指向新的 SessionStore，否则后续操作会写入旧会话目录。

        Args:
            store: 新的 SessionStore 实例。
        """
        self.store = store
        self.session_id = store.session_id
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

    # ── 内部方法 ────────────────────────────────────────────────

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

    def _active_task_recovery_projection(self) -> dict[str, object] | None:
        """Return unfinished task recovery state for the next Agent run."""
        projection = self.task_recovery.active_projection()
        return dict(projection) if projection is not None else None
