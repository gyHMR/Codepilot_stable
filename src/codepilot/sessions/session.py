from __future__ import annotations

"""AgentSession 负责编排一次应用级别的 Agent 对话。

主要职责：
1) 管理工作区会话目录。
2) 持久化 Agent 事件和消息。
3) 提供稳定的 run（发送任务）和 continue（继续运行）入口。
4) 执行上下文溢出检测与压缩（compaction）。
"""

from pathlib import Path
import inspect
import logging
from typing import Callable

from codepilot.llm.overflow import estimate_context_tokens, is_context_overflow
from codepilot.llm.api_registry import complete_simple
from codepilot.protocols import (
    AgentRunResult,
    AssistantMessage,
    Context,
    Message,
    SimpleStreamOptions,
    TextContent,
    UserMessage,
)
from codepilot.core import Agent, AgentEvent, AgentMessage, AgentOptions

from codepilot.extensions.types import ExtensionLifecycleContext
from .branching import fork_session as branch_fork_session
from .branching import switch_session as branch_switch_session
from .branching import switch_to_entry as branch_switch_to_entry
from .checkpoint import SessionCheckpoint, record_checkpoint
from .compaction import (
    COMPACTION_SYSTEM_PROMPT,
    build_compacted_context,
    fallback_summary,
    format_messages_for_summary,
)
from .store import SessionStore, new_session_id
from .types import AgentSessionOptions

logger = logging.getLogger("codepilot.sessions.session")


class AgentSession:
    """Agent 会话管理类。

    封装了一个完整的 Agent 对话生命周期，包括：
    - 消息的收发与持久化存储
    - 上下文窗口的溢出检测与自动压缩
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

        # 加载已持久化的历史消息；优先读取会话消息，若无则读取上下文消息
        persisted_messages = self.store.load_session_messages()
        if not persisted_messages:
            persisted_messages = self.store.load_context_messages()
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
            get_api_key=options.get_api_key,
            before_tool_call=options.before_tool_call,
            after_tool_call=options.after_tool_call,
            retry_enabled=options.retry_enabled,
            max_model_retries=options.max_retries,
            retry_base_delay_ms=options.retry_base_delay_ms,
            session_id=self.session_id,
        )
        if options.convert_to_llm is not None:
            agent_opts.convert_to_llm = options.convert_to_llm

        # 创建核心 Agent 实例
        self.agent = Agent(agent_opts)

        # 上下文管理相关配置
        self.max_context_messages = options.max_context_messages       # 消息数量上限
        self.max_context_tokens = options.max_context_tokens           # token 数量上限
        self.retain_recent_messages = options.retain_recent_messages   # 压缩时保留的最近消息数
        self.summary_builder = options.summary_builder                 # 自定义摘要构建器

        self.tool_execution = options.tool_execution
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

    async def continue_run(self) -> AgentRunResult:
        """继续上一次未完成的 Agent 运行（例如工具调用后的延续）。

        Returns:
            继续运行产生的结构化 Run 结果。
        """
        await self._run_lifecycle_hooks(text="", is_continue=True, hooks=self.before_prompt_hooks)
        result = await self.agent.continue_run()
        self.store.append_run_result(result)
        await self._compact_context_if_needed()
        await self._run_lifecycle_hooks(text="", is_continue=True, hooks=self.after_prompt_hooks)
        return result

    async def run(
        self,
        text: str,
        *,
        images: list[str] | None = None,
    ) -> AgentRunResult:
        """Run one user task and persist its structured result."""

        await self._run_lifecycle_hooks(
            text=text,
            is_continue=False,
            hooks=self.before_prompt_hooks,
        )
        self._check_context_freshness()
        await self._check_and_compact_before_prompt()
        result = await self.agent.run(text, images=images)
        self.store.append_run_result(result)
        await self._compact_context_if_needed()
        await self._run_lifecycle_hooks(
            text=text,
            is_continue=False,
            hooks=self.after_prompt_hooks,
        )
        return result

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

    def record_checkpoint(self, label: str, details: dict | None = None) -> SessionCheckpoint:
        """在当前会话中记录一个检查点（checkpoint）。

        Args:
            label: 检查点的标签名称。
            details: 可选的附加详情字典。

        Returns:
            创建的 SessionCheckpoint 实例。
        """
        return record_checkpoint(self.store, session_id=self.session_id, label=label, details=details)

    # ── 内部方法 ────────────────────────────────────────────────

    async def _on_agent_event(self, event: AgentEvent) -> None:
        """Agent 事件回调：将事件持久化到存储，并在消息结束时保存上下文消息。"""
        self.store.append_event(event)
        if event["type"] == "message_end":
            message = event["message"]
            self.store.append_context_message(message)

    def _check_context_freshness(self) -> None:
        freshness = self.store.run_store.evaluate_freshness()
        if not freshness.checked_paths and freshness.status == "valid":
            return
        payload = freshness.to_event_payload()
        self.store.append_event(
            {
                "type": "context_freshness_checked",
                "sessionId": self.session_id,
                "freshness": payload,
            }
        )
        if freshness.status == "valid":
            return
        lines = [
            "[Context Freshness]",
            f"status={freshness.status}",
        ]
        if freshness.changed_paths:
            lines.append("changed_files=" + ", ".join(freshness.changed_paths))
        if freshness.missing_paths:
            lines.append("missing_files=" + ", ".join(freshness.missing_paths))
        lines.append("旧工具结果可能已过期；依赖这些文件前请重新读取。")
        self.agent.add_steering_message(
            UserMessage(
                content=[TextContent(text="\n".join(lines))],
                metadata={"context_freshness": payload},
            )
        )

    async def _run_lifecycle_hooks(
        self,
        *,
        text: str,
        is_continue: bool,
        hooks: list,
    ) -> None:
        """执行生命周期钩子列表。

        钩子可以是同步或异步函数，统一通过 inspect.isawaitable 兼容处理。

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

    async def _check_and_compact_before_prompt(self) -> None:
        """在调用 LLM 之前检测上下文是否溢出。

        如果已溢出，则强制触发上下文压缩，以确保请求不会因超出窗口而失败。
        """
        model = self.agent.state.model
        ctx = Context(
            messages=self.agent.state.messages,
            system_prompt=self.agent.state.system_prompt,
            tools=self.agent.state.tools,
        )
        if is_context_overflow(model, ctx):
            logger.warning(
                "context overflow detected before prompt, triggering compaction session_id=%s",
                self.session_id,
            )
            await self._compact_context_if_needed(force=True)

    async def _compact_context_if_needed(self, *, force: bool = False) -> None:
        """根据需要压缩上下文消息。

        触发条件（满足任一即触发）：
        - force=True（强制压缩）
        - 消息数量超过 max_context_messages
        - 估算 token 数超过 max_context_tokens

        压缩策略：
        1. 保留最近的 retain_recent_messages 条消息不动
        2. 将更早的消息交给 LLM 生成摘要（或使用自定义摘要构建器）
        3. 用一条摘要消息替换所有旧消息

        Args:
            force: 是否强制执行压缩（忽略阈值判断）。
        """
        max_messages = self.max_context_messages
        max_tokens = self.max_context_tokens
        over_message_limit = bool(max_messages and max_messages > 0 and len(self.agent.state.messages) > max_messages)
        estimated_tokens = estimate_context_tokens(self.agent.state.messages, self.agent.state.system_prompt)
        over_token_limit = bool(max_tokens and max_tokens > 0 and estimated_tokens > max_tokens)

        # 未触发任何压缩条件，直接返回
        if not force and not over_message_limit and not over_token_limit:
            return

        messages = list(self.agent.state.messages)
        # 至少保留 2 条消息，最多保留 retain_recent_messages 条
        retain = max(2, min(self.retain_recent_messages, len(messages) - 1))
        if len(messages) <= retain:
            return

        # 分割：旧消息（待压缩）和近期消息（保留原样）
        older = messages[:-retain]

        # 生成摘要：优先使用自定义构建器，否则调用 LLM
        if self.summary_builder:
            summary_text = self.summary_builder(older).strip()
        else:
            summary_text = await self._llm_summary(older)

        # LLM 摘要失败时，使用基于规则的降级摘要
        if not summary_text:
            summary_text = self._fallback_summary(older)

        reason = "overflow" if force else ("token_threshold" if over_token_limit else "message_threshold")
        compacted_context = build_compacted_context(
            messages=messages,
            summary_text=summary_text,
            retain_recent_messages=self.retain_recent_messages,
            reason=reason,
            system_prompt=self.agent.state.system_prompt,
        )

        # 更新 Agent 状态和持久化存储
        self.agent.set_messages(compacted_context.messages)
        self.store.rewrite_context_messages(compacted_context.messages)
        # 记录压缩事件，便于调试和审计
        self.store.append_event(
            {
                "type": "context_compacted",
                "sessionId": self.session_id,
                "before_count": len(messages),
                "after_count": len(compacted_context.messages),
                "retained_recent": retain,
                "estimated_tokens_before": estimated_tokens,
                "reason": reason,
                "report": compacted_context.report,
            }
        )
        logger.info(
            "context compacted session_id=%s before=%d after=%d",
            self.session_id, len(messages), len(compacted_context.messages),
        )

    async def _llm_summary(self, messages: list[Message]) -> str:
        """使用当前 LLM 对历史消息生成上下文摘要。

        Args:
            messages: 需要压缩的旧消息列表。

        Returns:
            生成的摘要文本；失败时返回空字符串。
        """
        formatted = self._format_messages_for_summary(messages)
        if not formatted.strip():
            return ""

        try:
            summary_context = Context(
                messages=[UserMessage(content=f"请压缩以下对话历史为简明摘要：\n\n{formatted}")],
                system_prompt=COMPACTION_SYSTEM_PROMPT,
            )
            model = self.agent.state.model
            api_key = None
            if self.get_api_key is not None:
                value = self.get_api_key(model.provider)
                api_key = await value if inspect.isawaitable(value) else value
            result = await complete_simple(
                model,
                summary_context,
                SimpleStreamOptions(max_tokens=2000, api_key=api_key),
            )
            text_parts = [b.text for b in result.content if isinstance(b, TextContent)]
            summary = "\n".join(text_parts).strip()
            if summary:
                logger.info("LLM compaction summary generated chars=%d", len(summary))
                return summary
        except Exception as exc:
            logger.warning("LLM compaction failed, using fallback: %s", exc)

        return ""

    @staticmethod
    def _format_messages_for_summary(messages: list[Message]) -> str:
        """将消息列表格式化为供摘要使用的文本格式。"""
        return format_messages_for_summary(messages)

    @staticmethod
    def _fallback_summary(messages: list[Message]) -> str:
        """当 LLM 摘要不可用时，使用基于规则的方式生成降级摘要。"""
        return fallback_summary(messages)
