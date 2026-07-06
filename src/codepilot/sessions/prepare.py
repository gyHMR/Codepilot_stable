from __future__ import annotations

# 新手导读：prepare.py 拥有会话打开、run 准备、上下文投影和 run 前副作用。
# 关注点：SessionController 调它获得 PreparedAgentRun；core 不接触 live session 对象。

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from codepilot.core.contracts import (
    AgentContext,
    AgentLoopInput,
    AgentLoopLimits,
    AgentResumeInput,
    ContextPreparationRequest,
    PreparedContext,
    RetryPolicy,
    RunCorrelation,
    TaskStrategy,
)
from codepilot.core.contracts import AgentMessage
from codepilot.core.loop import maybe_await
from codepilot.core.task import ensure_planning_budget_profile, ensure_task_mode
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import AgentEvent, Message, UserMessage
from codepilot.protocols.commands import SessionLifecycleContext, SessionLifecycleView

from . import commands as command_state
from .context.freshness import build_context_freshness_notice
from .context.governor import ContextGovernor
from .context.state import SessionContextState
from .contracts import (
    PreparedAgentRun,
    RollbackBaselineRef,
    SessionOptions,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionView,
)
from .conversation import SessionConversationState
from .history.git_rollback import GitRollbackBaseline
from .history.task_recovery import TaskRecoveryStore
from .memory import MemoryRetriever, MemoryStore, MemoryWriter
from .storage import SessionStore, new_session_id

if TYPE_CHECKING:
    pass

logger = logging.getLogger("codepilot.sessions.prepare")


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
        self._rollback_baselines: dict[str, GitRollbackBaseline] = {}

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

def new_v2_run_id() -> str:
    return f"run_{uuid4().hex[:12]}"

def capture_rollback_baseline_ref(session_id: str, run_id: str) -> RollbackBaselineRef:
    return RollbackBaselineRef(session_id=session_id, run_id=run_id)

def describe_runtime_session(session: Any, *, last_run_id: str | None) -> SessionView:
    """Return the controller-facing view for a live session runtime."""

    return SessionView(
        session_id=session.session_id,
        message_count=len(session.conversation.messages),
        last_run_id=last_run_id,
        task_mode=session.task_mode,
        context=runtime_session_state(session),
    )

def runtime_session_state(session: Any) -> dict[str, Any]:
    """Return command/runtime-facing session state without exposing live stores."""

    return {
        "session_id": session.session_id,
        "message_count": len(session.conversation.messages),
        "entry_ids": command_state.list_entry_ids(session),
        "entries": command_state.list_entries(session),
        "tree": command_state.get_session_tree(session),
        "leaf_id": command_state.get_leaf_id(session),
        "task_mode": session.task_mode,
        "planning_budget_profile": session.planning_budget_profile,
    }

def close_runtime_session(session: Any) -> None:
    """Close a live session runtime without exposing its private close hook."""

    session._close()

def _session_messages_for_loop(session: Any) -> list[Message]:
    """Return the current session transcript plus pending steering messages."""

    return [
        *session.conversation.messages,
        *session.conversation.drain_steering_messages(),
    ]

def _loop_context(session: Any) -> PreparedContext:
    return PreparedContext(
        {
            "system_prompt": session.conversation.system_prompt,
            "session_id": session.session_id,
        }
    )

async def prepare_runtime_run(
    session: Any,
    intent: SessionRunIntent,
    *,
    run_id: str,
    model: ModelDescriptor,
) -> PreparedAgentRun:
    """Prepare a live session runtime for a new agent loop."""

    rollback_baseline = await begin_run_lifecycle(
        session,
        text=intent.text,
        run_id=run_id,
        is_continue=False,
    )
    session_messages = _session_messages_for_loop(session)
    return PreparedAgentRun(
        run_id=run_id,
        session_id=session.session_id,
        loop_input=AgentLoopInput(
            run_id=run_id,
            correlation=RunCorrelation(session_id=session.session_id),
            messages=session_messages,
            user_prompt=intent.text,
            context=_loop_context(session),
            model=model,
            tools=[],
            task_strategy=runtime_task_strategy(session, mode_hint=intent.mode_hint),
            limits=runtime_loop_limits(session),
            retry_policy=runtime_retry_policy(session),
        ),
        context_port=RuntimeSessionContextPort(session),
        input_messages=[UserMessage(content=intent.text)],
        rollback_baseline=_remember_rollback_baseline(session, run_id, rollback_baseline),
        context_refs={"governor": "session_context_governor"},
        memory_refs={"enabled": session.memory_enabled},
        recovery_refs={"projection": session._active_task_recovery_projection()},
    )

async def prepare_runtime_resume(
    session: Any,
    intent: SessionResumeIntent,
    *,
    run_id: str,
    model: ModelDescriptor,
) -> PreparedAgentRun:
    """Prepare a live session runtime for approval resume."""

    rollback_baseline = await begin_run_lifecycle(
        session,
        text="",
        run_id=run_id,
        is_continue=True,
    )
    session_messages = _session_messages_for_loop(session)
    resume_input = AgentResumeInput(
        run_id=run_id,
        correlation=RunCorrelation(session_id=session.session_id),
        messages=session_messages,
        context=_loop_context(session),
        model=model,
        tools=[],
        approval_id=intent.approval_id,
        decision=intent.decision,
        reason=intent.reason,
        task_strategy=runtime_task_strategy(session),
        retry_policy=runtime_retry_policy(session),
    )
    return PreparedAgentRun(
        run_id=run_id,
        session_id=session.session_id,
        loop_input=AgentLoopInput(
            run_id=run_id,
            correlation=RunCorrelation(session_id=session.session_id),
            messages=session_messages,
            context=_loop_context(session),
            model=model,
            tools=[],
            task_strategy=runtime_task_strategy(session),
            limits=runtime_loop_limits(session),
            retry_policy=runtime_retry_policy(session),
        ),
        resume_input=resume_input,
        context_port=RuntimeSessionContextPort(session),
        rollback_baseline=_remember_rollback_baseline(session, run_id, rollback_baseline),
        recovery_refs={"projection": session._active_task_recovery_projection()},
    )

def runtime_retry_policy(session: Any) -> RetryPolicy:
    return RetryPolicy(
        enabled=bool(getattr(session, "retry_enabled", False)),
        max_retries=_int_or_default(getattr(session, "max_retries", 0), default=0),
        base_delay_ms=_int_or_default(
            getattr(session, "retry_base_delay_ms", 0),
            default=0,
        ),
    )

def _remember_rollback_baseline(
    session: Any,
    run_id: str,
    baseline: GitRollbackBaseline,
) -> RollbackBaselineRef:
    baselines = getattr(session, "_rollback_baselines", None)
    if baselines is None:
        baselines = {}
        session._rollback_baselines = baselines
    baselines[run_id] = baseline
    return RollbackBaselineRef(session_id=session.session_id, run_id=run_id)

def _int_or_default(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value

def runtime_task_strategy(
    session: Any,
    *,
    mode_hint: str | None = None,
) -> TaskStrategy:
    return TaskStrategy(
        enabled=bool(getattr(session, "task_control_enabled", False)),
        mode=mode_hint or getattr(session, "task_mode", "build"),
        recovery_projection=session._active_task_recovery_projection(),
        planning_budget_profile=getattr(
            session,
            "planning_budget_profile",
            "balanced",
        ),
        max_replans_per_run=_int_or_default(
            getattr(session, "max_task_replans_per_run", 2),
            default=2,
        ),
    )

def runtime_loop_limits(session: Any) -> AgentLoopLimits:
    return AgentLoopLimits(
        max_tool_calls_per_turn=getattr(session, "max_tool_calls_per_turn", None),
    )

class RuntimeSessionContextPort:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._session
        request_context = request.get("context")
        request_context = request_context if isinstance(request_context, dict) else {}
        raw_task_signal = (
            request_context.get("task_control_signal")
            or request_context.get("task_signal")
        )
        task_signal = raw_task_signal if isinstance(raw_task_signal, dict) else None
        prepared = await maybe_await(
            session.prepare_context(
                AgentContext(
                    system_prompt=str(request.get("system_prompt", "")),
                    messages=list(request.get("messages", ())),
                    tools=list(request.get("tools", ())),
                    current_task=(
                        str(request_context.get("current_task"))
                        if request_context.get("current_task") is not None
                        else None
                    ),
                    task_recovery_projection=session._active_task_recovery_projection(),
                    task_signal=task_signal,
                ),
                ContextPreparationRequest(
                    session_id=session.session_id,
                    model_context_window=session.conversation.model.context_window,
                    model_max_output_tokens=session.conversation.model.max_tokens,
                ),
            )
        )
        report = prepared.report.to_dict()
        session.latest_context_report = report
        session.store.append_event(
            {
                "type": "context_prepared",
                "sessionId": session.session_id,
                "report": report,
            }
        )
        memory_ids = report.get("retrieved_memory_ids")
        if session.memory_enabled and isinstance(memory_ids, list) and memory_ids:
            session.store.append_event(
                {
                    "type": "memory_retrieved",
                    "sessionId": session.session_id,
                    "runId": request.get("run_id"),
                    "memoryIds": memory_ids,
                    "reasons": report.get("memory_retrieval_reasons", {}),
                }
            )
        return {
            "system_prompt": prepared.system_prompt,
            "messages": list(prepared.messages),
            "tools": list(prepared.tools),
            "context_report": report,
        }

async def begin_run_lifecycle(
    session: Any,
    *,
    text: str,
    run_id: str,
    is_continue: bool,
    rollback_baseline: GitRollbackBaseline | None = None,
) -> GitRollbackBaseline:
    """Prepare session-owned state before core runs the agent loop."""

    rollback_baseline = rollback_baseline or command_state.capture_run_rollback_baseline(session)
    await run_lifecycle_hooks(
        session,
        text=text,
        is_continue=is_continue,
        hooks=session.before_prompt_hooks,
    )

    if not is_continue:
        if session.memory_enabled:
            admit_prompt_memory(session, text, run_id=run_id)
        begin_task_recovery(session, text, run_id=run_id)

    check_context_freshness(session)
    return rollback_baseline

def admit_prompt_memory(session: Any, text: str, *, run_id: str | None) -> None:
    """Admit durable project memory from the user prompt when policy allows it."""

    try:
        record = session.memory_writer.admit_prompt_memory(text, run_id=run_id)
        if record is None:
            return
        session.store.append_event(
            {
                "type": "memory_updated",
                "sessionId": session.session_id,
                "memoryId": record.id,
                "kind": record.type,
            }
        )
    except Exception as exc:
        logger.warning("failed to admit prompt memory: %s", exc)
        session.store.append_event(
            {
                "type": "memory_warning",
                "sessionId": session.session_id,
                "operation": "prompt_memory_admission",
                "message": str(exc),
            }
        )

def begin_task_recovery(session: Any, text: str, *, run_id: str | None) -> None:
    """Persist the current task projection outside durable memory."""

    try:
        projection = session.task_recovery.begin_task(text, run_id=run_id)
        session.store.append_event(
            {
                "type": "task_recovery_updated",
                "sessionId": session.session_id,
                "runId": run_id,
                "goal": projection.get("goal"),
            }
        )
    except Exception as exc:
        logger.warning("failed to write task recovery: %s", exc)
        session.store.append_event(
            {
                "type": "task_recovery_warning",
                "sessionId": session.session_id,
                "operation": "task_recovery_begin",
                "message": str(exc),
            }
        )

def check_context_freshness(session: Any) -> None:
    """Check whether previous run-tracked context has become stale."""

    freshness = session.store.run_store.evaluate_freshness()
    if not freshness.should_record_event():
        return
    payload = freshness.to_event_payload()
    session.store.append_event(
        {
            "type": "context_freshness_checked",
            "sessionId": session.session_id,
            "freshness": payload,
        }
    )
    if not freshness.requires_steering():
        return
    notice = build_context_freshness_notice(freshness)
    if notice is not None:
        session.conversation.add_steering_message(notice)

async def run_lifecycle_hooks(
    session: Any,
    *,
    text: str,
    is_continue: bool,
    hooks: list,
) -> None:
    """Run prompt lifecycle hooks with a snapshot view of the session."""

    if not hooks:
        return
    ctx = SessionLifecycleContext(
        text=text,
        is_continue=is_continue,
        message_count=len(session.conversation.messages),
        session_view=SessionLifecycleView(
            session_id=session.session_id,
            workspace_dir=str(session.workspace_dir),
            message_count=len(session.conversation.messages),
            task_mode=str(session.task_mode),
        ),
    )
    for hook in hooks:
        value = hook(ctx)
        if inspect.isawaitable(value):
            await value

__all__ = [
    "SessionRuntime",
    "capture_rollback_baseline_ref",
    "close_runtime_session",
    "describe_runtime_session",
    "new_v2_run_id",
    "prepare_runtime_resume",
    "prepare_runtime_run",
    "runtime_loop_limits",
    "runtime_retry_policy",
    "runtime_session_state",
    "runtime_task_strategy",
]
