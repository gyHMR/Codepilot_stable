from __future__ import annotations

"""会话分支与切换辅助函数。"""

from typing import TYPE_CHECKING

from ..persistence.store import SessionStore, new_session_id
from ..types import AgentSessionOptions

if TYPE_CHECKING:
    from ..session import AgentSession


def build_session_options_from_existing(
    session: "AgentSession",
    *,
    session_id: str,
    system_prompt: str | None = None,
) -> AgentSessionOptions:
    """从已有会话构建底层选项，保留运行时配置（模型、工具、钩子等）。"""

    return AgentSessionOptions(
        model=session.agent.state.model,
        workspace_dir=session.workspace_dir,
        system_prompt=system_prompt if system_prompt is not None else session.agent.state.system_prompt,
        tools=list(session.agent.state.tools),
        session_id=session_id,
        messages=[],
        thinking_level=session.agent.state.thinking_level,
        tool_execution=session.tool_execution,
        max_tool_calls_per_turn=session.max_tool_calls_per_turn,
        memory_enabled=session.memory_enabled,
        task_mode=session.task_mode,
        planning_budget_profile=session.planning_budget_profile,
        get_api_key=session.get_api_key,
        retry_enabled=session.retry_enabled,
        max_retries=session.max_retries,
        retry_base_delay_ms=session.retry_base_delay_ms,
        extension_commands=session.extension_commands,
        before_prompt_hooks=session.before_prompt_hooks,
        after_prompt_hooks=session.after_prompt_hooks,
        before_tool_call=session.before_tool_call,
        after_tool_call=session.after_tool_call,
        stream_fn=session.stream_fn,
    )


def create_fresh_session(old: "AgentSession") -> "AgentSession":
    """创建一个空的兄弟会话（保留运行时设置，清空消息历史）。"""

    from ..session import AgentSession

    return AgentSession(
        build_session_options_from_existing(
            old,
            session_id=new_session_id(),
        )
    )


def fork_session(session: "AgentSession", from_entry_id: str | None = None) -> "AgentSession":
    """从指定条目或当前叶子分叉一个新会话。"""

    from ..session import AgentSession

    new_id = new_session_id()
    fork_store = session.store.fork_to(new_id, from_entry_id=from_entry_id)
    meta = fork_store.read_meta() or {}
    system_prompt = str(meta.get("system_prompt", session.agent.state.system_prompt))
    return AgentSession(
        build_session_options_from_existing(
            session,
            session_id=new_id,
            system_prompt=system_prompt,
        )
    )


def switch_to_entry(session: "AgentSession", entry_id: str) -> None:
    """切换当前会话叶子到指定条目，并恢复该分支的消息。"""

    session.store.set_leaf(entry_id)
    restored = session.store.load_session_messages(leaf_id=entry_id)
    session.agent.set_messages(restored)
    session.store.append_event(
        {
            "type": "session_switch_entry",
            "session_id": session.session_id,
            "entry_id": entry_id,
        }
    )


def switch_session(session: "AgentSession", session_id: str) -> None:
    """切换到同一工作区中的另一个已持久化的会话。"""

    new_store = SessionStore(session.workspace_dir, session_id)
    meta = new_store.read_meta()
    if not meta:
        raise ValueError(f"Session not found: {session_id}")

    session.session_id = session_id
    session.rebind_store(new_store)
    restored = new_store.load_session_messages()
    session.agent.set_messages(restored)
