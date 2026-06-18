from __future__ import annotations

"""Session branching and switching helpers."""

from typing import TYPE_CHECKING

from .store import SessionStore, new_session_id
from .types import AgentSessionOptions

if TYPE_CHECKING:
    from .session import AgentSession


def build_session_options_from_existing(
    session: "AgentSession",
    *,
    session_id: str,
    system_prompt: str | None = None,
) -> AgentSessionOptions:
    """Create low-level options that preserve an existing session runtime shape."""

    return AgentSessionOptions(
        model=session.agent.state.model,
        workspace_dir=session.workspace_dir,
        system_prompt=system_prompt if system_prompt is not None else session.agent.state.system_prompt,
        tools=list(session.agent.state.tools),
        session_id=session_id,
        messages=[],
        thinking_level=session.agent.state.thinking_level,
        tool_execution=session.tool_execution,
        get_api_key=session.get_api_key,
        max_context_messages=session.max_context_messages,
        max_context_tokens=session.max_context_tokens,
        retain_recent_messages=session.retain_recent_messages,
        summary_builder=session.summary_builder,
        retry_enabled=session.retry_enabled,
        max_retries=session.max_retries,
        retry_base_delay_ms=session.retry_base_delay_ms,
        extension_commands=session.extension_commands,
        before_prompt_hooks=session.before_prompt_hooks,
        after_prompt_hooks=session.after_prompt_hooks,
        before_tool_call=session.before_tool_call,
        after_tool_call=session.after_tool_call,
    )


def create_fresh_session(old: "AgentSession") -> "AgentSession":
    """Create an empty sibling session while preserving runtime settings."""

    from .session import AgentSession

    return AgentSession(
        build_session_options_from_existing(
            old,
            session_id=new_session_id(),
        )
    )


def fork_session(session: "AgentSession", from_entry_id: str | None = None) -> "AgentSession":
    """Fork a session from a specific entry or the current leaf."""

    from .session import AgentSession

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
    """Switch the current session leaf and restore messages from that entry."""

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
    """Switch this facade to another persisted session id in the same workspace."""

    new_store = SessionStore(session.workspace_dir, session_id)
    meta = new_store.read_meta()
    if not meta:
        raise ValueError(f"Session not found: {session_id}")

    session.session_id = session_id
    session.store = new_store
    restored = new_store.load_session_messages()
    if not restored:
        restored = new_store.load_context_messages()
    session.agent.set_messages(restored)
