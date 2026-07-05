from __future__ import annotations

# 新手导读：tool_hooks.py 定义工具调用 hook 的跨层协议。
# 关注点：hook 只能看到工具调用事实和只读上下文快照，不能接触 core/session live object。

"""Tool-call hook contracts shared by extensions, runtime, sessions, and tools."""

from dataclasses import dataclass, field
from typing import Any

from .content import ImageContent, TextContent
from .messages import AssistantMessage, Message
from .tools import Tool, ToolCall, ToolResult


@dataclass(frozen=True)
class ToolHookContextSnapshot:
    """Protocol-level snapshot visible to tool hooks.

    This replaces the old core ``AgentContext`` dependency.  Hooks may inspect
    the prompt, transcript, tool catalog, and task signal, but they never receive
    a live session, store, memory writer, or agent object.
    """

    run_id: str = ""
    session_id: str | None = None
    system_prompt: str = ""
    messages: tuple[Message, ...] = ()
    tools: tuple[Tool, ...] = ()
    task_signal: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeforeToolCallResult:
    """Result returned by a before-tool hook."""

    block: bool = False
    reason: str | None = None


@dataclass
class AfterToolCallResult:
    """Patch returned by an after-tool hook."""

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None


@dataclass
class BeforeToolCallContext:
    """Context passed before a tool call is executed."""

    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: dict[str, Any]
    context: ToolHookContextSnapshot


@dataclass
class AfterToolCallContext:
    """Context passed after a tool call has produced a result."""

    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: dict[str, Any]
    result: ToolResult
    is_error: bool
    context: ToolHookContextSnapshot


__all__ = [
    "AfterToolCallContext",
    "AfterToolCallResult",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "ToolHookContextSnapshot",
]
