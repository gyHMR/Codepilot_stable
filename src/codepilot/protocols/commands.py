from __future__ import annotations

# 新手导读：commands.py 定义外部能力暴露 slash command、生命周期 hook 和工具 hook 的协议。
# 关注点：这里不执行命令，也不持有 session；extensions 生产这些能力，sessions 消费这些能力。

"""Command and hook capability contracts shared across layers."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, cast

from .conversation import AssistantMessage, ImageContent, Message, TextContent, ToolCall
from .tools import Tool, ToolResult


CommandSource = Literal["extension", "skill", "builtin", "prompt"]
_COMMAND_SOURCES = frozenset({"extension", "skill", "builtin", "prompt"})


@dataclass(frozen=True)
class SessionLifecycleView:
    """Read-only session facts available to lifecycle hooks."""

    session_id: str
    workspace_dir: str
    message_count: int
    task_mode: str


@dataclass(frozen=True)
class SessionLifecycleContext:
    """Session lifecycle hook context passed to before/after prompt hooks."""

    text: str
    is_continue: bool
    message_count: int
    session_view: SessionLifecycleView | None = None


LifecycleHook = Callable[[SessionLifecycleContext], None | Awaitable[None]]


@dataclass(frozen=True)
class SessionCommandView:
    """Read-only session facts available to extension and skill commands."""

    session_id: str
    workspace_dir: str
    message_count: int
    task_mode: str
    leaf_id: str | None = None


@dataclass(frozen=True)
class SessionCommandContext:
    """Slash-command execution context shared by extensions, skills, and sessions."""

    name: str
    args: list[str]
    raw_text: str
    session_view: SessionCommandView | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _require_command_text(self.name, field_name="command context name"),
        )
        if isinstance(self.args, (str, bytes)):
            raise TypeError("SessionCommandContext.args must be a list of strings")
        object.__setattr__(self, "args", [str(arg).strip() for arg in self.args if str(arg).strip()])
        object.__setattr__(self, "raw_text", str(self.raw_text or ""))
        if self.session_view is not None and not isinstance(
            self.session_view,
            SessionCommandView,
        ):
            raise TypeError("SessionCommandContext.session_view must be SessionCommandView")


CommandHandler = Callable[[SessionCommandContext], str | None | Awaitable[str | None]]


@dataclass(frozen=True)
class RegisteredCommand:
    """Registered slash command exposed through command routing."""

    name: str
    handler: CommandHandler
    description: str | None = None
    source: CommandSource = "extension"

    def __post_init__(self) -> None:
        name = _require_command_text(self.name, field_name="command name").lstrip("/")
        if not name:
            raise ValueError("command name cannot be empty")
        if any(char.isspace() for char in name):
            raise ValueError("command name cannot contain whitespace")
        if not callable(self.handler):
            raise TypeError("RegisteredCommand.handler must be callable")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "description",
            _optional_command_text(self.description, field_name="command description"),
        )
        object.__setattr__(self, "source", _ensure_command_source(self.source))


@dataclass(frozen=True)
class ToolHookContextSnapshot:
    """Read-only context visible to before/after tool hooks."""

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


def _require_command_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_command_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_command_text(value, field_name=field_name)


def _ensure_command_source(value: object) -> CommandSource:
    text = _require_command_text(value, field_name="command source")
    if text not in _COMMAND_SOURCES:
        raise ValueError(f"Unknown command source: {value}")
    return cast(CommandSource, text)


__all__ = [
    "AfterToolCallContext",
    "AfterToolCallResult",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "CommandHandler",
    "CommandSource",
    "LifecycleHook",
    "RegisteredCommand",
    "SessionCommandContext",
    "SessionCommandView",
    "SessionLifecycleContext",
    "SessionLifecycleView",
    "ToolHookContextSnapshot",
]
