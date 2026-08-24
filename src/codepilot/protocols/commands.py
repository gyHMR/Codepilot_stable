"""定义跨层共享的命令与生命周期钩子契约。

本模块只描述扩展、Skill 与 Session 之间交换的只读视图、调用上下文和返回结果。
它不解析命令、不执行处理器，也不持有 Session 状态；能力由扩展层提供，由运行时或
Session 协调器负责路由。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, cast



CommandSource = Literal["extension", "skill", "builtin", "prompt"]
_COMMAND_SOURCES = frozenset({"extension", "skill", "builtin", "prompt"})


@dataclass(frozen=True)
class SessionLifecycleView:
    """生命周期钩子可读取的 Session 事实快照，调用方不得通过它修改会话。"""

    session_id: str
    workspace_dir: str
    message_count: int
    current_mode: str


@dataclass(frozen=True)
class SessionLifecycleContext:
    """提示词处理前后传给生命周期钩子的上下文。"""

    text: str
    is_continue: bool
    message_count: int
    session_view: SessionLifecycleView | None = None


LifecycleHook = Callable[[SessionLifecycleContext], None | Awaitable[None]]


@dataclass(frozen=True)
class SessionCommandView:
    """扩展命令和 Skill 命令可读取的 Session 事实快照。"""

    session_id: str
    workspace_dir: str
    message_count: int
    current_mode: str
    leaf_id: str | None = None


@dataclass(frozen=True)
class SessionCommandContext:
    """斜杠命令的统一调用上下文。

    ``name`` 不含前导斜杠；``args`` 在初始化时去除空白项；``session_view`` 只提供
    当前会话的只读投影，命令实现不能把它当作 Session 权威状态。
    """

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


@dataclass(frozen=True)
class CommandOutcome:
    """命令的结构化结果，可直接输出文本或转入一次普通模型 Run。

    ``output`` 表示命令已经产生可展示结果，``prompt`` 表示应把文本交给正常 Run
    主链路继续处理；两者至少存在一个。
    """

    output: str | None = None
    prompt: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _optional_command_text(self.output, field_name="command output"))
        object.__setattr__(self, "prompt", _optional_command_text(self.prompt, field_name="command prompt"))
        if self.output is None and self.prompt is None:
            raise ValueError("CommandOutcome requires output or prompt")


CommandHandlerResult = str | CommandOutcome | None
CommandHandler = Callable[
    [SessionCommandContext],
    CommandHandlerResult | Awaitable[CommandHandlerResult],
]


@dataclass(frozen=True)
class RegisteredCommand:
    """注册到统一命令路由中的斜杠命令声明。"""

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
    "CommandHandler",
    "CommandHandlerResult",
    "CommandOutcome",
    "CommandSource",
    "LifecycleHook",
    "RegisteredCommand",
    "SessionCommandContext",
    "SessionCommandView",
    "SessionLifecycleContext",
    "SessionLifecycleView",
]
