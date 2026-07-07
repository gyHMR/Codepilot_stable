from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

from codepilot.core.task import TaskMode, ensure_task_mode

from .config import RuntimePermissionMode


CommandSource = Literal["builtin", "extension", "skill", "prompt"]
_COMMAND_SOURCES = frozenset({"builtin", "extension", "skill", "prompt"})
_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})


@dataclass(frozen=True)
class CommandDescriptor:
    """Runtime command definition rendered by interfaces."""

    name: str
    description: str
    source: CommandSource

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_command_name(self.name))
        object.__setattr__(
            self,
            "description",
            _require_text(self.description, field_name="description"),
        )
        object.__setattr__(self, "source", _ensure_command_source(self.source))

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
        }


@dataclass(frozen=True)
class SessionStatus:
    """Session status view rendered by interfaces."""

    session_id: str
    model_id: str
    workspace: str
    permission_mode: RuntimePermissionMode
    message_count: int
    leaf_id: str
    task_mode: TaskMode = "build"
    is_running: bool = False
    credential_source: str = "unknown"
    warnings: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, field_name="session_id"))
        object.__setattr__(self, "model_id", _require_text(self.model_id, field_name="model_id"))
        object.__setattr__(self, "workspace", _require_text(self.workspace, field_name="workspace"))
        object.__setattr__(self, "permission_mode", _ensure_permission_mode(self.permission_mode))
        object.__setattr__(self, "task_mode", ensure_task_mode(self.task_mode))
        object.__setattr__(
            self,
            "message_count",
            _ensure_non_negative_int(self.message_count, field_name="message_count"),
        )
        object.__setattr__(self, "leaf_id", _require_text(self.leaf_id, field_name="leaf_id"))
        if not isinstance(self.is_running, bool):
            raise TypeError("SessionStatus.is_running must be bool")
        object.__setattr__(
            self,
            "credential_source",
            _require_text(self.credential_source, field_name="credential_source"),
        )
        object.__setattr__(self, "warnings", _clean_warnings(self.warnings))


def builtin_commands() -> list[CommandDescriptor]:
    """Return built-in application commands rendered by interfaces."""

    return [
        CommandDescriptor(name="help", description="显示可用命令", source="builtin"),
        CommandDescriptor(name="status", description="查看模型、工作区、会话和权限状态", source="builtin"),
        CommandDescriptor(name="mode", description="查看或切换任务模式：read/plan/build", source="builtin"),
        CommandDescriptor(name="session", description="查看当前会话与叶子节点", source="builtin"),
        CommandDescriptor(name="tree", description="查看当前会话树", source="builtin"),
        CommandDescriptor(name="path", description="查看指定节点路径", source="builtin"),
        CommandDescriptor(name="fork", description="从指定节点分叉新会话", source="builtin"),
        CommandDescriptor(name="new", description="等价于从当前叶子分叉新会话", source="builtin"),
        CommandDescriptor(name="switch", description="切换到指定叶子节点", source="builtin"),
        CommandDescriptor(name="clear", description="清空上下文，创建新会话", source="builtin"),
        CommandDescriptor(name="context", description="查看最近一次上下文投影治理报告", source="builtin"),
        CommandDescriptor(name="memory", description="查看、添加、提升或删除结构化记忆", source="builtin"),
        CommandDescriptor(name="rollback", description="预览或执行最近一次 run 的 Git 回退", source="builtin"),
        CommandDescriptor(name="tools", description="查看当前可用工具", source="builtin"),
        CommandDescriptor(name="approve", description="批准等待中的工具调用：/approve <approval_id>", source="builtin"),
        CommandDescriptor(name="deny", description="拒绝等待中的工具调用：/deny <approval_id>", source="builtin"),
        CommandDescriptor(name="model", description="查看当前模型信息", source="builtin"),
        CommandDescriptor(name="usage", description="查看 token 用量和费用", source="builtin"),
        CommandDescriptor(name="exit", description="退出 Codepilot", source="builtin"),
    ]


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name=field_name)


def _ensure_command_source(value: object) -> CommandSource:
    if isinstance(value, str):
        value = value.strip()
    if value not in _COMMAND_SOURCES:
        raise ValueError(f"Unknown command source: {value}")
    return cast(CommandSource, value)


def _ensure_permission_mode(value: object) -> RuntimePermissionMode:
    if value not in _PERMISSION_MODES:
        raise ValueError(f"Unknown permission_mode: {value}")
    return cast(RuntimePermissionMode, value)


def _normalize_command_name(value: object) -> str:
    text = _require_text(value, field_name="command name").lstrip("/")
    if not text:
        raise ValueError("CommandDescriptor.name cannot be empty")
    if any(char.isspace() for char in text):
        raise ValueError("CommandDescriptor.name cannot contain whitespace")
    return text


def _ensure_non_negative_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an int")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _clean_warnings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError("SessionStatus.warnings must be a sequence of strings")
    warnings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("SessionStatus.warnings must contain strings")
        text = item.strip()
        if text:
            warnings.append(text)
    return tuple(warnings)


__all__ = [
    "builtin_commands",
    "CommandDescriptor",
    "CommandSource",
    "SessionStatus",
]
