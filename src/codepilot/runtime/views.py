from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

from codepilot.core.plan import RunMode, ensure_run_mode

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
    usage: str = ""
    group: str = "general"
    visible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_command_name(self.name))
        object.__setattr__(
            self,
            "description",
            _require_text(self.description, field_name="description"),
        )
        object.__setattr__(self, "source", _ensure_command_source(self.source))
        object.__setattr__(self, "usage", _optional_text(self.usage, field_name="usage") or f"/{self.name}")
        object.__setattr__(self, "group", _optional_text(self.group, field_name="group") or "general")
        if not isinstance(self.visible, bool):
            raise TypeError("CommandDescriptor.visible must be bool")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "usage": self.usage,
            "group": self.group,
            "visible": str(self.visible).lower(),
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
    current_mode: RunMode = "build"
    is_running: bool = False
    credential_source: str = "unknown"
    warnings: tuple[str, ...] | None = None
    plan_summary: dict[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, field_name="session_id"))
        object.__setattr__(self, "model_id", _require_text(self.model_id, field_name="model_id"))
        object.__setattr__(self, "workspace", _require_text(self.workspace, field_name="workspace"))
        object.__setattr__(self, "permission_mode", _ensure_permission_mode(self.permission_mode))
        object.__setattr__(self, "current_mode", ensure_run_mode(self.current_mode))
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
        object.__setattr__(self, "plan_summary", _clean_plan_summary(self.plan_summary))


def builtin_commands() -> list[CommandDescriptor]:
    """Return built-in application commands rendered by interfaces."""

    return [
        CommandDescriptor(name="help", description="显示可用命令", source="builtin", group="core"),
        CommandDescriptor(name="status", description="查看模型、工作区、会话、权限和计划摘要", source="builtin", group="core"),
        CommandDescriptor(name="resume", description="列出历史会话或切换到指定会话", source="builtin", usage="/resume [number|session_id]", group="session"),
        CommandDescriptor(name="new", description="创建空白新会话并切换", source="builtin", group="session"),
        CommandDescriptor(name="fork", description="从当前会话复制一份新会话并切换", source="builtin", group="session"),
        CommandDescriptor(name="mode", description="查看或切换运行模式：read/plan/build", source="builtin", usage="/mode [read|plan|build]", group="workflow"),
        CommandDescriptor(name="plan", description="查看、批准、拒绝或清除当前计划", source="builtin", usage="/plan [approve|reject|clear]", group="workflow"),
        CommandDescriptor(name="tools", description="查看当前可用工具", source="builtin", group="workflow"),
        CommandDescriptor(name="model", description="查看当前模型信息", source="builtin", group="system"),
        CommandDescriptor(name="usage", description="查看 token 用量和费用", source="builtin", group="system"),
        CommandDescriptor(name="rollback", description="预览或执行最近一次 run 的 Git 回退", source="builtin", usage="/rollback [apply] [run_id]", group="system"),
        CommandDescriptor(name="memory", description="查看、添加、提升或删除结构化记忆", source="builtin", usage="/memory [list|search|add|approve|edit|disable|delete]", group="system"),
        CommandDescriptor(name="context", description="查看最近一次上下文投影治理报告", source="builtin", group="system"),
        CommandDescriptor(name="exit", description="退出 Codepilot", source="builtin", group="core"),
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
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    return text or None


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


def _clean_plan_summary(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("SessionStatus.plan_summary must be a dict")
    return dict(value)


__all__ = [
    "builtin_commands",
    "CommandDescriptor",
    "CommandSource",
    "SessionStatus",
]
