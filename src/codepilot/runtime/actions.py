from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, cast
from uuid import uuid4

from codepilot.core.plan import RunMode, ensure_run_mode
from codepilot.protocols import AssistantMessage, Message, ToolResultMessage, UserMessage
from codepilot.sessions.contracts import SessionCommandRecord, SessionRunRecord, SessionView
from codepilot.tools.contracts import ToolRegistration
from codepilot.tools.security import ApprovalChallenge

from .config import RuntimePermissionMode


ApprovalDecisionValue = Literal["approve", "deny"]
CommandSource = Literal["builtin", "extension", "skill", "prompt"]
_COMMAND_SOURCES = frozenset({"builtin", "extension", "skill", "prompt"})
_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})


@dataclass(frozen=True)
class SessionOpenIntent:
    """Everything an interface can ask for when opening a runtime session."""

    workspace_dir: str | Path
    session_id: str | None = None
    model: Any | None = None
    provider: str | None = None
    model_id: str | None = None
    get_api_key: Any | None = None
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    tools: list[ToolRegistration] = field(default_factory=list)
    memory_enabled: bool = True
    current_mode: str | None = None
    planning_budget_profile: str | None = None
    load_workspace_resources: bool = True
    tool_permission_mode: str | None = None
    enabled_builtin_tools: list[str] | None = None
    stream_fn: Any | None = None
    thinking_level: str | None = None
    max_tool_calls_per_turn: int | None = None
    retry_enabled: bool | None = None
    max_retries: int | None = None
    retry_base_delay_ms: int | None = None
    run_timeout_seconds: int | None = None
    model_context_window: int | None = None
    model_max_output_tokens: int | None = None
    edit_require_unique_match: bool | None = None
    prompt_guidelines: list[str] | None = None
    append_system_prompt: str | None = None
    extension_paths: list[str] | None = None
    skill_paths: list[str] | None = None
    prompt_debug_sources: bool | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    mcp_transport_factory: Any | None = None
    shell_timeout_seconds: int | None = None
    shell_max_timeout_seconds: int | None = None
    shell_stdout_limit: int | None = None
    shell_stderr_limit: int | None = None
    shell_allowed_env: list[str] | None = None
    extension_commands: dict[str, Any] = field(default_factory=dict)
    before_prompt_hooks: list[Any] = field(default_factory=list)
    after_prompt_hooks: list[Any] = field(default_factory=list)
    prepare_context: Any | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage))
            for message in self.messages
        ):
            raise TypeError("SessionOpenIntent.messages must contain protocol Message values")
        if any(not isinstance(tool, ToolRegistration) for tool in self.tools):
            raise TypeError("SessionOpenIntent.tools must contain ToolRegistration values")
        object.__setattr__(self, "messages", list(self.messages))
        object.__setattr__(self, "tools", list(self.tools))


@dataclass(frozen=True)
class SessionRef:
    session_id: str


@dataclass(frozen=True)
class ApprovalView:
    approval_id: str
    session_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    reason: str = ""
    risk_level: str = "unknown"


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
        object.__setattr__(self, "description", _require_view_text(self.description, "description"))
        object.__setattr__(self, "source", _ensure_command_source(self.source))
        object.__setattr__(self, "usage", _optional_text(self.usage) or f"/{self.name}")
        object.__setattr__(self, "group", _optional_text(self.group) or "general")
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
        object.__setattr__(self, "session_id", _require_view_text(self.session_id, "session_id"))
        object.__setattr__(self, "model_id", _require_view_text(self.model_id, "model_id"))
        object.__setattr__(self, "workspace", _require_view_text(self.workspace, "workspace"))
        object.__setattr__(self, "permission_mode", _ensure_permission_mode(self.permission_mode))
        object.__setattr__(self, "current_mode", ensure_run_mode(self.current_mode))
        object.__setattr__(self, "message_count", _ensure_non_negative_int(self.message_count, "message_count"))
        object.__setattr__(self, "leaf_id", _require_view_text(self.leaf_id, "leaf_id"))
        if not isinstance(self.is_running, bool):
            raise TypeError("SessionStatus.is_running must be bool")
        object.__setattr__(self, "credential_source", _require_view_text(self.credential_source, "credential_source"))
        object.__setattr__(self, "warnings", _clean_warnings(self.warnings))
        object.__setattr__(self, "plan_summary", _clean_plan_summary(self.plan_summary))


@dataclass(frozen=True)
class AppSessionView:
    session: SessionView
    status: SessionStatus
    state: Mapping[str, Any] | None = None
    commands: tuple[CommandDescriptor, ...] = ()
    pending_approvals: tuple[ApprovalView, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", MappingProxyType(dict(self.state or {})))
        object.__setattr__(self, "commands", tuple(self.commands))
        object.__setattr__(self, "pending_approvals", tuple(self.pending_approvals))


@dataclass(frozen=True)
class PromptSubmitted:
    text: str
    request_id: str = field(default_factory=lambda: f"request_{uuid4().hex}")
    images: tuple[str, ...] | list[str] | None = None
    mode_hint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "prompt text"))
        object.__setattr__(self, "request_id", _require_text(self.request_id, "request_id"))
        object.__setattr__(self, "images", _clean_images(self.images))
        object.__setattr__(self, "mode_hint", _optional_text(self.mode_hint))


@dataclass(frozen=True)
class CommandSubmitted:
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text(self.text, "command text"))


@dataclass(frozen=True)
class ApprovalDecided:
    approval_id: str
    decision: ApprovalDecisionValue
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "approval_id", _require_text(self.approval_id, "approval_id"))
        object.__setattr__(self, "decision", _approval_decision(self.decision))
        object.__setattr__(self, "reason", _optional_text(self.reason) or "")


@dataclass(frozen=True)
class RunCancelled:
    reason: str = "user"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _optional_text(self.reason) or "user")


UserAction = PromptSubmitted | CommandSubmitted | ApprovalDecided | RunCancelled


@dataclass(frozen=True)
class ProgressFrame:
    event: dict[str, Any]
    kind: Literal["progress"] = field(default="progress", init=False)


@dataclass(frozen=True)
class ApprovalRequiredFrame:
    approval: ApprovalChallenge
    kind: Literal["approval_required"] = field(default="approval_required", init=False)


@dataclass(frozen=True)
class RunFinishedFrame:
    record: SessionRunRecord
    kind: Literal["run_finished"] = field(default="run_finished", init=False)


@dataclass(frozen=True)
class RunPausedFrame:
    record: SessionRunRecord
    checkpoint: dict[str, Any] = field(default_factory=dict)
    kind: Literal["run_paused"] = field(default="run_paused", init=False)


@dataclass(frozen=True)
class CommandFinishedFrame:
    record: SessionCommandRecord
    kind: Literal["command_finished"] = field(default="command_finished", init=False)


@dataclass(frozen=True)
class CancelledFrame:
    session_id: str
    cancelled: bool = False
    reason: str = "user"
    kind: Literal["cancelled"] = field(default="cancelled", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))


@dataclass(frozen=True)
class FailedFrame:
    error: Any
    kind: Literal["failed"] = field(default="failed", init=False)


RuntimeFrame = (
    ProgressFrame
    | ApprovalRequiredFrame
    | RunPausedFrame
    | RunFinishedFrame
    | CommandFinishedFrame
    | CancelledFrame
    | FailedFrame
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text must be a string or None")
    return value.strip() or None


def _clean_images(value: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("images must be a sequence of strings")
    images: list[str] = []
    for item in value:
        images.append(_require_text(item, "image"))
    return tuple(images)


def _approval_decision(value: object) -> ApprovalDecisionValue:
    text = _require_text(value, "approval decision").lower()
    if text not in {"approve", "deny"}:
        raise ValueError(f"Unknown approval decision: {value}")
    return text  # type: ignore[return-value]


def _require_view_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _ensure_command_source(value: object) -> CommandSource:
    normalized = value.strip() if isinstance(value, str) else value
    if normalized not in _COMMAND_SOURCES:
        raise ValueError(f"Unknown command source: {value}")
    return cast(CommandSource, normalized)


def _ensure_permission_mode(value: object) -> RuntimePermissionMode:
    if value not in _PERMISSION_MODES:
        raise ValueError(f"Unknown permission_mode: {value}")
    return cast(RuntimePermissionMode, value)


def _normalize_command_name(value: object) -> str:
    text = _require_view_text(value, "command name").lstrip("/")
    if not text:
        raise ValueError("CommandDescriptor.name cannot be empty")
    if any(char.isspace() for char in text):
        raise ValueError("CommandDescriptor.name cannot contain whitespace")
    return text


def _ensure_non_negative_int(value: object, field_name: str) -> int:
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
    "AppSessionView",
    "ApprovalDecided",
    "ApprovalDecisionValue",
    "ApprovalRequiredFrame",
    "ApprovalView",
    "CancelledFrame",
    "CommandDescriptor",
    "CommandFinishedFrame",
    "CommandSource",
    "CommandSubmitted",
    "FailedFrame",
    "ProgressFrame",
    "PromptSubmitted",
    "RunCancelled",
    "RunFinishedFrame",
    "RunPausedFrame",
    "RuntimeFrame",
    "SessionOpenIntent",
    "SessionRef",
    "SessionStatus",
    "UserAction",
]
