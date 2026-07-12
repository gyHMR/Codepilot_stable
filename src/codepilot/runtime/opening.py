from __future__ import annotations

"""Public session-opening intent and views returned by the runtime gateway."""

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from codepilot.protocols import AssistantMessage, Message, ToolResultMessage, UserMessage
from codepilot.sessions.contracts import SessionView
from codepilot.tools.contracts import ToolRegistration

from .approvals import ApprovalView
from .views import CommandDescriptor, SessionStatus


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
    tool_execution: str | None = None
    max_tool_calls_per_turn: int | None = None
    retry_enabled: bool | None = None
    max_retries: int | None = None
    retry_base_delay_ms: int | None = None
    model_context_window: int | None = None
    model_max_output_tokens: int | None = None
    block_dangerous_bash: bool | None = None
    bash_allow_patterns: list[str] | None = None
    bash_block_patterns: list[str] | None = None
    edit_require_unique_match: bool | None = None
    prompt_guidelines: list[str] | None = None
    append_system_prompt: str | None = None
    tool_snippets: dict[str, str] | None = None
    extension_paths: list[str] | None = None
    skill_paths: list[str] | None = None
    prompt_debug_sources: bool | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    mcp_client: Any | None = None
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


__all__ = [
    "AppSessionView",
    "SessionOpenIntent",
    "SessionRef",
]
