from __future__ import annotations

# 新手导读：session_opening.py 定义接口层打开会话时能提交的应用意图。
# 关注点：SessionOpenIntent 是 runtime public DTO；RuntimeAssemblyIntent 是 runtime 内部装配 DTO。

"""Public session-opening intent plus runtime-internal assembly conversion."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from codepilot.sessions.contracts import SessionView

from .approvals import ApprovalView
from .assemble import RuntimeAssemblyIntent
from .views import CommandDescriptor, SessionStatus


@dataclass(frozen=True)
class SessionOpenIntent:
    workspace_dir: str | Path
    session_id: str | None = None
    model: Any | None = None
    provider: str | None = None
    model_id: str | None = None
    get_api_key: Any | None = None
    tools: list[Any] | None = None
    memory_enabled: bool = True
    task_control_enabled: bool = True
    task_mode: str | None = None
    planning_budget_profile: str | None = None
    read_only_mode: bool | None = None
    load_workspace_resources: bool = True
    tool_permission_mode: str | None = None
    enabled_builtin_tools: list[str] | None = None
    approval_provider: Any | None = None
    stream_fn: Any | None = None
    retry_enabled: bool | None = None
    max_retries: int | None = None
    retry_base_delay_ms: int | None = None


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


def _to_runtime_assembly_intent(intent: SessionOpenIntent) -> RuntimeAssemblyIntent:
    """Convert public open intent into the richer internal assembly request."""

    return RuntimeAssemblyIntent(
        workspace_dir=intent.workspace_dir,
        model=intent.model,
        provider=intent.provider,
        model_id=intent.model_id,
        get_api_key=intent.get_api_key,
        tools=list(intent.tools or ()),
        session_id=intent.session_id,
        memory_enabled=intent.memory_enabled,
        task_control_enabled=intent.task_control_enabled,
        task_mode=intent.task_mode,  # type: ignore[arg-type]
        planning_budget_profile=intent.planning_budget_profile,  # type: ignore[arg-type]
        read_only_mode=intent.read_only_mode,
        load_workspace_resources=intent.load_workspace_resources,
        tool_permission_mode=intent.tool_permission_mode,  # type: ignore[arg-type]
        enabled_builtin_tools=intent.enabled_builtin_tools,
        approval_provider=intent.approval_provider,
        stream_fn=intent.stream_fn,
        retry_enabled=intent.retry_enabled,
        max_retries=intent.max_retries,
        retry_base_delay_ms=intent.retry_base_delay_ms,
    )


__all__ = [
    "AppSessionView",
    "SessionOpenIntent",
    "SessionRef",
]
