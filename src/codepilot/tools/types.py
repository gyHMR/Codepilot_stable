from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from codepilot.core import AgentTool, AgentToolResult

ToolRiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    category: str
    read_only: bool
    concurrency_safe: bool
    exclusive: bool
    requires_approval: bool
    risk_level: ToolRiskLevel
    resource_scope: tuple[str, ...]
    network_access: bool = False
    credential_required: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolRuntimeRequest:
    tool_call_id: str
    name: str
    params: dict[str, Any]
    source: str = "agent"


@dataclass(frozen=True)
class ToolRuntimeResult:
    result: AgentToolResult
    is_error: bool = False
    approved: bool = True
    approval_id: str | None = None


__all__ = [
    "AgentTool",
    "AgentToolResult",
    "ToolMetadata",
    "ToolRiskLevel",
    "ToolRuntimeRequest",
    "ToolRuntimeResult",
]
