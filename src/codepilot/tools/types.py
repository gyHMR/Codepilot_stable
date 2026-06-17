from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codepilot.core import AgentTool, AgentToolResult
from codepilot.protocols.tools import (
    ToolMetadata,
    ToolResultStatus,
    ToolRiskLevel,
)


@dataclass(frozen=True)
class ToolRuntimeRequest:
    tool_call_id: str
    name: str
    params: dict[str, Any]
    source: str = "agent"


@dataclass(frozen=True)
class ToolRuntimeResult:
    result: AgentToolResult
    status: ToolResultStatus = "success"
    is_error: bool = False
    approved: bool = True
    approval_id: str | None = None


__all__ = [
    "AgentTool",
    "AgentToolResult",
    "ToolMetadata",
    "ToolResultStatus",
    "ToolRiskLevel",
    "ToolRuntimeRequest",
    "ToolRuntimeResult",
]
