"""
Tooling boundary for Codepilot.

Tools live below this package so runtime and interfaces can share one security
model instead of embedding filesystem and shell checks in entry points.
"""

from .approval import ApprovalDecision, ApprovalProvider, DenyApprovalProvider
from .builtin import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES, create_builtin_tools
from .permissions import PermissionPolicy, ToolDecision, ToolRequest
from .registry import ToolRegistry
from .runtime import ToolRuntime
from .sandbox import WorkspaceSandbox
from .types import (
    AgentTool,
    AgentToolResult,
    ToolMetadata,
    ToolResultStatus,
    ToolRuntimeRequest,
    ToolRuntimeResult,
)


__all__ = [
    "AgentTool",
    "AgentToolResult",
    "ApprovalDecision",
    "ApprovalProvider",
    "DenyApprovalProvider",
    "MUTATING_TOOL_NAMES",
    "PermissionPolicy",
    "READ_ONLY_TOOL_NAMES",
    "ToolDecision",
    "ToolMetadata",
    "ToolRegistry",
    "ToolRequest",
    "ToolResultStatus",
    "ToolRuntime",
    "ToolRuntimeRequest",
    "ToolRuntimeResult",
    "WorkspaceSandbox",
    "create_builtin_tools",
]
