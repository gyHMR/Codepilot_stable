"""
Tooling boundary for Codepilot.

Tools live below this package so runtime and interfaces can share one security
model instead of embedding filesystem and shell checks in entry points.
"""

from .builtin import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES, create_builtin_tools
from .permissions import PermissionPolicy, ToolDecision, ToolRequest
from .sandbox import WorkspaceSandbox
from .types import AgentTool, AgentToolResult

__all__ = [
    "AgentTool",
    "AgentToolResult",
    "MUTATING_TOOL_NAMES",
    "PermissionPolicy",
    "READ_ONLY_TOOL_NAMES",
    "ToolDecision",
    "ToolRequest",
    "WorkspaceSandbox",
    "create_builtin_tools",
]
