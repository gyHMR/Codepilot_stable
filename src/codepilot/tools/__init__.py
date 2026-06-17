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
from .approval import ApprovalDecision, ApprovalProvider, DenyApprovalProvider
from .base import AgentTool, AgentToolResult, ToolMetadata
from .diff import DiffRecorder, FileDiff
from .permissions import PermissionPolicy, ToolDecision, ToolRequest
from .registry import ToolRegistry
from .runtime import ToolRuntime
from .types import ToolRuntimeRequest, ToolRuntimeResult

__all__ = [
    "AgentTool",
    "AgentToolResult",
    "ApprovalDecision",
    "ApprovalProvider",
    "DenyApprovalProvider",
    "DiffRecorder",
    "FileDiff",
    "PermissionPolicy",
    "ToolDecision",
    "ToolMetadata",
    "ToolRegistry",
    "ToolRequest",
    "ToolRuntime",
    "ToolRuntimeRequest",
    "ToolRuntimeResult",
]
