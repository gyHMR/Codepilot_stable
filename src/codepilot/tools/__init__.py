"""
Codepilot 工具层。

工具位于本包之下，使 runtime 和 interfaces 共享统一的安全模型，
而非在各入口点分散嵌入文件系统和 shell 检查逻辑。
"""

from .approval import (
    ApprovalDecision,
    ApprovalProvider,
    ApprovalRequest,
    DeferredApprovalProvider,
)
from .builtin import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES, create_builtin_tools
from .permissions import PermissionPolicy, ToolDecision, ToolPermissionMode, ToolRequest
from .registry import ToolRegistry
from .runtime import ToolRuntime
from .sandbox import WorkspaceSandbox
from .types import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    ToolMetadata,
    ToolResultStatus,
    ToolRuntimeRequest,
    ToolRuntimeResult,
)


__all__ = [
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "ApprovalDecision",
    "ApprovalProvider",
    "ApprovalRequest",
    "DeferredApprovalProvider",
    "MUTATING_TOOL_NAMES",
    "PermissionPolicy",
    "READ_ONLY_TOOL_NAMES",
    "ToolDecision",
    "ToolPermissionMode",
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
