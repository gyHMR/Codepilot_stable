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
from .builtins import create_builtin_tools
from .metadata import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES
from .policy import PermissionPolicy, ToolDecision, ToolPermissionMode, ToolRequest
from .registry import ToolRegistry
from .result_safety import ToolResultGuard, apply_result_guard
from .execution import ToolRuntime
from .workspace_safety import WorkspaceSandbox
from .argument_schema import SchemaValidationResult, SchemaValidator, validate_tool_arguments
from .contracts import (
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
    "SchemaValidationResult",
    "SchemaValidator",
    "ToolResultGuard",
    "apply_result_guard",
    "create_builtin_tools",
    "validate_tool_arguments",
]
