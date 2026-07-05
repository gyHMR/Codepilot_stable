# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：tools 层是工具执行安全边界，统一处理契约、权限、校验、审批和结果防护。

"""
Codepilot 工具层。

工具位于本包之下，使 runtime 和 interfaces 共享统一的安全模型，
而非在各入口点分散嵌入文件系统和 shell 检查逻辑。
"""

from .authoring import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    MUTATING_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    ToolMetadata,
    ToolRegistry,
    ToolResultStatus,
)
from .builtins import create_builtin_tools
from .policy import (
    ApprovalDecision,
    ApprovalProvider,
    ApprovalRequest,
    DeferredApprovalProvider,
    PermissionPolicy,
    ToolDecision,
    ToolPermissionMode,
    ToolRequest,
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
    "create_builtin_tools",
]
