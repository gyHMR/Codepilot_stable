from __future__ import annotations

"""
Codepilot 工具层（Tools Layer）公共接口。

本模块是工具层的统一入口，导出所有核心类型和函数。
工具层负责：
  - 工具定义与注册（ToolDefinition、ToolRegistry）
  - 权限决策与安全策略（PermissionPolicy）
  - 审批流程管理（ApprovalProvider）
  - 统一执行管线（ToolRuntime）
  - 结果规范化与脱敏（ToolResultPolicy）

层级位置：protocols → tools → core → sessions → runtime → interfaces
工具层只依赖 protocols 层，不依赖 core/sessions/runtime。
"""

from .approvals import (
    ApprovalDecision,
    ApprovalProvider,
    ApprovalRequest,
    DeferredApprovalProvider,
)
from .builtins import create_builtin_tools
from .contracts import (
    PreparedToolCall,
    PreparedToolCallResult,
    ToolCallRequest,
    ToolCatalogItem,
    ToolCatalogView,
    ToolDefinition,
    ToolInterruption,
    ToolInvocation,
    ToolMetadata,
    ToolObservation,
    ToolPolicyContext,
    ToolPort,
    ToolResult,
    ToolResumeDecision,
    ToolRiskView,
)
from .permissions import PermissionPolicy, ToolDecision, ToolPermissionMode
from .registry import ToolRegistry, get_builtin_tool_metadata
from .runtime import ToolRuntime

__all__ = [
    "ApprovalDecision",
    "ApprovalProvider",
    "ApprovalRequest",
    "DeferredApprovalProvider",
    "PermissionPolicy",
    "PreparedToolCall",
    "PreparedToolCallResult",
    "ToolCallRequest",
    "ToolCatalogItem",
    "ToolCatalogView",
    "ToolDecision",
    "ToolDefinition",
    "ToolInterruption",
    "ToolInvocation",
    "ToolMetadata",
    "ToolObservation",
    "ToolPermissionMode",
    "ToolPolicyContext",
    "ToolPort",
    "ToolRegistry",
    "ToolResult",
    "ToolResumeDecision",
    "ToolRiskView",
    "ToolRuntime",
    "create_builtin_tools",
    "get_builtin_tool_metadata",
]
