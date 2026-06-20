from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from codepilot.protocols.tools import (
    Tool,
    ToolMetadata,
    ToolResult,
    ToolResultStatus,
    ToolRiskLevel,
)


# 工具执行结果类型（复用 protocols 中的 ToolResult）
AgentToolResult = ToolResult
# 工具执行过程中的增量更新回调
AgentToolUpdateCallback = Callable[[AgentToolResult], None]


class ToolExecuteFn(Protocol):
    """工具执行函数协议：接收调用ID、参数、信号和更新回调，返回结果。"""
    def __call__(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> Awaitable[AgentToolResult] | AgentToolResult:
        ...


@dataclass
class AgentTool:
    """工具层拥有的可执行工具定义。"""
    name: str                        # 工具名称
    label: str                       # 人类可读标签
    description: str                 # 工具描述
    parameters: dict[str, Any]       # JSON Schema 参数定义
    execute: ToolExecuteFn           # 执行函数
    runtime_managed: bool = False    # 是否由 ToolRuntime 管理
    metadata: ToolMetadata | None = None  # 工具元数据

    def to_spec(self) -> Tool:
        """返回面向 LLM provider 的工具描述（不含执行器）。"""

        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass(frozen=True)
class ToolRuntimeRequest:
    """工具运行时请求：封装一次工具调用的完整信息。"""
    tool_call_id: str                # 调用唯一标识
    name: str                        # 工具名称
    params: dict[str, Any]           # 调用参数
    source: str = "agent"            # 调用来源


@dataclass(frozen=True)
class ToolRuntimeResult:
    """工具运行时结果：封装执行结果和权限审批状态。"""
    result: AgentToolResult
    status: ToolResultStatus = "success"
    is_error: bool = False
    approved: bool = True
    approval_id: str | None = None


__all__ = [
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "ToolExecuteFn",
    "ToolMetadata",
    "ToolResultStatus",
    "ToolRiskLevel",
    "ToolRuntimeRequest",
    "ToolRuntimeResult",
]
