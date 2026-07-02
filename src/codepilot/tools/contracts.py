from __future__ import annotations

"""
工具层类型定义模块。

定义了工具层拥有的核心类型，与 protocols/tools.py 的区别：
- protocols/tools.py: 定义跨层共享的稳定协议（Tool、ToolResult 等）
- tools/types.py: 定义工具层内部的可执行类型（AgentTool、ToolRuntimeRequest 等）

主要类型：
    - AgentTool: 可执行的工具定义（包含 execute 函数）
    - AgentToolResult: 工具执行结果（复用 ToolResult）
    - ToolRuntimeRequest: 工具运行时请求
    - ToolRuntimeResult: 工具运行时结果（包含权限审批状态）
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from codepilot.protocols.tools import (
    Tool,
    ToolMetadata,
    ToolResult,
    ToolResultStatus,
    ToolRiskLevel,
    ensure_tool_result_status,
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

    def __post_init__(self) -> None:
        self.name = _require_tool_definition_text(self.name, field_name="tool name")
        self.label = _require_tool_definition_text(self.label, field_name="label")
        self.description = _require_tool_definition_text(
            self.description,
            field_name="description",
        )
        if not isinstance(self.parameters, dict):
            raise TypeError("AgentTool parameters must be a dict")
        self.parameters = deepcopy(self.parameters)
        if not callable(self.execute):
            raise TypeError("AgentTool execute must be callable")
        if not isinstance(self.runtime_managed, bool):
            raise TypeError("AgentTool runtime_managed must be bool")

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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_call_id",
            _require_tool_runtime_text(self.tool_call_id, field_name="tool_call_id"),
        )
        object.__setattr__(
            self,
            "name",
            _require_tool_runtime_text(self.name, field_name="tool name"),
        )
        if not isinstance(self.params, dict):
            raise TypeError("ToolRuntimeRequest params must be a dict")
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(
            self,
            "source",
            _require_tool_runtime_text(self.source, field_name="source"),
        )


@dataclass(frozen=True)
class ToolRuntimeResult:
    """工具运行时结果：封装执行结果和权限审批状态。"""
    result: AgentToolResult
    status: ToolResultStatus = "success"
    is_error: bool = False
    approved: bool = True
    approval_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, ToolResult):
            raise TypeError("ToolRuntimeResult result must be AgentToolResult")
        status = ensure_tool_result_status(self.status)
        is_error = bool(self.is_error)
        if is_error and status == "success":
            status = "error"
        elif status != "success":
            is_error = True
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "is_error", is_error)
        if not isinstance(self.approved, bool):
            raise TypeError("ToolRuntimeResult approved must be bool")
        if self.approval_id is not None:
            object.__setattr__(
                self,
                "approval_id",
                _require_tool_runtime_text(self.approval_id, field_name="approval_id"),
            )


def _clean_tool_runtime_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_tool_definition_text(value: object, *, field_name: str) -> str:
    text = _clean_tool_runtime_text(value)
    if not text:
        raise ValueError(f"AgentTool {field_name} cannot be empty")
    return text


def _require_tool_runtime_text(value: object, *, field_name: str) -> str:
    text = _clean_tool_runtime_text(value)
    if not text:
        raise ValueError(f"Tool runtime {field_name} cannot be empty")
    return text


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
