from __future__ import annotations

# 新手导读：工具层契约文件：定义可执行 AgentTool、运行时请求和运行时结果。
# 关注点：protocols 只描述跨层数据，这里才包含 execute 函数等工具层内部概念。

"""
工具层类型定义模块。

定义了工具层拥有的核心类型，与 protocols/tools.py 的区别：
- protocols/tools.py: 定义跨层共享的稳定协议（Tool、ToolResult 等）
- tools/contracts.py: 定义工具层内部的可执行类型（AgentTool、ToolRuntimeRequest 等）

主要类型：
    - AgentTool: 可执行的工具定义（包含 execute 函数）
    - AgentToolResult: 工具执行结果（复用 ToolResult）
    - ToolRuntimeRequest: 工具运行时请求
    - ToolRuntimeResult: 工具运行时结果（包含权限审批状态）
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from codepilot.protocols.tools import (
    TASK_CONTROL_UPDATE_TOOL,
    Tool,
    ToolMetadata,
    ToolResult,
    ToolResultStatus,
    ToolRiskLevel,
    ensure_tool_result_status,
    TASK_CONTROL_COMPLETE_TOOL,
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



# ── Tool metadata and registry ─────────────────────────────────────

READ_ONLY_TOOL_NAMES = {
    "read",
    "grep",
    "find",
    "ls",
    "workspace_status",
    TASK_CONTROL_COMPLETE_TOOL,
    TASK_CONTROL_UPDATE_TOOL,
}
MUTATING_TOOL_NAMES = {"write", "edit", "bash"}


def get_builtin_tool_metadata(name: str) -> ToolMetadata | None:
    """Return static metadata for a built-in tool name."""
    return _BUILTIN_TOOL_METADATA.get(name)


def infer_tool_metadata(tool: AgentTool) -> ToolMetadata:
    """Infer conservative metadata for caller, extension, and MCP tools."""
    name = tool.name
    read_only = name in READ_ONLY_TOOL_NAMES
    mutating = name in MUTATING_TOOL_NAMES
    category = _infer_category(name)
    external = category in {"extension", "mcp"}
    if external:
        return ToolMetadata(
            name=name,
            category=category,
            read_only=False,
            concurrency_safe=False,
            exclusive=True,
            requires_approval=True,
            risk_level=_infer_risk(name, mutating=False),
            resource_scope=(category,),
            network_access=True,
            credential_required=False,
            extra={"metadata_inferred": True},
        )
    return ToolMetadata(
        name=name,
        category=category,
        read_only=read_only,
        concurrency_safe=read_only,
        exclusive=not read_only,
        requires_approval=external,
        risk_level=_infer_risk(name, mutating),
        resource_scope=(category,),
        network_access=False,
        credential_required=False,
    )


def _builtin_metadata(
    name: str,
    *,
    category: str,
    read_only: bool,
    risk_level: ToolRiskLevel,
    resource_scope: tuple[str, ...],
) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        category=category,
        read_only=read_only,
        concurrency_safe=read_only,
        exclusive=not read_only,
        requires_approval=False,
        risk_level=risk_level,
        resource_scope=resource_scope,
        network_access=False,
        credential_required=False,
        extra={
            "capabilities": [
                (
                    "filesystem.read"
                    if read_only and category in {"filesystem", "search"}
                    else "filesystem.write"
                    if category == "filesystem"
                    else "process.execute"
                    if category == "shell"
                    else "workspace.read"
                )
            ]
        },
    )


_BUILTIN_TOOL_METADATA: dict[str, ToolMetadata] = {
    "ls": _builtin_metadata(
        "ls",
        category="filesystem",
        read_only=True,
        risk_level="low",
        resource_scope=("workspace",),
    ),
    "read": _builtin_metadata(
        "read",
        category="filesystem",
        read_only=True,
        risk_level="low",
        resource_scope=("workspace",),
    ),
    "write": _builtin_metadata(
        "write",
        category="filesystem",
        read_only=False,
        risk_level="medium",
        resource_scope=("workspace",),
    ),
    "edit": _builtin_metadata(
        "edit",
        category="filesystem",
        read_only=False,
        risk_level="medium",
        resource_scope=("workspace",),
    ),
    "grep": _builtin_metadata(
        "grep",
        category="search",
        read_only=True,
        risk_level="low",
        resource_scope=("workspace",),
    ),
    "find": _builtin_metadata(
        "find",
        category="search",
        read_only=True,
        risk_level="low",
        resource_scope=("workspace",),
    ),
    "bash": _builtin_metadata(
        "bash",
        category="shell",
        read_only=False,
        risk_level="medium",
        resource_scope=("workspace", "process"),
    ),
    "workspace_status": _builtin_metadata(
        "workspace_status",
        category="workspace",
        read_only=True,
        risk_level="low",
        resource_scope=("workspace", "git"),
    ),
    TASK_CONTROL_COMPLETE_TOOL: _builtin_metadata(
        TASK_CONTROL_COMPLETE_TOOL,
        category="task_control",
        read_only=True,
        risk_level="low",
        resource_scope=("task",),
    ),
    TASK_CONTROL_UPDATE_TOOL: _builtin_metadata(
        TASK_CONTROL_UPDATE_TOOL,
        category="task_control",
        read_only=True,
        risk_level="low",
        resource_scope=("task",),
    ),
}


def _infer_category(name: str) -> str:
    if name in {"read", "write", "edit", "ls", "workspace_status"}:
        return "filesystem"
    if name in {"grep", "find"}:
        return "search"
    if name == "bash":
        return "shell"
    if name in {TASK_CONTROL_COMPLETE_TOOL, TASK_CONTROL_UPDATE_TOOL}:
        return "task_control"
    if name.startswith("mcp_"):
        return "mcp"
    return "extension"


def _infer_risk(name: str, mutating: bool) -> ToolRiskLevel:
    if name.startswith("mcp_"):
        return "medium"
    if _infer_category(name) == "extension":
        return "medium"
    if name == "bash":
        return "medium"
    return "medium" if mutating else "low"

@dataclass
class ToolRegistry:
    """工具注册表：由 runtime 装配和未来插件加载使用。"""
    _tools: dict[str, AgentTool] = field(default_factory=dict)       # 名称 -> 工具
    _metadata: dict[str, ToolMetadata] = field(default_factory=dict) # 名称 -> 元数据

    def register(self, tool: AgentTool, *, metadata: ToolMetadata | None = None, replace: bool = True) -> None:
        if not isinstance(tool, AgentTool):
            raise TypeError("ToolRegistry.register expects an AgentTool")
        if metadata is not None:
            if not isinstance(metadata, ToolMetadata):
                raise TypeError("ToolRegistry metadata must be ToolMetadata")
            if metadata.name != tool.name:
                raise ValueError(
                    f"Tool metadata name must match tool name: {metadata.name} != {tool.name}"
                )
        if not replace and tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self._metadata[tool.name] = metadata or tool.metadata or infer_tool_metadata(tool)

    def extend(self, tools: list[AgentTool], *, replace: bool = True) -> None:
        for tool in tools:
            self.register(tool, replace=replace)

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def metadata_for(self, name: str) -> ToolMetadata | None:
        return self._metadata.get(name)

    def list(self) -> list[AgentTool]:
        return list(self._tools.values())

    def list_metadata(self) -> list[ToolMetadata]:
        return list(self._metadata.values())


__all__ = [
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "MUTATING_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "ToolExecuteFn",
    "ToolMetadata",
    "ToolRegistry",
    "ToolResultStatus",
    "ToolRiskLevel",
    "ToolRuntimeRequest",
    "ToolRuntimeResult",
    "get_builtin_tool_metadata",
    "infer_tool_metadata",
]
