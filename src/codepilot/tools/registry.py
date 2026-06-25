from __future__ import annotations

"""
工具注册表模块。

ToolRegistry 是工具的中心化存储，由 runtime 装配和插件加载使用。
它维护工具名称到 AgentTool 实例和 ToolMetadata 的映射。

使用方式：
    registry = ToolRegistry()
    registry.register(my_tool, metadata=my_metadata)
    tool = registry.get("my_tool")
    tools = registry.list()
"""

from dataclasses import dataclass, field

from .types import AgentTool, ToolMetadata


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
        self._metadata[tool.name] = metadata or infer_tool_metadata(tool)

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


def infer_tool_metadata(tool: AgentTool) -> ToolMetadata:
    """从工具定义推断元数据（用于非内置工具）。"""
    name = tool.name
    read_only = name in {"read", "grep", "find", "ls", "workspace_status"}
    mutating = name in {"write", "edit", "bash"}
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


def _infer_category(name: str) -> str:
    if name in {"read", "write", "edit", "ls", "workspace_status"}:
        return "filesystem"
    if name in {"grep", "find"}:
        return "search"
    if name == "bash":
        return "shell"
    if name.startswith("mcp_"):
        return "mcp"
    return "extension"


def _infer_risk(name: str, mutating: bool) -> str:
    if name.startswith("mcp_"):
        return "medium"
    if _infer_category(name) == "extension":
        return "medium"
    if name == "bash":
        return "medium"
    return "medium" if mutating else "low"
