from __future__ import annotations

# 新手导读：ToolRegistry 是工具表，只保存工具实例和对应 metadata。
# 关注点：它故意不做执行决策，避免注册表变成“大杂烩”。

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

from .contracts import AgentTool, ToolMetadata
from .metadata import infer_tool_metadata


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
