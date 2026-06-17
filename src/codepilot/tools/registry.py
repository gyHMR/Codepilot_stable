from __future__ import annotations

from dataclasses import dataclass, field

from .types import AgentTool


@dataclass
class ToolRegistry:
    """Small registry used by runtime assembly and future plugin loading."""

    _tools: dict[str, AgentTool] = field(default_factory=dict)

    def register(self, tool: AgentTool, *, replace: bool = True) -> None:
        if not replace and tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def extend(self, tools: list[AgentTool], *, replace: bool = True) -> None:
        for tool in tools:
            self.register(tool, replace=replace)

    def list(self) -> list[AgentTool]:
        return list(self._tools.values())
