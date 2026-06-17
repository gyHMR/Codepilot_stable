from __future__ import annotations

from dataclasses import dataclass, field

from .types import AgentTool, ToolMetadata


@dataclass
class ToolRegistry:
    """Small registry used by runtime assembly and future plugin loading."""

    _tools: dict[str, AgentTool] = field(default_factory=dict)
    _metadata: dict[str, ToolMetadata] = field(default_factory=dict)

    def register(self, tool: AgentTool, *, metadata: ToolMetadata | None = None, replace: bool = True) -> None:
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
    name = tool.name
    read_only = name in {"read", "read_file", "grep", "find", "ls", "list_dir"}
    mutating = name in {"write", "write_file", "edit", "bash"}
    category = _infer_category(name)
    external = category in {"extension", "mcp"}
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
    if name in {"read", "read_file", "write", "write_file", "edit", "ls", "list_dir"}:
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
