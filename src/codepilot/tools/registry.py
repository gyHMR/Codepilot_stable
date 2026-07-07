from __future__ import annotations

"""Tool registration and metadata inference."""

from dataclasses import dataclass, field
from typing import Iterable

from codepilot.protocols.tools import (
    TASK_CONTROL_COMPLETE_TOOL,
    TASK_CONTROL_UPDATE_TOOL,
    ToolMetadata,
    ToolRiskLevel,
)

from .authoring import AgentTool


READ_ONLY_TOOL_NAMES = {
    "ls",
    "read",
    "grep",
    "find",
    "workspace_status",
    TASK_CONTROL_COMPLETE_TOOL,
    TASK_CONTROL_UPDATE_TOOL,
}
MUTATING_TOOL_NAMES = {"write", "edit", "bash"}


@dataclass
class ToolRegistry:
    """Small in-memory registry used while a runtime session is open."""

    _tools: dict[str, AgentTool] = field(default_factory=dict)
    _metadata: dict[str, ToolMetadata] = field(default_factory=dict)

    def register(
        self,
        tool: AgentTool,
        *,
        metadata: ToolMetadata | None = None,
        replace: bool = True,
    ) -> None:
        if not isinstance(tool, AgentTool):
            raise TypeError("ToolRegistry.register expects an AgentTool")
        if metadata is not None:
            _validate_metadata(tool, metadata)
        if not replace and tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self._metadata[tool.name] = metadata or tool.metadata or infer_tool_metadata(tool)

    def extend(self, tools: Iterable[AgentTool], *, replace: bool = True) -> None:
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


def get_builtin_tool_metadata(name: str) -> ToolMetadata | None:
    return _BUILTIN_TOOL_METADATA.get(name)


def infer_tool_metadata(tool: AgentTool) -> ToolMetadata:
    """Infer conservative metadata for caller, extension, skill, and MCP tools."""

    category = _infer_category(tool.name)
    read_only = tool.name in READ_ONLY_TOOL_NAMES
    mutating = tool.name in MUTATING_TOOL_NAMES
    external = category in {"extension", "mcp"}
    if external:
        return ToolMetadata(
            name=tool.name,
            category=category,
            read_only=False,
            concurrency_safe=False,
            exclusive=True,
            requires_approval=True,
            risk_level=_infer_risk(tool.name, mutating=False),
            resource_scope=(category,),
            network_access=True,
            credential_required=False,
            extra={"metadata_inferred": True},
        )
    return ToolMetadata(
        name=tool.name,
        category=category,
        read_only=read_only,
        concurrency_safe=read_only,
        exclusive=not read_only,
        requires_approval=False,
        risk_level=_infer_risk(tool.name, mutating),
        resource_scope=(category,),
        network_access=False,
        credential_required=False,
    )


def _validate_metadata(tool: AgentTool, metadata: ToolMetadata) -> None:
    if not isinstance(metadata, ToolMetadata):
        raise TypeError("ToolRegistry metadata must be ToolMetadata")
    if metadata.name != tool.name:
        raise ValueError(
            f"Tool metadata name must match tool name: {metadata.name} != {tool.name}"
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
        extra={"capabilities": [_capability(category, read_only=read_only)]},
    )


def _capability(category: str, *, read_only: bool) -> str:
    if category in {"filesystem", "search"}:
        return "filesystem.read" if read_only else "filesystem.write"
    if category == "shell":
        return "process.execute"
    return "workspace.read"


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
    if name in {"ls", "read", "write", "edit"}:
        return "filesystem"
    if name in {"grep", "find"}:
        return "search"
    if name == "bash":
        return "shell"
    if name == "workspace_status":
        return "workspace"
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


__all__ = [
    "MUTATING_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "ToolRegistry",
    "get_builtin_tool_metadata",
    "infer_tool_metadata",
]
