from __future__ import annotations

# 新手导读：工具 metadata 描述工具风险、只读/写入能力、并发安全和默认推断。
# 关注点：registry 只负责登记，工具语义判断集中在这里和 policy.py。

"""Tool metadata catalog and conservative metadata inference."""

from .contracts import AgentTool, ToolMetadata, ToolRiskLevel


READ_ONLY_TOOL_NAMES = {"read", "grep", "find", "ls", "workspace_status"}
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
}


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
    "get_builtin_tool_metadata",
    "infer_tool_metadata",
]
