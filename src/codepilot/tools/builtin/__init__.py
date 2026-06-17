from __future__ import annotations

from pathlib import Path

from codepilot.core import AgentTool

from codepilot.tools.permissions import (
    MUTATING_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
)
from codepilot.tools.sandbox import WorkspaceSandbox
from codepilot.tools.types import ToolMetadata, ToolRiskLevel

from .file_tools import create_file_tools
from .search_tools import create_search_tools
from .shell_tools import create_shell_tools


def create_builtin_tools(
    workspace_dir: str | Path,
    enabled_names: list[str] | None = None,
    *,
    edit_require_unique_match: bool = True,
) -> list[AgentTool]:
    workspace = Path(workspace_dir)
    sandbox = WorkspaceSandbox(workspace)
    enabled = set(enabled_names) if enabled_names else None

    def allow(name: str) -> bool:
        return enabled is None or name in enabled

    tools: list[AgentTool] = []
    tools.extend(
        create_file_tools(
            sandbox,
            allow=allow,
            edit_require_unique_match=edit_require_unique_match,
        )
    )
    tools.extend(create_search_tools(sandbox, allow=allow))
    tools.extend(create_shell_tools(sandbox, allow=allow))
    return tools


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
    )


_BUILTIN_TOOL_METADATA: dict[str, ToolMetadata] = {
    "ls": _builtin_metadata(
        "ls",
        category="filesystem",
        read_only=True,
        risk_level="low",
        resource_scope=("workspace",),
    ),
    "list_dir": _builtin_metadata(
        "list_dir",
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
    "read_file": _builtin_metadata(
        "read_file",
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
    "write_file": _builtin_metadata(
        "write_file",
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
}


def get_builtin_tool_metadata(name: str) -> ToolMetadata | None:
    return _BUILTIN_TOOL_METADATA.get(name)


__all__ = [
    "MUTATING_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "create_builtin_tools",
    "get_builtin_tool_metadata",
]
