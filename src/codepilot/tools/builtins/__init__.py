from __future__ import annotations

"""Built-in tool factory."""

from pathlib import Path

from codepilot.tools.contracts import ToolDefinition
from codepilot.tools.registry import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES, get_builtin_tool_metadata
from codepilot.tools.sandbox import ShellExecutionPolicy, WorkspaceSandbox

from .files import create_file_tools
from .plan import create_plan_tools
from .search import create_search_tools
from .shell import create_shell_tools
from .workspace import create_workspace_tools


def create_builtin_tools(
    workspace_dir: str | Path,
    enabled_names: list[str] | None = None,
    *,
    edit_require_unique_match: bool = True,
    shell_policy: ShellExecutionPolicy | None = None,
) -> list[ToolDefinition]:
    sandbox = WorkspaceSandbox(Path(workspace_dir))
    enabled = set(enabled_names) if enabled_names else None

    def allow(name: str) -> bool:
        return enabled is None or name in enabled

    tools: list[ToolDefinition] = []
    tools.extend(
        create_file_tools(
            sandbox,
            allow=allow,
            edit_require_unique_match=edit_require_unique_match,
        )
    )
    tools.extend(create_search_tools(sandbox, allow=allow))
    tools.extend(create_workspace_tools(sandbox, allow=allow))
    tools.extend(create_shell_tools(sandbox, allow=allow, policy=shell_policy))
    tools.extend(create_plan_tools(allow=allow))
    return tools


__all__ = [
    "MUTATING_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "create_builtin_tools",
    "create_plan_tools",
    "get_builtin_tool_metadata",
]
