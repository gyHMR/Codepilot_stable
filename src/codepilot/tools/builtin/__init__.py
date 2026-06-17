from __future__ import annotations

from pathlib import Path

from codepilot.core import AgentTool

from codepilot.tools.permissions import (
    MUTATING_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    PermissionPolicy,
)
from codepilot.tools.sandbox import WorkspaceSandbox

from .file_tools import create_file_tools
from .search_tools import create_search_tools
from .shell_tools import create_shell_tools


def create_builtin_tools(
    workspace_dir: str | Path,
    enabled_names: list[str] | None = None,
    *,
    block_dangerous_bash: bool = True,
    bash_allow_patterns: list[str] | None = None,
    bash_block_patterns: list[str] | None = None,
    edit_require_unique_match: bool = True,
    permission_policy: PermissionPolicy | None = None,
) -> list[AgentTool]:
    workspace = Path(workspace_dir)
    sandbox = WorkspaceSandbox(workspace)
    policy = permission_policy or PermissionPolicy(
        block_dangerous_bash=block_dangerous_bash,
        bash_allow_patterns=bash_allow_patterns,
        bash_block_patterns=bash_block_patterns,
    )
    enabled = set(enabled_names) if enabled_names else None

    def allow(name: str) -> bool:
        return enabled is None or name in enabled

    tools: list[AgentTool] = []
    tools.extend(create_file_tools(sandbox, policy=policy, allow=allow, edit_require_unique_match=edit_require_unique_match))
    tools.extend(create_search_tools(sandbox, allow=allow))
    tools.extend(create_shell_tools(sandbox, policy=policy, allow=allow))
    return tools


__all__ = [
    "MUTATING_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "create_builtin_tools",
]
