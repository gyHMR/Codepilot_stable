from __future__ import annotations

# 新手导读：包门面文件：集中导出本层最常用的类型和入口，降低学习时的导入成本。
# 关注点：tools 层是工具执行安全边界，统一处理契约、权限、校验、审批和结果防护。

"""内置工具包：提供文件操作、搜索、shell 和工作区状态等基础工具。"""

from pathlib import Path

from codepilot.tools.contracts import AgentTool
from codepilot.tools.metadata import (
    MUTATING_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    get_builtin_tool_metadata,
)
from codepilot.tools.workspace_safety import WorkspaceSandbox
from codepilot.tools.shell_safety import ShellExecutionPolicy

from .files import create_file_tools
from .search import create_search_tools
from .shell import create_shell_tools
from .workspace_status import create_workspace_tools


def create_builtin_tools(
    workspace_dir: str | Path,
    enabled_names: list[str] | None = None,
    *,
    edit_require_unique_match: bool = True,
    shell_policy: ShellExecutionPolicy | None = None,
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
    tools.extend(create_workspace_tools(sandbox, allow=allow))
    tools.extend(create_shell_tools(sandbox, allow=allow, policy=shell_policy))
    return tools


__all__ = [
    "MUTATING_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "create_builtin_tools",
    "get_builtin_tool_metadata",
]
