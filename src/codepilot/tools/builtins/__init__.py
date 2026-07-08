from __future__ import annotations

"""
内置工具工厂模块。

本模块是所有内置工具的创建入口，负责按类别创建并合并所有内置工具。
内置工具包括五类：文件操作、搜索、Shell执行、工作区状态、计划更新。

每种工具都通过对应的 create_*_tools() 函数创建，最终在 create_builtin_tools()
中按配置过滤后合并返回。工具名称与 metadata 定义在 registry.py 的
_BUILTIN_METADATA 字典中统一管理。
"""

from pathlib import Path

from codepilot.tools.contracts import ToolDefinition
from codepilot.tools.registry import (
    MUTATING_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    get_builtin_tool_metadata,
)
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
    """
    创建所有内置工具的工厂函数。

    从五类工具各自的工厂函数创建工具列表，按 enabled_names 过滤。

    参数:
        workspace_dir: 工作区根目录路径，用于沙箱路径校验。
        enabled_names: 启用的工具名称白名单，为 None 时全部启用。
        edit_require_unique_match: edit 工具是否要求 old_text 唯一匹配（默认 True）。
        shell_policy: Shell 执行策略（超时、输出限制等），为 None 时使用默认值。

    返回:
        过滤后的 ToolDefinition 列表，已包含完整的 metadata 和 execute 函数。

    实现细节:
        - 每种工具的定义（名称、参数 schema、metadata）在各自的工厂函数中
        - 工具的元数据（read_only、risk_level、scopes 等）来自 registry._BUILTIN_METADATA
        - allow() 闭包用于按名称白名单过滤
    """
    # 创建 WorkspaceSandbox 用于工具内部的路径校验
    sandbox = WorkspaceSandbox(Path(workspace_dir))
    # 如果未指定白名单则全部允许
    enabled = set(enabled_names) if enabled_names else None

    def allow(name: str) -> bool:
        """检查工具名称是否在白名单内，None 表示全部允许。"""
        return enabled is None or name in enabled

    tools: list[ToolDefinition] = []

    # 依次创建五类工具：
    # 1. 文件操作工具: ls, read, write, edit, apply_patch
    tools.extend(
        create_file_tools(
            sandbox,
            allow=allow,
            edit_require_unique_match=edit_require_unique_match,
        )
    )
    # 2. 搜索工具: grep, find
    tools.extend(create_search_tools(sandbox, allow=allow))
    # 3. 工作区状态工具: workspace_status
    tools.extend(create_workspace_tools(sandbox, allow=allow))
    # 4. Shell 执行工具: bash
    tools.extend(create_shell_tools(sandbox, allow=allow, policy=shell_policy))
    # 5. 计划管理工具: update_plan
    tools.extend(create_plan_tools(allow=allow))

    return tools


__all__ = [
    "MUTATING_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "create_builtin_tools",
    "create_plan_tools",
    "get_builtin_tool_metadata",
]
