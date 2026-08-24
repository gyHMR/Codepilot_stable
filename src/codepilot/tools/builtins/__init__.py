"""规范的内置注册组合器 —— 收集所有内置工具并向调用方提供统一的创建接口。

本文件是 builtins 包的入口，组合了 files、search、shell、workspace
四个模块的工具创建函数，通过 create_builtin_registrations() 提供一站式
的内置工具注册创建。

内置工具清单：
1. 文件操作工具（files.py）—— ls / read / write / edit / apply_patch
2. 搜索工具（search.py）—— grep / glob
3. 命令工具（shell.py）—— command（受控命令）/ bash（完整 Shell）
4. 工作区状态工具（workspace.py）—— workspace_status

create_builtin_registrations 的 allow 机制支持选择性启用/禁用工具，
用于权限隔离场景（如 plan 模式下禁用写操作工具）。
"""

from pathlib import Path

from ..contracts import ToolRegistration
from ..sandbox import ShellExecutionPolicy, WorkspaceSandbox
from .files import create_file_registrations
from .search import create_search_registrations
from .shell import create_command_registration, create_shell_registration
from .workspace import create_workspace_status_registration


def create_builtin_registrations(
    workspace_dir: str | Path,
    enabled_names: list[str] | None = None,
    *,
    edit_require_unique_match: bool = True,
    shell_policy: ShellExecutionPolicy | None = None,
) -> list[ToolRegistration]:
    """创建所有启用的内置工具注册列表。

    这是 builtins 包的主入口，收集所有内置工具的注册信息并返回列表。

    处理流程：
    1. 创建工作区沙箱（WorkspaceSandbox）
    2. 根据 enabled_names 构建 allow 过滤器
    3. 依次创建各模块的注册：files → search → workspace_status → command → bash
    4. 返回合并后的注册列表

    参数:
        workspace_dir: 工作区根目录路径
        enabled_names: 要启用的工具名称列表
            - None: 启用所有工具
            - 列表: 只启用列表中的工具
        edit_require_unique_match: edit 工具是否要求唯一匹配（默认 True）
        shell_policy: Shell 执行策略（默认 None 使用 ShellExecutionPolicy 的默认值）

    返回:
        ToolRegistration 列表，按工具类型分组排列
    """
    # 创建沙箱（用于路径解析和边界检查）
    sandbox = WorkspaceSandbox(Path(workspace_dir))
    # 构建 allow 过滤器
    enabled = set(enabled_names) if enabled_names else None

    def allow(name: str) -> bool:
        """allow 过滤器 —— 根据 enabled_names 决定是否启用某工具。

        参数:
            name: 工具名称

        返回:
            True 表示允许注册此工具
        """
        return enabled is None or name in enabled

    registrations: list[ToolRegistration] = []
    # 文件操作工具（ls / read / write / edit / apply_patch）
    registrations.extend(
        create_file_registrations(
            sandbox,
            allow=allow,
            edit_require_unique_match=edit_require_unique_match,
        )
    )
    # 搜索工具（grep / glob）
    registrations.extend(create_search_registrations(sandbox, allow=allow))
    # 工作区状态工具
    if allow("workspace_status"):
        registrations.append(create_workspace_status_registration(sandbox))
    # 受控命令工具
    if allow("command"):
        registrations.append(create_command_registration(sandbox, policy=shell_policy))
    # Shell 命令工具
    if allow("bash"):
        registrations.append(create_shell_registration(sandbox, policy=shell_policy))
    return registrations


__all__ = [
    "create_builtin_registrations",
    "create_file_registrations",
    "create_search_registrations",
    "create_command_registration",
    "create_shell_registration",
    "create_workspace_status_registration",
]