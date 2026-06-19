from __future__ import annotations

"""
工具组装模块。

负责将来自不同来源的工具合并为统一的工具列表：
1) 内置工具（ls、find、read、grep、edit、write、bash 等）
2) 调用方通过 options.tools 传入的自定义工具
3) 扩展加载的工具（通过 extension_paths）
4) MCP 代理工具（通过 mcp_servers 配置）

组装流程：
- 按名称去重（后者覆盖前者）
- 注册到 ToolRegistry 并附加元数据
- 根据 read_only_mode 过滤只保留只读工具
- 应用权限策略（PermissionPolicy）
- 输出最终的 AgentTool 列表
"""

from dataclasses import dataclass
from pathlib import Path

from codepilot.extensions import load_extensions, load_skills
from codepilot.extensions.mcp import create_mcp_proxy_tools, parse_mcp_tool_configs
from codepilot.extensions.types import LoadedExtensions
from codepilot.tools.builtin import (
    create_builtin_tools,
    get_builtin_tool_metadata,
)
from codepilot.tools.permissions import PermissionPolicy
from codepilot.tools.registry import ToolRegistry
from codepilot.tools.runtime import ToolRuntime
from codepilot.tools.types import AgentTool

from .config import RuntimeConfig
from .types import CreateAgentSessionOptions


@dataclass(frozen=True)
class AssembledTools:
    """工具组装结果。

    Attributes:
        tools: 最终的 AgentTool 列表（已注册到 ToolRuntime 并应用权限策略）。
        loaded_extensions: 已加载的扩展信息（包含钩子、命令等）。
        loaded_skills: 已加载的技能信息（包含钩子、命令等）。
    """

    tools: list[AgentTool]
    loaded_extensions: LoadedExtensions
    loaded_skills: LoadedExtensions


def assemble_tools(
    workspace: Path,
    options: CreateAgentSessionOptions,
    config: RuntimeConfig,
) -> AssembledTools:
    """组装所有来源的工具并应用权限策略。

    流程：
    1. 加载扩展和技能（获取工具、钩子、命令等）。
    2. 创建 MCP 代理工具。
    3. 创建内置工具。
    4. 按优先级合并所有工具（后者覆盖同名前者）。
    5. 注册到 ToolRegistry 并附加元数据。
    6. 如果是只读模式，过滤掉非只读工具。
    7. 创建 ToolRuntime 并应用权限策略。

    Args:
        workspace: 工作区目录路径。
        options: 创建会话的配置选项。
        config: 已解析的运行时配置。

    Returns:
        AssembledTools 对象，包含最终工具列表和加载的扩展/技能信息。
    """
    # 加载扩展和技能
    loaded_extensions = load_extensions(workspace, configured_paths=config.extension_paths)
    loaded_skills = load_skills(workspace, configured_paths=config.skill_paths)
    # 创建 MCP 代理工具
    mcp_tools = create_mcp_proxy_tools(
        parse_mcp_tool_configs(config.mcp_servers),
        client=options.mcp_client,
    )

    # 创建内置工具
    builtin_tools = create_builtin_tools(
        workspace,
        enabled_names=config.enabled_builtin_tools,
        edit_require_unique_match=config.edit_require_unique_match,
    )

    # 按名称去重合并：内置 -> 自定义 -> 扩展 -> MCP（后者覆盖前者）
    tool_map = {tool.name: tool for tool in builtin_tools}
    for tool in options.tools:
        tool_map[tool.name] = tool
    for tool in loaded_extensions.tools:
        tool_map[tool.name] = tool
    for tool in mcp_tools:
        tool_map[tool.name] = tool

    # 注册到 ToolRegistry 并附加元数据
    registry = ToolRegistry()
    for tool in tool_map.values():
        metadata = get_builtin_tool_metadata(tool.name)
        registry.register(tool, metadata=metadata)

    # 只读模式：过滤掉非只读工具
    if config.read_only_mode:
        registry = _read_only_registry(registry)

    # 创建 ToolRuntime 并应用权限策略
    runtime = ToolRuntime(
        registry=registry,
        permission_policy=PermissionPolicy(
            read_only=config.read_only_mode,
            block_dangerous_bash=config.block_dangerous_bash,
            bash_allow_patterns=config.bash_allow_patterns,
            bash_block_patterns=config.bash_block_patterns,
        ),
    )

    return AssembledTools(
        tools=runtime.as_agent_tools(),
        loaded_extensions=loaded_extensions,
        loaded_skills=loaded_skills,
    )


def _read_only_registry(registry: ToolRegistry) -> ToolRegistry:
    """过滤工具注册表，只保留标记为 read_only 的工具。"""
    filtered = ToolRegistry()
    for tool in registry.list():
        metadata = registry.metadata_for(tool.name)
        if metadata is not None and metadata.read_only:
            filtered.register(tool, metadata=metadata)
    return filtered
