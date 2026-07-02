from __future__ import annotations

"""
工具组装模块。

负责将来自不同来源的工具合并为统一的工具列表：
1) 内置工具（ls、find、read、grep、edit、write、bash 等）
2) 调用方通过 options.tools 传入的自定义工具
3) 扩展加载的工具（通过 extension_paths）
4) MCP 代理工具（通过 mcp_servers 配置）

阶段C改进：
- 为工具记录来源和 metadata
- 增加单工具校验
- 增加冲突诊断
- Skill、MCP、Extension 错误统一进入 diagnostics
"""

from dataclasses import dataclass, field
from pathlib import Path

from codepilot.extensions import load_extensions, load_skills
from codepilot.extensions.mcp import create_mcp_proxy_tools, parse_mcp_tool_configs
from codepilot.extensions.types import LoadedExtensions
from codepilot.tools.builtins import (
    create_builtin_tools,
    get_builtin_tool_metadata,
)
from codepilot.tools.policy import PermissionPolicy
from codepilot.tools.approval import DeferredApprovalProvider
from codepilot.tools.metadata import infer_tool_metadata
from codepilot.tools.registry import ToolRegistry
from codepilot.tools.execution import ToolRuntime
from codepilot.tools.shell_safety import ShellExecutionPolicy
from codepilot.tools.contracts import AgentTool

from .config import RuntimeConfig
from .types import CreateAgentSessionOptions, RegisteredTool, RuntimeDiagnostic


@dataclass(frozen=True)
class AssembledTools:
    """工具组装结果。

    Attributes:
        tools: 最终的 AgentTool 列表（已注册到 ToolRuntime 并应用权限策略）。
        registered_tools: 已注册的工具详细信息列表。
        loaded_extensions: 已加载的扩展信息（包含钩子、命令等）。
        loaded_skills: 已加载的技能信息（包含钩子、命令等）。
        diagnostics: 装配诊断列表。
    """
    tools: list[AgentTool]
    registered_tools: list[RegisteredTool]
    loaded_extensions: LoadedExtensions
    loaded_skills: LoadedExtensions
    diagnostics: list[RuntimeDiagnostic] = field(default_factory=list)


def validate_tool_definition(tool: AgentTool) -> list[RuntimeDiagnostic]:
    """校验单个工具定义。

    Args:
        tool: 要校验的工具。

    Returns:
        诊断列表（空表示校验通过）。
    """
    diagnostics: list[RuntimeDiagnostic] = []

    # 检查 name 非空
    if not tool.name or not tool.name.strip():
        diagnostics.append(RuntimeDiagnostic(
            severity="error",
            code="tool.invalid_name",
            message="Tool name is empty",
        ))

    # 检查 description 非空
    if not tool.description or not tool.description.strip():
        diagnostics.append(RuntimeDiagnostic(
            severity="warning",
            code="tool.missing_description",
            message=f"Tool '{tool.name}' has no description",
        ))

    # 检查 parameters 是对象形式的 JSON Schema
    if not isinstance(tool.parameters, dict):
        diagnostics.append(RuntimeDiagnostic(
            severity="error",
            code="tool.invalid_parameters",
            message=f"Tool '{tool.name}' parameters must be a dict",
        ))
    elif "type" in tool.parameters and tool.parameters["type"] != "object":
        diagnostics.append(RuntimeDiagnostic(
            severity="warning",
            code="tool.parameters_not_object",
            message=f"Tool '{tool.name}' parameters type should be 'object'",
        ))

    # 检查 execute 可调用
    if not callable(tool.execute):
        diagnostics.append(RuntimeDiagnostic(
            severity="error",
            code="tool.execute_not_callable",
            message=f"Tool '{tool.name}' execute is not callable",
        ))

    return diagnostics


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
    5. 校验工具定义。
    6. 记录工具来源和 metadata。
    7. 检测名称冲突并生成诊断。
    8. 如果是只读模式，过滤掉非只读工具。
    9. 创建 ToolRuntime 并应用权限策略。

    Args:
        workspace: 工作区目录路径。
        options: 创建会话的配置选项。
        config: 已解析的运行时配置。

    Returns:
        AssembledTools 对象，包含最终工具列表和加载的扩展/技能信息。
    """
    diagnostics: list[RuntimeDiagnostic] = []

    # 加载扩展和技能
    loaded_extensions = load_extensions(workspace, configured_paths=config.extension_paths)
    loaded_skills = load_skills(workspace, configured_paths=config.skill_paths)

    # 收集扩展和技能的错误
    for error in loaded_extensions.errors:
        diagnostics.append(RuntimeDiagnostic(
            severity="warning",
            code="extension.load_error",
            message=error,
            source="extension",
        ))
    for error in loaded_skills.errors:
        diagnostics.append(RuntimeDiagnostic(
            severity="warning",
            code="skill.load_error",
            message=error,
            source="skill",
        ))

    # 创建 MCP 代理工具
    mcp_tools = create_mcp_proxy_tools(
        parse_mcp_tool_configs(config.mcp_servers),
        client=options.mcp_client,
    )

    # 检查 MCP client 缺失
    if config.mcp_servers and not options.mcp_client:
        diagnostics.append(RuntimeDiagnostic(
            severity="warning",
            code="mcp.client_missing",
            message="MCP servers configured but no MCP client provided",
        ))

    # 创建内置工具
    builtin_tools = create_builtin_tools(
        workspace,
        enabled_names=config.enabled_builtin_tools,
        edit_require_unique_match=config.edit_require_unique_match,
        shell_policy=ShellExecutionPolicy(
            timeout_seconds=config.shell_timeout_seconds,
            max_timeout_seconds=config.shell_max_timeout_seconds,
            stdout_limit=config.shell_stdout_limit,
            stderr_limit=config.shell_stderr_limit,
            allowed_env=tuple(config.shell_allowed_env or ()),
        ),
    )

    # 按名称去重合并：内置 -> 自定义 -> 扩展 -> MCP（后者覆盖前者）
    tool_map: dict[str, tuple[AgentTool, str, str | None]] = {}

    # 内置工具
    for tool in builtin_tools:
        if tool.name in tool_map:
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.name_conflict",
                message=f"Tool '{tool.name}' from builtin overrides previous registration",
            ))
        tool_map[tool.name] = (tool, "builtin", None)

    # 调用方工具
    for tool in options.tools:
        if get_builtin_tool_metadata(tool.name) is not None:
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.reserved_name",
                message=f"Tool '{tool.name}' from caller uses a reserved builtin name",
                source="caller",
            ))
            continue
        if tool.name in tool_map:
            prev_source = tool_map[tool.name][1]
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.name_conflict",
                message=f"Tool '{tool.name}' from caller overrides {prev_source}",
            ))
        tool_map[tool.name] = (tool, "caller", None)

    # 扩展工具
    for tool in loaded_extensions.tools:
        if get_builtin_tool_metadata(tool.name) is not None:
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.reserved_name",
                message=f"Tool '{tool.name}' from extension uses a reserved builtin name",
                source="extension",
            ))
            continue
        if tool.name in tool_map:
            prev_source = tool_map[tool.name][1]
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.name_conflict",
                message=f"Tool '{tool.name}' from extension overrides {prev_source}",
            ))
        tool_map[tool.name] = (tool, "extension", None)

    # MCP 工具
    for tool in mcp_tools:
        if get_builtin_tool_metadata(tool.name) is not None:
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.reserved_name",
                message=f"Tool '{tool.name}' from MCP uses a reserved builtin name",
                source="mcp",
            ))
            continue
        if tool.name in tool_map:
            prev_source = tool_map[tool.name][1]
            diagnostics.append(RuntimeDiagnostic(
                severity="warning",
                code="tool.name_conflict",
                message=f"Tool '{tool.name}' from MCP overrides {prev_source}",
            ))
        tool_map[tool.name] = (tool, "mcp", None)

    # 校验工具定义并构建 RegisteredTool 列表
    registered_tools: list[RegisteredTool] = []
    for name, (tool, source, origin) in tool_map.items():
        tool_diagnostics = validate_tool_definition(tool)
        diagnostics.extend(tool_diagnostics)
        if any(diag.severity == "error" for diag in tool_diagnostics):
            continue

        metadata = (
            get_builtin_tool_metadata(tool.name)
            if source == "builtin"
            else tool.metadata or infer_tool_metadata(tool)
        )

        registered_tools.append(RegisteredTool(
            name=name,
            tool=tool,
            metadata=metadata,
            source=source,
            origin=origin,
        ))

    # 注册到 ToolRegistry
    registry = ToolRegistry()
    for reg_tool in registered_tools:
        registry.register(reg_tool.tool, metadata=reg_tool.metadata)

    # 只读模式：过滤掉非只读工具
    if config.read_only_mode:
        registry = _read_only_registry(registry)

    # 创建 ToolRuntime 并应用权限策略
    runtime = ToolRuntime(
        registry=registry,
        permission_policy=PermissionPolicy(
            mode=config.tool_permission_mode,  # type: ignore[arg-type]
            block_dangerous_bash=config.block_dangerous_bash,
            bash_allow_patterns=config.bash_allow_patterns,
            bash_block_patterns=config.bash_block_patterns,
        ),
        approval_provider=options.approval_provider or DeferredApprovalProvider(),
    )

    return AssembledTools(
        tools=runtime.as_agent_tools(),
        registered_tools=registered_tools,
        loaded_extensions=loaded_extensions,
        loaded_skills=loaded_skills,
        diagnostics=diagnostics,
    )


def _read_only_registry(registry: ToolRegistry) -> ToolRegistry:
    """过滤工具注册表，只保留标记为 read_only 的工具。"""
    filtered = ToolRegistry()
    for tool in registry.list():
        metadata = registry.metadata_for(tool.name)
        if metadata is not None and metadata.read_only:
            filtered.register(tool, metadata=metadata)
    return filtered
