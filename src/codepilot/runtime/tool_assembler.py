from __future__ import annotations

"""Tool assembly for runtime sessions."""

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
    tools: list[AgentTool]
    loaded_extensions: LoadedExtensions
    loaded_skills: LoadedExtensions


def assemble_tools(
    workspace: Path,
    options: CreateAgentSessionOptions,
    config: RuntimeConfig,
) -> AssembledTools:
    loaded_extensions = load_extensions(workspace, configured_paths=config.extension_paths)
    loaded_skills = load_skills(workspace, configured_paths=config.skill_paths)
    mcp_tools = create_mcp_proxy_tools(
        parse_mcp_tool_configs(config.mcp_servers),
        client=options.mcp_client,
    )

    builtin_tools = create_builtin_tools(
        workspace,
        enabled_names=config.enabled_builtin_tools,
        edit_require_unique_match=config.edit_require_unique_match,
    )

    tool_map = {tool.name: tool for tool in builtin_tools}
    for tool in options.tools:
        tool_map[tool.name] = tool
    for tool in loaded_extensions.tools:
        tool_map[tool.name] = tool
    for tool in mcp_tools:
        tool_map[tool.name] = tool

    registry = ToolRegistry()
    for tool in tool_map.values():
        metadata = get_builtin_tool_metadata(tool.name)
        registry.register(tool, metadata=metadata)

    if config.read_only_mode:
        registry = _read_only_registry(registry)
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
    filtered = ToolRegistry()
    for tool in registry.list():
        metadata = registry.metadata_for(tool.name)
        if metadata is not None and metadata.read_only:
            filtered.register(tool, metadata=metadata)
    return filtered
