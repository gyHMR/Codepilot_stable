from __future__ import annotations

"""Build the tool runtime for one opened session."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codepilot.extensions import load_extensions, load_skills
from codepilot.extensions.mcp import create_mcp_proxy_tools, parse_mcp_tool_configs
from codepilot.protocols import Tool
from codepilot.protocols.commands import RegisteredCommand
from codepilot.tools import AgentTool
from codepilot.tools.approval import DeferredApprovalProvider
from codepilot.tools.builtins import create_builtin_tools
from codepilot.tools.engine import ToolRuntime
from codepilot.tools.policy import PermissionPolicy
from codepilot.tools.registry import (
    ToolRegistry,
    get_builtin_tool_metadata,
    infer_tool_metadata,
)
from codepilot.tools.workspace import ShellExecutionPolicy

from .config import RuntimeConfig
from .opening import SessionOpenIntent


@dataclass
class RuntimeTools:
    specs: list[Tool]
    runtime: ToolRuntime
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_tool_hooks: list[Any] = field(default_factory=list)
    after_tool_hooks: list[Any] = field(default_factory=list)
    before_prompt_hooks: list[Any] = field(default_factory=list)
    after_prompt_hooks: list[Any] = field(default_factory=list)
    prompt_guidelines: list[str] = field(default_factory=list)
    append_prompts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_runtime_tools(
    workspace: Path,
    intent: SessionOpenIntent,
    config: RuntimeConfig,
) -> RuntimeTools:
    """Load tools and extension capabilities, then create the ToolRuntime."""

    warnings: list[str] = []
    loaded_extensions = load_extensions(workspace, configured_paths=config.extension_paths)
    loaded_skills = load_skills(workspace, configured_paths=config.skill_paths)
    warnings.extend(f"extension: {error}" for error in loaded_extensions.errors)
    warnings.extend(f"skill: {error}" for error in loaded_skills.errors)
    warnings.extend(f"skill: {item}" for item in loaded_skills.diagnostics)

    mcp_tools = create_mcp_proxy_tools(
        parse_mcp_tool_configs(config.mcp_servers),
        client=intent.mcp_client,
    )
    if config.mcp_servers and intent.mcp_client is None:
        warnings.append("MCP servers configured but no MCP client was provided")

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

    registry = ToolRegistry()
    _register_tools(registry, builtin_tools, source="builtin", warnings=warnings)
    _register_tools(registry, list(intent.tools), source="caller", warnings=warnings)
    _register_tools(registry, loaded_skills.tools, source="skill", warnings=warnings)
    _register_tools(registry, loaded_extensions.tools, source="extension", warnings=warnings)
    _register_tools(registry, mcp_tools, source="mcp", warnings=warnings)

    if config.read_only_mode:
        registry = _read_only_registry(registry)

    runtime = ToolRuntime(
        registry=registry,
        permission_policy=PermissionPolicy(
            mode=config.tool_permission_mode,
            block_dangerous_bash=config.block_dangerous_bash,
            bash_allow_patterns=config.bash_allow_patterns,
            bash_block_patterns=config.bash_block_patterns,
        ),
        approval_provider=intent.approval_provider or DeferredApprovalProvider(),
    )

    return RuntimeTools(
        specs=[tool.to_spec() for tool in registry.list()],
        runtime=runtime,
        commands={
            **loaded_skills.commands,
            **loaded_extensions.commands,
        },
        before_tool_hooks=[
            *loaded_extensions.before_tool_hooks,
            *loaded_skills.before_tool_hooks,
        ],
        after_tool_hooks=[
            *loaded_extensions.after_tool_hooks,
            *loaded_skills.after_tool_hooks,
        ],
        before_prompt_hooks=[
            *loaded_extensions.before_prompt_hooks,
            *loaded_skills.before_prompt_hooks,
        ],
        after_prompt_hooks=[
            *loaded_extensions.after_prompt_hooks,
            *loaded_skills.after_prompt_hooks,
        ],
        prompt_guidelines=[
            *(config.prompt_guidelines or []),
            *loaded_extensions.prompt_guidelines,
            *loaded_skills.prompt_guidelines,
        ],
        append_prompts=[
            *([config.append_system_prompt] if config.append_system_prompt else []),
            *loaded_extensions.append_prompts,
            *loaded_skills.append_prompts,
            *(_debug_prompt_sources(loaded_extensions, loaded_skills) if config.prompt_debug_sources else []),
        ],
        warnings=warnings,
    )


def _register_tools(
    registry: ToolRegistry,
    tools: list[AgentTool],
    *,
    source: str,
    warnings: list[str],
) -> None:
    for tool in tools:
        if source != "builtin" and get_builtin_tool_metadata(tool.name) is not None:
            warnings.append(f"{source} tool '{tool.name}' uses a reserved builtin name")
            continue
        if registry.get(tool.name) is not None:
            warnings.append(f"{source} tool '{tool.name}' overrides an earlier tool")
        if not tool.description.strip():
            warnings.append(f"{source} tool '{tool.name}' has no description")
        metadata = (
            get_builtin_tool_metadata(tool.name)
            if source == "builtin"
            else tool.metadata or infer_tool_metadata(tool)
        )
        registry.register(tool, metadata=metadata)


def _read_only_registry(registry: ToolRegistry) -> ToolRegistry:
    filtered = ToolRegistry()
    for tool in registry.list():
        metadata = registry.metadata_for(tool.name)
        if metadata is not None and metadata.read_only:
            filtered.register(tool, metadata=metadata)
    return filtered


def _debug_prompt_sources(extensions: Any, skills: Any) -> list[str]:
    lines: list[str] = ["## Prompt Sources", "### extensions"]
    lines.extend([f"- {path}" for path in extensions.loaded_paths] or ["- (none)"])
    lines.append("### skills")
    lines.extend([f"- {path}" for path in skills.loaded_paths] or ["- (none)"])
    errors = [*extensions.errors, *skills.errors]
    if errors:
        lines.append("### errors")
        lines.extend(f"- {error}" for error in errors)
    if skills.diagnostics:
        lines.append("### diagnostics")
        lines.extend(f"- {item}" for item in skills.diagnostics)
    return ["\n".join(lines)]


__all__ = [
    "RuntimeTools",
    "build_runtime_tools",
]
