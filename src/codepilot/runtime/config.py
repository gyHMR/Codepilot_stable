from __future__ import annotations

"""Runtime inputs and effective configuration resolution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from codepilot.core import ToolExecutionMode
from codepilot.sessions.store import SessionStore

from .resources import WorkspaceResourceLoader, WorkspaceResources
from .types import CreateAgentSessionOptions

ConfigSource = str
T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeDefaults:
    system_prompt: str = ""
    thinking_level: str = "off"
    tool_execution: ToolExecutionMode = "parallel"
    max_context_messages: int | None = None
    retain_recent_messages: int = 24
    max_context_tokens: int | None = None
    retry_enabled: bool = True
    max_retries: int = 2
    retry_base_delay_ms: int = 1200
    read_only_mode: bool = False
    block_dangerous_bash: bool = True
    bash_allow_patterns: list[str] | None = None
    bash_block_patterns: list[str] | None = None
    edit_require_unique_match: bool = True
    extension_paths: list[str] | None = None
    skill_paths: list[str] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    prompt_guidelines: list[str] | None = None
    append_system_prompt: str | None = None
    prompt_debug_sources: bool = False
    tool_snippets: dict[str, str] | None = None
    enabled_builtin_tools: list[str] | None = None


@dataclass(frozen=True)
class RuntimeInputs:
    workspace: Path
    resources: WorkspaceResources | None
    restored_meta: dict[str, Any] | None


@dataclass(frozen=True)
class ResolvedRuntimeConfig:
    system_prompt: str
    thinking_level: str
    tool_execution: ToolExecutionMode
    max_context_messages: int | None
    retain_recent_messages: int
    max_context_tokens: int | None
    retry_enabled: bool
    max_retries: int
    retry_base_delay_ms: int
    read_only_mode: bool
    block_dangerous_bash: bool
    bash_allow_patterns: list[str] | None
    bash_block_patterns: list[str] | None
    edit_require_unique_match: bool
    extension_paths: list[str] | None
    skill_paths: list[str] | None
    mcp_servers: list[dict[str, Any]] | None
    prompt_guidelines: list[str] | None
    append_system_prompt: str | None
    prompt_debug_sources: bool
    tool_snippets: dict[str, str] | None
    enabled_builtin_tools: list[str] | None
    sources: dict[str, ConfigSource] = field(default_factory=dict)


RuntimeConfig = ResolvedRuntimeConfig


def load_runtime_inputs(options: CreateAgentSessionOptions) -> RuntimeInputs:
    workspace, resources = load_workspace_resources(options)
    return RuntimeInputs(
        workspace=workspace,
        resources=resources,
        restored_meta=read_restored_session_meta(workspace, options.session_id),
    )


def load_workspace_resources(options: CreateAgentSessionOptions) -> tuple[Path, WorkspaceResources | None]:
    workspace = Path(options.workspace_dir)
    resources = WorkspaceResourceLoader(workspace).load() if options.load_workspace_resources else None
    return workspace, resources


def read_restored_session_meta(workspace: Path, session_id: str | None) -> dict[str, Any] | None:
    if not session_id:
        return None
    return SessionStore(workspace_dir=workspace, session_id=session_id).read_meta()


def resolve_runtime_config(
    options: CreateAgentSessionOptions,
    inputs: RuntimeInputs,
    defaults: RuntimeDefaults = RuntimeDefaults(),
) -> ResolvedRuntimeConfig:
    resources = inputs.resources
    settings = resources.settings if resources is not None else None
    restored = inputs.restored_meta or {}
    sources: dict[str, ConfigSource] = {}

    def choose(name: str, *candidates: tuple[ConfigSource, T | None], default: T) -> T:
        for source, value in candidates:
            if value is not None:
                sources[name] = source
                return value
        sources[name] = "default"
        return default

    system_prompt = choose(
        "system_prompt",
        ("options", options.system_prompt),
        ("restored_session", restored.get("system_prompt") if isinstance(restored.get("system_prompt"), str) else None),
        ("workspace", resources.prompt if resources is not None else None),
        ("workspace", settings.system_prompt if settings is not None else None),
        default=defaults.system_prompt,
    )
    thinking_level = choose(
        "thinking_level",
        ("options", options.thinking_level),
        ("workspace", settings.thinking_level if settings is not None else None),
        default=defaults.thinking_level,
    )
    tool_execution = choose(
        "tool_execution",
        ("options", options.tool_execution),
        ("workspace", settings.tool_execution if settings is not None else None),
        default=defaults.tool_execution,
    )
    max_context_messages = choose(
        "max_context_messages",
        ("options", options.max_context_messages),
        ("workspace", settings.max_context_messages if settings is not None else None),
        default=defaults.max_context_messages,
    )
    retain_recent_messages = choose(
        "retain_recent_messages",
        ("options", options.retain_recent_messages),
        ("workspace", settings.retain_recent_messages if settings is not None else None),
        default=defaults.retain_recent_messages,
    )
    max_context_tokens = choose(
        "max_context_tokens",
        ("options", options.max_context_tokens),
        ("workspace", settings.max_context_tokens if settings is not None else None),
        default=defaults.max_context_tokens,
    )
    retry_enabled = choose(
        "retry_enabled",
        ("options", options.retry_enabled),
        ("workspace", settings.retry_enabled if settings is not None else None),
        default=defaults.retry_enabled,
    )
    max_retries = choose(
        "max_retries",
        ("options", options.max_retries),
        ("workspace", settings.max_retries if settings is not None else None),
        default=defaults.max_retries,
    )
    retry_base_delay_ms = choose(
        "retry_base_delay_ms",
        ("options", options.retry_base_delay_ms),
        ("workspace", settings.retry_base_delay_ms if settings is not None else None),
        default=defaults.retry_base_delay_ms,
    )
    read_only_mode = choose(
        "read_only_mode",
        ("options", options.read_only_mode),
        ("workspace", settings.read_only_mode if settings is not None else None),
        default=defaults.read_only_mode,
    )
    block_dangerous_bash = choose(
        "block_dangerous_bash",
        ("options", options.block_dangerous_bash),
        ("workspace", settings.block_dangerous_bash if settings is not None else None),
        default=defaults.block_dangerous_bash,
    )
    bash_allow_patterns = choose(
        "bash_allow_patterns",
        ("options", options.bash_allow_patterns),
        ("workspace", settings.bash_allow_patterns if settings is not None else None),
        default=defaults.bash_allow_patterns,
    )
    bash_block_patterns = choose(
        "bash_block_patterns",
        ("options", options.bash_block_patterns),
        ("workspace", settings.bash_block_patterns if settings is not None else None),
        default=defaults.bash_block_patterns,
    )
    edit_require_unique_match = choose(
        "edit_require_unique_match",
        ("options", options.edit_require_unique_match),
        ("workspace", settings.edit_require_unique_match if settings is not None else None),
        default=defaults.edit_require_unique_match,
    )
    extension_paths = choose(
        "extension_paths",
        ("options", options.extension_paths),
        ("workspace", settings.extension_paths if settings is not None else None),
        default=defaults.extension_paths,
    )
    skill_paths = choose(
        "skill_paths",
        ("options", options.skill_paths),
        ("workspace", settings.skill_paths if settings is not None else None),
        default=defaults.skill_paths,
    )
    mcp_servers = choose(
        "mcp_servers",
        ("options", options.mcp_servers),
        ("workspace", settings.mcp_servers if settings is not None else None),
        default=defaults.mcp_servers,
    )
    prompt_guidelines = choose(
        "prompt_guidelines",
        ("options", options.prompt_guidelines),
        ("workspace", settings.prompt_guidelines if settings is not None else None),
        default=defaults.prompt_guidelines,
    )
    append_system_prompt = choose(
        "append_system_prompt",
        ("options", options.append_system_prompt),
        ("workspace", settings.append_system_prompt if settings is not None else None),
        default=defaults.append_system_prompt,
    )
    prompt_debug_sources = choose(
        "prompt_debug_sources",
        ("options", options.prompt_debug_sources),
        ("workspace", settings.prompt_debug_sources if settings is not None else None),
        default=defaults.prompt_debug_sources,
    )
    tool_snippets = choose(
        "tool_snippets",
        ("options", options.tool_snippets),
        ("workspace", settings.tool_snippets if settings is not None else None),
        default=defaults.tool_snippets,
    )
    enabled_builtin_tools = choose(
        "enabled_builtin_tools",
        ("options", options.enabled_builtin_tools),
        ("workspace", resources.enabled_tools if resources is not None else None),
        default=defaults.enabled_builtin_tools,
    )

    return ResolvedRuntimeConfig(
        system_prompt=system_prompt,
        thinking_level=thinking_level,
        tool_execution=tool_execution,
        max_context_messages=max_context_messages,
        retain_recent_messages=retain_recent_messages,
        max_context_tokens=max_context_tokens,
        retry_enabled=retry_enabled,
        max_retries=max_retries,
        retry_base_delay_ms=retry_base_delay_ms,
        read_only_mode=read_only_mode,
        block_dangerous_bash=block_dangerous_bash,
        bash_allow_patterns=bash_allow_patterns,
        bash_block_patterns=bash_block_patterns,
        edit_require_unique_match=edit_require_unique_match,
        extension_paths=extension_paths,
        skill_paths=skill_paths,
        mcp_servers=mcp_servers,
        prompt_guidelines=prompt_guidelines,
        append_system_prompt=append_system_prompt,
        prompt_debug_sources=prompt_debug_sources,
        tool_snippets=tool_snippets,
        enabled_builtin_tools=enabled_builtin_tools,
        sources=sources,
    )
