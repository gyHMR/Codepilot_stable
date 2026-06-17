from __future__ import annotations

"""Runtime configuration loading and option resolution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codepilot.core import ToolExecutionMode
from codepilot.sessions.store import SessionStore

from .resources import WorkspaceResourceLoader, WorkspaceResources
from .types import CreateAgentSessionOptions


@dataclass(frozen=True)
class RuntimeConfig:
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
    resources: WorkspaceResources | None,
    restored_meta: dict[str, Any] | None,
) -> RuntimeConfig:
    system_prompt = options.system_prompt
    if not system_prompt and resources and resources.prompt:
        system_prompt = resources.prompt
    if not system_prompt and resources and resources.settings.system_prompt:
        system_prompt = resources.settings.system_prompt
    if not system_prompt and restored_meta and isinstance(restored_meta.get("system_prompt"), str):
        system_prompt = restored_meta["system_prompt"]

    thinking_level = options.thinking_level
    if thinking_level == "off" and resources and resources.settings.thinking_level:
        thinking_level = resources.settings.thinking_level

    tool_execution = options.tool_execution
    if tool_execution == "parallel" and resources and resources.settings.tool_execution:
        tool_execution = resources.settings.tool_execution

    max_context_messages = options.max_context_messages
    if max_context_messages is None and resources and resources.settings.max_context_messages is not None:
        max_context_messages = resources.settings.max_context_messages

    retain_recent_messages = options.retain_recent_messages
    if retain_recent_messages == 24 and resources and resources.settings.retain_recent_messages is not None:
        retain_recent_messages = resources.settings.retain_recent_messages

    max_context_tokens = options.max_context_tokens
    if max_context_tokens is None and resources and resources.settings.max_context_tokens is not None:
        max_context_tokens = resources.settings.max_context_tokens

    retry_enabled = options.retry_enabled
    if resources and resources.settings.retry_enabled is not None:
        retry_enabled = resources.settings.retry_enabled

    max_retries = options.max_retries
    if resources and resources.settings.max_retries is not None and options.max_retries == 2:
        max_retries = resources.settings.max_retries

    retry_base_delay_ms = options.retry_base_delay_ms
    if resources and resources.settings.retry_base_delay_ms is not None and options.retry_base_delay_ms == 1200:
        retry_base_delay_ms = resources.settings.retry_base_delay_ms

    read_only_mode = options.read_only_mode
    if resources and resources.settings.read_only_mode is not None:
        read_only_mode = resources.settings.read_only_mode

    block_dangerous_bash = options.block_dangerous_bash
    if resources and resources.settings.block_dangerous_bash is not None:
        block_dangerous_bash = resources.settings.block_dangerous_bash

    bash_allow_patterns = options.bash_allow_patterns
    if bash_allow_patterns is None and resources and resources.settings.bash_allow_patterns is not None:
        bash_allow_patterns = resources.settings.bash_allow_patterns

    bash_block_patterns = options.bash_block_patterns
    if bash_block_patterns is None and resources and resources.settings.bash_block_patterns is not None:
        bash_block_patterns = resources.settings.bash_block_patterns

    edit_require_unique_match = options.edit_require_unique_match
    if resources and resources.settings.edit_require_unique_match is not None:
        edit_require_unique_match = resources.settings.edit_require_unique_match

    extension_paths = options.extension_paths
    if extension_paths is None and resources and resources.settings.extension_paths is not None:
        extension_paths = resources.settings.extension_paths

    skill_paths = options.skill_paths
    if skill_paths is None and resources and resources.settings.skill_paths is not None:
        skill_paths = resources.settings.skill_paths

    mcp_servers = options.mcp_servers
    if mcp_servers is None and resources and resources.settings.mcp_servers is not None:
        mcp_servers = resources.settings.mcp_servers

    prompt_guidelines = options.prompt_guidelines
    if prompt_guidelines is None and resources and resources.settings.prompt_guidelines is not None:
        prompt_guidelines = resources.settings.prompt_guidelines

    append_system_prompt = options.append_system_prompt
    if append_system_prompt is None and resources and resources.settings.append_system_prompt is not None:
        append_system_prompt = resources.settings.append_system_prompt

    prompt_debug_sources = options.prompt_debug_sources
    if not prompt_debug_sources and resources and resources.settings.prompt_debug_sources:
        prompt_debug_sources = True

    tool_snippets = options.tool_snippets
    if tool_snippets is None and resources and resources.settings.tool_snippets is not None:
        tool_snippets = resources.settings.tool_snippets

    enabled_builtin_tools = options.enabled_builtin_tools
    if enabled_builtin_tools is None and resources:
        enabled_builtin_tools = resources.enabled_tools

    return RuntimeConfig(
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
    )
