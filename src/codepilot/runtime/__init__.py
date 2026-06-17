"""
Codepilot runtime assembly layer.

- AgentSession
- session persistence (context/events JSONL)
- create_agent_session factory
"""

from codepilot.extensions import discover_extension_paths, discover_skill_paths, load_extensions, load_skills
from codepilot.extensions.mcp import create_mcp_proxy_tools, parse_mcp_tool_configs
from codepilot.sessions.session import AgentSession
from codepilot.tools.builtin import create_builtin_tools

from .command_registry import format_commands_for_help, list_runtime_commands
from .convert_to_llm import convert_to_llm
from .factory import create_agent_session
from .resources import WorkspaceResourceLoader, WorkspaceResources, WorkspaceSettings
from .service import CreateSessionRequest, RuntimeService, SessionHandle, UserInput
from .system_prompt import SystemPromptBuildOptions, build_default_system_prompt, build_system_prompt
from .types import AgentSessionOptions, CreateAgentSessionOptions

_CLI_EXPORTS = {"RunOptions", "build_parser", "main", "run", "run_print", "run_interactive", "run_rpc"}


def __getattr__(name: str):
    if name in _CLI_EXPORTS:
        from codepilot import interfaces
        from codepilot.interfaces import cli

        _ = interfaces
        return getattr(cli, name)
    raise AttributeError(f"module 'codepilot.runtime' has no attribute {name!r}")


__all__ = [
    "AgentSession",
    "AgentSessionOptions",
    "CreateAgentSessionOptions",
    "convert_to_llm",
    "create_agent_session",
    "RuntimeService",
    "CreateSessionRequest",
    "SessionHandle",
    "UserInput",
    "create_builtin_tools",
    "WorkspaceResourceLoader",
    "WorkspaceResources",
    "WorkspaceSettings",
    "build_default_system_prompt",
    "build_system_prompt",
    "SystemPromptBuildOptions",
    "discover_extension_paths",
    "discover_skill_paths",
    "load_extensions",
    "load_skills",
    "format_commands_for_help",
    "list_runtime_commands",
    "parse_mcp_tool_configs",
    "create_mcp_proxy_tools",
    "build_parser",
    "main",
    "RunOptions",
    "run",
    "run_print",
    "run_interactive",
    "run_rpc",
]
