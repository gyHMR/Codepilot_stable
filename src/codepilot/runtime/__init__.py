"""
Codepilot runtime assembly layer.

- create_agent_session factory
- RuntimeService shared by user-facing interfaces
- workspace resource and command assembly helpers
"""

from .command_registry import format_commands_for_help, list_runtime_commands
from .factory import build_agent_session_options, create_agent_session
from .prompt import SystemPromptBuildOptions, build_default_system_prompt, build_system_prompt
from .resources import WorkspaceModelConfig, WorkspaceResourceLoader, WorkspaceResources, WorkspaceSettings
from .service import CreateSessionRequest, RuntimeService, SessionHandle, UserInput
from .types import CreateAgentSessionOptions


__all__ = [
    "CreateAgentSessionOptions",
    "build_agent_session_options",
    "create_agent_session",
    "RuntimeService",
    "CreateSessionRequest",
    "SessionHandle",
    "UserInput",
    "WorkspaceResourceLoader",
    "WorkspaceModelConfig",
    "WorkspaceResources",
    "WorkspaceSettings",
    "build_default_system_prompt",
    "build_system_prompt",
    "SystemPromptBuildOptions",
    "format_commands_for_help",
    "list_runtime_commands",
]
